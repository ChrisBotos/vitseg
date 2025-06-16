#!/usr/bin/env python3
"""patch_clusterer_rewrite.py

A robust and well‑documented pipeline for clustering Vision Transformer (DINO‑ViT) patch
embeddings extracted from DAPI‑stained microscopy images and visualising the results on
Cellpose‑defined nuclear masks. The script is intended for high‑quality figure generation
and reproducible analyses.

Usage example
-------------
python cluster_vit_patches.py \
    -i IRI_regist_cropped.tif \
    -m segmentation_masks.npy \
    -c VIT_dynamic_patches/coords_IRI_regist_cropped.csv \
    -f VIT_dynamic_patches/features_IRI_regist_cropped.csv \
    -k 12 \
    --region 0 1 0 1 \
    --auto-k dbi \
    --outdir ./my_experiment \
    --log my_experiment/pipeline.log
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

###############################################################################
# Helper functions                                                             #
###############################################################################

def set_global_seed(seed: int) -> None:
    """Set the RNG seed across `random`, NumPy, and scikit‑learn for reproducibility."""
    random.seed(seed)  # Standard library RNG.
    np.random.seed(seed)  # NumPy RNG.
    try:
        import torch  # Optional import to seed PyTorch if available.

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ModuleNotFoundError:
        pass  # PyTorch not installed; silently continue.


def configure_logging(level: str = "INFO", logfile: Path | None = None) -> None:
    """Configure console and optional file logging with a consistent format."""
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="w", encoding="utf‑8"))

    logging.basicConfig(level=level.upper(), format=fmt, datefmt=datefmt, handlers=handlers)


def parse_arguments() -> argparse.Namespace:
    """Parse command‑line arguments and return the populated namespace."""
    parser = argparse.ArgumentParser(
        description="Cluster DINO‑ViT patch features and overlay results on Cellpose masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = parser.add_argument_group("Input/Output")
    io.add_argument("-i", "--image", type=Path, required=True, help="Microscopy image (e.g. DAPI).")
    io.add_argument("-m", "--mask", type=Path, required=True, help="`*.npy` mask label image produced by Cellpose.")
    io.add_argument("-c", "--coords", type=Path, required=True, help="Patch centroid coordinates CSV (x_center, y_center).")
    io.add_argument("-f", "--features", type=Path, required=True, help="Patch feature CSV (rows align with `--coords`).")
    io.add_argument("--outdir", type=Path, default=Path("results"), help="Directory for all generated artefacts.")
    io.add_argument("--log", type=Path, default=None, help="Optional path to a log file.")

    model = parser.add_argument_group("Model")
    model.add_argument("-k", "--clusters", type=int, default=10, help="Initial K for K‑Means.")
    model.add_argument(
        "--auto-k",
        choices=["none", "silhouette", "dbi"],
        default="none",
        help="Criterion for automatic K selection (silhouette or Davies–Bouldin).",
    )
    model.add_argument("--seed", type=int, default=0, help="Random seed for deterministic output.")

    viz = parser.add_argument_group("Visualisation")
    viz.add_argument(
        "--region",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=(0.0, 1.0, 0.0, 1.0),
        help="Fractional crop region for overlays (values in [0, 1]).",
    )

    return parser.parse_args()


def choose_optimal_k(features: np.ndarray, k_max: int, criterion: str) -> Tuple[int, pd.DataFrame]:
    """Determine the optimal K using silhouette or Davies–Bouldin indices.

    Parameters
    ----------
    features : np.ndarray
        Scaled feature matrix (n_samples × n_features).
    k_max : int
        Maximum number of clusters to consider (inclusive).
    criterion : str
        Either "silhouette" for maximising silhouette score or "dbi" for minimising Davies–Bouldin index.

    Returns
    -------
    best_k : int
        The selected optimal cluster count.
    scores_df : pd.DataFrame
        DataFrame with scores for each K tested for downstream inspection.
    """
    logging.info("Searching for optimal K using %s criterion…", criterion.capitalize())

    candidate_ks = range(2, k_max + 1)
    results: List[Tuple[int, float]] = []

    for k in candidate_ks:
        model = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = model.fit_predict(features)

        if criterion == "silhouette":
            score = silhouette_score(features, labels)
        elif criterion == "dbi":
            score = davies_bouldin_score(features, labels)
        else:
            raise ValueError(f"Unsupported criterion: {criterion}.")

        logging.debug("K=%d | %s=%.4f", k, criterion, score)
        results.append((k, score))

    scores_df = pd.DataFrame(results, columns=["k", criterion])

    if criterion == "silhouette":
        best_k = scores_df.loc[scores_df[criterion].idxmax(), "k"]
    else:  # Davies–Bouldin must be minimised.
        best_k = scores_df.loc[scores_df[criterion].idxmin(), "k"]

    logging.info("Optimal K selected: %d.", best_k)
    return best_k, scores_df


def compute_label_cluster_map(mask: np.ndarray, clusters: pd.Series) -> Dict[int, int]:
    """Map each Cellpose object ID to its corresponding cluster label.

    This function calculates object centroids in pixel space, then assigns the cluster
    label that belongs to the *nearest* patch centroid within that object. Alternative
    strategies (e.g. majority vote) can be implemented as needed.
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Ignore background label 0.

    centroids: List[Tuple[int, Tuple[float, float]]] = []
    for label_id in unique_labels:
        ys, xs = np.where(mask == label_id)
        if xs.size == 0:  # Empty object guard.
            continue
        centroids.append((label_id, (xs.mean(), ys.mean())))

    # We assume clusters are ordered identically between `coords` and `clusters`.
    mapping = {label_id: int(cluster) for (label_id, _), cluster in zip(centroids, clusters)}
    return mapping


def generate_color_palette(n: int, alpha: int = 128) -> Dict[int, Tuple[int, int, int, int]]:
    """Generate an RGBA palette with visually distinct colours using Matplotlib cycler."""
    cmap = plt.get_cmap("tab20")
    palette = {}
    for i in range(n):
        r, g, b, _ = cmap(i % cmap.N)
        palette[i] = (int(r * 255), int(g * 255), int(b * 255), alpha)
    return palette


def overlay_masks(
    base: Image.Image,
    mask: np.ndarray,
    label_to_colour: Dict[int, Tuple[int, int, int, int]],
) -> Image.Image:
    """Overlay colour‑coded masks onto a base RGB image and return the composite RGBA image."""
    composite = base.convert("RGBA")
    overlay_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)

    for label_id, colour in label_to_colour.items():
        overlay_rgba[mask == label_id] = colour

    overlay_img = Image.fromarray(overlay_rgba, mode="RGBA")
    return Image.alpha_composite(composite, overlay_img)


def main() -> None:  # noqa: C901 – Function intentionally lengthy for linear CLI flow.
    """Entrypoint executed when the script is run as a module."""
    args = parse_arguments()
    configure_logging(logfile=args.log)
    set_global_seed(args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", args.outdir.resolve())

    # ---------------------------------------------------------------------
    # Load input data.
    # ---------------------------------------------------------------------
    img = Image.open(args.image).convert("RGB")
    mask = np.load(args.mask)

    coords_df = pd.read_csv(args.coords)
    features_df = pd.read_csv(args.features)

    if len(coords_df) != len(features_df):
        raise ValueError("Mismatch between rows in coords and features CSVs.")

    # ---------------------------------------------------------------------
    # Feature standardisation.
    # ---------------------------------------------------------------------
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.values)

    # ---------------------------------------------------------------------
    # Optional automatic K selection.
    # ---------------------------------------------------------------------
    k = args.clusters
    scores_summary = None
    if args.auto_k != "none":
        k, scores_summary = choose_optimal_k(features_scaled, k_max=k, criterion=args.auto_k)
        scores_path = args.outdir / "cluster_selection_scores.csv"
        scores_summary.to_csv(scores_path, index=False)
        logging.info("Cluster selection scores saved: %s", scores_path.name)

    # ---------------------------------------------------------------------
    # Clustering.
    # ---------------------------------------------------------------------
    kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init="auto")
    labels = kmeans.fit_predict(features_scaled)
    coords_df["cluster"] = labels

    clusters_csv = args.outdir / "patch_clusters.csv"
    coords_df.to_csv(clusters_csv, index=False)
    logging.info("Cluster assignments saved: %s", clusters_csv.name)

    # Save model and scaler for downstream usage.
    import joblib  # Only import when needed.

    joblib.dump(kmeans, args.outdir / "kmeans_model.joblib")
    joblib.dump(scaler, args.outdir / "scaler.joblib")

    # ---------------------------------------------------------------------
    # Map Cellpose objects to clusters.
    # ---------------------------------------------------------------------
    label_cluster_map = compute_label_cluster_map(mask, coords_df["cluster"])

    # ---------------------------------------------------------------------
    # Determine crop region (fractions → pixels).
    # ---------------------------------------------------------------------
    width, height = img.size
    xmin_f, xmax_f, ymin_f, ymax_f = args.region
    x0, x1 = int(width * xmin_f), int(width * xmax_f)
    y0, y1 = int(height * ymin_f), int(height * ymax_f)

    if x1 <= x0 or y1 <= y0:
        raise ValueError("Provided `--region` arguments yield an empty crop.")

    img_crop = img.crop((x0, y0, x1, y1))
    mask_crop = mask[y0:y1, x0:x1]

    # ---------------------------------------------------------------------
    # Visualisations.
    # ---------------------------------------------------------------------
    colour_palette = generate_color_palette(k)

    # 1) Random‑mask overlay (each object random colour).
    rng = np.random.default_rng(args.seed)
    random_palette = {
        label_id: tuple((rng.random(3) * 255).astype(int)) + (128,)
        for label_id in np.unique(mask_crop)[1:]
    }
    rand_overlay = overlay_masks(img_crop, mask_crop, random_palette)
    rand_overlay.save(args.outdir / "overlay_random.png")

    # 2) Cluster‑coloured overlay.
    cluster_palette = {lab: colour_palette[cluster] for lab, cluster in label_cluster_map.items()}
    cluster_overlay = overlay_masks(img_crop, mask_crop, cluster_palette)
    cluster_overlay.save(args.outdir / "overlay_clusters.png")

    # 3) PCA embedding.
    pca = PCA(n_components=2, random_state=args.seed)
    pcs = pca.fit_transform(features_scaled)
    pcs_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pcs_df["cluster"] = labels
    pcs_df.to_csv(args.outdir / "patch_pca_components.csv", index=False)

    plt.figure(figsize=(5, 5))
    for cl in range(k):
        subset = pcs_df[pcs_df["cluster"] == cl]
        colour = np.array(colour_palette[cl])[:3] / 255
        plt.scatter(subset["PC1"], subset["PC2"], s=8, label=f"Cluster {cl}", color=colour)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("PCA of DINO‑ViT Patch Features")
    plt.legend(frameon=False, loc="best", markerscale=2)
    plt.tight_layout()
    pca_png = args.outdir / "pca_clusters.png"
    plt.savefig(pca_png, dpi=300)
    plt.close()

    logging.info("All outputs written to %s.", args.outdir.resolve())


if __name__ == "__main__":
    main()
