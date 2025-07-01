#!/usr/bin/env python3
"""patch_clusterer_rewrite.py

A robust and well‑documented pipeline for clustering Vision Transformer (DINO‑ViT) patch
embeddings extracted from DAPI‑stained microscopy images and visualising the results on
segmentation‑defined nuclear masks. The script is intended for high‑quality figure generation
and reproducible analyses.

Usage example
-------------
python cluster_vit_patches.py \
    -i img/IRI_regist_cropped.tif \
    -m filtered_results/filtered_passed_masks.npy \
    -c VIT_dynamic_patches_16x16/coords_IRI_regist_cropped.csv \
    -f VIT_dynamic_patches_16x16/features_IRI_regist_cropped.csv \
    -k 12 \
    --region 0 1 0 1 \
    --auto-k dbi \
    --outdir ./clustered_dynamic_vit_patches_16x16 \
    --log clustered_vit_patches.log
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import joblib

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
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
        logfile = logfile.expanduser()
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, mode="w", encoding="utf‑8"))

    logging.basicConfig(level=level.upper(), format=fmt, datefmt=datefmt, handlers=handlers)


def parse_arguments() -> argparse.Namespace:
    """Parse command‑line arguments and return the populated namespace."""
    parser = argparse.ArgumentParser(
        description="Cluster DINO‑ViT patch features and overlay results on segmentation masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    """I/O always required"""
    io = parser.add_argument_group("Input / Output")
    io.add_argument("-i", "--image", type=Path, required=True,
                    help="Microscopy image (e.g. DAPI).")
    io.add_argument("-c", "--coords", type=Path, required=True,
                    help="Patch-centroid CSV (x_center, y_center).")
    io.add_argument("-f", "--features", type=Path, required=True,
                    help="Patch-feature CSV (rows align with --coords).")
    io.add_argument("--outdir", type=Path, default=Path("results"),
                    help="Directory for all generated artefacts.")
    io.add_argument("--log", type=Path,
                    help="Optional path to a log file.")

    """Choose exactly ONE mask input"""
    mx = io.add_mutually_exclusive_group(required=True)
    mx.add_argument("-m", "--mask", type=Path,
                    help="2-D label map OR 3-D Boolean stack (.npy).")
    mx.add_argument("--labels", type=Path,
                    help="1-D label list (.npy).")

    """Only needed when you use --labels"""
    parser.add_argument("--label_map", type=Path,
                        help="Original 2-D segmentation map; REQUIRED with --labels.")

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


def save_overlay_from_masks(
    img: Image.Image,
    masks: np.ndarray,                      # Shape → (N, H, W) boolean.
    labels: np.ndarray,                     # Cluster id per mask slice.
    palette: dict[int, tuple[int, int, int, int]],
    out_path: Path,
    alpha: float = 0.35,
    region: tuple[float, float, float, float] | None = None,
) -> None:
    """Blend coloured nuclei on top of *img*; optionally restrict to `region`."""

    # ── 1 ▸ Crop to region if requested ─────────────────────────────────────────
    if region is not None and region != (0.0, 1.0, 0.0, 1.0):
        xmin, xmax, ymin, ymax = region
        w, h = img.size
        l, r = int(xmin * w), int(xmax * w)
        t, b = int(ymin * h), int(ymax * h)

        img = img.crop((l, t, r, b))                       # PIL object.
        masks = masks[:, t:b, l:r]                         # NumPy slice.

    # ── 2 ▸ Prepare RGB layers"""─
    rgb     = np.array(img.convert("RGB"), dtype=np.uint8)
    overlay = rgb.copy()

    # ── 3 ▸ Blend every nucleus slice ──────────────────────────────────────────
    for idx, m in enumerate(masks):
        cl          = int(labels[idx])
        R, G, B, A  = palette.get(cl, (160, 160, 160, int(alpha * 255)))
        colour      = np.array([R, G, B], dtype=np.uint8)
        a           = (A / 255.0) * alpha

        mask_bool   = m.astype(bool)
        overlay[mask_bool] = (
            (1 - a) * overlay[mask_bool] + a * colour
        ).astype(np.uint8)

    # ── 4 ▸ Write TIFF to disk"""─
    imsave(str(out_path), overlay)


def compute_label_cluster_map(mask: np.ndarray, coords: pd.DataFrame) -> Dict[int, int]:
    """Return a dict {segmentation-label ➜ majority-cluster}.

    Works for 2-D (H×W) masks *and* 3-D stacks such as (1, H, W).
    """
    # 1 ▸ Ensure we are working on a 2-D mask ────────────────────────────
    if mask.ndim > 2:
        mask_2d = mask[0]             # use first z-slice; adjust if needed
    else:
        mask_2d = mask

    clusters_by_label: Dict[int, List[int]] = {}

    # Column 0 = x, column 1 = y (pixel units)
    x_col, y_col = coords.columns[:2]

    # 2 ▸ Walk through every patch centre ────────────────────────────────
    for _, row in coords.iterrows():
        x = int(round(row[x_col]))
        y = int(round(row[y_col]))
        cl = int(row["cluster"])

        # Skip out-of-bounds coordinates (guards against bad CSV entries)
        if not (0 <= x < mask_2d.shape[1] and 0 <= y < mask_2d.shape[0]):
            continue

        label_id = int(mask_2d[y, x])     # mask is indexed (row = y, col = x)
        if label_id == 0:                 # 0 = background
            continue

        clusters_by_label.setdefault(label_id, []).append(cl)

    # 3 ▸ Majority vote per nucleus ──────────────────────────────────────
    return {
        lab: max(set(votes), key=votes.count)
        for lab, votes in clusters_by_label.items()
    }


def generate_color_palette(n: int, alpha: int = 200) -> Dict[int, Tuple[int, int, int, int]]:
    """Generate an RGBA palette with visually distinct colors using Matplotlib cycler."""
    cmap = plt.get_cmap("tab20")
    palette = {}
    for i in range(n):
        r, g, b, _ = cmap(i % cmap.N)
        palette[i] = (int(r * 255), int(g * 255), int(b * 255), alpha)
    return palette


def main() -> None:  # noqa: C901 – Function intentionally lengthy for linear CLI flow.
    """Entrypoint executed when the script is run as a module."""
    args = parse_arguments()

    # Prepare output directory and logging.
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.log and not args.log.is_absolute():
        args.log = args.outdir / args.log
    configure_logging(logfile=args.log)
    set_global_seed(args.seed)
    logging.info("Output directory: %s", args.outdir.resolve())

    # ---------------------------------------------------------------------
    # Load inputs: raw image, segmentation mask, coordinates & features.
    # ---------------------------------------------------------------------
    img = Image.open(args.image).convert("RGB")  # Load raw RGB image.

    if args.labels is not None:

        if args.label_map is None:
            logging.error("--label_map is required when --labels is used.")
            sys.exit(1)

        label_ids = np.load(args.labels).astype(int)
        label_map = np.load(args.label_map, mmap_mode="r")
        # Build a Boolean stack on-the-fly for overlay only:
        mask = np.array([label_map == lab for lab in label_ids], dtype=bool)

    else:
        mask = np.load(args.mask, mmap_mode="r")

    coords_df = pd.read_csv(args.coords)         # Patch coords for clustering.
    features_df = pd.read_csv(args.features)     # DINO-ViT features per patch.
    if len(coords_df) != len(features_df):
        raise ValueError("Mismatch between rows in coords and features CSVs.")

    # ---------------------------------------------------------------------
    # Feature standardisation and clustering.
    # ---------------------------------------------------------------------
    scaler = StandardScaler()  # Scale to zero mean & unit variance.
    features_scaled = scaler.fit_transform(features_df.values)

    k = args.clusters
    if args.auto_k != "none":
        k, scores_summary = choose_optimal_k(features_scaled, k_max=k, criterion=args.auto_k)
        scores_summary.to_csv(args.outdir / "cluster_selection_scores.csv", index=False)
        logging.info("Cluster selection scores saved.")

    kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init="auto")
    labels = kmeans.fit_predict(features_scaled)
    coords_df["cluster"] = labels
    coords_df.to_csv(args.outdir / "patch_clusters.csv", index=False)
    logging.info("Cluster assignments saved.")
    joblib.dump(kmeans, args.outdir / "kmeans_model.joblib")
    joblib.dump(scaler, args.outdir / "scaler.joblib")

    # ---------------------------------------------------------------------
    # Overlay clusters on image via NumPy blending (handles any mask shape).
    # ---------------------------------------------------------------------
    # Determine slices: channels vs Z-stack vs single 2D
    if mask.ndim == 3 and mask.shape[2] <= 4:
        # treat as multi-channel mask: one slice per channel if desired
        slice_indices = [0]
        get_slice = lambda m, i: m[:, :, i]
    elif mask.ndim == 3:
        # treat first axis as Z-stack
        slice_indices = range(mask.shape[0])
        get_slice = lambda m, i: m[i]
    elif mask.ndim == 2:
        slice_indices = [None]
        get_slice = lambda m, _: m
    else:
        raise ValueError(f"Unexpected mask ndim={mask.ndim}. Expected 2 or 3.")

    # Prepare color palette for clustering overlay: vivid & semi-opaque.
    raw_palette = generate_color_palette(k, alpha=230)  # RGBA tuples per cluster.
    default_grey = (160, 160, 160, 230)

    out_path = args.outdir / "overlay_clusters.tif"

    # Overlay clusters on (optionally) cropped image.
    logging.info("Overlay crop region: %s", args.region)
    save_overlay_from_masks(
        img=img,
        masks=mask,
        labels=labels,
        palette=raw_palette,
        out_path=args.outdir / "overlay_clusters.tif",
        alpha=0.35,
        region=tuple(args.region),
    )

    logging.info("Overlay saved: %s", out_path.name)

    # ---------------------------------------------------------------------
    # PCA embedding plot of patch features.
    # ---------------------------------------------------------------------
    pca = PCA(n_components=2, random_state=args.seed)
    pcs = pca.fit_transform(features_scaled)
    pcs_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pcs_df["cluster"] = labels
    pcs_df.to_csv(args.outdir / "patch_pca_components.csv", index=False)

    plt.figure(figsize=(5, 5))
    for cl in range(k):
        subset = pcs_df[pcs_df["cluster"] == cl]
        color = np.array(raw_palette[cl])[:3] / 255.0
        plt.scatter(subset["PC1"], subset["PC2"], s=8,
                    label=f"Cluster {cl}", color=color)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("PCA of DINO‑ViT Patch Features")
    plt.legend(frameon=False, loc="best", markerscale=2)
    plt.tight_layout()
    plt.savefig(args.outdir / "pca_clusters.png", dpi=300)
    plt.close()

    logging.info("All outputs written to %s.", args.outdir.resolve())


if __name__ == "__main__":
    main()
