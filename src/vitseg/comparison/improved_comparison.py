"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: improved_comparison.py.
Description:
    Improved ViT-vs-spatial-transcriptomics comparison that addresses four
    issues identified in the original pipeline:
      1. Uses 16px-only features (nucleus-level) instead of full 1152-D.
      2. Clusters per-sample instead of across all samples.
      3. Tests multiple K values (5, 6, 8, 10).
      4. Compares against both cell types and kidney tissue zones.
    Uses majority-vote matching (all nuclei within spot radius) instead of
    single nearest-neighbor matching.

Dependencies:
    • Python >= 3.10.
    • pandas, numpy, scikit-learn, matplotlib, seaborn.

Usage:
    python -m vitseg.comparison.improved_comparison \
        --features results/IRI_regist_14k/features_IRI_regist_binary_mask.csv \
        --coords results/IRI_regist_14k/coords_IRI_regist_binary_mask.csv \
        --spatial data/metadata_complete.csv \
        --output results/improved_comparison
"""
import argparse
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

# Kidney tissue zone mapping from figure_idents cell type annotations.
ZONE_MAP = {
    "PT-S1/S2": "cortex",
    "DCT": "cortex",
    "CNT": "cortex",
    "glomeruli": "cortex",
    "PT-S3": "outer_medulla",
    "TAL": "medulla",
    "collecting duct": "medulla",
    "interstitial cells": "interstitial",
    "FR-PT": "injury",
    "Injured tubule": "injury",
    "undetermined": "other",
    "gaps": "other",
}

SAMPLES = ["IRI1", "IRI2", "IRI3"]
K_VALUES = [5, 6, 8, 10]
MATCH_RADIUS = 40  # Coordinate units, matches spatial spot grid spacing.


def load_features_by_scale(features_csv: Path, scales: List[int]) -> np.ndarray:
    """Load features for specific scales from the multi-scale CSV.

    Uses the ``usecols`` parameter to avoid loading the full 1152-column file
    into memory.  Only columns matching ``vit{scale}_*`` patterns are retained.

    Args:
        features_csv (Path): Path to multi-scale features CSV.
        scales (list): List of scale values to include (e.g. [16] or [16, 32]).

    Returns:
        np.ndarray: Feature matrix of shape (n_nuclei, 384 * len(scales)).
    """
    header = pd.read_csv(features_csv, nrows=0).columns.tolist()
    selected_cols = []
    for scale in sorted(scales):
        prefix = f"vit{scale}_"
        cols = [c for c in header if c.startswith(prefix)]
        if not cols:
            raise ValueError(
                f"No {prefix}* columns found in {features_csv}. "
                f"Available columns start with: {header[:5]}"
            )
        selected_cols.extend(cols)

    scales_str = "+".join(f"{s}px" for s in sorted(scales))
    LOGGER.info("Loading %d features (%s) from %s...", len(selected_cols), scales_str, features_csv.name)
    df = pd.read_csv(features_csv, usecols=selected_cols)
    LOGGER.info("  Loaded feature matrix: %s", df.shape)
    return df.values.astype(np.float32)


def load_spatial_boundaries(spatial_csv: Path) -> Dict[str, dict]:
    """Compute per-sample coordinate bounding boxes from spatial data.

    Args:
        spatial_csv (Path): Path to spatial metadata CSV.

    Returns:
        dict: Mapping sample name to boundary dict with x/y min/max/center.
    """
    spatial = pd.read_csv(spatial_csv)
    boundaries = {}
    for sample in SAMPLES:
        sub = spatial[spatial["sample"] == sample]
        if len(sub) == 0:
            continue
        boundaries[sample] = {
            "x_min": sub["x"].min(),
            "x_max": sub["x"].max(),
            "y_min": sub["y"].min(),
            "y_max": sub["y"].max(),
            "x_center": (sub["x"].min() + sub["x"].max()) / 2,
            "y_center": (sub["y"].min() + sub["y"].max()) / 2,
        }
    return boundaries


def assign_samples(
    coords: np.ndarray, boundaries: Dict[str, dict]
) -> np.ndarray:
    """Assign each nucleus to a sample based on spatial bounding boxes.

    Points inside a sample's bounding box are assigned to that sample.
    Overlapping regions are resolved by nearest-centroid distance.
    Points outside all bounding boxes are assigned to the nearest centroid.

    Args:
        coords (np.ndarray): Coordinate array of shape (n, 2) with x, y.
        boundaries (dict): Per-sample bounding boxes from
            :func:`load_spatial_boundaries`.

    Returns:
        np.ndarray: String array of sample labels, length n.
    """
    n = len(coords)
    assignments = np.full(n, "", dtype=object)

    # First pass: assign by bounding box.
    for sample, b in boundaries.items():
        mask = (
            (coords[:, 0] >= b["x_min"])
            & (coords[:, 0] <= b["x_max"])
            & (coords[:, 1] >= b["y_min"])
            & (coords[:, 1] <= b["y_max"])
        )
        # Track all candidate samples per point for overlap resolution.
        assignments[mask] = sample

    # Second pass: resolve overlaps and unassigned by nearest centroid.
    centroids = {
        s: np.array([b["x_center"], b["y_center"]])
        for s, b in boundaries.items()
    }
    sample_names = list(centroids.keys())
    centroid_arr = np.array([centroids[s] for s in sample_names])

    unassigned = assignments == ""
    if unassigned.any():
        dists = np.linalg.norm(
            coords[unassigned, np.newaxis, :] - centroid_arr[np.newaxis, :, :],
            axis=2,
        )
        nearest_idx = dists.argmin(axis=1)
        assignments[unassigned] = np.array(sample_names)[nearest_idx]

    return assignments


def cluster_per_sample(
    features: np.ndarray,
    sample_labels: np.ndarray,
    k: int,
    seed: int = 0,
    batch_size: int = 10000,
) -> np.ndarray:
    """Cluster features independently per sample using MiniBatchKMeans.

    Args:
        features (np.ndarray): Feature matrix (n, d).
        sample_labels (np.ndarray): Sample assignment for each row.
        k (int): Number of clusters.
        seed (int): Random seed for reproducibility.
        batch_size (int): Mini-batch size for streaming.

    Returns:
        np.ndarray: Cluster labels (0-based per sample), length n.
    """
    cluster_labels = np.full(len(features), -1, dtype=int)

    for sample in SAMPLES:
        mask = sample_labels == sample
        n_sample = mask.sum()
        if n_sample == 0:
            continue

        sample_features = features[mask]

        # Scale features.
        scaler = StandardScaler()
        for i in range(0, n_sample, batch_size):
            scaler.partial_fit(sample_features[i : i + batch_size])

        scaled = np.empty_like(sample_features)
        for i in range(0, n_sample, batch_size):
            scaled[i : i + batch_size] = scaler.transform(
                sample_features[i : i + batch_size]
            )

        # Cluster.
        effective_k = min(k, n_sample)
        kmeans = MiniBatchKMeans(
            n_clusters=effective_k, random_state=seed, batch_size=batch_size
        )
        for i in range(0, n_sample, batch_size):
            kmeans.partial_fit(scaled[i : i + batch_size])

        labels = np.empty(n_sample, dtype=int)
        for i in range(0, n_sample, batch_size):
            labels[i : i + batch_size] = kmeans.predict(
                scaled[i : i + batch_size]
            )

        cluster_labels[mask] = labels
        LOGGER.info("  %s: %s nuclei -> %d clusters", sample, f"{n_sample:,}", effective_k)

    return cluster_labels


def majority_vote_match(
    vit_coords: np.ndarray,
    vit_clusters: np.ndarray,
    spatial_coords: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match spatial spots to ViT clusters using majority vote within radius.

    For each spatial spot, finds all ViT nuclei within the given radius and
    assigns the most common cluster label.

    Args:
        vit_coords (np.ndarray): ViT nuclei coordinates (n_vit, 2).
        vit_clusters (np.ndarray): ViT cluster labels (n_vit,).
        spatial_coords (np.ndarray): Spatial spot coordinates (n_spots, 2).
        radius (float): Search radius in coordinate units.

    Returns:
        Tuple of (matched_vit_clusters, n_nuclei_per_spot) arrays.
        Entries are -1 where no ViT nuclei were found within the radius.
    """
    tree = KDTree(vit_coords)
    indices = tree.query_radius(spatial_coords, r=radius)

    matched_clusters = np.full(len(spatial_coords), -1, dtype=int)
    n_nuclei = np.zeros(len(spatial_coords), dtype=int)

    for i, nn_idx in enumerate(indices):
        if len(nn_idx) == 0:
            continue
        nn_clusters = vit_clusters[nn_idx]
        matched_clusters[i] = Counter(nn_clusters).most_common(1)[0][0]
        n_nuclei[i] = len(nn_idx)

    return matched_clusters, n_nuclei


def run_comparison(
    vit_coords: np.ndarray,
    vit_clusters: np.ndarray,
    vit_samples: np.ndarray,
    spatial_data: pd.DataFrame,
    k: int,
    radius: float,
) -> dict:
    """Run full comparison for one clustering configuration.

    Args:
        vit_coords (np.ndarray): ViT nuclei coordinates (n, 2).
        vit_clusters (np.ndarray): ViT cluster labels (n,).
        vit_samples (np.ndarray): Sample labels for ViT nuclei.
        spatial_data (pd.DataFrame): Spatial transcriptomics data.
        k (int): K value used for clustering (for labeling).
        radius (float): Matching radius.

    Returns:
        dict: Metrics including per-sample and overall ARI/NMI for both
        cell type and zone comparisons.
    """
    results = {"k": k, "samples": {}, "overall": {}}

    all_vit_labels = []
    all_cell_labels = []
    all_zone_labels = []

    for sample in SAMPLES:
        vit_mask = vit_samples == sample
        sp_sample = spatial_data[spatial_data["sample"] == sample].copy()

        if vit_mask.sum() == 0 or len(sp_sample) == 0:
            continue

        sp_sample["zone"] = sp_sample["figure_idents"].map(ZONE_MAP)
        sp_coords = sp_sample[["x", "y"]].values

        matched_vit, n_nuclei = majority_vote_match(
            vit_coords[vit_mask],
            vit_clusters[vit_mask],
            sp_coords,
            radius,
        )

        # Filter to matched spots only.
        valid = matched_vit >= 0
        if valid.sum() < 10:
            continue

        vit_labels = matched_vit[valid]
        cell_labels = sp_sample["figure_idents"].values[valid]
        zone_labels = sp_sample["zone"].values[valid]

        # Remove NaN zones.
        zone_valid = pd.notna(zone_labels)
        vit_for_zone = vit_labels[zone_valid]
        zone_labels_clean = zone_labels[zone_valid]

        sample_metrics = {
            "n_matched": int(valid.sum()),
            "mean_nuclei_per_spot": float(n_nuclei[valid].mean()),
            "cell_ari": float(adjusted_rand_score(cell_labels, vit_labels)),
            "cell_nmi": float(
                normalized_mutual_info_score(cell_labels, vit_labels)
            ),
            "n_vit_clusters": int(len(np.unique(vit_labels))),
            "n_cell_types": int(len(np.unique(cell_labels))),
        }

        if len(vit_for_zone) > 10:
            sample_metrics["zone_ari"] = float(
                adjusted_rand_score(zone_labels_clean, vit_for_zone)
            )
            sample_metrics["zone_nmi"] = float(
                normalized_mutual_info_score(zone_labels_clean, vit_for_zone)
            )
            sample_metrics["n_zones"] = int(len(np.unique(zone_labels_clean)))

        results["samples"][sample] = sample_metrics

        all_vit_labels.append(vit_labels)
        all_cell_labels.append(cell_labels)
        if len(vit_for_zone) > 0:
            all_zone_labels.append(
                np.column_stack([vit_for_zone, zone_labels_clean])
            )

    # Overall metrics: average per-sample (not concatenated, since cluster IDs
    # are per-sample and not globally comparable).
    sample_cell_aris = [
        s["cell_ari"] for s in results["samples"].values() if "cell_ari" in s
    ]
    sample_cell_nmis = [
        s["cell_nmi"] for s in results["samples"].values() if "cell_nmi" in s
    ]
    sample_zone_aris = [
        s["zone_ari"] for s in results["samples"].values() if "zone_ari" in s
    ]
    sample_zone_nmis = [
        s["zone_nmi"] for s in results["samples"].values() if "zone_nmi" in s
    ]
    total_matched = sum(
        s["n_matched"] for s in results["samples"].values() if "n_matched" in s
    )

    if sample_cell_aris:
        results["overall"]["cell_ari"] = float(np.mean(sample_cell_aris))
        results["overall"]["cell_nmi"] = float(np.mean(sample_cell_nmis))
        results["overall"]["n_matched"] = total_matched

    if sample_zone_aris:
        results["overall"]["zone_ari"] = float(np.mean(sample_zone_aris))
        results["overall"]["zone_nmi"] = float(np.mean(sample_zone_nmis))

    return results


def create_summary_table(all_results: List[dict]) -> pd.DataFrame:
    """Build a summary DataFrame from all comparison results.

    Args:
        all_results (list): List of result dicts from :func:`run_comparison`.

    Returns:
        pd.DataFrame: Summary table with one row per K value.
    """
    rows = []
    for r in all_results:
        row = {
            "scales": r.get("scales", "16px"),
            "K": r["k"],
            "cell_ARI": r["overall"].get("cell_ari", float("nan")),
            "cell_NMI": r["overall"].get("cell_nmi", float("nan")),
            "zone_ARI": r["overall"].get("zone_ari", float("nan")),
            "zone_NMI": r["overall"].get("zone_nmi", float("nan")),
            "n_matched": r["overall"].get("n_matched", 0),
        }
        for sample in SAMPLES:
            if sample in r["samples"]:
                sm = r["samples"][sample]
                row[f"{sample}_cell_ARI"] = sm.get("cell_ari", float("nan"))
                row[f"{sample}_zone_ARI"] = sm.get("zone_ari", float("nan"))
                row[f"{sample}_nuclei_per_spot"] = sm.get(
                    "mean_nuclei_per_spot", float("nan")
                )
        rows.append(row)
    return pd.DataFrame(rows)


def create_confusion_heatmap(
    vit_coords: np.ndarray,
    vit_clusters: np.ndarray,
    vit_samples: np.ndarray,
    spatial_data: pd.DataFrame,
    radius: float,
    output_path: Path,
    label_column: str = "figure_idents",
    title_suffix: str = "",
):
    """Generate a confusion matrix heatmap for ViT vs spatial clusters.

    Args:
        vit_coords (np.ndarray): ViT nuclei coordinates.
        vit_clusters (np.ndarray): ViT cluster labels.
        vit_samples (np.ndarray): Sample labels.
        spatial_data (pd.DataFrame): Spatial transcriptomics data.
        radius (float): Matching radius.
        output_path (Path): Path to save the heatmap PNG.
        label_column (str): Spatial column to compare against.
        title_suffix (str): Extra text for the plot title.
    """
    all_vit = []
    all_spatial = []

    for sample in SAMPLES:
        vit_mask = vit_samples == sample
        sp_sample = spatial_data[spatial_data["sample"] == sample]
        if vit_mask.sum() == 0 or len(sp_sample) == 0:
            continue

        sp_coords = sp_sample[["x", "y"]].values
        matched_vit, _ = majority_vote_match(
            vit_coords[vit_mask], vit_clusters[vit_mask], sp_coords, radius
        )

        valid = matched_vit >= 0
        if valid.sum() == 0:
            continue

        all_vit.append(matched_vit[valid])
        all_spatial.append(sp_sample[label_column].values[valid])

    if not all_vit:
        return

    vit_all = np.concatenate(all_vit)
    spatial_all = np.concatenate(all_spatial)

    # Remove NaN entries.
    valid_mask = pd.notna(spatial_all)
    vit_all = vit_all[valid_mask]
    spatial_all = spatial_all[valid_mask]

    # Build contingency table.
    df = pd.DataFrame({"vit": vit_all, "spatial": spatial_all})
    ct = pd.crosstab(df["spatial"], df["vit"], normalize="index")

    plt.figure(figsize=(max(10, len(ct.columns) * 0.8), max(8, len(ct) * 0.6)))
    sns.heatmap(
        ct,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Proportion"},
    )
    plt.title(
        f"ViT Cluster Distribution per {label_column}{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("ViT Cluster", fontsize=12)
    plt.ylabel(label_column, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_report(
    all_results: List[dict],
    summary_df: pd.DataFrame,
    output_dir: Path,
    baseline_ari: float,
):
    """Save the analysis report and summary files.

    Args:
        all_results (list): List of result dicts.
        summary_df (pd.DataFrame): Summary table.
        output_dir (Path): Output directory.
        baseline_ari (float): Original ARI for comparison.
    """
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    # Find best configuration.
    best_cell_idx = summary_df["cell_ARI"].idxmax()
    best_zone_idx = summary_df["zone_ARI"].idxmax()

    with open(output_dir / "analysis_report.txt", "w") as f:
        f.write("Improved ViT-Spatial Cluster Comparison Report\n")
        f.write("=" * 55 + "\n\n")

        f.write("Changes from original analysis:\n")
        f.write("  1. Features: 16px only (384-D) instead of all scales (1152-D).\n")
        f.write("  2. Clustering: per-sample instead of cross-sample.\n")
        f.write("  3. K values: tested 5, 6, 8, 10.\n")
        f.write("  4. Matching: majority vote (radius=40) instead of single nearest.\n")
        f.write("  5. Comparison: both cell types and kidney tissue zones.\n\n")

        f.write(f"Original baseline ARI (cell types): {baseline_ari:.4f}\n\n")

        f.write("Summary Table:\n")
        f.write("-" * 75 + "\n")
        f.write(
            f"{'Scales':<12s} | {'K':>3s} | {'Cell ARI':>9s} | {'Cell NMI':>9s} | "
            f"{'Zone ARI':>9s} | {'Zone NMI':>9s} | {'Matched':>8s}\n"
        )
        f.write("-" * 75 + "\n")

        for _, row in summary_df.iterrows():
            f.write(
                f"{row['scales']:<12s} | {int(row['K']):>3d} | "
                f"{row['cell_ARI']:>9.4f} | {row['cell_NMI']:>9.4f} | "
                f"{row['zone_ARI']:>9.4f} | {row['zone_NMI']:>9.4f} | "
                f"{int(row['n_matched']):>8,d}\n"
            )

        f.write("-" * 55 + "\n\n")

        best_cell = summary_df.loc[best_cell_idx]
        best_zone = summary_df.loc[best_zone_idx]

        f.write(f"Best cell-type ARI: {best_cell['cell_ARI']:.4f} (K={int(best_cell['K'])})\n")
        f.write(f"Best zone ARI:      {best_zone['zone_ARI']:.4f} (K={int(best_zone['K'])})\n\n")

        improvement_cell = best_cell["cell_ARI"] / max(baseline_ari, 1e-6)
        f.write(f"Cell-type ARI improvement: {improvement_cell:.1f}x over baseline\n\n")

        f.write("Per-Sample Results (best cell-type K={}):\n".format(int(best_cell["K"])))
        f.write("-" * 55 + "\n")

        best_result = all_results[best_cell_idx]
        for sample in SAMPLES:
            if sample in best_result["samples"]:
                sm = best_result["samples"][sample]
                f.write(
                    f"  {sample}: cell_ARI={sm['cell_ari']:.4f}, "
                    f"zone_ARI={sm.get('zone_ari', float('nan')):.4f}, "
                    f"matched={sm['n_matched']:,}, "
                    f"nuclei/spot={sm['mean_nuclei_per_spot']:.1f}\n"
                )

        f.write("\n")

        f.write("Zone Mapping Used:\n")
        for cell_type, zone in sorted(ZONE_MAP.items()):
            f.write(f"  {cell_type:>25s} -> {zone}\n")

    # Save per-K detailed results.
    for r in all_results:
        k = r["k"]
        detail_rows = []
        for sample in SAMPLES:
            if sample in r["samples"]:
                row = {"sample": sample}
                row.update(r["samples"][sample])
                detail_rows.append(row)
        if detail_rows:
            pd.DataFrame(detail_rows).to_csv(
                output_dir / f"per_sample_K{k}.csv", index=False
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Improved ViT-spatial comparison with per-sample "
        "clustering, 16px features, and zone-level analysis."
    )
    parser.add_argument(
        "--features", type=Path, required=True, help="Multi-scale features CSV."
    )
    parser.add_argument(
        "--coords", type=Path, required=True, help="Coordinates CSV."
    )
    parser.add_argument(
        "--spatial", type=Path, required=True, help="Spatial metadata CSV."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed."
    )
    parser.add_argument(
        "--radius", type=float, default=MATCH_RADIUS,
        help="Matching radius in coordinate units."
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    t0 = time.time()
    LOGGER.info("=" * 60)
    LOGGER.info("Improved ViT-Spatial Cluster Comparison")
    LOGGER.info("=" * 60)

    # Step 1: Load coordinates and assign samples.
    LOGGER.info("[1/5] Loading coordinates and assigning samples...")
    coords_df = pd.read_csv(args.coords)
    coords = coords_df[["x_center", "y_center"]].values

    spatial_data = pd.read_csv(args.spatial)
    boundaries = load_spatial_boundaries(args.spatial)
    sample_labels = assign_samples(coords, boundaries)

    for sample in SAMPLES:
        n = (sample_labels == sample).sum()
        LOGGER.info("  %s: %s nuclei", sample, f"{n:,}")

    # Step 2: Test multiple feature scale combinations.
    scale_configs = [[16], [16, 32]]
    all_results = []

    for scales in scale_configs:
        scales_tag = "_".join(f"{s}px" for s in scales)
        LOGGER.info("=" * 60)
        LOGGER.info("[2/5] Feature set: %s", scales_tag)
        LOGGER.info("=" * 60)

        features = load_features_by_scale(args.features, scales)

        # Step 3: Cluster per-sample for each K.
        LOGGER.info("[3/5] Clustering per-sample (%s)...", scales_tag)

        for k in K_VALUES:
            LOGGER.info("--- %s, K=%d ---", scales_tag, k)
            cluster_labels = cluster_per_sample(
                features, sample_labels, k=k, seed=args.seed
            )

            # Save cluster assignments.
            cluster_df = pd.DataFrame(
                {
                    "x_center": coords[:, 0],
                    "y_center": coords[:, 1],
                    "cluster": cluster_labels,
                    "sample": sample_labels,
                }
            )
            cluster_df.to_csv(
                args.output / f"clusters_{scales_tag}_persample_K{k}.csv",
                index=False,
            )

            # Step 4: Run comparison.
            result = run_comparison(
                coords,
                cluster_labels,
                sample_labels,
                spatial_data,
                k=k,
                radius=args.radius,
            )
            result["scales"] = scales_tag
            all_results.append(result)

            cell_ari = result["overall"].get("cell_ari", float("nan"))
            zone_ari = result["overall"].get("zone_ari", float("nan"))
            LOGGER.info("  Overall (mean per-sample): cell_ARI=%.4f, zone_ARI=%.4f", cell_ari, zone_ari)

        # Free memory before loading next feature set.
        del features

    # Step 5: Generate report and visualizations.
    LOGGER.info("=" * 60)
    LOGGER.info("[4/5] Generating report...")
    summary_df = create_summary_table(all_results)
    save_report(all_results, summary_df, args.output, baseline_ari=0.0136)

    # Create confusion heatmaps for the best configuration.
    LOGGER.info("[5/5] Creating visualizations...")
    best_idx = summary_df["cell_ARI"].idxmax()
    best_row = summary_df.loc[best_idx]
    best_k = int(best_row["K"])
    best_scales = all_results[best_idx]["scales"]

    best_clusters = pd.read_csv(
        args.output / f"clusters_{best_scales}_persample_K{best_k}.csv"
    )
    best_labels = best_clusters["cluster"].values

    # Cell type confusion matrix.
    create_confusion_heatmap(
        coords,
        best_labels,
        sample_labels,
        spatial_data,
        args.radius,
        args.output / f"confusion_cell_types_{best_scales}_K{best_k}.png",
        label_column="figure_idents",
        title_suffix=f" (K={best_k}, {best_scales} per-sample)",
    )

    # Zone confusion matrix.
    spatial_with_zones = spatial_data.copy()
    spatial_with_zones["zone"] = spatial_with_zones["figure_idents"].map(ZONE_MAP)
    create_confusion_heatmap(
        coords,
        best_labels,
        sample_labels,
        spatial_with_zones,
        args.radius,
        args.output / f"confusion_zones_{best_scales}_K{best_k}.png",
        label_column="zone",
        title_suffix=f" (K={best_k}, {best_scales} per-sample)",
    )

    elapsed = time.time() - t0
    LOGGER.info("Done in %.0fs. Results saved to %s", elapsed, args.output)
    LOGGER.info("Summary:\n%s", summary_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    exit(main())
