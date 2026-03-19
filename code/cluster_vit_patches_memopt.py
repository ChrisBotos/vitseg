"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: cluster_vit_patches_memopt.py.
Description:
    Memory-optimized clustering of ViT patch embeddings with MiniBatchKMeans and streaming scaling,
    preserving full overlay, palette generation, PCA plotting, and CSV fallback.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scikit-learn, joblib, matplotlib, PIL, seaborn, scipy.

Usage:
    python cluster_vit_patches_memopt.py --image data.tif \
        --labels filtered_passed_labels.npy --label_map segmentation_masks.npy \
        --coords coords.csv --features_npy features.npy --features_csv features.csv \
        --clusters 10 --auto-k silhouette --batch-size 10000 \
        --outdir results --seed 0 --region 0.1 0.9 0.1 0.9

"""
import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import random

from generate_contrast_colors import generate_color_palette, colors_to_hex_list

"""Compute slicing indices from fractional region specification."""
def _slice_region(height: int, width: int, region: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """
    Compute integer slice indices (x0, y0, x1, y1) from fractional region.
    Region format: (xmin, xmax, ymin, ymax) as fractions of width and height.
    """
    xmin, xmax, ymin, ymax = region
    x0 = max(0, int(width * xmin))
    x1 = min(width, int(width * xmax))
    y0 = max(0, int(height * ymin))
    y1 = min(height, int(height * ymax))
    return x0, x1, y0, y1

"""Build a lookup table from label IDs to cluster indices."""
def _build_lut(max_label: int, labels: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    """
    Create a lookup table (LUT) of size max_label+1 mapping original labels to cluster index+1.
    Entry 0 remains 0 (transparent background).
    Maps cluster 0->1, cluster 1->2, etc. for proper color indexing.
    """
    lut = np.zeros(max_label + 1, dtype=np.int32)

    for lab, cid in zip(labels, cluster_ids):
        if 0 <= lab <= max_label:
            lut[int(lab)] = int(cid) + 1  # Map cluster 0->1, cluster 1->2, etc.

    return lut

''' Configure root logger '''
def configure_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

''' Set global random seeds for reproducibility '''
def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

''' Compute mapping from cluster IDs back onto full segmentation mask '''
def compute_label_cluster_map(label_map, passed_labels, coords_df, cluster_ids):
    mapped = np.zeros_like(label_map, dtype=np.int32)
    for lab, cid in zip(passed_labels, cluster_ids):
        mask = (label_map == lab)
        mapped[mask] = cid + 1  # 0 reserved.
    return mapped

''' Save RGBA overlay of clusters on the image '''
def save_overlay(img_path: Path, seg_map_path: Path, passed_labels: np.ndarray, cluster_ids: np.ndarray,
                color_palette: Dict[int, Tuple[int, int, int, int]], region, out_path: Path, down: int = 1) -> None:
    """
    Create RGBA overlay of cluster colors on microscopy image.

    Args:
        img_path: Path to base microscopy image.
        seg_map_path: Path to segmentation mask array.
        passed_labels: Array of cell labels that passed filtering.
        cluster_ids: Array of cluster assignments for each label.
        color_palette: Dictionary mapping cluster index to RGBA color.
        region: Crop region as (xmin, xmax, ymin, ymax) fractions.
        out_path: Output path for overlay image.
        down: Downsampling factor for memory efficiency.
    """
    print(f"DEBUG: Creating overlay with {len(color_palette)} colors")
    print(f"DEBUG: Region: {region}, downsample: {down}")

    # Load and crop base image.
    base_full = Image.open(img_path).convert('RGBA')
    w, h = base_full.size
    x0, x1, y0, y1 = _slice_region(h, w, region)
    base_crop = base_full.crop((x0, y0, x1, y1))

    # Load and crop segmentation mask.
    seg = np.load(seg_map_path, mmap_mode='r')[y0:y1, x0:x1]

    if down > 1:
        seg = seg[::down, ::down]
        base_crop = base_crop.resize((seg.shape[1], seg.shape[0]), Image.BILINEAR)

    # Create lookup table mapping labels to cluster colors.
    lut = _build_lut(int(seg.max()), passed_labels, cluster_ids)
    cluster_map = lut[seg]

    # Build RGBA color array with background as transparent.
    k = len(color_palette)
    rgba_array = np.zeros((k + 1, 4), dtype=np.uint8)  # +1 for background (index 0).

    # Populate color array with palette colors.
    # cluster_map uses 1-based indexing (0=background, 1=cluster0, 2=cluster1, etc.).
    for cluster_idx, (r, g, b, a) in color_palette.items():
        array_idx = cluster_idx + 1  # Map cluster 0->index 1, cluster 1->index 2, etc.
        if array_idx < len(rgba_array):
            rgba_array[array_idx] = [r, g, b, a]

    # Create overlay image using cluster map as indices.
    overlay_data = rgba_array[cluster_map]
    overlay = Image.fromarray(overlay_data, mode='RGBA')

    # Composite overlay onto base image.
    composite = Image.alpha_composite(base_crop, overlay)
    composite.save(out_path)

    logging.info(f'Overlay saved with {np.sum(cluster_map > 0)} colored pixels → {out_path}')
    print(f"DEBUG: Overlay dimensions: {composite.size}, unique clusters: {len(np.unique(cluster_map[cluster_map > 0]))}")

''' Choose optimal K via silhouette or Davies-Bouldin '''
def choose_optimal_k(features, k_max, criterion, sample_size=5000):
    idx = np.random.RandomState(0).choice(features.shape[0], min(sample_size, features.shape[0]), replace=False)
    sample = features[idx]
    scores = []
    for k in range(2, k_max+1):
        km = MiniBatchKMeans(n_clusters=k, random_state=0)
        labels = km.fit_predict(sample)
        score = silhouette_score(sample, labels) if criterion=='silhouette' else davies_bouldin_score(sample, labels)
        scores.append((k, score))
    df = pd.DataFrame(scores, columns=['k', criterion])
    best = int(df.loc[df[criterion].idxmax() if criterion=='silhouette' else df[criterion].idxmin(), 'k'])
    logging.info(f'Optimal K selected: {best}')
    return best, df

''' Load features from .npy or CSV fallback '''
def load_features(path_npy: Path, path_csv: Path):
    if path_npy.exists():
        return np.load(path_npy, mmap_mode='r')
    if path_csv.exists():
        return pd.read_csv(path_csv).values
    logging.error('No feature array found at .npy or .csv')
    sys.exit(1)

''' Stream-scaling using partial StandardScaler '''
def scale_features(data, batch_size):
    scaler = StandardScaler()
    n = data.shape[0]
    for i in range(0, n, batch_size):
        scaler.partial_fit(data[i:i+batch_size])
    return scaler

''' Stream-clustering with MiniBatchKMeans '''
def cluster_streaming(data, batch_size, k, seed):
    mbk = MiniBatchKMeans(n_clusters=k, random_state=seed)
    n = data.shape[0]
    for i in range(0, n, batch_size):
        mbk.partial_fit(data[i:i+batch_size])
    return mbk

''' Main pipeline '''
def main():
    parser = argparse.ArgumentParser(description='Memory-efficient clustering of ViT patch embeddings.')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--label_map', type=Path, required=True)
    parser.add_argument('--coords', type=Path, required=True)
    parser.add_argument('--features_npy', type=Path, default=Path('features.npy'))
    parser.add_argument('--features_csv', type=Path, default=Path('features.csv'))
    parser.add_argument('--clusters', type=int, default=10)
    parser.add_argument('--auto-k', choices=['none','silhouette','dbi'], default='none')
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--outdir', type=Path, default=Path('results'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--region', type=float, nargs=4)
    parser.add_argument('--downsample', type=int, default=1, help='Downsampling factor for overlay (integer > 1).')
    parser.add_argument('--test', action='store_true', help='Run unit tests.')

    args = parser.parse_args()

    if args.test:
        import unittest
        class TestUtils(unittest.TestCase):
            def test_slice_region(self):
                h, w = 100, 200
                region = (0.1, 0.5, 0.2, 0.6)
                x0, x1, y0, y1 = _slice_region(h, w, region)
                self.assertEqual((x0, x1, y0, y1), (20, 100, 20, 60))

            def test_build_lut(self):
                labels = np.array([1, 3, 5])
                clusters = np.array([0, 2, 4])
                lut = _build_lut(6, labels, clusters)
                expected = np.zeros(7, dtype=np.int32)
                expected[1] = 1
                expected[3] = 3
                expected[5] = 5
                np.testing.assert_array_equal(lut, expected)

        unittest.main(argv=[sys.argv[0]])
        return

    configure_logging()
    set_global_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    logging.info('Loading features...')
    features = load_features(args.features_npy, args.features_csv)

    logging.info('Scaling features...')
    scaler = scale_features(features, args.batch_size)
    joblib.dump(scaler, args.outdir/'scaler.joblib')

    logging.info('Transforming features and saving memmap...')
    n, m = features.shape
    scaled_fp = args.outdir/'features_scaled.npy'
    scaled = np.memmap(scaled_fp, dtype='float32', mode='w+', shape=(n, m))
    for i in range(0, n, args.batch_size):
        scaled[i:i+args.batch_size] = scaler.transform(features[i:i+args.batch_size])

    if args.auto_k!='none':
        k, df_scores = choose_optimal_k(scaled, args.clusters, args.auto_k)
        df_scores.to_csv(args.outdir/'cluster_selection_scores.csv', index=False)
    else:
        k = args.clusters

    # Generate high-contrast colors optimized for scientific visualization.
    print(f"DEBUG: Generating enhanced color palette for {k} clusters")

    # Try to load color configuration if available.
    try:
        from color_config import load_color_config
        color_config = load_color_config()
        print("DEBUG: Using ColorConfig system for enhanced color generation")
        color_palette = color_config.generate_palette(n=k)
    except ImportError:
        print("DEBUG: ColorConfig not available, using direct color generation")
        color_palette = generate_color_palette(
            n=k,
            alpha=255,   # (0 to 255) 0=transparent, 255=opaque.
            background="dark",  # Dark background for high-contrast PCA plots.
            saturation=0.95,    # High saturation for better distinction.
            contrast_ratio=10, # High contrast for clear visibility.
            hue_start=0.0      # Offset to avoid starting with red.
        )

    # Convert to hex format for matplotlib/seaborn compatibility.
    hex_colors = colors_to_hex_list(color_palette)

    print(f"DEBUG: Generated {len(color_palette)} RGBA colors and {len(hex_colors)} hex colors")

    # Display first few colors for verification.
    for i in range(min(3, len(color_palette))):
        r, g, b, a = color_palette[i]
        hex_color = hex_colors[i] if i < len(hex_colors) else "N/A"
        print(f"DEBUG: Color {i}: RGB({r}, {g}, {b}) -> {hex_color}")


    logging.info(f'Clustering into {k} clusters...')
    kmeans = cluster_streaming(scaled, args.batch_size, k, args.seed)
    joblib.dump(kmeans, args.outdir/'kmeans_model.joblib')

    logging.info('Predicting cluster labels...')
    labels = np.zeros(n, dtype=int)
    for i in range(0, n, args.batch_size):
        labels[i:i+args.batch_size] = kmeans.predict(scaled[i:i+args.batch_size])

    coords_df = pd.read_csv(args.coords)
    coords_df['cluster'] = labels
    coords_df.to_csv(args.outdir/'patch_clusters.csv', index=False)


    logging.info('Plotting PCA scatter...')
    sample_idx = np.random.RandomState(args.seed).choice(n, min(5000, n), replace=False)
    pcs = PCA(2).fit_transform(scaled[sample_idx])

    # Create high-resolution PCA plot with dark background.
    plt.figure(figsize=(12, 10))
    plt.style.use('dark_background')

    # Use hex colors for seaborn compatibility.
    sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=labels[sample_idx],
                   palette=hex_colors, legend=True, s=80, alpha=0.8,
                   edgecolor='black', linewidth=0.3)

    # High-quality text with better visibility.
    plt.title('PCA Visualization of Patch Clusters', fontsize=20, fontweight='bold',
              color='white', pad=25)
    plt.xlabel('First Principal Component', fontsize=16, color='white', fontweight='bold')
    plt.ylabel('Second Principal Component', fontsize=16, color='white', fontweight='bold')

    # Improve tick labels.
    plt.xticks(fontsize=14, color='white', fontweight='bold')
    plt.yticks(fontsize=14, color='white', fontweight='bold')

    # Enhanced legend with better visibility.
    legend = plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left',
                       frameon=True, fancybox=True, shadow=True,
                       facecolor='black', edgecolor='white', fontsize=12)
    legend.get_title().set_color('white')
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(14)

    for text in legend.get_texts():
        text.set_color('white')
        text.set_fontweight('bold')

    # Subtle grid for better readability.
    plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')

    plt.tight_layout()
    plt.savefig(args.outdir/'pca_clusters.png', dpi=400, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()

    logging.info(f'PCA plot saved with {len(hex_colors)} distinct colors.')
    print(f"DEBUG: PCA plot uses {len(np.unique(labels[sample_idx]))} unique cluster labels")

    logging.info('Creating overlay…')
    passed = np.load(args.labels)

    print(f"DEBUG: Creating overlay for {len(passed)} passed labels with {len(color_palette)} cluster colors")

    save_overlay(args.image, args.label_map, passed, labels, color_palette,
                region=args.region, out_path=args.outdir / 'overlay_clusters.tif',
                down=args.downsample)

    print("DEBUG: Overlay creation completed successfully")

if __name__ == '__main__':
    main()
