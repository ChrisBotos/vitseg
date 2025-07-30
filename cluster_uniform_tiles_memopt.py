"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: cluster_uniform_tiles_memopt.py.
Description:
    Memory-optimized clustering of uniform tile ViT embeddings with MiniBatchKMeans
    and streaming scaling. Creates tile-based overlays and PCA visualizations for
    spatial pattern analysis in binary mask images.

    Key features for bioinformatician users:
        • **Tile-based clustering** – Clusters uniform tiles rather than individual
          cell patches, enabling analysis of tissue architecture and spatial patterns.
        • **Memory-efficient processing** – Uses streaming algorithms and batch
          processing to handle large numbers of tiles without memory overflow.
        • **Spatial overlay generation** – Creates overlays showing cluster assignments
          mapped back to tile positions in the original tissue image.
        • **Enhanced visualizations** – Generates high-quality PCA plots and color-coded
          overlays optimized for scientific publication.

    Scientific context:
        This approach is ideal for analyzing regional tissue heterogeneity, identifying
        spatial patterns in kidney injury models, and understanding tissue organization
        at the architectural level rather than individual cell level.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scikit-learn, joblib, matplotlib, PIL, seaborn, scipy.

Usage:
    python cluster_uniform_tiles_memopt.py \
        --image binary_mask.tif \
        --coords coords.csv \
        --features_npy features.npy \
        --clusters 10 \
        --auto-k silhouette \
        --outdir results

Arguments:
    --image            Path to binary mask image (TIFF format).
    --coords           CSV file with tile center coordinates.
    --features_npy     NumPy array of tile feature embeddings.
    --features_csv     CSV fallback for features if NPY not available.
    --clusters         Initial number of clusters for K-means.
    --auto-k           Auto-K selection method ('none', 'silhouette', 'dbi').
    --batch-size       Batch size for streaming processing.
    --outdir           Output directory for results.
    --seed             Random seed for reproducibility.
    --region           Crop region for overlay (xmin xmax ymin ymax).
    --downsample       Downsampling factor for overlay generation.

Inputs:
    • features.npy     Memory-mapped tile feature array (n_tiles, n_features).
    • coords.csv       Tile center coordinates (x_center, y_center).
    • binary_mask.tif  Binary mask image for overlay generation.

Outputs:
    • tile_clusters.csv        Cluster assignments per tile.
    • kmeans_model.joblib      Trained MiniBatchKMeans model.
    • scaler.joblib            StandardScaler for features.
    • overlay_clusters.tif     Tile cluster overlay on binary image.
    • pca_clusters.png         PCA scatter plot of tile clusters.

Key Features:
    • Streaming algorithms for memory efficiency with large tile datasets.
    • High-contrast color palettes optimized for scientific visualization.
    • Tile-based spatial overlay generation without segmentation masks.
    • Publication-quality PCA plots with enhanced readability.

Notes:
    • Each tile is treated as an independent sample for clustering.
    • Cluster assignments are mapped back to tile positions for spatial analysis.
    • Background tiles (containing no nuclei) will still be clustered and visualized.
"""

import argparse
import logging
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# Import color generation utilities.
try:
    from generate_contrast_colors import generate_color_palette, colors_to_hex_list
except ImportError:
    print("WARNING: generate_contrast_colors not found, using basic color generation")
    
    def generate_color_palette(n, alpha=255, **kwargs):
        """Basic color palette generation fallback."""
        colors = []
        for i in range(n):
            hue = (i * 360 / n) % 360
            # Convert HSV to RGB (simplified).
            c = 1.0
            x = c * (1 - abs((hue / 60) % 2 - 1))
            m = 0
            
            if 0 <= hue < 60:
                r, g, b = c, x, 0
            elif 60 <= hue < 120:
                r, g, b = x, c, 0
            elif 120 <= hue < 180:
                r, g, b = 0, c, x
            elif 180 <= hue < 240:
                r, g, b = 0, x, c
            elif 240 <= hue < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
                
            colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255), alpha))
        return colors
    
    def colors_to_hex_list(colors):
        """Convert RGBA colors to hex format."""
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b, a in colors]


def _slice_region(height: int, width: int, region: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    """
    Convert fractional region coordinates to pixel coordinates.
    
    Parameters:
        height: Image height in pixels.
        width: Image width in pixels.
        region: Fractional coordinates (xmin, xmax, ymin, ymax).
        
    Returns:
        Pixel coordinates (x0, x1, y0, y1).
    """
    xmin, xmax, ymin, ymax = region
    x0 = int(xmin * width)
    x1 = int(xmax * width)
    y0 = int(ymin * height)
    y1 = int(ymax * height)
    return x0, x1, y0, y1


def configure_logging():
    """Configure root logger for informative output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def set_global_seed(seed: int):
    """Set global random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def save_tile_overlay(
    image_path: Path,
    coords_df: pd.DataFrame,
    cluster_ids: np.ndarray,
    color_palette: Dict[int, Tuple[int, int, int, int]],
    region: Tuple[float, float, float, float],
    out_path: Path,
    patch_size: int = 64,
    down: int = 1
) -> None:
    """
    Create RGBA overlay of tile clusters on binary mask image.
    
    Parameters:
        image_path: Path to binary mask image.
        coords_df: DataFrame with tile coordinates (x_center, y_center).
        cluster_ids: Array of cluster assignments for each tile.
        color_palette: Dictionary mapping cluster index to RGBA color.
        region: Crop region as (xmin, xmax, ymin, ymax) fractions.
        out_path: Output path for overlay image.
        patch_size: Size of tiles in pixels.
        down: Downsampling factor for memory efficiency.
    """
    print(f"DEBUG: Creating tile overlay with {len(color_palette)} colors")
    print(f"DEBUG: Region: {region}, downsample: {down}")

    # Load and crop base image.
    base_full = Image.open(image_path).convert('RGBA')
    w, h = base_full.size
    x0, x1, y0, y1 = _slice_region(h, w, region)
    base_crop = base_full.crop((x0, y0, x1, y1))

    if down > 1:
        base_crop = base_crop.resize((base_crop.width // down, base_crop.height // down), Image.BILINEAR)
        patch_size = patch_size // down

    # Create overlay canvas.
    overlay = Image.new('RGBA', base_crop.size, (0, 0, 0, 0))
    
    # Draw colored rectangles for each tile.
    from PIL import ImageDraw
    draw = ImageDraw.Draw(overlay)
    
    tiles_drawn = 0
    for idx, (_, row) in enumerate(coords_df.iterrows()):
        if idx >= len(cluster_ids):
            break
            
        # Get tile center coordinates.
        tile_x = int((row['x_center'] - x0) / down)
        tile_y = int((row['y_center'] - y0) / down)
        
        # Check if tile is within crop region.
        if (tile_x < patch_size // 2 or tile_x >= base_crop.width - patch_size // 2 or
            tile_y < patch_size // 2 or tile_y >= base_crop.height - patch_size // 2):
            continue
            
        # Get cluster color.
        cluster_id = cluster_ids[idx]
        if cluster_id in color_palette:
            color = color_palette[cluster_id]
            
            # Draw tile rectangle.
            left = tile_x - patch_size // 2
            top = tile_y - patch_size // 2
            right = left + patch_size
            bottom = top + patch_size
            
            draw.rectangle([left, top, right, bottom], fill=color, outline=None)
            tiles_drawn += 1

    # Composite overlay onto base image.
    composite = Image.alpha_composite(base_crop, overlay)
    composite.save(out_path)

    logging.info(f'Tile overlay saved with {tiles_drawn} colored tiles → {out_path}')
    print(f"DEBUG: Overlay dimensions: {composite.size}, tiles drawn: {tiles_drawn}")


def choose_optimal_k(features, k_max, criterion, sample_size=5000):
    """Choose optimal K using silhouette or Davies-Bouldin criterion."""
    idx = np.random.RandomState(0).choice(features.shape[0], min(sample_size, features.shape[0]), replace=False)
    sample = features[idx]
    scores = []
    
    for k in range(2, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=0)
        labels = km.fit_predict(sample)
        
        if criterion == 'silhouette':
            score = silhouette_score(sample, labels)
        else:  # davies_bouldin
            score = davies_bouldin_score(sample, labels)
            
        scores.append((k, score))
    
    df = pd.DataFrame(scores, columns=['k', criterion])
    
    if criterion == 'silhouette':
        best = int(df.loc[df[criterion].idxmax(), 'k'])
    else:
        best = int(df.loc[df[criterion].idxmin(), 'k'])
        
    logging.info(f'Optimal K selected: {best}')
    return best, df


def load_features(path_npy: Path, path_csv: Path):
    """Load features from .npy or CSV fallback."""
    if path_npy.exists():
        return np.load(path_npy, mmap_mode='r')
    if path_csv.exists():
        return pd.read_csv(path_csv).values
    logging.error('No feature array found at .npy or .csv')
    sys.exit(1)


def scale_features(data, batch_size):
    """Stream-scaling using partial StandardScaler."""
    scaler = StandardScaler()
    n = data.shape[0]
    for i in range(0, n, batch_size):
        scaler.partial_fit(data[i:i + batch_size])
    return scaler


def cluster_streaming(data, batch_size, k, seed):
    """Stream-clustering with MiniBatchKMeans."""
    mbk = MiniBatchKMeans(n_clusters=k, random_state=seed)
    n = data.shape[0]
    for i in range(0, n, batch_size):
        mbk.partial_fit(data[i:i + batch_size])
    return mbk


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Memory-efficient clustering of uniform tile ViT embeddings.')
    
    parser.add_argument('--image', type=Path, required=True, help='Path to binary mask image.')
    parser.add_argument('--coords', type=Path, required=True, help='CSV file with tile coordinates.')
    parser.add_argument('--features_npy', type=Path, default=Path('features.npy'), help='NumPy feature array.')
    parser.add_argument('--features_csv', type=Path, default=Path('features.csv'), help='CSV feature fallback.')
    parser.add_argument('--clusters', type=int, default=10, help='Initial number of clusters.')
    parser.add_argument('--auto-k', choices=['none', 'silhouette', 'dbi'], default='none', help='Auto-K selection.')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for streaming.')
    parser.add_argument('--outdir', type=Path, default=Path('results'), help='Output directory.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--region', type=float, nargs=4, default=[0, 1, 0, 1], help='Crop region fractions.')
    parser.add_argument('--downsample', type=int, default=1, help='Downsampling factor for overlay.')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    try:
        args = parse_arguments()
        
        configure_logging()
        set_global_seed(args.seed)
        args.outdir.mkdir(parents=True, exist_ok=True)

        logging.info('Loading tile features...')
        features = load_features(args.features_npy, args.features_csv)
        print(f"DEBUG: Loaded {features.shape[0]} tiles with {features.shape[1]} features each")

        logging.info('Scaling features...')
        scaler = scale_features(features, args.batch_size)
        joblib.dump(scaler, args.outdir / 'scaler.joblib')

        logging.info('Transforming features and saving memmap...')
        n, m = features.shape
        scaled_fp = args.outdir / 'features_scaled.npy'
        scaled = np.memmap(scaled_fp, dtype='float32', mode='w+', shape=(n, m))
        
        for i in range(0, n, args.batch_size):
            scaled[i:i + args.batch_size] = scaler.transform(features[i:i + args.batch_size])

        # Determine optimal number of clusters.
        if args.auto_k != 'none':
            k, df_scores = choose_optimal_k(scaled, args.clusters, args.auto_k)
            df_scores.to_csv(args.outdir / 'cluster_selection_scores.csv', index=False)
        else:
            k = args.clusters

        # Generate color palette for clusters.
        print(f"DEBUG: Generating color palette for {k} clusters")
        color_palette = generate_color_palette(
            n=k,
            alpha=180,  # Semi-transparent for overlay.
            background="dark",
            saturation=0.95,
            contrast_ratio=10,
            hue_start=0.0
        )
        hex_colors = colors_to_hex_list(color_palette)
        
        # Convert to dictionary format.
        color_dict = {i: color_palette[i] for i in range(len(color_palette))}

        logging.info(f'Clustering into {k} clusters...')
        kmeans = cluster_streaming(scaled, args.batch_size, k, args.seed)
        joblib.dump(kmeans, args.outdir / 'kmeans_model.joblib')

        logging.info('Predicting cluster labels...')
        labels = np.zeros(n, dtype=int)
        for i in range(0, n, args.batch_size):
            labels[i:i + args.batch_size] = kmeans.predict(scaled[i:i + args.batch_size])

        # Save cluster assignments with coordinates.
        coords_df = pd.read_csv(args.coords)
        coords_df['cluster'] = labels
        coords_df.to_csv(args.outdir / 'tile_clusters.csv', index=False)

        logging.info('Creating PCA visualization...')
        sample_idx = np.random.RandomState(args.seed).choice(n, min(5000, n), replace=False)
        pcs = PCA(2).fit_transform(scaled[sample_idx])

        # Create high-resolution PCA plot.
        plt.figure(figsize=(12, 10))
        plt.style.use('dark_background')

        sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=labels[sample_idx],
                       palette=hex_colors, legend=True, s=80, alpha=0.8,
                       edgecolor='black', linewidth=0.3)

        plt.title('PCA Visualization of Tile Clusters', fontsize=20, fontweight='bold',
                  color='white', pad=25)
        plt.xlabel('First Principal Component', fontsize=16, color='white', fontweight='bold')
        plt.ylabel('Second Principal Component', fontsize=16, color='white', fontweight='bold')

        plt.xticks(fontsize=14, color='white', fontweight='bold')
        plt.yticks(fontsize=14, color='white', fontweight='bold')

        legend = plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left',
                           frameon=True, fancybox=True, shadow=True,
                           facecolor='black', edgecolor='white', fontsize=12)
        legend.get_title().set_color('white')
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_fontsize(14)

        for text in legend.get_texts():
            text.set_color('white')
            text.set_fontweight('bold')

        plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
        plt.tight_layout()
        plt.savefig(args.outdir / 'pca_clusters.png', dpi=400, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        plt.close()

        logging.info('Creating tile overlay...')
        save_tile_overlay(
            args.image, coords_df, labels, color_dict,
            region=tuple(args.region), out_path=args.outdir / 'overlay_clusters.tif',
            patch_size=64, down=args.downsample
        )

        print("✓ Uniform tile clustering completed successfully.")
        print(f"  • Processed {n} tiles")
        print(f"  • Generated {k} clusters")
        print(f"  • Results saved to: {args.outdir}")

    except Exception as e:
        print(f"ERROR: Uniform tile clustering failed: {str(e)}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
