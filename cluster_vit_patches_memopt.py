"""
Author: Christos Botos.
Affiliation: Institute of Molecular Biology and Biotechnology
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: cluster_vit_patches_memopt.py
Description:
    Memory-optimized clustering of ViT patch embeddings with MiniBatchKMeans and streaming scaling,
    preserving full overlay, palette generation, PCA plotting, and CSV fallback.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scikit-learn, joblib, matplotlib, PIL, seaborn, scipy.

Usage:
    python cluster_vit_patches_memopt.py --image img.tif \
        --labels filtered_passed_labels.npy --label_map segmentation_masks.npy \
        --coords coords.csv --features_npy features.npy --features_csv features.csv \
        --clusters 10 --auto-k silhouette --batch-size 10000 \
        --outdir results --seed 0 --region 0.1 0.9 0.1 0.9

Positional Arguments:
    None.

Optional Arguments:
    --image            Path to raw microscopy image (RGB).
    --labels           Numpy .npy list of passed labels.
    --label_map        Original segmentation map .npy.
    --coords           CSV of patch centroids.
    --features_npy     Numpy .npy feature array for streaming.
    --features_csv     CSV of features (fallback if no .npy).
    --clusters         Initial K for MiniBatchKMeans.
    --auto-k           Criterion for auto-K selection ('none','silhouette','dbi').
    --batch-size       Batch size for streaming.
    --outdir           Output directory.
    --seed             Random seed for reproducibility.
    --region           Fractional crop region for overlay: xmin xmax ymin ymax.

Inputs:
    • features.npy     Memory-mapped feature array (n_samples, n_features).
    • features.csv     CSV features fallback.
    • coords.csv       Coordinates CSV.
    • segmentation_masks.npy Full segmentation map.

Outputs:
    • patch_clusters.csv       Cluster assignments per patch.
    • kmeans_model.joblib      Trained MiniBatchKMeans model.
    • scaler.joblib            StandardScaler fitted.
    • overlay_clusters.tif     Overlay of clusters on image.
    • pca_clusters.png         PCA scatter plot of clusters.
    • cluster_selection_scores.csv K vs score.

"""
import argparse
import logging
import sys
from pathlib import Path
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

''' Configure root logger '''
def configure_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

''' Set global random seeds for reproducibility '''
def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

''' Generate distinct color palette for clusters '''
def generate_color_palette(n_clusters):
    palette = sns.color_palette('hsv', n_clusters)
    return [(int(r*255), int(g*255), int(b*255)) for r, g, b in palette]

''' Compute mapping from cluster IDs back onto full segmentation mask '''
def compute_label_cluster_map(label_map, passed_labels, coords_df, cluster_ids):
    mapped = np.zeros_like(label_map, dtype=np.int32)
    for lab, cid in zip(passed_labels, cluster_ids):
        mask = (label_map == lab)
        mapped[mask] = cid + 1  # 0 reserved
    return mapped

''' Save RGBA overlay of clusters on the image '''
def save_overlay(img_path: Path, seg_map_path: Path, passed_labels: np.ndarray, cluster_ids: np.ndarray,
                palette, region, out_path: Path, alpha: float = 0.35, down: int = 1) -> None:
    """Write an RGBA overlay without materialising full‑frame masks.

    The pipeline is:
        1. Memory‑map the segmentation mask.
        2. Crop *before* any heavy processing.
        3. Vector‑map label → cluster via LUT.
        4. Convert to RGBA using a pre‑baked NumPy palette.
        5. Alpha‑composite onto the (cropped, optionally down‑sampled) source image.
    """
    # Load base image once
    base_full = Image.open(img_path).convert('RGBA')
    w, h = base_full.size
    x0, x1, y0, y1 = _slice_region(h, w, region)

    # Crop first to minimise RAM.
    base_crop = base_full.crop((x0, y0, x1, y1))
    seg = np.load(seg_map_path, mmap_mode='r')[y0:y1, x0:x1]

    # Optional overlay down‑sampling.
    if down > 1:
        seg = seg[::down, ::down]
        base_crop = base_crop.resize((seg.shape[1], seg.shape[0]), Image.BILINEAR)

    # Label → cluster map via LUT (fully vectorised).
    lut = _build_lut(int(seg.max()), passed_labels, cluster_ids)
    cluster_map = lut[seg]

    # Build an (k+1, 4) RGBA palette. Row 0 is transparent background.
    k = len(palette)
    rgba = np.zeros((k + 1, 4), dtype=np.uint8)
    for idx, (r, g, b) in enumerate(palette, start=1):
        rgba[idx, :3] = (r, g, b)
        rgba[idx, 3] = int(alpha * 255)

    overlay = Image.fromarray(rgba[cluster_map], mode='RGBA')
    composite = Image.alpha_composite(base_crop, overlay)
    composite.save(out_path)
    logging.info('Overlay saved → %s', out_path)

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

    args = parser.parse_args()

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

    palette = generate_color_palette(k)

    logging.info('Plotting PCA scatter...')
    sample_idx = np.random.RandomState(args.seed).choice(n, min(5000, n), replace=False)
    pcs = PCA(2).fit_transform(scaled[sample_idx])
    plt.figure()
    sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=labels[sample_idx], palette=palette, legend=False)
    plt.title('PCA of patch clusters')
    plt.savefig(args.outdir/'pca_clusters.png')
    logging.info('PCA plot saved.')

    logging.info('Creating overlay…')
    passed = np.load(args.labels)
    palette = generate_color_palette(k)
    save_overlay(args.image, args.label_map, passed, labels, palette,
                region=args.region, out_path=args.outdir / 'overlay_clusters.tif',
                alpha=0.35, down=args.downsample)

if __name__=='__main__':
    main()
