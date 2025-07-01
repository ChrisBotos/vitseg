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
def save_overlay_from_masks(image_path, overlay_map, palette, alpha, region, out_path):
    base = Image.open(image_path).convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0,0,0,0))
    data = np.array(overlay_map)
    for cid, color in enumerate(palette, start=1):
        mask = (data == cid)
        color_img = Image.new('RGBA', base.size, color + (0,))
        mask_img = Image.fromarray((mask * int(255*alpha)).astype(np.uint8))
        color_img.putalpha(mask_img)
        overlay = Image.alpha_composite(overlay, color_img)
    composite = Image.alpha_composite(base, overlay)
    if region:
        xmin, xmax, ymin, ymax = region
        w, h = base.size
        crop = (int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h))
        composite = composite.crop(crop)
    composite.save(out_path)
    logging.info(f'Overlay saved → {out_path}')

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

    logging.info('Creating overlay map...')
    passed = np.load(args.labels)
    seg_map = np.load(args.label_map, mmap_mode='r')
    overlay_map = compute_label_cluster_map(seg_map, passed, coords_df, labels)
    palette = generate_color_palette(k)
    save_overlay_from_masks(args.image, overlay_map, palette, alpha=0.35, region=args.region,
                             out_path=args.outdir/'overlay_clusters.tif')

    logging.info('Plotting PCA scatter...')
    sample_idx = np.random.RandomState(args.seed).choice(n, min(5000, n), replace=False)
    pcs = PCA(2).fit_transform(scaled[sample_idx])
    plt.figure()
    sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=labels[sample_idx], palette=palette, legend=False)
    plt.title('PCA of patch clusters')
    plt.savefig(args.outdir/'pca_clusters.png')
    logging.info('PCA plot saved.')

if __name__=='__main__':
    main()
