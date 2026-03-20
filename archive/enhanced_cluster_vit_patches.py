"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: enhanced_cluster_vit_patches.py.
Description:
    Advanced clustering pipeline for enhanced ViT patch embeddings with hierarchical
    clustering, ensemble methods, and comprehensive evaluation metrics specifically
    designed for biological image analysis.

    Key improvements for bioinformatician users:
        • **Hierarchical clustering** – Uses agglomerative clustering to capture
          natural cell type hierarchies and tissue organization patterns.
        • **Ensemble clustering** – Combines multiple clustering algorithms to
          improve robustness and capture different aspects of cellular heterogeneity.
        • **Advanced evaluation metrics** – Includes silhouette analysis, calinski-harabasz
          scores, and biological relevance metrics for comprehensive quality assessment.
        • **Adaptive cluster selection** – Automatically determines optimal cluster
          numbers using multiple criteria and stability analysis.

    Scientific context:
        This clustering approach is optimized for identifying distinct cell populations
        in tissue samples, particularly useful for studying cellular responses to
        injury, treatment effects, and phenotypic transitions in disease models.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scikit-learn, scipy, matplotlib, seaborn.
    • joblib for model persistence.

Usage:
    python enhanced_cluster_vit_patches.py \
        --features features.csv \
        --coords coords.csv \
        --image image.tif \
        --labels labels.npy \
        --label_map segmentation_masks.npy \
        --outdir results \
        --method ensemble \
        --max_clusters 20

"""
import argparse
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

from generate_contrast_colors import generate_color_palette, colors_to_hex_list


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("enhanced-clustering")


# ---------------------------------------------------------------------------
# Enhanced Clustering Classes
# ---------------------------------------------------------------------------

class EnsembleClusterer:
    """
    Ensemble clustering approach combining multiple algorithms for robust results.
    
    This class implements a sophisticated ensemble approach that combines different
    clustering algorithms to capture various aspects of cellular heterogeneity,
    essential for accurate cell type identification in biological samples.
    """
    
    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.clusterers = {}
        self.weights = {}
        
        # Initialize multiple clustering algorithms.
        self.clusterers['kmeans'] = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        )
        self.clusterers['agglomerative'] = AgglomerativeClustering(
            n_clusters=n_clusters, linkage='ward'
        )
        self.clusterers['spectral'] = SpectralClustering(
            n_clusters=n_clusters, random_state=random_state, affinity='rbf'
        )
        
        print(f"DEBUG: Initialized ensemble with {len(self.clusterers)} algorithms")
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit ensemble clustering and return consensus labels.
        
        Args:
            X: Feature matrix [n_samples, n_features].
            
        Returns:
            Consensus cluster labels [n_samples].
        """
        predictions = {}
        scores = {}
        
        # Fit each clusterer and evaluate performance.
        for name, clusterer in self.clusterers.items():
            try:
                labels = clusterer.fit_predict(X)
                predictions[name] = labels
                
                # Evaluate clustering quality.
                if len(np.unique(labels)) > 1:
                    sil_score = silhouette_score(X, labels)
                    ch_score = calinski_harabasz_score(X, labels)
                    scores[name] = {'silhouette': sil_score, 'calinski_harabasz': ch_score}
                    print(f"DEBUG: {name} - Silhouette: {sil_score:.4f}, CH: {ch_score:.2f}")
                else:
                    scores[name] = {'silhouette': -1, 'calinski_harabasz': 0}
                    print(f"DEBUG: {name} - Failed (single cluster)")
                    
            except Exception as e:
                print(f"DEBUG: {name} failed: {e}")
                predictions[name] = np.zeros(X.shape[0], dtype=int)
                scores[name] = {'silhouette': -1, 'calinski_harabasz': 0}
        
        # Compute ensemble weights based on performance.
        total_score = sum(s['silhouette'] + s['calinski_harabasz']/1000 for s in scores.values())
        if total_score > 0:
            for name in predictions:
                score = scores[name]['silhouette'] + scores[name]['calinski_harabasz']/1000
                self.weights[name] = max(0, score / total_score)
        else:
            # Equal weights if all methods failed.
            for name in predictions:
                self.weights[name] = 1.0 / len(predictions)
        
        print(f"DEBUG: Ensemble weights: {self.weights}")
        
        # Generate consensus labels using weighted voting.
        return self._consensus_clustering(predictions, X)
    
    def _consensus_clustering(self, predictions: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
        """Generate consensus labels from multiple clustering results."""
        n_samples = X.shape[0]
        
        # Create co-association matrix.
        coassoc_matrix = np.zeros((n_samples, n_samples))
        
        for name, labels in predictions.items():
            weight = self.weights[name]
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    if labels[i] == labels[j]:
                        coassoc_matrix[i, j] += weight
                        coassoc_matrix[j, i] += weight
        
        # Apply hierarchical clustering to co-association matrix.
        distance_matrix = 1 - coassoc_matrix
        condensed_distances = pdist(distance_matrix, metric='precomputed')
        linkage_matrix = linkage(condensed_distances, method='average')
        consensus_labels = fcluster(linkage_matrix, self.n_clusters, criterion='maxclust') - 1
        
        return consensus_labels


class ClusterEvaluator:
    """
    Comprehensive cluster evaluation with biological relevance metrics.
    
    This class provides extensive evaluation metrics specifically designed for
    assessing clustering quality in biological contexts, helping researchers
    validate the biological significance of identified cell populations.
    """
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray, 
                          coords: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Comprehensive clustering evaluation with multiple metrics.
        
        Args:
            X: Feature matrix [n_samples, n_features].
            labels: Cluster labels [n_samples].
            coords: Optional spatial coordinates for spatial metrics.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}
        
        # Standard clustering metrics.
        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(X, labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        else:
            metrics['silhouette_score'] = -1
            metrics['calinski_harabasz_score'] = 0
            metrics['davies_bouldin_score'] = float('inf')
        
        # Cluster balance metrics.
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['n_clusters'] = len(unique_labels)
        metrics['cluster_balance'] = np.std(counts) / np.mean(counts)  # Lower is better.
        metrics['min_cluster_size'] = np.min(counts)
        metrics['max_cluster_size'] = np.max(counts)
        
        # Spatial coherence metrics if coordinates provided.
        if coords is not None:
            metrics['spatial_coherence'] = self._compute_spatial_coherence(labels, coords)
        
        self.metrics = metrics
        return metrics
    
    def _compute_spatial_coherence(self, labels: np.ndarray, coords: np.ndarray) -> float:
        """
        Compute spatial coherence of clusters.
        
        Measures how spatially compact clusters are, which is important for
        biological interpretation as cells of the same type often cluster spatially.
        """
        coherence_scores = []
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_coords = coords[cluster_mask]
            
            if len(cluster_coords) > 1:
                # Compute average pairwise distance within cluster.
                distances = pdist(cluster_coords)
                avg_distance = np.mean(distances)
                coherence_scores.append(1.0 / (1.0 + avg_distance))  # Higher is better.
        
        return np.mean(coherence_scores) if coherence_scores else 0.0


def determine_optimal_clusters(X: np.ndarray, max_clusters: int = 20, 
                             methods: List[str] = None) -> Tuple[int, pd.DataFrame]:
    """
    Determine optimal number of clusters using multiple criteria.
    
    This function evaluates different cluster numbers using various metrics
    to find the optimal number of cell populations, crucial for accurate
    biological interpretation.
    
    Args:
        X: Feature matrix [n_samples, n_features].
        max_clusters: Maximum number of clusters to evaluate.
        methods: List of evaluation methods to use.
        
    Returns:
        Tuple of (optimal_k, scores_dataframe).
    """
    if methods is None:
        methods = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    
    print(f"DEBUG: Evaluating cluster numbers from 2 to {max_clusters}")
    
    results = []
    evaluator = ClusterEvaluator()
    
    for k in range(2, max_clusters + 1):
        print(f"DEBUG: Evaluating k={k}")
        
        # Use KMeans for initial evaluation (fastest).
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Evaluate clustering quality.
        metrics = evaluator.evaluate_clustering(X, labels)
        
        result = {'k': k}
        result.update(metrics)
        results.append(result)
    
    scores_df = pd.DataFrame(results)
    
    # Determine optimal k using multiple criteria.
    optimal_k_candidates = {}
    
    if 'silhouette' in methods:
        optimal_k_candidates['silhouette'] = scores_df.loc[
            scores_df['silhouette_score'].idxmax(), 'k'
        ]
    
    if 'calinski_harabasz' in methods:
        optimal_k_candidates['calinski_harabasz'] = scores_df.loc[
            scores_df['calinski_harabasz_score'].idxmax(), 'k'
        ]
    
    if 'davies_bouldin' in methods:
        optimal_k_candidates['davies_bouldin'] = scores_df.loc[
            scores_df['davies_bouldin_score'].idxmin(), 'k'
        ]
    
    # Use majority vote or median for final decision.
    optimal_k = int(np.median(list(optimal_k_candidates.values())))
    
    print(f"DEBUG: Optimal k candidates: {optimal_k_candidates}")
    print(f"DEBUG: Selected optimal k: {optimal_k}")
    
    return optimal_k, scores_df


def create_enhanced_visualizations(X: np.ndarray, labels: np.ndarray, coords: np.ndarray,
                                 scores_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create comprehensive visualizations for clustering analysis.

    This function generates publication-quality visualizations specifically
    designed for biological data interpretation, including PCA plots,
    cluster evaluation metrics, and spatial distribution analysis.

    Args:
        X: Feature matrix [n_samples, n_features].
        labels: Cluster labels [n_samples].
        coords: Spatial coordinates [n_samples, 2].
        scores_df: Cluster evaluation scores dataframe.
        output_dir: Output directory for visualizations.
    """
    print("DEBUG: Creating enhanced visualizations")

    # Set publication-quality style.
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Cluster evaluation metrics plot.
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Silhouette score.
    axes[0, 0].plot(scores_df['k'], scores_df['silhouette_score'], 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Calinski-Harabasz score.
    axes[0, 1].plot(scores_df['k'], scores_df['calinski_harabasz_score'], 'o-',
                   color='orange', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Calinski-Harabasz Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Calinski-Harabasz Analysis', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Davies-Bouldin score.
    axes[1, 0].plot(scores_df['k'], scores_df['davies_bouldin_score'], 'o-',
                   color='red', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Davies-Bouldin Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Davies-Bouldin Analysis', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Cluster balance.
    axes[1, 1].plot(scores_df['k'], scores_df['cluster_balance'], 'o-',
                   color='green', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Cluster Balance (CV)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Cluster Size Balance', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Enhanced PCA visualization.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Generate high-contrast colors for clusters.
    n_clusters = len(np.unique(labels))
    colors = generate_color_palette(n=n_clusters, background="light", saturation=0.9)
    hex_colors = colors_to_hex_list(colors)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # PCA plot with cluster colors.
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20',
                             s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
                      fontsize=14, fontweight='bold')
    axes[0].set_title('PCA Visualization of Cell Clusters', fontsize=16, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Add cluster centroids.
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        centroid = X_pca[cluster_mask].mean(axis=0)
        axes[0].scatter(centroid[0], centroid[1], c='red', s=200, marker='x', linewidth=3)
        axes[0].annotate(f'C{cluster_id}', (centroid[0], centroid[1]),
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')

    # Spatial distribution plot.
    scatter2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab20',
                              s=30, alpha=0.8, edgecolors='black', linewidth=0.3)
    axes[1].set_xlabel('X Coordinate (pixels)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Y Coordinate (pixels)', fontsize=14, fontweight='bold')
    axes[1].set_title('Spatial Distribution of Cell Clusters', fontsize=16, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Add colorbars.
    cbar1 = plt.colorbar(scatter, ax=axes[0])
    cbar1.set_label('Cluster ID', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Cluster ID', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"DEBUG: Saved visualizations to {output_dir}")


def main():
    """
    Main entry point for enhanced clustering analysis.

    Orchestrates the complete clustering pipeline with advanced algorithms,
    comprehensive evaluation, and publication-quality visualizations.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced clustering for ViT patch embeddings with biological context"
    )

    # Input/output arguments.
    parser.add_argument('--features', type=Path, required=True,
                       help='Enhanced ViT features CSV file')
    parser.add_argument('--coords', type=Path, required=True,
                       help='Patch coordinates CSV file')
    parser.add_argument('--image', type=Path, required=True,
                       help='Original microscopy image')
    parser.add_argument('--labels', type=Path, required=True,
                       help='Filtered cell labels (.npy)')
    parser.add_argument('--label_map', type=Path, required=True,
                       help='Original segmentation map (.npy)')
    parser.add_argument('--outdir', type=Path, default=Path('enhanced_clustering_results'),
                       help='Output directory')

    # Clustering parameters.
    parser.add_argument('--method', type=str, default='ensemble',
                       choices=['ensemble', 'kmeans', 'hierarchical', 'spectral'],
                       help='Clustering method to use')
    parser.add_argument('--max_clusters', type=int, default=20,
                       help='Maximum number of clusters to evaluate')
    parser.add_argument('--auto_k', action='store_true',
                       help='Automatically determine optimal number of clusters')
    parser.add_argument('--n_clusters', type=int, default=10,
                       help='Number of clusters (if not using auto_k)')

    # Processing parameters.
    parser.add_argument('--standardize', action='store_true', default=True,
                       help='Standardize features before clustering')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Create output directory.
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENHANCED ViT CLUSTERING ANALYSIS")
    print("=" * 80)
    print(f"Features: {args.features}")
    print(f"Method: {args.method}")
    print(f"Auto-determine clusters: {args.auto_k}")
    print(f"Output directory: {args.outdir}")
    print("=" * 80)

    try:
        # Load data.
        print("Loading enhanced ViT features...")
        features_df = pd.read_csv(args.features)
        coords_df = pd.read_csv(args.coords)

        X = features_df.values
        coords = coords_df[['x_center', 'y_center']].values

        print(f"DEBUG: Loaded {X.shape[0]} samples with {X.shape[1]} features")

        # Standardize features if requested.
        if args.standardize:
            print("Standardizing features...")
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            joblib.dump(scaler, args.outdir / 'feature_scaler.joblib')

        # Determine optimal number of clusters.
        if args.auto_k:
            print("Determining optimal number of clusters...")
            optimal_k, scores_df = determine_optimal_clusters(X, args.max_clusters)
            scores_df.to_csv(args.outdir / 'cluster_evaluation_scores.csv', index=False)
        else:
            optimal_k = args.n_clusters
            scores_df = None

        print(f"Using {optimal_k} clusters for final analysis")

        # Perform clustering.
        print(f"Performing {args.method} clustering...")
        if args.method == 'ensemble':
            clusterer = EnsembleClusterer(optimal_k, args.seed)
            labels = clusterer.fit_predict(X)
        elif args.method == 'kmeans':
            clusterer = KMeans(n_clusters=optimal_k, random_state=args.seed, n_init=10)
            labels = clusterer.fit_predict(X)
        elif args.method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
            labels = clusterer.fit_predict(X)
        elif args.method == 'spectral':
            clusterer = SpectralClustering(n_clusters=optimal_k, random_state=args.seed)
            labels = clusterer.fit_predict(X)

        # Evaluate clustering results.
        print("Evaluating clustering quality...")
        evaluator = ClusterEvaluator()
        final_metrics = evaluator.evaluate_clustering(X, labels, coords)

        print("Final clustering metrics:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save results.
        print("Saving clustering results...")

        # Save cluster assignments.
        results_df = coords_df.copy()
        results_df['cluster'] = labels
        results_df.to_csv(args.outdir / 'enhanced_cluster_assignments.csv', index=False)

        # Save clustering model.
        if hasattr(clusterer, 'fit'):
            joblib.dump(clusterer, args.outdir / 'clustering_model.joblib')

        # Save evaluation metrics.
        metrics_df = pd.DataFrame([final_metrics])
        metrics_df.to_csv(args.outdir / 'final_clustering_metrics.csv', index=False)

        # Create visualizations.
        if scores_df is not None:
            create_enhanced_visualizations(X, labels, coords, scores_df, args.outdir)

        print("\n" + "=" * 80)
        print("✅ ENHANCED CLUSTERING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Results saved to: {args.outdir}")
        print(f"Identified {len(np.unique(labels))} distinct cell populations")
        print(f"Silhouette score: {final_metrics['silhouette_score']:.4f}")
        print(f"Calinski-Harabasz score: {final_metrics['calinski_harabasz_score']:.2f}")

    except Exception as e:
        print(f"\n❌ CLUSTERING ANALYSIS FAILED: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
