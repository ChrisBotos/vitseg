"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: analyze_vit_improvements.py.
Description:
    Comprehensive analysis script to evaluate and compare ViT clustering improvements.
    Provides detailed performance metrics, feature quality assessment, and biological
    relevance analysis for enhanced ViT implementations.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scikit-learn, matplotlib, seaborn.
    • scipy for statistical analysis.

Usage:
    python analyze_vit_improvements.py \
        --old_features old_features.csv \
        --new_features new_features.csv \
        --coords coords.csv \
        --old_clusters old_clusters.csv \
        --new_clusters new_clusters.csv \
        --outdir analysis_results
"""
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


class ViTImprovementAnalyzer:
    """Comprehensive analyzer for ViT clustering improvements.

    This class provides detailed analysis of feature quality, clustering
    performance, and biological relevance to validate the effectiveness
    of ViT enhancements for biological image analysis.
    """
    
    def __init__(self):
        self.results = {}
        
    def analyze_feature_quality(self, old_features: np.ndarray,
                               new_features: np.ndarray) -> Dict[str, float]:
        """Analyze and compare feature quality between old and new implementations.

        This analysis evaluates feature discriminability, information content,
        and dimensionality effectiveness to validate improvements.

        Args:
            old_features (np.ndarray): Original ViT features [n_samples, n_features].
            new_features (np.ndarray): Enhanced ViT features [n_samples, n_features].

        Returns:
            Dict[str, float]: Dictionary of feature quality metrics.
        """
        print("DEBUG: Analyzing feature quality...")
        
        metrics = {}
        
        # Dimensionality comparison.
        metrics['old_dimensionality'] = old_features.shape[1]
        metrics['new_dimensionality'] = new_features.shape[1]
        metrics['dimensionality_reduction'] = (
            old_features.shape[1] - new_features.shape[1]
        ) / old_features.shape[1]
        
        # Feature variance and information content.
        old_variance = np.var(old_features, axis=0)
        new_variance = np.var(new_features, axis=0)
        
        metrics['old_mean_variance'] = np.mean(old_variance)
        metrics['new_mean_variance'] = np.mean(new_variance)
        metrics['old_variance_cv'] = np.std(old_variance) / np.mean(old_variance)
        metrics['new_variance_cv'] = np.std(new_variance) / np.mean(new_variance)
        
        # Feature correlation analysis.
        old_corr_matrix = np.corrcoef(old_features.T)
        new_corr_matrix = np.corrcoef(new_features.T)
        
        # Remove diagonal elements for correlation analysis.
        old_corr_off_diag = old_corr_matrix[np.triu_indices_from(old_corr_matrix, k=1)]
        new_corr_off_diag = new_corr_matrix[np.triu_indices_from(new_corr_matrix, k=1)]
        
        metrics['old_mean_correlation'] = np.mean(np.abs(old_corr_off_diag))
        metrics['new_mean_correlation'] = np.mean(np.abs(new_corr_off_diag))
        metrics['correlation_reduction'] = (
            metrics['old_mean_correlation'] - metrics['new_mean_correlation']
        ) / metrics['old_mean_correlation']
        
        # PCA analysis for information preservation.
        old_pca = PCA()
        new_pca = PCA()
        
        old_pca.fit(old_features)
        new_pca.fit(new_features)
        
        # Compare explained variance ratios.
        old_cumvar_50 = np.sum(old_pca.explained_variance_ratio_[:50])
        new_cumvar_50 = np.sum(new_pca.explained_variance_ratio_[:min(50, len(new_pca.explained_variance_ratio_))])
        
        metrics['old_cumvar_50'] = old_cumvar_50
        metrics['new_cumvar_50'] = new_cumvar_50
        metrics['information_preservation'] = new_cumvar_50 / old_cumvar_50
        
        # Effective dimensionality (number of components for 95% variance).
        old_eff_dim = np.argmax(np.cumsum(old_pca.explained_variance_ratio_) >= 0.95) + 1
        new_eff_dim = np.argmax(np.cumsum(new_pca.explained_variance_ratio_) >= 0.95) + 1
        
        metrics['old_effective_dim'] = old_eff_dim
        metrics['new_effective_dim'] = new_eff_dim
        metrics['effective_dim_reduction'] = (old_eff_dim - new_eff_dim) / old_eff_dim
        
        print(f"DEBUG: Feature quality analysis completed")
        print(f"DEBUG: Dimensionality: {old_features.shape[1]} → {new_features.shape[1]}")
        print(f"DEBUG: Effective dimensionality: {old_eff_dim} → {new_eff_dim}")
        print(f"DEBUG: Information preservation: {metrics['information_preservation']:.4f}")
        
        return metrics
    
    def compare_clustering_performance(self, old_features: np.ndarray, new_features: np.ndarray,
                                     old_labels: np.ndarray, new_labels: np.ndarray) -> Dict[str, float]:
        """Compare clustering performance between old and new implementations.

        This analysis evaluates clustering quality improvements using multiple
        metrics to validate the biological significance of enhancements.

        Args:
            old_features (np.ndarray): Original ViT features.
            new_features (np.ndarray): Enhanced ViT features.
            old_labels (np.ndarray): Original cluster labels.
            new_labels (np.ndarray): Enhanced cluster labels.

        Returns:
            Dict[str, float]: Dictionary of clustering performance metrics.
        """
        print("DEBUG: Comparing clustering performance...")
        
        metrics = {}
        
        # Basic clustering metrics.
        old_sil = silhouette_score(old_features, old_labels)
        new_sil = silhouette_score(new_features, new_labels)
        
        old_ch = calinski_harabasz_score(old_features, old_labels)
        new_ch = calinski_harabasz_score(new_features, new_labels)
        
        old_db = davies_bouldin_score(old_features, old_labels)
        new_db = davies_bouldin_score(new_features, new_labels)
        
        metrics['old_silhouette'] = old_sil
        metrics['new_silhouette'] = new_sil
        metrics['silhouette_improvement'] = (new_sil - old_sil) / abs(old_sil)
        
        metrics['old_calinski_harabasz'] = old_ch
        metrics['new_calinski_harabasz'] = new_ch
        metrics['ch_improvement'] = (new_ch - old_ch) / old_ch
        
        metrics['old_davies_bouldin'] = old_db
        metrics['new_davies_bouldin'] = new_db
        metrics['db_improvement'] = (old_db - new_db) / old_db  # Lower is better
        
        # Cluster stability analysis.
        metrics['old_n_clusters'] = len(np.unique(old_labels))
        metrics['new_n_clusters'] = len(np.unique(new_labels))
        
        # Cluster size distribution.
        old_sizes = np.bincount(old_labels)
        new_sizes = np.bincount(new_labels)
        
        metrics['old_cluster_balance'] = np.std(old_sizes) / np.mean(old_sizes)
        metrics['new_cluster_balance'] = np.std(new_sizes) / np.mean(new_sizes)
        metrics['balance_improvement'] = (
            metrics['old_cluster_balance'] - metrics['new_cluster_balance']
        ) / metrics['old_cluster_balance']
        
        # Agreement between old and new clustering.
        if len(old_labels) == len(new_labels):
            metrics['adjusted_rand_index'] = adjusted_rand_score(old_labels, new_labels)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(old_labels, new_labels)
        
        print(f"DEBUG: Clustering performance comparison completed")
        print(f"DEBUG: Silhouette: {old_sil:.4f} → {new_sil:.4f} ({metrics['silhouette_improvement']:+.2%})")
        print(f"DEBUG: Calinski-Harabasz: {old_ch:.2f} → {new_ch:.2f} ({metrics['ch_improvement']:+.2%})")
        print(f"DEBUG: Davies-Bouldin: {old_db:.4f} → {new_db:.4f} ({metrics['db_improvement']:+.2%})")
        
        return metrics
    
    def analyze_spatial_coherence(self, coords: np.ndarray, old_labels: np.ndarray,
                                new_labels: np.ndarray) -> Dict[str, float]:
        """Analyze spatial coherence of clustering results.

        This analysis evaluates how well clusters preserve spatial relationships,
        which is crucial for biological interpretation of tissue organization.

        Args:
            coords (np.ndarray): Spatial coordinates [n_samples, 2].
            old_labels (np.ndarray): Original cluster labels.
            new_labels (np.ndarray): Enhanced cluster labels.

        Returns:
            Dict[str, float]: Dictionary of spatial coherence metrics.
        """
        print("DEBUG: Analyzing spatial coherence...")
        
        metrics = {}
        
        def compute_spatial_coherence(labels):
            coherence_scores = []
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_coords = coords[cluster_mask]
                
                if len(cluster_coords) > 1:
                    # Compute average pairwise distance within cluster.
                    distances = pdist(cluster_coords)
                    avg_distance = np.mean(distances)
                    coherence_scores.append(1.0 / (1.0 + avg_distance))
            
            return np.mean(coherence_scores) if coherence_scores else 0.0
        
        old_coherence = compute_spatial_coherence(old_labels)
        new_coherence = compute_spatial_coherence(new_labels)
        
        metrics['old_spatial_coherence'] = old_coherence
        metrics['new_spatial_coherence'] = new_coherence
        metrics['coherence_improvement'] = (new_coherence - old_coherence) / old_coherence
        
        # Nearest neighbor analysis.
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        
        def compute_nn_purity(labels):
            purities = []
            for i, neighbors in enumerate(indices):
                neighbor_labels = labels[neighbors[1:]]  # Exclude self
                same_cluster = np.sum(neighbor_labels == labels[i])
                purity = same_cluster / len(neighbor_labels)
                purities.append(purity)
            return np.mean(purities)
        
        old_nn_purity = compute_nn_purity(old_labels)
        new_nn_purity = compute_nn_purity(new_labels)
        
        metrics['old_nn_purity'] = old_nn_purity
        metrics['new_nn_purity'] = new_nn_purity
        metrics['nn_purity_improvement'] = (new_nn_purity - old_nn_purity) / old_nn_purity
        
        print(f"DEBUG: Spatial coherence analysis completed")
        print(f"DEBUG: Spatial coherence: {old_coherence:.4f} → {new_coherence:.4f} ({metrics['coherence_improvement']:+.2%})")
        print(f"DEBUG: NN purity: {old_nn_purity:.4f} → {new_nn_purity:.4f} ({metrics['nn_purity_improvement']:+.2%})")
        
        return metrics
    
    def create_comparison_visualizations(self, old_features: np.ndarray, new_features: np.ndarray,
                                       coords: np.ndarray, old_labels: np.ndarray,
                                       new_labels: np.ndarray, output_dir: Path) -> None:
        """Create comprehensive comparison visualizations.

        This function generates publication-quality visualizations comparing
        old and new ViT implementations across multiple dimensions.

        Args:
            old_features (np.ndarray): Original ViT features.
            new_features (np.ndarray): Enhanced ViT features.
            coords (np.ndarray): Spatial coordinates.
            old_labels (np.ndarray): Original cluster labels.
            new_labels (np.ndarray): Enhanced cluster labels.
            output_dir (Path): Output directory for visualizations.
        """
        print("DEBUG: Creating comparison visualizations...")
        
        # Set publication style.
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Feature space comparison with PCA.
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # PCA of old features.
        old_pca = PCA(n_components=2)
        old_pca_features = old_pca.fit_transform(old_features)
        
        scatter1 = axes[0, 0].scatter(old_pca_features[:, 0], old_pca_features[:, 1], 
                                     c=old_labels, cmap='tab20', s=30, alpha=0.7)
        axes[0, 0].set_title('Original ViT Features (PCA)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel(f'PC1 ({old_pca.explained_variance_ratio_[0]:.2%})', fontweight='bold')
        axes[0, 0].set_ylabel(f'PC2 ({old_pca.explained_variance_ratio_[1]:.2%})', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # PCA of new features.
        new_pca = PCA(n_components=2)
        new_pca_features = new_pca.fit_transform(new_features)
        
        scatter2 = axes[0, 1].scatter(new_pca_features[:, 0], new_pca_features[:, 1], 
                                     c=new_labels, cmap='tab20', s=30, alpha=0.7)
        axes[0, 1].set_title('Enhanced ViT Features (PCA)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel(f'PC1 ({new_pca.explained_variance_ratio_[0]:.2%})', fontweight='bold')
        axes[0, 1].set_ylabel(f'PC2 ({new_pca.explained_variance_ratio_[1]:.2%})', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Spatial distribution comparison.
        scatter3 = axes[1, 0].scatter(coords[:, 0], coords[:, 1], c=old_labels, 
                                     cmap='tab20', s=20, alpha=0.8)
        axes[1, 0].set_title('Original Clustering (Spatial)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('X Coordinate', fontweight='bold')
        axes[1, 0].set_ylabel('Y Coordinate', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        scatter4 = axes[1, 1].scatter(coords[:, 0], coords[:, 1], c=new_labels, 
                                     cmap='tab20', s=20, alpha=0.8)
        axes[1, 1].set_title('Enhanced Clustering (Spatial)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('X Coordinate', fontweight='bold')
        axes[1, 1].set_ylabel('Y Coordinate', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'vit_improvement_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"DEBUG: Comparison visualizations saved to {output_dir}")
    
    def generate_improvement_report(self, all_metrics: Dict[str, Dict], output_dir: Path) -> None:
        """Generate comprehensive improvement report.

        This function creates a detailed report summarizing all improvements
        and their statistical significance for method validation.

        Args:
            all_metrics (Dict[str, Dict]): Dictionary containing all analysis metrics.
            output_dir (Path): Output directory for the report.
        """
        print("DEBUG: Generating improvement report...")
        
        # Combine all metrics into a single dataframe.
        report_data = []
        for category, metrics in all_metrics.items():
            for metric, value in metrics.items():
                report_data.append({
                    'category': category,
                    'metric': metric,
                    'value': value
                })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_dir / 'improvement_analysis_report.csv', index=False)
        
        # Create summary statistics.
        summary_stats = {}
        
        # Feature quality improvements.
        if 'feature_quality' in all_metrics:
            fq = all_metrics['feature_quality']
            summary_stats['dimensionality_reduction'] = fq.get('dimensionality_reduction', 0)
            summary_stats['information_preservation'] = fq.get('information_preservation', 1)
            summary_stats['correlation_reduction'] = fq.get('correlation_reduction', 0)
        
        # Clustering performance improvements.
        if 'clustering_performance' in all_metrics:
            cp = all_metrics['clustering_performance']
            summary_stats['silhouette_improvement'] = cp.get('silhouette_improvement', 0)
            summary_stats['ch_improvement'] = cp.get('ch_improvement', 0)
            summary_stats['db_improvement'] = cp.get('db_improvement', 0)
        
        # Spatial coherence improvements.
        if 'spatial_coherence' in all_metrics:
            sc = all_metrics['spatial_coherence']
            summary_stats['coherence_improvement'] = sc.get('coherence_improvement', 0)
            summary_stats['nn_purity_improvement'] = sc.get('nn_purity_improvement', 0)
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(output_dir / 'improvement_summary.csv', index=False)
        
        print(f"DEBUG: Improvement report saved to {output_dir}")
        
        # Print key improvements.
        print("\n" + "=" * 60)
        print("KEY IMPROVEMENTS SUMMARY")
        print("=" * 60)
        for key, value in summary_stats.items():
            if 'improvement' in key or 'reduction' in key or 'preservation' in key:
                print(f"{key}: {value:+.2%}")
        print("=" * 60)


def main():
    """Main entry point for ViT improvement analysis.

    Orchestrates comprehensive analysis comparing old and new ViT implementations
    across feature quality, clustering performance, and biological relevance.
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis of ViT clustering improvements"
    )

    # Input files.
    parser.add_argument('--old_features', type=Path, required=True,
                       help='Original ViT features CSV file')
    parser.add_argument('--new_features', type=Path, required=True,
                       help='Enhanced ViT features CSV file')
    parser.add_argument('--coords', type=Path, required=True,
                       help='Patch coordinates CSV file')
    parser.add_argument('--old_clusters', type=Path, required=True,
                       help='Original clustering results CSV file')
    parser.add_argument('--new_clusters', type=Path, required=True,
                       help='Enhanced clustering results CSV file')

    # Output directory.
    parser.add_argument('--outdir', type=Path, default=Path('improvement_analysis'),
                       help='Output directory for analysis results')

    # Analysis options.
    parser.add_argument('--standardize', action='store_true',
                       help='Standardize features before analysis')

    args = parser.parse_args()

    # Create output directory.
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ViT IMPROVEMENT ANALYSIS")
    print("=" * 80)
    print(f"Old features: {args.old_features}")
    print(f"New features: {args.new_features}")
    print(f"Output directory: {args.outdir}")
    print("=" * 80)

    try:
        # Initialize analyzer.
        analyzer = ViTImprovementAnalyzer()

        # Load data.
        print("Loading data files...")
        old_features_df = pd.read_csv(args.old_features)
        new_features_df = pd.read_csv(args.new_features)
        coords_df = pd.read_csv(args.coords)
        old_clusters_df = pd.read_csv(args.old_clusters)
        new_clusters_df = pd.read_csv(args.new_clusters)

        # Extract arrays.
        old_features = old_features_df.values
        new_features = new_features_df.values
        coords = coords_df[['x_center', 'y_center']].values
        old_labels = old_clusters_df['cluster'].values
        new_labels = new_clusters_df['cluster'].values

        print(f"DEBUG: Loaded old features: {old_features.shape}")
        print(f"DEBUG: Loaded new features: {new_features.shape}")
        print(f"DEBUG: Loaded coordinates: {coords.shape}")

        # Standardize features if requested.
        if args.standardize:
            print("Standardizing features...")
            old_scaler = StandardScaler()
            new_scaler = StandardScaler()
            old_features = old_scaler.fit_transform(old_features)
            new_features = new_scaler.fit_transform(new_features)

        # Perform comprehensive analysis.
        all_metrics = {}

        # 1. Feature quality analysis.
        print("\n1. Analyzing feature quality...")
        feature_metrics = analyzer.analyze_feature_quality(old_features, new_features)
        all_metrics['feature_quality'] = feature_metrics

        # 2. Clustering performance comparison.
        print("\n2. Comparing clustering performance...")
        clustering_metrics = analyzer.compare_clustering_performance(
            old_features, new_features, old_labels, new_labels
        )
        all_metrics['clustering_performance'] = clustering_metrics

        # 3. Spatial coherence analysis.
        print("\n3. Analyzing spatial coherence...")
        spatial_metrics = analyzer.analyze_spatial_coherence(coords, old_labels, new_labels)
        all_metrics['spatial_coherence'] = spatial_metrics

        # 4. Create visualizations.
        print("\n4. Creating comparison visualizations...")
        analyzer.create_comparison_visualizations(
            old_features, new_features, coords, old_labels, new_labels, args.outdir
        )

        # 5. Generate comprehensive report.
        print("\n5. Generating improvement report...")
        analyzer.generate_improvement_report(all_metrics, args.outdir)

        print("\n" + "=" * 80)
        print("✅ ViT IMPROVEMENT ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Analysis results saved to: {args.outdir}")

        # Print key findings.
        if 'clustering_performance' in all_metrics:
            cp = all_metrics['clustering_performance']
            print(f"\nKey Performance Improvements:")
            print(f"• Silhouette Score: {cp['silhouette_improvement']:+.2%}")
            print(f"• Calinski-Harabasz Score: {cp['ch_improvement']:+.2%}")
            print(f"• Davies-Bouldin Score: {cp['db_improvement']:+.2%}")

        if 'feature_quality' in all_metrics:
            fq = all_metrics['feature_quality']
            print(f"\nFeature Quality Improvements:")
            print(f"• Dimensionality Reduction: {fq['dimensionality_reduction']:+.2%}")
            print(f"• Information Preservation: {fq['information_preservation']:.4f}")
            print(f"• Correlation Reduction: {fq['correlation_reduction']:+.2%}")

    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
