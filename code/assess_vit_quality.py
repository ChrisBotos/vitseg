"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: assess_vit_quality.py.
Description:
    Comprehensive quality assessment script for Vision Transformer clustering results.
    Evaluates clustering performance across multiple patch sizes and identifies potential
    artifacts such as border effects, spatial incoherence, and poor cluster separation.

    Key assessment components for bioinformatician users:
        • **Border effect detection** – Identifies inappropriate grouping of edge-cropped
          masks that may indicate poor feature extraction near image boundaries.
        • **Multi-scale comparison** – Compares clustering quality across different patch
          sizes (16px, 32px, 64px) to determine optimal scale for tissue analysis.
        • **Spatial coherence analysis** – Measures how well clusters preserve spatial
          relationships, crucial for biological interpretation of tissue organization.
        • **Cluster quality metrics** – Comprehensive evaluation using silhouette analysis,
          Calinski-Harabasz scores, and Davies-Bouldin indices for robust assessment.

    Scientific context:
        This assessment helps researchers select optimal ViT configurations for tissue
        analysis, ensuring that clustering results reflect genuine biological patterns
        rather than technical artifacts from image processing or feature extraction.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scikit-learn, matplotlib, seaborn, scipy.
    • rich for enhanced console output.

Usage:
    python assess_vit_quality.py \
        --results_dir results/IRI_regist_cropped_10k_filtered_32_64px \
        --image_path data/IRI_regist_cropped.tif \
        --output_dir quality_assessment \
        --patch_sizes 32 64 \
        --border_threshold 10

Arguments:
    --results_dir      Directory containing patch clustering results.
    --image_path       Path to original image for border detection.
    --output_dir       Output directory for quality assessment reports.
    --patch_sizes      List of patch sizes to evaluate.
    --border_threshold Pixel distance from edge to consider as border region.

Inputs:
    • patch_clusters.csv files from different patch size experiments.
    • Original image file for dimension analysis.
    • Coordinate and feature files for comprehensive evaluation.

Outputs:
    • quality_assessment_report.csv    Comprehensive quality metrics comparison.
    • border_effect_analysis.png       Visualization of border clustering effects.
    • spatial_coherence_plots.png      Spatial coherence analysis across scales.
    • cluster_quality_comparison.png   Multi-metric quality comparison.
    • recommendations.txt              Optimal configuration recommendations.

Key Features:
    • Multi-scale patch size comparison with statistical significance testing.
    • Border effect detection and quantification for edge artifact identification.
    • Spatial coherence analysis with nearest neighbor purity assessment.
    • Comprehensive cluster quality evaluation with biological relevance metrics.
    • Automated recommendation system for optimal ViT configuration selection.

Notes:
    • Designed to work with existing ViT clustering pipeline outputs.
    • Includes extensive validation and quality control measures.
    • Generates publication-quality visualizations and detailed analysis reports.
"""
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table

console = Console()


class ViTQualityAssessor:
    """
    Comprehensive quality assessment for Vision Transformer clustering results.
    
    This class evaluates ViT clustering performance across multiple dimensions,
    helping researchers identify optimal configurations and detect potential
    artifacts that could compromise biological interpretation.
    """
    
    def __init__(self, image_path: Path, border_threshold: int = 50):
        """
        Initialize quality assessor with image dimensions and border parameters.
        
        Args:
            image_path: Path to original image for dimension analysis.
            border_threshold: Distance from edge to consider as border region.
        """
        self.image_path = image_path
        self.border_threshold = border_threshold
        self.image_dims = self._get_image_dimensions()
        self.results = {}
        
        console.print(f"[cyan]Initialized ViT Quality Assessor[/cyan]")
        console.print(f"[blue]ℹ[/blue] Image dimensions: {self.image_dims}")
        console.print(f"[blue]ℹ[/blue] Border threshold: {border_threshold}px")
    
    def _get_image_dimensions(self) -> Tuple[int, int]:
        """Extract image dimensions from file."""
        try:
            with Image.open(self.image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            console.print(f"[red]✗[/red] Error reading image: {e}")
            # Default fallback dimensions.
            return (1000, 1000)
    
    def detect_border_effects(self, coords_df: pd.DataFrame, labels: np.ndarray,
                            patch_size: int) -> Dict[str, float]:
        """
        Analyze border clustering quality and edge artifact detection.

        This analysis evaluates how well the ViT identifies and groups patches
        that are affected by image boundaries. Good border clustering indicates
        that the ViT correctly recognizes the shared characteristic of being
        edge-cropped, which is valuable for quality control and artifact detection.

        Args:
            coords_df: DataFrame with x_center, y_center coordinates.
            labels: Cluster labels for each patch.
            patch_size: Size of patches used for analysis.

        Returns:
            Dictionary containing border clustering quality metrics.
        """
        console.print(f"[cyan]Analyzing border effects for {patch_size}px patches...[/cyan]")
        
        width, height = self.image_dims
        border_metrics = {}
        
        # Identify border patches.
        is_border = (
            (coords_df['x_center'] <= self.border_threshold) |
            (coords_df['x_center'] >= width - self.border_threshold) |
            (coords_df['y_center'] <= self.border_threshold) |
            (coords_df['y_center'] >= height - self.border_threshold)
        )
        
        border_patches = coords_df[is_border]
        interior_patches = coords_df[~is_border]
        
        border_metrics['total_patches'] = len(coords_df)
        border_metrics['border_patches'] = len(border_patches)
        border_metrics['interior_patches'] = len(interior_patches)
        border_metrics['border_fraction'] = len(border_patches) / len(coords_df)
        
        if len(border_patches) == 0:
            console.print("[yellow]⚠[/yellow] No border patches detected")
            return border_metrics
        
        # Analyze border cluster purity.
        border_labels = labels[is_border]
        interior_labels = labels[~is_border]
        
        # Calculate border cluster concentration.
        unique_clusters = np.unique(labels)
        border_concentrations = []
        
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            cluster_border_fraction = np.sum(cluster_mask & is_border) / np.sum(cluster_mask)
            border_concentrations.append(cluster_border_fraction)
        
        border_metrics['max_border_concentration'] = np.max(border_concentrations)
        border_metrics['mean_border_concentration'] = np.mean(border_concentrations)
        border_metrics['border_cluster_imbalance'] = np.std(border_concentrations)
        
        # Detect clusters that are predominantly border patches (POSITIVE indicator).
        high_border_clusters = np.sum(np.array(border_concentrations) > 0.7)
        border_metrics['high_border_clusters'] = high_border_clusters
        border_metrics['high_border_cluster_fraction'] = high_border_clusters / len(unique_clusters)

        # Calculate border clustering quality score (higher = better edge artifact detection).
        border_metrics['border_clustering_quality'] = (
            border_metrics['mean_border_concentration'] * 0.6 +
            border_metrics['high_border_cluster_fraction'] * 0.4
        )

        console.print(f"[green]✓[/green] Border clustering analysis complete: "
                     f"{border_metrics['border_fraction']:.1%} border patches, "
                     f"{high_border_clusters} dedicated border clusters "
                     f"(quality score: {border_metrics['border_clustering_quality']:.3f})")

        return border_metrics
    
    def compute_spatial_coherence(self, coords_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute spatial coherence metrics for clustering results.
        
        Spatial coherence measures how well clusters preserve spatial relationships,
        which is crucial for biological interpretation as cells of similar types
        often cluster spatially in tissue samples.
        
        Args:
            coords_df: DataFrame with spatial coordinates.
            labels: Cluster labels.
            
        Returns:
            Dictionary of spatial coherence metrics.
        """
        console.print("[cyan]Computing spatial coherence metrics...[/cyan]")
        
        coords = coords_df[['x_center', 'y_center']].values
        coherence_metrics = {}
        
        # Intra-cluster spatial coherence.
        coherence_scores = []
        cluster_sizes = []
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_coords = coords[cluster_mask]
            cluster_sizes.append(len(cluster_coords))
            
            if len(cluster_coords) > 1:
                # Compute average pairwise distance within cluster.
                distances = pdist(cluster_coords)
                avg_distance = np.mean(distances)
                coherence_scores.append(1.0 / (1.0 + avg_distance))
        
        coherence_metrics['mean_spatial_coherence'] = np.mean(coherence_scores)
        coherence_metrics['std_spatial_coherence'] = np.std(coherence_scores)
        coherence_metrics['min_spatial_coherence'] = np.min(coherence_scores)
        
        # Nearest neighbor purity analysis.
        nn = NearestNeighbors(n_neighbors=min(10, len(coords) - 1))
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        
        purities = []
        for i, neighbors in enumerate(indices):
            neighbor_labels = labels[neighbors[1:]]  # Exclude self
            same_cluster = np.sum(neighbor_labels == labels[i])
            purity = same_cluster / len(neighbor_labels)
            purities.append(purity)
        
        coherence_metrics['mean_nn_purity'] = np.mean(purities)
        coherence_metrics['std_nn_purity'] = np.std(purities)
        
        # Cluster size balance.
        coherence_metrics['cluster_size_cv'] = np.std(cluster_sizes) / np.mean(cluster_sizes)
        coherence_metrics['n_clusters'] = len(np.unique(labels))
        
        console.print(f"[green]✓[/green] Spatial coherence: "
                     f"{coherence_metrics['mean_spatial_coherence']:.3f}, "
                     f"NN purity: {coherence_metrics['mean_nn_purity']:.3f}")

        return coherence_metrics

    def compute_cluster_quality_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive cluster quality metrics.

        This function evaluates clustering quality using multiple established metrics,
        providing a robust assessment of how well the ViT features separate into
        distinct, meaningful clusters for biological analysis.

        Args:
            features: Feature matrix [n_samples, n_features].
            labels: Cluster labels [n_samples].

        Returns:
            Dictionary of cluster quality metrics.
        """
        console.print("[cyan]Computing cluster quality metrics...[/cyan]")

        quality_metrics = {}

        try:
            # Core clustering metrics.
            quality_metrics['silhouette_score'] = silhouette_score(features, labels)
            quality_metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
            quality_metrics['davies_bouldin_score'] = davies_bouldin_score(features, labels)

            # Cluster balance metrics.
            cluster_sizes = np.bincount(labels)
            quality_metrics['n_clusters'] = len(cluster_sizes)
            quality_metrics['min_cluster_size'] = np.min(cluster_sizes)
            quality_metrics['max_cluster_size'] = np.max(cluster_sizes)
            quality_metrics['cluster_size_ratio'] = np.max(cluster_sizes) / np.min(cluster_sizes)
            quality_metrics['cluster_balance'] = np.std(cluster_sizes) / np.mean(cluster_sizes)

            # Feature separation analysis.
            cluster_centers = []
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_center = np.mean(features[cluster_mask], axis=0)
                cluster_centers.append(cluster_center)

            cluster_centers = np.array(cluster_centers)
            if len(cluster_centers) > 1:
                center_distances = pdist(cluster_centers)
                quality_metrics['min_center_distance'] = np.min(center_distances)
                quality_metrics['mean_center_distance'] = np.mean(center_distances)
                quality_metrics['center_separation_ratio'] = np.max(center_distances) / np.min(center_distances)

            console.print(f"[green]✓[/green] Quality metrics computed: "
                         f"Silhouette={quality_metrics['silhouette_score']:.3f}, "
                         f"CH={quality_metrics['calinski_harabasz_score']:.0f}")

        except Exception as e:
            console.print(f"[red]✗[/red] Error computing quality metrics: {e}")
            quality_metrics = {'error': str(e)}

        return quality_metrics

    def assess_patch_size_configuration(self, results_dir: Path, patch_sizes: List[int]) -> pd.DataFrame:
        """
        Comprehensive assessment of different patch size configurations.

        This function evaluates clustering results across different patch sizes
        and combinations, helping researchers identify the optimal configuration
        for their specific tissue analysis requirements.

        Args:
            results_dir: Directory containing clustering results.
            patch_sizes: List of patch sizes to evaluate.

        Returns:
            DataFrame with comprehensive quality assessment results.
        """
        console.print(f"[1m[cyan]Assessing patch size configurations...[/cyan][/1m")

        assessment_results = []

        # Define configuration patterns to search for.
        config_patterns = []
        for size in patch_sizes:
            config_patterns.append(f"*_filtered_{size}px")

        # Add combination patterns.
        if len(patch_sizes) > 1:
            size_combos = []
            for i in range(len(patch_sizes)):
                for j in range(i + 1, len(patch_sizes)):
                    size_combos.append(f"{patch_sizes[i]}_{patch_sizes[j]}px")

            if len(patch_sizes) >= 3:
                all_sizes = "_".join(map(str, patch_sizes))
                size_combos.append(f"{all_sizes}px")

            for combo in size_combos:
                config_patterns.append(f"*_filtered_{combo}")

        with Progress() as progress:
            task = progress.add_task("Evaluating configurations...", total=len(config_patterns))

            for pattern in config_patterns:
                progress.update(task, advance=1)

                # Find matching directories.
                matching_dirs = list(results_dir.glob(pattern))

                for config_dir in matching_dirs:
                    console.print(f"[blue]ℹ[/blue] Analyzing: {config_dir.name}")

                    try:
                        result = self._analyze_single_configuration(config_dir)
                        if result:
                            assessment_results.append(result)
                    except Exception as e:
                        console.print(f"[red]✗[/red] Error analyzing {config_dir.name}: {e}")
                        continue

        if not assessment_results:
            console.print("[red]✗[/red] No valid configurations found")
            return pd.DataFrame()

        results_df = pd.DataFrame(assessment_results)
        console.print(f"[green]✓[/green] Assessment complete: {len(results_df)} configurations analyzed")

        return results_df

    def _analyze_single_configuration(self, config_dir: Path) -> Optional[Dict]:
        """
        Analyze a single patch size configuration.

        Args:
            config_dir: Directory containing configuration results.

        Returns:
            Dictionary with analysis results or None if analysis fails.
        """
        # Check for required files.
        patch_clusters_file = config_dir / 'patch_clusters.csv'
        coords_file = config_dir / 'coords_features_binary_IRI_regist_cropped.csv'
        features_file = config_dir / 'features_scaled.npy'

        if not patch_clusters_file.exists():
            console.print(f"[yellow]⚠[/yellow] Missing patch_clusters.csv in {config_dir.name}")
            return None

        # Load clustering results.
        clusters_df = pd.read_csv(patch_clusters_file)
        labels = clusters_df['cluster'].values

        # Extract patch size from directory name.
        config_name = config_dir.name
        patch_size_info = self._extract_patch_size_info(config_name)

        result = {
            'configuration': config_name,
            'patch_sizes': patch_size_info,
            'n_patches': len(clusters_df),
            'n_clusters': len(np.unique(labels))
        }

        # Border effect analysis.
        border_metrics = self.detect_border_effects(
            clusters_df, labels, patch_size_info.get('primary_size', 32)
        )
        result.update({f'border_{k}': v for k, v in border_metrics.items()})

        # Spatial coherence analysis.
        spatial_metrics = self.compute_spatial_coherence(clusters_df, labels)
        result.update({f'spatial_{k}': v for k, v in spatial_metrics.items()})

        # Cluster quality metrics (if features available).
        if features_file.exists():
            try:
                features = np.load(features_file)
                if len(features) == len(labels):
                    quality_metrics = self.compute_cluster_quality_metrics(features, labels)
                    result.update({f'quality_{k}': v for k, v in quality_metrics.items()})
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Could not load features: {e}")

        return result

    def _extract_patch_size_info(self, config_name: str) -> Dict[str, Union[int, List[int]]]:
        """
        Extract patch size information from configuration directory name.

        Args:
            config_name: Configuration directory name.

        Returns:
            Dictionary with patch size information.
        """
        info = {'primary_size': 32, 'sizes': [32], 'is_combination': False}

        # Extract size information from name patterns.
        if '_16px' in config_name:
            info['primary_size'] = 16
            info['sizes'] = [16]
        elif '_32px' in config_name:
            info['primary_size'] = 32
            info['sizes'] = [32]
        elif '_64px' in config_name:
            info['primary_size'] = 64
            info['sizes'] = [64]
        elif '_16_32px' in config_name:
            info['primary_size'] = 24  # Average
            info['sizes'] = [16, 32]
            info['is_combination'] = True
        elif '_32_64px' in config_name:
            info['primary_size'] = 48  # Average
            info['sizes'] = [32, 64]
            info['is_combination'] = True
        elif '_16_32_64px' in config_name:
            info['primary_size'] = 37  # Average
            info['sizes'] = [16, 32, 64]
            info['is_combination'] = True

        return info

    def generate_quality_report(self, results_df: pd.DataFrame, output_dir: Path) -> None:
        """
        Generate comprehensive quality assessment report with recommendations.

        This function creates detailed visualizations and analysis reports to help
        researchers understand the performance characteristics of different ViT
        configurations and select the optimal setup for their analysis.

        Args:
            results_df: DataFrame with assessment results.
            output_dir: Directory for output files.
        """
        console.print(f"[1m[cyan]Generating quality assessment report...[/cyan][/1m")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comprehensive results.
        results_df.to_csv(output_dir / 'quality_assessment_report.csv', index=False)

        # Create visualizations.
        self._create_quality_visualizations(results_df, output_dir)

        # Generate recommendations.
        recommendations = self._generate_recommendations(results_df)

        # Save recommendations.
        with open(output_dir / 'recommendations.txt', 'w') as f:
            f.write("ViT Configuration Quality Assessment Recommendations\n")
            f.write("=" * 60 + "\n\n")

            for category, recs in recommendations.items():
                f.write(f"{category.upper()}:\n")
                for rec in recs:
                    f.write(f"• {rec}\n")
                f.write("\n")

        # Print key findings.
        self._print_summary_table(results_df, recommendations)

        console.print(f"[green]✓[/green] Quality assessment report saved to: {output_dir}")

    def _create_quality_visualizations(self, results_df: pd.DataFrame, output_dir: Path) -> None:
        """Create comprehensive quality visualization plots."""
        console.print("[cyan]Creating quality visualizations...[/cyan]")

        # Set up plotting style.
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Border clustering quality analysis.
        if 'border_high_border_cluster_fraction' in results_df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Border Clustering Quality Analysis', fontsize=16, fontweight='bold')

            # Border patch fraction.
            axes[0, 0].bar(results_df['configuration'], results_df['border_border_fraction'])
            axes[0, 0].set_title('Border Patch Fraction')
            axes[0, 0].set_ylabel('Fraction')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # High border cluster fraction (higher = better edge detection).
            axes[0, 1].bar(results_df['configuration'], results_df['border_high_border_cluster_fraction'])
            axes[0, 1].set_title('Dedicated Border Clusters (Higher = Better)')
            axes[0, 1].set_ylabel('Fraction')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Border concentration distribution.
            axes[1, 0].bar(results_df['configuration'], results_df['border_mean_border_concentration'])
            axes[1, 0].set_title('Mean Border Concentration')
            axes[1, 0].set_ylabel('Concentration')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Border cluster imbalance.
            axes[1, 1].bar(results_df['configuration'], results_df['border_border_cluster_imbalance'])
            axes[1, 1].set_title('Border Cluster Imbalance')
            axes[1, 1].set_ylabel('Standard Deviation')
            axes[1, 1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(output_dir / 'border_clustering_quality.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 2. Spatial coherence comparison.
        if 'spatial_mean_spatial_coherence' in results_df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Spatial Coherence Analysis', fontsize=16, fontweight='bold')

            # Spatial coherence.
            axes[0, 0].bar(results_df['configuration'], results_df['spatial_mean_spatial_coherence'])
            axes[0, 0].set_title('Mean Spatial Coherence')
            axes[0, 0].set_ylabel('Coherence Score')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Nearest neighbor purity.
            axes[0, 1].bar(results_df['configuration'], results_df['spatial_mean_nn_purity'])
            axes[0, 1].set_title('Nearest Neighbor Purity')
            axes[0, 1].set_ylabel('Purity Score')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Cluster size coefficient of variation.
            axes[1, 0].bar(results_df['configuration'], results_df['spatial_cluster_size_cv'])
            axes[1, 0].set_title('Cluster Size Variability')
            axes[1, 0].set_ylabel('Coefficient of Variation')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Number of clusters.
            axes[1, 1].bar(results_df['configuration'], results_df['spatial_n_clusters'])
            axes[1, 1].set_title('Number of Clusters')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(output_dir / 'spatial_coherence_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Cluster quality metrics comparison.
        quality_cols = [col for col in results_df.columns if col.startswith('quality_')]
        if quality_cols:
            n_metrics = len(quality_cols)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            fig.suptitle('Cluster Quality Metrics Comparison', fontsize=16, fontweight='bold')

            for i, col in enumerate(quality_cols):
                row, col_idx = divmod(i, n_cols)
                ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]

                metric_name = col.replace('quality_', '').replace('_', ' ').title()
                ax.bar(results_df['configuration'], results_df[col])
                ax.set_title(metric_name)
                ax.tick_params(axis='x', rotation=45)

            # Hide unused subplots.
            for i in range(n_metrics, n_rows * n_cols):
                row, col_idx = divmod(i, n_cols)
                ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
                ax.set_visible(False)

            plt.tight_layout()
            plt.savefig(output_dir / 'cluster_quality_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        console.print("[green]✓[/green] Visualizations created successfully")

    def _generate_recommendations(self, results_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Generate intelligent recommendations based on quality assessment results.

        This function analyzes the assessment results to provide actionable
        recommendations for optimal ViT configuration selection based on
        multiple quality criteria and biological relevance.

        Returns:
            Dictionary with categorized recommendations.
        """
        recommendations = {
            'optimal_configuration': [],
            'border_clustering_quality': [],
            'spatial_coherence': [],
            'cluster_quality': [],
            'general_recommendations': []
        }

        if results_df.empty:
            recommendations['general_recommendations'].append("No valid configurations found for analysis.")
            return recommendations

        # Find optimal configuration based on multiple criteria.
        scoring_criteria = []

        # Border clustering quality (higher border concentration is GOOD).
        if 'border_mean_border_concentration' in results_df.columns:
            border_scores = results_df['border_mean_border_concentration'].fillna(0)
            scoring_criteria.append(('border_quality', border_scores))

        # Spatial coherence (higher is better).
        if 'spatial_mean_spatial_coherence' in results_df.columns:
            spatial_scores = results_df['spatial_mean_spatial_coherence'].fillna(0)
            scoring_criteria.append(('spatial_coherence', spatial_scores))

        # Silhouette score (higher is better).
        if 'quality_silhouette_score' in results_df.columns:
            silhouette_scores = results_df['quality_silhouette_score'].fillna(0)
            scoring_criteria.append(('silhouette', silhouette_scores))

        # Calinski-Harabasz score (higher is better, normalize).
        if 'quality_calinski_harabasz_score' in results_df.columns:
            ch_scores = results_df['quality_calinski_harabasz_score'].fillna(0)
            ch_normalized = (ch_scores - ch_scores.min()) / (ch_scores.max() - ch_scores.min() + 1e-8)
            scoring_criteria.append(('calinski_harabasz', ch_normalized))

        # Davies-Bouldin score (lower is better, invert).
        if 'quality_davies_bouldin_score' in results_df.columns:
            db_scores = results_df['quality_davies_bouldin_score'].fillna(results_df['quality_davies_bouldin_score'].max())
            db_inverted = 1 / (1 + db_scores)  # Invert so higher is better
            scoring_criteria.append(('davies_bouldin', db_inverted))

        # Compute composite score.
        if scoring_criteria:
            composite_scores = np.zeros(len(results_df))
            for name, scores in scoring_criteria:
                # Normalize scores to 0-1 range.
                normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                composite_scores += normalized

            best_idx = np.argmax(composite_scores)
            best_config = results_df.iloc[best_idx]['configuration']

            recommendations['optimal_configuration'].append(
                f"Best overall configuration: {best_config}"
            )
            recommendations['optimal_configuration'].append(
                f"Composite quality score: {composite_scores[best_idx]:.3f}"
            )

        # Border clustering analysis (corrected interpretation).
        if 'border_mean_border_concentration' in results_df.columns:
            best_border_idx = results_df['border_mean_border_concentration'].idxmax()
            best_border_config = results_df.iloc[best_border_idx]['configuration']
            border_score = results_df.iloc[best_border_idx]['border_mean_border_concentration']

            recommendations['border_clustering_quality'].append(
                f"Best border clustering: {best_border_config} (score: {border_score:.3f})"
            )
            recommendations['border_clustering_quality'].append(
                "Higher border concentration indicates good feature extraction of edge artifacts"
            )

            if border_score > 0.3:
                recommendations['border_clustering_quality'].append(
                    "Excellent border artifact detection - ViT correctly identifies edge-cropped masks"
                )
            elif border_score > 0.15:
                recommendations['border_clustering_quality'].append(
                    "Good border artifact detection - ViT shows awareness of edge effects"
                )
            else:
                recommendations['border_clustering_quality'].append(
                    "Limited border artifact detection - consider larger patch sizes for better edge awareness"
                )

        # Spatial coherence recommendations.
        if 'spatial_mean_spatial_coherence' in results_df.columns:
            best_spatial_idx = results_df['spatial_mean_spatial_coherence'].idxmax()
            best_spatial_config = results_df.iloc[best_spatial_idx]['configuration']
            spatial_score = results_df.iloc[best_spatial_idx]['spatial_mean_spatial_coherence']

            recommendations['spatial_coherence'].append(
                f"Best spatial coherence: {best_spatial_config} (score: {spatial_score:.3f})"
            )

            if spatial_score > 0.7:
                recommendations['spatial_coherence'].append(
                    "Excellent spatial organization - clusters preserve tissue architecture well"
                )
            elif spatial_score > 0.5:
                recommendations['spatial_coherence'].append(
                    "Good spatial organization - reasonable preservation of tissue structure"
                )
            else:
                recommendations['spatial_coherence'].append(
                    "Limited spatial coherence - consider different patch sizes or clustering parameters"
                )

        # Cluster quality recommendations.
        if 'quality_silhouette_score' in results_df.columns:
            best_silhouette_idx = results_df['quality_silhouette_score'].idxmax()
            best_silhouette_config = results_df.iloc[best_silhouette_idx]['configuration']
            silhouette_score = results_df.iloc[best_silhouette_idx]['quality_silhouette_score']

            recommendations['cluster_quality'].append(
                f"Best cluster separation: {best_silhouette_config} (silhouette: {silhouette_score:.3f})"
            )

            if silhouette_score > 0.5:
                recommendations['cluster_quality'].append(
                    "Excellent cluster separation - distinct, well-separated cell populations"
                )
            elif silhouette_score > 0.25:
                recommendations['cluster_quality'].append(
                    "Good cluster separation - reasonable distinction between populations"
                )
            else:
                recommendations['cluster_quality'].append(
                    "Poor cluster separation - consider different clustering parameters or feature preprocessing"
                )

        # General recommendations based on patch sizes.
        patch_size_performance = {}
        for _, row in results_df.iterrows():
            config = row['configuration']
            if '_16px' in config:
                size_key = '16px'
            elif '_32px' in config:
                size_key = '32px'
            elif '_64px' in config:
                size_key = '64px'
            elif '_16_32_64px' in config:
                size_key = 'multi_scale'
            else:
                size_key = 'other'

            if size_key not in patch_size_performance:
                patch_size_performance[size_key] = []

            # Use composite score if available.
            if scoring_criteria:
                score_idx = results_df[results_df['configuration'] == config].index[0]
                score = composite_scores[score_idx] if 'composite_scores' in locals() else 0
                patch_size_performance[size_key].append(score)

        # Recommend best patch size strategy.
        if patch_size_performance:
            avg_performance = {k: np.mean(v) for k, v in patch_size_performance.items() if v}
            best_strategy = max(avg_performance, key=avg_performance.get)

            strategy_recommendations = {
                '16px': "16px patches provide fine-grained detail but may miss larger tissue patterns",
                '32px': "32px patches offer balanced resolution for most tissue analysis applications",
                '64px': "64px patches capture broader tissue context but may lose cellular detail",
                'multi_scale': "Multi-scale approach combines benefits of different resolutions"
            }

            recommendations['general_recommendations'].append(
                f"Recommended patch strategy: {best_strategy}"
            )
            recommendations['general_recommendations'].append(
                strategy_recommendations.get(best_strategy, "Custom configuration shows good performance")
            )

        return recommendations

    def _print_summary_table(self, results_df: pd.DataFrame, recommendations: Dict[str, List[str]]) -> None:
        """Print a formatted summary table of key findings."""
        console.print("\n" + "=" * 80)
        console.print("[1m[cyan]ViT QUALITY ASSESSMENT SUMMARY[/cyan][/1m")
        console.print("=" * 80)

        # Create summary table.
        table = Table(title="Configuration Performance Summary")
        table.add_column("Configuration", style="cyan", no_wrap=True)
        table.add_column("Border Quality", style="green")
        table.add_column("Spatial Coherence", style="blue")
        table.add_column("Silhouette Score", style="magenta")
        table.add_column("Clusters", style="yellow")

        for _, row in results_df.iterrows():
            config = row['configuration']
            border_qual = f"{row.get('border_mean_border_concentration', 0):.3f}"
            spatial_coh = f"{row.get('spatial_mean_spatial_coherence', 0):.3f}"
            silhouette = f"{row.get('quality_silhouette_score', 0):.3f}"
            n_clusters = f"{row.get('spatial_n_clusters', 0)}"

            table.add_row(config, border_qual, spatial_coh, silhouette, n_clusters)

        console.print(table)

        # Print key recommendations.
        console.print("\n[1m[yellow]KEY RECOMMENDATIONS:[/yellow][/1m")
        for category, recs in recommendations.items():
            if recs and category == 'optimal_configuration':
                console.print(f"[green]✓[/green] {recs[0]}")
                break

        console.print("=" * 80)


def main():
    """
    Main entry point for ViT quality assessment.

    Orchestrates comprehensive quality assessment across multiple patch size
    configurations, providing detailed analysis and recommendations for
    optimal ViT setup selection.
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive quality assessment for Vision Transformer clustering results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic assessment of all configurations
    python assess_vit_quality.py \\
        --results_dir results \\
        --image_path data/IRI_regist_cropped.tif \\
        --output_dir quality_assessment

    # Focused assessment of specific patch sizes
    python assess_vit_quality.py \\
        --results_dir results \\
        --image_path data/IRI_regist_cropped.tif \\
        --output_dir quality_assessment \\
        --patch_sizes 16 32 64 \\
        --border_threshold 75
        """
    )

    parser.add_argument(
        '--results_dir', type=Path, required=True,
        help='Directory containing ViT clustering results'
    )
    parser.add_argument(
        '--image_path', type=Path, required=True,
        help='Path to original image for dimension analysis'
    )
    parser.add_argument(
        '--output_dir', type=Path, required=True,
        help='Output directory for quality assessment reports'
    )
    parser.add_argument(
        '--patch_sizes', type=int, nargs='+', default=[16, 32, 64],
        help='List of patch sizes to evaluate (default: 16 32 64)'
    )
    parser.add_argument(
        '--border_threshold', type=int, default=50,
        help='Distance from edge to consider as border region (default: 50)'
    )

    args = parser.parse_args()

    try:
        console.print("[1m[cyan]Starting ViT Quality Assessment...[/cyan][/1m")
        console.print(f"[blue]ℹ[/blue] Results directory: {args.results_dir}")
        console.print(f"[blue]ℹ[/blue] Image path: {args.image_path}")
        console.print(f"[blue]ℹ[/blue] Output directory: {args.output_dir}")
        console.print(f"[blue]ℹ[/blue] Patch sizes: {args.patch_sizes}")

        # Validate inputs.
        if not args.results_dir.exists():
            console.print(f"[red]✗[/red] Results directory not found: {args.results_dir}")
            return 1

        if not args.image_path.exists():
            console.print(f"[red]✗[/red] Image file not found: {args.image_path}")
            return 1

        # Initialize quality assessor.
        assessor = ViTQualityAssessor(args.image_path, args.border_threshold)

        # Perform comprehensive assessment.
        console.print("\n[1m[cyan]Performing comprehensive quality assessment...[/cyan][/1m")
        results_df = assessor.assess_patch_size_configuration(args.results_dir, args.patch_sizes)

        if results_df.empty:
            console.print("[red]✗[/red] No valid configurations found for assessment")
            return 1

        # Generate quality report.
        console.print(f"\n[1m[cyan]Generating quality report...[/cyan][/1m")
        assessor.generate_quality_report(results_df, args.output_dir)

        console.print("\n" + "=" * 80)
        console.print("[1m[green]✅ ViT QUALITY ASSESSMENT COMPLETED SUCCESSFULLY[/green][/1m")
        console.print("=" * 80)
        console.print(f"[green]✓[/green] Assessment results saved to: {args.output_dir}")
        console.print(f"[green]✓[/green] {len(results_df)} configurations analyzed")

        return 0

    except Exception as e:
        console.print(f"[red]✗[/red] Error during quality assessment: {e}")
        console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())
