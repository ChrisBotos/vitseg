"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: cluster_metrics.py.
Description:
    Comprehensive module for calculating statistical metrics to evaluate alignment
    between ViT-derived clusters and spatial spot clusters. Implements rigorous
    statistical methods including ARI, NMI, silhouette analysis, and spatial
    correlation metrics for quantitative cluster comparison.

Dependencies:
    • Python >= 3.10.
    • pandas, numpy, scipy, scikit-learn.
    • rich (for enhanced console output).
    • statsmodels (for spatial statistics).

Usage:
    python -m vigseg.comparison.cluster_metrics \
        --vit_data results/spot_nuclei_clustering/spot_clusters.csv \
        --spatial_data data/metadata_complete.csv \
        --output comparison_analysis/results/metrics/ \
        --samples IRI1 IRI2 IRI3 \
        --cluster_column figure_idents
"""
import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, 
    silhouette_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table

console = Console()


def load_and_align_data(vit_path: Path, spatial_path: Path, 
                       samples: List[str], cluster_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and align ViT cluster data with spatial cluster data.
    
    Args:
        vit_path: Path to ViT cluster assignments.
        spatial_path: Path to spatial metadata.
        samples: List of sample names to include.
        cluster_column: Column name for spatial clusters.
        
    Returns:
        Tuple of aligned (vit_data, spatial_data) DataFrames.
        
    This function handles coordinate system alignment and ensures both datasets
    contain the same spatial locations for valid comparison.
    """
    console.print(f"[cyan]Loading ViT cluster data from {vit_path}...[/cyan]")
    vit_data = pd.read_csv(vit_path)
    
    console.print(f"[cyan]Loading spatial cluster data from {spatial_path}...[/cyan]")
    spatial_data = pd.read_csv(spatial_path)
    
    # Filter for specified samples.
    vit_filtered = vit_data[vit_data['sample'].isin(samples)].copy()
    spatial_filtered = spatial_data[spatial_data['sample'].isin(samples)].copy()
    
    console.print(f"[green]✓[/green] Loaded {len(vit_filtered):,} ViT spots and {len(spatial_filtered):,} spatial spots")
    
    # Align datasets by finding nearest spatial coordinates.
    aligned_vit, aligned_spatial = align_coordinates(vit_filtered, spatial_filtered)
    
    console.print(f"[green]✓[/green] Aligned {len(aligned_vit):,} matching locations")
    
    return aligned_vit, aligned_spatial


def align_coordinates(vit_data: pd.DataFrame, spatial_data: pd.DataFrame, 
                     max_distance: float = 50.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align ViT and spatial data by matching nearest coordinates.
    
    Args:
        vit_data: ViT cluster data with spot_x, spot_y coordinates.
        spatial_data: Spatial data with x, y coordinates.
        max_distance: Maximum distance for coordinate matching.
        
    Returns:
        Tuple of aligned DataFrames with matching spatial locations.
        
    This function ensures both datasets reference the same tissue locations
    for valid cluster comparison analysis.
    """
    console.print("[cyan]Aligning coordinate systems...[/cyan]")
    
    aligned_pairs = []
    
    for _, vit_row in track(vit_data.iterrows(), total=len(vit_data), description="Matching coordinates"):
        vit_x, vit_y = vit_row['spot_x'], vit_row['spot_y']
        sample = vit_row['sample']
        
        # Find spatial points in same sample.
        sample_spatial = spatial_data[spatial_data['sample'] == sample]
        
        if len(sample_spatial) == 0:
            continue
            
        # Calculate distances to all spatial points.
        distances = np.sqrt((sample_spatial['x'] - vit_x)**2 + (sample_spatial['y'] - vit_y)**2)
        min_distance = distances.min()
        
        if min_distance <= max_distance:
            closest_idx = distances.idxmin()
            spatial_row = spatial_data.loc[closest_idx]
            
            aligned_pairs.append({
                'vit_data': vit_row,
                'spatial_data': spatial_row,
                'distance': min_distance
            })
    
    if len(aligned_pairs) == 0:
        raise ValueError(f"No coordinate matches found within {max_distance} pixels")
    
    # Create aligned DataFrames.
    aligned_vit = pd.DataFrame([pair['vit_data'] for pair in aligned_pairs])
    aligned_spatial = pd.DataFrame([pair['spatial_data'] for pair in aligned_pairs])
    
    # Add distance information.
    distances = [pair['distance'] for pair in aligned_pairs]
    aligned_vit['alignment_distance'] = distances
    aligned_spatial['alignment_distance'] = distances
    
    console.print(f"[blue]ℹ[/blue] Mean alignment distance: {np.mean(distances):.2f} pixels")
    console.print(f"[blue]ℹ[/blue] Max alignment distance: {np.max(distances):.2f} pixels")
    
    return aligned_vit, aligned_spatial


def calculate_adjusted_rand_index(vit_clusters: np.ndarray, spatial_clusters: np.ndarray) -> Dict[str, float]:
    """
    Calculate Adjusted Rand Index with confidence intervals.
    
    Args:
        vit_clusters: ViT cluster assignments.
        spatial_clusters: Spatial cluster assignments.
        
    Returns:
        Dictionary with ARI score and confidence intervals.
        
    ARI measures the similarity between two clusterings, adjusted for chance.
    Values range from -1 to 1, with 1 indicating perfect agreement.
    """
    console.print("[cyan]Calculating Adjusted Rand Index...[/cyan]")
    
    # Calculate ARI.
    ari_score = adjusted_rand_score(vit_clusters, spatial_clusters)
    
    # Bootstrap confidence intervals.
    n_bootstrap = 1000
    bootstrap_scores = []
    
    for _ in track(range(n_bootstrap), description="Bootstrap ARI"):
        # Resample with replacement.
        indices = np.random.choice(len(vit_clusters), size=len(vit_clusters), replace=True)
        boot_vit = vit_clusters[indices]
        boot_spatial = spatial_clusters[indices]
        
        boot_ari = adjusted_rand_score(boot_vit, boot_spatial)
        bootstrap_scores.append(boot_ari)
    
    # Calculate confidence intervals.
    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)
    
    results = {
        'ari_score': float(ari_score),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'bootstrap_std': float(np.std(bootstrap_scores))
    }
    
    console.print(f"[green]✓[/green] ARI: {ari_score:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")
    
    return results


def calculate_normalized_mutual_information(vit_clusters: np.ndarray, spatial_clusters: np.ndarray) -> Dict[str, float]:
    """
    Calculate Normalized Mutual Information with statistical testing.
    
    Args:
        vit_clusters: ViT cluster assignments.
        spatial_clusters: Spatial cluster assignments.
        
    Returns:
        Dictionary with NMI score and statistical significance.
        
    NMI measures the amount of information shared between two clusterings,
    normalized to range from 0 to 1.
    """
    console.print("[cyan]Calculating Normalized Mutual Information...[/cyan]")
    
    # Calculate NMI.
    nmi_score = normalized_mutual_info_score(vit_clusters, spatial_clusters)
    
    # Permutation test for significance.
    n_permutations = 1000
    permutation_scores = []
    
    for _ in track(range(n_permutations), description="Permutation test"):
        # Randomly permute one clustering.
        permuted_spatial = np.random.permutation(spatial_clusters)
        perm_nmi = normalized_mutual_info_score(vit_clusters, permuted_spatial)
        permutation_scores.append(perm_nmi)
    
    # Calculate p-value.
    p_value = np.mean(np.array(permutation_scores) >= nmi_score)
    
    results = {
        'nmi_score': float(nmi_score),
        'p_value': float(p_value),
        'permutation_mean': float(np.mean(permutation_scores)),
        'permutation_std': float(np.std(permutation_scores))
    }
    
    console.print(f"[green]✓[/green] NMI: {nmi_score:.4f} (p-value: {p_value:.4f})")
    
    return results


def calculate_silhouette_analysis(coordinates: np.ndarray, vit_clusters: np.ndarray, 
                                spatial_clusters: np.ndarray) -> Dict[str, Any]:
    """
    Perform comprehensive silhouette analysis for both clusterings.
    
    Args:
        coordinates: Spatial coordinates for distance calculation.
        vit_clusters: ViT cluster assignments.
        spatial_clusters: Spatial cluster assignments.
        
    Returns:
        Dictionary with silhouette scores and detailed analysis.
        
    Silhouette analysis measures how well-separated clusters are in space,
    providing insight into cluster quality and spatial coherence.
    """
    console.print("[cyan]Performing silhouette analysis...[/cyan]")
    
    # Calculate silhouette scores.
    vit_silhouette = silhouette_score(coordinates, vit_clusters)
    spatial_silhouette = silhouette_score(coordinates, spatial_clusters)
    
    # Per-sample silhouette scores.
    from sklearn.metrics import silhouette_samples
    
    vit_sample_scores = silhouette_samples(coordinates, vit_clusters)
    spatial_sample_scores = silhouette_samples(coordinates, spatial_clusters)
    
    # Cluster-wise analysis.
    vit_cluster_scores = {}
    spatial_cluster_scores = {}
    
    for cluster in np.unique(vit_clusters):
        mask = vit_clusters == cluster
        vit_cluster_scores[str(cluster)] = {
            'mean_score': float(np.mean(vit_sample_scores[mask])),
            'std_score': float(np.std(vit_sample_scores[mask])),
            'n_points': int(np.sum(mask))
        }
    
    for cluster in np.unique(spatial_clusters):
        mask = spatial_clusters == cluster
        spatial_cluster_scores[str(cluster)] = {
            'mean_score': float(np.mean(spatial_sample_scores[mask])),
            'std_score': float(np.std(spatial_sample_scores[mask])),
            'n_points': int(np.sum(mask))
        }
    
    results = {
        'vit_silhouette': float(vit_silhouette),
        'spatial_silhouette': float(spatial_silhouette),
        'silhouette_difference': float(vit_silhouette - spatial_silhouette),
        'vit_cluster_scores': vit_cluster_scores,
        'spatial_cluster_scores': spatial_cluster_scores,
        'vit_sample_scores': vit_sample_scores.tolist(),
        'spatial_sample_scores': spatial_sample_scores.tolist()
    }
    
    console.print(f"[green]✓[/green] ViT Silhouette: {vit_silhouette:.4f}")
    console.print(f"[green]✓[/green] Spatial Silhouette: {spatial_silhouette:.4f}")
    console.print(f"[blue]ℹ[/blue] Difference: {vit_silhouette - spatial_silhouette:.4f}")
    
    return results


def create_confusion_matrix_analysis(vit_clusters: np.ndarray, spatial_clusters: np.ndarray) -> Dict[str, Any]:
    """
    Create and analyze confusion matrix between clusterings.
    
    Args:
        vit_clusters: ViT cluster assignments.
        spatial_clusters: Spatial cluster assignments.
        
    Returns:
        Dictionary with confusion matrix and derived statistics.
        
    Confusion matrix analysis reveals how cluster assignments correspond
    between the two clustering approaches.
    """
    console.print("[cyan]Creating confusion matrix analysis...[/cyan]")
    
    # Create confusion matrix.
    cm = confusion_matrix(spatial_clusters, vit_clusters)
    
    # Get unique labels.
    spatial_labels = np.unique(spatial_clusters)
    vit_labels = np.unique(vit_clusters)
    
    # Calculate derived statistics.
    total_points = cm.sum()
    diagonal_sum = np.trace(cm)
    accuracy = diagonal_sum / total_points if total_points > 0 else 0.0
    
    # Per-cluster statistics.
    cluster_stats = {}
    for i, spatial_label in enumerate(spatial_labels):
        if i < cm.shape[0]:
            row_sum = cm[i, :].sum()
            max_overlap = cm[i, :].max()
            best_match_idx = cm[i, :].argmax()
            best_match_vit = vit_labels[best_match_idx] if best_match_idx < len(vit_labels) else None
            
            cluster_stats[str(spatial_label)] = {
                'total_points': int(row_sum),
                'max_overlap': int(max_overlap),
                'overlap_fraction': float(max_overlap / row_sum) if row_sum > 0 else 0.0,
                'best_match_vit_cluster': str(best_match_vit) if best_match_vit is not None else None
            }
    
    results = {
        'confusion_matrix': cm.tolist(),
        'spatial_labels': spatial_labels.tolist(),
        'vit_labels': vit_labels.tolist(),
        'overall_accuracy': float(accuracy),
        'total_points': int(total_points),
        'diagonal_sum': int(diagonal_sum),
        'cluster_statistics': cluster_stats
    }
    
    console.print(f"[green]✓[/green] Overall accuracy: {accuracy:.4f}")
    console.print(f"[blue]ℹ[/blue] Matrix shape: {cm.shape}")

    return results


def calculate_effect_sizes(vit_clusters: np.ndarray, spatial_clusters: np.ndarray,
                          coordinates: np.ndarray) -> Dict[str, float]:
    """
    Calculate effect sizes for practical significance assessment.

    Args:
        vit_clusters: ViT cluster assignments.
        spatial_clusters: Spatial cluster assignments.
        coordinates: Spatial coordinates.

    Returns:
        Dictionary with various effect size measures.

    Effect sizes provide measures of practical significance beyond
    statistical significance, indicating the magnitude of clustering differences.
    """
    console.print("[cyan]Calculating effect sizes...[/cyan]")

    # Cohen's kappa for agreement.
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(vit_clusters, spatial_clusters)

    # Cramer's V for association strength.
    cm = confusion_matrix(spatial_clusters, vit_clusters)
    chi2 = stats.chi2_contingency(cm)[0]
    n = cm.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(cm.shape) - 1)))

    # Spatial coherence difference.
    vit_silhouette = silhouette_score(coordinates, vit_clusters)
    spatial_silhouette = silhouette_score(coordinates, spatial_clusters)
    coherence_effect = (vit_silhouette - spatial_silhouette) / spatial_silhouette if spatial_silhouette != 0 else 0

    results = {
        'cohens_kappa': float(kappa),
        'cramers_v': float(cramers_v),
        'coherence_effect_size': float(coherence_effect),
        'chi2_statistic': float(chi2)
    }

    console.print(f"[green]✓[/green] Cohen's κ: {kappa:.4f}")
    console.print(f"[green]✓[/green] Cramer's V: {cramers_v:.4f}")

    return results


def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Save comprehensive analysis results to files.

    Args:
        results: Dictionary containing all analysis results.
        output_dir: Directory for saving results.

    This function creates multiple output formats for different use cases:
    JSON for programmatic access, CSV for data analysis, and TXT for reports.
    """
    console.print("[cyan]Saving analysis results...[/cyan]")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comprehensive JSON results.
    json_path = output_dir / 'alignment_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]✓[/green] Saved metrics to {json_path}")

    # Save silhouette analysis as CSV.
    if 'silhouette_analysis' in results:
        silhouette_data = []
        for cluster, stats in results['silhouette_analysis']['vit_cluster_scores'].items():
            silhouette_data.append({
                'cluster_type': 'ViT',
                'cluster_id': cluster,
                'mean_score': stats['mean_score'],
                'std_score': stats['std_score'],
                'n_points': stats['n_points']
            })

        for cluster, stats in results['silhouette_analysis']['spatial_cluster_scores'].items():
            silhouette_data.append({
                'cluster_type': 'Spatial',
                'cluster_id': cluster,
                'mean_score': stats['mean_score'],
                'std_score': stats['std_score'],
                'n_points': stats['n_points']
            })

        silhouette_df = pd.DataFrame(silhouette_data)
        silhouette_path = output_dir / 'silhouette_analysis.csv'
        silhouette_df.to_csv(silhouette_path, index=False)
        console.print(f"[green]✓[/green] Saved silhouette analysis to {silhouette_path}")

    # Save confusion matrix as CSV.
    if 'confusion_matrix' in results:
        cm_data = results['confusion_matrix']
        cm_df = pd.DataFrame(
            cm_data['confusion_matrix'],
            index=[f"Spatial_{label}" for label in cm_data['spatial_labels']],
            columns=[f"ViT_{label}" for label in cm_data['vit_labels']]
        )
        cm_path = output_dir / 'confusion_matrix.csv'
        cm_df.to_csv(cm_path)
        console.print(f"[green]✓[/green] Saved confusion matrix to {cm_path}")

    # Create human-readable summary.
    summary_path = output_dir / 'metrics_summary.txt'
    create_summary_report(results, summary_path)


def create_summary_report(results: Dict[str, Any], output_path: Path) -> None:
    """
    Create human-readable summary report of analysis results.

    Args:
        results: Dictionary containing all analysis results.
        output_path: Path for saving the summary report.

    This function generates a comprehensive text report suitable for
    scientific documentation and interpretation.
    """
    with open(output_path, 'w') as f:
        f.write("ViT-Spatial Cluster Alignment Analysis Summary\n")
        f.write("=" * 50 + "\n\n")

        # ARI results.
        if 'ari' in results:
            ari = results['ari']
            f.write("Adjusted Rand Index (ARI)\n")
            f.write("-" * 25 + "\n")
            f.write(f"ARI Score: {ari['ari_score']:.4f}\n")
            f.write(f"95% Confidence Interval: [{ari['ci_lower']:.4f}, {ari['ci_upper']:.4f}]\n")
            f.write(f"Bootstrap Standard Error: {ari['bootstrap_std']:.4f}\n\n")

            # Interpretation.
            if ari['ari_score'] > 0.7:
                interpretation = "Strong alignment"
            elif ari['ari_score'] > 0.3:
                interpretation = "Moderate alignment"
            else:
                interpretation = "Weak alignment"
            f.write(f"Interpretation: {interpretation}\n\n")

        # NMI results.
        if 'nmi' in results:
            nmi = results['nmi']
            f.write("Normalized Mutual Information (NMI)\n")
            f.write("-" * 35 + "\n")
            f.write(f"NMI Score: {nmi['nmi_score']:.4f}\n")
            f.write(f"P-value (permutation test): {nmi['p_value']:.4f}\n")
            f.write(f"Null distribution mean: {nmi['permutation_mean']:.4f}\n")
            f.write(f"Null distribution std: {nmi['permutation_std']:.4f}\n\n")

            significance = "Significant" if nmi['p_value'] < 0.05 else "Not significant"
            f.write(f"Statistical significance: {significance}\n\n")

        # Silhouette analysis.
        if 'silhouette_analysis' in results:
            sil = results['silhouette_analysis']
            f.write("Silhouette Analysis\n")
            f.write("-" * 20 + "\n")
            f.write(f"ViT Clustering Silhouette: {sil['vit_silhouette']:.4f}\n")
            f.write(f"Spatial Clustering Silhouette: {sil['spatial_silhouette']:.4f}\n")
            f.write(f"Difference (ViT - Spatial): {sil['silhouette_difference']:.4f}\n\n")

            better_clustering = "ViT" if sil['silhouette_difference'] > 0 else "Spatial"
            f.write(f"Better spatial coherence: {better_clustering} clustering\n\n")

        # Effect sizes.
        if 'effect_sizes' in results:
            eff = results['effect_sizes']
            f.write("Effect Sizes\n")
            f.write("-" * 12 + "\n")
            f.write(f"Cohen's Kappa: {eff['cohens_kappa']:.4f}\n")
            f.write(f"Cramer's V: {eff['cramers_v']:.4f}\n")
            f.write(f"Coherence Effect Size: {eff['coherence_effect_size']:.4f}\n\n")

        # Confusion matrix summary.
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            f.write("Confusion Matrix Summary\n")
            f.write("-" * 25 + "\n")
            f.write(f"Overall Accuracy: {cm['overall_accuracy']:.4f}\n")
            f.write(f"Total Data Points: {cm['total_points']:,}\n")
            f.write(f"Matrix Dimensions: {len(cm['spatial_labels'])} x {len(cm['vit_labels'])}\n\n")

            f.write("Best Cluster Matches:\n")
            for spatial_cluster, stats in cm['cluster_statistics'].items():
                f.write(f"  Spatial {spatial_cluster} -> ViT {stats['best_match_vit_cluster']} ")
                f.write(f"({stats['overlap_fraction']:.2%} overlap)\n")

    console.print(f"[green]✓[/green] Summary report saved to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Calculate comprehensive cluster alignment metrics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--vit_data', type=Path, required=True,
                       help='Path to ViT cluster assignments CSV')
    parser.add_argument('--spatial_data', type=Path, required=True,
                       help='Path to spatial metadata CSV')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for results')
    parser.add_argument('--samples', nargs='+', default=['IRI1', 'IRI2', 'IRI3'],
                       help='Sample names to include')
    parser.add_argument('--cluster_column', default='figure_idents',
                       help='Column name for spatial clusters')
    parser.add_argument('--max_distance', type=float, default=50.0,
                       help='Maximum distance for coordinate alignment')

    args = parser.parse_args()

    try:
        console.print(f"[1m[36mCluster Alignment Metrics Analysis[/36m[/1m")
        console.print(f"[blue]ℹ[/blue] Samples: [1m{', '.join(args.samples)}[/1m")
        console.print(f"[blue]ℹ[/blue] Spatial clusters: [1m{args.cluster_column}[/1m")

        # Load and align data.
        vit_data, spatial_data = load_and_align_data(
            args.vit_data, args.spatial_data, args.samples, args.cluster_column
        )

        # Prepare cluster arrays.
        vit_clusters = vit_data['cluster'].values
        spatial_clusters = spatial_data[args.cluster_column].values
        coordinates = vit_data[['spot_x', 'spot_y']].values

        # Encode string labels to integers.
        le_spatial = LabelEncoder()
        spatial_encoded = le_spatial.fit_transform(spatial_clusters)

        # Calculate all metrics.
        results = {}

        results['ari'] = calculate_adjusted_rand_index(vit_clusters, spatial_encoded)
        results['nmi'] = calculate_normalized_mutual_information(vit_clusters, spatial_encoded)
        results['silhouette_analysis'] = calculate_silhouette_analysis(coordinates, vit_clusters, spatial_encoded)
        results['confusion_matrix'] = create_confusion_matrix_analysis(vit_clusters, spatial_encoded)
        results['effect_sizes'] = calculate_effect_sizes(vit_clusters, spatial_encoded, coordinates)

        # Add metadata.
        results['metadata'] = {
            'n_points': len(vit_clusters),
            'n_vit_clusters': len(np.unique(vit_clusters)),
            'n_spatial_clusters': len(np.unique(spatial_encoded)),
            'samples': args.samples,
            'cluster_column': args.cluster_column,
            'spatial_labels': le_spatial.classes_.tolist()
        }

        # Save results.
        save_results(results, args.output)

        # Display summary table.
        table = Table(title="Analysis Summary", style="cyan")
        table.add_column("Metric", style="white")
        table.add_column("Value", style="green")

        table.add_row("Data Points", f"{len(vit_clusters):,}")
        table.add_row("ViT Clusters", f"{len(np.unique(vit_clusters))}")
        table.add_row("Spatial Clusters", f"{len(np.unique(spatial_encoded))}")
        table.add_row("ARI Score", f"{results['ari']['ari_score']:.4f}")
        table.add_row("NMI Score", f"{results['nmi']['nmi_score']:.4f}")
        table.add_row("Overall Accuracy", f"{results['confusion_matrix']['overall_accuracy']:.4f}")

        console.print(table)
        console.print(f"[green]✓[/green] [1mAnalysis complete![/1m")

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
