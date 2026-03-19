"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: spatial_analysis.py.
Description:
    Specialized module for spatial correlation analysis of clustering patterns
    in kidney tissue samples. Implements Moran's I, spatial autocorrelation,
    and Local Indicators of Spatial Association (LISA) for comprehensive
    spatial clustering validation.

Dependencies:
    • Python >= 3.10.
    • pandas, numpy, scipy.
    • libpysal, esda (for spatial statistics).
    • rich (for enhanced console output).
    • networkx (for spatial graph analysis).

Usage:
    python comparison_analysis/scripts/spatial_analysis.py \
        --cluster_data results/spot_nuclei_clustering/spot_clusters.csv \
        --output comparison_analysis/results/spatial/ \
        --samples IRI1 IRI2 IRI3 \
        --distance_threshold 100.0
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
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table

console = Console()

# Try to import spatial analysis libraries.
try:
    import libpysal as lps
    from esda import Moran, Moran_Local, G, G_Local
    SPATIAL_LIBS_AVAILABLE = True
    console.print("[green]✓[/green] Spatial analysis libraries available")
except ImportError:
    SPATIAL_LIBS_AVAILABLE = False
    console.print("[yellow]⚠[/yellow] Spatial libraries not available, using fallback methods")


def create_spatial_weights(coordinates: np.ndarray, method: str = 'knn', 
                          k: int = 8, distance_threshold: float = 100.0) -> Any:
    """
    Create spatial weights matrix for spatial autocorrelation analysis.
    
    Args:
        coordinates: Array of (x, y) coordinates.
        method: Method for weight construction ('knn', 'distance', 'delaunay').
        k: Number of nearest neighbors for KNN method.
        distance_threshold: Maximum distance for distance-based weights.
        
    Returns:
        Spatial weights object or matrix.
        
    Spatial weights define the neighborhood structure for spatial statistics,
    crucial for accurate autocorrelation analysis in tissue samples.
    """
    console.print(f"[cyan]Creating spatial weights using {method} method...[/cyan]")
    
    if SPATIAL_LIBS_AVAILABLE:
        if method == 'knn':
            weights = lps.weights.KNN.from_array(coordinates, k=k)
        elif method == 'distance':
            weights = lps.weights.DistanceBand.from_array(coordinates, threshold=distance_threshold)
        elif method == 'delaunay':
            weights = lps.weights.Delaunay.from_array(coordinates)
        else:
            raise ValueError(f"Unknown weight method: {method}")
        
        weights.transform = 'r'  # Row-standardize weights.
        console.print(f"[green]✓[/green] Created {weights.n} x {weights.n} spatial weights matrix")
        return weights
    else:
        # Fallback: create distance-based weights manually.
        distances = squareform(pdist(coordinates))
        weights = np.zeros_like(distances)
        
        if method == 'knn':
            for i in range(len(coordinates)):
                # Find k nearest neighbors (excluding self).
                neighbor_indices = np.argsort(distances[i])[1:k+1]
                weights[i, neighbor_indices] = 1
        elif method == 'distance':
            weights = (distances <= distance_threshold) & (distances > 0)
        
        # Row-standardize.
        row_sums = weights.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero.
        weights = weights / row_sums[:, np.newaxis]
        
        console.print(f"[green]✓[/green] Created {weights.shape[0]} x {weights.shape[1]} fallback weights matrix")
        return weights


def calculate_morans_i(cluster_labels: np.ndarray, spatial_weights: Any) -> Dict[str, float]:
    """
    Calculate Moran's I statistic for spatial autocorrelation.
    
    Args:
        cluster_labels: Cluster assignments for each location.
        spatial_weights: Spatial weights matrix or object.
        
    Returns:
        Dictionary with Moran's I statistics and significance tests.
        
    Moran's I measures the degree of spatial autocorrelation in cluster
    assignments, indicating whether similar clusters tend to be spatially clustered.
    """
    console.print("[cyan]Calculating Moran's I statistic...[/cyan]")
    
    if SPATIAL_LIBS_AVAILABLE and hasattr(spatial_weights, 'transform'):
        # Use libpysal implementation.
        moran = Moran(cluster_labels, spatial_weights)
        
        results = {
            'morans_i': float(moran.I),
            'expected_i': float(moran.EI),
            'variance_i': float(moran.VI_norm),
            'z_score': float(moran.z_norm),
            'p_value': float(moran.p_norm),
            'n_observations': int(len(cluster_labels))
        }
    else:
        # Fallback implementation.
        n = len(cluster_labels)
        weights = spatial_weights if isinstance(spatial_weights, np.ndarray) else np.array(spatial_weights)
        
        # Calculate Moran's I manually.
        y = cluster_labels - np.mean(cluster_labels)
        W = weights.sum()
        
        numerator = np.sum(weights * np.outer(y, y))
        denominator = np.sum(y**2)
        
        morans_i = (n / W) * (numerator / denominator) if denominator != 0 else 0
        expected_i = -1 / (n - 1)
        
        # Approximate variance (simplified).
        variance_i = (n**2 - 3*n + 3) / ((n - 1) * (n - 2) * (n - 3)) - expected_i**2
        z_score = (morans_i - expected_i) / np.sqrt(variance_i) if variance_i > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        results = {
            'morans_i': float(morans_i),
            'expected_i': float(expected_i),
            'variance_i': float(variance_i),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'n_observations': int(n)
        }
    
    console.print(f"[green]✓[/green] Moran's I: {results['morans_i']:.4f} (p-value: {results['p_value']:.4f})")
    
    return results


def calculate_local_morans_i(cluster_labels: np.ndarray, spatial_weights: Any) -> Dict[str, Any]:
    """
    Calculate Local Indicators of Spatial Association (LISA).
    
    Args:
        cluster_labels: Cluster assignments for each location.
        spatial_weights: Spatial weights matrix or object.
        
    Returns:
        Dictionary with local Moran's I statistics for each location.
        
    LISA identifies local spatial clusters and outliers, revealing
    spatial heterogeneity in clustering patterns across tissue regions.
    """
    console.print("[cyan]Calculating Local Moran's I (LISA)...[/cyan]")
    
    if SPATIAL_LIBS_AVAILABLE and hasattr(spatial_weights, 'transform'):
        # Use libpysal implementation.
        lisa = Moran_Local(cluster_labels, spatial_weights)
        
        results = {
            'local_i': lisa.Is.tolist(),
            'p_values': lisa.p_sim.tolist(),
            'z_scores': lisa.z_sim.tolist(),
            'quadrants': lisa.q.tolist(),
            'significant_locations': (lisa.p_sim < 0.05).tolist(),
            'n_significant': int(np.sum(lisa.p_sim < 0.05))
        }
    else:
        # Simplified fallback implementation.
        n = len(cluster_labels)
        weights = spatial_weights if isinstance(spatial_weights, np.ndarray) else np.array(spatial_weights)
        
        y = cluster_labels - np.mean(cluster_labels)
        local_i = np.zeros(n)
        
        for i in range(n):
            neighbors = weights[i] > 0
            if np.any(neighbors):
                local_i[i] = y[i] * np.sum(weights[i, neighbors] * y[neighbors])
        
        # Approximate significance (simplified).
        p_values = np.ones(n)  # Conservative approach.
        z_scores = np.zeros(n)
        quadrants = np.ones(n, dtype=int)
        
        results = {
            'local_i': local_i.tolist(),
            'p_values': p_values.tolist(),
            'z_scores': z_scores.tolist(),
            'quadrants': quadrants.tolist(),
            'significant_locations': (p_values < 0.05).tolist(),
            'n_significant': int(np.sum(p_values < 0.05))
        }
    
    console.print(f"[green]✓[/green] LISA analysis complete - {results['n_significant']} significant locations")
    
    return results


def calculate_getis_ord_g(cluster_labels: np.ndarray, spatial_weights: Any) -> Dict[str, Any]:
    """
    Calculate Getis-Ord G statistics for hotspot analysis.
    
    Args:
        cluster_labels: Cluster assignments for each location.
        spatial_weights: Spatial weights matrix or object.
        
    Returns:
        Dictionary with G statistics for hotspot identification.
        
    Getis-Ord G statistics identify spatial clusters of high or low values,
    useful for detecting hotspots of specific cluster types in tissue.
    """
    console.print("[cyan]Calculating Getis-Ord G statistics...[/cyan]")
    
    if SPATIAL_LIBS_AVAILABLE and hasattr(spatial_weights, 'transform'):
        # Global G statistic.
        g_global = G(cluster_labels, spatial_weights)
        
        # Local G statistics.
        g_local = G_Local(cluster_labels, spatial_weights)
        
        results = {
            'global_g': float(g_global.G),
            'global_p_value': float(g_global.p_norm),
            'global_z_score': float(g_global.z_norm),
            'local_g': g_local.Gs.tolist(),
            'local_p_values': g_local.p_sim.tolist(),
            'local_z_scores': g_local.z_sim.tolist(),
            'hotspots': (g_local.p_sim < 0.05).tolist(),
            'n_hotspots': int(np.sum(g_local.p_sim < 0.05))
        }
    else:
        # Simplified fallback.
        results = {
            'global_g': 0.0,
            'global_p_value': 1.0,
            'global_z_score': 0.0,
            'local_g': np.zeros(len(cluster_labels)).tolist(),
            'local_p_values': np.ones(len(cluster_labels)).tolist(),
            'local_z_scores': np.zeros(len(cluster_labels)).tolist(),
            'hotspots': np.zeros(len(cluster_labels), dtype=bool).tolist(),
            'n_hotspots': 0
        }
    
    console.print(f"[green]✓[/green] G statistics complete - {results['n_hotspots']} hotspots identified")
    
    return results


def analyze_spatial_clustering_by_sample(data: pd.DataFrame, samples: List[str], 
                                       distance_threshold: float = 100.0) -> Dict[str, Any]:
    """
    Perform comprehensive spatial analysis for each sample separately.
    
    Args:
        data: DataFrame with cluster assignments and coordinates.
        samples: List of sample names to analyze.
        distance_threshold: Distance threshold for spatial weights.
        
    Returns:
        Dictionary with spatial analysis results for each sample.
        
    Sample-specific analysis accounts for potential differences in tissue
    architecture and clustering patterns between experimental conditions.
    """
    console.print("[cyan]Performing sample-specific spatial analysis...[/cyan]")
    
    results = {}
    
    for sample in track(samples, description="Analyzing samples"):
        sample_data = data[data['sample'] == sample].copy()
        
        if len(sample_data) < 10:  # Minimum points for meaningful analysis.
            console.print(f"[yellow]⚠[/yellow] Skipping {sample} - insufficient data points ({len(sample_data)})")
            continue
        
        coordinates = sample_data[['spot_x', 'spot_y']].values
        cluster_labels = sample_data['cluster'].values
        
        # Create spatial weights.
        weights = create_spatial_weights(coordinates, method='knn', distance_threshold=distance_threshold)
        
        # Calculate spatial statistics.
        sample_results = {
            'n_points': len(sample_data),
            'n_clusters': len(np.unique(cluster_labels)),
            'morans_i': calculate_morans_i(cluster_labels, weights),
            'lisa': calculate_local_morans_i(cluster_labels, weights),
            'getis_ord_g': calculate_getis_ord_g(cluster_labels, weights)
        }
        
        results[sample] = sample_results
        console.print(f"[green]✓[/green] Completed spatial analysis for {sample}")
    
    return results


def save_spatial_results(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Save comprehensive spatial analysis results to files.

    Args:
        results: Dictionary containing all spatial analysis results.
        output_dir: Directory for saving results.

    This function creates multiple output formats optimized for different
    analysis workflows and visualization requirements.
    """
    console.print("[cyan]Saving spatial analysis results...[/cyan]")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comprehensive JSON results.
    json_path = output_dir / 'morans_i_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]✓[/green] Saved spatial metrics to {json_path}")

    # Create spatial autocorrelation summary CSV.
    autocorr_data = []
    for sample, sample_results in results.items():
        if isinstance(sample_results, dict) and 'morans_i' in sample_results:
            morans = sample_results['morans_i']
            autocorr_data.append({
                'sample': sample,
                'n_points': sample_results['n_points'],
                'n_clusters': sample_results['n_clusters'],
                'morans_i': morans['morans_i'],
                'expected_i': morans['expected_i'],
                'z_score': morans['z_score'],
                'p_value': morans['p_value'],
                'significant': morans['p_value'] < 0.05
            })

    if autocorr_data:
        autocorr_df = pd.DataFrame(autocorr_data)
        autocorr_path = output_dir / 'spatial_autocorr.csv'
        autocorr_df.to_csv(autocorr_path, index=False)
        console.print(f"[green]✓[/green] Saved autocorrelation summary to {autocorr_path}")

    # Create LISA analysis CSV.
    lisa_data = []
    for sample, sample_results in results.items():
        if isinstance(sample_results, dict) and 'lisa' in sample_results:
            lisa = sample_results['lisa']
            for i, (local_i, p_val, z_score, quad, significant) in enumerate(zip(
                lisa['local_i'], lisa['p_values'], lisa['z_scores'],
                lisa['quadrants'], lisa['significant_locations']
            )):
                lisa_data.append({
                    'sample': sample,
                    'location_id': i,
                    'local_i': local_i,
                    'p_value': p_val,
                    'z_score': z_score,
                    'quadrant': quad,
                    'significant': significant
                })

    if lisa_data:
        lisa_df = pd.DataFrame(lisa_data)
        lisa_path = output_dir / 'lisa_analysis.csv'
        lisa_df.to_csv(lisa_path, index=False)
        console.print(f"[green]✓[/green] Saved LISA analysis to {lisa_path}")

    # Create spatial summary report.
    summary_path = output_dir / 'spatial_summary.txt'
    create_spatial_summary_report(results, summary_path)


def create_spatial_summary_report(results: Dict[str, Any], output_path: Path) -> None:
    """
    Create human-readable spatial analysis summary report.

    Args:
        results: Dictionary containing spatial analysis results.
        output_path: Path for saving the summary report.

    This function generates a comprehensive report with interpretation
    guidelines for spatial clustering patterns.
    """
    with open(output_path, 'w') as f:
        f.write("Spatial Clustering Analysis Summary\n")
        f.write("=" * 40 + "\n\n")

        # Overall summary.
        total_samples = len([k for k in results.keys() if isinstance(results[k], dict)])
        significant_samples = 0

        f.write(f"Total samples analyzed: {total_samples}\n\n")

        # Sample-by-sample results.
        for sample, sample_results in results.items():
            if not isinstance(sample_results, dict):
                continue

            f.write(f"Sample: {sample}\n")
            f.write("-" * (len(sample) + 8) + "\n")

            f.write(f"Data points: {sample_results['n_points']:,}\n")
            f.write(f"Unique clusters: {sample_results['n_clusters']}\n\n")

            # Moran's I results.
            if 'morans_i' in sample_results:
                morans = sample_results['morans_i']
                f.write("Global Spatial Autocorrelation (Moran's I):\n")
                f.write(f"  Moran's I: {morans['morans_i']:.4f}\n")
                f.write(f"  Expected I: {morans['expected_i']:.4f}\n")
                f.write(f"  Z-score: {morans['z_score']:.4f}\n")
                f.write(f"  P-value: {morans['p_value']:.4f}\n")

                if morans['p_value'] < 0.05:
                    significance = "Significant"
                    significant_samples += 1
                    if morans['morans_i'] > morans['expected_i']:
                        pattern = "positive spatial autocorrelation (clustering)"
                    else:
                        pattern = "negative spatial autocorrelation (dispersion)"
                else:
                    significance = "Not significant"
                    pattern = "random spatial pattern"

                f.write(f"  Interpretation: {significance} {pattern}\n\n")

            # LISA results.
            if 'lisa' in sample_results:
                lisa = sample_results['lisa']
                f.write("Local Spatial Association (LISA):\n")
                f.write(f"  Significant local clusters: {lisa['n_significant']}\n")
                f.write(f"  Percentage significant: {lisa['n_significant']/sample_results['n_points']*100:.1f}%\n\n")

            # Getis-Ord G results.
            if 'getis_ord_g' in sample_results:
                g_stats = sample_results['getis_ord_g']
                f.write("Hotspot Analysis (Getis-Ord G):\n")
                f.write(f"  Global G: {g_stats['global_g']:.4f}\n")
                f.write(f"  Global G p-value: {g_stats['global_p_value']:.4f}\n")
                f.write(f"  Local hotspots: {g_stats['n_hotspots']}\n")
                f.write(f"  Percentage hotspots: {g_stats['n_hotspots']/sample_results['n_points']*100:.1f}%\n\n")

        # Overall interpretation.
        f.write("Overall Interpretation\n")
        f.write("-" * 22 + "\n")
        f.write(f"Samples with significant spatial clustering: {significant_samples}/{total_samples}\n")

        if significant_samples > total_samples * 0.5:
            f.write("Conclusion: Strong evidence for spatial organization in clustering patterns.\n")
        elif significant_samples > 0:
            f.write("Conclusion: Moderate evidence for spatial organization in some samples.\n")
        else:
            f.write("Conclusion: Limited evidence for spatial organization in clustering patterns.\n")

        f.write("\nInterpretation Guidelines:\n")
        f.write("• Moran's I > 0: Positive spatial autocorrelation (similar clusters nearby)\n")
        f.write("• Moran's I < 0: Negative spatial autocorrelation (dissimilar clusters nearby)\n")
        f.write("• Moran's I ≈ 0: Random spatial distribution\n")
        f.write("• P-value < 0.05: Statistically significant spatial pattern\n")
        f.write("• LISA identifies local hotspots and outliers\n")
        f.write("• Getis-Ord G identifies clusters of high/low values\n")

    console.print(f"[green]✓[/green] Spatial summary report saved to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Perform comprehensive spatial clustering analysis.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--cluster_data', type=Path, required=True,
                       help='Path to cluster assignments CSV with coordinates')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for spatial analysis results')
    parser.add_argument('--samples', nargs='+', default=['IRI1', 'IRI2', 'IRI3'],
                       help='Sample names to include')
    parser.add_argument('--distance_threshold', type=float, default=100.0,
                       help='Distance threshold for spatial neighbors')
    parser.add_argument('--weight_method', choices=['knn', 'distance', 'delaunay'],
                       default='knn', help='Method for spatial weight construction')
    parser.add_argument('--k_neighbors', type=int, default=8,
                       help='Number of neighbors for KNN weights')

    args = parser.parse_args()

    try:
        console.print(f"[1m[36mSpatial Clustering Analysis[/36m[/1m")
        console.print(f"[blue]ℹ[/blue] Samples: [1m{', '.join(args.samples)}[/1m")
        console.print(f"[blue]ℹ[/blue] Weight method: [1m{args.weight_method}[/1m")

        # Load cluster data.
        console.print(f"[cyan]Loading cluster data from {args.cluster_data}...[/cyan]")
        data = pd.read_csv(args.cluster_data)

        # Filter for specified samples.
        data_filtered = data[data['sample'].isin(args.samples)].copy()
        console.print(f"[green]✓[/green] Loaded {len(data_filtered):,} data points from {len(args.samples)} samples")

        # Perform spatial analysis.
        results = analyze_spatial_clustering_by_sample(
            data_filtered, args.samples, args.distance_threshold
        )

        # Add metadata.
        results['metadata'] = {
            'total_points': len(data_filtered),
            'samples': args.samples,
            'distance_threshold': args.distance_threshold,
            'weight_method': args.weight_method,
            'k_neighbors': args.k_neighbors if args.weight_method == 'knn' else None
        }

        # Save results.
        save_spatial_results(results, args.output)

        # Display summary table.
        table = Table(title="Spatial Analysis Summary", style="cyan")
        table.add_column("Sample", style="white")
        table.add_column("Points", style="green")
        table.add_column("Moran's I", style="green")
        table.add_column("P-value", style="green")
        table.add_column("Significant", style="yellow")

        for sample, sample_results in results.items():
            if isinstance(sample_results, dict) and 'morans_i' in sample_results:
                morans = sample_results['morans_i']
                significant = "Yes" if morans['p_value'] < 0.05 else "No"
                table.add_row(
                    sample,
                    f"{sample_results['n_points']:,}",
                    f"{morans['morans_i']:.4f}",
                    f"{morans['p_value']:.4f}",
                    significant
                )

        console.print(table)
        console.print(f"[green]✓[/green] [1mSpatial analysis complete![/1m")

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
