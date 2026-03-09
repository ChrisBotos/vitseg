"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: cluster_spots_by_nuclei_features.py.
Description:
    Attaches nuclei to closest spots from metadata, aggregates ViT features per spot,
    and performs clustering based on nuclei features rather than figure_idents.
    
    This approach enables spatial transcriptomics-style analysis where spots represent
    tissue locations and clustering is based on the morphological features of nuclei
    within each spot's vicinity, providing insights into tissue heterogeneity patterns.

Dependencies:
    • Python >= 3.10.
    • pandas, numpy, scikit-learn, matplotlib, seaborn.
    • rich (for enhanced console output).
    • scipy (for spatial distance calculations).

Usage:
    python code/cluster_spots_by_nuclei_features.py \
    --nuclei_coords results/IRI_regist_14k/coords_IRI_regist_binary_mask.csv \
    --nuclei_features results/IRI_regist_14k/features_IRI_regist_binary_mask.csv \
    --spots_metadata data/metadata_complete.csv \
    --output results/spot_nuclei_clustering_14k \
    --samples IRI1 IRI2 IRI3 \
    --clusters 15 \
    --max_distance 1000 \
    --aggregation_method mean

Arguments:
    --nuclei_coords        Path to nuclei coordinates CSV file.
    --nuclei_features      Path to nuclei ViT features CSV file.
    --spots_metadata       Path to spots metadata CSV file.
    --output               Output directory for results.
    --samples              Sample names to include (e.g., IRI1 IRI2 IRI3).
    --clusters             Number of clusters for K-means.
    --min_nuclei_per_spot  Minimum nuclei required per spot to include in analysis.
    --aggregation_method   Method to aggregate nuclei features per spot (mean, median, max).
    --max_distance         Maximum distance to attach nucleus to spot (pixels).

Inputs:
    • coords_*.csv         Nuclei center coordinates (x_center, y_center).
    • features_*.csv       ViT features for each nucleus.
    • metadata_complete.csv Spot locations and annotations.

Outputs:
    • spot_nuclei_assignments.csv    Nucleus-to-spot assignments.
    • spot_aggregated_features.csv   Aggregated features per spot.
    • spot_clusters.csv              Cluster assignments per spot.
    • spot_cluster_visualization.png Visualization of clustered spots.
    • spot_cluster_stats.txt         Statistics and analysis summary.

Key Features:
    • Spatial assignment of nuclei to nearest spots using Euclidean distance.
    • Robust handling of spots with varying nuclei counts (0 to many).
    • Multiple aggregation strategies for combining nuclei features per spot.
    • Publication-quality visualizations with scientific color palettes.
    • Comprehensive statistics and quality control metrics.

Notes:
    • Spots without nuclei within max_distance are excluded from clustering.
    • Features are standardized before clustering for optimal performance.
    • Clustering results can be compared with figure_idents for validation.
"""
import argparse
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table

console = Console()

# Try to import enhanced color generation system.
try:
    from generate_contrast_colors import generate_color_palette, colors_to_hex_list
    ENHANCED_COLORS_AVAILABLE = True
    console.print("[green]OK[/green] Using enhanced color generation system")
except ImportError:
    ENHANCED_COLORS_AVAILABLE = False
    console.print("[yellow]WARNING[/yellow] Enhanced colors not available, using fallback")


def generate_high_contrast_colors(n_colors: int) -> List[str]:
    """
    Generate high-contrast colors optimized for scientific visualization.
    
    Args:
        n_colors: Number of distinct colors needed.
        
    Returns:
        List of hex color codes.
        
    This function creates visually distinct colors suitable for publication-quality
    figures with good contrast on both light and dark backgrounds.
    """
    if ENHANCED_COLORS_AVAILABLE:
        color_palette = generate_color_palette(
            n=n_colors,
            background="dark",
            contrast_ratio=4.5,
            saturation=0.95
        )
        return colors_to_hex_list(color_palette)
    else:
        # Fallback to seaborn palette.
        colors = sns.color_palette("husl", n_colors)
        return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                for r, g, b in colors]


def load_data(nuclei_coords_path: Path, nuclei_features_path: Path, 
              spots_metadata_path: Path, samples: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load nuclei coordinates, features, and spot metadata.
    
    Args:
        nuclei_coords_path: Path to nuclei coordinates CSV.
        nuclei_features_path: Path to nuclei features CSV.
        spots_metadata_path: Path to spots metadata CSV.
        samples: List of sample names to filter.
        
    Returns:
        Tuple of (nuclei_coords, nuclei_features, spots_metadata) DataFrames.
        
    This function loads and validates all required data files, ensuring
    consistency between nuclei coordinates and features.
    """
    console.print(f"[cyan]Loading nuclei coordinates from {nuclei_coords_path}...[/cyan]")
    nuclei_coords = pd.read_csv(nuclei_coords_path)
    console.print(f"[green]OK[/green] Loaded {len(nuclei_coords):,} nuclei coordinates")
    
    console.print(f"[cyan]Loading nuclei features from {nuclei_features_path}...[/cyan]")
    nuclei_features = pd.read_csv(nuclei_features_path)
    console.print(f"[green]OK[/green] Loaded {len(nuclei_features):,} nuclei with {nuclei_features.shape[1]} features")
    
    console.print(f"[cyan]Loading spots metadata from {spots_metadata_path}...[/cyan]")
    spots_metadata = pd.read_csv(spots_metadata_path)
    
    # Filter for specified samples.
    spots_filtered = spots_metadata[spots_metadata['sample'].isin(samples)].copy()
    console.print(f"[green]OK[/green] Loaded {len(spots_filtered):,} spots from {len(samples)} samples")
    
    # Validate data consistency.
    if len(nuclei_coords) != len(nuclei_features):
        raise ValueError(f"Mismatch: {len(nuclei_coords)} coordinates vs {len(nuclei_features)} features")
    
    console.print(f"[blue]INFO[/blue] Samples included: {sorted(spots_filtered['sample'].unique())}")
    console.print(f"[blue]INFO[/blue] Feature dimensions: {nuclei_features.shape[1]}")
    
    return nuclei_coords, nuclei_features, spots_filtered


def create_synthetic_spots(nuclei_coords: pd.DataFrame, grid_size: int = 50,
                          samples: List[str] = None) -> pd.DataFrame:
    """
    Create synthetic spots based on nuclei locations using a regular grid.

    Args:
        nuclei_coords: DataFrame with x_center, y_center columns.
        grid_size: Size of grid cells for creating spots.
        samples: List of sample names to assign to spots.

    Returns:
        DataFrame with synthetic spot locations.

    This function creates a regular grid of spots covering the nuclei coordinate
    space, useful when the original spots and nuclei are in different coordinate systems.
    """
    console.print(f"[cyan]Creating synthetic spots with grid size {grid_size}...[/cyan]")

    # Get coordinate bounds.
    x_min, x_max = nuclei_coords['x_center'].min(), nuclei_coords['x_center'].max()
    y_min, y_max = nuclei_coords['y_center'].min(), nuclei_coords['y_center'].max()

    # Create grid points.
    x_coords = np.arange(x_min, x_max + grid_size, grid_size)
    y_coords = np.arange(y_min, y_max + grid_size, grid_size)

    # Create all combinations.
    xx, yy = np.meshgrid(x_coords, y_coords)
    spot_positions = np.column_stack([xx.ravel(), yy.ravel()])

    # Create synthetic spots DataFrame.
    n_spots = len(spot_positions)
    synthetic_spots = pd.DataFrame({
        'x': spot_positions[:, 0],
        'y': spot_positions[:, 1],
        'sample': np.random.choice(samples or ['synthetic'], n_spots),
        'condition': ['synthetic'] * n_spots,
        'figure_idents': ['synthetic_spot'] * n_spots
    })

    console.print(f"[green]OK[/green] Created {len(synthetic_spots):,} synthetic spots")
    console.print(f"[blue]INFO[/blue] Grid covers X({x_min:.0f}-{x_max:.0f}), Y({y_min:.0f}-{y_max:.0f})")

    return synthetic_spots


def assign_nuclei_to_spots(nuclei_coords: pd.DataFrame, spots_metadata: pd.DataFrame,
                          max_distance: float = 100.0) -> pd.DataFrame:
    """
    Assign each nucleus to the closest spot within maximum distance.

    Args:
        nuclei_coords: DataFrame with x_center, y_center columns.
        spots_metadata: DataFrame with x, y columns for spot locations.
        max_distance: Maximum distance in pixels to assign nucleus to spot.

    Returns:
        DataFrame with nucleus assignments including spot_id and distance.

    This function uses Euclidean distance to find the nearest spot for each nucleus,
    handling cases where no spot is within the maximum distance threshold.
    """
    console.print(f"[cyan]Assigning {len(nuclei_coords):,} nuclei to {len(spots_metadata):,} spots...[/cyan]")

    # Extract coordinate arrays.
    nuclei_positions = nuclei_coords[['x_center', 'y_center']].values
    spot_positions = spots_metadata[['x', 'y']].values

    # Debug coordinate ranges.
    console.print(f"[blue]INFO[/blue] Nuclei coordinate ranges: X({nuclei_positions[:, 0].min():.0f}-{nuclei_positions[:, 0].max():.0f}), Y({nuclei_positions[:, 1].min():.0f}-{nuclei_positions[:, 1].max():.0f})")
    console.print(f"[blue]INFO[/blue] Spot coordinate ranges: X({spot_positions[:, 0].min():.0f}-{spot_positions[:, 0].max():.0f}), Y({spot_positions[:, 1].min():.0f}-{spot_positions[:, 1].max():.0f})")

    # Calculate pairwise distances.
    console.print("[yellow]WARNING[/yellow] Computing distance matrix...")
    distances = cdist(nuclei_positions, spot_positions, metric='euclidean')

    # Find closest spot for each nucleus.
    closest_spot_indices = np.argmin(distances, axis=1)
    closest_distances = np.min(distances, axis=1)

    # Debug distance statistics.
    console.print(f"[blue]INFO[/blue] Distance statistics: Min={closest_distances.min():.1f}, Max={closest_distances.max():.1f}, Mean={closest_distances.mean():.1f}")

    # Create assignment DataFrame.
    assignments = pd.DataFrame({
        'nucleus_id': range(len(nuclei_coords)),
        'x_center': nuclei_coords['x_center'].values,
        'y_center': nuclei_coords['y_center'].values,
        'spot_index': closest_spot_indices,
        'distance_to_spot': closest_distances
    })

    # Add spot information.
    spot_info = spots_metadata.iloc[closest_spot_indices][['x', 'y', 'sample', 'condition']].reset_index()
    spot_info.columns = ['spot_id', 'spot_x', 'spot_y', 'sample', 'condition']
    assignments = pd.concat([assignments, spot_info], axis=1)

    # Filter by maximum distance.
    valid_assignments = assignments[assignments['distance_to_spot'] <= max_distance].copy()

    console.print(f"[green]OK[/green] Assigned {len(valid_assignments):,} nuclei to spots")
    console.print(f"[yellow]WARNING[/yellow] Excluded {len(assignments) - len(valid_assignments):,} nuclei (distance > {max_distance})")

    # If no assignments, suggest a better max_distance.
    if len(valid_assignments) == 0:
        suggested_distance = np.percentile(closest_distances, 10)  # 10th percentile.
        console.print(f"[red]ERROR[/red] No nuclei assigned! Consider increasing max_distance to at least {suggested_distance:.0f}")

        # Check for coordinate system mismatch.
        nuclei_range = (nuclei_positions[:, 0].max() - nuclei_positions[:, 0].min() +
                       nuclei_positions[:, 1].max() - nuclei_positions[:, 1].min())
        spot_range = (spot_positions[:, 0].max() - spot_positions[:, 0].min() +
                     spot_positions[:, 1].max() - spot_positions[:, 1].min())

        if spot_range > nuclei_range * 10:  # Spots have much larger coordinate range.
            console.print("[yellow]WARNING[/yellow] Detected coordinate system mismatch (spots >> nuclei)")
            console.print("[yellow]WARNING[/yellow] Consider using --create_synthetic_spots")

    return valid_assignments


def aggregate_features_per_spot(assignments: pd.DataFrame, nuclei_features: pd.DataFrame,
                               aggregation_method: str = 'mean', min_nuclei_per_spot: int = 1) -> pd.DataFrame:
    """
    Aggregate nuclei features for each spot using specified method.
    
    Args:
        assignments: DataFrame with nucleus-to-spot assignments.
        nuclei_features: DataFrame with ViT features for each nucleus.
        aggregation_method: Method to aggregate features ('mean', 'median', 'max').
        min_nuclei_per_spot: Minimum nuclei required per spot.
        
    Returns:
        DataFrame with aggregated features per spot.
        
    This function handles spots with varying numbers of nuclei by applying
    robust aggregation methods and filtering spots with insufficient data.
    """
    console.print(f"[cyan]Aggregating features using {aggregation_method} method...[/cyan]")
    
    # Group assignments by spot.
    spot_groups = assignments.groupby('spot_id')
    
    aggregated_spots = []
    feature_columns = nuclei_features.columns
    
    for spot_id, group in track(spot_groups, description="Processing spots"):
        if len(group) < min_nuclei_per_spot:
            continue
            
        # Get features for nuclei in this spot.
        nucleus_indices = group['nucleus_id'].values
        spot_features = nuclei_features.iloc[nucleus_indices]
        
        # Aggregate features.
        if aggregation_method == 'mean':
            aggregated = spot_features.mean()
        elif aggregation_method == 'median':
            aggregated = spot_features.median()
        elif aggregation_method == 'max':
            aggregated = spot_features.max()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Add spot metadata.
        spot_info = group.iloc[0]
        spot_data = {
            'spot_id': spot_id,
            'spot_x': spot_info['spot_x'],
            'spot_y': spot_info['spot_y'],
            'sample': spot_info['sample'],
            'condition': spot_info['condition'],
            'n_nuclei': len(group),
            'mean_distance': group['distance_to_spot'].mean()
        }
        
        # Combine with aggregated features.
        for col in feature_columns:
            spot_data[f'agg_{col}'] = aggregated[col]
        
        aggregated_spots.append(spot_data)
    
    result_df = pd.DataFrame(aggregated_spots)
    console.print(f"[green]OK[/green] Created {len(result_df):,} spots with aggregated features")

    # Show nuclei distribution statistics if we have spots.
    if len(result_df) > 0:
        nuclei_stats = result_df['n_nuclei'].describe()
        console.print(f"[blue]INFO[/blue] Nuclei per spot - Mean: {nuclei_stats['mean']:.1f}, "
                     f"Median: {nuclei_stats['50%']:.1f}, Max: {int(nuclei_stats['max'])}")
    else:
        console.print("[red]ERROR[/red] No spots created! Check nucleus-to-spot assignments and min_nuclei_per_spot threshold.")

    return result_df


def cluster_spots(spot_features: pd.DataFrame, n_clusters: int, random_state: int = 42) -> Tuple[np.ndarray, KMeans, StandardScaler]:
    """
    Perform K-means clustering on aggregated spot features.
    
    Args:
        spot_features: DataFrame with aggregated features per spot.
        n_clusters: Number of clusters for K-means.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (cluster_labels, kmeans_model, scaler).
        
    This function standardizes features before clustering and evaluates
    cluster quality using silhouette score.
    """
    console.print(f"[cyan]Clustering {len(spot_features):,} spots into {n_clusters} clusters...[/cyan]")
    
    # Extract feature columns (those starting with 'agg_').
    feature_cols = [col for col in spot_features.columns if col.startswith('agg_')]
    features = spot_features[feature_cols].values
    
    console.print(f"[blue]INFO[/blue] Using {len(feature_cols)} aggregated features")
    
    # Standardize features.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform clustering.
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Evaluate clustering quality.
    if len(np.unique(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        console.print(f"[green]OK[/green] Clustering complete - Silhouette score: {silhouette_avg:.3f}")
    else:
        console.print("[yellow]WARNING[/yellow] Only one cluster found")
    
    return cluster_labels, kmeans, scaler


def create_visualization(spot_features: pd.DataFrame, output_path: Path) -> Dict[str, Any]:
    """
    Create visualization of clustered spots with colored circles.

    Args:
        spot_features: DataFrame with spot coordinates and cluster assignments.
        output_path: Path for saving the visualization.

    Returns:
        Dictionary with visualization statistics.

    This function creates a publication-quality visualization showing spots
    colored by cluster assignment, with separate subplots for each sample.
    """
    console.print("[cyan]Creating cluster visualization...[/cyan]")

    # Get unique clusters and generate colors.
    unique_clusters = sorted(spot_features['cluster'].unique())
    colors = generate_high_contrast_colors(len(unique_clusters))
    color_map = dict(zip(unique_clusters, colors))

    # Get unique samples for subplots.
    samples = sorted(spot_features['sample'].unique())
    n_samples = len(samples)

    # Create figure with subplots.
    fig, axes = plt.subplots(1, n_samples, figsize=(6*n_samples, 6), facecolor='black')
    if n_samples == 1:
        axes = [axes]

    fig.suptitle('Spot Clustering Based on Nuclei ViT Features',
                 fontsize=16, color='white', y=0.95)

    stats = {'samples': {}, 'total_spots': 0, 'cluster_counts': {}}

    for idx, sample in enumerate(samples):
        ax = axes[idx]
        sample_df = spot_features[spot_features['sample'] == sample]

        console.print(f"[yellow]WARNING[/yellow] Plotting {sample}: {len(sample_df):,} spots")

        # Set black background.
        ax.set_facecolor('black')

        # Plot spots by cluster.
        for cluster_id in track(unique_clusters, description=f"Plotting {sample}"):
            cluster_data = sample_df[sample_df['cluster'] == cluster_id]

            if len(cluster_data) > 0:
                color = color_map[cluster_id]
                ax.scatter(cluster_data['spot_x'], cluster_data['spot_y'],
                          c=color, s=50, alpha=0.8, edgecolors='white', linewidth=0.5,
                          label=f'Cluster {cluster_id}')

                # Update statistics.
                if cluster_id not in stats['cluster_counts']:
                    stats['cluster_counts'][cluster_id] = 0
                stats['cluster_counts'][cluster_id] += len(cluster_data)

        # Configure subplot.
        ax.set_title(f'{sample}', fontsize=14, color='white', pad=10)
        ax.set_xlabel('X Coordinate', fontsize=12, color='white')
        ax.set_ylabel('Y Coordinate', fontsize=12, color='white')

        # Set axis limits with padding.
        x_min, x_max = sample_df['spot_x'].min(), sample_df['spot_x'].max()
        y_min, y_max = sample_df['spot_y'].min(), sample_df['spot_y'].max()

        x_range = x_max - x_min
        y_range = y_max - y_min
        x_padding = x_range * 0.05
        y_padding = y_range * 0.05

        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Set equal aspect ratio.
        ax.set_aspect('equal', adjustable='box')

        # Style axes.
        ax.tick_params(colors='white', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('white')

        # Add legend for first subplot only.
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     facecolor='black', edgecolor='white', labelcolor='white')

        # Store sample statistics.
        stats['samples'][sample] = {
            'n_spots': len(sample_df),
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max)
        }
        stats['total_spots'] += len(sample_df)

    plt.tight_layout()
    plt.savefig(output_path, facecolor='black', edgecolor='none',
                dpi=300, bbox_inches='tight')
    plt.close()

    console.print(f"[green]OK[/green] Visualization saved to {output_path}")

    return stats, color_map


def save_statistics(spot_features: pd.DataFrame, stats: Dict[str, Any],
                   color_map: Dict, output_path: Path) -> None:
    """
    Save comprehensive statistics about the clustering analysis.

    Args:
        spot_features: DataFrame with clustering results.
        stats: Statistics dictionary from visualization.
        color_map: Color mapping dictionary.
        output_path: Path for saving statistics.

    This function creates a detailed report with clustering statistics,
    nuclei distribution analysis, and quality metrics.
    """
    console.print("[cyan]Saving comprehensive statistics...[/cyan]")

    with open(output_path, 'w') as f:
        f.write("Spot-Nuclei Clustering Analysis Statistics\n")
        f.write("=" * 45 + "\n\n")

        f.write(f"Total spots analyzed: {stats['total_spots']:,}\n")
        f.write(f"Number of samples: {len(stats['samples'])}\n")
        f.write(f"Number of clusters: {len(color_map)}\n\n")

        # Sample statistics.
        f.write("Sample Statistics:\n")
        f.write("-" * 20 + "\n")
        for sample, sample_stats in stats['samples'].items():
            f.write(f"{sample}:\n")
            f.write(f"  Spots: {sample_stats['n_spots']:,}\n")
            f.write(f"  X range: {sample_stats['x_range'][0]:.0f} - {sample_stats['x_range'][1]:.0f}\n")
            f.write(f"  Y range: {sample_stats['y_range'][0]:.0f} - {sample_stats['y_range'][1]:.0f}\n\n")

        # Cluster distribution.
        f.write("Cluster Distribution:\n")
        f.write("-" * 20 + "\n")
        for cluster_id, count in sorted(stats['cluster_counts'].items()):
            percentage = (count / stats['total_spots']) * 100
            f.write(f"Cluster {cluster_id}: {count:,} spots ({percentage:.1f}%)\n")

        # Nuclei per spot statistics.
        f.write(f"\nNuclei per Spot Statistics:\n")
        f.write("-" * 30 + "\n")
        nuclei_stats = spot_features['n_nuclei'].describe()
        f.write(f"Mean nuclei per spot: {nuclei_stats['mean']:.2f}\n")
        f.write(f"Median nuclei per spot: {nuclei_stats['50%']:.2f}\n")
        f.write(f"Min nuclei per spot: {int(nuclei_stats['min'])}\n")
        f.write(f"Max nuclei per spot: {int(nuclei_stats['max'])}\n")
        f.write(f"Std deviation: {nuclei_stats['std']:.2f}\n")

        # Distance statistics.
        f.write(f"\nDistance Statistics:\n")
        f.write("-" * 20 + "\n")
        distance_stats = spot_features['mean_distance'].describe()
        f.write(f"Mean distance to spot: {distance_stats['mean']:.2f} pixels\n")
        f.write(f"Median distance to spot: {distance_stats['50%']:.2f} pixels\n")
        f.write(f"Max distance to spot: {distance_stats['max']:.2f} pixels\n")

        # Color mapping.
        f.write(f"\nColor Mapping:\n")
        f.write("-" * 15 + "\n")
        for cluster_id, color in color_map.items():
            f.write(f"Cluster {cluster_id}: {color}\n")

    console.print(f"[green]OK[/green] Statistics saved to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Cluster spots based on aggregated nuclei ViT features.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nuclei_coords', type=Path, required=True,
                       help='Path to nuclei coordinates CSV file')
    parser.add_argument('--nuclei_features', type=Path, required=True,
                       help='Path to nuclei ViT features CSV file')
    parser.add_argument('--spots_metadata', type=Path, required=True,
                       help='Path to spots metadata CSV file')
    parser.add_argument('--output', type=Path, default=Path('results/spot_nuclei_clustering'),
                       help='Output directory for results')
    parser.add_argument('--samples', nargs='+', default=['IRI1', 'IRI2', 'IRI3'],
                       help='Sample names to include')
    parser.add_argument('--clusters', type=int, default=15,
                       help='Number of clusters for K-means')
    parser.add_argument('--min_nuclei_per_spot', type=int, default=1,
                       help='Minimum nuclei required per spot')
    parser.add_argument('--aggregation_method', choices=['mean', 'median', 'max'],
                       default='mean', help='Method to aggregate nuclei features per spot')
    parser.add_argument('--max_distance', type=float, default=100.0,
                       help='Maximum distance to attach nucleus to spot (pixels)')
    parser.add_argument('--create_synthetic_spots', action='store_true',
                       help='Create synthetic spots based on nuclei locations when coordinate systems mismatch')
    parser.add_argument('--spot_grid_size', type=int, default=50,
                       help='Grid size for synthetic spots (pixels)')

    args = parser.parse_args()

    try:
        # Create output directory.
        args.output.mkdir(parents=True, exist_ok=True)

        console.print(f"[1m[36mSpot-Nuclei Clustering Analysis[/36m[/1m")
        console.print(f"[blue]INFO[/blue] Samples: [1m{', '.join(args.samples)}[/1m")
        console.print(f"[blue]INFO[/blue] Clusters: [1m{args.clusters}[/1m")
        console.print(f"[blue]INFO[/blue] Aggregation: [1m{args.aggregation_method}[/1m")

        # Load data.
        nuclei_coords, nuclei_features, spots_metadata = load_data(
            args.nuclei_coords, args.nuclei_features, args.spots_metadata, args.samples
        )

        # Check if we should create synthetic spots.
        if args.create_synthetic_spots:
            console.print("[yellow]WARNING[/yellow] Using synthetic spots based on nuclei locations")
            spots_metadata = create_synthetic_spots(nuclei_coords, args.spot_grid_size, args.samples)

        # Assign nuclei to spots.
        assignments = assign_nuclei_to_spots(
            nuclei_coords, spots_metadata, args.max_distance
        )

        # If no assignments and not using synthetic spots, suggest using them.
        if len(assignments) == 0 and not args.create_synthetic_spots:
            console.print("[yellow]WARNING[/yellow] Consider using --create_synthetic_spots to handle coordinate system mismatch")
            return 1

        # Save assignments.
        assignments_path = args.output / 'spot_nuclei_assignments.csv'
        assignments.to_csv(assignments_path, index=False)
        console.print(f"[green]OK[/green] Saved assignments to {assignments_path}")

        # Aggregate features per spot.
        spot_features = aggregate_features_per_spot(
            assignments, nuclei_features, args.aggregation_method, args.min_nuclei_per_spot
        )

        # Check if we have any spots to work with.
        if len(spot_features) == 0:
            console.print("[red]ERROR[/red] No spots with sufficient nuclei found. Try:")
            console.print("  • Increasing --max_distance")
            console.print("  • Decreasing --min_nuclei_per_spot")
            console.print("  • Checking coordinate system alignment")
            return 1

        # Save aggregated features.
        features_path = args.output / 'spot_aggregated_features.csv'
        spot_features.to_csv(features_path, index=False)
        console.print(f"[green]OK[/green] Saved aggregated features to {features_path}")

        # Perform clustering.
        cluster_labels, kmeans_model, scaler = cluster_spots(spot_features, args.clusters)

        # Add cluster labels to spot features.
        spot_features['cluster'] = cluster_labels

        # Save clustering results.
        clusters_path = args.output / 'spot_clusters.csv'
        spot_features.to_csv(clusters_path, index=False)
        console.print(f"[green]OK[/green] Saved clustering results to {clusters_path}")

        # Create visualization.
        viz_path = args.output / 'spot_cluster_visualization.png'
        stats, color_map = create_visualization(spot_features, viz_path)

        # Save comprehensive statistics.
        stats_path = args.output / 'spot_cluster_stats.txt'
        save_statistics(spot_features, stats, color_map, stats_path)

        # Display summary table.
        table = Table(title="Analysis Summary", style="cyan")
        table.add_column("Metric", style="white")
        table.add_column("Value", style="green")

        table.add_row("Total Spots", f"{stats['total_spots']:,}")
        table.add_row("Samples", f"{len(stats['samples'])}")
        table.add_row("Clusters", f"{len(color_map)}")
        table.add_row("Mean Nuclei/Spot", f"{spot_features['n_nuclei'].mean():.1f}")
        table.add_row("Output Files", "5 (assignments, features, clusters, viz, stats)")

        console.print(table)
        console.print(f"[green]OK[/green] [1mAnalysis complete![/1m")

    except Exception as e:
        console.print(f"[red]ERROR[/red] Error: {e}")
        console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
