"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: visualize_clusters_circles.py.
Description:
    Visualizes tissue samples as colored circles on a black background.
    Each circle represents one metadata entry with position from x,y coordinates.
    Colors are assigned based on either figure_idents or banksy clusters.

    This script is designed for bioinformatician users to visualize spatial patterns
    in kidney tissue samples, enabling analysis of cell type distributions and
    clustering patterns across different conditions (IRI, sham, etc.).

Dependencies:
    • Python >= 3.10.
    • pandas, matplotlib, numpy, seaborn.
    • rich (for enhanced console output).

Usage:
    python visualize_clusters_circles.py \
        --metadata ../data/metadata_complete.csv \
        --output ../results/cluster_visualization.png \
        --color-by figure_idents \
        --radius 5 \
        --samples sham1 sham2 sham3 \
        --figsize 80 60
"""
import argparse
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table

console = Console()

# Try to import the enhanced color generation system.
try:
    from vigseg.utilities.color_generation import generate_color_palette, colors_to_hex_list
    ENHANCED_COLORS_AVAILABLE = True
    console.print("[green]✓[/green] Using enhanced color generation system")
except ImportError:
    ENHANCED_COLORS_AVAILABLE = False
    console.print("[yellow]⚠[/yellow] Enhanced colors not available, using fallback")

def generate_high_contrast_colors(n_colors: int) -> List[str]:
    """
    Generate high-contrast colors optimized for scientific visualization.

    Args:
        n_colors: Number of distinct colors needed.

    Returns:
        List of hex color codes.

    This function creates visually distinct colors that work well on black
    backgrounds and are suitable for publication-quality figures.
    """
    if ENHANCED_COLORS_AVAILABLE:
        # Use the enhanced color generation system.
        color_palette = generate_color_palette(
            n=n_colors,
            background="dark",
            contrast_ratio=4.5,
            saturation=0.95
        )
        return colors_to_hex_list(color_palette)
    else:
        # Fallback to improved manual palette.
        if n_colors <= 12:
            # Use a curated palette optimized for scientific visualization.
            base_colors = [
                '#FF4444', '#44FF44', '#4444FF', '#FFFF44', '#FF44FF', '#44FFFF',
                '#FF8844', '#88FF44', '#4488FF', '#FF4488', '#88FF88', '#8844FF'
            ]
            return base_colors[:n_colors]
        else:
            # Use seaborn's husl palette for larger numbers.
            colors = sns.color_palette("husl", n_colors)
            return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                    for r, g, b in colors]


def load_and_filter_data(metadata_path: Path, samples: List[str]) -> pd.DataFrame:
    """
    Load metadata and filter for specified samples.

    Args:
        metadata_path: Path to metadata CSV file.
        samples: List of sample names to include.

    Returns:
        Filtered DataFrame with specified samples only.

    This function loads the complete metadata and filters for the specified
    samples, ensuring data quality and consistency.
    """
    console.print(f"[cyan]Loading metadata from {metadata_path}...[/cyan]")

    try:
        df = pd.read_csv(metadata_path)
        console.print(f"[green]✓[/green] Loaded {len(df):,} total entries")

        # Filter for specified samples.
        filtered_df = df[df['sample'].isin(samples)].copy()

        console.print(f"[green]✓[/green] Filtered to {len(filtered_df):,} entries")

        if len(filtered_df) == 0:
            available_samples = sorted(df['sample'].unique())
            console.print(f"[red]✗[/red] No data found for specified samples: {samples}")
            console.print(f"[yellow]⚠[/yellow] Available samples in metadata: {available_samples}")
            raise ValueError(f"No data found for samples: {samples}")

        console.print(f"[blue]ℹ[/blue] Samples included: {sorted(filtered_df['sample'].unique())}")

        # Show conditions found.
        conditions = sorted(filtered_df['condition'].unique())
        console.print(f"[blue]ℹ[/blue] Conditions found: {conditions}")

        return filtered_df

    except Exception as e:
        console.print(f"[red]✗[/red] Error loading metadata: {e}")
        raise


def create_cluster_visualization(
    df: pd.DataFrame,
    color_by: str,
    radius: float,
    figsize: Tuple[int, int],
    alpha: float,
    output_path: Path,
    flip_y: bool = False
) -> Dict[str, Any]:
    """
    Create the main cluster visualization with colored circles.

    Args:
        df: Filtered DataFrame with IRI samples.
        color_by: Column name to use for coloring.
        radius: Circle radius in pixels.
        figsize: Figure size as (width, height).
        alpha: Circle transparency.
        output_path: Path for saving the visualization.

    Returns:
        Dictionary with visualization statistics.

    This function creates the main visualization with separate subplots for
    each IRI sample, using high-contrast colors on a black background.
    """
    console.print(f"[cyan]Creating visualization colored by {color_by}...[/cyan]")

    # Get unique values for coloring and generate colors.
    unique_values = sorted(df[color_by].dropna().unique())
    colors = generate_high_contrast_colors(len(unique_values))
    color_map = dict(zip(unique_values, colors))

    # Get unique samples for subplots.
    samples = sorted(df['sample'].unique())
    n_samples = len(samples)

    # Create figure with subplots.
    fig, axes = plt.subplots(1, n_samples, figsize=figsize, facecolor='black')
    if n_samples == 1:
        axes = [axes]

    fig.suptitle(f'Tissue Samples - Colored by {color_by}',
                 fontsize=16, color='white', y=0.95)

    stats = {'samples': {}, 'total_points': 0, 'color_counts': {}}

    for idx, sample in enumerate(samples):
        ax = axes[idx]
        sample_df = df[df['sample'] == sample]

        console.print(f"[yellow]⚠[/yellow] Processing {sample}: {len(sample_df):,} points")

        # Set black background.
        ax.set_facecolor('black')

        # Use scatter plot for efficiency - group by cluster value.
        for cluster_value in track(unique_values,
                                 description=f"Plotting {sample}"):
            cluster_data = sample_df[sample_df[color_by] == cluster_value]

            if len(cluster_data) > 0:
                color = color_map[cluster_value]
                # Handle coordinate system - flip Y if requested.
                x_coords = cluster_data['x']
                y_coords = cluster_data['y']
                if flip_y:
                    # Flip Y coordinates if images appear transposed.
                    y_max = sample_df['y'].max()
                    y_coords = y_max - y_coords

                ax.scatter(x_coords, y_coords,
                          c=color, s=radius**2, alpha=alpha,
                          edgecolors='none', label=str(cluster_value))

                # Update statistics.
                if cluster_value not in stats['color_counts']:
                    stats['color_counts'][cluster_value] = 0
                stats['color_counts'][cluster_value] += len(cluster_data)

        # Configure subplot.
        ax.set_title(f'{sample}', fontsize=14, color='white', pad=10)
        ax.set_xlabel('X Coordinate', fontsize=12, color='white')
        ax.set_ylabel('Y Coordinate', fontsize=12, color='white')

        # Set axis limits with some padding.
        x_min, x_max = sample_df['x'].min(), sample_df['x'].max()
        y_min, y_max = sample_df['y'].min(), sample_df['y'].max()

        # Calculate padding as 5% of the range.
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_padding = x_range * 0.05
        y_padding = y_range * 0.05

        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Set equal aspect ratio to prevent distortion.
        ax.set_aspect('equal', adjustable='box')

        # Style axes.
        ax.tick_params(colors='white', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('white')

        # Store sample statistics.
        stats['samples'][sample] = {
            'n_points': len(sample_df),
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max)
        }
        stats['total_points'] += len(sample_df)

    plt.tight_layout()

    # Save the main visualization.
    console.print(f"[cyan]Saving visualization to {output_path}...[/cyan]")
    plt.savefig(output_path, facecolor='black', edgecolor='none',
                dpi=300, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓[/green] Visualization saved successfully")

    return stats, color_map


def create_legend(color_map: Dict, output_path: Path) -> None:
    """
    Create a separate legend figure for the color mapping.
    
    Args:
        color_map: Dictionary mapping cluster values to colors.
        output_path: Path for saving the legend.
        
    This function creates a clean, publication-ready legend that can be
    used alongside the main visualization.
    """
    console.print("[cyan]Creating legend...[/cyan]")
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(color_map) * 0.4)), 
                          facecolor='black')
    ax.set_facecolor('black')
    
    # Create legend entries.
    legend_elements = []
    for value, color in color_map.items():
        legend_elements.append(patches.Patch(color=color, label=str(value)))
    
    ax.legend(handles=legend_elements, loc='center', fontsize=12,
              facecolor='black', edgecolor='white', labelcolor='white',
              ncol=min(3, len(color_map)))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor='black', edgecolor='none',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓[/green] Legend saved to {output_path}")


def save_statistics(stats: Dict[str, Any], color_map: Dict, output_path: Path) -> None:
    """
    Save comprehensive statistics about the visualization.
    
    Args:
        stats: Statistics dictionary from visualization creation.
        color_map: Color mapping dictionary.
        output_path: Path for saving statistics.
        
    This function creates a detailed text report with statistics about
    the data and visualization for documentation purposes.
    """
    console.print("[cyan]Saving statistics...[/cyan]")
    
    with open(output_path, 'w') as f:
        f.write("Cluster Visualization Statistics\n")
        f.write("=" * 35 + "\n\n")
        
        f.write(f"Total data points: {stats['total_points']:,}\n")
        f.write(f"Number of samples: {len(stats['samples'])}\n")
        f.write(f"Number of unique clusters: {len(color_map)}\n\n")
        
        f.write("Sample Statistics:\n")
        f.write("-" * 20 + "\n")
        for sample, sample_stats in stats['samples'].items():
            f.write(f"{sample}:\n")
            f.write(f"  Points: {sample_stats['n_points']:,}\n")
            f.write(f"  X range: {sample_stats['x_range'][0]:.0f} - {sample_stats['x_range'][1]:.0f}\n")
            f.write(f"  Y range: {sample_stats['y_range'][0]:.0f} - {sample_stats['y_range'][1]:.0f}\n\n")
        
        f.write("Cluster Counts:\n")
        f.write("-" * 15 + "\n")
        for cluster, count in sorted(stats['color_counts'].items()):
            percentage = (count / stats['total_points']) * 100
            f.write(f"{cluster}: {count:,} ({percentage:.1f}%)\n")
        
        f.write(f"\nColor Mapping:\n")
        f.write("-" * 15 + "\n")
        for value, color in color_map.items():
            f.write(f"{value}: {color}\n")
    
    console.print(f"[green]✓[/green] Statistics saved to {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize IRI samples as colored circles on black background.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--metadata', type=Path, required=True,
                       help='Path to metadata CSV file')
    parser.add_argument('--output', type=Path, 
                       default=Path('results/cluster_visualization.png'),
                       help='Output path for visualization')
    parser.add_argument('--color-by', choices=['figure_idents', 'banksy'],
                       default='figure_idents',
                       help='Column to use for coloring')
    parser.add_argument('--radius', type=float, default=5.0,
                       help='Circle radius in pixels')
    parser.add_argument('--samples', nargs='+', 
                       default=['IRI1', 'IRI2', 'IRI3'],
                       help='Sample names to include')
    parser.add_argument('--figsize', nargs=2, type=int, default=[20, 15],
                       help='Figure size as width height')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for output image')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Circle transparency (0-1)')
    parser.add_argument('--flip-y', action='store_true',
                       help='Flip Y coordinates (use if images appear transposed)')

    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    try:
        args = parse_arguments()
        
        # Create output directory.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[1m[36mCluster Visualization[/36m[/1m")
        console.print(f"[blue]ℹ[/blue] Coloring by: [1m{args.color_by}[/1m")
        console.print(f"[blue]ℹ[/blue] Circle radius: [1m{args.radius}[/1m pixels")
        console.print(f"[blue]ℹ[/blue] Samples: [1m{', '.join(args.samples)}[/1m")
        
        # Load and filter data.
        df = load_and_filter_data(args.metadata, args.samples)
        
        # Create visualization.
        stats, color_map = create_cluster_visualization(
            df, args.color_by, args.radius, tuple(args.figsize),
            args.alpha, args.output, args.flip_y
        )
        
        # Create legend.
        legend_path = args.output.with_name(args.output.stem + '_legend.png')
        create_legend(color_map, legend_path)
        
        # Save statistics.
        stats_path = args.output.with_name(args.output.stem + '_stats.txt')
        save_statistics(stats, color_map, stats_path)
        
        # Display summary table.
        table = Table(title="Visualization Summary", style="cyan")
        table.add_column("Metric", style="white")
        table.add_column("Value", style="green")
        
        table.add_row("Total Points", f"{stats['total_points']:,}")
        table.add_row("Samples", f"{len(stats['samples'])}")
        table.add_row("Unique Clusters", f"{len(color_map)}")
        table.add_row("Output Files", "3 (main, legend, stats)")
        
        console.print(table)
        console.print(f"[green]✓[/green] [1mVisualization complete![/1m")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
