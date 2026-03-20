"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: prepare_vit_data.py.
Description:
    Prepares ViT cluster data from IRI_regist_14k for spatial alignment
    verification by assigning sample labels based on coordinate ranges and
    reformatting the data to match the expected format.

Dependencies:
    • Python >= 3.10.
    • pandas, numpy.
    • rich (for enhanced console output).

Usage:
    python -m vitseg.utilities.prepare_vit_data \
        --vit_clusters results/IRI_regist_14k/patch_clusters.csv \
        --spatial_data data/metadata_complete.csv \
        --output results/IRI_regist_14k/vit_clusters_formatted.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_spatial_sample_boundaries(spatial_data_path: Path) -> dict:
    """
    Load spatial data and determine sample coordinate boundaries.
    
    Args:
        spatial_data_path: Path to spatial metadata CSV.
        
    Returns:
        Dictionary with sample boundaries and centroids.
    """
    console.print("[cyan]Loading spatial data to determine sample boundaries...[/cyan]")
    
    spatial_data = pd.read_csv(spatial_data_path)
    iri_data = spatial_data[spatial_data['sample'].isin(['IRI1', 'IRI2', 'IRI3'])]
    
    boundaries = {}
    for sample in ['IRI1', 'IRI2', 'IRI3']:
        sample_data = iri_data[iri_data['sample'] == sample]
        boundaries[sample] = {
            'x_min': sample_data['x'].min(),
            'x_max': sample_data['x'].max(),
            'y_min': sample_data['y'].min(),
            'y_max': sample_data['y'].max(),
            'x_center': (sample_data['x'].min() + sample_data['x'].max()) / 2,
            'y_center': (sample_data['y'].min() + sample_data['y'].max()) / 2,
            'n_points': len(sample_data)
        }
    
    # Display boundaries.
    table = Table(title="Sample Coordinate Boundaries")
    table.add_column("Sample", style="cyan")
    table.add_column("X Range", style="green")
    table.add_column("Y Range", style="green")
    table.add_column("Center", style="yellow")
    table.add_column("Points", style="blue")
    
    for sample, bounds in boundaries.items():
        table.add_row(
            sample,
            f"[{bounds['x_min']}, {bounds['x_max']}]",
            f"[{bounds['y_min']}, {bounds['y_max']}]",
            f"({bounds['x_center']:.0f}, {bounds['y_center']:.0f})",
            f"{bounds['n_points']:,}"
        )
    
    console.print(table)
    return boundaries


def assign_vit_points_to_samples(vit_data: pd.DataFrame, boundaries: dict) -> pd.DataFrame:
    """
    Assign ViT points to samples based on coordinate overlap.
    
    Args:
        vit_data: ViT cluster data with x_center, y_center columns.
        boundaries: Sample coordinate boundaries.
        
    Returns:
        ViT data with sample assignments.
    """
    console.print("[cyan]Assigning ViT points to samples based on coordinates...[/cyan]")
    
    # Initialize sample assignment.
    vit_data = vit_data.copy()
    vit_data['sample'] = 'unassigned'
    vit_data['condition'] = 'IRI'
    
    assignment_stats = {}
    
    for sample, bounds in boundaries.items():
        # Find points within sample boundaries.
        mask = (
            (vit_data['x_center'] >= bounds['x_min']) &
            (vit_data['x_center'] <= bounds['x_max']) &
            (vit_data['y_center'] >= bounds['y_min']) &
            (vit_data['y_center'] <= bounds['y_max'])
        )
        
        vit_data.loc[mask, 'sample'] = sample
        assignment_stats[sample] = mask.sum()
    
    # Handle overlapping regions by distance to centroid.
    unassigned_mask = vit_data['sample'] == 'unassigned'
    if unassigned_mask.sum() > 0:
        console.print(f"[yellow]⚠[/yellow] {unassigned_mask.sum():,} points outside sample boundaries")
        console.print("[cyan]Assigning based on distance to sample centroids...[/cyan]")
        
        for idx in vit_data[unassigned_mask].index:
            x, y = vit_data.loc[idx, 'x_center'], vit_data.loc[idx, 'y_center']
            
            # Calculate distance to each sample centroid.
            distances = {}
            for sample, bounds in boundaries.items():
                dist = np.sqrt((x - bounds['x_center'])**2 + (y - bounds['y_center'])**2)
                distances[sample] = dist
            
            # Assign to closest sample.
            closest_sample = min(distances, key=distances.get)
            vit_data.loc[idx, 'sample'] = closest_sample
            assignment_stats[closest_sample] += 1
    
    # Display assignment statistics.
    table = Table(title="ViT Point Assignment Statistics")
    table.add_column("Sample", style="cyan")
    table.add_column("Assigned Points", style="green")
    table.add_column("Percentage", style="yellow")
    
    total_points = len(vit_data)
    for sample in ['IRI1', 'IRI2', 'IRI3']:
        count = assignment_stats.get(sample, 0)
        percentage = (count / total_points) * 100
        table.add_row(sample, f"{count:,}", f"{percentage:.1f}%")
    
    table.add_row("TOTAL", f"{total_points:,}", "100.0%", style="bold")
    console.print(table)
    
    return vit_data


def format_vit_data_for_verification(vit_data: pd.DataFrame) -> pd.DataFrame:
    """
    Format ViT data to match verification script expectations.
    
    Args:
        vit_data: ViT data with sample assignments.
        
    Returns:
        Formatted ViT data with expected column names.
    """
    console.print("[cyan]Formatting data for verification script...[/cyan]")
    
    # Rename columns to match verification script expectations.
    formatted_data = vit_data.rename(columns={
        'x_center': 'spot_x',
        'y_center': 'spot_y'
    })
    
    # Ensure required columns are present.
    required_columns = ['spot_x', 'spot_y', 'sample', 'condition', 'cluster']
    for col in required_columns:
        if col not in formatted_data.columns:
            console.print(f"[red]✗[/red] Missing required column: {col}")
            
    console.print(f"[green]✓[/green] Formatted data: {len(formatted_data):,} points")
    console.print(f"[green]✓[/green] Columns: {list(formatted_data.columns)}")
    
    return formatted_data


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Prepare ViT data for spatial alignment verification")
    parser.add_argument("--vit_clusters", type=Path, required=True,
                       help="Path to ViT cluster CSV file")
    parser.add_argument("--spatial_data", type=Path, required=True,
                       help="Path to spatial metadata CSV file")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output path for formatted ViT data")
    
    args = parser.parse_args()
    
    # Display header.
    header = Panel(
        "[bold]ViT Data Preparation for Spatial Alignment Verification[/bold]\n"
        "[dim]Assigning sample labels and formatting data[/dim]",
        border_style="cyan"
    )
    console.print(header)
    
    try:
        # Load ViT data.
        console.print(f"[blue]ℹ[/blue] Loading ViT clusters from {args.vit_clusters}")
        vit_data = pd.read_csv(args.vit_clusters)
        console.print(f"[green]✓[/green] Loaded {len(vit_data):,} ViT points")
        
        # Load spatial boundaries.
        boundaries = load_spatial_sample_boundaries(args.spatial_data)
        
        # Assign samples.
        vit_data_with_samples = assign_vit_points_to_samples(vit_data, boundaries)
        
        # Format for verification.
        formatted_data = format_vit_data_for_verification(vit_data_with_samples)
        
        # Save formatted data.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        formatted_data.to_csv(args.output, index=False)
        console.print(f"[green]✓[/green] Formatted ViT data saved to {args.output}")
        
        console.print(f"[blue]ℹ[/blue] Ready for spatial alignment verification!")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Preparation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
