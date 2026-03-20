#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_spot_nuclei_coordinates.py.
Description:
    Quick test script to verify coordinate compatibility between nuclei and spots
    before running the full spot-nuclei clustering analysis. Loads a small sample
    of coordinates, analyzes ranges and overlaps, and suggests optimal parameters.

Dependencies:
    • Python >= 3.10.
    • pandas, numpy, rich, scipy.

Usage:
    python code/test_spot_nuclei_coordinates.py
"""

import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from scipy.spatial.distance import cdist

console = Console()


def analyze_coordinate_compatibility():
    """
    Analyze coordinate compatibility between nuclei and spots.
    
    This function loads sample data and analyzes whether the coordinate systems
    are compatible, suggesting optimal parameters for the full analysis.
    """
    console.print("[cyan][1m Testing Coordinate Compatibility [/1m[/cyan]")
    console.print()
    
    # Load sample nuclei coordinates (first 1000 for speed).
    nuclei_path = Path("results/IRI_regist_14k/coords_IRI_regist_binary_mask.csv")
    spots_path = Path("data/metadata_complete.csv")
    
    if not nuclei_path.exists():
        console.print(f"[red]ERROR[/red] Nuclei coordinates not found: {nuclei_path}")
        return False
        
    if not spots_path.exists():
        console.print(f"[red]ERROR[/red] Spots metadata not found: {spots_path}")
        return False
    
    console.print(f"[cyan]Loading sample nuclei coordinates...[/cyan]")
    nuclei_df = pd.read_csv(nuclei_path, nrows=1000)  # Sample first 1000.
    console.print(f"[green]OK[/green] Loaded {len(nuclei_df):,} sample nuclei")
    
    console.print(f"[cyan]Loading spots metadata...[/cyan]")
    spots_df = pd.read_csv(spots_path)
    
    # Filter for IRI samples.
    iri_spots = spots_df[spots_df['sample'].isin(['IRI1', 'IRI2', 'IRI3'])].copy()
    console.print(f"[green]OK[/green] Loaded {len(iri_spots):,} IRI spots")
    
    # Analyze coordinate ranges.
    console.print()
    console.print("[cyan][1m Coordinate Range Analysis [/1m[/cyan]")
    
    table = Table(title="Coordinate Ranges")
    table.add_column("Dataset", style="cyan")
    table.add_column("X Min", justify="right")
    table.add_column("X Max", justify="right") 
    table.add_column("Y Min", justify="right")
    table.add_column("Y Max", justify="right")
    table.add_column("Range", justify="right")
    
    # Nuclei ranges.
    nuclei_x_min, nuclei_x_max = nuclei_df['x_center'].min(), nuclei_df['x_center'].max()
    nuclei_y_min, nuclei_y_max = nuclei_df['y_center'].min(), nuclei_df['y_center'].max()
    nuclei_range = max(nuclei_x_max - nuclei_x_min, nuclei_y_max - nuclei_y_min)
    
    table.add_row(
        "Nuclei", 
        f"{nuclei_x_min:.0f}", f"{nuclei_x_max:.0f}",
        f"{nuclei_y_min:.0f}", f"{nuclei_y_max:.0f}",
        f"{nuclei_range:.0f}"
    )
    
    # Spots ranges.
    spots_x_min, spots_x_max = iri_spots['x'].min(), iri_spots['x'].max()
    spots_y_min, spots_y_max = iri_spots['y'].min(), iri_spots['y'].max()
    spots_range = max(spots_x_max - spots_x_min, spots_y_max - spots_y_min)
    
    table.add_row(
        "Spots",
        f"{spots_x_min:.0f}", f"{spots_x_max:.0f}",
        f"{spots_y_min:.0f}", f"{spots_y_max:.0f}",
        f"{spots_range:.0f}"
    )
    
    console.print(table)
    
    # Check overlap.
    console.print()
    console.print("[cyan][1m Overlap Analysis [/1m[/cyan]")
    
    x_overlap = not (nuclei_x_max < spots_x_min or spots_x_max < nuclei_x_min)
    y_overlap = not (nuclei_y_max < spots_y_min or spots_y_max < nuclei_y_min)
    has_overlap = x_overlap and y_overlap
    
    if has_overlap:
        console.print("[green]OK[/green] Coordinate systems have overlap - compatible!")
        
        # Calculate overlap region.
        overlap_x_min = max(nuclei_x_min, spots_x_min)
        overlap_x_max = min(nuclei_x_max, spots_x_max)
        overlap_y_min = max(nuclei_y_min, spots_y_min)
        overlap_y_max = min(nuclei_y_max, spots_y_max)
        
        console.print(f"[blue]INFO[/blue] Overlap region: X({overlap_x_min:.0f}-{overlap_x_max:.0f}), Y({overlap_y_min:.0f}-{overlap_y_max:.0f})")
        
    else:
        console.print("[red]ERROR[/red] No coordinate overlap - systems are incompatible!")
        console.print("[yellow]WARNING[/yellow] You will need to use --create_synthetic_spots")
        return False
    
    # Test distance calculations.
    console.print()
    console.print("[cyan][1m Distance Analysis [/1m[/cyan]")
    
    # Sample 100 nuclei and 1000 spots for distance testing.
    sample_nuclei = nuclei_df.sample(min(100, len(nuclei_df)))
    sample_spots = iri_spots.sample(min(1000, len(iri_spots)))
    
    nuclei_positions = sample_nuclei[['x_center', 'y_center']].values
    spot_positions = sample_spots[['x', 'y']].values
    
    console.print(f"[cyan]Computing distances for {len(sample_nuclei)} nuclei to {len(sample_spots)} spots...[/cyan]")
    distances = cdist(nuclei_positions, spot_positions, metric='euclidean')
    
    # Find closest distances.
    closest_distances = distances.min(axis=1)
    
    # Analyze distance distribution.
    distance_stats = {
        'min': closest_distances.min(),
        'max': closest_distances.max(),
        'mean': closest_distances.mean(),
        'median': np.median(closest_distances),
        'p10': np.percentile(closest_distances, 10),
        'p90': np.percentile(closest_distances, 90)
    }
    
    console.print(f"[blue]INFO[/blue] Distance statistics:")
    console.print(f"  • Minimum: {distance_stats['min']:.1f} pixels")
    console.print(f"  • Maximum: {distance_stats['max']:.1f} pixels")
    console.print(f"  • Mean: {distance_stats['mean']:.1f} pixels")
    console.print(f"  • Median: {distance_stats['median']:.1f} pixels")
    console.print(f"  • 10th percentile: {distance_stats['p10']:.1f} pixels")
    console.print(f"  • 90th percentile: {distance_stats['p90']:.1f} pixels")
    
    # Suggest optimal max_distance.
    console.print()
    console.print("[cyan][1m Recommendations [/1m[/cyan]")
    
    # Test different max_distance values.
    test_distances = [100, 200, 500, 1000, 2000]
    
    console.print("[blue]INFO[/blue] Assignment success rates for different max_distance values:")
    
    best_distance = None
    best_rate = 0
    
    for max_dist in test_distances:
        assigned = (closest_distances <= max_dist).sum()
        rate = assigned / len(closest_distances) * 100
        
        if rate >= 80 and best_distance is None:  # First distance giving >=80% success.
            best_distance = max_dist
            best_rate = rate
        
        status = "OK" if rate >= 80 else "WARNING" if rate >= 50 else "ERROR"
        color = "green" if rate >= 80 else "yellow" if rate >= 50 else "red"
        console.print(f"  [{color}]{status}[/{color}] max_distance={max_dist}: {rate:.1f}% ({assigned}/{len(closest_distances)})")
    
    if best_distance:
        console.print()
        console.print(f"[green]OK[/green] [1m Recommended max_distance: {best_distance} pixels [/1m")
        console.print(f"[green]OK[/green] Expected assignment rate: {best_rate:.1f}%")
        
        # Suggest command.
        console.print()
        console.print("[cyan][1m Suggested Command [/1m[/cyan]")
        console.print(f"""
python code/cluster_spots_by_nuclei_features.py \\
    --nuclei_coords results/IRI_regist_14k/coords_IRI_regist_binary_mask.csv \\
    --nuclei_features results/IRI_regist_14k/features_IRI_regist_binary_mask.csv \\
    --spots_metadata data/metadata_complete.csv \\
    --output results/spot_nuclei_clustering_14k \\
    --samples IRI1 IRI2 IRI3 \\
    --clusters 15 \\
    --max_distance {best_distance} \\
    --aggregation_method mean
        """.strip())
        
        return True
    else:
        console.print()
        console.print("[red]ERROR[/red] No suitable max_distance found")
        console.print("[yellow]WARNING[/yellow] Consider using --create_synthetic_spots")
        return False


def main():
    """Run coordinate compatibility test and print recommendations."""
    try:
        success = analyze_coordinate_compatibility()
        
        if success:
            console.print()
            console.print("[green][1m OK Coordinate systems are compatible! [/1m[/green]")
            console.print("[green]You can proceed with the full analysis using the suggested parameters.[/green]")
        else:
            console.print()
            console.print("[red][1m ERROR Coordinate systems need adjustment [/1m[/red]")
            console.print("[yellow]Consider using synthetic spots or adjusting parameters.[/yellow]")
            
        return 0 if success else 1
        
    except Exception as e:
        console.print(f"[red]ERROR[/red] Error during analysis: {e}")
        console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())
