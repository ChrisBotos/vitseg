"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: spatial_alignment.py.
Description:
    Spatial alignment verification script for ViT-derived clusters and spatial
    transcriptomics data. Creates overlay visualizations to verify coordinate
    system alignment before running comprehensive comparison analysis.

Dependencies:
    • Python >= 3.10.
    • pandas, numpy, matplotlib, seaborn.
    • rich (for enhanced console output).
    • scipy (for spatial statistics).

Usage:
    python -m vigseg.utilities.spatial_alignment \
        --vit_clusters results/spot_nuclei_clustering/spot_clusters.csv \
        --spatial_data data/metadata_complete.csv \
        --output results/spatial_alignment_verification \
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table
from rich.panel import Panel

console = Console()

# Try to import enhanced color generation.
try:
    from vigseg.utilities.color_generation import generate_color_palette, colors_to_hex_list
    ENHANCED_COLORS_AVAILABLE = True
    console.print("[green]✓[/green] Enhanced color generation available")
except ImportError:
    ENHANCED_COLORS_AVAILABLE = False
    console.print("[yellow]⚠[/yellow] Using fallback color generation")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def generate_consistent_colors(n_colors: int, palette_type: str = 'scientific') -> List[str]:
    """
    Generate consistent color palette for visualizations.
    
    Args:
        n_colors: Number of distinct colors needed.
        palette_type: Type of color palette.
        
    Returns:
        List of hex color codes.
        
    This function ensures consistent color schemes across all visualizations
    for proper visual comparison and interpretation.
    """
    if ENHANCED_COLORS_AVAILABLE:
        color_palette = generate_color_palette(
            n=n_colors,
            background="light",
            contrast_ratio=4.5,
            saturation=0.85
        )
        return colors_to_hex_list(color_palette)
    else:
        # Fallback to seaborn palettes.
        if palette_type == 'scientific':
            colors = sns.color_palette("Set2", n_colors)
        elif palette_type == 'accessible':
            colors = sns.color_palette("colorblind", n_colors)
        else:
            colors = sns.color_palette("husl", n_colors)
        
        return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in colors]


def load_and_validate_data(vit_path: Path, spatial_path: Path, samples: List[str],
                          cluster_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate both datasets with comprehensive error checking.
    
    Args:
        vit_path: Path to ViT cluster assignments.
        spatial_path: Path to spatial metadata.
        samples: List of sample names to include.
        cluster_column: Column name for spatial clusters.
        
    Returns:
        Tuple of (vit_data, spatial_data) DataFrames.
        
    This function performs comprehensive data validation and filtering
    to ensure both datasets are compatible for alignment verification.
    """
    console.print("[cyan]Loading and validating datasets...[/cyan]")
    
    # Load ViT cluster data.
    console.print(f"[blue]ℹ[/blue] Loading ViT clusters from {vit_path}")
    vit_data = pd.read_csv(vit_path)
    
    # Validate ViT data structure.
    required_vit_cols = ['spot_x', 'spot_y', 'sample', 'cluster']
    missing_vit_cols = [col for col in required_vit_cols if col not in vit_data.columns]
    if missing_vit_cols:
        raise ValueError(f"ViT data missing required columns: {missing_vit_cols}")

    # Load spatial data.
    console.print(f"[blue]ℹ[/blue] Loading spatial data from {spatial_path}")
    spatial_data = pd.read_csv(spatial_path)

    # Validate spatial data structure.
    required_spatial_cols = ['x', 'y', 'sample', cluster_column]
    missing_spatial_cols = [col for col in required_spatial_cols if col not in spatial_data.columns]
    if missing_spatial_cols:
        raise ValueError(f"Spatial data missing required columns: {missing_spatial_cols}")

    # Filter for specified samples - handle different sample naming conventions.
    # ViT data uses IRI1, IRI2, IRI3 while spatial data might use different naming.
    console.print(f"[blue]ℹ[/blue] Available ViT samples: {sorted(vit_data['sample'].unique())}")
    console.print(f"[blue]ℹ[/blue] Available spatial samples: {sorted(spatial_data['sample'].unique())}")

    # Map sample names if needed.
    sample_mapping = {}
    for sample in samples:
        # Check if sample exists directly in both datasets.
        if sample in vit_data['sample'].values and sample in spatial_data['sample'].values:
            sample_mapping[sample] = sample
        else:
            # Try to find matching samples with different naming conventions.
            vit_matches = [s for s in vit_data['sample'].unique() if sample.lower() in s.lower()]
            spatial_matches = [s for s in spatial_data['sample'].unique() if sample.lower() in s.lower()]
            if vit_matches and spatial_matches:
                sample_mapping[sample] = {'vit': vit_matches[0], 'spatial': spatial_matches[0]}
                console.print(f"[yellow]⚠[/yellow] Mapped {sample} to ViT: {vit_matches[0]}, Spatial: {spatial_matches[0]}")

    # Filter data based on sample mapping.
    vit_filtered_list = []
    spatial_filtered_list = []

    for sample in samples:
        if sample in sample_mapping:
            mapping = sample_mapping[sample]
            if isinstance(mapping, str):
                # Direct mapping.
                vit_sample_data = vit_data[vit_data['sample'] == mapping].copy()
                spatial_sample_data = spatial_data[spatial_data['sample'] == mapping].copy()
            else:
                # Different names for ViT and spatial.
                vit_sample_data = vit_data[vit_data['sample'] == mapping['vit']].copy()
                spatial_sample_data = spatial_data[spatial_data['sample'] == mapping['spatial']].copy()

            # Standardize sample names.
            vit_sample_data['sample'] = sample
            spatial_sample_data['sample'] = sample

            vit_filtered_list.append(vit_sample_data)
            spatial_filtered_list.append(spatial_sample_data)

    if vit_filtered_list:
        vit_filtered = pd.concat(vit_filtered_list, ignore_index=True)
    else:
        vit_filtered = pd.DataFrame()

    if spatial_filtered_list:
        spatial_filtered = pd.concat(spatial_filtered_list, ignore_index=True)
    else:
        spatial_filtered = pd.DataFrame()

    # Remove rows with missing cluster assignments.
    if len(vit_filtered) > 0:
        vit_filtered = vit_filtered.dropna(subset=['cluster'])
    if len(spatial_filtered) > 0:
        spatial_filtered = spatial_filtered.dropna(subset=[cluster_column])
    
    console.print(f"[green]✓[/green] ViT data: {len(vit_filtered):,} points from {len(vit_filtered['sample'].unique())} samples")
    console.print(f"[green]✓[/green] Spatial data: {len(spatial_filtered):,} points from {len(spatial_filtered['sample'].unique())} samples")
    
    return vit_filtered, spatial_filtered


def analyze_coordinate_systems(vit_data: pd.DataFrame, spatial_data: pd.DataFrame,
                              samples: List[str]) -> Dict[str, Any]:
    """
    Analyze coordinate systems and calculate alignment statistics.
    
    Args:
        vit_data: ViT cluster assignments with coordinates.
        spatial_data: Spatial metadata with coordinates.
        samples: List of sample names.
        
    Returns:
        Dictionary containing coordinate system analysis results.
        
    This function provides comprehensive coordinate system diagnostics
    including range analysis, overlap detection, and alignment metrics.
    """
    console.print("[cyan]Analyzing coordinate systems...[/cyan]")
    
    analysis = {
        'samples': {},
        'overall': {}
    }
    
    # Analyze each sample separately.
    for sample in samples:
        vit_sample = vit_data[vit_data['sample'] == sample]
        spatial_sample = spatial_data[spatial_data['sample'] == sample]
        
        if len(vit_sample) == 0 or len(spatial_sample) == 0:
            console.print(f"[yellow]⚠[/yellow] Sample {sample} missing in one or both datasets")
            continue
        
        # Calculate coordinate ranges.
        vit_x_range = (vit_sample['spot_x'].min(), vit_sample['spot_x'].max())
        vit_y_range = (vit_sample['spot_y'].min(), vit_sample['spot_y'].max())
        spatial_x_range = (spatial_sample['x'].min(), spatial_sample['x'].max())
        spatial_y_range = (spatial_sample['y'].min(), spatial_sample['y'].max())
        
        # Calculate overlap percentages.
        x_overlap = max(0, min(vit_x_range[1], spatial_x_range[1]) - max(vit_x_range[0], spatial_x_range[0]))
        y_overlap = max(0, min(vit_y_range[1], spatial_y_range[1]) - max(vit_y_range[0], spatial_y_range[0]))
        
        vit_x_span = vit_x_range[1] - vit_x_range[0]
        vit_y_span = vit_y_range[1] - vit_y_range[0]
        spatial_x_span = spatial_x_range[1] - spatial_x_range[0]
        spatial_y_span = spatial_y_range[1] - spatial_y_range[0]
        
        x_overlap_pct = (x_overlap / max(vit_x_span, spatial_x_span)) * 100 if max(vit_x_span, spatial_x_span) > 0 else 0
        y_overlap_pct = (y_overlap / max(vit_y_span, spatial_y_span)) * 100 if max(vit_y_span, spatial_y_span) > 0 else 0
        
        analysis['samples'][sample] = {
            'vit_points': len(vit_sample),
            'spatial_points': len(spatial_sample),
            'vit_x_range': vit_x_range,
            'vit_y_range': vit_y_range,
            'spatial_x_range': spatial_x_range,
            'spatial_y_range': spatial_y_range,
            'x_overlap': x_overlap,
            'y_overlap': y_overlap,
            'x_overlap_pct': x_overlap_pct,
            'y_overlap_pct': y_overlap_pct,
            'coordinate_alignment': 'good' if x_overlap_pct > 50 and y_overlap_pct > 50 else 'poor'
        }
        
        console.print(f"[blue]ℹ[/blue] {sample}: X overlap {x_overlap_pct:.1f}%, Y overlap {y_overlap_pct:.1f}%")
    
    # Calculate overall statistics.
    all_vit_x = vit_data['spot_x']
    all_vit_y = vit_data['spot_y']
    all_spatial_x = spatial_data['x']
    all_spatial_y = spatial_data['y']
    
    analysis['overall'] = {
        'total_vit_points': len(vit_data),
        'total_spatial_points': len(spatial_data),
        'vit_x_range': (all_vit_x.min(), all_vit_x.max()),
        'vit_y_range': (all_vit_y.min(), all_vit_y.max()),
        'spatial_x_range': (all_spatial_x.min(), all_spatial_x.max()),
        'spatial_y_range': (all_spatial_y.min(), all_spatial_y.max()),
        'coordinate_scale_ratio_x': (all_vit_x.max() - all_vit_x.min()) / (all_spatial_x.max() - all_spatial_x.min()),
        'coordinate_scale_ratio_y': (all_vit_y.max() - all_vit_y.min()) / (all_spatial_y.max() - all_spatial_y.min())
    }
    
    return analysis


def create_side_by_side_visualization(vit_data: pd.DataFrame, spatial_data: pd.DataFrame,
                                     samples: List[str], cluster_column: str,
                                     output_path: Path, flip_y: bool = False,
                                     alpha: float = 0.6) -> None:
    """
    Create side-by-side visualization comparing both datasets.
    
    Args:
        vit_data: ViT cluster assignments.
        spatial_data: Spatial metadata.
        samples: List of sample names.
        cluster_column: Column name for spatial clusters.
        output_path: Path for saving the visualization.
        flip_y: Whether to flip Y coordinates.
        alpha: Transparency level.
        
    This function creates publication-quality side-by-side comparisons
    showing both clustering approaches for direct visual inspection.
    """
    console.print("[cyan]Creating side-by-side visualization...[/cyan]")
    
    n_samples = len(samples)
    fig, axes = plt.subplots(2, n_samples, figsize=(6*n_samples, 12), facecolor='white')
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    # Generate colors for ViT clusters.
    vit_clusters = sorted(vit_data['cluster'].unique())
    vit_colors = generate_consistent_colors(len(vit_clusters), 'scientific')
    vit_color_map = dict(zip(vit_clusters, vit_colors))
    
    # Generate colors for spatial clusters.
    spatial_clusters = sorted(spatial_data[cluster_column].dropna().unique())
    spatial_colors = generate_consistent_colors(len(spatial_clusters), 'accessible')
    spatial_color_map = dict(zip(spatial_clusters, spatial_colors))
    
    for i, sample in enumerate(samples):
        # Filter data for current sample.
        vit_sample = vit_data[vit_data['sample'] == sample]
        spatial_sample = spatial_data[spatial_data['sample'] == sample]
        
        if len(vit_sample) == 0 or len(spatial_sample) == 0:
            console.print(f"[yellow]⚠[/yellow] Skipping {sample} - insufficient data")
            continue
        
        # Apply coordinate transformation if needed.
        vit_x, vit_y = vit_sample['spot_x'], vit_sample['spot_y']
        spatial_x, spatial_y = spatial_sample['x'], spatial_sample['y']
        
        if flip_y:
            vit_y = vit_y.max() - vit_y
            spatial_y = spatial_y.max() - spatial_y
        
        # Plot ViT clusters (top row).
        ax_vit = axes[0, i]
        for cluster in vit_clusters:
            cluster_mask = vit_sample['cluster'] == cluster
            if cluster_mask.sum() > 0:
                ax_vit.scatter(vit_x[cluster_mask], vit_y[cluster_mask],
                             c=vit_color_map[cluster], s=20, alpha=alpha,
                             edgecolors='white', linewidth=0.3,
                             label=f'ViT {cluster}')
        
        ax_vit.set_title(f'{sample} - ViT Clusters', fontsize=14, fontweight='bold')
        ax_vit.set_xlabel('X Coordinate', fontsize=12)
        ax_vit.set_ylabel('Y Coordinate', fontsize=12)
        ax_vit.set_aspect('equal', adjustable='box')
        ax_vit.grid(True, alpha=0.3)
        
        # Plot spatial clusters (bottom row).
        ax_spatial = axes[1, i]
        for cluster in spatial_clusters:
            cluster_mask = spatial_sample[cluster_column] == cluster
            if cluster_mask.sum() > 0:
                ax_spatial.scatter(spatial_x[cluster_mask], spatial_y[cluster_mask],
                                 c=spatial_color_map[cluster], s=20, alpha=alpha,
                                 edgecolors='white', linewidth=0.3,
                                 label=cluster)
        
        ax_spatial.set_title(f'{sample} - Spatial Clusters ({cluster_column})', fontsize=14, fontweight='bold')
        ax_spatial.set_xlabel('X Coordinate', fontsize=12)
        ax_spatial.set_ylabel('Y Coordinate', fontsize=12)
        ax_spatial.set_aspect('equal', adjustable='box')
        ax_spatial.grid(True, alpha=0.3)
        
        # Add legends for first column only.
        if i == 0:
            ax_vit.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax_spatial.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.suptitle('Spatial Alignment Verification: Side-by-Side Comparison', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"[green]✓[/green] Side-by-side visualization saved to {output_path}")


def create_overlay_visualization(vit_data: pd.DataFrame, spatial_data: pd.DataFrame,
                                samples: List[str], cluster_column: str,
                                output_path: Path, flip_y: bool = False,
                                alpha: float = 0.4) -> None:
    """
    Create overlay visualization with both datasets superimposed.
    
    Args:
        vit_data: ViT cluster assignments.
        spatial_data: Spatial metadata.
        samples: List of sample names.
        cluster_column: Column name for spatial clusters.
        output_path: Path for saving the visualization.
        flip_y: Whether to flip Y coordinates.
        alpha: Transparency level.
        
    This function creates overlay plots showing both datasets on the same
    coordinate system to verify spatial alignment and identify mismatches.
    """
    console.print("[cyan]Creating overlay visualization...[/cyan]")
    
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(8*n_samples, 8), facecolor='white')
    if n_samples == 1:
        axes = [axes]
    
    for i, sample in enumerate(samples):
        # Filter data for current sample.
        vit_sample = vit_data[vit_data['sample'] == sample]
        spatial_sample = spatial_data[spatial_data['sample'] == sample]
        
        if len(vit_sample) == 0 or len(spatial_sample) == 0:
            console.print(f"[yellow]⚠[/yellow] Skipping {sample} - insufficient data")
            continue
        
        # Apply coordinate transformation if needed.
        vit_x, vit_y = vit_sample['spot_x'], vit_sample['spot_y']
        spatial_x, spatial_y = spatial_sample['x'], spatial_sample['y']
        
        if flip_y:
            vit_y = vit_y.max() - vit_y
            spatial_y = spatial_y.max() - spatial_y
        
        ax = axes[i]
        
        # Plot ViT clusters in blue tones.
        ax.scatter(vit_x, vit_y, c='blue', s=15, alpha=alpha,
                  edgecolors='darkblue', linewidth=0.2, label='ViT Clusters')
        
        # Plot spatial clusters in red tones.
        ax.scatter(spatial_x, spatial_y, c='red', s=15, alpha=alpha,
                  edgecolors='darkred', linewidth=0.2, label=f'Spatial ({cluster_column})')
        
        ax.set_title(f'{sample} - Overlay Verification', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add coordinate range annotations.
        vit_x_range = f"ViT X: [{vit_x.min():.0f}, {vit_x.max():.0f}]"
        vit_y_range = f"ViT Y: [{vit_y.min():.0f}, {vit_y.max():.0f}]"
        spatial_x_range = f"Spatial X: [{spatial_x.min():.0f}, {spatial_x.max():.0f}]"
        spatial_y_range = f"Spatial Y: [{spatial_y.min():.0f}, {spatial_y.max():.0f}]"
        
        ax.text(0.02, 0.98, f"{vit_x_range}\n{vit_y_range}\n{spatial_x_range}\n{spatial_y_range}",
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Spatial Alignment Verification: Overlay Analysis', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"[green]✓[/green] Overlay visualization saved to {output_path}")


def save_diagnostic_report(analysis: Dict[str, Any], output_path: Path) -> None:
    """
    Save comprehensive diagnostic report with coordinate system analysis.
    
    Args:
        analysis: Coordinate system analysis results.
        output_path: Path for saving the diagnostic report.
        
    This function creates a detailed text report with coordinate system
    diagnostics, alignment statistics, and recommendations.
    """
    console.print("[cyan]Saving diagnostic report...[/cyan]")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Spatial Alignment Verification Report\n")
        f.write("=" * 40 + "\n\n")
        
        # Overall summary.
        overall = analysis['overall']
        f.write("Overall Dataset Summary:\n")
        f.write(f"  Total ViT points: {overall['total_vit_points']:,}\n")
        f.write(f"  Total spatial points: {overall['total_spatial_points']:,}\n")
        f.write(f"  ViT X range: [{overall['vit_x_range'][0]:.1f}, {overall['vit_x_range'][1]:.1f}]\n")
        f.write(f"  ViT Y range: [{overall['vit_y_range'][0]:.1f}, {overall['vit_y_range'][1]:.1f}]\n")
        f.write(f"  Spatial X range: [{overall['spatial_x_range'][0]:.1f}, {overall['spatial_x_range'][1]:.1f}]\n")
        f.write(f"  Spatial Y range: [{overall['spatial_y_range'][0]:.1f}, {overall['spatial_y_range'][1]:.1f}]\n")
        f.write(f"  X scale ratio: {overall['coordinate_scale_ratio_x']:.3f}\n")
        f.write(f"  Y scale ratio: {overall['coordinate_scale_ratio_y']:.3f}\n\n")
        
        # Sample-by-sample analysis.
        f.write("Sample-by-Sample Analysis:\n")
        f.write("-" * 30 + "\n")
        
        for sample, sample_data in analysis['samples'].items():
            f.write(f"\nSample: {sample}\n")
            f.write(f"  ViT points: {sample_data['vit_points']:,}\n")
            f.write(f"  Spatial points: {sample_data['spatial_points']:,}\n")
            f.write(f"  X overlap: {sample_data['x_overlap']:.1f} units ({sample_data['x_overlap_pct']:.1f}%)\n")
            f.write(f"  Y overlap: {sample_data['y_overlap']:.1f} units ({sample_data['y_overlap_pct']:.1f}%)\n")
            f.write(f"  Alignment quality: {sample_data['coordinate_alignment']}\n")
        
        # Recommendations.
        f.write("\nRecommendations:\n")
        f.write("-" * 15 + "\n")
        
        good_alignment_samples = [s for s, d in analysis['samples'].items() if d['coordinate_alignment'] == 'good']
        poor_alignment_samples = [s for s, d in analysis['samples'].items() if d['coordinate_alignment'] == 'poor']
        
        if len(good_alignment_samples) == len(analysis['samples']):
            f.write("✓ All samples show good coordinate alignment (>50% overlap)\n")
            f.write("✓ Proceed with comprehensive comparison analysis\n")
        elif len(good_alignment_samples) > 0:
            f.write(f"⚠ Mixed alignment quality:\n")
            f.write(f"  Good alignment: {', '.join(good_alignment_samples)}\n")
            f.write(f"  Poor alignment: {', '.join(poor_alignment_samples)}\n")
            f.write("⚠ Consider coordinate transformation or filtering problematic samples\n")
        else:
            f.write("✗ Poor coordinate alignment across all samples\n")
            f.write("✗ Coordinate system transformation required before analysis\n")
            f.write("✗ Consider using --flip_y option or manual coordinate adjustment\n")
        
        # Scale ratio analysis.
        x_ratio = overall['coordinate_scale_ratio_x']
        y_ratio = overall['coordinate_scale_ratio_y']
        
        if abs(x_ratio - 1.0) > 0.1 or abs(y_ratio - 1.0) > 0.1:
            f.write(f"\n⚠ Coordinate scale mismatch detected:\n")
            f.write(f"  X scale ratio: {x_ratio:.3f} (should be ~1.0)\n")
            f.write(f"  Y scale ratio: {y_ratio:.3f} (should be ~1.0)\n")
            f.write("⚠ Consider coordinate normalization or scaling adjustment\n")
    
    console.print(f"[green]✓[/green] Diagnostic report saved to {output_path}")


def save_alignment_statistics(analysis: Dict[str, Any], output_path: Path) -> None:
    """
    Save quantitative alignment statistics as JSON.
    
    Args:
        analysis: Coordinate system analysis results.
        output_path: Path for saving the statistics.
        
    This function saves machine-readable alignment statistics for
    programmatic analysis and integration with other tools.
    """
    console.print("[cyan]Saving alignment statistics...[/cyan]")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder)
    
    console.print(f"[green]✓[/green] Alignment statistics saved to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Verify spatial alignment between ViT and spatial transcriptomics data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--vit_clusters', type=Path, required=True,
                       help='Path to ViT cluster assignments CSV')
    parser.add_argument('--spatial_data', type=Path, required=True,
                       help='Path to spatial metadata CSV')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for verification results')
    parser.add_argument('--samples', nargs='+', default=['IRI1', 'IRI2', 'IRI3'],
                       help='Sample names to include')
    parser.add_argument('--cluster_column', default='figure_idents',
                       help='Column name for spatial clusters')
    parser.add_argument('--flip_y', action='store_true',
                       help='Apply Y-coordinate flipping for alignment')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='Transparency for overlay plots')
    
    args = parser.parse_args()
    
    try:
        # Display header.
        header = Panel.fit(
            "[bold cyan]Spatial Alignment Verification[/bold cyan]\n"
            "[dim]ViT-Spatial Transcriptomics Coordinate System Analysis[/dim]",
            border_style="cyan"
        )
        console.print(header)
        console.print(f"[blue]ℹ[/blue] Samples: [bold]{', '.join(args.samples)}[/bold]")
        console.print(f"[blue]ℹ[/blue] Spatial clusters: [bold]{args.cluster_column}[/bold]")
        console.print(f"[blue]ℹ[/blue] Y-flip: [bold]{args.flip_y}[/bold]")
        
        # Create output directory.
        args.output.mkdir(parents=True, exist_ok=True)
        
        # Load and validate data.
        vit_data, spatial_data = load_and_validate_data(
            args.vit_clusters, args.spatial_data, args.samples, args.cluster_column
        )
        
        # Analyze coordinate systems.
        analysis = analyze_coordinate_systems(vit_data, spatial_data, args.samples)
        
        # Create visualizations.
        create_side_by_side_visualization(
            vit_data, spatial_data, args.samples, args.cluster_column,
            args.output / 'alignment_verification.png', args.flip_y, args.alpha
        )
        
        create_overlay_visualization(
            vit_data, spatial_data, args.samples, args.cluster_column,
            args.output / 'overlay_visualization.png', args.flip_y, args.alpha
        )
        
        # Save diagnostic outputs.
        save_diagnostic_report(analysis, args.output / 'coordinate_diagnostics.txt')
        save_alignment_statistics(analysis, args.output / 'alignment_statistics.json')
        
        # Display summary table.
        table = Table(title="Alignment Verification Summary", style="cyan")
        table.add_column("Sample", style="white")
        table.add_column("ViT Points", style="green")
        table.add_column("Spatial Points", style="green")
        table.add_column("X Overlap %", style="yellow")
        table.add_column("Y Overlap %", style="yellow")
        table.add_column("Alignment", style="red")
        
        for sample, sample_data in analysis['samples'].items():
            alignment_color = "green" if sample_data['coordinate_alignment'] == 'good' else "red"
            table.add_row(
                sample,
                f"{sample_data['vit_points']:,}",
                f"{sample_data['spatial_points']:,}",
                f"{sample_data['x_overlap_pct']:.1f}%",
                f"{sample_data['y_overlap_pct']:.1f}%",
                f"[{alignment_color}]{sample_data['coordinate_alignment']}[/{alignment_color}]"
            )
        
        console.print(table)
        
        # Final recommendation.
        good_samples = sum(1 for d in analysis['samples'].values() if d['coordinate_alignment'] == 'good')
        total_samples = len(analysis['samples'])
        
        if good_samples == total_samples:
            console.print(f"[green]✓[/green] [bold]All samples show good alignment - proceed with comparison analysis![/bold]")
        elif good_samples > 0:
            console.print(f"[yellow]⚠[/yellow] [bold]{good_samples}/{total_samples} samples show good alignment - review diagnostics[/bold]")
        else:
            console.print(f"[red]✗[/red] [bold]Poor alignment detected - coordinate transformation needed[/bold]")
        
        console.print(f"[blue]ℹ[/blue] All results saved to {args.output}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Verification failed: {str(e)}")
        console.print("[red]Traceback:[/red]")
        console.print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
