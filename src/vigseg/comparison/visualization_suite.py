"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: visualization_suite.py.
Description:
    Comprehensive visualization suite for cluster comparison analysis.
    Creates publication-quality visualizations including side-by-side cluster maps,
    Sankey diagrams for cluster transitions, scatter plots of centroids,
    and heatmaps of overlap matrices.

Dependencies:
    • Python >= 3.10.
    • pandas, numpy, matplotlib, seaborn, plotly.
    • rich (for enhanced console output).
    • networkx (for network visualizations).

Usage:
    python -m vigseg.comparison.visualization_suite \
        --metrics_data comparison_analysis/results/metrics/alignment_metrics.json \
        --cluster_data results/spot_nuclei_clustering/spot_clusters.csv \
        --spatial_data data/metadata_complete.csv \
        --output comparison_analysis/visualizations/ \
        --samples IRI1 IRI2 IRI3
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

console = Console()

# Try to import enhanced visualization libraries.
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
    console.print("[green]✓[/green] Plotly available for interactive visualizations")
except ImportError:
    PLOTLY_AVAILABLE = False
    console.print("[yellow]⚠[/yellow] Plotly not available, using matplotlib only")

try:
    from vigseg.utilities.color_generation import generate_color_palette, colors_to_hex_list
    ENHANCED_COLORS_AVAILABLE = True
    console.print("[green]✓[/green] Enhanced color generation available")
except ImportError:
    ENHANCED_COLORS_AVAILABLE = False
    console.print("[yellow]⚠[/yellow] Using fallback color generation")


def generate_consistent_colors(n_colors: int, palette_type: str = 'scientific') -> List[str]:
    """
    Generate consistent color palette for all visualizations.
    
    Args:
        n_colors: Number of distinct colors needed.
        palette_type: Type of color palette ('scientific', 'accessible', 'vibrant').
        
    Returns:
        List of hex color codes.
        
    This function ensures all visualizations use the same color scheme
    for consistent interpretation across figures.
    """
    if ENHANCED_COLORS_AVAILABLE:
        color_palette = generate_color_palette(
            n=n_colors,
            background="white",
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


def create_side_by_side_cluster_maps(vit_data: pd.DataFrame, spatial_data: pd.DataFrame,
                                    samples: List[str], output_path: Path) -> None:
    """
    Create side-by-side cluster visualization maps.
    
    Args:
        vit_data: ViT cluster assignments with coordinates.
        spatial_data: Spatial cluster assignments with coordinates.
        samples: List of sample names to visualize.
        output_path: Path for saving the visualization.
        
    This function creates publication-quality side-by-side comparisons
    showing both clustering approaches for direct visual comparison.
    """
    console.print("[cyan]Creating side-by-side cluster maps...[/cyan]")
    
    # Set up the figure with subplots.
    n_samples = len(samples)
    fig, axes = plt.subplots(2, n_samples, figsize=(6*n_samples, 12), facecolor='white')
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    # Generate consistent colors for ViT clusters.
    vit_clusters = sorted(vit_data['cluster'].unique())
    vit_colors = generate_consistent_colors(len(vit_clusters), 'scientific')
    vit_color_map = dict(zip(vit_clusters, vit_colors))
    
    # Generate consistent colors for spatial clusters.
    spatial_clusters = sorted(spatial_data['figure_idents'].dropna().unique())
    spatial_colors = generate_consistent_colors(len(spatial_clusters), 'accessible')
    spatial_color_map = dict(zip(spatial_clusters, spatial_colors))
    
    for i, sample in enumerate(samples):
        # Filter data for current sample.
        vit_sample = vit_data[vit_data['sample'] == sample]
        spatial_sample = spatial_data[spatial_data['sample'] == sample]
        
        # Plot ViT clusters (top row).
        ax_vit = axes[0, i]
        for cluster in vit_clusters:
            cluster_data = vit_sample[vit_sample['cluster'] == cluster]
            if len(cluster_data) > 0:
                ax_vit.scatter(cluster_data['spot_x'], cluster_data['spot_y'],
                             c=vit_color_map[cluster], s=30, alpha=0.8,
                             edgecolors='white', linewidth=0.5,
                             label=f'ViT {cluster}')
        
        ax_vit.set_title(f'{sample} - ViT Clusters', fontsize=14, fontweight='bold')
        ax_vit.set_xlabel('X Coordinate', fontsize=12)
        ax_vit.set_ylabel('Y Coordinate', fontsize=12)
        ax_vit.set_aspect('equal', adjustable='box')
        ax_vit.grid(True, alpha=0.3)
        
        # Plot spatial clusters (bottom row).
        ax_spatial = axes[1, i]
        for cluster in spatial_clusters:
            cluster_data = spatial_sample[spatial_sample['figure_idents'] == cluster]
            if len(cluster_data) > 0:
                ax_spatial.scatter(cluster_data['x'], cluster_data['y'],
                                 c=spatial_color_map[cluster], s=30, alpha=0.8,
                                 edgecolors='white', linewidth=0.5,
                                 label=cluster)
        
        ax_spatial.set_title(f'{sample} - Spatial Clusters', fontsize=14, fontweight='bold')
        ax_spatial.set_xlabel('X Coordinate', fontsize=12)
        ax_spatial.set_ylabel('Y Coordinate', fontsize=12)
        ax_spatial.set_aspect('equal', adjustable='box')
        ax_spatial.grid(True, alpha=0.3)
        
        # Add legends for first column only.
        if i == 0:
            ax_vit.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax_spatial.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.suptitle('ViT vs Spatial Clustering Comparison', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"[green]✓[/green] Side-by-side cluster maps saved to {output_path}")


def create_sankey_diagram(confusion_matrix_data: Dict[str, Any], output_path: Path) -> None:
    """
    Create Sankey diagram showing cluster membership transitions.
    
    Args:
        confusion_matrix_data: Confusion matrix results from metrics analysis.
        output_path: Path for saving the interactive diagram.
        
    Sankey diagrams effectively visualize how cluster memberships
    transition between the two clustering approaches.
    """
    console.print("[cyan]Creating Sankey cluster flow diagram...[/cyan]")
    
    if not PLOTLY_AVAILABLE:
        console.print("[yellow]⚠[/yellow] Plotly not available, skipping Sankey diagram")
        return
    
    # Extract confusion matrix data.
    cm = np.array(confusion_matrix_data['confusion_matrix'])
    spatial_labels = confusion_matrix_data['spatial_labels']
    vit_labels = confusion_matrix_data['vit_labels']
    
    # Prepare data for Sankey diagram.
    source_indices = []
    target_indices = []
    values = []
    
    # Create node labels.
    node_labels = [f"Spatial_{label}" for label in spatial_labels] + [f"ViT_{label}" for label in vit_labels]
    
    # Build connections.
    for i, spatial_label in enumerate(spatial_labels):
        for j, vit_label in enumerate(vit_labels):
            if i < cm.shape[0] and j < cm.shape[1] and cm[i, j] > 0:
                source_indices.append(i)  # Spatial cluster index.
                target_indices.append(len(spatial_labels) + j)  # ViT cluster index (offset).
                values.append(int(cm[i, j]))
    
    # Create Sankey diagram.
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="lightblue"
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color="rgba(0,0,255,0.3)"
        )
    )])
    
    fig.update_layout(
        title_text="Cluster Membership Flow: Spatial → ViT",
        font_size=12,
        width=1200,
        height=800
    )
    
    # Save as HTML.
    fig.write_html(str(output_path))
    console.print(f"[green]✓[/green] Sankey diagram saved to {output_path}")


def create_centroid_scatter_plot(vit_data: pd.DataFrame, spatial_data: pd.DataFrame,
                                samples: List[str], output_path: Path) -> None:
    """
    Create scatter plot of cluster centroids in feature space.
    
    Args:
        vit_data: ViT cluster data with feature information.
        spatial_data: Spatial cluster data.
        samples: List of sample names.
        output_path: Path for saving the plot.
        
    Centroid analysis reveals how cluster centers relate between
    the two clustering approaches in high-dimensional feature space.
    """
    console.print("[cyan]Creating cluster centroid scatter plot...[/cyan]")
    
    # Calculate centroids for ViT clusters.
    feature_cols = [col for col in vit_data.columns if col.startswith('agg_')]
    
    if len(feature_cols) < 2:
        console.print("[yellow]⚠[/yellow] Insufficient features for centroid analysis")
        return
    
    vit_centroids = []
    for cluster in sorted(vit_data['cluster'].unique()):
        cluster_data = vit_data[vit_data['cluster'] == cluster]
        centroid = cluster_data[feature_cols].mean()
        vit_centroids.append({
            'cluster': cluster,
            'type': 'ViT',
            'x': centroid.iloc[0],  # First feature dimension.
            'y': centroid.iloc[1],  # Second feature dimension.
            'size': len(cluster_data)
        })
    
    # For spatial clusters, we'll use spatial coordinates as proxy.
    spatial_centroids = []
    for cluster in sorted(spatial_data['figure_idents'].dropna().unique()):
        cluster_data = spatial_data[spatial_data['figure_idents'] == cluster]
        if len(cluster_data) > 0:
            spatial_centroids.append({
                'cluster': cluster,
                'type': 'Spatial',
                'x': cluster_data['x'].mean(),
                'y': cluster_data['y'].mean(),
                'size': len(cluster_data)
            })
    
    # Create the plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    
    # Plot ViT centroids.
    vit_df = pd.DataFrame(vit_centroids)
    if len(vit_df) > 0:
        scatter1 = ax1.scatter(vit_df['x'], vit_df['y'], s=vit_df['size']*2,
                              alpha=0.7, c=range(len(vit_df)), cmap='Set2',
                              edgecolors='black', linewidth=1)
        
        for _, row in vit_df.iterrows():
            ax1.annotate(f"ViT {row['cluster']}", (row['x'], row['y']),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax1.set_title('ViT Cluster Centroids\n(Feature Space)', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'Feature Dimension 1', fontsize=12)
    ax1.set_ylabel(f'Feature Dimension 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot spatial centroids.
    spatial_df = pd.DataFrame(spatial_centroids)
    if len(spatial_df) > 0:
        scatter2 = ax2.scatter(spatial_df['x'], spatial_df['y'], s=spatial_df['size']*2,
                              alpha=0.7, c=range(len(spatial_df)), cmap='Set3',
                              edgecolors='black', linewidth=1)
        
        for _, row in spatial_df.iterrows():
            ax2.annotate(row['cluster'], (row['x'], row['y']),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax2.set_title('Spatial Cluster Centroids\n(Coordinate Space)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Coordinate', fontsize=12)
    ax2.set_ylabel('Y Coordinate', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Cluster Centroid Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"[green]✓[/green] Centroid scatter plot saved to {output_path}")


def create_overlap_heatmap(confusion_matrix_data: Dict[str, Any], output_path: Path) -> None:
    """
    Create heatmap visualization of cluster overlap matrix.

    Args:
        confusion_matrix_data: Confusion matrix results from metrics analysis.
        output_path: Path for saving the heatmap.

    Heatmaps provide intuitive visualization of cluster correspondence
    patterns and highlight the strongest associations between clusterings.
    """
    console.print("[cyan]Creating cluster overlap heatmap...[/cyan]")

    # Extract confusion matrix data.
    cm = np.array(confusion_matrix_data['confusion_matrix'])
    spatial_labels = confusion_matrix_data['spatial_labels']
    vit_labels = confusion_matrix_data['vit_labels']

    # Normalize by row to show proportions.
    cm_normalized = cm.astype(float)
    row_sums = cm_normalized.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero.
    cm_normalized = cm_normalized / row_sums[:, np.newaxis]

    # Create the heatmap.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')

    # Raw counts heatmap.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'ViT_{label}' for label in vit_labels],
                yticklabels=[f'Spatial_{label}' for label in spatial_labels],
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Cluster Overlap Matrix\n(Raw Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('ViT Clusters', fontsize=12)
    ax1.set_ylabel('Spatial Clusters', fontsize=12)

    # Normalized proportions heatmap.
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=[f'ViT_{label}' for label in vit_labels],
                yticklabels=[f'Spatial_{label}' for label in spatial_labels],
                ax=ax2, cbar_kws={'label': 'Proportion'})
    ax2.set_title('Cluster Overlap Matrix\n(Row Proportions)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('ViT Clusters', fontsize=12)
    ax2.set_ylabel('Spatial Clusters', fontsize=12)

    plt.suptitle('Cluster Correspondence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    console.print(f"[green]✓[/green] Overlap heatmap saved to {output_path}")


def create_silhouette_comparison(silhouette_data: Dict[str, Any], output_path: Path) -> None:
    """
    Create silhouette score comparison visualization.

    Args:
        silhouette_data: Silhouette analysis results from metrics.
        output_path: Path for saving the comparison plot.

    Silhouette comparison shows cluster quality differences between
    the two clustering approaches with statistical context.
    """
    console.print("[cyan]Creating silhouette score comparison...[/cyan]")

    # Extract silhouette data.
    vit_silhouette = silhouette_data['vit_silhouette']
    spatial_silhouette = silhouette_data['spatial_silhouette']
    vit_cluster_scores = silhouette_data['vit_cluster_scores']
    spatial_cluster_scores = silhouette_data['spatial_cluster_scores']

    # Prepare data for plotting.
    cluster_data = []

    for cluster, stats in vit_cluster_scores.items():
        cluster_data.append({
            'cluster': f'ViT_{cluster}',
            'type': 'ViT',
            'mean_score': stats['mean_score'],
            'std_score': stats['std_score'],
            'n_points': stats['n_points']
        })

    for cluster, stats in spatial_cluster_scores.items():
        cluster_data.append({
            'cluster': f'Spatial_{cluster}',
            'type': 'Spatial',
            'mean_score': stats['mean_score'],
            'std_score': stats['std_score'],
            'n_points': stats['n_points']
        })

    cluster_df = pd.DataFrame(cluster_data)

    # Create the comparison plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')

    # Overall comparison.
    methods = ['ViT Clustering', 'Spatial Clustering']
    scores = [vit_silhouette, spatial_silhouette]
    colors = ['#2E86AB', '#A23B72']

    bars = ax1.bar(methods, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Overall Silhouette Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Silhouette Score', fontsize=12)
    ax1.set_ylim(0, max(scores) * 1.2)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars.
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # Per-cluster comparison.
    vit_data = cluster_df[cluster_df['type'] == 'ViT']
    spatial_data = cluster_df[cluster_df['type'] == 'Spatial']

    x_pos_vit = np.arange(len(vit_data))
    x_pos_spatial = np.arange(len(spatial_data)) + len(vit_data) + 1

    ax2.bar(x_pos_vit, vit_data['mean_score'], yerr=vit_data['std_score'],
           color='#2E86AB', alpha=0.8, label='ViT Clusters', capsize=5)
    ax2.bar(x_pos_spatial, spatial_data['mean_score'], yerr=spatial_data['std_score'],
           color='#A23B72', alpha=0.8, label='Spatial Clusters', capsize=5)

    # Set labels.
    all_labels = list(vit_data['cluster']) + [''] + list(spatial_data['cluster'])
    ax2.set_xticks(list(x_pos_vit) + [len(vit_data)] + list(x_pos_spatial))
    ax2.set_xticklabels(all_labels, rotation=45, ha='right')
    ax2.set_title('Per-Cluster Silhouette Scores', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Cluster Quality Assessment', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    console.print(f"[green]✓[/green] Silhouette comparison saved to {output_path}")


def create_metrics_summary_plot(metrics_data: Dict[str, Any], output_path: Path) -> None:
    """
    Create comprehensive metrics summary visualization.

    Args:
        metrics_data: Complete metrics analysis results.
        output_path: Path for saving the summary plot.

    This function creates a dashboard-style visualization summarizing
    all key alignment metrics with confidence intervals and interpretations.
    """
    console.print("[cyan]Creating comprehensive metrics summary...[/cyan]")

    # Create figure with subplots.
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ARI with confidence intervals.
    ax1 = fig.add_subplot(gs[0, 0])
    ari_data = metrics_data['ari']
    ax1.bar(['ARI'], [ari_data['ari_score']], color='#2E86AB', alpha=0.8)
    ax1.errorbar(['ARI'], [ari_data['ari_score']],
                yerr=[[ari_data['ari_score'] - ari_data['ci_lower']],
                      [ari_data['ci_upper'] - ari_data['ari_score']]],
                fmt='none', color='black', capsize=10, capthick=2)
    ax1.set_title('Adjusted Rand Index', fontweight='bold')
    ax1.set_ylabel('ARI Score')
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)

    # Add interpretation text.
    if ari_data['ari_score'] > 0.7:
        interpretation = "Strong\nAlignment"
        color = 'green'
    elif ari_data['ari_score'] > 0.3:
        interpretation = "Moderate\nAlignment"
        color = 'orange'
    else:
        interpretation = "Weak\nAlignment"
        color = 'red'

    ax1.text(0, ari_data['ari_score'] + 0.1, interpretation,
            ha='center', va='bottom', fontweight='bold', color=color)

    # NMI with significance.
    ax2 = fig.add_subplot(gs[0, 1])
    nmi_data = metrics_data['nmi']
    bar_color = '#A23B72' if nmi_data['p_value'] < 0.05 else '#CCCCCC'
    ax2.bar(['NMI'], [nmi_data['nmi_score']], color=bar_color, alpha=0.8)
    ax2.set_title('Normalized Mutual Information', fontweight='bold')
    ax2.set_ylabel('NMI Score')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)

    # Add significance indicator.
    significance = "Significant" if nmi_data['p_value'] < 0.05 else "Not Significant"
    ax2.text(0, nmi_data['nmi_score'] + 0.05, f"p = {nmi_data['p_value']:.3f}\n{significance}",
            ha='center', va='bottom', fontsize=10)

    # Effect sizes.
    ax3 = fig.add_subplot(gs[0, 2])
    effect_data = metrics_data['effect_sizes']
    effects = ['Cohen\'s κ', 'Cramer\'s V']
    values = [effect_data['cohens_kappa'], effect_data['cramers_v']]
    colors = ['#F18F01', '#C73E1D']

    bars = ax3.bar(effects, values, color=colors, alpha=0.8)
    ax3.set_title('Effect Sizes', fontweight='bold')
    ax3.set_ylabel('Effect Size')
    ax3.set_ylim(0, max(values) * 1.2)
    ax3.grid(True, alpha=0.3)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Confusion matrix visualization (simplified).
    ax4 = fig.add_subplot(gs[1, :])
    cm_data = metrics_data['confusion_matrix']
    cm = np.array(cm_data['confusion_matrix'])

    # Normalize for better visualization.
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero.

    im = ax4.imshow(cm_norm, cmap='Blues', aspect='auto')
    ax4.set_title('Cluster Correspondence Matrix (Normalized)', fontweight='bold')
    ax4.set_xlabel('ViT Clusters')
    ax4.set_ylabel('Spatial Clusters')

    # Add colorbar.
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion')

    # Metadata summary.
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    # Create summary text.
    metadata = metrics_data['metadata']
    summary_text = f"""
    Analysis Summary:
    • Total Data Points: {metadata['n_points']:,}
    • ViT Clusters: {metadata['n_vit_clusters']}
    • Spatial Clusters: {metadata['n_spatial_clusters']}
    • Samples: {', '.join(metadata['samples'])}
    • Overall Accuracy: {cm_data['overall_accuracy']:.3f}

    Key Findings:
    • ARI Score: {ari_data['ari_score']:.3f} (95% CI: {ari_data['ci_lower']:.3f}-{ari_data['ci_upper']:.3f})
    • NMI Score: {nmi_data['nmi_score']:.3f} (p-value: {nmi_data['p_value']:.3f})
    • Cohen's Kappa: {effect_data['cohens_kappa']:.3f}
    • Cramer's V: {effect_data['cramers_v']:.3f}
    """

    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle('Cluster Alignment Analysis - Comprehensive Summary',
                fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    console.print(f"[green]✓[/green] Metrics summary plot saved to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive cluster comparison visualizations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--metrics_data', type=Path, required=True,
                       help='Path to alignment metrics JSON file')
    parser.add_argument('--cluster_data', type=Path, required=True,
                       help='Path to ViT cluster assignments CSV')
    parser.add_argument('--spatial_data', type=Path, required=True,
                       help='Path to spatial metadata CSV')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for visualizations')
    parser.add_argument('--samples', nargs='+', default=['IRI1', 'IRI2', 'IRI3'],
                       help='Sample names to include')
    parser.add_argument('--cluster_column', default='figure_idents',
                       help='Column name for spatial clusters')

    args = parser.parse_args()

    try:
        console.print(f"[1m[36mCluster Comparison Visualization Suite[/36m[/1m")
        console.print(f"[blue]ℹ[/blue] Samples: [1m{', '.join(args.samples)}[/1m")

        # Create output directories.
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'cluster_maps').mkdir(exist_ok=True)
        (args.output / 'sankey_diagrams').mkdir(exist_ok=True)
        (args.output / 'scatter_plots').mkdir(exist_ok=True)
        (args.output / 'heatmaps').mkdir(exist_ok=True)

        # Load data.
        console.print(f"[cyan]Loading metrics data from {args.metrics_data}...[/cyan]")
        with open(args.metrics_data, 'r') as f:
            metrics_data = json.load(f)

        console.print(f"[cyan]Loading cluster data from {args.cluster_data}...[/cyan]")
        vit_data = pd.read_csv(args.cluster_data)
        vit_filtered = vit_data[vit_data['sample'].isin(args.samples)]

        console.print(f"[cyan]Loading spatial data from {args.spatial_data}...[/cyan]")
        spatial_data = pd.read_csv(args.spatial_data)
        spatial_filtered = spatial_data[spatial_data['sample'].isin(args.samples)]

        console.print(f"[green]✓[/green] Loaded data for {len(args.samples)} samples")

        # Generate visualizations.
        visualizations = [
            ("Side-by-side cluster maps", create_side_by_side_cluster_maps,
             (vit_filtered, spatial_filtered, args.samples, args.output / 'cluster_maps' / 'cluster_comparison_maps.png')),

            ("Sankey cluster flow diagram", create_sankey_diagram,
             (metrics_data['confusion_matrix'], args.output / 'sankey_diagrams' / 'sankey_cluster_flow.html')),

            ("Cluster centroid scatter plot", create_centroid_scatter_plot,
             (vit_filtered, spatial_filtered, args.samples, args.output / 'scatter_plots' / 'centroid_scatter.png')),

            ("Cluster overlap heatmap", create_overlap_heatmap,
             (metrics_data['confusion_matrix'], args.output / 'heatmaps' / 'overlap_heatmap.png')),

            ("Silhouette score comparison", create_silhouette_comparison,
             (metrics_data['silhouette_analysis'], args.output / 'silhouette_comparison.png')),

            ("Comprehensive metrics summary", create_metrics_summary_plot,
             (metrics_data, args.output / 'metrics_summary_dashboard.png'))
        ]

        # Create all visualizations.
        for description, func, args_tuple in track(visualizations, description="Creating visualizations"):
            try:
                func(*args_tuple)
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to create {description}: {e}")

        # Display summary table.
        table = Table(title="Visualization Summary", style="cyan")
        table.add_column("Visualization", style="white")
        table.add_column("Output File", style="green")
        table.add_column("Status", style="yellow")

        for description, _, args_tuple in visualizations:
            output_file = args_tuple[-1]  # Last argument is always output path.
            status = "✓ Created" if output_file.exists() else "✗ Failed"
            table.add_row(description, str(output_file.name), status)

        console.print(table)
        console.print(f"[green]✓[/green] [1mVisualization suite complete![/1m")
        console.print(f"[blue]ℹ[/blue] All visualizations saved to {args.output}")

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
