"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: cluster_comparison_simple.py.
Description:
    Runs a simplified cluster comparison analysis between ViT-derived clusters
    and spatial transcriptomics clusters, avoiding Unicode encoding issues on
    Windows systems.

Dependencies:
    • Python >= 3.10.
    • pandas, numpy, matplotlib, seaborn, scikit-learn.

Usage:
    python -m vitseg.utilities.cluster_comparison_simple \
        --vit_clusters results/IRI_regist_14k/vit_clusters_formatted.csv \
        --spatial_data data/metadata_complete.csv \
        --output results/simple_cluster_comparison \
        --samples IRI1 IRI2 IRI3 \
        --cluster_column figure_idents
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KDTree
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless operation.
plt.switch_backend('Agg')


def load_and_prepare_data(vit_path: Path, spatial_path: Path, samples: list, cluster_column: str):
    """
    Load and prepare ViT and spatial data for comparison.
    
    Args:
        vit_path: Path to ViT cluster data.
        spatial_path: Path to spatial metadata.
        samples: List of sample names to include.
        cluster_column: Column name for spatial clusters.
        
    Returns:
        Tuple of (vit_data, spatial_data, merged_data).
    """
    print("Loading data...")
    
    # Load ViT data.
    vit_data = pd.read_csv(vit_path)
    vit_data = vit_data[vit_data['sample'].isin(samples)]
    print(f"ViT data: {len(vit_data):,} points from {len(vit_data['sample'].unique())} samples")
    
    # Load spatial data.
    spatial_data = pd.read_csv(spatial_path)
    spatial_data = spatial_data[spatial_data['sample'].isin(samples)]
    print(f"Spatial data: {len(spatial_data):,} points from {len(spatial_data['sample'].unique())} samples")
    
    # Merge datasets on coordinates using efficient KDTree matching.
    print("Merging datasets on coordinates using KDTree...")
    merged_data = []

    for sample in samples:
        vit_sample = vit_data[vit_data['sample'] == sample]
        spatial_sample = spatial_data[spatial_data['sample'] == sample]

        print(f"Processing {sample}: {len(vit_sample):,} ViT points, {len(spatial_sample):,} spatial points")

        if len(vit_sample) == 0 or len(spatial_sample) == 0:
            continue

        # Build KDTree for efficient spatial matching.
        vit_coords = vit_sample[['spot_x', 'spot_y']].values
        tree = KDTree(vit_coords)

        # Find closest ViT point for each spatial point.
        spatial_coords = spatial_sample[['x', 'y']].values
        distances, indices = tree.query(spatial_coords, k=1)

        # Create matched pairs.
        for i, (_, spatial_point) in enumerate(spatial_sample.iterrows()):
            distance = distances[i][0]
            vit_idx = vit_sample.iloc[indices[i][0]]

            # Only include if distance is reasonable (< 100 units).
            if distance < 100:
                merged_data.append({
                    'sample': sample,
                    'x': spatial_point['x'],
                    'y': spatial_point['y'],
                    'vit_cluster': vit_idx['cluster'],
                    'spatial_cluster': spatial_point[cluster_column],
                    'distance': distance
                })
    
    merged_df = pd.DataFrame(merged_data)
    print(f"Merged data: {len(merged_df):,} matched points")
    
    return vit_data, spatial_data, merged_df


def calculate_cluster_metrics(merged_data: pd.DataFrame):
    """
    Calculate cluster comparison metrics.
    
    Args:
        merged_data: DataFrame with matched ViT and spatial clusters.
        
    Returns:
        Dictionary of metrics.
    """
    print("Calculating cluster metrics...")
    
    # Encode cluster labels.
    le_vit = LabelEncoder()
    le_spatial = LabelEncoder()
    
    vit_labels = le_vit.fit_transform(merged_data['vit_cluster'])
    spatial_labels = le_spatial.fit_transform(merged_data['spatial_cluster'])
    
    # Calculate metrics.
    ari = adjusted_rand_score(spatial_labels, vit_labels)
    nmi = normalized_mutual_info_score(spatial_labels, vit_labels)
    
    # Calculate per-sample metrics.
    sample_metrics = {}
    for sample in merged_data['sample'].unique():
        sample_data = merged_data[merged_data['sample'] == sample]
        if len(sample_data) > 10:  # Minimum points for meaningful metrics.
            sample_vit = le_vit.transform(sample_data['vit_cluster'])
            sample_spatial = le_spatial.transform(sample_data['spatial_cluster'])
            
            sample_metrics[sample] = {
                'ari': adjusted_rand_score(sample_spatial, sample_vit),
                'nmi': normalized_mutual_info_score(sample_spatial, sample_vit),
                'n_points': len(sample_data),
                'n_vit_clusters': len(sample_data['vit_cluster'].unique()),
                'n_spatial_clusters': len(sample_data['spatial_cluster'].unique())
            }
    
    metrics = {
        'overall_ari': ari,
        'overall_nmi': nmi,
        'n_total_points': len(merged_data),
        'n_vit_clusters': len(merged_data['vit_cluster'].unique()),
        'n_spatial_clusters': len(merged_data['spatial_cluster'].unique()),
        'sample_metrics': sample_metrics
    }
    
    return metrics


def create_confusion_matrix_plot(merged_data: pd.DataFrame, output_dir: Path):
    """
    Create confusion matrix visualization.

    Args:
        merged_data: DataFrame with matched clusters.
        output_dir: Output directory for plots.
    """
    print("Creating confusion matrix plot...")

    # Encode labels to handle mixed types.
    le_spatial = LabelEncoder()
    le_vit = LabelEncoder()

    spatial_encoded = le_spatial.fit_transform(merged_data['spatial_cluster'])
    vit_encoded = le_vit.fit_transform(merged_data['vit_cluster'])

    # Create confusion matrix.
    cm = confusion_matrix(spatial_encoded, vit_encoded)

    # Get unique labels.
    spatial_labels = le_spatial.classes_
    vit_labels = le_vit.classes_
    
    # Create plot.
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=vit_labels, yticklabels=spatial_labels)
    plt.title('Confusion Matrix: Spatial vs ViT Clusters')
    plt.xlabel('ViT Clusters')
    plt.ylabel('Spatial Clusters')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_cluster_distribution_plot(merged_data: pd.DataFrame, output_dir: Path):
    """
    Create cluster distribution plots.
    
    Args:
        merged_data: DataFrame with matched clusters.
        output_dir: Output directory for plots.
    """
    print("Creating cluster distribution plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ViT cluster distribution.
    vit_counts = merged_data['vit_cluster'].value_counts().sort_index()
    axes[0, 0].bar(range(len(vit_counts)), vit_counts.values)
    axes[0, 0].set_title('ViT Cluster Distribution')
    axes[0, 0].set_xlabel('Cluster ID')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xticks(range(len(vit_counts)))
    axes[0, 0].set_xticklabels(vit_counts.index)
    
    # Spatial cluster distribution.
    spatial_counts = merged_data['spatial_cluster'].value_counts().sort_index()
    axes[0, 1].bar(range(len(spatial_counts)), spatial_counts.values)
    axes[0, 1].set_title('Spatial Cluster Distribution')
    axes[0, 1].set_xlabel('Cluster ID')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticks(range(len(spatial_counts)))
    axes[0, 1].set_xticklabels(spatial_counts.index)
    
    # Sample distribution.
    sample_counts = merged_data['sample'].value_counts()
    axes[1, 0].bar(sample_counts.index, sample_counts.values)
    axes[1, 0].set_title('Points per Sample')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Count')
    
    # Distance distribution.
    axes[1, 1].hist(merged_data['distance'], bins=50, alpha=0.7)
    axes[1, 1].set_title('Matching Distance Distribution')
    axes[1, 1].set_xlabel('Distance (units)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_results(metrics: dict, merged_data: pd.DataFrame, output_dir: Path):
    """
    Save analysis results to files.
    
    Args:
        metrics: Dictionary of calculated metrics.
        merged_data: DataFrame with matched data.
        output_dir: Output directory.
    """
    print("Saving results...")
    
    # Save metrics to CSV.
    metrics_df = pd.DataFrame([{
        'metric': 'Overall ARI',
        'value': metrics['overall_ari']
    }, {
        'metric': 'Overall NMI', 
        'value': metrics['overall_nmi']
    }, {
        'metric': 'Total Points',
        'value': metrics['n_total_points']
    }, {
        'metric': 'ViT Clusters',
        'value': metrics['n_vit_clusters']
    }, {
        'metric': 'Spatial Clusters',
        'value': metrics['n_spatial_clusters']
    }])
    
    metrics_df.to_csv(output_dir / 'overall_metrics.csv', index=False)
    
    # Save per-sample metrics.
    sample_metrics_list = []
    for sample, sample_data in metrics['sample_metrics'].items():
        sample_metrics_list.append({
            'sample': sample,
            'ari': sample_data['ari'],
            'nmi': sample_data['nmi'],
            'n_points': sample_data['n_points'],
            'n_vit_clusters': sample_data['n_vit_clusters'],
            'n_spatial_clusters': sample_data['n_spatial_clusters']
        })
    
    sample_df = pd.DataFrame(sample_metrics_list)
    sample_df.to_csv(output_dir / 'sample_metrics.csv', index=False)
    
    # Save merged data.
    merged_data.to_csv(output_dir / 'matched_clusters.csv', index=False)
    
    # Create summary report.
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write("ViT-Spatial Cluster Comparison Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"- Adjusted Rand Index (ARI): {metrics['overall_ari']:.4f}\n")
        f.write(f"- Normalized Mutual Information (NMI): {metrics['overall_nmi']:.4f}\n")
        f.write(f"- Total matched points: {metrics['n_total_points']:,}\n")
        f.write(f"- ViT clusters: {metrics['n_vit_clusters']}\n")
        f.write(f"- Spatial clusters: {metrics['n_spatial_clusters']}\n\n")
        
        f.write("Per-Sample Metrics:\n")
        for sample, sample_data in metrics['sample_metrics'].items():
            f.write(f"- {sample}:\n")
            f.write(f"  * ARI: {sample_data['ari']:.4f}\n")
            f.write(f"  * NMI: {sample_data['nmi']:.4f}\n")
            f.write(f"  * Points: {sample_data['n_points']:,}\n")
            f.write(f"  * ViT clusters: {sample_data['n_vit_clusters']}\n")
            f.write(f"  * Spatial clusters: {sample_data['n_spatial_clusters']}\n\n")
        
        # Interpretation.
        f.write("Interpretation:\n")
        if metrics['overall_ari'] > 0.5:
            f.write("- EXCELLENT alignment between ViT and spatial clusters\n")
        elif metrics['overall_ari'] > 0.3:
            f.write("- GOOD alignment between ViT and spatial clusters\n")
        elif metrics['overall_ari'] > 0.1:
            f.write("- MODERATE alignment between ViT and spatial clusters\n")
        else:
            f.write("- POOR alignment between ViT and spatial clusters\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Simple cluster comparison analysis")
    parser.add_argument("--vit_clusters", type=Path, required=True,
                       help="Path to ViT cluster CSV file")
    parser.add_argument("--spatial_data", type=Path, required=True,
                       help="Path to spatial metadata CSV file")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output directory for results")
    parser.add_argument("--samples", nargs='+', default=['IRI1', 'IRI2', 'IRI3'],
                       help="Sample names to include")
    parser.add_argument("--cluster_column", default='figure_idents',
                       help="Column name for spatial clusters")
    
    args = parser.parse_args()
    
    print("Simple ViT-Spatial Cluster Comparison Analysis")
    print("=" * 50)
    print(f"Samples: {', '.join(args.samples)}")
    print(f"Spatial clusters: {args.cluster_column}")
    print(f"Output: {args.output}")
    print()
    
    try:
        # Create output directory.
        args.output.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data.
        vit_data, spatial_data, merged_data = load_and_prepare_data(
            args.vit_clusters, args.spatial_data, args.samples, args.cluster_column
        )
        
        if len(merged_data) == 0:
            print("ERROR: No matched points found between datasets")
            return 1
        
        # Calculate metrics.
        metrics = calculate_cluster_metrics(merged_data)
        
        # Create visualizations.
        create_confusion_matrix_plot(merged_data, args.output)
        create_cluster_distribution_plot(merged_data, args.output)
        
        # Save results.
        save_results(metrics, merged_data, args.output)
        
        print()
        print("Analysis Results:")
        print(f"- Overall ARI: {metrics['overall_ari']:.4f}")
        print(f"- Overall NMI: {metrics['overall_nmi']:.4f}")
        print(f"- Matched points: {metrics['n_total_points']:,}")
        print()
        print(f"Results saved to: {args.output}")
        print("Analysis complete!")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
