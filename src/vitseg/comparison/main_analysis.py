"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: main_analysis.py.
Description:
    Main orchestration script for comprehensive ViT-spatial cluster comparison
    analysis. Coordinates the complete workflow including data loading, alignment
    metrics calculation, spatial analysis, visualization generation, and
    comprehensive report creation.

Dependencies:
    • Python >= 3.10.
    • All modules in vitseg.comparison.
    • pandas, numpy, scipy, scikit-learn.
    • rich (for enhanced console output).

Usage:
    python -m vitseg.comparison.main_analysis \
        --vit_clusters results/spot_nuclei_clustering/spot_clusters.csv \
        --spatial_clusters data/metadata_complete.csv \
        --output comparison_analysis/results/complete_analysis \
        --samples IRI1 IRI2 IRI3 \
        --cluster_column figure_idents
"""
import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import subprocess
import sys

import pandas as pd
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table
from rich.panel import Panel

console = Console()


def run_subprocess_analysis(module_name: str, args: List[str], description: str) -> bool:
    """Run a subprocess analysis module with error handling.

    Args:
        module_name (str): Fully qualified module name (e.g. 'vitseg.comparison.cluster_metrics').
        args (list[str]): List of command-line arguments for the module.
        description (str): Description of the analysis for progress tracking.

    Returns:
        bool: Whether the subprocess completed successfully.
    """
    console.print(f"[cyan]Running {description}...[/cyan]")

    try:
        # Construct the command using -m module invocation.
        cmd = [sys.executable, "-m", module_name] + args
        
        # Run the subprocess.
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        console.print(f"[green]✓[/green] {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗[/red] {description} failed with return code {e.returncode}")
        console.print(f"[red]Error output:[/red] {e.stderr}")
        return False
    except Exception as e:
        console.print(f"[red]✗[/red] {description} failed with error: {e}")
        return False


def validate_input_files(vit_clusters: Path, spatial_clusters: Path, samples: List[str]) -> bool:
    """
    Validate that all required input files exist and contain expected data.
    
    Args:
        vit_clusters: Path to ViT cluster assignments.
        spatial_clusters: Path to spatial metadata.
        samples: List of sample names to validate.
        
    Returns:
        Boolean indicating whether all validations passed.
        
    This function performs comprehensive input validation to prevent
    analysis failures due to missing or malformed data.
    """
    console.print("[cyan]Validating input files...[/cyan]")
    
    # Check file existence.
    if not vit_clusters.exists():
        console.print(f"[red]✗[/red] ViT clusters file not found: {vit_clusters}")
        return False
    
    if not spatial_clusters.exists():
        console.print(f"[red]✗[/red] Spatial clusters file not found: {spatial_clusters}")
        return False
    
    try:
        # Load and validate ViT data.
        vit_data = pd.read_csv(vit_clusters)
        required_vit_cols = ['spot_x', 'spot_y', 'sample', 'cluster']
        missing_vit_cols = [col for col in required_vit_cols if col not in vit_data.columns]
        
        if missing_vit_cols:
            console.print(f"[red]✗[/red] ViT data missing columns: {missing_vit_cols}")
            return False
        
        # Check for samples in ViT data.
        vit_samples = set(vit_data['sample'].unique())
        missing_vit_samples = [s for s in samples if s not in vit_samples]
        
        if missing_vit_samples:
            console.print(f"[yellow]⚠[/yellow] ViT data missing samples: {missing_vit_samples}")
        
        # Load and validate spatial data.
        spatial_data = pd.read_csv(spatial_clusters)
        required_spatial_cols = ['x', 'y', 'sample']
        missing_spatial_cols = [col for col in required_spatial_cols if col not in spatial_data.columns]
        
        if missing_spatial_cols:
            console.print(f"[red]✗[/red] Spatial data missing columns: {missing_spatial_cols}")
            return False
        
        # Check for samples in spatial data.
        spatial_samples = set(spatial_data['sample'].unique())
        missing_spatial_samples = [s for s in samples if s not in spatial_samples]
        
        if missing_spatial_samples:
            console.print(f"[yellow]⚠[/yellow] Spatial data missing samples: {missing_spatial_samples}")
        
        console.print(f"[green]✓[/green] Input validation passed")
        console.print(f"[blue]ℹ[/blue] ViT data: {len(vit_data):,} points, {len(vit_samples)} samples")
        console.print(f"[blue]ℹ[/blue] Spatial data: {len(spatial_data):,} points, {len(spatial_samples)} samples")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗[/red] Input validation failed: {e}")
        return False


def create_analysis_directories(output_dir: Path) -> Dict[str, Path]:
    """
    Create all necessary output directories for the analysis.
    
    Args:
        output_dir: Base output directory.
        
    Returns:
        Dictionary mapping directory names to Path objects.
        
    This function ensures all required directories exist before
    starting the analysis workflow.
    """
    console.print("[cyan]Creating analysis directories...[/cyan]")
    
    directories = {
        'base': output_dir,
        'metrics': output_dir / 'metrics',
        'spatial': output_dir / 'spatial',
        'visualizations': output_dir / 'visualizations',
        'reports': output_dir / 'reports'
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created {name} directory: {path}")
    
    return directories


def run_complete_analysis(vit_clusters: Path, spatial_clusters: Path, output_dirs: Dict[str, Path],
                         samples: List[str], cluster_column: str) -> Dict[str, bool]:
    """
    Execute the complete analysis workflow.
    
    Args:
        vit_clusters: Path to ViT cluster assignments.
        spatial_clusters: Path to spatial metadata.
        output_dirs: Dictionary of output directories.
        samples: List of sample names.
        cluster_column: Column name for spatial clusters.
        
    Returns:
        Dictionary indicating success/failure of each analysis step.
        
    This function orchestrates the complete analysis workflow,
    running each component in the correct order with proper dependencies.
    """
    console.print("[cyan]Starting complete analysis workflow...[/cyan]")
    
    results = {}

    # Step 1: Calculate alignment metrics.
    metrics_args = [
        '--vit_data', str(vit_clusters),
        '--spatial_data', str(spatial_clusters),
        '--output', str(output_dirs['metrics']),
        '--samples'] + samples + [
        '--cluster_column', cluster_column
    ]

    results['metrics'] = run_subprocess_analysis(
        'vitseg.comparison.cluster_metrics',
        metrics_args,
        "Cluster alignment metrics calculation"
    )

    # Step 2: Spatial correlation analysis.
    spatial_args = [
        '--cluster_data', str(vit_clusters),
        '--output', str(output_dirs['spatial']),
        '--samples'] + samples

    results['spatial'] = run_subprocess_analysis(
        'vitseg.comparison.spatial_analysis',
        spatial_args,
        "Spatial correlation analysis"
    )

    # Step 3: Generate visualizations (only if metrics succeeded).
    if results['metrics']:
        viz_args = [
            '--metrics_data', str(output_dirs['metrics'] / 'alignment_metrics.json'),
            '--cluster_data', str(vit_clusters),
            '--spatial_data', str(spatial_clusters),
            '--output', str(output_dirs['visualizations']),
            '--samples'] + samples + [
            '--cluster_column', cluster_column
        ]

        results['visualizations'] = run_subprocess_analysis(
            'vitseg.comparison.visualization_suite',
            viz_args,
            "Visualization generation"
        )
    else:
        console.print("[yellow]⚠[/yellow] Skipping visualizations due to metrics failure")
        results['visualizations'] = False
    
    return results


def create_analysis_summary(output_dirs: Dict[str, Path], analysis_results: Dict[str, bool],
                           samples: List[str]) -> None:
    """
    Create comprehensive analysis summary and index.
    
    Args:
        output_dirs: Dictionary of output directories.
        analysis_results: Results of each analysis step.
        samples: List of sample names analyzed.
        
    This function creates a comprehensive summary of all analysis results
    with links to detailed outputs and interpretive guidance.
    """
    console.print("[cyan]Creating analysis summary...[/cyan]")
    
    summary_path = output_dirs['base'] / 'analysis_summary.md'
    
    with open(summary_path, 'w') as f:
        f.write("# ViT-Spatial Cluster Comparison Analysis Summary\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Samples Analyzed:** {', '.join(samples)}\n\n")
        
        # Analysis status.
        f.write("## Analysis Status\n\n")
        for step, success in analysis_results.items():
            status = "✅ Completed" if success else "❌ Failed"
            f.write(f"- **{step.title()}:** {status}\n")
        f.write("\n")
        
        # Results overview.
        f.write("## Results Overview\n\n")
        
        if analysis_results.get('metrics', False):
            f.write("### Statistical Alignment Metrics\n")
            f.write("- **Location:** `metrics/alignment_metrics.json`\n")
            f.write("- **Summary:** `metrics/metrics_summary.txt`\n")
            f.write("- **Confusion Matrix:** `metrics/confusion_matrix.csv`\n")
            f.write("- **Silhouette Analysis:** `metrics/silhouette_analysis.csv`\n\n")
        
        if analysis_results.get('spatial', False):
            f.write("### Spatial Correlation Analysis\n")
            f.write("- **Moran's I Results:** `spatial/morans_i_results.json`\n")
            f.write("- **Autocorrelation Summary:** `spatial/spatial_autocorr.csv`\n")
            f.write("- **LISA Analysis:** `spatial/lisa_analysis.csv`\n")
            f.write("- **Summary Report:** `spatial/spatial_summary.txt`\n\n")
        
        if analysis_results.get('visualizations', False):
            f.write("### Visualizations\n")
            f.write("- **Cluster Comparison Maps:** `visualizations/cluster_maps/`\n")
            f.write("- **Sankey Diagrams:** `visualizations/sankey_diagrams/`\n")
            f.write("- **Scatter Plots:** `visualizations/scatter_plots/`\n")
            f.write("- **Heatmaps:** `visualizations/heatmaps/`\n")
            f.write("- **Metrics Dashboard:** `visualizations/metrics_summary_dashboard.png`\n\n")
        
        # Interpretation guidelines.
        f.write("## Interpretation Guidelines\n\n")
        f.write("### Alignment Metrics\n")
        f.write("- **ARI > 0.7:** Strong cluster alignment\n")
        f.write("- **ARI 0.3-0.7:** Moderate alignment\n")
        f.write("- **ARI < 0.3:** Weak alignment\n")
        f.write("- **NMI > 0.8:** High information overlap\n")
        f.write("- **p-value < 0.05:** Statistically significant alignment\n\n")
        
        f.write("### Spatial Metrics\n")
        f.write("- **Moran's I > 0:** Positive spatial autocorrelation (clustering)\n")
        f.write("- **Moran's I < 0:** Negative spatial autocorrelation (dispersion)\n")
        f.write("- **p-value < 0.05:** Significant spatial pattern\n\n")
        
        # Next steps.
        f.write("## Next Steps\n\n")
        f.write("1. Review the metrics summary for overall alignment assessment\n")
        f.write("2. Examine spatial analysis for tissue organization patterns\n")
        f.write("3. Use visualizations for publication and presentation\n")
        f.write("4. Consider biological interpretation of cluster correspondences\n")
        f.write("5. Validate findings with additional samples or methods\n\n")
    
    console.print(f"[green]✓[/green] Analysis summary saved to {summary_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Complete ViT-spatial cluster comparison analysis.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--vit_clusters', type=Path, required=True,
                       help='Path to ViT cluster assignments CSV')
    parser.add_argument('--spatial_clusters', type=Path, required=True,
                       help='Path to spatial metadata CSV')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for complete analysis')
    parser.add_argument('--samples', nargs='+', default=['IRI1', 'IRI2', 'IRI3'],
                       help='Sample names to include')
    parser.add_argument('--cluster_column', default='figure_idents',
                       help='Column name for spatial clusters')
    
    args = parser.parse_args()
    
    try:
        # Display header.
        header = Panel.fit(
            "[bold cyan]ViT-Spatial Cluster Comparison Analysis[/bold cyan]\n"
            "[dim]Comprehensive statistical and spatial analysis suite[/dim]",
            border_style="cyan"
        )
        console.print(header)
        console.print(f"[blue]ℹ[/blue] Samples: [bold]{', '.join(args.samples)}[/bold]")
        console.print(f"[blue]ℹ[/blue] Spatial clusters: [bold]{args.cluster_column}[/bold]")
        console.print(f"[blue]ℹ[/blue] Output directory: [bold]{args.output}[/bold]")
        
        # Validate inputs.
        if not validate_input_files(args.vit_clusters, args.spatial_clusters, args.samples):
            console.print("[red]✗[/red] Input validation failed. Aborting analysis.")
            return 1
        
        # Create output directories.
        output_dirs = create_analysis_directories(args.output)
        
        # Run complete analysis.
        analysis_results = run_complete_analysis(
            args.vit_clusters, args.spatial_clusters, output_dirs,
            args.samples, args.cluster_column
        )
        
        # Create summary.
        create_analysis_summary(output_dirs, analysis_results, args.samples)
        
        # Display final results.
        success_count = sum(analysis_results.values())
        total_count = len(analysis_results)
        
        table = Table(title="Analysis Results Summary", style="cyan")
        table.add_column("Component", style="white")
        table.add_column("Status", style="green")
        table.add_column("Output Location", style="blue")
        
        component_dirs = {
            'metrics': 'metrics/',
            'spatial': 'spatial/',
            'visualizations': 'visualizations/'
        }
        
        for component, success in analysis_results.items():
            status = "✅ Success" if success else "❌ Failed"
            location = component_dirs.get(component, "N/A")
            table.add_row(component.title(), status, location)
        
        console.print(table)
        
        if success_count == total_count:
            console.print(f"[green]✓[/green] [bold]Complete analysis successful![/bold]")
            console.print(f"[blue]ℹ[/blue] All results saved to {args.output}")
        else:
            console.print(f"[yellow]⚠[/yellow] Analysis completed with {total_count - success_count} failures")
            console.print(f"[blue]ℹ[/blue] Partial results saved to {args.output}")
        
        return 0 if success_count == total_count else 1
        
    except Exception as e:
        console.print(f"[red]✗[/red] Analysis failed: {e}")
        console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())
