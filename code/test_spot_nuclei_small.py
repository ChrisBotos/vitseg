#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_spot_nuclei_small.py.
Description:
    Test the spot-nuclei clustering pipeline with a small subset of data
    to verify everything works correctly before running on the full dataset.
    Uses first 5000 nuclei for quick testing.

Dependencies:
    • Python >= 3.10.
    • pandas, numpy, rich, scikit-learn, matplotlib.

Usage:
    python code/test_spot_nuclei_small.py
"""

import traceback
import subprocess
import sys
from pathlib import Path
from rich.console import Console

console = Console()


def run_small_test():
    """
    Run the spot-nuclei clustering pipeline with a small subset of data.
    
    This function tests the pipeline with 5000 nuclei to ensure everything
    works correctly before running on the full 628k nuclei dataset.
    """
    console.print("[cyan][1m Testing Spot-Nuclei Clustering Pipeline [/1m[/cyan]")
    console.print()
    
    # Create small test data files.
    console.print("[cyan]Creating small test dataset (5000 nuclei)...[/cyan]")
    
    # Read and sample nuclei coordinates.
    import pandas as pd
    
    coords_path = Path("results/IRI_regist_14k/coords_IRI_regist_binary_mask.csv")
    features_path = Path("results/IRI_regist_14k/features_IRI_regist_binary_mask.csv")
    
    if not coords_path.exists() or not features_path.exists():
        console.print(f"[red]✗[/red] Required files not found:")
        console.print(f"  • {coords_path}")
        console.print(f"  • {features_path}")
        return False
    
    # Sample first 5000 rows.
    coords_df = pd.read_csv(coords_path, nrows=5000)
    features_df = pd.read_csv(features_path, nrows=5000)
    
    # Create test directory.
    test_dir = Path("results/test_spot_nuclei_small")
    test_dir.mkdir(exist_ok=True)
    
    # Save test files.
    test_coords_path = test_dir / "coords_test.csv"
    test_features_path = test_dir / "features_test.csv"
    
    coords_df.to_csv(test_coords_path, index=False)
    features_df.to_csv(test_features_path, index=False)
    
    console.print(f"[green]✓[/green] Created test files with {len(coords_df):,} nuclei")
    
    # Run the clustering pipeline.
    console.print()
    console.print("[cyan]Running spot-nuclei clustering pipeline...[/cyan]")
    
    cmd = [
        sys.executable, "code/cluster_spots_by_nuclei_features.py",
        "--nuclei_coords", str(test_coords_path),
        "--nuclei_features", str(test_features_path),
        "--spots_metadata", "data/metadata_complete.csv",
        "--output", str(test_dir / "clustering_results"),
        "--samples", "IRI1", "IRI2", "IRI3",
        "--clusters", "10",  # Fewer clusters for small test.
        "--max_distance", "1000",
        "--aggregation_method", "mean"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout.
        
        if result.returncode == 0:
            console.print("[green]✓[/green] Pipeline completed successfully!")
            
            # Check output files.
            output_dir = test_dir / "clustering_results"
            expected_files = [
                "spot_nuclei_assignments.csv",
                "spot_aggregated_features.csv", 
                "spot_clusters.csv",
                "spot_cluster_visualization.png",
                "spot_cluster_stats.txt"
            ]
            
            console.print()
            console.print("[cyan]Checking output files...[/cyan]")
            
            all_files_exist = True
            for filename in expected_files:
                filepath = output_dir / filename
                if filepath.exists():
                    console.print(f"[green]✓[/green] {filename}")
                else:
                    console.print(f"[red]✗[/red] {filename} - Missing!")
                    all_files_exist = False
            
            if all_files_exist:
                # Show some basic statistics.
                console.print()
                console.print("[cyan]Pipeline Results Summary:[/cyan]")
                
                # Read assignments.
                assignments_df = pd.read_csv(output_dir / "spot_nuclei_assignments.csv")
                console.print(f"[blue]ℹ[/blue] Nuclei assigned to spots: {len(assignments_df):,}")
                
                # Read aggregated features.
                features_df = pd.read_csv(output_dir / "spot_aggregated_features.csv")
                console.print(f"[blue]ℹ[/blue] Spots with aggregated features: {len(features_df):,}")
                
                # Read clusters.
                clusters_df = pd.read_csv(output_dir / "spot_clusters.csv")
                n_clusters = clusters_df['cluster'].nunique()
                console.print(f"[blue]ℹ[/blue] Number of clusters: {n_clusters}")
                
                # Show assignment rate.
                assignment_rate = len(assignments_df) / len(coords_df) * 100
                console.print(f"[blue]ℹ[/blue] Assignment success rate: {assignment_rate:.1f}%")
                
                console.print()
                console.print("[green][1m ✓ Small scale test completed successfully! [/1m[/green]")
                console.print("[green]The pipeline is ready for full-scale analysis.[/green]")
                
                return True
            else:
                console.print()
                console.print("[red][1m ✗ Some output files are missing [/1m[/red]")
                return False
                
        else:
            console.print(f"[red]✗[/red] Pipeline failed with return code {result.returncode}")
            console.print(f"[red]Error output:[/red]")
            console.print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        console.print("[red]✗[/red] Pipeline timed out after 5 minutes")
        return False
    except Exception as e:
        console.print(f"[red]✗[/red] Error running pipeline: {e}")
        return False


def main():
    """Run the small scale test and print suggested full-analysis command."""
    try:
        success = run_small_test()
        
        if success:
            console.print()
            console.print("[cyan][1m Ready for Full Analysis [/1m[/cyan]")
            console.print()
            console.print("You can now run the full analysis with:")
            console.print()
            console.print("[yellow]python code/cluster_spots_by_nuclei_features.py \\[/yellow]")
            console.print("[yellow]    --nuclei_coords results/IRI_regist_14k/coords_IRI_regist_binary_mask.csv \\[/yellow]")
            console.print("[yellow]    --nuclei_features results/IRI_regist_14k/features_IRI_regist_binary_mask.csv \\[/yellow]")
            console.print("[yellow]    --spots_metadata data/metadata_complete.csv \\[/yellow]")
            console.print("[yellow]    --output results/spot_nuclei_clustering_14k \\[/yellow]")
            console.print("[yellow]    --samples IRI1 IRI2 IRI3 \\[/yellow]")
            console.print("[yellow]    --clusters 15 \\[/yellow]")
            console.print("[yellow]    --max_distance 1000 \\[/yellow]")
            console.print("[yellow]    --aggregation_method mean[/yellow]")
            
        else:
            console.print()
            console.print("[red][1m ✗ Small scale test failed [/1m[/red]")
            console.print("[yellow]Please check the error messages above and fix any issues.[/yellow]")
            
        return 0 if success else 1
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error during test: {e}")
        console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())
