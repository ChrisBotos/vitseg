"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_cluster_metrics.py.
Description:
    Comprehensive test suite for cluster metrics analysis module.
    Tests all statistical functions including ARI, NMI, silhouette analysis,
    and confusion matrix calculations with synthetic and real data scenarios.

Dependencies:
    • Python >= 3.10.
    • pytest, numpy, pandas, scikit-learn.
    • rich (for enhanced console output).

Usage:
    pytest comparison_analysis/tests/test_cluster_metrics.py -v
"""
import pytest
import numpy as np
import pandas as pd
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any
import tempfile
import sys

# Add the scripts directory to the path.
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from cluster_metrics import (
    load_and_align_data, align_coordinates, calculate_adjusted_rand_index,
    calculate_normalized_mutual_information, calculate_silhouette_analysis,
    create_confusion_matrix_analysis, calculate_effect_sizes
)

from rich.console import Console
console = Console()


class TestClusterMetrics:
    """Test suite for cluster metrics analysis functions."""
    
    @pytest.fixture
    def synthetic_perfect_alignment(self):
        """Create synthetic data with perfect cluster alignment."""
        np.random.seed(42)
        n_points = 1000
        n_clusters = 5
        
        # Generate cluster assignments (identical for both methods).
        cluster_labels = np.random.randint(0, n_clusters, n_points)
        
        # Generate coordinates with some spatial structure.
        coordinates = []
        for cluster in range(n_clusters):
            cluster_mask = cluster_labels == cluster
            n_cluster_points = np.sum(cluster_mask)
            
            # Create cluster centers.
            center_x = np.random.uniform(0, 1000)
            center_y = np.random.uniform(0, 1000)
            
            # Generate points around center.
            x_coords = np.random.normal(center_x, 50, n_cluster_points)
            y_coords = np.random.normal(center_y, 50, n_cluster_points)
            
            coordinates.extend(list(zip(x_coords, y_coords)))
        
        coordinates = np.array(coordinates)
        
        # Create DataFrames.
        vit_data = pd.DataFrame({
            'spot_x': coordinates[:, 0],
            'spot_y': coordinates[:, 1],
            'sample': ['TEST'] * n_points,
            'cluster': cluster_labels
        })
        
        spatial_data = pd.DataFrame({
            'x': coordinates[:, 0] + np.random.normal(0, 1, n_points),  # Small noise.
            'y': coordinates[:, 1] + np.random.normal(0, 1, n_points),
            'sample': ['TEST'] * n_points,
            'figure_idents': [f'cluster_{label}' for label in cluster_labels]
        })
        
        return vit_data, spatial_data, coordinates, cluster_labels
    
    @pytest.fixture
    def synthetic_imperfect_alignment(self):
        """Create synthetic data with imperfect cluster alignment."""
        np.random.seed(123)
        n_points = 500

        # Generate cluster assignments with the same label set so the contingency
        # table has no structurally empty rows or columns.  This avoids zero
        # expected frequencies that would cause chi2_contingency to fail.
        n_clusters = 4
        vit_clusters = np.random.randint(0, n_clusters, n_points)
        spatial_clusters = np.random.randint(0, n_clusters, n_points)

        # Add some correlation.
        correlation_mask = np.random.random(n_points) < 0.6
        spatial_clusters[correlation_mask] = vit_clusters[correlation_mask]

        coordinates = np.random.uniform(0, 1000, (n_points, 2))

        vit_data = pd.DataFrame({
            'spot_x': coordinates[:, 0],
            'spot_y': coordinates[:, 1],
            'sample': ['TEST'] * n_points,
            'cluster': vit_clusters
        })

        spatial_data = pd.DataFrame({
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'sample': ['TEST'] * n_points,
            'figure_idents': [f'spatial_{label}' for label in spatial_clusters]
        })

        return vit_data, spatial_data, coordinates, vit_clusters, spatial_clusters
    
    def test_coordinate_alignment_perfect_match(self, synthetic_perfect_alignment):
        """Test coordinate alignment with perfect matches."""
        vit_data, spatial_data, _, _ = synthetic_perfect_alignment
        
        aligned_vit, aligned_spatial = align_coordinates(vit_data, spatial_data, max_distance=10.0)
        
        # Should align most points with small noise.
        assert len(aligned_vit) > len(vit_data) * 0.8
        assert len(aligned_spatial) == len(aligned_vit)
        assert 'alignment_distance' in aligned_vit.columns
        assert np.all(aligned_vit['alignment_distance'] <= 10.0)
    
    def test_coordinate_alignment_no_matches(self):
        """Test coordinate alignment with no matches."""
        # Create data with completely different coordinate systems.
        vit_data = pd.DataFrame({
            'spot_x': [0, 1, 2],
            'spot_y': [0, 1, 2],
            'sample': ['TEST'] * 3,
            'cluster': [0, 1, 2]
        })
        
        spatial_data = pd.DataFrame({
            'x': [1000, 1001, 1002],
            'y': [1000, 1001, 1002],
            'sample': ['TEST'] * 3,
            'figure_idents': ['A', 'B', 'C']
        })
        
        with pytest.raises(ValueError, match="No coordinate matches found"):
            align_coordinates(vit_data, spatial_data, max_distance=10.0)
    
    def test_adjusted_rand_index_perfect_alignment(self, synthetic_perfect_alignment):
        """Test ARI calculation with perfect alignment."""
        _, _, _, cluster_labels = synthetic_perfect_alignment
        
        # Perfect alignment should give ARI = 1.0.
        results = calculate_adjusted_rand_index(cluster_labels, cluster_labels)
        
        assert results['ari_score'] == pytest.approx(1.0, abs=1e-10)
        assert 'ci_lower' in results
        assert 'ci_upper' in results
        assert 'bootstrap_std' in results
        assert results['ci_lower'] <= results['ari_score'] <= results['ci_upper']
    
    def test_adjusted_rand_index_random_alignment(self):
        """Test ARI calculation with random alignment."""
        np.random.seed(42)
        n_points = 1000
        
        # Generate completely random cluster assignments.
        clusters1 = np.random.randint(0, 5, n_points)
        clusters2 = np.random.randint(0, 5, n_points)
        
        results = calculate_adjusted_rand_index(clusters1, clusters2)
        
        # Random alignment should give ARI close to 0.
        assert -0.1 <= results['ari_score'] <= 0.1
        assert results['bootstrap_std'] > 0
    
    def test_normalized_mutual_information_perfect(self, synthetic_perfect_alignment):
        """Test NMI calculation with perfect alignment."""
        _, _, _, cluster_labels = synthetic_perfect_alignment
        
        results = calculate_normalized_mutual_information(cluster_labels, cluster_labels)
        
        assert results['nmi_score'] == pytest.approx(1.0, abs=1e-10)
        assert results['p_value'] <= 0.05  # Should be highly significant.
        assert 'permutation_mean' in results
        assert 'permutation_std' in results
    
    def test_normalized_mutual_information_random(self):
        """Test NMI calculation with random alignment."""
        np.random.seed(42)
        n_points = 500
        
        clusters1 = np.random.randint(0, 4, n_points)
        clusters2 = np.random.randint(0, 4, n_points)
        
        results = calculate_normalized_mutual_information(clusters1, clusters2)
        
        # Random alignment should have low NMI and high p-value.
        assert 0 <= results['nmi_score'] <= 0.3
        assert results['p_value'] > 0.05
    
    def test_silhouette_analysis_valid_clusters(self, synthetic_perfect_alignment):
        """Test silhouette analysis with valid cluster structure."""
        _, _, coordinates, cluster_labels = synthetic_perfect_alignment
        
        # Create second clustering (slightly different).
        cluster_labels2 = cluster_labels.copy()
        # Randomly reassign 10% of points.
        reassign_mask = np.random.random(len(cluster_labels)) < 0.1
        cluster_labels2[reassign_mask] = np.random.randint(0, 5, np.sum(reassign_mask))
        
        results = calculate_silhouette_analysis(coordinates, cluster_labels, cluster_labels2)
        
        assert 'vit_silhouette' in results
        assert 'spatial_silhouette' in results
        assert 'silhouette_difference' in results
        assert 'vit_cluster_scores' in results
        assert 'spatial_cluster_scores' in results
        
        # Silhouette scores should be reasonable.
        assert -1 <= results['vit_silhouette'] <= 1
        assert -1 <= results['spatial_silhouette'] <= 1
    
    def test_confusion_matrix_analysis(self, synthetic_imperfect_alignment):
        """Test confusion matrix analysis."""
        _, _, _, vit_clusters, spatial_clusters = synthetic_imperfect_alignment
        
        results = create_confusion_matrix_analysis(vit_clusters, spatial_clusters)
        
        assert 'confusion_matrix' in results
        assert 'spatial_labels' in results
        assert 'vit_labels' in results
        assert 'overall_accuracy' in results
        assert 'cluster_statistics' in results
        
        # Check matrix dimensions.
        # sklearn's confusion_matrix uses the union of all unique labels from both
        # clusterings, so the matrix is square with size equal to the total number
        # of distinct labels across both arrays.
        cm = np.array(results['confusion_matrix'])
        all_labels = np.union1d(results['spatial_labels'], results['vit_labels'])
        assert cm.shape[0] == len(all_labels)
        assert cm.shape[1] == len(all_labels)
        
        # Accuracy should be between 0 and 1.
        assert 0 <= results['overall_accuracy'] <= 1
    
    def test_effect_sizes_calculation(self, synthetic_imperfect_alignment):
        """Test effect size calculations."""
        _, _, coordinates, vit_clusters, spatial_clusters = synthetic_imperfect_alignment
        
        results = calculate_effect_sizes(vit_clusters, spatial_clusters, coordinates)
        
        assert 'cohens_kappa' in results
        assert 'cramers_v' in results
        assert 'coherence_effect_size' in results
        assert 'chi2_statistic' in results
        
        # Effect sizes should be in reasonable ranges.
        assert -1 <= results['cohens_kappa'] <= 1
        assert 0 <= results['cramers_v'] <= 1
        assert results['chi2_statistic'] >= 0
    
    def test_edge_case_single_cluster(self):
        """Test handling of single cluster case."""
        n_points = 100
        coordinates = np.random.uniform(0, 100, (n_points, 2))
        
        # All points in same cluster.
        single_cluster = np.zeros(n_points, dtype=int)
        multi_cluster = np.random.randint(0, 3, n_points)
        
        # Should handle gracefully without errors.
        ari_result = calculate_adjusted_rand_index(single_cluster, multi_cluster)
        assert 'ari_score' in ari_result
        
        nmi_result = calculate_normalized_mutual_information(single_cluster, multi_cluster)
        assert 'nmi_score' in nmi_result
    
    def test_edge_case_empty_data(self):
        """Test handling of empty data."""
        empty_clusters = np.array([])

        # sklearn's adjusted_rand_score returns 1.0 for two empty arrays
        # (perfect agreement on the vacuous case), so the function succeeds.
        result = calculate_adjusted_rand_index(empty_clusters, empty_clusters)
        assert 'ari_score' in result
    
    def test_data_type_consistency(self, synthetic_perfect_alignment):
        """Test that all returned values have consistent data types."""
        _, _, coordinates, cluster_labels = synthetic_perfect_alignment
        
        # Test ARI.
        ari_results = calculate_adjusted_rand_index(cluster_labels, cluster_labels)
        for key, value in ari_results.items():
            assert isinstance(value, (int, float)), f"ARI {key} should be numeric"
        
        # Test NMI.
        nmi_results = calculate_normalized_mutual_information(cluster_labels, cluster_labels)
        for key, value in nmi_results.items():
            assert isinstance(value, (int, float)), f"NMI {key} should be numeric"
        
        # Test silhouette.
        sil_results = calculate_silhouette_analysis(coordinates, cluster_labels, cluster_labels)
        assert isinstance(sil_results['vit_silhouette'], (int, float))
        assert isinstance(sil_results['spatial_silhouette'], (int, float))
    
    def test_reproducibility(self, synthetic_perfect_alignment):
        """Test that results are reproducible with same input."""
        _, _, coordinates, cluster_labels = synthetic_perfect_alignment
        
        # Run analysis twice.
        results1 = calculate_adjusted_rand_index(cluster_labels, cluster_labels)
        results2 = calculate_adjusted_rand_index(cluster_labels, cluster_labels)
        
        # Core metric should be identical.
        assert results1['ari_score'] == results2['ari_score']
        
        # Bootstrap results may vary slightly, but should be close.
        assert abs(results1['bootstrap_std'] - results2['bootstrap_std']) < 0.1


def test_integration_with_file_io():
    """Test integration with file I/O operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data files.
        vit_data = pd.DataFrame({
            'spot_x': [0, 10, 20, 30],
            'spot_y': [0, 10, 20, 30],
            'sample': ['TEST'] * 4,
            'cluster': [0, 1, 0, 1]
        })
        
        spatial_data = pd.DataFrame({
            'x': [1, 11, 21, 31],
            'y': [1, 11, 21, 31],
            'sample': ['TEST'] * 4,
            'figure_idents': ['A', 'B', 'A', 'B']
        })
        
        vit_path = temp_path / 'vit_clusters.csv'
        spatial_path = temp_path / 'spatial_clusters.csv'
        
        vit_data.to_csv(vit_path, index=False)
        spatial_data.to_csv(spatial_path, index=False)
        
        # Test loading and alignment.
        aligned_vit, aligned_spatial = load_and_align_data(
            vit_path, spatial_path, ['TEST'], 'figure_idents'
        )
        
        assert len(aligned_vit) > 0
        assert len(aligned_spatial) == len(aligned_vit)
        assert 'alignment_distance' in aligned_vit.columns


if __name__ == "__main__":
    # Run tests with verbose output.
    pytest.main([__file__, "-v", "--tb=short"])
