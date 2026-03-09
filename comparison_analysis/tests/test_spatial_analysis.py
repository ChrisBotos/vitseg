"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_spatial_analysis.py.
Description:
    Comprehensive test suite for spatial analysis module.
    Tests spatial autocorrelation calculations, LISA analysis, and
    Getis-Ord G statistics with synthetic spatial data scenarios.
    
    This test suite validates the accuracy of spatial statistics
    calculations and ensures proper handling of various spatial
    patterns for bioinformaticians analyzing tissue organization.

Dependencies:
    • Python >= 3.10.
    • pytest, numpy, pandas, scipy.
    • rich (for enhanced console output).

Usage:
    pytest comparison_analysis/tests/test_spatial_analysis.py -v

Key Features:
    • Synthetic spatial pattern generation.
    • Spatial autocorrelation validation.
    • Edge case handling for sparse data.
    • Performance testing with large datasets.
    • Statistical significance validation.

Notes:
    • Tests cover clustered, dispersed, and random spatial patterns.
    • Validates proper handling of coordinate systems.
    • Ensures statistical significance calculations are accurate.
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

from spatial_analysis import (
    create_spatial_weights, calculate_morans_i, calculate_local_morans_i,
    calculate_getis_ord_g, analyze_spatial_clustering_by_sample
)

from rich.console import Console
console = Console()


class TestSpatialAnalysis:
    """Test suite for spatial analysis functions."""
    
    @pytest.fixture
    def clustered_spatial_data(self):
        """Create synthetic data with strong spatial clustering."""
        np.random.seed(42)
        
        # Create 3 distinct clusters.
        cluster_centers = [(100, 100), (300, 300), (500, 100)]
        cluster_labels = []
        coordinates = []
        
        for i, (cx, cy) in enumerate(cluster_centers):
            n_points = 50
            # Generate points around cluster center.
            x_coords = np.random.normal(cx, 20, n_points)
            y_coords = np.random.normal(cy, 20, n_points)
            
            coordinates.extend(list(zip(x_coords, y_coords)))
            cluster_labels.extend([i] * n_points)
        
        coordinates = np.array(coordinates)
        cluster_labels = np.array(cluster_labels)
        
        data = pd.DataFrame({
            'spot_x': coordinates[:, 0],
            'spot_y': coordinates[:, 1],
            'sample': ['CLUSTERED'] * len(coordinates),
            'cluster': cluster_labels
        })
        
        return data, coordinates, cluster_labels
    
    @pytest.fixture
    def random_spatial_data(self):
        """Create synthetic data with random spatial distribution."""
        np.random.seed(123)
        n_points = 100
        
        # Completely random coordinates.
        coordinates = np.random.uniform(0, 600, (n_points, 2))
        cluster_labels = np.random.randint(0, 4, n_points)
        
        data = pd.DataFrame({
            'spot_x': coordinates[:, 0],
            'spot_y': coordinates[:, 1],
            'sample': ['RANDOM'] * n_points,
            'cluster': cluster_labels
        })
        
        return data, coordinates, cluster_labels
    
    @pytest.fixture
    def dispersed_spatial_data(self):
        """Create synthetic data with dispersed spatial pattern."""
        np.random.seed(456)
        
        # Create regular grid with some noise.
        grid_size = 8
        spacing = 50
        coordinates = []
        cluster_labels = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * spacing + np.random.normal(0, 5)
                y = j * spacing + np.random.normal(0, 5)
                coordinates.append((x, y))
                # Checkerboard pattern for clusters.
                cluster_labels.append((i + j) % 2)
        
        coordinates = np.array(coordinates)
        cluster_labels = np.array(cluster_labels)
        
        data = pd.DataFrame({
            'spot_x': coordinates[:, 0],
            'spot_y': coordinates[:, 1],
            'sample': ['DISPERSED'] * len(coordinates),
            'cluster': cluster_labels
        })
        
        return data, coordinates, cluster_labels
    
    def test_spatial_weights_knn(self, clustered_spatial_data):
        """Test KNN spatial weights creation."""
        data, coordinates, _ = clustered_spatial_data
        
        weights = create_spatial_weights(coordinates, method='knn', k=5)
        
        # Check weights properties.
        assert weights.shape[0] == len(coordinates)
        assert weights.shape[1] == len(coordinates)
        
        # Each point should have exactly k neighbors (or fewer if not enough points).
        for i in range(len(coordinates)):
            n_neighbors = np.sum(weights[i, :] > 0)
            assert n_neighbors <= 5
            assert n_neighbors > 0  # Should have at least one neighbor.
    
    def test_spatial_weights_distance(self, clustered_spatial_data):
        """Test distance-based spatial weights creation."""
        data, coordinates, _ = clustered_spatial_data
        
        threshold = 50.0
        weights = create_spatial_weights(coordinates, method='distance', distance_threshold=threshold)
        
        # Check that weights respect distance threshold.
        for i in range(len(coordinates)):
            for j in range(len(coordinates)):
                if i != j and weights[i, j] > 0:
                    distance = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
                    assert distance <= threshold
    
    def test_morans_i_clustered_data(self, clustered_spatial_data):
        """Test Moran's I calculation with clustered data."""
        data, coordinates, cluster_labels = clustered_spatial_data
        
        weights = create_spatial_weights(coordinates, method='knn', k=8)
        results = calculate_morans_i(cluster_labels, weights)
        
        assert 'morans_i' in results
        assert 'expected_i' in results
        assert 'z_score' in results
        assert 'p_value' in results
        
        # Clustered data should show positive spatial autocorrelation.
        assert results['morans_i'] > results['expected_i']
        assert results['z_score'] > 0
        assert results['p_value'] < 0.05  # Should be significant.
    
    def test_morans_i_random_data(self, random_spatial_data):
        """Test Moran's I calculation with random data."""
        data, coordinates, cluster_labels = random_spatial_data
        
        weights = create_spatial_weights(coordinates, method='knn', k=8)
        results = calculate_morans_i(cluster_labels, weights)
        
        # Random data should show little spatial autocorrelation.
        assert abs(results['morans_i'] - results['expected_i']) < 0.3
        assert results['p_value'] > 0.01  # Should not be highly significant.
    
    def test_morans_i_dispersed_data(self, dispersed_spatial_data):
        """Test Moran's I calculation with dispersed data."""
        data, coordinates, cluster_labels = dispersed_spatial_data
        
        weights = create_spatial_weights(coordinates, method='knn', k=4)
        results = calculate_morans_i(cluster_labels, weights)
        
        # Dispersed (checkerboard) pattern should show negative autocorrelation.
        assert results['morans_i'] < results['expected_i']
        assert results['z_score'] < 0
    
    def test_local_morans_i(self, clustered_spatial_data):
        """Test Local Moran's I (LISA) calculation."""
        data, coordinates, cluster_labels = clustered_spatial_data
        
        weights = create_spatial_weights(coordinates, method='knn', k=8)
        results = calculate_local_morans_i(cluster_labels, weights)
        
        assert 'local_i' in results
        assert 'p_values' in results
        assert 'z_scores' in results
        assert 'quadrants' in results
        assert 'significant_locations' in results
        assert 'n_significant' in results
        
        # Check array lengths.
        n_points = len(cluster_labels)
        assert len(results['local_i']) == n_points
        assert len(results['p_values']) == n_points
        assert len(results['z_scores']) == n_points
        assert len(results['quadrants']) == n_points
        assert len(results['significant_locations']) == n_points
        
        # Should find some significant local clusters.
        assert results['n_significant'] > 0
    
    def test_getis_ord_g(self, clustered_spatial_data):
        """Test Getis-Ord G statistics calculation."""
        data, coordinates, cluster_labels = clustered_spatial_data
        
        weights = create_spatial_weights(coordinates, method='knn', k=8)
        results = calculate_getis_ord_g(cluster_labels, weights)
        
        assert 'global_g' in results
        assert 'global_p_value' in results
        assert 'local_g' in results
        assert 'local_p_values' in results
        assert 'local_z_scores' in results
        assert 'hotspots' in results
        assert 'coldspots' in results
        assert 'n_hotspots' in results
        assert 'n_coldspots' in results
        
        # Check array lengths.
        n_points = len(cluster_labels)
        assert len(results['local_g']) == n_points
        assert len(results['local_p_values']) == n_points
        assert len(results['local_z_scores']) == n_points
        assert len(results['hotspots']) == n_points
        assert len(results['coldspots']) == n_points
    
    def test_analyze_spatial_clustering_by_sample(self, clustered_spatial_data, random_spatial_data):
        """Test complete spatial analysis by sample."""
        # Combine datasets.
        clustered_data, _, _ = clustered_spatial_data
        random_data, _, _ = random_spatial_data
        
        combined_data = pd.concat([clustered_data, random_data], ignore_index=True)
        samples = ['CLUSTERED', 'RANDOM']
        
        results = analyze_spatial_clustering_by_sample(combined_data, samples, distance_threshold=100.0)
        
        # Check that both samples are analyzed.
        assert 'CLUSTERED' in results
        assert 'RANDOM' in results
        
        # Check structure of results.
        for sample in samples:
            sample_results = results[sample]
            assert 'n_points' in sample_results
            assert 'n_clusters' in sample_results
            assert 'morans_i' in sample_results
            assert 'lisa' in sample_results
            assert 'getis_ord_g' in sample_results
        
        # Clustered data should show stronger spatial autocorrelation.
        clustered_morans = results['CLUSTERED']['morans_i']['morans_i']
        random_morans = results['RANDOM']['morans_i']['morans_i']
        
        assert clustered_morans > random_morans
    
    def test_edge_case_few_points(self):
        """Test handling of datasets with very few points."""
        # Create minimal dataset.
        data = pd.DataFrame({
            'spot_x': [0, 10, 20],
            'spot_y': [0, 10, 20],
            'sample': ['MINIMAL'] * 3,
            'cluster': [0, 1, 0]
        })
        
        # Should handle gracefully without errors.
        results = analyze_spatial_clustering_by_sample(data, ['MINIMAL'], distance_threshold=50.0)
        
        assert 'MINIMAL' in results
        assert results['MINIMAL']['n_points'] == 3
    
    def test_edge_case_single_cluster(self):
        """Test handling of data with only one cluster."""
        np.random.seed(42)
        coordinates = np.random.uniform(0, 100, (20, 2))
        
        data = pd.DataFrame({
            'spot_x': coordinates[:, 0],
            'spot_y': coordinates[:, 1],
            'sample': ['SINGLE'] * 20,
            'cluster': [0] * 20  # All same cluster.
        })
        
        # Should handle single cluster case.
        results = analyze_spatial_clustering_by_sample(data, ['SINGLE'], distance_threshold=50.0)
        
        assert 'SINGLE' in results
        assert results['SINGLE']['n_clusters'] == 1
    
    def test_coordinate_scaling_invariance(self, clustered_spatial_data):
        """Test that results are invariant to coordinate scaling."""
        data, coordinates, cluster_labels = clustered_spatial_data
        
        # Calculate with original coordinates.
        weights1 = create_spatial_weights(coordinates, method='knn', k=8)
        results1 = calculate_morans_i(cluster_labels, weights1)
        
        # Scale coordinates by factor of 10.
        scaled_coordinates = coordinates * 10
        weights2 = create_spatial_weights(scaled_coordinates, method='knn', k=8)
        results2 = calculate_morans_i(cluster_labels, weights2)
        
        # Moran's I should be approximately the same.
        assert abs(results1['morans_i'] - results2['morans_i']) < 0.01
    
    def test_weight_matrix_symmetry(self, clustered_spatial_data):
        """Test that spatial weight matrices are symmetric."""
        data, coordinates, _ = clustered_spatial_data
        
        # Test KNN weights (may not be symmetric).
        knn_weights = create_spatial_weights(coordinates, method='knn', k=5)
        
        # Test distance weights (should be symmetric).
        dist_weights = create_spatial_weights(coordinates, method='distance', distance_threshold=50.0)
        
        # Distance weights should be symmetric.
        assert np.allclose(dist_weights, dist_weights.T)
    
    def test_statistical_significance_consistency(self, clustered_spatial_data):
        """Test consistency of statistical significance calculations."""
        data, coordinates, cluster_labels = clustered_spatial_data
        
        weights = create_spatial_weights(coordinates, method='knn', k=8)
        
        # Run analysis multiple times.
        results1 = calculate_morans_i(cluster_labels, weights)
        results2 = calculate_morans_i(cluster_labels, weights)
        
        # Results should be identical (no randomness in calculation).
        assert results1['morans_i'] == results2['morans_i']
        assert results1['p_value'] == results2['p_value']
        assert results1['z_score'] == results2['z_score']
    
    def test_data_type_consistency(self, clustered_spatial_data):
        """Test that all returned values have consistent data types."""
        data, coordinates, cluster_labels = clustered_spatial_data
        
        weights = create_spatial_weights(coordinates, method='knn', k=8)
        
        # Test Moran's I.
        morans_results = calculate_morans_i(cluster_labels, weights)
        for key, value in morans_results.items():
            assert isinstance(value, (int, float)), f"Moran's I {key} should be numeric"
        
        # Test LISA.
        lisa_results = calculate_local_morans_i(cluster_labels, weights)
        assert isinstance(lisa_results['n_significant'], int)
        assert all(isinstance(x, (int, float)) for x in lisa_results['local_i'])
        assert all(isinstance(x, (int, float)) for x in lisa_results['p_values'])
    
    def test_performance_large_dataset(self):
        """Test performance with larger dataset."""
        np.random.seed(42)
        n_points = 1000
        
        # Generate large random dataset.
        coordinates = np.random.uniform(0, 1000, (n_points, 2))
        cluster_labels = np.random.randint(0, 10, n_points)
        
        data = pd.DataFrame({
            'spot_x': coordinates[:, 0],
            'spot_y': coordinates[:, 1],
            'sample': ['LARGE'] * n_points,
            'cluster': cluster_labels
        })
        
        # Should complete without timeout or memory issues.
        results = analyze_spatial_clustering_by_sample(data, ['LARGE'], distance_threshold=100.0)
        
        assert 'LARGE' in results
        assert results['LARGE']['n_points'] == n_points


def test_integration_with_real_data_structure():
    """Test integration with realistic data structure."""
    # Create data that mimics real spot clustering results.
    np.random.seed(42)
    n_points = 200
    
    data = pd.DataFrame({
        'spot_x': np.random.uniform(0, 500, n_points),
        'spot_y': np.random.uniform(0, 500, n_points),
        'sample': np.random.choice(['IRI1', 'IRI2'], n_points),
        'cluster': np.random.randint(0, 5, n_points),
        'spot_id': [f'spot_{i}' for i in range(n_points)]
    })
    
    samples = ['IRI1', 'IRI2']
    results = analyze_spatial_clustering_by_sample(data, samples, distance_threshold=75.0)
    
    # Should analyze both samples.
    assert len(results) >= 2
    for sample in samples:
        if sample in results:
            assert 'morans_i' in results[sample]
            assert 'lisa' in results[sample]


if __name__ == "__main__":
    # Run tests with verbose output.
    pytest.main([__file__, "-v", "--tb=short"])
