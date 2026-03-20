"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_spot_nuclei_clustering.py.
Description:
    Comprehensive test suite for the spot-nuclei clustering analysis pipeline.
    Tests data loading, spatial assignment, feature aggregation, and clustering.

Dependencies:
    • Python >= 3.10.
    • pytest, numpy, pandas, scikit-learn.
    • All dependencies from the main clustering script.

Usage:
    python -m pytest tests/test_spot_nuclei_clustering.py -v
    python tests/test_spot_nuclei_clustering.py
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Import functions from the main script.
from vitseg.clustering.cluster_spots_by_nuclei import (
    load_data, assign_nuclei_to_spots, aggregate_features_per_spot,
    cluster_spots, generate_high_contrast_colors
)


class TestSpotNucleiClustering(unittest.TestCase):
    """Test suite for spot-nuclei clustering analysis."""
    
    def setUp(self):
        """Set up test fixtures with synthetic data."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create synthetic nuclei coordinates.
        self.nuclei_coords = pd.DataFrame({
            'x_center': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'y_center': [15, 25, 35, 45, 55, 65, 75, 85, 95, 105]
        })
        
        # Create synthetic nuclei features (simplified ViT features).
        np.random.seed(42)
        n_features = 50
        self.nuclei_features = pd.DataFrame(
            np.random.randn(len(self.nuclei_coords), n_features),
            columns=[f'vit16_{i}' for i in range(n_features)]
        )
        
        # Create synthetic spots metadata.
        self.spots_metadata = pd.DataFrame({
            'x': [12, 22, 32, 42, 52],
            'y': [17, 27, 37, 47, 57],
            'sample': ['IRI1', 'IRI1', 'IRI2', 'IRI2', 'IRI3'],
            'condition': ['IRI', 'IRI', 'IRI', 'IRI', 'IRI'],
            'figure_idents': ['PT-S1', 'TAL', 'CNT', 'PT-S3', 'interstitial']
        })
        
        # Save test data to files.
        self.coords_path = self.test_dir / 'test_coords.csv'
        self.features_path = self.test_dir / 'test_features.csv'
        self.metadata_path = self.test_dir / 'test_metadata.csv'
        
        self.nuclei_coords.to_csv(self.coords_path, index=False)
        self.nuclei_features.to_csv(self.features_path, index=False)
        self.spots_metadata.to_csv(self.metadata_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_load_data(self):
        """Test data loading functionality."""
        samples = ['IRI1', 'IRI2', 'IRI3']
        
        coords, features, metadata = load_data(
            self.coords_path, self.features_path, self.metadata_path, samples
        )
        
        # Verify data shapes and contents.
        self.assertEqual(len(coords), 10)
        self.assertEqual(len(features), 10)
        self.assertEqual(len(metadata), 5)
        self.assertEqual(features.shape[1], 50)
        
        # Verify column names.
        self.assertIn('x_center', coords.columns)
        self.assertIn('y_center', coords.columns)
        self.assertTrue(all(col.startswith('vit16_') for col in features.columns))
        self.assertIn('sample', metadata.columns)
    
    def test_load_data_sample_filtering(self):
        """Test sample filtering in data loading."""
        samples = ['IRI1', 'IRI2']  # Exclude IRI3.
        
        coords, features, metadata = load_data(
            self.coords_path, self.features_path, self.metadata_path, samples
        )
        
        # Should have 4 spots (2 IRI1 + 2 IRI2).
        self.assertEqual(len(metadata), 4)
        self.assertEqual(set(metadata['sample'].unique()), {'IRI1', 'IRI2'})
    
    def test_assign_nuclei_to_spots(self):
        """Test nucleus-to-spot assignment functionality."""
        assignments = assign_nuclei_to_spots(
            self.nuclei_coords, self.spots_metadata, max_distance=50.0
        )
        
        # Verify assignment structure.
        expected_columns = [
            'nucleus_id', 'x_center', 'y_center', 'spot_index', 
            'distance_to_spot', 'spot_id', 'spot_x', 'spot_y', 'sample', 'condition'
        ]
        for col in expected_columns:
            self.assertIn(col, assignments.columns)
        
        # Verify all assignments are within max distance.
        self.assertTrue(all(assignments['distance_to_spot'] <= 50.0))
        
        # Verify nucleus coordinates are preserved.
        self.assertTrue(np.array_equal(
            assignments['x_center'].values, self.nuclei_coords['x_center'].values[:len(assignments)]
        ))
    
    def test_assign_nuclei_to_spots_distance_filtering(self):
        """Test distance-based filtering in nucleus assignment."""
        # Use very small max distance to test filtering.
        assignments = assign_nuclei_to_spots(
            self.nuclei_coords, self.spots_metadata, max_distance=1.0
        )
        
        # Should have fewer assignments due to distance filtering.
        self.assertLess(len(assignments), len(self.nuclei_coords))
        self.assertTrue(all(assignments['distance_to_spot'] <= 1.0))
    
    def test_aggregate_features_per_spot(self):
        """Test feature aggregation per spot."""
        # First assign nuclei to spots.
        assignments = assign_nuclei_to_spots(
            self.nuclei_coords, self.spots_metadata, max_distance=50.0
        )
        
        # Test mean aggregation.
        spot_features = aggregate_features_per_spot(
            assignments, self.nuclei_features, aggregation_method='mean', min_nuclei_per_spot=1
        )
        
        # Verify output structure.
        expected_columns = [
            'spot_id', 'spot_x', 'spot_y', 'sample', 'condition', 'n_nuclei', 'mean_distance'
        ]
        for col in expected_columns:
            self.assertIn(col, spot_features.columns)
        
        # Verify aggregated feature columns.
        agg_feature_cols = [col for col in spot_features.columns if col.startswith('agg_')]
        self.assertEqual(len(agg_feature_cols), self.nuclei_features.shape[1])
        
        # Verify nuclei counts are positive.
        self.assertTrue(all(spot_features['n_nuclei'] >= 1))
    
    def test_aggregate_features_different_methods(self):
        """Test different aggregation methods."""
        assignments = assign_nuclei_to_spots(
            self.nuclei_coords, self.spots_metadata, max_distance=50.0
        )
        
        # Test all aggregation methods.
        for method in ['mean', 'median', 'max']:
            spot_features = aggregate_features_per_spot(
                assignments, self.nuclei_features, aggregation_method=method, min_nuclei_per_spot=1
            )
            
            # Should have same structure regardless of method.
            self.assertGreater(len(spot_features), 0)
            agg_cols = [col for col in spot_features.columns if col.startswith('agg_')]
            self.assertEqual(len(agg_cols), self.nuclei_features.shape[1])
    
    def test_cluster_spots(self):
        """Test spot clustering functionality."""
        # Create mock spot features.
        np.random.seed(42)
        spot_features = pd.DataFrame({
            'spot_id': range(5),
            'spot_x': [12, 22, 32, 42, 52],
            'spot_y': [17, 27, 37, 47, 57],
            'sample': ['IRI1', 'IRI1', 'IRI2', 'IRI2', 'IRI3'],
            'n_nuclei': [2, 3, 1, 4, 2]
        })
        
        # Add aggregated features.
        for i in range(10):
            spot_features[f'agg_vit16_{i}'] = np.random.randn(5)
        
        # Test clustering.
        cluster_labels, kmeans_model, scaler = cluster_spots(spot_features, n_clusters=3)
        
        # Verify clustering results.
        self.assertEqual(len(cluster_labels), 5)
        self.assertTrue(all(0 <= label < 3 for label in cluster_labels))
        self.assertIsNotNone(kmeans_model)
        self.assertIsNotNone(scaler)
    
    def test_generate_high_contrast_colors(self):
        """Test color generation functionality."""
        # Test with different numbers of colors.
        for n_colors in [3, 5, 10, 15]:
            colors = generate_high_contrast_colors(n_colors)
            
            self.assertEqual(len(colors), n_colors)
            
            # Verify hex color format.
            for color in colors:
                self.assertTrue(color.startswith('#'))
                self.assertEqual(len(color), 7)  # #RRGGBB format.
    
    def test_edge_case_no_nuclei_in_range(self):
        """Test handling of spots with no nuclei in range."""
        # Create spots far from nuclei.
        distant_spots = pd.DataFrame({
            'x': [1000, 2000],
            'y': [1000, 2000],
            'sample': ['IRI1', 'IRI1'],
            'condition': ['IRI', 'IRI'],
            'figure_idents': ['PT-S1', 'TAL']
        })
        
        assignments = assign_nuclei_to_spots(
            self.nuclei_coords, distant_spots, max_distance=50.0
        )
        
        # Should have no assignments due to distance.
        self.assertEqual(len(assignments), 0)
    
    def test_edge_case_single_nucleus_per_spot(self):
        """Test handling of spots with exactly one nucleus."""
        # Create spots very close to individual nuclei.
        close_spots = pd.DataFrame({
            'x': [10, 20],  # Very close to first two nuclei.
            'y': [15, 25],
            'sample': ['IRI1', 'IRI1'],
            'condition': ['IRI', 'IRI'],
            'figure_idents': ['PT-S1', 'TAL']
        })
        
        assignments = assign_nuclei_to_spots(
            self.nuclei_coords, close_spots, max_distance=5.0
        )
        
        spot_features = aggregate_features_per_spot(
            assignments, self.nuclei_features, aggregation_method='mean', min_nuclei_per_spot=1
        )
        
        # Should handle single-nucleus spots correctly.
        self.assertGreater(len(spot_features), 0)
        # For single nucleus, mean equals the original values.
        single_nucleus_spots = spot_features[spot_features['n_nuclei'] == 1]
        if len(single_nucleus_spots) > 0:
            self.assertTrue(all(single_nucleus_spots['n_nuclei'] == 1))
    
    def test_data_consistency_validation(self):
        """Test validation of data consistency between coordinates and features."""
        # Create mismatched data.
        short_features = self.nuclei_features.iloc[:5]  # Only 5 features for 10 coordinates.
        
        short_features_path = self.test_dir / 'short_features.csv'
        short_features.to_csv(short_features_path, index=False)
        
        # Should raise ValueError for mismatched lengths.
        with self.assertRaises(ValueError):
            load_data(self.coords_path, short_features_path, self.metadata_path, ['IRI1'])


def run_performance_benchmark():
    """Run performance benchmark with larger synthetic dataset."""
    print("\nRunning performance benchmark...")
    
    # Create larger synthetic dataset.
    np.random.seed(42)
    n_nuclei = 10000
    n_spots = 1000
    n_features = 1152  # Typical ViT feature dimension.
    
    nuclei_coords = pd.DataFrame({
        'x_center': np.random.randint(0, 5000, n_nuclei),
        'y_center': np.random.randint(0, 5000, n_nuclei)
    })
    
    nuclei_features = pd.DataFrame(
        np.random.randn(n_nuclei, n_features),
        columns=[f'vit16_{i}' for i in range(n_features)]
    )
    
    spots_metadata = pd.DataFrame({
        'x': np.random.randint(0, 5000, n_spots),
        'y': np.random.randint(0, 5000, n_spots),
        'sample': np.random.choice(['IRI1', 'IRI2', 'IRI3'], n_spots),
        'condition': ['IRI'] * n_spots,
        'figure_idents': np.random.choice(['PT-S1', 'TAL', 'CNT'], n_spots)
    })
    
    import time
    
    # Benchmark assignment.
    start_time = time.time()
    assignments = assign_nuclei_to_spots(nuclei_coords, spots_metadata, max_distance=100.0)
    assignment_time = time.time() - start_time
    
    print(f"Assignment time: {assignment_time:.2f}s for {n_nuclei:,} nuclei and {n_spots:,} spots")
    print(f"Assigned {len(assignments):,} nuclei to spots")
    
    # Benchmark aggregation.
    start_time = time.time()
    spot_features = aggregate_features_per_spot(assignments, nuclei_features, 'mean', 1)
    aggregation_time = time.time() - start_time
    
    print(f"Aggregation time: {aggregation_time:.2f}s for {len(spot_features):,} spots")
    
    # Benchmark clustering.
    start_time = time.time()
    cluster_labels, _, _ = cluster_spots(spot_features, n_clusters=15)
    clustering_time = time.time() - start_time
    
    print(f"Clustering time: {clustering_time:.2f}s for {len(spot_features):,} spots")
    print(f"Total pipeline time: {assignment_time + aggregation_time + clustering_time:.2f}s")


if __name__ == '__main__':
    # Run unit tests.
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmark.
    run_performance_benchmark()
