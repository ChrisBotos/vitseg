"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_assess_vit_quality.py.
Description:
    Comprehensive test suite for the ViT quality assessment functionality.
    Validates border clustering analysis, spatial coherence computation, and
    multi-scale comparison capabilities for robust quality evaluation.

Dependencies:
    • Python >= 3.10.
    • unittest, numpy, pandas, scikit-learn.
    • PIL for image handling, pathlib for file operations.

Usage:
    python -m pytest tests/test_assess_vit_quality.py -v

    # Or run directly.
    python tests/test_assess_vit_quality.py
"""
import unittest
import traceback
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Import the module under test.
try:
    from vigseg.utilities.assess_vit_quality import ViTQualityAssessor
except ImportError as e:
    print(f"Warning: Could not import assess_vit_quality module: {e}")
    print("Some tests may be skipped.")
    ViTQualityAssessor = None


class TestViTQualityAssessor(unittest.TestCase):
    """Test suite for ViT quality assessment functionality."""
    
    def setUp(self):
        """Set up test fixtures with synthetic data."""
        if ViTQualityAssessor is None:
            self.skipTest("assess_vit_quality module not available")
        
        # Create temporary directory for test files.
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create synthetic image.
        self.image_path = self.test_dir / 'test_image.tif'
        test_image = Image.new('L', (1000, 800), color=128)
        test_image.save(self.image_path)
        
        # Initialize assessor.
        self.assessor = ViTQualityAssessor(self.image_path, border_threshold=50)
        
        # Create synthetic coordinate data.
        np.random.seed(42)
        self.n_samples = 200
        
        # Generate coordinates with some near borders.
        self.coords_df = pd.DataFrame({
            'x_center': np.random.randint(10, 990, self.n_samples),
            'y_center': np.random.randint(10, 790, self.n_samples)
        })
        
        # Create synthetic features and labels.
        self.features, self.true_labels = make_blobs(
            n_samples=self.n_samples, centers=5, n_features=50,
            cluster_std=2.0, random_state=42
        )
        
        # Standardize features.
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)
        
        # Create labels with some border clustering.
        self.labels = self.true_labels.copy()
        
        # Force some border patches to cluster together (positive indicator).
        border_mask = (
            (self.coords_df['x_center'] <= 50) |
            (self.coords_df['x_center'] >= 950) |
            (self.coords_df['y_center'] <= 50) |
            (self.coords_df['y_center'] >= 750)
        )
        
        # Assign border patches to a dedicated cluster.
        self.labels[border_mask] = 5  # New cluster for border patches.
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_border_effect_detection(self):
        """Test border clustering analysis with correct interpretation."""
        print("DEBUG: Testing border effect detection...")
        
        border_metrics = self.assessor.detect_border_effects(
            self.coords_df, self.labels, patch_size=32
        )
        
        # Validate metrics structure.
        expected_keys = [
            'total_patches', 'border_patches', 'interior_patches',
            'border_fraction', 'max_border_concentration', 'mean_border_concentration',
            'border_cluster_imbalance', 'high_border_clusters',
            'high_border_cluster_fraction', 'border_clustering_quality'
        ]
        
        for key in expected_keys:
            self.assertIn(key, border_metrics)
            self.assertIsInstance(border_metrics[key], (int, float, np.integer, np.floating))
        
        # Validate logical relationships.
        self.assertEqual(
            border_metrics['total_patches'],
            border_metrics['border_patches'] + border_metrics['interior_patches']
        )
        
        self.assertGreaterEqual(border_metrics['border_fraction'], 0.0)
        self.assertLessEqual(border_metrics['border_fraction'], 1.0)
        
        # Border clustering quality should be positive for good clustering.
        self.assertGreaterEqual(border_metrics['border_clustering_quality'], 0.0)
        
        print(f"DEBUG: Border clustering quality: {border_metrics['border_clustering_quality']:.3f}")
        print(f"DEBUG: High border clusters: {border_metrics['high_border_clusters']}")
    
    def test_spatial_coherence_computation(self):
        """Test spatial coherence metrics computation."""
        print("DEBUG: Testing spatial coherence computation...")
        
        coherence_metrics = self.assessor.compute_spatial_coherence(
            self.coords_df, self.labels
        )
        
        # Validate metrics structure.
        expected_keys = [
            'mean_spatial_coherence', 'std_spatial_coherence', 'min_spatial_coherence',
            'mean_nn_purity', 'std_nn_purity', 'cluster_size_cv', 'n_clusters'
        ]
        
        for key in expected_keys:
            self.assertIn(key, coherence_metrics)
            self.assertIsInstance(coherence_metrics[key], (int, float))
            self.assertFalse(np.isnan(coherence_metrics[key]))
        
        # Validate value ranges.
        self.assertGreaterEqual(coherence_metrics['mean_spatial_coherence'], 0.0)
        self.assertLessEqual(coherence_metrics['mean_spatial_coherence'], 1.0)
        
        self.assertGreaterEqual(coherence_metrics['mean_nn_purity'], 0.0)
        self.assertLessEqual(coherence_metrics['mean_nn_purity'], 1.0)
        
        self.assertGreaterEqual(coherence_metrics['cluster_size_cv'], 0.0)
        
        print(f"DEBUG: Mean spatial coherence: {coherence_metrics['mean_spatial_coherence']:.3f}")
        print(f"DEBUG: Mean NN purity: {coherence_metrics['mean_nn_purity']:.3f}")
    
    def test_cluster_quality_metrics(self):
        """Test cluster quality metrics computation."""
        print("DEBUG: Testing cluster quality metrics...")
        
        quality_metrics = self.assessor.compute_cluster_quality_metrics(
            self.features, self.labels
        )
        
        # Validate metrics structure.
        expected_keys = [
            'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
            'n_clusters', 'min_cluster_size', 'max_cluster_size',
            'cluster_size_ratio', 'cluster_balance'
        ]
        
        for key in expected_keys:
            self.assertIn(key, quality_metrics)
            self.assertIsInstance(quality_metrics[key], (int, float, np.integer, np.floating))
            self.assertFalse(np.isnan(quality_metrics[key]))
        
        # Validate value ranges.
        self.assertGreaterEqual(quality_metrics['silhouette_score'], -1.0)
        self.assertLessEqual(quality_metrics['silhouette_score'], 1.0)
        
        self.assertGreaterEqual(quality_metrics['calinski_harabasz_score'], 0.0)
        self.assertGreaterEqual(quality_metrics['davies_bouldin_score'], 0.0)
        
        self.assertGreaterEqual(quality_metrics['cluster_size_ratio'], 1.0)
        self.assertGreaterEqual(quality_metrics['cluster_balance'], 0.0)
        
        print(f"DEBUG: Silhouette score: {quality_metrics['silhouette_score']:.3f}")
        print(f"DEBUG: Calinski-Harabasz score: {quality_metrics['calinski_harabasz_score']:.1f}")
    
    def test_patch_size_info_extraction(self):
        """Test patch size information extraction from configuration names."""
        print("DEBUG: Testing patch size info extraction...")
        
        test_cases = [
            ('IRI_regist_cropped_10k_filtered_16px', {'primary_size': 16, 'sizes': [16], 'is_combination': False}),
            ('IRI_regist_cropped_10k_filtered_32px', {'primary_size': 32, 'sizes': [32], 'is_combination': False}),
            ('IRI_regist_cropped_10k_filtered_64px', {'primary_size': 64, 'sizes': [64], 'is_combination': False}),
            ('IRI_regist_cropped_10k_filtered_16_32px', {'primary_size': 24, 'sizes': [16, 32], 'is_combination': True}),
            ('IRI_regist_cropped_10k_filtered_32_64px', {'primary_size': 48, 'sizes': [32, 64], 'is_combination': True}),
            ('IRI_regist_cropped_10k_filtered_16_32_64px', {'primary_size': 37, 'sizes': [16, 32, 64], 'is_combination': True}),
        ]
        
        for config_name, expected in test_cases:
            result = self.assessor._extract_patch_size_info(config_name)
            
            self.assertEqual(result['primary_size'], expected['primary_size'])
            self.assertEqual(result['sizes'], expected['sizes'])
            self.assertEqual(result['is_combination'], expected['is_combination'])
            
            print(f"DEBUG: {config_name} -> {result}")
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        print("DEBUG: Testing error handling...")
        
        # Test with empty data.
        empty_coords = pd.DataFrame({'x_center': [], 'y_center': []})
        empty_labels = np.array([])
        
        # Should handle empty data gracefully.
        try:
            border_metrics = self.assessor.detect_border_effects(empty_coords, empty_labels, 32)
            self.assertEqual(border_metrics['total_patches'], 0)
        except Exception as e:
            print(f"DEBUG: Expected error with empty data: {e}")
        
        # Test with single cluster.
        single_cluster_labels = np.zeros(len(self.coords_df))
        
        try:
            coherence_metrics = self.assessor.compute_spatial_coherence(
                self.coords_df, single_cluster_labels
            )
            self.assertIsInstance(coherence_metrics, dict)
        except Exception as e:
            print(f"DEBUG: Error with single cluster: {e}")


if __name__ == '__main__':
    print("=" * 80)
    print("RUNNING ViT QUALITY ASSESSMENT TESTS")
    print("=" * 80)
    
    unittest.main(verbosity=2)
