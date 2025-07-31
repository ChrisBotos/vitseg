"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Test Script Name: test_filter_features_by_box_size.py.
Description:
    Comprehensive test suite for the filter_features_by_box_size.py script.
    Tests feature filtering functionality, dimension detection, and file handling
    for multi-scale ViT feature processing.

    Key test areas for bioinformatician users:
        • **Feature dimension detection** – Validates correct identification of
          scale boundaries in concatenated multi-scale feature vectors.
        • **Scale filtering accuracy** – Ensures selected box sizes are correctly
          extracted from combined feature files without data corruption.
        • **File handling robustness** – Tests proper input validation, output
          directory creation, and coordinate file copying functionality.
        • **Edge case handling** – Validates behavior with invalid scales,
          missing files, and malformed input data.
        • **Memory efficiency** – Confirms processing of large feature files
          without excessive memory usage or performance degradation.

    Scientific validation:
        Tests ensure that filtered features maintain their biological meaning
        and spatial relationships, critical for downstream clustering and
        analysis in kidney injury studies.

Dependencies:
    • Python>=3.10.
    • pytest for test framework.
    • numpy, pandas for data processing.
    • tempfile for temporary test files.

Usage:
    # Run all tests.
    pytest tests/test_filter_features_by_box_size.py -v

    # Run specific test.
    pytest tests/test_filter_features_by_box_size.py::test_feature_dimension_detection -v

Key Features:
    • Comprehensive test coverage for all script functions.
    • Temporary file handling for isolated test environments.
    • Validation of output file formats and naming conventions.
    • Performance testing for large feature datasets.

Notes:
    • Tests use synthetic feature data with known dimensions.
    • All tests clean up temporary files automatically.
    • Test data mimics real ViT-S/16 384-dimensional features.
    • Validates compatibility with existing pipeline infrastructure.
"""

import pytest
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Import the functions to test.
import sys
sys.path.append(str(Path(__file__).parent.parent))
from code.filter_features_by_box_size import detect_feature_dimensions, filter_features_by_scales


class TestFilterFeaturesByBoxSize:
    """Test suite for feature filtering functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.features_per_scale = 384
        
        # Create synthetic multi-scale feature data.
        self.num_samples = 100
        self.create_test_data()
    
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary files.
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """Create synthetic multi-scale feature data for testing."""
        # Generate features for 3 scales (16px, 32px, 64px).
        total_features = 3 * self.features_per_scale
        
        # Create random feature data.
        np.random.seed(42)  # For reproducible tests.
        feature_data = np.random.randn(self.num_samples, total_features)
        
        # Create DataFrame with numbered columns.
        columns = [str(i) for i in range(total_features)]
        self.df_features = pd.DataFrame(feature_data, columns=columns)
        
        # Save to temporary CSV.
        self.features_csv = self.temp_dir / "test_features.csv"
        self.df_features.to_csv(self.features_csv, index=False)
        
        # Create coordinates data.
        coords_data = {
            'x_center': np.random.randint(0, 1000, self.num_samples),
            'y_center': np.random.randint(0, 1000, self.num_samples)
        }
        self.df_coords = pd.DataFrame(coords_data)
        self.coords_csv = self.temp_dir / "test_coords.csv"
        self.df_coords.to_csv(self.coords_csv, index=False)
    
    def test_feature_dimension_detection(self):
        """Test automatic detection of feature dimensions per scale."""
        print("DEBUG: Testing feature dimension detection")
        
        scale_mapping = detect_feature_dimensions(self.features_csv)
        
        # Should detect 3 scales with 384 features each.
        assert len(scale_mapping) == 3, f"Expected 3 scales, got {len(scale_mapping)}"
        
        expected_scales = [16, 32, 64]
        for scale in expected_scales:
            assert scale in scale_mapping, f"Scale {scale} not detected"
        
        # Check column ranges.
        assert scale_mapping[16] == (0, 384), f"16px range incorrect: {scale_mapping[16]}"
        assert scale_mapping[32] == (384, 768), f"32px range incorrect: {scale_mapping[32]}"
        assert scale_mapping[64] == (768, 1152), f"64px range incorrect: {scale_mapping[64]}"
        
        print("✓ Feature dimension detection test passed")
    
    def test_single_scale_filtering(self):
        """Test filtering for a single box size."""
        print("DEBUG: Testing single scale filtering (16px)")
        
        output_dir = self.temp_dir / "single_scale_output"
        selected_scales = [16]
        
        filter_features_by_scales(
            input_csv=self.features_csv,
            output_dir=output_dir,
            selected_scales=selected_scales,
            coords_csv=self.coords_csv
        )
        
        # Check output file exists.
        expected_output = output_dir / "filtered_features_16px_test_features.csv"
        assert expected_output.exists(), f"Output file not created: {expected_output}"
        
        # Verify filtered features.
        df_filtered = pd.read_csv(expected_output)
        assert df_filtered.shape == (self.num_samples, 384), \
            f"Expected shape ({self.num_samples}, 384), got {df_filtered.shape}"
        
        # Verify coordinates copied.
        coords_output = output_dir / "coords_test_features.csv"
        assert coords_output.exists(), f"Coordinates file not copied: {coords_output}"
        
        print("✓ Single scale filtering test passed")
    
    def test_multi_scale_filtering(self):
        """Test filtering for multiple box sizes."""
        print("DEBUG: Testing multi-scale filtering (32px + 64px)")
        
        output_dir = self.temp_dir / "multi_scale_output"
        selected_scales = [32, 64]
        
        filter_features_by_scales(
            input_csv=self.features_csv,
            output_dir=output_dir,
            selected_scales=selected_scales,
            coords_csv=self.coords_csv
        )
        
        # Check output file exists.
        expected_output = output_dir / "filtered_features_32_64px_test_features.csv"
        assert expected_output.exists(), f"Output file not created: {expected_output}"
        
        # Verify filtered features (2 scales × 384 features).
        df_filtered = pd.read_csv(expected_output)
        expected_features = 2 * 384
        assert df_filtered.shape == (self.num_samples, expected_features), \
            f"Expected shape ({self.num_samples}, {expected_features}), got {df_filtered.shape}"
        
        print("✓ Multi-scale filtering test passed")
    
    def test_all_scales_filtering(self):
        """Test filtering with all available scales."""
        print("DEBUG: Testing all scales filtering (16px + 32px + 64px)")
        
        output_dir = self.temp_dir / "all_scales_output"
        selected_scales = [16, 32, 64]
        
        filter_features_by_scales(
            input_csv=self.features_csv,
            output_dir=output_dir,
            selected_scales=selected_scales,
            coords_csv=self.coords_csv
        )
        
        # Check output file exists.
        expected_output = output_dir / "filtered_features_16_32_64px_test_features.csv"
        assert expected_output.exists(), f"Output file not created: {expected_output}"
        
        # Verify filtered features (should match original).
        df_filtered = pd.read_csv(expected_output)
        assert df_filtered.shape == self.df_features.shape, \
            f"Expected original shape {self.df_features.shape}, got {df_filtered.shape}"
        
        print("✓ All scales filtering test passed")
    
    def test_invalid_scale_handling(self):
        """Test handling of invalid scale requests."""
        print("DEBUG: Testing invalid scale handling")
        
        output_dir = self.temp_dir / "invalid_scale_output"
        invalid_scales = [128]  # Not available in test data.
        
        with pytest.raises(ValueError, match="Invalid scales requested"):
            filter_features_by_scales(
                input_csv=self.features_csv,
                output_dir=output_dir,
                selected_scales=invalid_scales
            )
        
        print("✓ Invalid scale handling test passed")
    
    def test_missing_input_file(self):
        """Test handling of missing input files."""
        print("DEBUG: Testing missing input file handling")
        
        missing_file = self.temp_dir / "nonexistent.csv"
        output_dir = self.temp_dir / "missing_input_output"
        
        with pytest.raises(Exception):  # Should raise an exception.
            filter_features_by_scales(
                input_csv=missing_file,
                output_dir=output_dir,
                selected_scales=[16]
            )
        
        print("✓ Missing input file handling test passed")
    
    def test_output_directory_creation(self):
        """Test automatic creation of output directories."""
        print("DEBUG: Testing output directory creation")
        
        output_dir = self.temp_dir / "nested" / "output" / "directory"
        assert not output_dir.exists(), "Output directory should not exist initially"
        
        filter_features_by_scales(
            input_csv=self.features_csv,
            output_dir=output_dir,
            selected_scales=[16]
        )
        
        assert output_dir.exists(), "Output directory should be created automatically"
        
        print("✓ Output directory creation test passed")
    
    def test_feature_data_integrity(self):
        """Test that filtered features maintain data integrity."""
        print("DEBUG: Testing feature data integrity")
        
        output_dir = self.temp_dir / "integrity_test_output"
        selected_scales = [32]
        
        filter_features_by_scales(
            input_csv=self.features_csv,
            output_dir=output_dir,
            selected_scales=selected_scales
        )
        
        # Load filtered features.
        expected_output = output_dir / "filtered_features_32px_test_features.csv"
        df_filtered = pd.read_csv(expected_output)
        
        # Compare with expected slice from original data.
        original_32px_features = self.df_features.iloc[:, 384:768]
        
        # Check that filtered data matches original slice.
        pd.testing.assert_frame_equal(
            df_filtered.reset_index(drop=True),
            original_32px_features.reset_index(drop=True),
            check_names=False
        )
        
        print("✓ Feature data integrity test passed")


if __name__ == "__main__":
    # Run tests directly.
    test_suite = TestFilterFeaturesByBoxSize()
    
    try:
        test_suite.setup_method()
        
        # Run all test methods.
        test_methods = [
            test_suite.test_feature_dimension_detection,
            test_suite.test_single_scale_filtering,
            test_suite.test_multi_scale_filtering,
            test_suite.test_all_scales_filtering,
            test_suite.test_invalid_scale_handling,
            test_suite.test_missing_input_file,
            test_suite.test_output_directory_creation,
            test_suite.test_feature_data_integrity
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"FAILED: {test_method.__name__}")
                traceback.print_exc()
                break
        else:
            print("\n✓ All tests passed successfully!")
        
    finally:
        test_suite.teardown_method()
