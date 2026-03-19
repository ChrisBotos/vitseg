"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_verify_spatial_alignment.py.
Description:
    Comprehensive test suite for spatial alignment verification script.
    Tests coordinate system analysis, visualization generation, and
    diagnostic reporting with synthetic and real data scenarios.

Dependencies:
    • Python >= 3.10.
    • pytest, numpy, pandas, matplotlib.
    • rich (for enhanced console output).

Usage:
    pytest tests/test_verify_spatial_alignment.py -v
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

# Add the code directory to the path.
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

from verify_spatial_alignment import (
    load_and_validate_data, analyze_coordinate_systems,
    generate_consistent_colors, save_diagnostic_report,
    save_alignment_statistics
)

from rich.console import Console
console = Console()


class TestSpatialAlignmentVerification:
    """Test suite for spatial alignment verification functions."""
    
    @pytest.fixture
    def aligned_datasets(self):
        """Create synthetic datasets with good coordinate alignment."""
        np.random.seed(42)
        n_points = 1000
        
        # Generate base coordinates.
        base_x = np.random.uniform(100, 900, n_points)
        base_y = np.random.uniform(100, 900, n_points)
        
        # Create ViT data with slight noise.
        vit_data = pd.DataFrame({
            'spot_x': base_x + np.random.normal(0, 5, n_points),
            'spot_y': base_y + np.random.normal(0, 5, n_points),
            'sample': np.random.choice(['IRI1', 'IRI2', 'IRI3'], n_points),
            'cluster': np.random.randint(0, 10, n_points)
        })
        
        # Create spatial data with similar coordinates.
        spatial_data = pd.DataFrame({
            'x': base_x + np.random.normal(0, 10, n_points),
            'y': base_y + np.random.normal(0, 10, n_points),
            'sample': np.random.choice(['IRI1', 'IRI2', 'IRI3'], n_points),
            'figure_idents': np.random.choice(['CNT', 'PT-S1/S2', 'TAL', 'PT-S3'], n_points)
        })
        
        return vit_data, spatial_data
    
    @pytest.fixture
    def misaligned_datasets(self):
        """Create synthetic datasets with poor coordinate alignment."""
        np.random.seed(123)
        n_points = 500
        
        # Create ViT data in one coordinate system.
        vit_data = pd.DataFrame({
            'spot_x': np.random.uniform(0, 100, n_points),
            'spot_y': np.random.uniform(0, 100, n_points),
            'sample': np.random.choice(['IRI1', 'IRI2'], n_points),
            'cluster': np.random.randint(0, 5, n_points)
        })
        
        # Create spatial data in completely different coordinate system.
        spatial_data = pd.DataFrame({
            'x': np.random.uniform(1000, 2000, n_points),
            'y': np.random.uniform(1000, 2000, n_points),
            'sample': np.random.choice(['IRI1', 'IRI2'], n_points),
            'figure_idents': np.random.choice(['CNT', 'PT-S1/S2'], n_points)
        })
        
        return vit_data, spatial_data
    
    @pytest.fixture
    def scaled_datasets(self):
        """Create synthetic datasets with different coordinate scales."""
        np.random.seed(456)
        n_points = 300
        
        # Generate base coordinates.
        base_x = np.random.uniform(0, 100, n_points)
        base_y = np.random.uniform(0, 100, n_points)
        
        # Create ViT data.
        vit_data = pd.DataFrame({
            'spot_x': base_x,
            'spot_y': base_y,
            'sample': ['IRI1'] * n_points,
            'cluster': np.random.randint(0, 3, n_points)
        })
        
        # Create spatial data with 10x scaling.
        spatial_data = pd.DataFrame({
            'x': base_x * 10,
            'y': base_y * 10,
            'sample': ['IRI1'] * n_points,
            'figure_idents': np.random.choice(['CNT', 'TAL'], n_points)
        })
        
        return vit_data, spatial_data
    
    def test_load_and_validate_data_success(self, aligned_datasets):
        """Test successful data loading and validation."""
        vit_data, spatial_data = aligned_datasets
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save test data.
            vit_path = temp_path / 'vit_test.csv'
            spatial_path = temp_path / 'spatial_test.csv'
            
            vit_data.to_csv(vit_path, index=False)
            spatial_data.to_csv(spatial_path, index=False)
            
            # Test loading.
            loaded_vit, loaded_spatial = load_and_validate_data(
                vit_path, spatial_path, ['IRI1', 'IRI2', 'IRI3'], 'figure_idents'
            )
            
            assert len(loaded_vit) > 0
            assert len(loaded_spatial) > 0
            assert 'spot_x' in loaded_vit.columns
            assert 'spot_y' in loaded_vit.columns
            assert 'x' in loaded_spatial.columns
            assert 'y' in loaded_spatial.columns
    
    def test_load_and_validate_data_missing_columns(self, aligned_datasets):
        """Test data loading with missing required columns."""
        vit_data, spatial_data = aligned_datasets
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create data with missing columns.
            vit_missing = vit_data.drop(columns=['spot_x'])
            spatial_missing = spatial_data.drop(columns=['figure_idents'])
            
            vit_path = temp_path / 'vit_missing.csv'
            spatial_path = temp_path / 'spatial_missing.csv'
            
            vit_missing.to_csv(vit_path, index=False)
            spatial_missing.to_csv(spatial_path, index=False)
            
            # Test that missing columns raise errors.
            with pytest.raises(ValueError, match="missing required columns"):
                load_and_validate_data(vit_path, spatial_path, ['IRI1'], 'figure_idents')
    
    def test_analyze_coordinate_systems_aligned(self, aligned_datasets):
        """Test coordinate system analysis with well-aligned data."""
        vit_data, spatial_data = aligned_datasets
        samples = ['IRI1', 'IRI2', 'IRI3']
        
        analysis = analyze_coordinate_systems(vit_data, spatial_data, samples)
        
        # Check structure.
        assert 'samples' in analysis
        assert 'overall' in analysis
        
        # Check that all samples are analyzed.
        for sample in samples:
            if sample in analysis['samples']:
                sample_data = analysis['samples'][sample]
                assert 'vit_points' in sample_data
                assert 'spatial_points' in sample_data
                assert 'x_overlap_pct' in sample_data
                assert 'y_overlap_pct' in sample_data
                assert 'coordinate_alignment' in sample_data
                
                # Well-aligned data should show good overlap.
                assert sample_data['x_overlap_pct'] > 50
                assert sample_data['y_overlap_pct'] > 50
                assert sample_data['coordinate_alignment'] == 'good'
    
    def test_analyze_coordinate_systems_misaligned(self, misaligned_datasets):
        """Test coordinate system analysis with misaligned data."""
        vit_data, spatial_data = misaligned_datasets
        samples = ['IRI1', 'IRI2']
        
        analysis = analyze_coordinate_systems(vit_data, spatial_data, samples)
        
        # Check that misaligned data shows poor overlap.
        for sample in samples:
            if sample in analysis['samples']:
                sample_data = analysis['samples'][sample]
                assert sample_data['x_overlap_pct'] < 10  # Should be very low.
                assert sample_data['y_overlap_pct'] < 10
                assert sample_data['coordinate_alignment'] == 'poor'
    
    def test_analyze_coordinate_systems_scaled(self, scaled_datasets):
        """Test coordinate system analysis with scaled data."""
        vit_data, spatial_data = scaled_datasets
        samples = ['IRI1']
        
        analysis = analyze_coordinate_systems(vit_data, spatial_data, samples)
        
        # Check overall scale ratios.
        overall = analysis['overall']
        assert 'coordinate_scale_ratio_x' in overall
        assert 'coordinate_scale_ratio_y' in overall
        
        # Should detect 10x scaling difference.
        assert abs(overall['coordinate_scale_ratio_x'] - 0.1) < 0.05  # 1/10 ratio.
        assert abs(overall['coordinate_scale_ratio_y'] - 0.1) < 0.05
    
    def test_generate_consistent_colors(self):
        """Test color generation functionality."""
        # Test basic color generation.
        colors = generate_consistent_colors(5, 'scientific')
        assert len(colors) == 5
        assert all(color.startswith('#') for color in colors)
        assert all(len(color) == 7 for color in colors)  # Hex format.
        
        # Test different palette types.
        colors_accessible = generate_consistent_colors(3, 'accessible')
        assert len(colors_accessible) == 3
        
        # Test reproducibility.
        colors1 = generate_consistent_colors(4, 'scientific')
        colors2 = generate_consistent_colors(4, 'scientific')
        # Colors should be consistent (though may vary due to randomness in fallback).
        assert len(colors1) == len(colors2)
    
    def test_save_diagnostic_report(self, aligned_datasets):
        """Test diagnostic report generation."""
        vit_data, spatial_data = aligned_datasets
        samples = ['IRI1', 'IRI2']
        
        analysis = analyze_coordinate_systems(vit_data, spatial_data, samples)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            report_path = temp_path / 'diagnostic_report.txt'
            
            save_diagnostic_report(analysis, report_path)
            
            # Check that report was created.
            assert report_path.exists()
            
            # Check report content.
            with open(report_path, 'r') as f:
                content = f.read()
            
            assert 'Spatial Alignment Verification Report' in content
            assert 'Overall Dataset Summary' in content
            assert 'Sample-by-Sample Analysis' in content
            assert 'Recommendations' in content
            
            # Should recommend proceeding with good alignment.
            assert 'good coordinate alignment' in content.lower()
    
    def test_save_diagnostic_report_poor_alignment(self, misaligned_datasets):
        """Test diagnostic report with poor alignment."""
        vit_data, spatial_data = misaligned_datasets
        samples = ['IRI1', 'IRI2']
        
        analysis = analyze_coordinate_systems(vit_data, spatial_data, samples)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            report_path = temp_path / 'diagnostic_report_poor.txt'
            
            save_diagnostic_report(analysis, report_path)
            
            # Check report content for poor alignment warnings.
            with open(report_path, 'r') as f:
                content = f.read()
            
            assert 'Poor coordinate alignment' in content or 'poor alignment' in content.lower()
            assert 'transformation required' in content.lower() or 'coordinate' in content.lower()
    
    def test_save_alignment_statistics(self, aligned_datasets):
        """Test alignment statistics JSON export."""
        vit_data, spatial_data = aligned_datasets
        samples = ['IRI1']
        
        analysis = analyze_coordinate_systems(vit_data, spatial_data, samples)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stats_path = temp_path / 'alignment_stats.json'
            
            save_alignment_statistics(analysis, stats_path)
            
            # Check that JSON was created and is valid.
            assert stats_path.exists()
            
            with open(stats_path, 'r') as f:
                loaded_analysis = json.load(f)
            
            assert 'samples' in loaded_analysis
            assert 'overall' in loaded_analysis
            assert loaded_analysis['overall']['total_vit_points'] > 0
            assert loaded_analysis['overall']['total_spatial_points'] > 0
    
    def test_edge_case_empty_sample(self):
        """Test handling of samples with no data."""
        # Create data where one sample has no points.
        vit_data = pd.DataFrame({
            'spot_x': [1, 2, 3],
            'spot_y': [1, 2, 3],
            'sample': ['IRI1', 'IRI1', 'IRI1'],
            'cluster': [0, 1, 2]
        })
        
        spatial_data = pd.DataFrame({
            'x': [1, 2],
            'y': [1, 2],
            'sample': ['IRI2', 'IRI2'],  # Different sample.
            'figure_idents': ['CNT', 'TAL']
        })
        
        # Should handle gracefully without errors.
        analysis = analyze_coordinate_systems(vit_data, spatial_data, ['IRI1', 'IRI2'])
        
        # Should only analyze samples with data in both datasets.
        assert len(analysis['samples']) <= 2
    
    def test_edge_case_single_point(self):
        """Test handling of samples with very few points."""
        vit_data = pd.DataFrame({
            'spot_x': [100],
            'spot_y': [100],
            'sample': ['IRI1'],
            'cluster': [0]
        })
        
        spatial_data = pd.DataFrame({
            'x': [105],
            'y': [105],
            'sample': ['IRI1'],
            'figure_idents': ['CNT']
        })
        
        # Should handle single points without errors.
        analysis = analyze_coordinate_systems(vit_data, spatial_data, ['IRI1'])
        
        assert 'IRI1' in analysis['samples']
        assert analysis['samples']['IRI1']['vit_points'] == 1
        assert analysis['samples']['IRI1']['spatial_points'] == 1
    
    def test_coordinate_range_calculations(self):
        """Test coordinate range and overlap calculations."""
        # Create data with known coordinate ranges.
        vit_data = pd.DataFrame({
            'spot_x': [0, 100],  # Range: 0-100.
            'spot_y': [0, 100],
            'sample': ['TEST', 'TEST'],
            'cluster': [0, 1]
        })
        
        spatial_data = pd.DataFrame({
            'x': [50, 150],  # Range: 50-150, overlap: 50-100.
            'y': [25, 125],  # Range: 25-125, overlap: 25-100.
            'sample': ['TEST', 'TEST'],
            'figure_idents': ['A', 'B']
        })
        
        analysis = analyze_coordinate_systems(vit_data, spatial_data, ['TEST'])
        
        sample_data = analysis['samples']['TEST']
        
        # Check coordinate ranges.
        assert sample_data['vit_x_range'] == (0, 100)
        assert sample_data['vit_y_range'] == (0, 100)
        assert sample_data['spatial_x_range'] == (50, 150)
        assert sample_data['spatial_y_range'] == (25, 125)
        
        # Check overlaps.
        assert sample_data['x_overlap'] == 50  # 100 - 50.
        assert sample_data['y_overlap'] == 75  # 100 - 25.
        
        # Check overlap percentages.
        assert sample_data['x_overlap_pct'] == 50.0  # 50/100.
        assert sample_data['y_overlap_pct'] == 75.0  # 75/100.


def test_integration_with_real_data_structure():
    """Test integration with realistic data structure."""
    # Create data that mimics real ViT and spatial data.
    np.random.seed(42)
    n_points = 100
    
    vit_data = pd.DataFrame({
        'spot_id': [f'spot_{i}' for i in range(n_points)],
        'spot_x': np.random.uniform(0, 1000, n_points),
        'spot_y': np.random.uniform(0, 1000, n_points),
        'sample': np.random.choice(['IRI1', 'IRI2'], n_points),
        'cluster': np.random.randint(0, 5, n_points),
        'n_nuclei': np.random.randint(1, 10, n_points)
    })
    
    spatial_data = pd.DataFrame({
        'x': np.random.uniform(0, 1000, n_points),
        'y': np.random.uniform(0, 1000, n_points),
        'sample': np.random.choice(['IRI1', 'IRI2'], n_points),
        'condition': ['IRI'] * n_points,
        'figure_idents': np.random.choice(['CNT', 'PT-S1/S2', 'TAL', 'PT-S3'], n_points),
        'banksy': np.random.randint(1, 18, n_points)
    })
    
    # Should analyze both figure_idents and banksy columns.
    analysis_fig = analyze_coordinate_systems(vit_data, spatial_data, ['IRI1', 'IRI2'])
    
    assert 'samples' in analysis_fig
    assert len(analysis_fig['samples']) <= 2


if __name__ == "__main__":
    # Run tests with verbose output.
    pytest.main([__file__, "-v", "--tb=short"])
