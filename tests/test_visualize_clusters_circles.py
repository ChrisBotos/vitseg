"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_visualize_clusters_circles.py.
Description:
    Comprehensive test suite for the visualize_clusters_circles.py script.
    Tests data loading, filtering, color generation, visualization creation,
    and output file generation for tissue sample cluster visualization.

Dependencies:
    • Python >= 3.10.
    • pytest, pandas, numpy, matplotlib.
    • pathlib, tempfile for test file management.

Usage:
    pytest tests/test_visualize_clusters_circles.py -v
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add the code directory to the path for imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from visualize_clusters_circles import (
    generate_high_contrast_colors,
    load_and_filter_data,
    create_cluster_visualization,
    create_legend,
    save_statistics
)


class TestColorGeneration:
    """Test color generation functionality."""
    
    def test_small_color_count(self):
        """Test color generation for small numbers of clusters."""
        colors = generate_high_contrast_colors(5)
        assert len(colors) == 5
        assert all(color.startswith('#') for color in colors)
        assert all(len(color) == 7 for color in colors)
    
    def test_large_color_count(self):
        """Test color generation for large numbers of clusters."""
        colors = generate_high_contrast_colors(20)
        assert len(colors) == 20
        assert all(color.startswith('#') for color in colors)
        assert len(set(colors)) == 20  # All colors should be unique.
    
    def test_edge_cases(self):
        """Test edge cases for color generation."""
        # Test single color.
        colors = generate_high_contrast_colors(1)
        assert len(colors) == 1
        
        # Test zero colors.
        colors = generate_high_contrast_colors(0)
        assert len(colors) == 0


class TestDataLoading:
    """Test data loading and filtering functionality."""
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        data = {
            'condition': ['IRI', 'IRI', 'sham', 'IRI', 'sham'],
            'sample': ['IRI1', 'IRI2', 'sham1', 'IRI3', 'sham2'],
            'x': [100, 200, 300, 400, 500],
            'y': [150, 250, 350, 450, 550],
            'figure_idents': ['PT-S1/S2', 'TAL', 'PT-S1/S2', 'CNT', 'TAL'],
            'banksy': [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        return pd.DataFrame(data)
    
    def test_load_and_filter_data(self, sample_metadata):
        """Test loading and filtering of metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_metadata.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            # Test filtering for IRI samples.
            filtered_df = load_and_filter_data(temp_path, ['IRI1', 'IRI2', 'IRI3'])
            
            assert len(filtered_df) == 3  # Should have 3 IRI samples.
            assert all(filtered_df['condition'] == 'IRI')
            assert set(filtered_df['sample']) == {'IRI1', 'IRI2', 'IRI3'}
            
        finally:
            temp_path.unlink()
    
    def test_filter_specific_samples(self, sample_metadata):
        """Test filtering for specific sample names."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_metadata.to_csv(f.name, index=False)
            temp_path = Path(f.name)

        try:
            # Test filtering for only IRI1 and IRI2.
            filtered_df = load_and_filter_data(temp_path, ['IRI1', 'IRI2'])

            assert len(filtered_df) == 2
            assert set(filtered_df['sample']) == {'IRI1', 'IRI2'}

            # Test filtering for sham samples.
            sham_df = load_and_filter_data(temp_path, ['sham1', 'sham2'])

            assert len(sham_df) == 2
            assert set(sham_df['sample']) == {'sham1', 'sham2'}
            assert all(sham_df['condition'] == 'sham')

        finally:
            temp_path.unlink()


class TestVisualizationCreation:
    """Test visualization creation functionality."""
    
    @pytest.fixture
    def sample_iri_data(self):
        """Create sample IRI data for testing."""
        np.random.seed(42)  # For reproducible tests.
        data = {
            'sample': ['IRI1'] * 100 + ['IRI2'] * 100,
            'x': np.random.randint(0, 1000, 200),
            'y': np.random.randint(0, 1000, 200),
            'figure_idents': np.random.choice(['PT-S1/S2', 'TAL', 'CNT'], 200),
            'banksy': np.random.choice([1.0, 2.0, 3.0], 200)
        }
        return pd.DataFrame(data)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_cluster_visualization(self, mock_close, mock_savefig, sample_iri_data):
        """Test cluster visualization creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_viz.png'
            
            stats, color_map = create_cluster_visualization(
                sample_iri_data, 'figure_idents', 5.0, (12, 8), 0.7, output_path, False
            )
            
            # Check statistics structure.
            assert 'samples' in stats
            assert 'total_points' in stats
            assert 'color_counts' in stats
            assert stats['total_points'] == 200
            assert len(stats['samples']) == 2  # IRI1 and IRI2.
            
            # Check color mapping.
            assert len(color_map) == len(sample_iri_data['figure_idents'].unique())
            assert all(color.startswith('#') for color in color_map.values())
            
            # Verify matplotlib functions were called.
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_legend(self, mock_close, mock_savefig):
        """Test legend creation."""
        color_map = {'PT-S1/S2': '#FF0000', 'TAL': '#00FF00', 'CNT': '#0000FF'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_legend.png'
            
            create_legend(color_map, output_path)
            
            # Verify matplotlib functions were called.
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    def test_save_statistics(self, sample_iri_data):
        """Test statistics file creation."""
        stats = {
            'samples': {
                'IRI1': {'n_points': 100, 'x_range': (0, 1000), 'y_range': (0, 1000)},
                'IRI2': {'n_points': 100, 'x_range': (0, 1000), 'y_range': (0, 1000)}
            },
            'total_points': 200,
            'color_counts': {'PT-S1/S2': 70, 'TAL': 80, 'CNT': 50}
        }
        color_map = {'PT-S1/S2': '#FF0000', 'TAL': '#00FF00', 'CNT': '#0000FF'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_stats.txt'
            
            save_statistics(stats, color_map, output_path)
            
            # Verify file was created and contains expected content.
            assert output_path.exists()
            content = output_path.read_text()
            assert 'Total data points: 200' in content
            assert 'Number of samples: 2' in content
            assert 'PT-S1/S2: 70' in content
            assert '#FF0000' in content


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_full_workflow_figure_idents(self):
        """Test complete workflow with figure_idents coloring."""
        # Create sample data.
        data = {
            'condition': ['IRI'] * 50,
            'sample': ['IRI1'] * 25 + ['IRI2'] * 25,
            'x': np.random.randint(0, 1000, 50),
            'y': np.random.randint(0, 1000, 50),
            'figure_idents': np.random.choice(['PT-S1/S2', 'TAL'], 50),
            'banksy': np.random.choice([1.0, 2.0], 50)
        }
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save test data.
            metadata_path = Path(temp_dir) / 'metadata.csv'
            df.to_csv(metadata_path, index=False)
            
            # Load and filter data.
            filtered_df = load_and_filter_data(metadata_path, ['IRI1', 'IRI2'])
            assert len(filtered_df) == 50
            
            # Generate colors.
            unique_values = sorted(filtered_df['figure_idents'].unique())
            colors = generate_high_contrast_colors(len(unique_values))
            assert len(colors) == len(unique_values)
    
    def test_full_workflow_banksy(self):
        """Test complete workflow with banksy coloring."""
        # Create sample data.
        data = {
            'condition': ['IRI'] * 30,
            'sample': ['IRI1'] * 30,
            'x': np.random.randint(0, 1000, 30),
            'y': np.random.randint(0, 1000, 30),
            'figure_idents': np.random.choice(['PT-S1/S2', 'TAL'], 30),
            'banksy': np.random.choice([1.0, 2.0, 3.0], 30)
        }
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save test data.
            metadata_path = Path(temp_dir) / 'metadata.csv'
            df.to_csv(metadata_path, index=False)
            
            # Load and filter data.
            filtered_df = load_and_filter_data(metadata_path, ['IRI1'])
            assert len(filtered_df) == 30
            
            # Check banksy values.
            unique_banksy = sorted(filtered_df['banksy'].unique())
            assert len(unique_banksy) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
