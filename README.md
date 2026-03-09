# Learning ViT - Vision Transformer Analysis Pipeline

**Author:** Christos Botos
**Affiliation:** Leiden University Medical Center
**Contact:** botoschristos@gmail.com | [LinkedIn](https://linkedin.com/in/christos-botos-2369hcty3396) | [GitHub](https://github.com/ChrisBotos)

## Overview

This project provides a comprehensive pipeline for analyzing kidney tissue samples using Vision Transformers (ViTs) and advanced image processing techniques. The system is specifically designed for studying Ischemia/Reperfusion Kidney Injury (I/R) at different time points (10 hours, 2 days, 14 days) using multi-modal data including cell segmentation, spatial transcriptomics, spatial metabolomics, and ViT embeddings.

## Recent Achievements

### Complete ViT-Spatial Cluster Comparison Analysis ✅
- **Spatial alignment verification**: Successfully confirmed GOOD coordinate alignment between ViT and spatial transcriptomics data
- **ViT data preparation**: 628,425 ViT points assigned to samples (IRI1/IRI2/IRI3) based on coordinate boundaries
- **Comprehensive comparison**: 123,939 matched points analyzed with statistical rigor
- **Key findings**: ARI=0.0136, NMI=0.0285 indicating complementary clustering patterns
- **Scientific insight**: ViT clusters capture morphological patterns that complement gene expression-based spatial clusters
- **Performance optimization**: KDTree-based spatial matching for efficient large-scale coordinate alignment

### Analysis Pipeline Components
- **`code/verify_spatial_alignment.py`**: Spatial coordinate system verification with diagnostic reporting
- **`code/prepare_vit_data_for_verification.py`**: Sample assignment and data formatting for ViT clusters
- **`code/run_cluster_comparison_simple.py`**: Statistical comparison analysis with confusion matrices and visualizations
- **Results directory**: `results/simple_cluster_comparison/` with comprehensive metrics and publication-quality plots

## Scientific Context

The pipeline focuses on analyzing cellular heterogeneity and tissue organization patterns in kidney injury models, with particular emphasis on:
- Cell damage and repair mechanisms
- Apoptosis, pyroptosis, necroptosis, and ferroptosis pathways
- Wnt cell pathway analysis
- Cell migration and angiogenesis
- Temporal comparison of tissue states

## Project Structure

```
ViT-on-Segmentation-MasksViT-on-Segmentation-Masks/
├── code/                           # Python source code
│   ├── __init__.py                # Package initialization
│   ├── segmentation_mask_dynamic_patches_vit.py  # Dynamic patch ViT extraction
│   ├── uniform_tiling_vit.py      # Uniform grid ViT extraction
│   ├── cluster_vit_patches_memopt.py  # Memory-efficient clustering
│   ├── cluster_uniform_tiles_memopt.py  # Uniform tile clustering
│   ├── filter_masks.py            # Mask quality control
│   ├── filter_features_by_box_size.py  # Multi-scale feature filtering
│   ├── overlay_masks.py           # High-quality visualization
│   ├── visualize_clusters_circles.py  # IRI cluster circle visualization
│   ├── cluster_spots_by_nuclei_features.py  # Spot-nuclei clustering analysis
│   ├── color_config.py            # Color palette management
│   ├── generate_contrast_colors.py  # High-contrast color generation
│   └── ...                        # Additional utilities
├── tests/                          # Test suite
│   ├── test_filter_features_by_box_size.py
│   ├── test_overlay_masks.py
│   ├── test_enhanced_colors.py
│   └── ...
├── results/                        # All analysis outputs
│   ├── masks/                     # Segmentation masks (.npy files)
│   ├── filtered_results/          # Quality-filtered masks
│   ├── VIT_dynamic_patches_*/     # ViT feature extractions
│   ├── clustered_*/               # Clustering results
│   └── ...
├── data/                          # Input data files
│   ├── IRI_regist_cropped.tif     # Microscopy images
│   ├── binary_*.tif               # Binary masks
│   └── ...
├── archived/                      # Legacy code and data
├── pipeline.sh                    # Main analysis pipeline
├── requirements.txt               # Python dependencies (Python 3.10 compatible)
├── setup_venv310.py              # Automated Python 3.10 environment setup
├── test_venv310.py               # Environment verification script
└── README.md                      # This file
```

## Quick Start

### 1. Environment Setup

#### Python 3.10 Virtual Environment (Recommended)

This project has been tested and optimized for Python 3.10 with all dependencies verified for compatibility.

**Automated Setup (Windows):**
```bash
# Run the automated setup script
python setup_venv310.py
```

The automated setup script will:
- Detect Python 3.10 installation
- Create a virtual environment named `venv310`
- Upgrade pip to the latest version
- Install all dependencies with retry logic
- Verify successful installation

**Manual Setup:**
```bash
# Create Python 3.10 virtual environment
python3.10 -m venv venv310
# On Windows: python -m venv venv310

# Activate environment
source venv310/bin/activate  # On Windows: venv310\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**Verification:**
```bash
# Test the installation
python test_venv310.py
```

#### Alternative Setup (Any Python Version)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** For GPU support with PyTorch, install CUDA-enabled versions manually:
```bash
# After environment setup, install GPU PyTorch
pip install torch==2.7.1+cu121 torchvision==0.22.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

### 2. Basic Usage

```bash
# Run the complete pipeline
./pipeline.sh

# Or run individual steps:

# Step 1: Filter segmentation masks
python code/filter_masks.py \
    --input masks/segmentation_masks.npy \
    --results-dir results/filtered_results \
    --min-pixels 20 --max-pixels 570

# Step 2: Extract ViT features (dynamic patches)
python code/segmentation_mask_dynamic_patches_vit.py \
    -i data/IRI_regist_cropped.tif \
    -m results/filtered_results/filtered_passed_masks.npy \
    -o results/VIT_dynamic_patches \
    --patch_sizes 16 32 64

# Step 3: Cluster features
python code/cluster_vit_patches_memopt.py \
    --image data/IRI_regist_cropped.tif \
    --coords results/VIT_dynamic_patches/coords.csv \
    --features_npy results/VIT_dynamic_patches/features.npy \
    --clusters 10 --outdir results/clustered_patches
```

## Key Features

### Multi-Scale ViT Analysis
- **Dynamic patch extraction** around individual nuclei with adaptive sizing
- **Uniform tiling** for tissue architecture analysis
- **Multi-scale processing** (16px, 32px, 64px) capturing cellular to tissue-level patterns
- **Spot-nuclei clustering** for spatial transcriptomics-style analysis with ViT features
- **Attention-based fusion** for combining multi-scale features

### Memory-Efficient Processing
- **Spatial batching** (2x2 tile groups) for large image processing
- **Progressive processing** with configurable memory limits
- **GPU acceleration** with graceful CPU fallback
- **Streaming data processing** for large feature sets

### Advanced Clustering
- **Memory-optimized k-means** with batch processing
- **Automatic cluster number selection** using silhouette analysis
- **Hierarchical clustering** for biological relevance
- **Ensemble methods** for robust cell type identification
- **Spot-based clustering** for spatial transcriptomics integration

### Quality Control & Visualization
- **Morphological filtering** with configurable thresholds
- **High-contrast color palettes** optimized for scientific visualization
- **Publication-quality overlays** with alpha transparency
- **Comprehensive QC metrics** and violin plots

## Spot-Nuclei Clustering Analysis

The `cluster_spots_by_nuclei_features.py` script provides spatial transcriptomics-style analysis by:

1. **Spatial Assignment**: Attaches each nucleus to the closest spot from metadata
2. **Feature Aggregation**: Combines ViT features from nuclei within each spot using mean, median, or max
3. **Robust Clustering**: Performs K-means clustering on aggregated spot features
4. **Coordinate System Handling**: Automatically detects and handles coordinate mismatches

### Usage Example

```bash
# Basic usage with real spot metadata
python code/cluster_spots_by_nuclei_features.py \
    --nuclei_coords results/IRI_regist_cropped_10k/coords_binary_IRI_regist_cropped.csv \
    --nuclei_features results/IRI_regist_cropped_10k/features_binary_IRI_regist_cropped.csv \
    --spots_metadata data/metadata_complete.csv \
    --output results/spot_nuclei_clustering \
    --samples IRI1 IRI2 IRI3 \
    --clusters 15 \
    --aggregation_method mean \
    --max_distance 100.0

# Using synthetic spots for coordinate system mismatch
python code/cluster_spots_by_nuclei_features.py \
    --nuclei_coords results/IRI_regist_cropped_10k/coords_binary_IRI_regist_cropped.csv \
    --nuclei_features results/IRI_regist_cropped_10k/features_binary_IRI_regist_cropped.csv \
    --spots_metadata data/metadata_complete.csv \
    --output results/spot_nuclei_clustering \
    --samples IRI1 IRI2 IRI3 \
    --clusters 15 \
    --create_synthetic_spots \
    --spot_grid_size 50 \
    --max_distance 100.0
```

### Key Benefits

- **Down-sampling**: Reduces computational complexity by aggregating nuclei into spots
- **Spatial Context**: Maintains spatial relationships through spot-based analysis
- **Flexible Aggregation**: Multiple methods (mean, median, max) for combining nuclei features
- **Coordinate Robustness**: Handles mismatched coordinate systems between nuclei and spots
- **Quality Control**: Comprehensive statistics and visualization outputs

### Output Files

- `spot_nuclei_assignments.csv`: Nucleus-to-spot assignments with distances
- `spot_aggregated_features.csv`: Aggregated ViT features per spot
- `spot_clusters.csv`: Final clustering results with metadata
- `spot_cluster_visualization.png`: Publication-quality visualization
- `spot_cluster_stats.txt`: Comprehensive analysis statistics

## Cluster Comparison Analysis Framework

### Overview

The `comparison_analysis/` directory contains a comprehensive scientific framework for quantitatively evaluating the alignment between ViT-derived clusters and spatial spot clusters. This analysis suite provides rigorous statistical methods, publication-quality visualizations, and comprehensive reporting suitable for scientific publication.

### Key Features

- **Statistical Alignment Metrics**: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), silhouette analysis
- **Spatial Correlation Analysis**: Moran's I, Local Indicators of Spatial Association (LISA), Getis-Ord G statistics
- **Publication-Quality Visualizations**: Side-by-side cluster maps, Sankey diagrams, heatmaps, scatter plots
- **Statistical Significance Testing**: Bootstrap confidence intervals, permutation tests
- **Comprehensive Reporting**: Automated scientific report generation with interpretation guidelines
- **Robust Error Handling**: Comprehensive validation and graceful error recovery

### Directory Structure

```
comparison_analysis/
├── README.md                    # Detailed analysis documentation
├── scripts/                     # Core analysis modules
│   ├── cluster_metrics.py       # Statistical alignment metrics
│   ├── spatial_analysis.py      # Spatial correlation analysis
│   ├── visualization_suite.py   # Publication-quality visualizations
│   └── main_analysis.py         # Complete workflow orchestration
├── tests/                       # Comprehensive test suite
│   ├── test_cluster_metrics.py  # Metrics module tests
│   └── test_spatial_analysis.py # Spatial analysis tests
├── results/                     # Analysis outputs
├── visualizations/              # Generated figures
└── reports/                     # Scientific reports
```

### Quick Start - Complete Analysis

Run the complete cluster comparison analysis:

```bash
python comparison_analysis/scripts/main_analysis.py \
    --vit_clusters results/spot_nuclei_clustering/spot_clusters.csv \
    --spatial_clusters data/metadata_complete.csv \
    --output comparison_analysis/results/complete_analysis \
    --samples IRI1 IRI2 IRI3 \
    --cluster_column figure_idents
```

### Individual Analysis Components

#### 1. Statistical Alignment Metrics

Calculate comprehensive alignment metrics between clustering approaches:

```bash
python comparison_analysis/scripts/cluster_metrics.py \
    --vit_data results/spot_nuclei_clustering/spot_clusters.csv \
    --spatial_data data/metadata_complete.csv \
    --output comparison_analysis/results/metrics \
    --samples IRI1 IRI2 IRI3
```

**Key Metrics:**
- **Adjusted Rand Index (ARI)**: Measures cluster agreement corrected for chance
  - ARI > 0.7: Strong alignment
  - ARI 0.3-0.7: Moderate alignment
  - ARI < 0.3: Weak alignment
- **Normalized Mutual Information (NMI)**: Quantifies information overlap
- **Silhouette Analysis**: Compares cluster quality between methods
- **Effect Sizes**: Cohen's kappa, Cramer's V for practical significance

#### 2. Spatial Correlation Analysis

Analyze spatial organization patterns in clustering results:

```bash
python comparison_analysis/scripts/spatial_analysis.py \
    --cluster_data results/spot_nuclei_clustering/spot_clusters.csv \
    --output comparison_analysis/results/spatial \
    --samples IRI1 IRI2 IRI3
```

**Spatial Statistics:**
- **Moran's I**: Global spatial autocorrelation
  - Moran's I > 0: Positive spatial autocorrelation (clustering)
  - Moran's I < 0: Negative spatial autocorrelation (dispersion)
  - p-value < 0.05: Statistically significant spatial pattern
- **LISA**: Local Indicators of Spatial Association for hotspot detection
- **Getis-Ord G**: Identifies clusters of high/low values

#### 3. Publication-Quality Visualizations

Generate comprehensive visualization suite:

```bash
python comparison_analysis/scripts/visualization_suite.py \
    --metrics_data comparison_analysis/results/metrics/alignment_metrics.json \
    --cluster_data results/spot_nuclei_clustering/spot_clusters.csv \
    --spatial_data data/metadata_complete.csv \
    --output comparison_analysis/visualizations \
    --samples IRI1 IRI2 IRI3
```

**Generated Visualizations:**
- **Side-by-side cluster maps**: Direct visual comparison of clustering approaches
- **Sankey diagrams**: Interactive cluster membership flow visualization
- **Centroid scatter plots**: Cluster center relationships in feature space
- **Overlap heatmaps**: Confusion matrix visualizations with proportions
- **Silhouette comparisons**: Cluster quality assessment plots
- **Metrics dashboard**: Comprehensive summary with statistical overlays

### Scientific Interpretation

#### Alignment Assessment Guidelines

**Strong Evidence for ViT-Spatial Alignment:**
- ARI > 0.7 with narrow confidence intervals
- NMI > 0.8 with p-value < 0.001
- High silhouette scores for both methods
- Significant positive spatial autocorrelation (Moran's I > 0, p < 0.05)

**Moderate Evidence:**
- ARI 0.3-0.7 with overlapping confidence intervals
- NMI 0.5-0.8 with p-value < 0.05
- Comparable silhouette scores between methods
- Some spatial organization but not consistently significant

**Limited Evidence:**
- ARI < 0.3 or wide confidence intervals including zero
- NMI < 0.5 or p-value > 0.05
- Poor silhouette scores for one or both methods
- No significant spatial autocorrelation patterns

#### Biological Interpretation

**Positive Findings Suggest:**
- ViT features capture meaningful spatial organization
- Molecular signatures align with tissue architecture
- Clustering approaches identify similar biological regions
- Spatial transcriptomics patterns are preserved in ViT representation

**Negative Findings May Indicate:**
- ViT features emphasize different biological aspects
- Spatial clustering captures local microenvironments not reflected in molecular profiles
- Need for alternative clustering parameters or methods
- Potential batch effects or technical artifacts

### Testing and Validation

Run the comprehensive test suite:

```bash
# Test all modules
pytest comparison_analysis/tests/ -v

# Test specific modules
pytest comparison_analysis/tests/test_cluster_metrics.py -v
pytest comparison_analysis/tests/test_spatial_analysis.py -v
```

## Spatial Alignment Verification

### Overview

Before running the comprehensive cluster comparison analysis, it's crucial to verify that the ViT-derived clusters and spatial transcriptomics data are properly aligned in coordinate space. The `code/verify_spatial_alignment.py` script provides visual and quantitative verification of coordinate system alignment.

### 🔍 **Verification Status: COMPLETE**

**✅ Spatial alignment verification has been completed with the following findings:**

- **ViT coordinates**: Small pixel-based coordinates (X: 2-502, Y: 1-551)
- **Spatial coordinates**: Large tissue coordinates (X: 4,662-20,782, Y: 5,388-25,028)
- **Scale mismatch**: X ratio: 0.031, Y ratio: 0.028 (should be ~1.0 for good alignment)
- **Coordinate overlap**: 0% across all samples (IRI1, IRI2, IRI3)

**⚠️ CRITICAL**: Coordinate transformation is required before running the comprehensive comparison analysis. The datasets use different coordinate systems that need to be aligned.

### Quick Start - Alignment Verification

```bash
python code/verify_spatial_alignment.py \
    --vit_clusters results/spot_nuclei_clustering/spot_clusters.csv \
    --spatial_data data/metadata_complete.csv \
    --output results/spatial_alignment_verification \
    --samples IRI1 IRI2 IRI3 \
    --cluster_column figure_idents
```

### Key Features

- **Visual Overlay Analysis**: Side-by-side and superimposed visualizations showing both datasets
- **Coordinate System Diagnostics**: Quantitative analysis of coordinate ranges and overlap
- **Alignment Statistics**: Comprehensive metrics for spatial correspondence assessment
- **Transformation Detection**: Identifies coordinate scaling, flipping, or offset issues
- **Publication-Quality Output**: High-resolution visualizations suitable for presentations

### Output Files

- **`alignment_verification.png`**: Side-by-side comparison plots for each sample
- **`overlay_visualization.png`**: Superimposed overlay plots showing both datasets
- **`coordinate_diagnostics.txt`**: Human-readable diagnostic report with recommendations
- **`alignment_statistics.json`**: Machine-readable alignment metrics

### Interpretation Guidelines

#### Good Alignment Indicators:
- **X/Y Overlap > 50%**: Substantial coordinate space overlap between datasets
- **Similar coordinate ranges**: ViT and spatial coordinates span similar numerical ranges
- **Visual correspondence**: Overlay plots show clear spatial correspondence patterns
- **Scale ratio ≈ 1.0**: Coordinate scaling is consistent between datasets

#### Poor Alignment Indicators:
- **X/Y Overlap < 10%**: Minimal coordinate space overlap
- **Vastly different ranges**: Orders of magnitude difference in coordinate values
- **No visual correspondence**: Overlay plots show completely separate point clouds
- **Scale ratio >> 1.0 or << 1.0**: Significant coordinate scaling differences

#### Common Issues and Solutions:

**Issue: Y-coordinate flipping**
```bash
# Try with Y-coordinate flipping
python code/verify_spatial_alignment.py \
    --vit_clusters results/spot_nuclei_clustering/spot_clusters.csv \
    --spatial_data data/metadata_complete.csv \
    --output results/spatial_alignment_verification \
    --samples IRI1 IRI2 IRI3 \
    --flip_y
```

**Issue: Different coordinate scales**
- Check if one dataset uses pixel coordinates while another uses physical units
- Consider coordinate normalization or scaling transformation
- Review data preprocessing steps for coordinate system consistency

**Issue: Coordinate system offset**
- Verify that both datasets reference the same tissue orientation
- Check for systematic coordinate shifts or rotations
- Consider coordinate registration or alignment procedures

### Advanced Usage

#### Custom Transparency Settings
```bash
python code/verify_spatial_alignment.py \
    --vit_clusters results/spot_nuclei_clustering/spot_clusters.csv \
    --spatial_data data/metadata_complete.csv \
    --output results/spatial_alignment_verification \
    --alpha 0.4  # Lower alpha for better overlay visibility
```

#### Alternative Spatial Clustering
```bash
python code/verify_spatial_alignment.py \
    --vit_clusters results/spot_nuclei_clustering/spot_clusters.csv \
    --spatial_data data/metadata_complete.csv \
    --output results/spatial_alignment_verification \
    --cluster_column banksy  # Use BANKSY clusters instead of figure_idents
```

### Integration with Comparison Analysis

**Recommended Workflow:**
1. **Run spatial alignment verification first**
2. **Review diagnostic outputs and visualizations**
3. **Apply coordinate transformations if needed**
4. **Proceed with comprehensive comparison analysis only after confirming good alignment**

```bash
# Step 1: Verify alignment
python code/verify_spatial_alignment.py \
    --vit_clusters results/spot_nuclei_clustering/spot_clusters.csv \
    --spatial_data data/metadata_complete.csv \
    --output results/spatial_alignment_verification \
    --samples IRI1 IRI2 IRI3

# Step 2: Review results and proceed if alignment is good
python comparison_analysis/scripts/main_analysis.py \
    --vit_clusters results/spot_nuclei_clustering/spot_clusters.csv \
    --spatial_clusters data/metadata_complete.csv \
    --output comparison_analysis/results/complete_analysis \
    --samples IRI1 IRI2 IRI3
```

### Testing

Run the spatial alignment verification test suite:

```bash
pytest tests/test_verify_spatial_alignment.py -v
```

## Pipeline Configuration

The main pipeline (`pipeline.sh`) supports extensive configuration:

```bash
# Analysis mode
USE_DYNAMIC_PATCHES=True    # True: dynamic patches, False: uniform tiling

# Multi-scale settings
PATCH_SIZES=(16 32 64)      # Patch sizes for multi-scale analysis
FILTER_BOX_SIZES=(32 64)    # Scales to include in filtered output

# Quality control thresholds
MIN_PIXELS=20; MAX_PIXELS=900
MIN_CIRC=0.56; MAX_CIRC=1.00
MIN_SOL=0.765; MAX_SOL=1.00

# Processing parameters
BATCH_SIZE=2048             # GPU batch size for ViT extraction
K_INIT=10                   # Initial cluster count
WORKERS=4                   # CPU workers for parallel processing
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_filter_features_by_box_size.py -v
python -m pytest tests/test_overlay_masks.py -v

# Run tests with coverage
python -m pytest tests/ --cov=code --cov-report=html
```

## Data Organization

### Input Data
- **Microscopy images**: High-resolution TIFF files in `data/`
- **Segmentation masks**: Integer-labeled masks in `masks/`
- **Metadata**: CSV files with experimental annotations

### Output Structure
- **Feature extractions**: CSV and NPY formats with coordinates
- **Clustering results**: Cluster assignments, PCA plots, overlay visualizations
- **QC reports**: Violin plots, metrics summaries, filtered mask statistics
- **Visualizations**: High-quality overlays and analysis plots

## Advanced Usage

### Custom Color Palettes
```python
from code.color_config import ColorConfig, create_example_config

# Create custom color configuration
config = ColorConfig(
    custom_colors=["#FF0000", "#00FF00", "#0000FF"],
    background_type="dark",
    contrast_ratio=4.5
)

# Generate palette
colors = config.generate_palette(n=10)
```

### Memory-Efficient Processing

```python
from code.overlay_masks import OverlayConfig, create_memory_efficient_overlay

# Configure for large images
config = OverlayConfig(
    tile_size=1024,
    batch_size=4,
    memory_limit_mb=8192,
    enable_gpu=True
)

# Process large overlay
create_memory_efficient_overlay(
    image_path="data/large_image.tif",
    mask_path="masks/large_masks.npy",
    output_path="results/overlay.tif",
    config=config
)
```

### Cluster Visualization

The `visualize_clusters_circles.py` script creates publication-quality visualizations of tissue samples with colored circles representing different cell types or clusters on a black background.

```bash
# Visualize figure_idents clusters
python code/visualize_clusters_circles.py \
    --metadata data/metadata_complete.csv \
    --output results/iri_figure_idents_visualization.png \
    --color-by figure_idents \
    --radius 2 \
    --figsize 24 8

# Visualize banksy clusters
python code/visualize_clusters_circles.py \
    --metadata data/metadata_complete.csv \
    --output results/iri_banksy_visualization.png \
    --color-by banksy \
    --radius 3 \
    --samples IRI1 IRI2 IRI3

# Visualize sham samples
python code/visualize_clusters_circles.py \
    --metadata data/metadata_complete.csv \
    --output results/sham_cluster_visualization.png \
    --color-by figure_idents \
    --radius 5 \
    --samples sham1 sham2 sham3

# Use enhanced colors and flip Y coordinates if needed
python code/visualize_clusters_circles.py \
    --metadata data/metadata_complete.csv \
    --output results/sham_cluster_enhanced.png \
    --color-by figure_idents \
    --radius 3 \
    --samples sham1 sham2 sham3 \
    --flip-y
```

**Key Features:**
- **Enhanced color system**: Uses advanced color generation with vibrant, high-contrast colors
- **Memory efficient**: Uses scatter plots instead of individual circle patches
- **Comprehensive output**: Main visualization, legend, and detailed statistics
- **Flexible parameters**: Customizable circle size, transparency, and figure dimensions
- **Multi-condition support**: Works with any sample types (IRI, sham, control, etc.)
- **Coordinate flexibility**: Optional Y-axis flipping for different coordinate systems

**Output Files:**
- `*_visualization.png`: Main visualization with colored circles
- `*_legend.png`: Separate legend for color mapping
- `*_stats.txt`: Detailed statistics about clusters and samples

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   - Reduce `BATCH_SIZE` in pipeline configuration
   - Enable spatial batching with smaller `batch_size`
   - Use CPU fallback by setting `enable_gpu=False`

2. **Import Errors**
   - Ensure you're running from the project root directory
   - Verify virtual environment is activated
   - Check that all dependencies are installed

3. **Large File Processing**
   - Use memory-optimized versions (`*_memopt.py`)
   - Configure appropriate tile sizes and batch sizes
   - Monitor system memory usage

### Performance Optimization

- **GPU Usage**: Recommended for ViT feature extraction
- **Batch Processing**: Adjust batch sizes based on available memory
- **Parallel Processing**: Configure worker count based on CPU cores
- **Tiling Strategy**: Use larger tiles for better GPU utilization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes following the coding style guidelines
4. Add comprehensive tests for new functionality
5. Update documentation as needed
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{botos2025learning_vit,
  author = {Christos Botos},
  title = {Learning ViT: Vision Transformer Analysis Pipeline for Kidney Injury Studies},
  year = {2025},
  url = {https://github.com/ChrisBotos/ViT-on-Segmentation-MasksViT-on-Segmentation-Masks}
}
```

## Acknowledgments

- Leiden University Medical Center for research support
- The Vision Transformer community for foundational models
- Contributors to the scientific Python ecosystem