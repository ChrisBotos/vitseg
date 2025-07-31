# Learning ViT - Vision Transformer Analysis Pipeline

**Author:** Christos Botos
**Affiliation:** Leiden University Medical Center
**Contact:** botoschristos@gmail.com | [LinkedIn](https://linkedin.com/in/christos-botos-2369hcty3396) | [GitHub](https://github.com/ChrisBotos)

## Overview

This project provides a comprehensive pipeline for analyzing kidney tissue samples using Vision Transformers (ViTs) and advanced image processing techniques. The system is specifically designed for studying Ischemia/Reperfusion Kidney Injury (I/R) at different time points (10 hours, 2 days, 14 days) using multi-modal data including cell segmentation, spatial transcriptomics, spatial metabolomics, and ViT embeddings.

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
│   ├── filter_masks_memopt.py     # Memory-optimized mask filtering
│   ├── filter_features_by_box_size.py  # Multi-scale feature filtering
│   ├── overlay_masks.py           # High-quality visualization
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
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Run the complete pipeline
./pipeline.sh

# Or run individual steps:

# Step 1: Filter segmentation masks
python code/filter_masks_memopt.py \
    --input results/masks/segmentation_masks.npy \
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

### Quality Control & Visualization
- **Morphological filtering** with configurable thresholds
- **High-contrast color palettes** optimized for scientific visualization
- **Publication-quality overlays** with alpha transparency
- **Comprehensive QC metrics** and violin plots

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
- **Segmentation masks**: Integer-labeled masks in `results/masks/`
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
    mask_path="results/masks/large_masks.npy",
    output_path="results/overlay.tif",
    config=config
)
```

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