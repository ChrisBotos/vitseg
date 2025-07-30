# Uniform Tiling ViT Analysis

## Overview

The pipeline now supports two distinct approaches for Vision Transformer (ViT) feature extraction and clustering:

1. **Dynamic Patches** (original method): Extracts patches around individual nuclei masks
2. **Uniform Tiling** (new method): Splits the entire binary image into regular grid tiles

## Usage

### Switching Between Methods

In `pipeline.sh`, set the `USE_DYNAMIC_PATCHES` parameter:

```bash
# For dynamic patches around nuclei (original behavior)
USE_DYNAMIC_PATCHES=True

# For uniform tiling across entire image (new method)
USE_DYNAMIC_PATCHES=False
```

### Dynamic Patches Method

**When to use:**
- Individual cell morphology analysis
- Cell type classification
- Single-cell feature extraction

**Requirements:**
- Filtered segmentation masks
- Original segmentation maps
- Binary mask image

**Process:**
1. Filters segmentation masks based on morphological criteria
2. Extracts patches centered on each filtered nucleus
3. Clusters patches representing individual cells
4. Creates overlays mapping clusters back to original nuclei

### Uniform Tiling Method

**When to use:**
- Tissue architecture analysis
- Spatial pattern recognition
- Regional heterogeneity studies
- Whole-tissue organization analysis

**Requirements:**
- Binary mask image only
- No segmentation filtering needed

**Process:**
1. Divides binary image into regular grid tiles
2. Extracts ViT features from each tile
3. Clusters tiles based on tissue patterns
4. Creates overlays showing spatial cluster distribution

## Scientific Applications

### Dynamic Patches
- **Cell damage analysis**: Identify damaged vs. healthy cell morphologies
- **Cell type classification**: Distinguish different cell populations
- **Apoptosis detection**: Recognize apoptotic cell features
- **Individual cell tracking**: Follow specific cells across conditions

### Uniform Tiling
- **Tissue architecture**: Analyze overall tissue organization
- **Spatial patterns**: Identify regional tissue heterogeneity
- **Injury zones**: Map areas of tissue damage vs. recovery
- **Angiogenesis patterns**: Detect vascular organization changes
- **Inflammatory regions**: Identify areas of immune cell infiltration

## Technical Details

### File Outputs

**Dynamic Patches:**
- Uses `segmentation_mask_dynamic_patches_vit.py`
- Clustering with `cluster_vit_patches_memopt.py`
- Output: `patch_clusters.csv`

**Uniform Tiling:**
- Uses `uniform_tiling_vit.py`
- Clustering with `cluster_uniform_tiles_memopt.py`
- Output: `tile_clusters.csv`

### Parameters

**Uniform Tiling Specific:**
- `--patch_size`: Size of square tiles (default: 64 pixels)
- `--stride`: Distance between tile centers (default: 64 for non-overlapping)
- Tiles containing only background will still generate features

### Memory Considerations

**Dynamic Patches:**
- Memory usage depends on number of filtered nuclei
- Typically lower memory requirements
- Scales with cell density

**Uniform Tiling:**
- Memory usage depends on image size and tile size
- Higher memory requirements for large images
- Scales with total image area

## Configuration Examples

### For Individual Cell Analysis
```bash
USE_DYNAMIC_PATCHES=True
RUN_FILTER_MASKS=True
RUN_BINARY_CONVERSION=True
RUN_VIT_EXTRACTION=True
RUN_CLUSTERING=True
```

### For Tissue Architecture Analysis
```bash
USE_DYNAMIC_PATCHES=False
RUN_FILTER_MASKS=False      # Not needed for uniform tiling
RUN_BINARY_CONVERSION=True  # Ensure binary image exists
RUN_VIT_EXTRACTION=True
RUN_CLUSTERING=True
```

## Visualization Differences

### Dynamic Patches
- Overlays show cluster colors mapped to individual nuclei
- PCA plots represent individual cell features
- Colors correspond to cell types or morphological clusters

### Uniform Tiling
- Overlays show cluster colors mapped to spatial tiles
- PCA plots represent tissue region features
- Colors correspond to tissue architecture patterns

## Best Practices

1. **Use dynamic patches** when studying individual cell properties, morphology, or cell-type specific responses.

2. **Use uniform tiling** when studying tissue organization, spatial patterns, or regional heterogeneity.

3. **Combine both methods** for comprehensive analysis:
   - Run dynamic patches to identify cell types
   - Run uniform tiling to understand spatial organization
   - Compare results to understand cell-tissue relationships

4. **Adjust tile size** for uniform tiling based on tissue features:
   - Smaller tiles (16-32px) for fine spatial details
   - Larger tiles (64-128px) for broader tissue patterns

## Troubleshooting

### Common Issues

**Uniform Tiling:**
- Large images may require increased batch size or memory
- Very small tiles may not capture meaningful tissue features
- Background-only tiles will still be clustered (this is expected)

**Dynamic Patches:**
- Requires successful mask filtering step
- Empty filtered masks will cause clustering to fail
- Segmentation quality directly affects results

### Performance Tips

- Use GPU acceleration when available
- Adjust batch sizes based on available memory
- Consider downsampling very large images for initial analysis
- Use appropriate patch/tile sizes for your tissue type

## Integration with Existing Workflow

The new uniform tiling method integrates seamlessly with existing pipeline infrastructure:

- Same color palette system
- Same clustering algorithms
- Same output formats
- Same visualization tools
- Compatible with existing analysis scripts

This allows users to easily switch between methods or run both for comprehensive tissue analysis.
