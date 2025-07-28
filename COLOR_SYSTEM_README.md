# Enhanced Color Generation System

## Overview

The enhanced color generation system provides high-contrast, vibrant colors optimized for scientific visualization, particularly microscopy images and cell segmentation overlays. The system has been completely redesigned to address the limitations of the original weak, low-saturation color palette.

## Key Improvements

### 🎨 **Vibrant Predefined Colors**
- **20 carefully selected high-contrast colors** with strong saturation and brightness
- Colors like strong red (#FF0000), neon green (#00FF00), bright blue (#0080FF), magenta (#FF00FF)
- All colors tested for minimum 4.0 contrast ratio on dark backgrounds
- Optimized for rapid visual distinction in scientific imaging contexts

### ⚙️ **Custom Color Palette Support**
- **Flexible configuration system** allowing users to specify custom color palettes
- Support for hex color codes (#FF0000) and RGB values
- Automatic validation and error handling for invalid color specifications
- Graceful fallback to algorithmic generation when custom palettes are insufficient

### 🔄 **Hybrid Generation Approach**
- **Predefined colors first** (up to 20 colors) for optimal visual distinction
- **Algorithmic fallback** for large color sets (>20 colors)
- **Custom colors take precedence** when specified in configuration
- Seamless integration between different color sources

### 🌓 **Background Adaptation**
- **Automatic adjustment** for light and dark backgrounds
- Dark background: bright, vibrant colors for fluorescence microscopy
- Light background: darker, adapted colors for brightfield microscopy
- Maintains contrast ratios across different imaging contexts

## Usage Examples

### Basic Usage (Predefined Colors)
```python
from generate_contrast_colors import generate_color_palette

# Generate 8 vibrant colors for dark background
colors = generate_color_palette(n=8, background="dark")
# Returns: {0: (255, 0, 0, 255), 1: (0, 255, 0, 255), ...}
```

### Custom Color Palette
```python
# Specify custom colors
custom_colors = ["#FF0000", "#00FF00", "#0080FF", "#FF00FF"]
colors = generate_color_palette(n=6, custom_colors=custom_colors)
# First 4 colors will be custom, remaining 2 will be generated
```

### Configuration System
```python
from color_config import ColorConfig, load_color_config

# Create custom configuration
config = ColorConfig(
    background="dark",
    alpha=200,
    saturation=0.95,
    custom_colors=["#FF0000", "#00FF00", "#0080FF"]
)

# Generate colors using configuration
colors = config.generate_palette(n=8)
```

### Load from Configuration File
```python
# Load configuration from JSON file
config = load_color_config("my_color_config.json")
colors = config.generate_palette(n=10)
```

## Configuration File Format

Create a JSON configuration file to customize color generation:

```json
{
  "colors": {
    "background": "dark",
    "alpha": 200,
    "saturation": 0.95,
    "contrast_ratio": 5.0,
    "custom_colors": [
      "#FF0000",
      "#00FF00", 
      "#0080FF",
      "#FF00FF",
      "#FF8C00",
      "#00FFFF"
    ],
    "use_predefined": true,
    "validate_contrast": true,
    "fallback_to_algorithmic": true
  }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `background` | string | "dark" | Background type: "light" or "dark" |
| `alpha` | integer | 255 | Alpha transparency (0-255, 255=opaque) |
| `saturation` | float | 0.95 | Color saturation (0.0-1.0, higher=more vivid) |
| `contrast_ratio` | float | 4.5 | Minimum WCAG contrast ratio |
| `custom_colors` | array | null | List of hex color codes |
| `use_predefined` | boolean | true | Whether to use predefined vibrant colors |
| `validate_contrast` | boolean | true | Whether to validate contrast ratios |
| `fallback_to_algorithmic` | boolean | true | Whether to use algorithmic fallback |

## Integration with Existing Code

### Clustering Scripts
The enhanced system integrates seamlessly with existing clustering scripts:

```python
# In cluster_vit_patches_memopt.py
try:
    from color_config import load_color_config
    color_config = load_color_config("color_config.json")
    color_palette = color_config.generate_palette(n=k)
except ImportError:
    # Fallback to direct generation
    color_palette = generate_color_palette(n=k, background="dark")
```

### Overlay Functions
Update overlay functions to use the enhanced colors:

```python
from generate_contrast_colors import generate_color_palette

# Generate enhanced colors for overlays
colors = generate_color_palette(n=num_clusters, background="dark")
hex_colors = colors_to_hex_list(colors)
```

## Scientific Visualization Benefits

### 🔬 **Microscopy Optimization**
- **High contrast ratios** ensure visibility against dark microscopy backgrounds
- **Vibrant saturation** enables rapid visual distinction between cell types
- **Background adaptation** works with both fluorescence and brightfield imaging

### 🧬 **Cell Segmentation**
- **Maximum visual separation** between different segmented regions
- **Consistent color quality** across different numbers of clusters
- **Optimized RGB distance** between colors for clear boundaries

### 📊 **Data Analysis**
- **Publication-quality colors** suitable for scientific figures
- **Consistent color schemes** across different analysis sessions
- **Customizable palettes** for specific research requirements

## Performance Characteristics

- **Fast generation**: Predefined colors provide instant access to optimal palettes
- **Scalable**: Handles both small (2-10) and large (50+) color sets efficiently
- **Memory efficient**: Minimal overhead for color generation and storage
- **Backward compatible**: Drop-in replacement for existing color generation

## Testing and Validation

The system includes comprehensive test suites:

```bash
# Run all tests
python test_enhanced_colors.py

# Test specific functionality
python generate_contrast_colors.py
```

### Test Coverage
- ✅ Vibrant color generation validation
- ✅ Custom color palette functionality
- ✅ Configuration system testing
- ✅ Performance benchmarking
- ✅ Scientific visualization context validation
- ✅ Contrast ratio verification
- ✅ Background adaptation testing

## Migration Guide

### From Original System
1. **No code changes required** for basic usage - the enhanced system is backward compatible
2. **Optional**: Add configuration files for custom color palettes
3. **Optional**: Update clustering scripts to use ColorConfig system
4. **Recommended**: Test with your specific imaging data to verify improvements

### Configuration Migration
```python
# Old approach
colors = generate_color_palette(n=8, background="dark", saturation=0.85)

# New approach with configuration
config = ColorConfig(background="dark", saturation=0.95)
colors = config.generate_palette(n=8)
```

## Troubleshooting

### Common Issues

**Q: Colors appear too bright/dark for my images**
A: Adjust the `background` parameter ("light" vs "dark") and `contrast_ratio` in your configuration.

**Q: Custom colors not appearing**
A: Verify hex color format (#RRGGBB) and check console for validation warnings.

**Q: Need more than 20 colors**
A: The system automatically uses algorithmic generation for large color sets while maintaining quality.

**Q: Colors not distinct enough**
A: Increase `saturation` parameter or use custom colors with higher RGB distances.

## Future Enhancements

- 🎯 **Colorblind-friendly palettes** with accessibility optimization
- 🎨 **Perceptually uniform color spaces** (LAB, LUV) for better visual distinction
- 📱 **Interactive color picker** for real-time palette customization
- 🔄 **Automatic palette optimization** based on image content analysis

---

**Author**: Christos Botos  
**Contact**: botoschristos@gmail.com  
**Affiliation**: Leiden University Medical Center  
**Version**: 2.0 - Enhanced Color Generation System
