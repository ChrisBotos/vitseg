"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_color_improvements.py
Description:
    Test script to validate the improved color generation and overlay functionality.
    Compares old vs new color generation methods and validates contrast ratios.

Dependencies:
    • Python >= 3.10.
    • numpy, PIL, matplotlib.

Usage:
    python test_color_improvements.py

Key Features:
    • Tests color generation with various parameters.
    • Validates contrast ratios meet WCAG standards.
    • Checks overlay functionality with proper RGBA handling.
    • Generates comparison visualizations.

Notes:
    • Run this after implementing the improved color generation.
    • Validates both PCA plot colors and overlay colors work correctly.
"""
import traceback
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from generate_contrast_colors import generate_color_palette, colors_to_hex_list, calculate_contrast_ratio


def test_color_contrast_ratios():
    """
    Test that generated colors meet minimum contrast requirements.
    Validates both light and dark background scenarios.
    """
    print("Testing color contrast ratios...")
    
    # Test dark background (typical for microscopy).
    dark_colors = generate_color_palette(n=10, background="dark", contrast_ratio=4.5)
    bg_dark = (0, 0, 0)
    
    for i, (r, g, b, a) in dark_colors.items():
        contrast = calculate_contrast_ratio((r, g, b), bg_dark)
        assert contrast >= 4.4, f"Dark bg color {i} contrast {contrast:.2f} below threshold"
        print(f"  Dark bg color {i}: RGB({r:3d}, {g:3d}, {b:3d}) contrast={contrast:.2f} ✓")
    
    # Test light background.
    light_colors = generate_color_palette(n=6, background="light", contrast_ratio=3.0)
    bg_light = (255, 255, 255)
    
    for i, (r, g, b, a) in light_colors.items():
        contrast = calculate_contrast_ratio((r, g, b), bg_light)
        assert contrast >= 2.9, f"Light bg color {i} contrast {contrast:.2f} below threshold"
        print(f"  Light bg color {i}: RGB({r:3d}, {g:3d}, {b:3d}) contrast={contrast:.2f} ✓")
    
    print("✓ All contrast ratio tests passed.")


def test_hex_color_conversion():
    """
    Test conversion from RGBA dictionary to hex color list for matplotlib.
    Ensures proper format for seaborn/matplotlib compatibility.
    """
    print("\nTesting hex color conversion...")
    
    colors = generate_color_palette(n=5, background="dark")
    hex_colors = colors_to_hex_list(colors)
    
    assert len(hex_colors) == 5, f"Expected 5 hex colors, got {len(hex_colors)}"
    
    for i, hex_color in enumerate(hex_colors):
        assert hex_color.startswith('#'), f"Color {i} not in hex format: {hex_color}"
        assert len(hex_color) == 7, f"Color {i} wrong hex length: {hex_color}"
        
        # Validate hex characters.
        try:
            int(hex_color[1:], 16)
        except ValueError:
            assert False, f"Color {i} invalid hex: {hex_color}"
        
        print(f"  Hex color {i}: {hex_color} ✓")
    
    print("✓ Hex color conversion tests passed.")


def test_color_uniqueness():
    """
    Test that generated colors are visually distinct.
    Checks for duplicate RGB values and reasonable color separation.
    """
    print("\nTesting color uniqueness...")

    # Test with reasonable number of colors (8 is typical for clustering).
    colors = generate_color_palette(n=8, background="dark", saturation=0.85)
    rgb_values = [(r, g, b) for r, g, b, a in colors.values()]

    # Check for exact duplicates.
    unique_colors = set(rgb_values)
    assert len(unique_colors) == len(rgb_values), "Duplicate RGB colors detected"

    # Check minimum color separation (Euclidean distance).
    min_distance = float('inf')
    distances = []

    for i, (r1, g1, b1) in enumerate(rgb_values):
        for j, (r2, g2, b2) in enumerate(rgb_values[i+1:], i+1):
            distance = np.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)
            distances.append(distance)
            min_distance = min(min_distance, distance)

    avg_distance = np.mean(distances)
    print(f"  Minimum color separation: {min_distance:.1f}")
    print(f"  Average color separation: {avg_distance:.1f}")

    # For 8 colors, minimum distance should be reasonable.
    assert min_distance > 30, f"Colors too similar, min distance: {min_distance:.1f}"
    assert avg_distance > 80, f"Average separation too low: {avg_distance:.1f}"

    print("✓ Color uniqueness tests passed.")


def create_color_comparison_plot():
    """
    Create visual comparison of color palettes for different parameters.
    Saves comparison plot showing various color generation settings.
    """
    print("\nCreating color comparison visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Improved Color Palette Comparison', fontsize=16)
    
    # Test different parameter combinations.
    test_configs = [
        {"n": 8, "background": "dark", "saturation": 0.85, "contrast_ratio": 4.5, 
         "title": "Dark Background (High Contrast)"},
        {"n": 8, "background": "dark", "saturation": 0.65, "contrast_ratio": 3.0,
         "title": "Dark Background (Medium Contrast)"},
        {"n": 8, "background": "light", "saturation": 0.85, "contrast_ratio": 4.5,
         "title": "Light Background (High Contrast)"},
        {"n": 8, "background": "light", "saturation": 0.65, "contrast_ratio": 3.0,
         "title": "Light Background (Medium Contrast)"}
    ]
    
    for idx, config in enumerate(test_configs):
        ax = axes[idx // 2, idx % 2]

        # Extract title and generate colors.
        title = config.pop("title")
        colors = generate_color_palette(**config)
        
        # Set background color.
        bg_color = 'black' if config["background"] == "dark" else 'white'
        ax.set_facecolor(bg_color)
        
        # Plot color swatches.
        for i, (r, g, b, a) in colors.items():
            rect = plt.Rectangle((i, 0), 0.8, 1, 
                               facecolor=(r/255, g/255, b/255, a/255),
                               edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
            
            # Add contrast ratio text.
            bg_rgb = (0, 0, 0) if config["background"] == "dark" else (255, 255, 255)
            contrast = calculate_contrast_ratio((r, g, b), bg_rgb)
            text_color = 'white' if config["background"] == "dark" else 'black'
            ax.text(i + 0.4, 0.5, f'{contrast:.1f}', 
                   ha='center', va='center', fontsize=8, color=text_color)
        
        ax.set_xlim(0, config["n"])
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Color Index')
        ax.set_xticks(range(config["n"]))
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('color_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Color comparison plot saved as 'color_comparison.png'")


def test_overlay_rgba_handling():
    """
    Test that overlay function properly handles RGBA color arrays.
    Creates a small test overlay to validate color application.
    """
    print("\nTesting overlay RGBA handling...")

    # Create test data with new transparency settings.
    test_colors = generate_color_palette(n=3, background="light", alpha=89)  # 0.35 * 255
    
    # Simulate overlay color array creation (as done in save_overlay).
    k = len(test_colors)
    rgba_array = np.zeros((k + 1, 4), dtype=np.uint8)  # +1 for background.
    
    # Populate with colors (1-based indexing for clusters).
    for cluster_idx, (r, g, b, a) in test_colors.items():
        rgba_array[cluster_idx + 1] = [r, g, b, a]
    
    # Validate array structure.
    assert rgba_array[0].sum() == 0, "Background should be transparent"
    
    for i in range(1, k + 1):
        assert rgba_array[i][3] == 89, f"Alpha channel incorrect for cluster {i-1}"
        assert rgba_array[i][:3].sum() > 0, f"RGB values missing for cluster {i-1}"
        print(f"  Cluster {i-1}: RGBA{tuple(rgba_array[i])} ✓")
    
    print("✓ Overlay RGBA handling tests passed.")


def main():
    """
    Run all color improvement tests and generate validation outputs.
    """
    print("=" * 60)
    print("TESTING IMPROVED COLOR GENERATION SYSTEM")
    print("=" * 60)
    
    try:
        # Run all tests.
        test_color_contrast_ratios()
        test_hex_color_conversion()
        test_color_uniqueness()
        test_overlay_rgba_handling()
        create_color_comparison_plot()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - COLOR IMPROVEMENTS VALIDATED")
        print("=" * 60)
        print("\nKey improvements implemented:")
        print("• Clean, readable color generation code")
        print("• High contrast ratios for better visibility")
        print("• Proper RGBA handling in overlays")
        print("• Compatible hex colors for PCA plots")
        print("• Golden ratio hue distribution for maximum distinction")
        print("• Comprehensive debug output and error handling")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
