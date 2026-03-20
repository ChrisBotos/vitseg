"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: demo_enhanced_colors.py.
Description:
    Demonstration script showcasing the enhanced color generation system.
    Compares original vs enhanced color palettes and shows configuration options.

Dependencies:
    • Python >= 3.10.
    • matplotlib (for visualization).
    • generate_contrast_colors module.
    • color_config module.

Usage:
    python demo_enhanced_colors.py
"""
import traceback
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

from code.generate_contrast_colors import generate_color_palette, colors_to_hex_list
from code.color_config import ColorConfig, create_example_config


def create_color_comparison_chart():
    """
    Create visual comparison between original and enhanced color systems.
    
    Generates side-by-side comparison showing the improvement in color
    vibrancy and visual distinction achieved by the enhanced system.
    """
    print("Creating color comparison chart...")
    
    # Generate enhanced colors (new system).
    enhanced_colors = generate_color_palette(n=10, background="dark")
    enhanced_hex = colors_to_hex_list(enhanced_colors)
    
    # Simulate original weak colors for comparison.
    original_colors = {
        0: (180, 150, 150, 255),  # Weak pink
        1: (150, 180, 150, 255),  # Weak green
        2: (150, 150, 180, 255),  # Weak blue
        3: (180, 180, 150, 255),  # Weak yellow
        4: (180, 150, 180, 255),  # Weak magenta
        5: (150, 180, 180, 255),  # Weak cyan
        6: (170, 160, 150, 255),  # Weak brown
        7: (160, 170, 150, 255),  # Weak olive
        8: (150, 160, 170, 255),  # Weak gray-blue
        9: (170, 150, 160, 255),  # Weak purple
    }
    original_hex = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b, a in original_colors.values()]
    
    # Create comparison visualization.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original colors (left side).
    ax1.set_title("Original Color System\n(Weak, Low-Saturation)", fontsize=14, fontweight='bold')
    for i, (color, hex_color) in enumerate(zip(original_colors.values(), original_hex)):
        r, g, b, a = color
        rect = patches.Rectangle((0, i), 1, 0.8, 
                               facecolor=(r/255, g/255, b/255), 
                               edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(1.1, i + 0.4, f"Color {i}: {hex_color} RGB({r}, {g}, {b})", 
                va='center', fontsize=10)
    
    ax1.set_xlim(0, 4)
    ax1.set_ylim(-0.5, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Enhanced colors (right side).
    ax2.set_title("Enhanced Color System\n(Vibrant, High-Contrast)", fontsize=14, fontweight='bold')
    for i, (color, hex_color) in enumerate(zip(enhanced_colors.values(), enhanced_hex)):
        r, g, b, a = color
        rect = patches.Rectangle((0, i), 1, 0.8, 
                               facecolor=(r/255, g/255, b/255), 
                               edgecolor='black', linewidth=1)
        ax2.add_patch(rect)
        ax2.text(1.1, i + 0.4, f"Color {i}: {hex_color} RGB({r}, {g}, {b})", 
                va='center', fontsize=10)
    
    ax2.set_xlim(0, 4)
    ax2.set_ylim(-0.5, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("color_comparison_chart.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Color comparison chart saved as 'color_comparison_chart.png'")


def demonstrate_custom_palettes():
    """
    Demonstrate custom color palette functionality.
    
    Shows how users can specify their own color palettes and how the
    system handles various input formats and edge cases.
    """
    print("\nDemonstrating custom color palettes...")
    
    # Example 1: Scientific color palette.
    scientific_colors = ["#FF0000", "#00FF00", "#0080FF", "#FF00FF", "#FF8C00"]
    sci_palette = generate_color_palette(n=5, custom_colors=scientific_colors)
    
    print("Scientific Color Palette:")
    for i, (r, g, b, a) in sci_palette.items():
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        print(f"  Color {i}: {hex_color} RGB({r}, {g}, {b})")
    
    # Example 2: Mixed custom and generated.
    mixed_palette = generate_color_palette(n=8, custom_colors=["#FF0000", "#00FF00"])
    
    print("\nMixed Custom + Generated Palette:")
    for i, (r, g, b, a) in mixed_palette.items():
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        source = "Custom" if i < 2 else "Generated"
        print(f"  Color {i}: {hex_color} RGB({r}, {g}, {b}) [{source}]")
    
    # Example 3: Error handling.
    robust_palette = generate_color_palette(n=4, custom_colors=["#FF0000", "invalid", "#00FF00"])
    
    print("\nRobust Error Handling (with invalid color):")
    for i, (r, g, b, a) in robust_palette.items():
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        print(f"  Color {i}: {hex_color} RGB({r}, {g}, {b})")


def demonstrate_configuration_system():
    """
    Demonstrate the ColorConfig system and JSON configuration loading.
    
    Shows how users can create, save, and load color configurations
    for consistent color schemes across analysis sessions.
    """
    print("\nDemonstrating configuration system...")
    
    # Create example configuration.
    config = ColorConfig(
        background="dark",
        alpha=200,
        saturation=0.95,
        contrast_ratio=5.0,
        custom_colors=["#FF0000", "#00FF00", "#0080FF", "#FF00FF"]
    )
    
    # Generate colors using configuration.
    config_colors = config.generate_palette(n=6)
    
    print("Configuration-based Color Generation:")
    print(f"  Background: {config.background}")
    print(f"  Alpha: {config.alpha}")
    print(f"  Saturation: {config.saturation}")
    print(f"  Contrast Ratio: {config.contrast_ratio}")
    print(f"  Custom Colors: {len(config.custom_colors) if config.custom_colors else 0}")
    print("  Generated Colors:")
    
    for i, (r, g, b, a) in config_colors.items():
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        print(f"    Color {i}: {hex_color} RGB({r}, {g}, {b}) Alpha={a}")


def demonstrate_background_adaptation():
    """
    Demonstrate automatic background adaptation for different imaging contexts.
    
    Shows how the system adapts colors for both dark (fluorescence) and
    light (brightfield) microscopy backgrounds.
    """
    print("\nDemonstrating background adaptation...")
    
    # Dark background (fluorescence microscopy).
    dark_colors = generate_color_palette(n=5, background="dark")
    print("Dark Background (Fluorescence Microscopy):")
    for i, (r, g, b, a) in dark_colors.items():
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        print(f"  Color {i}: {hex_color} RGB({r}, {g}, {b})")
    
    # Light background (brightfield microscopy).
    light_colors = generate_color_palette(n=5, background="light")
    print("\nLight Background (Brightfield Microscopy):")
    for i, (r, g, b, a) in light_colors.items():
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        print(f"  Color {i}: {hex_color} RGB({r}, {g}, {b})")


def demonstrate_large_color_sets():
    """
    Demonstrate hybrid approach for large color sets.
    
    Shows how the system handles requests for many colors using a
    combination of predefined and algorithmically generated colors.
    """
    print("\nDemonstrating large color set generation...")
    
    # Generate large color set.
    large_palette = generate_color_palette(n=25, background="dark")
    
    print(f"Large Color Set (n=25):")
    print(f"  Total colors generated: {len(large_palette)}")
    print("  First 5 colors (predefined):")
    
    for i in range(5):
        r, g, b, a = large_palette[i]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        print(f"    Color {i}: {hex_color} RGB({r}, {g}, {b})")
    
    print("  Last 5 colors (algorithmic):")
    for i in range(20, 25):
        r, g, b, a = large_palette[i]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        print(f"    Color {i}: {hex_color} RGB({r}, {g}, {b})")


def create_scientific_visualization_demo():
    """
    Create demonstration showing colors in scientific visualization context.
    
    Simulates how the enhanced colors would appear in typical scientific
    imaging scenarios like cell segmentation overlays.
    """
    print("\nCreating scientific visualization demonstration...")
    
    # Generate colors for demonstration.
    colors = generate_color_palette(n=8, background="dark")
    
    # Create simulated segmentation overlay.
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Simulate dark microscopy background.
    ax.set_facecolor('black')
    
    # Create simulated cell regions with enhanced colors.
    np.random.seed(42)  # For reproducible demo.
    
    for i, (r, g, b, a) in colors.items():
        # Create random cell-like shapes.
        center_x = np.random.uniform(0.1, 0.9)
        center_y = np.random.uniform(0.1, 0.9)
        width = np.random.uniform(0.08, 0.15)
        height = np.random.uniform(0.08, 0.15)
        
        # Create elliptical cell region.
        ellipse = patches.Ellipse((center_x, center_y), width, height,
                                angle=np.random.uniform(0, 180),
                                facecolor=(r/255, g/255, b/255, 0.7),
                                edgecolor=(r/255, g/255, b/255, 1.0),
                                linewidth=2)
        ax.add_patch(ellipse)
        
        # Add cell label.
        ax.text(center_x, center_y, f"Cell {i+1}", 
               ha='center', va='center', fontsize=10, fontweight='bold',
               color='white', bbox=dict(boxstyle="round,pad=0.3", 
                                      facecolor='black', alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Enhanced Colors in Scientific Visualization\n(Simulated Cell Segmentation Overlay)", 
                fontsize=14, fontweight='bold', color='white')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("scientific_visualization_demo.png", dpi=300, bbox_inches='tight', 
                facecolor='black')
    plt.show()
    
    print("✓ Scientific visualization demo saved as 'scientific_visualization_demo.png'")


def main():
    """
    Run complete demonstration of enhanced color generation system.
    
    Executes all demonstration functions to showcase the improvements
    and capabilities of the enhanced color system.
    """
    print("ENHANCED COLOR GENERATION SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Visual comparison.
        create_color_comparison_chart()
        
        # Custom palette functionality.
        demonstrate_custom_palettes()
        
        # Configuration system.
        demonstrate_configuration_system()
        
        # Background adaptation.
        demonstrate_background_adaptation()
        
        # Large color sets.
        demonstrate_large_color_sets()
        
        # Scientific visualization context.
        create_scientific_visualization_demo()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Key improvements demonstrated:")
        print("✓ Vibrant, high-contrast colors vs weak original colors")
        print("✓ Custom color palette support with error handling")
        print("✓ Flexible configuration system with JSON support")
        print("✓ Automatic background adaptation for different imaging")
        print("✓ Hybrid approach for large color sets")
        print("✓ Scientific visualization context optimization")
        print("\nGenerated files:")
        print("• color_comparison_chart.png - Visual comparison")
        print("• scientific_visualization_demo.png - Usage example")
        
    except Exception as e:
        print(f"ERROR during demonstration: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
