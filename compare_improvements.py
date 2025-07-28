"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: compare_improvements.py
Description:
    Visual comparison of the color generation improvements.
    Shows before/after examples of PCA plots and overlay transparency.

Dependencies:
    • Python >= 3.10.
    • matplotlib, numpy.

Usage:
    python compare_improvements.py

Key Features:
    • Side-by-side comparison of old vs new color schemes.
    • Demonstrates improved contrast and transparency.
    • Shows publication-quality formatting improvements.

Notes:
    • Run after implementing the color improvements.
    • Generates comparison plots for documentation.
"""
import traceback
import matplotlib.pyplot as plt
import numpy as np
from generate_contrast_colors import generate_color_palette, colors_to_hex_list


def create_pca_comparison():
    """
    Create side-by-side comparison of old vs new PCA plot styling.
    Shows the improvement from dark to light background.
    """
    print("Creating PCA plot comparison...")
    
    # Generate sample PCA data.
    np.random.seed(42)
    n_points = 500
    n_clusters = 6
    
    # Create clustered data.
    cluster_centers = np.random.randn(n_clusters, 2) * 3
    points = []
    labels = []
    
    for i in range(n_clusters):
        cluster_points = np.random.randn(n_points // n_clusters, 2) * 0.8 + cluster_centers[i]
        points.append(cluster_points)
        labels.extend([i] * (n_points // n_clusters))
    
    pca_data = np.vstack(points)
    labels = np.array(labels)
    
    # Generate colors for both styles.
    dark_colors = generate_color_palette(n=n_clusters, background="dark", alpha=200)
    light_colors = generate_color_palette(n=n_clusters, background="light", alpha=89)
    
    dark_hex = colors_to_hex_list(dark_colors)
    light_hex = colors_to_hex_list(light_colors)
    
    # Create comparison plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Old style (dark background).
    ax1.set_facecolor('black')
    for i in range(n_clusters):
        mask = labels == i
        ax1.scatter(pca_data[mask, 0], pca_data[mask, 1], 
                   c=dark_hex[i], s=50, alpha=0.8, label=f'Cluster {i}')
    
    ax1.set_title('OLD: Dark Background PCA Plot', fontsize=14, color='white', pad=20)
    ax1.set_xlabel('First Principal Component', fontsize=12, color='white')
    ax1.set_ylabel('Second Principal Component', fontsize=12, color='white')
    ax1.tick_params(colors='white')
    ax1.legend(loc='upper right', facecolor='black', edgecolor='white')
    for text in ax1.legend().get_texts():
        text.set_color('white')
    
    # New style (light background).
    ax2.set_facecolor('white')
    for i in range(n_clusters):
        mask = labels == i
        ax2.scatter(pca_data[mask, 0], pca_data[mask, 1], 
                   c=light_hex[i], s=60, alpha=0.7, 
                   edgecolor='white', linewidth=0.5, label=f'Cluster {i}')
    
    ax2.set_title('NEW: Clean White Background PCA Plot', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('First Principal Component', fontsize=12)
    ax2.set_ylabel('Second Principal Component', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('pca_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ PCA comparison saved as 'pca_comparison.png'")


def create_transparency_comparison():
    """
    Create comparison showing different alpha transparency levels.
    Demonstrates the improvement from alpha=200 to alpha=89.
    """
    print("Creating transparency comparison...")
    
    # Create a sample "microscopy" background.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create mock microscopy background.
    x, y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    background = np.sin(x) * np.cos(y) + 0.5 * np.random.randn(100, 100)
    
    # Generate colors.
    colors = generate_color_palette(n=4, background="light")
    
    # Different alpha levels.
    alpha_levels = [255, 200, 89]  # Opaque, old setting, new setting
    alpha_names = ["Opaque (α=1.0)", "Old Setting (α=0.78)", "New Setting (α=0.35)"]
    
    for idx, (alpha, name) in enumerate(zip(alpha_levels, alpha_names)):
        ax = axes[idx]
        
        # Show background.
        ax.imshow(background, cmap='gray', alpha=0.8)
        
        # Add colored overlays.
        for i, (r, g, b, _) in colors.items():
            # Create circular regions.
            circle_x = 30 + (i % 2) * 40
            circle_y = 30 + (i // 2) * 40
            
            y_coords, x_coords = np.ogrid[:100, :100]
            mask = (x_coords - circle_x)**2 + (y_coords - circle_y)**2 <= 15**2
            
            # Create RGBA overlay.
            overlay = np.zeros((100, 100, 4))
            overlay[mask] = [r/255, g/255, b/255, alpha/255]
            
            ax.imshow(overlay)
        
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Overlay Transparency Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('transparency_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Transparency comparison saved as 'transparency_comparison.png'")


def create_contrast_comparison():
    """
    Create comparison showing contrast ratio improvements.
    Demonstrates better visibility with higher contrast ratios.
    """
    print("Creating contrast ratio comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Test configurations.
    configs = [
        {"background": "light", "contrast_ratio": 3.0, "title": "Old: Light Background (3.0 contrast)"},
        {"background": "light", "contrast_ratio": 4.5, "title": "New: Light Background (4.5 contrast)"},
        {"background": "dark", "contrast_ratio": 3.0, "title": "Old: Dark Background (3.0 contrast)"},
        {"background": "dark", "contrast_ratio": 4.5, "title": "New: Dark Background (4.5 contrast)"}
    ]
    
    for idx, config in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        
        # Generate colors.
        title = config.pop("title")
        colors = generate_color_palette(n=8, **config)
        
        # Set background.
        bg_color = 'black' if config["background"] == "dark" else 'white'
        ax.set_facecolor(bg_color)
        
        # Plot color swatches.
        for i, (r, g, b, a) in colors.items():
            rect = plt.Rectangle((i, 0), 0.8, 1, 
                               facecolor=(r/255, g/255, b/255),
                               edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
            
            # Add contrast ratio text.
            from generate_contrast_colors import calculate_contrast_ratio
            bg_rgb = (0, 0, 0) if config["background"] == "dark" else (255, 255, 255)
            contrast = calculate_contrast_ratio((r, g, b), bg_rgb)
            text_color = 'white' if config["background"] == "dark" else 'black'
            ax.text(i + 0.4, 0.5, f'{contrast:.1f}', 
                   ha='center', va='center', fontsize=10, 
                   color=text_color, fontweight='bold')
        
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Color Index')
        ax.set_xticks(range(8))
        ax.set_yticks([])
    
    plt.suptitle('Contrast Ratio Improvements', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('contrast_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Contrast comparison saved as 'contrast_comparison.png'")


def main():
    """
    Generate all comparison visualizations.
    """
    print("=" * 60)
    print("GENERATING COLOR IMPROVEMENT COMPARISONS")
    print("=" * 60)
    
    try:
        create_pca_comparison()
        create_transparency_comparison()
        create_contrast_comparison()
        
        print("\n" + "=" * 60)
        print("✅ ALL COMPARISONS GENERATED SUCCESSFULLY")
        print("=" * 60)
        print("\nGenerated files:")
        print("• pca_comparison.png - Before/after PCA plot styling")
        print("• transparency_comparison.png - Alpha transparency levels")
        print("• contrast_comparison.png - Contrast ratio improvements")
        print("\nKey improvements demonstrated:")
        print("• Clean white background for PCA plots (publication quality)")
        print("• Better transparency (α=0.35) for overlay visibility")
        print("• Higher contrast ratios (4.5+) for better color distinction")
        
    except Exception as e:
        print(f"\n❌ COMPARISON GENERATION FAILED: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
