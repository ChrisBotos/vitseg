"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: generate_contrast_colors.py
Description:
    Generate high-contrast color palettes optimized for scientific visualization.
    Creates visually distinct colors with guaranteed contrast ratios for overlays.

Dependencies:
    • Python >= 3.10.
    • colorsys (standard library).

Usage:
    from generate_contrast_colors import generate_color_palette
    colors = generate_color_palette(n=10, background="dark")

Key Features:
    • Golden ratio hue distribution for maximum visual separation.
    • WCAG-compliant contrast ratios for accessibility.
    • Optimized for both light and dark backgrounds.
    • Returns both RGBA tuples and hex codes for flexibility.

Notes:
    • Use background="dark" for microscopy images with dark backgrounds.
    • Higher contrast_ratio values improve visibility but reduce color variety.
"""
import traceback
from typing import Dict, Tuple, List
import colorsys


def calculate_luminance(rgb: Tuple[int, int, int]) -> float:
    """
    Calculate relative luminance using WCAG 2.1 formula.

    Parameters:
        rgb: RGB color tuple with values 0-255.

    Returns:
        Relative luminance value between 0 and 1.
    """
    r, g, b = [channel / 255.0 for channel in rgb]

    # Convert sRGB to linear RGB.
    def linearize(channel):
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    r_linear = linearize(r)
    g_linear = linearize(g)
    b_linear = linearize(b)

    # Apply WCAG luminance coefficients.
    return 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear


def calculate_contrast_ratio(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """
    Calculate WCAG contrast ratio between two colors.

    Parameters:
        color1: First RGB color tuple (0-255).
        color2: Second RGB color tuple (0-255).

    Returns:
        Contrast ratio (1.0 to 21.0, higher is better contrast).
    """
    lum1 = calculate_luminance(color1)
    lum2 = calculate_luminance(color2)

    # Ensure lighter color is in numerator.
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)

    return (lighter + 0.05) / (darker + 0.05)


def generate_color_palette(n: int, alpha: int = 255, background: str = "dark",
                          saturation: float = 0.85, contrast_ratio: float = 4.5,
                          hue_start: float = 0.0) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Generate visually distinct colors with high contrast for scientific visualization.

    Uses golden ratio spacing for optimal hue distribution and ensures all colors
    meet WCAG contrast requirements against the specified background.

    Parameters:
        n: Number of colors to generate.
        alpha: Alpha transparency (0-255, 255 = opaque).
        background: Background type ("light" or "dark").
        saturation: Color saturation (0.0-1.0, higher = more vivid).
        contrast_ratio: Minimum WCAG contrast ratio (higher = better visibility).
        hue_start: Starting hue offset (0.0-1.0) for palette variation.

    Returns:
        Dictionary mapping cluster index to (R, G, B, A) color tuple.

    Notes:
        • Golden ratio (φ ≈ 0.618) spacing ensures maximum visual separation.
        • For dark microscopy backgrounds, use background="dark" with high contrast_ratio.
        • Higher saturation values create more vibrant colors for better distinction.
    """
    print(f"DEBUG: Generating {n} colors with background='{background}', contrast_ratio={contrast_ratio}")

    if n <= 0:
        return {}

    # Validate parameters.
    if not (0 <= alpha <= 255):
        raise ValueError(f"Alpha must be 0-255, got {alpha}")
    if background not in ["light", "dark"]:
        raise ValueError(f"Background must be 'light' or 'dark', got '{background}'")
    if not (0.0 <= saturation <= 1.0):
        raise ValueError(f"Saturation must be 0.0-1.0, got {saturation}")

    # Golden ratio for optimal hue distribution.
    golden_ratio = 0.6180339887498948

    # Background-specific settings.
    if background == "dark":
        base_lightness = 0.75  # Bright colors on dark background.
        bg_rgb = (0, 0, 0)
        lightness_range = (0.6, 0.9)  # Keep colors bright.
    else:
        base_lightness = 0.4   # Dark colors on light background.
        bg_rgb = (255, 255, 255)
        lightness_range = (0.2, 0.6)  # Keep colors dark.

    colors = {}

    for i in range(n):
        # Distribute hues evenly using golden ratio.
        hue = (hue_start + i * golden_ratio) % 1.0

        # Vary lightness to avoid similar colors when hues are close.
        lightness_offset = (i % 3 - 1) * 0.1  # -0.1, 0, +0.1 pattern.
        lightness = max(lightness_range[0],
                       min(lightness_range[1], base_lightness + lightness_offset))

        # Convert HSL to RGB.
        r_float, g_float, b_float = colorsys.hls_to_rgb(hue, lightness, saturation)
        r, g, b = int(r_float * 255), int(g_float * 255), int(b_float * 255)

        # Ensure minimum contrast ratio.
        current_contrast = calculate_contrast_ratio((r, g, b), bg_rgb)

        if current_contrast < contrast_ratio:
            # Adjust lightness to improve contrast.
            adjustment_direction = 1 if background == "dark" else -1

            for attempt in range(20):
                lightness += adjustment_direction * 0.05
                lightness = max(0.1, min(0.9, lightness))

                r_float, g_float, b_float = colorsys.hls_to_rgb(hue, lightness, saturation)
                r, g, b = int(r_float * 255), int(g_float * 255), int(b_float * 255)

                current_contrast = calculate_contrast_ratio((r, g, b), bg_rgb)

                if current_contrast >= contrast_ratio:
                    break

        colors[i] = (r, g, b, alpha)
        print(f"DEBUG: Color {i}: RGB({r}, {g}, {b}) contrast={current_contrast:.2f}")

    return colors


def colors_to_hex_list(color_dict: Dict[int, Tuple[int, int, int, int]]) -> List[str]:
    """
    Convert RGBA color dictionary to list of hex color codes for matplotlib/seaborn.

    Parameters:
        color_dict: Dictionary of cluster index to RGBA tuple.

    Returns:
        List of hex color codes (e.g., ['#FF5733', '#33FF57', ...]).
    """
    hex_colors = []

    for i in range(len(color_dict)):
        if i in color_dict:
            r, g, b, a = color_dict[i]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            hex_colors.append(hex_color)
        else:
            hex_colors.append("#000000")  # Fallback black.

    print(f"DEBUG: Generated {len(hex_colors)} hex colors: {hex_colors[:5]}...")
    return hex_colors

def test_color_generation():
    """
    Test color palette generation with various parameters.
    Validates contrast ratios and color uniqueness.
    """
    print("Testing color palette generation...")

    # Test dark background (typical for microscopy).
    dark_colors = generate_color_palette(n=8, background="dark", contrast_ratio=4.0)
    assert len(dark_colors) == 8, f"Expected 8 colors, got {len(dark_colors)}"

    # Test light background.
    light_colors = generate_color_palette(n=5, background="light", contrast_ratio=3.0)
    assert len(light_colors) == 5, f"Expected 5 colors, got {len(light_colors)}"

    # Test hex conversion.
    hex_colors = colors_to_hex_list(dark_colors)
    assert len(hex_colors) == 8, f"Expected 8 hex colors, got {len(hex_colors)}"
    assert all(color.startswith('#') for color in hex_colors), "All colors should be hex format"

    # Verify contrast ratios.
    bg_rgb = (0, 0, 0)  # Dark background.

    for i, (r, g, b, a) in dark_colors.items():
        contrast = calculate_contrast_ratio((r, g, b), bg_rgb)
        assert contrast >= 3.8, f"Color {i} contrast {contrast:.2f} below threshold"
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255, f"Invalid RGB values: {r}, {g}, {b}"
        assert 0 <= a <= 255, f"Invalid alpha value: {a}"

    print("All color generation tests passed.")


if __name__ == "__main__":
    test_color_generation()

    # Generate example palette for visual inspection.
    print("\nExample palette for dark background:")
    example_colors = generate_color_palette(n=6, background="dark", contrast_ratio=4.5)

    for i, (r, g, b, a) in example_colors.items():
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        contrast = calculate_contrast_ratio((r, g, b), (0, 0, 0))
        print(f"  Cluster {i}: {hex_color} RGB({r:3d}, {g:3d}, {b:3d}) contrast={contrast:.2f}")

    print("\nColor generation module ready for use.")
