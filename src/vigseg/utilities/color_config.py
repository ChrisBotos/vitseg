"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: color_config.py.
Description:
    Configuration system for color palette management in scientific visualization.
    Provides centralized control over color generation parameters and custom palettes.

Dependencies:
    • Python >= 3.10.
    • typing (standard library).
    • pathlib (standard library).
    • json (standard library).

Usage:
    from vigseg.utilities.color_config import ColorConfig, load_color_config
    config = load_color_config("config.json")
    colors = config.generate_palette(n=10)
"""
import logging
import traceback
import json
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)

from vigseg.utilities.color_generation import generate_color_palette


@dataclass
class ColorConfig:
    """
    Configuration class for color palette generation and management.
    
    This class centralizes all color-related parameters and provides methods
    for generating color palettes with consistent settings across the pipeline.
    All parameters follow lowercase naming conventions for consistency.
    """
    # Core color generation parameters.
    background: str = "dark"                    # Background type: "light" or "dark".
    alpha: int = 255                           # Alpha transparency (0-255, 255=opaque).
    saturation: float = 0.95                   # Color saturation (0.0-1.0, higher=more vivid).
    contrast_ratio: float = 4.5                # Minimum WCAG contrast ratio.
    hue_start: float = 0.0                     # Starting hue offset (0.0-1.0).
    
    # Custom color palette options.
    custom_colors: Optional[List[str]] = None   # List of hex color codes.
    custom_rgb_colors: Optional[List[Tuple[int, int, int]]] = None  # List of RGB tuples.
    
    # Advanced generation parameters.
    use_predefined: bool = True                # Whether to use predefined vibrant colors.
    max_predefined: int = 20                   # Maximum number of predefined colors to use.
    
    # Validation and fallback settings.
    validate_contrast: bool = True             # Whether to validate contrast ratios.
    fallback_to_algorithmic: bool = True       # Whether to use algorithmic fallback.
    
    def __post_init__(self):
        """
        Validate configuration parameters after initialization.
        
        Ensures all parameters are within valid ranges and converts custom
        RGB colors to hex format for consistency.
        """
        # Validate core parameters.
        if self.background not in ["light", "dark"]:
            raise ValueError(f"background must be 'light' or 'dark', got '{self.background}'")
        
        if not (0 <= self.alpha <= 255):
            raise ValueError(f"alpha must be 0-255, got {self.alpha}")
            
        if not (0.0 <= self.saturation <= 1.0):
            raise ValueError(f"saturation must be 0.0-1.0, got {self.saturation}")
            
        if not (1.0 <= self.contrast_ratio <= 21.0):
            raise ValueError(f"contrast_ratio must be 1.0-21.0, got {self.contrast_ratio}")
            
        if not (0.0 <= self.hue_start <= 1.0):
            raise ValueError(f"hue_start must be 0.0-1.0, got {self.hue_start}")
        
        # Convert RGB tuples to hex colors if provided.
        if self.custom_rgb_colors:
            if not self.custom_colors:
                self.custom_colors = []
            
            for r, g, b in self.custom_rgb_colors:
                if not all(0 <= val <= 255 for val in [r, g, b]):
                    raise ValueError(f"RGB values must be 0-255, got ({r}, {g}, {b})")
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                self.custom_colors.append(hex_color)
        
        # Validate custom hex colors.
        if self.custom_colors:
            validated_colors = []
            for color in self.custom_colors:
                color = color.strip()
                if not color.startswith('#'):
                    color = '#' + color
                
                if len(color) != 7:
                    LOGGER.warning(f"Invalid hex color '{color}', skipping")
                    continue

                try:
                    int(color[1:], 16)  # Validate hex format.
                    validated_colors.append(color.upper())
                except ValueError:
                    LOGGER.warning(f"Invalid hex color '{color}', skipping")
                    continue
            
            self.custom_colors = validated_colors
            LOGGER.debug(f"Validated {len(self.custom_colors)} custom colors")

    def generate_palette(self, n: int) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Generate color palette using current configuration.
        
        Args:
            n: Number of colors to generate.
            
        Returns:
            Dictionary mapping color index to (R, G, B, A) tuple.
            
        This method applies all configuration settings to generate a color
        palette optimized for scientific visualization contexts.
        """
        LOGGER.debug(f"Generating palette with ColorConfig: n={n}, background={self.background}")
        LOGGER.debug(f"Config - alpha={self.alpha}, saturation={self.saturation}, contrast_ratio={self.contrast_ratio}")
        
        if self.custom_colors:
            LOGGER.debug(f"Using {len(self.custom_colors)} custom colors: {self.custom_colors[:3]}...")
        
        return generate_color_palette(
            n=n,
            alpha=self.alpha,
            background=self.background,
            saturation=self.saturation,
            contrast_ratio=self.contrast_ratio,
            hue_start=self.hue_start,
            custom_colors=self.custom_colors
        )

    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of configuration.
        """
        return {
            'background': self.background,
            'alpha': self.alpha,
            'saturation': self.saturation,
            'contrast_ratio': self.contrast_ratio,
            'hue_start': self.hue_start,
            'custom_colors': self.custom_colors,
            'custom_rgb_colors': self.custom_rgb_colors,
            'use_predefined': self.use_predefined,
            'max_predefined': self.max_predefined,
            'validate_contrast': self.validate_contrast,
            'fallback_to_algorithmic': self.fallback_to_algorithmic
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ColorConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters.
            
        Returns:
            ColorConfig instance with specified parameters.
        """
        return cls(**config_dict)


def load_color_config(config_path: Union[str, Path] = None) -> ColorConfig:
    """
    Load color configuration from file or return default configuration.
    
    Args:
        config_path: Path to JSON configuration file (optional).
        
    Returns:
        ColorConfig instance with loaded or default parameters.
        
    This function provides flexible configuration loading with graceful
    fallback to defaults when configuration files are not available.
    """
    if config_path is None:
        LOGGER.debug("No config path provided, using default ColorConfig")
        return ColorConfig()
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        LOGGER.debug(f"Config file {config_path} not found, using default ColorConfig")
        return ColorConfig()
    
    try:
        LOGGER.debug(f"Loading color configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Extract color-specific configuration if nested.
        if 'colors' in config_data:
            config_data = config_data['colors']
        elif 'color_config' in config_data:
            config_data = config_data['color_config']
        
        config = ColorConfig.from_dict(config_data)
        LOGGER.debug("Successfully loaded color configuration")
        return config
        
    except Exception as e:
        LOGGER.warning(f"Failed to load color config from {config_path}: {e}")
        LOGGER.debug("Using default ColorConfig")
        return ColorConfig()


def save_color_config(config: ColorConfig, config_path: Union[str, Path]) -> None:
    """
    Save color configuration to JSON file.
    
    Args:
        config: ColorConfig instance to save.
        config_path: Path where to save the configuration.
        
    This function enables users to save their custom color configurations
    for reuse across different analysis sessions.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        LOGGER.debug(f"Color configuration saved to {config_path}")
        
    except Exception as e:
        print(f"ERROR: Failed to save color config to {config_path}: {e}")
        raise


def create_example_config() -> ColorConfig:
    """
    Create example configuration with custom colors for demonstration.
    
    Returns:
        ColorConfig with example custom color palette.
        
    This function provides a template for users to understand how to
    configure custom color palettes for their specific needs.
    """
    example_colors = [
        "#FF0000",  # Strong red.
        "#00FF00",  # Neon green.
        "#0080FF",  # Bright blue.
        "#FF00FF",  # Magenta.
        "#FF8C00",  # Deep orange.
        "#00FFFF",  # Cyan.
        "#FFFF00",  # Yellow.
        "#8000FF",  # Purple.
    ]
    
    return ColorConfig(
        background="dark",
        alpha=200,
        saturation=0.95,
        contrast_ratio=5.0,
        custom_colors=example_colors
    )


if __name__ == "__main__":
    # Test configuration system.
    print("Testing color configuration system...")
    
    # Test default configuration.
    default_config = ColorConfig()
    colors = default_config.generate_palette(n=5)
    print(f"Generated {len(colors)} colors with default config")
    
    # Test custom color configuration.
    custom_config = create_example_config()
    custom_colors = custom_config.generate_palette(n=8)
    print(f"Generated {len(custom_colors)} colors with custom config")
    
    # Test configuration serialization.
    test_path = Path("test_color_config.json")
    save_color_config(custom_config, test_path)
    loaded_config = load_color_config(test_path)
    
    # Cleanup test file.
    if test_path.exists():
        test_path.unlink()
    
    print("Color configuration system tests completed successfully.")
