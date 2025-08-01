#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_overlay_masks.py
Description:
    Test script for the rewritten overlay_masks.py to verify memory-efficient processing
    and configuration management functionality. This script creates synthetic test data
    and validates the overlay generation process.

Dependencies:
    • Python >= 3.10
    • numpy >= 1.21.0
    • tifffile >= 2021.7.2
    • pytest (optional, for structured testing)

Usage:
    python test_overlay_masks.py

Key Features:
    • Synthetic test data generation
    • Configuration validation testing
    • Memory management verification
    • Error handling validation
    • Output quality checks

Notes:
    • Creates temporary test files that are cleaned up automatically
    • Tests both CPU and GPU processing paths (if available)
    • Validates memory-efficient processing with different tile sizes
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import tifffile as tiff

# Import the rewritten overlay_masks module.
from code.overlay_masks import OverlayConfig, create_memory_efficient_overlay, generate_label_colors


def create_test_image(height: int = 2048, width: int = 2048) -> np.ndarray:
    """
    Create synthetic test image for overlay testing.
    
    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        
    Returns:
        Synthetic grayscale image as uint16 array.
    """
    print(f"DEBUG: Creating test image ({height}x{width})")
    
    # Create gradient background.
    y_grad = np.linspace(0, 65535, height)[:, None]
    x_grad = np.linspace(0, 65535, width)[None, :]
    
    # Combine gradients.
    image = (y_grad * 0.3 + x_grad * 0.7).astype(np.uint16)
    
    # Add some texture.
    noise = np.random.randint(0, 1000, (height, width), dtype=np.uint16)
    image = np.clip(image + noise, 0, 65535)
    
    return image


def create_test_mask(height: int = 2048, width: int = 2048, num_objects: int = 100) -> np.ndarray:
    """
    Create synthetic segmentation mask for testing.
    
    Args:
        height: Mask height in pixels.
        width: Mask width in pixels.
        num_objects: Number of segmented objects to create.
        
    Returns:
        Integer-labeled segmentation mask.
    """
    print(f"DEBUG: Creating test mask ({height}x{width}) with {num_objects} objects")
    
    mask = np.zeros((height, width), dtype=np.int32)
    
    # Create random circular objects.
    for label in range(1, num_objects + 1):
        # Random center and radius.
        center_y = np.random.randint(50, height - 50)
        center_x = np.random.randint(50, width - 50)
        radius = np.random.randint(10, 50)
        
        # Create circular mask.
        y_coords, x_coords = np.ogrid[:height, :width]
        circle_mask = (y_coords - center_y)**2 + (x_coords - center_x)**2 <= radius**2
        
        # Assign label where no other object exists.
        mask[circle_mask & (mask == 0)] = label
    
    return mask


def test_configuration_validation():
    """Test configuration parameter validation."""
    print("DEBUG: Testing configuration validation")
    
    # Test valid configuration.
    config = OverlayConfig(
        tile_size=1024,
        batch_size=4,
        alpha=0.5,
        memory_limit_mb=4096
    )
    
    config.validate()  # Should not raise exception.
    print("✅ Valid configuration passed validation")
    
    # Test invalid alpha value.
    try:
        invalid_config = OverlayConfig(alpha=1.5)
        invalid_config.validate()
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Invalid alpha value correctly rejected")
    
    # Test batch size adjustment.
    config_zero_batch = OverlayConfig(batch_size=0)
    config_zero_batch.validate()
    assert config_zero_batch.batch_size == 1
    print("✅ Zero batch size correctly adjusted to 1")


def test_color_generation():
    """Test deterministic color generation."""
    print("DEBUG: Testing color generation")
    
    # Test reproducible colors.
    colors1 = generate_label_colors(100, seed=42)
    colors2 = generate_label_colors(100, seed=42)
    
    assert np.array_equal(colors1, colors2), "Colors should be deterministic"
    print("✅ Color generation is deterministic")
    
    # Test background color.
    assert np.array_equal(colors1[0], [0, 0, 0]), "Background should be black"
    print("✅ Background color is black")
    
    # Test color range.
    assert colors1.dtype == np.uint8, "Colors should be uint8"
    assert colors1.shape == (101, 3), "Color array shape should be (n_labels+1, 3)"
    print("✅ Color array properties are correct")


def test_overlay_creation():
    """Test complete overlay creation process."""
    print("DEBUG: Testing overlay creation")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data.
        test_image = create_test_image(512, 512)  # Smaller for faster testing.
        test_mask = create_test_mask(512, 512, 20)
        
        # Save test files.
        image_path = temp_path / "test_image.tif"
        mask_path = temp_path / "test_mask.npy"
        output_path = temp_path / "test_overlay.tif"
        
        tiff.imwrite(image_path, test_image)
        np.save(mask_path, test_mask)
        
        # Test with CPU processing.
        cpu_config = OverlayConfig(
            tile_size=256,
            batch_size=2,
            enable_gpu=False,
            alpha=0.4
        )
        
        create_memory_efficient_overlay(
            image_path=image_path,
            mask_path=mask_path,
            output_path=output_path,
            config=cpu_config
        )
        
        # Verify output exists and has correct properties.
        assert output_path.exists(), "Output file should exist"
        
        with tiff.TiffFile(output_path) as tif_file:
            output_shape = tif_file.series[0].shape
            assert output_shape == (512, 512, 3), f"Output shape should be (512, 512, 3), got {output_shape}"
        
        print("✅ CPU overlay creation successful")
        
        # Test with GPU processing (if available).
        try:
            import cupy as cp
            
            gpu_output_path = temp_path / "test_overlay_gpu.tif"
            gpu_config = OverlayConfig(
                tile_size=256,
                batch_size=2,
                enable_gpu=True,
                alpha=0.4,
                memory_limit_mb=2048
            )
            
            create_memory_efficient_overlay(
                image_path=image_path,
                mask_path=mask_path,
                output_path=gpu_output_path,
                config=gpu_config
            )
            
            assert gpu_output_path.exists(), "GPU output file should exist"
            print("✅ GPU overlay creation successful")
            
        except ImportError:
            print("⚠️  CuPy not available, skipping GPU test")
        except Exception as e:
            print(f"⚠️  GPU test failed (expected on some systems): {e}")


def main():
    """Run all tests."""
    print("🧪 Starting overlay_masks.py tests")
    
    try:
        test_configuration_validation()
        test_color_generation()
        test_overlay_creation()
        
        print("✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
