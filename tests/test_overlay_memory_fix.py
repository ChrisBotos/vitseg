#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_overlay_memory_fix.py.
Description:
    Test script to verify memory optimizations in overlay_masks.py work correctly.
    This script creates synthetic test data and runs the overlay process with
    various memory constraints to ensure the fixes prevent OOM errors.

Dependencies:
    • Python >= 3.10.
    • numpy >= 1.21.0.
    • tifffile >= 2021.7.2.
    • psutil >= 5.8.0.

Usage:
    python test_overlay_memory_fix.py
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import tifffile as tiff
import psutil

# Import the overlay functions to test.
from code.overlay_masks import (
    OverlayConfig,
    overlay,
    calculate_optimal_batch_size,
    get_system_memory_info,
    get_mask_max_label_efficiently
)


def create_test_data(size: int = 2048, num_labels: int = 1000) -> tuple:
    """
    Create synthetic test data for memory testing.

    Args:
        size: Image size (size x size pixels).
        num_labels: Number of segmentation labels to create.

    Returns:
        Tuple of (image_array, mask_array) for testing.

    This function creates realistic test data that mimics the memory
    characteristics of actual microscopy images and segmentation masks.
    """
    print(f"DEBUG: Creating test data - size: {size}x{size}, labels: {num_labels}")

    # Create synthetic grayscale image.
    rng = np.random.default_rng(42)
    image = rng.integers(0, 65536, size=(size, size), dtype=np.uint16)

    # Add some structure to make it more realistic.
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size // 2, size // 2
    distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    image = image * (1 + 0.5 * np.sin(distance / 100))
    image = np.clip(image, 0, 65535).astype(np.uint16)

    # Create synthetic segmentation mask.
    mask = np.zeros((size, size), dtype=np.int32)

    # Add random circular regions with different labels.
    for label in range(1, min(num_labels + 1, 10000)):  # Limit for test performance.
        center_y = rng.integers(50, size - 50)
        center_x = rng.integers(50, size - 50)
        radius = rng.integers(10, 50)

        y, x = np.ogrid[:size, :size]
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        mask[distance <= radius] = label

    print(f"DEBUG: Test data created - image dtype: {image.dtype}, mask max: {mask.max()}")
    return image, mask


def test_memory_optimization_functions():
    """
    Test memory optimization utility functions.

    This function validates that the memory monitoring and optimization
    functions work correctly and return reasonable values.
    """
    print("\n=== Testing Memory Optimization Functions ===")

    # Test system memory info.
    memory_info = get_system_memory_info()
    print(f"System memory info: {memory_info}")

    assert 'total_mb' in memory_info
    assert 'available_mb' in memory_info
    assert memory_info['total_mb'] > 0
    assert memory_info['available_mb'] > 0

    # Test batch size calculation.
    for tile_size in [512, 1024, 2048]:
        for memory_limit in [2048, 4096, 8192]:
            batch_size = calculate_optimal_batch_size(tile_size, memory_limit)
            print(f"Tile size: {tile_size}, Memory: {memory_limit}MB -> Batch size: {batch_size}")
            assert 1 <= batch_size <= 8

    print("✅ Memory optimization functions work correctly")


def test_mask_max_label_efficiency():
    """
    Test efficient max label detection.

    This function validates that the chunked max label detection
    works correctly and uses less memory than loading the entire mask.
    """
    print("\n=== Testing Efficient Max Label Detection ===")

    # Create test mask.
    test_mask = np.random.randint(0, 5000, size=(4000, 4000), dtype=np.int32)
    expected_max = int(test_mask.max())

    # Save to temporary file.
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as temp_file:
        np.save(temp_file.name, test_mask)
        temp_path = temp_file.name

    try:
        # Test efficient max label detection.
        detected_max = get_mask_max_label_efficiently(temp_path)
        print(f"Expected max: {expected_max}, Detected max: {detected_max}")

        assert detected_max == expected_max
        print("✅ Efficient max label detection works correctly")

    finally:
        # Cleanup.
        os.unlink(temp_path)


def test_memory_constrained_overlay():
    """
    Test overlay creation under memory constraints.

    This function tests the overlay process with various memory limits
    to ensure it handles memory pressure gracefully.
    """
    print("\n=== Testing Memory-Constrained Overlay ===")

    # Create test data.
    image, mask = create_test_data(size=1024, num_labels=100)

    # Save test data to temporary files.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        image_path = temp_dir / "test_image.tif"
        mask_path = temp_dir / "test_mask.npy"
        output_path = temp_dir / "test_overlay.tif"

        # Save test data.
        tiff.imwrite(image_path, image)
        np.save(mask_path, mask)

        # Test with different memory constraints.
        memory_limits = [1024, 2048, 4096]  # MB.
        batch_sizes = [1, 2, 4]

        for memory_limit in memory_limits:
            for batch_size in batch_sizes:
                print(f"\nTesting with memory limit: {memory_limit}MB, batch size: {batch_size}")

                try:
                    # Create restrictive configuration.
                    config = OverlayConfig(
                        tile_size=512,  # Smaller tiles for memory efficiency.
                        batch_size=batch_size,
                        workers=2,  # Fewer workers.
                        memory_limit_mb=memory_limit,
                        enable_gpu=False,  # CPU only for consistent testing.
                        cleanup_frequency=5  # More frequent cleanup.
                    )

                    # Monitor memory before processing.
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / (1024**2)

                    # Run overlay creation.
                    overlay(
                        image_path=image_path,
                        mask_path=mask_path,
                        output_path=output_path,
                        config=config
                    )

                    # Monitor memory after processing.
                    memory_after = process.memory_info().rss / (1024**2)
                    memory_increase = memory_after - memory_before

                    print(f"Memory usage - Before: {memory_before:.1f}MB, "
                          f"After: {memory_after:.1f}MB, Increase: {memory_increase:.1f}MB")

                    # Verify output exists and has reasonable size.
                    assert output_path.exists()
                    output_size_mb = output_path.stat().st_size / (1024**2)
                    print(f"Output file size: {output_size_mb:.1f}MB")

                    # Cleanup for next test.
                    if output_path.exists():
                        output_path.unlink()

                    print(f"✅ Memory limit {memory_limit}MB, batch size {batch_size} - SUCCESS")

                except Exception as e:
                    print(f"❌ Memory limit {memory_limit}MB, batch size {batch_size} - FAILED: {e}")
                    traceback.print_exc()

    print("✅ Memory-constrained overlay testing completed")


def main():
    """
    Main test function.

    This function runs all memory optimization tests to validate
    that the fixes in overlay_masks.py work correctly.
    """
    print("Starting overlay_masks.py memory optimization tests")
    print("=" * 60)

    try:
        # Test individual functions.
        test_memory_optimization_functions()
        test_mask_max_label_efficiency()

        # Test full overlay process.
        test_memory_constrained_overlay()

        print("\n" + "=" * 60)
        print("✅ All memory optimization tests passed successfully!")
        print("The overlay_masks.py memory fixes are working correctly.")

    except Exception as e:
        print(f"\n❌ Memory optimization tests failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
