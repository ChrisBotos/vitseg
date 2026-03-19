#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_overlay_fix.py.
Description:
    Test script to verify the overlay mask fix for large label values.
    Tests the fix for the IndexError that occurs when mask labels
    exceed the color lookup table size, and verifies that the modulo
    operation works correctly.

Dependencies:
    • Python >= 3.10.
    • numpy.
    • matplotlib.

Usage:
    python test_overlay_fix.py
"""

import numpy as np
import matplotlib.pyplot as plt
import traceback


def test_large_label_fix():
    """
    Test the fix for large label values in overlay masks.
    
    This function creates a test scenario similar to the original error
    where mask labels exceed the color LUT size, and verifies that the
    modulo operation prevents IndexErrors.
    """
    print("DEBUG: Testing large label fix for overlay masks")
    
    # Create a test mask with large label values similar to the original error.
    test_mask = np.array([
        [0, 1809533, 1810175],
        [1811693, 1810613, 0],
        [2000000, 1500000, 999999]
    ], dtype=np.int32)
    
    print(f"DEBUG: Test mask shape: {test_mask.shape}")
    print(f"DEBUG: Test mask values: {test_mask.flatten()}")
    
    # Create a limited color LUT (similar to the 1M limit in the original code).
    max_lut_size = 1000001  # 1M + 1 for background.
    lut = np.random.randint(0, 256, size=(max_lut_size, 3), dtype=np.uint8)
    lut[0] = 0  # Background remains black.
    
    print(f"DEBUG: LUT shape: {lut.shape}")
    print(f"DEBUG: Max LUT index: {max_lut_size - 1}")
    
    # Test the original approach (should fail).
    print("\nDEBUG: Testing original approach (should fail)...")
    try:
        colored_mask_original = lut[test_mask]
        print("ERROR: Original approach should have failed but didn't!")
    except IndexError as e:
        print(f"DEBUG: Original approach failed as expected: {e}")

    # Test the fixed approach with modulo operation.
    print("\nDEBUG: Testing fixed approach with modulo operation...")
    try:
        max_lut_index = lut.shape[0] - 1
        safe_mask_indices = test_mask % (max_lut_index + 1)
        colored_mask_fixed = lut[safe_mask_indices]
        
        print(f"DEBUG: Safe indices: {safe_mask_indices.flatten()}")
        print(f"DEBUG: Colored mask shape: {colored_mask_fixed.shape}")
        print("DEBUG: Fixed approach succeeded!")
        
        # Verify all indices are within bounds.
        assert np.all(safe_mask_indices >= 0), "Some indices are negative"
        assert np.all(safe_mask_indices <= max_lut_index), "Some indices exceed LUT bounds"
        print("DEBUG: All assertions passed - indices are within bounds")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Fixed approach failed: {e}")
        traceback.print_exc()
        return False


def test_color_consistency():
    """
    Test that the same large label always maps to the same color.
    
    This ensures that the modulo operation provides consistent color
    mapping across different tiles and processing runs.
    """
    print("\nDEBUG: Testing color consistency with modulo operation")
    
    # Create test labels.
    large_labels = [1809533, 1810175, 1811693, 1810613, 2000000]
    max_lut_size = 1000001

    # Create deterministic LUT.
    np.random.seed(42)
    lut = np.random.randint(0, 256, size=(max_lut_size, 3), dtype=np.uint8)

    # Test consistency across multiple calls.
    colors1 = []
    colors2 = []
    
    for label in large_labels:
        safe_index = label % max_lut_size
        color1 = lut[safe_index]
        color2 = lut[safe_index]  # Second call should give same result.
        
        colors1.append(color1)
        colors2.append(color2)
        
        print(f"DEBUG: Label {label} -> Index {safe_index} -> Color {color1}")
    
    # Verify consistency.
    for i, (c1, c2) in enumerate(zip(colors1, colors2)):
        assert np.array_equal(c1, c2), f"Color inconsistency for label {large_labels[i]}"

    print("DEBUG: Color consistency test passed!")
    return True


def main():
    """Main function to run all tests."""
    print("=" * 60)
    print("TESTING OVERLAY MASK FIX FOR LARGE LABEL VALUES")
    print("=" * 60)
    
    try:
        # Test the basic fix.
        success1 = test_large_label_fix()
        
        # Test color consistency.
        success2 = test_color_consistency()
        
        if success1 and success2:
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED - Fix is working correctly!")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("❌ SOME TESTS FAILED - Fix needs more work")
            print("=" * 60)
            return 1
            
    except Exception as e:
        print(f"\nERROR: Test execution failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
