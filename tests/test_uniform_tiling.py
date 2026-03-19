"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_uniform_tiling.py.
Description:
    Test script for uniform tiling ViT feature extraction functionality.
    Creates a synthetic binary mask image and tests the uniform tiling pipeline
    to ensure proper integration with the existing clustering infrastructure.

Dependencies:
    • Python >= 3.10.
    • numpy, PIL, pathlib.

Usage:
    python test_uniform_tiling.py
"""

import traceback
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile
import shutil


def create_synthetic_binary_mask(width: int = 512, height: int = 512, num_regions: int = 20) -> np.ndarray:
    """
    Create a synthetic binary mask image for testing uniform tiling.
    
    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        num_regions (int): Number of white regions to create.

    Returns:
        np.ndarray: Binary mask array (0 = background, 255 = foreground).
    """
    print(f"DEBUG: Creating synthetic binary mask ({width}x{height}) with {num_regions} regions")
    
    # Start with black background.
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Add random white regions to simulate nuclei.
    np.random.seed(42)  # For reproducible test results.

    for i in range(num_regions):
        # Random center position.
        center_x = np.random.randint(50, width - 50)
        center_y = np.random.randint(50, height - 50)
        
        # Random region size.
        radius = np.random.randint(10, 30)
        
        # Create circular region.
        y, x = np.ogrid[:height, :width]
        mask_region = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask[mask_region] = 255
    
    print(f"DEBUG: Created mask with {np.sum(mask > 0)} foreground pixels")
    return mask


def test_uniform_tiling_extraction():
    """
    Test the uniform tiling ViT feature extraction pipeline.
    
    Returns:
        True if test passes, False otherwise.
    """
    print("="*60)
    print("TESTING UNIFORM TILING VIT EXTRACTION")
    print("="*60)
    
    try:
        # Create temporary directory for test files.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            print(f"DEBUG: Using temporary directory: {temp_path}")
            
            # Create synthetic binary mask.
            mask_array = create_synthetic_binary_mask(width=256, height=256, num_regions=10)
            
            # Save as TIFF image.
            mask_image_path = temp_path / "test_binary_mask.tif"
            mask_image = Image.fromarray(mask_array, mode='L')
            mask_image.save(mask_image_path, format='TIFF')
            print(f"DEBUG: Saved test mask to {mask_image_path}")
            
            # Create output directory.
            output_dir = temp_path / "test_output"
            output_dir.mkdir(exist_ok=True)
            
            # Test parameters.
            patch_size = 32
            stride = 32
            batch_size = 16
            
            print(f"DEBUG: Testing with patch_size={patch_size}, stride={stride}, batch_size={batch_size}")
            
            # Import and test the uniform tiling module.
            try:
                import sys
                sys.path.append('.')  # Add current directory to path.
                from uniform_tiling_vit import extract_uniform_features, UniformTileDataset
                import torch
                from transformers import ViTImageProcessor, ViTModel
                
                print("DEBUG: Successfully imported uniform tiling modules")
                
                # Check if we can load the ViT model.
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"DEBUG: Using device: {device}")
                
                model_name = "facebook/dino-vits16"
                print(f"DEBUG: Loading ViT model: {model_name}")
                
                processor = ViTImageProcessor.from_pretrained(model_name)
                model = ViTModel.from_pretrained(model_name, output_hidden_states=True, use_safetensors=True)
                
                print("DEBUG: ViT model loaded successfully")
                
                # Test the dataset creation.
                test_image = Image.open(mask_image_path)
                dataset = UniformTileDataset(test_image, patch_size, stride)
                
                expected_tiles = ((256 - patch_size) // stride + 1) ** 2
                actual_tiles = len(dataset)
                
                print(f"DEBUG: Expected {expected_tiles} tiles, got {actual_tiles}")
                
                if actual_tiles != expected_tiles:
                    print(f"ERROR: Tile count mismatch")
                    return False
                
                # Test feature extraction (with small batch for speed).
                print("DEBUG: Testing feature extraction...")
                extract_uniform_features(
                    image_path=mask_image_path,
                    model=model,
                    processor=processor,
                    patch_size=patch_size,
                    stride=stride,
                    batch_size=batch_size,
                    device=device,
                    output_dir=output_dir,
                    workers=1  # Single worker for testing.
                )
                
                # Check output files.
                expected_files = [
                    f"features_test_binary_mask.csv",
                    f"features_test_binary_mask.npy",
                    f"coords_test_binary_mask.csv"
                ]
                
                for filename in expected_files:
                    filepath = output_dir / filename
                    if not filepath.exists():
                        print(f"ERROR: Expected output file not found: {filepath}")
                        return False
                    print(f"DEBUG: Found expected output file: {filename}")
                
                # Validate file contents.
                import pandas as pd
                
                coords_df = pd.read_csv(output_dir / "coords_test_binary_mask.csv")
                features_df = pd.read_csv(output_dir / "features_test_binary_mask.csv")
                features_npy = np.load(output_dir / "features_test_binary_mask.npy")
                
                print(f"DEBUG: Coordinates shape: {coords_df.shape}")
                print(f"DEBUG: Features CSV shape: {features_df.shape}")
                print(f"DEBUG: Features NPY shape: {features_npy.shape}")
                
                # Check consistency.
                if len(coords_df) != len(features_df) or len(coords_df) != len(features_npy):
                    print("ERROR: Inconsistent number of samples between output files")
                    return False
                
                if features_df.shape[0] != actual_tiles:
                    print(f"ERROR: Feature count {features_df.shape[0]} doesn't match tile count {actual_tiles}")
                    return False
                
                print("DEBUG: All output files have consistent dimensions")
                
                # Check coordinate ranges.
                x_coords = coords_df['x_center'].values
                y_coords = coords_df['y_center'].values
                
                if np.min(x_coords) < patch_size // 2 or np.max(x_coords) >= 256 - patch_size // 2:
                    print("ERROR: X coordinates out of expected range")
                    return False
                    
                if np.min(y_coords) < patch_size // 2 or np.max(y_coords) >= 256 - patch_size // 2:
                    print("ERROR: Y coordinates out of expected range")
                    return False
                
                print("DEBUG: Coordinate ranges are valid")
                
                print("✓ Uniform tiling extraction test PASSED")
                return True
                
            except ImportError as e:
                print(f"ERROR: Failed to import required modules: {e}")
                return False
            except Exception as e:
                print(f"ERROR: Feature extraction failed: {e}")
                traceback.print_exc()
                return False
                
    except Exception as e:
        print(f"ERROR: Test setup failed: {e}")
        traceback.print_exc()
        return False


def test_clustering_compatibility():
    """
    Test compatibility with the clustering pipeline.
    
    Returns:
        True if test passes, False otherwise.
    """
    print("\n" + "="*60)
    print("TESTING CLUSTERING COMPATIBILITY")
    print("="*60)
    
    try:
        # Test import of clustering module.
        import sys
        sys.path.append('.')
        from cluster_uniform_tiles_memopt import parse_arguments, load_features
        
        print("DEBUG: Successfully imported clustering module")
        
        # Test argument parsing.
        test_args = [
            '--image', 'test.tif',
            '--coords', 'coords.csv',
            '--features_npy', 'features.npy',
            '--clusters', '5',
            '--outdir', 'results'
        ]
        
        # Temporarily modify sys.argv for testing.
        original_argv = sys.argv
        sys.argv = ['cluster_uniform_tiles_memopt.py'] + test_args
        
        try:
            args = parse_arguments()
            print("DEBUG: Argument parsing successful")
            
            # Check that all required arguments are present.
            required_attrs = ['image', 'coords', 'features_npy', 'clusters', 'outdir']
            for attr in required_attrs:
                if not hasattr(args, attr):
                    print(f"ERROR: Missing required argument: {attr}")
                    return False
            
            print("DEBUG: All required arguments present")
            
        finally:
            sys.argv = original_argv
        
        print("✓ Clustering compatibility test PASSED")
        return True
        
    except Exception as e:
        print(f"ERROR: Clustering compatibility test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test execution function."""
    print("UNIFORM TILING PIPELINE TEST SUITE")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Uniform tiling extraction.
    if not test_uniform_tiling_extraction():
        all_tests_passed = False
    
    # Test 2: Clustering compatibility.
    if not test_clustering_compatibility():
        all_tests_passed = False
    
    print("\n" + "="*60)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED")
        print("The uniform tiling pipeline is ready for use.")
        print("\nTo use uniform tiling in your pipeline:")
        print("1. Set USE_DYNAMIC_PATCHES=False in pipeline.sh")
        print("2. Ensure RUN_BINARY_CONVERSION=True")
        print("3. Run the pipeline as usual")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the error messages above and fix any issues.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
