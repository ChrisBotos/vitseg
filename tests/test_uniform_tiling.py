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
    • numpy, PIL, pathlib, pytest.

Usage:
    pytest tests/test_uniform_tiling.py -v
"""

import sys
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile


def create_synthetic_binary_mask(width: int = 512, height: int = 512, num_regions: int = 20) -> np.ndarray:
    """
    Create a synthetic binary mask image for testing uniform tiling.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        num_regions: Number of white regions to create.

    Returns:
        Binary mask array (0 = background, 255 = foreground).
    """
    # Start with black background.
    mask = np.zeros((height, width), dtype=np.uint8)

    # Add random white regions to simulate nuclei.
    rng = np.random.RandomState(42)

    for i in range(num_regions):
        center_x = rng.randint(50, width - 50)
        center_y = rng.randint(50, height - 50)
        radius = rng.randint(10, 30)

        y, x = np.ogrid[:height, :width]
        mask_region = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask[mask_region] = 255

    return mask


def test_uniform_tiling_extraction():
    """Test the uniform tiling ViT feature extraction pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create synthetic binary mask.
        mask_array = create_synthetic_binary_mask(width=256, height=256, num_regions=10)

        # Save as TIFF image.
        mask_image_path = temp_path / "test_binary_mask.tif"
        mask_image = Image.fromarray(mask_array, mode='L')
        mask_image.save(mask_image_path, format='TIFF')

        # Create output directory.
        output_dir = temp_path / "test_output"
        output_dir.mkdir(exist_ok=True)

        # Test parameters.
        patch_size = 32
        stride = 32
        batch_size = 16
        image_size = 256

        from uniform_tiling_vit import extract_uniform_features, UniformTileDataset
        import torch
        from transformers import ViTImageProcessor, ViTModel

        # Load ViT model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "facebook/dino-vits16"
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name, output_hidden_states=True, use_safetensors=True)

        # Test dataset creation.
        test_image = Image.open(mask_image_path)
        dataset = UniformTileDataset(test_image, patch_size, stride)

        # Ceiling division: number of grid cells that cover the image.
        expected_tiles_x = (image_size + stride - 1) // stride
        expected_tiles_y = (image_size + stride - 1) // stride
        expected_tiles = expected_tiles_x * expected_tiles_y
        assert len(dataset) == expected_tiles, (
            f"Tile count mismatch: expected {expected_tiles}, got {len(dataset)}"
        )

        # Test feature extraction.
        extract_uniform_features(
            image_path=mask_image_path,
            model=model,
            processor=processor,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            workers=1
        )

        # Check output files exist.
        expected_files = [
            "features_test_binary_mask.csv",
            "features_test_binary_mask.npy",
            "coords_test_binary_mask.csv"
        ]

        for filename in expected_files:
            filepath = output_dir / filename
            assert filepath.exists(), f"Expected output file not found: {filepath}"

        # Validate file contents.
        import pandas as pd
        coords_df = pd.read_csv(output_dir / "coords_test_binary_mask.csv")
        features_df = pd.read_csv(output_dir / "features_test_binary_mask.csv")
        features_npy = np.load(output_dir / "features_test_binary_mask.npy")

        # Check consistency across output formats.
        assert len(coords_df) == len(features_df), "Coords and CSV features row count mismatch"
        assert len(coords_df) == len(features_npy), "Coords and NPY features row count mismatch"
        assert features_df.shape[0] == expected_tiles, (
            f"Feature count {features_df.shape[0]} doesn't match tile count {expected_tiles}"
        )

        # Check coordinate ranges. Centres can be up to stride // 2 past the
        # image boundary (the last grid cell's centre for an image that is an
        # exact multiple of the stride).
        x_coords = coords_df['x_center'].values
        y_coords = coords_df['y_center'].values

        max_center = (expected_tiles_x - 1) * stride + stride // 2
        min_center = stride // 2

        assert np.min(x_coords) >= min_center, (
            f"Min X coordinate {np.min(x_coords)} below expected {min_center}"
        )
        assert np.max(x_coords) <= max_center, (
            f"Max X coordinate {np.max(x_coords)} above expected {max_center}"
        )
        assert np.min(y_coords) >= min_center, (
            f"Min Y coordinate {np.min(y_coords)} below expected {min_center}"
        )
        assert np.max(y_coords) <= max_center, (
            f"Max Y coordinate {np.max(y_coords)} above expected {max_center}"
        )


def test_clustering_compatibility():
    """Test compatibility with the clustering pipeline."""
    from cluster_uniform_tiles_memopt import parse_arguments

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

        # Check that all required arguments are present.
        required_attrs = ['image', 'coords', 'features_npy', 'clusters', 'outdir']
        for attr in required_attrs:
            assert hasattr(args, attr), f"Missing required argument: {attr}"
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    import pytest
    exit(pytest.main([__file__, "-v"]))
