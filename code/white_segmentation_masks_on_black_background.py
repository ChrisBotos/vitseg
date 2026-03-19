"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: white_segmentation_masks_on_black_background.py.
Description:
    Generate a binary mask image for ViT input where pixels inside any mask region
    are set to 1 and all other pixels to 0.

Dependencies:
    • Python >= 3.10.
    • numpy, pillow.

Usage:
    python code/white_segmentation_masks_on_black_background.py \\
        --mask masks/segmentation_masks.npy \\
        --output data/binary_mask.tif
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(mask_path: Path) -> np.ndarray:
    """Load mask array and collapse to a 2D boolean mask.

    Args:
        mask_path (Path): Path to the numpy mask file.

    Returns:
        np.ndarray: 2D boolean mask where True indicates a masked region.

    Raises:
        ValueError: If the mask array has an unsupported shape.
    """
    masks = np.load(mask_path, mmap_mode='r')
    # If stack of masks.
    if masks.ndim == 3:
        # If object type, convert to boolean.
        if masks.dtype == object:
            masks = np.stack([m.astype(bool) for m in masks], axis=0)
        # Collapse across stack: any True indicates mask.
        binary = np.any(masks, axis=0)
    elif masks.ndim == 2:
        # Label map: any label > 0 indicates mask.
        if masks.dtype == bool:
            binary = masks
        else:
            binary = masks > 0
    else:
        raise ValueError(f'Unsupported mask array shape: {masks.shape}')
    return binary


def save_binary_image(binary_mask: np.ndarray, output_path: Path) -> None:
    """Save the 2D boolean mask as a binary image (0 or 255) in TIFF format.

    Args:
        binary_mask (np.ndarray): 2D boolean mask.
        output_path (Path): Path to save the binary TIFF image.
    """
    # Convert boolean mask to uint8 0/255.
    img_array = binary_mask.astype(np.uint8) * 255
    img = Image.fromarray(img_array, mode='L')
    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as TIFF.
    img.save(output_path, format='TIFF')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with mask and output paths.
    """
    parser = argparse.ArgumentParser(
        description='Create binary TIFF mask image for ViT input.'
    )
    parser.add_argument(
        '--mask', type=Path, required=True,
        help='Path to the numpy file containing masks.'
    )
    parser.add_argument(
        '--output', type=Path, required=True,
        help='Path to save the binary mask image (TIFF).'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    binary = load_mask(args.mask)

    save_binary_image(binary, args.output)


if __name__ == '__main__':
    main()
