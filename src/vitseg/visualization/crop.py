#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: crop.py.
Description:
    Crop the IRI_regist.tif image in ./data using a flexible crop function.
    Supports both relative (0-1) and absolute pixel coordinates.

Dependencies:
    • Python >= 3.10.
    • numpy, tifffile.

Usage:
    python -m vitseg.visualization.crop
"""

import os
import logging
import numpy as np
import tifffile

from vitseg.utilities.logging_setup import setup_logging


def crop_image(image: np.ndarray, crop_cell_centered_patch, logger=None) -> np.ndarray:
    """Crop image to a user-defined bounding box.

    Args:
        image (np.ndarray): Input image.
        crop_cell_centered_patch (tuple): Tuple of (y0, y1, x0, x1), either relative (0-1) or absolute.
        logger (logging.Logger): Logger object.

    Returns:
        np.ndarray: Cropped image.

    Raises:
        ValueError: If the resulting crop dimensions are invalid.
    """
    h, w = image.shape[:2]
    y0, y1, x0, x1 = crop_cell_centered_patch

    # Interpret relative coordinates if all values are in [0, 1].
    if all(0 <= val <= 1 for val in crop_cell_centered_patch):
        y0, y1 = int(y0 * h), int(y1 * h)
        x0, x1 = int(x0 * w), int(x1 * w)
        if logger:
            logger.info(f"Cropping with relative coordinates -> rows {y0}:{y1}, cols {x0}:{x1}")
    else:
        # Absolute pixel coordinates.
        y0, y1, x0, x1 = map(int, crop_cell_centered_patch)
        if logger:
            logger.info(f"Cropping with absolute coordinates -> rows {y0}:{y1}, cols {x0}:{x1}")

    # Clamp to image bounds.
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)

    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Invalid crop dimensions: y=[{y0}:{y1}], x=[{x0}:{x1}]")

    return image[y0:y1, x0:x1]


def main():
    """Crop the IRI_regist.tif image using predefined relative coordinates."""
    # Set up logging.
    _, __ = setup_logging("crop")
    logger = logging.getLogger(__name__)

    # Locate files relative to the repo root (script is at src/vitseg/visualization/crop.py).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    img_dir = os.path.join(repo_root, "data")
    input_path = os.path.join(img_dir, "IRI_regist.tif")
    output_path = os.path.join(img_dir, "IRI_regist_cropped.tif")

    # Check that the input file exists.
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Reading image: {input_path}")
    image = tifffile.imread(input_path)

    # Define a crop box using relative coordinates (ymin, ymax, xmin, xmax).
    crop_cell_centered_patch = (0.5, 0.52, 0.66, 0.68)

    logger.info("Performing crop.")
    cropped = crop_image(image, crop_cell_centered_patch, logger=logger)

    logger.info(f"Saving cropped image to: {output_path}")
    tifffile.imwrite(output_path, cropped)

    logger.info("Done.")


if __name__ == "__main__":
    main()
