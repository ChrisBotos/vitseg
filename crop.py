#!/usr/bin/env python3
"""
A minimal script to crop the IRI_regist.tif image in ./img using a flexible crop function.
"""

import os
import logging
import numpy as np
import tifffile

def crop_image(image: np.ndarray, crop_cell_centered_patch, logger=None) -> np.ndarray:
    """
    Crop image to a user-defined bounding box.

    Args:
        image (np.ndarray): Input image.
        crop_cell_centered_patch: Tuple of (y0, y1, x0, x1), either relative (0–1) or absolute.
        logger: Logger object.

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape[:2]
    y0, y1, x0, x1 = crop_cell_centered_patch

    # interpret relative cell_centered_patch
    if all(0 <= val <= 1 for val in crop_cell_centered_patch):
        y0, y1 = int(y0 * h), int(y1 * h)
        x0, x1 = int(x0 * w), int(x1 * w)
        if logger:
            logger.info(f"Cropping with relative cell_centered_patch → rows {y0}:{y1}, cols {x0}:{x1}")
    else:
        # absolute coords
        y0, y1, x0, x1 = map(int, crop_cell_centered_patch)
        if logger:
            logger.info(f"Cropping with absolute cell_centered_patch → rows {y0}:{y1}, cols {x0}:{x1}")

    # clamp to image bounds
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)

    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Invalid crop dimensions: y=[{y0}:{y1}], x=[{x0}:{x1}]")

    return image[y0:y1, x0:x1]

def main():
    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)

    # locate files relative to this script
    base_dir = os.path.abspath(os.path.dirname(__file__))
    img_dir  = os.path.join(base_dir, "img")
    input_path  = os.path.join(img_dir, "IRI_regist.tif")
    output_path = os.path.join(img_dir, "IRI_regist_cropped.tif")

    # check input
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Reading image: {input_path}")
    image = tifffile.imread(input_path)

    # define a crop box: relative coordinates xmin,xmax,ymin,ymax.
    crop_cell_centered_patch = (0.5,0.52,0.66,0.68)

    logger.info("Performing crop")
    cropped = crop_image(image, crop_cell_centered_patch, logger=logger)

    logger.info(f"Saving cropped image to: {output_path}")
    tifffile.imwrite(output_path, cropped)

    logger.info("Done.")

if __name__ == "__main__":
    main()
