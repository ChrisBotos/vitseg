"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: overlay_masks.py
Description:
    Overlay an integer-label mask (.npy) on a very large microscopy TIFF.
    Processing is tile-wise, parallel, and optional GPU-accelerated, so RAM
    stays bounded by a single tile.  The output is first written as a
    contiguous BigTIFF mem-map; workers fill their rectangles in-place.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tifffile as tiff
from tqdm import tqdm
import multiprocessing as mp

# ---------- helper functions -------------------------------------------------

def _generate_label_colours(max_label: int, seed: int = 42) -> np.ndarray:
    """Return a deterministic RGB colour for every label from 0..max_label."""
    rng = np.random.default_rng(seed)
    lut = rng.integers(0, 256, size=(max_label + 1, 3), dtype=np.uint8)
    lut[0] = 0      # background stays black
    return lut


def _blend_tile(
    tile_img: np.ndarray,
    tile_mask: np.ndarray,
    lut: np.ndarray,
    alpha: float,
    *,
    gpu: bool = False,
) -> np.ndarray:
    """
    Colour-map *tile_mask* and alpha-blend it onto *tile_img*.

    If *gpu* is True we try CuPy first.  Any failure to import CuPy or to
    access a working CUDA runtime triggers an automatic, silent fallback to
    pure NumPy on the CPU, so the caller never has to handle GPU errors.
    """
    # Decide which array backend to use.
    xp = np  # Default to NumPy.
    if gpu:
        try:
            import cupy as cp
            # Accessing a trivial CUDA property forces the driver to load.
            _ = cp.cuda.runtime.getDeviceCount()
            xp = cp
        except Exception:
            # Any import or runtime error → fall back to NumPy.
            xp = np

    # Convert inputs to the chosen backend.
    img = xp.asarray(tile_img, dtype=xp.float32)
    mask = xp.asarray(tile_mask, dtype=xp.int32)
    lut_xp = xp.asarray(lut, dtype=xp.float32)

    # Map labels to RGB and alpha-blend.
    coloured = lut_xp[mask]                       # Shape (H, W, 3).
    blended = img * (1.0 - alpha) + coloured * alpha
    blended = xp.clip(blended, 0, 255).astype(xp.uint8)

    # Ensure the result is a NumPy array for the caller.
    return xp.asnumpy(blended) if xp is not np else blended


# ---------- worker -----------------------------------------------------------

def _process_tile(
    args: Tuple[int, int, int, int, str, str, np.ndarray, float, bool]
) -> Tuple[int, int, int, int, np.ndarray]:
    """Read one tile from disk, blend mask, and return the RGB result."""
    y0, y1, x0, x1, img_path, mask_path, lut, alpha, gpu = args

    # Lazy, read-only memory-maps: only the sliced region is paged into RAM.
    img_mm = tiff.memmap(img_path, mode="r")
    tile_img = img_mm[y0:y1, x0:x1]

    mask_mm = np.load(mask_path, mmap_mode="r")
    tile_mask = mask_mm[y0:y1, x0:x1]

    if tile_img.ndim == 2:                  # promote grayscale to RGB
        tile_img = np.repeat(tile_img[..., None], 3, axis=2)

    blended = _blend_tile(tile_img, tile_mask, lut, alpha, gpu=gpu)
    return y0, y1, x0, x1, blended


# ---------- main overlay routine --------------------------------------------

def overlay(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    out_path: Union[str, Path],
    *,
    tile: int = 1024,
    workers: Union[int, str, None] = "auto",
    alpha: float = 0.4,
    seed: int = 42,
    gpu: bool = False,
) -> None:
    """
    Overlay a labelled segmentation mask on a large microscopy TIFF.

    The function streams the job tile by tile, so peak RAM usage is bounded by
    a single tile plus a small buffer.  Tiles are processed in parallel across
    CPU processes, with an optional GPU path that falls back gracefully if the
    CUDA tool-chain is unavailable.  After all tiles are written into a sparse
    BigTIFF mem-map, the file is repacked into a fully populated, tiled,
    JPEG-compressed BigTIFF that any standard viewer can open.
    """
    import multiprocessing as mp
    import os

    image_path = Path(image_path)
    mask_path = Path(mask_path)
    out_path = Path(out_path)

    # Inspect dimensions without reading the entire image.
    with tiff.TiffFile(image_path, mode="r") as tif:
        height, width = tif.series[0].shape[:2]

    mask_mm = np.load(mask_path, mmap_mode="r")
    if mask_mm.shape != (height, width):
        raise ValueError("Image and mask dimensions do not match.")

    lut = _generate_label_colours(int(mask_mm.max()), seed)

    # Create an empty, contiguous BigTIFF mem-map for in-place writing.
    tiff.memmap(
        out_path,
        shape=(height, width, 3),
        dtype=np.uint8,
        photometric="rgb",
        bigtiff=True,
    )

    # Build the list of tile tasks.
    tasks = [
        (
            y0,
            min(y0 + tile, height),
            x0,
            min(x0 + tile, width),
            str(image_path),
            str(mask_path),
            lut,
            alpha,
            gpu,
        )
        for y0 in range(0, height, tile)
        for x0 in range(0, width, tile)
    ]

    # Resolve the number of worker processes.
    if isinstance(workers, str):
        n_workers = os.cpu_count() if workers.lower() == "auto" else int(workers)
    else:
        n_workers = workers or os.cpu_count()
    if n_workers is None or n_workers < 1:
        raise ValueError("Number of workers must be a positive integer.")

    # Process tiles in parallel using the spawn context.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        out_map = tiff.memmap(out_path, mode="r+")
        try:
            for y0, y1, x0, x1, blended in tqdm(
                executor.map(_process_tile, tasks), total=len(tasks)
            ):
                out_map[y0:y1, x0:x1] = blended
        finally:
            out_map.flush()
            del out_map

    # Repack the sparse mem-map into a fully allocated, viewer-friendly TIFF.
    tmp_path = out_path.with_suffix(".repack.tif")
    repack_mm = tiff.memmap(out_path, mode="r")

    try:
        import imagecodecs  # noqa: F401
        compression = "jpeg"
    except ImportError:
        compression = "none"

    tiff.imwrite(
        tmp_path,
        repack_mm,
        bigtiff=True,
        tile=(tile, tile),
        compression=compression,
        photometric="rgb",
    )

    os.replace(tmp_path, out_path)
    sys.stdout.write(f"✅ Overlay complete → {out_path}\n")





# ---------- CLI --------------------------------------------------------------

def _get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Overlay segmentation masks onto huge TIFFs tile-wise."
    )
    p.add_argument("--image", required=True, help="Path to input TIFF.")
    p.add_argument("--mask", required=True, help="Path to label mask (.npy).")
    p.add_argument("--out", default="overlay.tif", help="Output BigTIFF path.")
    p.add_argument("--tile", type=int, default=1024, help="Tile edge length.")
    p.add_argument("--workers", default="auto",
                   help="'auto' or an integer worker count.")
    p.add_argument("--alpha", type=float, default=0.4,
                   help="Overlay transparency in [0, 1].")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible colours.")
    p.add_argument("--gpu", action="store_true",
                   help="Enable CuPy acceleration when available.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _get_parser().parse_args(argv)
    overlay(
        image_path=args.image,
        mask_path=args.mask,
        out_path=args.out,
        tile=args.tile,
        workers=args.workers,
        alpha=args.alpha,
        seed=args.seed,
        gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
