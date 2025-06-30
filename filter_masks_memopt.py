"""
Author: Christos Botos.
Affiliation: Institute of Molecular Biology and Biotechnology
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: filter_masks_memopt.py.
Description:
    Drop‑in replacement for *filter_masks.py* that eliminates the per‑nucleus
    full‑frame binary copies that previously caused >200GB peaks for large
    label maps.  The new version works **directly on the label image**, streams
    metrics in constant RAM, and writes the accepted / rejected nuclei to
    memory‑mapped boolean stacks so that downstream scripts remain unchanged.

Key Improvements:
    • `np.load(..., mmap_mode="r")` ensures the label map is never duplicated.
    • Metrics computed via `skimage.measure.regionprops_table` on the label map
      (O(1) extra memory) instead of converting to a list of masks.
    • `straight_fraction` is evaluated *per nucleus* in a for‑loop so only one
      temporary mask is alive at any time.
    • Output boolean stacks are created as `np.memmap`, filled incrementally,
      and finally flushed to `.npy` – peak RAM is ≈ 2× the image size, no more.
    • Overlay generation colours each nucleus on‑the‑fly; no stack in memory.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pandas, matplotlib, scikit‑image.

Usage Example:
    python filter_masks_memopt.py \
        --input segmentation_masks.npy \
        --results-dir filtered_results \
        --output-prefix filtered_ \
        --min-pixels 20 --max-pixels 570 \
        --max-straight-fraction 0.25 \
        --summary-csv --overlay --raw-image img/IRI_regist_cropped.tif

Tests:
    Run `pytest filter_masks_memopt.py -q` to execute the built‑in unit tests.
"""
from __future__ import annotations

###############################################################################
# Imports & logging.
###############################################################################
import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
from skimage.color import gray2rgb
from skimage.io import imsave
from skimage.measure import find_contours, regionprops_table

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("filter‑masks‑memopt")

###############################################################################
# Dataclasses.
###############################################################################

@dataclass
class Thresholds:
    """Symmetric min‑max thresholds for every metric."""

    # Size and compactness.
    min_pixels: int = 0
    max_pixels: int = int(1e9)
    min_circularity: float = 0.0
    max_circularity: float = 1.0

    # Solidity.
    min_solidity: float = 0.0
    max_solidity: float = 1.0

    # Elongation.
    min_eccentricity: float = 0.0
    max_eccentricity: float = 1.0
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = float("inf")

    # Raggedness.
    min_hole_fraction: float = 0.0
    max_hole_fraction: float = 1.0

    # Straightness.
    min_straight_fraction: float = 0.0
    max_straight_fraction: float = 1.0

    # Intensity.
    min_mean_intensity: float = float("-inf")
    max_mean_intensity: float = float("inf")

    # Miscellaneous.
    exclude_border: bool = False

    # ------------------------ Validation ------------------------ #
    def validate(self) -> None:
        assert 0 <= self.min_pixels <= self.max_pixels, "Invalid pixel range."
        assert 0 <= self.min_circularity <= self.max_circularity <= 1, "Invalid circularity range."
        assert 0 <= self.min_solidity <= self.max_solidity <= 1, "Invalid solidity range."
        assert 0 <= self.min_eccentricity <= self.max_eccentricity <= 1, "Invalid eccentricity range."
        assert 0 <= self.min_hole_fraction <= self.max_hole_fraction <= 1, "Invalid hole‑fraction range."


@dataclass
class Config:
    masks_path: Path
    intensity_path: Path | None
    out_dir: Path
    prefix: str
    save_plots: bool
    save_summary_csv: bool
    th: Thresholds
    raw_image: str | None
    overlay: bool

###############################################################################
# Helper functions – metrics.
###############################################################################

def straight_fraction_from_coords(coords: np.ndarray, min_run: int = 4, angle_tol: float = 2.0) -> float:
    """Return the fraction of boundary pixels that lie on long almost‑straight runs."""

    # vectors between successive points (row, col)
    vecs = np.diff(coords, axis=0, append=coords[:1])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit = vecs / norms
    dot = (unit * np.roll(unit, -1, axis=0)).sum(axis=1).clip(-1, 1)
    angles = np.degrees(np.arccos(dot))
    straight = angles < angle_tol
    runs = np.diff(np.where(np.concatenate(([0], straight, [0])))[0])
    long_runs = runs[runs >= min_run]
    return long_runs.sum() / len(coords) if len(coords) else 0.0


def compute_metrics(label_map: np.ndarray, intensity: np.ndarray | None) -> pd.DataFrame:
    """Compute morphology metrics for every non‑zero label in *label_map*."""

    props = [
        "label",
        "area",
        "perimeter",
        "solidity",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "filled_area",
        "bbox",
    ]
    if intensity is not None:
        props.append("mean_intensity")

    df = pd.DataFrame(
        regionprops_table(label_map, properties=props, intensity_image=intensity)
    ).rename_axis("mask_id")

    df = df.rename(columns={
        "bbox-0": "bbox_0",
        "bbox-1": "bbox_1",
        "bbox-2": "bbox_2",
        "bbox-3": "bbox_3",
    })

    # Derived metrics.
    df["aspect_ratio"] = df.major_axis_length / df.minor_axis_length.replace(0, np.nan)
    df["hole_fraction"] = (df.filled_area - df.area) / df.area
    df["circularity"] = (4 * np.pi * df.area / (df.perimeter.replace(0, np.nan) ** 2)).clip(0, 1)

    # Straight fraction per nucleus (streamed).
    straight_vals: List[float] = []
    for lab in df.label.astype(int):
        mask = label_map == lab
        contours = find_contours(mask.astype(float), 0.5)
        straight_vals.append(
            straight_fraction_from_coords(contours[0]) if contours else 0.0
        )
    df["straight_fraction"] = straight_vals

    h, w = label_map.shape
    df["border_touch"] = (
        (df.bbox_0 == 0) | (df.bbox_2 == h) | (df.bbox_1 == 0) | (df.bbox_3 == w)
    )

    LOGGER.info("Computed metrics for %d nuclei.", len(df))
    return df


def apply_thresholds(df: pd.DataFrame, th: Thresholds) -> np.ndarray:
    """Return Boolean mask of rows that satisfy *th*."""

    conds = [
        df.area.between(th.min_pixels, th.max_pixels),
        df.circularity.between(th.min_circularity, th.max_circularity),
        df.solidity.between(th.min_solidity, th.max_solidity),
        df.eccentricity.between(th.min_eccentricity, th.max_eccentricity),
        df.aspect_ratio.between(th.min_aspect_ratio, th.max_aspect_ratio),
        df.hole_fraction.between(th.min_hole_fraction, th.max_hole_fraction),
        df.straight_fraction.between(th.min_straight_fraction, th.max_straight_fraction),
    ]
    if "mean_intensity" in df.columns:
        conds.append(df.mean_intensity.between(th.min_mean_intensity, th.max_mean_intensity))
    if th.exclude_border:
        conds.append(~df.border_touch)

    passed = np.logical_and.reduce(conds)
    LOGGER.info("%d / %d nuclei pass all thresholds.", passed.sum(), len(passed))
    return passed

###############################################################################
# Overlay generation (streamed – constant memory).
###############################################################################

def paint_overlay(
    raw: np.ndarray,
    label_map: np.ndarray,
    passed_labels: set[int],
    out_path: Path,
    alpha: float = 0.35,
) -> None:
    """Colour passed nuclei green and failed red without loading mask stack."""

    if raw.ndim == 2:
        base = gray2rgb((raw / raw.max() * 255).astype(np.uint8))
    elif raw.ndim == 3 and raw.shape[2] == 3:
        base = raw.astype(np.uint8)
    else:
        raise ValueError("Raw image must be grayscale H×W or RGB H×W×3.")

    overlay = base.copy()
    green = np.array([0, 255, 0], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)

    unique_labels = np.unique(label_map)
    unique_labels = unique_labels[unique_labels != 0]

    for lab in unique_labels:
        colour = green if lab in passed_labels else red
        mask = label_map == lab
        overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * colour).astype(np.uint8)

    imsave(str(out_path), overlay)
    LOGGER.info("Overlay saved → %s", out_path.name)

###############################################################################
# I/O helpers.
###############################################################################

def load_label_map(path: Path) -> np.ndarray:
    """Return a 2‑D integer label map, loading with mmap when possible."""

    arr = np.load(path, allow_pickle=True, mmap_mode="r")
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
        return arr
    if arr.ndim == 3:
        # Binary stack ➜ need to pack into label map first for streaming.
        label = np.zeros(arr.shape[1:], dtype=np.int32)
        for i, sl in enumerate(arr, 1):
            label[sl.astype(bool)] = i
        return label
    raise ValueError("Unsupported mask array shape – expected label map or stack.")


def write_masks(
    label_map: np.ndarray,
    labels: List[int],
    out_file: Path,
) -> None:
    """Write the selected labels to *out_file* as a Boolean stack.

    The function streams one nucleus at a time into a memory-mapped array
    created with *open_memmap*, which writes a valid .npy header up front.
    This prevents the ‘UnpicklingError: invalid load key 0x01’ that occurs
    when using bare np.memmap without a header.
    """
    h, w = label_map.shape
    n = len(labels)
    LOGGER.info("Writing %d masks → %s.", n, out_file.name)

    # Create a .npy file with a correct header and obtain a writable memmap.
    mm = open_memmap(out_file, mode="w+", dtype=np.bool_, shape=(n, h, w))

    # Fill the memmap slice by slice. Only one H×W array is resident at a time.
    for idx, lab in enumerate(labels):
        mm[idx] = label_map == lab

    mm.flush()   # Ensure header and data are flushed to disk.
    del mm       # Close the memmap and release the file handle.

###############################################################################
# CLI parsing.
###############################################################################

def add_threshold_args(p: argparse.ArgumentParser) -> None:
    for f, default in asdict(Thresholds()).items():
        flag = f"--{f.replace('_', '-')}"
        if isinstance(default, bool):
            p.add_argument(flag, action="store_true")
        else:
            p.add_argument(flag, type=type(default), default=default)


def parse_cli() -> Config:
    a = argparse.ArgumentParser(
        description="Memory‑efficient mask filtering by morphology and intensity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    a.add_argument("-i", "--input", required=True, type=Path, help="Input label map or binary stack (.npy).")
    a.add_argument("--intensity-image", type=Path, help="Optional grayscale image (.npy) for intensity filters.")
    a.add_argument("-d", "--results-dir", type=Path, default=Path("filtered_results"))
    a.add_argument("-o", "--output-prefix", default="")
    a.add_argument("--no-plots", action="store_true")
    a.add_argument("--summary-csv", action="store_true")
    a.add_argument("--raw-image", type=Path, help="Raw microscopy image for overlays (tif or npy).")
    a.add_argument("--overlay", action="store_true")
    a.add_argument("--verbose", action="store_true")

    add_threshold_args(a)
    args = a.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    th = Thresholds(**{k: getattr(args, k) for k in Thresholds.__annotations__})
    th.validate()

    return Config(
        masks_path=args.input,
        intensity_path=args.intensity_image,
        out_dir=args.results_dir,
        prefix=args.output_prefix,
        save_plots=not args.no_plots,
        save_summary_csv=args.summary_csv,
        th=th,
        raw_image=str(args.raw_image) if args.raw_image else None,
        overlay=args.overlay,
    )

###############################################################################
# Main entry point.
###############################################################################

def main() -> None:  # noqa: C901 – Linear CLI flow.
    cfg = parse_cli()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # 1 ▸ Load label map & optional intensity image (both via mmap).
    label_map = load_label_map(cfg.masks_path)
    intensity = None
    if cfg.intensity_path:
        intensity = np.load(cfg.intensity_path, mmap_mode="r")
        assert intensity.shape == label_map.shape, "Intensity image shape mismatch."

    # 2 ▸ Metrics & filtering (constant RAM).
    metrics_df = compute_metrics(label_map, intensity)
    passed_mask = apply_thresholds(metrics_df, cfg.th)

    passed_labels = metrics_df.label[passed_mask].astype(int).tolist()
    failed_labels = metrics_df.label[~passed_mask].astype(int).tolist()

    # 3 ▸ Write outputs – streamed.
    if passed_labels:
        write_masks(label_map, passed_labels, cfg.out_dir / f"{cfg.prefix}passed_masks.npy")
    if failed_labels:
        write_masks(label_map, failed_labels, cfg.out_dir / f"{cfg.prefix}failed_masks.npy")

    if cfg.save_summary_csv:
        metrics_df.assign(passed=passed_mask).to_csv(cfg.out_dir / f"{cfg.prefix}metrics.csv", index=False)
        LOGGER.info("Metrics CSV written.")

    # 4 ▸ Overlay (optional).
    if cfg.overlay:
        assert cfg.raw_image is not None, "--raw-image required for overlay."
        if str(cfg.raw_image).lower().endswith(".npy"):
            raw = np.load(cfg.raw_image, mmap_mode="r")
        else:
            from skimage.io import imread
            raw = imread(cfg.raw_image)
        paint_overlay(raw, label_map, set(passed_labels), cfg.out_dir / f"{cfg.prefix}overlay.tif")

    LOGGER.info("✓ Done.")

###############################################################################
# Minimal unit tests (pytest).
###############################################################################

def _synthetic_label_map() -> Tuple[np.ndarray, np.ndarray]:
    """Return a 100×100 label map with two circles and two squares."""
    lab = np.zeros((100, 100), np.int32)
    rr, cc = np.ogrid[:100, :100]
    circle1 = (rr - 25) ** 2 + (cc - 25) ** 2 <= 10 ** 2
    circle2 = (rr - 75) ** 2 + (cc - 25) ** 2 <= 8 ** 2
    square1 = (20 <= rr) & (rr <= 40) & (60 <= cc) & (cc <= 80)
    square2 = (60 <= rr) & (rr <= 80) & (60 <= cc) & (cc <= 80)
    lab[circle1] = 1
    lab[circle2] = 2
    lab[square1] = 3
    lab[square2] = 4
    intensity = (lab > 0).astype(np.float32) * 100  # uniform intensity
    return lab, intensity


def test_compute_metrics_shapes():
    lab, inten = _synthetic_label_map()
    df = compute_metrics(lab, inten)
    assert df.shape[0] == 4
    required_cols = {"area", "circularity", "straight_fraction"}
    assert required_cols.issubset(df.columns)


def test_apply_thresholds():
    lab, _ = _synthetic_label_map()
    df = compute_metrics(lab, None)
    th = Thresholds(min_pixels=50, max_pixels=4000)
    passed = apply_thresholds(df, th)
    # All four nuclei have area ≈ 300‑400 px so should pass.
    assert passed.sum() == 4


if __name__ == "__main__":
    sys.exit(main())
