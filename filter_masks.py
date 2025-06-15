#!/usr/bin/env python3
"""
python filter_masks.py \
  --input segmentation_masks.npy \
  --results-dir my_results \
  --output-prefix filtered_ \
  --min-pixels 20  --max-pixels 550 \
  --min-circularity 0.66  --max-circularity 1.0 \
  --min-solidity 0.80      --max-solidity 1.0 \
  --min-eccentricity 0.0   --max-eccentricity 0.98 \
  --min-aspect-ratio 0.5   --max-aspect-ratio 3.2 \
  --min-hole-fraction 0.0  --max-hole-fraction 0.001 \
  --max-straight-fraction 0.25 \
  --exclude-border \
  --summary-csv \
  --raw-image img/IRI_regist_cropped.tif \
  --overlay
"""

from __future__ import annotations

###############################################################################
# Imports & logging.
###############################################################################
import argparse
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.measure import find_contours
from skimage.color import gray2rgb
from skimage.io import imsave


logger = logging.getLogger("filter_masks")

###############################################################################
# Dataclass definitions.
###############################################################################

@dataclass
class Thresholds:
    """Numeric thresholds that determine which masks are kept.

    Every metric has a symmetric minimum and maximum bound so the CLI is
    predictable. Defaults are permissive, effectively disabling that metric
    unless the user supplies tighter values.
    """

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

    def validate(self) -> None:
        """Ensure that every lower bound is not greater than its upper bound."""
        assert 0 <= self.min_pixels <= self.max_pixels, "Pixel range is invalid."
        assert 0 <= self.min_circularity <= self.max_circularity <= 1, "Circularity range is invalid."
        assert 0 <= self.min_solidity <= self.max_solidity <= 1, "Solidity range is invalid."
        assert 0 <= self.min_eccentricity <= self.max_eccentricity <= 1, "Eccentricity range is invalid."
        assert 0 <= self.min_hole_fraction <= self.max_hole_fraction <= 1, "Hole-fraction range is invalid."

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
# CLI helpers.
###############################################################################

def add_threshold_args(parser: argparse.ArgumentParser, tmpl: Thresholds) -> None:
    """Create one `--flag` per Thresholds field automatically."""
    for field, default in asdict(tmpl).items():
        flag = f"--{field.replace('_', '-')}"
        if isinstance(default, bool):
            parser.add_argument(flag, action="store_true", help=f"Enable {field.replace('_', ' ')} filter.")
        else:
            parser.add_argument(flag, type=type(default), default=default, help=f"Threshold for {field.replace('_', ' ')}.")


def parse_cli() -> Config:
    parser = argparse.ArgumentParser(description="Filter segmentation masks by morphology and intensity.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input", required=True, help="Input .npy label map, binary stack, or object array.")
    parser.add_argument("--intensity-image", help="Optional grayscale image (.npy) for intensity thresholds.")
    parser.add_argument("-d", "--results-dir", default="filtered_mask_results", help="Directory for outputs.")
    parser.add_argument("-o", "--output-prefix", default="", help="Prefix for all output files.")

    add_threshold_args(parser, Thresholds())

    parser.add_argument("--no-plots", action="store_true", help="Skip violin QC plots.")
    parser.add_argument("--summary-csv", action="store_true", help="Save per-mask metrics as CSV & parquet.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging (DEBUG level).")
    parser.add_argument("--raw-image", help="Path to the original grayscale image (*.npy or *.tif).")
    parser.add_argument("--overlay", action="store_true", help="Save a red/green overlay of failed/passed masks.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[%(levelname)s] %(message)s")

    th = Thresholds(**{k: getattr(args, k) for k in Thresholds.__annotations__})
    th.validate()

    return Config(masks_path=Path(args.input),
                  intensity_path=Path(args.intensity_image) if args.intensity_image else None,
                  out_dir=Path(args.results_dir),
                  prefix=args.output_prefix,
                  save_plots=not args.no_plots,
                  save_summary_csv=args.summary_csv,
                  th=th,
                  raw_image=args.raw_image,
                  overlay=args.overlay)


###############################################################################
# Overlay helper.
###############################################################################


def save_overlay(
    raw: np.ndarray,
    masks: list[np.ndarray],
    passed: np.ndarray,
    out_path: Path,
    alpha: float = 0.35,   # Fraction of overlay colour to mix in.
) -> None:
    """
    Blend passed masks in translucent green and failed masks in translucent red.
    The original grayscale image remains visible beneath the overlay.
    """

    # Convert raw to uint8 RGBA in the 0-255 range.
    if raw.ndim == 2:
        base = gray2rgb((raw / raw.max() * 255).astype(np.uint8))
    elif raw.ndim == 3 and raw.shape[2] == 3:
        base = raw.astype(np.uint8)
    else:
        raise ValueError("Raw image must be grayscale (H×W) or RGB (H×W×3).")

    overlay = base.copy()
    green = np.array([0, 255, 0], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)

    # ------------------------------------------------------------------
    # Choose a vivid colour for every *passed* mask so neighbouring
    # objects are easy to tell apart. Failed masks stay red.
    # ------------------------------------------------------------------
    palette = np.array(
        [  # RGB tuples in uint8 range.
            [0, 255, 255],  # Cyan.
            [0, 0, 255],  # Blue.
            [0, 200, 255],
            [0, 255, 0],  # Green.
            [0, 150, 255],
            [0, 255, 200],
            [0, 200, 200],
            [0, 255, 150],
            [0, 200, 150],
            [0, 150, 200]
        ],
        dtype=np.uint8,
    )
    logger.info("Using %d colours for passed masks.", len(palette))

    for idx, m in enumerate(masks):
        if passed[idx]:
            colour = palette[idx % len(palette)]  # Cycle through palette.
        else:
            colour = red  # Failed mask => red.

        mask_bool = m.astype(bool)
        overlay[mask_bool] = (
                (1 - alpha) * overlay[mask_bool] + alpha * colour
        ).astype(np.uint8)

    imsave(out_path, overlay)  # skimage automatically saves TIFF format.
    logger.info("Saved translucent overlay → %s.", out_path)


###############################################################################
# Mask utilities.
###############################################################################

def load_masks(path: Path) -> List[np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
        labels = [l for l in np.unique(arr) if l]
        logger.info("Label map → %d masks.", len(labels))
        return [(arr == l).astype(np.uint8) for l in labels]
    if arr.ndim == 3:
        logger.info("Binary stack → %d masks.", arr.shape[0])
        return [(arr[i] > 0).astype(np.uint8) for i in range(arr.shape[0])]
    if arr.ndim == 2:
        logger.info("Single binary mask detected.")
        return [(arr > 0).astype(np.uint8)]
    
    def flatten(obj):
        out: List[np.ndarray] = []
        for el in np.asarray(obj, dtype=object).flat:
            if isinstance(el, np.ndarray):
                out.extend(flatten(el))
            elif isinstance(el, (list, tuple)):
                out.extend(flatten(np.array(el, dtype=object)))
            else:
                raise TypeError(f"Unsupported element {type(el)} in container.")
        return out
    
    masks = flatten(arr)
    logger.info("Nested container → %d masks.", len(masks))
    return masks

def split_small_masks(
    masks: list[np.ndarray],
    min_area: int = 1,
    min_perim: float = 1.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Separate masks that are too small to analyse.

    A mask is sent to *discarded* if either its pixel area is
    ≤ `min_area` **or** its perimeter (skimage.measure.perimeter)
    is ≤ `min_perim`. This prevents zero-area or one-pixel masks
    from entering the metrics table.

    Returns
    -------
    kept, discarded : list[np.ndarray], list[np.ndarray]
    """

    from skimage.measure import perimeter

    kept: list[np.ndarray] = []
    discarded: list[np.ndarray] = []

    for m in masks:
        area = int(m.sum())
        perim = perimeter(m.astype(bool))
        if area <= min_area or perim <= min_perim:
            discarded.append(m)
        else:
            kept.append(m)

    logger.info(
        "Discarded %d masks with area ≤ %d px or perimeter ≤ %.1f px.",
        len(discarded),
        min_area,
        min_perim,
    )
    return kept, discarded


###############################################################################
# Metric computation.
###############################################################################

def binary_list_to_label(masks: List[np.ndarray]) -> np.ndarray:
    label = np.zeros_like(masks[0], dtype=int)
    for i, m in enumerate(masks, 1):
        label[m.astype(bool)] = i
    return label

def straight_fraction(
    mask: np.ndarray,
    min_run: int = 4,
    angle_tol_deg: float = 2.0,
) -> float:
    """
    Return the fraction of boundary pixels that lie on long, almost-straight segments.

    A segment is considered straight when the direction change between two
    consecutive vectors stays below `angle_tol_deg` for at least `min_run`
    boundary steps. The result ranges from 0 (no straight edges) to 1
    (entire boundary is straight).
    """

    # Get a single sub-pixel contour for the mask.
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return 0.0
    pts = contours[0]  # Nx2 array (row, col).

    # Vectorised direction vectors between successive contour points.
    vecs = np.diff(pts, axis=0, append=pts[:1])

    # Normalise vectors and compute angle change between neighbours.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Prevent divide-by-zero.
    unit = vecs / norms
    dot = (unit * np.roll(unit, -1, axis=0)).sum(axis=1).clip(-1, 1)
    angles = np.degrees(np.arccos(dot))           # In degrees.

    straight_mask = angles < angle_tol_deg
    # Identify contiguous runs of straight segments.
    run_lengths = np.diff(np.where(np.concatenate(([0], straight_mask, [0])))[0])
    long_runs = run_lengths[run_lengths >= min_run]
    straight_px = long_runs.sum()

    return straight_px / len(pts)


def compute_metrics(masks: List[np.ndarray], intensity: np.ndarray | None = None) -> pd.DataFrame:
    props = ["area", "perimeter", "solidity", "eccentricity", "major_axis_length", "minor_axis_length", "filled_area", "bbox"]
    if intensity is not None:
        props.append("mean_intensity")

    df = pd.DataFrame(regionprops_table(binary_list_to_label(masks), properties=props, intensity_image=intensity)).rename_axis("mask_id")
    df = df.rename(columns={"bbox-0": "bbox_0", "bbox-1": "bbox_1", "bbox-2": "bbox_2", "bbox-3": "bbox_3"})

    df["aspect_ratio"] = df.major_axis_length / df.minor_axis_length.replace(0, np.nan)
    df["hole_fraction"] = (df.filled_area - df.area) / df.area
    df["straight_fraction"] = [straight_fraction(m) for m in masks] # Straight-edge fraction identifies masks with extended flat borders.
    df["circularity"] = (4 * np.pi * df.area / (df.perimeter.replace(0, np.nan) ** 2)).clip(0, 1)

    h, w = masks[0].shape
    df["border_touch"] = (df.bbox_0 == 0) | (df.bbox_2 == h) | (df.bbox_1 == 0) | (df.bbox_3 == w)

    logger.info("Computed metrics for %d masks.", len(df))
    return df


###############################################################################
# Filtering.
###############################################################################


def filter_masks(df: pd.DataFrame, th: Thresholds) -> np.ndarray:
    """Return a Boolean mask indicating which rows satisfy every threshold."""

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
        conds.append(
            df.mean_intensity.between(
                th.min_mean_intensity,
                th.max_mean_intensity,
                inclusive="both",
            )
        )

    if th.exclude_border:
        conds.append(~df.border_touch)

    passed = np.logical_and.reduce(conds)
    logger.info("%d / %d masks pass all thresholds.", passed.sum(), len(passed))
    return passed


###############################################################################
# Plotting.
###############################################################################


def violin_scatter(
    ax: plt.Axes,
    data: np.ndarray,
    lo: float,
    hi: float,
    ylabel: str,
    title: str
) -> None:
    """Draw a violin outline with jittered points and a colour bar."""

    # Violin outline.
    vp = ax.violinplot([data], positions=[1], showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor("none")
        body.set_edgecolor("black")

    # Jittered scatter coloured by value.
    rng = np.random.default_rng(0)
    sc = ax.scatter(
        rng.normal(1, 0.07, len(data)),
        data,
        c=data,
        cmap="viridis",
        s=10,
        edgecolors="none",
    )

    # Threshold lines.
    ax.hlines([lo, hi], 0.5, 1.5, linestyles="--", colors="black")
    ax.set_xticks([1])
    ax.set_xticklabels([title])
    ax.set_ylabel(ylabel)

    # Add colour bar on the right.
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=ylabel)



def save_violin_plots(df: pd.DataFrame, th: Thresholds, out_dir: Path, prefix: str) -> None:
    """Generate and save violin-scatter plots for every metric with finite bounds."""

    plots = [
        ("area", th.min_pixels, th.max_pixels, "Pixel Count", "Size"),
        ("circularity", th.min_circularity, th.max_circularity, "Circularity", "Circularity"),
        ("solidity", th.min_solidity, th.max_solidity, "Solidity", "Solidity"),
        ("eccentricity", th.min_eccentricity, th.max_eccentricity, "Eccentricity", "Eccentricity"),
        ("aspect_ratio", th.min_aspect_ratio, th.max_aspect_ratio, "Aspect Ratio", "Aspect Ratio"),
        ("hole_fraction", th.min_hole_fraction, th.max_hole_fraction, "Hole Fraction", "Hole Fraction"),
        ("straight_fraction", th.min_straight_fraction, th.max_straight_fraction, "Straight Fraction", "Straight Fraction"),
        ("mean_intensity", th.min_mean_intensity, th.max_mean_intensity, "Mean Intensity", "Mean Intensity"),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)

    for col, lo, hi, ylabel, title in plots:
        if col not in df.columns or (np.isneginf(lo) and np.isposinf(hi)):
            continue  # Skip metrics that are absent or completely unbounded.

        fig, ax = plt.subplots(figsize=(4, 6), constrained_layout=True)
        violin_scatter(ax, df[col].values, lo, hi, ylabel, title)

        # Write the PNG while silencing the NumPy det() warning that occasionally
        # appears when constrained-layout solves a single-axis figure.

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in det",
                module="numpy.linalg",
            )
            fig.savefig(out_dir / f"{prefix}{col}_violin.png", dpi=300)

        plt.close(fig)
        logger.info("Saved %s violin plot.", col)


###############################################################################
# Main.
###############################################################################


def main():
    cfg = parse_cli()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load masks and split off tiny / empty ones.
    # ------------------------------------------------------------------
    masks_raw = load_masks(cfg.masks_path)
    masks, discarded = split_small_masks(masks_raw)  # NEW helper call.

    # Persist discarded masks immediately.
    np.save(cfg.out_dir / f"{cfg.prefix}discarded_masks.npy", np.array(discarded, dtype=object))

    # ------------------------------------------------------------------
    # 2. Optional intensity image.
    # ------------------------------------------------------------------
    intensity = None
    if cfg.intensity_path:
        intensity = np.load(cfg.intensity_path)
        assert intensity.shape == masks[0].shape, "Intensity image shape mismatch."

    # ------------------------------------------------------------------
    # 3. Metrics, filtering, and downstream steps (unchanged).
    # ------------------------------------------------------------------
    df = compute_metrics(masks, intensity)
    passed = filter_masks(df, cfg.th)


    # Save masks.
    def dump(sel, name):
        np.save(cfg.out_dir / f"{cfg.prefix}{name}.npy", np.array([masks[i] for i in np.where(sel)[0]], dtype=object))
    dump(passed, "passed_masks"); dump(~passed, "failed_masks")

    # Summaries.
    if cfg.save_summary_csv:
        df.assign(passed=passed).to_csv(cfg.out_dir / f"{cfg.prefix}metrics.csv", index=False)
        df.assign(passed=passed).to_parquet(cfg.out_dir / f"{cfg.prefix}metrics.parquet")
        logger.info("Saved metrics table.")

    # QC plots.
    if cfg.save_plots:
        save_violin_plots(df, cfg.th, cfg.out_dir, cfg.prefix)

    # Overlay generation if requested by the user.
    if cfg.overlay:
        assert cfg.raw_image is not None, "Use --raw-image to supply the source image."
        if cfg.raw_image.endswith((".npy", ".NPY")):
            raw_img = np.load(cfg.raw_image)
        else:
            from skimage.io import imread
            raw_img = imread(cfg.raw_image)
        save_overlay(
            raw=raw_img,
            masks=masks,
            passed=passed,
            out_path=cfg.out_dir / f"{cfg.prefix}overlay.tif",
        )


if __name__ == "__main__":
    sys.exit(main())
