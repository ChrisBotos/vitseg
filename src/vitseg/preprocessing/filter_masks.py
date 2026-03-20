"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: filter_masks.py.
Description:
    Memory-efficient filtering of segmentation masks based on morphological and intensity
    metrics. This script eliminates the per-nucleus full-frame binary copies that
    previously caused >200GB memory peaks for large label maps. The implementation works
    directly on the label image, streams metrics in constant RAM, and writes the
    accepted / rejected nuclei to memory-mapped boolean stacks.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, matplotlib, scikit-image.

Usage:
    python filter_masks.py \
        --input segmentation_masks.npy \
        --results-dir filtered_results \
        --output-prefix filtered_ \
        --min-pixels 20 --max-pixels 570 \
        --summary-csv --overlay --raw-image data/IRI_regist_cropped.tif
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
import warnings

import numpy as np
from numpy.lib.format import open_memmap
import matplotlib
import json
import pandas as pd
from skimage.color import gray2rgb
from skimage.io import imsave
from typing import Tuple
from skimage.measure import regionprops_table

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("filter‑masks")

###############################################################################
# Dataclasses.
###############################################################################

@dataclass
class Thresholds:
    """Symmetric min-max thresholds for every metric."""

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
    region: Tuple[float, float, float, float] | None = None
    no_stack: bool = True

###############################################################################
# Helper functions – metrics.
###############################################################################

def compute_metrics(label_map: np.ndarray, intensity: np.ndarray | None) -> pd.DataFrame:
    """Compute morphology metrics for every non-zero label in *label_map*."""

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
    ]
    if "mean_intensity" in df.columns:
        conds.append(df.mean_intensity.between(th.min_mean_intensity, th.max_mean_intensity))
    if th.exclude_border:
        conds.append(~df.border_touch)

    passed = np.logical_and.reduce(conds)
    LOGGER.info("%d / %d nuclei pass all thresholds.", passed.sum(), len(passed))
    return passed

def violin_scatter(
    ax: "matplotlib.axes.Axes",
    data: np.ndarray,
    lo: float,
    hi: float,
    ylabel: str,
    title: str,
) -> None:
    """Draw a split violin with width proportional to absolute point count.

    Overlays raw observations and annotates threshold values together
    with how many observations fall outside them.

    Args:
        ax (matplotlib.axes.Axes): Axes object to draw on.
        data (np.ndarray): 1-D array of metric values.
        lo (float): Lower threshold bound.
        hi (float): Upper threshold bound.
        ylabel (str): Label for the y-axis.
        title (str): Title for the x-tick label.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    # --------------------------------------------------------------------- #
    # Basic statistics.                                                     #
    # --------------------------------------------------------------------- #
    n_total: int = len(data)
    finite_lo: bool = np.isfinite(lo)
    finite_hi: bool = np.isfinite(hi)

    # Boolean mask of excluded observations.
    excl_mask = (
        (data < lo if finite_lo else False) |
        (data > hi if finite_hi else False)
    )
    n_excluded: int = int(excl_mask.sum())

    # --------------------------------------------------------------------- #
    # Violin outline. Width ∝ absolute counts (density_norm="count").              #
    # --------------------------------------------------------------------- #
    sns.violinplot(
        y=data,
        ax=ax,
        inner=None,      # We handle raw points ourselves.
        density_norm="count",   # Area – and therefore width – scales with N.
        cut=0,           # Do not extrapolate beyond the data range.
        fill=False,
        linewidth=1.2,
        color="black",
    )

    # --------------------------------------------------------------------- #
    # Scatter of individual observations.                                   #
    # --------------------------------------------------------------------- #
    rng = np.random.default_rng(0)
    sc = ax.scatter(
        rng.normal(1.0, 0.04, n_total),   # Horizontal jitter.
        data,
        c=data,
        cmap="viridis",
        s=12,
        edgecolors="none",
        alpha=0.85,
    )

    # --------------------------------------------------------------------- #
    # Threshold guide lines on both halves, plus numeric annotations.       #
    # --------------------------------------------------------------------- #
    if finite_lo:
        ax.hlines(lo, -0.25, 0.25,  linestyles="--", colors="black", linewidth=0.9)
        ax.hlines(lo,  0.75, 1.25,  linestyles="--", colors="black", linewidth=0.9)
        ax.text(1.3, lo, f"min = {lo:.3g}", va="center", fontsize="x-small")

    if finite_hi:
        ax.hlines(hi, -0.25, 0.25,  linestyles="--", colors="black", linewidth=0.9)
        ax.hlines(hi,  0.75, 1.25,  linestyles="--", colors="black", linewidth=0.9)
        ax.text(1.3, hi, f"max = {hi:.3g}", va="center", fontsize="x-small")

    # --------------------------------------------------------------------- #
    # Axis cosmetics and meta-information.                                  #
    # --------------------------------------------------------------------- #
    ax.set_xlim(-0.25, 1.35)           # Extra space on the right for labels.
    ax.set_xticks([1.0])
    ax.set_xticklabels([title])
    ax.set_ylabel(ylabel)

    # Annotation of sample size and exclusion count.
    ax.text(
        0.5,
        1.02,
        f"n = {n_total}   excluded = {n_excluded}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize="small",
    )

    # Colour bar keyed to data values.
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=ylabel)


def save_violin_plots(
    df: pd.DataFrame,
    th: Thresholds,
    out_dir: Path,
    prefix: str,
) -> None:
    """Generate violin-scatter plots for every bounded metric.

    Saves each figure as both PDF and PNG in *out_dir*.
    Full absolute file paths are logged.

    Args:
        df (pd.DataFrame): Metrics DataFrame with one row per nucleus.
        th (Thresholds): Threshold configuration for annotation.
        out_dir (Path): Output directory for saved figures.
        prefix (str): Filename prefix for saved figures.
    """

    import matplotlib.pyplot as plt
    import warnings

    plots = [
        ("area",              th.min_pixels,         th.max_pixels,         "Pixel Count",      "Size"),
        ("circularity",       th.min_circularity,    th.max_circularity,    "Circularity",      "Circularity"),
        ("solidity",          th.min_solidity,       th.max_solidity,       "Solidity",         "Solidity"),
        ("eccentricity",      th.min_eccentricity,   th.max_eccentricity,   "Eccentricity",     "Eccentricity"),
        ("aspect_ratio",      th.min_aspect_ratio,   th.max_aspect_ratio,   "Aspect Ratio",     "Aspect Ratio"),
        ("hole_fraction",     th.min_hole_fraction,  th.max_hole_fraction,  "Hole Fraction",    "Hole Fraction"),
        ("straight_fraction", getattr(th, "min_straight_fraction", 0.0),
                              getattr(th, "max_straight_fraction", 1.0),     "Straight Fraction","Straight Fraction"),
        ("mean_intensity",    th.min_mean_intensity, th.max_mean_intensity, "Mean Intensity",   "Mean Intensity"),
    ]

    # Ensure the output directory exists.
    out_dir.mkdir(parents=True, exist_ok=True)

    for col, lo, hi, ylabel, title in plots:
        # Skip metrics that are absent or completely unbounded.
        if col not in df.columns or (np.isneginf(lo) and np.isposinf(hi)):
            continue

        fig, ax = plt.subplots(figsize=(4, 6), constrained_layout=True)

        # Draw the violin–scatter with annotations.
        violin_scatter(ax, df[col].values, lo, hi, ylabel, title)

        # Construct exact file paths.
        pdf_path = (out_dir / f"{prefix}{col}_violin.pdf").resolve()
        png_path = (out_dir / f"{prefix}{col}_violin.png").resolve()

        with warnings.catch_warnings():
            # Suppress the occasional NumPy det() warning triggered by constrained-layout.
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in det",
                module="numpy.linalg",
            )
            # Save as both vector (PDF) and high-resolution raster (PNG).
            fig.savefig(pdf_path, format="pdf")    # Example: /abs/path/out_dir/prefixarea_violin.pdf.
            fig.savefig(png_path, dpi=300)         # Example: /abs/path/out_dir/prefixarea_violin.png.

        plt.close(fig)
        LOGGER.info("Saved violin plot for %s (%s, %s).", col, pdf_path, png_path)


###############################################################################
# Overlay generation.
###############################################################################

def paint_overlay(
    raw: np.ndarray,
    label_map: np.ndarray,
    passed_labels: set[int],
    out_path: Path,
    region: Tuple[float, float, float, float] | None = None,
    alpha: float = 0.35,
) -> None:
    """Colour passed nuclei green and failed nuclei red, honouring an optional crop box.

    Args:
        raw (np.ndarray): Raw microscopy image (grayscale H x W or RGB H x W x 3).
        label_map (np.ndarray): 2-D integer label map.
        passed_labels (set[int]): Set of label IDs that passed filtering.
        out_path (Path): Destination path for the overlay image.
        region (Tuple[float, float, float, float] | None): Optional fractional crop box.
        alpha (float): Blending alpha for the overlay colour.
    """

    # ❶ Crop raw and label_map if a region was given.
    if region is not None and region != (0.0, 1.0, 0.0, 1.0):
        xmin, xmax, ymin, ymax = region
        h, w = label_map.shape
        l, r = int(xmin * w), int(xmax * w)
        t, b = int(ymin * h), int(ymax * h)
        raw       = raw[t:b, l:r]       if raw.ndim == 2 else raw[t:b, l:r, :]
        label_map = label_map[t:b, l:r]

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
        color = green if lab in passed_labels else red
        mask = label_map == lab
        overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * color).astype(np.uint8)

    imsave(str(out_path), overlay)
    LOGGER.info("Overlay saved → %s", out_path.name)

###############################################################################
# I/O helpers.
###############################################################################

def load_label_map(path: Path) -> np.ndarray:
    """Return a 2-D integer label map, loading with mmap when possible.

    Args:
        path (Path): Path to the .npy file containing the label map or binary stack.

    Returns:
        np.ndarray: 2-D integer label map.

    Raises:
        ValueError: If the array shape is unsupported.
    """

    arr = np.load(path, allow_pickle=True, mmap_mode="r")
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
        return arr
    if arr.ndim == 3:
        # Binary stack needs to be packed into a label map first for streaming.
        label = np.zeros(arr.shape[1:], dtype=np.int32)
        for i, sl in enumerate(arr, 1):
            label[sl.astype(bool)] = i
        return label
    raise ValueError("Unsupported mask array shape – expected label map or stack.")

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
    a.add_argument("--region", nargs=4, type=float,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="Restrict overlay to this fractional box (0-1 coordinates).")
    a.add_argument("--no_stack", action="store_true",
                   help="Skip writing the Boolean mask stack.")

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
        region=tuple(args.region) if args.region else None,
        no_stack=args.no_stack,
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
    np.save(cfg.out_dir / f"{cfg.prefix}passed_labels.npy",
                     np.array(passed_labels, dtype=np.int32))
    np.save(cfg.out_dir / f"{cfg.prefix}failed_labels.npy",
                     np.array(failed_labels, dtype=np.int32))

    # Optional one-liner overview that is easy to inspect with `cat`.
    with open(cfg.out_dir / f"{cfg.prefix}counts.json", "w") as fh:
            json.dump({"passed": len(passed_labels),
                       "failed": len(failed_labels)}, fh)
    LOGGER.info("Saved label lists – no Boolean stacks written.")

    if cfg.save_summary_csv:
        metrics_df.assign(passed=passed_mask).to_csv(cfg.out_dir / f"{cfg.prefix}metrics.csv", index=False)
        LOGGER.info("Metrics CSV written.")

    if cfg.save_plots:
        save_violin_plots(metrics_df.assign(passed=passed_mask), cfg.th, cfg.out_dir, cfg.prefix)

    # 4 ▸ Overlay (optional).
    if cfg.overlay:
        assert cfg.raw_image is not None, "--raw-image required for overlay."

        # Load the raw fluorescence or bright-field image.
        if str(cfg.raw_image).lower().endswith(".npy"):
            raw = np.load(cfg.raw_image, mmap_mode="r")
        else:
            from skimage.io import imread
            raw = imread(cfg.raw_image)

        # Informative log.
        LOGGER.info("Saving overlay for region: %s",
                    cfg.region if cfg.region else "full frame")

        # Draw red/green QC overlay, optionally cropped to cfg.region.
        paint_overlay(
            raw=raw,
            label_map=label_map,
            passed_labels=set(passed_labels),
            out_path=cfg.out_dir / f"{cfg.prefix}overlay.tif",
            region=cfg.region,
            alpha=0.35,
        )

    '''Persist boolean mask stack for downstream scripts.'''
    if cfg.no_stack:
        LOGGER.info("Flag --no_stack set – skipping Boolean-stack export.")
        LOGGER.info("✓ Done.")
        return  # ← Early exit; everything else is already written.

    stack_path = cfg.out_dir / f"{cfg.prefix}passed_masks.npy"
    height, width = label_map.shape
    passed_stack = open_memmap(
        stack_path, mode="w+", dtype=np.bool_, shape=(len(passed_labels), height, width)
    )

    for i, lab in enumerate(passed_labels, 1):
        passed_stack[i - 1] = label_map == lab  # Write one slice, then flush.

    del passed_stack  # Ensures the mem-mapped array is written to disk.
    LOGGER.info("Boolean stack of passed nuclei saved → %s", stack_path.name)

    LOGGER.info("✓ Done.")


if __name__ == "__main__":
    sys.exit(main())
