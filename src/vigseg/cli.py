"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: cli.py.
Description:
    Command-line interface for the vigseg pipeline. Provides the ``vigseg``
    entry point that orchestrates the full analysis workflow: mask filtering,
    binary conversion, ViT feature extraction, feature filtering, and
    clustering.  Replicates the logic previously contained in pipeline.sh.

Dependencies:
    * Python >= 3.10.
    * All vigseg subpackage dependencies.

Usage:
    vigseg --help
    vigseg --steps filter_masks binary vit_extraction filtering clustering
    vigseg --steps clustering --use-dynamic-patches
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the vigseg CLI."""
    parser = argparse.ArgumentParser(
        prog="vigseg",
        description="vigseg analysis pipeline.",
    )

    # Pipeline step selection.
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[
            "filter_masks",
            "binary",
            "vit_extraction",
            "filtering",
            "clustering",
            "comparison",
        ],
        default=[],
        help="Pipeline steps to run (default: none).",
    )

    # ViT extraction mode.
    parser.add_argument(
        "--use-dynamic-patches",
        action="store_true",
        default=True,
        help="Use dynamic patches around nuclei (default: True).",
    )
    parser.add_argument(
        "--use-uniform-tiling",
        action="store_true",
        default=False,
        help="Use uniform tiling instead of dynamic patches.",
    )

    # Input files.
    parser.add_argument("--image", type=str, default="data/IRI_regist_cropped.tif",
                        help="Original high-resolution slide.")
    parser.add_argument("--binary-image", type=str, default="data/binary_IRI_regist_cropped.tif",
                        help="8-bit binary mask image.")
    parser.add_argument("--raw-masks", type=str, default="masks/segmentation_masks.npy",
                        help="Full segmentation map.")

    # Multi-scale parameters.
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[16, 32, 64],
                        help="Patch sizes in pixels for multi-scale analysis.")
    parser.add_argument("--filter-box-sizes", type=int, nargs="+", default=[16, 32],
                        help="Box sizes to include in filtered features.")

    # Clustering parameters.
    parser.add_argument("--k-init", type=int, default=10,
                        help="Initial cluster count for k-means.")
    parser.add_argument("--auto-k", type=str, default="none",
                        choices=["none", "silhouette", "dbi"],
                        help="Auto-k behaviour.")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="GPU batch size for ViT embedding extraction.")
    parser.add_argument("--clust-batch-size", type=int, default=10000,
                        help="Batch size for clustering stage.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Python multiprocess workers.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Down-sampling factor for final overlays.")

    # Visualisation ROI.
    parser.add_argument("--viz-box", type=float, nargs=4, default=[0.57, 0.67, 0.46, 0.56],
                        help="Visualisation ROI (xmin xmax ymin ymax).")

    # Morphological thresholds.
    parser.add_argument("--min-pixels", type=int, default=20)
    parser.add_argument("--max-pixels", type=int, default=900)
    parser.add_argument("--min-circularity", type=float, default=0.56)
    parser.add_argument("--max-circularity", type=float, default=1.00)
    parser.add_argument("--min-solidity", type=float, default=0.765)
    parser.add_argument("--max-solidity", type=float, default=1.00)
    parser.add_argument("--min-eccentricity", type=float, default=0.00)
    parser.add_argument("--max-eccentricity", type=float, default=0.975)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.50)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.20)
    parser.add_argument("--min-hole-fraction", type=float, default=0.00)
    parser.add_argument("--max-hole-fraction", type=float, default=0.001)

    # Comparison parameters.
    parser.add_argument("--spatial-data", type=str,
                        default="data/metadata_complete.csv",
                        help="Path to spatial transcriptomics metadata CSV.")
    parser.add_argument("--comparison-output", type=str,
                        default="results/improved_comparison",
                        help="Output directory for comparison results.")
    parser.add_argument("--match-radius", type=float, default=40,
                        help="Matching radius for spatial comparison.")

    # Output directories.
    parser.add_argument("--results-dir", type=str,
                        default="results/IRI_regist_cropped_10k",
                        help="Output directory for results.")

    return parser


def _run_module(module: str, args: list[str]) -> None:
    """Run a vigseg module as a subprocess."""
    cmd = [sys.executable, "-m", module] + args
    LOGGER.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    return result


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the vigseg CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.use_uniform_tiling:
        args.use_dynamic_patches = False

    steps = args.steps
    if not steps:
        parser.print_help()
        return

    results_dir = args.results_dir
    binary_stem = Path(args.binary_image).stem

    coords_csv = f"{results_dir}/coords_{binary_stem}.csv"
    feats_csv = f"{results_dir}/features_{binary_stem}.csv"
    feats_npy = f"{results_dir}/features_{binary_stem}.npy"

    print("\n========================================")
    print("Vision Transformer Analysis Pipeline")
    print("========================================\n")

    # Step 1: Filter segmentation masks.
    if "filter_masks" in steps:
        print(">>> Filtering segmentation masks ...")
        _run_module("vigseg.preprocessing.filter_masks", [
            "--input", args.raw_masks,
            "--results-dir", results_dir,
            "--output-prefix", "filtered_",
            "--min-pixels", str(args.min_pixels),
            "--max-pixels", str(args.max_pixels),
            "--min-circularity", str(args.min_circularity),
            "--max-circularity", str(args.max_circularity),
            "--min-solidity", str(args.min_solidity),
            "--max-solidity", str(args.max_solidity),
            "--min-eccentricity", str(args.min_eccentricity),
            "--max-eccentricity", str(args.max_eccentricity),
            "--min-aspect-ratio", str(args.min_aspect_ratio),
            "--max-aspect-ratio", str(args.max_aspect_ratio),
            "--min-hole-fraction", str(args.min_hole_fraction),
            "--max-hole-fraction", str(args.max_hole_fraction),
            "--summary-csv",
            "--raw-image", args.image,
            "--overlay",
            "--region", *[str(v) for v in args.viz_box],
            "--no_stack",
        ])

    # Step 2: Binary conversion.
    if "binary" in steps:
        print(">>> Building binary mask image ...")
        _run_module("vigseg.preprocessing.binary_conversion", [
            "--mask", args.raw_masks,
            "--output", args.binary_image,
        ])

    # Step 3: ViT extraction.
    if "vit_extraction" in steps:
        if args.use_dynamic_patches:
            print(">>> Extracting ViT embeddings (dynamic patches) ...")
            required_masks = f"{results_dir}/filtered_passed_labels.npy"
            _run_module("vigseg.extraction.dynamic_patches_vit", [
                "--image", args.binary_image,
                "--mask", required_masks,
                "--label_map", args.raw_masks,
                "--output", results_dir,
                "--patch_sizes", *[str(s) for s in args.patch_sizes],
                "--batch_size", str(args.batch_size),
                "--model_name", "facebook/dino-vits16",
                "--viz_crop_region", *[str(v) for v in args.viz_box],
                "--save_numpy",
                "--no_compile",
            ])
        else:
            print(">>> Extracting ViT embeddings (uniform tiling) ...")
            uniform_batch = max(256, args.batch_size // 4)
            _run_module("vigseg.extraction.uniform_tiling_vit", [
                "--image", args.binary_image,
                "--output", results_dir,
                "--patch_sizes", *[str(s) for s in args.patch_sizes],
                "--stride", str(args.patch_sizes[0]),
                "--batch_size", str(uniform_batch),
                "--model_name", "facebook/dino-vits16",
                "--workers", "1",
                "--enhance_binary",
                "--fusion_method", "concatenate",
            ])

    # Step 4: Feature filtering.
    if "filtering" in steps:
        print(">>> Filtering features by selected box sizes ...")
        filter_scales_str = "_".join(str(s) for s in args.filter_box_sizes)
        filtered_dir = f"{results_dir}_filtered_{filter_scales_str}px"
        _run_module("vigseg.clustering.filter_features", [
            "--input", feats_csv,
            "--output", filtered_dir,
            "--box_sizes", *[str(s) for s in args.filter_box_sizes],
            "--coords", coords_csv,
        ])
        # Update paths for clustering.
        input_stem = Path(feats_csv).stem
        feats_csv = f"{filtered_dir}/filtered_features_{filter_scales_str}px_{input_stem}.csv"
        coords_csv = f"{filtered_dir}/coords_{input_stem}.csv"
        feats_npy = f"{filtered_dir}/filtered_features_{filter_scales_str}px_{input_stem}.npy"
        results_dir = filtered_dir

    # Step 5: Clustering.
    if "clustering" in steps:
        print(">>> Clustering patch embeddings ...")
        if "filtering" in steps:
            filter_scales_str = "_".join(str(s) for s in args.filter_box_sizes)
            cluster_dir = f"{args.results_dir}_filtered_{filter_scales_str}px"
        else:
            cluster_dir = results_dir

        if args.use_dynamic_patches:
            _run_module("vigseg.clustering.cluster_dynamic_patches", [
                "--image", args.image,
                "--labels", f"{args.results_dir}/filtered_passed_labels.npy",
                "--label_map", args.raw_masks,
                "--coords", coords_csv,
                "--features_npy", feats_npy,
                "--features_csv", feats_csv,
                "--clusters", str(args.k_init),
                "--auto-k", args.auto_k,
                "--batch-size", str(args.clust_batch_size),
                "--seed", str(args.seed),
                "--outdir", cluster_dir,
                "--region", "0", "1", "0", "1",
                "--downsample", str(args.downsample),
            ])
        else:
            _run_module("vigseg.clustering.cluster_uniform_tiles", [
                "--image", args.binary_image,
                "--coords", coords_csv,
                "--features_npy", feats_npy,
                "--features_csv", feats_csv,
                "--clusters", str(args.k_init),
                "--auto-k", args.auto_k,
                "--batch-size", str(args.clust_batch_size),
                "--seed", str(args.seed),
                "--outdir", cluster_dir,
                "--region", "0", "1", "0", "1",
                "--downsample", str(args.downsample),
                "--patch-size", str(args.patch_sizes[0]),
            ])

    # Step 6: Comparison.
    if "comparison" in steps:
        print(">>> Running improved ViT-spatial comparison ...")
        comparison_args = [
            "--features", feats_csv,
            "--coords", coords_csv,
            "--spatial", args.spatial_data,
            "--output", args.comparison_output,
            "--seed", str(args.seed),
            "--radius", str(args.match_radius),
        ]
        _run_module("vigseg.comparison.improved_comparison", comparison_args)

    print("\n>>> Pipeline completed successfully.")


if __name__ == "__main__":
    main()
