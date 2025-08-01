#!/usr/bin/env bash

# Author: Christos Botos.
# Script Name: pipeline_region_support.sh.
# Description: End‑to‑end nuclei‑clustering pipeline with region support and
#              deterministic file naming. Supports both dynamic patch extraction
#              around individual nuclei and uniform tiling across entire images.
#              • Uses X‑first VIZ_BOX coordinates (xmin xmax ymin ymax).
#              • Feeds the same ROI to every Python stage so visual and numeric
#                outputs match perfectly.
#              • Works on a binary mask image but preserves its own stem when
#                handing feature files to the clustering stage, avoiding any
#                stem mismatch errors.
#              • Dynamic patches: Extracts patches around filtered nuclei masks.
#              • Uniform tiling: Splits entire binary image into regular tiles.

set -euo pipefail  # Fail on the first error, unset variable, or failed pipe.

###############################################################################
# 1 ┃ Pipeline step control flags.                                            .
###############################################################################
# Set to True to run each step, False to skip.
# This allows selective execution of pipeline components for development,
# debugging, or when resuming from intermediate results.
#
# Usage examples:
#   • Run full pipeline: Set all flags to True.
#   • Skip to clustering: Set only RUN_CLUSTERING=True (current default).
#   • Run only mask filtering: Set only RUN_FILTER_MASKS=True.
#   • Resume from ViT step: Set RUN_VIT_EXTRACTION=True and RUN_CLUSTERING=True.
#   • Filter features only: Set RUN_FEATURE_FILTERING=True and RUN_CLUSTERING=True.
#   • Test feature filtering: Set RUN_VIT_EXTRACTION=True, RUN_FEATURE_FILTERING=True.
#
RUN_FILTER_MASKS=False       # Step 3.1: Filter segmentation masks.
RUN_BINARY_CONVERSION=False  # Step 3.2: Convert mask set to binary TIFF.
RUN_VIT_EXTRACTION=False     # Step 3.3: Extract ViT patch embeddings.
RUN_FEATURE_FILTERING=True  # Step 3.4: Filter features by box sizes.
RUN_CLUSTERING=True          # Step 3.5: Cluster the embeddings.

# ViT extraction method selection.
# Set to True for dynamic patches around masks (current behavior).
# Set to False for uniform tiling across the entire binary image.
#
# Dynamic patches (True):
#   • Extracts patches centered on filtered nuclei masks.
#   • Uses segmentation_mask_dynamic_patches_vit.py.
#   • Requires filtered masks and segmentation maps.
#   • Ideal for individual cell morphology analysis.
#
# Uniform tiling (False):
#   • Splits entire binary image into regular grid tiles.
#   • Uses uniform_tiling_vit.py.
#   • Works directly with binary mask images.
#   • Ideal for tissue architecture and spatial pattern analysis.
#
USE_DYNAMIC_PATCHES=True     # True: dynamic patches, False: uniform tiling.

###############################################################################
# 2 ┃ User‑editable parameters.                                                .
###############################################################################
IMAGE="data/IRI_regist_cropped.tif"                # Original high‑resolution slide.
BINARY_IMAGE="data/binary_IRI_regist_cropped.tif"         # 8‑bit binary mask (1 = nucleus).
RAW_MASKS="masks/segmentation_masks.npy"   # Full segmentation map.

# ViT patch sizes in pixels for multi-scale analysis.
# Multiple sizes capture different spatial patterns: fine details → tissue architecture.
# Recommended: 32 (cell details), 64 (local patterns), 128 (tissue regions).
PATCH_SIZES=(16 32 64)

# Feature filtering by box sizes (optional step after ViT extraction).
# When enabled, creates filtered feature files containing only selected scales.
# Useful for focused analysis on specific spatial resolutions.
#
# FILTER_BOX_SIZES: Space-separated list of box sizes to include in filtered output.
# Must be a subset of PATCH_SIZES. Examples:
#   • (16)       - Only fine cellular details
#   • (32 64)    - Local patterns and tissue architecture (skip fine details)
#   • (16 32 64) - All scales (equivalent to no filtering)
#
FILTER_BOX_SIZES=(32)     # Box sizes to include in filtered features.

K_INIT=10                 # Initial cluster count for k‑means.
AUTO_K="none"              # Auto‑k behaviour ("none", "silhouette", "dbi").
BATCH_SIZE=2048            # GPU batch size for ViT embedding extraction.
CLUST_BATCH_SIZE=10000     # Batch size for clustering stage.
WORKERS=4                  # Python multiprocess workers.
SEED=0                     # Random seed for reproducibility.
DOWNSAMPLE=1               # Down‑sampling factor for final overlays (>1 ⇒ subsample).

# Visualisation ROI in normalised slide coordinates (xmin xmax ymin ymax).
VIZ_BOX=(0.57 0.67 0.46 0.56)

# Morphological thresholds.
MIN_PIXELS=20 ; MAX_PIXELS=900
MIN_CIRC=0.56 ; MAX_CIRC=1.00
MIN_SOL=0.765 ; MAX_SOL=1.00
MIN_ECC=0.00 ; MAX_ECC=0.975
MIN_AR=0.50 ; MAX_AR=3.20
MIN_HOLE=0.00 ; MAX_HOLE=0.001

NO_STACK=True              # Disable stacking of overlays to save RAM.

###############################################################################
# 2 ┃ Derived paths.                                    .
###############################################################################
BASE_RAW="$(basename "${IMAGE%.*}")"        # → IRI_regist.
BASE_BIN="$(basename "${BINARY_IMAGE%.*}")"  # → binary_mask.

FILTER_DIR="results/IRI_regist_cropped_10k"                       # QA filter outputs.
PATCH_DIR="results/IRI_regist_cropped_10k"              # ViT features and coordinates.
CLUSTER_DIR="results/IRI_regist_cropped_10k"            # Final clustering outputs.

# Feature and coordinate files derived from the *binary* stem.
COORDS_CSV="${PATCH_DIR}/coords_${BASE_BIN}.csv"
FEATS_CSV="${PATCH_DIR}/features_${BASE_BIN}.csv"
FEATS_NPY="${PATCH_DIR}/features_${BASE_BIN}.npy"

###############################################################################
# 3 ┃ Step‑wise pipeline.                                                     .
###############################################################################

printf '\n========================================\n'
printf 'Vision Transformer Analysis Pipeline\n'
printf '========================================\n'

# Display pipeline configuration.
printf '\nPipeline Steps:\n'
if [[ "${RUN_FILTER_MASKS}" == "True" ]]; then
    printf '  ✓ Step 3.1: Filter segmentation masks\n'
else
    printf '  ⏭ Step 3.1: Filter segmentation masks (SKIPPED)\n'
fi

if [[ "${RUN_BINARY_CONVERSION}" == "True" ]]; then
    printf '  ✓ Step 3.2: Convert mask set to binary TIFF\n'
else
    printf '  ⏭ Step 3.2: Convert mask set to binary TIFF (SKIPPED)\n'
fi

if [[ "${RUN_VIT_EXTRACTION}" == "True" ]]; then
    if [[ "${USE_DYNAMIC_PATCHES}" == "True" ]]; then
        printf '  ✓ Step 3.3: Extract ViT patch embeddings (DYNAMIC PATCHES)\n'
    else
        printf '  ✓ Step 3.3: Extract ViT patch embeddings (UNIFORM TILING)\n'
    fi
else
    printf '  ⏭ Step 3.3: Extract ViT patch embeddings (SKIPPED)\n'
fi

if [[ "${RUN_FEATURE_FILTERING}" == "True" ]]; then
    printf '  ✓ Step 3.4: Filter features by box sizes (%s)\n' "${FILTER_BOX_SIZES[*]}"
else
    printf '  ⏭ Step 3.4: Filter features by box sizes (SKIPPED)\n'
fi

if [[ "${RUN_CLUSTERING}" == "True" ]]; then
    printf '  ✓ Step 3.5: Cluster the embeddings\n'
else
    printf '  ⏭ Step 3.5: Cluster the embeddings (SKIPPED)\n'
fi

printf '\nConfiguration:\n'
printf '  • Patch sizes: %s\n' "${PATCH_SIZES[*]}"
if [[ "${RUN_FEATURE_FILTERING}" == "True" ]]; then
    printf '  • Filtered scales: %s\n' "${FILTER_BOX_SIZES[*]}"
fi
printf '  • Output directory: %s\n' "${PATCH_DIR}"
printf '\n========================================\n'

########################################
# 3.1 ┃ Filter segmentation masks.                                           .
########################################
if [[ "${RUN_FILTER_MASKS}" == "True" ]]; then
    printf '\n➤ Filtering segmentation masks …\n'
    python code/filter_masks_memopt.py \
        --input             "${RAW_MASKS}" \
        --results-dir       "${FILTER_DIR}" \
        --output-prefix     "filtered_" \
        --min-pixels        "${MIN_PIXELS}"       --max-pixels        "${MAX_PIXELS}" \
        --min-circularity   "${MIN_CIRC}"         --max-circularity   "${MAX_CIRC}" \
        --min-solidity      "${MIN_SOL}"          --max-solidity      "${MAX_SOL}" \
        --min-eccentricity  "${MIN_ECC}"         --max-eccentricity  "${MAX_ECC}" \
        --min-aspect-ratio  "${MIN_AR}"           --max-aspect-ratio  "${MAX_AR}"  \
        --min-hole-fraction "${MIN_HOLE}"        --max-hole-fraction "${MAX_HOLE}" \
        --summary-csv \
        --raw-image         "${IMAGE}" \
        --overlay \
        --region            "${VIZ_BOX[@]}" \
        ${NO_STACK:+--no_stack}
else
    printf '\n⏭ Skipping mask filtering step (RUN_FILTER_MASKS=False)\n'
fi

########################################
# 3.2 ┃ Convert mask set to binary TIFF.                                     .
########################################
if [[ "${RUN_BINARY_CONVERSION}" == "True" ]]; then
    printf '\n➤ Building binary mask image …\n'
    python code/white_segmentation_masks_on_black_background.py \
           --mask   "${RAW_MASKS}" \
           --output "${BINARY_IMAGE}"
else
    printf '\n⏭ Skipping binary conversion step (RUN_BINARY_CONVERSION=False)\n'
fi

########################################
# 3.3 ┃ Extract ViT patch embeddings.                                        .
########################################
if [[ "${RUN_VIT_EXTRACTION}" == "True" ]]; then
    if [[ "${USE_DYNAMIC_PATCHES}" == "True" ]]; then
        printf '\n➤ Extracting Vision‑Transformer patch embeddings (dynamic patches) …\n'

        # Check for required filtered masks dependency.
        REQUIRED_MASKS="${FILTER_DIR}/filtered_passed_labels.npy"
        if [[ ! -f "${REQUIRED_MASKS}" ]]; then
            printf '\nERROR: Dynamic patches extraction requires filtered masks.\n'
            printf 'Missing file: %s\n' "${REQUIRED_MASKS}"
            printf '\nTo fix this issue, choose one of:\n'
            printf '  1. Enable mask filtering: Set RUN_FILTER_MASKS=True\n'
            printf '  2. Use uniform tiling: Set USE_DYNAMIC_PATCHES=False\n'
            printf '  3. Provide existing filtered masks in: %s\n' "${FILTER_DIR}/"
            exit 1
        fi

        # Build repeated "--patch_sizes" arguments.
        PATCH_SIZE_ARGS=()
        for S in "${PATCH_SIZES[@]}"; do PATCH_SIZE_ARGS+=("$S"); done

        python code/segmentation_mask_dynamic_patches_vit.py \
            --image            "${BINARY_IMAGE}" \
            --mask             "${REQUIRED_MASKS}" \
            --label_map        "${RAW_MASKS}" \
            --output           "${PATCH_DIR}" \
            --patch_sizes      "${PATCH_SIZE_ARGS[@]}" \
            --workers          "${WORKERS}" \
            --batch_size       "${BATCH_SIZE}" \
            --model_name       "facebook/dino-vits16" \
            --viz_crop_region  "${VIZ_BOX[@]}" \
            --no_compile
    else
        printf '\n➤ Extracting Vision‑Transformer patch embeddings (uniform tiling) …\n'

        # Check for required binary image dependency.
        if [[ ! -f "${BINARY_IMAGE}" ]]; then
            printf '\nERROR: Uniform tiling extraction requires binary image.\n'
            printf 'Missing file: %s\n' "${BINARY_IMAGE}"
            printf '\nTo fix this issue, choose one of:\n'
            printf '  1. Enable binary conversion: Set RUN_BINARY_CONVERSION=True\n'
            printf '  2. Provide existing binary image at: %s\n' "${BINARY_IMAGE}"
            exit 1
        fi

        # Use the first patch size for uniform tiling (can be extended for multi-scale).
        UNIFORM_PATCH_SIZE="${PATCH_SIZES[0]}"

        # Use smaller batch size for uniform tiling to handle large numbers of tiles.
        UNIFORM_BATCH_SIZE=$((BATCH_SIZE / 4))
        if [[ ${UNIFORM_BATCH_SIZE} -lt 256 ]]; then
            UNIFORM_BATCH_SIZE=256
        fi

        python code/uniform_tiling_vit.py \
            --image            "${BINARY_IMAGE}" \
            --output           "${PATCH_DIR}" \
            --patch_sizes      "${PATCH_SIZES[@]}" \
            --stride           "${UNIFORM_PATCH_SIZE}" \
            --batch_size       "${UNIFORM_BATCH_SIZE}" \
            --model_name       "facebook/dino-vits16" \
            --workers          1 \
            --enhance_binary \
            --fusion_method    concatenate
    fi
else
    printf '\n⏭ Skipping ViT extraction step (RUN_VIT_EXTRACTION=False)\n'
fi

########################################
# 3.4 ┃ Filter features by box sizes.                                        .
########################################
if [[ "${RUN_FEATURE_FILTERING}" == "True" ]]; then
    printf '\n➤ Filtering features by selected box sizes …\n'

    # Check for required features file dependency.
    if [[ ! -f "${FEATS_CSV}" ]]; then
        printf '\nERROR: Feature filtering requires ViT features file.\n'
        printf 'Missing file: %s\n' "${FEATS_CSV}"
        printf '\nTo fix this issue:\n'
        printf '  1. Enable ViT extraction: Set RUN_VIT_EXTRACTION=True\n'
        printf '  2. Or provide existing features file at: %s\n' "${FEATS_CSV}"
        exit 1
    fi

    # Validate that FILTER_BOX_SIZES is a subset of PATCH_SIZES.
    for FILTER_SIZE in "${FILTER_BOX_SIZES[@]}"; do
        FOUND=false
        for PATCH_SIZE in "${PATCH_SIZES[@]}"; do
            if [[ "${FILTER_SIZE}" == "${PATCH_SIZE}" ]]; then
                FOUND=true
                break
            fi
        done
        if [[ "${FOUND}" == "false" ]]; then
            printf 'ERROR: Filter box size %s not found in PATCH_SIZES (%s)\n' "${FILTER_SIZE}" "${PATCH_SIZES[*]}"
            exit 1
        fi
    done

    # Create filtered features directory.
    FILTER_SCALES_STR=$(IFS=_; echo "${FILTER_BOX_SIZES[*]}")
    FILTERED_DIR="${PATCH_DIR}_filtered_${FILTER_SCALES_STR}px"

    printf '  • Input features: %s\n' "${FEATS_CSV}"
    printf '  • Selected scales: %s\n' "${FILTER_BOX_SIZES[*]}"
    printf '  • Output directory: %s\n' "${FILTERED_DIR}"

    python code/filter_features_by_box_size.py \
        --input         "${FEATS_CSV}" \
        --output        "${FILTERED_DIR}" \
        --box_sizes     "${FILTER_BOX_SIZES[@]}" \
        --coords        "${COORDS_CSV}"

    # Update file paths to use filtered features for clustering.
    # Note: The filter script uses the full input filename stem, so we need to match that pattern.
    FILTERED_SCALES_STR=$(IFS=_; echo "${FILTER_BOX_SIZES[*]}")
    INPUT_STEM="$(basename "${FEATS_CSV%.*}")"  # Extract stem from features CSV filename
    FILTERED_FEATS_CSV="${FILTERED_DIR}/filtered_features_${FILTERED_SCALES_STR}px_${INPUT_STEM}.csv"
    FILTERED_COORDS_CSV="${FILTERED_DIR}/coords_${INPUT_STEM}.csv"

    # The filter script only creates CSV files, so we need to use a non-existent .npy path
    # to force the clustering script to use the CSV file.
    FILTERED_FEATS_NPY="${FILTERED_DIR}/filtered_features_${FILTERED_SCALES_STR}px_${INPUT_STEM}.npy"

    printf '  • Filtered features: %s\n' "${FILTERED_FEATS_CSV}"
    printf '  • Filtered coordinates: %s\n' "${FILTERED_COORDS_CSV}"
    printf '  • Note: Using CSV features (no .npy file created by filtering)\n'

else
    printf '\n⏭ Skipping feature filtering step (RUN_FEATURE_FILTERING=False)\n'

    # Use original features for clustering.
    FILTERED_FEATS_CSV="${FEATS_CSV}"
    FILTERED_COORDS_CSV="${COORDS_CSV}"
    FILTERED_FEATS_NPY="${FEATS_NPY}"
fi

########################################
# 3.5 ┃ Cluster the embeddings.                                              .
########################################
if [[ "${RUN_CLUSTERING}" == "True" ]]; then
    printf '\n➤ Clustering patch embeddings …\n'

    # Update cluster output directory to reflect filtered features if used.
    if [[ "${RUN_FEATURE_FILTERING}" == "True" ]]; then
        FILTER_SCALES_STR=$(IFS=_; echo "${FILTER_BOX_SIZES[*]}")
        FILTERED_CLUSTER_DIR="${CLUSTER_DIR}_filtered_${FILTER_SCALES_STR}px"
        printf '  • Using filtered features for clustering\n'
        printf '  • Filtered cluster output: %s\n' "${FILTERED_CLUSTER_DIR}"
    else
        FILTERED_CLUSTER_DIR="${CLUSTER_DIR}"
        printf '  • Using original features for clustering\n'
    fi

    if [[ "${USE_DYNAMIC_PATCHES}" == "True" ]]; then
        # Dynamic patches clustering (uses filtered labels and segmentation masks).
        python code/cluster_vit_patches_memopt.py \
            --image         "${IMAGE}" \
            --labels        "${FILTER_DIR}/filtered_passed_labels.npy" \
            --label_map     "${RAW_MASKS}" \
            --coords        "${FILTERED_COORDS_CSV}" \
            --features_npy  "${FILTERED_FEATS_NPY}" \
            --features_csv  "${FILTERED_FEATS_CSV}" \
            --clusters      "${K_INIT}" \
            --auto-k        "${AUTO_K}" \
            --batch-size    "${CLUST_BATCH_SIZE}" \
            --seed          "${SEED}" \
            --outdir        "${FILTERED_CLUSTER_DIR}" \
            --region        0 1 0 1 \
            --downsample    "${DOWNSAMPLE}"
    else
        # Uniform tiling clustering (uses tile coordinates directly).
        python code/cluster_uniform_tiles_memopt.py \
            --image         "${BINARY_IMAGE}" \
            --coords        "${FILTERED_COORDS_CSV}" \
            --features_npy  "${FILTERED_FEATS_NPY}" \
            --features_csv  "${FILTERED_FEATS_CSV}" \
            --clusters      "${K_INIT}" \
            --auto-k        "${AUTO_K}" \
            --batch-size    "${CLUST_BATCH_SIZE}" \
            --seed          "${SEED}" \
            --outdir        "${FILTERED_CLUSTER_DIR}" \
            --region        0 1 0 1 \
            --downsample    "${DOWNSAMPLE}"
    fi
else
    printf '\n⏭ Skipping clustering step (RUN_CLUSTERING=False)\n'
fi

printf '\n✓ Pipeline completed successfully.\n'
