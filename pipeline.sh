#!/usr/bin/env bash

# Author: Christos Botos.
# Script Name: pipeline_region_support.sh.
# Description: End‑to‑end nuclei‑clustering pipeline with region support and
#              deterministic file naming.
#              • Uses X‑first VIZ_BOX coordinates (xmin xmax ymin ymax).
#              • Feeds the same ROI to every Python stage so visual and numeric
#                outputs match perfectly.
#              • Works on a binary mask image but preserves its own stem when
#                handing feature files to the clustering stage, avoiding any
#                stem mismatch errors.

set -euo pipefail  # Fail on the first error, unset variable, or failed pipe.

###############################################################################
# 1 ┃ User‑editable parameters.                                                .
###############################################################################
IMAGE="img/IRI_regist_cropped.tif"                 # Original high‑resolution slide.
BINARY_IMAGE="img/binary_mask.tif"         # 8‑bit binary mask (1 = nucleus).
RAW_MASKS="segmentation_masks.npy"   # Full segmentation map.

# ViT patch sizes in pixels (smallest → largest).
PATCH_SIZES=(16 32 64)

K_INIT=20                  # Initial cluster count for k‑means.
AUTO_K="none"              # Auto‑k behaviour ("none", "silhouette", "dbi").
BATCH_SIZE=2048            # GPU batch size for ViT embedding extraction.
CLUST_BATCH_SIZE=10000     # Batch size for clustering stage.
WORKERS=8                  # Python multiprocess workers.
SEED=0                     # Random seed for reproducibility.
DOWNSAMPLE=1               # Down‑sampling factor for final overlays (>1 ⇒ subsample).

# Visualisation ROI in normalised slide coordinates (xmin xmax ymin ymax).
VIZ_BOX=(0.57 0.67 0.46 0.56)

# Morphological thresholds.
MIN_PIXELS=20 ; MAX_PIXELS=900
MIN_CIRC=0.62 ; MAX_CIRC=1.00
MIN_SOL=0.765 ; MAX_SOL=1.00
MIN_ECC=0.00 ; MAX_ECC=0.975
MIN_AR=0.50 ; MAX_AR=3.20
MIN_HOLE=0.00 ; MAX_HOLE=0.001

NO_STACK=true              # Disable stacking of overlays to save RAM.

###############################################################################
# 2 ┃ Derived paths (do NOT edit by hand).                                    .
###############################################################################
BASE_RAW="$(basename "${IMAGE%.*}")"        # → IRI_regist.
BASE_BIN="$(basename "${BINARY_IMAGE%.*}")"  # → binary_mask.

FILTER_DIR="new_test"                       # QA filter outputs.
PATCH_DIR="new_test"              # ViT features and coordinates.
CLUSTER_DIR="new_test"            # Final clustering outputs.

# Feature and coordinate files derived from the *binary* stem.
COORDS_CSV="${PATCH_DIR}/coords_${BASE_BIN}.csv"
FEATS_CSV="${PATCH_DIR}/features_${BASE_BIN}.csv"
FEATS_NPY="${PATCH_DIR}/features_${BASE_BIN}.npy"

###############################################################################
# 3 ┃ Step‑wise pipeline.                                                     .
###############################################################################

########################################
# 3.1 ┃ Filter segmentation masks.                                           .
########################################
printf '\n➤ Filtering segmentation masks …\n'
python filter_masks_memopt.py \
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

#########################################
## 3.2 ┃ Convert mask set to binary TIFF.                                     .
#########################################
#printf '\n➤ Building binary mask image …\n'
#python white_segmentation_masks_on_black_background.py \
#       --mask   "${RAW_MASKS}" \
#       --output "${BINARY_IMAGE}"
#
#########################################
## 3.3 ┃ Extract ViT patch embeddings.                                        .
#########################################
#printf '\n➤ Extracting Vision‑Transformer patch embeddings …\n'
#
## Build repeated "--patch_sizes" arguments.
#PATCH_SIZE_ARGS=()
#for S in "${PATCH_SIZES[@]}"; do PATCH_SIZE_ARGS+=("$S"); done
#
#python segmentation_mask_dynamic_patches_vit.py \
#    --image            "${BINARY_IMAGE}" \
#    --mask             "${FILTER_DIR}/filtered_passed_labels.npy" \
#    --label_map        "${RAW_MASKS}" \
#    --output           "${PATCH_DIR}" \
#    --patch_sizes      "${PATCH_SIZE_ARGS[@]}" \
#    --workers          "${WORKERS}" \
#    --batch_size       "${BATCH_SIZE}" \
#    --model_name       "facebook/dino-vits16" \
#    --viz_crop_region  "${VIZ_BOX[@]}" \
#    --no_compile
#     The default file stem BASE_BIN is preserved so downstream stages.
#     read the correct feature files.
#
#########################################
## 3.4 ┃ Cluster the embeddings.                                              .
#########################################
#printf '\n➤ Clustering patch embeddings …\n'
#python cluster_vit_patches_memopt.py \
#    --image         "${IMAGE}" \
#    --labels        "${FILTER_DIR}/filtered_passed_labels.npy" \
#    --label_map     "${RAW_MASKS}" \
#    --coords        "${COORDS_CSV}" \
#    --features_npy  "${FEATS_NPY}" \
#    --features_csv  "${FEATS_CSV}" \
#    --clusters      "${K_INIT}" \
#    --auto-k        "${AUTO_K}" \
#    --batch-size    "${CLUST_BATCH_SIZE}" \
#    --seed          "${SEED}" \
#    --outdir        "${CLUSTER_DIR}" \
#    --region        0 1 0 1 \
#    --downsample    "${DOWNSAMPLE}"
#
#printf '\n✓ Pipeline completed successfully.\n'
