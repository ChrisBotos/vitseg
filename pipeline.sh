#!/usr/bin/env bash

# Author: Christos Botos.
# Patch Name: pipeline_region_support.sh
# Description:
#     • Uses x-first VIZ_BOX coordinates (xmin xmax ymin ymax).
#     • Feeds the same box to **all** three Python stages.
#     • Optionally crops the ViT embedding stage (`--crop_region`) so only
#       nuclei inside the ROI are embedded, which speeds up big slides.

set -euo pipefail

###############################################################################
# User-editable parameters.
###############################################################################
IMAGE="img/IRI_regist.tif"
BINARY_IMAGE="img/binary_mask.tif"
RAW_MASKS="segmentation_masks_whole.npy"

# ViT crop sizes in pixels.
PATCH_SIZES=(16 32 64)

K_INIT=20
AUTO_K="none"
BATCH_SIZE=512
WORKERS=8

# Visualisation box – note the X-first order.
VIZ_BOX=(0.57 0.67 0.46 0.56)   # xmin xmax ymin ymax

# Morphology thresholds.
MIN_PIXELS=20       ; MAX_PIXELS=1000
MIN_CIRC=0.6       ; MAX_CIRC=1.00
MIN_SOL=0.75       ; MAX_SOL=1.00
MIN_ECC=0.00        ; MAX_ECC=0.98
MIN_AR=0.50         ; MAX_AR=3.20
MIN_HOLE=0.00       ; MAX_HOLE=0.001

# Extra.
NO_STACK=true

###############################################################################
# Derived paths.
###############################################################################
BASE=$(basename "${IMAGE%.*}")
FILTER_DIR="whole_run"
PATCH_DIR="whole_run_20k_binary"
CLUSTER_DIR="whole_run_20k_binary"

################################################################################
## 1. Filter segmentation masks.
################################################################################
#echo "➤ Filtering masks …"
#python filter_masks_memopt.py \
#    --input  "${RAW_MASKS}" \
#    --results-dir "${FILTER_DIR}" \
#    --output-prefix "filtered_" \
#    --min-pixels       "${MIN_PIXELS}"       --max-pixels       "${MAX_PIXELS}" \
#    --min-circularity  "${MIN_CIRC}"         --max-circularity  "${MAX_CIRC}" \
#    --min-solidity     "${MIN_SOL}"          --max-solidity     "${MAX_SOL}" \
#    --min-eccentricity "${MIN_ECC}"          --max-eccentricity "${MAX_ECC}" \
#    --min-aspect-ratio "${MIN_AR}"           --max-aspect-ratio "${MAX_AR}"  \
#    --min-hole-fraction "${MIN_HOLE}"        --max-hole-fraction "${MAX_HOLE}" \
#    --summary-csv \
#    --raw-image "${IMAGE}" \
#    --overlay \
#    --region "${VIZ_BOX[@]}" \
#    ${NO_STACK:+--no_stack}
#

###############################################################################
# 2. Turn the image to binary, 1 for pixels in masks 0 for all others.
###############################################################################
python segmentation_mask_to_binary_vit_input.py \
         --mask "${RAW_MASKS}" \
         --output "${BINARY_IMAGE}"

###############################################################################
# 3. Extract ViT patch embeddings.
###############################################################################
echo "➤ Extracting Vision-Transformer patch embeddings …"
PATCH_SIZE_ARGS=()
for S in "${PATCH_SIZES[@]}"; do PATCH_SIZE_ARGS+=( "$S" ); done

python segmentation_mask_dynamic_patches_vit.py \
    --image "${BINARY_IMAGE}" \
    --mask  "${FILTER_DIR}/filtered_passed_labels.npy" \
    --label_map "${RAW_MASKS}" \
    --output "${PATCH_DIR}" \
    --patch_sizes "${PATCH_SIZE_ARGS[@]}" \
    --workers "${WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    --model_name "facebook/dino-vits16" \
    --viz_crop_region "${VIZ_BOX[@]}" \
    --no_compile


###############################################################################
# 4. Cluster the embeddings and overlay results.
###############################################################################
echo "➤ Clustering patch embeddings …"

python cluster_vit_patches_memopt.py \
    --image     "${IMAGE}" \
    --labels    "${FILTER_DIR}/filtered_passed_labels.npy" \
    --label_map "${RAW_MASKS}" \
    --coords    "${PATCH_DIR}/coords_${BASE}.csv" \
    --features_csv "${PATCH_DIR}/features_${BASE}.csv" \
    --clusters  "${K_INIT}" \
    --auto-k    "${AUTO_K}" \
    --outdir    "${CLUSTER_DIR}" \
    --region    0 1 0 1

echo "✓ Pipeline completed successfully."
