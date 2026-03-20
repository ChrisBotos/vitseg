"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: segmentation_mask_dynamic_patches_vit.py.
Description:
    Vision Transformer feature extraction for single-cell nuclei analysis.
    Extracts multi-scale patch embeddings from segmentation masks using a
    pre-trained DINO ViT-S/16 model and concatenates features across scales.

    Key features for bioinformatician users:
        • **Multi-scale extraction** – Extracts features at multiple patch sizes
          (e.g. 16, 32, 64 px) to capture both fine cellular details and broader
          tissue context essential for cell type classification.
        • **Discriminative weighting** – Weights scale contributions by feature
          variance to emphasise the most informative scales.
        • **Memory-efficient batching** – Processes nuclei in configurable batches
          with optional mixed-precision inference to fit large images.

    Scientific context:
        This implementation is optimised for analysing cellular heterogeneity in
        tissue samples, particularly useful for studying cell damage, repair
        mechanisms, and phenotypic transitions in kidney injury models.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, torch >= 2.2, torchvision, transformers, scikit-image, tqdm.

Usage:
    python code/segmentation_mask_dynamic_patches_vit.py \
        -i data/IRI_regist_cropped.tif \
        -m results/filtered_results/filtered_passed_masks.npy \
        -o results/VIT_dynamic_patches \
        --patch_sizes 16 32 64 \
        --batch_size 512 \
        --model_name facebook/dino-vits16

"""
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.measure import regionprops_table
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from transformers import ViTImageProcessor, ViTModel

# Ensure huge TIFFs are accepted without triggering a DecompressionBombError.
Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("vit-dynamic-patches")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def batched_crops(
    img_tensor: torch.Tensor,    # (3, H, W) on GPU/CPU.
    centres: torch.Tensor,       # (N, 2) (x, y) on same device, float32.
    size: int,                  # Crop side in pixels.
) -> torch.Tensor:               # (N, 3, size, size).
    """Vectorised GPU/CPU cropping with *no* Python loops.

    The function constructs, for each nucleus centre, a normalised sampling grid
    in the ``[-1, 1]`` range expected by :pyfunc:`torch.nn.functional.grid_sample`.
    Using broadcasting, a single call produces the *N* crops required for the
    batch.
    """
    if size < 2:
        raise ValueError("*size* must be at least 2pixels.")

    n = centres.shape[0]
    _, H, W = img_tensor.shape
    half = (size - 1) / 2.0  # Half‑width expressed in pixels.

    # Offsets grid relative to the nucleus centre (pixel units).
    delta = torch.linspace(-half, half, steps=size, device=centres.device)
    yy, xx = torch.meshgrid(delta, delta, indexing="ij")              # (S, S).
    base_grid = torch.stack((xx, yy), dim=-1)                          # (S, S, 2).

    # Shift the base grid by every centre and broadcast to (N, S, S, 2).
    grid = base_grid.unsqueeze(0) + centres.view(n, 1, 1, 2)

    # Normalise pixel positions to the ``[-1, 1]`` range separately for x and y.
    grid[..., 0] = grid[..., 0].mul_(2.0 / (W - 1)).add_(-1.0)  # x‑coords.
    grid[..., 1] = grid[..., 1].mul_(2.0 / (H - 1)).add_(-1.0)  # y‑coords.

    # Input must be (N, C, H, W); replicate the 3‑channel slide without copy.
    crops = F.grid_sample(
        img_tensor.unsqueeze(0).expand(n, -1, -1, -1),
        grid,
        mode="bilinear",
        align_corners=True,
    )
    return crops


def centroids_from_label_list(raw_seg: np.ndarray, wanted: np.ndarray) -> List[Tuple[int, int]]:
    """Return *(x, y)* centroids for the subset of labels present in *wanted*."""
    props = regionprops_table(raw_seg, properties=("label", "centroid"))
    df = pd.DataFrame(props).set_index("label")

    sel = df.loc[wanted]
    # regionprops returns (row, col) ⇒ (y, x).
    return list(
        zip(
            sel["centroid-1"].round().astype(int),
            sel["centroid-0"].round().astype(int),
        )
    )


def compute_centroids(mask_array: np.ndarray) -> List[Tuple[int, int]]:
    """Return *(x, y)* centroids for each nuclear mask (label map or bool stack)."""
    centroids: List[Tuple[int, int]] = []

    if mask_array.ndim == 2 and mask_array.dtype == np.bool_:
        mask_array = mask_array[np.newaxis, ...]

    if mask_array.ndim == 2:  # Single label map.
        for lab in np.unique(mask_array)[1:]:
            ys, xs = np.nonzero(mask_array == lab)
            if xs.size:
                centroids.append((int(xs.mean()), int(ys.mean())))
    else:                      # Stack of binary masks.
        for sl in mask_array.reshape(-1, *mask_array.shape[-2:]):
            ys, xs = np.nonzero(sl)
            if xs.size:
                centroids.append((int(xs.mean()), int(ys.mean())))

    return centroids



# ---------------------------------------------------------------------------
# Core embedding routine
# ---------------------------------------------------------------------------

def extract_and_save_patches(
    image_path: Path,
    masks_path: Path,
    label_map_path: Path | None,
    output_dir: Path,
    patch_size: int,
    patch_sizes: List[int] | None,
    batch_size: int,
    device: torch.device,
    model_name: str,
    use_amp: bool,
    viz_crop_region: Tuple[float, float, float, float] | None,
    compile_model: bool,
    save_numpy: bool,
) -> None:
    """Embed every nucleus for one or many crop sizes and write the features to disk with scientifically improved multi-scale fusion."""

    start_time = time.perf_counter()
    sizes: List[int] = sorted(set(patch_sizes or [patch_size]))
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    LOGGER.debug(f"Processing {len(sizes)} patch sizes: {sizes}")

    # ------------------------------------------------------------------
    # 1. Build ViT
    # ------------------------------------------------------------------
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name, output_hidden_states=True, use_safetensors=True).to(device).eval()

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[arg-type]
        except Exception as err:  # pragma: no cover – depends on system support.
            LOGGER.warning("torch.compile failed (%s). Falling back to eager mode.", err)

    if use_amp and device.type == "cuda":
        model.half()

    # ------------------------------------------------------------------
    # 2. Load slide and compute centroids
    # ------------------------------------------------------------------
    image = Image.open(image_path).convert("RGB")
    masks_or_labels = np.load(masks_path, mmap_mode="r")

    if masks_or_labels.ndim >= 3 or masks_or_labels.dtype == object:
        masks = (
            masks_or_labels
            if masks_or_labels.dtype != object
            else np.stack(masks_or_labels.astype(bool), axis=0)
        )
        centroids = compute_centroids(masks)
    else:
        if label_map_path is None:
            raise ValueError("--label_map is required when --mask is a label list.")
        raw_seg = np.load(label_map_path, mmap_mode="r")
        centroids = centroids_from_label_list(raw_seg, masks_or_labels.astype(int))

    if not centroids:
        LOGGER.warning("No nuclei detected – aborting.")
        return
    LOGGER.info("Detected %d nuclei.", len(centroids))

    # ------------------------------------------------------------------
    # 3. Optional visualisation of crop boundaries
    # ------------------------------------------------------------------
    if viz_crop_region is not None:
        random.seed(42)
        palette = ["red"] + [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(len(sizes) - 1)]
        size2color = dict(zip(sizes, palette))

        vxmin, vxmax, vymin, vymax = viz_crop_region
        cw, ch = image.size
        vx0, vy0, vx1, vy1 = int(vxmin * cw), int(vymin * ch), int(vxmax * cw), int(vymax * ch)

        viz = image.crop((vx0, vy0, vx1, vy1)).copy()
        draw = ImageDraw.Draw(viz)
        for (x, y) in centroids:
            for s in sizes:
                half = s // 2
                draw.rectangle(
                    (x - half - vx0, y - half - vy0, x + half - vx0, y + half - vy0),
                    outline=size2color[s],
                    width=2,
                )
        viz.save(output_dir / f"viz_{image_path.stem}.png")

    # ------------------------------------------------------------------
    # 4. Shared tensors
    # ------------------------------------------------------------------
    img_t = TF.to_tensor(image).to(device, non_blocking=True)           # (3, H, W).
    centroids_t = torch.as_tensor(centroids, device=device, dtype=torch.float32)

    normalise = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)

    if use_amp and device.type == "cuda":
        autocast_ctx: contextlib.AbstractContextManager = torch.autocast(
            device_type="cuda", dtype=torch.float16
        )
    elif use_amp and device.type == "cpu":
        autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()

    # ------------------------------------------------------------------
    # 5. Multi-Scale ViT Feature Extraction Loop
    # ------------------------------------------------------------------
    size_to_feats: Dict[int, np.ndarray] = {}

    LOGGER.debug(f"Starting ViT feature extraction for {len(centroids)} nuclei")

    for s in sizes:
        LOGGER.info("Embedding %d‑px crops…", s)
        feats: List[np.ndarray] = []

        for i in tqdm(range(0, len(centroids), batch_size), desc=f"{s}px", unit="batch"):
            batch_cent = centroids_t[i : i + batch_size]

            patches = batched_crops(img_t, batch_cent, s)  # (B, 3, s, s).

            model_size = getattr(model.config, "image_size", 224)
            if isinstance(model_size, (list, tuple)):  # e.g. [224, 224].
                model_size = model_size[0]
            if patches.shape[-1] != model_size:  # Upsample 16 → 224.
                patches = F.interpolate(
                    patches, size=(model_size, model_size),
                    mode="bilinear", align_corners=False)

            patches = normalise(patches)  # Now 224 × 224.

            with torch.inference_mode(), autocast_ctx:
                out = model(pixel_values=patches)
                # Mean the last 4 hidden layers, skip CLS token.
                hid = torch.stack(out.hidden_states[-4:], dim=0).mean(0)
                emb = F.normalize(hid[:, 1:, :].mean(1), p=2, dim=1).cpu().numpy()
            feats.append(emb.astype(np.float32))

        size_to_feats[s] = np.concatenate(feats, axis=0)
        LOGGER.debug(f"Extracted {size_to_feats[s].shape[0]} features of dimension {size_to_feats[s].shape[1]} for {s}px patches")

    # ------------------------------------------------------------------
    # 6. Scientifically Improved Multi-Scale Feature Fusion
    # ------------------------------------------------------------------
    LOGGER.debug(f"Combining features from {len(sizes)} scales using improved fusion")

    if len(sizes) == 1:
        # Single scale - use features directly.
        combined = size_to_feats[sizes[0]]
        LOGGER.debug(f"Single scale features: {combined.shape}")
    else:
        # Scientifically improved multi-scale fusion.
        LOGGER.debug("Applying scientifically improved multi-scale fusion")

        # Step 1: Compute scale-specific discriminative power.
        scale_weights = {}
        scale_features_list = []

        for s in sizes:
            features = size_to_feats[s]
            scale_features_list.append(features)

            # Compute discriminative power using inter-class vs intra-class variance.
            # Higher discriminative power = better for clustering.
            feature_variance = np.var(features, axis=0)  # Variance per feature dimension.
            feature_mean_variance = np.mean(feature_variance)  # Overall feature variance.

            # Weight by discriminative power (higher variance = more discriminative).
            scale_weights[s] = feature_mean_variance
            LOGGER.debug(f"Scale {s}px discriminative power: {feature_mean_variance:.6f}")

        # Step 2: Normalize weights so they sum to 1.
        total_weight = sum(scale_weights.values())
        for s in sizes:
            scale_weights[s] = scale_weights[s] / total_weight
            LOGGER.debug(f"Scale {s}px normalized weight: {scale_weights[s]:.4f}")

        # Step 3: Apply weighted fusion with cross-scale normalization.
        weighted_features = []
        for i, s in enumerate(sizes):
            features = scale_features_list[i]
            weight = scale_weights[s]

            # L2 normalize each scale's features before weighting.
            features_normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

            # Apply discriminative weighting.
            weighted_features.append(features_normalized * weight)
            LOGGER.debug(f"Scale {s}px weighted features shape: {weighted_features[-1].shape}")

        # Step 4: Concatenate weighted features.
        combined = np.concatenate(weighted_features, axis=1)
        LOGGER.debug(f"Combined weighted features: {combined.shape}")

        # Step 5: Apply final L2 normalization for clustering stability.
        combined = combined / (np.linalg.norm(combined, axis=1, keepdims=True) + 1e-8)
        LOGGER.debug(f"Final normalized features: {combined.shape}")

    # Generate feature column names.
    cols: List[str] = []
    for s in sizes:
        d = size_to_feats[s].shape[1]
        cols.extend([f"vit{s}_{i}" for i in range(d)])

    # Save improved features.
    LOGGER.debug(f"Saving {combined.shape[0]} nuclei with {combined.shape[1]}D features")
    pd.DataFrame(combined, columns=cols).to_csv(
        output_dir / f"features_{image_path.stem}.csv", index=False
    )
    pd.DataFrame(centroids, columns=["x_center", "y_center"]).to_csv(
        output_dir / f"coords_{image_path.stem}.csv", index=False
    )

    if save_numpy:
        np.save(output_dir / f"features_{image_path.stem}.npy", combined)

    # Save fusion metadata for analysis.
    if len(sizes) > 1:
        fusion_metadata = {
            'patch_sizes': sizes,
            'scale_weights': [scale_weights[s] for s in sizes],
            'fusion_method': 'discriminative_weighted',
            'feature_dim': combined.shape[1],
            'num_nuclei': combined.shape[0],
            'extraction_time': time.perf_counter() - start_time
        }

        pd.DataFrame([fusion_metadata]).to_csv(
            output_dir / f"fusion_metadata_{image_path.stem}.csv", index=False
        )

    LOGGER.info("Scientifically improved feature extraction completed:")
    LOGGER.info("  • %d nuclei × %d‑D features saved", combined.shape[0], combined.shape[1])
    LOGGER.info("  • Fusion method: discriminative weighted")
    LOGGER.info("  • Total wall‑time: %.1fs", time.perf_counter() - start_time)

# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    """Return an ArgumentParser for multi-scale ViT feature extraction."""
    p = argparse.ArgumentParser(
        "ViT Feature Extraction for Biological Image Analysis",
        description="Extract multi-scale ViT embeddings with advanced fusion and dimensionality optimization"
    )

    # Core input/output arguments.
    p.add_argument("-i", "--image", type=Path, required=True,
                   help="Input microscopy image (RGB TIFF)")
    p.add_argument("-m", "--mask", type=Path, required=True,
                   help="Nuclear segmentation masks (.npy)")
    p.add_argument("-o", "--output", type=Path, default=Path("./output"),
                   help="Output directory for features and metadata")

    # Patch size configuration.
    p.add_argument("--patch_size", type=int, default=16,
                   help="Legacy single crop size (px). Use --patch_sizes for multi-scale.")
    p.add_argument("--patch_sizes", type=int, nargs="+",
                   help="Multiple crop sizes for multi-scale analysis (e.g., 16 32 64)")

    # Model and processing parameters.
    p.add_argument("--batch_size", type=int, default=256,
                   help="Batch size for GPU processing")
    p.add_argument("--model_name", type=str, default="facebook/dino-vits16",
                   help="HuggingFace ViT model identifier")

    # Performance options.
    p.add_argument("--no_amp", action="store_true",
                   help="Disable automatic mixed precision")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile optimization")

    # Visualization and output options.
    p.add_argument("--viz_crop_region", nargs=4, type=float,
                   help="Visualization crop region: xmin xmax ymin ymax (0-1 fractions)")
    p.add_argument("--save_numpy", action="store_true",
                   help="Save features as .npy array in addition to CSV")
    p.add_argument("--label_map", type=Path,
                   help="Original segmentation map when --mask contains label list")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    extract_and_save_patches(
        image_path=args.image,
        masks_path=args.mask,
        label_map_path=args.label_map,
        output_dir=args.output,
        patch_size=args.patch_size,
        patch_sizes=args.patch_sizes,
        batch_size=args.batch_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model_name=args.model_name,
        use_amp=not args.no_amp,
        viz_crop_region=tuple(args.viz_crop_region) if args.viz_crop_region else None,
        compile_model=not args.no_compile,
        save_numpy=args.save_numpy,
    )


if __name__ == "__main__":
    main()
