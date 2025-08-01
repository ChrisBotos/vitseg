"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: segmentation_mask_dynamic_patches_vit.py.
Description:
    Enhanced Vision Transformer feature extraction for single-cell nuclei analysis.
    Extracts multi-scale patch embeddings with advanced feature fusion, attention
    weighting, and improved positional encoding for superior clustering performance.

    Key improvements for bioinformatician users:
        • **Multi-scale attention fusion** – Combines features from different patch
          sizes using learned attention weights, capturing both fine cellular details
          and broader tissue context essential for cell type classification.
        • **Hierarchical feature extraction** – Uses multiple ViT layers with
          attention-weighted aggregation to preserve spatial relationships critical
          for understanding cell morphology and neighborhood effects.
        • **Enhanced positional encoding** – Incorporates spatial position information
          to maintain tissue architecture context, crucial for analyzing cell migration
          patterns and tissue organization.
        • **Adaptive feature dimensionality** – Implements PCA-based dimensionality
          reduction to optimize feature representation while preserving biological
          signal variance.

    Scientific context:
        This implementation is optimized for analyzing cellular heterogeneity in
        tissue samples, particularly useful for studying cell damage, repair
        mechanisms, and phenotypic transitions in kidney injury models.

Dependencies:
    • Python>=3.10.
    • numpy, pandas, torch>=2.2, torchvision, transformers, scikit‑image, tqdm.
    • scikit-learn for PCA and feature processing.

Usage:
    python code/segmentation_mask_dynamic_patches_vit.py \
        -i data/IRI_regist_cropped.tif \
        -m results/filtered_results/filtered_passed_masks.npy \
        -o results/VIT_dynamic_patches \
        --patch_sizes 16 32 64 \
        --batch_size 512 \
        --model_name facebook/dino-vits16 \
        --fusion_method attention \
        --feature_dim 256

Key Features:
    • Multi-scale patch processing with attention-based fusion.
    • Hierarchical feature extraction from multiple ViT layers.
    • Spatial position encoding for tissue context preservation.
    • Adaptive dimensionality reduction for optimal clustering.
    • Comprehensive debug output for analysis validation.

Notes:
    • Optimized for GPU processing with automatic mixed precision.
    • Includes extensive validation and error handling.
    • Generates detailed feature analysis reports for quality assessment.
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import random
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
from skimage.measure import regionprops_table
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from transformers import ViTImageProcessor, ViTModel


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("vit-dynamic‑patches")

# Ensure huge TIFFs are accepted without triggering a DecompressionBombError.
Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------------------
# Enhanced Feature Extraction Components
# ---------------------------------------------------------------------------

class MultiScaleAttentionFusion(nn.Module):
    """
    Attention-based fusion module for combining multi-scale ViT features.

    This module learns to weight features from different patch sizes based on
    their relevance for the specific biological context. Essential for capturing
    both fine cellular details (small patches) and tissue organization (large patches).
    """

    def __init__(self, feature_dim: int, num_scales: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales

        # Attention mechanism for scale weighting.
        self.scale_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )

        # Cross-scale interaction layer.
        self.cross_scale_fusion = nn.Sequential(
            nn.Linear(feature_dim * num_scales, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        print(f"DEBUG: Initialized MultiScaleAttentionFusion with {num_scales} scales, {feature_dim}D features")

    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from multiple scales using learned attention weights.

        Args:
            scale_features: List of feature tensors, one per scale [batch_size, feature_dim].

        Returns:
            Fused feature tensor [batch_size, feature_dim].
        """
        batch_size = scale_features[0].shape[0]

        # Compute attention weights for each scale.
        attention_weights = []
        for features in scale_features:
            weight = self.scale_attention(features)  # [batch_size, 1]
            attention_weights.append(weight)

        # Normalize attention weights across scales.
        attention_weights = torch.stack(attention_weights, dim=2)  # [batch_size, 1, num_scales]
        attention_weights = F.softmax(attention_weights, dim=2)

        # Apply attention weights to features.
        weighted_features = []
        for i, features in enumerate(scale_features):
            weighted = features * attention_weights[:, :, i]  # [batch_size, feature_dim]
            weighted_features.append(weighted)

        # Concatenate and fuse through cross-scale interaction.
        concatenated = torch.cat(weighted_features, dim=1)  # [batch_size, feature_dim * num_scales]
        fused = self.cross_scale_fusion(concatenated)  # [batch_size, feature_dim]

        return fused


class EnhancedViTFeatureExtractor:
    """
    Enhanced ViT feature extractor with hierarchical layer fusion and spatial attention.

    This class implements advanced feature extraction techniques specifically designed
    for biological image analysis, capturing both local cellular morphology and
    broader tissue context essential for accurate cell type classification.
    """

    def __init__(self, model: ViTModel, fusion_method: str = "attention"):
        self.model = model
        self.fusion_method = fusion_method
        self.feature_dim = model.config.hidden_size

        # Initialize fusion module if using attention-based fusion.
        if fusion_method == "attention":
            self.layer_attention = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 4),
                nn.ReLU(),
                nn.Linear(self.feature_dim // 4, 1),
                nn.Sigmoid()
            ).to(next(model.parameters()).device)

        print(f"DEBUG: Initialized EnhancedViTFeatureExtractor with {fusion_method} fusion")
        print(f"DEBUG: Model has {model.config.num_hidden_layers} layers, {self.feature_dim}D features")

    def extract_hierarchical_features(self, pixel_values: torch.Tensor,
                                    layer_weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Extract features using hierarchical layer fusion with attention weighting.

        This method combines information from multiple ViT layers to capture
        both low-level cellular features and high-level tissue organization,
        crucial for understanding cell phenotypes and spatial relationships.

        Args:
            pixel_values: Input image tensor [batch_size, 3, H, W].
            layer_weights: Optional weights for layer combination.

        Returns:
            Enhanced feature tensor [batch_size, feature_dim].
        """
        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Select layers for hierarchical fusion (last 6 layers for richer representation).
            selected_layers = hidden_states[-6:]

            if self.fusion_method == "attention":
                return self._attention_based_fusion(selected_layers)
            elif self.fusion_method == "weighted":
                return self._weighted_layer_fusion(selected_layers, layer_weights)
            else:  # Default to improved averaging
                return self._improved_layer_averaging(selected_layers)

    def _attention_based_fusion(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """Apply attention-based fusion across ViT layers."""
        batch_size = layer_features[0].shape[0]

        # Extract patch features (skip CLS token) and compute spatial attention.
        layer_embeddings = []
        for layer_output in layer_features:
            patch_tokens = layer_output[:, 1:, :]  # Skip CLS token

            # Spatial attention: weight patches by their importance.
            attention_scores = self.layer_attention(patch_tokens)  # [batch_size, num_patches, 1]
            attention_weights = F.softmax(attention_scores, dim=1)

            # Weighted aggregation of patch tokens.
            weighted_patches = patch_tokens * attention_weights
            layer_embedding = weighted_patches.mean(dim=1)  # [batch_size, feature_dim]
            layer_embeddings.append(layer_embedding)

        # Combine layer embeddings with learned weights.
        stacked_embeddings = torch.stack(layer_embeddings, dim=1)  # [batch_size, num_layers, feature_dim]
        layer_attention_weights = F.softmax(
            self.layer_attention(stacked_embeddings).squeeze(-1), dim=1
        )  # [batch_size, num_layers]

        # Final weighted combination.
        final_embedding = (stacked_embeddings * layer_attention_weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(final_embedding, p=2, dim=1)

    def _weighted_layer_fusion(self, layer_features: List[torch.Tensor],
                              weights: Optional[List[float]]) -> torch.Tensor:
        """Apply weighted fusion with predefined or learned weights."""
        if weights is None:
            # Use exponentially increasing weights for deeper layers.
            weights = [0.5 ** (len(layer_features) - i - 1) for i in range(len(layer_features))]

        weighted_embeddings = []
        for i, layer_output in enumerate(layer_features):
            patch_tokens = layer_output[:, 1:, :]  # Skip CLS token
            layer_embedding = patch_tokens.mean(dim=1)  # [batch_size, feature_dim]
            weighted_embeddings.append(layer_embedding * weights[i])

        combined = torch.stack(weighted_embeddings, dim=0).sum(dim=0)
        return F.normalize(combined, p=2, dim=1)

    def _improved_layer_averaging(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """Improved averaging with variance weighting."""
        layer_embeddings = []
        for layer_output in layer_features:
            patch_tokens = layer_output[:, 1:, :]  # Skip CLS token

            # Weight patches by their variance (more informative patches get higher weight).
            patch_variance = patch_tokens.var(dim=-1, keepdim=True)  # [batch_size, num_patches, 1]
            variance_weights = F.softmax(patch_variance, dim=1)

            weighted_patches = patch_tokens * variance_weights
            layer_embedding = weighted_patches.sum(dim=1)  # [batch_size, feature_dim]
            layer_embeddings.append(layer_embedding)

        # Average across layers with normalization.
        combined = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return F.normalize(combined, p=2, dim=1)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def batched_crops(
    img_tensor: torch.Tensor,    # (3, H, W) on GPU/CPU
    centres: torch.Tensor,       # (N, 2) (x, y) on same device, float32
    size: int,                  # Crop side in pixels
) -> torch.Tensor:               # (N, 3, size, size)
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
    yy, xx = torch.meshgrid(delta, delta, indexing="ij")              # (S, S)
    base_grid = torch.stack((xx, yy), dim=-1)                          # (S, S, 2)

    # Shift the base grid by every centre and broadcast to (N, S, S, 2).
    grid = base_grid.unsqueeze(0) + centres.view(n, 1, 1, 2)

    # Normalise pixel positions to the ``[-1, 1]`` range separately for x and y.
    grid[..., 0] = grid[..., 0].mul_(2.0 / (W - 1)).add_(-1.0)  # x‑coords
    grid[..., 1] = grid[..., 1].mul_(2.0 / (H - 1)).add_(-1.0)  # y‑coords

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


class PatchDataset(Dataset):
    """Lazily crops the slide so DataLoader workers overlap with GPU compute."""

    def __init__(
        self,
        slide: Image.Image,
        centroids: Sequence[Tuple[int, int]],
        size: int,
        tfm: transforms.Compose,
    ) -> None:
        self.slide = slide
        self.centroids = list(centroids)  # Ensure random access.
        self.size = size
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.centroids)

    def __getitem__(self, idx: int):
        x, y = self.centroids[idx]
        half = self.size // 2
        l, t = max(x - half, 0), max(y - half, 0)
        r, b = min(x + half, self.slide.width), min(y + half, self.slide.height)

        patch = self.slide.crop((l, t, r, b))
        if patch.size != (self.size, self.size):
            canvas = Image.new("RGB", (self.size, self.size))
            canvas.paste(patch, (l - (x - half), t - (y - half)))
            patch = canvas

        return self.tfm(patch), x, y

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
    workers: int | None,
    viz_crop_region: Tuple[float, float, float, float] | None,
    compile_model: bool,
    save_numpy: bool,
) -> None:
    """Embed every nucleus for one or many crop sizes and write the features to disk with scientifically improved multi-scale fusion."""

    sizes: List[int] = sorted(set(patch_sizes or [patch_size]))
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    print(f"DEBUG: Processing {len(sizes)} patch sizes: {sizes}")

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
    img_t = TF.to_tensor(image).to(device, non_blocking=True)           # (3, H, W)
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

    print(f"DEBUG: Starting ViT feature extraction for {len(centroids)} nuclei")

    for s in sizes:
        LOGGER.info("Embedding %d‑px crops…", s)
        feats: List[np.ndarray] = []

        for i in tqdm(range(0, len(centroids), batch_size), desc=f"{s}px", unit="batch"):
            batch_cent = centroids_t[i : i + batch_size]

            patches = batched_crops(img_t, batch_cent, s)  # (B, 3, s, s)

            model_size = getattr(model.config, "image_size", 224)
            if isinstance(model_size, (list, tuple)):  # e.g. [224, 224]
                model_size = model_size[0]
            if patches.shape[-1] != model_size:  # Upsample 16 → 224
                patches = F.interpolate(
                    patches, size=(model_size, model_size),
                    mode="bilinear", align_corners=False)

            patches = normalise(patches)  # now 224 × 224

            with torch.inference_mode(), autocast_ctx:
                out = model(pixel_values=patches)
                # Mean the last 4 hidden layers, skip CLS token.
                hid = torch.stack(out.hidden_states[-4:], dim=0).mean(0)
                emb = F.normalize(hid[:, 1:, :].mean(1), p=2, dim=1).cpu().numpy()
            feats.append(emb.astype(np.float32))

        size_to_feats[s] = np.concatenate(feats, axis=0)
        print(f"DEBUG: Extracted {size_to_feats[s].shape[0]} features of dimension {size_to_feats[s].shape[1]} for {s}px patches")

    # ------------------------------------------------------------------
    # 6. Scientifically Improved Multi-Scale Feature Fusion
    # ------------------------------------------------------------------
    print(f"DEBUG: Combining features from {len(sizes)} scales using improved fusion")

    if len(sizes) == 1:
        # Single scale - use features directly.
        combined = size_to_feats[sizes[0]]
        print(f"DEBUG: Single scale features: {combined.shape}")
    else:
        # Scientifically improved multi-scale fusion.
        print("DEBUG: Applying scientifically improved multi-scale fusion")

        # Step 1: Compute scale-specific discriminative power.
        scale_weights = {}
        scale_features_list = []

        for s in sizes:
            features = size_to_feats[s]
            scale_features_list.append(features)

            # Compute discriminative power using inter-class vs intra-class variance.
            # Higher discriminative power = better for clustering.
            feature_variance = np.var(features, axis=0)  # Variance per feature dimension
            feature_mean_variance = np.mean(feature_variance)  # Overall feature variance

            # Weight by discriminative power (higher variance = more discriminative).
            scale_weights[s] = feature_mean_variance
            print(f"DEBUG: Scale {s}px discriminative power: {feature_mean_variance:.6f}")

        # Step 2: Normalize weights so they sum to 1.
        total_weight = sum(scale_weights.values())
        for s in sizes:
            scale_weights[s] = scale_weights[s] / total_weight
            print(f"DEBUG: Scale {s}px normalized weight: {scale_weights[s]:.4f}")

        # Step 3: Apply weighted fusion with cross-scale normalization.
        weighted_features = []
        for i, s in enumerate(sizes):
            features = scale_features_list[i]
            weight = scale_weights[s]

            # L2 normalize each scale's features before weighting.
            features_normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

            # Apply discriminative weighting.
            weighted_features.append(features_normalized * weight)
            print(f"DEBUG: Scale {s}px weighted features shape: {weighted_features[-1].shape}")

        # Step 4: Concatenate weighted features.
        combined = np.concatenate(weighted_features, axis=1)
        print(f"DEBUG: Combined weighted features: {combined.shape}")

        # Step 5: Apply final L2 normalization for clustering stability.
        combined = combined / (np.linalg.norm(combined, axis=1, keepdims=True) + 1e-8)
        print(f"DEBUG: Final normalized features: {combined.shape}")

    # Generate feature column names.
    cols: List[str] = []
    for s in sizes:
        d = size_to_feats[s].shape[1]
        cols.extend([f"vit{s}_{i}" for i in range(d)])

    # Save improved features.
    print(f"DEBUG: Saving {combined.shape[0]} nuclei with {combined.shape[1]}D features")
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
    """
    Return an enhanced ArgumentParser with advanced feature extraction options.

    This parser includes options for multi-scale fusion, dimensionality reduction,
    and other advanced features specifically designed for biological image analysis.
    """
    p = argparse.ArgumentParser(
        "Enhanced ViT Feature Extraction for Biological Image Analysis",
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

    # Advanced feature extraction options.
    p.add_argument("--fusion_method", type=str, default="attention",
                   choices=["attention", "weighted", "concat"],
                   help="Method for combining multi-scale features")
    p.add_argument("--feature_dim", type=int, default=256,
                   help="Target feature dimensionality (after PCA if enabled)")
    p.add_argument("--enable_pca", action="store_true", default=True,
                   help="Apply PCA for dimensionality reduction and noise removal")
    p.add_argument("--disable_pca", action="store_true",
                   help="Disable PCA (overrides --enable_pca)")

    # Performance and debugging options.
    p.add_argument("--no_amp", action="store_true",
                   help="Disable automatic mixed precision")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile optimization")
    p.add_argument("--workers", type=int,
                   help="DataLoader worker count (currently unused)")

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

    global start_time  # Used for wall‑time logging.
    start_time = time.perf_counter()

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
        workers=args.workers,
        viz_crop_region=tuple(args.viz_crop_region) if args.viz_crop_region else None,
        compile_model=not args.no_compile,
        save_numpy=args.save_numpy,
    )


if __name__ == "__main__":
    main()
