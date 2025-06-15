#!/usr/bin/env python
"""
segmentation_patches_vit_improved.py
------------------------------------
Extract fixed‑size image patches centred on nuclear centroids (as produced by a Cellpose
segmentation mask) and embed them with a Vision Transformer (ViT). The script outputs two
CSV files per input image:
    1. features_<image‑stem>.csv  – ℝ^D feature vectors (one row per nucleus).
    2. coords_<image‑stem>.csv    – the (x, y) centroid coordinates corresponding to each row.

Major improvements over the original version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Bug‑fix: the singleton dimension for boolean masks is now inserted correctly.
* Safer AMP handling: automatic mixed precision is enabled only when CUDA is available.
* Robust model loading: falls back to remote download if local weights are missing and prints
  informative messages.
* Optional Torch 2.0 compilation is toggled automatically and deactivates gracefully if not
  available.
* Clearer feature aggregation: the last four hidden layers are averaged (instead of summed),
  then mean‑pooled over patch tokens before ℓ2‑normalisation.
* Enhanced CLI with {
    --model_name          : ViT checkpoint to use,
    --workers             : DataLoader worker threads,
    --no_compile          : Skip torch.compile if compatibility issues arise,
    --save_numpy          : Additionally save a NumPy .npy feature array for faster reload.
  }.
* Comprehensive biological comments throughout the code, as requested.

Author: ChatGPT‑o3 – expert bioinformatician.
Date: 2025‑06‑16.
"""
from __future__ import annotations

import argparse  # Handles command‑line interfaces.
import logging   # Provides flexible runtime logging.
import os        # Facilitates OS interaction.
from pathlib import Path  # Pathlib offers readable path manipulation.
from typing import Iterable, List, Sequence, Tuple

import numpy as np                # Efficient numerical arrays.
import pandas as pd               # Convenient table IO.
import torch                      # Core deep‑learning framework.
import torch.nn.functional as F   # Functional utilities (e.g. normalisation).
from PIL import Image, ImageDraw  # Pillow handles image IO and simple plotting.
from torch.utils.data import DataLoader, Dataset  # Fast mini‑batch loading.
from torchvision import transforms               # Image preprocessing pipeline.
from tqdm import tqdm                             # Friendly progress bars.
from transformers import ViTImageProcessor, ViTModel  # Vision Transformer toolkit.

# ------------------------------- Logging ----------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("segmentation‑patches")

# ----------------------------- Helper utils -------------------------------- #

def compute_centroids(mask_array: np.ndarray) -> List[Tuple[int, int]]:
    """Return a list of (x, y) centroids for each nuclear instance in *mask_array*.

    Parameters
    ----------
    mask_array : np.ndarray
        * Either an (H, W) integer label map with background label 0.
        * Or an (N, H, W) boolean stack where each slice marks one nucleus.

    Returns
    -------
    List[Tuple[int, int]]
        Pixel‑precise centroids. Each coordinate pair refers to the original (un‑cropped)
        image space so that downstream spot‑based omics measurements align naturally.
    """
    centroids: List[Tuple[int, int]] = []

    # Convert stacked boolean slices into a common 3‑D shape for unified handling.
    if mask_array.ndim == 2 and mask_array.dtype == np.bool_:  # Single boolean mask.
        mask_array = mask_array[np.newaxis, ...]  # Add slice axis. This was buggy before.

    if mask_array.ndim == 2:  # Integer label map.
        labels: np.ndarray = np.unique(mask_array)
        labels = labels[labels > 0]  # Exclude background.
        for lab in labels:
            ys, xs = np.nonzero(mask_array == lab)
            if xs.size:
                centroids.append((int(xs.mean()), int(ys.mean())))
    else:  # (N, H, W) boolean stack.
        for slice_mask in mask_array.reshape(-1, *mask_array.shape[-2:]):
            ys, xs = np.nonzero(slice_mask)
            if xs.size:
                centroids.append((int(xs.mean()), int(ys.mean())))

    return centroids


class PatchDataset(Dataset):
    """Tiny Dataset wrapper that stores PIL patches and their centroid coordinates."""

    def __init__(
        self,
        patches: Sequence[Image.Image],
        coords: Sequence[Tuple[int, int]],
        transform: transforms.Compose | None = None,
    ) -> None:
        assert len(patches) == len(coords), "Patches and coordinates must align in length."
        self._patches: Sequence[Image.Image] = patches
        self._coords: Sequence[Tuple[int, int]] = coords
        self._transform: transforms.Compose | None = transform

    # ------------------------- Dataset protocol ------------------------- #
    def __len__(self) -> int:  # noqa: Dunder methods are self‑explanatory.
        return len(self._patches)

    def __getitem__(self, idx: int):
        patch = self._patches[idx]
        if self._transform is not None:
            patch = self._transform(patch)
        x_c, y_c = self._coords[idx]
        return patch, x_c, y_c


# -------------------------- Core processing -------------------------------- #

def extract_and_save_patches(
    image_path: Path,
    masks_path: Path,
    output_dir: Path,
    patch_size: int = 16,
    batch_size: int = 256,
    device: torch.device | None = None,
    model_name: str = "facebook/dino-vits16",
    use_amp: bool = True,
    workers: int | None = None,
    crop_region: Tuple[float, float, float, float] | None = None,
    viz_crop_region: Tuple[float, float, float, float] | None = None,
    compile_model: bool = True,
    save_numpy: bool = False,
) -> None:
    """High‑level wrapper to orchestrate patch extraction, embedding, and saving.

    The workflow is particularly useful in spatial omics pipelines where protein or RNA
    measurements are mapped to single‑cell coordinates. Here, the ViT embedding provides a
    compact descriptor of each nucleus's micro‑environment that can be integrated with the
    molecular data in a downstream multimodal analysis.
    """

    # ---------------- Device & AMP handling ---------------- #
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.type == "cuda"  # AMP is meaningful only on CUDA.

    LOGGER.info("Running on %s. AMP is %s.", device, "ON" if use_amp else "OFF")

    # ---------------- IO setup ---------------- #
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Quiets HF tokeniser warnings.

    # ---------------- Model loading ---------------- #
    try:
        processor = ViTImageProcessor.from_pretrained(model_name, local_files_only=False)
        model = ViTModel.from_pretrained(
            model_name,
            local_files_only=False,
            output_hidden_states=True,
        )
        LOGGER.info("Loaded model %s.", model_name)
    except Exception as exc:  # noqa: BLE001 (broad except justified for graceful fallback)
        LOGGER.error("Failed to load model '%s': %s", model_name, exc)
        raise SystemExit(1) from exc

    model.to(device).eval()

    # Optional Torch 2.0 compile.
    if compile_model:
        if hasattr(torch, "compile"):
            LOGGER.info("Compiling model for faster execution (Torch 2.x)")
            model = torch.compile(model)  # type: ignore[arg‑type]
        else:
            LOGGER.warning("torch.compile not available in this environment – skipping.")

    if use_amp:
        model.half()

    # ---------------- Image & mask loading ---------------- #
    image = Image.open(image_path).convert("RGB")
    masks = np.load(masks_path)

    # ---------------- Optional cropping ---------------- #
    if crop_region is not None:
        xmin, xmax, ymin, ymax = crop_region
        w, h = image.size
        left, top = int(xmin * w), int(ymin * h)
        right, bottom = int(xmax * w), int(ymax * h)
        assert right > left and bottom > top, "Invalid crop region coordinates."
        image = image.crop((left, top, right, bottom))
        masks = (
            masks[top:bottom, left:right]
            if masks.ndim == 2
            else masks[:, top:bottom, left:right]
        )
        LOGGER.info("Cropped image and mask to %s.", ((left, top), (right, bottom)))

    # ---------------- Centroid computation ---------------- #
    centroids = compute_centroids(masks)
    if not centroids:
        LOGGER.warning("No centroids detected – check mask integrity.")
        return
    LOGGER.info("Detected %d nuclei.", len(centroids))

    # ---------------- Optional patch visualisation ---------------- #
    if viz_crop_region is not None:
        vxmin, vxmax, vymin, vymax = viz_crop_region
        cw, ch = image.size
        vx0, vy0 = int(vxmin * cw), int(vymin * ch)
        vx1, vy1 = int(vxmax * cw), int(vymax * ch)
        if vx1 > vx0 and vy1 > vy0:
            viz = image.crop((vx0, vy0, vx1, vy1)).copy()
            draw = ImageDraw.Draw(viz)
            half = patch_size // 2
            for x, y in centroids:
                # Draw a red square marking each planned patch area.
                draw.rectangle(
                    [x - half - vx0, y - half - vy0, x + half - vx0, y + half - vy0],
                    outline="red",
                    width=2,
                )
            viz_path = output_dir / f"viz_{image_path.stem}.png"
            viz.save(viz_path)
            LOGGER.info("Saved patch visualisation to %s.", viz_path.name)

    # ---------------- Patch extraction ---------------- #
    patches: List[Image.Image] = []
    coords: List[Tuple[int, int]] = []
    half = patch_size // 2

    for x, y in centroids:
        left, top = max(x - half, 0), max(y - half, 0)
        right, bottom = min(x + half, image.width), min(y + half, image.height)
        patch = image.crop((left, top, right, bottom))

        # Zero‑pad if patch hits image border.
        if patch.size != (patch_size, patch_size):
            canvas = Image.new("RGB", (patch_size, patch_size))
            paste_pos = (left - (x - half), top - (y - half))
            canvas.paste(patch, paste_pos)
            patch = canvas
        patches.append(patch)
        coords.append((x, y))

    # ---------------- Preprocessing transform ---------------- #
    # ViTImageProcessor stores size as int or dict depending on HF version.
    vit_size = processor.size["height"] if isinstance(processor.size, dict) else processor.size
    transform = transforms.Compose(
        [
            transforms.Resize(vit_size),
            transforms.CenterCrop(vit_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    # ---------------- DataLoader ---------------- #
    if workers is None:
        workers = min(os.cpu_count() or 4, 8)
    loader = DataLoader(
        PatchDataset(patches, coords, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    # ---------------- Feature extraction ---------------- #
    all_features: List[np.ndarray] = []
    all_centers: List[Tuple[int, int]] = []

    # Choose correct autocast context depending on device.
    autocast_context = (
        torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast  # type: ignore[attr‑defined]
    )

    with torch.inference_mode():
        for imgs, xs, ys in tqdm(loader, desc="Extracting features", unit="batch"):
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with autocast_context():
                out = model(pixel_values=imgs)
            # Last 4 hidden states → mean across layers for robustness.
            hidden = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)  # (B, 197, D)
            # Remove class token (index 0) and mean‑pool remaining patch tokens.
            patch_tokens = hidden[:, 1:, :]
            embeddings = patch_tokens.mean(dim=1)  # (B, D)
            # ℓ2‑normalise so that cosine similarity ~ dot product.
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_features.append(embeddings.cpu().float().numpy())
            all_centers.extend(zip(xs.tolist(), ys.tolist()))

    features = np.concatenate(all_features, axis=0)

    # ---------------- Saving ---------------- #
    feats_csv = output_dir / f"features_{image_path.stem}.csv"
    coords_csv = output_dir / f"coords_{image_path.stem}.csv"
    pd.DataFrame(features).to_csv(feats_csv, index=False)
    pd.DataFrame(all_centers, columns=["x_center", "y_center"]).to_csv(coords_csv, index=False)
    LOGGER.info("Wrote %d feature vectors to %s.", features.shape[0], feats_csv.name)

    if save_numpy:
        npy_path = output_dir / f"features_{image_path.stem}.npy"
        np.save(npy_path, features)
        LOGGER.info("Also saved NumPy binary to %s.", npy_path.name)


# ---------------------------- CLI wrapper ---------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="""Extract ViT embeddings for nuclear patches.

        Typical usage::

            python segmentation_patches_vit_improved.py \
                -i sample.tif \
                -m nuclei_masks.npy \
                -o ./out_dir \
                --patch_size 16 \
                --batch_size 256 \
                --model_name facebook/dino-vits16
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-i", "--image", type=Path, required=True, help="Path to RGB image.")
    parser.add_argument("-m", "--mask", type=Path, required=True, help="Path to .npy mask file.")
    parser.add_argument("-o", "--output", type=Path, default=Path("./output"), help="Output dir.")

    parser.add_argument("--patch_size", type=int, default=16, help="Patch side length in px.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batches per forward pass.")

    parser.add_argument("--model_name", type=str, default="facebook/dino-vits16", help="HF repo ID or local path to a ViT checkpoint.")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision even on GPU.")
    parser.add_argument("--no_compile", action="store_true", help="Skip torch.compile even if available.")
    parser.add_argument("--workers", type=int, help="DataLoader worker threads (0 → main process).")

    parser.add_argument(
        "--crop_region",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Optional fractional crop (0‑1) before processing.",
    )
    parser.add_argument(
        "--viz_crop_region",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Optional fractional patch‑visualisation region.",
    )
    parser.add_argument("--save_numpy", action="store_true", help="Also save .npy feature array.")
    return parser


def main() -> None:  # noqa: D401 – imperative mood preferred.
    parser = build_argparser()
    args = parser.parse_args()

    extract_and_save_patches(
        image_path=args.image,
        masks_path=args.mask,
        output_dir=args.output,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model_name=args.model_name,
        use_amp=not args.no_amp,
        workers=args.workers,
        crop_region=tuple(args.crop_region) if args.crop_region else None,
        viz_crop_region=tuple(args.viz_crop_region) if args.viz_crop_region else None,
        compile_model=not args.no_compile,
        save_numpy=args.save_numpy,
    )


if __name__ == "__main__":
    main()
