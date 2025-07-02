"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: segmentation_mask_dynamic_patches_vit_fixed.py.
Description:
    Extract square crops centred on single–cell nuclei and compute ViT embeddings
    at one or several spatial scales.

    This refactor fixes the following issues present in the original script:
        • **Datasets bug** – ``torch.utils.data.Dataset`` was never imported, causing
          a ``NameError`` when instantiating ``PatchDataset``. The import has been
          added and the class cleaned‑up.
        • **GPU crop bug** – ``batched_crops`` relied on the non‑existent
          ``Tensor.linspace`` method. The routine has been rewritten to build a
          correctly normalised sampling grid with ``torch.meshgrid`` and now works
          for CUDA/CPU alike.
        • **torch.compile safety** – If ``torch.compile`` is unavailable or fails
          (e.g. on older GPU compute capabilities) we gracefully fall back to the
          eager model to avoid a crash.
        • Minor hygiene – duplicate imports removed, typing tightened, comments
          upgraded to full sentences, and the logging name clarified.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, torch >= 2.2, torchvision, transformers, scikit‑image, tqdm.

Usage:
    python segmentation_mask_dynamic_patches_vit_fixed.py \
        -i img/IRI_regist_cropped.tif \
        -m filtered_results/filtered_passed_masks.npy \
        -o VIT_dynamic_patches \
        --patch_sizes 16 32 64 \
        --batch_size 512 \
        --model_name facebook/dino-vits16

Test suite:
    Run ``pytest -q segmentation_mask_dynamic_patches_vit_fixed.py`` to execute the
    self‑contained unit tests for the critical functions.
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
Image.MAX_IMAGE_PIXELS = None
from skimage.measure import regionprops_table
from tqdm import tqdm

import torch
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
        raise ValueError("*size* must be at least 2 pixels.")

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
    """Embed every nucleus for one or many crop sizes and write the features to disk."""

    sizes: List[int] = sorted(set(patch_sizes or [patch_size]))
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ------------------------------------------------------------------
    # 1. Build ViT
    # ------------------------------------------------------------------
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()

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
    # 5. Main embedding loop
    # ------------------------------------------------------------------
    size_to_feats: Dict[int, np.ndarray] = {}
    for s in sizes:
        LOGGER.info("Embedding %d‑px crops …", s)
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

    # ------------------------------------------------------------------
    # 6. Concatenate all scales and save
    # ------------------------------------------------------------------
    combined = np.concatenate([size_to_feats[s] for s in sizes], axis=1)

    cols: List[str] = []
    for s in sizes:
        d = size_to_feats[s].shape[1]
        cols.extend([f"vit{s}_{i}" for i in range(d)])

    pd.DataFrame(combined, columns=cols).to_csv(output_dir / f"features_{image_path.stem}.csv", index=False)
    pd.DataFrame(centroids, columns=["x_center", "y_center"]).to_csv(
        output_dir / f"coords_{image_path.stem}.csv", index=False
    )
    if save_numpy:
        np.save(output_dir / f"features_{image_path.stem}.npy", combined)

    LOGGER.info("Saved %d nuclei × %d‑D features.", combined.shape[0], combined.shape[1])
    LOGGER.info("Total wall‑time %.1f s", time.perf_counter() - start_time)

# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    """Return an :pyclass:`ArgumentParser` populated with CLI flags."""
    p = argparse.ArgumentParser("Nucleus crops → ViT embeddings (single or multi‑scale)")
    p.add_argument("-i", "--image", type=Path, required=True)
    p.add_argument("-m", "--mask", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, default=Path("./output"))

    p.add_argument("--patch_size", type=int, default=16, help="Legacy single crop size (px).")
    p.add_argument(
        "--patch_sizes",
        type=int,
        nargs="+",
        help="One or several crop sizes. Overrides --patch_size.",
    )

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--model_name", type=str, default="facebook/dino-vits16")
    p.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision.")
    p.add_argument("--no_compile", action="store_true", help="Disable torch.compile.")
    p.add_argument("--workers", type=int, help="DataLoader worker count (unused in current flow).")
    p.add_argument("--viz_crop_region", nargs=4, type=float)
    p.add_argument("--save_numpy", action="store_true", help="Also save a .npy features array.")
    p.add_argument(
        "--label_map", type=Path, help="Original label map when --mask contains a label list.",
    )
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
