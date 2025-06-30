#!/usr/bin/env python
"""
segmentation_mask_dynamic_patches_vit.py
=======================================
Single‑cell ViT embeddings with **optional multi‑scale context**.

*Back‑compatible*: if you give just `--patch_size 16` the script behaves exactly
like the original version.  If you want extra context windows, simply add one
or more values to `--patch_sizes`, e.g. `--patch_sizes 16 32 64`.

Key points
~~~~~~~~~~
* **Both flags accepted** – `--patch_size` (legacy) and `--patch_sizes` (new).
  When both are present, `--patch_sizes` wins.
* **No other behaviour removed** – AMP, torch.compile, visualisation, CSV/NPY
  output, workers, etc. work as before.
* **Column names** annotate which scale each dimension came from, e.g.
  `vit16_0, … vit32_0, … vit64_0`.
* The rest of the script remains unchanged so you can keep your existing
  pipelines and logs.


Example:
python new.py     -i img/IRI_regist_cropped.tif     -m filtered_results/filtered_passed_masks.npy     -o VIT_dynamic_patches     --patch_sizes 16 32 64     --batch_size 512     --model_name facebook/dino-vits16     --viz_crop_region 0 1 0 1


Author: Christos Botos – Mahfouz Lab
Date: 15-06-2025.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel

# ------------------------------- Logging ---------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("nucleus‑patches")

# ----------------------------- Helper utils ------------------------------- #

def compute_centroids(mask_array: np.ndarray) -> List[Tuple[int, int]]:
    """Return (x, y) centroids for each nuclear mask (label map or bool stack)."""
    c: List[Tuple[int, int]] = []
    if mask_array.ndim == 2 and mask_array.dtype == np.bool_:
        mask_array = mask_array[np.newaxis, ...]
    if mask_array.ndim == 2:
        for lab in np.unique(mask_array)[1:]:
            ys, xs = np.nonzero(mask_array == lab)
            if xs.size:
                c.append((int(xs.mean()), int(ys.mean())))
    else:
        for sl in mask_array.reshape(-1, *mask_array.shape[-2:]):
            ys, xs = np.nonzero(sl)
            if xs.size:
                c.append((int(xs.mean()), int(ys.mean())))
    return c


class PatchDataset(Dataset):
    """Dataset that resizes a list of PIL patches and returns tensors + coords."""

    def __init__(self, patches: Sequence[Image.Image], coords: Sequence[Tuple[int, int]], tfm):
        self.patches, self.coords, self.tfm = patches, coords, tfm

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.tfm(self.patches[idx]), *self.coords[idx]


# ---------------------------- Core function ------------------------------- #

def extract_and_save_patches(
    image_path: Path,
    masks_path: Path,
    output_dir: Path,
    patch_size: int,
    patch_sizes: List[int] | None,
    batch_size: int,
    device: torch.device,
    model_name: str,
    use_amp: bool,
    workers: int | None,
    crop_region: Tuple[float, float, float, float] | None,
    viz_crop_region: Tuple[float, float, float, float] | None,
    compile_model: bool,
    save_numpy: bool,
):
    """Embeds every nucleus for one or many crop sizes and saves concatenated feats."""

    sizes = patch_sizes or [patch_size]
    sizes = sorted(set(sizes))

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[arg‑type]
    if use_amp and device.type == "cuda":
        model.half()

    image = Image.open(image_path).convert("RGB")
    masks = np.load(masks_path, allow_pickle=True)
    if masks.dtype == object:
        masks = np.stack(masks.astype(bool), axis=0)

    if crop_region is not None:
        xmin, xmax, ymin, ymax = crop_region
        w, h = image.size
        l, t, r, b = int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)
        image = image.crop((l, t, r, b))
        masks = masks[:, t:b, l:r] if masks.ndim == 3 else masks[t:b, l:r]

    centroids = compute_centroids(masks)
    if not centroids:
        LOGGER.warning("No nuclei detected – aborting.")
        return
    LOGGER.info("Detected %d nuclei.", len(centroids))

    # ---------------- Optional patch visualisation ---------------- #
    if viz_crop_region is not None:
        # deterministic but distinct colors by crop size
        random.seed(42)
        palette = ["red"] + [f"#{random.randint(0, 0xFFFFFF):06x}"
                             for _ in range(len(sizes) - 1)]
        size2color = dict(zip(sizes, palette))

        vxmin, vxmax, vymin, vymax = viz_crop_region
        cw, ch = image.size
        vx0, vy0 = int(vxmin * cw), int(vymin * ch)
        vx1, vy1 = int(vxmax * cw), int(vymax * ch)

        viz = image.crop((vx0, vx0, vx1, vy1)).copy()
        draw = ImageDraw.Draw(viz)

        for x, y in centroids:         # loop over nuclei
            for s in sizes:            # loop over crop sizes
                half = s // 2
                color = size2color[s]
                # --- rectangle must receive a tuple of four ints ---
                draw.rectangle(
                    (x - half - vx0, y - half - vy0,
                     x + half - vx0, y + half - vy0),
                    outline=color,
                    width=2,
                )

        viz.save(output_dir / f"viz_{image_path.stem}.png")


    # ---------------- TF + loader -------------------------------------- # -------------------------------------- #
    vit_size = processor.size["height"] if isinstance(processor.size, dict) else processor.size
    tfm = transforms.Compose([
        transforms.Resize(vit_size), transforms.CenterCrop(vit_size),
        transforms.ToTensor(), transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    size_to_feats: Dict[int, np.ndarray] = {}
    autocast = torch.cuda.amp.autocast if (use_amp and device.type=="cuda") else torch.cpu.amp.autocast  # type: ignore[attr‑defined]

    for s in sizes:
        LOGGER.info("Embedding %d‑px crops …", s)
        h = s//2
        patches, coords = [], []
        for x, y in centroids:
            l, t = max(x-h,0), max(y-h,0)
            r, b = min(x+h, image.width), min(y+h, image.height)
            patch = image.crop((l, t, r, b))
            if patch.size != (s, s):
                canvas = Image.new("RGB", (s, s)); canvas.paste(patch, (l-(x-h), t-(y-h))); patch = canvas
            patches.append(patch); coords.append((x, y))

        loader = DataLoader(PatchDataset(patches, coords, tfm), batch_size=batch_size, shuffle=False,
                            num_workers=workers or 0, pin_memory=device.type=="cuda",
                            persistent_workers=(workers or 0) > 0)

        feats: List[np.ndarray] = []
        with torch.inference_mode():
            for imgs, *_ in tqdm(loader, desc=f"{s}px", unit="batch"):
                imgs = imgs.to(device, non_blocking=True)
                with autocast():
                    out = model(pixel_values=imgs)
                hid = torch.stack(out.hidden_states[-4:], 0).mean(0)
                emb = F.normalize(hid[:,1:,:].mean(1), p=2, dim=1).cpu().float().numpy()
                feats.append(emb)
        size_to_feats[s] = np.concatenate(feats, 0)

    combined = np.concatenate([size_to_feats[s] for s in sizes], 1)
    cols: List[str] = []
    for s in sizes:
        d = size_to_feats[s].shape[1]
        cols.extend([f"vit{s}_{i}" for i in range(d)])

    pd.DataFrame(combined, columns=cols).to_csv(output_dir/f"features_{image_path.stem}.csv", index=False)
    pd.DataFrame(centroids, columns=["x_center", "y_center"]).to_csv(output_dir/f"coords_{image_path.stem}.csv", index=False)
    LOGGER.info("Saved %d nuclei × %d‑dim features.", combined.shape[0], combined.shape[1])
    if save_numpy:
        np.save(output_dir/f"features_{image_path.stem}.npy", combined)


# ---------------------------- CLI wrapper --------------------------------- #


def build_argparser():
    p = argparse.ArgumentParser("Nucleus crops → ViT embeddings (single or multi‑scale)")
    p.add_argument("-i", "--image", type=Path, required=True)
    p.add_argument("-m", "--mask", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, default=Path("./output"))
    p.add_argument("--patch_size", type=int, default=16, help="Legacy single crop size (px).")
    p.add_argument("--patch_sizes", type=int, nargs="+", help="Give one or several crop sizes. Overrides --patch_size.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--model_name", type=str, default="facebook/dino-vits16")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--workers", type=int)
    p.add_argument("--crop_region", nargs=4, type=float)
    p.add_argument("--viz_crop_region", nargs=4, type=float)
    p.add_argument("--save_numpy", action="store_true")
    return p


def main():
    a = build_argparser().parse_args()
    extract_and_save_patches(
        image_path=a.image,
        masks_path=a.mask,
        output_dir=a.output,
        patch_size=a.patch_size,
        patch_sizes=a.patch_sizes,
        batch_size=a.batch_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model_name=a.model_name,
        use_amp=not a.no_amp,
        workers=a.workers,
        crop_region=tuple(a.crop_region) if a.crop_region else None,
        viz_crop_region=tuple(a.viz_crop_region) if a.viz_crop_region else None,
        compile_model=not a.no_compile,
        save_numpy=a.save_numpy,
    )


if __name__ == "__main__":
    main()
