"""
python segmentation_mask_dynamic_patches_vit.py \
  -i img/IRI_regist_cropped.tif \
  -m segmentation_masks.npy \
  -o ./ViT_cell_patches \
  --patch_size 16 \
  --batch_size 512 \
  --crop_region 0 1 0 1 \
  --viz_crop_region 0 1 0 1
"""


import os
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw

def compute_centroids(masks: np.ndarray):
    """
    Given a mask array, compute centroids for each individual cell mask.
    Supports:
      - 2D label maps (H, W) where each integer >0 is a cell ID.
      - 3D boolean masks (N, H, W).
    Returns list of tuples (x_center, y_center).
    """
    centroids = []
    if masks.ndim == 2 and np.issubdtype(masks.dtype, np.integer):
        labels = np.unique(masks)
        labels = labels[labels > 0]
        for lab in labels:
            mask_bool = (masks == lab)
            ys, xs = np.where(mask_bool)
            if xs.size == 0:
                continue
            cx = xs.mean()
            cy = ys.mean()
            centroids.append((int(cx), int(cy)))
    else:
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        for mask in masks:
            mask_bool = mask.astype(bool)
            ys, xs = np.where(mask_bool)
            if xs.size == 0:
                continue
            cx = xs.mean()
            cy = ys.mean()
            centroids.append((int(cx), int(cy)))
    return centroids


def extract_and_save_patches(
    image_path: Path,
    masks_path: Path,
    output_dir: Path,
    patch_size: int,
    batch_size: int,
    device: torch.device,
    crop_region: tuple[float, float, float, float] | None = None,
    viz_crop_region: tuple[float, float, float, float] | None = None
):
    # Inform user about model loading.
    print("[Init] Loading DINO-ViT model and processor.")
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from transformers import ViTImageProcessor, ViTModel
    import pandas as pd
    from tqdm import tqdm

    # Load the processor and model for DINO-ViT.
    processor = ViTImageProcessor.from_pretrained(
        'facebook/dino-vits16', local_files_only=True
    )
    model = ViTModel.from_pretrained(
        'facebook/dino-vits16', local_files_only=True
    )
    model.to(device).eval()
    print("[Init] Model and processor ready.")

    # Load image and masks.
    img_full = Image.open(image_path).convert('RGB')
    masks = np.load(masks_path)
    orig_w, orig_h = img_full.width, img_full.height

    # Primary crop region interpreted as xmin, xmax, ymin, ymax fractions.
    if crop_region is not None:
        xmin_frac, xmax_frac, ymin_frac, ymax_frac = crop_region
        x0 = int(orig_w * xmin_frac)
        x1 = int(orig_w * xmax_frac)
        y0 = int(orig_h * ymin_frac)
        y1 = int(orig_h * ymax_frac)
        # Validate region
        if x1 <= x0 or y1 <= y0:
            print(f"Warning: Invalid crop_region {crop_region}, skipping primary crop.")
            img = img_full
        else:
            img = img_full.crop((x0, y0, x1, y1))
            if masks.ndim == 2:
                masks = masks[y0:y1, x0:x1]
            else:
                masks = masks[:, y0:y1, x0:x1]
            print(f"[Info] Primary crop applied: region=({x0},{y0}) to ({x1},{y1}).")
    else:
        img = img_full

    # Compute centroids and validate.
    centers = compute_centroids(masks)
    if not centers:
        print("Warning: No centroids found. Please check the mask file or crop_region settings.")
        return

    # Visualization crop region interpreted as xmin, xmax, ymin, ymax fractions on primary-cropped image.
    if viz_crop_region is not None:
        vxmin_frac, vxmax_frac, vymin_frac, vymax_frac = viz_crop_region
        cw, ch = img.width, img.height
        vx0 = int(cw * vxmin_frac)
        vx1 = int(cw * vxmax_frac)
        vy0 = int(ch * vymin_frac)
        vy1 = int(ch * vymax_frac)
        if vx1 <= vx0 or vy1 <= vy0:
            print(f"Warning: Invalid viz_crop_region {viz_crop_region}, skipping viz crop.")
        else:
            img_viz = img.crop((vx0, vy0, vx1, vy1))
            draw = ImageDraw.Draw(img_viz)
            half = patch_size // 2
            for cx, cy in centers:
                # adjust centroid to viz coords
                lx = cx - vx0 - half
                ty = cy - vy0 - half
                rx = cx - vx0 + half
                by = cy - vy0 + half
                draw.rectangle([lx, ty, rx, by], outline='red', width=2)
            img_viz.save(output_dir / f"viz_crop_{image_path.stem}.png")
            print(f"[Info] Visualization crop saved: region=({vx0},{vy0}) to ({vx1},{vy1}).")

    # Extract and pad patches around each centroid.
    half = patch_size // 2
    patches, coords = [], []
    for cx, cy in centers:
        left = cx - half
        top = cy - half
        right = cx + half
        bottom = cy + half
        crop_left = max(left, 0)
        crop_top = max(top, 0)
        crop_right = min(right, img.width)
        crop_bottom = min(bottom, img.height)
        patch = img.crop((crop_left, crop_top, crop_right, crop_bottom))
        if patch.size != (patch_size, patch_size):
            padded = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
            offset_x = crop_left - left
            offset_y = crop_top - top
            padded.paste(patch, (offset_x, offset_y))
            patch = padded
        patches.append(patch)
        coords.append((cx, cy))

    # Prepare transforms for the model.
    size_param = processor.size
    size = size_param if isinstance(size_param, (int, tuple)) else size_param['height']
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(processor.image_mean, processor.image_std)
    ])

    # Define dataset and dataloader.
    class PatchDataset(Dataset):
        def __init__(self, patches, coords, transform=None):
            self.patches = patches
            self.coords = coords
            self.transform = transform

        def __len__(self) -> int:
            return len(self.patches)

        def __getitem__(self, idx):
            patch = self.patches[idx]
            if self.transform:
                patch = self.transform(patch)
            x_c, y_c = self.coords[idx]
            return patch, x_c, y_c

    loader = DataLoader(
        PatchDataset(patches, coords, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Extract features in batches.
    features_list, centers_list = [], []
    for imgs_batch, xs_batch, ys_batch in tqdm(loader, desc="Extracting DINO features."):
        imgs_batch = imgs_batch.to(device)
        with torch.no_grad():
            outputs = model(pixel_values=imgs_batch)
            features_list.append(outputs.last_hidden_state[:, 0, :].cpu())
        centers_list.extend(zip(xs_batch.tolist(), ys_batch.tolist()))

    if not features_list:
        print("Warning: No features were extracted. Ensure that patches are correctly generated.")
        return

    # Save features and coordinates to CSV files.
    feats_all = torch.cat(features_list, dim=0).numpy()
    pd.DataFrame(feats_all).to_csv(
        output_dir / f"features_{image_path.stem}.csv", index=False
    )
    pd.DataFrame(centers_list, columns=['x_center', 'y_center']).to_csv(
        output_dir / f"coords_{image_path.stem}.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract fixed-size DINO-ViT patches centered on each mask centroid with optional visualization crop."
    )
    parser.add_argument(
        "-i", "--image", required=True, type=Path,
        help="Path to the input image file."
    )
    parser.add_argument(
        "-m", "--masks", required=True, type=Path,
        help="Path to the .npy file containing segmentation masks."
    )
    parser.add_argument(
        "-o", "--output", default=Path("./ViT_cell_patches"), type=Path,
        help="Directory to save output files."
    )
    parser.add_argument(
        "--patch_size", default=16, type=int,
        help="Size of the square patch around each centroid."
    )
    parser.add_argument(
        "--batch_size", default=512, type=int,
        help="Number of patches to process per batch."
    )
    parser.add_argument(
        "--crop_region", nargs=4, type=float, metavar=('XMIN','XMAX','YMIN','YMAX'),
        help="Optional primary crop region as fractions (xmin, xmax, ymin, ymax) of the full image."
    )
    parser.add_argument(
        "--viz_crop_region", nargs=4, type=float, metavar=('XMIN','XMAX','YMIN','YMAX'),
        help="Optional visualization crop region as fractions (xmin, xmax, ymin, ymax) on the primary-cropped image."
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_region = tuple(args.crop_region) if args.crop_region else None
    viz_crop_region = tuple(args.viz_crop_region) if args.viz_crop_region else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.")

    extract_and_save_patches(
        image_path=args.image,
        masks_path=args.masks,
        output_dir=output_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        device=device,
        crop_region=crop_region,
        viz_crop_region=viz_crop_region
    )


if __name__ == "__main__":
    main()
