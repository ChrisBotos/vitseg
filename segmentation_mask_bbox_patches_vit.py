"""
python segmentation_mask_bbox_patches_vit.py \
  -i img/IRI_regist_cropped.tif \
  -m segmentation_masks.npy \
  -o ./ViT_cell_patches \
  --margin_factor 1.2 \
  --batch_size 64

-i, --image <path>
Path to your input image file (e.g. img/IRI_regist_cropped.tif). This is the RGB image on which your segmentation masks were generated. The script will open it, draw bounding boxes, and crop patches from it.

-m, --masks <path>
Path to the NumPy file containing your segmentation masks (e.g. segmentation_masks.npy). This should be an array of shape (N, H, W) where each mask[i] is a Boolean or binary mask for one detected cell. The script computes tight bounding boxes from these masks.

-o, --output <dir>
Directory where all outputs will be saved. Defaults to ./ViT_outputs. Under this folder you will find:

IRI_regist_cropped_bboxes.png – the original image overlaid with red boxes.

features_IRI_regist_cropped.csv – a table of DINO-ViT embeddings (one row per patch).

coords_IRI_regist_cropped.csv – the (x_center, y_center) of each crop.

--margin_factor <float>
Multiplier to pad each mask’s tight bounding box before cropping.

A value of 1.0 means “no padding” – you crop exactly the mask’s box.

1.2 adds 20% extra on the longer side (our default), giving a bit of pericellular context.

Larger values (e.g. 1.5 or 2.0) include more surrounding tissue.

--batch_size <int>
Number of patches to process through DINO-ViT at once. Adjust this to fit your GPU memory.

Higher values (e.g. 64, 128) tend to run faster but use more VRAM.

Lower values (e.g. 16, 32) are safer on cards with limited memory.
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
import pandas as pd
from tqdm import tqdm

class MaskPatchDataset(Dataset):
    """
    Dataset that holds variable-size patches extracted from segmentation masks and applies preprocessing.
    """
    def __init__(self, patches, centers, transform=None):
        """
        Initialize with a list of PIL.Image patches, their center coordinates, and an optional transform.
        """
        self.patches = patches
        self.centers = centers
        self.transform = transform

    def __len__(self):
        """
        Return the total number of patches available.
        """
        return len(self.patches)

    def __getitem__(self, idx):
        """
        Return the transformed patch tensor and its center (x, y) coordinate.
        """
        patch = self.patches[idx]
        if self.transform:
            patch = self.transform(patch)
        x_c, y_c = self.centers[idx]
        return patch, x_c, y_c


def compute_bounding_boxes(masks: np.ndarray):
    """
    Given a boolean array of shape (N, H, W), compute tight bounding boxes for each mask.
    Returns list of tuples (x0, y0, x1, y1).
    """
    bboxes = []
    for mask in masks:
        # Find indices where mask is True.
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue  # skip empty masks.
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bboxes.append((x0, y0, x1, y1))
    return bboxes


def extract_and_save_patches(
    image_path: Path,
    masks_path: Path,
    output_dir: Path,
    margin_factor: float,
    batch_size: int,
    device: torch.device
):
    """
    Load image and segmentation masks, draw boxes, extract padded patches, run DINO ViT and save results.
    """
    # Load original image.
    img = Image.open(image_path).convert('RGB')
    width, height = img.size

    # Load masks from .npy file expecting shape (N, H, W).
    masks = np.load(masks_path)

    # Compute bounding boxes for each mask.
    bboxes = compute_bounding_boxes(masks)

    # Prepare to draw bounding boxes in red.
    img_boxes = img.copy()
    draw = ImageDraw.Draw(img_boxes)

    patches, centers = [], []
    for (x0, y0, x1, y1) in bboxes:
        # determine box dimensions and center.
        w, h = x1 - x0 + 1, y1 - y0 + 1
        cx, cy = x0 + w/2, y0 + h/2
        # compute padded square side length.
        side = int(min(max(margin_factor * max(w, h), 1), max(width, height)))
        # calculate top-left of the square patch.
        left = int(max(cx - side/2, 0))
        top = int(max(cy - side/2, 0))
        right = int(min(left + side, width))
        bottom = int(min(top + side, height))
        # draw rectangle.
        draw.rectangle([left, top, right, bottom], outline='red', width=2)
        # crop patch and record.
        patch = img.crop((left, top, right, bottom))
        patches.append(patch)
        centers.append((int(cx), int(cy)))

    # Ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save image with drawn boxes.
    boxes_file = output_dir / f"{image_path.stem}_bboxes.png"
    img_boxes.save(boxes_file)
    print(f"Saved bounding-box image to {boxes_file}")

    # Prepare DINO ViT.
    extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
    model = ViTModel.from_pretrained('facebook/dino-vits16')
    model.to(device).eval()

    # Define preprocessing transform using extractor stats.
    transform = transforms.Compose([
        transforms.Resize(extractor.size),
        transforms.CenterCrop(extractor.size),
        transforms.ToTensor(),
        transforms.Normalize(extractor.image_mean, extractor.image_std)
    ])

    # Build dataset and loader.
    dataset = MaskPatchDataset(patches, centers, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    feats, coords = [], []
    # Process patches through DINO ViT.
    for batch in tqdm(loader, desc="Extracting DINO features"):
        imgs, xs, ys = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            outputs = model(pixel_values=imgs)
            # take class token features.
            out_feats = outputs.last_hidden_state[:, 0, :].cpu()
        feats.append(out_feats)
        coords.extend(zip(xs.numpy().tolist(), ys.numpy().tolist()))

    # Concatenate and save.
    feats_all = torch.cat(feats, dim=0).numpy()
    df_feats = pd.DataFrame(feats_all)
    df_coords = pd.DataFrame(coords, columns=['x_center', 'y_center'])
    feats_file = output_dir / f"features_{image_path.stem}.csv"
    coords_file = output_dir / f"coords_{image_path.stem}.csv"
    df_feats.to_csv(feats_file, index=False)
    df_coords.to_csv(coords_file, index=False)
    print(f"Saved features to {feats_file}")
    print(f"Saved coordinates to {coords_file}")


def main():
    """
    Parse arguments and invoke the patch extraction + feature pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Extract DINO-ViT features from cell-based bounding-box patches."
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
        "-o", "--output", default=Path("./ViT_outputs"), type=Path,
        help="Directory to save output files (CSV features, coords, bbox image)."
    )
    parser.add_argument(
        "--margin_factor", default=1.2, type=float,
        help="Factor to enlarge bounding boxes uniformly around each cell."
    )
    parser.add_argument(
        "--batch_size", default=64, type=int,
        help="Number of patches to process per batch through the model."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    extract_and_save_patches(
        image_path=args.image,
        masks_path=args.masks,
        output_dir=args.output,
        margin_factor=args.margin_factor,
        batch_size=args.batch_size,
        device=device
    )

if __name__ == "__main__":
    main()
