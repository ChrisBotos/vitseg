"""
python segmentation_mask_bbox_patches_vit.py \
  -i img/IRI_regist_cropped.tif \
  -m segmentation_masks.npy \
  -o ./ViT_cell_patches \
  --margin_factor 1.2 \
  --batch_size 64 \
  --mask_overlay

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
from PIL import Image, ImageDraw, ImageColor
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
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
    Given a mask array, compute tight bounding boxes for each individual cell mask.
    Supports:
      - 2D label maps (H, W) where each integer >0 is a cell ID.
      - 3D boolean masks (N, H, W).
    Returns list of tuples (x0, y0, x1, y1).
    """
    bboxes = []
    # Case 1: 2D labeled mask map
    if masks.ndim == 2 and np.issubdtype(masks.dtype, np.integer):
        labels = np.unique(masks)
        labels = labels[labels > 0]  # ignore background 0
        print(f"Detected {len(labels)} unique labels in mask map.")
        for lab in labels:
            mask_bool = masks == lab
            ys, xs = np.where(mask_bool)
            if xs.size == 0:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            bboxes.append((int(x0), int(y0), int(x1), int(y1)))
    else:
        # Ensure 3D mask array: (N, H, W)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        print(f"Loaded {masks.shape[0]} mask planes from .npy file.")
        for mask in masks:
            mask_bool = mask.astype(bool)
            ys, xs = np.where(mask_bool)
            if xs.size == 0:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            bboxes.append((int(x0), int(y0), int(x1), int(y1)))
    print(f"Computed {len(bboxes)} bounding boxes from masks.")
    return bboxes


def overlay_masks(image: Image.Image, masks: np.ndarray, output_path: Path):
    """
    Overlay segmentation masks in random semi-transparent colors on the input image and save.
    """
    base = image.convert('RGBA')
    overlay = Image.new('RGBA', base.size)
    rng = np.random.default_rng()
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    for mask in masks:
        # generate a random RGB color
        r, g, b = rng.integers(0, 256, size=3)
        color = (int(r), int(g), int(b), 100)
        # create a solid color image
        mask_layer = Image.new('RGBA', base.size, color)
        # create a mask image for alpha composite
        mask_img = Image.fromarray((mask.astype('uint8') * 255))
        # composite mask_layer over overlay using mask_img as alpha
        overlay = Image.composite(mask_layer, overlay, mask_img)
    result = Image.alpha_composite(base, overlay)
    result.convert('RGB').save(output_path)
    print(f"Saved mask overlay image to {output_path}")


def extract_and_save_patches(
    image_path: Path,
    masks_path: Path,
    output_dir: Path,
    margin_factor: float,
    batch_size: int,
    device: torch.device,
    mask_overlay: bool
):
    """
    Load image and segmentation masks, optionally overlay masks, draw boxes, extract padded patches,
    run DINO ViT and save results.
    """
    img = Image.open(image_path).convert('RGB')
    masks = np.load(masks_path)

    # Optional mask overlay
    if mask_overlay:
        overlay_file = output_dir / f"{image_path.stem}_mask_overlay.png"
        overlay_masks(img, masks, overlay_file)

    # Compute bounding boxes and debug print counts
    bboxes = compute_bounding_boxes(masks)

    # Draw bounding boxes on image
    img_boxes = img.copy()
    draw = ImageDraw.Draw(img_boxes)
    patches, centers = [], []
    for x0, y0, x1, y1 in bboxes:
        w, h = x1 - x0 + 1, y1 - y0 + 1
        cx, cy = x0 + w/2, y0 + h/2
        side = int(max(1, margin_factor * max(w, h)))
        left = max(int(cx - side/2), 0)
        top = max(int(cy - side/2), 0)
        right = min(left + side, img.width)
        bottom = min(top + side, img.height)
        draw.rectangle([left, top, right, bottom], outline='red', width=2)
        patches.append(img.crop((left, top, right, bottom)))
        centers.append((int(cx), int(cy)))
    boxes_file = output_dir / f"{image_path.stem}_bboxes.png"
    img_boxes.save(boxes_file)
    print(f"Saved bounding-box image to {boxes_file}")

    # Initialize processor and model only once
    processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
    model = ViTModel.from_pretrained('facebook/dino-vits16')
    model.to(device).eval()

    size = processor.size if isinstance(processor.size, (int, tuple)) else processor.size['height']
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(processor.image_mean, processor.image_std)
    ])

    dataset = MaskPatchDataset(patches, centers, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    feats_list, coords = [], []
    for imgs, xs, ys in tqdm(loader, desc="Extracting DINO features"):
        imgs = imgs.to(device)
        with torch.no_grad():
            outputs = model(pixel_values=imgs)
            feats_list.append(outputs.last_hidden_state[:, 0, :].cpu())
        coords.extend(zip(xs.tolist(), ys.tolist()))

    feats_all = torch.cat(feats_list, dim=0).numpy()
    pd.DataFrame(feats_all).to_csv(output_dir / f"features_{image_path.stem}.csv", index=False)
    pd.DataFrame(coords, columns=['x_center', 'y_center']).to_csv(
        output_dir / f"coords_{image_path.stem}.csv", index=False)
    print(f"Saved features and coordinates to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINO-ViT features from cell-based bounding-box patches."
    )
    parser.add_argument("-i", "--image", required=True, type=Path,
        help="Path to the input image file.")
    parser.add_argument("-m", "--masks", required=True, type=Path,
        help="Path to the .npy file containing segmentation masks.")
    parser.add_argument("-o", "--output", default=Path("./ViT_outputs"), type=Path,
        help="Directory to save output files.")
    parser.add_argument("--margin_factor", default=1.2, type=float,
        help="Factor to enlarge bounding boxes around each cell.")
    parser.add_argument("--batch_size", default=64, type=int,
        help="Number of patches to process per batch.")
    parser.add_argument("--mask_overlay", action="store_true",
        help="If set, save an overlay of segmentation masks in random colors.")
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    extract_and_save_patches(
        image_path=args.image,
        masks_path=args.masks,
        output_dir=output_dir,
        margin_factor=args.margin_factor,
        batch_size=args.batch_size,
        device=device,
        mask_overlay=args.mask_overlay
    )

if __name__ == "__main__":
    main()
