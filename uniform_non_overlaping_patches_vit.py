"""
python uniform_non_overlaping_patches_vit.py \
  -i img/IRI_regist_cropped.tif \
  -o ./ViT_outputs \
  --patch_size 16 \
  --stride 16 \
  --batch_size 64


-i, --image <path>
Path to your input image file. In your case: img/IRI_regist_cropped.tif. The script will open and split this image into patches.

-o, --output <dir>
Directory where the CSVs will be written. It defaults to ./ViT_outputs if you don’t supply one. You’ll find

features_<image_stem>.csv (the patch embeddings)

coords_<image_stem>.csv (the patch center coordinates)

--patch_size <int>
The height & width (in pixels) of each square patch.

A value of 16 means you crop 16×16 px tiles.

Patches larger than the model’s input size will be resized down by the processor.

--stride <int>
How many pixels you shift the cropping window each step.

A stride of 16 with patch_size 16 gives non-overlapping patches.

A smaller stride yields overlapping patches (more samples, finer coverage).

--batch_size <int>
Number of patches sent through the model at once.

Larger batches are more GPU-efficient but use more memory.

If you hit out-of-memory errors, lower this number.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
import pandas as pd
from tqdm import tqdm

class PatchDataset(Dataset):
    """
    Dataset that extracts patches from a large image for efficient batch processing.
    """
    def __init__(self, image: Image.Image, patch_size: int, stride: int, transform=None):
        """
        Initialize the dataset with an image, patch size, stride, and optional transform.
        """
        self.image = image.convert('RGB')
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.width, self.height = self.image.size
        self.coords = self._compute_patch_centers()

    def _compute_patch_centers(self):
        """
        Compute the center coordinates for each patch given the patch size and stride.
        """
        centers = []
        for y in range(0, self.height - self.patch_size + 1, self.stride):
            for x in range(0, self.width - self.patch_size + 1, self.stride):
                # Compute center of patch for mapping back to original image.
                centers.append((x + self.patch_size // 2, y + self.patch_size // 2))
        return centers

    def __len__(self):
        """
        Return the total number of patches in the image.
        """
        return len(self.coords)

    def __getitem__(self, idx):
        """
        Extract and return the patch tensor and its center coordinate at index idx.
        """
        x_c, y_c = self.coords[idx]
        left = x_c - self.patch_size // 2
        top = y_c - self.patch_size // 2
        patch = self.image.crop((left, top, left + self.patch_size, top + self.patch_size))
        if self.transform:
            patch = self.transform(patch)
        return patch, x_c, y_c


def extract_features(
    image_path: Path,
    model: ViTModel,
    processor: ViTImageProcessor,
    patch_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    output_dir: Path
):
    """
    Extract ViT features for each patch of the image and save coordinates and features.
    """
    # Load the image from disk.
    img = Image.open(image_path)

    # Create patch dataset with preprocessing transform from the ViT processor.
    transform = transforms.Compose([
        transforms.Resize(processor.size['height']),  # Resize to model input size.
        transforms.CenterCrop(processor.size['height']),  # Center crop if needed.
        transforms.ToTensor(),  # Convert PIL image to tensor.
        transforms.Normalize(processor.image_mean, processor.image_std)  # Normalize as per model.
    ])
    dataset = PatchDataset(img, patch_size, stride, transform=transform)

    # Create DataLoader for batching patches.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    features_list = []
    coords_list = []

    model.to(device)
    model.eval()

    # Iterate over batches of patches and extract features.
    for batch in tqdm(loader, desc="Extracting patch features"):
        patches, xs, ys = batch
        patches = patches.to(device)
        with torch.no_grad():
            outputs = model(pixel_values=patches)
            # Use the last hidden state of class token as feature vector
            feats = outputs.last_hidden_state[:, 0, :].cpu()
        features_list.append(feats)
        coords_list.extend(zip(xs.numpy().tolist(), ys.numpy().tolist()))

    # Concatenate all batch features into a single tensor.
    all_features = torch.cat(features_list, dim=0).numpy()

    # Create DataFrame for features and coordinates.
    df_feats = pd.DataFrame(all_features)
    df_coords = pd.DataFrame(coords_list, columns=["x_center", "y_center"])

    # Ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features and coordinates to CSV files.
    feats_file = output_dir / f"features_{image_path.stem}.csv"
    coords_file = output_dir / f"coords_{image_path.stem}.csv"
    df_feats.to_csv(feats_file, index=False)
    df_coords.to_csv(coords_file, index=False)
    print(f"Saved features to {feats_file}")
    print(f"Saved coordinates to {coords_file}")


def main():
    """
    Parse command-line arguments and run feature extraction pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Extract ViT features from large images by dividing into patches."
    )
    parser.add_argument(
        "-i", "--image", required=True, type=Path,
        help="Path to the input image file."
    )
    parser.add_argument(
        "-o", "--output", default=Path("./ViT_outputs"), type=Path,
        help="Directory to save output feature and coordinate files."
    )
    parser.add_argument(
        "--patch_size", default=16, type=int,
        help="Size of each square patch in pixels."
    )
    parser.add_argument(
        "--stride", default=16, type=int,
        help="Stride between patches in pixels."
    )
    parser.add_argument(
        "--batch_size", default=64, type=int,
        help="Number of patches to process per batch."
    )
    args = parser.parse_args()

    # Detect device (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize ViT processor and model from Hugging Face.
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', output_hidden_states=True, use_safe_tensors=True).to(device).eval()

    # Run the extraction.
    extract_features(
        image_path=args.image,
        model=model,
        processor=processor,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()
