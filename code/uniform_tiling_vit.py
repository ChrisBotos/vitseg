"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: uniform_tiling_vit.py.
Description:
    Uniform tiling Vision Transformer feature extraction for binary mask analysis.
    Splits the entire binary image into regular grid tiles and extracts ViT embeddings
    for clustering analysis, providing an alternative to dynamic patch extraction.

    Key features for bioinformatician users:
        • **Uniform grid tiling** – Divides the binary image into regular non-overlapping
          tiles starting from top-left corner, ensuring complete coverage of the entire
          tissue section including edge regions.
        • **Edge handling with black padding** – Border tiles that extend beyond image
          boundaries are automatically padded with black pixels, ensuring consistent
          tile sizes while preserving edge information.
        • **Binary mask processing** – Works directly with binary mask images where
          white pixels (255) represent nuclei regions and black pixels (0) represent
          background, enabling whole-tissue analysis.
        • **Standard ViT clustering** – Extracts features from all tiles regardless of
          content, allowing clustering algorithms to identify tissue patterns and
          spatial organization at the tile level.
        • **Memory-efficient processing** – Uses batch processing and memory mapping
          to handle large tissue images without memory overflow.

    Scientific context:
        This approach is particularly useful for analyzing tissue architecture,
        spatial patterns, and regional heterogeneity in kidney injury models where
        understanding broader tissue organization is as important as individual
        cell characteristics.

Dependencies:
    • Python>=3.10.
    • numpy, pandas, torch>=2.2, torchvision, transformers, PIL, tqdm.
    • scikit-learn for feature processing.

Usage:
    python code/uniform_tiling_vit.py \
        --image data/ss_bIRI2_binary_mask.tif \
        --output results/VIT_uniform_tiles \
        --patch_size 64 \
        --stride 64 \
        --batch_size 512 \
        --model_name facebook/dino-vits16

"""

import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

# Optional import for binary enhancement.
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    LOGGER.warning("opencv-python not available. Binary enhancement will be disabled.")

# Disable PIL image size limit for large microscopy images.
Image.MAX_IMAGE_PIXELS = None


class UniformTileDataset(Dataset):
    """
    Dataset for extracting uniform tiles from binary mask images.
    
    This dataset divides the input image into regular grid tiles and provides
    efficient batch processing for ViT feature extraction.
    """
    
    def __init__(self, image: Image.Image, patch_size: int, stride: int, transform=None):
        """
        Initialize dataset with image and tiling parameters.
        
        Args:
            image: PIL Image object (binary mask).
            patch_size: Size of square tiles in pixels.
            stride: Stride between tile centers in pixels.
            transform: Optional preprocessing transform for tiles.
        """
        self.image = image.convert('RGB')  # Ensure RGB format for ViT.
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.width, self.height = self.image.size
        self.coords = self._compute_tile_centers()
        
        LOGGER.debug(f"Image dimensions: {self.width}x{self.height}")
        LOGGER.debug(f"Patch size: {patch_size}, stride: {stride}")
        LOGGER.debug(f"Generated {len(self.coords)} tiles (includes edge tiles with black padding)")

        # Calculate how many tiles will need padding.
        edge_tiles = 0
        for x_center, y_center in self.coords:
            left = x_center - self.patch_size // 2
            top = y_center - self.patch_size // 2
            right = left + self.patch_size
            bottom = top + self.patch_size

            if (left < 0 or top < 0 or right > self.width or bottom > self.height):
                edge_tiles += 1

        LOGGER.debug(f"{edge_tiles} tiles will require black padding for edge coverage")

    def _compute_tile_centers(self):
        """
        Compute center coordinates for each tile in the regular grid.

        Tiles start from top-left corner and cover the entire image.
        Border tiles that extend beyond image boundaries will be padded with black pixels.

        Returns:
            List of (x_center, y_center) tuples for each tile.
        """
        centers = []

        # Calculate number of tiles needed to cover entire image including edges.
        tiles_x = (self.width + self.stride - 1) // self.stride  # Ceiling division.
        tiles_y = (self.height + self.stride - 1) // self.stride  # Ceiling division.

        LOGGER.debug(f"Image coverage requires {tiles_x} x {tiles_y} = {tiles_x * tiles_y} tiles")

        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                # Calculate tile top-left corner starting from (0,0).
                left = tile_x * self.stride
                top = tile_y * self.stride

                # Calculate tile center coordinates.
                center_x = left + self.patch_size // 2
                center_y = top + self.patch_size // 2
                centers.append((center_x, center_y))

        return centers

    def __len__(self):
        """Return total number of tiles."""
        return len(self.coords)

    def __getitem__(self, idx):
        """
        Extract tile at given index and return with coordinates.

        Handles border tiles by padding with black pixels when they extend
        beyond image boundaries.

        Args:
            idx: Tile index.

        Returns:
            Tuple of (tile_tensor, x_center, y_center).
        """
        x_center, y_center = self.coords[idx]

        # Calculate tile boundaries.
        left = x_center - self.patch_size // 2
        top = y_center - self.patch_size // 2
        right = left + self.patch_size
        bottom = top + self.patch_size

        # Check if tile extends beyond image boundaries.
        needs_padding = (left < 0 or top < 0 or
                        right > self.width or bottom > self.height)

        if needs_padding:
            # Create black tile and paste image portion onto it.
            tile = self._create_padded_tile(left, top, right, bottom)
        else:
            # Extract tile normally if it fits within image boundaries.
            tile = self.image.crop((left, top, right, bottom))

        # Apply preprocessing transform if provided.
        if self.transform:
            tile = self.transform(tile)

        return tile, x_center, y_center

    def _create_padded_tile(self, left: int, top: int, right: int, bottom: int):
        """
        Create a padded tile with black background for border cases.

        When tiles extend beyond image boundaries, this method creates a black
        tile of the correct size and pastes the available image portion onto it.

        Args:
            left, top, right, bottom: Tile boundaries (may extend beyond image).

        Returns:
            PIL Image of size (patch_size, patch_size) with black padding.
        """
        # Create black tile of correct size.
        padded_tile = Image.new('RGB', (self.patch_size, self.patch_size), color=(0, 0, 0))

        # Calculate intersection with actual image boundaries.
        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(self.width, right)
        crop_bottom = min(self.height, bottom)

        # Only proceed if there's an actual intersection.
        if crop_left < crop_right and crop_top < crop_bottom:
            # Extract the available image portion.
            image_portion = self.image.crop((crop_left, crop_top, crop_right, crop_bottom))

            # Calculate where to paste this portion in the padded tile.
            paste_x = crop_left - left
            paste_y = crop_top - top

            # Paste image portion onto black background.
            padded_tile.paste(image_portion, (paste_x, paste_y))

        return padded_tile


def extract_uniform_features(
    image_path: Path,
    model: ViTModel,
    processor: ViTImageProcessor,
    patch_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    workers: int = 4
):
    """
    Extract ViT features from uniform tiles across the entire image.
    
    Args:
        image_path: Path to binary mask image.
        model: Pre-loaded ViT model.
        processor: ViT image processor for preprocessing.
        patch_size: Size of square tiles in pixels.
        stride: Stride between tiles in pixels.
        batch_size: Number of tiles per processing batch.
        device: PyTorch device (CPU or CUDA).
        output_dir: Directory to save output files.
        workers: Number of CPU workers for data loading.
    """
    LOGGER.debug(f"Loading image from {image_path}")

    # Load binary mask image.
    image = Image.open(image_path)
    LOGGER.debug(f"Image loaded successfully, size: {image.size}")

    # Check for very large images and suggest downsampling.
    width, height = image.size
    total_pixels = width * height

    if total_pixels > 100_000_000:  # 100 megapixels.
        LOGGER.warning(f"Very large image detected ({width}x{height} = {total_pixels:,} pixels)")
        LOGGER.warning(f"This will generate {((width-patch_size)//stride+1) * ((height-patch_size)//stride+1):,} tiles")
        LOGGER.warning(f"Consider downsampling the image or using larger patch sizes for better performance")

        # Optionally downsample very large images.
        if total_pixels > 200_000_000:  # 200 megapixels.
            downsample_factor = 2
            new_width = width // downsample_factor
            new_height = height // downsample_factor
            LOGGER.warning(f"Auto-downsampling by factor {downsample_factor} to {new_width}x{new_height}")
            image = image.resize((new_width, new_height), Image.LANCZOS)
            patch_size = patch_size // downsample_factor
            stride = stride // downsample_factor
            LOGGER.debug(f"Adjusted patch_size to {patch_size}, stride to {stride}")

    # Create preprocessing transform matching ViT requirements.
    transform = transforms.Compose([
        transforms.Resize(processor.size['height']),  # Resize to model input size.
        transforms.CenterCrop(processor.size['height']),  # Center crop if needed.
        transforms.ToTensor(),  # Convert to tensor.
        transforms.Normalize(processor.image_mean, processor.image_std)  # Normalize.
    ])

    # Create tile dataset.
    dataset = UniformTileDataset(image, patch_size, stride, transform=transform)

    # Adjust batch size for very large datasets to prevent memory issues.
    if len(dataset) > 500000:
        adjusted_batch_size = min(batch_size, 512)
        LOGGER.debug(f"Large dataset detected ({len(dataset)} tiles), reducing batch size to {adjusted_batch_size}")
        batch_size = adjusted_batch_size
    elif len(dataset) > 200000:
        adjusted_batch_size = min(batch_size, 1024)
        LOGGER.debug(f"Medium dataset detected ({len(dataset)} tiles), reducing batch size to {adjusted_batch_size}")
        batch_size = adjusted_batch_size
    
    # Create data loader for batch processing with memory-efficient settings.
    # Reduce workers and disable pin_memory for large datasets to prevent OOM.
    effective_workers = min(workers, 2) if len(dataset) > 100000 else workers
    use_pin_memory = device.type == 'cuda' and len(dataset) < 50000

    LOGGER.debug(f"Using {effective_workers} workers, pin_memory={use_pin_memory}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=use_pin_memory,
        persistent_workers=False  # Don't keep workers alive between epochs.
    )

    # Initialize storage for features and coordinates.
    # For large datasets, we'll save features incrementally to avoid memory issues.
    features_list = []
    coords_list = []

    # Create temporary files for incremental saving if dataset is very large.
    save_incrementally = len(dataset) > 100000
    if save_incrementally:
        LOGGER.debug(f"Large dataset detected, will save features incrementally")
        temp_features_file = output_dir / f"temp_features_{image_path.stem}.npy"
        temp_coords_file = output_dir / f"temp_coords_{image_path.stem}.csv"

        # Determine feature dimension from model configuration.
        feature_dim = model.config.hidden_size

        # Create memory-mapped array for incremental saving.
        temp_features_mmap = np.memmap(
            temp_features_file,
            dtype=np.float32,
            mode='w+',
            shape=(len(dataset), feature_dim)
        )

        # Initialize coordinates list for incremental saving.
        temp_coords_list = []

    # Move model to device and set to evaluation mode.
    model.to(device)
    model.eval()

    LOGGER.debug(f"Processing {len(dataset)} tiles in {len(loader)} batches")

    # Estimate memory usage.
    feature_dim = model.config.hidden_size
    estimated_memory_gb = (len(dataset) * feature_dim * 4) / (1024**3)  # 4 bytes per float32.
    LOGGER.debug(f"Estimated memory usage for features: {estimated_memory_gb:.2f} GB")

    if estimated_memory_gb > 8:
        LOGGER.warning(f"High memory usage expected ({estimated_memory_gb:.2f} GB)")
        LOGGER.warning(f"Consider using smaller patch sizes or downsampling the image")

    # Process tiles in batches with memory management.
    current_sample_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Extracting tile features")):
            try:
                tiles, x_coords, y_coords = batch
                tiles = tiles.to(device, non_blocking=True)

                # Extract features using ViT model.
                outputs = model(pixel_values=tiles)

                # Use CLS token embeddings as tile features.
                tile_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                # Store features and coordinates.
                if save_incrementally:
                    # Save directly to memory-mapped array.
                    batch_size_actual = tile_features.shape[0]
                    temp_features_mmap[current_sample_idx:current_sample_idx + batch_size_actual] = tile_features
                    temp_coords_list.extend(zip(x_coords.numpy().tolist(), y_coords.numpy().tolist()))
                    current_sample_idx += batch_size_actual
                else:
                    # Store in memory for smaller datasets.
                    features_list.append(torch.from_numpy(tile_features))
                    coords_list.extend(zip(x_coords.numpy().tolist(), y_coords.numpy().tolist()))

                # Clear GPU memory periodically.
                if (batch_idx + 1) % 20 == 0:
                    torch.cuda.empty_cache() if device.type == 'cuda' else None

                # Progress reporting for large datasets.
                if (batch_idx + 1) % max(1, len(loader) // 20) == 0:
                    progress_pct = (batch_idx + 1) / len(loader) * 100
                    LOGGER.debug(f"Processed {batch_idx + 1}/{len(loader)} batches ({progress_pct:.1f}%)")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    LOGGER.warning(f"GPU memory error at batch {batch_idx + 1}, clearing cache and retrying...")
                    torch.cuda.empty_cache() if device.type == 'cuda' else None

                    # Retry with individual tile processing.
                    for i in range(len(tiles)):
                        single_tile = tiles[i:i+1]
                        single_output = model(pixel_values=single_tile)
                        single_feature = single_output.last_hidden_state[:, 0, :].cpu().numpy()

                        if save_incrementally:
                            temp_features_mmap[current_sample_idx] = single_feature[0]
                            temp_coords_list.append((x_coords[i].item(), y_coords[i].item()))
                            current_sample_idx += 1
                        else:
                            if i == 0:
                                batch_features = single_feature
                            else:
                                batch_features = np.vstack([batch_features, single_feature])

                    if not save_incrementally:
                        features_list.append(torch.from_numpy(batch_features))
                        coords_list.extend(zip(x_coords.numpy().tolist(), y_coords.numpy().tolist()))
                else:
                    raise e

    # Handle final feature processing based on storage method.
    if save_incrementally:
        LOGGER.debug(f"Using incremental save method")
        # Features are already saved in memory-mapped file.
        all_features = np.array(temp_features_mmap)
        coords_list = temp_coords_list

        # Flush memory-mapped array.
        del temp_features_mmap

        LOGGER.debug(f"Final feature array shape: {all_features.shape}")
    else:
        LOGGER.debug(f"Using in-memory method")
        # Concatenate all features from memory.
        all_features = torch.cat(features_list, dim=0).numpy()
        LOGGER.debug(f"Final feature array shape: {all_features.shape}")

    # Create DataFrames for features and coordinates.
    df_features = pd.DataFrame(all_features)
    df_coords = pd.DataFrame(coords_list, columns=["x_center", "y_center"])

    # Ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output file names using image stem.
    image_stem = image_path.stem
    features_csv = output_dir / f"features_{image_stem}.csv"
    features_npy = output_dir / f"features_{image_stem}.npy"
    coords_csv = output_dir / f"coords_{image_stem}.csv"

    # Save features in both CSV and NPY formats.
    df_features.to_csv(features_csv, index=False)
    np.save(features_npy, all_features)
    df_coords.to_csv(coords_csv, index=False)

    LOGGER.debug(f"Features saved to {features_csv}")
    LOGGER.debug(f"Features saved to {features_npy}")
    LOGGER.debug(f"Coordinates saved to {coords_csv}")
    
    print(f"✓ Uniform tiling feature extraction completed successfully.")
    print(f"  • Processed {len(dataset)} tiles")
    print(f"  • Feature dimensionality: {all_features.shape[1]}")
    print(f"  • Output files saved to: {output_dir}")


def enhance_binary_image(image_array):
    """
    Apply morphological enhancement to binary mask for better ViT processing.

    Args:
        image_array: NumPy array of binary mask.

    Returns:
        Enhanced binary mask array.
    """
    if not CV2_AVAILABLE:
        LOGGER.warning("opencv-python not available. Returning original image.")
        return image_array

    # Apply morphological operations to enhance structure.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Morphological closing to fill small gaps.
    closed = cv2.morphologyEx(image_array, cv2.MORPH_CLOSE, kernel)

    # Edge detection to highlight boundaries.
    edges = cv2.Canny(closed, 50, 150)

    # Combine original mask with edge information.
    enhanced = np.maximum(closed, edges)

    # Apply slight Gaussian blur to create smoother gradients for ViT.
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)

    return enhanced


def extract_multiscale_uniform_features(
    image_path: Path,
    model,
    processor,
    patch_sizes: list,
    stride: int,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    workers: int = 4,
    enhance_binary: bool = False,
    fusion_method: str = "concatenate"
):
    """
    Extract multi-scale ViT features from uniform tiles across the entire image.

    Args:
        image_path: Path to binary mask image.
        model: Pre-loaded ViT model.
        processor: ViT image processor for preprocessing.
        patch_sizes: List of patch sizes to extract features for.
        stride: Stride between tiles in pixels.
        batch_size: Number of tiles per processing batch.
        device: PyTorch device (CPU or CUDA).
        output_dir: Directory to save output files.
        workers: Number of CPU workers for data loading.
        enhance_binary: Apply morphological enhancement to binary masks.
        fusion_method: Method to combine multi-scale features.
    """
    LOGGER.debug(f"Loading image from {image_path}")

    # Load binary mask image.
    image = Image.open(image_path)
    LOGGER.debug(f"Image loaded successfully, size: {image.size}")

    # Apply binary enhancement if requested.
    if enhance_binary:
        LOGGER.debug(f"Applying binary enhancement...")
        image_array = np.array(image.convert('L'))
        enhanced_array = enhance_binary_image(image_array)
        image = Image.fromarray(enhanced_array).convert('RGB')
    else:
        image = image.convert('RGB')

    # Use the largest patch size for grid generation to ensure complete coverage.
    base_patch_size = max(patch_sizes)

    LOGGER.debug(f"Multi-scale extraction with patch sizes: {patch_sizes}")
    LOGGER.debug(f"Base patch size for grid: {base_patch_size}, stride: {stride}")

    # Storage for multi-scale features.
    all_scale_features = {}
    coords_list = []

    # Process each patch size.
    for patch_size in patch_sizes:
        LOGGER.debug(f"Processing patch size {patch_size}...")

        # Create dataset for this patch size.
        dataset = UniformTileDataset(image, patch_size, stride, transform=None)

        # Create preprocessing transform matching ViT requirements.
        transform = transforms.Compose([
            transforms.Resize(processor.size['height']),
            transforms.CenterCrop(processor.size['height']),
            transforms.ToTensor(),
            transforms.Normalize(processor.image_mean, processor.image_std)
        ])

        # Apply transform to dataset.
        dataset.transform = transform

        # Store grid-based coordinates (using stride as the grid cell size) only
        # once. The grid positions are the same for all scales; only the crop
        # extent around each centre changes.
        if not coords_list:
            coords_list = [
                (tile_x * stride + stride // 2, tile_y * stride + stride // 2)
                for tile_y in range((dataset.height + stride - 1) // stride)
                for tile_x in range((dataset.width + stride - 1) // stride)
            ]

        # Adjust batch size for this scale.
        scale_batch_size = min(batch_size, max(32, batch_size // len(patch_sizes)))

        # Create data loader.
        loader = DataLoader(
            dataset,
            batch_size=scale_batch_size,
            shuffle=False,
            num_workers=min(workers, 2),
            pin_memory=device.type == 'cuda' and len(dataset) < 50000,
            persistent_workers=False
        )

        # Move model to device and set to evaluation mode.
        model.to(device)
        model.eval()

        LOGGER.debug(f"Processing {len(dataset)} tiles for patch size {patch_size}")

        # Initialize storage for this scale.
        features_list = []

        # Process tiles in batches.
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Scale {patch_size}px")):
                try:
                    tiles, x_coords, y_coords = batch
                    tiles = tiles.to(device, non_blocking=True)

                    # Extract features using ViT model.
                    outputs = model(pixel_values=tiles)

                    # Use CLS token embeddings as tile features.
                    tile_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    features_list.append(tile_features)

                    # Clear GPU memory periodically.
                    if (batch_idx + 1) % 20 == 0:
                        torch.cuda.empty_cache() if device.type == 'cuda' else None

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        LOGGER.warning(f"GPU memory error at batch {batch_idx + 1}, retrying per-tile...")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()

                        # Retry failed batch one tile at a time to avoid data loss.
                        for i in range(len(tiles)):
                            single_tile = tiles[i:i+1]
                            single_output = model(pixel_values=single_tile)
                            single_feature = single_output.last_hidden_state[:, 0, :].cpu().numpy()
                            features_list.append(single_feature)
                    else:
                        raise e

        # Concatenate features for this scale.
        scale_features = np.vstack(features_list)
        all_scale_features[patch_size] = scale_features

        # Verify tile count matches grid coordinate count.
        if scale_features.shape[0] != len(coords_list):
            raise RuntimeError(
                f"Tile count mismatch at scale {patch_size}px: "
                f"got {scale_features.shape[0]} features but {len(coords_list)} grid positions"
            )

        LOGGER.debug(f"Scale {patch_size}: extracted {scale_features.shape[0]} features of dimension {scale_features.shape[1]}")

    # Combine multi-scale features.
    LOGGER.debug(f"Combining multi-scale features using {fusion_method}")

    if fusion_method == "concatenate":
        # Concatenate features from all scales.
        combined_features = np.hstack([all_scale_features[size] for size in sorted(patch_sizes)])
    elif fusion_method == "average":
        # Average features across scales.
        combined_features = np.mean([all_scale_features[size] for size in patch_sizes], axis=0)
    elif fusion_method == "max":
        # Element-wise maximum across scales.
        combined_features = np.maximum.reduce([all_scale_features[size] for size in patch_sizes])
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")

    LOGGER.debug(f"Combined features shape: {combined_features.shape}")

    # Ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output file names using image stem.
    image_stem = image_path.stem

    # Save individual scale features.
    for patch_size in patch_sizes:
        scale_features_csv = output_dir / f"features_{patch_size}px_{image_stem}.csv"
        scale_features_npy = output_dir / f"features_{patch_size}px_{image_stem}.npy"

        df_scale = pd.DataFrame(all_scale_features[patch_size])
        df_scale.to_csv(scale_features_csv, index=False)
        np.save(scale_features_npy, all_scale_features[patch_size])

        LOGGER.debug(f"Scale {patch_size}px features saved to {scale_features_npy}")

    # Save combined multi-scale features.
    features_csv = output_dir / f"features_{image_stem}.csv"
    features_npy = output_dir / f"features_{image_stem}.npy"
    coords_csv = output_dir / f"coords_{image_stem}.csv"

    # Generate feature column names matching the vit{size}_{i} convention
    # used by filter_features_by_box_size.py and the dynamic patches script.
    feature_columns: list[str] = []
    for size in sorted(patch_sizes):
        dim = all_scale_features[size].shape[1]
        feature_columns.extend([f"vit{size}_{i}" for i in range(dim)])

    # Save combined features in both CSV and NPY formats.
    df_features = pd.DataFrame(combined_features, columns=feature_columns)
    df_features.to_csv(features_csv, index=False)
    np.save(features_npy, combined_features)

    # Save coordinates.
    df_coords = pd.DataFrame(coords_list, columns=["x_center", "y_center"])
    df_coords.to_csv(coords_csv, index=False)

    LOGGER.debug(f"Combined features saved to {features_csv}")
    LOGGER.debug(f"Combined features saved to {features_npy}")
    LOGGER.debug(f"Coordinates saved to {coords_csv}")

    print(f"✓ Multi-scale uniform tiling feature extraction completed successfully.")
    print(f"  • Processed {len(coords_list)} tiles at {len(patch_sizes)} scales")
    print(f"  • Individual scale dimensions: {[all_scale_features[size].shape[1] for size in patch_sizes]}")
    print(f"  • Combined feature dimensionality: {combined_features.shape[1]}")
    print(f"  • Output files saved to: {output_dir}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract ViT features using uniform tiling across binary mask images."
    )

    parser.add_argument(
        "--image", type=Path, required=True,
        help="Path to binary mask image file (TIFF format)."
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for feature and coordinate files."
    )
    parser.add_argument(
        "--patch_sizes", type=int, nargs='+', default=[64],
        help="List of patch sizes in pixels for multi-scale extraction (default: [64])."
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help="Stride between tiles in pixels (default: largest patch_size for non-overlapping)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=512,
        help="Number of tiles to process per batch (default: 512)."
    )
    parser.add_argument(
        "--model_name", type=str, default="facebook/dino-vits16",
        help="Hugging Face ViT model identifier (default: facebook/dino-vits16)."
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of CPU workers for data loading (default: 4)."
    )
    parser.add_argument(
        "--enhance_binary", action="store_true",
        help="Apply morphological enhancement to improve ViT feature quality on binary masks."
    )
    parser.add_argument(
        "--fusion_method", type=str, default="concatenate",
        choices=['concatenate', 'average', 'max'],
        help="Method to combine multi-scale features (default: concatenate)."
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    try:
        # Parse command-line arguments.
        args = parse_arguments()

        # Set default stride to largest patch size if not specified.
        if args.stride is None:
            args.stride = max(args.patch_sizes)

        LOGGER.debug(f"Starting uniform tiling ViT feature extraction")
        LOGGER.debug(f"Input image: {args.image}")
        LOGGER.debug(f"Output directory: {args.output}")
        LOGGER.debug(f"Patch sizes: {args.patch_sizes}, stride: {args.stride}")
        LOGGER.debug(f"Multi-scale fusion: {args.fusion_method}")
        LOGGER.debug(f"Binary enhancement: {args.enhance_binary}")
        LOGGER.debug(f"Batch size: {args.batch_size}")
        LOGGER.debug(f"Model: {args.model_name}")

        # Detect available device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.debug(f"Using device: {device}")

        # Load ViT model and processor.
        LOGGER.debug(f"Loading ViT model and processor...")
        processor = ViTImageProcessor.from_pretrained(args.model_name)
        model = ViTModel.from_pretrained(
            args.model_name,
            output_hidden_states=True,
            use_safetensors=True
        )

        LOGGER.debug(f"Model loaded successfully")

        # Extract features using multi-scale uniform tiling.
        extract_multiscale_uniform_features(
            image_path=args.image,
            model=model,
            processor=processor,
            patch_sizes=args.patch_sizes,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output,
            workers=args.workers,
            enhance_binary=args.enhance_binary,
            fusion_method=args.fusion_method
        )

    except Exception as e:
        print(f"ERROR: Uniform tiling feature extraction failed: {str(e)}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
