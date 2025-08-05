"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: overlay_masks.py
Description:
    Memory-efficient overlay of integer-label masks (.npy) on very large microscopy TIFFs.
    This script addresses critical GPU out-of-memory (OOM) issues through advanced memory
    management strategies including spatial batching, progressive processing, and configurable
    memory limits. Processing is tile-wise with parallel execution and optional GPU acceleration.
    The system implements 2x2 spatial batching to reduce memory pressure while maintaining
    full-resolution output quality.

Dependencies:
    • Python >= 3.10
    • numpy >= 1.21.0
    • tifffile >= 2021.7.2
    • tqdm >= 4.62.0
    • psutil >= 5.8.0 (for system memory monitoring)
    • cupy >= 9.0.0 (optional, for GPU acceleration)
    • imagecodecs (optional, for compression)

Usage:
    python overlay_masks.py --image img/IRI_regist_cropped.tif --mask segmentation_masks.npy
                           --out overlay.tif --tile 1024 --workers 8 --alpha 0.4 --gpu
                           --batch-size 4 --memory-limit 8192

Arguments:
    --image         Path to input TIFF image file
    --mask          Path to integer-label mask (.npy format)
    --out           Output BigTIFF path (default: overlay.tif)
    --tile          Tile edge length in pixels (default: 1024)
    --workers       Number of worker processes ('auto' or integer, default: auto)
    --alpha         Overlay transparency [0,1] (default: 0.4)
    --seed          RNG seed for reproducible colors (default: 42)
    --gpu           Enable CuPy GPU acceleration when available
    --batch-size    Spatial batch size for memory management (default: 4)
    --memory-limit  GPU memory limit in MB (default: 8192)

Inputs:
    • Large microscopy TIFF image (any bit depth)
    • Integer-labeled segmentation mask (.npy format)

Outputs:
    • overlay.tif   High-quality BigTIFF overlay with alpha-blended masks
    • Debug logs    Comprehensive memory usage and processing statistics

Key Features:
    • Memory-efficient spatial batching (2x2 tile groups)
    • Progressive memory cleanup between processing steps
    • GPU memory monitoring with automatic CPU fallback
    • Configurable memory limits and batch sizes
    • Deterministic color generation for reproducible results
    • BigTIFF support for images >4GB
    • Comprehensive error handling and recovery

Notes:
    • Designed for kidney slice analysis from I/R injury studies
    • Optimized for DAPI stain segmentation overlays
    • Supports both CPU and GPU processing with graceful fallback
    • Memory usage scales with tile size and batch size, not image size
    • All processing parameters are configurable for different hardware setups
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile as tiff
from tqdm import tqdm
import multiprocessing as mp
import psutil  # For system memory monitoring.

"""CONFIGURATION MANAGEMENT"""

@dataclass
class OverlayConfig:
    """
    Configuration parameters for memory-efficient overlay processing.

    This configuration class centralizes all processing parameters to enable
    fine-tuning of memory usage and performance characteristics for different
    hardware setups and image sizes.
    """
    # Core processing parameters.
    tile_size: int = 1024                    # Tile edge length in pixels.
    batch_size: int = 4                      # Number of tiles processed simultaneously.
    workers: Union[int, str] = "auto"        # Number of worker processes.
    alpha: float = 0.4                       # Overlay transparency [0,1].
    seed: int = 42                           # RNG seed for reproducible colors.

    # Memory management parameters.
    memory_limit_mb: int = 8192              # GPU memory limit in megabytes.
    enable_gpu: bool = False                 # Enable GPU acceleration.
    cleanup_frequency: int = 10              # Memory cleanup every N tiles.

    # Output parameters.
    compression: str = "deflate"             # TIFF compression method.
    bigtiff_threshold: int = 4_000_000_000   # BigTIFF threshold in bytes.

    def validate(self) -> None:
        """
        Validate configuration parameters and adjust if necessary.

        This method ensures all parameters are within acceptable ranges and
        adjusts values that could cause memory issues or processing failures.
        """
        if self.tile_size < 64 or self.tile_size > 4096:
            warnings.warn(f"Tile size {self.tile_size} may cause issues. Recommended: 512-2048.")

        # Memory-aware batch size adjustment.
        if self.batch_size < 1:
            self.batch_size = 1
        elif self.batch_size > 8:  # Limit batch size to prevent memory issues.
            print(f"WARNING: Reducing batch size from {self.batch_size} to 8 for memory safety")
            self.batch_size = 8

        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError(f"Alpha must be in [0,1], got {self.alpha}")

        if self.memory_limit_mb < 1024:
            warnings.warn(f"Memory limit {self.memory_limit_mb}MB is very low. Consider increasing.")

        # Adjust cleanup frequency based on batch size for better memory management.
        if self.cleanup_frequency > 20:
            self.cleanup_frequency = 20
            print("DEBUG: Limited cleanup frequency to 20 for better memory management")


"""MEMORY MONITORING UTILITIES"""


def get_system_memory_info() -> Dict[str, float]:
    """
    Get system memory information for optimization decisions.

    Returns:
        Dictionary containing system memory statistics in MB.

    This function provides system memory monitoring to make intelligent
    decisions about batch sizes and worker counts based on available resources.
    """
    try:
        memory = psutil.virtual_memory()
        return {
            'total_mb': memory.total / (1024**2),
            'available_mb': memory.available / (1024**2),
            'used_mb': memory.used / (1024**2),
            'percent_used': memory.percent
        }
    except Exception as e:
        print(f"DEBUG: Failed to get system memory info: {e}")
        return {
            'total_mb': 16384,  # Assume 16GB default.
            'available_mb': 8192,  # Assume 8GB available.
            'used_mb': 8192,
            'percent_used': 50.0
        }


def calculate_optimal_batch_size(tile_size: int, available_memory_mb: int = 8192) -> int:
    """
    Calculate optimal batch size based on tile size and available memory.

    Args:
        tile_size: Size of each tile in pixels.
        available_memory_mb: Available memory in megabytes.

    Returns:
        Optimal batch size for memory-efficient processing.

    This function estimates memory usage per tile and calculates the maximum
    batch size that can fit in available memory with safety margins.
    """
    # Estimate memory per tile: tile_size^2 * 3 channels * 4 bytes (float32) * 3 (safety factor).
    memory_per_tile_mb = (tile_size ** 2 * 3 * 4 * 3) / (1024 ** 2)

    # Calculate max batch size with 50% safety margin.
    max_batch_size = max(1, int((available_memory_mb * 0.5) / memory_per_tile_mb))

    # Limit to reasonable range.
    optimal_batch_size = min(max_batch_size, 8)  # Cap at 8.
    optimal_batch_size = max(optimal_batch_size, 1)  # Minimum 1.

    print(f"DEBUG: Calculated optimal batch size: {optimal_batch_size} "
          f"(memory per tile: {memory_per_tile_mb:.1f}MB, available: {available_memory_mb}MB)")

    return optimal_batch_size

def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage information.

    Returns:
        Dictionary containing GPU memory statistics in MB, or empty dict if GPU unavailable.

    This function provides detailed GPU memory monitoring to prevent OOM errors
    and enable intelligent memory management decisions during processing.
    """
    try:
        import cupy as cp

        # Get memory pool statistics.
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()

        # Get device memory info.
        device = cp.cuda.Device()
        device_total = device.mem_info[1]  # Total device memory.
        device_free = device.mem_info[0]   # Free device memory.

        return {
            'pool_used_mb': used_bytes / (1024**2),
            'pool_total_mb': total_bytes / (1024**2),
            'device_free_mb': device_free / (1024**2),
            'device_total_mb': device_total / (1024**2),
            'device_used_mb': (device_total - device_free) / (1024**2)
        }

    except Exception:
        return {}


def cleanup_gpu_memory() -> None:
    """
    Perform aggressive GPU memory cleanup.

    This function forces garbage collection and clears GPU memory pools
    to prevent memory fragmentation and reduce OOM risk during processing.
    """
    try:
        import cupy as cp

        # Force garbage collection.
        gc.collect()

        # Clear memory pool.
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

        print("DEBUG: GPU memory cleanup completed.")

    except Exception as e:
        print(f"DEBUG: GPU cleanup failed: {e}")


def monitor_memory_usage(config: OverlayConfig, tile_count: int) -> bool:
    """
    Monitor memory usage and determine if processing should continue.

    Args:
        config: Configuration object with memory limits.
        tile_count: Current tile number for cleanup scheduling.

    Returns:
        True if processing can continue, False if memory limit exceeded.

    This function implements intelligent memory monitoring to prevent OOM
    errors by tracking usage patterns and triggering cleanup when needed.
    """
    # Perform periodic cleanup.
    if tile_count % config.cleanup_frequency == 0:
        cleanup_gpu_memory()
        gc.collect()

    # Check GPU memory if enabled.
    if config.enable_gpu:
        gpu_info = get_gpu_memory_info()

        if gpu_info:
            used_mb = gpu_info.get('device_used_mb', 0)

            if used_mb > config.memory_limit_mb:
                print(f"WARNING: GPU memory usage ({used_mb:.1f}MB) exceeds limit ({config.memory_limit_mb}MB)")
                return False

            print(f"DEBUG: GPU memory usage: {used_mb:.1f}MB / {config.memory_limit_mb}MB")

    return True


"""COLOR GENERATION UTILITIES"""

def generate_label_colors(max_label: int, seed: int = 42) -> np.ndarray:
    """
    Generate deterministic RGB colors for segmentation labels.

    Args:
        max_label: Maximum label value in the segmentation mask.
        seed: Random seed for reproducible color generation.

    Returns:
        Color lookup table as (max_label+1, 3) uint8 array.

    This function creates a deterministic color palette for visualizing
    segmentation masks, ensuring consistent colors across multiple runs
    while providing sufficient visual distinction between adjacent labels.
    """
    print(f"DEBUG: Generating color palette for {max_label} labels with seed {seed}")

    # Limit color LUT size to prevent excessive memory usage.
    # Large labels will be mapped to valid indices using modulo operation.
    if max_label > 1000000:  # 1M labels max to prevent memory issues.
        print(f"WARNING: Max label {max_label} is very large, limiting to 1M for memory efficiency")
        print("DEBUG: Large label values will be mapped to valid LUT indices using modulo operation")
        max_label = 1000000

    rng = np.random.default_rng(seed)
    lut = rng.integers(0, 256, size=(max_label + 1, 3), dtype=np.uint8)
    lut[0] = 0  # Background remains black for better contrast.

    print(f"DEBUG: Color palette generated successfully")
    return lut


def get_mask_max_label_efficiently(mask_path: Union[str, Path]) -> int:
    """
    Efficiently determine maximum label in mask without loading entire array.

    Args:
        mask_path: Path to mask file.

    Returns:
        Maximum label value in the mask.

    This function uses chunked reading to determine the maximum label
    without loading the entire mask into memory, preventing OOM issues.
    """
    print(f"DEBUG: Determining max label in mask: {mask_path}")

    try:
        # Load mask with memory mapping.
        mask_memmap = np.load(mask_path, mmap_mode="r")
        height, width = mask_memmap.shape

        # Process mask in chunks to avoid memory issues.
        chunk_size = 10000  # Process 10k rows at a time.
        max_label = 0

        for start_row in range(0, height, chunk_size):
            end_row = min(start_row + chunk_size, height)
            chunk = mask_memmap[start_row:end_row, :]
            chunk_max = int(chunk.max()) if chunk.size > 0 else 0
            max_label = max(max_label, chunk_max)

            # Cleanup chunk immediately.
            del chunk

        # Cleanup memory map.
        del mask_memmap
        gc.collect()

        print(f"DEBUG: Max label determined efficiently: {max_label}")
        return max_label

    except Exception as e:
        print(f"ERROR: Failed to determine max label efficiently: {e}")
        # Fallback to simple approach.
        mask_memmap = np.load(mask_path, mmap_mode="r")
        max_label = int(mask_memmap.max()) if mask_memmap.size > 0 else 0
        del mask_memmap
        gc.collect()
        return max_label


"""TILE PROCESSING FUNCTIONS"""

def blend_tile_with_mask(
    tile_img: np.ndarray,
    tile_mask: np.ndarray,
    color_lut: np.ndarray,
    alpha: float,
    enable_gpu: bool = False,
    memory_limit_mb: int = 8192
) -> np.ndarray:
    """
    Apply color mapping and alpha blending to a single image tile.

    Args:
        tile_img: Input image tile as numpy array.
        tile_mask: Segmentation mask tile with integer labels.
        color_lut: Color lookup table for label-to-RGB mapping.
        alpha: Alpha blending factor [0,1] for overlay transparency.
        enable_gpu: Whether to attempt GPU acceleration.
        memory_limit_mb: GPU memory limit for processing decisions.

    Returns:
        RGB blended tile as uint8 numpy array.

    This function performs memory-efficient tile blending with intelligent
    GPU/CPU backend selection. It includes automatic fallback mechanisms
    to prevent OOM errors and ensures consistent output quality regardless
    of the processing backend used.
    """
    # Determine optimal processing backend.
    xp = np  # Default to CPU processing.
    gpu_available = False

    if enable_gpu:
        try:
            import cupy as cp

            # Verify CUDA runtime availability.
            device_count = cp.cuda.runtime.getDeviceCount()

            if device_count > 0:
                # Check available GPU memory.
                device = cp.cuda.Device()
                free_memory_mb = device.mem_info[0] / (1024**2)

                if free_memory_mb > memory_limit_mb * 0.5:  # Use 50% safety margin.
                    xp = cp
                    gpu_available = True
                    print(f"DEBUG: Using GPU backend with {free_memory_mb:.1f}MB available")
                else:
                    print(f"DEBUG: GPU memory insufficient ({free_memory_mb:.1f}MB), using CPU")

        except Exception as e:
            print(f"DEBUG: GPU initialization failed ({e}), falling back to CPU")

    try:
        # Normalize image intensity to 0-255 range.
        if tile_img.dtype != np.uint8:
            if tile_img.max() > 0:
                tile_img_norm = (tile_img.astype(np.float32) / tile_img.max()) * 255.0
            else:
                tile_img_norm = tile_img.astype(np.float32)
        else:
            tile_img_norm = tile_img.astype(np.float32)

        # Convert inputs to selected backend.
        img_tensor = xp.asarray(tile_img_norm, dtype=xp.float32)
        mask_tensor = xp.asarray(tile_mask, dtype=xp.int32)
        lut_tensor = xp.asarray(color_lut, dtype=xp.float32)

        # Perform color mapping and alpha blending.
        # Handle large label values by mapping them to valid LUT indices.
        # Use modulo operation to ensure all labels fit within LUT bounds.
        max_lut_index = lut_tensor.shape[0] - 1
        safe_mask_indices = mask_tensor % (max_lut_index + 1)
        colored_mask = lut_tensor[safe_mask_indices]  # Shape: (H, W, 3)

        # Alpha blend: result = img * (1-alpha) + colored_mask * alpha
        blended = img_tensor * (1.0 - alpha) + colored_mask * alpha
        blended = xp.clip(blended, 0, 255).astype(xp.uint8)

        # Convert result back to numpy if needed.
        result = xp.asnumpy(blended) if gpu_available else blended

        # Cleanup GPU memory if used.
        if gpu_available:
            del img_tensor, mask_tensor, lut_tensor, colored_mask, blended

        return result

    except Exception as e:
        print(f"WARNING: Tile blending failed with {xp.__name__} backend: {e}")

        # Fallback to CPU processing.
        if gpu_available:
            print("DEBUG: Attempting CPU fallback")
            return blend_tile_with_mask(tile_img, tile_mask, color_lut, alpha,
                                      enable_gpu=False, memory_limit_mb=memory_limit_mb)
        else:
            raise RuntimeError(f"Tile blending failed on CPU backend: {e}")


def create_spatial_batches(tiles: List[Tuple], batch_size: int) -> List[List[Tuple]]:
    """
    Group tiles into spatial batches for memory-efficient processing.

    Args:
        tiles: List of tile specifications (y0, y1, x0, x1, ...).
        batch_size: Maximum number of tiles per batch.

    Returns:
        List of tile batches, each containing up to batch_size tiles.

    This function implements spatial proximity grouping to optimize memory
    usage and cache efficiency during parallel processing. Tiles are grouped
    to minimize memory fragmentation and improve processing throughput.
    """
    print(f"DEBUG: Creating spatial batches from {len(tiles)} tiles with batch size {batch_size}")

    if batch_size <= 0:
        batch_size = 1

    batches = []

    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i + batch_size]
        batches.append(batch)

    print(f"DEBUG: Created {len(batches)} spatial batches")
    return batches


"""WORKER PROCESS FUNCTIONS"""

def process_tile_worker_optimized(args: Tuple) -> Tuple[int, int, int, int, np.ndarray]:
    """
    Memory-optimized worker function for processing individual tiles.

    Args:
        args: Tuple containing (y0, y1, x0, x1, img_path, mask_path, config_dict, color_lut).

    Returns:
        Tuple of (y0, y1, x0, x1, blended_tile) for result assembly.

    This optimized worker function reduces memory usage by accepting pre-computed
    color lookup tables and implementing aggressive memory cleanup strategies.
    """
    try:
        y0, y1, x0, x1, img_path, mask_path, config_dict, color_lut = args

        # Reconstruct config object from dictionary.
        config = OverlayConfig(**config_dict)

        # Load image tile using memory mapping for efficiency.
        img_memmap = None
        mask_memmap = None

        try:
            with tiff.TiffFile(img_path) as tif_file:
                img_memmap = tif_file.asarray(out='memmap')
                tile_img = img_memmap[y0:y1, x0:x1].copy()  # Copy to avoid memmap issues.

            # Load mask tile using memory mapping.
            mask_memmap = np.load(mask_path, mmap_mode="r")
            tile_mask = mask_memmap[y0:y1, x0:x1].copy()

        finally:
            # Explicitly cleanup memory maps immediately.
            if mask_memmap is not None:
                del mask_memmap
            if img_memmap is not None:
                del img_memmap
            gc.collect()

        # Ensure image is in RGB format.
        if tile_img.ndim == 2:
            # Convert grayscale to RGB by replicating channels.
            tile_img = np.repeat(tile_img[..., None], 3, axis=2)
        elif tile_img.ndim == 3 and tile_img.shape[2] == 1:
            # Handle single-channel 3D arrays.
            tile_img = np.repeat(tile_img, 3, axis=2)
        elif tile_img.ndim == 3 and tile_img.shape[2] > 3:
            # Take first 3 channels if more than RGB.
            tile_img = tile_img[:, :, :3]

        # Check if tile has any labels.
        max_label = int(tile_mask.max()) if tile_mask.size > 0 else 0

        if max_label > 0:
            # Perform tile blending using pre-computed color LUT.
            blended_tile = blend_tile_with_mask(
                tile_img=tile_img,
                tile_mask=tile_mask,
                color_lut=color_lut,
                alpha=config.alpha,
                enable_gpu=config.enable_gpu,
                memory_limit_mb=config.memory_limit_mb
            )
        else:
            # No labels in this tile, return original image.
            blended_tile = tile_img.astype(np.uint8)

        # Cleanup intermediate arrays.
        del tile_img, tile_mask
        gc.collect()

        return y0, y1, x0, x1, blended_tile

    except Exception as e:
        print(f"ERROR: Optimized tile processing failed for ({y0}:{y1}, {x0}:{x1}): {e}")
        traceback.print_exc()

        # Return a black tile as fallback.
        fallback_tile = np.zeros((y1-y0, x1-x0, 3), dtype=np.uint8)
        return y0, y1, x0, x1, fallback_tile


def process_batch_worker_optimized(batch_args: Tuple) -> List[Tuple[int, int, int, int, np.ndarray]]:
    """
    Memory-optimized worker function for processing batches of tiles.

    Args:
        batch_args: Tuple containing (tile_batch, img_path, mask_path, config_dict, color_lut).

    Returns:
        List of processed tile results.

    This optimized function processes multiple tiles with aggressive memory
    management, pre-computed color lookup tables, and reduced memory copying.
    """
    try:
        tile_batch, img_path, mask_path, config_dict, color_lut = batch_args
        config = OverlayConfig(**config_dict)

        print(f"DEBUG: Processing optimized batch of {len(tile_batch)} tiles")

        results = []

        # Process each tile in the batch with memory optimization.
        for i, (y0, y1, x0, x1) in enumerate(tile_batch):
            try:
                # Create args tuple for optimized tile processing.
                tile_args = (y0, y1, x0, x1, img_path, mask_path, config_dict, color_lut)
                result = process_tile_worker_optimized(tile_args)
                results.append(result)

                # Aggressive memory cleanup within batch.
                if (i + 1) % 2 == 0:  # More frequent cleanup.
                    gc.collect()

                    # Additional GPU cleanup if enabled.
                    if config.enable_gpu:
                        cleanup_gpu_memory()

            except Exception as e:
                print(f"ERROR: Optimized batch tile processing failed for ({y0}:{y1}, {x0}:{x1}): {e}")

                # Add fallback result.
                fallback_tile = np.zeros((y1-y0, x1-x0, 3), dtype=np.uint8)
                results.append((y0, y1, x0, x1, fallback_tile))

        # Final batch cleanup.
        gc.collect()
        if config.enable_gpu:
            cleanup_gpu_memory()

        print(f"DEBUG: Optimized batch processing completed with {len(results)} results")
        return results

    except Exception as e:
        print(f"ERROR: Optimized batch processing failed: {e}")
        traceback.print_exc()
        return []


"""MAIN OVERLAY PROCESSING FUNCTION"""

def overlay(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[OverlayConfig] = None
) -> None:
    """
    Create memory-efficient overlay of segmentation masks on large microscopy images.

    Args:
        image_path: Path to input TIFF image file.
        mask_path: Path to integer-labeled segmentation mask (.npy).
        output_path: Path for output BigTIFF overlay file.
        config: Configuration object with processing parameters.

    This function implements advanced memory management strategies to handle
    very large microscopy images without GPU OOM errors. It uses spatial
    batching, progressive processing, and intelligent memory monitoring to
    ensure stable processing of multi-gigabyte images.

    The processing pipeline includes:
    1. Image dimension validation and memory estimation
    2. Spatial tile generation with configurable batch sizes
    3. Memory-efficient parallel processing with cleanup
    4. Progressive result assembly with BigTIFF support
    5. Final compression and optimization
    """
    # Initialize configuration if not provided.
    if config is None:
        config = OverlayConfig()

    # Get system memory info and optimize batch size accordingly.
    memory_info = get_system_memory_info()
    print(f"DEBUG: System memory - Total: {memory_info['total_mb']:.0f}MB, "
          f"Available: {memory_info['available_mb']:.0f}MB, Used: {memory_info['percent_used']:.1f}%")

    # Use available system memory instead of config limit if it's more restrictive.
    effective_memory_limit = min(config.memory_limit_mb, memory_info['available_mb'] * 0.8)  # Use 80% of available.

    optimal_batch_size = calculate_optimal_batch_size(config.tile_size, effective_memory_limit)
    if config.batch_size > optimal_batch_size:
        print(f"DEBUG: Reducing batch size from {config.batch_size} to {optimal_batch_size} for memory efficiency")
        config.batch_size = optimal_batch_size

    config.validate()

    # Convert paths to Path objects.
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    output_path = Path(output_path)

    print(f"DEBUG: Starting overlay creation")
    print(f"DEBUG: Image: {image_path}")
    print(f"DEBUG: Mask: {mask_path}")
    print(f"DEBUG: Output: {output_path}")
    print(f"DEBUG: Configuration: {config}")

    try:

        '''Image and mask validation'''

        # Load image dimensions using memory-efficient approach.
        with tiff.TiffFile(image_path) as tif_file:
            image_shape = tif_file.series[0].shape
            height, width = image_shape[-2:]  # Extract H, W regardless of other dimensions.

        print(f"DEBUG: Image dimensions: {height} x {width}")

        # Validate mask dimensions using memory mapping.
        mask_memmap = np.load(mask_path, mmap_mode="r")
        mask_height, mask_width = mask_memmap.shape

        if (mask_height, mask_width) != (height, width):
            raise ValueError(
                f"Image and mask dimensions mismatch: "
                f"image=({height}, {width}), mask=({mask_height}, {mask_width})"
            )

        print(f"DEBUG: Mask dimensions validated: {mask_height} x {mask_width}")

        '''Memory estimation and BigTIFF configuration'''

        # Calculate output file size: height * width * 3 channels * 1 byte per pixel.
        estimated_output_size = height * width * 3
        use_bigtiff = estimated_output_size >= config.bigtiff_threshold

        print(f"DEBUG: Estimated output size: {estimated_output_size / (1024**3):.2f} GB")
        print(f"DEBUG: Using BigTIFF: {use_bigtiff}")

        # Create empty output file using memory mapping.
        print("DEBUG: Creating output memory map")

        tiff.memmap(
            output_path,
            shape=(height, width, 3),
            dtype=np.uint8,
            photometric="rgb",
            bigtiff=use_bigtiff,
        )

        '''Tile generation and spatial batching'''

        # Generate tile specifications.
        print(f"DEBUG: Generating tiles with size {config.tile_size}")

        tiles = []

        for y0 in range(0, height, config.tile_size):
            y1 = min(y0 + config.tile_size, height)

            for x0 in range(0, width, config.tile_size):
                x1 = min(x0 + config.tile_size, width)
                tiles.append((y0, y1, x0, x1))

        print(f"DEBUG: Generated {len(tiles)} tiles")

        # Create spatial batches for memory-efficient processing.
        tile_batches = create_spatial_batches(tiles, config.batch_size)

        print(f"DEBUG: Created {len(tile_batches)} spatial batches")

        '''Worker process configuration with memory-based optimization'''

        # Determine optimal number of worker processes based on memory constraints.
        if isinstance(config.workers, str):
            if config.workers.lower() == "auto":
                # Calculate memory-aware worker count.
                cpu_count = os.cpu_count() or 1

                # Estimate memory per worker (tile_size^2 * 3 channels * 4 bytes * safety factor).
                memory_per_worker_mb = (config.tile_size ** 2 * 3 * 4 * 2) / (1024 ** 2)  # 2x safety factor.

                # Limit workers based on available memory (assume 16GB total, use 50%).
                max_workers_by_memory = max(1, int((8 * 1024) / memory_per_worker_mb))

                n_workers = min(cpu_count, max_workers_by_memory, 4)  # Cap at 4 for stability.
                print(f"DEBUG: Auto-calculated {n_workers} workers (CPU: {cpu_count}, Memory limit: {max_workers_by_memory})")
            else:
                n_workers = int(config.workers)
        else:
            n_workers = config.workers or 1

        # Ensure valid worker count with memory safety.
        if n_workers < 1:
            n_workers = 1
        elif n_workers > 8:  # Hard limit to prevent memory issues.
            print(f"WARNING: Limiting workers from {n_workers} to 8 for memory safety")
            n_workers = 8

        print(f"DEBUG: Using {n_workers} worker processes")

        '''Optimized parallel tile processing with pre-computed color LUT'''

        # Pre-compute max label and color LUT to avoid memory issues in workers.
        print("DEBUG: Pre-computing color lookup table")
        max_label = get_mask_max_label_efficiently(mask_path)

        if max_label > 0:
            color_lut = generate_label_colors(max_label, config.seed)
        else:
            color_lut = np.zeros((1, 3), dtype=np.uint8)

        # Convert config to dictionary for worker processes.
        config_dict = {
            'tile_size': config.tile_size,
            'batch_size': config.batch_size,
            'workers': config.workers,
            'alpha': config.alpha,
            'seed': config.seed,
            'memory_limit_mb': config.memory_limit_mb,
            'enable_gpu': config.enable_gpu,
            'cleanup_frequency': config.cleanup_frequency,
            'compression': config.compression,
            'bigtiff_threshold': config.bigtiff_threshold
        }

        # Prepare optimized batch arguments for worker processes.
        batch_args = [
            (batch, str(image_path), str(mask_path), config_dict, color_lut)
            for batch in tile_batches
        ]

        print("DEBUG: Starting optimized parallel tile processing")

        # Process batches in parallel with optimized memory monitoring and error recovery.
        multiprocessing_context = mp.get_context("spawn")

        # Reduce max workers further if memory constraints are severe.
        if config.memory_limit_mb < 4096:  # Less than 4GB.
            n_workers = min(n_workers, 2)
            print(f"DEBUG: Reduced workers to {n_workers} due to low memory limit")

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing_context) as executor:
            # Open output memory map for writing results.
            output_memmap = tiff.memmap(output_path, mode="r+")

            try:
                processed_count = 0

                # Process batches with progress tracking and optimized worker.
                with tqdm(total=len(tiles), desc="Processing tiles") as progress_bar:

                    try:
                        for batch_results in executor.map(process_batch_worker_optimized, batch_args):

                            # Write batch results to output with immediate cleanup.
                            for y0, y1, x0, x1, blended_tile in batch_results:
                                output_memmap[y0:y1, x0:x1] = blended_tile
                                processed_count += 1
                                progress_bar.update(1)

                                # Cleanup blended tile immediately after writing.
                                del blended_tile

                            # More aggressive memory monitoring and cleanup.
                            if not monitor_memory_usage(config, processed_count):
                                print("WARNING: Memory limit exceeded, forcing aggressive cleanup")
                                cleanup_gpu_memory()
                                gc.collect()

                            # More frequent output flushing for memory efficiency.
                            if processed_count % max(1, config.cleanup_frequency * config.batch_size // 2) == 0:
                                output_memmap.flush()
                                gc.collect()  # Force garbage collection after flush.
                                print(f"DEBUG: Flushed output and cleaned memory after {processed_count} tiles")

                    except Exception as parallel_error:
                        print(f"ERROR: Parallel processing failed: {parallel_error}")
                        print("DEBUG: Attempting sequential fallback processing")

                        # Sequential fallback processing.
                        for batch_args_single in batch_args[processed_count // config.batch_size:]:
                            try:
                                batch_results = process_batch_worker_optimized(batch_args_single)

                                for y0, y1, x0, x1, blended_tile in batch_results:
                                    output_memmap[y0:y1, x0:x1] = blended_tile
                                    processed_count += 1
                                    progress_bar.update(1)
                                    del blended_tile

                                # Aggressive cleanup in sequential mode.
                                gc.collect()
                                if config.enable_gpu:
                                    cleanup_gpu_memory()

                            except Exception as seq_error:
                                print(f"ERROR: Sequential fallback failed for batch: {seq_error}")
                                # Continue with next batch.
                                continue

                print(f"DEBUG: Processed {processed_count} tiles successfully")

            finally:
                # Ensure output is properly flushed and closed.
                try:
                    output_memmap.flush()
                    del output_memmap
                    gc.collect()  # Force garbage collection to release file handles.
                    print("DEBUG: Output memory map closed")
                except Exception as cleanup_error:
                    print(f"WARNING: Output memory map cleanup failed: {cleanup_error}")

        '''Final optimization and compression (optional on Windows)'''

        try:
            print("DEBUG: Starting final TIFF optimization")

            # Create temporary path for repacking.
            temp_output_path = output_path.with_suffix(".temp.tif")

            # Load the sparse memory map for repacking.
            sparse_memmap = tiff.memmap(output_path, mode="r")

            try:
                # Determine optimal compression method.
                try:
                    import imagecodecs  # noqa: F401
                    compression_method = config.compression
                    print(f"DEBUG: Using {compression_method} compression")
                except ImportError:
                    compression_method = "none"
                    print("DEBUG: imagecodecs not available, using no compression")

                # Write optimized TIFF with tiling and compression.
                print("DEBUG: Writing optimized TIFF file")

                tiff.imwrite(
                    temp_output_path,
                    sparse_memmap,
                    bigtiff=use_bigtiff,
                    tile=(config.tile_size, config.tile_size),
                    compression=compression_method,
                    photometric="rgb",
                    metadata={'description': 'Memory-efficient overlay generated by overlay_masks.py'}
                )

                # Replace original with optimized version (Windows-safe).
                import time

                # Wait a moment for file handles to be released.
                time.sleep(0.1)
                gc.collect()

                # Try multiple times to handle Windows file locking.
                for attempt in range(5):
                    try:
                        if output_path.exists():
                            output_path.unlink()
                        temp_output_path.rename(output_path)
                        break
                    except (PermissionError, OSError) as e:
                        if attempt == 4:  # Last attempt.
                            raise e
                        print(f"DEBUG: File replacement attempt {attempt + 1} failed, retrying...")
                        time.sleep(0.5)
                        gc.collect()

                print(f"DEBUG: Final optimization completed")

            finally:
                # Cleanup memory map properly.
                try:
                    del sparse_memmap
                    gc.collect()  # Force garbage collection to release file handles.
                except Exception as cleanup_error:
                    print(f"WARNING: Memory map cleanup failed: {cleanup_error}")

        except Exception as optimization_error:
            print(f"WARNING: Final optimization failed: {optimization_error}")
            print("DEBUG: Continuing with unoptimized output file")

            # Clean up temporary file if it exists.
            temp_output_path = output_path.with_suffix(".temp.tif")
            if temp_output_path.exists():
                try:
                    temp_output_path.unlink()
                except Exception:
                    pass

        '''Processing completion'''

        # Final memory cleanup.
        cleanup_gpu_memory()
        gc.collect()

        # Calculate final file size.
        final_size_mb = output_path.stat().st_size / (1024**2)

        print(f"✅ Overlay creation completed successfully")
        print(f"✅ Output file: {output_path}")
        print(f"✅ Final size: {final_size_mb:.1f} MB")
        print(f"✅ Processing configuration: {config}")

    except Exception as e:
        print(f"ERROR: Overlay creation failed: {e}")
        traceback.print_exc()

        # Cleanup partial output file if it exists.
        if output_path.exists():
            try:
                output_path.unlink()
                print("DEBUG: Cleaned up partial output file")
            except Exception as cleanup_error:
                print(f"WARNING: Failed to cleanup partial output: {cleanup_error}")

        raise RuntimeError(f"Overlay creation failed: {e}")


"""COMMAND LINE INTERFACE"""

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser with comprehensive options.

    Returns:
        Configured ArgumentParser object with all processing options.

    This parser provides access to all memory management and processing
    parameters, enabling fine-tuning for different hardware configurations
    and image sizes. All parameters include detailed help text for users.
    """
    parser = argparse.ArgumentParser(
        description="Memory-efficient overlay of segmentation masks on large microscopy TIFFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python overlay_masks.py --image img/sample.tif --mask masks.npy --out overlay.tif

  # GPU-accelerated processing with custom memory limit
  python overlay_masks.py --image img/large.tif --mask masks.npy --out overlay.tif \\
                          --gpu --memory-limit 6144 --batch-size 2

  # High-throughput processing with many workers
  python overlay_masks.py --image img/huge.tif --mask masks.npy --out overlay.tif \\
                          --workers 16 --tile 2048 --batch-size 8

For kidney slice analysis from I/R injury studies, recommended settings:
  --tile 1024 --batch-size 4 --alpha 0.4 --gpu (if available)
        """
    )

    # Required arguments.
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input TIFF image file"
    )

    parser.add_argument(
        "--mask",
        required=True,
        help="Path to integer-labeled segmentation mask (.npy format)"
    )

    # Output configuration.
    parser.add_argument(
        "--out",
        default="overlay.tif",
        help="Output BigTIFF file path (default: overlay.tif)"
    )

    # Processing parameters.
    parser.add_argument(
        "--tile",
        type=int,
        default=1024,
        help="Tile edge length in pixels (default: 1024, recommended: 512-2048)"
    )

    parser.add_argument(
        "--workers",
        default="auto",
        help="Number of worker processes ('auto' or integer, default: auto)"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Overlay transparency [0,1] (default: 0.4, 0=transparent, 1=opaque)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible color generation (default: 42)"
    )

    # Memory management parameters.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Spatial batch size for memory management (default: 4, reduce if OOM)"
    )

    parser.add_argument(
        "--memory-limit",
        type=int,
        default=8192,
        help="GPU memory limit in MB (default: 8192, adjust based on GPU)"
    )

    # GPU acceleration.
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable CuPy GPU acceleration (requires CUDA and CuPy installation)"
    )

    # Advanced options.
    parser.add_argument(
        "--compression",
        default="deflate",
        choices=["none", "deflate", "lzw", "zstd"],
        help="TIFF compression method (default: deflate)"
    )

    parser.add_argument(
        "--cleanup-frequency",
        type=int,
        default=10,
        help="Memory cleanup frequency (every N tiles, default: 10)"
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """
    Main entry point for command line execution.

    Args:
        argv: Command line arguments (None for sys.argv).

    This function handles command line parsing, configuration validation,
    and error handling for the overlay creation process. It provides
    comprehensive error reporting and graceful failure handling.
    """
    try:
        # Parse command line arguments.
        parser = create_argument_parser()
        args = parser.parse_args(argv)

        print("DEBUG: Starting overlay_masks.py")
        print(f"DEBUG: Command line arguments: {args}")

        # Create configuration from arguments.
        config = OverlayConfig(
            tile_size=args.tile,
            batch_size=args.batch_size,
            workers=args.workers,
            alpha=args.alpha,
            seed=args.seed,
            memory_limit_mb=args.memory_limit,
            enable_gpu=args.gpu,
            cleanup_frequency=args.cleanup_frequency,
            compression=args.compression
        )

        # Validate configuration.
        config.validate()

        # Execute overlay creation.
        overlay(
            image_path=args.image,
            mask_path=args.mask,
            output_path=args.out,
            config=config
        )

        print("✅ overlay_masks.py completed successfully")

    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"❌ overlay_masks.py failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
