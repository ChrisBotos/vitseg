"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: filter_features_by_box_size.py.
Description:
    Filter multi-scale ViT features CSV by box sizes (16px, 32px, 64px).
    Creates filtered copies of feature files containing only selected box size
    combinations, enabling focused analysis on specific spatial scales.

    Key features for bioinformatician users:
        • **Multi-scale feature filtering** – Selectively extract features from
          specific box sizes (16px, 32px, 64px) from combined multi-scale CSV files.
        • **Flexible scale combinations** – Choose any combination of the three
          available scales to create custom feature sets for targeted analysis.
        • **Automatic feature dimension detection** – Intelligently identifies
          feature boundaries between different scales in concatenated feature vectors.
        • **Consistent file naming** – Maintains original naming conventions with
          clear scale indicators for easy identification and pipeline integration.
        • **Memory-efficient processing** – Handles large feature files without
          loading entire datasets into memory simultaneously.

    Scientific context:
        Different box sizes capture distinct biological information:
        • 16px patches – Fine cellular details and nuclear morphology.
        • 32px patches – Local tissue patterns and cell neighborhoods.
        • 64px patches – Broader tissue architecture and regional organization.
        Filtering allows focused analysis on specific spatial scales relevant
        to particular research questions in kidney injury and repair studies.

Dependencies:
    • Python>=3.10.
    • numpy, pandas for data processing.
    • pathlib for file handling.

Usage:
    # Extract only 16px features.
    python filter_features_by_box_size.py \
        --input features_binary_image.csv \
        --output filtered_features \
        --box_sizes 16

    # Extract 32px and 64px features.
    python filter_features_by_box_size.py \
        --input features_binary_image.csv \
        --output filtered_features \
        --box_sizes 32 64

    # Extract all three scales (equivalent to original file).
    python filter_features_by_box_size.py \
        --input features_binary_image.csv \
        --output filtered_features \
        --box_sizes 16 32 64

Arguments:
    --input          Path to combined multi-scale features CSV file.
    --output         Output directory for filtered feature files.
    --box_sizes      Space-separated list of box sizes to include (16, 32, 64).
    --coords         Optional path to coordinates CSV file to copy alongside.

Inputs:
    • Combined multi-scale features CSV with concatenated embeddings.
    • Optional coordinates CSV file for spatial information.

Outputs:
    • filtered_features_<scales>_<image_stem>.csv    Filtered feature embeddings.
    • coords_<image_stem>.csv                        Copied coordinates (if provided).

Key Features:
    • Selective scale extraction from multi-scale feature vectors.
    • Automatic detection of feature dimensions per scale.
    • Preservation of spatial coordinate information.
    • Clear output file naming with scale indicators.

Notes:
    • Assumes standard 384-dimensional features per scale from ViT-S/16 model.
    • Combined features are expected to be concatenated in order: 16px, 32px, 64px.
    • Output files maintain row order and indexing from original CSV.
    • Scale information is embedded in output filenames for easy identification.
    • Compatible with existing clustering and analysis pipeline infrastructure.
"""

import argparse
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def detect_feature_dimensions(csv_path: Path) -> dict:
    """
    Detect feature dimensions for each scale in multi-scale CSV.
    
    Assumes standard ViT-S/16 384-dimensional features per scale.
    
    Parameters:
        csv_path: Path to combined features CSV file.
        
    Returns:
        Dictionary mapping scale sizes to feature column ranges.
    """
    print(f"DEBUG: Analyzing feature dimensions in {csv_path}")
    
    # Read first row to determine total feature count.
    df_sample = pd.read_csv(csv_path, nrows=1)
    total_features = len(df_sample.columns)
    
    print(f"DEBUG: Total feature columns detected: {total_features}")
    
    # Standard ViT-S/16 produces 384-dimensional features.
    features_per_scale = 384
    num_scales = total_features // features_per_scale
    
    if total_features % features_per_scale != 0:
        print(f"WARNING: Total features ({total_features}) not divisible by {features_per_scale}")
        print("WARNING: May indicate non-standard feature dimensions or mixed scales")
    
    print(f"DEBUG: Detected {num_scales} scales with {features_per_scale} features each")
    
    # Map scales to column ranges assuming order: 16px, 32px, 64px.
    scale_mapping = {}
    available_scales = [16, 32, 64][:num_scales]
    
    for i, scale in enumerate(available_scales):
        start_col = i * features_per_scale
        end_col = (i + 1) * features_per_scale
        scale_mapping[scale] = (start_col, end_col)
        print(f"DEBUG: Scale {scale}px mapped to columns {start_col}-{end_col-1}")
    
    return scale_mapping


def filter_features_by_scales(input_csv: Path, output_dir: Path, 
                             selected_scales: List[int], coords_csv: Optional[Path] = None):
    """
    Filter multi-scale features CSV by selected box sizes.
    
    Parameters:
        input_csv: Path to combined multi-scale features CSV.
        output_dir: Output directory for filtered files.
        selected_scales: List of box sizes to include (e.g., [16, 32]).
        coords_csv: Optional coordinates CSV to copy alongside features.
    """
    print(f"DEBUG: Starting feature filtering for scales: {selected_scales}")
    print(f"DEBUG: Input file: {input_csv}")
    print(f"DEBUG: Output directory: {output_dir}")
    
    # Ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect feature dimensions for each scale.
    scale_mapping = detect_feature_dimensions(input_csv)
    available_scales = list(scale_mapping.keys())
    
    # Validate selected scales.
    invalid_scales = [s for s in selected_scales if s not in available_scales]
    if invalid_scales:
        raise ValueError(f"Invalid scales requested: {invalid_scales}. "
                        f"Available scales: {available_scales}")
    
    print(f"DEBUG: Loading features from {input_csv}")
    
    # Load the full features CSV.
    df_features = pd.read_csv(input_csv)
    total_rows = len(df_features)
    print(f"DEBUG: Loaded {total_rows} feature vectors")
    
    # Extract columns for selected scales.
    selected_columns = []
    for scale in sorted(selected_scales):  # Maintain order.
        start_col, end_col = scale_mapping[scale]
        scale_columns = list(range(start_col, end_col))
        selected_columns.extend(scale_columns)
        print(f"DEBUG: Including {len(scale_columns)} features from {scale}px scale")
    
    # Create filtered dataframe.
    df_filtered = df_features.iloc[:, selected_columns]
    print(f"DEBUG: Filtered features shape: {df_filtered.shape}")
    
    # Generate output filename with scale indicators.
    input_stem = input_csv.stem
    scales_str = "_".join(map(str, sorted(selected_scales)))
    output_filename = f"filtered_features_{scales_str}px_{input_stem}.csv"
    output_path = output_dir / output_filename
    
    # Save filtered features.
    df_filtered.to_csv(output_path, index=False)
    print(f"DEBUG: Filtered features saved to {output_path}")
    
    # Copy coordinates file if provided.
    if coords_csv and coords_csv.exists():
        coords_output = output_dir / f"coords_{input_stem}.csv"
        df_coords = pd.read_csv(coords_csv)
        df_coords.to_csv(coords_output, index=False)
        print(f"DEBUG: Coordinates copied to {coords_output}")
    elif coords_csv:
        print(f"WARNING: Coordinates file not found: {coords_csv}")
    
    print(f"✓ Feature filtering completed successfully.")
    print(f"  • Input features: {df_features.shape}")
    print(f"  • Selected scales: {selected_scales}")
    print(f"  • Output features: {df_filtered.shape}")
    print(f"  • Output file: {output_path}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Filter multi-scale ViT features by box sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input", 
        type=Path, 
        required=True,
        help="Path to combined multi-scale features CSV file"
    )
    
    parser.add_argument(
        "--output", 
        type=Path, 
        required=True,
        help="Output directory for filtered feature files"
    )
    
    parser.add_argument(
        "--box_sizes", 
        type=int, 
        nargs="+", 
        choices=[16, 32, 64],
        required=True,
        help="Box sizes to include in filtered output (16, 32, 64)"
    )
    
    parser.add_argument(
        "--coords", 
        type=Path, 
        help="Optional coordinates CSV file to copy alongside features"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input file.
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        if not args.input.suffix.lower() == '.csv':
            raise ValueError(f"Input file must be CSV format: {args.input}")
        
        # Filter features by selected scales.
        filter_features_by_scales(
            input_csv=args.input,
            output_dir=args.output,
            selected_scales=args.box_sizes,
            coords_csv=args.coords
        )
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
