"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: filter_features_by_box_size.py.
Description:
    Filter multi-scale ViT features CSV by box sizes (16px, 32px, 64px).
    Creates filtered copies of feature files containing only selected box size
    combinations, enabling focused analysis on specific spatial scales.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas.

Usage:
    python filter_features_by_box_size.py \
        --input features_binary_image.csv \
        --output filtered_features \
        --box_sizes 16 32 64
"""

import argparse
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def detect_feature_dimensions(csv_path: Path) -> dict:
    """Detect feature dimensions for each scale in multi-scale CSV.

    Assumes standard ViT-S/16 384-dimensional features per scale.

    Args:
        csv_path (Path): Path to combined features CSV file.

    Returns:
        dict: Dictionary mapping scale sizes to feature column ranges.
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
        print("WARNING: May indicate non-standard feature dimensions or mixed scales.")

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
    """Filter multi-scale features CSV by selected box sizes.

    Args:
        input_csv (Path): Path to combined multi-scale features CSV.
        output_dir (Path): Output directory for filtered files.
        selected_scales (List[int]): List of box sizes to include (e.g., [16, 32]).
        coords_csv (Optional[Path]): Optional coordinates CSV to copy alongside features.
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
    
    # Extract columns for the selected scales.
    selected_columns = []
    for scale in sorted(selected_scales):  # Maintain order.
        start_col, end_col = scale_mapping[scale]
        scale_columns = list(range(start_col, end_col))
        selected_columns.extend(scale_columns)
        print(f"DEBUG: Including {len(scale_columns)} features from {scale}px scale")
    
    # Create the filtered dataframe.
    df_filtered = df_features.iloc[:, selected_columns]
    print(f"DEBUG: Filtered features shape: {df_filtered.shape}")
    
    # Generate the output filename with scale indicators.
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
    """Main entry point for the command-line interface."""
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
        # Validate the input file.
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")

        if not args.input.suffix.lower() == '.csv':
            raise ValueError(f"Input file must be CSV format: {args.input}")

        # Filter features by the selected scales.
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
