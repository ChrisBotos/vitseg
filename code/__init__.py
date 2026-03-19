"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: __init__.py.
Description:
    Package initialization for the ViT-on-Segmentation-Masks code module.
    Contains core modules for analyzing kidney tissue samples using Vision
    Transformers and advanced image processing techniques.

Key Modules:
    • segmentation_mask_dynamic_patches_vit: Dynamic patch extraction around nuclei.
    • uniform_tiling_vit: Uniform grid-based ViT feature extraction.
    • cluster_vit_patches_memopt: Memory-efficient clustering of ViT features.
    • cluster_uniform_tiles_memopt: Clustering for uniform tile features.
    • filter_masks: Quality control filtering of segmentation masks.
    • filter_features_by_box_size: Multi-scale feature filtering.
    • overlay_masks: High-quality mask overlay visualization.
    • color_config: Color palette configuration system.
    • generate_contrast_colors: High-contrast color generation.
"""

__version__ = "1.0.0"
__author__ = "Christos Botos"
__email__ = "botoschristos@gmail.com"
