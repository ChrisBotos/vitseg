"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: __init__.py.
Description:
    Package initialization for vitseg — Vision Transformer analysis pipeline
    for kidney tissue samples. Contains core modules for multi-scale ViT
    embedding extraction, clustering, and comparison against spatial
    transcriptomics data.

Key Subpackages:
    * preprocessing: Mask filtering and binary conversion.
    * extraction: Dynamic-patch and uniform-tiling ViT feature extraction.
    * clustering: Feature filtering, k-means clustering, and spot-nuclei analysis.
    * visualization: Overlay masks, cluster circle plots, and image cropping.
    * comparison: Cluster metrics, spatial analysis, and visualization suite.
    * utilities: Color generation, configuration, spatial alignment, and helpers.
"""

__version__ = "1.0.0"
__author__ = "Christos Botos"
__email__ = "botoschristos@gmail.com"

__all__ = [
    "preprocessing",
    "extraction",
    "clustering",
    "visualization",
    "comparison",
    "utilities",
]
