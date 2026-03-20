"""Visualization subpackage: overlay masks, cluster circles, and image cropping."""

from vitseg.visualization.overlay_masks import (
    OverlayConfig,
    overlay,
    generate_label_colors,
)
from vitseg.visualization.cluster_circles import (
    generate_high_contrast_colors,
    load_and_filter_data,
    create_cluster_visualization,
    create_legend,
    save_statistics,
)
from vitseg.visualization.crop import crop_image

__all__ = [
    "OverlayConfig",
    "overlay",
    "generate_label_colors",
    "generate_high_contrast_colors",
    "load_and_filter_data",
    "create_cluster_visualization",
    "create_legend",
    "save_statistics",
    "crop_image",
]
