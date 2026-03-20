"""Comparison subpackage: cluster metrics and spatial analysis."""

from vitseg.comparison.cluster_metrics import (
    load_and_align_data,
    align_coordinates,
    calculate_adjusted_rand_index,
    calculate_normalized_mutual_information,
    calculate_silhouette_analysis,
    create_confusion_matrix_analysis,
    calculate_effect_sizes,
)
from vitseg.comparison.spatial_analysis import (
    create_spatial_weights,
    calculate_morans_i,
    calculate_local_morans_i,
    calculate_getis_ord_g,
    analyze_spatial_clustering_by_sample,
)

__all__ = [
    "load_and_align_data",
    "align_coordinates",
    "calculate_adjusted_rand_index",
    "calculate_normalized_mutual_information",
    "calculate_silhouette_analysis",
    "create_confusion_matrix_analysis",
    "calculate_effect_sizes",
    "create_spatial_weights",
    "calculate_morans_i",
    "calculate_local_morans_i",
    "calculate_getis_ord_g",
    "analyze_spatial_clustering_by_sample",
]
