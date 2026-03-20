"""Comparison subpackage: cluster metrics, spatial analysis, and improved comparison."""

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


def __getattr__(name):
    """Lazy import for improved_comparison to avoid RuntimeWarning when run as __main__."""
    _lazy = {"run_comparison", "majority_vote_match", "ZONE_MAP"}
    if name in _lazy:
        from vitseg.comparison import improved_comparison
        return getattr(improved_comparison, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "run_comparison",
    "majority_vote_match",
    "ZONE_MAP",
]
