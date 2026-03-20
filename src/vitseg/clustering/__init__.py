"""Clustering subpackage: feature filtering and k-means clustering."""

from vitseg.clustering.filter_features import (
    detect_feature_dimensions,
    filter_features_by_scales,
)

__all__ = [
    "detect_feature_dimensions",
    "filter_features_by_scales",
]
