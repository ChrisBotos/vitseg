"""Tests for vigseg.comparison.improved_comparison."""

import numpy as np
import pandas as pd
import pytest

from vigseg.comparison.improved_comparison import (
    ZONE_MAP,
    assign_samples,
    majority_vote_match,
)


class TestZoneMap:
    """Tests for the ZONE_MAP constant."""

    def test_zone_map_has_expected_cell_types(self):
        """All known cell types from figure_idents are present."""
        expected = {
            "PT-S1/S2", "DCT", "CNT", "glomeruli", "PT-S3",
            "TAL", "collecting duct", "interstitial cells",
            "FR-PT", "Injured tubule", "undetermined", "gaps",
        }
        assert set(ZONE_MAP.keys()) == expected

    def test_zone_map_values_are_valid(self):
        """All zone values are one of the expected categories."""
        valid_zones = {"cortex", "outer_medulla", "medulla", "interstitial", "injury", "other"}
        for cell_type, zone in ZONE_MAP.items():
            assert zone in valid_zones, f"{cell_type} maps to unexpected zone: {zone}"

    def test_zone_map_nonempty(self):
        """Zone map is not empty."""
        assert len(ZONE_MAP) > 0


class TestAssignSamples:
    """Tests for the assign_samples function."""

    def test_basic_assignment(self):
        """Points inside bounding boxes get assigned to the correct sample."""
        boundaries = {
            "IRI1": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100,
                      "x_center": 50, "y_center": 50},
            "IRI2": {"x_min": 200, "x_max": 300, "y_min": 200, "y_max": 300,
                      "x_center": 250, "y_center": 250},
        }
        coords = np.array([
            [50, 50],     # Inside IRI1.
            [250, 250],   # Inside IRI2.
        ])

        result = assign_samples(coords, boundaries)
        assert result[0] == "IRI1"
        assert result[1] == "IRI2"

    def test_unassigned_goes_to_nearest_centroid(self):
        """Points outside all bounding boxes are assigned to nearest centroid."""
        boundaries = {
            "IRI1": {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10,
                      "x_center": 5, "y_center": 5},
            "IRI2": {"x_min": 90, "x_max": 100, "y_min": 90, "y_max": 100,
                      "x_center": 95, "y_center": 95},
        }
        # Point at (20, 20) is outside both boxes, closer to IRI1.
        coords = np.array([[20, 20]])
        result = assign_samples(coords, boundaries)
        assert result[0] == "IRI1"

    def test_all_assigned(self):
        """No empty assignments remain after processing."""
        boundaries = {
            "IRI1": {"x_min": 0, "x_max": 50, "y_min": 0, "y_max": 50,
                      "x_center": 25, "y_center": 25},
        }
        coords = np.array([[100, 100], [200, 200], [25, 25]])
        result = assign_samples(coords, boundaries)
        assert all(r != "" for r in result)


class TestMajorityVoteMatch:
    """Tests for the majority_vote_match function."""

    def test_basic_majority_vote(self):
        """Spot gets assigned the most common cluster within radius."""
        vit_coords = np.array([
            [10, 10], [11, 10], [12, 10],  # 3 nuclei near (10, 10).
            [50, 50],                       # 1 nucleus far away.
        ])
        vit_clusters = np.array([0, 0, 1, 2])  # Majority near spot = cluster 0.
        spatial_coords = np.array([[10, 10]])

        matched, n_nuclei = majority_vote_match(
            vit_coords, vit_clusters, spatial_coords, radius=5.0
        )
        assert matched[0] == 0  # Majority vote = cluster 0.
        assert n_nuclei[0] == 3  # 3 nuclei within radius.

    def test_no_match_returns_minus_one(self):
        """Spots with no nuclei within radius get -1."""
        vit_coords = np.array([[100, 100]])
        vit_clusters = np.array([0])
        spatial_coords = np.array([[0, 0]])

        matched, n_nuclei = majority_vote_match(
            vit_coords, vit_clusters, spatial_coords, radius=5.0
        )
        assert matched[0] == -1
        assert n_nuclei[0] == 0

    def test_multiple_spots(self):
        """Multiple spatial spots are matched independently."""
        vit_coords = np.array([
            [10, 10], [11, 10],  # Near spot A.
            [50, 50], [51, 50],  # Near spot B.
        ])
        vit_clusters = np.array([0, 0, 1, 1])
        spatial_coords = np.array([[10, 10], [50, 50]])

        matched, n_nuclei = majority_vote_match(
            vit_coords, vit_clusters, spatial_coords, radius=5.0
        )
        assert matched[0] == 0
        assert matched[1] == 1
        assert n_nuclei[0] == 2
        assert n_nuclei[1] == 2
