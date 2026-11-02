"""
Unit tests for WildGuard hotspot analytics and validation metrics.
Uses tiny synthetic fixtures exclusively.
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

from src.analysis.hotspots import calculate_modified_z_scores
from src.analysis.validation import (
    compute_metrics_bundle,
    compute_iou_with_tolerance,
    bootstrap_confidence_intervals,
)


def test_modified_z_score_with_positive_mad():
    # Counts with variance: median=1, MAD=1
    # 0, 1, 1, 1, 2, 2, 1, 20
    counts = np.array([0, 1, 1, 1, 2, 2, 1, 25])
    mod_z, is_hotspot, stats = calculate_modified_z_scores(counts)

    assert stats["mad"] > 0
    assert stats["used_fallback"] == False
    # The count 25 should be flagged as a hotspot (M_i > 3.5)
    assert is_hotspot[-1] == True
    assert mod_z[-1] > 3.5
    # The normal counts should not be hotspots
    assert is_hotspot[1] == False


def test_modified_z_score_with_zero_mad_fallback():
    # Sparse counts where median=0 and MAD=0
    # Over 50% are zeros
    counts = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 15])
    mod_z, is_hotspot, stats = calculate_modified_z_scores(counts, percentile_fallback=90.0)

    assert stats["mad"] == 0.0
    assert stats["used_fallback"] == True
    assert stats["threshold_rule"].startswith("90.0th Percentile")
    # 15 is >= 90th percentile of non-zero elements [1, 2, 15] (90th pct is 12.4)
    assert is_hotspot[-1] == True
    # Zeros must never be hotspots
    assert np.all(is_hotspot[:8] == False)


def test_compute_metrics_bundle():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.85, 0.3, 0.05, 0.2])

    metrics = compute_metrics_bundle(y_true, y_pred, scores, k_fractions=[0.2, 0.4])

    assert metrics["tp"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 5
    assert metrics["precision"] == 0.75  # 3 / 4
    assert metrics["recall"] == 0.75     # 3 / 4
    assert metrics["f1"] == 0.75
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_compute_iou_with_tolerance():
    # 3 collinear segments spaced by 1 km
    # seg 0: (0, 0) to (0.01, 0)
    # seg 1: (0.01, 0) to (0.02, 0)
    # seg 2: (0.05, 0) to (0.06, 0)
    lines = [
        LineString([(77.0, 23.0), (77.009, 23.0)]),
        LineString([(77.009, 23.0), (77.018, 23.0)]),
        LineString([(77.050, 23.0), (77.059, 23.0)]),
    ]

    pred_gdf = gpd.GeoDataFrame({
        "segment_id": ["s1", "s2", "s3"],
        "is_hotspot": [True, False, False],
        "geometry": lines,
    }, crs="EPSG:4326")

    # True hotspot is segment 2 (adjacent to segment 1, within 1 km)
    test_gdf = gpd.GeoDataFrame({
        "segment_id": ["s1", "s2", "s3"],
        "is_hotspot": [False, True, False],
        "geometry": lines,
    }, crs="EPSG:4326")

    iou = compute_iou_with_tolerance(pred_gdf, test_gdf, tolerance_km=1.5)
    # Since s1 and s2 are adjacent (< 1.5 km), tolerance matching should recognize overlap
    assert iou > 0.0


def test_bootstrap_confidence_intervals():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    scores = np.linspace(0.1, 0.9, len(y_true))

    ci = bootstrap_confidence_intervals(y_true, y_pred, scores, n_iterations=100)

    assert "f1" in ci
    assert "precision" in ci
    assert "recall" in ci
    low, high = ci["f1"]
    assert 0.0 <= low <= high <= 1.0
