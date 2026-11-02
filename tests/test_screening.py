"""
Unit tests for Highway Exposure Screening module.
Uses tiny synthetic fixtures exclusively.
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point

from src.analysis.screening import calculate_exposure_scores, PREDICTED_EXPOSURE_LABEL


def test_exposure_scoring_synthetic():
    # 2 synthetic highway segments:
    # Seg 1: Close to Kanha (~22.33, 80.61), motorway
    # Seg 2: Far from reserves (~25.0, 75.0), secondary
    seg1 = LineString([(80.60, 22.30), (80.61, 22.31)])
    seg2 = LineString([(75.00, 25.00), (75.01, 25.01)])

    segments_gdf = gpd.GeoDataFrame({
        "segment_id": ["seg_near", "seg_far"],
        "highway_name": ["NH-44 Express", "Rural Link"],
        "highway_type": ["motorway", "secondary"],
        "ref": ["NH-44", "SH-12"],
        "geometry": [seg1, seg2],
    }, crs="EPSG:4326")

    # Synthetic sightings near seg1
    sight_pts = [
        Point(80.605, 22.305),
        Point(80.608, 22.308),
    ]
    sightings_gdf = gpd.GeoDataFrame({
        "key": [101, 102],
        "species": ["Panthera tigris", "Macaca mulatta"],
        "decimalLatitude": [22.305, 22.308],
        "decimalLongitude": [80.605, 80.608],
        "uncertainty_tag": ["blurred", "precise"],
        "geometry": sight_pts,
    }, crs="EPSG:4326")

    scored_gdf = calculate_exposure_scores(segments_gdf, sightings_gdf)

    assert len(scored_gdf) == 2
    # Seg near must rank higher than seg far
    assert scored_gdf.iloc[0]["segment_id"] == "seg_near"
    assert scored_gdf.iloc[0]["exposure_score"] > scored_gdf.iloc[1]["exposure_score"]

    # Check mandatory label
    assert scored_gdf.iloc[0]["screening_label"] == PREDICTED_EXPOSURE_LABEL
    assert scored_gdf.iloc[1]["screening_label"] == PREDICTED_EXPOSURE_LABEL

    # Check risk categories
    assert scored_gdf.iloc[0]["risk_category"] in ["High Risk", "Moderate Risk"]
    assert scored_gdf.iloc[0]["risk_color"].startswith("#")
    assert scored_gdf.iloc[0]["survey_rank"] == 1
    assert scored_gdf.iloc[1]["survey_rank"] == 2


def test_exposure_scores_with_empty_sightings():
    seg = LineString([(80.0, 22.0), (80.01, 22.01)])
    segments_gdf = gpd.GeoDataFrame({
        "segment_id": ["s1"],
        "highway_name": ["Test Road"],
        "highway_type": ["primary"],
        "ref": ["NH-1"],
        "geometry": [seg],
    }, crs="EPSG:4326")

    scored = calculate_exposure_scores(segments_gdf, gpd.GeoDataFrame())
    assert len(scored) == 1
    assert "exposure_score" in scored.columns
    assert 0 <= scored.iloc[0]["exposure_score"] <= 100

