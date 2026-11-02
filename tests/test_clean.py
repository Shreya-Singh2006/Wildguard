"""
Unit tests for WildGuard data cleaning and quality tagging rules.
Uses tiny synthetic fixtures exclusively.
"""

import pytest
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd

from src.clean.cleaning import (
    tag_uncertainty,
    clean_sightings_data,
    clean_roadkill_data,
)


def test_tag_uncertainty_precise():
    assert tag_uncertainty(500) == "precise"
    assert tag_uncertainty(1000) == "precise"
    assert tag_uncertainty(0) == "precise"


def test_tag_uncertainty_blurred():
    assert tag_uncertainty(20000) == "blurred"
    assert tag_uncertainty(30000) == "blurred"
    assert tag_uncertainty(50000) == "blurred"


def test_tag_uncertainty_other():
    assert tag_uncertainty(1001) == "other"
    assert tag_uncertainty(5000) == "other"
    assert tag_uncertainty(19999) == "other"


def test_tag_uncertainty_missing():
    assert tag_uncertainty(None) == "missing"
    assert tag_uncertainty(np.nan) == "missing"
    assert tag_uncertainty("invalid") == "missing"


def test_threatened_species_forced_blurring():
    # Tiger should always be tagged as blurred regardless of small reported uncertainty
    assert tag_uncertainty(100, species_name="Panthera tigris") == "blurred"
    assert tag_uncertainty(None, species_name="Panthera pardus") == "blurred"
    assert tag_uncertainty(500, species_name="Bos gaurus") == "blurred"
    assert tag_uncertainty(200, species_name="Melursus ursinus") == "blurred"
    assert tag_uncertainty(100, species_name="Cuon alpinus") == "blurred"
    # Non-threatened species keeps normal precision tag
    assert tag_uncertainty(500, species_name="Macaca mulatta") == "precise"


def test_clean_sightings_pipeline_synthetic():
    synthetic_records = [
        # Valid record 1: precise common mammal
        {
            "key": 1,
            "species": "Macaca mulatta",
            "scientificName": "Macaca mulatta",
            "decimalLatitude": 23.2,
            "decimalLongitude": 77.4,
            "coordinateUncertaintyInMeters": 250,
            "year": 2018,
            "month": 4,
            "day": 12,
            "eventDate": "2018-04-12",
            "basisOfRecord": "HUMAN_OBSERVATION",
            "hasGeospatialIssue": False,
        },
        # Valid record 2: threatened blurred tiger
        {
            "key": 2,
            "species": "Panthera tigris",
            "scientificName": "Panthera tigris",
            "decimalLatitude": 22.3,
            "decimalLongitude": 80.6,
            "coordinateUncertaintyInMeters": 28000,
            "year": 2021,
            "month": 3,
            "day": 5,
            "eventDate": "2021-03-05",
            "basisOfRecord": "HUMAN_OBSERVATION",
            "hasGeospatialIssue": False,
        },
        # Drop 1: Missing coordinates
        {
            "key": 3,
            "species": "Canis aureus",
            "scientificName": "Canis aureus",
            "decimalLatitude": None,
            "decimalLongitude": 78.0,
            "coordinateUncertaintyInMeters": 500,
            "year": 2019,
            "basisOfRecord": "HUMAN_OBSERVATION",
            "hasGeospatialIssue": False,
        },
        # Drop 2: Coordinate at 0, 0
        {
            "key": 4,
            "species": "Canis aureus",
            "scientificName": "Canis aureus",
            "decimalLatitude": 0.0,
            "decimalLongitude": 0.0,
            "coordinateUncertaintyInMeters": 500,
            "year": 2019,
            "basisOfRecord": "HUMAN_OBSERVATION",
            "hasGeospatialIssue": False,
        },
        # Drop 3: Geospatial issue flag
        {
            "key": 5,
            "species": "Axis axis",
            "scientificName": "Axis axis",
            "decimalLatitude": 24.1,
            "decimalLongitude": 79.5,
            "coordinateUncertaintyInMeters": 500,
            "year": 2020,
            "basisOfRecord": "HUMAN_OBSERVATION",
            "hasGeospatialIssue": True,
        },
        # Drop 4: Year before 2000
        {
            "key": 6,
            "species": "Semnopithecus entellus",
            "scientificName": "Semnopithecus entellus",
            "decimalLatitude": 23.5,
            "decimalLongitude": 77.2,
            "coordinateUncertaintyInMeters": 500,
            "year": 1995,
            "basisOfRecord": "HUMAN_OBSERVATION",
            "hasGeospatialIssue": False,
        },
        # Drop 5: Preserved museum specimen
        {
            "key": 7,
            "species": "Herpestes edwardsi",
            "scientificName": "Herpestes edwardsi",
            "decimalLatitude": 23.8,
            "decimalLongitude": 78.1,
            "coordinateUncertaintyInMeters": 500,
            "year": 2015,
            "basisOfRecord": "PRESERVED_SPECIMEN",
            "hasGeospatialIssue": False,
        },
        # Drop 6: Exact duplicate of record 1
        {
            "key": 8,
            "species": "Macaca mulatta",
            "scientificName": "Macaca mulatta",
            "decimalLatitude": 23.2,
            "decimalLongitude": 77.4,
            "coordinateUncertaintyInMeters": 250,
            "year": 2018,
            "month": 4,
            "day": 12,
            "eventDate": "2018-04-12",
            "basisOfRecord": "HUMAN_OBSERVATION",
            "hasGeospatialIssue": False,
        },
    ]

    # Simple bounding box poly covering MP coords
    mp_poly = Polygon([(73.0, 20.0), (84.0, 20.0), (84.0, 28.0), (73.0, 28.0), (73.0, 20.0)])
    mp_boundary = gpd.GeoDataFrame([{"name": "MP", "geometry": mp_poly}], crs="EPSG:4326")

    clean_gdf, audit = clean_sightings_data(synthetic_records, mp_boundary)

    # Out of 8 synthetic records:
    # 2 valid (key 1 and key 2), 6 dropped
    assert len(clean_gdf) == 2
    assert audit["initial_count"] == 8
    assert audit["final_count"] == 2

    # Check uncertainty tagging
    rec1 = clean_gdf[clean_gdf["key"] == 1].iloc[0]
    assert rec1["uncertainty_tag"] == "precise"
    assert rec1["is_threatened_blurred"] == False

    rec2 = clean_gdf[clean_gdf["key"] == 2].iloc[0]
    assert rec2["uncertainty_tag"] == "blurred"
    assert rec2["is_threatened_blurred"] == True
