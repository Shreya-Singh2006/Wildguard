"""
Unit tests for WildGuard storage, database initialization, and fresh-clone loading.
Ensures fresh environments with non-existent or empty databases never crash with DatabaseError.
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import pytest

from src.store import (
    get_cleaning_audit_from_db,
    get_db_connection,
    get_metadata,
    init_database,
    load_highway_screening,
    load_roadkill,
    load_sightings,
    load_validation_results,
    set_metadata,
)


def test_fresh_database_auto_initialization(tmp_path: Path):
    """Test that a non-existent database file is automatically initialized on connection."""
    test_db = tmp_path / "subdir" / "test_wildguard.db"
    assert not test_db.exists()

    conn = get_db_connection(test_db)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    conn.close()

    assert test_db.exists()
    assert "metadata" in tables
    assert "sightings" in tables
    assert "roadkill" in tables
    assert "highway_screening" in tables
    assert "validation_metrics"
    assert "cleaning_audit" in tables


def test_load_functions_on_empty_database(tmp_path: Path):
    """Test that load_* functions return empty structures without raising DatabaseError."""
    test_db = tmp_path / "fresh.db"

    # Direct query before any data exists
    sightings = load_sightings(test_db)
    assert isinstance(sightings, (gpd.GeoDataFrame, pd.DataFrame))
    assert sightings.empty

    roadkill = load_roadkill(test_db)
    assert isinstance(roadkill, (gpd.GeoDataFrame, pd.DataFrame))
    assert roadkill.empty

    screening = load_highway_screening(test_db)
    assert isinstance(screening, (gpd.GeoDataFrame, pd.DataFrame))
    assert screening.empty

    val = load_validation_results(test_db)
    assert isinstance(val, dict)
    assert len(val) == 0

    meta = get_metadata("non_existent_key", default="fallback", db_path=test_db)
    assert meta == "fallback"

    audit = get_cleaning_audit_from_db("test_ds", db_path=test_db)
    assert isinstance(audit, list)
    assert len(audit) == 0


def test_set_and_get_metadata(tmp_path: Path):
    """Test writing and reading metadata."""
    test_db = tmp_path / "meta_test.db"
    set_metadata("sample_key", {"count": 42, "country": "India"}, db_path=test_db)

    val = get_metadata("sample_key", db_path=test_db)
    assert isinstance(val, dict)
    assert val["count"] == 42
    assert val["country"] == "India"

