"""
Storage and Persistence Layer for WildGuard.
Provides SQLite database tables and Parquet file serialization for:
- Datasets metadata and selection status
- Cleaned MP mammal sightings
- Cleaning audit histories
- Cleaned roadkill records
- 1 km highway segments and screening scores
- Hotspot validation metrics and bootstrap confidence intervals
- Monthly climate normals
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import geopandas as gpd
import pandas as pd

from src.config import PROCESSED_DATA_DIR, SQLITE_DB_PATH

logger = logging.getLogger(__name__)


def get_db_connection(db_path: Path = SQLITE_DB_PATH) -> sqlite3.Connection:
    """Create or return a SQLite connection."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: Path = SQLITE_DB_PATH) -> None:
    """Initialize SQLite database schema for WildGuard."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cleaning_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_name TEXT,
        step_name TEXT,
        record_count INTEGER,
        removed_count INTEGER,
        created_at TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sightings (
        key INTEGER PRIMARY KEY,
        species TEXT,
        scientific_name TEXT,
        vernacular_name TEXT,
        latitude REAL,
        longitude REAL,
        coordinate_uncertainty REAL,
        year INTEGER,
        month INTEGER,
        day INTEGER,
        event_date TEXT,
        uncertainty_tag TEXT,
        is_threatened_blurred INTEGER
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS roadkill (
        key INTEGER PRIMARY KEY,
        species TEXT,
        scientific_name TEXT,
        latitude REAL,
        longitude REAL,
        coordinate_uncertainty REAL,
        year INTEGER,
        uncertainty_tag TEXT,
        country TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS highway_screening (
        segment_id TEXT PRIMARY KEY,
        highway_name TEXT,
        highway_type TEXT,
        ref TEXT,
        length_m REAL,
        pa_proximity_score REAL,
        wildlife_density_score REAL,
        road_vulnerability_score REAL,
        exposure_score REAL,
        risk_category TEXT,
        risk_color TEXT,
        screening_label TEXT,
        survey_rank INTEGER,
        precise_sightings_nearby INTEGER,
        blurred_sightings_nearby INTEGER,
        center_lat REAL,
        center_lon REAL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS validation_metrics (
        model_name TEXT PRIMARY KEY,
        precision REAL,
        recall REAL,
        f1 REAL,
        iou_tolerance_1km REAL,
        roc_auc REAL,
        pr_auc REAL,
        p_at_5 REAL,
        p_at_10 REAL,
        p_at_20 REAL,
        ci_json TEXT,
        split_info_json TEXT,
        updated_at TEXT
    );
    """)

    conn.commit()
    conn.close()
    logger.info("WildGuard SQLite schema initialized.")


def set_metadata(key: str, value: Any, db_path: Path = SQLITE_DB_PATH) -> None:
    """Save or update metadata key-value pair."""
    conn = get_db_connection(db_path)
    val_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    now = datetime.now().isoformat()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value, updated_at) VALUES (?, ?, ?)",
            (key, val_str, now),
        )
    conn.close()


def get_metadata(key: str, default: Any = None, db_path: Path = SQLITE_DB_PATH) -> Any:
    """Retrieve metadata key-value pair."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT value FROM metadata WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return default
    val_str = row["value"]
    try:
        return json.loads(val_str)
    except Exception:
        return val_str


def save_cleaning_audit_to_db(audit_data: Dict[str, Any], db_path: Path = SQLITE_DB_PATH) -> None:
    """Save cleaning audit steps to SQLite."""
    conn = get_db_connection(db_path)
    dataset_name = audit_data.get("dataset", "Unknown")
    now = audit_data.get("processed_at", datetime.now().isoformat())
    with conn:
        conn.execute("DELETE FROM cleaning_audit WHERE dataset_name = ?", (dataset_name,))
        for s in audit_data.get("steps", []):
            conn.execute(
                """INSERT INTO cleaning_audit (dataset_name, step_name, record_count, removed_count, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (dataset_name, s.get("step", ""), s.get("count", 0), s.get("removed", 0), now),
            )
    conn.close()


def get_cleaning_audit_from_db(dataset_name: str, db_path: Path = SQLITE_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve cleaning audit steps from SQLite."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT step_name, record_count, removed_count FROM cleaning_audit WHERE dataset_name = ? ORDER BY id ASC",
        (dataset_name,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def save_sightings(gdf: gpd.GeoDataFrame, db_path: Path = SQLITE_DB_PATH) -> None:
    """Save cleaned sightings to SQLite and Parquet."""
    if gdf.empty:
        return

    # Parquet
    parquet_path = PROCESSED_DATA_DIR / "sightings.parquet"
    gdf.to_parquet(parquet_path, index=False)

    # SQLite
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("DELETE FROM sightings")
        records = []
        for _, row in gdf.iterrows():
            records.append((
                int(row.get("key", 0)),
                str(row.get("species", "")),
                str(row.get("scientificName", "")),
                str(row.get("vernacularName", "")),
                float(row.get("decimalLatitude", 0.0)),
                float(row.get("decimalLongitude", 0.0)),
                float(row.get("coordinateUncertaintyInMeters", 0.0)) if pd.notna(row.get("coordinateUncertaintyInMeters")) else None,
                int(row.get("year", 0)) if pd.notna(row.get("year")) else 0,
                int(row.get("month", 0)) if pd.notna(row.get("month")) else None,
                int(row.get("day", 0)) if pd.notna(row.get("day")) else None,
                str(row.get("eventDate", "")),
                str(row.get("uncertainty_tag", "other")),
                1 if row.get("is_threatened_blurred", False) else 0,
            ))
        conn.executemany(
            """INSERT OR REPLACE INTO sightings
               (key, species, scientific_name, vernacular_name, latitude, longitude,
                coordinate_uncertainty, year, month, day, event_date, uncertainty_tag, is_threatened_blurred)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            records,
        )
    conn.close()
    set_metadata("sightings_count", len(gdf))
    set_metadata("sightings_saved_at", datetime.now().isoformat())
    logger.info(f"Saved {len(gdf)} sightings to SQLite and Parquet.")


def load_sightings(db_path: Path = SQLITE_DB_PATH) -> gpd.GeoDataFrame:
    """Load cleaned sightings from Parquet if available, else SQLite."""
    parquet_path = PROCESSED_DATA_DIR / "sightings.parquet"
    if parquet_path.exists():
        try:
            return gpd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning(f"Error reading sightings parquet: {e}")

    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM sightings", conn)
    conn.close()
    if df.empty:
        return gpd.GeoDataFrame()

    geometry = gpd.points_from_xy(df["longitude"], df["latitude"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf = gdf.rename(columns={
        "latitude": "decimalLatitude",
        "longitude": "decimalLongitude",
        "scientific_name": "scientificName",
        "vernacular_name": "vernacularName",
        "coordinate_uncertainty": "coordinateUncertaintyInMeters",
        "event_date": "eventDate",
    })
    gdf["is_threatened_blurred"] = gdf["is_threatened_blurred"].astype(bool)
    return gdf


def save_roadkill(gdf: gpd.GeoDataFrame, country: str = "Taiwan", db_path: Path = SQLITE_DB_PATH) -> None:
    """Save cleaned roadkill records to SQLite and Parquet."""
    if gdf.empty:
        return

    parquet_path = PROCESSED_DATA_DIR / "roadkill.parquet"
    gdf.to_parquet(parquet_path, index=False)

    conn = get_db_connection(db_path)
    with conn:
        conn.execute("DELETE FROM roadkill")
        records = []
        for _, row in gdf.iterrows():
            records.append((
                int(row.get("key", 0)),
                str(row.get("species", "")),
                str(row.get("scientificName", "")),
                float(row.get("decimalLatitude", 0.0)),
                float(row.get("decimalLongitude", 0.0)),
                float(row.get("coordinateUncertaintyInMeters", 0.0)) if pd.notna(row.get("coordinateUncertaintyInMeters")) else None,
                int(row.get("year", 0)) if pd.notna(row.get("year")) else 0,
                str(row.get("uncertainty_tag", "precise")),
                country,
            ))
        conn.executemany(
            """INSERT OR REPLACE INTO roadkill
               (key, species, scientific_name, latitude, longitude,
                coordinate_uncertainty, year, uncertainty_tag, country)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            records,
        )
    conn.close()
    set_metadata("roadkill_count", len(gdf))
    set_metadata("roadkill_saved_at", datetime.now().isoformat())
    logger.info(f"Saved {len(gdf)} roadkill records to SQLite and Parquet.")


def load_roadkill(db_path: Path = SQLITE_DB_PATH) -> gpd.GeoDataFrame:
    """Load cleaned roadkill records from Parquet if available, else SQLite."""
    parquet_path = PROCESSED_DATA_DIR / "roadkill.parquet"
    if parquet_path.exists():
        try:
            return gpd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning(f"Error reading roadkill parquet: {e}")

    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM roadkill", conn)
    conn.close()
    if df.empty:
        return gpd.GeoDataFrame()

    geometry = gpd.points_from_xy(df["longitude"], df["latitude"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf = gdf.rename(columns={
        "latitude": "decimalLatitude",
        "longitude": "decimalLongitude",
        "scientific_name": "scientificName",
        "coordinate_uncertainty": "coordinateUncertaintyInMeters",
    })
    return gdf


def save_highway_screening(gdf: gpd.GeoDataFrame, db_path: Path = SQLITE_DB_PATH) -> None:
    """Save ranked highway screening segments to SQLite and Parquet."""
    if gdf.empty:
        return

    parquet_path = PROCESSED_DATA_DIR / "screening_segments.parquet"
    gdf.to_parquet(parquet_path, index=False)

    conn = get_db_connection(db_path)
    with conn:
        conn.execute("DELETE FROM highway_screening")
        records = []
        centroids = gdf.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
        for idx, row in gdf.iterrows():
            c = centroids.iloc[idx]
            records.append((
                str(row.get("segment_id", "")),
                str(row.get("highway_name", "Unnamed Highway")),
                str(row.get("highway_type", "primary")),
                str(row.get("ref", "")),
                float(row.get("length_m", 1000.0)),
                float(row.get("pa_proximity_score", 0.0)),
                float(row.get("wildlife_density_score", 0.0)),
                float(row.get("road_vulnerability_score", 0.0)),
                float(row.get("exposure_score", 0.0)),
                str(row.get("risk_category", "Low Risk")),
                str(row.get("risk_color", "#2a9d8f")),
                str(row.get("screening_label", "Predicted exposure, not observed roadkill")),
                int(row.get("survey_rank", 0)),
                int(row.get("precise_sightings_nearby", 0)),
                int(row.get("blurred_sightings_nearby", 0)),
                float(c.y),
                float(c.x),
            ))
        conn.executemany(
            """INSERT OR REPLACE INTO highway_screening
               (segment_id, highway_name, highway_type, ref, length_m,
                pa_proximity_score, wildlife_density_score, road_vulnerability_score,
                exposure_score, risk_category, risk_color, screening_label, survey_rank,
                precise_sightings_nearby, blurred_sightings_nearby, center_lat, center_lon)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            records,
        )
    conn.close()
    set_metadata("screening_segments_count", len(gdf))
    set_metadata("screening_saved_at", datetime.now().isoformat())
    logger.info(f"Saved {len(gdf)} highway screening segments to SQLite and Parquet.")


def load_highway_screening(db_path: Path = SQLITE_DB_PATH) -> gpd.GeoDataFrame:
    """Load ranked highway screening segments."""
    parquet_path = PROCESSED_DATA_DIR / "screening_segments.parquet"
    if parquet_path.exists():
        try:
            return gpd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning(f"Error reading screening segments parquet: {e}")

    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM highway_screening ORDER BY survey_rank ASC", conn)
    conn.close()
    if df.empty:
        return gpd.GeoDataFrame()

    geometry = gpd.points_from_xy(df["center_lon"], df["center_lat"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def save_validation_results(validation_dict: Dict[str, Any], db_path: Path = SQLITE_DB_PATH) -> None:
    """Save validation results and baselines to SQLite and JSON."""
    json_path = PROCESSED_DATA_DIR / "validation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(validation_dict, f, indent=2)

    conn = get_db_connection(db_path)
    split_json = json.dumps(validation_dict.get("split_info", {}))
    ci_json = json.dumps(validation_dict.get("confidence_intervals_95", {}))
    now = datetime.now().isoformat()

    models = [
        ("Outlier Hotspot Model (Primary)", validation_dict.get("model_metrics", {})),
        ("Random Ranking Baseline", validation_dict.get("baselines", {}).get("random_ranking", {})),
        ("Density-Only Baseline", validation_dict.get("baselines", {}).get("density_only", {})),
    ]

    with conn:
        for m_name, m in models:
            if not m:
                continue
            conn.execute(
                """INSERT OR REPLACE INTO validation_metrics
                   (model_name, precision, recall, f1, iou_tolerance_1km, roc_auc, pr_auc,
                    p_at_5, p_at_10, p_at_20, ci_json, split_info_json, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    m_name,
                    m.get("precision", 0.0),
                    m.get("recall", 0.0),
                    m.get("f1", 0.0),
                    m.get("iou_tolerance_1km", 0.0),
                    m.get("roc_auc", 0.5),
                    m.get("pr_auc", 0.0),
                    m.get("p_at_5", 0.0),
                    m.get("p_at_10", 0.0),
                    m.get("p_at_20", 0.0),
                    ci_json if "Primary" in m_name else "",
                    split_json,
                    now,
                ),
            )
    conn.close()
    logger.info("Saved validation results to SQLite and JSON.")


def load_validation_results(db_path: Path = SQLITE_DB_PATH) -> Dict[str, Any]:
    """Load validation results."""
    json_path = PROCESSED_DATA_DIR / "validation_results.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading validation results json: {e}")

    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM validation_metrics", conn)
    conn.close()
    if df.empty:
        return {}
    return {"models_df": df}
