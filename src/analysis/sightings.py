"""
Wildlife Sightings Analysis for Madhya Pradesh.
Implements:
1. Grid cell aggregation (0.25-degree cells ~25-30 km, matching blurred species scale).
2. District-level aggregation using MP administrative polygons.
3. Strict separation of precise sightings vs deliberately blurred threatened species (~30 km).
   - Never plotted as points.
   - Never counted together with precise records.
   - Represented as wide grid cells.
4. Safari route and tourist seasonality bias analysis (peak April, monsoon dip July-September).
NOTE: DBSCAN is strictly prohibited on sightings per product specification.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from src.config import MP_CONFIG, THREATENED_BLURRED_SPECIES

logger = logging.getLogger(__name__)


_GRID_CACHE = {}


def generate_mp_grid(
    cell_size_deg: float = 0.25,
    bbox: Optional[Dict[str, float]] = None,
) -> gpd.GeoDataFrame:
    """
    Generate regular spatial grid cells across Madhya Pradesh.
    Cached in memory for instantaneous reuse.
    """
    if bbox is None:
        bbox = MP_CONFIG["bbox"]

    cache_key = (cell_size_deg, bbox["min_lon"], bbox["max_lon"], bbox["min_lat"], bbox["max_lat"])
    if cache_key in _GRID_CACHE:
        return _GRID_CACHE[cache_key].copy()

    min_lon = bbox["min_lon"]
    max_lon = bbox["max_lon"]
    min_lat = bbox["min_lat"]
    max_lat = bbox["max_lat"]

    lon_steps = np.arange(min_lon, max_lon, cell_size_deg)
    lat_steps = np.arange(min_lat, max_lat, cell_size_deg)

    cells = []
    cell_id = 0
    for lon in lon_steps:
        for lat in lat_steps:
            cell_geom = box(lon, lat, lon + cell_size_deg, lat + cell_size_deg)
            cells.append({
                "cell_id": f"cell_{cell_id:04d}",
                "min_lon": lon,
                "max_lon": lon + cell_size_deg,
                "min_lat": lat,
                "max_lat": lat + cell_size_deg,
                "center_lon": lon + cell_size_deg / 2.0,
                "center_lat": lat + cell_size_deg / 2.0,
                "geometry": cell_geom,
            })
            cell_id += 1

    gdf = gpd.GeoDataFrame(cells, crs="EPSG:4326")
    _GRID_CACHE[cache_key] = gdf
    return gdf.copy()


def aggregate_sightings_by_grid(
    sightings_gdf: gpd.GeoDataFrame,
    grid_gdf: Optional[gpd.GeoDataFrame] = None,
    cell_size_deg: float = 0.25,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Aggregate sightings into grid cells with O(1) vectorized math for ultra-fast UI rendering.
    Strictly splits:
    1. Precise sightings grid (records with uncertainty <= 1,000m)
    2. Blurred threatened species grid (wide cells, never mixed into precise counts)
    Returns: (precise_grid_gdf, blurred_grid_gdf)
    """
    # Handle if cell_size_deg was passed as 2nd positional argument
    if isinstance(grid_gdf, (int, float)):
        cell_size_deg = float(grid_gdf)
        grid_gdf = None

    if grid_gdf is None or grid_gdf.empty:
        grid_gdf = generate_mp_grid(cell_size_deg=cell_size_deg)

    if sightings_gdf.empty:
        g1 = grid_gdf.copy()
        g1["precise_count"] = 0
        g2 = grid_gdf.copy()
        g2["blurred_count"] = 0
        return g1, g2

    bbox = MP_CONFIG["bbox"]
    min_lon = bbox["min_lon"]
    max_lon = bbox["max_lon"]
    min_lat = bbox["min_lat"]
    max_lat = bbox["max_lat"]

    lon_steps = np.arange(min_lon, max_lon, cell_size_deg)
    lat_steps = np.arange(min_lat, max_lat, cell_size_deg)
    n_lats = len(lat_steps)
    total_cells = len(grid_gdf)

    def fast_cell_assign(df_pts: pd.DataFrame) -> pd.Series:
        lons = df_pts["decimalLongitude"].values
        lats = df_pts["decimalLatitude"].values
        c = np.floor((lons - min_lon) / cell_size_deg).astype(int)
        r = np.floor((lats - min_lat) / cell_size_deg).astype(int)
        valid = (c >= 0) & (c < len(lon_steps)) & (r >= 0) & (r < n_lats)
        indices = np.where(valid, c * n_lats + r, -1)
        valid_indices = indices[indices >= 0]
        # Format as cell_XXXX
        cell_labels = [f"cell_{idx:04d}" for idx in valid_indices if idx < total_cells]
        return pd.Series(cell_labels).value_counts()

    # Separate precise vs blurred
    precise_mask = (sightings_gdf["uncertainty_tag"] == "precise") & (~sightings_gdf["is_threatened_blurred"])
    blurred_mask = (sightings_gdf["uncertainty_tag"] == "blurred") | (sightings_gdf["is_threatened_blurred"])

    precise_pts = sightings_gdf[precise_mask]
    blurred_pts = sightings_gdf[blurred_mask]

    precise_grid = grid_gdf.copy()
    if not precise_pts.empty:
        p_df = fast_cell_assign(precise_pts).rename_axis("cell_id").reset_index(name="precise_count")
        precise_grid = precise_grid.merge(p_df, on="cell_id", how="left")
        precise_grid["precise_count"] = precise_grid["precise_count"].fillna(0).astype(int)
    else:
        precise_grid["precise_count"] = 0

    blurred_grid = grid_gdf.copy()
    if not blurred_pts.empty:
        b_df = fast_cell_assign(blurred_pts).rename_axis("cell_id").reset_index(name="blurred_count")
        blurred_grid = blurred_grid.merge(b_df, on="cell_id", how="left")
        blurred_grid["blurred_count"] = blurred_grid["blurred_count"].fillna(0).astype(int)
    else:
        blurred_grid["blurred_count"] = 0

    return precise_grid, blurred_grid


def aggregate_sightings_by_district(
    sightings_gdf: gpd.GeoDataFrame,
    districts_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Aggregate sightings by administrative district in MP.
    Separates precise vs blurred counts per district.
    """
    if districts_gdf.empty:
        return districts_gdf

    result_gdf = districts_gdf.copy()
    if sightings_gdf.empty:
        result_gdf["total_sightings"] = 0
        result_gdf["precise_sightings"] = 0
        result_gdf["blurred_sightings"] = 0
        return result_gdf

    # Spatial join
    sj = gpd.sjoin(sightings_gdf, districts_gdf, predicate="within")

    if not sj.empty:
        district_col = "district" if "district" in sj.columns else districts_gdf.columns[0]
        
        counts = sj.groupby(district_col).agg(
            total_sightings=("key", "count"),
            precise_sightings=("uncertainty_tag", lambda s: (s == "precise").sum()),
            blurred_sightings=("uncertainty_tag", lambda s: (s == "blurred").sum()),
        ).reset_index()

        result_gdf = result_gdf.merge(counts, on=district_col, how="left")
        result_gdf["total_sightings"] = result_gdf["total_sightings"].fillna(0).astype(int)
        result_gdf["precise_sightings"] = result_gdf["precise_sightings"].fillna(0).astype(int)
        result_gdf["blurred_sightings"] = result_gdf["blurred_sightings"].fillna(0).astype(int)
    else:
        result_gdf["total_sightings"] = 0
        result_gdf["precise_sightings"] = 0
        result_gdf["blurred_sightings"] = 0

    return result_gdf


def analyze_sampling_bias(sightings_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Analyze known safari route and seasonal tourist biases in sightings:
    1. Monthly distribution: April peak vs July-September monsoon collapse.
    2. Proximity concentration around major MP Tiger Reserves (Kanha, Pench, Bandhavgarh, Panna).
    """
    if sightings_gdf.empty:
        return {"error": "No sightings data"}

    # Monthly breakdown
    months = sightings_gdf["month"].dropna().astype(int)
    month_counts = {m: int((months == m).sum()) for m in range(1, 13)}

    # Month names
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    monthly_trend = [
        {"month_num": m, "month_name": month_names[m - 1], "count": month_counts.get(m, 0)}
        for m in range(1, 13)
    ]

    total_valid_months = sum(month_counts.values()) or 1
    april_count = month_counts.get(4, 0)
    monsoon_count = sum(month_counts.get(m, 0) for m in [7, 8, 9])

    april_share_pct = round((april_count / total_valid_months) * 100, 1)
    monsoon_share_pct = round((monsoon_count / total_valid_months) * 100, 1)

    # Safari Tiger Reserves in MP
    reserves = [
        {"name": "Kanha Tiger Reserve", "lat": 22.33, "lon": 80.61},
        {"name": "Pench Tiger Reserve", "lat": 21.76, "lon": 79.33},
        {"name": "Bandhavgarh Tiger Reserve", "lat": 23.68, "lon": 81.03},
        {"name": "Panna Tiger Reserve", "lat": 24.65, "lon": 80.01},
        {"name": "Satpura Tiger Reserve", "lat": 22.45, "lon": 78.43},
    ]

    # Calculate distance to nearest tiger reserve
    coords_rad = np.radians(sightings_gdf[["decimalLatitude", "decimalLongitude"]].values)
    near_reserve_count = 0

    for lat, lon in zip(sightings_gdf["decimalLatitude"], sightings_gdf["decimalLongitude"]):
        # Approx distance in km to nearest reserve
        dists = [
            111.0 * np.sqrt((lat - res["lat"])**2 + ((lon - res["lon"]) * np.cos(np.radians(lat)))**2)
            for res in reserves
        ]
        if min(dists) <= 30.0:  # Within 30 km of tiger reserve
            near_reserve_count += 1

    reserve_share_pct = round((near_reserve_count / len(sightings_gdf)) * 100, 1)

    return {
        "monthly_trend": monthly_trend,
        "april_peak_count": april_count,
        "april_share_pct": april_share_pct,
        "monsoon_dip_count": monsoon_count,
        "monsoon_share_pct": monsoon_share_pct,
        "near_reserve_count": near_reserve_count,
        "reserve_share_pct": reserve_share_pct,
        "bias_summary": (
            f"{reserve_share_pct}% of mammal records cluster within 30 km of 5 major tiger reserves, "
            f"and April accounts for {april_share_pct}% of records while monsoon (July-Sept) plummets to "
            f"{monsoon_share_pct}% due to park closures. These observations reflect tourist presence "
            f"and safari accessibility, NOT biological animal abundance or seasonal migration."
        ),
    }
