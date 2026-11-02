"""
OpenStreetMap Highway Data Fetcher for WildGuard.
Queries the Overpass API for major highways (motorway, trunk, primary) within study areas,
converts the response to GeoPandas GeoDataFrame (LineStrings), and caches to GeoJSON.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
import geopandas as gpd
from shapely.geometry import LineString

from src.config import CACHE_MAX_AGE_DAYS, RAW_DATA_DIR

logger = logging.getLogger(__name__)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]


def _query_overpass(query: str, max_retries: int = 3, timeout: int = 35) -> Optional[Dict[str, Any]]:
    """Execute Overpass QL query across mirrors with retry and backoff."""
    for attempt in range(max_retries):
        endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            logger.info(f"Querying Overpass endpoint ({endpoint})... attempt {attempt + 1}")
            resp = requests.post(
                endpoint,
                data={"data": query},
                timeout=timeout,
                headers={"User-Agent": "WildGuard-RoadEcology-Tool/1.0"},
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning("Overpass rate limited (429). Backing off...")
                time.sleep(5 * (attempt + 1))
            else:
                logger.warning(f"Overpass HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            logger.warning(f"Error querying {endpoint}: {e}")
        time.sleep(2 * (attempt + 1))
    return None


def fetch_highways_osm(
    bbox: Dict[str, float],
    area_name: str,
    highway_types: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """
    Fetch major highway ways in bbox from OpenStreetMap via Overpass API.
    bbox format: {'min_lat': ..., 'min_lon': ..., 'max_lat': ..., 'max_lon': ...}
    Saves to data/raw/highways_{area_name}.geojson
    """
    if highway_types is None:
        highway_types = ["motorway", "trunk", "primary"]

    cache_path = RAW_DATA_DIR / f"highways_{area_name}.geojson"
    meta_path = RAW_DATA_DIR / f"highways_{area_name}_meta.json"

    # Check cache
    if not force_refresh and cache_path.exists():
        try:
            gdf = gpd.read_file(cache_path)
            if not gdf.empty:
                logger.info(f"Loaded {len(gdf)} highway segments for {area_name} from cache.")
                return gdf
        except Exception as e:
            logger.warning(f"Failed to read cached highway GeoJSON: {e}")

    # Build Overpass query
    # Overpass bbox format is (south, west, north, east)
    s = bbox["min_lat"]
    w = bbox["min_lon"]
    n = bbox["max_lat"]
    e = bbox["max_lon"]

    types_regex = "|".join(highway_types)
    query = f"""
    [out:json][timeout:90];
    (
      way["highway"~"^({types_regex})$"]({s},{w},{n},{e});
    );
    out body;
    >;
    out skel qt;
    """

    data = _query_overpass(query)
    elements = data.get("elements", []) if data else []

    # Parse nodes and ways
    nodes: Dict[int, tuple] = {}
    ways: List[Dict[str, Any]] = []

    for el in elements:
        el_type = el.get("type")
        if el_type == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
        elif el_type == "way":
            ways.append(el)

    features = []
    for w in ways:
        node_ids = w.get("nodes", [])
        pts = [nodes[nid] for nid in node_ids if nid in nodes]
        if len(pts) >= 2:
            geom = LineString(pts)
            tags = w.get("tags", {})
            features.append({
                "osm_id": w.get("id"),
                "highway": tags.get("highway", "primary"),
                "name": tags.get("name", "Unnamed Highway"),
                "ref": tags.get("ref", ""),
                "maxspeed": tags.get("maxspeed", ""),
                "lanes": tags.get("lanes", ""),
                "oneway": tags.get("oneway", "no"),
                "geometry": geom,
            })

    if not features:
        logger.warning(f"No highway features returned from Overpass for {area_name}. Checking fallback.")
        # If cache exists even if expired, fallback
        if cache_path.exists():
            return gpd.read_file(cache_path)
        # Create minimal representative highways if completely offline
        gdf = gpd.GeoDataFrame(
            columns=["osm_id", "highway", "name", "ref", "maxspeed", "lanes", "oneway", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
        return gdf

    gdf = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")
    logger.info(f"Fetched {len(gdf)} highway ways for {area_name} from Overpass API.")

    # Save to GeoJSON
    gdf.to_file(cache_path, driver="GeoJSON")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "area_name": area_name,
            "bbox": bbox,
            "highway_types": highway_types,
            "cached_at": datetime.now().isoformat(),
            "count": len(gdf),
        }, f, indent=2)

    return gdf
