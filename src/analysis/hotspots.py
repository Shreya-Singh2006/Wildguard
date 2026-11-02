"""
Roadkill Hotspot Analysis Module for WildGuard.
Implements:
1. Metric highway segmentation into 1 km stretches.
2. Spatial roadkill count aggregation per segment.
3. Robust outlier rule:
   - Modified Z-Score > 3.5 using Median and MAD:
     M_i = 0.6745 * (count - median) / MAD
   - If MAD == 0: fallback to 90th percentile of non-zero segments.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring

from src.config import HOTSPOT_CONFIG

logger = logging.getLogger(__name__)


def segment_highways_1km(
    highways_gdf: gpd.GeoDataFrame,
    segment_length_m: float = 1000.0,
    min_segment_length_m: float = 200.0,
) -> gpd.GeoDataFrame:
    """
    Split highway LineStrings into ~1 km metric segments.
    Projects to EPSG:3857 (Web Mercator) for distance calculation in meters,
    then reprojects back to EPSG:4326 for GIS and web mapping.
    """
    if highways_gdf.empty:
        return gpd.GeoDataFrame(
            columns=["segment_id", "highway_name", "highway_type", "ref", "length_m", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    # Ensure projected CRS for metric length
    gdf_proj = highways_gdf.to_crs(epsg=3857)

    segments = []
    seg_idx = 0

    for _, row in gdf_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        hw_name = row.get("name", "Unnamed Highway")
        hw_type = row.get("highway", "primary")
        hw_ref = row.get("ref", "")
        osm_id = row.get("osm_id", "")

        # Handle LineString and MultiLineString
        geoms = [geom] if geom.geom_type == "LineString" else list(geom.geoms)

        for g in geoms:
            total_len = g.length
            if total_len <= segment_length_m:
                segments.append({
                    "segment_id": f"seg_{seg_idx:05d}",
                    "osm_id": osm_id,
                    "highway_name": hw_name,
                    "highway_type": hw_type,
                    "ref": hw_ref,
                    "length_m": round(total_len, 1),
                    "geometry": g,
                })
                seg_idx += 1
            else:
                curr_dist = 0.0
                while curr_dist < total_len:
                    end_dist = min(curr_dist + segment_length_m, total_len)
                    sub = substring(g, curr_dist, end_dist)
                    sub_len = sub.length

                    # If remaining snippet is tiny, append to previous segment if available
                    if sub_len < min_segment_length_m and segments:
                        # Skip or keep
                        pass
                    else:
                        segments.append({
                            "segment_id": f"seg_{seg_idx:05d}",
                            "osm_id": osm_id,
                            "highway_name": hw_name,
                            "highway_type": hw_type,
                            "ref": hw_ref,
                            "length_m": round(sub_len, 1),
                            "geometry": sub,
                        })
                        seg_idx += 1

                    curr_dist += segment_length_m

    seg_gdf_proj = gpd.GeoDataFrame(segments, geometry="geometry", crs="EPSG:3857")
    seg_gdf = seg_gdf_proj.to_crs(epsg=4326)
    logger.info(f"Split {len(highways_gdf)} highway ways into {len(seg_gdf)} 1-km segments.")
    return seg_gdf


def count_records_per_segment(
    segments_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    buffer_meters: float = 200.0,
) -> gpd.GeoDataFrame:
    """
    Count points within buffer_meters of each 1 km highway segment.
    """
    if segments_gdf.empty:
        return segments_gdf

    result = segments_gdf.copy()
    if points_gdf.empty:
        result["roadkill_count"] = 0
        return result

    # Project to EPSG:3857 for accurate buffering in meters
    seg_proj = result.to_crs(epsg=3857)
    pts_proj = points_gdf.to_crs(epsg=3857)

    # Buffer segments
    buffered_seg = seg_proj.copy()
    buffered_seg["geometry"] = buffered_seg.geometry.buffer(buffer_meters)

    # Spatial join
    sj = gpd.sjoin(pts_proj, buffered_seg[["segment_id", "geometry"]], predicate="within")

    if not sj.empty:
        counts = sj.groupby("segment_id").size().rename("roadkill_count")
        result = result.join(counts, on="segment_id")
        result["roadkill_count"] = result["roadkill_count"].fillna(0).astype(int)
    else:
        result["roadkill_count"] = 0

    return result


def calculate_modified_z_scores(
    counts: np.ndarray,
    percentile_fallback: float = 90.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute modified z-scores and hotspot flags using Median and MAD.
    Formula:
      M_i = 0.6745 * (count - median) / MAD
    If MAD == 0:
      Fallback: Flag segments >= 90th percentile of non-zero segments.
    Returns:
      (modified_z_scores, is_hotspot_mask, stats_dict)
    """
    n = len(counts)
    if n == 0:
        return np.array([]), np.array([], dtype=bool), {}

    counts = np.asarray(counts, dtype=float)
    med = float(np.median(counts))
    abs_dev = np.abs(counts - med)
    mad = float(np.median(abs_dev))

    stats = {
        "total_segments": n,
        "segments_with_records": int(np.sum(counts > 0)),
        "max_count": int(np.max(counts)) if n > 0 else 0,
        "mean_count": round(float(np.mean(counts)), 2) if n > 0 else 0.0,
        "median": med,
        "mad": mad,
        "used_fallback": False,
        "threshold_value": 0.0,
    }

    if mad > 0:
        mod_z = 0.6745 * (counts - med) / mad
        is_hotspot = mod_z > HOTSPOT_CONFIG["modified_z_threshold"]
        stats["threshold_rule"] = "Modified Z-Score > 3.5 (Median + MAD)"
        stats["threshold_value"] = round(med + (3.5 * mad / 0.6745), 2)
    else:
        # Fallback when MAD == 0 (e.g., majority of segments have 0 or 1 roadkill)
        stats["used_fallback"] = True
        stats["threshold_rule"] = f"{percentile_fallback}th Percentile of Non-Zero Segments"
        non_zeros = counts[counts > 0]
        if len(non_zeros) > 0:
            p_val = float(np.percentile(non_zeros, percentile_fallback))
            # Require at least 1 record to be a hotspot
            thresh = max(1.0, p_val)
            stats["threshold_value"] = thresh
            is_hotspot = (counts >= thresh) & (counts > 0)
            # Impute nominal modified z for ranking
            mod_z = np.where(counts > 0, counts / max(1.0, np.mean(non_zeros)), 0.0)
        else:
            is_hotspot = np.zeros(n, dtype=bool)
            mod_z = np.zeros(n, dtype=float)
            stats["threshold_value"] = 1.0

    stats["hotspot_count"] = int(np.sum(is_hotspot))
    stats["hotspot_pct"] = round((stats["hotspot_count"] / n) * 100, 2) if n > 0 else 0.0

    return mod_z, is_hotspot, stats


def identify_hotspots(
    segments_with_counts: gpd.GeoDataFrame,
    count_col: str = "roadkill_count",
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Run the robust outlier rule on segment counts and annotate GeoDataFrame.
    """
    gdf = segments_with_counts.copy()
    if gdf.empty or count_col not in gdf.columns:
        return gdf, {}

    counts = gdf[count_col].values
    mod_z, is_hotspot, stats = calculate_modified_z_scores(counts)

    gdf["modified_z"] = np.round(mod_z, 2)
    gdf["is_hotspot"] = is_hotspot

    logger.info(
        f"Identified {stats['hotspot_count']} hotspots out of {len(gdf)} segments "
        f"({stats['hotspot_pct']}%) using rule: {stats['threshold_rule']}."
    )
    return gdf, stats
