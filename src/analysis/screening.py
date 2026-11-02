"""
Highway Screening Module for Madhya Pradesh (Kanha-Pench Corridor).
Ranks 1 km highway stretches for field surveys based on:
1. Overlap with protected areas and wildlife corridors (Kanha & Pench National Parks).
2. Proximity to wildlife sightings (precise points & wide grid cells).
3. Highway classification and traffic speed vulnerability.

MANDATORY LABEL: "Predicted exposure, not observed roadkill"
Provides multi-criteria ranking, categorical risk (Red/Amber/Green + Text),
detailed segment breakdown, and CSV export.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.analysis.hotspots import segment_highways_1km
from src.config import KANHA_PENCH_CONFIG

logger = logging.getLogger(__name__)

PREDICTED_EXPOSURE_LABEL = "Predicted exposure, not observed roadkill"


def calculate_exposure_scores(
    segments_gdf: gpd.GeoDataFrame,
    sightings_gdf: Optional[gpd.GeoDataFrame] = None,
) -> gpd.GeoDataFrame:
    """
    Score 1 km highway segments for wildlife exposure near Kanha-Pench.
    Composite Exposure Score (0 to 100):
    - Protected Area / Corridor Proximity: 40 points
    - Wildlife Sightings Density (within 10 km): 40 points
    - Road Type & Traffic Vulnerability: 20 points
    """
    if segments_gdf.empty:
        return segments_gdf

    df = segments_gdf.copy()
    # Project to EPSG:3857 for accurate metric centroid, then reproject back to EPSG:4326
    centroids = df.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)

    # 1. Protected Area & Corridor Proximity Score (Max 40 pts)
    # Kanha: ~22.33 N, 80.61 E; Pench: ~21.76 N, 79.33 E
    reserves = KANHA_PENCH_CONFIG["protected_areas"]
    pa_scores = []

    for pt in centroids:
        min_km = 999.0
        for pa in reserves:
            # Approximate Euclidean distance in km
            d_km = 111.0 * np.sqrt((pt.y - pa["lat"])**2 + ((pt.x - pa["lon"]) * np.cos(np.radians(pt.y)))**2)
            if d_km < min_km:
                min_km = d_km

        # Score decays with distance from protected areas
        if min_km <= 5.0:
            score = 40.0
        elif min_km <= 15.0:
            score = 30.0 + (15.0 - min_km)
        elif min_km <= 30.0:
            score = 15.0 + ((30.0 - min_km) / 15.0) * 15.0
        elif min_km <= 50.0:
            score = 5.0 + ((50.0 - min_km) / 20.0) * 10.0
        else:
            score = max(0.0, 5.0 - (min_km - 50.0) / 10.0)
        pa_scores.append(round(score, 1))

    df["pa_proximity_score"] = pa_scores

    # 2. Wildlife Sightings Density Score (Max 40 pts)
    # Counts sightings in vicinity and checks blurred vs precise share
    sightings_scores = []
    precise_counts = []
    blurred_counts = []

    if sightings_gdf is not None and not sightings_gdf.empty:
        # Distance calculation
        sight_coords = sightings_gdf[["decimalLatitude", "decimalLongitude"]].dropna().values
        unc_tags = sightings_gdf["uncertainty_tag"].values

        for pt in centroids:
            # Lat/Lon distance approx in degrees (0.1 deg ~ 11 km)
            dists_deg = np.sqrt((sight_coords[:, 0] - pt.y)**2 + (sight_coords[:, 1] - pt.x)**2)
            nearby_mask = dists_deg <= 0.12  # within ~13 km
            n_near = int(np.sum(nearby_mask))

            p_cnt = int(np.sum(nearby_mask & (unc_tags == "precise")))
            b_cnt = int(np.sum(nearby_mask & (unc_tags == "blurred")))
            precise_counts.append(p_cnt)
            blurred_counts.append(b_cnt)

            # Log-scaled score up to 40
            if n_near > 0:
                s_score = min(40.0, 8.0 * np.log1p(n_near) + (p_cnt * 0.5))
            else:
                s_score = 0.0
            sightings_scores.append(round(s_score, 1))
    else:
        # Default placeholder density if no sightings layer provided
        for _ in range(len(df)):
            precise_counts.append(0)
            blurred_counts.append(0)
            sightings_scores.append(10.0)

    df["wildlife_density_score"] = sightings_scores
    df["precise_sightings_nearby"] = precise_counts
    df["blurred_sightings_nearby"] = blurred_counts

    # 3. Road Type & Speed Vulnerability Score (Max 20 pts)
    # Motorway/trunk (e.g. NH-44) has high traffic volume and vehicle speeds
    road_scores = []
    for _, row in df.iterrows():
        hw = str(row.get("highway_type", "")).lower()
        ref = str(row.get("ref", "")).upper()
        if "motorway" in hw or "trunk" in hw or "NH" in ref:
            r_score = 20.0
        elif "primary" in hw or "SH" in ref:
            r_score = 14.0
        elif "secondary" in hw:
            r_score = 8.0
        else:
            r_score = 5.0
        road_scores.append(r_score)

    df["road_vulnerability_score"] = road_scores

    # 4. Total Composite Score
    df["exposure_score"] = np.round(
        df["pa_proximity_score"] + df["wildlife_density_score"] + df["road_vulnerability_score"], 1
    )
    # Clip to 0..100
    df["exposure_score"] = np.clip(df["exposure_score"], 0.0, 100.0)

    # 5. Risk Category (Red/Amber/Green + Text Label)
    def assign_risk(score):
        if score >= 65.0:
            return "High Risk", "#e63946"
        elif score >= 40.0:
            return "Moderate Risk", "#f4a261"
        else:
            return "Low Risk", "#2a9d8f"

    risk_info = df["exposure_score"].apply(assign_risk)
    df["risk_category"] = [r[0] for r in risk_info]
    df["risk_color"] = [r[1] for r in risk_info]

    # Mandatory Label
    df["screening_label"] = PREDICTED_EXPOSURE_LABEL

    # Rank descending
    df = df.sort_values(by="exposure_score", ascending=False).reset_index(drop=True)
    df["survey_rank"] = range(1, len(df) + 1)

    logger.info(f"Calculated exposure scores for {len(df)} highway segments. Top score: {df['exposure_score'].max()}")
    return df
