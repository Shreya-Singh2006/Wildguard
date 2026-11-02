"""
Common UI components, styling, sidebar filters, and data helpers for WildGuard.
Ensures consistency across all dashboard pages:
- Projector and laptop responsive CSS styling
- Universal sidebar filters (species, years, uncertainty)
- Standard source line and limit line on all maps and tables
- Red / Amber / Green risk badges with explicit text labels
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import geopandas as gpd
import pandas as pd
import streamlit as st

from src.config import (
    KANHA_PENCH_CONFIG,
    MP_CONFIG,
    SQLITE_DB_PATH,
    TAIWAN_ROADKILL_CONFIG,
    THREATENED_BLURRED_SPECIES,
)
from src.store import (
    get_metadata,
    load_highway_screening,
    load_roadkill,
    load_sightings,
    load_validation_results,
)


def apply_custom_css():
    """Inject polished, modern CSS styling suitable for laptops and projectors."""
    st.markdown(
        """
        <style>
        /* Base page styling */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 95%;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 12px 18px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        /* Risk badges */
        .badge-risk-high {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
            border: 1px solid #f87171;
            display: inline-block;
        }
        .badge-risk-moderate {
            background-color: #fef3c7;
            color: #92400e;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
            border: 1px solid #fcd34d;
            display: inline-block;
        }
        .badge-risk-low {
            background-color: #d1fae5;
            color: #065f46;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
            border: 1px solid #6ee7b7;
            display: inline-block;
        }

        /* Mandatory notice callout */
        .screening-banner {
            background-color: #fffbeb;
            border-left: 5px solid #f59e0b;
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 1rem;
            font-size: 0.95rem;
        }

        /* Source & Limit footnote box */
        .map-footnote-box {
            background-color: #f1f5f9;
            border: 1px solid #cbd5e1;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.82rem;
            color: #334155;
            margin-top: 6px;
            margin-bottom: 12px;
        }
        .map-footnote-box b {
            color: #0f172a;
        }

        /* Projector friendly high contrast headers */
        h1, h2, h3 {
            color: #1e293b;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=600)
def get_cached_sightings() -> gpd.GeoDataFrame:
    """Load cleaned MP sightings GeoDataFrame."""
    return load_sightings()


@st.cache_data(ttl=600)
def get_cached_roadkill() -> gpd.GeoDataFrame:
    """Load cleaned roadkill records GeoDataFrame."""
    return load_roadkill()


@st.cache_data(ttl=600)
def get_cached_screening() -> gpd.GeoDataFrame:
    """Load ranked highway screening segments."""
    return load_highway_screening()


@st.cache_data(ttl=600)
def get_cached_validation() -> Dict[str, Any]:
    """Load validation metrics and bootstrap CIs."""
    return load_validation_results()


def render_sidebar_filters(sightings_gdf: Optional[gpd.GeoDataFrame] = None) -> Dict[str, Any]:
    """
    Render global sidebar filters available on all pages:
    - Species filter
    - Year slider (2000-2026)
    - Coordinate uncertainty filter
    - Refresh data button
    """
    st.sidebar.markdown("## 🛡️ WildGuard")
    st.sidebar.caption("Wildlife-Highway Conflict Screening & Roadkill Method Validation")
    st.sidebar.markdown("---")

    # Status & Fallback Indicator
    selected_country = get_metadata("roadkill_selected_country", "Taiwan")
    last_fetched = get_metadata("pipeline_last_run", "Not yet fetched")
    if last_fetched != "Not yet fetched":
        try:
            dt = datetime.fromisoformat(last_fetched)
            last_fetched = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

    st.sidebar.info(
        f"**Roadkill Dataset Country:** {selected_country} *(TaiRON)*\n\n"
        f"**Sightings Focus:** Madhya Pradesh, India\n\n"
        f"**Last Data Fetch:** {last_fetched}"
    )

    st.sidebar.markdown("### 🔍 Global Filters")

    # Species options
    species_options = []
    if sightings_gdf is not None and not sightings_gdf.empty and "species" in sightings_gdf.columns:
        unique_sp = sorted([s for s in sightings_gdf["species"].dropna().unique() if str(s).strip() != ""])
        species_options = unique_sp

    selected_species = st.sidebar.multiselect(
        "Filter by Species:",
        options=species_options,
        default=[],
        help="Select one or more species to filter wildlife records.",
    )

    # Year Range
    min_year = 2000
    max_year = 2026
    if sightings_gdf is not None and not sightings_gdf.empty and "year" in sightings_gdf.columns:
        valid_yrs = sightings_gdf["year"].dropna()
        if not valid_yrs.empty:
            min_year = int(max(2000, valid_yrs.min()))
            max_year = int(min(2026, valid_yrs.max()))

    year_range = st.sidebar.slider(
        "Year Range:",
        min_value=2000,
        max_value=2026,
        value=(min_year, max_year),
        step=1,
    )

    # Uncertainty Tag Filter
    uncertainty_choice = st.sidebar.selectbox(
        "Coordinate Uncertainty Filter:",
        options=[
            "All Records",
            "Precise Only (<= 1,000m)",
            "Blurred Threatened Species (~30km blur)",
        ],
        index=0,
        help="Precise records have uncertainty <= 1km. Threatened species (tiger, leopard, etc.) are blurred ~30km by iNaturalist.",
    )

    st.sidebar.markdown("---")

    # Refresh Data Action
    st.sidebar.markdown("### 🔄 Data Pipeline")
    refresh_clicked = st.sidebar.button("Refresh Open Data", help="Refetches raw data from GBIF and Overpass API")

    return {
        "selected_species": selected_species,
        "year_range": year_range,
        "uncertainty_choice": uncertainty_choice,
        "refresh_clicked": refresh_clicked,
    }


def render_map_footnote(source_text: str, limit_text: str):
    """Render standard source line and limit line directly beneath maps."""
    st.markdown(
        f"""
        <div class="map-footnote-box">
            <b>Source:</b> {source_text}<br>
            <b>Limitation:</b> {limit_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_sightings(
    gdf: gpd.GeoDataFrame,
    selected_species: List[str],
    year_range: Tuple[int, int],
    uncertainty_choice: str,
) -> gpd.GeoDataFrame:
    """Apply sidebar filters to sightings GeoDataFrame."""
    if gdf.empty:
        return gdf

    filtered = gdf.copy()

    # Filter year
    if "year" in filtered.columns:
        filtered = filtered[
            (filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])
        ]

    # Filter species
    if selected_species and "species" in filtered.columns:
        filtered = filtered[filtered["species"].isin(selected_species)]

    # Filter uncertainty
    if uncertainty_choice == "Precise Only (<= 1,000m)":
        filtered = filtered[filtered["uncertainty_tag"] == "precise"]
    elif uncertainty_choice == "Blurred Threatened Species (~30km blur)":
        filtered = filtered[filtered["uncertainty_tag"] == "blurred"]

    return filtered

