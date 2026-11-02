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


SPECIES_COMMON_NAMES = {
    "Panthera tigris": "🐅 Tiger (Panthera tigris)",
    "Panthera pardus": "🐆 Leopard (Panthera pardus)",
    "Melursus ursinus": "🐻 Sloth Bear (Melursus ursinus)",
    "Cuon alpinus": "🐕 Dhole / Wild Dog (Cuon alpinus)",
    "Bos frontalis": "🐂 Gaur / Indian Bison (Bos frontalis/gaurus)",
    "Rusa unicolor": "🦌 Sambar (Rusa unicolor)",
    "Rucervus duvaucelii": "🦌 Barasingha / Swamp Deer (Rucervus duvaucelii)",
    "Axis axis": "🦌 Chital / Spotted Deer (Axis axis)",
    "Semnopithecus entellus": "🐒 Gray Langur (Semnopithecus entellus)",
    "Macaca mulatta": "🐒 Rhesus Macaque (Macaca mulatta)",
    "Canis aureus": "🐺 Golden Jackal (Canis aureus)",
    "Canis lupus": "🐺 Indian Wolf (Canis lupus)",
    "Sus scrofa": "🐗 Wild Boar (Sus scrofa)",
    "Boselaphus tragocamelus": "🦌 Nilgai / Blue Bull (Boselaphus tragocamelus)",
    "Antilope cervicapra": "🦌 Blackbuck (Antilope cervicapra)",
    "Muntiacus muntjak": "🦌 Barking Deer (Muntiacus muntjak)",
    "Tetracerus quadricornis": "🦌 Four-horned Antelope (Tetracerus quadricornis)",
    "Ratufa indica": "🐿️ Indian Giant Squirrel (Ratufa indica)",
    "Funambulus pennantii": "🐿️ Northern Palm Squirrel (Funambulus pennantii)",
    "Funambulus palmarum": "🐿️ Indian Palm Squirrel (Funambulus palmarum)",
    "Herpestes smithii": "🦦 Ruddy Mongoose (Herpestes smithii)",
    "Herpestes edwardsi": "🦦 Indian Grey Mongoose (Herpestes edwardsi)",
    "Pteropus medius": "🦇 Indian Flying Fox (Pteropus medius)",
    "Hyaena hyaena": "🐺 Striped Hyena (Hyaena hyaena)",
    "Felis chaus": "🐱 Jungle Cat (Felis chaus)",
    "Prionailurus viverrinus": "🐱 Fishing Cat (Prionailurus viverrinus)",
}


def apply_custom_css():
    """Inject polished, modern high-contrast CSS styling for laptop, dark mode, and projectors."""
    st.markdown(
        """
        <style>
        /* Base page styling */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 95%;
        }

        /* Metric cards - high contrast dark card */
        div[data-testid="stMetric"] {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            padding: 14px 18px !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25) !important;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
            color: #94a3b8 !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] *,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] div {
            color: #f8fafc !important;
            font-weight: 800 !important;
            font-size: 1.65rem !important;
        }

        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] *,
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] div {
            font-size: 0.85rem !important;
        }

        /* Mandatory notice callout - crisp high contrast */
        .screening-banner {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            border-left: 6px solid #f59e0b !important;
            padding: 14px 18px !important;
            border-radius: 0 10px 10px 0 !important;
            margin-bottom: 1.2rem !important;
            font-size: 0.95rem !important;
            color: #f1f5f9 !important;
            line-height: 1.5 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }

        .screening-banner b, .screening-banner strong {
            color: #fbbf24 !important;
        }

        .screening-banner i, .screening-banner em {
            color: #cbd5e1 !important;
        }

        .screening-banner code {
            background-color: #0f172a !important;
            color: #38bdf8 !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
        }

        /* Info card box */
        .info-card-box {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            color: #f1f5f9 !important;
            padding: 16px !important;
            border-radius: 10px !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25) !important;
        }

        .info-card-box h4 {
            color: #38bdf8 !important;
            margin-top: 0 !important;
            margin-bottom: 8px !important;
        }

        .info-card-box b, .info-card-box strong {
            color: #f8fafc !important;
        }

        /* Source & Limit footnote box */
        .map-footnote-box {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            padding: 10px 14px !important;
            border-radius: 8px !important;
            font-size: 0.84rem !important;
            color: #cbd5e1 !important;
            margin-top: 8px !important;
            margin-bottom: 14px !important;
            line-height: 1.4 !important;
        }

        .map-footnote-box b, .map-footnote-box strong {
            color: #38bdf8 !important;
        }

        /* Risk badges */
        .badge-risk-high {
            background-color: #450a0a !important;
            color: #fca5a5 !important;
            padding: 4px 10px !important;
            border-radius: 6px !important;
            font-weight: 700 !important;
            border: 1px solid #ef4444 !important;
            display: inline-block !important;
        }

        .badge-risk-moderate {
            background-color: #451a03 !important;
            color: #fde68a !important;
            padding: 4px 10px !important;
            border-radius: 6px !important;
            font-weight: 700 !important;
            border: 1px solid #f59e0b !important;
            display: inline-block !important;
        }

        .badge-risk-low {
            background-color: #022c22 !important;
            color: #6ee7b7 !important;
            padding: 4px 10px !important;
            border-radius: 6px !important;
            font-weight: 700 !important;
            border: 1px solid #10b981 !important;
            display: inline-block !important;
        }

        /* Headers */
        h1, h2, h3 {
            color: #f8fafc !important;
            font-weight: 700 !important;
        }

        h4, h5, h6 {
            color: #e2e8f0 !important;
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
    - Species filter with friendly common names & emojis
    - Year slider (2000-2026)
    - Coordinate uncertainty filter
    - Clear Filters button
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

    # If reset was clicked in session_state, handle reset
    if st.sidebar.button("🔄 Reset / Clear Filters", help="Reset all filters to show full dataset", use_container_width=True):
        st.session_state["filter_species"] = []
        st.session_state["filter_years"] = (2000, 2026)
        st.session_state["filter_unc"] = "All Records"
        st.rerun()

    # Species options sorted by observation count so common species appear first
    species_options = []
    if sightings_gdf is not None and not sightings_gdf.empty and "species" in sightings_gdf.columns:
        counts = sightings_gdf["species"].value_counts()
        species_options = [s for s in counts.index if str(s).strip() != ""]

    selected_species = st.sidebar.multiselect(
        "Filter by Species:",
        options=species_options,
        default=st.session_state.get("filter_species", []),
        key="filter_species",
        format_func=lambda s: SPECIES_COMMON_NAMES.get(s, s),
        help="Select one or more species. Search by common or scientific name (e.g. Tiger, Leopard, Sambar, Chital).",
    )

    # Year Range
    min_year = 2000
    max_year = 2026
    if sightings_gdf is not None and not sightings_gdf.empty and "year" in sightings_gdf.columns:
        valid_yrs = sightings_gdf["year"].dropna()
        if not valid_yrs.empty:
            min_year = int(max(2000, valid_yrs.min()))
            max_year = int(min(2026, valid_yrs.max()))

    default_years = st.session_state.get("filter_years", (min_year, max_year))
    year_range = st.sidebar.slider(
        "Year Range:",
        min_value=2000,
        max_value=2026,
        value=default_years,
        key="filter_years",
        step=1,
    )

    # Uncertainty Tag Filter
    unc_options = [
        "All Records",
        "Precise Only (<= 1,000m)",
        "Blurred Threatened Species (~30km blur)",
    ]
    default_unc = st.session_state.get("filter_unc", "All Records")
    unc_idx = unc_options.index(default_unc) if default_unc in unc_options else 0

    uncertainty_choice = st.sidebar.selectbox(
        "Coordinate Uncertainty Filter:",
        options=unc_options,
        index=unc_idx,
        key="filter_unc",
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


def is_filter_active(filters: Dict[str, Any], default_years: Tuple[int, int] = (2000, 2026)) -> bool:
    """Return True if user has modified any filter from default."""
    if not filters:
        return False
    has_sp = bool(filters.get("selected_species"))
    yr = filters.get("year_range", default_years)
    has_yr = (yr[0] > 2000 or yr[1] < 2026)
    has_unc = filters.get("uncertainty_choice") != "All Records"
    return has_sp or has_yr or has_unc


def render_filter_status_banner(filters: Dict[str, Any], total_count: int, filtered_count: int):
    """Render a prominent, high-contrast banner when filters are actively restricting records."""
    if not is_filter_active(filters):
        return

    sp_list = filters.get("selected_species", [])
    if sp_list:
        friendly = [SPECIES_COMMON_NAMES.get(s, s).split("(")[0].strip() for s in sp_list]
        sp_desc = f"<b>Species ({len(sp_list)}):</b> {', '.join(friendly[:3])}{'...' if len(friendly)>3 else ''}"
    else:
        sp_desc = "<b>Species:</b> All"

    yr = filters.get("year_range", (2000, 2026))
    yr_desc = f"<b>Years:</b> {yr[0]}–{yr[1]}"
    unc_desc = f"<b>Uncertainty:</b> {filters.get('uncertainty_choice', 'All Records')}"

    st.markdown(
        f"""
        <div style="background-color: #0f172a; border: 1px solid #3b82f6; border-left: 6px solid #38bdf8; padding: 12px 16px; border-radius: 8px; margin-bottom: 1.2rem; color: #f8fafc; font-size: 0.92rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <span style="color:#38bdf8; font-weight:700;">🎯 Active Filters Applied:</span> &nbsp;
                {sp_desc} &nbsp;|&nbsp; {yr_desc} &nbsp;|&nbsp; {unc_desc}
            </div>
            <div style="background-color: #1e3a8a; border: 1px solid #3b82f6; color: #93c5fd; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                Matching {filtered_count:,} of {total_count:,} records ({round(filtered_count/max(1, total_count)*100, 1)}%)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        filtered = filtered[
            (filtered["uncertainty_tag"] == "precise") & (~filtered["is_threatened_blurred"])
        ]
    elif uncertainty_choice == "Blurred Threatened Species (~30km blur)":
        filtered = filtered[
            (filtered["uncertainty_tag"] == "blurred") | (filtered["is_threatened_blurred"])
        ]

    return filtered

