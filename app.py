"""
WildGuard - Main Streamlit Dashboard Application.
Evidence-based road ecology decision support tool for Madhya Pradesh, India.
Features:
1. Wildlife sightings explorer for MP, India
2. Roadkill hotspot analysis on real roadkill records (Taiwan TaiRON fallback validation)
3. Highway screening ranking for Kanha-Pench ecological corridor
4. Honest data quality audit of sampling bias and coordinate blurring
"""

import streamlit as st

st.set_page_config(
    page_title="WildGuard | Wildlife-Highway Conflict Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from datetime import datetime
import folium
from streamlit_folium import st_folium
import pandas as pd

from src.config import MP_CONFIG, TAIWAN_ROADKILL_CONFIG
from src.pipeline import run_full_pipeline
from src.store import get_metadata
from src.ui_common import (
    apply_custom_css,
    filter_sightings,
    get_cached_roadkill,
    get_cached_screening,
    get_cached_sightings,
    get_cached_validation,
    is_filter_active,
    render_filter_status_banner,
    render_sidebar_filters,
)

apply_custom_css()

# Load cached data
sightings_gdf = get_cached_sightings()
roadkill_gdf = get_cached_roadkill()
screening_gdf = get_cached_screening()
validation_data = get_cached_validation()

# Render universal sidebar filters
filters = render_sidebar_filters(sightings_gdf)
filtered_sightings = filter_sightings(
    sightings_gdf,
    filters["selected_species"],
    filters["year_range"],
    filters["uncertainty_choice"],
)

# Handle Refresh Data button
if filters.get("refresh_clicked"):
    with st.status("Refreshing open data from GBIF and Overpass...", expanded=True) as status:
        st.write("Checking open data endpoints and validating thresholds...")
        res = run_full_pipeline(force_refresh=True)
        status.update(label="Data refresh complete!", state="complete", expanded=False)
    st.cache_data.clear()
    st.rerun()

# Check if data exists; if completely empty, run automatic first-time fetch
if sightings_gdf.empty or screening_gdf.empty:
    st.info("👋 Welcome to WildGuard! Initializing and fetching open data from GBIF and OpenStreetMap...")
    with st.status("Running initial WildGuard data pipeline...", expanded=True) as status:
        res = run_full_pipeline(force_refresh=False)
        status.update(label="Initial setup complete!", state="complete", expanded=False)
    st.cache_data.clear()
    st.rerun()

# Main Header
st.title("🛡️ WildGuard: Wildlife & Highway Conflict Screening")
st.markdown(
    ":green-badge[Live Open Data] :blue-badge[Madhya Pradesh, India] :orange-badge[Kanha-Pench NH-44 Corridor] :violet-badge[Taiwan TaiRON Validation]"
)
st.markdown(
    "**Evidence-based road ecology decision support tool.** "
    "Connecting open biodiversity occurrence archives, robust spatial outlier hotspot statistics, and multi-criteria highway exposure screening."
)

# Mandatory Fallback Callout
selected_country = get_metadata("roadkill_selected_country", "Taiwan")
selection_reason = get_metadata(
    "roadkill_selection_reason",
    "India Roadkill Monitoring Project has only 491 records nationwide (none in MP). "
    "Falling back to Taiwan Roadkill Observation Network (TaiRON, 46,416 records) for method validation.",
)

st.markdown(
    f"""
    <div class="screening-banner">
        <b>🌐 Dataset & Fallback Status:</b> Roadkill method validation is conducted on <b>{selected_country} (TaiRON)</b>.<br>
        <i>Reason:</i> {selection_reason}<br>
        <b>📍 Wildlife Sightings & Highway Screening:</b> Strictly focused on <b>Madhya Pradesh, India</b> (Kanha-Pench Corridor).
    </div>
    """,
    unsafe_allow_html=True,
)

# Render active filter status chip if filters are modified
render_filter_status_banner(filters, len(sightings_gdf), len(filtered_sightings))

# Top KPI Metric Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    sightings_delta = f"Filtered from {len(sightings_gdf):,}" if is_filter_active(filters) else "All clean MP records"
    st.metric(
        "MP Mammal Sightings",
        f"{len(filtered_sightings):,}",
        delta=sightings_delta,
        delta_color="normal" if not is_filter_active(filters) else "inverse",
        help="Cleaned human observation mammal sightings in MP (2000-2026)",
    )
with kpi2:
    blurred_count = int(filtered_sightings["is_threatened_blurred"].sum()) if not filtered_sightings.empty else 0
    st.metric(
        "Threatened Blurred Records",
        f"{blurred_count:,}",
        help="Tiger, leopard, gaur, etc. with ~30km deliberate blur radius",
    )
with kpi3:
    st.metric(
        f"{selected_country} Roadkill Records",
        f"{len(roadkill_gdf):,}",
        help="Georeferenced roadkill records used for validation (uncertainty <= 1km)",
    )
with kpi4:
    screened_km = len(screening_gdf) if not screening_gdf.empty else 0
    st.metric(
        "Screened Highways",
        f"{screened_km:,} km",
        help="1 km highway stretches evaluated in Kanha-Pench corridor",
    )
with kpi5:
    high_risk_count = (
        int((screening_gdf["risk_category"] == "High Risk").sum())
        if not screening_gdf.empty and "risk_category" in screening_gdf.columns
        else 0
    )
    st.metric(
        "High Exposure Stretches",
        f"{high_risk_count}",
        delta=f"Ranked for field survey",
        delta_color="inverse",
    )

st.markdown("---")

# Quick Navigation Guide with polished cards
st.markdown("### 🧭 Interactive Modules")
col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)

with col_nav1:
    with st.container(border=True):
        st.markdown("#### 🐾 [Wildlife Explorer](/Wildlife_Explorer)")
        st.caption("Interactive map of MP mammal records with strict separation between precise points and 0.25° grid cells for endangered taxa.")

with col_nav2:
    with st.container(border=True):
        st.markdown("#### 🎯 [Roadkill Validation](/Roadkill_Validation)")
        st.caption("Statistical validation of 1 km Modified Z-score hotspot detection against density and random baselines with 95% bootstrap CIs.")

with col_nav3:
    with st.container(border=True):
        st.markdown("#### 🛣️ [Highway Screening](/Highway_Screening)")
        st.caption("Multi-criteria survey priority ranking for NH-44 and connecting roads between Kanha and Pench Tiger Reserves with CSV export.")

with col_nav4:
    with st.container(border=True):
        st.markdown("#### 🔍 [Data Quality & Limits](/Data_Quality)")
        st.caption("Transparent audit of data cleaning attrition, coordinate uncertainty distribution, safari clustering, and monsoon dip bias.")

st.markdown("---")

# Methodology & System Architecture Overview in Tabs
st.markdown("### 📐 Scientific Integrity & System Design")
tab_method, tab_arch = st.tabs([
    ":material/menu_book: Scientific Methodology",
    ":material/memory: System Architecture & Data Pipeline",
])

with tab_method:
    st.markdown(
        """
        1. **Zero Data Fabrication:** Every single record originates directly from official open APIs (GBIF, OpenStreetMap Overpass, geoBoundaries, IMD).
        2. **Precautionary Species Privacy:** iNaturalist injects deliberate ~30 km obfuscation for 7 threatened species (*Tiger, Leopard, Gaur, Sambar, Sloth Bear, Dhole, Barasingha*). We never convert blurred centroids into false pinpoint coordinates; they are strictly aggregated into 0.25° cells.
        3. **Robust Outlier Detection:** Rather than naive fatality counts, we employ the Modified Z-Score using Median and Median Absolute Deviation ($M_i = 0.6745 \\times (x_i - \\text{median}) / \\text{MAD} > 3.5$), falling back to the 90th percentile if MAD is 0.
        4. **Temporal Holdout Validation:** Hotspot models are trained on past records ($\le 2015$) and evaluated on future unseen records ($> 2015$), benchmarking against density-only and random baselines.
        5. **Exposure vs. Mortality Distinction:** Screening identifies **predicted exposure to traffic**, NOT observed road mortalities, directing forest department survey budgets with scientific honesty.
        """
    )

with tab_arch:
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown(
            """
            - **Platform:** Python 3.11 & Streamlit Wide Layout
            - **Spatial Engine:** GeoPandas & Shapely 2.0+ Vector Geometry
            - **Statistical Metrics:** scikit-learn & 1,000-iteration Bootstrap CIs
            - **Basemap Engine:** High-performance OpenStreetMap Leaflet layers
            """
        )
    with col_a2:
        st.markdown(
            """
            - **Persistence Layer:** SQLite (`data/wildguard.db`) & Apache Parquet
            - **Caching Policy:** 30-Day TTL with automatic freshness validation
            - **Network Resiliency:** Exponential backoff retry with offline fallback
            - **API Credentials:** 100% Open Access (Zero API keys required)
            """
        )

