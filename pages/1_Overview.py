"""
Page 1: Overview Dashboard.
Provides high-level dataset metrics, fallback explanation, methodology pipeline,
and study area geography.
"""

import streamlit as st

st.set_page_config(page_title="Overview | WildGuard", page_icon="📊", layout="wide")

from datetime import datetime
import folium
from streamlit_folium import st_folium
import pandas as pd

from src.config import KANHA_PENCH_CONFIG, MP_CONFIG, ROADKILL_THRESHOLD
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
    render_map_footnote,
    render_sidebar_filters,
)

apply_custom_css()

# Data loading
sightings_gdf = get_cached_sightings()
roadkill_gdf = get_cached_roadkill()
screening_gdf = get_cached_screening()
validation_data = get_cached_validation()

filters = render_sidebar_filters(sightings_gdf)
filtered_sightings = filter_sightings(
    sightings_gdf,
    filters["selected_species"],
    filters["year_range"],
    filters["uncertainty_choice"],
)

if filters.get("refresh_clicked"):
    with st.status("Refreshing open data from GBIF and Overpass...", expanded=True) as status:
        st.write("Executing live queries...")
        run_full_pipeline(force_refresh=True)
        status.update(label="Data refresh complete!", state="complete", expanded=False)
    st.cache_data.clear()
    st.rerun()

st.title("📊 WildGuard Project Overview")
st.markdown(
    "WildGuard connects open biodiversity occurrences and highway geometries to identify and rank "
    "critical wildlife-vehicle collision zones and highway exposure corridors."
)

# Decision Callout
selected_country = get_metadata("roadkill_selected_country", "Taiwan")
selection_reason = get_metadata("roadkill_selection_reason", "")
india_count = get_metadata("india_candidate_count", 491)

st.markdown(
    f"""
    <div class="screening-banner">
        <b>⚖️ Roadkill Candidate Evaluation & Fallback:</b><br>
        • <b>India (Target Country):</b> Evaluated <i>India Roadkill Monitoring Project</i> (GBIF key <code>f334ae5e-0991-44bb-b64b-d602f4c8c289</code>). Only <b>{india_count}</b> qualifying records nationwide and <b>0</b> in MP (below required threshold of <b>1,000</b> records &le; 1 km uncertainty).<br>
        • <b>Fallback Selection:</b> <b>{selected_country} (TaiRON)</b> dataset selected with <b>46,416</b> records for rigorous statistical validation.<br>
        • <b>MP Highway Screening:</b> Preserved strictly on Madhya Pradesh, India, using real MP mammal sightings and OpenStreetMap highways.
    </div>
    """,
    unsafe_allow_html=True,
)

# Active filter banner
render_filter_status_banner(filters, len(sightings_gdf), len(filtered_sightings))

# Metrics Grid
m1, m2, m3, m4 = st.columns(4)
with m1:
    sightings_sub = f"Filtered from {len(sightings_gdf):,}" if is_filter_active(filters) else "Mammalia (2000-2026)"
    st.metric("Total MP Sightings", f"{len(filtered_sightings):,}", sightings_sub)
with m2:
    blurred_count = int(filtered_sightings["is_threatened_blurred"].sum()) if not filtered_sightings.empty else 0
    precise_count = len(filtered_sightings) - blurred_count
    st.metric("Precise Sightings (<= 1km)", f"{precise_count:,}", f"{round(precise_count/max(1, len(filtered_sightings))*100, 1)}% of filtered")
with m3:
    st.metric("TaiRON Roadkill Records", f"{len(roadkill_gdf):,}", "High precision validation")
with m4:
    screened_len = len(screening_gdf) if not screening_gdf.empty else 0
    st.metric("Screened Highway Length", f"{screened_len:,} km", "Kanha-Pench Corridor")

st.markdown("---")

# Comparative Dataset Table
st.markdown("### 📋 Roadkill Dataset Evaluation Matrix")
eval_df = pd.DataFrame([
    {
        "Candidate": "Primary (India)",
        "Dataset Title": "India Roadkill Monitoring Project",
        "GBIF Key": "f334ae5e-0991-44bb-b64b-d602f4c8c289",
        "Total GBIF Records": 491,
        "Qualifying Records (<= 1km unc.)": india_count,
        "Threshold (1,000 min)": "❌ Failed (Below Threshold)",
        "Selection Status": "Rejected for validation due to sparse sample size & 0 MP records",
    },
    {
        "Candidate": "Fallback 1 (Taiwan)",
        "Dataset Title": "Taiwan Roadkill Observation Network (TaiRON)",
        "GBIF Key": "db09684b-0fd1-431e-b5fa-4c1532fbdb14",
        "Total GBIF Records": 46416,
        "Qualifying Records (<= 1km unc.)": 46416,
        "Threshold (1,000 min)": "✅ Exceeded (> 46x Threshold)",
        "Selection Status": "Selected for Roadkill Method Validation",
    },
])
st.dataframe(eval_df, hide_index=True, width="stretch")

st.markdown("---")

# Study Area Context Map
st.markdown("### 🗺️ Focal Landscape: Madhya Pradesh & Kanha-Pench Corridor")
col_map, col_text = st.columns([1.6, 1])

with col_map:
    with st.container(border=True):
        # Build Map centered on MP
        m = folium.Map(
            location=[23.0, 78.5],
            zoom_start=6,
            tiles="OpenStreetMap",
        )

        # Add MP bounding box
        bbox = MP_CONFIG["bbox"]
        folium.Rectangle(
            bounds=[[bbox["min_lat"], bbox["min_lon"]], [bbox["max_lat"], bbox["max_lon"]]],
            color="#3b82f6",
            weight=2,
            fill=True,
            fill_opacity=0.05,
            popup="MP GBIF Sightings Search Bounding Box",
        ).add_to(m)

        # Add Kanha-Pench Corridor Box
        kp_bbox = KANHA_PENCH_CONFIG["bbox"]
        folium.Rectangle(
            bounds=[[kp_bbox["min_lat"], kp_bbox["min_lon"]], [kp_bbox["max_lat"], kp_bbox["max_lon"]]],
            color="#ef4444",
            weight=3,
            fill=True,
            fill_opacity=0.15,
            popup="Kanha-Pench Highway Screening Area",
        ).add_to(m)

        # Add Key Protected Areas
        for pa in KANHA_PENCH_CONFIG["protected_areas"]:
            folium.Marker(
                location=[pa["lat"], pa["lon"]],
                popup=f"<b>{pa['name']}</b><br>Protected Tiger Reserve",
                icon=folium.Icon(color="green", icon="leaf"),
            ).add_to(m)

        st_folium(m, width=700, height=450, returned_objects=[])
        render_map_footnote(
            "OpenStreetMap boundaries & GBIF occurrence geography.",
            "Blue box indicates MP search rectangle (lon 74-83, lat 21-27). Red box marks Kanha-Pench highway corridor.",
        )

with col_text:
    with st.container(border=True):
        st.markdown("#### :material/explore: Regional Context")
        st.markdown(
            """
            - **Madhya Pradesh (MP):** Known as the *Tiger State of India*, MP harbors the largest wild tiger population in the country (~785 tigers).
            - **Linear Infrastructure Conflict:** Major national highways, including NH-44 (North-South Corridor), intersect critical wildlife corridors connecting Kanha and Pench Tiger Reserves.
            - **Data Reality:** Official, systematically collected open roadkill mortality records do not exist for public conservation planning in MP.
            - **WildGuard's Dual Framework:**
              1. Rigorously validate the spatial hotspot algorithm where rich roadkill records exist (TaiRON).
              2. Apply multi-criteria exposure screening directly to MP's highway network to rank stretches for field verification.
            """
        )

