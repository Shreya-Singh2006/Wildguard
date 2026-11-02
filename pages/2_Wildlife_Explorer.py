"""
Page 2: Wildlife Sightings Explorer (Madhya Pradesh, India).
Visualizes mammal sightings across MP with strict separation of precise observations
vs deliberately blurred threatened species (~30 km blur).
NO DBSCAN clustering is performed per specification.
"""

import streamlit as st

st.set_page_config(page_title="Wildlife Explorer | WildGuard", page_icon="🐾", layout="wide")

from datetime import datetime
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np

from src.config import KANHA_PENCH_CONFIG, MP_CONFIG, THREATENED_BLURRED_SPECIES
from src.analysis.sightings import aggregate_sightings_by_grid
from src.ui_common import (
    apply_custom_css,
    filter_sightings,
    get_cached_sightings,
    render_filter_status_banner,
    render_map_footnote,
    render_sidebar_filters,
)

apply_custom_css()

# Data loading
sightings_gdf = get_cached_sightings()
filters = render_sidebar_filters(sightings_gdf)

st.title("🐾 Madhya Pradesh Wildlife Sightings Explorer")
st.markdown(
    "Explore open mammal observations across Madhya Pradesh (2000–2026). "
    "To protect endangered wildlife, threatened species are deliberately blurred (~30 km) and displayed "
    "strictly as wide grid cells, never as pinpoint locations."
)

if sightings_gdf.empty:
    st.warning("⚠️ No sightings data currently available in database. Please click 'Refresh Open Data' in the sidebar.")
    st.stop()

# Apply active sidebar filters
filtered_gdf = filter_sightings(
    sightings_gdf,
    filters["selected_species"],
    filters["year_range"],
    filters["uncertainty_choice"],
)

# Prominent active filter status chip
render_filter_status_banner(filters, len(sightings_gdf), len(filtered_gdf))

# Metric Summary Bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Filtered Records", f"{len(filtered_gdf):,}", f"out of {len(sightings_gdf):,}")
with col2:
    p_cnt = int((filtered_gdf["uncertainty_tag"] == "precise").sum())
    st.metric("Precise Points (<= 1km)", f"{p_cnt:,}", "Safe for point plotting")
with col3:
    b_cnt = int(filtered_gdf["is_threatened_blurred"].sum())
    st.metric("Threatened Blurred Records", f"{b_cnt:,}", "~30 km iNaturalist blur")
with col4:
    unique_sp_cnt = filtered_gdf["species"].nunique() if "species" in filtered_gdf.columns else 0
    st.metric("Unique Species", f"{unique_sp_cnt}")

st.markdown("---")

# Main Interactive Map
st.markdown("### 🗺️ Spatial Sightings Distribution Map")
st.caption("Green points indicate high-precision sightings (<= 1 km). Orange/purple grid boxes indicate wide cells for blurred threatened species.")

col_map, col_controls = st.columns([2.5, 1])

with col_controls:
    st.markdown("#### Layer Visibility")
    show_precise = st.checkbox("Show Precise Sightings (Points)", value=True)
    show_blurred = st.checkbox("Show Threatened Species (Wide Grid Cells)", value=True)
    show_reserves = st.checkbox("Show Major Tiger Reserves", value=True)

    st.markdown("#### Threatened Species Legend")
    st.markdown(
        """
        *Deliberately blurred by ~30 km by iNaturalist:*
        - 🐅 Tiger (*Panthera tigris*)
        - 🐆 Leopard (*Panthera pardus*)
        - 🐂 Gaur (*Bos gaurus*)
        - 🦌 Sambar (*Rusa unicolor*)
        - 🐻 Sloth Bear (*Melursus ursinus*)
        - 🐕 Dhole (*Cuon alpinus*)
        - 🦌 Barasingha (*Rucervus duvaucelii*)
        """
    )
    st.info("💡 **Methodology Note:** Grid cell size is 0.25° (~27 km), aligning with the ~30 km blur radius.")

with col_map:
    # Initialize Folium Map centered on MP
    mp_center = [23.5, 78.5]
    fmap = folium.Map(location=mp_center, zoom_start=7, tiles="OpenStreetMap")

    # Major Tiger Reserves
    reserves = [
        {"name": "Kanha Tiger Reserve", "lat": 22.33, "lon": 80.61},
        {"name": "Pench Tiger Reserve", "lat": 21.76, "lon": 79.33},
        {"name": "Bandhavgarh Tiger Reserve", "lat": 23.68, "lon": 81.03},
        {"name": "Panna Tiger Reserve", "lat": 24.65, "lon": 80.01},
        {"name": "Satpura Tiger Reserve", "lat": 22.45, "lon": 78.43},
    ]

    if show_reserves:
        for res in reserves:
            folium.Marker(
                location=[res["lat"], res["lon"]],
                popup=f"<b>{res['name']}</b><br>National Park / Tiger Reserve",
                icon=folium.Icon(color="darkgreen", icon="tree-conifer", prefix="glyphicon"),
            ).add_to(fmap)

    # 1. Wide Grid Cells for Blurred Threatened Species
    if show_blurred:
        # Generate grid and aggregate
        precise_grid, blurred_grid = aggregate_sightings_by_grid(filtered_gdf, cell_size_deg=0.25)
        active_blurred_cells = blurred_grid[blurred_grid["blurred_count"] > 0]

        max_b_count = max(1, active_blurred_cells["blurred_count"].max()) if not active_blurred_cells.empty else 1

        for _, cell in active_blurred_cells.iterrows():
            c_cnt = cell["blurred_count"]
            # Color intensity based on count
            opacity = min(0.65, 0.15 + (c_cnt / max_b_count) * 0.5)
            bounds = [
                [cell["min_lat"], cell["min_lon"]],
                [cell["max_lat"], cell["max_lon"]],
            ]
            folium.Rectangle(
                bounds=bounds,
                color="#ea580c",
                weight=1.5,
                fill=True,
                fill_color="#f97316",
                fill_opacity=opacity,
                popup=(
                    f"<b>Threatened Species Grid Cell ({cell['cell_id']})</b><br>"
                    f"Blurred Sightings: <b>{c_cnt}</b><br>"
                    f"<i>Coordinates blurred ~30 km by iNaturalist. Not a point location.</i>"
                ),
            ).add_to(fmap)

    # 2. Precise Points (<= 1,000 m uncertainty)
    if show_precise:
        precise_pts = filtered_gdf[
            (filtered_gdf["uncertainty_tag"] == "precise") & (~filtered_gdf["is_threatened_blurred"])
        ]
        # Sample to 350 points for snappy, instantaneous Leaflet rendering
        total_precise = len(precise_pts)
        if total_precise > 350:
            precise_pts = precise_pts.sample(350, random_state=42)

        for _, row in precise_pts.iterrows():
            lat = row.get("decimalLatitude")
            lon = row.get("decimalLongitude")
            if pd.notna(lat) and pd.notna(lon):
                sp_name = row.get("species") or row.get("scientificName") or "Mammal"
                unc = row.get("coordinateUncertaintyInMeters", "N/A")
                yr = row.get("year", "N/A")
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3.5,
                    color="#0d9488",
                    fill=True,
                    fill_color="#14b8a6",
                    fill_opacity=0.75,
                    tooltip=f"{sp_name} ({yr})",
                    popup=f"<b>{sp_name}</b><br>Year: {yr}<br>Coordinate Uncertainty: {unc}m",
                ).add_to(fmap)

    st_folium(fmap, width=850, height=550, returned_objects=[])
    render_map_footnote(
        "GBIF Occurrence API (taxonKey: 359 Mammalia, country: IN, basisOfRecord: HUMAN_OBSERVATION, 2000-2026).",
        "Threatened species (tigers, leopards, gaurs, sloth bears, dholes, sambar, barasingha) are strictly displayed as 0.25° (~27 km) wide grid cells due to deliberate iNaturalist blurring. Point markers are restricted to non-threatened records with <= 1 km uncertainty. Do NOT use as animal census or migration corridors.",
    )

st.markdown("---")

# Analytics: Top Species & Temporal Breakdown
st.markdown("### 📊 Sightings Composition")
c_sp, c_yr = st.columns(2)

with c_sp:
    st.markdown("#### Top Observed Species")
    if "species" in filtered_gdf.columns and not filtered_gdf.empty:
        top_sp = filtered_gdf["species"].value_counts().head(10).reset_index()
        top_sp.columns = ["Species", "Observation Count"]
        st.bar_chart(top_sp.set_index("Species"))
    else:
        st.info("No species data for chart.")

with c_yr:
    st.markdown("#### Observations by Year (2000–2026)")
    if "year" in filtered_gdf.columns and not filtered_gdf.empty:
        yr_counts = filtered_gdf["year"].value_counts().sort_index().reset_index()
        yr_counts.columns = ["Year", "Count"]
        st.line_chart(yr_counts.set_index("Year"))
    else:
        st.info("No yearly data for chart.")

st.markdown("---")

# Data Table Preview
with st.expander("📄 View Filtered Wildlife Records Table", expanded=False):
    cols_to_show = [
        "key", "species", "scientificName", "year", "month",
        "uncertainty_tag", "coordinateUncertaintyInMeters", "is_threatened_blurred"
    ]
    avail_cols = [c for c in cols_to_show if c in filtered_gdf.columns]
    st.dataframe(filtered_gdf[avail_cols].head(100), use_container_width=True)

