"""
Page 4: Highway Exposure Screening (Kanha-Pench Corridor, Madhya Pradesh).
Ranks 1 km highway stretches for field surveys based on composite exposure scores:
- Protected Area Proximity (Kanha & Pench Tiger Reserves)
- Wildlife Sightings Density (Precise points & wide blurred cells)
- Road Type & Traffic Speed Vulnerability (NH-44 vs State Highways)

MANDATORY LABEL: "Predicted exposure, not observed roadkill"
Includes Red/Amber/Green risk classification, interactive map, segment detail drawer,
and CSV export.
"""

import streamlit as st

st.set_page_config(page_title="Highway Screening | WildGuard", page_icon="🛣️", layout="wide")

import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np

from src.config import KANHA_PENCH_CONFIG
from src.ui_common import (
    apply_custom_css,
    get_cached_screening,
    get_cached_sightings,
    render_map_footnote,
    render_sidebar_filters,
)

apply_custom_css()

# Data loading
screening_gdf = get_cached_screening()
if not screening_gdf.empty and "geometry" in screening_gdf.columns:
    if "center_lat" not in screening_gdf.columns or "center_lon" not in screening_gdf.columns:
        rep = screening_gdf.geometry.representative_point()
        screening_gdf["center_lat"] = rep.y
        screening_gdf["center_lon"] = rep.x

sightings_gdf = get_cached_sightings()
filters = render_sidebar_filters(sightings_gdf)

st.title("🛣️ Highway Screening: Kanha-Pench Ecological Corridor")
st.markdown(
    "Multi-criteria screening model ranking 1 km highway stretches for field survey prioritization. "
    "Focuses on National Highway 44 (NH-44) and major primary corridors connecting **Kanha** and **Pench** Tiger Reserves."
)

# Mandatory Label Notice
st.markdown(
    """
    <div class="screening-banner">
        <b>⚠️ MANDATORY SCIENTIFIC CLASSIFICATION: "Predicted exposure, not observed roadkill"</b><br>
        This screening model scores environmental exposure (habitat proximity, wildlife presence, and road speed), 
        <b>NOT confirmed animal fatalities</b>. It provides an objective tool for forest departments to target field 
        surveys and wildlife crossing infrastructure.
    </div>
    """,
    unsafe_allow_html=True,
)

if screening_gdf.empty:
    st.warning("⚠️ No highway screening data available. Please click 'Refresh Open Data' in the sidebar.")
    st.stop()

# Summary Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Stretches Evaluated", f"{len(screening_gdf):,}", "1 km Segments")
with col2:
    high_n = int((screening_gdf["risk_category"] == "High Risk").sum())
    st.metric("High Exposure Stretches", f"{high_n}", f"{round(high_n/len(screening_gdf)*100, 1)}% of corridor")
with col3:
    mod_n = int((screening_gdf["risk_category"] == "Moderate Risk").sum())
    st.metric("Moderate Exposure", f"{mod_n}", f"{round(mod_n/len(screening_gdf)*100, 1)}%")
with col4:
    low_n = int((screening_gdf["risk_category"] == "Low Risk").sum())
    st.metric("Low Exposure", f"{low_n}", f"{round(low_n/len(screening_gdf)*100, 1)}%")
with col5:
    max_sc = round(float(screening_gdf["exposure_score"].max()), 1)
    st.metric("Peak Exposure Score", f"{max_sc} / 100")

st.markdown("---")

# Filter by Risk Level
st.markdown("### 🎛️ Filter Ranked Stretches")
col_f1, col_f2 = st.columns([1.2, 1.8], vertical_alignment="center")
with col_f1:
    risk_filter = st.pills(
        "Filter by Risk Category:",
        options=["High Risk", "Moderate Risk", "Low Risk"],
        default=["High Risk", "Moderate Risk", "Low Risk"],
        selection_mode="multi",
    ) or ["High Risk", "Moderate Risk", "Low Risk"]
with col_f2:
    min_score = st.slider("Minimum Composite Score:", min_value=0.0, max_value=100.0, value=0.0, step=5.0)

filtered_stretches = screening_gdf[
    (screening_gdf["risk_category"].isin(risk_filter))
    & (screening_gdf["exposure_score"] >= min_score)
].copy()

# Segment Inspection Selector
selected_seg_id = st.selectbox(
    "Select a 1 km Segment to Inspect Details:",
    options=filtered_stretches["segment_id"].tolist(),
    index=0 if not filtered_stretches.empty else None,
    format_func=lambda sid: f"Rank #{filtered_stretches.loc[filtered_stretches['segment_id']==sid, 'survey_rank'].values[0]} | {sid} - {filtered_stretches.loc[filtered_stretches['segment_id']==sid, 'highway_name'].values[0]} ({filtered_stretches.loc[filtered_stretches['segment_id']==sid, 'ref'].values[0]}) - Score: {filtered_stretches.loc[filtered_stretches['segment_id']==sid, 'exposure_score'].values[0]}",
)

# Detail Card if selected
if selected_seg_id:
    row = filtered_stretches[filtered_stretches["segment_id"] == selected_seg_id].iloc[0]
    badge_cls = "badge-risk-high" if row["risk_category"] == "High Risk" else ("badge-risk-moderate" if row["risk_category"] == "Moderate Risk" else "badge-risk-low")
    c_lat = float(row.get("center_lat", row.geometry.centroid.y if row.geometry else 22.0))
    c_lon = float(row.get("center_lon", row.geometry.centroid.x if row.geometry else 80.0))

    with st.container():
        st.markdown(
            f"""
            <div class="info-card-box">
                <h4>📍 Segment Detail: {row['segment_id']} — {row['highway_name']} ({row['ref']})</h4>
                <b>Survey Priority Rank:</b> #{row['survey_rank']} &nbsp;|&nbsp;
                <b>Classification:</b> <span class="{badge_cls}">{row['risk_category']}</span> &nbsp;|&nbsp;
                <b>Status:</b> <i>{row['screening_label']}</i><br><br>
                <b>Composite Exposure Score:</b> <span style="font-size:1.4rem; font-weight:800; color:#38bdf8;">{row['exposure_score']} / 100</span><br>
                • <b>Protected Area Proximity:</b> {row['pa_proximity_score']} / 40 pts<br>
                • <b>Wildlife Sightings Density:</b> {row['wildlife_density_score']} / 40 pts ({row['precise_sightings_nearby']} precise nearby, {row['blurred_sightings_nearby']} blurred nearby)<br>
                • <b>Road Type & Speed Vulnerability:</b> {row['road_vulnerability_score']} / 20 pts ({row['highway_type']})<br>
                • <b>Segment Coordinates:</b> Lat {round(c_lat, 4)}, Lon {round(c_lon, 4)} (Length: {round(row['length_m'], 0)} m)
            </div>
            """,
            unsafe_allow_html=True,
        )

# Map & Table Layout
col_map, col_table = st.columns([1.5, 1])

with col_map:
    with st.container(border=True):
        st.markdown("#### 🗺️ Corridor Screening Map")
        m_screen = folium.Map(location=[22.1, 80.0], zoom_start=8, tiles="OpenStreetMap")

        # Add Kanha and Pench reserve locations
        for pa in KANHA_PENCH_CONFIG["protected_areas"]:
            folium.Marker(
                location=[pa["lat"], pa["lon"]],
                popup=f"<b>{pa['name']}</b>",
                icon=folium.Icon(color="green", icon="leaf"),
            ).add_to(m_screen)

        # Fast GeoJSON rendering of highway segments
        if not filtered_stretches.empty:
            disp_gdf = filtered_stretches[[
                "segment_id", "highway_name", "ref", "exposure_score", "risk_category", "survey_rank", "geometry"
            ]].copy()
            
            high_mod = disp_gdf[disp_gdf["risk_category"].isin(["High Risk", "Moderate Risk"])]
            low = disp_gdf[disp_gdf["risk_category"] == "Low Risk"]
            if len(low) > 800:
                low = low.sample(800, random_state=42)
            disp_gdf = pd.concat([high_mod, low]) if not low.empty else high_mod
            
            disp_gdf["geometry"] = disp_gdf["geometry"].simplify(tolerance=0.0005)

            def style_fn(feature):
                cat = feature["properties"].get("risk_category", "Low Risk")
                if cat == "High Risk":
                    return {"color": "#dc2626", "weight": 4, "opacity": 0.9}
                elif cat == "Moderate Risk":
                    return {"color": "#f59e0b", "weight": 3, "opacity": 0.8}
                else:
                    return {"color": "#10b981", "weight": 2, "opacity": 0.5}

            geojson_layer = folium.GeoJson(
                disp_gdf,
                name="Highway Segments",
                style_function=style_fn,
                tooltip=folium.GeoJsonTooltip(
                    fields=["survey_rank", "segment_id", "highway_name", "exposure_score", "risk_category"],
                    aliases=["Rank #:", "Segment:", "Road:", "Score:", "Risk:"],
                    localize=True,
                ),
            )
            geojson_layer.add_to(m_screen)

            # Highlight selected segment prominently if present
            if selected_seg_id:
                sel_match = filtered_stretches[filtered_stretches["segment_id"] == selected_seg_id]
                if not sel_match.empty:
                    sel_row = sel_match.iloc[0]
                    sel_geom = sel_row.geometry
                    if sel_geom is not None and sel_geom.geom_type == "LineString":
                        coords = [[pt[1], pt[0]] for pt in sel_geom.coords]
                        folium.PolyLine(
                            locations=coords,
                            color="#2563eb",
                            weight=7,
                            opacity=1.0,
                            tooltip=f"Selected: {selected_seg_id}",
                        ).add_to(m_screen)
                        m_lat = float(sel_row.get("center_lat", sel_geom.centroid.y if sel_geom else 22.0))
                        m_lon = float(sel_row.get("center_lon", sel_geom.centroid.x if sel_geom else 80.0))
                        folium.Marker(
                            location=[m_lat, m_lon],
                            icon=folium.Icon(color="blue", icon="info-sign"),
                            tooltip=f"Selected: {selected_seg_id} (Rank #{sel_row['survey_rank']})",
                        ).add_to(m_screen)

        st_folium(m_screen, width=650, height=520, returned_objects=[])
        render_map_footnote(
            "OpenStreetMap Overpass API (highways: motorway, trunk, primary) & GBIF MP mammal occurrences.",
            "Labels read 'Predicted exposure, not observed roadkill'. Colors: Red (High Risk, Score >= 65), Amber (Moderate Risk, Score 40-64), Green (Low Risk, Score < 40).",
        )

with col_table:
    with st.container(border=True):
        st.markdown("#### :material/table_chart: Prioritized Survey Stretches")
        display_df = filtered_stretches[[
            "survey_rank", "segment_id", "highway_name", "ref", "exposure_score", "risk_category"
        ]].rename(columns={
            "survey_rank": "Rank",
            "segment_id": "Segment",
            "highway_name": "Highway",
            "ref": "Route Ref",
            "exposure_score": "Score",
            "risk_category": "Risk",
        })
        st.dataframe(display_df.head(25), hide_index=True, width="stretch")

        # CSV Export Button
        export_candidates = [
            "survey_rank", "segment_id", "highway_name", "highway_type", "ref",
            "length_m", "pa_proximity_score", "wildlife_density_score",
            "road_vulnerability_score", "exposure_score", "risk_category",
            "screening_label", "center_lat", "center_lon"
        ]
        avail_export_cols = [c for c in export_candidates if c in filtered_stretches.columns]
        csv_data = filtered_stretches[avail_export_cols].to_csv(index=False)

        st.download_button(
            label="Export Ranked Stretches (CSV)",
            icon=":material/download:",
            data=csv_data,
            file_name="wildguard_kanha_pench_ranked_stretches.csv",
            mime="text/csv",
            width="stretch",
            help="Export full ranked candidate stretches for field survey planning",
        )

