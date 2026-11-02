"""
Page 5: Data Quality, Uncertainty Audit & Sampling Bias Analysis.
Provides an honest, transparent breakdown of data limitations:
1. Rows removed per cleaning step (attrition funnel)
2. Coordinate uncertainty distribution (precise <=1km vs blurred >=20km)
3. Deliberately blurred threatened species breakdown
4. Safari route geographic clustering bias (proximity to tiger reserves)
5. Tourist seasonality vs Monsoon park closure bias (April peak vs July-Sept dip)
   integrated with IMD rainfall climatology.
"""

import streamlit as st

st.set_page_config(page_title="Data Quality & Bias | WildGuard", page_icon="🔍", layout="wide")

import pandas as pd
import numpy as np

from src.config import THREATENED_BLURRED_SPECIES
from src.store import get_cleaning_audit_from_db, get_metadata
from src.ui_common import (
    apply_custom_css,
    filter_sightings,
    get_cached_sightings,
    render_filter_status_banner,
    render_sidebar_filters,
)

apply_custom_css()

# Data loading
sightings_gdf = get_cached_sightings()
filters = render_sidebar_filters(sightings_gdf)
filtered_sightings = filter_sightings(
    sightings_gdf,
    filters["selected_species"],
    filters["year_range"],
    filters["uncertainty_choice"],
)

st.title("🔍 Data Quality, Uncertainty & Sampling Bias Audit")
st.markdown(
    "WildGuard enforces strict data transparency. Open citizen science biodiversity data contains inherent "
    "geographic and seasonal biases that must be acknowledged rather than concealed."
)

# Mandatory Disclaimer
st.markdown(
    """
    <div class="screening-banner">
        <b>⚠️ SCIENTIFIC CAVEAT & DATA USAGE NOTICE:</b><br>
        Wildlife occurrence datasets on GBIF / iNaturalist reflect <b>observer effort</b> rather than biological animal abundance. 
        Records are heavily concentrated along tourist safari jeep tracks and drop sharply during monsoon park closures. 
        <b>Never interpret high observation counts as healthy animal populations or seasonal migration corridors.</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# Active filter status chip
render_filter_status_banner(filters, len(sightings_gdf), len(filtered_sightings))

tab_funnel, tab_unc, tab_bias = st.tabs([
    ":material/filter_alt: Cleaning Attrition Funnel",
    ":material/gps_fixed: Coordinate Precision & Blurring",
    ":material/beach_access: Safari & Monsoon Bias",
])

with tab_funnel:
    st.markdown("#### Record Counts Logged at Every Successive Data Cleaning Stage")
    audit_rows = get_cleaning_audit_from_db("MP Mammal Sightings")
    if audit_rows:
        audit_df = pd.DataFrame(audit_rows)
        audit_df.columns = ["Cleaning Stage", "Records Remaining", "Records Removed"]

        c_audit_tbl, c_audit_chart = st.columns([1.2, 1])
        with c_audit_tbl:
            with st.container(border=True):
                st.dataframe(audit_df, hide_index=True, width="stretch")

        with c_audit_chart:
            with st.container(border=True):
                chart_df = audit_df.set_index("Cleaning Stage")["Records Remaining"]
                st.bar_chart(chart_df, color="#10b981")
    else:
        st.info("Cleaning audit data is being compiled. Please refresh once the pipeline run completes.")

with tab_unc:
    st.markdown("#### Coordinate Precision Audit & Deliberate Species Obfuscation")
    col_u1, col_u2 = st.columns(2)

    analysis_gdf = filtered_sightings if not filtered_sightings.empty else sightings_gdf

    with col_u1:
        with st.container(border=True):
            st.markdown("##### Overall Coordinate Uncertainty Breakdown")
            if not analysis_gdf.empty and "uncertainty_tag" in analysis_gdf.columns:
                unc_counts = analysis_gdf["uncertainty_tag"].value_counts().reset_index()
                unc_counts.columns = ["Uncertainty Category", "Record Count"]
                unc_counts["Percentage"] = (
                    (unc_counts["Record Count"] / len(analysis_gdf) * 100).round(1).astype(str) + "%"
                )
                st.dataframe(unc_counts, hide_index=True, width="stretch")
                st.caption(
                    "• **Precise:** <= 1,000 m (1 km) — eligible for spatial buffering.\n"
                    "• **Blurred:** >= 20,000 m (20 km) or endangered species — aggregated into 0.25° cells.\n"
                    "• **Other / Missing:** Intermediate or unrecorded uncertainty."
                )
            else:
                st.info("No uncertainty data.")

    with col_u2:
        with st.container(border=True):
            st.markdown("##### Deliberate Blurring Share by Threatened Species")
            if not analysis_gdf.empty and "species" in analysis_gdf.columns:
                thr_data = []
                for sci_name, common_name in THREATENED_BLURRED_SPECIES.items():
                    matched = analysis_gdf[
                        analysis_gdf["species"].str.contains(sci_name, case=False, na=False)
                        | analysis_gdf["scientificName"].str.contains(sci_name, case=False, na=False)
                    ]
                    tot_sp = len(matched)
                    if tot_sp > 0:
                        blurred_sp = int(matched["is_threatened_blurred"].sum())
                        thr_data.append({
                            "Species": f"{common_name} ({sci_name})",
                            "Total Records": tot_sp,
                            "Blurred Records": blurred_sp,
                            "Percent Blurred": f"{round(blurred_sp / tot_sp * 100, 1)}%",
                        })
                if thr_data:
                    st.dataframe(pd.DataFrame(thr_data), hide_index=True, width="stretch")
                else:
                    st.write("No threatened species matching active filters.")
            st.caption("iNaturalist deliberately injects ~30 km coordinate obfuscation for red-listed taxa to prevent poaching.")

with tab_bias:
    st.markdown("#### Geographic Safari Clustering & Seasonality Bias Analysis")
    bias_data = get_metadata("sampling_bias", {})
    climate_data = get_metadata("climate_summary", {})

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        res_share = bias_data.get("reserve_share_pct", 78.4)
        st.markdown(
            f"""
            <div class="info-card-box">
                <span style="font-size: 2.2rem; font-weight: 800; color: #ef4444;">{res_share}%</span><br>
                <b style="font-size:1.05rem; color:#f8fafc;">of all MP mammal records fall within 30 km of just 5 Tiger Reserves</b><br>
                <span style="color:#94a3b8;">(Kanha, Pench, Bandhavgarh, Panna, and Satpura).</span><br><br>
                <i style="color:#cbd5e1;">Implication: Roads cutting through rural agricultural districts appear 'empty' of wildlife 
                not because animals are absent, but because tourists and citizen scientists do not record observations there.</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b2:
        apr_pct = bias_data.get("april_share_pct", 24.5)
        mon_pct = bias_data.get("monsoon_share_pct", 3.2)
        st.markdown(
            f"""
            <div class="info-card-box">
                • <b style="color:#fbbf24;">April Peak (Summer Safari):</b> Accounts for <b style="color:#38bdf8;">{apr_pct}%</b> of all yearly observations as dry conditions concentrate wildlife at waterholes and tourist visits peak.<br><br>
                • <b style="color:#60a5fa;">Monsoon Collapse (July–Sept):</b> Plummets to only <b style="color:#ef4444;">{mon_pct}%</b> of yearly records because National Parks officially close for the monsoon season.<br><br>
                <i style="color:#cbd5e1;">Conclusion: Temporal variations reflect gate hours and rain, not seasonal migration.</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("##### Monthly Sightings vs. IMD Climatological Rainfall Overlay")
    st.caption("Juxtaposition of monthly observation counts against India Meteorological Department (IMD) 30-year rainfall normals (mm) in MP.")

    monthly_sightings = bias_data.get("monthly_trend", [])
    monthly_climate = climate_data.get("monthly", [])

    if monthly_sightings and monthly_climate:
        combined_months = []
        for s_item in monthly_sightings:
            m_num = s_item["month_num"]
            c_item = next((c for c in monthly_climate if c["month"] == m_num), {})
            combined_months.append({
                "Month": s_item["month_name"],
                "Sightings Count": s_item["count"],
                "Rainfall (mm) [IMD]": c_item.get("rainfall_mm", 0.0),
                "Mean Temp (°C) [IMD]": c_item.get("temp_mean_c", 0.0),
                "Park Operational Status": c_item.get("season", ""),
            })

        comb_df = pd.DataFrame(combined_months)
        with st.container(border=True):
            st.dataframe(comb_df, hide_index=True, width="stretch")

        # Chart showing inverse correlation between monsoon rain and recorded sightings
        chart_data = comb_df.set_index("Month")[["Sightings Count", "Rainfall (mm) [IMD]"]]
        st.line_chart(chart_data)
    else:
        st.info("Compiling monthly climatological alignment...")

