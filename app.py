import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Parking Intelligence Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
  [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 18px 22px; text-align: center;
  }
  .metric-val { font-size: 2rem; font-weight: 700; color: #58a6ff; }
  .metric-lbl { font-size: 0.78rem; color: #8b949e; margin-top: 4px; text-transform: uppercase; letter-spacing: .06em; }
  .tier-critical { color: #f85149; font-weight: 700; }
  .tier-high     { color: #e3a119; font-weight: 700; }
  .tier-medium   { color: #f0c21f; font-weight: 600; }
  .tier-low      { color: #3fb950; font-weight: 600; }
  .rec-box {
    background: #0d2235; border-left: 4px solid #58a6ff;
    padding: 14px 18px; border-radius: 6px; margin-top: 10px;
  }
  h1, h2, h3 { color: #e6edf3 !important; }
  .stSelectbox label, .stSlider label { color: #c9d1d9 !important; }
  [data-testid="stMetricValue"] { color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_zones():
    df = pd.read_csv("enforcement_priority_zones.csv")
    return df

@st.cache_data
def load_violations():
    df = pd.read_csv("jan_to_may_police_violation_anonymized791b166__1_.csv")
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], format="mixed", utc=True)
    df["hour"] = df["created_datetime"].dt.hour
    df["day_of_week"] = df["created_datetime"].dt.day_name()
    df["date"] = df["created_datetime"].dt.date
    df["month"] = df["created_datetime"].dt.strftime("%b %Y")
    return df

zones = load_zones()
viol  = load_violations()

# ── Helpers ────────────────────────────────────────────────────────────────────
TIER_ORDER = {"🔴 CRITICAL": 0, "🟠 HIGH": 1, "🟡 MEDIUM": 2, "🟢 LOW": 3}

def tier_color(tier):
    if "CRITICAL" in tier: return "#f85149"
    if "HIGH"     in tier: return "#e3a119"
    if "MEDIUM"   in tier: return "#f0c21f"
    return "#3fb950"

def recommend(crs):
    if crs >= 50:
        return [
            "🚨 Deploy towing units immediately",
            "📡 Install real-time monitoring cameras",
            "🚧 Erect temporary no-parking barriers during peak hours",
            "👮 Increase patrol frequency by 60%",
            "📢 Issue public warning via traffic broadcast",
        ]
    elif crs >= 40:
        return [
            "🚔 Deploy additional patrol vehicles",
            "⚠️ Issue formal warning notices to repeat offenders",
            "📋 Conduct peak-hour targeted enforcement",
            "🔭 Monitor with CCTV during evening peak",
        ]
    elif crs >= 25:
        return [
            "📝 Routine enforcement patrols",
            "🗺️ Update signage in the zone",
            "📊 Continue weekly monitoring",
        ]
    else:
        return [
            "✅ Standard periodic inspection",
            "📁 Log for trend analysis",
        ]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Navigation")
    page = st.radio("", [
        "📊 Executive Overview",
        "🗺️ Hotspot Map",
        "🏆 Enforcement Priorities",
        "🔍 Zone Explorer",
        "📈 Trend Analytics",
        "⚡ Impact Simulator",
    ])
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"🗂️ {len(viol):,} violations")
    st.caption(f"📍 {len(zones)} enforcement zones")
    st.caption(f"📅 Nov 2023 – Apr 2024")
    st.markdown("---")
    st.caption("Flipkart Gridlock Hackathon · Topic 1")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.markdown("# 🚦 AI Parking Intelligence Dashboard")
    st.markdown("##### Bengaluru Traffic Enforcement · Powered by DBSCAN + LightGBM")
    st.markdown("---")

    # KPI row
    critical = zones[zones["enforcement_tier"].str.contains("HIGH|CRITICAL")]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(viol):,}</div><div class="metric-lbl">Total Violations</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(zones)}</div><div class="metric-lbl">Hotspot Zones</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(critical)}</div><div class="metric-lbl">High-Risk Zones</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{zones["crs_predicted"].mean():.1f}</div><div class="metric-lbl">Avg Risk Score</div></div>', unsafe_allow_html=True)
    with c5:
        peak_hour = viol["hour"].mode()[0]
        st.markdown(f'<div class="metric-card"><div class="metric-val">{peak_hour}:00</div><div class="metric-lbl">Peak Violation Hour</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top-risk zones bar chart + vehicle breakdown
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("### Top 15 High-Risk Enforcement Zones")
        top15 = zones.nsmallest(15, "enforcement_rank").copy()
        top15["label"] = "Zone " + top15["cluster_id"].astype(str)
        fig = px.bar(
            top15, x="crs_predicted", y="label", orientation="h",
            color="crs_predicted",
            color_continuous_scale=["#3fb950","#f0c21f","#e3a119","#f85149"],
            labels={"crs_predicted": "Congestion Risk Score", "label": "Zone"},
            text=top15["crs_predicted"].round(1),
        )
        fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0),
            height=380,
        )
        fig.update_traces(textposition="outside", textfont_color="#e6edf3")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### Vehicle Type Distribution")
        vc = viol["vehicle_type"].value_counts().head(6)
        fig2 = px.pie(
            values=vc.values, names=vc.index,
            color_discrete_sequence=["#58a6ff","#3fb950","#e3a119","#f85149","#bc8cff","#79c0ff"],
            hole=0.5,
        )
        fig2.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(font_color="#c9d1d9"), height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Tier summary
    st.markdown("### Zone Risk Distribution")
    tier_counts = zones["enforcement_tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]
    tier_counts["Color"] = tier_counts["Tier"].apply(tier_color)
    fig3 = px.bar(
        tier_counts, x="Tier", y="Count", color="Tier",
        color_discrete_map={t: tier_color(t) for t in tier_counts["Tier"]},
        text="Count",
    )
    fig3.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#c9d1d9", showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0), height=260,
    )
    fig3.update_traces(textposition="outside", textfont_color="#e6edf3")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Citywide Action Plan ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Citywide Action Plan — Resource Deployment Summary")
    st.markdown(
        "<span style='color:#8b949e;'>Aggregated from per-zone recommendations, "
        "based on each zone's Congestion Risk Score tier.</span>",
        unsafe_allow_html=True,
    )

    tow_zones    = zones[zones["crs_predicted"] >= 50]
    patrol_zones = zones[(zones["crs_predicted"] >= 40) & (zones["crs_predicted"] < 50)]
    routine_zones = zones[(zones["crs_predicted"] >= 25) & (zones["crs_predicted"] < 40)]

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            f'<div class="metric-card" style="border-left:4px solid #f85149;">'
            f'<div class="metric-val">{len(tow_zones)}</div>'
            f'<div class="metric-lbl">Towing Units Needed</div>'
            f'<div style="color:#8b949e;font-size:0.78rem;margin-top:4px;">'
            f'1 per HIGH-tier zone (CRS ≥ 50) · covers {int(tow_zones["violation_count"].sum()):,} violations</div>'
            f'</div>', unsafe_allow_html=True)
    with a2:
        st.markdown(
            f'<div class="metric-card" style="border-left:4px solid #e3a119;">'
            f'<div class="metric-val">{len(patrol_zones)}</div>'
            f'<div class="metric-lbl">Extra Patrol Vehicles</div>'
            f'<div style="color:#8b949e;font-size:0.78rem;margin-top:4px;">'
            f'1 per zone, CRS 40–50 · covers {int(patrol_zones["violation_count"].sum()):,} violations</div>'
            f'</div>', unsafe_allow_html=True)
    with a3:
        st.markdown(
            f'<div class="metric-card" style="border-left:4px solid #f0c21f;">'
            f'<div class="metric-val">{len(routine_zones)}</div>'
            f'<div class="metric-lbl">Zones on Routine Watch</div>'
            f'<div style="color:#8b949e;font-size:0.78rem;margin-top:4px;">'
            f'CRS 25–40 · signage + weekly monitoring only</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:#8b949e;font-size:0.85rem;'>"
        f"💡 This is a deployment <b>starting point</b> derived directly from the zone tiers above — "
        f"see the <b>Zone Explorer</b> page for the full recommendation breakdown per individual zone, "
        f"or the <b>Impact Simulator</b> to model how added patrols affect a zone's projected risk score."
        f"</span>", unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HOTSPOT MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Hotspot Map":
    st.markdown("# 🗺️ Interactive Hotspot Map")
    st.markdown("DBSCAN-detected parking violation clusters across Bengaluru, colored by Congestion Risk Score.")

    col1, col2 = st.columns([1, 3])
    with col1:
        map_type = st.radio("Layer", ["Heatmap", "Cluster Markers", "Both"])
        show_tier = st.multiselect(
            "Filter by Tier",
            options=zones["enforcement_tier"].unique().tolist(),
            default=zones["enforcement_tier"].unique().tolist(),
        )

    filtered = zones[zones["enforcement_tier"].isin(show_tier)]

    m = folium.Map(
        location=[12.97, 77.59],
        zoom_start=11,
        tiles="CartoDB dark_matter",
    )

    if map_type in ["Heatmap", "Both"]:
        heat_data = [[r.centre_lat, r.centre_lon, r.crs_predicted] for _, r in filtered.iterrows()]
        HeatMap(heat_data, radius=25, blur=18, min_opacity=0.4,
                gradient={"0.2": "#3fb950", "0.5": "#f0c21f", "0.8": "#e3a119", "1.0": "#f85149"}).add_to(m)

    if map_type in ["Cluster Markers", "Both"]:
        mc = MarkerCluster().add_to(m)
        for _, r in filtered.iterrows():
            color = tier_color(r["enforcement_tier"])
            folium.CircleMarker(
                location=[r["centre_lat"], r["centre_lon"]],
                radius=max(5, r["violation_count"] / 200),
                color=color, fill=True, fill_color=color, fill_opacity=0.75,
                tooltip=folium.Tooltip(
                    f"<b>Zone {r['cluster_id']}</b><br>"
                    f"Violations: {r['violation_count']:,}<br>"
                    f"CRS: {r['crs_predicted']:.1f}<br>"
                    f"Tier: {r['enforcement_tier']}<br>"
                    f"Rank: #{int(r['enforcement_rank'])}"
                ),
                popup=folium.Popup(
                    f"<b>Zone {r['cluster_id']}</b><br>"
                    f"Density: {r['violation_density']:.0f}/km²<br>"
                    f"Peak-hour frac: {r['peak_hour_frac']:.1%}<br>"
                    f"Dist to centre: {r['dist_to_centre_km']:.1f} km",
                    max_width=220,
                ),
            ).add_to(mc)

    with col2:
        st_folium(m, width=None, height=620, returned_objects=[])

    st.caption(f"Showing {len(filtered)} of {len(zones)} zones")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ENFORCEMENT PRIORITIES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Enforcement Priorities":
    st.markdown("# 🏆 Enforcement Priority Table")

    col1, col2, col3 = st.columns(3)
    with col1:
        tier_filter = st.multiselect(
            "Tier", zones["enforcement_tier"].unique().tolist(),
            default=zones["enforcement_tier"].unique().tolist()
        )
    with col2:
        top_n = st.selectbox("Show top N zones", [10, 20, 50, 157], index=1)
    with col3:
        sort_by = st.selectbox("Sort by", ["enforcement_rank", "crs_predicted", "violation_count"])

    display = (
        zones[zones["enforcement_tier"].isin(tier_filter)]
        .sort_values(sort_by if sort_by == "enforcement_rank" else sort_by, ascending=(sort_by == "enforcement_rank"))
        .head(top_n)
        [["enforcement_rank","cluster_id","violation_count","crs_predicted",
          "violation_density","peak_hour_frac","dist_to_centre_km","enforcement_tier"]]
        .rename(columns={
            "enforcement_rank": "Rank",
            "cluster_id": "Zone ID",
            "violation_count": "Violations",
            "crs_predicted": "CRS Score",
            "violation_density": "Density/km²",
            "peak_hour_frac": "Peak Hour %",
            "dist_to_centre_km": "Dist (km)",
            "enforcement_tier": "Tier",
        })
    )
    display["Peak Hour %"] = (display["Peak Hour %"] * 100).round(1).astype(str) + "%"
    display["CRS Score"] = display["CRS Score"].round(2)
    display["Density/km²"] = display["Density/km²"].round(0).astype(int)
    display["Dist (km)"] = display["Dist (km)"].round(2)

    st.dataframe(display, use_container_width=True, hide_index=True, height=560)
    st.caption(f"Showing {len(display)} zones | Sorted by {sort_by}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ZONE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Zone Explorer":
    st.markdown("# 🔍 Zone Explorer")

    zone_options = zones.sort_values("enforcement_rank")["cluster_id"].tolist()
    selected = st.selectbox(
        "Select Zone (sorted by enforcement priority)",
        zone_options,
        format_func=lambda z: f"Zone {z} — Rank #{int(zones.loc[zones['cluster_id']==z,'enforcement_rank'].values[0])}"
    )

    r = zones[zones["cluster_id"] == selected].iloc[0]

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Congestion Risk Score", f"{r['crs_predicted']:.2f}")
    c2.metric("Violations", f"{int(r['violation_count']):,}")
    c3.metric("Enforcement Rank", f"#{int(r['enforcement_rank'])}")
    c4.metric("Tier", r["enforcement_tier"])

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Zone Characteristics")
        chars = {
            "Violation Density": f"{r['violation_density']:.0f} /km²",
            "Peak Hour Fraction": f"{r['peak_hour_frac']:.1%}",
            "Morning Peak Frac": f"{r['morning_peak_frac']:.1%}",
            "Evening Peak Frac": f"{r['evening_peak_frac']:.1%}",
            "Weekend Fraction": f"{r['weekend_frac']:.1%}",
            "Distance to City Centre": f"{r['dist_to_centre_km']:.2f} km",
            "Cluster Spread": f"{r['spread_km']:.3f} km",
            "Unique Active Days": f"{int(r['unique_days'])}",
            "Recurrence Score": f"{r['recurrence_score']:.3f}",
        }
        for k, v in chars.items():
            col_k, col_v = st.columns([2, 1])
            col_k.write(k)
            col_v.write(f"**{v}**")

        st.markdown("### Vehicle Mix")
        veh_cols = [c for c in zones.columns if c.startswith("viol_") and c.endswith("_frac")]
        veh_labels = [c.replace("viol_","").replace("_frac","").replace("_"," ") for c in veh_cols]
        veh_vals = [r[c] * 100 for c in veh_cols]
        fig_veh = px.bar(
            x=veh_labels, y=veh_vals,
            labels={"x": "Vehicle Type", "y": "% of Violations"},
            color=veh_vals,
            color_continuous_scale=["#58a6ff","#3fb950","#f0c21f","#e3a119","#f85149"],
        )
        fig_veh.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0), height=240,
        )
        st.plotly_chart(fig_veh, use_container_width=True)

    with col_right:
        st.markdown("### Mini Map")
        mini_map = folium.Map(location=[r["centre_lat"], r["centre_lon"]], zoom_start=14, tiles="CartoDB dark_matter")
        folium.CircleMarker(
            [r["centre_lat"], r["centre_lon"]],
            radius=18, color=tier_color(r["enforcement_tier"]),
            fill=True, fill_color=tier_color(r["enforcement_tier"]), fill_opacity=0.6,
            tooltip=f"Zone {selected}",
        ).add_to(mini_map)
        st_folium(mini_map, width=None, height=280, returned_objects=[])

        st.markdown("### Enforcement Recommendation")
        recs = recommend(r["crs_predicted"])
        rec_html = "".join(f"<div style='margin:6px 0'>{rec}</div>" for rec in recs)
        st.markdown(f'<div class="rec-box">{rec_html}</div>', unsafe_allow_html=True)

        # CRS gauge
        st.markdown("### Risk Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=r["crs_predicted"],
            number={"font": {"color": "#e6edf3", "size": 36}},
            gauge={
                "axis": {"range": [0, 60], "tickcolor": "#8b949e"},
                "bar": {"color": tier_color(r["enforcement_tier"]), "thickness": 0.25},
                "bgcolor": "#161b22",
                "steps": [
                    {"range": [0, 25],  "color": "#0d1117"},
                    {"range": [25, 40], "color": "#0d2030"},
                    {"range": [40, 50], "color": "#1a1a00"},
                    {"range": [50, 60], "color": "#1f0000"},
                ],
                "threshold": {"line": {"color": "#f85149", "width": 3}, "value": 50},
            },
            title={"text": "CRS", "font": {"color": "#8b949e"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0d1117", font_color="#c9d1d9",
            margin=dict(l=20, r=20, t=30, b=10), height=200,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — TREND ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend Analytics":
    st.markdown("# 📈 Trend Analytics")

    tab1, tab2, tab3, tab4 = st.tabs(["Hourly Pattern", "Daily Trend", "Day of Week", "Monthly"])

    with tab1:
        st.markdown("### Violations by Hour of Day")
        hourly = viol.groupby("hour").size().reset_index(name="count")
        fig = px.area(
            hourly, x="hour", y="count",
            labels={"hour": "Hour of Day", "count": "Violations"},
            color_discrete_sequence=["#58a6ff"],
        )
        fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", margin=dict(l=0, r=0, t=10, b=0), height=360,
            xaxis=dict(tickmode="linear", dtick=1),
        )
        fig.update_traces(fill="tozeroy", fillcolor="rgba(88,166,255,0.15)")
        st.plotly_chart(fig, use_container_width=True)
        peak = int(hourly.loc[hourly["count"].idxmax(), "hour"])
        st.info(f"🕐 Peak violation hour: **{peak}:00 – {peak+1}:00** with {int(hourly['count'].max()):,} violations")

    with tab2:
        st.markdown("### Daily Violation Volume + Rolling 7-day Average")
        daily = viol.groupby("date").size().reset_index(name="count")
        daily["rolling_7"] = daily["count"].rolling(7).mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=daily["date"], y=daily["count"], name="Daily", marker_color="#30363d"))
        fig2.add_trace(go.Scatter(x=daily["date"], y=daily["rolling_7"], name="7-day avg", line=dict(color="#58a6ff", width=2)))
        fig2.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#c9d1d9",
            legend=dict(bgcolor="#161b22"), margin=dict(l=0, r=0, t=10, b=0), height=360,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### Violations by Day of Week")
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = viol.groupby("day_of_week").size().reset_index(name="count")
        dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=dow_order, ordered=True)
        dow = dow.sort_values("day_of_week")
        fig3 = px.bar(
            dow, x="day_of_week", y="count",
            color="count", color_continuous_scale=["#3fb950","#f0c21f","#f85149"],
            labels={"day_of_week": "", "count": "Violations"},
            text=dow["count"],
        )
        fig3.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0), height=340,
        )
        fig3.update_traces(textposition="outside", textfont_color="#e6edf3")
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.markdown("### Monthly Violation Trend")
        monthly = viol.groupby("month").size().reset_index(name="count")
        # Sort chronologically
        monthly["sort_key"] = pd.to_datetime(monthly["month"], format="%b %Y")
        monthly = monthly.sort_values("sort_key")
        fig4 = px.bar(
            monthly, x="month", y="count",
            color="count", color_continuous_scale=["#3fb950","#58a6ff","#f85149"],
            labels={"month": "Month", "count": "Violations"},
            text="count",
        )
        fig4.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0), height=340,
        )
        fig4.update_traces(textposition="outside", textfont_color="#e6edf3")
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — IMPACT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Impact Simulator":
    st.markdown("# ⚡ What-If Impact Simulator")
    st.markdown("Simulate the effect of enforcement interventions on Congestion Risk Score across zones.")

    col1, col2 = st.columns([1, 1])
    with col1:
        sim_scope = st.radio("Simulation Scope", ["Single Zone", "All Zones"])
        reduction = st.slider("🎯 Violation Reduction Target (%)", 0, 100, 30, step=5)
        enforcement_boost = st.slider("👮 Patrol Frequency Increase (%)", 0, 200, 50, step=10)
        barrier_impact = st.slider("🚧 Temporary Barrier Coverage (%)", 0, 100, 20, step=5)
        st.markdown("---")
        st.markdown("**Intervention Model**")
        st.caption("Combined reduction = violation reduction × (1 + 0.3×patrol_boost/100 + 0.15×barrier_coverage/100)")

        effective_reduction = min(
            reduction * (1 + 0.3 * enforcement_boost/100 + 0.15 * barrier_impact/100),
            95.0
        )
        st.info(f"📉 Effective combined reduction: **{effective_reduction:.1f}%**")

    with col2:
        if sim_scope == "Single Zone":
            zone_id = st.selectbox(
                "Select Zone",
                zones.sort_values("enforcement_rank")["cluster_id"].tolist(),
                format_func=lambda z: f"Zone {z} (Rank #{int(zones.loc[zones['cluster_id']==z,'enforcement_rank'].values[0])})"
            )
            r = zones[zones["cluster_id"] == zone_id].iloc[0]
            current_crs = r["crs_predicted"]
            projected_crs = current_crs * (1 - effective_reduction / 100)
            improvement = current_crs - projected_crs

            st.markdown("### Simulation Result")
            m1, m2, m3 = st.columns(3)
            m1.metric("Current CRS", f"{current_crs:.2f}")
            m2.metric("Projected CRS", f"{projected_crs:.2f}", delta=f"-{improvement:.2f}", delta_color="inverse")
            m3.metric("Risk Reduction", f"{improvement:.2f}")

            # Before / After gauge
            fig_sim = go.Figure()
            for val, label, color in [
                (current_crs, "Current", "#f85149"),
                (projected_crs, "Projected", "#3fb950"),
            ]:
                fig_sim.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=val,
                    number={"font": {"color": "#e6edf3", "size": 28}},
                    gauge={
                        "axis": {"range": [0, 60]},
                        "bar": {"color": color, "thickness": 0.25},
                        "bgcolor": "#161b22",
                        "steps": [{"range": [0, 60], "color": "#0d1117"}],
                    },
                    title={"text": label, "font": {"color": "#8b949e"}},
                    domain={"row": 0, "column": 0 if label == "Current" else 1},
                ))
            fig_sim.update_layout(
                grid={"rows": 1, "columns": 2, "pattern": "independent"},
                paper_bgcolor="#0d1117", font_color="#c9d1d9",
                margin=dict(l=10, r=10, t=30, b=10), height=220,
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            recs = recommend(projected_crs)
            rec_html = "".join(f"<div style='margin:5px 0'>{rec}</div>" for rec in recs)
            st.markdown(f'<div class="rec-box"><b>Recommended interventions for projected CRS {projected_crs:.1f}:</b><br>{rec_html}</div>', unsafe_allow_html=True)

        else:
            st.markdown("### All-Zone Impact Preview")
            sim_zones = zones.copy()
            sim_zones["projected_crs"] = sim_zones["crs_predicted"] * (1 - effective_reduction / 100)
            sim_zones["crs_reduction"] = sim_zones["crs_predicted"] - sim_zones["projected_crs"]
            sim_zones["label"] = "Zone " + sim_zones["cluster_id"].astype(str)

            top20 = sim_zones.nsmallest(20, "enforcement_rank")
            fig_all = go.Figure()
            fig_all.add_trace(go.Bar(
                y=top20["label"], x=top20["crs_predicted"],
                name="Current CRS", orientation="h", marker_color="#f85149", opacity=0.7,
            ))
            fig_all.add_trace(go.Bar(
                y=top20["label"], x=top20["projected_crs"],
                name="Projected CRS", orientation="h", marker_color="#3fb950", opacity=0.8,
            ))
            fig_all.update_layout(
                barmode="overlay",
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#c9d1d9",
                yaxis=dict(autorange="reversed"),
                legend=dict(bgcolor="#161b22"),
                margin=dict(l=0, r=0, t=10, b=0), height=500,
                xaxis_title="CRS",
            )
            st.plotly_chart(fig_all, use_container_width=True)

            total_before = sim_zones["crs_predicted"].sum()
            total_after  = sim_zones["projected_crs"].sum()
            st.success(
                f"🏙️ City-wide CRS sum: **{total_before:.1f}** → **{total_after:.1f}** "
                f"(reduction of **{total_before - total_after:.1f}** points, "
                f"**{(total_before - total_after)/total_before*100:.1f}%** improvement)"
            )
