"""
Timber Truss Fire Reliability — Monte Carlo Simulation (Streamlit App)
======================================================================
UI layer over mcs_simulation.py engine.
Presents research-methodology-aligned outputs for the thesis:
  - Reliability Index β and Pf with 95% CI (Clopper-Pearson)
  - FM1-FM8 failure mode breakdown
  - Parametric b × h heatmap
  - Convergence tracking
  - Sensitivity (Spearman rank correlation)
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import scipy.stats as stats

from mcs_simulation import (
    SPECIES_DATA, FIRE_SCENARIOS, TRUSS_CONFIGS,
    run_simulation, run_parametric_sweep,
    analyze_results, compute_sensitivity,
)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Timber Truss Fire Reliability",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# GLOBAL CSS — premium dark glassmorphism
# ─────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  /* Dark base */
  .stApp {
      background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
      color: #e6edf3;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background: rgba(22, 27, 34, 0.95);
      border-right: 1px solid rgba(48, 54, 61, 0.8);
      backdrop-filter: blur(12px);
  }
  section[data-testid="stSidebar"] * {
      color: #e6edf3 !important;
  }

  /* Cards */
  .metric-card {
      background: rgba(22, 27, 34, 0.8);
      border: 1px solid rgba(48, 54, 61, 0.9);
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 12px;
      backdrop-filter: blur(8px);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .metric-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
  }
  .metric-value {
      font-size: 2.2rem;
      font-weight: 700;
      background: linear-gradient(120deg, #58a6ff, #bc8cff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      line-height: 1.1;
  }
  .metric-label {
      font-size: 0.75rem;
      font-weight: 500;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-top: 4px;
  }
  .metric-sub {
      font-size: 0.82rem;
      color: #58a6ff;
      margin-top: 6px;
  }

  /* Status badges */
  .badge-pass {
      display: inline-block;
      background: rgba(35, 134, 54, 0.25);
      color: #3fb950;
      border: 1px solid rgba(63, 185, 80, 0.4);
      border-radius: 20px;
      padding: 3px 12px;
      font-size: 0.78rem;
      font-weight: 600;
  }
  .badge-fail {
      display: inline-block;
      background: rgba(218, 54, 51, 0.2);
      color: #f85149;
      border: 1px solid rgba(248, 81, 73, 0.4);
      border-radius: 20px;
      padding: 3px 12px;
      font-size: 0.78rem;
      font-weight: 600;
  }
  .badge-warn {
      display: inline-block;
      background: rgba(210, 153, 34, 0.2);
      color: #d29922;
      border: 1px solid rgba(210, 153, 34, 0.4);
      border-radius: 20px;
      padding: 3px 12px;
      font-size: 0.78rem;
      font-weight: 600;
  }

  /* Section headers */
  .section-title {
      font-size: 0.7rem;
      font-weight: 600;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin: 16px 0 8px 0;
  }

  /* Header banner */
  .header-banner {
      background: linear-gradient(135deg, rgba(33, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%);
      border: 1px solid rgba(48, 54, 61, 0.8);
      border-radius: 16px;
      padding: 28px 32px;
      margin-bottom: 28px;
      backdrop-filter: blur(12px);
  }
  .header-title {
      font-size: 1.8rem;
      font-weight: 700;
      color: #e6edf3;
      margin: 0 0 6px 0;
  }
  .header-sub {
      font-size: 0.9rem;
      color: #8b949e;
      margin: 0;
  }

  /* REI badge row */
  .rei-row {
      display: flex;
      gap: 8px;
      margin: 8px 0;
  }
  .rei-chip {
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
      background: rgba(22, 27, 34, 0.6);
      border-radius: 10px;
      padding: 4px;
      gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
      border-radius: 8px;
      color: #8b949e !important;
      font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
      background: rgba(88, 166, 255, 0.15) !important;
      color: #58a6ff !important;
  }

  /* DataFrame */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* Button */
  .stButton > button {
      background: linear-gradient(135deg, #1f6feb, #388bfd);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 12px 28px;
      font-weight: 600;
      font-size: 0.95rem;
      width: 100%;
      transition: all 0.2s ease;
      box-shadow: 0 4px 15px rgba(31, 111, 235, 0.35);
  }
  .stButton > button:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(31, 111, 235, 0.5);
  }

  /* Sliders */
  .stSlider [data-testid="stThumbValue"] { color: #58a6ff; }
  .stSelectbox div[data-baseweb="select"] {
      background: rgba(22, 27, 34, 0.8);
      border-color: rgba(48, 54, 61, 0.9);
  }
  hr { border-color: rgba(48, 54, 61, 0.6); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
PLOT_BG = "rgba(13,17,23,0)"
GRID_COLOR = "rgba(48,54,61,0.6)"
ACCENT_BLUE = "#58a6ff"
ACCENT_PURPLE = "#bc8cff"
FM_COLORS = [
    "#58a6ff", "#bc8cff", "#3fb950", "#d29922",
    "#f85149", "#ffa657", "#79c0ff", "#56d364"
]


def plotly_layout(title, xlab="", ylab=""):
    return dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        title=dict(text=title, font=dict(size=15, color="#e6edf3"), x=0),
        xaxis=dict(title=xlab, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                   title_font_color="#8b949e", tickfont_color="#8b949e"),
        yaxis=dict(title=ylab, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                   title_font_color="#8b949e", tickfont_color="#8b949e"),
        font=dict(family="Inter", color="#e6edf3"),
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(bgcolor="rgba(22,27,34,0.7)", bordercolor="rgba(48,54,61,0.5)",
                    borderwidth=1, font_color="#e6edf3"),
    )


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌲 Simulation Controls")
    st.markdown('<div class="section-title">Species & Treatment</div>', unsafe_allow_html=True)

    species_options = {v['name']: k for k, v in SPECIES_DATA.items()}
    species_display = st.selectbox("**Wood Species**", list(species_options.keys()))
    species_key = species_options[species_display]

    treatment_display = st.selectbox("**Treatment**", ["Untreated", "Treated"])
    treatment = "borax" if treatment_display == "Treated" else "untreated"

    st.markdown('<div class="section-title">Structural Configuration</div>', unsafe_allow_html=True)
    truss_options = list(TRUSS_CONFIGS.keys())
    truss_type = st.selectbox("**Truss Configuration**", truss_options)

    st.markdown('<div class="section-title">Fire Scenario</div>', unsafe_allow_html=True)
    scenario_options = {f"({k}) {v['name']}": k for k, v in FIRE_SCENARIOS.items()}
    scenario_display = st.selectbox("**Fire Scenario**", list(scenario_options.keys()))
    scenario_key = scenario_options[scenario_display]

    rei_options = {
        "REI 30 (30 min)": 30,
        "REI 45 (45 min)": 45,
        "REI 60 (60 min)": 60,
    }
    rei_label = st.selectbox("**Required Fire Resistance**", list(rei_options.keys()))
    rei_duration = rei_options[rei_label]

    st.markdown('<div class="section-title">Cross-Section Dimensions (TRUSS)</div>',
                unsafe_allow_html=True)
    st.caption("Applies to all TRUSS_CONFIG members. SPECIES_DATA specimens are fixed.")

    b_range = st.select_slider(
        "**Width b (mm)** — range",
        options=list(range(50, 251, 25)),
        value=(50, 250),
    )
    h_range = st.select_slider(
        "**Depth h (mm)** — range",
        options=list(range(100, 301, 25)),
        value=(100, 300),
    )

    b_values = list(range(b_range[0], b_range[1] + 1, 25))
    h_values = list(range(h_range[0], h_range[1] + 1, 25))

    st.markdown('<div class="section-title">Simulation</div>', unsafe_allow_html=True)
    N = st.select_slider(
        "**Iterations (N)**",
        options=[500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000],
        value=5_000,
    )

    st.markdown("---")
    run_btn = st.button("🚀  Run Simulation", use_container_width=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown(f"""
<div class="header-banner">
  <div class="header-title">🔥 Timber Truss Fire Reliability</div>
  <p class="header-sub">
    Monte Carlo Probabilistic Assessment · Anogeissus leiocarpa &amp; Erythrophleum suaveolens ·
    Failure Modes FM1–FM8 per EN 1995-1-2
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "sweep_df" not in st.session_state:
    st.session_state.sweep_df = None
if "samples_df" not in st.session_state:
    st.session_state.samples_df = None
if "convergence" not in st.session_state:
    st.session_state.convergence = None
if "summary" not in st.session_state:
    st.session_state.summary = None

# ─────────────────────────────────────────
# RUN SIMULATION
# ─────────────────────────────────────────
if run_btn:
    total_combos = len(b_values) * len(h_values)

    with st.spinner(f"Running {N:,} iterations × {total_combos} b/h combinations…"):
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def on_progress(current, total):
            progress_bar.progress(current / total)
            status_text.markdown(
                f"<span style='color:#8b949e; font-size:0.85rem;'>"
                f"Sweep {current}/{total} (b={b_values[(current-1)//len(h_values)]}mm, "
                f"h={h_values[(current-1)%len(h_values)]}mm)</span>",
                unsafe_allow_html=True
            )

        sweep_df = run_parametric_sweep(
            N=N,
            species_key=species_key,
            scenario_key=scenario_key,
            treatment=treatment,
            truss_type=truss_type,
            rei_duration=rei_duration,
            b_values=b_values,
            h_values=h_values,
            progress_callback=on_progress,
        )
        progress_bar.empty()
        status_text.empty()

        # Also run a single "nominal" simulation (mid b/h) with convergence + sensitivity
        b_nom = b_values[len(b_values) // 2]
        h_nom = h_values[len(h_values) // 2]

        res_df, samples_df, convergence = run_simulation(
            N=N,
            species_key=species_key,
            scenario_key=scenario_key,
            treatment=treatment,
            truss_type=truss_type,
            b_override=b_nom,
            h_override=h_nom,
            rei_duration=rei_duration,
            sensitivity=True,
            track_convergence=True,
        )
        scenario_label = f"{FIRE_SCENARIOS[scenario_key]['name']} (REI {rei_duration})"
        summary = analyze_results(res_df, scenario_label,
                                  SPECIES_DATA[species_key]['name'], treatment)

    st.session_state.sweep_df   = sweep_df
    st.session_state.samples_df = samples_df
    st.session_state.convergence = convergence
    st.session_state.summary    = summary
    st.session_state.b_nom      = b_nom
    st.session_state.h_nom      = h_nom

# ─────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────
if st.session_state.summary is None:
    st.info("👈  Configure the parameters in the sidebar and click **Run Simulation** to begin.")
    st.stop()

summary    = st.session_state.summary
sweep_df   = st.session_state.sweep_df
samples_df = st.session_state.samples_df
convergence = st.session_state.convergence
b_nom      = st.session_state.b_nom
h_nom      = st.session_state.h_nom

# ── KPI strip ─────────────────────────────
def beta_badge(beta_val):
    if beta_val >= 3.8:
        return '<span class="badge-pass">RELIABLE</span>'
    elif beta_val >= 2.5:
        return '<span class="badge-warn">MARGINAL</span>'
    else:
        return '<span class="badge-fail">CRITICAL</span>'

pf = summary['Pf_System']
beta = summary['Beta_System']
n_fail = summary['Failures']

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Reliability Index β (nominal b={b_nom}, h={h_nom})</div>
      <div class="metric-value">{beta:.3f}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    pf_str = f"{pf:.3e}" if pf > 0 else "< 1/N"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Probability of Failure Pf</div>
      <div class="metric-value">{pf_str}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Iterations / Failures</div>
      <div class="metric-value">{summary['N']:,}</div>
      <div class="metric-sub">{n_fail:,} failures recorded</div>
    </div>""", unsafe_allow_html=True)
with col4:
    fm_cols = [c for c in summary if c.endswith('%')]
    dominant_fm = max(fm_cols, key=lambda c: summary[c]) if fm_cols else "—"
    dom_pct = summary.get(dominant_fm, 0)
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Dominant Failure Mode</div>
      <div class="metric-value" style="font-size:1.3rem">{dominant_fm.replace('%','')}</div>
      <div class="metric-sub">{dom_pct:.1f}% of failures</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Summary Table",
    "📊 Failure Modes",
    "🗺️ Parametric Heatmap",
    "📈 Convergence",
    "🎯 Sensitivity",
])

# ─────────────────────────────────────────
# TAB 1 — SUMMARY TABLE (per methodology Table)
# ─────────────────────────────────────────
with tab1:
    st.subheader("Reliability Summary — All b/h Combinations")
    st.caption(
        f"**{FIRE_SCENARIOS[scenario_key]['name']}** · {SPECIES_DATA[species_key]['name']} · "
        f"{treatment.capitalize()} · {truss_type} · REI {rei_duration}"
    )

    display_df = sweep_df[["b", "h", "Pf_System", "Beta_System", "Failures"]].copy()
    display_df.columns = [
        "b (mm)", "h (mm)", "Pf", "β", "Failures"
    ]

    # Format
    display_df["Pf"] = display_df["Pf"].apply(lambda x: f"{x:.4e}")
    display_df["β"] = display_df["β"].apply(lambda x: f"{x:.4f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Download button
    csv = sweep_df.to_csv(index=False)
    st.download_button(
        label="⬇️  Export Full Results CSV",
        data=csv,
        file_name=f"mcs_{species_key}_{scenario_key}_REI{rei_duration}.csv",
        mime="text/csv",
    )

    # Methodology reference box
    with st.expander("📖 Methodology Reference (Table 3.2 — Limit State Functions)"):
        st.markdown("""
| FM | Member | Limit State Function |
|---|---|---|---|
| FM1 | Top Chord | G₁ = k_c,y · f_c,0,d,fi · A_ef − N_Ed,fi |
| FM2 | Top Chord | G₂ = 1 − [(σ_c,0,d / k_c,y·f_c,0,d,fi)² + σ_m,y,d/f_m,d,fi] |
| FM4 | Bottom Chord | G₄ = f_m,d,fi · W_ef,y − M_Ed,fi |
| FM4a | Bottom Chord | G₄ₐ = 1 − (σ_t,0,d/f_t,0,d,fi + σ_m,y,d/f_m,d,fi) |
| FM5 | Bottom Chord | G₅ = k_crit · f_m,d,fi · W_ef,y − M_Ed,fi |
| FM6 | Web (Comp.) | G₆ = k_c · f_c,0,d,fi · A_ef − N_Ed,fi,web |
| FM7 | Web (Tension) | G₇ = f_t,0,d,fi · A_ef − N_Ed,fi,web |
| FM8 | All Members | G₈ = f_v,d,fi · A_ef,shear − V_Ed,fi |
        """)

# ─────────────────────────────────────────
# TAB 2 — FAILURE MODE DISTRIBUTION
# ─────────────────────────────────────────
with tab2:
    st.subheader("Failure Mode Distribution (FM1–FM8)")
    st.caption(f"Nominal cross-section: b={b_nom} mm, h={h_nom} mm")

    fm_map = {
        "FM1 Chord Buckling":     summary["FM1_Buckling%"],
        "FM2 Comb.Bend+Comp":     summary["FM2_CombBendComp%"],
        "FM3 Chord Tension":      summary["FM3_Tension%"],
        "FM4 Bending":            summary["FM4_Bending%"],
        "FM4a Comb.Ten+Bend":     summary["FM4a_CombTenBend%"],
        "FM5 LTB":                summary["FM5_LTB%"],
        "FM6 Web Buckling":       summary["FM6_WBuckling%"],
        "FM7 Web Tension":        summary["FM7_WTension%"],
        "FM8 Shear":              summary["FM8_Shear%"]
    }
    fm_df = pd.DataFrame({"Failure Mode": list(fm_map.keys()),
                          "Percentage (%)": list(fm_map.values())})
    fm_df = fm_df[fm_df["Percentage (%)"] > 0].sort_values("Percentage (%)", ascending=True)

    if fm_df.empty:
        st.info("No failures recorded in this run — all members survived the fire duration.")
    else:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            fig_bar = go.Figure(go.Bar(
                y=fm_df["Failure Mode"],
                x=fm_df["Percentage (%)"],
                orientation='h',
                marker=dict(
                    color=FM_COLORS[:len(fm_df)],
                    line=dict(color="rgba(0,0,0,0.2)", width=1)
                ),
                text=[f"{v:.1f}%" for v in fm_df["Percentage (%)"]],
                textposition="outside",
                textfont=dict(color="#e6edf3"),
            ))
            fig_bar.update_layout(
                **plotly_layout("Failure Mode Breakdown", "Relative Frequency (%)", ""),
                height=400,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            fig_pie = go.Figure(go.Pie(
                labels=fm_df["Failure Mode"],
                values=fm_df["Percentage (%)"],
                marker=dict(colors=FM_COLORS[:len(fm_df)],
                            line=dict(color="#0d1117", width=2)),
                hole=0.45,
                textfont=dict(color="#e6edf3", size=11),
            ))
            fig_pie.update_layout(
                **plotly_layout(""),
                height=400,
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Stacked bar across all b/h combinations
    st.subheader("Failure Mode Distribution Across b/h Combinations")
    fm_sweep_cols = [c for c in sweep_df.columns if c.endswith('%')]
    if sweep_df[fm_sweep_cols].sum().sum() > 0:
        sweep_melt = sweep_df[["b", "h"] + fm_sweep_cols].copy()
        sweep_melt["label"] = sweep_melt.apply(lambda r: f"b={int(r.b)}, h={int(r.h)}", axis=1)
        fig_stack = go.Figure()
        for i, col in enumerate(fm_sweep_cols):
            fig_stack.add_trace(go.Bar(
                name=col.replace('%', '').replace('_', ' '),
                x=sweep_melt["label"],
                y=sweep_melt[col],
                marker_color=FM_COLORS[i % len(FM_COLORS)],
            ))
        fig_stack.update_layout(
            **plotly_layout("Failure Mode % by b/h Configuration", "Cross-section", "% of Failures"),
            barmode="stack",
            height=380,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_stack, use_container_width=True)

# ─────────────────────────────────────────
# TAB 3 — PARAMETRIC HEATMAP
# ─────────────────────────────────────────
with tab3:
    st.subheader("Parametric b × h Heatmap")
    st.caption("Reliability Index β over all width/depth combinations")

    metric_choice = st.radio(
        "Display metric:", ["β (Reliability Index)", "Pf (Failure Probability)"],
        horizontal=True
    )
    show_pf = "Pf" in metric_choice

    pivot_col = "Pf_System" if show_pf else "Beta_System"
    pivot = sweep_df.pivot(index="h", columns="b", values=pivot_col)
    z = pivot.values
    x_labels = [f"b={int(v)}" for v in pivot.columns]
    y_labels = [f"h={int(v)}" for v in pivot.index]

    color_scale = "Reds" if show_pf else "Blues_r"
    z_text = [[f"{v:.3e}" if show_pf else f"{v:.3f}" for v in row] for row in z]

    fig_heat = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=color_scale,
        zmin=np.nanmin(z),
        zmax=np.nanmax(z),
        colorbar=dict(
            title=dict(text="Pf" if show_pf else "β", font=dict(color="#e6edf3")),
            tickfont=dict(color="#e6edf3"),
        ),
        text=z_text,
        texttemplate="%{text}",
        textfont=dict(color="white", size=11),
        hoverongaps=False,
    ))
    fig_heat.update_layout(
        **plotly_layout(
            f"{'Probability of Failure (Pf)' if show_pf else 'Reliability Index (β)'} — b × h Grid",
            "Width b (mm)", "Depth h (mm)"
        ),
        height=520,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # 3D surface alternative
    with st.expander("🧊 Show 3D Surface"):
        fig_3d = go.Figure(go.Surface(
            z=z,
            x=[int(v) for v in pivot.columns],
            y=[int(v) for v in pivot.index],
            colorscale=color_scale,
            colorbar=dict(title=dict(text="β" if not show_pf else "Pf",
                                     font=dict(color="#e6edf3")),
                          tickfont=dict(color="#e6edf3")),
        ))
        fig_3d.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=PLOT_BG,
            scene=dict(
                xaxis=dict(title="b (mm)", gridcolor=GRID_COLOR),
                yaxis=dict(title="h (mm)", gridcolor=GRID_COLOR),
                zaxis=dict(title="β" if not show_pf else "Pf", gridcolor=GRID_COLOR),
            ),
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# ─────────────────────────────────────────
# TAB 4 — CONVERGENCE
# ─────────────────────────────────────────
with tab4:
    st.subheader("Simulation Convergence")
    st.caption(f"Running Pf estimate vs. N iterations (b={b_nom} mm, h={h_nom} mm)")

    if convergence:
        conv_df = pd.DataFrame(convergence, columns=["N", "Pf_running"])
        # Compute running beta
        conv_df["Beta_running"] = conv_df["Pf_running"].apply(
            lambda p: -float(stats.norm.ppf(p)) if 0 < p < 1 else (5.0 if p == 0 else -5.0)
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig_conv_pf = go.Figure()
            fig_conv_pf.add_trace(go.Scatter(
                x=conv_df["N"], y=conv_df["Pf_running"],
                mode="lines", name="Pf (running)",
                line=dict(color=ACCENT_BLUE, width=2),
            ))
            fig_conv_pf.add_hline(y=summary["Pf_System"], line_dash="dash",
                                   line_color=ACCENT_PURPLE, annotation_text="Final Pf")
            fig_conv_pf.update_layout(
                **plotly_layout("Pf Convergence", "Iterations N", "Probability of Failure"),
                height=360,
            )
            st.plotly_chart(fig_conv_pf, use_container_width=True)

        with col_b:
            fig_conv_beta = go.Figure()
            fig_conv_beta.add_trace(go.Scatter(
                x=conv_df["N"], y=conv_df["Beta_running"],
                mode="lines", name="β (running)",
                line=dict(color=ACCENT_PURPLE, width=2),
            ))
            fig_conv_beta.add_hline(y=summary["Beta_System"], line_dash="dash",
                                     line_color=ACCENT_BLUE, annotation_text=f"β = {summary['Beta_System']:.3f}")
            fig_conv_beta.update_layout(
                **plotly_layout("β Convergence", "Iterations N", "Reliability Index β"),
                height=360,
            )
            st.plotly_chart(fig_conv_beta, use_container_width=True)

        # Convergence check (Eqn 3.90)
        if len(conv_df) > 1:
            pf_n    = conv_df["Pf_running"].iloc[-1]
            pf_half = conv_df["Pf_running"].iloc[len(conv_df) // 2]
            if pf_half > 0:
                rel_change = abs(pf_n - pf_half) / pf_half
                converged = rel_change <= 0.05
                badge = "✅ CONVERGED" if converged else "⚠️ NOT YET CONVERGED"
                st.markdown(
                    f"**Convergence Check (Eqn 3.90):** |Pf(N) - Pf(N/2)| / Pf(N/2) = "
                    f"`{rel_change:.4f}` → **{badge}**"
                )

# ─────────────────────────────────────────
# TAB 5 — SENSITIVITY
# ─────────────────────────────────────────
with tab5:
    st.subheader("Sensitivity Analysis — Spearman Rank Correlation")
    st.caption("Correlation of input variables with structural failure (per Eqn 3.87)")

    if samples_df.empty:
        st.info("No sample data available — run with N ≥ 500 to see sensitivity results.")
    else:
        corr = compute_sensitivity(samples_df)
        if corr.empty or corr.isna().all():
            st.info("Insufficient failure variability to compute Spearman correlations.")
        else:
            corr_df = corr.reset_index()
            corr_df.columns = ["Variable", "Spearman ρ"]
            corr_df = corr_df.dropna().sort_values("Spearman ρ", ascending=True)

            colors = [
                "#f85149" if v > 0 else "#3fb950"
                for v in corr_df["Spearman ρ"]
            ]
            fig_tornado = go.Figure(go.Bar(
                y=corr_df["Variable"],
                x=corr_df["Spearman ρ"],
                orientation='h',
                marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.2)", width=1)),
                text=[f"{v:.3f}" for v in corr_df["Spearman ρ"]],
                textposition="outside",
                textfont=dict(color="#e6edf3"),
            ))
            fig_tornado.add_vline(x=0, line_color=GRID_COLOR, line_width=1)
            fig_tornado.update_layout(
                **plotly_layout(
                    "Tornado Chart — Input Variable Sensitivity",
                    "Spearman Rank Correlation with Failure", ""
                ),
                height=400,
            )
            st.plotly_chart(fig_tornado, use_container_width=True)

            st.markdown("""
            **Interpretation:**
            - 🔴 **Positive ρ** → variable increases the likelihood of failure
            - 🟢 **Negative ρ** → variable decreases the likelihood of failure
            - Variables closest to ±1 are the most critical design parameters
            """)

            with st.expander("📊 Full Correlation Table"):
                corr_display = corr_df.copy()
                corr_display["Spearman ρ"] = corr_display["Spearman ρ"].apply(
                    lambda x: f"{x:+.4f}"
                )
                st.dataframe(corr_display, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8b949e; font-size:0.78rem;'>"
    "Monte Carlo Timber Truss Fire Reliability Engine · "
    "EN 1995-1-2 · ISO 834 / Parametric Fire · "
    "Anogeissus leiocarpa &amp; Erythrophleum suaveolens"
    "</p>",
    unsafe_allow_html=True
)
