"""
B2B Sales Automation & Analytics Dashboard
Renaissance Dark/Brown Theme
Course: Applied Programming Tools for B2B Business
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="B2B Sales Intelligence",
    page_icon="⚜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────
# RENAISSANCE THEME CSS
# ────────────────────────────────────────────────
RENAISSANCE_CSS = """
<style>
/* ── Import Renaissance-style fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

/* ── Root palette ── */
:root {
    --parchment:   #f5ead4;
    --vellum:      #ede0c4;
    --ink:         #1a1008;
    --sepia-dark:  #2c1a0e;
    --sepia-mid:   #5c3d1e;
    --sepia-light: #8b5e3c;
    --gold:        #c9a84c;
    --gold-bright: #e8c97a;
    --gold-dim:    #7a5c1e;
    --crimson:     #8b1a1a;
    --forest:      #1a4a2e;
    --slate:       #2a3040;
    --bg-main:     #0f0a05;
    --bg-card:     #1a1008;
    --bg-panel:    #140d06;
    --border-gold: #c9a84c55;
}

/* ── Global background ── */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    background-image:
        radial-gradient(ellipse at 20% 20%, #2c1a0e22 0%, transparent 60%),
        radial-gradient(ellipse at 80% 80%, #1a0a0222 0%, transparent 60%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0d05 0%, #0f0a03 100%) !important;
    border-right: 1px solid var(--gold-dim) !important;
}
[data-testid="stSidebar"] * { color: var(--vellum) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: var(--gold) !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--gold) !important;
    font-family: 'Cinzel', serif !important;
}

/* ── Main text ── */
.stApp p, .stApp li, .stApp span, .stApp div,
[data-testid="stMarkdownContainer"] {
    color: var(--vellum) !important;
    font-family: 'Crimson Text', Georgia, serif !important;
}

/* ── Title and headers ── */
h1, h2, h3, h4, .stApp h1, .stApp h2, .stApp h3 {
    font-family: 'Cinzel', serif !important;
    color: var(--gold) !important;
    letter-spacing: 0.06em !important;
    text-shadow: 0 0 30px #c9a84c44;
}

/* ── Tab styling ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--gold-dim) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Cinzel', serif !important;
    font-size: 0.75rem !important;
    color: var(--sepia-light) !important;
    letter-spacing: 0.1em !important;
    padding: 0.6rem 1.2rem !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
    border-radius: 4px 4px 0 0 !important;
    background: transparent !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--gold) !important;
    background: linear-gradient(180deg, #2c1a0e 0%, #1a0d05 100%) !important;
    border-color: var(--gold-dim) !important;
    text-shadow: 0 0 15px #c9a84c66 !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--gold-bright) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e1205 0%, #150d04 100%) !important;
    border: 1px solid var(--gold-dim) !important;
    border-radius: 6px !important;
    padding: 1rem !important;
    box-shadow: inset 0 1px 0 #c9a84c22, 0 4px 20px #00000066 !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-family: 'Cinzel', serif !important;
    font-size: 0.65rem !important;
    color: var(--gold-dim) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Cinzel', serif !important;
    font-size: 1.6rem !important;
    color: var(--gold) !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: var(--forest) !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--gold-dim) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

/* ── Divider ── */
hr { border-color: var(--gold-dim) !important; }

/* ── Buttons ── */
.stButton button {
    font-family: 'Cinzel', serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    color: var(--ink) !important;
    background: linear-gradient(135deg, var(--gold) 0%, var(--gold-dim) 100%) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 4px !important;
    padding: 0.5rem 1.5rem !important;
    box-shadow: 0 2px 8px #c9a84c44 !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, var(--gold-bright) 0%, var(--gold) 100%) !important;
    box-shadow: 0 4px 16px #c9a84c88 !important;
}

/* ── Inputs ── */
.stTextInput input, .stSelectbox select {
    background: #1a0d05 !important;
    border: 1px solid var(--gold-dim) !important;
    color: var(--vellum) !important;
    border-radius: 4px !important;
    font-family: 'Crimson Text', serif !important;
}

/* ── Info/Success boxes ── */
.stSuccess, .stInfo, .stWarning {
    border-radius: 6px !important;
    border-left: 3px solid var(--gold) !important;
    background: #1a1008 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--gold) !important;
}

/* ── Caption ── */
.stApp .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--sepia-light) !important;
    font-family: 'Crimson Text', serif !important;
    font-style: italic !important;
    font-size: 0.85rem !important;
}

/* ── Scroll bars ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 3px; }
</style>
"""

st.markdown(RENAISSANCE_CSS, unsafe_allow_html=True)

# ────────────────────────────────────────────────
# MATPLOTLIB RENAISSANCE THEME
# ────────────────────────────────────────────────
PALETTE = {
    "bg":          "#0f0a05",
    "bg_card":     "#1a1008",
    "bg_axes":     "#130c04",
    "gold":        "#c9a84c",
    "gold_bright": "#e8c97a",
    "gold_dim":    "#7a5c1e",
    "crimson":     "#8b1a1a",
    "forest":      "#1a4a2e",
    "vellum":      "#ede0c4",
    "sepia":       "#8b5e3c",
    "bars":        ["#c9a84c", "#8b5e3c", "#5c3d1e", "#e8c97a", "#7a5c1e",
                    "#8b1a1a", "#1a4a2e", "#2a3040"],
}

def ren_fig(w=6, h=4):
    """Return a pre-styled Figure with Renaissance colours."""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PALETTE["bg_card"])
    ax.set_facecolor(PALETTE["bg_axes"])
    ax.tick_params(colors=PALETTE["vellum"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["gold"])
    ax.yaxis.label.set_color(PALETTE["gold"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["gold_dim"])
    ax.title.set_color(PALETTE["gold"])
    return fig, ax

# ────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("B2B_Leads_Dataset_1000Records.xlsx")
    except FileNotFoundError:
        st.error("⚠ Dataset not found. Place 'B2B_Leads_Dataset_1000Records.xlsx' in the same folder as app.py.")
        st.stop()

    required = ["Lead_ID", "Client_Name", "Industry", "Region",
                "Lead_Source", "Revenue", "Status", "Follow_Up_Time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()
    return df

df = load_data()

# Derived columns
df["Converted"] = (df["Status"] == "Converted").astype(int)

# ────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:1.6rem 0 0.4rem 0;">
  <div style="font-family:'Cinzel',serif; font-size:0.7rem; color:#7a5c1e; letter-spacing:0.3em; text-transform:uppercase; margin-bottom:0.3rem;">
    ✦ &nbsp; Applied Programming Tools for B2B Business &nbsp; ✦
  </div>
  <h1 style="font-family:'Cinzel',serif; font-size:2rem; font-weight:900; color:#c9a84c;
             margin:0; text-shadow:0 0 40px #c9a84c55; letter-spacing:0.08em;">
    B2B Sales Intelligence Codex
  </h1>
  <div style="font-family:'Crimson Text',serif; font-size:1.05rem; color:#8b5e3c;
              font-style:italic; margin-top:0.3rem;">
    Lead Analytics · Conversion Insights · Strategic Intelligence
  </div>
  <div style="height:1px; background:linear-gradient(90deg,transparent,#c9a84c,transparent);
              margin:1rem auto; width:60%;"></div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# SIDEBAR FILTERS
# ────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding:0.8rem 0;">
  <div style="font-family:'Cinzel',serif; font-size:1rem; color:#c9a84c; letter-spacing:0.1em;">
    ⚜ FILTERS ⚜
  </div>
  <div style="height:1px; background:linear-gradient(90deg,transparent,#c9a84c,transparent);
              margin:0.5rem 0;"></div>
</div>
""", unsafe_allow_html=True)

regions    = sorted(df["Region"].dropna().unique())
industries = sorted(df["Industry"].dropna().unique())
sources    = sorted(df["Lead_Source"].dropna().unique())

sel_region   = st.sidebar.multiselect("Region",      regions,    default=regions)
sel_industry = st.sidebar.multiselect("Industry",    industries, default=industries)
sel_source   = st.sidebar.multiselect("Lead Source", sources,    default=sources)

rev_min, rev_max = float(df["Revenue"].min()), float(df["Revenue"].max())
sel_rev = st.sidebar.slider(
    "Revenue Range (USD)",
    min_value=rev_min, max_value=rev_max,
    value=(rev_min, rev_max),
    format="$%.0f"
)

st.sidebar.markdown("""
<div style="margin-top:1.5rem; padding:0.8rem; border:1px solid #7a5c1e33;
            border-radius:4px; background:#1a0d05;">
  <div style="font-family:'Cinzel',serif; font-size:0.6rem; color:#7a5c1e;
              letter-spacing:0.15em; text-align:center; margin-bottom:0.4rem;">
    DATASET
  </div>
  <div style="font-family:'Crimson Text',serif; font-size:0.8rem; color:#8b5e3c;
              text-align:center; font-style:italic;">
    1 000 B2B Leads<br>8 Features · 2 Classes
  </div>
</div>
""", unsafe_allow_html=True)

# ── Apply filters ──
filtered = df[
    df["Region"].isin(sel_region) &
    df["Industry"].isin(sel_industry) &
    df["Lead_Source"].isin(sel_source) &
    (df["Revenue"] >= sel_rev[0]) &
    (df["Revenue"] <= sel_rev[1])
].copy()

# ────────────────────────────────────────────────
# TABS
# ────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Dashboard",
    "🤖  ML Predictor",
    "💡  Insights & Strategy",
    "📜  Data Compendium",
])

# ══════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════
with tab1:

    n       = len(filtered)
    n_conv  = int(filtered["Converted"].sum())
    conv_r  = (n_conv / n * 100) if n else 0
    avg_fup = filtered["Follow_Up_Time"].mean() if n else 0
    tot_rev = filtered["Revenue"].sum() if n else 0

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("⚜ Total Leads",       f"{n:,}")
    k2.metric("✅ Converted",         f"{n_conv:,}")
    k3.metric("📈 Conversion Rate",   f"{conv_r:.1f}%")
    k4.metric("⏱ Avg Follow-Up",     f"{avg_fup:.1f} hrs")
    k5.metric("💰 Total Revenue",     f"${tot_rev:,.0f}")

    st.divider()

    # ── Row 1: Bar + Donut ──
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### Leads by Region")
        fig, ax = ren_fig(6, 3.5)
        region_counts = filtered["Region"].value_counts()
        bars = ax.bar(region_counts.index, region_counts.values,
                      color=PALETTE["bars"][:len(region_counts)], edgecolor="#0f0a05", linewidth=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(int(bar.get_height())), ha='center', va='bottom',
                    color=PALETTE["gold"], fontsize=7, fontfamily='serif')
        ax.set_ylabel("Lead Count")
        ax.set_title("")
        ax.grid(axis='y', color=PALETTE["gold_dim"], alpha=0.15, linestyle='--')
        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        st.caption("Regional distribution of all filtered leads.")

    with col2:
        st.markdown("#### Conversion Status")
        fig2, ax2 = ren_fig(4, 3.5)
        status_counts = filtered["Status"].value_counts()
        wedge_colors = [PALETTE["gold"], PALETTE["crimson"]]
        wedge_props  = dict(width=0.45, edgecolor=PALETTE["bg_card"], linewidth=2)
        ax2.pie(status_counts.values,
                labels=status_counts.index,
                autopct="%1.1f%%",
                colors=wedge_colors,
                startangle=90,
                wedgeprops=wedge_props,
                textprops={"color": PALETTE["vellum"], "fontsize": 9})
        ax2.set_title("")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig2)
        st.caption("Converted vs Not Converted leads.")

    st.divider()

    # ── Row 2: Conversion by Industry + Revenue Trend ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Conversion Rate by Industry")
        fig3, ax3 = ren_fig(6, 3.8)
        ind_conv = filtered.groupby("Industry")["Converted"].mean().sort_values() * 100
        colors3 = [PALETTE["crimson"] if v < 40 else PALETTE["gold"] if v < 60 else PALETTE["forest"]
                   for v in ind_conv.values]
        bars3 = ax3.barh(ind_conv.index, ind_conv.values, color=colors3, edgecolor="#0f0a05", linewidth=0.4)
        for bar in bars3:
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{bar.get_width():.1f}%", va='center',
                     color=PALETTE["vellum"], fontsize=7)
        ax3.set_xlabel("Conversion Rate (%)")
        ax3.axvline(50, color=PALETTE["gold_dim"], linestyle='--', linewidth=0.8, alpha=0.5)
        ax3.grid(axis='x', color=PALETTE["gold_dim"], alpha=0.1, linestyle='--')
        ax3.set_title("")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig3)
        legend_items = [
            mpatches.Patch(color=PALETTE["crimson"], label="< 40%"),
            mpatches.Patch(color=PALETTE["gold"],    label="40–60%"),
            mpatches.Patch(color=PALETTE["forest"],  label="> 60%"),
        ]
        st.caption("Gold dashed line = 50% benchmark.")

    with col4:
        st.markdown("#### Revenue Distribution by Lead Source")
        fig4, ax4 = ren_fig(6, 3.8)
        src_rev = [
            filtered.loc[filtered["Lead_Source"] == src, "Revenue"].dropna()
            for src in sources if src in filtered["Lead_Source"].values
        ]
        src_labels = [s for s in sources if s in filtered["Lead_Source"].values]
        if src_rev:
            bp = ax4.boxplot(src_rev, labels=src_labels, patch_artist=True,
                             medianprops=dict(color=PALETTE["gold_bright"], linewidth=1.5),
                             flierprops=dict(marker='o', markerfacecolor=PALETTE["gold_dim"],
                                             markersize=3, linestyle='none'),
                             whiskerprops=dict(color=PALETTE["sepia"]),
                             capprops=dict(color=PALETTE["sepia"]))
            for patch, color in zip(bp['boxes'], PALETTE["bars"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax4.set_ylabel("Revenue (USD)")
        ax4.tick_params(axis='x', labelrotation=30, labelsize=7)
        ax4.grid(axis='y', color=PALETTE["gold_dim"], alpha=0.15, linestyle='--')
        ax4.set_title("")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig4)
        st.caption("Revenue spread per lead acquisition channel.")

    st.divider()

    # ── Row 3: Lead Source bar + Follow-Up scatter ──
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#### Lead Source Analysis")
        fig5, ax5 = ren_fig(6, 3.5)
        src_conv = filtered.groupby("Lead_Source").agg(
            Total=("Lead_ID", "count"),
            Converted=("Converted", "sum")
        ).reset_index()
        x = np.arange(len(src_conv))
        w = 0.38
        ax5.bar(x - w/2, src_conv["Total"],     width=w, label="Total",
                color=PALETTE["sepia"], edgecolor="#0f0a05", linewidth=0.4)
        ax5.bar(x + w/2, src_conv["Converted"], width=w, label="Converted",
                color=PALETTE["gold"],  edgecolor="#0f0a05", linewidth=0.4)
        ax5.set_xticks(x)
        ax5.set_xticklabels(src_conv["Lead_Source"], rotation=30, ha='right', fontsize=7)
        ax5.legend(facecolor=PALETTE["bg_card"], edgecolor=PALETTE["gold_dim"],
                   labelcolor=PALETTE["vellum"], fontsize=7)
        ax5.grid(axis='y', color=PALETTE["gold_dim"], alpha=0.12, linestyle='--')
        ax5.set_title("")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig5)
        st.caption("Total leads vs converted leads per source.")

    with col6:
        st.markdown("#### Follow-Up Time vs Conversion")
        fig6, ax6 = ren_fig(6, 3.5)
        conv_mask = filtered["Converted"] == 1
        ax6.scatter(filtered.loc[~conv_mask, "Follow_Up_Time"],
                    filtered.loc[~conv_mask, "Revenue"],
                    c=PALETTE["crimson"], alpha=0.4, s=12, label="Not Converted")
        ax6.scatter(filtered.loc[conv_mask,  "Follow_Up_Time"],
                    filtered.loc[conv_mask,  "Revenue"],
                    c=PALETTE["gold"],   alpha=0.6, s=14, label="Converted")
        ax6.set_xlabel("Follow-Up Time (hrs)")
        ax6.set_ylabel("Revenue (USD)")
        ax6.legend(facecolor=PALETTE["bg_card"], edgecolor=PALETTE["gold_dim"],
                   labelcolor=PALETTE["vellum"], fontsize=7)
        ax6.grid(color=PALETTE["gold_dim"], alpha=0.1, linestyle='--')
        ax6.set_title("")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig6)
        st.caption("Faster follow-up typically correlates with conversion.")

    st.divider()

    # ── Top 20 Leads ──
    st.markdown("#### ⚜ Top 20 High-Revenue Leads")
    top20 = filtered.sort_values("Revenue", ascending=False).head(20)
    st.dataframe(
        top20[["Lead_ID", "Client_Name", "Industry", "Region",
               "Lead_Source", "Revenue", "Status", "Follow_Up_Time"]],
        use_container_width=True, hide_index=True
    )

    # Download
    st.download_button(
        "⬇  Download Filtered Data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_b2b_leads.csv",
        mime="text/csv"
    )


# ══════════════════════════════════════════════════
# TAB 2 — ML PREDICTOR
# ══════════════════════════════════════════════════
with tab2:
    st.markdown("### 🤖 Decision Tree Classifier — Lead Conversion Prediction")
    st.caption("Predicts whether a lead will convert using behavioural and firmographic signals.")

    feature_cols = ["Industry", "Region", "Lead_Source", "Revenue", "Follow_Up_Time"]
    X = df[feature_cols].copy()
    y = df["Converted"]
    X = pd.get_dummies(X, drop_first=True)

    depth = st.slider("Decision Tree Max Depth", min_value=2, max_value=12, value=5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc  = accuracy_score(y_test, pred)
    cm   = confusion_matrix(y_test, pred)

    m1, m2, m3, m4 = st.columns(4)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    m1.metric("Accuracy",  f"{acc*100:.2f}%")
    m2.metric("Precision", f"{precision*100:.2f}%")
    m3.metric("Recall",    f"{recall*100:.2f}%")
    m4.metric("Test Size", f"{len(X_test)} leads")

    st.divider()

    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion Matrix")
        fig_cm, ax_cm = ren_fig(4, 3.5)
        im = ax_cm.imshow(cm, cmap="YlOrBr", aspect="auto")
        ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred: No", "Pred: Yes"], color=PALETTE["vellum"], fontsize=8)
        ax_cm.set_yticklabels(["Actual: No", "Actual: Yes"], color=PALETTE["vellum"], fontsize=8)
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center',
                           color=PALETTE["ink"], fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax_cm)
        ax_cm.set_title("Prediction Accuracy Matrix", color=PALETTE["gold"], fontsize=9)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig_cm)

    with col_fi:
        st.markdown("#### Feature Importance — Top Conversion Drivers")
        importances = pd.Series(model.feature_importances_, index=X.columns)\
                        .sort_values(ascending=True).tail(10)
        fig_fi, ax_fi = ren_fig(5, 3.5)
        colors_fi = [PALETTE["gold"] if v == importances.max() else PALETTE["sepia"]
                     for v in importances.values]
        ax_fi.barh(importances.index, importances.values,
                   color=colors_fi, edgecolor="#0f0a05", linewidth=0.4)
        ax_fi.set_xlabel("Importance Score")
        ax_fi.grid(axis='x', color=PALETTE["gold_dim"], alpha=0.12, linestyle='--')
        ax_fi.set_title("")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig_fi)

    st.markdown("""
<div style="background:#1a1008; border:1px solid #7a5c1e; border-left:3px solid #c9a84c;
            border-radius:6px; padding:1rem 1.2rem; margin-top:0.5rem;">
  <div style="font-family:'Cinzel',serif; font-size:0.7rem; color:#c9a84c;
              letter-spacing:0.1em; margin-bottom:0.4rem;">⚜ MODEL INTERPRETATION</div>
  <div style="font-family:'Crimson Text',serif; font-size:0.95rem; color:#ede0c4;">
    The Decision Tree splits data on the most informative features first.
    Features with higher importance scores create larger information gains at
    each branch — they are the primary determinants of whether a lead converts.
    Revenue and Follow-Up Time consistently rank as the strongest predictors.
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# TAB 3 — INSIGHTS & STRATEGY
# ══════════════════════════════════════════════════
with tab3:
    st.markdown("### 💡 Business Insights & Strategic Recommendations")

    n_f     = len(filtered)
    if n_f == 0:
        st.warning("No data matches current filters.")
    else:
        # Compute key analytics
        best_region   = filtered.groupby("Region")["Converted"].mean().idxmax()
        best_reg_rate = filtered.groupby("Region")["Converted"].mean().max() * 100
        best_source   = filtered.groupby("Lead_Source")["Converted"].mean().idxmax()
        best_src_rate = filtered.groupby("Lead_Source")["Converted"].mean().max() * 100

        conv_fup  = filtered[filtered["Converted"]==1]["Follow_Up_Time"].mean()
        nconv_fup = filtered[filtered["Converted"]==0]["Follow_Up_Time"].mean()
        fup_diff  = nconv_fup - conv_fup

        best_ind   = filtered.groupby("Industry")["Converted"].mean().idxmax()
        best_i_rt  = filtered.groupby("Industry")["Converted"].mean().max() * 100
        worst_ind  = filtered.groupby("Industry")["Converted"].mean().idxmin()
        worst_i_rt = filtered.groupby("Industry")["Converted"].mean().min() * 100

        # Insight cards
        ins_data = [
            {
                "icon": "🗺",
                "title": "Best Performing Region",
                "body": f"<strong style='color:#c9a84c;'>{best_region}</strong> leads with a "
                        f"<strong>{best_reg_rate:.1f}%</strong> conversion rate. "
                        "Direct sales resources toward this region and replicate its playbook elsewhere.",
            },
            {
                "icon": "🎯",
                "title": "Top Lead Source",
                "body": f"<strong style='color:#c9a84c;'>{best_source}</strong> delivers "
                        f"<strong>{best_src_rate:.1f}%</strong> conversion — the highest of all channels. "
                        "Increase budget allocation to this source for maximum ROI.",
            },
            {
                "icon": "⏱",
                "title": "Follow-Up Speed Impact",
                "body": f"Converted leads received follow-up in <strong>{conv_fup:.1f} hrs</strong> on average, "
                        f"versus <strong>{nconv_fup:.1f} hrs</strong> for non-converted leads "
                        f"(a difference of <strong>{fup_diff:.1f} hrs</strong>). "
                        "Faster follow-up is strongly correlated with higher conversion.",
            },
            {
                "icon": "🏭",
                "title": "Industry Intelligence",
                "body": f"<strong style='color:#c9a84c;'>{best_ind}</strong> converts at "
                        f"<strong>{best_i_rt:.1f}%</strong> — prioritise it. "
                        f"<strong>{worst_ind}</strong> underperforms at <strong>{worst_i_rt:.1f}%</strong>; "
                        "revisit messaging and qualification criteria for this segment.",
            },
        ]

        for card in ins_data:
            st.markdown(f"""
<div style="background:linear-gradient(135deg,#1e1205,#150d04);
            border:1px solid #7a5c1e; border-left:3px solid #c9a84c;
            border-radius:6px; padding:1rem 1.2rem; margin-bottom:0.75rem;">
  <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">
    <span style="font-size:1.1rem;">{card['icon']}</span>
    <span style="font-family:'Cinzel',serif; font-size:0.75rem; color:#c9a84c;
                 letter-spacing:0.08em; text-transform:uppercase;">{card['title']}</span>
  </div>
  <div style="font-family:'Crimson Text',serif; font-size:0.98rem; color:#ede0c4; line-height:1.5;">
    {card['body']}
  </div>
</div>
""", unsafe_allow_html=True)

        st.divider()
        st.markdown("#### ⚜ Three Strategic Recommendations")

        strategies = [
            {
                "num": "I",
                "title": "Accelerate Lead Response Protocol",
                "body": "Implement a <strong>same-day follow-up rule</strong> (target < 4 hours). "
                        "Deploy Make.com automation to trigger immediate acknowledgement emails and "
                        "assign a sales agent within 30 minutes of form submission. "
                        "Data shows converted leads are contacted significantly faster — "
                        "standardising this across all regions will lift overall conversion rate.",
            },
            {
                "num": "II",
                "title": "Double Down on High-ROI Channels",
                "body": f"Reallocate 30–40% of acquisition budget to <strong>{best_source}</strong> "
                        "and invest in content/partnerships that strengthen that channel. "
                        "Run A/B tests on underperforming channels before cutting them entirely — "
                        "some may convert poorly due to targeting issues rather than channel quality.",
            },
            {
                "num": "III",
                "title": "Industry-Specific Nurture Sequences",
                "body": f"Build tailored email and content sequences for <strong>{best_ind}</strong> "
                        "(high converters) to increase deal size, and develop a dedicated "
                        f"re-engagement campaign for <strong>{worst_ind}</strong> leads. "
                        "Segment CRM by industry and assign specialist account executives — "
                        "personalisation consistently outperforms generic outreach by 2–4×.",
            },
        ]

        for s in strategies:
            st.markdown(f"""
<div style="background:#1a1008; border:1px solid #7a5c1e33;
            border-radius:6px; padding:1rem 1.2rem; margin-bottom:0.75rem;
            display:flex; gap:1rem;">
  <div style="font-family:'Cinzel',serif; font-size:1.4rem; color:#7a5c1e;
              min-width:2rem; text-align:center; padding-top:0.1rem;">{s['num']}</div>
  <div>
    <div style="font-family:'Cinzel',serif; font-size:0.78rem; color:#c9a84c;
                letter-spacing:0.06em; margin-bottom:0.3rem;">{s['title']}</div>
    <div style="font-family:'Crimson Text',serif; font-size:0.96rem;
                color:#ede0c4; line-height:1.55;">{s['body']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Make.com workflow note
        st.divider()
        st.markdown("#### 🔄 Make.com Automation Workflow (Part B Overview)")
        st.markdown("""
<div style="background:#1a1008; border:1px solid #7a5c1e33; border-radius:6px; padding:1rem 1.2rem;">
  <div style="font-family:'Crimson Text',serif; font-size:0.96rem; color:#ede0c4; line-height:1.7;">
    <strong style="color:#c9a84c;">Trigger:</strong> Google Form submission (new lead capture)<br>
    <strong style="color:#c9a84c;">Action 1:</strong> Send automated confirmation email to the lead (personalised via data mapper)<br>
    <strong style="color:#c9a84c;">Action 2:</strong> Append lead record to Google Sheets CRM<br>
    <strong style="color:#c9a84c;">Action 3:</strong> Notify sales team via email + Slack channel with lead summary<br><br>
    <strong style="color:#c9a84c;">Business Benefits:</strong>
    Response time reduced from hours to seconds · Zero manual data entry ·
    Sales team notified instantly · Full audit trail in Sheets ·
    Scales to unlimited lead volume without additional headcount.
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# TAB 4 — DATA COMPENDIUM
# ══════════════════════════════════════════════════
with tab4:
    st.markdown("### 📜 Data Compendium — Full Dataset Preview")
    st.caption("All 1 000 records · Apply sidebar filters to narrow the view.")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Rows (filtered)", len(filtered))
    s2.metric("Industries",      filtered["Industry"].nunique())
    s3.metric("Regions",         filtered["Region"].nunique())
    s4.metric("Lead Sources",    filtered["Lead_Source"].nunique())

    st.divider()

    col_s, col_d = st.columns([1, 2])
    with col_s:
        st.markdown("#### Numeric Summary")
        st.dataframe(
            filtered[["Revenue", "Follow_Up_Time"]].describe().round(2),
            use_container_width=True
        )
    with col_d:
        st.markdown("#### Categorical Breakdown")
        cat_df = pd.DataFrame({
            "Industry":    filtered["Industry"].value_counts(),
            "Region":      filtered["Region"].value_counts(),
            "Lead_Source": filtered["Lead_Source"].value_counts(),
        }).fillna(0).astype(int)
        st.dataframe(cat_df, use_container_width=True)

    st.divider()
    st.markdown("#### Full Record Table")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=400)

    st.download_button(
        "⬇  Export Full Filtered Dataset (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="b2b_leads_export.csv",
        mime="text/csv"
    )

# ────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:1.5rem 0 0.5rem 0; margin-top:1rem;">
  <div style="height:1px; background:linear-gradient(90deg,transparent,#c9a84c,transparent);
              margin-bottom:0.8rem; width:50%; margin-left:auto; margin-right:auto;"></div>
  <div style="font-family:'Cinzel',serif; font-size:0.55rem; color:#5c3d1e;
              letter-spacing:0.2em; text-transform:uppercase;">
    ✦ &nbsp; B2B Sales Intelligence Codex · Applied Programming Tools for B2B Business &nbsp; ✦
  </div>
</div>
""", unsafe_allow_html=True)
