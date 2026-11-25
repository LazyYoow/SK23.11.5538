# Sistem Klasifikasi Kategori Indeks Massa Tubuh Menggunakan Simulasi
# Image Processing dan Machine Learning Berbasis Data Antropometri
# Disusun oleh: Satrio Raja Nabil Rizky (23.11.5538)
# Dosen Pembimbing: Mulia Sulistiyono, S.Kom., M.Kom.
import streamlit as st
import streamlit.components.v1 as components
import io
import html as _html
import pandas as pd
import numpy as np
import pickle
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
import warnings
from datetime import datetime, timedelta, timezone
warnings.filterwarnings("ignore")

# Zona waktu GMT+7 (WIB), fallback jika tzdata tidak tersedia.
try:
    from zoneinfo import ZoneInfo
    WIB = ZoneInfo("Asia/Jakarta")
except Exception:
    WIB = timezone(timedelta(hours=7))

HARI  = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
BULAN = ["Januari","Februari","Maret","April","Mei","Juni","Juli",
         "Agustus","September","Oktober","November","Desember"]

def fmt_hari_tanggal(dt):
    return f"{HARI[dt.weekday()]}, {dt.day} {BULAN[dt.month-1]} {dt.year}"

def fmt_waktu_riwayat(dt):
    return f"{dt.day:02d}/{dt.month:02d}/{dt.year} {dt:%H:%M:%S}"

st.set_page_config(
    page_title="Klasifikasi Kategori IMT | Skripsi S1 Informatika - 23.11.5538",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []
if "aff_idx" not in st.session_state:
    st.session_state.aff_idx = 0
if "last_idx" not in st.session_state:
    st.session_state.last_idx = None
if "inp_height" not in st.session_state:
    st.session_state.inp_height = 170
if "inp_weight" not in st.session_state:
    st.session_state.inp_weight = 70
if "inp_target" not in st.session_state:
    st.session_state.inp_target = 65

# Tema tunggal: Obsidian Precision Dark UI (WCAG AA >= 5.17:1 di seluruh permukaan)
BG_APP       = "#090A0D"
BG_SIDE      = "#0E1015"
BG_CARD      = "#0C0D10"
BG_SURFACE   = "#14161B"
BG_RAISE     = "#1B1E26"
BORDER       = "#242832"
BORDER_2     = "#313644"
INPUT_BORDER = "#667085"
TXT_MAIN     = "#F4F6F9"
TXT_BODY     = "#E2E6EE"
TXT_SUB      = "#A1A8B8"
TXT_TERT     = "#8A92A3"
PRIMARY      = "#2563EB"
PRIMARY_HOV  = "#1D4ED8"
PRIMARY_ACT  = "#1E40AF"
PRIMARY_TXT  = "#60A5FA"
SECONDARY    = "#94A3B8"
SUCCESS      = "#34D399"
WARNING      = "#FBBF24"
ERROR        = "#FB7185"
ACCENT       = "#60A5FA"
ACCENT_HI    = "#93C5FD"
ACCENT_2     = "#94A3B8"
PIE_TXT      = "#090A0D"
SIL_REF      = "#F4F6F9"
SH_A1, SH_A2 = 0.48, 0.32
CONTOUR_RGB  = (96, 165, 250)
CM_CMAP      = "Blues"
CAT_COLORS   = ["#60A5FA", "#38BDF8", "#34D399", "#FBBF24", "#FB923C", "#FB7185"]
CHART_COLORS = CAT_COLORS

plt.rcParams.update({
    "figure.facecolor": BG_CARD,
    "axes.facecolor": BG_CARD,
    "savefig.facecolor": BG_CARD,
    "text.color": TXT_MAIN,
    "axes.labelcolor": TXT_SUB,
    "axes.edgecolor": BORDER,
    "xtick.color": TXT_SUB,
    "ytick.color": TXT_SUB,
    "font.family": "sans-serif",
})

def style_ax(ax, grid_axis=None):
    ax.set_facecolor(BG_CARD)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(BORDER)
    ax.tick_params(colors=TXT_SUB, labelsize=8)
    if grid_axis:
        ax.set_axisbelow(True)
        ax.grid(axis=grid_axis, color=BORDER_2, linestyle=":", linewidth=0.85, alpha=0.75)
    return ax

def _svg_arc_path(cx, cy, r, deg_start, deg_end):
    """Path SVG busur lingkaran dari deg_start (kiri) ke deg_end (kanan) searah jarum jam."""
    a0 = np.radians(deg_start)
    a1 = np.radians(deg_end)
    x0, y0 = cx + r * np.cos(a0), cy - r * np.sin(a0)
    x1, y1 = cx + r * np.cos(a1), cy - r * np.sin(a1)
    return f"M {x0:.2f} {y0:.2f} A {r:.2f} {r:.2f} 0 0 1 {x1:.2f} {y1:.2f}"

def _bmi_to_gauge_angle(bmi_val):
    """Petakan BMI ke sudut busur 180..0 derajat sesuai 6 segmen kategori WHO."""
    b = float(np.clip(bmi_val, 10.0, 48.0))
    zones = [
        (10.0, 16.0, 178.0, 151.0),
        (16.0, 18.5, 145.0, 116.5),
        (18.5, 25.0, 110.5, 69.5),
        (25.0, 30.0, 63.5,  45.5),
        (30.0, 40.0, 39.5,  22.5),
        (40.0, 48.0, 16.5,  2.0),
    ]
    for b0, b1, d0, d1 in zones:
        if b <= b1:
            t = (b - b0) / max(b1 - b0, 1e-6)
            return d0 + t * (d1 - d0)
    return 2.0

def build_bmi_gauge_svg(bmi_val, active_idx, label_text):
    """Bangun SVG Semi-Circular Segmented BMI Gauge berlapis ganda dengan penunjuk segitiga putih."""
    cx, cy = 190.0, 150.0
    r_out, r_in = 114.0, 92.0
    segs = [
        (0, 180.0, 149.5, CAT_COLORS[0]),
        (1, 146.0, 115.5, CAT_COLORS[1]),
        (2, 112.0,  68.0, CAT_COLORS[2]),
        (3,  64.5,  44.5, CAT_COLORS[3]),
        (4,  41.0,  21.5, CAT_COLORS[4]),
        (5,  18.0,   0.0, CAT_COLORS[5]),
    ]
    paths = []
    for idx_s, d0, d1, col in segs:
        is_act = (idx_s == active_idx)
        w_out = 18 if is_act else 15
        op_out = 1.0 if is_act else 0.78
        op_in = 0.34 if is_act else 0.18
        p_in = _svg_arc_path(cx, cy, r_in, d0, d1)
        p_out = _svg_arc_path(cx, cy, r_out, d0, d1)
        paths.append(f'<path d="{p_in}" fill="none" stroke="{col}" stroke-width="11" stroke-opacity="{op_in}" stroke-linecap="butt"/>')
        paths.append(f'<path d="{p_out}" fill="none" stroke="{col}" stroke-width="{w_out}" stroke-opacity="{op_out}" stroke-linecap="butt"/>')

    # Label ambang WHO di tepi luar busur
    ticks = [(147.8, "16"), (113.8, "18,5"), (66.2, "25"), (42.8, "30"), (19.8, "40")]
    for d_t, lbl_t in ticks:
        rt = 136.0
        xt = cx + rt * np.cos(np.radians(d_t))
        yt = cy - rt * np.sin(np.radians(d_t))
        paths.append(
            f'<text x="{xt:.1f}" y="{yt:.1f}" fill="{TXT_SUB}" font-size="11.5" font-weight="600" '
            f'text-anchor="middle" dominant-baseline="middle">{lbl_t}</text>'
        )

    # Segitiga putih penunjuk posisi BMI saat ini di tepi luar busur
    theta = _bmi_to_gauge_angle(bmi_val)
    rad = np.radians(theta)
    ux, uy = np.cos(rad), -np.sin(rad)
    tx_p, ty_p = -uy, ux
    tip_r, base_r, half_w = 115.0, 128.0, 6.5
    p1x, p1y = cx + tip_r * ux, cy + tip_r * uy
    bx, by = cx + base_r * ux, cy + base_r * uy
    p2x, p2y = bx + half_w * tx_p, by + half_w * ty_p
    p3x, p3y = bx - half_w * tx_p, by - half_w * ty_p
    paths.append(f'<polygon points="{p1x:.2f},{p1y:.2f} {p2x:.2f},{p2y:.2f} {p3x:.2f},{p3y:.2f}" fill="#FFFFFF"/>')

    bmi_str = f"{bmi_val:.1f}".replace(".", ",")
    badge_txt = f"IMT · {label_text}"
    pill_w = max(104, len(badge_txt) * 7 + 22)
    paths.append(
        f'<text x="{cx:.0f}" y="116" fill="{TXT_MAIN}" font-size="38" font-weight="600" '
        f'text-anchor="middle" letter-spacing="-0.02em">{bmi_str}</text>'
    )
    paths.append(
        f'<rect x="{cx - pill_w/2:.1f}" y="129" width="{pill_w}" height="24" rx="12" fill="#171A21" stroke="#282C37" stroke-width="1"/>'
    )
    paths.append(
        f'<text x="{cx:.0f}" y="145" fill="{TXT_SUB}" font-size="11" font-weight="600" text-anchor="middle">{_html.escape(badge_txt)}</text>'
    )
    return (
        '<div class="gauge-well">'
        '<svg class="gauge-svg" viewBox="0 0 380 166" width="380" height="166" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gauge Indeks Massa Tubuh">'
        + "".join(paths) +
        '</svg></div>'
    )

def render_table(df):
    """Tabel HTML responsif bergaya Obsidian Well tanpa kanvas putih."""
    headers = "".join(f"<th>{_html.escape(str(c))}</th>" for c in df.columns)
    body_rows = []
    for row in df.itertuples(index=False):
        tds = "".join(f"<td>{_html.escape(str(v))}</td>" for v in row)
        body_rows.append(f"<tr>{tds}</tr>")
    st.markdown(
        f'<div class="tbl-wrap"><table class="data-tbl">'
        f'<thead><tr>{headers}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

# ============================================================
# REDESIGN UI v2 (hanya tampilan; tidak menyentuh logika ML)
# ============================================================
HIDE_STREAMLIT_TOOLBAR = True   # True: sembunyikan Stop/menu titik tiga saat demo atau sidang

REDESIGN_CSS = """
/* ================================================================
   REDESIGN LAYER v2 (Antislop Matte Pass):
   Tanpa gradien dekoratif, tanpa pendar/glow, tanpa garis aksen dekoratif,
   tanpa dot status kosong, dan target sentuh >= 44px.
   ================================================================ */
.stApp { -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }
.block-container { padding-top: 2.2rem !important; max-width: 1280px !important; }

/* ---------- HEADER ---------- */
.page-header {
    border-bottom: none !important;
    padding: 0 0 18px !important;
    margin-bottom: 4px !important;
    align-items: center !important;
}
.page-title {
    font-size: clamp(24px, 2.5vw, 32px) !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    margin: 0 0 6px !important;
    padding: 0 !important;
}
.page-sub { font-size: 14px !important; color: #A1A8B8 !important; max-width: 640px; }
.page-id {
    position: relative;
    padding: 14px 18px !important;
    border-radius: 12px !important;
    background: #12141B !important;
    border: 1px solid #232836 !important;
    box-shadow: none !important;
    min-width: 280px;
}
.page-id-kicker {
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    color: #8A92A3 !important;
    margin-bottom: 3px !important;
}
.page-id strong { font-size: 14.5px !important; letter-spacing: -0.01em; margin-bottom: 2px !important; }
.page-id-line { display: block; font-size: 12px; color: #A1A8B8; line-height: 1.5; }

/* ---------- TABS ---------- */
div[data-baseweb="tab-list"] {
    gap: 4px !important;
    border-bottom: 1px solid #1E222C !important;
    margin-bottom: 20px !important;
}
button[data-baseweb="tab"] {
    padding: 10px 16px !important;
    min-height: 44px !important;
    border-radius: 8px 8px 0 0 !important;
    font-size: 14px !important;
    color: #A1A8B8 !important;
}
button[data-baseweb="tab"]:hover { background-color: #11141B !important; color: #F4F6F9 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #FFFFFF !important; font-weight: 600 !important; }
div[data-baseweb="tab-highlight"] { background-color: #60A5FA !important; height: 2px !important; border-radius: 0; }

/* ---------- KARTU UTAMA (container border) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #12141A !important;
    border: 1px solid #1F232D !important;
    border-radius: 16px !important;
    box-shadow: none !important;
    padding: 20px !important;
}
div[data-testid="stExpander"], div[data-testid="stExpander"] details {
    border-radius: 12px !important;
    background-color: #12141A !important;
    border-color: #1F232D !important;
}
div[data-testid="stExpander"] summary { background-color: #12141A !important; border-radius: 12px !important; }
div[data-testid="stVerticalBlockBorderWrapper"] h4 {
    font-size: 15px !important; font-weight: 600 !important; letter-spacing: -0.01em; margin-bottom: 4px !important;
}
div[data-testid="stImage"] img { border-radius: 10px !important; }

/* ---------- SIDEBAR (Flat Matte Solid) ---------- */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
    background: #0D0F14 !important;
    border-right: 1px solid #1A1D26 !important;
}
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
    padding-top: 1.5rem !important;
    padding-left: 18px !important;
    padding-right: 18px !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 14px !important; }

.sb-head { padding: 0 2px 2px !important; }
.sb-title { font-size: 16px !important; font-weight: 600 !important; letter-spacing: -0.015em; margin: 0; }
.sb-sub { font-size: 12.5px !important; color: #8A92A3 !important; margin: 3px 0 0 !important; }

/* Satu kartu kontrol matte (satu level border) */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #12141B !important;
    border: 1px solid #1F232E !important;
    border-radius: 14px !important;
    padding: 18px 18px 14px !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* Segmented switch matte tanpa gradien/glow, tap target >= 44px */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
    background-color: #0A0B10 !important;
    border: 1px solid #1E222D !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 4px !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {
    min-height: 44px !important;
    border-radius: 7px !important;
    border: 1px solid transparent !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"]:hover,
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background-color: #151924 !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"]:has(input:checked),
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: #1E2433 !important;
    border-color: #5B657A !important;
    box-shadow: none !important;
}
.stApp section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"] p,
.stApp section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label p {
    color: #9CA3AF !important; font-size: 13.5px !important; font-weight: 500 !important;
}
.stApp section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"] p,
.stApp section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"]:has(input:checked) p,
.stApp section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #F4F6F9 !important; font-weight: 600 !important;
}
.stApp section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stWidgetLabel"] p {
    font-size: 13px !important; font-weight: 500 !important; letter-spacing: 0 !important;
    text-transform: none !important; color: #A1A8B8 !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stWidgetLabel"] { margin-bottom: 8px !important; }

/* Baris slider seragam: sentence case + nilai chip datar */
.ctrl-row {
    border-top: none !important;
    margin-top: 18px !important;
    padding-top: 0 !important;
    margin-bottom: 4px;
    align-items: center !important;
}
.ctrl-lbl { font-size: 13px !important; font-weight: 500 !important; letter-spacing: 0 !important; text-transform: none !important; color: #A1A8B8 !important; }
.ctrl-readout {
    font-size: 14.5px !important; font-weight: 600 !important; color: #F4F6F9 !important;
    background: #0A0B10; border: 1px solid #232834; border-radius: 6px; padding: 2px 9px;
}
.ctrl-readout small { font-size: 11.5px !important; color: #8A92A3 !important; margin-left: 3px !important; }
section[data-testid="stSidebar"] div[data-testid="stSlider"] { padding: 4px 4px 0 !important; }
section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] {
    width: 16px !important; height: 16px !important;
    background-color: #F4F6F9 !important;
    border: 2px solid #2563EB !important;
    box-shadow: none !important;
}

/* Kartu IMT sidebar: matte solid dengan mistar 6 segmen diskrit WHO */
.rail-card {
    background: #13161E !important;
    border: 1px solid #222733 !important;
    border-radius: 14px !important;
    padding: 18px !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.rail-lbl { font-size: 12.5px !important; font-weight: 500 !important; letter-spacing: 0 !important; text-transform: none !important; color: #A1A8B8 !important; }
.rail-hero { align-items: center !important; margin-top: 6px; }
.rail-val { font-size: 34px !important; font-weight: 700 !important; letter-spacing: -0.03em !important; }
.rail-unit { font-size: 13px !important; margin-left: 5px !important; }
.rail-hero .status-tag { padding: 4px 10px; border-radius: 6px; font-size: 12px; }
.rail-meter {
    position: relative; height: 6px; border-radius: 3px; margin: 16px 0 4px;
    background: linear-gradient(90deg,
        #60A5FA 0% 14.4%, #13161E 14.4% 15.2%,
        #38BDF8 15.2% 20.8%, #13161E 20.8% 21.6%,
        #34D399 21.6% 37.1%, #13161E 37.1% 37.9%,
        #FBBF24 37.9% 49.6%, #13161E 49.6% 50.4%,
        #FB923C 50.4% 74.6%, #13161E 74.6% 75.4%,
        #FB7185 75.4% 100%);
    opacity: 0.9;
}
.rail-meter i {
    position: absolute; top: -5px; width: 3px; height: 16px; border-radius: 2px;
    background: #FFFFFF; transform: translateX(-50%); box-shadow: 0 0 0 2px #13161E;
}
.rail-ledger { border-top: 1px solid #1F2531 !important; margin-top: 14px !important; padding-top: 14px !important; gap: 10px !important; }
.rail-row { font-size: 13px !important; }

/* Tombol aksi utama solid tanpa gradien dan tanpa glow */
section[data-testid="stSidebar"] .stButton > button {
    min-height: 46px !important;
    border-radius: 10px !important;
    background: #2563EB !important;
    border: 1px solid #3B82F6 !important;
    color: #FFFFFF !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #1D4ED8 !important;
    border-color: #60A5FA !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stButton > button:active {
    background: #1E40AF !important;
    transform: translateY(1px);
    box-shadow: none !important;
}

/* ---------- KARTU METRIK (Matte Netral tanpa Garis Warna Dekoratif) ---------- */
.diag-hero {
    background-color: #12141A !important;
    border: 1px solid #1F232D !important;
    border-radius: 16px !important;
    box-shadow: none !important;
    padding: 24px !important;
}
.hero-top { align-items: flex-start !important; margin-bottom: 18px !important; }
.hero-title { font-size: 22px !important; font-weight: 650 !important; letter-spacing: -0.02em !important; }
.hero-sub { font-size: 13px !important; color: #A1A8B8 !important; margin-top: 4px !important; }
.tint-pill {
    padding: 5px 12px !important; border-radius: 6px !important;
    font-size: 12.5px !important; line-height: 1.3; white-space: nowrap;
}
.tile-grid { gap: 12px !important; margin-bottom: 18px !important; }
.metric-tile {
    position: relative;
    display: flex; flex-direction: column; gap: 5px;
    min-height: 112px;
    padding: 15px 16px !important;
    border-radius: 12px !important;
    background-color: #14171F !important;
    border: 1px solid #242A36 !important;
    box-shadow: none !important;
}
.tile-lbl {
    font-size: 12.5px !important; font-weight: 500 !important;
    letter-spacing: 0 !important; text-transform: none !important;
    color: #A1A8B8 !important; margin: 0 !important;
}
.tile-blue .tile-lbl    { color: #A1A8B8 !important; }
.tile-emerald .tile-lbl { color: #A1A8B8 !important; }
.tile-amber .tile-lbl   { color: #A1A8B8 !important; }
.tile-rose .tile-lbl    { color: #A1A8B8 !important; }
.tile-val {
    font-size: 22px !important; font-weight: 700 !important;
    letter-spacing: -0.02em; line-height: 1.18 !important;
    margin: 2px 0 0 !important; overflow-wrap: anywhere;
}
.tile-emerald .tile-val { color: #34D399; }
.tile-rose .tile-val    { color: #FB7185; }
.tile-meta { font-size: 12.5px !important; color: #A1A8B8 !important; margin: auto 0 0 !important; padding-top: 4px; line-height: 1.4; }
.conf-bar { height: 4px; border-radius: 2px; background: #1E2433; margin-top: 8px; overflow: hidden; }
.conf-bar i { display: block; height: 100%; border-radius: 2px; background: #60A5FA; }

.gauge-well, .capsule-bar-wrap {
    background-color: #0D0F14 !important;
    border: 1px solid #1B1F28 !important;
    border-radius: 12px !important;
}
.capsule-bar-wrap { padding: 16px 18px 12px !important; }
.capsule-title { font-size: 15px !important; }
.capsule-status { font-size: 15px !important; font-weight: 600 !important; }
.capsule-track { height: 12px !important; background-color: #171A22 !important; }
.ruler-ticks { font-size: 12px !important; color: #A1A8B8 !important; font-weight: 500; }

/* Kartu simulasi target & afirmasi */
.sim-box { border-radius: 10px !important; }

@media (max-width: 1120px) {
    .page-id { min-width: 0; }
    .tile-val { font-size: 20px !important; }
}
@media (max-width: 640px) {
    .page-title { font-size: 22px !important; }
    .tile-val { font-size: 19px !important; }
    .metric-tile { min-height: 0; }
}
"""

HIDE_TOOLBAR_CSS = """
[data-testid="stStatusWidget"], [data-testid="stMainMenu"], [data-testid="stToolbarActions"] { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
""" if HIDE_STREAMLIT_TOOLBAR else ""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --color-primary: {PRIMARY};
    --color-primary-hover: {PRIMARY_HOV};
    --color-primary-active: {PRIMARY_ACT};
    --color-primary-text: {PRIMARY_TXT};
    --color-on-primary: #FFFFFF;
    --color-secondary: {SECONDARY};

    --color-neutral-50: #F4F6F9;
    --color-neutral-100: #E2E6EE;
    --color-neutral-200: #C3CAD9;
    --color-neutral-300: #A1A8B8;
    --color-neutral-400: #8A92A3;
    --color-neutral-500: #5E6675;
    --color-neutral-600: #313644;
    --color-neutral-700: #242832;
    --color-neutral-800: #14161B;
    --color-neutral-900: #090A0D;

    --color-success: {SUCCESS};
    --color-warning: {WARNING};
    --color-error: {ERROR};

    --color-cat-0: {CAT_COLORS[0]};
    --color-cat-1: {CAT_COLORS[1]};
    --color-cat-2: {CAT_COLORS[2]};
    --color-cat-3: {CAT_COLORS[3]};
    --color-cat-4: {CAT_COLORS[4]};
    --color-cat-5: {CAT_COLORS[5]};

    --color-bg: {BG_APP};
    --color-surface: {BG_SURFACE};
    --color-well: {BG_CARD};
    --color-surface-subtle: {BG_RAISE};
    --color-border: {BORDER};
    --color-border-strong: {BORDER_2};
    --color-input-border: {INPUT_BORDER};
    --color-text-primary: {TXT_MAIN};
    --color-text-secondary: {TXT_SUB};
    --color-text-tertiary: {TXT_TERT};

    --background-color: var(--color-bg);
    --secondary-background-color: var(--color-surface);
    --text-color: var(--color-text-primary);
    --primary-color: var(--color-primary);

    --text-xs: 12px;
    --text-sm: 14px;
    --text-base: 16px;
    --text-lg: 20px;
    --text-xl: 24px;
    --text-2xl: 32px;

    --space-1: 4px;
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-6: 24px;
    --space-8: 32px;
    --space-12: 48px;
    --space-16: 64px;

    --radius-md: 8px;
    --radius-lg: 16px;
}}

/* Kanvas utama: Obsidian matte tenang tanpa grid dekoratif */
html, body, [class*="css"], .stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    font-size: var(--text-sm);
    font-weight: 400;
    line-height: 1.5;
    font-variant-numeric: tabular-nums;
    background-color: var(--color-bg) !important;
    background-image: none !important;
    color: var(--color-text-primary) !important;
}}
div[data-testid="stAppViewContainer"], section[data-testid="stMain"] {{
    background-color: transparent !important;
    color: var(--color-text-primary) !important;
}}
header[data-testid="stHeader"] {{
    background-color: transparent !important;
}}
header[data-testid="stHeader"] button, header[data-testid="stHeader"] svg {{
    color: var(--color-text-secondary) !important;
    fill: var(--color-text-secondary) !important;
}}
[data-testid="stAppDeployButton"] {{ display: none !important; }}
.block-container {{
    padding-top: var(--space-4);
    padding-bottom: var(--space-12);
    max-width: 1240px;
}}

a[href], button, summary, select, label, [role="button"], [role="tab"], [role="radio"], [role="slider"], input[type="checkbox"], input[type="radio"] {{
    cursor: pointer;
}}
button:disabled, [aria-disabled="true"] {{
    cursor: not-allowed;
}}
button:focus-visible, a:focus-visible, summary:focus-visible, [role="tab"]:focus-visible, [role="slider"]:focus-visible, input:focus-visible {{
    outline: 2px solid var(--color-primary-text) !important;
    outline-offset: 2px !important;
    box-shadow: none !important;
}}

/* Sidebar Architectural Instrument Panel */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{
    background-color: {BG_SIDE} !important;
    border-right: 1px solid var(--color-border);
    color: var(--color-text-primary) !important;
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: var(--space-4) !important;
    padding-bottom: var(--space-6) !important;
    padding-left: var(--space-4) !important;
    padding-right: var(--space-4) !important;
}}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{
    gap: 10px !important;
}}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: #13161D !important;
    border: 1px solid #252A36 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: none !important;
}}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {{
    gap: 4px !important;
}}
section[data-testid="stSidebar"] svg {{
    fill: var(--color-text-secondary) !important;
    color: var(--color-text-secondary) !important;
}}
/* Tactile Matte Segmented Switch untuk Jenis Kelamin (44px tap target, 3.3:1 boundary) */
section[data-testid="stSidebar"] div.stElementContainer:has(div[data-testid="stRadio"]),
section[data-testid="stSidebar"] div[data-testid="stRadio"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
    width: 100% !important;
    max-width: 100% !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stWidgetLabel"] {{
    margin-bottom: 6px !important;
    min-height: 0 !important;
}}
.stApp section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stWidgetLabel"] p {{
    font-size: 13px !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: #A1A8B8 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {{
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    width: 100% !important;
    box-sizing: border-box !important;
    gap: 4px !important;
    background-color: #0B0D12 !important;
    border: 1px solid #242936 !important;
    border-radius: 9px !important;
    padding: 3px !important;
    min-height: 0 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > div {{
    width: 100% !important;
    min-width: 0 !important;
    margin: 0 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {{
    margin: 0 !important;
    padding: 8px 10px !important;
    min-height: 44px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    border-radius: 6px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    gap: 0 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div:first-child,
section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-of-type {{
    display: none !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"]:hover,
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
    background-color: #151923 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"]:has(input:checked),
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
    background-color: #1E2330 !important;
    border-color: #5B657A !important;
    box-shadow: none !important;
}}
.stApp section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"] p,
.stApp section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label p {{
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #9CA3AF !important;
    text-align: center !important;
    margin: 0 !important;
}}
.stApp section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"] p,
.stApp section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"]:has(input:checked) p,
.stApp section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {{
    color: #F4F6F9 !important;
    font-weight: 600 !important;
}}
/* Symmetrical Sliders & Pure Typographic Readouts */
section[data-testid="stSidebar"] div[data-testid="stSlider"] {{
    padding-top: 2px !important;
    padding-bottom: 4px !important;
    margin-bottom: 0 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stThumbValue"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stTickBarMax"] {{
    display: none !important;
}}
.sb-head {{
    padding: 0 2px 6px;
    margin-bottom: 0;
}}
.sb-title {{
    font-size: 15px;
    font-weight: 600;
    color: var(--color-text-primary);
    line-height: 1.3;
    margin: 0;
    letter-spacing: -0.015em;
}}
.sb-sub {{
    font-size: 12px;
    font-weight: 400;
    color: var(--color-text-tertiary);
    margin: 2px 0 0;
    line-height: 1.4;
}}
.ctrl-row {{
    border-top: 1px solid #1E222D;
    margin-top: 8px;
    padding-top: 12px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}}
.ctrl-lbl {{
    font-size: 13px;
    font-weight: 500;
    color: #A1A8B8;
}}
.ctrl-readout {{
    font-size: 14px;
    font-weight: 600;
    color: #F4F6F9;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
}}
.ctrl-readout small {{
    font-size: 12px;
    font-weight: 400;
    color: #8A92A3 !important;
    margin-left: 2px;
}}

/* Headings & Body */
h1, h2, h3, h4 {{
    color: var(--color-text-primary) !important;
    font-weight: 600 !important;
    line-height: 1.25 !important;
    letter-spacing: -0.01em;
}}
h4 {{
    font-size: var(--text-base) !important;
    margin-bottom: var(--space-2) !important;
}}
.stApp p, .stApp li, .stApp label, .stApp span[data-baseweb] {{
    color: var(--color-text-primary);
    font-weight: 400;
    line-height: 1.5;
}}
.stApp .stCaption, .stApp small, div[data-testid="stCaptionContainer"] p {{
    color: var(--color-text-secondary) !important;
    font-size: var(--text-xs) !important;
    line-height: 1.5 !important;
}}
div[data-testid="stWidgetLabel"] p, div[data-testid="stRadio"] label p, div[data-testid="stCheckbox"] label p {{
    color: var(--color-text-primary) !important;
    font-size: var(--text-sm) !important;
    line-height: 1.5 !important;
}}

/* Inputs & Native Controls */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploaderDropzone"] {{
    background-color: var(--color-well) !important;
    border: 1px solid var(--color-input-border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--color-text-primary) !important;
    box-shadow: none !important;
}}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {{
    border-color: var(--color-primary-text) !important;
    outline: 2px solid var(--color-primary-text) !important;
    outline-offset: 2px !important;
    box-shadow: none !important;
}}
div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p {{
    color: var(--color-text-primary) !important;
    font-size: var(--text-sm) !important;
}}
div[data-baseweb="select"] svg,
div[data-testid="stFileUploader"] svg,
div[data-testid="stExpander"] summary svg {{
    fill: var(--color-text-secondary) !important;
    color: var(--color-text-secondary) !important;
}}
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[role="listbox"] {{
    background-color: var(--color-surface) !important;
    border: 1px solid var(--color-border-strong) !important;
    border-radius: var(--radius-md) !important;
    color: var(--color-text-primary) !important;
}}
li[role="option"] {{
    background-color: var(--color-surface) !important;
    color: var(--color-text-primary) !important;
    font-size: var(--text-sm) !important;
}}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {{
    background-color: var(--color-surface-subtle) !important;
    color: var(--color-text-primary) !important;
}}
span[data-baseweb="tag"] {{
    background-color: #111C30 !important;
    color: #60A5FA !important;
    border: 1px solid #1E3A5F !important;
    border-radius: var(--radius-md) !important;
    font-size: var(--text-xs) !important;
}}
span[data-baseweb="tag"] span {{ color: var(--color-text-primary) !important; }}
span[data-baseweb="tag"] svg {{ fill: #60A5FA !important; }}

div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"],
div[data-testid="stSlider"] [data-testid="stThumbValue"] {{
    color: var(--color-text-secondary) !important;
    font-size: var(--text-xs) !important;
}}
div[data-testid="stToast"] {{
    background-color: var(--color-surface) !important;
    color: var(--color-text-primary) !important;
    border: 1px solid var(--color-border-strong) !important;
    border-radius: var(--radius-md) !important;
}}
div[data-testid="stAlert"] {{
    background-color: var(--color-surface) !important;
    border: 1px solid var(--color-border-strong) !important;
    border-radius: var(--radius-lg) !important;
    color: var(--color-text-primary) !important;
}}
div[data-testid="stAlert"] p {{
    color: var(--color-text-primary) !important;
    font-size: var(--text-sm) !important;
}}
div[data-testid="stProgress"] > div > div {{
    background-color: var(--color-well) !important;
    border-radius: var(--radius-md) !important;
}}
div[data-testid="stProgress"] > div > div > div {{
    background-color: var(--color-primary) !important;
    border-radius: var(--radius-md) !important;
}}

/* Tone & Category Utilities */
.tone-primary {{ color: var(--color-primary-text) !important; }}
.tone-c0 {{ color: var(--color-cat-0) !important; }}
.tone-c1 {{ color: var(--color-cat-1) !important; }}
.tone-c2 {{ color: var(--color-cat-2) !important; }}
.tone-c3 {{ color: var(--color-cat-3) !important; }}
.tone-c4 {{ color: var(--color-cat-4) !important; }}
.tone-c5 {{ color: var(--color-cat-5) !important; }}

.tag-c0 {{ color: var(--color-cat-0); border-color: #1E3A5F; background-color: #111C30; }}
.tag-c1 {{ color: var(--color-cat-1); border-color: #164E63; background-color: #0F222D; }}
.tag-c2 {{ color: var(--color-cat-2); border-color: #144334; background-color: #10231D; }}
.tag-c3 {{ color: var(--color-cat-3); border-color: #4D3814; background-color: #241D11; }}
.tag-c4 {{ color: var(--color-cat-4); border-color: #542814; background-color: #271710; }}
.tag-c5 {{ color: var(--color-cat-5); border-color: #541B29; background-color: #261419; }}

.bar-c0 {{ background-color: var(--color-cat-0); }}
.bar-c1 {{ background-color: var(--color-cat-1); }}
.bar-c2 {{ background-color: var(--color-cat-2); }}
.bar-c3 {{ background-color: var(--color-cat-3); }}
.bar-c4 {{ background-color: var(--color-cat-4); }}
.bar-c5 {{ background-color: var(--color-cat-5); }}

.spacer-2 {{ height: var(--space-2); }}
.spacer-3 {{ height: var(--space-3); }}
.spacer-4 {{ height: var(--space-4); }}

/* Double-Bezel Obsidian Cards */
div[data-testid="stVerticalBlockBorderWrapper"] > div > div > div[data-testid="stVerticalBlock"] {{
    gap: var(--space-3);
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--color-border) !important;
    background-color: var(--color-surface) !important;
    box-shadow: inset 0 0 0 1px #1C1F28 !important;
    padding: var(--space-4) !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stImage"] img,
div[data-testid="stExpander"] div[data-testid="stImage"] img {{
    border-radius: var(--radius-md);
    border: 1px solid #1C1F27;
}}

div[data-testid="stExpander"], div[data-testid="stExpander"] details {{
    background-color: var(--color-surface) !important;
    border: 1px solid var(--color-border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: none !important;
}}
div[data-testid="stExpander"] details {{ border: none !important; }}
div[data-testid="stExpander"] summary {{
    min-height: var(--space-12);
    display: flex;
    align-items: center;
    background-color: var(--color-surface) !important;
    border-radius: var(--radius-lg) !important;
}}
div[data-testid="stExpander"] summary p {{
    color: var(--color-text-primary) !important;
    font-weight: 600 !important;
    font-size: var(--text-sm) !important;
}}

/* Buttons */
.stButton > button, .stDownloadButton > button {{
    background-color: var(--color-primary) !important;
    color: var(--color-on-primary) !important;
    border: 1px solid var(--color-primary) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: var(--text-sm) !important;
    line-height: 1.5 !important;
    padding: var(--space-2) var(--space-4) !important;
    min-height: var(--space-12);
    width: 100%;
    box-shadow: none !important;
    transition: background-color 0.15s ease, border-color 0.15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background-color: var(--color-primary-hover) !important;
    border-color: var(--color-primary-hover) !important;
    color: var(--color-on-primary) !important;
}}
.stButton > button:active, .stDownloadButton > button:active {{
    background-color: var(--color-primary-active) !important;
    border-color: var(--color-primary-active) !important;
}}
.st-key-clear_hist > div > button,
.st-key-aff_btn > div > button,
.st-key-demo_batch_btn > div > button,
.st-key-reset_demo_btn > div > button,
div[data-testid="stFileUploader"] button {{
    background-color: var(--color-well) !important;
    color: var(--color-text-primary) !important;
    border: 1px solid var(--color-input-border) !important;
    border-radius: var(--radius-md) !important;
}}
.st-key-clear_hist > div > button:hover,
.st-key-aff_btn > div > button:hover,
.st-key-demo_batch_btn > div > button:hover,
.st-key-reset_demo_btn > div > button:hover,
div[data-testid="stFileUploader"] button:hover {{
    background-color: var(--color-surface-subtle) !important;
    border-color: var(--color-primary-text) !important;
    color: var(--color-text-primary) !important;
}}

/* Page Header: 2-Column Precision Grid (Judul Kiri, Kartu Identitas Kanan) */
.page-header {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-6);
    padding: 0 0 var(--space-4);
    margin-bottom: var(--space-4);
    border-bottom: 1px solid var(--color-border);
}}
.page-title {{
    font-size: clamp(20px, 2.2vw, 26px);
    font-weight: 700;
    line-height: 1.2;
    margin: 0 0 4px;
    color: var(--color-text-primary);
    letter-spacing: -0.02em;
}}
.page-sub {{
    font-size: 13px;
    font-weight: 400;
    color: var(--color-text-secondary);
    margin: 0;
    line-height: 1.45;
}}
.page-id {{
    justify-self: end;
    text-align: left;
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    line-height: 1.55;
    background-color: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-4);
    white-space: nowrap;
}}
.page-id-kicker {{
    display: block;
    font-size: 11.5px;
    font-weight: 500;
    letter-spacing: 0;
    color: var(--color-text-tertiary);
    margin-bottom: 2px;
}}
.page-id strong {{
    color: var(--color-text-primary);
    font-weight: 600;
    display: block;
    font-size: var(--text-sm);
    margin-bottom: 2px;
}}

/* Tabs */
div[data-baseweb="tab-list"] {{
    gap: var(--space-6);
    border-bottom: 1px solid var(--color-border);
    flex-wrap: wrap;
    margin-bottom: var(--space-4);
}}
button[data-baseweb="tab"] {{
    color: var(--color-text-secondary) !important;
    font-weight: 500;
    font-size: var(--text-sm) !important;
    padding: var(--space-3) 0 !important;
    min-height: var(--space-12);
    background-color: transparent !important;
    transition: color 0.15s ease;
}}
button[data-baseweb="tab"]:hover {{ color: var(--color-text-primary) !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--color-text-primary) !important;
    font-weight: 600;
}}
div[data-baseweb="tab-highlight"] {{
    background-color: var(--color-primary-text) !important;
    height: 2px !important;
}}

/* Architectural Status Tags & Metric Tiles */
.pill-row {{
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin-top: var(--space-3);
}}
.tint-pill {{
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: var(--text-xs);
    font-weight: 500;
    line-height: 1.5;
    border: 1px solid;
}}
.tint-pill b {{
    color: var(--color-text-primary);
    font-weight: 600;
}}
.pill-emerald {{
    background-color: #10231D;
    border-color: #1F5C46;
    color: #34D399;
}}
.pill-blue {{
    background-color: #111C30;
    border-color: #274C7E;
    color: #60A5FA;
}}
.pill-amber {{
    background-color: #241D11;
    border-color: #664B1B;
    color: #FBBF24;
}}
.pill-rose {{
    background-color: #261419;
    border-color: #6E2436;
    color: #FB7185;
}}

/* Sidebar Architectural Readout Card (Focal Point + Clean Tabular Ledger) */
.rail-card {{
    background-color: #13161D;
    border: 1px solid #252A36;
    border-radius: 12px;
    padding: 16px;
    margin: 2px 0 6px;
    box-shadow: none;
}}
.rail-lbl {{
    display: block;
    font-size: 12px;
    font-weight: 500;
    color: #A1A8B8;
    margin: 0 0 6px;
}}
.rail-hero {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
}}
.rail-val {{
    font-size: 28px;
    font-weight: 600;
    color: #F4F6F9;
    line-height: 1.1;
    letter-spacing: -0.025em;
    font-variant-numeric: tabular-nums;
    margin: 0;
}}
.rail-unit {{
    font-size: 13px;
    font-weight: 400;
    color: #8A92A3;
    letter-spacing: 0;
    margin-left: 4px;
}}
.rail-ledger {{
    border-top: 1px solid #1F2430;
    margin-top: 14px;
    padding-top: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}}
.rail-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
    font-size: 12.5px;
    line-height: 1.35;
}}
.rail-k {{
    color: #A1A8B8;
    font-weight: 400;
}}
.rail-v {{
    color: #F4F6F9;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    text-align: right;
}}
section[data-testid="stSidebar"] .stButton > button {{
    min-height: 44px !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    background-color: #1A1E28 !important;
    color: #F4F6F9 !important;
    border: 1px solid #5B6478 !important;
    box-shadow: none !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.1s ease !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background-color: #232938 !important;
    border-color: #7D89A3 !important;
    color: #FFFFFF !important;
}}
section[data-testid="stSidebar"] .stButton > button:active {{
    background-color: #13161E !important;
    border-color: #5B6478 !important;
    transform: translateY(1px);
}}

/* Status Tag */
.status-tag {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: var(--text-xs);
    font-weight: 600;
    border: 1px solid var(--color-border);
    line-height: 1.25;
}}
.status-tag-sm {{
    margin-left: var(--space-2);
}}

/* Tab 1 Random Forest Main Spotlight Card */
.rf-spotlight {{
    background-color: #101726;
    border: 1px solid #1E3A5F;
    border-radius: var(--radius-lg);
    box-shadow: inset 0 0 0 1px #172A46;
    padding: var(--space-4) var(--space-6);
    margin-bottom: var(--space-4);
}}
.rf-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin-bottom: var(--space-3);
    padding-bottom: var(--space-3);
    border-bottom: 1px solid #1B2D48;
}}
.rf-kicker {{
    font-size: var(--text-xs);
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #60A5FA;
    text-transform: uppercase;
}}
.rf-grid {{
    display: grid;
    grid-template-columns: 1.15fr 1fr;
    gap: var(--space-4);
    align-items: center;
}}
.rf-main-lbl {{
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    margin: 0 0 var(--space-1);
}}
.rf-main-val {{
    font-size: var(--text-xl);
    font-weight: 700;
    color: var(--color-text-primary);
    margin: 0 0 var(--space-2);
    line-height: 1.25;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2);
}}
.rf-meta {{
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    margin: 0;
    line-height: 1.5;
}}
.rf-meta b {{
    color: var(--color-text-primary);
    font-weight: 600;
}}
.rf-probs {{
    background-color: var(--color-well);
    border: 1px solid #1C2536;
    border-radius: var(--radius-md);
    padding: var(--space-3);
}}
.rf-prob-title {{
    font-size: var(--text-xs);
    font-weight: 600;
    color: var(--color-text-secondary);
    margin: 0 0 var(--space-2);
}}
.rf-mini-row {{
    margin-bottom: var(--space-2);
}}
.rf-mini-row:last-child {{
    margin-bottom: 0;
}}
.rf-mini-head {{
    display: flex;
    justify-content: space-between;
    font-size: var(--text-xs);
    color: var(--color-text-primary);
    margin-bottom: 3px;
}}
.rf-mini-track {{
    height: 6px;
    background-color: #161B24;
    border-radius: 12px;
    overflow: hidden;
}}
.rf-mini-fill {{
    height: 100%;
    border-radius: 12px;
}}

/* Tab 5 Calorie & Energy Expenditure Grid */
.cal-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: var(--space-3);
    margin: var(--space-3) 0 var(--space-4);
}}
.cal-summary-box {{
    background-color: var(--color-well);
    border: 1px solid #1C1F27;
    border-radius: var(--radius-lg);
    padding: var(--space-4);
    margin-bottom: var(--space-4);
    font-size: var(--text-sm);
    color: var(--color-text-secondary);
    line-height: 1.6;
}}
.cal-summary-box strong {{
    color: var(--color-text-primary);
    font-weight: 600;
}}

/* Tab 1 Luxury Diagnostic Card (Inspirasi Kartu Kiri & Kanan) */
.diag-hero {{
    background-color: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    box-shadow: inset 0 0 0 1px #1C1F28;
    padding: var(--space-6);
    margin-bottom: var(--space-4);
}}
.hero-top {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: var(--space-3);
    margin-bottom: var(--space-4);
}}
.hero-title {{
    font-size: var(--text-xl);
    font-weight: 600;
    color: var(--color-text-primary);
    margin: 0;
    letter-spacing: -0.015em;
}}
.hero-sub {{
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    margin: var(--space-1) 0 0;
}}
.hero-reading {{
    font-size: var(--text-xl);
    font-weight: 500;
    color: var(--color-text-secondary);
    margin: 0;
}}
.hero-reading b {{
    color: var(--color-text-primary);
    font-weight: 600;
}}

/* 3 Tinted Metric Tiles Row */
.tile-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-3);
    margin-bottom: var(--space-4);
}}
.metric-tile {{
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-4);
    border: 1px solid;
}}
.tile-lbl {{
    font-size: var(--text-xs);
    font-weight: 600;
    margin: 0 0 var(--space-1);
}}
.tile-val {{
    font-size: var(--text-base);
    font-weight: 600;
    color: var(--color-text-primary);
    margin: 0;
    line-height: 1.25;
}}
.tile-meta {{
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    margin: var(--space-1) 0 0;
}}
.tile-emerald {{
    background-color: #10231D;
    border-color: #184233;
}}
.tile-emerald .tile-lbl {{ color: #34D399; }}
.tile-amber {{
    background-color: #241D11;
    border-color: #4D3814;
}}
.tile-amber .tile-lbl {{ color: #FBBF24; }}
.tile-blue {{
    background-color: #111C30;
    border-color: #1E3A5F;
}}
.tile-blue .tile-lbl {{ color: #60A5FA; }}
.tile-rose {{
    background-color: #261419;
    border-color: #541B29;
}}
.tile-rose .tile-lbl {{ color: #FB7185; }}

/* Recessed Well untuk Semi-Circular BMI Arc Gauge */
.gauge-well {{
    background-color: var(--color-well);
    border: 1px solid #1C1F27;
    border-radius: var(--radius-lg);
    padding: var(--space-4) var(--space-4) var(--space-2);
    margin-bottom: var(--space-4);
    display: flex;
    justify-content: center;
    align-items: center;
}}
.gauge-svg {{
    width: 100%;
    max-width: 390px;
    height: auto;
    display: block;
}}

/* Bottom Linear BMI Capsule Slider + White Upward Pointer (Inspirasi Kartu Kiri) */
.capsule-bar-wrap {{
    background-color: var(--color-well);
    border: 1px solid #1C1F27;
    border-radius: var(--radius-lg);
    padding: var(--space-4);
}}
.capsule-head {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: var(--space-3);
}}
.capsule-title {{
    font-size: var(--text-base);
    font-weight: 500;
    color: var(--color-text-primary);
}}
.capsule-title b {{
    font-weight: 600;
}}
.capsule-status {{
    font-size: var(--text-base);
    font-weight: 500;
    color: var(--color-text-secondary);
}}
.capsule-track {{
    position: relative;
    height: var(--space-3);
    border-radius: 24px;
    background-color: #16181F;
    border: 1px solid #222631;
    overflow: visible;
}}
.capsule-active-zone {{
    position: absolute;
    top: 0;
    bottom: 0;
    border-radius: 24px;
}}
.zone-0 {{ left: 0%; width: 15%; background-color: var(--color-cat-0); }}
.zone-1 {{ left: 15%; width: 6.25%; background-color: var(--color-cat-1); }}
.zone-2 {{ left: 21.25%; width: 16.25%; background-color: var(--color-cat-2); }}
.zone-3 {{ left: 37.5%; width: 12.5%; background-color: var(--color-cat-3); }}
.zone-4 {{ left: 50%; width: 25%; background-color: var(--color-cat-4); }}
.zone-5 {{ left: 75%; width: 25%; background-color: var(--color-cat-5); }}
.capsule-pointer {{
    position: absolute;
    bottom: -6px;
    transform: translateX(-50%);
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-bottom: 8px solid #FFFFFF;
    transition: left 0.2s ease;
}}
.ruler-ticks {{
    position: relative;
    height: var(--space-6);
    margin-top: var(--space-2);
    font-size: var(--text-xs);
    color: var(--color-text-tertiary);
}}
.ruler-ticks span {{
    position: absolute;
    transform: translateX(-50%);
    white-space: nowrap;
}}
.tick-16 {{ left: 15%; }}
.tick-185 {{ left: 21.25%; }}
.tick-25 {{ left: 37.5%; }}
.tick-30 {{ left: 50%; }}
.tick-40 {{ left: 75%; }}

/* Data Tables & WHO Table */
.tbl-wrap {{
    width: 100%;
    overflow-x: auto;
    background-color: var(--color-well);
    border: 1px solid #1C1F27;
    border-radius: var(--radius-md);
}}
.data-tbl, .who-tbl {{
    width: 100%;
    border-collapse: collapse;
    font-size: var(--text-sm);
    line-height: 1.5;
}}
.data-tbl thead tr, .who-tbl thead tr {{
    border-bottom: 1px solid var(--color-border);
    background-color: var(--color-surface);
}}
.data-tbl th, .who-tbl th {{
    text-align: left;
    padding: var(--space-2) var(--space-3);
    color: var(--color-text-secondary);
    font-weight: 600;
    font-size: var(--text-xs);
    white-space: nowrap;
}}
.data-tbl td, .who-tbl td {{
    padding: var(--space-2) var(--space-3);
    color: var(--color-text-primary);
    border-bottom: 1px solid #1A1D24;
    white-space: normal;
}}
.data-tbl tbody tr:last-child td, .who-tbl tbody tr:last-child td {{ border-bottom: none; }}
.data-tbl tbody tr:hover td {{ background-color: var(--color-surface); }}
.who-row-active {{ background-color: var(--color-surface-subtle); }}
.who-row-active .who-range {{ font-weight: 600; }}
.who-range {{ white-space: nowrap; color: var(--color-text-primary); }}
.who-label {{ font-weight: 600; }}
.who-en {{ font-size: var(--text-xs); color: var(--color-text-secondary); margin-left: var(--space-1); }}
.who-group {{ font-size: var(--text-xs); color: var(--color-text-secondary); }}

/* WHO Formula Strip */
.formula-strip {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2);
    padding: var(--space-3) var(--space-4);
    margin-bottom: var(--space-3);
    background-color: var(--color-well);
    border: 1px solid #1C1F27;
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
    color: var(--color-text-secondary);
}}
.formula-eq {{
    font-size: var(--text-base);
    font-weight: 600;
    color: var(--color-text-primary);
}}
.formula-val {{
    color: var(--color-text-primary);
    font-weight: 600;
}}

/* Target Weight Simulation */
.sim-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-3);
    margin: var(--space-2) 0 var(--space-3);
}}
.sim-box {{
    background-color: var(--color-well);
    border: 1px solid #1C1F27;
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-4);
}}
.sim-lbl {{
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    margin: 0 0 var(--space-1);
}}
.sim-val {{
    font-size: var(--text-xl);
    font-weight: 600;
    color: var(--color-text-primary);
    margin: 0;
    line-height: 1.25;
}}

/* Tab 2 Stat Strip */
.stat-strip {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: var(--space-3);
    margin-bottom: var(--space-4);
}}
.stat-cell {{
    background-color: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--space-4);
}}
.stat-lbl {{
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    margin: 0 0 var(--space-2);
}}
.stat-val {{
    font-size: var(--text-xl);
    font-weight: 600;
    color: var(--color-text-primary);
    margin: 0;
    line-height: 1.25;
}}
.stat-sub {{
    font-size: var(--text-xs);
    font-weight: 400;
    color: var(--color-text-secondary);
}}

/* Tab 3 Probability Rows */
.prob-row {{ margin-bottom: var(--space-4); }}
.prob-head {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: var(--text-sm);
    margin-bottom: var(--space-2);
    color: var(--color-text-primary);
}}
.prob-sub {{
    color: var(--color-text-secondary);
    font-size: var(--text-xs);
    margin-left: var(--space-2);
    font-weight: 400;
}}
.prob-val {{
    font-weight: 600;
    font-size: var(--text-base);
}}
.prob-track {{
    height: var(--space-2);
    background-color: var(--color-well);
    border: 1px solid #1C1F27;
    border-radius: 24px;
    overflow: hidden;
}}
.prob-fill {{
    height: 100%;
    border-radius: 24px;
    transition: width 0.2s ease;
}}

/* Tab 5 Recommendations */
.cat-banner {{
    background-color: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--space-6);
    margin-bottom: var(--space-4);
}}
.cat-banner-title {{
    font-size: var(--text-xl);
    font-weight: 600;
    line-height: 1.25;
    margin: 0 0 var(--space-2);
}}
.cat-banner-meta {{
    font-size: var(--text-sm);
    color: var(--color-text-secondary);
    margin: 0;
}}
.cat-meta-em {{
    color: var(--color-text-primary);
    font-weight: 600;
}}
.section-lead {{
    font-size: var(--text-sm);
    color: var(--color-text-secondary);
    margin: 0 0 var(--space-2);
}}

.rec-item {{
    display: flex;
    gap: var(--space-4);
    align-items: flex-start;
    background-color: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--space-4);
    margin-bottom: var(--space-1);
}}
.rec-num {{
    flex: 0 0 var(--space-8);
    height: var(--space-8);
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #111C30;
    border: 1px solid #1E3A5F;
    border-radius: var(--radius-md);
    font-weight: 600;
    font-size: var(--text-xs);
    color: var(--color-primary-text);
}}
.rec-title {{
    font-weight: 600;
    color: var(--color-text-primary);
    margin: 0 0 var(--space-1);
    font-size: var(--text-sm);
    line-height: 1.25;
}}
.rec-desc {{
    color: var(--color-text-secondary);
    margin: 0;
    font-size: var(--text-sm);
    line-height: 1.5;
}}

.aff-text {{
    background-color: var(--color-well);
    border: 1px solid #1C1F27;
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-4);
    font-size: var(--text-base);
    font-weight: 400;
    color: var(--color-text-primary);
    line-height: 1.5;
    margin: var(--space-3) 0 0;
}}
.aff-note {{
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    margin-top: var(--space-2);
}}

/* Live WIB Clock */
.clock {{
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-3);
    background-color: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--space-3) var(--space-4);
    font-size: var(--text-sm);
    color: var(--color-text-primary);
    margin-bottom: var(--space-3);
}}
.clock b {{
    font-size: var(--text-base);
    font-weight: 600;
    color: var(--color-text-primary);
}}
.clock-dot {{
    width: var(--space-2);
    height: var(--space-2);
    border-radius: var(--radius-md);
    background-color: var(--color-success);
}}
.clock-tz {{
    margin-left: auto;
    font-size: var(--text-xs);
    font-weight: 500;
    color: var(--color-text-secondary);
    background-color: var(--color-well);
    border: 1px solid var(--color-border-strong);
    border-radius: 6px;
    padding: 3px 10px;
}}
div[data-testid="stCustomComponentV1"]:has(iframe[height="0"]) {{
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
}}

.empty-box {{
    background-color: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--space-4);
    color: var(--color-text-secondary);
    font-size: var(--text-sm);
    line-height: 1.5;
}}
.empty-box strong {{
    color: var(--color-text-primary);
    display: block;
    margin-bottom: var(--space-1);
    font-weight: 600;
}}

.gold-rule {{
    height: 1px;
    border: 0;
    margin: var(--space-6) 0;
    background-color: var(--color-border);
}}

/* Multi-state responsive reflow: tablet/small-laptop (<= 1120px) dan mobile (<= 640px) */
@media (max-width: 1120px) {{
    .block-container {{
        padding-top: var(--space-4);
        padding-bottom: var(--space-8);
    }}
    .page-header {{
        grid-template-columns: 1fr;
        gap: var(--space-3);
    }}
    .page-id {{
        justify-self: start;
        white-space: normal;
    }}
    .diag-hero {{ padding: var(--space-4); }}
    .rf-spotlight {{ padding: var(--space-4); }}
    .rf-grid {{ grid-template-columns: 1fr; }}
    .tile-grid {{ grid-template-columns: 1fr 1fr; }}
    .cal-grid {{ grid-template-columns: 1fr 1fr; }}
    .stat-strip {{ grid-template-columns: 1fr 1fr; }}
    div[data-baseweb="tab-list"] {{ gap: var(--space-4); }}
    .clock-tz {{ margin-left: 0; }}
}}
@media (max-width: 640px) {{
    .block-container {{
        padding-top: var(--space-3);
        padding-bottom: var(--space-6);
        padding-left: var(--space-3);
        padding-right: var(--space-3);
    }}
    .diag-hero, .cat-banner, .stat-cell {{ padding: var(--space-3); }}
    .tile-grid, .cal-grid, .stat-strip, .sim-grid {{ grid-template-columns: 1fr; }}
    .hero-title, .cat-banner-title {{ font-size: 18px; }}
    div[data-baseweb="tab-list"] {{ gap: var(--space-3); }}
}}
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }}
}}
{REDESIGN_CSS}
{HIDE_TOOLBAR_CSS}
</style>
""", unsafe_allow_html=True)

LABEL_MAP   = {0:"Extremely Weak",1:"Weak",2:"Normal",3:"Overweight",4:"Obesity",5:"Extreme Obesity"}
LABEL_ID    = {0:"Sangat Kurus",  1:"Kurus",  2:"Normal", 3:"Overweight",4:"Obesitas",5:"Sangat Obesitas"}

RECOMMENDATIONS = {
    0:{"title":"Perlu Perhatian Ekstra","items":[
        ("01","Perbanyak Kalori Bergizi","Makanan padat nutrisi: kacang, alpukat, nasi merah, protein hewani berkualitas."),
        ("02","Latihan Pembentuk Otot","Latihan beban 3×/minggu. Target massa otot bukan sekadar angka timbangan."),
        ("03","Konsultasi Dokter Gizi","BMI sangat rendah bisa menandakan kondisi medis tersembunyi."),
        ("04","Tidur 7–9 Jam Cukup","Tidur berkualitas mendukung hormon pertumbuhan dan nafsu makan.")]},
    1:{"title":"Tingkatkan Nutrisi","items":[
        ("01","Protein & Karbohidrat Kompleks","Target 1.5–2g protein/kg BB. Makan 5–6 kali porsi kecil per hari."),
        ("02","Olahraga Ringan Rutin","30 menit jalan cepat atau bersepeda per hari untuk membangun metabolisme."),
        ("03","Kelola Stres","Stres menekan nafsu makan. Latihan pernapasan atau relaksasi 10 menit/hari membantu.")]},
    2:{"title":"Pertahankan Pola Hidup Sehat","items":[
        ("01","Jaga Konsistensi Pola Makan","BMI berada di rentang ideal. Pertahankan komposisi gizi seimbang dan jadwal makan teratur."),
        ("02","Olahraga Rutin & Gizi Seimbang","150 menit aktivitas sedang/minggu. Perbanyak sayur, buah, dan protein."),
        ("03","Cek Rutin 3–6 Bulan","Monitor BMI secara periodik agar perubahan kecil lebih mudah ditangani sejak dini."),
        ("04","Hidrasi Optimal","2–2.5 liter air/hari. Batasi minuman berpemanis tambahan.")]},
    3:{"title":"Kendalikan Berat Badan","items":[
        ("01","Tingkatkan Aktivitas Harian","Targetkan 8.000–10.000 langkah/hari atau 45 menit latihan kardio teratur."),
        ("02","Kurangi Gula & Gorengan","Utamakan bahan pangan utuh dan periksa label nutrisi sebelum membeli."),
        ("03","Defisit Kalori Terukur","Defisit 300–500 kkal/hari mendukung penurunan 0.3–0.5 kg/minggu secara bertahap."),
        ("04","Perbaiki Kualitas Tidur","Kurang tidur meningkatkan hormon ghrelin pemicu lapar. Targetkan 7–9 jam/malam.")]},
    4:{"title":"Prioritaskan Kesehatan","items":[
        ("01","Konsultasi Dokter atau Ahli Gizi","Evaluasi profil metabolik untuk mencegah risiko diabetes, hipertensi, dan penyakit kardiovaskular."),
        ("02","Olahraga Rendah Benturan","Renang, jalan kaki, atau sepeda statis lebih aman bagi persendian. Mulai 20 menit/hari."),
        ("03","Pola Makan Terstruktur","Hindari diet ekstrem. Terapkan 3 kali makan utama dengan porsi serat dan protein terukur."),
        ("04","Dukungan Lingkungan","Libatkan keluarga atau komunitas sehat agar perubahan kebiasaan bertahan jangka panjang.")]},
    5:{"title":"Pendampingan Medis Diperlukan","items":[
        ("01","Program Medis Terpadu","BMI di atas 40 memerlukan supervisi dokter dan ahli gizi untuk perencanaan yang aman."),
        ("02","Pemantauan Tekanan Darah & Gula Darah","Lakukan pemeriksaan berkala dan kurangi asupan natrium serta pangan ultra-proses."),
        ("03","Aktivitas Fisik Bertahap","Mulai dengan berjalan ringan 10 menit 3×/hari, lalu tingkatkan sesuai toleransi tubuh."),
        ("04","Pendampingan Perilaku Makan","Konseling membantu mengelola pola makan emosional secara berkelanjutan.")]}
}

AFFIRMATIONS = {
    0:["Berat badan bukan ukuran nilai dirimu. Mengurus tubuh dengan lebih baik adalah langkah yang patut dihargai.",
       "Kekurangan energi bisa diperbaiki pelan-pelan. Kamu tidak harus sempurna hari ini.",
       "Mencari bantuan profesional adalah bentuk keberanian, bukan kelemahan."],
    1:["Setiap porsi makan yang seimbang adalah investasi kecil untuk dirimu sendiri.",
       "Kamu sudah mengambil langkah pertama dengan mengetahui kondisimu.",
       "Pelan tapi konsisten jauh lebih berarti daripada perubahan besar yang cepat hilang."],
    2:["Kamu berada di rentang sehat. Pertahankan kebiasaan yang sudah berjalan baik.",
       "Keseimbangan adalah hasil dari kebiasaan kecil yang dijaga setiap hari.",
       "Tubuhmu bekerja baik, dan kamu yang menjaganya. Terus lanjutkan."],
    3:["Perubahan kecil hari ini akan terasa besar dalam beberapa bulan ke depan.",
       "Mulailah dari satu kebiasaan kecil, misalnya jalan kaki 10 menit setelah makan.",
       "Kamu tidak perlu sempurna, cukup lebih baik dari kemarin."],
    4:["Langkah kecil yang dimulai hari ini tetap sangat berarti untuk kesehatanmu.",
       "Kamu layak mendapat dukungan. Berbicara dengan dokter adalah langkah yang tepat.",
       "Tidak apa-apa untuk mulai pelan. Yang penting kamu sudah mulai."],
    5:["Kamu tidak sendirian. Dukungan medis dan orang terdekat bisa membantu langkah berikutnya.",
       "Prioritaskan keselamatanmu. Hubungi tenaga kesehatan untuk rencana yang aman.",
       "Setiap hari adalah kesempatan baru untuk merawat diri dengan lebih baik."],
}

@st.cache_resource
def load_artifacts():
    with open("model_bmi.pkl","rb") as f:  model  = pickle.load(f)
    with open("scaler_bmi.pkl","rb") as f: scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.sidebar.warning(f"Gagal memuat model: {e}")

# Citra biner 128x128 harus identik dengan notebook pelatihan karena menjadi
# sumber 5 fitur morfologi yang masuk ke StandardScaler dan Random Forest.
def generate_body_silhouette(height_cm, weight_kg, gender, img_size=128):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    h_norm = np.clip((height_cm - 140) / (200 - 140), 0.0, 1.2)
    w_norm = np.clip((weight_kg - 35)  / (165 - 35), 0.0, 1.2)
    body_h = max(20, int(0.5*img_size + h_norm*0.35*img_size))
    body_w = max(8, int(0.15*img_size + w_norm*0.3*img_size))
    shoulder_w = max(10, int(body_w*(1.3 if gender=="Male" else 1.1)))
    cx,cy = img_size//2, img_size//2
    cv2.ellipse(img,(cx,cy),(body_w,body_h//2),0,0,360,255,-1)
    cv2.ellipse(img,(cx,cy-body_h//2+10),(shoulder_w,10),0,0,360,255,-1)
    head_r = max(8, int(body_w*0.5))
    cv2.circle(img,(cx,cy-body_h//2-head_r+5),head_r,255,-1)
    return img

def extract_image_features(img):
    _,binary = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    pixel_area = np.sum(binary>0)
    coords = np.argwhere(binary>0)
    if len(coords)>0:
        y0,x0=coords.min(axis=0); y1,x1=coords.max(axis=0)
        bbox_h=y1-y0+1; bbox_w=x1-x0+1
    else: bbox_h=bbox_w=1
    aspect_ratio=bbox_h/(bbox_w+1e-5)
    body_density=pixel_area/((bbox_h*bbox_w)+1e-5)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    perimeter=cv2.arcLength(contours[0],True) if contours else 0
    compactness=(4*np.pi*pixel_area)/(perimeter**2+1e-5)
    return [pixel_area,round(aspect_ratio,4),round(body_density,4),round(perimeter,4),round(compactness,4)]

def normalize_gender(g):
    """Seragamkan variasi penulisan gender (male/M/Laki-laki/pria menjadi Male, selainnya Female)."""
    s = str(g).strip().lower()
    return "Male" if s in ("male", "m", "l", "laki-laki", "laki laki", "laki", "pria", "cowok", "1") else "Female"

@st.cache_data(show_spinner=False, max_entries=128)
def build_feature_vector(gender, height, weight):
    gender=normalize_gender(gender)
    h_m=height/100; bmi=weight/(h_m**2)
    img_feats=extract_image_features(generate_body_silhouette(height,weight,gender))
    g_num=1 if gender=="Male" else 0
    h_cat=0 if height<=155 else 1 if height<=170 else 2 if height<=185 else 3
    w_cat=0 if weight<=60  else 1 if weight<=80  else 2 if weight<=100  else 3
    return [g_num,height,weight,h_m,bmi,bmi**2,weight/height,weight**2,height**2,h_cat,w_cat]+img_feats

@st.cache_data(show_spinner=False)
def load_dataset():
    df_d = pd.read_csv("bmi.csv")
    label_map_en = {0:"Extremely Weak",1:"Weak",2:"Normal",3:"Overweight",4:"Obesity",5:"Extreme Obesity"}
    df_d["Label"] = df_d["Index"].map(label_map_en)
    df_d["BMI"] = df_d["Weight"]/(df_d["Height"]/100)**2
    return df_d

def bmi_index(bmi):
    if bmi<16: return 0
    if bmi<18.5: return 1
    if bmi<25: return 2
    if bmi<30: return 3
    if bmi<40: return 4
    return 5

SUB_LABEL_EN = {
    0: "Extremely Weak",
    1: "Weak / Underweight",
    2: "Ideal Weight",
    3: "Pre-Obese",
    4: "Obesity Class I-II",
    5: "Extreme Obesity Class III",
}

def format_sub_label(idx):
    return SUB_LABEL_EN.get(idx, LABEL_MAP[idx])

SUB_LABEL_ID = {
    0: "Kekurangan Berat Parah",
    1: "Kurang Berat Badan",
    2: "Berat Ideal",
    3: "Pra-Obesitas",
    4: "Obesitas Kelas I-II",
    5: "Obesitas Kelas III",
}

def format_sub_label_id(idx):
    return SUB_LABEL_ID.get(idx, LABEL_ID[idx])

@st.cache_data(show_spinner=False)
def simulate_curve(gender, height, w_lo, w_hi):
    rows=[]
    for w in range(w_lo, w_hi+1):
        bmi_w = w/((height/100)**2)
        if model_loaded:
            fv=build_feature_vector(gender,height,w)
            p=int(model.predict(scaler.transform([fv]))[0])
        else:
            p=bmi_index(bmi_w)
        rows.append((w,round(bmi_w,2),p))
    return pd.DataFrame(rows,columns=["Berat","BMI","Pred"])

# Siluet anatomi tampak depan khusus tampilan UI (bukan input fitur ML).
# Proporsi kanon 8 kepala (satuan tinggi tubuh = 1):
# dagu 0.125, bahu 0.18, pusar 0.38, selangkangan 0.50, lutut 0.72, mata kaki 0.95.
SIL_X0, SIL_X1, SIL_Y0, SIL_Y1, SIL_RES = -0.36, 0.36, -0.01, 1.01, 640

def _smooth(xk, yk, xx):
    """Interpolasi monoton halus (PCHIP); fallback linear jika scipy tidak ada."""
    try:
        from scipy.interpolate import PchipInterpolator
        return PchipInterpolator(xk, yk)(xx)
    except Exception:
        return np.interp(xx, xk, yk)

def _arc(cx, cy, r, a0, a1, n=24):
    t = np.linspace(a0, a1, n)
    return np.c_[cx + r*np.cos(t), cy + r*np.sin(t)]

def _limb(ys, cx, wo, wi, side, cap_top=False, cap_bot=False, n=90):
    """Anggota tubuh: garis tengah cx(y), setengah lebar sisi luar wo(y) & dalam wi(y)."""
    yy = np.linspace(ys[0], ys[-1], n)
    c = _smooth(ys, cx, yy); o = _smooth(ys, wo, yy); i = _smooth(ys, wi, yy)
    outer = np.c_[side*(c+o), yy]; inner = np.c_[side*(c-i), yy]
    pts = [outer]
    if cap_bot:
        r = (o[-1]+i[-1])/2; m = side*(c[-1]+(o[-1]-i[-1])/2)
        a = _arc(m, yy[-1], r, 0, np.pi) if side > 0 else _arc(m, yy[-1], r, np.pi, 0)
        pts.append(a)
    pts.append(inner[::-1])
    if cap_top:
        r = (o[0]+i[0])/2; m = side*(c[0]+(o[0]-i[0])/2)
        a = _arc(m, yy[0], r, np.pi, 2*np.pi) if side > 0 else _arc(m, yy[0], r, 2*np.pi, np.pi)
        pts.append(a)
    return np.vstack(pts)

def body_polygons(gender, bmi):
    """Kumpulan poligon (satuan tinggi tubuh) membentuk siluet tampak depan."""
    male = gender == "Male"
    f = float(np.clip((bmi - 22.0) / 18.0, -0.45, 1.10))   # 0 = BMI 22 (ideal)
    grow = lambda c: 1.0 + c*f
    polys = []

    # Torso: (y, setengah lebar, koefisien pertumbuhan lemak)
    if male:
        T = [(0.150,0.030,0.35),(0.160,0.052,0.30),(0.172,0.082,0.20),(0.186,0.104,0.28),
             (0.205,0.110,0.32),(0.235,0.104,0.35),(0.265,0.101,0.50),(0.300,0.094,0.70),
             (0.340,0.087,0.95),(0.375,0.085,1.05),(0.410,0.089,0.90),(0.450,0.096,0.65),
             (0.485,0.094,0.55),(0.508,0.084,0.55)]
    else:
        T = [(0.150,0.026,0.35),(0.160,0.046,0.30),(0.172,0.072,0.20),(0.186,0.092,0.28),
             (0.205,0.096,0.32),(0.232,0.090,0.40),(0.258,0.096,0.55),(0.288,0.088,0.65),
             (0.330,0.077,0.90),(0.362,0.075,1.00),(0.405,0.092,0.85),(0.452,0.108,0.65),
             (0.485,0.106,0.55),(0.508,0.094,0.55)]
    ty = np.array([t[0] for t in T])
    tw = np.array([t[1]*grow(t[2]) for t in T])
    yy = np.linspace(ty[0], ty[-1], 120)
    ww = _smooth(ty, tw, yy)
    # Bagian bawah torso melengkung ke selangkangan (bukan potongan datar)
    polys.append(np.vstack([np.c_[ww, yy], [[0.0, ty[-1] + 0.014]], np.c_[-ww[::-1], yy[::-1]]]))
    torso_w = lambda y: float(np.interp(y, yy, ww, left=0.0, right=0.0))   # lebar torso di level y

    # Leher (trapezius menyatu ke bahu)
    nw = (0.029 if male else 0.023) * grow(0.35)
    polys.append(np.array([[-nw,0.100],[nw,0.100],[nw*1.08,0.150],[nw*1.35,0.168],
                           [-nw*1.35,0.168],[-nw*1.08,0.150]]))

    # Kepala: bentuk telur, rahang menyempit, pipi sedikit melebar saat BMI tinggi
    a = (0.049 if male else 0.045) * grow(0.12); b = 0.064; cy = 0.066
    t = np.linspace(0, 2*np.pi, 120)
    hx = a*np.sin(t); hy = cy - b*np.cos(t)
    low = hy > cy
    hx[low] *= 1 - (0.22 if male else 0.30) * ((hy[low]-cy)/b)**2 * (1 - 0.5*max(f, 0))
    polys.append(np.c_[hx, hy])

    # Lengan: deltoid, bisep, siku, lengan bawah, pergelangan, dan tangan.
    ay = [0.198,0.228,0.265,0.315,0.360,0.420,0.468,0.500,0.535,0.566]
    if male:
        acx = [0.104,0.114,0.122,0.128,0.133,0.139,0.145,0.148,0.149,0.149]
        awo = [0.030,0.031,0.028,0.022,0.025,0.022,0.016,0.020,0.020,0.013]
        awi = [0.024,0.026,0.024,0.019,0.022,0.019,0.014,0.019,0.017,0.010]
    else:
        acx = [0.092,0.102,0.110,0.117,0.123,0.130,0.137,0.140,0.141,0.141]
        awo = [0.025,0.026,0.024,0.019,0.021,0.019,0.013,0.017,0.017,0.011]
        awi = [0.020,0.022,0.020,0.016,0.019,0.016,0.012,0.016,0.015,0.009]
    ac = [0.35,0.55,0.60,0.30,0.45,0.35,0.12,0.15,0.12,0.08]
    awo_g = [w*grow(c) for w, c in zip(awo, ac)]
    awi_g = [w*grow(c) for w, c in zip(awi, ac)]
    # Geser lengan keluar agar tidak menembus torso saat BMI tinggi.
    gap = [0.0, 0.0, 0.004, 0.008, 0.009, 0.010, 0.012, 0.012, 0.012, 0.012]
    base_x = [x + (wg - w) for x, w, wg in zip(acx, awi, awi_g)]
    req = [torso_w(y) + wg + gp if gp > 0 else -1 for y, wg, gp in zip(ay, awi_g, gap)]
    shift = max(0.0, max(r - x for r, x in zip(req, base_x)))
    wt = [0.45, 0.65, 0.85, 1, 1, 1, 1, 1, 1, 1]
    acx_g = [max(x + shift*k, r) for x, r, k in zip(base_x, req, wt)]
    for s in (-1, 1):
        polys.append(_limb(ay, acx_g, awo_g, awi_g, s, cap_top=True, cap_bot=True))

    # Tungkai: paha, lutut, betis, dan pergelangan kaki.
    ly = [0.440,0.500,0.560,0.630,0.700,0.725,0.765,0.805,0.860,0.920,0.952]
    if male:
        lcx = [0.050,0.054,0.055,0.053,0.050,0.049,0.049,0.049,0.047,0.045,0.044]
        lwo = [0.046,0.050,0.046,0.040,0.031,0.030,0.034,0.035,0.028,0.019,0.017]
        lwi = [0.046,0.046,0.040,0.033,0.025,0.025,0.028,0.028,0.021,0.015,0.014]
    else:
        lcx = [0.056,0.058,0.056,0.051,0.046,0.045,0.046,0.047,0.046,0.044,0.043]
        lwo = [0.052,0.056,0.050,0.041,0.030,0.028,0.032,0.032,0.025,0.017,0.015]
        lwi = [0.050,0.048,0.041,0.032,0.023,0.023,0.026,0.026,0.019,0.013,0.012]
    lc = [0.55,0.60,0.55,0.45,0.30,0.26,0.34,0.34,0.22,0.10,0.08]
    lwo_g = [w*grow(c) for w, c in zip(lwo, lc)]
    lwi_g = [w*grow(c) for w, c in zip(lwi, lc)]
    lcx_g = [x + 0.6*(wg - w) for x, w, wg in zip(lcx, lwi, lwi_g)]
    for s in (-1, 1):
        polys.append(_limb(ly, lcx_g, lwo_g, lwi_g, s))
        # Telapak kaki (tampak depan, ujung sedikit mengarah keluar)
        ax_ = lcx_g[-1]; fw = 0.021*grow(0.10)
        polys.append(np.array([[s*(ax_-fw*0.75),0.945],[s*(ax_+fw*0.85),0.945],
                               [s*(ax_+fw*1.15),0.975],[s*(ax_+fw*1.05),0.990],
                               [s*(ax_-fw*0.80),0.990],[s*(ax_-fw*0.90),0.970]]))
        e = _arc(0, 0, 1.0, 0, 2*np.pi, 40)
        polys.append(np.c_[s*(ax_+fw*0.30) + e[:,0]*fw*1.15, 0.988 + e[:,1]*0.012])
    return polys

def _rasterize(polys):
    """Gabungkan poligon menjadi mask anti-alias (tanpa sambungan antar bagian)."""
    S = SIL_RES
    w = int((SIL_X1-SIL_X0)*S); h = int((SIL_Y1-SIL_Y0)*S)
    mask = np.zeros((h, w), np.uint8)
    for p in polys:   # diisi satu per satu agar bagian yang tumpang-tindih tidak berlubang
        pts = np.round(np.c_[(p[:,0]-SIL_X0)*S, (p[:,1]-SIL_Y0)*S]*16).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_AA, shift=4)
    return mask

def _shade(mask, hex_color):
    """Pencahayaan volumetrik halus dari height-map multi-skala ke Lambert dan spekular."""
    m = mask.astype(np.float32)/255.0
    z = (0.50*cv2.GaussianBlur(m, (0, 0), 3.5) +
         0.35*cv2.GaussianBlur(m, (0, 0), 9.0) +
         0.15*cv2.GaussianBlur(m, (0, 0), 20.0))
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=5)
    k = 0.55
    nx, ny, nz = -gx*k, -gy*k, np.ones_like(z)
    nn = np.sqrt(nx**2 + ny**2 + nz**2); nx, ny, nz = nx/nn, ny/nn, nz/nn
    L = np.array([-0.45, -0.50, 0.74]); L = L/np.linalg.norm(L)
    diff = np.clip(nx*L[0] + ny*L[1] + nz*L[2], 0, 1)
    base = np.array(matplotlib.colors.to_rgb(hex_color))
    shade = 0.28 + 0.82*diff
    spec = np.clip(diff, 0, 1)**30*0.30
    rgb = base[None, None, :]*shade[..., None] + spec[..., None]
    # Oklusi lembut di tepi tubuh (memberi kesan kedalaman)
    edge = cv2.GaussianBlur(m, (0, 0), 2.0)
    rgb = rgb*(0.70 + 0.30*np.clip(edge*1.6 - 0.3, 0, 1))[..., None]
    rgba = np.dstack([np.clip(rgb, 0, 1), mask.astype(np.float32)/255.0])
    return rgba

@st.cache_data(show_spinner=False)
def _ideal_contours(gender):
    m_ref = _rasterize(body_polygons(gender, 22.0))
    cnts, _ = cv2.findContours((m_ref > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cn[:, 0, :].astype(float) for cn in cnts]

def draw_human_silhouette(height, weight, gender, idx):
    """
    Siluet tampak depan yang realistis & proporsional:
    - Proporsi kanon ±8 kepala, diskalakan sesuai tinggi badan (relatif 175 cm)
    - Bentuk bahu, pinggang, pinggul, dan tungkai berbeda untuk laki-laki/perempuan
    - Distribusi lemak per area (perut, pinggul, paha, lengan) mengikuti BMI
    - Garis putus-putus = siluet pembanding pada BMI ideal (22) dengan tinggi yang sama
    Ini ILUSTRASI skematis, bukan representasi wajah atau bentuk tubuh spesifik.
    """
    bmi = weight/((height/100)**2)
    H = height/175.0
    c = CHART_COLORS[idx]

    fig, ax = plt.subplots(figsize=(3.6, 5.6))
    fig.patch.set_facecolor(BG_CARD); ax.set_facecolor(BG_CARD)

    # Bayangan lantai
    ax.add_patch(Ellipse((0, 1.0*H+0.006), 0.42*H, 0.030*H, color="#000000", alpha=SH_A1, zorder=1, linewidth=0))
    ax.add_patch(Ellipse((0, 1.0*H+0.006), 0.26*H, 0.016*H, color="#000000", alpha=SH_A2, zorder=1, linewidth=0))

    # Tubuh (render volumetrik)
    mask = _rasterize(body_polygons(gender, bmi))
    ax.imshow(_shade(mask, c), extent=(SIL_X0*H, SIL_X1*H, SIL_Y1*H, SIL_Y0*H),
              interpolation="bilinear", zorder=3)

    # Garis pembanding BMI ideal
    if abs(bmi - 22.0) > 2.0:
        for p in _ideal_contours(gender):
            px = (p[:, 0]/SIL_RES + SIL_X0)*H; py = (p[:, 1]/SIL_RES + SIL_Y0)*H
            ax.plot(np.r_[px, px[0]], np.r_[py, py[0]], color=SIL_REF, linewidth=0.7,
                    linestyle=(0, (3, 2)), alpha=0.55, zorder=4)

    # Garis dimensi tinggi
    xd = 0.33*H
    ax.plot([xd, xd], [0.0, 1.0*H], color=TXT_SUB, linewidth=0.8, zorder=2)
    for yv in (0.0, 1.0*H):
        ax.plot([xd-0.015, xd+0.015], [yv, yv], color=TXT_SUB, linewidth=0.8)
    ax.text(xd+0.035, 0.5*H, f"{height} cm", color=TXT_SUB, fontsize=9, va="center", ha="left", rotation=90)

    ax.text(0, 1.0*H+0.09, f"{LABEL_ID[idx]}  ·  IMT {bmi:.1f}".replace(".", ","), ha="center", fontsize=9,
            fontweight="bold", color=TXT_BODY)
    if abs(bmi - 22.0) > 2.0:
        ax.text(0, 1.0*H+0.15, "- - -  siluet IMT ideal (22)", ha="center", fontsize=7, color=TXT_SUB)

    ax.set_xlim(-0.46, 0.50); ax.set_ylim(max(1.36, 1.0*H + 0.22), -0.06)
    ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig

def _clock_body():
    now = datetime.now(WIB)
    st.markdown(
        f'<div class="clock"><span class="clock-dot"></span>'
        f'<span id="wib-clock-date">{fmt_hari_tanggal(now)}</span>'
        f'<b id="wib-clock-time">{now:%H:%M:%S}</b>'
        f'<span class="clock-tz">GMT+7 · WIB</span></div>',
        unsafe_allow_html=True)
    components.html("""
    <script>
    (function() {
        const HARI = ["Minggu","Senin","Selasa","Rabu","Kamis","Jumat","Sabtu"];
        const BULAN = ["Januari","Februari","Maret","April","Mei","Juni","Juli",
                       "Agustus","September","Oktober","November","Desember"];
        function tick() {
            try {
                const doc = window.parent.document;
                const elTime = doc.getElementById("wib-clock-time");
                const elDate = doc.getElementById("wib-clock-date");
                if (!elTime) return;
                const now = new Date(Date.now() + 7 * 3600 * 1000);
                const hh = String(now.getUTCHours()).padStart(2, "0");
                const mm = String(now.getUTCMinutes()).padStart(2, "0");
                const ss = String(now.getUTCSeconds()).padStart(2, "0");
                elTime.textContent = hh + ":" + mm + ":" + ss;
                if (elDate) {
                    elDate.textContent = HARI[now.getUTCDay()] + ", " +
                        now.getUTCDate() + " " + BULAN[now.getUTCMonth()] + " " + now.getUTCFullYear();
                }
            } catch (e) {}
        }
        if (window.parent && !window.parent.__wibClockTimer) {
            window.parent.__wibClockTimer = window.parent.setInterval(tick, 1000);
        }
        tick();
    })();
    </script>
    """, height=0, width=0)

live_clock = _clock_body

with st.sidebar:
    st.markdown(
        '<div class="sb-head">'
        '<p class="sb-title">Parameter Tubuh</p>'
        '<p class="sb-sub">Prediksi diperbarui otomatis</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        gender = st.radio(
            "Jenis Kelamin",
            ["Male", "Female"],
            horizontal=True,
            format_func=lambda x: "Laki-laki" if x == "Male" else "Perempuan",
        )
        st.markdown(
            f'<div class="ctrl-row"><span class="ctrl-lbl">Tinggi Badan</span>'
            f'<span class="ctrl-readout">{st.session_state.inp_height}<small>cm</small></span></div>',
            unsafe_allow_html=True,
        )
        height = st.slider("Tinggi Badan (cm)", 140, 200, key="inp_height", label_visibility="collapsed")

        st.markdown(
            f'<div class="ctrl-row"><span class="ctrl-lbl">Berat Badan</span>'
            f'<span class="ctrl-readout">{st.session_state.inp_weight}<small>kg</small></span></div>',
            unsafe_allow_html=True,
        )
        weight = st.slider("Berat Badan (kg)", 35, 165, key="inp_weight", label_visibility="collapsed")

        st.markdown(
            f'<div class="ctrl-row"><span class="ctrl-lbl">Target Berat</span>'
            f'<span class="ctrl-readout">{st.session_state.inp_target}<small>kg</small></span></div>',
            unsafe_allow_html=True,
        )
        target_w = st.slider("Target Berat Badan (kg)", 35, 165, key="inp_target", label_visibility="collapsed")

    bmi_val=weight/(height/100)**2
    bi=bmi_index(bmi_val)

    prob_main = np.zeros(6)
    if model_loaded:
        fv_main = build_feature_vector(gender, height, weight)
        fv_main_sc = scaler.transform([fv_main])
        pred_main = int(model.predict(fv_main_sc)[0])
        proba_raw = model.predict_proba(fv_main_sc)[0]
        for idx_c, cls_c in enumerate(model.classes_):
            prob_main[int(cls_c)] = proba_raw[idx_c]
        conf_main = float(np.max(prob_main) * 100.0)
    else:
        pred_main = bi
        prob_main[bi] = 1.0
        conf_main = 100.0

    if st.session_state.last_idx is not None and st.session_state.last_idx != pred_main:
        st.toast(f"Prediksi Random Forest: {LABEL_ID[pred_main]}")
    st.session_state.last_idx = pred_main

    il=round(18.5*(height/100)**2); ih=round(24.9*(height/100)**2)
    bmi_t_side=target_w/((height/100)**2)
    bmi_side_str = f"{bmi_val:.1f}".replace(".", ",")
    bmit_side_str = f"{bmi_t_side:.1f}".replace(".", ",")
    bp_side = min(max((bmi_val - 10.0) / 40.0 * 100.0, 2.0), 98.0)
    st.markdown(
        f'<div class="rail-card">'
        f'<span class="rail-lbl">Indeks Massa Tubuh</span>'
        f'<div class="rail-hero">'
        f'<p class="rail-val">{bmi_side_str}<span class="rail-unit">kg/m²</span></p>'
        f'<span class="status-tag tag-c{pred_main}">{LABEL_ID[pred_main]}</span>'
        f'</div>'
        f'<div class="rail-meter"><i style="left:{bp_side:.2f}%"></i></div>'
        f'<div class="rail-ledger">'
        f'<div class="rail-row"><span class="rail-k">Prediksi RF</span><span class="rail-v">{LABEL_ID[pred_main]} · {conf_main:.0f}%</span></div>'
        f'<div class="rail-row"><span class="rail-k">Rentang Ideal</span><span class="rail-v">{il}–{ih} kg</span></div>'
        f'<div class="rail-row"><span class="rail-k">Target Berat</span><span class="rail-v">{target_w} kg · IMT {bmit_side_str}</span></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("Simpan ke Riwayat", type="primary", use_container_width=True, icon=":material/bookmark_add:"):
        pred_lbl = LABEL_ID[pred_main] if model_loaded else "-"
        st.session_state.riwayat.append({
            "Waktu (WIB)": fmt_waktu_riwayat(datetime.now(WIB)),
            "Gender": "Laki-laki" if gender=="Male" else "Perempuan",
            "Tinggi (cm)": height,
            "Berat (kg)": weight,
            "BMI": round(bmi_val,2),
            "Kategori (IMT)": LABEL_ID[bi],
            "Prediksi Random Forest": pred_lbl,
        })
        st.toast("Pengukuran tersimpan ke riwayat")

st.markdown("""
<div class="page-header">
<div>
<h1 class="page-title">Klasifikasi Indeks Massa Tubuh (IMT)</h1>
<p class="page-sub">Simulasi Citra Siluet Biner OpenCV &amp; Algoritma Random Forest Berbasis Data Antropometri</p>
</div>
<div class="page-id">
<span class="page-id-kicker">Skripsi S1 Informatika · AMIKOM Yogyakarta</span>
<strong>Satrio Raja Nabil Rizky (23.11.5538)</strong>
<span class="page-id-line">Pembimbing: Mulia Sulistiyono, S.Kom., M.Kom.</span>
</div>
</div>
""", unsafe_allow_html=True)

t1,t2,t3,t4,t5=st.tabs([
    "Klasifikasi IMT",
    "Analisis Data (n=3.700)",
    "Distribusi Probabilitas",
    "Evaluasi 6 Model ML",
    "Rekomendasi & Kalori",
])

with t1:
    ca,cb=st.columns([1,1.85])
    with ca:
        with st.container(border=True):
            st.markdown("#### Siluet Tubuh")
            fig_s=draw_human_silhouette(height,weight,gender,bi)
            st.pyplot(fig_s,use_container_width=True); plt.close(fig_s)
            st.caption("Ilustrasi 3D proporsional (±8 kepala). Garis putus-putus = pembanding IMT ideal 22.")

        with st.expander("Citra Biner OpenCV & 5 Fitur Morfologi", expanded=False):
            img_cv = generate_body_silhouette(height, weight, gender)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
            cnts, _ = cv2.findContours(img_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_rgb, cnts, -1, CONTOUR_RGB, 2)
            st.image(img_rgb, caption="Citra siluet biner OpenCV (128×128 px) dengan kontur", use_container_width=True)
            f_img = extract_image_features(img_cv)
            render_table(pd.DataFrame({
                "Fitur Morfologi": ["pixel_area", "aspect_ratio", "body_density", "contour_perimeter", "compactness"],
                "Nilai": [f"{int(f_img[0])} px²", f"{f_img[1]:.4f}", f"{f_img[2]:.4f}", f"{f_img[3]:.4f}", f"{f_img[4]:.4f}"],
            }))

    with cb:
        if weight < il:
            diff = il - weight
            gap_short = f"-{diff:.1f} kg".replace(".", ",")
            gap_sub = f"Rentang WHO {il}–{ih} kg"
            gap_tile_cls = "tile-amber"
        elif weight > ih:
            diff = weight - ih
            gap_short = f"+{diff:.1f} kg".replace(".", ",")
            gap_sub = f"Rentang WHO {il}–{ih} kg"
            gap_tile_cls = "tile-rose"
        else:
            gap_short = "0,0 kg (Ideal)"
            gap_sub = f"Rentang WHO {il}–{ih} kg"
            gap_tile_cls = "tile-emerald"

        bmi_t = target_w / ((height / 100) ** 2)
        cat_t = bmi_index(bmi_t)
        d_t = weight - target_w
        delta_t_str = (f"-{d_t:.0f} kg" if d_t > 0 else f"+{-d_t:.0f} kg" if d_t < 0 else "Tercapai")

        pred = pred_main
        conf = conf_main
        gauge_svg_html = build_bmi_gauge_svg(bmi_val, pred, LABEL_ID[pred])
        bp = min(max((bmi_val - 10.0) / 40.0 * 100.0, 2.0), 98.0)
        bmi_comma = f"{bmi_val:.1f}".replace(".", ",")
        conf_comma = f"{conf:.1f}%".replace(".", ",")
        weight_comma = f"{weight:.0f}"
        bmit_comma = f"{bmi_t:.1f}".replace(".", ",")
        gender_id = "Laki-laki" if gender == "Male" else "Perempuan"

        # Satu Kartu Utama Terpadu: Sorotan Random Forest + 3 Metrik Kunci + Arc Gauge + Bar IMT
        st.markdown(
            f'<div class="diag-hero">'
            f'<div class="hero-top">'
            f'<div>'
            f'<p class="hero-title">Prediksi Random Forest</p>'
            f'<p class="hero-sub">Model Terbaik (Akurasi 99,73% · 16 Fitur) · {gender_id}, {height} cm, {weight_comma} kg</p>'
            f'</div>'
            f'<span class="tint-pill pill-emerald">Akurasi RF: <b>99,73%</b> (Puncak 99,78%)</span>'
            f'</div>'
            f'<div class="tile-grid">'
            f'<div class="metric-tile tile-blue">'
            f'<p class="tile-lbl">Klasifikasi Random Forest</p>'
            f'<p class="tile-val tone-c{pred}">{LABEL_ID[pred]}</p>'
            f'<p class="tile-meta">Keyakinan {conf_comma} · {format_sub_label_id(pred)}</p>'
            f'<div class="conf-bar"><i style="width:{min(conf, 100.0):.1f}%"></i></div>'
            f'</div>'
            f'<div class="metric-tile {gap_tile_cls}">'
            f'<p class="tile-lbl">Selisih Berat Ideal</p>'
            f'<p class="tile-val">{gap_short}</p>'
            f'<p class="tile-meta">{gap_sub}</p>'
            f'</div>'
            f'<div class="metric-tile tile-amber">'
            f'<p class="tile-lbl">Target Berat</p>'
            f'<p class="tile-val">{target_w} kg ({delta_t_str})</p>'
            f'<p class="tile-meta">IMT {bmit_comma} · {LABEL_ID[cat_t]}</p>'
            f'</div>'
            f'</div>'
            f'{gauge_svg_html}'
            f'<div class="capsule-bar-wrap">'
            f'<div class="capsule-head">'
            f'<span class="capsule-title">Indeks Massa Tubuh: <b>{bmi_comma} kg/m²</b></span>'
            f'<span class="capsule-status tone-c{pred}">{LABEL_ID[pred]} ({format_sub_label_id(pred)})</span>'
            f'</div>'
            f'<div class="capsule-track">'
            f'<div class="capsule-active-zone zone-{pred}"></div>'
            f'<div class="capsule-pointer" style="left:{bp:.2f}%"></div>'
            f'</div>'
            f'<div class="ruler-ticks">'
            f'<span class="tick-16">16</span>'
            f'<span class="tick-185">18,5</span>'
            f'<span class="tick-25">25</span>'
            f'<span class="tick-30">30</span>'
            f'<span class="tick-40">40</span>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Rumus & Acuan Kategori BMI / IMT (Standar WHO)", expanded=True):
            st.markdown(
                f'<div class="formula-strip">'
                f'<span class="formula-eq">IMT = Berat (kg) / Tinggi (m)²</span>'
                f'<span>Saat ini: {weight} / ({height/100:.2f})² = <b class="formula-val">{bmi_val:.2f} kg/m²</b> ({LABEL_ID[bi]})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            who_rows = [
                (0, "< 16,0",        "Sangat Kurus",    "Extremely Weak",  "Underweight (< 18,5)"),
                (1, "16,0 – 18,49",  "Kurus",           "Weak",            "Underweight (< 18,5)"),
                (2, "18,5 – 24,99",  "Normal",          "Normal (Ideal)",  "Normal (18,5 – 24,9)"),
                (3, "25,0 – 29,99",  "Overweight",      "Overweight",      "Overweight (25,0 – 29,9)"),
                (4, "30,0 – 39,99",  "Obesitas",        "Obesity",         "Obesitas (≥ 30,0)"),
                (5, "≥ 40,0",        "Sangat Obesitas", "Extreme Obesity", "Obesitas (≥ 30,0)"),
            ]
            rows_html = []
            for idx_r, rng, id_lbl, en_lbl, who_grp in who_rows:
                active = (idx_r == bi)
                row_cls = ' class="who-row-active"' if active else ""
                mark = f' <span class="status-tag status-tag-sm tag-c{idx_r}">Saat ini</span>' if active else ""
                rows_html.append(
                    f'<tr{row_cls}>'
                    f'<td class="who-range">{rng}</td>'
                    f'<td><span class="who-label tone-c{idx_r}">{id_lbl}</span>'
                    f'<span class="who-en">({en_lbl})</span>{mark}</td>'
                    f'<td class="who-group">{who_grp}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div class="tbl-wrap"><table class="who-tbl">'
                f'<thead><tr>'
                f'<th>Rentang IMT</th>'
                f'<th>Kategori Model (6 Kelas)</th>'
                f'<th>Kelompok WHO</th>'
                f'</tr></thead><tbody>{"".join(rows_html)}</tbody></table></div>',
                unsafe_allow_html=True,
            )

        with st.expander("Rincian Atribut Antropometri & 5 Fitur Morfologi Citra", expanded=False):
            info_df=pd.DataFrame({
                "Atribut / Fitur":["Tinggi Badan","Berat Badan","BMI / IMT","Kategori (EN)","Kategori (ID)","Rentang Berat Ideal","Pixel Area (Citra)","Aspect Ratio (Citra)","Body Density (Citra)","Contour Perimeter","Compactness"],
                "Nilai":[f"{height} cm",f"{weight} kg",f"{bmi_val:.2f} kg/m²",LABEL_MAP[bi],LABEL_ID[bi],f"{il}–{ih} kg",f"{int(f_img[0])} px²",f"{f_img[1]:.4f}",f"{f_img[2]:.4f}",f"{f_img[3]:.4f}",f"{f_img[4]:.4f}"]
            })
            render_table(info_df)
        st.caption("Hasil ini ditujukan untuk edukasi dan simulasi akademik, bukan diagnosis medis.")

    st.markdown('<div class="spacer-3"></div>', unsafe_allow_html=True)
    cc1,cc2=st.columns([1.15, 1])
    with cc1:
        with st.container(border=True):
            st.markdown("#### Simulasi Target Berat")
            bmi_t=target_w/((height/100)**2)
            cat_t=bmi_index(bmi_t)
            d_t=weight-target_w
            if d_t>0: msg=f"Perlu turun {d_t} kg untuk mencapai target {target_w} kg."
            elif d_t<0: msg=f"Perlu naik {-d_t} kg untuk mencapai target {target_w} kg."
            else: msg="Berat saat ini sudah sama dengan target."
            st.markdown(
                f'<div class="sim-grid">'
                f'<div class="sim-box"><p class="sim-lbl">BMI pada target ({target_w} kg)</p><p class="sim-val">{bmi_t:.1f}</p></div>'
                f'<div class="sim-box"><p class="sim-lbl">Kategori pada target</p><p class="sim-val tone-c{cat_t}">{LABEL_ID[cat_t]}</p></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(msg)

    with cc2:
        with st.container(border=True):
            st.markdown("#### Pesan Afirmasi")
            if st.button("Tampilkan pesan lain", key="aff_btn"):
                st.session_state.aff_idx += 1
            msgs=AFFIRMATIONS[bi]
            idx_msg=st.session_state.aff_idx % len(msgs)
            st.markdown(f'<p class="aff-text">"{msgs[idx_msg]}"</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="aff-note">Pesan {idx_msg+1} dari {len(msgs)} untuk kategori {LABEL_ID[bi]}</p>', unsafe_allow_html=True)

    st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
    st.markdown("#### Riwayat Pengukuran (sesi ini)")
    live_clock()
    if st.session_state.riwayat:
        dfr=pd.DataFrame(st.session_state.riwayat)
        render_table(dfr)
        if len(dfr)>=2:
            fig,ax=plt.subplots(figsize=(8,3.2))
            style_ax(ax,grid_axis="y")
            xs=list(range(1,len(dfr)+1))
            ax.fill_between(xs,dfr["BMI"],color=PRIMARY,alpha=0.14,zorder=1)
            ax.plot(xs,dfr["BMI"],color=PRIMARY,linewidth=1.8,zorder=2)
            ax.scatter(xs,dfr["BMI"],color=ACCENT,s=42,zorder=3,edgecolors=BG_CARD,linewidths=1.5)
            ax.axhline(18.5,color=ACCENT_2,linestyle=":",linewidth=0.9)
            ax.axhline(25,color=ACCENT_2,linestyle=":",linewidth=0.9)
            ax.set_xlabel("Urutan pengukuran"); ax.set_ylabel("BMI")
            ax.set_xticks(xs)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        if st.button("Hapus riwayat", key="clear_hist"):
            st.session_state.riwayat=[]
            st.rerun()
    else:
        st.markdown(
            '<div class="empty-box"><strong>Belum ada catatan pengukuran pada sesi ini.</strong>'
            'Atur tinggi dan berat badan di panel kiri, lalu tekan tombol "Simpan ke Riwayat" untuk mulai merekam perubahan BMI.</div>',
            unsafe_allow_html=True,
        )

with t2:
    try:
        df_data=load_dataset()
        label_map_en={0:"Extremely Weak",1:"Weak",2:"Normal",3:"Overweight",4:"Obesity",5:"Extreme Obesity"}

        st.markdown(f"""
<div class="stat-strip">
  <div class="stat-cell"><p class="stat-lbl">Total dataset gabungan</p><p class="stat-val">{len(df_data):,}</p></div>
  <div class="stat-cell"><p class="stat-lbl">Fitur prediktor</p><p class="stat-val">16 <span class="stat-sub">(11 + 5)</span></p></div>
  <div class="stat-cell"><p class="stat-lbl">Kategori IMT</p><p class="stat-val">6 kelas</p></div>
  <div class="stat-cell"><p class="stat-lbl">Akurasi terbaik (RF 75:25)</p><p class="stat-val tone-primary">99,78%</p></div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="spacer-3"></div>', unsafe_allow_html=True)
        fc1,fc2=st.columns([1,2])
        with fc1:
            g_pick=st.radio("Filter gender",["Semua","Laki-laki","Perempuan"],horizontal=True)
        with fc2:
            cat_pick=st.multiselect("Filter kategori IMT",list(label_map_en.values()),default=list(label_map_en.values()))

        df_view=df_data.copy()
        if g_pick!="Semua":
            df_view=df_view[df_view["Gender"]==("Male" if g_pick=="Laki-laki" else "Female")]
        df_view=df_view[df_view["Label"].isin(cat_pick)]
        st.caption(f"Menampilkan {len(df_view):,} dari {len(df_data):,} sampel")

        st.markdown('<div class="spacer-2"></div>', unsafe_allow_html=True)
        ca2,cb2=st.columns(2)
        with ca2:
            with st.container(border=True):
                st.markdown("#### Distribusi Kategori IMT (Dataset Gabungan)")
                fig,ax=plt.subplots(figsize=(6,4))
                style_ax(ax,grid_axis="y")
                cnt=df_data["Index"].value_counts().sort_index()
                bars=ax.bar([label_map_en[i] for i in cnt.index],cnt.values,color=[CAT_COLORS[i] for i in cnt.index],alpha=0.88,edgecolor=BG_CARD,linewidth=0.5,width=0.6)
                for b,v in zip(bars,cnt.values): ax.text(b.get_x()+b.get_width()/2,b.get_height()+10,str(v),ha="center",va="bottom",color=TXT_MAIN,fontsize=9)
                ax.set_ylabel("Jumlah Sampel"); ax.tick_params(axis="x",rotation=25,labelsize=7)
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        with cb2:
            with st.container(border=True):
                st.markdown("#### Distribusi Gender")
                fig,ax=plt.subplots(figsize=(4,4))
                gc=df_data["Gender"].value_counts()
                wedges,texts,autotexts=ax.pie(gc,labels=gc.index,autopct="%1.1f%%",colors=[ACCENT,ACCENT_2],startangle=90,wedgeprops={"edgecolor":BG_CARD,"linewidth":2},textprops={"color":TXT_BODY})
                for a in autotexts: a.set_color(PIE_TXT); a.set_fontweight("bold")
                ax.set_facecolor(BG_CARD)
                st.pyplot(fig,use_container_width=True); plt.close(fig)

        with st.container(border=True):
            st.markdown("#### Scatter Plot: Tinggi vs Berat (Sesuai Filter)")
            if len(df_view) == 0:
                st.markdown(
                    '<div class="empty-box"><strong>Tidak ada data yang cocok dengan filter saat ini.</strong>'
                    'Pilih minimal satu kategori IMT pada kolom filter di atas untuk menampilkan sebaran tinggi dan berat badan.</div>',
                    unsafe_allow_html=True,
                )
            else:
                fig,ax=plt.subplots(figsize=(10,5))
                style_ax(ax,grid_axis="both")
                for idx2 in sorted(df_view["Index"].unique()):
                    sub=df_view[df_view["Index"]==idx2]
                    ax.scatter(sub["Height"],sub["Weight"],c=CAT_COLORS[idx2],label=label_map_en[idx2],alpha=0.8,s=26,edgecolors="none")
                ax.set_xlabel("Tinggi Badan (cm)"); ax.set_ylabel("Berat Badan (kg)")
                ax.legend(fontsize=8,labelcolor=TXT_BODY,facecolor=BG_RAISE,edgecolor=BORDER_2)
                st.pyplot(fig,use_container_width=True); plt.close(fig)

        st.markdown("#### Preview Dataset (15 Baris Pertama, Sesuai Filter)")
        if len(df_view) > 0:
            render_table(df_view[["Gender","Height","Weight","BMI","Label"]].head(15).assign(BMI=lambda d: d["BMI"].round(2)))
        else:
            st.caption("Tabel pratinjau kosong karena tidak ada kategori yang dipilih.")

    except FileNotFoundError:
        st.warning("Pastikan file bmi.csv tersedia di repositori.")

with t3:
    st.markdown("#### Distribusi Probabilitas Prediksi")

    if model_loaded:
        fv2=build_feature_vector(gender,height,weight)
        fv2_sc=scaler.transform([fv2])
        proba2=model.predict_proba(fv2_sc)[0]
        pred2=model.predict(fv2_sc)[0]

        prob_full=np.zeros(6)
        for i,cls in enumerate(model.classes_): prob_full[cls]=proba2[i]

        st.markdown(f"Berdasarkan **Random Forest** (Accuracy 99.73% · split 80:20) untuk BMI saat ini: **{bmi_val:.1f}**")

        with st.container(border=True):
            for i,(cat,p) in enumerate(zip(LABEL_ID.values(),prob_full)):
                pct=p*100
                tag = f' <span class="status-tag status-tag-sm tag-c{i}">Prediksi</span>' if i==pred2 else ""
                st.markdown(
                    '<div class="prob-row">'
                    '<div class="prob-head">'
                    f'<span>{cat}<span class="prob-sub">{LABEL_MAP[i]}</span>{tag}</span>'
                    f'<span class="prob-val tone-c{i}">{pct:.1f}%</span>'
                    '</div>'
                    '<div class="prob-track">'
                    f'<div class="prob-fill bar-c{i}" style="width:{pct:.1f}%"></div>'
                    '</div>'
                    '</div>', unsafe_allow_html=True)

        ca3,cb3=st.columns(2)
        with ca3:
            with st.container(border=True):
                st.markdown("#### Distribusi Probabilitas (Bar)")
                fig,ax=plt.subplots(figsize=(6,4))
                style_ax(ax,grid_axis="y")
                bars=ax.bar(list(LABEL_ID.values()),prob_full*100,color=CAT_COLORS,alpha=0.88,edgecolor=BG_CARD,linewidth=0.5,width=0.6)
                for b,v in zip(bars,prob_full*100): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,f"{v:.1f}%",ha="center",va="bottom",color=TXT_MAIN,fontsize=8)
                ax.set_ylabel("Probabilitas (%)"); ax.set_ylim(0,115); ax.tick_params(axis="x",rotation=20,labelsize=7)
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        with cb3:
            with st.container(border=True):
                st.markdown("#### Distribusi Probabilitas (Pie)")
                fig,ax=plt.subplots(figsize=(5.4,4.2))
                nz_idx=[i for i,p in enumerate(prob_full) if p>0]
                nz_vals=[prob_full[i] for i in nz_idx]
                nz_clrs=[CHART_COLORS[i] for i in nz_idx]
                ws,_,ats=ax.pie(
                    nz_vals,
                    labels=None,
                    autopct=lambda p:f"{p:.1f}%" if p>=8 else "",
                    pctdistance=0.79,
                    colors=nz_clrs,
                    startangle=90,
                    wedgeprops={"width":0.38,"edgecolor":BG_CARD,"linewidth":1.5},
                )
                for x in ats: x.set_color(PIE_TXT); x.set_fontsize(8); x.set_fontweight("bold")
                ax.text(0,0.06,f"{prob_full[pred2]*100:.1f}%",ha="center",va="center",color=TXT_MAIN,fontsize=15,fontweight="bold")
                ax.text(0,-0.14,LABEL_ID[pred2],ha="center",va="center",color=TXT_SUB,fontsize=8.5)
                leg_handles=[mpatches.Patch(color=CHART_COLORS[i],label=f"{LABEL_ID[i]} ({prob_full[i]*100:.1f}%)") for i in range(6)]
                ax.legend(
                    handles=leg_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5,-0.02),
                    ncol=2,
                    frameon=False,
                    fontsize=7.8,
                    labelcolor=TXT_BODY,
                    handlelength=1.1,
                    columnspacing=1.2,
                )
                ax.set_aspect("equal")
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

        render_table(pd.DataFrame({"Kategori ID":list(LABEL_ID.values()),"Kategori EN":list(LABEL_MAP.values()),"Probabilitas (%)":[f"{p*100:.2f}%" for p in prob_full],"Prediksi":["YA" if i==pred2 else "-" for i in range(6)]}))

        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
        st.markdown("#### Simulasi Skenario Berat Badan")
        st.caption(f"Tinggi {height} cm · {'Laki-laki' if gender=='Male' else 'Perempuan'} · prediksi ML untuk setiap berat dalam rentang yang dipilih")
        w_lo,w_hi=st.slider("Rentang berat simulasi (kg)",35,165,(50,90))
        if not (w_lo<=weight<=w_hi):
            st.caption("Berat saat ini berada di luar rentang simulasi.")
        df_sim=simulate_curve(gender,height,w_lo,w_hi)
        with st.container(border=True):
            fig,ax=plt.subplots(figsize=(10,4))
            style_ax(ax,grid_axis="y")
            ax.plot(df_sim["Berat"],df_sim["BMI"],color=BORDER_2,linewidth=1,zorder=1)
            ax.scatter(df_sim["Berat"],df_sim["BMI"],c=[CAT_COLORS[p] for p in df_sim["Pred"]],s=22,edgecolors="none",zorder=3)
            for y in (18.5,25,30):
                ax.axhline(y,color=TXT_SUB,linestyle=":",linewidth=0.8)
            ax.scatter([weight],[bmi_val],marker="*",s=180,color=ACCENT_HI,zorder=4,label="Kondisi saat ini")
            ax.set_xlabel("Berat badan (kg)"); ax.set_ylabel("BMI")
            ax.legend(fontsize=8,labelcolor=TXT_BODY,facecolor=BG_RAISE,edgecolor=BORDER_2)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    else:
        st.warning("Model belum dimuat.")

with t4:
    st.markdown("#### Perbandingan Performa 6 Algoritma Machine Learning")
    st.markdown("Hasil eksperimen pada dataset gabungan `bmi.csv` · **n = 3,700 sampel** · **5 Skenario Rasio Split (90:10, 80:20, 75:25, 70:30, 60:40)** · **16 Fitur** (11 Fitur Antropometri + 5 Fitur Morfologi Citra Siluet)")

    REAL_RESULTS=[
        {"Model":"Random Forest",    "Acc (90:10)":"99.73%","Acc (80:20)":"99.73%","Acc (75:25)":"99.78%","Acc (70:30)":"99.64%","Acc (60:40)":"99.66%","F1-Score (80:20)":"99.73%"},
        {"Model":"Gradient Boosting","Acc (90:10)":"99.46%","Acc (80:20)":"99.59%","Acc (75:25)":"99.68%","Acc (70:30)":"99.55%","Acc (60:40)":"99.66%","F1-Score (80:20)":"99.60%"},
        {"Model":"Decision Tree",    "Acc (90:10)":"99.46%","Acc (80:20)":"99.59%","Acc (75:25)":"99.46%","Acc (70:30)":"99.19%","Acc (60:40)":"99.39%","F1-Score (80:20)":"99.60%"},
        {"Model":"SVM (RBF)",        "Acc (90:10)":"97.30%","Acc (80:20)":"97.57%","Acc (75:25)":"97.62%","Acc (70:30)":"97.39%","Acc (60:40)":"97.16%","F1-Score (80:20)":"97.54%"},
        {"Model":"Logistic Reg.",    "Acc (90:10)":"97.57%","Acc (80:20)":"97.03%","Acc (75:25)":"96.86%","Acc (70:30)":"96.94%","Acc (60:40)":"96.89%","F1-Score (80:20)":"96.86%"},
        {"Model":"KNN",              "Acc (90:10)":"96.22%","Acc (80:20)":"96.49%","Acc (75:25)":"96.65%","Acc (70:30)":"96.58%","Acc (60:40)":"95.34%","F1-Score (80:20)":"96.44%"},
    ]
    render_table(pd.DataFrame(REAL_RESULTS))

    SPLIT_INFO=[
        {"Skenario Rasio Split":"Eksperimen 1 (90:10)","Data Latih (Train)":"3,330 sampel (90%)","Data Uji (Testing)":"370 sampel (10%)","Akurasi Terbaik (RF)":"99.73%"},
        {"Skenario Rasio Split":"Eksperimen 2 (80:20)","Data Latih (Train)":"2,960 sampel (80%)","Data Uji (Testing)":"740 sampel (20%)","Akurasi Terbaik (RF)":"99.73%"},
        {"Skenario Rasio Split":"Eksperimen 3 (75:25)","Data Latih (Train)":"2,775 sampel (75%)","Data Uji (Testing)":"925 sampel (25%)","Akurasi Terbaik (RF)":"99.78%"},
        {"Skenario Rasio Split":"Eksperimen 4 (70:30)","Data Latih (Train)":"2,590 sampel (70%)","Data Uji (Testing)":"1,110 sampel (30%)","Akurasi Terbaik (RF)":"99.64%"},
        {"Skenario Rasio Split":"Eksperimen 5 (60:40)","Data Latih (Train)":"2,220 sampel (60%)","Data Uji (Testing)":"1,480 sampel (40%)","Akurasi Terbaik (RF)":"99.66%"},
    ]
    with st.expander("Lihat rincian pembagian jumlah data train & testing (5 varian split)"):
        render_table(pd.DataFrame(SPLIT_INFO))

    st.markdown('<div class="spacer-4"></div>', unsafe_allow_html=True)
    ca4,cb4=st.columns(2)
    with ca4:
        with st.container(border=True):
            SPLITS=["90:10","80:20","75:25","70:30","60:40"]
            sp=st.selectbox("Pilih skenario rasio split",SPLITS,index=1)
            st.markdown(f"#### Akurasi Model (Split {sp})")
            col_name=f"Acc ({sp})"
            pairs=sorted([(r["Model"],float(r[col_name].strip("%"))) for r in REAL_RESULTS],key=lambda x:x[1])
            fig,ax=plt.subplots(figsize=(6,4))
            style_ax(ax,grid_axis="x")
            names=[p[0] for p in pairs]; vals=[p[1] for p in pairs]
            best=max(vals)
            colors=[(ACCENT if v==best else TXT_SUB) for v in vals]
            brs=ax.barh(names,vals,color=colors,alpha=0.9,edgecolor=BG_CARD,linewidth=0.5)
            for b,v in zip(brs,vals): ax.text(v+0.1,b.get_y()+b.get_height()/2,f"{v:.2f}%",ha="left",va="center",color=TXT_MAIN,fontsize=9)
            ax.set_xlim(min(vals)-2,100.5); ax.set_xlabel("Accuracy (%)")
            ax.tick_params(labelsize=9)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
            st.caption("Model terbaik disorot dengan warna biru safir. F1-Score ditampilkan untuk skenario split 80:20.")

    with cb4:
        with st.container(border=True):
            st.markdown("#### Feature Importance (Random Forest)")
            FEAT_IMP=[("BMI²",32.71),("BMI",29.36),("Weight/Height",14.11),("Weight²",5.33),("Weight",5.17),("aspect_ratio",3.77),("Weight_category",2.53),("body_density",1.70)]
            fig,ax=plt.subplots(figsize=(6,4))
            style_ax(ax,grid_axis="x")
            img_feats_names=["aspect_ratio","pixel_area","body_density","contour_perimeter","compactness"]
            fn=[f[0] for f in FEAT_IMP]; fi=[f[1] for f in FEAT_IMP]
            clrs2=[(ACCENT_2 if f in img_feats_names else ACCENT) for f in fn]
            ax.barh(fn[::-1],fi[::-1],color=clrs2[::-1],alpha=0.9,edgecolor=BG_CARD,linewidth=0.5)
            ax.set_xlabel("Importance (%)"); ax.tick_params(labelsize=8)
            ind_p=mpatches.Patch(color=ACCENT,label="Fitur Antropometri"); teal_p=mpatches.Patch(color=ACCENT_2,label="Fitur Citra Siluet")
            ax.legend(handles=[ind_p,teal_p],fontsize=8,labelcolor=TXT_BODY,facecolor=BG_RAISE,edgecolor=BORDER_2)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    with st.container(border=True):
        st.markdown("#### Uji Kontribusi Fitur (Studi Ablasi): Peran 5 Fitur Morfologi Citra Siluet")
        st.caption(
            "Pengujian Random Forest pada 5 random seed (42, 7, 123, 2024, 999) dengan split 80:20 (n = 3.700) "
            "untuk mengukur daya diskriminatif 5 fitur morfologi citra biner (pixel_area, aspect_ratio, body_density, contour_perimeter, compactness)."
        )
        ABLATION_RESULTS = [
            {"Konfigurasi Fitur": "A. Tabular Lengkap (11 fitur antropometri, tanpa siluet)", "Jumlah Fitur": 11, "Akurasi (Mean ± Std)": "99,57% ± 0,16%", "Macro-F1 (Mean ± Std)": "98,70% ± 0,93%", "Catatan Evaluasi": "Baseline fitur numerik penuh (termasuk BMI & BMI²)"},
            {"Konfigurasi Fitur": "B. Gabungan Penuh (11 fitur antropometri + 5 fitur siluet)", "Jumlah Fitur": 16, "Akurasi (Mean ± Std)": "99,57% ± 0,16%", "Macro-F1 (Mean ± Std)": "98,70% ± 0,95%", "Catatan Evaluasi": "Konfigurasi model utama skripsi (mencapai 99,73% pada seed=42)"},
            {"Konfigurasi Fitur": "C. Tabular Tanpa Turunan BMI (8 fitur non-BMI, tanpa siluet)", "Jumlah Fitur": 8, "Akurasi (Mean ± Std)": "97,59% ± 0,22%", "Macro-F1 (Mean ± Std)": "94,84% ± 1,19%", "Catatan Evaluasi": "Tanpa variabel BMI, BMI², dan Weight/Height"},
            {"Konfigurasi Fitur": "D. Tabular Tanpa Turunan BMI + 5 Fitur Siluet", "Jumlah Fitur": 13, "Akurasi (Mean ± Std)": "97,14% ± 0,42%", "Macro-F1 (Mean ± Std)": "94,94% ± 1,00%", "Catatan Evaluasi": "Macro-F1 meningkat +0,10 poin saat fitur siluet ditambahkan"},
            {"Konfigurasi Fitur": "E. Antropometri Dasar Saja (Gender + Height + Weight)", "Jumlah Fitur": 3, "Akurasi (Mean ± Std)": "97,16% ± 0,15%", "Macro-F1 (Mean ± Std)": "94,83% ± 0,48%", "Catatan Evaluasi": "Hanya 3 variabel input mentah tanpa rekayasa fitur"},
            {"Konfigurasi Fitur": "F. Antropometri Dasar + 5 Fitur Morfologi Siluet", "Jumlah Fitur": 8, "Akurasi (Mean ± Std)": "97,24% ± 0,60%", "Macro-F1 (Mean ± Std)": "95,03% ± 1,29%", "Catatan Evaluasi": "Akurasi (+0,08%) & Macro-F1 (+0,20%) naik di atas input mentah"},
            {"Konfigurasi Fitur": "G. Gender + 5 Fitur Morfologi Citra Siluet Saja (Tanpa Tinggi, Berat, & BMI)", "Jumlah Fitur": 6, "Akurasi (Mean ± Std)": "95,81% ± 0,69%", "Macro-F1 (Mean ± Std)": "92,61% ± 1,68%", "Catatan Evaluasi": "Bukti mandiri: geometri siluet 128×128 px mampu mengklasifikasikan 6 kelas IMT dengan akurasi 95,81%"},
        ]
        render_table(pd.DataFrame(ABLATION_RESULTS))

    if model_loaded:
        try:
            @st.cache_data
            def get_confusion_matrix():
                df_test=pd.read_csv("bmi.csv")
                df_test["Height_m"]=df_test["Height"]/100
                df_test["BMI"]=df_test["Weight"]/(df_test["Height_m"]**2)
                df_test["BMI_squared"]=df_test["BMI"]**2
                df_test["Weight_Height_ratio"]=df_test["Weight"]/df_test["Height"]
                df_test["Weight_squared"]=df_test["Weight"]**2
                df_test["Height_squared"]=df_test["Height"]**2
                df_test["Gender_num"]=(df_test["Gender"]=="Male").astype(int)
                df_test["Height_category"]=pd.cut(df_test["Height"],bins=[0,155,170,185,300],labels=[0,1,2,3]).astype(int)
                df_test["Weight_category"]=pd.cut(df_test["Weight"],bins=[0,60,80,100,300],labels=[0,1,2,3]).astype(int)
                img_feats_all=[extract_image_features(generate_body_silhouette(r.Height,r.Weight,r.Gender)) for _,r in df_test.iterrows()]
                df_img2=pd.DataFrame(img_feats_all,columns=["pixel_area","aspect_ratio","body_density","contour_perimeter","compactness"])
                feat_cols=["Gender_num","Height","Weight","Height_m","BMI","BMI_squared","Weight_Height_ratio","Weight_squared","Height_squared","Height_category","Weight_category","pixel_area","aspect_ratio","body_density","contour_perimeter","compactness"]
                X_all=pd.concat([df_test[feat_cols[:11]],df_img2],axis=1)
                X_sc_all=scaler.transform(X_all)
                y_all=df_test["Index"]
                from sklearn.metrics import confusion_matrix as cm_fn
                from sklearn.model_selection import train_test_split as tts
                Xtr,Xte,ytr,yte=tts(pd.DataFrame(X_sc_all,columns=feat_cols),y_all,test_size=0.2,random_state=42,stratify=y_all)
                yp=model.predict(Xte)
                return cm_fn(yte,yp)

            cm_mat = get_confusion_matrix()
            with st.container(border=True):
                st.markdown("#### Confusion Matrix: Random Forest (Split 80:20)")
                fig,ax=plt.subplots(figsize=(8,5.5))
                tnames=["Ext.Weak","Weak","Normal","Overweight","Obesity","Ext.Obesity"]
                sns.heatmap(cm_mat,annot=True,fmt="d",cmap=CM_CMAP,ax=ax,xticklabels=tnames,yticklabels=tnames,linewidths=0.5,linecolor=BORDER)
                ax.set_xlabel("Predicted Label"); ax.set_ylabel("Actual Label")
                ax.tick_params(colors=TXT_BODY,labelsize=8); ax.tick_params(axis="x",rotation=25)
                ax.set_title("Confusion Matrix: Random Forest (Data Uji 20% = 740 Sampel)",color=TXT_MAIN,fontsize=11)
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        except Exception as e:
            st.warning(f"Gagal memuat Confusion Matrix: {e}")

with t5:
    st.markdown("#### Rekomendasi Kesehatan & Kalkulasi Pembakaran Kalori Personal")
    rec=RECOMMENDATIONS[pred_main]
    gender_lbl_t5 = "Laki-laki" if gender == "Male" else "Perempuan"
    st.markdown(
        f'<div class="cat-banner">'
        f'<p class="cat-banner-title tone-c{pred_main}">{rec["title"]}</p>'
        f'<p class="cat-banner-meta">Hasil Klasifikasi Random Forest: <strong class="cat-meta-em">{LABEL_ID[pred_main]}</strong> '
        f'({format_sub_label_id(pred_main)}) · IMT <strong class="cat-meta-em">{bmi_val:.2f} kg/m²</strong> · '
        f'{gender_lbl_t5}, Tinggi <strong class="cat-meta-em">{height} cm</strong>, Berat <strong class="cat-meta-em">{weight} kg</strong></p>'
        f'</div>', unsafe_allow_html=True)

    # Kalkulator BMR, TDEE, Defisit/Surplus Energi, dan Target Pembakaran Kalori Olahraga (Mifflin-St Jeor + Wishnofsky + METs)
    with st.container(border=True):
        st.markdown("#### Kalkulasi Kebutuhan Energi & Target Pembakaran Kalori (Berbasis BB & TB)")
        st.caption(
            "Dihitung menggunakan persamaan klinis Mifflin-St Jeor (BMR & TDEE), kesetaraan energi jaringan adiposa Wishnofsky "
            "(1 kg ≈ 7.700 kkal), dan rumus Metabolic Equivalent of Task (MET) sesuai berat badan Anda."
        )
        kc1, kc2 = st.columns([1, 1.6])
        with kc1:
            usia_thn = st.slider("Usia (Tahun)", 15, 80, 22, key="cal_age")
        with kc2:
            PAL_OPTIONS = [
                ("Sangat Jarang Olahraga / Sedenter (×1,20)", 1.20),
                ("Aktivitas Ringan: Olahraga 1–3 hari/minggu (×1,375)", 1.375),
                ("Aktivitas Sedang: Olahraga 3–5 hari/minggu (×1,55)", 1.55),
                ("Aktivitas Berat: Olahraga 6–7 hari/minggu (×1,725)", 1.725),
            ]
            pal_choice = st.selectbox(
                "Tingkat Aktivitas Fisik Harian",
                options=range(len(PAL_OPTIONS)),
                index=1,
                format_func=lambda i: PAL_OPTIONS[i][0],
                key="cal_pal",
            )
            pal_factor = PAL_OPTIONS[pal_choice][1]

        bmr_val = (10.0 * weight) + (6.25 * height) - (5.0 * usia_thn) + (5.0 if gender == "Male" else -161.0)
        tdee_val = bmr_val * pal_factor
        min_safe_intake = 1500 if gender == "Male" else 1200

        def _fmt_ribuan(n):
            return f"{int(round(n)):,}".replace(",", ".")

        if weight > ih:
            selisih_ideal_kg = weight - ih
            total_kalori_ideal = selisih_ideal_kg * 7700.0
            daily_deficit = 500 if pred_main <= 3 else 600
            burn_sport_day = 300 if pred_main <= 3 else 350
            diet_cut_day = daily_deficit - burn_sport_day
            intake_target = max(min_safe_intake, round(tdee_val - diet_cut_day))
            est_days = max(1, int(np.ceil(total_kalori_ideal / daily_deficit)))
            est_weeks = round(est_days / 7.0, 1)
            burn_tile_cls = "tile-rose"
            burn_lbl = "Target Kalori Dibakar (Olahraga)"
            burn_val_str = f"{_fmt_ribuan(burn_sport_day)} kkal/hari"
            burn_meta_str = f"+ Kurangi makan {_fmt_ribuan(diet_cut_day)} kkal = Defisit {_fmt_ribuan(daily_deficit)} kkal/hari"
            narasi_kalori = (
                f"Dengan tinggi <strong>{height} cm</strong> dan berat <strong>{weight} kg</strong>, berat badan berada "
                f"<strong>+{selisih_ideal_kg:.1f} kg</strong> di atas batas atas rentang ideal WHO ({il}–{ih} kg). "
                f"Untuk mencapai berat ideal <strong>{ih} kg</strong>, total kelebihan energi yang perlu dibakar adalah sekitar "
                f"<strong>{_fmt_ribuan(total_kalori_ideal)} kkal</strong>. Melalui kombinasi pembakaran olahraga rutin "
                f"<strong>{_fmt_ribuan(burn_sport_day)} kkal/hari</strong> dan pengaturan asupan makan <strong>{_fmt_ribuan(intake_target)} kkal/hari</strong> "
                f"(total defisit aman {_fmt_ribuan(daily_deficit)} kkal/hari), rentang berat ideal diperkirakan tercapai dalam "
                f"<strong>±{est_days} hari (±{str(est_weeks).replace('.', ',')} minggu)</strong>."
            )
        elif weight < il:
            selisih_ideal_kg = il - weight
            total_kalori_ideal = selisih_ideal_kg * 7700.0
            surplus_day = 400
            burn_sport_day = 180
            intake_target = round(tdee_val + surplus_day + burn_sport_day)
            est_days = max(1, int(np.ceil(total_kalori_ideal / surplus_day)))
            est_weeks = round(est_days / 7.0, 1)
            burn_tile_cls = "tile-amber"
            burn_lbl = "Batas Kalori Latihan Otot"
            burn_val_str = f"{_fmt_ribuan(burn_sport_day)} kkal/hari"
            burn_meta_str = f"Jaga surplus bersih +{_fmt_ribuan(surplus_day)} kkal/hari untuk naik massa otot"
            narasi_kalori = (
                f"Dengan tinggi <strong>{height} cm</strong> dan berat <strong>{weight} kg</strong>, berat badan masih kurang "
                f"<strong>{selisih_ideal_kg:.1f} kg</strong> dari batas bawah ideal WHO ({il}–{ih} kg), setara kebutuhan surplus total "
                f"<strong>{_fmt_ribuan(total_kalori_ideal)} kkal</strong>. Fokus utama bukan membakar lemak berlebih, melainkan "
                f"menjaga asupan padat gizi sebesar <strong>{_fmt_ribuan(intake_target)} kkal/hari</strong> (+{_fmt_ribuan(surplus_day)} kkal surplus bersih) "
                f"disertai latihan beban terukur (<strong>±{_fmt_ribuan(burn_sport_day)} kkal/hari</strong>) selama "
                f"<strong>±{est_days} hari (±{str(est_weeks).replace('.', ',')} minggu)</strong>."
            )
        else:
            total_kalori_ideal = 0.0
            burn_sport_day = 250
            intake_target = round(tdee_val)
            burn_tile_cls = "tile-emerald"
            burn_lbl = "Kalori Olahraga Pemeliharaan"
            burn_val_str = f"{_fmt_ribuan(burn_sport_day)} kkal/hari"
            burn_meta_str = "Setara anjuran kebugaran WHO 150 menit/minggu"
            narasi_kalori = (
                f"Dengan tinggi <strong>{height} cm</strong> dan berat <strong>{weight} kg</strong>, Anda sudah berada di dalam "
                f"rentang berat ideal WHO (<strong>{il}–{ih} kg</strong>). Pertahankan keseimbangan energi harian di sekitar "
                f"<strong>{_fmt_ribuan(intake_target)} kkal/hari</strong> dan bakar sekitar <strong>{_fmt_ribuan(burn_sport_day)} kkal/hari</strong> "
                f"melalui aktivitas fisik aerobik/kebugaran agar komposisi otot dan metabolisme tetap optimal."
            )

        # Tambahan informasi selisih kalori menuju Target Berat Badan pilihan user di Sidebar
        diff_target_user = weight - target_w
        if diff_target_user > 0:
            target_cal_note = f"Menuju target pribadi ({target_w} kg): perlu membakar total {_fmt_ribuan(diff_target_user * 7700)} kkal (-{diff_target_user} kg)."
        elif diff_target_user < 0:
            target_cal_note = f"Menuju target pribadi ({target_w} kg): perlu surplus total {_fmt_ribuan((-diff_target_user) * 7700)} kkal (+{-diff_target_user} kg)."
        else:
            target_cal_note = f"Berat saat ini sudah tepat berada pada target pribadi ({target_w} kg)."

        st.markdown(
            f'<div class="cal-grid">'
            f'<div class="metric-tile tile-blue">'
            f'<p class="tile-lbl">Energi Basal (BMR)</p>'
            f'<p class="tile-val">{_fmt_ribuan(bmr_val)} kkal/hari</p>'
            f'<p class="tile-meta">Kebutuhan energi saat istirahat penuh</p>'
            f'</div>'
            f'<div class="metric-tile tile-emerald">'
            f'<p class="tile-lbl">Kebutuhan Harian (TDEE)</p>'
            f'<p class="tile-val">{_fmt_ribuan(tdee_val)} kkal/hari</p>'
            f'<p class="tile-meta">Total energi harian (BMR × {str(pal_factor).replace(".", ",")})</p>'
            f'</div>'
            f'<div class="metric-tile {burn_tile_cls}">'
            f'<p class="tile-lbl">{burn_lbl}</p>'
            f'<p class="tile-val">{burn_val_str}</p>'
            f'<p class="tile-meta">{burn_meta_str}</p>'
            f'</div>'
            f'<div class="metric-tile tile-amber">'
            f'<p class="tile-lbl">Rekomendasi Asupan Makan</p>'
            f'<p class="tile-val">{_fmt_ribuan(intake_target)} kkal/hari</p>'
            f'<p class="tile-meta">Batas aman minimum: {_fmt_ribuan(min_safe_intake)} kkal/hari</p>'
            f'</div>'
            f'</div>'
            f'<div class="cal-summary-box">{narasi_kalori}<br><strong>Target Pribadi Sidebar:</strong> {target_cal_note}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(f"#### Estimasi Durasi Olahraga untuk Membakar {_fmt_ribuan(burn_sport_day)} kkal/hari (Pada Berat {weight} kg)")
        MET_ACTIVITIES = [
            ("Jalan Kaki Cepat (5,5 km/jam)", 4.3, "Aman untuk semua kategori & persendian"),
            ("Bersepeda Santai / Sepeda Statis", 5.5, "Rendah benturan lutut, cocok untuk Overweight/Obesitas"),
            ("Berenang Gaya Bebas Ringan", 6.0, "Sangat direkomendasikan untuk Obesitas (bebas beban sendi)"),
            ("Latihan Beban / Resistance Training", 5.0, "Optimal menjaga & membangun massa otot"),
            ("Senam Aerobik / Kardio Ritmik", 6.5, "Efektif meningkatkan detak jantung & pembakaran kalori"),
            ("Jogging / Lari Ringan (8 km/jam)", 7.5, "Pembakaran cepat (disarankan bila IMT < 30 kg/m²)"),
        ]
        met_rows = []
        for act_name, met_val, note_act in MET_ACTIVITIES:
            kcal_per_min = (met_val * 3.5 * weight) / 200.0
            kcal_30m = round(kcal_per_min * 30.0)
            durasi_target_mnt = max(5, int(round(burn_sport_day / max(kcal_per_min, 0.1))))
            met_rows.append({
                "Aktivitas Fisik / Olahraga": act_name,
                "Nilai MET": str(met_val).replace(".", ","),
                f"Terbakar per 30 Menit (BB {weight} kg)": f"{_fmt_ribuan(kcal_30m)} kkal",
                f"Durasi Mencapai Target ({_fmt_ribuan(burn_sport_day)} kkal)": f"±{durasi_target_mnt} menit/hari",
                "Catatan Kesesuaian": note_act,
            })
        render_table(pd.DataFrame(met_rows))

    st.markdown('<div class="spacer-3"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-lead">Centang kebiasaan harian pendukung rekomendasi di bawah ini:</p>', unsafe_allow_html=True)
    prog_ph=st.empty()
    done_count=0
    for i,(em,tit,desc) in enumerate(rec["items"]):
        st.markdown(
            '<div class="rec-item">'
            f'<div class="rec-num">{em}</div>'
            '<div>'
            f'<p class="rec-title">{tit}</p>'
            f'<p class="rec-desc">{desc}</p>'
            '</div>'
            '</div>', unsafe_allow_html=True)
        checked=st.checkbox("Sudah saya jalani", key=f"habit_{pred_main}_{i}")
        if checked: done_count+=1
        st.markdown('<div class="spacer-2"></div>', unsafe_allow_html=True)
    prog_ph.progress(done_count/len(rec["items"]), text=f"Progres kebiasaan: {done_count} dari {len(rec['items'])}")
    if done_count==len(rec["items"]):
        st.success("Semua kebiasaan sudah dijalani. Pertahankan konsistensi ini.")

    st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
    st.markdown("#### Panduan Berat Badan Berdasarkan Tinggi Badan Anda")
    td=pd.DataFrame({"Kategori (ID)":list(LABEL_ID.values()),"Kategori (EN)":list(LABEL_MAP.values()),"BMI Range":["< 16","16–18.5","18.5–25","25–30","30–40","> 40"],"Rentang Berat (kg)":[f"{round(lo*(height/100)**2)}–{round(hi*(height/100)**2)}" for lo,hi in [(10,16),(16,18.5),(18.5,25),(25,30),(30,40),(40,55)]]})
    render_table(td)

st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
st.markdown("#### Prediksi Batch (Upload File CSV)")
st.caption("Format kolom CSV: Gender (Male/Female atau Laki-laki/Perempuan), Height (cm), Weight (kg)")

if "use_demo_batch" not in st.session_state:
    st.session_state.use_demo_batch = False

bc1, bc2 = st.columns([2.2, 1])
with bc1:
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
with bc2:
    st.markdown('<div class="spacer-4"></div>', unsafe_allow_html=True)
    if not st.session_state.use_demo_batch:
        if st.button("Muat Contoh Data Uji Sidang (12 Sampel)", key="demo_batch_btn", use_container_width=True):
            st.session_state.use_demo_batch = True
            st.rerun()
    else:
        if st.button("Tutup Contoh Data Uji", key="reset_demo_btn", use_container_width=True):
            st.session_state.use_demo_batch = False
            st.rerun()

batch_source_df = None
if uploaded_file is not None:
    batch_source_df = pd.read_csv(uploaded_file, sep=None, engine="python")
elif st.session_state.use_demo_batch:
    try:
        batch_source_df = pd.read_csv("sample_batch_demo_sidang.csv")
    except Exception:
        batch_source_df = pd.DataFrame({
            "Gender": ["Male","Female","Male","Female","Male","Female","Male","Female","Male","Female","Male","Female"],
            "Height": [172, 165, 175, 160, 170, 162, 168, 158, 174, 160, 166, 155],
            "Weight": [44, 41, 53, 45, 65, 56, 78, 68, 106, 90, 122, 105],
        })
    st.caption("Menampilkan 12 sampel representatif (mencakup seluruh 6 kategori IMT standar WHO) dari sample_batch_demo_sidang.csv.")

if batch_source_df is not None and model_loaded:
    df_batch = batch_source_df.copy()
    col_map = {c.lower().strip(): c for c in df_batch.columns}
    if all(k in col_map for k in ["gender", "height", "weight"]):
        df_batch = df_batch.rename(columns={col_map["gender"]:"Gender", col_map["height"]:"Height", col_map["weight"]:"Weight"})
        df_batch["Height"] = pd.to_numeric(df_batch["Height"], errors="coerce")
        df_batch["Weight"] = pd.to_numeric(df_batch["Weight"], errors="coerce")
        n_awal = len(df_batch)
        df_batch = df_batch.dropna(subset=["Gender","Height","Weight"])
        df_batch = df_batch[(df_batch["Height"] > 50) & (df_batch["Height"] < 260) & (df_batch["Weight"] > 10) & (df_batch["Weight"] < 350)]
        df_batch["Gender"] = df_batch["Gender"].map(normalize_gender)
        st.write(f"Data valid ditemukan: **{len(df_batch)} baris**" + (f" ({n_awal-len(df_batch)} baris dilewati karena kosong/tidak wajar)" if n_awal > len(df_batch) else ""))
        if df_batch.empty:
            st.error("Tidak ada baris valid untuk diprediksi.")
            st.stop()
        with st.spinner("Memproses prediksi batch..."):
            batch_fv=[build_feature_vector(str(r.Gender).strip(),float(r.Height),float(r.Weight)) for _,r in df_batch.iterrows()]
            batch_sc=scaler.transform(batch_fv)
            df_batch["BMI"]=(df_batch["Weight"]/(df_batch["Height"]/100)**2).round(2)
            df_batch["Prediksi_Index"]=model.predict(batch_sc)
            df_batch["Prediksi_EN"]=df_batch["Prediksi_Index"].map(LABEL_MAP)
            df_batch["Prediksi_ID"]=df_batch["Prediksi_Index"].map(LABEL_ID)
        render_table(df_batch.head(30))
        st.success(f"Prediksi selesai untuk {len(df_batch)} data.")

        # CSV ramah Excel Indonesia (header sep=, + UTF-8 BOM agar langsung terpisah rapi per kolom)
        csv_excel_bytes = ("sep=,\r\n" + df_batch.to_csv(index=False)).encode("utf-8-sig")

        def _build_excel_bytes(df_out):
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter
            wb = Workbook()
            ws = wb.active
            ws.title = "Hasil Prediksi BMI"
            hdr_fill = PatternFill("solid", fgColor="1E293B")
            hdr_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
            body_font = Font(name="Calibri", size=11, color="111827")
            zebra_fill = PatternFill("solid", fgColor="F8FAFC")
            thin = Side(style="thin", color="CBD5E1")
            border = Border(left=thin, right=thin, top=thin, bottom=thin)
            cols = list(df_out.columns)
            ws.append(cols)
            for c_idx in range(1, len(cols) + 1):
                cell = ws.cell(row=1, column=c_idx)
                cell.fill = hdr_fill
                cell.font = hdr_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = border
            ws.row_dimensions[1].height = 24
            for r_idx, row in enumerate(df_out.itertuples(index=False), start=2):
                ws.append(list(row))
                for c_idx, val in enumerate(row, start=1):
                    cell = ws.cell(row=r_idx, column=c_idx)
                    cell.font = body_font
                    cell.border = border
                    if r_idx % 2 == 1:
                        cell.fill = zebra_fill
                    col_name = cols[c_idx - 1]
                    if col_name == "BMI":
                        cell.number_format = "0.00"
                        cell.alignment = Alignment(horizontal="right", vertical="center")
                    elif col_name in ("Height", "Weight", "Prediksi_Index"):
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                    else:
                        cell.alignment = Alignment(horizontal="left", vertical="center")
            for c_idx, col_name in enumerate(cols, start=1):
                max_len = max([len(str(col_name))] + [len(str(v)) for v in df_out[col_name]])
                ws.column_dimensions[get_column_letter(c_idx)].width = max(max_len + 4, 12)
            ws.freeze_panes = "A2"
            buf = io.BytesIO()
            wb.save(buf)
            return buf.getvalue()

        try:
            xlsx_bytes = _build_excel_bytes(df_batch)
            st.download_button(
                "Download Hasil Prediksi (Excel)",
                xlsx_bytes,
                "hasil_prediksi_bmi_skripsi.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        except Exception:
            st.download_button(
                "Download Hasil Prediksi (CSV)",
                csv_excel_bytes,
                "hasil_prediksi_bmi_skripsi.csv",
                "text/csv",
                type="primary",
                use_container_width=True,
            )
    else:
        st.error("Format kolom CSV tidak sesuai. Pastikan memiliki kolom: Gender, Height, Weight.")
elif batch_source_df is not None:
    st.warning("Model belum dimuat.")
