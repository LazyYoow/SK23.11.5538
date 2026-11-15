# BMI Analyzer: Klasifikasi Kategori Indeks Massa Tubuh Menggunakan Simulasi
# Image Processing dan Machine Learning Berbasis Data Antropometri
# Disusun oleh: Satrio Raja Nabil Rizky (23.11.5538)
# Dosen Pembimbing: Mulia Sulistiyono, S.Kom., M.Kom.
import streamlit as st
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
    page_title="BMI Analyzer | Skripsi S1 Informatika",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []
if "aff_idx" not in st.session_state:
    st.session_state.aff_idx = 0
if "last_idx" not in st.session_state:
    st.session_state.last_idx = None
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

IS_LIGHT = st.session_state.theme_mode == "Light"

# Token warna permukaan dan tipografi untuk Dark Mode & White Mode (semua pasangan >= 4.5:1 WCAG AA)
if IS_LIGHT:
    BG_APP      = "#FAF9F5"
    BG_SIDE     = "#F3F1EC"
    BG_CARD     = "#FFFFFF"
    BG_RAISE    = "#F1EFEA"
    BORDER      = "#E2E0D8"
    BORDER_2    = "#C8C4B8"
    TXT_MAIN    = "#14141B"
    TXT_BODY    = "#2D2C38"
    TXT_SUB     = "#5C5969"
    ACCENT      = "#4F46E5"
    ACCENT_HI   = "#3730A3"
    ACCENT_DEEP = "#4F46E5"
    ACCENT_2    = "#0F766E"
    PIE_TXT     = "#FFFFFF"
    SIL_REF     = "#14141B"
    SH_A1, SH_A2 = 0.14, 0.10
    PILL_BG     = "rgba(20,20,27,0.04)"
    TZ_BORDER   = "rgba(15,118,110,0.35)"
    CAT_COLORS  = ["#1D4ED8","#15803D","#0F766E","#B45309","#B91C1C","#6D28D9"]
else:
    BG_APP      = "#09090E"
    BG_SIDE     = "#0E0E15"
    BG_CARD     = "#12121A"
    BG_RAISE    = "#1A1A24"
    BORDER      = "#23232F"
    BORDER_2    = "#34343F"
    TXT_MAIN    = "#F5F3EE"
    TXT_BODY    = "#D6D3CC"
    TXT_SUB     = "#8E8B99"
    ACCENT      = "#818CF8"
    ACCENT_HI   = "#A5B4FC"
    ACCENT_DEEP = "#4F46E5"
    ACCENT_2    = "#7DD3C0"
    PIE_TXT     = "#09090E"
    SIL_REF     = "#F5F3EE"
    SH_A1, SH_A2 = 0.45, 0.35
    PILL_BG     = "rgba(255,255,255,0.03)"
    TZ_BORDER   = "rgba(125,211,192,0.35)"
    CAT_COLORS  = ["#60A5FA","#4ADE80","#7DD3C0","#FBBF24","#F87171","#A78BFA"]

plt.rcParams.update({
    "figure.facecolor": BG_CARD,
    "axes.facecolor": BG_CARD,
    "savefig.facecolor": BG_CARD,
    "text.color": TXT_MAIN,
    "axes.labelcolor": TXT_BODY,
    "axes.edgecolor": BORDER_2,
    "xtick.color": TXT_BODY,
    "ytick.color": TXT_BODY,
    "font.family": "sans-serif",
})

def style_ax(ax, grid_axis=None):
    ax.set_facecolor(BG_CARD)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(BORDER_2)
    ax.tick_params(colors=TXT_BODY, labelsize=8)
    if grid_axis:
        ax.set_axisbelow(True)
        ax.grid(axis=grid_axis, color=BORDER, linestyle="-", linewidth=0.6, alpha=0.9)
    return ax

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif !important;
}}
.stApp {{
    background: {BG_APP} !important;
    color: {TXT_MAIN} !important;
}}
header[data-testid="stHeader"] {{ background: transparent !important; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3.5rem; max-width: 1280px; }}

a[href], button, summary, select, label, [role="button"], [role="tab"] {{
    cursor: pointer;
}}
button:disabled, [aria-disabled="true"] {{
    cursor: not-allowed;
}}
button:focus-visible, a:focus-visible, summary:focus-visible, [role="tab"]:focus-visible {{
    outline: 2px solid {ACCENT_HI} !important;
    outline-offset: 2px !important;
}}

section[data-testid="stSidebar"] {{
    background: {BG_SIDE} !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.75rem; }}
section[data-testid="stSidebar"] h3 {{
    font-family: 'Playfair Display', serif !important;
    font-weight: 600 !important; font-size: 1.35rem !important;
}}

h1, h2, h3, h4 {{ color: {TXT_MAIN} !important; letter-spacing: -0.01em; }}
h4 {{ font-weight: 600 !important; font-size: 1.05rem !important; color: {TXT_MAIN} !important; }}
.stApp p, .stApp li, .stApp label, .stApp span[data-baseweb] {{ color: {TXT_BODY}; }}
.stApp .stCaption, .stApp small, div[data-testid="stCaptionContainer"] p {{ color: {TXT_SUB} !important; }}
div[data-testid="stWidgetLabel"] p, div[data-testid="stRadio"] label p, div[data-testid="stCheckbox"] label p {{
    color: {TXT_BODY} !important;
}}
div[data-baseweb="select"] > div, div[data-testid="stFileUploader"] section {{
    background: {BG_CARD} !important;
    border-color: {BORDER} !important;
    color: {TXT_MAIN} !important;
}}

.page-header {{
    position: relative; padding: 0.25rem 0 1.5rem; margin-bottom: 1.5rem;
    border-bottom: 1px solid {BORDER};
}}
.page-meta {{
    font-size: 0.82rem; font-weight: 500; color: {ACCENT_HI};
    letter-spacing: .04em; margin: 0 0 8px;
}}
.page-title {{
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2rem, 4vw, 3rem); font-weight: 600; line-height: 1.12;
    margin: 0 0 10px; color: {TXT_MAIN};
}}
.page-sub {{ font-size: 0.98rem; color: {TXT_BODY}; margin: 0; font-weight: 400; }}
.page-author {{ font-size: 0.85rem; color: {TXT_SUB}; margin: 10px 0 0; font-weight: 400; }}

div[data-baseweb="tab-list"] {{ gap: 28px; border-bottom: 1px solid {BORDER}; flex-wrap: wrap; }}
button[data-baseweb="tab"] {{
    color: {TXT_SUB} !important; font-weight: 400; font-size: 13.5px; letter-spacing: .01em;
    padding: 12px 4px 14px !important; min-height: 44px; background: transparent !important;
    transition: color .15s ease;
}}
button[data-baseweb="tab"]:hover {{ color: {TXT_MAIN} !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {TXT_MAIN} !important; font-weight: 600; }}
div[data-baseweb="tab-highlight"] {{
    background: {ACCENT} !important; height: 2px !important;
}}

div[data-testid="stMetric"], div[data-testid="metric-container"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER}; border-radius: 12px;
    padding: 1.1rem 1.3rem 1.2rem;
}}
div[data-testid="stMetricLabel"] p {{
    color: {TXT_SUB} !important; font-size: 11px !important; font-weight: 500 !important;
    text-transform: uppercase; letter-spacing: .08em;
}}
div[data-testid="stMetricValue"] {{
    color: {TXT_MAIN} !important; font-weight: 500;
    font-family: 'Playfair Display', serif !important; font-size: 2rem !important;
    overflow-wrap: break-word;
}}

div[data-testid="stVerticalBlockBorderWrapper"] > div > div > div[data-testid="stVerticalBlock"] {{ gap: 0.8rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 12px !important; border: 1px solid {BORDER} !important;
    background: {BG_CARD};
    padding: 0.4rem 0.6rem;
}}

div[data-testid="stExpander"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER} !important; border-radius: 12px !important;
}}
div[data-testid="stExpander"] summary {{ min-height: 44px; display: flex; align-items: center; }}
div[data-testid="stExpander"] summary p {{ color: {TXT_MAIN} !important; font-weight: 500; }}

/* Tombol utama: kontras #FFFFFF di atas #4F46E5 = 6.29:1 (lulus WCAG AA di kedua mode) */
.stButton > button, .stDownloadButton > button {{
    background: {ACCENT_DEEP};
    color: #FFFFFF !important; border: 1px solid #6366F1; border-radius: 10px;
    font-weight: 600; font-size: 13.5px; letter-spacing: .01em;
    padding: 0.6rem 1rem; min-height: 44px; width: 100%;
    transition: background-color .15s ease, border-color .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background: #4338CA; border-color: {ACCENT}; color: #FFFFFF !important;
}}
.stButton > button:active, .stDownloadButton > button:active {{
    background: #3730A3;
}}
.st-key-clear_hist > div > button {{
    background: transparent !important; color: {TXT_MAIN} !important;
    border: 1px solid {BORDER_2} !important;
}}
.st-key-clear_hist > div > button:hover {{
    background: {BG_RAISE} !important; border-color: {ACCENT} !important; color: {ACCENT_HI} !important;
}}

.bmi-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 18px;
}}
.card-label {{
    font-size: 11px; font-weight: 500; color: {TXT_SUB};
    text-transform: uppercase; letter-spacing: .08em; margin: 0 0 8px;
}}
.card-value {{
    font-family: 'Playfair Display', serif; font-size: 2.1rem; font-weight: 500;
    color: {TXT_MAIN}; margin: 0; line-height: 1.1;
}}
.card-sub {{ font-size: 12.5px; color: {TXT_SUB}; margin: 6px 0 0; }}

.pill {{
    display: inline-block; padding: 4px 10px; border-radius: 6px;
    font-size: 11.5px; font-weight: 600; letter-spacing: .02em;
    border: 1px solid; background: {PILL_BG};
}}

.scale-wrap {{ margin: 0.25rem 0 0.5rem; }}
.scale-label {{
    font-size: 11px; color: {TXT_SUB}; font-weight: 500;
    text-transform: uppercase; letter-spacing: .08em; margin: 0 0 14px;
}}
.scale-track {{
    position: relative; height: 6px; border-radius: 3px;
    background: linear-gradient(90deg, #60A5FA 0%, #4ADE80 28%, #FBBF24 52%, #F87171 76%, #A78BFA 100%);
}}
.scale-marker {{
    position: absolute; top: 50%; transform: translate(-50%, -50%);
    width: 18px; height: 18px; border-radius: 50%;
    background: {BG_APP}; border: 2px solid {TXT_MAIN};
    box-shadow: 0 0 0 4px rgba(79,70,229,0.24);
    transition: left .3s ease;
}}
.scale-ticks {{
    display: flex; justify-content: space-between; font-size: 11px;
    color: {TXT_SUB}; margin-top: 12px;
}}

.pred-box {{
    border: 1px solid {BORDER}; border-radius: 12px; padding: 18px 20px;
    background: {BG_CARD};
}}
.pred-title {{
    font-size: 11px; font-weight: 500; color: {TXT_SUB};
    text-transform: uppercase; letter-spacing: .08em; margin: 0 0 12px;
}}
.pred-meta {{ font-size: 12.5px; color: {TXT_SUB}; margin: 12px 0 0; line-height: 1.6; }}

.prob-row {{ margin-bottom: 18px; }}
.prob-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 14px; margin-bottom: 9px; color: {TXT_MAIN};
}}
.prob-sub {{ color: {TXT_SUB}; font-size: 12px; margin-left: 8px; font-weight: 400; }}
.prob-val {{ font-weight: 500; font-family: 'Playfair Display', serif; font-size: 1.05rem; }}
.prob-track {{ height: 6px; background: {BG_RAISE}; border-radius: 4px; overflow: hidden; }}
.prob-fill {{ height: 100%; border-radius: 4px; transition: width .35s ease; }}

.cat-banner {{
    background: {BG_CARD};
    border: 1px solid {BORDER}; border-radius: 12px;
    padding: 1.4rem 1.6rem; margin-bottom: 1.4rem;
}}
.cat-banner-title {{ font-family: 'Playfair Display', serif; font-size: 1.55rem; font-weight: 500; margin: 0 0 8px; }}
.cat-banner-meta {{ font-size: 13.5px; color: {TXT_SUB}; margin: 0; }}
.rec-item {{
    display: flex; gap: 16px; align-items: flex-start;
    background: {BG_CARD};
    border: 1px solid {BORDER}; border-radius: 10px;
    padding: 16px 18px; margin-bottom: 4px;
}}
.rec-num {{
    flex: 0 0 36px; height: 36px; border-radius: 8px; background: {BG_RAISE};
    border: 1px solid {BORDER_2}; color: {ACCENT_HI};
    font-family: 'Playfair Display', serif; font-weight: 600; font-size: 0.95rem;
    display: flex; align-items: center; justify-content: center;
}}
.rec-title {{ font-weight: 500; color: {TXT_MAIN}; margin: 0 0 5px; font-size: 0.97rem; }}
.rec-desc {{ color: {TXT_BODY}; margin: 0; font-size: 0.89rem; line-height: 1.6; font-weight: 400; }}

.aff-text {{
    font-family: 'Playfair Display', serif; font-size: 1.25rem; font-style: italic;
    color: {TXT_MAIN}; line-height: 1.6; margin: 6px 0 0;
}}
.aff-note {{ font-size: 12px; color: {TXT_SUB}; margin-top: 10px; }}

.clock {{
    display: flex; align-items: center; flex-wrap: wrap; gap: 10px;
    background: {BG_CARD};
    border: 1px solid {BORDER}; border-radius: 10px; padding: 10px 14px;
    font-size: 13px; color: {TXT_BODY}; margin-bottom: 12px;
}}
.clock b {{
    font-family: 'Playfair Display', serif; font-size: 1.15rem; font-weight: 600;
    color: {ACCENT_HI}; letter-spacing: .02em;
}}
.clock-dot {{
    width: 8px; height: 8px; border-radius: 50%; background: #16A34A;
}}
.clock-tz {{
    margin-left: auto; font-size: 11px; font-weight: 600; letter-spacing: .08em;
    color: {ACCENT_2}; border: 1px solid {TZ_BORDER}; border-radius: 6px; padding: 3px 9px;
}}

div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; }}

.gold-rule {{
    height: 1px; border: 0; margin: 1.5rem 0;
    background: {BORDER};
}}

@media (max-width: 768px) {{
    .block-container {{ padding-top: 1.25rem; padding-bottom: 2rem; }}
    div[data-baseweb="tab-list"] {{ gap: 14px; }}
    .clock-tz {{ margin-left: 0; }}
}}
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }}
}}
</style>
""", unsafe_allow_html=True)

LABEL_MAP   = {0:"Extremely Weak",1:"Weak",2:"Normal",3:"Overweight",4:"Obesity",5:"Extreme Obesity"}
LABEL_ID    = {0:"Sangat Kurus",  1:"Kurus",  2:"Normal", 3:"Overweight",4:"Obesitas",5:"Sangat Obesitas"}
CAT_BG      = ["#1e3a5f","#14532d","#065f46","#78350f","#7f1d1d","#3b0764"]

RECOMMENDATIONS = {
    0:{"title":"Perlu Perhatian Ekstra","color":CAT_COLORS[0],"items":[
        ("01","Perbanyak Kalori Bergizi","Makanan padat nutrisi: kacang, alpukat, nasi merah, protein hewani berkualitas."),
        ("02","Latihan Pembentuk Otot","Latihan beban 3×/minggu. Target massa otot bukan sekadar angka timbangan."),
        ("03","Konsultasi Dokter Gizi","BMI sangat rendah bisa menandakan kondisi medis tersembunyi."),
        ("04","Tidur 7–9 Jam Cukup","Tidur berkualitas mendukung hormon pertumbuhan dan nafsu makan.")]},
    1:{"title":"Tingkatkan Nutrisi","color":CAT_COLORS[1],"items":[
        ("01","Protein & Karbohidrat Kompleks","Target 1.5–2g protein/kg BB. Makan 5–6 kali porsi kecil per hari."),
        ("02","Olahraga Ringan Rutin","30 menit jalan cepat atau bersepeda per hari untuk membangun metabolisme."),
        ("03","Kelola Stres","Stres menekan nafsu makan. Latihan pernapasan atau relaksasi 10 menit/hari membantu.")]},
    2:{"title":"Pertahankan Pola Hidup Sehat","color":CAT_COLORS[2],"items":[
        ("01","Jaga Konsistensi Pola Makan","BMI berada di rentang ideal. Pertahankan komposisi gizi seimbang dan jadwal makan teratur."),
        ("02","Olahraga Rutin & Gizi Seimbang","150 menit aktivitas sedang/minggu. Perbanyak sayur, buah, dan protein."),
        ("03","Cek Rutin 3–6 Bulan","Monitor BMI secara periodik agar perubahan kecil lebih mudah ditangani sejak dini."),
        ("04","Hidrasi Optimal","2–2.5 liter air/hari. Batasi minuman berpemanis tambahan.")]},
    3:{"title":"Kendalikan Berat Badan","color":CAT_COLORS[3],"items":[
        ("01","Tingkatkan Aktivitas Harian","Targetkan 8.000–10.000 langkah/hari atau 45 menit latihan kardio teratur."),
        ("02","Kurangi Gula & Gorengan","Utamakan bahan pangan utuh dan periksa label nutrisi sebelum membeli."),
        ("03","Defisit Kalori Terukur","Defisit 300–500 kkal/hari mendukung penurunan 0.3–0.5 kg/minggu secara bertahap."),
        ("04","Perbaiki Kualitas Tidur","Kurang tidur meningkatkan hormon ghrelin pemicu lapar. Targetkan 7–9 jam/malam.")]},
    4:{"title":"Prioritaskan Kesehatan","color":CAT_COLORS[4],"items":[
        ("01","Konsultasi Dokter atau Ahli Gizi","Evaluasi profil metabolik untuk mencegah risiko diabetes, hipertensi, dan penyakit kardiovaskular."),
        ("02","Olahraga Rendah Benturan","Renang, jalan kaki, atau sepeda statis lebih aman bagi persendian. Mulai 20 menit/hari."),
        ("03","Pola Makan Terstruktur","Hindari diet ekstrem. Terapkan 3 kali makan utama dengan porsi serat dan protein terukur."),
        ("04","Dukungan Lingkungan","Libatkan keluarga atau komunitas sehat agar perubahan kebiasaan bertahan jangka panjang.")]},
    5:{"title":"Pendampingan Medis Diperlukan","color":CAT_COLORS[5],"items":[
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

def build_feature_vector(gender, height, weight):
    gender=normalize_gender(gender)
    h_m=height/100; bmi=weight/(h_m**2)
    img_feats=extract_image_features(generate_body_silhouette(height,weight,gender))
    g_num=1 if gender=="Male" else 0
    h_cat=0 if height<=155 else 1 if height<=170 else 2 if height<=185 else 3
    w_cat=0 if weight<=60  else 1 if weight<=80  else 2 if weight<=100  else 3
    return [g_num,height,weight,h_m,bmi,bmi**2,weight/height,weight**2,height**2,h_cat,w_cat]+img_feats

def bmi_index(bmi):
    if bmi<16: return 0
    if bmi<18.5: return 1
    if bmi<25: return 2
    if bmi<30: return 3
    if bmi<40: return 4
    return 5

def format_sub_label(idx):
    return "Ideal (Normal Weight)" if idx == 2 else LABEL_MAP[idx]

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
    c = CAT_COLORS[idx]

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
        m_ref = _rasterize(body_polygons(gender, 22.0))
        cnts, _ = cv2.findContours((m_ref > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cn in cnts:
            p = cn[:, 0, :].astype(float)
            px = (p[:, 0]/SIL_RES + SIL_X0)*H; py = (p[:, 1]/SIL_RES + SIL_Y0)*H
            ax.plot(np.r_[px, px[0]], np.r_[py, py[0]], color=SIL_REF, linewidth=0.7,
                    linestyle=(0, (3, 2)), alpha=0.55, zorder=4)

    # Garis dimensi tinggi
    xd = 0.33*H
    ax.plot([xd, xd], [0.0, 1.0*H], color=TXT_SUB, linewidth=0.8, zorder=2)
    for yv in (0.0, 1.0*H):
        ax.plot([xd-0.015, xd+0.015], [yv, yv], color=TXT_SUB, linewidth=0.8)
    ax.text(xd+0.035, 0.5*H, f"{height} cm", color=TXT_SUB, fontsize=8, va="center", ha="left", rotation=90)

    ax.text(0, 1.0*H+0.09, f"{LABEL_ID[idx]}  ·  BMI {bmi:.1f}", ha="center", fontsize=9,
            fontweight="bold", color=TXT_BODY)
    if abs(bmi - 22.0) > 2.0:
        ax.text(0, 1.0*H+0.15, "- - -  siluet BMI ideal (22)", ha="center", fontsize=7, color=TXT_SUB)

    ax.set_xlim(-0.46, 0.50); ax.set_ylim(1.36, -0.06)
    ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig

def _clock_body():
    now = datetime.now(WIB)
    st.markdown(
        f'<div class="clock"><span class="clock-dot"></span>'
        f'<span>{fmt_hari_tanggal(now)}</span>'
        f'<b>{now:%H:%M:%S}</b>'
        f'<span class="clock-tz">GMT+7 · WIB</span></div>',
        unsafe_allow_html=True)

if hasattr(st, "fragment"):
    live_clock = st.fragment(run_every="1s")(_clock_body)
else:
    live_clock = _clock_body   # Fallback statis untuk Streamlit < 1.37.

with st.sidebar:
    st.radio(
        "Mode Tampilan",
        ["Dark", "Light"],
        horizontal=True,
        key="theme_mode",
        format_func=lambda x: "Dark Mode" if x == "Dark" else "White Mode",
    )
    st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
    st.markdown("### Input Data Antropometri")
    st.caption("Hasil diperbarui otomatis saat slider digeser.")
    st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
    gender=st.radio("Jenis Kelamin",["Male","Female"],format_func=lambda x:"Laki-laki" if x=="Male" else "Perempuan")
    height=st.slider("Tinggi Badan (cm)",140,200,170)
    weight=st.slider("Berat Badan (kg)",35,165,70)
    target_w=st.slider("Target Berat Badan (kg)",35,165,65)

    bmi_val=weight/(height/100)**2
    bi=bmi_index(bmi_val)
    c2=CAT_COLORS[bi]

    if st.session_state.last_idx is not None and st.session_state.last_idx != bi:
        st.toast(f"Kategori berubah menjadi {LABEL_ID[bi]}")
    st.session_state.last_idx = bi

    st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
    st.markdown(f"""
<div class="bmi-card">
<p class="card-label">BMI / IMT</p>
<p class="card-value">{bmi_val:.1f}</p>
<div style="margin-top:12px">
<span class="pill" style="color:{c2};border-color:{c2}55">{LABEL_ID[bi]}</span>
</div>
<p class="card-sub">{format_sub_label(bi)}</p>
</div>""", unsafe_allow_html=True)
    il=round(18.5*(height/100)**2); ih=round(24.9*(height/100)**2)
    st.markdown(f"<p class='card-sub' style='margin-top:14px'><span style='color:{TXT_SUB}'>Berat Ideal</span>&nbsp;&nbsp;<b style='color:{TXT_MAIN};font-weight:500'>{il}–{ih} kg</b></p>", unsafe_allow_html=True)
    st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)

    if st.button("Simpan ke Riwayat", use_container_width=True):
        pred_lbl="-"
        if model_loaded:
            fv_s=build_feature_vector(gender,height,weight)
            pred_lbl=LABEL_ID[int(model.predict(scaler.transform([fv_s]))[0])]
        st.session_state.riwayat.append({
            "Waktu (WIB)": fmt_waktu_riwayat(datetime.now(WIB)),
            "Gender": "Laki-laki" if gender=="Male" else "Perempuan",
            "Tinggi (cm)": height,
            "Berat (kg)": weight,
            "BMI": round(bmi_val,2),
            "Kategori (IMT)": LABEL_ID[bi],
            "Prediksi ML": pred_lbl,
        })
        st.toast("Pengukuran tersimpan ke riwayat")

st.markdown("""
<div class="page-header">
<p class="page-meta">Skripsi S1 Informatika · Universitas AMIKOM Yogyakarta</p>
<h1 class="page-title">BMI Analyzer</h1>
<p class="page-sub">Klasifikasi kategori IMT melalui simulasi citra biner dan machine learning</p>
<p class="page-author">Satrio Raja Nabil Rizky &nbsp;·&nbsp; 23.11.5538</p>
</div>
""", unsafe_allow_html=True)

t1,t2,t3,t4,t5=st.tabs(["Kalkulator BMI","Analisis Data (n=3,700)","Distribusi Probabilitas","Evaluasi 6 Model ML","Rekomendasi"])

with t1:
    ca,cb=st.columns([1,1.8])
    with ca:
        with st.container(border=True):
            st.markdown("#### Siluet Tubuh")
            fig_s=draw_human_silhouette(height,weight,gender,bi)
            st.pyplot(fig_s,use_container_width=True); plt.close()
            st.caption("Ilustrasi 3D tampak depan (proporsi anatomi ±8 kepala, distribusi lemak mengikuti BMI). Garis putus-putus = pembanding BMI ideal 22.")

        with st.expander("Citra Biner OpenCV & Fitur Morfologi", expanded=True):
            img_cv = generate_body_silhouette(height, weight, gender)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
            cnts, _ = cv2.findContours(img_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_rgb, cnts, -1, (248, 140, 129), 2)   # indigo dalam format BGR
            st.image(img_rgb, caption="Citra siluet biner OpenCV (128×128 px) dengan kontur", use_container_width=True)
            f_img = extract_image_features(img_cv)
            st.json({
                "pixel_area": int(f_img[0]),
                "aspect_ratio": f_img[1],
                "body_density": f_img[2],
                "contour_perimeter": f_img[3],
                "compactness": f_img[4]
            })

    with cb:
        st.markdown("#### Hasil Analisis IMT")
        m1,m2,m3=st.columns(3)
        m1.metric("BMI / IMT",f"{bmi_val:.1f}")
        m2.metric("Kategori (ID)",LABEL_ID[bi])
        if weight < il:
            diff = il - weight
            m3.metric("Gap Ideal", f"-{diff:.0f} kg", delta=f"Kurang {diff:.0f} kg", delta_color="inverse")
        elif weight > ih:
            diff = weight - ih
            m3.metric("Gap Ideal", f"+{diff:.0f} kg", delta=f"Lebih {diff:.0f} kg", delta_color="inverse")
        else:
            m3.metric("Gap Ideal", "0 kg", delta="Sudah Ideal ✓", delta_color="normal")

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        bp=min(max((bmi_val-10)/50*100,0),100)
        st.markdown(
            '<div class="scale-wrap">'
            '<p class="scale-label">Skala IMT</p>'
            '<div class="scale-track">'
            f'<div class="scale-marker" style="left:{bp}%"></div>'
            '</div>'
            '<div class="scale-ticks"><span>Kurus</span><span>Normal</span><span>Gemuk</span><span>Obesitas</span><span>Parah</span></div>'
            '</div>', unsafe_allow_html=True)

        st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)

        if model_loaded:
            fv=build_feature_vector(gender,height,weight)
            fv_sc=scaler.transform([fv])
            pred=model.predict(fv_sc)[0]
            proba=model.predict_proba(fv_sc)[0]
            conf=max(proba)*100
            c3=CAT_COLORS[pred]
            st.markdown(
                f'<div class="pred-box" style="border-left:2px solid {c3}">'
                '<p class="pred-title">Prediksi Machine Learning</p>'
                f'<span class="pill" style="color:{c3};border-color:{c3}55">{LABEL_ID[pred]}</span>'
                f'<span style="margin-left:10px;color:{TXT_MAIN};font-family:Playfair Display,serif;font-size:1.05rem">{format_sub_label(pred)}</span>'
                f'<p class="pred-meta">Confidence {conf:.1f}% &nbsp;·&nbsp; Model Random Forest (Acc 99.73% · split 80:20 · n=3,700)</p>'
                '</div>', unsafe_allow_html=True)
        else:
            st.warning("Pastikan model_bmi.pkl dan scaler_bmi.pkl tersedia di repositori.")

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        info_df=pd.DataFrame({
            "Atribut / Fitur":["Tinggi Badan","Berat Badan","BMI / IMT","Kategori (EN)","Kategori (ID)","Rentang Berat Ideal","Pixel Area (Citra)","Aspect Ratio (Citra)","Body Density (Citra)","Contour Perimeter","Compactness"],
            "Nilai":[f"{height} cm",f"{weight} kg",f"{bmi_val:.2f} kg/m²",LABEL_MAP[bi],LABEL_ID[bi],f"{il}–{ih} kg",f"{int(f_img[0])} px²",f"{f_img[1]:.4f}",f"{f_img[2]:.4f}",f"{f_img[3]:.4f}",f"{f_img[4]:.4f}"]
        })
        st.dataframe(info_df,hide_index=True,use_container_width=True)
        st.caption("Hasil ini untuk edukasi dan simulasi akademik, bukan diagnosis medis.")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    cc1,cc2=st.columns(2)
    with cc1:
        with st.container(border=True):
            st.markdown("#### Simulasi Target Berat")
            bmi_t=target_w/((height/100)**2)
            cat_t=bmi_index(bmi_t)
            d_t=weight-target_w
            if d_t>0: msg=f"Perlu turun {d_t} kg untuk mencapai target"
            elif d_t<0: msg=f"Perlu naik {-d_t} kg untuk mencapai target"
            else: msg="Berat saat ini sudah sama dengan target"
            t_a,t_b=st.columns(2)
            t_a.metric("BMI pada target",f"{bmi_t:.1f}")
            t_b.metric("Kategori pada target",LABEL_ID[cat_t])
            st.caption(msg)

    with cc2:
        with st.container(border=True):
            st.markdown("#### Pesan Afirmasi")
            if st.button("Pesan lain", key="aff_btn"):
                st.session_state.aff_idx += 1
            msgs=AFFIRMATIONS[bi]
            idx_msg=st.session_state.aff_idx % len(msgs)
            st.markdown(f'<p class="aff-text">"{msgs[idx_msg]}"</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="aff-note">Pesan {idx_msg+1} dari {len(msgs)} · kategori {LABEL_ID[bi]}</p>', unsafe_allow_html=True)

    st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
    st.markdown("#### Riwayat Pengukuran (sesi ini)")
    live_clock()
    if st.session_state.riwayat:
        dfr=pd.DataFrame(st.session_state.riwayat)
        st.dataframe(dfr,hide_index=True,use_container_width=True)
        if len(dfr)>=2:
            fig,ax=plt.subplots(figsize=(8,3.2))
            style_ax(ax,grid_axis="y")
            xs=list(range(1,len(dfr)+1))
            ax.plot(xs,dfr["BMI"],color=ACCENT,linewidth=1.6,zorder=2)
            ax.scatter(xs,dfr["BMI"],color=ACCENT_HI,s=38,zorder=3,edgecolors=BG_CARD)
            ax.axhline(18.5,color=ACCENT_2,linestyle=":",linewidth=0.9)
            ax.axhline(25,color=ACCENT_2,linestyle=":",linewidth=0.9)
            ax.set_xlabel("Urutan pengukuran"); ax.set_ylabel("BMI")
            ax.set_xticks(xs)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        if st.button("Hapus riwayat", key="clear_hist"):
            st.session_state.riwayat=[]
            st.rerun()
    else:
        st.caption("Belum ada data. Klik \"Simpan ke Riwayat\" di sidebar untuk mulai mencatat.")

with t2:
    try:
        df_data=pd.read_csv("bmi.csv")
        label_map_en={0:"Extremely Weak",1:"Weak",2:"Normal",3:"Overweight",4:"Obesity",5:"Extreme Obesity"}
        df_data["Label"]=df_data["Index"].map(label_map_en)
        df_data["BMI"]=df_data["Weight"]/(df_data["Height"]/100)**2

        s1,s2,s3,s4=st.columns(4)
        s1.metric("Total Dataset Gabungan",f"{len(df_data):,}")
        s2.metric("Fitur (11 Tabular + 5 Citra)","16")
        s3.metric("Kategori IMT","6 Kelas")
        s4.metric("Best Accuracy (RF · 75:25)","99.78%")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
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

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
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
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        with cb2:
            with st.container(border=True):
                st.markdown("#### Distribusi Gender")
                fig,ax=plt.subplots(figsize=(4,4))
                gc=df_data["Gender"].value_counts()
                wedges,texts,autotexts=ax.pie(gc,labels=gc.index,autopct="%1.1f%%",colors=[ACCENT,ACCENT_2],startangle=90,wedgeprops={"edgecolor":BG_CARD,"linewidth":2},textprops={"color":TXT_BODY})
                for a in autotexts: a.set_color(PIE_TXT); a.set_fontweight("bold")
                ax.set_facecolor(BG_CARD)
                st.pyplot(fig,use_container_width=True); plt.close()

        with st.container(border=True):
            st.markdown("#### Scatter Plot: Tinggi vs Berat (Sesuai Filter)")
            fig,ax=plt.subplots(figsize=(10,5))
            style_ax(ax,grid_axis="both")
            for idx2 in sorted(df_view["Index"].unique()):
                sub=df_view[df_view["Index"]==idx2]
                ax.scatter(sub["Height"],sub["Weight"],c=CAT_COLORS[idx2],label=label_map_en[idx2],alpha=0.8,s=26,edgecolors="none")
            ax.set_xlabel("Tinggi Badan (cm)"); ax.set_ylabel("Berat Badan (kg)")
            if len(df_view)>0:
                ax.legend(fontsize=8,labelcolor=TXT_BODY,facecolor=BG_RAISE,edgecolor=BORDER_2)
            st.pyplot(fig,use_container_width=True); plt.close()

        st.markdown("#### Preview Dataset (15 Baris Pertama, Sesuai Filter)")
        st.dataframe(df_view[["Gender","Height","Weight","BMI","Label"]].head(15),hide_index=True,use_container_width=True)

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
                pct=p*100; ind="✓ " if i==pred2 else ""
                st.markdown(
                    '<div class="prob-row">'
                    '<div class="prob-head">'
                    f'<span>{ind}{cat}<span class="prob-sub">{LABEL_MAP[i]}</span></span>'
                    f'<span class="prob-val" style="color:{CAT_COLORS[i]}">{pct:.1f}%</span>'
                    '</div>'
                    '<div class="prob-track">'
                    f'<div class="prob-fill" style="width:{pct}%;background:{CAT_COLORS[i]}"></div>'
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
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        with cb3:
            with st.container(border=True):
                st.markdown("#### Distribusi Probabilitas (Pie)")
                fig,ax=plt.subplots(figsize=(5,4))
                ws,ts,ats=ax.pie(prob_full+0.001,labels=list(LABEL_ID.values()),autopct=lambda p:f"{p:.1f}%" if p>1 else "",colors=CAT_COLORS,startangle=90,wedgeprops={"edgecolor":BG_CARD,"linewidth":1.5},textprops={"color":TXT_BODY})
                for x in ts: x.set_fontsize(7)
                for x in ats: x.set_color(PIE_TXT); x.set_fontsize(7); x.set_fontweight("bold")
                st.pyplot(fig,use_container_width=True); plt.close()

        st.dataframe(pd.DataFrame({"Kategori ID":list(LABEL_ID.values()),"Kategori EN":list(LABEL_MAP.values()),"Probabilitas (%)":[f"{p*100:.2f}%" for p in prob_full],"Prediksi":["YA" if i==pred2 else "-" for i in range(6)]}),hide_index=True,use_container_width=True)

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
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

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
    st.dataframe(pd.DataFrame(REAL_RESULTS),hide_index=True,use_container_width=True)

    SPLIT_INFO=[
        {"Skenario Rasio Split":"Eksperimen 1 (90:10)","Data Latih (Train)":"3,330 sampel (90%)","Data Uji (Testing)":"370 sampel (10%)","Akurasi Terbaik (RF)":"99.73%"},
        {"Skenario Rasio Split":"Eksperimen 2 (80:20)","Data Latih (Train)":"2,960 sampel (80%)","Data Uji (Testing)":"740 sampel (20%)","Akurasi Terbaik (RF)":"99.73%"},
        {"Skenario Rasio Split":"Eksperimen 3 (75:25)","Data Latih (Train)":"2,775 sampel (75%)","Data Uji (Testing)":"925 sampel (25%)","Akurasi Terbaik (RF)":"99.78%"},
        {"Skenario Rasio Split":"Eksperimen 4 (70:30)","Data Latih (Train)":"2,590 sampel (70%)","Data Uji (Testing)":"1,110 sampel (30%)","Akurasi Terbaik (RF)":"99.64%"},
        {"Skenario Rasio Split":"Eksperimen 5 (60:40)","Data Latih (Train)":"2,220 sampel (60%)","Data Uji (Testing)":"1,480 sampel (40%)","Akurasi Terbaik (RF)":"99.66%"},
    ]
    with st.expander("Lihat rincian pembagian jumlah data train & testing (5 varian split)"):
        st.dataframe(pd.DataFrame(SPLIT_INFO),hide_index=True,use_container_width=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
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
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
            st.caption("Model terbaik disorot dengan warna indigo. F1-Score hanya tersedia untuk split 80:20.")

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
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

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
                sns.heatmap(cm_mat,annot=True,fmt="d",cmap="Blues",ax=ax,xticklabels=tnames,yticklabels=tnames,linewidths=0.5,linecolor=BORDER)
                ax.set_xlabel("Predicted Label"); ax.set_ylabel("Actual Label")
                ax.tick_params(colors=TXT_BODY,labelsize=8); ax.tick_params(axis="x",rotation=25)
                ax.set_title("Confusion Matrix: Random Forest (Data Uji 20% = 740 Sampel)",color=TXT_MAIN,fontsize=11)
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        except Exception as e:
            st.warning(f"Gagal memuat Confusion Matrix: {e}")

with t5:
    st.markdown("#### Rekomendasi Kesehatan Personal")
    rec=RECOMMENDATIONS[bi]; rc=rec["color"]
    st.markdown(
        f'<div class="cat-banner" style="border-left:2px solid {rc}">'
        f'<p class="cat-banner-title" style="color:{rc}">{rec["title"]}</p>'
        f'<p class="cat-banner-meta">Kategori: <span style="color:{TXT_MAIN}">{LABEL_ID[bi]}</span> ({LABEL_MAP[bi]}) &nbsp;·&nbsp; '
        f'BMI: <span style="color:{TXT_MAIN}">{bmi_val:.1f}</span> &nbsp;·&nbsp; {height} cm &nbsp;·&nbsp; {weight} kg</p>'
        '</div>', unsafe_allow_html=True)

    st.markdown("<p class='card-sub' style='margin-bottom:6px'>Centang kebiasaan yang sudah kamu jalani:</p>", unsafe_allow_html=True)
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
        checked=st.checkbox("Sudah saya jalani", key=f"habit_{bi}_{i}")
        if checked: done_count+=1
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    prog_ph.progress(done_count/len(rec["items"]), text=f"Progres kebiasaan: {done_count} dari {len(rec['items'])}")
    if done_count==len(rec["items"]):
        st.success("Semua kebiasaan sudah dijalani. Pertahankan konsistensi ini.")

    st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
    st.markdown("#### Panduan Berat Badan Berdasarkan Tinggi Badan Anda")
    td=pd.DataFrame({"Kategori (ID)":list(LABEL_ID.values()),"Kategori (EN)":list(LABEL_MAP.values()),"BMI Range":["< 16","16–18.5","18.5–25","25–30","30–40","> 40"],"Rentang Berat (kg)":[f"{round(lo*(height/100)**2)}–{round(hi*(height/100)**2)}" for lo,hi in [(10,16),(16,18.5),(18.5,25),(25,30),(30,40),(40,55)]]})
    st.dataframe(td,hide_index=True,use_container_width=True)

st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
st.markdown("#### Prediksi Batch (Upload File CSV)")
st.caption("Format kolom CSV: Gender (Male/Female), Height (cm), Weight (kg)")
uploaded_file=st.file_uploader("Upload file CSV",type=["csv"])
if uploaded_file and model_loaded:
    df_batch=pd.read_csv(uploaded_file)
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
        st.dataframe(df_batch.head(30),use_container_width=True)
        st.success(f"Prediksi selesai untuk {len(df_batch)} data.")
        st.download_button("Download Hasil CSV",df_batch.to_csv(index=False).encode("utf-8"),"hasil_prediksi_bmi_skripsi.csv","text/csv")
    else:
        st.error("Format kolom CSV tidak sesuai. Pastikan memiliki kolom: Gender, Height, Weight.")
elif uploaded_file:
    st.warning("Model belum dimuat.")
