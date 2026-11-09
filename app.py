# ============================================================
# BMI ANALYZER v2 — MINIMALIST ACADEMIC EDITION
# Klasifikasi Kategori Indeks Massa Tubuh Menggunakan Simulasi
# Image Processing dan Machine Learning Berbasis Data Antropometri
# Disusun Oleh: Satrio Raja Nabil Rizky (23.11.5538)
# Dosen Pembimbing: Mulia Sulistiyono, S.Kom., M.Kom.
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyBboxPatch
import seaborn as sns
import io, warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="BMI Analyzer — Skripsi S1 Informatika",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── MINIMALIST CLEAN CSS (Nordic / Clinical White & Slate) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #0F172A !important;
}

/* Background Utama & Sidebar Minimalis */
.stApp, .main {
    background-color: #F8FAFC !important;
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}

section[data-testid="stSidebar"] * {
    color: #1E293B !important;
}

/* Tipografi Header */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    color: #0F172A !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* Tabs Minimalis */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #F1F5F9;
    border-radius: 10px;
    padding: 6px;
    border: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    font-weight: 500;
    color: #64748B !important;
    font-size: 13.5px;
    padding: 8px 16px;
    background-color: transparent !important;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
}

/* Kartu Metrik (Metric Container) */
div[data-testid="metric-container"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
}

div[data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-weight: 700 !important;
    font-size: 1.75rem !important;
}

div[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* Tombol Minimalis */
.stButton > button {
    background-color: #0F172A !important;
    color: #FFFFFF !important;
    border: 1px solid #0F172A !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 0.55rem 1.2rem !important;
    width: 100%;
    transition: all 0.15s ease;
}

.stButton > button * {
    color: #FFFFFF !important;
}

.stButton > button:hover {
    background-color: #1E293B !important;
    border-color: #1E293B !important;
    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.1) !important;
}

/* Kartu Konten Minimalis */
.min-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    margin-bottom: 16px;
}

/* Garis Pemisah */
hr {
    border: none !important;
    border-top: 1px solid #E2E8F0 !important;
    margin: 1.25rem 0 !important;
}

/* Expander & Tabel */
div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────
LABEL_MAP  = {0:"Extremely Weak", 1:"Weak", 2:"Normal", 3:"Overweight", 4:"Obesity", 5:"Extreme Obesity"}
LABEL_ID   = {0:"Sangat Kurus",   1:"Kurus", 2:"Normal", 3:"Overweight", 4:"Obesitas", 5:"Sangat Obesitas"}
# Palet warna minimalis profesional (Muted Clinical Palette)
CAT_COLORS = ["#3B82F6", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

RECOMMENDATIONS = {
    0:{"title":"Perlu Perhatian Gizi Ekstra","color":"#3B82F6","items":[
        ("🥗","Surplus Kalori Padat Nutrisi","Utamakan sumber kalori sehat seperti kacang-kacangan, alpukat, beras merah, dan protein hewani."),
        ("🏋️","Latihan Beban Terukur","Lakukan latihan resistensi 3×/minggu untuk membangun massa otot secara bertahap."),
        ("🩺","Konsultasi Medis & Gizi","Evaluasi status metabolisme bersama dokter atau ahli gizi klinis."),
        ("😴","Istirahat Berkualitas","Tidur 7–9 jam per malam untuk mengoptimalkan pemulihan jaringan tubuh.")]},
    1:{"title":"Tingkatkan Asupan Nutrisi Harian","color":"#0EA5E9","items":[
        ("🍳","Protein & Karbohidrat Kompleks","Targetkan 1.5–2g protein/kg BB dengan frekuensi makan 4–5 porsi teratur per hari."),
        ("🚴","Aktivitas Fisik Ringan-Sedang","Olahraga rutin 30 menit/hari untuk menjaga kebugaran kardiovaskular dan nafsu makan."),
        ("🧘","Manajemen Stres","Kelola stres harian agar pola makan dan penyerapan nutrisi tetap optimal.")]},
    2:{"title":"Status Gizi Ideal — Pertahankan!","color":"#10B981","items":[
        ("🎯","Komposisi Tubuh Optimal","IMT Anda berada pada rentang ideal WHO. Pertahankan keseimbangan asupan dan aktivitas."),
        ("❤️","Olahraga Rutin 150 Menit/Minggu","Kombinasikan latihan aerobik dan kekuatan otot secara konsisten."),
        ("🔄","Pemantauan Berkala","Lakukan pengecekan antropometri rutin setiap 3–6 bulan."),
        ("💧","Hidrasi Cukup","Penuhi kebutuhan cairan 2–2.5 liter air putih per hari.")]},
    3:{"title":"Kendalikan Berat Badan Secara Bertahap","color":"#F59E0B","items":[
        ("🚶","Tingkatkan Aktivitas Harian","Targetkan 8.000–10.000 langkah/hari atau latihan kardio 40 menit."),
        ("🥗","Batasi Gula Tambahan & Lemak Jenuh","Perbanyak porsi serat dari sayuran hijau, buah utuh, dan protein rendah lemak."),
        ("📊","Defisit Kalori Terkontrol","Defisit 300–500 kkal/hari mendukung penurunan berat 0.3–0.5 kg/minggu secara aman."),
        ("😴","Jaga Pola Tidur","Kurang tidur dapat meningkatkan hormon ghrelin yang memicu rasa lapar berlebih.")]},
    4:{"title":"Prioritaskan Program Penurunan Berat Badan","color":"#EF4444","items":[
        ("🏥","Pendampingan Medis & Ahli Gizi","Obesitas meningkatkan risiko sindrom metabolik; susun program diet terstruktur bersama ahli."),
        ("🏊","Olahraga Low-Impact","Pilih aktivitas ramah sendi seperti berenang, jalan cepat, atau sepeda statis."),
        ("🥦","Pola Makan Tinggi Serat & Protein","Hindari diet ekstrem; terapkan defisit kalori bertahap dengan gizi seimbang."),
        ("👥","Konsistensi Jangka Panjang","Fokus pada perubahan kebiasaan harian yang berkelanjutan.")]},
    5:{"title":"Intervensi Medis & Gizi Terpadu Diperlukan","color":"#8B5CF6","items":[
        ("🏥","Konsultasi Dokter Spesialis","IMT ≥ 40 memerlukan evaluasi medis menyeluruh dan pengawasan klinis terstruktur."),
        ("❤️","Pemantauan Parameter Klinis","Pantau tekanan darah, profil lipid, dan gula darah secara berkala."),
        ("🚶","Aktivitas Fisik Bertahap","Mulai dari aktivitas ringan berdurasi pendek sesuai toleransi sendi dan kardiovaskular."),
        ("🧠","Pendampingan Holistik","Kombinasikan terapi nutrisi medis dengan dukungan perubahan perilaku sehat.")]}
}

# ── LOAD ARTIFACTS ──────────────────────────────────────────
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
    st.sidebar.warning(f"⚠️ Model belum dimuat: {e}")

# ── IMAGE PROCESSING (Konsisten dengan Notebook Skripsi) ───
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

def build_feature_vector(gender, height, weight):
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

# ── MINIMALIST ARCHITECTURAL SILHOUETTE (Clean & Modern) ───
def draw_human_silhouette(height, weight, gender, idx):
    fig, ax = plt.subplots(figsize=(3.4, 5.2), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(-95, 95)
    ax.set_ylim(-195, 135)
    ax.set_aspect("equal")
    ax.axis("off")

    c = CAT_COLORS[idx]
    hn = np.clip((height - 140) / (200 - 140), 0, 1)
    wn = np.clip((weight - 35) / (165 - 35), 0, 1)

    # Garis grid pengukur minimalis di latar belakang
    for y_grid in range(-160, 121, 40):
        ax.axhline(y_grid, color="#F1F5F9", lw=0.8, zorder=0)

    bh = 78 + hn * 46
    bw = 17 + wn * 40
    sw = bw * (1.38 if gender == "Male" else 1.15)
    hipw = bw * (1.05 if gender == "Male" else 1.32)
    hr = 20 + wn * 3
    lh = 68 + hn * 24
    lw = bw * 0.36
    aw = 7.5 + wn * 7.5

    # Lingkaran Halo Minimalis di sekitar Kepala
    ax.add_patch(plt.Circle((0, 96), hr + 6, fill=False, edgecolor=c, alpha=0.25, lw=1.5, zorder=2))
    ax.add_patch(plt.Circle((0, 96), hr, color=c, alpha=0.92, zorder=3))

    # Leher & Bahu
    ax.add_patch(plt.Rectangle((-5.5, 96 - hr - 12), 11, 14, color=c, alpha=0.92, zorder=3))
    ax.add_patch(Ellipse((0, 96 - hr - 9), sw * 2, 16, color=c, alpha=0.92, zorder=3))

    # Torso Proporsional
    pts = np.array([
        [-sw, 96 - hr - 12],
        [-hipw, 96 - hr - bh],
        [hipw, 96 - hr - bh],
        [sw, 96 - hr - 12]
    ])
    ax.add_patch(plt.Polygon(pts, closed=True, color=c, alpha=0.92, zorder=3))
    ax.add_patch(Ellipse((0, 96 - hr - bh * 0.52), bw * 2 + 2, bh * 0.92, color=c, alpha=0.92, zorder=2))

    # Lengan Kiri & Kanan
    ax.add_patch(Ellipse((-sw - aw * 0.45, 96 - hr - bh * 0.42), aw + 1.5, bh * 0.68, angle=-5, color=c, alpha=0.85, zorder=2))
    ax.add_patch(Ellipse((sw + aw * 0.45, 96 - hr - bh * 0.42), aw + 1.5, bh * 0.68, angle=5, color=c, alpha=0.85, zorder=2))

    # Kaki Kiri & Kanan
    lt = 96 - hr - bh
    ax.add_patch(Ellipse((-hipw * 0.36, lt - lh * 0.5), lw * 1.9, lh, angle=2, color=c, alpha=0.92, zorder=3))
    ax.add_patch(Ellipse((hipw * 0.36, lt - lh * 0.5), lw * 1.9, lh, angle=-2, color=c, alpha=0.92, zorder=3))

    # Label Kategori & Ukuran di Bawah Siluet
    ax.text(0, lt - lh - 22, f"{LABEL_ID[idx].upper()}", ha="center", fontsize=9.5, fontweight="bold", color="#0F172A")
    ax.text(0, lt - lh - 35, f"{height} cm  ·  {weight} kg", ha="center", fontsize=8, color="#64748B")

    plt.tight_layout(pad=0.2)
    return fig

# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Parameter Antropometri")
    st.caption("Masukkan data fisik untuk simulasi citra dan klasifikasi IMT.")
    st.markdown("---")
    gender = st.radio("Jenis Kelamin", ["Male", "Female"], format_func=lambda x: "Laki-laki (Male)" if x == "Male" else "Perempuan (Female)")
    height = st.slider("Tinggi Badan (cm)", 140, 200, 170)
    weight = st.slider("Berat Badan (kg)", 35, 165, 70)

    bmi_val = weight / (height / 100) ** 2
    bi = bmi_index(bmi_val)
    c2 = CAT_COLORS[bi]

    st.markdown("---")
    st.markdown(f"""
    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-left:4px solid {c2}; border-radius:10px; padding:14px;">
        <div style="font-size:11.5px; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;">Indeks Massa Tubuh</div>
        <div style="font-size:1.85rem; font-weight:800; color:#0F172A; margin:2px 0;">{bmi_val:.1f} <span style="font-size:0.85rem; font-weight:500; color:#64748B;">kg/m²</span></div>
        <div style="font-size:0.95rem; font-weight:700; color:{c2};">{LABEL_ID[bi]} <span style="font-weight:400; color:#64748B; font-size:0.8rem;">({LABEL_MAP[bi]})</span></div>
    </div>
    """, unsafe_allow_html=True)

    il = round(18.5 * (height / 100) ** 2)
    ih = round(24.9 * (height / 100) ** 2)
    st.caption(f"Rentang Berat Ideal: **{il} – {ih} kg**")
    st.markdown("---")
    predict_btn = st.button("Jalankan Prediksi Model", use_container_width=True)
    if st.button("Catat Log ke MLflow", use_container_width=True):
        try:
            import mlflow
            mlflow.set_experiment("BMI_Analyzer_v2_Skripsi")
            with mlflow.start_run():
                mlflow.log_params({"gender": gender, "height": height, "weight": weight})
                mlflow.log_metrics({"bmi": round(bmi_val, 2), "category_index": bi})
            st.sidebar.success("Tercatat di MLflow!")
        except Exception as e:
            st.sidebar.error(str(e))

# ── HEADER MINIMALIS ────────────────────────────────────────
st.markdown("""
<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:22px 28px; margin-bottom:20px; box-shadow:0 1px 2px rgba(15,23,42,0.03);">
    <div style="font-size:11.5px; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">
        Sistem Pendukung Analisis Antropometri · Universitas AMIKOM Yogyakarta
    </div>
    <h1 style="font-size:1.75rem; font-weight:800; color:#0F172A; margin:0 0 4px 0;">
        Klasifikasi Kategori Indeks Massa Tubuh
    </h1>
    <p style="color:#475569; font-size:0.9rem; margin:0;">
        Simulasi <i>Image Processing</i> Morfologi Siluet &amp; <i>Machine Learning</i> &nbsp;·&nbsp; <b>Satrio Raja Nabil Rizky (23.11.5538)</b>
    </p>
</div>
""", unsafe_allow_html=True)

# ── TABS ────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "Kalkulator & Siluet",
    "Eksplorasi Data (n=3.700)",
    "Distribusi Probabilitas",
    "Evaluasi 6 Model ML",
    "Rekomendasi Klinis"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — KALKULATOR BMI & SILUET
# ═══════════════════════════════════════════════════════════
with t1:
    ca, cb = st.columns([1, 1.85], gap="large")
    with ca:
        st.markdown("#### Visualisasi Proporsi Siluet")
        fig_s = draw_human_silhouette(height, weight, gender, bi)
        st.pyplot(fig_s, use_container_width=True)
        plt.close()

        with st.expander("Lihat Citra Biner OpenCV & Fitur Morfologi", expanded=True):
            img_cv = generate_body_silhouette(height, weight, gender)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
            cnts, _ = cv2.findContours(img_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_rgb, cnts, -1, (16, 185, 129), 2)
            st.image(img_rgb, caption="Citra Siluet Biner OpenCV (128×128 px) + Kontur", use_container_width=True)
            f_img = extract_image_features(img_cv)
            st.json({
                "pixel_area": int(f_img[0]),
                "aspect_ratio": f_img[1],
                "body_density": f_img[2],
                "contour_perimeter": f_img[3],
                "compactness": f_img[4]
            })

    with cb:
        st.markdown("#### Ringkasan Analisis Antropometri")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nilai IMT / BMI", f"{bmi_val:.1f} kg/m²")
        m2.metric("Kategori Status Gizi", LABEL_ID[bi])
        diff = weight - il if weight < il else (weight - ih if weight > ih else 0)
        status_gap = "Ideal (0 kg)" if diff == 0 else (f"+{abs(diff):.0f} kg di atas ideal" if diff > 0 else f"-{abs(diff):.0f} kg di bawah ideal")
        m3.metric("Selisih Berat Ideal", status_gap)

        # Indikator Posisi Skala IMT Minimalis
        bp = min(max((bmi_val - 10) / 45 * 100, 0), 100)
        st.markdown(f"""
        <div class="min-card" style="margin-top:14px; padding:16px 20px;">
            <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:600; color:#475569; margin-bottom:8px;">
                <span>Posisi pada Skala IMT (WHO)</span>
                <span style="color:#0F172A;">{bmi_val:.2f} kg/m²</span>
            </div>
            <div style="position:relative; height:10px; border-radius:6px; background:linear-gradient(to right, #3B82F6 0%, #0EA5E9 18%, #10B981 35%, #F59E0B 55%, #EF4444 78%, #8B5CF6 100%); margin-bottom:8px;">
                <div style="position:absolute; top:50%; transform:translate(-50%,-50%); width:16px; height:16px; border-radius:50%; background:#0F172A; border:3px solid #FFFFFF; left:{bp}%; box-shadow:0 1px 4px rgba(0,0,0,0.25);"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#64748B;">
                <span>&lt;16 (Sangat Kurus)</span><span>18.5–24.9 (Normal)</span><span>25–29.9 (Overweight)</span><span>30–39.9 (Obesitas)</span><span>&ge;40 (Ekstrem)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if model_loaded:
            fv = build_feature_vector(gender, height, weight)
            fv_sc = scaler.transform([fv])
            pred = model.predict(fv_sc)[0]
            proba = model.predict_proba(fv_sc)[0]
            conf = max(proba) * 100
            c3 = CAT_COLORS[pred]
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {c3}; border-radius:10px; padding:16px 20px; margin-bottom:16px; box-shadow:0 1px 2px rgba(15,23,42,0.03);">
                <div style="font-size:11.5px; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:0.04em;">Hasil Klasifikasi Model Machine Learning</div>
                <div style="font-size:1.25rem; font-weight:700; color:#0F172A; margin-top:4px;">
                    {LABEL_ID[pred]} <span style="font-weight:500; color:{c3};">({LABEL_MAP[pred]})</span>
                </div>
                <div style="font-size:12.5px; color:#64748B; margin-top:4px;">
                    Tingkat Keyakinan (<i>Confidence</i>): <b>{conf:.1f}%</b> &nbsp;·&nbsp; Algoritma: <b>Random Forest</b> (Akurasi 99.73% pada n=3.700)
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Pastikan model_bmi.pkl dan scaler_bmi.pkl tersedia di repositori.")

        info_df = pd.DataFrame({
            "Atribut / Fitur Ekstraksi": [
                "Tinggi Badan (Height)", "Berat Badan (Weight)", "Indeks Massa Tubuh (BMI)",
                "Kategori Target (EN)", "Kategori Target (ID)", "Rentang Berat Ideal",
                "Luas Piksel Siluet (pixel_area)", "Rasio Aspek Siluet (aspect_ratio)",
                "Kepadatan Piksel (body_density)", "Keliling Kontur (contour_perimeter)", "Kekompakan Bentuk (compactness)"
            ],
            "Nilai Parameter": [
                f"{height} cm", f"{weight} kg", f"{bmi_val:.2f} kg/m²",
                LABEL_MAP[bi], LABEL_ID[bi], f"{il} – {ih} kg",
                f"{int(f_img[0])} piksel²", f"{f_img[1]:.4f}",
                f"{f_img[2]:.4f}", f"{f_img[3]:.4f} piksel", f"{f_img[4]:.4f}"
            ]
        })
        st.dataframe(info_df, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2 — ANALISIS DATA (MINIMALIST LIGHT CHARTS)
# ═══════════════════════════════════════════════════════════
with t2:
    try:
        df_data = pd.read_csv("bmi.csv")
        label_map_en = {0:"Extremely Weak", 1:"Weak", 2:"Normal", 3:"Overweight", 4:"Obesity", 5:"Extreme Obesity"}
        df_data["Label"] = df_data["Index"].map(label_map_en)
        df_data["BMI"] = df_data["Weight"] / (df_data["Height"] / 100) ** 2

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Sampel Dataset", f"{len(df_data):,}")
        s2.metric("Jumlah Fitur Prediktor", "16 Fitur")
        s3.metric("Kelas Target IMT", "6 Kelas")
        s4.metric("Akurasi Tertinggi (RF)", "99.78%")

        st.markdown("---")
        ca2, cb2 = st.columns([1.3, 1], gap="large")
        with ca2:
            st.markdown("#### Distribusi Kelas Target IMT")
            fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")
            cnt = df_data["Index"].value_counts().sort_index()
            bars = ax.bar([label_map_en[i] for i in cnt.index], cnt.values, color=[CAT_COLORS[i] for i in cnt.index], alpha=0.9, edgecolor="#0F172A", linewidth=0.4, width=0.58)
            for b, v in zip(bars, cnt.values):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 18, str(v), ha="center", va="bottom", color="#0F172A", fontsize=8.5, fontweight="bold")
            ax.set_ylabel("Jumlah Responden", color="#334155", fontsize=9)
            ax.set_ylim(0, max(cnt.values) * 1.15)
            ax.tick_params(colors="#334155", labelsize=8)
            ax.tick_params(axis="x", rotation=20)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#CBD5E1")
            ax.grid(axis="y", color="#F1F5F9", linestyle="--")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with cb2:
            st.markdown("#### Proporsi Jenis Kelamin")
            fig, ax = plt.subplots(figsize=(4.5, 4), dpi=150)
            fig.patch.set_facecolor("#FFFFFF")
            gc = df_data["Gender"].value_counts()
            wedges, texts, autotexts = ax.pie(
                gc, labels=gc.index, autopct="%1.1f%%",
                colors=["#0F172A", "#94A3B8"], startangle=90,
                wedgeprops={"edgecolor": "#FFFFFF", "linewidth": 2, "width": 0.45}
            )
            for t in texts: t.set_color("#0F172A"); t.set_fontsize(9); t.set_fontweight("600")
            for a in autotexts: a.set_color("#FFFFFF"); a.set_fontsize(8.5); a.set_fontweight("bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("#### Sebaran Tinggi terhadap Berat Badan Berdasarkan Kategori IMT")
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        for idx2 in sorted(df_data["Index"].unique()):
            sub = df_data[df_data["Index"] == idx2]
            ax.scatter(sub["Height"], sub["Weight"], c=CAT_COLORS[idx2], label=label_map_en[idx2], alpha=0.75, s=24, edgecolors="none")
        ax.set_xlabel("Tinggi Badan (cm)", color="#334155", fontsize=9.5)
        ax.set_ylabel("Berat Badan (kg)", color="#334155", fontsize=9.5)
        ax.tick_params(colors="#334155", labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#CBD5E1")
        ax.legend(fontsize=8.5, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")
        ax.grid(color="#F1F5F9", linestyle="--")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("#### Sampel Data Antropometri (15 Baris Pertama)")
        st.dataframe(df_data[["Gender", "Height", "Weight", "BMI", "Label"]].head(15), hide_index=True, use_container_width=True)

    except FileNotFoundError:
        st.warning("Pastikan file bmi.csv tersedia di repositori.")

# ═══════════════════════════════════════════════════════════
# TAB 3 — DISTRIBUSI PROBABILITAS
# ═══════════════════════════════════════════════════════════
with t3:
    st.markdown("#### Distribusi Probabilitas Prediksi Kelas")

    if model_loaded:
        fv2 = build_feature_vector(gender, height, weight)
        fv2_sc = scaler.transform([fv2])
        proba2 = model.predict_proba(fv2_sc)[0]
        pred2 = model.predict(fv2_sc)[0]

        prob_full = np.zeros(6)
        for i, cls in enumerate(model.classes_):
            prob_full[cls] = proba2[i]

        st.caption(f"Estimasi probabilitas model **Random Forest** untuk profil input saat ini (IMT: **{bmi_val:.1f} kg/m²**):")

        for i, (cat, p) in enumerate(zip(LABEL_ID.values(), prob_full)):
            pct = p * 100
            ind = "● " if i == pred2 else ""
            fw = "700" if i == pred2 else "500"
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
                    <span style="color:#0F172A; font-weight:{fw};">{ind}{cat} <span style="color:#64748B; font-weight:400;">({LABEL_MAP[i]})</span></span>
                    <span style="color:#0F172A; font-weight:700;">{pct:.1f}%</span>
                </div>
                <div style="height:8px; background:#E2E8F0; border-radius:4px; overflow:hidden;">
                    <div style="height:100%; width:{pct}%; background:{CAT_COLORS[i]}; border-radius:4px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.dataframe(
            pd.DataFrame({
                "Kategori (ID)": list(LABEL_ID.values()),
                "Kategori (EN)": list(LABEL_MAP.values()),
                "Probabilitas Prediksi": [f"{p*100:.2f}%" for p in prob_full],
                "Status Keputusan": ["Terpilih (Prediksi Utama)" if i == pred2 else "—" for i in range(6)]
            }),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("Model belum dimuat.")

# ═══════════════════════════════════════════════════════════
# TAB 4 — EVALUASI 6 MODEL ML (5 VARIAN SPLIT)
# ═══════════════════════════════════════════════════════════
with t4:
    st.markdown("#### Perbandingan Performa 6 Algoritma Machine Learning")
    st.caption("Evaluasi pada dataset gabungan `bmi.csv` (**n = 3.700 sampel**) · **5 Skenario Rasio Split (90:10, 80:20, 75:25, 70:30, 60:40)** · **16 Fitur**")

    REAL_RESULTS = [
        {"Model":"Random Forest",     "Acc (90:10)":"99.73%", "Acc (80:20)":"99.73%", "Acc (75:25)":"99.78%", "Acc (70:30)":"99.64%", "Acc (60:40)":"99.66%", "F1-Score (80:20)":"99.73%"},
        {"Model":"Gradient Boosting", "Acc (90:10)":"99.46%", "Acc (80:20)":"99.59%", "Acc (75:25)":"99.68%", "Acc (70:30)":"99.55%", "Acc (60:40)":"99.66%", "F1-Score (80:20)":"99.60%"},
        {"Model":"Decision Tree",     "Acc (90:10)":"99.46%", "Acc (80:20)":"99.59%", "Acc (75:25)":"99.46%", "Acc (70:30)":"99.19%", "Acc (60:40)":"99.39%", "F1-Score (80:20)":"99.60%"},
        {"Model":"SVM (RBF)",         "Acc (90:10)":"97.30%", "Acc (80:20)":"97.57%", "Acc (75:25)":"97.62%", "Acc (70:30)":"97.39%", "Acc (60:40)":"97.16%", "F1-Score (80:20)":"97.54%"},
        {"Model":"Logistic Reg.",     "Acc (90:10)":"97.57%", "Acc (80:20)":"97.03%", "Acc (75:25)":"96.86%", "Acc (70:30)":"96.94%", "Acc (60:40)":"96.89%", "F1-Score (80:20)":"96.86%"},
        {"Model":"KNN",               "Acc (90:10)":"96.22%", "Acc (80:20)":"96.49%", "Acc (75:25)":"96.65%", "Acc (70:30)":"96.58%", "Acc (60:40)":"95.34%", "F1-Score (80:20)":"96.44%"},
    ]
    st.dataframe(pd.DataFrame(REAL_RESULTS), hide_index=True, use_container_width=True)

    SPLIT_INFO = [
        {"Skenario Rasio Split":"Eksperimen 1 (90:10)", "Data Latih (Train)":"3.330 sampel (90%)", "Data Uji (Testing)":"370 sampel (10%)",   "Akurasi Terbaik (RF)":"99.73%"},
        {"Skenario Rasio Split":"Eksperimen 2 (80:20)", "Data Latih (Train)":"2.960 sampel (80%)", "Data Uji (Testing)":"740 sampel (20%)",   "Akurasi Terbaik (RF)":"99.73%"},
        {"Skenario Rasio Split":"Eksperimen 3 (75:25)", "Data Latih (Train)":"2.775 sampel (75%)", "Data Uji (Testing)":"925 sampel (25%)",   "Akurasi Terbaik (RF)":"99.78%"},
        {"Skenario Rasio Split":"Eksperimen 4 (70:30)", "Data Latih (Train)":"2.590 sampel (70%)", "Data Uji (Testing)":"1.110 sampel (30%)", "Akurasi Terbaik (RF)":"99.64%"},
        {"Skenario Rasio Split":"Eksperimen 5 (60:40)", "Data Latih (Train)":"2.220 sampel (60%)", "Data Uji (Testing)":"1.480 sampel (40%)", "Akurasi Terbaik (RF)":"99.66%"},
    ]
    with st.expander("Lihat Rincian Pembagian Jumlah Data Train & Testing (5 Varian Split)"):
        st.dataframe(pd.DataFrame(SPLIT_INFO), hide_index=True, use_container_width=True)

    st.markdown("---")
    ca4, cb4 = st.columns(2, gap="large")
    with ca4:
        st.markdown("#### Grafik Akurasi Model (Split 80:20)")
        fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        nms = ["Random Forest", "Gradient Boosting", "Decision Tree", "SVM (RBF)", "Logistic Reg.", "KNN"]
        acs = [99.73, 99.59, 99.59, 97.57, 97.03, 96.49]
        cls_c = ["#0F172A" if a >= 99.5 else "#475569" if a >= 97 else "#94A3B8" for a in acs]
        brs = ax.barh(nms[::-1], acs[::-1], color=cls_c[::-1], height=0.58)
        for b, v in zip(brs, acs[::-1]):
            ax.text(v + 0.15, b.get_y() + b.get_height() / 2, f"{v:.2f}%", ha="left", va="center", color="#0F172A", fontsize=8.5, fontweight="600")
        ax.set_xlim(92, 101.8)
        ax.set_xlabel("Accuracy (%)", color="#334155", fontsize=9)
        ax.tick_params(colors="#334155", labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#CBD5E1")
        ax.grid(axis="x", color="#F1F5F9", linestyle="--")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with cb4:
        st.markdown("#### Kontribusi Fitur (Random Forest)")
        FEAT_IMP = [("BMI²", 32.71), ("BMI", 29.36), ("Weight/Height", 14.11), ("Weight²", 5.33), ("Weight", 5.17), ("aspect_ratio", 3.77), ("Weight_category", 2.53), ("body_density", 1.70)]
        fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        img_feats_names = ["aspect_ratio", "pixel_area", "body_density", "contour_perimeter", "compactness"]
        fn = [f[0] for f in FEAT_IMP]
        fi = [f[1] for f in FEAT_IMP]
        clrs2 = ["#10B981" if f in img_feats_names else "#0F172A" for f in fn]
        ax.barh(fn[::-1], fi[::-1], color=clrs2[::-1], height=0.58)
        ax.set_xlabel("Importance (%)", color="#334155", fontsize=9)
        ax.tick_params(colors="#334155", labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#CBD5E1")
        ax.grid(axis="x", color="#F1F5F9", linestyle="--")
        green_p = mpatches.Patch(color="#10B981", label="Fitur Citra Siluet")
        slate_p = mpatches.Patch(color="#0F172A", label="Fitur Antropometri")
        ax.legend(handles=[green_p, slate_p], fontsize=8, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    if model_loaded:
        try:
            @st.cache_data
            def get_confusion_matrix():
                df_test = pd.read_csv("bmi.csv")
                df_test["Height_m"] = df_test["Height"] / 100
                df_test["BMI"] = df_test["Weight"] / (df_test["Height_m"] ** 2)
                df_test["BMI_squared"] = df_test["BMI"] ** 2
                df_test["Weight_Height_ratio"] = df_test["Weight"] / df_test["Height"]
                df_test["Weight_squared"] = df_test["Weight"] ** 2
                df_test["Height_squared"] = df_test["Height"] ** 2
                df_test["Gender_num"] = (df_test["Gender"] == "Male").astype(int)
                df_test["Height_category"] = pd.cut(df_test["Height"], bins=[0,155,170,185,300], labels=[0,1,2,3]).astype(int)
                df_test["Weight_category"] = pd.cut(df_test["Weight"], bins=[0,60,80,100,300], labels=[0,1,2,3]).astype(int)
                img_feats_all = [extract_image_features(generate_body_silhouette(r.Height, r.Weight, r.Gender)) for _, r in df_test.iterrows()]
                df_img2 = pd.DataFrame(img_feats_all, columns=["pixel_area","aspect_ratio","body_density","contour_perimeter","compactness"])
                feat_cols = ["Gender_num","Height","Weight","Height_m","BMI","BMI_squared","Weight_Height_ratio","Weight_squared","Height_squared","Height_category","Weight_category","pixel_area","aspect_ratio","body_density","contour_perimeter","compactness"]
                X_all = pd.concat([df_test[feat_cols[:11]], df_img2], axis=1)
                X_sc_all = scaler.transform(X_all)
                y_all = df_test["Index"]
                from sklearn.metrics import confusion_matrix as cm_fn
                from sklearn.model_selection import train_test_split as tts
                Xtr, Xte, ytr, yte = tts(pd.DataFrame(X_sc_all, columns=feat_cols), y_all, test_size=0.2, random_state=42, stratify=y_all)
                yp = model.predict(Xte)
                return cm_fn(yte, yp)

            cm_mat = get_confusion_matrix()
            st.markdown("#### Confusion Matrix — Random Forest (Skenario Split 80:20 · 740 Sampel Uji)")
            fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")
            tnames = ["Ext.Weak", "Weak", "Normal", "Overweight", "Obesity", "Ext.Obesity"]
            sns.heatmap(
                cm_mat, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=tnames, yticklabels=tnames,
                linewidths=0.8, linecolor="#E2E8F0"
            )
            ax.set_xlabel("Label Prediksi Model", color="#0F172A", fontsize=9)
            ax.set_ylabel("Label Aktual", color="#0F172A", fontsize=9)
            ax.tick_params(colors="#334155", labelsize=8)
            ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        except Exception as e:
            st.warning(f"Gagal memuat Confusion Matrix: {e}")

# ═══════════════════════════════════════════════════════════
# TAB 5 — REKOMENDASI KLINIS MINIMALIS
# ═══════════════════════════════════════════════════════════
with t5:
    st.markdown("#### Rekomendasi Kesehatan & Gizi Personal")
    rec = RECOMMENDATIONS[bi]
    rc = rec["color"]
    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {rc}; border-radius:12px; padding:18px 22px; margin-bottom:18px;">
        <div style="font-size:1.15rem; font-weight:700; color:#0F172A; margin-bottom:4px;">{rec["title"]}</div>
        <div style="color:#64748B; font-size:0.88rem;">
            Status Gizi: <b style="color:{rc};">{LABEL_ID[bi]}</b> ({LABEL_MAP[bi]}) &nbsp;·&nbsp;
            IMT: <b>{bmi_val:.1f} kg/m²</b> &nbsp;·&nbsp;
            Tinggi: <b>{height} cm</b> &nbsp;·&nbsp; Berat: <b>{weight} kg</b>
        </div>
    </div>""", unsafe_allow_html=True)

    c_rec1, c_rec2 = st.columns(2, gap="medium")
    for idx_item, (em, tit, desc) in enumerate(rec["items"]):
        target_col = c_rec1 if idx_item % 2 == 0 else c_rec2
        with target_col:
            st.markdown(f"""
            <div class="min-card" style="padding:16px 18px; margin-bottom:12px;">
                <div style="display:flex; align-items:flex-start; gap:12px;">
                    <span style="font-size:1.4rem;">{em}</span>
                    <div>
                        <div style="font-weight:600; color:#0F172A; font-size:0.92rem; margin-bottom:3px;">{tit}</div>
                        <div style="color:#64748B; font-size:0.84rem; line-height:1.55;">{desc}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Tabel Acuan Berat Badan Berdasarkan Tinggi Badan Anda")
    td = pd.DataFrame({
        "Kategori Status Gizi (ID)": list(LABEL_ID.values()),
        "Kategori Status Gizi (EN)": list(LABEL_MAP.values()),
        "Rentang IMT (kg/m²)": ["< 16.0", "16.0 – 18.49", "18.5 – 24.99", "25.0 – 29.99", "30.0 – 39.99", "≥ 40.0"],
        f" Estimasi Berat untuk Tinggi {height} cm": [
            f"{round(lo*(height/100)**2)} – {round(hi*(height/100)**2)} kg"
            for lo, hi in [(10,16), (16,18.5), (18.5,25), (25,30), (30,40), (40,55)]
        ]
    })
    st.dataframe(td, hide_index=True, use_container_width=True)

# ── BATCH PREDICTION ────────────────────────────────────────
st.markdown("---")
st.markdown("#### Prediksi Massal (Batch Upload CSV)")
st.caption("Unggah file CSV dengan kolom: `Gender` (Male/Female), `Height` (cm), dan `Weight` (kg).")
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
if uploaded_file and model_loaded:
    df_batch = pd.read_csv(uploaded_file)
    col_map = {c.lower().strip(): c for c in df_batch.columns}
    if all(k in col_map for k in ["gender", "height", "weight"]):
        df_batch = df_batch.rename(columns={col_map["gender"]:"Gender", col_map["height"]:"Height", col_map["weight"]:"Weight"})
        df_batch = df_batch.dropna(subset=["Gender", "Height", "Weight"])
        st.write(f"Data valid ditemukan: **{len(df_batch)} baris**")
        batch_fv = [build_feature_vector(str(r.Gender).strip(), float(r.Height), float(r.Weight)) for _, r in df_batch.iterrows()]
        batch_sc = scaler.transform(batch_fv)
        df_batch["BMI"] = (df_batch["Weight"] / (df_batch["Height"] / 100) ** 2).round(2)
        df_batch["Prediksi_Index"] = model.predict(batch_sc)
        df_batch["Prediksi_EN"] = df_batch["Prediksi_Index"].map(LABEL_MAP)
        df_batch["Prediksi_ID"] = df_batch["Prediksi_Index"].map(LABEL_ID)
        st.dataframe(df_batch.head(30), use_container_width=True)
        st.success(f"Prediksi selesai untuk {len(df_batch)} data responden.")
        st.download_button("Unduh Hasil Prediksi (CSV)", df_batch.to_csv(index=False).encode("utf-8"), "hasil_prediksi_bmi_minimalist.csv", "text/csv")
    else:
        st.error("Format kolom CSV tidak sesuai. Pastikan terdapat kolom: Gender, Height, Weight.")
elif uploaded_file:
    st.warning("Model belum dimuat.")
