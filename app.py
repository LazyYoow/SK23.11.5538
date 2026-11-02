# ============================================================
# BMI ANALYZER v2 — SKRIPSI EDITION
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
from matplotlib.patches import Ellipse
import seaborn as sns
import io, warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="BMI Analyzer v2 — Skripsi S1 Informatika",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS (Dark Aesthetic) ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif!important;}
.main{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);}
section[data-testid="stSidebar"]{background:rgba(15,12,41,0.95)!important;border-right:1px solid rgba(255,255,255,0.08);}
h1,h2,h3{font-family:'Plus Jakarta Sans',sans-serif!important;font-weight:800!important;}
.stTab [data-baseweb="tab-list"]{gap:6px;background:rgba(255,255,255,0.04);border-radius:14px;padding:4px;}
.stTab [data-baseweb="tab"]{border-radius:10px;font-weight:600;color:rgba(255,255,255,0.5);font-size:13px;}
.stTab [aria-selected="true"]{background:linear-gradient(135deg,#6C63FF,#FF6B9D)!important;color:white!important;}
div[data-testid="metric-container"]{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:16px;padding:1rem;}
.stSlider .stMarkdown{color:rgba(255,255,255,0.7);}
.stButton>button{background:linear-gradient(135deg,#6C63FF,#FF6B9D);color:white;border:none;border-radius:12px;font-weight:700;font-size:14px;padding:0.6rem 1.5rem;width:100%;}
.stButton>button:hover{opacity:0.9;transform:translateY(-1px);}
.glow-card{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:20px;padding:1.5rem;backdrop-filter:blur(10px);margin-bottom:1rem;}
.rec-card{border-radius:16px;padding:1.2rem;margin-bottom:10px;border-left:4px solid;}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;font-size:13px;margin-top:6px;}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────
LABEL_MAP   = {0:"Extremely Weak",1:"Weak",2:"Normal",3:"Overweight",4:"Obesity",5:"Extreme Obesity"}
LABEL_ID    = {0:"Sangat Kurus",  1:"Kurus",  2:"Normal", 3:"Overweight",4:"Obesitas",5:"Sangat Obesitas"}
CAT_COLORS  = ["#3B82F6","#22C55E","#00D4A1","#F59E0B","#EF4444","#7C3AED"]
CAT_BG      = ["#1e3a5f","#14532d","#065f46","#78350f","#7f1d1d","#3b0764"]

RECOMMENDATIONS = {
    0:{"title":"⚠️ Perlu Perhatian Ekstra","color":"#3B82F6","items":[
        ("🥗","Perbanyak Kalori Bergizi","Makanan padat nutrisi: kacang, alpukat, nasi merah, protein hewani berkualitas."),
        ("🏋️","Latihan Pembentuk Otot","Latihan beban 3×/minggu. Target massa otot bukan sekadar angka timbangan."),
        ("🩺","Konsultasi Dokter Gizi","BMI sangat rendah bisa menandakan kondisi medis tersembunyi."),
        ("😴","Tidur 7–9 Jam Cukup","Tidur berkualitas mendukung hormon pertumbuhan & nafsu makan.")]},
    1:{"title":"📈 Tingkatkan Nutrisi","color":"#22C55E","items":[
        ("🍳","Protein & Karbohidrat Kompleks","Target 1.5–2g protein/kg BB. Makan 5–6 kali porsi kecil per hari."),
        ("🚴","Olahraga Ringan Rutin","30 menit jalan cepat atau bersepeda per hari. Bangun metabolisme."),
        ("🧘","Kelola Stres","Stres menekan nafsu makan. Meditasi 10 menit/hari sangat membantu.")]},
    2:{"title":"🏆 Pertahankan Pola Hidup Sehat!","color":"#00D4A1","items":[
        ("🎯","Kamu Sudah di Jalur Tepat!","BMI ideal! Pertahankan pola makan seimbang dan olahraga rutin."),
        ("❤️","Olahraga Rutin & Gizi Seimbang","150 menit aktivitas sedang/minggu. Perbanyak sayur, buah, protein."),
        ("🔄","Cek Rutin 3–6 Bulan","Monitor BMI periodik. Perubahan kecil lebih mudah ditangani dini."),
        ("💧","Hidrasi Optimal","2–2.5 liter air/hari. Kurangi minuman manis dan soda.")]},
    3:{"title":"⚡ Kendalikan Berat Badan","color":"#F59E0B","items":[
        ("🚶","Mulai Aktif Sekarang","10.000 langkah/hari atau 45 menit cardio. Gunakan tangga lebih sering."),
        ("🥗","Kurangi Gula & Gorengan","Ganti dengan whole food. Biasakan membaca label nutrisi sebelum beli."),
        ("📊","Defisit Kalori Aman","300–500 kkal/hari = turun 0.3–0.5 kg/minggu secara berkelanjutan."),
        ("😴","Perbaiki Kualitas Tidur","Kurang tidur naikkan hormon lapar. Target 7–9 jam/malam.")]},
    4:{"title":"🚨 Prioritaskan Kesehatan","color":"#EF4444","items":[
        ("🏥","Konsultasi Dokter Segera","Obesitas tingkatkan risiko diabetes, hipertensi, penyakit jantung."),
        ("🏊","Olahraga Rendah Benturan","Renang atau sepeda statis aman untuk sendi. Mulai 20 menit/hari."),
        ("🥦","Diet Terstruktur","Hindari crash diet. Makan 3×/hari porsi terkontrol, banyak serat."),
        ("👥","Dukungan Sosial","Bergabung komunitas kesehatan. Konsistensi lebih penting dari sempurna.")]},
    5:{"title":"🆘 Tindakan Segera Diperlukan","color":"#7C3AED","items":[
        ("🏥","Program Medis Profesional","BMI >40: intervensi medis mendesak. Hubungi dokter spesialis obesitas."),
        ("❤️","Monitor Tekanan Darah","Cek rutin. Target <120/80 mmHg. Kurangi garam & makanan olahan."),
        ("🚶","Mulai dari Langkah Kecil","Berjalan 10 menit 3×/hari sudah bermakna klinis. Tingkatkan perlahan."),
        ("🧠","Dukungan Psikologis","Konseling perilaku makan membantu atasi akar hubungan emosi-makanan.")]}
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
    st.sidebar.warning(f"⚠️ Error: {e}")

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

# ── SILHOUETTE MATPLOTLIB (human-like) ──────────────────────
def draw_human_silhouette(height, weight, gender, idx):
    fig,ax=plt.subplots(figsize=(3.5,5.5))
    fig.patch.set_facecolor("#0f0c29"); ax.set_facecolor("#0f0c29")
    ax.set_xlim(-100,100); ax.set_ylim(-200,140); ax.set_aspect("equal"); ax.axis("off")
    c=CAT_COLORS[idx]
    hn=(height-140)/(200-140); wn=min(max((weight-35)/(165-35),0),1)
    bh=80+hn*50; bw=18+wn*44
    sw=bw*(1.45 if gender=="Male" else 1.18); hipw=bw*(1.08 if gender=="Male" else 1.38)
    hr=22+wn*4; lh=70+hn*25; lw=bw*0.38; aw=8+wn*8
    ax.add_patch(plt.Circle((0,100),hr+14,color=c,alpha=0.12))
    if gender=="Female":
        ax.add_patch(Ellipse((0,100+hr*0.25),hr*2.1,hr*1.4,color="#5C4033"))
    ax.add_patch(plt.Circle((0,100),hr,color=c,zorder=3))
    ax.add_patch(plt.Rectangle((-6,100-hr-14),12,16,color=c,zorder=3))
    ax.add_patch(Ellipse((0,100-hr-10),sw*2,18,color=c,zorder=3))
    pts=np.array([[-sw,100-hr-14],[-hipw,100-hr-bh],[hipw,100-hr-bh],[sw,100-hr-14]])
    ax.add_patch(plt.Polygon(pts,closed=True,color=c,zorder=3))
    ax.add_patch(Ellipse((0,100-hr-bh*0.5),bw*2+4,bh,color=c,zorder=2))
    ax.add_patch(Ellipse((-sw-aw*0.5,100-hr-bh*0.4),aw+2,bh*0.72,angle=-6,color=c,zorder=2))
    ax.add_patch(Ellipse((sw+aw*0.5,100-hr-bh*0.4),aw+2,bh*0.72,angle=6,color=c,zorder=2))
    lt=100-hr-bh
    ax.add_patch(Ellipse((-hipw*0.38,lt-lh*0.5),lw*2,lh,angle=3,color=c,zorder=3))
    ax.add_patch(Ellipse((hipw*0.38,lt-lh*0.5),lw*2,lh,angle=-3,color=c,zorder=3))
    ax.add_patch(Ellipse((-hipw*0.38,lt-lh-6),lw*1.4,10,color=c,zorder=3))
    ax.add_patch(Ellipse((hipw*0.38,lt-lh-6),lw*1.4,10,color=c,zorder=3))
    smx=np.linspace(-8,8,20); smy=92+(smx/8)**2*5
    ax.plot([-7,-4],[102,104],"white",lw=1.5,alpha=0.8,zorder=5)
    ax.plot([4,7],[104,102],"white",lw=1.5,alpha=0.8,zorder=5)
    ax.plot(smx,smy,"white",lw=1.5,alpha=0.8,zorder=5)
    ax.strokeColor=c+"55"
    ax.add_patch(plt.Circle((0,100),hr+7,fill=False,edgecolor=c,alpha=0.4,lw=2.5))
    ax.text(0,lt-lh-28,LABEL_ID[idx],ha="center",fontsize=9,fontweight="bold",color=c)
    plt.tight_layout(pad=0); return fig

# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Input Data Antropometri")
    st.markdown("---")
    gender=st.radio("Jenis Kelamin",["Male","Female"],format_func=lambda x:"👨 Laki-laki" if x=="Male" else "👩 Perempuan")
    height=st.slider("📏 Tinggi Badan (cm)",140,200,170)
    weight=st.slider("⚖️ Berat Badan (kg)",35,165,70)

    bmi_val=weight/(height/100)**2
    bi=bmi_index(bmi_val)
    c2=CAT_COLORS[bi]
    st.markdown("---")
    st.markdown(f"### BMI: **{bmi_val:.1f}**")
    st.markdown(f'<div style="background:{c2}33;border:1px solid {c2};border-left:4px solid {c2};border-radius:12px;padding:10px;text-align:center"><span style="color:{c2};font-weight:800;font-size:1.1rem">{LABEL_ID[bi]}</span><br><span style="color:rgba(255,255,255,0.5);font-size:11px">{LABEL_MAP[bi]}</span></div>',unsafe_allow_html=True)
    il=round(18.5*(height/100)**2); ih=round(24.9*(height/100)**2)
    st.markdown(f"**Berat Ideal:** {il}–{ih} kg")
    st.markdown("---")
    predict_btn=st.button("🔍 Prediksi IMT",use_container_width=True)
    if st.button("📝 Log ke MLflow",use_container_width=True):
        try:
            import mlflow
            mlflow.set_experiment("BMI_Analyzer_v2_Skripsi")
            with mlflow.start_run():
                mlflow.log_params({"gender":gender,"height":height,"weight":weight})
                mlflow.log_metrics({"bmi":round(bmi_val,2),"category_index":bi})
            st.sidebar.success("✅ Logged ke MLflow!")
        except Exception as e:
            st.sidebar.error(str(e))

# ── HEADER ──────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.5rem 0 1rem">
    <h1 style="font-size:2.6rem;font-weight:800;
        background:linear-gradient(135deg,#6C63FF,#FF6B9D,#00D4A1);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px">
        💪 BMI Analyzer v2 — Simulasi Citra & ML
    </h1>
    <p style="color:rgba(255,255,255,0.6);font-size:0.95rem">
         Satrio Raja Nabil Rizky (23.11.5538) · Skripsi S1 Informatika · Universitas AMIKOM Yogyakarta
    </p>
</div>
""",unsafe_allow_html=True)

# ── TABS ────────────────────────────────────────────────────
t1,t2,t3,t4,t5=st.tabs(["🧮 Kalkulator BMI","📊 Analisis Data (n=3,700)","📈 Distribusi Probabilitas","🤖 Evaluasi 6 Model ML","💡 Rekomendasi"])

# ═══════════════════════════════════════════════════════════
# TAB 1 — KALKULATOR BMI
# ═══════════════════════════════════════════════════════════
with t1:
    ca,cb=st.columns([1,1.8])
    with ca:
        st.markdown("### Siluet Tubuh")
        fig_s=draw_human_silhouette(height,weight,gender,bi)
        st.pyplot(fig_s,use_container_width=True); plt.close()

        with st.expander("🔬 Lihat Citra Biner OpenCV & Fitur Morfologi", expanded=True):
            img_cv = generate_body_silhouette(height, weight, gender)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
            cnts, _ = cv2.findContours(img_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_rgb, cnts, -1, (0, 212, 161), 2)
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
        st.markdown("### Hasil Analisis IMT")
        m1,m2,m3=st.columns(3)
        m1.metric("BMI / IMT",f"{bmi_val:.1f}")
        m2.metric("Kategori (ID)",LABEL_ID[bi])
        diff=weight-il; m3.metric("Gap Ideal",f"{abs(diff):.0f} kg",delta=f"{'Lebih' if diff>0 else 'Kurang'} {abs(diff):.0f}kg",delta_color="inverse" if diff>0 else "off")

        st.markdown("---")
        bp=min(max((bmi_val-10)/50*100,0),100)
        st.markdown(f"""
        <p style="font-size:12px;color:rgba(255,255,255,0.5);margin-bottom:6px">Skala IMT</p>
        <div style="position:relative;height:24px;border-radius:12px;overflow:hidden;background:linear-gradient(to right,#3B82F6 0%,#22C55E 25%,#F59E0B 50%,#EF4444 75%,#7C3AED 100%);margin-bottom:4px">
            <div style="position:absolute;top:50%;transform:translate(-50%,-50%);width:20px;height:20px;border-radius:50%;background:white;border:3px solid rgba(0,0,0,.3);left:{bp}%;box-shadow:0 2px 8px rgba(0,0,0,.5)"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:rgba(255,255,255,0.4)">
            <span>Kurus</span><span>Normal</span><span>Gemuk</span><span>Obesitas</span><span>Parah</span>
        </div>""",unsafe_allow_html=True)
        st.markdown("---")

        if model_loaded:
            fv=build_feature_vector(gender,height,weight)
            fv_sc=scaler.transform([fv])
            pred=model.predict(fv_sc)[0]
            proba=model.predict_proba(fv_sc)[0]
            conf=max(proba)*100
            c3=CAT_COLORS[pred]
            st.markdown(f'<div style="background:{c3}22;border:1px solid {c3};border-left:4px solid {c3};border-radius:12px;padding:12px;margin-bottom:8px"><span style="color:{c3};font-weight:700;font-size:16px">✅ Prediksi Machine Learning: {LABEL_MAP[pred]} ({LABEL_ID[pred]})</span><br><span style="color:rgba(255,255,255,0.6);font-size:12px">Confidence: {conf:.1f}% · Model: Random Forest (Acc 99.73% pada n=3,700)</span></div>',unsafe_allow_html=True)
        else:
            st.warning("Pastikan model_bmi.pkl dan scaler_bmi.pkl tersedia di repositori.")

        info_df=pd.DataFrame({
            "Atribut / Fitur":["Tinggi Badan","Berat Badan","BMI / IMT","Kategori (EN)","Kategori (ID)","Rentang Berat Ideal","Pixel Area (Citra)","Aspect Ratio (Citra)","Body Density (Citra)","Contour Perimeter","Compactness"],
            "Nilai":[f"{height} cm",f"{weight} kg",f"{bmi_val:.2f} kg/m²",LABEL_MAP[bi],LABEL_ID[bi],f"{il}–{ih} kg",f"{int(f_img[0])} px²",f"{f_img[1]:.4f}",f"{f_img[2]:.4f}",f"{f_img[3]:.4f}",f"{f_img[4]:.4f}"]
        })
        st.dataframe(info_df,hide_index=True,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2 — ANALISIS DATA
# ═══════════════════════════════════════════════════════════
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
        s4.metric("Best Accuracy (RF)","99.78%")

        st.markdown("---")
        ca2,cb2=st.columns(2)
        with ca2:
            st.markdown("#### Distribusi Kategori IMT (Dataset Gabungan)")
            fig,ax=plt.subplots(figsize=(6,4)); fig.patch.set_facecolor("#0f0c29"); ax.set_facecolor("#0f0c29")
            cnt=df_data["Index"].value_counts().sort_index()
            bars=ax.bar([label_map_en[i] for i in cnt.index],cnt.values,color=[CAT_COLORS[i] for i in cnt.index],alpha=0.85,edgecolor="white",linewidth=0.5,width=0.6)
            for b,v in zip(bars,cnt.values): ax.text(b.get_x()+b.get_width()/2,b.get_height()+10,str(v),ha="center",va="bottom",color="white",fontsize=9,fontweight="bold")
            ax.set_ylabel("Jumlah Sampel",color="white"); ax.tick_params(colors="white",labelsize=7); ax.tick_params(axis="x",rotation=25)
            ax.spines[["top","right"]].set_visible(False); ax.spines[["bottom","left"]].set_color((1,1,1,0.15)); ax.set_facecolor("none")
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        with cb2:
            st.markdown("#### Distribusi Gender")
            fig,ax=plt.subplots(figsize=(4,4)); fig.patch.set_facecolor("#0f0c29")
            gc=df_data["Gender"].value_counts()
            wedges,texts,autotexts=ax.pie(gc,labels=gc.index,autopct="%1.1f%%",colors=["#6C63FF","#FF6B9D"],startangle=90,wedgeprops={"edgecolor":"white","linewidth":2})
            for t in texts: t.set_color("white")
            for a in autotexts: a.set_color("white"); a.set_fontweight("bold")
            ax.set_facecolor("none")
            st.pyplot(fig,use_container_width=True); plt.close()

        st.markdown("#### Scatter Plot: Tinggi vs Berat (Berdasarkan Kategori IMT)")
        fig,ax=plt.subplots(figsize=(10,5)); fig.patch.set_facecolor("#0f0c29"); ax.set_facecolor("#111827")
        for idx2 in sorted(df_data["Index"].unique()):
            sub=df_data[df_data["Index"]==idx2]
            ax.scatter(sub["Height"],sub["Weight"],c=CAT_COLORS[idx2],label=label_map_en[idx2],alpha=0.65,s=28,edgecolors="none")
        ax.set_xlabel("Tinggi Badan (cm)",color="white"); ax.set_ylabel("Berat Badan (kg)",color="white"); ax.tick_params(colors="white")
        ax.spines[["top","right"]].set_color("none"); ax.spines[["bottom","left"]].set_color((1,1,1,0.15))
        ax.legend(fontsize=8,labelcolor="white",facecolor="#1f2937",edgecolor=(1,1,1,0.1))
        ax.grid(color=(1,1,1,0.05),linestyle="--")
        st.pyplot(fig,use_container_width=True); plt.close()

        st.markdown("#### Preview Dataset (15 Baris Pertama)")
        st.dataframe(df_data[["Gender","Height","Weight","BMI","Label"]].head(15),hide_index=True,use_container_width=True)

    except FileNotFoundError:
        st.warning("Pastikan file bmi.csv tersedia di repositori.")

# ═══════════════════════════════════════════════════════════
# TAB 3 — DISTRIBUSI PROBABILITAS
# ═══════════════════════════════════════════════════════════
with t3:
    st.markdown("### Distribusi Probabilitas Prediksi")

    if model_loaded:
        fv2=build_feature_vector(gender,height,weight)
        fv2_sc=scaler.transform([fv2])
        proba2=model.predict_proba(fv2_sc)[0]
        pred2=model.predict(fv2_sc)[0]

        prob_full=np.zeros(6)
        for i,cls in enumerate(model.classes_): prob_full[cls]=proba2[i]

        st.markdown(f"Berdasarkan **Random Forest** (Accuracy 99.73%) untuk BMI saat ini: **{bmi_val:.1f}**")

        for i,(cat,p) in enumerate(zip(LABEL_ID.values(),prob_full)):
            pct=p*100; ind="✅ " if i==pred2 else ""
            st.markdown(f"""
            <div style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;color:white;font-size:.9rem;margin-bottom:4px">
                    <span>{ind}<b>{cat}</b> <span style="color:rgba(255,255,255,0.4);font-size:11px">({LABEL_MAP[i]})</span></span>
                    <span style="color:{CAT_COLORS[i]};font-weight:700">{pct:.1f}%</span>
                </div>
                <div style="height:10px;background:rgba(255,255,255,0.08);border-radius:5px;overflow:hidden">
                    <div style="height:100%;width:{pct}%;background:{CAT_COLORS[i]};border-radius:5px;transition:width 0.5s"></div>
                </div>
            </div>""",unsafe_allow_html=True)

        st.markdown("---")
        ca3,cb3=st.columns(2)
        with ca3:
            fig,ax=plt.subplots(figsize=(6,4)); fig.patch.set_facecolor("#0f0c29"); ax.set_facecolor("none")
            bars=ax.bar(list(LABEL_ID.values()),prob_full*100,color=CAT_COLORS,alpha=0.85,edgecolor="white",linewidth=0.5,width=0.6)
            for b,v in zip(bars,prob_full*100): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,f"{v:.1f}%",ha="center",va="bottom",color="white",fontsize=8)
            ax.set_ylabel("Probabilitas (%)",color="white"); ax.set_ylim(0,115); ax.tick_params(colors="white",labelsize=7); ax.tick_params(axis="x",rotation=20)
            ax.spines[["top","right"]].set_visible(False); ax.spines[["bottom","left"]].set_color((1,1,1,0.15)); ax.set_facecolor("none")
            ax.set_title("Distribusi Probabilitas (Bar)",color="white",fontsize=11); plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        with cb3:
            fig,ax=plt.subplots(figsize=(5,4)); fig.patch.set_facecolor("#0f0c29")
            ws,ts,ats=ax.pie(prob_full+0.001,labels=list(LABEL_ID.values()),autopct=lambda p:f"{p:.1f}%" if p>1 else "",colors=CAT_COLORS,startangle=90,wedgeprops={"edgecolor":"white","linewidth":1.5})
            for x in ts: x.set_color("white"); x.set_fontsize(7)
            for x in ats: x.set_color("white"); x.set_fontsize(7)
            ax.set_title("Distribusi Probabilitas (Pie)",color="white",fontsize=11)
            st.pyplot(fig,use_container_width=True); plt.close()

        st.dataframe(pd.DataFrame({"Kategori ID":list(LABEL_ID.values()),"Kategori EN":list(LABEL_MAP.values()),"Probabilitas (%)":[f"{p*100:.2f}%" for p in prob_full],"Prediksi":["✅ YA" if i==pred2 else "—" for i in range(6)]}),hide_index=True,use_container_width=True)
    else:
        st.warning("Model belum dimuat.")

# ═══════════════════════════════════════════════════════════
# TAB 4 — AKURASI MODEL
# ═══════════════════════════════════════════════════════════
with t4:
    st.markdown("### Perbandingan Performa 6 Algoritma Machine Learning")
    st.markdown("Hasil eksperimen pada dataset gabungan `bmi.csv` · **n = 3,700 sampel** · 3 Skenario Rasio Split · **16 Fitur** (11 Fitur Antropometri + 5 Fitur Morfologi Citra Siluet)")

    REAL_RESULTS=[
        {"Model":"Random Forest",    "Acc (80:20)":"99.73%","Acc (75:25)":"99.78%","Acc (70:30)":"99.64%","F1-Score (80:20)":"99.73%"},
        {"Model":"Gradient Boosting","Acc (80:20)":"99.59%","Acc (75:25)":"99.68%","Acc (70:30)":"99.55%","F1-Score (80:20)":"99.60%"},
        {"Model":"Decision Tree",    "Acc (80:20)":"99.59%","Acc (75:25)":"99.46%","Acc (70:30)":"99.19%","F1-Score (80:20)":"99.60%"},
        {"Model":"SVM (RBF)",        "Acc (80:20)":"97.57%","Acc (75:25)":"97.62%","Acc (70:30)":"97.39%","F1-Score (80:20)":"97.54%"},
        {"Model":"Logistic Reg.",    "Acc (80:20)":"97.03%","Acc (75:25)":"96.86%","Acc (70:30)":"96.94%","F1-Score (80:20)":"96.86%"},
        {"Model":"KNN",              "Acc (80:20)":"96.49%","Acc (75:25)":"96.65%","Acc (70:30)":"96.58%","F1-Score (80:20)":"96.44%"},
    ]
    st.dataframe(pd.DataFrame(REAL_RESULTS),hide_index=True,use_container_width=True)
    st.markdown("---")

    ca4,cb4=st.columns(2)
    with ca4:
        st.markdown("#### Perbandingan Akurasi Model (Split 80:20)")
        fig,ax=plt.subplots(figsize=(6,4)); fig.patch.set_facecolor("#0f0c29"); ax.set_facecolor("none")
        nms=["Random Forest","Gradient Boosting","Decision Tree","SVM (RBF)","Logistic Reg.","KNN"]
        acs=[99.73,99.59,99.59,97.57,97.03,96.49]
        cls_c=[("#00D4A1" if a>=99.5 else "#6C63FF" if a>=97 else "#F59E0B") for a in acs]
        brs=ax.barh(nms[::-1],acs[::-1],color=cls_c[::-1],alpha=0.85,edgecolor="white",linewidth=0.5)
        for b,v in zip(brs,acs[::-1]): ax.text(v+0.15,b.get_y()+b.get_height()/2,f"{v:.2f}%",ha="left",va="center",color="white",fontsize=9)
        ax.set_xlim(90,102); ax.set_xlabel("Accuracy (%)",color="white"); ax.tick_params(colors="white",labelsize=9)
        ax.spines[["top","right"]].set_visible(False); ax.spines[["bottom","left"]].set_color((1,1,1,0.15)); ax.set_facecolor("none")
        ax.grid(axis="x",color=(1,1,1,0.06),linestyle="--")
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

    with cb4:
        st.markdown("#### Feature Importance (Random Forest)")
        FEAT_IMP=[("BMI²",32.71),("BMI",29.36),("Weight/Height",14.11),("Weight²",5.33),("Weight",5.17),("aspect_ratio",3.77),("Weight_category",2.53),("body_density",1.70)]
        fig,ax=plt.subplots(figsize=(6,4)); fig.patch.set_facecolor("#0f0c29"); ax.set_facecolor("none")
        img_feats_names=["aspect_ratio","pixel_area","body_density","contour_perimeter","compactness"]
        fn=[f[0] for f in FEAT_IMP]; fi=[f[1] for f in FEAT_IMP]
        clrs2=[("#EF4444" if f in img_feats_names else "#6C63FF") for f in fn]
        ax.barh(fn[::-1],fi[::-1],color=clrs2[::-1],alpha=0.85,edgecolor="white",linewidth=0.5)
        ax.set_xlabel("Importance (%)",color="white"); ax.tick_params(colors="white",labelsize=8)
        ax.spines[["top","right"]].set_visible(False); ax.spines[["bottom","left"]].set_color((1,1,1,0.15)); ax.set_facecolor("none")
        ax.grid(axis="x",color=(1,1,1,0.06),linestyle="--")
        red_p=mpatches.Patch(color="#EF4444",label="Fitur Citra Siluet"); blue_p=mpatches.Patch(color="#6C63FF",label="Fitur Antropometri")
        ax.legend(handles=[red_p,blue_p],fontsize=8,labelcolor="white",facecolor="#1f2937",edgecolor=(1,1,1,0.1))
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
            fig,ax=plt.subplots(figsize=(8,5.5)); fig.patch.set_facecolor("#0f0c29")
            tnames=["Ext.Weak","Weak","Normal","Overweight","Obesity","Ext.Obesity"]
            sns.heatmap(cm_mat,annot=True,fmt="d",cmap="Purples",ax=ax,xticklabels=tnames,yticklabels=tnames,linewidths=0.5,linecolor="rgba(255,255,255,0.1)")
            ax.set_xlabel("Predicted Label",color="white"); ax.set_ylabel("Actual Label",color="white"); ax.tick_params(colors="white",labelsize=8); ax.tick_params(axis="x",rotation=25)
            ax.set_title("Confusion Matrix — Random Forest (Data Uji 20% = 740 Sampel)",color="white",fontsize=11); ax.set_facecolor("none"); fig.patch.set_facecolor("none")
            plt.tight_layout(); st.markdown("#### Confusion Matrix — Random Forest (Split 80:20)"); st.pyplot(fig,use_container_width=True); plt.close()
        except Exception as e:
            st.warning(f"Gagal memuat Confusion Matrix: {e}")

# ═══════════════════════════════════════════════════════════
# TAB 5 — REKOMENDASI
# ═══════════════════════════════════════════════════════════
with t5:
    st.markdown("### 💡 Rekomendasi Kesehatan Personal")
    rec=RECOMMENDATIONS[bi]; rc=rec["color"]
    st.markdown(f"""
    <div style="background:{rc}18;border:1px solid {rc};border-left:5px solid {rc};border-radius:18px;padding:1.5rem;margin-bottom:1.5rem">
        <h2 style="color:{rc};margin:0 0 .5rem;font-size:1.5rem">{rec["title"]}</h2>
        <p style="color:rgba(255,255,255,0.65);margin:0">
            Kategori: <b style="color:{rc}">{LABEL_ID[bi]}</b> ({LABEL_MAP[bi]}) &nbsp;·&nbsp;
            BMI: <b>{bmi_val:.1f}</b> &nbsp;·&nbsp;
            {height} cm &nbsp;·&nbsp; {weight} kg
        </p>
    </div>""",unsafe_allow_html=True)

    for em,tit,desc in rec["items"]:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-left:4px solid {rc};border-radius:16px;padding:16px;margin-bottom:12px">
            <div style="display:flex;align-items:flex-start;gap:12px">
                <span style="font-size:1.8rem">{em}</span>
                <div><p style="font-weight:700;color:white;margin:0 0 4px;font-size:1rem">{tit}</p>
                <p style="color:rgba(255,255,255,0.6);margin:0;font-size:.9rem;line-height:1.6">{desc}</p></div>
            </div>
        </div>""",unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Panduan Berat Badan Berdasarkan Tinggi Badan Anda")
    td=pd.DataFrame({"Kategori (ID)":list(LABEL_ID.values()),"Kategori (EN)":list(LABEL_MAP.values()),"BMI Range":["< 16","16–18.5","18.5–25","25–30","30–40","> 40"],"Rentang Berat (kg)":[f"{round(lo*(height/100)**2)}–{round(hi*(height/100)**2)}" for lo,hi in [(10,16),(16,18.5),(18.5,25),(25,30),(30,40),(40,55)]]})
    st.dataframe(td,hide_index=True,use_container_width=True)

# ── BATCH PREDICTION ────────────────────────────────────────
st.markdown("---")
st.markdown("### 📁 Prediksi Batch (Upload File CSV)")
st.caption("Format Kolom CSV: Gender (Male/Female), Height (cm), Weight (kg)")
uploaded_file=st.file_uploader("Upload file CSV",type=["csv"])
if uploaded_file and model_loaded:
    df_batch=pd.read_csv(uploaded_file)
    col_map = {c.lower().strip(): c for c in df_batch.columns}
    if all(k in col_map for k in ["gender", "height", "weight"]):
        df_batch = df_batch.rename(columns={col_map["gender"]:"Gender", col_map["height"]:"Height", col_map["weight"]:"Weight"})
        df_batch = df_batch.dropna(subset=["Gender","Height","Weight"])
        st.write(f"Data valid ditemukan: **{len(df_batch)} baris**")
        batch_fv=[build_feature_vector(str(r.Gender).strip(),float(r.Height),float(r.Weight)) for _,r in df_batch.iterrows()]
        batch_sc=scaler.transform(batch_fv)
        df_batch["BMI"]=(df_batch["Weight"]/(df_batch["Height"]/100)**2).round(2)
        df_batch["Prediksi_Index"]=model.predict(batch_sc)
        df_batch["Prediksi_EN"]=df_batch["Prediksi_Index"].map(LABEL_MAP)
        df_batch["Prediksi_ID"]=df_batch["Prediksi_Index"].map(LABEL_ID)
        st.dataframe(df_batch.head(30),use_container_width=True)
        st.success(f"✅ Prediksi selesai untuk {len(df_batch)} data!")
        st.download_button("⬇️ Download Hasil CSV",df_batch.to_csv(index=False).encode("utf-8"),"hasil_prediksi_bmi_skripsi.csv","text/csv")
    else:
        st.error("❌ Format kolom CSV tidak sesuai! Pastikan memiliki kolom: Gender, Height, Weight.")
elif uploaded_file:
    st.warning("Model belum dimuat.")
