import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from scipy.stats import skew
from typing import Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# ─────────────────────────────────────────────────────────────
# 1. KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bsurf - Sistem Uji Daging Sapi",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# 2. CUSTOM CSS — desain bersih, modern, merah-daging
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background halaman */
.stApp { background-color: #0f1117; color: #e8eaed; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 40%, #b91c1c 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "🥩";
    font-size: 160px;
    position: absolute;
    right: -10px;
    top: -20px;
    opacity: 0.12;
    line-height: 1;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #fff;
    margin: 0 0 6px 0;
    line-height: 1.25;
}
.hero-sub {
    font-size: 0.9rem;
    color: #fca5a5;
    margin: 0;
    letter-spacing: 0.02em;
}

/* Kartu info pipeline */
.pipeline-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    color: #c9d1d9;
}
.pipeline-step:last-child { margin-bottom: 0; }
.step-badge {
    background: #b91c1c;
    color: #fff;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    white-space: nowrap;
    min-width: 60px;
    text-align: center;
}

/* Upload area */
div[data-testid="stFileUploader"] > label { display: none; }
div[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #30363d !important;
    border-radius: 12px !important;
    background: #161b22 !important;
    transition: border-color 0.2s;
}
div[data-testid="stFileUploadDropzone"]:hover {
    border-color: #b91c1c !important;
}

/* Tombol analisis */
div.stButton > button {
    background: linear-gradient(90deg, #991b1b, #b91c1c) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 14px 0 !important;
    width: 100%;
    letter-spacing: 0.02em;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.88; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
}
div[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #e8eaed !important;
}

/* Result box */
.result-box {
    border-radius: 12px;
    padding: 20px 24px;
    margin: 20px 0;
    border-left: 5px solid;
}
.result-segar   { background:#052e16; border-color:#16a34a; color:#bbf7d0; }
.result-setengah { background:#431407; border-color:#ea580c; color:#fed7aa; }
.result-busuk   { background:#450a0a; border-color:#dc2626; color:#fecaca; }

/* Label teknis di expander */
.tech-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 6px;
}

/* Divider */
hr { border-color: #30363d !important; }

/* Caption */
figcaption, .stImage > div > span { color: #8b949e !important; font-size: 0.8rem !important; }

/* Expander */
details { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 3. KELAS & FUNGSI INTI
# ─────────────────────────────────────────────────────────────
class FeatureWeighter(BaseEstimator, TransformerMixin):
    def __init__(self, hsv_weight: float = 1.0, num_hsv_features: int = 57):
        self.hsv_weight = hsv_weight
        self.num_hsv_features = num_hsv_features
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_w = X.copy()
        X_w[:, :self.num_hsv_features] *= self.hsv_weight
        return X_w


@st.cache_resource
def load_models():
    try:
        model_fusi  = joblib.load('models/fusi_model.pkl')
        model_kmeans = joblib.load('models/kmeans_vocab.pkl')
        return model_fusi, model_kmeans
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None


def remove_background(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tinggi, lebar = image.shape[:2]
    margin = 10
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    top    = hsv[0:margin, :]
    bottom = hsv[tinggi-margin:tinggi, :]
    left   = hsv[margin:tinggi-margin, 0:margin]
    right  = hsv[margin:tinggi-margin, lebar-margin:lebar]
    border_pixels = np.concatenate([top.reshape(-1,3), bottom.reshape(-1,3),
                                    left.reshape(-1,3), right.reshape(-1,3)])
    avg_sat = np.mean(border_pixels[:, 1])
    if avg_sat > 55:
        return image, np.ones(image.shape[:2], dtype=np.uint8) * 255
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd  = np.zeros((1,65), np.float64)
    fgd  = np.zeros((1,65), np.float64)
    rect = (margin, margin, lebar-2*margin, tinggi-2*margin)
    try:
        cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask_d = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        if (np.count_nonzero(mask_d) / mask_d.size) * 100 < 15:
            return image, np.ones(image.shape[:2], dtype=np.uint8) * 255
        kernel = np.ones((5,5), np.uint8)
        mask_d = cv2.morphologyEx(mask_d, cv2.MORPH_CLOSE, kernel)
        return image * mask_d[:,:,np.newaxis], mask_d * 255
    except:
        return image, np.ones(image.shape[:2], dtype=np.uint8) * 255


def extract_hsv_features(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if mask is None: mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255
    valid = mask > 0
    moments = []
    for i in range(3):
        px = hsv[:,:,i][valid]
        if len(px) > 0:
            moments.extend([np.mean(px), np.std(px), skew(px)])
        else:
            moments.extend([0.0, 0.0, 0.0])
    h = cv2.calcHist([hsv],[0],mask,[16],[0,180])
    s = cv2.calcHist([hsv],[1],mask,[16],[0,256])
    v = cv2.calcHist([hsv],[2],mask,[16],[0,256])
    cv2.normalize(h,h); cv2.normalize(s,s); cv2.normalize(v,v)
    return np.concatenate([moments, h.flatten(), s.flatten(), v.flatten()])


def extract_surf_descriptors(image: np.ndarray):
    surf = cv2.xfeatures2d.SURF_create(300)
    kp, desc = surf.detectAndCompute(image, None)
    return kp, desc


def create_bovw_histogram(descriptor: Optional[np.ndarray], kmeans_model) -> np.ndarray:
    if descriptor is not None:
        words = kmeans_model.predict(descriptor)
        hist, _ = np.histogram(words, bins=np.arange(kmeans_model.n_clusters+1), density=True)
        return hist
    return np.zeros(kmeans_model.n_clusters)


# ─────────────────────────────────────────────────────────────
# 4. FUNGSI GRAFIK HASIL
# ─────────────────────────────────────────────────────────────
def build_proba_chart(probas: np.ndarray, pred_idx: int) -> plt.Figure:
    """Bar chart probabilitas per kelas dengan desain gelap."""
    labels = ['Segar', 'Setengah\nSegar', 'Busuk']
    colors = ['#16a34a', '#ea580c', '#dc2626']
    edge   = ['#4ade80', '#fb923c', '#f87171']
    alpha  = [1.0 if i == pred_idx else 0.35 for i in range(3)]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    bars = ax.bar(labels, probas * 100, color=colors, edgecolor=edge,
                  linewidth=1.4, width=0.55)
    for bar, a in zip(bars, alpha):
        bar.set_alpha(a)

    for bar, val in zip(bars, probas * 100):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1.5,
                f"{val:.1f}%",
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#e8eaed')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Probabilitas (%)', color='#8b949e', fontsize=9)
    ax.set_title('Distribusi Keyakinan Model', color='#e8eaed',
                 fontsize=11, fontweight='bold', pad=12)
    ax.tick_params(colors='#8b949e', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.yaxis.label.set_color('#8b949e')
    ax.set_facecolor('#161b22')
    fig.tight_layout()
    return fig


def build_hsv_chart(hsv_features: np.ndarray) -> plt.Figure:
    """Plot histogram H, S, V dari fitur yang sudah diekstraksi."""
    # Indeks 9 s.d. 56 = 3 histogram x 16 bins (setelah 9 momen warna)
    hist_h = hsv_features[9:25]
    hist_s = hsv_features[25:41]
    hist_v = hsv_features[41:57]
    bins   = np.arange(16)

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    fig.patch.set_facecolor('#161b22')
    configs = [
        (hist_h, '#f97316', 'Hue'),
        (hist_s, '#22c55e', 'Saturation'),
        (hist_v, '#60a5fa', 'Value'),
    ]
    for ax, (data, color, label) in zip(axes, configs):
        ax.set_facecolor('#0f1117')
        ax.bar(bins, data, color=color, alpha=0.75, width=0.8, edgecolor=color)
        ax.plot(bins, data, color=color, linewidth=1.5, marker='o', markersize=3)
        ax.set_title(label, color='#e8eaed', fontsize=9, fontweight='600')
        ax.tick_params(colors='#8b949e', labelsize=7)
        for spine in ax.spines.values(): spine.set_edgecolor('#30363d')
        ax.set_xlabel('Bins (16)', color='#8b949e', fontsize=7)

    fig.suptitle('Histogram Fitur Warna HSV (16 Bins per Kanal)',
                 color='#c9d1d9', fontsize=10, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


def build_bovw_chart(bovw_hist: np.ndarray) -> plt.Figure:
    """Distribusi Visual Word BoVW."""
    k = len(bovw_hist)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#0f1117')
    ax.fill_between(range(k), bovw_hist, color='#a78bfa', alpha=0.5)
    ax.plot(range(k), bovw_hist, color='#a78bfa', linewidth=1.5)
    ax.set_title('Distribusi Visual Words (BoVW — SURF Codebook)',
                 color='#e8eaed', fontsize=10, fontweight='bold')
    ax.set_xlabel(f'Visual Word (0–{k-1})', color='#8b949e', fontsize=8)
    ax.set_ylabel('Frekuensi', color='#8b949e', fontsize=8)
    ax.tick_params(colors='#8b949e', labelsize=7)
    for spine in ax.spines.values(): spine.set_edgecolor('#30363d')
    fig.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────
# 5. SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 4px 0;">
        <span style="font-size:2.4rem;">🥩</span>
        <h2 style="font-family:'Space Grotesk',sans-serif;
                   color:#e8eaed; margin:8px 0 2px 0; font-size:1.2rem;">
            Bsurf -  SURF-BoVW
        </h2>
        <p style="color:#8b949e; font-size:0.78rem; margin:0;">
            Sistem Klasifikasi Kesegaran Daging Sapi
        </p>
    </div>
    <hr style="border-color:#30363d; margin:16px 0;">
    """, unsafe_allow_html=True)

    st.markdown("**Pipeline Sistem:**")
    st.markdown("""
    <div class="pipeline-card">
        <div class="pipeline-step">
            <span class="step-badge">GrabCut</span>
            <span>Segmentasi adaptif & isolasi ROI daging</span>
        </div>
        <div class="pipeline-step">
            <span class="step-badge">HSV</span>
            <span>9 momen warna + 48 histogram (57 fitur)</span>
        </div>
        <div class="pipeline-step">
            <span class="step-badge">SURF</span>
            <span>Deskriptor tekstur multi-skala</span>
        </div>
        <div class="pipeline-step">
            <span class="step-badge">BoVW</span>
            <span>Kamus visual K-Means (K=100)</span>
        </div>
        <div class="pipeline-step">
            <span class="step-badge">SVM</span>
            <span>Klasifikasi fusi fitur warna + tekstur</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Kelas Prediksi:**")
    for icon, label, color in [
        ("🟢", "Segar", "#16a34a"),
        ("🟡", "Setengah Segar", "#ea580c"),
        ("🔴", "Busuk", "#dc2626"),
    ]:
        st.markdown(
            f"<span style='color:{color}; font-weight:600;'>{icon} {label}</span>",
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color:#30363d; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.75rem; color:#8b949e; text-align:center;'>"
        "Dikembangkan oleh <b style='color:#c9d1d9;'>Abrar Wahid</b><br>"
        "Universitas Majalengka · 2026</p>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# 6. AREA UTAMA
# ─────────────────────────────────────────────────────────────
model_fusi, model_kmeans = load_models()

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">Sistem Uji Prediksi Kesegaran Daging Sapi</p>
    <p class="hero-sub">FUSI FITUR · SURF-BoVW + MOMEN WARNA HSV · SUPPORT VECTOR MACHINE</p>
</div>
""", unsafe_allow_html=True)

# Layout: upload kiri, hasil kanan
col_upload, col_gap, col_result = st.columns([1.1, 0.08, 1.6])

with col_upload:
    st.markdown("##### Unggah Citra Daging")
    uploaded_file = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png", "webp"],
        help="Foto fokus pada permukaan daging sapi. Format: JPG, PNG, WEBP."
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        img_array = np.array(pil_image)
        if img_array.ndim == 3 and img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        st.image(pil_image, caption="Citra Input", use_column_width=True)
        st.markdown(
            f"<p style='font-size:0.8rem;color:#8b949e;'>"
            f"📐 {pil_image.width} × {pil_image.height} px · "
            f"{uploaded_file.type}</p>",
            unsafe_allow_html=True
        )
        run_btn = st.button("🔬 Analisis Kesegaran Daging", use_container_width=True,
                            disabled=(model_fusi is None))
    else:
        st.markdown("""
        <div style="border:2px dashed #30363d; border-radius:12px;
                    padding:48px 20px; text-align:center; color:#8b949e;">
            <div style="font-size:2.5rem;">📷</div>
            <p style="margin:8px 0 0 0; font-size:0.9rem;">
                Seret & lepas gambar di sini,<br>atau klik untuk memilih file
            </p>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

with col_result:
    if uploaded_file is None:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:center;
                    height:380px; border:1px solid #30363d; border-radius:14px;
                    color:#4b5563; font-size:0.95rem; text-align:center;">
            <div>
                <div style="font-size:2.8rem; margin-bottom:10px;">🔬</div>
                Unggah gambar daging terlebih dahulu,<br>
                lalu klik tombol analisis
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded_file is not None and not run_btn:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:center;
                    height:380px; border:1px solid #30363d; border-radius:14px;
                    color:#4b5563; font-size:0.95rem; text-align:center;">
            <div>
                <div style="font-size:2.8rem; margin-bottom:10px;">⬅️</div>
                Gambar siap diproses.<br>
                Tekan <b style="color:#c9d1d9;">Analisis Kesegaran Daging</b> untuk mulai.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None and run_btn and model_fusi is not None:
        with st.spinner("⚙️ Memproses citra — GrabCut → HSV → SURF → SVM …"):

            img_resized = cv2.resize(img_bgr, (512, 512))
            img_cleaned, mask = remove_background(img_resized)

            fitur_hsv   = extract_hsv_features(img_cleaned, mask)
            keypoints, desc_surf = extract_surf_descriptors(img_cleaned)
            fitur_bovw  = create_bovw_histogram(desc_surf, model_kmeans)

            fitur_gabungan = np.concatenate([fitur_hsv, fitur_bovw]).reshape(1, -1)
            pred_idx   = model_fusi.predict(fitur_gabungan)[0]
            probas     = model_fusi.predict_proba(fitur_gabungan)[0]
            confidence = probas[pred_idx] * 100

        # ── Tampilkan Hasil ──────────────────────────────────
        KELAS  = ['Segar', 'Setengah Segar', 'Busuk']
        CSS_CL = ['result-segar', 'result-setengah', 'result-busuk']
        IKON   = ['🥩', '⚠️', '🦠']

        hasil_kelas = KELAS[pred_idx]
        ikon        = IKON[pred_idx]
        css_cl      = CSS_CL[pred_idx]

        st.markdown(f"""
        <div class="result-box {css_cl}">
            <div style="font-size:1.7rem; font-weight:700; margin-bottom:4px;">
                {ikon} {hasil_kelas}
            </div>
            <div style="font-size:0.88rem; opacity:0.85; line-height:1.5;">
                Fusi fitur HSV + SURF-BoVW memprediksi citra ini sebagai
                <b>Daging {hasil_kelas}</b> dengan keyakinan
                <b>{confidence:.2f}%</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrik 2 kolom
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Prediksi", f"{ikon} {hasil_kelas}")
        with m2:
            st.metric("Keyakinan", f"{confidence:.2f}%")
        with m3:
            st.metric("Titik SURF", f"{len(keypoints)}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── TAB: Grafik + Bukti Teknis ───────────────────────
        tab1, tab2, tab3 = st.tabs(["📊 Grafik Hasil", "🎨 Fitur Warna", "🔬 Bukti Segmentasi"])

        with tab1:
            st.markdown("#### Distribusi Keyakinan Model")
            fig_proba = build_proba_chart(probas, pred_idx)
            st.image(fig_to_bytes(fig_proba), use_column_width=True)
            plt.close(fig_proba)

            st.markdown("#### Distribusi Visual Words (BoVW)")
            fig_bovw = build_bovw_chart(fitur_bovw)
            st.image(fig_to_bytes(fig_bovw), use_column_width=True)
            plt.close(fig_bovw)

        with tab2:
            st.markdown("#### Histogram Fitur Warna HSV")
            fig_hsv = build_hsv_chart(fitur_hsv)
            st.image(fig_to_bytes(fig_hsv), use_column_width=True)
            plt.close(fig_hsv)

            # Tabel momen warna
            st.markdown("#### Momen Warna (Mean · Std · Skewness)")
            kanal_nama = ['Hue', 'Saturation', 'Value']
            rows = []
            for i, nm in enumerate(kanal_nama):
                m, s, sk = fitur_hsv[i*3], fitur_hsv[i*3+1], fitur_hsv[i*3+2]
                rows.append({"Kanal": nm, "Mean": f"{m:.4f}",
                              "Std Dev": f"{s:.4f}", "Skewness": f"{sk:.4f}"})
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        with tab3:
            st.markdown("#### Proses Segmentasi & Deteksi Tekstur")
            img_cleaned_rgb = cv2.cvtColor(img_cleaned, cv2.COLOR_BGR2RGB)
            img_orig_rgb    = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            b1, b2 = st.columns(2)
            with b1:
                st.markdown('<p class="tech-label">1 · Input (512×512)</p>',
                            unsafe_allow_html=True)
                st.image(img_orig_rgb, use_column_width=True)

            with b2:
                st.markdown('<p class="tech-label">2 · Hasil GrabCut (ROI)</p>',
                            unsafe_allow_html=True)
                st.image(img_cleaned_rgb, use_column_width=True)

            if keypoints:
                img_kp = cv2.drawKeypoints(
                    img_cleaned_rgb, keypoints, None,
                    color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
                )
                st.markdown(
                    f'<p class="tech-label">3 · Titik Tekstur SURF '
                    f'({len(keypoints)} keypoints terdeteksi)</p>',
                    unsafe_allow_html=True
                )
                st.image(img_kp, use_column_width=True)
            else:
                st.info("Tidak ada titik tekstur SURF yang terdeteksi pada citra ini.")


# ─────────────────────────────────────────────────────────────
# 7. FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; font-size:0.78rem; color:#4b5563;">
    Sistem Pakar Daging Sapi · Fusi Fitur SURF-BoVW & Momen Warna HSV · SVM Classification<br>
    Dikembangkan oleh <b style="color:#8b949e;">Abrar Wahid</b> · Universitas Majalengka
</p>
""", unsafe_allow_html=True)