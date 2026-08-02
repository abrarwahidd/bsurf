# Sistem Klasifikasi Kesegaran Daging Sapi

> **Beef Inspection & Meat Analysis** — Sistem klasifikasi kesegaran daging sapi berbasis Computer Vision menggunakan Fusi Fitur SURF-BoVW & Momen Warna HSV dengan Support Vector Machine.

**Penulis:** Abrar Wahid  
**Institusi:** Program Studi Informatika, Fakultas Teknik, Universitas Majalengka  
**Tahun:** 2026

---

## Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Hasil Penelitian](#-hasil-penelitian)
- [Struktur Direktori](#-struktur-direktori)
- [Persyaratan Sistem](#-persyaratan-sistem)
- [Instalasi](#-instalasi)
- [Dataset](#-dataset)
- [Penggunaan](#-penggunaan)
- [Pipeline Teknis](#-pipeline-teknis)
- [Evaluasi Model](#-evaluasi-model)
- [Referensi](#-referensi)

---

## Tentang Proyek

Inspeksi visual kesegaran daging sapi secara manual di industri pangan sangat rentan terhadap subjektivitas dan kesulitan mendeteksi **fase transisi pembusukan** secara akurat. Berdasarkan Laporan Tahunan BPOM RI (2024), agen mikrobiologi mendominasi penyebab KLB Keracunan Pangan di Indonesia dengan total 98 kejadian.

Penelitian ini mengusulkan sistem klasifikasi otomatis berbasis **Computer Vision** yang mengintegrasikan dua sumber informasi visual secara komplementer:

| Fitur | Algoritma | Dimensi | Keunggulan |
|-------|-----------|---------|-----------|
| Tekstur Lokal | SURF → Bag of Visual Words | 100D | Tahan rotasi & skala |
| Warna Global | HSV Color Moments + Histogram 1D | 57D | Tahan variasi iluminasi |
| **Fusi Hibrida** | **Early Feature Fusion** | **157D** | **Komplementer & diskriminatif** |

**Kelas Klasifikasi:** 🟢 Segar · 🟡 Setengah Segar · 🔴 Busuk

---

## Arsitektur Sistem

```
Citra Input (JPG/PNG)
        │
        ▼
┌─────────────────────────────┐
│    Pra-pemrosesan Citra      │
│  • Resize → 512×512 px      │
│  • Gaussian Blur (7×7)      │
│  • Adaptive Background      │
│    Removal (HSV Saturation  │
│    + GrabCut + Morphology)  │
└──────────────┬──────────────┘
               │ ROI Daging
       ┌───────┴───────┐
       ▼               ▼
┌────────────┐  ┌──────────────────────┐
│  Ekstraksi │  │    Ekstraksi Tekstur  │
│   Warna    │  │                      │
│            │  │  SURF Keypoints      │
│ HSV Color  │  │  (Hessian Threshold  │
│  Moments   │  │   = 300)             │
│ (Mean/Std/ │  │        │             │
│  Skewness) │  │  MiniBatch K-Means   │
│ + Hist 1D  │  │  Clustering (K=100)  │
│ (16 bins   │  │        │             │
│  per kanal)│  │  Bag of Visual Words │
│            │  │  Histogram           │
│   57 Dim   │  │   100 Dim            │
└─────┬──────┘  └──────────┬───────────┘
      │                    │
      └─────────┬──────────┘
                ▼
     Konkatenasi → 157 Dimensi
                │
                ▼
┌───────────────────────────────────┐
│       Scikit-Learn Pipeline        │
│                                   │
│  [1] StandardScaler               │
│      (Z-score normalization)      │
│                                   │
│  [2] FeatureWeighter              │
│      (hsv_weight × 57 Dim HSV)    │
│      → mengatasi Curse of         │
│        Dimensionality             │
│                                   │
│  [3] SVM (Kernel RBF)             │
│      + class_weight='balanced'    │
│      + GridSearchCV (C, γ, w)     │
└───────────────┬───────────────────┘
                │
                ▼
     Prediksi Kelas Kesegaran
  🟢 Segar | 🟡 Setengah Segar | 🔴 Busuk
```

---

## 📊 Hasil Penelitian

### Ablation Study — Perbandingan Skenario Klasifikasi

| Skenario Pemodelan | Akurasi | Presisi (Macro) | Recall (Macro) | F1-Score (Macro) |
|--------------------|---------|-----------------|----------------|------------------|
| Skenario 1 — Tekstur SURF saja | 65.25% | 0.66 | 0.65 | 0.65 |
| Skenario 2 — Warna HSV saja | 84.50% | 0.85 | 0.85 | 0.84 |
| **Skenario 3 — Fusi Fitur (Diusulkan)** | **89.00%** | **0.89** | **0.89** | **0.89** |

### Laporan Klasifikasi Model Fusi (Per Kelas)

| Kelas | Precision | Recall | F1-Score | Jumlah Sampel |
|-------|-----------|--------|----------|---------------|
| 🟢 Segar | 0.92 | 0.88 | 0.90 | 138 |
| 🟡 Setengah Segar | 0.83 | 0.85 | 0.84 | 130 |
| 🔴 Busuk | 0.92 | 0.95 | 0.93 | 132 |
| **Macro Average** | **0.89** | **0.89** | **0.89** | **400** |

### Parameter Optimal (GridSearchCV)

- **Kernel SVM:** Radial Basis Function (RBF)
- **hsv_weight (Feature Balancing):** 3.0
- **Validasi:** Stratified 5-Fold Cross Validation
- **Dataset:** 400 citra (138 Segar · 130 Setengah Segar · 132 Busuk)

---

## 📁 Struktur Direktori

```
bsurf/
│
├── app.py                      # Aplikasi web Streamlit (Interface)
├── surf_final.ipynb            # Notebook pelatihan & evaluasi model
├── README.md                   # Dokumentasi proyek ini
│
├── models/                     # Model terlatih (dibuat setelah training)
│   ├── fusi_model.pkl          # Pipeline SVM + StandardScaler + FeatureWeighter
│   └── kmeans_vocab.pkl        # Visual Vocabulary MiniBatch K-Means (K=100)
│
└── dataset/                    # Dataset citra daging sapi
    ├── Segar/                  # 138 citra daging segar
    ├── SetengahSegar/          # 130 citra daging setengah segar
    └── Busuk/                  # 132 citra daging busuk
```

> **Catatan:** Folder `models/` akan otomatis terbuat setelah menjalankan notebook `surf_final.ipynb` hingga selesai.

---

## ⚙️ Persyaratan Sistem

### Perangkat Keras (Rekomendasi)
- **Processor:** Intel Core i5/i7 generasi ke-10 ke atas
- **RAM:** Minimum 8 GB (16 GB+ direkomendasikan untuk training)
- **Storage:** Minimum 2 GB ruang kosong

### Perangkat Lunak
- **OS:** Windows 10/11, Ubuntu 20.04+, atau macOS 12+
- **Python:** 3.7.x (wajib — SURF hanya tersedia di OpenCV versi lama)

---

## 🚀 Instalasi


Pastikan Anda sudah menginstal [Miniconda](https://docs.anaconda.com/miniconda/) atau Anaconda di komputer Anda sebelum memulai langkah-langkah berikut.

### 1. Clone Repositori

```bash
git clone https://github.com/abrarwahidd/bsurf.git
cd bsurf
```

### 2. Buat Virtual Environment

```bash
# Membuat environment baru bernama "surf_env" dengan Python 3.8
conda create -n surf_env python=3.7 -y

# Mengaktifkan environment
conda activate surf_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **⚠️ PENTING — Versi OpenCV:**
> Algoritma SURF dipatenkan dan hanya tersedia di `opencv-contrib-python` versi `3.4.2.16`. Versi lebih baru (4.x) **tidak** menyertakan SURF. Pastikan menggunakan versi yang tepat.

Verifikasi instalasi SURF:
```python
import cv2
surf = cv2.xfeatures2d.SURF_create(300)
print("SURF tersedia dan siap digunakan!")
```

### 4. Verifikasi Instalasi

```bash
python -c "import cv2; print(cv2.__version__)"
# Output yang diharapkan: 3.4.2
```

---

## 📦 Dataset

Dataset menggunakan citra daging sapi sekunder dari repositori publik:

| Sumber | Nama Dataset | Publisher |
|--------|-------------|-----------|
| [Roboflow](https://roboflow.com) | *meat freshness Computer Vision Model* | University of Southeastern Philippines |
| [Kaggle](https://kaggle.com) | *Meat Freshness Image Dataset* | Vinayak Shanawad |

### Struktur Dataset

```
dataset/
├── Segar/         → 138 citra (.jpg) — daging merah cerah, oxymyoglobin dominan
├── SetengahSegar/ → 130 citra (.jpg) — transisi oksidasi, warna mulai berubah
└── Busuk/         → 132 citra (.jpg) — metmyoglobin dominan, kecokelatan/kehitaman
```

Letakkan folder `dataset/` di direktori yang sama dengan file notebook (`surf_final.ipynb`).

---

## 💻 Penggunaan

### A. Melatih Model (Training)

Jalankan seluruh sel notebook secara berurutan:

```bash
jupyter notebook surf_final.ipynb
```

Notebook akan melakukan:
1. Verifikasi ketersediaan SURF
2. Pembangunan Kamus Visual (MiniBatch K-Means, K=100)
3. Ekstraksi fitur HSV (57D) dan SURF-BoVW (100D) untuk seluruh dataset
4. Training Scikit-Learn Pipeline dengan GridSearchCV
5. Evaluasi Ablation Study (3 skenario)
6. Menyimpan model ke `models/fusi_model.pkl` dan `models/kmeans_vocab.pkl`

> Proses training memerlukan waktu 15–45 menit tergantung spesifikasi hardware.

### B. Menjalankan Aplikasi Web (Interface)

Setelah model tersedia di folder `models/`:

```bash
streamlit run app.py
```

Buka browser dan akses `http://localhost:8501`

**Cara penggunaan antarmuka:**
1. Unggah foto daging sapi (JPG, PNG, atau WEBP) melalui tombol upload
2. Klik **🔬 Analisis Kesegaran Daging**
3. Sistem akan menampilkan:
   - Prediksi kelas (Segar / Setengah Segar / Busuk)
   - Persentase keyakinan model
   - Jumlah titik SURF yang terdeteksi
   - Grafik distribusi probabilitas kelas
   - Histogram Visual Words (BoVW)
   - Histogram fitur warna HSV per kanal
   - Tabel Momen Warna (Mean, Std, Skewness)
   - Visualisasi proses segmentasi GrabCut & deteksi keypoints

---

## 🔧 Pipeline Teknis

### Pra-pemrosesan — Adaptive Background Removal

```python
def remove_background(image):
    # 1. Analisis saturasi rata-rata piksel tepi (10px border)
    # 2. Jika avg_sat > 55 → gambar full daging, skip GrabCut
    # 3. Jika ada background → jalankan GrabCut (5 iterasi)
    # 4. Safety Net: jika ROI < 15% frame → kembalikan asli
    # 5. Morphological Closing (5×5 kernel) untuk menutup lubang
```

### Ekstraksi Fitur Warna HSV (57 Dimensi)

```
Kanal H, S, V masing-masing:
  • Mean     → distribusi intensitas rata-rata
  • Std Dev  → dispersi / variansi warna
  • Skewness → kecondongan distribusi warna
= 3 kanal × 3 momen = 9 dimensi

Histogram 1D per kanal (16 bins, dinormalisasi):
= 3 kanal × 16 bins = 48 dimensi

Total: 9 + 48 = 57 dimensi
```

### Ekstraksi Fitur Tekstur SURF-BoVW (100 Dimensi)

```
1. SURF Keypoint Detection (Hessian Threshold = 300)
   → Setiap keypoint: deskriptor 64 dimensi

2. MiniBatch K-Means Clustering (K=100)
   → Membangun kamus visual dari seluruh deskriptor latih

3. Bag of Visual Words Histogram
   → Frekuensi kemunculan setiap visual word
   → Output: 100 dimensi per citra
```

### Scikit-Learn Pipeline

```python
Pipeline([
    ('scaler', StandardScaler()),          # Z-score normalization
    ('weighter', FeatureWeighter(          # Feature Balancing
        hsv_weight=3.0,                    # Parameter optimal
        num_hsv_features=57
    )),
    ('svm', SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True
    ))
])
```

### Hyperparameter Tuning (GridSearchCV)

```python
param_grid = {
    'svm__C':          [0.1, 1, 10, 100],
    'svm__gamma':      ['scale', 'auto', 0.001, 0.01, 0.1],
    'svm__kernel':     ['linear', 'rbf'],
    'weighter__hsv_weight': np.arange(0.1, 3.1, 0.5)
}
# Evaluasi internal: Stratified 5-Fold Cross Validation
```

---

## 📈 Evaluasi Model

### Confusion Matrix — Model Fusi

```
                 Prediksi
              Segar  ½Segar  Busuk
Aktual Segar  [ 121    16      1  ]   Precision: 0.92
       ½Segar [  10   110     10  ]   Precision: 0.83
       Busuk  [   0     7    125  ]   Precision: 0.92
```

### Analisis Pengaruh hsv_weight

| hsv_weight | Akurasi Validasi |
|------------|-----------------|
| 0.1 | 59.2% |
| 0.5 | 80.5% |
| 1.0 | 84.7% |
| 1.5 | 87.2% |
| 2.0 | 86.5% |
| **3.0** | **89.0% ✓ Optimal** |

Parameter `hsv_weight = 3.0` mengamplifikasi 57 dimensi fitur warna untuk menyeimbangkan dominasi 100 dimensi fitur tekstur, mengatasi **Curse of Dimensionality**.

### Catatan Ambiguitas Fase Transisi

Kelas "Setengah Segar" mencatat Recall terendah (0.85) karena:
- Oksidasi mikrobiologi terjadi **heterogen** (tidak merata pada permukaan daging)
- Sebagian sampel memiliki tepi jaringan yang sudah menghitam namun area tengah masih segar
- Ini adalah **ambiguitas fisik benda organik**, bukan kelemahan arsitektur algoritma

---

## 📚 Referensi

- Bay, H., Tuytelaars, T., and Gool, L.V. (2008). *SURF: Speeded Up Robust Features.*
- Arsalane, A., Klilou, A., and El Barbri, N. (2024). *Performance evaluation of machine learning algorithms for meat freshness assessment.* International Journal of Electrical and Computer Engineering, 14(5).
- Kapoor, S. and Narayanan, A. (2023). *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns, 4(9).
- Shehzad, K., Ali, U., and Munir, A. (2025). *Computer Vision for Food Quality Assessment: Advances and Challenges.* Global Journal of Machine Learning and Computing, 1(1).
- Laporan Tahunan Badan Pengawas Obat dan Makanan (BPOM RI). (2024).

---

## 📄 Lisensi

Proyek ini dikembangkan sebagai bagian dari penelitian Tugas Akhir pada Program Studi Informatika, Fakultas Teknik, Universitas Majalengka. Penggunaan kode untuk keperluan akademis dan non-komersial diperbolehkan dengan mencantumkan atribusi yang sesuai.

---

<p align="center">
  Dikembangkan oleh <b>Abrar Wahid</b> · Universitas Majalengka · 2026
</p>