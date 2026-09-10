# ============================================================
# config/settings.py
# Semua konstanta dan konfigurasi global proyek.
# Import dari sini di seluruh modul — jangan hardcode di tempat lain.
# ============================================================

import numpy as np
import random

# ── Path ────────────────────────────────────────────────────
DATASET_PATH    = "dataset"
OUTPUT_IMG_DIR  = "output-img"
MODEL_DIR       = "models"
CACHE_DIR       = "cache"

# ── Reproduksibilitas ───────────────────────────────────────
RANDOM_STATE    = 42
_GLOBAL_SEED    = 42
np.random.seed(_GLOBAL_SEED)
random.seed(_GLOBAL_SEED)

# ── Dataset ─────────────────────────────────────────────────
CLASSES_NAME    = ["Segar", "SetengahSegar", "Busuk"]

# ── Preprocessing ───────────────────────────────────────────
IMG_SIZE        = (512, 512)
GRABCUT_MARGIN  = 10
GRABCUT_SAT_THRESHOLD = 55      # avg border saturation → full-meat flag
GRABCUT_MIN_FOREGROUND = 15.0   # % minimum foreground setelah GrabCut

# ── Feature Extraction ──────────────────────────────────────
SURF_HESSIAN_THRESHOLD = 300
HSV_HIST_BINS   = 16            # bins per kanal → 16 × 3 = 48 dim
NUM_HSV_FEATURES = 57           # 9 color moments + 48 histogram 1D

# ── BoVW ────────────────────────────────────────────────────
K_CLUSTERS      = 100
KMEANS_BATCH    = 2000

# ── Cross-Validation ────────────────────────────────────────
N_FOLDS         = 5
N_INNER_FOLDS   = 3

# ── Runtime ─────────────────────────────────────────────────
N_JOBS          = -1            # -1 = semua core; set 1 untuk debugging

# ── Hyperparameter Grid ─────────────────────────────────────
PARAM_GRID_BASE = {
    "svm__C":      [0.1, 1, 10, 50, 100],
    "svm__gamma":  ["scale", 0.1, 0.01, 0.001],
    "svm__kernel": ["rbf", "linear"],
}
PARAM_GRID_FUSION = {
    **PARAM_GRID_BASE,
    "weighter__hsv_weight": [0.1, 0.5, 1.0, 1.5, 2.0, 3.0],
}
