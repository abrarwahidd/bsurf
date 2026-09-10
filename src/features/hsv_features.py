# ============================================================
# features/hsv_features.py
# Ekstraksi fitur warna 57 dimensi dari ruang warna HSV:
#   - 9D  : Color Moments (Mean, Std, Skewness) per kanal H, S, V
#   - 48D : Histogram 1D (16 bins × 3 kanal), dinormalisasi L2
# ============================================================

import cv2
import numpy as np
from scipy.stats import skew
from typing import Optional

from src.config.settings import HSV_HIST_BINS


def extract_hsv_features(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Mengekstrak 57 dimensi fitur warna dari citra BGR.

    Args:
        image : ndarray BGR (sudah di-resize dan di-segment)
        mask  : ndarray uint8 optional; nilai > 0 = piksel valid.
                Jika None, semua piksel digunakan.

    Returns:
        fitur : ndarray float32 shape (57,)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if mask is None:
        mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255
    valid = mask > 0

    # ── 9 Color Moments ────────────────────────────────────
    moments: list = []
    for ch in range(3):
        pixels = hsv[:, :, ch][valid]
        if len(pixels) > 0:
            moments.extend([
                float(np.mean(pixels)),
                float(np.std(pixels)),
                float(skew(pixels)),
            ])
        else:
            moments.extend([0.0, 0.0, 0.0])

    # ── 48 Histogram 1D (L2-normalized) ───────────────────
    ranges = [(0, 180), (0, 256), (0, 256)]
    hists  = []
    for ch, (lo, hi) in enumerate(ranges):
        h = cv2.calcHist([hsv], [ch], mask, [HSV_HIST_BINS], [lo, hi])
        cv2.normalize(h, h)
        hists.append(h.flatten())
    hist_features = np.concatenate(hists)

    return np.concatenate([moments, hist_features]).astype(np.float32)
