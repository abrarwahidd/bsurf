# ============================================================
# preprocessing/segmentation.py
# Adaptive background removal menggunakan analisis saturasi HSV
# dan GrabCut dengan safety-net fallback.
# ============================================================

import cv2
import numpy as np
from typing import Tuple

from src.config.settings import (
    GRABCUT_MARGIN,
    GRABCUT_SAT_THRESHOLD,
    GRABCUT_MIN_FOREGROUND,
)


def remove_background(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive Background Removal.

    Strategi:
      1. Analisis saturasi HSV di tepi gambar (margin piksel).
      2. Jika avg_sat > GRABCUT_SAT_THRESHOLD → citra dianggap
         'full daging', kembalikan langsung tanpa GrabCut.
      3. Sebaliknya, jalankan GrabCut dengan safety-net:
         jika area foreground < GRABCUT_MIN_FOREGROUND% → fallback.

    Returns:
        img_bersih : ndarray BGR hasil segmentasi
        clean_mask : ndarray uint8 mask (255=foreground, 0=background)
    """
    tinggi, lebar = image.shape[:2]
    m = GRABCUT_MARGIN

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    border_pixels = np.concatenate([
        hsv[0:m, :].reshape(-1, 3),
        hsv[tinggi - m:tinggi, :].reshape(-1, 3),
        hsv[m:tinggi - m, 0:m].reshape(-1, 3),
        hsv[m:tinggi - m, lebar - m:lebar].reshape(-1, 3),
    ])
    avg_sat = np.mean(border_pixels[:, 1])

    if avg_sat > GRABCUT_SAT_THRESHOLD:
        return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

    # GrabCut
    mask     = np.zeros(image.shape[:2], np.uint8)
    bgd_mdl  = np.zeros((1, 65), np.float64)
    fgd_mdl  = np.zeros((1, 65), np.float64)
    rect     = (m, m, lebar - 2 * m, tinggi - 2 * m)

    try:
        cv2.grabCut(image, mask, rect, bgd_mdl, fgd_mdl, 5,
                    cv2.GC_INIT_WITH_RECT)
        mask_daging = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

        fg_pct = (np.count_nonzero(mask_daging) / mask_daging.size) * 100
        if fg_pct < GRABCUT_MIN_FOREGROUND:
            return image, np.ones(image.shape[:2], dtype=np.uint8) * 255

        kernel      = np.ones((5, 5), np.uint8)
        mask_daging = cv2.morphologyEx(mask_daging, cv2.MORPH_CLOSE, kernel)
        clean_mask  = mask_daging * 255
        img_bersih  = image * mask_daging[:, :, np.newaxis]
        return img_bersih, clean_mask

    except Exception:
        return image, np.ones(image.shape[:2], dtype=np.uint8) * 255
