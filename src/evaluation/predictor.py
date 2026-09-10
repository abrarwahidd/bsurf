# ============================================================
# evaluation/predictor.py
# Inferensi satu citra menggunakan model final dari model_trainer.
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional, Tuple
from sklearn.cluster import MiniBatchKMeans

from src.config.settings import IMG_SIZE, CLASSES_NAME, SURF_HESSIAN_THRESHOLD
from src.preprocessing.segmentation import remove_background
from src.features.hsv_features import extract_hsv_features
from src.features.surf_bovw import extract_surf_descriptors, create_bovw_histogram
from src.utils.io_helpers import save_current_figure, make_safe_filename


def predict_image(
    image_path: str,
    pipeline: Any,
    vocab: MiniBatchKMeans,
    image_size: Tuple[int, int] = IMG_SIZE,
    show_plot: bool = True,
) -> Tuple[str, float]:
    """
    Prediksi kesegaran daging sapi dari satu citra.

    Parameters
    ----------
    image_path : path ke file citra
    pipeline   : Pipeline sklearn (dari train_final_model atau load_models)
    vocab      : MiniBatchKMeans vocab (konsisten dengan pipeline)
    image_size : ukuran resize
    show_plot  : tampilkan visualisasi matplotlib

    Returns
    -------
    label      : string kelas prediksi
    confidence : float persen keyakinan
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    # ── Preprocessing & Feature Extraction ──────────────────
    img_resized         = cv2.resize(img, image_size)
    img_clean, mask     = remove_background(img_resized)
    fitur_hsv           = extract_hsv_features(img_clean, mask)
    desc_surf           = extract_surf_descriptors(img_clean)
    fitur_bovw          = create_bovw_histogram(desc_surf, vocab)
    fitur_gabungan      = np.concatenate([fitur_hsv, fitur_bovw]).reshape(1, -1)

    # ── Keypoints SURF untuk visualisasi ────────────────────
    surf                = cv2.xfeatures2d.SURF_create(SURF_HESSIAN_THRESHOLD)
    keypoints, _        = surf.detectAndCompute(img_clean, None)
    img_kp              = cv2.drawKeypoints(
        img_clean, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
    )

    # ── Prediksi ─────────────────────────────────────────────
    idx         = pipeline.predict(fitur_gabungan)[0]
    confidence  = pipeline.predict_proba(fitur_gabungan)[0][idx] * 100
    label       = CLASSES_NAME[idx]
    label_clean = label.replace("SetengahSegar", "Setengah Segar")

    print("HASIL DIAGNOSIS SISTEM:")
    print(f"  Prediksi Kesegaran : {label_clean}")
    print(f"  Tingkat Keyakinan  : {confidence:.2f}%")
    print(f"  Titik Tekstur SURF : {len(keypoints)} keypoints")

    if show_plot:
        color = {"Segar": "green", "SetengahSegar": "orange", "Busuk": "red"}.get(label, "gray")
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        plt.title("1. Citra Input", fontweight="bold"); plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        plt.title(f"2. {len(keypoints)} Titik SURF", fontweight="bold"); plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB))
        plt.title(f"3. Prediksi: {label_clean} ({confidence:.1f}%)",
                  fontweight="bold", color=color)
        plt.axis("off")
        plt.tight_layout()
        nama = make_safe_filename(image_path.split("/")[-1].split(".")[0])
        save_current_figure(f"prediksi_{nama}.png")
        plt.show()

    return label_clean, confidence
