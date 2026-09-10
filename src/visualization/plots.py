# ============================================================
# visualization/plots.py
# Semua fungsi visualisasi hasil eksperimen:
#   - Ablation Study bar chart
#   - Confusion Matrix heatmap
#   - HSV weight impact plot
#   - HSV decomposition & histogram
#   - Preprocessing visualization (GrabCut + SURF)
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List
from sklearn.metrics import confusion_matrix

from src.config.settings import CLASSES_NAME, IMG_SIZE, SURF_HESSIAN_THRESHOLD
from src.preprocessing.segmentation import remove_background
from src.utils.io_helpers import save_current_figure, make_safe_filename


# ── Ablation Study ──────────────────────────────────────────

def plot_accuracy_comparison(
    acc_surf: float,
    acc_hsv: float,
    acc_fusi: float,
) -> None:
    """Bar chart perbandingan akurasi tiga skenario ablation."""
    sns.set_theme(style="whitegrid",
                  rc={"axes.edgecolor": "0.15", "axes.linewidth": 1.25})
    methods    = ["Hanya SURF\n(Tekstur)", "Hanya HSV\n(Warna)", "FUSI\n(SURF + HSV)"]
    accuracies = [acc_surf, acc_hsv, acc_fusi]
    colors     = ["#e74c3c", "#2980b9", "#27ae60"]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(methods, accuracies, color=colors,
                   edgecolor="black", linewidth=1.5, width=0.55)
    bars[2].set_hatch("//")
    plt.ylim(0, 105)
    plt.ylabel("Akurasi Validasi (%)", fontsize=13, fontweight="bold", labelpad=12)
    plt.title("Perbandingan Akurasi Skenario Uji (Ablation Study)",
              fontsize=16, fontweight="bold", pad=20)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1.5,
                 f"{yval:.2f}%", ha="center", va="bottom",
                 fontsize=13, fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor="black",
                           boxstyle="round,pad=0.2", alpha=0.8))
    plt.xticks(fontsize=12, fontweight="bold")
    sns.despine(left=True, top=True, right=True)
    plt.tight_layout()
    save_current_figure("evaluasi_perbandingan_akurasi.png")
    plt.show()


# ── Confusion Matrix ─────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    classes: List[str] = CLASSES_NAME,
) -> None:
    """Heatmap Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5.5))
    ax = sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu",
        xticklabels=classes, yticklabels=classes,
        annot_kws={"size": 14, "weight": "bold"},
        linewidths=1.5, linecolor="black", cbar_kws={"shrink": 0.8},
    )
    for _, spine in ax.spines.items():
        spine.set_visible(True); spine.set_linewidth(1.5)
    plt.title(title, fontsize=15, fontweight="bold", pad=15)
    plt.ylabel("Label Aktual",         fontsize=13, fontweight="bold", labelpad=10)
    plt.xlabel("Label Prediksi Sistem", fontsize=13, fontweight="bold", labelpad=10)
    plt.xticks(fontsize=12); plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    save_current_figure(f"evaluasi_{make_safe_filename(title)}.png")
    plt.show()


# ── HSV Weight Impact ────────────────────────────────────────

def plot_hsv_weight_impact(grid_search_obj) -> None:
    """
    Grafik pengaruh hsv_weight terhadap akurasi validasi inner CV,
    diambil dari cv_results_ GridSearch fold terakhir.
    """
    results     = pd.DataFrame(grid_search_obj.cv_results_)
    bp          = grid_search_obj.best_params_
    best_kernel = bp.get("svm__kernel")
    best_c      = bp.get("svm__C")
    best_gamma  = bp.get("svm__gamma")
    best_weight = bp.get("weighter__hsv_weight")

    filtered = results[
        (results["param_svm__kernel"] == best_kernel) &
        (results["param_svm__C"]      == best_c)      &
        (results["param_svm__gamma"]  == best_gamma)
    ].sort_values("param_weighter__hsv_weight")

    weights = filtered["param_weighter__hsv_weight"].tolist()
    scores  = (filtered["mean_test_score"] * 100).tolist()

    plt.figure(figsize=(9, 5.5))
    plt.plot(weights, scores, marker="o", linestyle="-",
             color="#8e44ad", linewidth=2, markersize=8)
    plt.axvline(x=best_weight, color="r", linestyle="--",
                label=f"Bobot Optimal ({best_weight})")
    plt.ylim(min(scores) - 1.5, max(scores) + 3.5)
    plt.title("Pengaruh hsv_weight Terhadap Akurasi Validasi",
              fontsize=14, fontweight="bold")
    plt.xlabel("Nilai hsv_weight", fontsize=12)
    plt.ylabel("Akurasi (%)", fontsize=12)
    for i, txt in enumerate(scores):
        plt.annotate(f"{txt:.1f}%", (weights[i], scores[i]),
                     textcoords="offset points", xytext=(0, 10), ha="center")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_current_figure("evaluasi_pengaruh_hsv_weight.png")
    plt.show()


# ── HSV Decomposition ────────────────────────────────────────

def plot_hsv_decomposition(image_path: str) -> None:
    """Visualisasi kanal HSV dan histogram 1D (16 bins) per kanal."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Citra tidak ditemukan: {image_path}"); return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    hist_h = cv2.calcHist([img_hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([img_hsv], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([img_hsv], [2], None, [16], [0, 256]).flatten()

    plt.figure(figsize=(16, 7))
    plt.suptitle("Dekomposisi Ruang Warna HSV dan Ekstraksi Histogram 1D",
                 fontsize=16, fontweight="bold", y=1.02)

    for i, (img_ch, cmap, title) in enumerate(
        [(img_rgb, None, "Citra Asli (RGB)"),
         (h, "hsv", "Kanal Hue (H)"),
         (s, "gray", "Kanal Saturation (S)"),
         (v, "gray", "Kanal Value (V)")]
    ):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img_ch, cmap=cmap)
        plt.title(title, fontweight="bold", fontsize=12); plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.text(0.5, 0.5, "Ekstraksi\nHistogram 1D\n(16 Bins)",
             fontsize=14, ha="center", va="center", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="gray"))
    plt.axis("off")

    for idx, (hist, color, label) in enumerate(zip(
        [hist_h, hist_s, hist_v],
        ["red", "green", "blue"],
        ["Histogram Hue", "Histogram Saturation", "Histogram Value"],
    )):
        plt.subplot(2, 4, 6 + idx)
        plt.plot(hist, color=color, marker="o", linewidth=2)
        plt.title(f"{label} (16 Bins)", fontsize=11)
        plt.xlabel("Bins (0–15)")
        if idx == 0: plt.ylabel("Frekuensi Piksel")
        plt.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    nama = make_safe_filename(image_path.split("/")[-1].split(".")[0])
    save_current_figure(f"hsv_dekomposisi_{nama}.png")
    plt.show()


# ── Preprocessing Visualization ──────────────────────────────

def plot_preprocessing(image_path: str) -> None:
    """Visualisasi tiga tahap preprocessing: asli, GrabCut, SURF keypoints."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Citra tidak ditemukan: {image_path}"); return

    img_resized          = cv2.resize(img_bgr, IMG_SIZE)
    img_clean, _         = remove_background(img_resized)
    surf                 = cv2.xfeatures2d.SURF_create(SURF_HESSIAN_THRESHOLD)
    keypoints, _         = surf.detectAndCompute(img_clean, None)
    img_kp               = cv2.drawKeypoints(
        img_clean, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
    )

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title("1. Citra Asli", fontsize=14, fontweight="bold"); plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB))
    plt.title("2. Segmentasi (GrabCut)", fontsize=14, fontweight="bold"); plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.title(f"3. Deteksi Tekstur ({len(keypoints)} Titik SURF)",
              fontsize=14, fontweight="bold"); plt.axis("off")
    plt.tight_layout()
    save_current_figure("preprocessing_grabcut_surf.png")
    plt.show()
