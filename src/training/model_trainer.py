# ============================================================
# training/model_trainer.py
# Final model training pada seluruh dataset menggunakan
# hyperparameter terbaik dari Ablation Study, dan
# fungsi save/load artefak model.
# ============================================================

import os
import glob
import numpy as np
import joblib
import cv2
from typing import Any, Dict, Tuple
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score

from src.config.settings import (
    DATASET_PATH, CLASSES_NAME, IMG_SIZE, K_CLUSTERS,
    RANDOM_STATE, KMEANS_BATCH, MODEL_DIR,
)
from src.preprocessing.segmentation import remove_background
from src.features.hsv_features import extract_hsv_features
from src.features.surf_bovw import extract_surf_descriptors, build_vocabulary, create_bovw_histogram
from src.training.pipeline_builder import build_final_pipeline


def train_final_model(
    best_params: Dict,
    k: int = K_CLUSTERS,
    image_size: Tuple[int, int] = IMG_SIZE,
    save_dir: str = ".",
) -> Tuple[Any, MiniBatchKMeans]:
    """
    Latih model final pada SELURUH dataset menggunakan hyperparameter
    terbaik hasil Ablation Study.

    Langkah:
      1. Ekstraksi HSV features + SURF descriptors semua citra
      2. Bangun vocab KMeans global dari seluruh deskriptor
      3. Buat BoVW histogram → fusi 157D
      4. Fit Pipeline (Scaler → Weighter → SVM) dengan best_params
      5. Simpan model + vocab ke disk

    Akurasi in-sample dilaporkan hanya informatif —
    akurasi generalisasi yang valid ada di Ablation Study (nested CV).

    Returns
    -------
    final_pipeline : Pipeline sklearn yang sudah difit
    final_vocab    : MiniBatchKMeans vocab global
    """
    print("\n" + "=" * 65)
    print("FINAL MODEL TRAINING (seluruh dataset)")
    print(f"  Hyperparameter: {best_params}")
    print("=" * 65)

    label_map  = {cls: i for i, cls in enumerate(CLASSES_NAME)}
    all_hsv, all_descs, all_labels = [], [], []

    for cls in CLASSES_NAME:
        folder = os.path.join(DATASET_PATH, cls)
        paths  = sorted(glob.glob(os.path.join(folder, "*.*")))
        print(f"  Memuat {cls}: {len(paths)} citra...")
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, image_size)
            img_clean, mask = remove_background(img)
            all_hsv.append(extract_hsv_features(img_clean, mask))
            all_descs.append(extract_surf_descriptors(img_clean))
            all_labels.append(label_map[cls])

    print(f"\n  Membangun vocab KMeans global...")
    final_vocab = build_vocabulary(all_descs, k=k)
    print("  ✅ Vocab KMeans global selesai.")

    all_bovw = [create_bovw_histogram(d, final_vocab) for d in all_descs]
    X_final  = np.column_stack([all_hsv, all_bovw]).astype(np.float32)
    y_final  = np.array(all_labels, dtype=np.int32)
    print(f"  Dimensi fitur fusi final: {X_final.shape}")

    final_pipeline = build_final_pipeline(best_params)
    final_pipeline.fit(X_final, y_final)

    train_acc = accuracy_score(y_final, final_pipeline.predict(X_final)) * 100
    print(f"  Akurasi training (in-sample, informatif): {train_acc:.2f}%")
    print("  (Akurasi generalisasi: lihat hasil Ablation Study)")

    save_models(final_pipeline, final_vocab, save_dir)
    return final_pipeline, final_vocab


def save_models(
    pipeline: Any,
    vocab: MiniBatchKMeans,
    save_dir: str = ".",
) -> None:
    """Simpan Pipeline SVM fusi dan vocab KMeans ke disk."""
    model_dir = os.path.join(save_dir, MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)

    svm_path    = os.path.join(model_dir, "fusi_model.pkl")
    kmeans_path = os.path.join(model_dir, "kmeans_vocab.pkl")

    try:
        joblib.dump(pipeline, svm_path)
        print(f"  [BERHASIL] Model SVM Fusi  → {svm_path}")
        if vocab is not None:
            joblib.dump(vocab, kmeans_path)
            print(f"  [BERHASIL] Vocab KMeans    → {kmeans_path}")
    except Exception as exc:
        print(f"  [GAGAL] {exc}")


def load_models(
    save_dir: str = ".",
) -> Tuple[Any, MiniBatchKMeans]:
    """Muat Pipeline SVM fusi dan vocab KMeans dari disk."""
    model_dir   = os.path.join(save_dir, MODEL_DIR)
    svm_path    = os.path.join(model_dir, "fusi_model.pkl")
    kmeans_path = os.path.join(model_dir, "kmeans_vocab.pkl")

    pipeline = joblib.load(svm_path)
    vocab    = joblib.load(kmeans_path)
    print(f"  Model dimuat: {svm_path}")
    print(f"  Vocab dimuat: {kmeans_path}")
    return pipeline, vocab
