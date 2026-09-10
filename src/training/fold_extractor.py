# ============================================================
# training/fold_extractor.py
# Ekstraksi fitur per fold untuk nested CV yang bebas leakage.
#
# Kunci metodologi:
#   - Vocab KMeans HANYA dibangun dari train_idx setiap fold
#   - val_idx menggunakan vocab fold tersebut → zero leakage
#   - Raw descriptors (HSV + SURF) dimuat sekali dari cache
#     agar tidak di-ekstraksi ulang di setiap fold
# ============================================================

import os
import glob
import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import MiniBatchKMeans

import cv2

from src.config.settings import (
    DATASET_PATH, CLASSES_NAME, K_CLUSTERS, IMG_SIZE, RANDOM_STATE,
)
from src.preprocessing.segmentation import remove_background
from src.features.hsv_features import extract_hsv_features
from src.features.surf_bovw import extract_surf_descriptors, build_vocabulary, create_bovw_histogram


# ── Raw Feature Cache (in-memory, satu kali ekstraksi per sesi) ─

_RAW_CACHE: Optional[dict] = None


def get_ordered_paths(
    dataset_path: str = DATASET_PATH,
    classes: List[str] = CLASSES_NAME,
) -> Tuple[List[str], np.ndarray]:
    """
    Kembalikan (ordered_paths, y_all) — path citra terurut per kelas
    beserta label integer-nya. Digunakan sebagai indeks untuk StratifiedKFold.
    """
    label_map     = {cls: i for i, cls in enumerate(classes)}
    ordered_paths = []
    y_all         = []
    for cls in classes:
        folder = os.path.join(dataset_path, cls)
        paths  = sorted(glob.glob(os.path.join(folder, "*.*")))
        ordered_paths.extend(paths)
        y_all.extend([label_map[cls]] * len(paths))
    return ordered_paths, np.array(y_all, dtype=np.int32)


def load_raw_features(
    ordered_paths: List[str],
    image_size: Tuple[int, int] = IMG_SIZE,
    classes: List[str] = CLASSES_NAME,
    use_cache: bool = True,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], np.ndarray]:
    """
    Muat dan ekstraksi raw features (HSV + SURF descriptors) dari semua citra.
    Hasil disimpan di cache in-memory sehingga saat dipanggil berulang kali
    (di setiap fold dan skenario), ekstraksi hanya terjadi SATU kali per sesi.

    Returns:
        all_hsv   : list of ndarray float32 (57,)
        all_descs : list of ndarray float32 (N,64) atau None
        y_all     : ndarray int32 label
    """
    global _RAW_CACHE

    cache_key = (tuple(ordered_paths), image_size)

    if use_cache and _RAW_CACHE is not None and _RAW_CACHE.get("key") == cache_key:
        print("  [Cache] Raw features dimuat dari memori (tidak re-ekstraksi).")
        return _RAW_CACHE["hsv"], _RAW_CACHE["descs"], _RAW_CACHE["y"]

    print(f"  Mengekstrak raw features dari {len(ordered_paths)} citra...")
    label_map = {cls: i for i, cls in enumerate(classes)}

    all_hsv, all_descs, y_all = [], [], []
    for path in ordered_paths:
        cls_name = os.path.basename(os.path.dirname(path))
        img = cv2.imread(path)
        if img is None or cls_name not in label_map:
            continue
        img = cv2.resize(img, image_size)
        img_clean, mask = remove_background(img)
        all_hsv.append(extract_hsv_features(img_clean, mask))
        all_descs.append(extract_surf_descriptors(img_clean))
        y_all.append(label_map[cls_name])

    y_arr = np.array(y_all, dtype=np.int32)

    if use_cache:
        _RAW_CACHE = {"key": cache_key, "hsv": all_hsv, "descs": all_descs, "y": y_arr}
        print("  [Cache] Raw features disimpan ke memori.")

    return all_hsv, all_descs, y_arr


def clear_raw_cache() -> None:
    """Hapus cache in-memory (berguna saat ganti dataset atau image_size)."""
    global _RAW_CACHE
    _RAW_CACHE = None
    print("  [Cache] Raw feature cache dihapus.")


def extract_fold_features(
    all_hsv: List[np.ndarray],
    all_descs: List[Optional[np.ndarray]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    k: int = K_CLUSTERS,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,   # X_fusi_tr, X_hsv_tr, X_surf_tr
    np.ndarray, np.ndarray, np.ndarray,   # X_fusi_val, X_hsv_val, X_surf_val
    MiniBatchKMeans,                       # vocab_fold
]:
    """
    Bangun vocab KMeans dari train_idx dan buat fitur fusi/HSV/SURF
    untuk training dan validation fold secara terpisah.

    Vocab dibangun HANYA dari descriptors train_idx → zero leakage [FIX #1].

    Returns:
        X_fusi_tr, X_hsv_tr, X_surf_tr  : fitur training
        X_fusi_val, X_hsv_val, X_surf_val : fitur validation
        vocab_fold                        : KMeans yang difit dari train
    """
    # Pisahkan raw features per split
    hsv_tr   = [all_hsv[i]   for i in train_idx]
    descs_tr = [all_descs[i] for i in train_idx]
    hsv_val   = [all_hsv[i]   for i in val_idx]
    descs_val = [all_descs[i] for i in val_idx]

    # Vocab hanya dari train
    vocab_fold = build_vocabulary(descs_tr, k=k)

    # BoVW histograms
    bovw_tr  = [create_bovw_histogram(d, vocab_fold) for d in descs_tr]
    bovw_val = [create_bovw_histogram(d, vocab_fold) for d in descs_val]

    X_fusi_tr  = np.column_stack([hsv_tr,  bovw_tr]).astype(np.float32)
    X_hsv_tr   = np.array(hsv_tr,  dtype=np.float32)
    X_surf_tr  = np.array(bovw_tr, dtype=np.float32)
    X_fusi_val = np.column_stack([hsv_val, bovw_val]).astype(np.float32)
    X_hsv_val  = np.array(hsv_val,  dtype=np.float32)
    X_surf_val = np.array(bovw_val, dtype=np.float32)

    return (X_fusi_tr, X_hsv_tr, X_surf_tr,
            X_fusi_val, X_hsv_val, X_surf_val,
            vocab_fold)
