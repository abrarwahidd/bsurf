# ============================================================
# features/surf_bovw.py
# Ekstraksi deskriptor SURF 64D dan pemetaan ke histogram
# Bag-of-Visual-Words (BoVW) menggunakan KMeans vocabulary.
# ============================================================

import cv2
import numpy as np
from typing import Optional
from sklearn.cluster import MiniBatchKMeans

from src.config.settings import SURF_HESSIAN_THRESHOLD, K_CLUSTERS, KMEANS_BATCH, RANDOM_STATE


def extract_surf_descriptors(
    image: np.ndarray,
    hessian_threshold: int = SURF_HESSIAN_THRESHOLD,
) -> Optional[np.ndarray]:
    """
    Deteksi keypoints dan ekstraksi deskriptor SURF 64D.

    Returns:
        descriptors : ndarray shape (N, 64) atau None jika tidak ada keypoint
    """
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    _, descriptors = surf.detectAndCompute(image, None)
    return descriptors


def build_vocabulary(
    descriptors_list: list,
    k: int = K_CLUSTERS,
) -> MiniBatchKMeans:
    """
    Bangun kamus visual KMeans dari daftar deskriptor SURF.

    Args:
        descriptors_list : list of ndarray, masing-masing shape (N_i, 64)
        k                : jumlah visual words (cluster)

    Returns:
        vocab : MiniBatchKMeans yang sudah difit
    """
    valid = [d for d in descriptors_list if d is not None and len(d) > 0]
    if not valid:
        raise ValueError("Tidak ada deskriptor SURF yang valid untuk membangun vocab.")
    stacked = np.vstack(valid)
    vocab = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=KMEANS_BATCH,
    )
    vocab.fit(stacked)
    return vocab


def create_bovw_histogram(
    descriptor: Optional[np.ndarray],
    vocab: MiniBatchKMeans,
) -> np.ndarray:
    """
    Petakan deskriptor SURF ke histogram BoVW (normalized frequency, sum=1).

    Normalisasi: proporsi tiap visual word — nilai dalam [0, 1],
    lebih konvensional dan intuitif dibanding probability density (PDF).

    Returns:
        histogram : ndarray float32 shape (k,)
    """
    k = vocab.n_clusters
    if descriptor is not None and len(descriptor) > 0:
        words  = vocab.predict(descriptor)
        counts, _ = np.histogram(words, bins=np.arange(k + 1))
        return (counts.astype(np.float32) / (counts.sum() + 1e-8))
    return np.zeros(k, dtype=np.float32)
