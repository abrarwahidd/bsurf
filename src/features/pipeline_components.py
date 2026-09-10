# ============================================================
# features/pipeline_components.py
# Custom sklearn Transformer: FeatureWeighter
# Mengamplifikasi blok fitur HSV agar kontribusinya seimbang
# dengan fitur SURF-BoVW yang berdimensi lebih besar.
# ============================================================

import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.settings import NUM_HSV_FEATURES


class FeatureWeighter(BaseEstimator, TransformerMixin):
    """
    Stateless transformer yang mengalikan fitur warna HSV
    (indeks 0 .. num_hsv_features-1) dengan hsv_weight,
    sedangkan fitur tekstur SURF-BoVW dibiarkan.

    Digunakan di dalam Pipeline sklearn sehingga hsv_weight
    bisa di-tune lewat GridSearchCV.

    Parameters
    ----------
    hsv_weight        : pengali fitur warna (default 1.0 = tidak berubah)
    num_hsv_features  : jumlah dimensi fitur warna di depan vektor fitur
    """

    def __init__(
        self,
        hsv_weight: float = 1.0,
        num_hsv_features: int = NUM_HSV_FEATURES,
    ) -> None:
        self.hsv_weight       = hsv_weight
        self.num_hsv_features = num_hsv_features

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureWeighter":
        return self  # stateless

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X.copy()
        X_out[:, : self.num_hsv_features] *= self.hsv_weight
        return X_out
