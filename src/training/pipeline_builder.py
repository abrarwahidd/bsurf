# ============================================================
# training/pipeline_builder.py
# Factory function untuk membangun Pipeline sklearn.
# Dipisahkan agar mudah diganti tanpa menyentuh logika evaluasi.
# ============================================================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config.settings import RANDOM_STATE, NUM_HSV_FEATURES
from src.features.pipeline_components import FeatureWeighter


def build_pipeline(is_fusion: bool) -> Pipeline:
    """
    Buat Pipeline sklearn sesuai skenario.

    Skenario non-fusi : Scaler → SVM
    Skenario fusi     : Scaler → FeatureWeighter → SVM

    SVC selalu menggunakan probability=True dan class_weight='balanced'
    agar bisa dipakai predict_proba() dan robust terhadap imbalance.
    """
    svm = SVC(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        probability=True,
    )
    if is_fusion:
        return Pipeline([
            ("scaler",   StandardScaler()),
            ("weighter", FeatureWeighter(num_hsv_features=NUM_HSV_FEATURES)),
            ("svm",      svm),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    svm),
    ])


def build_final_pipeline(best_params: dict) -> Pipeline:
    """
    Buat Pipeline final (untuk train_final_model) dengan hyperparameter
    yang sudah diketahui dari Ablation Study.
    Selalu menggunakan arsitektur fusi.
    """
    return Pipeline([
        ("scaler",   StandardScaler()),
        ("weighter", FeatureWeighter(
            hsv_weight=best_params.get("weighter__hsv_weight", 1.0),
            num_hsv_features=NUM_HSV_FEATURES,
        )),
        ("svm", SVC(
            C=best_params.get("svm__C", 10),
            kernel=best_params.get("svm__kernel", "rbf"),
            gamma=best_params.get("svm__gamma", "scale"),
            random_state=RANDOM_STATE,
            class_weight="balanced",
            probability=True,
        )),
    ])
