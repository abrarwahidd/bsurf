# ============================================================
# evaluation/cross_validator.py
# Nested cross-validation yang metodologis bersih.
#
# Outer loop : StratifiedKFold N_FOLDS (true hold-out per fold)
# Inner loop : GridSearchCV StratifiedKFold N_INNER_FOLDS (HP tuning)
# Akurasi    : accuracy_score(y_outer_val, y_pred_outer) — bukan best_score_
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config.settings import (
    CLASSES_NAME, IMG_SIZE, K_CLUSTERS, N_FOLDS, N_INNER_FOLDS,
    N_JOBS, PARAM_GRID_BASE, PARAM_GRID_FUSION, RANDOM_STATE,
)
from src.training.fold_extractor import (
    get_ordered_paths, load_raw_features, extract_fold_features,
)
from src.training.pipeline_builder import build_pipeline


def evaluate_scenario(
    scenario: str,
    n_folds: int = N_FOLDS,
    k: int = K_CLUSTERS,
    image_size: Tuple[int, int] = IMG_SIZE,
    use_cache: bool = True,
) -> Tuple[float, float, np.ndarray, Any, Dict, Any]:
    """
    Evaluasi satu skenario dengan nested CV yang bebas leakage.

    Parameters
    ----------
    scenario  : 'fusion' | 'hsv' | 'surf'
    n_folds   : jumlah outer fold
    k         : jumlah visual words KMeans
    image_size: ukuran resize citra
    use_cache : gunakan raw feature cache in-memory

    Returns
    -------
    mean_acc      : rata-rata akurasi outer fold hold-out (%)
    std_acc       : standar deviasi akurasi (%)
    y_pred_oof    : prediksi out-of-fold lengkap (untuk Confusion Matrix)
    best_estimator: Pipeline terbaik dari fold dengan akurasi tertinggi
    best_params   : dict hyperparameter terbaik
    last_grid     : objek GridSearch fold terakhir (untuk plot hsv_weight)
    best_vocab    : MiniBatchKMeans vocab dari fold terbaik (deployment)
    """
    is_fusion  = (scenario == "fusion")
    param_grid = PARAM_GRID_FUSION if is_fusion else PARAM_GRID_BASE
    outer_cv   = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=RANDOM_STATE)

    ordered_paths, y_all = get_ordered_paths()
    all_hsv, all_descs, _ = load_raw_features(
        ordered_paths, image_size=image_size, use_cache=use_cache
    )

    fold_scores        = []
    fold_details: List[dict] = []
    oof_preds          = np.empty(len(y_all), dtype=np.int32)
    best_estimator     = None
    best_score_global  = -1.0
    best_params_global: Dict = {}
    best_vocab_global  = None
    last_grid          = None

    for fold_idx, (train_idx, val_idx) in enumerate(
        outer_cv.split(ordered_paths, y_all), 1
    ):
        print(f"  Fold {fold_idx}/{n_folds}...", end=" ", flush=True)

        # Bangun fitur per fold — vocab hanya dari train [FIX #1]
        (X_fusi_tr, X_hsv_tr, X_surf_tr,
         X_fusi_val, X_hsv_val, X_surf_val,
         vocab_fold) = extract_fold_features(
            all_hsv, all_descs, train_idx, val_idx, k=k
        )

        X_tr  = {"fusion": X_fusi_tr,  "hsv": X_hsv_tr,  "surf": X_surf_tr }[scenario]
        X_val = {"fusion": X_fusi_val, "hsv": X_hsv_val, "surf": X_surf_val}[scenario]
        y_tr  = y_all[train_idx]
        y_val = y_all[val_idx]

        # Inner GridSearchCV
        inner_cv    = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                                       random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            build_pipeline(is_fusion),
            param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=N_JOBS,
            refit=True,
        )
        grid_search.fit(X_tr, y_tr)
        last_grid = grid_search

        # Outer hold-out evaluation [FIX #2 / #7]
        y_val_pred         = grid_search.best_estimator_.predict(X_val)
        fold_acc           = accuracy_score(y_val, y_val_pred) * 100
        oof_preds[val_idx] = y_val_pred
        fold_scores.append(fold_acc)

        fold_details.append({
            "Fold":        fold_idx,
            "Akurasi (%)": round(fold_acc, 2),
            "n_train":     len(y_tr),
            "n_val":       len(y_val),
            "kernel":      grid_search.best_params_.get("svm__kernel", "-"),
            "C":           grid_search.best_params_.get("svm__C", "-"),
        })
        print(f"acc={fold_acc:.2f}% | {grid_search.best_params_}")

        if fold_acc > best_score_global:
            best_score_global  = fold_acc
            best_estimator     = grid_search.best_estimator_
            best_params_global = grid_search.best_params_
            best_vocab_global  = vocab_fold

    mean_acc = float(np.mean(fold_scores))
    std_acc  = float(np.std(fold_scores))
    print(f"  ➜ Mean: {mean_acc:.2f}% ± {std_acc:.2f}%  "
          f"[{min(fold_scores):.2f} – {max(fold_scores):.2f}]")
    print("\n" + pd.DataFrame(fold_details).to_string(index=False))

    return (mean_acc, std_acc, oof_preds,
            best_estimator, best_params_global,
            last_grid, best_vocab_global)


def run_ablation_study(
    use_cache: bool = True,
) -> dict:
    """
    Jalankan tiga skenario Ablation Study secara berurutan:
      1. SURF-BoVW saja (tekstur)
      2. HSV saja (warna)
      3. Fusi Berbobot (HSV + SURF)

    Returns dict dengan kunci per skenario berisi semua return value
    dari evaluate_scenario().
    """
    results = {}
    scenarios = [
        ("surf",   "[1/3] HANYA TEKSTUR (SURF-BoVW 100D)"),
        ("hsv",    "[2/3] HANYA WARNA (HSV 57D)"),
        ("fusion", "[3/3] FUSI FITUR BERBOBOT (157D)"),
    ]

    for key, label in scenarios:
        print(f"\n{'='*65}")
        print(label)
        print("="*65)
        out = evaluate_scenario(key, use_cache=use_cache)
        results[key] = {
            "mean_acc":      out[0],
            "std_acc":       out[1],
            "oof_preds":     out[2],
            "best_estimator": out[3],
            "best_params":   out[4],
            "last_grid":     out[5],
            "best_vocab":    out[6],
        }
        print(f"  ✅ Akurasi {key.upper()}: {out[0]:.2f}%")

    print(f"\n{'='*65}")
    print("RINGKASAN ABLATION STUDY")
    for key, label in scenarios:
        tag = key.upper().ljust(6)
        print(f"  {tag}: {results[key]['mean_acc']:.2f}% ± {results[key]['std_acc']:.2f}%")
    print("="*65)
    print("Akurasi dihitung dari outer fold hold-out (true nested CV).")

    return results
