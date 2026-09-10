#!/usr/bin/env python3
# ============================================================
# run_experiment.py
# Entry point utama — menggantikan Cell 8, 9, 10, 12 notebook.
# Jalankan: python run_experiment.py
# ============================================================

import os
import numpy as np
from sklearn.metrics import classification_report

from src.config.settings import CLASSES_NAME
from src.utils.io_helpers import (
    load_env_file, start_wandb_run, log_wandb_metrics,
    finish_wandb_run,
)
from src.evaluation.cross_validator import run_ablation_study
from src.training.model_trainer import train_final_model
from src.visualization.plots import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_hsv_weight_impact,
)

# ── WandB (opsional) ────────────────────────────────────────
load_env_file(".env")
wandb_run = start_wandb_run(
    project_name="beefresearch-TA",
    run_name="svm-surf-hsv-modular",
    api_key=os.getenv("WANDB_API_KEY", ""),
    config={
        "k_clusters": 100,
        "n_folds": 5,
        "random_state": 42,
        "architecture": "HSV+SURF-BoVW+SVM",
    },
)

# ── 1. Ablation Study ────────────────────────────────────────
results = run_ablation_study(use_cache=True)

acc_surf  = results["surf"]["mean_acc"]
acc_hsv   = results["hsv"]["mean_acc"]
acc_fusi  = results["fusion"]["mean_acc"]
pred_surf  = results["surf"]["oof_preds"]
pred_hsv   = results["hsv"]["oof_preds"]
pred_fusi  = results["fusion"]["oof_preds"]
grid_fusi  = results["fusion"]["last_grid"]
params_fusi = results["fusion"]["best_params"]

if wandb_run:
    log_wandb_metrics({
        "accuracy_surf": acc_surf,
        "accuracy_hsv":  acc_hsv,
        "accuracy_fusi": acc_fusi,
    })

# Ground truth untuk confusion matrix
from src.training.fold_extractor import get_ordered_paths
_, y_label = get_ordered_paths()

# ── 2. Visualisasi ──────────────────────────────────────────
plot_accuracy_comparison(acc_surf, acc_hsv, acc_fusi)

for preds, label in [
    (pred_surf, "Confusion Matrix - Skenario 1 (SURF)"),
    (pred_hsv,  "Confusion Matrix - Skenario 2 (HSV)"),
    (pred_fusi, "Confusion Matrix - Metode Fusi (HSV + SURF)"),
]:
    plot_confusion_matrix(y_label, preds, title=label)

print("\nLAPORAN KLASIFIKASI RINCI — MODEL FUSI:")
print(classification_report(y_label, pred_fusi, target_names=CLASSES_NAME))

# HSV weight impact
if grid_fusi is not None:
    plot_hsv_weight_impact(grid_fusi)

# ── 3. Final Model Training ──────────────────────────────────
final_model, final_vocab = train_final_model(
    best_params=params_fusi,
    save_dir=".",
)
print("\n✅ Model final tersimpan di ./models/")
print("   Gunakan predict.py untuk inferensi.")

if wandb_run:
    finish_wandb_run()
