# ============================================================
# utils/io_helpers.py
# Helper I/O: .env loader, output directory, figure saving,
# filename sanitizer, dan WandB wrapper.
# ============================================================

import os
import matplotlib.pyplot as plt
from typing import Any, Optional

from src.config.settings import OUTPUT_IMG_DIR


# ── .env & OS Env ───────────────────────────────────────────

def load_env_file(env_path: str = ".env") -> None:
    """Muat variabel dari file .env ke os.environ."""
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


# ── Output Directory & Figure ───────────────────────────────

def ensure_output_dir(path: str = OUTPUT_IMG_DIR) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_safe_filename(text: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")
    return "_".join(part for part in safe.split("_") if part)


def save_current_figure(filename: str, dpi: int = 300,
                        output_dir: str = OUTPUT_IMG_DIR) -> str:
    output_path = os.path.join(ensure_output_dir(output_dir), filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"  Gambar disimpan: {output_path}")
    return output_path


# ── WandB Wrapper ───────────────────────────────────────────

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


def start_wandb_run(
    project_name: str,
    config: dict,
    api_key: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Any:
    if not _WANDB_AVAILABLE:
        return None
    if not api_key:
        print("WANDB_API_KEY belum diisi. Logging WandB dilewati.")
        return None
    key = api_key.strip()
    if key.startswith("WANDB_API_KEY="):
        key = key.split("=", 1)[1].strip()
    if not key:
        print("WANDB_API_KEY kosong. Logging WandB dilewati.")
        return None
    os.environ["WANDB_API_KEY"] = key
    try:
        return wandb.init(project=project_name, name=run_name,
                          config=config, reinit=True)
    except Exception as exc:
        print(f"Gagal inisialisasi WandB: {exc}")
        return None


def log_wandb_metrics(metrics: dict, step: Optional[int] = None) -> None:
    if _WANDB_AVAILABLE and wandb.run is not None:
        if step is None:
            wandb.log(metrics)
        else:
            wandb.log(metrics, step=step)


def finish_wandb_run() -> None:
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
