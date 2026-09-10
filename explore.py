#!/usr/bin/env python3
# ============================================================
# explore.py
# Visualisasi eksplorasi dataset — menggantikan Cell 5 & 11.
# Jalankan: python explore.py --image dataset/Segar/contoh.jpg
# ============================================================

import argparse
from src.visualization.plots import plot_preprocessing, plot_hsv_decomposition


def main():
    parser = argparse.ArgumentParser(description="Eksplorasi visual citra dataset")
    parser.add_argument("--image", required=True, help="Path ke citra")
    parser.add_argument(
        "--mode",
        choices=["preprocessing", "hsv", "all"],
        default="all",
        help="Jenis visualisasi (default: all)",
    )
    args = parser.parse_args()

    if args.mode in ("preprocessing", "all"):
        plot_preprocessing(args.image)
    if args.mode in ("hsv", "all"):
        plot_hsv_decomposition(args.image)


if __name__ == "__main__":
    main()
