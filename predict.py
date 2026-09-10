#!/usr/bin/env python3
# ============================================================
# predict.py
# Entry point inferensi satu citra.
# Jalankan: python predict.py --image test_img/segar4.webp
# ============================================================

import argparse
from src.training.model_trainer import load_models
from src.evaluation.predictor import predict_image


def main():
    parser = argparse.ArgumentParser(
        description="Prediksi kesegaran daging sapi dari citra"
    )
    parser.add_argument("--image",  required=True, help="Path ke citra input")
    parser.add_argument("--model_dir", default=".", help="Direktori model (default: .)")
    parser.add_argument("--no_plot", action="store_true", help="Skip visualisasi")
    args = parser.parse_args()

    pipeline, vocab = load_models(save_dir=args.model_dir)
    label, confidence = predict_image(
        image_path=args.image,
        pipeline=pipeline,
        vocab=vocab,
        show_plot=not args.no_plot,
    )
    print(f"\nHasil: {label} ({confidence:.2f}%)")


if __name__ == "__main__":
    main()
