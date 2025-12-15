# src/predict_hour.py

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from .utils import extract_rgb_hsv_stats


MODELS_DIR = Path("models")


def load_model():
    model_path = MODELS_DIR / "best_advanced.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu {model_path}. "
            f"Najpierw uruchom: python -m src.baseline_advanced"
        )
    return joblib.load(model_path)


def predict_for_image(image_path: Path):
    feats = extract_rgb_hsv_stats(image_path)
    if feats is None:
        raise RuntimeError(f"Nie udało się wczytać obrazu: {image_path}")

    X = feats.reshape(1, -1)  # (1, n_features)
    model = load_model()

    y_pred = model.predict(X)[0]

    # jeśli model umie zwracać prawdopodobieństwa:
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]

    return int(y_pred), proba


def main():
    parser = argparse.ArgumentParser(
        description="Przewidywanie godziny na podstawie pojedynczego obrazu"
    )
    parser.add_argument("image", type=str, help="Ścieżka do pliku JPG/PNG")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Plik nie istnieje: {img_path}")

    hour, proba = predict_for_image(img_path)
    print(f"Przewidywana godzina: {hour}")

    if proba is not None:
        # wypisz top 5 najbardziej prawdopodobnych godzin
        top_k = 5
        indices = np.argsort(proba)[::-1][:top_k]
        print("\nTop godziny:")
        for i in indices:
            print(f"  {i:2d}: {proba[i]:.3f}")


if __name__ == "__main__":
    main()
