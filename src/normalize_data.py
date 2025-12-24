# src/normalize_data.py — Run: python -m src.normalize_data features_robust.csv

# Opis:
#     Skrypt do standaryzacji (mean=0, std=1) cech tabelarycznych po precomputingu.
#     - Normalizacja liczona WYŁĄCZNIE na zbiorze treningowym,
#       a następnie stosowana do całego datasetu (train/val/test)
#     - Brak data leakage
#     - Deterministyczny split (random_state)
#     - NIE normalizujemy: filepath, datetime, hour

# Wejście:
#     - CSV z cechami (np. features_advanced.csv, features_robust.csv)

# Wyjście:
#     - *_normalized.csv        -> znormalizowane dane
#     - *_scaler.npz            -> mean, std, feature_cols (do inference)

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


META_COLS = {"filepath", "datetime", "hour"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "features_csv",
        type=Path,
        help="Ścieżka do pliku z cechami (np. features_robust.csv)",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.8,
        help="Udział danych treningowych (domyślnie 0.8)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed losowości (domyślnie 42)",
    )
    args = parser.parse_args()

    features_path: Path = args.features_csv
    if not features_path.exists():
        raise FileNotFoundError(f"Brak pliku: {features_path}")

    print(f"[INFO] Wczytywanie danych z: {features_path}")
    df = pd.read_csv(features_path)
    print(f"[INFO] Shape: {df.shape}")

    if "hour" not in df.columns:
        raise ValueError("Brak kolumny 'hour' – nie wiem jak zrobić stratify")

    # --- wybór kolumn cech ---
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"[INFO] Liczba cech do normalizacji: {len(feature_cols)}")

    X = df[feature_cols].astype(np.float32).to_numpy()
    y_hour = df["hour"].astype(int).to_numpy()

    # --- split tylko do wyliczenia statystyk ---
    X_train, _X_rest = train_test_split(
        X,
        train_size=args.train_size,
        random_state=args.random_state,
        stratify=y_hour,
    )

    print(f"[INFO] Liczenie mean/std na TRAIN: {X_train.shape}")

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6

    # --- normalizacja całego datasetu ---
    X_norm = (X - mean) / std

    # --- składamy dataframe wyjściowy ---
    df_norm = df.copy()
    df_norm[feature_cols] = X_norm

    # --- ścieżki wyjściowe ---
    out_csv = features_path.with_name(features_path.stem + "_normalized.csv")
    out_scaler = features_path.with_name(features_path.stem + "_scaler.npz")

    df_norm.to_csv(out_csv, index=False)
    np.savez(
        out_scaler,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feature_cols=np.array(feature_cols),
    )

    print(f"[INFO] Zapisano znormalizowany CSV: {out_csv}")
    print(f"[INFO] Zapisano scaler: {out_scaler}")


if __name__ == "__main__":
    main()
