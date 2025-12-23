# src/train_robust_time.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


FEATURES_PATH = Path("features_robust.csv")


@dataclass(frozen=True)
class BinningConfig:
    """
    Konfiguracja binning'u.
    Przykład: bin_size=4 => 6 klas: 0-3, 4-7, 8-11, 12-15, 16-19, 20-23
    """
    bin_size: int = 4  # możesz zmienić np. na 3 albo 6


def hour_to_bin(hour: int, bin_size: int) -> int:
    return int(hour // bin_size)


def bin_to_hour_center(bin_id: int, bin_size: int) -> float:
    # środek przedziału w godzinach, np. bin 0 (0-3) => 1.5
    start = bin_id * bin_size
    end = start + bin_size - 1
    return (start + end) / 2.0


def circular_hour_error(pred_hour: float, true_hour: float) -> float:
    """
    Błąd na kole 24h: różnica minimalna po okręgu.
    Np. pred=23, true=1 => błąd 2, a nie 22.
    """
    diff = abs(pred_hour - true_hour)
    return float(min(diff, 24.0 - diff))


def top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """
    Top-k accuracy dla klasyfikacji wieloklasowej:
    czy prawdziwa klasa jest w k najbardziej prawdopodobnych.
    """
    topk = np.argsort(-proba, axis=1)[:, :k]
    hits = (topk == y_true.reshape(-1, 1)).any(axis=1)
    return float(hits.mean())


def make_models() -> List[Tuple[str, object]]:
    """
    Modele: dobieramy takie, które potrafią dać predict_proba (dla top-k i uncertain).
    RF i GB zwykle sprawują się dobrze na cechach tablicowych.
    """
    return [
        ("logreg", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=None)),
        ])),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=1,
        )),
        ("gb", GradientBoostingClassifier(random_state=42)),
    ]


def main() -> None:
    import time
    start = time.time()

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku {FEATURES_PATH}. "
            "Najpierw uruchom: python -m src.precompute_features_robust"
        )

    # --- KROK 1: Wczytanie cech ---
    df = pd.read_csv(FEATURES_PATH)
    print("Wczytano:", df.shape)

    # --- KROK 2: Przygotowanie listy cech (wszystko poza metadanymi) ---
    drop_cols = {"filepath", "datetime", "hour"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].astype(np.float32)

    # --- KROK 3: Binning godzin do przedziałów ---
    cfg = BinningConfig(bin_size=4)
    y_hour = df["hour"].astype(int).to_numpy()
    y = np.array([hour_to_bin(h, cfg.bin_size) for h in y_hour], dtype=int)

    n_classes = int(np.max(y) + 1)
    print(f"Bin size = {cfg.bin_size}, liczba klas = {n_classes}")

    # --- KROK 4: Split train/test ---
    X_train, X_test, y_train, y_test, y_hour_train, y_hour_test = train_test_split(
        X, y, y_hour,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --- KROK 5: Trening i porównanie modeli ---
    best_name = None
    best_model = None
    best_acc = -1.0

    for name, model in make_models():
        print("\n" + "=" * 80)
        print("Trening modelu:", name)

        # --- BLOK AKWIZYCJI: dopasowanie modelu do danych ---
        model.fit(X_train, y_train)

        # --- KROK 6: Predykcje i metryki ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # predict_proba do top-k i uncertain
        proba = model.predict_proba(X_test)

        acc_top3 = top_k_accuracy(proba, y_test, k=3)
        acc_top2 = top_k_accuracy(proba, y_test, k=2)

        print(f"Accuracy (top-1): {acc:.4f}")
        print(f"Accuracy (top-2): {acc_top2:.4f}")
        print(f"Accuracy (top-3): {acc_top3:.4f}")

        # --- KROK 7: Błąd w godzinach (circular) na bazie środka przedziału ---
        pred_bins = y_pred
        pred_hour_center = np.array([bin_to_hour_center(b, cfg.bin_size) for b in pred_bins], dtype=float)
        true_hour = y_hour_test.astype(float)

        circ_err = np.array([circular_hour_error(ph, th) for ph, th in zip(pred_hour_center, true_hour)], dtype=float)
        print(f"Circular MAE (w godzinach): {circ_err.mean():.3f}")

        # --- KROK 8: Mechanizm 'uncertain' (odrzuć niepewne predykcje) ---
        # Proste i skuteczne: jeśli max(prob) < threshold => uncertain
        threshold = 0.45
        maxp = proba.max(axis=1)
        keep = maxp >= threshold

        if keep.any():
            acc_confident = accuracy_score(y_test[keep], y_pred[keep])
            coverage = float(keep.mean())
            print(f"Confident accuracy (thr={threshold}): {acc_confident:.4f} przy coverage={coverage:.3f}")
        else:
            print(f"Confident accuracy: brak predykcji powyżej progu {threshold}")

        # --- KROK 9: Wybór najlepszego modelu ---
        # Wybieramy po top-1, ale równie dobrze możesz wybierać po circular MAE albo top-3.
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    # --- KROK 10: Raport dla najlepszego modelu ---
    assert best_model is not None and best_name is not None
    print("\n" + "#" * 80)
    print(f"Najlepszy model: {best_name} (top-1 accuracy = {best_acc:.4f})")
    print("#" * 80)

    y_pred_best = best_model.predict(X_test)
    print("\nClassification report (binning):")
    print(classification_report(y_test, y_pred_best))

    print("\nConfusion matrix (binning):")
    print(confusion_matrix(y_test, y_pred_best))

    # Podsumowanie czasu trenowania
    elapsed = time.time() - start
    import datetime
    print("Czas trenowania:", str(datetime.timedelta(seconds=int(elapsed))))


if __name__ == "__main__":
    main()
