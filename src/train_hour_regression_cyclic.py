# train_hour_regression_cyclic.py — Run: python -m src.train_hour_regression_cyclic
# src/train_hour_regression_cyclic.py
# -----------------------------------------------------------------------------
# Cel: REGRESJA dokładnej godziny (0..24) z obrazu, ale w sposób zgodny z naturą
#      czasu (cykliczność 24h). Zamiast klasyfikować 24 klasy, uczymy regresję
#      na okręgu: przewidujemy (sin, cos) kąta odpowiadającego godzinie.
#
# Dlaczego (sin, cos)?
# - godzina jest cykliczna: 23 jest blisko 0
# - klasyczna regresja na liczbie "hour" nie rozumie zawijania (wrap-around)
# - sin/cos koduje pozycję na kole i pozwala modelowi uczyć się gładko
# -----------------------------------------------------------------------------

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES_PATH = Path("features_robust.csv")


# -----------------------------------------------------------------------------
# Proste logowanie do terminala, z flush=True żeby logi pojawiały się od razu.
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


# -----------------------------------------------------------------------------
# KROK A: Zamiana godzin na wektor 2D (sin, cos) na okręgu 24h.
# hour: (N,) w [0..23] lub [0..24)
# wynik: (N,2) => [sin(theta), cos(theta)]
# -----------------------------------------------------------------------------
def hour_to_sin_cos(hour: np.ndarray) -> np.ndarray:
    theta = 2.0 * np.pi * (hour.astype(np.float32) / 24.0)
    return np.stack([np.sin(theta), np.cos(theta)], axis=1)


# -----------------------------------------------------------------------------
# KROK B: Zamiana predykcji (sin, cos) z powrotem na godzinę w [0..24).
# Uwaga: używamy atan2(sin, cos), a potem mapujemy kąt do [0, 2pi).
# -----------------------------------------------------------------------------
def sin_cos_to_hour(y_sc: np.ndarray) -> np.ndarray:
    theta = np.arctan2(y_sc[:, 0], y_sc[:, 1])           # [-pi, pi]
    theta = theta % (2.0 * np.pi)                        # [0, 2pi)
    hour = theta * 24.0 / (2.0 * np.pi)                  # [0, 24)
    return hour


# -----------------------------------------------------------------------------
# KROK C: Błąd na okręgu 24h (circular error).
# pred=23, true=1 => błąd 2, a nie 22.
# -----------------------------------------------------------------------------
def circular_hour_error(pred_hour: np.ndarray, true_hour: np.ndarray) -> np.ndarray:
    diff = np.abs(pred_hour - true_hour)
    return np.minimum(diff, 24.0 - diff)


# -----------------------------------------------------------------------------
# KROK D: Definicja modeli do porównania.
# - ridge: szybki liniowy baseline (zawsze warto mieć)
# - hgb:  zwykle najlepszy stosunek jakość/czas na cechach tablicowych
# - rf:   mocny, ale bywa wolniejszy (szczególnie na 471k próbek)
#
# Uwaga: Uczymy MultiOutputRegressor, bo mamy dwa cele: sin i cos.
# -----------------------------------------------------------------------------
def make_models():
    return [
        (
            "ridge",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("reg", MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))),
                ]
            ),
        ),
        (
            "hgb",
            MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    random_state=42,
                )
            ),
        ),
        (
            "rf",
            MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=None,
                    min_samples_leaf=1,
                )
            ),
        ),
    ]


def main() -> None:
    # -----------------------------------------------------------------------------
    # KROK 0: start stopera całego programu
    # -----------------------------------------------------------------------------
    t_program_start = time.time()
    log("Start: regresja cykliczna godziny (sin/cos)")

    # -----------------------------------------------------------------------------
    # KROK 1: sprawdzenie pliku z cechami
    # -----------------------------------------------------------------------------
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku {FEATURES_PATH}. Najpierw uruchom: python -m src.precompute_features_robust"
        )

    # -----------------------------------------------------------------------------
    # KROK 2: wczytanie cech
    # -----------------------------------------------------------------------------
    log(f"Wczytywanie danych z: {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)
    log(f"Wczytano dataframe: shape={df.shape}")

    # -----------------------------------------------------------------------------
    # KROK 3: wybór kolumn cech (odrzucamy metadane)
    # -----------------------------------------------------------------------------
    drop_cols = {"filepath", "datetime", "hour"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    if "hour" not in df.columns:
        raise ValueError("Brak kolumny 'hour' w features_robust.csv")

    log(f"Liczba cech wejściowych: {len(feature_cols)}")
    log(f"Przykładowe cechy: {feature_cols[:10]}{' ...' if len(feature_cols) > 10 else ''}")

    # -----------------------------------------------------------------------------
    # KROK 4: budowa macierzy X oraz wektora prawdziwych godzin
    # -----------------------------------------------------------------------------
    log("Budowa macierzy X (float32) i wektora y_hour")
    X = df[feature_cols].astype(np.float32).to_numpy()
    y_hour = df["hour"].astype(np.float32).to_numpy()
    log(f"X shape={X.shape}, y_hour shape={y_hour.shape}")

    # -----------------------------------------------------------------------------
    # KROK 5: transformacja celu do (sin, cos)
    # -----------------------------------------------------------------------------
    log("Transformacja celu: hour -> (sin, cos)")
    y_sc = hour_to_sin_cos(y_hour)
    log(f"y_sc shape={y_sc.shape}")

    # -----------------------------------------------------------------------------
    # KROK 6: split train/test
    # Stratify po godzinie (int), żeby rozkład godzin był podobny w train i test.
    # -----------------------------------------------------------------------------
    log("Podział train/test (80/20) ze stratify po godzinie")
    X_train, X_test, y_train_sc, y_test_sc, hour_train, hour_test = train_test_split(
        X,
        y_sc,
        y_hour,
        test_size=0.2,
        random_state=42,
        stratify=df["hour"].astype(int),
    )
    log(f"Train: X={X_train.shape}, Test: X={X_test.shape}")

    # -----------------------------------------------------------------------------
    # KROK 7: trening i porównanie modeli
    # Metryki:
    # - Cyclic MAE (h): średni błąd w godzinach na okręgu 24h (najważniejsza)
    # - Naive MAE (h): zwykłe MAE (mniej sensowne przy cyklu, ale bywa informacyjne)
    # - P90/P95 cyclic error: pokazuje ogon błędów (czy czasem mocno się myli)
    # -----------------------------------------------------------------------------
    best = None  # (cmae, name)

    for name, model in make_models():
        log("-" * 80)
        log(f"Trening modelu: {name}")

        t_fit_start = time.time()

        # --- BLOK AKWIZYCJI: dopasowanie modelu do danych ---
        model.fit(X_train, y_train_sc)

        fit_time = time.time() - t_fit_start
        log(f"Koniec fit: {fit_time:.2f}s")

        # --- Predykcja na teście ---
        log("Predykcja na zbiorze testowym")
        pred_sc = model.predict(X_test)

        # --- Konwersja (sin,cos) -> godzina ---
        pred_hour = sin_cos_to_hour(pred_sc)

        # --- Metryki błędu ---
        circ_err = circular_hour_error(pred_hour, hour_test)
        cmae = float(circ_err.mean())
        mae_naive = float(mean_absolute_error(hour_test, pred_hour))
        p90 = float(np.percentile(circ_err, 90))
        p95 = float(np.percentile(circ_err, 95))

        log(f"Wyniki [{name}]")
        print(f"  Cyclic MAE (h): {cmae:.3f}", flush=True)
        print(f"  Naive  MAE (h): {mae_naive:.3f}", flush=True)
        print(f"  P90 cyclic err (h): {p90:.3f}", flush=True)
        print(f"  P95 cyclic err (h): {p95:.3f}", flush=True)
        print(f"  Fit time: {fit_time:.2f}s", flush=True)

        if best is None or cmae < best[0]:
            best = (cmae, name)

    # -----------------------------------------------------------------------------
    # KROK 8: podsumowanie + czas całkowity
    # -----------------------------------------------------------------------------
    total_time = time.time() - t_program_start
    log("=" * 80)
    log(f"Najlepszy model wg Cyclic MAE: {best[1]} (Cyclic MAE={best[0]:.3f}h)")
    log(f"Całkowity czas wykonania: {total_time:.2f}s")


if __name__ == "__main__":
    main()
