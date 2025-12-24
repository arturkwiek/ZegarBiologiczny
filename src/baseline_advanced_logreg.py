# src/baseline_advanced_logreg.py — Run: python -m src.baseline_advanced_logreg
# baseline_advanced_logreg.py

# Opis:
#     Skrypt do trenowania bazowego modelu klasyfikacji godzin na podstawie rozszerzonych cech (features_advanced.csv).
#     - Wczytuje cechy z features_advanced.csv
#     - Wybiera wszystkie kolumny numeryczne poza "hour" jako cechy wejściowe
#     - Dzieli dane na zbiór treningowy i testowy
#     - Trenuje model LogisticRegression w pipeline ze skalowaniem (StandardScaler)
#     - Zapisuje wytrenowany model do pliku baseline_advanced_logreg_model.pkl
#     - Wyświetla metryki klasyfikacji (accuracy, classification report, confusion matrix)

# Zadania realizowane przez skrypt:
#     1. Wczytanie cech rozszerzonych z pliku CSV
#     2. Przygotowanie macierzy cech i etykiet godzin
#     3. Podział na zbiór treningowy i testowy
#     4. Trening modelu LogisticRegression (z normalizacją cech)
#     5. Zapis modelu do pliku w katalogu models/
#     6. Wyświetlenie metryk jakości na zbiorze testowym

from pathlib import Path

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES_PATH = Path("features_advanced.csv")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku {FEATURES_PATH}. "
            f"Najpierw uruchom: python -m src.precompute_features_advanced"
        )

    logging.info("Wczytywanie cech z pliku CSV...")
    df = pd.read_csv(FEATURES_PATH)
    logging.info(f"Wczytano cechy rozszerzone: {df.shape}")

    if "hour" not in df.columns:
        raise ValueError("Brak kolumny 'hour' w pliku features_advanced.csv")

    # --- wybór cech: tylko kolumny numeryczne ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "hour"]

    if not feature_cols:
        raise ValueError("Nie znaleziono żadnych cech numerycznych")

    logging.info(f"Użyte cechy: {feature_cols}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["hour"].to_numpy(dtype=np.int64)

    logging.info("Podział na zbiór treningowy i testowy...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    logging.info(f"Rozmiar zbioru treningowego: {X_train.shape}, testowego: {X_test.shape}")

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=5000,
                    n_jobs=-1,
                    multi_class="auto",
                    verbose=1,
                ),
            ),
        ]
    )

    logging.info("Rozpoczynam trenowanie modelu LogisticRegression...")
    clf.fit(X_train, y_train)
    logging.info("Trenowanie zakończone.")

    # Zapisz wytrenowany model do katalogu 'models' w katalogu głównym repozytorium
    import pickle
    models_path = Path(__file__).resolve().parent.parent / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    model_path = models_path / "baseline_advanced_logreg_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model zapisano do pliku: {model_path}")

    logging.info("Predykcja na zbiorze testowym...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n=== Wyniki modelu bazowego (cechy advanced + LogisticRegression) ===")
    print("Accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
