# src/baseline_advanced.py — Run: python -m src.baseline_advanced

# Opis:
#     Skrypt do trenowania bazowego modelu klasyfikacji godzin na podstawie rozszerzonych cech obrazu
#     (features_advanced.csv: mean/std RGB + statystyki HSV z kanałów H/S/V).
#     - Wczytuje cechy z features_advanced.csv
#     - Używa wybranych kolumn cech (r_mean, g_mean, b_mean, r_std, g_std, b_std, s_mean, v_mean)
#     - Dzieli dane na zbiór treningowy i testowy
#     - Trenuje kilka modeli klasyfikacyjnych (LogisticRegression, KNN, RandomForest, GradientBoosting)
#     - Wybiera model o najwyższym accuracy i wyświetla szczegółowe metryki

# Zadania realizowane przez skrypt:
#     1. Wczytanie cech rozszerzonych i przygotowanie macierzy X oraz etykiet godzin y
#     2. Podział danych na zbiory treningowy i testowy (stratyfikacja po godzinach)
#     3. Zbudowanie zestawu modeli bazowych (build_models)
#     4. Trening i ocena każdego modelu oraz wybór najlepszego wg accuracy
#     5. Wypisanie classification report i macierzy pomyłek dla najlepszego modelu

from __future__ import annotations

from pathlib import Path

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES_PATH = Path("features_advanced.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)


def load_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku {FEATURES_PATH}. "
            f"Najpierw uruchom: python -m src.precompute_features_advanced"
        )

    logging.info("Wczytywanie cech z pliku CSV...")
    df = pd.read_csv(FEATURES_PATH)
    logging.info(f"Wczytano cechy rozszerzone: {df.shape}")

    # Pełny zestaw cech (nowsza wersja może mieć dodatkowo h_mean, h_std)
    all_feature_cols = [
        "r_mean",
        "g_mean",
        "b_mean",
        "r_std",
        "g_std",
        "b_std",
        "h_mean",
        "h_std",
        "s_mean",
        "v_mean",
    ]

    # Dostosuj się do rzeczywistego nagłówka CSV (na starszych wersjach brak h_mean/h_std)
    present_cols = [c for c in all_feature_cols if c in df.columns]
    missing_cols = [c for c in all_feature_cols if c not in df.columns]
    if missing_cols:
        logging.warning(
            "Brak kolumn %s w features_advanced.csv – trenowanie tylko na: %s",
            missing_cols,
            present_cols,
        )

    if not present_cols:
        raise ValueError("Brak jakichkolwiek oczekiwanych kolumn cech w features_advanced.csv")

    X = df[present_cols].to_numpy(dtype=np.float32)
    y = df["hour"].to_numpy(dtype=np.int64)

    return X, y


def build_models():
    """
    Zwraca słownik nazw -> modeli do porównania.
    """

    models = {}

    # 1) Logistic Regression na zeskalowanych cechach
    models["logreg"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                # multi_class="auto" bywa domyślne, ale w nowszych wersjach
                # LogisticRegression nie zawsze przyjmuje ten parametr –
                # usuwamy go dla kompatybilności wstecznej.
                LogisticRegression(
                    max_iter=5000,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 2) KNN - też potrzebuje skalowania
    models["knn"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]
    )

    # 3) Random Forest - wersja odchudzona (mniejszy model, lepszy na RPi)
    models["rf"] = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )

    # 4) Gradient Boosting - wolniejszy, ale często dobry
    models["gb"] = GradientBoostingClassifier(
        random_state=42,
    )

    return models


def main() -> None:

    import time
    start = time.time()

    # 1. Wczytanie cech
    logging.info("Rozpoczynam wczytywanie cech...")
    X, y = load_features()
    logging.info("Cechy wczytane.")

    # 2. Podział na train/test
    logging.info("Podział na zbiór treningowy i testowy...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    logging.info(f"Rozmiar zbioru treningowego: {X_train.shape}, testowego: {X_test.shape}")

    models = build_models()

    results = {}

    # 3. Trening i ocena każdego modelu
    import pickle
    from pathlib import Path
    # Modele bazowe zapisujemy domyślnie do katalogu PC
    models_path = Path(__file__).resolve().parent.parent / "models" / "pc"
    models_path.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        logging.info(f"Trenowanie modelu: {name}")
        print(f"\n=== Trenowanie modelu: {name} ===")
        model.fit(X_train, y_train)
        logging.info(f"Model {name} wytrenowany.")

        # Zapisz wytrenowany model do pliku pickle
        model_path = models_path / f"baseline_advanced_{name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model {name} zapisano do pliku: {model_path}")

        y_pred = model.predict(X_test)
        logging.info(f"Predykcja zakończona dla modelu: {name}")

        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            "acc": acc,
            "y_pred": y_pred,
        }
        print(f"Accuracy ({name}): {acc:.4f}")

    # 4. Wybór najlepszego modelu po accuracy
    best_name = max(results.keys(), key=lambda n: results[n]["acc"])
    best_acc = results[best_name]["acc"]
    best_pred = results[best_name]["y_pred"]

    logging.info(f"Najlepszy model: {best_name} (accuracy = {best_acc:.4f})")

    print("\n" + "#" * 80)
    print(f"Najlepszy model: {best_name} (accuracy = {best_acc:.4f})")
    print("#" * 80)

    print("\nPełny raport dla najlepszego modelu:\n")
    print("Classification report:")
    print(classification_report(y_test, best_pred))

    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, best_pred))

    # Podsumowanie czasu trenowania
    elapsed = time.time() - start
    import datetime
    print("Czas trenowania:", str(datetime.timedelta(seconds=int(elapsed))))


if __name__ == "__main__":
    main()
