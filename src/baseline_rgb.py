# src/baseline_rgb.py
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

FEATURES_PATH = Path("features_mean_rgb.csv")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku {FEATURES_PATH}. "
            f"Najpierw uruchom: python -m src.precompute_mean_rgb"
        )

    df = pd.read_csv(FEATURES_PATH)
    print("Wczytano cechy:", df.shape)

    X = df[["r_mean", "g_mean", "b_mean"]].to_numpy(dtype=np.float32)
    y = df["hour"].to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    clf = LogisticRegression(
        max_iter=5000,
        n_jobs=-1,
        multi_class="auto",
        verbose=1,
    )

    clf.fit(X_train, y_train)

    # Zapisz wytrenowany model do pliku pickle
    import pickle
    models_path = Path(__file__).resolve().parent.parent / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    model_path = models_path / "baseline_rgb_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model zapisano do pliku: {model_path}")

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n=== Wyniki modelu bazowego (średnie RGB z cache) ===")
    print("Accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
