# src/baseline_advanced_logreg.py
from pathlib import Path

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


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku {FEATURES_PATH}. "
            f"Najpierw uruchom: python -m src.precompute_features_advanced"
        )

    df = pd.read_csv(FEATURES_PATH)
    print("Wczytano cechy rozszerzone:", df.shape)

    # wszystkie cechy oprócz etykiety
    feature_cols = [c for c in df.columns if c != "hour"]

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["hour"].to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # Pipeline = scaler + ten sam model co w RGB
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=5000,
                    n_jobs=-1,
                    multi_class="auto",
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n=== Wyniki modelu bazowego (cechy advanced) ===")
    print("Accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
