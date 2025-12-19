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
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n=== Wyniki modelu bazowego (średnie RGB z cache) ===")
    print("Accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, y_pred))

    num_epochs = 10  # example value, set as needed
    train_loader = None  # replace with actual data loader

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs} started.")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # train step
            loss = None  # replace with actual loss computation
            running_loss += loss.item()
            if i % 10 == 9:  # log every 10 batches
                logging.info(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 10:.4f}")
                running_loss = 0.0
        logging.info(f"Epoch {epoch+1} finished.")

    logging.info("Training completed.")


if __name__ == "__main__":
    main()
