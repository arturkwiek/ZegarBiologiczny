# src/precompute_features_robust.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .load_data import load_labels
from src.settings import DATA_DIR, LABELS_CSV
from .utils_robust import extract_robust_image_features

OUTPUT_PATH = Path("features_robust.csv")
DATA_ROOT = Path(DATA_DIR)


def main() -> None:
    # --- KROK 1: Wczytanie labels.csv ---
    labels_path = LABELS_CSV or Path("labels.csv")
    df = load_labels(labels_path)
    print("Liczba rekordów w labels.csv:", len(df))

    rows = []


    # --- KROK 2: Iteracja po obrazach i ekstrakcja cech ---
    pbar = tqdm(df.iterrows(), total=len(df))
    for _, row in pbar:
        rel_path = str(row["filepath"])
        full_path = DATA_ROOT / rel_path

        # --- BLOK AKWIZYCJI: pobierz cechy z obrazu ---
        feats = extract_robust_image_features(full_path)
        if feats is None:
            continue

        # --- KROK 3: Zbuduj rekord (cechy + metadane) ---
        record = {
            "filepath": rel_path,
            "datetime": row["datetime"],
            "hour": int(row["hour"]),
        }
        record.update(feats)
        rows.append(record)


    # --- KROK 4: Zapis do CSV ---
    out_df = pd.DataFrame(rows)
    print("Gotowe cechy:", out_df.shape)

    out_df.to_csv(OUTPUT_PATH, index=False)
    print("Zapisano do:", OUTPUT_PATH)

    # Podsumowanie czasu z tqdm
    elapsed = pbar.format_dict['elapsed']
    import datetime
    print("Czas przetwarzania (tqdm):", str(datetime.timedelta(seconds=int(elapsed))))


if __name__ == "__main__":
    main()
