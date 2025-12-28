# src/precompute_features_advanced.py — Run: python -m src.precompute_features_advanced
# precompute_features_advanced.py

# Opis:
#     Skrypt do ekstrakcji i zapisu rozszerzonych cech RGB/HSV z obrazów na podstawie pliku labels.csv.
#     Dla każdego obrazu:
#       - Wczytuje ścieżki i metadane z labels.csv
#       - Wylicza cechy: średnie i odchylenia RGB oraz statystyki H/S/V (funkcja extract_rgb_hsv_stats)
#       - Zapisuje wynikowe cechy do pliku features_advanced.csv
#       - Wyświetla postęp przetwarzania (tqdm) oraz podsumowanie czasu

# Zadania realizowane przez skrypt:
#     1. Wczytanie etykiet i ścieżek do obrazów
#     2. Ekstrakcja cech RGB/HSV dla każdego obrazu
#     3. Zapis cech do pliku CSV
#     4. Podsumowanie czasu przetwarzania

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .load_data import load_labels
from src.settings import DATA_DIR, LABELS_CSV
from .utils import extract_rgb_hsv_stats

OUTPUT_PATH = Path("features_advanced.csv")
DATA_ROOT = Path(DATA_DIR)


def main() -> None:
    labels_path = LABELS_CSV or Path("labels.csv")
    df = load_labels(labels_path)
    print("Liczba rekordów w labels.csv:", len(df))


    rows = []
    pbar = tqdm(df.iterrows(), total=len(df))
    for _, row in pbar:
        rel_path = str(row["filepath"])
        full_path = DATA_ROOT / rel_path

        feat = extract_rgb_hsv_stats(full_path)
        if feat is None:
            continue

        (
            r_mean,
            g_mean,
            b_mean,
            r_std,
            g_std,
            b_std,
            h_mean,
            h_std,
            s_mean,
            v_mean,
        ) = feat.tolist()

        rows.append(
            {
                "filepath": rel_path,
                "datetime": row["datetime"],
                "hour": int(row["hour"]),
                "r_mean": float(r_mean),
                "g_mean": float(g_mean),
                "b_mean": float(b_mean),
                "r_std": float(r_std),
                "g_std": float(g_std),
                "b_std": float(b_std),
                "h_mean": float(h_mean),
                "h_std": float(h_std),
                "s_mean": float(s_mean),
                "v_mean": float(v_mean),
            }
        )


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
