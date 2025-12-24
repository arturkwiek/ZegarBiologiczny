# src/precompute_mean_rgb.py — Run: python -m src.precompute_mean_rgb
# 
# Opis:
#     Skrypt do ekstrakcji i zapisu średnich wartości RGB z obrazów na podstawie pliku labels.csv.
#     Dla każdego obrazu:
#       - Wczytuje ścieżki i metadane z labels.csv
#       - Wylicza średnie wartości kanałów R, G, B (funkcja extract_mean_rgb)
#       - Zapisuje wynikowe cechy do pliku features_mean_rgb.csv
#       - Wyświetla postęp przetwarzania (tqdm) oraz podsumowanie czasu

# Zadania realizowane przez skrypt:
#     1. Wczytanie etykiet i ścieżek do obrazów
#     2. Ekstrakcja średnich wartości RGB dla każdego obrazu
#     3. Zapis cech do pliku CSV
#     4. Podsumowanie czasu przetwarzania


from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .load_data import load_labels
from src.settings import DATA_DIR, LABELS_CSV

from .utils import extract_mean_rgb

OUTPUT_PATH = Path("features_mean_rgb.csv")


def main() -> None:
    labels_path = LABELS_CSV or Path("labels.csv")
    df = load_labels(labels_path)
    print("Liczba rekordów w labels.csv:", len(df))


    rows = []
    pbar = tqdm(df.iterrows(), total=len(df))
    for _, row in pbar:
        rel_path = str(row["filepath"])
        full_path = DATA_DIR / rel_path

        feat = extract_mean_rgb(full_path)
        if feat is None:
            # np. plik nie istnieje / nie udało się wczytać
            continue

        r_mean, g_mean, b_mean = feat

        rows.append(
            {
                "filepath": rel_path,
                "datetime": row["datetime"],
                "hour": int(row["hour"]),
                "r_mean": float(r_mean),
                "g_mean": float(g_mean),
                "b_mean": float(b_mean),
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
