import csv
import tarfile
from io import BytesIO
from pathlib import Path

"""
Skrypt do przygotowania zbioru w postaci 24 archiwów TAR (po jednym na każdą
godzinę), zgodnie z bieżącą strukturą projektu.

Założenia dostosowane do projektu:
- Używamy konfiguracji z src.settings (DATA_DIR, LABELS_CSV)
- Plik etykiet to domyślnie dataset/2025/labels.csv
- Kolumna "filepath" w CSV jest ŚCIEŻKĄ WZGLĘDNĄ względem DATA_DIR
    (np. "11/30/12/20251130_124132.jpg")
- Wyjściowe archiwa zapisujemy w podkatalogu DATA_DIR / "shards_by_hour"
"""

from src.settings import DATA_DIR, LABELS_CSV

# Gdzie zapisać wynikowe archiwa
OUT_DIR = DATA_DIR / "shards_by_hour"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ścieżki z CSV są względne względem DATA_DIR
BASE_DIR = DATA_DIR

def main() -> None:
    # Otwórz 24 tary naraz (bez kompresji)
    tars: dict[int, tarfile.TarFile] = {}
    counters: dict[int, int] = {h: 0 for h in range(24)}

    for h in range(24):
        tars[h] = tarfile.open(OUT_DIR / f"hour_{h:02d}.tar", mode="w")

    missing = 0
    total = 0

    labels_path = LABELS_CSV or Path("labels.csv")
    if not labels_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku z etykietami: {labels_path}")

    print(f"Używany plik etykiet: {labels_path}")
    print(f"Katalog danych (BASE_DIR): {BASE_DIR}")
    print(f"Katalog wyjściowy shardów: {OUT_DIR}")

    try:
        with open(labels_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1

                try:
                    hour = int(row["hour"])
                except (KeyError, ValueError):
                    # Pomijamy wiersze bez poprawnej godziny
                    continue

                if hour not in range(24):
                    # Nietypowa godzina – pomijamy
                    continue

                img_path = BASE_DIR / row["filepath"]

                if not img_path.exists():
                    missing += 1
                    continue

                # Unikalna nazwa w obrębie danej godziny
                idx = counters[hour]
                counters[hour] += 1

                jpg_name = f"{idx:09d}.jpg"
                cls_name = f"{idx:09d}.cls"

                # Dodaj jpg (bez wczytywania całego do RAM)
                tars[hour].add(img_path, arcname=jpg_name)

                # Dodaj etykietę jako mały plik tekstowy
                label_bytes = f"{hour}\n".encode("utf-8")
                info = tarfile.TarInfo(name=cls_name)
                info.size = len(label_bytes)
                tars[hour].addfile(info, BytesIO(label_bytes))

                if total % 50000 == 0:
                    print(
                        f"Przetworzono {total:,} wierszy... "
                        f"brakujących plików: {missing:,}"
                    )

    finally:
        for tar in tars.values():
            tar.close()

    print("DONE")
    print("Liczność na godzinę:", counters)
    print("Brakujące pliki:", missing)


if __name__ == "__main__":
    main()