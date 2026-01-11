# MLDailyHourClock.py — Run: python MLDailyHourClock.py
# MLDailyHourClock.py

# Opis:
#     Skrypt do ciągłego zbierania obrazów z kamerki i zapisywania ich w strukturze katalogów
#     dataset/YYYY/MM/DD/HH wraz z etykietą godziny w pliku labels.csv.
#     - Otwiera kamerę (CAMERA_INDEX)
#     - Co sekundę zapisuje klatkę do odpowiedniego podkatalogu godzinowego
#     - Uzupełnia/zakłada plik labels.csv ze ścieżką względną, godziną i timestampem
#     - Tworzy brakujące katalogi i pliki (dataset/, labels.csv)
#     - Zatrzymanie skryptu: Ctrl+C w terminalu

# Zadania realizowane przez skrypt:
#     1. Przygotowanie katalogów dataset oraz pliku labels.csv
#     2. Otwarcie strumienia z kamery i obsługa błędów
#     3. Okresowe zapisywanie obrazów w strukturze YYYY/MM/DD/HH
#     4. Dopisywanie rekordów do labels.csv (filepath, hour, datetime)
#     5. Łagodne zakończenie pracy po przerwaniu przez użytkownika

import argparse
import cv2
import time
from datetime import datetime
from pathlib import Path

from src.settings import DATA_DIR, LABELS_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="Indeks kamerki (zwykle 0)")
    parser.add_argument("--width", type=int, default=1280, help="Sugerowana szerokosc obrazu")
    parser.add_argument("--height", type=int, default=720, help="Sugerowana wysokosc obrazu")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Odstep czasu miedzy kolejnymi zdjeciami [s] (domyslnie 1.0)",
    )
    parser.add_argument(
        "--temp-copy",
        type=str,
        default="~/www/camera_hour.jpg",
        help="Sciezka do pliku tymczasowego z kopia zdjecia (pusty napis aby wylaczyc)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir: Path = DATA_DIR.resolve()
    csv_path: Path = LABELS_CSV.resolve()
    temp_copy_path: Path | None = None
    if args.temp_copy:
        temp_copy_path = Path(args.temp_copy).expanduser().resolve()

    # --- przygotowanie katalogów ---
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_copy_path is not None:
        temp_copy_path.parent.mkdir(parents=True, exist_ok=True)

    # --- kamera ---
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"❌ Nie można otworzyć kamery o indeksie {args.cam}.")
        return

    # próbujemy wymusić rozdzielczość (nie każda kamerka to respektuje)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))

    print("▶ Start zbierania danych (Ctrl+C aby przerwać)")
    print(f"Dataset : {output_dir}")
    print(f"Labels  : {csv_path}")

    # --- CSV nagłówek ---
    if not csv_path.exists():
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("filepath,hour,datetime\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Błąd odczytu z kamery.")
                break

            now = datetime.now()

            year = now.strftime("%Y")
            month = now.strftime("%m")
            day = now.strftime("%d")
            hour_int = now.hour
            hour_str = f"{hour_int:02d}"
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            # dataset/YYYY/MM/DD/HH/
            hour_dir = output_dir / year / month / day / hour_str
            hour_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{timestamp}.jpg"
            full_path = hour_dir / filename

            # zapis obrazu
            cv2.imwrite(str(full_path), frame)

            # 👉 ZAPIS ŚCIEŻKI WZGLĘDNEJ
            relative_path = full_path.relative_to(output_dir)

            with csv_path.open("a", encoding="utf-8") as f:
                f.write(f"{relative_path.as_posix()},{hour_int},{now.isoformat()}\n")

            print(f"✔ {relative_path} (godzina={hour_int})")

            if temp_copy_path is not None:
                suffix = temp_copy_path.suffix or ".jpg"
                base_name = temp_copy_path.stem if temp_copy_path.suffix else temp_copy_path.name
                tmp_copy = temp_copy_path.parent / f".{base_name}.tmp{suffix}"
                cv2.imwrite(str(tmp_copy), frame)
                tmp_copy.replace(temp_copy_path)

            time.sleep(max(0.0, float(args.interval)))

    except KeyboardInterrupt:
        print("\n⏹ Przerwano przez użytkownika.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✔ Kamera zwolniona, koniec pracy.")


if __name__ == "__main__":
    main()
