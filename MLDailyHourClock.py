import cv2
import os
import time
from datetime import datetime

# Konfiguracja
OUTPUT_DIR = "dataset"     # katalog bazowy na dane
CSV_PATH = "labels.csv"    # plik z etykietami (ścieżka_do_pliku,godzina,datetime)
CAMERA_INDEX = 0           # indeks kamery (0 – zazwyczaj domyślna kamera)

def main():
    # Utworzenie katalogu bazowego, jeśli nie istnieje
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Inicjalizacja kamery
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Nie można otworzyć kamery. Sprawdź podłączenie i indeks kamery.")
        return

    print("Start zbierania danych. Przerwij działanie skryptu Ctrl+C.")

    # Utworzenie pliku CSV z nagłówkiem, jeśli nie istnieje
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", encoding="utf-8") as f:
            f.write("filepath,hour,datetime\n")

    try:
        while True:
            # Pobranie klatki z kamery
            ret, frame = cap.read()
            if not ret:
                print("Błąd odczytu z kamery.")
                break

            now = datetime.now()

            year = now.strftime("%Y")
            month = now.strftime("%m")
            day = now.strftime("%d")
            hour_int = now.hour              # etykieta: godzina jako int 0–23
            hour_str = f"{hour_int:02d}"     # wersja z zerem wiodącym
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            # Struktura katalogów:
            # dataset/YYYY/MM/DD/HH/
            hour_dir = os.path.join(OUTPUT_DIR, year, month, day, hour_str)
            os.makedirs(hour_dir, exist_ok=True)

            # nazwa pliku: np. 20251129_153045.jpg
            filename = f"{timestamp}.jpg"
            filepath = os.path.join(hour_dir, filename)

            # zapis obrazu
            cv2.imwrite(filepath, frame)

            # dopisanie etykiety do CSV
            # datetime w ISO ułatwia później analizy (np. filtr po dacie)
            with open(CSV_PATH, "a", encoding="utf-8") as f:
                f.write(f"{filepath},{hour_int},{now.isoformat()}\n")

            print(f"Zapisano: {filepath} (godzina={hour_int})")

            # czekamy ~1 sekundę do kolejnego zrzutu
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika. Kończenie...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Kamera zwolniona, koniec pracy.")

if __name__ == "__main__":
    main()
