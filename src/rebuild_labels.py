# rebuild_labels.py
# Run: python -m src.rebuild_labels [--data-dir path/to/dataset] [--out path/to/labels.csv] [--ext .jpg .png] [--strict-structure] [--check-hour-vs-folder]

# Opis:
#     Skrypt do odbudowy pliku labels.csv na podstawie aktualnej zawartości katalogu dataset.
#     - Rekurencyjnie przechodzi po drzewie katalogów DATA_DIR
#     - Szuka plików o zadanych rozszerzeniach (domyślnie .jpg, .jpeg, .png)
#     - Odczytuje datę i czas z nazwy pliku w formacie YYYYMMDD_HHMMSS (jak w clock.py)
#     - Wylicza godzinę na podstawie timestampu i zapisuje ją jako etykietę
#     - (Opcjonalnie) weryfikuje spójność między nazwą pliku a folderem HH i raportuje rozjazdy

# Zadania realizowane przez skrypt:
#     1. Wczytanie konfiguracji ścieżek z src.settings (DATA_DIR, LABELS_CSV)
#     2. Parsowanie argumentów CLI (nadpisanie katalogu danych, wyjściowego CSV, listy rozszerzeń, trybu strict)
#     3. Przechodzenie po plikach w katalogu dataset i filtrowanie po rozszerzeniach
#     4. Ekstrakcja daty/godziny z nazw plików i opcjonalne sprawdzenie zgodności z folderem HH
#     5. Posortowanie rekordów i zapis nowego labels.csv (filepath, hour, datetime)

from __future__ import annotations

import re
import csv
from pathlib import Path
from datetime import datetime
import argparse

from src.settings import DATA_DIR, LABELS_CSV


NAME_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})\.(?P<ext>jpg|jpeg|png)$", re.IGNORECASE)


def resolve_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def parse_args():
    p = argparse.ArgumentParser(description="Odbudowa labels.csv po czyszczeniu datasetu (format clock.py)")
    p.add_argument("--data-dir", type=Path, default=None, help="(opcjonalnie) nadpisz DATA_DIR")
    p.add_argument("--out", type=Path, default=None, help="(opcjonalnie) nadpisz LABELS_CSV")
    p.add_argument("--ext", nargs="*", default=[".jpg", ".jpeg", ".png"], help="Rozszerzenia obrazów")
    p.add_argument(
        "--strict-structure",
        action="store_true",
        help="Wymuś strukturę YYYY/MM/DD/HH; pomijaj wszystko inne",
    )
    p.add_argument(
        "--check-hour-vs-folder",
        action="store_true",
        help="Sprawdź czy godzina w nazwie pliku == folder HH; raportuj rozjazdy",
    )
    return p.parse_args()


def looks_like_strict_structure(rel_path: Path) -> bool:
    parts = rel_path.parts
    if len(parts) < 5:
        return False
    yyyy, mm, dd, hh = parts[0], parts[1], parts[2], parts[3]
    return (
        len(yyyy) == 4 and yyyy.isdigit()
        and len(mm) == 2 and mm.isdigit()
        and len(dd) == 2 and dd.isdigit()
        and len(hh) == 2 and hh.isdigit()
    )


def hour_from_folder(rel_path: Path) -> int | None:
    parts = rel_path.parts
    if len(parts) < 5:
        return None
    hh = parts[3]
    if not (len(hh) == 2 and hh.isdigit()):
        return None
    h = int(hh)
    return h if 0 <= h <= 23 else None


def dt_from_name(filename: str) -> datetime | None:
    m = NAME_RE.match(filename)
    if not m:
        return None
    ts = f"{m.group('date')}_{m.group('time')}"
    return datetime.strptime(ts, "%Y%m%d_%H%M%S")


def main():
    args = parse_args()

    print("== rebuild_labels: start ==")
    print("Settings loaded")
    print(f"DATA_DIR = {DATA_DIR}")
    print(f"LABELS_CSV = {LABELS_CSV}")

    # ✅ domyślnie bierzemy z settings.py, CLI tylko nadpisuje
    data_dir = resolve_path(args.data_dir) if args.data_dir else DATA_DIR.expanduser().resolve()
    out_csv = resolve_path(args.out) if args.out else LABELS_CSV.expanduser().resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Nie istnieje data-dir: {data_dir}")

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext}

    rows: list[tuple[str, int, str]] = []
    total_images = 0
    skipped_structure = 0
    skipped_badname = 0
    skipped_nohour = 0
    hour_mismatch = 0
    mismatch_examples: list[tuple[str, int, int]] = []
    
    print("Reading input data...")
    
    for path in data_dir.rglob("*"):
        print(f"Processed {path} records")
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue

        total_images += 1
        rel = path.relative_to(data_dir)

        if args.strict_structure and not looks_like_strict_structure(rel):
            skipped_structure += 1
            continue

        # ✅ nazwy muszą być jak z clock.py (bo powiedziałeś, że wszystkie takie są)
        dt = dt_from_name(path.name)
        if dt is None:
            skipped_badname += 1
            continue

        hour_from_name = dt.hour
        folder_hour = hour_from_folder(rel)

        if folder_hour is None and args.strict_structure:
            skipped_nohour += 1
            continue

        if args.check_hour_vs_folder and folder_hour is not None:
            if folder_hour != hour_from_name:
                hour_mismatch += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append((rel.as_posix(), folder_hour, hour_from_name))

        rows.append((rel.as_posix(), hour_from_name, dt.isoformat()))

    rows.sort(key=lambda r: r[0])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "hour", "datetime"])
        w.writerows(rows)

    print(f"DATA_DIR   : {data_dir}")
    print(f"OUT_CSV    : {out_csv}")
    print(f"Obrazy     : {total_images}")
    print(f"Zapisane   : {len(rows)}")
    if args.strict_structure:
        print(f"Pominięte (zła struktura): {skipped_structure}")
        print(f"Pominięte (brak HH): {skipped_nohour}")
    print(f"Pominięte (zła nazwa): {skipped_badname}")
    if args.check_hour_vs_folder:
        print(f"Rozjazdy hour (folder vs nazwa): {hour_mismatch}")
        if mismatch_examples:
            print("Przykłady (plik, folder_HH, name_HH):")
            for ex in mismatch_examples:
                print("  ", ex)


if __name__ == "__main__":
    main()
