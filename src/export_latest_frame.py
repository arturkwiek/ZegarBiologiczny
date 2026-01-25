# src/export_latest_frame.py — Run: python -m src.export_latest_frame
#
# Opis:
#     Skrypt do kopiowania najnowszego obrazu z datasetu do katalogu WWW.
#     - Wyszukuje najświeższy plik JPG zapisany przez MLDailyHourClock
#     - Co zadany interwał kopiuje go pod wskazaną ścieżkę (np. www/camera_hour.jpg)
#     - Używa zapisu atomowego (tmp + rename), aby uniknąć częściowo skopiowanych plików
#
# Zadania realizowane przez skrypt:
#     1. Odnajdywanie najnowszej klatki na podstawie struktury YYYY/MM/DD/HH
#     2. Monitorowanie datasetu w pętli ze stałym interwałem
#     3. Kopiowanie ostatniego obrazu do katalogu WWW w formie pojedynczego pliku

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

from src.settings import DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DATA_DIR,
        help="Katalog datasetu - domyślnie DATA_DIR z settings.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./www/camera_hour.jpg"),
        help="Ścieżka docelowa (np. ~/www/camera_hour.jpg)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Czas między kolejnymi próbami (sekundy)",
    )
    return parser.parse_args()


def _iter_dirs_desc(path: Path) -> Iterable[Path]:
    try:
        dirs = [p for p in path.iterdir() if p.is_dir()]
    except FileNotFoundError:
        return []
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def _find_latest_image(source_dir: Path) -> Optional[Path]:
    """Zwraca najnowszy plik JPG z datasetu albo None, jeśli brak."""
    if not source_dir.exists():
        return None

    for year_dir in _iter_dirs_desc(source_dir):
        for month_dir in _iter_dirs_desc(year_dir):
            for day_dir in _iter_dirs_desc(month_dir):
                for hour_dir in _iter_dirs_desc(day_dir):
                    try:
                        candidates = sorted(hour_dir.glob("*.jpg"), reverse=True)
                    except FileNotFoundError:
                        continue
                    if candidates:
                        return candidates[0]
    return None


def _copy_atomically(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp_path)
    tmp_path.replace(dst)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser()
    output_path = args.output.expanduser()
    interval = max(0.1, float(args.interval))

    print(f"[INFO] Monitoruję katalog: {source_dir}")
    print(f"[INFO] Docelowy plik WWW: {output_path}")

    last_seen: Optional[Path] = None
    last_mtime: float = 0.0
    reported_empty = False

    try:
        while True:
            latest = _find_latest_image(source_dir)
            if latest is None:
                if not reported_empty:
                    print("[INFO] Brak plików w dataset (jeszcze). Czekam na pierwszą klatkę...")
                    reported_empty = True
                time.sleep(interval)
                continue

            reported_empty = False
            try:
                mtime = latest.stat().st_mtime
            except FileNotFoundError:
                time.sleep(interval)
                continue

            if last_seen is None or mtime > last_mtime or latest != last_seen:
                try:
                    _copy_atomically(latest, output_path)
                    rel = latest.relative_to(source_dir)
                except ValueError:
                    rel = latest
                print(f"[INFO] Skopiowano {rel} -> {output_path}")
                last_seen = latest
                last_mtime = mtime
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[INFO] Przerwano przez użytkownika.")


if __name__ == "__main__":
    main()
