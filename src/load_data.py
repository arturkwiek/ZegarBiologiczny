from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm

try:
    from src.settings import DATA_DIR, LABELS_CSV
except ImportError:
    DATA_DIR = None
    LABELS_CSV = None

def _resolve_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def load_labels(labels_path: str | Path) -> pd.DataFrame:
    """
    Wczytuje labels.csv, wymuszając filepath jako string (żeby nie ucinało zer itd.)
    """
    labels_path = LABELS_CSV or Path("labels.csv")
    if not labels_path.exists():
        raise FileNotFoundError(f"Nie znaleziono labels.csv: {labels_path}")

    df = pd.read_csv(labels_path, dtype={"filepath": str})
    return df


def check_files_exist(df: pd.DataFrame, data_root: str | Path) -> list[str]:
    """
    Sprawdza czy pliki wskazane w kolumnie 'filepath' istnieją względem data_root.
    Zwraca listę brakujących ścieżek (takich jak w CSV).
    """
    data_root = DATA_DIR or Path("dataset")
    if not data_root.exists():
        raise FileNotFoundError(f"Nie znaleziono folderu datasetu: {data_root}")

    if "filepath" not in df.columns:
        raise KeyError("Brak kolumny 'filepath' w labels.csv")

    missing: list[str] = []
    for fp in tqdm(df["filepath"].astype(str), desc="Sprawdzanie plików"):
        full = data_root / fp
        if not full.exists():
            missing.append(fp)
    return missing


def get_default_paths(project_root: Path | None = None) -> tuple[Path, Path]:
    """
    Ustala domyślne ścieżki:
    - priorytet: zmienne środowiskowe
    - potem: domyślne w projekcie
    """
    root = _resolve_path(project_root or Path(__file__).resolve().parents[1])

    data_root = os.environ.get("DATA_DIR")
    labels_path = os.environ.get("LABELS_CSV")

    data_root = _resolve_path(data_root) if data_root else (root / "dataset").resolve()
    labels_path = _resolve_path(labels_path) if labels_path else (root / "labels.csv").resolve()

    return data_root, labels_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None, help="Folder z danymi (root dla filepath)")
    parser.add_argument("--labels", type=Path, default=None, help="Ścieżka do labels.csv")
    args = parser.parse_args()

    default_data_root, default_labels_path = get_default_paths()

    data_root = args.data_dir or default_data_root
    labels_path = args.labels or default_labels_path

    df = load_labels(labels_path)
    print("Pierwsze wiersze labels.csv:")
    print(df.head())
    print("\nLiczba rekordów:", len(df))

    missing = check_files_exist(df, data_root=data_root)
    print("\nBrakujące pliki:", len(missing))
    if missing:
        print(missing[:10])
