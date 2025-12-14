from pathlib import Path
import pandas as pd

DATA_ROOT = Path(".")          # root projektu
LABELS_PATH = Path("labels.csv")


def load_labels() -> pd.DataFrame:
    """
    Wczytuje labels.csv poprawnie wymuszając filepath jako string.
    """
    df = pd.read_csv(
        LABELS_PATH,
        dtype={"filepath": str}    # <- To naprawia problem
    )
    return df


from tqdm import tqdm

def check_files_exist(df: pd.DataFrame):
    missing = []
    for fp in tqdm(df["filepath"].astype(str), desc="Sprawdzanie plików"):
        full = DATA_ROOT / fp
        if not full.exists():
            missing.append(fp)
    return missing



if __name__ == "__main__":
    df = load_labels()
    print("Pierwsze wiersze labels.csv:")
    print(df.head())

    print("\nLiczba rekordów:", len(df))

    missing = check_files_exist(df)
    print("\nBrakujące pliki:", len(missing))
    if len(missing) > 0:
        print(missing[:10])
