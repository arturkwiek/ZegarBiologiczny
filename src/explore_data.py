# explore_data.py
# Run: python -m src.explore_data

# Opis:
#     Prosty skrypt eksploracyjny do podglądu rozkładu danych w labels.csv.
#     - Wczytuje labels.csv z domyślnej lokalizacji (settings.LABELS_CSV lub ./labels.csv)
#     - Wypisuje podstawowe informacje o liczbie rekordów i przykładach wierszy
#     - Pokazuje rozkład godzin (0–23) jako wykres słupkowy
#     - Pokazuje liczbę próbek na dzień jako wykres słupkowy po dacie
#     - Ułatwia szybkie sprawdzenie, czy dataset jest w miarę zbalansowany po godzinach/dniach

# Zadania realizowane przez skrypt:
#     1. Wczytanie labels.csv (z użyciem src.load_data.load_labels)
#     2. Wyświetlenie podstawowych statystyk w terminalu
#     3. Wizualizacja rozkładu godzin (plot_hour_distribution)
#     4. Wizualizacja liczby próbek na dzień (plot_samples_per_day)

import sys
from pathlib import Path

# gdy uruchamiany jako skrypt, dopisz root projektu do sys.path
if __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.load_data import load_labels
try:
    from src.settings import DATA_DIR, LABELS_CSV
except ImportError:
    DATA_DIR = None
    LABELS_CSV = None

def add_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje kolumnę 'date' (YYYY-MM-DD) wyciągniętą z kolumny 'datetime'.
    Zakładamy, że datetime jest w formacie ISO, np. 2025-11-30T16:59:56.088524
    """
    if "date" not in df.columns:
        df["date"] = df["datetime"].str.slice(0, 10)
    return df


def plot_hour_distribution(df: pd.DataFrame):
    """
    Rysuje rozkład liczby próbek na każdą godzinę (0–23).
    """
    plt.figure(figsize=(10, 4))
    sns.countplot(x=df["hour"])
    plt.title("Rozkład liczby próbek na godziny")
    plt.xlabel("Godzina (0–23)")
    plt.ylabel("Liczba próbek")
    plt.tight_layout()
    plt.show()


def plot_samples_per_day(df: pd.DataFrame):
    """
    Rysuje liczbę próbek na każdy dzień.
    """
    df = add_date_column(df)
    counts = df["date"].value_counts().sort_index()

    plt.figure(figsize=(10, 4))
    counts.plot(kind="bar")
    plt.title("Liczba próbek na dzień")
    plt.xlabel("Data")
    plt.ylabel("Liczba próbek")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    labels_path = LABELS_CSV or Path("labels.csv")
    df = load_labels(labels_path)
    print("Liczba rekordów:", len(df))
    print("Przykładowe rekordy:")
    print(df.head())

    print("\nRozkład godzin:")
    print(df["hour"].value_counts().sort_index())

    plot_hour_distribution(df)
    plot_samples_per_day(df)
