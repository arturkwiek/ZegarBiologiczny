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
    df = load_labels()
    print("Liczba rekordów:", len(df))
    print("Przykładowe rekordy:")
    print(df.head())

    print("\nRozkład godzin:")
    print(df["hour"].value_counts().sort_index())

    plot_hour_distribution(df)
    plot_samples_per_day(df)
