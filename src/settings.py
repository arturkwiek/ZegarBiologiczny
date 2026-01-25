
# settings.py

# Opis:
# 	Plik konfiguracyjny projektu.
# 	- Definiuje ścieżki do katalogu z danymi i pliku z etykietami
# 	- Ułatwia centralne zarządzanie lokalizacją danych

# Zadania realizowane przez skrypt:
# 	1. Definicja ścieżek do danych i etykiet
# 	2. Centralna konfiguracja dla pozostałych modułów

from pathlib import Path

DATA_DIR = Path("./dataset/2026/")
LABELS_CSV = Path("./dataset/2026/labels.csv")