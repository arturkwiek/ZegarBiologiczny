# Cechy generowane przez `precompute_mean_rgb.py`

Ten plik opisuje wszystkie kolumny dostępne w pliku `features_mean_rgb.csv`, generowanym przez moduł `precompute_mean_rgb.py`.

## Kolumny meta-danych

- **filepath**  
  Ścieżka względna do pliku obrazu względem katalogu projektu lub katalogu danych.  
  Przykład: `dataset/2025/11/29/13/20251129_135945.jpg`.

- **datetime**  
  Dokładny znacznik czasu utworzenia klatki, w formacie ISO (np. `2025-11-29T13:59:45.123456`).  
  Może być używany do filtrowania danych po dacie, porze roku, dniu tygodnia itp.

- **hour**  
  Godzina doby (0–23), pełniąca rolę etykiety klasy w zadaniu przewidywania godziny.  
  Typ: liczba całkowita.

## Cechy z obrazu (wektor wejściowy)

Wszystkie cechy są obliczane na podstawie pełnego obrazu RGB wczytywanego przez OpenCV.  
OpenCV czyta obraz w formacie BGR, ale w funkcji `extract_mean_rgb` następuje konwersja do przestrzeni RGB.

- **r_mean**  
  Średnia wartość kanału R (Red) po wszystkich pikselach obrazu.  
  - Zakres: ~0–255 (dla obrazów 8‑bitowych).  
  - Interpretacja: globalna „czerwoność” sceny. Wyższa wartość oznacza ogólnie cieplejsze tony lub silniejsze światło w kanale czerwonym.

- **g_mean**  
  Średnia wartość kanału G (Green).  
  - Zakres: ~0–255.  
  - Interpretacja: globalna ilość składowej zielonej; dla scen z dużą ilością roślinności lub mocno zielonego oświetlenia wartości mogą być wyższe.

- **b_mean**  
  Średnia wartość kanału B (Blue).  
  - Zakres: ~0–255.  
  - Interpretacja: globalna „niebieskość” sceny; wieczorem lub przy zimnym świetle wartości mogą rosnąć.

## Podsumowanie

Po uruchomieniu `precompute_mean_rgb.py` dla każdej klatki z `labels.csv` otrzymujesz jeden wiersz w `features_mean_rgb.csv` o postaci:

- meta‑dane: `filepath`, `datetime`, `hour`
- wektor cech: `[r_mean, g_mean, b_mean]`

Jest to najprostsza, 3‑wymiarowa reprezentacja koloru całego obrazu, przydatna jako baseline dla prostych modeli tablicowych (np. regresja logistyczna).