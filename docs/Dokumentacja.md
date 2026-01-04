<!-- Alternatywny README / dokument ogólny projektu -->

# Dokumentacja projektu „Zegar biologiczny”

Projekt „Zegar biologiczny” ma na celu zbudowanie systemu, który na podstawie pojedynczego obrazu sceny zewnętrznej
potrafi oszacować porę doby (godzinę) w sposób zgodny z cykliczną naturą czasu. Rozwiązanie łączy klasyczne
przetwarzanie obrazu z metodami uczenia maszynowego, obejmując pełen proces: od akwizycji danych, przez
przygotowanie cech i trening modeli, aż po ich uruchomienie w środowisku rzeczywistym (kamera + overlay).

Aby ten cel zrealizować, projekt został podzielony na kilka spójnych modułów, tworzących kompletny łańcuch
przetwarzania (pipeline):

- **część akwizycyjna** – pozyskiwanie obrazów z kamery i ich trwały zapis na dysku,
- **część kontrolna i weryfikacyjna** – sprawdzanie spójności pliku `labels.csv` oraz rozkładu danych w czasie,
- **część korekcyjna** – porządkowanie i rekonstrukcja etykiet tam, gdzie surowe dane są niekompletne lub niespójne,
- **część przygotowująca dane** – wyznaczanie cech numerycznych z obrazów oraz ich normalizacja,
- **część trenująca** – uczenie i porównywanie modeli klasyfikacji i regresji czasu, w tym modeli cyklicznych,
- **część implementacyjna** – wykorzystanie wytrenowanych modeli w trybie on‑line (overlay godziny na obrazie,
  praca na żywym strumieniu z kamery).

W kolejnych rozdziałach przedstawiono zwięzły opis każdej z części wraz z odwołaniem do kluczowych skryptów
oraz artefaktów (pliki CSV z cechami, modele, logi).

---

## 1. Moduł akwizycji danych

Cel tej części stanowi cykliczne zapisywanie obrazów z kamery oraz budowa surowego zbioru danych.

Główne elementy:

- `MLDailyHourClock.py` – aplikacja wysokiego poziomu sterująca eksperymentem (akwizycja w dłuższym okresie, integracja z Raspberry Pi / komputerem PC),
- struktura folderu `dataset/2025/...` – surowe obrazy oraz plik `dataset/2025/labels.csv`
	zawierający m.in. kolumny `filepath`, `datetime`, `hour`.

Rezultatem części akwizycyjnej są **obrazy** oraz **etykiety czasu**, które stanowią wejście dla kolejnych etapów pipeline'u ML.

---

## 2. Moduł kontroli i weryfikacji danych

Celem tej części jest weryfikacja, czy zestaw danych jest spójny, kompletny i w sposób sensowny rozłożony w czasie.

Główne skrypty:

- `src/load_data.py`
	- funkcja `load_labels()` wczytuje `labels.csv` (z `src/settings.py` lub z pliku lokalnego), zapewniając poprawny typ kolumny `filepath`,
	- funkcja `check_files_exist()` sprawdza, czy wszystkie ścieżki z `labels.csv` istnieją względem katalogu `DATA_DIR`,
	- moduł może zostać uruchomiony jako narzędzie CLI do szybkiej walidacji zbioru danych.
- `src/explore_data.py`
	- podstawowa eksploracja: rozkład godzin (`hour`), liczba próbek na dzień (`datetime → date`),
	- generuje wykresy (matplotlib/seaborn) oraz wypisuje kluczowe statystyki opisowe.

Część kontrolna i weryfikacyjna umożliwia wczesne wykrycie brakujących plików, luk w rozkładzie godzin lub nierównomiernego próbkowania w czasie.

---

## 3. Moduł korekcji etykiet

Zadaniem tej części jest porządkowanie oraz ewentualna rekonstrukcja etykiet czasowych.

Główne skrypty:

- `src/rebuild_labels.py` – generuje i koryguje `labels.csv` na podstawie struktury katalogów i nazw plików,
	co zapewnia spójność etykiet z rzeczywistym zbiorem obrazów,
- `src/utils.py`, `src/utils_robust.py` – zbiór funkcji pomocniczych (m.in. wczytywanie obrazu, wyliczanie statystyk kanałów, obsługa błędów I/O).

Produktem części korekcyjnej jest **spójny plik etykiet**, na którym można bezpiecznie opierać dalsze etapy przetwarzania i budowy cech.

---

## 4. Moduł przygotowania danych (feature engineering i normalizacja)

Celem tej części jest przekształcenie surowych obrazów do postaci tabelarycznej (cechy numeryczne),
odpowiedniej dla modeli klasycznych oraz sieci neuronowych.

Główne skrypty:

- `src/precompute_mean_rgb.py`
	- dla każdego obrazu liczy średnie wartości kanałów R/G/B,
	- zapisuje wynik do `features_mean_rgb.csv`.
- `src/precompute_features_advanced.py`
	- wylicza rozszerzony zestaw cech: średnie i odchylenia RGB oraz statystyki HSV
		(funkcja `extract_rgb_hsv_stats()`),
	- zapisuje do `features_advanced.csv`.
- `src/precompute_features_robust.py`
	- generuje cechy „robust” (bardziej odporne na warunki oświetleniowe, chmury itd.),
	- zapisuje do `features_robust.csv`.
- `src/normalize_data.py`
	- normalizuje wybrane pliki cech (`features_*.csv`) z użyciem odpowiedniego skalera (Standard/Robust itp.),
	- zapisuje zarówno znormalizowane dane (`*_normalized.csv`), jak i parametry skalera (`*.npz`).

We wszystkich przypadkach statystyki normalizacji wyznaczane są **wyłącznie na zbiorze treningowym**,
a następnie stosowane do walidacji i testu, co ogranicza ryzyko przecieku informacji.

---

## 5. Moduł treningu modeli ML

Celem tej części jest wytrenowanie zestawu modeli przewidujących godzinę na podstawie cech wyznaczonych z obrazów.

Modele bazowe oparte na prostych cechach RGB:

- `src/baseline_rgb.py` – klasyfikacja godzin na podstawie `features_mean_rgb.csv`.
- `src/baseline_advanced.py` – modele na ręcznie wybranych cechach z `features_advanced.csv`.
- `src/baseline_advanced_logreg.py` – LogisticRegression na pełnym zestawie cech advanced.

Modele operujące na cechach robust:

- `src/train_robust_time.py`
	- klasyfikacja godzin z binningiem (przedziały czasowe),
	- LogisticRegression / RandomForest / GradientBoosting, metryki top‑k i circular MAE.

Modele regresji cyklicznej:

- `src/train_hour_regression_cyclic.py`
	- regresja (sin, cos) z użyciem modeli sklearn (Ridge, HGB, RF) na `features_robust.csv`,
	- metryki: Cyclic MAE, P90/P95, czas trenowania.
- `src/train_hour_nn_cyclic.py`
	- sieć MLP w PyTorch (wejście: cechy robust, wyjście: `[sin, cos]`),
	- **czasowy split danych**: `time_based_split()` dzieli dane chronologicznie na train/val/test
		po całych dniach (`datetime → date`),
	- normalizacja cech liczona tylko na train, identyczna transformacja stosowana do val/test,
	- zapisywany jest najlepszy checkpoint `models/best_mlp_cyclic.pt` według walidacyjnego Cyclic MAE.

Modele CNN na pełnym obrazie:

- `src/train_hour_cnn.py`
	- konwolucyjna sieć neuronowa w PyTorch uczona bezpośrednio na obrazach (bez ręcznego feature engineeringu),
	- wejście: obraz RGB przeskalowany do zadanej rozdzielczości kwadratowej (np. 224×224),
	- wyjście: klasy godzin 0..23 lub ich rozkład prawdopodobieństwa,
	- zapis checkpointu (stan modelu oraz metadane, m.in. `num_classes`, `img_size`) do pliku `models/rpi/best_cnn_hour.pt`.

Wytrenowane modele klasyczne zapisywane są jako pliki `.pkl` w katalogu `models/`,
natomiast model MLP jako plik `.pt` (stan sieci oraz parametry normalizacji).

---

## 6. Moduł implementacyjny (predykcja on‑line)

Celem tej części jest wykorzystanie wytrenowanych modeli do przewidywania godziny na nowych obrazach
oraz prezentowanie tej informacji użytkownikowi w sposób czytelny i ciągły.

Główne skrypty:

- `src/predict_hour.py`
	- wczytuje zapisane modele oraz skalery,
	- przyjmuje ścieżkę do obrazu, wylicza cechy (robust / advanced) i zwraca przewidywaną godzinę (tryb batch/off‑line).
- `src/camera_hour_overlay.py`
	- prosty overlay czasu na obrazie (tryb podglądu / debug),
	- może korzystać z prostszych modeli baseline.
- `src/camera_hour_overlay_advanced.py`, `src/camera_hour_overlay_rpi.py`
	- integracja z OpenCV i/lub Raspberry Pi,
	- pobranie klatki z kamery, predykcja godziny **aktualnym modelem produkcyjnym** (np. `best_mlp_cyclic.pt`),
	- naniesienie opisu (overlay) i zapis / wyświetlenie w pętli.

- `src/camera_hour_overlay_mlp.py`, `src/camera_hour_overlay_mlp_rpi.py`
	- warianty overlay oparte o regresję cykliczną MLP na cechach robust oraz tryb fallback (RandomForest na cechach advanced),
	- wersja RPi obsługuje dodatkowo model CNN trenowany na pełnym obrazie (`best_cnn_hour.pt`, przełącznik `--use_cnn`),
	- skrypty zapisują klatki z nałożoną godziną do pliku JPG oraz opcjonalny log CSV z predykcjami.

Powyższe skrypty pełnią rolę **warstwy implementacyjnej oraz narzędzia do weryfikacji jakości modelu w czasie rzeczywistym**:
umożliwiają wizualną ocenę predykcji (porównanie przewidywanej godziny z rzeczywistą)
bezpośrednio na obrazie z kamery.

---

## 7. Orkiestracja pipeline'u i logowanie

Cały pipeline może zostać uruchomiony jednym poleceniem z katalogu głównego repozytorium:

```bash
./run_full_pipeline.sh
```

Skrypt:

- wykonuje kolejne kroki: `load_data`, `explore_data`, `precompute_mean_rgb`, `normalize_mean_rgb`,
	`baseline_rgb`, `precompute_features_advanced`, `normalize_advanced`, `baseline_advanced`,
	`baseline_advanced_logreg`, `precompute_features_robust`, `normalize_robust`,
	`train_robust_time`, `train_hour_regression_cyclic`, `train_hour_nn_cyclic`,
- zapisuje logi do katalogu `Logs/YYYY.MM.DD/`,
- utrzymuje **checkpointy kroków** w `Logs/YYYY.MM.DD/checkpoints/` (`*.done`),
	co pozwala wznawiać pipeline od wybranego etapu (np. `./run_full_pipeline.sh precompute_features_advanced`).

---

## 8. Podsumowanie

Projekt „Zegar biologiczny” realizuje kompletny łańcuch przetwarzania obejmujący:

1. Pozyskanie obrazów i zapis etykiet czasu.
2. Walidację i eksplorację zbioru danych.
3. Korektę i rekonstrukcję etykiet tam, gdzie jest to konieczne.
4. Ekstrakcję cech RGB/HSV/robust oraz ich normalizację.
5. Trening wielu modeli (baseline oraz modeli zaawansowanych, w tym MLP z regresją cykliczną).
6. Integrację modelu z kamerą i możliwość pracy w trybie on‑line.

Niniejszy dokument uzupełnia pozostałe materiały w katalogu `docs/` (np. szczegółowe opisy cech i eksperymentów)
i ma służyć jako zwięzły, techniczny przegląd architektury projektu.