# Rejestr modeli (model registry)

Ten plik podsumowuje dostępne modele w katalogu `models/` oraz proponuje konwencję wersjonowania na przyszłość.

## 1. Aktualne pliki w `models/`

- `baseline_advanced_gb_model.pkl`  
  - Typ: GradientBoostingClassifier (features_advanced)  
  - Przeznaczenie: baseline / eksperyment  
  - Trening: `python -m src.baseline_advanced`

- `baseline_advanced_knn_model.pkl`  
  - Typ: KNeighborsClassifier (features_advanced)  
  - Przeznaczenie: baseline / eksperyment  
  - Trening: `python -m src.baseline_advanced`

- `baseline_advanced_logreg_model.pkl`  
  - Typ: LogisticRegression (features_advanced)  
  - Przeznaczenie: baseline / eksperyment  
  - Trening: `python -m src.baseline_advanced`

- `baseline_advanced_rf_model.pkl`  
  - Typ: RandomForestClassifier (features_advanced)  
  - Przeznaczenie: model klasyfikacyjny godzin (fallback) używany m.in. w `camera_hour_overlay_mlp_rpi`  
  - Trening: `python -m src.baseline_advanced`

- `baseline_rgb_model.pkl`  
  - Typ: model RGB (baseline)  
  - Trening: `python -m src.baseline_rgb`

- `best_mlp_cyclic.pt`  
  - Typ: MLP (regresja cykliczna sin/cos) na cechach ROBUST  
  - Przeznaczenie: główny model używany w skryptach MLP (`train_hour_nn_cyclic*.py`, `camera_hour_overlay_mlp*.py`)  

## 2. Proponowana konwencja nazewnictwa

Na potrzeby kolejnych wersji modeli warto stosować schemat:

`<zadanie>_<cechy>_<algorytm>_<wariant>_<platforma>_<data>_<kluczowe_parametry>.<ext>`

Przykłady dla tego repo:

- `hour_advanced_rf_class_pc_2025-12-28_n200_depthNone.pkl`
- `hour_advanced_rf_class_rpi_2025-12-29_n50_depth12.pkl`
- `hour_robust_mlp_cyclic_pc_2025-12-20_in58_h256-128.pt`

Gdzie:

- `zadanie` – np. `hour` (predykcja godziny),
- `cechy` – `advanced`, `robust`, `rgb`,
- `algorytm` – `rf`, `mlp`, `logreg`, `knn`, `gb`,
- `wariant` – np. `class` / `reg` / `cyclic`,
- `platforma` – `pc`, `rpi`, `server`, jeśli ma znaczenie,
- `data` – `YYYY-MM-DD`,
- `kluczowe_parametry` – np. `n200_depthNone`, `in58_h256-128`.

## 3. Alias na "aktualny" model

Aby nie zmieniać kodu za każdym razem, można utrzymywać aliasy:

- `baseline_advanced_rf_current_pc.pkl`
- `baseline_advanced_rf_current_rpi.pkl`

Kod (np. `camera_hour_overlay_mlp_rpi.py`) może korzystać z aliasu, a prawdziwy plik wersjonowany trzymać osobno w katalogu `models/` lub w podkatalogu `models/archive/`.

Przykład organizacji:

```text
models/
  baseline_advanced_rf_current_pc.pkl      # alias / kopia aktualnego modelu PC
  baseline_advanced_rf_current_rpi.pkl     # alias / kopia aktualnego modelu RPi
  hour_advanced_rf_class_pc_2025-12-28_n200_depthNone.pkl
  hour_advanced_rf_class_rpi_2025-12-29_n50_depth12.pkl
  archive/
    ... stare wersje ...
```

## 4. Metadane obok modeli

Dla każdego nowego modelu warto zapisać metadane w pliku `.json` o tej samej nazwie bazowej, np.:

Plik modelu:

- `hour_advanced_rf_class_rpi_2025-12-29_n50_depth12.pkl`

Metadane:

- `hour_advanced_rf_class_rpi_2025-12-29_n50_depth12.meta.json`

Przykładowa zawartość:

```json
{
  "dataset": "features_advanced.csv",
  "features": "advanced (r,g,b mean/std + hsv)",
  "target": "hour_class_0_23",
  "algo": "RandomForestClassifier",
  "params": {
    "n_estimators": 50,
    "max_depth": 12,
    "min_samples_leaf": 20
  },
  "train_sklearn_version": "1.8.0",
  "train_date": "2025-12-29",
  "train_script": "src/baseline_advanced.py",
  "notes": "Wersja zoptymalizowana pod Raspberry Pi (mniejszy model)."
}
```

## 5. Minimalny workflow przy dodawaniu nowego modelu

1. Wytrenować model odpowiednim skryptem (np. `python -m src.baseline_advanced`).
2. Zapisany model skopiować / przenieść do `models/` pod nazwą zgodną ze schematem.
3. Utworzyć plik `.meta.json` z podstawowymi informacjami (dane, parametry, metryki, wersje bibliotek).
4. Jeśli model ma być domyślny dla PC/RPi – podmienić odpowiedni alias (`*_current_pc.pkl` / `*_current_rpi.pkl`).
5. Dodać w tym pliku krótką notkę o nowym modelu (sekcja 1 lub nowy podrozdział typu „Historia modeli”).

Ten rejestr możesz swobodnie rozwijać, np. dopisując dokładne metryki (accuracy, MAE, błąd godzinowy) dla poszczególnych modeli.

## 6. Changelog – 2025-12-29

- Wydzielono katalogi na modele:
  - `models/pc` – modele trenowane i używane na PC.
  - `models/rpi` – modele przeznaczone na Raspberry Pi.
- Zmieniono skrypty treningowe tak, by zapisywały modele do `models/pc`:
  - [src/baseline_advanced.py](../src/baseline_advanced.py)
  - [src/baseline_rgb.py](../src/baseline_rgb.py)
  - [src/train_hour_nn_cyclic.py](../src/train_hour_nn_cyclic.py)
  - [src/train_hour_nn_cyclic_2.py](../src/train_hour_nn_cyclic_2.py)
- Skrypty inference na PC przełączono na `models/pc`:
  - [src/predict_hour.py](../src/predict_hour.py)
  - [src/camera_hour_overlay_mlp.py](../src/camera_hour_overlay_mlp.py).
- Skrypt overlay na RPi ([src/camera_hour_overlay_mlp_rpi.py](../src/camera_hour_overlay_mlp_rpi.py)) ustawiono na korzystanie z `models/rpi` (oddzielne ścieżki do `best_mlp_cyclic.pt` i `baseline_advanced_rf_model.pkl`).
- Odchudzono RandomForest w [src/baseline_advanced.py](../src/baseline_advanced.py) (mniej drzew, ograniczona głębokość, `min_samples_leaf`), aby model był lżejszy i nadawał się na RPi.
- Uelastyczniono wczytywanie `features_advanced.csv` (obsługa przypadków bez kolumn `h_mean`, `h_std`).
- Poprawiono kompatybilność z aktualną wersją scikit-learn (usunięto problematyczny parametr `multi_class` z LogisticRegression) oraz zaakceptowano ostrzeżenia typu FutureWarning.
