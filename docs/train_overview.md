# Train: przegląd skryptów treningowych

Ten dokument opisuje, co robią skrypty `train_*`, na jakich cechach pracują,
jak kodują godzinę (klasy vs regresja cykliczna), jakie algorytmy wykorzystują
oraz jakie artefakty zapisują.

---

## 1. `train_hour_nn_cyclic.py` – MLP (sin, cos) na cechach robust

**Cel:**

- Nauczyć sieć neuronową przewidywania dokładnej godziny \(h \in [0,24)\)
  jako punktu na okręgu: \((\sin \theta, \cos \theta)\), gdzie
  \( \theta = 2\pi h / 24 \).
- Dzięki temu godziny 23 i 0 są blisko siebie na kole (rozwiązanie problemu
  wrap-around), a metryka błędu jest cykliczna.

**Wejście / wyjście:**

- Wejście: `features_robust.csv` (z `precompute_features_robust.py`).
- Cechy wejściowe: wszystkie kolumny numeryczne poza `filepath`, `datetime`, `hour`.
- Cel: wektor 2D `y_sc = [sin(theta), cos(theta)]` zbudowany z kolumny `hour`.
- Wyjście:
  - wytrenowany model MLP zapisany do `models/pc/best_mlp_cyclic.pt`,
  - zapisane w checkpointcie `mean` i `std` do standaryzacji cech (wykorzystywane
    później w overlay MLP).

**Kodowanie celu i metryki:**

- Kodowanie:
  - funkcja `hour_to_sin_cos(hour)` przekształca wektor godzin na \((\sin, \cos)\).
  - podczas ewaluacji `sin_cos_to_hour` odwraca ten proces.
- Metryka:
  - definiuje się błąd cykliczny
    \( e = \min(|h_{pred} - h_{true}|, 24 - |h_{pred} - h_{true}|) \),
  - raportowane są: Cyclic MAE (średni błąd w godzinach), P90 i P95 błędu.

**Model i trening:**

- MLP z kilkoma warstwami `Linear + BatchNorm1d + GELU + Dropout`, wyjście 2D.
- Dane dzielone są na train/val/test (ok. 80/10/10) z **losową stratyfikacją po godzinie**.
- Standaryzacja cech (mean=0, std=1) liczona wyłącznie na zbiorze train,
  następnie stosowana do val/test.
- Loss: `SmoothL1Loss` (Huber) na przestrzeni \((\sin, \cos)\), z lekkim
  „ściąganiem” predykcji do okręgu przez normalizację wektorów.
- Optymalizator: `AdamW` z redukcją LR na plateau (`ReduceLROnPlateau`) i
  early stopping po braku poprawy metryki walidacyjnej.

**Zastosowanie:**

- To główny model MLP wykorzystywany w overlay (`camera_hour_overlay_mlp.py`,
  `camera_hour_overlay_mlp_rpi.py`) w trybie „robust features + MLP cyclic”.

---

## 2. `train_hour_nn_cyclic_2.py` – MLP (sin, cos) z podziałem czasowym

**Cel:**

- Taki sam jak powyżej (regresja cykliczna \((\sin, \cos)\) na cechach robust),
  ale z **bardziej realistycznym podziałem danych w czasie**.
- Ma lepiej odzwierciedlać scenariusz „trenujemy na przeszłości, testujemy na
  przyszłości”, ograniczając przeciek informacji.

**Wejście / wyjście:**

- Wejście: `features_robust.csv`.
- Cechy i kodowanie celu: takie same jak w `train_hour_nn_cyclic.py`.
- Wyjście: analogiczny model MLP (zapisany do `models/pc`) + te same metryki.

**Kluczowe różnice vs `train_hour_nn_cyclic.py`:**

- Zamiast losowego splitu z `train_test_split`, użyty jest
  `time_based_split(df, datetime_col="datetime", ratios=(0.7, 0.15, 0.15))`:
  - dane sortowane po `datetime`,
  - grupowanie po dniach, przydział całych dni do zbiorów train/val/test,
  - zapewnione co najmniej po jednym dniu w każdym zbiorze.
- Ewaluacja zawsze normalizuje \((\sin, \cos)\) do długości 1, żeby być spójnym
  z treningiem.

**Zastosowanie:**

- Eksperymentalny wariant MLP, bardziej „time-aware”, dobry do porównań
  z klasycznym, losowym splitem.

---

## 3. `train_hour_regression_cyclic.py` – klasyczne modele regresyjne (sin, cos)

**Cel:**

- Zamiast sieci neuronowej – użyć **klasycznych modeli regresyjnych** na
  cechach robust, ciągle w paradygmacie \((\sin, \cos)\) + błąd cykliczny.

**Wejście / wyjście:**

- Wejście: `features_robust.csv`.
- Cechy: wszystkie kolumny numeryczne poza `filepath`, `datetime`, `hour`.
- Cel: \( y_{sc} = [\sin, \cos] \) z `hour_to_sin_cos`.
- Wyjście: brak pojedynczego pliku modelu (skrypt skupia się na porównaniu
  jakości), za to szczegółowy log metryk dla kilku modeli.

**Modele porównywane (`make_models`):**

- `ridge`:
  - `Pipeline(StandardScaler → Ridge)` zapakowany w `MultiOutputRegressor`.
  - Liniowy, szybki baseline regresyjny.
- `hgb`:
  - `HistGradientBoostingRegressor` w `MultiOutputRegressor`.
  - Bardzo wydajny booster na cechach tablicowych.
- `rf`:
  - `RandomForestRegressor` (300 drzew, `n_jobs=-1`) w `MultiOutputRegressor`.
  - Nieliniowy model zespołowy, dobry na duże tablice.

**Metryki:**

- Dla każdego modelu:
  - predykcja \((\sin, \cos)\) na teście, konwersja do godzin
    przez `sin_cos_to_hour`,
  - `Cyclic MAE (h)`: główna miara błędu,
  - `Naive MAE (h)`: klasyczne MAE bez uwzględniania cykliczności (do porównań),
  - P90/P95 błędu cyklicznego,
  - czas trenowania.
- Na końcu wybierany jest najlepszy model wg Cyclic MAE i logowane jest
  podsumowanie.

**Zastosowanie:**

- Referencyjne porównanie: jak daleko dojdziemy klasycznymi regresorami
  w modelu cyklicznym vs MLP.

---

## 4. `train_robust_time.py` – klasyfikacja godzin w binach z cech robust

**Cel:**

- Uprościć zadanie do **klasyfikacji godzin w przedziały** (biny czasowe), a
  następnie zadawać pytania typu „z jaką dokładnością trafiamy w przedział
  lub jego najbliższe sąsiedztwo”.

**Wejście / wyjście:**

- Wejście: `features_robust.csv`.
- Cechy: wszystkie kolumny poza `filepath`, `datetime`, `hour`.
- Etykiety: numery binów, np. przy `bin_size=4` mamy 6 klas (0–3, 4–7, ..., 20–23).
- Wyjście: brak zapisu modelu do pliku – skrypt służy głównie do analizy metryk.

**Binning i metryki:**

- `BinningConfig(bin_size=4)` – domyślnie godziny dzielone na przedziały po 4.
- Metryki:
  - top‑1 accuracy (klasyczna dokładność),
  - top‑2 i top‑3 accuracy (czy prawdziwy bin jest w 2/3 najbardziej
    prawdopodobnych klasach),
  - circular MAE liczony z **środka binu** jako reprezentatywnej godziny,
  - „confident accuracy”: accuracy po odrzuceniu predykcji z niską pewnością
    (np. max(prob) < 0.45), wraz z coverage.

**Modele (`make_models`):**

- `logreg` – StandardScaler + LogisticRegression (wieloklasowy).
- `rf` – RandomForestClassifier (300 drzew).
- `gb` – GradientBoostingClassifier.

**Zastosowanie:**

- Ocenia, na ile modele na cechach robust potrafią „wpisać się w okno
  czasowe” zamiast trafiać dokładną godzinę.
- Wyniki (top‑k, confident) pomagają dobrać strategie typu „pokaż predykcję
  tylko, gdy jesteśmy wystarczająco pewni”.

---

## 5. `train_hour_cnn.py` – CNN na pełnych obrazach

**Cel:**

- Odejść od ręcznie projektowanych cech i nauczyć **konwolucyjną sieć
  neuronową**, która bierze na wejście cały obraz i bezpośrednio klasyfikuje
  godzinę (0–23).

**Wejście / wyjście:**

- Wejście:
  - `labels.csv` / `dataset/2025/labels.csv` (filepath, hour, datetime),
  - obrazy z `DATA_DIR`.
- Transformacje wejścia:
  - obraz BGR → RGB,
  - resize do `img_size×img_size` (domyślnie 224×224),
  - skala [0,1], zamiana na tensor `(3, H, W)`.
- Etykieta: `hour` jako klasa 0–23.
- Wyjście: wytrenowany model CNN zapisany do `models/pc/best_cnn_hour.pt`.

**Model (`SmallHourCNN`):**

- 3 bloki: `Conv2d → ReLU → MaxPool2d` (kolejno 16, 32, 64 kanałów),
  co daje stopniowe downsamplingowanie obrazu.
- `AdaptiveAvgPool2d((1,1))` → globalne uśrednianie przestrzenne.
- Klasyfikator: `Linear(64→64) + ReLU + Linear(64→24)`.
- Loss: `CrossEntropyLoss`, optymalizator `Adam`.

**Trening:**

- Dataset `HourImageDataset` budowany na podstawie labels.csv.
- Podział danych: train/val z `random_split` (proporcje `val_frac`, domyślnie 0.1).
- Metryki w trakcie treningu:
  - logowane per epoka: `train_loss`, `train_acc`, `val_loss`, `val_acc`.
- Najlepszy model według `val_loss` jest zapisywany jako
  `models/pc/best_cnn_hour.pt`.

**Zastosowanie:**

- Eksperymentalny, end‑to‑end model wizyjny.
- Na RPi może być używany jako alternatywny backend (`--use_cnn` w
  `camera_hour_overlay_mlp_rpi.py`), po skopiowaniu modelu do `models/rpi/`.

---

## 6. Podsumowanie roli skryptów `train_*`

- `train_hour_nn_cyclic.py` / `train_hour_nn_cyclic_2.py` – MLP na cechach robust,
  uczą się regresji cyklicznej (sin, cos) i są głównymi dostawcami modelu MLP
  do overlay.
- `train_hour_regression_cyclic.py` – klasyczne regresory (Ridge/HGB/RF) w tym
  samym paradygmacie sin/cos, do porównań z MLP.
- `train_robust_time.py` – klasyfikacja godzin do binów czasowych + top‑k,
  confident accuracy; bardziej „robustny” punkt widzenia na zadanie.
- `train_hour_cnn.py` – end‑to‑end CNN na pełnych obrazach.

Wszystkie te skrypty razem tworzą „laboratorium” modeli: od prostych liniowych
po nieliniowe MLP/CNN, bazujące na różnych reprezentacjach (cechy robust vs pełen obraz).
