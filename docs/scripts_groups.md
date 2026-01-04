# Grupy skryptów w `src/`

## 1. Skrypty treningowe (`train_*`)

Skrypty odpowiedzialne za właściwe trenowanie modeli (MLP, CNN, regresje
cykliczne) na wcześniej wyliczonych cechach lub bezpośrednio na obrazach.

- `train_hour_cnn.py` – trening CNN na pełnych obrazach (klasy godzin 0–23).
- `train_hour_nn_cyclic.py` – trening MLP regresji cyklicznej (sin/cos) na cechach robust.
- `train_hour_nn_cyclic_2.py` – alternatywny/eksperymentalny wariant MLP cyklicznego.
- `train_hour_regression_cyclic.py` – regresja cykliczna (inne modele) dla godziny.
- `train_robust_time.py` – modele na cechach robust (regresja czasu).

## 2. Skrypty precompute / normalize

Skrypty przygotowujące dane wejściowe: wyliczanie cech z obrazów, ich
normalizacja oraz podstawowa analiza/eksploracja danych.

- `precompute_mean_rgb.py` – wyliczanie cech mean RGB dla wszystkich obrazów.
- `precompute_features_advanced.py` – wyliczanie cech advanced (RGB + HSV) do RF.
- `precompute_features_robust.py` – wyliczanie cech robust (bardziej złożone statystyki).
- `normalize_data.py` – normalizacja wyliczonych cech (RGB/advanced/robust).
- `load_data.py` – wczytywanie i walidacja labels.csv oraz plików datasetu.
- `explore_data.py` – analizy i statystyki danych/cech.

## 3. Skrypty baseline (`baseline_*`, `predict_hour`)

Prostsze modele bazowe (baseline), które służą jako punkt odniesienia
dla bardziej zaawansowanych sieci oraz do szybkiego sprawdzenia pipeline'u.

- `baseline_rgb.py` – baseline na prostych cechach mean RGB.
- `baseline_advanced.py` – baseline na cechach advanced (RandomForest + inne modele).
- `baseline_advanced_logreg.py` – baseline z uproszczonym/logistycznym modelem na cechach advanced.
- `predict_hour.py` – prosty skrypt do offline’owej predykcji godziny z zapisanych cech/modeli.

## 4. Skrypty kamera / overlay (`camera_*`)

Skrypty pracujące „online” z kamerą – generują bieżącą predykcję godziny
na obrazie z kamery, rysują overlay i/lub zapisują wynik do pliku.

- `camera_hour_overlay.py` – podgląd z kamery na PC z overlay (MLP robust / RF advanced).
- `camera_hour_overlay_advanced.py` – wariant nastawiony na cechy advanced.
- `camera_hour_overlay_mlp.py` – główny overlay na PC korzystający z MLP i/lub RF.
- `camera_hour_overlay_mlp_rpi.py` – wariant na Raspberry Pi (MLP robust / RF / CNN, zapis do JPG).
- `camera_hour_overlay_rpi.py` – prostszy wariant overlay na RPi (sterowany parametrem --model).

## 5. Pozostałe

Pomocnicze moduły konfiguracyjne i narzędziowe, które wspierają główny
pipeline (ścieżki, utils, przebudowa labels).

- `settings.py` – centralne ścieżki do danych i etykiet.
- `rebuild_labels.py` – przebudowa/aktualizacja pliku labels.csv.
- `utils.py`, `utils_robust.py` – funkcje pomocnicze (cechy, konwersje, itp.).
