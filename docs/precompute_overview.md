# Precompute: przegląd cech i normalizacji

Ten dokument opisuje, co dokładnie robią skrypty precompute/normalize, jakie cechy
wyliczają, jakie algorytmy są stosowane i w jakim celu.

---

## 1. `precompute_mean_rgb.py` – średnie wartości RGB

**Cel:**

- Zbudować bardzo prosty opis obrazu: trzy liczby odpowiadające średnim jasnościom
  kanałów R, G, B. Ten zestaw cech służy do najprostszego baseline'u `baseline_rgb.py`.

**Wejście / wyjście:**

- Wejście: `labels.csv` (kolumny: filepath, hour, datetime) + obrazy z `DATA_DIR`.
- Wyjście: `features_mean_rgb.csv` z kolumnami:
  - `filepath`, `datetime`, `hour`,
  - `r_mean`, `g_mean`, `b_mean`.

**Algorytm / cechy:**

- Dla każdego wiersza z `labels.csv`:
  - Wczytaj obraz z dysku.
  - Przekonwertuj do RGB (wewnątrz `extract_mean_rgb`).
  - Policz średnią wartość pikseli osobno dla kanałów R, G, B:
    - \( r_{mean} = \frac{1}{N} \sum_i R_i \)
    - \( g_{mean} = \frac{1}{N} \sum_i G_i \)
    - \( b_{mean} = \frac{1}{N} \sum_i B_i \)
- Zapisz wynik do CSV.

**Zastosowanie:**

- Szybki, tani obliczeniowo baseline do przewidywania godziny na podstawie ogólnego
  zabarwienia sceny (np. różny kolor nieba rano/wieczorem).

---

## 2. `precompute_features_advanced.py` – cechy RGB + HSV

**Cel:**

- Wyliczyć bogatszy wektor cech do modeli klasycznych (RandomForest, itp.), który
  opisuje nie tylko średni kolor, ale też rozrzut (odchylenie standardowe) oraz
  podstawowe statystyki w przestrzeni HSV.
- Te cechy są głównym wejściem do `baseline_advanced.py`.

**Wejście / wyjście:**

- Wejście: `labels.csv` + obrazy z `DATA_DIR`.
- Wyjście: `features_advanced.csv` z kolumnami:
  - `filepath`, `datetime`, `hour`,
  - `r_mean`, `g_mean`, `b_mean`,
  - `r_std`, `g_std`, `b_std`,
  - `h_mean`, `h_std`,
  - `s_mean`, `v_mean`.

**Algorytm / cechy (funkcja `extract_rgb_hsv_stats`):**

1. Wczytaj obraz i przekonwertuj do przestrzeni RGB (0–255).
2. Policz średnie i odchylenia standardowe kanałów RGB:
   - \( r_{mean}, g_{mean}, b_{mean} \)
   - \( r_{std}, g_{std}, b_{std} \).
3. Przekonwertuj obraz z RGB do HSV (OpenCV):
   - H w zakresie [0, 179], S i V w zakresie [0, 255].
4. Dla kanału H (odcień):
   - policz \( h_{mean} \) i \( h_{std} \) – średni odcień i jego rozrzut.
5. Dla kanałów S (nasycenie) i V (wartość/jasność):
   - policz \( s_{mean} \) i \( v_{mean} \).

**Zastosowanie:**

- Dane wejściowe dla `baseline_advanced.py` (RandomForest + inne klasyczne modele).
- Umożliwia modelom wykorzystanie:
  - ogólnego poziomu jasności i koloru (RGB),
  - informacji o „typie” światła (odcień, nasycenie) i jego zmienności.

---

## 3. `precompute_features_robust.py` – cechy robust

**Cel:**

- Wyliczyć znacznie bogatszy i bardziej „robust” zestaw cech opisujących obraz
  niż same statystyki kolorów. Te cechy są używane głównie przez modele typu MLP
  i regresje robust (`train_hour_nn_cyclic.py`, `train_robust_time.py`).

**Wejście / wyjście:**

- Wejście: `labels.csv` + obrazy z `DATA_DIR`.
- Wyjście: `features_robust.csv` z kolumnami:
  - `filepath`, `datetime`, `hour`,
  - + dziesiątki kolumn cech numerycznych zwróconych przez
    `extract_robust_image_features` (np. statystyki jasności, tekstury, lokalnych
    kontrastów – dokładny zestaw zależy od implementacji w `utils_robust.py`).

**Algorytm / cechy (funkcja `extract_robust_image_features`):**

- Dla każdego obrazu:
  - Wczytaj obraz (BGR) z dysku.
  - Oblicz szereg cech, m.in. (przykłady typowych robust features; szczegóły są
    w `utils_robust.py`):
    - statystyki intensywności/jasności (średnia, odchylenie, kwartyle),
    - statystyki lokalnego kontrastu i krawędzi (np. gradienty, filtry Laplace’a),
    - histogramy wartości w wybranych przestrzeniach barw,
    - inne znormalizowane wskaźniki odporne na zmiany ekspozycji.
  - Funkcja zwraca słownik `{nazwa_cechy: wartość}`, który jest łączony z
    metadanymi (`filepath`, `datetime`, `hour`).

**Zastosowanie:**

- Wejście do bardziej złożonych modeli:
  - MLP regresji cyklicznej na cechach robust (`train_hour_nn_cyclic.py`),
  - inne regresje/klasyfikacje w `train_robust_time.py`.
- Celem jest lepsza odporność na zmiany ekspozycji/kamery niż przy samym RGB/HSV.

---

## 4. `normalize_data.py` – standaryzacja cech

**Cel:**

- Ujednolicić skalę cech numerycznych (np. z `features_advanced.csv`,
  `features_robust.csv`) przez standaryzację \( z' = \frac{z - \mu}{\sigma} \),
  przy czym \( \mu \) i \( \sigma \) liczone są **tylko na zbiorze treningowym**.
- Zapobiega to „data leakage” i stabilizuje trening modeli wrażliwych na skalę.

**Wejście / wyjście:**

- Wejście: dowolny plik z cechami, np.:
  - `features_advanced.csv`,
  - `features_robust.csv`,
  - `features_mean_rgb.csv`.
- Wyjście:
  - `*_normalized.csv` – te same rekordy, ale kolumny cech zastąpione
    znormalizowanymi wartościami,
  - `*_scaler.npz` – zapisane `mean`, `std`, `feature_cols` do późniejszego użycia
    w inference (np. w skryptach overlay/MLP).

**Algorytm:**

1. Wczytaj CSV z cechami.
2. Rozdziel kolumny na:
   - **meta**: `filepath`, `datetime`, `hour`,
   - **cechy numeryczne**: wszystkie pozostałe.
3. Zbuduj macierz \( X \) z cech, wektor \( y \) = `hour`.
4. Użyj `train_test_split` (z `stratify=y`, `train_size` domyślnie 0.8) **tylko po to**,
   aby wydzielić część danych jako „train” do policzenia statystyk.
5. Policz na `X_train`:
   - \( \mu = mean(X_{train}, axis=0) \),
   - \( \sigma = std(X_{train}, axis=0) + 10^{-6} \).
6. Znormalizuj całe \( X \):
   - \( X_{norm} = \frac{X - \mu}{\sigma} \).
7. Złóż znormalizowany `df_norm` i zapisz jako `*_normalized.csv`.
8. Zapisz `mean`, `std` i listę `feature_cols` do `*_scaler.npz`.

**Zastosowanie:**

- Stanowi pomost między precompute a treningiem modeli, które zakładają
  znormalizowane wejścia (szczególnie MLP i regresje).
- Te same statystyki normalizacji są potem używane w skryptach inference
  (np. `camera_hour_overlay_mlp.py`, `camera_hour_overlay_mlp_rpi.py`) tak, aby
  cechy online były w tej samej skali, co podczas treningu.

---

## 5. Podsumowanie przepływu precompute → normalize → train

1. **Precompute**:
   - uruchamiasz kolejno:
     - `python -m src.precompute_mean_rgb`,
     - `python -m src.precompute_features_advanced`,
     - `python -m src.precompute_features_robust`.
2. **Normalize**:
   - dla wybranych plików cech:
     - `python -m src.normalize_data features_advanced.csv`,
     - `python -m src.normalize_data features_robust.csv`,
     - (opcjonalnie) `python -m src.normalize_data features_mean_rgb.csv`.
3. **Train**:
   - modele RGB/advanced/robust korzystają z odpowiednich
     `*_normalized.csv` + zapisanych scalerów.

Ten łańcuch sprawia, że wszystkie modele uczą się na powtarzalnych,
spójnie wyliczonych cechach – zarówno prostych (RGB), jak i bogatych (advanced/robust).
