# Baseline: przegląd modeli bazowych

Ten dokument opisuje, co robią skrypty baseline, na jakich cechach pracują,
jakie algorytmy wykorzystują i w jakim celu są używane.

---

## 1. `baseline_rgb.py` – LogisticRegression na średnich RGB

**Cel:**

- Stworzyć najprostszy możliwy model klasyfikacji godzin na podstawie
  minimalnego zestawu cech: średnich wartości kanałów R, G, B.
- Służy jako tani, szybko trenowalny **baseline referencyjny**.

**Wejście / wyjście:**

- Wejście: `features_mean_rgb_normalized.csv` (z `precompute_mean_rgb.py` + `normalize_data.py`).
- Cechy wejściowe: `r_mean`, `g_mean`, `b_mean` (3 liczby na obraz).
- Etykieta: `hour` (klasa 0–23).
- Wyjście: wytrenowany model `LogisticRegression` zapisany do
  `models/pc/baseline_rgb_model.pkl`.

**Algorytm / pipeline:**

1. Wczytaj znormalizowane cechy RGB.
2. Podziel dane na zbiór treningowy/testowy (`train_test_split`, stratyfikacja po `hour`).
3. Naucz klasyfikator `LogisticRegression` (wieloklasowy, max_iter=5000, równoległy przez `n_jobs=-1`).
4. Zapisz model do `models/pc/baseline_rgb_model.pkl` (pickle).
5. Wypisz:
   - accuracy na zbiorze testowym,
   - `classification_report`,
   - `confusion_matrix`,
   - czas trenowania.

**Zastosowanie:**

- Bardzo lekki punkt odniesienia – sprawdza, ile można „wycisnąć” wyłącznie
  z koloru sceny.

---

## 2. `baseline_advanced.py` – kilka modeli na cechach RGB+HSV

**Cel:**

- Zbudować bogatszy baseline na rozszerzonych cechach (RGB+HSV),
  porównując kilka klasycznych algorytmów i wybierając najlepszy.

**Wejście / wyjście:**

- Wejście: `features_advanced.csv` z `precompute_features_advanced.py`.
- Cechy wejściowe (dynamiczne w zależności od wersji CSV):
  - pełna lista docelowa:
    - `r_mean`, `g_mean`, `b_mean`,
    - `r_std`, `g_std`, `b_std`,
    - `h_mean`, `h_std`,
    - `s_mean`, `v_mean`.
  - skrypt potrafi działać też bez `h_mean`/`h_std` (starsze wersje cech) – wtedy
    korzysta z podzbioru dostępnych kolumn i ostrzega w logach.
- Etykieta: `hour` (klasa 0–23).
- Wyjścia (modele zapisane w `models/pc/`):
  - `baseline_advanced_logreg_model.pkl` (pipeline StandardScaler + LogisticRegression),
  - `baseline_advanced_knn_model.pkl` (StandardScaler + KNeighborsClassifier),
  - `baseline_advanced_rf_model.pkl` (RandomForestClassifier – odchudzony, pod RPi),
  - `baseline_advanced_gb_model.pkl` (GradientBoostingClassifier).

**Algorytmy:**

- `logreg`:
  - Pipeline: `StandardScaler` → `LogisticRegression(max_iter=5000, n_jobs=-1)`.
  - Model liniowy w przestrzeni zeskalowanych cech, dobre odniesienie bazowe.
- `knn`:
  - Pipeline: `StandardScaler` → `KNeighborsClassifier(n_neighbors=15)`.
  - Klasyfikacja k‑NN, sprawdza, na ile „bliskie sąsiedztwo” w przestrzeni cech
    odpowiada podobnym godzinom.
- `rf` (RandomForestClassifier):
  - Parametry odchudzone (dla mniejszego rozmiaru modelu / RPi):
    - `n_estimators=50`, `max_depth=12`, `min_samples_leaf=20`, `n_jobs=-1`, `random_state=42`.
  - Silniejszy, nieliniowy model zespołowy, dobry kompromis między jakością
    a kosztem obliczeniowym.
- `gb` (GradientBoostingClassifier):
  - Domyślne parametry, `random_state=42`.
  - Jeszcze inny typ modelu zespołowego, często dający dobre wyniki na tablicowych cechach.

**Pipeline skryptu:**

1. Wczytaj `features_advanced.csv` i przygotuj macierz X oraz wektor y (`hour`).
2. Podziel dane na train/test (`train_test_split`, stratify=y, test_size=0.3).
3. Zbuduj zestaw modeli (`build_models`).
4. Dla każdego modelu:
   - wytrenuj na train,
   - zapisz do pliku `baseline_advanced_<nazwa>_model.pkl` w `models/pc`,
   - oblicz accuracy na teście.
5. Wybierz model o najwyższym accuracy.
6. Wypisz szczegółowe metryki (`classification_report`, `confusion_matrix`) dla najlepszego modelu.

**Zastosowanie:**

- Główny baseline tablicowy dla całego projektu – dostarcza zarówno konkretny model
  (szczególnie RF do użycia na RPi), jak i porównanie kilku klasycznych algorytmów.

---

## 3. `baseline_advanced_logreg.py` – osobny baseline z LogisticRegression

**Cel:**

- Mieć dedykowaną, prostą ścieżkę treningu **tylko** LogisticRegression
  na wszystkich numerycznych cechach z `features_advanced.csv`.
- Pozwala szybko sprawdzić, jak daleko zajdziemy liniowym modelem
  bez ręcznego wybierania kolumn.

**Wejście / wyjście:**

- Wejście: `features_advanced.csv`.
- Cechy wejściowe:
  - wszystkie kolumny numeryczne oprócz `hour` (wybierane automatycznie przez
    `df.select_dtypes(include=[np.number])`).
- Etykieta: `hour`.
- Wyjście: `models/baseline_advanced_logreg_model.pkl` (uwaga: ten skrypt zapisuje
  jeszcze bez rozbicia na `models/pc`/`models/rpi`).

**Algorytm:**

1. Wczytaj `features_advanced.csv`.
2. Wybierz wszystkie kolumny numeryczne ≠ `hour` jako cechy.
3. Podziel dane na train/test (stratyfikacja po `hour`).
4. Trenuj pipeline:
   - `StandardScaler` → `LogisticRegression(max_iter=5000, n_jobs=-1, multi_class="auto")`.
5. Zapisz model jako `baseline_advanced_logreg_model.pkl` w `models/`.
6. Wypisz accuracy, classification report, confusion matrix.

**Zastosowanie:**

- Niezależna, szybka ścieżka sprawdzająca, jak radzi sobie wyłącznie
  LogisticRegression na pełnym zestawie cech advanced.

---

## 4. `predict_hour.py` – predykcja z pojedynczego obrazu

**Cel:**

- Dać prosty interfejs CLI do sprawdzenia, co „widzi” model baseline
  dla pojedynczego obrazu (np. zdjęcia z dysku).

**Wejście / wyjście:**

- Wejście:
  - ścieżka do obrazu `JPG/PNG` podana w CLI,
  - wytrenowany model zapisany jako `models/pc/best_advanced.joblib`.
- Wyjście:
  - na stdout: przewidywana godzina,
  - jeśli model ma `predict_proba` – top 5 godzin z największym
    prawdopodobieństwem wraz z wartościami.

**Algorytm / pipeline:**

1. Wczytaj model z `models/pc/best_advanced.joblib` (joblib.load).
2. Wczytaj obraz z dysku i wylicz cechy `extract_rgb_hsv_stats` z `src.utils`:
   - dokładnie taki sam format jak w `features_advanced.csv`.
3. Ułóż wektor cech w kształcie `(1, n_features)` i wykonaj `model.predict`.
4. Jeśli model udostępnia `predict_proba`, oblicz rozkład prawdopodobieństwa po klasach
   i wypisz top 5.

**Zastosowanie:**

- Szybki test „na żywo” dla klasycznych modeli bazowych (szczególnie
  tych wytrenowanych przez `baseline_advanced.py`).
- Umożliwia ręczne debugowanie: podejrzenie, dla jakich godzin model ma największą
  pewność dla konkretnego zdjęcia.

---

## 5. Jak baseline'y wpisują się w cały projekt

- `baseline_rgb.py` i `baseline_advanced.py` odpowiadają za klasyczne modele
  na cechach tablicowych (mean RGB / advanced RGB+HSV).
- `baseline_advanced_logreg.py` jest dodatkowym, czystym baseline'em liniowym.
- `predict_hour.py` umożliwia pojedynczą predykcję z linii komend.
- Modele z `baseline_advanced.py` służą m.in. jako:
  - źródło lekkiego modelu RF dla Raspberry Pi (kopiowanego do `models/rpi/`),
  - punkt odniesienia dla nowszych podejść (MLP na cechach robust, CNN na pełnych obrazach).
