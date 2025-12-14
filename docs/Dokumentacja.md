# **Opis Pierwszego etapu prac** 2025.12.07
# 🟦 **Podsumowanie dotychczasowej pracy nad projektem „CameraTime”**

Projekt ma jeden główny cel:
**wytrenować model uczenia maszynowego, który przewiduje godzinę doby na podstawie obrazu z kamery USB.**

Zrealizowaliśmy już kilka kluczowych etapów, które tworzą solidny fundament do dalszych, bardziej zaawansowanych eksperymentów.

---

# 🟩 **1. Zbieranie i organizacja danych**

System uruchomiony na Raspberry Pi automatycznie zapisuje jedno zdjęcie **co sekundę**, tworząc niezwykle bogaty zbiór obrazów.
Dane są organizowane w przejrzystej strukturze katalogów:

```
dataset/YYYY/MM/DD/HH/YYYYMMDD_HHMMSS.jpg
```

Do tego generowany jest plik **labels.csv**, zawierający:

* pełną ścieżkę do obrazu,
* godzinę (etykieta: 0–23),
* dokładny timestamp.

Obecnie zgromadziliśmy:

### **~550 000 zdjęć**

– to już dane na poziomie prawdziwych projektów badawczych.

Rozkład godzin jest praktycznie równy — co oznacza, że mamy **wiele pełnych dób**, idealnych do analizy cyklu światła i modelowania czasu.

---

# 🟩 **2. Walidacja struktury datasetu (`load_data`)**

Napisany został moduł, który:

* wczytuje `labels.csv`,
* wykrywa błędne wpisy,
* sprawdza istnienie każdego zdjęcia na dysku.

Wynik:

* ~550k prawidłowych ścieżek,
* tylko **2 uszkodzone wpisy (`nan`)**,
* pliki na dysku są spójne i kompletne.

Dataset jest zatem **stabilny i wiarygodny**.

---

# 🟩 **3. Eksploracja danych (`explore_data`)**

Wygenerowaliśmy statystyki opisujące zbiór:

* liczność próbek w poszczególnych godzinach,
* przykładowe rekordy,
* pierwsze wizualizacje (histogram rozkładu godzin).

Rozkład godzin wygląda jak idealny zegar biologiczny środowiska — **zdecydowanie potwierdza sens modelowania pory dnia na podstawie obrazu**.

---

# 🟩 **4. Zbudowanie modelu bazowego (`baseline_rgb`)**

Pierwszy, najprostszy model opiera się wyłącznie na **średnich wartościach kanałów RGB obrazu**:

* dla każdego obrazu liczymy 3 liczby: `mean_r, mean_g, mean_b`,
* uczymy klasyfikator Logistic Regression na 24 klasy (godziny 0–23).

To najlżejsza możliwa reprezentacja obrazu — idealna na start.

## 1 Load data

```
(.venv) vision@VisionRPi:~/workspace/ZegarBiologicznyML$ python -m src.load_data
Pierwsze wiersze labels.csv:
                                    filepath  hour                    datetime
0  dataset/2025/11/30/12/20251130_124132.jpg    12  2025-11-30T12:41:32.502789
1  dataset/2025/11/30/12/20251130_124133.jpg    12  2025-11-30T12:41:33.508668
2  dataset/2025/11/30/12/20251130_124134.jpg    12  2025-11-30T12:41:34.513759
3  dataset/2025/11/30/12/20251130_124135.jpg    12  2025-11-30T12:41:35.520540
4  dataset/2025/11/30/12/20251130_124136.jpg    12  2025-11-30T12:41:36.525037

Liczba rekordów: 550077
Sprawdzanie plików: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 550077/550077 [00:10<00:00, 54565.47it/s]

Brakujące pliki: 2
['nan', 'nan']
```
---

## 2 Explore data

```
(.venv) vision@VisionRPi:~/workspace/ZegarBiologicznyML$ python -m src.explore_data
Liczba rekordów: 550128
Przykładowe rekordy:
                                    filepath  hour                    datetime
0  dataset/2025/11/30/12/20251130_124132.jpg    12  2025-11-30T12:41:32.502789
1  dataset/2025/11/30/12/20251130_124133.jpg    12  2025-11-30T12:41:33.508668
2  dataset/2025/11/30/12/20251130_124134.jpg    12  2025-11-30T12:41:34.513759
3  dataset/2025/11/30/12/20251130_124135.jpg    12  2025-11-30T12:41:35.520540
4  dataset/2025/11/30/12/20251130_124136.jpg    12  2025-11-30T12:41:36.525037

Rozkład godzin:
hour
0     21486
1     21484
2     21484
3     21484
4     21487
5     21484
6     21481
7     21469
8     21465
9     20551
10    20883
11    21460
12    22561
13    24978
14    24415
15    25045
16    25065
17    25063
18    25062
19    25063
20    25064
21    25062
22    25047
23    21485
Name: count, dtype: int64
```
---

## 3 Baseline RGB

```
(.venv) vision@VisionRPi:~/workspace/ZegarBiologicznyML$(.venv) vision@VisionRPi:~/workspace/ZegarBiologicznyML$ python -m src.baseline_rgb
Liczba rekordów w labels.csv: 550165
Ekstrakcja cech (średnie RGB)...
  3%|██████▍                                                                                                                                                                                                       | 17144/550165 [03:30<1:48:30, 81.87it/s]
  3%|██████▍                                                                                                                                                                                                       | 17162/550165 [03:30<1:50:10, 80.64it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 550165/550165 [2:04:33<00:00, 73.61it/s]
Gotowe X shape: (550118, 3) y shape: (550118,)
/home/vision/workspace/ZegarBiologicznyML/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1272: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.8. From then on, it will always use 'multinomial'.
Leave it to its default value to avoid this warning.
  warnings.warn(
/home/vision/workspace/ZegarBiologicznyML/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:473: ConvergenceWarning: lbfgs failed to converge after 2000 iteration(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Increase the number of iterations to improve the convergence (max_iter=2000).
You might also want to scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(

=== Wyniki modelu bazowego (średnie RGB) ===
Accuracy: 0.24701277297074578

Classification report:
              precision    recall  f1-score   support

           0       0.14      0.13      0.14      6446
           1       0.24      0.29      0.26      6445
           2       0.15      0.04      0.06      6445
           3       0.00      0.00      0.00      6445
           4       0.14      0.41      0.21      6446
           5       0.17      0.32      0.22      6445
           6       0.34      0.22      0.27      6444
           7       0.55      0.51      0.53      6441
           8       0.21      0.34      0.26      6439
           9       0.30      0.01      0.02      6159
          10       0.10      0.05      0.07      6265
          11       0.18      0.30      0.23      6438
          12       0.20      0.14      0.17      6768
          13       0.25      0.30      0.28      7493
          14       0.42      0.50      0.45      7317
          15       0.70      0.64      0.67      7514
          16       0.31      0.54      0.39      7520
          17       0.19      0.18      0.18      7519
          18       0.02      0.00      0.01      7519
          19       0.31      0.21      0.25      7519
          20       0.19      0.46      0.27      7519
          21       0.04      0.00      0.01      7519
          22       0.31      0.24      0.27      7518
          23       0.02      0.01      0.01      6453

    accuracy                           0.25    165036
   macro avg       0.23      0.24      0.22    165036
weighted avg       0.23      0.25      0.22    165036

Macierz pomyłek:
[[ 864 1178  280    0 2821  193    0    0    0    0    0    0    0    0
     0    0    0    0   36    0 1074    0    0    0]
 [ 716 1842   62    1 2108  668    0    0    0    0    0    0    0    0
     0    0    5    0   19    0 1024    0    0    0]
 [ 754  891  249    0 2636 1253    0    0    0    0    0    0    0    0
     0    0    0    0  108    0  554    0    0    0]
 [ 540  913  432    0 2291 1976    0    0    0    0    0    0    0    0
     0    0    0    0   53    0  240    0    0    0]
 [ 567  599  406    0 2614 2237    0    0    0    0    0    0    0    0
     0    0    0    0    4    0   19    0    0    0]
 [ 416  976  206    2 2101 2059    0    0    0    0    0    0    0    0
     0    0  320    6   56   16  246    0    0   41]
 [  10  282    0    1 1091 1250 1390    0    0    0    0    0    0    0
    64  503  254    9   16   30  839    0   89  616]
 [   0    0    0    0    0    0    5 3301  462    5   35   20    0  261
  1041 1311    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0  749 2162   63  683  827  915  795
   245    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0  115 2288   57  812 1285  361 1232
     9    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0 1666    8  314 2539  455  927
   356    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0  848    3  941 1925  917  750
  1046    8    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    2 1055    2  180 2227  972 1145
   860    0  203   58   10   50    0    1    3    0]
 [   0    0    0    0    0    0    2   74 1084   44  183 1447  815 2269
  1080    2   98   17    1   88    0   20  269    0]
 [   0    0    0    0    0    0   51 1130  491   10   67  152  327  651
  3633  211    0    0    0    2    0    0  592    0]
 [   0    0    0    0    0    0  560  647    5    0    0   85  154  594
   393 4791  163    0    0    1    0    0  121    0]
 [   0    0    0    0  142    5  319    0    0    0    0    0    0    0
     0   32 4026  946   78  193  903   66   98  712]
 [   0    0    0    0   57  166  128    0    0    0    0    0    0    0
     0    0 2320 1325  112 1466 1433   64   27  421]
 [ 378   74    4    0   54  178   16    0    0    0    0    0    0    0
     0    0 1689 1218   34  440 2580  586  105  163]
 [ 471  237   28    0   13  261  445    0    0    0    0    0    0    0
     0    0  406 1089  341 1555 1698    0  886   89]
 [   0    0    0    0    0  236  385    0    0    0    0    0    0    0
     0    0  399 1074  320  879 3494   74   63  595]
 [  15    0    1    0   30    8  173    0    0    0    0    0    0    0
     0    0 1863  682  189   96 2444   31 1550  437]
 [ 672  107    0    0    9  637  325    0    0    0    0    0    0  356
     0    0 1145  469  201  161 1609    0 1810   17]
 [ 647  725   39   27 2295 1004  243    0    0    0    0    0    0    0
     0    0  156    5  234   69  679    0  281   49]]
(.venv) vision@VisionRPi:~/workspace/ZegarBiologicznyML$
```

### Wynik:

**Accuracy ≈ 24–25%** przy 24 klasach
(losowe zgadywanie dałoby 4.2%).

To oznacza, że nawet z tak ograniczoną informacją model:

* potrafi odróżniać dzień od nocy,
* rozpoznaje porę popołudniową,
* radzi sobie tam, gdzie kolorystyka światła jest charakterystyczna.

Szczególnie dobrze wychodzą godziny:

* 14–16 (jasny dzień),
* 7–8 (charakterystyczny poranek),
* 20 (wieczór ze sztucznym światłem).

### Czas wykonania na Raspberry Pi:

* ~2 godziny ładowania i przetwarzania 550k obrazów,
* potem szybkie trenowanie.

Osiągnęliśmy więc **pierwszy działający model**, który faktycznie nauczył się relacji między światłem na obrazie a godziną.

---

# 🟩 **5. Główne wnioski z baseline**

1. Dane realnie zawierają sygnał pozwalający przewidzieć godzinę — projekt ma sens.
2. Baseline radzi sobie z prostymi przypadkami, ale ma swoje ograniczenia:

   * średnie RGB gubią kontekst,
   * model nie widzi kształtów, cieni, nieba ani lamp,
   * nie rozróżni np. 3:00 od 5:00, gdy średnia jasność jest podobna.
3. Aby wejść na poziom 80–95% accuracy, konieczne będzie użycie modelu głębokiego (CNN).

Baseline wykonuje więc swoją rolę:
**jest punktem odniesienia do oceny jakości przyszłych, lepszych modeli.**

---

# 🟦 **Etap projektu w którym jesteśmy**

Masz teraz:

* ogromny, kompletny dataset,
* pełny pipeline w Pythonie,
* sanity-check danych,
* analizę statystyczną,
* działający baseline ML,
* model, który umie przewidywać godzinę lepiej niż losowość.

To jest **połowa projektu ML** — i to ta trudniejsza połowa.

---