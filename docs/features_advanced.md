# Cechy generowane przez `precompute_features_advanced.py`

Ten plik opisuje wszystkie kolumny dostępne w pliku `features_advanced.csv`, generowanym przez moduł `precompute_features_advanced.py`.

## Kolumny meta-danych

- **filepath**  
  Ścieżka względna do pliku obrazu, np. `dataset/2025/11/29/13/20251129_135945.jpg`.

- **datetime**  
  Dokładny znacznik czasu utworzenia klatki w formacie ISO, użyteczny przy analizach sezonowości, dni tygodnia itp.

- **hour**  
  Godzina doby (0–23), wykorzystywana jako etykieta klasy.  
  Typ: liczba całkowita.

## Cechy z obrazu – przestrzeń RGB

Cechy wyliczane są na podstawie obrazu RGB (po konwersji z BGR). Obraz jest spłaszczany do tablicy pikseli, a następnie dla każdego kanału liczone są średnia i odchylenie standardowe.

- **r_mean**  
  Średnia wartość kanału R (Red) po wszystkich pikselach.  
  - Zakres: ~0–255.  
  - Interpretacja: ogólny poziom „ciepła”/czerwieni w scenie.

- **g_mean**  
  Średnia wartość kanału G (Green).  
  - Zakres: ~0–255.  
  - Interpretacja: globalna ilość zieleni; często powiązana z roślinnością lub tonem oświetlenia.

- **b_mean**  
  Średnia wartość kanału B (Blue).  
  - Zakres: ~0–255.  
  - Interpretacja: globalna „niebieskość” sceny; może rosnąć np. przy chłodnym świetle, zmierzchu lub nocą.

- **r_std**  
  Odchylenie standardowe kanału R.  
  - Mierzy zróżnicowanie jasności w kanale czerwonym.  
  - Wyższa wartość oznacza większy kontrast / większe zróżnicowanie obszarów ciemnych i jasnych w czerwieni.

- **g_std**  
  Odchylenie standardowe kanału G.  
  - Pokazuje, jak bardzo zróżnicowany jest kanał zielony na obrazie.

- **b_std**  
  Odchylenie standardowe kanału B.  
  - Informuje o zróżnicowaniu w kanale niebieskim (np. przejścia między jasnym niebem a ciemnymi obiektami).

## Cechy z obrazu – przestrzeń HSV

Obraz RGB jest dodatkowo konwertowany do przestrzeni HSV (Hue, Saturation, Value) w implementacji OpenCV:

- `H` w zakresie [0, 179]
- `S` w zakresie [0, 255]
- `V` w zakresie [0, 255]

Z tego obrazu wykorzystywane są wszystkie kanały H (odcień), S (nasycenie) i V (jasność).

- **h_mean**  
  Średnia wartość kanału Hue (H).  
  - Zakres: ~0–179 (konwencja OpenCV).  
  - Interpretacja: przybliżony dominujący odcień sceny (z zastrzeżeniem, że H jest wielkością cykliczną na kole barw).

- **h_std**  
  Odchylenie standardowe kanału Hue (H).  
  - Informuje o zróżnicowaniu odcieni na obrazie – niskie wartości oznaczają scenę „jednobarwną”, wysokie dużą różnorodność barw.

- **s_mean**  
  Średnia wartość kanału Saturation (S).  
  - Zakres: ~0–255.  
  - Interpretacja: globalny stopień nasycenia kolorów na obrazie.  
  - Niskie wartości: scena „wyprana z kolorów” (mgła, silne zachmurzenie, noc z małą ilością światła).  
  - Wysokie wartości: żywe, nasycone barwy (np. słoneczny dzień, kolorowe obiekty).

- **v_mean**  
  Średnia wartość kanału Value (V) – jasność.  
  - Zakres: ~0–255.  
  - Interpretacja: ogólna jasność sceny.  
  - Niskie wartości: noc / silnie zacienione ujęcia.  
  - Wysokie wartości: jasny dzień, mocno oświetlone wnętrze.

## Podsumowanie

Po uruchomieniu `precompute_features_advanced.py` dla każdej klatki z `labels.csv` otrzymujesz w `features_advanced.csv` wiersz o postaci:

- meta‑dane: `filepath`, `datetime`, `hour`
- cechy RGB: `r_mean`, `g_mean`, `b_mean`, `r_std`, `g_std`, `b_std`
- cechy HSV: `h_mean`, `h_std`, `s_mean`, `v_mean`

Łącznie 10 liczbowych cech opisujących globalny kolor i kontrast sceny oraz podstawową charakterystykę odcienia, nasycenia i jasności.