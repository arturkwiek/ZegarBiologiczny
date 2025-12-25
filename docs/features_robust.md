# Cechy generowane przez `precompute_features_robust.py`

Ten plik opisuje wszystkie kolumny dostępne w pliku `features_robust.csv`, generowanym przez moduł `precompute_features_robust.py` na podstawie funkcji `extract_robust_image_features` z `utils_robust.py`.

W porównaniu z prostymi cechami RGB/HSV jest to dużo bogatszy, „robust” zestaw, zaprojektowany tak, aby lepiej radzić sobie z:

- różnymi warunkami pogodowymi (chmury, mgła, deszcz, śnieg),
- zmianami sezonowymi,
- różnicami jasności i nasycenia,
- strukturą obrazu (krawędzie, gradient jasności góra–dół).

## Kolumny meta-danych

- **filepath**  
  Ścieżka względna do obrazu, np. `dataset/2025/11/29/13/20251129_135945.jpg`.

- **datetime**  
  Dokładny znacznik czasu utworzenia klatki w formacie ISO.

- **hour**  
  Godzina doby (0–23), etykieta klasy.  
  Typ: liczba całkowita.

## Podstawowe statystyki RGB i HSV

Obraz BGR jest konwertowany do RGB i HSV, a następnie spłaszczany do wektorów pikseli.

- **r_mean**, **g_mean**, **b_mean**  
  Średnie wartości kanałów R, G, B po wszystkich pikselach.  
  - Zakres: ~0–255.  
  - Interpretacja: globalna kolorystyka sceny (jak w prostszych modułach).

- **r_std**, **g_std**, **b_std**  
  Odchylenia standardowe kanałów R, G, B.  
  - Informują o zróżnicowaniu jasności w danym kanale.  
  - Wyższe wartości → większy kontrast / większa różnorodność barw i natężenia światła.

- **h_mean**, **s_mean**, **v_mean**  
  Średnie wartości kanałów H (odcień), S (nasycenie) i V (jasność) w przestrzeni HSV.  
  - H (0–179 w OpenCV): pozycja na kole barw; średnia daje przybliżony, dominujący odcień sceny (z zastrzeżeniem cykliczności tego kanału).  
  - S (0–255): im większe, tym bardziej nasycone kolory.  
  - V (0–255): im większe, tym jaśniejsza scena.

- **h_std**, **s_std**, **v_std**  
  Odchylenia standardowe H, S i V.  
  - Mierzą zróżnicowanie odcienia, nasycenia i jasności.  
  - Wysokie wartości mogą oznaczać mieszankę bardzo różnych barw oraz bardzo jasnych i bardzo ciemnych obszarów (silny kontrast).

## Percentyle i rozstępy dla S i V

Z wektorów `s` i `v` liczone są wybrane percentyle oraz rozstępy między nimi. To lepiej opisuje rozkład niż sama średnia.

- **v_p10**, **v_p50**, **v_p90**  
  10., 50. (mediana) i 90. percentyl jasności V.  
  - v_p10 – poziom jasności, poniżej którego znajduje się 10% najciemniejszych pikseli.  
  - v_p50 – mediana jasności (połowa pikseli jest jaśniejsza, połowa ciemniejsza).  
  - v_p90 – poziom jasności, poniżej którego znajduje się 90% pikseli (czyli 10% najjaśniejszych jest powyżej tej wartości).

- **v_iqr_90_10**  
  Różnica `v_p90 - v_p10`.  
  - Przybliżona miara „rozpiętości” jasności/lokalnego kontrastu.  
  - Niska wartość → obraz w miarę jednolicie jasny (np. szara, zamglona scena).  
  - Wysoka wartość → duży rozstrzał pomiędzy najciemniejszymi i najjaśniejszymi obszarami.

- **s_p10**, **s_p50**, **s_p90**  
  Analogi percentyli dla kanału S (nasycenie).

- **s_iqr_90_10**  
  Różnica `s_p90 - s_p10`, opisująca rozpiętość nasycenia kolorów.

## Histogramy jasności V i nasycenia S

Dla kanałów V i S wyliczane są histogramy o zadanej liczbie binów (domyślnie 16).  
Histogramy są normalizowane tak, aby suma wszystkich binów wynosiła 1. Każdy bin staje się oddzielną cechą.

### Histogram V (jasność)

- **v_hist_00**, **v_hist_01**, …, **v_hist_15**  
  - Każda cecha reprezentuje udział pikseli, których jasność V mieści się w danym przedziale.  
  - Przedziały są równomiernie rozmieszczone w zakresie [0, 255].  
  - Wartości są nieujemne i sumują się (w przybliżeniu) do 1.

Intuicja:

- jeśli większość masy jest w binach o niskiej jasności → scena ogólnie ciemna, 
- jeśli histogram jest przesunięty w stronę wysokich wartości → scena jasna, dobrze oświetlona, 
- kształt histogramu pozwala odróżnić np. "płasko oświetlone" ujęcia od tych z wyraźnym podziałem na ciemne i jasne obszary.

### Histogram S (nasycenie)

- **s_hist_00**, **s_hist_01**, …, **s_hist_15**  
  - Analogicznie jak dla V – udział pikseli w kolejnych przedziałach nasycenia S (0–255).  
  - Pozwala odróżniać sceny mało nasycone (mgła, śnieg, noc) od scen z bardzo żywymi kolorami.

## Entropia histogramów

Z znormalizowanych histogramów liczone są entropie Shannona. Służy do oceny „równomierności” rozkładu.

- **v_entropy**  
  Entropia histogramu jasności V.  
  - Wysoka entropia → szeroki, „rozlany” rozkład (dużo różnych poziomów jasności, mniej dominujących wartości).  
  - Niska entropia → większość pikseli ma podobną jasność (np. bardzo ciemna lub bardzo jasna scena).

- **s_entropy**  
  Entropia histogramu nasycenia S.  
  - Wysoka entropia → duża różnorodność nasycenia (od bardzo wyblakłych do bardzo nasyconych obszarów).  
  - Niska entropia → scena o dość jednolitym nasyceniu.

## Gradient pionowy jasności (góra vs dół)

Obraz HSV jest analizowany w pionie, aby uchwycić różnicę jasności między górną i dolną częścią kadru.  
Domyślnie górne i dolne fragmenty stanowią po 25% wysokości obrazu.

- **v_top_mean**  
  Średnia jasność V w górnej części obrazu.  
  - Może odpowiadać np. niebu lub górnym partiom sceny.

- **v_bottom_mean**  
  Średnia jasność V w dolnej części obrazu.  
  - Zwykle odpowiada ziemi, ulicy, budynkom, wnętrzom itp.

- **v_top_minus_bottom**  
  Różnica `v_top_mean - v_bottom_mean`.  
  - Dodatnia: góra jaśniejsza niż dół (np. jasne niebo, ciemniejszy dół).  
  - Ujemna: dół jaśniejszy niż góra (np. mocno oświetlone wnętrze, jasna podłoga, ciemne niebo).

Taka cecha bywa bardziej stabilna niż absolutna jasność przy zmianach ekspozycji, a jednocześnie dobrze oddaje strukturę sceny.

## Gęstość krawędzi (edge density)

Z obrazu BGR wyliczany jest obraz w skali szarości, a następnie detektor krawędzi Canny'ego.  
Wynik binarny (krawędź / brak krawędzi) jest sprowadzany do jednej liczby:

- **edge_density**  
  Udział pikseli zaklasyfikowanych jako krawędzie:  
  `edge_density = liczba_pikseli_krawędzi / liczba_wszystkich_pikseli`.  
  - Zakres: [0, 1].  
  - Wysoka wartość → scena bogata w detale i krawędzie (np. miasto, ostre kontury).  
  - Niska wartość → scena gładka, mało kontrastowa (np. jednolite niebo, mgła, śnieg, rozmyte ujęcia).

## Podsumowanie

Po uruchomieniu `precompute_features_robust.py` dla każdej klatki otrzymujesz w `features_robust.csv` wiersz zawierający:

- meta‑dane: `filepath`, `datetime`, `hour`
- globalne statystyki RGB/HSV: średnie i odchylenia (`r_mean`, `g_mean`, `b_mean`, `r_std`, `g_std`, `b_std`, `h_mean`, `s_mean`, `v_mean`, `h_std`, `s_std`, `v_std`)
- percentyle i rozstępy dla S i V (`*_p10`, `*_p50`, `*_p90`, `*_iqr_90_10`)
- znormalizowane histogramy V i S (`v_hist_00..15`, `s_hist_00..15`)
- entropie histogramów (`v_entropy`, `s_entropy`)
- cechy gradientu pionowego jasności (`v_top_mean`, `v_bottom_mean`, `v_top_minus_bottom`)
- gęstość krawędzi (`edge_density`)

Łącznie daje to kilkadziesiąt liczbowych cech opisujących nie tylko jasność i kolor, ale także strukturę i rozkład informacji w obrazie. To dobra baza pod modele, które mają być bardziej odporne na zmienne warunki oświetleniowe i pogodowe niż proste średnie RGB.