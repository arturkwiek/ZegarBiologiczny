# 1 Co robi ten kod (baseline.py)?

To jest prosty **skrypt bazowy (baseline)** do sprawdzenia, *czy da się przewidywać godzinę (`hour`) na podstawie średnich kolorów RGB obrazu*. Innymi słowy: „czy sama informacja o tym, jak czerwone/zielone/niebieskie jest zdjęcie, coś mówi o porze dnia”.

Przejdźmy przez to jak przez preparat pod mikroskopem 🧠🔬.

Najpierw importy.
Kod używa NumPy i Pandas do pracy z danymi, scikit-learn do uczenia modelu (regresja logistyczna), oraz kilku metryk do oceny jakości klasyfikacji.

Stała `FEATURES_PATH` wskazuje na plik `features_mean_rgb.csv`. To jest **cache z wcześniej policzonymi cechami** – średnie wartości R, G i B dla każdego obrazu.

Funkcja `main()` zaczyna od sprawdzenia, czy ten plik istnieje.
Jeśli nie – program przerywa działanie i mówi wprost: *najpierw uruchom skrypt, który liczy średnie RGB*. To zabezpieczenie przed trenowaniem modelu na „niczym”.

Potem:

* CSV jest wczytywany do DataFrame’a,
* wypisywany jest jego rozmiar (ile przykładów i kolumn).

Z danych wyciągane są:

* `X`: macierz cech → tylko trzy kolumny: `r_mean`, `g_mean`, `b_mean`,
* `y`: etykieta → kolumna `hour` (najpewniej godzina wykonania zdjęcia, np. 0–23).

Czyli model dostaje **3 liczby** i ma zgadnąć **jedną z wielu klas (godzin)**. Bardzo minimalistyczne podejście – i o to chodzi w baseline.

Następnie dane są dzielone:

* 70% na trening,
* 30% na test,
* `stratify=y` pilnuje, żeby rozkład godzin był podobny w obu zbiorach,
* `random_state=42` gwarantuje powtarzalność wyników (magiczna liczba naukowców).

Model:

* to **LogisticRegression**, ale w wersji wieloklasowej,
* `max_iter=5000` daje mu dużo czasu na zbieżność,
* `n_jobs=-1` każe użyć wszystkich rdzeni CPU.

Model jest trenowany (`fit`), potem przewiduje godziny dla danych testowych (`predict`).

Na koniec kod:

* liczy accuracy (procent poprawnych predykcji),
* drukuje raport klasyfikacji (precision, recall, f1 dla każdej godziny),
* drukuje macierz pomyłek (kto z kim się myli).

Podsumowując w jednym zdaniu:
**ten kod sprawdza, jak dobrze da się przewidzieć godzinę na podstawie samego „koloru” obrazów, używając bardzo prostego modelu jako punktu odniesienia**.

# 2 Czy to jest uczenie maszynowe (ML)?

To jest intelektualny odpowiednik pytania:
„Zanim odpalimy rakietę neuronową, sprawdźmy, czy wystarczy rower”.

Jeśli chcesz, kolejnym naturalnym krokiem jest porównanie tego baseline’u z:

* cechami bardziej złożonymi (histogramy, HSV),
* albo modelem nieliniowym (Random Forest, SVM),
* albo siecią neuronową, która *widzi* coś więcej niż średnią barwę.

# 2 
Tak — **to jest model uczenia maszynowego**, i to w najczystszej, podręcznikowej postaci. 🧠

Konkretnie:

* używa **uczenia nadzorowanego** (supervised learning),
* ma **dane wejściowe** (`r_mean`, `g_mean`, `b_mean`),
* ma **etykiety** (`hour`),
* **uczy się parametrów** na danych treningowych (`fit`),
* a potem **generalizuje** na danych testowych (`predict`).

Algorytm, który tu pracuje, to **regresja logistyczna**. Mimo mylącej nazwy, w tej konfiguracji jest to **klasyfikator**, nie regresja w sensie „przewidywania liczb ciągłych”. Model uczy się granic decyzyjnych w trójwymiarowej przestrzeni RGB i mówi: „ten punkt najbardziej pasuje do godziny 14”, „ten do 7”, itd.

Ważny niuans filozoficzny (i praktyczny):
To jest **model bardzo prosty** i celowo „głupi”. Nie ma pojęcia o kształtach, obiektach ani słońcu na horyzoncie. Widzi tylko średni kolor całego obrazu. Ale właśnie dlatego jest cenny — daje punkt odniesienia. Jeśli bardziej złożony model nie przebije tego baseline’u, to znaczy, że gdzieś po drodze oszukujemy samych siebie.

Czyli:

* tak, to ML,
* tak, naprawdę się uczy,
* nie, to nie jest „AI z filmów”,
* i tak, jego prostota jest jego supermocą.

Dalej świat się robi ciekawszy: można zapytać, *dlaczego* w ogóle kolor miałby kodować porę dnia i kiedy ta hipoteza zaczyna się rozpadać.

# 3 Jak dokładnie działa to „uczenie” w tym kodzie?
Świetne pytanie, bo tu dotykamy sedna „uczenia” zamiast magii. Ten algorytm niczego **nie rozumie** o świecie. On tylko **dopasowuje liczby do liczb** w sprytny sposób. Zobaczmy, co naprawdę się dzieje pod maską.

To, co tu działa, to **regresja logistyczna w wersji wieloklasowej**. Najpierw uproszczenie obrazu świata: każdy obraz zostaje sprowadzony do punktu w 3-wymiarowej przestrzeni. Oś X to średnia czerwieni, Y – zieleni, Z – niebieskiego. Każde zdjęcie to jeden punkt w tej chmurze RGB.

Do każdego punktu przypisana jest etykieta: godzina. Algorytm dostaje więc mnóstwo przykładów typu: „ten punkt → godzina 8”, „tamten punkt → godzina 17”.

Uczenie polega na znalezieniu **zestawu wag i biasów** (czyli liczb), które pozwalają jak najlepiej oddzielić punkty różnych godzin od siebie. Dla każdej godziny model buduje funkcję, która mówi: „jak bardzo ten punkt pasuje do mnie”. Matematycznie to jest prosta kombinacja liniowa: wagi razy RGB plus przesunięcie.

Te surowe wyniki są przepuszczane przez funkcję softmax, która zamienia je na coś w rodzaju prawdopodobieństw. Model nie mówi tylko „to jest 14”, ale w tle liczy: „14 ma 42%, 15 ma 30%, 13 ma 20%, reszta marginalnie”.

Kluczowy moment: **jak model się poprawia?**
Na początku wagi są losowe, więc odpowiedzi są kiepskie. Model porównuje swoje przewidywania z prawdziwymi godzinami i liczy **funkcję straty** (log-loss). Ta strata mówi jedną liczbą, jak bardzo model się myli.

Potem uruchamia się mechanizm uczenia właściwego: **gradient descent**. Algorytm oblicza, w którą stronę i o ile zmienić każdą wagę, żeby strata była mniejsza. To jest czysta matematyka: pochodne, nachylenia, małe kroki w dół „zbocza błędu”.

Ten proces powtarza się setki lub tysiące razy:
– policz przewidywania
– policz błąd
– popraw wagi
– zrób to jeszcze raz

Uczenie kończy się, gdy model przestaje się istotnie poprawiać albo dojdzie do limitu iteracji (`max_iter=5000`).

Efekt końcowy nie jest „wiedzą o porach dnia”. To jest **układ płaszczyzn decyzyjnych** przecinających przestrzeń RGB. Każda płaszczyzna to matematyczna granica: „po tej stronie częściej była godzina 10, po tamtej 11”.

Dlatego ten model:
– jest szybki,
– jest interpretowalny (można spojrzeć na wagi),
– ale jest ślepy na wszystko, co nieliniowe i przestrzenne.

Uczenie tutaj to nie odkrywanie praw natury, tylko **optymalizacja geometrii w trzech wymiarach**. A mimo to czasem działa zaskakująco dobrze, bo świat bywa leniwy i koduje informacje tam, gdzie się ich nie spodziewamy.
