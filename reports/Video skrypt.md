# 1. Celem projektu było:
- Sprawdzić, czy w spektrum światła widzianego kryje się informacja o godzinie. Wydaje się, że maszyny widzą więcej niż my, nawet w prostych systemach zbierania i analizy danych.
- Podstawą miały być bardzo proste, tanie urządzenia oraz jak najprostszy model obliczeniowy.
- W zakresie samego opracowania i implementacji przyjęto założenie całkowitego zdania się na modele językowe i aktualne rozwiązania generatywnej sztucznej inteligencji.

# 2. Akwizycja danych
Pierwszym krokiem było opracowanie aplikacji pozwalającej na zebranie danych.
Posłużył do tego skrypt zapisujący jedną ramkę obrazu pozyskaną z prostej kamery USB co sekundę. Pozwolił on zbierać 3600 próbek na godzinę.
W pierwszym etapie akwizycji danych wykorzystano bardzo tanią kamerę inspekcyjną.
To właśnie ona pozwoliła w przeciągu kilkunastu dni na przełomie 2025 i 2026 roku zebrać blisko 500 tysięcy obrazów.

# 3. Przygotowanie modeli
Jeżeli chodzi o przygotowanie modeli, tu również postanowiłem zdać się na modele językowe generatywnej sztucznej inteligencji - częściowo jako pomysłodawcę użyłem ChatGPT, a w dalszej implementacji, integracji i ciągłym rozwoju korzystałem w VS Code z narzędzia Github Copilot.

Początkowo zależało mi najpierw na czymś bardzo prostym – czymś, co opierałoby się na prostych obliczeniach, a nie skomplikowanym wielowymiarowym lub złożonym modelu.

Dlatego z propozycji uzyskanych z LLM-u wybrałem regresję jako najbardziej intuicyjną metodę interpolacji danych.

W ciągu dalszych prac wielokrotnie ten model był modyfikowany i uzupełniany na bieżąco.

W końcowej fazie zaplanowane zostało wykorzystanie konwolucyjnej sieci neuronowej do porównania wyników.

# 4. Repozytorium

Widząc, co wydarzyło się do dnia dzisiejszego od końcówki lat 80., kiedy to pierwszy raz przepisałem linijkę kodu w języku Basic z książki Atari do mojego Atari 800 XL, postanowiłem całkowicie zdać się na generatywne modele tworzenia kodu.

Ostatecznie w wyniku tej kilku tygodniowej pracy wygenerowane zostało:

- Kod Pythona: 30 plików, łącznie 5798 linii (średnio ok. 190 linii na plik).
- Skrypty shell (.sh): 3 pliki, łącznie 854 linie (średnio ~285 linii na plik – dość rozbudowane skrypty, zwłaszcza synchro_dataset.sh).
- Dokumentacja Markdown (.md): 30 plików, łącznie 4996 linii (średnio ~165 linii na plik).

Łącznie w samym kodzie (Python + SH) masz ok. 6650 linii, a dokumentacja to kolejne ~5000 linii – czyli repo jest mocno udokumentowane (prawie tyle samo linii MD co kodu).

[Podział skryptów i pipeline — zobacz: docs/scripts_groups.md](../docs/scripts_groups.md)

Oczywiście w trakcie prac i analiz powstało również szereg dodatkowych, ostatecznie zbędnych plików, jednak dokumentują one w pewien sposób proces rozwoju projektu. 

# 5. Trenowanie

- Trenowanie odbywało się wielotopowo w miarę przerostu kodu i danych wielokrotnie realizowane od zera aby ocenić jakość, przebieg i wyniki takiego trenowania.

- W końcowej fazie przygotowany został ogólny pipeline trenowania, który automatyzował cały ten proces. 
- Najbardziej czasochłonnym elementem było tworzenie cech na podstawie zebranych danych. 
- Najdłuższy cykl iteracyjny zajął blisko 42 godziny nieprzerwanych obliczeń na komputerze z procesorem Intel i7. 

[Zobacz podsumowanie trenowania w logach: Logs/2025.12.26/Summary.md](../Logs/2025.12.26/Summary.md)

# 6. Wyniki

Wyniki doświadczeń okazały się umiarkowanie zadowalające. 
Początkowo zakładałem, że uzyskany model i aparat będzie bardziej precyzyjny w swoich predykcjach. 

Domyślałem się, że będą występowały znaczne błędy "przypadkowe"/"graniczne", Związany z jakimiś zakłóceniami i zmianami pogodowymi, ale oczekiwałem raczej bardziej dokładnych wyników w typowych warunkach atmosferycznych.

Podsumowując jednak, można stwierdzić, że informacja z przeciętnej, niskobudżetowej kamery USB niesie ze sobą pewną wiedzę na temat bieżącego momentu w cyklu dobowym, jednak nie na tyle dokładną, aby można było z tej informacji czerpać wiedzę na temat konkretnej godziny w ciągu dnia. 

Być może ważnym i godnym uwagi byłoby dodanie kolejnego wskaźnika, a mianowicie upływu czasu lub gradientu zmian na przestrzeni ostatniego przedziału czasowego. Jest to w końcu typowy sposób w jaki ludzie szacują bieżącą godzinę w ciągu dnia. Być może zostanie wzięte to pod uwagę, jeżeli eksperyment będzie kontynuowany w najbliższym czasie.

