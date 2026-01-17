# „Zegar biologiczny – przewidywanie godziny doby na podstawie obrazu z kamery”

Autor: …  
Data: …

---

## 1. Wprowadzenie i cel projektu

- Krótkie tło: cykl dobowy, zmienność oświetlenia, „zegar z obrazu”.
- Sformułowanie problemu:
  - Wejście: pojedynczy obraz z kamery USB.
  - Wyjście: przewidywana godzina doby (0–23).
- Główny cel:
  - Zbudować model ML, który na podstawie obrazu szacuje godzinę.
- Cele szczegółowe:
  - Zaprojektować i zebrać własny zbiór danych.
  - Przygotować cechy/obrazy do treningu modeli.
  - Wytrenować i porównać kilka modeli.
  - Ocenić, czy model spełnia założenia zadania i kiedy zawodzi.

---

## 2. Opis i przygotowanie danych

### 2.1. Źródło danych

- Opis kamery i środowiska (np. widok za oknem / scena wewnętrzna).
- Okres zbierania danych (daty, pory roku).
- Informacja o częstotliwości próbkowania (np. 1 klatka / sekundę).

### 2.2. Struktura zbioru danych

- Opis struktury katalogów (np. `dataset/YYYY/MM/DD/HH/…`).
- Przykładowe ścieżki do plików.
- Format obrazów (JPEG, rozdzielczość, RGB).

### 2.3. Etykiety i plik `labels.csv`

- Kolumny (np. `filepath`, `hour`, `datetime`, ewentualne dodatkowe).
- Jak powstaje plik etykiet (skrypty, logika).
- Sprawdzenie spójności:
  - Istnienie plików z `filepath`.
  - Zakres godzin (0–23).
  - Ewentualne problemy (brakujące pliki, duplikaty) i jak zostały rozwiązane.

### 2.4. Wstępna analiza danych

- Rozkład liczby próbek na:
  - godziny doby,
  - dni / miesiące / pory roku.
- Identyfikacja nierównomierności (np. mniej danych w nocy).
- Przykładowe wizualizacje:
  - histogram godzin,
  - liczba próbek na dzień,
  - przykładowe obrazy z różnych godzin.

---

## 3. Przygotowanie danych do trenowania

### 3.1. Podział na zbiory treningowe, walidacyjne i testowe

- Zastosowana strategia:
  - Podział po dniach / przedziałach czasowych (uzasadnienie – unikanie przecieku czasowego).
- Konkretne proporcje (np. 70%/15%/15% dni).
- Liczby próbek w poszczególnych zbiorach.

### 3.2. Ekstrakcja cech

- **Cechy proste (baseline)**:
  - Średnie kanałów RGB, ewentualnie odchylenia, HSV.
- **Cechy „advanced” / „robust”**:
  - Jakie statystyki są liczone (np. percentyle, kanały HSV, normalizacja).
  - Uzasadnienie, dlaczego takie cechy mogą być stabilniejsze (np. różne warunki oświetleniowe).

### 3.3. Normalizacja i skalowanie

- Jakie skalery zastosowano (StandardScaler, RobustScaler itd.).
- Zasada:
  - Dopasowanie skalera na zbiorze treningowym.
  - Zastosowanie tego samego przekształcenia do walidacji i testu.
- Krótkie uzasadnienie (unikanie przecieku informacji).

---

## 4. Modele i metody

### 4.1. Modele bazowe

- Klasyfikator na średnich RGB:
  - np. logistyczna regresja / Random Forest.
- Czego oczekujemy od baseline’u (prosty punkt odniesienia, niska złożoność).

### 4.2. Modele na cechach zaawansowanych / robust

- Skrócony opis:
  - jakie algorytmy: LogisticRegression, RandomForest, Gradient Boosting itp.
  - jakie cechy wejściowe: `features_advanced`, `features_robust`.
- Uzasadnienie:
  - dlaczego cechy robust,
  - jakie problemy mają rozwiązać (np. różne pory roku, pogoda).

### 4.3. Modele cykliczne

- Koncepcja reprezentacji czasu jako punktu na okręgu:
  - kodowanie godziny jako $(\sin, \cos)$.
- Krótki opis:
  - regresja cykliczna w sklearn,
  - MLP w PyTorchu (wejście: cechy robust → wyjście: $(\sin, \cos)$ → dekodowanie na godzinę).
- Dlaczego podejście cykliczne jest lepsze niż klasyczne 0–23:
  - 23:00 i 0:00 są „blisko”, ale w klasycznej numeracji są „daleko”.

### 4.4. Model CNN na obrazach

- Architektura (np. prosty CNN lub transfer learning).
- Preprocessing obrazów (skalowanie do stałej rozdzielczości, normalizacja).
- Różnica w stosunku do modeli na cechach:
  - CNN uczy się cech automatycznie z obrazu.

---

## 5. Eksperymenty i wyniki

### 5.1. Metryki oceny

- Accuracy (dokładność) na zbiorze testowym.
- Top-k accuracy (np. top-3).
- Macierz pomyłek (confusion matrix) dla klas 0–23.
- Błąd cykliczny:
  - np. średni błąd w godzinach (MAE liczone na okręgu).

### 5.2. Wyniki modeli bazowych

- Tabela:
  - model, cechy, accuracy (train/val/test), uwagi.
- Krótkie omówienie:
  - kiedy baseline działa całkiem dobrze, kiedy się myli.

### 5.3. Wyniki modeli zaawansowanych / robust

- Tabela porównawcza:
  - LogisticRegression, RF, itp. na cechach advanced/robust.
- Wykresy:
  - accuracy per klasa (godzina),
  - porównanie z baseline.

### 5.4. Wyniki modeli cyklicznych (MLP / regresja)

- Tabela:
  - model, metryki cykliczne, porównanie z klasyczną klasyfikacją.
- Macierz pomyłek / histogram błędu w godzinach.
- Przykładowe przypadki:
  - godziny najłatwiejsze i najtrudniejsze.

### 5.5. Wyniki CNN (jeśli trenowany)

- Krótkie zestawienie:
  - accuracy / błąd cykliczny,
  - czas treningu,
  - porównanie do modeli na cechach.

---

## 6. Dyskusja i wnioski

### 6.1. Interpretacja wyników

- Które modele sprawdzają się najlepiej i dlaczego.
- Jak wygląda różnica pomiędzy baseline a modelami zaawansowanymi.
- Jakie godziny są najczęściej mylone i z jakimi (analiza macierzy pomyłek).

### 6.2. Ocena względem wymagań zadania

- Czy przygotowane dane są:
  - spójne, wyczyszczone, gotowe do trenowania?
- Czy model spełnia oczekiwania:
  - np. osiąga accuracy powyżej X%,
  - błąd zwykle nie przekracza 1–2 godzin?
- Jeśli nie wszystkie wymagania są spełnione:
  - jasne uzasadnienie (np. ograniczona ilość danych nocnych, silna zmienność pogody).

### 6.3. Ograniczenia projektu

- Ograniczona liczba lokalizacji (jedna scena).
- Możliwe przeuczenie do konkretnego miejsca / kamery.
- Nierównomierny rozkład próbek w czasie.

### 6.4. Możliwe kierunki rozwoju

- Dodanie drugiej sceny / kamery.
- Wykorzystanie sekwencji (kilka klatek zamiast jednej).
- Dalsze udoskonalenie architektury CNN / modeli hybrydowych.

---

## 7. Podsumowanie

- Jedno–dwa akapity streszczenia:
  - co zrobiono (dane → cechy → modele → wyniki),
  - jakie są główne wnioski,
  - czy projekt można uznać za udany wg kryteriów zadania.

---

## 8. Załączniki

### 8.1. Kod projektu

- Odwołanie do repozytorium:
  - struktura katalogów (`src/`, `docs/`, `dataset/`, `models/`, `Logs/`).
- Lista kluczowych skryptów (krótkie opisy).

### 8.2. Lista bibliotek i narzędzi

- Zawartość `requirements.txt` (ew. streszczona),
- narzędzia dodatkowe (tmux, Raspberry Pi, itd.).

### 8.3. Źródła danych

- Opis własnego zbioru „CameraTime” (kamera, lokalizacja, okres).
- Jeśli były używane publiczne dane – lista i linki (tu raczej nie).

### 8.4. Dodatkowe wykresy / tabele

- Pełne macierze pomyłek,
- dodatkowe histogramy i wykresy pomocnicze.
