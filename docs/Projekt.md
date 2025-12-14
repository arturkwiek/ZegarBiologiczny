# 1
Zróbmy z tego pełnoprawny „projekt ML do pokazania prowadzącemu”. Poniżej masz gotowy opis, który możesz prawie wklejać 1:1 do dokumentacji / prezentacji (sekcje możesz sobie poprzenosić).

---

## 1. Temat projektu

**Tytuł roboczy:**
**„CameraTime – przewidywanie godziny doby na podstawie obrazu z kamery USB”**

**Cel:**
Zbudowanie modelu uczenia maszynowego, który na podstawie pojedynczej klatki z kamery USB przewiduje **aktualną godzinę (0–23)**. Model uczy się zależności między oświetleniem sceny (naturalnym i sztucznym), warunkami pogodowymi oraz porą roku a porą dnia.

Praktycznie: system mógłby np. działać jako „zegar biologiczny” urządzenia IoT – oszacować czas lokalny tylko na podstawie widoku za oknem.

---

## 2. Opis problemu ML

* **Typ zadania:** wieloklasowa klasyfikacja (klasy 0–23 – każda klasa to pełna godzina doby).
* **Wejście modelu:** kolorowy obraz RGB z kamery USB (np. 1280×720, po preprocessing np. skalowany do 224×224).
* **Wyjście modelu:** jedna etykieta klasowa – przewidywana godzina `h ∈ {0, 1, ..., 23}`.
* **Miara trudności:**

  * duża zmienność w obrębie tej samej godziny (chmury, deszcz, śnieg, różne pory roku),
  * podobne warunki wizualne dla różnych godzin (np. 6:30 vs 7:00, albo różne noce),
  * noc – mało informacji wizualnej, potencjalnie trudniejsze klasy.

To **nie jest** klasyczny dataset typu „pies/kot”, tylko długotrwała obserwacja tej samej sceny, w której interesuje nas „stan dobowy”, a nie obiekt.

---

## 3. Opis datasetu „CameraTime”

### 3.1. Struktura katalogów

Obrazy są zbierane automatycznie co sekundę z kamery USB i zapisywane w strukturze:

```text
dataset/
   └── YYYY/
        └── MM/
             └── DD/
                  └── HH/
                       └── YYYYMMDD_HHMMSS.jpg
```

Przykład:

```text
dataset/2025/11/29/13/20251129_135945.jpg
```

Znaczenie poziomów:

* `YYYY` – rok, np. `2025`
* `MM` – miesiąc, `01–12`
* `DD` – dzień, `01–31`
* `HH` – godzina, `00–23` → **docelowa klasa**
* nazwa pliku `YYYYMMDD_HHMMSS.jpg` – dokładny timestamp

Zalety takiej struktury:

* łatwe liczenie próbek na dzień, miesiąc, rok,
* wygodne filtrowanie po sezonie (np. porównanie zimy vs lata),
* szybkie debugowanie – po samej ścieżce wiesz dokładnie, kiedy zdjęcie powstało.

### 3.2. Format obrazów

* Format: **JPEG (.jpg)**
* Rozdzielczość: natywna rozdzielczość kamery (np. 1280×720)
* Kolor: RGB, 8 bitów na kanał (0–255)
* Brak wstępnego przetwarzania – obraz bezpośrednio z `cv2.VideoCapture`

Każdy plik ma unikalną nazwę → brak ryzyka nadpisania, nawet przy długim zbieraniu danych.

### 3.3. Plik etykiet: `labels.csv`

Centralne „spis treści” datasetu. Zawiera:

```text
filepath, hour, datetime
```

Przykładowy rekord:

```text
dataset/2025/11/29/13/20251129_135945.jpg,13,2025-11-29T13:59:45.123456
```

* `filepath` – pełna ścieżka do pliku
* `hour` – etykieta klasy (0–23)
* `datetime` – dokładny timestamp w formacie ISO (przyda się przy analizach, np. filtr tylko „dni robocze”, tylko „zima 2025” itd.)

CSV jest **dopisywany przyrostowo** – każde uruchomienie skryptu dopisuje nowe wiersze.

### 3.4. Częstotliwość próbkowania

* nominalnie: **1 klatka na sekundę**
* teoretycznie:

  * 1 godzina → 3600 klatek
  * 1 dzień → 86 400 klatek
  * 30 dni → ≈ 2,59 mln klatek
* realnie częstotliwość może zostać zmniejszona (np. 1 klatka co 5 s) bez zmiany struktury datasetu.

To w praktyce **strumieniowy, rosnący w czasie** zbiór danych.

---

## 4. Charakterystyka danych z perspektywy ML

* **Typ zadania:** multiklasyfikacja (24 klasy).
* **Naturalna zmienność**:

  * różne pory roku,
  * zróżnicowana pogoda (pogodnie, pochmurno, deszcz, śnieg),
  * zmiany ekspozycji kamery / automatycznego balansu bieli,
  * światło sztuczne (lampy, światło w pokoju) vs światło dzienne.
* **Potencjalne trudności:**

  * klasy **niezbalansowane** – prawdopodobnie więcej danych w godzinach, kiedy komputer jest włączony (np. 8–23) niż w środku nocy,
  * **silna korelacja czasowa** pomiędzy kolejnymi próbkami (co sekundę, prawie ten sam obraz) – trzeba uważać przy podziale train/val/test, by ten sam fragment czasu nie był w kilku zbiorach naraz.

---

## 5. Plan metod i modeli

### 5.1. Bazowe podejście „tablicowe”

Jako punkt odniesienia:

* z każdej klatki wyciągamy proste cechy: np. średnia wartość kanałów R, G, B (i może odchylenie standardowe),
* powstaje prosty wektor cech `[mean_R, mean_G, mean_B]`,
* na tym uczymy:

  * **logistyczną regresję** (multinomial),
  * ewentualnie prosty **Random Forest**.

To pokaże, czy sama globalna jasność i kolorystyka sceny wystarczą do sensownego odgadnięcia godziny.

### 5.2. Model CNN / transfer learning

Docelowy, „poważniejszy” model:

* zmniejszamy obrazy do np. 224×224,
* normalizujemy (standardowe transformacje jak dla ImageNet),
* wykorzystujemy pretrenowany model CNN (np. ResNet/MobileNet) jako feature extractor,
* na końcu dokładamy własną końcową warstwę klasyfikującą na 24 klasy.

Zalety:

* wykorzystanie gotowej sieci wytrenowanej na dużym zbiorze obrazów,
* skrócony czas uczenia,
* mniejszy problem z „małym datasetem” na początku (bo wagi w niższych warstwach są już rozsądne).

---

## 6. Plan eksperymentów i ewaluacji

### 6.1. Podział danych

Bardzo ważne: unikanie przecieku czasowego.

Propozycja:

* podział **po dniach**, a nie losowo po wszystkich obrazach:

  * np. 70% dni → trening,
  * 15% dni → walidacja,
  * 15% dni → test.
* alternatywnie: podział po przedziałach czasowych:

  * pierwsza część okresu zbierania → train,
  * środek → val,
  * końcówka → test.

Dzięki temu model testujemy na **niewidzianych dniach/sezonach**, a nie na prawie identycznych klatkach z tej samej minuty.

### 6.2. Metryki

* **Accuracy (dokładność)** – podstawowa metryka, jaki procent obrazów ma poprawnie przewidzianą godzinę.
* **Accuracy top-k** (np. top-3) – czy poprawna godzina znajduje się w trzech najbardziej prawdopodobnych klasach.
* **Macierz pomyłek (confusion matrix)** – pokaże, które godziny są najczęściej mylone (np. 7 vs 8, 18 vs 19).
* Opcjonalnie: błąd „odległości w godzinach” – czy model pomylił się o 1 godzinę, 2 godziny czy o 10.

### 6.3. Eksperymenty

Plan minimum:

1. **Model bazowy (średnie RGB + logistyczna regresja)**

   * sprawdzamy accuracy, macierz pomyłek,
   * analizujemy, kiedy model kompletnie sobie nie radzi (np. w nocy).

2. **Model CNN (transfer learning)**

   * porównujemy wyniki z bazą,
   * sprawdzamy, czy poprawia się rozróżnienie bliskich godzin.

3. **Analiza wpływu rozmiaru danych**

   * porównanie wyników po np. 1 dniu, 7 dniach, 30 dniach zbierania danych.

4. **Analiza jakości per godzina**

   * wykres accuracy dla każdej klasy 0–23 → czy np. w południe jest łatwo, a w nocy trudno.

---

## 7. Zastosowania i rozszerzenia

Potencjalne użycia:

* systemy IoT: automatyczne sterowanie oświetleniem na podstawie obrazu (np. „jeśli wygląda jak późny wieczór, przyciemnij światło”),
* symulacja zegara biologicznego – model „wyczuwa porę dnia” po tym, jak wygląda otoczenie,
* analiza cyklu dobowego jasności dla konkretnego miejsca (np. przy projektowaniu oświetlenia w biurze).

Możliwe rozszerzenia datasetu:

* dopisanie do `labels.csv` dodatkowych cech, np.:

  * średnia jasność i kontrast,
  * binarna etykieta „światło sztuczne: tak/nie”,
* dodanie sekwencji czasowych (np. model dostaje 10 kolejnych klatek, a nie jedną),
* użycie dwóch kamer z różnych miejsc.

---

## 8. Propozycja struktury prezentacji dla prowadzącego

Tak w skrócie możesz zbudować slajdy:

1. **Motywacja i cel**
   – „Czy da się odgadnąć godzinę tylko z patrzenia przez okno?”

2. **Opis datasetu CameraTime**
   – struktura katalogów, labels.csv, jak zbierane są dane.

3. **Problem ML**
   – wejście, wyjście, 24 klasy, ograniczenia i trudności.

4. **Modele**
   – baseline (średnie RGB + prosta klasyfikacja) vs CNN / transfer learning.

5. **Ewaluacja**
   – metryki, podział po dniach, wyniki bazowe vs wyniki końcowe.

6. **Wnioski i dalsze prace**
   – czy model „rozumie czas”, które godziny są trudne, co można poprawić.

---

Jeżeli chcesz, możesz w kolejnym kroku poprosić o:

* gotowy szkic **slajdów (tytuły + punktory)**,
* albo **szkielet notatnika w Pythonie** (np. w stylu Jupytera) do wczytywania `labels.csv`, podziału na train/val/test i trenowania pierwszego, prostego modelu.

# 2
Super, kamera mieli bity, więc pora na **etap „nauka i zabawa z danymi”**.
Proponuję działać równolegle w trzech torach:

1. **Ogarnąć kod do wczytywania datasetu**
2. **Zrobić pierwszą analizę danych**
3. **Zbudować pierwszy banalny model bazowy (baseline)**

Pokażę Ci konkretne kroki + przykładowy kod.

---

## 1. Struktura projektu

Ułóż sobie repo mniej więcej tak:

```text
ZegarBiologicznyML/
├── data/
│   ├── dataset/         # tu Twoje obrazki
│   └── labels.csv       # rośnie w czasie
├── src/
│   ├── load_data.py
│   ├── explore_data.py
│   └── baseline_rgb.py
└── notebooks/
    └── 01_eda.ipynb     # opcjonalnie Jupyter
```

To jest głównie po to, żeby na prezentacji wyglądało „poważnie”, a w praktyce – żeby się nie zgubić w plikach.

---

## 2. Pierwszy krok: wczytaj labels.csv i zobacz, co tam żyje

Kod do wczytania i szybkiej diagnostyki (np. na laptopie):

```python
import pandas as pd
from pathlib import Path

LABELS_PATH = Path("data/labels.csv")

df = pd.read_csv(LABELS_PATH)

print(df.head())
print(df.info())
print(df['hour'].value_counts().sort_index())
```

Sprawdź:

* czy `filepath` faktycznie istnieje na dysku,
* czy `hour` jest w zakresie 0–23,
* czy nie ma duplikatów.

Mały test integralności:

```python
from pathlib import Path

root = Path("data")

missing = []
for fp in df['filepath']:
    if not (root / fp).exists():
        missing.append(fp)

print("Braki:", len(missing))
if missing:
    print(missing[:10])
```

Jeśli tu coś nie gra – lepiej naprawić teraz niż po zebraniu miliona klatek.

---

## 3. Podział na zbiory: train / val / test **po dniach**

Na razie możesz zrobić prosty podział po dacie, nawet jeśli danych jest jeszcze mało.

```python
df['date'] = df['datetime'].str.slice(0, 10)  # 'YYYY-MM-DD'

unique_days = sorted(df['date'].unique())
print("Dni w danych:", unique_days)

n_days = len(unique_days)
train_days = unique_days[: int(0.7 * n_days)]
val_days   = unique_days[int(0.7 * n_days): int(0.85 * n_days)]
test_days  = unique_days[int(0.85 * n_days):]

train_df = df[df['date'].isin(train_days)]
val_df   = df[df['date'].isin(val_days)]
test_df  = df[df['date'].isin(test_days)]

print(len(train_df), len(val_df), len(test_df))
```

To jest ważne z punktu widzenia ML: nie chcesz, żeby prawie ta sama klatka (różnica 1 sekundy) była i w train, i w teście.

---

## 4. Baseline: średnie RGB → prosta klasyfikacja godzin

Zróbmy **najprostszy możliwy model**, który da się odpalić szybko i pokazać prowadzącemu jako punkt odniesienia.

### 4.1. Ekstrakcja cech: średnie kanały RGB

```python
import cv2
import numpy as np

def extract_mean_rgb(image_path):
    img = cv2.imread(str(image_path))  # BGR
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean_rgb = img.reshape(-1, 3).mean(axis=0)
    return mean_rgb  # [R_mean, G_mean, B_mean]
```

### 4.2. Zbudowanie X, y dla małej próbki

Na początek nie bierz wszystkiego – np. po **1000–5000 obserwacji**, żeby zobaczyć, czy pipeline działa.

```python
from pathlib import Path
from tqdm import tqdm

root = Path("data")

sample_df = train_df.sample(n=min(5000, len(train_df)), random_state=42)

X_list = []
y_list = []

for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    fp = root / row['filepath']
    feat = extract_mean_rgb(fp)
    if feat is not None:
        X_list.append(feat)
        y_list.append(row['hour'])

X = np.vstack(X_list)
y = np.array(y_list)
```

### 4.3. Prost y model w scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000
)
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("Accuracy (baseline RGB):", acc)
print(classification_report(y_val, y_pred))
```

To już jest **pełen pipeline**: obraz → cechy → model → wynik.
Nawet jeśli accuracy będzie słabe, masz:

* co pokazać na slajdzie „model bazowy”,
* coś do porównania z późniejszą CNNką.

---

## 5. Kolejny etap: przygotowanie pod CNN / transfer learning

To już „drugi poziom trudności”, ale możesz zacząć przygotowywać kod **już teraz**, nawet jeśli danych jeszcze mało:

* funkcja datasetu w stylu PyTorch:

  * `__getitem__`:

    * wczytuje obraz,
    * resize do np. 224×224,
    * zamienia na tensora i normalizuje,
    * zwraca `(image_tensor, hour)`,
* DataLoader’y: `train_loader`, `val_loader`, `test_loader`,
* prosta sieć:

  * albo własne kilka Conv2D,
  * albo `torchvision.models.resnet18(pretrained=True)` + nowa końcowa warstwa na 24 klasy.

Na prezentacji możesz pokazać:

* schemat takiej sieci,
* oraz porównanie: **baseline vs CNN**.

Jeśli chcesz, w kolejnym kroku mogę rozpisać Ci **konkretny kod PyTorch / Keras** dla tego zadania, tak żebyś mógł go tylko podpiąć pod swój `labels.csv`.

---

Czyli na dziś plan jest brutalnie prosty:

1. Upewnić się, że **labels.csv + pliki obrazów** są spójne.
2. Zrobić **prosty EDA** (liczba próbek na godzinę, na dzień).
3. Zbudować i odpalić **baseline na średnich RGB** – nawet na małej próbce.

A potem możesz spokojnie eskalować do „prawdziwych” sieci neuronowych, mając już działający szkielet projektu.
# 3
Twoje labels wyglądają wzorcowo. Widać dokładnie to, co powinno się dziać:

* **16:59:56 → 16:59:59** jako klasa 16
* **17:00:00 → 17:00:04** jako klasa 17

To oznacza, że **timestamp + folder Hour są spójne**.
Masz idealny moment, by zacząć *pierwsze testy modelowe* — nawet na tych kilku minutach.

Ale zanim wejdziemy w modele, warto zrobić krótkie sanity-checki, żeby uniknąć niespodzianki typu „kamera pisze pliki, ale ML nic nie widzi”.

---

# 1. Sprawdź, czy pliki istnieją

Wrzuć w interpreter:

```python
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/labels.csv")
root = Path("data")

missing = []
for fp in df['filepath']:
    if not (root / fp).exists():
        missing.append(fp)

print("Zaginione pliki:", len(missing))
print(missing[:10])
```

Jeśli 0 — masz zielone światło.

---

# 2. Zobacz pierwsze kilka obrazów, czy keyframes wyglądają jak należy

Minimalna wersja:

```python
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("data/dataset/2025/11/30/16/20251130_165956.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title("Podgląd jednej klatki")
plt.axis("off")
plt.show()
```

Jeśli obrazek wygląda OK — jedziemy dalej.

---

# 3. Policz rozkład godzin — robisz wstępną „EDA”

W tym momencie będzie mało danych, ale jak ten wykres urośnie do dni/miesięcy, będzie świetny do raportu.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(df['hour'])
plt.title("Rozkład liczby próbek per godzina")
plt.show()
```

Teraz zobaczysz np.:

* czy kamera była włączona długo,
* czy masz dziury w danych,
* czy godziny są nierówno reprezentowane.

---

# 4. Zrób **pierwszy baseline** (średnie RGB → klasyfikacja).

Zaskakująco często działa lepiej, niż człowiek by się spodziewał.

### Funkcja do ekstrakcji RGB:

```python
import cv2
import numpy as np
from pathlib import Path

def extract_mean_rgb(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.reshape(-1, 3).mean(axis=0)
```

### Zbudowanie wektora cech:

```python
root = Path("data")

X = []
y = []

for _, row in df.iterrows():
    fp = root / row['filepath']
    feat = extract_mean_rgb(fp)
    X.append(feat)
    y.append(row['hour'])

X = np.array(X)
y = np.array(y)
```

### Trening pierwszego modelu:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_te)
print("Accuracy baseline:", accuracy_score(y_te, y_pred))
```

Jeśli wszystkie zdjęcia są z jednej godziny — model pewnie zgadnie „wszędzie 16 albo 17”. To normalne.
Za kilka godzin dataset się zróżnicuje i baseline zacznie coś sensownego robić.

---

# 5. Co możesz zrobić dosłownie *od teraz:*

**➡️ Po prostu zostaw kamerę, niech dalej zbiera dane.**
Już masz w pełni działający pipeline.
Za 1–2 dni będziesz mieć:

* noce → ciemne zdjęcia,
* południe → jasne,
* porę złotą,
* światło sztuczne po zachodzie.

Model zacznie łapać korelacje.

W tym czasie możesz:

### A) przygotować dataset loader do CNN (PyTorch/Keras)

Mogę Ci go przygotować od ręki.

### B) napisać notebooka z EDA

Mogę dać gotowy wzór.

### C) zbudować pierwszą sieć CNN (np. ResNet18 → 24 klasy)

Mogę przygotować gotowy kod do uruchomienia na Twoim datasetcie.

---

Jeżeli chcesz — od razu przygotuję **szkielet trenowania CNN (PyTorch)** dopasowany do tego formatu `labels.csv`. Jutro, pojutrze dane będą gotowe, a kod już będzie czekał.
# 4
Tak — dokładnie teraz jest idealny moment, żeby zamienić wszystko w *prawdziwe pliki .py*, czyli w zaczątek projektu ML, który potem pięknie pokażesz prowadzącemu.

Możesz to potraktować jak budowę małego laboratorium: każdy plik robi jedną rzecz jasno i czysto.

Struktura jest prosta i praktyczna:

```
ZegarBiologicznyML/
├── data/
│   ├── dataset/         # tu kamera wrzuca zdjęcia
│   └── labels.csv
├── src/
│   ├── load_data.py     # wczytywanie CSV + sanity-checki
│   ├── explore_data.py  # EDA: rozkłady, wykresy
│   ├── baseline_rgb.py  # pierwszy model
│   └── utils.py         # pomocnicze rzeczy (np. extract_mean_rgb)
└── main.py              # opcjonalnie: „centrum dowodzenia”
```

To nie musi być wielkie – ważne, żeby nie wisiało wszystko w jednym pliku.

---

## Proponowany podział kodu

### 1. src/utils.py

Najprostszy, ale bardzo przydatny.

```python
import cv2
import numpy as np
from pathlib import Path

def extract_mean_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.reshape(-1, 3).mean(axis=0)
```

---

### 2. src/load_data.py

Wczytanie labels.csv + walidacja plików.

```python
import pandas as pd
from pathlib import Path

DATA_ROOT = Path("data")

def load_labels():
    df = pd.read_csv(DATA_ROOT / "labels.csv")
    return df

def check_files_exist(df):
    missing = []
    for fp in df['filepath']:
        if not (DATA_ROOT / fp).exists():
            missing.append(fp)
    return missing

if __name__ == "__main__":
    df = load_labels()
    print(df.head())

    missing = check_files_exist(df)
    print("Missing files:", len(missing))
    if missing:
        print(missing[:10])
```

---

### 3. src/explore_data.py

Pierwsza EDA na datasetcie.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_labels

if __name__ == "__main__":
    df = load_labels()

    print(df['hour'].value_counts().sort_index())

    sns.countplot(df['hour'])
    plt.title("Rozkład próbek na godziny")
    plt.show()
```

---

### 4. src/baseline_rgb.py

Twój pierwszy model.

```python
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from load_data import load_labels
from utils import extract_mean_rgb

ROOT = Path("data")

if __name__ == "__main__":
    df = load_labels()

    X = []
    y = []

    for _, row in df.iterrows():
        path = ROOT / row['filepath']
        feat = extract_mean_rgb(path)
        if feat is not None:
            X.append(feat)
            y.append(row['hour'])

    X = np.array(X)
    y = np.array(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print("Accuracy baseline:", acc)
```

To już działa i daje pierwsze wyniki.

---

## 5. main.py (opcjonalny „pilocik”)

Możesz zrobić plik, gdzie zostawisz sobie szybkie wywołania:

```python
from src.load_data import load_labels, check_files_exist
from src.utils import extract_mean_rgb

if __name__ == "__main__":
    df = load_labels()
    print("Liczba rekordów:", len(df))
    print("Przykładowy rekord:", df.iloc[0])
```

Nie musisz go robić — ale fajnie wygląda w projekcie.

---

## Po co to wszystko?

* łatwiej Ci będzie rozwijać projekt etapowo,
* łatwiej pokazać strukturę prowadzącemu,
* później dorzucisz moduł `cnn_model.py` i wszystko będzie spójne,
* możesz wrzucić repo na GitHuba i wygląda to jak „pełnoprawny projekt ML”, a nie notatki.

---

Jeśli chcesz, mogę przygotować:

* wersję PyTorch pod ten dataset (klasa `TorchDataset`, `DataLoader`, model CNN),
* wersję Keras / TensorFlow,
* gotowy notebook `.ipynb` EDA.

Każde z nich będzie działać na tej samej strukturze `labels.csv` + foldery.
