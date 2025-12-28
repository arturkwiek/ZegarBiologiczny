Jasne — poniżej masz **czystą, uporządkowaną instrukcję od zera**, bez dygresji. Taki **README-style setup**, który po prostu robisz punkt po punkcie.

---

# 🧩 Setup: OpenCV + HTTP uruchomione w `tmux` (Raspberry Pi)

Cel:

* kamera + OpenCV zapisuje obraz co sekundę
* lokalny serwer WWW pokazuje obraz w przeglądarce
* wszystko działa **w tle**, nawet po wylogowaniu (dzięki `tmux`)

---

## 0️⃣ Założenia

* Ubuntu Server na Raspberry Pi
* użytkownik: `vision`
* kamera działa (`/dev/video0`)
* Python + OpenCV zainstalowane

---

## 1️⃣ Struktura katalogów

```text
/home/vision/
├── vision_app/
│   └── camera.py
└── www/
    ├── index.html
    └── camera_hour.jpg
```

---

## 2️⃣ Kod OpenCV zapisujący obraz

**`~/vision_app/camera.py`**

```python
import cv2
import os
import time

cap = cv2.VideoCapture(0)

TMP = "/home/vision/www/camera_hour.jpg.tmp"
DST = "/home/vision/www/camera_hour.jpg"

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    cv2.putText(
        frame,
        time.strftime("%Y-%m-%d %H:%M:%S"),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imwrite(TMP, frame)
    os.replace(TMP, DST)   # zapis atomowy

    time.sleep(1)
```

---

## 3️⃣ Strona HTML

**`~/www/index.html`**

```html
<!doctype html>
<html>
<body style="margin:0; background:#111; display:flex; justify-content:center;">
  <img id="img" style="max-width:100%; height:auto;">
  <script>
    const img = document.getElementById("img");
    function refresh() {
      img.src = "/camera_hour.jpg?t=" + Date.now();
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
```

---

## 4️⃣ Instalacja tmux

```bash
sudo apt update
sudo apt install -y tmux
```

---

## 5️⃣ Uruchomienie wszystkiego w tmux (ręcznie)

### Start sesji

```bash
tmux new -s vision
```

### Panel 1 — OpenCV

```bash
cd ~/vision_app
python3 camera.py
```

### Podziel ekran

```
Ctrl + b
%
```

### Panel 2 — serwer HTTP

```bash
cd ~/www
python3 -m http.server 8080
```

---

## 6️⃣ Odłączenie i powrót

* **odłącz (procesy dalej działają)**

  ```
  Ctrl + b
  d
  ```

* **powrót do sesji**

  ```bash
  tmux attach -t vision
  ```

---

## 7️⃣ Dostęp z przeglądarki

```
http://IP_RASPBERRY:8080/
```

Test bez HTML:

```
http://IP_RASPBERRY:8080/camera_hour.jpg
```

---

## 8️⃣ Jednolinijkowy start (opcjonalnie)

Jeśli chcesz odpalać wszystko jednym poleceniem:

```bash
tmux new -s vision \; \
  send-keys "cd ~/vision_app && python3 camera.py" C-m \; \
  split-window -h \; \
  send-keys "cd ~/www && python3 -m http.server 8080" C-m
```

---

## 9️⃣ Minimalna ściąga tmux

* nowa sesja: `tmux new -s vision`
* powrót: `tmux a -t vision`
* odłącz: `Ctrl+b d`
* zmiana panelu: `Ctrl+b` + strzałki
* zamknij panel: `Ctrl+d`

---

## ✅ Efekt końcowy

* OpenCV działa non-stop
* HTTP serwuje aktualny obraz
* SSH możesz zamknąć
* wszystko żyje w `tmux`

---

# Uruchomienie pełnego pipeline'u ML (ekstrakcja cech + modele)

Repozytorium zawiera skrypt, który odpala **cały pipeline**: walidację danych, eksplorację, ekstrakcję cech (RGB, advanced, robust), normalizację oraz trening wszystkich modeli.

## Wymagania

- zainstalowane zależności z requirements.txt
- uruchamiasz z katalogu głównego repozytorium ZegarBiologiczny
- dostępny `python` odpowiadający środowisku projektu

## Jedno polecenie

```bash
./run_full_pipeline.sh
