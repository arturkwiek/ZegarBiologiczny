# Zegar Biologiczny – przewidywanie godziny z obrazu nieba

Repozytorium implementuje model „biologicznego zegara”, który uczy się
przewidywać godzinę w ciągu doby na podstawie zdjęć nieba.

Projekt składa się z kilku głównych części:

- zbieranie danych z kamery i budowa `dataset/YYYY/MM/DD/HH`,
- ekstrakcja cech (RGB, advanced, robust),
- normalizacja cech i trening modeli (baseline, MLP, CNN),
- uruchamianie overlay na PC i Raspberry Pi.

Szczegółowy opis pipeline'u znajdziesz w:

- docs/quick_guide.md – skrócona ściąga komend,
- docs/train_overview.md – przegląd skryptów treningowych,
- docs/pipeline_pc_rpi.md – ścieżka PC → modele → Raspberry Pi.

---

## 1. Wymagania i instalacja

- Python 3.10+ (zalecane wirtualne środowisko),
- pakiety z requirements.txt,
- Git Bash / WSL lub inny shell z obsługą skryptów .sh (na Windows).

Instalacja zależności:

```bash
pip install -r requirements.txt
```

Większość komend zakłada, że jesteś w katalogu głównym repozytorium
ZegarBiologiczny.

---

## 2. Dane: struktura i konfiguracja

Aktualny rok/katalog danych i ścieżka do labels.csv są zdefiniowane w
pliku konfiguracyjnym:

- src/settings.py

Domyślnie dane są w katalogu:

- dataset/YYYY/ (np. dataset/2026/),
- dataset/YYYY/labels.csv – kolumny: filepath, hour, datetime.

Wszystkie nowe skrypty powinny korzystać z tych stałych zamiast
hard‑kodować ścieżki.

Do wczytywania i walidacji etykiet służy moduł:

- src/load_data.py – funkcje load_labels, check_files_exist, get_default_paths.

---

## 3. Zbieranie nowego datasetu z kamery

Za zbieranie obrazów odpowiada skrypt główny projektu:

- MLDailyHourClock.py

Tworzy on strukturę katalogów dataset/YYYY/MM/DD/HH oraz dopisuje wpisy
do labels.csv.

Przed startem zbierania ustaw odpowiedni rok/katalog w src/settings.py,
np. na 2026. Przykładowe uruchomienie:

```bash
python MLDailyHourClock.py \
  --cam 0 \
  --width 1280 --height 720 \
  --interval 1.0 \
  --temp-copy ~/www/camera_hour.jpg
```

Parametry:

- --cam – indeks kamery (0, 1, …),
- --width, --height – żądana rozdzielczość,
- --interval – odstęp w sekundach między kolejnymi zdjęciami,
- --temp-copy – opcjonalna ścieżka do „ostatniej klatki” (JPG) zapisywanej
  atomowo, np. do katalogu WWW na RPi.

Więcej przykładów: docs/quick_guide.md.

---

## 4. Pełny pipeline ML (PC)

Kluczowy skrypt uruchamiający cały pipeline znajduje się w katalogu głównym:

- run_full_pipeline.sh

Wykonuje on kolejno m.in.:

- load_data, explore_data,
- precompute_mean_rgb, normalize_mean_rgb, baseline_rgb,
- precompute_features_advanced, normalize_advanced,
  baseline_advanced, baseline_advanced_logreg,
- precompute_features_robust, normalize_robust,
  train_robust_time, train_hour_regression_cyclic,
  train_hour_nn_cyclic, train_hour_cnn,
- prepare_rpi_rf – kopiowanie modelu RF do models/rpi/.

Podstawowe wywołanie:

```bash
./run_full_pipeline.sh
```

Skrypt:

- loguje wyniki do Logs/YYYY.MM.DD/,
- tworzy checkpointy w Logs/YYYY.MM.DD/checkpoints/<step>.done,
- pozwala wznowić od wybranego kroku:

```bash
./run_full_pipeline.sh precompute_features_advanced
```

Szczegóły: docs/pipeline_pc_rpi.md.

---

## 5. Ręczne uruchamianie najważniejszych kroków

Jeśli nie chcesz używać pełnego pipeline'u, możesz odpalać wybrane
etapy ręcznie (z katalogu głównego repozytorium):

- Walidacja danych:

  ```bash
  python -m src.load_data
  ```

- Ekstrakcja cech:

  ```bash
  python -m src.precompute_mean_rgb
  python -m src.precompute_features_advanced
  python -m src.precompute_features_robust
  ```

- Normalizacja:

  ```bash
  python -m src.normalize_data features_mean_rgb.csv
  python -m src.normalize_data features_advanced.csv
  python -m src.normalize_data features_robust.csv
  ```

- Modele bazowe i zaawansowane:

  ```bash
  python -m src.baseline_rgb
  python -m src.baseline_advanced
  python -m src.baseline_advanced_logreg
  python -m src.train_robust_time
  python -m src.train_hour_regression_cyclic
  python -m src.train_hour_nn_cyclic
  python -m src.train_hour_cnn
  ```

Pełna ściąga komend: docs/quick_guide.md.

---

## 6. Uruchamianie overlay (PC i Raspberry Pi)

### 6.1. Overlay na PC

Po wytrenowaniu modeli w models/pc/ możesz wyświetlać overlay z kamery
na desktopie:

- Prosty overlay (modele baseline):

  ```bash
  python -m src.camera_hour_overlay --cam 0 --width 1280 --height 720
  ```

- Overlay z MLP (cechy robust) lub fallback RF:

  ```bash
  python -m src.camera_hour_overlay_mlp --cam 0 --width 1280 --height 720
  python -m src.camera_hour_overlay_mlp --cam 0 --use_fallback
  ```

### 6.2. Overlay na Raspberry Pi

Na Raspberry Pi modele trzymane są w katalogu:

- models/rpi/

Kopiowanie z PC (przykład):

```bash
scp models/rpi/baseline_advanced_rf_model.pkl vision@VISION_RPI:~/workspace/ZegarBiologiczny/models/rpi/
scp models/pc/best_mlp_cyclic.pt           vision@VISION_RPI:~/workspace/ZegarBiologiczny/models/rpi/
```

Uruchomienie overlay na RPi:

- Fallback RF (cechy advanced):

  ```bash
  python3 -m src.camera_hour_overlay_mlp_rpi \
    --cam 0 --width 1280 --height 720 --interval 5.0 --use_fallback
  ```

- MLP (cechy robust):

  ```bash
  python3 -m src.camera_hour_overlay_mlp_rpi \
    --cam 0 --width 1280 --height 720
  ```

Opis pełnego pipeline'u PC → RPi: docs/pipeline_pc_rpi.md.

---

## 7. Struktura repozytorium (skrót)

- dataset/ – zebrane dane (rok/dzień/godzina) + labels.csv,
- src/ – kod źródłowy (ekstrakcja cech, trening, overlay, utils),
- models/pc/ – modele trenowane na PC,
- models/rpi/ – modele przygotowane do deploymentu na Raspberry Pi,
- docs/ – dokumentacja techniczna i ściągi,
- Logs/ – logi z pipeline'u oraz skryptów overlay.

---

## 8. Dodatkowe materiały i debugowanie

- docs/features_mean_rgb.md, docs/features_advanced.md,
  docs/features_robust.md – opis cech,
- docs/baseline_overview.md, docs/model_registry.md – przegląd modeli,
- Logs/* – przykładowe logi z wcześniejszych eksperymentów.

W razie problemów przydatne logi i checkpointy znajdziesz w katalogu Logs.

---

## 9. (Opcjonalnie) Prosty setup OpenCV + HTTP w tmux na RPi

Jeżeli potrzebujesz minimalnego środowiska na Raspberry Pi do serwowania
ostatniej klatki z kamery w HTTP, możesz użyć uproszczonego setupu
OpenCV + python -m http.server uruchomionego w tmux.

Przykładowy szkic:

1. Struktura katalogów:

   ```text
   /home/vision/
   ├── vision_app/
   │   └── camera.py
   └── www/
       ├── index.html
       └── camera_hour.jpg
   ```

2. Skrypt camera.py zapisujący co sekundę klatkę do /home/vision/www/camera_hour.jpg.

3. Serwer HTTP:

   ```bash
   cd ~/www
   python3 -m http.server 8080
   ```

4. Oba procesy można trzymać w jednej sesji tmux, żeby działały po
   wylogowaniu z SSH.

Ten prosty setup dobrze współgra z parametrem --temp-copy w
MLDailyHourClock.py lub z wariantami overlay na RPi.
