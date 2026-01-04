## Szybka ściąga: uruchamianie skryptów

Poniżej zebrano najważniejsze przykłady wywołań skryptów z katalogu `src/`.
Wszystkie komendy zakładają, że jesteś w katalogu głównym repozytorium
(`ZegarBiologiczny`) i używasz odpowiedniego środowiska Pythona.

---

### 0. Pełny pipeline

- Skrypt: `run_full_pipeline.sh`
- Opis: walidacja danych, ekstrakcja cech, normalizacja, trening wszystkich modeli.

Przykład:

```bash
./run_full_pipeline.sh
./run_full_pipeline.sh precompute_features_advanced   # wznowienie od wybranego kroku
```

---

### 1. Walidacja i eksploracja danych

- Skrypt: `src/load_data.py`
- Opis: sprawdzenie spójności `labels.csv` i istnienia plików.

```bash
python -m src.load_data --data-dir dataset/2025 --labels dataset/2025/labels.csv
```

- Skrypt: `src/explore_data.py`
- Opis: podstawowa eksploracja rozkładu godzin i liczby próbek na dzień.

```bash
python -m src.explore_data
```

- Skrypt: `src/rebuild_labels.py`
- Opis: rekonstrukcja / naprawa pliku `labels.csv` na podstawie struktury katalogów.

```bash
python -m src.rebuild_labels
```

---

### 2. Ekstrakcja cech i normalizacja

- Skrypt: `src/precompute_mean_rgb.py`
- Opis: wyznaczenie średnich wartości kanałów R/G/B (`features_mean_rgb.csv`).

```bash
python -m src.precompute_mean_rgb
```

- Skrypt: `src/precompute_features_advanced.py`
- Opis: cechy advanced (RGB + HSV) → `features_advanced.csv`.

```bash
python -m src.precompute_features_advanced
```

- Skrypt: `src/precompute_features_robust.py`
- Opis: cechy robust → `features_robust.csv`.

```bash
python -m src.precompute_features_robust
```

- Skrypt: `src/normalize_data.py`
- Opis: normalizacja wybranego pliku cech.

```bash
python -m src.normalize_data features_mean_rgb.csv
python -m src.normalize_data features_advanced.csv
python -m src.normalize_data features_robust.csv
```

---

### 3. Modele bazowe (baseline)

- Skrypt: `src/baseline_rgb.py`
- Opis: model bazowy na średnich RGB.

```bash
python -m src.baseline_rgb
```

- Skrypt: `src/baseline_advanced.py`
- Opis: modele na ręcznie wybranych cechach advanced.

```bash
python -m src.baseline_advanced
```

- Skrypt: `src/baseline_advanced_logreg.py`
- Opis: LogisticRegression na pełnym zestawie cech advanced.

```bash
python -m src.baseline_advanced_logreg
```

---

### 4. Modele na cechach robust

- Skrypt: `src/train_robust_time.py`
- Opis: klasyfikacja godzin (binning) na cechach robust.

```bash
python -m src.train_robust_time
```

- Skrypt: `src/train_hour_regression_cyclic.py`
- Opis: regresja cykliczna (sin/cos) w sklearn na `features_robust.csv`.

```bash
python -m src.train_hour_regression_cyclic
```

- Skrypt: `src/train_hour_nn_cyclic.py`
- Opis: sieć MLP (PyTorch) dla regresji cyklicznej.

```bash
python -m src.train_hour_nn_cyclic
```

---

### 5. Predykcja z gotowych modeli

- Skrypt: `src/predict_hour.py`
- Opis: offline'owa predykcja godziny dla pojedynczego obrazu.

```bash
python -m src.predict_hour --image path/to/image.jpg
```

---

### 6. Overlay z kamery (desktop)

- Skrypt: `src/camera_hour_overlay.py`
- Opis: prosty podgląd z kamery z overlayem godziny (modele baseline).

```bash
python -m src.camera_hour_overlay --cam 0 --width 1280 --height 720
# z logowaniem predykcji do domyślnego pliku camera_hour_overlay_log.txt
python -m src.camera_hour_overlay --cam 0 --log-csv ""
# własna ścieżka do logu
python -m src.camera_hour_overlay --cam 0 --log-csv Logs/camera_overlay_pc.csv
```

- Skrypt: `src/camera_hour_overlay_mlp.py`
- Opis: podgląd z kamery z użyciem MLP (robust) lub fallback RF (advanced).

```bash
python -m src.camera_hour_overlay_mlp --cam 0 --width 1280 --height 720
python -m src.camera_hour_overlay_mlp --cam 0 --use_fallback

# logowanie predykcji (domyślnie do camera_hour_overlay_mlp_log.txt)
python -m src.camera_hour_overlay_mlp --cam 0 --log-csv ""
```

---

### 7. Overlay i zapis na Raspberry Pi

- Skrypt: `src/camera_hour_overlay_rpi.py`
- Opis: okresowe zapisywanie obrazu z overlayem godziny do pliku JPG,
	przy użyciu modelu baseline RGB.

```bash
python -m src.camera_hour_overlay_rpi \
	--model models/baseline_rgb_model.pkl \
	--cam 0 --width 1280 --height 720 \
	--interval 2.0 --output ~/www/camera_hour.jpg

# domyślne logowanie do camera_hour_overlay_rpi_log.txt
python -m src.camera_hour_overlay_rpi --model models/baseline_rgb_model.pkl --log-csv ""
```

- Skrypt: `src/camera_hour_overlay_mlp_rpi.py`
- Opis: wariant RPi korzystający z MLP robust (domyślnie) lub fallback RF advanced,
	z możliwością użycia CNN.

```bash
python -m src.camera_hour_overlay_mlp_rpi \
	--cam 0 --width 1280 --height 720 \
	--interval 2.0 --output ~/www/camera_hour.jpg

python -m src.camera_hour_overlay_mlp_rpi --cam 0 --use_fallback

# domyślne logowanie do camera_hour_overlay_mlp_rpi_log.txt
python -m src.camera_hour_overlay_mlp_rpi --cam 0 --log-csv ""
```