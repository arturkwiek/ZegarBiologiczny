# Pełny pipeline ML: PC → modele → Raspberry Pi

## 1. Uruchomienie pipeline'u na PC

Z katalogu głównego repozytorium:

```bash
./run_full_pipeline.sh
```

Skrypt wykona kolejno m.in.:

- `load_data`, `explore_data`
- `precompute_mean_rgb`, `normalize_mean_rgb`, `baseline_rgb`
- `precompute_features_advanced`, `normalize_advanced`, `baseline_advanced`
- `precompute_features_robust`, `normalize_robust`, `train_robust_time`
- `train_hour_regression_cyclic`, `train_hour_nn_cyclic`
- `prepare_rpi_rf` – nowy krok, który kopiuje model RF do `models/rpi/`.

Po zakończeniu otrzymasz:

- `models/pc/baseline_advanced_rf_model.pkl` – model RF trenowany na PC,
- `models/rpi/baseline_advanced_rf_model.pkl` – artefakt gotowy do użycia na RPi,
- `models/pc/best_mlp_cyclic.pt` – checkpoint MLP (cykliczna regresja robust).

## 2. Ponowne trenowanie tylko RF dla RPi

Jeśli chcesz zaktualizować wyłącznie model RF (bez ruszania wcześniejszych kroków):

```bash
./run_full_pipeline.sh baseline_advanced
```

To odpali kroki od `baseline_advanced` do końca, w tym `prepare_rpi_rf`.

## 3. Deploy modeli na Raspberry Pi

Na Raspberry Pi (w katalogu projektu `~/workspace/ZegarBiologiczny`) modele umieszczamy w:

- `models/rpi/baseline_advanced_rf_model.pkl` – wykorzystywane przez `camera_hour_overlay_mlp_rpi.py` w trybie `--use_fallback`,
- `models/rpi/best_mlp_cyclic.pt` – wykorzystywane przez `camera_hour_overlay_mlp_rpi.py` w trybie MLP.

Przykładowe kopiowanie z PC na RPi:

```bash
scp models/rpi/baseline_advanced_rf_model.pkl vision@VISION_RPI:~/workspace/ZegarBiologiczny/models/rpi/
scp models/pc/best_mlp_cyclic.pt           vision@VISION_RPI:~/workspace/ZegarBiologiczny/models/rpi/
```

## 4. Uruchomienie overlay na Raspberry Pi

Fallback RF (cechy advanced):

```bash
python3 -m src.camera_hour_overlay_mlp_rpi \
  --cam 0 --width 1280 --height 720 --interval 5.0 --use_fallback
```

MLP (cechy robust), jeśli dostępny ekstraktor robust i checkpoint MLP:

```bash
python3 -m src.camera_hour_overlay_mlp_rpi \
  --cam 0 --width 1280 --height 720
```

Panel overlay na RPi pokazuje:

- przewidywaną godzinę,
- datę i czas systemowy,
- tryb i nazwę modelu (MLP lub RF fallback),
- pasek pewności + wartość procentową,
- FPS przetwarzania.
