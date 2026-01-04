# Statystyki kodu (Python)

Data generowania: 2025-12-25

- Liczba plików Pythona w projekcie (bez `.venv` i `__pycache__`): **21**
- Łączna liczba linii kodu w tych plikach: **3028**

Podsumowanie oparte na komendzie:

```bash
find . -path './.venv' -prune -o -type f -name '*.py' -not -path '*__pycache__*' -print0 | xargs -0 wc -l
```

---

## Logi z pracy modeli online (`camera_*`)

Skrypty pracujące z kamerą (`camera_hour_overlay*.py`) oprócz wyświetlania
overlayu generują także logi z predykcjami, które można później analizować
w celu oceny skuteczności modeli w warunkach rzeczywistych.

### Domyślne pliki logów

Jeżeli parametr `--log-csv`/`--log_csv` nie zostanie podany albo zostanie
podany pusty string (`""`), każdy skrypt zapisuje log do pliku o nazwie
pochodzącej od nazwy pliku Pythona:

- `camera_hour_overlay.py` → `camera_hour_overlay_log.txt`
- `camera_hour_overlay_advanced.py` → `camera_hour_overlay_advanced_log.txt`
- `camera_hour_overlay_mlp.py` → `camera_hour_overlay_mlp_log.txt`
- `camera_hour_overlay_rpi.py` → `camera_hour_overlay_rpi_log.txt`
- `camera_hour_overlay_mlp_rpi.py` → `camera_hour_overlay_mlp_rpi_log.txt`

Ścieżka jest liczona względem katalogu, z którego uruchamiany jest skrypt,
chyba że jawnie podasz inną w `--log-csv`.

### Format wiersza logu

Większość skryptów `camera_*` (zarówno na PC, jak i RPi) zapisuje w
każdym kroku predykcji wiersz o strukturze:

```text
timestamp_iso;true_hour;pred_hour;pred_label;confidence;model_name
```

Gdzie:

- `timestamp_iso` – znacznik czasu systemowego (`datetime.now().isoformat()`).
- `true_hour` – bieżący czas systemowy przeliczony na godzinę dziesiętną
	w przedziale ~[0, 24), np. `19.2333`.
- `pred_hour` – przewidywana godzina w formie liczby zmiennoprzecinkowej:
	- dla modeli klasyfikacyjnych (RF / baseline / CNN): indeks klasy 0–23,
	- dla MLP cyklicznego: wartość ciągła w [0, 24).
- `pred_label` – sformatowana etykieta tekstowa, np. `17:00` lub `17:14`.
- `confidence` – miara pewności predykcji:
	- maksymalne prawdopodobieństwo klasy z `predict_proba`,
	- pseudo‑pewność oparta o normę wektora (MLP sin/cos),
	- puste pole, jeśli model nie zwraca pewności.
- `model_name` – nazwa modelu/trybu, np. `LogisticRegression`,
	`MLP cyclic (robust)`, `RF classify (advanced) [fallback]`,
	`CNN (full frame)`.

### Przykładowe użycia

Kilka typowych wywołań z włączonym logowaniem (domyślna nazwa pliku):

```bash
python -m src.camera_hour_overlay --cam 0 --log-csv ""
python -m src.camera_hour_overlay_mlp --cam 0 --log-csv ""
python -m src.camera_hour_overlay_rpi --model models/baseline_rgb_model.pkl --log-csv ""
python -m src.camera_hour_overlay_mlp_rpi --cam 0 --log-csv ""
```

lub z własną ścieżką:

```bash
python -m src.camera_hour_overlay --cam 0 --log-csv Logs/camera_pc_session1.csv
```

Tak zebrane logi można następnie ładować do Pythona / Pandas i liczyć
np. błąd kołowy między `true_hour` a `pred_hour` w zależności od pory
dnia, typu modelu czy wartości `confidence`.
