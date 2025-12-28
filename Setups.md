# Wymagania środowiskowe

sudo apt update

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate
pip install -r r
pip install opencv-python numpy

sudo apt install -y libgl1 libglib2.0-0 libxcb1 libx11-6 libxext6 libxrender1 libxfixes3 libxi6 libxkbcommon0

sudo apt install -y libxcb-xinerama0 libxcb-randr0 libxcb-render0 libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-util1 libxcb-xkb1

python src\camera_hour_overlay.py --model models\baseline_rgb_model.pkl --cam 0
python src/camera_hour_overlay_advanced.py   --model models/baseline_advanced_logreg_model.pkl   --features_csv features_advanced.csv   --cam 0


# Serwer www

zamknięcie działającego serwera

lsof -t -i:8080 | xargs kill -9

cd ~/www
python3 -m http.server 8080


# Pipeline ML i checkpointy

Do uruchomienia pełnego pipeline'u (walidacja danych, ekstrakcja cech, normalizacja, trening modeli) służy skrypt:

```bash
./run_full_pipeline.sh
```

Domyślnie korzysta on z **checkpointów** zapisywanych w katalogu:

```text
Logs/YYYY.MM.DD/checkpoints/
```

Każdy krok po udanym wykonaniu tworzy tam plik:

```text
<nazwa_kroku>.done
```

Przy kolejnym uruchomieniu tego samego dnia kroki z istniejącym `.done` są pomijane.

Lista głównych kroków:

- `load_data`
- `explore_data`
- `precompute_mean_rgb`
- `normalize_mean_rgb`
- `baseline_rgb`
- `precompute_features_advanced`
- `normalize_advanced`
- `baseline_advanced`
- `baseline_advanced_logreg`
- `precompute_features_robust`
- `normalize_robust`
- `train_robust_time`
- `train_hour_regression_cyclic`
- `train_hour_nn_cyclic`

## Wznowienie od konkretnego kroku

Jeśli któryś krok zakończy się błędem, w logu pojawi się podpowiedź, jak wznowić od tego miejsca, np.:

```bash
./run_full_pipeline.sh normalize_advanced
```

Podanie nazwy kroku jako argumentu oznacza:

- wcześniejsze kroki są pomijane po **nazwie kroku** (nie po checkpointach)
- wskazany krok oraz wszystkie kolejne są wykonywane ponownie (checkpointy są ignorowane)

## Reset checkpointów dla dnia

Aby całkowicie powtórzyć pipeline dla konkretnej daty (np. 2025.12.26), usuń pliki `.done`:

```bash
rm Logs/2025.12.26/checkpoints/*.done
```

lub cały katalog z checkpointami tego dnia:

```bash
rm -r Logs/2025.12.26/checkpoints
```

Po takim resecie `./run_full_pipeline.sh` wykona wszystkie kroki od nowa.