#!/usr/bin/env bash

# run_full_pipeline.sh
# Kompletny pipeline: walidacja datasetu, eksploracja, ekstrakcja cech, normalizacja, trening wszystkich modeli.
# Uruchamiaj z katalogu głównego repozytorium.
#
# Użycie (domyślnie z checkpointami):
#   ./run_full_pipeline.sh
#       - uruchamia pipeline i pomija kroki, które mają ukończony checkpoint
#         (pliki Logs/YYYY.MM.DD/checkpoints/<nazwa_kroku>.done)
#
#   ./run_full_pipeline.sh STEP_NAME
#       - ignoruje checkpointy i startuje od zadanego kroku, wykonując go od nowa
#         oraz wszystkie kolejne kroki.
#
# Nazwy kroków (STEP_NAME):
#   load_data
#   explore_data
#   precompute_mean_rgb
#   normalize_mean_rgb
#   baseline_rgb
#   precompute_features_advanced
#   normalize_advanced
#   baseline_advanced
#   baseline_advanced_logreg
#   precompute_features_robust
#   normalize_robust
#   train_robust_time
#   train_hour_regression_cyclic
#   train_hour_nn_cyclic

set -euo pipefail

# Opcjonalny argument: nazwa kroku, od którego chcemy wznowić pipeline.
START_FROM_STEP="${1:-}"
WAITING_FOR_STEP=""

if [ -n "${START_FROM_STEP}" ]; then
  WAITING_FOR_STEP="${START_FROM_STEP}"
  echo "[INFO] Uruchamianie pipeline od kroku: ${START_FROM_STEP}"
fi

# Domyślny interpreter Pythona (możesz nadpisać zmienną PY_CMD w środowisku)
PY_CMD="${PY_CMD:-python3}"

if ! command -v "${PY_CMD}" >/dev/null 2>&1; then
  echo "[ERROR] Interpreter Python '${PY_CMD}' nie został znaleziony w PATH." >&2
  echo "[ERROR] Zainstaluj Python 3 lub uruchom skrypt z ustawionym PY_CMD=/sciezka/do/python." >&2
  exit 127
fi

# Data w formacie YYYY.MM.DD (spójna z istniejącymi logami)
TODAY="$(date +%Y.%m.%d)"
LOG_DIR="Logs/${TODAY}"
mkdir -p "${LOG_DIR}"

# Katalog na pliki checkpointów (po jednym .done na krok w danym dniu)
CHECKPOINT_DIR="${LOG_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

log_step() {
  local name="$1"; shift
  local logfile="${LOG_DIR}/${name}_${TODAY}_log.txt"
  echo "[INFO] === ${name} ===" | tee -a "${logfile}"
  echo "[INFO] Log: ${logfile}"
  # Komendę uruchamiamy w podpowłoce i jasno logujemy ewentualny błąd
  if ! ("$@") >>"${logfile}" 2>&1; then
    echo "[ERROR] Krok '${name}' zakończył się błędem. Sprawdź log: ${logfile}" | tee -a "${logfile}" >&2
    echo "[ERROR] Aby wznowić pipeline od tego kroku, uruchom:" | tee -a "${logfile}" >&2
    echo "[ERROR]   ./run_full_pipeline.sh ${name}" | tee -a "${logfile}" >&2
    exit 1
  fi
}

# Wrapper na log_step, który obsługuje pomijanie kroków do momentu START_FROM_STEP.
run_step() {
  local name="$1"; shift

  # Jeżeli użytkownik podał krok START_FROM_STEP, to do tego momentu tylko pomijamy.
  # W tym trybie ignorujemy checkpointy, bo intencją jest ponowne wykonanie od wybranego kroku.

  # Jeżeli użytkownik podał krok START_FROM_STEP, to do tego momentu tylko pomijamy.
  if [ -n "${WAITING_FOR_STEP}" ]; then
    if [ "${name}" != "${WAITING_FOR_STEP}" ]; then
      echo "[INFO] Pomijam krok '${name}' (czekam na '${WAITING_FOR_STEP}')."
      return 0
    else
      echo "[INFO] Wznawiam pipeline od kroku '${name}'."
      WAITING_FOR_STEP=""
    fi
  fi

  # Tryb bez START_FROM_STEP: korzystamy z checkpointów, żeby pomijać gotowe kroki.
  if [ -z "${START_FROM_STEP}" ]; then
    local checkpoint="${CHECKPOINT_DIR}/${name}.done"
    if [ -f "${checkpoint}" ]; then
      echo "[INFO] Pomijam krok '${name}' (istnieje checkpoint: ${checkpoint})."
      return 0
    fi
  fi

  log_step "${name}" "$@"

  # Po udanym kroku zapisujemy checkpoint (również w trybie START_FROM_STEP).
  local checkpoint="${CHECKPOINT_DIR}/${name}.done"
  : > "${checkpoint}"
}

# 1. Walidacja datasetu (labels + pliki)
run_step "load_data" "${PY_CMD}" -m src.load_data

# 2. Eksploracja danych (rozklady godzin/dni)
#    Uwaga: ten krok otwiera wykresy w matplotlib; jeśli uruchamiasz na serwerze bez GUI,
#    możesz go pominąć lub ustawić backend bezokienkowy w explore_data.py.
run_step "explore_data" "${PY_CMD}" -m src.explore_data

# 3. Ekstrakcja prostych cech RGB (mean)
run_step "precompute_mean_rgb" "${PY_CMD}" -m src.precompute_mean_rgb

# 4. Normalizacja prostych cech RGB
run_step "normalize_mean_rgb" "${PY_CMD}" -m src.normalize_data features_mean_rgb.csv

# 5. Trening modelu baseline na średnich RGB
run_step "baseline_rgb" "${PY_CMD}" -m src.baseline_rgb

# 6. Ekstrakcja cech advanced (RGB + HSV z H/S/V)
run_step "precompute_features_advanced" "${PY_CMD}" -m src.precompute_features_advanced

# 7. Normalizacja cech advanced
run_step "normalize_advanced" "${PY_CMD}" -m src.normalize_data features_advanced.csv

# 8. Trening modeli advanced (ręcznie wybrane cechy)
run_step "baseline_advanced" "${PY_CMD}" -m src.baseline_advanced

# 9. Trening LogisticRegression na wszystkich cechach advanced
run_step "baseline_advanced_logreg" "${PY_CMD}" -m src.baseline_advanced_logreg

# 10. Ekstrakcja cech robust
run_step "precompute_features_robust" "${PY_CMD}" -m src.precompute_features_robust

# 11. Normalizacja cech robust
run_step "normalize_robust" "${PY_CMD}" -m src.normalize_data features_robust.csv

# 12. Trening klasyfikacji godzin na cechach robust (binning)
run_step "train_robust_time" "${PY_CMD}" -m src.train_robust_time

# 13. Trening regresji cyklicznej (tablicowe cechy robust)
run_step "train_hour_regression_cyclic" "${PY_CMD}" -m src.train_hour_regression_cyclic

# 14. Trening sieci MLP do regresji cyklicznej (torch, cechy robust)
run_step "train_hour_nn_cyclic" "${PY_CMD}" -m src.train_hour_nn_cyclic

echo "[INFO] Pipeline zakończony. Logi: ${LOG_DIR}"