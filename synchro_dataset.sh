#!/usr/bin/env bash
set -Eeuo pipefail

# synchro_dataset.sh — dwufazowy worker ingestu RPi → Windows z odpornością na przerwania.

SCRIPT_VERSION="2026-01-10.1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#############################################
# Konfiguracja (dostosuj do swojego środowiska)
#############################################

SRC_DIR="/home/vision/workspace/ZegarBiologiczny/dataset"
STAGING_DIR="/home/vision/workspace/ZegarBiologiczny/.staging_dataset"

DEST_USER="vision"
DEST_HOST="192.168.0.199"
DEST_PATH="/g/RPiZegarBiologiczny/dataset"

# Uwierzytelnianie SSH: ustaw PASS lub SSHPASS dla sshpass, albo wskaż klucz prywatny.
PASS="root"
SSH_IDENTITY_FILE=""

LOOP_SLEEP_SEC=60            # częstotliwość prób gdy brak pracy lub host offline
RETRY_BACKOFF_SEC=30         # opóźnienie po błędzie transferu
MIN_FILE_AGE_SEC=30          # bufor czasowy, by nie chwytać plików w trakcie zapisu
RSYNC_TIMEOUT_SEC=60         # odcina wiszące sesje przy śpiącym Windows

# Skanowanie źródła:
# Przy setkach tysięcy plików pełne "find" w każdym cyklu jest kosztowne.
# Dostępne tryby:
# - full: skanuje całe SRC_DIR (dotychczasowe zachowanie)
# - recent-hours: skanuje tylko katalogi dataset/YYYY/MM/DD/HH z ostatnich N godzin + labels.csv
# SCAN_MODE="full"
SCAN_MODE="recent-hours"
SCAN_RECENT_HOURS=23

# rsync (3.2.x) nie pozwala łączyć: --append-verify z --delay-updates ani z --partial-dir.
# Wybierz tryb:
# - append-verify: maks. wznawialność po przerwaniu (domyślnie; bez --delay-updates/--partial-dir)
# - delay-updates: atomowe "publikowanie" plików (bez wznawiania; używa --partial-dir)
RSYNC_TRANSFER_MODE="append-verify"

# --- Opcjonalnie: Wake-on-LAN (wybudzanie Windows tylko gdy jest co synchronizować) ---
# Wymaga zwykle połączenia Ethernet (Wi‑Fi WoL bywa ograniczone) + włączonego WoL w BIOS/UEFI i Windows.
# Ustaw MAC karty sieciowej Windows, np. "AA:BB:CC:DD:EE:FF". Jeśli nie ustawisz MAC, WoL nie zadziała.
WOL_ENABLED=0
WOL_MAC=""
WOL_BROADCAST="255.255.255.255"
WOL_PORT=9
WOL_COOLDOWN_SEC=300         # nie wysyłaj magic packet częściej niż co 5 min
WOL_WAIT_AFTER_SEC=25        # po wysłaniu WoL odczekaj zanim zrobisz kolejną próbę

# Widoczność rsync w tmux/stdout:
# - none: nic nie wypisuje na ekran (tylko log na dysku)
# - progress: jedna linia postępu (domyślnie)
# - files: wypisuje każdą linię rsync na ekran (bardzo gadatliwe)
RSYNC_STDOUT_MODE="progress"
RSYNC_STDOUT_EVERY_N=1       # co ile plików aktualizować postęp (1 = na bieżąco, ale nadal 1 linia)
RSYNC_STDOUT_EVERY_SEC=1     # oraz nie rzadziej niż co ile sekund (gdy N jest duże)

# Wydajność: przy JPG/PNG kompresja SSH/rsync zwykle tylko spowalnia.
# Zostawiamy --compress (dla np. CSV), ale wyłączamy ją dla typowo już skompresowanych formatów.
RSYNC_SKIP_COMPRESS_EXTS="jpg,jpeg,png,gif,webp,mp4,mov,avi,mkv,zip,gz,bz2,xz,7z,rar,pdf"

LOG_FILE="${SCRIPT_DIR}/rsync_ingest_dataset.log"
LOCK_FILE="/tmp/synchro_dataset.lock"

EXCLUDE_GLOBS=("*.tmp" "*.part" "*.swp" ".DS_Store" "Thumbs.db" "*/.rsync-partial/*")

SSH_OPTS=("-o" "StrictHostKeyChecking=no" "-o" "UserKnownHostsFile=/home/vision/.ssh/known_hosts" "-o" "ConnectTimeout=10" "-o" "ServerAliveInterval=15" "-o" "ServerAliveCountMax=3")

STAGED_LAST_COUNT=0
WORKER_STATE="init"
DEST_PATH_WINDOWS=""

ENSURE_REMOTE_LAST_OUTPUT=""
ENSURE_REMOTE_LAST_STATUS=0
LAST_WOL_TS=0

SCAN_ROOTS=()

#############################################
# Inicjalizacja i walidacja
#############################################

umask 077
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"
}

log_rsync_line() {
  local line=$1
  # Tylko do pliku: pełna data+czas. Ekran kontrolujemy osobno (tryby stdout).
  printf '[%s] RSYNC: %s\n' "$(date '+%F %T')" "$line" >>"$LOG_FILE"
}

rsync_stdout_progress() {
  local transferred_count=$1
  local last_file=$2
  local items_per_min=$3
  # \r = nadpisanie bieżącej linii w tmux/terminalu
  printf '\r[%s] RSYNC: %d items (%d/min)... last=%s' "$(date '+%T')" "$transferred_count" "$items_per_min" "$last_file"
}

fatal() {
  log "FATAL: $*"
  exit 1
}

set_state_once() {
  local new_state=$1
  local message=$2
  if [[ "$WORKER_STATE" != "$new_state" ]]; then
    log "$message"
    WORKER_STATE="$new_state"
  fi
}

to_windows_path() {
  local input=$1
  if [[ $input =~ ^/[A-Za-z]/ ]]; then
    local drive_letter=${input:1:1}
    local remainder=${input:3}
    remainder=${remainder#/}
    remainder=${remainder//\//\\}
    if [[ -n $remainder ]]; then
      printf '%s:\\%s' "${drive_letter^^}" "$remainder"
    else
      printf '%s:\\' "${drive_letter^^}"
    fi
    return 0
  fi
  printf '%s' "$input"
}

DEST_PATH_WINDOWS=$(to_windows_path "$DEST_PATH")

[[ -d "$SRC_DIR" ]] || fatal "SRC_DIR nie istnieje: $SRC_DIR"
[[ "$SRC_DIR" != "$STAGING_DIR" ]] || fatal "STAGING_DIR musi być innym katalogiem niż SRC_DIR"
mkdir -p "$STAGING_DIR"

command -v rsync >/dev/null 2>&1 || fatal "Brak rsync w PATH"
command -v ssh >/dev/null 2>&1 || fatal "Brak ssh w PATH"

if [[ -n "$SSH_IDENTITY_FILE" ]]; then
  [[ -r "$SSH_IDENTITY_FILE" ]] || fatal "Klucz SSH niedostępny: $SSH_IDENTITY_FILE"
  SSH_CMD=(ssh "${SSH_OPTS[@]}" -i "$SSH_IDENTITY_FILE")
  RSYNC_RSH="ssh ${SSH_OPTS[*]} -i $SSH_IDENTITY_FILE"
else
  if [[ -z "${SSHPASS:-}" ]]; then
    : "${PASS:?Ustaw PASS lub SSHPASS dla uwierzytelnienia hasłem}"
    export SSHPASS="$PASS"
  fi
  command -v sshpass >/dev/null 2>&1 || fatal "Brak sshpass w PATH"
  SSH_CMD=(sshpass -e ssh "${SSH_OPTS[@]}")
  RSYNC_RSH="sshpass -e ssh ${SSH_OPTS[*]}"
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  fatal "Inna instancja pracuje (lock: $LOCK_FILE)"
fi

trap 'log "Worker kończy (code=$?)"' EXIT

#############################################
# Funkcje narzędziowe
#############################################

should_exclude() {
  local rel=$1
  for pattern in "${EXCLUDE_GLOBS[@]}"; do
    if [[ $rel == $pattern ]]; then
      return 0
    fi
  done
  return 1
}

cleanup_empty_dirs() {
  local base=$1
  find "$base" -mindepth 1 -type d -empty -delete 2>/dev/null || true
}

staging_has_payload() {
  find "$STAGING_DIR" -mindepth 1 ! -path '*/.rsync-partial/*' -print -quit >/dev/null 2>&1
}

ensure_remote_path() {
  local cmd
  if [[ "$DEST_PATH_WINDOWS" == "$DEST_PATH" ]]; then
    cmd=("mkdir" "-p" "--" "$DEST_PATH")
  else
    cmd=("cmd.exe" "/c" "if not exist \"$DEST_PATH_WINDOWS\" mkdir \"$DEST_PATH_WINDOWS\"")
  fi

  # Dlaczego: gdy Windows zwróci błąd (np. uprawnienia/ścieżka), chcemy widzieć treść,
  # a nie mylnie raportować "host offline".
  local output status
  set +e
  output=$("${SSH_CMD[@]}" "${DEST_USER}@${DEST_HOST}" "${cmd[@]}" 2>&1)
  status=$?
  set -e

  ENSURE_REMOTE_LAST_OUTPUT="$output"
  ENSURE_REMOTE_LAST_STATUS=$status

  if (( status != 0 )); then
    log "WARN: Zdalne utworzenie katalogu nie powiodło się (exit=${status}): ${DEST_HOST}:${DEST_PATH}"
    if [[ -n "${output}" ]]; then
      log "WARN: Remote output: ${output}"
    fi
    return 1
  fi
  return 0
}

wol_send_magic_packet() {
  # Minimalna implementacja WoL bez zależności (używa python3, które zwykle jest na RPi).
  # Zwraca 0 jeśli wysłano, !=0 jeśli nie.
  local mac=$1
  local bcast=$2
  local port=$3

  command -v python3 >/dev/null 2>&1 || return 2

  python3 - <<'PY'
import os, socket, sys

mac = os.environ.get('WOL_MAC', '').strip()
bcast = os.environ.get('WOL_BROADCAST', '255.255.255.255').strip()
port = int(os.environ.get('WOL_PORT', '9'))

def parse_mac(m: str) -> bytes:
    m = m.replace('-', ':').replace('.', ':')
    parts = [p for p in m.split(':') if p]
    if len(parts) == 6:
        return bytes(int(p, 16) for p in parts)
    # allow AABBCCDDEEFF
    m2 = ''.join(c for c in m if c.isalnum())
    if len(m2) == 12:
        return bytes(int(m2[i:i+2], 16) for i in range(0, 12, 2))
    raise ValueError('bad mac format')

try:
    mac_bytes = parse_mac(mac)
except Exception as e:
    print(f'WOL: invalid MAC: {e}', file=sys.stderr)
    sys.exit(3)

packet = b'\xff'*6 + mac_bytes*16
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.sendto(packet, (bcast, port))
sock.close()
sys.exit(0)
PY
}

maybe_wake_destination() {
  (( WOL_ENABLED == 1 )) || return 1
  [[ -n "${WOL_MAC}" ]] || return 2

  local now
  now=$(date +%s)
  if (( now - LAST_WOL_TS < WOL_COOLDOWN_SEC )); then
    return 3
  fi

  # Tylko gdy błąd wygląda jak brak hosta / uśpienie / trasa.
  local out
  out="${ENSURE_REMOTE_LAST_OUTPUT}"
  if [[ "$out" == *"No route to host"* || "$out" == *"Connection timed out"* || "$out" == *"Connection refused"* || "$out" == *"Operation timed out"* ]]; then
    log "WARN: Host wygląda na niedostępny/sleep — wysyłam Wake-on-LAN (MAC=${WOL_MAC})."
    LAST_WOL_TS=$now
    WOL_MAC="$WOL_MAC" WOL_BROADCAST="$WOL_BROADCAST" WOL_PORT="$WOL_PORT" wol_send_magic_packet "$WOL_MAC" "$WOL_BROADCAST" "$WOL_PORT" || {
      log "WARN: Nie udało się wysłać WoL (python3/MAC?)."
      return 4
    }
    log "INFO: WoL wysłany — czekam ${WOL_WAIT_AFTER_SEC}s na start Windows/SSH."
    sleep "$WOL_WAIT_AFTER_SEC"
    return 0
  fi

  return 5
}

compute_scan_roots() {
  SCAN_ROOTS=()

  case "${SCAN_MODE}" in
    full)
      SCAN_ROOTS+=("$SRC_DIR")
      ;;
    recent-hours)
      local i rel_dir abs_dir
      for ((i=0; i<SCAN_RECENT_HOURS; i++)); do
        rel_dir=$(date -d "-${i} hour" '+%Y/%m/%d/%H' 2>/dev/null || true)
        [[ -n "$rel_dir" ]] || continue
        abs_dir="$SRC_DIR/$rel_dir"
        [[ -d "$abs_dir" ]] || continue
        SCAN_ROOTS+=("$abs_dir")
      done
      ;;
    *)
      log "WARN: Nieznany SCAN_MODE='${SCAN_MODE}' (używam full)."
      SCAN_ROOTS+=("$SRC_DIR")
      ;;
  esac

  # labels.csv: zwykle dataset/YYYY/labels.csv (sync istotny, a nie chcemy skanować całego drzewa).
  local year year_prev
  year=$(date '+%Y')
  year_prev=$(date -d '-1 day' '+%Y' 2>/dev/null || true)
  [[ -f "$SRC_DIR/$year/labels.csv" ]] && SCAN_ROOTS+=("$SRC_DIR/$year/labels.csv")
  [[ -n "$year_prev" && "$year_prev" != "$year" && -f "$SRC_DIR/$year_prev/labels.csv" ]] && SCAN_ROOTS+=("$SRC_DIR/$year_prev/labels.csv")

  # Fallback: jeśli nic nie istnieje (np. świeży start/inna struktura), wróć do pełnego skanu.
  if (( ${#SCAN_ROOTS[@]} == 0 )); then
    SCAN_ROOTS+=("$SRC_DIR")
  fi
}

stage_ready_items() {
  STAGED_LAST_COUNT=0
  local now rel dest dest_dir mtime prefix
  now=$(date +%s)
  prefix="${SRC_DIR%/}/"
  compute_scan_roots
  while IFS= read -r -d '' file; do
    rel=${file#"$prefix"}
    if should_exclude "$rel"; then
      continue
    fi
    mtime=$(stat -c %Y "$file") || continue
    if (( now - mtime < MIN_FILE_AGE_SEC )); then
      continue
    fi
    dest="$STAGING_DIR/$rel"
    dest_dir=$(dirname -- "$dest")
    mkdir -p "$dest_dir"
    if mv "$file" "$dest"; then
      ((STAGED_LAST_COUNT++))
    else
      log "WARN: Nie mogę przenieść pliku do staging: $file"
    fi
  done < <(find "${SCAN_ROOTS[@]}" -type f ! -path '*/.rsync-partial/*' -print0) || true
  cleanup_empty_dirs "$SRC_DIR"
}

purge_staging_after_success() {
  local tmp="${STAGING_DIR}.purge.$$"
  rm -rf "$tmp"
  mv "$STAGING_DIR" "$tmp"
  mkdir -p "$STAGING_DIR"
  rm -rf "$tmp"
  log "INFO: Lokalne staging wyczyszczone po potwierdzonym transferze."
}

run_rsync() {
  local rsync_opts=("-a" "--compress" "--partial" "--timeout=$RSYNC_TIMEOUT_SEC" "--prune-empty-dirs")

  case "$RSYNC_TRANSFER_MODE" in
    append-verify)
      rsync_opts+=("--append-verify")
      ;;
    delay-updates)
      rsync_opts+=("--partial-dir=.rsync-partial")
      rsync_opts+=("--delay-updates")
      ;;
    *)
      log "FATAL: Nieznany RSYNC_TRANSFER_MODE='$RSYNC_TRANSFER_MODE' (dozwolone: append-verify|delay-updates)"
      return 2
      ;;
  esac

  if [[ -n "${RSYNC_SKIP_COMPRESS_EXTS}" ]]; then
    rsync_opts+=("--skip-compress=${RSYNC_SKIP_COMPRESS_EXTS}")
  fi

  local pattern
  for pattern in "${EXCLUDE_GLOBS[@]}"; do
    rsync_opts+=("--exclude=$pattern")
  done

  # Widoczność w tmux: loguj co jest synchronizowane.
  # -i + out-format daje czytelną listę zmienianych plików.
  rsync_opts+=("--itemize-changes")
  rsync_opts+=("--out-format=%i %n%L")

  rsync_opts+=("-e" "$RSYNC_RSH")
  rsync_opts+=("${STAGING_DIR%/}/" "${DEST_USER}@${DEST_HOST}:${DEST_PATH%/}/")

  # Dlaczego: rsync (szczególnie kod 1) bez stderr jest nieużyteczny diagnostycznie.
  # Logujemy pełny output do pliku i na ekran w tmux, zachowując kod wyjścia rsync.
  local status
  set +e

  log "INFO: RSYNC_TRANSFER_MODE=${RSYNC_TRANSFER_MODE}"
  log "INFO: rsync args: ${rsync_opts[*]}"

  # rsync output: zawsze do logu; na ekran według RSYNC_STDOUT_MODE.
  local line transferred_count=0 last_file="" last_emit_ts
  last_emit_ts=$(date +%s)
  local start_ts
  start_ts=$last_emit_ts

  # Umożliwia aktualizację zmiennych w pętli while po pipe (bash >= 4, job control off).
  shopt -s lastpipe 2>/dev/null || true

  rsync "${rsync_opts[@]}" 2>&1 | while IFS= read -r line; do
    log_rsync_line "$line"

    case "$RSYNC_STDOUT_MODE" in
      none)
        ;;
      files)
        printf '[%s] RSYNC: %s\n' "$(date '+%T')" "$line"
        ;;
      progress)
        # Zliczamy tylko linie itemize-changes (format: <f...<spacja>plik lub >f...)
        # Uwaga: na części bashów regex z "<"/">" w [[ =~ ]] potrafi sypnąć parse error,
        # więc robimy to bezpiecznie przez dopasowanie prefiksu i obcięcie do 1. spacji.
        if [[ "$line" == \<* || "$line" == \>* ]]; then
          if [[ "$line" == *" "* ]]; then
            last_file=${line#* }
            ((transferred_count++))
          fi
        fi

        local now_ts
        now_ts=$(date +%s)
        if (( transferred_count > 0 )) && { (( transferred_count % RSYNC_STDOUT_EVERY_N == 0 )) || (( now_ts - last_emit_ts >= RSYNC_STDOUT_EVERY_SEC )); }; then
          last_emit_ts=$now_ts
          local elapsed items_per_min
          elapsed=$(( now_ts - start_ts ))
          if (( elapsed <= 0 )); then
            items_per_min=0
          else
            items_per_min=$(( transferred_count * 60 / elapsed ))
          fi
          rsync_stdout_progress "$transferred_count" "$last_file" "$items_per_min"
        fi
        ;;
      *)
        # Bezpieczny fallback: nie spamuj, ale nie ukrywaj, że tryb jest zły.
        printf '[%s] WARN: Nieznany RSYNC_STDOUT_MODE=%s (używam progress)\n' "$(date '+%T')" "$RSYNC_STDOUT_MODE" >&2
        RSYNC_STDOUT_MODE="progress"
        ;;
    esac
  done
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$RSYNC_STDOUT_MODE" == "progress" ]]; then
    printf '\n'
  fi
  if (( status == 0 )); then
    return 0
  fi

  log "WARN: rsync zwrócił kod ${status} — staging pozostaje nietknięty."
  return $status
}

#############################################
# Pętla robocza
#############################################

log "Worker start: SRC=$SRC_DIR, STAGING=$STAGING_DIR, DEST=${DEST_USER}@${DEST_HOST}:${DEST_PATH}"
log "Worker version: ${SCRIPT_VERSION}"

while true; do
  log "INFO: Rozpoczynam skanowanie źródła (min age ${MIN_FILE_AGE_SEC}s)."
  stage_ready_items

  if (( STAGED_LAST_COUNT > 0 )); then
    log "INFO: Wykryto ${STAGED_LAST_COUNT} nowych plików, staging gotowy do transferu."
  else
    log "INFO: Skanowanie zakończone – brak nowych plików spełniających warunki."
  fi

  if ! staging_has_payload; then
    set_state_once "idle" "INFO: Brak nowych plików, śpię ${LOOP_SLEEP_SEC}s."
    sleep "$LOOP_SLEEP_SEC"
    continue
  fi

  set_state_once "pending" "INFO: Staging zawiera dane, rozpoczynam transfer."

  if ! ensure_remote_path; then
    maybe_wake_destination || true
    set_state_once "offline" "WARN: Host docelowy niedostępny, ponawiam za ${RETRY_BACKOFF_SEC}s."
    sleep "$RETRY_BACKOFF_SEC"
    continue
  fi

  if run_rsync; then
    set_state_once "draining" "INFO: Transfer potwierdzony, czyszczę staging."
    purge_staging_after_success
  else
    set_state_once "retry" "INFO: Transfer nieudany, spróbuję ponownie po ${RETRY_BACKOFF_SEC}s."
    sleep "$RETRY_BACKOFF_SEC"
    continue
  fi

  set_state_once "cooldown" "INFO: Oczekiwanie na kolejną porcję danych (${LOOP_SLEEP_SEC}s)."
  sleep "$LOOP_SLEEP_SEC"
done
