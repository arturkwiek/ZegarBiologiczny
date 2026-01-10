#!/usr/bin/env bash
set -Eeuo pipefail

############################
# synchro_dataset.sh
# Transfer (ingest): RPi -> Windows + usuń źródło po poprawnym skopiowaniu
# Autoryzacja: hasło przez sshpass (ENV: PASS=... lub SSHPASS=...)
############################

############################
# Konfiguracja (ZMIENIAJ TU)
############################

# Źródło na RPi (katalog z datasetem)
SRC_DIR="/home/vision/workspace/ZegarBiologiczny/dataset"

# Windows (SSH)
WIN_USER="vision"
WIN_HOST="192.168.0.199"
PASS="root"

# Cel na Windows (dysk G:)
DEST_DIR_WIN="/g/RPiZegarBiologiczny/dataset"

# Co ile sekund ponawiać próbę
INTERVAL_SEC=60

# Logi i lock
LOG_FILE="/home/vision/rsync_ingest_dataset.log"
LOCK_FILE="/tmp/synchro_dataset.lock"

# Wykluczenia (pliki tymczasowe itp.)
EXCLUDES=(
  "--exclude=*.tmp"
  "--exclude=*.part"
  "--exclude=*.swp"
  "--exclude=.DS_Store"
  "--exclude=Thumbs.db"
  "--exclude=.rsync-partial/"
)

############################
# Hasło (ENV)
############################
# Możesz ustawić:
#   export PASS='twoje_haslo'
# albo bezpośrednio:
#   export SSHPASS='twoje_haslo'
#
# sshpass -e czyta WYŁĄCZNIE SSHPASS, więc mapujemy PASS -> SSHPASS.

if [[ -z "${SSHPASS:-}" ]]; then
  : "${PASS:?Ustaw zmienną PASS (np. export PASS='pass') albo SSHPASS (np. export SSHPASS='pass')}"
  export SSHPASS="$PASS"
fi

############################
# SSH/Rsync komendy
############################
# Uwaga: StrictHostKeyChecking=no usuwa interakcję (pierwsze połączenie),
# ale jest mniej "paranoicznie bezpieczne". W LAN często akceptowalne.
SSH_OPTS=(
  "-o" "StrictHostKeyChecking=no"
  "-o" "UserKnownHostsFile=/home/vision/.ssh/known_hosts"
  "-o" "ConnectTimeout=10"
  "-o" "ServerAliveInterval=15"
  "-o" "ServerAliveCountMax=3"
)

SSH_CMD=(sshpass -e ssh "${SSH_OPTS[@]}")
RSYNC_RSH="sshpass -e ssh ${SSH_OPTS[*]}"

############################
# Funkcje pomocnicze
############################
log() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"
}

cleanup_empty_dirs() {
  # usuń puste katalogi w SRC po rsync --remove-source-files
  find "$SRC_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
}

ensure_remote_dir() {
  # Tworzy katalog docelowy na Windows, jeśli nie istnieje.
  # Zdalnie jesteśmy w cmd.exe, więc mkdir + ścieżka Windows.
  # Zamiana G:/.. na G:\.. dla cmd.exe:
  local win_path_backslashes
  win_path_backslashes="${DEST_DIR_WIN//\//\\}"

  "${SSH_CMD[@]}" "${WIN_USER}@${WIN_HOST}" "mkdir \"$win_path_backslashes\" 2>NUL" \
    >/dev/null 2>&1 || true
}

rsync_once() {
  # --remove-source-files: usuwa pliki na RPi po poprawnym skopiowaniu
  # --partial + --partial-dir: odporność na zerwane połączenia
  rsync -a -v --progress \
    --remove-source-files \
    --partial --partial-dir=".rsync-partial" \
    --prune-empty-dirs \
    "${EXCLUDES[@]}" \
    -e "$RSYNC_RSH" \
    "${SRC_DIR%/}/" \
    "${WIN_USER}@${WIN_HOST}:${DEST_DIR_WIN%/}/"
}

############################
# Lock (żeby nie odpalić 2x)
############################
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Inny proces synchro_dataset.sh już działa (lock: $LOCK_FILE)." >&2
  exit 1
fi

############################
# Start
############################
log "Start ingest worker (password auth). SRC=${SRC_DIR} -> ${WIN_HOST}:${DEST_DIR_WIN}"

if [[ ! -d "$SRC_DIR" ]]; then
  log "BŁĄD: katalog źródłowy nie istnieje: $SRC_DIR"
  exit 2
fi

# Szybki test połączenia (opcjonalny, ale pomaga w diagnostyce)
if ! "${SSH_CMD[@]}" "${WIN_USER}@${WIN_HOST}" "echo ok" >/dev/null 2>&1; then
  log "UWAGA: Nie mogę połączyć się SSH do Windows (${WIN_USER}@${WIN_HOST}). Sprawdź hasło/Firewall/sshd."
fi

while true; do
  ensure_remote_dir

  if rsync_once; then
    cleanup_empty_dirs
    log "OK: rsync zakończony, usunięto źródłowe pliki i puste katalogi."
  else
    log "UWAGA: rsync nieudany, ponawiam po ${INTERVAL_SEC}s."
  fi

  sleep "$INTERVAL_SEC"
done
