#!/bin/bash

# PROJECT_DIR="$HOME/workspace/ZegarBiologicznyML"
# cd "$PROJECT_DIR" || exit 1

echo "=== STATUS ZEGAR BIOLOGICZNY ML ==="
echo
echo "[Kod]"
git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "Brak repo git"
git log -1 --oneline 2>/dev/null || echo "Brak historii git"

echo
echo "[Dane]"
if [ -f labels.csv ]; then
  echo -n "labels.csv: "
  wc -l labels.csv
else
  echo "Brak labels.csv"
fi

echo
echo "[Cechy]"
ls -lh features_*.csv 2>/dev/null || echo "Brak plików cech"

echo
echo "[Modele]"
ls -lh models 2>/dev/null || echo "Brak katalogu models"

echo
echo "[Ostatnie wyniki]"
ls -1 results/baseline_advanced_*.json 2>/dev/null | tail -n 5 || echo "Brak zapisanych wyników"

echo
echo "=== KONIEC STATUSU ==="
