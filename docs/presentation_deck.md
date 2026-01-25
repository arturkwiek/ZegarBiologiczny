# Prezentacja: ZegarBiologiczny — „biologiczny zegar” z obrazów nieba

Poniżej jest gotowy szkielet slajdów (12–15 min, ~14 slajdów). Tekst jest po polsku i opiera się o to, co jest w repo: pipeline, modele, deploy na RPi i metryki z logów w `Logs/`.

---

## 1. Tytuł
**ZegarBiologiczny**

- Predykcja godziny (0–23) na podstawie zdjęć nieba
- ML + pipeline: data → cechy → modele → Raspberry Pi overlay

_Notatki:_ 1 zdanie o motywacji i zastosowaniu (np. automatyczne „wyczucie pory dnia” z obrazu, nawet przy zmiennych warunkach).

---

## 2. Problem i cel
- Wejście: obraz z kamery (niebo / scena zewnętrzna)
- Wyjście: godzina dnia (klasa 0–23) lub czas na okręgu 24h (sin/cos)
- Wyzwania: chmury, ekspozycja, sezonowość, inne kamery

---

## 3. Dane i etykiety
- Dane zbierane przez: `MLDailyHourClock.py`
- Struktura: `dataset/YYYY/MM/DD/HH/*.jpg`
- Etykiety: `labels.csv` z kolumnami: `filepath,hour,datetime`

_Notatki:_ podkreśl „źródło prawdy” ładowania/validacji w `src/load_data.py`.

---

## 4. Pipeline end-to-end
- Walidacja: `python -m src.load_data`
- Ekstrakcja cech: `precompute_*` (RGB / advanced / robust)
- Normalizacja bez leakage: `python -m src.normalize_data <features.csv>`
- Trening modeli: baseline + robust + MLP/CNN
- Artefakty: `models/pc/` i przygotowanie `models/rpi/`

_Notatki:_ pokaż 1 slajd „jak uruchomić”: `./run_full_pipeline.sh`.

---

## 5. Cechy: od prostych do robust
- Mean RGB: szybki baseline (3 liczby)
- Advanced: RGB+HSV statystyki (więcej sygnału)
- Robust: histogramy/entropia/gradient/edge density (odporność na chmury)

---

## 6. Modele (przegląd)
- Baseline RGB: LogisticRegression
- Baseline Advanced: porównanie kilku klasyków (RF/GB/KNN/LogReg), RF jako praktyczny model do RPi
- Robust: podejście cykliczne (sin/cos) + MLP, plus porównania regresorów
- CNN: eksperymentalnie „end-to-end”

---

## 7. Wyniki: baseline RGB (punkt odniesienia)
- Accuracy ~0.24 (naturalny limit prostego meanRGB)
- Wniosek: sam „kolor” daje słaby, ale szybki benchmark

_Źródło (przykład):_ `Logs/baseline_rgb_log.txt`

---

## 8. Wyniki: advanced (RF)
- RandomForest na cechach advanced osiąga accuracy ~0.79
- Wniosek: ręcznie zaprojektowane cechy + klasyk ML potrafią działać zaskakująco dobrze

_Źródło (przykład):_ `Logs/baseline_advanced_log.txt`

---

## 9. Wyniki: MLP cykliczny (robust)
- Predykcja czasu na okręgu 24h: `(sin θ, cos θ)` gdzie `θ = 2πh/24`
- Przykładowo: TEST Cyclic MAE ~0.54h (~33 min), P90 ~1.46h, P95 ~2.15h

_Źródło (przykład):_ `Logs/2025.12.26/train_hour_nn_cyclic_2025.12.26_log.txt`

---

## 10. Deploy: Raspberry Pi overlay
- Desktop: `src/camera_hour_overlay*.py` (podgląd z overlayem)
- RPi: `src/camera_hour_overlay_mlp_rpi.py` (MLP albo fallback RF)
- Artefakty: `models/rpi/…`

_Notatki:_ podkreśl, że pipeline przygotowuje model RF do RPi (`prepare_rpi_rf`).

---

## 11. Synchronizacja danych RPi → PC/Windows
- Worker: `synchro_dataset.sh`
- Dwufazowo: `SRC_DIR` → `STAGING_DIR` → `rsync` → czyszczenie staging po sukcesie
- Optymalizacja skanowania przy dużym dataset: `SCAN_MODE=recent-hours`

---

## 12. Roadmap / dalsze kroki
- Walidacja na różnych kamerach / lokalizacjach
- Time-based split jako standard (redukcja leakage)
- Kompresja / sampling danych, monitoring driftu
- Ewentualnie: model z transfer learning (mała sieć + augmentacje)

---

## Załączniki (opcjonalnie)
- Demo: krótki film/gif z overlay
- 2–3 przykładowe zdjęcia: „rano / południe / wieczór / pochmurno”
