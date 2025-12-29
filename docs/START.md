# START – przewodnik po dokumentacji

Lista głównych plików dokumentacji w projekcie z krótkim opisem.

## Plik główny

- [README.md](../README.md) – szybki setup kamery + OpenCV + HTTP na Raspberry Pi oraz uruchomienie pełnego pipeline’u `./run_full_pipeline.sh`.

## Dokumenty ogólne

- [docs/Projekt.md](Projekt.md) – opis koncepcji projektu i głównych założeń.
- [docs/Dokumentacja.md](Dokumentacja.md) – ogólna dokumentacja techniczna (szerszy opis niż README).
- [docs/Podsumowanie.md](Podsumowanie.md) – zbiorcze podsumowania etapów prac i wniosków.
- [docs/Podsumowanie etapu.md](Podsumowanie%20etapu.md) – szczegółowe podsumowanie wybranego etapu eksperymentów.
- [docs/CODE_Statistic.md](CODE_Statistic.md) – statystyki / notatki dotyczące kodu i eksperymentów.
- [docs/quick_guide.md](quick_guide.md) – krótka ściąga „jak odpalić” projekt i najważniejsze komendy.
- [docs/short_cmd.md](short_cmd.md) – lista skróconych komend (CLI) do najczęściej używanych zadań.

## Precompute i cechy

- [docs/precompute_overview.md](precompute_overview.md) – szczegółowy opis skryptów precompute (`precompute_*`) i `normalize_data.py`: jakie cechy liczą, po co i jak.
- [docs/features_mean_rgb.md](features_mean_rgb.md) – opis cech `mean RGB` i ich wykorzystania.
- [docs/features_advanced.md](features_advanced.md) – opis cech „advanced” (RGB+HSV) do modeli klasycznych.
- [docs/features_robust.md](features_robust.md) – opis bogatszych cech „robust” używanych przez MLP/regresje.

## Modele baseline

- [docs/baseline_overview.md](baseline_overview.md) – przegląd skryptów baseline (`baseline_*`, `predict_hour.py`), używanych algorytmów i powiązanych modeli.
- [docs/Opis modelu baseline.md](Opis%20modelu%20baseline.md) – narracyjny opis modelu bazowego i jego zachowania.

## Skrypty treningowe i grupy skryptów

- [docs/train_overview.md](train_overview.md) – szczegółowy opis wszystkich skryptów `train_*` (MLP, regresje cykliczne, CNN).
- [docs/scripts_groups.md](scripts_groups.md) – podział skryptów w `src/` na grupy (train, precompute, camera, baseline) z krótkim opisem.

## Modele i pipeline PC↔RPi

- [docs/model_registry.md](model_registry.md) – rejestr modeli (plików w `models/pc` i `models/rpi`), konwencje nazewnicze, changelog zmian.
- [docs/pipeline_pc_rpi.md](pipeline_pc_rpi.md) – pełny opis pipeline’u PC → modele → Raspberry Pi (w tym krok `prepare_rpi_rf`, kopiowanie modeli i uruchamianie overlay na RPi).

## Dodatkowe analizy i notatki

- [docs/Analiza codex.md.ini](Analiza%20codex.md.ini) – notatki/analityka związana z eksperymentami (np. wyniki, obserwacje).

---

Jeśli szukasz konkretnego tematu:

- **Jak liczone są cechy?** → [docs/precompute_overview.md](precompute_overview.md)
- **Jak działają modele bazowe?** → [docs/baseline_overview.md](baseline_overview.md)
- **Jak trenowane są MLP/CNN?** → [docs/train_overview.md](train_overview.md)
- **Jakie są grupy skryptów w `src/`?** → [docs/scripts_groups.md](scripts_groups.md)
- **Jak wygląda pełny pipeline PC → RPi?** → [docs/pipeline_pc_rpi.md](pipeline_pc_rpi.md)
- **Jakie modele i pliki są w `models/`?** → [docs/model_registry.md](model_registry.md)
