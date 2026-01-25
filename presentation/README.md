# Prezentacja (PowerPoint)

Masz dwie opcje:

1) Outline slajdów (łatwe ręczne dopracowanie treści):
- docs/presentation_deck.md

2) Automatyczne wygenerowanie `.pptx` (tytuły + bullet points + notatki):

```bash
pip install python-pptx
python tools/make_presentation_pptx.py
# Windows alternatywnie:
py -3 tools/make_presentation_pptx.py

# jeśli chcesz pominąć parsowanie metryk z Logs/:
python tools/make_presentation_pptx.py --no-log-metrics
```

Wyjście:
- presentation/ZegarBiologiczny.pptx
