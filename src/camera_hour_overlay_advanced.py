import argparse
import pickle
from pathlib import Path


import cv2
import numpy as np
import pandas as pd
import datetime


# ---------- 1) PRÓBA użycia Twojej ekstrakcji advanced ----------
def try_import_repo_feature_extractor():
    """
    Spróbuj zaimportować funkcję z Twojego repo.
    Dostosuj nazwy jeśli u Ciebie są inne.

    Oczekujemy funkcji:
      extract_features_advanced(frame_bgr: np.ndarray) -> dict[str, float]
    """
    candidates = [
        ("src.precompute_features_advanced", "extract_features_advanced"),
        ("src.precompute_features_advanced", "compute_features_advanced"),
        ("src.precompute_features_advanced", "extract_features_from_frame"),
    ]
    for module_name, fn_name in candidates:
        try:
            mod = __import__(module_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name)
            return fn, f"{module_name}.{fn_name}"
        except Exception:
            continue
    return None, None


# ---------- 2) FALLBACK (tylko jako szkielet, zwykle NIEZGODNY z treningiem) ----------
def fallback_extract_features_advanced(frame_bgr: np.ndarray) -> dict[str, float]:
    """
    Minimalny przykład "advanced" cech.
    UWAGA: prawie na pewno NIE będzie zgodny z Twoim features_advanced.csv.
    To jest tylko awaryjny szkielet, żebyś miał działający pipeline end-to-end.
    """
    # BGR -> RGB / HSV / Gray
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    feats: dict[str, float] = {}

    # Statystyki RGB
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    feats["r_mean"] = float(r.mean())
    feats["g_mean"] = float(g.mean())
    feats["b_mean"] = float(b.mean())
    feats["r_std"] = float(r.std())
    feats["g_std"] = float(g.std())
    feats["b_std"] = float(b.std())

    # Jasność / kontrast (na gray)
    feats["gray_mean"] = float(gray.mean())
    feats["gray_std"] = float(gray.std())

    # HSV: średnia saturacja i value
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    feats["sat_mean"] = float(s.mean())
    feats["val_mean"] = float(v.mean())
    feats["sat_std"] = float(s.std())
    feats["val_std"] = float(v.std())

    # Prosty histogram jasności (np. 8 binów)
    hist = cv2.calcHist([gray], [0], None, [8], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    for i, p in enumerate(hist):
        feats[f"gray_hist_{i}"] = float(p)

    return feats


def pretty_hour_label(hour: int) -> str:
    if 0 <= int(hour) <= 23:
        return f"{int(hour):02d}:00"
    return str(int(hour))


def draw_overlay(frame: np.ndarray, text: str, confidence: float | None, fps: float | None) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = 90

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # Tekst główny: cień czarny + biały
    # Data i czas systemowy
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Tekst główny: cień czarny + biały
    cv2.putText(frame, text, (20+2, 38+2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Data/czas w prawym górnym rogu panelu
    date_x = w - 350
    date_y = 38
    cv2.putText(
        frame,
        now_str,
        (date_x+2, date_y+2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        now_str,
        (date_x, date_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if confidence is not None:
        conf_text = f"pewnosc: {confidence*100:5.1f}%"
        cv2.putText(frame, conf_text, (20+2, 72+2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, conf_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        bar_x, bar_y = 220, 54
        bar_w, bar_h = 260, 18
        # Ramka: czarna i biała
        cv2.rectangle(frame, (bar_x-1, bar_y-1), (bar_x + bar_w+1, bar_y + bar_h+1), (0, 0, 0), 4)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
        fill_w = int(bar_w * max(0.0, min(1.0, confidence)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (80, 220, 120), -1)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:4.1f}", (w - 150+2, h - 20+2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:4.1f}", (w - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="np. models/baseline_advanced_logreg_model.pkl")
    ap.add_argument("--features_csv", default="features_advanced.csv", help="CSV użyty do treningu (żeby wziąć kolejność kolumn)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--smooth", type=float, default=0.6)
    args = ap.parse_args()

    model_path = Path(args.model)
    csv_path = Path(args.features_csv)

    if not model_path.exists():
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono {csv_path}. To potrzebne, żeby wziąć identyczną listę cech i kolejność.\n"
            f"Podaj prawidłową ścieżkę przez --features_csv."
        )

    # Wczytaj model (Pipeline)
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)

    # Wczytaj kolejność cech tak samo jak w treningu
    df = pd.read_csv(csv_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "hour"]
    if not feature_cols:
        raise RuntimeError("Nie znaleziono numerycznych cech w features_advanced.csv (poza 'hour').")

    # Spróbuj użyć repo-funkcji, jeśli istnieje
    repo_fn, repo_name = try_import_repo_feature_extractor()
    if repo_fn is not None:
        extractor = repo_fn
        print(f"[OK] Używam ekstraktora z repo: {repo_name}")
    else:
        extractor = fallback_extract_features_advanced
        print(
            "[UWAGA] Nie znalazłem funkcji ekstrakcji advanced w repo. Używam fallback.\n"
            "To prawie na pewno NIE będzie zgodne z treningiem i wyniki mogą być losowe.\n"
            "Najlepiej: dopasuj try_import_repo_feature_extractor() do nazw w Twoim precompute_features_advanced.py."
        )

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Nie mogę otworzyć kamerki {args.cam}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps = None
    last_t = cv2.getTickCount()

    smooth = max(0.0, min(0.99, float(args.smooth)))
    smoothed_conf = None

    window = "USB camera -> predicted hour (ADVANCED) (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # FPS update
        now_t = cv2.getTickCount()
        dt = (now_t - last_t) / cv2.getTickFrequency()
        last_t = now_t
        if dt > 0:
            inst = 1.0 / dt
            fps = inst if fps is None else (0.9 * fps + 0.1 * inst)

        # 1) Wyciągnij cechy jako dict
        feats_dict = extractor(frame)  # dict[str, float]

        # 2) Złóż wektor w dokładnej kolejności feature_cols
        #    Brakujące cechy -> 0.0 (lepsze niż crash, ale jeśli brakuje ich dużo, to model nie ma sensu)
        x = np.array([[float(feats_dict.get(c, 0.0)) for c in feature_cols]], dtype=np.float32)

        # 3) Predykcja
        pred = pipe.predict(x)[0]
        label = pretty_hour_label(pred)

        confidence_to_show = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(x)[0]
            conf = float(np.max(proba))
            smoothed_conf = conf if smoothed_conf is None else (smooth * smoothed_conf + (1 - smooth) * conf)
            confidence_to_show = smoothed_conf

        out = draw_overlay(frame, f"Godzina: {label}", confidence_to_show, fps)
        cv2.imshow(window, out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
