# camera_hour_overlay_rpi.py
# Run: python -m src.camera_hour_overlay_rpi --model models/baseline_rgb_model.pkl [--cam 0 --width 1280 --height 720 --smooth 0.6 --interval 2.0 --output ~/www/camera_hour.jpg]

# Opis:
#     Skrypt pomocniczy do Raspberry Pi, który okresowo zapisuje klatkę z kamerki USB
#     z nałożoną przewidywaną godziną (overlay) do pliku JPG.
#     - Wczytuje wytrenowany model klasyfikacji godzin (np. baseline_rgb_model.pkl)
#     - Co zadany interwał (interval w sekundach) pobiera klatkę z kamerki
#     - Liczy cechy mean RGB i wykonuje predykcję godziny
#     - Rysuje overlay z godziną, pewnością i FPS na obrazie
#     - Zapisuje wynikowy obraz do pliku (np. ~/www/camera_hour.jpg) do użycia przez serwer WWW

# Zadania realizowane przez skrypt:
#     1. Wczytanie modelu i sprawdzenie liczby oczekiwanych cech (load_model, get_expected_n_features)
#     2. Konfiguracja źródła wideo i parametrów kamery (rozmiar, indeks kamery)
#     3. Okresowe pobieranie klatek i wyliczanie cech mean RGB (mean_rgb_features)
#     4. Predykcja godziny oraz (jeśli dostępne) prawdopodobieństwa klas
#     5. Rysowanie overlay z godziną, pewnością i FPS oraz zapis obrazu do pliku wyjściowego

import argparse
import pickle
from pathlib import Path
import time
import os
import cv2
import numpy as np
import datetime

def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def get_expected_n_features(model) -> int | None:
    if hasattr(model, "named_steps") and "logreg" in model.named_steps:
        lr = model.named_steps["logreg"]
        if hasattr(lr, "n_features_in_"):
            return int(lr.n_features_in_)
        return None
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    return None

def mean_rgb_features(frame_bgr: np.ndarray) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    means = frame_rgb.mean(axis=(0, 1)).astype(np.float32)
    return means.reshape(1, 3)

def pretty_hour_label(hour: int) -> str:
    if 0 <= int(hour) <= 23:
        return f"{int(hour):02d}:00"
    return str(int(hour))

def draw_overlay(frame: np.ndarray, text: str, confidence: float | None, fps: float | None) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    alpha = 0.55
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Data i czas systemowy
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Najpierw czarny cień, potem biały tekst
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
    ap.add_argument("--model", required=True, help="Sciezka do .pkl (np. models/baseline_rgb_model.pkl)")
    ap.add_argument("--cam", type=int, default=0, help="Indeks kamerki (zwykle 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--smooth", type=float, default=0.6, help="Wygładzanie pewności 0..1 (większe = mocniej wygładza)")
    ap.add_argument("--interval", type=float, default=2.0, help="Czas w sekundach między zdjęciami")
    ap.add_argument("--output", type=str, default="~/www/camera_hour.jpg", help="Ścieżka do pliku wyjściowego JPG")
    ap.add_argument(
        "--log-csv",
        dest="log_csv",
        type=str,
        default="",
        help="Ścieżka do pliku CSV z logiem predykcji (pusty lub brak = <nazwa_skryptu>_log.txt)",
    )
    # alias dla wstecznej kompatybilności
    ap.add_argument(
        "--log_csv",
        dest="log_csv",
        type=str,
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

    model = load_model(model_path)
    n_features = get_expected_n_features(model)
    if n_features is not None and n_features != 3:
        raise SystemExit(
            f"Ten model oczekuje {n_features} cech, a ten podgląd z kamerki liczy tylko 3 (mean RGB).\n"
            f"Użyj modelu baseline_rgb_model.pkl albo dopisz ekstrakcję advanced cech na żywo."
        )

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Nie mogę otworzyć kamerki o indeksie {args.cam}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    last_t = cv2.getTickCount()
    fps = None
    smooth = float(args.smooth)
    smooth = max(0.0, min(0.99, smooth))
    smoothed_conf = None

    output_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # konfiguracja logowania do CSV
    raw_log = (getattr(args, "log_csv", "") or "").strip()
    if not raw_log:
        log_path = f"{Path(__file__).stem}_log.txt"
    else:
        log_path = os.path.expanduser(raw_log)

    log_header_written = False
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    try:
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            log_header_written = True
    except OSError:
        pass

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Błąd odczytu z kamerki!")
            time.sleep(args.interval)
            continue
        now_t = cv2.getTickCount()
        dt = (now_t - last_t) / cv2.getTickFrequency()
        last_t = now_t
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = inst_fps if fps is None else (0.9 * fps + 0.1 * inst_fps)
        x = mean_rgb_features(frame)
        pred = model.predict(x)[0]
        label = pretty_hour_label(pred)
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)[0]
            confidence = float(np.max(proba))
            if smoothed_conf is None:
                smoothed_conf = confidence
            else:
                smoothed_conf = smooth * smoothed_conf + (1 - smooth) * confidence
            confidence_to_show = smoothed_conf
        else:
            confidence_to_show = None
        text = f"Godzina: {label}"
        frame_out = draw_overlay(frame, text, confidence_to_show, fps)
        cv2.imwrite(output_path, frame_out)
        print(f"Zapisano obraz do {output_path} z predykcją {label}")

        # Logowanie do CSV (timestamp, rzeczywista godzina, predykcja, pewność, model)
        if log_path is not None:
            now_dt = datetime.datetime.now()
            ts_str = now_dt.isoformat()
            true_hour = (
                now_dt.hour
                + now_dt.minute / 60.0
                + now_dt.second / 3600.0
                + now_dt.microsecond / 3_600_000_000.0
            )
            try:
                pred_hour = float(pred)
            except Exception:
                pred_hour = float(int(pred))
            conf_val = "" if confidence is None else f"{float(confidence):.4f}"
            model_name = type(model).__name__
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    if not log_header_written:
                        f.write(
                            "timestamp_iso;true_hour;pred_hour;pred_label;confidence;model_name\n"
                        )
                        log_header_written = True
                    f.write(
                        f"{ts_str};{true_hour:.4f};{pred_hour:.4f};"
                        f"{label};{conf_val};{model_name}\n"
                    )
            except Exception as e:
                print(f"[WARN] Nie udało się dopisać do loga {log_path}: {e}")
        time.sleep(args.interval)

    cap.release()

if __name__ == "__main__":
    main()
