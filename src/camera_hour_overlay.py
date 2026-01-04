# camera_hour_overlay.py
# Run: python -m src.camera_hour_overlay --model models/baseline_rgb_model.pkl [--cam 0 --width 1280 --height 720 --smooth 0.6]

# Opis:
#     Skrypt do podglądu na żywo z kamerki USB z nałożoną przewidywaną godziną.
#     - Wczytuje wytrenowany model klasyfikacji godzin (np. baseline_rgb_model.pkl)
#     - Otwiera strumień z kamerki (OpenCV)
#     - Dla każdej klatki liczy średnie wartości RGB (mean_rgb_features)
#     - Wykonuje predykcję godziny i (opcjonalnie) prawdopodobieństw klas
#     - Rysuje overlay z godziną, pewnością i FPS, wyświetlany w oknie OpenCV

# Zadania realizowane przez skrypt:
#     1. Wczytanie i sprawdzenie modelu (load_model, get_expected_n_features)
#     2. Konfiguracja kamerki (indeks, rozdzielczość)
#     3. Pętla pobierająca klatki i licząca cechy mean RGB
#     4. Predykcja godziny i wygładzanie pewności (jeśli dostępne predict_proba)
#     5. Rysowanie overlay (draw_overlay) i obsługa klawisza wyjścia (q / ESC)

import argparse
import os
import pickle
from pathlib import Path
import cv2
import numpy as np
import datetime


def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def get_expected_n_features(model) -> int | None:
    """
    Próbuje ustalić liczbę cech wejściowych.
    Działa dla LogisticRegression i Pipeline(StandardScaler->LogReg).
    """
    # Pipeline?
    if hasattr(model, "named_steps") and "logreg" in model.named_steps:
        lr = model.named_steps["logreg"]
        if hasattr(lr, "n_features_in_"):
            return int(lr.n_features_in_)
        return None

    # Goły estimator
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    return None


def mean_rgb_features(frame_bgr: np.ndarray) -> np.ndarray:
    """
    OpenCV daje obraz w BGR. Przeliczamy na RGB i liczymy średnie.
    Zwraca shape (1,3): [r_mean, g_mean, b_mean]
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    means = frame_rgb.mean(axis=(0, 1)).astype(np.float32)  # [R,G,B]
    return means.reshape(1, 3)


def pretty_hour_label(hour: int) -> str:
    # Jeśli hour to 0..23, pokażemy 00:xx. Jeśli to inne klasy, pokażemy po prostu liczbę.
    if 0 <= int(hour) <= 23:
        return f"{int(hour):02d}:00"
    return str(int(hour))


def draw_overlay(
    frame: np.ndarray,
    text: str,
    confidence: float | None,
    fps: float | None,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Panel u góry
    panel_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    alpha = 0.55
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Data i czas systemowy
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Tekst główny: cień czarny + biały
    cv2.putText(
        frame,
        text,
        (20+2, 38+2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

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

    # Confidence + pasek
    if confidence is not None:
        conf_text = f"pewnosc: {confidence*100:5.1f}%"
        cv2.putText(
            frame,
            conf_text,
            (20+2, 72+2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            conf_text,
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        bar_x, bar_y = 220, 54
        bar_w, bar_h = 260, 18
        # Ramka: czarna i biała
        cv2.rectangle(frame, (bar_x-1, bar_y-1), (bar_x + bar_w+1, bar_y + bar_h+1), (0, 0, 0), 4)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)

        fill_w = int(bar_w * max(0.0, min(1.0, confidence)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (80, 220, 120), -1)

    # FPS w rogu
    if fps is not None:
        cv2.putText(
            frame,
            f"FPS: {fps:4.1f}",
            (w - 150+2, h - 20+2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS: {fps:4.1f}",
            (w - 150, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Sciezka do .pkl (np. models/baseline_rgb_model.pkl)")
    ap.add_argument("--cam", type=int, default=0, help="Indeks kamerki (zwykle 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--smooth", type=float, default=0.6, help="Wygładzanie pewności 0..1 (większe = mocniej wygładza)")
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
    model_name = type(model).__name__

    # Wspieramy "mean RGB" (3 cechy). Jeśli model oczekuje czegoś innego, przerwiemy z jasnym komunikatem.
    if n_features is not None and n_features != 3:
        raise SystemExit(
            f"Ten model oczekuje {n_features} cech, a ten podgląd z kamerki liczy tylko 3 (mean RGB).\n"
            f"Użyj modelu baseline_rgb_model.pkl albo dopisz ekstrakcję advanced cech na żywo."
        )

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Nie mogę otworzyć kamerki o indeksie {args.cam}.")

    # Ustawienia rozdzielczości (nie każda kamerka to respektuje, ale próbujemy)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    last_t = cv2.getTickCount()
    fps = None

    # Wygładzanie pewności (żeby overlay nie skakał)
    smooth = float(args.smooth)
    smooth = max(0.0, min(0.99, smooth))
    smoothed_conf = None

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

    window = "USB camera -> predicted hour (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # FPS
        now_t = cv2.getTickCount()
        dt = (now_t - last_t) / cv2.getTickFrequency()
        last_t = now_t
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = inst_fps if fps is None else (0.9 * fps + 0.1 * inst_fps)

        # Cechy mean RGB
        x = mean_rgb_features(frame)

        # Predykcja
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

        text = f"Godzina: {label}"
        frame_out = draw_overlay(frame, text, confidence_to_show, fps)

        cv2.imshow(window, frame_out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q albo ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
