# camera_hour_overlay_mlp_rpi.py
# Run: python -m src.camera_hour_overlay_mlp_rpi [--cam 0 --width 1280 --height 720 \
#        --interval 2.0 --output ~/www/camera_hour.jpg --use_fallback]
# -----------------------------------------------------------------------------
# Cel:
#     Wariant skryptu overlay dostosowany do Raspberry Pi.
#     Okresowo pobiera klatkę z kamerki USB, wyznacza godzinę przy użyciu
#     modelu MLP (regresja cykliczna na cechach robust) lub – w trybie
#     awaryjnym – modelu RandomForest na cechach advanced, a następnie
#     zapisuje obraz z nałożoną godziną do pliku JPG.
#
#     - Tryb podstawowy (domyślny): MLP cyclic + cechy ROBUST
#       wymaga pliku models/best_mlp_cyclic.pt oraz funkcji ekstrakcji
#       cech robust z pojedynczej klatki (por. _try_import_robust_extractor).
#     - Tryb awaryjny (--use_fallback): RandomForest na cechach ADVANCED
#       wymaga pliku models/baseline_advanced_rf_model.pkl.
#
#     Skrypt nie otwiera okna z podglądem – zapisuje kolejne obrazy do pliku,
#     tak jak camera_hour_overlay_rpi.py.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import math
import datetime
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .camera_hour_overlay_mlp import (
    MLP,
    _try_import_robust_extractor,
    extract_advanced_features_from_bgr,
    load_mlp_checkpoint,
    sincos_to_hour,
)
from .train_hour_cnn import SmallHourCNN

MODELS_DIR = Path("models") / "rpi"
MLP_CKPT = MODELS_DIR / "best_mlp_cyclic.pt"
ADV_RF_PKL = MODELS_DIR / "baseline_advanced_rf_model.pkl"
CNN_CKPT = MODELS_DIR / "best_cnn_hour.pt"

# Stałe filtra czasu na kole 24h
MAX_RATE_H_PER_SEC = 0.000333  # maks. szybkość zmiany estymaty [h/s], np. ~0.02 h/min
MAX_DT_SEC = 10.0  # zabezpieczenie na długie przerwy między pomiarami
ALPHA_MIN = 0.01
ALPHA_MAX = 0.12
CONF_FLOOR = 0.5
ANOMALY_H = 2.0
ANOMALY_COUNT_THRESHOLD = 3


def wrap24(h: float) -> float:
    """Zawijanie godziny na przedział [0, 24) (czas cykliczny)."""
    return float(h % 24.0)


def cyclic_delta(pred: float, state: float) -> float:
    """Najmniejsza różnica pred-state na okręgu 24h w zakresie [-12, 12]."""
    diff = (pred - state + 12.0) % 24.0 - 12.0
    return float(diff)


def format_hour_hhmm(h: float) -> str:
    """Formatuje godzinę float jako HH:MM na kole 24h."""
    h_wrapped = wrap24(h)
    hh = int(h_wrapped) % 24
    mm = int(round((h_wrapped - hh) * 60.0)) % 60
    return f"{hh:02d}:{mm:02d}"

import pickle


def draw_overlay(
    frame: np.ndarray,
    text_raw: str,
    text_filt: str,
    mode_name: str,
    confidence: float | None,
    fps: float | None,
) -> np.ndarray:
    """Rysuje panel informacyjny u góry obrazu (styl jak w camera_hour_overlay_rpi)."""
    h, w = frame.shape[:2]
    panel_h = 160

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    alpha = 0.55
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Wspólne wiersze tekstu w panelu
    y_filt = 36
    y_raw = 64
    y_date = 92
    y_mode = 120
    y_conf = 148

    # Filtrowana godzina (wiersz 1)
    cv2.putText(
        frame,
        text_filt,
        (20 + 2, y_filt + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text_filt,
        (20, y_filt),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Surowa godzina (wiersz 2)
    cv2.putText(
        frame,
        text_raw,
        (20 + 2, y_raw + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text_raw,
        (20, y_raw),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Data/czas (wiersz 3, wyrównany do prawej)
    (text_w, _), _ = cv2.getTextSize(now_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    date_x = max(20, w - text_w - 20)
    date_y = y_date
    cv2.putText(
        frame,
        now_str,
        (date_x + 2, date_y + 2),
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

    # Tryb/model (wiersz 4)
    mode_text = f"Model: {mode_name}"
    cv2.putText(
        frame,
        mode_text,
        (20 + 2, y_mode + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        mode_text,
        (20, y_mode),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Pasek pewności (0..1) (wiersz 5)
    if confidence is not None:
        conf_clamped = max(0.0, min(1.0, float(confidence)))
        conf_text = f"pewnosc: {conf_clamped * 100:5.1f}%"
        cv2.putText(
            frame,
            conf_text,
            (20 + 2, y_conf + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            conf_text,
            (20, y_conf),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # Pasek pod tekstem pewności
        bar_x, bar_y = 260, y_conf - 18
        bar_w, bar_h = 260, 18
        cv2.rectangle(frame, (bar_x - 1, bar_y - 1), (bar_x + bar_w + 1, bar_y + bar_h + 1), (0, 0, 0), 4)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
        fill_w = int(bar_w * conf_clamped)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (80, 220, 120), -1)

    # FPS w prawym dolnym rogu
    if fps is not None:
        cv2.putText(
            frame,
            f"FPS: {fps:4.1f}",
            (w - 150 + 2, h - 20 + 2),
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Indeks kamerki (zwykle 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30, help="Docelowa liczba FPS kamery")
    ap.add_argument("--interval", type=float, default=2.0, help="Interwał (s) między kolejnymi zapisami")
    ap.add_argument(
        "--output",
        type=str,
        default="~/www/camera_hour.jpg",
        help="Ścieżka do pliku wyjściowego JPG",
    )
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
    ap.add_argument(
        "--smooth",
        type=float,
        default=0.6,
        help="Wygładzanie pewności 0..1 (większe = mocniej wygładza)",
    )
    ap.add_argument(
        "--use_fallback",
        action="store_true",
        help="Wymuś tryb awaryjny: RandomForest na cechach advanced",
    )
    ap.add_argument(
        "--use_cnn",
        action="store_true",
        help="Użyj modelu CNN na całym obrazie (eksperymentalne)",
    )
    args = ap.parse_args()

    # --- Wybór trybu: CNN (pełny obraz) vs MLP (robust) vs fallback RF (advanced) ---
    use_cnn = bool(args.use_cnn)
    robust_extractor: Optional[callable] = None if (args.use_fallback or use_cnn) else _try_import_robust_extractor()
    use_mlp = (not use_cnn) and (robust_extractor is not None) and MLP_CKPT.exists()

    mlp_model: Optional[MLP]
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]
    rf_clf = None
    cnn_model = None
    expected_feats: Optional[int] = None
    cnn_img_size: int = 224

    if use_cnn:
        import torch

        if not CNN_CKPT.exists():
            raise FileNotFoundError(
                f"Nie znaleziono pliku {CNN_CKPT}. "
                "Uruchom: python -m src.train_hour_cnn (żeby zapisać best_cnn_hour.pt)."
            )

        ckpt = torch.load(CNN_CKPT, map_location="cpu")

        # Obsługa zarówno zapisu pełnego modelu (torch.save(model))
        # jak i zapisu słownika z 'model_state' (obecny train_hour_cnn).
        if hasattr(ckpt, "eval") and hasattr(ckpt, "state_dict") and not isinstance(ckpt, dict):
            # Zapisano cały obiekt modelu
            cnn_model = ckpt
            # Spróbuj wydedukować liczbę klas z ostatniej warstwy liniowej
            last_linear = getattr(getattr(cnn_model, "classifier", None), "[-1]", None)
            num_classes = getattr(last_linear, "out_features", 24) if last_linear is not None else 24
            cnn_img_size = 224  # brak metadanych w tym trybie
        else:
            # Zapisano słownik (np. {"model_state": ..., "num_classes": ...})
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model_state", ckpt)
                num_classes = int(ckpt.get("num_classes", 24))
                cnn_img_size = int(ckpt.get("img_size", 224))
            else:
                # Ostatnia linia obrony: potraktuj jako state_dict bez metadanych
                state_dict = ckpt
                num_classes = 24
                cnn_img_size = 224

            cnn_model = SmallHourCNN(num_classes=num_classes)
            cnn_model.load_state_dict(state_dict)

        cnn_model.eval()

        mean = std = None
        mode_name = "CNN (full frame)"
        print(f"[INFO] Załadowano CNN z {CNN_CKPT}")
    elif use_mlp:
        mlp_model, mean, std = load_mlp_checkpoint(MLP_CKPT)
        mlp_model.eval()
        mode_name = "MLP cyclic (robust)"
        print(f"[INFO] Załadowano MLP z {MLP_CKPT}")
    else:
        if not ADV_RF_PKL.exists():
            raise FileNotFoundError(
                f"Nie znaleziono pliku {ADV_RF_PKL}. "
                "Uruchom: python -m src.baseline_advanced (żeby zapisać baseline_advanced_rf_model.pkl)."
            )
        with open(ADV_RF_PKL, "rb") as f:
            rf_clf = pickle.load(f)
        mean = std = None
        mode_name = "RF classify (advanced) [fallback]"
        print(f"[INFO] Załadowano fallback RF z {ADV_RF_PKL}")
        expected_feats = getattr(rf_clf, "n_features_in_", None)
        if expected_feats is not None:
            print(f"[INFO] RF oczekuje {expected_feats} cech wejściowych")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Nie mogę otworzyć kamerki o indeksie {args.cam}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    last_t = cv2.getTickCount()
    fps: Optional[float] = None

    smooth = float(args.smooth)
    smooth = max(0.0, min(0.99, smooth))
    smoothed_conf: Optional[float] = None

    output_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    raw_log = (getattr(args, "log_csv", "") or "").strip()
    if not raw_log:
        log_path = f"{Path(__file__).stem}_log.txt"
    else:
        log_path = os.path.expanduser(raw_log)

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Stan filtra czasu
    state_hour: Optional[float] = None
    anomaly_count: int = 0
    last_state_ts: Optional[float] = None
    raw_delta: float = 0.0
    dt_sec: float = 0.0

    # Stan nagłówka logu CSV
    log_header_written = False
    if os.path.exists(log_path):
        try:
            if os.path.getsize(log_path) > 0:
                log_header_written = True
        except OSError:
            pass

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Błąd odczytu z kamerki – ponowna próba po przerwie.")
            time.sleep(args.interval)
            continue

        # FPS
        now_t = cv2.getTickCount()
        dt = (now_t - last_t) / cv2.getTickFrequency()
        last_t = now_t
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = inst_fps if fps is None else (0.9 * fps + 0.1 * inst_fps)

        # --- Predykcja godziny ---
        if use_cnn:
            # pełny obraz przez CNN (klasy 0..23)
            import torch
            device = next(cnn_model.parameters()).device if hasattr(cnn_model, "parameters") else torch.device("cpu")

            frame_small = cv2.resize(
                frame, (cnn_img_size, cnn_img_size), interpolation=cv2.INTER_AREA
            )
            img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_chw = np.transpose(img_rgb, (2, 0, 1))
            x = torch.from_numpy(img_chw).unsqueeze(0).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                logits = cnn_model(x)
                prob = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred_class = int(np.argmax(prob))
            hour_float = float(pred_class)

            raw_conf = float(np.max(prob))
            if smoothed_conf is None:
                smoothed_conf = raw_conf
            else:
                smoothed_conf = smooth * smoothed_conf + (1.0 - smooth) * raw_conf
            conf_to_show = smoothed_conf

        elif use_mlp:
            # cechy robust (na downskalowanej klatce dla oszczędności CPU)
            frame_small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            feats = robust_extractor(frame_small).astype(np.float32).reshape(1, -1)
            # standaryzacja jak w treningu
            feats = (feats - mean) / (std + 1e-6)

            import torch  # lokalny import, żeby nie psuć startu jeśli torch nie jest dostępny globalnie

            with torch.no_grad():
                x = torch.from_numpy(feats)
                y_sc = mlp_model(x).numpy().reshape(-1)

            # norm przed normalizacją traktujemy jako surogat pewności
            norm = float(np.linalg.norm(y_sc) + 1e-9)
            y_sc = y_sc / norm

            pred_hour = sincos_to_hour(y_sc)
            hour_float = float(pred_hour)

            # kalibracja pewności (sigmoida)
            t = 0.9
            k = 6.0
            raw_conf = 1.0 / (1.0 + np.exp(-k * (norm - t)))
            if smoothed_conf is None:
                smoothed_conf = raw_conf
            else:
                smoothed_conf = smooth * smoothed_conf + (1.0 - smooth) * raw_conf
            conf_to_show = smoothed_conf
        else:
            # fallback: RF na cechach advanced
            feats_adv = extract_advanced_features_from_bgr(frame).reshape(1, -1)
            if expected_feats is not None and feats_adv.shape[1] != expected_feats:
                raise RuntimeError(
                    f"RF: niezgodna liczba cech ({feats_adv.shape[1]} != {expected_feats})"
                )
            proba = rf_clf.predict_proba(feats_adv)[0]
            pred_class = int(np.argmax(proba))
            hour_float = float(pred_class)

            raw_conf = float(np.max(prob))
            if smoothed_conf is None:
                smoothed_conf = raw_conf
            else:
                smoothed_conf = smooth * smoothed_conf + (1.0 - smooth) * raw_conf
            conf_to_show = smoothed_conf

        # --- Filtracja predykcji na kole 24h ---
        now_ts = time.perf_counter()
        if last_state_ts is None:
            dt_sec = 0.0
        else:
            dt_sec = max(0.0, now_ts - last_state_ts)
        last_state_ts = now_ts

        if state_hour is None:
            state_hour = wrap24(hour_float)
            raw_delta = 0.0
            anomaly_count = 0
        else:
            raw_delta = cyclic_delta(hour_float, state_hour)
            is_anomaly = abs(raw_delta) > ANOMALY_H
            if is_anomaly:
                anomaly_count += 1
            else:
                anomaly_count = 0

            allow_update = (not is_anomaly) or (anomaly_count >= ANOMALY_COUNT_THRESHOLD)
            if allow_update:
                dt_eff = min(dt_sec, MAX_DT_SEC)
                max_jump_h = MAX_RATE_H_PER_SEC * dt_eff

                conf_for_alpha = max(0.0, min(1.0, float(raw_conf)))
                alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * conf_for_alpha
                if conf_for_alpha < CONF_FLOOR:
                    alpha = ALPHA_MIN

                # Ograniczenie skoku na kole 24h
                delta = max(-max_jump_h, min(max_jump_h, raw_delta))
                state_hour = wrap24(state_hour + alpha * delta)

                if is_anomaly:
                    anomaly_count = 0

        if state_hour is None:
            # Bezpiecznik: nie powinno się zdarzyć, ale dla pewności.
            state_hour = wrap24(hour_float)
            raw_delta = 0.0

        text_raw = f"Godzina (raw): {format_hour_hhmm(hour_float)}"
        text_filt = f"Godzina (filt): {format_hour_hhmm(state_hour)}"

        frame_out = draw_overlay(frame, text_raw, text_filt, mode_name, conf_to_show, fps)
        cv2.imwrite(output_path, frame_out)
        print(f"[INFO] Zapisano obraz do {output_path} z predykcjami RAW/FILT: {text_raw} / {text_filt} ({mode_name})")

        if log_path is not None:
            now_dt = datetime.datetime.now()
            ts_str = now_dt.isoformat()
            conf_val = "" if raw_conf is None else f"{float(raw_conf):.4f}"
            dt_val = f"{dt_sec:.3f}"
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    if not log_header_written:
                        f.write(
                            "timestamp_iso;pred_float;state_hour;raw_delta;confidence;dt_sec;model_name\n"
                        )
                        log_header_written = True
                    f.write(
                        f"{ts_str};{hour_float:.4f};{state_hour:.4f};"
                        f"{raw_delta:.4f};{conf_val};{dt_val};{mode_name}\n"
                    )
            except Exception as e:
                print(f"[WARN] Nie udało się dopisać do loga {log_path}: {e}")

        time.sleep(args.interval)

    cap.release()


if __name__ == "__main__":
    main()
