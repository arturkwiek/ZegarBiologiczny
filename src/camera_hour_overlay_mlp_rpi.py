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
import datetime
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .camera_hour_overlay_mlp import (
    MODELS_DIR,
    MLP_CKPT,
    ADV_RF_PKL,
    MLP,
    _try_import_robust_extractor,
    extract_advanced_features_from_bgr,
    load_mlp_checkpoint,
    sincos_to_hour,
)

import pickle


def draw_overlay(
    frame: np.ndarray,
    text: str,
    mode_name: str,
    confidence: float | None,
    fps: float | None,
) -> np.ndarray:
    """Rysuje panel informacyjny u góry obrazu (styl jak w camera_hour_overlay_rpi)."""
    h, w = frame.shape[:2]
    panel_h = 100

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    alpha = 0.55
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Główna linia z godziną
    cv2.putText(
        frame,
        text,
        (20 + 2, 38 + 2),
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

    # Tryb/model
    mode_text = f"Model: {mode_name}"
    cv2.putText(
        frame,
        mode_text,
        (20 + 2, 72 + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        mode_text,
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Data/czas systemowy w prawym górnym rogu panelu
    date_x = w - 350
    date_y = 38
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

    # Pasek pewności (0..1)
    if confidence is not None:
        conf_clamped = max(0.0, min(1.0, float(confidence)))
        conf_text = f"pewnosc: {conf_clamped * 100:5.1f}%"
        cv2.putText(
            frame,
            conf_text,
            (20 + 2, 104 + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            conf_text,
            (20, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        bar_x, bar_y = 260, 86
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
    args = ap.parse_args()

    # --- Wybór trybu: MLP (robust) vs fallback RF (advanced) ---
    robust_extractor: Optional[callable] = None if args.use_fallback else _try_import_robust_extractor()
    use_mlp = (robust_extractor is not None) and MLP_CKPT.exists()

    mlp_model: Optional[MLP]
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]
    rf_clf = None

    if use_mlp:
        mlp_model, mean, std = load_mlp_checkpoint(MLP_CKPT)
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
        if use_mlp:
            # cechy robust
            feats = robust_extractor(frame).astype(np.float32).reshape(1, -1)
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

            # „pewność” mapujemy do 0..1 i wygładzamy
            raw_conf = max(0.0, min(1.0, norm))
            if smoothed_conf is None:
                smoothed_conf = raw_conf
            else:
                smoothed_conf = smooth * smoothed_conf + (1.0 - smooth) * raw_conf
            conf_to_show = smoothed_conf
        else:
            # fallback: RF na cechach advanced
            feats_adv = extract_advanced_features_from_bgr(frame).reshape(1, -1)
            proba = rf_clf.predict_proba(feats_adv)[0]
            pred_class = int(np.argmax(proba))
            hour_float = float(pred_class)

            raw_conf = float(np.max(proba))
            if smoothed_conf is None:
                smoothed_conf = raw_conf
            else:
                smoothed_conf = smooth * smoothed_conf + (1.0 - smooth) * raw_conf
            conf_to_show = smoothed_conf

        hh = int(hour_float) % 24
        mm = int(round((hour_float - hh) * 60.0)) % 60
        text = f"Godzina (pred): {hh:02d}:{mm:02d}"

        frame_out = draw_overlay(frame, text, mode_name, conf_to_show, fps)
        cv2.imwrite(output_path, frame_out)
        print(f"[INFO] Zapisano obraz do {output_path} z predykcją {hh:02d}:{mm:02d} ({mode_name})")

        time.sleep(args.interval)

    cap.release()


if __name__ == "__main__":
    main()
