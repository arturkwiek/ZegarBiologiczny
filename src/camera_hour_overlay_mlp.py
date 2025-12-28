# src/camera_hour_overlay.py
# -----------------------------------------------------------------------------
# Cel: Live podgląd z kamery + overlay przewidywanej godziny.
#
# Tryb A (preferowany): MLP cykliczny (sin/cos) + cechy ROBUST (58)
#   - wymaga: models/best_mlp_cyclic.pt oraz możliwości wyliczenia cech robust
#   - próbuje zaimportować extractor cech z repo (patrz: _try_import_robust_extractor)
#
# Tryb B (fallback): RandomForest klasyfikacyjny na cechach ADVANCED (10)
#   - wymaga: models/baseline_advanced_rf_model.pkl
#   - cechy advanced liczymy w locie (mean/std RGB + HSV)
#
# Uruchom:
#   python -m src.camera_hour_overlay --cam 0
#   python -m src.camera_hour_overlay --cam 0 --width 1280 --height 720
#
# Sterowanie:
#   q / ESC  -> wyjście
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

# OpenCV do kamery i rysowania overlay
import cv2

# Torch tylko jeśli użyjemy MLP
try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None


MODELS_DIR = Path("models")
MLP_CKPT = MODELS_DIR / "best_mlp_cyclic.pt"
ADV_RF_PKL = MODELS_DIR / "baseline_advanced_rf_model.pkl"


# ----------------------------- Utils: time geometry -----------------------------

def sincos_to_hour(sin_cos: np.ndarray) -> float:
    """(sin, cos) -> hour in [0, 24)."""
    s = float(sin_cos[0])
    c = float(sin_cos[1])
    theta = math.atan2(s, c)  # [-pi, pi]
    if theta < 0:
        theta += 2.0 * math.pi
    return theta * 24.0 / (2.0 * math.pi)


def circular_err(pred_h: float, true_h: float) -> float:
    diff = abs(pred_h - true_h)
    return min(diff, 24.0 - diff)


# ----------------------------- Feature extraction ------------------------------

def extract_advanced_features_from_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Fallback: 10 cech jak w baseline_advanced:
    r_mean,g_mean,b_mean,r_std,g_std,b_std,h_mean,h_std,s_mean,v_mean
    """
    # BGR -> RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r = rgb[:, :, 0].reshape(-1)
    g = rgb[:, :, 1].reshape(-1)
    b = rgb[:, :, 2].reshape(-1)

    r_mean, g_mean, b_mean = float(r.mean()), float(g.mean()), float(b.mean())
    r_std, g_std, b_std = float(r.std()), float(g.std()), float(b.std())

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    # OpenCV H: [0..179], S,V: [0..255]
    h = (hsv[:, :, 0].reshape(-1) / 179.0)
    s = (hsv[:, :, 1].reshape(-1) / 255.0)
    v = (hsv[:, :, 2].reshape(-1) / 255.0)

    h_mean, h_std = float(h.mean()), float(h.std())
    s_mean, v_mean = float(s.mean()), float(v.mean())

    feats = np.array([r_mean, g_mean, b_mean, r_std, g_std, b_std, h_mean, h_std, s_mean, v_mean], dtype=np.float32)
    return feats


def _try_import_robust_extractor() -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """
    Szukamy w repo funkcji, która liczy cechy robust z pojedynczego obrazu.
    Dopasuj nazwy importu, jeśli masz je inaczej nazwane.

    Oczekujemy: extractor(frame_bgr) -> np.ndarray shape=(58,) float32
    """
    candidates = [
        ("src.precompute_features_robust", "extract_features_robust_from_bgr"),
        ("src.precompute_features_robust", "extract_robust_features_from_bgr"),
        ("src.utils", "extract_features_robust_from_bgr"),
        ("src.utils", "extract_robust_features_from_bgr"),
    ]

    for mod_name, fn_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name)
            return fn
        except Exception:
            continue
    return None


# ----------------------------- MLP model definition ----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_mlp_checkpoint(ckpt_path: Path) -> Tuple[MLP, np.ndarray, np.ndarray]:
    """
    Ładuje checkpoint MLP: model + mean/std do standaryzacji.
    Obsługuje kilka typowych formatów checkpointu.
    """
    if torch is None:
        raise RuntimeError("Brak torch. Zainstaluj PyTorch albo użyj fallbacku advanced RF.")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Format 1: dict ze stanem
    if isinstance(ckpt, dict):
        # próbujemy znaleźć klucze
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model")
        mean = ckpt.get("mean")
        std = ckpt.get("std")
        in_dim = ckpt.get("in_dim")

        if state is None:
            # czasem cały checkpoint to state_dict
            if all(isinstance(k, str) and k.startswith(("net.", "0.", "1.", "2.", "3.")) for k in ckpt.keys()):
                state = ckpt

        if mean is None or std is None:
            raise ValueError("Checkpoint nie zawiera mean/std do standaryzacji cech.")

        mean = np.array(mean, dtype=np.float32).reshape(1, -1)
        std = np.array(std, dtype=np.float32).reshape(1, -1)

        if in_dim is None:
            in_dim = int(mean.shape[1])

        model = MLP(in_dim=in_dim)
        model.load_state_dict(state)
        model.eval()
        return model, mean, std

    raise ValueError("Nieznany format checkpointu MLP.")


# ----------------------------- Overlay drawing --------------------------------

def draw_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    x, y = 20, 30
    for i, line in enumerate(lines):
        yy = y + i * 28
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ----------------------------- Main loop --------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="ID kamery (0 zazwyczaj domyślna).")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--every", type=int, default=2, help="Predykcja co N klatek (dla stabilności/CPU).")
    ap.add_argument("--use_fallback", action="store_true", help="Wymuś fallback advanced RF.")
    args = ap.parse_args()

    # --- Wybór trybu ---
    robust_extractor = None if args.use_fallback else _try_import_robust_extractor()
    use_mlp = (robust_extractor is not None) and MLP_CKPT.exists()

    mlp_model = None
    mean = std = None

    if use_mlp:
        mlp_model, mean, std = load_mlp_checkpoint(MLP_CKPT)
        mode_name = "MLP cyclic (robust)"
    else:
        # fallback: RF klasyfikacyjny (advanced)
        if not ADV_RF_PKL.exists():
            raise FileNotFoundError(
                f"Nie znalazłem {ADV_RF_PKL}. "
                "Uruchom: python -m src.baseline_advanced (żeby zapisać baseline_advanced_rf_model.pkl)"
            )
        with open(ADV_RF_PKL, "rb") as f:
            rf_clf = pickle.load(f)
        mode_name = "RF classify (advanced) [fallback]"

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Nie mogę otworzyć kamery: {args.cam}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    last_pred_hour = None
    last_conf = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        if frame_idx % max(1, args.every) == 0:
            if use_mlp:
                # 1) cechy robust
                feats = robust_extractor(frame).astype(np.float32).reshape(1, -1)
                # 2) standaryzacja jak w treningu
                feats = (feats - mean) / (std + 1e-6)

                with torch.no_grad():
                    x = torch.from_numpy(feats)
                    y_sc = mlp_model(x).numpy().reshape(-1)

                # normalizacja na okrąg (bezpiecznik)
                norm = float(np.linalg.norm(y_sc) + 1e-9)
                y_sc = y_sc / norm

                pred_hour = sincos_to_hour(y_sc)
                last_pred_hour = pred_hour
                last_conf = float(norm)  # pseudo-confidence: im bliżej okręgu „bez normalizacji”, tym lepiej
            else:
                feats = extract_advanced_features_from_bgr(frame).reshape(1, -1)
                proba = rf_clf.predict_proba(feats)[0]
                pred_hour = int(np.argmax(proba))
                last_pred_hour = float(pred_hour)
                last_conf = float(np.max(proba))

        # --- overlay ---
        lines = [f"Mode: {mode_name}"]
        if last_pred_hour is None:
            lines.append("Pred: ...")
        else:
            h = last_pred_hour
            hh = int(h) % 24
            mm = int(round((h - hh) * 60.0)) % 60
            lines.append(f"Pred hour: {hh:02d}:{mm:02d}")
            if last_conf is not None:
                lines.append(f"Conf: {last_conf:.3f}")
        lines.append("q/ESC -> quit")

        shown = draw_overlay(frame, lines)
        cv2.imshow("camera_hour_overlay", shown)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
