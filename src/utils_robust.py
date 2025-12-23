# src/utils_robust.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


def _safe_entropy_from_hist(hist: np.ndarray) -> float:
    """
    Oblicza entropię Shannona z histogramu (już znormalizowanego do sumy 1).
    Dodajemy epsilon, żeby nie trafić w log(0).
    """
    eps = 1e-12
    p = np.clip(hist, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def extract_robust_image_features(
    image_path: Path,
    *,
    bins_v: int = 16,
    bins_s: int = 16,
    top_frac: float = 0.25,
    bottom_frac: float = 0.25,
) -> Optional[Dict[str, float]]:
    """
    Ekstrakcja cech "odpornych na chmury" wyłącznie z obrazu.

    Zwraca słownik cech liczbowych albo None, jeśli obraz jest nie do odczytu.

    Cechy (intuicyjnie):
    - Globalne statystyki RGB i HSV (jak wcześniej, ale rozszerzone)
    - Percentyle V (jasność): V10, V50, V90 + rozstęp V90-V10
    - Histogramy V i S (zabezpiecza przed "średnia nic nie mówi")
    - Entropia histogramu V i S (miara "płaskości"/złożoności)
    - Gradient pionowy: jasność góra vs dół (często stabilniejsze niż absolute V)
    - Edge density (Canny): sygnał warunków, nie tylko sceny
    """
    # --- KROK 1: Wczytanie obrazu ---
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    # --- KROK 2: Konwersje kolorów ---
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # --- KROK 3: Spłaszczenie do wektorów dla statystyk globalnych ---
    rgb = img_rgb.reshape(-1, 3).astype(np.float32)
    hsv = img_hsv.reshape(-1, 3).astype(np.float32)

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    # W HSV (OpenCV): H w [0..179], S w [0..255], V w [0..255]
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]

    feats: Dict[str, float] = {}

    # --- KROK 4: Proste statystyki (zgodne z Twoim dotychczasowym podejściem) ---
    feats["r_mean"] = float(r.mean())
    feats["g_mean"] = float(g.mean())
    feats["b_mean"] = float(b.mean())
    feats["r_std"] = float(r.std())
    feats["g_std"] = float(g.std())
    feats["b_std"] = float(b.std())
    feats["s_mean"] = float(s.mean())
    feats["v_mean"] = float(v.mean())
    feats["s_std"] = float(s.std())
    feats["v_std"] = float(v.std())

    # --- KROK 5: Percentyle i rozstęp kontrastu (bardzo ważne przy chmurach) ---
    v10, v50, v90 = np.percentile(v, [10, 50, 90])
    feats["v_p10"] = float(v10)
    feats["v_p50"] = float(v50)
    feats["v_p90"] = float(v90)
    feats["v_iqr_90_10"] = float(v90 - v10)  # "kontrast rozstępu"

    s10, s50, s90 = np.percentile(s, [10, 50, 90])
    feats["s_p10"] = float(s10)
    feats["s_p50"] = float(s50)
    feats["s_p90"] = float(s90)
    feats["s_iqr_90_10"] = float(s90 - s10)

    # --- KROK 6: Histogramy V i S (kształt rozkładu zamiast średniej) ---
    # Normalizujemy do sumy 1 -> można liczyć entropię
    v_hist, _ = np.histogram(v, bins=bins_v, range=(0, 255), density=False)
    v_hist = v_hist.astype(np.float64)
    v_hist = v_hist / max(v_hist.sum(), 1.0)

    s_hist, _ = np.histogram(s, bins=bins_s, range=(0, 255), density=False)
    s_hist = s_hist.astype(np.float64)
    s_hist = s_hist / max(s_hist.sum(), 1.0)

    # Dodajemy histogram jako osobne cechy v_hist_00 ... v_hist_15
    for i, p in enumerate(v_hist):
        feats[f"v_hist_{i:02d}"] = float(p)
    for i, p in enumerate(s_hist):
        feats[f"s_hist_{i:02d}"] = float(p)

    # --- KROK 7: Entropia histogramów (miara "płaskości" sceny) ---
    feats["v_entropy"] = _safe_entropy_from_hist(v_hist)
    feats["s_entropy"] = _safe_entropy_from_hist(s_hist)

    # --- KROK 8: Gradient pionowy jasności (góra vs dół) ---
    # Idea: wschód/zachód często zmienia gradient; chmury spłaszczają, ale nadal informują.
    h_img, w_img = img_hsv.shape[:2]
    top_h = max(1, int(h_img * top_frac))
    bot_h = max(1, int(h_img * bottom_frac))

    v_img = img_hsv[:, :, 2].astype(np.float32)
    v_top = v_img[:top_h, :].mean()
    v_bottom = v_img[h_img - bot_h :, :].mean()

    feats["v_top_mean"] = float(v_top)
    feats["v_bottom_mean"] = float(v_bottom)
    feats["v_top_minus_bottom"] = float(v_top - v_bottom)

    # --- KROK 9: Edge density (Canny) ---
    # Uwaga: to jest bardziej "warunki + ostrość" niż czas, ale model lubi mieć ten sygnał.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    feats["edge_density"] = float((edges > 0).mean())

    return feats
