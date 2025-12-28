# utils.py — Run: used as a library module (import src.utils), not meant to be executed directly.

# Opis:
#     Zbiór funkcji pomocniczych do ekstrakcji cech z obrazów (średnie RGB, statystyki HSV).
#     - Wczytywanie obrazów i obliczanie cech
#     - Konwersje kolorów (BGR -> RGB)
#     - Funkcje wykorzystywane w precompute_mean_rgb.py, precompute_features_advanced.py

# Zadania realizowane przez skrypt:
#     1. Ekstrakcja średnich wartości RGB z obrazu
#     2. Przetwarzanie obrazów na potrzeby ekstrakcji cech
#     3. Pomocnicze funkcje do obsługi danych obrazowych

from pathlib import Path
import cv2
import numpy as np

def extract_mean_rgb(path: Path):
    """
    Wczytuje obraz z podanej ścieżki i zwraca średnie wartości kanałów RGB.
    Zwraca np.ndarray o kształcie (3,) lub None jeśli obrazka nie da się wczytać.
    """
    img = cv2.imread(str(path))
    if img is None:
        return None

    # OpenCV czyta w BGR, konwersja na RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # policz średnie po wszystkich pikselach
    mean_rgb = img.reshape(-1, 3).mean(axis=0)
    return mean_rgb.astype(np.float32)

def extract_rgb_hsv_stats(path):
    """
    Zwraca rozszerzone cechy z obrazu:
    [r_mean, g_mean, b_mean,
     r_std, g_std, b_std,
     h_mean, h_std,
     s_mean, v_mean]
    """
    img = cv2.imread(str(path))
    if img is None:
        return None

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
    r_std, g_std, b_std = r.std(), g.std(), b.std()

    # RGB -> HSV (OpenCV: H in [0,179], S,V in [0,255])
    img_hsv = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    h = img_hsv[:, :, 0].astype(np.float32)
    s = img_hsv[:, :, 1].astype(np.float32)
    v = img_hsv[:, :, 2].astype(np.float32)

    h_mean = h.mean()
    h_std = h.std()
    s_mean = s.mean()
    v_mean = v.mean()

    return np.array(
        [
            r_mean,
            g_mean,
            b_mean,
            r_std,
            g_std,
            b_std,
            h_mean,
            h_std,
            s_mean,
            v_mean,
        ],
        dtype=np.float32,
    )
