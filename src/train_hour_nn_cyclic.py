# src/train_hour_nn_cyclic.py — Run: python -m src.train_hour_nn_cyclic
# -----------------------------------------------------------------------------
# Cel: regresja dokładnej godziny jako problem cykliczny (24h).
# Uczymy sieć przewidywać (sin, cos) zamiast liczby hour, co rozwiązuje 23≈0.
#
# Wejście: cechy tabelaryczne z features_robust.csv (np. 56 cech)
# Wyjście: 2 wartości: [sin(theta), cos(theta)]
# Metryka: Cyclic MAE (w godzinach) + P90/P95
#------------------------------------------------------------------------------

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


FEATURES_PATH = Path("features_robust.csv")
MODEL_OUT = Path("models") / "pc"
MODEL_OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


# --- cykliczne kodowanie celu ---
def hour_to_sin_cos(hour: np.ndarray) -> np.ndarray:
    theta = 2.0 * np.pi * (hour.astype(np.float32) / 24.0)
    return np.stack([np.sin(theta), np.cos(theta)], axis=1)


def sin_cos_to_hour(y_sc: np.ndarray) -> np.ndarray:
    theta = np.arctan2(y_sc[:, 0], y_sc[:, 1])  # [-pi, pi]
    theta = theta % (2.0 * np.pi)               # [0, 2pi)
    return theta * 24.0 / (2.0 * np.pi)         # [0, 24)


def circular_hour_error(pred_hour: np.ndarray, true_hour: np.ndarray) -> np.ndarray:
    diff = np.abs(pred_hour - true_hour)
    return np.minimum(diff, 24.0 - diff)


# --- dataset pytorchowy ---
class TabularTimeDataset(Dataset):
    def __init__(self, X: np.ndarray, y_sc: np.ndarray, y_hour: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y_sc = torch.from_numpy(y_sc).float()
        self.y_hour = torch.from_numpy(y_hour).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_sc[idx], self.y_hour[idx]


# --- prosta, mocna MLP ---
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(128, 2),  # [sin, cos]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainConfig:
    batch_size: int = 4096
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    num_workers: int = 0  # na Windows/WSL czasem 0 jest najpewniejsze


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    all_pred = []
    all_true_hour = []

    for Xb, yb_sc, yb_hour in loader:
        Xb = Xb.to(device)
        pred_sc = model(Xb).cpu().numpy()
        all_pred.append(pred_sc)
        all_true_hour.append(yb_hour.numpy())

    pred_sc = np.vstack(all_pred)
    true_hour = np.concatenate(all_true_hour)

    pred_hour = sin_cos_to_hour(pred_sc)
    circ_err = circular_hour_error(pred_hour, true_hour)

    cmae = float(circ_err.mean())
    p90 = float(np.percentile(circ_err, 90))
    p95 = float(np.percentile(circ_err, 95))
    return cmae, p90, p95


def main() -> None:
    t0 = time.time()
    cfg = TrainConfig()

    # --- urządzenie ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # --- wczytanie danych ---
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku {FEATURES_PATH}. Najpierw uruchom precompute_features_robust."
        )

    log(f"Wczytywanie danych z: {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)
    log(f"Shape: {df.shape}")

    if "hour" not in df.columns:
        raise ValueError("Brak kolumny 'hour' w features_robust.csv")

    drop_cols = {"filepath", "datetime", "hour"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    log(f"Liczba cech: {len(feature_cols)}")

    # --- macierze ---
    X = df[feature_cols].astype(np.float32).to_numpy()
    y_hour = df["hour"].astype(np.float32).to_numpy()
    y_sc = hour_to_sin_cos(y_hour)

    # --- split: train/val/test (80/10/10) ---
    # stratify po godzinie, aby rozkład był równy
    log("Split: train/val/test z zachowaniem proporcji godzin")
    rng = np.random.RandomState(42)
    hour_int = df["hour"].astype(int).to_numpy()

    # najpierw wydzielamy test 10%
    from sklearn.model_selection import train_test_split
    X_tr, X_te, ysc_tr, ysc_te, yh_tr, yh_te, h_tr, h_te = train_test_split(
        X, y_sc, y_hour, hour_int,
        test_size=0.10,
        random_state=42,
        stratify=hour_int,
    )
    # potem z train wydzielamy val ~11.11% z pozostałego => ~10% całości
    X_tr, X_va, ysc_tr, ysc_va, yh_tr, yh_va, h_tr, h_va = train_test_split(
        X_tr, ysc_tr, yh_tr, h_tr,
        test_size=0.1111,
        random_state=42,
        stratify=h_tr,
    )

    log(f"Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")

    # --- normalizacja cech (ważne dla NN) ---
    log("Standaryzacja cech (mean=0, std=1) na bazie TRAIN")
    mean = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr = (X_tr - mean) / std
    X_va = (X_va - mean) / std
    X_te = (X_te - mean) / std

    # --- dataloadery ---
    train_ds = TabularTimeDataset(X_tr, ysc_tr, yh_tr)
    val_ds = TabularTimeDataset(X_va, ysc_va, yh_va)
    test_ds = TabularTimeDataset(X_te, ysc_te, yh_te)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))

    # --- model ---
    model = MLP(in_dim=X_tr.shape[1]).to(device)
    log(f"Model: MLP(in_dim={X_tr.shape[1]})")

    # --- loss / opt ---
    # Huber bywa stabilniejszy niż MSE przy ogonie błędów (chmury, ekstremalne warunki)
    loss_fn = nn.SmoothL1Loss(beta=0.5)  # Huber
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # mały scheduler: obniż LR, jeśli walidacja stoi
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best_val = float("inf")
    best_epoch = -1
    best_path = MODEL_OUT / "best_mlp_cyclic.pt"
    bad_epochs = 0

    log("Start treningu")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t_ep = time.time()

        running = 0.0
        n = 0

        for Xb, yb_sc, _yb_hour in train_loader:
            Xb = Xb.to(device)
            yb_sc = yb_sc.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(Xb)

            # (opcjonalnie) lekkie "ściąganie" predykcji do okręgu:
            # normalizuje wektor [sin, cos] do długości 1, żeby nie "odpływał".
            pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)

            loss = loss_fn(pred, yb_sc)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            opt.step()

            running += float(loss.item()) * Xb.size(0)
            n += Xb.size(0)

        train_loss = running / max(n, 1)

        # walidacja (Cyclic MAE)
        val_cmae, val_p90, val_p95 = evaluate(model, val_loader, device)
        scheduler.step(val_cmae)

        lr_now = opt.param_groups[0]["lr"]
        log(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.5f} | val_cmae={val_cmae:.3f}h | "
            f"p90={val_p90:.3f}h | p95={val_p95:.3f}h | lr={lr_now:.2e} | "
            f"time={time.time() - t_ep:.1f}s"
        )

        # early stopping
        if val_cmae + 1e-6 < best_val:
            best_val = val_cmae
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                    "feature_cols": feature_cols,
                    "best_val_cmae": best_val,
                    "best_epoch": best_epoch,
                },
                best_path,
            )
            log(f"Zapisano best model -> {best_path} (val_cmae={best_val:.3f}h)")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log(f"Early stopping: brak poprawy przez {cfg.patience} epok (best_epoch={best_epoch})")
                break

    # --- test na najlepszym checkpoint ---
    log(f"Ładowanie najlepszego modelu z: {best_path}")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_cmae, test_p90, test_p95 = evaluate(model, test_loader, device)

    log("=" * 80)
    log(f"BEST epoch: {ckpt['best_epoch']}, best_val_cmae={ckpt['best_val_cmae']:.3f}h")
    log(f"TEST  Cyclic MAE: {test_cmae:.3f}h  (~{test_cmae*60:.1f} min)")
    log(f"TEST  P90:        {test_p90:.3f}h  (~{test_p90*60:.1f} min)")
    log(f"TEST  P95:        {test_p95:.3f}h  (~{test_p95*60:.1f} min)")
    log(f"Całkowity czas:   {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
