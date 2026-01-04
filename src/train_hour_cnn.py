# train_hour_cnn.py
# -----------------------------------------------------------------------------
# Trening prostego modelu CNN na pełnych obrazach do przewidywania godziny.
# - wejście: obraz RGB z labels.csv (kolumny: filepath, hour, datetime)
# - wyjście: klasyfikacja godziny (24 klasy: 0..23)
# - zapis modelu: models/pc/best_cnn_hour.pt (torch.save(model))
#
# Uruchom (z katalogu głównego repo):
#   python -m src.train_hour_cnn \
#       --epochs 10 --batch-size 64 --img-size 224
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .load_data import load_labels
from .settings import DATA_DIR, LABELS_CSV


class HourImageDataset(Dataset):
    """Dataset oparty o labels.csv (filepath, hour, datetime)."""

    def __init__(self, labels_df, root_dir: Path, img_size: int = 224) -> None:
        self.df = labels_df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.img_size = int(img_size)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.df)

    def __getitem__(self, idx: int):  # type: ignore[override]
        row = self.df.iloc[idx]
        rel_path = str(row["filepath"])
        full_path = self.root_dir / rel_path

        img = cv2.imread(str(full_path))
        if img is None:
            # awaryjnie: czarny obraz
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32) / 255.0  # [0,1]
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        x = torch.from_numpy(img)  # float32 tensor, shape (3, H, W)

        hour = int(row["hour"])
        y = torch.tensor(hour, dtype=torch.long)
        return x, y


class SmallHourCNN(nn.Module):
    """Lekki CNN do klasyfikacji godziny (24 klasy)."""

    def __init__(self, num_classes: int = 24) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/4
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/8
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc=f"Train {epoch}/{total_epochs}", leave=False)):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def eval_one_epoch(
    model,
    loader,
    criterion,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Val   {epoch}/{total_epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            running_loss += float(loss.item()) * x.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(x.size(0))

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def main() -> None:
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-frac", type=float, default=0.1, help="Ułamek danych na walidację (0..1)")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Używane urządzenie: {device}")

    # 1) Wczytanie labels.csv (domyślne ścieżki z settings.py)
    labels_path = LABELS_CSV if LABELS_CSV is not None else Path("labels.csv")
    data_root = DATA_DIR if DATA_DIR is not None else Path("dataset")
    print(f"[INFO] Wczytuję etykiety z: {labels_path}")
    print(f"[INFO] Katalog z danymi: {data_root}")

    df = load_labels(labels_path)
    print(f"[INFO] Liczba rekordów w labels.csv: {len(df)}")

    # 2) Dataset i podział na train/val
    dataset = HourImageDataset(df, root_dir=data_root, img_size=args.img_size)
    n_total = len(dataset)
    n_val = int(args.val_frac * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"[INFO] Podział: train={n_train}, val={n_val}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # 3) Model, optymalizator, loss
    model = SmallHourCNN(num_classes=24).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4) Trening z wyborem najlepszego modelu po val_loss
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device, epoch, args.epochs
        )

        epoch_time = time.time() - t_epoch_start

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
            f"time={epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            print(f"[INFO] Nowy najlepszy model (val_loss={val_loss:.4f})")

    # 5) Zapis najlepszego modelu (state_dict + metadane, bez picklowania całego obiektu)
    models_dir = Path(__file__).resolve().parent.parent / "models" / "pc"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "best_cnn_hour.pt"

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = {
        "model_state": model.state_dict(),
        "num_classes": 24,
        "img_size": args.img_size,
    }
    torch.save(ckpt, ckpt_path)
    print(f"[INFO] Zapisano model CNN do: {ckpt_path}")
    total_time = time.time() - t0
    print(f"[INFO] Całkowity czas trenowania: {total_time:.2f}s")


if __name__ == "__main__":
    main()
