"""
Fine-tuning SegFormera na danych WMS z etykietami 'hint'.
Używamy hint jako pseudo-etykiet — obszary geograficznie dobrane.
"""

import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from src.dataset import CLASSES
from src.model import NUM_CLASSES, build_model, export_to_onnx
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.transforms import v2
from tqdm import tqdm

DATASET_DIR = "data/wms_dataset"
OUTPUT_DIR = "data/processed"
IMAGE_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = torch.tensor([0.485, 0.456, 0.406, 0.35])
STD = torch.tensor([0.229, 0.224, 0.225, 0.15])

CONFIG = {
    "batch_size": 32,
    "epochs": 15,
    "lr": 5e-5,  # bardzo mały LR — fine-tuning, nie trening od zera
    "weight_decay": 1e-2,
    "val_split": 0.2,
    "patience": 5,
}

# Wagi klas — wyrównanie nadreprezentowanego Residential
CLASS_WEIGHTS = {
    "AnnualCrop": 1.5,
    "Forest": 1.5,
    "HerbaceousVegetation": 1.5,
    "Highway": 1.0,
    "Industrial": 1.0,
    "Pasture": 1.5,
    "PermanentCrop": 1.5,
    "Residential": 0.5,  # nadreprezentowany
    "River": 1.0,
    "SeaLake": 1.0,
}


class WMSDataset(Dataset):
    def __init__(self, entries: list, train: bool = True):
        self.entries = entries
        self.train = train
        ops = []
        if train:
            ops += [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=90),
            ]
        ops.append(v2.Normalize(mean=MEAN.tolist(), std=STD.tolist()))
        self.transforms = v2.Compose(ops)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = Image.open(f"{DATASET_DIR}/images/{entry['file']}").convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        arr = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        arr = arr.permute(2, 0, 1)  # (3, H, W)
        nir = arr.mean(dim=0, keepdim=True)  # symulowany NIR
        arr = torch.cat([arr, nir], dim=0)  # (4, H, W)
        arr = self.transforms(arr)

        label = CLASSES.index(entry["hint"])
        return arr, label


def get_dataloaders(manifest_path: str):
    with open(manifest_path) as f:
        entries = json.load(f)

    # Filtruj patche bez hinta
    entries = [e for e in entries if e.get("hint") in CLASSES]
    print(f"Dataset: {len(entries)} patchów")

    n_val = int(len(entries) * CONFIG["val_split"])
    n_train = len(entries) - n_val
    train_e, val_e = random_split(
        entries, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_ds = WMSDataset(list(train_e), train=True)
    val_ds = WMSDataset(list(val_e), train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(pixel_values=images).logits
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, acc_metric, f1_metric):
    model.eval()
    acc_metric.reset()
    f1_metric.reset()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="Val  ", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits = model(pixel_values=images).logits
            loss = criterion(logits, labels)
        total_loss += loss.item()
        acc_metric.update(logits.argmax(dim=1), labels)
        f1_metric.update(logits.argmax(dim=1), labels)
    return (
        total_loss / len(loader),
        acc_metric.compute().item(),
        f1_metric.compute().item(),
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader = get_dataloaders(f"{DATASET_DIR}/manifest.json")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    print(f"Device: {DEVICE}")

    # Załaduj najlepszy checkpoint z poprzedniego treningu
    model = build_model(pretrained=False).to(DEVICE)
    ckpt = os.path.join(OUTPUT_DIR, "best_model.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"Załadowano checkpoint: {ckpt}")
    else:
        print("Brak checkpointu — trening od pretrained ImageNet")

    # Wagi klas
    weights = torch.tensor(
        [CLASS_WEIGHTS[cls] for cls in CLASSES], dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler = torch.amp.GradScaler("cuda")

    acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES).to(DEVICE)
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(DEVICE)

    best_f1 = 0.0
    no_improve = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, acc_metric, f1_metric
        )
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
            f"loss {train_loss:.4f} → {val_loss:.4f} | "
            f"acc {val_acc:.4f} | f1 {val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(
                model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_wms.pt")
            )
            print(f"  ✓ Nowy najlepszy model WMS (f1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"  Early stopping po {epoch} epokach.")
                break

    # Eksport najlepszego modelu WMS jako ONNX
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "best_model_wms.pt"), map_location="cpu")
    )
    onnx_path = os.path.join(OUTPUT_DIR, "model_wms.onnx")
    export_to_onnx(model.cpu(), onnx_path, IMAGE_SIZE)
    print(f"\nGotowe. Najlepszy F1: {best_f1:.4f}")
    print(f"Model WMS: {onnx_path}")


if __name__ == "__main__":
    main()
