"""
Fine-tuning SegFormera na etykietach Esri LULC.
Prawdziwe etykiety z Impact Observatory zamiast pseudo-labeli.
"""
import json
import os
import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.transforms import v2
from tqdm import tqdm

from src.dataset import CLASSES, NUM_CLASSES
from src.model import build_model, export_to_onnx, NUM_CLASSES
from src.train import FocalLoss

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "data/processed"
IMAGE_SIZE = 256

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CONFIG = {
    "batch_size":   16,
    "epochs":       15,
    "lr":           5e-5,   # mały LR — fine-tuning
    "weight_decay": 1e-2,
    "patience":     5,
}

# Wagi klas — kompensacja dominacji Residential
CLASS_WEIGHTS = {
    "AnnualCrop":           1.0,
    "Forest":               2.0,
    "HerbaceousVegetation": 3.0,
    "Highway":              4.0,
    "Industrial":           4.0,
    "Pasture":              3.0,
    "PermanentCrop":        3.0,
    "Residential":          0.3,   # nadreprezentowany
    "River":                4.0,
    "SeaLake":              4.0,
}


class EsriLabeledDataset(Dataset):
    def __init__(self, entries: list, train: bool = True):
        self.entries = entries
        ops = [v2.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if train:
            ops += [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(90),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        ops.append(v2.Normalize(mean=MEAN, std=STD))
        self.transforms = v2.Compose(ops)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e   = self.entries[idx]
        img = Image.open(f"data/wms_dataset/images/{e['file']}").convert("RGB")
        arr = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        arr = arr.permute(2, 0, 1)
        arr = self.transforms(arr)
        lbl = torch.tensor(CLASSES.index(e["label"]), dtype=torch.long)
        return arr, lbl


def get_dataloaders():
    with open("data/wms_dataset/labeled_esri.json") as f:
        entries = json.load(f)

    torch.manual_seed(42)
    perm  = torch.randperm(len(entries)).tolist()
    n_val = int(len(entries) * 0.2)
    val_e   = [entries[i] for i in perm[:n_val]]
    train_e = [entries[i] for i in perm[n_val:]]

    train_ds = EsriLabeledDataset(train_e, train=True)
    val_ds   = EsriLabeledDataset(val_e,   train=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_loader, val_loader = get_dataloaders()
    print(f"Device: {DEVICE}")

    # Załaduj najlepszy model EuroSAT jako punkt startowy
    model = build_model(pretrained=False).to(DEVICE)
    ckpt  = os.path.join(OUTPUT_DIR, "best_model.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"Załadowano checkpoint EuroSAT: {ckpt}")

    weights = torch.tensor(
        [CLASS_WEIGHTS[cls] for cls in CLASSES], dtype=torch.float32
    ).to(DEVICE)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"],
                      weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler    = torch.amp.GradScaler("cuda")
    acc_m     = MulticlassAccuracy(num_classes=NUM_CLASSES).to(DEVICE)
    f1_m      = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(DEVICE)

    best_f1 = 0.0; no_improve = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total = 0.0
        for imgs, labels in tqdm(train_loader, desc="Train", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = criterion(model(pixel_values=imgs).logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        train_loss = total / len(train_loader)

        model.eval(); acc_m.reset(); f1_m.reset(); val_total = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Val  ", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast("cuda"):
                    logits = model(pixel_values=imgs).logits
                    val_total += criterion(logits, labels).item()
                acc_m.update(logits.argmax(1), labels)
                f1_m.update(logits.argmax(1), labels)
        val_loss = val_total / len(val_loader)
        val_acc  = acc_m.compute().item()
        val_f1   = f1_m.compute().item()
        scheduler.step()

        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"loss {train_loss:.4f} → {val_loss:.4f} | "
              f"acc {val_acc:.4f} | f1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1; no_improve = 0
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model_esri.pt")
            print(f"  ✓ Nowy najlepszy model Esri (f1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"  Early stopping po {epoch} epokach.")
                break

    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model_esri.pt",
                                     map_location="cpu"))
    export_to_onnx(model.cpu(), f"{OUTPUT_DIR}/model_esri.onnx")
    print(f"\nGotowe. Najlepszy F1: {best_f1:.4f}")
    print(f"Model: {OUTPUT_DIR}/model_esri.onnx")


if __name__ == "__main__":
    main()
