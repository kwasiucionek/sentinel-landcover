"""
Fine-tuning ResNet50 (SSL4EO-S12) na EuroSAT + WMS dataset.
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.transforms import v2
from tqdm import tqdm

from src.dataset import CLASSES
from src.model_geo import NUM_CLASSES, PATCH_SIZE, build_geo_model, export_geo_to_onnx

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "data/processed"
CONFIG     = {
    "batch_size": 32, "epochs": 20, "lr": 1e-4,
    "weight_decay": 1e-2, "val_split": 0.2, "patience": 6,
}
MEAN = [0.485, 0.456, 0.406, 0.35]
STD  = [0.229, 0.224, 0.225, 0.15]


class EuroSATGeoDataset(Dataset):
    def __init__(self, split: str = "train"):
        from torchgeo.datasets import EuroSAT
        self.base = EuroSAT(root="data/raw", split=split, download=False)
        train = split == "train"
        ops = []
        if train:
            ops += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
                    v2.RandomRotation(90)]
        ops.append(v2.Normalize(mean=MEAN, std=STD))
        self.transforms = v2.Compose(ops)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        img    = sample["image"][[3, 2, 1, 7]].float() / 10000.0
        img    = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear"
        ).squeeze(0)
        img    = self.transforms(img)
        label  = sample["label"].long()
        return img, label


class WMSGeoDataset(Dataset):
    def __init__(self, entries: list, train: bool = True):
        self.entries = entries
        ops = []
        if train:
            ops += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
                    v2.RandomRotation(90)]
        ops.append(v2.Normalize(mean=MEAN, std=STD))
        self.transforms = v2.Compose(ops)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e   = self.entries[idx]
        img = Image.open(f"data/wms_dataset/images/{e['file']}").convert("RGB")
        img = img.resize((PATCH_SIZE, PATCH_SIZE))
        arr = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        arr = arr.permute(2, 0, 1)
        nir = arr.mean(dim=0, keepdim=True)
        arr = torch.cat([arr, nir], dim=0)
        arr = self.transforms(arr)
        lbl = torch.tensor(CLASSES.index(e["hint"]), dtype=torch.long)
        return arr, lbl


def get_dataloaders():
    with open("data/wms_dataset/manifest.json") as f:
        all_entries = [e for e in json.load(f) if e.get("hint") in CLASSES]

    torch.manual_seed(42)
    perm  = torch.randperm(len(all_entries)).tolist()
    n_val = int(len(all_entries) * CONFIG["val_split"])
    val_entries   = [all_entries[i] for i in perm[:n_val]]
    train_entries = [all_entries[i] for i in perm[n_val:]]

    train_ds = ConcatDataset([
        EuroSATGeoDataset(split="train"),
        WMSGeoDataset(train_entries, train=True),
    ])
    val_ds = ConcatDataset([
        EuroSATGeoDataset(split="val"),
        WMSGeoDataset(val_entries, train=False),
    ])

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total = 0.0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, acc_m, f1_m):
    model.eval()
    acc_m.reset(); f1_m.reset()
    total = 0.0
    for imgs, labels in tqdm(loader, desc="Val  ", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits = model(imgs)
            total += criterion(logits, labels).item()
        acc_m.update(logits.argmax(1), labels)
        f1_m.update(logits.argmax(1), labels)
    return total / len(loader), acc_m.compute().item(), f1_m.compute().item()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_loader, val_loader = get_dataloaders()
    print(f"Device: {DEVICE}")

    model     = build_geo_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"],
                      weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler    = torch.amp.GradScaler("cuda")
    acc_m     = MulticlassAccuracy(num_classes=NUM_CLASSES).to(DEVICE)
    f1_m      = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(DEVICE)

    best_f1 = 0.0; no_improve = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, acc_m, f1_m)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"loss {train_loss:.4f} → {val_loss:.4f} | "
              f"acc {val_acc:.4f} | f1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1; no_improve = 0
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model_geo.pt")
            print(f"  ✓ Nowy najlepszy model geo (f1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"  Early stopping po {epoch} epokach.")
                break

    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model_geo.pt",
                                     map_location="cpu"))
    export_geo_to_onnx(model.cpu(), f"{OUTPUT_DIR}/model_geo.onnx")
    print(f"\nGotowe. Najlepszy F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
