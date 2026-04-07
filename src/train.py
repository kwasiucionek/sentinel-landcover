"""
Trening SegFormer-B2 na EuroSAT.
RGB 256x256, Focal Loss, early stopping.
"""
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm

from .model import build_model, export_to_onnx, NUM_CLASSES
from .dataset import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "output_dir":   "data/processed",
    "batch_size":   16,
    "epochs":       30,
    "lr":           3e-4,
    "weight_decay": 1e-2,
    "patience":     7,
}


class FocalLoss(nn.Module):
    """Focal Loss — lepiej radzi sobie z trudnymi przykładami."""
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets,
                               label_smoothing=self.label_smoothing,
                               reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total = 0.0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(pixel_values=images).logits
            loss   = criterion(logits, labels)
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
    for images, labels in tqdm(loader, desc="Val  ", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits = model(pixel_values=images).logits
            total += criterion(logits, labels).item()
        acc_m.update(logits.argmax(1), labels)
        f1_m.update(logits.argmax(1), labels)
    return total / len(loader), acc_m.compute().item(), f1_m.compute().item()


def train():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    train_loader, val_loader = get_dataloaders(CONFIG["batch_size"])
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    print(f"Device: {DEVICE}")

    model     = build_model(pretrained=True).to(DEVICE)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"],
                      weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler    = torch.amp.GradScaler("cuda")
    acc_m     = MulticlassAccuracy(num_classes=NUM_CLASSES).to(DEVICE)
    f1_m      = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(DEVICE)

    best_f1 = 0.0; no_improve = 0; history = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, acc_m, f1_m)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"loss {train_loss:.4f} → {val_loss:.4f} | "
              f"acc {val_acc:.4f} | f1 {val_f1:.4f}")
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })
        if val_f1 > best_f1:
            best_f1 = val_f1; no_improve = 0
            ckpt = os.path.join(CONFIG["output_dir"], "best_model.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Nowy najlepszy model (f1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"  Early stopping po {epoch} epokach.")
                break

    model.load_state_dict(torch.load(
        os.path.join(CONFIG["output_dir"], "best_model.pt"), map_location="cpu"))
    export_to_onnx(model.cpu(),
                   os.path.join(CONFIG["output_dir"], "model.onnx"))
    with open(os.path.join(CONFIG["output_dir"], "history.json"), "w") as f:
        json.dump({"config": CONFIG, "history": history}, f, indent=2)
    print(f"\nGotowe. Najlepszy F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()
