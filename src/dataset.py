"""
EuroSAT dataset — 3 kanały RGB, resize 256x256.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT
from torchvision.transforms import v2

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

CLASS_COLORS = [
    (255, 211, 0),
    (34, 139, 34),
    (144, 238, 144),
    (128, 128, 128),
    (139, 0, 0),
    (0, 200, 100),
    (255, 165, 0),
    (255, 69, 0),
    (0, 105, 148),
    (0, 191, 255),
]

NUM_CLASSES = len(CLASSES)

# Statystyki RGB dla EuroSAT (kanały B04, B03, B02 / 10000)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 256


class EuroSATDataset(torch.utils.data.Dataset):
    def __init__(self, split: str = "train", download: bool = False):
        self.base   = EuroSAT(root="data/raw", split=split, download=download)
        self.train  = split == "train"
        self.transforms = self._build_transforms()

    def _build_transforms(self):
        ops = [v2.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if self.train:
            ops += [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=90),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ]
        ops.append(v2.Normalize(mean=MEAN, std=STD))
        return v2.Compose(ops)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        # B04=Red, B03=Green, B02=Blue — 3 kanały RGB
        img   = sample["image"][[3, 2, 1]].float() / 10000.0
        img   = self.transforms(img)
        label = sample["label"].long()
        return img, label


def get_dataloaders(batch_size: int = 16):
    train_ds = EuroSATDataset(split="train")
    val_ds   = EuroSATDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader
