import sys
sys.path.insert(0, ".")
import matplotlib.pyplot as plt
import numpy as np
from torchgeo.datasets import EuroSAT100
from src.dataset import CLASSES, CLASS_COLORS

# Surowy dataset bez normalizacji
base = EuroSAT100(root="data/raw", download=False)
print(f"Dataset size: {len(base)}")

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
indices = np.random.choice(len(base), 9, replace=False)

for ax, idx in zip(axes.flat, indices):
    sample = base[idx]
    image = sample["image"][[3, 2, 1], :, :]   # tylko RGB (bez NIR)
    label = sample["label"]

    rgb = image.permute(1, 2, 0).numpy().astype(float)
    rgb = rgb / rgb.max() * 255 if rgb.max() > 0 else rgb  # stretch do [0,255]
    rgb = rgb.clip(0, 255).astype("uint8")

    ax.imshow(rgb)
    color = [c/255 for c in CLASS_COLORS[label]]
    ax.set_title(CLASSES[label], color=color, fontweight="bold")
    ax.axis("off")

plt.suptitle("EuroSAT — próbki Sentinel-2", fontsize=14)
plt.tight_layout()
plt.savefig("data/processed/eda_grid.png", dpi=150)
plt.show()
print("Zapisano: data/processed/eda_grid.png")
