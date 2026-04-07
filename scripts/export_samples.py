import sys
sys.path.insert(0, ".")
import numpy as np
import os
from PIL import Image
from torchgeo.datasets import EuroSAT
from src.dataset import CLASSES

os.makedirs("data/samples", exist_ok=True)

ds = EuroSAT(root="data/raw", split="val", download=False)

# Eksportuj 3 próbki z każdej klasy
exported = {cls: 0 for cls in CLASSES}
for i, sample in enumerate(ds):
    label = sample["label"].item()
    cls = CLASSES[label]
    if exported[cls] >= 3:
        continue
    img = sample["image"][[3, 2, 1]].numpy().astype(float)  # RGB
    img = (img / img.max() * 255).clip(0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)  # HWC
    pil = Image.fromarray(img).resize((256, 256), Image.NEAREST)
    fname = f"data/samples/{cls}_{exported[cls]+1}.png"
    pil.save(fname)
    exported[cls] += 1
    if all(v >= 3 for v in exported.values()):
        break

print("Zapisano próbki do data/samples/:")
for f in sorted(os.listdir("data/samples")):
    print(f"  {f}")
