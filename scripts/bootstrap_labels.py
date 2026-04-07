"""
Bootstrap labeling — używa obecnego modelu ONNX do auto-labelowania patchów WMS.
Patche z pewnością > CONFIDENCE_THRESHOLD są zatwierdzone automatycznie.
Reszta odrzucana.
"""
import os, sys, json
sys.path.insert(0, ".")

import numpy as np
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort

from src.dataset import CLASSES
from src.dataset import CLASSES

CONFIDENCE_THRESHOLD = 0.80
MODEL_PATH  = "data/processed/model.onnx"
DATASET_DIR = "data/wms_dataset"
IMAGE_SIZE  = 64
MEAN = np.array([0.485, 0.456, 0.406, 0.35], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225, 0.15],  dtype=np.float32)


def preprocess(patch: np.ndarray) -> np.ndarray:
    img = Image.fromarray(patch).resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    nir = arr.mean(axis=0, keepdims=True)
    arr = np.concatenate([arr, nir], axis=0)
    arr = (arr - MEAN[:, None, None]) / STD[:, None, None]
    return arr[None].astype(np.float32)


def main():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    with open(f"{DATASET_DIR}/manifest.json") as f:
        manifest = json.load(f)

    approved  = []
    rejected  = 0
    class_counts = {cls: 0 for cls in CLASSES}

    for entry in tqdm(manifest, desc="Bootstrap"):
        img_path = f"{DATASET_DIR}/images/{entry['file']}"
        patch    = np.array(Image.open(img_path).convert("RGB"))
        inp      = preprocess(patch)

        logits   = session.run(["logits"], {"pixel_values": inp})[0][0]
        probs    = np.exp(logits) / np.exp(logits).sum()
        cls_idx  = int(np.argmax(probs))
        conf     = float(probs[cls_idx])

        if conf >= CONFIDENCE_THRESHOLD:
            entry["label"]      = CLASSES[cls_idx]
            entry["confidence"] = round(conf, 4)
            approved.append(entry)
            class_counts[CLASSES[cls_idx]] += 1
        else:
            rejected += 1

    # Zapisz zatwierdzony dataset
    out_path = f"{DATASET_DIR}/labeled.json"
    with open(out_path, "w") as f:
        json.dump(approved, f, indent=2, ensure_ascii=False)

    total = len(manifest)
    print(f"\n=== Bootstrap wyniki ===")
    print(f"Łącznie:    {total}")
    print(f"Zatwierdzone: {len(approved)} ({len(approved)/total:.1%})")
    print(f"Odrzucone:  {rejected} ({rejected/total:.1%})")
    print(f"\nRozkład zatwierdzonych klas:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        if cnt > 0:
            print(f"  {cls}: {cnt}")

    # Ostrzeżenie jeśli klasa ma < 50 próbek
    print("\n⚠ Klasy z < 50 próbkami (może być za mało do treningu):")
    for cls, cnt in class_counts.items():
        if cnt < 50:
            print(f"  {cls}: {cnt}")


if __name__ == "__main__":
    main()
