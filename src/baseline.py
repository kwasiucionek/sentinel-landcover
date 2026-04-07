"""
XGBoost baseline — klasyfikacja pokrycia terenu na podstawie
cech spektralnych wyciągniętych z patchów Sentinel-2.
Służy jako punkt odniesienia dla SegFormera.
"""
import json
import os
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401  — rejestruje accessor .rio
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torchgeo.datasets import EuroSAT

BANDS = [3, 2, 1, 7]  # R, G, B, NIR — te same co SegFormer
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]


def extract_features(dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Wyciąga cechy spektralne z każdego patcha:
    - średnia per kanał (4)
    - odchylenie std per kanał (4)
    - min/max per kanał (8)
    - NDVI: (NIR - Red) / (NIR + Red + eps) (1)
    Razem: 17 cech per próbka.
    """
    X, y = [], []
    for sample in dataset:
        img = sample["image"][BANDS].float().numpy() / 10000.0  # (4, H, W)
        mean = img.mean(axis=(1, 2))       # (4,)
        std  = img.std(axis=(1, 2))        # (4,)
        mn   = img.min(axis=(1, 2))        # (4,)
        mx   = img.max(axis=(1, 2))        # (4,)
        red, nir = img[0], img[3]
        ndvi = ((nir - red) / (nir + red + 1e-8)).mean()
        features = np.concatenate([mean, std, mn, mx, [ndvi]])
        X.append(features)
        y.append(sample["label"].item())
    return np.array(X), np.array(y)


def train_baseline(data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print("Ładowanie danych...")
    train_ds = EuroSAT(root=data_dir, split="train", download=False)
    val_ds   = EuroSAT(root=data_dir, split="val",   download=False)

    print("Ekstrakcja cech treningowych...")
    X_train, y_train = extract_features(train_ds)
    print("Ekstrakcja cech walidacyjnych...")
    X_val, y_val = extract_features(val_ds)

    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
        device="cpu",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    preds = model.predict(X_val)
    report = classification_report(y_val, preds, target_names=CLASSES, output_dict=True)

    print("\n=== XGBoost Baseline ===")
    print(classification_report(y_val, preds, target_names=CLASSES))

    results = {
        "model": "XGBoost",
        "features": "mean+std+min+max+ndvi per channel",
        "n_features": X_train.shape[1],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "report": report,
    }
    out_path = os.path.join(output_dir, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wyniki zapisane: {out_path}")

    return model, report["macro avg"]["f1-score"]


if __name__ == "__main__":
    model, f1 = train_baseline("data/raw", "data/processed")
    print(f"\nXGBoost Macro F1: {f1:.4f}")
    print("(porównaj z SegFormer F1 z data/processed/history.json)")
