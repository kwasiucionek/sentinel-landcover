"""
Zbiera patche WMS z Sentinel Hub i tworzy dataset do fine-tuningu.
Strategia: siatka lokalizacji Wrocławia x kilka dat = ~1000 patchów.
"""
import os
import sys
import json
sys.path.insert(0, ".")

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from src.cdse_client import CDSEClient
from src.dataset import CLASSES

load_dotenv()

# ── Konfiguracja ──────────────────────────────────────────────────────────────
OUTPUT_DIR   = "data/wms_dataset"
PATCH_SIZE   = 64
DATES        = ["2026-03-07", "2026-03-14", "2026-03-22"]
TILE_SIZE    = 512   # pobieramy 512x512 → kroimy na patche 64x64

# Siatka obszarów Wrocławia — różne typy pokrycia terenu
AREAS = {
    # (lon_min, lat_min, lon_max, lat_max, dominant_class_hint)
    "centrum_miasto":     (16.978, 51.095, 17.045, 51.130, "Residential"),
    "psie_pole":          (17.040, 51.130, 17.110, 51.165, "Residential"),
    "lasy_poludniowe":    (16.950, 51.020, 17.020, 51.060, "Forest"),
    "odra_centrum":       (16.990, 51.105, 17.060, 51.130, "River"),
    "fabryczna_przemysl": (16.880, 51.075, 16.950, 51.110, "Industrial"),
    "pola_dlugoleka":     (17.100, 51.180, 17.180, 51.220, "AnnualCrop"),
    "lasy_oborniki":      (16.850, 51.200, 16.920, 51.240, "Forest"),
    "jezioro_siedlec":    (16.820, 51.150, 16.880, 51.180, "SeaLake"),
    "krzyki_mieszk":      (17.010, 51.065, 17.080, 51.100, "Residential"),
    "autostrada_a4":      (16.900, 51.090, 16.970, 51.110, "Highway"),
}


def tile_to_patches(tile: Image.Image) -> list[np.ndarray]:
    """Kroi tile 512x512 na siatkę patchów 64x64."""
    arr    = np.array(tile.convert("RGB"))
    h, w   = arr.shape[:2]
    n_rows = h // PATCH_SIZE
    n_cols = w // PATCH_SIZE
    patches = []
    for r in range(n_rows):
        for c in range(n_cols):
            patch = arr[r*PATCH_SIZE:(r+1)*PATCH_SIZE,
                        c*PATCH_SIZE:(c+1)*PATCH_SIZE]
            patches.append(patch)
    return patches


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

    client      = CDSEClient()
    instance_id = os.getenv("CDSE_INSTANCE_ID")
    manifest    = []
    patch_idx   = 0

    for area_name, (lon_min, lat_min, lon_max, lat_max, hint) in AREAS.items():
        for date in DATES:
            print(f"Pobieranie: {area_name} / {date}...", end=" ")
            tile = client.fetch_wms_preview(
                instance_id, lon_min, lat_min, lon_max, lat_max,
                layer="TRUE-COLOR", size=TILE_SIZE, date=date,
            )
            if tile is None:
                print("BRAK SCENY")
                continue

            patches = tile_to_patches(tile)
            print(f"{len(patches)} patchów")

            for patch in patches:
                fname = f"{patch_idx:05d}.png"
                Image.fromarray(patch).save(f"{OUTPUT_DIR}/images/{fname}")
                manifest.append({
                    "file":     fname,
                    "area":     area_name,
                    "date":     date,
                    "hint":     hint,   # sugerowana klasa (do weryfikacji)
                    "label":    None,   # wypełni bootstrap
                })
                patch_idx += 1

    # Zapisz manifest
    with open(f"{OUTPUT_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nGotowe: {patch_idx} patchów → {OUTPUT_DIR}/")
    print(f"Manifest: {OUTPUT_DIR}/manifest.json")


if __name__ == "__main__":
    main()
