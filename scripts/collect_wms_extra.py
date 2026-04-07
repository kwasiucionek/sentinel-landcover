"""Dozbiera patche dla brakujących klas."""
import os, sys, json
sys.path.insert(0, ".")
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from src.cdse_client import CDSEClient

load_dotenv()

OUTPUT_DIR = "data/wms_dataset"
PATCH_SIZE = 64
DATES      = ["2026-03-07", "2026-03-14", "2026-03-22"]

EXTRA_AREAS = {
    # łąki i pastwiska — okolice Siechnic
    "laki_siechnice":    (17.100, 51.050, 17.170, 51.090, "HerbaceousVegetation"),
    "pastwiska_dlugoleka": (17.150, 51.160, 17.220, 51.200, "Pasture"),
    # sady i uprawy wieloletnie — okolice Oleśnicy
    "sady_olesnica":     (17.350, 51.180, 17.420, 51.220, "PermanentCrop"),
    # mniej Residential — małe osiedla
    "osiedle_biskupin":  (17.030, 51.120, 17.070, 51.145, "Residential"),
}

def tile_to_patches(tile):
    arr = np.array(tile.convert("RGB"))
    h, w = arr.shape[:2]
    patches = []
    for r in range(h // PATCH_SIZE):
        for c in range(w // PATCH_SIZE):
            patches.append(arr[r*PATCH_SIZE:(r+1)*PATCH_SIZE,
                               c*PATCH_SIZE:(c+1)*PATCH_SIZE])
    return patches

def main():
    with open(f"{OUTPUT_DIR}/manifest.json") as f:
        manifest = json.load(f)
    patch_idx = len(manifest)

    client      = CDSEClient()
    instance_id = os.getenv("CDSE_INSTANCE_ID")

    for area_name, (lon_min, lat_min, lon_max, lat_max, hint) in EXTRA_AREAS.items():
        for date in DATES:
            print(f"Pobieranie: {area_name} / {date}...", end=" ")
            tile = client.fetch_wms_preview(
                instance_id, lon_min, lat_min, lon_max, lat_max,
                layer="TRUE-COLOR", size=512, date=date,
            )
            if tile is None:
                print("BRAK")
                continue
            patches = tile_to_patches(tile)
            print(f"{len(patches)} patchów")
            for patch in patches:
                fname = f"{patch_idx:05d}.png"
                Image.fromarray(patch).save(f"{OUTPUT_DIR}/images/{fname}")
                manifest.append({
                    "file": fname, "area": area_name,
                    "date": date,  "hint": hint, "label": None,
                })
                patch_idx += 1

    with open(f"{OUTPUT_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    from collections import Counter
    hints = Counter(p["hint"] for p in manifest)
    print(f"\nGotowe: {patch_idx} patchów total")
    print("Rozkład:", dict(hints.most_common()))

if __name__ == "__main__":
    main()
