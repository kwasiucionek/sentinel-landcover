"""
Labelowanie patchów WMS używając Esri LULC jako ground truth.
Zamiast pseudo-etykiet (hint) używamy oficjalnej klasyfikacji
Impact Observatory / Microsoft / Esri (10m, 2017-2024).

Mapowanie klas Esri → EuroSAT:
1  Water              → SeaLake
2  Trees              → Forest
3  Flooded vegetation → HerbaceousVegetation
4  Crops              → AnnualCrop
5  Built area         → Residential
6  Bare ground        → Highway
7  Snow/Ice           → None (odrzucamy)
8  Clouds             → None (odrzucamy)
9  Rangeland          → Pasture
"""
import json
import os
import sys
sys.path.insert(0, ".")

import numpy as np
import requests
import io
from PIL import Image
from tqdm import tqdm

from src.esri_lulc_client import LULC_SERVICE, ESRI_CLASSES
from src.dataset import CLASSES
# Bbox-y z collect_wms_dataset.py i collect_wms_extra.py
WROCLAW_BBOXES = {
    'centrum_miasto':     (16.978, 51.095, 17.045, 51.130),
    'psie_pole':          (17.040, 51.130, 17.110, 51.165),
    'lasy_poludniowe':    (16.950, 51.020, 17.020, 51.060),
    'odra_centrum':       (16.990, 51.105, 17.060, 51.130),
    'fabryczna_przemysl': (16.880, 51.075, 16.950, 51.110),
    'pola_dlugoleka':     (17.100, 51.180, 17.180, 51.220),
    'lasy_oborniki':      (16.850, 51.200, 16.920, 51.240),
    'jezioro_siedlec':    (16.820, 51.150, 16.880, 51.180),
    'krzyki_mieszk':      (17.010, 51.065, 17.080, 51.100),
    'autostrada_a4':      (16.900, 51.090, 16.970, 51.110),
    'laki_siechnice':     (17.100, 51.050, 17.170, 51.090),
    'pastwiska_dlugoleka':(17.150, 51.160, 17.220, 51.200),
    'sady_olesnica':      (17.350, 51.180, 17.420, 51.220),
    'osiedle_biskupin':   (17.030, 51.120, 17.070, 51.145),
}

DATASET_DIR = "data/wms_dataset"
OUTPUT_PATH = f"{DATASET_DIR}/labeled_esri.json"
PATCH_SIZE  = 64   # rozmiar patcha w pikselach WMS
TILE_SIZE   = 512  # rozmiar tile WMS

# Mapowanie Esri → EuroSAT
ESRI_TO_EUROSAT = {
    1:  "SeaLake",
    2:  "Forest",
    3:  "HerbaceousVegetation",
    4:  "AnnualCrop",
    5:  "Residential",
    6:  "Highway",
    7:  None,   # Snow/Ice — odrzucamy
    8:  None,   # Clouds — odrzucamy
    9:  "Pasture",
    10: "Pasture",
    11: "HerbaceousVegetation",
}

# Minimalna pewność etykiety (% dominującej klasy w patchu)
MIN_CONFIDENCE = 0.60


# Mapowanie kolorów RGB Esri → ID klasy
ESRI_COLOR_TO_CLASS = {
    (26,  91,  171): 1,   # Water
    (53,  130,  33): 2,   # Trees
    (135, 209, 158): 3,   # Flooded vegetation / Rangeland
    (255, 219,  92): 4,   # Crops
    (237,   2,  42): 5,   # Built area
    (239, 207, 168): 6,   # Bare ground
    (237, 233, 228): 7,   # Snow/Ice
}


def rgb_to_class_arr(rgb_arr: np.ndarray) -> np.ndarray:
    """Mapuje obraz RGB Esri LULC na tablicę klas (H, W)."""
    H, W = rgb_arr.shape[:2]
    class_arr = np.zeros((H, W), dtype=np.uint8)
    for color, class_id in ESRI_COLOR_TO_CLASS.items():
        mask = (
            (rgb_arr[:, :, 0] == color[0]) &
            (rgb_arr[:, :, 1] == color[1]) &
            (rgb_arr[:, :, 2] == color[2])
        )
        class_arr[mask] = class_id
    return class_arr


def fetch_esri_lulc_raw_arr(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    year: int = 2023,
    size: int = TILE_SIZE,
) -> np.ndarray | None:
    """Pobiera LULC jako numpy array (H, W) wartości klas 1-11."""
    params = {
        "bbox":          f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "bboxSR":        "4326",
        "size":          f"{size},{size}",
        "imageSR":       "4326",
        "format":        "png",
        "f":             "image",
        "renderingRule": json.dumps({"rasterFunction": "Cartographic Renderer for Visualization and Analysis"}),
        "mosaicRule":    json.dumps({
            "mosaicMethod": "esriMosaicAttribute",
            "sortField": "Year",
            "sortValue": str(year),
            "ascending": False,
            "mosaicOperation": "MT_FIRST",
            "where": f"Year={year}",
        }),
    }
    try:
        r = requests.get(
            f"{LULC_SERVICE}/exportImage",
            params=params,
            timeout=20,
            headers={"User-Agent": "sentinel-landcover-portfolio/1.0"},
        )
        if r.status_code in (401, 499):
            return None
        r.raise_for_status()
        if "json" in r.headers.get("Content-Type", "").lower():
            return None
        rgb = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
        return rgb_to_class_arr(rgb)
    except Exception as e:
        print(f"    Błąd LULC: {e}")
        return None


def get_patch_label(
    lulc_arr: np.ndarray,
    r: int, c: int,
    patch_size: int = PATCH_SIZE,
) -> tuple[str | None, float]:
    """
    Wyznacza etykietę EuroSAT dla patcha na podstawie dominującej klasy Esri.
    Zwraca (etykieta, pewność) lub (None, 0) jeśli odrzucony.
    """
    patch = lulc_arr[
        r * patch_size:(r + 1) * patch_size,
        c * patch_size:(c + 1) * patch_size,
    ]
    if patch.size == 0:
        return None, 0.0

    # Zlicz piksele per klasa Esri
    unique, counts = np.unique(patch, return_counts=True)
    total = patch.size

    # Dominująca klasa Esri
    best_esri = None
    best_count = 0
    for val, cnt in zip(unique, counts):
        esri_id = int(val)
        if esri_id in ESRI_TO_EUROSAT and cnt > best_count:
            best_esri  = esri_id
            best_count = cnt

    if best_esri is None:
        return None, 0.0

    confidence    = best_count / total
    eurosat_class = ESRI_TO_EUROSAT.get(best_esri)

    if eurosat_class is None:
        return None, 0.0  # Snow/Ice lub Clouds — odrzucamy

    if confidence < MIN_CONFIDENCE:
        return None, confidence  # zbyt mieszany patch — odrzucamy

    return eurosat_class, confidence


def main():
    with open(f"{DATASET_DIR}/manifest.json") as f:
        manifest = json.load(f)

    print(f"Patchów w manifeście: {len(manifest)}")
    print(f"Pobieranie etykiet Esri LULC 2023...")

    labeled    = []
    rejected   = 0
    class_counts = {cls: 0 for cls in CLASSES}

    # Grupuj patche po obszarze i dacie żeby pobierać tile tylko raz
    from collections import defaultdict
    groups = defaultdict(list)
    for entry in manifest:
        key = (entry["area"], entry["date"])
        groups[key].append(entry)

    for (area, date), entries in tqdm(groups.items(), desc="Obszary"):
        if area not in WROCLAW_BBOXES:
            continue
        lon_min, lat_min, lon_max, lat_max = WROCLAW_BBOXES[area]

        lulc_arr = fetch_esri_lulc_raw_arr(
            lon_min, lat_min, lon_max, lat_max, year=2023
        )
        if lulc_arr is None:
            rejected += len(entries)
            continue

        n_rows = lulc_arr.shape[0] // PATCH_SIZE
        n_cols = lulc_arr.shape[1] // PATCH_SIZE

        for entry in entries:
            # Odtwórz pozycję patcha z nazwy pliku
            file_idx = int(os.path.splitext(entry["file"])[0])
            # Pozycja w siatce — odtwórz z globalnego indeksu
            # (patche są numerowane sekwencyjnie row by row)
            patches_in_tile = n_rows * n_cols
            local_idx       = file_idx % patches_in_tile
            r = local_idx // n_cols
            c = local_idx  % n_cols

            if r >= n_rows or c >= n_cols:
                rejected += 1
                continue

            eurosat_cls, conf = get_patch_label(lulc_arr, r, c)

            if eurosat_cls is None:
                rejected += 1
                continue

            new_entry = {**entry, "label": eurosat_cls, "confidence": round(conf, 3)}
            labeled.append(new_entry)
            class_counts[eurosat_cls] += 1

    with open(OUTPUT_PATH, "w") as f:
        json.dump(labeled, f, indent=2, ensure_ascii=False)

    total = len(manifest)
    print(f"\n=== Wyniki labelowania Esri LULC ===")
    print(f"Łącznie:     {total}")
    print(f"Zaetykietowane: {len(labeled)} ({len(labeled)/total:.1%})")
    print(f"Odrzucone:   {rejected} ({rejected/total:.1%})")
    print(f"\nRozkład klas:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        if cnt > 0:
            bar = "█" * (cnt // 10)
            print(f"  {cls:25s}: {cnt:4d} {bar}")
    print(f"\nZapisano: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
