"""
Klient Esri Sentinel-2 10m Land Cover (Impact Observatory / Microsoft / Esri).
Pobiera referencyjną klasyfikację LULC z ArcGIS Living Atlas.

Serwis: https://ic.imagery1.arcgis.com/arcgis/rest/services/Sentinel2_10m_LandCover/ImageServer
Dane: 2017-2024, 9 klas, 10m rozdzielczość, cały świat
"""
import io
import json
from typing import Optional

import requests
from PIL import Image

LULC_SERVICE = (
    "https://ic.imagery1.arcgis.com/arcgis/rest/services"
    "/Sentinel2_10m_LandCover/ImageServer"
)

# 9 klas Esri LULC → kolory i mapowanie na nasze klasy EuroSAT
ESRI_CLASSES = {
    1:  {"name": "Water",              "color": (0, 105, 148),   "eurosat": "SeaLake"},
    2:  {"name": "Trees",              "color": (34, 139, 34),   "eurosat": "Forest"},
    3:  {"name": "Flooded vegetation", "color": (0, 180, 130),   "eurosat": "HerbaceousVegetation"},
    4:  {"name": "Crops",              "color": (255, 211, 0),   "eurosat": "AnnualCrop"},
    5:  {"name": "Built area",         "color": (200, 50, 50),   "eurosat": "Residential"},
    6:  {"name": "Bare ground",        "color": (165, 110, 65),  "eurosat": "Highway"},
    7:  {"name": "Snow/Ice",           "color": (220, 240, 255), "eurosat": None},
    8:  {"name": "Clouds",             "color": (200, 200, 200), "eurosat": None},
    9:  {"name": "Rangeland",          "color": (144, 200, 100), "eurosat": "HerbaceousVegetation"},
    10: {"name": "Rangeland",          "color": (144, 200, 100), "eurosat": "Pasture"},
    11: {"name": "Rangeland",          "color": (144, 200, 100), "eurosat": "Pasture"},
}


def fetch_esri_lulc(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    year: int = 2023,
    size: int = 512,
    timeout: int = 20,
) -> Optional[Image.Image]:
    """
    Pobiera klasyfikację LULC z Esri ImageServer jako kolorowy PNG.
    Zwraca None jeśli serwis niedostępny lub wymaga autoryzacji.
    """
    mosaic_rule = json.dumps({
        "mosaicMethod": "esriMosaicAttribute",
        "sortField": "Year",
        "sortValue": str(year),
        "ascending": False,
        "mosaicOperation": "MT_FIRST",
        "where": f"Year={year}",
    })

    rendering_rule = json.dumps({
        "rasterFunction": "Cartographic Renderer for Visualization and Analysis"
    })

    params = {
        "bbox":          f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "bboxSR":        "4326",
        "size":          f"{size},{size}",
        "imageSR":       "4326",
        "format":        "png",
        "pixelType":     "U8",
        "mosaicRule":    mosaic_rule,
        "renderingRule": rendering_rule,
        "f":             "image",
    }

    try:
        r = requests.get(
            f"{LULC_SERVICE}/exportImage",
            params=params,
            timeout=timeout,
            headers={"User-Agent": "sentinel-landcover-portfolio/1.0"},
        )
        if r.status_code == 499 or r.status_code == 401:
            return None  # wymaga auth
        r.raise_for_status()
        if "json" in r.headers.get("Content-Type", "").lower():
            return None  # błąd serwisu
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def fetch_esri_lulc_raw(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    year: int = 2023,
    size: int = 512,
    timeout: int = 20,
) -> Optional[Image.Image]:
    """
    Pobiera surowe wartości klasowe (bez renderowania kolorami Esri).
    Zwraca obraz w trybie 'L' (grayscale) gdzie wartości = klasy 1-11.
    """
    mosaic_rule = json.dumps({
        "mosaicMethod": "esriMosaicAttribute",
        "sortField": "Year",
        "sortValue": str(year),
        "ascending": False,
        "mosaicOperation": "MT_FIRST",
        "where": f"Year={year}",
    })

    params = {
        "bbox":       f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "bboxSR":     "4326",
        "size":       f"{size},{size}",
        "imageSR":    "4326",
        "format":     "png",
        "pixelType":  "U8",
        "mosaicRule": mosaic_rule,
        "f":          "image",
    }

    try:
        r = requests.get(
            f"{LULC_SERVICE}/exportImage",
            params=params,
            timeout=timeout,
            headers={"User-Agent": "sentinel-landcover-portfolio/1.0"},
        )
        if r.status_code in (401, 499):
            return None
        r.raise_for_status()
        if "json" in r.headers.get("Content-Type", "").lower():
            return None
        return Image.open(io.BytesIO(r.content)).convert("L")
    except Exception:
        return None


def lulc_to_stats(raw_img: Image.Image) -> dict:
    """
    Przelicza surowy obraz klas na rozkład procentowy klas Esri.
    Zwraca dict: {nazwa_klasy: udział}
    """
    import numpy as np
    arr   = np.array(raw_img)
    total = arr.size
    stats = {}
    for class_id, info in ESRI_CLASSES.items():
        count = (arr == class_id).sum()
        if count > 0:
            stats[info["name"]] = count / total
    return stats


def lulc_to_eurosat_stats(raw_img: Image.Image) -> dict:
    """
    Mapuje klasy Esri na klasy EuroSAT — do porównania z naszym modelem.
    """
    import numpy as np
    from src.dataset import CLASSES

    arr   = np.array(raw_img)
    total = arr.size
    counts = {cls: 0 for cls in CLASSES}

    for class_id, info in ESRI_CLASSES.items():
        eurosat_cls = info.get("eurosat")
        if eurosat_cls and eurosat_cls in counts:
            counts[eurosat_cls] += int((arr == class_id).sum())

    return {cls: cnt / total for cls, cnt in counts.items()}


def colorize_lulc(raw_img: Image.Image) -> Image.Image:
    """
    Koloruje surowy obraz klas kolorami Esri LULC.
    """
    import numpy as np
    arr     = np.array(raw_img)
    colored = np.zeros((*arr.shape, 3), dtype=np.uint8)
    colored[:] = (100, 100, 100)  # domyślny kolor dla nieznanych klas

    for class_id, info in ESRI_CLASSES.items():
        mask = arr == class_id
        colored[mask] = info["color"]

    return Image.fromarray(colored)


if __name__ == "__main__":
    import os
    os.makedirs("data/samples", exist_ok=True)

    # Wrocław Centrum
    lon_min, lat_min, lon_max, lat_max = 16.978, 51.095, 17.045, 51.130

    print("Test Esri LULC serwisu...")
    for year in [2022, 2023]:
        print(f"  Pobieranie {year}...", end=" ")
        raw = fetch_esri_lulc_raw(lon_min, lat_min, lon_max, lat_max, year=year)
        if raw is None:
            print("BRAK (wymaga auth lub serwis niedostępny)")
            continue
        colored = colorize_lulc(raw)
        colored.save(f"data/samples/esri_lulc_{year}.png")
        stats = lulc_to_stats(raw)
        print(f"OK → data/samples/esri_lulc_{year}.png")
        print(f"    Klasy: {', '.join(f'{k}: {v:.1%}' for k,v in sorted(stats.items(), key=lambda x:x[1], reverse=True) if v > 0)}")
