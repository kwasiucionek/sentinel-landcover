"""
Klient WMS dla danych GUGiK / Geoportal.gov.pl.

Pobiera:
- Ortofotomapę Sentinel-2 (ORTO) jako wejście do klasyfikacji
- Warstwę referencyjną "Klasyfikacja pokrycia terenu 2025" (POLSA/GUGiK)

Dokumentacja WMS: https://www.geoportal.gov.pl/pl/usluga/uslugi-przegladania-wms-i-wmts/
"""
import io
import math
import os
from dataclasses import dataclass
from typing import Optional

import requests
from PIL import Image

# ── Endpointy WMS GUGiK ───────────────────────────────────────────────────────
WMS_ORTO = (
    "https://mapy.geoportal.gov.pl/wss/service/img/guest/ORTO/MapServer/WMSServer"
)
WMS_KPTO_2025 = os.getenv(
    "WMS_KPTO_2025",
    # URL zostanie opublikowany przez GUGiK — śledź:
    # https://www.geoportal.gov.pl/pl/usluga/uslugi-przegladania-wms-i-wmts/
    "https://mapy.geoportal.gov.pl/wss/service/POLSA/guest/services/KPTO_2025/MapServer/WMSServer",
)

# Klasy S2GLC PL2020 → mapowanie na nasze klasy EuroSAT
# https://s2glc.cbk.waw.pl/
S2GLC_TO_EUROSAT = {
    "Clouds":                  None,
    "Artificial surfaces":     "Residential",
    "Cultivated areas":        "AnnualCrop",
    "Vineyards":               "PermanentCrop",
    "Broadleaf trees":         "Forest",
    "Coniferous trees":        "Forest",
    "Herbaceous":              "HerbaceousVegetation",
    "Moors & heathland":       "HerbaceousVegetation",
    "Sclerophyllous veg.":     "HerbaceousVegetation",
    "Marshes":                 "HerbaceousVegetation",
    "Peatbogs":                "HerbaceousVegetation",
    "Natural material surfaces": "Highway",
    "Permanent snow":          None,
    "Water bodies":            "SeaLake",
    "Wetlands":                "River",
}


@dataclass
class BBox:
    """Bounding box w układzie EPSG:4326 (WGS84)."""
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

    def to_wms_string(self) -> str:
        # WMS 1.3.0 dla EPSG:4326 kolejność: lat,lon
        return f"{self.lat_min},{self.lon_min},{self.lat_max},{self.lon_max}"

    def to_epsg3857(self) -> "BBox":
        """Przelicza do Web Mercator (EPSG:3857) dla WMS 1.1.1."""
        def to_mercator(lon, lat):
            x = lon * 20037508.34 / 180.0
            y = math.log(math.tan((90 + lat) * math.pi / 360.0)) * 20037508.34 / math.pi
            return x, y
        x_min, y_min = to_mercator(self.lon_min, self.lat_min)
        x_max, y_max = to_mercator(self.lon_max, self.lat_max)
        return BBox(x_min, y_min, x_max, y_max)


# Predefiniowane obszary Wrocławia
WROCLAW_BBOXES = {
    "Centrum":        BBox(16.978, 51.095, 17.045, 51.130),
    "Psie Pole":      BBox(17.040, 51.130, 17.110, 51.165),
    "Fabryczna":      BBox(16.880, 51.075, 16.950, 51.110),
    "Krzyki":         BBox(17.010, 51.065, 17.080, 51.100),
    "Odra (centrum)": BBox(16.990, 51.105, 17.060, 51.130),
}


def fetch_wms_image(
    wms_url: str,
    layers: str,
    bbox: BBox,
    width: int = 512,
    height: int = 512,
    srs: str = "EPSG:4326",
    version: str = "1.3.0",
    image_format: str = "image/png",
    timeout: int = 15,
) -> Optional[Image.Image]:
    """
    Wykonuje zapytanie WMS GetMap i zwraca PIL Image.
    Zwraca None jeśli serwis niedostępny.
    """
    crs_param = "CRS" if version == "1.3.0" else "SRS"
    if version == "1.3.0" and srs == "EPSG:4326":
        bbox_str = bbox.to_wms_string()
    else:
        b = bbox.to_epsg3857()
        bbox_str = f"{b.lon_min},{b.lat_min},{b.lon_max},{b.lat_max}"
        srs = "EPSG:3857"

    params = {
        "SERVICE": "WMS",
        "VERSION": version,
        "REQUEST": "GetMap",
        "LAYERS": layers,
        "BBOX": bbox_str,
        crs_param: srs,
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": image_format,
        "TRANSPARENT": "TRUE",
        "STYLES": "",
    }
    try:
        resp = requests.get(wms_url, params=params, timeout=timeout)
        resp.raise_for_status()
        if "xml" in resp.headers.get("Content-Type", ""):
            return None  # WMS zwrócił błąd XML
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def fetch_orto(bbox: BBox, size: int = 512) -> Optional[Image.Image]:
    """Pobiera ortofotomapę z GUGiK ORTO WMS."""
    return fetch_wms_image(
        wms_url=WMS_ORTO,
        layers="Raster",
        bbox=bbox,
        width=size,
        height=size,
    )


def fetch_kpto_2025(bbox: BBox, size: int = 512) -> Optional[Image.Image]:
    """
    Pobiera warstwę klasyfikacji pokrycia terenu 2025 (POLSA/GUGiK).
    Zwraca None jeśli warstwa jeszcze niedostępna pod tym URL.
    """
    return fetch_wms_image(
        wms_url=WMS_KPTO_2025,
        layers="0",
        bbox=bbox,
        width=size,
        height=size,
    )


def side_by_side(
    img_a: Image.Image,
    img_b: Image.Image,
    label_a: str = "Model",
    label_b: str = "GUGiK 2025",
) -> Image.Image:
    """Skleja dwa obrazy poziomo z etykietami."""
    from PIL import ImageDraw
    w, h = img_a.size
    result = Image.new("RGB", (w * 2 + 10, h + 30), (240, 240, 240))
    result.paste(img_a, (0, 30))
    result.paste(img_b, (w + 10, 30))
    draw = ImageDraw.Draw(result)
    draw.text((w // 2 - 30, 5),  label_a, fill=(30, 30, 30))
    draw.text((w + w // 2 - 30, 5), label_b, fill=(30, 30, 30))
    return result


if __name__ == "__main__":
    print("Test pobierania ortofotomapy GUGiK...")
    for name, bbox in WROCLAW_BBOXES.items():
        img = fetch_orto(bbox, size=256)
        if img:
            path = f"data/samples/wroclaw_{name.replace(' ', '_')}.png"
            os.makedirs("data/samples", exist_ok=True)
            img.save(path)
            print(f"  ✓ {name} → {path}")
        else:
            print(f"  ✗ {name} — błąd pobierania")

    print("\nTest warstwy KPTO 2025...")
    img = fetch_kpto_2025(WROCLAW_BBOXES["Centrum"])
    if img:
        print("  ✓ Warstwa KPTO 2025 dostępna")
    else:
        print("  ✗ Warstwa KPTO 2025 niedostępna (URL do aktualizacji gdy GUGiK opublikuje)")
