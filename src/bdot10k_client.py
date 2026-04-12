"""Klient WMTS/WFS dla BDOT10k i EGiB — GUGiK Geoportal."""
import io
import logging
import time
import requests
from PIL import Image
from pyproj import Transformer

logger = logging.getLogger(__name__)

BDOT10K_WMTS = "https://mapy.geoportal.gov.pl/wss/service/WMTS/guest/wmts/BDOT10k-BDOO"
EGIB_WFS     = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/EGIB/WFS/UslugaZbiorcza"

_TOP_N   = 850000.0
_TOP_E   = 100000.0
_SCALE_8 = 23623.559151785714
_TILE_M  = _SCALE_8 * 0.00028 * 512  # ≈ 3386.7 m

_t = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)


def _get(url, params=None, timeout=15, retries=3):
    for i in range(retries):
        try:
            return requests.get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            if i < retries - 1:
                time.sleep(2)
            else:
                raise


def _to_tile(lon: float, lat: float) -> tuple[int, int]:
    e, n = _t.transform(lon, lat)
    return int((_TOP_N - n) / _TILE_M), int((e - _TOP_E) / _TILE_M)


def fetch_bdot10k_area(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    zoom: int = 8,
) -> Image.Image | None:
    row_nw, col_nw = _to_tile(lon_min, lat_max)  # NW
    row_se, col_se = _to_tile(lon_max, lat_min)  # SE

    rows = min(max(1, row_se - row_nw + 1), 3)
    cols = min(max(1, col_se - col_nw + 1), 3)

    grid = Image.new("RGBA", (512 * cols, 512 * rows))
    ok = 0
    for dr in range(rows):
        for dc in range(cols):
            r = _get(BDOT10K_WMTS, params={
                "Service": "WMTS", "Request": "GetTile", "Version": "1.0.0",
                "Layer": "BDOT10k", "Style": "default",
                "TileMatrixSet": "EPSG:2180", "TileMatrix": f"EPSG:2180:{zoom}",
                "TileRow": row_nw + dr, "TileCol": col_nw + dc,
                "Format": "image/png",
            })
            if r.ok and "image" in r.headers.get("Content-Type", ""):
                try:
                    t = Image.open(io.BytesIO(r.content)).convert("RGBA")
                    grid.paste(t, (dc * 512, dr * 512))
                    ok += 1
                except Exception as ex:
                    logger.warning("Kafel (%d,%d) błąd: %s", row_nw+dr, col_nw+dc, ex)
    return grid.resize((512, 512), Image.LANCZOS) if ok else None


def fetch_egib_parcels(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    max_count: int = 200,
) -> list[dict]:
    import xml.etree.ElementTree as ET
    try:
        r = _get(EGIB_WFS, params={
            "SERVICE": "WFS", "REQUEST": "GetFeature", "VERSION": "2.0.0",
            "TYPENAMES": "ms:dzialki",
            "BBOX": f"{lat_min},{lon_min},{lat_max},{lon_max},EPSG:4326",
            "COUNT": max_count,
            "OUTPUTFORMAT": "application/gml+xml; version=3.2",
        }, timeout=30)
        if not r.ok:
            return []
        ns = {"ms": "http://mapserver.gis.umn.edu/mapserver"}
        root = ET.fromstring(r.text)
        return [
            {"id": m.get("{http://www.opengis.net/gml/3.2}id", "")}
            for m in root.findall(".//ms:dzialki", ns)
        ]
    except Exception as e:
        logger.error("EGiB błąd: %s", e)
        return []


def detect_bdot_discrepancies(
    sentinel_classes: dict[str, float],
    bdot_img: Image.Image | None,
) -> list[str]:
    if bdot_img is None or not sentinel_classes:
        return ["⚠️ Brak danych BDOT10k"]
    dominant = max(sentinel_classes.items(), key=lambda x: x[1])[0]
    pixels = list(bdot_img.convert("RGB").getdata())
    total  = len(pixels)
    green  = sum(1 for r,g,b in pixels if g>140 and g>r+20 and g>b+20) / total
    blue   = sum(1 for r,g,b in pixels if b>140 and b>r+20 and b>g+20) / total
    brown  = sum(1 for r,g,b in pixels if r>150 and g>80 and b<100 and r>g+30) / total

    if dominant == "Forest" and green < 0.15:
        return ["⚠️ Sentinel: Las — BDOT10k nie pokazuje roślinności"]
    if dominant == "Residential" and brown < 0.05:
        return ["⚠️ Sentinel: Zabudowa — BDOT10k nie potwierdza"]
    if dominant in ("River", "SeaLake") and blue < 0.05:
        return ["⚠️ Sentinel: Woda — BDOT10k nie pokazuje wody"]
    if dominant == "AnnualCrop" and brown > 0.3:
        return ["⚠️ Pole uprawne — BDOT10k pokazuje zabudowę (zmiana użytkowania?)"]
    return [f"✅ Klasa '{dominant}' zgodna z BDOT10k"]


# ── MPZP — Miejscowe Plany Zagospodarowania Przestrzennego ──────────────────

MPZP_WFS = "http://gis1.um.wroc.pl/arcgis/services/ogc/OGC_mpzp/MapServer/WFSServer"

# Mapowanie klasyfikacji MPZP → EuroSAT
MPZP_TO_EUROSAT = {
    "mieszkaniowe":       "Residential",
    "usługi":             "Residential",
    "przemysłowe":        "Industrial",
    "zieleń":             "HerbaceousVegetation",
    "las":                "Forest",
    "woda":               "SeaLake",
    "rolne":              "AnnualCrop",
    "komunikacja":        "Highway",
    "sportu i rekreacji": "HerbaceousVegetation",
}


def fetch_mpzp_przeznaczenie(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    max_features: int = 50,
) -> list[dict]:
    """Pobierz przeznaczenie terenu MPZP dla bbox WGS84."""
    from xml.etree import ElementTree as ET
    try:
        r = _get(MPZP_WFS, params={
            "service":     "wfs",
            "request":     "GetFeature",
            "version":     "1.1.0",
            "typeName":    "OGC_mpzp:przeznaczenie_terenu_-_uproszczona_klasyfikacja",
            "bbox":        f"{lat_min},{lon_min},{lat_max},{lon_max},EPSG:4326",
            "maxFeatures": max_features,
        }, timeout=20)
        if not r.ok:
            return []

        root = ET.fromstring(r.content)
        ns = {"OGC_mpzp": "http://gis1.um.wroc.pl/arcgis/services/ogc/OGC_mpzp/MapServer/WFSServer"}
        results = []
        for f in root.findall(".//OGC_mpzp:przeznaczenie_terenu_-_uproszczona_klasyfikacja", ns):
            klasyfikacja = f.find("OGC_mpzp:uproszczona_klasyfikacja_przeznaczenia", ns)
            symbol       = f.find("OGC_mpzp:symbol_literowy", ns)
            opis         = f.find("OGC_mpzp:opis_w_legendzie", ns)
            results.append({
                "klasyfikacja": klasyfikacja.text if klasyfikacja is not None else "",
                "symbol":       symbol.text if symbol is not None else "",
                "opis":         opis.text if opis is not None else "",
            })
        return results
    except Exception as e:
        logger.error("MPZP WFS błąd: %s", e)
        return []


def detect_mpzp_discrepancies(
    sentinel_classes: dict[str, float],
    mpzp_features: list[dict],
) -> list[str]:
    """Porównaj klasyfikację Sentinel z przeznaczeniem MPZP."""
    if not mpzp_features or not sentinel_classes:
        return ["⚠️ Brak danych MPZP dla tego obszaru"]

    dominant = max(sentinel_classes.items(), key=lambda x: x[1])[0]

    # Zlicz przeznaczenia MPZP
    from collections import Counter
    klasy = Counter(f["klasyfikacja"].lower() for f in mpzp_features if f["klasyfikacja"])
    # Pomiń komunikację jako dominującą (drogi są wszędzie)
    klasy_bez_kom = {k: v for k, v in klasy.items() if "komunikacja" not in k}
    dominant_mpzp = max(klasy_bez_kom, key=klasy_bez_kom.get) if klasy_bez_kom else (klasy.most_common(1)[0][0] if klasy else "")
    expected = MPZP_TO_EUROSAT.get(dominant_mpzp, "")

    alerts = []
    if expected and expected != dominant:
        alerts.append(
            f"⚠️ Sentinel: **{dominant}** — MPZP wskazuje: **{dominant_mpzp}** "
            f"({klasy.most_common(1)[0][1]} działek)"
        )
    elif dominant_mpzp:
        alerts.append(
            f"✅ Sentinel: **{dominant}** zgodne z MPZP: **{dominant_mpzp}**"
        )

    # Pokaż top 3 przeznaczenia
    for kl, cnt in klasy.most_common(3):
        alerts.append(f"  • {kl}: {cnt} terenów")

    return alerts
