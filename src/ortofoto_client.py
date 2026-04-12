"""Klient WMTS ortofoto 2025 — Urząd Miasta Wrocław."""
import io
import logging
import time
import requests
from PIL import Image
from pyproj import Transformer


def _get_with_retry(url, timeout=15, retries=3, delay=2):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            return r
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < retries - 1:
                logging.warning("Retry %d/%d: %s", attempt+1, retries, e)
                time.sleep(delay)
            else:
                raise
    return None

logger = logging.getLogger(__name__)

ORTOFOTO_URL = (
    "http://gis1.um.wroc.pl/arcgis/rest/services/ogc/OGC_ortofoto_2025"
    "/MapServer/WMTS/tile/1.0.0/ogc_OGC_ortofoto_2025"
    "/default/default028mm/{zoom}/{row}/{col}"
)

# EPSG:2177 — CS2000/17 (strefa Wrocław)
# TopLeftCorner: (N=10001300, E=877300) — odwrócone osie!
_TOP_N = 10001300.0
_TOP_E = 877300.0

_transformer = Transformer.from_crs("EPSG:4326", "EPSG:2177", always_xy=True)

# ScaleDenominator per zoom level
_SCALES = {
    0: 472471.18303754483,
    1: 236235.59151877242,
    2: 165364.9140631407,
    3: 94494.23660750895,
    4: 47247.118303754476,
    5: 23623.559151877,
    6: 11811.779575938621,
}


def _tile_m(zoom: int) -> float:
    return _SCALES[zoom] * 0.00028 * 256


def _latlon_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    e, n = _transformer.transform(lon, lat)
    tm = _tile_m(zoom)
    row = int((_TOP_N - n) / tm)
    col = int((e - _TOP_E) / tm)
    return row, col


def fetch_ortofoto_tile(lon: float, lat: float, zoom: int = 5) -> Image.Image | None:
    """Pobierz kafel ortofoto 2025 dla punktu (lon, lat) WGS84."""
    row, col = _latlon_to_tile(lon, lat, zoom)
    url = ORTOFOTO_URL.format(zoom=zoom, row=row, col=col)
    try:
        r = _get_with_retry(url, timeout=15)
        if r.ok and len(r.content) > 1000:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        logger.warning("Ortofoto: pusty kafel row=%d col=%d", row, col)
        return None
    except Exception as e:
        logger.error("Ortofoto błąd: %s", e)
        return None


def fetch_ortofoto_area(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    zoom: int = 5,
) -> Image.Image | None:
    """Pobierz i złóż kafle ortofoto dla bbox WGS84."""
    row_min, col_min = _latlon_to_tile(lon_min, lat_max, zoom)  # NW
    row_max, col_max = _latlon_to_tile(lon_max, lat_min, zoom)  # SE

    cols = max(1, col_max - col_min + 1)
    rows = max(1, row_max - row_min + 1)
    cols = min(cols, 4)
    rows = min(rows, 4)

    grid = Image.new("RGB", (256 * cols, 256 * rows))
    ok = 0
    for dr in range(rows):
        for dc in range(cols):
            url = ORTOFOTO_URL.format(
                zoom=zoom, row=row_min + dr, col=col_min + dc
            )
            try:
                r = _get_with_retry(url, timeout=15)
                if r.ok and len(r.content) > 1000:
                    t = Image.open(io.BytesIO(r.content)).convert("RGB")
                    grid.paste(t, (dc * 256, dr * 256))
                    ok += 1
            except Exception:
                pass

    if ok == 0:
        return None
    return grid.resize((512, 512), Image.LANCZOS)
