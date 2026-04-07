"""
Klient Copernicus Data Space Ecosystem (CDSE).
Wyszukuje i pobiera sceny Sentinel-2 dla zadanego obszaru.

Dokumentacja: https://documentation.dataspace.copernicus.eu/
"""

import io
import os
from datetime import datetime, timedelta
from typing import Optional

import requests
from PIL import Image

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
CDSE_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1"
CDSE_WMS_URL = "https://sh.dataspace.copernicus.eu/ogc/wms"

WROCLAW_BBOXES = {
    "Centrum": (16.978, 51.095, 17.045, 51.130),
    "Psie Pole": (17.040, 51.130, 17.110, 51.165),
    "Fabryczna": (16.880, 51.075, 16.950, 51.110),
    "Krzyki": (17.010, 51.065, 17.080, 51.100),
    "Odra (centrum)": (16.990, 51.105, 17.060, 51.130),
}


class CDSEClient:
    def __init__(self):
        self.client_id = os.getenv("CDSE_CLIENT_ID", "")
        self.client_secret = os.getenv("CDSE_CLIENT_SECRET", "")
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    def get_token(self) -> str:
        if self._token and self._token_expiry and datetime.now() < self._token_expiry:
            return self._token
        resp = requests.post(
            CDSE_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = datetime.now() + timedelta(seconds=data["expires_in"] - 30)
        return self._token

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_token()}"}

    def search_scenes(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        date_from: str = None,
        date_to: str = None,
        max_cloud: int = 20,
        max_results: int = 5,
    ) -> list[dict]:
        if not date_to:
            date_to = datetime.now().strftime("%Y-%m-%d")
        if not date_from:
            date_from = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        bbox_wkt = (
            f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
            f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
        )
        params = {
            "$filter": (
                f"Collection/Name eq 'SENTINEL-2' "
                f"and OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}') "
                f"and ContentDate/Start gt {date_from}T00:00:00.000Z "
                f"and ContentDate/Start lt {date_to}T23:59:59.000Z "
                f"and Attributes/OData.CSC.DoubleAttribute/any("
                f"att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {max_cloud}.00)"
            ),
            "$orderby": "ContentDate/Start desc",
            "$top": max_results,
        }
        resp = requests.get(
            f"{CDSE_ODATA_URL}/Products",
            params=params,
            headers=self._auth_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("value", [])

    def fetch_wms_preview(
        self,
        instance_id: str,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        layer: str = "TRUE-COLOR",
        size: int = 512,
        date: str = None,
    ) -> Optional[Image.Image]:
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": layer,
            "BBOX": f"{lat_min},{lon_min},{lat_max},{lon_max}",
            "CRS": "EPSG:4326",
            "WIDTH": size,
            "HEIGHT": size,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
            "TIME": date,
            "MAXCC": "20",
            "STYLES": "",
        }
        try:
            r = requests.get(
                f"{CDSE_WMS_URL}/{instance_id}",
                params=params,
                headers=self._auth_headers(),
                timeout=20,
            )
            r.raise_for_status()
            if "xml" in r.headers.get("Content-Type", "").lower():
                return None
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            return None

    def list_available_layers(self, instance_id: str) -> list[str]:
        params = {"SERVICE": "WMS", "REQUEST": "GetCapabilities"}
        r = requests.get(
            f"{CDSE_WMS_URL}/{instance_id}",
            params=params,
            headers=self._auth_headers(),
            timeout=15,
        )
        import xml.etree.ElementTree as ET

        root = ET.fromstring(r.content)
        ns = {"wms": "http://www.opengis.net/wms"}
        return [el.text for el in root.findall(".//wms:Layer/wms:Name", ns) if el.text]


def test_wms(instance_id: str, client: CDSEClient):
    print(f"\nTest WMS Sentinel Hub (instance: {instance_id[:8]}...)...")
    lon_min, lat_min, lon_max, lat_max = WROCLAW_BBOXES["Centrum"]
    os.makedirs("data/samples", exist_ok=True)
    for layer in ["TRUE-COLOR", "FALSE-COLOR", "NDVI"]:
        img = client.fetch_wms_preview(
            instance_id,
            lon_min,
            lat_min,
            lon_max,
            lat_max,
            layer=layer,
            size=512,
            date="2026-03-22",
        )
        if img:
            path = f"data/samples/cdse_{layer.lower().replace('-', '_')}.png"
            img.save(path)
            print(f"  ✓ {layer} → {path}")
        else:
            print(f"  ✗ {layer} — błąd")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    client = CDSEClient()

    print("Test autoryzacji CDSE...")
    try:
        token = client.get_token()
        print(f"  ✓ Token OK ({token[:20]}...)")
    except Exception as e:
        print(f"  ✗ Błąd: {e}")
        exit(1)

    print("\nWyszukiwanie scen Sentinel-2 dla Wrocławia (ostatnie 60 dni)...")
    lon_min, lat_min, lon_max, lat_max = WROCLAW_BBOXES["Centrum"]
    scenes = client.search_scenes(lon_min, lat_min, lon_max, lat_max, max_cloud=30)
    if not scenes:
        print("  ✗ Brak scen")
    else:
        print(f"  ✓ Znaleziono {len(scenes)} scen:")
        for s in scenes:
            name = s.get("Name", "?")
            date = s.get("ContentDate", {}).get("Start", "?")[:10]
            size = s.get("ContentLength", 0) / 1e9
            print(f"    [{date}] {name[:60]} ({size:.1f} GB)")

    instance_id = os.getenv("CDSE_INSTANCE_ID", "")
    if instance_id:
        test_wms(instance_id, client)
    else:
        print("\n✗ Brak CDSE_INSTANCE_ID w .env")
