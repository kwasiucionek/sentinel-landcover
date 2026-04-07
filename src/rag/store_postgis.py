"""
PostGIS storage dla wyników analiz satelitarnych.
Zastępuje SQLite — dodaje prawdziwe wsparcie geometrii przestrzennej.
"""
import json
import os
from dataclasses import dataclass
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection


@dataclass
class AnalysisRecord:
    tile_name:     str
    analyzed_at:   str
    class_stats:   dict
    mask_path:     str
    tile_path:     str
    location:      Optional[str]  = None
    notes:         Optional[str]  = None
    ndvi_mean:     Optional[float] = None
    shannon_idx:   Optional[float] = None
    dominant_cls:  Optional[str]   = None
    dominant_conf: Optional[float] = None
    patch_count:   Optional[int]   = None
    # PostGIS: bbox jako WKT polygon EPSG:4326
    # np. "POLYGON((16.978 51.095, 17.045 51.095, ...))"
    bbox_wkt:      Optional[str]   = None
    id:            Optional[int]   = None


def get_connection() -> PgConnection:
    return psycopg2.connect(
        host=os.getenv("POSTGIS_HOST", "localhost"),
        port=int(os.getenv("POSTGIS_PORT", 5432)),
        dbname=os.getenv("POSTGIS_DB", "sentinel_landcover"),
        user=os.getenv("POSTGIS_USER", "sentinel"),
        password=os.getenv("POSTGIS_PASSWORD", "sentinel_pass"),
    )


def init_db():
    """Tworzy tabelę z kolumną geometry jeśli nie istnieje."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id            SERIAL PRIMARY KEY,
                    tile_name     TEXT NOT NULL,
                    analyzed_at   TIMESTAMPTZ NOT NULL,
                    class_stats   JSONB NOT NULL,
                    mask_path     TEXT NOT NULL,
                    tile_path     TEXT NOT NULL,
                    location      TEXT,
                    notes         TEXT,
                    ndvi_mean     REAL,
                    shannon_idx   REAL,
                    dominant_cls  TEXT,
                    dominant_conf REAL,
                    patch_count   INTEGER,
                    bbox          geometry(Polygon, 4326)
                );
            """)
            # Indeks przestrzenny — przyspiesza zapytania ST_DWithin itp.
            cur.execute("""
                CREATE INDEX IF NOT EXISTS analyses_bbox_idx
                ON analyses USING GIST(bbox);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS analyses_location_idx
                ON analyses(location);
            """)
        conn.commit()


def save_analysis(record: AnalysisRecord) -> int:
    """Zapisuje analizę do PostGIS, zwraca id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            bbox_expr = (
                f"ST_GeomFromText('{record.bbox_wkt}', 4326)"
                if record.bbox_wkt else "NULL"
            )
            cur.execute(f"""
                INSERT INTO analyses
                (tile_name, analyzed_at, class_stats, mask_path, tile_path,
                 location, notes, ndvi_mean, shannon_idx,
                 dominant_cls, dominant_conf, patch_count, bbox)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        {bbox_expr})
                RETURNING id
            """, (
                record.tile_name,
                record.analyzed_at,
                json.dumps(record.class_stats),
                record.mask_path,
                record.tile_path,
                record.location,
                record.notes,
                record.ndvi_mean,
                record.shannon_idx,
                record.dominant_cls,
                record.dominant_conf,
                record.patch_count,
            ))
            row_id = cur.fetchone()[0]
        conn.commit()
    return row_id


def get_all_analyses() -> list[AnalysisRecord]:
    """Zwraca wszystkie analizy posortowane od najnowszej."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, tile_name, analyzed_at::text, class_stats,
                       mask_path, tile_path, location, notes,
                       ndvi_mean, shannon_idx, dominant_cls, dominant_conf,
                       patch_count,
                       ST_AsText(bbox) AS bbox_wkt
                FROM analyses
                ORDER BY analyzed_at DESC
            """)
            return [_row_to_record(r) for r in cur.fetchall()]


def get_analyses_by_location(location: str) -> list[AnalysisRecord]:
    """Zwraca analizy dla danej lokalizacji."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, tile_name, analyzed_at::text, class_stats,
                       mask_path, tile_path, location, notes,
                       ndvi_mean, shannon_idx, dominant_cls, dominant_conf,
                       patch_count, ST_AsText(bbox) AS bbox_wkt
                FROM analyses
                WHERE location = %s
                ORDER BY analyzed_at DESC
            """, (location,))
            return [_row_to_record(r) for r in cur.fetchall()]


def get_analyses_near_point(
    lon: float, lat: float, radius_m: float = 5000
) -> list[AnalysisRecord]:
    """
    Zwraca analizy w promieniu radius_m metrów od punktu (lon, lat).
    Przykład: wszystkie analizy w promieniu 5km od centrum Wrocławia.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, tile_name, analyzed_at::text, class_stats,
                       mask_path, tile_path, location, notes,
                       ndvi_mean, shannon_idx, dominant_cls, dominant_conf,
                       patch_count, ST_AsText(bbox) AS bbox_wkt,
                       ST_Distance(
                           ST_Centroid(bbox)::geography,
                           ST_MakePoint(%s, %s)::geography
                       ) AS dist_m
                FROM analyses
                WHERE bbox IS NOT NULL
                  AND ST_DWithin(
                      ST_Centroid(bbox)::geography,
                      ST_MakePoint(%s, %s)::geography,
                      %s
                  )
                ORDER BY dist_m ASC
            """, (lon, lat, lon, lat, radius_m))
            return [_row_to_record(r) for r in cur.fetchall()]


def get_analyses_intersecting(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
) -> list[AnalysisRecord]:
    """
    Zwraca analizy których bbox przecina podany obszar.
    Użyteczne do: "pokaż analizy dla obszaru widocznego na mapie".
    """
    bbox_wkt = (
        f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
        f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
    )
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, tile_name, analyzed_at::text, class_stats,
                       mask_path, tile_path, location, notes,
                       ndvi_mean, shannon_idx, dominant_cls, dominant_conf,
                       patch_count, ST_AsText(bbox) AS bbox_wkt
                FROM analyses
                WHERE bbox IS NOT NULL
                  AND ST_Intersects(bbox,
                      ST_GeomFromText(%s, 4326))
                ORDER BY analyzed_at DESC
            """, (bbox_wkt,))
            return [_row_to_record(r) for r in cur.fetchall()]


def _row_to_record(row: dict) -> AnalysisRecord:
    return AnalysisRecord(
        id=row["id"],
        tile_name=row["tile_name"],
        analyzed_at=str(row["analyzed_at"]),
        class_stats=row["class_stats"] if isinstance(row["class_stats"], dict)
                    else json.loads(row["class_stats"]),
        mask_path=row["mask_path"],
        tile_path=row["tile_path"],
        location=row["location"],
        notes=row["notes"],
        ndvi_mean=row["ndvi_mean"],
        shannon_idx=row["shannon_idx"],
        dominant_cls=row["dominant_cls"],
        dominant_conf=row["dominant_conf"],
        patch_count=row["patch_count"],
        bbox_wkt=row.get("bbox_wkt"),
    )
