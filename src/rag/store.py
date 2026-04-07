import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

SQLITE_PATH = "data/analyses/analyses.db"
QDRANT_COLLECTION = "analyses"


@dataclass
class AnalysisRecord:
    tile_name:    str
    analyzed_at:  str
    class_stats:  dict        # {"Forest": 0.34, ...}
    mask_path:    str
    tile_path:    str
    location:     Optional[str] = None
    notes:        Optional[str] = None
    # B) cechy ilościowe
    ndvi_mean:    Optional[float] = None
    shannon_idx:  Optional[float] = None
    dominant_cls: Optional[str]   = None
    dominant_conf: Optional[float] = None
    patch_count:  Optional[int]   = None
    id:           Optional[int]   = None


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                tile_name     TEXT NOT NULL,
                analyzed_at   TEXT NOT NULL,
                class_stats   TEXT NOT NULL,
                mask_path     TEXT NOT NULL,
                tile_path     TEXT NOT NULL,
                location      TEXT,
                notes         TEXT,
                ndvi_mean     REAL,
                shannon_idx   REAL,
                dominant_cls  TEXT,
                dominant_conf REAL,
                patch_count   INTEGER
            )
        """)
        conn.commit()


def save_analysis(record: AnalysisRecord) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO analyses
               (tile_name, analyzed_at, class_stats, mask_path, tile_path,
                location, notes, ndvi_mean, shannon_idx,
                dominant_cls, dominant_conf, patch_count)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.tile_name, record.analyzed_at,
                json.dumps(record.class_stats, ensure_ascii=False),
                record.mask_path, record.tile_path,
                record.location, record.notes,
                record.ndvi_mean, record.shannon_idx,
                record.dominant_cls, record.dominant_conf,
                record.patch_count,
            )
        )
        conn.commit()
        return cursor.lastrowid


def get_all_analyses() -> list[AnalysisRecord]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY analyzed_at DESC"
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def get_analyses_by_location(location: str) -> list[AnalysisRecord]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE location = ? ORDER BY analyzed_at DESC",
            (location,)
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def _row_to_record(row: sqlite3.Row) -> AnalysisRecord:
    return AnalysisRecord(
        id=row["id"],
        tile_name=row["tile_name"],
        analyzed_at=row["analyzed_at"],
        class_stats=json.loads(row["class_stats"]),
        mask_path=row["mask_path"],
        tile_path=row["tile_path"],
        location=row["location"],
        notes=row["notes"],
        ndvi_mean=row["ndvi_mean"],
        shannon_idx=row["shannon_idx"],
        dominant_cls=row["dominant_cls"],
        dominant_conf=row["dominant_conf"],
        patch_count=row["patch_count"],
    )
