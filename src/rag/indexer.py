"""
Indeksowanie analiz satelitarnych w pgvector.
Zastępuje Qdrant — embeddingi trzymane w PostgreSQL obok geometrii PostGIS.
"""
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from src.rag.store_postgis import get_connection

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def init_vector_db():
    """Tworzy tabelę embeddingów z pgvector."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analysis_embeddings (
                    id          SERIAL PRIMARY KEY,
                    analysis_id INTEGER REFERENCES analyses(id) ON DELETE CASCADE,
                    embedding   vector(384),
                    text        TEXT NOT NULL,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            # Używamy zwykłego indeksu - ivfflat wymaga wielu rekordów
            pass
        conn.commit()


def build_text(record) -> str:
    """Buduje tekst do embeddingu z rekordu analizy."""
    dominant = record.dominant_cls or ""
    location = record.location or ""
    notes    = record.notes or ""
    stats    = ", ".join(
        f"{cls}: {pct:.0%}"
        for cls, pct in sorted(
            record.class_stats.items(), key=lambda x: x[1], reverse=True
        )[:3]
    )
    ndvi = f"NDVI={record.ndvi_mean:.2f}" if record.ndvi_mean else ""
    return (
        f"{location} {record.tile_name} {dominant} "
        f"{stats} {ndvi} {notes} {record.analyzed_at[:10]}"
    ).strip()


def index_analysis(record) -> bool:
    """Indeksuje analizę w pgvector. Zwraca True jeśli sukces."""
    try:
        init_vector_db()
        text      = build_text(record)
        embedding = get_model().encode(text).tolist()
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Usuń stary embedding jeśli istnieje
                cur.execute(
                    "DELETE FROM analysis_embeddings WHERE analysis_id = %s",
                    (record.id,)
                )
                cur.execute(
                    """INSERT INTO analysis_embeddings (analysis_id, embedding, text)
                       VALUES (%s, %s, %s)""",
                    (record.id, embedding, text)
                )
            conn.commit()
        return True
    except Exception as e:
        print(f"Błąd indeksowania: {e}")
        return False


def search_similar(query: str, limit: int = 5) -> list[dict]:
    """Wyszukuje analizy podobne do zapytania."""
    try:
        init_vector_db()
        q_emb = get_model().encode(query).tolist()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET enable_indexscan = off;")
                cur.execute("""
                    SELECT
                        a.id, a.tile_name, a.analyzed_at, a.location,
                        a.dominant_cls, a.ndvi_mean, a.class_stats,
                        a.notes, a.mask_path,
                        ST_AsText(a.bbox) AS bbox_wkt,
                        1 - (e.embedding <=> %s::vector) AS similarity,
                        e.text
                    FROM analysis_embeddings e
                    JOIN analyses a ON a.id = e.analysis_id
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                    -- bez indeksu ivfflat dla małych zbiorów
                """, (q_emb, q_emb, limit))
                rows = cur.fetchall()
        return [
            {
                "id":          r[0],
                "tile_name":   r[1],
                "analyzed_at": str(r[2]),
                "location":    r[3],
                "dominant_cls": r[4],
                "ndvi_mean":   r[5],
                "class_stats": r[6] if isinstance(r[6], dict) else json.loads(r[6]),
                "notes":       r[7],
                "mask_path":   r[8],
                "bbox_wkt":    r[9],
                "similarity":  float(r[10]),
                "text":        r[11],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"Błąd wyszukiwania: {e}")
        return []


# ── Baza wiedzy EO ────────────────────────────────────────────────────────────

def init_knowledge_table():
    """Utwórz tabelę knowledge_base jeśli nie istnieje."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(768),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        conn.commit()


def index_knowledge_base(force: bool = False) -> int:
    """Zaindeksuj dokumenty wiedzy EO. Zwraca liczbę zaindeksowanych."""
    from src.rag.eo_knowledge import DOCUMENTS
    init_knowledge_table()
    model = get_model()
    indexed = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Sprawdź ile już zaindeksowano
            cur.execute("SELECT COUNT(*) FROM knowledge_base")
            existing = cur.fetchone()[0]
            if existing >= len(DOCUMENTS) and not force:
                return 0  # Już zaindeksowane

            # Wyczyść i zaindeksuj od nowa
            cur.execute("DELETE FROM knowledge_base")
            for doc in DOCUMENTS:
                text = f"{doc['title']}: {doc['content']}"
                emb = model.encode(text).tolist()
                cur.execute("""
                    INSERT INTO knowledge_base (title, content, embedding)
                    VALUES (%s, %s, %s::vector)
                """, (doc["title"], doc["content"], emb))
                indexed += 1
        conn.commit()
    return indexed


def search_knowledge(query: str, limit: int = 3) -> list[dict]:
    """Wyszukaj dokumenty wiedzy EO podobne do zapytania."""
    try:
        init_knowledge_table()
        q_emb = get_model().encode(query).tolist()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET enable_indexscan = off;")
                cur.execute("""
                    SELECT title, content,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM knowledge_base
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (q_emb, q_emb, limit))
                rows = cur.fetchall()
        return [
            {"title": r[0], "content": r[1], "similarity": float(r[2])}
            for r in rows
        ]
    except Exception as e:
        logger.error("search_knowledge błąd: %s", e)
        return []
