import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from .store import AnalysisRecord, QDRANT_COLLECTION

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384

_embedder = None
_qdrant   = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        _qdrant = QdrantClient(host=host, port=port)
        _ensure_collection(_qdrant)
    return _qdrant


def _ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def record_to_text(record: AnalysisRecord) -> str:
    stats_str = ", ".join(
        f"{cls} {pct:.1%}"
        for cls, pct in sorted(record.class_stats.items(),
                                key=lambda x: x[1], reverse=True)
        if pct > 0
    )
    location = record.location or "nieznana lokalizacja"
    notes    = f" Notatki: {record.notes}." if record.notes else ""

    quant = ""
    if record.ndvi_mean is not None:
        quant += f" NDVI średnie: {record.ndvi_mean:.3f}."
    if record.shannon_idx is not None:
        quant += f" Indeks różnorodności Shannona: {record.shannon_idx:.3f}."
    if record.dominant_cls and record.dominant_conf is not None:
        quant += (f" Dominująca klasa: {record.dominant_cls} "
                  f"(pewność {record.dominant_conf:.1%}).")
    if record.patch_count is not None:
        quant += f" Liczba przeanalizowanych patchów: {record.patch_count}."

    return (
        f"Analiza satelitarna: {record.tile_name}. "
        f"Data: {record.analyzed_at[:10]}. "
        f"Lokalizacja: {location}. "
        f"Pokrycie terenu: {stats_str}.{quant}{notes}"
    )


def index_analysis(record: AnalysisRecord):
    text    = record_to_text(record)
    vector  = _get_embedder().encode(text).tolist()
    _get_qdrant().upsert(
        collection_name=QDRANT_COLLECTION,
        points=[PointStruct(
            id=record.id,
            vector=vector,
            payload={
                "tile_name":    record.tile_name,
                "analyzed_at":  record.analyzed_at,
                "location":     record.location,
                "class_stats":  record.class_stats,
                "ndvi_mean":    record.ndvi_mean,
                "shannon_idx":  record.shannon_idx,
                "dominant_cls": record.dominant_cls,
                "text":         text,
            }
        )]
    )
