"""
RAG retriever — pgvector + kimi-k2.5 (Ollama Cloud).
"""
import base64
import io
import os
from PIL import Image
import requests

from src.rag.indexer import search_similar

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "kimi-k2.5:cloud")


def _ollama_headers() -> dict:
    token = os.getenv("OLLAMA_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _img_to_b64(img: Image.Image, max_size: int = 512) -> str:
    img = img.copy()
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def ask(query: str, model: str | None = None) -> str:
    """RAG chat — wyszukuje podobne analizy i odpowiada."""
    model = model or OLLAMA_MODEL
    hits  = search_similar(query, limit=5)

    if not hits:
        context = "Brak zapisanych analiz satelitarnych."
    else:
        parts = []
        for h in hits:
            stats = ", ".join(
                f"{k}: {v:.0%}"
                for k, v in sorted(
                    h["class_stats"].items(), key=lambda x: x[1], reverse=True
                )[:4]
            )
            parts.append(
                f"[{h['analyzed_at'][:10]}] {h['location'] or h['tile_name']} "
                f"— dominująca: {h['dominant_cls']}, {stats}, "
                f"NDVI={h['ndvi_mean']:.2f}" if h['ndvi_mean'] else
                f"[{h['analyzed_at'][:10]}] {h['location'] or h['tile_name']} "
                f"— {h['dominant_cls']}"
            )
        context = "\n".join(parts)

    prompt = f"""Jesteś asystentem analizującym dane satelitarne Sentinel-2 dla obszaru Wrocławia.

Kontekst — ostatnie analizy pokrycia terenu:
{context}

Pytanie: {query}

Odpowiedz po polsku, konkretnie i zwięźle."""

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            headers=_ollama_headers(),
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return f"Błąd modelu: {e}"


def classify_patch_with_llm(
    patch: Image.Image,
    ollama_host: str,
    model: str | None = None,
) -> tuple[str, str]:
    """Klasyfikuje patch satelitarny przez LLM vision."""
    from src.dataset import CLASSES
    model    = model or OLLAMA_MODEL
    b64      = _img_to_b64(patch)
    classes  = ", ".join(CLASSES)
    prompt   = (
        f"Classify this Sentinel-2 satellite image patch into exactly one of these "
        f"land cover classes: {classes}.\n"
        f"Reply with JSON: {{\"class\": \"ClassName\", \"reason\": \"brief explanation\"}}"
    )
    try:
        r = requests.post(
            f"{ollama_host}/api/chat",
            headers=_ollama_headers(),
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }],
                "stream": False,
            },
            timeout=60,
        )
        r.raise_for_status()
        import json, re
        text = r.json()["message"]["content"]
        m    = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            cls  = data.get("class", "")
            from src.dataset import CLASSES
            if cls in CLASSES:
                return cls, data.get("reason", "")
        # Fallback — szukaj nazwy klasy w tekście
        from src.dataset import CLASSES
        for c in CLASSES:
            if c.lower() in text.lower():
                return c, text[:100]
        return "Residential", text[:100]
    except Exception as e:
        return "Residential", str(e)


def describe_change_detection(
    result_a: dict,
    result_b: dict,
    date_a: str,
    date_b: str,
    area: str,
    model: str | None = None,
) -> str:
    """Generuje raport change detection przez LLM."""
    model = model or OLLAMA_MODEL

    def fmt_stats(stats: dict) -> str:
        return ", ".join(
            f"{k}: {v:.0%}"
            for k, v in sorted(stats.items(), key=lambda x: x[1], reverse=True)
            if v > 0.01
        )

    prompt = f"""Jesteś ekspertem analizy danych satelitarnych Sentinel-2.

Obszar: {area}
Data A ({date_a}): {fmt_stats(result_a['class_stats'])}
Data B ({date_b}): {fmt_stats(result_b['class_stats'])}

Opisz po polsku zmiany pokrycia terenu między tymi datami.
Zwróć uwagę na: zmiany lesistości, urbanizację, zmiany rolnicze.
Oceń czy zmiany są znaczące czy mieszczą się w błędzie pomiaru.
Odpowiedź max 3-4 zdania."""

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            headers=_ollama_headers(),
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return f"Błąd generowania raportu: {e}"
