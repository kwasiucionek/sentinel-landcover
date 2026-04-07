import io
import os
import base64
import numpy as np
from PIL import Image
from .indexer import _get_embedder, _get_qdrant, QDRANT_COLLECTION

TOP_K = 5


def search_similar(query: str, top_k: int = TOP_K) -> list[dict]:
    vector  = _get_embedder().encode(query).tolist()
    results = _get_qdrant().query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload for hit in results.points]


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_str = "\n\n".join(
        f"[Analiza {i+1}] {ctx['text']}"
        for i, ctx in enumerate(contexts)
    )
    return f"""Jesteś ekspertem ds. teledetekcji i obserwacji Ziemi (Earth Observation).
Odpowiadasz na pytania dotyczące analiz satelitarnych pokrycia terenu wykonanych modelem SegFormer.

Dostępne analizy z historii:
{context_str}

Pytanie użytkownika: {question}

Odpowiedz po polsku, konkretnie i rzeczowo. Jeśli dane są niewystarczające, powiedz o tym wprost."""


def ask(question: str, ollama_client) -> str:
    contexts = search_similar(question)
    if not contexts:
        return "Brak analiz w historii. Wykonaj najpierw analizę satelitarną."
    prompt   = build_prompt(question, contexts)
    model    = os.getenv("OLLAMA_MODEL", "qwen3-vl:235b-cloud")
    response = ollama_client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content


def describe_change_detection(
    area: str,
    date_a: str,
    date_b: str,
    result_a: dict,
    result_b: dict,
    diff_stats: dict,
    change_pct: float,
    ollama_client,
) -> str:
    """Generuje opis zmiany pokrycia terenu przez LLM."""
    stats_a_str = ", ".join(
        f"{cls} {pct:.1%}" for cls, pct in
        sorted(result_a["class_stats"].items(), key=lambda x: x[1], reverse=True)
        if pct > 0
    )
    stats_b_str = ", ".join(
        f"{cls} {pct:.1%}" for cls, pct in
        sorted(result_b["class_stats"].items(), key=lambda x: x[1], reverse=True)
        if pct > 0
    )
    significant = {k: v for k, v in diff_stats.items() if abs(v) >= 0.02}
    diff_str = ", ".join(
        f"{cls} {'wzrósł' if v > 0 else 'zmalał'} o {abs(v):.1%}"
        for cls, v in sorted(significant.items(), key=lambda x: abs(x[1]), reverse=True)
    ) or "brak istotnych zmian"

    prompt = f"""Jesteś ekspertem ds. teledetekcji i analiz przestrzennych pracującym dla firmy GIS.
Przygotuj krótki raport analityczny (3-4 zdania) na podstawie danych z klasyfikacji pokrycia terenu.

Obszar: {area}
Porównanie: {date_a} → {date_b}

Pokrycie terenu {date_a}: {stats_a_str}
Pokrycie terenu {date_b}: {stats_b_str}
Powierzchnia zmian: {change_pct:.1%} obszaru
Główne zmiany: {diff_str}

Napisz rzeczowy raport po polsku. Wskaż możliwe przyczyny zmian i ich znaczenie \
dla planowania przestrzennego lub zarządzania środowiskiem.
Jeśli zmiany są małe (<5%), zaznacz że obszar jest stabilny."""

    model    = os.getenv("OLLAMA_MODEL", "qwen3-vl:235b-cloud")
    response = ollama_client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content


def classify_patch_with_llm(
    patch_img: Image.Image,
    ollama_client,
    model: str = None,
) -> tuple[str, str]:
    """
    Klasyfikuje patch satelitarny przez multimodalny LLM.
    Zwraca (klasa, uzasadnienie).
    """
    from src.dataset import CLASSES

    model = model or os.getenv("OLLAMA_MODEL", "qwen3-vl:235b-cloud")

    # Powiększ patch do 256x256 — lepiej widoczny dla LLM
    img_large = patch_img.resize((256, 256), Image.NEAREST)
    buf       = io.BytesIO()
    img_large.save(buf, format="PNG")
    img_b64   = base64.b64encode(buf.getvalue()).decode()

    classes_str = ", ".join(CLASSES)
    prompt = f"""You are an expert in satellite image analysis and Earth Observation.
Classify this Sentinel-2 satellite image patch into exactly one of these land cover classes:
{classes_str}

Rules:
- Reply with ONLY the class name on the first line
- Then one sentence explaining what you see
- Use the exact class name from the list above

Example response:
Forest
Dense coniferous forest visible with dark green uniform texture."""

    response = ollama_client.chat(
        model=model,
        messages=[{
            "role":    "user",
            "content": prompt,
            "images":  [img_b64],
        }],
    )
    text  = response.message.content.strip()
    lines = text.split("\n", 1)
    predicted_class = lines[0].strip()
    reasoning       = lines[1].strip() if len(lines) > 1 else ""

    # Walidacja
    if predicted_class not in CLASSES:
        for cls in CLASSES:
            if cls.lower() in predicted_class.lower():
                predicted_class = cls
                break
        else:
            predicted_class = "Residential"  # fallback

    return predicted_class, reasoning
