"""
Sentinel Land Cover Analyzer
Streamlit app: upload tile → segmentacja ONNX → RAG chat z historią analiz
"""

import io
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.dataset import CLASS_COLORS, CLASSES
from src.rag.indexer import index_analysis
from src.rag.retriever import ask, classify_patch_with_llm, describe_change_detection
from src.rag.store_postgis import (
    AnalysisRecord,
    get_all_analyses,
    get_analyses_by_location,
    init_db,
    save_analysis,
)

# ── Stałe ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "data/processed/model.onnx"
MODEL_WMS_PATH = "data/processed/model_wms.onnx"
ANALYSES_DIR = "data/analyses"
IMAGE_SIZE = 256
IMAGE_SIZE_GEO = 256  # ten sam rozmiar co model główny
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

os.makedirs(ANALYSES_DIR, exist_ok=True)
init_db()

# Reindeksuj analizy które nie mają embeddingów
try:
    from src.rag.indexer import index_analysis as _idx
    from src.rag.store_postgis import get_all_analyses as _get_all, get_connection as _gc
    _records = _get_all()
    with _gc() as _conn:
        with _conn.cursor() as _cur:
            _cur.execute("SELECT analysis_id FROM analysis_embeddings")
            _indexed = {r[0] for r in _cur.fetchall()}
    for _r in _records:
        if _r.id not in _indexed:
            _idx(_r)
except Exception as _e:
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


@st.cache_resource
def load_model_wms():
    # Priorytety: Esri LULC labels > WMS pseudo > EuroSAT
    for path, name in [
        ("data/processed/model_esri.onnx", "SegFormer Esri LULC"),
        ("data/processed/model_geo.onnx",  "ResNet50 SSL4EO"),
        ("data/processed/model_wms.onnx",  "SegFormer WMS"),
        (MODEL_PATH,                        "SegFormer EuroSAT"),
    ]:
        if os.path.exists(path):
            print(f"Używam modelu WMS: {name} ({path})")
            return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


@st.cache_resource
def load_ollama():
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


@st.cache_resource
def load_cdse():
    from src.cdse_client import CDSEClient

    return CDSEClient()


def preprocess_patch(patch: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    img = Image.fromarray(patch).resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # (3, H, W) — RGB bez NIR
    arr = (arr - MEAN[:, None, None]) / STD[:, None, None]
    return arr[None].astype(np.float32)


def compute_ndvi(arr_rgb: np.ndarray) -> float:
    r = arr_rgb[:, :, 0].astype(float) / 255.0
    nir = arr_rgb.mean(axis=2).astype(float) / 255.0
    return float(((nir - r) / (nir + r + 1e-8)).mean())


def compute_shannon(class_stats: dict) -> float:
    h = 0.0
    for p in class_stats.values():
        if p > 0:
            h -= p * math.log(p)
    return h


def analyze_grid(
    img: Image.Image, session: ort.InferenceSession, patch_size: int = IMAGE_SIZE
) -> dict:
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    n_rows = max(1, h // patch_size)
    n_cols = max(1, w // patch_size)
    arr = arr[: n_rows * patch_size, : n_cols * patch_size]
    mask_arr = np.zeros((n_rows * patch_size, n_cols * patch_size, 3), dtype=np.uint8)
    class_counts = {cls: 0 for cls in CLASSES}
    confidences = []
    for r in range(n_rows):
        for c in range(n_cols):
            patch = arr[
                r * patch_size : (r + 1) * patch_size,
                c * patch_size : (c + 1) * patch_size,
            ]
            inp = preprocess_patch(patch, size=patch_size)
            logits = (
                session.run(["logits"], {"pixel_values": inp})[0][0]
                if "pixel_values" in [i.name for i in session.get_inputs()]
                else session.run(["logits"], {"x": inp})[0][0]
            )
            probs = np.exp(logits) / np.exp(logits).sum()
            cls_idx = int(np.argmax(probs))
            confidences.append(float(probs[cls_idx]))
            class_counts[CLASSES[cls_idx]] += 1
            mask_arr[
                r * patch_size : (r + 1) * patch_size,
                c * patch_size : (c + 1) * patch_size,
            ] = CLASS_COLORS[cls_idx]
    total = n_rows * n_cols
    class_stats = {cls: cnt / total for cls, cnt in class_counts.items()}
    mask_img = Image.fromarray(mask_arr).resize(img.size, Image.NEAREST)
    return {
        "class_stats": class_stats,
        "mask_img": mask_img,
        "softmax_mean": float(np.mean(confidences)),
        "ndvi_mean": compute_ndvi(arr),
        "patch_count": total,
    }


def compute_change_detection(result_a: dict, result_b: dict) -> dict:
    stats_a = result_a["class_stats"]
    stats_b = result_b["class_stats"]
    diff_stats = {cls: stats_b.get(cls, 0.0) - stats_a.get(cls, 0.0) for cls in CLASSES}
    mask_a = np.array(result_a["mask_img"])
    mask_b = np.array(result_b["mask_img"])
    changed = np.any(mask_a != mask_b, axis=2)
    change_pct = float(changed.mean())
    change_arr = np.where(changed[:, :, None], mask_b, np.array([180, 180, 180]))
    change_mask = Image.fromarray(change_arr.astype(np.uint8))
    return {
        "diff_stats": diff_stats,
        "change_mask": change_mask,
        "change_pct": change_pct,
    }


def save_analysis_record(
    tile_name: str,
    img: Image.Image,
    result: dict,
    tile_path: str,
    location: str,
    notes: str,
    bbox_wkt: str = None,
) -> AnalysisRecord:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_path = os.path.join(ANALYSES_DIR, f"{ts}_{tile_name}_mask.png")
    result["mask_img"].save(mask_path)
    class_stats = result["class_stats"]
    dominant_cls = max(class_stats, key=class_stats.get)
    record = AnalysisRecord(
        tile_name=tile_name,
        analyzed_at=datetime.now().isoformat(),
        class_stats=class_stats,
        mask_path=mask_path,
        tile_path=tile_path,
        location=location or None,
        notes=notes or None,
        ndvi_mean=result["ndvi_mean"],
        shannon_idx=compute_shannon(class_stats),
        dominant_cls=dominant_cls,
        dominant_conf=result["softmax_mean"],
        patch_count=result["patch_count"],
        bbox_wkt=bbox_wkt,
    )
    record.id = save_analysis(record)
    index_analysis(record)
    return record


def show_metrics(dominant_cls, dominant_conf, ndvi_mean, shannon_idx):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Klasa dominująca",
        dominant_cls,
        help="Klasa pokrycia terenu zajmująca największą powierzchnię",
    )
    m2.metric(
        "Pewność modelu",
        f"{dominant_conf:.1%}",
        help=">80% dobra klasyfikacja, <60% obszar trudny do sklasyfikowania",
    )
    m3.metric(
        "NDVI",
        f"{ndvi_mean:.3f}",
        help="Indeks roślinności: <0 woda/beton, 0-0.3 niska roślinność, >0.3 las",
    )
    m4.metric(
        "Shannon",
        f"{shannon_idx:.3f}",
        help="Różnorodność terenu: 0 = jeden typ, ~2.3 = maks. różnorodność (10 klas)",
    )


def plot_class_distribution(class_stats: dict):
    import matplotlib.pyplot as plt

    active = {k: v for k, v in class_stats.items() if v > 0}
    colors = [
        "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS[CLASSES.index(k)]) for k in active
    ]
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.bar(active.keys(), active.values(), color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Udział")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def compare_with_previous(location: str, current_stats: dict) -> str | None:
    if not location:
        return None
    prev = get_analyses_by_location(location)
    if not prev:
        return None
    lines = []
    for cls, cur_val in current_stats.items():
        diff = cur_val - prev[0].class_stats.get(cls, 0.0)
        if abs(diff) >= 0.02:
            lines.append(f"{cls}: {'▲' if diff > 0 else '▼'} {diff:+.1%}")
    return (
        ("Zmiany vs poprzednia analiza:\n" + "\n".join(lines))
        if lines
        else "Brak istotnych zmian."
    )


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel Land Cover Analyzer", page_icon="🛰️", layout="wide"
)
st.title("🛰️ Sentinel Land Cover Analyzer")
st.caption(
    "Segmentacja pokrycia terenu · SegFormer-B2 · Copernicus Sentinel-2 · RAG chat"
)

session = load_model()
session_wms = load_model_wms()
ollama = load_ollama()
cdse = load_cdse()

tab_analyze, tab_history, tab_chat, tab_geo, tab_change, tab_llm = st.tabs(
    [
        "📡 Analiza",
        "📋 Historia",
        "💬 Chat RAG",
        "🗺️ Geoportal",
        "🔄 Change Detection",
        "🤖 LLM vs DL",
    ]
)

# ── Tab: Analiza ──────────────────────────────────────────────────────────────
with tab_analyze:
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Wgraj obraz satelitarny")
        st.caption("Minimum 64×64 px. Większe obrazy krojone na siatkę patchów.")
        uploaded = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"])
        location = st.text_input("Lokalizacja", placeholder="np. Wrocław, Psie Pole")
        notes = st.text_area("Notatki", height=80)
        analyze_btn = st.button("▶ Analizuj", type="primary", disabled=uploaded is None)

    with col_right:
        if uploaded and analyze_btn:
            img = Image.open(uploaded)
            result = analyze_grid(img, session)
            tile_name = os.path.splitext(uploaded.name)[0]
            tile_path = os.path.join(ANALYSES_DIR, uploaded.name)
            img.save(tile_path)
            record = save_analysis_record(
                tile_name, img, result, tile_path, location, notes
            )

            class_stats = result["class_stats"]
            st.success(f"Przeanalizowano {result['patch_count']} patchów")
            c1, c2 = st.columns(2)
            c1.image(img, caption="Oryginalny obraz", use_container_width=True)
            c2.image(
                result["mask_img"],
                caption="Maska pokrycia terenu",
                use_container_width=True,
            )
            show_metrics(
                record.dominant_cls,
                record.dominant_conf,
                record.ndvi_mean,
                record.shannon_idx,
            )
            plot_class_distribution(class_stats)
            if location:
                diff = compare_with_previous(location, class_stats)
                if diff:
                    st.info(diff)
            st.caption(f"Analiza zapisana · id={record.id}")
            st.rerun()

# ── Tab: Historia ─────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("Historia analiz")

    # Wyszukiwanie przestrzenne
    with st.expander("🔍 Wyszukaj przestrzennie"):
        from src.rag.store_postgis import get_analyses_near_point, get_analyses_intersecting
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            search_lon = st.number_input("Longitude", value=17.038, format="%.4f")
            search_lat = st.number_input("Latitude",  value=51.107, format="%.4f")
            radius_km  = st.slider("Promień (km)", 1, 50, 5)
        with col_s2:
            st.markdown("**Centrum Wrocławia:** 17.038, 51.107")
            st.markdown("**Psie Pole:** 17.075, 51.148")
            st.markdown("**Fabryczna:** 16.915, 51.093")
        if st.button("🗺️ Szukaj w pobliżu"):
            nearby = get_analyses_near_point(search_lon, search_lat, radius_km * 1000)
            if nearby:
                st.success(f"Znaleziono {len(nearby)} analiz w promieniu {radius_km} km")
                for r in nearby:
                    st.write(f"📍 [{r.id}] {r.tile_name} · {r.location} · {r.dominant_cls}")
            else:
                st.info("Brak analiz w tym obszarze")

    records = get_all_analyses()
    if not records:
        st.info("Brak analiz. Wykonaj pierwszą analizę w zakładce 📡.")
    else:
        for rec in records:
            dominant = rec.dominant_cls or max(rec.class_stats, key=rec.class_stats.get)
            with st.expander(
                f"[{rec.id}] {rec.tile_name} · {rec.analyzed_at[:16]} · {dominant}"
            ):
                c1, c2 = st.columns([1, 2])
                if os.path.exists(rec.mask_path):
                    c1.image(rec.mask_path, caption="Maska", use_container_width=True)
                with c2:
                    plot_class_distribution(rec.class_stats)
                cols = c2.columns(3)
                if rec.ndvi_mean is not None:
                    cols[0].metric("NDVI", f"{rec.ndvi_mean:.3f}")
                if rec.shannon_idx is not None:
                    cols[1].metric("Shannon", f"{rec.shannon_idx:.3f}")
                if rec.patch_count is not None:
                    cols[2].metric("Patche", rec.patch_count)
                if rec.location:
                    c2.write(f"📍 {rec.location}")
                if rec.notes:
                    c2.write(f"📝 {rec.notes}")

# ── Tab: Chat RAG ─────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Chat z historią analiz")
    st.caption(
        "LLM odpowiada na podstawie zapisanych wyników analiz satelitarnych"
    )
    chat_model = st.selectbox(
        "Model LLM",
        [
            "kimi-k2.5:cloud",
            "qwen3-vl:235b-instruct-cloud",
            "gemma4:31b-cloud",
            "mistral-large-3:675b-cloud",
        ],
        key="chat_model",
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Zapytaj o historię analiz..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("AI analizuje historię..."):
                answer = ask(prompt, model=chat_model)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Tab: Geoportal ────────────────────────────────────────────────────────────
with tab_geo:
    from src.cdse_client import WROCLAW_BBOXES

    st.subheader("🗺️ Sentinel-2 — Wrocław (Copernicus Data Space)")
    st.caption(
        "Prawdziwe sceny satelitarne pobierane na żywo z CDSE · klasyfikacja naszym modelem"
    )

    instance_id = os.getenv("CDSE_INSTANCE_ID", "")
    if not instance_id:
        st.error("Brak CDSE_INSTANCE_ID w .env")
    else:
        col_left, col_right = st.columns([1, 2])
        with col_left:
            area = st.selectbox("Obszar Wrocławia", list(WROCLAW_BBOXES.keys()))
            layer = st.selectbox("Warstwa", ["TRUE-COLOR", "FALSE-COLOR", "NDVI"])
            date = st.date_input("Data sceny", value=datetime(2026, 3, 22))
            fetch_btn = st.button("📥 Pobierz scenę", type="primary")

        with col_right:
            if fetch_btn:
                lon_min, lat_min, lon_max, lat_max = WROCLAW_BBOXES[area]
                date_str = date.strftime("%Y-%m-%d")
                with st.spinner(f"Pobieranie {layer} dla {area}..."):
                    img = cdse.fetch_wms_preview(
                        instance_id,
                        lon_min,
                        lat_min,
                        lon_max,
                        lat_max,
                        layer=layer,
                        size=2500,
                        date=date_str,
                    )
                if img is None:
                    st.error("Błąd pobierania — sprawdź datę lub zwiększ zakres chmur")
                else:
                    with st.spinner("Klasyfikacja siatką patchów..."):
                        result = analyze_grid(
                            img, session_wms, patch_size=IMAGE_SIZE_GEO
                        )
                    st.session_state["geo_img"] = img
                    st.session_state["geo_result"] = result
                    st.session_state["geo_area"] = area
                    st.session_state["geo_layer"] = layer
                    st.session_state["geo_date"] = date_str

            if st.session_state.get("geo_img") is not None:
                img = st.session_state["geo_img"]
                result = st.session_state["geo_result"]
                area_s = st.session_state["geo_area"]
                layer_s = st.session_state["geo_layer"]
                date_str = st.session_state["geo_date"]
                lon_min, lat_min, lon_max, lat_max = WROCLAW_BBOXES[area_s]

                st.image(
                    img,
                    caption=f"Sentinel-2 {layer_s} · {area_s} · {date_str}",
                    use_container_width=True,
                )

                # Konwertuj do bytes do pobrania
                import io as _io

                buf = _io.BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Pobierz obraz 2500×2500",
                    data=buf.getvalue(),
                    file_name=f"sentinel2_{layer_s}_{area_s}_{date_str}.png",
                    mime="image/png",
                )

                if layer_s == "TRUE-COLOR":
                    st.markdown("---")
                    class_stats = result["class_stats"]
                    dominant_cls = max(class_stats, key=class_stats.get)
                    shannon_idx = compute_shannon(class_stats)
                    c1, c2 = st.columns(2)
                    c1.image(img, caption="Sentinel-2 RGB", use_container_width=True)
                    c2.image(
                        result["mask_img"],
                        caption="Maska naszego modelu",
                        use_container_width=True,
                    )
                    show_metrics(
                        dominant_cls,
                        result["softmax_mean"],
                        result["ndvi_mean"],
                        shannon_idx,
                    )
                    plot_class_distribution(class_stats)

                    # Warstwy referencyjne
                    with st.expander("🗺️ Warstwy referencyjne — BDOT10k / Ortofoto / MPZP"):
                        from src.bdot10k_client import (
                            fetch_bdot10k_area, fetch_egib_parcels,
                            fetch_mpzp_przeznaczenie, detect_mpzp_discrepancies,
                            detect_bdot_discrepancies,
                        )
                        from src.ortofoto_client import fetch_ortofoto_area

                        col_ref1, col_ref2 = st.columns(2)
                        with col_ref1:
                            with st.spinner("BDOT10k..."):
                                bdot = fetch_bdot10k_area(lon_min, lat_min, lon_max, lat_max)
                            if bdot:
                                st.image(bdot, caption="BDOT10k (GUGiK)", use_container_width=True)
                        with col_ref2:
                            with st.spinner("Ortofoto 2025..."):
                                orto = fetch_ortofoto_area(lon_min, lat_min, lon_max, lat_max)
                            if orto:
                                st.image(orto, caption="Ortofoto 2025 (UM Wrocław)", use_container_width=True)

                        col_ref3, col_ref4 = st.columns(2)
                        with col_ref3:
                            st.markdown("**BDOT10k:**")
                            for a in detect_bdot_discrepancies(class_stats, bdot):
                                st.markdown(a)
                        with col_ref4:
                            with st.spinner("MPZP..."):
                                mpzp = fetch_mpzp_przeznaczenie(lon_min, lat_min, lon_max, lat_max)
                            st.markdown("**MPZP:**")
                            for a in detect_mpzp_discrepancies(class_stats, mpzp):
                                st.markdown(a)

                    if st.button("💾 Zapisz do historii"):
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        tile_name = f"cdse_{area_s.replace(' ', '_')}_{date_str}"
                        tile_path = os.path.join(ANALYSES_DIR, f"{ts}_{tile_name}.png")
                        img.save(tile_path)
                        lon_min_s, lat_min_s, lon_max_s, lat_max_s = WROCLAW_BBOXES[area_s]
                        bbox_wkt_s = (f"POLYGON(({lon_min_s} {lat_min_s},{lon_max_s} {lat_min_s},"
                                      f"{lon_max_s} {lat_max_s},{lon_min_s} {lat_max_s},{lon_min_s} {lat_min_s}))")
                        record = save_analysis_record(
                            tile_name,
                            img,
                            result,
                            tile_path,
                            f"Wrocław, {area_s}",
                            f"Sentinel-2 {layer_s} z CDSE, data sceny: {date_str}",
                            bbox_wkt=bbox_wkt_s,
                        )
                        st.session_state["geo_img"] = None
                        st.success(f"Zapisano do historii (id={record.id})")
                        st.rerun()

                with st.expander("📋 Dostępne sceny Sentinel-2 dla tego obszaru"):
                    with st.spinner("Wyszukiwanie w CDSE..."):
                        scenes = cdse.search_scenes(
                            lon_min,
                            lat_min,
                            lon_max,
                            lat_max,
                            max_cloud=30,
                            max_results=10,
                        )
                    if scenes:
                        for s in scenes:
                            sdate = s.get("ContentDate", {}).get("Start", "?")[:10]
                            size = s.get("ContentLength", 0) / 1e9
                            st.text(
                                f"[{sdate}] {s.get('Name', '?')[:55]} ({size:.1f} GB)"
                            )
                    else:
                        st.info("Brak scen dla tego obszaru i zakresu dat")

# ── Tab: Change Detection ─────────────────────────────────────────────────────
with tab_change:
    from src.cdse_client import WROCLAW_BBOXES

    st.subheader("🔄 Change Detection — porównanie dwóch dat")
    st.caption("Wykrywa zmiany pokrycia terenu między dwiema scenami Sentinel-2")
    st.warning(
        "⚠️ Wiarygodne wyniki wymagają scen z podobnymi warunkami atmosferycznymi. "
        "Zalecane: ten sam miesiąc, różne lata (np. marzec 2025 vs marzec 2026)."
    )

    instance_id = os.getenv("CDSE_INSTANCE_ID", "")
    if not instance_id:
        st.error("Brak CDSE_INSTANCE_ID w .env")
    else:
        col_l, col_r = st.columns(2)
        with col_l:
            area_cd = st.selectbox("Obszar", list(WROCLAW_BBOXES.keys()), key="cd_area")
            date_a = st.date_input(
                "Data A (starsza)", value=datetime(2025, 3, 22), key="cd_date_a"
            )
            date_b = st.date_input(
                "Data B (nowsza)", value=datetime(2026, 3, 22), key="cd_date_b"
            )
            run_cd = st.button("🔍 Wykryj zmiany", type="primary")
        with col_r:
            st.markdown("**Jak czytać wyniki:**")
            st.markdown("- Szary = brak zmiany klasy")
            st.markdown("- Kolor = nowa klasa pokrycia terenu")
            st.markdown("- % zmiany = ile patchów zmieniło klasę")

        if run_cd:
            lon_min, lat_min, lon_max, lat_max = WROCLAW_BBOXES[area_cd]
            date_a_str = date_a.strftime("%Y-%m-%d")
            date_b_str = date_b.strftime("%Y-%m-%d")
            if date_a_str >= date_b_str:
                st.error("Data A musi być starsza niż Data B")
            else:
                with st.spinner(f"Pobieranie sceny A ({date_a_str})..."):
                    img_a = cdse.fetch_wms_preview(
                        instance_id,
                        lon_min,
                        lat_min,
                        lon_max,
                        lat_max,
                        layer="TRUE-COLOR",
                        size=2500,
                        date=date_a_str,
                    )
                with st.spinner(f"Pobieranie sceny B ({date_b_str})..."):
                    img_b = cdse.fetch_wms_preview(
                        instance_id,
                        lon_min,
                        lat_min,
                        lon_max,
                        lat_max,
                        layer="TRUE-COLOR",
                        size=2500,
                        date=date_b_str,
                    )
                if img_a is None or img_b is None:
                    st.error("Nie udało się pobrać jednej ze scen — spróbuj innych dat")
                else:
                    with st.spinner("Klasyfikacja i analiza zmian..."):
                        result_a = analyze_grid(
                            img_a, session_wms, patch_size=IMAGE_SIZE_GEO
                        )
                        result_b = analyze_grid(
                            img_b, session_wms, patch_size=IMAGE_SIZE_GEO
                        )
                        change = compute_change_detection(result_a, result_b)
                    st.session_state["cd_results"] = {
                        "img_a": img_a,
                        "img_b": img_b,
                        "result_a": result_a,
                        "result_b": result_b,
                        "change": change,
                        "area": area_cd,
                        "date_a": date_a_str,
                        "date_b": date_b_str,
                    }

        if "cd_results" in st.session_state:
            cd = st.session_state["cd_results"]
            result_a = cd["result_a"]
            result_b = cd["result_b"]
            change = cd["change"]
            date_a_str = cd["date_a"]
            date_b_str = cd["date_b"]

            c1, c2, c3 = st.columns(3)
            c1.image(
                result_a["mask_img"],
                caption=f"Maska A · {date_a_str}",
                use_container_width=True,
            )
            c2.image(
                change["change_mask"], caption="Mapa zmian", use_container_width=True
            )
            c3.image(
                result_b["mask_img"],
                caption=f"Maska B · {date_b_str}",
                use_container_width=True,
            )

            # Warstwa referencyjna Esri LULC
            st.markdown("---")
            st.markdown("**🌍 Warstwa referencyjna — Esri LULC (Impact Observatory)**")
            st.caption("Globalna klasyfikacja 10m Sentinel-2 · dane 2017-2024 · 9 klas")

            from src.esri_lulc_client import ESRI_CLASSES, fetch_esri_lulc

            col_e1, col_e2 = st.columns(2)
            for col, year in zip([col_e1, col_e2], [2022, 2023]):
                with st.spinner(f"Pobieranie Esri LULC {year}..."):
                    lon_min, lat_min, lon_max, lat_max = WROCLAW_BBOXES[cd["area"]]
                    esri_img = fetch_esri_lulc(
                        lon_min,
                        lat_min,
                        lon_max,
                        lat_max,
                        year=year,
                        size=2500,
                    )
                if esri_img:
                    col.image(
                        esri_img,
                        caption=f"Esri LULC {year} · {cd['area']}",
                        use_container_width=True,
                    )
                else:
                    col.warning(f"Esri LULC {year} niedostępny")

            # Legenda Esri
            with st.expander("📋 Legenda Esri LULC"):
                cols = st.columns(3)
                for i, (cls_id, info) in enumerate(ESRI_CLASSES.items()):
                    if info["name"] not in [
                        ESRI_CLASSES[j]["name"] for j in range(1, cls_id)
                    ]:
                        color_hex = "#{:02x}{:02x}{:02x}".format(*info["color"])
                        cols[i % 3].markdown(
                            f"<span style='color:{color_hex}'>■</span> "
                            f"**{info['name']}** → {info['eurosat'] or 'brak'}",
                            unsafe_allow_html=True,
                        )

            # Warstwy referencyjne — BDOT10k + Ortofoto 2025
            st.markdown("---")
            st.markdown("**🗺️ Warstwy referencyjne GUGiK / UM Wrocław**")

            from src.bdot10k_client import (
                fetch_bdot10k_area, fetch_egib_parcels, detect_bdot_discrepancies
            )
            from src.ortofoto_client import fetch_ortofoto_area

            lon_min_r, lat_min_r, lon_max_r, lat_max_r = WROCLAW_BBOXES[cd["area"]]

            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                with st.spinner("BDOT10k..."):
                    bdot_img = fetch_bdot10k_area(
                        lon_min_r, lat_min_r, lon_max_r, lat_max_r
                    )
                if bdot_img:
                    st.image(bdot_img, caption="BDOT10k (GUGiK)", use_container_width=True)
                else:
                    st.warning("BDOT10k niedostępny")

            with col_r2:
                with st.spinner("Ortofoto 2025..."):
                    orto_img = fetch_ortofoto_area(
                        lon_min_r, lat_min_r, lon_max_r, lat_max_r
                    )
                if orto_img:
                    st.image(orto_img, caption="Ortofoto 2025 (UM Wrocław)", use_container_width=True)
                else:
                    st.warning("Ortofoto niedostępne")

            with col_r3:
                with st.spinner("EGiB + MPZP..."):
                    parcels = fetch_egib_parcels(
                        lon_min_r, lat_min_r, lon_max_r, lat_max_r
                    )
                    from src.bdot10k_client import (
                        fetch_mpzp_przeznaczenie, detect_mpzp_discrepancies
                    )
                    mpzp = fetch_mpzp_przeznaczenie(
                        lon_min_r, lat_min_r, lon_max_r, lat_max_r
                    )
                st.metric("Działki EGiB", len(parcels))
                st.markdown("**BDOT10k:**")
                for a in detect_bdot_discrepancies(result_b["class_stats"], bdot_img):
                    st.markdown(a)
                st.markdown("**MPZP:**")
                for a in detect_mpzp_discrepancies(result_b["class_stats"], mpzp):
                    st.markdown(a)

            st.metric(
                "Powierzchnia zmian",
                f"{change['change_pct']:.1%}",
                help="Procent patchów które zmieniły klasę pokrycia terenu",
            )

            st.markdown("**Zmiany udziału klas (Data B − Data A):**")
            significant = {
                k: v for k, v in change["diff_stats"].items() if abs(v) >= 0.02
            }
            if not significant:
                st.info(
                    "Brak istotnych zmian (próg: ±2%). Spróbuj dat z większym odstępem."
                )
            else:
                for cls, delta in sorted(
                    significant.items(), key=lambda x: abs(x[1]), reverse=True
                ):
                    bar_a = result_a["class_stats"].get(cls, 0)
                    bar_b = result_b["class_stats"].get(cls, 0)
                    color = "green" if delta > 0 else "red"
                    st.markdown(
                        f"**{cls}**: {bar_a:.1%} → {bar_b:.1%} "
                        f"<span style='color:{color}'>{'▲' if delta > 0 else '▼'} {delta:+.1%}</span>",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            st.markdown("**🤖 Analiza AI**")
            if st.button("📝 Generuj raport AI"):
                with st.spinner("AI analizuje zmiany..."):
                    report = describe_change_detection(
                        area=cd["area"],
                        date_a=date_a_str,
                        date_b=date_b_str,
                        result_a=result_a,
                        result_b=result_b,
                        diff_stats=change["diff_stats"],
                        change_pct=change["change_pct"],
                        ollama_client=ollama,
                    )
                st.session_state["cd_report"] = report
            if "cd_report" in st.session_state:
                st.info(st.session_state["cd_report"])

            if st.button("💾 Zapisz obie sceny do historii"):
                for img, result, date_str, suffix in [
                    (cd["img_a"], result_a, date_a_str, "A"),
                    (cd["img_b"], result_b, date_b_str, "B"),
                ]:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    tile_name = f"cd_{cd['area'].replace(' ', '_')}_{date_str}"
                    tile_path = os.path.join(ANALYSES_DIR, f"{ts}_{tile_name}.png")
                    img.save(tile_path)
                    save_analysis_record(
                        tile_name,
                        img,
                        result,
                        tile_path,
                        f"Wrocław, {cd['area']}",
                        f"Change Detection scena {suffix} · {date_a_str} vs {date_b_str}",
                    )
                st.success("Obie sceny zapisane do historii")
                st.rerun()

# ── Tab: LLM vs DL ───────────────────────────────────────────────────────────
with tab_llm:
    from src.cdse_client import WROCLAW_BBOXES

    st.subheader("🤖 LLM vs DL — porównanie klasyfikatorów")
    st.caption(
        "Porównanie klasyfikacji satelitarnej: "
        "SegFormer Esri LULC vs LLM vision (multimodal)"
    )

    instance_id = os.getenv("CDSE_INSTANCE_ID", "")
    if not instance_id:
        st.error("Brak CDSE_INSTANCE_ID w .env")
    else:
        col_l, col_r = st.columns([1, 2])
        with col_l:
            area_llm = st.selectbox(
                "Obszar", list(WROCLAW_BBOXES.keys()), key="llm_area"
            )
            date_llm = st.date_input(
                "Data sceny", value=datetime(2026, 3, 22), key="llm_date"
            )
            n_patches = st.slider("Liczba patchów do porównania", 4, 16, 8)
            llm_model = st.selectbox(
                "Model LLM",
                [
                    "qwen3-vl:235b-instruct-cloud",
                    "gemma4:31b-cloud",
                    "mistral-large-3:675b-cloud",
                    "kimi-k2.5:cloud",
                ],
                key="llm_model",
            )
            run_llm = st.button("🔍 Porównaj", type="primary")
        with col_r:
            st.markdown("**Metodologia:**")
            st.markdown("1. Pobieramy scenę Sentinel-2 z CDSE (Copernicus Data Space)")
            st.markdown("2. Kroimy na patche 256×256 px")
            st.markdown("3. Każdy patch klasyfikuje SegFormer Esri LULC (DL) i wybrany LLM vision")
            st.markdown("4. Porównujemy wyniki — ✅ zgodność, ❌ rozbieżność")

        if run_llm:
            lon_min, lat_min, lon_max, lat_max = WROCLAW_BBOXES[area_llm]
            date_str = date_llm.strftime("%Y-%m-%d")
            # Pobierz mały tile do wyświetlania
            with st.spinner("Pobieranie sceny..."):
                img_display = cdse.fetch_wms_preview(
                    instance_id,
                    lon_min,
                    lat_min,
                    lon_max,
                    lat_max,
                    layer="TRUE-COLOR",
                    size=512,
                    date=date_str,
                )
                img_full = cdse.fetch_wms_preview(
                    instance_id,
                    lon_min,
                    lat_min,
                    lon_max,
                    lat_max,
                    layer="TRUE-COLOR",
                    size=2500,
                    date=date_str,
                )
            if img_display is None or img_full is None:
                st.error("Błąd pobierania sceny")
            else:
                arr = np.array(img_full.convert("RGB"))
                n_rows = arr.shape[0] // IMAGE_SIZE_GEO
                n_cols = arr.shape[1] // IMAGE_SIZE_GEO
                all_pos = [(r, c) for r in range(n_rows) for c in range(n_cols)]
                random.seed(42)
                positions = random.sample(all_pos, min(n_patches, len(all_pos)))
                st.session_state["llm_model_selected"] = llm_model
                st.session_state["llm_data"] = {
                    "img": img_display,
                    "arr": arr,
                    "positions": positions,
                    "date": date_str,
                    "area": area_llm,
                }

        if "llm_data" in st.session_state:
            data = st.session_state["llm_data"]
            img = data["img"]
            arr = data["arr"]
            positions = data["positions"]

            results = []
            agree = 0
            progress = st.progress(0, text="Klasyfikacja...")

            for i, (r, c) in enumerate(positions):
                patch = arr[
                    r * IMAGE_SIZE_GEO : (r + 1) * IMAGE_SIZE_GEO,
                    c * IMAGE_SIZE_GEO : (c + 1) * IMAGE_SIZE_GEO,
                ]
                patch_pil = Image.fromarray(patch)

                # SegFormer WMS
                inp = preprocess_patch(patch, size=IMAGE_SIZE_GEO)
                logits = session_wms.run(
                    ["logits"],
                    {
                        "x"
                        if "x" in [i.name for i in session_wms.get_inputs()]
                        else "pixel_values": inp
                    },
                )[0][0]
                probs = np.exp(logits) / np.exp(logits).sum()
                dl_class = CLASSES[int(np.argmax(probs))]
                dl_conf = float(probs.max())

                # qwen3-vl
                llm_class, llm_reason = classify_patch_with_llm(patch_pil, ollama, model=st.session_state.get("llm_model_selected", llm_model))

                match = dl_class == llm_class
                if match:
                    agree += 1

                results.append(
                    {
                        "patch": patch_pil,
                        "dl_class": dl_class,
                        "dl_conf": dl_conf,
                        "llm_class": llm_class,
                        "llm_reason": llm_reason,
                        "match": match,
                    }
                )
                progress.progress(
                    (i + 1) / len(positions), text=f"Patch {i + 1}/{len(positions)}..."
                )

            progress.empty()

            agreement_pct = agree / len(results)
            st.metric(
                "Zgodność SegFormer vs " + st.session_state.get("llm_model_selected", "LLM").split(":")[0],
                f"{agreement_pct:.1%}",
                help="Procent patchów gdzie oba modele wskazały tę samą klasę",
            )

            # Warstwy referencyjne
            with st.expander("🗺️ Warstwy referencyjne — BDOT10k / Ortofoto / MPZP"):
                from src.bdot10k_client import (
                    fetch_bdot10k_area, fetch_mpzp_przeznaczenie,
                    detect_mpzp_discrepancies, detect_bdot_discrepancies,
                )
                from src.ortofoto_client import fetch_ortofoto_area

                lon_min_l, lat_min_l, lon_max_l, lat_max_l = WROCLAW_BBOXES[data["area"]]
                col_l1, col_l2 = st.columns(2)
                with col_l1:
                    with st.spinner("BDOT10k..."):
                        bdot_l = fetch_bdot10k_area(lon_min_l, lat_min_l, lon_max_l, lat_max_l)
                    if bdot_l:
                        st.image(bdot_l, caption="BDOT10k (GUGiK)", use_container_width=True)
                with col_l2:
                    with st.spinner("Ortofoto 2025..."):
                        orto_l = fetch_ortofoto_area(lon_min_l, lat_min_l, lon_max_l, lat_max_l)
                    if orto_l:
                        st.image(orto_l, caption="Ortofoto 2025 (UM Wrocław)", use_container_width=True)

                with st.spinner("MPZP..."):
                    mpzp_l = fetch_mpzp_przeznaczenie(lon_min_l, lat_min_l, lon_max_l, lat_max_l)

                # Dominant klasa z wyników DL
                dl_classes = {}
                for res in results:
                    dl_classes[res["dl_class"]] = dl_classes.get(res["dl_class"], 0) + 1
                dominant_dl = max(dl_classes, key=dl_classes.get) if dl_classes else "Residential"
                dl_stats = {k: v/len(results) for k, v in dl_classes.items()}

                col_l3, col_l4 = st.columns(2)
                with col_l3:
                    st.markdown("**BDOT10k:**")
                    for a in detect_bdot_discrepancies(dl_stats, bdot_l):
                        st.markdown(a)
                with col_l4:
                    st.markdown("**MPZP:**")
                    for a in detect_mpzp_discrepancies(dl_stats, mpzp_l):
                        st.markdown(a)

            cols_per_row = 2
            for i in range(0, len(results), cols_per_row):
                row_results = results[i : i + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, res in zip(cols, row_results):
                    col.image(res["patch"], width=256)
                    dl_color = "#{:02x}{:02x}{:02x}".format(
                        *CLASS_COLORS[CLASSES.index(res["dl_class"])]
                    )
                    col.markdown(
                        f"**DL:** <span style='color:{dl_color}'>■</span> "
                        f"{res['dl_class']} ({res['dl_conf']:.0%})",
                        unsafe_allow_html=True,
                    )
                    llm_color = "#{:02x}{:02x}{:02x}".format(
                        *CLASS_COLORS[CLASSES.index(res["llm_class"])]
                    )
                    match_icon = "✅" if res["match"] else "❌"
                    col.markdown(
                        f"**LLM:** <span style='color:{llm_color}'>■</span> "
                        f"{res['llm_class']} {match_icon}",
                        unsafe_allow_html=True,
                    )
                    if not res["match"] and res["llm_reason"]:
                        col.caption(res["llm_reason"][:80] + "...")
