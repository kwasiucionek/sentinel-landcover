# 🛰️ Sentinel Land Cover Analyzer

Narzędzie do automatycznej klasyfikacji i monitoringu pokrycia terenu
na podstawie zobrazowań satelitarnych Sentinel-2, zbudowane z myślą
o zastosowaniach w systemach geoinformacyjnych — geoportalach miejskich,
monitoringu środowiskowym, gospodarce wodnej i zarządzaniu kryzysowym.



---

## Motywacja i zastosowania biznesowe

GISPartner realizuje projekty w obszarach, które bezpośrednio korzystają
z automatycznej analizy danych satelitarnych:

### 🏗️ Monitoring nielegalnej eksploatacji surowców
Automatyczna klasyfikacja pokrycia terenu z Sentinel-2 pozwala wykryć
nowe wyrobiska i zmiany w użytkowaniu gruntu bez kosztownych pomiarów
naziemnych. Change detection między latami identyfikuje obszary gdzie
klasa "Bare Ground" nagle zastępuje "Forest" lub "Cropland".

### 💧 Gospodarka wodna i zagrożenia hydrologiczne
Wskaźnik NDWI (Normalized Difference Water Index) obliczany na bieżąco
z danych Sentinel-2 umożliwia monitoring zbiorników wodnych i wczesne
wykrywanie podtopień — wprost wspierając systemy zarządzania ryzykiem
powodziowym.

### 🗺️ Geoportal Wrocławia i planowanie przestrzenne
Analiza pokrycia terenu dostarcza aktualnych danych o urbanizacji,
lesistości i powierzchniach zieleni — kluczowych dla geoportali
miejskich i systemów planowania przestrzennego.

### 🚨 Zarządzanie kryzysowe
Szybka klasyfikacja zobrazowań po zdarzeniach ekstremalnych (powódź,
pożar) pozwala ocenić zasięg szkód i wspiera służby ratownicze.

---

## Architektura

```
Dane Sentinel-2 (Copernicus Data Space)
        ↓
┌─────────────────────────────────────────┐
│  XGBoost baseline    — cechy spektralne │
│  SegFormer-B2        — RGB 256px        │
│  SegFormer Esri LULC — fine-tuning WMS  │
│  ResNet50 SSL4EO-S12 — 3M scen S2       │
└─────────────────────────────────────────┘
        ↓
   Eksport ONNX → inferencja CPU (VPS)
        ↓
┌─────────────────────────────────────────┐
│  PostGIS  — geometrie bbox EPSG:4326    │
│  Qdrant   — embeddingi (RAG)            │
│  kimi-k2.5 via Ollama Cloud — LLM RAG   │
│  gemma4 / qwen3-vl / mistral — LLM vis  │
└─────────────────────────────────────────┘
        ↓
   Streamlit dashboard (6 zakładek)
```

---

## Wyniki modeli

| Model | Macro F1 | Dane treningu | Etykiety |
|---|---|---|---|
| XGBoost (baseline) | 0.58 | EuroSAT cechy spektralne | prawdziwe |
| SegFormer-B2 (stary) | 0.71 | EuroSAT 64px, fake NIR | prawdziwe |
| SegFormer WMS pseudo | 0.86* | EuroSAT + WMS Wrocław | pseudo (hint) |
| ResNet50 SSL4EO | 0.66 | EuroSAT + WMS | prawdziwe + hint |
| **SegFormer EuroSAT v2** | **0.97** | EuroSAT 256px, RGB | prawdziwe |
| **SegFormer Esri LULC** | **0.78** | WMS Wrocław | **Esri LULC (prawdziwe)** |

*wynik zawyżony przez spójność dystrybucji val/train

**Kluczowe ulepszenia pipeline v2:**
- Usunięto fake NIR (`nir = arr.mean()`) → 3 kanały RGB
- Zwiększono patch z 64×64 → 256×256 px (+10 p.p. dokładności)
- Focal Loss zamiast CrossEntropy (gamma=2.0, label_smoothing=0.1)
- Augmentacje: ColorJitter, GaussianBlur, flip, rotation
- Etykiety Esri LULC (Impact Observatory) zamiast pseudo-labeli

**Uwaga o domain shift:** model trenowany na scenach EuroSAT (lato)
gorzej radzi sobie ze scenami zimowymi/wiosennymi (brak liści, brązowe
pola). Modele LLM vision są odporne na ten efekt dzięki rozumieniu
kontekstu wizualnego.

**Change detection szum** (te same warunki, 2 tygodnie różnicy):
- SegFormer EuroSAT: 21.9% — wysoki szum
- SegFormer Esri LULC: **5.9%** — po domain adaptation

---

## Porównanie LLM vs DL

Eksperyment klasyfikacji patch-by-patch (256×256px, sceny Sentinel-2):

| Model | Zgodność z Esri LULC | Szybkość | Uwagi |
|---|---|---|---|
| SegFormer Esri LULC | referencja | ~50ms/patch | wrażliwy na sezon |
| kimi-k2.5 (cloud) | ~25% | ~3s/patch | dobry ogólnie |
| **gemma4:31b (cloud)** | **~25-40%** | ~2s/patch | **najlepszy vision** |
| qwen3-vl:235b (cloud) | ~12% | ~4s/patch | słabszy na EO |
| mistral-large-3 (cloud) | ~15% | ~3s/patch | średni |

**Wniosek:** Gemma 4 (31B) przewyższa znacznie większe modele na
zadaniach klasyfikacji satelitarnej. Optymalne rozwiązanie produkcyjne
to ensemble — DL jako szybki klasyfikator pierwszego rzutu, LLM do
weryfikacji trudnych przypadków i generowania raportów.

---

## Funkcjonalności

### 📡 Analiza satelitarna
- Wgranie obrazu PNG/JPG
- Klasyfikacja siatką patchów 256×256 px
- Rozkład 10 klas pokrycia terenu EuroSAT
- NDVI, indeks Shannona, pewność modelu
- Porównanie z poprzednią analizą lokalizacji

### 🗺️ Geoportal (Copernicus Data Space)
- Pobieranie scen Sentinel-2 na żywo dla Wrocławia
- 5 predefiniowanych obszarów (Centrum, Psie Pole, Fabryczna, Krzyki, Odra)
- Warstwy: TRUE-COLOR, FALSE-COLOR, NDVI
- Wyszukiwanie dostępnych scen z metadanymi
- Warstwa referencyjna Esri LULC (Impact Observatory, 2017-2024)

### 🔄 Change Detection
- Porównanie dwóch dat z tego samego obszaru
- Mapa zmian z kolorowym kodowaniem
- Automatyczny raport AI (kimi-k2.5) w języku polskim
- Ostrzeżenia o warunkach atmosferycznych

### 🤖 LLM vs DL
- Klasyfikacja patch-by-patch: SegFormer Esri vs wybrany LLM vision
- Wybór modelu: gemma4, qwen3-vl, mistral-large-3, kimi-k2.5
- Porównanie zgodności z uzasadnieniem LLM
- Analiza rozbieżności i domain shift

### 📋 Historia analiz (RAG + PostGIS)
- Każda analiza zapisywana w PostGIS z geometrią bbox EPSG:4326
- Wyszukiwanie przestrzenne (ST_DWithin, ST_Intersects)
- Semantic search po historii (Qdrant)
- Chat w języku naturalnym z historią obserwacji

---

## Stos technologiczny

| Kategoria | Technologie |
|---|---|
| Dane EO | `torchgeo`, `rasterio`, `rioxarray`, `xarray`, `geopandas` |
| Źródła danych | Copernicus Data Space (CDSE WMS), Esri Living Atlas LULC |
| ML/DL | `PyTorch`, `transformers` (SegFormer-B2), `XGBoost`, `timm` |
| Foundation models | ResNet50 SSL4EO-S12 (3M scen Sentinel-2, MoCo) |
| Inferencja | `onnxruntime` (CPU, bez GPU na produkcji) |
| LLM / Vision | `gemma4:31b`, `qwen3-vl:235b`, `kimi-k2.5` via Ollama Cloud |
| RAG | `Qdrant`, `sentence-transformers` |
| Baza danych | `PostGIS 16` (geometrie), `Qdrant` (wektory) |
| Konteneryzacja | `Docker`, `docker-compose` |
| Frontend | `Streamlit` |
| Środowisko | `conda` (conda-forge dla GDAL/PROJ) |

---

## Instalacja

### Lokalnie (trening + development)

```bash
git clone https://github.com/<your-username>/sentinel-landcover
cd sentinel-landcover

conda create -n sentinel-landcover python=3.11
conda activate sentinel-landcover
conda install -c conda-forge "rasterio<1.4" rioxarray geopandas shapely -y

# PyTorch z CUDA (Blackwell RTX 5090)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

### Docker (produkcja / VPS, tylko CPU)

```bash
cp .env.example .env
# Uzupełnij: CDSE_CLIENT_ID, CDSE_CLIENT_SECRET, CDSE_INSTANCE_ID
# Opcjonalnie: OLLAMA_HOST, OLLAMA_MODEL

docker compose up -d
```

Aplikacja dostępna pod `http://localhost:8501`.

---

## Użycie

```bash
# Baseline XGBoost
python -m src.baseline

# Trening SegFormer v2 (RGB 256px, Focal Loss)
python -m src.train

# Trening ResNet50 SSL4EO
python scripts/train_geo.py

# Zbieranie danych WMS Wrocław
python scripts/collect_wms_dataset.py
python scripts/collect_wms_extra.py

# Labelowanie Esri LULC
python scripts/label_wms_from_esri.py

# Fine-tuning na etykietach Esri
python scripts/finetune_esri.py

# Eksport próbek z EuroSAT
python scripts/export_samples.py
```

---

## Struktura projektu

```
sentinel-landcover/
├── src/
│   ├── dataset.py           # EuroSAT wrapper, transformacje RGB 256px
│   ├── model.py             # SegFormer-B2, eksport ONNX
│   ├── model_geo.py         # ResNet50 SSL4EO-S12
│   ├── train.py             # Trening + Focal Loss
│   ├── baseline.py          # XGBoost baseline
│   ├── cdse_client.py       # Copernicus Data Space API
│   ├── esri_lulc_client.py  # Esri Living Atlas LULC
│   └── rag/
│       ├── store_postgis.py # PostGIS — geometrie analiz
│       ├── indexer.py       # Qdrant — embeddingi
│       └── retriever.py     # RAG + LLM vision
├── app/
│   └── app.py               # Streamlit (6 zakładek)
├── scripts/
│   ├── collect_wms_dataset.py
│   ├── collect_wms_extra.py
│   ├── label_wms_from_esri.py
│   ├── finetune_esri.py
│   ├── train_geo.py
│   └── export_samples.py
├── data/
│   ├── raw/                 # EuroSAT dataset (27k TIF)
│   ├── processed/           # Modele ONNX, checkpointy
│   ├── wms_dataset/         # Dane WMS Wrocław (2688 patchów)
│   └── analyses/            # Historia analiz (PNG maski)
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Znane ograniczenia i dalszy rozwój

### Ograniczenia modelu
Model trenowany na EuroSAT (lato) wykazuje domain shift przy scenach
zimowych/wiosennych. Fine-tuning na danych WMS z etykietami Esri LULC
zmniejszył szum change detection z 21.9% do 5.9%. Wiarygodne wyniki
change detection wymagają scen z podobnym kątem słońca i sezonem.

### Możliwe rozszerzenia

- [ ] Pixel-wise segmentation (U-Net zamiast patch classification)
- [ ] Pobieranie danych przez STAC (`pystac-client`)
- [ ] Wieloczasowe serie Sentinel-2 (Prithvi-EO-2.0 foundation model)
- [ ] Integracja danych SAR Sentinel-1 (widzenie przez chmury)
- [ ] Orkiestracja pipeline ETL (Prefect lub Dagster)
- [ ] Monitoring modelu (Prometheus + Grafana)
- [ ] Korekcja atmosferyczna (Sen2Cor) przed klasyfikacją
- [ ] Dane hiperspektralne (KP Labs Intuition-1)

---

## Licencja

MIT — dane EuroSAT: [Helber et al. 2019](https://doi.org/10.1109/jstars.2019.2918242)

Dane Sentinel-2: © ESA / Copernicus Data Space Ecosystem

Warstwa LULC: © Impact Observatory, Microsoft, Esri (CC BY 4.0)
