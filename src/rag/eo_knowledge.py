"""Baza wiedzy EO — dokumenty do RAG."""

DOCUMENTS = [
    {
        "title": "Klasy pokrycia terenu EuroSAT",
        "content": """EuroSAT definiuje 10 klas pokrycia terenu na podstawie zobrazowań Sentinel-2:
- AnnualCrop: uprawy jednoroczne (zboża, rzepak, kukurydza). Charakterystyczne prostokątne poletka, brązowo-beżowe w marcu, zielone latem.
- Forest: lasy iglaste i liściaste. Ciemnozielona tekstura, nieregularne granice.
- HerbaceousVegetation: łąki, nieużytki, niska roślinność. Jasnozielone, jednolite.
- Highway: drogi, autostrady. Szare linie, regularne.
- Industrial: tereny przemysłowe, magazyny, fabryki. Duże szarobrązowe budynki.
- Pasture: pastwiska. Zielone, często otoczone ogrodzeniem.
- PermanentCrop: sady, winnice. Regularne rzędy roślin.
- Residential: zabudowa mieszkaniowa. Gęste skupisko budynków, dachy, ulice.
- River: rzeki, kanały. Niebieskie linie wodne.
- SeaLake: jeziora, zbiorniki wodne. Duże niebieskie obszary."""
    },
    {
        "title": "Indeksy spektralne Sentinel-2",
        "content": """Kluczowe indeksy spektralne obliczane z danych Sentinel-2:
- NDVI (Normalized Difference Vegetation Index): (NIR - RED) / (NIR + RED). Wartości: <0 woda, 0-0.2 gleba/skały, 0.2-0.4 niska roślinność, 0.4-0.7 zdrowa roślinność, >0.7 gęsta roślinność. Pasma: B8 (NIR, 842nm) i B4 (RED, 665nm).
- NDWI (Normalized Difference Water Index): (GREEN - NIR) / (GREEN + NIR). Wartości >0 wskazują wodę. Pasma: B3 (GREEN, 560nm) i B8 (NIR).
- NDBI (Normalized Difference Built-up Index): (SWIR - NIR) / (SWIR + NIR). Wartości >0 wskazują zabudowę. Pasma: B11 (SWIR, 1610nm) i B8.
- EVI (Enhanced Vegetation Index): 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1). Lepszy niż NDVI na gęstą roślinność."""
    },
    {
        "title": "Pasma spektralne Sentinel-2",
        "content": """Sentinel-2 MSI (MultiSpectral Instrument) posiada 13 pasm spektralnych:
- B1 (443nm, 60m): aerozole
- B2 (490nm, 10m): niebieski (BLUE)
- B3 (560nm, 10m): zielony (GREEN)
- B4 (665nm, 10m): czerwony (RED)
- B5 (705nm, 20m): red edge 1
- B6 (740nm, 20m): red edge 2
- B7 (783nm, 20m): red edge 3
- B8 (842nm, 10m): bliski podczerwień NIR (najważniejszy)
- B8A (865nm, 20m): NIR narrow
- B9 (945nm, 60m): para wodna
- B10 (1375nm, 60m): cirrus
- B11 (1610nm, 20m): SWIR 1
- B12 (2190nm, 20m): SWIR 2
Rozdzielczość 10m dla B2/B3/B4/B8 — najlepsza do klasyfikacji pokrycia terenu."""
    },
    {
        "title": "Produkty Sentinel-2 L1C i L2A",
        "content": """Sentinel-2 dostarcza dwa poziomy przetwarzania:
- L1C (Top-of-Atmosphere): surowe dane z korekcją geometryczną, reflektancja ToA. Wymaga korekcji atmosferycznej przed analizą pokrycia terenu.
- L2A (Bottom-of-Atmosphere): po korekcji atmosferycznej Sen2Cor, reflektancja BoA. Gotowy do klasyfikacji pokrycia terenu. Dostępny przez Copernicus Data Space Ecosystem (CDSE).
Copernicus Data Space Ecosystem (CDSE) udostępnia dane przez API STAC, OData i WMS."""
    },
    {
        "title": "Domain shift w klasyfikacji Sentinel-2",
        "content": """Domain shift to problem gdy model trenowany na danych z jednego okresu gorzej działa na danych z innego okresu:
- Model trenowany na EuroSAT (lato, Europa Zachodnia) gorzej klasyfikuje sceny zimowe/wiosenne.
- Pola uprawne w marcu są brązowe (po orce) — model może je mylić z Bare Ground lub Industrial.
- Lasy liściaste bez liści (zima) mają mniejsze NDVI i wyglądają inaczej niż latem.
- Fine-tuning na lokalnych danych (np. Esri LULC dla Wrocławia) redukuje domain shift.
- Change detection wymaga scen z podobnymi warunkami atmosferycznymi i porą roku."""
    },
    {
        "title": "Change Detection — wykrywanie zmian pokrycia terenu",
        "content": """Change detection porównuje klasyfikację pokrycia terenu między dwiema datami:
- Zalecane: ten sam miesiąc, różne lata (np. marzec 2025 vs marzec 2026).
- Szum change detection: procent patchów zmieniających klasę przy braku rzeczywistych zmian.
- SegFormer EuroSAT: szum ~22%, SegFormer Esri LULC: szum ~6%.
- Istotne zmiany: >5% powierzchni, trwałe przez kilka obserwacji.
- Typowe wykrywane zmiany: wylesianie, nowa zabudowa, zmiany rolnicze, podtopienia."""
    },
    {
        "title": "Indeks Shannona różnorodności pokrycia terenu",
        "content": """Indeks Shannona (H) mierzy różnorodność klas pokrycia terenu w obszarze:
H = -sum(p_i * log(p_i)) gdzie p_i to udział klasy i.
- H=0: jeden typ pokrycia (np. czysta woda lub monokultura).
- H=1-2: umiarkowana różnorodność (np. miasto z parkami).
- H>2: wysoka różnorodność (np. mozaika rolna z lasami i wodą).
Dla Wrocławia typowe wartości: Centrum ~1.8-2.2, tereny rolne ~0.5-1.0."""
    },
    {
        "title": "Wrocław — obszary i charakterystyka pokrycia terenu",
        "content": """Charakterystyka obszarów Wrocławia w klasyfikacji Sentinel-2:
- Centrum: dominuje Residential (60-70%), Highway (15%), rzeka Odra (River/SeaLake 5-10%).
- Psie Pole: mix Residential (50%) i AnnualCrop (30%) — obszar peri-urban.
- Fabryczna: Industrial (35%), Residential (40%), Airport (Highway).
- Krzyki: Residential (65%), HerbaceousVegetation (15%) — parki i ogrody.
- Odra (centrum): River/SeaLake (20%), Residential (50%), Forest (wyspy).
Wrocław leży na Nizinie Śląskiej, rzeka Odra dzieli miasto na części."""
    },
    {
        "title": "BDOT10k — Baza Danych Obiektów Topograficznych",
        "content": """BDOT10k to wektorowa baza danych topograficznych GUGiK dla Polski:
- Skala 1:10 000, aktualizowana przez Główny Urząd Geodezji i Kartografii.
- Zawiera: budynki, drogi, rzeki, lasy, tereny zielone, linie kolejowe.
- Dostępna bezpłatnie przez WMTS: mapy.geoportal.gov.pl/wss/service/WMTS/guest/wmts/BDOT10k-BDOO
- Układ współrzędnych: EPSG:2180 (PL-1992).
- Klasy obiektów: PT_LAS (las), BU_BUDYNEK (budynek), SK_RZEKA (rzeka), SK_JEZIORO (jezioro)."""
    },
    {
        "title": "MPZP — Miejscowe Plany Zagospodarowania Przestrzennego Wrocławia",
        "content": """MPZP to oficjalne dokumenty planistyczne określające przeznaczenie terenu:
- Dane UM Wrocław dostępne przez WFS: gis1.um.wroc.pl/arcgis/services/ogc/OGC_mpzp/MapServer/WFSServer
- Klasy przeznaczenia: zabudowa mieszkaniowa jednorodzinna/wielorodzinna, usługi, komunikacja publiczna, aktywność gospodarcza, zieleń, parki.
- Rozbieżność między klasyfikacją Sentinel a MPZP może wskazywać na nielegalną zmianę użytkowania.
- Warstwa przeznaczenie_terenu_-_uproszczona_klasyfikacja zawiera uproszczone kategorie."""
    },
    {
        "title": "EGiB — Ewidencja Gruntów i Budynków",
        "content": """EGiB to oficjalny rejestr działek ewidencyjnych i budynków w Polsce:
- Prowadzony przez starostów powiatowych, koordynowany przez GUGiK.
- Dostępny bezpłatnie przez WFS: mapy.geoportal.gov.pl/wss/service/PZGIK/EGIB/WFS/UslugaZbiorcza
- Bezpłatnie dostępne: geometria działek + identyfikatory.
- Klasy użytku gruntu (płatne): grunty orne (R), lasy (Ls), zabudowa (B), wody (W).
- Działki ewidencyjne identyfikowane przez: numer obrębu + numer działki."""
    },
]
