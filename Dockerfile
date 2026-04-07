FROM python:3.11-slim

WORKDIR /app

# Zależności systemowe (GDAL/PROJ bez conda)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libnetcdf-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Zależności Python
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Kod aplikacji
COPY src/ ./src/
COPY app/ ./app/
COPY data/processed/model.onnx ./data/processed/model.onnx

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
