FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PJA_INDEX_DIR=data/local/chroma \
    PJA_COLLECTION_NAME=patient_journey_chunks \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-deps .

COPY app.py ./
COPY scripts/container-entrypoint.sh ./scripts/container-entrypoint.sh
COPY data/source_manifest.csv ./data/source_manifest.csv
COPY data/benchmark/ ./data/benchmark/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

ENTRYPOINT ["/bin/sh", "scripts/container-entrypoint.sh"]
