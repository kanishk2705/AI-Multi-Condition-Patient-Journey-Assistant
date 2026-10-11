#!/bin/sh
set -eu

CHUNKS_PATH="data/local/processed/chunks.jsonl"
INDEX_DIR="data/local/chroma"
INDEX_HASH_FILE="data/local/.index-corpus-sha256"

mkdir -p data/local/processed data/local

if [ ! -s "$CHUNKS_PATH" ]; then
  echo "No local corpus found. Fetching only source-manifest rows approved by the project policy."
  python -m patient_journey_assistant.ingest \
    --manifest data/source_manifest.csv \
    --output-dir data/local/processed \
    --fetch-approved \
    --delay-seconds 1
fi

if [ ! -s "$CHUNKS_PATH" ]; then
  echo "Ingestion did not produce a non-empty chunk corpus; refusing to start." >&2
  exit 1
fi

CURRENT_CORPUS_HASH="$(sha256sum "$CHUNKS_PATH" | cut -d ' ' -f 1)"
NEED_REBUILD=1

if [ -s "$INDEX_HASH_FILE" ] && [ -f "$INDEX_DIR/chroma.sqlite3" ]; then
  STORED_CORPUS_HASH="$(cat "$INDEX_HASH_FILE")"
  if [ "$STORED_CORPUS_HASH" = "$CURRENT_CORPUS_HASH" ]; then
    NEED_REBUILD=0
  fi
fi

if [ "$NEED_REBUILD" -eq 1 ]; then
  echo "Building the local embedding index for the current approved corpus. First startup may download model weights."
  python -m patient_journey_assistant.build_index \
    --chunks "$CHUNKS_PATH" \
    --index-dir "$INDEX_DIR"
  printf '%s\n' "$CURRENT_CORPUS_HASH" > "$INDEX_HASH_FILE.tmp"
  mv "$INDEX_HASH_FILE.tmp" "$INDEX_HASH_FILE"
fi

PORT_VALUE="${PORT:-8501}"
exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="$PORT_VALUE" \
  --server.headless=true \
  --browser.gatherUsageStats=false
