"""Build and open a local Chroma index from approved, processed chunks."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .embeddings import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEncoder
from .manifest import ALLOWED_CONDITIONS, ALLOWED_STAGES

DEFAULT_CHUNKS_PATH = Path("data/local/processed/chunks.jsonl")
DEFAULT_INDEX_DIR = Path("data/local/chroma")
DEFAULT_COLLECTION_NAME = "patient_journey_chunks"
INDEX_SCHEMA_VERSION = 1

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "source_id",
    "condition",
    "stage_scope",
    "title",
    "publisher",
    "source_url",
    "source_content_sha256",
    "text",
}


def _import_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB is required to build or open the index. "
            "Install the project requirements with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return chromadb


def load_chunk_records(path: str | Path = DEFAULT_CHUNKS_PATH) -> list[dict[str, Any]]:
    """Load and validate JSONL chunks without changing the generated corpus."""
    chunks_path = Path(path)
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"Processed chunks not found at {chunks_path}. Run the approved-source "
            "ingestion command first."
        )

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        chunks_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on chunks line {line_number}.") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Chunks line {line_number} must contain a JSON object.")

        missing = sorted(
            field for field in REQUIRED_CHUNK_FIELDS
            if field not in record or record[field] in (None, "")
        )
        if missing:
            raise ValueError(
                f"Chunks line {line_number} is missing required fields: "
                + ", ".join(missing)
            )
        chunk_id = record["chunk_id"]
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id {chunk_id!r} in processed corpus.")
        seen_ids.add(chunk_id)

        if record["condition"] not in ALLOWED_CONDITIONS:
            raise ValueError(
                f"Chunk {chunk_id!r} has unsupported condition {record['condition']!r}."
            )
        stages = record["stage_scope"]
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"Chunk {chunk_id!r} must have a non-empty stage_scope list.")
        if set(stages) - ALLOWED_STAGES:
            raise ValueError(f"Chunk {chunk_id!r} contains unsupported stage labels.")
        if not isinstance(record["text"], str) or not record["text"].strip():
            raise ValueError(f"Chunk {chunk_id!r} has empty text.")

        records.append(record)

    if not records:
        raise ValueError(f"No usable chunks found in {chunks_path}.")
    return records


def chunk_to_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert chunk provenance to Chroma-compatible scalar metadata."""
    stages = list(chunk["stage_scope"])
    metadata: dict[str, str | int | float | bool] = {
        "source_id": str(chunk["source_id"]),
        "condition": str(chunk["condition"]),
        "title": str(chunk["title"]),
        "publisher": str(chunk["publisher"]),
        "source_url": str(chunk["source_url"]),
        "source_content_sha256": str(chunk["source_content_sha256"]),
        "stage_scope_csv": "|".join(stages),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "word_count": int(chunk.get("word_count", len(chunk["text"].split()))),
    }

    for field in ("source_type", "extraction_mode", "checked_date", "fetched_at"):
        value = chunk.get(field)
        if value not in (None, ""):
            metadata[field] = str(value)

    # Boolean fields permit a simple, explicit filter for one selected stage.
    for stage in sorted(ALLOWED_STAGES):
        metadata[f"stage_{stage}"] = stage in stages
    return metadata


def build_persistent_index(
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 32,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Create a fresh local Chroma collection from the current processed corpus.

    Existing collection records are replaced so chunks removed during source
    extraction cannot remain as stale index entries. Embeddings are computed
    before the existing collection is replaced.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    chunks = load_chunk_records(chunks_path)
    embedding_model = encoder or SentenceTransformerEncoder(model_name)

    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch_texts = [item["text"] for item in chunks[start:start + batch_size]]
        batch_vectors = embedding_model.encode(batch_texts)
        if len(batch_vectors) != len(batch_texts):
            raise ValueError("Embedding model returned a different number of vectors than texts.")
        if batch_vectors and any(len(vector) != len(batch_vectors[0]) for vector in batch_vectors):
            raise ValueError("Embedding model returned vectors with inconsistent dimensions.")
        vectors.extend(batch_vectors)

    if len(vectors) != len(chunks):
        raise ValueError("Not every corpus chunk received an embedding.")

    chromadb = _import_chromadb()
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_path))

    listed = client.list_collections()
    existing_names = {
        item if isinstance(item, str) else item.name
        for item in listed
    }
    if collection_name in existing_names:
        client.delete_collection(name=collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": model_name,
            "index_schema_version": INDEX_SCHEMA_VERSION,
        },
    )

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            ids=[item["chunk_id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[chunk_to_metadata(item) for item in batch],
            embeddings=vectors[start:start + batch_size],
        )

    return {
        "index_dir": str(index_path),
        "collection": collection_name,
        "embedding_model": model_name,
        "distance_metric": "cosine",
        "chunks_indexed": len(chunks),
        "embedding_dimensions": len(vectors[0]),
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def open_persistent_collection(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Open an existing collection and return it with its recorded model name."""
    chromadb = _import_chromadb()
    index_path = Path(index_dir)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index directory {index_path} does not exist. Build the index first."
        )

    client = chromadb.PersistentClient(path=str(index_path))
    listed = client.list_collections()
    names = {item if isinstance(item, str) else item.name for item in listed}
    if collection_name not in names:
        raise FileNotFoundError(
            f"Collection {collection_name!r} was not found in {index_path}. "
            "Build the index first."
        )

    collection = client.get_collection(name=collection_name)
    metadata = collection.metadata or {}
    recorded_model = metadata.get("embedding_model")
    if not recorded_model:
        raise ValueError(
            "The collection does not record its embedding model. Rebuild the index."
        )
    return collection, str(recorded_model)
