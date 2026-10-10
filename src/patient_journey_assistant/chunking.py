"""Deterministic text chunking and provenance-preserving IDs."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from .manifest import SourceRecord
from .text_processing import normalize_text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_chunk_id(source_id: str, content_sha256: str, chunk_index: int, chunk_text: str) -> str:
    """Stable for the same source revision, chunk index, and exact chunk text."""
    identity = f"{source_id}|{content_sha256}|{chunk_index}|{chunk_text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def chunk_source_text(
    source: SourceRecord,
    text: str,
    *,
    max_words: int = 180,
    overlap_words: int = 30,
    fetched_at: str | None = None,
) -> list[dict[str, Any]]:
    """Create deterministic overlapping word-window chunks with source metadata."""
    if max_words < 20:
        raise ValueError("max_words must be at least 20.")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be >= 0 and smaller than max_words.")

    normalized = normalize_text(text)
    words = normalized.split()
    if not words:
        return []

    content_hash = sha256_text(normalized)
    step = max_words - overlap_words
    timestamp = fetched_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    chunks: list[dict[str, Any]] = []

    for chunk_index, start in enumerate(range(0, len(words), step)):
        segment = words[start:start + max_words]
        if not segment:
            break
        chunk_text = " ".join(segment)
        chunks.append({
            "chunk_id": stable_chunk_id(source.source_id, content_hash, chunk_index, chunk_text),
            "source_id": source.source_id,
            "source_content_sha256": content_hash,
            "chunk_index": chunk_index,
            "condition": source.condition,
            "stage_scope": list(source.stage_scope),
            "title": source.title,
            "publisher": source.publisher,
            "source_url": source.url,
            "source_type": source.source_type,
            "extraction_mode": source.extraction_mode,
            "checked_date": source.checked_date.isoformat(),
            "fetched_at": timestamp,
            "word_count": len(segment),
            "text": chunk_text,
        })
        if start + max_words >= len(words):
            break

    return chunks
