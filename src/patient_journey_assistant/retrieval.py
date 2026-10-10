"""Comparable baseline and condition/stage-aware semantic retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .embeddings import SentenceTransformerEncoder
from .manifest import ALLOWED_CONDITIONS, ALLOWED_STAGES
from .vector_index import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_INDEX_DIR,
    open_persistent_collection,
)


class QueryEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode text with the same model used to build the index."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved corpus passage and its provenance."""

    chunk_id: str
    similarity: float
    text: str
    metadata: dict[str, Any]


def build_context_filter(condition: str, stage: str) -> dict[str, Any]:
    """Return a Chroma filter for the selected condition and journey stage."""
    if condition not in ALLOWED_CONDITIONS:
        raise ValueError(f"Unsupported condition {condition!r}.")
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"Unsupported journey stage {stage!r}.")
    return {
        "$and": [
            {"condition": {"$eq": condition}},
            {f"stage_{stage}": {"$eq": True}},
        ]
    }


class SemanticRetriever:
    """Retrieve the same embedded corpus using two distinct policies.

    Baseline retrieval does not filter by context. Context-aware retrieval
    filters eligible chunks first and then uses the same embedding model,
    distance metric, and top-k setting. The current curated corpus is small,
    so each query ranks all eligible chunks before deterministic tie-breaking.
    """

    def __init__(self, collection: Any, encoder: QueryEncoder) -> None:
        self.collection = collection
        self.encoder = encoder

    def retrieve_baseline(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        return self._retrieve(query=query, k=k, where=None)

    def retrieve_contextual(
        self,
        query: str,
        *,
        condition: str,
        stage: str,
        k: int = 5,
    ) -> list[RetrievedChunk]:
        where = build_context_filter(condition, stage)
        return self._retrieve(query=query, k=k, where=where)

    def _retrieve(
        self,
        *,
        query: str,
        k: int,
        where: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if k < 1:
            raise ValueError("k must be at least 1.")

        # Chroma's Collection.count() returns the total collection size and
        # does not accept a metadata filter. Use the total as an upper bound
        # for query n_results; Chroma applies the where filter inside query()
        # and returns only the eligible matches.
        corpus_count = self.collection.count()
        if corpus_count < 1:
            return []

        vectors = self.encoder.encode([query.strip()])
        if len(vectors) != 1 or not vectors[0]:
            raise ValueError("The query encoder did not return exactly one non-empty vector.")

        # Rank every eligible item so the top-k boundary is stable if distances tie.
        query_args: dict[str, Any] = {
            "query_embeddings": [vectors[0]],
            "n_results": corpus_count,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_args["where"] = where
        result = self.collection.query(**query_args)

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for position, chunk_id in enumerate(ids):
            if position >= len(documents) or position >= len(metadatas):
                continue
            distance = distances[position] if position < len(distances) else None
            if distance is None:
                raise ValueError("The vector index did not return distances.")
            metadata = dict(metadatas[position] or {})
            stage_csv = metadata.pop("stage_scope_csv", "")
            metadata["stage_scope"] = stage_csv.split("|") if stage_csv else []
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    similarity=1.0 - float(distance),
                    text=str(documents[position] or ""),
                    metadata=metadata,
                )
            )

        retrieved.sort(key=lambda item: (-item.similarity, item.chunk_id))
        return retrieved[:k]


def create_persistent_retriever(
    index_dir: str = str(DEFAULT_INDEX_DIR),
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    model_name: str | None = None,
) -> SemanticRetriever:
    """Open the persisted index with its recorded embedding model."""
    collection, recorded_model = open_persistent_collection(index_dir, collection_name)
    if model_name and model_name != recorded_model:
        raise ValueError(
            f"Index uses {recorded_model!r}, but {model_name!r} was requested. "
            "Use the recorded model or rebuild the index."
        )
    encoder = SentenceTransformerEncoder(recorded_model)
    return SemanticRetriever(collection, encoder)
