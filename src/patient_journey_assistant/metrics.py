"""Small, transparent retrieval metrics for the annotated benchmark."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from statistics import mean
from typing import Any


def _validate_inputs(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> set[str]:
    if k < 1:
        raise ValueError("k must be at least 1.")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Each answerable benchmark question needs a relevant chunk ID.")
    if any(not isinstance(item, str) or not item for item in retrieved_ids):
        raise ValueError("Retrieved chunk IDs must be non-empty strings.")
    if any(not isinstance(item, str) or not item for item in relevant):
        raise ValueError("Relevant chunk IDs must be non-empty strings.")
    return relevant


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int = 5,
) -> float:
    """Fraction of all labelled relevant chunks retrieved within the first k."""
    relevant = _validate_inputs(retrieved_ids, relevant_ids, k)
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int = 5,
) -> float:
    """Reciprocal rank of the first relevant chunk within the first k, else zero."""
    relevant = _validate_inputs(retrieved_ids, relevant_ids, k)
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def retrieval_coverage(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int = 5,
) -> float:
    """Return 1.0 when at least one relevant chunk is in top-k, otherwise 0.0."""
    relevant = _validate_inputs(retrieved_ids, relevant_ids, k)
    return float(any(chunk_id in relevant for chunk_id in retrieved_ids[:k]))


def evaluate_rankings(
    retrieved_by_query: Mapping[str, Sequence[str]],
    relevant_by_query: Mapping[str, Iterable[str]],
    k: int = 5,
) -> dict[str, Any]:
    """Macro-average per-query Recall@k, MRR@k, and retrieval coverage."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not retrieved_by_query:
        raise ValueError("At least one benchmark query is required.")
    if set(retrieved_by_query) != set(relevant_by_query):
        raise ValueError("Retrieved and relevant mappings must contain the same query IDs.")

    per_query: dict[str, dict[str, float]] = {}
    for query_id in sorted(retrieved_by_query):
        retrieved = retrieved_by_query[query_id]
        relevant = relevant_by_query[query_id]
        per_query[query_id] = {
            "recall_at_k": recall_at_k(retrieved, relevant, k),
            "mrr_at_k": reciprocal_rank_at_k(retrieved, relevant, k),
            "covered": retrieval_coverage(retrieved, relevant, k),
        }

    return {
        "query_count": len(per_query),
        "k": k,
        "recall_at_k_macro": mean(item["recall_at_k"] for item in per_query.values()),
        "mrr_at_k_macro": mean(item["mrr_at_k"] for item in per_query.values()),
        "retrieval_coverage": mean(item["covered"] for item in per_query.values()),
        "per_query": per_query,
    }
