"""Evaluate the no-key extractive fallback against provisional evidence labels.

This is a deterministic citation/relevance diagnostic, not a medical correctness metric.
It writes IDs and aggregate scores, never retrieved text or raw questions.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .answer_generation import answer_question
from .evaluate_retrieval import load_benchmark, validate_relevant_ids
from .manifest import ALLOWED_CONDITIONS, ALLOWED_STAGES
from .retrieval import SemanticRetriever, create_persistent_retriever
from .vector_index import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_INDEX_DIR,
    load_chunk_records,
)

DEFAULT_BENCHMARK = Path("data/benchmark/answerable_questions.json")
DEFAULT_OUTPUT_DIR = Path("data/local/evaluation")


def evaluate_extractive_fallback(
    questions: Sequence[Mapping[str, Any]],
    retriever: SemanticRetriever,
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Compare extractive answer/citation behavior for both retrieval policies."""
    if not questions:
        raise ValueError("At least one question is required.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    outcomes: list[dict[str, Any]] = []

    for question in questions:
        condition, stage = str(question["condition"]), str(question["stage"])
        relevant = set(question["relevant_chunk_ids"])
        if not relevant:
            raise ValueError(f"{question['question_id']} has no labelled relevant passage.")

        pair: dict[str, Any] = {"question_id": str(question["question_id"])}
        for method in ("baseline", "contextual"):
            response = answer_question(
                str(question["question"]),
                condition=condition,
                stage=stage,
                retriever=retriever,
                generator=None,
                method=method,
                generation_mode="extractive",
                k=k,
            )
            cited = list(dict.fromkeys(
                chunk_id
                for claim in response.claims
                for chunk_id in claim.source_chunk_ids
            ))
            cited_set = set(cited)
            hits = cited_set & relevant
            context_concordant_rate = None
            collection = getattr(retriever, "collection", None)
            if cited and collection is not None:
                indexed = collection.get(ids=cited, include=["metadatas"])
                metadata_by_id = {
                    chunk_id: (meta or {})
                    for chunk_id, meta in zip(indexed.get("ids", []), indexed.get("metadatas", []))
                }
                concordant = 0
                for chunk_id in cited:
                    meta = metadata_by_id.get(chunk_id, {})
                    raw_scope = meta.get("stage_scope_csv", "")
                    scope = raw_scope.split("|") if raw_scope else meta.get("stage_scope", [])
                    if meta.get("condition") == condition and stage in scope:
                        concordant += 1
                context_concordant_rate = concordant / len(cited)
            pair[method] = {
                "status": response.status,
                "reason_code": response.reason_code,
                "claim_count": len(response.claims),
                "cited_chunk_ids": cited,
                "relevant_chunk_ids": sorted(relevant),
                "labelled_relevant_citation_count": len(hits),
                "citation_relevance_precision": len(hits) / len(cited) if cited else 0.0,
                "citation_relevance_recall": len(hits) / len(relevant),
                "has_labelled_relevant_citation": bool(hits),
                "source_link_count": len(response.sources),
                "source_links_validated": all(
                    source.source_url.startswith("https://")
                    for source in response.sources
                ),
                "context_concordant_citation_rate": context_concordant_rate,
            }
        pair["paired"] = {
            "same_status": pair["baseline"]["status"] == pair["contextual"]["status"],
            "baseline_status": pair["baseline"]["status"],
            "contextual_status": pair["contextual"]["status"],
            "citation_relevance_precision_delta": (
                pair["contextual"]["citation_relevance_precision"]
                - pair["baseline"]["citation_relevance_precision"]
            ),
            "citation_relevance_recall_delta": (
                pair["contextual"]["citation_relevance_recall"]
                - pair["baseline"]["citation_relevance_recall"]
            ),
        }
        outcomes.append(pair)

    summaries: dict[str, Any] = {}
    for method in ("baseline", "contextual"):
        rows = [outcome[method] for outcome in outcomes]
        summaries[method] = {
            "question_count": len(rows),
            "answered_count": sum(row["status"] == "answered" for row in rows),
            "abstained_count": sum(row["status"] == "abstained" for row in rows),
            "answer_rate": mean(float(row["status"] == "answered") for row in rows),
            "citation_relevance_precision_macro": mean(
                row["citation_relevance_precision"] for row in rows
            ),
            "citation_relevance_recall_macro": mean(
                row["citation_relevance_recall"] for row in rows
            ),
            "labelled_relevant_citation_coverage": mean(
                float(row["has_labelled_relevant_citation"]) for row in rows
            ),
            "context_concordant_citation_rate": (
                mean(
                    row["context_concordant_citation_rate"]
                    for row in rows
                    if row["context_concordant_citation_rate"] is not None
                )
                if any(row["context_concordant_citation_rate"] is not None for row in rows)
                else None
            ),
            "validated_source_link_count": sum(row["source_link_count"] for row in rows),
            "invalid_https_source_rows": sum(not row["source_links_validated"] for row in rows),
        }

    delta_keys = (
        "citation_relevance_precision_macro",
        "citation_relevance_recall_macro",
        "labelled_relevant_citation_coverage",
        "answer_rate",
    )
    differences = {
        f"{key}_contextual_minus_baseline": round(
            summaries["contextual"][key] - summaries["baseline"][key], 6
        )
        for key in delta_keys
    }
    return {
        "question_count": len(outcomes),
        "k": k,
        "generation_mode": "extractive_verbatim_sentences",
        "metric_provenance": "citation overlap with assistant-adjudicated provisional benchmark labels",
        "method_summaries": summaries,
        "paired_differences_contextual_minus_baseline": differences,
        "per_question": outcomes,
        "limitations": [
            "The fallback copies selected sentences verbatim from source passages; it does not generate a novel natural-language synthesis.",
            "Relevance labels are assistant-adjudicated provisional annotations, not independently verified ground truth.",
            "Citation-label overlap is a retrieval/citation relevance diagnostic, not a measure of medical truth or clinical safety.",
            "The source collection has only 39 chunks and five approved pages; results are exploratory.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "heldout", "all"), default="dev")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive.")

    try:
        benchmark = load_benchmark(args.benchmark)
        corpus = load_chunk_records(args.chunks)
        questions = benchmark["questions"]
        if args.split != "all":
            questions = [question for question in questions if question["split"] == args.split]
        validate_relevant_ids(questions, (chunk["chunk_id"] for chunk in corpus))
        for question in questions:
            if question["condition"] not in ALLOWED_CONDITIONS or question["stage"] not in ALLOWED_STAGES:
                raise ValueError(f"Unsupported benchmark context for {question['question_id']}.")
        retriever = create_persistent_retriever(args.index_dir, args.collection)
        if retriever.collection.count() != len(corpus):
            raise ValueError("Local index count differs from the chunk corpus; rebuild it.")
        results = evaluate_extractive_fallback(questions, retriever, k=args.k)
        output = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"extractive_answers_{args.split}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "benchmark_id": benchmark.get("benchmark_id"),
            "benchmark_annotation_status": benchmark.get("annotation_status"),
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "split": args.split,
            "corpus_chunks": len(corpus),
            "indexed_chunks": retriever.collection.count(),
            "embedding_model": (retriever.collection.metadata or {}).get("embedding_model"),
            "distance_metric": (retriever.collection.metadata or {}).get("hnsw:space", "unknown"),
            "results": results,
        }
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Extractive evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({
        "output": str(output),
        "split": args.split,
        "question_count": results["question_count"],
        "method_summaries": results["method_summaries"],
        "paired_differences": results["paired_differences_contextual_minus_baseline"],
        "metric_provenance": results["metric_provenance"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
