"""Run the answerable-question retrieval benchmark without an LLM."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .manifest import ALLOWED_CONDITIONS, ALLOWED_STAGES
from .metrics import recall_at_k, reciprocal_rank_at_k, retrieval_coverage
from .retrieval import RetrievedChunk, SemanticRetriever, create_persistent_retriever
from .vector_index import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_INDEX_DIR,
    load_chunk_records,
)

DEFAULT_BENCHMARK_PATH = Path("data/benchmark/answerable_questions.json")
DEFAULT_OUTPUT_DIR = Path("data/local/evaluation")
REQUIRED_QUESTION_FIELDS = {
    "question_id",
    "split",
    "condition",
    "stage",
    "question",
    "relevant_chunk_ids",
}


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK_PATH) -> dict[str, Any]:
    """Load and validate the draft benchmark's structure and six-cell split."""
    benchmark_path = Path(path)
    try:
        data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in benchmark file {benchmark_path}: {exc}") from exc

    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Benchmark must be a JSON object with schema_version=1.")
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Benchmark must contain a non-empty 'questions' list.")

    seen_ids: set[str] = set()
    counts: dict[tuple[str, str, str], int] = {}
    split_counts = {"dev": 0, "heldout": 0}
    for position, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Benchmark question {position} must be an object.")
        missing = sorted(REQUIRED_QUESTION_FIELDS - set(question))
        if missing:
            raise ValueError(
                f"Benchmark question {position} is missing required fields: "
                + ", ".join(missing)
            )
        question_id = question["question_id"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Benchmark question {position} has an invalid question_id.")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate benchmark question_id {question_id!r}.")
        seen_ids.add(question_id)

        split = question["split"]
        condition = question["condition"]
        stage = question["stage"]
        query = question["question"]
        relevant_ids = question["relevant_chunk_ids"]
        if split not in split_counts:
            raise ValueError(f"Question {question_id!r} has unsupported split {split!r}.")
        if condition not in ALLOWED_CONDITIONS:
            raise ValueError(f"Question {question_id!r} has unsupported condition {condition!r}.")
        if stage not in ALLOWED_STAGES:
            raise ValueError(f"Question {question_id!r} has unsupported stage {stage!r}.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Question {question_id!r} has an empty question.")
        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise ValueError(f"Question {question_id!r} needs at least one relevant chunk ID.")
        if any(not isinstance(item, str) or not item for item in relevant_ids):
            raise ValueError(f"Question {question_id!r} has an invalid relevant chunk ID.")
        if len(set(relevant_ids)) != len(relevant_ids):
            raise ValueError(f"Question {question_id!r} repeats a relevant chunk ID.")

        split_counts[split] += 1
        key = (condition, stage, split)
        counts[key] = counts.get(key, 0) + 1

    # Freeze the declared research design: four development and two held-out
    # questions in each of the six condition-stage cells.
    expected_counts = {
        (condition, stage, split): expected
        for condition in sorted(ALLOWED_CONDITIONS)
        for stage in sorted(ALLOWED_STAGES)
        for split, expected in (("dev", 4), ("heldout", 2))
    }
    if counts != expected_counts:
        raise ValueError(
            "Benchmark is not balanced as specified. Expected four dev and two heldout "
            "questions in each condition-stage cell."
        )
    if split_counts != {"dev": 24, "heldout": 12}:
        raise ValueError("Benchmark split totals must be 24 dev and 12 heldout questions.")

    return data


def validate_relevant_ids(questions: Sequence[Mapping[str, Any]], corpus_ids: Iterable[str]) -> None:
    """Reject stale or misspelled gold labels before running any retrieval."""
    known_ids = set(corpus_ids)
    unknown: list[tuple[str, str]] = []
    for question in questions:
        for chunk_id in question["relevant_chunk_ids"]:
            if chunk_id not in known_ids:
                unknown.append((question["question_id"], chunk_id))
    if unknown:
        details = "; ".join(f"{qid}: {chunk_id}" for qid, chunk_id in unknown)
        raise ValueError(f"Relevant chunk IDs are absent from the current corpus: {details}")


def _metric_record(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> dict[str, float]:
    return {
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
        "mrr_at_k": reciprocal_rank_at_k(retrieved_ids, relevant_ids, k),
        "retrieval_coverage": retrieval_coverage(retrieved_ids, relevant_ids, k),
    }


def _ranked_record(items: Sequence[RetrievedChunk], relevant_ids: set[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rank, item in enumerate(items, start=1):
        metadata = item.metadata or {}
        output.append({
            "rank": rank,
            "chunk_id": item.chunk_id,
            "similarity": round(float(item.similarity), 8),
            "is_relevant": item.chunk_id in relevant_ids,
            "title": metadata.get("title"),
            "source_id": metadata.get("source_id"),
            "source_url": metadata.get("source_url"),
            "condition": metadata.get("condition"),
            "stage_scope": metadata.get("stage_scope", []),
        })
    return output


def evaluate_questions(
    questions: Sequence[Mapping[str, Any]],
    retriever: SemanticRetriever,
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Evaluate both retrieval methods and preserve per-question paired results."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not questions:
        raise ValueError("At least one question is required.")

    per_question: list[dict[str, Any]] = []

    for question in questions:
        qid = question["question_id"]
        relevant = set(question["relevant_chunk_ids"])
        baseline = retriever.retrieve_baseline(question["question"], k=k)
        contextual = retriever.retrieve_contextual(
            question["question"],
            condition=question["condition"],
            stage=question["stage"],
            k=k,
        )
        base_ranked_ids = [item.chunk_id for item in baseline]
        context_ranked_ids = [item.chunk_id for item in contextual]

        base_metrics = _metric_record(base_ranked_ids, relevant, k)
        context_metrics = _metric_record(context_ranked_ids, relevant, k)
        deltas = {
            key: round(context_metrics[key] - base_metrics[key], 8)
            for key in base_metrics
        }
        per_question.append({
            "question_id": qid,
            "condition": question["condition"],
            "stage": question["stage"],
            "question": question["question"],
            "relevant_chunk_ids": sorted(relevant),
            "reference_answer_note": question.get("reference_answer_note"),
            "methods": {
                "baseline": {
                    "retrieved": _ranked_record(baseline, relevant),
                    "metrics": base_metrics,
                },
                "contextual": {
                    "retrieved": _ranked_record(contextual, relevant),
                    "metrics": context_metrics,
                },
            },
            "contextual_minus_baseline": deltas,
        })

    baseline_aggregate = _aggregate_from_per_query(per_question, "baseline")
    contextual_aggregate = _aggregate_from_per_query(per_question, "contextual")
    paired_differences = {
        key: round(
            contextual_aggregate[key] - baseline_aggregate[key],
            8,
        )
        for key in ("recall_at_k_macro", "mrr_at_k_macro", "retrieval_coverage")
    }
    return {
        "question_count": len(questions),
        "k": k,
        "method_summaries": {
            "baseline": baseline_aggregate,
            "contextual": contextual_aggregate,
        },
        "contextual_minus_baseline": paired_differences,
        "per_question": per_question,
    }


def _aggregate_from_per_query(
    per_question: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, Any]:
    output = {
        "question_count": len(per_question),
        "recall_at_k_macro": mean(
            item["methods"][method]["metrics"]["recall_at_k"] for item in per_question
        ),
        "mrr_at_k_macro": mean(
            item["methods"][method]["metrics"]["mrr_at_k"] for item in per_question
        ),
        "retrieval_coverage": mean(
            item["methods"][method]["metrics"]["retrieval_coverage"] for item in per_question
        ),
    }
    return {
        key: int(value) if key == "question_count" else round(float(value), 8)
        for key, value in output.items()
    }


def _default_output_path(split: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"retrieval_{split}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
        help="Evaluate only one split. Development is the default; held-out results are written only when explicitly requested.",
    )
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default=None, help="Output JSON path; defaults under data/local/.")
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1.")

    try:
        benchmark = load_benchmark(args.benchmark)
        corpus = load_chunk_records(args.chunks)
        questions = [
            item for item in benchmark["questions"]
            if item["split"] == args.split
        ]
        validate_relevant_ids(questions, (item["chunk_id"] for item in corpus))
        retriever = create_persistent_retriever(args.index_dir, args.collection)
        indexed_count = retriever.collection.count()
        if indexed_count != len(corpus):
            raise ValueError(
                f"The local index has {indexed_count} chunks but the current corpus has "
                f"{len(corpus)}. Rebuild the index before evaluation."
            )
        evaluated = evaluate_questions(questions, retriever, k=args.k)
        collection_metadata = retriever.collection.metadata or {}
        report = {
            "benchmark_id": benchmark.get("benchmark_id"),
            "benchmark_annotation_status": benchmark.get("annotation_status"),
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "split": args.split,
            "k": args.k,
            "corpus_chunks": len(corpus),
            "indexed_chunks": retriever.collection.count(),
            "embedding_model": collection_metadata.get("embedding_model"),
            "distance_metric": collection_metadata.get("hnsw:space", "unknown"),
            "method_definitions": {
                "baseline": "Semantic ranking across the full indexed corpus, without condition/stage filters.",
                "contextual": "Filter by selected condition and stage, then rank using the same embedding model and cosine distance.",
            },
            "results": evaluated,
            "limitations": [
                "Questions and relevance labels were assistant-adjudicated against the current corpus but are not independently human-validated; treat all metrics as preliminary.",
                "This is a small exploratory benchmark over a curated 39-chunk corpus.",
                "Retrieval metrics do not establish answer correctness, clinical safety, or real-world impact.",
                "The challenge set is not scored by this retrieval-only command.",
            ],
        }
        output_path = Path(args.output) if args.output else _default_output_path(args.split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Benchmark evaluation failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "output": str(output_path),
        "split": args.split,
        "question_count": evaluated["question_count"],
        "k": args.k,
        "corpus_chunks": len(corpus),
        "indexed_chunks": retriever.collection.count(),
        "embedding_model": collection_metadata.get("embedding_model"),
        "baseline": evaluated["method_summaries"]["baseline"],
        "contextual": evaluated["method_summaries"]["contextual"],
        "contextual_minus_baseline": evaluated["contextual_minus_baseline"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
