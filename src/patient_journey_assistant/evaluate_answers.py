"""Paired answer-quality evaluation for baseline and contextual retrieval.

By default, evaluate one development question from each condition-stage cell to
limit provider calls. The judge is an LLM-as-judge diagnostic, not human ground truth.
All answer artifacts are written locally under data/local/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .answer_generation import (
    AnswerResponse,
    GenerationConfigurationError,
    GoogleGenAITextGenerator,
    answer_question,
)
from .evaluate_retrieval import load_benchmark, validate_relevant_ids
from .manifest import ALLOWED_CONDITIONS, ALLOWED_STAGES
from .retrieval import RetrievedChunk, SemanticRetriever, create_persistent_retriever
from .vector_index import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_INDEX_DIR,
    load_chunk_records,
)

DEFAULT_BENCHMARK_PATH = Path("data/benchmark/answerable_questions.json")
DEFAULT_OUTPUT_DIR = Path("data/local/evaluation")
DIMENSIONS = (
    "evidence_support",
    "stage_relevance",
    "coverage",
    "citation_quality",
)

JUDGE_SYSTEM_INSTRUCTION = """You are an evaluator for a narrowly scoped educational retrieval-augmented generation prototype.
Use only the supplied question, reference-answer note, candidate answer, and retrieved passage text.
Do not use outside medical knowledge to add facts. Judge whether the claims and citations are supported
by the supplied passages and whether the answer fits the selected information stage. Do not reward
verbosity or confident tone. An abstention is handled outside this scoring rubric and should not be scored.
Return the required structured 0–2 scores, each with a concise evidence-based rationale.
This LLM-as-judge output is only a provisional diagnostic and must not be represented as human ground truth.
"""


class AnswerQualityJudgment(BaseModel):
    """LLM-as-judge scores based on the project's frozen 0–2 rubric."""

    model_config = ConfigDict(extra="forbid")

    evidence_support: int = Field(ge=0, le=2)
    evidence_support_rationale: str = Field(min_length=1, max_length=500)
    stage_relevance: int = Field(ge=0, le=2)
    stage_relevance_rationale: str = Field(min_length=1, max_length=500)
    coverage: int = Field(ge=0, le=2)
    coverage_rationale: str = Field(min_length=1, max_length=500)
    citation_quality: int = Field(ge=0, le=2)
    citation_quality_rationale: str = Field(min_length=1, max_length=500)
    major_concerns: list[str] = Field(default_factory=list, max_length=8)

    @property
    def total_score(self) -> int:
        return sum(int(getattr(self, dimension)) for dimension in DIMENSIONS)

    def to_record(self) -> dict[str, Any]:
        record = self.model_dump()
        record["total_score"] = self.total_score
        return record


class QualityJudge(Protocol):
    model_name: str

    def score(
        self,
        *,
        question: str,
        condition: str,
        stage: str,
        reference_answer_note: str,
        response: AnswerResponse,
        retrieved: Sequence[RetrievedChunk],
    ) -> AnswerQualityJudgment: ...


class GoogleGenAIQualityJudge:
    """Structured Gemini judge sharing the answer model alias and environment key."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if client is None and not resolved_key:
            raise GenerationConfigurationError(
                "No Gemini API key configured for answer-quality judging."
            )
        self.model_name = model_name or os.getenv(
            "PJA_GENERATION_MODEL", "gemini-flash-latest"
        )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GenerationConfigurationError(
                "Google Gen AI SDK is missing; install the application requirements."
            ) from exc
        self._types = types
        self._client = client or genai.Client(api_key=resolved_key)

    def score(
        self,
        *,
        question: str,
        condition: str,
        stage: str,
        reference_answer_note: str,
        response: AnswerResponse,
        retrieved: Sequence[RetrievedChunk],
    ) -> AnswerQualityJudgment:
        evidence = []
        for item in retrieved:
            metadata = item.metadata or {}
            evidence.append({
                "chunk_id": item.chunk_id,
                "title": metadata.get("title", ""),
                "condition": metadata.get("condition", ""),
                "stage_scope": metadata.get("stage_scope", []),
                "source_url": metadata.get("source_url", ""),
                "text": item.text,
            })
        candidate = {
            "status": response.status,
            "message": response.message,
            "claims": [
                {"text": claim.text, "source_chunk_ids": list(claim.source_chunk_ids)}
                for claim in response.claims
            ],
            "cited_sources": [
                {
                    "chunk_id": source.chunk_id,
                    "title": source.title,
                    "url": source.source_url,
                }
                for source in response.sources
            ],
        }
        prompt = (
            "Score this answer using the fixed 0–2 rubric below.\n"
            "Evidence support: 0 = material claims unsupported/contradictory; "
            "1 = mixed support or an important gap; 2 = material claims supported by evidence.\n"
            "Stage relevance: 0 = off-stage/no answer; 1 = partly relevant; "
            "2 = directly answers for the selected stage.\n"
            "Coverage: 0 = misses key supported information; 1 = partial; 2 = covers the key "
            "points supported by the collection.\n"
            "Citation quality: 0 = invalid/missing/non-supporting citations; 1 = some support; "
            "2 = cited sources are real retrieved sources and support the associated claims.\n\n"
            + json.dumps({
                "condition": condition,
                "stage": stage,
                "question": question,
                "reference_answer_note": reference_answer_note,
                "candidate_answer": candidate,
                "retrieved_evidence": evidence,
            }, ensure_ascii=False)
        )
        config = self._types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=AnswerQualityJudgment,
            temperature=0,
            max_output_tokens=1200,
        )
        output = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        response_text = getattr(output, "text", None)
        if not isinstance(response_text, str) or not response_text.strip():
            raise RuntimeError("The answer-quality judge returned an empty response.")
        try:
            return AnswerQualityJudgment.model_validate_json(response_text)
        except (ValidationError, ValueError, TypeError) as exc:
            raise RuntimeError("The answer-quality judge returned invalid structured output.") from exc


class _CapturedRetriever:
    """Return exactly the previously recorded top-k lists for a paired experiment."""

    def __init__(
        self,
        baseline: Sequence[RetrievedChunk],
        contextual: Sequence[RetrievedChunk],
    ) -> None:
        self._baseline = list(baseline)
        self._contextual = list(contextual)

    def retrieve_baseline(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        return self._baseline[:k]

    def retrieve_contextual(
        self, query: str, *, condition: str, stage: str, k: int = 5
    ) -> list[RetrievedChunk]:
        return self._contextual[:k]


def select_balanced_questions(
    questions: Sequence[Mapping[str, Any]],
    *,
    per_cell: int = 1,
) -> list[Mapping[str, Any]]:
    """Select the first N questions per condition-stage cell; zero means all."""
    if per_cell < 0:
        raise ValueError("per_cell cannot be negative.")
    if per_cell == 0:
        return list(questions)
    counts: Counter[tuple[str, str]] = Counter()
    selected: list[Mapping[str, Any]] = []
    for question in questions:
        cell = (str(question["condition"]), str(question["stage"]))
        if cell not in {(c, s) for c in ALLOWED_CONDITIONS for s in ALLOWED_STAGES}:
            raise ValueError(f"Unexpected condition-stage cell: {cell!r}")
        if counts[cell] < per_cell:
            selected.append(question)
            counts[cell] += 1
    expected = len(ALLOWED_CONDITIONS) * len(ALLOWED_STAGES) * per_cell
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} questions for {per_cell} per cell, selected {len(selected)}."
        )
    return selected


def _response_record(
    response: AnswerResponse,
    retrieved: Sequence[RetrievedChunk],
    *,
    condition: str,
    stage: str,
    judge: QualityJudge,
    question: Mapping[str, Any],
) -> dict[str, Any]:
    retrieved_by_id = {item.chunk_id: item for item in retrieved}
    cited_ids = [
        chunk_id
        for claim in response.claims
        for chunk_id in claim.source_chunk_ids
    ]
    context_cited = 0
    for chunk_id in cited_ids:
        item = retrieved_by_id.get(chunk_id)
        if item and item.metadata.get("condition") == condition and stage in item.metadata.get("stage_scope", []):
            context_cited += 1
    context_match_rate = context_cited / len(cited_ids) if cited_ids else None

    judgment = None
    not_scored_reason = None
    if response.status == "answered":
        judgment = judge.score(
            question=str(question["question"]),
            condition=condition,
            stage=stage,
            reference_answer_note=str(question.get("reference_answer_note", "")),
            response=response,
            retrieved=retrieved,
        ).to_record()
    else:
        not_scored_reason = f"response_status:{response.status}:{response.reason_code}"

    return {
        "status": response.status,
        "reason_code": response.reason_code,
        "model_name": response.model_name,
        "claims": [
            {"text": claim.text, "source_chunk_ids": list(claim.source_chunk_ids)}
            for claim in response.claims
        ],
        "cited_sources": [
            {
                "chunk_id": source.chunk_id,
                "title": source.title,
                "publisher": source.publisher,
                "source_url": source.source_url,
            }
            for source in response.sources
        ],
        "retrieved_top_k": [
            {
                "rank": rank,
                "chunk_id": item.chunk_id,
                "similarity": round(float(item.similarity), 8),
                "title": item.metadata.get("title"),
                "source_url": item.metadata.get("source_url"),
                "condition": item.metadata.get("condition"),
                "stage_scope": item.metadata.get("stage_scope", []),
            }
            for rank, item in enumerate(retrieved, start=1)
        ],
        "context_concordant_citation_rate": context_match_rate,
        "quality_judgment": judgment,
        "not_scored_reason": not_scored_reason,
    }


def _summarize_method(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    answered = [r for r in records if r["status"] == "answered"]
    scored = [r["quality_judgment"] for r in answered if r["quality_judgment"]]
    summary: dict[str, Any] = {
        "question_count": len(records),
        "answered_count": len(answered),
        "answer_rate": (len(answered) / len(records)) if records else 0.0,
        "abstained_count": sum(r["status"] == "abstained" for r in records),
        "redirected_count": sum(r["status"] in {"redirected", "professional_redirect"} for r in records),
        "service_unavailable_count": sum(r["status"] in {"service_unavailable", "configuration_missing"} for r in records),
        "judge_scored_answers": len(scored),
    }
    for dimension in DIMENSIONS:
        summary[f"{dimension}_mean"] = (
            round(mean(item[dimension] for item in scored), 4) if scored else None
        )
    totals = [sum(int(item[d]) for d in DIMENSIONS) for item in scored]
    summary["total_score_mean_out_of_8"] = round(mean(totals), 4) if totals else None
    rates = [r["context_concordant_citation_rate"] for r in records if r["context_concordant_citation_rate"] is not None]
    summary["context_concordant_citation_rate"] = round(mean(rates), 4) if rates else None
    return summary


def evaluate_answer_quality(
    questions: Sequence[Mapping[str, Any]],
    retriever: SemanticRetriever,
    *,
    generator: Any,
    judge: QualityJudge,
    k: int = 5,
) -> dict[str, Any]:
    """Generate paired answers from the exact baseline/contextual retrieval results."""
    if not questions:
        raise ValueError("At least one question is required.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    outcomes: list[dict[str, Any]] = []

    for question in questions:
        condition, stage = str(question["condition"]), str(question["stage"])
        baseline_chunks = retriever.retrieve_baseline(question["question"], k=k)
        contextual_chunks = retriever.retrieve_contextual(
            question["question"], condition=condition, stage=stage, k=k
        )
        captured = _CapturedRetriever(baseline_chunks, contextual_chunks)

        paired: dict[str, Any] = {"question_id": question["question_id"]}
        for method, items in (("baseline", baseline_chunks), ("contextual", contextual_chunks)):
            response = answer_question(
                str(question["question"]),
                condition=condition,
                stage=stage,
                retriever=captured,
                generator=generator,
                k=k,
                method=method,
            )
            paired[method] = _response_record(
                response,
                items,
                condition=condition,
                stage=stage,
                judge=judge,
                question=question,
            )
        paired["paired_status"] = {
            "same_status": paired["baseline"]["status"] == paired["contextual"]["status"],
            "baseline": paired["baseline"]["status"],
            "contextual": paired["contextual"]["status"],
        }
        outcomes.append(paired)

    by_method = {
        method: [outcome[method] for outcome in outcomes]
        for method in ("baseline", "contextual")
    }
    summaries = {method: _summarize_method(records) for method, records in by_method.items()}
    paired_scores: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        deltas = []
        for outcome in outcomes:
            a = outcome["baseline"]["quality_judgment"]
            b = outcome["contextual"]["quality_judgment"]
            if a is not None and b is not None:
                deltas.append(float(b[dimension]) - float(a[dimension]))
        paired_scores[f"{dimension}_contextual_minus_baseline_mean"] = (
            round(mean(deltas), 4) if deltas else None
        )
    paired_scores["paired_judged_answer_count"] = sum(
        outcome["baseline"]["quality_judgment"] is not None
        and outcome["contextual"]["quality_judgment"] is not None
        for outcome in outcomes
    )
    return {
        "question_count": len(outcomes),
        "k": k,
        "score_provenance": "llm_as_judge_provisional",
        "answer_generation_model": getattr(generator, "model_name", None),
        "answer_quality_judge_model": getattr(judge, "model_name", None),
        "judge_rubric_dimensions": list(DIMENSIONS),
        "method_summaries": summaries,
        "paired_score_differences_contextual_minus_baseline": paired_scores,
        "per_question": outcomes,
        "limitations": [
            "LLM-as-judge scores are provisional diagnostics and are not human ground truth.",
            "Only a sampled question set may be run; report the sampling policy and question count.",
            "Citation provenance is checked by code, but semantic evidence entailment is judged automatically.",
            "The answerable benchmark labels remain assistant-adjudicated provisional annotations.",
            "Retrieval and answer-quality results do not establish clinical accuracy or safety.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    parser.add_argument(
        "--per-cell",
        type=int,
        default=1,
        help="Questions to evaluate per condition-stage cell; zero evaluates the full split.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.k < 1 or args.per_cell < 0:
        parser.error("--k must be positive and --per-cell cannot be negative.")

    try:
        benchmark = load_benchmark(args.benchmark)
        corpus = load_chunk_records(args.chunks)
        validate_relevant_ids(
            [q for q in benchmark["questions"] if q["split"] == args.split],
            (item["chunk_id"] for item in corpus),
        )
        questions = select_balanced_questions(
            [q for q in benchmark["questions"] if q["split"] == args.split],
            per_cell=args.per_cell,
        )
        retriever = create_persistent_retriever(args.index_dir, args.collection)
        if retriever.collection.count() != len(corpus):
            raise ValueError("Local index count differs from the current chunk corpus; rebuild it.")
        generator = GoogleGenAITextGenerator()
        judge = GoogleGenAIQualityJudge(model_name=generator.model_name)
        evaluated = evaluate_answer_quality(
            questions, retriever, generator=generator, judge=judge, k=args.k
        )
        result = {
            "benchmark_id": benchmark.get("benchmark_id"),
            "benchmark_annotation_status": benchmark.get("annotation_status"),
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "split": args.split,
            "per_cell_limit": args.per_cell,
            "sampling_policy": (
                "all questions in split" if args.per_cell == 0
                else f"first {args.per_cell} question(s) per condition-stage cell"
            ),
            "corpus_chunks": len(corpus),
            "indexed_chunks": retriever.collection.count(),
            "embedding_model": (retriever.collection.metadata or {}).get("embedding_model"),
            "distance_metric": (retriever.collection.metadata or {}).get("hnsw:space", "unknown"),
            "results": evaluated,
        }
        output = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"answer_quality_{args.split}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except GenerationConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Answer-quality evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({
        "output": str(output),
        "split": args.split,
        "question_count": result["results"]["question_count"],
        "sampling_policy": result["sampling_policy"],
        "score_provenance": result["results"]["score_provenance"],
        "method_summaries": result["results"]["method_summaries"],
        "paired_differences": result["results"]["paired_score_differences_contextual_minus_baseline"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
