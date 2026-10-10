"""Offline checks for the Phase 4 benchmark and evaluator."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from patient_journey_assistant.evaluate_retrieval import (
    evaluate_questions,
    load_benchmark,
    validate_relevant_ids,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "benchmark" / "answerable_questions.json"
CHALLENGES = ROOT / "data" / "benchmark" / "challenge_questions.json"
INVENTORY = ROOT / "data" / "benchmark" / "chunk_inventory.csv"


def test_answerable_benchmark_has_balanced_dev_and_heldout_splits():
    benchmark = load_benchmark(BENCHMARK)
    questions = benchmark["questions"]
    assert len(questions) == 36
    assert sum(item["split"] == "dev" for item in questions) == 24
    assert sum(item["split"] == "heldout" for item in questions) == 12

    for condition in ("asthma", "type_2_diabetes"):
        for stage in (
            "understanding_condition",
            "appointment_preparation",
            "ongoing_care_information",
        ):
            assert sum(
                item["condition"] == condition
                and item["stage"] == stage
                and item["split"] == "dev"
                for item in questions
            ) == 4
            assert sum(
                item["condition"] == condition
                and item["stage"] == stage
                and item["split"] == "heldout"
                for item in questions
            ) == 2


def test_all_relevance_labels_resolve_to_uploaded_corpus_inventory():
    benchmark = load_benchmark(BENCHMARK)
    with INVENTORY.open(encoding="utf-8", newline="") as file:
        chunk_ids = {row["chunk_id"] for row in csv.DictReader(file)}
    assert len(chunk_ids) == 39
    validate_relevant_ids(benchmark["questions"], chunk_ids)


def test_relevance_rationales_match_relevant_chunk_ids():
    benchmark = load_benchmark(BENCHMARK)
    for question in benchmark["questions"]:
        rationale_ids = [item["chunk_id"] for item in question["relevance_rationales"]]
        assert set(rationale_ids) == set(question["relevant_chunk_ids"])
        assert question["annotation_status"] == "draft_requires_human_review"


def test_challenge_set_has_twelve_expected_behavior_rubrics():
    data = json.loads(CHALLENGES.read_text(encoding="utf-8"))
    challenges = data["challenges"]
    assert len(challenges) == 12
    assert len({item["id"] for item in challenges}) == 12
    assert all(item["expected_behavior"] and item["expected_behavior_note"] for item in challenges)


def test_evaluator_reports_paired_retrieval_metrics():
    benchmark = load_benchmark(BENCHMARK)
    question = next(item for item in benchmark["questions"] if item["split"] == "dev")
    gold = question["relevant_chunk_ids"][0]

    class FakeRetriever:
        def retrieve_baseline(self, query, k=5):
            return [
                SimpleNamespace(chunk_id="unrelated", similarity=0.9, metadata={}),
                SimpleNamespace(chunk_id=gold, similarity=0.8, metadata={"title": "Relevant"}),
            ][:k]

        def retrieve_contextual(self, query, *, condition, stage, k=5):
            return [
                SimpleNamespace(chunk_id=gold, similarity=0.8, metadata={"title": "Relevant"}),
                SimpleNamespace(chunk_id="other", similarity=0.7, metadata={}),
            ][:k]

    result = evaluate_questions([question], FakeRetriever(), k=5)
    assert result["question_count"] == 1
    assert result["method_summaries"]["baseline"]["mrr_at_k_macro"] == 0.5
    assert result["method_summaries"]["contextual"]["mrr_at_k_macro"] == 1.0
    assert result["contextual_minus_baseline"]["mrr_at_k_macro"] == 0.5
    assert result["per_question"][0]["methods"]["contextual"]["retrieved"][0]["is_relevant"] is True


def test_evaluator_rejects_unknown_gold_chunk_ids():
    benchmark = load_benchmark(BENCHMARK)
    with pytest.raises(ValueError, match="absent from the current corpus"):
        validate_relevant_ids(benchmark["questions"], {"known-chunk-only"})
