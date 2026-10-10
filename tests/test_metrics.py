import pytest

from patient_journey_assistant.metrics import (
    evaluate_rankings,
    recall_at_k,
    reciprocal_rank_at_k,
    retrieval_coverage,
)


def test_recall_at_k_uses_all_relevant_chunks_as_denominator():
    assert recall_at_k(
        ["a", "x", "b", "c"],
        {"a", "b", "missing"},
        k=3,
    ) == pytest.approx(2 / 3)


def test_mrr_uses_first_relevant_rank_and_zero_when_absent():
    assert reciprocal_rank_at_k(["x", "b", "a"], {"a", "b"}, k=3) == 0.5
    assert reciprocal_rank_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_retrieval_coverage_is_binary_per_query():
    assert retrieval_coverage(["x", "a"], {"a"}, k=2) == 1.0
    assert retrieval_coverage(["x", "y"], {"a"}, k=2) == 0.0


def test_macro_metrics_are_averaged_per_query():
    result = evaluate_rankings(
        retrieved_by_query={
            "q1": ["a", "x"],
            "q2": ["x", "y"],
        },
        relevant_by_query={
            "q1": {"a"},
            "q2": {"b"},
        },
        k=2,
    )

    assert result["query_count"] == 2
    assert result["recall_at_k_macro"] == pytest.approx(0.5)
    assert result["mrr_at_k_macro"] == pytest.approx(0.5)
    assert result["retrieval_coverage"] == pytest.approx(0.5)


def test_metrics_reject_missing_annotations_and_mismatched_query_ids():
    with pytest.raises(ValueError, match="relevant chunk ID"):
        recall_at_k(["a"], set())
    with pytest.raises(ValueError, match="same query IDs"):
        evaluate_rankings({"q1": ["a"]}, {"q2": {"a"}})
