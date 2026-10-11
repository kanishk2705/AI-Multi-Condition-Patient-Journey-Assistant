from patient_journey_assistant.evaluate_extractive import evaluate_extractive_fallback
from patient_journey_assistant.retrieval import RetrievedChunk


CHUNK_ID = "chunk-extractive-1"


def make_chunk(chunk_id=CHUNK_ID, *, condition="asthma", stage_scope=None, similarity=0.9):
    return RetrievedChunk(
        chunk_id=chunk_id,
        similarity=similarity,
        text=(
            "Lung function tests measure how much air a person can breathe in and out. "
            "They also measure how quickly air moves in and out of the lungs."
        ),
        metadata={
            "condition": condition,
            "stage_scope": stage_scope or ["appointment_preparation"],
            "source_url": "https://medlineplus.gov/lab-tests/lung-function-tests/",
            "title": "Lung Function Tests",
            "publisher": "MedlinePlus",
            "source_id": "MEDLINEPLUS-LUNG-FUNCTION-TESTS",
        },
    )


class FakeRetriever:
    def __init__(self, baseline=None, contextual=None):
        self.baseline = list(baseline if baseline is not None else [make_chunk()])
        self.contextual = list(contextual if contextual is not None else [make_chunk()])
        self.calls = []

    def retrieve_baseline(self, query, k=5):
        self.calls.append(("baseline", query, k))
        return self.baseline[:k]

    def retrieve_contextual(self, query, *, condition, stage, k=5):
        self.calls.append(("contextual", query, condition, stage, k))
        return self.contextual[:k]


def question(relevant_ids=None):
    return {
        "question_id": "A-P-01",
        "condition": "asthma",
        "stage": "appointment_preparation",
        "question": "What do lung function tests measure?",
        "relevant_chunk_ids": relevant_ids or [CHUNK_ID],
    }


def test_extractive_fallback_evaluation_is_paired_and_uses_label_overlap_only():
    retriever = FakeRetriever()
    result = evaluate_extractive_fallback([question()], retriever, k=5)

    assert result["question_count"] == 1
    assert result["generation_mode"] == "extractive_verbatim_sentences"
    assert result["metric_provenance"].startswith("citation overlap")
    assert retriever.calls[0][0] == "baseline"
    assert retriever.calls[1][0] == "contextual"
    assert result["method_summaries"]["baseline"]["answered_count"] == 1
    assert result["method_summaries"]["contextual"]["answered_count"] == 1
    assert result["method_summaries"]["contextual"]["citation_relevance_precision_macro"] == 1.0
    assert result["method_summaries"]["contextual"]["citation_relevance_recall_macro"] == 1.0


def test_extractive_evaluation_records_zero_gold_overlap_without_relabelling():
    retriever = FakeRetriever()
    result = evaluate_extractive_fallback(
        [question(relevant_ids=["another-labelled-chunk"])],
        retriever,
        k=5,
    )
    row = result["per_question"][0]["contextual"]
    assert row["status"] == "answered"
    assert row["labelled_relevant_citation_count"] == 0
    assert row["citation_relevance_precision"] == 0.0
    assert row["citation_relevance_recall"] == 0.0
    assert result["method_summaries"]["contextual"]["labelled_relevant_citation_coverage"] == 0.0


def test_extractive_evaluation_output_omits_question_and_passage_text():
    retriever = FakeRetriever()
    result = evaluate_extractive_fallback([question()], retriever, k=5)
    serialized = str(result)
    assert "What do lung function tests measure?" not in serialized
    assert "Lung function tests measure how much air" not in serialized
