import json

from patient_journey_assistant.answer_generation import AnswerResponse
from patient_journey_assistant.evaluate_answers import (
    AnswerQualityJudgment,
    evaluate_answer_quality,
    select_balanced_questions,
)
from patient_journey_assistant.retrieval import RetrievedChunk


CHUNK_ID = "chunk-lung-test"


def make_chunk():
    return RetrievedChunk(
        chunk_id=CHUNK_ID,
        similarity=0.82,
        text="Lung function tests measure the volume and flow of air breathed in and out.",
        metadata={
            "condition": "asthma",
            "stage_scope": ["appointment_preparation"],
            "source_url": "https://medlineplus.gov/lab-tests/lung-function-tests/",
            "title": "Lung Function Tests",
            "publisher": "MedlinePlus",
            "source_id": "MEDLINEPLUS-LUNG-FUNCTION-TESTS",
        },
    )


class FakeRetriever:
    def __init__(self):
        self.chunk = make_chunk()
        self.calls = []

    def retrieve_baseline(self, query, k=5):
        self.calls.append(("baseline", query, k))
        return [self.chunk]

    def retrieve_contextual(self, query, *, condition, stage, k=5):
        self.calls.append(("contextual", query, condition, stage, k))
        return [self.chunk]


class FakeGenerator:
    model_name = "fake-generation-model"

    def __init__(self):
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return json.dumps({
            "status": "answered",
            "claims": [{
                "text": "Lung function tests measure the volume and flow of air breathed in and out.",
                "source_chunk_ids": [CHUNK_ID],
            }],
            "reason": "",
        })


class FakeJudge:
    model_name = "fake-quality-judge"

    def __init__(self, score=2):
        self.calls = []
        self.score_value = score

    def score(self, *, question, condition, stage, reference_answer_note, response, retrieved):
        self.calls.append({
            "question": question,
            "condition": condition,
            "stage": stage,
            "response_status": response.status,
            "retrieved_ids": [item.chunk_id for item in retrieved],
        })
        return AnswerQualityJudgment(
            evidence_support=self.score_value,
            evidence_support_rationale="Claim matches the supplied evidence.",
            stage_relevance=self.score_value,
            stage_relevance_rationale="It answers the selected stage.",
            coverage=self.score_value,
            coverage_rationale="It covers the reference note.",
            citation_quality=self.score_value,
            citation_quality_rationale="The citation is a retrieved approved source.",
            major_concerns=[],
        )


def question():
    return {
        "question_id": "A-P-01",
        "split": "dev",
        "condition": "asthma",
        "stage": "appointment_preparation",
        "question": "What do lung function tests measure?",
        "reference_answer_note": "They assess volume and flow of breathed air.",
        "relevant_chunk_ids": [CHUNK_ID],
    }


def test_balanced_question_selection_returns_one_per_condition_stage_cell():
    data = json.loads(
        open("data/benchmark/answerable_questions.json", encoding="utf-8").read()
    )
    dev = [q for q in data["questions"] if q["split"] == "dev"]
    selected = select_balanced_questions(dev, per_cell=1)
    assert len(selected) == 6
    assert len({(q["condition"], q["stage"]) for q in selected}) == 6


def test_zero_per_cell_returns_the_entire_split():
    data = json.loads(
        open("data/benchmark/answerable_questions.json", encoding="utf-8").read()
    )
    dev = [q for q in data["questions"] if q["split"] == "dev"]
    assert len(select_balanced_questions(dev, per_cell=0)) == 24


def test_paired_answer_quality_evaluation_scores_both_methods_with_one_setting():
    retriever = FakeRetriever()
    generator = FakeGenerator()
    judge = FakeJudge()
    result = evaluate_answer_quality(
        [question()], retriever, generator=generator, judge=judge, k=5
    )

    assert result["question_count"] == 1
    assert result["score_provenance"] == "llm_as_judge_provisional"
    assert len(generator.calls) == 2
    assert len(judge.calls) == 2
    assert [call[0] for call in retriever.calls] == ["baseline", "contextual"]

    summaries = result["method_summaries"]
    assert summaries["baseline"]["answered_count"] == 1
    assert summaries["contextual"]["answered_count"] == 1
    assert summaries["baseline"]["evidence_support_mean"] == 2
    assert summaries["contextual"]["citation_quality_mean"] == 2
    assert result["paired_score_differences_contextual_minus_baseline"]["evidence_support_contextual_minus_baseline_mean"] == 0


def test_non_answered_responses_are_not_sent_to_quality_judge():
    # A generator that asks the answer layer to abstain exercises evaluator handling
    # without requiring a real external API provider.
    class AbstainingGenerator:
        model_name = "fake-abstain-model"

        def generate(self, prompt):
            return json.dumps({
                "status": "insufficient_evidence",
                "claims": [],
                "reason": "No supporting evidence.",
            })

    retriever = FakeRetriever()
    judge = FakeJudge()
    result = evaluate_answer_quality(
        [question()], retriever, generator=AbstainingGenerator(), judge=judge, k=5
    )
    assert result["method_summaries"]["baseline"]["answered_count"] == 0
    assert result["method_summaries"]["baseline"]["judge_scored_answers"] == 0
    assert result["method_summaries"]["contextual"]["abstained_count"] == 1
    assert judge.calls == []
