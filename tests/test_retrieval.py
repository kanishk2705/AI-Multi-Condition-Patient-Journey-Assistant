import pytest

from patient_journey_assistant.retrieval import SemanticRetriever, build_context_filter


class FakeEncoder:
    def __init__(self):
        self.queries = []

    def encode(self, texts):
        self.queries.extend(texts)
        return [[1.0, 0.0] for _ in texts]


class FakeCollection:
    def __init__(self, records):
        self.records = records
        self.query_calls = []

    @staticmethod
    def matches(metadata, where):
        if where is None:
            return True
        for condition in where["$and"]:
            key, rule = next(iter(condition.items()))
            expected = rule["$eq"]
            if metadata.get(key) != expected:
                return False
        return True

    def count(self, where=None):
        return sum(self.matches(record["metadata"], where) for record in self.records)

    def query(self, *, query_embeddings, n_results, include, where=None):
        self.query_calls.append({
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include,
            "where": where,
        })
        selected = [
            record for record in self.records
            if self.matches(record["metadata"], where)
        ]
        selected.sort(key=lambda item: (item["distance"], item["id"]))
        selected = selected[:n_results]
        return {
            "ids": [[record["id"] for record in selected]],
            "documents": [[record["text"] for record in selected]],
            "metadatas": [[record["metadata"] for record in selected]],
            "distances": [[record["distance"] for record in selected]],
        }


def sample_records():
    return [
        {
            "id": "chunk-z",
            "distance": 0.1,
            "text": "Asthma appointment preparation evidence.",
            "metadata": {
                "condition": "asthma",
                "stage_appointment_preparation": True,
                "stage_scope_csv": "appointment_preparation",
                "title": "Lung Function Tests",
            },
        },
        {
            "id": "chunk-b",
            "distance": 0.1,
            "text": "Diabetes ongoing care evidence.",
            "metadata": {
                "condition": "type_2_diabetes",
                "stage_appointment_preparation": False,
                "stage_scope_csv": "ongoing_care_information",
                "title": "Blood Glucose Test",
            },
        },
        {
            "id": "chunk-a",
            "distance": 0.1,
            "text": "Asthma appointment evidence, second passage.",
            "metadata": {
                "condition": "asthma",
                "stage_appointment_preparation": True,
                "stage_scope_csv": "appointment_preparation",
                "title": "Lung Function Tests",
            },
        },
        {
            "id": "chunk-c",
            "distance": 0.3,
            "text": "Asthma ongoing care evidence.",
            "metadata": {
                "condition": "asthma",
                "stage_appointment_preparation": False,
                "stage_scope_csv": "ongoing_care_information",
                "title": "Asthma",
            },
        },
    ]


def test_baseline_retrieval_is_unfiltered_and_breaks_ties_by_chunk_id():
    collection = FakeCollection(sample_records())
    retriever = SemanticRetriever(collection, FakeEncoder())

    results = retriever.retrieve_baseline("What happens at the appointment?", k=3)

    assert [item.chunk_id for item in results] == ["chunk-a", "chunk-b", "chunk-z"]
    assert collection.query_calls[0]["where"] is None
    assert collection.query_calls[0]["n_results"] == 4
    assert results[0].similarity == pytest.approx(0.9)


def test_contextual_retrieval_filters_condition_and_stage_before_ranking():
    collection = FakeCollection(sample_records())
    retriever = SemanticRetriever(collection, FakeEncoder())

    results = retriever.retrieve_contextual(
        "What happens at the appointment?",
        condition="asthma",
        stage="appointment_preparation",
        k=5,
    )

    assert [item.chunk_id for item in results] == ["chunk-a", "chunk-z"]
    where = collection.query_calls[0]["where"]
    assert where == build_context_filter("asthma", "appointment_preparation")
    assert all(item.metadata["condition"] == "asthma" for item in results)
    assert all("appointment_preparation" in item.metadata["stage_scope"] for item in results)


def test_contextual_retrieval_returns_empty_when_no_chunks_are_eligible():
    collection = FakeCollection([])
    retriever = SemanticRetriever(collection, FakeEncoder())

    results = retriever.retrieve_contextual(
        "Explain the question.",
        condition="asthma",
        stage="appointment_preparation",
    )

    assert results == []
    assert collection.query_calls == []


def test_filter_rejects_unknown_condition_or_stage():
    with pytest.raises(ValueError, match="Unsupported condition"):
        build_context_filter("copd", "appointment_preparation")
    with pytest.raises(ValueError, match="Unsupported journey stage"):
        build_context_filter("asthma", "emergency_triage")


def test_retriever_rejects_blank_query_and_invalid_k():
    retriever = SemanticRetriever(FakeCollection(sample_records()), FakeEncoder())
    with pytest.raises(ValueError, match="non-empty"):
        retriever.retrieve_baseline("   ")
    with pytest.raises(ValueError, match="at least 1"):
        retriever.retrieve_baseline("asthma", k=0)
