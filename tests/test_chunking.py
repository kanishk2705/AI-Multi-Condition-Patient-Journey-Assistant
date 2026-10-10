import pytest
from datetime import date

from patient_journey_assistant.chunking import chunk_source_text
from patient_journey_assistant.manifest import SourceRecord


def make_source():
    return SourceRecord(
        source_id="TEST-SOURCE",
        condition="asthma",
        stage_scope=("understanding_condition",),
        title="Test",
        publisher="MedlinePlus",
        url="https://medlineplus.gov/asthma.html",
        source_type="patient_topic_summary",
        checked_date=date(2026, 10, 10),
        status="approved",
        extraction_mode="medlineplus_topic_summary",
        reuse_basis="public domain summary",
        notes="",
    )


def test_chunking_is_repeatable_and_preserves_metadata():
    source = make_source()
    text = " ".join(f"word{i}" for i in range(420))
    first = chunk_source_text(source, text, max_words=180, overlap_words=30, fetched_at="fixed")
    second = chunk_source_text(source, text, max_words=180, overlap_words=30, fetched_at="fixed")
    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]
    assert len(first) == 3
    assert first[0]["stage_scope"] == ["understanding_condition"]
    assert first[0]["source_url"] == source.url
    assert len(first[0]["text"].split()) == 180
    assert len(first[1]["text"].split()) == 180


def test_invalid_chunk_parameters_raise():
    source = make_source()
    with pytest.raises(ValueError):
        chunk_source_text(source, "some text", max_words=30, overlap_words=30)
