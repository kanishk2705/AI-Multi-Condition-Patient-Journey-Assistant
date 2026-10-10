from patient_journey_assistant.vector_index import chunk_to_metadata


def test_chroma_metadata_uses_scalars_and_preserves_stage_filters():
    chunk = {
        "source_id": "TEST-SOURCE",
        "condition": "asthma",
        "stage_scope": ["appointment_preparation", "understanding_condition"],
        "title": "Asthma",
        "publisher": "MedlinePlus",
        "source_url": "https://medlineplus.gov/asthma.html",
        "source_content_sha256": "abc123",
        "chunk_index": 2,
        "word_count": 80,
        "text": "A short sample.",
    }

    metadata = chunk_to_metadata(chunk)

    assert metadata["condition"] == "asthma"
    assert metadata["stage_appointment_preparation"] is True
    assert metadata["stage_understanding_condition"] is True
    assert metadata["stage_ongoing_care_information"] is False
    assert metadata["stage_scope_csv"] == (
        "appointment_preparation|understanding_condition"
    )
    assert all(not isinstance(value, (list, dict)) for value in metadata.values())
