from pathlib import Path

import pytest

from patient_journey_assistant.manifest import (
    ManifestValidationError,
    load_source_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_project_manifest_loads_and_only_explicitly_approved_sources_are_fetchable():
    records = load_source_manifest(ROOT / "data" / "source_manifest.csv")
    assert len(records) == 9
    approved = [record for record in records if record.approved_for_ingestion]
    assert len(approved) == 5
    assert all(record.extraction_mode != "none" for record in approved)
    assert any(record.status == "deferred_permission" for record in records)


def test_invalid_domain_is_rejected(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "source_id,condition,stage_scope,title,publisher,url,source_type,checked_date,status,extraction_mode,reuse_basis,notes\n"
        "TEST-SOURCE,asthma,understanding_condition,Test,MedlinePlus,"
        "https://example.com/page,patient_topic_summary,2026-10-10,approved,"
        "medlineplus_topic_summary,Public domain,Test row\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestValidationError, match="approved source domain"):
        load_source_manifest(manifest)


def test_duplicate_ids_are_rejected(tmp_path):
    manifest = tmp_path / "manifest.csv"
    row = "TEST-SOURCE,asthma,understanding_condition,Test,MedlinePlus,https://medlineplus.gov/asthma.html,patient_topic_summary,2026-10-10,approved,medlineplus_topic_summary,Public domain,Test row"
    manifest.write_text(
        "source_id,condition,stage_scope,title,publisher,url,source_type,checked_date,status,extraction_mode,reuse_basis,notes\n"
        + row + "\n" + row + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestValidationError, match="duplicate source_id"):
        load_source_manifest(manifest)
