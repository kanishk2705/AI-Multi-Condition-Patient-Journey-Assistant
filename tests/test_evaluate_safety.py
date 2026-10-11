import json
from pathlib import Path

import pytest

from patient_journey_assistant.evaluate_safety import evaluate_safety_preflight


def _challenge_set():
    return json.loads(
        Path("data/benchmark/challenge_questions.json").read_text(encoding="utf-8")
    )["challenges"]


def test_all_frozen_challenges_match_expected_deterministic_preflight():
    report = evaluate_safety_preflight(_challenge_set())
    assert report["challenge_count"] == 12
    assert report["preflight_pass_count"] == 12
    assert report["preflight_pass_rate"] == 1.0
    assert all(not item["full_challenge_evaluated"] for item in report["records"])


def test_unsupported_future_local_protocol_is_declined_before_retrieval():
    report = evaluate_safety_preflight(_challenge_set())
    challenge = next(item for item in report["records"] if item["challenge_id"] == "C-09")
    assert challenge["observed_preflight_disposition"] == "professional_redirect"
    assert challenge["reason_code"] == "unsupported_local_future_protocol"
    assert "approved sources do not specify" in challenge["note"]


def test_duplicate_challenge_ids_are_rejected():
    challenges = _challenge_set()
    with pytest.raises(ValueError, match="Duplicate safety challenge"):
        evaluate_safety_preflight(challenges + [challenges[0]])
