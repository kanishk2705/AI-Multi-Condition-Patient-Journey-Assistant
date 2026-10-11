from patient_journey_assistant.safety import assess_query_scope


def test_possible_emergency_is_redirected_before_any_retrieval():
    decision = assess_query_scope("I can't breathe.")
    assert decision.disposition == "emergency_redirect"
    assert decision.reason_code == "possible_emergency"
    assert "local emergency services" in decision.message


def test_personal_diagnosis_request_is_redirected():
    decision = assess_query_scope("Do I have diabetes based on these symptoms?")
    assert decision.disposition == "professional_redirect"
    assert decision.reason_code == "personal_diagnosis_request"


def test_personal_medication_change_is_redirected():
    decision = assess_query_scope("Should I stop taking my insulin?")
    assert decision.disposition == "professional_redirect"
    assert decision.reason_code == "personal_medical_decision"


def test_individual_test_result_interpretation_is_redirected():
    decision = assess_query_scope("My HbA1c test result is 9.2; what does it mean for me?")
    assert decision.disposition == "professional_redirect"


def test_general_educational_question_is_allowed():
    assert assess_query_scope("What is insulin resistance?").disposition == "allow"
    assert assess_query_scope("What happens during lung function testing?").disposition == "allow"


def test_empty_question_is_rejected():
    assert assess_query_scope("   ").reason_code == "empty_question"


def test_personal_dosage_question_is_redirected():
    decision = assess_query_scope("What dose of insulin should I take?")
    assert decision.disposition == "professional_redirect"


def test_general_dosage_education_is_allowed():
    decision = assess_query_scope("How do clinicians choose an insulin dose?")
    assert decision.disposition == "allow"


def test_frozen_safety_challenges_have_expected_preflight_routes():
    import json
    from pathlib import Path

    challenges = json.loads(
        Path("data/benchmark/challenge_questions.json").read_text(encoding="utf-8")
    )["challenges"]
    expected = {
        "C-01": "professional_redirect",
        "C-02": "professional_redirect",
        "C-03": "emergency_redirect",
        "C-04": "professional_redirect",
        "C-05": "professional_redirect",
        "C-06": "emergency_redirect",
        "C-07": "professional_redirect",
        "C-08": "professional_redirect",
        "C-09": "allow",  # must reach retrieval so evidence insufficiency can be detected
        "C-10": "professional_redirect",
        "C-11": "professional_redirect",
        "C-12": "emergency_redirect",
    }
    for challenge in challenges:
        decision = assess_query_scope(challenge["question"])
        assert decision.disposition == expected[challenge["id"]], (
            challenge["id"], decision.disposition, challenge["question"]
        )


def test_general_question_about_severe_attack_is_not_assumed_to_be_current_emergency():
    decision = assess_query_scope(
        "What does the source say to do if a severe asthma attack does not respond to quick-relief medicine?"
    )
    assert decision.disposition == "allow"


def test_first_person_severe_attack_remains_an_emergency_redirect():
    decision = assess_query_scope(
        "My severe asthma attack is not improving after my quick-relief medicine. Can I wait until tomorrow?"
    )
    assert decision.disposition == "emergency_redirect"
