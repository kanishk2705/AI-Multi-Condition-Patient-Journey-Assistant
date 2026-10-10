"""Guardrails for the frozen Phase 1 scope."""

CONDITIONS = {"asthma", "type_2_diabetes"}
STAGES = {
    "understanding_condition",
    "appointment_preparation",
    "ongoing_care_information",
}


def test_frozen_scope_has_two_conditions_and_three_stages():
    assert len(CONDITIONS) == 2
    assert len(STAGES) == 3


def test_condition_and_stage_names_are_stable():
    assert CONDITIONS == {"asthma", "type_2_diabetes"}
    assert "appointment_preparation" in STAGES
