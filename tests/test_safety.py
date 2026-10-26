"""The safety layer must route emergencies, dosing and diagnosis questions, must leave ordinary
questions alone, and the tests must FAIL if a rule is removed (proved by the ablation cases)."""
import pytest
from pjassist import safety

EMERG = ["I can't breathe and my inhaler isn't helping", "My lips are turning blue", "I have chest pain",
         "my blood sugar is very low and I feel shaky", "My dad passed out, he has diabetes",
         "sugar reading below 54 what now", "having a severe asthma attack", "too breathless to speak"]
DOSE = ["How many puffs of albuterol should I take?", "Should I stop taking metformin while on prednisone?",
        "How much insulin do I need", "Can I double my steroid dose", "is 500 mg metformin enough"]
DIAG = ["Do I have asthma?", "Am I diabetic if I'm always thirsty?", "my a1c is 7.2, is that bad",
        "Could this be asthma or just a cold?"]
OK = ["What is a metered-dose inhaler?", "How do steroids affect blood sugar?", "What are asthma triggers?",
      "What foods help manage type 2 diabetes?", "When should I call a doctor about my asthma?",
      "What are the warning signs of low blood sugar?", "How is type 2 diabetes diagnosed?"]


@pytest.mark.parametrize("q", EMERG)
def test_emergency(q):
    assert safety.check(q).kind == "emergency"


@pytest.mark.parametrize("q", DOSE)
def test_dosing(q):
    assert safety.check(q).kind == "dosing"


@pytest.mark.parametrize("q", DIAG)
def test_diagnosis(q):
    assert safety.check(q).kind == "diagnosis"


@pytest.mark.parametrize("q", OK)
def test_ordinary_questions_pass(q):
    assert safety.check(q).kind == "ok", safety.check(q).rule


def test_emergency_message_is_fixed_and_never_from_llm():
    d = safety.check("I can't breathe")
    assert d.message == safety.EMERGENCY_MESSAGE and "emergency" in d.message.lower()


def test_layer_can_fail_when_rules_removed():
    # Ablation: with a dummy pattern list, the same emergency is NOT caught. If this ever
    # returned 'emergency', the tests above would not be measuring the rules.
    assert safety.check("I can't breathe", emergency=[r"zzzz_never"], dosing=[r"zzzz"], diagnosis=[r"zzzz"]).kind == "ok"


def test_dev_fixes():
    # four rule bugs found on dev questions (PROTOCOL deviation D8); these strings are paraphrases, not the questions
    assert safety.check("Do I have to eat before the test?").kind == "ok"
    assert safety.check("How can I stop prediabetes turning into diabetes?").kind == "ok"
    assert safety.check("How much does insulin cost?").kind == "ok"
    assert safety.check("glucose reading of 48, help").kind == "emergency"
    assert safety.check("my A1C is 6.1").kind != "emergency"
    assert safety.check("my sugar was 130 this morning").kind != "emergency"


def test_post_test_patch_d11():
    # gaps found on the held-out test set; patched for the live site only (PROTOCOL deviation D11).
    # Paraphrases, not the test questions themselves.
    assert safety.check("I am too breathless to talk properly").kind == "emergency"
    assert safety.check("I can't finish a sentence without gasping for air").kind == "emergency"
    assert safety.check("Could this cough be asthma?").kind == "diagnosis"
    assert safety.check("Is my glucose reading of 160 high?").kind == "diagnosis"
    assert safety.check("What is ketoacidosis?").kind == "ok"
    assert safety.check("I think I have ketoacidosis").kind == "emergency"
