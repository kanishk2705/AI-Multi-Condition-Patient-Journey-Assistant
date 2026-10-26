from pjassist.classifier import classify


def test_conditions():
    assert classify("How do I use my inhaler?")["conditions"] == ("asthma",)
    assert classify("What should my blood sugar be?")["conditions"] == ("diabetes",)
    assert classify("Do steroid tablets raise blood sugar?")["conditions"] == ("asthma", "diabetes")
    assert classify("What is a healthy weight?")["conditions"] == ()


def test_stage():
    assert classify("What causes asthma?")["stage"] == "understand"
    assert classify("How can I manage my diabetes every day?")["stage"] == "manage"
    assert classify("When should I go to the emergency room for asthma?")["stage"] == "warning"
    assert classify("asthma")["stage"] is None
