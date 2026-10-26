"""The grounding checker must accept supported sentences and reject unsupported, uncited or
wrongly cited ones. A stub NLI makes the tests deterministic; test_checker_can_fail shows that
an always-entail stub is caught by the lexical check, and an always-reject stub fails everything."""
from pjassist.grounding import check_answer, split_sentences

P = {"P1": "Oral steroids can raise blood glucose levels. People with diabetes may need to check their glucose more often.",
     "P2": "Asthma is a long-term lung disease. It causes the airways to narrow."}


def nli_overlap(pairs):
    # crude stand-in for an NLI model: entailment if every hypothesis word appears in the premise
    out = []
    for prem, hyp in pairs:
        h = {w.strip('.,').lower() for w in hyp.split()}
        p = {w.strip('.,').lower() for w in prem.split()}
        out.append(1.0 if h <= p else 0.0)
    return out


def test_supported_sentence():
    r = check_answer("Oral steroids can raise blood glucose levels. [P1]", P, nli_overlap)
    assert r["fully_grounded"] and r["n_supported"] == 1


def test_wrong_citation_rejected():
    r = check_answer("Oral steroids can raise blood glucose levels. [P2]", P, nli_overlap)
    assert not r["fully_grounded"]


def test_uncited_and_invented_rejected():
    r = check_answer("Steroids cure asthma permanently. Asthma is a long-term lung disease. [P2]", P, nli_overlap)
    assert r["n_sentences"] == 2 and r["n_supported"] == 1
    assert r["sentences"][0]["reason"] == "uncited"


def test_nonexistent_label_rejected():
    r = check_answer("Asthma is a long-term lung disease. [P9]", P, nli_overlap)
    assert not r["fully_grounded"] and r["sentences"][0]["invalid_cites"] == ["P9"]


def test_scope_note_excluded():
    r = check_answer("Asthma is a long-term lung disease. [P2] The sources shown do not cover diabetes.", P, nli_overlap)
    assert r["fully_grounded"] and len(r["scope_notes"]) == 1


def test_checker_can_fail():
    always = lambda pairs: [1.0] * len(pairs)
    never = lambda pairs: [0.0] * len(pairs)
    # lexical guard catches an invented sentence even if the NLI model wrongly says "entailed"
    assert not check_answer("Bananas prevent seizures in children. [P1]", P, always)["fully_grounded"]
    # and a broken NLI model that rejects everything rejects even a verbatim sentence
    assert not check_answer("Oral steroids can raise blood glucose levels. [P1]", P, never)["fully_grounded"]


def test_split():
    assert len(split_sentences("One. Two? Three!")) == 3


def test_citation_styles_normalised():
    for a in ("Oral steroids can raise blood glucose levels【P1】.", "Oral steroids can raise blood glucose levels (P1).",
              "Oral steroids can raise blood glucose levels [P1, P2]."):
        r = check_answer(a, P, nli_overlap)
        assert r["sentences"][0]["cited"][0] == "P1", a


def test_max_windows_keeps_best_window():
    r = check_answer("Asthma is a long-term lung disease. [P2]", P, nli_overlap, max_windows=1)
    assert r["fully_grounded"]
