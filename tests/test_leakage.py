"""Leakage guard tests (PROTOCOL.md 5.2). Each guard is tested on the real data AND shown to fail
on a deliberately leaked input, so a passing test means something."""
import json
from pathlib import Path
import pytest
from pjassist import classifier, leakage

ROOT = Path(__file__).resolve().parents[1]
P = [json.loads(l) for l in open(ROOT / "data/corpus/passages.jsonl")]
Q = json.loads((ROOT / "data/questions/dev.json").read_text()) + json.loads((ROOT / "data/questions/test.json").read_text())


def test_lexicon_clean():
    assert leakage.lexicon_violations(classifier.CONDITION_CUES, classifier.STAGE_CUES, P, Q) == []


def test_lexicon_guard_fails_on_leak():
    leaked = dict(classifier.STAGE_CUES, manage=classifier.STAGE_CUES["manage"] + [Q[0]["question"]])
    assert leakage.lexicon_violations(classifier.CONDITION_CUES, leaked, P, Q)
    six = " ".join(P[10]["text"].split()[:12])
    leaked2 = dict(classifier.CONDITION_CUES, diabetes=classifier.CONDITION_CUES["diabetes"] + [six])
    assert leakage.lexicon_violations(leaked2, classifier.STAGE_CUES, P, Q, max_words=99)


def test_metadata_clean():
    assert leakage.metadata_violations(P, Q) == []


def test_metadata_guard_fails_on_leak():
    bad = dict(P[0], section=Q[3]["question"])
    assert leakage.metadata_violations([bad], Q)
    bad2 = dict(P[0], gold_for=["a01"])
    assert leakage.metadata_violations([bad2], Q)
    bad3 = dict(P[0], title="answers a01")
    assert leakage.metadata_violations([bad3], Q)


def test_classifier_reads_no_data():
    assert leakage.source_violations(leakage.classifier_source()) == []


def test_source_guard_fails_on_leak():
    assert leakage.source_violations("x = json.load(open('data/questions/test.json'))")
