"""Parity: the JavaScript fallback (docs/app/lexical.js) must reproduce the Python lexical
pipeline exactly: tokens, BM25 top-10 ranking and scores, classifier output and the extractive
answer, on fixed queries. Skipped if Node is not installed."""
import json, shutil, subprocess
from pathlib import Path
import numpy as np
import pytest
from pjassist.bm25 import BM25
from pjassist.classifier import classify
from pjassist.generate import extractive_answer
from pjassist.text import tokens
from pjassist import safety

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ["How do I use my inhaler before exercise?", "What are the symptoms of low blood sugar?",
           "Can steroid tablets raise blood glucose?", "When should I go to the emergency room for asthma?",
           "What is the A1C test?", "flu shot for people with diabetes and asthma", "dust mites mold",
           "What does insulin resistance mean?", "Why is my reliever making me shaky", "What is the 15-15 rule?",
           "How much physical activity per week", "quitting smoking benefits", "What causes asthma?",
           "nighttime low blood sugar signs", "Is the nasal spray vaccine ok?",
           "I can't breathe and my inhaler isn't helping", "How many puffs should I take?", "Do I have asthma?",
           "my glucose reading is 48", "What is a metered-dose inhaler?", "Do I have to fast before the test?"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_matches_python():
    data = json.loads((ROOT / "docs/app/data/index.json").read_text())
    P = data["passages"]
    bm = BM25([p["title"] + " " + (p["section"] or "") + " " + p["text"] for p in P])
    idf = BM25([p["text"] for p in P]).idf
    js = json.loads(subprocess.run(["node", str(ROOT / "tests/parity_runner.js"), json.dumps(QUERIES)],
                                   capture_output=True, text=True, check=True).stdout)
    for q, j in zip(QUERIES, js):
        s = np.asarray(bm.scores(q))
        order = list(np.argsort(-s, kind="stable")[:10])
        assert j["tokens"] == tokens(q), q
        assert j["rank"] == [int(i) for i in order], q
        assert np.allclose(j["scores"], s[order], rtol=1e-9, atol=1e-12), q
        m = classify(q)
        assert j["meta"]["conditions"] == list(m["conditions"]) and j["meta"]["stage"] == m["stage"], q
        assert j["safety"] == safety.check(q).kind, q
        assert j["extractive"] == extractive_answer(q, [P[i] for i in order[:4]], idf)["text"], q


def test_parity_can_fail():
    # sanity: a different tokeniser output would not compare equal
    assert tokens("inhalers") != tokens("inhaler wheezing")
