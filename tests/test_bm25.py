from pjassist.bm25 import BM25
from pjassist.text import tokens


def test_tokeniser():
    assert tokens("Inhalers, wheezing & the A1C!") == ["inhaler", "wheez", "a1c"]


def test_bm25_ranks_relevant_first():
    docs = ["Asthma inhalers open the airways.", "Insulin lowers blood glucose.", "Eat vegetables daily."]
    s = BM25(docs).scores("how do inhalers help asthma")
    assert max(range(3), key=lambda i: s[i]) == 0 and s[2] == 0
