"""Retrieval systems compared in the study (PROTOCOL.md section 5).

  dense   plain semantic retrieval: cosine similarity of BAAI/bge-small-en-v1.5 embeddings
  bm25    lexical Okapi BM25 (own implementation, see bm25.py)
  hybrid  reciprocal-rank fusion of dense and bm25 (k = 60)
  meta    hybrid candidates re-scored with condition/stage agreement between the passage
          metadata and the metadata INFERRED from the question by classifier.py
  meta_ce meta, then the top candidates re-ranked by a cross-encoder
          (cross-encoder/ms-marco-MiniLM-L-6-v2) with the metadata bonus kept as a tie-breaker

Embeddings and cross-encoder scores are computed on CI (scripts/embed.py) and cached in
data/ as arrays, because the build sandbox cannot download models; every system can then
run anywhere from the cached arrays.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .bm25 import BM25
from .classifier import classify

RRF_K = 60


def rrf(rank_lists: list[list[int]], n: int, k: int = RRF_K) -> np.ndarray:
    s = np.zeros(n)
    for ranks in rank_lists:
        for r, i in enumerate(ranks):
            s[i] += 1.0 / (k + r + 1)
    return s


def meta_bonus(p: dict, q_conds: tuple, q_stage: str | None, wc: float, ws: float, wboth: float) -> float:
    b = 0.0
    pc = set(p["conditions"])
    if q_conds:
        qc = set(q_conds)
        if pc & qc:
            b += wc
        else:
            b -= wc  # passage about the other condition only
        if len(qc) == 2 and len(pc) == 2:
            b += wboth
    if q_stage and p["stage"] == q_stage:
        b += ws
    return b


@dataclass
class Params:
    cand: int = 30          # hybrid candidates considered by meta / meta_ce
    wc: float = 0.010       # condition agreement bonus (RRF scale: rank-1 RRF ~ 0.033)
    ws: float = 0.005       # stage agreement bonus
    wboth: float = 0.005    # extra bonus when question and passage both concern the two conditions
    ce_top: int = 20
    ce_meta_scale: float = 1.0  # bonus added to the CE logit is bonus * ce_meta_scale / wc


@dataclass
class Index:
    passages: list[dict]
    emb: np.ndarray                                   # (n, d) L2-normalised passage embeddings
    bm25: BM25 = field(init=False)

    def __post_init__(self):
        self.bm25 = BM25([p["title"] + " " + (p.get("section") or "") + " " + p["text"] for p in self.passages])
        self.n = len(self.passages)

    # --- base scorers -------------------------------------------------------------------
    def dense_scores(self, qvec: np.ndarray) -> np.ndarray:
        return self.emb @ qvec

    def rank_dense(self, qvec) -> list[int]:
        return list(np.argsort(-self.dense_scores(qvec), kind="stable"))

    def rank_bm25(self, q: str) -> list[int]:
        return list(np.argsort(-np.asarray(self.bm25.scores(q)), kind="stable"))

    def hybrid_scores(self, q: str, qvec) -> np.ndarray:
        return rrf([self.rank_dense(qvec), self.rank_bm25(q)], self.n)

    # --- systems ---------------------------------------------------------------------
    def search(self, system: str, q: str, qvec, params: Params = Params(), meta: dict | None = None,
               ce_fn=None) -> dict:
        """Returns {'ranking': [passage idx...], 'score': [...], 'meta': inferred metadata}.
        `meta` lets the caller supply oracle metadata for the ablation; default is inferred."""
        m = meta or classify(q)
        if system == "dense":
            s = self.dense_scores(qvec)
        elif system == "bm25":
            s = np.asarray(self.bm25.scores(q))
        elif system == "hybrid":
            s = self.hybrid_scores(q, qvec)
        elif system in ("meta", "meta_ce"):
            h = self.hybrid_scores(q, qvec)
            cand = np.argsort(-h, kind="stable")[: params.cand]
            s = np.full(self.n, -1e9)
            for i in cand:
                s[i] = h[i] + meta_bonus(self.passages[i], m["conditions"], m["stage"], params.wc, params.ws, params.wboth)
            if system == "meta_ce":
                top = np.argsort(-s, kind="stable")[: params.ce_top]
                ce = ce_fn(q, [int(i) for i in top])
                s2 = np.full(self.n, -1e9)
                for i, c in zip(top, ce):
                    bonus = meta_bonus(self.passages[i], m["conditions"], m["stage"], params.wc, params.ws, params.wboth)
                    s2[i] = c + params.ce_meta_scale * bonus / max(params.wc, 1e-9)
                s = s2
        else:
            raise ValueError(system)
        order = np.argsort(-s, kind="stable")
        return {"ranking": [int(i) for i in order], "score": [float(s[i]) for i in order], "meta": m}
