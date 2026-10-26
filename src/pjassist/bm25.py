"""Okapi BM25 (k1=1.2, b=0.75, Lucene-style idf). Own implementation so the browser
fallback can reproduce it exactly (docs/app/lexical.js)."""
from __future__ import annotations
import math
from collections import Counter
from .text import tokens

K1, B = 1.2, 0.75


class BM25:
    def __init__(self, texts: list[str]):
        self.docs = [Counter(tokens(t)) for t in texts]
        self.len = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.len) / max(1, len(self.len))
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for d, dl in zip(self.docs, self.len):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / self.avgdl))
            out.append(s)
        return out
