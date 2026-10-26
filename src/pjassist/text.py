"""Tokeniser shared by BM25 (Python) and the browser fallback (docs/app/lexical.js).

Kept deliberately simple so the JavaScript port is line-for-line identical:
lower-case, split on anything that is not a-z/0-9, drop stop words and tokens of
length 1, then a tiny suffix stripper. Any change here must be mirrored in JS;
tests/test_parity.py runs both and compares rankings.
"""
from __future__ import annotations
import re

STOP = frozenset("""a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have having he
her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now
of off on once only or other our ours ourselves out over own same she should so some such than that the their
theirs them themselves then there these they this those through to too under until up very was we were what when
where which while who whom why will with would you your yours yourself yourselves also may might must shall get
gets got many much one two""".split())

_SPLIT = re.compile(r"[^a-z0-9]+")


def stem(t: str) -> str:
    if len(t) > 5 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 5 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 4 and t.endswith("ed"):
        return t[:-2]
    if len(t) > 4 and t.endswith("es") and not t.endswith("ses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokens(text: str) -> list[str]:
    out = []
    for t in _SPLIT.split(text.lower()):
        if len(t) < 2 or t in STOP:
            continue
        out.append(stem(t))
    return out
