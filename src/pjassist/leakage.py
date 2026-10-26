"""Leakage guard: no question text or gold label may reach the index metadata or the query
classifier (PROTOCOL.md 5.2). Functions return a list of violations; tests assert it is empty
on the real data and non-empty on deliberately leaked inputs."""
from __future__ import annotations
import inspect, re
from .text import tokens

META_FIELDS = ("id", "source", "title", "section", "parent_section", "url", "conditions", "stage", "license")
N = 6


def ngrams(text: str, n: int = N) -> set:
    t = tokens(text)
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def lexicon_violations(cond_cues: dict, stage_cues: dict, passages: list, questions: list, max_words: int = 4) -> list:
    out = []
    cues = [c for v in list(cond_cues.values()) + list(stage_cues.values()) for c in v]
    for c in cues:
        if len(c.split()) > max_words:
            out.append(f"cue longer than {max_words} words: {c!r}")
    cue_grams = set()
    for c in cues:
        cue_grams |= ngrams(c)
    for p in passages:
        if ngrams(p["text"]) & cue_grams:
            out.append(f"passage text in lexicon: {p['id']}")
    for q in questions:
        if ngrams(q["question"]) & cue_grams:
            out.append(f"question text in lexicon: {q['qid']}")
    return out


def metadata_violations(passages: list, questions: list) -> list:
    out = []
    qgrams = {}
    for q in questions:
        for g in ngrams(q["question"], 5):
            qgrams[g] = q["qid"]
    qids = {q["qid"] for q in questions}
    for p in passages:
        meta = " ".join(str(p.get(f) or "") for f in META_FIELDS)
        for g in ngrams(meta, 5):
            if g in qgrams:
                out.append(f"question {qgrams[g]} text in metadata of {p['id']}")
        for w in re.findall(r"\b[a-z]\d{2}\b", meta.lower()):
            if w in qids:
                out.append(f"question id {w} in metadata of {p['id']}")
        extra = set(p) - set(META_FIELDS) - {"text", "retrieved_utc", "page_sha256"}
        if extra:
            out.append(f"unexpected metadata fields {sorted(extra)} in {p['id']}")
    return out


def source_violations(src: str) -> list:
    bad = [w for w in ("questions", "gold", "dev.json", "test.json", "open(", "read_text", "json.load")
           if w in src]
    return [f"classifier source mentions {w!r}" for w in bad]


def classifier_source() -> str:
    """Classifier source code with docstrings removed (the docstring explains, in words, that
    no gold data was used; only executable code is checked)."""
    import ast
    from . import classifier
    tree = ast.parse(inspect.getsource(classifier))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and \
                isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            body.pop(0)
    return ast.unparse(tree)
