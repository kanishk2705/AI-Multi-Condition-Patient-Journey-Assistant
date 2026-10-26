"""Independent grounding checker (PROTOCOL.md section 7).

An answer sentence is SUPPORTED only if
  (1) it cites at least one passage that was actually retrieved for this question, and
  (2) NLI: some window of 1-3 consecutive sentences from a cited passage entails it with
      P(entailment) >= NLI_T, using an NLI cross-encoder (not the LLM that wrote the answer), and
  (3) lexical: at least LEX_T of the sentence's content tokens occur in the cited passages.
Both thresholds were fixed before any answer was generated. The NLI model is passed in as
a function so tests can substitute a stub and show the checker can fail.
"""
from __future__ import annotations
import re
from typing import Callable, Iterable
from .text import tokens

NLI_T = 0.5
LEX_T = 0.6
CITE = re.compile(r"\[(P\d+)\]")
SCOPE_PHRASES = ("sources shown do not cover", "sources do not cover", "passages do not cover")
_SENT = re.compile(r"([.!?](?:\s*\[P\d+\])*)\s+(?=[A-Z0-9\"'(])")

NliFn = Callable[[list[tuple[str, str]]], list[float]]  # (premise, hypothesis) -> P(entailment)


def normalise_citations(text: str) -> str:
    """Models write citation labels in several styles; map them all to [P1] before checking.
    Handles full-width brackets (【P1】, 【P1†L3】), parentheses (P1) and comma lists [P1, P2]."""
    text = re.sub(r"【\s*(P\d+)[^】]*】", r"[\1]", text)
    text = re.sub(r"\(\s*(P\d+)\s*\)", r"[\1]", text)
    text = re.sub(r"\[\s*(P\d+(?:\s*[,;]\s*P\d+)+)\s*\]",
                  lambda m: "".join(f"[{x.strip()}]" for x in re.split(r"[,;]", m.group(1))), text)
    return text


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in _SENT.sub(lambda m: m.group(1) + "\n", text).split("\n") if s.strip()]


def strip_cites(s: str) -> str:
    return re.sub(r"\s*\[P\d+\]", "", s).strip()


def windows(passage: str, max_len: int = 3) -> list[str]:
    ss = split_sentences(passage)
    out = []
    for i in range(len(ss)):
        for L in range(1, max_len + 1):
            if i + L <= len(ss):
                out.append(" ".join(ss[i:i + L]))
    return out or [passage]


def lexical_support(sentence: str, passages: Iterable[str]) -> float:
    st = set(tokens(strip_cites(sentence)))
    if not st:
        return 1.0
    pt = set()
    for p in passages:
        pt |= set(tokens(p))
    return len(st & pt) / len(st)


def check_answer(answer: str, labels: dict[str, str], nli: NliFn, max_windows: int | None = None) -> dict:
    """max_windows=None (the evaluated setting) tests every window; the live site passes a small
    number to fit a slow free server (PROTOCOL deviation D14).
    labels maps a citation label like 'P1' to the retrieved passage text shown to the generator."""
    # A sentence that only states what the sources do NOT cover (generator rule 5) makes no
    # health claim; it is counted separately and excluded from the grounding denominator.
    all_sents = split_sentences(normalise_citations(answer))
    scope = [s for s in all_sents if not CITE.search(s) and any(p in s.lower() for p in SCOPE_PHRASES)]
    sents = [s for s in all_sents if s not in scope]
    rows, pairs, owners = [], [], []
    for i, s in enumerate(sents):
        cited = [c for c in CITE.findall(s) if c in labels]
        bad = [c for c in CITE.findall(s) if c not in labels]
        hyp = strip_cites(s)
        lex = lexical_support(s, [labels[c] for c in cited]) if cited else 0.0
        rows.append({"sentence": s, "cited": cited, "invalid_cites": bad, "lexical": lex, "nli": 0.0})
        cand = [w for c in cited for w in windows(labels[c])]
        if max_windows:  # live-site speed option: keep the windows sharing most words with the sentence
            ht = set(tokens(hyp))
            cand = sorted(cand, key=lambda w: -len(ht & set(tokens(w))))[:max_windows]
        for w in cand:
            pairs.append((w, hyp)); owners.append(i)
    if pairs:
        probs = nli(pairs)
        for i, p in zip(owners, probs):
            rows[i]["nli"] = max(rows[i]["nli"], float(p))
    for r in rows:
        r["supported"] = bool(r["cited"]) and r["nli"] >= NLI_T and r["lexical"] >= LEX_T
        r["reason"] = ("supported" if r["supported"] else "uncited" if not r["cited"]
                       else "nli" if r["nli"] < NLI_T and r["lexical"] >= LEX_T
                       else "lexical" if r["nli"] >= NLI_T else "nli+lexical")
    n = len(rows)
    k = sum(r["supported"] for r in rows)
    return {"n_sentences": n, "n_supported": k, "fully_grounded": n > 0 and k == n, "sentences": rows,
            "scope_notes": scope}
