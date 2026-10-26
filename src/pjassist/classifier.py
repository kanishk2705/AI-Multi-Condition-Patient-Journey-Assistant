"""Infer (conditions, stage) from a question with a hand-written keyword lexicon.

Written before the question set and without looking at gold passages (PROTOCOL.md 4.3).
It reads nothing from data/ at runtime: the lexicon is the constant below, so the
leakage tests can inspect it. Matching is on the shared tokeniser's stems, so it behaves
the same in the browser fallback.

Conditions: every condition with at least one cue; steroid/prednisone cues add both
(oral steroids for asthma raise blood glucose). No cue -> () = unknown, no filter.
Stage: the stage with the most cue hits; ties broken warning > manage > understand
(safer to over-route to warning signs). No cue -> None.
"""
from __future__ import annotations
import re
from .text import tokens

CONDITION_CUES = {
    "asthma": ["asthma", "asthmatic", "inhaler", "puffer", "wheeze", "wheezing", "wheezy", "breath", "breathe",
               "breathing", "breathless", "shortness of breath", "airway", "airways", "lung", "lungs",
               "spirometry", "peak flow", "nebulizer", "spacer", "allergen", "allergens", "pollen", "dust mites",
               "mold", "cough", "coughing", "bronchodilator", "albuterol", "salbutamol", "reliever", "rescue",
               "controller", "inhaled", "attack", "flare", "flare up"],
    "diabetes": ["diabetes", "diabetic", "sugar", "glucose", "insulin", "metformin", "a1c", "hba1c",
                 "hypoglycemia", "hyperglycemia", "carbohydrate", "carbohydrates", "carbs", "carb", "prediabetes",
                 "glucometer", "ketones", "foot", "feet", "neuropathy", "kidney", "kidneys", "retinopathy",
                 "meal plan", "sweets", "thirsty", "urinate"],
    "both": ["steroid", "steroids", "prednisone", "prednisolone", "corticosteroid", "corticosteroids",
             "both conditions", "asthma and diabetes", "diabetes and asthma"],
}
STAGE_CUES = {
    "warning": ["emergency", "urgent", "hospital", "911", "ambulance", "warning sign", "warning signs", "severe",
                "worse", "worsening", "danger", "dangerous", "when to call", "when should i call", "when to see",
                "when should i see", "get help", "seek care", "call my doctor", "call a doctor", "symptoms of low",
                "signs of low", "signs of high", "symptoms of high", "too low", "too high", "attack", "red flag",
                "go to the er", "emergency room"],
    "manage": ["manage", "managing", "control", "controlling", "daily", "every day", "routine", "diet", "eat",
               "eating", "food", "foods", "meal", "exercise", "physical activity", "active", "walk", "check",
               "monitor", "monitoring", "track", "avoid", "prevent", "preventing", "plan", "action plan",
               "vaccine", "vaccines", "flu shot", "sick day", "sick days", "travel", "technique", "use my inhaler",
               "use an inhaler", "take care", "lifestyle", "weight", "sleep", "stress", "smoking", "quit"],
    "understand": ["what is", "what are", "what does", "what causes", "cause", "causes", "why", "mean",
                   "means", "diagnosed", "diagnosis", "test", "tests", "risk factor", "risk factors", "type 2",
                   "type 1", "difference", "how common", "who gets", "explain", "definition", "happens in"],
}
STAGE_ORDER = ("warning", "manage", "understand")


def _phrases(cues):
    return [tuple(tokens(c)) for c in cues if tokens(c)]


_COND = {k: _phrases(v) for k, v in CONDITION_CUES.items()}
_STAGE = {k: _phrases(v) for k, v in STAGE_CUES.items()}
# "what is" etc. are all stop words; keep them by matching on raw lower-case text too
_RAW_STAGE = {k: [c for c in v if not tokens(c)] for k, v in STAGE_CUES.items()}


def _count(tok: list[str], phrases) -> int:
    n = 0
    for p in phrases:
        L = len(p)
        n += sum(1 for i in range(len(tok) - L + 1) if tuple(tok[i:i + L]) == p)
    return n


def classify(question: str) -> dict:
    tok = tokens(question)
    low = " " + " ".join(re.sub(r"[^a-z0-9' ]+", " ", question.lower().replace("’", "'")).split()) + " "
    hits = {k: _count(tok, v) for k, v in _COND.items()}
    conds = set()
    if hits["both"]:
        conds |= {"asthma", "diabetes"}
    for c in ("asthma", "diabetes"):
        if hits[c]:
            conds.add(c)
    s_hits = {k: _count(tok, _STAGE[k]) + sum(low.count(" " + c + " ") for c in _RAW_STAGE[k]) for k in STAGE_ORDER}
    best = max(s_hits.values())
    stage = None if best == 0 else next(s for s in STAGE_ORDER if s_hits[s] == best)
    return {"conditions": tuple(sorted(conds)), "stage": stage, "condition_hits": hits, "stage_hits": s_hits}
