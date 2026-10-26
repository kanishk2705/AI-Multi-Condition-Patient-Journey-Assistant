"""Rule-based safety layer (live-site version, deviation D11; the evaluated version is
results/final/safety_rules_evaluated.py), applied BEFORE retrieval and generation.

1. Emergency phrases -> a fixed "seek urgent care now" message; no retrieval, no LLM.
2. Personal dosing / medicine-change questions -> declined with a fixed message.
3. Personal diagnosis questions -> declined with a fixed message.

Rules are written by hand (PROTOCOL.md section 6), not learned, so their behaviour is
inspectable. They are deliberately broad: a false alarm costs a reader one extra
sentence, a miss could cost much more. Precision/recall on the authored question set are
reported, and tests/test_safety.py shows the layer fails when a rule is removed.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

EMERGENCY_MESSAGE = (
    "This sounds like it could be an emergency. Seek urgent care now: call your local emergency "
    "number (112 in India, 911 in the US) or go to the nearest emergency department. "
    "If you have a rescue inhaler or fast-acting sugar and have been told how to use it, use it while you get help. "
    "This tool cannot help in an emergency and does not give medical advice."
)
DOSING_MESSAGE = (
    "I can't answer questions about how much medicine to take, or about starting, stopping or changing a medicine. "
    "Those decisions depend on you personally; please ask your doctor, nurse or pharmacist. "
    "I can share general information from public health sources about how these medicines work."
)
DIAGNOSIS_MESSAGE = (
    "I can't tell whether you have a condition or what is causing your symptoms. Only a health professional "
    "who can examine you can do that. I can share general information about how asthma and type 2 diabetes "
    "are diagnosed."
)

_EMERGENCY = [
    r"\b(can'?t|cannot|can not|unable to|struggling to|hard to) (breathe|breath|catch (my|his|her|their) breath)\b",
    r"\bnot breathing\b", r"\bgasping\b", r"\b(lips|fingernails|face) (\w+ ){0,2}(blue|grey|gray)\b",
    r"\bchest pain\b", r"\bpain in (my|the) chest\b", r"\bchest (is )?(tight|tightness) and\b.*\b(can'?t|cannot)\b",
    r"\b(passed out|fainted|unconscious|unresponsive|not responding|won'?t wake|can'?t wake)\b",
    r"\b(seizure|fitting|convulsion)s?\b",
    r"\bvery low (blood )?(sugar|glucose)\b", r"\b(sugar|glucose) (is |level is )?(very|really|dangerously) low\b",
    r"\b(sugar|glucose)\b.*\b(below|under) (40|45|50|54)\b",
    r"\b(sugar|glucose|meter|reading)\b[^.?!]*?\b([2-4]\d|5[0-4])\b(?!\s*(%|percent|years?|minutes?|hours?|grams?|g\b|kg|lbs?))",
    r"\b(inhaler|reliever|puffer)\b.*\b(not|isn'?t|doesn'?t|didn'?t) (help|work)",
    r"\bsevere (asthma )?attack\b", r"\b(too|so) (breathless|short of breath|out of breath) to\b",
    r"\b(can'?t|cannot|unable to) (finish|complete|get out) (a |my )?(sentence|words?)\b",
    r"\b(confused|slurred speech|can'?t speak)\b.*\b(sugar|glucose|diabet)", r"\b(sugar|glucose|diabet)\w*\b.*\b(confused|slurred speech)\b",
    r"\b(i|i'm|i am|he|she|they|my \w+) (think|might|may|have|has|got|is|are)\b.*\bketoacidosis\b", r"\b(i|my)\b.*\bhigh ketones\b", r"\bkeep vomiting\b", r"\bcan'?t stop vomiting\b",
    r"\b(emergency|ambulance|911|112)\b.*\b(now|right now|need)\b", r"\bdying\b",
]
_DOSING = [
    r"\bhow (many|much)\b(?!.*\b(cost|costs|price|pay|money)\b).*\b(puffs?|units?|mg|milligrams?|tablets?|pills?|doses?|insulin|metformin|prednisone|steroids?)\b",
    r"\b(what|which) (dose|dosage)\b", r"\b(my|the right|correct|maximum|max) (dose|dosage)\b",
    r"\b(should|can|could|may) i (take|skip|double|increase|decrease|reduce|lower|raise|change|switch|start)\b",
    r"\b(increase|double|halve|reduce|lower|raise|adjust|change|skip|stop) (my|the) (dose|insulin|metformin|inhaler|steroids?|prednisone|medicines?|medications?|tablets?|pills?)\b",
    r"\b(stop|quit|come off) (taking )?(my )?(insulin|metformin|inhaler|steroids?|prednisone|medicines?|medications?)\b",
    r"\b\d+\s?(mg|units?|puffs?)\b",
]
_DIAGNOSIS = [
    r"\bdo i have (?!to\b)", r"\bam i (diabetic|asthmatic|prediabetic|pre-diabetic)\b", r"\bdoes (he|she|my \w+) have\b",
    r"\b(is|could) (this|it|that) (be )?(asthma|diabetes|copd|a heart attack|an? infection)\b",
    r"\bwhat'?s wrong with me\b", r"\bdiagnose (me|my)\b", r"\bmy (test|a1c|hba1c|glucose|sugar|spirometry|peak flow) (result|reading|number|level)s? (is|was|are|were)\b",
    r"\bmy (a1c|hba1c) is \d",
    r"\bcould (my|this|it|these) [\w\s]{0,30}\bbe (asthma|diabetes|prediabetes|copd|an? \w+)\b",
    r"\bis my [\w\s]{0,30}\b(reading|level|number|result|peak flow|sugar|glucose)\b[\w\s]{0,20}\b(normal|ok|okay|good|bad|high|low)\b", r"\bis my (a1c|hba1c|sugar|glucose|reading|peak flow) (normal|bad|ok|okay|too high|too low)\b",
]
_NEUTRALISE = [r"metered[- ]dose", r"dry powder inhaler"]  # device names that contain trigger words


@dataclass(frozen=True)
class SafetyDecision:
    kind: str      # "emergency" | "dosing" | "diagnosis" | "ok"
    message: str
    rule: str = ""


def _norm(q: str) -> str:
    q = q.lower().replace("’", "'")
    for p in _NEUTRALISE:
        q = re.sub(p, " device ", q)
    return re.sub(r"\s+", " ", q)


def check(question: str, emergency=None, dosing=None, diagnosis=None) -> SafetyDecision:
    """Pattern lists can be overridden (tests use this to prove the layer can fail)."""
    q = _norm(question)
    for kind, pats, msg in (("emergency", emergency or _EMERGENCY, EMERGENCY_MESSAGE),
                            ("dosing", dosing or _DOSING, DOSING_MESSAGE),
                            ("diagnosis", diagnosis or _DIAGNOSIS, DIAGNOSIS_MESSAGE)):
        for p in pats:
            if re.search(p, q):
                return SafetyDecision(kind, msg, p)
    return SafetyDecision("ok", "")


EMERGENCY_PATTERNS, DOSING_PATTERNS, DIAGNOSIS_PATTERNS = _EMERGENCY, _DOSING, _DIAGNOSIS
