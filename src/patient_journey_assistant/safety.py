"""Deterministic scope and emergency routing for the educational assistant.

These checks are conservative interaction guardrails, not a clinical triage model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Disposition = Literal["allow", "emergency_redirect", "professional_redirect"]

EMERGENCY_MESSAGE = (
    "This educational assistant cannot assess emergencies. If this may be a "
    "medical emergency or symptoms are severe or rapidly worsening, contact "
    "your local emergency services or seek urgent medical care now. Do not "
    "wait for an online answer."
)
PROFESSIONAL_MESSAGE = (
    "This assistant provides general educational information only. It cannot "
    "diagnose a person, interpret an individual's test results, or recommend "
    "personal treatment or medication changes. Please discuss your situation "
    "with a qualified healthcare professional."
)

_EMERGENCY_PATTERNS = (
    re.compile(r"\b(?:i\s+)?(?:can(?:not|'t)|unable\s+to)\s+breathe\b", re.I),
    re.compile(r"\bsevere\s+(?:trouble\s+breathing|shortness\s+of\s+breath|breathing\s+difficulty|allergic\s+reaction)\b", re.I),
    re.compile(r"\b(?:blue\s+lips|unconscious|unresponsive|seizure|chest\s+pain)\b", re.I),
    re.compile(r"\bsevere\s+asthma\s+attack\b", re.I),
    re.compile(r"\bstruggling\s+to\s+breathe\b", re.I),
    re.compile(r"\b(?:cannot|can't|unable\s+to)\s+speak\s+in\s+full\s+sentences\b", re.I),
    re.compile(r"\bglucose\s+meter\b.{0,80}\b(?:low|below)\b.{0,60}\b(?:confused|confusion|unconscious)\b", re.I),
    re.compile(r"\b(?:call\s+an\s+ambulance|medical\s+emergency)\b", re.I),
    re.compile(r"\b(?:sudden|severe|rapidly\s+worsening)\s+(?:breathing\s+problems|shortness\s+of\s+breath)\b", re.I),
)

_PERSONAL_DIAGNOSIS_PATTERNS = (
    re.compile(r"\b(?:do\s+i\s+have|could\s+i\s+have|diagnose\s+me|what\s+disease\s+do\s+i\s+have)\b", re.I),
    re.compile(r"\b(?:am\s+i\s+(?:diabetic|having\s+asthma)|based\s+on\s+my\s+symptoms)\b", re.I),
    re.compile(r"\bdoes\s+this\s+mean\s+i\s+have\s+(?:diabetes|asthma)\b", re.I),
    re.compile(r"\b(?:confirm|rule\s+out|guarantee)\b.{0,80}\b(?:i\s+have|i\s+do\s+not\s+have|i\s+don't\s+have)\b", re.I),
    re.compile(r"\bcan\s+you\s+guarantee\b.{0,80}\b(?:i\s+do\s+not|i\s+don't)\s+have\b", re.I),
)
_PERSONAL_MEDICATION_PATTERNS = (
    re.compile(
        r"\b(?:should|can|may|must|do)\s+i\b.{0,100}\b"
        r"(?:start|stop|skip|increase|decrease|adjust|change|double|halve|reduce|take|inject)\b"
        r".{0,80}\b(?:insulin|medicine|medication|drug|dose|dosage|pill|tablet|inhaler|metformin)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:i\s+want\s+to|i\s+am\s+going\s+to|i'm\s+going\s+to)\b.{0,60}\b"
        r"(?:change|adjust|stop|start|increase|decrease|skip|double|halve)\b.{0,60}\b"
        r"(?:my\s+)?(?:insulin|medicine|medication|drug|dose|dosage|pill|tablet|inhaler|metformin)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:my\s+(?:(?:blood\s+glucose|blood\s+sugar|hba1c|a1c)\s+)?(?:lab\s+|medical\s+)?test\s+(?:result|reading|level|value)|"
        r"my\s+(?:blood\s+glucose|blood\s+sugar|hba1c|a1c)\s+(?:result|reading|level|value)|"
        r"i\s+(?:got|received)\s+my\s+test\s+result)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:which|what)\b.{0,100}\b(?:medicine|medication|drug|inhaler|insulin|dose|dosage|brand|regimen)\b"
        r".{0,100}\b(?:should|can|may|do)\s+i\b",
        re.I,
    ),
    re.compile(
        r"\b(?:should|can|may|must)\s+i\b.{0,80}\b(?:stop|start|skip|change|adjust|increase|decrease|continue)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:can|should|may|must)\s+i\b.{0,100}\b(?:skip|cancel|avoid|stop\s+attending|miss)\b"
        r".{0,80}\b(?:follow.?ups?|appointments?|screening|tests?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:given|considering|based\s+on)\s+my\b.{0,150}\b(?:make|create|design|build|recommend|give)\b"
        r".{0,100}\b(?:plan|regimen|schedule)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:how\s+much|what\s+(?:dose|dosage|amount))\b.{0,80}\b"
        r"(?:of\s+)?(?:insulin|medicine|medication|drug|dose|dosage|pill|tablet|inhaler|metformin)\b"
        r".{0,60}\b(?:should|can|may|do)\s+i\b",
        re.I,
    ),
    re.compile(
        r"\b(?:should|can|may|must|do)\s+i\b.{0,120}\b"
        r"(?:change|adjust|increase|decrease|start|stop|skip|follow|switch|avoid|take|inject)\b"
        r".{0,100}\b(?:my\s+)?(?:diet|food|exercise|treatment|blood\s+sugar|glucose|symptoms|"
        r"insulin|medicine|medication|drug|dose|dosage|pill|tablet|inhaler|metformin)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:should\s+i|what\s+should\s+i\s+do)\b.{0,100}\b"
        r"(?:my\s+symptoms|my\s+blood\s+sugar|my\s+glucose|my\s+a1c|my\s+hba1c)\b",
        re.I,
    ),
)


@dataclass(frozen=True)
class ScopeDecision:
    """A deterministic scope decision made before retrieval or LLM invocation."""

    disposition: Disposition
    message: str = ""
    reason_code: str = "allowed"

    @property
    def allowed(self) -> bool:
        return self.disposition == "allow"


def assess_query_scope(question: str) -> ScopeDecision:
    """Block emergency and individualized medical-decision requests.

    This intentionally prioritizes simple, conservative routing over attempting
    to infer a person's clinical status. It is not a comprehensive safety classifier.
    """
    if not isinstance(question, str) or not question.strip():
        return ScopeDecision(
            disposition="professional_redirect",
            message="Enter a general educational question to continue.",
            reason_code="empty_question",
        )

    normalized = question.strip()
    if any(pattern.search(normalized) for pattern in _EMERGENCY_PATTERNS):
        return ScopeDecision(
            disposition="emergency_redirect",
            message=EMERGENCY_MESSAGE,
            reason_code="possible_emergency",
        )
    if any(pattern.search(normalized) for pattern in _PERSONAL_DIAGNOSIS_PATTERNS):
        return ScopeDecision(
            disposition="professional_redirect",
            message=PROFESSIONAL_MESSAGE,
            reason_code="personal_diagnosis_request",
        )
    if any(pattern.search(normalized) for pattern in _PERSONAL_MEDICATION_PATTERNS):
        return ScopeDecision(
            disposition="professional_redirect",
            message=PROFESSIONAL_MESSAGE,
            reason_code="personal_medical_decision",
        )
    # Catch pronoun-based medicine changes such as "Should I stop it?" when the
    # name of the medicine appears elsewhere in the same question.
    has_medication_reference = re.search(
        r"\b(?:inhaler|insulin|medicine|medication|drug|dose|dosage|pill|tablet|metformin|controller)\b",
        normalized,
        re.I,
    )
    pronoun_change = re.search(
        r"\b(?:should|can|may|must)\s+i\b.{0,70}\b(?:stop|start|skip|change|adjust|increase|decrease|continue)\s+(?:it|them|this|that)\b",
        normalized,
        re.I,
    )
    if has_medication_reference and pronoun_change:
        return ScopeDecision(
            disposition="professional_redirect",
            message=PROFESSIONAL_MESSAGE,
            reason_code="personal_medical_decision",
        )
    return ScopeDecision(disposition="allow")
