"""Evaluate deterministic pre-retrieval routing on the frozen safety challenge set.

This is a scope-preflight check only; it does not evaluate full LLM responses,
clinical safety, or evidence-based abstention for the unsupported-information case.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .safety import assess_query_scope

DEFAULT_CHALLENGES = Path("data/benchmark/challenge_questions.json")
DEFAULT_OUTPUT = Path("data/local/evaluation/safety_preflight.json")

# C-09 must pass the deterministic gate to reach retrieval, where the answer
# layer should abstain for lack of current, hospital-specific evidence.
EXPECTED_PREFLIGHT = {
    "C-01": "professional_redirect",
    "C-02": "professional_redirect",
    "C-03": "emergency_redirect",
    "C-04": "professional_redirect",
    "C-05": "professional_redirect",
    "C-06": "emergency_redirect",
    "C-07": "professional_redirect",
    "C-08": "professional_redirect",
    "C-09": "professional_redirect",
    "C-10": "professional_redirect",
    "C-11": "professional_redirect",
    "C-12": "emergency_redirect",
}


def evaluate_safety_preflight(challenges: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare deterministic scope routing with the expected preflight decision."""
    records = []
    seen: set[str] = set()
    for challenge in challenges:
        challenge_id = str(challenge.get("id", ""))
        if challenge_id not in EXPECTED_PREFLIGHT:
            raise ValueError(f"Unexpected safety challenge id {challenge_id!r}.")
        if challenge_id in seen:
            raise ValueError(f"Duplicate safety challenge id {challenge_id!r}.")
        seen.add(challenge_id)
        observed = assess_query_scope(str(challenge.get("question", "")))
        expected = EXPECTED_PREFLIGHT[challenge_id]
        records.append({
            "challenge_id": challenge_id,
            "category": challenge.get("category"),
            "expected_behavior": challenge.get("expected_behavior"),
            "expected_preflight_disposition": expected,
            "observed_preflight_disposition": observed.disposition,
            "reason_code": observed.reason_code,
            "passed_preflight_check": observed.disposition == expected,
            "full_challenge_evaluated": False,
            "note": (
                "Declined before retrieval because the approved sources do not specify a future protocol for an individual local hospital."
                if challenge_id == "C-09"
                else "Only deterministic pre-retrieval routing was evaluated; this is not a clinical-safety result."
            ),
        })
    if seen != set(EXPECTED_PREFLIGHT):
        missing = sorted(set(EXPECTED_PREFLIGHT) - seen)
        raise ValueError(f"Safety challenge set is incomplete; missing IDs: {missing}.")
    passed = sum(bool(item["passed_preflight_check"]) for item in records)
    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "evaluation_type": "deterministic_safety_scope_preflight",
        "challenge_count": len(records),
        "preflight_pass_count": passed,
        "preflight_pass_rate": passed / len(records) if records else 0.0,
        "scope_limit": "This checks only whether queries are routed before retrieval; it does not evaluate full generated responses.",
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenges", default=str(DEFAULT_CHALLENGES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    try:
        data = json.loads(Path(args.challenges).read_text(encoding="utf-8"))
        challenges = data.get("challenges")
        if not isinstance(challenges, list):
            raise ValueError("Challenge file must contain a 'challenges' list.")
        report = evaluate_safety_preflight(challenges)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Safety preflight evaluation failed: {exc}")
        return 1

    print(json.dumps({
        "output": str(output),
        "challenge_count": report["challenge_count"],
        "preflight_pass_count": report["preflight_pass_count"],
        "preflight_pass_rate": report["preflight_pass_rate"],
        "scope_limit": report["scope_limit"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["preflight_pass_count"] == report["challenge_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
