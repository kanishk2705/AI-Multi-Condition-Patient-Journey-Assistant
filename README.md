# AI Multi-Condition Patient Journey Assistant

An undergraduate AI project investigating whether condition- and journey-stage-aware retrieval improves the relevance and source-grounding of educational health-information answers.

**Status:** Phase 2 — source validation and ingestion implementation. GitHub Actions passed 15 offline tests and verified the manifest dry-run (9 rows: 5 approved, 4 skipped). Actual source fetching and manual corpus inspection remain pending. No model-performance results are claimed.

## Problem and research question

A generic semantic retriever may return text that is topically similar to a health question but not appropriate for the user's selected condition or information-seeking stage.

**Research question:** Does condition- and journey-stage-aware retrieval return more relevant evidence and support more faithfully grounded answers than ordinary semantic retrieval?

## Frozen scope

- **Conditions:** asthma and type 2 diabetes.
- **Stages:** understanding the condition; preparing for an appointment; understanding general ongoing-care information.
- **Approach:** compare semantic retrieval over the full approved collection against semantic retrieval constrained by the selected condition and stage.
- **Evaluation:** 36 answerable questions plus 12 safety/insufficient-evidence challenges, using pre-labelled relevant sources, Recall@5, MRR@5, citation support, answer quality, and abstention behaviour.

See the [project charter](docs/PROJECT_CHARTER.md) and [evaluation plan](docs/EVALUATION_PLAN.md).

## Current Phase 2 work

The source manifest validator checks required fields, domain allowlists, statuses, stage labels, dates, duplicate IDs/URLs, approved extraction modes, and reuse basis. The ingestion CLI is dry-run by default and fetches only approved rows when explicitly requested. It checks robots.txt, HTTPS, redirect domains, response type/size, and uses restrictive page-specific extraction modes.

The initial approved collection uses:
- MedlinePlus health-topic **Summary sections only**, which the official MedlinePlus usage policy identifies as public-domain content.
- MedlinePlus **medical-test information**, also identified in that policy as public-domain content.

WHO entries are deferred until the exact licence/permission basis is established. NHS entries remain candidates until page-specific content and exclusions are manually reviewed. See the [data and safety policy](docs/DATA_AND_SAFETY_POLICY.md) and [source manifest](data/source_manifest.csv).

## Setup

Requires Python 3.11 or compatible.

    python -m pip install -e .
    python -m pip install -r requirements-dev.txt
    pytest

Install the full application requirements before implementing the embedding index or Streamlit interface:

    python -m pip install -r requirements.txt

## Validate source manifest (offline dry-run)

    python -m patient_journey_assistant.ingest --manifest data/source_manifest.csv

No network requests are made by default.

## Explicitly fetch approved sources

Only do this after checking the manifest and source policy:

    python -m patient_journey_assistant.ingest --manifest data/source_manifest.csv --fetch-approved

Generated chunks and the ingestion report are written under data/local/processed/ and are excluded from Git. The fetcher respects robots.txt and skips any source not marked approved. Inspect the local text and report before using the corpus further.

## Safety and limitations

This is an educational information-navigation prototype, not a diagnostic or treatment tool, not a medical device, and not validated for clinical use. It must not diagnose, triage, prescribe, advise medication changes, or replace a healthcare professional. It should abstain when approved evidence is missing or conflicting. Do not enter identifying or personal medical information. The application is designed not to persist questions or answers.

No source performance, answer accuracy, deployment, or user-impact claims are made until these are measured.
