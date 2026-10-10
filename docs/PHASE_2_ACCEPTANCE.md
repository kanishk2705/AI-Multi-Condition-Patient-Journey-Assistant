# Phase 2 Acceptance — Source Validation and Ingestion

## Implemented

- [x] Validate manifest schema, source IDs, conditions, stage labels, dates, URL schemes/domains, duplicate IDs/URLs, statuses, extraction modes, and reuse basis.
- [x] Fetch only records explicitly marked approved; dry-run is the default.
- [x] Check robots.txt before each approved page request and fail closed if the policy cannot be established.
- [x] Reject non-HTTPS redirects, redirects outside the approved source domain, non-HTML responses, and oversized pages.
- [x] Restrict extraction to the MedlinePlus health-topic Summary section or medical-test article text.
- [x] Stop medical-test extraction before recognized bibliography headings.
- [x] Skip WHO entries deferred for permission and NHS entries that still require page-level content review.
- [x] Normalize text, accept original HTML bytes for parser encoding detection, and remove whitespace before punctuation.
- [x] Produce deterministic overlapping chunks with stable IDs and source provenance.
- [x] Write generated chunks and ingestion reports only beneath git-ignored local data.
- [x] Add offline tests for manifest validation, extractor boundaries, deterministic chunking, and robots handling.
- [x] Run the tests and actual approved-source ingestion in the developer's local environment.

## Local verification record — 10 October 2026

The developer's reported local run showed:
- Automated tests before Phase 3 additions: **18 passed**.
- Manifest: 9 rows total, 5 approved and 4 skipped by policy.
- Actual ingestion: 5 sources fetched, 0 source failures.
- Corpus: 39 chunks generated; the report's chunk count matched the JSONL file.
- Corpus audit: no duplicate chunk IDs, empty chunks, or missing required metadata were reported.
- Encoding and formatting audit: zero occurrences of the checked Unicode corruption markers, zero chunks with spaces before punctuation.
- Reference spot checks did not find the checked bibliography phrases.
- Git reported the generated local data as ignored.

These are local execution results reported by the developer. They do not by themselves prove all medical content is accurate or suitable.

## Reproduce

    python -m pip install -e . --no-deps
    python -m pip install -r requirements-dev.txt
    pytest
    python -m patient_journey_assistant.ingest --manifest data/source_manifest.csv
    python -m patient_journey_assistant.ingest --manifest data/source_manifest.csv --fetch-approved

Fetched chunks and the ingestion report are stored under `data/local/processed/` and must not be committed. The project also stores its local embedding index under `data/local/chroma/`.

