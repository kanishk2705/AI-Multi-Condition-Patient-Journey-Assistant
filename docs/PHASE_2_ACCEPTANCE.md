# Phase 2 Acceptance — Source Validation and Ingestion

## Implemented in this phase

- [x] Validate manifest schema, source IDs, conditions, stage labels, dates, URL schemes/domains, duplicate IDs/URLs, statuses, extraction modes, and reuse basis.
- [x] Fetch only records explicitly marked approved.
- [x] Default to dry-run; network retrieval requires --fetch-approved.
- [x] Check robots.txt before each approved page request and fail closed if the policy cannot be established.
- [x] Reject non-HTTPS redirects, redirects outside the approved source domain, non-HTML responses, and oversized pages.
- [x] Restrict extraction to the MedlinePlus health-topic Summary section or MedlinePlus medical-test article text.
- [x] Skip WHO entries deferred for permission and NHS entries that still require page-level content review.
- [x] Normalize text and produce deterministic overlapping chunks with stable IDs and complete source provenance.
- [x] Write generated chunks and an ingestion report only beneath git-ignored local data.
- [x] Add offline unit tests for manifest validation, extractor boundaries, deterministic chunking, and robots.txt behaviour.

## Run

    python -m pip install -e .
    python -m pip install -r requirements-dev.txt
    pytest
    python -m patient_journey_assistant.ingest --manifest data/source_manifest.csv

The final command validates the manifest and reports approved/skipped sources without making network requests.

To attempt a source fetch after reviewing the manifest:

    python -m patient_journey_assistant.ingest --manifest data/source_manifest.csv --fetch-approved

Fetched text chunks and the ingestion report are written below data/local/processed/, which is excluded from Git. If any source fails robots, network, content-type, redirect, or extraction checks, the run records the failure and exits non-zero.

## Verification record

GitHub Actions run 13 passed on 10 October 2026:
- Clean Python 3.11 environment created.
- Offline manifest dry-run completed successfully: 9 rows, 5 approved, 4 skipped; no network requests made.
- Automated tests: 15 passed.

## Still required before Phase 2 is considered fully accepted

- [ ] Run the same setup and tests in the developer's local environment.
- [ ] Fetch approved sources only after reviewing the manifest; inspect the actual extracted text and ingestion report.
- [ ] Confirm no excluded/copyrighted material or unsafe treatment/emergency instructions entered the generated local corpus.
- [ ] Record actual fetched-source count, character count, chunk count, and failures from the real fetch run; do not invent counts.
