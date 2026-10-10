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

## Still required before Phase 2 is considered fully accepted

- [ ] Install dependencies in a clean local environment.
- [ ] Run pytest and review the actual test results.
- [ ] Run the manifest dry-run and review the five approved and four skipped sources.
- [ ] Attempt source fetching only when the environment permits access; manually review the downloaded text and report.
- [ ] Confirm no excluded/copyrighted material or safety-sensitive emergency instructions entered the generated local corpus.
- [ ] Record the actual source count, character count, chunk count, and any failures; do not invent counts.
