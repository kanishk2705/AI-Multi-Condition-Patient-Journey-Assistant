# Phase 3 Acceptance — Shared Embedding Index and Retrieval Comparison

## Implemented

- [x] Validate the processed JSONL chunk corpus before indexing; reject duplicate IDs, empty text, missing required fields, and unsupported condition/stage labels.
- [x] Encode corpus passages and query text with the same configurable Sentence Transformers model; normalize embeddings.
- [x] Build a persistent local Chroma collection using cosine distance and preserve source provenance in scalar metadata.
- [x] Replace the collection on rebuild so chunks removed from a later corpus version do not remain as stale index records.
- [x] Implement Method A baseline semantic retrieval across the full collection, without metadata filters.
- [x] Implement Method B retrieval filtered by selected condition and journey stage before semantic ranking.
- [x] Use the same query encoder and similarity metric for both methods.
- [x] Return passage text, chunk ID, score, source title, source URL, condition, and stage scope.
- [x] Return fewer than k results when fewer eligible chunks exist; return an empty list when no chunks are eligible.
- [x] Sort tied similarities deterministically by chunk ID.
- [x] Add Recall@k, reciprocal rank/MRR@k, retrieval coverage, and macro-average evaluation helpers.
- [x] Add offline unit tests using fake encoders and a fake collection; these tests do not need ChromaDB, model downloads, or an LLM key.
- [x] Provide index-build and side-by-side retrieval CLI entry points.

## Commands

Install full application requirements after the lightweight test setup:

    python -m pip install -r requirements.txt

Build the persistent local index from the reviewed corpus:

    python -m patient_journey_assistant.build_index --chunks data/local/processed/chunks.jsonl --index-dir data/local/chroma

Compare both retrieval methods for one query:

    python -m patient_journey_assistant.compare_retrieval --condition asthma --stage appointment_preparation --query "What happens during lung function testing?" --k 5

Default embedding model: `sentence-transformers/all-MiniLM-L6-v2`. First use may download model weights. No Google Gen AI key is required for indexing or retrieval.

## Verification still required on a developer machine

- [ ] Install the full requirements successfully in the target Python environment.
- [ ] Run all automated tests; confirm the new retrieval/metric tests pass alongside Phase 2 tests.
- [ ] Build the index from the current 39-chunk corpus and verify the indexed count.
- [ ] Run at least one query for each condition and each stage, reviewing source IDs and stage eligibility.
- [ ] Verify that rebuilding after corpus changes removes stale index records.
- [ ] Record the actual model identifier, index count, and CLI output; do not claim an improvement before running the frozen benchmark.

## Boundaries

Phase 3 implements retrieval only. It does not generate answers, call an LLM provider, implement the Streamlit interface, or establish medical correctness. Actual retrieval metrics must come from the manually annotated benchmark defined in `docs/EVALUATION_PLAN.md`.
