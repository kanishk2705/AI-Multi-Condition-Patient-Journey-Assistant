# Phase 5 Acceptance — Evidence-Grounded Answer Generation and Interface

## Goal

Complete the first end-to-end educational RAG path: user chooses one of the frozen conditions and stages, asks a general educational question, the system retrieves eligible approved passages, the model drafts claim-level evidence-linked output, and the application displays only answers whose citations pass deterministic validation.

## Implemented in the Phase 5 branch

- [x] Conservative deterministic request-scope checks execute before retrieval and before any LLM call.
- [x] Emergency-like wording routes to an immediate generic emergency-services message; the program does not attempt triage.
- [x] Individual diagnosis, individual test-result interpretation, and personal medication/treatment-change requests route to a professional boundary message.
- [x] Contextual retrieval is lazy-loaded, so a blocked question does not initialize the embedding index.
- [x] Google Gen AI SDK adapter uses an environment-configured model, structured JSON output and low temperature; unit tests use a fake provider and need no key/network.
- [x] Claim-level citations must reference retrieved chunk IDs from the selected condition/stage.
- [x] Source titles and HTTPS links are taken only from indexed metadata, never from generated model output.
- [x] Unsupported model citations, URL-bearing claim text, personal medication directives, malformed output and insufficient evidence fail closed.
- [x] Streamlit interface includes condition/stage controls, a scope disclaimer, validated source links, and no custom persistent question storage or query logging.
- [x] Added `.env.example`; no real secrets are included.
- [ ] Run the application interactively in an authenticated environment.
- [ ] Evaluate generated answers with the fixed four-dimension rubric on both retrieval methods.
- [ ] Score behavior on the 12 safety challenge prompts after provider configuration.
- [ ] Deploy a public demo only after secret management and runtime checks are in place.

## Core modules

- `src/patient_journey_assistant/safety.py`
- `src/patient_journey_assistant/answer_generation.py`
- `app.py`
- `tests/test_safety.py`
- `tests/test_answer_generation.py`

## Run locally

Install project dependencies, ingest/build the local index, configure `GEMINI_API_KEY`, and run:

    streamlit run app.py

The actual API key must be supplied as a local environment variable or deployment secret. It is intentionally not stored in GitHub files or committed to the repository.

## Important limitations

- The safety checks are conservative keyword rules, not a comprehensive medical safety system and not a triage model.
- Citation validation checks provenance and condition/stage eligibility; it cannot prove every generated statement is semantically entailed by its cited passage.
- An automated pass of mocked-provider tests is not proof of safety or answer correctness.
- A live answer-generation API key was not available to this automated build session, so end-to-end API inference and deployment are not claimed.
- The retrieval benchmark found no held-out improvement from context filtering. Generated-answer evaluation is a separate open question and must use the same generation settings and retrieved-evidence formatting for both methods.
