# AI Multi-Condition Patient Journey Assistant

An undergraduate AI project investigating whether condition- and journey-stage-aware retrieval improves the relevance and source-grounding of educational health-information answers.

**Status:** Phase 1 — project specification and repository scaffold. Implementation and evaluation have not yet been completed.

## Problem

A general semantic search system can retrieve text that is topically similar to a health question but not appropriate to the user's selected condition or information-seeking stage. This project compares ordinary semantic retrieval with retrieval conditioned on both condition and journey stage.

## Initial scope

- **Conditions:** asthma and type 2 diabetes.
- **Information stages:** understanding the condition; preparing for an appointment; understanding general ongoing-care information.
- **Inputs:** selected condition, selected stage, and a question.
- **Output:** an educational explanation grounded in retrieved passages, with direct links to the sources.
- **Sources:** a small curated collection of official public-health and government health-information pages.
- **Evaluation:** 36 answerable benchmark questions plus 12 safety/insufficient-evidence challenge questions.

## What makes this an investigation

The system will compare two retrieval methods on the same documents, questions, embedding model, and generation settings:

1. Semantic retrieval over the entire collection (baseline).
2. Semantic retrieval restricted or prioritized by condition and journey-stage metadata (proposed method).

Retrieval metrics, citation support, answer quality, and abstention behaviour will be evaluated and reported honestly, including negative or inconclusive results.

## Architecture

`Official sources → ingestion → clean text and metadata → chunking → embeddings → vector index → retrieval → grounded generation → cited answer`

The initial stack is Python, Sentence Transformers, ChromaDB, the Google Gen AI SDK, Streamlit, and pytest. The LLM model is configurable through environment variables.

## Safety and privacy boundaries

- This is an educational information-navigation prototype, not a medical device or clinical decision-support system.
- It must not diagnose, triage, prescribe, recommend medication changes, or replace a qualified health professional.
- It must abstain when its approved sources do not support an answer.
- Urgent or emergency requests must not be handled as routine information queries; the interface should direct users to local emergency services or an appropriate healthcare professional.
- Do not ask users for names, contact details, medical records, or other identifying/personal health information.
- Do not store or log user questions in the MVP.
- Source text will not be republished as a general-purpose mirror. Preserve source attribution and links, respect source terms, and ingest only the selected pages needed for the project.

## Repository map

- `docs/PROJECT_CHARTER.md` — frozen requirements and scope.
- `docs/EVALUATION_PLAN.md` — experiment design and metrics.
- `docs/DATA_AND_SAFETY_POLICY.md` — source handling, privacy, and safety rules.
- `data/source_manifest.csv` — initial candidate-source register.
- `src/patient_journey_assistant/` — application package.
- `tests/` — automated tests and evaluation checks.

## Current status and next steps

- [x] Freeze the initial problem, scope, architecture, safety boundaries, and evaluation plan.
- [x] Create the initial source manifest.
- [ ] Validate candidate sources and ingest the initial curated collection.
- [ ] Implement and test both retrieval methods.
- [ ] Build the question/evidence benchmark and run the comparison.
- [ ] Add grounded answer generation and the Streamlit interface.
- [ ] Deploy, document measured results, and publish screenshots/demo.

No performance results are claimed yet.
