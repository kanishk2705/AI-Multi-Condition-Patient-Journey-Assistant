# Phase 6 Acceptance — Answer-Quality Evaluation and Portable Runtime

## Implemented

- [x] Add a paired answer-generation evaluator comparing baseline semantic retrieval against condition/stage-aware retrieval using the same query, generation model alias, temperature, output schema and evidence serialization.
- [x] Reuse the exact recorded top-k retrieval results when generating and judging each method's answer.
- [x] Add structured 0–2 answer-quality rubric scores for evidence support, stage relevance, coverage and citation quality.
- [x] Label automated LLM-as-judge results as provisional diagnostics, not human ground truth.
- [x] Add balanced sampling (one question per condition-stage cell by default) to limit external model calls; allow explicitly requested full-split runs.
- [x] Add a machine-readable deterministic preflight report for the 12 frozen safety challenge prompts.
- [x] Add a Docker runtime that fetches only manifest-approved sources, builds the local index on first start, and starts Streamlit.
- [x] Add Docker build CI; API secrets and local corpus/index files remain excluded from the image.
- [x] Add offline tests for selection, paired evaluation, abstention, and challenge preflight routing.

## Automated validation commands

    pytest
    python -m patient_journey_assistant.evaluate_safety
    python -m patient_journey_assistant.evaluate_answers --split dev --per-cell 1

The answer-quality evaluator requires GEMINI_API_KEY or GOOGLE_API_KEY. It writes reports only under data/local/evaluation and sends selected questions/evidence to the configured provider. Do not use personal health details. The default evaluates six development questions (one per condition-stage cell), producing up to 12 answer generations plus judge calls; it may use paid provider quota.

## Container

Build:

    docker build -t patient-journey-assistant .

Run with a configured provider key:

    docker run --rm -p 8501:8501 -e GEMINI_API_KEY=... patient-journey-assistant

On first startup the container fetches the five approved source records from the manifest, creates 39-derived chunks if the source snapshot is unchanged, downloads the embedding model if needed, builds the Chroma index, and launches Streamlit. Generated corpus and index live inside the writable container filesystem unless a persistent volume is configured.

## Limitations and release gates

- LLM-as-judge scores are not independent human labels; they must be reported as preliminary.
- The 12-command safety check measures deterministic preflight routing only. It does not establish that the generated answers are safe or medically correct.
- Citation provenance checks verify retrieved IDs, context labels and approved HTTPS hosts; they cannot prove semantic entailment.
- No cloud account, provider secret, or hosting service is provisioned by this repository. A public deployment is not claimed just because the container can be built.
- The app is an educational information-navigation prototype, not a diagnostic or treatment tool.
