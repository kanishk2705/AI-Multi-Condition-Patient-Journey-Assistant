# Phase 4 Acceptance — Benchmark and Retrieval Evaluation

## Status

**Draft implementation; human annotation review required.** The question set, reference notes, and relevance labels were drafted from the 39-chunk corpus uploaded on 10 October 2026. They are not yet frozen ground truth. No retrieval-performance claim is made by creating these files or by passing structural tests.

## Benchmark files

- `data/benchmark/answerable_questions.json` — 36 answerable questions, split into 24 development and 12 held-out questions; each question has a reference-answer note, one or more relevant chunk IDs, and a rationale for every assigned chunk.
- `data/benchmark/challenge_questions.json` — 12 safety/insufficient-evidence prompts with expected behavior rubrics. They are not run through retrieval metrics and have no observed pass/fail scores yet.
- `data/benchmark/chunk_inventory.csv` — metadata-only inventory of the 39 uploaded chunk IDs and source/stage provenance. It does not contain source text.
- `data/benchmark/relevance_review.csv` — adjudication log with the original proposals, explicit KEEP/REMOVE decisions, rationales, and source links. It contains 55 rows including the rejected proposals and their replacements.

Question allocation is balanced across the six condition-stage cells: four development and two held-out questions per cell. The held-out split is logically separated in the data and the evaluator defaults to development; because the repository is public, the held-out prompts are not secret. Avoid tuning against held-out results.

## What the evaluator measures

The evaluator runs both methods on the exact same query, embedding model, cosine index, and `k`:

- **Method A / baseline:** semantic ranking over the whole indexed corpus without condition/stage filters.
- **Method B / contextual:** filter by the selected condition and journey stage, then rank the eligible chunks semantically.

For each split it stores per-question rankings and source metadata, Recall@k, MRR@k, retrieval coverage, and paired contextual-minus-baseline differences. The default `k` is 5. It writes result JSON under `data/local/evaluation/`, which is intended to remain local and ignored by Git.

## Commands

Run the structural/offline tests:

    pytest

Build or refresh the local index from the locally generated corpus:

    python -m patient_journey_assistant.build_index --chunks data/local/processed/chunks.jsonl --index-dir data/local/chroma

Evaluate the development split first:

    python -m patient_journey_assistant.evaluate_retrieval --split dev

After manually reviewing and freezing the question set and annotations, run held-out evaluation separately:

    python -m patient_journey_assistant.evaluate_retrieval --split heldout

Options such as `--k`, `--chunks`, `--index-dir`, `--collection`, and `--output` can be supplied when needed. The evaluator verifies that every annotated relevant chunk ID exists in the current corpus before retrieval starts.

## Required human review before freezing

Review every item in `answerable_questions.json`, using the source URLs and local chunk text:

1. Confirm each question belongs to its declared condition and journey stage and is answerable using this corpus.
2. Confirm each reference-answer note is fully supported by the listed source chunks.
3. Confirm each relevant chunk really supports the requested answer. Add or remove chunk IDs as appropriate; do not assign labels based on whichever chunks a retrieval method happens to return.
4. Confirm no important answer-bearing chunk is missing from each relevance set. The current relevance sets are proposals, not exhaustive expert judgments.
5. Confirm the 24/12 split is unchanged after review. Use development questions for debugging; keep held-out results out of design decisions.
6. Confirm the 12 challenge rubrics describe expected system behavior without treating this small set as proof of clinical safety.

The metadata coverage is uneven: the asthma appointment-preparation cell relies on the lung-function-test source, while understanding/ongoing asthma questions rely mostly on the asthma topic summary. This mirrors the supplied corpus and should be reported as a limitation, not hidden by rebalancing the sources.

## Acceptance checklist

- [x] Inspect the uploaded corpus: 39 unique, non-empty chunk records across five MedlinePlus source records.
- [x] Draft 36 answerable questions across six cells with 24/12 development/held-out allocation.
- [x] Attach proposed relevant chunk IDs and per-chunk rationales to each answerable question.
- [x] Adjudicate the proposed labels against the exact uploaded corpus text; retain rejected proposals in the CSV audit log.
- [x] Correct the unsupported prediabetes-definition question, remove two overinclusive gold labels, and preserve the six-cell 24/12 balance.
- [x] Draft 12 safety/insufficient-evidence challenge prompts with expected-behavior rubrics.
- [x] Add an offline evaluator for per-question and macro-average paired retrieval metrics.
- [ ] Human-review and freeze questions, reference notes, and chunk relevance labels.
- [ ] Run automated tests in CI after integration.
- [ ] Build/use the local 39-chunk index and run the development benchmark.
- [ ] Inspect development results and correct only defensible benchmark or implementation issues.
- [ ] Run the held-out split once after freezing the benchmark and report actual outcomes, including ties or regressions.
- [ ] Implement answer generation before scoring actual challenge pass/fail or generated-answer quality.

## Reporting boundaries

- Benchmark construction and test-suite success do not demonstrate retrieval improvement.
- A high retrieval score does not establish clinical accuracy or safety.
- No challenge pass rate exists until the generation behavior is implemented and reviewed.
- The public repository exposes held-out questions. The held-out split is therefore a process separation, not a secret or external blind test.
- Keep `data/local/processed/`, `data/local/chroma/`, evaluation outputs, API keys, and any real user input out of Git.
