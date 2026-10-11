# Phase 6 Results — Extractive Answer and Citation Evaluation

**Evaluation date:** 11 October 2026  
**CI workflow:** [Phase 6 answer-quality and safety checks](https://github.com/kanishk2705/AI-Multi-Condition-Patient-Journey-Assistant/actions/runs/38116149245)  
**Evaluated code commit:** eb3f82453557fb8f0343b8b23ab5116cf3fad319  
**Corpus:** five approved MedlinePlus sources, 39 chunks  
**Embedding/index:** sentence-transformers/all-MiniLM-L6-v2, 384 dimensions, ChromaDB, cosine distance, top-5  
**Benchmark label status:** assistant-adjudicated provisional

## What was run

The workflow rebuilt the approved-source corpus, verified the exact frozen chunk-ID inventory, rebuilt the vector index, ran the automated test suite, checked the 12 deterministic safety challenge routes, then evaluated the extractive answer fallback on every answerable development and held-out question.

The fallback is not an LLM answer. It selects verbatim sentences from retrieved source passages, validates source metadata and citations, and displays links taken from the index. It makes no external-model call. In baseline mode, the retrieval top five are unfiltered; the answer layer only displays statements from sources that pass the chosen condition/stage safety constraints.

## Automated checks

| Check | Result |
|---|---:|
| Unit tests on Python 3.11 | 76 passed |
| Frozen source snapshot | Exact 39 chunk IDs matched |
| Approved source fetch | 5 of 5 sources |
| Deterministic safety preflight | 12 of 12 expected routes |
| Development questions evaluated | 24 |
| Held-out questions evaluated | 12 |
| Answer-generation/LLM-as-judge API calls | Skipped: no Gemini API key is configured |
| Invalid source-link rows in extractive reports | 0 |
| Citation condition/stage concordance in extractive reports | 1.0000 on both splits and methods |

The safety score is strictly for pre-retrieval routing. It does not show that a generated answer is medically correct or that the application can reliably detect every emergency.

## Extractive answer/citation results

Citation precision and citation recall below are macro averages of overlap between cited chunk IDs and the benchmark's labelled relevant chunk IDs. They are not human scores of factual entailment, helpfulness, or clinical accuracy.

### Development split — 24 questions

| Metric | Method A: baseline semantic | Method B: condition + stage |
|---|---:|---:|
| Questions | 24 | 24 |
| Extractive answer rate | 1.0000 | 1.0000 |
| Macro citation relevance precision | 0.5486 | 0.5139 |
| Macro citation relevance recall | 0.8125 | 0.7708 |
| Questions with at least one labelled-relevant citation | 0.9167 | 0.9167 |
| Context-concordant citation rate | 1.0000 | 1.0000 |
| Validated source links emitted | 53 | 53 |

**Paired difference (Method B minus Method A):** citation precision −0.0347; citation recall −0.0417; relevant-citation coverage 0.0000; answer rate 0.0000.

### Held-out split — 12 questions

| Metric | Method A: baseline semantic | Method B: condition + stage |
|---|---:|---:|
| Questions | 12 | 12 |
| Extractive answer rate | 1.0000 | 1.0000 |
| Macro citation relevance precision | 0.5556 | 0.5139 |
| Macro citation relevance recall | 0.8333 | 0.8333 |
| Questions with at least one labelled-relevant citation | 0.9167 | 0.9167 |
| Context-concordant citation rate | 1.0000 | 1.0000 |
| Validated source links emitted | 24 | 25 |

**Paired difference (Method B minus Method A):** citation precision −0.0417; citation recall 0.0000; relevant-citation coverage 0.0000; answer rate 0.0000.

## Interpretation

1. Both methods returned at least one source-derived sentence for all 24 development and all 12 held-out questions. This answer rate means only that an extractive output survived the current deterministic source/citation checks; it does not prove the output covered every required answer point.
2. Contextual retrieval did not improve extractive citation overlap against the current labels. Its macro citation precision was lower by about 3.5 percentage points on development and 4.2 points on held-out; citation recall fell on development and tied on held-out.
3. Both methods maintained 100% condition/stage-concordant citations in the final emitted responses. The answer layer prevents citations from mismatched conditions/stages from being presented.
4. The earlier raw retrieval benchmark remains separate: it also showed no held-out improvement from contextual filtering. These extractive results do not establish that contextual RAG generation would be worse under a real language model.
5. The 12/12 safety result measures only deterministic preflight routing. It is not a complete safety or clinical-quality result.

## Critical limitations

- The relevant chunk labels and reference notes were adjudicated by an AI assistant against the 39-chunk corpus, not independently validated by a human or domain expert.
- The held-out questions are public and the sample contains only 12 questions.
- Citation overlap measures shared IDs with provisional labels; it cannot prove a sentence fully supports a question or that all key facts were covered.
- No Gemini API key was configured in GitHub Actions. Live LLM generation and LLM-as-judge scores were therefore **not run**.
- The no-key fallback copies source sentences verbatim instead of generating a fluent synthesis. This makes it reproducible and source-grounded, but it is not a substitute for evaluation of generated answer quality.
- The project is an educational information-navigation prototype, not a diagnostic/treatment service or clinically validated application.

## Reproducibility

The machine-readable JSON results and source-ingestion report are stored in the 30-day GitHub Actions artifact named phase6-answer-quality-and-safety associated with the linked workflow. Generated chunk text, the Chroma index, questions, retrieved text and model prompts are not committed as build artifacts to the repository.
