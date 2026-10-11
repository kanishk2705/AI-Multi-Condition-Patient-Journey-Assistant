# Phase 4 Results — Retrieval Benchmark

**Evaluation date:** 11 October 2026 (UTC)  
**Workflow:** [Phase 4 retrieval benchmark run](https://github.com/kanishk2705/AI-Multi-Condition-Patient-Journey-Assistant/actions/runs/38114302251)  
**Evaluated commit:** `b658c1b13dd95c1673ce6d5fda5dc88582f93053`  
**Benchmark:** `patient-journey-retrieval-v1-draft`  
**Annotation status:** `assistant_adjudicated_provisional`

## Executive result

The first full retrieval benchmark completed successfully. Condition-and-stage-aware retrieval did **not** improve macro Recall@5 or retrieval coverage on either split. It produced a very small development-set MRR@5 increase and no MRR change on the held-out split. For this 39-chunk corpus and provisional annotations, the hypothesis that contextual filtering improves retrieval relevance is **not supported by the observed results**.

These are preliminary exploratory measurements. The question-to-chunk labels were checked by an AI assistant against the exact chunk text but were not independently human-validated. The public held-out questions are visible in the repository, and the held-out set has only 12 questions. Do not generalize these values to larger corpora or real-world healthcare use.

## Reproducibility record

| Setting | Recorded value |
|---|---|
| Python | 3.11 |
| Approved sources | 5 of 5 fetched successfully |
| Corpus snapshot | Exact 39 chunk IDs matched the frozen inventory |
| Indexed passages | 39 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Distance metric | Cosine |
| Top-k | 5 |
| Method A | Semantic ranking across the full corpus, no context filter |
| Method B | Filter by selected condition and stage, then rank with the same embeddings and distance |
| Automated tests | 37 passed |
| LLM generation | Not involved in this retrieval benchmark |

The workflow only fetches manifest entries marked approved. It refuses to evaluate if the fetched source revision produces chunk IDs different from the frozen 39-chunk inventory. Source text, vector-index files, and the local evaluation JSON files were not committed to Git. Metric JSONs were uploaded as the run artifact for 30 days.

## Results

All metrics are macro-averaged per-question metrics at (k=5). Retrieval coverage is the fraction of questions with at least one labelled relevant chunk in the top five.

| Split | Method | Questions | Recall@5 | MRR@5 | Retrieval coverage |
|---|---|---:|---:|---:|---:|
| Development | Method A — baseline semantic | 24 | 0.8958 | 0.7597 | 0.9167 |
| Development | Method B — condition + stage | 24 | 0.8958 | 0.7667 | 0.9167 |
| Held-out | Method A — baseline semantic | 12 | 0.8750 | 0.9167 | 0.9167 |
| Held-out | Method B — condition + stage | 12 | 0.8750 | 0.9167 | 0.9167 |

### Paired difference (Method B minus Method A)

| Split | (Delta) Recall@5 | (Delta) MRR@5 | (Delta) coverage |
|---|---:|---:|---:|
| Development | 0.0000 | +0.0069 | 0.0000 |
| Held-out | 0.0000 | 0.0000 | 0.0000 |

The development MRR change comes from one question, `D-P-04`, where the first relevant passage moved from rank 3 to rank 2. That single shift increases the macro MRR by ( (1/2 - 1/3) / 24 approx 0.0069 ). No question gained Recall@5 or coverage from contextual filtering in this run.

## Questions where no relevant passage appeared in the top five

The following questions were missed by both methods according to the current relevance labels:

| Split | Question ID | Topic | Relevant gold passage(s) |
|---|---|---|---|
| Development | `A-P-04` | General preparation guidance for lung function testing | `9fa74ac34d0bd2ec19ca8c7d` |
| Development | `D-O-05` | Ongoing checks in addition to glucose monitoring | `75aeea86b757f01e4713f15d` |
| Held-out | `D-O-03` | Interpreting blood glucose results and medical history | `c6f0364f6b5fed2954efb41a`, `4fc4b5ccb3252b4905d63497` |

These are useful error-analysis targets for future retrieval engineering. Do not remove or relabel them simply because retrieval failed; a relevance label should describe whether the passage supports the answer, not whether the model successfully retrieves it.

Additional partial-recall example: `D-O-01` retrieves one of two labelled relevant passages at rank 5, giving Recall@5 of 0.5 for that question.

## Interpretation

1. The context filter correctly constrains eligibility, but this corpus's stage tags are broad and most stage-eligible passages also score highly under baseline semantic search.
2. On the observed benchmark, contextual filtering had no effect on the set of questions with at least one relevant top-five result.
3. The single development MRR gain is too small to present as convincing evidence of general improvement; the held-out comparison is an exact tie on all three metrics.
4. The missed questions point to retrieval failures that could be explored with development-set engineering. Do not tune and repeatedly re-report the held-out set as if it remained an independent test.
5. Retrieval metrics do not measure whether generated explanations are factually correct, whether citations support each generated claim, or whether the assistant behaves safely.

## Limitations

- Only 39 chunks from five approved source pages were evaluated.
- The answerable benchmark contains 24 development and 12 held-out questions; the sample is small.
- The benchmark is public, so the held-out split is not a secret blind test.
- Relevance judgments and reference answer notes are assistant-adjudicated provisional annotations, not independent expert gold labels.
- The selected sources do not cover all three stages evenly; this is a property of the approved corpus.
- No generated answers were tested, and the 12 safety challenge rubrics were not scored because generation has not yet been implemented.
- No statistical significance or broad generalization claim is made.

## Next engineering milestone

Proceed to answer generation using the retrieved passages as the sole evidence context. The generation layer should return source-grounded answers with verified links/citations, refuse unsupported individual medical decisions, and abstain when evidence is missing or conflicting. The later answer-quality evaluation must separately score evidence support, stage relevance, coverage and citation quality. The current retrieval result must remain unchanged in reports: do not describe the contextual method as a proven improvement.
