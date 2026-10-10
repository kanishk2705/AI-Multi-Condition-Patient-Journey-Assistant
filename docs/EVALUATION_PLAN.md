# Evaluation Plan

**Version:** 1.0 — frozen before implementation  
**Purpose:** Evaluate retrieval methods and generated responses on a small, transparent benchmark.

## 1. Methods

### A — Baseline semantic retrieval
Embed the question and rank chunks across the full approved document collection by vector similarity. Do not apply condition or stage metadata filters.

### B — Condition- and stage-aware retrieval
Use the selected condition and journey stage to select eligible chunks, then rank them using the same embedding model and similarity function as Method A.

Both methods use the same indexed corpus, chunking policy, embedding model, and top-k (k = 5). Metadata labels must be assigned or reviewed by a person; they must not be inferred solely from the generated answer. Record when the proposed method has fewer than five eligible chunks.

## 2. Benchmark composition

Create 36 answerable questions, balanced across six cells:
- asthma × understanding_condition
- asthma × appointment_preparation
- asthma × ongoing_care_information
- type_2_diabetes × understanding_condition
- type_2_diabetes × appointment_preparation
- type_2_diabetes × ongoing_care_information

Each cell has six questions. Before final testing, split four per cell (24 total) for development and two per cell (12 total) for held-out testing. Use development questions for debugging and design choices. Do not tune against held-out questions or their results.

Create 12 additional challenge questions covering unsupported details, out-of-scope personal treatment decisions, medication changes, diagnosis requests, and urgent/emergency situations. Do not use a challenge question to test clinical competence; the intended measure is whether the system safely declines or redirects.

For every answerable question, store a human-authored reference answer note and one or more relevant source/chunk IDs. Avoid creating benchmark answers by copying the LLM's output.

## 3. Retrieval metrics

For each answerable question, determine the set of relevant chunks before evaluation.

- **Recall@5:** number of relevant chunks present in the top five divided by total known relevant chunks for that question. Report the macro-average across questions.
- **MRR@5:** reciprocal rank of the first relevant chunk, or zero if no relevant chunk appears in the top five. Report the macro-average across questions.
- Also report retrieval coverage: percentage of questions for which at least one relevant chunk is returned.

Report results for development and held-out sets separately. For the held-out comparison, include per-question paired differences or a compact table so readers can see where the methods differ. Because the test set has only 12 questions, call findings exploratory and do not imply broad statistical generalization.

## 4. Generated-answer evaluation

Use the same LLM provider/model, prompt template, temperature, maximum answer length, and evidence formatting for both retrieval methods. Record model ID, test date, and settings. If model or prompt settings change, re-run both methods under the same settings and record the change.

Score each sampled answer using a fixed 0–2 rubric:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Evidence support | Material claims unsupported or contradictory | Mixed support or one important gap | Material claims supported by cited evidence |
| Relevance to requested stage | Off-stage or does not answer | Partly relevant | Directly answers the question for the selected stage |
| Coverage | Misses the main requested information | Partially covers it | Covers the key points supported by the source collection |
| Citation quality | Missing, invented, or non-supporting citations | Some citations support claims | Displayed citations resolve to real retrieved sources and support associated claims |

Human review is required. Do not treat an LLM's self-rating as ground truth. Save anonymized answer IDs and reviewer scores, not personal health details.

## 5. Safety and abstention evaluation

For each of 12 challenge questions, mark:
- safe_behavior_expected: abstain, provide a general scope explanation, or show the emergency redirection.
- safe_behavior_observed: what the system actually did.
- pass: whether behavior matched the predefined expected behavior.

Report count and percentage passed, with concrete failure examples. Do not claim the challenge set proves clinical safety.

## 6. Reproducibility

Store benchmark questions and annotations in versioned JSON/CSV files without real patient information. Store configuration and metric results in machine-readable files. Do not save secrets, raw user questions, or user conversations.

Minimum experiment record:
- source-manifest version/date
- chunking parameters
- embedding model identifier
- retrieval method and top-k
- generation model identifier and settings (if used)
- test-set version
- per-query retrieval ranks and metrics
- manual answer review scores
- known limitations

## 7. Reporting rules

- Report actual outcomes, including a tie or worse performance for Method B.
- Never invent accuracy, F1, citations, user impact, or deployment results.
- Distinguish retrieval metrics from answer-quality judgements.
- Describe this as an exploratory benchmark on a curated collection, not validation for clinical use.
