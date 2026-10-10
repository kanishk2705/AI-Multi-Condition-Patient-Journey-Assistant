# Project Charter — AI Multi-Condition Patient Journey Assistant

**Version:** 1.0  
**Status:** Frozen for initial build on 10 October 2026  
**Project type:** Undergraduate applied AI/ML portfolio project  
**Target first portfolio release:** November 2026

## 1. Problem statement

People seeking information about a health condition may find material that is topically relevant but poorly matched to the information need at their current stage. A generic semantic retriever can return similar text without considering whether the user is trying to understand a condition, prepare for an appointment, or understand general ongoing-care information.

This project investigates whether explicitly conditioning retrieval on the selected condition and journey stage improves retrieval quality and the source-grounding of educational answers compared with ordinary semantic retrieval.

## 2. Research question and hypothesis

**Research question:** On a small, manually annotated health-information benchmark, does condition- and journey-stage-aware retrieval return more relevant evidence and support more faithfully grounded answers than ordinary semantic retrieval?

**Hypothesis:** Metadata-aware retrieval will improve retrieval relevance when query intent is aligned with metadata labels. The effect on generated-answer quality is empirical and may be small, mixed, or negative; the project will report observed results without forcing a positive conclusion.

## 3. Initial scope

### Conditions
- Asthma
- Type 2 diabetes

### Journey stages
- understanding_condition: basic concepts, terminology, and general understanding.
- appointment_preparation: understanding common tests, appointments, and questions to raise with a qualified professional.
- ongoing_care_information: general educational information about monitoring and ongoing care.

The stages are information-navigation categories. They do not infer a person's real clinical status or recommend a care pathway.

### User flow
1. The visitor chooses a condition.
2. The visitor chooses a journey stage.
3. The visitor enters a general educational question.
4. The system retrieves passages from a curated source collection.
5. The system either returns a concise, source-grounded explanation with numbered citations and links, or explains that its approved sources do not adequately support an answer.
6. The interface displays the cited source titles and URLs.

## 4. Out of scope

- Diagnosis, triage, prognosis, or clinical decision support.
- Personalized treatment recommendations, medication or dosage advice, or advice to start, stop, or change treatment.
- Emergency assessment or a chatbot that attempts to manage acute symptoms.
- Inferring a diagnosis, urgency level, or clinical stage from a conversation.
- Patient accounts, medical-record uploads, personal profiles, or persistent storage of user questions.
- Multi-agent orchestration, fine-tuning a language model, training a new embedding model, or building a production healthcare platform.
- Claims of clinical validation, medical-device status, or replacement of a clinician.

## 5. Knowledge sources and data

Start with a small curated set of public pages from reputable public-health or government health-information publishers, including WHO, MedlinePlus, and NHS pages where suitable. URLs, page titles, publisher, access date, source condition, stage labels, and content hash must be recorded in data/source_manifest.csv.

Source inclusion requires a basic suitability and terms/robots review. Preserve attribution and source URLs. Do not commit full copied source pages or large copyrighted text dumps to the repository. Store only the limited derived text needed for the experiments, with provenance, in local/generated data that is git-ignored. If reuse permissions or fetching rules are unclear, use manually prepared short notes or permitted access instead of bulk downloading.

## 6. Frozen technical approach

- Language: Python 3.11 or a compatible version supported by selected dependencies.
- Text preparation: transparent Python functions with deterministic tests.
- Embeddings: a small Sentence Transformers model, initially sentence-transformers/all-MiniLM-L6-v2.
- Vector index: ChromaDB, persistent locally for reproducible experiments.
- Retrieval: cosine/embedding similarity; deterministic tie handling where practical.
- Baseline: rank the whole indexed collection using semantic similarity only.
- Proposed method: filter eligible chunks by selected condition and journey stage, then rank by the same semantic similarity measure.
- Generation: an LLM accessed through the Google Gen AI SDK; model name configured by environment variable so it can be changed without source edits. Retrieval and evaluation must work without an API key.
- Interface: Streamlit.
- Tests: pytest.
- Evaluation: fixed query set, manually labelled relevant evidence, retrieval metrics, citation-support review, answer-quality rubric, and safety/abstention checks.

Use the same source corpus, embedding model, top-k, and generation settings for the baseline and proposed method. Report any unavoidable differences.

## 7. Evaluation plan summary

- 36 answerable questions: six for each of the two conditions × three stages.
- Split answerable questions into 24 development questions (four per condition-stage cell) and 12 held-out test questions (two per cell).
- 12 separate safety/insufficient-evidence challenge questions.
- Retrieval metrics: Recall@5 and MRR@5.
- Generated-answer review: citation support, factual support, relevance, and coverage using a fixed rubric.
- Safety review: correct abstention or safe redirection on the challenge set.
- Report sample size and limitations. The test set is small and supports an exploratory portfolio study, not a broad clinical claim.

The exact questions, gold answer notes, and relevant source/chunk labels must be written and frozen before running the final held-out test.

## 8. Safety and privacy

- Clear educational-only disclaimer in the UI and README.
- No diagnosis, prescribing, dosage advice, treatment changes, or individual medical decisions.
- Requests describing an immediate emergency must not receive routine generated education as a substitute for care. Show a concise message to contact local emergency services or an appropriate healthcare professional.
- Requests asking for unsupported individual medical guidance should be declined and directed to a qualified professional.
- Do not request identifying information or medical records.
- Do not persist conversation history or log raw user questions. Operational logs may contain event/error codes only; do not log prompts or retrieved medical context.
- Generated claims must cite supporting sources. If evidence is absent or conflicts, state that the collection is insufficient and abstain.

## 9. Acceptance criteria

The project is ready for its first portfolio release when:
1. Source entries and metadata are documented and traceable.
2. Ingestion/chunking is repeatable and creates stable IDs/content hashes.
3. Both retrieval methods can be run on the same indexed corpus.
4. Unit tests cover metadata filters, empty results, citation IDs, abstention, and source-link rendering.
5. Retrieval metrics are calculated from the frozen annotated benchmark.
6. Held-out results are recorded honestly, including regressions or inconclusive differences.
7. Generated answers display source links and do not invent citations.
8. Safety boundaries and no-storage policy are visible and implemented.
9. The application runs from documented setup instructions and has a deployed demo or clear recorded walkthrough.
10. README includes architecture, methodology, actual results, limitations, and a note that the system is not a medical tool.

## 10. Change control

The two conditions, three journey stages, comparison between ordinary semantic retrieval and metadata-aware retrieval, core evaluation metrics, and medical-safety boundaries are frozen for this first release. Changes require a specific reason such as unavailable source material, implementation failure, or a safety issue. New features that do not support the research question are deferred until after the application deadline.
