# AI Multi-Condition Patient Journey Assistant

An undergraduate applied-AI project investigating whether condition- and journey-stage-aware retrieval improves the relevance of evidence for educational health-information questions.

**Current status:** Phase 2 source ingestion has been locally verified. Phase 3 retrieval code is implemented on the `phase-3-retrieval` branch; building the local embedding index and running real-corpus comparisons remain to be verified. No retrieval-performance or answer-quality results are claimed.

## Research question

Does condition- and journey-stage-aware retrieval return more relevant evidence than ordinary semantic retrieval on a small, manually annotated benchmark?

## Frozen scope

- **Conditions:** asthma and type 2 diabetes.
- **Journey stages:** `understanding_condition`, `appointment_preparation`, and `ongoing_care_information`.
- **Method A:** semantic retrieval over the full approved corpus, without condition or stage filters.
- **Method B:** filter by the selected condition and journey stage, then rank eligible passages using the same embedding model and cosine-distance index.
- **Planned benchmark:** 36 answerable questions split into 24 development and 12 held-out questions, plus 12 safety/insufficient-evidence challenges.
- **Retrieval metrics:** Recall@5, MRR@5, and retrieval coverage. Report observed results, including ties or worse results for Method B.

See the [project charter](docs/PROJECT_CHARTER.md), [evaluation plan](docs/EVALUATION_PLAN.md), and [Phase 3 acceptance checklist](docs/PHASE_3_ACCEPTANCE.md).

## Phase 2 — source ingestion

The manifest contains nine candidate source records. Five MedlinePlus sources are approved for the current extraction policy; two NHS entries remain candidates, and two WHO entries are deferred pending a suitable reuse basis. The ingestion tool checks robots.txt, HTTPS, redirect domains, content type and size, and it extracts only the allowed health-topic Summary sections or medical-test article text. Reference sections are excluded from medical-test extraction.

**Local verification reported on 10 October 2026:** 18 tests passed; five sources fetched; zero source failures; 39 chunks written; report count matched file count; no duplicate chunk IDs, empty chunks, required-metadata omissions, checked encoding markers, or spaces-before-punctuation issues were found. Generated corpus and index files remain local and are excluded from Git.

See [data and safety policy](docs/DATA_AND_SAFETY_POLICY.md) and [source manifest](data/source_manifest.csv).

## Environment setup

Use Python 3.11 to match the version used by GitHub Actions for the project's automated tests. Check which Python versions are installed:

    py -0p

If Python 3.11 is installed, create an environment and install the requirements:

    py -3.11 -m venv .venv311
    .\.venv311\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e . --no-deps
    python -m pip install -r requirements-dev.txt
    pytest
    python -m pip install -r requirements.txt

The first real index build may download the Sentence Transformers model weights. The retrieval-only unit tests use fake encoders and do not require model downloads or an API key.

## Rebuild the local index

First run the approved-source ingestion command if the local chunk corpus is missing:

    python -m patient_journey_assistant.ingest --manifest data/source_manifest.csv --fetch-approved

Then build the local Chroma index:

    python -m patient_journey_assistant.build_index --chunks data/local/processed/chunks.jsonl --index-dir data/local/chroma

The index build replaces the selected collection so removed chunks from earlier corpus versions cannot remain as stale records. The embedding model and cosine distance metric are recorded in collection metadata.

## Compare retrieval methods

Example query for the asthma appointment-preparation stage:

    python -m patient_journey_assistant.compare_retrieval --condition asthma --stage appointment_preparation --query "What happens during lung function testing?" --k 5

The command prints results from both methods, including rank, cosine similarity, chunk ID, source title, source URL, and evidence excerpt. It does not generate an answer or call an LLM API. Method B may return fewer than five passages when fewer chunks are eligible.

## Safety and limitations

This is an educational information-navigation prototype, not a diagnostic or treatment tool, not a medical device, and not validated for clinical use. It must not diagnose, triage, prescribe, advise medication changes, or replace a healthcare professional. It should abstain when approved evidence is missing or conflicting. Do not enter identifying or personal medical information.

The corpus is small and curated. Retrieval scores are not medical accuracy or clinical-safety measures. No improvement, answer accuracy, deployment, or user-impact claims are made until those outcomes are actually evaluated.
