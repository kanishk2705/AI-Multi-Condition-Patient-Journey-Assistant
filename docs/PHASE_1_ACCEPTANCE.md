# Phase 1 Acceptance Checklist

Phase 1 means requirements and repository foundation, not a functioning RAG application.

- [x] Freeze the problem statement and research question.
- [x] Freeze the two conditions and three journey stages.
- [x] Freeze the baseline and proposed retrieval methods.
- [x] Define the benchmark size and development versus held-out split.
- [x] Define retrieval, answer-quality, and safety evaluation.
- [x] Define medical-safety and privacy boundaries.
- [x] Create candidate source manifest with canonical URLs.
- [x] Add Python dependency list, environment example, and safe Git ignore rules.
- [x] Add initial package and frozen-scope test.
- [ ] Verify all selected pages, page metadata, source policies, and reuse restrictions before ingesting content.
- [ ] Review dependency versions and complete a clean local environment installation.

## Next phase

Implement source-manifest validation and document ingestion, with stable source/chunk identifiers, metadata validation, repeatable chunking, and tests. Do not start LLM answer generation until retrieval and evaluation are working.
