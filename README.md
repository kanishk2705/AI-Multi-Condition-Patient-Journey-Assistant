# AI Multi-Condition Patient Journey Assistant

**Not medical advice.** A research and teaching project. The assistant gives educational
information from public U.S. health sources only. In an emergency, call your local emergency number.

**Project page:** https://kanishk2705.github.io/AI-Multi-Condition-Patient-Journey-Assistant/
**Try the assistant:** https://kanishk2705.github.io/AI-Multi-Condition-Patient-Journey-Assistant/app/

> On 63 held-out test questions with an answer in the sources, the selected system put a correct passage first in 87% of questions and in the top five in 97%. Condition and stage matching itself did not beat plain semantic search (MRR difference -0.010 (-0.078 to +0.055)); the gain came from the re-ranker. The abstention gate separated answerable from unanswerable questions with AUROC 0.97. 78% of LLM answer sentences were confirmed in their cited passage by an independent checker, against 100% for copied sentences.

## 1. Problem

People living with both asthma and type 2 diabetes need information that accounts for both
conditions (for example, steroid tablets for an asthma flare-up can affect blood sugar). Trusted
patient information is written one condition at a time: **0 of the 373 public passages
collected here are about both conditions.** General chatbots mix conditions, cite nothing and rarely
admit when they don't know. This project builds an assistant that answers only from public health
pages, cites a passage for every sentence, declines when the passages don't cover the question, and
never handles emergencies, doses or diagnoses. It then tests, with a pre-registered protocol,
whether retrieval that knows the reader's condition and stage of the journey (understanding,
daily management, warning signs) finds better passages than plain semantic search.

## 2. Design

`safety rules → condition/stage classifier → retrieval → abstention gate → LLM answer with a citation per sentence → independent grounding check`

* **Sources:** 373 passages from MedlinePlus, NIDDK, NHLBI and CDC (public domain or copyright-free;
  licences and hashes in [DATA.md](DATA.md)).
* **Retrieval compared:** plain semantic (`bge-small-en-v1.5`), BM25, hybrid (RRF), hybrid with a
  condition/stage bonus, and that plus a cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`).
* **Answers:** `openai/gpt-oss-120b` on Groq (no Llama chat model was available to this account; see
  deviation D1) versus an extractive copy-sentences baseline.
* **Grounding checker:** NLI cross-encoder (`nli-deberta-v3-base`) plus a word-overlap check, independent
  of the answer model; hand-audited.
* **Questions:** 148 written by Claude (not by patients or clinicians): answerable, cross-condition,
  trap, unanswerable and safety cases; 59 development / 89 held-out test, split before any system ran.
* **Protocol and hypotheses** frozen first: [PROTOCOL.md](PROTOCOL.md).

## 3. Implementation

```
src/pjassist/   text, bm25, classifier, retrieval, safety, generate, grounding, evaluate, report, pipeline
scripts/        collect_sources, build_corpus, build_questions, embed, run_dev, answer_dev, run_final, make_report, deploy_space
tests/          49 unit tests (grounding, safety and leakage guards shown to fail on broken input; Python/JS parity)
space/app.py    FastAPI backend; render/ + render.yaml deploy it on Render (ONNX models, free tier)
docs/           project page and the assistant (GitHub Pages)
results/        dev/ (tuning), final/ (held-out test, read once), llm_cache/ (every LLM response)
```

The build sandbox could not reach the health sites, Hugging Face or Groq, so those steps run in the
`ci-run` GitHub Actions workflow; keys exist only as GitHub and Space secrets.

## 4. Results (held-out test set, 63 questions with an answer in the sources)

| System | MRR@10 | Correct passage first | Correct passage in top 5 | Wrong-condition share (top 5) | Wrong-stage share (top 5) |
|---|---|---|---|---|---|
| Plain semantic search (dense) | 0.85 (0.77–0.91) | 76% (65%–86%) | 97% (92%–100%) | 0% (0%–0%) | 27% (20%–33%) |
| Keyword search (BM25) | 0.76 (0.67–0.84) | 65% (54%–76%) | 90% (83%–97%) | 10% (6%–15%) | 32% (26%–39%) |
| Hybrid (dense + BM25) | 0.84 (0.77–0.90) | 73% (62%–84%) | 97% (92%–100%) | 2% (0%–4%) | 27% (21%–34%) |
| Hybrid + condition match | 0.84 (0.77–0.90) | 73% (62%–84%) | 97% (92%–100%) | 1% (0%–2%) | 29% (23%–35%) |
| Hybrid + condition match + re-ranker (selected) | 0.91 (0.84–0.96) | 87% (79%–95%) | 97% (92%–100%) | 1% (0%–2%) | 31% (24%–38%) |

Brackets are 95% bootstrap intervals over questions.

* Classifier vs the author's labels: condition 89% (81%–95%), stage 44% (32%–57%).
* Abstention gate: AUROC 0.97; at the dev threshold it answered 87% (78%–95%) of answerable
  and 8% (0%–23%) of unanswerable test questions.
* Grounding: LLM sentences confirmed 78% (71%–85%), LLM answers fully confirmed
  56% (44%–69%); extractive sentences confirmed 100% (100%–100%).
* Hand audit of 60 randomly sampled answer sentences: the checker agreed with a careful reading on 54 of 60 (90%; Cohen's kappa 0.23). Where they disagreed, the checker was too strict 6 times and too lenient 0 times. For LLM sentences the careful reading judged 29 of 30 supported, so the checker's LLM figure is best read as a conservative lower bound (kappa is low mainly because almost every sampled sentence was supported).
* Safety: Emergencies routed to the urgent-care message: 4 of 5. Dosing questions declined: 4 of 4. Diagnosis questions declined: 2 of 4. Ordinary questions wrongly routed (false alarms): 1.

| | Hypothesis | Result | Verdict |
|---|---|---|---|
| H1 | Condition/stage-aware retrieval (`meta`) beats plain semantic retrieval on MRR@10 | -0.010 (-0.078 to +0.055) | **Not supported** |
| H2 | `meta` retrieves fewer wrong-condition passages (top 5) than `dense` | +0.006 (+0.000 to +0.019) | **Not supported** |
| H3 | The cross-encoder re-ranker (`meta_ce`) beats `meta` on MRR@10 | +0.069 (+0.014 to +0.127) | Supported |
| H4 | Hybrid beats both dense and BM25 on MRR@10 | vs dense -0.010 (-0.078 to +0.055); vs BM25 +0.080 (+0.020 to +0.145) | **Not supported** |
| H5 | On trap questions `meta` does not beat `dense` (expected) | -0.098 (-0.318 to +0.068) | Supported |
| H6 | LLM sentences are confirmed less often than extractive sentences | -0.222 (-0.312 to -0.142) | Supported |
| H7 | At least 80% of LLM answers are fully grounded | 56% (44%–69%) | **Not supported** |
| H8 | Abstention gate AUROC is at least 0.80 | 0.97 | Supported |
| H9 | Every emergency is routed to urgent care, with no false alarms on ordinary questions | 4/5 emergencies; 1 false alarm | **Not supported** |
| H10 | On cross-condition questions `meta` covers both conditions in the top 5 more often (descriptive) | -0.143 (-0.429 to +0.000) | **Not supported** |

**What was solved, and how well.** The assistant answers everyday asthma and type 2 diabetes questions
from trusted sources with a visible source for every sentence, declines most unanswerable questions and
routes 4 of 5 test emergencies to an urgent-care message (the missed one, “too breathless to finish a sentence”, and the other routing gaps found on the test set were patched afterwards for the live site only; see deviation D11). The specific hypothesis that condition and journey-stage
labels improve retrieval was **not supported** on this corpus: a strong semantic model plus a re-ranker
did the work, and the keyword stage classifier was too weak to help. Combining two conditions in one
answer stays hard because the sources rarely connect them.

## 5. Limitations

Small curated corpus (two conditions, four U.S. agencies, English, one snapshot); questions, gold
labels, rules and audits written by the same AI collaborator (circularity risk); no patient or
clinician input or review; small test subsets; a conservative automatic grounding checker; free-tier
hosting. No clinical-effectiveness claim is made.

## 6. Next steps

Questions from people living with both conditions and clinician review; sources that discuss the
conditions together; multi-hop retrieval; a trained stage classifier; a larger human-labelled set to
calibrate the grounding checker.

## Reproduce

```
pip install -r requirements-ci.txt
pytest -q                                   # 49 tests
python scripts/run_dev.py                   # dev tuning from cached model scores (data/cache)
python scripts/make_report.py               # page + README from results/final/results.json
```
Network-dependent steps (`collect_sources.py`, `embed.py`, `answer_dev.py`, `run_final.py`, `check_onnx.py`)
run in the `ci-run` workflow. `run_final.py` refuses to run again: the test split has been read.

## Data credit

Source passages: MedlinePlus (National Library of Medicine); NIDDK; NHLBI, National Institutes of
Health; CDC. Available free of charge on medlineplus.gov, niddk.nih.gov, nhlbi.nih.gov and cdc.gov.
Use here does not imply endorsement by NLM, NIH, CDC, HHS or the U.S. Government. Code: MIT licence.
