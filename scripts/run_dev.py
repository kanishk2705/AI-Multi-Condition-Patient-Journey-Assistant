"""Development run (PROTOCOL.md section 9, step 2). Uses dev questions only.

Needs data/cache/{passages_emb,dev_qemb,dev_ce}.npy (computed on CI by scripts/embed.py).
Writes results/dev/: retrieval.json (all systems, inferred and oracle metadata), tuning.json,
safety.json, gate.json, selection.json (the frozen choices for the final run) and
extractive_answers.json.
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pjassist import evaluate as ev, safety
from pjassist.bm25 import BM25
from pjassist.config import RESULTS, DATA
from pjassist.generate import SYSTEM_PROMPT, extractive_answer
from pjassist.models import EMB_MODEL, CE_MODEL, NLI_MODEL, LLM_MODEL, LLM_SETTINGS
from pjassist.retrieval import Params

OUT = RESULTS / "dev"
OUT.mkdir(parents=True, exist_ok=True)
Q = json.loads((DATA / "questions/dev.json").read_text())
P = ev.load_passages()
qemb = np.load(DATA / "cache/dev_qemb.npy")
ce = np.load(DATA / "cache/dev_ce.npy")
assert qemb.shape[0] == len(Q) and ce.shape == (len(Q), len(P))

# 1. safety routing on dev
saf = {q["qid"]: safety.check(q["question"]).kind for q in Q}
safety_rows = [{"qid": q["qid"], "expected": q["expected_route"], "got": saf[q["qid"]], "question": q["question"]} for q in Q]
(OUT / "safety.json").write_text(json.dumps({"rows": safety_rows,
    "confusion": Counter(f"{r['expected']}->{r['got']}" for r in safety_rows)}, indent=1))

# 2. tuning (dev only)
tun = ev.tune(Q, P, qemb, ce)
(OUT / "tuning.json").write_text(json.dumps(tun, indent=1))
params = {"meta": Params(**tun["meta"]), "meta_ce": Params(**tun["meta_ce"])}

# 3. retrieval evaluation of all systems, inferred and oracle metadata
res = ev.evaluate(Q, P, qemb, ce, ev.SYSTEMS, params)
res_or = ev.evaluate(Q, P, qemb, ce, ("meta", "meta_ce"), params, oracle=True)
summary = {"all": ev.summarize(res, Q), **{t: ev.summarize(res, Q, {t}) for t in ev.RETRIEVAL_TYPES},
           "oracle_all": ev.summarize(res_or, Q)}
cmp = {
    "meta_vs_dense_mrr": ev.paired(res, "meta", "dense", "rr", Q),
    "meta_vs_dense_wrong_condition": ev.paired(res, "meta", "dense", "wrong_condition@5", Q),
    "meta_vs_dense_wrong_stage": ev.paired(res, "meta", "dense", "wrong_stage@5", Q),
    "meta_ce_vs_meta_mrr": ev.paired(res, "meta_ce", "meta", "rr", Q),
    "meta_ce_vs_dense_mrr": ev.paired(res, "meta_ce", "dense", "rr", Q),
    "hybrid_vs_dense_mrr": ev.paired(res, "hybrid", "dense", "rr", Q),
    "hybrid_vs_bm25_mrr": ev.paired(res, "hybrid", "bm25", "rr", Q),
}
# classifier accuracy against the author's labels (dev)
inf = res["meta"]["inferred_meta"]
cls = {"condition_exact": float(np.mean([set(inf[q["qid"]]["conditions"]) == set(q["conditions"]) for q in Q if q["qid"] in inf and q["type"] != "unanswerable"])),
       "stage_exact": float(np.mean([inf[q["qid"]]["stage"] == q["stage"] for q in Q if q["qid"] in inf and q["type"] != "unanswerable"]))}
(OUT / "retrieval.json").write_text(json.dumps({
    "summary": summary, "paired": cmp, "classifier_vs_labels": cls,
    "per_question": {s: r["per_question"] for s, r in res.items()},
    "rankings": {s: ev.ids_rankings(r["rankings"], P) for s, r in res.items()},
    "inferred_meta": inf}, indent=1))

# 4. abstention gate on the selected system (meta_ce), coverage-risk
SEL = "meta_ce"
non_safety = [q for q in Q if q["expected_route"] == "ok"]
rk = res[SEL]["rankings"]
scores = ev.gate_scores(Q, ce, {q["qid"]: rk[q["qid"]] for q in non_safety})
tau = ev.choose_tau(non_safety, scores)
cr = ev.coverage_risk(non_safety, scores, ev.ids_rankings(rk, P))
au = ev.auroc([scores[q["qid"]] for q in non_safety if q["answerable"]],
              [scores[q["qid"]] for q in non_safety if q["type"] == "unanswerable"])
(OUT / "gate.json").write_text(json.dumps({"system": SEL, "tau": tau, "auroc": au, "aurc": cr["aurc"],
                                          "curve": cr["curve"], "scores": scores}, indent=1))

# 5. extractive answers on dev (same retrieval, same gate)
bm = BM25([p["text"] for p in P])
ext = {}
for q in non_safety:
    shown = [P[i] for i in rk[q["qid"]][:4]]
    a = extractive_answer(q["question"], shown, bm.idf)
    gated = scores[q["qid"]] < tau["tau"]
    ext[q["qid"]] = {"text": "" if gated else a["text"], "abstained": gated or a["abstained"],
                     "gate_abstained": gated, "shown": [p["id"] for p in shown]}
(OUT / "extractive_answers.json").write_text(json.dumps(ext, indent=1))

selection = {
    "retrieval_system": SEL, "params": {"meta": tun["meta"], "meta_ce": tun["meta_ce"]},
    "k_shown": 4, "gate_tau": tau["tau"], "models": {"embedding": EMB_MODEL, "cross_encoder": CE_MODEL,
    "nli": NLI_MODEL, "llm": LLM_MODEL, "llm_settings": LLM_SETTINGS},
    "extractive": {"max_sents": 3, "top_passages": 3},
    "llm_prompt": "SYSTEM_PROMPT (v2, deviation D10)",
    "llm_prompt_sha256": __import__("hashlib").sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    "dev_prompt_comparison": "v1: 0.493 sentences supported, 0.410 fully grounded; v2: see results/dev/summary.json",
}
(OUT / "selection.json").write_text(json.dumps(selection, indent=1))
print(json.dumps({"tuning": {"meta": tun["meta"], "meta_ce": tun["meta_ce"]},
                  "mrr": {s: round(summary["all"][s]["rr"]["mean"], 3) for s in summary["all"]},
                  "succ5": {s: round(summary["all"][s]["success@5"]["mean"], 3) for s in summary["all"]},
                  "wrong_cond": {s: round(summary["all"][s]["wrong_condition@5"]["mean"], 3) for s in summary["all"]},
                  "classifier": cls, "gate": {"tau": tau, "auroc": au, "aurc": cr["aurc"]},
                  "safety": Counter(f"{r['expected']}->{r['got']}" for r in safety_rows)}, indent=1))
