"""Summarise the development run with the same report code as the final run -> results/dev/summary.json."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pjassist import evaluate as ev, report, safety
from pjassist.config import DATA, RESULTS
from pjassist.retrieval import Params

Q = json.loads((DATA / "questions/dev.json").read_text())
P = ev.load_passages()
qemb, ce = np.load(DATA / "cache/dev_qemb.npy"), np.load(DATA / "cache/dev_ce.npy")
sel = json.loads((RESULTS / "dev/selection.json").read_text())
params = {"meta": Params(**sel["params"]["meta"]), "meta_ce": Params(**sel["params"]["meta_ce"])}
res = ev.evaluate(Q, P, qemb, ce, ev.SYSTEMS, params)
res_or = ev.evaluate(Q, P, qemb, ce, ("meta", "meta_ce"), params, oracle=True)
ns = [q for q in Q if q["expected_route"] == "ok"]
rk = {q["qid"]: res[sel["retrieval_system"]]["rankings"][q["qid"]] for q in ns}
scores = ev.gate_scores(Q, ce, rk)
out = {"note": "Development split (59 questions). Used for tuning and bug fixing; not evidence.",
       "safety": report.safety_report(Q, {q["qid"]: safety.check(q["question"]).kind for q in Q}),
       "retrieval": report.retrieval_report(Q, res, res_or),
       "classifier": report.classifier_report(Q, res["meta"]["inferred_meta"]),
       "gate": report.gate_report(ns, scores, ev.ids_rankings(rk, P), sel["gate_tau"])}
ap = RESULTS / "dev/answers.json"
if ap.exists():
    out["answers"] = report.answers_report(Q, json.loads(ap.read_text()))
(RESULTS / "dev/summary.json").write_text(json.dumps(out, indent=1))
a = out["retrieval"]["all"]
print({s: round(a[s]["rr"]["mean"], 3) for s in a})
print({s: round(v["rr"]["mean"], 3) for s, v in out["retrieval"]["oracle_all"].items()})
if "answers" in out:
    for k in ("llm", "extractive"):
        s = out["answers"][k]
        print(k, s["n_answers"], round(s["sentence_supported_rate"]["value"], 3), round(s["fully_grounded_rate"]["mean"], 3), s["sentence_reasons"])
