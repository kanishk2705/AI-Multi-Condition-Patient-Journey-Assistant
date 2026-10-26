"""CI check: do the live ONNX models reproduce the evaluated models? On the DEV split only.
Compares query/passage embeddings, re-ranker scores and the selected system's dev retrieval
(MRR, success@5), and the live NLI checker's verdicts against the evaluated checker on the dev
answers. Writes results/live_model_check.json."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pjassist import evaluate as ev
from pjassist.grounding import check_answer
from pjassist.models import QUERY_PREFIX, passage_text
from pjassist.onnx_models import OnnxCrossEncoder, OnnxEmbedder, fetch
from pjassist.retrieval import Params

d = Path(tempfile.mkdtemp()); fetch(d)
emb, ce, nli = OnnxEmbedder(d / "emb"), OnnxCrossEncoder(d / "ce"), OnnxCrossEncoder(d / "nli")
P = ev.load_passages(); Q = json.loads((ROOT / "data/questions/dev.json").read_text())
PE = np.load(ROOT / "data/cache/passages_emb.npy"); QE = np.load(ROOT / "data/cache/dev_qemb.npy"); CE = np.load(ROOT / "data/cache/dev_ce.npy")
pe = emb.encode([passage_text(p) for p in P]); qe = emb.encode([QUERY_PREFIX + q["question"] for q in Q])
cos_p = (pe * PE).sum(1); cos_q = (qe * QE).sum(1)
idx = list(range(0, len(P), 7))
ce_live = np.array([ce.predict([(q["question"], passage_text(P[j])) for j in idx]) for q in Q])
r = np.corrcoef(ce_live.ravel(), CE[:, idx].ravel())[0, 1]
sel = json.loads((ROOT / "results/dev/selection.json").read_text())
params = {"meta_ce": Params(**sel["params"]["meta_ce"])}
ce_full = np.array([ce.predict([(q["question"], passage_text(p)) for p in P]) for q in Q])
orig = ev.evaluate(Q, P, QE, CE, ("meta_ce",), params)
# live: ONNX query embeddings, the shipped passage embeddings, ONNX re-ranker
live = ev.evaluate(Q, P, qe, ce_full, ("meta_ce",), params)
s_o, s_l = ev.summarize(orig, Q)["meta_ce"], ev.summarize(live, Q)["meta_ce"]
same_top4 = np.mean([orig["meta_ce"]["rankings"][k][:4] == live["meta_ce"]["rankings"][k][:4] for k in orig["meta_ce"]["rankings"]])
A = json.loads((ROOT / "results/dev/answers.json").read_text())
ei = nli.entail_index()
f = lambda pairs: [float(x[ei]) for x in np.atleast_2d(nli.predict(pairs, apply_softmax=True))] if pairs else []
agree, n = 0, 0
for qid, a in A.items():
    g = a.get("llm", {}).get("grounding")
    if not g:
        continue
    labels = {f"P{i}": next(p["text"] for p in P if p["id"] == pid) for i, pid in enumerate(a["shown"], 1)}
    g2 = check_answer(a["llm"]["text"], labels, f)
    for s1, s2 in zip(g["sentences"], g2["sentences"]):
        n += 1; agree += s1["supported"] == s2["supported"]
out = {"split": "dev", "embedding_cosine_passages_min": float(cos_p.min()), "embedding_cosine_queries_min": float(cos_q.min()),
       "reranker_score_correlation": float(r), "selected_system_dev": {"evaluated_mrr": s_o["rr"]["mean"], "live_mrr": s_l["rr"]["mean"],
       "evaluated_success5": s_o["success@5"]["mean"], "live_success5": s_l["success@5"]["mean"], "same_top4_share": float(same_top4)},
       "checker_agreement_live_vs_evaluated": {"sentences": n, "agree": agree, "rate": agree / n if n else None}}
(ROOT / "results/live_model_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
