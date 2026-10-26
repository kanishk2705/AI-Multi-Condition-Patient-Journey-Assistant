"""Retrieval evaluation, tuning, abstention gate and coverage-risk (PROTOCOL.md sections 5-7).

Everything here runs offline from cached model outputs (passage/question embeddings and the
full question x passage cross-encoder matrix), so the same numbers are reproducible without
network access.
"""
from __future__ import annotations
import itertools, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .classifier import classify
from .config import DATA, SEED
from .retrieval import Index, Params
from .stats import ci_mean, ci_paired_diff

SYSTEMS = ("dense", "bm25", "hybrid", "meta", "meta_ce")
RETRIEVAL_TYPES = ("answerable", "trap", "cross")
K = 10


def load_passages():
    return [json.loads(l) for l in open(DATA / "corpus" / "passages.jsonl")]


def oracle_meta(q):
    return {"conditions": tuple(sorted(q["conditions"])), "stage": q["stage"]}


def run_system(index: Index, system: str, q: dict, qvec, ce_row, params: Params, oracle=False) -> dict:
    ce_fn = (lambda _q, idx: [float(ce_row[i]) for i in idx])
    meta = oracle_meta(q) if oracle else None
    return index.search(system, q["question"], qvec, params, meta=meta, ce_fn=ce_fn)


def question_metrics(q: dict, ranking: list[int], passages: list[dict]) -> dict:
    ids = [passages[i]["id"] for i in ranking[:K]]
    gold = set(q.get("gold", []))
    m = {}
    first = next((r for r, pid in enumerate(ids) if pid in gold), None)
    m["rr"] = 0.0 if first is None else 1.0 / (first + 1)
    for k in (1, 3, 5, 10):
        m[f"success@{k}"] = float(any(pid in gold for pid in ids[:k]))
        m[f"recall@{k}"] = len(gold & set(ids[:k])) / len(gold) if gold else 0.0
    top5 = [passages[i] for i in ranking[:5]]
    qc = set(q["conditions"])
    m["wrong_condition@5"] = float(np.mean([bool(p["conditions"]) and not (set(p["conditions"]) & qc) for p in top5]))
    m["wrong_stage@5"] = float(np.mean([p["stage"] != q["stage"] for p in top5]))
    if q["type"] == "cross":
        ids5 = set(ids[:5])
        m["both_coverage@5"] = float(all(set(v) & ids5 for v in q["gold_by_condition"].values()))
    return m


def evaluate(questions, passages, qemb, ce, systems, params_by_system, oracle=False) -> dict:
    """Returns {system: {"per_question": {qid: {...}}, "rankings": {qid: [...]}}} for retrieval questions."""
    index = Index(passages, np.load(DATA / "cache" / "passages_emb.npy"))
    out = {}
    for s in systems:
        per, ranks, metas = {}, {}, {}
        for i, q in enumerate(questions):
            if q["type"] not in RETRIEVAL_TYPES and q["type"] != "unanswerable":
                continue
            r = run_system(index, s, q, qemb[i], ce[i], params_by_system.get(s, Params()), oracle=oracle)
            ranks[q["qid"]] = r["ranking"][:K]
            metas[q["qid"]] = {"conditions": list(r["meta"]["conditions"]), "stage": r["meta"]["stage"]}
            if q["type"] in RETRIEVAL_TYPES:
                per[q["qid"]] = question_metrics(q, r["ranking"], passages)
        out[s] = {"per_question": per, "rankings": ranks, "inferred_meta": metas}
    return out


def summarize(res: dict, questions: list, subset=None) -> dict:
    """Mean and bootstrap CI of every metric, optionally on a subset of question types."""
    types = {q["qid"]: q["type"] for q in questions}
    out = {}
    for s, r in res.items():
        per = {k: v for k, v in r["per_question"].items() if subset is None or types[k] in subset}
        if not per:
            continue
        metrics = sorted({m for v in per.values() for m in v})
        out[s] = {m: ci_mean([v[m] for v in per.values() if m in v]) for m in metrics}
    return out


def paired(res: dict, a: str, b: str, metric: str, questions: list, subset=None) -> dict:
    types = {q["qid"]: q["type"] for q in questions}
    qids = [k for k in res[a]["per_question"] if (subset is None or types[k] in subset)
            and metric in res[a]["per_question"][k]]
    return ci_paired_diff([res[a]["per_question"][k][metric] for k in qids],
                          [res[b]["per_question"][k][metric] for k in qids])


def tune(questions, passages, qemb, ce) -> dict:
    """Grid search on DEV only (PROTOCOL 5.1). Returns chosen params and the full grid table."""
    index = Index(passages, np.load(DATA / "cache" / "passages_emb.npy"))
    retr = [(i, q) for i, q in enumerate(questions) if q["type"] in RETRIEVAL_TYPES]

    def score(system, p):
        rr, s5 = [], []
        for i, q in retr:
            r = run_system(index, system, q, qemb[i], ce[i], p)
            m = question_metrics(q, r["ranking"], passages)
            rr.append(m["rr"]); s5.append(m["success@5"])
        return float(np.mean(rr)), float(np.mean(s5))

    grid = []
    for wc, ws, wb in itertools.product((0.005, 0.01, 0.02), (0.0, 0.0025, 0.005, 0.01), (0.0, 0.005)):
        p = Params(wc=wc, ws=ws, wboth=wb)
        mrr, s5 = score("meta", p)
        grid.append({"system": "meta", "wc": wc, "ws": ws, "wboth": wb, "mrr": mrr, "success@5": s5})
    best = max((g for g in grid if g["system"] == "meta"),
               key=lambda g: (round(g["mrr"], 6), round(g["success@5"], 6), -(g["wc"] + g["ws"] + g["wboth"])))
    pm = Params(wc=best["wc"], ws=best["ws"], wboth=best["wboth"])
    grid_ce = []
    for sc in (0.0, 0.5, 1.0, 2.0):
        p = Params(wc=pm.wc, ws=pm.ws, wboth=pm.wboth, ce_meta_scale=sc)
        mrr, s5 = score("meta_ce", p)
        grid_ce.append({"system": "meta_ce", "ce_meta_scale": sc, "mrr": mrr, "success@5": s5})
    best_ce = max(grid_ce, key=lambda g: (round(g["mrr"], 6), round(g["success@5"], 6), -g["ce_meta_scale"]))
    pce = Params(wc=pm.wc, ws=pm.ws, wboth=pm.wboth, ce_meta_scale=best_ce["ce_meta_scale"])
    return {"meta": asdict(pm), "meta_ce": asdict(pce), "grid": grid + grid_ce}


# ------------------------------------------------------------------ abstention ---------------
def gate_scores(questions, ce, rankings: dict, k: int = 4) -> dict:
    """Gate score = highest cross-encoder logit among the k passages shown to the answerer."""
    pos = {q["qid"]: i for i, q in enumerate(questions)}
    return {qid: float(max(ce[pos[qid]][j] for j in r[:k])) for qid, r in rankings.items()}


def auroc(pos_scores, neg_scores) -> float:
    pos, neg = np.asarray(pos_scores), np.asarray(neg_scores)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def choose_tau(questions, scores: dict) -> dict:
    """Maximise balanced accuracy of answer/abstain on dev (PROTOCOL 5.4)."""
    ans = [scores[q["qid"]] for q in questions if q["qid"] in scores and q["answerable"]]
    un = [scores[q["qid"]] for q in questions if q["qid"] in scores and q["type"] == "unanswerable"]
    cands = sorted(set(ans + un))
    best = None
    for t in cands:
        tpr = np.mean([s >= t for s in ans]); tnr = np.mean([s < t for s in un])
        ba = (tpr + tnr) / 2
        if best is None or ba > best["balanced_accuracy"] + 1e-12:
            best = {"tau": float(t), "balanced_accuracy": float(ba), "answer_rate_answerable": float(tpr),
                    "abstain_rate_unanswerable": float(tnr)}
    # place tau midway between the chosen score and the next lower candidate, so it is not tied to one question
    lower = [c for c in cands if c < best["tau"]]
    if lower:
        best["tau"] = float((best["tau"] + max(lower)) / 2)
    return best


def coverage_risk(questions, scores: dict, rankings: dict, k: int = 4) -> dict:
    """`rankings` must hold passage ids (see ids_rankings). Coverage = share of non-safety questions answered; risk = share of answered that are wrong
    (unanswerable, or answerable with no gold passage among the k shown)."""
    rows = []
    for q in questions:
        if q["qid"] not in scores:
            continue
        ok = bool(q["answerable"] and set(q.get("gold", [])) & set(rankings[q["qid"]][:k]))
        rows.append((scores[q["qid"]], ok))
    rows.sort(key=lambda r: -r[0])
    curve, n = [], len(rows)
    for i in range(1, n + 1):
        ans = rows[:i]
        curve.append({"coverage": i / n, "risk": 1 - np.mean([a[1] for a in ans]), "threshold": ans[-1][0]})
    aurc = float(np.mean([c["risk"] for c in curve]))
    return {"curve": curve, "aurc": aurc, "n": n}


def ids_rankings(rankings: dict, passages) -> dict:
    return {qid: [passages[i]["id"] for i in r] for qid, r in rankings.items()}
