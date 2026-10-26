"""Turn per-question outputs into the reported statistics (same code for dev and final)."""
from __future__ import annotations
from collections import Counter
import numpy as np
from . import evaluate as ev
from .stats import ci_mean, ci_paired_diff, ci_ratio

ROUTES = ("emergency", "dosing", "diagnosis")


def safety_report(questions, routes: dict) -> dict:
    rows = [{"qid": q["qid"], "expected": q["expected_route"], "got": routes[q["qid"]]} for q in questions]
    out = {"confusion": dict(Counter(f"{r['expected']}->{r['got']}" for r in rows)), "rows": rows}
    for r in ROUTES:
        tp = sum(1 for x in rows if x["expected"] == r and x["got"] == r)
        fn = sum(1 for x in rows if x["expected"] == r and x["got"] != r)
        fp = sum(1 for x in rows if x["expected"] != r and x["got"] == r)
        out[r] = {"tp": tp, "fn": fn, "fp": fp, "recall": tp / (tp + fn) if tp + fn else None,
                  "precision": tp / (tp + fp) if tp + fp else None}
    out["false_alarms_on_non_safety"] = sum(1 for x in rows if x["expected"] == "ok" and x["got"] != "ok")
    out["accuracy"] = float(np.mean([x["expected"] == x["got"] for x in rows]))
    return out


def retrieval_report(questions, res, res_oracle) -> dict:
    rep = {"all": ev.summarize(res, questions)}
    for t in ev.RETRIEVAL_TYPES:
        rep[t] = ev.summarize(res, questions, {t})
    rep["oracle_all"] = ev.summarize(res_oracle, questions)
    rep["oracle_trap"] = ev.summarize(res_oracle, questions, {"trap"})
    pr = lambda a, b, m, sub=None: ev.paired(res, a, b, m, questions, sub)
    rep["hypotheses"] = {
        "H1_meta_vs_dense_mrr": pr("meta", "dense", "rr"),
        "H2_meta_vs_dense_wrong_condition": pr("meta", "dense", "wrong_condition@5"),
        "H3_meta_ce_vs_meta_mrr": pr("meta_ce", "meta", "rr"),
        "H4a_hybrid_vs_dense_mrr": pr("hybrid", "dense", "rr"),
        "H4b_hybrid_vs_bm25_mrr": pr("hybrid", "bm25", "rr"),
        "H5_trap_meta_vs_dense_mrr": pr("meta", "dense", "rr", {"trap"}),
        "H10_cross_meta_vs_dense_both_coverage": pr("meta", "dense", "both_coverage@5", {"cross"}),
    }
    rep["extra_paired"] = {
        "meta_ce_vs_dense_mrr": pr("meta_ce", "dense", "rr"),
        "meta_ce_vs_dense_success5": pr("meta_ce", "dense", "success@5"),
        "meta_vs_dense_wrong_stage": pr("meta", "dense", "wrong_stage@5"),
        "meta_ce_vs_dense_wrong_condition": pr("meta_ce", "dense", "wrong_condition@5"),
        "meta_ce_vs_dense_wrong_stage": pr("meta_ce", "dense", "wrong_stage@5"),
    }
    return rep


def classifier_report(questions, inferred: dict) -> dict:
    qs = [q for q in questions if q["qid"] in inferred and q["type"] in ev.RETRIEVAL_TYPES]
    cond = [set(inferred[q["qid"]]["conditions"]) == set(q["conditions"]) for q in qs]
    stage = [inferred[q["qid"]]["stage"] == q["stage"] for q in qs]
    none = [inferred[q["qid"]]["stage"] is None for q in qs]
    return {"n": len(qs), "condition_exact": ci_mean(cond), "stage_exact": ci_mean(stage),
            "stage_no_cue": float(np.mean(none)) if qs else None}


def gate_report(questions, scores: dict, rankings_ids: dict, tau: float) -> dict:
    ns = [q for q in questions if q["qid"] in scores]
    pos = [scores[q["qid"]] for q in ns if q["answerable"]]
    neg = [scores[q["qid"]] for q in ns if q["type"] == "unanswerable"]
    cr = ev.coverage_risk(ns, scores, rankings_ids)
    return {"auroc": ev.auroc(pos, neg), "aurc": cr["aurc"], "curve": cr["curve"], "tau": tau,
            "answer_rate_answerable": ci_mean([s >= tau for s in pos]),
            "false_answer_rate_unanswerable": ci_mean([s >= tau for s in neg]),
            "n_answerable": len(pos), "n_unanswerable": len(neg)}


def _ground_stats(rows, key):
    got = [(qid, r[key]["grounding"]) for qid, r in rows.items() if key in r and "grounding" in r[key]]
    num = [g["n_supported"] for _, g in got]
    den = [g["n_sentences"] for _, g in got]
    full = [g["fully_grounded"] for _, g in got]
    inval = sum(len(s["invalid_cites"]) for _, g in got for s in g["sentences"])
    reasons = Counter(s["reason"] for _, g in got for s in g["sentences"])
    return got, {"n_answers": len(got), "sentence_supported_rate": ci_ratio(num, den),
                 "fully_grounded_rate": ci_mean(full), "invalid_citations": inval,
                 "sentence_reasons": dict(reasons), "scope_notes": sum(len(g.get("scope_notes", [])) for _, g in got)}


def answers_report(questions, answers: dict) -> dict:
    qmap = {q["qid"]: q for q in questions}
    out = {}
    for key in ("llm", "extractive"):
        got, st = _ground_stats(answers, key)
        ans = [qid for qid, r in answers.items() if key in r and not r[key]["abstained"]]
        st["answered_answerable"] = sum(1 for qid in ans if qmap[qid]["answerable"])
        st["answered_unanswerable"] = sum(1 for qid in ans if qmap[qid]["type"] == "unanswerable")
        st["self_abstained_after_gate"] = sum(1 for r in answers.values() if key in r and r[key]["abstained"]
                                              and not r["gate_abstained"])
        cited_gold = []
        for qid, g in got:
            if qmap[qid]["answerable"]:
                labs = {s for row in g["sentences"] for s in row["cited"]}
                shown = answers[qid]["shown"]
                ids = {shown[int(l[1:]) - 1] for l in labs if 0 < int(l[1:]) <= len(shown)}
                cited_gold.append(bool(ids & set(qmap[qid].get("gold", []))))
        st["gold_cited_rate"] = ci_mean(cited_gold)
        out[key] = st
    # H6: paired per-question supported rate, questions answered by both
    both = [qid for qid, r in answers.items() if "grounding" in r.get("llm", {}) and "grounding" in r.get("extractive", {})]
    rate = lambda g: g["n_supported"] / g["n_sentences"] if g["n_sentences"] else 1.0
    out["H6_llm_minus_extractive_supported_rate"] = ci_paired_diff(
        [rate(answers[q]["llm"]["grounding"]) for q in both], [rate(answers[q]["extractive"]["grounding"]) for q in both])
    usage = [r["llm"].get("usage", {}) for r in answers.values() if "usage" in r.get("llm", {})]
    out["llm_tokens"] = {"calls": len(usage), "prompt": sum(u.get("prompt_tokens", 0) for u in usage),
                         "completion": sum(u.get("completion_tokens", 0) for u in usage)}
    return out
