"""One-time held-out test run (PROTOCOL.md section 4.4 and 9 step 4). Runs on CI.

Refuses to run if: the access ledger already exists; the test file hash differs from the one in
PROTOCOL.md; or results/dev/selection.json differs from the copy on branch frozen/selection.
Writes the ledger BEFORE reading any test question, then: test embeddings and cross-encoder
scores, safety routing, all retrieval systems (inferred and oracle metadata), abstention gate,
LLM and extractive answers with grounding checks, statistics -> results/final/results.json.
"""
import hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from pjassist import evaluate as ev, report, safety
from pjassist.answering import answer_split, check_prompt
from pjassist.config import DATA, RESULTS
from pjassist.retrieval import Params

TEST_SHA = "d8ef583865a2ec489cb983c34c9f2535425cbe8210f8f99d36131c532517db76"
OUT = RESULTS / "final"
LEDGER = OUT / "test_access.json"
SEL = RESULTS / "dev/selection.json"
TEST = DATA / "questions/test.json"


def sh(*a):
    return subprocess.run(a, cwd=ROOT, capture_output=True, text=True, check=True).stdout


def guard():
    if LEDGER.exists():
        sys.exit("REFUSED: test access ledger already exists; the test split has been read.")
    h = hashlib.sha256(TEST.read_bytes()).hexdigest()
    if h != TEST_SHA:
        sys.exit(f"REFUSED: test file hash {h} differs from PROTOCOL.md")
    sh("git", "fetch", "--depth=1", "origin", "+refs/heads/frozen/selection:refs/remotes/origin/frozen/selection")
    frozen = sh("git", "show", "origin/frozen/selection:results/dev/selection.json")
    if json.loads(frozen) != json.loads(SEL.read_text()):
        sys.exit("REFUSED: results/dev/selection.json differs from branch frozen/selection")
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = {"read_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_commit": sh("git", "rev-parse", "HEAD").strip(),
              "test_sha256": h, "selection_sha256": hashlib.sha256(SEL.read_bytes()).hexdigest(),
              "protocol_sha256": hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest()}
    LEDGER.write_text(json.dumps(ledger, indent=1))
    return ledger


def main(rehearse=False):
    """rehearse=True runs the identical pipeline on the DEV split (no ledger, cached LLM answers)
    so that a bug cannot crash the one-time test run after the ledger has been written."""
    global OUT
    check_prompt(json.loads(SEL.read_text()))
    if rehearse:
        OUT = RESULTS / "rehearsal"; OUT.mkdir(parents=True, exist_ok=True)
        ledger = {"rehearsal": True}
    else:
        ledger = guard()
    t0 = time.time()
    sel = json.loads(SEL.read_text())
    Q = json.loads(((DATA / "questions/dev.json") if rehearse else TEST).read_text())
    name = "rehearsal" if rehearse else "test"
    P = ev.load_passages()
    from embed import split
    qemb, ce = split(name, Q)

    routes = {q["qid"]: safety.check(q["question"]).kind for q in Q}
    params = {"meta": Params(**sel["params"]["meta"]), "meta_ce": Params(**sel["params"]["meta_ce"])}
    res = ev.evaluate(Q, P, qemb, ce, ev.SYSTEMS, params)
    res_or = ev.evaluate(Q, P, qemb, ce, ("meta", "meta_ce"), params, oracle=True)

    system = sel["retrieval_system"]
    non_safety = [q for q in Q if q["expected_route"] == "ok"]
    rk_idx = {q["qid"]: res[system]["rankings"][q["qid"]] for q in non_safety}
    scores = ev.gate_scores(Q, ce, rk_idx)
    rk_ids = ev.ids_rankings(rk_idx, P)
    # The pipeline answers only what the safety layer passes; evaluation of answers covers every
    # non-safety question so that a safety false alarm cannot hide an answer failure.
    answers = answer_split(non_safety, P, rk_ids, scores, sel["gate_tau"], RESULTS / ("llm_cache/dev.json" if rehearse else "llm_cache/test.json"))

    results = {
        "ledger": ledger, "selection": sel, "n_questions": len(Q),
        "safety": report.safety_report(Q, routes),
        "retrieval": report.retrieval_report(Q, res, res_or),
        "classifier": report.classifier_report(Q, res["meta"]["inferred_meta"]),
        "gate": report.gate_report(non_safety, scores, rk_ids, sel["gate_tau"]),
        "answers": report.answers_report(Q, answers),
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    (OUT / "per_question.json").write_text(json.dumps({
        "routes": routes, "rankings": {s: ev.ids_rankings(r["rankings"], P) for s, r in res.items()},
        "oracle_rankings": {s: ev.ids_rankings(r["rankings"], P) for s, r in res_or.items()},
        "retrieval_metrics": {s: r["per_question"] for s, r in res.items()},
        "inferred_meta": res["meta"]["inferred_meta"], "gate_scores": scores, "answers": answers}, indent=1, ensure_ascii=False))
    print("done", results["runtime_s"], "s")


if __name__ == "__main__":
    main(rehearse="--rehearse" in sys.argv)
