"""Build the labelled question set and split it into development and held-out test.

Inputs: scripts/questions_draft.py (questions, written by Claude) and the gold-label table
(data/questions/gold_labels.json, passage ids chosen by reading candidate passages found with
keyword searches; no retrieval system was run). Split: stratified by question type, 40% dev /
60% test, seed fixed in config. The test file's SHA-256 is recorded in PROTOCOL.md.
"""
import hashlib, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from questions_draft import Q
from pjassist.config import SEED

P = [json.loads(l) for l in open(ROOT / "data/corpus/passages.jsonl")]
ids = {p["id"] for p in P}
gold = json.loads((ROOT / "data/questions/gold_labels.json").read_text())

rows = []
for qid, typ, q, conds, stage, _ in Q:
    r = {"qid": qid, "type": typ, "question": q, "conditions": conds, "stage": stage,
         "answerable": typ in ("answerable", "cross", "trap"),
         "expected_route": typ if typ in ("emergency", "dosing", "diagnosis") else "ok"}
    if typ in ("answerable", "trap"):
        r["gold"] = gold["single"][qid]
    if typ == "cross":
        r["gold_by_condition"] = gold["cross"][qid]
        r["gold"] = sorted({g for v in r["gold_by_condition"].values() for g in v})
    if qid in gold.get("partial", {}):
        r["partial_support_note"] = gold["partial"][qid]
    if qid in gold.get("trap_note", {}):
        r["trap_note"] = gold["trap_note"][qid]
    for g in r.get("gold", []):
        assert g in ids, (qid, g)
    rows.append(r)

rng = random.Random(SEED)
dev, test = [], []
for typ in ("answerable", "cross", "trap", "unanswerable", "emergency", "dosing", "diagnosis"):
    grp = [r for r in rows if r["type"] == typ]
    if typ == "answerable":  # stratify answerable further by condition and stage
        strata = {}
        for r in grp:
            strata.setdefault((r["conditions"][0], r["stage"]), []).append(r)
        groups = list(strata.values())
    else:
        groups = [grp]
    for g in groups:
        g = sorted(g, key=lambda r: r["qid"]); rng.shuffle(g)
        k = round(0.4 * len(g))
        dev += g[:k]; test += g[k:]
dev.sort(key=lambda r: r["qid"]); test.sort(key=lambda r: r["qid"])
out = ROOT / "data/questions"
(out / "dev.json").write_text(json.dumps(dev, indent=1, ensure_ascii=False))
(out / "test.json").write_text(json.dumps(test, indent=1, ensure_ascii=False))
from collections import Counter
print("dev", len(dev), Counter(r["type"] for r in dev))
print("test", len(test), Counter(r["type"] for r in test))
print("test sha256", hashlib.sha256((out / "test.json").read_bytes()).hexdigest())
