"""Hand audit of the grounding checker (PROTOCOL 5.5), step 1: draw the sample BLIND.

Writes results/final/audit_sample.json with each sampled sentence and the full text of the
passage(s) it cites, but NOT the checker's verdict, so the human-style judgement is made first.
"""
import json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PQ = json.loads((ROOT / "results/final/per_question.json").read_text())
P = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/corpus/passages.jsonl")}
rng = random.Random(20261024)
pools = {"llm": [], "extractive": []}
for qid, a in sorted(PQ["answers"].items()):
    for key in pools:
        g = a.get(key, {}).get("grounding")
        if not g:
            continue
        for i, s in enumerate(g["sentences"]):
            if s["cited"]:
                pools[key].append((qid, key, i, s["sentence"], s["cited"], a["shown"]))
sample = []
for key in ("llm", "extractive"):
    pool = pools[key]
    sample += rng.sample(pool, min(30, len(pool)))
rng.shuffle(sample)
out = [{"n": k, "qid": qid, "answerer": key, "idx": i, "sentence": s,
        "cited_passages": {lab: P[shown[int(lab[1:]) - 1]]["text"] for lab in cited if 0 < int(lab[1:]) <= len(shown)}}
       for k, (qid, key, i, s, cited, shown) in enumerate(sample)]
(ROOT / "results/final/audit_sample.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(len(out), "sentences sampled")
