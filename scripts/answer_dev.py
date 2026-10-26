"""Dev answers (LLM + extractive) and grounding checks. Runs on CI (needs Groq and the NLI model)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pjassist.answering import answer_split
from pjassist.config import DATA, RESULTS

Q = json.loads((DATA / "questions/dev.json").read_text())
P = [json.loads(l) for l in open(DATA / "corpus/passages.jsonl")]
R = json.loads((RESULTS / "dev/retrieval.json").read_text())
G = json.loads((RESULTS / "dev/gate.json").read_text())
sel = json.loads((RESULTS / "dev/selection.json").read_text())
rk = R["rankings"][sel["retrieval_system"]]
non_safety = [q for q in Q if q["expected_route"] == "ok"]
out = answer_split(non_safety, P, {q["qid"]: rk[q["qid"]] for q in non_safety}, G["scores"], sel["gate_tau"],
                   RESULTS / "llm_cache/dev.json")
(RESULTS / "dev/answers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("answered", sum(1 for r in out.values() if "llm" in r and not r["llm"]["abstained"]))
