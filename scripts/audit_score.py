"""Hand audit step 2: compare the judgements (results/final/audit_judgments.json, written after
reading audit_sample.json) with the checker's verdicts -> results/final/audit.json."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results/final/audit_sample.json").read_text())
J = {int(k): v for k, v in json.loads((ROOT / "results/final/audit_judgments.json").read_text())["judgments"].items()}
PQ = json.loads((ROOT / "results/final/per_question.json").read_text())
rows = []
for s in S:
    chk = PQ["answers"][s["qid"]][s["answerer"]]["grounding"]["sentences"][s["idx"]]["supported"]
    rows.append({"n": s["n"], "answerer": s["answerer"], "human": J[s["n"]]["supported"], "checker": chk,
                 "note": J[s["n"]].get("note", "")})
n = len(rows); agree = sum(r["human"] == r["checker"] for r in rows)
po = agree / n
ph = sum(r["human"] for r in rows) / n; pcx = sum(r["checker"] for r in rows) / n
pe = ph * pcx + (1 - ph) * (1 - pcx)
kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
out = {"n": n, "agree": agree, "kappa": kappa,
       "checker_too_strict": sum(r["human"] and not r["checker"] for r in rows),
       "checker_too_lenient": sum(r["checker"] and not r["human"] for r in rows),
       "by_answerer": {a: {"n": sum(r["answerer"] == a for r in rows),
                           "agree": sum(r["answerer"] == a and r["human"] == r["checker"] for r in rows),
                           "human_supported": sum(r["answerer"] == a and r["human"] for r in rows)}
                       for a in ("llm", "extractive")},
       "auditor": "Claude (not independent of the author; independent of gpt-oss-120b and the NLI model)",
       "rows": rows}
(ROOT / "results/final/audit.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "rows"})
