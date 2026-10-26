"""Export the browser fallback index: docs/app/data/index.json (passages + tokeniser stop words +
classifier lexicon, so the JavaScript port cannot drift from Python)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pjassist.classifier import CONDITION_CUES, STAGE_CUES
from pjassist.text import STOP
from pjassist import safety

P = [json.loads(l) for l in open(ROOT / "data/corpus/passages.jsonl")]
keep = ("id", "source", "title", "section", "url", "text", "conditions", "stage")
out = {"stopwords": sorted(STOP), "condition_cues": CONDITION_CUES, "stage_cues": STAGE_CUES,
       "safety": {"emergency": safety.EMERGENCY_PATTERNS, "dosing": safety.DOSING_PATTERNS,
                  "diagnosis": safety.DIAGNOSIS_PATTERNS, "neutralise": safety._NEUTRALISE,
                  "messages": {"emergency": safety.EMERGENCY_MESSAGE, "dosing": safety.DOSING_MESSAGE,
                               "diagnosis": safety.DIAGNOSIS_MESSAGE}},
       "retrieved": "2026-10-24", "passages": [{k: p[k] for k in keep} for p in P]}
d = ROOT / "docs/app/data"; d.mkdir(parents=True, exist_ok=True)
(d / "index.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
print((d / "index.json").stat().st_size, "bytes")
