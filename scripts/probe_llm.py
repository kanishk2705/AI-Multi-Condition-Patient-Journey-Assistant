"""Compare available Groq chat models on one grounded-answer prompt and record rate limits."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "src")
import requests
from pjassist.generate import SYSTEM_PROMPT
P = [{"source": "MedlinePlus", "title": "Steroids", "section": None, "text": "Corticosteroids reduce inflammation and affect the immune system. You may need to take corticosteroids to treat asthma. Steroids are strong medicines, and they can have side effects, including weakened bones and cataracts."},
     {"source": "MedlinePlus", "title": "Hyperglycemia", "section": None, "text": "Hyperglycemia means high blood glucose. It can also happen if you are stressed or sick. Less commonly, people who don't have diabetes can also get hyperglycemia. It can be caused by conditions that can affect insulin or glucose levels in your blood, including certain medicines."}]
from pjassist.generate import user_prompt
QS = ["What side effects can steroid medicines have?", "Do steroid tablets for asthma raise blood sugar in people with diabetes?", "What is the best diet for weight loss?"]
h = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
out = {}
for model in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]:
    res = []
    for q in QS:
        u, _ = user_prompt(q, P)
        body = {"model": model, "temperature": 0, "max_tokens": 1200, "seed": 7,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": u}]}
        if model.startswith("openai/"):
            body["reasoning_effort"] = "low"
        if model.startswith("qwen/"):
            body["reasoning_effort"] = "none"
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body, headers=h, timeout=120)
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        res.append({"q": q, "status": r.status_code, "text": (j.get("choices") or [{}])[0].get("message", {}).get("content") if r.ok else r.text[:400],
                    "usage": j.get("usage"), "limits": {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")}})
        time.sleep(4)
    out[model] = res
Path("ci/out/probe_llm.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:6000])
