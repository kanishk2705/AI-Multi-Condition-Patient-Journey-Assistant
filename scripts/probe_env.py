"""CI environment probe: model downloads and Groq models/limits. Never prints secrets."""
import json, os, time
from pathlib import Path
import requests
out = {}
key = os.environ.get("GROQ_API_KEY", "")
out["groq_key_present"] = bool(key)
h = {"Authorization": f"Bearer {key}"}
r = requests.get("https://api.groq.com/openai/v1/models", headers=h, timeout=30)
out["groq_models_status"] = r.status_code
if r.ok:
    out["groq_models"] = sorted(m["id"] for m in r.json()["data"])
for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "meta-llama/llama-4-scout-17b-16e-instruct"]:
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=h, timeout=60,
                      json={"model": model, "messages": [{"role": "user", "content": "Reply with the word OK."}],
                            "max_tokens": 5, "temperature": 0})
    out[f"probe_{model}"] = {"status": r.status_code,
        "limits": {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")},
        "reply": (r.json()["choices"][0]["message"]["content"] if r.ok else r.text[:300])}
    time.sleep(3)
out["hf_token_present"] = bool(os.environ.get("HF_TOKEN"))
from huggingface_hub import whoami
try:
    w = whoami(token=os.environ.get("HF_TOKEN"))
    out["hf_user"] = w.get("name"); out["hf_token_role"] = w.get("auth", {}).get("accessToken", {}).get("role")
except Exception as e:
    out["hf_whoami_error"] = repr(e)[:300]
Path("ci/out/probe.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "groq_models"}, indent=1))
