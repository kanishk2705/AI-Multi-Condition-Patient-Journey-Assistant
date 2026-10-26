"""Backend API for the AI Multi-Condition Patient Journey Assistant. Runs on Render (render/Dockerfile,
ONNX models) or as a Docker Space (space/Dockerfile, PyTorch models).

Endpoints: GET /health, POST /ask, POST /compare. Keys come only from Space secrets
(GROQ_API_KEY). No tracking: nothing about visitors is stored; rate limiting keeps per-IP
counters in memory only and forgets them after a day.
"""
import os, threading, time
from collections import defaultdict, deque
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT / "src"))
from pjassist.pipeline import Pipeline  # noqa: E402

MAX_LEN = 300
PER_MIN, PER_DAY = 6, 40            # per visitor (IP) limits
LLM_DAILY_CAP = int(os.environ.get("LLM_DAILY_CAP", "120"))   # protects the free Groq quota
ALLOWED_ORIGINS = ["https://kanishk2705.github.io"]

app = FastAPI(title="Patient Journey Assistant API", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"], max_age=3600)

_pipe, _lock = None, threading.Lock()
_hits = defaultdict(deque)
_llm_calls = deque()


def pipe():
    global _pipe
    with _lock:
        if _pipe is None:
            be = os.environ.get("PJ_BACKEND", "torch")
            _pipe = Pipeline(ROOT, Path("/tmp/llm_cache.json"), backend=be,
                             models_dir=Path(os.environ["PJ_MODELS"]) if be == "onnx" else None)
    return _pipe


def limit(req: Request, cost: int = 1):
    ip = (req.headers.get("x-forwarded-for") or req.client.host or "?").split(",")[0].strip()
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 86400:
        q.popleft()
    if sum(1 for t in q if now - t < 60) + cost > PER_MIN or len(q) + cost > PER_DAY:
        raise HTTPException(429, "Too many questions from this connection. Please wait a minute and try again.")
    for _ in range(cost):
        q.append(now)


def llm_budget(n: int) -> bool:
    now = time.time()
    while _llm_calls and now - _llm_calls[0] > 86400:
        _llm_calls.popleft()
    if len(_llm_calls) + n > LLM_DAILY_CAP:
        return False
    for _ in range(n):
        _llm_calls.append(now)
    return True


class Ask(BaseModel):
    question: str = Field(..., min_length=3, max_length=MAX_LEN)
    conditions: list[str] | None = None
    stage: str | None = None


def clean(a: Ask):
    conds = [c for c in (a.conditions or []) if c in ("asthma", "diabetes")] or None
    stage = a.stage if a.stage in ("understand", "manage", "warning") else None
    return a.question.strip(), conds, stage


@app.get("/health")
def health():
    return {"ok": True, "loaded": _pipe is not None, "llm_calls_last_24h": len(_llm_calls), "llm_daily_cap": LLM_DAILY_CAP}


@app.post("/ask")
def ask(a: Ask, req: Request):
    limit(req)
    q, conds, stage = clean(a)
    p = pipe()
    use_llm = llm_budget(1)
    out = p.answer(q, conds, stage, use_llm=use_llm)
    out["llm_available"] = use_llm
    return out


@app.post("/compare")
def compare(a: Ask, req: Request):
    limit(req, cost=2)
    q, conds, stage = clean(a)
    p = pipe()
    use_llm = llm_budget(2)
    plain = p.answer(q, None, None, system="dense", use_llm=use_llm)
    full = p.answer(q, conds, stage, use_llm=use_llm)
    return {"plain": plain, "full": full, "llm_available": use_llm}


@app.on_event("startup")
def warm():
    threading.Thread(target=pipe, daemon=True).start()
