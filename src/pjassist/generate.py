"""Answer generation: (1) LLM via Groq, grounded in retrieved passages; (2) extractive baseline.

Both receive the same retrieved passages, labelled P1..Pk in rank order, and both must
cite a passage label in every sentence. Both may abstain.
"""
from __future__ import annotations
import hashlib, json, math, os, random, time
from collections import Counter
from pathlib import Path
from .grounding import normalise_citations, split_sentences
from .text import tokens

ABSTAIN = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT_V1 = """You answer health-information questions for people living with asthma, type 2 diabetes, or both.
You may use ONLY the numbered source passages provided. Rules:
1. Every sentence must end with the label of the passage that supports it, like [P2]. Use only labels that appear in the passages.
2. Do not add any fact, number, advice or reassurance that is not stated in the cited passage. Paraphrase closely; do not generalise.
3. If the passages do not contain the information needed to answer the question, reply with exactly: INSUFFICIENT_EVIDENCE
4. Never give a medicine dose, never tell the person to start, stop or change a medicine, and never say whether the person has a condition.
5. If the question is about both asthma and diabetes but the passages cover only one, answer only what the passages cover and say in one sentence that the sources shown do not cover the other condition; that sentence needs no citation.
6. Plain language, at most 5 sentences, no headings, no lists."""

# v2 (PROTOCOL deviation D10, chosen on dev): shorter, single-source sentences close to the source wording.
SYSTEM_PROMPT = """You answer health-information questions for people living with asthma, type 2 diabetes, or both.
You may use ONLY the numbered source passages provided. Rules:
1. Write short sentences. Each sentence restates one or two facts from ONE passage, in words close to that passage, and ends with that passage's label in square brackets, like [P2]. Use only labels that appear in the passages.
2. Do not combine facts from different passages in one sentence. Do not list more than three items in one sentence.
3. Do not add any fact, number, cause, explanation, advice or reassurance that the cited passage does not state. Do not generalise beyond the passage.
4. If the passages do not contain the information needed to answer the question, reply with exactly: INSUFFICIENT_EVIDENCE
5. Never give a medicine dose, never tell the person to start, stop or change a medicine, and never say whether the person has a condition.
6. If the question is about both asthma and diabetes but the passages cover only one, answer only what the passages cover and say in one sentence that the sources shown do not cover the other condition; that sentence needs no citation.
7. Plain language, at most 6 sentences, no headings, no lists."""

NO_CITE_OK = ("the sources shown do not cover", "these sources do not cover", "the passages do not cover")


def format_passages(passages: list[dict]) -> tuple[str, dict]:
    labels, parts = {}, []
    for i, p in enumerate(passages, 1):
        lab = f"P{i}"
        labels[lab] = p["text"]
        parts.append(f"[{lab}] ({p['source']}: {p['title']} - {p.get('section') or 'summary'})\n{p['text']}")
    return "\n\n".join(parts), labels


def user_prompt(question: str, passages: list[dict]) -> tuple[str, dict]:
    block, labels = format_passages(passages)
    return f"Source passages:\n\n{block}\n\nQuestion: {question}\n\nAnswer:", labels


class GroqClient:
    """Minimal Groq chat client with an on-disk cache and back-off on 429/5xx.
    The cache key covers model, settings and the full prompt, so cached answers can never be
    reused for a different prompt. The key is read from the environment only."""

    def __init__(self, model: str, cache_path: Path, temperature: float = 0.0, max_tokens: int = 700,
                 min_interval: float = 9.0, reasoning_effort: str | None = "low", seed: int = 7):
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.reasoning_effort, self.seed = reasoning_effort, seed
        self.cache_path = Path(cache_path)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.min_interval, self._last = min_interval, 0.0
        self.calls = 0

    def key(self, system: str, user: str) -> str:
        return hashlib.sha256(json.dumps([self.model, self.temperature, self.max_tokens, self.reasoning_effort,
                                          self.seed, system, user]).encode()).hexdigest()

    def complete(self, system: str, user: str) -> dict:
        k = self.key(system, user)
        if k in self.cache:
            return self.cache[k] | {"cached": True}
        import requests
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens, "seed": self.seed,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
        for attempt in range(8):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body, headers=headers, timeout=90)
            if r.status_code == 200:
                j = r.json()
                out = {"text": (j["choices"][0]["message"].get("content") or "").strip(), "usage": j.get("usage", {}),
                       "finish_reason": j["choices"][0].get("finish_reason"),
                       "model": j.get("model"), "system_fingerprint": j.get("system_fingerprint")}
                self.cache[k] = out
                self.calls += 1
                if self.calls % 10 == 0:
                    self.save()
                return out | {"cached": False}
            if r.status_code in (429, 500, 502, 503):
                ra = r.headers.get("retry-after")
                time.sleep(float(ra) + 1 if ra else min(120, 5 * 2 ** attempt) + random.random())
                continue
            raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")
        raise RuntimeError("Groq: too many retries")

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=0, sort_keys=True))


def llm_answer(client: GroqClient, question: str, passages: list[dict]) -> dict:
    user, labels = user_prompt(question, passages)
    out = client.complete(SYSTEM_PROMPT, user)
    text = normalise_citations(out["text"])
    abstained = ABSTAIN in text and len(text) < len(ABSTAIN) + 40
    return {"text": "" if abstained else text, "abstained": abstained, "labels": labels, "raw": text,
            "usage": out.get("usage", {}), "cached": out.get("cached", False)}


def extractive_answer(question: str, passages: list[dict], idf: dict, max_sents: int = 3,
                      min_score: float = 0.0, top_passages: int = 3) -> dict:
    """No LLM: pick the sentences (from the top passages) with the highest idf-weighted overlap
    with the question, keep original order within passage, cite each one."""
    q = Counter(tokens(question))
    cands = []
    labels = {f"P{i}": p["text"] for i, p in enumerate(passages, 1)}
    for i, p in enumerate(passages[:top_passages], 1):
        for j, s in enumerate(split_sentences(p["text"])):
            st = set(tokens(s))
            score = sum(idf.get(t, 0.0) for t in q if t in st) / math.sqrt(1 + len(st))
            cands.append((score, i, j, s))
    cands.sort(key=lambda c: -c[0])
    chosen = [c for c in cands[:max_sents] if c[0] > 0]
    top = chosen[0][0] if chosen else 0.0
    if not chosen or top < min_score:
        return {"text": "", "abstained": True, "labels": labels, "top_score": top}
    chosen.sort(key=lambda c: (c[1], c[2]))
    text = " ".join(f"{s.rstrip()} [P{i}]" for _, i, _, s in chosen)
    return {"text": text, "abstained": False, "labels": labels, "top_score": top}
