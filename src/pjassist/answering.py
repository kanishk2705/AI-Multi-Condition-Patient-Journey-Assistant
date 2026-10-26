"""Generate LLM and extractive answers for one split and check their grounding (runs on CI).

Shared by scripts/answer_dev.py and scripts/run_final.py so dev and test go through identical code.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .bm25 import BM25
from .generate import GroqClient, extractive_answer, llm_answer
from .grounding import check_answer
from .models import LLM_MODEL, LLM_SETTINGS, NLI_ENTAIL_IDX, NLI_MODEL


def make_nli():
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(NLI_MODEL)

    def nli(pairs):
        if not pairs:
            return []
        probs = m.predict(pairs, apply_softmax=True, batch_size=32, show_progress_bar=False)
        return [float(p[NLI_ENTAIL_IDX]) for p in np.atleast_2d(probs)]
    return nli


def check_prompt(selection: dict):
    import hashlib
    from .generate import SYSTEM_PROMPT
    h = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    if selection.get("llm_prompt_sha256") and selection["llm_prompt_sha256"] != h:
        raise SystemExit("REFUSED: generator prompt differs from the frozen selection")


def answer_split(questions, passages, rankings_ids: dict, gate: dict, tau: float, cache_path: Path,
                 k: int = 4, use_llm: bool = True, log=print) -> dict:
    """rankings_ids: qid -> ranked passage ids (selected system). gate: qid -> gate score.
    Only non-safety questions are answered; questions below tau are abstained by the gate and
    never sent to the LLM (PROTOCOL 5.4)."""
    by_id = {p["id"]: p for p in passages}
    bm = BM25([p["text"] for p in passages])
    nli = make_nli()
    client = GroqClient(LLM_MODEL, cache_path, temperature=LLM_SETTINGS["temperature"],
                        max_tokens=LLM_SETTINGS["max_tokens"], reasoning_effort=LLM_SETTINGS["reasoning_effort"],
                        seed=LLM_SETTINGS["seed"]) if use_llm else None
    out = {}
    try:
        for q in questions:
            qid = q["qid"]
            if qid not in rankings_ids:
                continue
            shown = [by_id[i] for i in rankings_ids[qid][:k]]
            gated = gate[qid] < tau
            row = {"gate_score": gate[qid], "gate_abstained": gated, "shown": [p["id"] for p in shown]}
            ext = extractive_answer(q["question"], shown, bm.idf)
            row["extractive"] = {"text": "" if gated else ext["text"], "abstained": gated or ext["abstained"]}
            if not row["extractive"]["abstained"]:
                row["extractive"]["grounding"] = check_answer(ext["text"], ext["labels"], nli)
            if use_llm and not gated:
                a = llm_answer(client, q["question"], shown)
                row["llm"] = {"text": a["text"], "raw": a["raw"], "abstained": a["abstained"], "usage": a["usage"],
                              "cached": a["cached"], "empty": not a["raw"]}
                if not a["abstained"] and a["text"]:
                    row["llm"]["grounding"] = check_answer(a["text"], a["labels"], nli)
                log(f"{qid} llm {'ABSTAIN' if a['abstained'] else 'answer'} tokens={a['usage'].get('total_tokens')}")
            elif use_llm:
                row["llm"] = {"text": "", "abstained": True, "gate": True}
            out[qid] = row
    finally:
        if client:
            client.save()
    return out
