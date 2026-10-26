"""The live question-answering pipeline used by the website backend.

It runs exactly the evaluated code: safety layer -> classifier (or the visitor's choices) ->
retrieval (the system selected on dev) -> abstention gate -> LLM answer -> grounding check.
Nothing here was used to produce the evaluation numbers; it reuses the same modules.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from . import safety
from .classifier import classify
from .generate import GroqClient, llm_answer
from .grounding import check_answer
from .models import CE_MODEL, EMB_MODEL, LLM_MODEL, LLM_SETTINGS, NLI_ENTAIL_IDX, NLI_MODEL, QUERY_PREFIX, passage_text
from .retrieval import Index, Params

NOT_ADVICE = "Educational information from public health sources only. Not medical advice."


class Pipeline:
    def __init__(self, root: Path, cache_path: Path, backend: str = "torch", models_dir: Path | None = None):
        self.passages = [json.loads(l) for l in open(root / "data/corpus/passages.jsonl")]
        self.index = Index(self.passages, np.load(root / "data/cache/passages_emb.npy"))
        sel = json.loads((root / "results/dev/selection.json").read_text())
        self.system = sel["retrieval_system"]
        self.params = Params(**sel["params"][self.system])
        self.tau = sel["gate_tau"]
        self.k = sel["k_shown"]
        self.max_windows = 4 if backend == "onnx" else None
        if backend == "onnx":   # live site on a 512 MB host: no PyTorch (see onnx_models.py)
            from .onnx_models import LIVE_NLI_NAME, OnnxCrossEncoder, OnnxEmbedder
            self.emb = OnnxEmbedder(models_dir / "emb")
            self.ce = OnnxCrossEncoder(models_dir / "ce")
            self.nli_model = OnnxCrossEncoder(models_dir / "nli")
            self.entail_idx, self.nli_name = self.nli_model.entail_index(), LIVE_NLI_NAME
        else:
            from sentence_transformers import CrossEncoder, SentenceTransformer
            self.emb = SentenceTransformer(EMB_MODEL)
            self.ce = CrossEncoder(CE_MODEL)
            self.nli_model = CrossEncoder(NLI_MODEL)
            self.entail_idx, self.nli_name = NLI_ENTAIL_IDX, NLI_MODEL
        self.client = GroqClient(LLM_MODEL, cache_path, temperature=LLM_SETTINGS["temperature"],
                                 max_tokens=LLM_SETTINGS["max_tokens"],
                                 reasoning_effort=LLM_SETTINGS["reasoning_effort"], seed=LLM_SETTINGS["seed"])

    def nli(self, pairs):
        if not pairs:
            return []
        probs = self.nli_model.predict(pairs, apply_softmax=True, batch_size=32, show_progress_bar=False)
        return [float(p[self.entail_idx]) for p in np.atleast_2d(probs)]

    def _ce(self, q):
        cache = {}

        def fn(_q, idx):
            need = [i for i in idx if i not in cache]
            if need:
                sc = self.ce.predict([(q, passage_text(self.passages[i])) for i in need], show_progress_bar=False)
                cache.update({i: float(s) for i, s in zip(need, sc)})
            return [cache[i] for i in idx]
        return fn

    def retrieve(self, question: str, system: str, meta: dict) -> tuple[list[int], dict]:
        qvec = self.emb.encode([QUERY_PREFIX + question], normalize_embeddings=True)[0]
        ce_fn = self._ce(question)
        r = self.index.search(system, question, qvec, self.params, meta=meta, ce_fn=ce_fn)
        top = r["ranking"][: self.k]
        gate = max(ce_fn(question, top))
        return top, {"gate_score": float(gate)}

    def passage_view(self, i: int, label: str) -> dict:
        p = self.passages[i]
        return {"label": label, "id": p["id"], "source": p["source"], "title": p["title"], "section": p["section"],
                "url": p["url"], "text": p["text"], "conditions": p["conditions"], "stage": p["stage"]}

    def answer(self, question: str, conditions=None, stage=None, system=None, use_llm=True) -> dict:
        system = system or self.system
        s = safety.check(question)
        if s.kind != "ok":
            return {"route": s.kind, "message": s.message, "notice": NOT_ADVICE}
        inferred = classify(question)
        meta = {"conditions": tuple(sorted(conditions)) if conditions else inferred["conditions"],
                "stage": stage or inferred["stage"]}
        top, g = self.retrieve(question, system, meta)
        shown = [self.passage_view(i, f"P{n}") for n, i in enumerate(top, 1)]
        out = {"route": "ok", "system": system, "inferred": {"conditions": list(inferred["conditions"]), "stage": inferred["stage"]},
               "used_meta": {"conditions": list(meta["conditions"]), "stage": meta["stage"]},
               "passages": shown, "gate_score": g["gate_score"], "gate_threshold": self.tau, "notice": NOT_ADVICE,
               "checker_model": self.nli_name}
        if g["gate_score"] < self.tau:
            out.update(abstained=True, abstain_reason="The retrieved passages do not match the question closely enough.")
            return out
        if not use_llm:
            return out
        a = llm_answer(self.client, question, [self.passages[i] for i in top])
        if a["abstained"] or not a["text"]:
            out.update(abstained=True, abstain_reason="The language model judged the passages insufficient.")
            return out
        gr = check_answer(a["text"], a["labels"], self.nli, max_windows=self.max_windows)
        out.update(abstained=False, answer=a["text"], sentences=gr["sentences"], scope_notes=gr["scope_notes"],
                   fully_grounded=gr["fully_grounded"])
        return out
