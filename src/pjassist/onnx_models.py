"""Lightweight ONNX versions of the models for the live backend (Render free tier, 512 MB RAM,
no PyTorch). Int8-quantised ONNX exports of the same embedding and re-ranker models as the
evaluation (Xenova/bge-small-en-v1.5, Xenova/ms-marco-MiniLM-L-6-v2); the live grounding checker uses the
smaller cross-encoder/nli-deberta-v3-xsmall (Xenova export, int8) instead of the evaluated
nli-deberta-v3-base, because the base model does not fit in memory. scripts/check_onnx.py
measures how closely the live models reproduce the evaluated ones."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ONNX_REPOS = {
    "emb": ("Xenova/bge-small-en-v1.5", "onnx/model_quantized.onnx"),
    "ce": ("Xenova/ms-marco-MiniLM-L-6-v2", "onnx/model_quantized.onnx"),
    "nli": ("Xenova/nli-deberta-v3-xsmall", "onnx/model_quantized.onnx"),
}
LIVE_NLI_NAME = "cross-encoder/nli-deberta-v3-xsmall (int8 ONNX)"


def fetch(dest: Path):
    from huggingface_hub import hf_hub_download
    dest.mkdir(parents=True, exist_ok=True)
    for key, (repo, fname) in ONNX_REPOS.items():
        d = dest / key
        d.mkdir(exist_ok=True)
        for f in (fname, "tokenizer.json", "config.json"):
            p = hf_hub_download(repo, f)
            (d / Path(f).name).write_bytes(Path(p).read_bytes())


class _Base:
    def __init__(self, d: Path, max_len: int = 512):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False   # keep peak memory low (512 MB host)
        onnx = next(d.glob("*.onnx"))
        self.sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        self.config = json.loads((d / "config.json").read_text())

    def _run(self, encs):
        feed = {"input_ids": np.array([e.ids for e in encs], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
        return self.sess.run(None, feed)[0], feed["attention_mask"]


class OnnxEmbedder(_Base):
    """bge-small: CLS pooling, L2-normalised (as in its sentence-transformers config)."""
    def encode(self, texts, normalize_embeddings=True, batch_size=16, **_):
        out = []
        for i in range(0, len(texts), batch_size):
            h, _m = self._run(self.tok.encode_batch(list(texts[i:i + batch_size])))
            v = h[:, 0, :]
            if normalize_embeddings:
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
            out.append(v)
        return np.vstack(out).astype(np.float32)


class OnnxCrossEncoder(_Base):
    def predict(self, pairs, apply_softmax=False, batch_size=16, **_):
        outs = []
        for i in range(0, len(pairs), batch_size):
            chunk = [(a, b) for a, b in pairs[i:i + batch_size]]
            logits, _m = self._run(self.tok.encode_batch(chunk))
            if apply_softmax:
                z = logits - logits.max(axis=1, keepdims=True)
                logits = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
            outs.append(logits)
        out = np.vstack(outs)
        return out[:, 0] if out.shape[1] == 1 else out

    def entail_index(self) -> int:
        lab = {int(k): v.lower() for k, v in self.config.get("id2label", {}).items()}
        return next(k for k, v in lab.items() if v.startswith("entail"))
