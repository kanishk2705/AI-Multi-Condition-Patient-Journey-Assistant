"""Compute model scores on CI (the sandbox cannot download models).

  python scripts/embed.py passages          -> data/cache/passages_emb.npy  (bge-small, L2-normalised)
  python scripts/embed.py split dev         -> data/cache/dev_qemb.npy, data/cache/dev_ce.npy
The test split is only embedded inside scripts/run_final.py, behind the access ledger.
Cross-encoder scores are computed for every (question, passage) pair, so any candidate list
chosen later can be re-ranked offline without re-running the model.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pjassist.models import EMB_MODEL, CE_MODEL, QUERY_PREFIX, passage_text

CACHE = ROOT / "data" / "cache"


def passages():
    from sentence_transformers import SentenceTransformer
    P = [json.loads(l) for l in open(ROOT / "data/corpus/passages.jsonl")]
    m = SentenceTransformer(EMB_MODEL)
    E = m.encode([passage_text(p) for p in P], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "passages_emb.npy", E.astype(np.float32))
    print("passages", E.shape)


def split(name, questions=None):
    from sentence_transformers import SentenceTransformer, CrossEncoder
    if questions is None:
        assert name != "test", "test questions are embedded only by run_final.py"
        questions = json.loads((ROOT / f"data/questions/{name}.json").read_text())
    P = [json.loads(l) for l in open(ROOT / "data/corpus/passages.jsonl")]
    m = SentenceTransformer(EMB_MODEL)
    Q = m.encode([QUERY_PREFIX + q["question"] for q in questions], normalize_embeddings=True, show_progress_bar=False)
    ce = CrossEncoder(CE_MODEL)
    pairs = [(q["question"], passage_text(p)) for q in questions for p in P]
    S = np.asarray(ce.predict(pairs, batch_size=64, show_progress_bar=False), dtype=np.float32).reshape(len(questions), len(P))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{name}_qemb.npy", Q.astype(np.float32))
    np.save(CACHE / f"{name}_ce.npy", S)
    print(name, Q.shape, S.shape)
    return Q, S


if __name__ == "__main__":
    if sys.argv[1] == "passages":
        passages()
    else:
        split(sys.argv[2])
