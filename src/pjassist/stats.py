"""Bootstrap intervals over questions (the unit of analysis). Fixed seed, 10,000 resamples,
percentile intervals. Paired comparisons resample the same question indices for both systems."""
from __future__ import annotations
import numpy as np
from .config import SEED

B = 10_000


def ci_mean(x, seed: int = SEED, b: int = B, alpha: float = 0.05) -> dict:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    m = x[idx].mean(axis=1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(m, alpha / 2)),
            "hi": float(np.quantile(m, 1 - alpha / 2)), "n": int(len(x))}


def ci_paired_diff(a, b_, seed: int = SEED, b: int = B, alpha: float = 0.05) -> dict:
    """Mean of (a - b) with a percentile bootstrap interval over the shared questions.
    Also returns the fraction of resamples with a positive difference."""
    a, b_ = np.asarray(a, dtype=float), np.asarray(b_, dtype=float)
    assert a.shape == b_.shape
    d = a - b_
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(b, len(d)))
    m = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.quantile(m, alpha / 2)),
            "hi": float(np.quantile(m, 1 - alpha / 2)), "p_gt0": float((m > 0).mean()), "n": int(len(d)),
            "excludes_zero": bool(np.quantile(m, alpha / 2) > 0 or np.quantile(m, 1 - alpha / 2) < 0)}


def ci_ratio(num, den, seed: int = SEED, b: int = B, alpha: float = 0.05) -> dict:
    """Ratio of sums (e.g. supported sentences / all sentences) resampled by question."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    keep = den > 0
    num, den = num[keep], den[keep]
    if len(num) == 0:
        return {"value": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(b, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return {"value": float(num.sum() / den.sum()), "lo": float(np.quantile(r, alpha / 2)),
            "hi": float(np.quantile(r, 1 - alpha / 2)), "n": int(len(num))}
