"""Shared embedding wrapper for documents and query text."""
from __future__ import annotations

from collections.abc import Sequence

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEncoder:
    """Encode documents and queries with the same normalized embedding model.

    The dependency is imported lazily so core retrieval tests can run offline
    without downloading model weights or importing PyTorch.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Sentence Transformers is required for embedding. "
                "Install the project requirements with: "
                "python -m pip install -r requirements.txt"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return L2-normalized vectors in a dependency-neutral Python format."""
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
