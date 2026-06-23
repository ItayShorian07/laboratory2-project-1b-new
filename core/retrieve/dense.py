"""Dense retrieval signal (MiniLM embeddings via NumPy dot products)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..embed import EmbeddingModel
from ..interfaces import PageScorer


class DenseRetriever(PageScorer):
    """Dense signal: MiniLM query embeddings vs. page vectors."""

    def __init__(self, page_vectors: np.ndarray, embedder: EmbeddingModel) -> None:
        self._page_vectors = page_vectors
        self._embedder = embedder

    def score(self, queries: Sequence[str]) -> np.ndarray:
        """Full per-page inner-product scores, aligned to page (column) order."""
        query_vectors = self._embedder.encode(list(queries))
        return np.asarray(query_vectors @ self._page_vectors.T, dtype=np.float32)
