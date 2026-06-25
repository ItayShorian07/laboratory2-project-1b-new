"""Dense retrieval signal."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..embed import EmbeddingModel
from ..interfaces import PageScorer


class DenseRetriever(PageScorer):
    """Scores pages with embedding similarity."""

    def __init__(self, page_vectors: np.ndarray, embedder: EmbeddingModel) -> None:
        self._page_vectors = page_vectors
        self._embedder = embedder

    def score(self, queries: Sequence[str]) -> np.ndarray:
        """Score pages for queries.

        Args:
            queries: Search queries.

        Returns:
            Score matrix with one row per query.
        """
        query_vectors = self._embedder.encode(list(queries))
        return np.asarray(query_vectors @ self._page_vectors.T, dtype=np.float32)
