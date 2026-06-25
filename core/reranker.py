"""Cross encoder reranking."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .interfaces import Reranker


class CrossEncoderReranker(Reranker):
    """Lazy cross encoder reranker."""

    def __init__(self, model_name: str, *, max_length: int = 512, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import CrossEncoder

            self._encoder = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._encoder

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Score query and passage pairs.

        Args:
            pairs: Query and passage pairs.

        Returns:
            Relevance scores.
        """
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        scores = self.encoder.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return np.asarray(scores, dtype=np.float32)

_RERANKERS: dict[str, CrossEncoderReranker] = {}


def get_reranker(model_name: str, max_length: int = 512) -> CrossEncoderReranker:
    reranker = _RERANKERS.get(model_name)
    if reranker is None:
        reranker = CrossEncoderReranker(model_name, max_length=max_length)
        _RERANKERS[model_name] = reranker
    return reranker


def score_pairs(
    model_name: str,
    pairs: List[Tuple[str, str]],
    *,
    max_length: int = 512,
    batch_size: int = 64,
) -> np.ndarray:
    reranker = get_reranker(model_name, max_length=max_length)
    reranker.batch_size = batch_size
    return reranker.score_pairs(pairs)
