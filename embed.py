"""Compatibility wrappers for text embeddings."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import EMBEDDING_MODEL_NAME

_EMBED_DIM = 384


class EmbeddingModel:
    """Lazy sentence embedding model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        """Encode text into normalized vectors.

        Args:
            texts: Text values to encode.
            batch_size: Model batch size.

        Returns:
            Float matrix with one row per text.
        """
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)
_DEFAULT_MODEL = EmbeddingModel()

def get_model() -> SentenceTransformer:
    return _DEFAULT_MODEL.model


def embed_texts(texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    return _DEFAULT_MODEL.encode(texts, batch_size=batch_size)


def embed_queries(queries: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    return _DEFAULT_MODEL.encode(queries, batch_size=batch_size)
