"""Score fusion."""
from __future__ import annotations

import numpy as np

from .normalizer import Normalizer


class HybridFuser:
    """Combines dense and lexical scores."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def fuse(self, dense: np.ndarray, lexical: np.ndarray) -> np.ndarray:
        """Fuse two score matrices.

        Args:
            dense: Dense retrieval scores.
            lexical: Lexical retrieval scores.

        Returns:
            Fused scores.
        """
        return (
            self.alpha * Normalizer.rows(dense)
            + (1.0 - self.alpha) * Normalizer.rows(lexical)
        )
