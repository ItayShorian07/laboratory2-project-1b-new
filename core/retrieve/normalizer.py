"""Score normalization."""
from __future__ import annotations

import numpy as np


class Normalizer:
    """Normalizes scores with min max scaling."""

    @staticmethod
    def rows(matrix: np.ndarray) -> np.ndarray:
        """Normalize each matrix row.

        Args:
            matrix: Score matrix.

        Returns:
            Normalized score matrix.
        """
        lo = matrix.min(axis=1, keepdims=True)
        hi = matrix.max(axis=1, keepdims=True)
        rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        return (matrix - lo) / rng

    @staticmethod
    def vector(values: np.ndarray) -> np.ndarray:
        """Normalize one vector.

        Args:
            values: Score values.

        Returns:
            Normalized score values.
        """
        lo = float(values.min())
        rng = float(values.max()) - lo
        return (values - lo) / (rng if rng > 1e-12 else 1.0)

_minmax_rows = Normalizer.rows
_minmax_1d = Normalizer.vector
