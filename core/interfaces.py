"""Retrieval interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import numpy as np


class PageScorer(ABC):
    """Scores pages for queries."""

    @abstractmethod
    def score(self, queries: Sequence[str]) -> np.ndarray:
        """Score pages for queries.

        Args:
            queries: Search queries.

        Returns:
            Score matrix with one row per query.
        """


class Reranker(ABC):
    """Scores candidate passages."""

    @abstractmethod
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Score query and passage pairs.

        Args:
            pairs: Query and passage pairs.

        Returns:
            Relevance scores.
        """
