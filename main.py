"""Entry points for the retrieval project."""
from __future__ import annotations

from typing import List

from core.index import build_index
from core.retrieve import search_batch


def run(queries: List[str]) -> List[List[int]]:
    """Rank pages for each query.

    Args:
        queries: Search queries.

    Returns:
        Ranked page ids for each query.
    """
    return search_batch(queries)


def build_offline_index() -> None:
    """Build local retrieval artifacts."""
    build_index()


if __name__ == "__main__":
    build_offline_index()
    print("Index built under artifacts/. Run: python scripts/eval_public.py")
