"""
Section B entry point.

The autograder calls run(queries) once with all evaluation queries.
Query embedding + retrieval must complete within the time limit (GPU available).
"""
from __future__ import annotations

from typing import List

from core.index import build_index
from core.retrieve import search_batch


def run(queries: List[str]) -> List[List[int]]:
    """Rank corpus pages for each query.

    Args:
        queries: Batch of query strings.

    Returns:
        One ranked list of page_id values per query, most relevant first.
    """
    return search_batch(queries)


def build_offline_index() -> None:
    """Build the submitted artifacts locally before grading."""
    build_index()


if __name__ == "__main__":
    build_offline_index()
    print("Index built under artifacts/. Run: python scripts/eval_public.py")
