"""Process level retrieval service."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from utils import K_EVAL

from ..index import load_index
from .pipeline import RetrievalPipeline

_PIPELINE: Optional[RetrievalPipeline] = None
_ARTIFACTS_KEY: Optional[Path] = None


def _get_pipeline(artifacts_dir: Optional[Path]) -> RetrievalPipeline:
    global _PIPELINE, _ARTIFACTS_KEY
    if _PIPELINE is None or _ARTIFACTS_KEY != artifacts_dir:
        _PIPELINE = RetrievalPipeline.from_index(load_index(artifacts_dir))
        _ARTIFACTS_KEY = artifacts_dir
    return _PIPELINE


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Rank pages for a batch of queries.

    Args:
        queries: Search queries.
        top_k: Number of page ids to return.
        artifacts_dir: Optional artifact directory.

    Returns:
        Ranked page ids for each query.
    """
    if not queries:
        return []
    return _get_pipeline(artifacts_dir).search(queries, top_k=top_k)
