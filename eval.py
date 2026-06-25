"""Evaluation helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from utils import K_EVAL, normalize_page_id

__all__ = [
    "K_EVAL",
    "dcg_at_k",
    "ndcg_at_k",
    "mean_ndcg_at_k",
    "load_query_file",
    "evaluate_run",
]


def dcg_at_k(gains: Sequence[float], k: int = K_EVAL) -> float:
    """Compute discounted cumulative gain.

    Args:
        gains: Relevance gains in rank order.
        k: Evaluation depth.

    Returns:
        Discounted cumulative gain.
    """
    gains = list(gains[:k])
    if not gains:
        return 0.0
    total = gains[0]
    for i, g in enumerate(gains[1:], start=2):
        total += g / math.log2(i)
    return total


def ndcg_at_k(
    ranked_ids: Sequence[int],
    relevant_ids: Set[int],
    k: int = K_EVAL,
) -> float:
    """Compute normalized discounted gain.

    Args:
        ranked_ids: Predicted page ids.
        relevant_ids: Relevant page ids.
        k: Evaluation depth.

    Returns:
        Normalized discounted gain.
    """
    seen: Set[int] = set()
    gains: List[float] = []
    for pid in ranked_ids:
        if pid in seen:
            continue
        seen.add(pid)
        gains.append(1.0 if pid in relevant_ids else 0.0)
        if len(gains) >= k:
            break

    dcg = dcg_at_k(gains, k)
    n_rel = min(len(relevant_ids), k)
    if n_rel == 0:
        return 0.0
    ideal = [1.0] * n_rel
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def mean_ndcg_at_k(
    all_ranked: Sequence[Sequence[int]],
    all_relevant: Sequence[Set[int]],
    k: int = K_EVAL,
) -> float:
    """Compute mean normalized discounted gain.

    Args:
        all_ranked: Predicted rankings.
        all_relevant: Relevant page id sets.
        k: Evaluation depth.

    Returns:
        Mean score across queries.
    """
    if not all_ranked:
        return 0.0
    scores = [
        ndcg_at_k(ranked, rel, k=k)
        for ranked, rel in zip(all_ranked, all_relevant)
    ]
    return float(sum(scores) / len(scores))


def load_query_file(path: Path) -> List[Dict[str, Any]]:
    """Load an evaluation query file.

    Args:
        path: Query file path.

    Returns:
        Query rows with relevant page id sets.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["relevant_page_ids"] = {
            normalize_page_id(x) for x in row["relevant_page_ids"]
        }
    return rows


def evaluate_run(
    queries: List[str],
    ground_truth: List[Set[int]],
    run_fn,
    *,
    k: int = K_EVAL,
) -> Dict[str, float]:
    """Evaluate a ranking function.

    Args:
        queries: Query texts.
        ground_truth: Relevant page id sets.
        run_fn: Function that returns rankings.
        k: Evaluation depth.

    Returns:
        Evaluation metrics.
    """
    ranked = run_fn(queries)
    if len(ranked) != len(queries):
        raise ValueError(
            f"run() returned {len(ranked)} lists for {len(queries)} queries"
        )
    for i, row in enumerate(ranked):
        if not isinstance(row, list):
            raise TypeError(f"run()[{i}] must be a list of page_id, got {type(row)}")
    score = mean_ndcg_at_k(ranked, ground_truth, k=k)
    return {"mean_ndcg@10": score, "num_queries": float(len(queries))}
