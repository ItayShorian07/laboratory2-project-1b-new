"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

from pathlib import Path
import heapq
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

from embed import embed_queries
from index import load_index, load_lexical_index
from utils import K_EVAL
from utils import tokenize

DENSE_CANDIDATES = 500
LEXICAL_CANDIDATES = 500
BM25_K1 = 1.2
BM25_B = 0.75
TITLE_BOOST = 0.5
DENSE_WEIGHT = 1.0
LEXICAL_WEIGHT = 1.0


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Return ranked page identifiers for each query.

    Args:
        queries: Query strings.
        top_k: Number of page identifiers to return per query.
        artifacts_dir: Optional artifacts directory override.

    Returns:
        Ranked page identifier lists, best first.
    """
    corpus_vectors, page_ids = load_index(artifacts_dir)
    lexical_index = load_lexical_index(artifacts_dir)
    lexical_page_ids = [int(x) for x in lexical_index["page_ids"]]
    if lexical_page_ids != page_ids:
        raise ValueError("Dense and lexical artifacts use different page_id ordering")

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    dense_scores = query_vectors @ corpus_vectors.T
    ranked: List[List[int]] = []
    for query, row in zip(queries, dense_scores):
        dense = _top_dense_scores(row, DENSE_CANDIDATES)
        lexical = _top_lexical_scores(query, lexical_index, LEXICAL_CANDIDATES)
        combined = _combine_scores(dense, lexical)
        best_docs = heapq.nlargest(top_k, combined.items(), key=lambda item: item[1])
        ranked.append([page_ids[doc_idx] for doc_idx, _ in best_docs])
    return ranked


def _top_dense_scores(row: np.ndarray, limit: int) -> Dict[int, float]:
    """Return the strongest dense scores for one query.

    Args:
        row: Dense similarity scores for all pages.
        limit: Maximum number of candidates to keep.

    Returns:
        Mapping from document index to dense score.
    """
    if row.size == 0:
        return {}
    keep = min(limit, row.size)
    if keep == row.size:
        order = np.argsort(-row)
    else:
        order = np.argpartition(-row, keep - 1)[:keep]
        order = order[np.argsort(-row[order])]
    return {int(idx): float(row[int(idx)]) for idx in order}


def _top_lexical_scores(query: str, lexical_index: dict, limit: int) -> Dict[int, float]:
    """Return top BM25 scores for one query.

    Args:
        query: Query string.
        lexical_index: Precomputed lexical index.
        limit: Maximum number of candidates to keep.

    Returns:
        Mapping from document index to BM25 score.
    """
    scores: dict[int, float] = defaultdict(float)
    query_terms = Counter(tokenize(query))
    postings = lexical_index["postings"]
    title_postings = lexical_index["title_postings"]
    doc_lengths = lexical_index["doc_lengths"]
    avg_doc_length = float(lexical_index["avg_doc_length"])
    total_docs = len(lexical_index["page_ids"])

    if avg_doc_length <= 0.0 or total_docs == 0:
        return {}

    for term in query_terms:
        rows = postings.get(term)
        if rows is None:
            continue
        doc_indices, freqs = rows
        doc_freq = len(doc_indices)
        idf = math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        lengths = doc_lengths[doc_indices]
        denom = freqs + BM25_K1 * (1.0 - BM25_B + BM25_B * lengths / avg_doc_length)
        term_scores = idf * (freqs * (BM25_K1 + 1.0)) / denom
        for doc_idx, score in zip(doc_indices, term_scores):
            scores[int(doc_idx)] += float(score)

        title_docs = title_postings.get(term)
        if title_docs is not None:
            for doc_idx in title_docs:
                scores[int(doc_idx)] += TITLE_BOOST * idf

    return dict(heapq.nlargest(limit, scores.items(), key=lambda item: item[1]))


def _combine_scores(
    dense_scores: Dict[int, float],
    lexical_scores: Dict[int, float],
) -> Dict[int, float]:
    """Combine dense and lexical scores after per-query normalization.

    Args:
        dense_scores: Dense candidate scores.
        lexical_scores: BM25 candidate scores.

    Returns:
        Fused scores keyed by document index.
    """
    dense_norm = _minmax(dense_scores)
    lexical_norm = _minmax(lexical_scores)
    combined: Dict[int, float] = {}
    for doc_idx in set(dense_norm) | set(lexical_norm):
        combined[doc_idx] = (
            DENSE_WEIGHT * dense_norm.get(doc_idx, 0.0)
            + LEXICAL_WEIGHT * lexical_norm.get(doc_idx, 0.0)
        )
    return combined


def _minmax(scores: Dict[int, float]) -> Dict[int, float]:
    """Normalize a score mapping to the ``[0, 1]`` range.

    Args:
        scores: Raw scores keyed by document index.

    Returns:
        Min-max normalized scores.
    """
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high <= low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}
