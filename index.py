"""Offline index build and load (not timed at grading)."""
from __future__ import annotations

import json
import gzip
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Optional, Tuple

import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries, tokenize

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"
LEXICAL_INDEX_NAME = "lexical_index.pkl.gz"
TITLE_TOKEN_WEIGHT = 5


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Build and persist dense and lexical retrieval artifacts.

    Args:
        entries_dir: Optional corpus directory override.
        artifacts_dir: Optional output directory override.

    Returns:
        A tuple ``(vectors, page_ids)`` for the dense index.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(iter_entries(entries_dir))
    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    page_ids = [c.page_id for c in chunks]

    np.save(out_dir / INDEX_VECTORS_NAME, vectors)
    meta = {
        "page_ids": page_ids,
        "chunk_ids": [c.chunk_id for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
    }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    build_lexical_index(records, out_dir)
    return vectors, page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load dense vectors and their page identifiers.

    Args:
        artifacts_dir: Optional artifacts directory override.

    Returns:
        A tuple ``(vectors, page_ids)``.
    """
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids


def build_lexical_index(records: List[dict], artifacts_dir: Path) -> None:
    """Build a compact BM25 index over the full page text.

    Args:
        records: Corpus records in the same order as the dense index.
        artifacts_dir: Directory where the lexical artifact is written.
    """
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    title_postings: dict[str, list[int]] = defaultdict(list)
    doc_lengths: List[int] = []
    page_ids: List[int] = []

    for doc_idx, record in enumerate(records):
        page_ids.append(int(record["page_id"]))
        title_tokens = tokenize(str(record.get("title", "")))
        body_tokens = tokenize(str(record.get("content", "")))
        term_counts = Counter(title_tokens * TITLE_TOKEN_WEIGHT + body_tokens)
        doc_lengths.append(sum(term_counts.values()))

        for term in set(title_tokens):
            title_postings[term].append(doc_idx)
        for term, freq in term_counts.items():
            postings[term].append((doc_idx, int(freq)))

    compact_postings = {
        term: (
            np.asarray([doc for doc, _ in rows], dtype=np.int32),
            np.asarray([freq for _, freq in rows], dtype=np.float32),
        )
        for term, rows in postings.items()
    }
    compact_titles = {
        term: np.asarray(rows, dtype=np.int32) for term, rows in title_postings.items()
    }
    index = {
        "page_ids": np.asarray(page_ids, dtype=np.int64),
        "doc_lengths": np.asarray(doc_lengths, dtype=np.float32),
        "avg_doc_length": float(np.mean(doc_lengths)) if doc_lengths else 0.0,
        "postings": compact_postings,
        "title_postings": compact_titles,
    }
    with gzip.open(artifacts_dir / LEXICAL_INDEX_NAME, "wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_lexical_index(artifacts_dir: Optional[Path] = None) -> dict:
    """Load the precomputed BM25 artifact.

    Args:
        artifacts_dir: Optional artifacts directory override.

    Returns:
        The lexical index dictionary produced by ``build_lexical_index``.
    """
    root = artifacts_dir or ARTIFACTS_DIR
    with gzip.open(root / LEXICAL_INDEX_NAME, "rb") as fh:
        return pickle.load(fh)
