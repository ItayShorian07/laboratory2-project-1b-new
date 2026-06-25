"""Artifact names and retrieval defaults."""
from __future__ import annotations

PAGE_VECTORS_NAME = "page_vectors.npy"
PAGE_IDS_NAME = "page_ids.npy"
PAGE_TEXTS_NAME = "page_texts.npy"
BM25_NAME = "bm25_index.npz"
CONFIG_NAME = "retrieval_config.json"

DEFAULT_ALPHA = 0.7

RERANK_MAX_CHARS = 2000

DEFAULT_RERANK = {
    "enabled": True,
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "pool": 120,
    "weight": 0.6,
    "max_length": 512,
}
