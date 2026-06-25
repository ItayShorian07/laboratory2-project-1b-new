"""Retrieval package API."""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "Normalizer": "normalizer",
    "_minmax_rows": "normalizer",
    "_minmax_1d": "normalizer",
    "DenseRetriever": "dense",
    "HybridFuser": "fusion",
    "RetrievalPipeline": "pipeline",
    "search_batch": "service",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__
