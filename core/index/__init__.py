"""Index package API."""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "PAGE_VECTORS_NAME": "config",
    "PAGE_IDS_NAME": "config",
    "PAGE_TEXTS_NAME": "config",
    "BM25_NAME": "config",
    "CONFIG_NAME": "config",
    "DEFAULT_ALPHA": "config",
    "RERANK_MAX_CHARS": "config",
    "DEFAULT_RERANK": "config",
    "LoadedIndex": "loaded_index",
    "IndexBuilder": "builder",
    "build_index": "builder",
    "IndexLoader": "loader",
    "load_index": "loader",
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
