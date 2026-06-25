"""Shared paths and data helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

STUDENT_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDENT_ROOT / "data"
ENTRIES_DIR = DATA_DIR / "Wikipedia Entries"
PUBLIC_QUERIES_PATH = DATA_DIR / "public_queries.json"
ARTIFACTS_DIR = STUDENT_ROOT / "artifacts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_EVAL = 10


def normalize_page_id(value: Any) -> int:
    """Convert a page id to an integer.

    Args:
        value: Page id from a data file.

    Returns:
        Page id as an integer.

    Raises:
        ValueError: If the value is not numeric.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Invalid page_id: {value!r}")


def load_public_queries(path: Path | None = None) -> List[Dict[str, Any]]:
    """Load public query rows.

    Args:
        path: Optional query file path.

    Returns:
        Query rows with normalized page ids.
    """
    path = path or PUBLIC_QUERIES_PATH
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["relevant_page_ids"] = [
            normalize_page_id(pid) for pid in row["relevant_page_ids"]
        ]
    return rows


def iter_entries(entries_dir: Path | None = None) -> Iterator[Dict[str, Any]]:
    """Yield corpus records.

    Args:
        entries_dir: Optional corpus directory.

    Yields:
        One normalized page record.

    Raises:
        FileNotFoundError: If the corpus directory is missing.
    """
    root = entries_dir or ENTRIES_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {root}. "
            "Expected student/data/Wikipedia Entries/ with one JSON file per page."
        )
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["page_id"] = normalize_page_id(data.get("page_id", path.stem))
        yield data


def entry_text(record: Dict[str, Any]) -> str:
    """Build searchable text for a page.

    Args:
        record: Page record.

    Returns:
        Title and content text.
    """
    title = record.get("title", "")
    content = record.get("content", "")
    if title:
        return f"{title}\n\n{content}".strip()
    return str(content).strip()


def ensure_artifacts_dir() -> Path:
    """Create the artifacts directory.

    Returns:
        Artifact directory path.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
