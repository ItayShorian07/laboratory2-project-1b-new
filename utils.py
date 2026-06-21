"""Shared paths and helpers for Section B."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

STUDENT_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDENT_ROOT / "data"
ENTRIES_DIR = DATA_DIR / "Wikipedia Entries"
PUBLIC_QUERIES_PATH = DATA_DIR / "public_queries.json"
ARTIFACTS_DIR = STUDENT_ROOT / "artifacts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_EVAL = 10

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?")
STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "being",
    "between",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "during",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "not",
    "of",
    "on",
    "one",
    "or",
    "our",
    "over",
    "per",
    "shall",
    "she",
    "should",
    "that",
    "the",
    "their",
    "them",
    "then",
    "they",
    "this",
    "through",
    "to",
    "together",
    "two",
    "under",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "will",
    "with",
    "without",
    "would",
    "you",
    "your",
}


def normalize_page_id(value: Any) -> int:
    """Coerce a JSON page identifier to an integer.

    Args:
        value: Page identifier read from JSON.

    Returns:
        The normalized integer page identifier.

    Raises:
        ValueError: If the value is not an integer or numeric string.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Invalid page_id: {value!r}")


def load_public_queries(path: Path | None = None) -> List[Dict[str, Any]]:
    """Load public queries with integer relevance labels.

    Args:
        path: Optional query file path.

    Returns:
        Query rows with normalized integer ``relevant_page_ids``.
    """
    path = path or PUBLIC_QUERIES_PATH
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["relevant_page_ids"] = [
            normalize_page_id(pid) for pid in row["relevant_page_ids"]
        ]
    return rows


def iter_entries(entries_dir: Path | None = None) -> Iterator[Dict[str, Any]]:
    """Yield one corpus record per JSON file.

    Args:
        entries_dir: Optional corpus directory override.

    Yields:
        Parsed page records with normalized integer ``page_id`` values.

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
    """Return the text used for dense document embedding.

    Args:
        record: A corpus page record.

    Returns:
        Title and content joined as one text block.
    """
    title = record.get("title", "")
    content = record.get("content", "")
    if title:
        return f"{title}\n\n{content}".strip()
    return str(content).strip()


def ensure_artifacts_dir() -> Path:
    """Create and return the artifacts directory.

    Returns:
        The local artifacts directory path.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def tokenize(text: str) -> List[str]:
    """Tokenize text for lexical retrieval.

    Args:
        text: Text to tokenize.

    Returns:
        Lowercase alphanumeric tokens with common stopwords removed.
    """
    tokens: List[str] = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("'-")
        if len(token) >= 2 and token not in STOPWORDS:
            tokens.append(token)
    return tokens
