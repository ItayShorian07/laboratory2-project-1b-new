"""Page chunk helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from utils import entry_text


@dataclass
class Chunk:
    """A retrieval text unit."""

    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Create chunks for one page.

    Args:
        record: Page record.

    Returns:
        Page chunks.
    """
    return [Chunk(page_id=int(record["page_id"]), chunk_id=0, text=entry_text(record))]


def chunk_corpus(records: Iterable[Dict[str, Any]]) -> List[Chunk]:
    """Create chunks for a corpus.

    Args:
        records: Page records.

    Returns:
        All page chunks.
    """
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
