"""Capability: a document's canonical outline and one or more whole sections.

These are the two structural building blocks the self-querying retriever uses,
exposed as tools so the UI (or a future agent) can browse a document before
searching: list its canonical section tree, then read a section by its
materialized-path id (``s4``, ``s4.1``).
"""

from app.core.config import SECTION_CHUNKS_LIMIT
from app.retrieval.retrieve import document_outline, section_chunks


def outline(scope: str, path: str) -> dict:
    """Title + canonical section tree of one document."""
    return document_outline(scope, path)


def section(
    scope: str,
    path: str,
    section_ids: str | list[str],
    limit: int = SECTION_CHUNKS_LIMIT,
    include_descendants: bool = True,
) -> list[dict]:
    """Every chunk of one or more sections, in reading order."""
    return section_chunks(
        scope,
        path,
        section_ids,
        limit=limit,
        include_descendants=include_descendants,
    )
