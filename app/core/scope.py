"""Scope helpers.

A *scope* is the folder the user is currently asking about. It is stored on every
point as the ``root`` payload field, so retrieval can be restricted to one folder
even though many folders share a single Qdrant collection.

Both ingestion and retrieval normalise the root the same way, otherwise the stored
``root`` and the query filter would not match.
"""

from pathlib import Path


def normalize_root(root: str | Path) -> str:
    """Canonical string used as the ``root`` payload value and as a query scope.

    Resolves to an absolute path and renders it with forward slashes so the same
    folder always produces the same key on a given machine.
    """
    return Path(root).expanduser().resolve().as_posix()