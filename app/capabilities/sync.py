"""Capability: keep a scope's index fresh (incremental)."""

from pathlib import Path

from app.core.scope import normalize_root
from app.ingestion.folder import sync_folder


def sync(scope: str) -> dict:
    """Re-index ``scope``. Only changed files are re-chunked/re-embedded."""
    if not scope:
        return {"error": "scope is required"}
    return sync_folder(Path(scope), reset=False)