"""Capability: scoped hybrid search + quick answer."""

from app.core.config import DEFAULT_TOP_K
from app.core.scope import normalize_root
from app.retrieval.retrieve import vector_search


def search(
    scope: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    paths: list[str] | None = None,
) -> list[dict]:
    """Hybrid search restricted to ``scope`` (a folder root) and, if given, ``paths``."""
    query = (query or "").strip()
    if not query or not scope:
        return []
    clean_paths = [p for p in dict.fromkeys(paths or []) if p] or None
    return vector_search(
        query, top_k=top_k, scope=normalize_root(scope), paths=clean_paths
    )