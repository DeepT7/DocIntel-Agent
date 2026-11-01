"""Capability: list the documents that belong to a scope."""

from qdrant_client.http import models

from app.core.config import QDRANT_COLLECTION, SCROLL_PAGE_SIZE
from app.core.scope import normalize_root
from app.retrieval.store import get_client


def list_sources(scope: str) -> list[dict]:
    """Return the distinct files indexed under ``scope``.

    Reads only payload metadata, so it is cheap and does not touch vectors.
    """
    if not scope:
        return []

    root = normalize_root(scope)
    qfilter = models.Filter(
        must=[models.FieldCondition(key="root", match=models.MatchValue(value=root))]
    )

    client = get_client()
    seen: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=qfilter,
        )
        for point in points:
            payload = point.payload or {}
            path = payload.get("path")
            if not path or path in seen:
                continue
            seen[path] = {
                "path": path,
                "title": payload.get("title", path),
                "file_id": payload.get("file_id"),
            }
        if offset is None:
            break

    return sorted(seen.values(), key=lambda item: item["path"])