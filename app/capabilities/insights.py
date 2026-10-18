"""Capability: Insights — cached per-document summaries.

Borrowed from Open Notebook's *Insight* concept: instead of re-reading a whole
document every time, we compute a summary once, cache it on disk, and reuse it as
a cheap, citable piece of context.

Short documents are summarized in one pass; long ones use map-reduce over their
indexed chunks. The cache is invalidated by the file content hash, so it follows
edits automatically.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client.http import models

from app.core.config import INDEX_DIR, QDRANT_COLLECTION
from app.core.prompts import (
    build_insight_map_prompt,
    build_insight_reduce_prompt,
    build_insight_single_prompt,
)
from app.core.scope import normalize_root
from app.ingestion.folder import file_hash
from app.reasoning.answer import _generate
from app.retrieval.store import get_client

INSIGHT_DIR = INDEX_DIR / "insights"

# A document whose full text fits here is summarized in a single call.
SINGLE_PASS_CHARS = 14000
# When map-reducing, chunks are packed into groups of about this size.
MAP_GROUP_CHARS = 6000


def _cache_path(root: str, rel_path: str) -> Path:
    root_key = hashlib.md5(root.encode("utf-8")).hexdigest()[:8]
    file_key = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:8]
    return INSIGHT_DIR / f"{root_key}_{file_key}.json"


def _load_cache(root: str, rel_path: str) -> dict | None:
    path = _cache_path(root, rel_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(root: str, rel_path: str, record: dict) -> None:
    path = _cache_path(root, rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def _file_chunks(root: str, rel_path: str) -> list[dict]:
    """All indexed chunks of one file, in document order."""
    qfilter = models.Filter(
        must=[
            models.FieldCondition(key="root", match=models.MatchValue(value=root)),
            models.FieldCondition(key="path", match=models.MatchValue(value=rel_path)),
        ]
    )
    client = get_client()
    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=1000,
        with_payload=True,
        with_vectors=False,
        scroll_filter=qfilter,
    )

    def order(point) -> int:
        chunk_id = (point.payload or {}).get("chunk_id", "")
        try:
            return int(chunk_id.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            return 0

    return [p.payload or {} for p in sorted(points, key=order)]


def _group(texts: list[str], max_chars: int) -> list[str]:
    """Pack successive texts into groups of at most ``max_chars`` characters."""
    groups: list[str] = []
    current = ""
    for text in texts:
        candidate = f"{current}\n\n{text}".strip() if current else text
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            groups.append(current)
        current = text
    if current:
        groups.append(current)
    return groups


def _compose(title: str, chunks: list[dict]) -> str:
    """One-pass summarize a short document."""
    full_text = "\n\n".join(chunk.get("text", "") for chunk in chunks)
    if len(full_text) <= SINGLE_PASS_CHARS:
        return _generate(build_insight_single_prompt(title, full_text), max_tokens=900)

    groups = _group([chunk.get("text", "") for chunk in chunks], MAP_GROUP_CHARS)
    partials = [
        _generate(build_insight_map_prompt(title, group), max_tokens=400)
        for group in groups
    ]
    return _generate(build_insight_reduce_prompt(title, partials), max_tokens=900)


def summarize_source(scope: str, rel_path: str, force: bool = False) -> dict:
    """Return a cached or freshly computed insight for one document."""
    root = normalize_root(scope)
    abs_path = Path(root) / rel_path
    if not abs_path.exists():
        return {"error": f"file not found: {rel_path}"}

    current_hash = file_hash(abs_path)
    cached = _load_cache(root, rel_path)
    if cached and cached.get("file_hash") == current_hash and not force:
        return {**cached, "cached": True}

    chunks = _file_chunks(root, rel_path)
    if not chunks:
        return {"error": "no indexed chunks for this file", "path": rel_path}

    title = Path(rel_path).stem
    summary = _compose(title, chunks)
    record = {
        "path": rel_path,
        "title": title,
        "file_hash": current_hash,
        "summary": summary,
        "chunks": len(chunks),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_cache(root, rel_path, record)
    return {**record, "cached": False}


def get_insight(scope: str, rel_path: str) -> dict | None:
    """Return the cached insight only (no LLM call)."""
    root = normalize_root(scope)
    cached = _load_cache(root, rel_path)
    if not cached:
        return None
    abs_path = Path(root) / rel_path
    if abs_path.exists() and cached.get("file_hash") == file_hash(abs_path):
        return {**cached, "stale": False}
    return {**cached, "stale": True}


def list_insights(scope: str) -> list[dict]:
    """Which files in the scope already have a (fresh or stale) insight."""
    from app.capabilities.sources import list_sources

    root = normalize_root(scope)
    output = []
    for item in list_sources(scope):
        cached = _load_cache(root, item["path"])
        if cached:
            abs_path = Path(root) / item["path"]
            stale = not abs_path.exists() or cached.get("file_hash") != file_hash(abs_path)
            output.append(
                {
                    "path": item["path"],
                    "title": item["title"],
                    "updated_at": cached.get("updated_at"),
                    "stale": stale,
                }
            )
    return output