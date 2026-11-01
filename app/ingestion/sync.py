"""Incremental folder sync: metadata diff -> re-index only what changed.

The sync is metadata-first: ``app.indexing.metadata`` records each file's size,
mtime and content hash, so the next run knows which files were added, modified
or deleted without opening them. A changed file is then hashed to confirm the
content really differs (a ``touch`` costs no embeddings), and only the changed
files are re-chunked and re-embedded.
"""

import json
from pathlib import Path

from app.core.config import (
    CACHE_PRUNE_MIN_EXTRA,
    CACHE_PRUNE_RATIO,
    DOCS_DIR,
    INDEX_VERSION,
    QDRANT_COLLECTION,
)
from app.core.scope import normalize_root
from app.ingestion.identity import file_hash
from app.ingestion.qdrant_index import (
    _ensure_collection,
    _serialized,
    cache_line_count,
    index_paths,
    prune_cache,
    remove_paths,
)
from app.retrieval.store import get_client


def _resolve_root(root: Path) -> Path:
    """Never create an arbitrary folder from a mistyped path.

    Only the built-in docs folder is allowed to be created on a fresh checkout.
    """
    root = Path(root)
    if not root.is_dir():
        if root == Path(DOCS_DIR):
            root.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Folder not found: {root.as_posix()}")
    return root


def _classify_changes(cache, scope: str, cached: dict, files: dict, diff) -> tuple[dict[str, str], list[str]]:
    """Hash files that mtime/size flagged; keep only real content changes.

    Mechanism 2: mtime/size says "maybe changed"; the hash decides for real.
    A handler-only change (``touch``, duplicate watcher event) is marked seen so
    the next diff treats it as unchanged.
    """
    hashes: dict[str, str] = {}
    changed: list[str] = []
    for rel_path in diff.changed:
        meta = files[rel_path]
        digest = file_hash(Path(meta.abs_path))
        hashes[rel_path] = digest
        row = cached.get(rel_path)
        if (
            row
            and row.get("content_hash") == digest
            and row.get("status") == "indexed"
            and row.get("index_version") == INDEX_VERSION
        ):
            cache.mark_seen(scope, meta, digest)
            continue
        changed.append(rel_path)
    return hashes, changed


def _group_by_path(chunks: list[dict]) -> dict[str, list[dict]]:
    chunks_by_path: dict[str, list[dict]] = {}
    for chunk in chunks:
        chunks_by_path.setdefault(chunk["path"], []).append(chunk)
    return chunks_by_path


def _sync_outlines(root: Path, hashes: dict, chunks_by_path: dict, deleted: list[str]) -> dict:
    """Refresh each Document Card's outline tree; never let cards break indexing."""
    try:
        from app.capabilities.registry import sync_outlines

        return sync_outlines(root, hashes, chunks_by_path, deleted)
    except Exception as error:
        print(f"warning: could not update document outlines: {error}")
        return {"updated": 0, "deleted": 0, "error": str(error)}


def _maybe_prune_cache(client, stats: dict) -> None:
    """Threshold-based cleanup: only when the cache has grown well past the index."""
    live_count = client.count(collection_name=QDRANT_COLLECTION).count
    cached_count = cache_line_count()
    if cached_count > live_count * CACHE_PRUNE_RATIO + CACHE_PRUNE_MIN_EXTRA:
        stats["cache_pruned"] = prune_cache()["removed"]


@_serialized
def sync_folder(root: Path = DOCS_DIR, reset: bool = False, auto_prune: bool = True) -> dict:
    """Scan ``root`` and index only new/changed files (metadata-first)."""
    from app.indexing.metadata import get_cache
    from app.indexing.scanner import snapshot

    root = _resolve_root(root)
    scope = normalize_root(root)
    client = get_client()
    recreated = _ensure_collection(client, reset=reset)

    cache = get_cache()
    if reset or recreated:
        # The whole collection was rebuilt, so every cached row is meaningless.
        cache.clear()

    # Mechanism 1: metadata only (no file contents are opened here).
    files = snapshot(root)
    cached = cache.rows(scope)
    diff = cache.diff(scope, files, INDEX_VERSION, force=reset or recreated)

    # Mechanism 2: the hash confirms which "maybe changed" files really changed.
    hashes, changed = _classify_changes(cache, scope, cached, files, diff)

    deleted = diff.deleted
    if deleted:
        remove_paths(deleted)

    chunks, embeddings_computed = index_paths(root, changed)
    chunks_by_path = _group_by_path(chunks)

    cache.mark_indexed_many(
        scope,
        [
            (
                files[rel_path],
                hashes[rel_path],
                len(chunks_by_path.get(rel_path, [])),
                INDEX_VERSION,
            )
            for rel_path in changed
        ],
    )
    for rel_path in changed:
        if not chunks_by_path.get(rel_path):
            print(f"warning: no extractable text in {rel_path} (scanned/image PDF?)")
    if deleted:
        cache.remove(scope, deleted)

    # Single source of truth: refresh each Document Card's outline tree from the
    # chunks just written (changed files) or from Qdrant once (missing/stale
    # cards). Chat then reads the tree from the RAM-cached registry instead of
    # scrolling Qdrant on every turn.
    changed_hashes = {rel_path: hashes[rel_path] for rel_path in changed}
    stats_cards = _sync_outlines(root, changed_hashes, chunks_by_path, deleted)

    stats = {
        "files": len(files),
        "changed": len(changed),
        "deleted": len(deleted),
        "chunks_reindexed": len(chunks),
        "embeddings_computed": embeddings_computed,
        "outlines": stats_cards,
    }

    cache.set_scope_state(
        scope,
        scanned=True,
        synced=True,
        stats=json.dumps(stats, ensure_ascii=False),
    )

    if auto_prune:
        _maybe_prune_cache(client, stats)
    return stats
