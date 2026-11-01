"""Bridge between the watcher queue and the existing ingestion pipeline.

A watcher batch contains file-level changes; this module applies exactly those
files (never the whole folder) to Qdrant, the document outlines and the metadata
cache. Heavy imports are local so the cache/scanner can be used without Qdrant
(e.g. in tests or a read-only process).
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import INDEX_VERSION
from app.core.scope import normalize_root
from app.indexing.metadata import MetadataCache
from app.indexing.scanner import file_meta
from app.indexing.worker import DELETE, Change


def apply_changes(
    scope: str,
    changes: list[Change],
    cache: MetadataCache,
) -> dict:
    """Index/remove the files in one debounced watcher batch."""
    from app.capabilities.registry import sync_outlines
    from app.ingestion.folder import file_hash, index_paths, remove_paths

    root = normalize_root(scope)
    root_path = Path(root)

    upserts: dict[str, Change] = {}
    deletes: dict[str, Change] = {}
    for change in changes:
        if change.action == DELETE:
            deletes[change.rel_path] = change
        else:
            upserts[change.rel_path] = change
    # If the same path was deleted and re-created in one batch, upsert wins.
    for rel_path in list(deletes):
        if rel_path in upserts:
            deletes.pop(rel_path)

    stats = {
        "indexed": 0,
        "deleted": 0,
        "skipped": 0,
        "chunks": 0,
        "embeddings": 0,
        "errors": [],
    }

    if deletes:
        paths = sorted(deletes)
        try:
            remove_paths(paths)
            cache.remove(root, paths)
            sync_outlines(root, {}, {}, paths)
            stats["deleted"] = len(paths)
        except Exception as error:
            stats["errors"].append(f"delete: {error}")

    for rel_path in sorted(upserts):
        meta = file_meta(root_path, rel_path)
        if meta is None:
            # Deleted between the event and the batch: treat as a removal.
            try:
                remove_paths([rel_path])
                cache.remove(root, [rel_path])
                sync_outlines(root, {}, {}, [rel_path])
                stats["deleted"] += 1
            except Exception as error:
                stats["errors"].append(f"{rel_path}: {error}")
            continue

        try:
            row = cache.get(root, rel_path)
            digest: str | None = None
            if row:
                # mtime/size changed but the bytes may be identical (touch/save).
                digest = file_hash(root_path / rel_path)
                if (
                    row.get("content_hash") == digest
                    and row.get("status") == "indexed"
                    and row.get("index_version") == INDEX_VERSION
                ):
                    cache.mark_seen(root, meta, digest)
                    stats["skipped"] += 1
                    continue

            chunks, embeddings = index_paths(root_path, [rel_path])
            if digest is None:
                digest = file_hash(root_path / rel_path)
            cache.mark_indexed(root, meta, digest, len(chunks), INDEX_VERSION)
            if chunks:
                sync_outlines(root, {rel_path: digest}, {rel_path: chunks}, [])
            stats["indexed"] += 1
            stats["chunks"] += len(chunks)
            stats["embeddings"] += embeddings
        except Exception as error:
            cache.mark_error(root, meta, rel_path, str(error), INDEX_VERSION)
            stats["errors"].append(f"{rel_path}: {error}")

    return stats
