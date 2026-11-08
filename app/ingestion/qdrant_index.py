"""Qdrant collection management and the single-file write path.

The full sync, the background watcher worker and the HTTP layer all use
``index_paths`` / ``remove_paths``, so a file is indexed exactly the same way no
matter what triggered it. Embeddings are cached by exact chunk text, so
unchanged text is reused instead of being re-sent to the embedding API.
"""

import functools
import json
import threading
from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import (
    DENSE_VECTOR_NAME,
    EMBED_BATCH_SIZE,
    EMBEDDING_DIM,
    INDEX_DIR,
    QDRANT_COLLECTION,
    SCROLL_PAGE_SIZE,
    SPARSE_MODEL,
    SPARSE_VECTOR_NAME,
    UPSERT_BATCH_SIZE,
)
from app.ingestion.chunking import build_chunks_for_file
from app.ingestion.identity import file_id_for, point_id_for
from app.ingestion.raw_assets import remove_raw_assets
from app.retrieval.embeddings import embed_texts, load_cached_embeddings, record_model
from app.retrieval.store import get_client

CACHE_PATH = INDEX_DIR / "embeddings_cache.jsonl"

# Re-exported under the original name: the canonical value lives in config.
BATCH_SIZE = EMBED_BATCH_SIZE

# One writer at a time: the watcher worker and an explicit /api/sync can overlap,
# and the embedded Qdrant client is shared by the whole process. RLock so
# sync_folder -> index_paths / remove_paths can nest.
INDEX_LOCK = threading.RLock()


def _serialized(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with INDEX_LOCK:
            return fn(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------- #
# Collection schema
# --------------------------------------------------------------------------- #
def _ensure_collection(client: QdrantClient, reset: bool = False) -> bool:
    """Ensure the collection exists with the current schema.

    Returns True when the collection was (re)created, so the caller can force a
    full re-index (the stored points are gone).
    """
    collections = [c.name for c in client.get_collections().collections]
    exists = QDRANT_COLLECTION in collections
    recreated = False

    if exists and reset:
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        exists = False

    if exists:
        params = client.get_collection(QDRANT_COLLECTION).config.params
        dense = params.vectors if isinstance(params.vectors, dict) else {}
        sparse = params.sparse_vectors if isinstance(params.sparse_vectors, dict) else {}
        dense_ok = DENSE_VECTOR_NAME in dense and dense[DENSE_VECTOR_NAME].size == EMBEDDING_DIM
        sparse_ok = SPARSE_VECTOR_NAME in sparse
        if not (dense_ok and sparse_ok):
            # A dense-dim change (new embedding model) or a missing sparse vector
            # (upgrade to native hybrid) invalidates the stored points, so rebuild.
            print("collection schema changed (dense dim / sparse vector); recreating collection")
            client.delete_collection(collection_name=QDRANT_COLLECTION)
            exists = False

    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=EMBEDDING_DIM, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        recreated = True
    return recreated


def _delete_file_points(client: QdrantClient, file_id: str) -> None:
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.Filter(
            must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))]
        ),
    )


# --------------------------------------------------------------------------- #
# Chunk -> point
# --------------------------------------------------------------------------- #
def _chunk_payload(chunk: dict) -> dict:
    """The payload stored with a point (retrieval reads these fields)."""
    return {
        "chunk_id": chunk["chunk_id"],
        "file_id": chunk["file_id"],
        "root": chunk["root"],
        "path": chunk["path"],
        "url": chunk["url"],
        "title": chunk["title"],
        "file_name": chunk.get("file_name"),
        "section_heading": chunk["section_heading"],
        "section_level": chunk.get("section_level", 1),
        "section_depth": chunk.get("section_depth", 1),
        "breadcrumbs": chunk.get("breadcrumbs", []),
        "breadcrumb_levels": chunk.get("breadcrumb_levels", []),
        "section_id": chunk["section_id"],
        "section_ancestors": chunk.get("section_ancestors", []),
        "section_titles": chunk.get("section_titles", []),
        "section_depths": chunk.get("section_depths", []),
        "section_numbers": chunk.get("section_numbers", []),
        "section_title_clean": chunk.get("section_title_clean"),
        "section_number": chunk.get("section_number", ""),
        "raw_heading": chunk.get("raw_heading"),
        "chunk_type": chunk.get("chunk_type", "text"),
        "context_header": chunk.get("context_header"),
        "doc_type": chunk["doc_type"],
        "char_count": chunk["char_count"],
        "text": chunk["text"],
        "caption": chunk.get("caption"),
        "table_number": chunk.get("table_number"),
        "asset_id": chunk.get("asset_id"),
        "asset_path": chunk.get("asset_path"),
        "headers": chunk.get("headers"),
        "row_count": chunk.get("row_count"),
    }


def _chunk_point(chunk: dict, embeddings: dict[str, list[float]]) -> models.PointStruct:
    """A point carrying both the dense and the BM25 sparse vector.

    Qdrant's client encodes the sparse vector from the text (via fastembed); the
    collection applies IDF when scoring.
    """
    return models.PointStruct(
        id=point_id_for(chunk["chunk_id"]),
        vector={
            DENSE_VECTOR_NAME: embeddings[chunk["text"]],
            SPARSE_VECTOR_NAME: models.Document(text=chunk["text"], model=SPARSE_MODEL),
        },
        payload=_chunk_payload(chunk),
    )


# --------------------------------------------------------------------------- #
# Embed + upsert
# --------------------------------------------------------------------------- #
def _embed_chunks(chunks: list[dict]) -> int:
    """Embed chunks whose text is not cached yet, then upsert all of them."""
    if not chunks:
        return 0

    model = record_model()
    embeddings = load_cached_embeddings(CACHE_PATH, model)
    pending = [chunk for chunk in chunks if chunk["text"] not in embeddings]

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            vectors = embed_texts([chunk["text"] for chunk in batch])
            for chunk, vector in zip(batch, vectors):
                embeddings[chunk["text"]] = vector
                handle.write(
                    json.dumps(
                        {"model": model, "text": chunk["text"], "embedding": vector},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    points = [_chunk_point(chunk, embeddings) for chunk in chunks]
    client = get_client()
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points[start : start + UPSERT_BATCH_SIZE],
        )
    return len(pending)


# --------------------------------------------------------------------------- #
# Public single-file operations
# --------------------------------------------------------------------------- #
@_serialized
def remove_paths(rel_paths: Iterable[str]) -> int:
    """Drop every vector that belongs to the given relative file paths."""
    paths = list(dict.fromkeys(rel_paths))
    if not paths:
        return 0
    client = get_client()
    _ensure_collection(client)
    for rel_path in paths:
        _delete_file_points(client, file_id_for(rel_path))
        remove_raw_assets(rel_path)
    return len(paths)


@_serialized
def index_paths(root: Path, rel_paths: Iterable[str]) -> tuple[list[dict], int]:
    """Index specific files (delete stale points, chunk, embed, upsert).

    Returns ``(chunks, embeddings_computed)``. This is the single-file entry
    point used by the watcher; it never scans the rest of the folder.
    """
    root = Path(root)
    client = get_client()
    _ensure_collection(client)

    chunks: list[dict] = []
    for rel_path in dict.fromkeys(rel_paths):
        path = root / rel_path
        if not path.is_file():
            continue
        _delete_file_points(client, file_id_for(rel_path))
        chunks.extend(build_chunks_for_file(root, path))

    embeddings_computed = _embed_chunks(chunks)
    return chunks, embeddings_computed


# --------------------------------------------------------------------------- #
# Embedding-cache maintenance
# --------------------------------------------------------------------------- #
def prune_cache() -> dict:
    """Drop embedding-cache entries whose text is no longer present in the index.

    The cache is keyed by text (not by file) and is append-only, so after files
    are deleted or their text changes, old entries linger. This removes the ones
    the current index no longer references.
    """
    client = get_client()
    live_texts: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            text = (point.payload or {}).get("text")
            if text:
                live_texts.add(text)
        if offset is None:
            break

    if not CACHE_PATH.exists():
        return {"cache_before": 0, "cache_after": 0, "removed": 0}

    kept_lines: list[str] = []
    before = 0
    with CACHE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            before += 1
            if json.loads(line).get("text") in live_texts:
                kept_lines.append(line)
    CACHE_PATH.write_text("".join(kept_lines), encoding="utf-8")
    return {
        "cache_before": before,
        "cache_after": len(kept_lines),
        "removed": before - len(kept_lines),
    }


def cache_line_count() -> int:
    if not CACHE_PATH.exists():
        return 0
    with CACHE_PATH.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)
