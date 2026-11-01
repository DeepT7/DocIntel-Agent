"""Folder-based knowledge base: ingest local documents into Qdrant incrementally.

This module is the public entry point for folder ingestion. The implementation
is split by responsibility:

- ``identity``     stable file ids, point ids and content hashes;
- ``chunking``     blocks -> chunk records (section tree, captions, splitting);
- ``qdrant_index`` collection schema, embedding cache, upsert/delete, pruning;
- ``sync``         metadata-first incremental folder sync.

The names below are re-exported so existing callers
(``from app.ingestion.folder import file_hash, index_paths, ...``) and
``python -m app.ingestion.folder`` keep working unchanged.

Design
------
- The **file-state cache** in ``app.indexing.metadata`` records size, mtime and a
  content hash per file, so the next sync knows which files were added,
  modified or deleted.
- Only changed files are re-chunked and re-embedded. Embeddings are cached by
  exact chunk text, so unchanged text is reused instead of re-sent to the API.
- Every chunk gets a **deterministic point id** (uuid5 of its chunk_id) and the
  file's chunks share a ``file_id``. On modify/delete we drop all points for that
  ``file_id`` first and then re-add the new ones, so nothing goes stale.

Result: adding/editing one note re-embeds only the chunks that actually changed.
"""

import argparse
from pathlib import Path

from app.core.config import DOCS_DIR
from app.ingestion.chunking import build_chunks_for_file
from app.ingestion.identity import file_hash, file_id_for, point_id_for
from app.ingestion.qdrant_index import (
    BATCH_SIZE,
    CACHE_PATH,
    INDEX_LOCK,
    cache_line_count,
    index_paths,
    prune_cache,
    remove_paths,
)
from app.ingestion.sync import sync_folder

__all__ = [
    "BATCH_SIZE",
    "CACHE_PATH",
    "INDEX_LOCK",
    "build_chunks_for_file",
    "cache_line_count",
    "file_hash",
    "file_id_for",
    "index_paths",
    "point_id_for",
    "prune_cache",
    "remove_paths",
    "sync_folder",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync a local document folder into Qdrant.")
    parser.add_argument("--root", default=str(DOCS_DIR), help="Folder to ingest")
    parser.add_argument("--reset", action="store_true", help="Rebuild the collection from scratch")
    parser.add_argument("--prune-cache", action="store_true", help="Drop cache entries not in the index")
    parser.add_argument("--no-prune", action="store_true", help="Disable the automatic cache cleanup")
    args = parser.parse_args()

    if args.prune_cache:
        print("prune:", prune_cache())
    else:
        print("sync:", sync_folder(Path(args.root), reset=args.reset, auto_prune=not args.no_prune))


if __name__ == "__main__":
    main()