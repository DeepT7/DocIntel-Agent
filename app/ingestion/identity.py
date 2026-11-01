"""Stable identifiers for files and chunks.

Every layer of ingestion derives ids from the same inputs, so they live here
instead of being duplicated across the chunker, the index writer and the
capability layer:

- ``file_hash``  content digest used for change detection;
- ``file_id_for`` per-path id that groups a file's points in Qdrant;
- ``point_id_for`` deterministic point id (uuid5 of the chunk id), so
  re-indexing the same chunk overwrites instead of duplicating.
"""

import hashlib
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_id_for(rel_path: str) -> str:
    return hashlib.md5(rel_path.encode("utf-8")).hexdigest()


def point_id_for(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, chunk_id))
