"""Mechanism 2a — SQLite metadata cache (file state).

One row per (scope, relative path) answers the question *"is this file already
indexed, and if not, why?"* without touching file contents or Qdrant:

* ``mtime_ns`` + ``size`` make the common "nothing changed" case a cheap stat
  comparison (a hash is only computed for the rare "maybe changed" file);
* ``content_hash`` is the exact identity used by the outline/card cache;
* ``status`` (``pending`` / ``indexed`` / ``error``) lets the watcher retry a
  failed file on the next event (deleted files are removed from the cache);
* ``index_version`` forces a re-index when the chunker changes.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from app.core.config import INDEX_DIR
from app.indexing.scanner import FileMeta

DEFAULT_DB_PATH = INDEX_DIR / "metadata.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS file_state (
    scope          TEXT    NOT NULL,
    rel_path       TEXT    NOT NULL,
    abs_path       TEXT    NOT NULL,
    name           TEXT    NOT NULL,
    ext            TEXT    NOT NULL DEFAULT '',
    size           INTEGER NOT NULL,
    mtime_ns       INTEGER NOT NULL,
    content_hash   TEXT,
    status         TEXT    NOT NULL DEFAULT 'pending',
    chunk_count    INTEGER NOT NULL DEFAULT 0,
    index_version  TEXT    NOT NULL DEFAULT '',
    error          TEXT,
    first_seen_at  REAL    NOT NULL,
    updated_at     REAL    NOT NULL,
    indexed_at     REAL,
    PRIMARY KEY (scope, rel_path)
);
CREATE INDEX IF NOT EXISTS idx_file_state_scope_status
    ON file_state(scope, status);
CREATE INDEX IF NOT EXISTS idx_file_state_scope_mtime
    ON file_state(scope, mtime_ns);

CREATE TABLE IF NOT EXISTS scope_state (
    scope         TEXT PRIMARY KEY,
    watcher       TEXT NOT NULL DEFAULT 'stopped',
    last_scan_at  REAL,
    last_sync_at  REAL,
    stats         TEXT
);
"""


@dataclass
class Diff:
    """Files that need work after comparing disk metadata with the cache."""

    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: int = 0

    @property
    def changed(self) -> list[str]:
        return [*self.added, *self.modified]


class MetadataCache:
    """Thread-safe access to the file-state database (one connection per call)."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self._lock = threading.RLock()
        self._ensure_schema()

    # ------------------------------------------------------------------ setup
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @contextmanager
    def _db(self):
        """One connection per operation, closed afterwards (WAL, thread-safe)."""
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ read
    def get(self, scope: str, rel_path: str) -> dict | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM file_state WHERE scope = ? AND rel_path = ?",
                (scope, rel_path),
            ).fetchone()
        return dict(row) if row else None

    def rows(self, scope: str) -> dict[str, dict]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM file_state WHERE scope = ?", (scope,)
            ).fetchall()
        return {row["rel_path"]: dict(row) for row in rows}

    def counts(self, scope: str) -> dict:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM file_state "
                "WHERE scope = ? GROUP BY status",
                (scope,),
            ).fetchall()
        by_status = {row["status"]: row["n"] for row in rows}
        by_status["total"] = sum(by_status.values())
        return by_status

    # ----------------------------------------------------------------- write
    def upsert(
        self,
        scope: str,
        meta: FileMeta,
        *,
        content_hash: str | None = None,
        status: str = "pending",
        chunk_count: int = 0,
        index_version: str = "",
        error: str | None = None,
    ) -> None:
        now = time.time()
        with self._db() as conn:
            conn.execute(
                """
                INSERT INTO file_state (
                    scope, rel_path, abs_path, name, ext, size, mtime_ns,
                    content_hash, status, chunk_count, index_version, error,
                    first_seen_at, updated_at, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(scope, rel_path) DO UPDATE SET
                    abs_path = excluded.abs_path,
                    name = excluded.name,
                    ext = excluded.ext,
                    size = excluded.size,
                    mtime_ns = excluded.mtime_ns,
                    status = excluded.status,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    scope,
                    meta.rel_path,
                    meta.abs_path,
                    meta.name,
                    meta.ext,
                    meta.size,
                    meta.mtime_ns,
                    content_hash,
                    status,
                    chunk_count,
                    index_version,
                    error,
                    now,
                    now,
                ),
            )

    def mark_indexed(
        self,
        scope: str,
        meta: FileMeta,
        content_hash: str,
        chunk_count: int,
        index_version: str,
    ) -> None:
        self.mark_indexed_many(scope, [(meta, content_hash, chunk_count, index_version)])

    def mark_indexed_many(
        self,
        scope: str,
        records: Iterable[tuple[FileMeta, str, int, str]],
    ) -> None:
        """Mark many files indexed in one transaction (first full sync)."""
        items = list(records)
        if not items:
            return
        now = time.time()
        with self._db() as conn:
            for meta, content_hash, chunk_count, index_version in items:
                conn.execute(
                    """
                    INSERT INTO file_state (
                        scope, rel_path, abs_path, name, ext, size, mtime_ns,
                        content_hash, status, chunk_count, index_version, error,
                        first_seen_at, updated_at, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'indexed', ?, ?, NULL, ?, ?, ?)
                    ON CONFLICT(scope, rel_path) DO UPDATE SET
                        abs_path = excluded.abs_path,
                        name = excluded.name,
                        ext = excluded.ext,
                        size = excluded.size,
                        mtime_ns = excluded.mtime_ns,
                        content_hash = excluded.content_hash,
                        status = 'indexed',
                        chunk_count = excluded.chunk_count,
                        index_version = excluded.index_version,
                        error = NULL,
                        updated_at = excluded.updated_at,
                        indexed_at = excluded.indexed_at
                    """,
                    (
                        scope,
                        meta.rel_path,
                        meta.abs_path,
                        meta.name,
                        meta.ext,
                        meta.size,
                        meta.mtime_ns,
                        content_hash,
                        chunk_count,
                        index_version,
                        now,
                        now,
                        now,
                    ),
                )

    def mark_seen(self, scope: str, meta: FileMeta, content_hash: str) -> None:
        """Same content, new mtime/size (e.g. ``touch``): avoid re-indexing."""
        with self._db() as conn:
            conn.execute(
                "UPDATE file_state SET size = ?, mtime_ns = ?, content_hash = ?, "
                "status = 'indexed', error = NULL, updated_at = ? "
                "WHERE scope = ? AND rel_path = ?",
                (
                    meta.size,
                    meta.mtime_ns,
                    content_hash,
                    time.time(),
                    scope,
                    meta.rel_path,
                ),
            )

    def mark_error(
        self,
        scope: str,
        meta: FileMeta | None,
        rel_path: str,
        message: str,
        index_version: str = "",
    ) -> None:
        if meta is not None:
            self.upsert(
                scope,
                meta,
                status="error",
                index_version=index_version,
                error=message,
            )
            return
        now = time.time()
        with self._db() as conn:
            conn.execute(
                "UPDATE file_state SET status = 'error', error = ?, updated_at = ? "
                "WHERE scope = ? AND rel_path = ?",
                (message, now, scope, rel_path),
            )

    def remove(self, scope: str, rel_paths: Iterable[str]) -> int:
        paths = list(dict.fromkeys(rel_paths))
        if not paths:
            return 0
        with self._db() as conn:
            removed = 0
            for rel_path in paths:
                cursor = conn.execute(
                    "DELETE FROM file_state WHERE scope = ? AND rel_path = ?",
                    (scope, rel_path),
                )
                removed += cursor.rowcount
        return removed

    def clear(self) -> None:
        with self._db() as conn:
            conn.execute("DELETE FROM file_state")
            conn.execute("DELETE FROM scope_state")

    # ------------------------------------------------------------------ diff
    def diff(
        self,
        scope: str,
        files: dict[str, FileMeta],
        index_version: str,
        *,
        force: bool = False,
    ) -> Diff:
        """Compare a metadata snapshot with the cached state.

        A file is *modified* when its size/mtime changed, when its last indexing
        failed, or when it was indexed with an older chunker version. The caller
        may confirm with a content hash before paying for extraction.
        """
        cached = self.rows(scope)
        result = Diff()
        for rel_path, meta in files.items():
            row = cached.get(rel_path)
            if row is None:
                result.added.append(rel_path)
            elif (
                force
                or row["size"] != meta.size
                or row["mtime_ns"] != meta.mtime_ns
                or row["index_version"] != index_version
                or row["status"] in {"pending", "error"}
            ):
                result.modified.append(rel_path)
            else:
                result.unchanged += 1
        result.deleted = [rel_path for rel_path in cached if rel_path not in files]
        return result

    # ----------------------------------------------------------- scope state
    def set_scope_state(
        self,
        scope: str,
        *,
        watcher: str | None = None,
        scanned: bool = False,
        synced: bool = False,
        stats: str | None = None,
    ) -> None:
        now = time.time()
        with self._db() as conn:
            conn.execute(
                "INSERT INTO scope_state (scope, watcher) VALUES (?, 'stopped') "
                "ON CONFLICT(scope) DO NOTHING",
                (scope,),
            )
            if watcher is not None:
                conn.execute(
                    "UPDATE scope_state SET watcher = ? WHERE scope = ?",
                    (watcher, scope),
                )
            if scanned:
                conn.execute(
                    "UPDATE scope_state SET last_scan_at = ? WHERE scope = ?",
                    (now, scope),
                )
            if synced:
                conn.execute(
                    "UPDATE scope_state SET last_sync_at = ? WHERE scope = ?",
                    (now, scope),
                )
            if stats is not None:
                conn.execute(
                    "UPDATE scope_state SET stats = ? WHERE scope = ?",
                    (stats, scope),
                )

    def scope_state(self, scope: str) -> dict | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM scope_state WHERE scope = ?", (scope,)
            ).fetchone()
        return dict(row) if row else None


_CACHE: MetadataCache | None = None
_CACHE_LOCK = threading.Lock()


def get_cache() -> MetadataCache:
    """Process-wide cache instance (one SQLite file, WAL mode)."""
    global _CACHE
    if _CACHE is None:
        with _CACHE_LOCK:
            if _CACHE is None:
                _CACHE = MetadataCache()
    return _CACHE
