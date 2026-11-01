"""Facade that wires the four mechanisms into one process-wide service.

* ``start(scope)``   — (re)start the watcher for a folder;
* ``sync(scope)``    — explicit full/incremental scan (``/api/sync``);
* ``tree(scope)``    — metadata-only folder tree (mechanism 1);
* ``read(...)``      — on-demand raw content (mechanism 4);
* ``status()``       — watcher mode + queue depth + cache counters.
"""

from __future__ import annotations

import json
import threading

from app.core.config import (
    WATCH_DEBOUNCE_SECONDS,
    WATCH_ENABLED,
    WATCH_POLL_SECONDS,
)
from app.core.scope import normalize_root
from app.indexing.metadata import MetadataCache, get_cache
from app.indexing.watcher import FolderWatcher
from app.indexing.worker import BackgroundIndexer, Change


class IndexManager:
    def __init__(self, cache: MetadataCache | None = None):
        self.cache = cache or get_cache()
        self._lock = threading.RLock()
        self._scope: str | None = None
        self._worker = BackgroundIndexer(
            self._handle_batch,
            debounce_seconds=WATCH_DEBOUNCE_SECONDS,
        )
        self._watcher = FolderWatcher(
            self._emit,
            debounce_seconds=WATCH_DEBOUNCE_SECONDS,
            poll_seconds=WATCH_POLL_SECONDS,
        )

    # ----------------------------------------------------------- lifecycle
    def start(self, scope: str | None) -> dict:
        """Watch ``scope`` (stops the previous watcher first)."""
        if not scope:
            self.stop_watcher()
            return self.status()
        root = normalize_root(scope)
        with self._lock:
            if root == self._scope and self._watcher.status()["mode"] != "stopped":
                return self.status(root)
            self._scope = root
            if WATCH_ENABLED:
                self._watcher.start(root)
            self.cache.set_scope_state(
                root, watcher=self._watcher.status()["mode"]
            )
        return self.status(root)

    def stop_watcher(self) -> None:
        with self._lock:
            self._watcher.stop()
            self._scope = None

    def shutdown(self) -> None:
        """Stop the watcher and the background worker (server shutdown)."""
        self.stop_watcher()
        self._worker.stop()

    def ensure_watcher(self, scope: str | None) -> None:
        if scope:
            self.start(scope)

    # ------------------------------------------------------------- changes
    def _emit(self, scope: str, change: Change) -> None:
        # Ignore late events from a watcher that was already replaced.
        with self._lock:
            if self._scope != scope:
                return
        self._worker.submit(scope, change)

    def _handle_batch(self, scope: str, changes: list[Change]) -> dict:
        from app.indexing.pipeline import apply_changes

        stats = apply_changes(scope, changes, self.cache)
        self.cache.set_scope_state(
            scope,
            synced=True,
            stats=json.dumps({**stats, "source": "watcher"}, ensure_ascii=False),
        )
        return stats

    # ---------------------------------------------------------- read helpers
    def tree(self, scope: str) -> dict:
        from app.indexing.scanner import scan_tree

        return scan_tree(normalize_root(scope))

    def read(self, scope: str, rel_path: str, max_bytes: int | None = None) -> dict:
        from app.indexing.content import read_content

        kwargs = {} if max_bytes is None else {"max_bytes": max_bytes}
        return read_content(normalize_root(scope), rel_path, **kwargs)

    # --------------------------------------------------------------- status
    def status(self, scope: str | None = None) -> dict:
        target = normalize_root(scope) if scope else self._scope
        watcher = self._watcher.status()
        payload = {
            "scope": target,
            "watcher": watcher,
            "worker": self._worker.status(),
            "watch_enabled": WATCH_ENABLED,
        }
        if target:
            payload["files"] = self.cache.counts(target)
            payload["state"] = self.cache.scope_state(target)
        return payload


_MANAGER: IndexManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> IndexManager:
    global _MANAGER
    if _MANAGER is None:
        with _MANAGER_LOCK:
            if _MANAGER is None:
                _MANAGER = IndexManager()
    return _MANAGER
