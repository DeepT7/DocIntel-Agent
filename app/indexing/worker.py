"""Mechanism 2b — debounced background index queue.

The watcher can fire dozens of events while a file is being saved. ``submits``
are stored as a set keyed by relative path (last action wins), then a single
background thread waits until the folder stays quiet for ``debounce_seconds``
before handing the batch to the indexer. The HTTP request that caused the change
never waits for embeddings.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

UPSERT = "upsert"
DELETE = "delete"

# Never starve: if events keep arriving, process the batch anyway after this
# multiple of the debounce window.
_MAX_SETTLE_MULTIPLIER = 5


@dataclass(frozen=True, slots=True)
class Change:
    """One file-level change requested by the watcher or a full scan."""

    action: str
    rel_path: str

    def __post_init__(self) -> None:
        if self.action not in (UPSERT, DELETE):
            raise ValueError(f"unknown change action: {self.action!r}")


Handler = Callable[[str, list[Change]], object]


class BackgroundIndexer:
    """Coalesce changes per scope and run the handler on one worker thread."""

    def __init__(
        self,
        handler: Handler,
        *,
        debounce_seconds: float = 1.0,
        poll_seconds: float = 0.25,
    ):
        self._handler = handler
        self._debounce = max(0.0, debounce_seconds)
        self._max_settle = max(self._debounce * _MAX_SETTLE_MULTIPLIER, self._debounce)
        self._poll = max(0.05, poll_seconds)
        self._cond = threading.Condition()
        self._pending: dict[str, dict[str, Change]] = {}
        self._running = True
        self._busy = False
        self._last_submit = 0.0
        self._processed = 0
        self._errors: list[str] = []
        self._thread = threading.Thread(
            target=self._run, name="docintel-index-worker", daemon=True
        )
        self._thread.start()

    # ---------------------------------------------------------------- public
    def submit(self, scope: str, change: Change) -> None:
        with self._cond:
            self._pending.setdefault(scope, {})[change.rel_path] = change
            self._last_submit = time.monotonic()
            self._cond.notify_all()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until the queue is empty and no batch is being processed."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._pending or self._busy:
                if deadline is None:
                    self._cond.wait(self._poll)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def stop(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        self._thread.join(timeout)

    def status(self) -> dict:
        with self._cond:
            return {
                "running": self._running,
                "busy": self._busy,
                "pending": sum(len(bucket) for bucket in self._pending.values()),
                "processed": self._processed,
                "errors": list(self._errors[-5:]),
            }

    # --------------------------------------------------------------- worker
    def _run(self) -> None:
        while True:
            with self._cond:
                while self._running and not self._pending:
                    self._cond.wait(self._poll)
                if not self._running:
                    return
                first_seen = time.monotonic()
                while self._running:
                    elapsed = time.monotonic() - first_seen
                    quiet_for = time.monotonic() - self._last_submit
                    settle = min(self._debounce - quiet_for, self._max_settle - elapsed)
                    if settle <= 0:
                        break
                    self._cond.wait(settle)
                if not self._running:
                    return
                batch = self._pending
                self._pending = {}
                self._busy = True

            for scope, changes in batch.items():
                try:
                    self._handler(scope, list(changes.values()))
                    with self._cond:
                        self._processed += len(changes)
                except Exception as error:  # keep the worker alive
                    with self._cond:
                        self._errors.append(f"{scope}: {error}")

            with self._cond:
                self._busy = False
                self._cond.notify_all()
