"""Mechanism 3 — real-time file-system watcher.

Prefers ``watchdog`` OS events (``on_created``, ``on_modified``, ``on_deleted``,
``on_moved``), debounced per path. When watchdog is not installed (or cannot
watch the path, e.g. some network shares), it falls back to a lightweight
metadata poller that only ``stat``s files — it never reads contents.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from app.core.config import SUPPORTED_EXTENSIONS
from app.indexing.scanner import snapshot
from app.indexing.worker import DELETE, UPSERT, Change

try:  # optional dependency: app still works with the polling fallback
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    _HAS_WATCHDOG = True
except ImportError:  # pragma: no cover - exercised on installs without watchdog
    FileSystemEventHandler = object  # type: ignore[assignment,misc]
    Observer = None  # type: ignore[assignment]
    _HAS_WATCHDOG = False


Schedule = Callable[[str, Change], None]
Emitter = Callable[[str, Change], None]


class _Debouncer:
    """One timer per path: a burst of writes fires the handler once."""

    def __init__(self, delay: float, callback: Callable[[Change], None]):
        self._delay = max(0.05, delay)
        self._callback = callback
        self._lock = threading.Lock()
        self._timers: dict[str, threading.Timer] = {}

    def schedule(self, key: str, change: Change) -> None:
        with self._lock:
            previous = self._timers.pop(key, None)
            if previous is not None:
                previous.cancel()
            timer = threading.Timer(self._delay, self._fire, args=(key, change))
            timer.daemon = True
            self._timers[key] = timer
            timer.start()

    def _fire(self, key: str, change: Change) -> None:
        with self._lock:
            self._timers.pop(key, None)
        try:
            self._callback(change)
        except Exception:  # a watcher callback must never kill the timer thread
            pass

    def cancel_all(self) -> None:
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for timer in timers:
            timer.cancel()


def _is_supported(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_EXTENSIONS


def _is_hidden(rel_path: str) -> bool:
    return any(part.startswith(".") for part in rel_path.split("/"))


if _HAS_WATCHDOG:

    class _WatchdogHandler(FileSystemEventHandler):  # type: ignore[misc]
        """Translate watchdog events into debounced upsert/delete changes."""

        def __init__(self, root: Path, schedule: Schedule):
            self._root = root
            self._schedule = schedule

        def _rel(self, raw_path: str) -> str | None:
            try:
                rel = Path(raw_path).relative_to(self._root).as_posix()
            except ValueError:
                return None
            if not rel or _is_hidden(rel) or not _is_supported(rel):
                return None
            return rel

        def on_created(self, event) -> None:
            if event.is_directory:
                return
            rel = self._rel(event.src_path)
            if rel:
                self._schedule(rel, Change(UPSERT, rel))

        def on_modified(self, event) -> None:
            if event.is_directory:
                return
            rel = self._rel(event.src_path)
            if rel:
                self._schedule(rel, Change(UPSERT, rel))

        def on_deleted(self, event) -> None:
            if event.is_directory:
                return
            rel = self._rel(event.src_path)
            if rel:
                self._schedule(rel, Change(DELETE, rel))

        def on_moved(self, event) -> None:
            if event.is_directory:
                return
            source = self._rel(event.src_path)
            if source:
                self._schedule(source, Change(DELETE, source))
            destination = self._rel(event.dest_path)
            if destination:
                self._schedule(destination, Change(UPSERT, destination))


class FolderWatcher:
    """Watch one folder; every event is coalesced before reaching ``on_change``."""

    def __init__(
        self,
        on_change: Emitter,
        *,
        debounce_seconds: float = 1.0,
        poll_seconds: float = 4.0,
    ):
        self._on_change = on_change
        self._debounce_seconds = debounce_seconds
        self._poll_seconds = max(0.5, poll_seconds)
        self._lock = threading.Lock()
        self._mode = "stopped"
        self._scope: str | None = None
        self._error: str | None = None
        self._observer = None
        self._debouncer: _Debouncer | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- public
    def start(self, scope: str) -> None:
        self.stop()
        root = Path(scope)
        if not root.is_dir():
            self._mode = "stopped"
            self._error = f"folder not found: {root.as_posix()}"
            self._scope = str(scope)
            return
        with self._lock:
            self._scope = str(scope)
            self._error = None
            self._debouncer = _Debouncer(
                self._debounce_seconds,
                lambda change: self._on_change(scope, change),
            )

        if _HAS_WATCHDOG:
            try:
                observer = Observer()
                observer.schedule(
                    _WatchdogHandler(root, self._debouncer.schedule),
                    str(root),
                    recursive=True,
                )
                observer.daemon = True
                observer.start()
                self._observer = observer
                self._mode = "watchdog"
                return
            except Exception as error:  # permission, inotify limits, ...
                with self._lock:
                    self._error = str(error)

        self._start_polling(root)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            observer, self._observer = self._observer, None
            debouncer, self._debouncer = self._debouncer, None
            thread, self._thread = self._thread, None
        if debouncer is not None:
            debouncer.cancel_all()
        if observer is not None:
            try:
                observer.stop()
                observer.join(timeout=5)
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        self._mode = "stopped"

    def status(self) -> dict:
        with self._lock:
            return {
                "mode": self._mode,
                "scope": self._scope,
                "error": self._error,
                "watchdog": _HAS_WATCHDOG,
            }

    # -------------------------------------------------------------- polling
    def _start_polling(self, root: Path) -> None:
        self._stop.clear()
        self._mode = "polling"
        self._thread = threading.Thread(
            target=self._poll_loop,
            args=(root,),
            name="docintel-watch-poll",
            daemon=True,
        )
        self._thread.start()

    def _poll_loop(self, root: Path) -> None:
        if self._stop.is_set():
            return
        previous = snapshot(root)
        while not self._stop.wait(self._poll_seconds):
            current = snapshot(root)
            debouncer = self._debouncer
            if debouncer is None:
                return
            for rel_path, meta in current.items():
                old = previous.get(rel_path)
                if old is None or old.mtime_ns != meta.mtime_ns or old.size != meta.size:
                    debouncer.schedule(rel_path, Change(UPSERT, rel_path))
            for rel_path in set(previous) - set(current):
                debouncer.schedule(rel_path, Change(DELETE, rel_path))
            previous = current
