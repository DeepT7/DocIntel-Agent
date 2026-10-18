"""Shared Qdrant client.

Embedded Qdrant (path mode) allows only one client per storage folder, so the
whole process must share a single instance. Retrieval and folder syncing both go
through ``get_client`` to avoid "storage already accessed" errors.

Why the error happens
---------------------
Embedded Qdrant keeps an advisory file lock (``output/qdrant_db/.lock``) while a
client is open. A second process that tries to open the same folder gets::

    RuntimeError: Storage folder ... is already accessed by another instance

This is by design: embedded mode has no server, so two writers to the same files
would corrupt the store. The usual causes are:

* the API server is already running (so the CLI/eval cannot open the DB), or
* ``uvicorn --reload`` starts a new worker before the old one released the lock, or
* a previous process crashed on a filesystem that does not release advisory locks
  (e.g. a Windows drive mounted under WSL) and left a stale lock behind.

What this module does about it
------------------------------
* retries briefly, so the short hand-over window of a reload does not fail;
* records the owner pid in ``output/qdrant_db/.docintel.pid`` so it can both
  name the holder and tell a *live* lock from a *stale* one;
* clears the lock automatically when the owning process is dead;
* otherwise raises a short, actionable message instead of the raw traceback.

For true concurrent access (server + CLI + eval at the same time), run a Qdrant
server and set ``QDRANT_URL`` (see ``app/core/config.py``).

Run ``python -m app.retrieval.store`` to see who holds the lock, or
``python -m app.retrieval.store --stop`` to stop that process.
"""

import atexit
import os
import sys
import threading
import time
from pathlib import Path

from qdrant_client import QdrantClient

from app.core.config import QDRANT_COLLECTION, QDRANT_PATH, QDRANT_URL

COLLECTION = QDRANT_COLLECTION

_client: QdrantClient | None = None

# FastAPI runs sync endpoints in a thread pool, so several requests may call
# ``get_client`` concurrently. Without this lock they all see ``_client is None``,
# one wins the embedded lock and the rest fail with "already accessed ... (in this
# process)". Double-checked locking makes creation atomic.
_client_lock = threading.Lock()

# Marker written while THIS process holds the embedded storage lock. Qdrant's own
# ``.lock`` holds an anonymous advisory lock with no owner info, so this file is
# how we name the holder and tell a stale lock from a live one.
_PID_FILE = QDRANT_PATH / ".docintel.pid"

# A reload or a quick restart may take a moment to release the lock; retry a few
# times before declaring a real conflict.
_OPEN_RETRIES = 5
_OPEN_RETRY_DELAY = 0.6


def _is_locked_error(error: BaseException) -> bool:
    return "already accessed by another instance" in str(error)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists but owned by someone else
        return True
    except OSError:
        return False
    return True


def _read_marker_pid() -> int | None:
    try:
        return int(_PID_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _lock_path() -> Path:
    return QDRANT_PATH / ".lock"


def _find_lock_holder() -> int | None:
    """Best-effort: the pid of the process holding ``.lock`` (Linux /proc scan)."""
    target = str(_lock_path().resolve())
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(f"{fd_dir}/{fd}") == target:
                    return int(entry)
            except OSError:
                continue
    return None


def _holder_pid() -> int | None:
    """Who currently holds the embedded lock, if we can tell."""
    marker = _read_marker_pid()
    if marker is not None and _pid_alive(marker):
        return marker
    return _find_lock_holder()


def _clear_stale_lock() -> None:
    """Remove the leftover lock + marker of a dead owner (unclean exit).

    Only ever called after confirming the recorded owner is gone, so a live
    client's lock is never removed (which would let two writers corrupt the DB).
    """
    for path in (_lock_path(), _PID_FILE):
        try:
            path.unlink()
        except OSError:
            pass


def _locked_message(last_error: BaseException | None) -> RuntimeError:
    holder = _holder_pid()
    if holder and holder == os.getpid():
        # Another thread in this process already opened it; get_client() should
        # normally return that instance, so this only happens on misuse.
        detail = "already held by this process"
        hint = "use app.retrieval.store.get_client() instead of creating a client."
    elif holder:
        detail = f"(held by pid {holder})"
        hint = (
            "Embedded mode allows only one process at a time, so stop the other "
            f"process first — e.g. `kill {holder}` or "
            "`python -m app.retrieval.store --stop` — then retry."
        )
    else:
        detail = ""
        hint = (
            "Embedded mode allows only one process at a time; stop the other "
            "process and retry."
        )
    return RuntimeError(
        f"Embedded Qdrant at '{QDRANT_PATH}' is already open {detail}. {hint} "
        "For concurrent access set QDRANT_URL to a Qdrant server."
    )


def _open_embedded() -> QdrantClient:
    last_error: RuntimeError | None = None
    for attempt in range(_OPEN_RETRIES):
        try:
            return QdrantClient(path=str(QDRANT_PATH))
        except RuntimeError as error:
            if not _is_locked_error(error):
                raise
            last_error = error
            marker = _read_marker_pid()
            if marker is not None and not _pid_alive(marker):
                # Our marker says the owner is gone. Double-check that no live
                # process actually holds `.lock` before touching it: unlinking a
                # lock a live client still holds would let two writers corrupt
                # the store.
                if _find_lock_holder() is None:
                    _clear_stale_lock()
                    continue
            if attempt < _OPEN_RETRIES - 1:
                # No marker yet, or a live owner that may still be shutting down.
                time.sleep(_OPEN_RETRY_DELAY)
                continue
            break
    raise _locked_message(last_error) from last_error


def get_client() -> QdrantClient:
    global _client
    # Fast path: already created (no lock needed once the instance exists).
    if _client is not None:
        return _client

    with _client_lock:
        # Double-check: another thread may have created it while we waited.
        if _client is not None:
            return _client

        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL)
        else:
            _client = _open_embedded()
            _write_marker()

        atexit.register(_close)
        return _client


def _write_marker() -> None:
    try:
        _PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass


def _close() -> None:
    global _client
    with _client_lock:
        if _client is None:
            return
        if not QDRANT_URL and _read_marker_pid() == os.getpid():
            try:
                _PID_FILE.unlink()
            except OSError:
                pass
        _client.close()
        _client = None


def status() -> dict:
    """Diagnostic: whether the embedded store is locked and by whom."""
    holder = _holder_pid()
    return {
        "mode": "server" if QDRANT_URL else "embedded",
        "url": QDRANT_URL or None,
        "path": str(QDRANT_PATH),
        "locked": holder is not None,
        "holder_pid": holder,
        "holder_alive": _pid_alive(holder) if holder else False,
    }


if __name__ == "__main__":
    info = status()
    for key, value in info.items():
        print(f"{key}: {value}")
    if "--stop" in sys.argv[1:]:
        pid = info.get("holder_pid")
        if not pid or pid == os.getpid():
            print("nothing to stop")
        else:
            os.kill(pid, 15)
            print(f"sent SIGTERM to pid {pid}")
