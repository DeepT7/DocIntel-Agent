"""Incremental indexing subsystem.

Four small mechanisms that keep a folder indexed without ever loading the whole
corpus into memory:

* ``scanner``  — lazy directory traversal (metadata only);
* ``metadata`` — SQLite file-state cache used for incremental diffing;
* ``worker``   — debounced background queue;
* ``watcher``  — OS file-system events (watchdog), with a polling fallback;
* ``content``  — on-demand raw content reads.

``manager`` wires them together for the HTTP layer.
"""

from app.indexing.content import read_content
from app.indexing.metadata import MetadataCache
from app.indexing.scanner import FileMeta, iter_files, scan_tree, snapshot
from app.indexing.worker import BackgroundIndexer, Change

__all__ = [
    "BackgroundIndexer",
    "Change",
    "FileMeta",
    "MetadataCache",
    "iter_files",
    "read_content",
    "scan_tree",
    "snapshot",
]
