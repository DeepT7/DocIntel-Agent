# Incremental indexing & file watching

How DocIntel keeps a folder indexed without loading the corpus into RAM and
without re-embedding files that did not change.

The subsystem lives in `app/indexing/` and plugs into the existing ingestion
pipeline (`app/ingestion/folder.py`) and document registry
(`app/capabilities/registry.py`).

---

## 1. Data flow

```
                    ┌──────────────────────────── app/indexing ───────────────────────────┐
 folder path ─────► │ scanner.snapshot()          metadata only: rel_path, size, mtime_ns  │
 (GET /api/tree)    │ scanner.scan_tree()         nested tree, contents never opened      │
                    └──────────────────────────┬──────────────────────────────────────────┘
                                               │
                                    ┌──────────▼───────────┐
                                    │  MetadataCache       │  SQLite: data/index/metadata.sqlite3
                                    │  (file_state)        │
                                    └──────┬───────┬───────┘
                        unchanged ─────────┘       │ changed / new / deleted
                        (stat only)                │
                                                   ▼
 GET/POST /api/sync ──► sync_folder() ──► BackgroundIndexer (queue + debounce)
                                                  ▲
 OS events ──► FolderWatcher (watchdog) ──────────┘   (polling fallback if unavailable)
               on_created / on_modified
               on_deleted / on_moved
                                                  │
                                                  ▼
                                  pipeline.apply_changes() ──► index_paths() / remove_paths()
                                                  │            chunk → embed → Qdrant
                                                  │            registry.sync_outlines()
                                                  ▼
                                  cache.mark_indexed / mark_error / remove

 GET /api/file?path=... ──► content.read_content()   raw bytes read from disk on demand
```

Two entry points share the same file-state cache and the same per-file pipeline:

* **explicit** — `/api/sync` (the **Update index** button) or
  `python -m app.ingestion.folder` performs a full metadata scan and diffs it;
* **real-time** — the watcher reacts to OS events and enqueues only the touched
  paths; nothing else is scanned.

**Why threads instead of asyncio:** Qdrant's client, the embedder and the file
reader are synchronous. A daemon thread (`BackgroundIndexer`) is the smallest
change that keeps the HTTP loop free while those blocking calls run. The FastAPI
lifespan owns the thread; it is stopped on shutdown. An async variant would need
async clients for Qdrant and the embedding provider, without changing this
design's boundaries.

---

## 2. Module / class structure

| File | API | Responsibility |
|------|-----|----------------|
| `app/indexing/scanner.py` | `FileMeta`, `iter_files()`, `snapshot()`, `file_meta()`, `scan_tree()` | **Mechanism 1.** Lazy `os.scandir` walk; yields path/name/size/mtime only. Hidden files, unsupported extensions and symlinks are skipped. No file content is read. |
| `app/indexing/metadata.py` | `MetadataCache`, `Diff`, `get_cache()` | **Mechanism 2a.** SQLite file-state cache. `diff()` compares a metadata snapshot with the stored rows; status transitions (`pending` → `indexed` / `error`) and `index_version` are tracked here. |
| `app/indexing/worker.py` | `BackgroundIndexer`, `Change`, `UPSERT`, `DELETE` | **Mechanism 2b.** Coalesces changes per `(scope, rel_path)` (last action wins) and runs one daemon thread. Waits for a quiet debounce window before calling the handler, so a burst of saves becomes one re-index. |
| `app/indexing/watcher.py` | `FolderWatcher`, `_WatchdogHandler`, `_Debouncer` | **Mechanism 3.** `watchdog` observer with recursive OS events; per-path debounce. Falls back to a metadata poller when watchdog is missing. Emits `Change(action, rel_path)` only. |
| `app/indexing/pipeline.py` | `apply_changes()` | Bridges one debounced batch to `index_paths()` / `remove_paths()`, `sync_outlines()`, and the cache. Hash-confirms before re-indexing, so `touch` / duplicate events are free. |
| `app/indexing/content.py` | `read_content()`, `ContentError` | **Mechanism 4.** Reads one file on demand, confined to the scope (blocks `../`), decodes UTF-8 text or returns base64, caps the read at `CONTENT_MAX_BYTES`. |
| `app/indexing/manager.py` | `IndexManager`, `get_manager()` | Process-wide facade: start/stop watcher, status, tree, raw read. |
| `app/ingestion/folder.py` | `index_paths()`, `remove_paths()`, `sync_folder()` | Existing chunker/embedder, refactored so one file can be indexed without scanning the folder. `sync_folder()` now uses the metadata cache for change detection. |
| `app/ingestion/assets.py` | `extract_assets()`, `sync_assets()`, `remove_assets()` | Smart visual-asset stage: extracts PDF picture regions / Office media, filters icons, logos, rules, blank and repeated template images, writes `data/index/assets/<file_id>/` + `manifest.json`. Best-effort: failures never block text indexing. |

```python
class IndexManager:
    cache: MetadataCache
    _worker: BackgroundIndexer          # debounced file queue
    _watcher: FolderWatcher             # OS events -> worker

    def start(scope) -> dict            # (re)start watcher for a folder
    def sync(scope) -> dict             # explicit scan (sync_folder)
    def tree(scope) -> dict             # metadata-only tree
    def read(scope, rel_path) -> dict   # on-demand content
    def status(scope=None) -> dict      # watcher mode, queue depth, counters
```

---

## 3. File-state cache schema

`data/index/metadata.sqlite3` (WAL mode, one connection per operation):

```sql
CREATE TABLE file_state (
    scope          TEXT    NOT NULL,     -- canonical folder root (normalize_root)
    rel_path       TEXT    NOT NULL,     -- POSIX path relative to scope
    abs_path       TEXT    NOT NULL,
    name           TEXT    NOT NULL,
    ext            TEXT    NOT NULL DEFAULT '',
    size           INTEGER NOT NULL,     -- fast change detection
    mtime_ns       INTEGER NOT NULL,     -- fast change detection
    content_hash   TEXT,                 -- SHA-256, confirms real changes
    status         TEXT    NOT NULL DEFAULT 'pending', -- pending|indexed|error
    chunk_count    INTEGER NOT NULL DEFAULT 0,
    index_version  TEXT    NOT NULL DEFAULT '',
    error          TEXT,
    first_seen_at  REAL    NOT NULL,
    updated_at     REAL    NOT NULL,
    indexed_at     REAL,
    PRIMARY KEY (scope, rel_path)
);
CREATE INDEX idx_file_state_scope_status ON file_state(scope, status);
CREATE INDEX idx_file_state_scope_mtime  ON file_state(scope, mtime_ns);

CREATE TABLE scope_state (
    scope         TEXT PRIMARY KEY,
    watcher       TEXT NOT NULL DEFAULT 'stopped', -- watchdog|polling|stopped
    last_scan_at  REAL,
    last_sync_at  REAL,
    stats         TEXT                              -- JSON of the last sync result
);
```

Change detection order (cheapest first):

1. row missing → **added**;
2. `size`/`mtime_ns` equal, status `indexed`, same `index_version` → **unchanged**;
3. otherwise hash the file:
   * same hash → metadata refreshed (`mark_seen`), no chunking/embedding;
   * different hash → **modified**;
4. cached path not on disk → **deleted** (vectors removed by `file_id`).

The old JSON manifest is no longer written or read: `file_state` is the single
source of truth for change detection.

---

## 4. HTTP API

| Method & path | Purpose |
|---------------|---------|
| `GET /api/tree?scope=` | Nested folder/file **metadata** tree (mechanism 1). |
| `POST /api/pick-folder` | Open the OS-native folder dialog (Modern Explorer `IFileOpenDialog` on WSL/Windows) and return the selected path. |
| `POST /api/sync` | Queue an explicit incremental scan (mechanism 2). Returns `{job_id}`; poll `/api/job/{id}` for `{files, changed, deleted, chunks_reindexed, embeddings_computed, outlines}`. |
| `GET /api/index/status?scope=` | Watcher mode, queue depth, processed count, cache counters. |
| `GET /api/file?scope=&path=&max_bytes=` | Raw content of one file, on demand (mechanism 4). Returns `text` (UTF-8) or `content` (base64), plus `size`, `mtime_ns`, `truncated`. |

The watcher starts automatically at server startup for the remembered folder and
whenever `POST /api/scope` selects a new folder. Setting a new folder stops the
previous observer.

At startup the saved folder is also re-scanned in the background as a `sync:*`
job, so edits made while the server was down are indexed and an explicit
`/api/sync` for the same scope is deduplicated. The UI runs the same incremental
scan after selecting a folder; the **Update index** button remains as a manual
fallback.

Examples:

```bash
curl "http://127.0.0.1:8756/api/tree"
curl "http://127.0.0.1:8756/api/file?path=guide.md"
curl "http://127.0.0.1:8756/api/index/status"
```

---

## 5. Sequences

**Startup**

1. FastAPI lifespan reads the saved scope.
2. `IndexManager.start(scope)` creates the cache DB if needed and starts the
   watcher (`watchdog`, or polling fallback).
3. A background catch-up `sync_folder()` diffs the folder against the metadata
   cache, indexing only files changed while the server was down.

**Watcher event** (`on_modified`)

1. Watchdog handler filters directories, hidden paths and unsupported extensions.
2. Per-path debounce timer (default 1 s) fires `Change("upsert", rel_path)`.
3. `BackgroundIndexer` coalesces the batch (quiet window, max 5× debounce).
4. `apply_changes()` hashes the file; if the hash is unchanged it just refreshes
   `mtime_ns`/`size`.
5. Otherwise `index_paths()` deletes the old points for `file_id`, chunks, embeds
   cache-aware, upserts, refreshes filtered visual assets, then `sync_outlines()`
   refreshes the document card.
6. Cache row → `indexed` with the new hash; failures → `error` (retried on the
   next event).

**Manual sync**

`/api/sync` queues a background job (same scope while one is queued/running is
deduplicated) → `sync_folder()` → `snapshot()` → `MetadataCache.diff()` →
hash-confirm modified files → `index_paths(changed)` + `remove_paths(deleted)` →
outlines → file-state cache + `scope_state`. The UI polls `/api/job/{id}` and
refreshes Sources when it finishes, so selecting a folder never waits for
embeddings.

**On-demand read**

`GET /api/file` → `read_content()` → resolve inside the scope → `open().read()` →
UTF-8 text or base64 → response. Nothing else in the corpus is touched.

---

## 6. Configuration

All values can be overridden with `DOCINTEL_INGESTION__*` env vars:

| Setting | Default | Meaning |
|---------|---------|---------|
| `watch_enabled` | `true` | Start the watcher when a folder is selected. |
| `watch_debounce_seconds` | `1.0` | Quiet window before a batch is indexed (and per-path event debounce). |
| `watch_poll_seconds` | `4.0` | Interval of the polling fallback (watchdog not installed / not permitted). |
| `content_max_bytes` | `4_000_000` | Maximum bytes `GET /api/file` reads per request. |

`watchdog==7.0.0` is optional: without it the app still works with the polling
fallback (metadata `stat` only, never contents).

---

## 7. Guarantees / edge cases

* **Memory:** scans hold only metadata; at most one file's chunks are alive per
  pipeline step (plus the embed batch, which was already the case).
* **No stale vectors:** a modified/deleted file's points are deleted by
  `file_id` before re-insert.
* **Touch is free:** mtime changes are confirmed with a hash; identical content is
  not re-embedded.
* **Rename** — watchdog reports `on_moved` as delete(old) + upsert(new); file
  identity remains `md5(rel_path)`, so a rename is re-indexed as a new file.
* **Failures** are stored on the cache row (`status='error'`, `error=...`); the
  next event or `/api/sync` retries them.
* **Multiple scopes** share one SQLite DB and one Qdrant collection, keyed by
  `root`; the watcher only runs for the active scope.
* **Qdrant is still embedded/single-process**; the watcher worker uses the same
  shared client (`app.retrieval.store.get_client`).
