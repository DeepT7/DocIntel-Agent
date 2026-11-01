"""Tests for the incremental indexing mechanisms (no Qdrant/LLM required).

Run with::

    python -m unittest discover -s tests -v
"""

import tempfile
import time
import unittest
from pathlib import Path

from app.indexing.content import ContentError, read_content
from app.indexing.metadata import MetadataCache
from app.indexing.scanner import file_meta, iter_files, scan_tree, snapshot
from app.indexing.watcher import FolderWatcher
from app.indexing.worker import UPSERT, BackgroundIndexer, Change


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ScannerTest(unittest.TestCase):
    def test_metadata_only_and_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "a.md", "# A")
            _write(root / "sub" / "b.txt", "b")
            _write(root / ".hidden.md", "hidden")
            _write(root / "sub" / "skip.bin", "bin")

            rels = sorted(meta.rel_path for meta in iter_files(root))
            self.assertEqual(rels, ["a.md", "sub/b.txt"])

            tree = scan_tree(root)
            self.assertEqual(tree["path"], "")
            names = sorted(child["name"] for child in tree["children"])
            self.assertEqual(names, ["a.md", "sub"])
            sub = next(child for child in tree["children"] if child["name"] == "sub")
            self.assertEqual(sub["children"][0]["path"], "sub/b.txt")

            meta = file_meta(root, "a.md")
            self.assertEqual(meta.size, 3)
            self.assertGreater(meta.mtime_ns, 0)
            self.assertIsNone(file_meta(root, "missing.md"))


class MetadataCacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write(self.root / "a.md", "one")
        self.cache = MetadataCache(self.root / "meta.sqlite3")
        self.scope = str(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_diff_lifecycle(self):
        files = snapshot(self.root)
        diff = self.cache.diff(self.scope, files, "v1")
        self.assertEqual(diff.added, ["a.md"])
        self.assertTrue(diff.changed)

        meta = files["a.md"]
        self.cache.mark_indexed(self.scope, meta, "hash-1", 2, "v1")
        diff = self.cache.diff(self.scope, files, "v1")
        self.assertEqual(diff.changed, [])
        self.assertEqual(diff.unchanged, 1)

        # Newer chunker version forces a re-index.
        self.assertEqual(self.cache.diff(self.scope, files, "v2").modified, ["a.md"])

        # Delete on disk is reported and can be removed from the cache.
        (self.root / "a.md").unlink()
        diff = self.cache.diff(self.scope, snapshot(self.root), "v1")
        self.assertEqual(diff.deleted, ["a.md"])
        self.cache.remove(self.scope, ["a.md"])
        self.assertEqual(self.cache.counts(self.scope)["total"], 0)

    def test_error_status_is_retried(self):
        meta = snapshot(self.root)["a.md"]
        self.cache.mark_error(self.scope, meta, "a.md", "boom", "v1")
        diff = self.cache.diff(self.scope, snapshot(self.root), "v1")
        self.assertEqual(diff.modified, ["a.md"])

    def test_touch_refresh(self):
        meta = snapshot(self.root)["a.md"]
        self.cache.mark_indexed(self.scope, meta, "hash-1", 2, "v1")
        time.sleep(0.01)
        (self.root / "a.md").touch()
        touched = snapshot(self.root)["a.md"]
        self.assertIn("a.md", self.cache.diff(self.scope, {**snapshot(self.root)}, "v1").modified)
        self.cache.mark_seen(self.scope, touched, "hash-1")
        self.assertEqual(
            self.cache.diff(self.scope, snapshot(self.root), "v1").changed, []
        )


class WorkerTest(unittest.TestCase):
    def test_batches_and_last_action_wins(self):
        batches = []
        worker = BackgroundIndexer(
            lambda scope, changes: batches.append((scope, changes)),
            debounce_seconds=0.2,
        )
        try:
            worker.submit("s", Change(UPSERT, "a.md"))
            worker.submit("s", Change(UPSERT, "b.md"))
            worker.submit("s", Change("delete", "b.md"))
            worker.submit("s", Change(UPSERT, "a.md"))
            self.assertTrue(worker.wait_idle(3))
        finally:
            worker.stop()
        self.assertEqual(len(batches), 1)
        scope, changes = batches[0]
        self.assertEqual(scope, "s")
        self.assertEqual(
            {(change.action, change.rel_path) for change in changes},
            {(UPSERT, "a.md"), ("delete", "b.md")},
        )


class ContentTest(unittest.TestCase):
    def test_reads_text_and_blocks_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "docs" / "a.md", "hello")
            payload = read_content(root, "docs/a.md")
            self.assertEqual(payload["text"], "hello")
            self.assertEqual(payload["path"], "docs/a.md")

            with self.assertRaises(ContentError):
                read_content(root, "../secret.txt")
            with self.assertRaises(ContentError):
                read_content(root, "missing.md")


class WatcherTest(unittest.TestCase):
    def test_polling_fallback_emits_upsert(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "a.md", "one")
            events = []
            watcher = FolderWatcher(
                lambda scope, change: events.append((change.action, change.rel_path)),
                debounce_seconds=0.05,
                poll_seconds=0.2,
            )
            watcher.start(str(root))
            try:
                time.sleep(0.4)
                _write(root / "new.md", "two")
                time.sleep(1.0)
            finally:
                watcher.stop()
            self.assertIn((UPSERT, "new.md"), events)


if __name__ == "__main__":
    unittest.main()
