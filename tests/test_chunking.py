"""Tests for raw table/image stubbing and page-break table merging (no Qdrant)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ingestion.chunking import (
    _merge_page_break_tables,
    build_chunks_for_file,
)
from app.ingestion.raw_assets import (
    load_tables,
    read_raw_table,
    save_tables,
    table_id_for,
)


class PageBreakMergeTest(unittest.TestCase):
    def test_merges_continuation_table_and_drops_duplicate_header(self):
        blocks = [
            {"type": "heading", "level": 1, "text": "Results"},
            {"type": "table", "rows": [["Name", "Value"], ["alpha", "1"]]},
            {"type": "text", "text": "2"},
            {"type": "table", "rows": [["Name", "Value"], ["beta", "2"]]},
        ]
        merged = _merge_page_break_tables(blocks)
        tables = [b for b in merged if b["type"] == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(
            tables[0]["rows"],
            [["Name", "Value"], ["alpha", "1"], ["beta", "2"]],
        )

    def test_does_not_merge_tables_with_different_column_counts(self):
        blocks = [
            {"type": "table", "rows": [["A", "B"], ["1", "2"]]},
            {"type": "table", "rows": [["X", "Y", "Z"], ["3", "4", "5"]]},
        ]
        merged = _merge_page_break_tables(blocks)
        self.assertEqual(len([b for b in merged if b["type"] == "table"]), 2)

    def test_does_not_merge_when_second_table_has_own_caption(self):
        blocks = [
            {"type": "table", "rows": [["A", "B"], ["1", "2"]]},
            {"type": "table", "rows": [["A", "B"], ["3", "4"]]},
            {"type": "text", "text": "Table 2: Other results"},
        ]
        merged = _merge_page_break_tables(blocks)
        self.assertEqual(len([b for b in merged if b["type"] == "table"]), 2)


class TableStubTest(unittest.TestCase):
    def _write(self, root: Path, text: str) -> Path:
        path = root / "doc.md"
        path.write_text(text, encoding="utf-8")
        return path

    def test_table_stub_has_columns_and_row_entities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md = self._write(
                root,
                "# Report\n\n## Costs\n\n| Item | Cost |\n| --- | --- |\n| alpha | 1 |\n| beta | 2 |\n",
            )
            chunks = build_chunks_for_file(root, md, assets_dir=root / "assets")
            stubs = [c for c in chunks if c["chunk_type"] == "table_stub"]
            self.assertEqual(len(stubs), 1)
            stub = stubs[0]
            self.assertIn("Columns: Item, Cost", stub["text"])
            self.assertIn("Row Entities: alpha, beta", stub["text"])
            self.assertIn("Total Rows: 2", stub["text"])
            self.assertIn("[TABLE STUB | ID:", stub["text"])
            self.assertEqual(stub["headers"], ["Item", "Cost"])
            self.assertEqual(stub["row_count"], 2)
            self.assertTrue(stub["asset_id"].endswith("_t0"))


class ImageStubTest(unittest.TestCase):
    def test_manifest_image_becomes_image_stub(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md = root / "doc.md"
            md.write_text("# Title\n\nsome prose", encoding="utf-8")
            manifest = {
                "assets": [
                    {"path": "images/abc/img.png", "caption": "Figure 1: Architecture"}
                ]
            }
            with patch(
                "app.ingestion.chunking.extract_assets", return_value=manifest
            ):
                chunks = build_chunks_for_file(root, md, assets_dir=root / "assets")
            stubs = [c for c in chunks if c["chunk_type"] == "image_stub"]
            self.assertEqual(len(stubs), 1)
            stub = stubs[0]
            self.assertEqual(stub["asset_path"], "images/abc/img.png")
            self.assertEqual(stub["caption"], "Figure 1: Architecture")
            self.assertTrue(stub["asset_id"].endswith("_s0_p0"))
            self.assertIn("[IMAGE STUB | ID:", stub["text"])
            self.assertIn("Caption: Figure 1: Architecture", stub["text"])


class RawTableRoundTripTest(unittest.TestCase):
    def test_save_load_and_read_back(self):
        file_id = "d" * 32
        table_id = table_id_for(file_id, "s0", 0)
        tables = [
            {
                "table_id": table_id,
                "file_id": file_id,
                "path": "doc.md",
                "section_id": "s0",
                "breadcrumb": "Overview",
                "caption": "Costs",
                "headers": ["Item", "Cost"],
                "rows": [["Item", "Cost"], ["alpha", "1"]],
                "row_count": 1,
                "col_count": 2,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "assets"
            save_tables(file_id, tables, out_dir=out_dir)
            loaded = load_tables(file_id, out_dir=out_dir)
            self.assertEqual(loaded, tables)
            self.assertEqual(read_raw_table(table_id, out_dir=out_dir), tables[0])
            self.assertIsNone(read_raw_table("missing_t0", out_dir=out_dir))


if __name__ == "__main__":
    unittest.main()