"""Tests for query-time table references, hydration and the raw-table tool."""

import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

from app.capabilities.registry import _structural_fields
from app.ingestion.chunking import build_chunks_for_file
from app.ingestion.raw_assets import load_tables, save_tables, table_id_for
from app.reasoning.answer import format_context
from app.reasoning.table_tools import (
    extract_table_references,
    hydrate_table_stubs,
    read_raw_table_tool,
    render_table_markdown,
    table_chunk_from_raw,
)

FILE_ID = "d" * 32


def _raw_table(table_id: str, rows: list[list[str]], **extra) -> dict:
    record = {
        "table_id": table_id,
        "file_id": FILE_ID,
        "path": "doc.md",
        "section_id": "s1",
        "breadcrumb": "Results",
        "caption": "Table 1: Costs",
        "table_number": "1",
        "headers": rows[0] if rows else [],
        "rows": rows,
        "row_count": max(len(rows) - 1, 0),
        "col_count": len(rows[0]) if rows else 0,
    }
    record.update(extra)
    return record


def _table_chunk(table_id: str) -> dict:
    return {
        "chunk_id": "chunk:1",
        "chunk_type": "table_stub",
        "asset_id": table_id,
        "text": (
            "[Document: doc.md | Section: Results]\n\n"
            "[TABLE STUB | ID: " + table_id + "]\n"
            "Caption: Table 1: Costs\n"
            "Columns: Item, Cost\n"
            "Row Entities: alpha, beta\n"
            "Total Rows: 2"
        ),
        "title": "doc",
        "url": "doc.md",
        "section_id": "s1",
        "section_titles": ["Results"],
        "section_heading": "Results",
        "caption": "Table 1: Costs",
        "table_number": "1",
        "headers": ["Item", "Cost"],
        "row_count": 2,
        "score": 0.5,
    }


class TableReferenceTest(unittest.TestCase):
    def test_extracts_english_and_vietnamese_references(self):
        refs = extract_table_references(
            "Compare Table 2 with Bảng 3.1 and table A.1; see the table of contents."
        )
        self.assertIn("2", refs)
        self.assertIn("3.1", refs)
        # Case variants are included so a lowercase reference matches a caption.
        self.assertIn("A.1", refs)
        self.assertIn("a.1", refs)

    def test_ignores_prose_without_a_number(self):
        self.assertEqual(extract_table_references("see the table below"), [])


class TableRenderingTest(unittest.TestCase):
    def test_renders_rows_as_markdown(self):
        raw = _raw_table(table_id_for(FILE_ID, "s1", 0), [["Item", "Cost"], ["alpha", "1"]])
        markdown = render_table_markdown(raw)
        self.assertIn("| Item | Cost |", markdown)
        self.assertIn("| alpha | 1 |", markdown)

    def test_truncates_large_tables_with_a_pointer(self):
        rows = [["Item", "Cost"]] + [[f"row{i}", str(i)] for i in range(50)]
        raw = _raw_table(table_id_for(FILE_ID, "s1", 0), rows)
        markdown = render_table_markdown(raw, max_rows=5)
        self.assertIn("| row4 | 4 |", markdown)
        self.assertNotIn("| row49 | 49 |", markdown)
        self.assertIn("Showing 5 of 50 rows", markdown)
        self.assertIn(raw["table_id"], markdown)

    def test_picture_derived_table_without_rows_has_no_markdown(self):
        raw = _raw_table(
            table_id_for(FILE_ID, "s1", 0),
            [],
            content="| A | B |\n| --- | --- |\n| 1 | 2 |",
        )
        self.assertEqual(render_table_markdown(raw), "")


class HydrationTest(unittest.TestCase):
    def test_hydration_injects_raw_rows_into_the_prompt_context(self):
        table_id = table_id_for(FILE_ID, "s1", 0)
        raw = _raw_table(table_id, [["Item", "Cost"], ["alpha", "1"]])
        with patch(
            "app.reasoning.table_tools.read_raw_table", return_value=raw
        ):
            chunks = hydrate_table_stubs([_table_chunk(table_id)])
        context = format_context(chunks)
        self.assertIn("| alpha | 1 |", context)
        self.assertTrue(chunks[0].get("table_hydrated"))

    def test_picture_table_falls_back_to_the_stub_text(self):
        table_id = table_id_for(FILE_ID, "s1", 0)
        raw = _raw_table(table_id, [], content="| A | B |")
        with patch(
            "app.reasoning.table_tools.read_raw_table", return_value=raw
        ):
            chunks = hydrate_table_stubs([_table_chunk(table_id)])
        self.assertEqual(chunks[0]["text"], _table_chunk(table_id)["text"])
        self.assertNotIn("table_hydrated", chunks[0])

    def test_missing_raw_table_keeps_the_stub(self):
        table_id = table_id_for(FILE_ID, "s1", 0)
        with patch("app.reasoning.table_tools.read_raw_table", return_value=None):
            chunks = hydrate_table_stubs([_table_chunk(table_id)])
        self.assertEqual(chunks[0]["text"], _table_chunk(table_id)["text"])

    def test_read_raw_table_tool_rejects_invalid_ids(self):
        self.assertIsNone(read_raw_table_tool("../../etc/passwd"))
        self.assertIsNone(read_raw_table_tool(""))

    def test_read_raw_table_tool_reads_a_saved_asset(self):
        table_id = table_id_for(FILE_ID, "s1", 0)
        raw = _raw_table(table_id, [["Item", "Cost"], ["alpha", "1"]])
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "assets"
            save_tables(FILE_ID, [raw], out_dir=out_dir)
            self.assertEqual(
                read_raw_table_tool(table_id, out_dir=out_dir)["table_id"], table_id
            )

    def test_tool_chunk_renders_rows_for_the_context(self):
        table_id = table_id_for(FILE_ID, "s1", 0)
        raw = _raw_table(table_id, [["Item", "Cost"], ["alpha", "1"]])
        chunk = table_chunk_from_raw(raw, table_number="1")
        self.assertEqual(chunk["chunk_type"], "table_stub")
        self.assertIn("| alpha | 1 |", format_context([chunk]))


class IngestMetadataTest(unittest.TestCase):
    def test_table_manifest_records_table_number_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "doc.md"
            path.write_text(
                "# Report\n\n| Item | Cost |\n| --- | --- |\n| alpha | 1 |\n\n"
                "Table 1: Costs\n",
                encoding="utf-8",
            )
            build_chunks_for_file(root, path, assets_dir=root / "assets")
            from app.ingestion.identity import file_id_for

            tables = load_tables(file_id_for("doc.md"), out_dir=root / "assets")
            self.assertEqual(tables[0]["table_number"], "1")
            self.assertEqual(tables[0]["rows"][1], ["alpha", "1"])

    def test_document_card_table_manifest_exposes_table_number(self):
        chunk = _table_chunk(table_id_for(FILE_ID, "s1", 0))
        chunk.update(
            {
                "section_ancestors": ["s1"],
                "section_depths": [1],
                "section_numbers": ["1"],
                "asset_path": f"tables/{FILE_ID}.json",
            }
        )
        card = _structural_fields([chunk], "doc.md", "hash")
        self.assertEqual(card["tables"][0]["table_number"], "1")
        self.assertEqual(card["tables"][0]["table_id"], chunk["asset_id"])


class TableNumberRetrievalTest(unittest.TestCase):
    def test_table_chunks_filters_on_the_payload_table_number(self):
        Point = namedtuple("Point", "payload")
        point = Point(
            payload={
                "chunk_id": "file:3",
                "chunk_type": "table_stub",
                "table_number": "2",
                "path": "doc.md",
                "root": "F:/docs",
                "text": "stub",
            }
        )

        class FakeClient:
            def __init__(self):
                self.kwargs = None

            def scroll(self, **kwargs):
                self.kwargs = kwargs
                return [point], None

        fake = FakeClient()
        with patch("app.retrieval.retrieve.get_client", return_value=fake):
            from app.retrieval.retrieve import table_chunks

            results = table_chunks("F:/docs", ["2"], paths=["doc.md"])

        self.assertEqual(results[0]["chunk_id"], "file:3")
        conditions = {condition.key: condition for condition in fake.kwargs["scroll_filter"].must}
        self.assertIn("chunk_type", conditions)
        self.assertIn("table_number", conditions)
        self.assertEqual(conditions["table_number"].match.any, ["2"])
        self.assertEqual(conditions["chunk_type"].match.value, "table_stub")


class AnswerNodeInjectionTest(unittest.TestCase):
    def test_answer_node_context_contains_raw_cell_values(self):
        table_id = table_id_for(FILE_ID, "s1", 0)
        raw = _raw_table(table_id, [["Item", "Cost"], ["alpha", "1"]])
        chunk = _table_chunk(table_id)

        captured = {}

        def fake_generate_answer(query, context, thinking=False):
            captured["context"] = context
            return "alpha costs 1."

        with patch(
            "app.reasoning.table_tools.read_raw_table", return_value=raw
        ), patch(
            "app.reasoning.rag_graph.generate_answer", side_effect=fake_generate_answer
        ):
            from app.reasoning.rag_graph import answer_node

            result = answer_node(
                {
                    "query": "What is the cost of alpha in Table 1?",
                    "relevant_chunks": [chunk],
                }
            )

        self.assertIn("| alpha | 1 |", captured["context"])
        self.assertIn("| alpha | 1 |", result["context"])

    def test_answer_question_hydrates_table_stubs(self):
        table_id = table_id_for(FILE_ID, "s1", 0)
        raw = _raw_table(table_id, [["Item", "Cost"], ["alpha", "1"]])
        chunk = _table_chunk(table_id)
        captured = {}

        def fake_generate(query, context, thinking=False):
            captured["context"] = context
            return "alpha costs 1."

        with patch(
            "app.reasoning.table_tools.read_raw_table", return_value=raw
        ), patch(
            "app.reasoning.answer.vector_search", return_value=[chunk]
        ), patch(
            "app.reasoning.answer.generate_answer", side_effect=fake_generate
        ):
            from app.reasoning.answer import answer_question

            result = answer_question("What is the cost of alpha?")

        self.assertIn("| alpha | 1 |", captured["context"])
        self.assertIn("| alpha | 1 |", result["context"])


if __name__ == "__main__":
    unittest.main()