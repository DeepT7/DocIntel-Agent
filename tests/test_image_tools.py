"""Tests for the raw image store reader, hydration and the view_image tool."""

import json
import tempfile
import unittest
from pathlib import Path

from app.ingestion.raw_assets import image_id_for, read_raw_image
from app.reasoning.answer import format_context
from app.reasoning.image_tools import hydrate_image_stubs, view_image

FILE_ID = "e" * 32


def _write_manifest(out_dir: Path, assets: list[dict]) -> None:
    target = out_dir / "images" / FILE_ID
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(
        json.dumps({"file_id": FILE_ID, "assets": assets}), encoding="utf-8"
    )


def _asset(name: str) -> dict:
    return {
        "asset_id": Path(name).stem,
        "file_name": name,
        "path": f"images/{FILE_ID}/{name}",
        "image_path": f"images/{FILE_ID}/{name}",
        "width": 640,
        "height": 480,
        "page": 4,
        "caption": "Figure 3: Architecture",
    }


def _image_chunk(image_id: str, image_path: str) -> dict:
    return {
        "chunk_id": "chunk:9",
        "chunk_type": "image_stub",
        "asset_id": image_id,
        "asset_path": image_path,
        "caption": "Figure 3: Architecture",
        "text": f"[IMAGE STUB | ID: {image_id} | Path: {image_path}]\nCaption: Figure 3: Architecture",
        "title": "doc",
        "url": "doc.md",
        "section_titles": ["Overview"],
        "section_heading": "Overview",
        "score": 0.4,
    }


class ReadRawImageTest(unittest.TestCase):
    def test_reads_by_image_id_and_adds_abs_path(self):
        image_id = image_id_for(FILE_ID, "s0", 0)
        name = f"{image_id}.png"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "assets"
            _write_manifest(out_dir, [_asset(name)])
            img = out_dir / "images" / FILE_ID / name
            img.parent.mkdir(parents=True, exist_ok=True)
            img.write_bytes(b"\x89PNG")

            record = read_raw_image(image_id, out_dir=out_dir)
            self.assertIsNotNone(record)
            self.assertEqual(record["abs_path"], str(img))

    def test_reads_by_image_path(self):
        image_id = image_id_for(FILE_ID, "s1", 1)
        name = f"{image_id}.png"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "assets"
            _write_manifest(out_dir, [_asset(name)])
            record = read_raw_image(
                image_path=f"images/{FILE_ID}/{name}", out_dir=out_dir
            )
            self.assertIsNotNone(record)
            self.assertEqual(record["file_name"], name)

    def test_missing_manifest_returns_none(self):
        self.assertIsNone(read_raw_image(image_id_for(FILE_ID, "s0", 0)))


class ImageHydrationTest(unittest.TestCase):
    def test_hydration_annotates_stub_with_stored_file(self):
        image_id = image_id_for(FILE_ID, "s0", 0)
        name = f"{image_id}.png"
        image_path = f"images/{FILE_ID}/{name}"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "assets"
            _write_manifest(out_dir, [_asset(name)])
            chunks = hydrate_image_stubs(
                [_image_chunk(image_id, image_path)], out_dir=out_dir
            )
        self.assertTrue(chunks[0].get("image_resolved"))
        self.assertIn("Figure 3: Architecture", chunks[0]["text"])
        self.assertIn("Page: 4", chunks[0]["text"])
        self.assertIn("640x480", chunks[0]["text"])

    def test_unresolved_image_keeps_the_stub(self):
        chunk = _image_chunk(image_id_for(FILE_ID, "s0", 0), "images/x/y.png")
        with tempfile.TemporaryDirectory() as tmp:
            chunks = hydrate_image_stubs([chunk], out_dir=Path(tmp) / "assets")
        self.assertEqual(chunks[0]["text"], chunk["text"])
        self.assertNotIn("image_resolved", chunks[0])

    def test_non_image_chunks_are_untouched(self):
        chunk = {"chunk_type": "text", "text": "hello"}
        self.assertEqual(hydrate_image_stubs([chunk]), [chunk])

    def test_hydrated_context_contains_the_figure_block(self):
        image_id = image_id_for(FILE_ID, "s0", 0)
        name = f"{image_id}.png"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "assets"
            _write_manifest(out_dir, [_asset(name)])
            chunks = hydrate_image_stubs(
                [_image_chunk(image_id, f"images/{FILE_ID}/{name}")], out_dir=out_dir
            )
        self.assertIn("[Image available", format_context(chunks))


class ViewImageToolTest(unittest.TestCase):
    def test_rejects_invalid_ids(self):
        self.assertIsNone(view_image("../../etc/passwd"))
        self.assertIsNone(view_image(""))

    def test_views_a_stored_image(self):
        image_id = image_id_for(FILE_ID, "s0", 0)
        name = f"{image_id}.png"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "assets"
            _write_manifest(out_dir, [_asset(name)])
            record = view_image(image_id, out_dir=out_dir)
        self.assertEqual(record["asset_id"], image_id)


if __name__ == "__main__":
    unittest.main()
