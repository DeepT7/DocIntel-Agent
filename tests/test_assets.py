"""Tests for smart visual-asset extraction and filtering (no Qdrant/LLM)."""

import tempfile
import unittest
import zipfile
from pathlib import Path

from app.ingestion.assets import (
    _Candidate,
    _ImageInfo,
    _reject_reason,
    extract_assets,
)
from app.ingestion.raw_assets import remove_raw_assets

try:
    import pymupdf
except ImportError:  # asset tests need PyMuPDF's image decoder
    pymupdf = None

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _png(width: int, height: int, *, draw: bool = True) -> bytes:
    """Render a simple chart-like (or blank) PNG with PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    if draw:
        page.draw_rect(pymupdf.Rect(5, 5, width - 5, height - 5), color=(0, 0, 0), width=1)
        page.draw_line((10, height - 10), (width - 10, 10), color=(0.2, 0.2, 0.8), width=2)
        page.insert_text((10, 20), "Chart", fontsize=10)
    else:
        page.draw_rect(page.rect, color=None, fill=(1, 1, 1))
    data = page.get_pixmap(dpi=150, alpha=False).tobytes("png")
    doc.close()
    return data


def _docx_with_images(root: Path) -> Path:
    chart = _png(400, 300)
    logo = _png(120, 60)
    document = f'''<w:document xmlns:w="{WORD_NS}" xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" xmlns:r="{R_NS}">
<w:body>
  <w:p><w:r><w:drawing><wp:inline>
    <wp:extent cx="3657600" cy="2743200"/>
    <a:graphic><a:graphicData><a:blip r:embed="rId1"/></a:graphicData></a:graphic>
  </wp:inline></w:drawing></w:r></w:p>
  <w:p><w:r><w:drawing><wp:inline>
    <wp:extent cx="457200" cy="457200"/>
    <a:graphic><a:graphicData><a:blip r:embed="rId2"/></a:graphicData></a:graphic>
  </wp:inline></w:drawing></w:r></w:p>
</w:body></w:document>'''
    rels = f'''<Relationships xmlns="{PKG_RELS_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/chart.png"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/logo.png"/>
</Relationships>'''
    path = root / "fake.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", rels)
        archive.writestr("word/media/chart.png", chart)
        archive.writestr("word/media/logo.png", logo)
        archive.writestr("word/media/unused.png", chart)
    return path


@unittest.skipUnless(pymupdf is not None, "PyMuPDF is required for asset tests")
class AssetExtractionTest(unittest.TestCase):
    def test_docx_keeps_chart_drops_small_logo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docx = _docx_with_images(root)
            out = root / "assets"

            manifest = extract_assets(docx, "fake.docx", out_dir=out)

            self.assertEqual(manifest["candidates"], 2)  # unused.png is not referenced
            self.assertEqual(manifest["kept"], 1)
            self.assertEqual(manifest["rejected"].get("small_on_page"), 1)
            kept = manifest["assets"][0]
            self.assertEqual(kept["source"], "docx")
            self.assertEqual(kept["display_in"], [4.0, 3.0])
            self.assertTrue((out / kept["path"]).is_file())
            self.assertTrue((out / "images" / manifest["file_id"] / "manifest.json").is_file())

    def test_remove_assets_and_unsupported_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "assets"
            docx = _docx_with_images(root)
            manifest = extract_assets(docx, "fake.docx", out_dir=out)
            asset_dir = out / "images" / manifest["file_id"]
            self.assertTrue(asset_dir.is_dir())

            remove_raw_assets("fake.docx", out_dir=out)
            self.assertFalse(asset_dir.exists())

            md = root / "note.md"
            md.write_text("# Note", encoding="utf-8")
            empty = extract_assets(md, "note.md", out_dir=out)
            self.assertEqual(empty["kept"], 0)
            self.assertFalse((out / "images" / empty["file_id"]).exists())


class AssetFilterTest(unittest.TestCase):
    def _candidate(self, *, width=400, height=300, ink=0.2, bbox=None, display=None):
        return _Candidate(
            data=b"x" * 4096,
            ext=".png",
            source="docx",
            info=_ImageInfo(width=width, height=height, entropy=2.0, ink_ratio=ink),
            bbox=bbox,
            display_w=display[0] if display else None,
            display_h=display[1] if display else None,
        )

    def test_visual_junk_is_rejected(self):
        self.assertEqual(_reject_reason(self._candidate(width=64, height=64), 1), "too_small")
        self.assertEqual(_reject_reason(self._candidate(width=1000, height=120), 1), "thin")
        self.assertEqual(_reject_reason(self._candidate(ink=0.0), 1), "blank")
        self.assertEqual(
            _reject_reason(self._candidate(bbox=(0, 0, 30, 30)), 1),
            "small_on_page",
        )
        self.assertEqual(
            _reject_reason(self._candidate(display=(0.5, 0.5)), 1),
            "small_on_page",
        )
        self.assertEqual(
            _reject_reason(self._candidate(width=200, height=100), 3),
            "repeated",
        )

    def test_informative_image_is_kept(self):
        self.assertIsNone(_reject_reason(self._candidate(), 1))


if __name__ == "__main__":
    unittest.main()
