"""Smart asset extraction: keep informative images, drop visual junk.

Different containers hide images in different places:

- **PDF** — ``pymupdf4llm``'s layout detects picture regions (figures, charts,
  scanned pages), and ``pymupdf`` renders each region to PNG. This catches both
  embedded raster images and vector charts, unlike an xref-only scan.
- **DOCX / PPTX** — Office files are ZIP containers. Only image parts referenced
  by the main document (``word/document.xml``) or by slides
  (``ppt/slides/slideN.xml``) are candidates, so template logos in headers,
  slide masters and unused media never enter the pipeline.

Every candidate is inspected with ``pymupdf`` (pixel size, aspect ratio, pixel
entropy) and rejected when it looks like visual noise: tiny icons, thin rules or
banners, blank blocks, or the same small template image repeated many times.
Kept assets are written to ``ASSETS_DIR/<file_id>/`` with a ``manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from app.core.config import (
    ASSET_MAX_ASPECT_RATIO,
    ASSET_MIN_AREA,
    ASSET_MIN_BYTES,
    ASSET_MIN_HEIGHT,
    ASSET_MIN_INK_RATIO,
    ASSET_MIN_WIDTH,
    ASSET_OFFICE_MIN_DISPLAY,
    ASSET_OFFICE_MIN_DISPLAY_AREA,
    ASSET_PDF_DPI,
    ASSET_PDF_MIN_DISPLAY,
    ASSET_PDF_MIN_DISPLAY_AREA,
    ASSET_REPEAT_COUNT,
    ASSET_REPEAT_MAX_AREA,
    ASSETS_DIR,
)
from app.ingestion.identity import file_id_for

# Raster formats PyMuPDF can inspect. Office vector formats (EMF/WMF/SVG) are
# skipped: they are usually logos/icons and cannot be reliably measured here.
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp",
    ".jp2", ".j2k", ".pnm", ".pbm", ".pgm", ".ppm", ".pam",
}
ASSET_SOURCES = {".pdf", ".docx", ".pptx", ".md", ".markdown", ".html", ".htm"}

_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")

_SLIDE_RELS_RE = re.compile(r"ppt/slides/_rels/slide(\d+)\.xml\.rels$")
_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P_NS = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_WP_NS = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
_EMU_PER_INCH = 914_400


@dataclass(slots=True)
class _ImageInfo:
    width: int
    height: int
    entropy: float
    ink_ratio: float


@dataclass(slots=True)
class _Candidate:
    data: bytes
    ext: str
    source: str
    info: _ImageInfo
    page: int | None = None
    slide: int | None = None
    part: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    display_w: float | None = None
    display_h: float | None = None
    caption: str = ""
    index: int = 0
    digest: str = ""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pixel_metrics(pix) -> tuple[float, float]:
    """Return (grayscale entropy, ink ratio).

    Entropy is useful metadata. ``ink_ratio`` is the fraction of sampled pixels
    that differ from the dominant background colour, so a sparse line chart
    still passes while a truly blank/uniform block is rejected.
    """
    samples = pix.samples
    components = pix.n
    pixels = pix.width * pix.height
    if not pixels or not components:
        return 0.0, 0.0

    # Sample at most ~4096 pixels; enough to distinguish blank/flat from detail.
    stride = max(1, pixels // 4096)
    values: list[int] = []
    for pixel in range(0, pixels, stride):
        offset = pixel * components
        if components >= 3:
            red, green, blue = samples[offset], samples[offset + 1], samples[offset + 2]
            values.append((red * 299 + green * 587 + blue * 114) // 1000)
        else:
            values.append(samples[offset])

    counts = Counter(values)
    total = len(values)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    background = counts.most_common(1)[0][0]
    ink = sum(1 for value in values if abs(value - background) > 16) / total
    return entropy, ink


def _inspect_image(data: bytes) -> _ImageInfo | None:
    """Decode an image with PyMuPDF; ``None`` means an unsupported format."""
    try:
        import pymupdf

        pix = pymupdf.Pixmap(data)
    except Exception:
        return None
    entropy, ink_ratio = _pixel_metrics(pix)
    return _ImageInfo(
        width=pix.width,
        height=pix.height,
        entropy=entropy,
        ink_ratio=ink_ratio,
    )


def _reject_reason(
    candidate: _Candidate,
    occurrences: int,
) -> str | None:
    info = candidate.info
    width, height = info.width, info.height
    area = width * height

    if len(candidate.data) < ASSET_MIN_BYTES:
        return "tiny_file"
    if width < ASSET_MIN_WIDTH or height < ASSET_MIN_HEIGHT:
        return "too_small"
    if area < ASSET_MIN_AREA:
        return "too_small_area"
    aspect = max(width / height, height / width)
    if aspect > ASSET_MAX_ASPECT_RATIO:
        return "thin"
    if info.ink_ratio < ASSET_MIN_INK_RATIO:
        return "blank"

    if candidate.bbox:
        x0, y0, x1, y1 = candidate.bbox
        display_w, display_h = x1 - x0, y1 - y0
        if (
            display_w < ASSET_PDF_MIN_DISPLAY
            or display_h < ASSET_PDF_MIN_DISPLAY
            or display_w * display_h < ASSET_PDF_MIN_DISPLAY_AREA
        ):
            return "small_on_page"

    if candidate.display_w is not None and candidate.display_h is not None:
        if (
            candidate.display_w < ASSET_OFFICE_MIN_DISPLAY
            or candidate.display_h < ASSET_OFFICE_MIN_DISPLAY
            or candidate.display_w * candidate.display_h < ASSET_OFFICE_MIN_DISPLAY_AREA
        ):
            return "small_on_page"

    if occurrences >= ASSET_REPEAT_COUNT and area < ASSET_REPEAT_MAX_AREA:
        return "repeated"
    return None


# --------------------------------------------------------------------------- #
# PDF: layout picture boxes -> rendered PNG
# --------------------------------------------------------------------------- #
def _pdf_caption(page, boxes: list[dict], picture: dict) -> str:
    """Use the nearest caption box below/above a picture when one exists."""
    import pymupdf

    px0, py0, px1, py1 = (float(value) for value in picture["bbox"])
    best: tuple[float, str] | None = None
    for box in boxes:
        if box.get("class") != "caption":
            continue
        cx0, cy0, cx1, cy1 = (float(value) for value in box["bbox"])
        if min(px1, cx1) - max(px0, cx0) <= 0:
            continue
        if cy0 >= py1:
            distance = cy0 - py1
        elif cy1 <= py0:
            distance = py0 - cy1
        else:
            distance = 0.0
        if distance > 100:
            continue
        try:
            text = page.get_textbox(pymupdf.Rect(cx0, cy0, cx1, cy1))
        except Exception:
            continue
        text = " ".join(text.split())
        if not text:
            continue
        if best is None or distance < best[0]:
            best = (distance, text[:300])
    return best[1] if best else ""


def _pdf_candidates(path: Path) -> list[_Candidate]:
    try:
        import pymupdf
        import pymupdf4llm
    except ImportError:  # PDF support is optional at runtime
        return []

    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    if not isinstance(chunks, list):
        return []

    candidates: list[_Candidate] = []
    doc = pymupdf.open(str(path))
    try:
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            page_number = int(metadata.get("page_number") or 0)
            if page_number < 1 or page_number > doc.page_count:
                continue
            page = doc[page_number - 1]
            boxes = chunk.get("page_boxes") or []
            picture_index = 0
            for box in boxes:
                if box.get("class") != "picture":
                    continue
                bbox = tuple(float(value) for value in box["bbox"])
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                try:
                    pix = page.get_pixmap(
                        clip=pymupdf.Rect(bbox),
                        dpi=ASSET_PDF_DPI,
                        alpha=False,
                    )
                    data = pix.tobytes("png")
                except Exception:
                    continue
                entropy, ink_ratio = _pixel_metrics(pix)
                candidates.append(
                    _Candidate(
                        data=data,
                        ext=".png",
                        source="pdf",
                        info=_ImageInfo(
                            width=pix.width,
                            height=pix.height,
                            entropy=entropy,
                            ink_ratio=ink_ratio,
                        ),
                        page=page_number,
                        bbox=bbox,
                        caption=_pdf_caption(page, boxes, box),
                        index=picture_index,
                    )
                )
                picture_index += 1
    finally:
        doc.close()
    return candidates


# --------------------------------------------------------------------------- #
# DOCX / PPTX: ZIP media referenced by the main document / slides
# --------------------------------------------------------------------------- #
def _resolve_part(base_dir: str, target: str) -> str:
    from posixpath import join, normpath

    target = target.replace("\\", "/")
    if target.startswith("/"):
        return normpath(target.lstrip("/"))
    return normpath(join(base_dir, target))


def _image_rel_map(
    archive: zipfile.ZipFile,
    names: set[str],
    rels_name: str,
    base_dir: str,
) -> dict[str, str]:
    """Map relationship ids to image parts for one Office rels file."""
    if rels_name not in names:
        return {}
    try:
        root = ElementTree.fromstring(archive.read(rels_name))
    except Exception:
        return {}

    mapping: dict[str, str] = {}
    for relationship in root.iter(f"{_NS}Relationship"):
        if relationship.get("TargetMode") == "External":
            continue
        if not (relationship.get("Type") or "").endswith("/image"):
            continue
        target = relationship.get("Target")
        rel_id = relationship.get("Id")
        if not target or not rel_id:
            continue
        part = _resolve_part(base_dir, target)
        if part in names and Path(part).suffix.lower() in IMAGE_EXTENSIONS:
            mapping[rel_id] = part
    return mapping


def _extent_inches(extent) -> tuple[float, float] | None:
    try:
        cx = int(extent.get("cx") or 0)
        cy = int(extent.get("cy") or 0)
    except (TypeError, ValueError):
        return None
    if cx <= 0 or cy <= 0:
        return None
    return cx / _EMU_PER_INCH, cy / _EMU_PER_INCH


def _docx_references(
    archive: zipfile.ZipFile,
    names: set[str],
) -> list[tuple[str, float | None, float | None]]:
    """Image references in the main DOCX body (headers/footers excluded)."""
    references: list[tuple[str, float | None, float | None]] = []
    for owner in ("document", "footnotes", "endnotes"):
        xml_name = f"word/{owner}.xml"
        rels_name = f"word/_rels/{owner}.xml.rels"
        if xml_name not in names or rels_name not in names:
            continue
        rel_map = _image_rel_map(archive, names, rels_name, "word")
        if not rel_map:
            continue
        try:
            root = ElementTree.fromstring(archive.read(xml_name))
        except Exception:
            continue
        for drawing in root.iter(f"{_W_NS}drawing"):
            blip = next(drawing.iter(f"{_A_NS}blip"), None)
            if blip is None:
                continue
            rel_id = blip.get(f"{_R_NS}embed") or blip.get(f"{_R_NS}link")
            part = rel_map.get(rel_id or "")
            if not part:
                continue
            extent = next(drawing.iter(f"{_WP_NS}extent"), None)
            if extent is None:
                extent = next(drawing.iter(f"{_A_NS}ext"), None)
            display = _extent_inches(extent) if extent is not None else None
            references.append(
                (part, display[0] if display else None, display[1] if display else None)
            )
    return references


def _pptx_references(
    archive: zipfile.ZipFile,
    names: set[str],
) -> list[tuple[int, str, float | None, float | None]]:
    """Image references in slides (masters/layouts/template logos excluded)."""
    references: list[tuple[int, str, float | None, float | None]] = []
    for rels_name in sorted(name for name in names if _SLIDE_RELS_RE.match(name)):
        match = _SLIDE_RELS_RE.search(rels_name)
        if not match:
            continue
        slide = int(match.group(1))
        rel_map = _image_rel_map(archive, names, rels_name, "ppt/slides")
        if not rel_map:
            continue
        xml_name = f"ppt/slides/slide{slide}.xml"
        if xml_name not in names:
            continue
        try:
            root = ElementTree.fromstring(archive.read(xml_name))
        except Exception:
            continue
        for picture in root.iter(f"{_P_NS}pic"):
            blip = next(picture.iter(f"{_A_NS}blip"), None)
            if blip is None:
                continue
            rel_id = blip.get(f"{_R_NS}embed") or blip.get(f"{_R_NS}link")
            part = rel_map.get(rel_id or "")
            if not part:
                continue
            extent = next(picture.iter(f"{_A_NS}ext"), None)
            display = _extent_inches(extent) if extent is not None else None
            references.append(
                (slide, part, display[0] if display else None, display[1] if display else None)
            )
    return references


def _office_candidates(path: Path) -> list[_Candidate]:
    suffix = path.suffix.lower()
    candidates: list[_Candidate] = []
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        return candidates

    with archive:
        names = set(archive.namelist())
        if suffix == ".docx":
            references: list[tuple[int | None, str, float | None, float | None]] = [
                (None, part, width, height)
                for part, width, height in _docx_references(archive, names)
            ]
            if not references:
                # XML parsing failed / VML images: use the body rels only, so
                # header/footer logos still never enter.
                references = [
                    (None, part, None, None)
                    for rels_name in (
                        "word/_rels/document.xml.rels",
                        "word/_rels/footnotes.xml.rels",
                        "word/_rels/endnotes.xml.rels",
                    )
                    for part in _image_rel_map(archive, names, rels_name, "word").values()
                ]
        else:
            references = [
                (slide, part, width, height)
                for slide, part, width, height in _pptx_references(archive, names)
            ]
            if not references:
                # Slide rels only: never fall back to masters/layout media.
                references = []
                for rels_name in sorted(name for name in names if _SLIDE_RELS_RE.match(name)):
                    match = _SLIDE_RELS_RE.search(rels_name)
                    slide = int(match.group(1)) if match else None
                    references.extend(
                        (slide, part, None, None)
                        for part in _image_rel_map(archive, names, rels_name, "ppt/slides").values()
                    )

        for index, (slide, part, display_w, display_h) in enumerate(references):
            try:
                data = archive.read(part)
            except KeyError:
                continue
            info = _inspect_image(data)
            if info is None:
                continue
            candidates.append(
                _Candidate(
                    data=data,
                    ext=Path(part).suffix.lower(),
                    source="docx" if suffix == ".docx" else "pptx",
                    info=info,
                    slide=slide,
                    part=part,
                    display_w=display_w,
                    display_h=display_h,
                    index=index,
                )
            )
    return candidates


def _markup_candidates(path: Path) -> list[_Candidate]:
    """Local image references in Markdown / HTML documents.

    Only a local ``src`` that resolves to a real image file is considered; the
    regular size/area/entropy filters then decide whether it is informative.
    """
    refs: list[tuple[str, str]] = []  # (src, alt/caption)
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        for alt, src in _MARKDOWN_IMAGE_RE.findall(text):
            refs.append((src, alt.strip()))
    else:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
        for img in soup.find_all("img"):
            src = img.get("src") or ""
            if src:
                refs.append((src, (img.get("alt") or "").strip()))

    candidates: list[_Candidate] = []
    for index, (src, caption) in enumerate(refs):
        if not src or src.startswith(("http://", "https://", "data:", "#")):
            continue
        src_clean = src.split("#", 1)[0].split("?", 1)[0]
        candidate_path = Path(src_clean)
        if not candidate_path.is_absolute():
            candidate_path = path.parent / candidate_path
        if not candidate_path.is_file():
            continue
        ext = candidate_path.suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        try:
            data = candidate_path.read_bytes()
        except OSError:
            continue
        info = _inspect_image(data)
        if info is None:
            continue
        candidates.append(
            _Candidate(
                data=data,
                ext=ext,
                source=suffix.lstrip("."),
                info=info,
                caption=caption,
                index=index,
            )
        )
    return candidates


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _asset_name(candidate: _Candidate, digest: str) -> str:
    if candidate.source == "pdf":
        prefix = f"pdf-p{candidate.page:04d}-{candidate.index:02d}"
    elif candidate.source == "pptx" and candidate.slide is not None:
        prefix = f"pptx-s{candidate.slide:03d}-{candidate.index:02d}"
    else:
        prefix = f"{candidate.source}-{candidate.index:02d}"
    return f"{prefix}-{digest[:10]}{candidate.ext}"


def extract_assets(
    source_path: Path,
    rel_path: str,
    *,
    out_dir: Path | None = None,
) -> dict:
    """Extract and filter one document's images into ``out_dir/<file_id>/``.

    Returns the manifest (also written to ``manifest.json``). Unsupported
    documents simply produce an empty manifest.
    """
    source_path = Path(source_path)
    suffix = source_path.suffix.lower()
    file_id = file_id_for(rel_path)
    target_dir = Path(out_dir or ASSETS_DIR) / "images" / file_id

    if suffix not in ASSET_SOURCES:
        shutil.rmtree(target_dir, ignore_errors=True)
        return {
            "file_id": file_id,
            "path": rel_path,
            "source": suffix.lstrip("."),
            "candidates": 0,
            "kept": 0,
            "rejected": {},
            "assets": [],
        }
    if suffix == ".pdf":
        candidates = _pdf_candidates(source_path)
    elif suffix in (".docx", ".pptx"):
        candidates = _office_candidates(source_path)
    else:
        candidates = _markup_candidates(source_path)

    for candidate in candidates:
        candidate.digest = _sha256(candidate.data)
    occurrences = Counter(candidate.digest for candidate in candidates)

    kept: list[dict] = []
    rejected: Counter[str] = Counter()
    shutil.rmtree(target_dir, ignore_errors=True)

    for candidate in candidates:
        reason = _reject_reason(candidate, occurrences[candidate.digest])
        if reason:
            rejected[reason] += 1
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        name = _asset_name(candidate, candidate.digest)
        (target_dir / name).write_bytes(candidate.data)
        record = {
            "asset_id": Path(name).stem,
            "file_name": name,
            "path": f"images/{file_id}/{name}",
            "image_path": f"images/{file_id}/{name}",
            "source": candidate.source,
            "width": candidate.info.width,
            "height": candidate.info.height,
            "bytes": len(candidate.data),
            "format": candidate.ext.lstrip("."),
            "sha256": candidate.digest,
            "entropy": round(candidate.info.entropy, 3),
            "ink_ratio": round(candidate.info.ink_ratio, 4),
        }
        if candidate.page is not None:
            record["page"] = candidate.page
        if candidate.slide is not None:
            record["slide"] = candidate.slide
        if candidate.part:
            record["part"] = candidate.part
        if candidate.bbox:
            record["bbox"] = [round(value, 2) for value in candidate.bbox]
        if candidate.display_w is not None and candidate.display_h is not None:
            record["display_in"] = [
                round(candidate.display_w, 2),
                round(candidate.display_h, 2),
            ]
        if candidate.caption:
            record["caption"] = candidate.caption
        kept.append(record)

    manifest = {
        "file_id": file_id,
        "path": rel_path,
        "source": suffix.lstrip("."),
        "candidates": len(candidates),
        "kept": len(kept),
        "rejected": dict(rejected),
        "assets": kept,
    }
    if candidates:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        shutil.rmtree(target_dir, ignore_errors=True)
    return manifest


def main() -> None:
    import argparse

    from app.core.config import DOCS_DIR
    from app.indexing.scanner import iter_files

    parser = argparse.ArgumentParser(
        description="Extract and filter visual assets from PDF/DOCX/PPTX files."
    )
    parser.add_argument("--root", default=str(DOCS_DIR), help="Folder to scan")
    parser.add_argument("--out", default=str(ASSETS_DIR), help="Asset output folder")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    files = kept = candidates = 0
    for meta in iter_files(root):
        if meta.ext not in ASSET_SOURCES:
            continue
        try:
            manifest = extract_assets(Path(meta.abs_path), meta.rel_path, out_dir=out_dir)
        except Exception as error:
            print(f"warning: {meta.rel_path}: {error}")
            continue
        files += 1
        candidates += manifest["candidates"]
        kept += manifest["kept"]
        rejected = ", ".join(
            f"{reason}={count}" for reason, count in sorted(manifest["rejected"].items())
        )
        suffix = f" (rejected: {rejected})" if rejected else ""
        print(f"{meta.rel_path}: {manifest['kept']}/{manifest['candidates']} kept{suffix}")
    print(f"assets: {kept}/{candidates} kept from {files} files -> {out_dir}")


if __name__ == "__main__":
    main()
