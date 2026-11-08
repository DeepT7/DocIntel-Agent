"""Convert supported documents to canonical Markdown, then to ordered blocks.

Every format is first normalised to one canonical Markdown string; a single
parser then turns that Markdown into ordered blocks:

    {"type": "heading", "level": int, "text": str}
    {"type": "text",    "text": str}
    {"type": "table",   "rows": [[str, ...], ...]}
    {"type": "picture", "text": str}

Canonicalisation per format:
- ``.md`` / ``.markdown``          -> used as-is (already canonical)
- ``.html`` / ``.htm``             -> boilerplate removed, then MarkItDown
- ``.docx`` / ``.pptx`` / ``.txt`` -> MarkItDown (Microsoft)
- ``.pdf``                         -> pymupdf4llm (PyMuPDF)
- anything else                    -> rejected (the scanner never enqueues it)

Keeping tables as blocks lets the chunker (a) preserve their position,
(b) treat each table as one atomic unit, and (c) attach a textual caption so
the table is semantically retrievable.
"""

from __future__ import annotations

import re
from io import BytesIO
from functools import lru_cache
from pathlib import Path

MARKDOWN_SOURCES = {".md", ".markdown"}
HTML_SOURCES = {".html", ".htm"}
MARKITDOWN_SOURCES = {".docx", ".pptx", ".txt"}

_PICTURE_START = "<!-- Start of picture text"
_PICTURE_END = "<!-- End of picture text -->"
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_FENCE_RE = re.compile(r"(?ms)^(```|~~~)[^\n]*\n.*?^\1[^\n]*$")


# --------------------------------------------------------------------------- #
# Table helpers
# --------------------------------------------------------------------------- #
def table_markdown(rows: list[list[str]]) -> str:
    """Render rows as a GitHub-style markdown table."""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(width)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator, *body])


def table_caption(rows: list[list[str]]) -> str:
    """A short natural-language description that embeds far better than a raw table."""
    if not rows:
        return "Table."
    columns = ", ".join(cell for cell in rows[0] if cell)
    return f"Table with columns: {columns}. Rows: {max(len(rows) - 1, 0)}."


def _is_separator(cells: list[str]) -> bool:
    return all(cell and set(cell) <= set("-: ") for cell in cells)


def _parse_markdown_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if _is_separator(cells):
            continue
        if any(cells):
            rows.append(cells)
    return rows


# --------------------------------------------------------------------------- #
# Canonical Markdown
# --------------------------------------------------------------------------- #
def _normalize_markdown(text: str) -> str:
    """Make converter output stable before it is parsed into blocks.

    MarkItDown/PDF output can contain HTML comments, image references, ``<br>``
    tags and non-breaking spaces. They add noise to chunk text, so they are
    removed here. Fenced code and PyMuPDF4LLM's picture markers are protected
    first so their contents (including HTML comments in code samples) survive.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    fences: list[str] = []

    def protect_fence(match: re.Match[str]) -> str:
        fences.append(match.group(0))
        return f"\x00FENCE{len(fences) - 1}\x00"

    text = _FENCE_RE.sub(protect_fence, text)
    text = text.replace(_PICTURE_START, "\x00PICTURE_START\x00")
    text = text.replace(_PICTURE_END, "\x00PICTURE_END\x00")

    text = _HTML_COMMENT_RE.sub("", text)
    text = _IMAGE_RE.sub("", text)
    text = _BR_RE.sub(" ", text)
    text = text.replace("\xa0", " ").replace("&nbsp;", " ")

    text = text.replace("\x00PICTURE_START\x00", _PICTURE_START)
    text = text.replace("\x00PICTURE_END\x00", _PICTURE_END)
    for index, fence in enumerate(fences):
        text = text.replace(f"\x00FENCE{index}\x00", fence)

    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


@lru_cache(maxsize=1)
def _markitdown():
    """One MarkItDown instance per process (it loads Magika once)."""
    from markitdown import MarkItDown

    return MarkItDown()


def _markitdown_html(path: Path) -> str:
    """Strip page boilerplate, then let MarkItDown convert the HTML body."""
    from bs4 import BeautifulSoup
    from markitdown import StreamInfo

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()
    cleaned = str(soup).encode("utf-8")
    result = _markitdown().convert_stream(
        BytesIO(cleaned),
        stream_info=StreamInfo(extension=".html", mimetype="text/html", charset="utf-8"),
    )
    return result.markdown or ""


def canonical_markdown(path: Path) -> str:
    """Convert one supported document to canonical Markdown."""
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_SOURCES:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        import pymupdf4llm

        text = pymupdf4llm.to_markdown(str(path))
    elif suffix in HTML_SOURCES:
        text = _markitdown_html(path)
    elif suffix in MARKITDOWN_SOURCES:
        text = _markitdown().convert(str(path)).markdown or ""
    else:
        raise ValueError(f"unsupported document format: {suffix or path.name}")
    return _normalize_markdown(text)


# --------------------------------------------------------------------------- #
# Markdown -> blocks
# --------------------------------------------------------------------------- #
def _blocks_from_markdown(text: str) -> list[dict]:
    blocks: list[dict] = []
    lines = text.splitlines()
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            joined = "\n".join(paragraph).strip()
            if joined:
                blocks.append({"type": "text", "text": joined})
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        # Fenced code is kept as text, so a "# comment" inside it is not a heading.
        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush_paragraph()
            fence = stripped[:3]
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith(fence):
                code_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            code = "\n".join(code_lines).strip()
            if code:
                blocks.append({"type": "text", "text": code})
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip().rstrip("#").strip()
            if title:
                blocks.append({"type": "heading", "level": level, "text": title})
            index += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = _parse_markdown_table(table_lines)
            if rows:
                blocks.append({"type": "table", "rows": rows})
            continue

        if stripped.startswith(_PICTURE_START):
            flush_paragraph()
            picture_lines: list[str] = []
            remainder = stripped[len(_PICTURE_START):].replace("-->", "", 1)
            if remainder.strip():
                picture_lines.append(remainder)
            index += 1
            while index < len(lines):
                if _PICTURE_END in lines[index]:
                    before = lines[index].split(_PICTURE_END)[0]
                    if before.strip():
                        picture_lines.append(before)
                    index += 1
                    break
                picture_lines.append(lines[index])
                index += 1
            content = "\n".join(picture_lines).replace("<br>", "\n").strip()
            if content:
                blocks.append({"type": "picture", "text": content})
            continue

        # HTML comments are dropped whole, unless they are picture markers.
        if stripped.startswith("<!--"):
            if "-->" in stripped:
                index += 1
                continue
            index += 1
            while index < len(lines) and "-->" not in lines[index]:
                index += 1
            if index < len(lines):
                index += 1
            continue
        if "<!--" in line:
            line = _HTML_COMMENT_RE.sub("", line)
            stripped = line.strip()
            if not stripped:
                index += 1
                continue

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        paragraph.append(line)
        index += 1

    flush_paragraph()
    return blocks


def extract_blocks(path: Path) -> list[dict]:
    """Return ordered blocks (headings, text, tables) for a supported file."""
    return _blocks_from_markdown(canonical_markdown(path))
