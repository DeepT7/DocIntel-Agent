"""Normalise files of different formats into ordered blocks before chunking.

Instead of a flat string, extraction returns a list of blocks in document order:

    {"type": "heading", "level": int, "text": str}
    {"type": "text",    "text": str}
    {"type": "table",   "rows": [[str, ...], ...]}

Keeping tables as blocks lets the chunker (a) preserve their position, (b) treat
each table as one atomic unit, and (c) attach a textual caption so the table is
semantically retrievable.

Format support:
- .md/.markdown            -> parsed into headings / paragraphs / pipe tables
- .docx                    -> paragraphs + tables in true order (iter_inner_content)
- .pdf                     -> PyMuPDF4LLM markdown (headings + tables) then parsed
- .html/.htm               -> headings / paragraphs / tables
- everything else (.txt, code, ...) -> a single text block
"""

from pathlib import Path

MARKDOWN_PARSED = {".md", ".markdown"}


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

        if stripped.startswith("#"):
            flush_paragraph()
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append({"type": "heading", "level": level, "text": stripped.lstrip("#").strip()})
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

        if stripped.startswith("<!-- Start of picture text"):
            flush_paragraph()
            picture_lines: list[str] = []
            remainder = stripped[len("<!-- Start of picture text"):].replace("-->", "", 1)
            if remainder.strip():
                picture_lines.append(remainder)
            index += 1
            while index < len(lines):
                if "<!-- End of picture text -->" in lines[index]:
                    before = lines[index].split("<!-- End of picture text -->")[0]
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

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        paragraph.append(line)
        index += 1

    flush_paragraph()
    return blocks


# --------------------------------------------------------------------------- #
# Format-specific extractors
# --------------------------------------------------------------------------- #
def _blocks_docx(path: Path) -> list[dict]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    blocks: list[dict] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            style = (item.style.name or "").lower()
            if style.startswith("heading"):
                level = int("".join(ch for ch in style if ch.isdigit()) or "1")
                blocks.append({"type": "heading", "level": level, "text": text})
            elif style == "title":
                blocks.append({"type": "heading", "level": 1, "text": text})
            else:
                blocks.append({"type": "text", "text": text})
        elif isinstance(item, Table):
            rows = [[cell.text.strip() for cell in row.cells] for row in item.rows]
            rows = [row for row in rows if any(row)]
            if rows:
                blocks.append({"type": "table", "rows": rows})
    return blocks


def _blocks_pdf(path: Path) -> list[dict]:
    import pymupdf4llm

    return _blocks_from_markdown(pymupdf4llm.to_markdown(str(path)))


def _blocks_html(path: Path) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()

    blocks: list[dict] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
        if element.name == "table":
            rows = [
                [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                for row in element.find_all("tr")
            ]
            rows = [row for row in rows if any(row)]
            if rows:
                blocks.append({"type": "table", "rows": rows})
        elif element.name.startswith("h"):
            text = element.get_text(" ", strip=True)
            if text:
                blocks.append({"type": "heading", "level": int(element.name[1]), "text": text})
        else:
            text = element.get_text(" ", strip=True)
            if text:
                blocks.append({"type": "text", "text": text})
    return blocks


def _blocks_text(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [{"type": "text", "text": text}] if text else []


def extract_blocks(path: Path) -> list[dict]:
    """Return ordered blocks (headings, text, tables) for a file."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _blocks_docx(path)
    if suffix == ".pdf":
        return _blocks_pdf(path)
    if suffix in {".html", ".htm"}:
        return _blocks_html(path)
    if suffix in MARKDOWN_PARSED:
        return _blocks_from_markdown(path.read_text(encoding="utf-8", errors="replace"))
    return _blocks_text(path)


def extract(path: Path) -> str:
    """Compatibility helper: flatten blocks back to markdown-ish text."""
    parts: list[str] = []
    for block in extract_blocks(path):
        if block["type"] == "heading":
            parts.append("#" * block["level"] + " " + block["text"])
        elif block["type"] == "table":
            parts.append(table_markdown(block["rows"]))
        else:
            parts.append(block["text"])
    return "\n\n".join(parts)