"""Turn extracted document blocks into embedded chunk records.

The chunker walks blocks in document order, maintains the section tree and
emits one record per text part or per atomic table. Records carry both the
embedded text (with a contextual header) and the retrieval metadata.
"""

import re
from pathlib import Path

from app.core.config import BREADCRUMB_MAX_DEPTH, CHUNK_MAX_CHARS, CHUNK_TARGET_CHARS
from app.core.scope import normalize_root
from app.core.sections import clean_heading, split_section_number
from app.ingestion.extract import extract_blocks, table_caption, table_markdown
from app.ingestion.identity import file_id_for


def _split_text(text: str, target_chars: int, max_chars: int) -> list[str]:
    """Greedily pack paragraphs up to target_chars, hard-splitting long ones."""
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= target_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        for start in range(0, len(paragraph), max_chars):
            chunks.append(paragraph[start : start + max_chars])

    if current:
        chunks.append(current)
    return chunks


# An author-written caption, e.g. "Table 3: Results", "Table 2.1: ...", "Table A.1:",
# "Figure 3.1:", "Bảng 3:". pymupdf4llm puts text taken from a picture (chart axis
# labels, a boxed table, ...) between "<!-- Start of picture text -->" markers;
# the caption right after tells us whether that region is a table (keep) or a
# figure (drop): figure text is meaningless without the image.
# A trailing ":" "." or "-" is required so a reference like "Table 3 shows" does
# not match. The number may be dotted (2.1, A.1, 3.2a) and is kept whole.
TABLE_RE = re.compile(r"^\s*\**\s*(table|bảng)\s*([A-Za-z]?\.?\d+(?:\.\d+)*[a-z]?)\s*[:.\-]", re.IGNORECASE)
FIGURE_RE = re.compile(r"^\s*\**\s*(figure|hình)\s*([A-Za-z]?\.?\d+(?:\.\d+)*[a-z]?)\s*[:.\-]", re.IGNORECASE)


def _adjacent_caption(blocks: list[dict], index: int) -> tuple[int, str, str, str] | None:
    """Return (caption_index, text, kind, number) for a caption next to index.

    The caption below the region is checked before the one above: a figure is
    usually followed by its own caption and may itself follow an unrelated table
    caption. ``kind`` is "table" or "figure".
    """
    for neighbour in (index + 1, index - 1):
        if neighbour < 0 or neighbour >= len(blocks):
            continue
        other = blocks[neighbour]
        if other["type"] != "text":
            continue
        text = other["text"].strip()
        table_match = TABLE_RE.match(text)
        if table_match:
            return neighbour, text, "table", table_match.group(2)
        figure_match = FIGURE_RE.match(text)
        if figure_match:
            return neighbour, text, "figure", figure_match.group(2)
    return None


def _attach_table_captions(blocks: list[dict]) -> list[dict]:
    """Resolve picture-text regions and attach author captions to tables.

    - A table with an adjacent author caption gets that caption (primary signal).
    - A picture-text region is kept only when its caption says "Table N" (its
      text is the table content); figure/chart text is dropped.
    - The caption block is removed from the body so it is not duplicated.
    """
    consumed: set[int] = set()
    promotions: dict[int, tuple[str, str]] = {}
    for index, block in enumerate(blocks):
        if block["type"] not in ("table", "picture"):
            continue
        found = _adjacent_caption(blocks, index)
        if not found:
            continue
        caption_index, caption, kind, number = found
        if kind != "table" or caption_index in consumed:
            continue
        consumed.add(caption_index)
        promotions[index] = (caption, number)

    output: list[dict] = []
    for index, block in enumerate(blocks):
        if index in consumed:
            continue
        if block["type"] == "picture":
            # Keep only regions promoted to a table by a "Table N" caption.
            if index in promotions:
                caption, number = promotions[index]
                output.append(
                    {
                        "type": "table",
                        "rows": [],
                        "content": block["text"],
                        "caption": caption,
                        "table_number": number,
                    }
                )
            continue
        if index in promotions:
            caption, number = promotions[index]
            block = {**block, "caption": caption, "table_number": number}
        output.append(block)
    return output


def _chunk_record(
    file_id: str,
    root: str,
    rel_path: str,
    file_name: str,
    title: str,
    section: dict,
    part: str,
    seq: int,
    extra: dict | None = None,
) -> dict:
    """One embedded chunk, prefixed with a contextual header.

    ``section`` is the materialized path produced by ``build_chunks_for_file``:
    ``id``, ``ancestors``, ``titles``, ``depths``, ``numbers``, ``title``,
    ``raw``, ``level`` and ``chunk_type``. The contextual header uses the clean
    titles (up to level 3), so the embedded text no longer contains Markdown
    stars or a synthetic ``(Table)`` suffix.
    """
    section_titles = list(section.get("titles") or [])
    section_depths = list(section.get("depths") or [])
    section_numbers = list(section.get("numbers") or [])
    section_title = section.get("title") or "Overview"
    raw_heading = section.get("raw") or section_title
    chunk_type = section.get("chunk_type", "text")
    header_titles = section_titles[:BREADCRUMB_MAX_DEPTH] or [section_title]
    breadcrumb = " > ".join(header_titles)
    header = f"[Document: {file_name} | Section: {breadcrumb}]"
    text = f"{header}\n\n{part}".strip()
    record = {
        "chunk_id": f"{file_id}:{seq}",
        "file_id": file_id,
        "root": root,
        "path": rel_path,
        "url": rel_path,
        "title": title,
        "file_name": file_name,
        # Display/evaluation metadata. Retrieval filters on section_id(s).
        "section_heading": raw_heading,
        "section_level": section.get("level", 1),
        "section_depth": section.get("depth", len(section_depths) or 1),
        "breadcrumbs": section_titles or [section_title],
        "breadcrumb_levels": section_depths or [1],
        # Canonical materialized path (the retrieval contract).
        "section_id": section.get("id") or "s0",
        "section_ancestors": list(section.get("ancestors") or ["s0"]),
        "section_titles": section_titles or [section_title],
        "section_depths": section_depths or [1],
        "section_numbers": section_numbers or [""],
        "section_title_clean": section_title,
        "section_number": section_numbers[-1] if section_numbers else "",
        "raw_heading": raw_heading,
        "chunk_type": chunk_type,
        "context_header": header,
        "doc_type": "local",
        "char_count": len(text),
        "text": text,
    }
    if extra:
        record.update(extra)
    return record


def _section_for_heading(stack: list[dict], block: dict, used_ids: set[str], top_count: int) -> tuple[dict, int]:
    """Build the tree node for a heading block and return (node, top_count).

    A new heading pops deeper/equal raw levels. The node id is a hybrid path:

    - a numbered heading under a numbered parent keeps its printed path
      ("4" -> "s4", "4.1" -> "s4.1"), which is the intuitive id;
    - a numbered list item under an unnumbered wrapper (e.g. the NeurIPS
      checklist) gets a structural child id, so it can never collide with the
      real section 4;
    - unnumbered headings get structural/``x`` ids.

    Duplicate printed paths keep their own id via a "-2" suffix.
    """
    raw_heading = block["text"]
    clean = clean_heading(raw_heading) or raw_heading
    number, _ = split_section_number(clean)
    level = int(block.get("level") or 1)
    while stack and stack[-1]["raw_level"] >= level:
        stack.pop()
    parent = stack[-1] if stack else None

    if parent:
        depth = parent["depth"] + 1
        parent_number = parent.get("number") or ""
        consistent = bool(
            number
            and parent_number
            and (number == parent_number or number.startswith(parent_number + "."))
        )
        if consistent:
            section_id = f"s{number}"
        else:
            parent["x_count"] = parent.get("x_count", 0) + 1
            section_id = f"{parent['id']}.x{parent['x_count']}"
    else:
        depth = 1
        if number:
            section_id = f"s{number}"
        else:
            top_count += 1
            section_id = f"x{top_count}"

    base_id = section_id
    suffix = 2
    while section_id in used_ids:
        section_id = f"{base_id}-{suffix}"
        suffix += 1
    used_ids.add(section_id)

    node = {
        "id": section_id,
        "title": clean,
        "number": number,
        "raw_level": level,
        "depth": depth,
        "x_count": 0,
    }
    return node, top_count


def build_chunks_for_file(root: Path, path: Path) -> list[dict]:
    rel_path = path.relative_to(root).as_posix()
    file_id = file_id_for(rel_path)
    root_str = normalize_root(root)
    title = path.stem
    file_name = path.name

    chunks: list[dict] = []
    seq = 0
    stack: list[dict] = []
    top_count = 0
    used_ids: set[str] = {"s0"}  # s0 is reserved for the pre-heading Overview
    overview = {
        "id": "s0",
        "title": "Overview",
        "number": "",
        "raw_level": 0,
        "depth": 1,
    }
    body: list[str] = []

    def section_info(chunk_type: str = "text", raw: str | None = None) -> dict:
        nodes = stack or [overview]
        current = nodes[-1]
        titles = [node["title"] for node in nodes]
        return {
            "id": current["id"],
            "ancestors": [node["id"] for node in nodes],
            "titles": titles,
            "depths": [node["depth"] for node in nodes],
            "numbers": [node["number"] for node in nodes],
            "title": current["title"],
            "raw": raw or current["title"],
            "level": current["raw_level"],
            "depth": current["depth"],
            "chunk_type": chunk_type,
        }

    def emit_body() -> None:
        nonlocal seq, body
        text = "\n\n".join(body).strip()
        body = []
        if not text:
            return
        section = section_info("text")
        for part in _split_text(text, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS):
            seq += 1
            chunks.append(
                _chunk_record(
                    file_id, root_str, rel_path, file_name, title,
                    section, part, seq,
                )
            )

    for block in _attach_table_captions(extract_blocks(path)):
        if block["type"] == "heading":
            emit_body()
            node, top_count = _section_for_heading(stack, block, used_ids, top_count)
            stack.append(node)
        elif block["type"] in ("text", "picture"):
            body.append(block["text"])
        elif block["type"] == "table":
            # A table is one atomic chunk (never split). It shares its section's
            # id, so asking for that section returns the prose AND the table. The
            # synthetic "(Table)" suffix stays display-only (section_heading).
            emit_body()
            seq += 1
            current_title = stack[-1]["title"] if stack else overview["title"]
            section = section_info("table", raw=f"{current_title} (Table)")
            real_caption = block.get("caption")
            caption = real_caption or table_caption(block["rows"])
            if block["rows"]:
                body_text = table_markdown(block["rows"])
            else:
                body_text = block.get("content", "")  # single-cell text-box table
            content_parts: list[str] = []
            if not real_caption:
                content_parts.append(current_title)
            content_parts.extend([caption, body_text])
            part = "\n\n".join(p for p in content_parts if p).strip()
            extra = {"caption": caption}
            if block.get("table_number"):
                extra["table_number"] = block["table_number"]
            chunks.append(
                _chunk_record(
                    file_id, root_str, rel_path, file_name, title,
                    section, part, seq, extra,
                )
            )

    emit_body()
    return chunks
