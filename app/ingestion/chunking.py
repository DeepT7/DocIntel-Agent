"""Turn extracted document blocks into embedded chunk records.

Tables and images are no longer embedded as raw text. This chunker emits
compact *stub* chunks (``table_stub`` / ``image_stub``) that carry the semantic
context needed for retrieval (columns, key row entities, caption), while the
full raw content is stored on disk under the local asset store
(``data/index/assets``) keyed by a deterministic ``asset_id``. A later Agent
tool reads the raw content back via that id.
"""

import re
from pathlib import Path

from app.core.config import BREADCRUMB_MAX_DEPTH, CHUNK_MAX_CHARS, CHUNK_TARGET_CHARS
from app.core.scope import normalize_root
from app.core.sections import clean_heading, split_section_number
from app.ingestion.assets import extract_assets
from app.ingestion.extract import extract_blocks, table_caption
from app.ingestion.identity import file_id_for
from app.ingestion.raw_assets import TABLES_DIRNAME, image_id_for, save_tables, table_id_for

MAX_ROW_KEYS = 40


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


TABLE_RE = re.compile(r"^\s*\**\s*(table|bảng)\s*([A-Za-z]?\.?\d+(?:\.\d+)*[a-z]?)\s*[:.\-]", re.IGNORECASE)
FIGURE_RE = re.compile(r"^\s*\**\s*(figure|hình)\s*([A-Za-z]?\.?\d+(?:\.\d+)*[a-z]?)\s*[:.\-]", re.IGNORECASE)
_PAGE_MARKER_RE = re.compile(r"^\s*(?:page\s*\d+|\d+|\(\d+\s+of\s+\d+\)|continued)\s*$", re.IGNORECASE)


def _is_page_marker(text: str) -> bool:
    """True when a text block is a page number / header / footer separator."""
    stripped = text.strip()
    if not stripped:
        return True
    return bool(_PAGE_MARKER_RE.match(stripped))


def _adjacent_caption(blocks: list[dict], index: int) -> tuple[int, str, str, str] | None:
    """Return (caption_index, text, kind, number) for a caption next to index."""
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


def _table_caption_index(blocks: list[dict], index: int) -> int | None:
    """Return the index of the adjacent author caption of a table, if any."""
    for neighbour in (index + 1, index - 1):
        if neighbour < 0 or neighbour >= len(blocks):
            continue
        other = blocks[neighbour]
        if other["type"] == "text" and TABLE_RE.match(other["text"].strip()):
            return neighbour
    return None


def _merge_page_break_tables(blocks: list[dict]) -> list[dict]:
    """Merge tables split across page breaks into one table.

    Two ``table`` blocks are merged when they are adjacent (or separated by a
    short page marker), the second one has no author caption of its own, and it
    shares the first table's column count. A repeated header row is dropped.
    """
    result: list[dict] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if block["type"] != "table":
            result.append(block)
            index += 1
            continue

        rows = [list(row) for row in block.get("rows", [])]
        while index + 1 < len(blocks):
            nxt = blocks[index + 1]
            if nxt["type"] == "table":
                if _table_caption_index(blocks, index + 1) is not None:
                    break
                nrows = nxt.get("rows", [])
                if not rows or not nrows:
                    break
                if len(nrows[0]) != len(rows[0]):
                    break
                if nrows[0] == rows[0]:
                    nrows = nrows[1:]
                rows.extend(list(row) for row in nrows)
                index += 1
                continue
            if nxt["type"] == "text" and _is_page_marker(nxt["text"]):
                # Skip the separator only when it is followed by a mergeable table.
                if index + 2 < len(blocks) and blocks[index + 2]["type"] == "table":
                    ahead = blocks[index + 2]
                    ahead_rows = ahead.get("rows") or []
                    if (
                        _table_caption_index(blocks, index + 2) is None
                        and rows
                        and ahead_rows
                        and len(ahead_rows[0]) == len(rows[0])
                    ):
                        index += 1
                        continue
            break

        merged = dict(block)
        merged["rows"] = rows
        result.append(merged)
        index += 1
    return result


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


def _table_stub_text(
    table_id: str,
    caption: str,
    headers: list[str],
    row_keys: list[str],
    row_count: int,
) -> str:
    return "\n".join(
        [
            f"[TABLE STUB | ID: {table_id}]",
            f"Caption: {caption}",
            f"Columns: {', '.join(headers)}",
            f"Row Entities: {', '.join(row_keys)}",
            f"Total Rows: {row_count}",
        ]
    )


def _image_stub_text(
    image_id: str,
    image_path: str,
    caption: str,
) -> str:
    lines = [
        f"[IMAGE STUB | ID: {image_id} | Path: {image_path}]",
    ]
    if caption:
        lines.append(f"Caption: {caption}")
    return "\n".join(lines)


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
    """One embedded chunk, prefixed with a contextual header."""
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
        "section_heading": raw_heading,
        "section_level": section.get("level", 1),
        "section_depth": section.get("depth", len(section_depths) or 1),
        "breadcrumbs": section_titles or [section_title],
        "breadcrumb_levels": section_depths or [1],
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
    """Build the tree node for a heading block and return (node, top_count)."""
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


def build_chunks_for_file(root: Path, path: Path, assets_dir: Path | None = None) -> list[dict]:
    rel_path = path.relative_to(root).as_posix()
    file_id = file_id_for(rel_path)
    root_str = normalize_root(root)
    title = path.stem
    file_name = path.name

    blocks = _attach_table_captions(_merge_page_break_tables(extract_blocks(path)))

    chunks: list[dict] = []
    seq = 0
    stack: list[dict] = []
    top_count = 0
    used_ids: set[str] = {"s0"}
    overview = {
        "id": "s0",
        "title": "Overview",
        "number": "",
        "raw_level": 0,
        "depth": 1,
    }
    body: list[str] = []
    tables_manifest: list[dict] = []
    table_index = 0
    pic_index = 0

    def section_info(chunk_type: str = "text", raw: str | None = None) -> dict:
        nodes = stack or [overview]
        current = nodes[-1]
        titles = [node["title"] for node in nodes]
        return {
            "id": current["id"],
            "ancestors": [node["id"] for node in nodes],
            "titles": titles,
            "depths": [node["depth"] for node in nodes],
            "numbers": [node.get("number", "") for node in nodes],
            "title": current["title"],
            "raw": raw or current["title"],
            "level": current["raw_level"],
            "depth": current["depth"],
            "chunk_type": chunk_type,
        }

    def breadcrumb_of(section: dict) -> str:
        header_titles = (section.get("titles") or [])[:BREADCRUMB_MAX_DEPTH]
        header_titles = header_titles or [section.get("title") or "Overview"]
        return " > ".join(header_titles)

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

    for block in blocks:
        if block["type"] == "heading":
            emit_body()
            node, top_count = _section_for_heading(stack, block, used_ids, top_count)
            stack.append(node)
        elif block["type"] in ("text", "picture"):
            body.append(block["text"])
        elif block["type"] == "table":
            emit_body()
            seq += 1
            current_title = stack[-1]["title"] if stack else overview["title"]
            section = section_info("table_stub", raw=f"{current_title} (Table)")
            rows = block.get("rows") or []
            real_caption = block.get("caption")
            caption = real_caption or table_caption(rows)
            if rows:
                headers = [str(cell) for cell in rows[0]]
                row_keys = [
                    str(row[0]).strip()
                    for row in rows[1:]
                    if row and str(row[0]).strip()
                ][:MAX_ROW_KEYS]
                row_count = max(len(rows) - 1, 0)
            else:
                headers = []
                row_keys = []
                row_count = 0
            table_id = table_id_for(file_id, section["id"], table_index)
            breadcrumb = breadcrumb_of(section)
            if rows:
                part = _table_stub_text(
                    table_id, caption, headers, row_keys, row_count
                )
            else:
                content = block.get("content") or ""
                part = "\n\n".join(
                    p
                    for p in [
                        f"[TABLE STUB | ID: {table_id}]",
                        f"Caption: {caption}",
                        content,
                    ]
                    if p
                )
            tables_manifest.append(
                {
                    "table_id": table_id,
                    "file_id": file_id,
                    "path": rel_path,
                    "section_id": section["id"],
                    "breadcrumb": breadcrumb,
                    "caption": caption,
                    "table_number": block.get("table_number") or "",
                    "headers": headers,
                    "rows": rows,
                    "content": block.get("content") or "",
                    "row_count": row_count,
                    "col_count": len(headers),
                }
            )
            extra = {
                "asset_id": table_id,
                "asset_path": f"{TABLES_DIRNAME}/{file_id}.json",
                "caption": caption,
                "headers": headers,
                "row_count": row_count,
            }
            if block.get("table_number"):
                extra["table_number"] = block["table_number"]
            chunks.append(
                _chunk_record(
                    file_id, root_str, rel_path, file_name, title,
                    section, part, seq, extra,
                )
            )
            body.append(f"[Ref Table: {table_id} - {caption}]")
            table_index += 1

    emit_body()

    try:
        manifest = extract_assets(path, rel_path, out_dir=assets_dir)
        images = manifest.get("assets") or []
    except Exception:
        images = []

    overview_section = {
        "id": "s0",
        "ancestors": ["s0"],
        "titles": ["Overview"],
        "depths": [1],
        "numbers": [""],
        "title": "Overview",
        "raw": "Overview",
        "level": 0,
        "depth": 1,
        "chunk_type": "image_stub",
    }
    for image in images:
        image_path = image.get("path") or image.get("image_path") or ""
        caption = image.get("caption") or ""
        image_id = image_id_for(file_id, "s0", pic_index)
        seq += 1
        part = _image_stub_text(image_id, image_path, caption)
        extra = {
            "asset_id": image_id,
            "asset_path": image_path,
            "caption": caption,
            "headers": [],
        }
        chunks.append(
            _chunk_record(
                file_id, root_str, rel_path, file_name, title,
                overview_section, part, seq, extra,
            )
        )
        body.append(f"[Ref Image: {image_id} - {caption or image_id}]")
        pic_index += 1

    emit_body()

    save_tables(file_id, tables_manifest, out_dir=assets_dir)
    return chunks