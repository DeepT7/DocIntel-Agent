"""Query-time access to the raw table store.

Ingestion keeps the compact ``table_stub`` in Qdrant but writes the full cell
matrix to ``data/index/assets/tables/<file_id>.json``. This module closes the
loop:

* parse explicit table references ("Table 2", "Bảng 3.1") from the user message;
* hydrate retrieved ``table_stub`` chunks with the raw Markdown table before the
  answer prompt is built (rows-empty / picture-derived tables keep their stub
  text, which already embeds the content);
* expose ``read_raw_table`` as a bounded "tool" the answer step can call once to
  fetch tables that were not retrieved or that were truncated inline;
* describe the available tables so the Document Card and the tool prompt share
  one manifest.

Nothing here mutates the index: hydration only produces new chunk dicts, so the
graph stays safe to retry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.core.config import (
    MAX_TOKENS,
    OPENROUTER_UTILITY_MODELS,
    TABLE_MAX_CHARS,
    TABLE_MAX_ROWS,
    TABLE_TOOL_CONTEXT_CHARS,
    TABLE_TOOL_MAX_TABLES,
)
from app.core.prompts import build_table_tool_prompt
from app.core.scope import normalize_root
from app.ingestion.extract import table_markdown
from app.ingestion.raw_assets import read_raw_table
from app.reasoning.answer import _generate

# "Table 2", "Table 2.1", "Bảng A.1", "table 3a". The number shape matches the
# caption regex used at ingestion (chunking.TABLE_RE), so payload values line up.
_TABLE_REF_RE = re.compile(
    r"(?<![\w-])(?:table|bảng|bang)\s*([A-Za-z]?\.?\d+(?:\.\d+)*[a-z]?)(?![\w])",
    re.IGNORECASE,
)

_TABLE_ID_RE = re.compile(r"[0-9a-f]{32}_[A-Za-z0-9.\-]+_t\d+")

# Describes the one tool the answer step may call. This is the advertised
# interface for provider-level tool calling; the utility-model selector below
# follows the same contract (``{"table_id": str}`` per call), so the two paths
# cannot drift.
READ_RAW_TABLE_TOOL = {
    "name": "read_raw_table",
    "description": (
        "Return one table's full raw content (headers + rows) by table id. "
        "Use it when a retrieved table stub is truncated or a needed table was "
        "not retrieved."
    ),
    "parameters": {
        "type": "object",
        "properties": {"table_id": {"type": "string"}},
        "required": ["table_id"],
    },
}


def _number_variants(number: str) -> list[str]:
    """Case variants so "table a.1" matches a stored "A.1" caption."""
    value = (number or "").strip().rstrip(".")
    if not value:
        return []
    return list(dict.fromkeys([value, value.upper(), value.lower()]))


def extract_table_references(query: str) -> list[str]:
    """Distinct table numbers named by the message, in order and de-duplicated."""
    numbers: list[str] = []
    seen: set[str] = set()
    for match in _TABLE_REF_RE.finditer(query or ""):
        for variant in _number_variants(match.group(1)):
            if variant not in seen:
                seen.add(variant)
                numbers.append(variant)
    return numbers


def render_table_markdown(
    table: dict | None,
    *,
    max_rows: int = TABLE_MAX_ROWS,
    max_chars: int = TABLE_MAX_CHARS,
) -> str:
    """Render raw rows as Markdown, truncating very large tables."""
    if not table:
        return ""
    rows = table.get("rows") or []
    if not rows:
        return ""

    body_rows = max(len(rows) - 1, 0)
    limit = max(1, max_rows)
    shown = rows[: limit + 1]  # header + limited body
    markdown = table_markdown(shown)
    if body_rows > limit:
        remaining = body_rows - limit
        markdown += (
            f"\n\n_(Showing {limit} of {body_rows} rows; "
            f"{remaining} more rows available via table id "
            f"`{table.get('table_id', '')}`.)_"
        )
    if len(markdown) > max_chars:
        cut = markdown[:max_chars]
        if "\n" in cut:
            cut = cut.rsplit("\n", 1)[0]
        markdown = cut + (
            f"\n\n_(Table truncated for length; full content available via "
            f"table id `{table.get('table_id', '')}`.)_"
        )
    return markdown


def hydrate_table_stubs(
    chunks: list[dict],
    *,
    max_rows: int = TABLE_MAX_ROWS,
    max_chars: int = TABLE_MAX_CHARS,
    out_dir: Path | None = None,
) -> list[dict]:
    """Append the raw Markdown table to every retrieved ``table_stub``.

    ``rows == []`` means the ingestion kept the content in the stub itself
    (picture-derived tables, for example), so the stub text is left untouched.
    """
    hydrated: list[dict] = []
    for chunk in chunks:
        if chunk.get("chunk_type") != "table_stub":
            hydrated.append(chunk)
            continue
        table_id = str(chunk.get("asset_id") or "").strip()
        if not table_id:
            hydrated.append(chunk)
            continue
        try:
            raw = read_raw_table(table_id, out_dir=out_dir)
        except Exception:
            raw = None
        markdown = render_table_markdown(
            raw, max_rows=max_rows, max_chars=max_chars
        )
        if not markdown:
            hydrated.append(chunk)
            continue
        copy = dict(chunk)
        base = str(chunk.get("text") or "").strip()
        copy["text"] = (
            f"{base}\n\nFull table (Markdown):\n{markdown}" if base else markdown
        )
        copy["table_hydrated"] = True
        copy["table_rows"] = max(len(raw.get("rows") or []) - 1, 0)
        hydrated.append(copy)
    return hydrated


def read_raw_table_tool(
    table_id: str, *, out_dir: Path | None = None
) -> dict | None:
    """The ``read_raw_table`` tool: validate the id, then read the asset."""
    value = (table_id or "").strip()
    if not _TABLE_ID_RE.fullmatch(value):
        return None
    try:
        return read_raw_table(value, out_dir=out_dir)
    except Exception:
        return None


def available_tables(scope: str | None, paths: list[str] | None = None) -> list[dict]:
    """Table metadata from the Document Cards (RAM-cached, no Qdrant scroll)."""
    if not scope:
        return []
    from app.capabilities.registry import list_cards

    wanted = {path for path in (paths or []) if path} or None
    tables: list[dict] = []
    seen: set[str] = set()
    for card in list_cards(normalize_root(scope)):
        path = card.get("path")
        if wanted and path not in wanted:
            continue
        for item in card.get("tables") or []:
            table_id = item.get("table_id")
            if not table_id or table_id in seen:
                continue
            seen.add(table_id)
            tables.append(
                {
                    "table_id": table_id,
                    "table_number": item.get("table_number") or "",
                    "caption": item.get("caption") or "",
                    "headers": item.get("headers") or [],
                    "row_count": item.get("row_count", 0),
                    "path": path,
                    "file_name": card.get("file_name") or Path(path or "").name,
                }
            )
    return tables


def _table_manifest_text(tables: list[dict]) -> str:
    lines = []
    for table in tables:
        columns = ", ".join(str(cell) for cell in table.get("headers") or [])
        number = table.get("table_number") or "?"
        lines.append(
            f"- {table.get('table_id')} | Table {number} | "
            f"{table.get('caption') or '(no caption)'} | "
            f"columns: {columns or '(unknown)'} | "
            f"rows: {table.get('row_count', 0)} | "
            f"{table.get('path') or table.get('file_name') or ''}"
        )
    return "\n".join(lines)


def request_table_ids(
    query: str,
    context: str,
    tables: list[dict],
    *,
    max_tables: int = TABLE_TOOL_MAX_TABLES,
) -> list[str]:
    """Ask the utility model which raw tables to fetch (bounded, fail-open)."""
    if not query.strip() or not tables:
        return []
    valid = {str(table.get("table_id")) for table in tables if table.get("table_id")}
    prompt = build_table_tool_prompt(
        query,
        context[:TABLE_TOOL_CONTEXT_CHARS],
        _table_manifest_text(tables),
        max_tables,
    )
    try:
        raw = _generate(
            prompt,
            models=OPENROUTER_UTILITY_MODELS,
            max_tokens=MAX_TOKENS["table_tool"],
        )
    except Exception as error:  # fail-open: answer from the hydrated context
        print(f"request_table_ids failed, skipping: {error}")
        return []

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    ids = data.get("table_ids") if isinstance(data, dict) else None
    if not isinstance(ids, list):
        return []
    selected: list[str] = []
    for item in ids:
        table_id = str(item).strip()
        if table_id in valid and table_id not in selected:
            selected.append(table_id)
        if len(selected) >= max_tables:
            break
    return selected


def table_chunk_from_raw(raw: dict, *, table_number: str = "") -> dict:
    """A RetrievedChunk-shaped dict for a table fetched by the tool."""
    table_id = raw.get("table_id") or ""
    path = raw.get("path") or ""
    caption = raw.get("caption") or ""
    markdown = render_table_markdown(raw)
    if not markdown:
        markdown = str(raw.get("content") or "").strip()
    heading = f"Table {table_number}".strip() if table_number else "Table"
    text = f"[{heading} | ID: {table_id}]\nCaption: {caption}"
    if markdown:
        text = f"{text}\n\nFull table (Markdown):\n{markdown}"
    return {
        "score": 0.0,
        "semantic_score": 0.0,
        "keyword_score": 0.0,
        "chunk_id": f"table-tool:{table_id}",
        "file_id": raw.get("file_id") or "",
        "doc_type": "local",
        "root": "",
        "path": path,
        "url": path,
        "title": Path(path).stem if path else table_id,
        "file_name": Path(path).name if path else table_id,
        "breadcrumbs": [raw.get("breadcrumb") or "Overview"],
        "section_heading": caption or heading,
        "section_level": 1,
        "section_depth": 1,
        "breadcrumb_levels": [1],
        "section_id": raw.get("section_id") or "",
        "section_ancestors": [raw.get("section_id") or "s0"],
        "section_titles": [raw.get("breadcrumb") or "Overview"],
        "section_depths": [1],
        "section_numbers": [""],
        "section_title_clean": caption or heading,
        "section_number": "",
        "raw_heading": heading,
        "chunk_type": "table_stub",
        "text": text,
        "caption": caption,
        "table_number": table_number or raw.get("table_number") or "",
        "asset_id": table_id,
        "asset_path": f"tables/{raw.get('file_id') or ''}.json",
        "headers": raw.get("headers") or [],
        "row_count": raw.get("row_count", 0),
        "table_rows": raw.get("row_count", 0),
        "table_hydrated": bool(markdown),
    }