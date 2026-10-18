"""Canonical section tree for indexed documents.

Why this module exists
----------------------
The old retrieval code identified a section by its *raw heading text*
(``section_heading``). That breaks for several reasons:

* PDF headings carry Markdown decoration (``**4.1 Experimental Setup**``).
* Tables get a synthetic suffix (``**4.1 Experimental Setup** (Table)``), so a
  heading lookup silently misses the table.
* A heading with no direct body text (``2 Background``) never appeared in the
  chunk-derived outline, even though its child sections did.
* Two sections can share a heading name; raw text cannot tell them apart.
* A parent heading did not expand to its child sections.

Instead, ingestion stores a **materialized path** per chunk::

    section_id:        "s4.1"
    section_ancestors: ["s4", "s4.1"]
    section_titles:    ["4 Experiments", "4.1 Setup"]
    section_depths:    [1, 2]

``section_id`` is structural (``s1``, ``s1.1``, ``s1.1.2``): it is generated
from the tree position, so it is unique and stable even when heading text is
duplicated or unnumbered. A query for ``s4`` matches every chunk whose
``section_ancestors`` contains ``s4`` -- i.e. the whole subtree. A query for
``s4.1`` matches just that branch, including its table chunks because tables
share their section's id.

The Document Card stores the same tree as ``outline_tree``. Retrieval reads it
from the card (RAM-cached) instead of scrolling Qdrant on every turn.
"""

from __future__ import annotations

import re

# Bump when the tree shape or metadata changes; cards without the current
# version are rebuilt from Qdrant during sync.
OUTLINE_VERSION = 1

_MARKDOWN_MARKS = re.compile(r"\*+")
_WHITESPACE = re.compile(r"\s+")
# A leading section number: "1", "2.1", "A.1", "B.2.3". The title must follow
# after whitespace, so identifiers like "3D Gaussian" are not treated as numbers.
_SECTION_NUMBER = re.compile(r"^((?:\d+|[A-Za-z])(?:\.\d+)*)\.?\s+(.+)$")


def clean_heading(text: str) -> str:
    """Return a display heading without Markdown emphasis and extra spaces."""
    cleaned = _MARKDOWN_MARKS.sub(" ", text or "")
    return _WHITESPACE.sub(" ", cleaned).strip()


def split_section_number(text: str) -> tuple[str, str]:
    """Split ``"4.1 Experimental Setup"`` into ``("4.1", "Experimental Setup")``.

    The number is presentation metadata only; section ids are structural. When
    the heading has no leading number, the number is empty.
    """
    match = _SECTION_NUMBER.match(text or "")
    if not match:
        return "", (text or "").strip()
    return match.group(1), match.group(2).strip()


def _node(
    section_id: str,
    title: str,
    level: int,
    number: str = "",
    synthetic: bool = False,
) -> dict:
    return {
        "id": section_id,
        "title": title,
        "level": max(int(level or 1), 1),
        "number": number,
        "synthetic": bool(synthetic),
        "children": [],
    }


def build_section_tree(chunks: list[dict]) -> list[dict]:
    """Build the nested outline from chunk section metadata.

    Chunks are expected in reading order and carry ``section_ancestors``,
    ``section_titles``, ``section_depths`` and (optionally) ``section_numbers``.
    A parent section with no direct chunk still appears because every descendant
    chunk carries the full ancestor path.
    """
    nodes: dict[str, dict] = {}
    roots: list[dict] = []

    for chunk in chunks:
        ids = list(chunk.get("section_ancestors") or [])
        if not ids:
            # Tolerate a single-section chunk (e.g. hand-built test data).
            section_id = chunk.get("section_id")
            if not section_id:
                continue
            ids = [section_id]
        titles = list(chunk.get("section_titles") or [])
        depths = list(chunk.get("section_depths") or [])
        numbers = list(chunk.get("section_numbers") or [])

        for index, section_id in enumerate(ids):
            if not section_id or section_id in nodes:
                continue
            title = (
                str(titles[index]).strip()
                if index < len(titles) and titles[index]
                else section_id
            )
            level = depths[index] if index < len(depths) else index + 1
            number = numbers[index] if index < len(numbers) else ""
            if not number:
                number = split_section_number(title)[0]
            node = _node(
                section_id,
                title,
                level,
                number,
                synthetic=section_id == "s0" and title == "Overview",
            )
            nodes[section_id] = node

            parent_id = ids[index - 1] if index > 0 else None
            if parent_id and parent_id in nodes:
                nodes[parent_id]["children"].append(node)
            else:
                roots.append(node)

    return roots


def flatten_sections(tree: list[dict]) -> list[dict]:
    """Depth-first, reading-order view of the tree (no ``children`` key)."""
    flat: list[dict] = []

    def walk(nodes: list[dict]) -> None:
        for node in nodes:
            flat.append(
                {
                    "id": node.get("id", ""),
                    "title": node.get("title", ""),
                    "level": int(node.get("level") or 1),
                    "number": node.get("number", ""),
                }
            )
            walk(node.get("children") or [])

    walk(tree or [])
    return flat


def outline_titles(tree: list[dict]) -> list[str]:
    """Flat list of clean section titles, in reading order."""
    return [item["title"] for item in flatten_sections(tree) if item.get("title")]


def render_outline(tree: list[dict]) -> str:
    """Compact indented text for a card (levels use the tree depth)."""
    lines: list[str] = []

    def walk(nodes: list[dict]) -> None:
        for node in nodes:
            level = max(int(node.get("level") or 1), 1)
            lines.append("  " * (level - 1) + f"- {node.get('title', '')}")
            walk(node.get("children") or [])

    walk(tree or [])
    return "\n".join(lines)
