import json

from app.core.config import CHUNK_DIR, CHUNK_MAX_CHARS, CHUNK_TARGET_CHARS, CLEAN_DIR
from app.core.models import ChunkRecord

CHUNK_DIR.mkdir(parents=True, exist_ok=True)


def _render_tables(tables: list[dict]) -> str:
    rendered_tables: list[str] = []

    for table in tables:
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        row_lines: list[str] = []

        for row in rows:
            if headers and len(headers) == len(row):
                row_lines.append("; ".join(f"{header}: {value}" for header, value in zip(headers, row)))
            else:
                row_lines.append(" | ".join(str(value) for value in row if str(value).strip()))

        table_parts: list[str] = []
        if headers:
            table_parts.append("Table columns: " + " | ".join(headers))
        if row_lines:
            table_parts.extend(f"Table row: {line}" for line in row_lines)

        table_text = "\n".join(part for part in table_parts if part).strip()
        if table_text:
            rendered_tables.append(table_text)

    return "\n\n".join(rendered_tables).strip()


def _split_text(text: str, target_chars: int, max_chars: int) -> list[str]:
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


def build_chunks(target_chars: int = CHUNK_TARGET_CHARS, max_chars: int = CHUNK_MAX_CHARS) -> list[ChunkRecord]:
    all_chunks: list[ChunkRecord] = []

    for clean_file in CLEAN_DIR.glob("*.json"):
        document = json.loads(clean_file.read_text(encoding="utf-8"))
        doc_chunks: list[ChunkRecord] = []

        for section_index, section in enumerate(document.get("sections", []), start=1):
            if not section.get("should_embed", True):
                continue
            section_heading = section.get("heading") or "Section"
            base_context = [
                document.get("title", ""),
                " > ".join(document.get("breadcrumbs", [])),
                section_heading,
            ]
            section_text = section.get("text", "").strip()
            code_blocks = section.get("code_blocks", [])
            table_text = _render_tables(section.get("tables", []))
            code_text = "\n\n".join(
                f"```{block.get('language', 'text')}\n{block.get('code', '').strip()}\n```"
                for block in code_blocks
                if block.get("code", "").strip()
            ).strip()

            body_parts = [part for part in [section_text, code_text, table_text] if part]
            if not body_parts:
                continue
            joined_body = "\n\n".join(body_parts)

            for part_index, content in enumerate(_split_text(joined_body, target_chars, max_chars), start=1):
                chunk_text = "\n\n".join(part for part in [*base_context, content] if part).strip()
                chunk = {
                    "chunk_id": f"{document.get('file_id', clean_file.stem)}:{section_index}:{part_index}",
                    "file_id": document.get("file_id", clean_file.stem),
                    "url": document.get("url", ""),
                    "title": document.get("title", ""),
                    "source_domain": document.get("source_domain", ""),
                    "doc_type": document.get("doc_type", ""),
                    "breadcrumbs": document.get("breadcrumbs", []),
                    "section_heading": section_heading,
                    "section_anchor": section.get("anchor", ""),
                    "section_role": section.get("section_role", "unknown"),
                    "priority_band": section.get("priority_band", "unknown"),
                    "importance_score": section.get("importance_score", 0),
                    "reasons": section.get("reasons", []),
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }
                doc_chunks.append(chunk)
                all_chunks.append(chunk)

        out_path = CHUNK_DIR / clean_file.name
        out_path.write_text(json.dumps(doc_chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    corpus_path = CHUNK_DIR / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    manifest = {
        "documents": len(list(CLEAN_DIR.glob("*.json"))),
        "chunks": len(all_chunks),
        "target_chars": target_chars,
        "max_chars": max_chars,
    }
    (CHUNK_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return all_chunks


if __name__ == "__main__":
    build_chunks()
