from urllib.parse import urljoin, urlparse
import json
import re
import unicodedata

from bs4 import BeautifulSoup, Comment
from app.core.config import CLEAN_DIR, RAW_DIR

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
CONTENT_TAGS = HEADING_TAGS | {"p", "li", "pre", "table", "blockquote"}
NOISE_TAGS = {
    "script",
    "style",
    "template",
    "noscript",
    "svg",
    "path",
    "button",
    "input",
    "header",
    "nav",
    "footer",
    "aside",
    "form",
}
NOISE_CLASS_HINTS = ("loading", "search", "sidebar", "theme-toggle", "chat", "feedback")
NOISE_TEXT_HINTS = (
    "ask a question",
    "enter to send",
    "new chat",
    "chat history",
    "was this page helpful",
    "thumbs up",
    "thumbs down",
)
MOJIBAKE_REPLACEMENTS = {
    "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â¹": "",
    "ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢": "â€¢",
    "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢": "'",
    "ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ": '"',
    "ÃƒÂ¢Ã¢â€šÂ¬\x9d": '"',
    "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“": "-",
    "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â": "-",
}
WHITESPACE_RE = re.compile(r"\n{3,}")
INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
LANGUAGE_RE = re.compile(r"(?:language|lang)-([a-zA-Z0-9_+-]+)")
STREAMED_INSERT_RE = re.compile(r'\$RS\("([^"]+)","([^"]+)"\)')


def _normalize_text(text: str) -> str:
    for wrong, right in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(wrong, right)
    text = unicodedata.normalize("NFKC", text).replace("\u200b", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return WHITESPACE_RE.sub("\n\n", "\n".join(lines)).strip()


def _normalize_inline_text(text: str) -> str:
    return INLINE_WHITESPACE_RE.sub(" ", _normalize_text(text)).strip()


def _normalize_code(code: str) -> str:
    for wrong, right in MOJIBAKE_REPLACEMENTS.items():
        code = code.replace(wrong, right)
    code = unicodedata.normalize("NFKC", code).replace("\u200b", "")
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in code.split("\n")]

    cleaned: list[str] = []
    blank_streak = 0
    for line in lines:
        if line.strip():
            blank_streak = 0
            cleaned.append(line)
            continue
        blank_streak += 1
        if blank_streak <= 2:
            cleaned.append("")

    return "\n".join(cleaned).strip()


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _clone_body(soup: BeautifulSoup) -> BeautifulSoup | None:
    if not soup.body:
        return None
    return BeautifulSoup(str(soup.body), "lxml")


def _resolve_streamed_content(soup: BeautifulSoup) -> None:
    for script in soup.find_all("script"):
        script_text = script.string or script.get_text()
        if not script_text:
            continue

        for source_id, target_id in STREAMED_INSERT_RE.findall(script_text):
            source = soup.find(id=source_id)
            target = soup.find(id=target_id)
            if not source or not target:
                continue

            nodes = list(source.contents)
            for node in nodes:
                target.insert_before(node.extract())

            source.decompose()
            target.decompose()


def _strip_noise(node: BeautifulSoup) -> None:
    for comment in node.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    for tag in node.find_all(NOISE_TAGS):
        tag.decompose()
    for tag in list(node.find_all(True)):
        classes = " ".join(tag.attrs.get("class", [])) if getattr(tag, "attrs", None) else ""
        if any(hint in classes for hint in NOISE_CLASS_HINTS):
            tag.decompose()


def _extract_language(tag) -> str:
    for candidate in [tag, *tag.parents]:
        attrs = getattr(candidate, "attrs", None)
        if not attrs:
            continue
        for class_name in attrs.get("class", []):
            match = LANGUAGE_RE.search(class_name)
            if match:
                return match.group(1).lower()
    return "text"


def _extract_table(tag) -> dict[str, list]:
    rows: list[list[str]] = []
    for row in tag.find_all("tr"):
        cells = [_normalize_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    return {
        "headers": rows[0] if rows else [],
        "rows": rows[1:] if len(rows) > 1 else [],
    }


def _new_section(heading: str, level: int, anchor: str) -> dict:
    return {
        "heading": heading,
        "level": level,
        "anchor": anchor,
        "text_blocks": [],
        "code_blocks": [],
        "tables": [],
        "links": [],
    }


def _is_content_node(tag) -> bool:
    """True when a tag carries section content we want to keep.

    LangChain docs render most paragraphs as ``<span data-as="p">`` instead of
    ``<p>``, so those spans must be treated as paragraphs too.
    """
    return tag.name in CONTENT_TAGS or tag.get("data-as") == "p"


def _is_top_level_content(tag) -> bool:
    parent = tag.parent
    while parent is not None:
        if _is_content_node(parent):
            return False
        parent = parent.parent
    return True


def _collect_links(tag, base_url: str) -> list[str]:
    links: list[str] = []
    for anchor in tag.find_all("a", href=True):
        href = anchor["href"].strip()
        if href and not href.startswith("#"):
            links.append(urljoin(base_url, href))
    return _dedupe_keep_order(links)


def _infer_breadcrumbs(url: str, title: str) -> list[str]:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    pretty = [part.replace("-", " ").replace("_", " ").strip().title() for part in parts]
    return pretty or ([title] if title else [])


def _infer_doc_type(url: str, metadata: dict, sections: list[dict]) -> str:
    if metadata.get("doc_type"):
        return metadata["doc_type"]
    if any(section["code_blocks"] for section in sections):
        return "reference"
    if any(section["tables"] for section in sections):
        return "structured_doc"
    return "article"


def _is_noise_section(section: dict) -> bool:
    heading = (section.get("heading", "") or "").lower()
    if any(hint in heading for hint in NOISE_TEXT_HINTS):
        return True
    text = (section.get("text", "") or "").strip().lower()
    # Only treat short blocks dominated by UI chatter as noise. Long documentation
    # sections may legitimately mention phrases like "chat history".
    return len(text) < 120 and any(hint in text for hint in NOISE_TEXT_HINTS)


def _extract_sections(html: str, metadata: dict) -> tuple[str, str, list[dict]]:
    soup = BeautifulSoup(html, "lxml")
    _resolve_streamed_content(soup)
    body = _clone_body(soup)
    if body:
        _strip_noise(body)

    title = _normalize_inline_text(soup.title.get_text(" ", strip=True)) if soup.title else metadata.get("url", "Untitled")
    description_meta = soup.find("meta", attrs={"name": "description"})
    description = _normalize_inline_text(description_meta.get("content", "")) if description_meta else ""

    sections: list[dict] = []
    current_section = _new_section("Overview", 1, "")
    seen_text: set[str] = set()
    seen_code: set[str] = set()

    if body:
        for tag in body.find_all(True):
            if not _is_content_node(tag):
                continue
            if not _is_top_level_content(tag):
                continue

            if tag.name in HEADING_TAGS:
                if current_section["text_blocks"] or current_section["code_blocks"] or current_section["tables"]:
                    sections.append(current_section)
                current_section = _new_section(
                    _normalize_inline_text(tag.get_text(" ", strip=True)) or "Section",
                    int(tag.name[1]),
                    tag.get("id", ""),
                )
                continue

            if tag.name == "pre":
                code = _normalize_code(tag.get_text("", strip=False))
                if code and code not in seen_code:
                    seen_code.add(code)
                    current_section["code_blocks"].append({"language": _extract_language(tag), "code": code})
                continue

            if tag.name == "table":
                table = _extract_table(tag)
                if table["headers"] or table["rows"]:
                    current_section["tables"].append(table)
                continue

            text = _normalize_text(tag.get_text(" ", strip=True))
            if text and text not in seen_text:
                seen_text.add(text)
                current_section["text_blocks"].append(text)
                current_section["links"].extend(_collect_links(tag, metadata.get("url", "")))

    if current_section["text_blocks"] or current_section["code_blocks"] or current_section["tables"]:
        sections.append(current_section)

    cleaned_sections: list[dict] = []
    for section in sections:
        section["links"] = _dedupe_keep_order(section["links"])
        section["text"] = "\n\n".join(section["text_blocks"]).strip()
        if _is_noise_section(section):
            continue
        section["section_role"] = "content"
        section["should_embed"] = bool(section["text"] or section["code_blocks"] or section["tables"])
        cleaned_sections.append(section)

    return title, description, cleaned_sections


def extract_document(html: str, metadata: dict) -> dict:
    title, description, sections = _extract_sections(html, metadata)
    doc_type = _infer_doc_type(metadata.get("url", ""), metadata, sections)

    plain_text_parts = [section["text"] for section in sections if section["should_embed"] and section["text"]]
    code_examples = [block for section in sections if section["should_embed"] for block in section["code_blocks"]]
    tables = [table for section in sections if section["should_embed"] for table in section["tables"]]

    return {
        "schema_version": 4,
        "file_id": metadata.get("file_id", ""),
        "url": metadata.get("url", ""),
        "source_url": metadata.get("source_url", metadata.get("url", "")),
        "source_domain": metadata.get("source_domain", urlparse(metadata.get("url", "")).netloc),
        "doc_type": doc_type,
        "title": title,
        "description": description,
        "breadcrumbs": _infer_breadcrumbs(metadata.get("url", ""), title),
        "headings": [section["heading"] for section in sections],
        "sections": sections,
        "plain_text": "\n\n".join(plain_text_parts).strip(),
        "code_examples": code_examples,
        "tables": tables,
        "source": {
            "crawled_at": metadata.get("crawled_at"),
            "status_code": metadata.get("status_code"),
            "content_type": metadata.get("content_type", ""),
            "discovered_from": metadata.get("discovered_from"),
            "anchor_text": metadata.get("anchor_text", ""),
        },
    }


def clean_all() -> None:
    for html_file in RAW_DIR.glob("*.html"):
        file_id = html_file.stem
        meta_file = RAW_DIR / f"{file_id}.json"
        if not meta_file.exists():
            continue

        html = html_file.read_text(encoding="utf-8")
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        metadata.setdefault("file_id", file_id)

        document = extract_document(html, metadata)
        out_file = CLEAN_DIR / f"{file_id}.json"
        out_file.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    clean_all()
