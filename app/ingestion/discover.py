from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from app.core.config import ALLOWED_PATH_PREFIXES, EXCLUDED_PATH_HINTS, SKIP_FILE_EXTENSIONS


def normalize_url(base_url: str, href: str) -> str | None:
    if not href:
        return None

    absolute = urljoin(base_url, href.strip())
    parsed = urlparse(absolute)
    if parsed.scheme not in ("http", "https"):
        return None

    clean_path = parsed.path.rstrip("/") or "/"
    normalized = parsed._replace(query="", fragment="", path=clean_path)
    return urlunparse(normalized)


def classify_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path

    if parsed.netloc == "docs.langchain.com":
        return "guide"

    parts = [part for part in path.split("/") if part]
    if not parts:
        return "reference_index"
    if len(parts) <= 3:
        return "reference_index"
    if len(parts) == 4:
        return "reference_topic"
    return "reference_symbol"


def is_allowed_url(url: str, allowed_domains: set[str]) -> bool:
    parsed = urlparse(url)
    if parsed.netloc not in allowed_domains:
        return False

    path = parsed.path.lower()
    if not any(path.startswith(prefix) for prefix in ALLOWED_PATH_PREFIXES):
        return False
    if any(hint in path for hint in EXCLUDED_PATH_HINTS):
        return False
    if any(path.endswith(ext) for ext in SKIP_FILE_EXTENSIONS):
        return False
    return True


def extract_links(html: str, base_url: str, allowed_domains: set[str]) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    links: dict[str, dict[str, str]] = {}

    for anchor in soup.find_all("a", href=True):
        url = normalize_url(base_url, anchor["href"])
        if not url or not is_allowed_url(url, allowed_domains):
            continue
        links[url] = {
            "url": url,
            "anchor_text": anchor.get_text(" ", strip=True),
            "doc_type": classify_url(url),
        }

    return [links[url] for url in sorted(links)]
