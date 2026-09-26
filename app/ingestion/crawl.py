from collections import deque
from datetime import datetime, timezone
from urllib.parse import urlparse
import hashlib
import json
import xml.etree.ElementTree as ET

import requests

from app.core.config import ALLOWED_DOMAINS, CRAWL_ROOTS, MAX_PAGES, RAW_DIR, REQUEST_TIMEOUT, SITEMAP_URLS, USER_AGENT
from app.ingestion.discover import classify_url, extract_links, is_allowed_url, normalize_url

RAW_DIR.mkdir(parents=True, exist_ok=True)


def url_to_filename(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def fetch(url: str) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response


def fetch_optional(url: str) -> requests.Response | None:
    try:
        return fetch(url)
    except Exception as exc:
        print(f"Skipping {url}: {exc}")
        return None


def _parse_sitemap(xml_text: str) -> tuple[list[str], list[str]]:
    root = ET.fromstring(xml_text)
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0] + "}"

    sitemap_nodes = root.findall(f".//{namespace}sitemap/{namespace}loc")
    url_nodes = root.findall(f".//{namespace}url/{namespace}loc")
    sitemap_urls = [node.text.strip() for node in sitemap_nodes if node.text and node.text.strip()]
    page_urls = [node.text.strip() for node in url_nodes if node.text and node.text.strip()]
    return sitemap_urls, page_urls


def discover_urls_from_sitemaps() -> list[dict[str, str]]:
    pending = deque(SITEMAP_URLS)
    seen_sitemaps: set[str] = set()
    discovered: dict[str, dict[str, str]] = {}

    while pending:
        sitemap_url = pending.popleft()
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)

        response = fetch_optional(sitemap_url)
        if not response:
            continue

        try:
            child_sitemaps, page_urls = _parse_sitemap(response.text)
        except ET.ParseError as exc:
            print(f"Invalid sitemap {sitemap_url}: {exc}")
            continue

        for child_url in child_sitemaps:
            pending.append(child_url)

        for page_url in page_urls:
            normalized = normalize_url(sitemap_url, page_url)
            if not normalized or not is_allowed_url(normalized, ALLOWED_DOMAINS):
                continue
            discovered[normalized] = {
                "url": normalized,
                "discovered_from": sitemap_url,
                "anchor_text": "",
                "doc_type": classify_url(normalized),
            }

    return [discovered[url] for url in sorted(discovered)]


def save_raw(
    response: requests.Response,
    *,
    discovered_from: str | None,
    anchor_text: str,
    queued_doc_type: str,
) -> str:
    final_url = response.url
    file_id = url_to_filename(final_url)
    html_path = RAW_DIR / f"{file_id}.html"
    meta_path = RAW_DIR / f"{file_id}.json"

    html_path.write_text(response.text, encoding="utf-8")

    metadata = {
        "file_id": file_id,
        "url": final_url,
        "source_url": response.request.url,
        "source_domain": urlparse(final_url).netloc,
        "doc_type": queued_doc_type or classify_url(final_url),
        "discovered_from": discovered_from,
        "anchor_text": anchor_text,
        "status_code": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return final_url


def crawl() -> list[str]:
    inventory = discover_urls_from_sitemaps()
    if inventory:
        queue = deque(inventory)
    else:
        queue = deque(
            {
                "url": url,
                "discovered_from": None,
                "anchor_text": "",
                "doc_type": classify_url(url),
            }
            for url in CRAWL_ROOTS
        )
    visited: set[str] = set()
    collected: list[str] = []

    while queue and len(collected) < MAX_PAGES:
        job = queue.popleft()
        url = job["url"]
        if url in visited:
            continue
        visited.add(url)

        try:
            response = fetch(url)
            final_url = save_raw(
                response,
                discovered_from=job["discovered_from"],
                anchor_text=job["anchor_text"],
                queued_doc_type=job["doc_type"],
            )
            collected.append(final_url)

            for link in extract_links(response.text, final_url, ALLOWED_DOMAINS):
                if link["url"] not in visited:
                    queue.append(
                        {
                            "url": link["url"],
                            "discovered_from": final_url,
                            "anchor_text": link["anchor_text"],
                            "doc_type": link["doc_type"],
                        }
                    )

            print(f"Crawled: {final_url}")
        except Exception as exc:
            print(f"Error processing {url}: {exc}")

    return collected


if __name__ == "__main__":
    crawl()
