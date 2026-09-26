from urllib.parse import urlparse
import json

from app.core.config import CLEAN_DIR, GRAPH_DIR

GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def _path_parent_url(url: str) -> str | None:
    normalized = _normalize_url(url)
    if not normalized:
        return None

    parsed = urlparse(normalized)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) <= 1:
        return None

    parent_path = "/" + "/".join(parts[:-1])
    return f"{parsed.scheme}://{parsed.netloc}{parent_path}"


def _collect_section_links(document: dict) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()

    for section in document.get("sections", []):
        for link in section.get("links", []):
            normalized = _normalize_url(link)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            links.append(normalized)

    return links


def _build_nodes(documents: list[dict]) -> tuple[list[dict], dict[str, str]]:
    nodes: list[dict] = []
    url_to_file_id: dict[str, str] = {}

    for document in documents:
        url = _normalize_url(document.get("url", ""))
        file_id = document.get("file_id", "")
        if not url or not file_id:
            continue

        url_to_file_id[url] = file_id
        nodes.append(
            {
                "file_id": file_id,
                "url": url,
                "title": document.get("title", ""),
                "doc_type": document.get("doc_type", ""),
                "source_domain": document.get("source_domain", ""),
                "breadcrumbs": document.get("breadcrumbs", []),
                "discovered_from": _normalize_url(document.get("source", {}).get("discovered_from", "") or ""),
                "path_parent_url": _path_parent_url(url),
                "outgoing_links": _collect_section_links(document),
            }
        )

    return nodes, url_to_file_id


def _append_edge(edges: list[dict], seen: set[tuple[str, str, str]], edge_type: str, from_file_id: str, to_file_id: str) -> None:
    key = (edge_type, from_file_id, to_file_id)
    if not from_file_id or not to_file_id or from_file_id == to_file_id or key in seen:
        return
    seen.add(key)
    edges.append({"type": edge_type, "from": from_file_id, "to": to_file_id})


def _build_edges(nodes: list[dict], url_to_file_id: dict[str, str]) -> list[dict]:
    edges: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for node in nodes:
        file_id = node["file_id"]

        discovered_from = node.get("discovered_from", "")
        if discovered_from in url_to_file_id:
            _append_edge(edges, seen, "discovered_from", url_to_file_id[discovered_from], file_id)

        path_parent_url = node.get("path_parent_url")
        if path_parent_url in url_to_file_id:
            _append_edge(edges, seen, "path_parent", url_to_file_id[path_parent_url], file_id)

        for outgoing_url in node.get("outgoing_links", []):
            if outgoing_url in url_to_file_id:
                _append_edge(edges, seen, "links_to", file_id, url_to_file_id[outgoing_url])

    return edges


def build_doc_graph() -> dict:
    documents: list[dict] = []
    for clean_file in CLEAN_DIR.glob("*.json"):
        documents.append(json.loads(clean_file.read_text(encoding="utf-8")))

    nodes, url_to_file_id = _build_nodes(documents)
    edges = _build_edges(nodes, url_to_file_id)

    graph = {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "documents": len(nodes),
            "edges": len(edges),
        },
    }

    (GRAPH_DIR / "doc_graph.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    return graph


if __name__ == "__main__":
    build_doc_graph()
