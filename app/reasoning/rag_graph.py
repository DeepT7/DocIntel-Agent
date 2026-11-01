import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, START, END

from app.capabilities.workspace import list_files
from app.core.models import ExternalSource, RAGState
from app.core.prompts import build_chat_prompt, build_workspace_prompt
from app.reasoning.answer import (
    _generate,
    classify_intent,
    condense_question,
    format_context,
    format_history,
    generate_answer,
    grade_documents,
    resolve_section,
    rewrite_query,
)
from app.core.config import (
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    GRADE_SKIP_THRESHOLD,
    MAX_REWRITES,
    MAX_SECTION_CHUNKS,
    MAX_SECTION_DOCS,
    MAX_TOKENS,
    SECTION_CAP_FLOOR,
    SECTION_EXPAND_MAX_CHUNKS,
    SECTION_EXPAND_PER_SECTION,
    SECTION_PER_DOC_FLOOR,
)
from app.core.sections import flatten_sections
from app.retrieval.retrieve import document_outline, section_chunks, vector_search
from app.core.tracing import observe, update_current_span, set_trace_io, trace_context, flush_traces
from app.core.text import clean_text

load_dotenv()

_SECTION_REF_RE = re.compile(
    # Explicit structural word + number (English/Vietnamese accelerator).
    r"\b(?:section|sections|subsection|chapter|mục|phần|chương)\s+"
    r"([A-Za-z]?\d+(?:\.\d+)*)"
    # Dotted number: 4.1, A.1, B.2.3 (not a model version like GPT-4.1).
    r"|(?<![A-Za-z0-9_-])([A-Za-z]?\d+(?:\.\d+)+)(?![A-Za-z0-9])"
    # Bare number, including CJK adjacency ("第4章"), but not a model version
    # ("GPT-4", "GPT-4.1", "model4"). The resolver validates the actual section.
    r"|(?<![A-Za-z0-9_.-])(\d+)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _section_references(query: str) -> list[str]:
    """Distinct section numbers named by the message, in order ("4", "4.1")."""
    refs: list[str] = []
    for match in _SECTION_REF_RE.finditer(query or ""):
        ref = next((group for group in match.groups() if group), "")
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _normalize_doc_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _mentions_doc(query: str, path: str) -> bool:
    normalized = _normalize_doc_text(query)
    if not normalized:
        return False
    stem = _normalize_doc_text(Path(path).stem)
    name = _normalize_doc_text(Path(path).name)
    return bool(
        (stem and stem in normalized) or (name and name in normalized)
    )


def _query_is_doc_specific(query: str, path: str, all_paths: list[str]) -> bool:
    """True when the query does not name a different document."""
    return not any(
        other != path and _mentions_doc(query, other) for other in all_paths
    )


_client = None


def _get_client():
    """Lazily build the Gemini client.

    Only ``external_search_node`` needs Gemini, and it is disabled by default for
    local-first folder Q&A. Importing this module must not require a Gemini key.
    """
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            return None
        _client = genai.Client(api_key=api_key)
    return _client

@observe("classify_node", capture_output=False)
def classify_node(state: RAGState) -> dict:
    """Route the turn: chat / followup / question.

    If the caller already classified (the server needs the label to choose the
    document scope), reuse it instead of paying for a second model call.
    """
    intent = state.get("intent")
    if intent not in ("chat", "followup", "question", "workspace"):
        intent = classify_intent(state["query"], state.get("chat_history"))
    update_current_span(intent=intent, query=state["query"])
    return {"intent": intent}


@observe("chat_node", capture_output=False)
def chat_node(state: RAGState) -> dict:
    """Reply to greetings/meta without touching the index."""
    query = state["query"]
    history = state.get("chat_history", []) or []
    prompt = build_chat_prompt(query, format_history(history))
    answer = _generate(prompt, max_tokens=MAX_TOKENS["chat_node"])
    update_current_span(output=answer, intent="chat")
    return {"context": "", "answer": answer, "sources": [], "intent": "chat"}


@observe("workspace_node", capture_output=False)
def workspace_node(state: RAGState) -> dict:
    """Answer questions about the workspace itself (list files, counts)."""
    scope = state.get("scope")
    files = list_files(scope) if scope else []
    prompt = build_workspace_prompt(scope or "", files, state["query"])
    answer = _generate(prompt, max_tokens=MAX_TOKENS["workspace_node"])
    update_current_span(output=answer, intent="workspace", num_files=len(files))
    return {"context": "", "answer": answer, "sources": [], "intent": "workspace"}


@observe("condense_node", capture_output=False)
def condense_node(state: RAGState) -> dict:
    """Rewrite a follow-up into a standalone query using chat history."""
    query = state["query"]
    history = state.get("chat_history", []) or []
    standalone = condense_question(query, history)
    update_current_span(
        query=query,
        standalone_query=standalone,
        has_history=bool(history),
        rewritten=standalone != query,
    )
    return {"standalone_query": standalone}


@observe("retrieve_node", as_type="retriever")
def retrieve_node(state: RAGState) -> dict:
    query = state.get("standalone_query") or state["query"]
    scope = state.get("scope")
    doc_paths = state.get("doc_paths")
    top_k = state.get("top_k") or DEFAULT_TOP_K

    # Canonical section filter: resolve structural ids per focused document. The
    # tree comes from the RAM-cached Document Card, so this adds one utility LLM
    # call per document (capped) but no Qdrant scroll. A parent id returns its
    # whole subtree; table chunks share their section's id.
    section_locks: dict[str, list[str]] = {}
    locked_chunk_ids: set[str] = set()
    results: list[dict] = []
    focus_paths = doc_paths[:MAX_SECTION_DOCS] if doc_paths else []
    router_queries = state.get("queries") or []
    # Best-effort comparison mapping: the router emits one rewritten query per
    # side, in the same order as target_doc_ids. Use it only when the counts line
    # up AND the queries are distinct; some models return the same mixed query
    # twice, and pairing that with each document would lock the wrong sections.
    section_queries = (
        router_queries
        if (
            len(focus_paths) > 1
            and len(router_queries) == len(focus_paths)
            and len(set(router_queries)) == len(router_queries)
            and all(
                _query_is_doc_specific(query, path, focus_paths)
                for query, path in zip(router_queries, focus_paths)
            )
        )
        else None
    )
    # Fallback comparison mapping when the router did not return usable per-side
    # queries. Section numbers are ordered as they appear in the message and are
    # paired with the ordered target documents. This handles the common
    # "section 4 of the first paper / section 3 of the second" shape without
    # guessing when the counts do not line up.
    if len(focus_paths) > 1 and not section_queries:
        refs = _section_references(query)
        if len(refs) == len(focus_paths):
            section_queries = [f"section {ref}" for ref in refs]
        elif len(refs) == 1:
            section_queries = [query] * len(focus_paths)
        elif len(refs) > 1:
            focus_paths = []
    if focus_paths:
        # Reserve room for vector retrieval when there are documents beyond the
        # section-resolution cap, then split the rest evenly across focused docs.
        # A wide comparison must not let the first huge chapter consume every
        # context slot.
        section_cap = MAX_SECTION_CHUNKS
        if doc_paths and len(doc_paths) > MAX_SECTION_DOCS:
            section_cap = max(MAX_SECTION_CHUNKS - top_k, SECTION_CAP_FLOOR)
        per_doc_budget = max(SECTION_PER_DOC_FLOOR, section_cap // len(focus_paths))
        for index, path in enumerate(focus_paths):
            outline = document_outline(scope, path)
            sections = flatten_sections(outline.get("tree") or [])
            section_query = section_queries[index] if section_queries else query
            section_ids = resolve_section(
                section_query, outline.get("title") or path, sections
            )
            if not section_ids:
                continue
            chunks = section_chunks(
                scope,
                path,
                section_ids,
                limit=section_cap,
                include_descendants=True,
            )
            if not chunks:
                continue
            section_locks[path] = section_ids
            added = 0
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id and chunk_id in locked_chunk_ids:
                    continue
                if chunk_id:
                    locked_chunk_ids.add(chunk_id)
                results.append(chunk)
                added += 1
                if added >= per_doc_budget or len(results) >= MAX_SECTION_CHUNKS:
                    break

    # Documents without a resolved section (or beyond the cap) keep vector
    # retrieval. A whole-folder question (doc_paths=None) never section-locks.
    unlocked_paths = (
        [path for path in doc_paths if path not in section_locks]
        if doc_paths
        else None
    )
    need_vector = not doc_paths or bool(unlocked_paths)
    if need_vector and len(results) < MAX_SECTION_CHUNKS:
        # Router sub-queries: run each, then merge (dedup by chunk_id). This is
        # the "decompose a comparison into per-side queries" step.
        queries = router_queries
        if len(queries) > 1:
            merged: list[dict] = []
            seen: set[str] = set(locked_chunk_ids)
            for sub_query in queries:
                for chunk in vector_search(
                    sub_query, top_k=top_k, scope=scope, paths=unlocked_paths
                ):
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id and chunk_id in seen:
                        continue
                    if chunk_id:
                        seen.add(chunk_id)
                    merged.append(chunk)
            vector_results = merged[: top_k * len(queries)]
        else:
            vector_results = vector_search(
                query, top_k=top_k, scope=scope, paths=unlocked_paths
            )

        seen = set(locked_chunk_ids)
        for chunk in vector_results:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id in seen:
                continue
            if chunk_id:
                seen.add(chunk_id)
            results.append(chunk)
            if len(results) >= MAX_SECTION_CHUNKS:
                break

    update_current_span(
        output=[
            {"chunk_id": r.get("chunk_id"), "title": r.get("title"), "score": round(r.get("score", 0.0), 4)}
            for r in results
        ],
        query=query,
        scope=scope,
        doc_paths=doc_paths,
        num_chunks=len(results),
        section_locks=section_locks,
    )
    return {
        "retrieved_chunks": results,
        "section_locks": section_locks,
        "section_locked_chunk_ids": sorted(locked_chunk_ids),
    }

@observe("grade_node", capture_output=False)
def grade_node(state: RAGState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    locked_ids = set(state.get("section_locked_chunk_ids") or [])
    locked = [chunk for chunk in chunks if chunk.get("chunk_id") in locked_ids]
    unlocked = [chunk for chunk in chunks if chunk.get("chunk_id") not in locked_ids]

    # Chunks from a resolved section are exactly what the user asked for; grade
    # only the vector-retrieved remainder. This makes mixed multi-document turns
    # (one document section-locked, another not) correct.
    if not unlocked:
        update_current_span(
            num_chunks=len(chunks),
            num_relevant=len(chunks),
            gated=True,
            section_locks=state.get("section_locks"),
        )
        return {"relevant_chunks": chunks, "graded_ok": bool(chunks)}

    query = state.get("standalone_query") or state["query"]
    # ``score`` is the fused RRF value; gate on the dense cosine when available.
    top_score = max(
        (
            chunk.get("semantic_score") or chunk.get("score", 0.0)
            for chunk in unlocked
        ),
        default=0.0,
    )
    # Skip the (slow) LLM grading call when retrieval is already confident.
    if top_score >= GRADE_SKIP_THRESHOLD:
        graded = unlocked
        gated = True
    else:
        indices = grade_documents(query, unlocked)
        graded = [unlocked[i] for i in indices]
        gated = False
    relevant = locked + graded
    update_current_span(
        num_chunks=len(chunks),
        num_relevant=len(relevant),
        gated=gated,
        locked=len(locked),
        section_locks=state.get("section_locks"),
    )
    return {"relevant_chunks": relevant, "graded_ok": bool(relevant)}
    

@observe("rewrite_node", capture_output=False)
def rewrite_node(state: RAGState) -> dict:
    query = state.get("standalone_query") or state["query"]
    rewritten = rewrite_query(query)
    update_current_span(original=query, rewritten=rewritten)
    return {"standalone_query": rewritten, "rewrite_count": state.get("rewrite_count", 0) + 1}

def route_after_grade(state: RAGState):
    if state.get("graded_ok"):
        return "answer"
    if state.get("rewrite_count", 0) < MAX_REWRITES: # Permitted to rewrite a limited number of times
        return "rewrite"
    # Local-first default: do not silently send the query to the web.
    if not state.get("allow_external", True):
        return "fallback"
    return "external_search"

def _expand_to_sections(
    chunks: list[dict],
    scope: str | None,
    per_section: int = SECTION_EXPAND_PER_SECTION,
    max_chunks: int = SECTION_EXPAND_MAX_CHUNKS,
) -> list[dict]:
    """Small-to-big: widen each retrieved chunk to its whole section.

    A section is often split across several chunks; the fact the user asked for
    can sit in a sibling chunk that scored slightly lower. Adding those siblings
    to the context lets the answer use the full section instead of a fragment.
    """
    expanded: list[dict] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id in seen:
            continue
        if chunk_id:
            seen.add(chunk_id)
        expanded.append(chunk)
        if len(expanded) >= max_chunks:
            break

        path = chunk.get("path")
        section_id = chunk.get("section_id")
        if not path or not section_id:
            continue
        added = 0
        for sibling in section_chunks(
            scope,
            path,
            [section_id],
            limit=per_section + 1,
            include_descendants=False,
        ):
            sid = sibling.get("chunk_id")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            expanded.append(sibling)
            added += 1
            if added >= per_section or len(expanded) >= max_chunks:
                break

    return expanded[:max_chunks]


@observe("answer_node", capture_output=False)
def answer_node(state: RAGState) -> dict:
    query = state.get("standalone_query") or state["query"]
    retrieved_chunks = state.get("relevant_chunks") or state.get("retrieved_chunks", [])
    sources = _expand_to_sections(retrieved_chunks, state.get("scope"))
    context = format_context(sources)
    answer = generate_answer(query, context, thinking=state.get("thinking", False))
    update_current_span(output=answer, num_sources=len(sources), answer_chars=len(answer))

    return {
        "context": context,
        "answer": answer,
        "sources": sources,
    }

def _extract_external_sources(response) -> list[ExternalSource]:
    sources: list[ExternalSource] = []
    seen_urls: set[str] = set()

    grounding_metadata = getattr(response, "grounding_metadata", None)
    if grounding_metadata is None:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            grounding_metadata = getattr(candidates[0], "grounding_metadata", None)

    grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
    for index, chunk in enumerate(grounding_chunks, start=1):
        web = getattr(chunk, "web", None)
        if web is None:
            continue
        uri = getattr(web, "uri", "") or ""
        if not uri or uri in seen_urls:
            continue
        seen_urls.add(uri)
        sources.append(
            {
                "title": getattr(web, "title", f"External Source {index}") or f"External Source {index}",
                "url": uri,
                "section_heading": "External Search",
                "score": 0.0,
                "breadcrumbs": [],
                "doc_type": "external_search",
                "text": "",
            }
        )

    return sources


def _format_external_context(answer_text: str, sources: list[ExternalSource]) -> str:
    source_lines = []
    for index, source in enumerate(sources, start=1):
        source_lines.append(
            f"[External Source {index}]\n"
            f"Title: {source.get('title', 'No Title')}\n"
            f"URL: {source.get('url', 'No URL')}"
        )

    joined_sources = "\n\n".join(source_lines)
    parts = [part for part in [f"External grounded answer:\n{answer_text}".strip(), joined_sources] if part]
    return "\n\n".join(parts).strip()


@observe("external_search_node", capture_output=False)
def external_search_node(state: RAGState) -> dict:
    query = (state.get("standalone_query") or state["query"]).strip()
    if not query:
        return {
            "external_ok": False,
            "external_reason": "Query must not be empty.",
            "external_context": "",
            "external_sources": [],
        }

    client = _get_client()
    if client is None:
        update_current_span(output=None, reason="no Gemini key")
        return {
            "external_ok": False,
            "external_reason": "External search unavailable (no Gemini key).",
            "external_context": "",
            "external_sources": [],
        }

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=f"Find relevant current public web information for this question and summarize only the most useful facts.\n\nQuestion: {query}",
        config=types.GenerateContentConfig(tools=[grounding_tool]),
    )

    external_answer = getattr(response, "text", "").strip()
    external_sources = _extract_external_sources(response)
    external_context = _format_external_context(external_answer, external_sources)
    update_current_span(
        output={"answer": external_answer, "urls": [s.get("url") for s in external_sources]},
        query=query,
        num_external_sources=len(external_sources),
    )

    if not external_answer and not external_sources:
        return {
            "external_ok": False,
            "external_reason": "External search did not return grounded results.",
            "external_context": "",
            "external_sources": [],
        }

    return {
        "external_ok": True,
        "external_reason": f"External search returned {len(external_sources)} source(s).",
        "external_context": external_context,
        "external_sources": external_sources,
    }


@observe("answer_with_fallback_node", capture_output=False)
def answer_with_fallback_node(state: RAGState) -> dict:
    query = state.get("standalone_query") or state["query"]
    internal_chunks = state.get("relevant_chunks") or state.get("retrieved_chunks", [])
    internal_chunks = _expand_to_sections(internal_chunks, state.get("scope"))
    internal_context = format_context(internal_chunks) if internal_chunks else ""
    external_context = state.get("external_context", "").strip()

    context_parts = []
    if internal_context:
        context_parts.append("Internal knowledge base context:\n" + internal_context)
    if external_context:
        context_parts.append("External grounded web context:\n" + external_context)

    combined_context = "\n\n".join(part for part in context_parts if part).strip()
    answer = generate_answer(query, combined_context, thinking=state.get("thinking", False))
    sources = [*internal_chunks, *state.get("external_sources", [])]
    update_current_span(output=answer, num_sources=len(sources), answer_chars=len(answer))

    return {
        "context": combined_context,
        "answer": answer,
        "sources": sources,
    }


@observe("fallback_node")
def fallback_node(state: RAGState) -> dict:
    reason = state.get("external_reason") or state.get("retrieval_reason", "Insufficient retrieved context.")
    answer = f"I could not find strong enough support in the knowledge base. {reason}"
    update_current_span(output=answer, reason=reason)
    return {
        "context": "",
        "answer": answer,
        "sources": [],
    }


def route_after_external_search(state: RAGState) -> str:
    if state.get("external_ok", False):
        return "answer_with_fallback"
    return "fallback"

builder = StateGraph(RAGState)
builder.add_node("classify", classify_node)
builder.add_node("chat", chat_node)
builder.add_node("workspace", workspace_node)
builder.add_node("condense", condense_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("answer", answer_node)
builder.add_node("grade", grade_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("external_search", external_search_node)
builder.add_node("answer_with_fallback", answer_with_fallback_node)
builder.add_node("fallback", fallback_node)

builder.add_edge(START, "classify")
# chat -> no retrieval; followup -> rewrite to standalone then retrieve;
# question -> already standalone, retrieve directly (no condense).
builder.add_conditional_edges(
    "classify",
    lambda state: state.get("intent", "question"),
    {"chat": "chat", "workspace": "workspace",
     "followup": "condense", "question": "retrieve"},
)
builder.add_edge("chat", END)
builder.add_edge("workspace", END)
builder.add_edge("condense", "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", route_after_grade,
                              {"answer": "answer", "rewrite": "rewrite",
                               "external_search": "external_search", "fallback": "fallback"})
builder.add_edge("rewrite", "retrieve")
builder.add_conditional_edges("external_search", route_after_external_search,
                             {
                                 "answer_with_fallback": "answer_with_fallback",
                                 "fallback": "fallback",
                             },)
builder.add_edge("answer", END)
builder.add_edge("answer_with_fallback", END)
builder.add_edge("fallback", END)

graph = builder.compile()

@observe("run_rag", capture_input=False, capture_output=False)
def run_rag(
    query: str,
    chat_history: list[dict] | None = None,
    scope: str | None = None,
    doc_paths: list[str] | None = None,
    top_k: int | None = None,
    allow_external: bool = True,
    thinking: bool = False,
    intent: str | None = None,
    queries: list[str] | None = None,
) -> RAGState:
    query = clean_text(query)
    history = chat_history or []
    with trace_context(
        trace_name="run_rag",
        tags=["rag"],
        metadata={"query": query, "turns": len(history) // 2, "scope": scope, "doc_paths": doc_paths},
    ):
        result = graph.invoke(
            {
                "query": query,
                "chat_history": history,
                "scope": scope,
                "doc_paths": doc_paths,
                "top_k": top_k,
                "allow_external": allow_external,
                "thinking": thinking,
                "intent": intent,
                "queries": queries or [],
            }
        )
    answer = result.get("answer", "")
    update_current_span(output=answer, query=query, num_sources=len(result.get("sources", [])))
    set_trace_io(input=query, output=answer)
    return result


def run_cli() -> None:
    """Interactive terminal harness: ask, inspect sources, read rendered answers."""
    import time

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    console = Console()
    console.print(
        Panel.fit(
            "[bold]LangGraph RAG[/bold] — hỏi đáp tài liệu LangChain/LangGraph\n"
            "Hỏi nối tiếp được (giữ ngữ cảnh). Lệnh: "
            "[cyan]:reset[/cyan] xoá lịch sử · [cyan]:web[/cyan] bật/tắt web · "
            "[cyan]:think[/cyan] bật/tắt suy luận · [cyan]exit[/cyan]/[cyan]quit[/cyan] thoát.",
            border_style="cyan",
        )
    )

    history: list[dict] = []
    allow_external = False
    thinking = False

    while True:
        turns = len(history) // 2
        prompt = f"\n[bold cyan]Question[/bold cyan][dim]({turns} turns)[/dim][bold cyan] >[/bold cyan] "
        try:
            query = clean_text(console.input(prompt)).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        lowered = query.lower()
        if lowered in {"exit", "quit", ":q"}:
            break
        if lowered in {":reset", "reset", "clear"}:
            history.clear()
            console.print("[dim]Đã xoá lịch sử hội thoại.[/dim]")
            continue
        if lowered in {":web", "web"}:
            allow_external = not allow_external
            console.print(f"[dim]Web search: {'ON' if allow_external else 'OFF'}[/dim]")
            continue
        if lowered in {":think", "think"}:
            thinking = not thinking
            console.print(f"[dim]Thinking: {'ON' if thinking else 'OFF'}[/dim]")
            continue
        if not query:
            continue

        started = time.perf_counter()
        console.print("[dim]Đang truy xuất và sinh câu trả lời...[/dim]")
        try:
            result = run_rag(
                query,
                chat_history=history,
                allow_external=allow_external,
                thinking=thinking,
            )
        except Exception as error:  # keep the REPL alive on provider failures
            console.print(f"[bold red]Lỗi:[/bold red] {error}")
            continue
        elapsed = time.perf_counter() - started

        standalone = result.get("standalone_query", query)
        if standalone and standalone != query:
            console.print(f"[dim]↳ Truy vấn độc lập:[/dim] [italic]{standalone}[/italic]")

        console.print(Rule("[bold green]Answer[/bold green]", style="green"))
        console.print(Markdown(result.get("answer", "")))

        sources = result.get("sources", [])
        if sources:
            table = Table(title=f"Sources ({len(sources)})", show_lines=False, expand=True)
            table.add_column("#", justify="right", style="dim", no_wrap=True)
            table.add_column("Title", style="bold", overflow="fold")
            table.add_column("Section", overflow="fold")
            table.add_column("URL", style="cyan", overflow="fold")
            for index, source in enumerate(sources, start=1):
                table.add_row(
                    str(index),
                    source.get("title", ""),
                    source.get("section_title_clean") or source.get("section_heading", ""),
                    source.get("url", ""),
                )
            console.print(table)

        console.print(f"[dim]⏱  {elapsed * 1000:.0f} ms[/dim]")

        answer = result.get("answer", "")
        if answer and not answer.startswith("I could not find"):
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})

    flush_traces()
    console.print("[dim]Đã thoát.[/dim]")


if __name__ == "__main__":
    run_cli()
