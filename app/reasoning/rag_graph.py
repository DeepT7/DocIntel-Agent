import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, START, END

from app.core.models import ExternalSource, RAGState
from app.reasoning.answer import condense_question, format_context, generate_answer, grade_documents, rewrite_query
from app.core.config import DEFAULT_MODEL, DEFAULT_TOP_K
from app.retrieval.retrieve import hybrid_search, vector_search
from app.core.tracing import observe, update_current_span, set_trace_io, trace_context, flush_traces
from app.core.text import clean_text

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY or API_KEY before running rag_graph.py.")
client = genai.Client(api_key=api_key)

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
    # results = hybrid_search(query, top_k=DEFAULT_TOP_K)
    results = vector_search(query, top_k=DEFAULT_TOP_K)
    update_current_span(
        output=[
            {"chunk_id": r.get("chunk_id"), "title": r.get("title"), "score": round(r.get("score", 0.0), 4)}
            for r in results
        ],
        query=query,
        num_chunks=len(results),
    )
    return {
        "retrieved_chunks": results,
    }

@observe("grade_node", capture_output=False)
def grade_node(state: RAGState) -> dict:
    query = state.get("standalone_query") or state["query"]
    chunks = state.get("retrieved_chunks", [])
    top_score = chunks[0].get("score", 0.0) if chunks else 0.0
    # Skip the (slow) LLM grading call when retrieval is already confident.
    if top_score >= 0.5:
        update_current_span(num_chunks=len(chunks), num_relevant=len(chunks), gated=True)
        return {"relevant_chunks": chunks, "graded_ok": bool(chunks)}
    indices = grade_documents(query, chunks)
    relevant = [chunks[i] for i in indices] if chunks else []
    update_current_span(num_chunks=len(chunks), num_relevant=len(relevant), gated=False)
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
    if state.get("rewrite_count", 0) < 1: # Permitted to rewrite one time
        return "rewrite"
    return "external_search"

@observe("answer_node", capture_output=False)
def answer_node(state: RAGState) -> dict:
    query = state.get("standalone_query") or state["query"]
    retrieved_chunks = state.get("relevant_chunks") or state.get("retrieved_chunks", [])
    context = format_context(retrieved_chunks)
    answer = generate_answer(query, context)
    update_current_span(output=answer, num_sources=len(retrieved_chunks), answer_chars=len(answer))

    return {
        "context": context,
        "answer": answer,
        "sources": retrieved_chunks,
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
    internal_chunks = state.get("relevant_chunks") or  state.get("retrieved_chunks", [])
    internal_context = format_context(internal_chunks) if internal_chunks else ""
    external_context = state.get("external_context", "").strip()

    context_parts = []
    if internal_context:
        context_parts.append("Internal knowledge base context:\n" + internal_context)
    if external_context:
        context_parts.append("External grounded web context:\n" + external_context)

    combined_context = "\n\n".join(part for part in context_parts if part).strip()
    answer = generate_answer(query, combined_context)
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
builder.add_node("condense", condense_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("answer", answer_node)
builder.add_node("grade", grade_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("external_search", external_search_node)
builder.add_node("answer_with_fallback", answer_with_fallback_node)
builder.add_node("fallback", fallback_node)

builder.add_edge(START, "condense")
builder.add_edge("condense", "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", route_after_grade,
                              {"answer": "answer", "rewrite": "rewrite", "external_search": "external_search"})
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
def run_rag(query: str, chat_history: list[dict] | None = None) -> RAGState:
    query = clean_text(query)
    history = chat_history or []
    with trace_context(
        trace_name="run_rag",
        tags=["rag"],
        metadata={"query": query, "turns": len(history) // 2},
    ):
        result = graph.invoke({"query": query, "chat_history": history})
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
            "[cyan]:reset[/cyan] xoá lịch sử · [cyan]exit[/cyan]/[cyan]quit[/cyan] thoát.",
            border_style="cyan",
        )
    )

    history: list[dict] = []

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
        if not query:
            continue

        started = time.perf_counter()
        console.print("[dim]Đang truy xuất và sinh câu trả lời...[/dim]")
        try:
            result = run_rag(query, chat_history=history)
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
                    source.get("section_heading", ""),
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
