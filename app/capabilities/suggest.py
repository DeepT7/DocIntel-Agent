"""Capability: grounded writing suggestions for the Quick window.

Combines the user's request and the text they are writing (clipboard) with
retrieved excerpts from the scope, then asks the model for a suggestion that
cites those excerpts.
"""

from app.capabilities.search import search
from app.core.config import DEFAULT_TOP_K
from app.core.prompts import build_suggest_prompt
from app.reasoning.answer import format_context, _generate

MAX_CLIPBOARD_CHARS = 2000


def suggest(
    scope: str,
    query: str,
    clipboard: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    query = (query or "").strip()
    clipboard = (clipboard or "").strip()[:MAX_CLIPBOARD_CHARS]

    if not query and not clipboard:
        return {"suggestion": "", "sources": []}

    # Selection/clipboard helps the retrieval query find related material too.
    search_query = " ".join(part for part in (query, clipboard) if part).strip()
    results = search(scope, search_query, top_k=top_k)

    context = format_context(results) if results else "(no matching documents)"
    prompt = build_suggest_prompt(
        query=query or "(no explicit request)",
        clipboard=clipboard or "(empty)",
        context=context,
    )
    suggestion = _generate(prompt, max_tokens=800)
    return {"suggestion": suggestion, "sources": results}