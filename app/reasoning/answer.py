from app.retrieval.retrieve import vector_search
import json
import os
import re
import time
import requests
from google import genai
from google.genai import types
from app.core.config import (
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    GENERATION_BACKOFF_BASE,
    GENERATION_EMPTY_RETRY_DELAY,
    GENERATION_MAX_RETRIES,
    GENERATION_PROVIDER,
    GENERATION_TIMEOUT,
    GRADE_CHUNK_CHARS,
    HISTORY_MAX_MESSAGES,
    HISTORY_MAX_MESSAGES_CLASSIFY,
    MAX_TOKENS,
    MODEL_COOLDOWN_SECONDS,
    OPENROUTER_BASE_URL,
    OPENROUTER_GENERATION_MODELS,
    OPENROUTER_UTILITY_MODELS,
    RETRYABLE_STATUS_CODES,
    SECTION_IDS_MAX,
    SECTION_QUERY_MIN_TITLE_CHARS,
    SECTION_QUERY_MIN_TITLE_WORDS,
)
from app.core.prompts import (
    build_answer_prompt,
    build_condense_prompt,
    build_grade_prompt,
    build_rewrite_prompt,
    build_route_prompt,
    build_section_route_prompt,
)
from app.core.models import AnswerResult, ExternalSource, RetrievedChunk
from app.core.sections import split_section_number
from dotenv import load_dotenv
load_dotenv()

_client = None


def _get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or API_KEY to use the Gemini provider.")
        _client = genai.Client(api_key=api_key)
    return _client

def format_context(results: list[RetrievedChunk | ExternalSource]) -> str:
    """
    Format the context from the search results into a string that can be used for answering the question.
    """
    context_parts = []
    for i, result in enumerate(results, start=1):
        title = result.get("title", "No Title")
        url = result.get("url", "No URL")
        section_heading = (
            result.get("section_title_clean") or result.get("section_heading", "")
        )
        breadcrumbs = " > ".join(
            result.get("section_titles") or result.get("breadcrumbs") or []
        )
        text = result.get("text", "")
        score = result.get("score", 0.0)
        context_part = (
            f"[Source {i}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Section: {section_heading}\n"
            f"Breadcrumbs: {breadcrumbs}\n"
            f"Score: {score:.4f}\n"
            f"Text: {text}"
        )
        context_parts.append(context_part)

    return "\n\n".join(context_parts)

# Models that recently failed are skipped for a while, so every call does not
# waste time re-trying a model that is rate-limited or down.
_MODEL_COOLDOWN: dict[str, float] = {}
COOLDOWN_SECONDS = MODEL_COOLDOWN_SECONDS


def _ready_models(models: list[str]) -> list[str]:
    now = time.time()
    ready = [m for m in models if _MODEL_COOLDOWN.get(m, 0.0) <= now]
    return ready or list(models)  # all cooling down -> try anyway


def _cooldown(model: str) -> None:
    _MODEL_COOLDOWN[model] = time.time() + COOLDOWN_SECONDS


def _generate_openrouter(
    prompt: str,
    models: list[str] | None = None,
    max_retries: int = GENERATION_MAX_RETRIES,
    max_tokens: int = 4096,
    system: str | None = None,
    thinking: bool = False,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY to use the OpenRouter provider.")
    models = _ready_models(models or OPENROUTER_GENERATION_MODELS)
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        # Let OpenRouter route to another provider when the first one flakes.
        "provider": {"allow_fallbacks": True},
        # Thinking mode: let reasoning-capable models spend hidden tokens before
        # answering. Off by default so a plain Q&A stays fast; ignored by models
        # that lack reasoning support.
        "reasoning": {"enabled": bool(thinking)},
    }

    last_error = "no models configured"
    for model in models:
        payload["model"] = model
        for attempt in range(max_retries):
            response = requests.post(url, headers=headers, json=payload, timeout=GENERATION_TIMEOUT)

            if response.status_code in RETRYABLE_STATUS_CODES:
                last_error = f"{model}: HTTP {response.status_code}"
                if attempt == max_retries - 1:
                    _cooldown(model)
                    print(f"generate: {model} failed ({last_error}), cooling down {COOLDOWN_SECONDS}s")
                    break
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else GENERATION_BACKOFF_BASE * (2**attempt)
                print(f"generate: {model} retry {attempt + 1}/{max_retries} after {delay:.0f}s (HTTP {response.status_code})")
                time.sleep(delay)
                continue

            response.raise_for_status()
            body = response.json()
            choice = (body.get("choices") or [{}])[0]
            content = (choice.get("message", {}).get("content") or "").strip()

            if content:
                used = body.get("model", model)
                provider = body.get("provider")
                suffix = f" (provider={provider})" if provider else ""
                print(f"generate: used model {used}{suffix}")
                return content

            # Empty content: usually a flaky provider, or reasoning tokens ate the
            # budget. Retry the same model before giving up on it.
            last_error = f"{model}: empty content"
            print(
                f"generate: {model} empty "
                f"(finish={choice.get('finish_reason')}, usage={body.get('usage')})"
            )
            if attempt < max_retries - 1:
                time.sleep(GENERATION_EMPTY_RETRY_DELAY)
                continue
            _cooldown(model)
            break

    raise RuntimeError(f"generate(openrouter) failed for all models: {last_error}")


def _gemini_thinking_config(thinking: bool) -> types.GenerateContentConfig:
    """Explicit thinking configuration for the Gemini fallback provider.

    Gemini 3 controls reasoning with ``thinking_level`` (no hard "off" — "low"
    is the minimum); Gemini 2.5 and older use a token budget where 0 disables
    thinking and -1 lets the model decide.
    """
    if DEFAULT_MODEL.startswith("gemini-3"):
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="high" if thinking else "low",
                include_thoughts=False,
            )
        )
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1 if thinking else 0,
            include_thoughts=False,
        )
    )


def _generate(
    prompt: str,
    models: list[str] | None = None,
    max_tokens: int = 4096,
    system: str | None = None,
    thinking: bool = False,
) -> str:
    """Send a raw prompt to the configured generation provider.

    ``system`` is sent as a real system message when the provider supports it
    (OpenRouter); the Gemini fallback prepends it to the prompt. ``thinking``
    turns on model reasoning for this call (answer generation only — the fast
    router/grader helpers keep it off).
    """
    if GENERATION_PROVIDER == "openrouter":
        return _generate_openrouter(
            prompt,
            models=models,
            max_tokens=max_tokens,
            system=system,
            thinking=thinking,
        )
    contents = f"{system}\n\n{prompt}" if system else prompt
    response = _get_gemini_client().models.generate_content(
        model=DEFAULT_MODEL,
        contents=contents,
        config=_gemini_thinking_config(thinking),
    )
    return getattr(response, "text", "").strip()


def generate_answer(query: str, context: str, thinking: bool = False) -> str:
    task = "answer_thinking" if thinking else "answer"
    return _generate(
        build_answer_prompt(query, context),
        max_tokens=MAX_TOKENS[task],
        thinking=thinking,
    )


def format_history(history: list[dict], max_messages: int = HISTORY_MAX_MESSAGES) -> str:
    """Render the most recent turns as plain text for the prompt."""
    lines = []
    for message in history[-max_messages:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {message.get('content', '')}")
    return "\n".join(lines)

def condense_question(question: str, history: list[dict]) -> str:
    """Resolve a follow-up question into a standalone one using chat history.

    Skips the LLM call entirely when there is no history, so single-turn usage
    stays free of extra latency and API cost.
    """
    if not history:
        return question
    prompt = build_condense_prompt(question, format_history(history))
    try:
        rewritten = _generate(
            prompt, models=OPENROUTER_UTILITY_MODELS, max_tokens=MAX_TOKENS["condense"]
        ).strip().strip('"').strip()
    except Exception as error:
        print(f"condense_question failed, using original question: {error}")
        return question

    # Some models echo the follow-up before the rewrite; drop that prefix.
    if rewritten.lower().startswith(question.lower()):
        rewritten = rewritten[len(question):].lstrip(" :.-?")
    return rewritten.strip() or question

def answer_question(query: str, top_k: int = DEFAULT_TOP_K) -> AnswerResult:
    """
    Answer a question based on the retrieved context and return the answer along with relevant context.
    """
    query = query.strip()
    if not query:
        return {
            "query": query,
            "answer": "Question must not be empty.",
            "context": "",
            "sources": [],
        }

    # Search for relevant records
    results = vector_search(query, top_k=top_k)

    if not results:
        return {
            "query": query,
            "answer": "No relevant information found.",
            "context": "",
            "sources": [],
        }

    # Resolve table_stub hits to their raw rows before prompting, so the model
    # sees cell values rather than only the compact stub. Imported here to avoid
    # a cycle (table_tools imports _generate from this module).
    from app.reasoning.image_tools import hydrate_image_stubs
    from app.reasoning.table_tools import hydrate_table_stubs

    results = hydrate_table_stubs(results)
    results = hydrate_image_stubs(results)
    context = format_context(results)
    answer = generate_answer(query, context)

    return {
        "query": query,
        "answer": answer,
        "context": context,
        "sources": results,
    }

def classify_intent(query: str, history: list[dict] | None = None) -> str:
    """Route a message to "chat", "followup" or "question".

    - "chat": social/meta, no retrieval.
    - "followup": continues the conversation; retrieval should stay on the
      documents already in play.
    - "question": a new standalone question; retrieval should search everything.

    Language-agnostic: the utility model decides from the message itself, so no
    hard-coded word lists. Fails safe to "question" (search everything).
    """
    query = (query or "").strip()
    if not query:
        return "chat"
    prompt = build_route_prompt(query, format_history(history or [], max_messages=HISTORY_MAX_MESSAGES_CLASSIFY))
    try:
        raw = _generate(prompt, models=OPENROUTER_UTILITY_MODELS, max_tokens=MAX_TOKENS["classify_intent"]).lower()
    except Exception as error:
        print(f"classify_intent failed, defaulting to question: {error}")
        return "question"
    match = re.search(r"\b(chat|followup|question|workspace)\b", raw)
    return match.group(1) if match else "question"


def _section_ids_from_response(raw: str, sections: list[dict]) -> list[str]:
    """Parse/validate the resolver's JSON, with a transparent title fallback."""
    valid = {str(section.get("id")): section for section in sections if section.get("id")}
    candidates: list[str] = []
    is_section_request: bool | None = None

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            flag = data.get("is_section_request")
            if isinstance(flag, bool):
                is_section_request = flag
            elif isinstance(flag, str) and flag.strip().lower() in ("true", "false"):
                is_section_request = flag.strip().lower() == "true"
            value = data.get("section_ids") or data.get("sections") or data.get("ids")
            if isinstance(value, str):
                candidates = [part.strip() for part in value.split(",") if part.strip()]
            elif isinstance(value, list):
                candidates = [str(item) for item in value]

    # An explicit "this is not a section request" always wins, even if the model
    # also emitted ids by mistake.
    if is_section_request is False:
        return []

    # Tolerate a bare list / comma-separated ids, and expand a value like
    # "s4, s4.1" if it slipped through as one string.
    expanded: list[str] = []
    for candidate in candidates:
        found = re.findall(r"\bs\d+(?:\.\d+)*\b", str(candidate))
        expanded.extend(found or [str(candidate)])
    if not expanded:
        expanded = re.findall(r"\bs\d+(?:\.\d+)*\b", raw)
    if not expanded and raw.strip() and raw.strip().upper() != "NONE":
        # Rare fallback: the model echoed a title instead of an id.
        target = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
        for section in sections:
            title = re.sub(
                r"[^a-z0-9]+", " ", str(section.get("title", "")).lower()
            ).strip()
            if title and (title == target or title in target or target in title):
                expanded = [str(section.get("id"))]
                break

    titles = {
        re.sub(r"[^a-z0-9]+", " ", str(section.get("title", "")).lower()).strip(): str(
            section.get("id")
        )
        for section in sections
        if section.get("id")
    }

    selected: list[str] = []
    for section_id in expanded:
        section_id = str(section_id).strip()
        if section_id not in valid:
            # The model may have returned a title inside the JSON; map it.
            key = re.sub(r"[^a-z0-9]+", " ", section_id.lower()).strip()
            section_id = titles.get(key, "")
            if not section_id:
                continue
        if section_id in selected:
            continue
        # A parent already covers its descendants; do not ask for both.
        if any(section_id.startswith(parent + ".") for parent in selected):
            continue
        selected = [
            parent for parent in selected if not parent.startswith(section_id + ".")
        ]
        selected.append(section_id)
    return selected[:SECTION_IDS_MAX]


# A cheap, deterministic gate in front of the resolver. The primary signals are
# language-independent: a section number (4.1, A.1, or a bare number that matches
# a number in THIS document) or a multi-word title from THIS document. The
# English/Vietnamese structural words are only an accelerator; they are not the
# contract. The resolver itself is multilingual once it is called.
_SECTION_WORD = re.compile(
    # Do not match a compound like "cross-section" or "sectional".
    r"(?<![\w-])(section|sections|subsection|chapter|mục|phần|chương)(?![\w-])",
    re.IGNORECASE,
)
# Dotted number, optionally letter-prefixed: 4.1, 2.3.1, A.1, B.2.
_DOTTED_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z]?\d+(?:\.\d+)+(?![A-Za-z0-9])"
)
# Bare number, including CJK adjacency ("第4章"), but not a model version
# ("GPT-4", "model4"). Used only when the number exists in the outline.
_BARE_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.-])(\d+)(?![A-Za-z0-9_])"
)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _looks_like_section_query(query: str, sections: list[dict]) -> bool:
    query = query or ""
    if _SECTION_WORD.search(query) or _DOTTED_NUMBER.search(query):
        return True

    # A bare number is a section reference only when it matches a number that
    # actually exists in this document. This works across languages:
    # "section 4", "chapitre 4", "Abschnitt 4", "第4章", "mục 4".
    section_numbers = {
        str(section.get("number") or "").strip()
        for section in sections
        if str(section.get("number") or "").strip()
    }
    for match in _BARE_NUMBER.finditer(query):
        if match.group(1) in section_numbers:
            return True

    normalized_query = _normalize_text(query)
    for section in sections:
        title = str(section.get("title") or "")
        _, title_without_number = split_section_number(title)
        candidate = _normalize_text(title_without_number or title)
        # A single generic word ("State", "Memory", "Results") is too weak a
        # signal; require a specific multi-word title.
        if len(candidate) < SECTION_QUERY_MIN_TITLE_CHARS or len(candidate.split()) < SECTION_QUERY_MIN_TITLE_WORDS:
            continue
        if candidate in normalized_query:
            return True
    return False


def resolve_section(query: str, title: str, sections: list[dict]) -> list[str]:
    """Self-query: which canonical section(s) is the user asking about?

    ``sections`` is the flattened outline (``id``, ``title``, ``level``). The
    model returns ids, so the result is a metadata filter rather than raw heading
    matching. A parent id intentionally expands to its whole subtree in
    ``section_chunks``; multiple ids support "compare section 3 and section 4".
    """
    if not sections or not _looks_like_section_query(query, sections):
        return []

    outline_lines = []
    for section in sections:
        level = max(int(section.get("level") or 1), 1)
        outline_lines.append(
            "  " * (level - 1)
            + f"[{section.get('id')}] {section.get('title', '')}"
        )
    prompt = build_section_route_prompt(title, "\n".join(outline_lines), query)
    try:
        raw = _generate(
            prompt, models=OPENROUTER_UTILITY_MODELS, max_tokens=MAX_TOKENS["resolve_section"]
        ).strip()
    except Exception as error:
        print(f"resolve_section failed, skipping section filter: {error}")
        return []
    if not raw or raw.strip().upper().startswith("NONE"):
        return []
    return _section_ids_from_response(raw, sections)


def grade_documents(question, chunks, max_chars=GRADE_CHUNK_CHARS):
    listing = []
    for i, c in enumerate(chunks):
        text = (c.get("text") or "")[:max_chars].replace("\n", " ")
        listing.append(
            f"[{i}] {c.get('section_title_clean') or c.get('section_heading', '')} | {text}"
        )
    prompt = build_grade_prompt(question, listing)
    raw = _generate(prompt, models=OPENROUTER_UTILITY_MODELS, max_tokens=MAX_TOKENS["grade"])
    m = re.search(r'"relevant_indices"\s*:\s*\[([^\]]*)\]', raw)
    if not m: 
        print(f"grade_documents: unparsable response, failing open: {raw[:120]!r}")
        return list(range(len(chunks)))
    return [int(x) for x in re.findall(r"\d+", m.group(1)) if int(x) < len(chunks)]

def rewrite_query(question):
    prompt = build_rewrite_prompt(question)
    return _generate(prompt, models=OPENROUTER_UTILITY_MODELS, max_tokens=MAX_TOKENS["rewrite"]).strip() or question.strip() or question


if __name__ == "__main__":
    query = input("Question: ").strip()
    if not query:
        raise SystemExit("Question must not be empty.")

    result = answer_question(query, top_k=DEFAULT_TOP_K)
    print("Answer:", result["answer"])
    print("\nSources:")
    for index, source in enumerate(result.get("sources", []), start=1):
        print(f"{index}. {source.get('title', 'No Title')} | {source.get('section_heading', '')}")
        print(f"   {source.get('url', 'No URL')}")
