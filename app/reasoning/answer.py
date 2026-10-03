from app.retrieval.retrieve import vector_search
import os
import re
import time
import requests
from google import genai
from app.core.config import (
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    GENERATION_PROVIDER,
    OPENROUTER_BASE_URL,
    OPENROUTER_GENERATION_MODELS,
    OPENROUTER_UTILITY_MODELS,
)
from app.core.prompts import (
    build_answer_prompt,
    build_condense_prompt,
    build_grade_prompt,
    build_rewrite_prompt
)
from app.core.models import AnswerResult, ExternalSource, RetrievedChunk
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
        section_heading = result.get("section_heading", "")
        breadcrumbs = " > ".join(result.get("breadcrumbs", []))
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
COOLDOWN_SECONDS = 300


def _ready_models(models: list[str]) -> list[str]:
    now = time.time()
    ready = [m for m in models if _MODEL_COOLDOWN.get(m, 0.0) <= now]
    return ready or list(models)  # all cooling down -> try anyway


def _cooldown(model: str) -> None:
    _MODEL_COOLDOWN[model] = time.time() + COOLDOWN_SECONDS


def _generate_openrouter(
    prompt: str,
    models: list[str] | None = None,
    max_retries: int = 3,
    max_tokens: int = 4096,
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
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # Let OpenRouter route to another provider when the first one flakes.
        "provider": {"allow_fallbacks": True},
        # Reasoning models otherwise spend seconds (and tokens) on hidden
        # chain-of-thought before answering. Ignored by models that lack it.
        "reasoning": {"enabled": False},
    }

    last_error = "no models configured"
    for model in models:
        payload["model"] = model
        for attempt in range(max_retries):
            response = requests.post(url, headers=headers, json=payload, timeout=180)

            if response.status_code in (429, 500, 502, 503, 504, 529):
                last_error = f"{model}: HTTP {response.status_code}"
                if attempt == max_retries - 1:
                    _cooldown(model)
                    print(f"generate: {model} failed ({last_error}), cooling down {COOLDOWN_SECONDS}s")
                    break
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 3.0 * (2**attempt)
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
                time.sleep(2)
                continue
            _cooldown(model)
            break

    raise RuntimeError(f"generate(openrouter) failed for all models: {last_error}")


def _generate(prompt: str, models: list[str] | None = None, max_tokens: int = 4096) -> str:
    """Send a raw prompt to the configured generation provider."""
    if GENERATION_PROVIDER == "openrouter":
        return _generate_openrouter(prompt, models=models, max_tokens=max_tokens)
    response = _get_gemini_client().models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
    )
    return getattr(response, "text", "").strip()


def generate_answer(query: str, context: str) -> str:
    return _generate(build_answer_prompt(query, context), max_tokens=700)


def format_history(history: list[dict], max_messages: int = 6) -> str:
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
            prompt, models=OPENROUTER_UTILITY_MODELS, max_tokens=128
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

    # Format the context from the search results 
    context = format_context(results)
    answer = generate_answer(query, context)

    return {
        "query": query,
        "answer": answer,
        "context": context,
        "sources": results,
    }

def grade_documents(question, chunks, max_chars=700):
    listing = []
    for i, c in enumerate(chunks):
        text = (c.get("text") or "")[:max_chars].replace("\n", " ")
        listing.append(f"[{i}] {c.get('section_heading', '')} | {text}")
    prompt = build_grade_prompt(question, listing)
    raw = _generate(prompt, models=OPENROUTER_UTILITY_MODELS, max_tokens=256)
    m = re.search(r'"relevant_indices"\s*:\s*\[([^\]]*)\]', raw)
    if not m: 
        print(f"grade_documents: unparsable response, failing open: {raw[:120]!r}")
        return list(range(len(chunks)))
    return [int(x) for x in re.findall(r"\d+", m.group(1)) if int(x) < len(chunks)]

def rewrite_query(question):
    prompt = build_rewrite_prompt(question)
    return _generate(prompt, models=OPENROUTER_UTILITY_MODELS, max_tokens=128).strip() or question.strip() or question


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
