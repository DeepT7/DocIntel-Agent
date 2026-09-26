from app.retrieval.retrieve import vector_search
import os
import time
import requests
from google import genai
from app.core.config import (
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    GENERATION_PROVIDER,
    OPENROUTER_BASE_URL,
    OPENROUTER_GENERATION_MODELS,
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


def build_prompt(query: str, context: str) -> str:
    """
    Build a prompt for the language model based on the query and context.
    """
    prompt = (
        "You are a documentation assistant.\n\n" 
        "Use the retrieved context as the primary source of truth.\n"
        "You may also use your general knowledge to provide additional explanation when it helps, but do not present unsupported details as if they were confirmed by the retrieved context.\n\n"

        "If the retrieved context clearly supports the answer:\n"
        "- answer directly\n"
        "- mention the most relevant sources"

        "\n\nIf the retrieved context only partially supports the answer:"
        "- answer with caution\n"
        "- clearly distinguish what is supported by the retrieved context and what is based on general knowledge or inference\n"
        "Do not invent specific APIs, parameters, or behaviors that are not grounded in the retrieved context."

        "User question:\n{query}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Answer:"
    )

    return prompt.format(query=query, context=context)


def _generate_openrouter(
    prompt: str,
    models: list[str] | None = None,
    max_retries: int = 3,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY to use the OpenRouter provider.")
    models = models or OPENROUTER_GENERATION_MODELS
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        # Generous cap: reasoning models spend tokens on hidden reasoning before
        # emitting content, and a low cap can leave the content empty.
        "max_tokens": 4096,
    }

    last_error = "no models configured"
    for model in models:
        for attempt in range(max_retries):
            payload["model"] = model
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            if response.status_code in (429, 500, 502, 503, 504, 529):
                last_error = f"{model}: HTTP {response.status_code}"
                if attempt == max_retries - 1:
                    print(f"generate: {model} failed ({last_error}), trying next model")
                    break
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 3.0 * (2**attempt)
                print(f"generate: {model} retry {attempt + 1}/{max_retries} after {delay:.0f}s (HTTP {response.status_code})")
                time.sleep(delay)
                continue
            response.raise_for_status()
            choices = response.json().get("choices", [])
            content = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
            if content:
                return content
            last_error = f"{model}: empty content"
            print(f"generate: {model} returned empty content, trying next model")
            break

    raise RuntimeError(f"generate(openrouter) failed for all models: {last_error}")


def _generate(prompt: str) -> str:
    """Send a raw prompt to the configured generation provider."""
    if GENERATION_PROVIDER == "openrouter":
        return _generate_openrouter(prompt)
    response = _get_gemini_client().models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
    )
    return getattr(response, "text", "").strip()


def generate_answer(query: str, context: str) -> str:
    return _generate(build_prompt(query, context))


def format_history(history: list[dict], max_messages: int = 6) -> str:
    """Render the most recent turns as plain text for the prompt."""
    lines = []
    for message in history[-max_messages:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {message.get('content', '')}")
    return "\n".join(lines)


def build_condense_prompt(question: str, history_text: str) -> str:
    return (
        "Given the conversation so far and a follow-up question, rewrite the "
        "follow-up into a single, self-contained question that can be used to "
        "search a documentation index.\n"
        "Keep the original language, all entities, and the full intent. Do not "
        "answer the question. Do not repeat or quote the follow-up question. "
        "If it is already self-contained, return it unchanged.\n"
        "Return ONLY the rewritten question, with no preamble, on one line.\n\n"
        f"Conversation:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )


def condense_question(question: str, history: list[dict]) -> str:
    """Resolve a follow-up question into a standalone one using chat history.

    Skips the LLM call entirely when there is no history, so single-turn usage
    stays free of extra latency and API cost.
    """
    if not history:
        return question
    prompt = build_condense_prompt(question, format_history(history))
    try:
        rewritten = _generate(prompt).strip().strip('"').strip()
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
