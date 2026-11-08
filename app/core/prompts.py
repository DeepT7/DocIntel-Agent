"""Central prompt store.

Every LLM prompt template lives here so prompts can be reviewed and tuned in one
place. Builders fill templates with explicit ``__PLACEHOLDER__`` tokens using
``str.replace`` -- never ``str.format`` -- because prompt bodies contain literal
braces (JSON examples), which ``str.format`` would try to interpret.
"""


def _fill(template: str, **values) -> str:
    for key, value in values.items():
        template = template.replace(f"__{key.upper()}__", value)
    return template


# --------------------------------------------------------------------------- #
# Answer generation
# --------------------------------------------------------------------------- #
ANSWER_PROMPT = """You are a documentation assistant.

Use the retrieved context as the primary source of truth.
You may also use your general knowledge to provide additional explanation when it
helps, but do not present unsupported details as if they were confirmed by the
retrieved context.

If the retrieved context clearly supports the answer:
- answer directly
- mention the most relevant sources
- when the question asks for specific models, names, numbers, or parameters, quote
  them exactly as written in the context (do not generalize or omit them)
- cite the source number, e.g. [Source 2], right after each specific fact
- if the context spans several documents, name the document each fact comes from

If the retrieved context only partially supports the answer:
- answer with caution
- clearly distinguish what is supported by the retrieved context and what is based
  on general knowledge or inference
Do not invent specific APIs, parameters, or behaviors that are not grounded in the
retrieved context.

User question:
__QUERY__

Retrieved context:
__CONTEXT__

Answer:
"""


def build_answer_prompt(query: str, context: str) -> str:
    return _fill(ANSWER_PROMPT, query=query, context=context)


# --------------------------------------------------------------------------- #
# Table tool: fetch full raw tables before answering
# --------------------------------------------------------------------------- #
TABLE_TOOL_PROMPT = """You decide whether the answer needs full raw table rows.

The context below may contain table stubs (columns and row entities only) or
tables truncated for length. You may request the complete raw table by id.

Available tables (id | table number | caption | columns | rows | file):
__TABLES__

Current context:
__CONTEXT__

User question:
__QUERY__

Rules:
- Request a table ONLY when its exact rows/cells are needed to answer and the
  context does not already contain them.
- Prefer the smallest set of tables that can answer the question.
- Use only ids listed above. Never invent an id.
- Return at most __MAX_TABLES__ ids.

Return ONLY JSON, no prose:
{"table_ids": ["<id>", ...]}
Use {"table_ids": []} when no fetch is needed.

JSON:"""


def build_table_tool_prompt(
    query: str, context: str, tables: str, max_tables: int
) -> str:
    return _fill(
        TABLE_TOOL_PROMPT,
        query=query,
        context=context or "(none)",
        tables=tables or "(none)",
        max_tables=str(max_tables),
    )


# --------------------------------------------------------------------------- #
# Intent routing (chat vs document question)
# --------------------------------------------------------------------------- #
# A language-agnostic router: the model classifies intent, so we never enumerate
# greetings/words. Mirrors how LlamaIndex RouterQueryEngine / agent tool-routing
# decide whether a turn needs retrieval.
ROUTE_PROMPT = """You route a message for a document Q&A assistant. Choose exactly one label.

- "chat": social or meta — greetings, thanks, acknowledgements, chit-chat, or
  questions about the assistant / how to use it. No document content is requested.
- "followup": the message continues the current topic and depends on earlier turns.
  It uses pronouns or ellipsis ("it", "they", "that paper", "why?", "and the data?"),
  or asks to compare documents already discussed, or builds on the last answer.
- "question": a new, self-contained question. It should be answered by searching the
  whole collection, and does not depend on earlier turns.
- "workspace": asks about the user's folder/collection itself, not document content —
  listing files, counting documents, what is in the folder, which folder is active.
  Answerable from workspace metadata.

Decide from the latest message; use the recent conversation only to tell a follow-up
from a new question. Answer with exactly one word: chat, followup, question, or workspace.

Examples (any language):
- "alo", "hola", "cảm ơn nhé", "ok" -> chat
- "bạn là ai?", "what can you do?" -> chat
- "list all files in this folder", "how many documents do I have?" -> workspace
- "liệt kê các file trong thư mục này" -> workspace
- "What datasets did they use?" (after discussing a paper) -> followup
- "why?", "còn short-term thì sao?" -> followup
- "compare the two papers" -> followup
- "Memory trong LangGraph chia thành mấy loại?" -> question
- "What is instruction tuning?" -> question

Recent conversation:
__HISTORY__

Latest message: __QUERY__

Label:"""


def build_route_prompt(query: str, history_text: str) -> str:
    return _fill(ROUTE_PROMPT, query=query, history=history_text or "(none)")


# --------------------------------------------------------------------------- #
# Self-querying: which section (if any) is the user asking about?
# --------------------------------------------------------------------------- #
# The section name is a FILTER over the document's real headings, not search
# text. The model picks from the actual outline, so it bridges the user's wording
# ("experiment" -> "Training Experiment and Setup") and answers NONE when the
# words are incidental ("introduction to X" as a topic, not the section). This
# replaces matching heading strings, which can neither understand intent nor
# synonyms.
SECTION_ROUTE_PROMPT = """You decide whether a user's message asks about specific sections of one document.

Document: __TITLE__

Sections of this document, in reading order. Indentation shows nesting; the id in brackets identifies each section:
__OUTLINE__

User message: __QUERY__

Return ONLY a JSON object, no prose:
{"is_section_request": true | false, "section_ids": ["<id>", ...]}

Rules:
- `is_section_request` is true ONLY when the message asks about a section of THIS document: by number ("section 4.1"), by title ("the experimental setup section"), or by a deictic reference ("in that section").
- A number used as a measurement, version, date, code line, percentage, or quantity is NOT a section reference: "5.2 seconds", "version 2.0", "GPT-4.1", "12%".
- A word like "section" inside a compound is NOT a section reference: "cross-section area", "sectional view".
- A general content question that merely contains a word appearing in a heading (for example "what datasets did they build?") is NOT a section request.
- If `is_section_request` is false, `section_ids` MUST be [].
- If the message asks about a parent section ("section 4"), return only the parent id. Its whole subtree is included automatically.
- Use only ids from the outline. Never invent an id.

Examples (any language):
- "section 5.2 nói gì?" -> {"is_section_request": true, "section_ids": ["<id of 5.2>"]}
- "5.2 seconds" -> {"is_section_request": false, "section_ids": []}
- "What is the cross-section area?" -> {"is_section_request": false, "section_ids": []}
- "what datasets did they build?" -> {"is_section_request": false, "section_ids": []}
- "the experimental setup section" -> {"is_section_request": true, "section_ids": ["<id>"]}
- "chapitre 4" -> {"is_section_request": true, "section_ids": ["<id of 4>"]}
- "第4章" -> {"is_section_request": true, "section_ids": ["<id of 4>"]}

JSON:"""


def build_section_route_prompt(title: str, outline: str, query: str) -> str:
    return _fill(
        SECTION_ROUTE_PROMPT,
        title=title or "(unknown)",
        outline=outline or "(none)",
        query=query,
    )


# --------------------------------------------------------------------------- #
# Workspace metadata (list files / counts) — grounded in the folder listing
# --------------------------------------------------------------------------- #
WORKSPACE_PROMPT = """You answer questions about the user's local workspace folder.
Use ONLY the file list below. Do not invent, add, or rename any file.
Reply in the user's language. If asked to list files, present them as a short
Markdown list, then state the total count.

Folder: __SCOPE__
Files (__COUNT__ total):
__FILES__

User: __QUERY__

Answer:"""


def build_workspace_prompt(scope: str, files: list[str], query: str) -> str:
    listing = "\n".join(f"- {name}" for name in files) or "(empty)"
    return _fill(
        WORKSPACE_PROMPT,
        scope=scope or "(no folder selected)",
        count=str(len(files)),
        files=listing,
        query=query,
    )


# --------------------------------------------------------------------------- #
# Small talk / meta (no retrieval)
# --------------------------------------------------------------------------- #
CHAT_PROMPT = """You are a friendly assistant for a local document workspace.
The user sent a short social or meta message (a greeting, acknowledgement, thanks,
or a question about you / how to use you). Reply briefly in the user's language.
Do NOT invent facts about their documents and do NOT use any retrieved content.
If they greeted you, greet back in one line and invite them to ask about their folder.

Conversation so far:
__HISTORY__

User: __QUERY__

Reply:"""


def build_chat_prompt(query: str, history_text: str) -> str:
    return _fill(CHAT_PROMPT, query=query, history=history_text or "(none)")


# --------------------------------------------------------------------------- #
# No-Context / Notes Only (every source switched off)
# --------------------------------------------------------------------------- #
NO_CONTEXT_SYSTEM = """You are a document assistant running in "No-Context / Notes Only" mode.

The user has switched every document source OFF for content Q&A. This only affects
reading document CONTENT; workspace metadata (the file list) is still available.
You must NOT read, quote, summarize or cite the content of any workspace file, and
you must not invent document content.

You may use:
1. the conversation history (for context and references, never as document content),
2. the open notes supplied by the user,
3. the workspace file names (metadata; safe to list and count),
4. your general knowledge.

Rules:
- Small talk (greetings, thanks, "what can you do?") -> reply briefly and naturally;
  do NOT mention documents or notes unless the user asks about them.
- If the user asks to list or count files, answer directly from the file list.
  Listing names is metadata, not reading content.
- If the open notes contain the answer -> use them and say the answer comes from the notes.
- If the question requires the content of a document ("summarize this file",
  "what does my report say about X", "tài liệu này nói gì") -> do not guess and do not
  reconstruct it from the conversation: tell the user, in their language, that no
  documents are enabled and ask them to enable at least one source.
- For any other general-knowledge answer, your FIRST sentence MUST say, in the user's
  language, that no documents are enabled so the answer is from general knowledge.
- Be honest about uncertainty. Reply in the user's language."""


NO_CONTEXT_PROMPT = """Conversation so far:
__HISTORY__

Files in this folder (__FILE_COUNT__ files; metadata — safe to list):
__FILES__

Open notes:
__NOTES__

User: __QUERY__

Answer:"""


def build_no_context_prompt(
    query: str, history_text: str, files_text: str, file_count: int, notes_text: str
) -> str:
    return _fill(
        NO_CONTEXT_PROMPT,
        history=history_text or "(none)",
        files=files_text or "(none)",
        file_count=str(file_count),
        notes=notes_text or "(none)",
        query=query,
    )


# --------------------------------------------------------------------------- #
# Follow-up condensation (multi-turn)
# --------------------------------------------------------------------------- #
CONDENSE_PROMPT = """Given the conversation so far and a follow-up question, rewrite the \
follow-up into a single, self-contained question that can be used to search a \
documentation index.
Keep the original language, all entities, and the full intent. Do not answer the \
question. Do not repeat or quote the follow-up question. If it is already \
self-contained, return it unchanged.
Return ONLY the rewritten question, with no preamble, on one line.

Conversation:
__HISTORY__

Follow-up question: __QUESTION__

Standalone question:
"""


def build_condense_prompt(question: str, history_text: str) -> str:
    return _fill(CONDENSE_PROMPT, question=question, history=history_text)


# --------------------------------------------------------------------------- #
# Corrective RAG: grade retrieved chunks
# --------------------------------------------------------------------------- #
GRADE_PROMPT = """You grade whether retrieved chunks are relevant to a question.

Question:
__QUESTION__

Chunks:
__CHUNKS__

A chunk is relevant if it contains information that helps answer the question.
Return ONLY JSON: {"relevant_indices": [<indices>]}  (empty list if none).
"""


def build_grade_prompt(question: str, chunk_lines: list[str]) -> str:
    return _fill(GRADE_PROMPT, question=question, chunks="\n\n".join(chunk_lines))


# --------------------------------------------------------------------------- #
# Corrective RAG: rewrite a query that failed to retrieve
# --------------------------------------------------------------------------- #
REWRITE_PROMPT = """The question below did not retrieve relevant documentation.
Rewrite it into a clearer, more specific search query that is likely to retrieve \
the right documents. Keep all entities and the original language.
Return ONLY the rewritten question, with no preamble.

Question: __QUESTION__
"""


def build_rewrite_prompt(question: str) -> str:
    return _fill(REWRITE_PROMPT, question=question)


# --------------------------------------------------------------------------- #
# Answer-level evaluation (LLM-as-judge)
# --------------------------------------------------------------------------- #
JUDGE_PROMPT = """You are a strict evaluator of a retrieval-augmented QA system.
Score the ANSWER on three criteria, each an integer from 1 to 5.

1. faithfulness: every factual claim in the ANSWER is supported by the RETRIEVED CONTEXT.
   - 5 = fully supported; 1 = mostly invented / unsupported.
2. answer_relevance: the ANSWER directly and completely addresses the QUESTION.
   - 5 = on-topic and complete; 1 = off-topic or does not answer.
3. citation_accuracy: sources cited in the ANSWER actually support the claims attributed to them.
   - 5 = all citations correct; 1 = citations wrong; use 3 if no citations are present.

Respond with ONLY a flat JSON object with exactly these keys:
{"faithfulness": <int>, "faithfulness_reason": "<string>",
 "answer_relevance": <int>, "answer_relevance_reason": "<string>",
 "citation_accuracy": <int>, "citation_accuracy_reason": "<string>"}

QUESTION:
__QUESTION__

RETRIEVED CONTEXT:
__CONTEXT__

ANSWER:
__ANSWER__
"""


def build_judge_prompt(question: str, context: str, answer: str) -> str:
    return _fill(JUDGE_PROMPT, question=question, context=context, answer=answer)


# --------------------------------------------------------------------------- #
# Insights: per-source summary artifacts (map-reduce)
# --------------------------------------------------------------------------- #
# An "insight" is a cached, reusable summary of one document. It is produced in
# one pass for short documents and map-reduce for long ones, then stored and
# cited like any other source. Inspired by Open Notebook's Insight concept.
INSIGHT_SINGLE_PROMPT = """Summarize the following document in the document's own language.
Return concise Markdown with: a 2-3 sentence overview, then the key points as
bullet points. Preserve numbers, proper nouns and technical terms.

Title: __TITLE__

Document:
__TEXT__
"""

INSIGHT_MAP_PROMPT = """Summarize this section of the document "__TITLE__" in the
document's own language as 3-5 bullet points. Keep only substantive content.

Section:
__TEXT__
"""

INSIGHT_REDUCE_PROMPT = """Merge the section summaries below into one coherent summary
of the document "__TITLE__" in the document's own language. Return concise Markdown:
a 2-3 sentence overview, then the key points as bullet points. Remove duplication.

Section summaries:
__PARTIALS__
"""


def build_insight_single_prompt(title: str, text: str) -> str:
    return _fill(INSIGHT_SINGLE_PROMPT, title=title, text=text)


def build_insight_map_prompt(title: str, text: str) -> str:
    return _fill(INSIGHT_MAP_PROMPT, title=title, text=text)


def build_insight_reduce_prompt(title: str, partials: list[str]) -> str:
    joined = "\n\n".join(f"- {part.strip()}" for part in partials if part.strip())
    return _fill(INSIGHT_REDUCE_PROMPT, title=title, partials=joined)


# --------------------------------------------------------------------------- #
# Document Registry: a per-file "card" (title, summary, keywords/entities)
# --------------------------------------------------------------------------- #
# Built once after indexing. Lets a file be found by *what it talks about* even
# when no chunk names it (e.g. "the Docker setup guide" for a file that only says
# "Open port 8502"). Also feeds document routing and gives the answer model a
# cheap overview without reading every chunk.
DOCUMENT_CARD_PROMPT = """Create a compact "document card" for one file so it can be found later by what it talks about, even when a chunk never names the file.

File name: __FILE_NAME__

Document:
__TEXT__

Return ONLY a JSON object, no prose:
{"title": "<the document's real title, in its own language>",
 "summary": "<3-5 sentences: what it is about, main topics, key facts/numbers>",
 "keywords": ["<5-12 important entities/terms: product names, versions, years, projects, people>"]}

JSON:"""


def build_document_card_prompt(file_name: str, text: str) -> str:
    return _fill(DOCUMENT_CARD_PROMPT, file_name=file_name or "(unknown)", text=text)


# --------------------------------------------------------------------------- #
# Turn router (self-querying): intent + which documents + sub-queries (Level 3)
# --------------------------------------------------------------------------- #
# ONE call replaces the old intent router *and* the title-matching ``_detect_docs``:
# a fast model reads the message + Document Cards and returns the intent and the
# documents the message is about — chosen by content, not by matching file names.
# ``target_doc_ids: []`` means "the whole folder" (no separate global/specific label).
TURN_ROUTER_PROMPT = """You route one message for a document Q&A assistant and decide which documents it is about.

Documents in the folder (id | file name | summary):
__CARDS__

Currently open documents (from the previous turn):
__CURRENT_DOCS__

Recent conversation:
__HISTORY__

Message: __QUERY__

Return ONLY a JSON object, no prose:
{"intent": "chat" | "followup" | "question" | "workspace",
 "scope_mode": "EXPLICIT_DOCS" | "INHERIT_PREVIOUS" | "GLOBAL_FOLDER",
 "target_doc_ids": [<ids>],
 "rewritten_queries": ["<query>", "<query>"]}

intent — pick one:
- "chat": social or meta (greeting, thanks, about the assistant). No document content.
- "followup": continues the conversation and depends on earlier turns. It is short, uses pronouns ("it", "they", "that paper", "the two"), or asks "why?", "and the data?", "compare the two". Classify as followup EVEN IF it also looks like a general question.
- "question": a new, self-contained question.
- "workspace": about the folder itself (list files, counts, which folder).

scope_mode — pick one:
- "EXPLICIT_DOCS": the message names or clearly describes one or more documents (file name, title, or content), including a NEW document that is not currently open. Put their ids in target_doc_ids; this list MUST be non-empty. If the message combines an open document with a new one ("compare it with C"), include BOTH ids.
- "INHERIT_PREVIOUS": the message continues the current thread (follow-up, deeper detail, "why?", "and the data?") and does NOT introduce a new document. Leave target_doc_ids [] (or repeat the open ids).
- "GLOBAL_FOLDER": the message asks about the whole folder, or switches to a new topic that does not match the currently open documents. Leave target_doc_ids [].

target_doc_ids — the documents the message is about (ids from the folder list above):
- only for EXPLICIT_DOCS; otherwise [].
- [] is invalid when scope_mode is EXPLICIT_DOCS.

rewritten_queries — only when scope_mode is EXPLICIT_DOCS or INHERIT_PREVIOUS and the answer will search specific documents: 1-3 self-contained search queries; split a comparison into one per side. Stay close to the message; do NOT add facts, numbers, names or constraints. Keep the user's language. Otherwise return [].

Examples:
- "hello" -> {"intent": "chat", "scope_mode": "GLOBAL_FOLDER", "target_doc_ids": [], "rewritten_queries": []}
- "list all files in this folder" -> {"intent": "workspace", "scope_mode": "GLOBAL_FOLDER", "target_doc_ids": [], "rewritten_queries": []}
- "what is instruction tuning?" -> {"intent": "question", "scope_mode": "GLOBAL_FOLDER", "target_doc_ids": [], "rewritten_queries": []}
- "the paper that imitates proprietary models — what datasets does it build?" -> {"intent": "question", "scope_mode": "EXPLICIT_DOCS", "target_doc_ids": [<its id>], "rewritten_queries": ["<same question>"]}
- (open: a paper) "what datasets did they use?" -> {"intent": "followup", "scope_mode": "INHERIT_PREVIOUS", "target_doc_ids": [], "rewritten_queries": ["<self-contained question>"]}
- (open: a paper) "compare it with <another file>" -> {"intent": "followup", "scope_mode": "EXPLICIT_DOCS", "target_doc_ids": [<open id>, <new id>], "rewritten_queries": ["<about the first>", "<about the second>"]}
- (open: a paper) "summarize the whole folder" -> {"intent": "question", "scope_mode": "GLOBAL_FOLDER", "target_doc_ids": [], "rewritten_queries": []}

JSON:"""


def build_turn_router_prompt(
    query: str,
    history_text: str,
    cards: list[dict],
    current_cards: list[dict] | None = None,
) -> str:
    def render(items: list[dict], with_summary: bool) -> str:
        lines = []
        for card in items:
            parts = [
                f"[{card['id']}] {card.get('file_name') or ''}".strip(),
                card.get("title") or "",
            ]
            if with_summary:
                parts.append(card.get("summary") or "")
            lines.append(" — ".join(part for part in parts if part))
        return "\n".join(lines)

    return _fill(
        TURN_ROUTER_PROMPT,
        cards=render(cards, True) or "(no documents)",
        current_docs=render(current_cards or [], False) or "(none)",
        history=history_text or "(none)",
        query=query,
    )