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