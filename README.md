# DocIntel Agent

A local-document intelligence RAG agent for PDF, Office, HTML, Markdown and text files.

It converts every supported document to canonical Markdown, turns it into structured
sections and context-rich chunks, embeds them into a local Qdrant index, and answers
questions through a LangGraph state machine that combines dense + keyword retrieval
with an optional grounded web-search fallback.

```
data/docs (PDF/DOCX/PPTX/HTML/MD/TXT)  →  canonical Markdown  →  blocks  →  chunks  →  embed  →  Qdrant
                                                                                              │
       question + chat history  →  condense  →  retrieve  →  judge ────┤
                                                          │      │
                                                (good)    │      │ (weak)
                                                          ▼      ▼
                                                       answer   external search
                                                          │      │
                                                          └──► answer / fallback
```

> **[PIPELINE.md](PIPELINE.md)** records the pipeline stages and how each one
> processes data (transformations, decisions, inputs/outputs).
>
> **[INDEXING.md](INDEXING.md)** documents the incremental indexing subsystem:
> metadata-only scanning, the SQLite file-state cache, the debounced background
> worker, real-time folder watching and on-demand content reads.

---

## Features

- **Canonical-Markdown ingestion** — MarkItDown converts `.docx`, `.pptx`, `.html`
  and text, pymupdf4llm converts `.pdf`, and `.md` is already canonical; one shared
  Markdown parser then preserves headings, lists, code fences and tables.
- **Smart asset extraction** — PDF picture regions are rendered with PyMuPDF and
  Office media is read from the ZIP container; tiny icons, logos, rules, blank
  blocks and repeated template images are filtered out before storage.
- **Structure-aware chunking** — every chunk carries title + breadcrumbs + section
  heading so retrieved text is self-describing; tables stay atomic with captions.
- **Incremental indexing** — metadata-only scan, content-hash confirmation, exact-text
  embedding cache, debounced watcher; only changed files are re-chunked.
- **Document cards & outlines** — one cached card per file (title, summary, keywords,
  canonical section tree) so chat can resolve document/section references.
- **Hybrid retrieval** — dense vector search (Qdrant) fused with a BM25 keyword
  signal (0.7 semantic / 0.3 keyword).
- **Grounded fallback** — when internal retrieval is weak, the graph queries Google
  Search grounding and answers from external context, or declines cleanly.
- **Multi-turn conversation** — the CLI keeps chat history and a `condense` node
  rewrites each follow-up into a standalone question before retrieval, so pronouns
  and elided references resolve against earlier turns. Single-turn use skips the
  rewrite entirely (no extra latency or API cost).
- **Evaluation harness** — `hit@k`, `recall@k`, `precision@k`, `MRR@k` over a
  49-question gold set, with caching and per-category / per-difficulty breakdowns.
- **Answer-level evaluation** — an LLM-as-judge scores faithfulness, answer relevance,
  and citation accuracy of the generated answers, using a judge model distinct from the
  generation chain to reduce self-preference bias.
- **Optional Langfuse tracing** — opt-in, no-op when keys are absent.
- **Provider-agnostic** — Gemini (`google-genai`) or OpenRouter for embeddings and
  answer generation, selectable via environment variables.

---

## Repository layout

```
.
├── app/
│   ├── core/
│   │   ├── config/            # typed settings (pydantic-settings), one file per group
│   │   ├── models.py          # TypedDicts for chunks, embeddings, retrieval, RAG state
│   │   ├── prompts.py         # prompt templates
│   │   ├── sections.py        # section-tree helpers
│   │   └── tracing.py         # optional Langfuse helpers (no-op when disabled)
│   ├── ingestion/
│   │   ├── extract.py         # any supported file -> canonical Markdown -> blocks
│   │   ├── assets.py          # smart visual-asset extraction + filtering
│   │   ├── chunking.py        # blocks -> section-aware chunks
│   │   ├── identity.py        # stable file/chunk ids + content hashes
│   │   ├── qdrant_index.py    # collection schema, embedding cache, upsert/delete
│   │   ├── sync.py            # metadata-first incremental folder sync
│   │   └── folder.py          # CLI / public ingestion entry point
│   ├── indexing/
│   │   ├── scanner.py         # metadata-only folder scan
│   │   ├── metadata.py        # SQLite file-state cache
│   │   ├── watcher.py         # real-time folder watcher (polling fallback)
│   │   ├── worker.py          # debounced background index queue
│   │   ├── pipeline.py        # watcher batch -> index writer
│   │   ├── manager.py         # per-scope watcher/indexer lifecycle
│   │   └── content.py         # on-demand raw file reads
│   ├── retrieval/
│   │   ├── embeddings.py      # provider-agnostic embeddings + cache
│   │   ├── store.py           # embedded/local Qdrant client
│   │   └── retrieve.py        # vector_search + hybrid_search
│   ├── reasoning/
│   │   ├── answer.py          # prompt building, generation, follow-up condensing
│   │   └── rag_graph.py       # LangGraph pipeline + interactive Rich CLI
│   ├── capabilities/          # HTTP capabilities: ask/search/outline/sync/registry
│   ├── evaluation/
│   │   ├── run_eval.py        # retrieval benchmark over the gold question set
│   │   └── answer_eval.py     # answer-level evaluation with an LLM-as-judge
│   └── server/                # FastAPI API + static browser UI
├── desktop/                   # optional two-window desktop shell
├── data/
│   ├── docs/                  # source documents (PDF/DOCX/PPTX/HTML/MD/TXT)
│   ├── eval/questions.json    # hand-curated gold questions
│   └── index/                 # metadata.sqlite3, embedding cache, registry/insights, assets/
├── output/                    # local Qdrant storage + evaluation reports
├── tests/
├── requirements.txt
└── .gitignore
```

---

## Requirements

- Python 3.13+
- A local Qdrant store (no server needed — the `qdrant-client` embedded path mode is used)
- API keys depending on the providers you enable (see below)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` in the project root. Only the keys for the providers you use are required.

| Variable | Purpose | Default |
| --- | --- | --- |
| `GEMINI_API_KEY` / `API_KEY` | Gemini access: Google Search grounding, Gemini embeddings/generation | — |
| `OPENROUTER_API_KEY` | OpenRouter embeddings + generation | — |
| `EMBEDDING_PROVIDER` | `gemini` or `openrouter` | `openrouter` |
| `OPENROUTER_EMBEDDING_MODEL` | Embedding model on OpenRouter | `google/gemini-embedding-2` |
| `GENERATION_PROVIDER` | `gemini` or `openrouter` | `openrouter` |
| `OPENROUTER_GENERATION_MODELS` | Comma-separated fallback chain, tried in order | Nemotron / Ling free models |
| `LANGFUSE_PUBLIC_KEY` | Enables tracing when set (with secret key) | — |
| `LANGFUSE_SECRET_KEY` | Enables tracing when set (with public key) | — |
| `LANGFUSE_BASE_URL` | Langfuse host (self-hosted / cloud region) | SDK default |
| `LANGFUSE_TRACING_ENABLED` | Set to `false` to force tracing off | `true` |

> **Important:** corpus embeddings and query embeddings must use the *same* provider
> and model. The embedding cache is scoped by model string, so switching providers
> triggers a full re-embed rather than mixing vector spaces.

### Pipeline constants

Configuration is a typed, nested settings tree in **`app/core/config/`**
(`pydantic-settings`), split one file per subsystem. New code reads it directly:

    from app.core.config import settings
    settings.retrieval.default_top_k

Every value can be overridden without touching code via the **`DOCINTEL_`**
environment or `.env`, nesting groups with a double underscore:

    DOCINTEL_RETRIEVAL__DEFAULT_TOP_K=10 uvicorn app.server.main:app --port 8756
    DOCINTEL_EMBEDDINGS__RETRYABLE_STATUS_CODES=429,500,503

Key groups (see each file for the full list and defaults):

- `paths.py`: `data_dir`, `docs_dir`, `index_dir`, `state_path`
- `ingestion.py`: `chunk_target_chars` (1200), `chunk_max_chars` (1800),
  `index_version`, `upsert_batch_size`, `breadcrumb_max_depth`, plus asset
  filters (`asset_min_width`, `asset_max_aspect_ratio`, `asset_min_ink_ratio`,
  `asset_pdf_dpi`, `asset_office_min_display`, `asset_repeat_count`, ...)
- `embeddings.py`: `provider`, `model_name`, `openrouter_model`, `dim`, plus
  retry (`max_retries`, `base_delay`, `request_timeout`, ...)
- `store.py`: `qdrant_url`, `qdrant_path`, `collection` (`docs_local`),
  `hybrid_prefetch_limit`, `scroll_page_size`, ...
- `retrieval.py`: `default_top_k` (5), `ask_top_k` (8), `reference_penalty`, ...
- `generation.py`: `default_model` (`gemini-3.5-flash`, grounded web search),
  `generation_provider`, model chains, retry/cooldown, token budgets
  (`MAX_TOKENS`), history windows
- `graph.py`: reasoning-graph caps and gates (`max_section_docs`,
  `max_section_chunks`, `grade_skip_threshold`, `max_rewrites`, ...)
- `capabilities.py`, `state.py`, `desktop.py`: card/insight/suggest/workspace
  budgets, session limits, and window sizes

The old flat imports (`from app.core.config import DEFAULT_TOP_K`) still work
through a compatibility facade in `config/__init__.py`; prefer `settings.*`.

Naming convention: `DOCINTEL_<GROUP>__<FIELD>`.

---

## Usage

Run every stage from the repository root so the `app` package imports resolve.

> **One process at a time (embedded Qdrant).** The vector store is embedded
> (`output/qdrant_db`) and keeps a file lock, so only **one** process may use it.
> Do not run the API server and a CLI/eval at the same time, and never start the
> server with `--reload`. Check who holds the lock with
> `python -m app.retrieval.store`, release it with
> `python -m app.retrieval.store --stop`, then retry. For genuine concurrent
> access, run a Qdrant server and set `QDRANT_URL` (embedded mode is bypassed).
>
> See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for the full write-up of the
> "Storage folder is already accessed" error.

### 1. Ingest and index

```bash
python -m app.ingestion.folder --root data/docs
```

Every supported file is converted to canonical Markdown first (MarkItDown for
`.docx` / `.pptx` / `.html` / `.txt`, pymupdf4llm for `.pdf`, read as-is for
`.md`), then parsed into ordered blocks and chunked. The sync is incremental:
metadata diffs and content hashes limit re-chunking/re-embedding to files that
really changed. Embeddings are cached by exact chunk text.

Use `--reset` to rebuild the collection from scratch and `--prune-cache` to drop
embedding-cache entries no longer referenced by the index.

Visual assets are extracted automatically for each changed PDF/DOCX/PPTX and
stored under `data/index/assets/<file_id>/`. To run only that stage:

```bash
python -m app.ingestion.assets --root data/docs
```

### 2. Ask questions

```bash
python -m app.reasoning.rag_graph        # interactive Rich CLI
```

The CLI renders the answer as Markdown and prints a source table with title,
section, and URL. It keeps **conversation history**, so follow-up questions work:
each follow-up is condensed into a standalone query before retrieval, and when
that rewrite changes the query the CLI prints it as `↳ Truy vấn độc lập:`.

| Command | Effect |
| --- | --- |
| `exit` / `quit` / `:q` | Leave the CLI |
| `:reset` / `reset` / `clear` | Clear conversation history and start a new topic |
| `:web` | Toggle web search (Google Search grounding) on/off |
| `:think` | Toggle thinking mode (model reasoning) on/off |

### Answer options (web + CLI + API)

Two per-question options are exposed on every surface; both default to **off**:

| Option | API field | Effect |
| --- | --- | --- |
| **Web search** | `allow_external` | When local retrieval is weak, allow a Gemini Google-Search-grounded lookup instead of declining. Off keeps answers strictly local. |
| **Think** | `thinking` | Let the answer model reason before answering. Slower and more expensive, but better on multi-step/comparison questions. Router/grader helpers stay non-thinking. |

In the web UI, the **Tools ▾** button (left of the send arrow) opens a menu with
**Web search** and **Think**. The button label shows the active modes
(`Web + Think`), and the choice is remembered in `localStorage` and sent with
each request:

```bash
curl -X POST http://127.0.0.1:8756/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"compare the two papers","allow_external":true,"thinking":true}'
```

Thinking mode uses the larger `answer_thinking` / `no_context_thinking` token
budgets (see `app/core/config/generation.py`), because hidden reasoning tokens
count against the completion.

### 3. Benchmark retrieval

```bash
python -m app.evaluation.run_eval --retriever all --k 1 3 5 10
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--retriever {vector,hybrid,all}` | Which retriever(s) to score |
| `--k 1 3 5 10` | Cutoffs to evaluate |
| `--limit N` | Run only the first N questions (writes a `_limitN` report) |
| `--no-cache` | Ignore `output/eval_cache.json` and re-retrieve |
| `--delay SECONDS` | Sleep between uncached questions to avoid rate limits |
| `--no-resolve-gold` | Use the hard-coded `chunk_ids` instead of resolving by section |

Gold labels are defined by *(url, section_heading)* and re-pointed at the current
corpus on every run, so chunker changes do not silently invalidate the benchmark.

### 4. Evaluate answer quality

```bash
python -m app.evaluation.answer_eval              # generate answers + judge (cached)
python -m app.evaluation.answer_eval --judge-only # score only, reuse cached answers
python -m app.evaluation.answer_eval --no-cache   # regenerate answers and judgements
```

Answers and judgements are cached under `output/answer_eval/`, so re-runs only fill in
missing entries. The aggregated report is written to `output/answer_eval/report.json`.

---

## The RAG graph

`app/reasoning/rag_graph.py` compiles this LangGraph state machine:

| Node | Role |
| --- | --- |
| `condense` | Rewrites a follow-up into a standalone query using `chat_history` (no-op without history) |
| `retrieve` | `hybrid_search` over Qdrant, top-k |
| `judge` | Accepts retrieval when the top fused score ≥ `0.1` |
| `answer` | Generates an answer from internal context only |
| `external_search` | Gemini + Google Search grounding for weak retrieval |
| `answer_with_fallback` | Generates from internal + external context combined |
| `fallback` | Declines gracefully when nothing is usable |

Conditional routing: `condense → retrieve`, then `judge → answer | external_search`,
then `external_search → answer_with_fallback | fallback`.

Every node is wrapped in an optional Langfuse span; when tracing is off the helpers
are no-ops.

---

## Retrieval

- `vector_search(query, top_k)` — cosine similarity via Qdrant `query_points`.
- `hybrid_search(query, top_k)` — dense candidates (2× top-k, min 10) fused with
  normalized BM25 scores using `0.7 * semantic + 0.3 * keyword`. The BM25 index is
  built lazily in memory from the stored payloads.

Latest benchmark (`output/eval_report.json`, 49 questions):

| Retriever | hit@1 | recall@1 | hit@5 | recall@5 | hit@10 | MRR@10 | latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vector | 0.816 | 0.611 | 0.980 | 0.947 | 1.000 | 0.881 | ~1105 ms |
| hybrid | 0.755 | 0.587 | 0.939 | 0.865 | 1.000 | 0.848 | ~1094 ms |

On this corpus, pure dense retrieval outperforms the current hybrid fusion;
the hybrid path is kept as an evaluated alternative.

### Answer-level evaluation (LLM-as-judge)

Retrieval metrics only tell us whether the right chunk was surfaced. To measure the
**final answer**, `app/evaluation/answer_eval.py` generates one answer per gold question
and scores it with an LLM judge on three axes:

| Criterion | Mean (n=49) |
| --- | --- |
| faithfulness | 4.94 |
| answer_relevance | 5.00 |
| citation_accuracy | 4.82 |

- The judge model (`qwen/qwen3-30b-a3b-instruct-2507`) is deliberately different from
  the generation chain to avoid **self-preference bias**.
- The judge is asked for a **flat JSON object**; the parser also tolerates nested or
  loose variants, and a malformed response skips that question instead of crashing.
- Answers and judgements are cached under `output/answer_eval/`, so re-runs only fill
  missing entries; the report is written to `output/answer_eval/report.json`.

---

## Notes

- Ingestion is scoped to one local folder. Standalone spreadsheets `.xlsx` /
  `.csv` / `.tsv`, images and source-code files are skipped by the scanner.
- `external_search_node` always uses Gemini with Google Search grounding, since
  OpenRouter does not expose that tool — hence `DEFAULT_MODEL` stays Gemini-only.
- Files in `data/` and `output/` are derived artifacts and safe to regenerate by
  re-running the pipeline stages in order.

---

## Design decisions & what we rejected

- **Custom pipeline over a framework.** Explicit retrieval/orchestration code was
  preferred over LlamaIndex/LangChain RAG abstractions, to keep full control and
  a clear understanding of every stage.
- **Cross-encoder reranking: tried and rejected.** We added a fastembed cross-encoder
  reranker (`ms-marco-MiniLM-L-6-v2` and `bge-reranker-base`) over a 20-candidate set.
  Both **lowered** MRR and hit@1/@5 versus plain dense retrieval, at every granularity
  (exact chunk, section, and document). Root cause: the bi-encoder already ranks
  strongly because chunks carry title/breadcrumb/heading prefixes, and the
  cross-encoder reordered near-miss sibling chunks above the gold section. The
  reranker was removed and the negative result kept as evidence. *Measure, don't assume.*
- **Hybrid fusion is kept but not the default.** BM25 scores are normalised while
  cosine similarity is not, so the `0.7 * semantic + 0.3 * keyword` blend
  underperforms dense-only on this corpus. It remains as an evaluated alternative.
- **Provider-agnostic embeddings/generation.** Corpus and query embeddings share the
  same provider + model; the embedding cache is scoped by model string, so switching
  providers re-embeds instead of mixing vector spaces. Generation uses a fallback
  chain across providers/models for reliability.

---

## Production considerations

This repository runs as a single local process. Moving it to a company-style
deployment changes scale and managed services, not the core patterns:

| Area | Here | At scale |
| --- | --- | --- |
| Serving | interactive CLI | FastAPI + async + streaming (SSE) |
| Vector DB | embedded Qdrant (local file) | managed (Pinecone / Weaviate / pgvector) |
| Orchestration | one process | Docker + Kubernetes / ECS |
| Config & secrets | `.env` | Vault / cloud secrets manager |
| Caching | eval cache only | Redis for retrieval + embeddings |
| Observability | Langfuse | OpenTelemetry + Prometheus / Grafana |
| CI/CD | manual | pipeline with an eval quality gate |
| Scale | single instance | autoscaling + load balancer |

The patterns used here (containerisation-friendly config, 12-factor env vars, traced
nodes, incremental indexing) transfer directly; the differences are scale, managed
services, and governance.

---

## Roadmap

- [ ] **Serving layer** — FastAPI (`/ingest`, streaming `/query`), Pydantic schemas, Docker + compose
- [x] **Answer-level evaluation** — faithfulness, groundedness, citation accuracy; LLM-as-judge
- [ ] **Agentic retrieval** — retrieval as a tool, grade → rewrite → re-retrieve loop
- [ ] **CI quality gate** — pytest + eval thresholds to block regressions
- [ ] **Production hardening** — Redis cache, rate limiting, guardrails / prompt-injection defence
- [ ] **Web-source ingestion** — crawl/import web pages on top of the local-folder pipeline
