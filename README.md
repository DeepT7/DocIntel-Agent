# DocIntel Agent

A documentation-intelligence RAG agent over the **LangChain / LangGraph Python docs**.

It crawls the official docs, turns every page into structured sections and context-rich
chunks, embeds them into a local Qdrant index, and answers questions through a
LangGraph state machine that combines dense + keyword retrieval with an optional
grounded web-search fallback.

```
LangChain docs  →  crawl  →  clean  →  chunk  →  embed  →  Qdrant
                                                               │
       question + chat history  →  condense  →  retrieve  →  judge ────┤
                                                          │      │
                                                (good)    │      │ (weak)
                                                          ▼      ▼
                                                       answer   external search
                                                          │      │
                                                          └──► answer / fallback
```

---

## Features

- **Sitemap-first crawler** — discovers allowed LangChain/LangGraph pages from
  `sitemap.xml`, then follows in-page links up to `MAX_PAGES`.
- **Structure-aware cleaning** — extracts headings, paragraphs, code blocks, tables,
  links and breadcrumbs into a versioned JSON schema, stripping nav/scripts/chat UI.
- **Context-prefixed chunking** — every chunk carries title + breadcrumbs + section
  heading so retrieved text is self-describing.
- **Document graph** — builds a `doc_graph.json` of `discovered_from`,
  `path_parent`, and `links_to` edges between pages.
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
- **Optional Langfuse tracing** — opt-in, no-op when keys are absent.
- **Provider-agnostic** — Gemini (`google-genai`) or OpenRouter for embeddings and
  answer generation, selectable via environment variables.

---

## Repository layout

```
.
├── app/
│   ├── core/
│   │   ├── config.py          # all constants: crawl scope, chunk sizes, models, paths
│   │   ├── models.py          # TypedDicts for chunks, embeddings, retrieval, RAG state
│   │   └── tracing.py         # optional Langfuse helpers (no-op when disabled)
│   ├── ingestion/
│   │   ├── discover.py        # URL normalization, classification, link extraction
│   │   ├── crawl.py           # sitemap discovery + BFS crawl → data/raw/
│   │   ├── clean.py           # HTML → structured document JSON → data/cleaned/
│   │   ├── chunk.py           # sections → chunks → data/chunks/
│   │   └── build_doc_graph.py # inter-document link graph → data/graph/
│   ├── retrieval/
│   │   ├── embed_chunks.py    # embed corpus (incremental) + upsert into Qdrant
│   │   └── retrieve.py        # vector_search + hybrid_search
│   ├── reasoning/
│   │   ├── answer.py          # prompt building, generation, follow-up condensing
│   │   └── rag_graph.py       # LangGraph pipeline + interactive Rich CLI
│   └── evaluation/
│       └── run_eval.py        # retrieval benchmark over the gold question set
├── data/
│   ├── raw/                   # crawled HTML + metadata JSON (keyed by md5 of URL)
│   ├── cleaned/               # structured documents
│   ├── chunks/                # per-document chunks + corpus.jsonl + manifest.json
│   ├── embeddings/            # corpus_embeddings.jsonl + manifest.json
│   ├── graph/                 # doc_graph.json
│   ├── eval/questions.json    # 49 hand-curated gold questions
│   └── samples/               # sample images for the scratch demos
├── output/
│   ├── qdrant_db/             # local Qdrant storage
│   ├── eval_report.json       # latest benchmark report
│   └── eval_cache.json        # cached retrieval results
├── scratch/                   # standalone learning/demo scripts (not part of the pipeline)
│   ├── core_chatbot.py        # raw Gemini text / multimodal / image demos
│   └── langgraph_simple.py    # minimal single-node LangGraph example
├── requirements.txt
└── .gitignore
```

> `scratch/` holds standalone learning/demo scripts. They are not part of the RAG pipeline.

---

## Requirements

- Python 3.13+
- A local Qdrant store (no server needed — the `qdrant-client` embedded path mode is used)
- API keys depending on the providers you enable (see below)

Install dependencies:

```bash
pip install requests beautifulsoup4 lxml python-dotenv \
            google-genai qdrant-client rank-bm25 langgraph rich
# optional
pip install langfuse
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

Tunable in `app/core/config.py`:

- Crawl scope: `CRAWL_ROOTS`, `SITEMAP_URLS`, `ALLOWED_DOMAINS`,
  `ALLOWED_PATH_PREFIXES`, `EXCLUDED_PATH_HINTS`, `MAX_PAGES`
- Chunking: `CHUNK_TARGET_CHARS` (1200), `CHUNK_MAX_CHARS` (1800)
- Retrieval: `DEFAULT_TOP_K` (5), `EMBEDDING_DIM` (3072), `QDRANT_PATH`,
  `QDRANT_COLLECTION` (`doc_chunks`)
- Models: `DEFAULT_MODEL` (`gemini-3.5-flash`, used for grounded web search)

---

## Usage

Run every stage from the repository root so the `app` package imports resolve.

### 1. Ingest

```bash
python -m app.ingestion.crawl            # sitemap + BFS crawl  → data/raw/
python -m app.ingestion.clean            # HTML → sections      → data/cleaned/
python -m app.ingestion.chunk            # sections → chunks    → data/chunks/
python -m app.ingestion.build_doc_graph  # link graph           → data/graph/
```

### 2. Embed and index

```bash
python -m app.retrieval.embed_chunks
```

This embeds `data/chunks/corpus.jsonl` incrementally (identical chunk text is
reused from the previous run) and then upserts the vectors into the local Qdrant
collection, recreating it. Batches are checkpointed to a `.partial` file, so a
crash never loses completed work. The Gemini path additionally paces requests to
respect the free-tier per-minute quota.

### 3. Ask questions

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

### 4. Benchmark retrieval

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

---

## Notes

- The crawler is scoped to LangChain docs (`docs.langchain.com` and
  `reference.langchain.com`, LangGraph Python paths). Change `CRAWL_ROOTS`,
  `ALLOWED_DOMAINS`, and `ALLOWED_PATH_PREFIXES` to target another site.
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
- [ ] **Answer-level evaluation** — faithfulness, groundedness, citation accuracy; LLM-as-judge
- [ ] **Agentic retrieval** — retrieval as a tool, grade → rewrite → re-retrieve loop
- [ ] **CI quality gate** — pytest + eval thresholds to block regressions
- [ ] **Production hardening** — Redis cache, rate limiting, guardrails / prompt-injection defence
- [ ] **Multi-source ingestion** — web crawl + local folders (code, markdown, PDF)
