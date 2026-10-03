# DocIntel Agent — Project Plan

Goal: a **CV-grade** LLM application. Target lane: **Applied LLM Engineer** with a thin
platform/serving layer. Depth over novelty: real metrics, runnable end-to-end, good docs.

Timeline: 2 weeks, "basic but complete" (one vertical slice).

---

## Week 1 — Quality & AI

| Day | Task | Status |
| --- | --- | --- |
| 1–2 | Clean code, `requirements.txt`, `.gitignore`, README + diagram, git init | ✅ Done |
| 3–5 | Answer-level evaluation (faithfulness, relevance, citation) + LLM-as-judge | ⏳ Almost |
| 6–7 | Agent + tools: retrieval as a tool, grade → rewrite → re-retrieve (Corrective RAG), memory | ⬜ |

### Day 3–5 — remaining checklist
- [ ] `answer_eval.py`: `JUDGE_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"`
- [ ] `answer_eval.py`: remove `"reasoning": {"enabled": False}` from `call_judge`
- [ ] `answer_eval.py`: fix typo `answer = generate_answers(...)` → `answers = ...`
- [ ] Run `python -m app.evaluation.answer_eval --no-cache` → expect `n=49`, no failures
- [ ] Refresh README numbers if they changed
- [ ] Commit `answer_eval.py` + `README.md`

---

## Week 2 — Production & packaging

| Day | Task | Status |
| --- | --- | --- |
| 8–10 | FastAPI: `/ingest`, `/query` streaming (SSE), Pydantic schemas, async | ⬜ |
| 11 | Docker + compose (app + Qdrant), env config, health check | ⬜ |
| 12 | pytest + eval as a CI quality gate | ⬜ |
| 13 | Deploy (Fly.io / Render) + public demo link | ⬜ |
| 14 | README polish (metrics, trade-offs), demo video | ⬜ |

---

## Cut for now (future work)

Redis cache, rate limiting, guardrails / PII, prompt-injection defence, MCP,
fine-tuning, multi-source web crawl, GraphRAG, a polished frontend.

---

## CV signals by area

- **Evals** — retrieval metrics + answer-level LLM-as-judge; eval as a ship gate.
- **Agents** — tool calling, corrective/agentic retrieval, memory.
- **Serving** — FastAPI, async, streaming, Pydantic structured outputs.
- **Platform** — Docker, env config, health checks, observability (Langfuse).
- **Presentation** — README with architecture, metrics table, and honest trade-offs.

---

## Deployment note (laptop vs company infra)

The patterns transfer; the differences are scale and managed services. In the README's
"Production considerations" section, document what would change at a company: managed
vector DB, Kubernetes/ECS, secrets manager, Redis, OpenTelemetry + Prometheus, CI/CD,
autoscaling. Being able to *articulate* this gap is itself a strong interview signal.

---

## Working agreement

- The user types the code; the assistant instructs, reviews, and diagnoses.
- Commit at the end of each day/phase with a clear message.