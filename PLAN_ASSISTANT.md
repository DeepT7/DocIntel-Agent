# PLAN — DocIntel: trợ lý tài liệu local-first (một lõi, nhiều bề mặt)

> Trạng thái: Phase 0 → 2 xong (test headless). Hướng mới: **API-first**, bỏ Qt/GTK/Electron cho cửa sổ chính.

## 1. Mục tiêu

Hệ thống local-first trên **thư mục tài liệu của người dùng**:

- **Main (đầy đủ)**: hỏi–đáp, tìm, tóm tắt/insight, hệ thống hóa; về sau flashcard/slide.
- **Quick (ambient)**: popup qua hotkey — hỏi nhanh, tìm, **gợi ý viết** từ đoạn đang viết (clipboard) + kho tài liệu.

Mọi bề mặt **chia sẻ một lõi**: cùng scope (folder), index Qdrant, lịch sử. User trỏ folder; file đổi được **sync incremental** khi cần.

## 2. Nguyên tắc

1. **API-first** — mọi capability qua HTTP; UI chỉ là client → Main là **web trong browser**, không Qt.
2. **Một process sở hữu Qdrant** — tránh "storage already accessed".
3. **Capabilities tách khỏi UI** — logic ở `app/capabilities/`, không UI nào tự đụng Qdrant.
4. **Ambient = clipboard**, không auto-capture ở MVP (không cần Accessibility).
5. **Sync theo yêu cầu**, không watcher liên tục. Save + sync trước khi hỏi là đủ.

## 3. Chắt lọc từ Open Notebook (lfnovo/open-notebook)

| Pattern của họ | Áp dụng tại đây | Trạng thái |
|---|---|---|
| **API-first**: UI là client | FastAPI `app/server/main.py`; `/` trả trang web Main | ✅ |
| **Insight**: summary per source, tái dùng | `capabilities/insights.py` (map-reduce + cache theo hash) | ✅ |
| **Async-first**: job nền + status | `app/jobs.py`; `POST` trả `job_id`, `GET` poll | ✅ |
| **Transformation-as-data** (task = prompt config) | dự kiến `capabilities/tasks.py` | ⏳ |
| **Provider abstraction** (Esperanto) | dự kiến `app/core/llm.py` | ⏳ |
| **Insight embed để search được** | dự kiến `doc_type=insight` vào Qdrant | ⏳ |
| **MCP** (agent dùng hệ thống) | dự kiến, xa | ⏳ |

**Không copy**: SurrealDB/Next.js/Docker (giữ Python/Qdrant); "add source thủ công" (giữ folder sync incremental); web-only (giữ đường ambient).

## 4. Kiến trúc

```
Một process Python
 ├─ app/ (lõi)
 │   ├─ ingestion/   extract · folder(sync, chunk, root payload)
 │   ├─ retrieval/   retrieve(hybrid + scope) · embeddings · store
 │   ├─ reasoning/   answer(_generate) · rag_graph
 │   ├─ capabilities/  search · sources · sync · suggest · ask · insights
 │   ├─ jobs.py      job nền + status
 │   ├─ app_state.py scope, recent, conversations[conv_id]
 │   └─ core/        config · prompts · scope · models
 ├─ app/server/main.py   FastAPI (API-first) ──► UI web (browser) = MAIN
 └─ desktop/             Quick popup native (pywebview) — TUỲ CHỌN (ambient)
```

- **Main** = trang web từ FastAPI (`http://localhost:8756`). Không Qt.
- **Quick** = popup native riêng (pywebview + hotkey + clipboard), gọi cùng API/bridge.
- Có thể thay Main bằng **frontend TypeScript (Vite)** gọi cùng HTTP API.

## 5. Mô hình khái niệm

| Khái niệm | Ở đây | Ghi chú |
|---|---|---|
| **Scope** | một folder root (`root` payload) | chuẩn hoá `normalize_root` |
| **Source** | một file trong scope | `path`, `file_id`, `title` |
| **Chunk** | đơn vị retrieval | `dense` + `bm25`, có `table_number`/`caption` |
| **Insight** | tóm tắt cache của 1 file | map-reduce, invalidate theo hash |
| **Answer** | câu trả lời hỏi–đáp trên folder | qua `rag_graph` có `scope`, đa lượt |
| **Suggestion** | gợi ý viết (query + clipboard) | có trích dẫn nguồn |
| **Job** | tác vụ nền | `queued\|running\|done\|error`, có `result` |

## 6. Cấu trúc file (hiện có)

```
app/core/scope.py            normalize_root
app/core/sections.py         canonical section tree: section_id/ancestors + render
app/core/prompts.py          answer/condense/grade/rewrite/judge + suggest + insight
app/ingestion/folder.py      sync_folder, chunk, materialized section paths (INDEX_VERSION=12)
app/retrieval/retrieve.py    vector_search(scope, paths) · section_chunks(section_id/ancestors)
app/retrieval/store.py       Qdrant client dùng chung (1 owner)
app/capabilities/search.py   search(scope, query, paths)
app/capabilities/sources.py  list_sources(scope)
app/capabilities/sync.py     sync(scope)
app/capabilities/suggest.py  suggest(scope, query, clipboard)
app/capabilities/ask.py      ask(scope, query, history, allow_external, doc_paths)
app/capabilities/workspace.py list_files(scope)
app/capabilities/insights.py summarize_source / get_insight / list_insights
app/reasoning/rag_graph.py   run_rag(query, history, scope, doc_paths, allow_external)
app/jobs.py                  submit / get / list_jobs
app/app_state.py             scope, recent, conversations[conv_id]{messages, active_docs, primary_docs, last_used_docs}
app/server/main.py           FastAPI + trang Main HTML
desktop/api.py, app.py       bridge + 2 cửa sổ (tuỳ chọn)
desktop/ui/*.html            Quick/Main native (tuỳ chọn)
```

## 7. Hợp đồng HTTP API

| Method | Endpoint | Body / Query | Trả về |
|---|---|---|---|
| GET | `/` | — | trang Main (HTML) |
| GET | `/api/health` | — | `{ok}` |
| GET | `/api/state` | `?conversation_id=` | `{scope, recent, excluded_docs, conversation_id, session, active_docs, primary_docs, last_used_docs}` |
| POST | `/api/scope` | `{path, conversation_id?}` | state mới (reset chat đang mở + working set mọi chat) |
| POST | `/api/sync` | `{scope?}` | `{files, changed, chunks_reindexed, embeddings_computed, outlines}` |
| GET | `/api/sources` | `?scope=` | `[{path,title,file_id}]` |
| GET | `/api/selection` | — | `{excluded_docs}` — file người dùng BỎ TÍCH |
| POST | `/api/selection` | `{paths}` | đặt loại trừ; trả `{excluded_docs}` |
| GET | `/api/outline` | `?scope=&path=` | `{title, sections:[…], tree:[{id,title,level,children}]}` |
| GET | `/api/section` | `?scope=&path=&section_id=` | chunk của các mục; `section_id` cách nhau dấu phẩy; cha → cả cây con |
| GET | `/api/registry` | `?scope=` | danh sách "thẻ căn cước" (title, summary, keywords, sections) |
| GET | `/api/card` | `?scope=&path=` | thẻ của một file |
| POST | `/api/registry` | `{scope?, force?}` | dựng thẻ còn thiếu (job nền) → `{job_id}` |
| POST | `/api/search` | `{query, scope?, path?, paths?, top_k}` | hits (`score`, `semantic_score`, `path`, `section_heading`) |
| POST | `/api/suggest` | `{query, clipboard, scope?}` | `{suggestion, sources}` |
| POST | `/api/ask` | `{query, scope?, conversation_id?, path?, paths?, top_k?, history?, use_session?, use_selection?, notes?, allow_external?, thinking?, save?}` | `{answer, intent, scope_mode, conversation_id, standalone_query, sources, scope, doc_paths, routed_queries}` |
| GET | `/api/session` | `?conversation_id=` | lịch sử hội thoại `[{role,content}]` |
| POST | `/api/session/clear` | `{conversation_id?}` | `{session: []}` |
| POST | `/api/conversation/new` | — | `{conversation_id}` (chat mới, rỗng) |
| GET | `/api/conversations` | — | `[{conversation_id, messages, updated_at}]` |
| GET | `/api/working-set` | `?conversation_id=` | `{conversation_id, active_docs, primary_docs}` |
| POST | `/api/working-set` | `{paths, conversation_id?}` | `{conversation_id, active_docs, primary_docs}` |
| POST | `/api/insight` | `{path, scope?, force?, background?}` | `{job_id}` hoặc insight trực tiếp |
| GET | `/api/insight` | `?scope=&path=` | insight đã cache (`stale`) |
| GET | `/api/insights` | `?scope=` | danh sách insight có/không fresh |
| GET | `/api/job/{id}` | — | trạng thái + result |
| GET | `/api/jobs` | — | tất cả job |

## 8. Luồng chính

**A. Gợi ý viết (Quick / Main)**
```
Ctrl+C → POST /api/suggest {query, clipboard} → sync scope (await)
 → vector_search(query+clipboard, scope) → prompt suggest + context → gợi ý [Source N]
```

**B. Tìm thông tin (Main)**
```
POST /api/search → hybrid RRF trong scope → hits → hiển thị + nguồn
```

**B2. Hỏi đáp trên folder (RAG đa lượt)**
```
POST /api/ask {query} (history từ session)
 → run_rag(scope): condense(có history) → retrieve(scope) → grade
   → yếu: rewrite 1 lần → grade lại
   → allow_external=false (mặc định) → fallback trung thực
   → answer + sources
 → append_turn vào session
```

**C. Tóm tắt file (Insight, nền)**
```
POST /api/insight {path} → job_id
 worker: đọc chunk file (root+path) → 1 pass ngắn / map-reduce dài
        → cache theo file_hash → GET /api/job/{id} → done
```

## 9. Lộ trình

- [x] **Phase 0 — lõi**: `root`/`scope`, `vector_search(scope)`, capabilities (`search`,`sources`,`sync`,`suggest`), `app_state`, prompt `suggest`.
- [x] **Phase 0.5 — API-first + Insight + jobs**: FastAPI, `jobs.py`, `capabilities/insights.py`. Test: `/`, sync incremental, sources, search, suggest, insight job + cache hit lần 2 (0 LLM call), list insights.
- [x] **Phase 1 — Quick popup native**: `desktop/*` (bridge + hotkey + clipboard), chưa chạy trên WSL (thiếu display).
- [x] **Phase 2 — Hỏi đáp trên folder**: `capabilities/ask.py`; `run_rag` nhận `scope` + `allow_external`; session đa lượt. Test: hỏi tiếng Việt → có nguồn `memory.md`, follow-up condense đúng, session + clear OK.
- [ ] **Phase 2b — Main hoàn thiện**: citation click mở file; streaming; hoặc frontend TS (Vite).
- [ ] **Phase 3 — Tasks (as data + job)**: `synthesize` → `flashcards` → `slides`; mỗi task là job nền.
- [ ] **Phase 4 — Insight search được + polish**: embed insight (`doc_type=insight`) để `search` trả cả insight; tray; đóng gói.
- [ ] **Phase 5 — Provider abstraction + MCP** (xa).

## 10. Chạy

```bash
# Main (browser, không Qt)
pip install fastapi uvicorn
uvicorn app.server.main:app --port 8756     # KHÔNG --reload với Qdrant embedded
# mở http://localhost:8756  (Q&A, search, suggest, insight)

# Quick popup native (tuỳ chọn, cần màn hình)
pip install pywebview pynput pyperclip
python -m desktop.app          # Ctrl+Alt+Space mở Quick
```

Chỉ mở **một** tiến trình dùng Qdrant tại một thời điểm.

## 11. Ghi chú kỹ thuật

- `root` chuẩn hoá `normalize_root` (absolute, forward slash); query lọc `root` **chính xác** (scope cha không lẫn con).
- `file_id = md5(rel_path)` — đa root thì `rel_path` trùng → trùng `file_id`; phải key theo `(root, rel_path)` khi mở rộng.
- `INDEX_VERSION=9` buộc re-upsert payload có `root` (dense tái dùng cache, không re-embed).
- Qdrant embedded chỉ một process: mọi truy cập qua `app/retrieval/store.get_client()`.
- Insight cache ở `data/index/insights/<root_key>_<file_key>.json`, invalidate theo `file_hash`; ngắn 1 pass, dài map-reduce (`SINGLE_PASS_CHARS`, `MAP_GROUP_CHARS`).
- Jobs = thread + dict trong process (đủ MVP); thay queue thật khi cần.

**Intent routing (chat / followup / question / workspace)**: mỗi lượt qua `classify_node` — utility model phân loại (đa ngữ, không liệt kê từ khoá). `chat` → `chat_node`; `workspace` → `workspace_node` (liệt kê file, **không** retrieval); `followup` → condense → retrieval trong hội thoại; `question` → retrieval **toàn folder** (bỏ condense). Fail-safe → `question`. Pattern LlamaIndex RouterQueryEngine / semantic router; zero-cost: thay embedding router. `workspace` dùng `capabilities/workspace.py:list_files`.

**Cô lập hội thoại (IMP-003)**: state hai cấp — app-level (`scope`, `recent`, `excluded_docs`) và conversation-level (`conversations[conv_id]` với `messages`, `active_docs`, `primary_docs`, `last_used_docs`). `new_conversation()` tạo id mới rỗng; mỗi `/api/ask` mang `conversation_id` → `last_used_docs` **per-conversation**, không rò rỉ. `set_scope` reset chat đang mở + working set mọi chat (path thuộc folder cũ). **Graph không đọc `app_state`**: `main.ask` nạp theo `conv_id` rồi truyền vào `run_rag` (DI qua `RAGState`); node chỉ đọc `state`.

**Cấp độ 1 — chọn nguồn tường minh (UI)**: panel **Sources** (checkbox mỗi file). Bỏ tích → `/api/selection` (`excluded_docs`); web gửi `use_selection: true` → `/api/ask` tính `selected = tất cả − excluded` **trên server**, làm phạm vi, **ưu tiên hơn router**. Cờ phân biệt "không lọc" với "bỏ hết": bỏ hết → **No-Context / Notes Only**, **không** fallback toàn folder. Mặc định "all" → không lọc. Lưu trong state; reset khi đổi folder.

**No-Context / Notes Only**: khi mọi nguồn Off, `/api/ask` **không gọi router**, không gửi Document Cards; gọi `answer_without_context()` với system prompt `NO_CONTEXT_SYSTEM`. Model chỉ dùng: lịch sử, notes đang mở, **tên file** (metadata), kiến thức chung. Câu hỏi cần nội dung tài liệu → xin user bật ≥1 nguồn (không bịa); kiến thức chung → nhắc "chưa bật tài liệu"; chat vẫn trả lời tự nhiên.

**Cấp độ 2a — Contextual Chunk Headers**: mỗi chunk gắn header `[Document: <file> | Section: <breadcrumb>]` **trước khi embed**; breadcrumb **tối đa 3 bậc** từ cây canonical (không `**`/hậu tố `(Table)`). Payload thêm: `section_id` (`s4`, `s4.1`), `section_ancestors`, `section_titles`, `section_depths`, `section_number`, `section_title_clean`, `raw_heading`, `chunk_type`. `INDEX_VERSION` → `12`.

**Cấp độ 2b — Document Registry (thẻ căn cước)**: `capabilities/registry.py` dựng thẻ/file (`title`, `summary`, `keywords/entities`, `sections`), cache theo content-hash, lưu `data/index/registry/<root>.json`. Thẻ lưu **`outline_tree`** (`id`, `title`, `level`, `children`) và `sections` là text thụt cấp của cùng cây. `POST /api/sync` cập nhật `outline_tree` từ chunk vừa index (hoặc scroll Qdrant **một lần** nếu thiếu); chat đọc cây từ **RAM cache**, không scroll mỗi lượt. Endpoint `/api/registry`, `/api/card`, `POST /api/registry` (job). Mục đích: tìm file theo **nội dung** dù chunk không nêu tên file.

**Cấp độ 3 — LLM Turn Router (thay `classify_intent` + `_detect_docs`)**: MỘT call `capabilities/router.py` đưa query + history + **Document Cards** (`id | file_name | title | summary`) + **`current_docs`** cho model nhanh, nhận JSON `{intent, scope_mode, target_doc_ids, rewritten_queries}`. `scope_mode` phân biệt **lý do** phạm vi:
- `EXPLICIT_DOCS` → câu nêu/mô tả tài liệu → dùng `target_doc_ids`;
- `INHERIT_PREVIOUS` → follow-up, không nêu file mới → kế thừa `last_used_docs`;
- `GLOBAL_FOLDER` → tổng hợp cả folder / đổi chủ đề → `target = None`.

Chọn theo **nội dung** (file_name + title + summary). `rewritten_queries` (EXPLICIT/INHERIT) tách câu so sánh → multi-query. **Guardrail** `resolve_scope`: router nói `followup` mà quên docs → lấy `last_used_docs` (không bung toàn folder); `last_used_docs` lưu sau mỗi lượt, reset khi Clear/đổi folder. **Ưu tiên**: `path(s)` request → L1 → `scope_mode`/`resolve_scope` → `None`. Đã gỡ `_detect_docs`, `_DOC_*`, `_COMPARATIVE_WORDS`, `_has_contiguous_run`, `_quoted_spans`.

**Câu hỏi cấu trúc (Canonical Section Tree + Self-Querying)**:
1. Ingestion sinh **materialized path** mỗi heading: `section_id`, `section_ancestors`, `section_titles`, `section_depths`. ID giữ **số mục in trên tài liệu** khi nhất quán với cha (`4` → `s4`, `4.1` → `s4.1`); heading không đánh số / list item dùng fallback (`x1`, `s4.x1`, `sF.x1`), trùng đường dẫn thêm `-2`. Bảng dùng **cùng `section_id`** với mục chứa nó (khác `chunk_type=table`).
2. Thẻ lưu `outline_tree`; đọc từ **RAM cache**, không scroll Qdrant.
3. Khi có `doc_paths`, resolver chạy **per-document** (tối đa 3 tài liệu/lượt): `resolve_section` trả **danh sách `section_id`** (cha hoặc nhiều mục). Gate `_looks_like_section_query` chỉ gọi khi query thật sự chỉ mục. Tín hiệu **độc lập ngôn ngữ**: số mục `4.1`/`A.1`, số khớp `section_number` (`chapitre 4`, `Abschnitt 4`, `第4章`, `4장`), hoặc khớp title nhiều từ. Từ khóa `section/mục/chương` chỉ accelerator. Resolver trả `is_section_request`; số dùng làm đơn vị/version/ngày/% (`5.2 seconds`, `version 2.0`, `12%`, `cross-section`) → `false`, không lock. Câu hỏi nội dung chung không bị khoá mục.
4. `section_chunks` lọc `section_ancestors` bằng `MatchAny`: mục con → đúng nhánh (+ bảng); mục cha → **toàn cây con** (kể cả cha không có text; trước đây trả 0); nhiều mục → `MatchAny(any=[s3, s4])`.
5. Chunk từ section đánh dấu `section_locked_chunk_ids`; `grade_node` chỉ chấm phần vector còn lại (multi-doc: mục khoá đi thẳng). Đã gỡ `MatchValue(section_heading)` và `heading_search`/`_heading_matches`.

**Small-to-big**: sau retrieval, mở rộng mỗi chunk trúng ra **cả mục của nó** (`section_chunks` exact, `include_descendants=False`) trước khi trả lời — fix "PDF có thông tin mà không trả lời được".

**Q&A local-first**: `/api/ask` mặc định `allow_external=false` → retrieval yếu trả lời trung thực, **không gửi query lên web**. Bật external (Google Search grounding qua Gemini) khi `allow_external=true`.

- `rag_graph` import **không bắt buộc** `GEMINI_API_KEY` (client lazy); Gemini chỉ cho external search → Q&A nội bộ chạy hoàn toàn bằng OpenRouter.
- Dev embedded Qdrant: tránh `--reload` (mỗi reload mở lại storage). `sources` rỗng → `POST /api/sync`. Ổn định hơn: bỏ `--reload` hoặc Qdrant server mode.
- Qt/GTK chỉ cần nếu chạy **Quick native trên Linux**; Main (browser) không cần. Windows/macOS: pywebview dùng WebView2/WKWebView.
