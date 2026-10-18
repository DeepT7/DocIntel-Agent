# PLAN — DocIntel: trợ lý tài liệu local-first (một lõi, nhiều bề mặt)

> Trạng thái: Phase 0 → Phase 1 đã xong (đã test headless). Hướng mới: **API-first**,
> bỏ phụ thuộc Qt/GTK/Electron cho cửa sổ chính.

## 1. Mục tiêu

Một hệ thống local-first trên **thư mục tài liệu của người dùng**:

- **Main (đầy đủ)**: hỏi–đáp, tìm thông tin, tóm tắt/insight, hệ thống hóa; về sau: flashcard/slide.
- **Quick (ambient)**: popup nhỏ qua hotkey — hỏi nhanh, tìm thông tin, **gợi ý viết**
  dựa trên đoạn đang viết (clipboard) + kho tài liệu trong folder.

Mọi bề mặt **chia sẻ một lõi**: cùng scope (folder), cùng index Qdrant, cùng lịch sử.
Người dùng trỏ tới một folder; file sửa liên tục được **sync incremental** khi cần.

## 2. Nguyên tắc

1. **API-first** — mọi capability đi qua HTTP; UI chỉ là client. Nhờ vậy Main window
   là **trang web trong trình duyệt**, không cần Qt/GTK/Electron.
2. **Một process sở hữu Qdrant** — tránh lỗi "storage already accessed".
3. **Capabilities tách khỏi UI** — logic nằm ở `app/capabilities/`, không UI nào tự
   đụng Qdrant.
4. **Ambient = clipboard, không auto-capture** ở MVP (không cần Accessibility permission).
5. **Sync theo yêu cầu**, không watcher liên tục. Save + sync trước khi hỏi là đủ.

## 3. Chắt lọc từ Open Notebook (lfnovo/open-notebook)

| Pattern của họ | Áp dụng tại đây | Trạng thái |
|---|---|---|
| **API-first**: UI là client | FastAPI `app/server/main.py`; `/` trả trang web Main | ✅ |
| **Insight**: summary per source, tái dùng | `capabilities/insights.py` (map-reduce + cache theo hash) | ✅ |
| **Async-first**: job nền + status | `app/jobs.py`; `POST` trả `job_id`, `GET` poll | ✅ |
| **Transformation-as-data** (task = prompt config) | dự kiến `capabilities/tasks.py` | ⏳ |
| **Provider abstraction** (Esperanto) | dự kiến `app/core/llm.py` | ⏳ |
| **Insight embed để search được** | dự kiến thêm `doc_type=insight` vào Qdrant | ⏳ |
| **MCP** (agent dùng hệ thống) | dự kiến, xa | ⏳ |

**Không copy**: SurrealDB/Next.js/Docker (giữ Python/Qdrant); "add source thủ công"
(giữ folder sync incremental); web-only (giữ đường ambient).

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

- **Main** = trang web phục vụ bởi FastAPI (`http://localhost:8756`). Không Qt.
- **Quick** = popup native riêng (pywebview + hotkey + clipboard), gọi cùng API/bridge.
- Có thể thay Main bằng **frontend TypeScript (Vite)** gọi cùng HTTP API.

## 5. Mô hình khái niệm

| Khái niệm | Ở đây | Ghi chú |
|---|---|---|
| **Scope** | một folder root (`root` payload) | được chuẩn hoá `normalize_root` |
| **Source** | một file trong scope | `path`, `file_id`, `title` |
| **Chunk** | đơn vị retrieval | `dense` + `bm25`, đã có `table_number`/`caption` |
| **Insight** | tóm tắt cache của 1 file | map-reduce, invalidate theo hash |
| **Answer** | câu trả lời hỏi–đáp trên folder | qua `rag_graph` có `scope`, đa lượt |
| **Suggestion** | gợi ý viết (query + clipboard) | có trích dẫn nguồn |
| **Job** | tác vụ nền | `queued|running|done|error`, có `result` |

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
app/capabilities/workspace.py list_files(scope)  (liệt kê file trong folder)
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
| GET | `/api/selection` | — | `{excluded_docs}` — file người dùng BỎ TÍCH (loại khỏi truy vấn) |
| POST | `/api/selection` | `{paths}` | đặt danh sách loại trừ; trả `{excluded_docs}` |
| GET | `/api/outline` | `?scope=&path=` | `{title, sections:[title…], tree:[{id,title,level,children}]}` |
| GET | `/api/section` | `?scope=&path=&section_id=` | chunk của một/nhiều mục; `section_id` cách nhau dấu phẩy; cha → cả cây con |
| GET | `/api/registry` | `?scope=` | danh sách "thẻ căn cước" tài liệu (title, summary, keywords, sections) |
| GET | `/api/card` | `?scope=&path=` | thẻ của một file |
| POST | `/api/registry` | `{scope?, force?}` | dựng thẻ còn thiếu (background job) → `{job_id}` |
| POST | `/api/search` | `{query, scope?, path?, paths?, top_k}` | hits (có `score`, `semantic_score`, `path`, `section_heading`) |
| POST | `/api/suggest` | `{query, clipboard, scope?}` | `{suggestion, sources}` |
| POST | `/api/ask` | `{query, scope?, conversation_id?, path?, paths?, top_k?, history?, use_session?, use_selection?, notes?, allow_external?, save?}` | `{answer, intent, scope_mode, conversation_id, standalone_query, sources, scope, doc_paths, routed_queries}` |
| GET | `/api/session` | `?conversation_id=` | lịch sử hội thoại (`[{role,content}]`) |
| POST | `/api/session/clear` | `{conversation_id?}` | `{session: []}` |
| POST | `/api/conversation/new` | — | `{conversation_id}` (chat mới, rỗng) |
| GET | `/api/conversations` | — | danh sách chat `[{conversation_id, messages, updated_at}]` |
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
user Ctrl+C (đoạn đang viết) → POST /api/suggest {query, clipboard}
 → sync scope (await) → vector_search(query+clipboard, scope)
 → prompt suggest + context → gợi ý có [Source N]
```

**B. Tìm thông tin (Main)**
```
POST /api/search → hybrid RRF trong scope → hits → hiển thị + nguồn
```

**B2. Hỏi đáp trên folder (RAG đa lượt)**
```
POST /api/ask {query}  (history lấy từ session)
 → run_rag(scope): condense(có history) → retrieve(scope) → grade
   → nếu yếu: rewrite 1 lần → grade lại
   → allow_external=false (mặc định local-first) → fallback trung thực
   → answer + sources (path trong folder)
 → append_turn vào session
```

**C. Tóm tắt file (Insight, chạy nền)**
```
POST /api/insight {path} → job_id
 worker: đọc chunk của file (theo root+path) → 1 pass nếu ngắn / map-reduce nếu dài
        → cache theo file_hash → GET /api/job/{id} → done
```

## 9. Lộ trình

- [x] **Phase 0 — lõi**: `root`/`scope`, `vector_search(scope)`, capabilities
  (`search`,`sources`,`sync`,`suggest`), `app_state`, prompt `suggest`.
- [x] **Phase 0.5 — API-first + Insight + jobs**: FastAPI, `jobs.py`,
  `capabilities/insights.py`. Test: `/`, sync incremental, sources, search, suggest,
  insight job + **cache hit lần 2 (0 LLM call)**, list insights.
- [x] **Phase 1 — Quick popup native**: `desktop/*` (bridge + hotkey + clipboard),
  chưa chạy trên WSL (thiếu display).
- [x] **Phase 2 — Hỏi đáp trên folder**: nối `rag_graph` vào API qua
  `capabilities/ask.py`; `run_rag` nhận `scope` + `allow_external`; `retrieve_node`
  lọc theo scope; session đa lượt. Test: hỏi tiếng Việt → trả lời có nguồn
  `memory.md`, follow-up được condense đúng, session + clear OK.
- [ ] **Phase 2b — Main hoàn thiện**: citation click mở file; hiển thị streaming;
  hoặc thay trang `/` bằng frontend TS (Vite).
- [ ] **Phase 3 — Tasks (as data + job)**:
  `synthesize` (hệ thống hóa) → `flashcards` → `slides`; mỗi task là job nền.
- [ ] **Phase 4 — Insight search được + polish**: embed insight (`doc_type=insight`)
  để `search` trả về cả insight; tray; đóng gói.
- [ ] **Phase 5 — Provider abstraction + MCP** (xa).

## 10. Chạy

```bash
# Main (browser, không Qt)
pip install fastapi uvicorn
uvicorn app.server.main:app --port 8756     # KHÔNG dùng --reload với Qdrant embedded
# mở http://localhost:8756  (Q&A, search, suggest, insight)

# Quick popup native (tuỳ chọn, cần desktop có màn hình)
pip install pywebview pynput pyperclip
python -m desktop.app          # Ctrl+Alt+Space mở Quick
```

Chỉ mở **một** tiến trình dùng Qdrant tại một thời điểm.

## 11. Ghi chú kỹ thuật

- `root` lưu trong payload, chuẩn hoá bằng `normalize_root` (absolute, forward slash);
  query lọc `root` khớp **chính xác** (scope cha không lẫn con).
- `file_id = md5(rel_path)` — nếu đa root, `rel_path` trùng sẽ trùng `file_id`;
  phải key theo `(root, rel_path)` khi mở rộng.
- `INDEX_VERSION=9` buộc re-upsert payload có `root` (dense tái dùng cache, không re-embed).
- Qdrant embedded chỉ một process: mọi truy cập qua `app/retrieval/store.get_client()`.
- Insight cache ở `data/index/insights/<root_key>_<file_key>.json`, invalidate theo
  `file_hash`; tài liệu ngắn 1 pass, dài map-reduce (`SINGLE_PASS_CHARS`, `MAP_GROUP_CHARS`).
- Jobs là thread + dict trong process (đủ cho MVP); thay bằng queue thật khi cần.
- **Intent routing (chat / followup / question / workspace)**: mỗi lượt qua
  `classify_node` — utility model phân loại (đa ngữ, không liệt kê từ khoá).
  `chat` → `chat_node`; `workspace` → `workspace_node` (liệt kê file trong folder,
  **không** retrieval); `followup` → condense → retrieval trong phạm vi hội thoại;
  `question` → retrieval **toàn folder** (bỏ condense). Fail-safe về `question`.
  Pattern chuẩn (LlamaIndex RouterQueryEngine / semantic router); zero-cost: thay bằng
  embedding router. `workspace` dùng `capabilities/workspace.py:list_files`.
- **Cô lập hội thoại (IMP-003)**: state hai cấp — app-level (`scope`, `recent`,
  `excluded_docs`) và conversation-level (`conversations[conv_id]` với `messages`,
  `active_docs`, `primary_docs`, `last_used_docs`). `new_conversation()` tạo id mới rỗng;
  mỗi request `/api/ask` mang `conversation_id`, nên `last_used_docs` là
  **per-conversation** và không rò rỉ giữa các chat. `set_scope` reset chat đang mở +
  working set của mọi chat (path thuộc folder cũ). **Graph không đọc `app_state`**:
  `main.ask` nạp dữ liệu theo `conv_id` rồi truyền vào `run_rag` (dependency injection
  qua `RAGState`); node chỉ đọc `state`, nên một turn không thể thấy state của chat khác.
- **Cấp độ 1 — chọn nguồn tường minh (UI)**: sidebar có panel **Sources** với checkbox mỗi
  file. Bỏ tích file nào → gửi vào `/api/selection` (`excluded_docs`); web gửi
  `use_selection: true` để `/api/ask` tính `selected = tất cả nguồn − excluded` **trên server**
  và dùng làm phạm vi, **ưu tiên hơn router** (không đoán khi người dùng đã chọn). Cờ này
  phân biệt "không lọc" với "bỏ chọn hết": bỏ hết → **No-Context / Notes Only** (xem dưới),
  **không** fallback về toàn folder. Mặc định "all" → không lọc (toàn folder). Lựa chọn lưu
  trong state; reset khi đổi folder.
- **No-Context / Notes Only**: khi mọi nguồn Off, `/api/ask` **không gọi router** và không gửi
  Document Cards; gọi `capabilities/ask.py:answer_without_context()` với **system prompt**
  `NO_CONTEXT_SYSTEM`. Model chỉ được dùng: lịch sử hội thoại, notes đang mở (`notes` trong
  request), **tên file** (metadata, để list/count) và kiến thức chung. Câu hỏi cần nội dung
  tài liệu → xin user bật ít nhất 1 nguồn (không bịa, không lấy từ lịch sử); câu hỏi kiến
  thức chung → mở đầu bằng câu nhắc "chưa bật tài liệu"; chat vẫn trả lời tự nhiên.
- **Cấp độ 2a — Contextual Chunk Headers (phân cấp, clean)**: mỗi chunk được gắn header
  `[Document: <file> | Section: <breadcrumb>]` **trước khi embed**; breadcrumb lấy **tối đa
  3 bậc** từ cây canonical (không còn `**` hay hậu tố `(Table)`). Payload lưu thêm:
  `section_id` (`s4`, `s4.1`), `section_ancestors` (gồm chính nó), `section_titles`,
  `section_depths`, `section_number`, `section_title_clean`, `raw_heading`, `chunk_type`.
  `INDEX_VERSION` → `12`.
- **Cấp độ 2b — Document Registry (thẻ căn cước)**: `capabilities/registry.py` dựng một
  thẻ/file (`title`, `summary`, `keywords/entities`, `sections`), cache theo content-hash,
  lưu `data/index/registry/<root>.json`. Thẻ lưu **`outline_tree`** — cây canonical
  (`id`, `title`, `level`, `children`) — và `sections` là bản text thụt cấp của cùng cây:
  ```
  - Chương 1: Quy định chung
    - 1.1. Phạm vi điều chỉnh
    - 1.2. Đối tượng áp dụng
  ```
  `POST /api/sync` cập nhật `outline_tree` từ chunk vừa index (hoặc scroll Qdrant **một lần**
  nếu thẻ thiếu); chat đọc cây từ **RAM cache**, **không scroll Qdrant mỗi lượt**.
  Endpoint `/api/registry`, `/api/card`, `POST /api/registry` (job). Mục đích: tìm file theo
  **nội dung** dù chunk không nêu tên file (vd "hướng dẫn Docker" cho file chỉ ghi "Mở port 8502").
- **Cấp độ 3 — LLM Turn Router (thay cả `classify_intent` lẫn `_detect_docs`)**: MỘT call
  `capabilities/router.py` đưa query + history + **Document Cards** (`id | file_name | title | summary`)
  + **`current_docs`** (scope chính xác của lượt trước) cho model nhanh, nhận JSON
  `{intent, scope_mode, target_doc_ids, rewritten_queries}`. `scope_mode` phân biệt **lý do**
  của phạm vi, nên `target_doc_ids: []` không còn mơ hồ:
  - `EXPLICIT_DOCS` → câu nêu/mô tả tài liệu (kể cả file mới) → dùng `target_doc_ids`;
  - `INHERIT_PREVIOUS` → follow-up, không nêu file mới → kế thừa `last_used_docs` của lượt trước;
  - `GLOBAL_FOLDER` → câu tổng hợp cả folder hoặc chuyển chủ đề → `target = None`.
  Chọn tài liệu theo **nội dung** (file_name + title + summary), không so tên file.
  `rewritten_queries` (khi EXPLICIT/INHERIT) tách câu so sánh → `retrieve_node` chạy multi-query.
  **Deterministic guardrail** (`resolve_scope`): nếu router nói `followup` mà quên docs →
  lấy `last_used_docs` thay vì bung ra toàn folder; `last_used_docs` lưu sau mỗi lượt và
  reset khi Clear chat/đổi folder. **Ưu tiên**: `path(s)` trong request → lựa chọn L1 →
  `scope_mode`/`resolve_scope` → `None`. Đã gỡ hẳn `_detect_docs`, `_DOC_*`,
  `_COMPARATIVE_WORDS`, `_has_contiguous_run`, `_quoted_spans`.
- **Câu hỏi cấu trúc (Canonical Section Tree + Self-Querying)**:
  (1) Ingestion sinh **materialized path** cho từng heading: `section_id` (`s1`, `s1.1`),
  `section_ancestors` (gồm chính nó), `section_titles`, `section_depths`. ID là **cấu trúc**,
  không phụ thuộc text, nên heading trùng tên / không đánh số vẫn tách biệt; bảng dùng
  **cùng `section_id`** với mục chứa nó (chỉ khác `chunk_type=table`).
  (2) Thẻ lưu `outline_tree`; `document_outline` đọc từ **RAM cache**, không scroll Qdrant.
  (3) Khi có `doc_paths`, resolver chạy **per-document** (tối đa 3 tài liệu/lượt):
  `resolve_section` trả **danh sách `section_id`** (JSON), có thể là mục cha hoặc nhiều mục
  ("so sánh mục 3 và 4"). Một **deterministic gate** (`_looks_like_section_query`) chỉ gọi
  resolver khi query thật sự chỉ mục (có "section/mục/chương", số dạng `4.1`, hoặc khớp
  tiêu đề nhiều từ); câu hỏi nội dung chung như "what datasets did they build?" **không**
  bị khoá mục.
  (4) `section_chunks` lọc `section_ancestors` bằng `MatchAny`:
  - mục con → đúng nhánh đó (kèm bảng);
  - mục cha → **toàn bộ cây con**, kể cả mục cha không có text trực tiếp (trước đây trả 0);
  - nhiều mục → một filter `MatchAny(any=[s3, s4])`.
  (5) Chunk lấy từ section được đánh dấu `section_locked_chunk_ids`; `grade_node` chỉ chấm
  phần vector còn lại (multi-doc: mục đã khoá đi thẳng, phần chưa khoá vẫn qua grader).
  **Đã gỡ hẳn** `MatchValue(section_heading)` và cơ chế khớp token tên mục cũ
  (`heading_search`/`_heading_matches`).
- **Small-to-big**: sau retrieval, mở rộng mỗi chunk trúng ra **cả mục của nó**
  (`section_chunks` với `section_id` exact, `include_descendants=False`) trước khi trả lời —
  vì một mục thường bị cắt thành nhiều chunk và câu trả lời nằm ở chunk anh em. Đây là fix
  cho "PDF có thông tin mà không trả lời được".
- **Q&A local-first**: `/api/ask` mặc định `allow_external=false` → khi retrieval yếu
  thì trả lời trung thực, **không gửi query lên web**. Chỉ bật external (Google Search
  grounding qua Gemini) khi request `allow_external=true`.
- `rag_graph` import **không còn bắt buộc** `GEMINI_API_KEY` (client tạo lazy); Gemini
  chỉ dùng cho external search. Vì vậy Q&A nội bộ chạy hoàn toàn bằng OpenRouter.
- Dev với embedded Qdrant: nên tránh `--reload` (mỗi reload mở lại storage). Nếu
  `sources` bỗng rỗng, gọi `POST /api/sync`. Chạy ổn định hơn: bỏ `--reload` hoặc
  chuyển Qdrant sang server mode.
- Qt/GTK chỉ cần nếu chạy **Quick native trên Linux**; Main (browser) không cần.
  Trên Windows/macOS, pywebview dùng WebView2/WKWebView — không Qt.

## 12. Việc tiếp theo (đề xuất)

1. **Embed insight** để search trả về insight (`doc_type=insight`, cite được).
2. **Transformation-as-data**: task = `{name, prompt, model_id}` → thêm task không cần code.
3. **Frontend TypeScript (Vite)** gọi API HTTP — thoả yêu cầu TS, vẫn không Qt.
4. **Đóng gói**: PyInstaller (Windows) cho lõi+API; Quick popup đóng gói riêng.