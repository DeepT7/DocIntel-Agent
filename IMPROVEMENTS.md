# Improvement notes

Các điểm cần sửa, ghi lại để sau xử lý: **tình huống** + **ở đâu**.

---

## IMP-006 — ĐÃ SỬA: canonical section tree + per-document resolution

**File:** `app/core/sections.py`, `app/ingestion/folder.py`, `app/capabilities/registry.py`, `app/retrieval/retrieve.py`, `app/reasoning/rag_graph.py`

**Đã làm:**

- Ingestion sinh **materialized path** cho từng chunk: `section_id` (`s4`, `s4.1`;
  fallback `x1`/`s4.x1`), `section_ancestors`, `section_titles`, `section_depths`,
  `section_number`, `section_title_clean`, `raw_heading`, `chunk_type`. ID giữ số mục in
  khi nhất quán với cha; trùng đường dẫn thêm `-2`. `INDEX_VERSION` → `13`,
  `OUTLINE_VERSION` → `2`.
- Bảng dùng **cùng `section_id`** với mục chứa nó; hỏi mục đó trả cả text + bảng.
- `section_chunks` lọc `section_ancestors` bằng `MatchAny`:
  - mục con → đúng nhánh;
  - mục cha → **toàn bộ cây con**, kể cả mục cha không có text trực tiếp;
  - nhiều mục → một filter `MatchAny(any=[s3, s4])`.
- `resolve_section` trả **danh sách `section_id`**; resolver chạy **per-document**
  (tối đa 3 tài liệu/lượt). `grade_node` chỉ chấm phần vector chưa khoá.
- Card lưu `outline_tree` canonical; `/api/sync` cập nhật từ chunk vừa index (hoặc scroll
  Qdrant một lần nếu thiếu); chat đọc từ RAM cache, không scroll Qdrant mỗi lượt.
- **Đã xoá** `MatchValue(section_heading)`, `_outline_text`, và nhánh scroll outline cũ.

**Còn lại:** xem IMP-014 (map sub-query ↔ tài liệu khi so sánh nhiều file).

---

## IMP-007 — Chưa có UI chuyển conversation

**File:** `app/server/static/app.js` (nút New chat), `app/server/static/index.html`

**Tình huống:** `POST /api/conversation/new` tạo chat rỗng mới, nhưng UI không có danh sách để quay lại chat cũ. Chat cũ vẫn nằm trong `state["conversations"]` (đến khi bị evict) nhưng không truy cập được.

**Hướng sửa:** dropdown/history panel dùng `GET /api/conversations`; chọn id → `storeConversation(id)` + `loadState()` + `loadSession()`.

---

## IMP-008 — Chat `default` cũ không tự hiện sau khi nâng cấp

**File:** `app/server/static/app.js` (`ensureConversation`)

**Tình huống:** state phẳng cũ được migrate vào `conversations["default"]`, nhưng web tạo UUID mới ngay lần đầu mở. Chat cũ vẫn còn trong file nhưng không được load lên UI.

**Hướng sửa:** nếu chỉ có `default` và `default.messages` non-empty, dùng `default` làm id đầu tiên; hoặc hiện một mục "Previous chat" một lần.

---

## IMP-009 — `set_scope` chỉ xoá messages của chat đang mở

**File:** `app/app_state.py` (`set_scope`)

**Tình huống:** `scope` là app-level. Đổi folder xoá messages của chat đang mở + working set của mọi chat, nhưng messages của các chat khác vẫn còn và trỏ về folder cũ. Nếu sau này có UI chuyển chat, các chat đó sẽ chạy trên corpus mới với history cũ.

**Hướng sửa:** lưu `scope` trong từng conversation và lọc theo scope hiện tại; hoặc xoá messages của mọi chat khi đổi folder (đơn giản nhưng mất lịch sử).

---

## IMP-010 — Cách ly graph là quy ước, chưa có test chặn

**File:** `app/reasoning/*`, `app/capabilities/*`, `app/retrieval/*`

**Tình huống:** hiện không node nào import `app_state` (đã audit). Nếu sau này thêm node đọc `app_state.get_active_docs()` trực tiếp, contamination giữa các conversation sẽ quay lại mà không có cảnh báo.

**Hướng sửa:** thêm test/CI grep cấm `app_state` trong `app/reasoning`, `app/capabilities`, `app/retrieval`; chỉ cho phép dữ liệu đi qua `RAGState`.

---

## IMP-011 — `RLock` chỉ đúng trong một process

**File:** `app/app_state.py` (`_LOCK`)

**Tình huống:** `threading.RLock` bảo vệ các thread trong cùng process. Chạy `uvicorn --workers > 1` (hoặc nhiều process) thì hai process vẫn có thể ghi đè `app_state.json`.

**Hướng sửa:** dùng `filelock` / `msvcrt.locking` / `fcntl` khi cần multi-process; hiện tại chấp nhận single-process.

---

## IMP-012 — Eviction có thể làm mất chat đang dùng

**File:** `app/app_state.py` (`MAX_CONVERSATIONS = 20`)

**Tình huống:** nếu `localStorage` giữ id của một conversation đã bị evict, lần `/api/ask` sau server tạo conversation **rỗng** cùng id đó → messages cũ mất.

**Hướng sửa:** tăng cap; hoặc đánh dấu chat đang mở là "pinned" không evict; hoặc client kiểm tra id còn trong `/api/conversations` trước khi dùng và tạo chat mới nếu không.

---

## IMP-013 — Bấm New chat giữa lúc request đang chạy

**File:** `app/server/static/app.js` (`send` + nút New chat)

**Tình huống:** response của request cũ (mang `conversation_id` cũ) vẫn render vào UI sau khi người dùng đã bấm New chat → câu trả lời nằm nhầm chat.

**Hướng sửa:** disable New chat khi đang gửi; hoặc kiểm tra `r.conversation_id === conversationId` trước khi render, nếu khác thì bỏ qua/cảnh báo.

---

## IMP-014 — ĐÃ SỬA (best-effort): map sub-query ↔ tài liệu khi so sánh

**File:** `app/reasoning/rag_graph.py` (`retrieve_node`)

**Đã làm:**

- Nếu `len(rewritten_queries) == len(doc_paths)`, các query **khác nhau**, và mỗi query
  **không nhắc tài liệu khác** → dùng từng query cho từng tài liệu theo thứ tự.
- Nếu router trả query trùng/mixed (đã gặp), fallback: lấy các **số mục theo đúng thứ tự
  xuất hiện** trong câu gốc (`4`, `3`) và ghép với `doc_paths` theo thứ tự
  (`section 4` → tài liệu 1, `section 3` → tài liệu 2).
- Nếu câu chỉ nhắc **một** số mục → dùng cùng query cho mọi tài liệu (so sánh cùng mục).
- Nếu có nhiều số mục nhưng không khớp số tài liệu → **không đoán**, fallback vector search.

**Còn lại:** câu so sánh không có số mục và không khớp title (chỉ dùng từ đồng nghĩa) có thể
không resolve được nếu router trả query mixed; khi đó vẫn fallback vector search.

---

## IMP-015 — Câu hỏi mục khi chưa chọn tài liệu là mơ hồ

**File:** `app/reasoning/rag_graph.py` (`retrieve_node`)

**Tình huống:** khi `doc_paths=None` (toàn folder, chat mới), "section 4.1" có thể khớp
nhiều tài liệu. Hệ thống **không** section-lock (đúng nguyên tắc không đoán), nên trả về
vector search + small-to-big. Muốn khoá đúng mục, người dùng chọn source ở sidebar hoặc
hỏi tiếp sau khi đã mở một tài liệu.

**Hướng sửa (tuỳ chọn):** nếu đúng **một** card có `section_number`/title khớp query thì
tự chọn tài liệu đó; nếu nhiều card khớp thì giữ nguyên hành vi hiện tại.

---

## IMP-016 — Cap chunk của section cha

**File:** `app/reasoning/rag_graph.py` (`MAX_SECTION_CHUNKS = 24`)

**Tình huống:** mục cha rất dài (ví dụ cả chương có >24 chunk) bị cắt theo thứ tự đọc.
Câu trả lời có thể thiếu phần cuối.

**Hướng sửa:** tăng cap theo ngân sách token của model; hoặc lấy toàn bộ mục rồi chọn
top-N chunk theo relevance trong nội bộ mục thay vì cắt thô theo thứ tự.

---

## IMP-017 — Cross-language synonym không có số mục / title

**File:** `app/reasoning/answer.py` (`_looks_like_section_query`)

**Tình huống:** gate đã độc lập ngôn ngữ với **số mục** (`4.1`, `chapitre 4`, `Abschnitt 4`,
`第4章`, `4장`) và **title nhiều từ** của chính tài liệu. Nhưng câu chỉ dùng từ đồng nghĩa
ở ngôn ngữ khác, không có số và không khớp title (ví dụ tài liệu tiếng Anh có mục
"3 Methodology", người dùng hỏi "la méthodologie"), sẽ **không gọi resolver** → trả lời
bằng vector search, không khoá mục.

**Đã giảm false positive:** resolver trả thêm `is_section_request`. Các câu
`5.2 seconds`, `version 2.0`, `12%`, `What is the cross-section area?` → `false` và
**không lock**; `section 5.2` → `s5.2`. Gate cũng không còn match `section` trong
`cross-section` / `sectional`.

**Hướng sửa (nếu cần):** dùng semantic similarity giữa query và title các mục (embedding
đã có) làm tín hiệu gate; hoặc gọi resolver không điều kiện cho chat 1 tài liệu và thêm
post-check để tránh false positive.
