# Improvement notes

Các điểm cần sửa, ghi lại để sau xử lý: **tình huống** + **ở đâu**.

---

## IMP-006 — ĐÃ SỬA: canonical section tree + per-document resolution

**File:** `app/core/sections.py`, `app/ingestion/chunking.py`, `app/capabilities/registry.py`, `app/retrieval/retrieve.py`, `app/reasoning/rag_graph.py`

**Đã làm:**

- Ingestion sinh **materialized path** cho từng chunk: `section_id` (`s4`, `s4.1`;
  fallback `x1`/`s4.x1`), `section_ancestors`, `section_titles`, `section_depths`,
  `section_number`, `section_title_clean`, `raw_heading`, `chunk_type`. ID giữ số mục in
  khi nhất quán với cha; trùng đường dẫn thêm `-2`. `INDEX_VERSION` → `14`,
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

## IMP-009 — ĐÃ SỬA MỘT PHẦN: `set_scope` chỉ xoá messages của chat đang mở

**File:** `app/app_state.py` (`set_scope`)

**Tình huống:** `scope` là app-level. Đổi folder xoá messages của chat đang mở + working set của mọi chat, nhưng messages của các chat khác vẫn còn và trỏ về folder cũ. Nếu sau này có UI chuyển chat, các chat đó sẽ chạy trên corpus mới với history cũ.

**Đã làm:** `set_scope(scope, conversation_id=...)` xoá `active_docs` / `primary_docs` / `last_used_docs` của **mọi** conversation (path của folder cũ không còn hợp lệ), nên working set không còn trỏ sang corpus mới.

**Còn lại:** messages của các conversation khác vẫn giữ nguyên; mở lại chat cũ sẽ thấy history của folder cũ nhưng retrieve trên folder hiện tại.

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

**File:** `app/app_state.py` (`_evict`), `app/core/config/state.py` (`max_conversations = 20`)

**Tình huống:** nếu `localStorage` giữ id của một conversation đã bị evict, lần `/api/ask` sau server tạo conversation **rỗng** cùng id đó → messages cũ mất. `_evict` hiện bảo vệ conversation đang dùng + `default`, nhưng không cứu được id cũ còn trong `localStorage`.

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

**File:** `app/reasoning/rag_graph.py` (`MAX_SECTION_CHUNKS`, `MAX_SECTION_DOCS`, per-doc budget), `app/core/config/graph.py`

**Tình huống:** mục cha rất dài (ví dụ cả chương có >24 chunk) bị cắt theo thứ tự đọc.
Câu trả lời có thể thiếu phần cuối. Khi so sánh nhiều tài liệu, `MAX_SECTION_DOCS = 3`
+ `SECTION_PER_DOC_FLOOR = 4` / `SECTION_CAP_FLOOR = 6` chia đều ngân sách để một
chương lớn không chiếm hết context, nhưng vẫn cắt thô theo thứ tự.

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

---

## IMP-018 — ĐÃ LÀM: ingestion canonical Markdown + smart assets + incremental

**File:** `app/ingestion/extract.py`, `app/ingestion/assets.py`, `app/ingestion/chunking.py`,
`app/ingestion/sync.py`, `app/ingestion/qdrant_index.py`, `app/core/config/ingestion.py`

**Đã làm:**

- **Canonical Markdown trước, block sau.** Mọi định dạng được đưa về **một** chuỗi
  Markdown rồi mới parse bằng một parser chung: `.md`/`.markdown` giữ nguyên,
  `.html`/`.htm` bỏ boilerplate rồi qua **MarkItDown**, `.docx`/`.pptx`/`.txt` qua
  **MarkItDown**, `.pdf` qua **pymupdf4llm**. Bỏ các extractor riêng lẻ; `.py`/`.json`
  không còn trong `SUPPORTED_EXTENSIONS` (scanner không đưa file không hỗ trợ vào index).
- **Chuẩn hoá trước khi parse:** bảo vệ fenced code + marker picture của PyMuPDF4LLM,
  xoá HTML comment ngoài code, xoá ảnh `![...](...)`, đổi `<br>`/`&nbsp;`/NBSP, gộp
  dòng trắng. Code fence vẫn giữ `#` bên trong là text, không thành heading.
- **Smart assets** (`ASSET_SOURCES = {.pdf, .docx, .pptx}`): PDF render **picture box**
  bằng PyMuPDF (bắt cả chart vector và trang scan, không chỉ ảnh nhúng); Office đọc
  media được tham chiếu từ `word/document.xml` / `ppt/slides/slideN.xml` (bỏ header,
  master, layout, media không dùng). Lọc `tiny_file`, `too_small*`, `thin`, `blank`,
  `small_on_page`, `repeated`, `unsupported_format`; ghi
  `data/index/assets/<file_id>/` + `manifest.json`; `remove_paths()` xoá kèm.
  Best-effort: lỗi asset không chặn index text (try/except trong `qdrant_index.py`).
- **Incremental indexing:** scan metadata-only, xác nhận bằng content hash, cache
  `file_state` SQLite, embedding cache theo text, watcher debounce (polling fallback).
- **Config tách nhóm** (`app/core/config/*`), `INDEX_VERSION` → `14`; tests mới
  `tests/test_extract.py`, `tests/test_assets.py`.

**Còn lại:**

- Asset **đã lưu nhưng chưa được dùng**: chưa có API/UI trả ảnh, chưa cite hình trong
  câu trả lời, retrieval vẫn chỉ dùng text chunk.
- Ảnh vector Office (EMF/WMF/SVG) bị loại (`unsupported_format`) vì không đo được đáng tin.
- PDF không có text layer (scan) không được OCR — chỉ text layer của PyMuPDF được dùng.
- Asset gắn với `file_id`; đổi tên/duplicate file tạo id mới nên asset được trích lại.
