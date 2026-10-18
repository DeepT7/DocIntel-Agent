const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ status */
let statusTimer;
function status(msg, isErr = false) {
  const el = $('status');
  el.textContent = msg;
  el.classList.toggle('err', isErr);
  el.classList.add('show');
  clearTimeout(statusTimer);
  statusTimer = setTimeout(() => el.classList.remove('show'), 2400);
}

/* --------------------------------------------------------------------- api */
async function api(url, { method = 'GET', body } = {}) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    opt.headers['Content-Type'] = 'application/json';
    opt.body = JSON.stringify(body);
  }
  const res = await fetch(url, opt);
  const raw = await res.text();
  let data = null;
  try { data = raw ? JSON.parse(raw) : null; } catch { data = raw; }
  if (!res.ok) throw new Error((data && data.detail) || res.statusText);
  return data;
}

// Shared wrapper: shows a spinner on the button while the task runs.
async function run(btn, okMsg, fn) {
  if (btn) { btn.classList.add('loading'); btn.disabled = true; }
  let ok = true;
  try {
    await fn();
    if (okMsg) status(okMsg);
  } catch (e) {
    ok = false;
    status(e.message || 'Request failed', true);
  } finally {
    if (btn) { btn.classList.remove('loading'); btn.disabled = false; }
  }
  return ok;
}

/* ------------------------------------------------------------ conversation */
// Each chat gets its own id, so messages/working-set/inherited scope never mix
// between conversations (or between two browser windows).
const CONV_KEY = 'docintel_conversation_id';
let conversationId = '';
try { conversationId = localStorage.getItem(CONV_KEY) || ''; } catch { conversationId = ''; }

function storeConversation(id) {
  conversationId = id;
  try { localStorage.setItem(CONV_KEY, id); } catch { /* private mode */ }
}

function convQuery() {
  return '?conversation_id=' + encodeURIComponent(conversationId);
}

async function ensureConversation() {
  if (conversationId) return conversationId;
  const r = await api('/api/conversation/new', { method: 'POST', body: {} });
  storeConversation(r.conversation_id);
  return conversationId;
}

/* --------------------------------------------------------------- markdown */
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function inlineMd(s) {
  s = escapeHtml(s);
  s = s.replace(/`([^`]+)`/g, '<code class="inline">$1</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/__([^_]+)__/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  s = s.replace(/(^|[^_\w])_([^_\n]+)_/g, '$1<em>$2</em>');
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}

// Self-contained Markdown -> HTML (escapes first, so it is XSS-safe).
function renderMarkdown(src) {
  if (!src) return '';
  const lines = String(src).replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  const isBlockStart = (l) =>
    /^```/.test(l) || /^#{1,6}\s/.test(l) || /^>\s?/.test(l) ||
    /^\s*[-*+]\s+/.test(l) || /^\s*\d+[.)]\s+/.test(l) ||
    /^\s*([-*_])\1{2,}\s*$/.test(l);

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    const fence = line.match(/^```(.*)$/);
    if (fence) {
      const lang = fence[1].trim();
      const buf = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++;
      out.push(`<pre class="code"><code${lang ? ` data-lang="${escapeHtml(lang)}"` : ''}>${escapeHtml(buf.join('\n'))}</code></pre>`);
      continue;
    }
    if (!line.trim()) { i++; continue; }

    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { out.push(`<h${h[1].length}>${inlineMd(h[2])}</h${h[1].length}>`); i++; continue; }

    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { out.push('<hr>'); i++; continue; }

    if (/^>\s?/.test(line)) {
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^>\s?/, '')); i++; }
      out.push(`<blockquote>${renderMarkdown(buf.join('\n'))}</blockquote>`);
      continue;
    }

    // table: header row + separator row
    if (line.includes('|') && i + 1 < lines.length &&
        lines[i + 1].includes('-') && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1])) {
      const cells = (l) => l.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes('|')) { rows.push(cells(lines[i])); i++; }
      out.push(
        '<table><thead><tr>' + head.map((c) => `<th>${inlineMd(c)}</th>`).join('') + '</tr></thead><tbody>' +
        rows.map((r) => '<tr>' + r.map((c) => `<td>${inlineMd(c)}</td>`).join('') + '</tr>').join('') +
        '</tbody></table>'
      );
      continue;
    }

    if (/^\s*[-*+]\s+/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) { buf.push(lines[i].replace(/^\s*[-*+]\s+/, '')); i++; }
      out.push('<ul>' + buf.map((x) => `<li>${inlineMd(x)}</li>`).join('') + '</ul>');
      continue;
    }

    if (/^\s*\d+[.)]\s+/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) { buf.push(lines[i].replace(/^\s*\d+[.)]\s+/, '')); i++; }
      out.push('<ol>' + buf.map((x) => `<li>${inlineMd(x)}</li>`).join('') + '</ol>');
      continue;
    }

    const buf = [line];
    i++;
    while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i])) { buf.push(lines[i]); i++; }
    out.push(`<p>${inlineMd(buf.join('\n')).replace(/\n/g, '<br>')}</p>`);
  }
  return out.join('\n');
}

/* ------------------------------------------------------------ shared bits */
function sourcesHtml(sources) {
  if (!sources || !sources.length) return '';
  const items = sources.map((s, i) => {
    const label = s.title || s.path || 'source';
    const sectionLabel = s.section_title_clean || s.section_heading;
    const sec = sectionLabel ? ' · ' + sectionLabel : '';
    const score = typeof s.score === 'number' ? ` · ${s.score.toFixed(3)}` : '';
    return `<span class="src"><b>[${i + 1}]</b> ${escapeHtml(label)}${escapeHtml(sec)}${score}</span>`;
  }).join('');
  return `<div class="sources"><div class="muted" style="margin-bottom:4px">Sources</div>${items}</div>`;
}

function preHtml(value) {
  return `<div class="md"><pre class="code"><code>${escapeHtml(JSON.stringify(value, null, 2))}</code></pre></div>`;
}

/* -------------------------------------------------------------------- chat */
function scrollDown() {
  const m = $('messages');
  m.scrollTop = m.scrollHeight;
}

function clearEmpty() {
  const e = $('msglist').querySelector('.empty');
  if (e) e.remove();
}

function showEmpty() {
  $('msglist').innerHTML =
    '<div class="empty"><h2>Ask your documents</h2>' +
    '<p>Pick a folder on the left, update the index, then ask anything.</p></div>';
}

function addUser(text) {
  clearEmpty();
  const el = document.createElement('div');
  el.className = 'msg user';
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  el.appendChild(b);
  $('msglist').appendChild(el);
  scrollDown();
  return el;
}

function addBot(text, thinking = false, sources = []) {
  clearEmpty();
  const el = document.createElement('div');
  el.className = 'msg bot';
  const av = document.createElement('div');
  av.className = 'avatar';
  av.textContent = 'AI';
  const c = document.createElement('div');
  c.className = 'content md' + (thinking ? ' thinking' : '');
  if (thinking) c.textContent = text;
  else c.innerHTML = renderMarkdown(text) + sourcesHtml(sources);
  el.append(av, c);
  $('msglist').appendChild(el);
  scrollDown();
  return el;
}

function autoGrow(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

async function send(btn) {
  const input = $('aq');
  const query = input.value.trim();
  if (!query) return;
  addUser(query);
  input.value = '';
  autoGrow(input);

  const thinking = addBot('Thinking…', true);
  if (btn) { btn.classList.add('loading'); btn.disabled = true; }
  try {
    await ensureConversation();
    // The server owns the selection (excluded_docs); the flag tells it to apply it.
    const body = { query, use_selection: true, conversation_id: conversationId };
    const r = await api('/api/ask', { method: 'POST', body });
    thinking.remove();
    addBot(r.answer || '(no answer)', false, r.sources || []);
    if (r.working_set) renderWorkingSet(r.working_set);
    status('Answer ready');
  } catch (e) {
    thinking.remove();
    addBot('Error: ' + (e.message || 'request failed'), false, []);
    status(e.message, true);
  } finally {
    if (btn) { btn.classList.remove('loading'); btn.disabled = false; }
    input.focus();
  }
}

async function loadSession() {
  try {
    const s = await api('/api/session' + convQuery());
    if (Array.isArray(s) && s.length) {
      $('msglist').innerHTML = '';
      s.forEach((m) => (m.role === 'user' ? addUser(m.content) : addBot(m.content, false, [])));
    } else {
      showEmpty();
    }
  } catch {
    showEmpty();
  }
}

/* ------------------------------------------------------------ working set */
let workingSet = [];

function baseName(p) {
  return String(p || '').split('/').filter(Boolean).pop() || p;
}

function renderWorkingSet(docs) {
  workingSet = Array.isArray(docs) ? docs : [];
  const el = $('workingset');
  el.innerHTML = workingSet.map((p) =>
    `<span class="chip" title="${escapeHtml(p)}">` +
      `<span class="chip-label">${escapeHtml(baseName(p))}</span>` +
      `<button class="chip-x" data-path="${escapeHtml(p)}" aria-label="Unpin">×</button>` +
    `</span>`
  ).join('');
}

async function unpin(path) {
  const remaining = workingSet.filter((p) => p !== path);
  try {
    const r = await api('/api/working-set', {
      method: 'POST',
      body: { paths: remaining, conversation_id: conversationId },
    });
    renderWorkingSet(r.active_docs || remaining);
    status('Unpinned ' + baseName(path));
  } catch (e) {
    status(e.message, true);
  }
}

/* -------------------------------------------------------- source toggles */
// Explicit selection: unchecked files are removed from the query scope. This
// beats guessing from the message text — the user's choice is authoritative.
let allSources = [];
let excludedSet = new Set();

function renderSources() {
  const el = $('sources');
  if (!allSources.length) {
    el.innerHTML = '<div class="muted">No indexed files. Pick a folder and Update index.</div>';
    updateSourceHint();
    return;
  }
  el.innerHTML = allSources.map((p) => {
    const on = !excludedSet.has(p);
    return '<label class="src-item">' +
      '<input type="checkbox" data-path="' + escapeHtml(p) + '"' + (on ? ' checked' : '') + '>' +
      '<span class="src-name" title="' + escapeHtml(p) + '">' + escapeHtml(baseName(p)) + '</span>' +
      '</label>';
  }).join('');
  updateSourceHint();
}

function updateSourceHint() {
  const total = allSources.length;
  const active = total - excludedSet.size;
  if (!total) {
    $('srcHint').textContent = '';
    return;
  }
  $('srcHint').textContent = active === 0
    ? `0/${total} selected · No-Context mode`
    : `${active}/${total} selected`;
}

async function loadSources() {
  try {
    const list = await api('/api/sources');
    allSources = (list || []).map((s) => s.path).filter(Boolean);
  } catch {
    allSources = [];
  }
  // Drop stale selections that no longer exist in the folder.
  excludedSet = new Set([...excludedSet].filter((p) => allSources.includes(p)));
  renderSources();
}

async function saveSelection() {
  try {
    await api('/api/selection', { method: 'POST', body: { paths: [...excludedSet] } });
  } catch (e) {
    status(e.message, true);
  }
}

function onSourceToggle(e) {
  const cb = e.target.closest('input[type="checkbox"]');
  if (!cb || !cb.dataset.path) return;
  if (cb.checked) excludedSet.delete(cb.dataset.path);
  else excludedSet.add(cb.dataset.path);
  updateSourceHint();
  saveSelection();
}

async function setAllSources(all) {
  excludedSet = all ? new Set() : new Set(allSources);
  renderSources();
  await saveSelection();
}

/* ------------------------------------------------------------ side tools */
async function loadState() {
  const s = await api('/api/state' + convQuery());
  $('state').textContent = 'scope: ' + (s.scope || '(none)');
  if (s.scope && !$('scope').value) $('scope').value = s.scope;
  renderWorkingSet(s.active_docs || []);
  excludedSet = new Set(s.excluded_docs || []);
}

async function useFolder(btn) {
  const path = $('scope').value.trim();
  if (!path) { status('Enter a folder path', true); return false; }
  return run(btn, 'Folder selected', async () => {
    await api('/api/scope', {
      method: 'POST',
      body: { path, conversation_id: conversationId },
    });
    // A new folder starts a fresh context for this conversation.
    $('msglist').innerHTML = '';
    renderWorkingSet([]);
    showEmpty();
    await loadState();
    await loadSources();
    await loadSession();
  });
}

async function syncIndex(btn) {
  return run(btn, 'Index updated', async () => {
    const r = await api('/api/sync', { method: 'POST', body: {} });
    $('out').innerHTML = preHtml(r);
    await loadSources();
  });
}

async function search(btn) {
  return run(btn, 'Search done', async () => {
    const list = await api('/api/search', { method: 'POST', body: { query: $('q').value, top_k: 5 } });
    if (!list || !list.length) { $('out').innerHTML = '<div class="muted">No results.</div>'; return; }
    $('out').innerHTML = list.map((r, i) => {
      const title = r.title || r.path || 'result';
      const score = typeof r.score === 'number' ? ' · ' + r.score.toFixed(3) : '';
      const snippet = (r.text || '').slice(0, 240);
      return `<div class="card"><div class="t">[${i + 1}] ${escapeHtml(title)}</div>` +
        `<div class="m">${escapeHtml(r.path || '')}${score}</div>` +
        `<div class="s">${escapeHtml(snippet)}</div></div>`;
    }).join('');
  });
}

async function suggest(btn) {
  return run(btn, 'Suggestion ready', async () => {
    const r = await api('/api/suggest', { method: 'POST', body: { query: $('q').value } });
    $('out').innerHTML = `<div class="md">${renderMarkdown(r.suggestion || '(no suggestion)')}</div>` +
      sourcesHtml(r.sources);
  });
}

async function createInsight(btn) {
  const path = $('path').value.trim();
  if (!path) { status('Enter a file path', true); return; }
  return run(btn, 'Insight ready', async () => {
    const r = await api('/api/insight', { method: 'POST', body: { path } });
    if (!r.job_id) { $('iout').innerHTML = preHtml(r); return; }
    $('iout').innerHTML = `<div class="muted">job ${escapeHtml(r.job_id)}: queued…</div>`;
    await new Promise((resolve) => {
      const timer = setInterval(async () => {
        try {
          const job = await api('/api/job/' + r.job_id);
          $('iout').innerHTML = `<div class="muted">job ${escapeHtml(r.job_id)}: ${escapeHtml(job.status)}</div>` +
            (job.status === 'done' ? preHtml(job.result ?? job)
              : job.status === 'error' ? `<div class="muted">${escapeHtml(job.error || '')}</div>` : '');
          if (job.status === 'done' || job.status === 'error') { clearInterval(timer); resolve(); }
        } catch (e) {
          clearInterval(timer);
          $('iout').innerHTML = `<div class="muted">Error: ${escapeHtml(e.message)}</div>`;
          resolve();
        }
      }, 1200);
    });
  });
}

async function listInsights(btn) {
  return run(btn, 'Insights loaded', async () => {
    const r = await api('/api/insights');
    $('iout').innerHTML = preHtml(r);
  });
}

// Build the Document Registry: one content-based "card" per file (background job).
async function buildCards(btn) {
  return run(btn, 'Building cards…', async () => {
    const r = await api('/api/registry', { method: 'POST', body: {} });
    if (!r.job_id) { status('Cards queued'); return; }
    await new Promise((resolve) => {
      const timer = setInterval(async () => {
        try {
          const job = await api('/api/job/' + r.job_id);
          if (job.status === 'done' || job.status === 'error') {
            clearInterval(timer);
            if (job.status === 'done') {
              const res = job.result || {};
              status(`Cards: ${(res.built || []).length} built, ${(res.skipped || []).length} cached`);
            } else {
              status(job.error || 'Card build failed', true);
            }
            resolve();
          }
        } catch (e) {
          clearInterval(timer);
          status(e.message, true);
          resolve();
        }
      }, 1200);
    });
  });
}

/* --------------------------------------------------------------- wiring */
function onEnter(input, handler) {
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); handler(); }
  });
}

function enableRipple(root) {
  root.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn || btn.disabled || btn.classList.contains('chip-x')) return;
    const rect = btn.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const span = document.createElement('span');
    span.className = 'ripple';
    span.style.width = span.style.height = `${size}px`;
    span.style.left = `${e.clientX - rect.left - size / 2}px`;
    span.style.top = `${e.clientY - rect.top - size / 2}px`;
    btn.appendChild(span);
    setTimeout(() => span.remove(), 520);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const btnUse = $('btnUse');
  const btnSync = $('btnSync');
  const btnAsk = $('btnAsk');
  const btnSearch = $('btnSearch');
  const btnInsight = $('btnInsight');
  const aq = $('aq');

  $('composer').addEventListener('submit', (e) => { e.preventDefault(); send(btnAsk); });
  aq.addEventListener('input', () => autoGrow(aq));
  aq.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(btnAsk); }
  });

  btnUse.addEventListener('click', () => useFolder(btnUse));
  btnSync.addEventListener('click', () => syncIndex(btnSync));
  $('btnClear').addEventListener('click', (e) => run(e.currentTarget, 'New chat', async () => {
    const r = await api('/api/conversation/new', { method: 'POST', body: {} });
    storeConversation(r.conversation_id);
    $('msglist').innerHTML = '';
    renderWorkingSet([]);
    showEmpty();
    await loadState();
  }));
  btnSearch.addEventListener('click', () => search(btnSearch));
  $('btnSuggest').addEventListener('click', (e) => suggest(e.currentTarget));
  btnInsight.addEventListener('click', () => createInsight(btnInsight));
  $('btnList').addEventListener('click', (e) => listInsights(e.currentTarget));

  $('workingset').addEventListener('click', (e) => {
    const btn = e.target.closest('.chip-x');
    if (btn) unpin(btn.dataset.path);
  });

  $('sources').addEventListener('change', onSourceToggle);
  $('btnAllSrc').addEventListener('click', (e) => run(e.currentTarget, 'All sources on', () => setAllSources(true)));
  $('btnNoneSrc').addEventListener('click', (e) => run(e.currentTarget, 'No sources selected', () => setAllSources(false)));
  $('btnCards').addEventListener('click', (e) => buildCards(e.currentTarget));

  // Enter works everywhere — no button needed.
  onEnter($('scope'), async () => { if (await useFolder(btnUse)) await syncIndex(btnSync); });
  onEnter($('q'), () => search(btnSearch));
  onEnter($('path'), () => createInsight(btnInsight));

  enableRipple(document);
  (async () => {
    try {
      await ensureConversation();
      await loadState();
      await loadSources();
      await loadSession();
    } catch (e) {
      status(e.message, true);
    }
  })();
});
