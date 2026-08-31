"use strict";
/* Reading runtime.  The markdown was rendered to HTML at build time; this file
   places it, tracks your work, and upgrades the interactive figures.  All state
   is per-browser localStorage under `mlss:*`, shared across every module file so
   the course home can add it up. */

const DATA = JSON.parse(document.getElementById('DATA').textContent);
const LECS = DATA.lectures;
const COURSE = DATA.course;
const MOD = DATA.slug;
const DAY = 864e5;

const store = {
  get(k, d){ try { const v = localStorage.getItem('mlss:' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v){ try { localStorage.setItem('mlss:' + k, JSON.stringify(v)); } catch (e) {} },
};
let DONE   = store.get('done', {});
let HL     = store.get('hl', {});
let NOTES  = store.get('notes', {});
let RATE   = store.get('rate', {});
let PHASE  = store.get('phase', {});
let REVIEW = store.get('review', {});
const save = () => { store.set('done', DONE); store.set('hl', HL); store.set('notes', NOTES);
  store.set('rate', RATE); store.set('phase', PHASE); store.set('review', REVIEW); store.set('srs', SRS); };

const $  = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const doc = $('#doc'), railL = $('#railL'), railRbody = $('#railRbody');
let view = 'reading', curLec = 0;

LECS.forEach(L => {
  // dividers ("Part A — Tools") carry no content — keep them out of progress math
  L._trackable = L.sections.filter(s => s.kind !== 'divider').map(s => s.id);
});
const lecDone = L => L._trackable.filter(id => DONE[id]).length;
const lecTotal = L => L._trackable.length;
function modCounts(slug){
  const c = COURSE.find(m => m.slug === slug) || { sections: 0 };
  const done = Object.keys(DONE).filter(k => k.startsWith(slug + '-') && DONE[k]).length;
  return { done, total: c.sections };
}

/* ── header ── */
function renderHeader(){
  const c = modCounts(MOD);
  const pct = c.total ? Math.round(c.done / c.total * 100) : 0;
  $('#mprogFill').style.width = pct + '%';
  $('#mprogPct').textContent = pct + '%';
  const due = srsCounts().due;
  const dot = $('#dueDot');
  dot.hidden = !due;
  dot.textContent = due > 99 ? '99+' : due;
}
function renderViews(){
  $('#views').innerHTML = Object.entries(VIEWS).map(([k, v]) =>
    `<button class="view-t" role="tab" data-view="${k}" aria-selected="${view === k}">${v.label}${v.count && v.count() ? ` <span style="opacity:.55">${v.count()}</span>` : ''}</button>`
  ).join('');
}

/* ── left rail: the course spine ── */
function renderLeft(){
  let h = `<div class="rl-h">This module</div><div class="spine-mod">${esc(DATA.name)}</div>`;
  LECS.forEach((L, ix) => {
    const tot = lecTotal(L), dn = lecDone(L);
    const pct = tot ? Math.round(dn / tot * 100) : 0;
    const label = L.kind === 'practicum' ? 'Practicum' : (L.spineLabel || 'Part ' + (+L.num));
    h += `<div class="lec">
      <button class="lec-b ${ix === curLec && view === 'reading' ? 'on' : ''}" data-lec="${ix}">
        <span class="lec-n">${L.kind === 'practicum' ? '★' : (+L.num)}</span>
        <span class="lec-lab">${esc(label)}</span>
        <span class="lec-bar"><i style="width:${pct}%"></i></span>
      </button>
      <ul class="secs ${ix === curLec && view === 'reading' ? 'open' : ''}" data-for="${ix}">
        ${(() => { let lastPart = null; return L.sections.map(s => {
          const partHdr = s.part && s.part !== lastPart ? (lastPart = s.part, `<li class="spine-part">${esc(s.part.replace(/\s*—.*/, '').replace(/^Part\s*/, 'Part '))}</li>`) : '';
          return partHdr + `<li><a href="#${s.id}" data-go="${s.id}" data-lec="${ix}" data-kind="${s.kind || 'content'}" class="${DONE[s.id] ? 'done' : ''}">
          <span class="n">${s.num || ''}</span><span>${esc(s.title)}</span></a></li>`;
        }).join(''); })()}
      </ul></div>`;
  });
  h += `<div class="rl-h">All modules</div><nav class="othermod">`;
  COURSE.forEach((m, i) => {
    const c = modCounts(m.slug);
    h += `<a href="./${m.slug}.html" class="${m.slug === MOD ? 'here' : ''}" ${m.slug === MOD ? 'aria-current="page"' : ''}>
      <span class="m-n">${i + 1}</span><span style="flex:1">${esc(m.name)}</span>
      <span class="m-p">${c.total ? Math.round(c.done / c.total * 100) : 0}%</span></a>`;
  });
  railL.innerHTML = h + `</nav>`;
}

/* ── right rail: your work on this page ── */
const PHASES = [
  ['first', 'First pass', 'read it through', null],
  ['second', 'Second pass', 'work every 🧪 on paper', 'view'],
  ['interview', 'Interview prep', 'threads · derivations · probes', 'view'],
];
function renderRight(){
  if (view !== 'reading') { railRbody.innerHTML = ''; return; }
  const L = LECS[curLec];
  const ph = PHASE[L.id] || {};
  const firstPct = lecTotal(L) ? Math.round(lecDone(L) / lecTotal(L) * 100) : 0;
  let h = `<div class="rl-h">Where you are in this lecture</div><div class="phase">` +
    PHASES.map(([k, label, hint, act]) => `<button class="phase-b ${ph[k] ? 'done' : ''}" data-phase="${k}" data-act="${act || 'toggle'}">
      <span class="box"></span>
      <span><b>${label}</b><small>${hint}</small></span>
      <span class="go">${act ? '→' : (k === 'first' ? firstPct + '%' : '')}</span>
    </button>`).join('') + `</div>`;

  const hls = Object.entries(HL).filter(([, v]) => v.mod === MOD && v.lec === L.id);
  h += `<div class="rl-h">Highlights</div>` + (hls.length
    ? `<ul class="rlist">` + hls.map(([k, v]) => `<li data-c="${v.c}">
        <button data-go="${v.sec}" data-hl="${k}">${esc(v.text.slice(0, 88))}${v.text.length > 88 ? '…' : ''}</button></li>`).join('') + `</ul>`
    : `<p class="empty">Select text, pick a colour. Terracotta (3) = “come back to this”.</p>`);

  const ns = Object.entries(NOTES).filter(([, v]) => v.mod === MOD && v.lec === L.id);
  h += `<div class="rl-h">Notes</div>` + (ns.length
    ? `<ul class="rlist">` + ns.map(([, v]) => `<li><button data-go="${v.sec}">${esc(v.text.slice(0, 88))}</button></li>`).join('') + `</ul>`
    : `<p class="empty">Hover a paragraph, press +.</p>`);
  railRbody.innerHTML = h;
}

/* ── show a lecture ── */
function showLecture(ix, anchor){
  curLec = ix; view = 'reading';
  const L = LECS[ix];
  doc.className = 'doc';
  const head = `<header class="lecture-head">
    <div class="kicker">${esc(DATA.name)}${L.kind === 'practicum' ? ' · Practicum' : ' · Part ' + (+L.num)}</div>
    <h1>${esc(L.subtitle || L.h1 || L.title)}</h1>
    <p class="meta">Lecture ${L.num}${L.instructor ? ' · ' + esc(L.instructor) : ''} · ${L.sections.length} sections · ~${lecHours(L)} first pass</p>
  </header>`;
  doc.innerHTML = head + L.html;
  applyState();
  upgradeInteractives();
  renderAll();
  store.set('last', { mod: MOD, lec: ix, sec: anchor || (L.sections[0] && L.sections[0].id) });
  history.replaceState(null, '', '#' + (anchor || L.id));
  if (anchor) { const el = document.getElementById(anchor); if (el) el.scrollIntoView(); }
  else window.scrollTo(0, 0);
}
const lecHours = L => {
  const mins = L.sections.reduce((a, s) => a + (s.mins || 0), 0);
  return mins >= 90 ? (mins / 60).toFixed(1).replace(/\.0$/, '') + ' h' : mins + ' min';
};

function showView(v){
  if (v === 'reading') { view = 'reading'; showLecture(curLec); return; }
  view = v;
  doc.className = 'doc';
  doc.innerHTML = `<div class="view on">${VIEWS[v].render()}</div>`;
  applyRatingsIn(doc);
  renderAll();
  window.scrollTo(0, 0);
}
function renderAll(){ renderHeader(); renderViews(); renderLeft(); renderRight(); }

/* ── phase-driven filtered views ── */
function showPhaseView(kind){
  const L = LECS[curLec];
  const t = document.createElement('div'); t.innerHTML = L.html;
  view = 'phase:' + kind;
  doc.className = 'doc';
  let body;
  if (kind === 'second') {
    const blocks = [...t.querySelectorAll('h3, h4')].filter(e => /🧪/.test(e.textContent));
    body = `<h1>Second pass — ${esc(L.spineLabel || 'this lecture')}</h1>
      <p class="phase-view-h">Every 🧪 worked example in this lecture, its solution blanked.
      Work each on paper first, then reveal. This is where the learning actually happens.</p>`;
    blocks.forEach((e, i) => {
      let inner = ''; let n = e.nextElementSibling;
      while (n && !/^H[1-4]$/.test(n.tagName) && inner.length < 14000) { inner += n.outerHTML; n = n.nextElementSibling; }
      body += `<div class="qcard"><h3 style="margin-top:0;display:flex;gap:10px;align-items:baseline">${e.innerHTML}
        <button class="tryfirst" data-reveal="${i}">Reveal solution</button></h3>
        <div class="masked" id="pv-${i}">${inner}</div></div>`;
    });
    if (!blocks.length) body += `<p class="view-lead">This lecture has no 🧪 blocks — it's mostly conceptual. A second read of the derivations is the equivalent.</p>`;
  } else {
    const wanted = [...t.querySelectorAll('section')].filter(s =>
      /^(interview prep|putting it together|check yourself)/i.test(s.dataset.title || ''));
    body = `<h1>Interview prep — ${esc(L.spineLabel || 'this lecture')}</h1>
      <p class="phase-view-h">The <em>Putting it together</em> threads, the whiteboard derivations, the
      depth-probe table and the check-yourself questions — the “before an interview” pass.</p>`;
    wanted.forEach(s => { body += s.outerHTML; });
    if (!wanted.length) body += `<p class="view-lead">No interview-prep section found in this lecture.</p>`;
  }
  const nav = `<div class="phase-nav">
    <button data-view="reading">‹ Back to reading</button>
    <button data-phase="second" data-act="view" ${kind === 'second' ? 'aria-current="true"' : ''}>Second pass</button>
    <button data-phase="interview" data-act="view" ${kind === 'interview' ? 'aria-current="true"' : ''}>Interview prep</button>
    <button data-phase="${kind}" data-act="toggle" class="phase-mark">Mark ${kind === 'second' ? 'second pass' : 'interview prep'} done</button>
  </div>`;
  doc.innerHTML = `<div class="view on">${nav}${body}</div>`;
  applyRatingsIn(doc);
  markDeadXrefs();
  renderAll();
  window.scrollTo(0, 0);
}

/* ── state into the current DOM ── */
function applyState(){
  $$('.sec-check').forEach(b => b.classList.toggle('on', !!DONE[b.dataset.sec]));
  Object.entries(HL).forEach(([k, v]) => {
    if (v.mod !== MOD || v.lec !== LECS[curLec].id) return;
    const sec = document.getElementById(v.sec);
    if (sec) applyHL(sec, v.start != null ? v.start : findOffset(sec, v.text), (v.len != null ? v.len : v.text.length), v.c, k);
  });
  Object.entries(NOTES).forEach(([k, v]) => {
    if (v.mod !== MOD || v.lec !== LECS[curLec].id) return;
    const sec = document.getElementById(v.sec); if (!sec) return;
    const p = sec.querySelectorAll('p')[v.p];
    if (p && !p.nextElementSibling?.classList.contains('note')) {
      const d = document.createElement('div'); d.className = 'note'; d.dataset.k = k;
      d.innerHTML = `<button class="note-x" title="Delete">✕</button>${esc(v.text)}`;
      p.after(d);
    }
  });
  $$('section p').forEach((p, ix) => {
    if (p.querySelector('.addn')) return;
    const b = document.createElement('button'); b.className = 'addn'; b.textContent = '+';
    b.title = 'Add a note'; b.dataset.p = ix; p.prepend(b);
  });
  $$('h3, h4').forEach(hh => {
    if (!/🧪/.test(hh.textContent) || hh.querySelector('.tryfirst')) return;
    hh.style.display = 'flex'; hh.style.alignItems = 'baseline'; hh.style.gap = '10px';
    const b = document.createElement('button'); b.className = 'tryfirst'; b.textContent = 'Try it first';
    hh.appendChild(b);
  });
  applyRatingsIn(doc);
  markDeadXrefs();
  buildSearch();
}
function applyRatingsIn(root){
  const byFront = {};
  cardPool().forEach(c => { byFront[c.front.slice(0, 48)] = c; });
  root.querySelectorAll('.recall').forEach(rc => {
    const front = (rc.querySelector('.recall-t')?.textContent || '').replace(/\s+/g, ' ').trim();
    const card = byFront[front.slice(0, 48)];
    if (card) {                                   // this recall card is a spaced-review card
      rc.dataset.card = card.key;
      const s = SRS[card.key];
      if (s) rc.dataset.rated = s.box >= 3 ? '1' : s.box >= 1 ? '2' : '3';
    } else {
      const key = recallKey(rc);
      if (RATE[key]) rc.dataset.rated = RATE[key];
    }
    if (rc.dataset.rated)
      rc.querySelectorAll('.rate-b').forEach(b => b.classList.toggle('on', b.dataset.r === rc.dataset.rated));
  });
}
const recallKey = rc => `${MOD}::${LECS[curLec] ? LECS[curLec].id : ''}::${rc.querySelector('.recall-t')?.textContent.trim().slice(0, 90)}`;

function markDeadXrefs(){
  $$('.xref').forEach(a => {
    if (a.dataset.xrefMod) {
      const m = COURSE.find(x => x.slug === a.dataset.xrefMod);
      a.title = m ? 'Open ' + m.name : '';
      a.classList.remove('dead');
      return;
    }
    const lec = a.dataset.xrefLec;
    if (!lec) return;
    if (lec.startsWith(MOD)) a.classList.remove('dead');
    else { a.classList.add('dead'); a.title = 'In another module'; }
  });
}

/* Highlights span multiple text nodes (a <strong> then plain text, etc.), so we
   store a character offset into the section's text and, on apply, wrap every
   text node the range touches — going back-to-front so earlier offsets hold. */
function sectionTextNodes(sec){
  const w = document.createTreeWalker(sec, NodeFilter.SHOW_TEXT);
  const out = []; let n;
  while ((n = w.nextNode())) {
    if (n.parentElement.closest('script,style,.addn,.note')) continue;
    out.push(n);
  }
  return out;
}
function offsetOfRange(sec, range){
  const nodes = sectionTextNodes(sec);
  const sc = range.startContainer, so = range.startOffset;
  let acc = 0;
  if (sc.nodeType === 3){
    for (const node of nodes){ if (node === sc) return acc + so; acc += node.nodeValue.length; }
    return acc;
  }
  const boundary = sc.childNodes ? sc.childNodes[so] : null;   // point is just before this child
  for (const node of nodes){
    if (boundary && (node === boundary || boundary.contains?.(node))) return acc;
    if (boundary && (node.compareDocumentPosition(boundary) & Node.DOCUMENT_POSITION_PRECEDING)) return acc;
    acc += node.nodeValue.length;
  }
  return acc;
}
function findOffset(sec, text){ return sec.textContent.indexOf(text); }
function applyHL(sec, start, len, c, k){
  if (!sec || start < 0 || sec.querySelector(`.hl[data-k="${k}"]`)) return;
  const end = start + len;
  const nodes = sectionTextNodes(sec);
  let acc = 0; const hits = [];
  for (const node of nodes){
    const s = acc, e = acc + node.nodeValue.length; acc = e;
    if (e <= start || s >= end) continue;
    hits.push({ node, a: Math.max(0, start - s), b: Math.min(node.nodeValue.length, end - s) });
  }
  for (let i = hits.length - 1; i >= 0; i--){
    const { node, a, b } = hits[i];
    if (a >= b || node.parentElement.closest('.hl')) continue;
    const r = document.createRange(); r.setStart(node, a); r.setEnd(node, b);
    const span = document.createElement('span'); span.className = 'hl'; span.dataset.c = c; span.dataset.k = k;
    try { r.surroundContents(span); } catch (e) {}
  }
}
function removeHL(k){
  document.querySelectorAll(`.hl[data-k="${k}"]`).forEach(el => {
    const p = el.parentNode; while (el.firstChild) p.insertBefore(el.firstChild, el); p.removeChild(el); p.normalize();
  });
}

/* ── module views ── */
const VIEWS = {
  reading: { label: 'Reading' },
  review: {
    label: 'Review',
    count(){ return srsCounts().due || ''; },
    render(){
      const cc = srsCounts();
      const now = Date.now();
      // upcoming: cards scheduled in the next 14 days, bucketed
      const buckets = {};
      Object.values(SRS).forEach(s => {
        if (s.due <= now) return;
        const d = Math.ceil((s.due - now) / DAY);
        if (d <= 14) buckets[d] = (buckets[d] || 0) + 1;
      });
      const bars = Array.from({ length: 14 }, (_, i) => i + 1).map(d => {
        const n = buckets[d] || 0;
        return `<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:3px">
          <div style="width:100%;height:${Math.min(60, n * 6)}px;background:var(--accent-dim);border-radius:2px 2px 0 0"></div>
          <span style="font-size:9px;color:var(--ink-3)">${d}</span></div>`;
      }).join('');
      return `<h1>Spaced review</h1>
        <p class="view-lead">Every glossary term and Check-yourself / Interview question in this module is a card.
        Cards you know move to 3-, 8-, 21-, then 60-day intervals; ones you miss reset. Retrieval + spacing is the
        part that actually sticks.</p>
        <div style="display:flex;gap:26px;font-family:var(--font-ui);font-size:12.5px;color:var(--ink-3);margin-bottom:22px;flex-wrap:wrap">
          <span><b style="display:block;font-size:22px;color:var(--accent);font-variant-numeric:tabular-nums">${cc.due}</b>due now</span>
          <span><b style="display:block;font-size:22px;color:var(--ink);font-variant-numeric:tabular-nums">${cc.fresh}</b>new (lecture started)</span>
          <span><b style="display:block;font-size:22px;color:var(--ink);font-variant-numeric:tabular-nums">${cc.mature}</b>known well</span>
          <span><b style="display:block;font-size:22px;color:var(--ink-3);font-variant-numeric:tabular-nums">${cc.total}</b>cards total</span>
        </div>
        <button class="rev-reveal" id="startReview" style="max-width:280px;padding:13px;font-size:14px;${cc.due || cc.fresh ? '' : 'opacity:.5;pointer-events:none'}">
          Start review${cc.due || cc.fresh ? ` &nbsp;·&nbsp; ${Math.min(cc.due + Math.min(cc.fresh, 12), 40)} cards` : ' — nothing due'}</button>
        <h3 style="margin-top:32px">Coming up (next 14 days)</h3>
        <div style="display:flex;align-items:flex-end;gap:3px;height:72px;margin:12px 0 8px">${bars}</div>
        <button class="rate-b" id="exportAnki" style="margin-top:20px">Export ${cc.total} cards for Anki</button>`;
    },
  },
  weak: {
    label: 'Weak spots',
    count(){ return Object.values(HL).filter(v => v.mod === MOD && v.c === '3').length || ''; },
    render(){
      const items = Object.entries(HL).filter(([, v]) => v.mod === MOD && v.c === '3');
      if (!items.length) return `<h1>Weak spots</h1><p class="view-lead">Highlight anything you don't yet
        understand in <strong style="color:var(--hl-3)">terracotta</strong> (select text, press <span class="kbd">3</span>).
        It gathers here — the most useful page you'll build in this module.</p>`;
      return `<h1>Weak spots</h1><p class="view-lead">${items.length} passage${items.length > 1 ? 's' : ''} marked not-yet-understood.</p>` +
        items.map(([, v]) => { const L = LECS.find(l => l.id === v.lec);
          return `<div class="qcard r3"><div class="from">${esc(L ? (L.spineLabel || 'L' + L.num) : '')}</div>
            <div style="font-size:16px">${esc(v.text)}</div></div>`; }).join('');
    },
  },
  gloss: {
    label: 'Glossary',
    render(){
      const rows = [];
      LECS.forEach(L => {
        const t = document.createElement('div'); t.innerHTML = L.html;
        [...t.querySelectorAll('section')].forEach(sec => {
          if (!/glossary/i.test(sec.dataset.title || '')) return;
          sec.querySelectorAll('table').forEach(tb => {
            const th = tb.querySelector('th');
            if (!th || !/term/i.test(th.textContent)) return;
            tb.querySelectorAll('tbody tr').forEach(tr => {
              const c = [...tr.children].map(x => x.innerHTML.trim());
              if (c.length >= 2) rows.push({ t: c[0], d: c.slice(1).join(' — '), from: L.spineLabel || 'L' + L.num });
            });
          });
        });
      });
      rows.sort((a, b) => a.t.replace(/\W/g, '').toLowerCase().localeCompare(b.t.replace(/\W/g, '').toLowerCase()));
      return `<h1>Glossary</h1><p class="view-lead">${rows.length} terms across the module, merged and sorted.</p>
        <input id="gf" placeholder="Filter terms…">
        <div class="table-wrap"><table><thead><tr><th>Term</th><th>Definition</th><th>From</th></tr></thead><tbody>` +
        rows.map(r => `<tr><td>${r.t}</td><td>${r.d}</td><td style="color:var(--ink-3);white-space:nowrap">${r.from}</td></tr>`).join('') +
        `</tbody></table></div>`;
    },
  },
  deriv: {
    label: 'Derivations',
    render(){
      let h = `<h1>Derivations</h1><p class="view-lead">Every 🧪 worked example and whiteboard-ready
        derivation in the module, one page — the fastest revision pass.</p>`;
      LECS.forEach(L => {
        const t = document.createElement('div'); t.innerHTML = L.html;
        const hits = [...t.querySelectorAll('h3, h4')].filter(e => /🧪|worked example|derivation|derive/i.test(e.textContent));
        if (!hits.length) return;
        h += `<h2 style="border:0;padding:0;margin:1.6em 0 .6em"><span class="sec-t">${esc(L.spineLabel || 'L' + L.num)}</span></h2>`;
        hits.forEach(e => {
          let body = ''; let n = e.nextElementSibling;
          while (n && !/^H[1-4]$/.test(n.tagName) && body.length < 12000) { body += n.outerHTML; n = n.nextElementSibling; }
          h += `<div class="qcard"><h3 style="margin-top:0">${e.innerHTML}</h3>${body}</div>`;
        });
      });
      return h;
    },
  },
};
/* ══════════ spaced review ══════════════════════════════════════════
   Every glossary term + Check-yourself / Interview question is a card.
   Leitner boxes 0–4 → 1, 3, 8, 21, 60-day intervals.  A card only enters
   the deck once its lecture has been started (a section marked done, or a
   first pass logged). */
const SRS_INTERVALS = [1, 3, 8, 21, 60];
let SRS = store.get('srs', {});
const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.random() * (i + 1) | 0; [a[i], a[j]] = [a[j], a[i]]; } return a; };
const cardKey = (lecId, type, front) => `${MOD}::${lecId}::${type}::${front.replace(/\s+/g, ' ').trim().slice(0, 64)}`;
const lecStarted = lecId => (PHASE[lecId] || {}).first || Object.keys(DONE).some(k => k.startsWith(lecId + '-') && DONE[k]);

let CARD_POOL = null;
function cardPool(){
  if (CARD_POOL) return CARD_POOL;
  const pool = [];
  const push = (L, type, front, back) => {
    front = String(front || '').replace(/\s+/g, ' ').trim();
    back = String(back || '').trim();
    if (front.length < 2 || !back) return;
    pool.push({ key: cardKey(L.id, type, front), lec: L.id, lecNum: L.num, type, front, back });
  };
  LECS.forEach(L => {
    const t = document.createElement('div'); t.innerHTML = L.html;
    t.querySelectorAll('section').forEach(sec => {
      const title = (sec.dataset.title || '').toLowerCase();
      if (/glossary/.test(title)){
        sec.querySelectorAll('table').forEach(tb => {
          const th = tb.querySelector('th');
          if (th && /term/i.test(th.textContent))
            tb.querySelectorAll('tbody tr').forEach(tr => {
              const c = [...tr.children];
              if (c.length >= 2) push(L, 'glossary', c[0].textContent, c.slice(1).map(x => x.innerHTML).join(' — '));
            });
        });
        sec.querySelectorAll('li').forEach(li => {
          const s = li.querySelector('strong');
          if (!s || !li.textContent.trim().startsWith(s.textContent.trim())) return;
          const i = li.innerHTML.search(/\s[—–-]\s/);
          if (i < 0) return;
          push(L, 'glossary', s.textContent, li.innerHTML.slice(i).replace(/^\s*[—–-]\s*/, ''));
        });
      }
      if (/check yourself/.test(title)){
        const ans = [];
        sec.querySelectorAll('.recall').forEach(rc => {
          rc.querySelector('.recall-a .rate')?.remove();
          [...rc.querySelectorAll('.recall-a ol > li')].forEach((li, i) => { ans[i] = li.innerHTML; });
        });
        const ol = [...sec.querySelectorAll('ol')].find(o => !o.closest('.recall'));
        if (ol) [...ol.children].forEach((li, i) => {
          const sref = (li.textContent.match(/\(§[^)]+\)\s*$/) || [''])[0].replace(/[()]/g, '');
          push(L, 'check', li.innerHTML.replace(/\s*\(§[^)]+\)\s*$/, ''),
            ans[i] || `Recall it from memory, then check ${sref || 'the section'}.`);
        });
      }
      if (/interview prep/.test(title)){
        sec.querySelectorAll('.recall').forEach(rc => {
          const q = rc.querySelector('.recall-t')?.textContent;
          const a = rc.querySelector('.recall-a')?.cloneNode(true);
          a?.querySelector('.rate')?.remove();
          if (q && a) push(L, 'interview', q, a.innerHTML);
        });
      }
    });
  });
  // de-dup by key (a term can appear in two lecture glossaries)
  const seen = new Set();
  CARD_POOL = pool.filter(c => !seen.has(c.key) && seen.add(c.key));
  return CARD_POOL;
}

function srsCounts(){
  const now = Date.now(); let due = 0, learning = 0, mature = 0, fresh = 0;
  for (const c of cardPool()){
    const s = SRS[c.key];
    if (!s) { if (lecStarted(c.lec)) fresh++; continue; }
    if (s.due <= now) due++;
    if (s.box >= 3) mature++; else learning++;
  }
  const c = { due, learning, mature, fresh, total: cardPool().length };
  // leave a summary the course home can read without loading this module's content
  try { const all = store.get('cardstats', {}); all[MOD] = c; store.set('cardstats', all); } catch (e) {}
  return c;
}
function reviewQueue(limit = 40){
  const now = Date.now(); const due = [], fresh = [];
  for (const c of cardPool()){
    const s = SRS[c.key];
    if (s) { if (s.due <= now) due.push(c); }
    else if (lecStarted(c.lec)) fresh.push(c);
  }
  return shuffle(due).slice(0, limit).concat(shuffle(fresh).slice(0, Math.max(0, Math.min(15, limit - due.length))));
}

/* study log — one entry per section completed, per day; the home page reads
   this to show honest weekly pace.  Keeps ~120 days. */
function logStudy(secId){
  const L = LECS.find(l => l._trackable && l._trackable.includes(secId));
  const mins = L ? (L.sections.find(s => s.id === secId) || {}).mins || 6 : 6;
  const day = new Date().toISOString().slice(0, 10);
  const log = store.get('log', {});
  log[day] = (log[day] || 0) + mins;
  const cut = Date.now() - 120 * DAY;
  for (const d in log) if (new Date(d).getTime() < cut) delete log[d];
  store.set('log', log);
}
function srsSchedule(key, grade){   // 0 = no idea, 1 = shaky, 2 = got it
  const now = Date.now();
  const s = SRS[key] || { box: 0, reps: 0 };
  if (grade === 2) s.box = Math.min(4, s.box + 1);
  else if (grade === 0) s.box = 0;
  s.reps = (s.reps || 0) + 1;
  s.due = now + (grade === 0 ? 6e4 : grade === 1 ? DAY : SRS_INTERVALS[s.box] * DAY);
  s.seen = now;
  SRS[key] = s;
  store.set('srs', SRS);
}

/* session driver */
let REV = null;
const revEl = id => document.getElementById(id);
function openReview(){
  const q = reviewQueue();
  if (!q.length) { flash('Nothing due — new cards unlock as you start each lecture.'); return; }
  REV = { queue: q, total: q.length, done: 0 };
  revEl('rev').classList.add('on');
  document.documentElement.style.overflow = 'hidden';
  revShow();
}
function revShow(){
  if (!REV.queue.length) return revFinish();
  const c = REV.queue[0];
  const L = LECS.find(x => x.id === c.lec);
  const kind = c.type === 'glossary' ? 'Define' : c.type === 'check' ? 'Check yourself' : 'Interview';
  revEl('revCount').textContent = `${REV.done} / ${REV.total}`;
  revEl('revProg').style.width = (REV.done / REV.total * 100) + '%';
  revEl('revNew').textContent = SRS[c.key] ? '' : 'new';
  revEl('revCard').innerHTML =
    `<div class="rev-kind">${kind}<span>· ${esc(L ? L.spineLabel : 'L' + c.lecNum)}</span></div>
     <div class="rev-front">${c.front}</div>
     <div class="rev-back hidden" id="revBack">${c.back}</div>`;
  revEl('revActions').innerHTML =
    `<button class="rev-reveal" id="revReveal">Show answer &nbsp;<span class="kbd">Space</span></button>`;
}
function revReveal(){
  const b = revEl('revBack'); if (!b || !b.classList.contains('hidden')) return;
  b.classList.remove('hidden');
  const s = SRS[REV.queue[0].key];
  const box = s ? Math.min(4, s.box + 1) : 0;
  revEl('revActions').innerHTML =
    `<button class="rev-grade" data-g="0">No idea<small>again now</small></button>
     <button class="rev-grade" data-g="1">Shaky<small>tomorrow</small></button>
     <button class="rev-grade" data-g="2">Got it<small>${SRS_INTERVALS[box]} days</small></button>`;
}
function revGrade(g){
  const c = REV.queue.shift();
  srsSchedule(c.key, g);
  if (g === 0) { REV.queue.push(c); REV.total++; }
  else REV.done++;
  renderHeader();
  revShow();
}
function revFinish(){
  const cc = srsCounts();
  revEl('revNew').textContent = '';
  revEl('revProg').style.width = '100%';
  revEl('revCount').textContent = `${REV.done} / ${REV.total}`;
  revEl('revCard').innerHTML = `<div class="rev-done">
    <div class="big">${REV.done}</div><h2>reviewed</h2>
    <p>${cc.due} still due today · ${cc.mature} card${cc.mature === 1 ? '' : 's'} you now know well ·
    ${cc.fresh} new card${cc.fresh === 1 ? '' : 's'} waiting for you to start their lecture.</p></div>`;
  revEl('revActions').innerHTML = `<button class="rev-reveal" id="revClose">Done</button>`;
}
function closeReview(){
  revEl('rev').classList.remove('on');
  document.documentElement.style.overflow = '';
  REV = null;
  if (view !== 'reading') showView(view); else renderHeader();
}
let flashT;
function flash(msg){
  let el = revEl('flashMsg');
  if (!el){ el = document.createElement('div'); el.id = 'flashMsg';
    el.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:90;background:var(--surface);border:1px solid var(--rule-2);border-radius:9px;padding:10px 16px;font-family:var(--font-ui);font-size:13px;color:var(--ink-2);box-shadow:var(--shadow)';
    document.body.appendChild(el); }
  el.textContent = msg; el.style.opacity = '1';
  clearTimeout(flashT); flashT = setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity .4s'; }, 2600);
}

/* Anki export — tab-separated, math kept as \(TeX\) for the MathJax add-on */
function ankiText(html){
  return String(html)
    .replace(/<math[^>]*data-tex="([^"]*)"[^>]*>[\s\S]*?<\/math>/g,
      (_, tex) => ' \\(' + tex.replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>') + '\\) ')
    .replace(/<[^>]+>/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/\s+/g, ' ').trim();
}
function exportAnki(){
  const rows = cardPool().map(c =>
    [ankiText(c.front), ankiText(c.back), `mlss::${MOD.replace(/-/g, '_')}::${c.type}`]
      .map(x => '"' + x.replace(/"/g, '""') + '"').join('\t'));
  const blob = new Blob([`#separator:tab\n#html:false\n#columns:Front\tBack\tTags\n#tags column:3\n${rows.join('\n')}\n`],
    { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `mlss-${MOD}.txt`;
  document.body.appendChild(a); a.click(); a.remove();
  flash(`${cardPool().length} cards → mlss-${MOD}.txt (Anki: File ▸ Import)`);
}

/* ── search ── */
let SIDX = [];
function buildSearch(){
  if (SIDX.length) return;
  LECS.forEach(L => {
    const t = document.createElement('div'); t.innerHTML = L.html;
    t.querySelectorAll('section').forEach(s => {
      SIDX.push({ lec: L.id, ls: L.spineLabel || 'L' + L.num, id: s.id,
        t: s.dataset.title || '', txt: s.textContent.replace(/\s+/g, ' ') });
    });
  });
}
function doSearch(q){
  const sr = $('#sr');
  if (!q.trim()) { sr.innerHTML = ''; return; }
  const rx = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'ig');
  const res = [];
  SIDX.forEach(s => {
    const inT = (s.t.match(rx) || []).length, inB = (s.txt.match(rx) || []).length;
    if (!inT && !inB) return;
    const i = s.txt.search(rx);
    res.push({ s, score: inT * 25 + inB, snip: i < 0 ? '' : s.txt.slice(Math.max(0, i - 55), i + 95) });
  });
  COURSE.forEach(m => { if (m.slug === MOD) return;
    (m.index || []).forEach(l => { if (rx.test(l.short) || rx.test(m.name))
      res.push({ s: { id: '', lec: '', ls: m.name, t: 'L' + l.num + ' · ' + l.short, href: `./${m.slug}.html` }, score: 4, snip: '', cross: m.name }); });
  });
  res.sort((a, b) => b.score - a.score);
  sr.innerHTML = res.slice(0, 25).map((r, ix) => `<button class="sri ${ix === 0 ? 'sel' : ''}"
    data-go="${r.s.id}" data-lec="${r.s.lec}" ${r.s.href ? `data-href="${r.s.href}"` : ''}>
    ${r.cross ? `<span class="scope">${esc(r.cross)}</span> ` : ''}<b>${esc(r.s.t)}</b>
    <span>${r.cross ? 'open module' : esc(r.s.ls) + ' · ' + esc(r.snip).replace(rx, m => `<mark>${m}</mark>`)}</span></button>`).join('')
    || `<p class="empty" style="padding:14px">No matches.</p>`;
}
function openSearch(){ const sd = $('#sd'); sd.classList.add('on'); const q = $('#sq'); q.value = ''; q.focus(); $('#sr').innerHTML = ''; }

/* ── navigation ── */
function goto(id, lecIx){
  $('#sd').classList.remove('on');
  if (lecIx != null && lecIx !== '' && +lecIx !== curLec) { showLecture(+lecIx, id); return; }
  if (view !== 'reading') { showLecture(curLec, id); return; }
  const el = document.getElementById(id);
  if (el) { el.scrollIntoView(); history.replaceState(null, '', '#' + id); }
}

/* ── events ── */
document.addEventListener('click', e => {
  const t = e.target;
  const vt = t.closest('[data-view]'); if (vt) { e.preventDefault(); showView(vt.dataset.view); return; }
  const lb = t.closest('.lec-b'); if (lb) {
    const ix = +lb.dataset.lec;
    if (ix === curLec && view === 'reading') lb.nextElementSibling.classList.toggle('open');
    else showLecture(ix);
    return;
  }
  const go = t.closest('[data-go]'); if (go) { e.preventDefault(); goto(go.dataset.go, go.dataset.lec); return; }
  const ph = t.closest('[data-phase]'); if (ph) {
    if (ph.dataset.act === 'view') { showPhaseView(ph.dataset.phase); return; }
    const L = LECS[curLec].id; PHASE[L] = PHASE[L] || {};
    PHASE[L][ph.dataset.phase] = !PHASE[L][ph.dataset.phase];
    save();
    if (view.startsWith('phase:')) showPhaseView(view.slice(6)); else renderRight();
    return;
  }
  const rev = t.closest('[data-reveal]'); if (rev) {
    const box = document.getElementById('pv-' + rev.dataset.reveal);
    if (box) { box.classList.toggle('masked'); rev.textContent = box.classList.contains('masked') ? 'Reveal solution' : 'Hide'; }
    return;
  }
  const chk = t.closest('.sec-check'); if (chk) {
    const s = chk.dataset.sec; DONE[s] = !DONE[s]; if (!DONE[s]) delete DONE[s]; else logStudy(s);
    save(); chk.classList.toggle('on', !!DONE[s]); renderHeader(); renderLeft(); renderRight(); return;
  }
  const rq = t.closest('.recall-q'); if (rq) {
    const rc = rq.closest('.recall'); const a = rc.querySelector('.recall-a');
    const open = a.hidden; a.hidden = !open; rc.toggleAttribute('data-open', open);
    rq.querySelector('.recall-x').textContent = open ? 'Hide' : 'Reveal';
    return;
  }
  const rb = t.closest('.rate-b'); if (rb && rb.dataset.r) {
    const rc = rb.closest('.recall'); if (!rc) return;
    const v = +rb.dataset.r;                       // 1 got it · 2 shaky · 3 no idea
    if (rc.dataset.card) {                         // a spaced-review card — schedule it
      srsSchedule(rc.dataset.card, 3 - v);
      const s = SRS[rc.dataset.card];
      rc.dataset.rated = s.box >= 3 ? '1' : s.box >= 1 ? '2' : '3';
      rc.querySelectorAll('.rate-b').forEach(b => b.classList.toggle('on', b.dataset.r === rc.dataset.rated));
      renderHeader(); renderViews();
      flash(v === 1 ? `Got it — next review in ${SRS_INTERVALS[s.box]} days` : v === 2 ? 'Back tomorrow' : 'Reset — back soon');
      return;
    }
    const key = recallKey(rc);
    if (RATE[key] === v) { delete RATE[key]; rc.removeAttribute('data-rated'); }
    else { RATE[key] = v; rc.dataset.rated = v; }
    rc.querySelectorAll('.rate-b').forEach(b => b.classList.toggle('on', RATE[key] && b.dataset.r === String(RATE[key])));
    save(); renderViews();
    return;
  }
  const tf = t.closest('.tryfirst'); if (tf && !tf.dataset.reveal) {
    const on = tf.textContent === 'Try it first';
    let n = tf.closest('h3,h4').nextElementSibling;
    while (n && !/^H[1-4]$/.test(n.tagName)) { if (on ? !n.dataset.boxed : true) n.classList.toggle('masked', on); n = n.nextElementSibling; }
    tf.textContent = on ? 'Show solution' : 'Try it first';
    return;
  }
  const an = t.closest('.addn'); if (an) {
    const txt = prompt('Note:'); if (!txt) return;
    const sec = an.closest('section'); const ps = [...sec.querySelectorAll('p')];
    NOTES['n' + Date.now()] = { mod: MOD, lec: LECS[curLec].id, sec: sec.id, p: ps.indexOf(an.parentElement), text: txt };
    save(); showLecture(curLec, sec.id); return;
  }
  const nx = t.closest('.note-x'); if (nx) {
    const d = nx.closest('.note'); delete NOTES[d.dataset.k]; save(); d.remove(); renderRight(); return;
  }
  const xr = t.closest('.xref'); if (xr) {
    if (xr.dataset.xrefMod) { location.href = `./${xr.dataset.xrefMod}.html`; return; }
    if (xr.dataset.xrefLec) {
      const num = xr.dataset.xrefLec.split('-').pop();
      const ix = LECS.findIndex(L => L.num === num);
      if (xr.dataset.xrefLec.startsWith(MOD) && ix >= 0) showLecture(ix);
    }
    return;
  }
  if (t.id === 'b-focus') { document.body.classList.toggle('focus'); return; }
  if (t.id === 'b-search') { openSearch(); return; }
  if (t.id === 'b-help') { $('#hd').showModal(); return; }
  if (t.id === 'b-review' || t.id === 'startReview') { openReview(); return; }
  if (t.id === 'revX' || t.id === 'revClose') { closeReview(); return; }
  if (t.id === 'revReveal') { revReveal(); return; }
  if (t.id === 'exportAnki') { exportAnki(); return; }
  const rg = t.closest('.rev-grade'); if (rg) { revGrade(+rg.dataset.g); return; }
  if (t.id === 'b-work' || t.id === 'rrClose') { $('#railR').classList.toggle('open'); return; }
  if (t.id === 'b-exp') {
    const blob = JSON.stringify({ done: DONE, hl: HL, notes: NOTES, rate: RATE, phase: PHASE, review: REVIEW, srs: SRS }, null, 2);
    navigator.clipboard?.writeText(blob).then(() => alert('Study data copied.'), () => prompt('Copy:', blob));
    return;
  }
  if (t.id === 'b-imp') {
    const s = prompt('Paste exported study data:'); if (!s) return;
    try { const o = JSON.parse(s);
      DONE = o.done || {}; HL = o.hl || {}; NOTES = o.notes || {}; RATE = o.rate || {}; PHASE = o.phase || {}; REVIEW = o.review || {}; SRS = o.srs || {};
      save(); showLecture(curLec); alert('Imported.');
    } catch (e) { alert('Could not read that.'); }
    return;
  }
  const sri = t.closest('.sri'); if (sri) {
    if (sri.dataset.href) { location.href = sri.dataset.href; return; }
    goto(sri.dataset.go, LECS.findIndex(L => L.id === sri.dataset.lec));
    return;
  }
  if (!t.closest('#hlbar')) $('#hlbar').classList.remove('on');
});

document.addEventListener('input', e => {
  if (e.target.id === 'sq') doSearch(e.target.value);
  if (e.target.id === 'gf') {
    const q = e.target.value.toLowerCase();
    $$('.view table tbody tr').forEach(tr => { tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none'; });
  }
});

/* ── highlight toolbar ── */
let SEL = null;
document.addEventListener('mouseup', () => setTimeout(() => {
  const s = getSelection(); const bar = $('#hlbar');
  if (!s || s.isCollapsed || !s.toString().trim() || view !== 'reading') return;
  const r = s.getRangeAt(0);
  const sec = (r.startContainer.nodeType === 1 ? r.startContainer : r.startContainer.parentElement)?.closest('section');
  if (!sec || !doc.contains(sec)) return;
  const b = r.getBoundingClientRect ? r.getBoundingClientRect()
    : (r.startContainer.parentElement || r.startContainer).getBoundingClientRect();
  bar.style.left = Math.max(8, Math.min(innerWidth - 185, b.left + b.width / 2 - 88)) + 'px';
  bar.style.top = (b.top > 62 ? b.top - 40 : b.bottom + 8) + 'px';
  bar.classList.add('on');
  const raw = s.toString();
  const lead = raw.length - raw.trimStart().length;
  SEL = { sec, text: raw.trim(), start: offsetOfRange(sec, r) + lead, len: raw.trim().length };
}, 10));
$('#hlbar').addEventListener('click', e => {
  const sw = e.target.closest('.sw'); if (!sw) return;
  const c = sw.dataset.c;
  if (c === '0') {
    const hl = getSelection().anchorNode?.parentElement?.closest('.hl');
    if (hl) { delete HL[hl.dataset.k]; save(); removeHL(hl.dataset.k); }
  } else if (SEL && SEL.sec) {
    const k = 'h' + Date.now();
    HL[k] = { mod: MOD, lec: LECS[curLec].id, sec: SEL.sec.id, text: SEL.text, start: SEL.start, len: SEL.len, c };
    save(); applyHL(SEL.sec, SEL.start, SEL.len, c, k);
  }
  getSelection().removeAllRanges();
  $('#hlbar').classList.remove('on'); renderRight();
});

/* ── keyboard ── */
addEventListener('keydown', e => {
  const sd = $('#sd');
  if (REV && revEl('rev').classList.contains('on')) {
    if (e.key === 'Escape') { closeReview(); return; }
    const back = revEl('revBack');
    if ((e.key === ' ' || e.key === 'Enter') && back && back.classList.contains('hidden')) { e.preventDefault(); revReveal(); return; }
    if (/^[123]$/.test(e.key) && back && !back.classList.contains('hidden')) { revGrade(+e.key - 1); return; }
    return;
  }
  if (e.key === 'Escape') { sd.classList.remove('on'); document.body.classList.remove('focus'); $('#railR').classList.remove('open'); return; }
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openSearch(); return; }
  if (sd.classList.contains('on')) {
    const items = [...sd.querySelectorAll('.sri')];
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      let i = items.findIndex(x => x.classList.contains('sel'));
      items[i]?.classList.remove('sel');
      i = (i + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
      items[i]?.classList.add('sel'); items[i]?.scrollIntoView({ block: 'nearest' });
    }
    if (e.key === 'Enter') { e.preventDefault(); sd.querySelector('.sri.sel')?.click(); }
    return;
  }
  if (/^(INPUT|TEXTAREA)$/.test(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  if (k === 'f') document.body.classList.toggle('focus');
  if (k === 'r') openReview();
  if (k === '?' || (e.shiftKey && k === '/')) $('#hd').showModal();
  if (k === '[' && curLec > 0) showLecture(curLec - 1);
  if (k === ']' && curLec < LECS.length - 1) showLecture(curLec + 1);
  if ((k === 'j' || k === 'k') && view === 'reading') {
    const secs = $$('section'); const y = scrollY + 76;
    let i = secs.findIndex(s => s.offsetTop > y);
    if (k === 'j') { if (i < 0) i = secs.length - 1; } else { i = Math.max(0, (i < 0 ? secs.length : i) - 2); }
    secs[i]?.scrollIntoView();
  }
  if ('1234'.includes(k) && SEL) document.querySelector(`.sw[data-c="${k}"]`)?.click();
  if (k === '0') document.querySelector('.sw[data-c="0"]')?.click();
});

/* ── scrollspy + reading bar ── */
let ticking = false;
addEventListener('scroll', () => {
  if (ticking) return; ticking = true;
  requestAnimationFrame(() => {
    ticking = false;
    const h = document.documentElement;
    $('#readbar').style.width = (h.scrollTop / (h.scrollHeight - h.clientHeight) * 100 || 0) + '%';
    if (view !== 'reading') return;
    const secs = $$('section'); if (!secs.length) return;
    let cur = secs[0];
    for (const s of secs) { if (s.getBoundingClientRect().top < 120) cur = s; else break; }
    secs.forEach(s => s.classList.toggle('live', s === cur));
    railL.querySelectorAll('.secs.open a').forEach(a => a.classList.toggle('cur', a.dataset.go === cur.id));
    history.replaceState(null, '', '#' + cur.id);
  });
}, { passive: true });

/* ── interactive figures ── */
function upgradeInteractives(){
  $$('.iv').forEach(fig => {
    let spec; try { spec = JSON.parse(fig.dataset.iv); } catch (e) { return; }
    const fn = (window.INTERACTIVE || {})[spec.title];
    if (!fn) return;
    const mount = fig.querySelector('.iv-mount');
    mount.hidden = false;
    fig.classList.add('is-live');
    fig.querySelector('.iv-cap').insertAdjacentHTML('beforeend', '<span class="iv-live">live</span>');
    try { fn(mount, spec); } catch (e) { mount.hidden = true; fig.classList.remove('is-live'); }
  });
}

/* ── boot ── */
(function(){
  const hash = location.hash.slice(1);
  let ix = 0, anchor = null;
  const bySec = LECS.findIndex(L => L.sections.some(s => s.id === hash));
  const byLec = LECS.findIndex(L => hash === L.id);
  if (bySec >= 0) { ix = bySec; anchor = hash; }
  else if (byLec >= 0) ix = byLec;
  else {
    const last = store.get('last', null);
    if (last && last.mod === MOD && last.lec < LECS.length) { ix = last.lec; anchor = last.sec; }
  }
  showLecture(ix, anchor);
  if (hash === 'review') { history.replaceState(null, '', location.pathname); openReview(); }
})();
