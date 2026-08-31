// notes markdown -> study-artifact HTML.  Content fidelity is the contract:
// nothing dropped, nothing paraphrased, every number verbatim.  The markdown is
// the source of truth; this file only renders it.
import MarkdownIt from 'markdown-it';
import deflist from 'markdown-it-deflist';
import Temml from 'temml';
import { parse as parseHTML } from 'node-html-parser';

// ── fixed callout map — identical meaning in every module (NOTES_PIPELINE.md) ──
export const CALLOUTS = {
  '📚': ['bg', 'Background'],
  '💡': ['key', 'Key insight'],
  '⚠️': ['warn', 'Careful'],
  '🧪': ['lab', 'Worked example'],
  '🎯': ['int', 'Interview'],
  '🔬': ['res', 'Research'],
  '🩹': ['recon', 'Reconstructed'],
};
const MARKERS = Object.keys(CALLOUTS);

const md = new MarkdownIt({ html: true, linkify: false, typographer: false, breaks: false })
  .use(deflist);

// ── math: pull every span out before markdown touches it, render with Temml ──
function extractMath(src, store) {
  // block first
  src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
    const i = store.length;
    store.push({ block: true, tex: tex.trim() });
    return `\n\nMATHPLACEHOLDER${i}E\n\n`;
  });
  // inline — not preceded/followed by $, allow escaped \$ inside
  src = src.replace(/(?<!\\)\$((?:[^$\n\\]|\\.)+?)(?<!\\)\$/g, (_, tex) => {
    const i = store.length;
    store.push({ block: false, tex: tex.replace(/\s*\n\s*/g, ' ').trim() });
    return `MATHPLACEHOLDER${i}E`;
  });
  return src;
}
function renderMath(m) {
  try {
    let html = Temml.renderToString(m.tex, { displayMode: m.block, throwOnError: true });
    html = html.replace(/^<math/, `<math data-tex="${escAttr(m.tex)}"`);
    if (m.block) return `<div class="mblk"${/\\boxed/.test(m.tex) ? ' data-boxed="1"' : ''}>${html}</div>`;
    return html;
  } catch (e) {
    return m.block
      ? `<div class="mblk mfail">${esc(m.tex)}</div>`
      : `<code class="mfail">${esc(m.tex)}</code>`;
  }
}

// ── interactive spec blocks: parsed here, realised by the client runtime ──
function extractInteractive(src, store) {
  return src.replace(/```interactive\n([\s\S]*?)```/g, (_, body) => {
    const spec = {};
    let key = null;
    for (const line of body.split('\n')) {
      const m = /^(\w+):\s?(.*)$/.exec(line);
      if (m) { key = m[1]; spec[key] = m[2]; }
      else if (key && line.trim()) spec[key] += ' ' + line.trim();
    }
    const i = store.length;
    store.push(spec);
    return `\n\nIVPLACEHOLDER${i}E\n\n`;
  });
}

const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const escAttr = s => esc(s).replace(/"/g, '&quot;');
// node-html-parser's parse() returns a transparent root; setAttribute on it is a
// no-op, so when we need a real mutable element we take its firstChild.
const frag = html => parseHTML(html).firstChild;

// ── cross-references -> in-artifact links ──
// "Part 2 §6", "Part 1 §8–11", "§15.4", "§1"  — every one becomes a real link,
// even when the surrounding sentence also contains inline math or code.
const XREF = /(?:Part\s*(\d+)\s*)?§\s*(\d+)(?:\.\d+)?(?:\s*[–-]\s*\d+)?/g;
// split an innerHTML string into [text, protected, text, protected, …] where
// "protected" is a whole <a>/<code>/<pre>/<math> subtree we must not rewrite
const PROTECT = /(<(a|code|pre|math)\b[\s\S]*?<\/\2>)/gi;
function linkifyXrefs(root, ctx) {
  const mods = (ctx.modules || []).filter(m => m.slug !== ctx.moduleSlug);
  const same = (whole, part) => {
    const lecNum = part ? String(part).padStart(2, '0') : ctx.lectureNum;
    return `<a class="xref" data-xref-lec="${ctx.moduleSlug}-${lecNum}" data-xref="${escAttr(whole)}" role="link" tabindex="0">${esc(whole)}</a>`;
  };
  const cross = (whole, slug) =>
    `<a class="xref" data-xref-mod="${slug}" data-xref="${escAttr(whole)}" role="link" tabindex="0">${esc(whole)}</a>`;
  const rewrite = str => str.replace(XREF, (whole, part, sec, off) => {
    if (part) {                                    // "Part N §M" may belong to another module
      const pre = str.slice(Math.max(0, off - 46), off);
      const m = mods.find(x => pre.includes(x.name) || (x.name === 'GenAI & LLM' && /\bGenAI\b|\bLLMs?\b/.test(pre)));
      if (m) return cross(whole, m.slug);
    }
    return same(whole, part);
  });
  for (const elm of root.querySelectorAll('p, li, td, dd, dt, h2, h3, h4, aside, blockquote, figcaption, .iv-fb-prose')) {
    if (!/§/.test(elm.textContent)) continue;
    const before = elm.innerHTML;
    // split() interleaves both capture groups: [text, subtree, tagname, text, …]
    const after = before.split(PROTECT).map((seg, i) =>
      i % 3 === 2 ? '' : i % 3 === 1 ? seg : rewrite(seg || '')
    ).join('');
    if (after !== before) elm.innerHTML = after;
  }
}

// ── main ──
export function renderLecture(mdSrc, ctx) {
  // ctx: { moduleSlug, moduleName, lectureNum, file }
  const math = [], iv = [];
  let src = mdSrc.replace(/\r\n?/g, '\n');
  const fm = /^---\n([\s\S]*?)\n---\n/.exec(src);
  const front = {};
  if (fm) {
    for (const line of fm[1].split('\n')) {
      const m = /^(\w+):\s*(.*)$/.exec(line);
      if (m) front[m[1]] = m[2].replace(/^["']|["']$/g, '');
    }
    src = src.slice(fm[0].length);
  }

  // Collapse the document's own opening headings into metadata so the reader
  // sees the title exactly once.  Pattern in every lecture:
  //     # Supervised Learning — Part 1
  //     ### Problem Formulation, Data Discipline, …
  src = src.replace(/^\s*#\s+(.+?)[ \t]*\n(?:[ \t]*#{2,4}\s+(.+?)[ \t]*\n)?/, (_, h1, sub) => {
    front.h1 = h1.trim();
    if (sub) front.subtitle = sub.trim();
    front.title = front.title || h1.trim();
    return '';
  });
  // "Module — Part N: Descriptive subtitle"  ->  split for the header
  const colon = /^(.*?—\s*Part\s*\d+)\s*:\s*(.+)$/.exec(front.title || '');
  if (colon) { front.h1 = front.h1 || colon[1].trim(); front.subtitle = front.subtitle || colon[2].trim(); }

  src = extractInteractive(src, iv);
  src = extractMath(src, math);

  let html = md.render(src);
  // restore placeholders on the string (they came through as plain text)
  html = html
    .replace(/<p>\s*(MATHPLACEHOLDER\d+E|IVPLACEHOLDER\d+E)\s*<\/p>/g, '$1')
    .replace(/MATHPLACEHOLDER(\d+)E/g, (_, i) => renderMath(math[+i]))
    .replace(/IVPLACEHOLDER(\d+)E/g, (_, i) => renderInteractive(iv[+i], +i, ctx));

  const root = parseHTML(`<div id="__root">${html}</div>`, { comment: false });
  const doc = root.querySelector('#__root');

  transformCallouts(doc);
  transformDetails(doc);
  transformCode(doc);
  bindSymbolTables(doc);
  linkifyXrefs(doc, ctx);
  wrapExternalLinks(doc);

  const sections = sectionize(doc, ctx);

  return {
    front,
    html: doc.innerHTML,
    sections,
    counts: {
      h2: sections.length,
      details: (mdSrc.match(/<details>/g) || []).length,
      rows: (mdSrc.split('\n').filter(l => l.trim().startsWith('|')).length),
      warn: (mdSrc.match(/⚠️/g) || []).length,
      interactive: iv.length,
    },
    interactive: iv,
  };
}

function transformCallouts(doc) {
  // rebuilding a callout's innerHTML re-parses any nested blockquote, so a
  // single pass can leave a marker-blockquote inside a just-built aside —
  // repeat until every blockquote is handled
  for (let pass = 0; pass < 4; pass++) {
    const pending = doc.querySelectorAll('blockquote');
    if (!pending.length) break;
    pending.forEach(convertCallout);
  }
}
function convertCallout(bq) {
    const first = bq.textContent.trimStart();
    const marker = MARKERS.find(m => first.startsWith(m));
    if (!marker) { bq.setAttribute('class', 'aside-quote'); return; }
    const [cls, label] = CALLOUTS[marker];
    // strip only the leading emoji marker — keep every word of the source,
    // including a "**bold lead-in:**"; the badge + label are added, not swapped
    let inner = bq.innerHTML.replace(new RegExp(`^\\s*(<p>)?\\s*${marker}\\uFE0F?\\s*`), '$1');
    bq.tagName = 'aside';
    bq.setAttribute('class', `callout callout-${cls}`);
    bq.innerHTML = `<p class="callout-h"><span class="callout-badge">${marker}</span>${label}</p>${inner}`;
}

function transformDetails(doc) {
  doc.querySelectorAll('details').forEach((d, k) => {
    const sum = d.querySelector('summary');
    const qRaw = sum ? sum.innerHTML : 'Show';
    if (sum) sum.remove();
    const num = /<b>\s*(\d+)\.?\s*<\/b>/.exec(qRaw) || /^\s*(\d+)\./.exec(d.textContent);
    const q = qRaw.replace(/<\/?b>/g, '').replace(/^\s*\d+\.\s*/, '').trim();
    const ans = d.innerHTML;
    const fig = parseHTML(`<div class="recall" data-recall>
      <button class="recall-q" type="button">
        <span class="recall-n">${num ? num[1] : 'Q'}</span>
        <span class="recall-t">${q}</span>
        <span class="recall-x">Reveal</span>
      </button>
      <div class="recall-a" hidden>${ans}
        <div class="rate" role="group" aria-label="How well did you know this?">
          <button type="button" class="rate-b" data-r="1">Got it</button>
          <button type="button" class="rate-b" data-r="2">Shaky</button>
          <button type="button" class="rate-b" data-r="3">No idea</button>
        </div>
      </div></div>`);
    d.replaceWith(fig);
  });
}

const unesc = s => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&');
function transformCode(doc) {
  // node-html-parser keeps <pre> content as raw text, so the <code> wrapper
  // sits inside pre.innerHTML as a literal string — work with that.
  doc.querySelectorAll('pre').forEach(pre => {
    const raw = pre.innerHTML;
    const lang = (/<code[^>]*class="language-(\w+)"/.exec(raw) || [])[1] || '';
    const inner = raw.replace(/^\s*<code[^>]*>/, '').replace(/<\/code>\s*$/, '').replace(/\n$/, '');
    const isArt = !lang && /[┌┐└┘├┤┬┴┼─│╱╲▲▼◄►●○◆■□▶◀↑↓⟶⟵]/.test(inner);
    if (isArt) {
      pre.tagName = 'figure';
      pre.setAttribute('class', 'ascii');
      pre.set_content(`<pre>${inner}</pre>`);
    } else {
      pre.setAttribute('class', lang ? `code code-${lang}` : 'code');
      pre.set_content(`<code${lang ? ` class="language-${lang}"` : ''}>${inner}</code>`);
    }
  });
}

// symbol-definition tables bind to the equation immediately above them so a
// scroll never separates a formula from what its symbols mean.
function bindSymbolTables(doc) {
  doc.querySelectorAll('table').forEach(tbl => {
    const th = tbl.querySelector('th');
    const isSym = th && /^symbol$/i.test(th.textContent.trim());
    const prev = tbl.previousElementSibling;
    const bound = isSym && prev && prev.classList && prev.classList.contains('mblk');
    const wrap = frag(`<div class="table-wrap${isSym ? ' symtab' : ''}${bound ? ' bound' : ''}"></div>`);
    tbl.replaceWith(wrap);
    wrap.appendChild(tbl);
    if (bound) prev.setAttribute('class', (prev.getAttribute('class') || 'mblk') + ' bound');
  });
}

function wrapExternalLinks(doc) {
  doc.querySelectorAll('a').forEach(a => {
    const href = a.getAttribute('href') || '';
    if (/^https?:/.test(href)) { a.setAttribute('target', '_blank'); a.setAttribute('rel', 'noopener'); }
    else if (/\.md/.test(href)) {
      // link to another lecture's markdown -> in-artifact xref
      const slugMatch = /([a-z-]+)-(\d+)\.md/.exec(href);
      a.setAttribute('class', 'xref');
      if (slugMatch) a.setAttribute('data-xref-lec', `${slugMatch[1]}-${slugMatch[2]}`);
      a.removeAttribute('href');
      a.setAttribute('role', 'link');
      a.setAttribute('tabindex', '0');
    }
  });
}

const WPM = 130; // honest: dense technical maths, not blog prose
function sectionize(doc, ctx) {
  const kids = [...doc.childNodes];
  const sections = [];
  let cur = null, pre = [], part = null, partIx = 0;
  const flush = () => { if (cur) sections.push(cur); };
  for (const node of kids) {
    // a bare "# Part N — …" inside the body is a macro-part divider grouping
    // the numbered sections under it — a landmark, not the lecture title
    if (node.nodeType === 1 && node.rawTagName === 'h1') {
      flush(); cur = null;
      part = { id: `${ctx.moduleSlug}-${ctx.lectureNum}-p${partIx++}`, title: node.textContent.trim(), _h1: node };
      continue;
    }
    if (node.nodeType === 1 && node.rawTagName === 'h2') {
      flush();
      const raw = node.textContent.trim();
      const m = /^(\d+(?:\.\d+)?)\.\s+(.*)$/.exec(raw);
      const num = m ? m[1] : '';
      const title = m ? m[2] : raw;
      const id = `${ctx.moduleSlug}-${ctx.lectureNum}-s${sections.length}`;
      cur = { id, num, title, nodes: [], _h2: node, part: part ? { id: part.id, title: part.title } : null, _partHead: part && !part._used ? part : null };
      if (part) part._used = true;
    } else if (cur) {
      cur.nodes.push(node);
    } else if (part && !part._used) {
      // content between the part heading and its first section
      (part._lead = part._lead || []).push(node);
    } else {
      pre.push(node);
    }
  }
  flush();

  // rebuild doc: front matter (pre) then <section> wrappers
  doc.set_content('');
  if (pre.length) {
    const f = frag('<div class="frontmatter"></div>');
    pre.forEach(n => f.appendChild(n));
    doc.appendChild(f);
  }
  for (const s of sections) {
    const words = s.nodes.map(n => n.textContent || '').join(' ').split(/\s+/).filter(Boolean).length;
    s.mins = Math.max(1, Math.round(words / WPM));
    s.kind = /^part\s+[a-z]\b/i.test(s.title) ? 'divider'
      : /key takeaways?/i.test(s.title) ? 'recap'
      : /^(interview prep|check yourself|glossary|going deeper|putting it together)/i.test(s.title) ? 'back'
      : 'content';
    if (s._partHead) {
      const ph = frag(`<div class="part-head" id="${s._partHead.id}"><h1>${esc(s._partHead.title)}</h1></div>`);
      (s._partHead._lead || []).forEach(n => ph.appendChild(n));
      doc.appendChild(ph);
    }
    const sec = frag(`<section id="${s.id}" data-title="${escAttr(s.title)}" data-kind="${s.kind}"${s.part ? ` data-part="${escAttr(s.part.title)}"` : ''}></section>`);
    const h = frag(`<h2>
      ${s.num ? `<span class="sec-n" aria-hidden="true">${s.num}</span>` : ''}
      <span class="sec-t">${s._h2.innerHTML.replace(/^\s*\d+(?:\.\d+)?\.\s+/, '')}</span>
      <span class="sec-meta"><span class="sec-time">${s.mins} min</span>
      <button class="sec-check" type="button" data-sec="${s.id}" aria-label="Mark this section understood" title="I understand this"></button></span>
    </h2>`);
    sec.appendChild(h);
    s.nodes.forEach(n => sec.appendChild(n));
    doc.appendChild(sec);
    delete s._h2; delete s.nodes; delete s._partHead;
  }
  return sections;
}

// The interactive runtime lives in the template; here we emit a spec the client
// upgrades in place, with the fallback rendered as complete standalone prose so
// a reader who never gets JS still has the whole teaching point.
// render `$math$` and `code` inside a plain spec string so the fallback reads
// as finished prose, not raw LaTeX
function inlineRich(s) {
  if (!s) return '';
  return String(s).split(/(\$[^$\n]+?\$|`[^`\n]+?`)/).map(seg => {
    if (seg[0] === '$' && seg[seg.length - 1] === '$') {
      try { return Temml.renderToString(seg.slice(1, -1).trim(), { throwOnError: true }); }
      catch (e) { return `<code class="mfail">${esc(seg.slice(1, -1))}</code>`; }
    }
    if (seg[0] === '`' && seg[seg.length - 1] === '`') return `<code>${esc(seg.slice(1, -1))}</code>`;
    return esc(seg);
  }).join('');
}
function renderInteractive(spec, i, ctx) {
  const id = `iv-${ctx.moduleSlug}-${ctx.lectureNum}-${i}`;
  const row = (k, v) => v ? `<div class="iv-row"><dt>${k}</dt><dd>${inlineRich(v)}</dd></div>` : '';
  return `<figure class="iv" id="${id}" data-iv="${escAttr(JSON.stringify(spec))}">
    <figcaption class="iv-cap"><span class="iv-kind">Interactive · ${esc(spec.type || 'figure')}</span>
      <span class="iv-title">${esc(spec.title || '')}</span></figcaption>
    <div class="iv-mount" hidden></div>
    <div class="iv-fallback">
      <dl class="iv-meta">
        ${row('Concept', spec.concept)}
        ${row('Try', spec.control)}
        ${row('Watch', spec.observe)}
        ${row('Insight', spec.insight)}
      </dl>
      <p class="iv-fb-prose">${inlineRich(spec.fallback || '')}</p>
    </div>
  </figure>`;
}
