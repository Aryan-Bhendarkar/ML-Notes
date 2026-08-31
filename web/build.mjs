// build.mjs — assemble one module's notes into a self-contained study artifact,
// and refresh the course home.  One module per invocation (each is 60–120k words).
//
//   node build.mjs "Supervised Learning"
//   node build.mjs --home        # rebuild only docs/index.html
//
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { renderLecture } from './lib/render.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const NOTES = path.join(ROOT, 'notes');
// build output — serve this folder with GitHub Pages (Settings ▸ Pages ▸ main /docs)
const DIST = path.join(ROOT, 'docs');
const slugify = s => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const readMd = p => fs.readFileSync(p, 'utf8').replace(/\r\n?/g, '\n');

// ── module discovery ────────────────────────────────────────────────
function moduleFiles(dir) {
  const files = fs.readdirSync(dir)
    .filter(f => f.endsWith('.md') && !/readme|quality_review/i.test(f))
    .map(f => {
      const lec = (/^lecture:\s*(\d+)/m.exec(readMd(path.join(dir, f))) || [])[1];
      return { f, lec: lec ? +lec : 999 };
    })
    .sort((a, b) => a.lec - b.lec || a.f.localeCompare(b.f));
  return files.map(x => path.join(dir, x.f));
}
function scanModules() {
  return fs.readdirSync(NOTES)
    .filter(d => fs.statSync(path.join(NOTES, d)).isDirectory())
    .map(name => {
      const files = moduleFiles(path.join(NOTES, name));
      const slug = slugify(name);
      const lectures = files.map((f, i) => {
        const src = readMd(f);
        const fm = /^---\n([\s\S]*?)\n---/.exec(src);
        const get = k => fm && (new RegExp(`^${k}:\\s*(.*)$`, 'm').exec(fm[1]) || [])[1]?.replace(/^["']|["']$/g, '');
        return {
          file: path.basename(f, '.md'),
          num: String(get('lecture') || i + 1).replace(/\D/g, '').padStart(2, '0'),
          bridge: !get('lecture'),                         // no `lecture:` field ⇒ a bridge / practicum, not a numbered lecture
          title: get('title') || path.basename(f, '.md'),
          instructor: get('instructor') || '',
          slides: get('slides') || '',
          h2: (src.match(/^## /gm) || []).length,
          // "## Part A — …" dividers carry no content; don't count them for progress
          trackable: (src.match(/^## /gm) || []).length - (src.match(/^## Part\s+[A-Z]\b/gm) || []).length,
          words: src.split(/\s+/).length,
        };
      });
      return {
        name, slug, dir: path.join(NOTES, name),
        lectures,
        sectionCount: lectures.reduce((a, l) => a + l.trackable, 0),
        wordCount: lectures.reduce((a, l) => a + l.words, 0),
      };
    });
}

const MODULE_ORDER = [
  'Supervised Learning', 'Deep Neural Networks', 'Dimensionality Reduction',
  'Unsupervised Learning', 'GenAI & LLM', 'Sequential Learning',
  'Causal Inference', 'Reinforcement Learning', 'Agentic AI',
];
function orderedModules() {
  const all = scanModules();
  return MODULE_ORDER.map(n => all.find(m => m.name === n)).filter(Boolean)
    .concat(all.filter(m => !MODULE_ORDER.includes(m.name)));
}

// short lecture label: "Part 2 · Losses, Optimisation, …"
function shortTitle(t) {
  const m = /(Part\s*\d+)\s*[:—-]\s*(.+)/.exec(t);
  let s = m ? `${m[1]} · ${m[2]}` : t;
  return s.length > 66 ? s.slice(0, 64).replace(/\s+\S*$/, '') + '…' : s;
}

// ── build one module ────────────────────────────────────────────────
function buildModule(mod, course) {
  const lectures = [];
  const counts = { h2: 0, details: 0, rows: 0, warn: 0, interactive: 0 };
  for (let i = 0; i < mod.lectures.length; i++) {
    const meta = mod.lectures[i];
    const src = readMd(path.join(mod.dir, meta.file + '.md'));
    const out = renderLecture(src, {
      moduleSlug: mod.slug, moduleName: mod.name,
      lectureNum: meta.num, file: meta.file,
      modules: course.map(m => ({ name: m.name, slug: m.slug })),
    });
    for (const k in counts) counts[k] += out.counts[k];
    lectures.push({
      id: `${mod.slug}-${meta.num}`,
      num: meta.num,
      file: meta.file,
      kind: meta.bridge ? 'practicum' : 'lecture',
      title: out.front.title || meta.title,
      h1: out.front.h1 || '',
      subtitle: out.front.subtitle || '',
      spineLabel: meta.bridge ? 'Practicum'
        : (/(Part\s*\d+)/i.exec(out.front.h1 || out.front.title || '') || [])[1] || ('Lecture ' + (+meta.num)),
      short: shortTitle(out.front.subtitle || out.front.title || meta.title),
      instructor: out.front.instructor || meta.instructor || '',
      html: out.html,
      sections: out.sections.map(s => ({ id: s.id, num: s.num, title: s.title, mins: s.mins, kind: s.kind, part: s.part ? s.part.title : '' })),
      interactiveCount: out.counts.interactive,
    });
  }

  const payload = {
    slug: mod.slug, name: mod.name,
    lectures,
    course: course.map(m => ({
      slug: m.slug, name: m.name,
      lectures: m.lectures.length, sections: m.sectionCount,
      index: m.lectures.map(l => ({ num: l.num, short: shortTitle(l.title) })),
    })),
  };

  const tpl = fs.readFileSync(path.join(HERE, 'app.template.html'), 'utf8');
  const appJs = fs.readFileSync(path.join(HERE, 'app.js'), 'utf8');
  const ivJs = fs.readFileSync(path.join(HERE, 'interactive.js'), 'utf8');
  const html = tpl
    .replace(/__MODULE__/g, esc(mod.name))
    .replace('__MATH_CSS__', () => mathCss())
    .replace('__IV_JS__', () => ivJs)
    .replace('__APP_JS__', () => appJs)
    .replace('__DATA__', () => JSON.stringify(payload).replace(/<\/script>/gi, '<\\/script>'));

  fs.mkdirSync(DIST, { recursive: true });
  const outPath = path.join(DIST, `${mod.slug}.html`);
  fs.writeFileSync(outPath, html);
  return { outPath, counts, lectures, srcCounts: srcFidelity(mod) };
}

// raw-source counts, straight from the .md — the number the build must match
function srcFidelity(mod) {
  let h2 = 0, details = 0, rows = 0, warn = 0, iv = 0;
  for (const l of mod.lectures) {
    const s = readMd(path.join(mod.dir, l.file + '.md'));
    h2 += (s.match(/^## /gm) || []).length;
    details += (s.match(/<details>/g) || []).length;
    rows += s.split('\n').filter(x => x.trim().startsWith('|')).length;
    warn += (s.match(/⚠️/g) || []).length;
    iv += (s.match(/^```interactive/gm) || []).length;
  }
  return { h2, details, rows, warn, interactive: iv };
}

function buildHome(course) {
  const tpl = fs.readFileSync(path.join(HERE, 'home.template.html'), 'utf8');
  const payload = {
    modules: course.map((m, i) => ({
      n: i + 1, slug: m.slug, name: m.name,
      lectures: m.lectures.map(l => ({
        num: l.num, kind: l.bridge ? 'practicum' : 'lecture',
        label: l.bridge ? 'Practicum' : (/(Part\s*\d+)/i.exec(l.title) || [])[1] || ('Lecture ' + (+l.num)),
        short: shortTitle(l.title), instructor: l.instructor, sections: l.trackable,
      })),
      lectureCount: m.lectures.filter(l => !l.bridge).length,
      sectionCount: m.sectionCount,
      wordCount: m.wordCount,
    })),
  };
  const html = tpl.replace('__DATA__', () => JSON.stringify(payload).replace(/<\/script>/gi, '<\\/script>'));
  fs.mkdirSync(DIST, { recursive: true });
  fs.writeFileSync(path.join(DIST, 'index.html'), html);
}

const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// Temml's stylesheet + its tiny script font, inlined so math renders offline
// with zero external requests (MathML Core is native in current browsers).
function mathCss() {
  const dir = path.join(HERE, 'node_modules', 'temml', 'dist');
  let css = fs.readFileSync(path.join(dir, 'Temml-Local.css'), 'utf8')
    .replace(/@font-face\s*{[^}]*}/, '');
  const b64 = fs.readFileSync(path.join(dir, 'Temml.woff2')).toString('base64');
  return `@font-face{font-family:'Temml';src:url(data:font/woff2;base64,${b64}) format('woff2');}\n${css}`;
}


// ── main ────────────────────────────────────────────────────────────
const arg = process.argv.slice(2).join(' ').trim();
const course = orderedModules();

if (arg === '--home') {
  buildHome(course);
  console.log('  wrote docs/index.html');
  process.exit(0);
}
if (!arg) {
  console.log('usage: node build.mjs "<Module Name>"   (or --home)\n\nmodules:');
  course.forEach(m => console.log(`  ${m.name}`));
  process.exit(1);
}
const mod = course.find(m => m.name.toLowerCase() === arg.toLowerCase() || m.slug === slugify(arg));
if (!mod) { console.error(`no module matching "${arg}"`); process.exit(1); }

const res = buildModule(mod, course);
buildHome(course);

// ── fidelity report ─────────────────────────────────────────────────
const kb = (fs.statSync(res.outPath).size / 1024).toFixed(0);
console.log(`\n  module   ${mod.name}`);
console.log(`  out      docs/${mod.slug}.html  (${kb} KB)\n`);
const row = (label, a, b) => {
  const ok = a === b ? 'ok ' : '!! ';
  console.log(`  ${ok}${label.padEnd(16)} source ${String(b).padStart(5)}   built ${String(a).padStart(5)}`);
};
row('## sections', res.counts.h2, res.srcCounts.h2);
row('<details>', res.counts.details, res.srcCounts.details);
row('table rows', res.counts.rows, res.srcCounts.rows);
row('⚠️ flags', res.counts.warn, res.srcCounts.warn);
row('interactive', res.counts.interactive, res.srcCounts.interactive);
console.log('\n  lectures:');
res.lectures.forEach(l =>
  console.log(`   ${l.num}  ${l.short.padEnd(50)} ${String(l.sections.length).padStart(3)} §  ${String(l.interactiveCount).padStart(2)} iv`));
const bad = ['h2', 'details', 'rows', 'warn', 'interactive'].filter(k => res.counts[k] !== res.srcCounts[k]);
console.log(bad.length ? `\n  FIDELITY MISMATCH: ${bad.join(', ')}\n` : `\n  fidelity: all counts match\n`);
