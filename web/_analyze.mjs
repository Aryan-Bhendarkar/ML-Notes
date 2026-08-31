import { JSDOM } from 'jsdom'; import fs from 'node:fs';
const files = fs.readdirSync('../docs').filter(f=>f.endsWith('.html') && f!=='index.html').sort();
const flags = {};
function add(mod, cat, detail){ (flags[cat] ??= []).push(`[${mod}] ${detail}`); }

for (const f of files){
  const mod = f.replace('.html','');
  const html = fs.readFileSync('../docs/'+f,'utf8');
  const m = /<script id="DATA"[^>]*>([\s\S]*?)<\/script>/.exec(html);
  const data = JSON.parse(m[1].replaceAll('<\/script>','</script>'));

  for (const L of data.lectures){
    const dom = new JSDOM(`<body>${L.html}</body>`);
    const d = dom.window.document;
    const tag = `${L.spineLabel||L.num}`;

    // 1. leftover markdown / raw tokens
    const bodyText = d.body.textContent;
    if (/```/.test(bodyText)) add(mod, 'raw-fence', `${tag}: literal \`\`\` in text`);
    if (/\|\s*-{3,}\s*\|/.test(bodyText)) add(mod, 'raw-table-sep', `${tag}: table separator row leaked as text`);
    const rawMd = bodyText.match(/(^|\s)#{2,4}\s+\w/g); // markdown headings as text
    if (rawMd) add(mod, 'raw-heading', `${tag}: ${rawMd.length} md heading(s) as plain text`);
    {
      const hits = (bodyText.match(/\$[^$\n]{2,40}\$/g)||[]).filter(x=>/\\[a-zA-Z]|_\{|\^\{/.test(x));
      if (hits.length) add(mod, 'raw-math', `${tag}: ${hits.length} unrendered inline math e.g. ${JSON.stringify(hits[0].slice(0,40))}`);
    }

    // 2. math anomalies
    d.querySelectorAll('.mfail').forEach(e=> add(mod,'math-fail',`${tag}: ${JSON.stringify(e.textContent.slice(0,50))}`));
    const emptyMath = [...d.querySelectorAll('math')].filter(mm=>!mm.textContent.trim());
    if (emptyMath.length) add(mod,'math-empty',`${tag}: ${emptyMath.length} empty <math>`);
    // \text or \mathrm rendered as raw
    [...d.querySelectorAll('math')].forEach(mm=>{ if(/\text|\mathrm|\begin|\frac/.test(mm.textContent)) add(mod,'math-raw-cmd',`${tag}: ${JSON.stringify(mm.getAttribute('data-tex')?.slice(0,45))}`); });

    // 3. callouts: any blockquote whose text starts with an emoji marker but wasn't converted
    d.querySelectorAll('blockquote').forEach(bq=>{
      const t = bq.textContent.trimStart();
      if (/^(📚|💡|⚠️|🧪|🎯|🔬|🩹)/.test(t)) add(mod,'callout-missed',`${tag}: blockquote starts with marker but rendered as quote`);
    });
    // callout with empty body
    d.querySelectorAll('.callout').forEach(c=>{ if(c.children.length<=1) add(mod,'callout-empty',`${tag}: ${c.className}`); });

    // 4. ascii vs code misclassification
    d.querySelectorAll('pre.code').forEach(p=>{ if(/[┌┐└┘│─]{2,}/.test(p.textContent)) add(mod,'ascii-as-code',`${tag}: box-drawing in <pre class=code>`); });
    d.querySelectorAll('figure.ascii').forEach(p=>{ if(/\bdef \w+\(|import \w+|return /.test(p.textContent) && !/[┌┐└┘]/.test(p.textContent)) add(mod,'code-as-ascii',`${tag}: looks like code in figure.ascii`); });

    // 5. tables
    d.querySelectorAll('table').forEach(tb=>{
      const hdr = tb.querySelectorAll('thead th').length;
      const rows = [...tb.querySelectorAll('tbody tr')];
      const ragged = rows.filter(r=>r.children.length!==hdr && r.children.length>0);
      if (hdr && ragged.length) add(mod,'table-ragged',`${tag}: ${ragged.length}/${rows.length} rows != ${hdr} cols`);
      if (!hdr) add(mod,'table-nohead',`${tag}: table with no <thead>`);
      // stray pipe or alignment colons in cells
      [...tb.querySelectorAll('td')].forEach(td=>{ if(/^:?-{2,}:?$/.test(td.textContent.trim())) add(mod,'table-sep-cell',`${tag}: separator row became a data row`); });
    });

    // 6. cross-refs
    const deadRatio = d.querySelectorAll('.xref').length;
    // any "§N" left as plain text (not linked)
    d.querySelectorAll('p,li,td').forEach(e=>{
      if (e.querySelector('.xref')) return;
      const txt = e.textContent;
      const m2 = txt.match(/(?<![\w#])§\s?\d/g);
      if (m2 && !e.closest('.xref')) { /* count only */ e.dataset._sref = (m2.length); }
    });
    const unlinkedSref = [...d.querySelectorAll('[data-_sref]')].reduce((a,e)=>a+ +e.dataset._sref,0);
    if (unlinkedSref>3) add(mod,'xref-unlinked',`${tag}: ~${unlinkedSref} "§N" not linked`);

    // 7. empty / thin sections
    d.querySelectorAll('section').forEach(s=>{
      if (s.dataset.kind === 'divider') return;
      const words = s.textContent.trim().split(/\s+/).length;
      if (words < 12) add(mod,'section-thin',`${tag}: "${s.dataset.title}" only ${words} words`);
    });

    // 8. details / recall
    d.querySelectorAll('.recall').forEach(r=>{
      if (!r.querySelector('.recall-a')?.textContent.trim()) add(mod,'recall-empty',`${tag}: recall card with no answer`);
      const q = r.querySelector('.recall-t')?.textContent.trim();
      if (!q) add(mod,'recall-noq',`${tag}: recall card with no question`);
    });

    // 9. interactive fallback completeness
    d.querySelectorAll('.iv').forEach(fig=>{
      const fb = fig.querySelector('.iv-fb-prose')?.textContent.trim() || '';
      if (fb.length < 30) add(mod,'iv-thin-fallback',`${tag}: iv fallback only ${fb.length} chars`);
    });

    // 10. stray HTML / entities
    if (/&lt;(details|summary|b)&gt;/.test(L.html)) add(mod,'escaped-html',`${tag}: escaped <details>/<summary>/<b> in output`);
    if (/&amp;(amp|lt|gt);/.test(L.html)) add(mod,'double-escape',`${tag}: double-escaped entity`);

    // 11. headings hierarchy: h3/h4 with no preceding h2 section
    // 12. list rendered as one <p> with bullets as text
    d.querySelectorAll('p').forEach(p=>{ if(/^\s*[-*]\s+\S.*\n?\s*[-*]\s+/.test(p.textContent)) add(mod,'list-as-p',`${tag}: bullet list stuck in a <p>`); });
  }
}

for (const [cat, items] of Object.entries(flags).sort((a,b)=>b[1].length-a[1].length)){
  console.log(`\n### ${cat}  (${items.length})`);
  items.slice(0,12).forEach(i=>console.log('  '+i));
  if (items.length>12) console.log(`  … +${items.length-12} more`);
}
if (!Object.keys(flags).length) console.log('no flags');
