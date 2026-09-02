// diagrams.mjs — render every ```mermaid block in notes/ to a static, theme-matched
// SVG once, cache it by content hash under web/.diagrams/.  render.mjs inlines the
// cached SVG at build time (a plain fs read — no browser needed for `npm run build`).
// Re-run this whenever a diagram changes:  npm run diagrams
//
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const NOTES = path.join(ROOT, 'notes');
const CACHE = path.join(HERE, '.diagrams');
const MERMAID = path.join(HERE, 'node_modules', 'mermaid', 'dist', 'mermaid.min.js');

export const diagHash = src => crypto.createHash('sha1').update(src.trim()).digest('hex').slice(0, 16);

// study-lamp palette — keep in sync with app.template.html / home.template.html
const THEME = {
  theme: 'base',
  themeVariables: {
    background: '#1B1A17',
    primaryColor: '#2C2820', primaryTextColor: '#EDE6D7', primaryBorderColor: '#4C4739',
    secondaryColor: '#242119', secondaryTextColor: '#EDE6D7', secondaryBorderColor: '#4C4739',
    tertiaryColor: '#16140F', tertiaryTextColor: '#B4AA95', tertiaryBorderColor: '#37332B',
    lineColor: '#7C7361', textColor: '#EDE6D7',
    mainBkg: '#2C2820', nodeBorder: '#4C4739', clusterBkg: '#16140F', clusterBorder: '#37332B',
    edgeLabelBackground: '#1B1A17', titleColor: '#EDE6D7',
    fontFamily: 'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif', fontSize: '13px',
    // flowchart accents
    nodeTextColor: '#EDE6D7',
    // sequence / state / class inherit from the above
  },
  flowchart: { curve: 'linear', htmlLabels: true, padding: 10, nodeSpacing: 34, rankSpacing: 40, useMaxWidth: true },
  sequence: { useMaxWidth: true, mirrorActors: false },
  gantt: { useMaxWidth: true },
  xyChart: {
    width: 560, height: 340, showTitle: true,
    plotReservedSpacePercent: 55,
    xAxis: { labelFontSize: 12, titleFontSize: 13 },
    yAxis: { labelFontSize: 12, titleFontSize: 13 },
  },
};
// distinct series colours for xychart-beta (it has no legend — name the lines in a caption)
THEME.themeVariables.xyChart = {
  backgroundColor: '#16140F', titleColor: '#EDE6D7',
  xAxisLabelColor: '#B4AA95', xAxisTitleColor: '#7C7361', xAxisLineColor: '#4C4739', xAxisTickColor: '#4C4739',
  yAxisLabelColor: '#B4AA95', yAxisTitleColor: '#7C7361', yAxisLineColor: '#4C4739', yAxisTickColor: '#4C4739',
  plotColorPalette: '#8CDCA6,#E89170,#93B0D6,#E6BA55',
};

function collect() {
  const seen = new Map();                                  // hash -> { src, where[] }
  for (const dir of fs.readdirSync(NOTES)) {
    const d = path.join(NOTES, dir);
    if (!fs.statSync(d).isDirectory()) continue;
    for (const f of fs.readdirSync(d).filter(x => x.endsWith('.md'))) {
      const md = fs.readFileSync(path.join(d, f), 'utf8').replace(/\r\n?/g, '\n');
      for (const m of md.matchAll(/```mermaid\n([\s\S]*?)```/g)) {
        const src = m[1].trim();
        const h = diagHash(src);
        if (!seen.has(h)) seen.set(h, { src, where: [] });
        seen.get(h).where.push(`${dir}/${f}`);
      }
    }
  }
  return seen;
}

// trim the fixed width/height mermaid bakes in so the figure scales to its column
function responsive(svg) {
  return svg
    .replace(/<svg([^>]*?)\sstyle="[^"]*"/, '<svg$1')
    .replace(/<svg([^>]*?)\swidth="[\d.]+(px)?"/, '<svg$1')
    .replace(/<svg([^>]*?)\sheight="[\d.]+(px)?"/, '<svg$1')
    .replace('<svg ', '<svg preserveAspectRatio="xMidYMid meet" style="max-width:100%;height:auto" ');
}

async function main() {
  const force = process.argv.includes('--force');
  fs.mkdirSync(CACHE, { recursive: true });
  const all = collect();
  const todo = [...all].filter(([h]) => force || !fs.existsSync(path.join(CACHE, h + '.svg')));
  console.log(`  ${all.size} unique mermaid diagrams · ${todo.length} to (re)render`);
  if (!todo.length) { prune(all); return; }

  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.setContent('<!doctype html><body style="background:#1B1A17">');
  await page.addScriptTag({ path: MERMAID });
  await page.evaluate(t => window.mermaid.initialize({ startOnLoad: false, securityLevel: 'loose', ...t }), THEME);

  let ok = 0, fail = 0;
  for (const [h, { src, where }] of todo) {
    try {
      const svg = await page.evaluate(async ([id, code]) => {
        const { svg } = await window.mermaid.render(id, code);
        return svg;
      }, ['d' + h, src]);
      fs.writeFileSync(path.join(CACHE, h + '.svg'), responsive(svg));
      ok++;
    } catch (e) {
      fail++;
      console.error(`  !! ${h}  (${where[0]})\n     ${String(e.message || e).split('\n')[0]}`);
    }
  }
  await browser.close();
  prune(all);
  console.log(`  rendered ${ok}${fail ? `,  ${fail} FAILED` : ''}`);
  if (fail) process.exit(1);
}

// drop cache files whose diagram no longer exists in the notes
function prune(all) {
  for (const f of fs.readdirSync(CACHE).filter(x => x.endsWith('.svg'))) {
    if (!all.has(f.replace('.svg', ''))) { fs.unlinkSync(path.join(CACHE, f)); console.log(`  pruned ${f}`); }
  }
}

main().catch(e => { console.error(e); process.exit(1); });
