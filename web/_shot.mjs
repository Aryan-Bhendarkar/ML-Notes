// _shot.mjs — visual QA for diagrams / interactive figures. Dev tool, not shipped.
//   node _shot.mjs diagram <hash|latest> [out] [width]     — render a cached .diagrams SVG in a themed frame
//   node _shot.mjs page <slug> <figure.diagram:N|figure.iv:N> [out] [width]  — grab from the built page
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const mode = process.argv[2];
const b = await chromium.launch();

const FRAME = w => `<!doctype html><meta charset=utf8><body style="margin:0;background:#1B1A17;padding:22px">
  <div style="max-width:${w}px;margin:0 auto;background:#16140F;border:1px solid #37332B;border-radius:10px;padding:20px 18px;text-align:center">
  <style>
    figure{margin:0} svg{max-width:100%;height:auto}
    .edgeLabel,.edgeLabel p{background:#16140F!important;color:#B4AA95!important}
  </style>__C__</div>`;

if (mode === 'diagram') {
  let id = process.argv[3];
  const dir = path.join(HERE, '.diagrams');
  if (id === 'latest') {
    id = fs.readdirSync(dir).filter(f => f.endsWith('.svg'))
      .map(f => [f, fs.statSync(path.join(dir, f)).mtimeMs]).sort((a, b) => b[1] - a[1])[0][0].replace('.svg', '');
  }
  const out = process.argv[4] || '../scratch_shot.png';
  const width = +(process.argv[5] || 760);
  const svg = fs.readFileSync(path.join(dir, id + '.svg'), 'utf8');
  const p = await b.newPage({ viewport: { width: width + 90, height: 900 }, deviceScaleFactor: 2 });
  await p.setContent(FRAME(width).replace('__C__', `<figure>${svg}</figure>`));
  await p.waitForTimeout(150);
  const box = await (await p.$('div')).boundingBox();
  await p.screenshot({ path: out, clip: { x: box.x, y: box.y, width: box.width, height: Math.min(box.height, 4000) } });
  console.log(`wrote ${out}  (${id}, ${Math.round(box.width)}×${Math.round(box.height)})`);
} else {
  const [slug, what, out = '../scratch_shot.png', width = '820'] = process.argv.slice(3);
  const p = await b.newPage({ viewport: { width: +width + 60, height: 1100 }, deviceScaleFactor: 2 });
  await p.goto(`http://localhost:5199/${slug}.html`, { waitUntil: 'networkidle' });
  await p.evaluate(() => {
    for (const e of document.querySelectorAll('*')) {
      const s = getComputedStyle(e);
      if ((s.position === 'fixed' || s.position === 'sticky') && +s.zIndex >= 5) e.style.display = 'none';
    }
  });
  const [tag, n] = what.split(':');
  const el = (await p.$$(tag))[+n || 0];
  if (!el) { console.error('not found', what); await b.close(); process.exit(1); }
  await el.scrollIntoViewIfNeeded(); await p.waitForTimeout(250);
  await el.screenshot({ path: out });
  const box = await el.boundingBox();
  console.log(`wrote ${out}  (${Math.round(box.width)}×${Math.round(box.height)})`);
}
await b.close();
