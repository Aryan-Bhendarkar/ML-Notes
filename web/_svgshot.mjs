// _svgshot.mjs — screenshot every ```svg / ```mermaid(xychart) block in a module's notes,
// each in a themed frame, to <scratch>/<file>_<n>.png.  dev tool.
//   node _svgshot.mjs "Supervised Learning" <outdir>
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const [mod, outDir] = process.argv.slice(2);
const dir = path.join(HERE, '..', 'notes', mod);
const cache = path.join(HERE, '.diagrams');
const crypto = await import('node:crypto');
const hash = s => crypto.createHash('sha1').update(s.trim()).digest('hex').slice(0, 16);
const b = await chromium.launch();
for (const f of fs.readdirSync(dir).filter(x => x.endsWith('.md'))) {
  const md = fs.readFileSync(path.join(dir, f), 'utf8').replace(/\r\n?/g, '\n');
  let k = 0;
  for (const m of md.matchAll(/```(svg|mermaid)\n([\s\S]*?)```/g)) {
    const kind = m[1], body = m[2].trim();
    let inner;
    if (kind === 'svg') inner = body;
    else {
      const p = path.join(cache, hash(body) + '.svg');
      if (!fs.existsSync(p)) { k++; continue; }
      inner = fs.readFileSync(p, 'utf8');
    }
    const pg = await b.newPage({ viewport: { width: 820, height: 520 }, deviceScaleFactor: 2 });
    await pg.setContent(`<body style="margin:0;background:#1B1A17;padding:16px"><div style="max-width:720px;margin:auto;background:#16140F;border:1px solid #37332B;border-radius:10px;padding:16px;text-align:center">${inner}</div>`);
    await pg.waitForTimeout(140);
    const box = await (await pg.$('div')).boundingBox();
    const name = `${f.replace('.md', '')}_${k}_${kind}.png`;
    await pg.screenshot({ path: path.join(outDir, name), clip: { x: box.x, y: box.y, width: box.width, height: Math.min(box.height, 3000) } });
    console.log(name, `${Math.round(box.width)}x${Math.round(box.height)}`);
    await pg.close(); k++;
  }
}
await b.close();
