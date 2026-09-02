// _convert.mjs — replace a fenced ASCII block under a given heading with new content.
// dev tool.  Run from repo root:
//   node web/_convert.mjs "notes/rel/path.md" "<heading substring>" "<replacement file>"
import fs from 'node:fs';
import path from 'node:path';
const [rel, headSub, newFile] = process.argv.slice(2);
const p = path.resolve(process.cwd(), rel);
let src = fs.readFileSync(p, 'utf8').replace(/\r\n?/g, '\n');
const repl = fs.readFileSync(path.resolve(process.cwd(), newFile), 'utf8').replace(/\r\n?/g, '\n').replace(/\n$/, '');
const hi = src.indexOf(headSub);
if (hi < 0) { console.error('heading not found:', headSub); process.exit(1); }
const after = src.slice(hi);
const fence = /```[a-z]*\n[\s\S]*?\n```/.exec(after);
if (!fence) { console.error('no fenced block after heading'); process.exit(1); }
const start = hi + fence.index, end = start + fence[0].length;
fs.writeFileSync(p, src.slice(0, start) + repl + src.slice(end));
console.log(`OK ${rel}  ::  ${headSub.slice(0, 48)}   (${fence[0].length} -> ${repl.length})`);
