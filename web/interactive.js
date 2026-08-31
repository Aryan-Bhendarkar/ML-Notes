"use strict";
/* Real interactive figures for Supervised Learning.  Keyed by the `title:` in
   the source `interactive` block.  Anything without an entry here keeps its
   fallback prose (which teaches the point completely on its own). */

const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent){
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
// a minimal cartesian plotter: data coords -> pixel coords, axes, grid
function mkPlot(mount, { w = 460, h = 300, xr, yr, xlab, ylab, pad = 34 }){
  const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, width: w, height: h });
  el('rect', { x: 0, y: 0, width: w, height: h, class: 'plot-bg', rx: 6 }, svg);
  const X = v => pad + (v - xr[0]) / (xr[1] - xr[0]) * (w - pad - 8);
  const Y = v => (h - pad) - (v - yr[0]) / (yr[1] - yr[0]) * (h - pad - 10);
  for (let i = 0; i <= 4; i++){
    const gx = xr[0] + (xr[1] - xr[0]) * i / 4, gy = yr[0] + (yr[1] - yr[0]) * i / 4;
    el('line', { x1: X(gx), y1: Y(yr[0]), x2: X(gx), y2: Y(yr[1]), class: 'plot-grid' }, svg);
    el('line', { x1: X(xr[0]), y1: Y(gy), x2: X(xr[1]), y2: Y(gy), class: 'plot-grid' }, svg);
    const tx = el('text', { x: X(gx), y: h - pad + 13, class: 'plot-label', 'text-anchor': 'middle' }, svg);
    tx.textContent = (+gx.toFixed(2)).toString();
    const ty = el('text', { x: pad - 5, y: Y(gy) + 3, class: 'plot-label', 'text-anchor': 'end' }, svg);
    ty.textContent = (+gy.toFixed(2)).toString();
  }
  el('line', { x1: X(xr[0]), y1: Y(yr[0]), x2: X(xr[1]), y2: Y(yr[0]), class: 'plot-axis' }, svg);
  el('line', { x1: X(xr[0]), y1: Y(yr[0]), x2: X(xr[0]), y2: Y(yr[1]), class: 'plot-axis' }, svg);
  if (xlab) { const t = el('text', { x: w - 6, y: h - pad + 13, class: 'plot-label', 'text-anchor': 'end' }, svg); t.textContent = xlab; }
  if (ylab) { const t = el('text', { x: pad - 5, y: 12, class: 'plot-label', 'text-anchor': 'end' }, svg); t.textContent = ylab; }
  mount.appendChild(svg);
  const api = {
    svg, X, Y,
    curve(fn, attrs){
      let d = '';
      for (let i = 0; i <= 240; i++){
        const x = xr[0] + (xr[1] - xr[0]) * i / 240;
        const y = fn(x);
        if (!isFinite(y)) { d += ' M'; continue; }
        d += `${d && d.slice(-2) !== ' M' ? 'L' : 'M'}${X(x).toFixed(1)},${Y(Math.max(yr[0], Math.min(yr[1], y))).toFixed(1)} `;
      }
      return el('path', Object.assign({ d, fill: 'none', 'stroke-width': 2 }, attrs), svg);
    },
    seg(x1, y1, x2, y2, attrs){ return el('line', Object.assign({ x1: X(x1), y1: Y(y1), x2: X(x2), y2: Y(y2), 'stroke-width': 2 }, attrs), svg); },
    dot(x, y, r, attrs){ return el('circle', Object.assign({ cx: X(x), cy: Y(y), r }, attrs), svg); },
  };
  return api;
}
function ctl(mount, defs, onChange){
  const bar = document.createElement('div'); bar.className = 'iv-ctl';
  const state = {};
  defs.forEach(d => {
    state[d.key] = d.value;
    const lab = document.createElement('label');
    lab.innerHTML = `${d.label} <span>${d.type === 'range'
      ? `<input type="range" min="${d.min}" max="${d.max}" step="${d.step}" value="${d.value}"><span class="val"> ${d.fmt ? d.fmt(d.value) : d.value}</span>`
      : d.options.map(o => `<label style="flex-direction:row;gap:4px;font-weight:500"><input type="radio" name="${d.key}" value="${o}" ${o === d.value ? 'checked' : ''}>${o}</label>`).join(' ')}</span>`;
    if (d.type === 'range') {
      const inp = lab.querySelector('input'), out = lab.querySelector('.val');
      inp.addEventListener('input', () => { state[d.key] = +inp.value; out.textContent = ' ' + (d.fmt ? d.fmt(+inp.value) : inp.value); onChange(state); });
    } else {
      lab.querySelectorAll('input').forEach(r => r.addEventListener('change', () => { state[d.key] = r.value; onChange(state); }));
    }
    bar.appendChild(lab);
  });
  mount.appendChild(bar);
  return state;
}
function readout(mount){ const d = document.createElement('div'); d.className = 'readout'; mount.appendChild(d); return d; }

const INTERACTIVE = {};

/* ── L2 §2 — One outlier, two fits ─────────────────────────────────── */
INTERACTIVE['One outlier, two fits'] = (mount, spec) => {
  const base = [];
  for (let i = 0; i < 18; i++){ const x = i / 17 * 8 + 1; base.push([x, 0.6 * x + 2 + (Math.sin(i * 91.7) * 0.7)]); }
  const P = mkPlot(mount, { xr: [0, 10], yr: [0, 14], xlab: 'x', ylab: 'y' });
  const fitLine = wfn => P.curve(wfn, {});
  const mseL = P.curve(x => x, { stroke: 'var(--c-warn)' });
  const maeL = P.curve(x => x, { stroke: 'var(--accent)' });
  const outDot = P.dot(6, 12, 6, { fill: 'var(--c-warn)', cursor: 'ns-resize' });
  base.forEach(([x, y]) => P.dot(x, y, 3, { fill: 'var(--ink-3)' }));
  const out = readout(mount);
  let oy = 12;
  function fitMSE(pts){ // least squares
    const n = pts.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
    pts.forEach(([x, y]) => { sx += x; sy += y; sxx += x * x; sxy += x * y; });
    const m = (n * sxy - sx * sy) / (n * sxx - sx * sx); return [m, (sy - m * sx) / n];
  }
  function fitMAE(pts){ // coordinate-descent-ish: minimise sum|y-mx-b| over a grid
    let best = [0.6, 2], bs = 1e9;
    for (let m = -0.5; m <= 2; m += 0.02) for (let b = -2; b <= 8; b += 0.1){
      let s = 0; for (const [x, y] of pts) s += Math.abs(y - m * x - b);
      if (s < bs) { bs = s; best = [m, b]; }
    }
    return best;
  }
  function draw(){
    const pts = base.concat([[6, oy]]);
    const [m1, b1] = fitMSE(pts), [m2, b2] = fitMAE(pts);
    mseL.setAttribute('d', pathFor(P, x => m1 * x + b1));
    maeL.setAttribute('d', pathFor(P, x => m2 * x + b2));
    outDot.setAttribute('cy', P.Y(oy));
    out.innerHTML = `outlier at y = <b>${oy.toFixed(1)}</b> &nbsp;·&nbsp;
      <span style="color:var(--c-warn)">MSE fit</span> ŷ = <b>${m1.toFixed(2)}</b>x + <b>${b1.toFixed(2)}</b> &nbsp;·&nbsp;
      <span style="color:var(--accent)">MAE fit</span> ŷ = <b>${m2.toFixed(2)}</b>x + <b>${b2.toFixed(2)}</b>`;
  }
  const pathFor = (P, fn) => { let d = ''; for (let i = 0; i <= 60; i++){ const x = i / 60 * 10; d += (i ? 'L' : 'M') + P.X(x).toFixed(1) + ',' + P.Y(fn(x)).toFixed(1) + ' '; } return d; };
  let drag = false;
  const svg = P.svg;
  const yAt = ev => { const r = svg.getBoundingClientRect(); const py = (ev.touches ? ev.touches[0].clientY : ev.clientY) - r.top;
    const vb = svg.viewBox.baseVal; const sy = py / r.height * vb.height;
    return Math.max(2, Math.min(14, (300 - 34 - sy) / (300 - 34 - 10) * 14)); };
  outDot.addEventListener('pointerdown', e => { drag = true; outDot.setPointerCapture(e.pointerId); });
  svg.addEventListener('pointermove', e => { if (!drag) return; oy = yAt(e); draw(); });
  svg.addEventListener('pointerup', () => drag = false);
  draw();
};

/* ── L2 §4 — The delta dial (Huber between MSE and MAE) ────────────── */
INTERACTIVE['The delta dial'] = (mount) => {
  const P = mkPlot(mount, { xr: [-6, 6], yr: [0, 8], xlab: 'error  e', ylab: 'loss' });
  P.curve(e => 0.5 * e * e, { stroke: 'var(--ink-3)', 'stroke-dasharray': '4 3' });     // MSE ½e²
  P.curve(e => Math.abs(e), { stroke: 'var(--ink-3)', 'stroke-dasharray': '1 3' });      // MAE |e|
  const hub = P.curve(e => e, { stroke: 'var(--accent)', 'stroke-width': 2.5 });
  const zoneL = P.seg(-2, 0, -2, 8, { stroke: 'var(--rule-2)', 'stroke-dasharray': '2 2' });
  const zoneR = P.seg(2, 0, 2, 8, { stroke: 'var(--rule-2)', 'stroke-dasharray': '2 2' });
  const out = readout(mount);
  const st = ctl(mount, [{ key: 'd', label: 'δ', type: 'range', min: 0.3, max: 5, step: 0.1, value: 2, fmt: v => v.toFixed(1) }], s => draw(s.d));
  function huber(e, d){ const a = Math.abs(e); return a <= d ? 0.5 * e * e : d * (a - d / 2); }
  function draw(d){
    let dd = ''; for (let i = 0; i <= 240; i++){ const e = -6 + 12 * i / 240; dd += (i ? 'L' : 'M') + P.X(e).toFixed(1) + ',' + P.Y(Math.min(8, huber(e, d))).toFixed(1) + ' '; }
    hub.setAttribute('d', dd);
    zoneL.setAttribute('x1', P.X(-d)); zoneL.setAttribute('x2', P.X(-d));
    zoneR.setAttribute('x1', P.X(d)); zoneR.setAttribute('x2', P.X(d));
    const at10 = huber(10, d);
    out.innerHTML = `δ = <b>${d.toFixed(1)}</b> — quadratic for |e| ≤ ${d.toFixed(1)}, linear beyond.
      An error of 10 costs <b>${at10.toFixed(1)}</b> under Huber vs <b>50</b> under ½e² and <b>10</b> under |e|.
      ${d < 0.6 ? ' At small δ it collapses onto MAE.' : d > 4 ? ' At large δ it hugs MSE.' : ''}`;
  }
  draw(2);
};

/* ── L2 §6 — The cost of confidence (BCE) ─────────────────────────── */
INTERACTIVE['The cost of confidence'] = (mount) => {
  const P = mkPlot(mount, { xr: [0, 1], yr: [0, 6], xlab: 'predicted p̂', ylab: '−log(·)' });
  P.curve(p => -Math.log(p), { stroke: 'var(--c-lab)' });          // loss if y=1
  P.curve(p => -Math.log(1 - p), { stroke: 'var(--c-warn)' });     // loss if y=0
  const m1 = P.dot(0.5, 0.69, 4, { fill: 'var(--c-lab)' });
  const m0 = P.dot(0.5, 0.69, 4, { fill: 'var(--c-warn)' });
  const out = readout(mount);
  ctl(mount, [{ key: 'p', label: 'p̂', type: 'range', min: 0.01, max: 0.99, step: 0.01, value: 0.5, fmt: v => v.toFixed(2) }], s => draw(s.p));
  function draw(p){
    const l1 = -Math.log(p), l0 = -Math.log(1 - p);
    m1.setAttribute('cx', P.X(p)); m1.setAttribute('cy', P.Y(Math.min(6, l1)));
    m0.setAttribute('cx', P.X(p)); m0.setAttribute('cy', P.Y(Math.min(6, l0)));
    out.innerHTML = `p̂ = <b>${p.toFixed(2)}</b> &nbsp;·&nbsp;
      if the truth is 1, loss = <b style="color:var(--c-lab)">${l1.toFixed(3)}</b>;
      if the truth is 0, loss = <b style="color:var(--c-warn)">${l0.toFixed(3)}</b>.
      ${p <= 0.02 || p >= 0.98 ? ' The curve is vertical here — no ceiling on the cost of being certain and wrong.' : ''}`;
  }
  draw(0.5);
};

/* ── L1 §9 — The U-curve, built from scratch ──────────────────────── */
INTERACTIVE['The U-curve, built from scratch'] = (mount) => {
  const P = mkPlot(mount, { xr: [1, 12], yr: [0, 5], xlab: 'model complexity', ylab: 'error' });
  const biasC = P.curve(c => 9 / (c + 1), { stroke: 'var(--c-bg)', 'stroke-dasharray': '4 3' });
  const varC = P.curve(c => 0.03 * c * c, { stroke: 'var(--c-key)', 'stroke-dasharray': '4 3' });
  const totC = P.curve(c => 9 / (c + 1) + 0.03 * c * c + 0.4, { stroke: 'var(--accent)', 'stroke-width': 2.5 });
  const marker = P.dot(6, 2, 5, { fill: 'var(--accent)' });
  const out = readout(mount);
  ctl(mount, [{ key: 'c', label: 'complexity', type: 'range', min: 1, max: 12, step: 0.5, value: 6, fmt: v => v.toFixed(1) }], s => draw(s.c));
  function draw(c){
    const b = 9 / (c + 1), v = 0.03 * c * c, t = b + v + 0.4;
    marker.setAttribute('cx', P.X(c)); marker.setAttribute('cy', P.Y(Math.min(5, t)));
    const star = 3;  // argmin of b+v here is near c≈? ; report qualitative
    out.innerHTML = `complexity <b>${c.toFixed(1)}</b> — <span style="color:var(--c-bg)">bias²</span> <b>${b.toFixed(2)}</b> +
      <span style="color:var(--c-key)">variance</span> <b>${v.toFixed(2)}</b> + noise 0.40 = <b>${t.toFixed(2)}</b>.
      ${c < 3 ? ' Underfit: bias dominates.' : c > 9 ? ' Overfit: variance dominates.' : ' Near the sweet spot.'}`;
  }
  draw(6);
};

/* ── L3 §2 — Impurity as the class mix changes ────────────────────── */
INTERACTIVE['Impurity as the class mix changes'] = (mount) => {
  const P = mkPlot(mount, { xr: [0, 1], yr: [0, 1], xlab: 'p (fraction positive)', ylab: 'impurity' });
  P.curve(p => 2 * p * (1 - p), { stroke: 'var(--accent)' });                              // Gini
  P.curve(p => p <= 0 || p >= 1 ? 0 : -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p)) / 2, { stroke: 'var(--c-key)', 'stroke-dasharray': '4 3' }); // entropy/2
  const g = P.dot(0.5, 0.5, 4, { fill: 'var(--accent)' });
  const e = P.dot(0.5, 0.5, 4, { fill: 'var(--c-key)' });
  const out = readout(mount);
  ctl(mount, [{ key: 'p', label: 'p', type: 'range', min: 0, max: 1, step: 0.01, value: 0.5, fmt: v => v.toFixed(2) }], s => draw(s.p));
  function draw(p){
    const gini = 2 * p * (1 - p);
    const ent = (p <= 0 || p >= 1) ? 0 : -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
    g.setAttribute('cx', P.X(p)); g.setAttribute('cy', P.Y(gini));
    e.setAttribute('cx', P.X(p)); e.setAttribute('cy', P.Y(ent / 2));
    out.innerHTML = `p = <b>${p.toFixed(2)}</b> &nbsp;·&nbsp; <span style="color:var(--accent)">Gini</span> = <b>${gini.toFixed(3)}</b>
      &nbsp;·&nbsp; <span style="color:var(--c-key)">entropy</span> = <b>${ent.toFixed(3)}</b> bits.
      Both peak at p = 0.5 (a coin flip) and hit 0 at a pure node.`;
  }
  draw(0.5);
};

/* ── L3 §7 — The correlation floor (ρσ²) ──────────────────────────── */
INTERACTIVE['The correlation floor'] = (mount) => {
  const P = mkPlot(mount, { xr: [1, 200], yr: [0, 1], xlab: 'number of trees  B', ylab: 'ensemble variance / σ²' });
  const curve = P.curve(B => 0.5 + (1 - 0.5) / B, { stroke: 'var(--accent)' });
  const floor = P.seg(1, 0.5, 200, 0.5, { stroke: 'var(--c-warn)', 'stroke-dasharray': '4 3' });
  const out = readout(mount);
  ctl(mount, [{ key: 'rho', label: 'ρ (tree correlation)', type: 'range', min: 0, max: 0.95, step: 0.05, value: 0.5, fmt: v => v.toFixed(2) }], s => draw(s.rho));
  function draw(rho){
    let d = ''; for (let i = 0; i <= 200; i++){ const B = 1 + i; d += (i ? 'L' : 'M') + P.X(B).toFixed(1) + ',' + P.Y(rho + (1 - rho) / B).toFixed(1) + ' '; }
    curve.setAttribute('d', d);
    floor.setAttribute('y1', P.Y(rho)); floor.setAttribute('y2', P.Y(rho));
    const at100 = rho + (1 - rho) / 100;
    out.innerHTML = `ρ = <b>${rho.toFixed(2)}</b> — variance floor is <b style="color:var(--c-warn)">${rho.toFixed(2)}·σ²</b>.
      At B = 100 you are already at <b>${at100.toFixed(3)}·σ²</b>; another 9,900 trees buy essentially nothing.
      This is why a Random Forest handicaps each tree — to push ρ down, not to add trees.`;
  }
  draw(0.5);
};

/* ════════════════════════════════════════════════════════════════════
   Modules 2–9.  Each figure is keyed by the source block's `title:`.
   Where a block appears with the same title in more than one lecture the
   registry entry is shared (see the alias lines at the bottom).
   ════════════════════════════════════════════════════════════════════ */

function logPlot(mount, opts){                       // y drawn on a log10 axis
  const P = mkPlot(mount, Object.assign({ yr: [0, 1] }, opts));
  return P;
}
const alias = (a, b) => { if (INTERACTIVE[b]) INTERACTIVE[a] = INTERACTIVE[b]; };

/* ── Agentic AI · L1 — Error compounding p^n ──────────────────────── */
INTERACTIVE['Error compounding — 0.95^n'] = (mount) => {
  const P = mkPlot(mount, { xr: [0, 100], yr: [0, 100], xlab: 'steps  n', ylab: 'end-to-end success %' });
  const curve = P.curve(n => 0, {});
  const out = readout(mount);
  ctl(mount, [{ key: 'p', label: 'per-step success', type: 'range', min: 0.8, max: 0.999, step: 0.001, value: 0.95, fmt: v => (v * 100).toFixed(1) + '%' }], s => draw(s.p));
  function draw(p){
    let d = ''; for (let i = 0; i <= 100; i++) d += (i ? 'L' : 'M') + P.X(i) + ',' + P.Y(Math.pow(p, i) * 100) + ' ';
    curve.setAttribute('d', d); curve.setAttribute('stroke', 'var(--accent)');
    const at40 = Math.pow(p, 40) * 100, at100 = Math.pow(p, 100) * 100;
    const half = Math.log(0.5) / Math.log(p);
    out.innerHTML = `a step that works <b>${(p * 100).toFixed(1)}%</b> of the time → the 40-step agent finishes
      <b>${at40.toFixed(1)}%</b> of the time, the 100-step agent <b>${at100.toFixed(1)}%</b>.
      Coin-flip reliability arrives at just <b>${Math.round(half)}</b> steps. The fix is guardrails, not a better model.`;
  }
  draw(0.95);
};

/* ── Agentic AI · L2 — tool-access strategy vs toolbox size ───────── */
INTERACTIVE['Cost & error rate vs. number of tools'] = (mount) => {
  const P = mkPlot(mount, { xr: [1, 200], yr: [0, 100], xlab: 'tools available', ylab: 'wrong-tool rate %' });
  P.curve(n => 4 * Math.sqrt(n), { stroke: 'var(--c-warn)' });                 // direct selection
  P.curve(n => 6 * Math.log2(n + 1), { stroke: 'var(--c-key)' });              // router
  P.curve(n => 3 + 0.02 * n, { stroke: 'var(--accent)' });                     // retrieval
  const mk = P.seg(20, 0, 20, 100, { stroke: 'var(--rule-2)', 'stroke-dasharray': '2 2' });
  const out = readout(mount);
  ctl(mount, [{ key: 'n', label: 'tools', type: 'range', min: 4, max: 200, step: 2, value: 20, fmt: v => v }], s => draw(s.n));
  function draw(n){
    mk.setAttribute('x1', P.X(n)); mk.setAttribute('x2', P.X(n));
    out.innerHTML = `at <b>${n}</b> tools — <span style="color:var(--c-warn)">put all in context</span> ≈ <b>${(4 * Math.sqrt(n)).toFixed(0)}%</b> wrong,
      <span style="color:var(--c-key)">router</span> ≈ <b>${(6 * Math.log2(n + 1)).toFixed(0)}%</b>,
      <span style="color:var(--accent)">retrieve then call</span> ≈ <b>${(3 + 0.02 * n).toFixed(0)}%</b>.
      All three are fine at a handful of tools; they only diverge once the toolbox is large — which is why "dump everything in" breaks in production, not in the demo.`;
  }
  draw(20);
};

/* ── Causal Inference · L2 — hidden confounder drift ──────────────── */
INTERACTIVE['Hidden-confounder strength vs. estimate drift'] = (mount) => {
  const TRUTH = 1794;
  const P = mkPlot(mount, { xr: [0, 1], yr: [0, 4000], xlab: 'strength of hidden cause  U', ylab: 'ATE estimate ($)' });
  P.seg(0, TRUTH, 1, TRUTH, { stroke: 'var(--accent)', 'stroke-dasharray': '5 3' });
  const est = P.curve(u => TRUTH * (1 + 1.4 * u), { stroke: 'var(--c-warn)' });
  const dot = P.dot(0, TRUTH, 4, { fill: 'var(--c-warn)' });
  const out = readout(mount);
  ctl(mount, [{ key: 'u', label: 'U strength', type: 'range', min: 0, max: 1, step: 0.02, value: 0, fmt: v => v < 0.03 ? 'none' : v.toFixed(2) }], s => draw(s.u));
  function draw(u){
    const e = TRUTH * (1 + 1.4 * u);
    dot.setAttribute('cx', P.X(u)); dot.setAttribute('cy', P.Y(e));
    out.innerHTML = `estimate <b>$${e.toFixed(0)}</b> vs truth <b style="color:var(--accent)">$${TRUTH}</b> —
      a gap of <b>$${(e - TRUTH).toFixed(0)}</b>.
      ${u < 0.03 ? 'At "none" they match.' : 'Nothing in the observed data reveals this drift.'}
      In real observational work you can't move this slider — you can only <em>assume</em> you're near "none".`;
  }
  draw(0);
};

/* ── DNN · L2 — variance propagation through depth ────────────────── */
INTERACTIVE['Variance propagation through depth'] = (mount) => {
  const N = 100;
  const P = mkPlot(mount, { xr: [0, 40], yr: [-12, 12], xlab: 'layer', ylab: 'log₁₀ activation variance' });
  P.seg(0, 0, 40, 0, { stroke: 'var(--accent)', 'stroke-dasharray': '4 3' });
  const curve = P.curve(() => 0, { stroke: 'var(--c-warn)' });
  const out = readout(mount);
  const st = ctl(mount, [
    { key: 's', label: 'init std σ', type: 'range', min: 0.02, max: 0.2, step: 0.005, value: 0.1, fmt: v => v.toFixed(3) },
    { key: 'L', label: 'depth', type: 'range', min: 1, max: 40, step: 1, value: 20, fmt: v => v },
  ], s => draw(s.s, s.L));
  function draw(s, L){
    const g = N * s * s;                                  // per-layer gain
    let d = '';
    for (let i = 0; i <= L; i++){ const y = i * Math.log10(g); d += (i ? 'L' : 'M') + P.X(i) + ',' + P.Y(Math.max(-12, Math.min(12, y))) + ' '; }
    curve.setAttribute('d', d);
    const final = Math.pow(g, L);
    out.innerHTML = `gain g = n·σ² = <b>${g.toFixed(3)}</b> per layer → after ${L} layers the variance is
      <b>${final < 1e-4 || final > 1e4 ? final.toExponential(1) : final.toFixed(2)}</b>×.
      ${g < 0.9 ? 'Signal is dying.' : g > 1.1 ? 'Signal is exploding.' : 'Stable — this is what σ = 1/√n buys you.'}
      The safe band around g = 1 is razor-thin, and it narrows as depth grows.`;
  }
  draw(0.1, 20);
};

/* ── DNN · L2 / L3, Sequential · L2 — exponential gradient decay ──── */
function gradDecay(mount, { xlab, unit }){
  const P = mkPlot(mount, { xr: [0, 60], yr: [-16, 6], xlab, ylab: 'log₁₀ gradient magnitude' });
  P.seg(0, -7.2, 60, -7.2, { stroke: 'var(--rule-2)', 'stroke-dasharray': '2 2' });   // ~float32 resolution
  const curve = P.curve(() => 0, { stroke: 'var(--c-warn)' });
  const out = readout(mount);
  ctl(mount, [
    { key: 'f', label: 'per-' + unit + ' factor', type: 'range', min: 0.5, max: 1.5, step: 0.01, value: 0.9, fmt: v => v.toFixed(2) },
    { key: 'n', label: unit + 's', type: 'range', min: 5, max: 60, step: 1, value: 30, fmt: v => v },
  ], s => draw(s.f, s.n));
  function draw(f, n){
    let d = '';
    for (let i = 0; i <= n; i++){ const y = i * Math.log10(f); d += (i ? 'L' : 'M') + P.X(i) + ',' + P.Y(Math.max(-16, Math.min(6, y))) + ' '; }
    curve.setAttribute('d', d);
    const to1pct = f < 1 ? Math.log(0.01) / Math.log(f) : Infinity;
    out.innerHTML = `factor <b>${f.toFixed(2)}</b> — on a log axis the decay is a <b>straight line</b>, so no amount of
      tuning rescues a factor below 1. Gradient falls to 1% of its start after
      <b>${isFinite(to1pct) ? Math.round(to1pct) : '∞'}</b> ${unit}s.
      ${f < 1 ? 'Moving that one number from 0.5 to 0.99 is exactly what gating/LSTM buys.' : 'Above 1 it explodes instead — clipping is the patch.'}`;
  }
  draw(0.9, 30);
}
INTERACTIVE['Gradient magnitude by depth'] = m => gradDecay(m, { xlab: 'layer index', unit: 'layer' });
INTERACTIVE['Gradient decay through time'] = m => gradDecay(m, { xlab: 'steps back in time', unit: 'step' });

/* ── DimRed · L3 / GenAI · L3 — LoRA rank vs parameter count ─────── */
INTERACTIVE['LoRA rank vs parameter count'] = (mount) => {
  const out = readout(mount);
  const bars = document.createElement('div'); bars.style.cssText = 'display:flex;flex-direction:column;gap:8px;font-family:var(--font-ui);font-size:12px';
  mount.insertBefore(bars, mount.firstChild);
  ctl(mount, [
    { key: 'd', label: 'hidden size d = k', type: 'range', min: 512, max: 16384, step: 512, value: 4096, fmt: v => v },
    { key: 'r', label: 'rank r', type: 'range', min: 1, max: 256, step: 1, value: 8, fmt: v => v },
  ], s => draw(s.d, s.r));
  function draw(d, r){
    const full = d * d, lora = r * 2 * d, ratio = full / lora;
    const row = (label, v, frac, col) => `<div><div style="display:flex;justify-content:space-between"><span>${label}</span><b>${(v / 1e6).toFixed(2)} M</b></div>
      <div style="height:10px;background:var(--rule);border-radius:3px;overflow:hidden"><i style="display:block;height:100%;width:${(frac * 100).toFixed(2)}%;background:${col}"></i></div></div>`;
    bars.innerHTML = row('full fine-tune  d×k', full, 1, 'var(--c-warn)') + row('LoRA  r(d+k)', lora, lora / full, 'var(--accent)');
    out.innerHTML = `<b>${ratio.toFixed(0)}×</b> fewer trainable parameters. The ratio is d/(2r) —
      doubling d doubles the saving, doubling r only halves it, so tune <b>r first</b>.
      With B initialised to zero the adapter starts as an exact no-op on the frozen base.`;
  }
  draw(4096, 8);
};

/* ── GenAI · L1 — the Chinchilla U-curve ─────────────────────────── */
INTERACTIVE['The Chinchilla U-curve'] = (mount) => {
  const P = mkPlot(mount, { xr: [0.1, 10], yr: [2, 5], xlab: 'model size (relative)', ylab: 'final loss' });
  const curve = P.curve(() => 3, { stroke: 'var(--accent)' });
  const dot = P.dot(1, 3, 4, { fill: 'var(--accent)' });
  const out = readout(mount);
  ctl(mount, [
    { key: 'n', label: 'model size', type: 'range', min: 0.1, max: 10, step: 0.1, value: 1, fmt: v => v.toFixed(1) + '×' },
    { key: 'C', label: 'compute budget', type: 'range', min: 0.5, max: 4, step: 0.5, value: 1, fmt: v => v.toFixed(1) + '×' },
  ], s => draw(s.n, s.C));
  const loss = (n, C) => 2.4 + 0.55 / n + 0.32 * n / C;      // underfit term + data-starvation term
  function draw(n, C){
    let d = ''; for (let i = 0; i <= 120; i++){ const x = 0.1 + i / 120 * 9.9; d += (i ? 'L' : 'M') + P.X(x) + ',' + P.Y(Math.min(5, loss(x, C))) + ' '; }
    curve.setAttribute('d', d);
    dot.setAttribute('cx', P.X(n)); dot.setAttribute('cy', P.Y(Math.min(5, loss(n, C))));
    const nstar = Math.sqrt(0.55 * C / 0.32);
    out.innerHTML = `loss <b>${loss(n, C).toFixed(3)}</b> at size ${n.toFixed(1)}× ·
      the U's minimum for this budget is near <b>${nstar.toFixed(1)}×</b>.
      ${n < nstar * 0.7 ? 'Too small — underfitting.' : n > nstar * 1.4 ? 'Too big — starving on data (Gopher\'s mistake).' : 'Near compute-optimal.'}
      Raise the budget and the whole U slides down-and-right.`;
  }
  draw(1, 1);
};

/* ── GenAI · L1 — KV-cache size vs context length ────────────────── */
INTERACTIVE['KV-cache size vs. context length'] = (mount) => {
  const HEADS = { 'MHA': 1, 'GQA-8': 1 / 4, 'MQA': 1 / 32, 'MLA': 1 / 64 };
  const P = mkPlot(mount, { xr: [0, 1e6], yr: [0, 100], xlab: 'context length (tokens)', ylab: 'KV-cache (GB)' });
  const lines = {};
  Object.keys(HEADS).forEach((k, i) => { lines[k] = P.curve(c => c * 2.7e-6 * HEADS[k] * 32, { stroke: ['var(--c-warn)', 'var(--c-key)', 'var(--accent)', 'var(--c-lab)'][i], 'stroke-width': 1.6 }); });
  const dot = P.dot(32000, 0, 4, { fill: 'var(--accent)' });
  const out = readout(mount);
  const st = ctl(mount, [
    { key: 'c', label: 'context', type: 'range', min: 1000, max: 1e6, step: 1000, value: 32000, fmt: v => v >= 1e6 ? '1M' : (v / 1000) + 'K' },
    { key: 'h', label: 'attention', type: 'radio', value: 'GQA-8', options: Object.keys(HEADS) },
  ], s => draw(s.c, s.h));
  const gb = (c, h) => c * 2.7e-6 * HEADS[h] * 32;
  function draw(c, h){
    dot.setAttribute('cx', P.X(c)); dot.setAttribute('cy', P.Y(Math.min(100, gb(c, h))));
    out.innerHTML = `at <b>${c >= 1e6 ? '1M' : (c / 1000) + 'K'}</b> tokens with <b>${h}</b> → <b>${gb(c, h).toFixed(1)} GB</b> of KV-cache.
      Linear in context length, but the attention variant scales it by
      <b>${(1 / HEADS[h]).toFixed(0)}×</b> — at long context that's the difference between one GPU and a rack.`;
  }
  draw(32000, 'GQA-8');
};

/* ── GenAI · L3 / Sequential · L3 — temperature + top-p on a distribution */
INTERACTIVE['Temperature and top-p, live'] = (mount) => {
  const PROMPTS = {
    'confident (“capital of France is …”)': [4.2, 1.4, 0.9, 0.6, 0.3, 0.1, 0.0, -0.3],
    'open-ended (“the weather was …”)': [1.1, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5],
  };
  const wrap = document.createElement('div'); mount.insertBefore(wrap, mount.firstChild);
  const out = readout(mount);
  const st = ctl(mount, [
    { key: 'prompt', label: 'prompt', type: 'radio', value: Object.keys(PROMPTS)[0], options: Object.keys(PROMPTS) },
    { key: 'T', label: 'temperature', type: 'range', min: 0.1, max: 2, step: 0.05, value: 1, fmt: v => v.toFixed(2) },
    { key: 'p', label: 'top-p', type: 'range', min: 0.1, max: 1, step: 0.05, value: 0.9, fmt: v => v.toFixed(2) },
  ], s => draw(s.prompt, s.T, s.p));
  function draw(prompt, T, p){
    const logits = PROMPTS[prompt];
    const ex = logits.map(z => Math.exp(z / T)); const Z = ex.reduce((a, b) => a + b, 0);
    const probs = ex.map(e => e / Z);
    const order = probs.map((v, i) => i).sort((a, b) => probs[b] - probs[a]);
    let cum = 0; const kept = new Set();
    for (const i of order){ kept.add(i); cum += probs[i]; if (cum >= p) break; }
    wrap.innerHTML = probs.map((v, i) => `<div style="display:flex;align-items:center;gap:8px;margin:2px 0;font-family:var(--font-ui);font-size:11px">
      <span style="width:22px;color:var(--ink-3)">t${i + 1}</span>
      <div style="flex:1;height:12px;background:var(--rule);border-radius:3px;overflow:hidden">
        <i style="display:block;height:100%;width:${(v * 100).toFixed(1)}%;background:${kept.has(i) ? 'var(--accent)' : 'var(--rule-2)'}"></i></div>
      <span style="width:38px;text-align:right;font-variant-numeric:tabular-nums">${(v * 100).toFixed(1)}%</span></div>`).join('');
    out.innerHTML = `T = <b>${T.toFixed(2)}</b>, top-p = <b>${p.toFixed(2)}</b> keeps <b>${kept.size} of ${probs.length}</b> tokens.
      Same p, different prompt → a different token count: top-p fixes a probability <em>mass</em>, not a count, so it adapts to how sure the model already is.`;
  }
  draw(Object.keys(PROMPTS)[0], 1, 0.9);
};

/* ── GenAI · L3 — the quantization cliff ─────────────────────────── */
INTERACTIVE['The quantization cliff'] = (mount) => {
  const P = mkPlot(mount, { xr: [2, 16], yr: [0, 4], xlab: 'bits per weight', ylab: 'perplexity increase vs FP16' });
  P.curve(b => Math.max(0, Math.pow(Math.max(0, 4 - b), 2) * 0.6), { stroke: 'var(--c-warn)' });
  const dot = P.dot(4, 0, 4, { fill: 'var(--accent)' });
  const out = readout(mount);
  ctl(mount, [{ key: 'b', label: 'bits', type: 'range', min: 2, max: 16, step: 0.5, value: 4, fmt: v => v }], s => draw(s.b));
  function draw(b){
    const dppl = Math.max(0, Math.pow(Math.max(0, 4 - b), 2) * 0.6);
    const size = (13 * b / 16).toFixed(1);
    dot.setAttribute('cx', P.X(b)); dot.setAttribute('cy', P.Y(Math.min(4, dppl)));
    out.innerHTML = `<b>${b}</b>-bit → model ≈ <b>${size} GB</b>, perplexity <b>+${dppl.toFixed(2)}</b> vs FP16.
      ${b >= 4 ? 'Flat — memory savings almost for free down to ~4 bits.' : 'Past the cliff — every further bit now costs real accuracy.'}
      The engineering call is knowing <em>where</em> the cliff sits, not just that quantization "helps".`;
  }
  draw(4);
};

/* ── GenAI · L4 — forward diffusion, scrub t ─────────────────────── */
INTERACTIVE['Diffusion step t, from clean image to pure noise'] = (mount) => {
  const G = 16, cell = 12;
  const svg = el('svg', { viewBox: `0 0 ${G * cell} ${G * cell}`, width: G * cell, height: G * cell });
  // a simple recognisable shape: a diagonal gradient disc
  const base = [];
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++){
    const r = Math.hypot(x - G / 2 + .5, y - G / 2 + .5);
    base.push(r < G * 0.36 ? 0.82 : 0.12);
  }
  const rects = base.map((v, i) => el('rect', { x: (i % G) * cell, y: ((i / G) | 0) * cell, width: cell, height: cell, fill: '#000' }, svg));
  mount.appendChild(svg);
  const out = readout(mount);
  ctl(mount, [{ key: 't', label: 'noise level  t', type: 'range', min: 0, max: 100, step: 1, value: 0, fmt: v => v + '%' }], s => draw(s.t));
  const rand = i => (Math.sin(i * 999.13 + 7.7) * 43758.5453) % 1;
  function draw(t){
    const a = t / 100;
    base.forEach((v, i) => {
      const n = 0.5 + 0.5 * Math.sin(i * 12.9898 + t * 0.6);   // per-frame pseudo-noise
      const g = Math.round(255 * ((1 - a) * v + a * n));
      rects[i].setAttribute('fill', `rgb(${g},${g},${g})`);
    });
    out.innerHTML = `t = <b>${t}%</b> — ${t === 0 ? 'the clean image x₀.' : t >= 99 ? 'pure Gaussian static x_T.' : 'a well-defined point x_t on one fixed path.'}
      The forward process has no parameters, so every x_t is reproducible; the reverse model's whole job is to undo one step of this.`;
  }
  draw(0);
};

/* ── GenAI · L4 — classifier-free guidance scale ─────────────────── */
INTERACTIVE['Guidance scale w'] = (mount) => {
  const P = mkPlot(mount, { xr: [0, 14], yr: [0, 100], xlab: 'guidance scale  w', ylab: '%' });
  const sig = (x, k, m) => 100 / (1 + Math.exp(-k * (x - m)));
  P.curve(w => sig(w, 0.6, 3), { stroke: 'var(--accent)' });        // adherence
  P.curve(w => 100 - sig(w, 0.55, 4), { stroke: 'var(--c-int)' });  // diversity
  P.seg(6, 0, 6, 100, { stroke: 'var(--rule-2)', 'stroke-dasharray': '2 2' });
  P.seg(10, 0, 10, 100, { stroke: 'var(--rule-2)', 'stroke-dasharray': '2 2' });
  const dA = P.dot(3, 50, 4, { fill: 'var(--accent)' }), dD = P.dot(3, 50, 4, { fill: 'var(--c-int)' });
  const out = readout(mount);
  ctl(mount, [{ key: 'w', label: 'w', type: 'range', min: 0, max: 14, step: 0.5, value: 3, fmt: v => v.toFixed(1) }], s => draw(s.w));
  function draw(w){
    const a = sig(w, 0.6, 3), d = 100 - sig(w, 0.55, 4);
    dA.setAttribute('cx', P.X(w)); dA.setAttribute('cy', P.Y(a));
    dD.setAttribute('cx', P.X(w)); dD.setAttribute('cy', P.Y(d));
    out.innerHTML = `w = <b>${w.toFixed(1)}</b> — <span style="color:var(--accent)">prompt adherence ${a.toFixed(0)}%</span>,
      <span style="color:var(--c-int)">diversity ${d.toFixed(0)}%</span>.
      They cross near w ≈ 3 and the sweet spot is ~6–10. They're one effect read two ways — past the band you buy adherence you can see and lose diversity you can see.`;
  }
  draw(3);
};

/* ── Sequential · L1 — TF-IDF weight vs document frequency ───────── */
INTERACTIVE['TF-IDF Weight as Document Frequency Changes'] = (mount) => {
  const N = 1000, TF = 3;
  const P = mkPlot(mount, { xr: [1, 1000], yr: [0, 25], xlab: 'documents containing the term  nₜ', ylab: 'tf-idf  (tf = 3)' });
  P.curve(n => TF * Math.log(N / n), { stroke: 'var(--accent)' });
  const dot = P.dot(5, 0, 4, { fill: 'var(--accent)' });
  P.dot(990, TF * Math.log(N / 990), 3, { fill: 'var(--c-warn)' });   // "the"
  P.dot(5, TF * Math.log(N / 5), 3, { fill: 'var(--c-lab)' });        // "fish"
  const out = readout(mount);
  ctl(mount, [{ key: 'n', label: 'nₜ', type: 'range', min: 1, max: 1000, step: 1, value: 5, fmt: v => v }], s => draw(s.n));
  function draw(n){
    const idf = Math.log(N / n), w = TF * idf;
    dot.setAttribute('cx', P.X(n)); dot.setAttribute('cy', P.Y(Math.min(25, w)));
    out.innerHTML = `nₜ = <b>${n}</b> → idf = ln(1000/${n}) = <b>${idf.toFixed(2)}</b>, tf-idf = <b>${w.toFixed(2)}</b>.
      "fish" (nₜ≈5) scores <b>${(TF * Math.log(N / 5)).toFixed(1)}</b>; "the" (nₜ≈990) scores <b>${(TF * Math.log(N / 990)).toFixed(2)}</b> —
      idf swings the weight by orders of magnitude, it isn't a small correction.`;
  }
  draw(5);
};

/* ── Unsupervised · L1 — cut the dendrogram ─────────────────────── */
INTERACTIVE['Cut the dendrogram'] = (mount) => {
  // 9 leaves, a fixed Ward merge sequence (height at which each pair joins)
  const merges = [ [0, 1, 2], [3, 4, 3], [5, 6, 4], [2, 7, 6], [8, -1, 7], [0, 3, 9], [5, 8, 12], [0, 5, 16] ];
  const W = 380, H = 240, pad = 24;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  el('rect', { x: 0, y: 0, width: W, height: H, class: 'plot-bg', rx: 6 }, svg);
  const leafX = i => pad + i * (W - 2 * pad) / 8;
  const Y = h => H - pad - h / 17 * (H - 2 * pad);
  // draw simple connectors bottom-up
  const pos = {}; for (let i = 0; i < 9; i++) pos[i] = { x: leafX(i), y: H - pad };
  let cid = 9;
  merges.forEach(([a, b, h]) => {
    const A = pos[a], B = pos[b] || A;
    const nx = (A.x + B.x) / 2, ny = Y(h);
    el('path', { d: `M${A.x},${A.y} V${ny} H${B.x} V${B.y}`, fill: 'none', stroke: 'var(--ink-3)', 'stroke-width': 1.4 }, svg);
    pos[cid++] = { x: nx, y: ny };
  });
  for (let i = 0; i < 9; i++){ const t = el('text', { x: leafX(i), y: H - 6, class: 'plot-label', 'text-anchor': 'middle' }, svg); t.textContent = String.fromCharCode(65 + i); }
  const cutLine = el('line', { x1: pad, x2: W - pad, y1: Y(10), y2: Y(10), stroke: 'var(--accent)', 'stroke-width': 2 }, svg);
  mount.appendChild(svg);
  const out = readout(mount);
  ctl(mount, [{ key: 'h', label: 'cut height', type: 'range', min: 1, max: 17, step: 0.5, value: 10, fmt: v => v.toFixed(1) }], s => draw(s.h));
  function draw(h){
    cutLine.setAttribute('y1', Y(h)); cutLine.setAttribute('y2', Y(h));
    const k = 9 - merges.filter(m => m[2] <= h).length;
    out.innerHTML = `cut at height <b>${h.toFixed(1)}</b> → <b>${Math.max(1, k)}</b> clusters.
      Nothing is recomputed — every K from 1 (above the root) to 9 (below every leaf) is already in this one tree,
      unlike K-Means' "re-run for every K".`;
  }
  draw(10);
};

/* ── RL · L1 — γ and the effective horizon ──────────────────────── */
INTERACTIVE["γ and the Agent's Time Horizon"] = (mount) => {
  const P = mkPlot(mount, { xr: [0, 60], yr: [0, 1], xlab: 'steps into the future  t', ylab: 'weight  γᵗ' });
  const curve = P.curve(t => Math.pow(0.9, t), { stroke: 'var(--accent)' });
  const mk = P.seg(10, 0, 10, 1, { stroke: 'var(--rule-2)', 'stroke-dasharray': '2 2' });
  const out = readout(mount);
  ctl(mount, [{ key: 'g', label: 'γ', type: 'range', min: 0, max: 0.99, step: 0.01, value: 0.9, fmt: v => v.toFixed(2) }], s => draw(s.g));
  function draw(g){
    let d = ''; for (let i = 0; i <= 60; i++) d += (i ? 'L' : 'M') + P.X(i) + ',' + P.Y(Math.pow(g, i)) + ' ';
    curve.setAttribute('d', d);
    const hor = g < 1 ? 1 / (1 - g) : Infinity;
    mk.setAttribute('x1', P.X(Math.min(60, hor))); mk.setAttribute('x2', P.X(Math.min(60, hor)));
    out.innerHTML = `γ = <b>${g.toFixed(2)}</b> → effective horizon 1/(1−γ) ≈ <b>${isFinite(hor) ? hor.toFixed(0) : '∞'}</b> steps.
      A reward ${Math.round(hor)} steps away is already discounted to ~37% of its face value.
      ${g < 0.5 ? 'Near-sighted: only the next move matters.' : g > 0.97 ? 'Far-sighted — and slow, high-variance to learn.' : ''}`;
  }
  draw(0.9);
};

/* ── DimRed · L1 — volume of a ball inside its cube ─────────────── */
function lgamma(x){
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1; let a = c[0]; const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
INTERACTIVE['Ball inside a cube, as dimension grows'] = (mount) => {
  const P = mkPlot(mount, { xr: [1, 20], yr: [-8, 0], xlab: 'dimension  d', ylab: 'log₁₀ (ball ÷ cube)' });
  const ratio = d => Math.exp((d / 2) * Math.log(Math.PI) - lgamma(d / 2 + 1)) / Math.pow(2, d);
  P.curve(d => Math.log10(ratio(d)), { stroke: 'var(--accent)' });
  const dot = P.dot(3, Math.log10(ratio(3)), 4, { fill: 'var(--accent)' });
  const out = readout(mount);
  ctl(mount, [{ key: 'd', label: 'dimension', type: 'range', min: 1, max: 20, step: 1, value: 3, fmt: v => v }], s => draw(s.d));
  function draw(d){
    const r = ratio(d);
    dot.setAttribute('cx', P.X(d)); dot.setAttribute('cy', P.Y(Math.max(-8, Math.log10(r))));
    out.innerHTML = `d = <b>${d}</b> — the inscribed ball fills <b>${r < 1e-3 ? r.toExponential(1) : (r * 100).toFixed(1) + '%'}</b> of the cube.
      By d = 10 it's under 0.25%; essentially all the cube's volume is in the corners the ball never reaches.
      That emptiness is the curse of dimensionality in one number.`;
  }
  draw(3);
};

/* ── DimRed · L1 — distance concentration ───────────────────────── */
INTERACTIVE['Distances concentrate as dimension grows'] = (mount) => {
  const P = mkPlot(mount, { xr: [1, 100], yr: [0, 1], xlab: 'dimension  d', ylab: '(max − min) ÷ min distance' });
  P.curve(d => 1.8 / Math.sqrt(d), { stroke: 'var(--accent)' });
  const dot = P.dot(2, 1.8 / Math.SQRT2, 4, { fill: 'var(--accent)' });
  const out = readout(mount);
  ctl(mount, [{ key: 'd', label: 'dimension', type: 'range', min: 2, max: 100, step: 1, value: 2, fmt: v => v }], s => draw(s.d));
  function draw(d){
    const c = 1.8 / Math.sqrt(d);
    dot.setAttribute('cx', P.X(d)); dot.setAttribute('cy', P.Y(Math.min(1, c)));
    out.innerHTML = `d = <b>${d}</b> — nearest and farthest point differ by only <b>${(c * 100).toFixed(0)}%</b> of the near distance.
      As d grows every pair of points drifts toward the same distance, so "nearest neighbour" stops meaning much — the ~1/√d shape is why.`;
  }
  draw(2);
};

/* ── aliases: same figure, different lecture ─────────────────────── */
alias('The Vanishing Gradient, Forward and Backward', 'Gradient decay through time');
alias('Sampling Strategy Comparison', 'Temperature and top-p, live');
alias('QLoRA memory picker', 'LoRA rank vs parameter count');
alias('Truncating a Matryoshka embedding', 'The quantization cliff');

window.INTERACTIVE = INTERACTIVE;
