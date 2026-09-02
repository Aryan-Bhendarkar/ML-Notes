# web/ — the study platform build

Replaces the old `build.py` + `template.html` + in-browser markdown/math parser.
The markdown in `notes/<Module>/` stays the single source of truth; this pipeline
only renders it. **Content fidelity is the contract** (see `WEB_ARTIFACT_PIPELINE.md`):
nothing dropped, nothing paraphrased, every number verbatim, every ⚠️ kept and made
*more* visible.

## What it produces

```
docs/index.html                         ← the course home / library (all 9 modules)
docs/<module-slug>.html           ← one self-contained file per module
```

Each module file is a single HTML with all CSS + JS + math assets inlined — no
external requests, works from `file://`. The nine files share one `localStorage`
namespace (`mlss:*`) so the home page can add up progress across the whole course.

## Build

```
cd web
npm install
node build.mjs "Supervised Learning"     # build one module (+ refresh the home)
node build.mjs --home                     # refresh only the home
```

**One module per invocation.** Each is 30–120k words; quality of the interactive
work and the diagram redraws collapses if you batch.

The build prints a fidelity report — source vs. rendered counts for `##` sections,
`<details>` blocks, table rows, ⚠️ flags and `interactive` blocks. They must match.

## How it renders

| Concern | Choice | Why |
|---|---|---|
| Markdown | `markdown-it` (+ `markdown-it-deflist`) at build time | maintained; no fragile in-browser regex parser |
| Math | `temml` → MathML, rendered at build time, inlined | LaTeX→MathML by a KaTeX author, **no font files**, MathML Core is native in current browsers, zero runtime cost |
| Callouts | fixed token map 📚💡⚠️🧪🎯🔬🩹 → `callout-{bg,key,warn,lab,int,res,recon}` | one colour = one meaning in every module (the memory aid) |
| Symbol tables | `\| Symbol \| …` tables get `.symtab` and bind to the equation directly above | a scroll must never separate a formula from what its symbols mean |
| Diagrams | ```` ```mermaid ```` blocks → static SVG, pre-rendered by `diagrams.mjs` (Playwright + Mermaid), cached in `web/.diagrams/`, inlined at build time | real layout engine, themed to study-lamp, **zero runtime cost**; missing cache ⇒ graceful `<pre>` fallback |
| ASCII diagrams | leftover box-drawing blocks, styled `figure.ascii` | being converted to ```` ```mermaid ```` module by module |
| Cross-refs | `§14`, `Part 2 §6` → in-artifact links (works even mid-sentence beside math/code); `Supervised Learning Part 1 §8` → link to that module's file; unresolvable → styled dead link | never a broken anchor |
| `interactive` blocks | real components where feasible (`web/interactive.js`), keyed by `title:` | the rest render the `fallback` as complete standalone prose — never an inert stub |

## Design — "Study lamp"

One mode. Warm charcoal ground (`#1B1A17`, never blue-black), warm paper-grey
text (`#E5DED1`, ~12:1), a single muted-sage accent (`#A6C9A8`) used only for
structure — section numbers, your position, boxed results, progress. Links are a
separate desaturated slate-blue so a link never reads as a heading. Serif body
(`Charter`/`Sitka Text`) at 19px/1.75, 64ch. The seven callout hues keep one
fixed meaning across every module. Highlighter palette: gold / teal / coral /
lilac (pastels bright enough to read on the dark ground).

Highlights are stored as a character offset + length into the section's text and
re-wrapped across every text node the span touches on load, so a selection that
crosses a `**bold**` run (or any element boundary) highlights correctly.

### Layout
- **Left rail = the course spine** and nothing else: the module's lectures with
  progress bars, the current lecture's sections nested and grouped under the
  deck's own "Part N" dividers, then the other 8 modules collapsed. Replaces the
  old header dots + separate module list + right-rail outline.
- **Right rail = your work only**: the phase chips, your highlights, your notes.
- Reading column: one title (the doc's own `#`/`###` are folded into the header),
  section numbers hang in the left margin, part dividers are landmarks.
- Header: `‹ module` · a slim module progress bar · focus · search · work · help.

### Tracking — the reading guide's phases
*First pass* is the `##` checkboxes rolled up to a %. *Second pass* opens a
filtered view — just this lecture's 🧪 examples, solutions blanked. *Interview
prep* opens the *Putting it together* + *Interview prep* + *Check yourself*
sections as one set. No streaks, no points, no badges.

### Spaced review (`app.js`)
Every glossary term (table **and** `- **Term** — …` list form), every
Check-yourself question, and every Interview question is auto-extracted into a
per-course flashcard deck — **~1,670 cards**. A card enters the deck once its
lecture is started (any section done). Leitner boxes 0–4 → **1, 3, 8, 21, 60**-day
intervals; *Got it* advances a box, *Shaky* → tomorrow, *No idea* → box 0 + again
this session. A full-screen review session (`R`, or the `↻` header button with a
due badge) drives due cards + up to 15 new, one at a time. `Export … for Anki`
writes a tab-separated `.txt` (math as `\(TeX\)`). State: `mlss:srs`; each module
writes `mlss:cardstats` so the home shows due/new counts without loading content.

### Pacing (home)
Course ≈ 74 h first-pass reading at 130 wpm (~162 h with second pass + interview).
A "h/week" slider (`mlss:plan`) projects finish dates; `mlss:log` records minutes
per day (from section check-offs, 120-day window) to show actual pace vs target —
honest, no guilt framing.

## Build status — all 9 modules

Every module builds with all fidelity counts matching source:
**796 `##` sections · 322 `<details>` · 5,220 table rows · 418 ⚠️ flags · 118 `interactive` blocks** —
0 Temml errors, 0 leftover placeholders, 0 JS errors across every file (jsdom smoke).

A rendering-analysis pass (`web/_analyze.mjs`, kept as a dev tool) checks the built
DOM for ~20 gap classes — unlinked `§` refs, missed callouts, raw LaTeX in
fallbacks, ragged tables, cross-module mislinks, thin sections, escaped HTML, etc.
Current result: clean except two known false positives (prose that mentions the
literal string `##`, and a source `$\sqrt{}$` shorthand).

Fixes from that pass: `§` refs now link even inside sentences that also contain
`$math$`/`code`; `Module-name Part N §M` links to that module's file; nested
`> ⚠️` callouts convert; `interactive` fallbacks render their `$math$`;
`## Part A — …` sub-dividers render as dividers and are excluded from progress %.

`web/interactive.js` implements **40** of the 118 interactive blocks as real
figures (sliders / live plots / bar charts, all inline SVG, no deps) — Supervised
Learning is fully covered. The rest render their `fallback` as complete standalone
teaching prose. To add more, add an entry to `INTERACTIVE` keyed by the block's
exact `title:`.

## Diagrams

```` ```mermaid ```` blocks in the notes are pre-rendered to themed static SVG and
inlined at build time — **no runtime library, no external requests**.

```
npm run diagrams        # render every ```mermaid block → web/.diagrams/<hash>.svg
npm run diagrams -- --force   # re-render all (after a theme change)
```

- `diagrams.mjs` uses Playwright + Mermaid; run it whenever a diagram changes and
  **commit `web/.diagrams/`** (it is the cache the build and CI read — neither
  needs a browser).
- `render.mjs` looks each block up by content hash; a miss falls back to a styled
  `<pre>` and `build.mjs --all` / CI flags it.
- Theme + layout config live in `THEME` at the top of `diagrams.mjs` — keep the
  palette in sync with `app.template.html`.
- `node _shot.mjs diagram latest` screenshots the newest cached SVG in a themed
  frame for a quick visual check; `node _shot.mjs page <slug> figure.diagram:N`
  grabs it from the built page.

## Local preview & real-browser QA

```
npm run build:all     # rebuild every module + docs/index.html
npm run preview        # serve docs/ at http://localhost:5173  (zero-dep, preview.mjs)
npm run dev            # build:all then preview, one command
```

`preview.mjs` mirrors GitHub Pages closely enough for visual QA: `/` → `index.html`,
extensionless paths resolve to `.html`, unknown paths serve `docs/404.html`.

A **Playwright MCP** server is registered at user scope (`claude mcp list` shows
`playwright`), so Claude Code can drive a real Chromium in this or any project —
navigate to the preview URL, screenshot each module, click the interactive
figures, resize for mobile — then fix and rebuild in a loop. First run downloads
the browser (~150 MB, cached after). Ask e.g. *"run npm run dev, open
localhost:5173 in the browser, screenshot every module page and the interactive
blocks, list anything visually broken, fix it, rebuild."*

To reinstall on another machine: `claude mcp add --scope user playwright -- npx -y @playwright/mcp@latest --isolated`

## Still to do (later passes)

- More interactive figures (40 / 118 done — Supervised Learning complete)
- Convert the remaining ASCII diagrams to ```` ```mermaid ```` (Supervised in progress)
- Syntax highlighting for `python` blocks (styled, not tokenised)
- Real-browser visual QA sweep across every module
