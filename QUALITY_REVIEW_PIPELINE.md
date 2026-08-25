# QUALITY_REVIEW_PIPELINE.md — auditing and upgrading notes to top-1% / Applied-Scientist bar

Claude Code: this file is your complete instruction set for a **quality pass over one already-written
module**. Read it fully, then execute all phases **for the one module you were given**.

> Companion to `NOTES_PIPELINE.md` (which writes a lecture's notes from slides) and
> `WEB_ARTIFACT_PIPELINE.md` (which renders a module's notes into a study environment). This file
> sits between them: it audits notes that already exist — whether written by a careful pass or a
> rushed one, by you or by another agent — against both documents' standards, compiles every gap into
> one file, and then fixes the `.md` source directly. It never touches a web artifact; if one exists
> for this module, flag it as stale in your summary so a rebuild can be scheduled.

---

## Invocation

> Run QUALITY_REVIEW_PIPELINE.md on the Sequential Learning module

**One module per invocation. Never batch.** A module is 3–4 lecture files plus a README, each
6,000–36,000 words. Reviewing two modules in one pass means skimming both instead of reading either.
If told "do all the modules," review the first module in the sequence below, apply its fixes, report
what you found and fixed, and name the next module — do not continue automatically.

**Module order** (go in this sequence unless told otherwise):

1. Supervised Learning
2. Deep Neural Networks
3. Dimensionality Reduction
4. Unsupervised Learning
5. GenAI & LLM
6. Sequential Learning
7. Causal Inference
8. Reinforcement Learning
9. Agentic AI

---

## Why this exists

A note file can be fluent, well-structured, and pass a casual read — and still fail the one thing
that matters: **it does not yet make its reader top 1% and interview-ready.** That failure mode is
invisible to a single read-through, because prose that sounds authoritative is easy to mistake for
prose that is complete. It only surfaces when someone deliberately checks the file against three
different standards, from three different vantage points, and against the actual source material.
That's this pipeline.

---

## The three lenses

Work each lecture file through all three lenses, in order, before moving to the next file. Do not
merge them into one pass — each lens catches a different failure mode, and going fast enough to skip
one is exactly how the gaps in the source material got there in the first place.

### Lens 1 — Teacher: does this actually teach?

You are auditing against `NOTES_PIPELINE.md`'s own teaching contract (re-read it now if it's not
fresh — the "reader," the "teaching contract," and the "self-review" table are the checklist here).
For every section, ask:

- Is every term explained on first use, with the four-part `> **Term** — ... *In everyday words:* ...
  *Concretely:* ... *Why it exists:* ...` pattern, or is something used before it's defined?
- Is every formula preceded by its plain-English reading, with a symbol table immediately after?
- Is every "it can be shown that" / "intuitively" / "beyond scope" phrase actually a skipped
  derivation? Derive it instead.
- Does every worked example end in an actual number, computed from real inputs — or does it wave at
  "and so on"?
- Are the interactive spec blocks (`type/title/concept/control/observe/insight/fallback`) proposed
  only where a concept is genuinely clearer *moving*, with every field filled and a real `fallback`?
- Interview prep: 8–12 questions, ranked easy→hard, at least 3 combining two concepts, each with a
  model answer you could actually say out loud? Whiteboard-ready derivations present and reproducible
  from the file alone? An applied Amazon scenario walking framing→data→model→metric→failure
  modes→what you'd ship, with explicit Leadership Principle ties?
- Glossary and Check-yourself: complete, and do the check-yourself questions actually probe
  understanding rather than recall of a definition?

### Lens 2 — Student: would *this specific reader* actually get there?

Now re-read the same file as the reader `NOTES_PIPELINE.md` defines: a programmer with basic ML
familiarity, aiming to defend any concept under interview pressure and land an Amazon Applied
Scientist Internship — i.e., **top 1%, not "passed the course."** Simulate being that reader, honestly:

- At every "you already know this" or brisk transition — do you, actually? Or would a real reader at
  this level stall here?
- Where a concept builds on an earlier one, is the dependency satisfied *in this file* (directly, or
  by an explicit, correct cross-reference), or does it silently assume something never taught?
- Is depth actually spent where the hard ideas are, or is coverage spread evenly so the genuinely
  difficult 20% gets the same treatment as the easy 80%? (Depth beats coverage — flag sections that
  are conspicuously thin relative to how hard the underlying idea is.)
- After reading the whole file cold, could you defend the material's central 2–3 claims under
  follow-up questioning ("why does that hold?", "when does it fail?", "what would you do instead?")
  using only what's in the file? If not, that's a gap — write down the specific question that would
  expose it.
- Does the file connect to *practice* — where a concept shows up in a real system, what breaks when
  it's done wrong, what a practitioner actually types — or does it stay abstract throughout?

### Lens 3 — Engineer: is it correct, complete, and ready to build on?

This is the fidelity and craftsmanship pass — the one a subject-matter reviewer, not a learner, would
run.

**Source fidelity — reopen the actual slides.** Never trust that the existing `.md` faithfully
reflects the deck. For each lecture:

- Confirm the file's own header claims which capture it used (`output/` raw frames vs.
  `slides_deduped/`). If it says `slides_deduped/`, or if the raw frame count in `output/<Lecture>/`
  is more than the file's frontmatter `slides:` count would suggest, **re-audit against the raw
  frames** per project memory `slides-deduped-is-lossy` — dedup and coarse OCR-only drafts have both
  measurably dropped named methods, benchmarks, and worked examples in this course before.
- Build contact sheets (see `contact_sheet.py`) across the full raw frame range and scan every one.
  Cross-check the file's own "Putting it together" / closing-summary section and any named
  method/algorithm/benchmark against what the deck's *own* closing or agenda slides name — a method
  the lecture names in its summary but never explains in the body is a real, recurring failure mode
  in this course (found repeatedly: a whole missing architecture, a missing bandit algorithm, an
  algorithm used throughout but never defined, missing benchmarks).
- Verify every citation (author, year, venue) against the actual slide footer or in-slide text, not
  from memory. A plausible-sounding but wrong citation is worse than an honestly-flagged unknown one
  — check project history for a concrete instance of exactly this failure before assuming citations
  in an existing file are safe.
- Verify every precise-looking number (thresholds, hyperparameters, benchmark scores, demo results)
  actually appears on a slide. A specific number with no slide source is a fabrication risk — either
  find its source or flag/replace it, never let invented precision stand uncaveated.
- Check every derivation and every "why X works" explanation *against its own stated formula* —
  re-derive it yourself and confirm the prose's claimed direction/mechanism actually follows from the
  math written two lines above it. This project has caught a mechanism explanation that flatly
  contradicted its own formula; assume that's possible anywhere until you've re-derived it.

**Internal consistency.** Stage counts, section numbering, and cross-references inside the file
should be self-consistent (a "five-stage procedure" that lists four stages is a bug independent of
what the slide says). Duplicated paragraphs, orphaned headings, and broken internal "§N" references
are bugs to fix on sight.

**Web-format readiness — read `WEB_ARTIFACT_PIPELINE.md`'s fidelity and math sections now.** The
markdown you're auditing is the *source of truth* for that build, so gaps here become artifact bugs
later. Specifically check:

- **LaTeX escaping.** Grep the file for a literal double backslash before a command (`\\tau`,
  `\\mathbb`, etc.) — this is a real, previously-found bug class that renders every formula broken.
  Every LaTeX command must be single-backslash; the *only* legitimate double-backslash is a genuine
  row-break inside `cases`/`bmatrix`/`align`. Use the surgical fix pattern (protect quadruple
  backslashes as a placeholder, halve doubles, restore) rather than a blind find-replace — verify
  before and after on a `grep -c` count.
- **Symbol tables bound to their formula.** Every `| Symbol | Read it as | What it means |` table
  must sit *immediately* after the formula it defines, with no unrelated prose between them — the
  artifact treats these as a bound unit, so if the markdown separates them, the artifact will too.
- **Callout emoji used per the fixed semantic map** (`📚` background, `💡` key insight, `⚠️` careful,
  `🧪` worked example, `🎯` interview, `🔬` research) — an emoji used outside its assigned meaning
  breaks the artifact's colour-coding, so correct any drift.
- **Cross-references phrased consistently** as `Part N §M` or `§M.M` — these become real links in the
  artifact; an inconsistent or ambiguous reference (e.g., a section number that doesn't exist, or
  prose like "as covered earlier" with no anchor) becomes a broken or missing link.
- **ASCII diagrams complete and accurate.** These get redrawn as SVG verbatim — if a diagram is
  wrong, mislabeled, or missing a state the prose describes, fix it in the markdown now rather than
  letting a later artifact build inherit the error.
- **`interactive` spec blocks well-formed** — every field present, `fallback` genuinely sufficient on
  its own (the artifact renders these as placeholder cards for now; the fallback text is what a
  reader actually sees).
- **Heading structure exportable.** Every `##`/`###` should be one the artifact's per-section
  checkbox and scrollspy outline can bind to — avoid headings that are pure decoration with no real
  section content beneath them, and don't let two sections share an ambiguous title within one file.

---

## Phase 1 — Read everything before writing anything

Read the module's `README.md` and every lecture `.md` in full. Do not start compiling findings from a
skim — Lens 2 in particular requires having actually followed the material as a reader would.

## Phase 2 — Run all three lenses per lecture, compile into one file

For each lecture file in the module, run Lens 1, then Lens 2, then Lens 3, taking notes as you go.
Then compile everything into a **single findings file** at the module root:
`notes/<Module>/QUALITY_REVIEW.md`. Structure it exactly like this project's established review
format:

```markdown
# Quality review — <Module> — [date]

Purpose / severity legend (reuse: 🔴 factual error or fabrication · 🟠 real content/pedagogy gap ·
🟡 polish / web-readiness).

## `notes/<Module>/<lecture-file>.md` (Lecture NN)

Source cross-checked: `output/<Lecture>/` (N raw frames), contact sheets + full-res spot checks on
slides [list].

### 🔴 ...
### 🟠 ...
### 🟡 ...
### Verified accurate / no action needed
### Not yet checked (if any, with a reason)

**Overall verdict:** ...
```

Cite section numbers, exact quotes, exact numbers, and exact slide numbers for every finding — every
item must be actionable without re-deriving anything. This file is the deliverable of Phases 1–2 and
must exist, complete, before Phase 3 starts.

## Phase 3 — Apply every finding

Work through the compiled findings file top to bottom, applying each fix directly to the lecture
`.md` (and the module `README.md` where a finding affects it — instructor names, word counts, section
tables, capture-quality claims). For 🔴 items, do not just patch the symptom — re-derive or re-verify
against the source before writing the fix, the way you would for a fresh `NOTES_PIPELINE.md` pass. For
🟡 web-readiness items, prefer a scripted, verifiable fix (e.g. the LaTeX-escaping repair pattern) over
a manual one wherever the issue is mechanical and repeats many times in one file.

After applying fixes, re-run the specific check that found each 🔴 and 🟠 item to confirm it's
actually resolved — do not mark an item fixed on the strength of having edited near it.

## Phase 4 — Update the module README and mark the review complete

Update `notes/<Module>/README.md`: instructor names, word counts, section tables, and capture-quality
notes should all reflect the post-fix state. Add a one-line pointer to `QUALITY_REVIEW.md` if it isn't
already linked. Mark the top of `QUALITY_REVIEW.md` with a completion banner (mirror the pattern used
in this project's `REVIEW_SCRATCH.md`) so it reads as an audit trail, not a pending list, for anyone
who opens it later.

---

## Self-review

Before reporting the module done, confirm:

| # | Check | Fail if |
|---|---|---|
| 1 | Every lecture in the module went through all three lenses | Any lecture file skipped a lens |
| 2 | Every 🔴 finding was re-verified against the actual slide, not memory | Any fix based on assumption rather than a re-checked source |
| 3 | Every fabricated-looking number/citation was resolved (fixed or honestly caveated) | Any number/citation left stated with unwarranted confidence |
| 4 | No `\\command` (double-backslash LaTeX) remains outside genuine matrix/cases row-breaks | `grep`/regex check still finds one |
| 5 | Every symbol table still sits directly under its formula | Any got separated during editing |
| 6 | Every internal `§N` cross-reference resolves to a real section | Any reference is stale after edits |
| 7 | The module README's instructor names, word counts, and section tables match the post-fix files | Any is stale |
| 8 | `QUALITY_REVIEW.md` exists, is complete, and is marked done | Missing, partial, or still phrased as a to-do list |
| 9 | Nothing was silently dropped to "simplify" — a finding was fixed or explicitly deferred with a reason, never ignored | Any compiled finding has no corresponding action |

---

## Honesty rules (inherited from `NOTES_PIPELINE.md`, restated because this pass will find violations of them)

- Never let a confident-sounding number or citation stand once you can't verify it against the slide
  — mark it `⚠️ verify this` or fix it. A confidently wrong fact is the worst outcome this pipeline
  can produce, worse than an honestly flagged gap.
- Never silently reconstruct missing content as if it were captured. If a gap is genuinely
  unrecoverable from the raw frames, say so explicitly, the way the existing notes already do
  elsewhere in this course.
- Never mark a finding "fixed" without re-checking it. "I edited that paragraph" is not the same as
  "I confirmed the number now matches the slide."

## Scope discipline

One module per invocation. Do not start the web artifact build in this pass — that's
`WEB_ARTIFACT_PIPELINE.md`'s job, and it should only run *after* this pipeline has cleared a module,
since it treats the markdown as ground truth and will faithfully reproduce whatever gaps are still in
it. If you finish a module with time/budget remaining, report completion and name the next module in
the sequence rather than continuing into it unprompted.
