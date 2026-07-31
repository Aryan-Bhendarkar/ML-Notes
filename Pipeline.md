# PIPELINE.md — Amazon MLSS Notes Pipeline

Claude Code: this file is your complete instruction set. Read it fully, then
execute all four phases for the lecture you were given.

---

## What you are building

The student is learning machine learning **for the first time**. They have
programming ability but **no prior ML knowledge whatsoever**. They watched a
50-minute lecture without pausing and captured screenshots of the slides.

Your job: turn those screenshots into a markdown study document so complete and
so clearly explained that the student can learn the entire lecture from the
notes alone, without rewatching, without googling terms, and without any prior
background.

These markdown files become an interactive notes site in a later phase. Write
clean, well-structured markdown — but focus your effort on **teaching quality**,
not formatting cleverness.

---

## Invocation

You will be told which lecture to process, e.g.:

> Execute PIPELINE.md for `topic-01-scaling-laws/lecture-02`

If the topic folder doesn't exist yet, create it. If screenshots are sitting
loose in `inbox/`, ask which topic and lecture they belong to before proceeding.

---

## Phase 1 — Compile screenshots into a PDF

**Input:** `raw/<topic>/<lecture>/` containing screenshot image files
**Output:** `pdfs/<topic>/<lecture>.pdf`

Steps:

1. **List the images and determine correct order.** This is the step that most
   often goes wrong. Screenshot filenames are usually epoch timestamps
   (`1785390208031_image.png`) or sequential (`Screenshot 2026-07-31 at
   14.03.22.png`). Sort **numerically by timestamp**, not lexicographically —
   naive string sort puts `10` before `2`.

2. **Verify the order before converting.** Read the first, middle, and last
   images. Confirm they form a sensible progression (title slide → content →
   summary). If the order looks wrong, sort by file mtime instead and re-verify.
   Report what ordering method you used.

3. **Deduplicate.** Video screenshots often capture the same slide multiple
   times. Compare adjacent images; if two are near-identical, keep one. Report
   how many duplicates you removed.

4. **Convert to PDF.** Prefer `img2pdf` (lossless, no re-encoding):
   ```bash
   img2pdf --output pdfs/<topic>/<lecture>.pdf <sorted images>
   ```
   Fall back to ImageMagick if unavailable:
   ```bash
   magick <sorted images> pdfs/<topic>/<lecture>.pdf
   ```
   Install with `pip install img2pdf` if neither exists.

5. **Confirm page count** matches the deduplicated image count.

---

## Phase 2 — Read and inventory the slides

**Output:** an internal working list. Do not write this to a file.

1. Read every page of the PDF.
2. Build a list of every **distinct concept, term, formula, diagram, and named
   result** that appears anywhere in the deck.
3. For each item, note whether the slide **explains** it or merely **mentions**
   it. Anything merely mentioned still gets a full explanation in the notes —
   the slide assuming knowledge is exactly the gap you exist to fill.
4. Identify the **prerequisite chain**: which concepts must be understood before
   which others. Your notes will follow this order, not slide order.

Also read these if they exist:
- `raw/<topic>/<lecture>/transcript.txt` — the instructor's spoken words. Treat
  this as **primary source material**, not a supplement. Slides are the
  skeleton; the transcript carries the intuition, caveats, and asides that make
  concepts click.
- `raw/<topic>/<lecture>/confusions.txt` — things the student found confusing
  while watching. Every item here gets extended, explicit treatment.

---

## Phase 3 — Write the notes

**Output:** `notes/<topic>/<lecture>.md`

### The teaching contract

This is the most important section of this file. Follow it exactly.

**1. Zero assumed knowledge.** The student does not know what a gradient is,
what a token is, what "training" means, or what a parameter is. If a word is
ML-specific, it gets defined the first time it appears — before it is used in a
sentence that assumes understanding.

**2. Every term gets the four-beat treatment on first appearance:**

> **Term** — one-sentence plain-English definition, no jargon inside it.
>
> *In everyday words:* an analogy or comparison to something familiar.
>
> *Concretely:* a specific, tiny example with real values.
>
> *Why it exists:* what problem it solves, what people did before it.

**3. Build in layers, never dump.** Introduce the simplest possible version of
an idea first, confirm it makes sense, then add the complication. Never present
the full formal version first and explain afterward.

The pattern is: **intuition → tiny example → general rule → formal notation →
edge cases**. Every concept walks that ladder.

**4. Explain equations in words before showing symbols.** Write what the
equation *says* as an English sentence first. Then show it. Then define every
single symbol in a table. Never show notation the student hasn't been walked
through.

Example of the required standard:

> The formula says: **the loss you'll get is determined by how much compute you
> spend, and it shrinks by a predictable fraction every time you multiply your
> compute.**
>
> $$L(C) \approx \left(\frac{C_{min}}{C}\right)^{\alpha}$$
>
> | Symbol | Read it as | What it means |
> |---|---|---|
> | $L$ | "loss" | How wrong the model is on average. Lower is better. |
> | $C$ | "compute" | Total arithmetic spent on training, measured in FLOPs. |
> | ... | | |

**5. Teach prerequisites the slide skipped.** If a slide says "the model
minimizes cross-entropy loss" and never explains cross-entropy, you explain
cross-entropy. Mark these clearly:

> 📚 **Background the slide assumed** — [concept]
> [full explanation]

**6. Simple language throughout.** Short sentences. Common words. If a technical
word is unavoidable, define it inline. Write as if explaining to a smart friend
who has never taken an ML course — not as if writing a textbook.

**7. No hand-waving.** Never write "this is beyond our scope", "intuitively it
works out", or "for now just accept that". If something is genuinely hard,
spend more words on it, not fewer. Hard concepts are the entire reason these
notes exist.

### Required document structure

```markdown
---
title: <lecture title>
topic: <topic slug>
lecture: <NN>
---

# <Lecture Title>

## What you'll understand after reading this
5-8 bullets, written as capabilities: "You'll be able to explain why...",
"You'll be able to calculate...". Not topic names — actual abilities.

## Before we start: what you need to know
Every prerequisite concept, fully taught. If the lecture assumes five things
the student doesn't know, teach all five here. This section can be long.
That is correct and expected.

## The big picture
In plain language, before any detail: what is this lecture actually about,
and why does it matter? Someone who reads only this section should understand
the point of the lecture.

## <Concept 1>
[four-beat treatment → build up → worked example → formal version]

### Worked example
Real numbers, computed step by step to a real answer. Show every step.
Never trail off with "and so on".

### Where people get confused
Specific misunderstandings, phrased as "you might think X — actually Y".

### Why this matters
Where this shows up in real ML systems.

## <Concept 2>
...

## Putting it together
How every concept in this lecture connects. An ASCII diagram of the
dependency structure, plus a paragraph walking through it.

## Glossary
Every term introduced in this lecture, alphabetical, one line each.
This is the student's quick-reference layer.

## Check yourself
12 questions, easy → hard. At least 3 requiring the student to combine
two concepts. Answers in a collapsed block:

<details><summary>Answers</summary>
...
</details>

## Going deeper
Papers and resources, ranked by importance, each with one line on why
it's worth reading and how hard it is.
```

### Formatting rules

- Math in LaTeX: `$inline$`, `$$display$$`. Required — the site renders KaTeX.
- **Bold** every term and number that must be remembered.
- Tables for comparisons and symbol definitions.
- Code blocks for algorithms, worked calculations, and step-by-step derivations.
- Callouts: `> 📚` for background, `> ⚠️` for uncertain content, `> 💡` for
  key insights.

### Honesty rules

- If a slide is unreadable or ambiguous, write
  `> ⚠️ Slide N unclear — verify against the recording` rather than guessing.
- If you are not confident in a fact, constant, date, or paper attribution, mark
  it `> ⚠️ verify this`. Confident fabrication is the worst possible failure
  here — the student cannot yet tell when you're wrong.
- If a slide oversimplifies or is now outdated, say so explicitly.
- If the field genuinely disagrees on something, present the disagreement.

### Length

Set by what teaching from zero actually requires — typically 3,000–6,000 words.
Do not pad, and do not compress. If the notes are under 2,500 words, you have
almost certainly summarized instead of taught.

---

## Phase 4 — Self-review and revise

After writing, re-read your own output against this checklist and **edit the
file in place** to fix failures. Do not write a report.

| # | Check | Fail if |
|---|---|---|
| 1 | Zero unexplained jargon | Any ML term used before it's defined |
| 2 | Equations explained in words first | Any formula appears before its plain-English version |
| 3 | Every symbol defined | Any variable in any equation lacks a table entry |
| 4 | Worked examples complete | Any example trails off without a final number |
| 5 | Prerequisites taught | Any assumed concept left unexplained |
| 6 | Layered build-up | Any concept presented formally before intuitively |
| 7 | Confusions addressed | Any item in `confusions.txt` not covered in depth |
| 8 | No hand-waving | Any "beyond scope" / "just accept" phrasing present |
| 9 | Uncertainty flagged | Any shaky claim stated confidently |
| 10 | Beginner-readable | Any paragraph requiring prior ML knowledge to parse |

Test for #1 and #10: reread the document imagining you know nothing about ML.
The first sentence that stops making sense marks a failure — fix it and
everything after it.

Then print a short completion summary: page count, ordering method used,
duplicates removed, word count, and any `⚠️` flags you left in the file.

---

## Repository layout

```
mlss/
├── PIPELINE.md              ← this file
├── raw/
│   └── topic-01-scaling-laws/
│       └── lecture-02/
│           ├── <screenshots>
│           ├── transcript.txt    (optional, high value)
│           └── confusions.txt    (optional, high value)
├── pdfs/
│   └── topic-01-scaling-laws/
│       └── lecture-02.pdf        ← Phase 1 output
└── notes/
    └── topic-01-scaling-laws/
        └── lecture-02.md         ← Phase 3 output
```

Topic folders: `topic-NN-slug`. Lecture folders: `lecture-NN`. The pipeline
derives all output paths from these names, so keep them exact.

---

## Scope discipline

Process **one lecture per invocation**. Do not batch multiple lectures into a
single run — slide images are token-expensive, and quality degrades noticeably
as context fills. One lecture, fresh context, full attention.

Do not build the interactive site. That is a later phase, after enough notes
exist to be worth presenting.