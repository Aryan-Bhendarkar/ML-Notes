# NOTES_PIPELINE.md — slides ➜ mastery-grade study notes

Claude Code: this file is your complete instruction set. Read it fully, then execute
all five phases **for the one lecture you were given**.

> Supersedes `Pipeline.md` phases 2–4. `Pipeline.md` described the old workflow that
> built notes from manually screenshotted PDFs; those PDFs were incomplete (the
> existing `notes/GenAI & LLM/genai-llm-01.md` records ~10 missing slides). Slides now
> come from a verified extraction pipeline instead. Phase 1 of `Pipeline.md`
> (screenshot ➜ PDF) is obsolete and must not be run.

---

## Invocation

> Execute NOTES_PIPELINE.md for lecture 14

One lecture per invocation. **Never batch.** Slide images are token-expensive and
teaching quality degrades badly as context fills. One lecture, fresh context, full
attention. If told to "do the whole module", process the first lecture and stop,
then report which lecture is next.

---

## The reader you are writing for

A programmer with **basic ML familiarity** — they know what a model, a dataset, and
training roughly are, and they can read Python. They are **not** a beginner and do
not need "what is a variable".

But treat *basic* as a floor, not a ceiling. Their goal is to reach the **top 1%**:
to explain any concept on these slides from first principles, defend it under
interview pressure, and land an **Amazon Applied Scientist Internship**. So:

- Every ML-specific term on a slide gets a real explanation the first time it appears.
- Every symbol in every equation gets defined.
- "You already know this" is never a reason to skip — it is a reason to be brisk,
  then go deeper than the slide did.
- Depth beats coverage. A concept explained until it genuinely clicks is worth more
  than five concepts summarised.

---

## Inputs

**Slides:** `slides_deduped/Lecture_NN - <title>/`

Numbered `slide_001.jpg …`, in lecture order. This is the deduplicated set — one
image per distinct slide state, redundant frames already removed. Use it.

- `slides_deduped/.../timestamps.txt` maps each slide to its position in the
  recording and to its original filename. Use the timestamp when you need to cite
  where something appears.
- `output/` holds the full un-deduplicated capture. Consult it **only** if a slide
  looks truncated mid-build and you need to see another state of it.

**Slide reading is mandatory.** Read every slide image in the folder. Do not infer
content from filenames, from `timestamps.txt`, or from what you already know about
the topic. If you have not looked at a slide, its content does not go in the notes.

**Note on animated slides:** the deck animates — a slide's title often appears before
its body. The dedup keeps the *most complete* state of each slide, so what you see
should be fully built. If you do find a title with an empty body and no fuller
version anywhere, flag it (see Honesty rules) rather than inventing the content.

---

## Outputs

```
notes/<Module Folder>/<module-slug>-NN.md      ← one file per lecture
notes/<Module Folder>/README.md                ← module index; create/update
```

Match the existing convention exactly — see `notes/GenAI & LLM/genai-llm-01.md` and
`notes/Supervised Learning/README.md` before writing anything.

### Lecture ➜ module map

| Lec | Module folder | File | Slides |
|-----|---------------|------|--------|
| 01–03 | `Supervised Learning` | `supervised-learning-01..03.md` | 26, 21, 78 |
| 04–06 | `Deep Neural Networks` | `deep-neural-networks-01..03.md` | 49, 41, 76 |
| 07–09 | `Dimensionality Reduction` | `dimensionality-reduction-01..03.md` | 42, 40, 37 |
| 10–13 | `Unsupervised Learning` | `unsupervised-learning-01..04.md` | 84, 14, 24, 22 |
| 14–17 | `GenAI & LLM` | `genai-llm-01..04.md` | 66, 22, 45, 34 |
| 18–20 | `Sequential Learning` | `sequential-learning-01..03.md` | 83, 54, 79 |
| 21–23 | `Causal Inference` | `causal-inference-01..03.md` | 19, 32, 36 |
| 24–26 | `Reinforcement Learning` | `reinforcement-learning-01..03.md` | 73, 70, 31 |
| 27–29 | `Agentic AI` | `agentic-ai-01..03.md` | 23, 21, 35 |

Slide counts are the expected number of images in the folder — a sanity check that
you are pointed at the right lecture, not a target. `Supervised Learning` and
`GenAI & LLM` already exist and already contain notes; **rewrite those files against
the new, more complete slides.** Read the existing file first and keep anything good
in it, but do not let its gaps propagate.

---

## Phase 1 — Read every slide, build the inventory

Read all slide images in order. Then produce, **in your reply and not in a file**:

1. **Concept inventory.** Every distinct concept, term, formula, diagram, algorithm,
   named result, dataset, and tool that appears anywhere in the deck. For each, mark:
   - `[explained]` the slide actually teaches it
   - `[mentioned]` the slide name-drops it and moves on
   - `[assumed]` the slide uses it without ever naming it as a thing to learn
   Every one of these gets taught in the notes. `[mentioned]` and `[assumed]` items
   are exactly the gaps you exist to fill.
2. **Keyword list.** Flat, alphabetical, interview-recall oriented.
3. **Prerequisite chain.** Which concepts must be understood before which. Your notes
   follow *this* order when it differs from slide order — say so when it does.

Keep this inventory open as you write. Before finishing, confirm every item is
covered.

---

## Phase 2 — Write the notes

### Required structure

```markdown
---
title: "<Module> — Part N: <descriptive subtitle>"
topic: <module-slug>
lecture: NN
source: "slides_deduped/Lecture_NN - <title>"
slides: <count>
---

# <Title>

## What you'll understand after reading this
6–10 bullets phrased as capabilities — "You'll be able to derive…", "You'll be able
to explain why…". Abilities, not topic names.

## Before we start: what you need to know
### Prerequisite 1 — <name>
Every assumed concept, fully taught. Long is correct here.

## The big picture
Plain language, before any detail. Someone who reads only this understands the point
of the lecture and how it connects to the rest of ML.

## 1. <Concept>
[intuition → tiny concrete example → general rule → formal notation → edge cases]

### Worked example
Real numbers, every step shown, ending in an actual answer. Never "and so on".

### Where people get confused
"You might think X — actually Y." Be specific.

### 💼 Interview questions
### 🔬 Research opportunity     ← where the field is still open (use where apt)

## 2. <Concept>
…

## Putting it together
How every concept connects. An ASCII dependency diagram plus a walkthrough.

## Interview prep — Amazon Applied Scientist
## Glossary
## Check yourself
## Going deeper
```

### The teaching contract

**1. Explain every term on first appearance.**

> **Term** — one-sentence plain-English definition, no jargon inside it.
>
> *In everyday words:* an analogy to something familiar.
>
> *Concretely:* a specific tiny example with real values.
>
> *Why it exists:* what problem it solves; what people did before it.

**2. Build in layers.** Simplest version first, confirm it lands, then complicate.
Never lead with the formal version.

**3. Words before symbols.** State what an equation *says* as an English sentence,
then show it, then define every symbol in a table. No exceptions.

> The formula says: **the loss you get is set by how much compute you spend, and it
> falls by a predictable fraction every time you multiply compute.**
>
> $$L(C) \approx \left(\frac{C_{min}}{C}\right)^{\alpha}$$
>
> | Symbol | Read it as | What it means |
> |---|---|---|
> | $L$ | "loss" | How wrong the model is on average. Lower is better. |
> | $C$ | "compute" | Total arithmetic spent training, in FLOPs. |

**4. Teach what the slide skipped.** If a slide says "minimises cross-entropy" and
never explains cross-entropy, you explain it:

> 📚 **Background the slide assumed** — [concept]

**5. Derive, don't assert.** For every key result, show *why* it's true — a short
derivation, a limiting case, or an argument from first principles. "It can be shown
that" is banned.

**6. No hand-waving.** Never "beyond our scope", "intuitively it works out", or "for
now just accept". Hard concepts get *more* words, not fewer. They are the reason
these notes exist.

**7. Connect to practice.** For each major concept: where it shows up in a real
system, what breaks when it's done wrong, and what a practitioner actually types.

### Formatting

- Math in LaTeX: `$inline$`, `$$display$$` — the site renders KaTeX.
- **Bold** every term and number worth remembering.
- Tables for comparisons and symbol definitions.
- Code blocks for algorithms and worked calculations. Prefer runnable PyTorch/NumPy.
- Callouts: `> 📚` background · `> 💡` key insight · `> ⚠️` uncertain · `> 🎯` interview
- Reference a slide as `[slide 14]` when pointing at a specific figure.

---

## Phase 3 — Interactive elements

These notes become an interactive web version. Wherever a concept is genuinely
clearer *moving* than static, emit a spec block. Do not decorate — propose an element
only where it does real explanatory work (a threshold sliding, a loss surface being
descended, attention weights lighting up).

Use exactly this fenced form so the site build can parse it:

````
```interactive
type: slider | animation | simulator | quiz | graph | diagram
title: Short name
concept: Which concept this teaches
control: What the reader manipulates
observe: What visibly changes
insight: The specific realisation this produces
fallback: What a static reader sees instead
```
````

Aim for roughly 3–6 per lecture, placed inline next to the concept they serve.
Every one needs a `fallback` — the notes must teach completely as plain text.

---

## Phase 4 — Interview preparation

The reader is targeting an **Amazon Applied Scientist Internship**. Every lecture
file ends with:

### `## Interview prep — Amazon Applied Scientist`

1. **Core questions (8–12).** Ranked easy → hard, drawn from *this* lecture's
   content. Each with a model answer in a `<details>` block. At least three should
   force combining two concepts.
2. **Depth probes.** The follow-up an interviewer asks when your first answer is
   good — "why does that hold?", "when does it fail?", "what would you do instead?"
3. **Whiteboard-ready derivations.** The 1–3 results from this lecture you should be
   able to derive cold. Give the derivation in steps you can reproduce.
4. **Applied scenario.** One realistic Amazon-flavoured problem (recommendations,
   demand forecasting, fraud, search ranking, Alexa NLU, delivery routing) that this
   lecture's content solves. Walk the full approach: framing → data → model →
   metric → failure modes → what you'd ship.
5. **Leadership Principles tie-in.** Amazon weights LPs as heavily as technical
   skill, and these decks reference them directly. Name 1–2 LPs this lecture's work
   naturally demonstrates (*Dive Deep*, *Learn and Be Curious*, *Bias for Action*,
   *Customer Obsession*, *Insist on the Highest Standards*) and give a concrete
   one-line example of how a project using this material would evidence it.

Be honest about level: mark anything genuinely beyond intern scope as
`🎯 stretch — nice to know, not expected`.

---

## Phase 5 — Resources, then self-review

### `## Going deeper`

Ranked by importance, each with **one line on why it's worth your time** and a
difficulty marker (`intro` / `solid` / `hard`). Mix: the original paper where one
exists, one genuinely good explainer (blog/video/visualisation), and one hands-on
resource. Prefer the canonical source over the popular one. **Do not invent citations
— if you are not certain a paper's title, authors, or year are right, say so with
`⚠️ verify this`.** A confidently wrong citation is worse than none.

### Self-review

Re-read your output against this table and **edit the file in place** to fix
failures. Do not write a report about it.

| # | Check | Fail if |
|---|-------|---------|
| 1 | Inventory complete | Any Phase-1 item missing from the notes |
| 2 | Every term explained | Any ML term used before it's defined |
| 3 | Words before symbols | Any formula appears before its plain-English version |
| 4 | Every symbol defined | Any variable lacks a table entry |
| 5 | Worked examples land | Any example without a final number |
| 6 | Prerequisites taught | Any assumed concept left unexplained |
| 7 | Layered build-up | Any concept formal-before-intuitive |
| 8 | Derivations present | Any key result asserted, not shown |
| 9 | No hand-waving | Any "beyond scope" / "just accept" phrasing |
| 10 | Interview section usable | Any answer you couldn't actually say out loud |
| 11 | Interactive blocks earn their place | Any element that's decoration |
| 12 | Citations trustworthy | Any unverified paper/date stated confidently |
| 13 | Slide fidelity | Any slide's content absent from the notes |

Then update `notes/<Module>/README.md`: one line per lecture — title, what it covers,
and its key takeaway.

Finally print a short summary: slides read, concepts inventoried, word count,
interactive blocks proposed, interview questions written, and any `⚠️` flags left.

---

## Honesty rules

- Unreadable or ambiguous slide → `> ⚠️ Slide N unclear — verify against the recording`
  (cite the timestamp). Never guess its content.
- Not confident in a fact, constant, date, or attribution → `> ⚠️ verify this`.
  **Confident fabrication is the worst possible failure here** — the reader cannot yet
  tell when you are wrong, and will repeat it in an interview.
- Slide is outdated or oversimplified → say so, and give the current view.
- Field genuinely disagrees → present the disagreement, don't pick silently.

## Length

Set by what teaching to mastery actually requires — typically **6,000–15,000 words**
depending on slide count. Do not pad; do not compress. Under ~4,000 words on a
40+ slide lecture means you summarised instead of taught. The existing
`genai-llm-01.md` (~159 KB) is the quality bar.

## Scope discipline

One lecture per invocation. Do not build the web site — that is a later phase.
