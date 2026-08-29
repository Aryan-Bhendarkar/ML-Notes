> ✅ **STATUS: ALL FINDINGS BELOW HAVE BEEN APPLIED AND INDEPENDENTLY RE-VERIFIED AGAINST THE LIVE
> FILES.** This file is kept as an audit trail of what was found and fixed during the Sequential
> Learning module's `QUALITY_REVIEW_PIPELINE.md` pass, not as a pending to-do list. Every finding
> below — whether reported by a per-lecture sub-agent or found independently by the coordinator's own
> direct re-inspection of the cited raw frames — was re-verified against the actual raw slide image
> (for the one 🔴 and every 🟠) or against the live `.md`/`README.md` content (for every finding)
> *after* being fixed, per this project's hard rule: **never mark a finding "Fixed" without
> re-reading the live file to confirm it landed.** The module `README.md` (instructor names, word
> counts, glossary counts, interactive-spec counts, the Part 3 deduped-frame count, capture-quality
> notes) now matches the post-fix state of all three files.

# Quality review — Sequential Learning — 2026-08-29

Purpose / severity legend (reuse: 🔴 factual error or fabrication · 🟠 real content/pedagogy gap ·
🟡 polish / web-readiness).

Method: each lecture file was run through all three lenses (Teacher / Student / Engineer) per
`QUALITY_REVIEW_PIPELINE.md`, executed as three parallel sub-agent passes (one per file), each
required to run a mandatory exhaustive `[slide N]` citation sweep and a closing/summary-slide
named-but-untaught check before compiling findings. The coordinator independently re-verified every
finding below against the actual raw frame in `output/Lecture_1{8,9}.../` or `output/Lecture_20.../`
directly before applying any fix, then re-read each edited region afterward to confirm it landed
before writing "Fixed" anywhere in this document. All three note files were already built from the
raw `output/` capture (150 / 136 / 160 frames respectively), not the lossy `slides_deduped/` sets —
confirmed independently by both the sub-agents and the coordinator, so none of this module's findings
trace to the `slides-deduped-is-lossy` failure mode itself (unlike Unsupervised Learning Lecture 13).

This is the cleanest module audited in this project to date: only **one** confirmed 🔴 across all
three files (a single wrong word, self-correcting two sections later in the same file), zero
fabricated citations or numbers, and every worked-example arithmetic chain (BPE, TF-IDF, negative
sampling, the full "I am happy" attention computation, WordPiece scores, Unigram segmentation, the
causal mask, GPT's next-token softmax, the notebook's 0.70→0.88 result) independently re-derived and
confirmed correct.

---

## `notes/Sequential Learning/sequential-learning-01.md` (Lecture 18)

Source cross-checked: `output/Lecture_18 - Module 6 Sequential Learning Part 1/` (150 raw frames,
confirmed via directory listing — matches the file's own frontmatter claim), ~30 `[slide N]`
citations swept exhaustively, plus the coordinator's own re-verification of `slide_096.jpg` and
`slide_094.jpg` directly.

### 🔴 Factual error or fabrication

None found. Every worked example (BPE merge sequence, TF-IDF cat/the/fish computation and the
"neural" interview quiz, negative-sampling loss direction, GloVe's ice/steam/water/fashion ratio
table and 4×4 co-occurrence matrix, FastText's enviroment/environment n-gram overlap) was
independently re-derived and matches both the arithmetic and the cited slide exactly.

### 🟠 Real content/pedagogy gap

1. **Negative-sampling worked example silently drops one of three negative words shown on the
   slide.** `slide_096.jpg` ("Negative Sampling") shows **three** negative examples for center word
   'king' — `'banana'`, `'table'`, **and `'purple'`** (all three visible as red `u_neg` ellipses with
   dashed arrows from the dot-product box) — but the note's prose worked example (§6.4, "🧪 Worked
   example [slide 96]") and the interactive block's fallback text (same section) mention only
   `'banana'` and `'table'`, omitting `'purple'` entirely. **Fix:** add `'purple'` to both the prose
   worked example and the interactive block's fallback text, matching all three negatives the slide
   actually shows.
2. **"Going deeper" overclaims what the lecture's own references slide verifies.** The closing
   caveat states the lecture's own references slide "[slide 150] confirms author names and years for
   all three papers" — but per direct inspection, slide 150 lists Word2Vec generically as a single
   "Mikolov et al. (2013)" line, not as two separately-attributed papers, while the note (correctly)
   cites two distinct real 2013 Mikolov papers in "Going deeper" #1–2. The papers themselves are real
   and correctly titled/dated from established field knowledge, but the claim that the *slide itself*
   separately confirms both is false. **Fix:** reworded the closing caveat to state that the slide
   cites Word2Vec generically as one line, and that the two separate paper citations in "Going
   deeper" come from established field knowledge rather than the slide's own detail.

### 🟡 Polish / web-readiness

3. **Frontmatter `slides: 83` contradicts the file's own stated capture source.** The header prose
   and `source:` field both correctly state the file was built from the 150-raw-frame `output/`
   capture (not the 83-image `slides_deduped/` set), but the frontmatter `slides:` field still shows
   the deduped count (83). **Fix:** changed to `slides: 150` for consistency with the file's own
   stated source.
4. **The slide's own internal inconsistency (K=5–15 vs. K=5–20) is silently resolved without a
   note.** `slide_096.jpg`'s fine print reads "K = 5-15 negatives sampled from $P_n(w)\propto
   f(w)^{3/4}$" while its headline bullet reads "K = 5-20 negatives per positive → ~100x faster!" —
   two different ranges on the same slide. The note uses "5–20" throughout without flagging that the
   source slide itself is inconsistent. **Fix:** added a one-line caveat noting the slide's own two
   different K-ranges (fine print says 5–15, headline says 5–20) and that the note follows the
   headline figure.
5. **Notation mismatch with the slide's own label ("2C" vs. the note's "$2m$"), unflagged.**
   `slide_094.jpg` ("CBOW vs Skip-Gram") literally reads "Slower (**2C** preds/window)" — using `C`
   for window size — while the note (§6.2–6.3) consistently uses `$m$` (established in §6.1's
   objective formula) and silently translates the slide's "2C" to "$2m$" with no note that the slide
   itself uses a different symbol. Not a numerical error (both denote the same quantity, and
   consistent notation within the note is good practice), but a reader checking the cited slide
   directly would see a different symbol than the note uses. **Fix:** added a parenthetical noting
   the slide's own label reads "2C"; this note uses `$m$` (window size) throughout for consistency
   with §6.1.

### Verified accurate / no action needed

- BPE merge example, tokenization table, stemming/lemmatization table, the full five-stage
  preprocessing pipeline demo, one-hot encoding, Bag-of-Words example, TF-IDF formula + worked
  example + interview quiz, N-grams example, the seven named failure modes and their concrete
  instances, the compositionality worked example, the sparse-vs-dense comparison table, the
  Distributional Hypothesis quote and attribution, the king/man/woman/queen analogy arithmetic,
  Word2Vec's architecture diagram, the Skip-Gram and CBOW objective formulas and their comparison
  table, GloVe's co-occurrence matrix and objective, FastText's n-gram sum formula and OOV example,
  and the closing evolution-of-representations timeline — all independently confirmed against the
  cited slides.
- Runtime "57:24" confirmed against `timestamps.txt`. LaTeX escaping clean (no illegitimate
  double-backslash instances). All 7 `interactive` blocks well-formed (all seven required fields
  present, fallback genuinely sufficient standalone). Symbol tables bound immediately to their
  formulas throughout. Cross-module check against the GenAI & LLM module's closing
  static-vs-contextual-embeddings framing found no conflict.

### Not yet checked

- Frames between explicitly cited slide numbers were not all individually opened at full resolution
  — the sweep targeted every actually-cited number; given zero numerical/content drift found across
  ~30 checks, this is assessed as low risk but not exhaustively proven.

**Overall verdict:** Zero 🔴s. Two genuine 🟠s (a dropped third negative-sampling example, an
overclaimed slide-verification statement) and three 🟡s (a stale frontmatter count, an unflagged
source-slide self-inconsistency, and an unflagged notation translation), all now fixed and
re-verified against source.

---

## `notes/Sequential Learning/sequential-learning-02.md` (Lecture 19)

Source cross-checked: `output/Lecture_19 - Module 6 Sequential Learning Part 2/` (136 raw frames),
35 of 136 frames spot-checked (every explicit citation plus title/closing frames), plus the
coordinator's own re-verification of `slide_090.jpg`, `slide_133.jpg`, and `slide_006.jpg` directly.

### 🔴 Factual error or fabrication

1. **§7.1 misidentifies which cell lights up in the lecture's own attention heatmap demo.** The note
   claims *"the cell at row `'it'`, column `'tired'` lights up with a real, non-trivial attention
   weight"* for the sentence "The animal didn't cross the street because it was too tired." Direct
   inspection of `slide_090.jpg` (the "Attention Mechanism Explorer" self-attention heatmap for this
   exact sentence) shows the bright gold cell in row `"it"` is at **column `"animal"`** (the 2nd
   column) — the cell at column `"tired"` (the 11th/last column) in row `"it"` is the same dark teal
   as the rest of that row, not highlighted. The note's own reasoning ("the model has learned to
   connect 'it' back toward the noun phrase it refers to") only supports "animal" (a noun), not
   "tired" (an adjective) — and the file's own §10.1 table two sections later correctly states
   *"Coreference: resolves pronouns ('it' looks at 'animal')"*, confirming "animal" is the
   demonstrably intended and correct claim. **Fix:** corrected "column `'tired'`" to "column
   `'animal'`" in §7.1. **Fixed**, re-verified by re-reading the corrected line in the live file and
   re-confirming against `slide_090.jpg` directly.

### 🟠 Real content/pedagogy gap

None found. This file's derivations and arithmetic are unusually solid throughout — see Verified
list below.

### 🟡 Polish / web-readiness

2. **§11's evolution-timeline hedge is more cautious than the source slide warrants, and conflates
   two distinct entries.** The note's `⚠️ verify this` callout lists "Bahdanau Attention 2014/2015"
   as a single uncertain entry, but `slide_133.jpg`'s own timeline graphic shows **two separate,
   individually-dated dots**: "Bahdanau Attention" at **2014** and "Luong Attention" at **2015** —
   directly, unambiguously readable on the slide, not small print. Every other year in the note's
   hedge (Transformer 2017, Flash Attention 2022, GQA/MQA 2023, MLA/Differential Attention 2024,
   Hybrid Mamba+Attention 2025) already matches the slide exactly. **Fix:** split the conflated
   "Bahdanau Attention 2014/2015" entry into "Bahdanau Attention (2014)" and "Luong Attention (2015)"
   as two separate, directly-confirmed timeline points, and removed the blanket `⚠️ verify this`
   hedge since the entire timeline is now confirmable directly from the slide's own graphic.
3. **Missing on-slide image attribution.** `slide_006.jpg` (the ELMo forward/backward LSTM diagram)
   carries a visible on-slide credit, *"Image: Jay Alammar, The Illustrated BERT"* — never carried
   into §2.1's discussion of that diagram or into "Going deeper" (which cites a *different* Jay
   Alammar work, "The Illustrated Transformer," for an unrelated diagram in §10). Not wrong, just
   incomplete — the ELMo diagram's actual credited source was never mentioned anywhere in the file.
   **Fix:** added the missing attribution as a caption note under the ELMo architecture diagram in
   §2.1.
4. **Cross-module notation divergence with GenAI & LLM, unflagged (not a bug).** This file uses
   $d_k$ throughout for the Q/K projection dimension; `genai-llm-01.md`'s independent derivation of
   the identical scaled-dot-product-attention mechanism uses plain $d$ for the same quantity. Both
   derivations are independently correct and reach the same conclusion — this is not a contradiction
   — but no cross-reference ties the two modules' coverage together for a reader moving between them.
   **Fix:** added a one-line cross-reference note in §9 pointing to `GenAI & LLM Part 1 §8` as an
   independent derivation of the same result, flagging the notation difference explicitly so a reader
   isn't confused encountering $d$ instead of $d_k$ there.

### Verified accurate / no action needed

- The instructor claim ("— not named in deck") independently reconfirmed by inspecting slides 1, 2,
  4, 5, 135, and 136 — no name overlay anywhere in the deck, unlike Lectures 18 and 20.
- §8's full worked attention computation (input embeddings, Q/K/V projection, the 3×3 score matrix,
  scaling by $\sqrt2$, the row-wise softmax, and the final weighted-sum outputs $\mathbf o_1,
  \mathbf o_2, \mathbf o_3$) independently re-derived by hand and matches `slide_091/095/099/102/
  103/106.jpg` exactly — this was the file's highest fabrication-risk content and it is completely
  correct.
- §9's $\sqrt{d_k}$ variance derivation and the $d=32$ unscaled-vs-scaled comparison table (ranges,
  entropy values, max probability) confirmed pixel-for-pixel against `slide_125.jpg`.
- §3.1's vanishing-gradient arithmetic ($0.70^9\approx0.040$) confirmed against the slide's own
  on-screen readout.
- Internal section numbering §1–§11 sequential, no gaps. Cross-references into
  `sequential-learning-01.md` (Part 1 §8 FastText, Prerequisite 3) both verified accurate.
- The closing section's claimed four building-blocks list and "next: assembling the Transformer"
  forward pointer confirmed exact against the deck's own closing slides.
- LaTeX escaping clean. All 5 `interactive` blocks well-formed. Symbol tables bound immediately to
  their formulas. Cross-module overlap with `GenAI & LLM §8` checked directly — same database-lookup
  analogy, same variance argument, same 512/8=64 split, same K/V-separation rationale — no
  contradictions found anywhere.

### Not yet checked

- Roughly 100 non-cited raw frames were not individually opened at full resolution — the sweep
  targeted every explicit citation, which is the scope this pipeline calls for.

**Overall verdict:** One 🔴 (a single misidentified column in an otherwise-correct heatmap claim,
already self-correcting two sections later in the same file) and three 🟡s (an over-conflated
timeline hedge, a missing image credit, an unflagged cross-module notation difference). Zero 🟠s —
this file's actual teaching content, derivations, and arithmetic were unusually solid throughout. All
four findings fixed and re-verified against source.

---

## `notes/Sequential Learning/sequential-learning-03.md` (Lecture 20)

Source cross-checked: `output/Lecture_20 - Module 6 Sequential Learning Part 3/` (160 raw frames),
also checked `slides_deduped/Lecture_20.../` (79 images + `timestamps.txt`, confirmed to exist even
though the note and README don't mention a deduped count for this lecture), ~27 full-resolution spot
checks, plus the coordinator's own re-verification of `slide_007.jpg`, `slide_063.jpg`, and
`slide_132`–`134.jpg` directly.

### 🔴 Factual error or fabrication

None found. Every precise number (the BPE merge example, the WordPiece score table, the Unigram
segmentation probabilities, the causal-mask matrix, the GPT prediction-head logits/softmax/loss
chain, the temperature-sampling values, and the hands-on notebook's headline 0.70/0.88 accuracy
result) was independently re-derived by hand and matches the cited slide digit for digit.

### 🟠 Real content/pedagogy gap

1. **§18's LoRA rung has no dedicated captured source frame — a real capture gap, not a content
   error.** The interactive 4-step PEFT walkthrough shows `slide_132.jpg` = "1 · Full fine-tuning"
   (step 1/4) and `slide_133.jpg` = "2 · Adapter tuning" (step 2/4), but the raw capture then jumps
   directly to `slide_134.jpg` = "4 · QLoRA" (step 4/4) — **step 3/4 ("3 · LoRA") was never
   captured as its own frame** (confirmed via `timestamps.txt`: only 58 seconds elapse between
   `slide_133` at 56:31 and `slide_134` at 57:29, versus 1.5 minutes for the previous step — the
   presenter evidently clicked through the LoRA step quickly enough that no stable frame was grabbed,
   the exact "segment tails" capture-loss pattern this project's memory warns about). The note's LoRA
   paragraph is not fabricated — cross-checked against `genai-llm-03.md` §8, which independently
   derives the identical mechanism (frozen base + additive low-rank update, merges back for zero
   inference latency) with real Llama-2-7B numbers — but it isn't verifiable from *this* lecture's
   own frames, and the note never flagged that. **Fix:** added an explicit `⚠️` capture-gap note
   under §18's LoRA paragraph, stating the step-3/4 frame was not captured in this lecture's raw
   output and that the description is corroborated via `GenAI & LLM Part 1... genai-llm-03.md §8`
   rather than this lecture's own slide.
2. **Zero `interactive` spec blocks, despite the source deck itself being built around at least five
   in-deck interactive widgets that were screenshotted only as static UI.** Confirmed capture
   evidence for: the BPE tokenizer demo (`slide_022.jpg`), the causal-mask "Apply causal mask" toggle
   (`slide_083.jpg`), the multi-head attention step-through (`slide_071.jpg`), the GPT sampling
   "Greedy / Temperature / Top-k / Top-p" control (`slide_121.jpg`), and the 4-step PEFT ladder
   (`slides_132`–`134.jpg`, directly re-confirmed by the coordinator). Lecture 19 has 5 well-formed
   `interactive` blocks for comparable content; this file — arguably richer in native, literally
   already-interactive source material — has zero. **Fix:** added 5 `interactive` blocks matching
   the five widgets above (BPE tokenizer step-through in §2, causal-mask toggle in §11.1, multi-head
   attention step-through in §9.1, the sampling-strategy comparison in §16.3, and the 4-step PEFT
   ladder in §18), each with all seven required fields and a fallback pointing to the existing static
   prose/table that already states the same content.

### 🟡 Polish / web-readiness

3. **§6 silently reorders content across the deck's own claimed tab boundaries.** The frontmatter
   states the file "follows" the deck's three explicit navigable tabs (1 Sub-word Tokenization / 2
   Transformer Architecture / 3 Language Models), and §6 ("A Model Sees Numbers, Not Text") is cited
   as opening "Part 2 — Transformer Architecture." Direct inspection of `slide_007.jpg` shows this
   content is actually the deck's 2nd slide overall, displayed while tab **"1 Sub-word Tokenization"**
   is still highlighted as active — several slides before "Attempt 1 — word-level" (`slide_010`,
   correctly the real start of tab 1's content). The note isn't wrong about the content, but a reader
   who opens `slide_007.jpg` to check §6 will see the wrong tab lit up. **Fix:** added a note at the
   top of §6 disclosing this is a deliberate pedagogical bridge — the slide displays under tab 1 in
   the recording, but its content ("what does a Transformer actually consume") is placed at the start
   of Part 2 in this file because that's where it teaches best, not because the file's Part-boundary
   claim was inaccurate.
4. **Module README's index table breaks its own established format for this lecture.** Rows 01/02
   show "(X deduped / Y raw frames)"; row 03 shows only "(160 raw frames)," silently omitting that
   `slides_deduped/Lecture_20.../` also exists (79 images) — not wrong (the note deliberately used
   raw frames per its frontmatter), but inconsistent with the table's own established format and
   could read as an oversight. **Fix:** addressed in the Phase 4 README rewrite — the new index row
   for Lecture 20 now states "(79 deduped / 160 raw frames)" matching rows 01/02's format.
5. **RoPE/ALiBi model-attribution hedge is more cautious than the source slide warrants.** The note's
   `⚠️ verify this` callout describes the model lists for RoPE (LLaMA, GPT-NeoX, PaLM, Qwen) and
   ALiBi (BLOOM, MPT) as read from "the slide's own small-print labels." Direct inspection of
   `slide_063.jpg` shows these are stated in the slide's plain, full-size body text ("Used by LLaMA,
   GPT-NeoX, PaLM, Qwen." / "Used by BLOOM and MPT.") — a directly quotable primary source, not small
   print requiring extra caution. **Fix:** downgraded the hedge to a plain citation, removing the
   "small-print" framing and the `⚠️ verify this` marker since the attribution is directly confirmable
   verbatim from the slide's own body text.

### Verified accurate / no action needed

- Frontmatter `slides: 160` confirmed accurate against the raw frame directory (`slide_001`–
  `slide_160.jpg`, no gaps).
- BPE/WordPiece/Unigram worked examples, the causal mask matrix, the GPT prediction-head example, the
  temperature-sampling example, the positional-encoding formula, and the multi-head attention worked
  example all independently re-derived and confirmed exact.
- The notebook's 0.70/0.88 result confirmed as the literal captured bar-chart value, not a paraphrase
  or rounding of a different number.
- BERT's NSP/RoBERTa aside quoted verbatim; the BPE citation (Gage 1994, Sennrich et al. 2016)
  verbatim. Section numbering §1–§21 contiguous, no gaps.
- Cross-module consistency check against `genai-llm-01.md` and `genai-llm-03.md` found full agreement
  on LoRA/QLoRA/PEFT mechanics and on the BERT/GPT/T5 architecture classification — genuine, expected
  duplication across course modules (both modules independently and correctly teach the same
  material), not a contradiction. Neither module's file cross-references the other; this is
  noted but not required to fix.

### Not yet checked

- Many uncited slides were not individually opened at full resolution — spot-checked via prose
  consistency against surrounding confirmed frames, with a 100% hit rate on everything actually
  verified.
- "Going deeper" citations' full bibliographic accuracy was not independently checked against
  external sources beyond the slide's own in-line citations — the file's own `⚠️ verify this` hedge
  on that section remains appropriate and was left as-is.

**Overall verdict:** Zero 🔴s. Two genuine 🟠s (a real capture gap in the LoRA rung, now disclosed
rather than silently passed through as if fully sourced; and a total absence of interactivity despite
the richest native interactive source material in the module) and three 🟡s (a silently-reordered
section relative to the deck's own tab structure, a README format inconsistency, and an
over-cautious citation hedge). All five findings fixed and re-verified against source.

---

## Module-wide observations

**Cross-module overlap with GenAI & LLM.** Sequential Learning (lectures 18–20) sits directly
adjacent in content to GenAI & LLM (lectures 14–17), which is taught *earlier* in the course.
`genai-llm-01.md` ("Foundations, Scale, and the Transformer," lecture 14) already fully derives
self-attention, Q/K/V, scaled dot-product attention, multi-head attention, and positional encoding
(including RoPE/ALiBi) — the same material `sequential-learning-02.md` (§7–§10) and
`sequential-learning-03.md` (§7–§12) independently re-derive "from scratch." Likewise,
`genai-llm-03.md` §8–9 covers LoRA/QLoRA in depth with real Llama-2-7B numbers, materially the same
content as `sequential-learning-03.md` §18's PEFT ladder. **No contradictions were found anywhere** —
every independently-checked claim, formula, and worked mechanism agrees between the two modules
(same database-lookup analogy, same variance argument for $\sqrt{d_k}$, same 512/8=64 head-splitting
example, same LoRA merge-back/zero-latency argument). The only friction found is notational (Sequential
Learning's $d_k$ vs. GenAI & LLM's plain $d$ for the same quantity, now cross-referenced per Lecture
19 finding 4 above) and a lack of cross-module pointers in either direction, which is a genuine,
expected consequence of the two modules being written independently rather than a defect — Sequential
Learning teaches the *general* NLP-representation lineage (Word2Vec → attention → Transformer) while
GenAI & LLM teaches the same architecture as *infrastructure* for building and serving LLMs; the
duplication is pedagogically defensible (a reader could plausibly read either module first) even
though it means a reader who completes both will see the Transformer derived twice, in different
voices, with consistent conclusions both times.

**Companion web artifact.** `web/` contains only `supervised-learning.html` plus build tooling —
confirmed via a direct grep for "Sequential" across `web/` returning no matches. **No companion web
artifact exists yet for this module**, so there is nothing to flag as stale.

**Module README.** Before this review, the README already had instructor names and a basic
index/key-takeaway table (a better starting point than GenAI & LLM's README had), but lacked a
capture-quality section, per-lecture detailed tables of contents, a callout legend, and a reading
guide — the same gaps found and fixed in the Unsupervised Learning and Supervised Learning modules'
own reviews. Phase 4 below brings this module's README up to that established standard.

---

## Summary — module-wide

| File | 🔴 | 🟠 | 🟡 | Status after Phase 3 |
|---|---|---|---|---|
| `sequential-learning-01.md` (Lecture 18) | 0 | 2 | 3 | All fixed |
| `sequential-learning-02.md` (Lecture 19) | 1 | 0 | 3 | All fixed |
| `sequential-learning-03.md` (Lecture 20) | 0 | 2 | 3 | All fixed |
| **Total** | **1** | **4** | **9** | **14/14 addressed** |

**Overall module verdict.** This is the cleanest module audited in this project's history: a single
🔴 (a misidentified heatmap column that the file itself corrects two sections later), zero fabricated
numbers or citations anywhere across three files and ~90 spot-checked citations, and every
substantial worked-example arithmetic chain re-derived and found correct on the first pass. The real
findings cluster around two recurring, minor patterns: (1) the note occasionally drops or
under-attributes one item from a multi-item slide (Lecture 18's missing third negative-sampling word,
Lecture 19's missing image credit), and (2) genuinely interactive source material — especially in
Lecture 20's code-heavy, demo-driven deck — was screenshotted and described in prose without being
converted into the `interactive` spec blocks `NOTES_PIPELINE.md` calls for, despite Lecture 19
demonstrating this is entirely achievable for the same kind of content. Both patterns are now fixed.
