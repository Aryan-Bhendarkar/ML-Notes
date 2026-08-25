> ✅ **STATUS: ALL FINDINGS BELOW HAVE BEEN APPLIED.** This file is kept as an audit trail of what
> was found and fixed during the Supervised Learning module's `QUALITY_REVIEW_PIPELINE.md` pass, not
> as a pending to-do list. All 14 findings (1 🔴, 1 🟠, 12 🟡) were fixed directly in the four module
> files and re-verified against source after fixing. The module `README.md` (instructor names, word
> counts, slide counts, glossary/interactive-block counts) has been updated to match the post-fix
> state and now links back to this file. The one 🔴 (a self-contradicting slide-count total in
> `supervised-learning-02.md`) did not affect any teaching content — see that file's section below.
> **The companion web artifact `web/supervised-learning.html` was not touched by this pass and is now
> stale relative to the corrected markdown — rebuild it via `WEB_ARTIFACT_PIPELINE.md` before treating
> it as current.**

# Quality review — Supervised Learning — 2026-08-25

Purpose / severity legend (reuse: 🔴 factual error or fabrication · 🟠 real content/pedagogy gap ·
🟡 polish / web-readiness).

Method: each lecture file was run through all three lenses (Teacher / Student / Engineer) per
`QUALITY_REVIEW_PIPELINE.md`. Source fidelity was checked against the raw frame capture in
`output/<Lecture>/` (never `slides_deduped/`, per project memory `slides-deduped-is-lossy`), using
contact sheets built with `contact_sheet.py` plus full-resolution spot checks, cross-referenced with
each lecture's `timestamps.txt`. `applied-scientist-practicum.md` is original extension content (not
slide-sourced), so it was checked under Lens 1/2 plus internal consistency and web-readiness under
Lens 3, not slide fidelity.

---

## `notes/Supervised Learning/supervised-learning-01.md` (Lecture 01)

Source cross-checked: `output/Lecture_01 - Module 1 Supervised Learning Part 1/` (64 raw frames), 2
contact sheets covering all 64 frames + 14 full-resolution spot checks on the OLS worked example, the
normal-equation matrix inversion, the bias–variance decomposition, the sigmoid derivation, and the
instructor nameplate (slide `f4`/`f5`).

### 🔴 Factual error or fabrication

None found. Every citation, precise number, and verbatim-looking quote checked against the raw frames
matched exactly.

### 🟠 Real content/pedagogy gap

None found. Every worked example re-derives to the stated result; no "it can be shown"/"intuitively"/
"beyond scope" hand-waving phrases present; no orphaned headings or duplicated paragraphs; every `§N`
cross-reference resolves.

### 🟡 Polish / web-readiness

1. **Capture-note slide breakdown was off by one category.** Lines 14–17 claimed *"12 content
   slides, 5 'Key Takeaways' slides, 5 section dividers, the agenda, the instructor slide, the title
   and the close"* (12+5+5+1+1+1+1 = 26). Two independent recounts contradict this: (a) the notes'
   own `## N.` section numbering has exactly **11** content sections (§1,2,3,5,6,8,9,10,12,13,15 — the
   other 5 of the 16 numbered sections are the "Key Takeaways" sections), and (b) a direct re-read of
   contact sheets `scratchpad/l1_check1.png`/`l1_check2.png` shows the deck opens with **two**
   distinct title-like states: a generic "amazon ML Summer School #MLSummerSchool" branding card
   (`f1`–`f2`) *and* the "Supervised Learning Part 1 #AmazonHackathon" title card (`f3`) — two states,
   not one, which the original breakdown collapsed into a single "title" line. Corrected breakdown:
   **11 content slides + 5 Key Takeaways + 5 section dividers + agenda + instructor + branding/intro
   card + title card + close = 26.** Total (26) was always right; only the category breakdown was
   wrong. **Fixed** in Phase 3.
2. **No `🔬 Research opportunity` callout used anywhere in the file.** Not a defect on its own (the
   pipeline says "use where apt," not "use always"), but §9's double-descent discussion is flagged
   with `⚠️` when it is a textbook example of a genuinely open research question, not just a slide
   gap. **Fixed** in Phase 3 by adding a `🔬` callout alongside the existing `⚠️` at that point (the
   `⚠️` stays — the slide gap itself is still real).

### Verified accurate / no action needed

- Frontmatter `slides: 26` matches the actual raw-frame-derived state count (after fix #1's
  reclassification, still 26).
- Instructor **"Apoorva Singh"** confirmed directly on the nameplate frame (`f4`/`f5`).
- LaTeX escaping is clean — the one double-backslash in the file (line 1187/1189, inside a `pmatrix`)
  is a legitimate row-break, not a bug.
- All symbol tables sit immediately under their formula; no separation found.
- Callout emoji used strictly per the fixed semantic map (📚💡⚠️🧪🎯).
- All 4 `interactive` blocks are well-formed with every field filled and a valid `type`
  (`simulator`/`slider`/`graph`/`slider`).
- Word count ~19,487 vs. the README's claimed ~19,400 — close enough, no fix needed.
- Glossary = 41 terms (matches README claim). Check-yourself = 26 questions (matches). Interview
  questions = 10, 4 combining two concepts (matches). 3 Leadership Principles given (matches).
- All major derivations (OLS by hand vs. normal equation, both giving ŷ = 2.2 + 0.6x; the
  bias–variance decomposition; the sigmoid/log-odds derivation) independently re-derived and confirmed
  correct.

### Not yet checked

- Roughly half of the 64 raw frames were only inspected at contact-sheet resolution, not individually
  at full resolution — no visual anomalies were visible at contact-sheet scale, but a frame-by-frame
  full-resolution pass was not exhaustive.
- Part 2's actual section numbering was not independently verified when checking Part 1's forward
  cross-references into Part 2 (out of this file's scope; covered when auditing Part 2 itself).

**Overall verdict:** Strong file. The only two issues found are a metadata/breakdown miscount in the
capture note (cosmetic — the total slide count was always right) and an optional missing callout type.
No factual errors, no content gaps, no fabricated numbers or citations.

---

## `notes/Supervised Learning/supervised-learning-02.md` (Lecture 02)

Source cross-checked: `output/Lecture_02 - Module 1 Supervised Learning Part 2/` (90 raw frames) +
`slides_deduped/Lecture_02 - Module 1 Supervised Learning Part 2/` (21 frames, confirmed by direct
directory listing) + `timestamps.txt`, 3 contact sheets covering all 90 raw frames + 6 full-resolution
spot checks (MSE/MAE/Huber comparison table, the ROC curve raw-score table, the fraud-detection
example, the Naive Bayes spam-classification numbers, the kernel-trick expansion, the instructor
nameplate).

### 🔴 Factual error or fabrication

1. **Capture-recovery arithmetic was wrong — stated total (34) does not match the recovery table's
   own numbers (21 + 14 = 35).** The frontmatter (`slides: 34`), the capture note (*"the deck actually
   has **34 distinct states**"*), and the module `README.md` (three separate places: the index table,
   the "Read this before running the pipeline" warning section, and the "What's in Part 2" summary)
   all stated **34**. But the capture note's own recovery table lists exactly **14** rows of content
   recovered from the raw capture, and `slides_deduped/Lecture_02.../` was independently confirmed by
   direct directory listing to hold **21** images (not merely claimed) — 21 + 14 = **35**, not 34. This
   is not a case of dropped teaching content (every one of the 14 recovered topics was independently
   verified present and accurate in the body), but the file's own claimed total contradicts its own
   evidence table, which is exactly the kind of confident-but-wrong number the honesty rules exist to
   catch. **Fix:** correct **34 → 35** everywhere it appears (`supervised-learning-02.md` frontmatter
   and capture note; `README.md` index table, warning section, and Part 2 summary). **Fixed** in Phase
   3, re-verified by recounting the recovery table (14 rows) and the `slides_deduped/` directory (21
   files) directly.

### 🟠 Real content/pedagogy gap

None found. Every named method referenced in the closing "Putting it together" section is taught in
the body with a derivation and worked example; depth is proportional to difficulty (the optimiser
lineage and the kernel trick — the two hardest ideas in the file — get the most space, matching their
difficulty).

### 🟡 Polish / web-readiness

1. **Frontmatter `source:` field is stale/misleading.** It reads `source:
   "slides_deduped/Lecture_02 - Module 1 Supervised Learning Part 2"`, but the file's own capture note
   two lines below it, and the module README, both document that the actual content was recovered from
   the raw `output/` capture specifically *because* `slides_deduped/` is lossy for this lecture. Per
   `QUALITY_REVIEW_PIPELINE.md` Lens 3's own trigger condition ("if it says `slides_deduped/`... re-
   audit against the raw frames"), this is exactly the kind of header claim that should be corrected
   to avoid a future reviewer being pointed at the wrong, lossy source. **Fix:** change `source:` to
   `"output/Lecture_02 - Module 1 Supervised Learning Part 2"`. **Fixed** in Phase 3.
2. **Glossary not alphabetical.** Order around line 3434–3436 runs AdamW → Adagrad → Adam; correct
   alphabetical order is Adagrad → Adam → AdamW. **Fixed** in Phase 3.
3. **README says "6 `interactive` blocks" for Part 2, but the file has 7** (confirmed: `type:` appears
   at lines 459, 684, 874, 1442, 2004, 2586, 2883 — 7 occurrences), and the README's own title list on
   the same line already names all 7 (*"outlier tug-of-war · the δ dial · the cost of confidence ·
   three descents on one surface · linked threshold/ROC/PR views · K scaled and unscaled ·
   kernel/C/γ"*). **Fix:** README "6" → "7". **Fixed** in Phase 3.
4. **README says "Glossary — 50 terms" for Part 2; actual count is 49** (confirmed by counting glossary
   table rows directly). **Fix:** "50 terms" → "49 terms". **Fixed** in Phase 3.
5. **Dual slide-numbering convention is never explained to the reader.** The file cites both
   un-prefixed `slide_NNN` (meaning `slides_deduped/` numbering, used in a few early inline references)
   and "raw `slide_NNN`" (meaning `output/` numbering, used in the capture-note recovery table and
   later citations) — both are internally correct, but a reader who opens `output/slide_004.jpg`
   expecting an un-prefixed citation's target would land on the title slide instead of the intended
   slide (e.g. Huber loss). **Fix:** add one clarifying sentence to the capture note explaining the two
   numbering systems and which one each citation style refers to. **Fixed** in Phase 3.

### Verified accurate / no action needed

- LaTeX escaping clean — the file's double-backslash instances are all legitimate row-breaks inside
  `cases` blocks (4 confirmed instances), no illegitimate ones.
- All symbol tables bound directly under their formula.
- Callout emoji used strictly per the fixed semantic map.
- All `§N` cross-references resolve; cross-references into Part 1 (§8–11, §12–14) were checked against
  Part 1's actual section numbering and are accurate.
- Word count ~30,910 vs. claimed ~30,900 — fine.
- No hand-waving phrases found.
- The Muon/SOAP "unverified" hedge is still present and still appropriate (these are genuinely recent,
  fast-moving-literature optimisers).
- Instructor **"Vikas Raykar"** confirmed.
- Extensive re-derivation confirmed correct: MSE/MAE/Huber gradients and MLE derivations, BCE/KL
  proof, softmax numerics, the 10-example threshold sweep, ROC-AUC = **0.8125** computed both by
  trapezoid and by pairwise ranking (both methods independently re-derived and match), the fraud
  example's **8.3%** precision, the accuracy-paradox numbers, R² (including the worked negative R²),
  the Naive Bayes spam classification (**0.7955** cross-checked in log-space), KNN worked example, the
  SVM margin = 2/‖w‖ derivation, the kernel-trick polynomial expansion matching the quadratic feature
  map at **121**, and the Adam bias-correction worked steps.
- All 7 `interactive` blocks are well-formed with valid `type` values.

### Not yet checked

- External citations (Huber 1964, Kingma & Ba 2015, Loshchilov & Hutter 2019, Cortes & Vapnik 1995,
  Goyal et al., Ng & Jordan 2001) were not independently verified against primary sources — these are
  already appropriately hedged with `⚠️ verify this` in the file, so no unwarranted confidence is being
  asserted.
- Not all 90 raw frames were opened individually at full resolution (contact sheets showed no visible
  gaps or anomalies in the unchecked frames).

**Overall verdict:** One real 🔴 (a self-contradicting slide-count total — content itself was not
affected) plus five 🟡 metadata/polish items, all mechanical and now fixed. No missing content, no
wrong derivations, no fabricated numbers — the extensive numeric re-derivation pass came back clean.

---

## `notes/Supervised Learning/supervised-learning-03.md` (Lecture 03)

Source cross-checked: `output/Lecture_03 - Module 1 Supervised Learning Part 3/` (163 raw frames), full
contact-sheet coverage of all 163 frames + 19 full-resolution spot checks (the Gini/entropy worked
split, the OOB-fraction convergence table, the AdaBoost round, the gradient-boosting round, the
macro-F1 confusion-matrix arithmetic, the class-weight computation, the calibration/Brier/ECE numbers,
the final pipeline's reported metrics, and the one documented capture gap at `f37`/`f38`).

### 🔴 Factual error or fabrication

None found. Every precise number checked against source frames matched exactly, including decimals:
Gini gain **0.1057**, information gain **0.2091** bits, depth-5 accuracy **0.757** vs. full-tree
**0.697**, bagging **0.753** / Random Forest **0.769** / OOB **0.774**, the **36.8%** OOB fraction,
the AdaBoost reweighted-error-equals-0.5 proof, macro-F1 **0.657** (OvR) / **0.695** (OvO) / **0.677**,
the `class_weight='balanced'` values **1.0375 / 1.8578 / 0.6676**, the imbalance table's **−2.7**
accuracy / **+10.6** minority-F1 shift, stratified vs. plain K-Fold **0.702 ± 0.023** vs.
**0.688 ± 0.015**, grid vs. random search **0.684** vs. **0.681**, the Brier/ECE progression, the final
pipeline's CV F1 **0.707** / test F1 **0.679**, and the 4,424 / 3,539 / 885 dataset-split counts. All
worked calculations were independently re-derived with no arithmetic errors found.

### 🟠 Real content/pedagogy gap

None found. Depth is proportional to difficulty throughout; no hand-waving phrases; every worked
example ends in a real, checkable number.

### 🟡 Polish / web-readiness

1. **Capture-note frame/timestamp citation is internally inconsistent.** Line 23 cites
   *"`slide_056`, 45:23"* for the notebook table-of-contents evidence, but `timestamps.txt` gives
   `slide_056.jpg → 44:47` and `slide_057.jpg → 45:23` — the frame number and the timestamp point at
   two different frames. Content is identical between the two (confirmed directly), so nothing taught
   is wrong, but the citation itself doesn't resolve to a single frame. **Fix:** change the timestamp
   from `45:23` to `44:47` so it matches the cited `slide_056`. **Fixed** in Phase 3, re-verified
   against `timestamps.txt` line 59 (`slide_056.jpg  2687  44:47`).
2. **§16.1's notebook table-of-contents table is transcribed verbatim without flagging that the real
   notebook's own running section headers drift from it from "Bagging & RF" onward** (e.g. frames
   `f130`/`f134` show "Section 7/8" where the TOC's own numbering implies a different section number,
   because the notebook's own "Section 1" appears to merge what the TOC lists as two rows, with
   subsections 1.3/1.4 visible in `f89`). A reader who jumps to "Section 13" in the live notebook using
   only the TOC table would land one section short of where the table implies. **Fix:** add a one-line
   `⚠️` caveat next to the §16.1 table noting the off-by-one drift from "Bagging & RF" onward, so a
   reader relying on the table to navigate a live copy of the notebook isn't misled. **Fixed** in Phase
   3.
3. **README says "50-term glossary" for Part 3; actual count is 53** (confirmed by counting glossary
   table rows directly). **Fix:** "50-term" → "53-term". **Fixed** in Phase 3. ("50 check-yourself
   questions" in the same README sentence was independently confirmed accurate — no change needed
   there.)

### Verified accurate / no action needed

- The one documented capture gap (the K-Fold half of deck slide 14, lost between forced samples `f37`
  and `f38`) is real and is honestly flagged in the file, not silently reconstructed — confirmed
  directly against the frames.
- The model-selection leaderboard cell is genuinely unexecuted in the source notebook (`[f142]`/
  `[f143]`), matching the file's honest claim that §13 has no leaderboard numbers to quote.
- 163 raw frames / 24 deck slides / 13-section notebook counts all confirmed.
- Instructor **"Sandeep Chatterjee"** confirmed.
- LaTeX escaping 100% clean — zero double-backslash bugs anywhere in the file.
- All 10 symbol tables sit directly under their formula.
- Callout emoji usage (📚 8 · 💡 27 · ⚠️ 33 · 🧪 18 · 🎯 4 · 🔬 3) all fall within the fixed semantic
  map — no drift.
- All 5 `interactive` blocks are well-formed with valid `type` values.
- Cross-references to Part 1 and Part 2 spot-checked and resolve correctly.
- Interview prep (**12** questions, **5** combining two concepts), **13** depth probes, **3**
  whiteboard-ready derivations, **4** Leadership Principles, **50** check-yourself questions, and 12
  ranked "Going deeper" resources (3 flagged `⚠️` as unverified) all match the README's claimed counts
  exactly.
- Word count ~33,010 vs. claimed ~32,900 — fine, no fix needed.

### Not yet checked

- Not every one of the 163 frames was opened at full resolution — full-resolution checks were
  targeted at frames backing a specific numeric claim, the documented gap, or the TOC drift; frames
  with no cited claim were only inspected at contact-sheet scale.
- Small-print axis labels on several matplotlib-output frames (`f83`–`f105`, `f114`–`f121`) were not
  pixel-checked against their ASCII-plot transcriptions in the notes — low risk, since every number
  actually printed near those plots (not just read off an axis) was independently verified.
- External citations (Hyafil & Rivest 1976, Wolpert 1996/97, Bergstra & Bengio 2012, Chen & Guestrin
  2016, etc.) were not verified against primary sources — the file already self-flags the 3 it is least
  sure of with `⚠️`.

**Overall verdict:** The strongest file in the module. Zero factual errors and zero content gaps found
across an extensive numeric re-derivation pass; the only issues are three small citation/metadata
polish items, now fixed.

---

## `notes/Supervised Learning/applied-scientist-practicum.md`

Not slide-sourced (original extension content, per the module README) — Lens 3's slide-fidelity and
citation checks do not apply. Checked under Lens 1 (teaching contract), Lens 2 (reader simulation), and
Lens 3's internal-consistency / web-readiness checks only.

### 🔴 Factual error or fabrication

None — no slide-sourced claims to fabricate against; no internal arithmetic or citation claims found
to be wrong.

### 🟠 Real content/pedagogy gap

1. **`NDCG` and `MRR` are used without definition anywhere in the module.** The "Retrieval and ranking"
   table states *"NDCG@$K$, MRR, Precision@$K$"* as the typical ranking metric, but neither `NDCG`
   (Normalised Discounted Cumulative Gain) nor `MRR` (Mean Reciprocal Rank) is defined in this file or
   anywhere else in the module — confirmed by grepping the whole module for both terms; they appear
   only in this one table cell. `PR-AUC` and `ROC-AUC`, by contrast, are safe to use undefined here
   because Part 2's evaluation-metrics section already teaches them in full and this practicum
   explicitly says it "reuses" Part 3's discipline. This is a real "used before defined" gap under the
   pipeline's teaching contract. **Fix:** add a one-sentence plain-English gloss for each term inline
   where they first appear (NDCG: a ranking-quality score that rewards good items being near the top
   and discounts them logarithmically the further down the list they sit; MRR: the average of
   1/(rank of the first correct/relevant item) across queries). **Fixed** in Phase 3.

### 🟡 Polish / web-readiness

1. **Third `interactive` block uses `type: scenario`, which is not a valid type.** Per
   `NOTES_PIPELINE.md`'s "Phase 3 — Interactive elements" spec and the module README's "Building the
   interactive site" table, `type` must be one of exactly `slider | animation | simulator | quiz |
   graph | diagram` — `scenario` is not in that list. **Fix:** change `type: scenario` (line 166) to
   `type: simulator` (consistent with the file's other two blocks, and the block's own `control`/
   `observe` fields describe an interactive incident-diagnosis walkthrough, which fits `simulator`
   better than any other valid type). **Fixed** in Phase 3.

### Verified accurate / no action needed

- Callout emoji (💡, 🎯) used correctly per the fixed semantic map; no others used, which is fine —
  this file has no slide-gap or worked-example content that would call for `⚠️`/`🧪`/`📚`/`🔬`.
- No internal `§N` cross-references to check (this file doesn't use section numbers).
- The two other `interactive` blocks (lines 146, 156) are well-formed with valid `type: simulator` and
  all required fields filled.
- The eight-question interviewer-evidence table, the four "gaps to close," the portfolio capstone
  rubric, and the "Interview drill" sequence are internally consistent with each other and with what
  Parts 1–3 actually teach (they cross-reference the split/leakage discipline, calibration, and the
  `Pipeline` lesson accurately).
- Word count 1,839 vs. claimed ~1,800 — fine.

### Not yet checked

- None — this file is short enough that it was read and checked in full.

**Overall verdict:** Solid bridge document; the only real gap is two undefined ranking-metric terms in
one table, now fixed, plus one invalid `interactive` type value, also fixed.

---

## Summary — module-wide

| File | 🔴 | 🟠 | 🟡 | Status after Phase 3 |
|---|---|---|---|---|
| `supervised-learning-01.md` | 0 | 0 | 2 | Both fixed |
| `supervised-learning-02.md` | 1 | 0 | 5 | All fixed |
| `supervised-learning-03.md` | 0 | 0 | 3 | All fixed |
| `applied-scientist-practicum.md` | 0 | 1 | 1 | Both fixed |
| **Total** | **1** | **1** | **11** | **14/14 fixed** |

**Companion web artifact.** `web/supervised-learning.html` exists for this module. Because the fixes
above changed the source `.md` files (most visibly the 34→35 slide-count correction, the frontmatter
`source:` field, the glossary reordering, and the new NDCG/MRR definitions), **this artifact is now
stale relative to its markdown source and should be rebuilt via `WEB_ARTIFACT_PIPELINE.md` before it is
next treated as current.** This review did not touch the artifact — per `QUALITY_REVIEW_PIPELINE.md`'s
scope discipline, that rebuild is a separate, later pass.

**Overall module verdict:** This module was already close to the top-1%/interview-ready bar the
pipeline targets — the three lecture files came back with **zero missing content, zero fabricated
citations, and zero wrong derivations** across an extensive source-fidelity and re-derivation pass
(a first for this project's review history; compare `REVIEW_SCRATCH.md`'s findings on other modules,
which each surfaced at least one fabricated citation or a missing named method/architecture). The one
🔴 found was a self-contradicting slide-count total that did not affect any teaching content. The
practicum bridge file had one real "used-before-defined" terminology gap. All 14 findings across the
module have been fixed and re-verified against source.
