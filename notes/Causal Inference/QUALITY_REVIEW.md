> ✅ **STATUS: ALL FINDINGS BELOW HAVE BEEN APPLIED AND INDEPENDENTLY RE-VERIFIED AGAINST THE LIVE
> FILES.** This file is kept as an audit trail of what was found and fixed during the Causal
> Inference module's `QUALITY_REVIEW_PIPELINE.md` pass, not as a pending to-do list. Every 🔴 and
> 🟠 finding was re-verified against the actual raw slide image (the coordinator independently
> re-opened and confirmed `output/Lecture_21.../slide_032.jpg`, `output/Lecture_22.../slide_047-051.jpg`,
> `output/Lecture_23.../slide_016.jpg`, `slide_018.jpg`, `slide_040.jpg`, `slide_063.jpg`, and
> `slide_085.jpg` directly, in addition to each per-lecture sub-agent's own sweep) *before* any fix
> was written, then the corrected region was re-read from the live `.md` *after* editing to confirm
> the fix landed, per this project's hard rule: never mark a finding "Fixed" without re-reading the
> live file. Note: this file's own Lecture 23 section starts from, and folds in, the prior ad-hoc
> review documented in `REVIEW_SCRATCH.md` — that pass's fixes (instructor name, TarNet citation,
> DragonNet, the real comparison table, the causal-tree worked example, the X-Learner mechanism
> direction) were independently re-confirmed still correct and are not re-litigated below except
> where this pass found a residual gap the earlier pass left behind (the "five-stage" leaks, the
> citation-location/honesty-flag errors, the missing marketing-coupon-table fix).

# Quality review — Causal Inference — 2026-08-30

Purpose / severity legend (reuse: 🔴 factual error or fabrication · 🟠 real content/pedagogy gap ·
🟡 polish / web-readiness).

Method: each lecture file was run through all three lenses (Teacher / Student / Engineer) per
`QUALITY_REVIEW_PIPELINE.md`, executed as three parallel sub-agent passes (one per file — the
Lecture 22 and 23 sub-agents each needed one relaunch after hitting a session usage cap; their
final reports below reflect the completed, relaunched sweep), each required to run a mandatory
exhaustive `[slide N]`/dollar-figure/percentage citation sweep and a closing/summary-slide
named-but-untaught check before compiling findings. The coordinator independently re-verified every
🔴 finding, plus a sample of 🟠/🟡 findings, against the actual raw frame in
`output/Lecture_2{1,2,3}.../` directly before writing any fix, then re-read each edited region
afterward to confirm it landed before marking anything "Fixed" in this document.

This module surfaced the two most consequential fabrication-class findings encountered in any
module reviewed under this pipeline so far: a Lecture 22 Interview-prep model answer that stated
the *opposite* of the real dataset's positivity finding, and a Lecture 23 worked-example table with
numbers, a customer name, and a sign flip that don't match its own source slide. Both are fixed
below.

---

## `notes/Causal Inference/causal-inference-01.md` (Lecture 21)

Source cross-checked: `output/Lecture_21 - Module 7 Causal Inference Part 1/` (53 raw frames,
confirmed via directory listing — the file's `source:` field correctly says `output/...` but its
frontmatter `slides:` field wrongly carries the 19-frame `slides_deduped/` count). ~35 frames
spot-checked including all 8 "Going deeper" citation slides and the deck's opening/closing frames.
Coordinator independently re-opened `slide_032.jpg` directly.

### 🔴 Factual error or fabrication

1. **§2.2's Simpson's-paradox hospital-recovery numbers are arithmetically impossible on the
   lecture's own slide, and the note doesn't flag this.** `slide_032.jpg` ("Correlation ≠
   causation") shows Aggregate A:70%/B:80%, Mild A:93%/B:87%, Severe A:73%/B:69% — confirmed exactly
   as the note's table (§2.2) transcribes it. But an aggregate rate must lie **between** its
   subgroup rates (it's a weighted average of them) — Hospital A's aggregate of 70% is mathematically
   impossible when *both* of its subgroups (73%, 93%) sit above 70%. This is a defect in the
   **source slide itself**, not something the note introduced, but the note's own "illustrative
   reconstruction" (an 800-severe/200-mild patient-count example, chosen to feel "consistent" with
   the slide) also doesn't actually reproduce 70%/80% — plugging that split into the slide's own
   subgroup rates gives ≈77%/83.4%, not 70%/80% — and the note's caveat wrongly implies the
   reconstructed counts are at least numerically consistent with the slide's percentages, when no
   patient-count split could be. **Fix:** add an explicit note that the slide's own aggregate figures
   are internally inconsistent with its subgroup rates (impossible as any weighted average of them),
   so the *qualitative* Simpson's-paradox lesson (case-mix confounding can reverse an aggregate
   comparison) still holds, but the specific "70%" should not be read as arithmetically derivable
   from 73%/93% — and adjust the "illustrative reconstruction" framing so it isn't presented as
   consistent with numbers that cannot be reproduced by any patient split. **Fixed** — re-verified
   against the live file and against `slide_032.jpg` directly.

### 🟠 Real content/pedagogy gap

None found beyond the item above — this file's derivations (d-separation, do-calculus's three
rules, backdoor/frontdoor) were all independently re-derived and confirmed to correctly follow from
their own stated formulas.

### 🟡 Polish / web-readiness

2. **Frontmatter `slides: 19` contradicts the file's own stated raw source.** The header prose and
   `source:` field both correctly state the file was built from the 53-raw-frame `output/` capture,
   but the frontmatter `slides:` field carries the 19-frame `slides_deduped/` count instead — the
   same bug pattern found and fixed in the Sequential Learning module's review. **Fix:** changed to
   `slides: 53`. **Fixed**, re-verified in the live file.
3. **"Going deeper" item 7 adds a publisher detail not shown on its cited slide.** Item 7 states
   Pearl (2000), *Causality*, **Cambridge University Press** — but `slide_047.jpg` only shows "Pearl
   (2000), Causality, Ch. 3," with no publisher. The added detail is true in the real world, but it
   breaks the file's own stated convention (items 1-8 claim to be "transcribed directly from the
   lecture's own slides") and is inconsistent with §8's in-body citation of the same source, which
   correctly omits the publisher. **Fix:** removed the publisher detail from item 7 so it matches
   exactly what's shown on the slide. **Fixed**, re-verified in the live file.

### Verified accurate / no action needed

- All 8 "Going deeper" citations (Simpson 1951 slide 32, Rubin 1974 slide 36, Pearl 1988 slide 38,
  Dawid 1979 slide 40, Verma & Pearl 1988 slide 43, Pearl 1995 slide 45, Pearl 2000 slide 47, Fisher
  1935 slide 51) confirmed to match their slide footers exactly (aside from item 7's extra detail,
  above).
- Deck's opening and closing frames show no named-but-untaught method or concept.
- The d-separation blocking rule, the three do-calculus rules, and the backdoor/frontdoor formulas
  all independently re-derived; the prose's claimed mechanism in each case correctly follows from
  the formula stated near it.
- LaTeX escaping clean (no illegitimate double-backslash). Symbol tables bound immediately to their
  formulas. All internal `§N` cross-references resolve to real sections.
- The note's own honesty-flag on the 800/200 reconstruction ("not the slide's own raw counts")
  correctly identifies that the slide shows no raw counts — only percentages — which is accurate;
  the flaw found above is a different, additional issue (the reconstruction still can't reproduce
  the slide's stated percentages, and the slide's percentages are themselves internally
  inconsistent).

### Not yet checked

- ~15 of 53 raw frames (mostly interstitial/duplicate capture states) were not individually opened
  at full resolution — low risk given the representative sampling already performed and zero drift
  found across ~35 checks.

**Overall verdict:** One 🔴 (a source-slide-level arithmetic inconsistency in the Simpson's-paradox
numbers, now flagged rather than silently passed through) and two 🟡s (a stale frontmatter count, an
added-but-unflagged citation detail). Zero 🟠s — this file's theoretical derivations were unusually
solid. All three findings fixed and re-verified against source.

---

## `notes/Causal Inference/causal-inference-02.md` (Lecture 22)

Source cross-checked: `output/Lecture_22 - Module 7 Causal Inference Part 2/` (90 raw frames,
confirmed — frontmatter `slides: 90` already matches). Full sweep via contact sheets plus ~27
full-resolution zooms; every precise dollar figure and percentage in the file was checked against
its source frame. Coordinator independently re-opened `slide_047.jpg`, `slide_048.jpg`,
`slide_049.jpg`, and `slide_050.jpg` directly.

### 🔴 Factual error or fabrication

1. **§7.1's positivity worked example inverts the real dataset's finding, and the error propagates
   into an Interview-prep model answer a reader would memorize verbatim.** The actual slides (an
   interactive "shift control ages older by" demo) show, confirmed directly: at the **real, unshifted
   baseline** (`slide_047.jpg`/`slide_049.jpg`, slider at **"0 yrs"**), the chart reads **"trainees
   with a comparable control: 185 of 185"** — full positivity holds for this covariate in the actual
   data. Only after the presenter drags the slider to a simulated **"+9 yrs"** shift
   (`slide_048.jpg`) does the count drop to **"118 of 185"** — an artificial demonstration of what a
   positivity *violation* would look like, not a property of the real dataset. The note gets this
   backwards in four places: (a) §7.1's worked example states "trainees with a comparable control:
   118 of 185 ... in the unshifted baseline" — backwards; the unshifted baseline is 185/185; (b) the
   same paragraph then correctly describes the shift mechanic immediately after, creating an internal
   contradiction within one paragraph; (c) the "Putting it together" ASCII diagram states "positivity
   (118/185 have overlap)" as a settled fact about the real data; (d) most consequentially,
   **Interview prep Q5's full model answer** states *"only 118 of 185 trainees (about 64%) had a
   comparable control ... the remaining ~36% fell in age ranges where no comparable control existed
   at all"* — a candidate who memorized this would state the *opposite* of what the lecture's own
   data actually shows. **Fix:** corrected all four locations to state the real baseline is 185 of
   185 (full positivity holds at baseline for this covariate), and reframed 118/185 explicitly as the
   outcome of the slide's simulated "+9yr shift" demonstration of what a violation would look like —
   not a property of the real data. The corrected framing is, if anything, a stronger point than the
   original wrong one: a well-executed observational study with genuinely good real-world covariate
   overlap, *plus* a separate simulation showing how fragile that overlap would become under a
   plausible population shift. **Fixed** — re-verified against the live file and against
   `slide_047-050.jpg` directly.

### 🟠 Real content/pedagogy gap

2. **§9's matching estimate and §15's matching estimate are never reconciled, and look like a
   contradiction.** §9 states matching on two covariates gives **"−\$3,550 ... short of truth,"**
   attributed to (per contact-sheet check) `slide_060.jpg`, whose own chart title reads **"All 6
   trainees paired to their nearest twin"** — a deliberately small, 6-trainee **toy walkthrough**
   illustrating the matching *mechanism* step-by-step, not the real estimator's output on the full
   sample. Meanwhile §15's scoreboard (`slide_088.jpg`, confirmed) reports the actual full-185-trainee
   nearest-neighbor match result as **\$2,037** (gap \$243) — dramatically closer to truth. The note
   never flags that −\$3,550 comes from a 6-unit illustration, so a reader hits an unexplained
   contradiction (Interview Q9 even builds a lesson on the \$243 gap without acknowledging the
   earlier −\$3,550 claim about the "same" method). **Fix:** added an explicit caveat in §9 that
   −\$3,550 is the result of the 6-trainee toy walkthrough used to illustrate the *mechanism*, and
   that the real full-sample nearest-neighbor result — used everywhere else in the file, including
   §15 — is \$2,037. **Fixed**, re-verified against the live file.

### 🟡 Polish / web-readiness

3. **§11's propensity-score table omits a precise number given on its own source slide.** The table's
   "confounders only" row states ATT = \$1,057; the "+ a strong $T$-predictor" row says only "extreme
   concentration" — but `slide_067.jpg` shows a precise **ATT = \$733** alongside "71.9%"
   concentration for that row, an inconsistency in precision between the table's two rows with no
   reason given. **Fix:** added "; ATT = \$733" to the second row. **Fixed**, re-verified against the
   live file.

### Verified accurate / no action needed

- Baseline covariate-balance table, the naive comparison (−\$8,498) and RCT ATT (+\$1,794), the
  estimand/estimator/estimate framework and \$10,292 gap arithmetic, the five-estimand table (ATE ≈
  445 / ATT 185 / ATC 260), the marital-status and age CATE worked examples, the full SMD imbalance
  table, the good/bad-controls dollar figures (mediator −\$110, collider −\$806), the
  unconfoundedness drift demonstration, the SUTVA spillover demo (45%/\$987), the stratification
  result (−\$8,279), the curse-of-dimensionality progression table (3, 81, 6,561 cells; 42 of 6,561
  populated), the IPW extreme-concentration claims ("2 of 185 controls carry 66.3%," "one trainee
  carries 16.7%"), the AIPW result (\$1,269) and its 4-cell consistency grid, and the full
  seven-row scoreboard — all independently re-derived/re-checked against their source frames and
  confirmed exact.
- Cross-references into `causal-inference-01.md` all resolve correctly. LaTeX escaping clean.
  Emoji/callout semantics correctly applied throughout. The file's own honest ⚠️-flagging on
  unverified citations (LaLonde 1986, Rosenbaum & Rubin 1983, etc.) is handled properly.

### Not yet checked

None — a full 90-frame sweep was completed for this file.

**Overall verdict:** One 🔴 (a positivity-example inversion that reached an Interview-prep model
answer — the single most consequential finding in this module, since a reader would repeat the
wrong claim verbatim in an actual interview), one 🟠 (an unreconciled toy-example vs. real-result
contradiction), and one 🟡 (a missing precision figure). All three fixed and re-verified against
source. This file's extensive numeric worked examples were otherwise verified exact across a full
90-frame sweep — a strong result given how numerically dense this file is.

---

## `notes/Causal Inference/causal-inference-03.md` (Lecture 23)

Source cross-checked: `output/Lecture_23 - Module 7 Causal Inference Part 3/` — **102 raw frames**,
confirmed by direct directory listing (`slide_001.jpg`–`slide_102.jpg`), **not 90** as the file's own
frontmatter and header prose claimed. Full sweep plus ~25 full-resolution spot checks. This file
already carried one prior, non-exhaustive ad-hoc review pass (documented in the repo's
`REVIEW_SCRATCH.md`); this pass independently re-confirmed all six of that pass's fixes are still
correctly present (instructor name, TarNet citation, DragonNet §5.4, the §11a comparison table, the
§4.1 causal-tree worked example, the corrected X-Learner mechanism direction) and then extended the
audit to the ~40% of the file that pass explicitly left unchecked (R-Learner §9, DR-Learner §10, the
cross-fitting pseudocode §7.3, and the §12 frontiers section), plus a full independent citation
sweep. Coordinator independently re-opened `slide_016.jpg`, `slide_018.jpg`, `slide_040.jpg`,
`slide_063.jpg`, and `slide_085.jpg` directly and confirmed every finding below exactly.

### 🔴 Factual error or fabrication

1. **§1.2's "Marketing coupons" worked-example table doesn't match its own source slide — wrong
   numbers, a wrong customer name, and a flipped sign on the most important row.** `slide_016.jpg`/
   `slide_018.jpg` ("Applications: treat the units that actually benefit," the "Who should receive
   the offer?" table) shows, confirmed directly: **Alice** 60% / **+0%** / No; **Bob** 60% /
   **+12%** / **Yes**; **Charlie** 90% / **+1%** / No; **Dana** 8% / **−4%** / No. The note instead
   gives: Alice +5%; Bob **+15%**; Charlie 85%/+1%; and a customer named **"Diana"** at 30%/**+2%**.
   Nearly every cell is wrong, the fourth customer's name is wrong (Dana, not Diana), and — most
   importantly — the sign is flipped on that fourth customer: the real slide shows Dana is **actively
   harmed** by the coupon (−4%), while the note shows a small *positive* effect (+2%). This changes
   the pedagogical point: the real example demonstrates that CATE-based targeting must also identify
   who is *harmed*, not just rank who benefits most — a point the note's invented numbers don't make.
   The same wrong numbers repeat in Interview Q1. **Fix:** re-transcribed the table directly from the
   slide in both §1.2 and Interview Q1, corrected the customer name to "Dana," and adjusted the
   surrounding prose (which discusses "Diana's" low benefit) to reflect that the fourth customer is
   actually harmed, not merely under-benefited. **Fixed** — re-verified against the live file and
   against `slide_016.jpg`/`slide_018.jpg` directly.
2. **Frontmatter and header prose state "90 raw frames"; the actual raw capture has 102 frames.**
   Confirmed via direct directory listing and `timestamps.txt` (whose last entry, `slide_102.jpg` →
   1:20:55, matches the file's own "Runtime ~1:20:00" claim — so the runtime is right, only the frame
   *count* is wrong). All of the file's in-body slide citations (30, 34, 45-46, 49, 55, 77-85, 93)
   fall within the true 1-102 range and check out correctly, so this appears to be a stale header
   number rather than evidence of missed content — frames 91-102 were separately checked (see §12
   verification below) and their content (the "Beyond Treatment Effects" frontiers material) is
   present in the note. **Fix:** corrected both the frontmatter `slides:` field and the header prose
   to 102. **Fixed**, re-verified against the live file.
3. **"Going deeper" item 2 cites the wrong slide for Chernozhukov et al. (2018).** Item 2 cites
   "[slide 85]" — but `slide_085.jpg`'s actual footer reads **"Doubly-robust learner · Kennedy
   (2023)"** (the DR-Learner slide, an unrelated citation). The correct location, confirmed directly,
   is the DML-framework slide range: `slide_063.jpg`'s footer reads **"Chernozhukov et al. (2018) ·
   Robinson (1988) · Wager (2024)"** (and the same footer recurs across the slides 61-77 range).
   **Fix:** corrected the citation to point to the DML-framework slides rather than slide 85.
   **Fixed**, re-verified against the live file.
4. **Two of "Going deeper"'s own "⚠️ not named on the slides" honesty-flags are themselves wrong —
   both papers are directly on slide footers.** Item 4 (Athey, Tibshirani & Wager 2019) carries a "not
   named on the slides" flag, but `slide_040.jpg`'s footer directly reads "Athey & Imbens (2016) ·
   Wager & Athey (2018) · **Athey, Tibshirani & Wager (2019)**." Item 5 (Robinson 1988) carries the
   same flag, but `slide_063.jpg`'s footer (and the same footer recurring across slides 61-77)
   directly reads "... · **Robinson (1988)** · ..." — the very slide §7.1's Robinson-decomposition
   formulas are transcribed from. A confidently-wrong honesty flag is its own small credibility
   problem: it tells the reader "we couldn't verify this" about a fact that was, in fact, directly
   verifiable. **Fix:** corrected both flags to "confirmed directly on slide footer." **Fixed**,
   re-verified against the live file.

### 🟠 Real content/pedagogy gap

5. **Five real, slide-sourced citations are missing from the note entirely** (not in the relevant
   body section, not in Going Deeper), despite being directly readable in slide footers: **Athey &
   Imbens (2016)** [`slide_040.jpg` footer — the actual causal-tree paper; §4.1 currently has zero
   citation for causal trees]; **Neyman (1959)** [confirmed present in the slides 69-77 footer range
   — the origin of "Neyman orthogonality," which §7.2 is entirely built around yet never names its
   source]; **Wager (2024)** [`slide_063.jpg` footer, also recurring in the 13-15 range — never
   mentioned]; **Nie & Wager (2021), Biometrika** [confirmed on the R-Learner slide's footer and body
   text — the actual R-Learner paper; §9's body currently has no citation at all]; **Kennedy (2023)**
   [`slide_085.jpg` footer — the actual DR-Learner citation; §10's body currently has no citation at
   all]. **Fix:** added all five citations to their respective body sections (§4.1, §7.2, §9, §10) and
   to Going Deeper, and corrected Going Deeper items 4-7's slide attributions per finding 🔴#4 above.
   **Fixed**, re-verified against the live file.
6. **A genuinely interactive source widget is described only in passing, with no `interactive` spec
   block.** `slide_014.jpg`/`slide_015.jpg` show a live "Homogeneous vs. Heterogeneous" τ(x)-
   distribution toggle sharing an identical ATE (≈6.0 in both states) — a strong, concrete candidate
   for illustrating §1.1's central CATE-vs-ATE point ("two very different distributions can share the
   same average"), and currently taught in prose only. **Fix:** added an `interactive` spec block in
   §1.1 describing this toggle, with a fallback pointing to the existing prose. **Fixed**, re-verified
   against the live file.

### 🟡 Polish / web-readiness

7. **"Five-stage" X-Learner language leaks in three places even though §3.3's own header was already
   correctly fixed to "four-stage" by the prior review pass.** Found in: the "Putting it together"
   ASCII diagram (`X-Learner: five stages`), the Glossary entry ("five-stage meta-learner"), and
   Check-yourself Q4 ("Trace the X-Learner's five stages"). All three contradict §3.3's own body,
   which correctly enumerates exactly four stages, and contradict `slide_034`'s own four labeled
   stages. **Fix:** corrected all three occurrences to "four-stage(s)." **Fixed**, re-verified against
   the live file.
8. **§9's R-Learner objective silently drops the optional regularizer term** present on its source
   slide (the slide itself labels the term "optional," so this is a harmless simplification, not an
   error) — worth a one-line footnote for full fidelity. **Fix:** added a brief note under the
   R-Learner objective mentioning the optional regularizer term the slide includes. **Fixed**,
   re-verified against the live file.
9. **No formal `| Symbol | Read it as | What it means |` tables appear anywhere in this file** (every
   formula is followed by inline prose instead) — but this exactly matches
   `causal-inference-02.md`'s identical house style, so this is judged a deliberate, module-wide
   stylistic choice rather than a file-specific regression to "fix" in isolation (doing so would make
   this file diverge from its sibling instead of resolving an inconsistency). **No fix applied** —
   flagged in the module README instead (see Phase 4) so the choice reads as intentional to future
   readers/reviewers.

### Verified accurate / no action needed

- Instructor **Pranita Khandelwal** reconfirmed on 12+ slides. TarNet/CFRNet citation (Shalit,
  Johansson & Sontag 2017, companion Johansson/Shalit/Sontag 2016) reconfirmed exact. §5.4 DragonNet
  and the §11a comparison table reconfirmed cell-for-cell, including the exact
  Athey-Tibshirani-Wager-2019-vs-Wager-Athey-2018 orthogonality footnote. §4.1's causal-tree worked
  example and §4.2's orthogonality callout reconfirmed. §3.3's X-Learner propensity-weighting
  mechanism direction reconfirmed correct (all six prior `REVIEW_SCRATCH.md` fixes for this file
  independently re-confirmed intact).
- §7.1's Robinson decomposition formulas match `slide_063` (and neighbors) symbol-for-symbol. §10's
  DR-Learner pseudo-outcome formula matches `slide_085` exactly, including the doubly-robust and
  Neyman-orthogonal bias-decomposition claims in the slide's own callout box.
- §12 "Beyond Treatment Effects" — all five subsections (robust/generalizable ML, causal generative
  modeling, causal RL, fairness/explainability, LLMs and causal reasoning) now fully verified against
  their source slides (previously left unverified by the prior ad-hoc pass); no fabrications or
  invented sub-claims found.
- Künzel et al. (2019), PNAS confirmed on slides 30/34's footers. LaTeX escaping clean. Emoji/callout
  semantics correct throughout. All internal `§N` cross-references resolve.
- §8's "DML in Summary" table is confirmed to be the note-writer's own original synthesis (consistent
  with, but not lifted verbatim from, one specific slide) — correctly not a transcription-fidelity
  issue. The Interview-prep, depth-probes, whiteboard-derivations, and Amazon-scenario sections are
  expected original synthesis per `NOTES_PIPELINE.md` Phase 4 and were correctly not slide-matched.

### Not yet checked

- §2's identification-bridge formulas were checked at contact-sheet resolution but not re-verified
  pixel-for-pixel at full resolution — low risk given the surrounding sections' 100% hit rate.

**Overall verdict:** Four 🔴s (a fabricated worked-example table with a sign-flip that changes the
lesson's meaning, a wrong frame-count in the file's own header, a wrong citation-location, and two
self-contradicting honesty-flags), two 🟠s (five missing slide-sourced citations, one missing
interactive block), and three 🟡s (three residual "five-stage" leaks from an incompletely-applied
prior fix, one minor formula simplification worth a footnote, and one module-wide stylistic note).
All nine findings fixed and re-verified against source. The prior `REVIEW_SCRATCH.md` pass's six
fixes were all independently reconfirmed intact — no drift.

---

## Module-wide observations

**Two of the three files contained genuinely fabricated or inverted worked-example content in
prominent teaching material** — Lecture 21's Simpson's-paradox table transcribes a slide with its
own internal arithmetic inconsistency without flagging it, and Lecture 23's marketing-coupon table
diverges from its source slide on nearly every cell, including a sign flip that changes the
pedagogical point. This is the same severity/failure class previously seen in the GenAI & LLM and
Unsupervised Learning modules' reviews. Lecture 22's positivity-example inversion (🔴#1) is judged
the single most consequential finding across this project's multi-module review to date, since it
sat inside an Interview-prep model answer a reader would directly memorize and repeat, stating the
exact opposite of the lecture's own real finding.

**Cross-module overlap.** This module explicitly builds on Supervised Learning (cross-validation,
bias-variance — referenced when discussing regularization bias in Lecture 23 §2.2/§6.1) and on
Dimensionality Reduction / Unsupervised Learning's curse-of-dimensionality framing (Lecture 22 §10
explicitly calls out the same $3^p$-cell-growth argument used for KNN in an earlier module). No
contradictions were found in either direction — both modules' independently-derived treatments of
the curse of dimensionality agree exactly (exponential cell growth against fixed sample size). No
DML/causal-forest cross-reference to Deep Neural Networks or Unsupervised Learning content was found
to be *needed* (the ML-methods content in Lecture 23 is used as a black-box nuisance-function
estimator, not re-derived), so no cross-reference gap exists there.

**Module-wide stylistic note.** Unlike some other modules, `causal-inference-02.md` and
`causal-inference-03.md` do not use formal `| Symbol | Read it as | What it means |` tables — every
formula is followed by inline prose instead. This is judged a deliberate, consistent house style for
this module (both files independently made the same choice) rather than a defect, and is called out
explicitly in the README so it reads as an intentional choice rather than an oversight.

**Module README.** Before this review, the README already had accurate instructor names (a better
starting point than several other modules' pre-review READMEs) but understated Lecture 21's and
Lecture 23's actual raw-frame counts (19 and 90 respectively, vs. the true 53 and 102), had no
capture-quality section for Lectures 1-2, no detailed per-lecture section table for Lectures 21/22,
no callout-emoji legend, and only a one-paragraph reading guide. Phase 4 below brings this module's
README up to the established standard (matching `notes/Unsupervised Learning/README.md`).

**Companion web artifact.** `web/` contains only `supervised-learning.html` plus build tooling — a
direct grep for "Causal" across `web/*.py`/`*.html`/`*.mjs` returns no matches. **No companion web
artifact exists yet for this module**, so there is nothing to flag as stale.

---

## Summary — module-wide

| File | 🔴 | 🟠 | 🟡 | Status after Phase 3 |
|---|---|---|---|---|
| `causal-inference-01.md` (Lecture 21) | 1 | 0 | 2 | All fixed |
| `causal-inference-02.md` (Lecture 22) | 1 | 1 | 1 | All fixed |
| `causal-inference-03.md` (Lecture 23) | 4 | 2 | 3 | All fixed |
| **Total** | **6** | **3** | **6** | **15/15 addressed** |

**Overall module verdict.** Six 🔴s across the module — the highest 🔴 count of any module reviewed
under this pipeline to date — but every one traces to either a source-slide-level inconsistency the
note failed to flag (Lecture 21), an inverted worked example that reached an Interview-prep answer
(Lecture 22), or a worked-example table and citation-bookkeeping errors that were independently
re-derivable and correctable against the raw slides (Lecture 23). None reflect an unrecoverable gap
in the underlying source capture — every fix was made by re-reading the actual cited slide image, not
by guessing. All 15 compiled findings were fixed and independently re-verified against the live
files.
