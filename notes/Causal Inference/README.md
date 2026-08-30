# Causal Inference — Amazon ML Summer School

Self-study notes built from the raw slide capture in [`output/`](../../output/) (not
`slides_deduped/`, which is lossy — see the warning in
[`notes/Supervised Learning/README.md`](../Supervised%20Learning/README.md)), following
[`NOTES_PIPELINE.md`](../../NOTES_PIPELINE.md).

These are **teaching documents, not summaries**. They define every term before using it, derive
every formula rather than asserting it, and work every example through to a final number. They are
written to be the only thing you need to read to master the lecture.

This module has gone through a full `QUALITY_REVIEW_PIPELINE.md` audit pass — see
[`QUALITY_REVIEW.md`](QUALITY_REVIEW.md) for the complete findings and fixes (six 🔴s, three 🟠s,
six 🟡s across the three files, all fixed and re-verified against source).

---

## Index

| # | Notes | Source deck | Instructor | Status | Covers |
|---|---|---|---|---|---|
| 01 | [causal-inference-01.md](causal-inference-01.md) | `Lecture_21 - Module 7 Causal Inference Part 1` (19 deduped / **53 raw frames**) | — (not named in deck) | ✅ Complete · reviewed | Why causality matters (prediction vs. intervention) · Correlation ≠ causation & Simpson's paradox · Potential outcomes framework (ATE/ATT/CATE) · Probabilistic graphical models & DAGs · Conditional independence (chain/fork/collider) · d-separation · do-calculus (3 rules) · Backdoor & frontdoor adjustment · RCTs as the gold standard |
| 02 | [causal-inference-02.md](causal-inference-02.md) | `Lecture_22 - Module 7 Causal Inference Part 2` (32 deduped / **90 raw frames**) | — (not named in deck) | ✅ Complete · reviewed | Treatment-effect estimands (ITE/ATE/ATT/ATC/CATE) · Estimand/estimator/estimate · Selection bias decomposition · Good vs. bad controls (confounder/mediator/collider) with real dollar figures · Unconfoundedness, positivity, SUTVA · Curse of dimensionality · Propensity score & IPW (+trimming) · Regression adjustment · Doubly robust (AIPW) estimation · Full estimator scoreboard against a known $1,794 truth |
| 03 | [causal-inference-03.md](causal-inference-03.md) | `Lecture_23 - Module 7 Causal Inference Part 3` (36 deduped / **102 raw frames**) | Pranita Khandelwal (Sr. Data Scientist, Amazon) | ✅ Complete · reviewed | Heterogeneous treatment effects (CATE) · Meta-learners (S/T/X-Learner) · Causal trees & forests (honest estimation) · Deep representation learning (TarNet, CFRNet, DragonNet) · DML framework (partialing out, Neyman orthogonality, cross-fitting) · R-Learner · DR-Learner · Full causal ML landscape mapped to canonical DAG · Frontiers (robust ML, causal generative models, causal RL, fairness, LLMs) |

⚠️ **Frame counts corrected during review.** Lecture 21's raw capture has **53** frames (not 19 — 19
is the lossy `slides_deduped/` count) and Lecture 23's raw capture has **102** frames (not 90 — a
stale header number in the file itself, corrected during the `QUALITY_REVIEW_PIPELINE.md` pass; see
`QUALITY_REVIEW.md`). Lecture 22's count (90 raw / 32 deduped) was already accurate.

## Key takeaway per lecture

| # | One-sentence takeaway |
|---|---|
| 01 | $P(Y\mid X) \ne P(Y\mid\text{do}(X))$ is the single fact this entire lecture exists to resolve, and every tool introduced — potential outcomes, DAGs, d-separation, do-calculus, backdoor/frontdoor adjustment — is a different route to the same destination: turning an unmeasurable interventional quantity into one that can actually be estimated from data, with a real randomized experiment being the one setting where that translation is unnecessary. |
| 02 | On one real dataset where the true answer is known ($1,794), a naive comparison gets the sign backwards (-$8,498) purely from pre-existing group differences — and every classical estimator (stratification, matching, propensity weighting, regression adjustment, doubly robust AIPW) is a different recipe for removing that selection bias, with the lecture's own scoreboard showing that covariate overlap, not estimator sophistication, is what actually determines how close an estimate lands to the truth. |
| 03 | Every causal ML method — from S-Learners to representation learners to DML — is a different mechanism for blocking the backdoor path in the canonical DAG; the progression from meta-learners (flexible but biased) to orthogonal learners (DML, R-Learner, DR-Learner) is the progression from accepting regularization bias to suppressing it to second order, enabling unbiased heterogeneous treatment effect estimation with valid confidence intervals from ML-estimated nuisance functions. |

---

## Capture quality

### ✅ Lecture 21 — good, one source-slide arithmetic quirk

53 raw frames over ~53 minutes. Unusually citation-dense — eight named papers, each confirmed
directly on its own slide footer (Simpson 1951, Rubin 1974, Pearl 1988/1995/2000, Dawid 1979, Verma
& Pearl 1988, Fisher 1935). No content gaps: the deck's opening and closing frames name nothing that
isn't taught in the body.

⚠️ **One flagged source-slide defect.** The Simpson's-paradox hospital-recovery numbers on the
deck's own slide (`slide_032.jpg`) are internally inconsistent — an aggregate rate must fall between
its subgroup rates, and the slide's own aggregate figure for Hospital A (70%) sits below *both* of
its subgroup rates (73%, 93%), which is arithmetically impossible for any patient split. This is a
defect in the source material, not something the note introduced; the note now flags it explicitly
(see `QUALITY_REVIEW.md`) rather than silently passing it through — the *qualitative* lesson
(case-mix confounding can reverse an aggregate comparison) remains sound.

**The instructor is not named** anywhere in the recording — the webcam tile carries no label.

### ✅ Lecture 22 — excellent, unusually numeric

90 raw frames over ~57 minutes, built entirely around one real running dataset (the National
Supported Work job-training program) with a known randomized-trial answer (ATT ≈ +\$1,794). Every
dollar figure and percentage in the note — dozens of them, across the SMD table, the good/bad
controls comparison, the curse-of-dimensionality progression, the IPW/AIPW worked examples, and the
seven-row final scoreboard — was independently re-verified against its source frame during this
review and confirmed exact, with three corrections (see below).

⚠️ **One significant correction.** The note originally inverted the lecture's own positivity finding
— stating "118 of 185 trainees have a comparable control" as the real dataset's *baseline*, when the
real baseline is actually **185 of 185** (full positivity holds); 118/185 is the result of an
interactive **simulated** "+9-year shift" of the control population's ages, not a property of the
real data. This error had propagated into an Interview-prep model answer. Corrected across all four
locations it appeared; see `QUALITY_REVIEW.md` for the full account.

**The instructor is not named** anywhere in the recording.

### ✅ Lecture 23 — good, header frame-count corrected

**102 raw frames** over ~1:20:00 (corrected from a stale "90" during this review — see
`QUALITY_REVIEW.md`). This lecture had already been through one earlier, non-exhaustive ad-hoc
review pass (documented in the repo's `REVIEW_SCRATCH.md`) that fixed the instructor name, a
fabricated TarNet citation, a missing DragonNet section, a missing comparison table, and a missing
worked example — all independently re-confirmed still correct by this pass. This review then found
and fixed a second wave of issues in the ~40% of the file the earlier pass hadn't checked: a
worked-example table (§1.2, the marketing-coupon example) that didn't match its own source slide on
nearly every cell including a sign flip, several citation-location/attribution errors (a wrong slide
number for the DML citation, two "not named on the slides" flags that were themselves wrong), and
five real slide-sourced citations that were missing entirely (Athey & Imbens 2016, Neyman 1959,
Wager 2024, Nie & Wager 2021, Kennedy 2023).

The instructor is **Pranita Khandelwal**, Senior Data Scientist, International Machine Learning
team at Amazon (buyer abuse prevention and payment experiences) — confirmed directly from the slide
nameplate on 12+ slides.

**A note on house style.** Unlike some other modules, Lectures 22 and 23 do not use formal
`| Symbol | Read it as | What it means |` tables — every formula is instead followed immediately by
inline prose defining its symbols. This is a deliberate, consistent choice across both files (not a
per-file inconsistency), reflecting this module's numerically dense, example-driven teaching style;
it was reviewed and judged intentional rather than a gap to fix.

---

## What's in Part 1

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — runtime and capture note · 8 capabilities · 4 prerequisites taught from zero
(conditional probability/expectation as population statements · i.i.d. and why deployment breaks it
· joint distributions · expectation notation) · a full-lecture ASCII map tracing one continuous
argument from $P(Y\mid X)\ne P(Y\mid\text{do}(X))$ through potential outcomes and DAGs to
do-calculus, backdoor/frontdoor, and RCTs

- §1 Why Causality Matters — the central $P(Y\mid X)\ne P(Y\mid\text{do}(X))$ distinction · the
  do-operator, defined in full · three paradigms contrasted directly (supervised ML / causal
  inference / RL) in one table · 🎯 a three-statement correlation-or-causation self-check
- §2 Correlation ≠ Causation — the firefighters/property-damage spurious-correlation example ·
  Simpson's Paradox (hospital recovery rates) worked through with an illustrative reconstruction,
  ⚠️ now flagging the source slide's own internal arithmetic inconsistency (see Capture quality above)
- §3 Potential Outcomes Framework — the Rubin switching equation, the fundamental problem of causal
  inference (only one potential outcome per unit is ever observed), and the three aggregate
  estimands (ATE/ATT/CATE) distinguished by business question
- §4 Probabilistic Graphical Models — DAGs as a factorization of the joint distribution · 💡 the
  single sentence connecting graphs back to §1: deleting edges = intervening
- §5 Conditional Independence — chain, fork, and collider, contrasted in one table · ⚠️ the
  counterintuitive collider case (Berkson's paradox), worked through with the celebrity
  attractiveness/talent example
- §6 D-Separation Theorem — the complete, mechanical procedure for reading conditional
  independencies off a graph, generalizing §5's three cases into one rule
- §7 do-Calculus — all three rules stated, explained in words before symbols, and unified as a
  proof system for eliminating $\text{do}(\cdot)$ from an expression
- §8 Backdoor and Frontdoor Criteria — both derived as named corollaries of §7's three rules, with
  explicit guidance on when to reach for which
- §9 RCTs — why randomization uniquely handles *unmeasured* confounders, and the four practical
  limitations (cost, ethics, external validity, compliance) that push real problems back toward
  observational methods

**Closing** — a full-lecture ASCII dependency map · three closing threads (identifiability as the
load-bearing word of the lecture; confounding's three graphical shapes) · **10 interview questions**
with model answers (2 combining concepts) · 3 depth probes · 3 whiteboard-ready derivations · an
Amazon marketing-attribution scenario · 2 Leadership Principles · 19-term glossary · 11
check-yourself questions · 9 ranked resources (all 8 primary citations confirmed directly against
slide footers, plus one companion textbook honestly flagged as not slide-sourced).

**Interactive specs:** none in this deck.

</details>

---

## What's in Part 2

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — runtime and capture note, including the explicit ⚠️ hedge that the National
Supported Work dataset's LaLonde-1986 attribution is background context, not a transcribed slide
citation · 10 capabilities · 4 prerequisites (SMD, logistic regression briefly, weighted averages) ·
a full-lecture ASCII map tracing the three-act structure (treatment effects → identification →
classical estimation → the ML bridge)

- §1 A Job-Training Experiment — the National Supported Work dataset, its known randomized-trial
  answer (ATT ≈ +\$1,794), and the baseline covariate-balance table confirming randomization worked
- §2 Two Estimates of One Treatment Effect — the same data giving opposite signs (+\$1,794 vs.
  −\$8,498) under two different comparison groups, framed as the puzzle the rest of the lecture
  resolves
- §3 Estimand, Estimator, Estimate — the three-way distinction, with the \$10,292 gap-to-target
  worked example
- §4 Treatment-Effect Estimands — all five estimands (ITE/ATE/ATT/ATC/CATE) in one table, plus
  §4.1's marital-status and age CATE worked examples
- §5 Selection Bias and Its Two Families — the naive-estimate decomposition (naive = ATT +
  selection bias), derived and then verified numerically as covariates are balanced one at a time
- §6 Good and Bad Controls — confounder vs. mediator vs. collider, with real contrasting dollar
  figures (\$1,794 correct vs. −\$110 mediator vs. −\$806 collider)
- §7 Unconfoundedness, §7.1 Positivity and Common Support, §7.2 SUTVA — all three identification
  assumptions, each with its own numeric demonstration and (added during this enhancement pass) an
  `interactive` spec block reproducing the deck's own live slider/toggle; §7.1 now corrected to
  state the real baseline (185/185 comparable controls) versus the slide's separate simulated-shift
  demonstration (118/185 under an artificial +9-year age shift) — see Capture quality above
- §8 Stratification, §9 Matching (with a flagged distinction between the 6-trainee toy walkthrough
  and the real full-sample result used elsewhere in the file), §10 The Curse of Dimensionality
  ($3^8=6{,}561$ cells, 42 populated), §11 Propensity Score, §12 Inverse-Probability Weighting
  (+trimming), §13 Regression Adjustment, §14 Doubly Robust Estimation (AIPW) — six classical
  estimators, each scored against the same known \$1,794 truth
- §15 Estimates Compared — the full seven-row scoreboard, with the lecture's own empirical finding
  that overlap matters more than estimator choice
- §16 From Classical Estimation to Machine Learning — the explicit bridge naming $e(x)$ and
  $\mu_t(x)$ as the two nuisance functions every method in §8–§14 reduces to, setting up Part 3

**Closing** — a full-lecture ASCII map · three closing threads · **10 interview questions** with
model answers (2 combining concepts) · 3 depth probes · 3 whiteboard-ready derivations · an Amazon
fulfillment-center process-change scenario · 2 Leadership Principles · 22-term glossary · 12
check-yourself questions · 5 ranked resources (1 slide-confirmed DML citation, 4 honestly flagged as
field-knowledge background rather than transcribed slide citations, including the LaLonde dataset
attribution itself).

**Interactive specs:** 3 — added during the enhancement pass: §7 the unconfoundedness
hidden-confounder-strength slider, §7.1 the positivity "shift control ages older by" slider (the
demo central to this file's major 🔴 positivity-inversion fix), and §7.2 the SUTVA
no-spillover/spillover-leaks toggle. All three were previously taught only as static prose worked
examples; each now also gets an `interactive` spec block whose `fallback` reproduces that same prose.

</details>

---

## What's in Part 3

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — runtime and capture note (corrected to 102 raw frames during this review) ·
instructor confirmed (Pranita Khandelwal) · 10 capabilities · 4 prerequisites (potential
outcomes/CATE recap, the classical estimation toolkit recap, regularization bias defined, the
factual MSE loss) · a full-lecture ASCII map tracing meta-learners → trees/forests → representation
learning → the DML framework → R/DR-Learners → the full landscape

- §1 From Average to Heterogeneous Treatment Effects — why ATE isn't enough, with an `interactive`
  spec block on the deck's own homogeneous-vs-heterogeneous τ(x) toggle (added during this review)
  and the corrected marketing-coupon worked example (Alice/Bob/Charlie/**Dana**, with Dana's
  negative CATE now correctly shown as active harm, not just weak benefit)
- §2 From CATE Definition to Estimation — the identification bridge, and why ML is both the solution
  and the problem (regularization bias)
- §3 Meta-Learners — S-Learner (§3.1), T-Learner (§3.2), and the **four-stage** X-Learner (§3.3,
  with the propensity-weighting mechanism direction independently re-confirmed correct)
- §4 Causal Trees and Forests — §4.1's honest-estimation splitting criterion with a full numeric
  worked example (the discount-coupon Split A vs. Split B comparison), now citing Athey & Imbens
  (2016); §4.2's causal forests, with the Athey-Tibshirani-Wager-2019-vs-Wager-Athey-2018
  orthogonality distinction
- §5 Deep Representation Learning — TarNet (§5.2, implicit balance), CFRNet (§5.3, explicit IPM
  penalty), and DragonNet (§5.4, propensity head + targeted regularization)
- §6–§8 The DML Framework — partialing out / the Robinson decomposition (§7.1), Neyman orthogonality
  (§7.2, now citing Neyman 1959 directly), and cross-fitting (§7.3)
- §9 R-Learner — DML generalized to heterogeneous effects, now citing Nie & Wager (2021) directly
- §10 DR-Learner — doubly robust + orthogonal + meta-learner, now citing Kennedy (2023) directly
- §11 The Full Landscape — §11a, the deck's own real "Comparison of HTE estimators" table
  (transcribed faithfully, including the orthogonality footnote); §11b, a labeled synthesis view
  mapping every method onto the canonical DAG
- §12 Beyond Treatment Effects — five frontier subsections (robust ML, causal generative modeling,
  causal RL, fairness/explainability, LLMs), independently verified against source slides during
  this review

**Closing** — a full-lecture ASCII map · three closing threads · **10 interview questions** with
model answers (2 combining concepts) · 3 depth probes · 3 whiteboard-ready derivations · an Amazon
promotional-targeting scenario · 2 Leadership Principles · 24-term glossary · 13 check-yourself
questions · 12 ranked resources (up from 7 before this review), now with 10 confirmed directly on
slide footers after this review corrected two wrong "not named on the slides" flags and added five
previously-missing citations.

**Interactive specs:** 1 — §1.1's homogeneous-vs-heterogeneous τ(x) distribution toggle, added
during this review.

</details>

---

## Reading guide

The three parts total ~28,500 words and form a single progression: from *why* causal inference is
needed (Part 1) through *what* classical methods can do on real data (Part 2) to *how* ML methods
scale these ideas to high-dimensional, heterogeneous-effect settings (Part 3).

**Prerequisites.** This module builds on
[`Supervised Learning`](../Supervised%20Learning/) (cross-validation and the bias-variance
trade-off, referenced when discussing regularization bias in Part 3) and on
[`Dimensionality Reduction`](../Dimensionality%20Reduction/) / [`Unsupervised
Learning`](../Unsupervised%20Learning/)'s curse-of-dimensionality framing (Part 2 §10 explicitly
reuses the same $3^p$-cell-growth argument applied to KNN in an earlier module — independently
verified consistent, no contradiction). No prerequisite gap or cross-module contradiction was found
during this review.

**First pass.** Read linearly: Part 1 → Part 2 → Part 3. Do not skip *Before we start* in any file —
each part explicitly builds on the previous one's vocabulary (potential outcomes, the do-operator,
confounder/mediator/collider) without re-deriving it.

**Second pass.** Work every 🧪 example on paper *before* reading the solution. The most important
across all three parts:
1. Part 1 §3 — the switching equation and why $\tau_i$ is fundamentally unobservable
2. Part 1 §8 — the backdoor adjustment formula, derived as a weighted average of within-stratum
   comparisons
3. Part 2 §5.1 — the naive-estimate decomposition (naive = ATT + selection bias), verified
   numerically as covariates are balanced one at a time
4. Part 2 §14 — doubly robust (AIPW) estimation, verified across all four cells of the
   correct/incorrect nuisance-specification grid
5. Part 3 §4.1 — the causal-tree splitting-criterion worked example (Split A vs. Split B)
6. Part 3 §7.1 — the Robinson decomposition, derived symbol-by-symbol
7. Part 3 §7.2 — Neyman orthogonality, and why the bias drops from $O(\delta)$ to $O(\delta^2)$

**Before an interview.** Each file's *Putting it together*, then all 9 whiteboard derivations (3 per
part), then the depth-probe tables. The three highest-value derivations: **the backdoor adjustment
formula** (Part 1 §8), **the naive-estimate decomposition** (Part 2 §5.1), and **the Robinson
decomposition** (Part 3 §7.1).

**The three questions this module is most likely to be examined on:**
1. *"Why does $P(Y\mid X) \ne P(Y\mid\text{do}(X))$, and what does that gap require to close?"*
   (Part 1 §1, §7–§8)
2. *"Walk through how you'd estimate a treatment effect from purely observational data, and how
   you'd know if your estimate is any good."* (Part 2 §6–§15)
3. *"How does DML suppress regularization bias in ML-estimated nuisance functions?"* (Part 3 §6–§8)

**When a naive treatment-vs-control comparison looks suspicious.** Part 2 §5.1's decomposition
first (is it a confounding problem or a wrong-control problem?), then §6's confounder/mediator/
collider classification before adjusting for anything.

**When someone asks "why not just throw ML at the outcome and take the difference?"** Part 3 §2.2's
regularization-bias framing, then §6–§8's DML framework for the principled fix.

**Callout legend**

| | Meaning |
|---|---|
| 📚 **Background** | A prerequisite the slide assumed you already knew |
| 💡 **Key insight** | The one sentence from that section worth memorising |
| ⚠️ **Careful** | A slide gap, an ambiguity, a source-slide defect, or a place the standard presentation misleads |
| 🧪 **Worked example** | Real numbers computed to a real answer |
| 🎯 **Interview** | Phrased the way you would actually say it out loud |
| 🔬 **Research opportunity** | Where the field is genuinely still open |

---

## Quality review

This module went through the full `QUALITY_REVIEW_PIPELINE.md` three-lens audit
(Teacher/Student/Engineer) on 2026-08-30. Full findings, severities, and fixes are in
[`QUALITY_REVIEW.md`](QUALITY_REVIEW.md). Summary: 6 🔴 (factual errors/fabrications), 3 🟠 (content
gaps), 6 🟡 (polish) across the three files — all 15 findings fixed and independently re-verified
against the raw source frames. No companion web artifact exists yet for this module (`web/` contains
only `supervised-learning.html`).
