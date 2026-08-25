# Supervised Learning — Amazon ML Summer School

Self-study notes built from the verified slide extraction in
[`slides_deduped/`](../../slides_deduped/) and [`output/`](../../output/), following
[`NOTES_PIPELINE.md`](../../NOTES_PIPELINE.md).

These are **teaching documents, not summaries**. They define every term before using it, derive
every formula rather than asserting it, and work every example through to a final number. They are
written to be the only thing you need to read to master the lecture.

> ✅ This module has completed a full [`QUALITY_REVIEW_PIPELINE.md`](../../QUALITY_REVIEW_PIPELINE.md)
> pass — see [`QUALITY_REVIEW.md`](QUALITY_REVIEW.md) for the audit trail of what was checked and
> fixed. The word counts, slide counts, and section tables below reflect the post-fix state.

---

## Index

| # | Notes | Source deck | Instructor | Status | Words | Covers |
|---|---|---|---|---|---|---|
| 01 | [supervised-learning-01.md](supervised-learning-01.md) | `Lecture_01 - Module 1 Supervised Learning Part 1` (26 distinct slides) | Apoorva Singh | ✅ Complete · verified | ~19,600 | Problem formulation · Data splits & leakage · Bias–variance · Linear & logistic regression |
| 02 | [supervised-learning-02.md](supervised-learning-02.md) | `Lecture_02 - Module 1 Supervised Learning Part 2` (35 distinct slides) | Vikas Raykar | ✅ Complete · verified | ~31,100 | Loss functions · Optimisation & solvers · Evaluation metrics · Naive Bayes · KNN · SVM |
| 03 | [supervised-learning-03.md](supervised-learning-03.md) | `Lecture_03 - Module 1 Supervised Learning Part 3` (24 deck slides + 13-section hands-on notebook) | Sandeep Chatterjee | ✅ Complete · one flagged gap | ~33,100 | Decision trees · Bagging & Random Forests · Boosting · Multi-class · Class imbalance · Cross-validation · Hyperparameter tuning · Model selection · Calibration · Pipelines |
| 04 | [applied-scientist-practicum.md](applied-scientist-practicum.md) | Original extension | — | ✅ Required bridge | ~2,100 | Experiment design · ranking/retrieval · production ML · responsible ML · Amazon-style capstone · evidence portfolio |

**Read them in order.** Part 3 leans on Part 1's bias–variance decomposition (every ensemble method
in it is an application of that one idea) and on Part 2's metrics section (precision/recall/F1,
ROC-AUC vs PR-AUC, and the accuracy paradox all reappear, extended to three classes). Part 1 is the
prerequisite for Part 2: it introduces train/validation/test
and leakage (used throughout Part 2's metrics section), bias–variance (which every Part 2
hyperparameter is an instance of), OLS and the normal equation (Part 2 §9 explains why it doesn't
scale), and the sigmoid (Part 2 §6 derives its gradient). Part 2 cross-references back rather than
repeating derivations.

> ⚠️ **Note on file numbering.** The file previously at `supervised-learning-01.md` was built from
> `PDF Notes/Supervised Learning/Supervised Learning - 1.pdf`, but that PDF turned out to be a
> partial capture of the **Lecture 02** video (same instructor, same 40:39 runtime, Optimization
> section at ~12:10). It has been rewritten from the complete slide set and now lives at
> `supervised-learning-02.md`. Lecture 01 is a genuinely different deck by a different instructor and
> is still to be written.

---

## ⚠️ Read this before running the pipeline on another lecture

**`slides_deduped/` is lossy.** For Lecture 02 it contained 21 images where the deck has **35
distinct states**. The de-duplication step merged genuinely different slides, silently dropping:
MSE, MAE, the regression-loss comparison table, BCE, mini-batch SGD, the entire classification-metrics
block (confusion matrix, precision/recall trade-off, ROC-AUC & PR-AUC), both KNN content slides, and
both SVM content slides — roughly **40% of the teaching content**, including ~8 minutes of
uninterrupted material between 21:12 and 29:07.

**The recovery procedure**, which should be used for every remaining lecture:

1. The raw capture in `output/<Lecture>/` samples a frame at **each end of every stable stretch** —
   pairs roughly 3 seconds apart at each transition.
2. Cluster raw frames into runs by file size and timestamp; each run is one slide state.
3. Read the **last** frame of each run — animated slides build, so the later frame is the more
   complete one.
4. Cross-check against `output/<Lecture>/timestamps.txt`: any gap longer than ~90 seconds between
   consecutive deduped slides means content was dropped.

Do not trust a slide count from `slides_deduped/` as evidence the deck is complete.

**Lecture 03 confirmed the same failure again.** `slides_deduped/` held **78** images for a deck of
**24 slides** — this time the problem was the mirror image: dozens of mid-build duplicates of the
same slide, with no reliable way to tell a build state from a genuinely new slide. The recovery
procedure above worked, with one refinement worth recording:

> **Refinement — cluster the raw frames by *timestamp gap*, not by file size.** A speaker webcam tile
> is composited into every frame, so JPEG file size varies by tens of kilobytes *within* one static
> slide and is useless as a similarity signal. The capture's own structure is the reliable one: a gap
> of **≤10 s** between consecutive frames marks a real slide transition (the detector fires and grabs
> both sides of it), while a gap of **≈90 s** is the `max_gap` forced sample *inside* one stable
> slide. Cluster on that, then read the **last** frame before each transition.
>
> Even so, **read the head of each run too, not just the tail.** Several slides in Lecture 03 replace
> their body content rather than appending to it — the Cross Validation slide swaps its K-Fold block
> for its LOOCV block — so a tail-only read silently loses the first half.

Lecture 03 also had one **genuine, unrecoverable** capture gap: the K-Fold half of deck slide 14 fell
between two forced 90-second samples and was never captured. It is flagged in the notes rather than
reconstructed silently.

---

## What's in Part 1

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — capture note · where this sits (the four things Part 2 inherits) · 12 capabilities
· 5 prerequisites taught from zero (dataset vocabulary · parameter vs hyperparameter · expectation
and variance, including *what* is random in bias–variance · matrix notation for the normal equation ·
derivatives) · the big picture: three of the deck's four sections happen *before* the algorithm

**Part 1 — Introduction to ML & Problem Formulation** (§1–4)
- The Arthur Samuel (1959) definition taken seriously; the three paradigms tabulated by what fails
- The labelling bottleneck, and why semi-/self-supervised learning exist — read off the deck's own
  Venn diagram
- The 7-stage workflow with **the characteristic failure of each stage**, and a realistic
  three-loop trace showing why it's a circle
- Data drift vs concept drift, and why they need different responses
- **Hypothesis space** as the idea that repays thought; the three model families' boundary shapes and
  what each *cannot* represent
- A vague "reduce churn" request formulated into six explicit decisions — including the one that
  silently changes the metric

**Part 2 — Data Splits & Data Leakage** (§5–7)
- Why **three** splits, with the multiple-comparisons inflation quantified (~2 points from picking
  the best of 30)
- Sizing a test set from the effect you need to detect, worked: 1,500 examples → ±1.5pp
- Stratified splits (with the binomial spread computed: 4–16 positives where you expected 10) ·
  temporal splits with an ASCII contrast · k-fold cross-validation
- **All four leakage mechanisms**, each with a concrete instance *and its observable symptom*
- The µ_all = 33.6 vs µ_train = 4.5 calculation that shows preprocessing-before-split numerically
- Why the `Pipeline` matters more than the rule; the feature-availability audit table

**Part 3 — Model Fitting & Bias–Variance** (§8–11)
- The **two-number diagnostic** table, including the fourth row (validation better than training =
  you have a bug)
- Same data, three complexities: why a degree-6 polynomial through 7 points has *zero* training error
  and no information
- **The bias–variance decomposition derived** in four lines, including why the cross-term vanishes
- ⚠️ Where the "irreducible noise" term went — the deck's bulleted list and boxed formula disagree,
  and the reconciliation is the practically useful part
- Three models decomposed numerically, with the zero-bias model coming **last**
- The U-curve read three ways; ⚠️ double descent flagged honestly as where this breaks
- Bagging vs boosting, with the $\rho\sigma^2$ variance floor that explains Random Forest's feature
  subsampling
- **Learning curves** taught properly — the ASCII shapes, the four-row reading table, and why this
  answers a six-figure question in an afternoon

**Part 4 — Linear Regression** (§12–14)
- Why squared, and why *vertical* distance
- **OLS fitted by hand** to a real dataset, then **verified via the normal equation** — both give
  $\hat y = 2.2 + 0.6x$
- Why residuals must sum to exactly zero, and what it means if yours don't
- The normal equation **derived**, and the three reasons it doesn't scale
- All four assumptions with what breaks · how to detect · how to fix, and the one plot that checks
  three of them
- Ridge shrinkage computed across five λ values, showing it approaches zero and **never arrives**
- **Why L1 gives exact zeros and L2 never does**, derived from the penalty *derivatives*, then shown
  again as soft-thresholding numbers, then again as diamond-vs-circle geometry
- ⚠️ Lasso's instability on correlated features — the caveat the comparison tables hide

**Part 5 — Logistic Regression** (§15–16)
- Three reasons OLS fails on a binary target, and why only the third is fatal
- The sigmoid's three anchor points; the **log-odds derivation** and what $e^{w_j}$ means
- MLE ⟺ binary cross-entropy proved in three lines
- One prediction worked end to end, cross-checked via the log-odds, then re-decided at a different
  threshold
- Why the boundary is **linear** despite the sigmoid being non-linear — and how you'd get a circular
  one

**Closing** — ASCII dependency map · four threads · **10 interview questions** with model answers (4
combining concepts) · 8 depth probes · **3 whiteboard derivations** · a full churn scenario including
the intervention-changes-the-label failure mode · 3 Leadership Principles · **41-term glossary** ·
**26 check-yourself questions** · 10 ranked resources

**Interactive specs:** 4 blocks — leak it then fix it · the U-curve built from scratch · will more
data help? · the sigmoid, the boundary and the threshold.

</details>

---

## What's in Part 2

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter**
- Capture note — the 14 recovered slides, each cited by raw frame number and video timestamp
- What you'll understand after reading this — 13 concrete capabilities
- Before we start — 7 prerequisites taught from zero: the supervised setup & notation ·
  model/parameters/hyperparameters/training/inference · derivatives, partial derivatives, gradients ·
  convexity, local vs global minima, saddle points · probability (Gaussian, Laplace, likelihood,
  MLE, log-likelihood) · mean vs median · Big-O
- The big picture — the four-box framework (model → loss → optimiser → metric) and why boxes 2 and 4
  are allowed to disagree

**Part 1 — Loss Functions** (§1–8)
- What a loss function is; loss vs cost vs objective vs risk
- **MSE** — the four slide properties unpacked; full MLE-under-Gaussian derivation; the gradient
  scaling with error; proof it optimises toward the **mean**
- **MAE** — sub-gradients taught from zero; MLE-under-Laplace derivation; proof it optimises toward
  the **median**; why the constant gradient makes it converge slowly
- **Huber** — continuity *and* differentiability proved at the boundary; where the $-\delta/2$ offset
  comes from; gradient capping as free gradient clipping; choosing δ and its scale sensitivity
- The slide's comparison table, plus three rows it omitted, plus a loss-selection decision tree
- ⚠️ The deck's own ½-factor inconsistency between the MSE and Huber slides, called out
- **BCE** — Bernoulli derivation; entropy, cross-entropy and **KL divergence** taught from zero, with
  the proof that minimising CE *is* minimising KL; the $\hat p - y$ gradient cancellation derived in
  full; numerical stability and the `BCEWithLogitsLoss` trap
- **Cross-entropy & softmax** — one-hot encoding; softmax computed by hand; shift invariance and the
  log-sum-exp trick; temperature; the softmax-vs-sigmoid multi-class/multi-label distinction
- **Hinge** — the ±1 convention explained; functional margin; the sparsity result **derived** from the
  gradient; hinge vs logistic loss compared row by row; 0/1 loss and surrogate losses

**Part 2 — Optimisation & Solvers** (§9–13)
- ERM, empirical risk, and why overfitting is created the moment you write $\frac1n\sum_i$
- The **normal equation derived**, its three scaling failures, and ridge as the fix
- **Batch GD** — five steps worked by hand; the $\eta < 2/L''$ divergence threshold **derived**;
  Hessian and eigenvalues explained; a learning-rate failure-mode table read off the loss curve
- What "till convergence" actually means — five stopping tests, early stopping, the patience trap
- **SGD** — the unbiasedness argument; epochs vs iterations (a million times more progress for the
  same compute); why noise helps; Robbins–Monro
- **Mini-batch SGD** — the $1/\sqrt B$ noise law tabulated against cost; why batch sizes are powers of
  two (GPU warps and tiling); why shuffle every epoch; the linear scaling rule and warmup
- **Improvements** — the deck lists eight names with no explanation; all eight taught as a lineage
  where each fixes the previous one's specific failure: momentum (with the $\frac{1}{1-\beta}=10\times$
  geometric series), Nesterov, Adagrad (and why it dies), RMSProp, **Adam with three steps worked by
  hand** showing the step is exactly η, bias correction, **AdamW's decoupling bug** explained, Muon
  and SOAP now cited and confirmed (Keller Jordan, Oct 2024; Vyas et al., ICLR 2025), cosine
  annealing with its endpoints checked
- The optimiser family tree as an ASCII diagram

**Part 3 — Evaluation Metrics** (§14–18)
- Loss ≠ metric, tabulated; train/validation/test; **data leakage** and its five common paths
- **Confusion matrix** — the two-word naming rule that makes it derivable rather than memorised;
  Type I/II; ⚠️ the sklearn orientation trap
- The **accuracy paradox** worked twice: a 99.0%-accurate useless model against a 94.8%-accurate good
  one; specificity, TPR, FPR, NPV, balanced accuracy, MCC
- **Precision–recall trade-off** — why it's mechanical, not empirical; a threshold sweep over 10
  examples producing five different products from one model; why **threshold moving is free** and
  should always precede SMOTE
- Why F1 uses the **harmonic mean**, demonstrated on a classifier the arithmetic mean rates at 0.505
  and the harmonic mean correctly rates at 0.0198; $F_\beta$; macro vs micro vs weighted
- **ROC curve built by hand from raw scores; AUC computed two independent ways** — trapezoid and
  pairwise ranking — both giving **0.8125**
- The fraud example where ROC-AUC looks superb (FPR 0.010) and precision is **8.3%**, with the
  denominator table explaining exactly why ROC hides it
- **Regression metrics** — RMSE vs MSE and why units matter; R² read as "fraction of the baseline's
  error removed"; a worked **negative R²**; why RMSE ≥ MAE always and what the gap diagnoses;
  MAPE, adjusted R², RMSLE
- Metric uncertainty — why a 2-point win on 100 examples is noise; standard errors, bootstrap CIs,
  and **paired** comparison

**Part 4 — Naive Bayes** (§19–20)
- Bayes' theorem derived in two lines; prior/likelihood/posterior/evidence; why $P(\mathbf{x})$ can be
  dropped and when it can't
- The naive assumption and the **counting argument** ($2^d$ vs $d$, tabulated to $d=10{,}000$)
- Why a false assumption still classifies well — the argmax is more robust than the values
- **Laplace smoothing** and **log-space computation**, both omitted by the slides and both mandatory
- Full worked spam classification: priors → smoothed likelihoods → products → normalisation →
  **log-space cross-check giving the identical 0.7955**
- Gaussian / Multinomial / Bernoulli variants compared by what they assume for $P(x_j \mid C_k)$
- Generative vs discriminative, tabulated; the Ng & Jordan (NIPS 2001) small-data result — confirmed
  and corrected: NB's error approaches its asymptote in $O(\log d)$ examples vs. logistic regression's
  $\Omega(d)$, not the same rate as an earlier draft stated
- **Naive Bayes as a linear classifier**, derived from the log-odds

**Part 5 — K-Nearest Neighbors** (§21–22)
- Lazy vs eager learners tabulated; the exact cost trade, and how vector databases pay it
  approximately
- Euclidean / Manhattan / Minkowski (with the $p=1$ collapse checked) / cosine, and when each applies
- **Worked example where K = 1, 3 and 5 give three different answers** for the same query
- Bias–variance taught from zero; why K=1 always gives 100% training accuracy and why that makes
  training accuracy useless for choosing K; why odd K; distance weighting
- **Curse of dimensionality** explained two ways — concentration of measure ($\sqrt d$ vs $d$) and a
  volume table showing $10^{20}$ points needed at $d=20$
- Feature scaling worked example where the prediction flips **purely because of units**
- KD-trees, Ball-trees, and where they stop working; HNSW / IVF-PQ / FAISS
- KNN for regression, and a clear statement that it **cannot extrapolate**

**Part 6 — Support Vector Machines** (§23–24)
- Why "which separating hyperplane?" is a real question; hyperplane geometry and the normal vector
- **Margin $= 2/\|w\|$ derived** from the scaling freedom — and the observation that the "1" in hinge
  loss *is* the margin normalisation
- Hard margin, and why it has no feasible solution on real data
- Soft margin, slack variables, and what C really does (including why C runs backwards,
  $\lambda = 1/2C$)
- **Proof that soft-margin SVM = hinge loss + L2 regularisation**, by eliminating the slack variables
- Margin computed by hand and cross-checked against both the point distance and the hinge loss
- **Kernel trick** — grounded in the actual slide, no longer reconstructed. XOR separated explicitly
  in 3-D with a concrete weight vector; the polynomial kernel expansion **proved** to equal the
  quadratic feature map, then both routes computed numerically to the same **121**
- Linear / RBF / polynomial compared; RBF computed; RBF-SVM related back to distance-weighted KNN;
  γ and C interaction and why they're tuned jointly

**Closing**
- Putting it together — full ASCII dependency map, plus **five threads** running through the lecture
- Interview prep — Amazon Applied Scientist: **12 core questions** with model answers (4 requiring
  two concepts combined), a depth-probe table, **3 whiteboard-ready derivations**, a full
  counterfeit-detection scenario (framing → data → model → metric → failure modes → what to ship),
  and 3 Leadership Principles with concrete evidence
- Glossary — **49 terms**, alphabetical
- Check yourself — **25 questions**, each tagged with the section that answers it
- Going deeper — 11 resources in 3 tiers with difficulty ratings; every citation that carried a
  ⚠️ verify-this flag (Huber 1964, Barron's general robust loss, Muon, SOAP, Goyal et al., Ng & Jordan,
  Rahimi & Recht, Jacot et al.) has since been checked against a primary source — all confirmed, one
  (the Huber efficiency constant, 1.35→1.345) and one (Ng & Jordan's sample-complexity rates) corrected

**Interactive specs:** 7 `interactive` blocks — outlier tug-of-war · the δ dial · the cost of
confidence · three descents on one surface · linked threshold/ROC/PR views · K scaled and unscaled ·
kernel/C/γ. Each has a `fallback` so the notes teach completely as plain text.

</details>

---

## What's in Part 3

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — capture note (raw-frame recovery + one honestly flagged gap) · 16 capabilities ·
7 prerequisites taught from zero (nodes and class proportions · log₂ and what a *bit* is ·
bias–variance recap · **the variance of an average of correlated estimators, derived** · the
bootstrap · residuals as negative gradients · macro-averaging) · the big picture: trees → prune →
bag → boost, then everything that isn't the model

**Part 1 — Decision trees** (§1–5)
- Recursive partitioning; why the boundary is **axis-parallel**, and what that costs on a 45° split
- **Gini derived** from "two random draws disagree"; why the 0.5 ceiling is binary-only and the real
  bound is $1-1/K$
- Entropy derived from bits; information gain, and why the size-weighting is the part that matters
- **One split worked with both criteria to a final number** — Gini gain 0.1057, IG 0.2091 bits
- Greedy splitting grounded in the actual NP-completeness result, not hand-waved
- **Tree variance identified as argmax variance amplified by recursion** — the mechanism, not the slogan
- The depth sweep read three ways: default full tree **0.697** vs depth-5 **0.757**
- Cost-complexity pruning, and the observation that it is ridge with leaves in place of weights
- ⚠️ The categorical trap the deck's "handles mixed types" hides, caught in the instructor's own code
- Feature importance, and **the three ways impurity importance lies**

**Part 2 — Ensembles** (§6–8)
- **Proof that bagging leaves bias untouched**, in two lines
- **The $\rho\sigma^2$ floor tabulated** — why 10,000 trees at ρ=0.9 is no better than 100
- Random Forest reframed: *deliberately handicap every tree to decorrelate the ensemble*
- **The 36.8% OOB fraction derived**, with a convergence table; and when OOB is invalid
- **One AdaBoost round by hand**, ending in the proof that the reweighted error is *exactly* 0.5
- Gradient boosting as descent in **function space**; one round worked at η=0.1 and η=1.0
- XGBoost/LightGBM/CatBoost compared, and ⚠️ why "parallelised" does *not* mean what it does in RF
- The lecture's own boosting table read as a diagnosis: HistGB at **train 0.999 / test 0.755**
- **The label-noise experiment**, with the honest note that RF's wobble is inside ±1.4pp

**Part 3 — The machinery around the model** (§9–15)
- OvR / OvO / softmax; **macro-F1 computed by hand from a 3×3 confusion matrix to 0.657**, then
  again for OvO to 0.695 — and the finding that *the entire gap is one minority class*
- **`class_weight='balanced'` computed on the real counts**, verified to give each class equal weight
- SMOTE's three failure modes; **exactly which rows contaminate which** when it precedes the split
- Focal loss tabulated at γ=2 — easy examples silenced 10,000×
- The imbalance table read the right way: **accuracy −2.7 points, minority F1 +10.6**
- ⚠️ Why SMOTE + class_weight is a **no-op by construction**, not a coincidence
- Cross-validation; **why stratification lowers variance, quantified** via the Binomial fold spread
- Why LOOCV is high-variance — and that it is the ρσ² floor again, in disguise
- **The 60-trial random-search result derived**, and why its independence from dimension is the point
- ⚠️ The lecture's grid search *beat* random search — the four reasons that isn't a contradiction
- No Free Lunch stated correctly, and what it does **not** imply
- Calibration: reliability diagrams, Platt vs isotonic, **Brier and ECE both worked to a number**
- **Why calibration can never change AUC**, and the two consequences
- The Pipeline leakage bug that passes code review, and the one-line fix

**Part 4 — The end-to-end demo** (§16)
- The 4,424-student dataset; the stratified split verified to 0.04pp
- **Every number the notebook printed, in one table** — baseline 0.499 through final test F1 0.679
- The final artifact: `GridSearchCV(Pipeline(ColumnTransformer, HistGB))` → calibrated → `joblib.dump`
- The one misclassified round-trip row, and the three earlier results that predicted it

**Closing** — ASCII dependency map with cross-validation drawn as the spine · **five threads**
(including *"three results in this lecture are inside the measurement error"*) · **12 interview
questions** with model answers (5 combining concepts) · 13 depth probes · **3 whiteboard
derivations** · a full delivery-delay scenario including the intervention-changes-the-label failure
mode · 4 Leadership Principles · **53-term glossary** · **50 check-yourself questions** ·
12 ranked resources — the 3 citations that carried a ⚠️ verify-this flag (Hyafil & Rivest 1976,
Wolpert's No Free Lunch 1996, and the GOSDT line of work) have since been confirmed against primary
sources

**Interactive specs:** 5 blocks — impurity as the class mix changes · prune it and watch both curves
· the correlation floor · the imbalance trade made visible · the reliability diagram before and after.

</details>

---

## Reading guide

**First pass (3–4 hours).** Read linearly. Do not skip *Before we start* even if parts look familiar
— later sections reference it by name. Skip the `interactive` spec blocks entirely.

**Second pass.** Work every 🧪 example on paper *before* reading the solution. This is where the
learning actually happens.

**Weekly.** Glossary + Check yourself. Both are built for spaced repetition.

**Before an interview.** The *Putting it together* threads, the 3 whiteboard derivations, and the
depth-probe table — in that order.

**Callout legend**

| | Meaning |
|---|---|
| 📚 **Background** | A prerequisite the slide assumed you already knew |
| 💡 **Key insight** | The one sentence from that section worth memorising |
| ⚠️ **Careful** | A slide gap, an ambiguity, or a place the standard presentation misleads |
| 🧪 **Worked example** | Real numbers computed to a real answer |
| 🎯 **Interview** | Phrased the way you would actually say it out loud |
| 🔬 **Research opportunity** | Where the field is genuinely still open |

---

## Building the interactive site

Every figure that teaches better as something you can *move* is followed by a fenced block tagged
`interactive`. It is YAML with these keys:

| Key | Meaning |
|---|---|
| `type` | `slider` · `animation` · `simulator` · `quiz` · `graph` · `diagram` |
| `title` | Short name |
| `concept` | Which concept this teaches |
| `control` | What the reader manipulates |
| `observe` | What visibly changes |
| `insight` | The specific realisation this produces |
| `fallback` | What a static reader sees instead — **always present** |

The site builder reads these; the prose never depends on them.

## Applied Scientist bridge — do this after Part 3

The three lectures make you strong at **choosing and evaluating a supervised model**. An Applied
Scientist interview and job additionally ask whether you can make a defensible decision in a changing
product: define the counterfactual, choose an online metric and guardrails, reason about ranking and
retrieval, ship within latency and cost budgets, and detect harm after launch. Those are not optional
"MLOps extras"; they decide whether an offline gain helps customers.

Read and complete [the Applied Scientist practicum](applied-scientist-practicum.md) after Part 3.
It turns the lecture material into one portfolio-quality Amazon-style case study, with a scoring rubric
and evidence checklist. The Markdown also contains three `interactive` specifications for the future
study site: offline-versus-online metric disagreement, a retrieval/ranking trade-off, and a monitoring
incident drill.

---

## Key takeaway per lecture

| # | One-sentence takeaway |
|---|---|
| 01 | Three of the four things that determine whether a model works — how the problem is formulated, whether the data pipeline is honest, and whether you are bias-limited or variance-limited — are settled before you pick an algorithm; and the two algorithms you do pick here, OLS and logistic regression, are the baselines everything else has to beat. |
| 02 | Choosing a loss function is choosing a noise model, choosing an optimiser is choosing how to trade gradient accuracy against gradient cost, and choosing a metric is choosing what the business actually means by "working" — and Naive Bayes, KNN and SVM are three different answers to what a model even is, with the SVM turning out to be nothing more than hinge loss plus L2. |
| 03 | A single decision tree has one fatal flaw — recursion amplifies the instability of every argmax it takes — and pruning, bagging and boosting are three answers to that one flaw, distinguishable by which half of the bias–variance decomposition each attacks; while the second half of the lecture is the reminder that a number coming out of a model is not yet a decision, and that cross-validation is the instrument every remaining choice depends on, which is why a differences-smaller-than-the-error-bar habit matters more than any single algorithm. |
