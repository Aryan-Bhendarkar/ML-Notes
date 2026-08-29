# Unsupervised Learning — Amazon ML Summer School

Self-study notes built from the verified slide extraction in
[`output/`](../../output/), following [`NOTES_PIPELINE.md`](../../NOTES_PIPELINE.md).

These are **teaching documents, not summaries**. They define every term before using it, derive
every formula rather than asserting it, and work every example through to a final number. They are
written to be the only thing you need to read to master the lecture.

This module has gone through a full `QUALITY_REVIEW_PIPELINE.md` audit pass — see
[`QUALITY_REVIEW.md`](QUALITY_REVIEW.md) for the complete findings and fixes.

---

## Index

| # | Notes | Source deck | Status | Words | Covers |
|---|---|---|---|---|---|
| 01 | [unsupervised-learning-01.md](unsupervised-learning-01.md) | `Lecture_10 - Module 4 Unsupervised Learning Part 1` (**~122 slides** + a full live-coded Jupyter demo) | ✅ Complete · no content gaps | ~23,200 | Distance as a modeling choice · Mahalanobis distance · **K-Means** (algorithm, convergence proof, K-Means++, elbow method) · **Hierarchical Clustering** (dendrograms, linkage) · **DBSCAN** (core/border/noise, ε-selection) · intrinsic/extrinsic evaluation metrics (including Davies-Bouldin) · a full live demo clustering real handwritten digits |
| 02 | [unsupervised-learning-02.md](unsupervised-learning-02.md) | `Lecture_11 - Module 4 Unsupervised Learning Part 2` (**50 slides**) | ✅ Complete · TOC-vs-content gap flagged | ~14,850 | **Gaussian Mixture Models & the EM Algorithm, derived in full**: the ELBO/KL decomposition derived from scratch · the E-step and M-step derived (not just stated), with a fully numeric worked example · the mountain-climbing intuition · a live responsibility-visualization demo · PLSA as GMM's discrete counterpart |
| 03 | [unsupervised-learning-03.md](unsupervised-learning-03.md) | `Lecture_12 - Module 4 Unsupervised Learning Part 3` (**59 slides**) | ✅ Complete · Diffusion/Flow Matching not in this video | ~17,560 | **Generative Modeling Overview** (K-Means→GMM→neural spectrum) · **VAEs derived in full** (ELBO via Jensen, reparameterization, posterior collapse) · **GANs derived in full** (density-ratio-via-classifier, JSD, minimax game, vanishing gradients, mode collapse, WGAN, DCGAN) |
| 04 | [unsupervised-learning-04.md](unsupervised-learning-04.md) | `Lecture_13 - Module 4 Unsupervised Learning Part 4` (**22 deduped slides**) | ✅ Complete · module-complete | ~12,900 | **Diffusion Models** (forward process, reverse process, hierarchical VAE connection, ELBO and denoising matching, noise prediction = score prediction, ancestral sampling, named production systems) · **Flow Matching** (probability paths, velocity fields, conditional flow matching loss, Euler ODE solver, unifying connection: diffusion is a special case of flow, named production systems) |


⚠️ Per this course's now-established pattern (confirmed in
[Dimensionality Reduction](../Dimensionality%20Reduction/README.md) and now again here), **a forward
reference — whether a prior lecture's preview or a deck's own table of contents — is a reasonable hint
but never a guarantee** of what one specific video actually delivers. Lecture 10's GMM preview turned
out accurate (confirmed from two independent points in that lecture). But Lecture 11's *own* Table of
Contents promised 5 sections beyond GMM/EM (Generative Modeling Overview, VAEs, GANs, Diffusion Models,
Flow Matching) that this specific video does **not** deliver — the recording ends at the GMM/EM summary
slide. Always verify a lecture's actual captured content against its own deck before writing notes,
even when the deck's own table of contents states otherwise.

**Prerequisites.** This module leans on
[`Dimensionality Reduction`](../Dimensionality%20Reduction/) throughout: distance concentration in high
dimensions (cited directly for DBSCAN's high-dimensional failure mode), mutual information (reused
verbatim as NMI, a clustering-evaluation metric), t-SNE's local-structure-only guarantee (used to
correctly interpret the live demo's visualization), and PCA as a pre-clustering/pre-evaluation step
(the mechanism behind the live demo's central silhouette-score resolution). Read Dimensionality
Reduction first if you haven't — Lecture 10 assumes it throughout rather than re-deriving it.

---

## Capture quality

### ✅ Lecture 10 — excellent

175 raw frames over 63 minutes. Every content slide has a fully-built state, every formula and figure
is legible, and the back half of the lecture is a **live-coded Jupyter notebook demo** — captured
densely enough that every printed number in the notebook's output is readable. This is the first
lecture in the course where a live coding demo is reproduced with its actual real-dataset numbers
(901 real handwritten digit images, real purity/silhouette/PCA results) rather than described from a
distance.

**No content gaps.** The one thing to flag: the deck ends on a Gaussian Mixture Models teaser and then
pivots straight into the live demo — GMM's actual mechanics are Part 2's subject, not this lecture's,
and the notes are explicit about not claiming otherwise.

**The instructor is not named** anywhere in the recording — the webcam tile carries no label —
matching [Dimensionality Reduction Part 1](../Dimensionality%20Reduction/dimensionality-reduction-01.md),
which had the identical gap.

### ✅ Lecture 11 — excellent, and unusually rigorous

50 raw frames over 52 minutes. Every content slide has a fully-built state, and — genuinely unusual for
this course — **every major equation is captured with the instructors' live handwritten derivation
annotations still visible on screen**: circled terms, arrows connecting symbols to their meaning, and a
hand-drawn sketch of the monotonic-improvement guarantee. No content gaps *within what was captured*.

⚠️ **But the deck's own Table of Contents promises more than this video delivers.** It lists seven
sections (Notation, GMM & EM, Generative Modeling Overview, VAEs, GANs, Diffusion Models, Flow
Matching); this recording covers only the first two. The recording ends cleanly at the GMM/EM summary
slide, whose final bullet is a one-sentence forward pointer to VAEs — not covered content. See the
Index table above.

Both instructors are named on the title slide: **Dhruv Bhardwaj** (Applied Scientist II) and
**Ayush Raj** (Applied Scientist).

### ✅ Lecture 12 — excellent, and confirms the Part 2 gap's hypothesis

59 raw frames over 50 minutes. Every content slide has a fully-built state, with the same live
handwritten-annotation capture quality as Lecture 11. **No content gaps** — the deck's own internal
page counter confirms the final captured slide (a GANs summary) is genuinely the deck's last slide, not
a truncated capture.

This lecture delivers exactly the "Generative Modeling Overview," "Variational Autoencoders," and
"GANs" sections Lecture 11's own Table of Contents promised but didn't deliver — confirming that
guess. It does **not** cover "Diffusion Models" or "Flow Matching," the two remaining sections from
that same original Table of Contents — the leading (unverified) hypothesis is that Lecture 13 covers
those.

**The instructor is not named** anywhere in the recording — matching Lecture 10 and
[Dimensionality Reduction Part 1](../Dimensionality%20Reduction/dimensionality-reduction-01.md).

### ✅ Lecture 13 — good, with dedup caveats

22 deduped slides over 40 minutes (from 52 raw frames). The instructor explicitly states at the start:
*"I'm going to cover these slides a bit quickly because we are short of time."* This is a rapid
survey lecture — the notes therefore teach each concept in full depth far beyond what the slides show.

⚠️ **Dedup caveats:** 30 of the 52 raw frames were not kept as a slide's representative frame (52
raw → 22 deduped). A frame-by-frame audit of all 30 dropped frames, done as part of this module's
quality review, confirmed every one is a genuine intermediate build-state or progressive-annotation
duplicate of an adjacent kept slide — no unique content lost to dedup mechanics itself. The full
YouTube transcript was obtained and verified against each slide; no significant content gaps were
identified. *(An earlier version of this note stated "12 out of 34 raw runs were dropped," which
does not reconcile with the actual 52-raw/22-deduped frame counts confirmed directly from
`output/` and `slides_deduped/`; corrected here.)*

This lecture delivers the "Diffusion Models" and "Flow Matching" sections that Lecture 11's Table
of Contents promised and Lecture 12 didn't cover — **completing the Unsupervised Learning module.**

**The instructor is not named** — matching Lectures 10 and 12.

---

## What's in Part 1

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — six sections with runtimes · capture notes · 9 capabilities · 6 prerequisites
taught from zero (supervised vs unsupervised as three questions, not one · the label-scarcity economic
argument, quantified · $L_p$ norms and their unit-ball *shapes* as the real point, not just the formula
· covariance whitening, recapped for Mahalanobis distance · the sum-of-squares objective and why
squaring specifically · entropy/mutual information, recapped for NMI) · a full-lecture ASCII map built
around the deck's own "one question drives everything" framing table (K-Means/GMM/LDA/VAE/GAN, five
answers to one question)

**PART A — Distance, and the Four Families** (§1–§4)
- §1 Distance as a modeling choice, made concrete: the same radius drawn as a diamond, circle, and
  square around one point, and what that costs you before any clustering algorithm even runs
- §2 **Mahalanobis distance derived in one line** from whitening ($d_M$ = Euclidean in whitened space),
  walked through the deck's own three-panel figure (raw Euclidean → Mahalanobis ellipses → whitened
  sphere) · 💡 the sentence that pre-loads all of Part 2: *"GMM learns one Mahalanobis ellipse per
  cluster — the only real difference from K-Means"*
- §3 Jaccard similarity and DTW, for when your data isn't a vector at all
- §4 **Four families of clustering**, run on identical data (two rings + a blob) so the *same figure*
  shows all four definitions of "cluster" failing or succeeding differently — the map the entire
  lecture is organised around

**PART B — K-Means** (§5–§10)
- §5 The five-step algorithm · 🧪 **the convergence proof derived properly**: both alternating steps
  shown individually non-increasing on $J$, with the centroid-update step's optimality proved by
  differentiating and setting to zero · 🧪 a full iteration worked by hand on six 1-D points, including
  a bad-initialization case that (coincidentally) self-corrects, with an explicit warning that this
  isn't guaranteed in general
- §6 The 3-line sklearn API, with 💡 **`n_init=10`'s real job**: silently hedging against the
  local-minimum risk §5 just proved is real, by brute-force restarting and keeping the best run
- §7 The elbow method · 🧪 **why distortion always decreases as K grows, proved in one line** (any
  $K$-solution splits into an equally-good $(K{+}1)$-solution) — and why that makes "minimize $J$"
  a degenerate, useless selection rule
- §8 Where K-Means fails, unified into **one** underlying cause: the objective implicitly assumes
  every cluster is an isotropic Gaussian blob of comparable size
- §9 K-Means++ · why $d^2$-weighting specifically (not $d$) · a demo figure with a genuinely
  interesting caveat flagged: two initializations landing at the *same* final distortion despite one
  being labeled "bad" — a low $J$ alone doesn't certify correctness, foreshadowing §29
- §10 Key takeaways, closing on 🎯 the "start with K-Means, let the failure pattern tell you what's
  next" rule of thumb

**PART C — Hierarchical Clustering** (§11–§16)
- §11 The transition, framed explicitly as "two independent relaxations of K-Means' two assumptions" —
  a structural device worth noticing on its own
- §12 The bottom-up algorithm · 🧪 $O(n^3)$ time complexity derived from first principles, with the
  practical ~10K-point ceiling this implies stated plainly
- §13 Dendrograms · 💡 the real payoff of "no K needed": **one run gives every value of K**, with zero
  additional computation, versus K-Means' full-rerun-per-K requirement
- §14 Linkage methods, run side by side on identical crescent-shaped data so the contrast is visible ·
  🧪 **Ward linkage's connection to K-Means derived explicitly**: its merge criterion *is* K-Means'
  objective function, applied incrementally rather than globally — the reason it's the right default
- §15 sklearn vs scipy, and the practical two-library workflow (scipy to explore, sklearn to fit final)
- §16 Key takeaways · 💡 the structural parallel to §5.2 named explicitly: greedy, irreversible merges
  carry the identical "locally optimal, not globally guaranteed" caveat K-Means' convergence proof did

**PART D — DBSCAN** (§17–§23)
- §17 The motivating comparison: DBSCAN succeeding on the exact crescent shape that broke both earlier
  algorithms
- §18 Core/border/noise, and 💡 why the noise category is DBSCAN's single structural advantage over
  every other algorithm in this lecture — no other method has a built-in "doesn't belong anywhere" label
- §19 The algorithm · ⚠️ the precise meaning of "noise (for now)" — provisional, not final · 🧪 a full
  6-point worked example run by hand, correctly separating two clusters and one outlier with no $K$
  specified
- §20 sklearn API · ⚠️ **two genuinely load-bearing warnings, both derived, not just stated**: why
  scaling matters *structurally* more for DBSCAN's absolute-radius $\epsilon$ than for K-Means, and
  exactly what "transductive, no `.predict()`" rules out in practice
- §21 Choosing ε and MinPts via the k-distance plot, with 💡 the structural parallel to the elbow
  method named explicitly (same "sort, plot, find the bend" pattern, applied to a different quantity)
- §22 Strengths and limitations, with two of the five limitations traced directly back to named results
  from [Dimensionality Reduction](../Dimensionality%20Reduction/) (distance concentration; the
  single-global-parameter problem shared with K-Means' own §8)
- §23 Key takeaways · 💡 the "cluster once, then train a classifier on the labels" pattern for scoring
  new points against a transductive algorithm — a genuinely reusable trick beyond DBSCAN itself

**PART E — Evaluating Clusters Without Labels, and the Live Demo** (§24–§30)
- §24 The GMM teaser, and an explicit, honest statement of what this lecture does and does not cover
  going forward
- §25–§27 Intrinsic metrics (distortion, silhouette, Calinski-Harabasz) and extrinsic metrics (ARI,
  NMI) defined and derived · 📚 NMI connected explicitly back to [Dimensionality Reduction Part
  1](../Dimensionality%20Reduction/dimensionality-reduction-01.md)'s mutual information — the identical
  quantity, reused · a which-metric-for-which-algorithm table, closing on 🎯 "metrics can hide
  pathologies a scatter plot reveals instantly"
- §28 🧪 **A complete, real Jupyter demo, reproduced with its actual output**: 901 real handwritten
  digit images (5 classes, verified sample counts), K-Means at K=5 with zero labels, **93.8% purity
  achieved unsupervised**, a t-SNE visualization, and centroid images that are literally the pixel-wise
  average of each recovered digit class
- §29 🧪 **The best worked demonstration in the lecture**: why silhouette = 0.187 on a 93.8%-*correct*
  clustering is not a contradiction — decomposed to real numbers ($a{=}8.06$, $b{=}9.87$) and resolved
  into the single sentence worth memorising: *"Purity asks 'is this point in the right cluster?'
  Silhouette asks 'is this point confidently inside its cluster?' Both are correct. They measure
  different things."*
- §30 🧪 **The fix, proved with a full real table**: silhouette climbing from 0.187 (64 raw dimensions)
  to 0.326 (10 PCA dimensions) while accuracy stays flat at ~93–94% throughout — direct numeric proof
  that the earlier "low" silhouette was a dimensionality artefact, not an algorithm failure, closing
  with a three-step practical checklist for evaluating real unsupervised clusterings

**Closing** — a full-lecture ASCII dependency map tracing "what is the simplest explanation of this
data?" through all three algorithms into evaluation and the live demo · 💡 two threads (the same
"locally optimal, not globally guaranteed" proof recurring across K-Means and Hierarchical clustering;
the live demo tying together nearly every earlier concept in one worked example) · **9 interview
questions** with model answers (2 combining concepts) · 8 depth probes · **3 whiteboard derivations** ·
a fraud-ring-detection scenario built directly on the deck's own named Amazon use case · 4 Leadership
Principles · **32-term glossary** · 12 check-yourself questions · 10 ranked resources.

**Interactive specs:** 4 — §5/§6 the K-Means assign/update loop, §9 K-Means++ seed selection, §13
cutting the dendrogram, §19 DBSCAN's core-point expansion. Beyond these, the deck's own richest
"interactivity" is the live-coded Jupyter demo itself, reproduced in full with real output in §28–§30.

</details>

---

## What's in Part 2

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — the deck's own promised-vs-delivered gap flagged explicitly before anything else ·
four sections with runtimes · capture notes · 8 capabilities · 4 prerequisites (the module-wide shared
notation, including $z$ as a first-class citizen alongside $y$ · KL divergence's non-negativity,
recapped as the single load-bearing fact the whole derivation rests on · marginalization · Jensen's
inequality, named as the alternative derivation route this lecture doesn't take) · a full-lecture ASCII
map tracing one continuous argument from Part 1's hard-clustering limitation through to PLSA and a
forward pointer to VAEs

**PART B — Gaussian Mixture Models** (§1–§2)
- §1 GMM as a universal density estimator, with 💡 the universal-approximation framing connected
  explicitly to [Deep Neural Networks Part 1](../Deep%20Neural%20Networks/deep-neural-networks-01.md)'s
  identical argument for neural networks, applied here to densities instead of functions
- §2 GMM as a latent-variable model · **Generation vs. Discovery**, named as the two directions of one
  model — the same duality, the notes argue, that organises everything the deck's own (undelivered)
  Table of Contents lists next

**PART C — Expectation-Maximization, Derived** (§3–§7)
- §3–§4 The problem stated precisely: why direct MLE is structurally intractable the moment a latent
  variable sits inside a log of a sum — not a lack of cleverness, a genuine absence of a closed form
- §5 🧪 **The ELBO/KL decomposition derived symbol-by-symbol from the definition of KL divergence**,
  verified as an *exact identity* (not an approximation) by adding the two terms back together and
  confirming they collapse to the original quantity · the "two horizontal bars" picture redrawn as the
  single image worth memorising over either update formula
- §6 🧪 **The E-step derived, not stated**: shown to be the unique KL-minimizing choice of $q$, arising
  mechanically from the decomposition itself · 🧪 **the M-step's three-inequality convergence chain
  derived in full** · 💡 the GMM M-step compared symbol-by-symbol against
  [Part 1](unsupervised-learning-01.md)'s K-Means update, with the precise limiting argument for why
  K-Means **is** GMM/EM with every responsibility collapsed to 0 or 1
- §7 The mountain-climbing analogy, with every term mapped back onto a specific symbol from §5–§6's
  derivation in one table · 🧪 the live responsibility-visualization demo, contrasted directly against
  every hard-clustering figure in [Part 1](unsupervised-learning-01.md)

**PART D — Related Topics & Summary** (§8–§9)
- §8 PLSA derived as the identical machinery applied to discrete text data · 📚 the connection back to
  [Dimensionality Reduction Part
  2](../Dimensionality%20Reduction/dimensionality-reduction-02.md)'s NMF-with-KL-loss equivalence,
  reached here from the opposite (probabilistic) direction — the same object, two independent routes
- §9 🧪 **"converges to a local optimum" unpacked as the exact structural twin of K-Means' own
  guarantee** · 🧪 **"continuous latents are completely intractable" unpacked as the precise reason the
  deck's own Table of Contents needed VAEs at all** — with the VAE's ELBO objective, approximate
  posterior, and joint-gradient-descent training explained as this lecture's exact machinery,
  generalised one level

**Closing** — a full-lecture ASCII map showing the derivation's three-act shape · 💡 the single thread
connecting K-Means, EM, and (looking ahead) VAEs as one reusable proof template · **9 interview
questions** with model answers (2 combining concepts) · 6 depth probes · **3 whiteboard derivations** ·
a soft-customer-segmentation scenario extending [Part 1](unsupervised-learning-01.md)'s own named
Amazon use case · 3 Leadership Principles · **18-term glossary** · 12 check-yourself questions · 10
ranked resources.

**Interactive specs:** none in this deck — its interactivity is a single live-coded
responsibility-visualization figure, reproduced in §7.

</details>

---

## What's in Part 3

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — the Part 2 promised-vs-delivered gap resolved explicitly (this lecture delivers
the first three of Part 2's five undelivered sections, and stops cleanly before Diffusion/Flow
Matching) · three sections with runtimes · capture notes · 8 capabilities · 5 prerequisites (everything
from Part 2 assumed fresh · Jensen's inequality, now used directly · Binary Cross-Entropy loss ·
Bayes' rule · sampling by transformation of random variables) · a full-lecture ASCII map showing the
VAE/GAN split as two different responses to Part 2's own closing limitation

**PART A — Generative Modeling Overview** (§1–§2)
- §1 What generative models are, and why an ordinary classifier isn't one · 🧪 the three-panel
  figure placing K-Means, GMM, and a neural generative model on one spectrum of increasingly
  expressive density approximations
- §2 Conditional generative models, read as one unifying pattern (text-to-image, image-to-text,
  image-to-image, speech-to-text, and translation are all "learn $p(x|c)$" with only the modality
  swapped)

**PART B — Variational Autoencoders** (§3–§9)
- §3 Why VAEs, and the two questions one VAE answers at once: can I sample, and can I score how
  typical a sample is
- §4 🧪 **The VAE's ELBO bound derived directly via Jensen's inequality**, the faster route
  Part 2's Prerequisite 4 named in advance
- §5 🧪 **The simplified ELBO derived from §4's bound**: reconstruction term minus regularisation
  term, split out symbol-by-symbol
- §6 🧪 **The reparameterization trick derived from necessity, not convenience**: why gradients
  cannot flow through a stochastic sampling node, and how moving randomness into a parameter-free
  $\epsilon$ fixes it
- §7 The reconstruction-vs-regularisation tension named explicitly, as the direct setup for §8
- §8 🧪 **Posterior collapse derived mechanistically**, not just named: a too-powerful decoder
  drives the encoder to ignore its input entirely
- §9 VAE takeaways · 📚 why "blurry samples" is specifically an artefact of the Gaussian
  likelihood · 🎯 the VAE's modern role as latent diffusion's compression layer, not a standalone
  sampler

**PART C — Generative Adversarial Networks** (§10–§18)
- §10 GANs as implicit generative models — "learning by comparison" instead of writing down $p(x)$
- §11 🧪 **The density-ratio-via-classifier trick derived in full**: an ordinary BCE-trained
  classifier's output secretly encodes $p(x)/p_\theta(x)$, via Bayes' rule run backwards
- §12 From density ratio to Jensen-Shannon divergence — why JSD, not plain KL, is the natural
  comparison for two distributions that may not overlap at all
- §13 🧪 **The minimax objective, read as two nested optimizations** · the forger-and-detective
  analogy · the live toy-GAN demo watched collapsing onto one mode in real time
- §14 🧪 **Vanishing gradients derived from the loss curve's actual shape**, with the
  non-saturating fix
- §15 🧪 **Mode collapse derived as a structural risk**, not a training bug: nothing in the
  objective rewards mode *coverage*
- §16 Wasserstein GAN: the Earth-Mover distance and why it stays useful even with no distribution
  overlap
- §17 DCGAN: fixing the architecture (strided convolutions, BatchNorm) rather than the objective
- §18 GAN takeaways · 🎯 the honest, current (2026) scorecard on where GANs still win over
  diffusion

**Closing** — a full-lecture ASCII map (VAE's honest route vs. GAN's implicit route, both
answering Part 2's closing limitation) · **9 interview questions** with model answers (2 combining
concepts) · 7 depth probes · **3 whiteboard derivations** · a synthetic-catalog-imagery scenario
(VAE vs. GAN) extending [Part 1](unsupervised-learning-01.md)'s and [Part
2](unsupervised-learning-02.md)'s named Amazon use cases · 3 Leadership Principles ·
**28-term glossary** · 12 check-yourself questions · 10 ranked resources.

**Interactive specs:** 2 — §6 the reparameterization pipeline traced step by step, §13 watching a
toy GAN collapse onto one mode over training steps.

</details>

---

## What's in Part 4

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — capture notes (22 deduped slides from 52 raw frames, all 30 dropped frames
individually audited, transcript as primary source) · 10 capabilities · 4 prerequisites taught from
zero (Parts 1–3 recap, hierarchical VAE, convolution of distributions, ODEs) · full-lecture ASCII
map tracing the spiral from VAE → diffusion → flow matching

**PART A — Diffusion Models** (§1–§9)
- §1 The core idea: destroy data with noise, learn to undo it · the information-theoretic
  argument for gradual noise addition
- §2 Diffusion models as hierarchical VAEs: detailed comparison table (encoder, decoder,
  dimension change, training objective)
- §3 The forward diffusion process: step-by-step formula, variance preservation proved, closed
  form via ᾱ_t, 🧪 worked example on a scalar, noise schedules (linear, cosine, sigmoid)
- §4 The reverse process: why it's Gaussian, the neural network's input/output/role of t,
  ε-prediction vs x₀-prediction
- §5 Training objective: ELBO derived from VAE framework, denoising matching reformulation, the
  simple MSE loss boxed, pseudocode for the full training loop
- §6 Noise prediction = score prediction: score function defined, 🧪 2D Gaussian score computed,
  the equivalence proved
- §7 Ancestral sampling: the algorithm, role of random noise z, speed problem and acceleration
  methods (DDIM, distillation, consistency models)
- §8 Score functions in depth: scores at every noise level, interpolation from data to Gaussian,
  connection to Langevin dynamics
- §9 Putting Part A together: ASCII pipeline diagram, key properties summary

**PART B — Flow Matching** (§10–§15)
- §10 Motivation: why not choose a better path? Convention shift to t ∈ [0,1]
- §11 Probability paths and velocity fields: definitions, continuity equation, the transport PDE
- §12 Conditional flow matching (CFM) loss: the breakthrough, straight-line paths, 🧪 1D worked
  example, why straight lines are special
- §13 Inference: Euler's ODE solver, 🧪 worked example, algorithm, speed advantage table
- §14 The unifying connection: diffusion is a special case of flow, comparison table (VAE vs GAN
  vs diffusion vs flow)
- §15 Putting Part B together: ASCII pipeline diagram, full-module dependency chain from
  K-Means through flow matching

**Closing** — 10 interview questions with model answers (in `<details>` blocks) · 3 depth probes ·
3 whiteboard derivations (CFM loss, noise=score, Euler's method) · Amazon delivery time scenario ·
2 Leadership Principles · 14-term glossary · 13 check-yourself questions · 6 ranked resources

**Interactive specs:** 1 — §3 a forward-diffusion animation stepping through the noising process.

</details>

---

## Reading guide

The four parts total ~68,500 words and form a single escalating argument: from discrete clustering
(Part 1) through soft probabilistic clustering (Part 2) to neural generative models (Parts 3–4).
Each part assumes the previous ones — do not skip ahead.

**Prerequisites.** This module leans on
[`Dimensionality Reduction`](../Dimensionality%20Reduction/) throughout. Read it first if you haven't.

**Part 1 → Part 2** (~38,000 words together). Part 2 assumes Part 1's K-Means convergence proof
directly — its own EM convergence proof is the identical argument one level more general, and its
central equivalence (K-Means **is** GMM/EM with every responsibility collapsed to 0 or 1) only
lands if Part 1's proof is fresh.

**Parts 3 → 4** (~30,500 words together). Part 3 covers VAEs and GANs; Part 4 covers Diffusion
Models and Flow Matching. Part 4 assumes the ELBO, reparameterization trick, and GAN framework
from Part 3 are fresh — it does not re-derive them.

**First pass.** Read linearly: Part 1 → Part 2 → Part 3 → Part 4. Do not skip *Before we start*
in any file. Parts 1–2's derivation sections (§5–§6 in each) and Parts 3–4's core derivations
are where each lecture's central lesson lands.

**Second pass.** Work every 🧪 example on paper *before* reading the solution. The most important
across all four parts:
1. Part 1 §5 — K-Means convergence proof and hand-worked iteration
2. Part 2 §5 — the ELBO/KL decomposition, derived from scratch
3. Part 2 §6 — the E-step and M-step, each derived rather than quoted
4. Part 3 §B — VAE derivation (ELBO via Jensen, reparameterization)
5. Part 3 §C — density-ratio-via-classifier trick for GANs
6. Part 4 §5 — denoising matching from the ELBO
7. Part 4 §6 — noise prediction = score prediction
8. Part 4 §12 — conditional flow matching loss derivation

**Before an interview.** Each file's *Putting it together*, then all 12 whiteboard derivations (3
per part), then the depth-probe tables. The three highest-value derivations: **the ELBO/KL
decomposition** (Part 2 §5), **noise prediction = score prediction** (Part 4 §6), and **the CFM
loss** (Part 4 §12).

**The three questions this module is most likely to be examined on:**
1. *"Derive the EM algorithm"* (Part 2 §5–§6)
2. *"How are diffusion models related to VAEs and score functions?"* (Part 4 §2, §5–§6)
3. *"Compare VAE, GAN, diffusion, and flow matching on key properties"* (Part 4 §14 table)

**When a GMM fit stalls or looks degenerate.** Part 2's Check-yourself question 10 first, then
re-initialize from a K-Means fit (Part 2's interview question 8).

**When someone asks "why not just use a GAN?"** Part 3 §C for the instability problems, then
Part 4 §14's comparison table for the full picture.

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

## Key takeaway per lecture

| # | One-sentence takeaway |
|---|---|
| 01 | Every clustering algorithm answers the same question — "what is a cluster?" — differently, and the disagreement is not a matter of implementation detail but of geometric commitment: K-Means says a cluster is the set of points closest to one representative centroid (and is therefore provably, by an alternating-minimization argument, guaranteed to converge but never to a global optimum — the identical structural caveat resurfaces in Hierarchical Clustering's greedy, irreversible merges), Hierarchical Clustering says a cluster is whatever a bottom-up chain of nearest-neighbour merges produces (trading $O(n^3)$ complexity for the genuine advantage of recovering every value of $K$ from a single run), and DBSCAN says a cluster is a dense region separated from other dense regions by sparse space (the only one of the three with a built-in notion of "doesn't belong anywhere," at the structural cost of needing a full re-run to score any new point); and because none of the three comes with a built-in measure of quality, evaluating any of them without labels reduces to intrinsic metrics like silhouette score — which the lecture's own live demonstration proves, with real numbers on 901 real handwritten digits, measures **compactness**, not **correctness**, so a clustering can be 94% right against ground truth and still score a mediocre 0.19 on silhouette with no contradiction at all, purely because the true underlying classes are themselves spread out in feature space rather than because the algorithm made any mistake. |
| 02 | Every hard clustering algorithm from Part 1 forces genuine ambiguity into an artificial single label, and Gaussian Mixture Models fix this by turning cluster membership into a real probability distribution instead — but making that work requires solving a genuinely intractable problem (the unknown cluster assignment sits inside a logarithm of a sum, with no closed-form maximum), and the Expectation-Maximization algorithm solves it by introducing a placeholder distribution $q(z)$ and decomposing the true log-likelihood into an *exact* identity — a lower bound (the ELBO) plus a KL-divergence penalty that can never be negative — so that alternating two moves, each provably unable to hurt the true likelihood (the E-step sets $q$ to the true posterior, driving the penalty to exactly zero; the M-step then climbs the now-tight bound), guarantees the true likelihood never decreases across any iteration, converging only to a local optimum by the identical structural argument that proved the same limitation for K-Means — which turns out not to be a coincidence, since K-Means **is** GMM's EM algorithm in the exact limit where every soft "responsibility" has collapsed to a hard 0 or 1. |
| 03 | GMM's EM algorithm breaks the moment the latent variable $z$ becomes continuous instead of discrete, because the exact posterior Bayes'-rule computation the E-step relies on no longer has a closed form — and this lecture answers that break in two genuinely different ways: Variational Autoencoders keep every piece of the honest ELBO machinery and simply *approximate* the one piece that became intractable, training a neural-network encoder to stand in for the now-uncomputable true posterior (paying for that honesty with blurry samples, since squared-error reconstruction's optimal solution under genuine ambiguity is an average, and with the structural risk of posterior collapse if the decoder becomes powerful enough to memorize inputs regardless of the latent), while Generative Adversarial Networks abandon density estimation altogether and train an ordinary binary classifier to distinguish real from generated samples — a genuinely elegant piece of mathematics (Bayes' rule run backwards) showing that classifier's output secretly estimates a density ratio, and that an optimal such classifier is computing the Jensen-Shannon divergence — buying sharper samples at the cost of a two-player adversarial game with no single loss guaranteed to improve monotonically, and a structural vulnerability (mode collapse) that has no analogue anywhere else in this module. |
| 04 | Diffusion models answer "how do you learn to generate data" with a strategy that is at bottom a hierarchical VAE pushed to its extreme: replace one hard encoding step with hundreds of tiny, easy ones (fixed Gaussian noise addition, no learning required), then train a single shared network to undo each tiny step — and the ELBO for this process cleanly decomposes into a reconstruction term, a fixed prior-matching term, and a denoising-matching term that carries essentially all the learning and simplifies to a plain mean-squared-error loss on predicted noise, an objective *provably equivalent* to learning the score function (the gradient of the log-density) at every noise level, which is what makes ancestral sampling work and what unifies diffusion with score-based generative modeling; Flow Matching then generalises the entire idea one level further by asking why the "corruption path" from data to noise has to be Gaussian at all, replacing it with an arbitrary, learnable probability path and a velocity field solved by an ordinary ODE — recovering diffusion as one special case of a strictly larger family, while typically needing an order of magnitude fewer sampling steps because straight-line paths are smoother than the curved trajectories a fixed noise schedule produces. |
