---
title: "Unsupervised Learning — Part 1: Clustering & Density Estimation"
topic: unsupervised-learning
lecture: 10
source: "output/Lecture_10 - Module 4 Unsupervised Learning Part 1"
slides: 122
video: "https://www.youtube.com/watch?v=CWfgSOx3ngc"
runtime: "1:02:54"
---

# Unsupervised Learning — Part 1
### Clustering & Density Estimation: four families, one modeling question

---

## 📋 About this lecture and its capture

This is the module's opening lecture, and it is organised around a single sentence that appears on
slide 18 and then structures everything that follows:

> *"Every clustering algorithm encodes an assumption about what 'natural partition' means. Get it
> wrong and no hyperparameter tuning will save you."*

The deck delivers three clustering algorithms (K-Means, Hierarchical, DBSCAN) as three different
answers to that one question, then a chapter on evaluating any of them without labels, then a long,
genuinely excellent live-coding demonstration that runs K-Means on real handwritten-digit data and
uses it to teach the single most counter-intuitive fact in unsupervised evaluation: a clustering can
be 94% *correct* and still score badly on the metric everyone reaches for first.

| | Section | Runtime | Covers |
|---|---|---|---|
| **A** | Why unsupervised, and the six questions | 0:00 – 4:17 | The label-scarcity motivation · six things you can ask of unlabeled data · the "one question drives everything" framing · Amazon/frontier-of-AI context |
| **B** | Distance is a modeling choice | 4:17 – 9:16 | $L_p$ norms · Mahalanobis distance · non-vector data (Jaccard, DTW) |
| **C** | K-Means | 9:16 – 30:28 | The algorithm · objective · K-Means++ · the elbow method · evaluation metrics · where it fails |
| **D** | Hierarchical Clustering | 30:28 – 44:55 | Agglomerative algorithm · dendrograms · linkage methods · complexity |
| **E** | DBSCAN | 45:44 – 50:59 | Core/border/noise points · the algorithm · choosing ε and MinPts · strengths and limitations |
| **F** | Evaluation, and the live demo | 50:59 – end | Intrinsic vs extrinsic metrics · **a full Jupyter walkthrough clustering 901 real digit images**, ending in a genuinely subtle lesson about silhouette score |

Reconstructed from the raw capture in `output/`, the deck contains roughly **122 distinct slide
states** before the live demo begins, and the demo itself — a real, screen-recorded Jupyter
notebook — runs for the remainder of the hour.

> ✅ **Capture quality: excellent.** 175 raw frames over 63 minutes. Every content slide has a
> fully-built state, every formula and figure is legible, and the live demo was captured densely
> enough that **every printed number in the notebook is readable**. That is why §24–§30 below
> reproduce the demo's actual output — real purity percentages, a real silhouette score, and a real
> table of accuracy-vs-silhouette across seven PCA dimensionalities — rather than describing it.
>
> **No content gaps.** The one thing to know before you start: the deck ends on a teaser for
> **Gaussian Mixture Models** (*"Next: Gaussian Mixture Models — P(point ∈ cluster k) instead of hard
> assignment"*) and then pivots straight into the live demo without ever returning to GMM. GMM is not
> covered in this lecture; it is Part 2's opening subject, and these notes do not claim otherwise.
>
> **The instructor is not named** anywhere in the recording — the webcam tile carries no label —
> matching [Dimensionality Reduction Part 1](../Dimensionality%20Reduction/dimensionality-reduction-01.md),
> which had the same gap.

---

## How to read this document

The three algorithms are not three unrelated recipes — they are three different formalisations of
"what is a cluster", and the deck's own **Four Families of Clustering** slide (§4) is the map:

```
"What is a cluster?"
        │
        ├─ a POINT everything is close to  ──────────▶  K-MEANS (centroid-based)     §5–§15
        │
        ├─ a set built by repeatedly MERGING
        │   the two nearest things  ─────────────────▶  HIERARCHICAL (connectivity)  §16–§21
        │
        └─ a DENSE REGION separated from other
            dense regions by sparse space  ───────────▶  DBSCAN (density-based)      §22–§27
```

*(A fourth family — clusters as **Gaussian distributions**, distribution-based — is named on the same
slide and previewed throughout this lecture, but its algorithm, GMM, belongs to Part 2.)*

If you are revising under time pressure: **§4, §12, and §29 are the interview core.** "What are the
four families of clustering and what does each assume?", "Derive why K-Means always converges but
never guarantees a global optimum", and "Why can a clustering be 94% correct and still have a low
silhouette score?" are the three questions this lecture is most likely to be examined on.

**§29–§30 (the live demo's silhouette paradox) are worth reading even if you skip everything else.**
It is the best worked demonstration of a genuinely subtle evaluation trap in the entire course, and it
resolves cleanly into a single, memorable sentence.

Everything in a `🧪 Worked example` block should be reproducible by you on paper.

---

## What you'll understand after reading this

- You'll be able to state the six questions you can ask of unlabeled data, and place clustering,
  density estimation, and representation learning as three different answers to a shared underlying
  question.
- You'll be able to explain why the choice of **distance metric** is itself a modeling assumption, and
  derive why Mahalanobis distance is "Euclidean distance after whitening by the covariance."
- You'll be able to write K-Means' objective function, **derive why the algorithm's alternating steps
  guarantee convergence but not global optimality**, and reproduce a full iteration by hand.
- You'll be able to explain **why K-Means++'s $d^2$-weighted seeding** fixes the specific failure mode
  of random initialization, and state its approximation guarantee.
- You'll be able to read a dendrogram, name all four linkage methods, and explain **why Ward linkage
  is usually the right default**.
- You'll be able to state DBSCAN's core/border/noise definitions from memory, run the algorithm by
  hand on a small point set, and use a k-distance plot to choose ε.
- You'll be able to compute silhouette score, Calinski-Harabasz, ARI and NMI by hand on a tiny example,
  and know **which metric to reach for, for which algorithm**.
- You'll be able to explain, with real numbers from a real dataset, **why 94% clustering accuracy can
  coexist with a silhouette score of 0.19** — and why that is not a contradiction.
- You'll be able to justify **why PCA before clustering** genuinely changes evaluation metrics without
  changing the clusters themselves.

---

## Before we start: what you need to know

### Prerequisite 1 — What "unsupervised" actually removes

> **Supervised learning** — given $(x, y)$ pairs, learn $f$ such that $f(x) \to y$.
> **Unsupervised learning** — given only $x$, ask one of three questions [slide 6]:
>
> - *"Are there natural groups?"* → **Clustering** — geometric
> - *"What distribution generated this data?"* → **Density estimation** — learn $p(x)$, the
>   probability distribution over the data. Once you have $p(x)$, you can evaluate how likely any
>   point is, and sample entirely new ones.
> - *"What hidden factors explain the variation?"* → **Representation learning** — compact
>   coordinates, more informative for any downstream task

*In everyday words:* a supervised model is handed the answer key and learns to reproduce it. An
unsupervised model is handed a pile of unlabeled examples and must find *any* structure worth naming —
groups, a distribution, or a compressed coordinate system — with no external signal for what "correct"
means.

*Why it matters here:* the deck is explicit that this is not three separate fields. The caption on
slide 8 reads — *"[opening clause cut off by the frame edge in the raw capture] lives inside clustering
(GMM), generation (VAE), and anomaly detection. Not a separate category."* [slide 8] ⚠️ The sentence's
subject is genuinely illegible in the capture (a caption bar clipped at the frame's left edge in
frames 6–8); everything from "lives inside" onward is transcribed exactly. Read at face value, the
point still stands: **anomaly detection is not a separate technique** — it is something clustering,
GMM, and generative models all do as a side effect (an unusually distant cluster, a low-probability
region under a fitted density, a poorly-reconstructed input), not a fourth category requiring its own
algorithm family. Every one of the "six questions" in Prerequisite 2 below is a specific instance of
one of the three underlying questions this slide organizes.

### Prerequisite 2 — Why unlabeled data is the norm, not the exception

The deck opens with the economic argument, and it is worth having the numbers [slide 3]:

> - *"ImageNet — 14 million labeled images, took 3 years of human effort (≈72 person-years of
>   labeling)."*
> - *"YouTube — 500 hours of video uploaded every single minute."*
> - *"A single genomics run — ~3 billion base pairs, almost none annotated."*
> - *"Your recommendation system's click log — billions of events/day, labels on ~0.1%."*
>
> *"**Labels don't scale. Structure is free — if you know how to find it.**"*

*Concretely:* if labeling cost holds at ImageNet's rate (roughly 1 label per 5 seconds of skilled human
time), one minute of YouTube's upload rate — 500 hours of video — would take **on the order of a
decade of continuous human labeling effort to annotate**, and that's *one minute's* worth of uploads.
The gap between how much data exists and how much can ever be labeled is not closing; it is widening
as data volume grows faster than labeling capacity ever could. This is why unsupervised methods are
not a niche technique — they are the only tool that scales with the data itself.

### Prerequisite 3 — Norms and $L_p$ distances (recap, extended)

You need the family of $L_p$ distances a little more precisely than a prior refresher gave it, because
this lecture makes the *shape* of the distance's unit ball the whole point [slide 29]:

$$d_p(x, y) = \left(\sum_i |x_i - y_i|^p\right)^{1/p}$$

| $p$ | Name | Unit ball shape | The deck's description |
|---|---|---|---|
| $p=1$ | Manhattan | **Diamond** | *"all points with $\|x\|+\|y\| \le r$"* |
| $p=2$ | Euclidean | **Circle** | *"all points with $\sqrt{x^2+y^2} \le r$"* |
| $p=\infty$ | Chebyshev | **Square** | *"all points with $\max(\|x\|,\|y\|) \le r$"* |

*Why the shape matters, not just the formula:* "distance from center — is a diamond for $p{=}1$, a
circle for $p{=}2$, a square for $p{=}\infty$" [slide 30]. A clustering algorithm built on Euclidean
distance is implicitly declaring that its notion of "nearby" is *circular* — equally permissive in
every direction. Switch to Manhattan and "nearby" becomes diamond-shaped: it costs the same to move
diagonally as to move along one axis by the same $L_1$ amount, which is a genuinely different
geometric commitment. **The metric you pick is not a computational detail. It is the modeling
assumption "distance is a modeling choice" refers to.**

### Prerequisite 4 — Covariance matrices and whitening (recap)

You need the fact that a covariance matrix $\Sigma$ can be used to "undo" correlation between
features, because this lecture builds directly on it for Mahalanobis distance (§3).

> **Whitening** — a linear transform $\tilde x = \Sigma^{-1/2}x$ that turns any correlated,
> differently-scaled data cloud into one with *identity* covariance: unit variance on every axis, zero
> correlation between axes.
>
> *In everyday words:* if your data cloud is a tilted, stretched ellipse, whitening is the rotation +
> stretch that turns it back into a perfect circle.
>
> *Why it exists:* Euclidean distance treats every direction as equally significant, which is wrong
> whenever your features are correlated or on different scales. Whitening first makes that assumption
> *true*, so that ordinary Euclidean distance, applied *after* whitening, becomes meaningful again.

### Prerequisite 5 — Variance and the sum-of-squares objective

You already know variance from the DR and DNN modules; the fact you need fresh here is that K-Means'
entire objective is a **sum of squared distances**, and minimizing that sum is mathematically identical
to minimizing within-cluster variance:

$$J = \sum_{k=1}^{K}\sum_{x \in C_k} \|x - \mu_k\|^2$$

*Why squaring, not absolute value:* squared distance is differentiable everywhere (absolute value has
a kink at zero), and its minimizer over a set of points is *exactly* their mean — a fact you will
derive properly in §7. That single algebraic convenience is why K-Means uses the *mean* as its
centroid rather than, say, the median.

### Prerequisite 6 — Entropy and mutual information (recap)

[Dimensionality Reduction Part 1](../Dimensionality%20Reduction/dimensionality-reduction-01.md)
introduced entropy and mutual information as a *feature-selection* tool. This lecture reuses the exact
same quantity as a *clustering-evaluation* tool (§28's Normalized Mutual Information), which is a nice
small piece of evidence that these are genuinely general-purpose information-theoretic tools rather
than one-off tricks:

$$H(X) = -\sum_x p(x)\log p(x), \qquad I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$$

*Why it matters here:* NMI asks *"how much does knowing the cluster assignment tell you about the true
label, and vice versa?"* — which is precisely mutual information, normalized into $[0,1]$ so it can be
compared across datasets with different numbers of classes.

---

## The big picture

The lecture's own framing device is the strongest in the course so far, and it is worth quoting in
full because everything else hangs off it [slide 11]:

> *"One question drives everything today: **'What is the simplest explanation of this data?'**"*
>
> | Method | Answer |
> |---|---|
> | **K-Means** | K centroids + hard assignments — the simplest geometric compression |
> | **GMM** | K Gaussians with learnable distributions — soft, probabilistic partition |
> | **LDA** | K latent topics mixing to produce each document — generative text model |
> | **VAE** | Smooth latent space with explicit $p(x)$ — you can evaluate and sample |
> | **GAN** | A generative model trained to produce indistinguishable samples — no $p(x)$ needed |
>
> *"Five different answers — different answers to the same question. VAE via an explicit distribution.
> GAN via a trained generator."*

**That table is the syllabus for the entire Unsupervised Learning module**, not just this lecture:
K-Means and GMM are Part 1–2's clustering answers; LDA, VAE and GAN are later parts' generative-modeling
answers. Every one of them is "compress the data into the simplest structure that still explains it" —
they just disagree about what "simplest" and "explains" mean.

**Today's lecture answers the geometric half of that question.** *"Start with the most geometric
question: what are the groups?"* [slide 17] — and the deck's own **Four Families** slide is the map of
how three genuinely different geometric commitments produce three genuinely different algorithms:

```
                    "What is a cluster?" — every algorithm answers differently        §4
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
   CENTROID-BASED              CONNECTIVITY-BASED               DENSITY-BASED
   "A cluster = the set        "A cluster = built by            "A cluster = a dense
    of points closest to        repeatedly merging the           region separated from
    ONE representative           two nearest clusters"            other dense regions"
    point"                              │                               │
        │                               ▼                               ▼
        ▼                        HIERARCHICAL                       DBSCAN
    K-MEANS                      §16 dendrogram                 §22 core/border/noise
  §5 the algorithm               §17 linkage methods             §23 the algorithm
  §7 the objective J             §20 complexity O(n³)            §25 choosing ε, MinPts
  §9 K-Means++ init              §21 fails: still assumes         §26 strengths/limits
  §11 elbow method                  compact clusters
  §12 where it fails
  §13 evaluation metrics
        │                               │                               │
        └───────────────────────────────┴───────────────────────────────┘
                                        │
                          ALL THREE PRODUCE HARD ASSIGNMENTS           §27
                          "What's still missing?" → soft membership
                          → Gaussian Mixture Models (Part 2)
                                        │
        ══════════════════════════════════════════════════════════════════
                     EVALUATING ANY OF THEM WITHOUT LABELS               §28
        intrinsic (silhouette, CH, DB) · extrinsic (ARI, NMI) · which for which

        ══════════════════════════════════════════════════════════════════
              LIVE DEMO: 901 real digit images, K-Means, PCA           §29–§34
        93.8% purity but silhouette = 0.187 — the paradox, resolved
```

**And the deck earns the abstraction with three concrete Amazon use cases before it ever writes an
algorithm** [slide 23]: *customer segmentation* (group millions of shoppers by behavior for
personalization), *delivery station grouping* (cluster addresses by geography for last-mile
logistics), *product catalog taxonomy* (auto-organize millions of ASINs into a browse tree), and *fraud
ring detection* (identify clusters of coordinated fake reviews). Every algorithm in this lecture is
introduced as the tool that makes one of those four systems possible.

### Why this matters beyond today

The closing slides of the intro (§13–16 of the deck) make an argument worth carrying through the whole
module: *"The frontier of AI is fundamentally unsupervised"* [slide 13] —

> - *GPT-4/Llama/Claude* — self-supervised language modeling, no human labels, predict next token
> - *Stable Diffusion/DALL-E 3* — latent diffusion + VAE (the generative models Part 3 explains
>   exactly how)
> - *DINOv2 (Meta, 2023)* — self-supervised vision transformer, outperforms supervised on downstream
>   tasks
> - *AlphaFold 2* — unsupervised pretraining on protein sequences, representation enables structure
>   prediction
> - *JEPA (LeCun, 2022)* — predicts in representation space, proposed path to world models and AGI
>
> *"'Self-supervised learning is the cake, supervised is the icing, RL is the cherry.' — Yann LeCun"*

> 💡 **The through-line to hold onto across all four parts of this module.** Clustering, density
> estimation, and representation learning are not academic curiosities sitting apart from "real"
> modern AI — they are the literal mechanism by which the largest, most consequential models of 2026
> are trained. Every algorithm in this lecture is a small, fully-understandable instance of the same
> underlying idea that scales up to GPT-scale self-supervision.

---

# PART A — Distance, and the Four Families

*4:17 – 9:16*

---

## 1. Distance is a modeling choice

> *"Every clustering algorithm needs to define what 'close' means. The metric you pick encodes the
> shape of your neighborhoods."* [slide 29]

Prerequisite 3 already gave you the shapes; the deck's own annotation on the same figure makes the
practical stakes concrete:

> *"Amber points = inside the neighbourhood of $x$."*

Picture the same radius $r$ drawn three ways around a point $x$: as a diamond ($p{=}1$), a circle
($p{=}2$), a square ($p{=}\infty$). **A point sitting diagonally from $x$** is treated very
differently by each: it can be *outside* the Manhattan neighbourhood while comfortably *inside* the
Chebyshev one, for the identical raw coordinate offset. Which points count as "close" — and therefore
which points a clustering algorithm will pull into the same group — depends on a choice you make
*before* you ever look at the data's actual shape.

> 💡 **Why this is the first slide of the lecture and not a footnote.** Every clustering algorithm you
> are about to learn is, underneath its specific mechanics, running a nearest-neighbour computation.
> Change the distance and you change what "nearest" means, and therefore you change the clusters —
> without touching a single line of the clustering algorithm itself. Debugging a bad clustering by
> tuning $K$ or $\epsilon$ before checking the distance metric is a common and avoidable mistake.

---

## 2. Mahalanobis distance

> *"Euclidean distance is blind to correlation between features. Mahalanobis normalises by the
> covariance structure."* [slide 32]

The deck's three-panel figure is the cleanest visual derivation of this idea available, and it's worth
walking through panel by panel.

**Panel 1 — Euclidean, ignores correlation.**

$$d_E = \sqrt{(x-\mu)^\top(x-\mu)}$$

*"A and B are similar Euclidean distance from center — circles are axis-aligned."* Two points, $A$
($d{=}3.0$) and $B$ ($d{=}2.2$), sit at different raw distances from the cluster's mean $\mu$ — but
the data cloud itself is a tilted ellipse. Euclidean distance draws its neighbourhoods as circles
regardless, so it cannot tell that $A$ sits *along* the natural spread of the data while $B$ sits
*across* it.

**Panel 2 — Mahalanobis, ellipses follow the data.**

$$d_M = \sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)}$$

*"A is close (inside 1 sigma ellipse). B is far — not following the cluster shape."* The same two
points are re-scored: $A$ is now $1.4\sigma$ (close, because it sits along the ellipse's natural major
axis) and $B$ is $4.2\sigma$ (far, because it sits across the ellipse's short axis, where the true
spread of the data is small). **The distances flip in relative ranking** — the point that looked
*closer* under Euclidean distance is the one Mahalanobis correctly identifies as the outlier.

**Panel 3 — the geometric reason it works: whitened space.**

$$\text{whitened: } \tilde x = \Sigma^{-1/2}x$$

*"After whitening, cluster = sphere. Mahalanobis = Euclidean in whitened space."* Apply the whitening
transform from Prerequisite 4 and the tilted ellipse becomes a perfect circle — and in that
transformed space, ordinary Euclidean distance is *exactly* Mahalanobis distance in the original
space. Mahalanobis distance is not a new, exotic metric; **it is Euclidean distance, computed after
first undoing the correlation structure of the data.**

### 🧪 Derive the connection in one line

$$d_M(x,\mu)^2 = (x-\mu)^\top\Sigma^{-1}(x-\mu) = (x-\mu)^\top\Sigma^{-1/2}\Sigma^{-1/2}(x-\mu) = \left(\Sigma^{-1/2}(x-\mu)\right)^\top\left(\Sigma^{-1/2}(x-\mu)\right) = \|\tilde x - \tilde\mu\|_2^2$$

using $\Sigma^{-1} = \Sigma^{-1/2}\Sigma^{-1/2}$ (valid since $\Sigma$ is symmetric positive-definite).
The last expression is just the squared Euclidean norm of the *whitened* difference. $\blacksquare$

> 💡 **Why this slide matters far beyond today.** The deck's closing line on this topic is the single
> most load-bearing sentence in the K-Means-vs-GMM comparison that Part 2 will build on: *"GMM learns
> one Mahalanobis ellipse per cluster (one Sigma per component) — the only real difference from
> K-Means."* Every distinction between K-Means and GMM that you will meet next lecture is, at bottom,
> the distinction between this slide's Panel 1 and Panel 2.

---

## 3. When your data is not a vector

> *"Euclidean distance requires numeric vectors. For sets, documents, or time series — you need a
> different distance entirely."* [slide 35]

**Jaccard Similarity**, for sets:

$$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$

*"$\{cat, dog, pet\}$ vs $\{cat, dog, fish\}$: Jaccard $= 2/4 = 0.67$"* — two shared elements
(cat, dog) out of four total distinct elements across both sets (cat, dog, pet, fish). Contrast with
*"$\{cat, dog, pet\}$ vs $\{car, bus, road\}$: Jaccard $= 0/6 = 0.00$"* — completely disjoint sets.

**Dynamic Time Warping (DTW)**, for time series:

*"Euclidean: align point-by-point. Same-index matching: misses the peak-to-peak structure."* Two
sequences that trace the *same shape* but are shifted in time — one peaks a few steps later than the
other — look very dissimilar under naive index-by-index Euclidean comparison, because it compares
timestep 5 of one to timestep 5 of the other regardless of whether that's where the corresponding
feature actually sits.

*"DTW: align by optimal warping. Warped matching: peaks align correctly despite time shift."* DTW
instead finds the best non-linear alignment between the two sequences' *indices*, so that a peak in
series 1 gets matched to the corresponding peak in series 2 even if it occurs several steps later —
"optimal warped alignment between sequences."

> 📚 **Background the slide assumed — why this matters practically.** Comparing two customers'
> purchase histories, two users' clickstream sessions, or two sensor readings almost never gives you
> clean fixed-length numeric vectors. Jaccard handles the "did they buy overlapping sets of items"
> question; DTW handles the "do these two time series have the same *shape*, allowing for the fact one
> happened faster or slower than the other" question. Neither is exotic — they are the default choice
> the moment your data is not naturally a fixed-length vector of comparable numbers.

---

# PART B — K-Means

*9:16 – 30:28*

---

## 4. Four families of clustering

> *"Same data. Four algorithms. Four different answers. Each one encodes a different assumption about
> what a cluster IS."* [slide 27]

The deck runs all four algorithms on one identical dataset — two concentric rings plus a separate
blob, with a few scattered outlier points — and shows four genuinely different results:

| | **1. Centroid-based** | **2. Connectivity-based** | **3. Density-based** | **4. Distribution-based** |
|---|---|---|---|---|
| Algorithm | K-Means | Hierarchical | DBSCAN | GMM |
| What happens on the rings | *"splits rings into wedges — assumes spherical clusters"* | *"also struggles with rings — merges by proximity, not density"* | *"finds rings + blob correctly — gray x's = noise points"* | *"fits a Gaussian per cluster — soft color blend = uncertain membership"* |
| Definition | *"Each cluster = one center point. Hard assignments."* | *"Clusters formed by merging nearby pairs."* | *"Clusters = dense regions. Gray = noise."* | *"Each cluster is a Gaussian. Soft membership."* |

**Read the ring result across all four, because it is the whole lecture in one picture.** K-Means
*must* produce convex, roughly-circular regions — its centroid-based definition mathematically cannot
represent a ring, so it fractures the ring into wedges radiating from a badly-placed centroid.
Hierarchical clustering, built on pairwise proximity, has almost the identical failure — nearby points
across the ring's curve get merged before points on opposite sides of the ring do. DBSCAN is the only
one of the three hard-clustering methods that succeeds, because "dense region separated by sparse
space" is a definition a ring satisfies perfectly, regardless of its shape. GMM succeeds too, but for a
completely different reason you'll meet in Part 2 — it can bend an ellipse to partially trace curvature
that a rigid circular assumption cannot.

> 🎯 **This is the single best answer to "how do you choose a clustering algorithm?"** Don't start
> with a hyperparameter search. Start by asking what shape you expect your clusters to have —
> spherical and similarly-sized (K-Means), nested/hierarchical structure with no natural $K$
> (Hierarchical), arbitrary shape with noise (DBSCAN), or overlapping/probabilistic membership (GMM) —
> and let *that* choose the family.

---

## 5. K-Means: the algorithm

> *"Partition N data points into K groups, where each point belongs to the group whose center it is
> nearest to."* [slide 39]

$$\text{Objective — minimise:} \qquad J = \sum_{k=1}^{K}\sum_{x\in C_k}\|x-\mu_k\|^2, \qquad \mu_k = \text{mean of all points assigned to cluster }k$$

| Symbol | Read it as | What it means |
|---|---|---|
| $K$ | "K" | Number of clusters — a **hyperparameter you choose**, not learned from data |
| $C_k$ | "C sub k" | The set of points currently assigned to cluster $k$ |
| $\mu_k$ | "mu sub k" | The centroid of cluster $k$ — the mean of every point assigned to it |
| $J$ | "J" | The total within-cluster sum of squared distances — what K-Means minimizes |

> *"Three things you need to decide before running K-Means: **K** — the number of clusters
> (hyperparameter, not learned). **Distance metric** — almost always Euclidean, but not always right.
> **Initialization** — random or smart (K-Means++). Matters a lot."*

### 5.1 The five steps

1. Choose $K$ — the number of clusters you want.
2. Place $K$ centroids randomly in the data space.
3. Assign each point to the nearest centroid.
4. Move each centroid to the mean of its assigned points.
5. Repeat steps 3–4 until assignments stop changing.

> *"Guaranteed to converge — objective J decreases at every step. Not guaranteed to find the global
> minimum — only a local one."*

### 5.2 🧪 Derive the convergence guarantee

**This is the exact interview question §12 flags, and the proof has two halves — one for each
alternating step.**

**Step 3 (assignment) can only decrease $J$, or leave it unchanged.** Reassigning a point to its
nearest centroid can only make that point's squared distance term *smaller or equal* to what it was —
it never gets moved to a *farther* centroid. Summed over all points, $J$ cannot increase.

**Step 4 (update) can only decrease $J$, or leave it unchanged.** For a fixed set of assigned points,
the value of $\mu$ that minimizes $\sum_x\|x-\mu\|^2$ is *exactly* the mean of those points — this is a
standard calculus fact: differentiate with respect to $\mu$, set to zero:

$$\frac{\partial}{\partial\mu}\sum_x\|x-\mu\|^2 = \sum_x -2(x-\mu) = 0 \implies \mu = \frac{1}{|C_k|}\sum_{x\in C_k}x$$

So moving each centroid to its cluster's mean is *provably the best possible centroid* for that fixed
assignment — it cannot make $J$ worse.

**Putting the two together:** $J$ is a sequence of non-negative numbers that never increases at any
step. A non-increasing sequence bounded below (here, by 0) must **converge** — it cannot decrease
forever. That gives you the guarantee. $\blacksquare$

**But note precisely what was *not* proven.** Nothing in that argument says the sequence converges to
the *global* minimum of $J$ over all possible assignments and centroid placements — only that it
converges to *some* point where neither step can improve it further, i.e. a **local minimum**. Which
local minimum you land in depends entirely on where you started — which is exactly why initialization
(§9) matters so much.

### 🧪 Worked example — one full iteration by hand

Six points in 1-D: $\{1, 2, 3, 10, 11, 12\}$, $K=2$, initial centroids $\mu_1 = 2$, $\mu_2 = 11$
(a lucky initialization, for clarity).

**Step 3 — assign.** Distance from each point to $\mu_1=2$ vs $\mu_2=11$:

| Point | $\|x-\mu_1\|$ | $\|x-\mu_2\|$ | Assigned to |
|---|---|---|---|
| 1 | 1 | 10 | $C_1$ |
| 2 | 0 | 9 | $C_1$ |
| 3 | 1 | 8 | $C_1$ |
| 10 | 8 | 1 | $C_2$ |
| 11 | 9 | 0 | $C_2$ |
| 12 | 10 | 1 | $C_2$ |

**Step 4 — update.** $\mu_1 = \frac{1+2+3}{3} = 2$, $\mu_2 = \frac{10+11+12}{3} = 11$ — **unchanged**,
so the algorithm has already converged in one iteration.

**Compute $J$ at this fixed point:**

$$J = \underbrace{(1-2)^2+(2-2)^2+(3-2)^2}_{=1+0+1=2} + \underbrace{(10-11)^2+(11-11)^2+(12-11)^2}_{=1+0+1=2} = \mathbf{4}$$

Now try a *bad* initialization: $\mu_1 = 1$, $\mu_2 = 2$ (both centroids inside the left cluster).

**Assign:** points 1, 2 both nearer $\mu_1$ or split between them by ties; suppose $\{1\} \to C_1$,
$\{2,3,10,11,12\} \to C_2$. **Update:** $\mu_1 = 1$, $\mu_2 = \frac{2+3+10+11+12}{5} = 7.6$.
Re-assign against the new centroids — $\{1,2,3\}\to C_1$ (nearer to 1 than 7.6), $\{10,11,12\}\to C_2$
— and it self-corrects to the same good answer this time. **But this is not guaranteed in general** —
with $K \ge 3$ or less-symmetric data, a bad start can converge to a genuinely worse local minimum that
never self-corrects. That is precisely the failure mode K-Means++ (§9) targets.

```python
from sklearn.cluster import KMeans
import numpy as np
X = np.array([1,2,3,10,11,12]).reshape(-1,1)
km = KMeans(n_clusters=2, init=np.array([[2.],[11.]]), n_init=1).fit(X)
print(km.cluster_centers_.ravel(), km.inertia_)   # [2. 11.]  4.0
```

```interactive
type: animation
title: K-Means, one step at a time
concept: The alternating assign/update loop and why J can only ever decrease or hold steady
control: A "step" button that advances one assign-step or one update-step at a time, plus a
  reset with a choice of good vs. bad initial centroids
observe: Points changing colour (reassignment) and centroids sliding to the mean of their new
  colour group, with a running plot of J ticking down after every step
insight: J visibly never goes up, and — with the "bad" initialization — you can watch it either
  self-correct (as in the worked example above) or lock into a visibly wrong partition, making
  §5.2's "local, not global, minimum" caveat concrete rather than abstract
fallback: The worked-example table above (six 1-D points, two initializations) plus the D2
  derivation already give the complete assign/update trace and the final J for both cases.
```

---

## 6. K-Means in Python: 3 lines

> [slide 49]

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, init='k-means++', random_state=42)
km.fit(X)

labels = km.labels_          # cluster assignment per point
centers = km.cluster_centers_  # K centroids (shape: K x d)
inertia = km.inertia_        # total within-cluster distortion
```

> *"That's it. sklearn handles initialization, convergence, and the objective function."*
>
> *"Key parameters: `n_clusters` — the K you choose · `init: 'k-means++'` is default ·
> `n_init`: number of restarts (default 10)."*

> 💡 **Read `n_init=10` carefully — it is silently doing the work of the "not guaranteed to find the
> global minimum" caveat from §5.** sklearn's default is to run the *entire* algorithm ten times from
> ten different K-Means++ initializations and **keep only the run with the lowest final $J$**. This is
> the standard, practical mitigation for local-minimum risk: you cannot guarantee a global optimum
> analytically, so you approximate it by brute-force restarts and taking the best. Setting `n_init=1`
> for speed silently reintroduces the local-minimum risk this default was designed to hedge against.

---

## 7. Choosing K — the Elbow Method

> *"Run K-Means for K = 2, 3, ..., N. Plot distortion against K."* [slide 61]
>
> *"Distortion always decreases as K grows."*
>
> *"Pick the K where the curve bends — diminishing returns beyond that point."*
>
> *"Limitation: elbow is sometimes ambiguous. Combine with silhouette for confirmation."*

### 🧪 Why distortion *always* decreases as K grows — a one-line proof

At $K=N$ (one cluster per point), every point is its own centroid, so every term
$\|x-\mu_k\|^2 = 0$ and $J=0$ exactly. More generally: **increasing $K$ by one can never force $J$
up**, because the old $(K{-}1)$-cluster solution is still a *valid* (if suboptimal) $K$-cluster
solution — just split one cluster into two identical copies and $J$ is unchanged; K-Means will then
only ever find something *at least as good*. This is exactly why "pick the $K$ that minimizes $J$" is
a broken selection rule on its own — the minimum is always at $K=N$, which is a useless clustering (one
point per cluster). **The elbow method exists specifically to work around this monotonicity**, by
looking for the point of diminishing *returns* rather than the point of minimum $J$.

---

## 8. When K-Means fails

> *"K-Means assumes clusters are: **Spherical** — equidistant from centroid in all directions.
> **Similar in size. Similar in density.**"* [slide 62]
>
> *"When violated: **Non-spherical shapes** — rings, crescents, elongated blobs. **Varying density** —
> dense and sparse together. **Outliers** — drag centroids away. **Initialization sensitivity** —
> different seeds, different results."*

The deck's three-panel figure demonstrates each failure directly: K-Means at $K{=}3$ works well on
*spherical, equal-size clusters* — clean separation. K-Means at $K{=}2$ on interleaved crescent
("moon") shapes *fails — non-spherical (moon-shaped)*, splitting each crescent's ends into the wrong
group. K-Means at $K{=}3$ on *clusters of very different density* fails because the objective $J$
implicitly weights every cluster by its spatial spread — a sparse cluster's points, being farther from
any centroid on average, exert more "pull," distorting where the boundary between a dense and a sparse
cluster actually falls.

> 💡 **The unifying reason all four failure modes are really one failure mode.** K-Means' objective is
> a sum of *squared Euclidean* distances to a *single* centroid per cluster. That is mathematically
> equivalent to assuming every cluster is an isotropic (spherical) Gaussian blob of roughly the same
> size and density — you will prove this equivalence formally in Part 2 when GMM is introduced with a
> shared, spherical covariance. Every one of the four bullet points above is just a different way that
> real data violates "isotropic Gaussian blob."

---

## 9. K-Means++ — smarter initialization

> *"Problem: random init can place two centroids in the same true cluster."* [slide 64]
>
> *"K-Means++ fix — spread seeds probabilistically: 1. Pick first centroid uniformly at random.
> 2. Compute $d$ = distance to nearest chosen centroid. 3. Pick next centroid with probability
> $\propto d^2$. 4. Repeat until K centroids chosen."*

$$P(\text{pick } x_i) \propto d(x_i, C)^2$$

*"A point twice as far is four times as likely to be picked."* *"Guarantee: $O(\log K)$ competitive
with optimal distortion."*

The deck's own side-by-side figure makes the mechanism vivid: *"Random init — two seeds in same
cluster"* shows two of three random starting centroids landing inside the same true blob, leaving a
third true blob unrepresented — the algorithm then converges to a bad local minimum with distortion
$J{=}153$, labeled *"bad local minimum."* *"K-Means++ — seeds spread by $d^2$ weighting"* shows the
probabilistic scheme deliberately favouring a point far from the already-chosen seeds — *"higher
$d^2 \Rightarrow$ more likely to be next seed"* — landing one centroid cleanly in each of the three true
blobs, and converging to the *same* final distortion $J{=}153$ this time, but now labeled *"correct
clustering."*

> ⚠️ **The demo's own numbers are worth sitting with.** Both runs converge to $J=153$ — because both
> are *local* minima that K-Means' alternating-minimization proof (§5.2) guarantees it will find *some*
> local minimum, and by coincidence both scenarios' final $J$ values match in this specific figure.
> What differs is *which* local minimum, and the "bad local minimum" one is explicitly mislabeling
> which points belong together, even though its numeric objective value happens to tie. **A low $J$
> alone does not certify a good clustering — you still need to inspect the assignment,** which is a
> theme §29's silhouette-score paradox will return to with a real dataset.

> 💡 **Why the probability is $d^2$ and not $d$.** Squaring amplifies the preference for far-away
> points more aggressively than a linear weighting would — "a point twice as far is four times as
> likely." This is a deliberate choice: it is aggressive enough to reliably escape "both seeds in the
> same cluster" while still leaving *some* chance of picking a nearby point (rather than deterministic
> farthest-point selection, which is brittle to a single extreme outlier dominating every future pick).

```interactive
type: simulator
title: K-Means++ seed selection
concept: Why d²-weighted probabilistic seeding avoids "two seeds, one cluster"
control: A slider on the exponent (d^1 vs d^2 vs "always pick the farthest point") and a
  re-roll button to re-sample which point gets picked next, on the deck's own three-blob layout
observe: Each new seed's location, and a running tally of how often two seeds land in the same
  true blob across many re-rolls, for each exponent setting
insight: d¹ still occasionally seeds two points in the same blob; d² does so far less often; pure
  "always farthest" seeding is deterministic and can fixate on one outlier every single run —
  d² is the aggressive-but-not-brittle middle ground the deck chose
fallback: The worked comparison above (random init landing two seeds in one blob vs. K-Means++
  landing one seed per blob) plus the "a point twice as far is four times as likely" rule already
  state the mechanism completely in words.
```

---

## 10. Key takeaways: K-Means

> [slide 68]
>
> 1. K-Means minimizes within-cluster distortion by alternating assignment and centroid update.
> 2. It converges fast but only to a local minimum — always run multiple restarts.
> 3. Use K-Means++ initialization (sklearn default) to avoid poor starting configurations.
> 4. The elbow method + silhouette score help choose K, but neither is definitive — visualize.
> 5. K-Means fails on non-spherical clusters, varying density, and outliers. Know when to reach for
>    something else.
>
> *"Rule of thumb: start with K-Means. If it works, ship it. If it fails, the failure pattern tells
> you which algorithm to try next."*

> 🎯 **That rule of thumb is worth memorising verbatim for an interview.** It reframes "which
> clustering algorithm should I use?" from an upfront design decision into a diagnostic loop: start
> cheap and simple, and let the *specific way K-Means fails* — non-spherical shapes point you to
> DBSCAN, no natural $K$ points you to Hierarchical, overlapping/uncertain membership points you to
> GMM — tell you what to reach for next.

---

# PART C — Hierarchical Clustering

*30:28 – 44:55*

---

## 11. Where K-Means falls short — the transition

> *"K-Means gave us a fast, intuitive algorithm. But it has two hard assumptions: **① You must choose
> K upfront** — what if you don't know the number of groups? **② Clusters must be spherical
> (convex)** — what if your data has rings, spirals, or chains?"* [slide 72]
>
> *"This section introduces two algorithms that relax these assumptions, one at a time.
> **Hierarchical Clustering** → Fixes: don't need K upfront. Still assumes compact clusters.
> **DBSCAN** → Fixes: finds arbitrary shapes. Bonus: identifies noise points.
> Neither needs K. Both expand the toolkit."*

This is a genuinely well-designed pedagogical structure worth naming explicitly: the lecture doesn't
present three independent algorithms — it presents K-Means, then **one relaxation** of K-Means's
assumptions (Hierarchical: drop the "choose K upfront" requirement, keep "compact"), then **a second,
independent relaxation** (DBSCAN: drop "compact," keep nothing about K at all). Each algorithm exists
because of a specific, named weakness in the one before it.

---

## 12. Agglomerative Clustering

> *"Promise: build a hierarchy from bottom up, no K needed."* [slide 76]
>
> *"The Algorithm (bottom-up): 1. Start: every point is its own cluster. 2. Find the two closest
> clusters. 3. Merge them into one. 4. Repeat until one cluster remains."*
>
> *"Complexity: $O(n^3)$ time, $O(n^2)$ space"*

The deck's three-panel figure traces the mechanism directly: **Step 1** shows five isolated points, each
its own cluster. **Step 2** shows the two nearest points (bottom-left pair) connected by a merge line —
the first, smallest merge. **Step 3** shows the fully-built hierarchy, with every point now connected
through a web of merge lines up to a single root.

> 💡 **The single sentence that distinguishes this from K-Means entirely.** K-Means asks "given $K$,
> what's the best partition?" — a question that requires $K$ as an input before you can even start.
> Agglomerative clustering instead asks "what is the complete history of which things are most similar
> to which other things?" — a question that needs *no* $K$ at all, because it doesn't stop at any
> particular number of clusters. **$K$ becomes something you extract from the finished hierarchy, not
> something you supply to the algorithm.**

### 🧪 Why $O(n^3)$ time — derive the complexity

At each of the $n-1$ merge steps, the algorithm must find the closest pair of clusters among however
many remain. A naive search over all pairs at one step costs $O(n^2)$ (comparing every pair of the
currently-remaining clusters). Repeating an $O(n^2)$ search across $O(n)$ merge steps gives
$O(n) \times O(n^2) = O(n^3)$ total. *(Optimized implementations using a priority queue can bring this
down to $O(n^2\log n)$, but the naive bound is what the deck states and is the one worth being able to
derive.)*

> ⚠️ **This complexity is the practical reason Hierarchical Clustering doesn't scale**, and the deck's
> own Key Takeaways slide (§14) states the concrete limit: *"impractical for datasets beyond ~10K
> points."* At $n = 100{,}000$, $n^3 = 10^{15}$ — simply not computable in any reasonable time on any
> single machine. This is the single most important practical fact to carry out of this section.

---

## 13. Dendrograms: reading the tree

> *"A dendrogram records the full merge history."* [slide 82]
>
> *"How to read it: **Y-axis = distance at merge**. **Horizontal line = a merge event**. **Cut at
> height h → get K clusters**."*
>
> *"Key insight: **One run → all values of K**. Choose K after seeing the structure."*

The deck's own dendrogram figure, built with Ward linkage on nine points, shows exactly this in action:
points $\{4, 0, 1\}$ merge at low heights on the left (an orange sub-tree), points $\{7, 6, 8\}$ merge
at low heights in the middle (green), and points $\{5, 2, 3\}$ merge at low-to-moderate heights on the
right (red). All three sub-trees then merge together at progressively greater heights, up to the root
at height $\approx 16$. A horizontal dashed line drawn at height $\approx 8$ — *"cut → K=3"* — crosses
exactly three of the tree's vertical branches, giving a 3-cluster partition read directly off the
figure with no re-running of the algorithm.

> 💡 **This is the practical payoff of the "no K needed" promise, made concrete.** Run agglomerative
> clustering **once**. The resulting dendrogram lets you extract the 2-cluster solution, the 3-cluster
> solution, the 7-cluster solution — *every* possible value of $K$ — by simply choosing where to draw a
> horizontal line, with zero additional computation. K-Means, by contrast, requires a completely fresh
> run for every candidate value of $K$ you want to compare (as the elbow method in §7 does explicitly).
> **This single-run-many-K property is the dendrogram's real value, not just its visual appeal.**

```interactive
type: slider
title: Cut the dendrogram
concept: One agglomerative run yields every value of K, for free
control: A vertical slider on cut-height h, dragged over the deck's own nine-point Ward
  dendrogram
observe: The horizontal cut line moving up and down, and the resulting cluster count / colouring
  of the nine points changing as it crosses different branches
insight: There is no re-computation at any cut height — every K from 1 (cut above the root) to 9
  (cut below every leaf) is already fully determined by the one tree, unlike K-Means' "re-run for
  every K" workflow
fallback: The worked description above (heights ≈8 → K=3, root ≈16) already states which branches
  a cut at each height crosses; walk the tree top-down by hand to get any other K.
```

---

## 14. Linkage methods

> *"How do we measure distance between two clusters?"* [slide 85]
>
> - *"**Single**: min pair-wise distance — finds chains, sensitive to noise"*
> - *"**Complete**: max pair-wise distance — compact clusters, breaks elongated ones"*
> - *"**Average**: mean pair-wise distance — a compromise"*
> - *"**Ward**: minimize total within-cluster variance — usually the best default"*

The deck runs all four linkages on the identical two-crescent dataset side by side, and the visible
contrast is the entire point of the slide: **Single linkage** correctly traces the two long, curved
crescents — because "distance between clusters = distance between their single nearest pair of points"
lets a cluster snake along a curved shape one point at a time, exactly like a chain. **Complete
linkage** instead breaks each crescent into disconnected fragments, because "distance = the *farthest*
pair" heavily penalizes any cluster that isn't compact and round — an elongated crescent has a large
worst-case pairwise distance within itself, so complete linkage refuses to keep it as one cluster.
**Average** and **Ward** linkage sit between these extremes, each also fragmenting the crescents to
varying degrees but less severely than complete linkage.

### 🧪 Why Ward is usually the right default — derive the connection to K-Means

**Ward's criterion is: at each merge step, choose the pair of clusters whose merger causes the
smallest possible *increase* in total within-cluster variance.** That is not a coincidental phrase —
it is *literally K-Means' objective function $J$ from §5*, applied incrementally at every merge rather
than optimized globally over a fixed $K$. This is why Ward linkage tends to produce clusters that
"look like" what K-Means would produce (compact, similarly-sized, roughly spherical) — because it is
optimizing the *same quantity*, just via bottom-up merging instead of alternating minimization.

> 💡 **The practical decision rule this gives you.** If you have no strong prior belief about your
> cluster shapes, start with Ward — it inherits K-Means' well-behaved, general-purpose bias toward
> compact clusters, which is right more often than it's wrong. **Reach for single linkage specifically
> when you expect chain-like or elongated true structure** (the crescents above, or genuinely
> hierarchical/phylogenetic data where "nearest-neighbour chaining" is the correct model) — and know
> that doing so trades away robustness to noise, since a single noisy point can bridge two otherwise
> distinct clusters into one long chain.

---

## 15. Hierarchical clustering in Python

> [slide 88]

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Fit clusters
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X)

# Plot dendrogram (scipy)
Z = linkage(X, method='ward')
dendrogram(Z, truncate_mode='level', p=5)
```

> *"Two libraries, complementary roles: **sklearn**: fit clusters, get labels. Simple API, no
> dendrogram. **scipy**: compute linkage matrix Z, plot dendrograms. More control."*
>
> *"Tip: set `n_clusters=None` + `distance_threshold` to cut by height instead of K."*

> ⚠️ **A subtlety worth flagging.** sklearn's `AgglomerativeClustering` requires you to *pre-commit* to
> either `n_clusters` or `distance_threshold` — it doesn't hand you the full dendrogram to explore
> interactively the way scipy's `linkage` + `dendrogram` does. In practice, the standard workflow is to
> build the dendrogram with **scipy** first to *look at* the structure and pick a sensible cut height
> visually, and only then fit the final model with **sklearn** using that chosen `n_clusters` (or the
> corresponding `distance_threshold`) for a clean, simple `.labels_` output.

---

## 16. Key takeaways: hierarchical clustering

> [slide 96]
>
> 1. No K needed upfront — the dendrogram shows all possible clusterings in one run.
> 2. Ward linkage is the best default. Use single linkage only if you specifically want chain-shaped
>    clusters.
> 3. $O(n^3)$ complexity makes it impractical for datasets beyond ~10K points. Use mini-batch or
>    approximate variants for scale.
> 4. Merges are greedy and irreversible — a bad early merge can never be undone.
> 5. Still assumes compact, roughly convex clusters. Arbitrary shapes need a different approach.
>
> *"Next: DBSCAN removes both the K requirement AND the shape assumption."*

### 🧪 Why merges being "greedy and irreversible" matters

Once two clusters are merged at some step, **no later step can ever split them back apart** — the
algorithm only ever merges, never un-merges. If an early, low-height merge happens to combine two
points that "should" belong to different final clusters — perhaps because of an unlucky noise point
bridging them — every subsequent decision inherits that mistake, and there is no mechanism in the
algorithm itself to correct it later. This is the direct structural analogue of K-Means' "guaranteed
convergence but not to a global optimum" caveat (§5.2): both algorithms make locally-optimal decisions
at each step with no guarantee those decisions compose into a globally optimal final answer.

---

# PART D — DBSCAN

*45:44 – 50:59*

---

## 17. What if clusters aren't blobs?

> *"Promise: find clusters of any shape, no K, no shape assumption, identifies outliers for free."*
> [slide 99]
>
> *"Core idea: a cluster is a dense region separated from other dense regions by sparse regions."*

The deck's own comparison figure states the motivation as plainly as possible: on the two-crescent
dataset, *"K-Means (K=2): fails on non-convex shapes"* — the familiar wedge-splitting failure from §8
— against *"DBSCAN ($\epsilon{=}0.2$, MinPts=5): finds arbitrary shapes"*, which traces both crescents
correctly with no shape assumption at all.

**This is a genuinely different definition of "cluster" than either K-Means or hierarchical
clustering use.** Both of those methods define a cluster in terms of *distance to a representative
point or to other cluster members*. DBSCAN instead defines a cluster purely in terms of **local point
density**, with no reference to shape, size, or any centroid at all — which is exactly what lets it
succeed where the other two fail.

---

## 18. Three types of points

> *"Parameters: $\epsilon$ (radius), MinPts (density threshold)"* [slide 101]
>
> - **Core point** — *"$\ge$ MinPts neighbors within $\epsilon$ radius. Forms the backbone of a
>   cluster."*
> - **Border point** — *"< MinPts neighbors, but within $\epsilon$ of a core point. On the edge of a
>   cluster."*
> - **Noise point** — *"< MinPts neighbors, not near any core point. Outlier, belongs to no cluster."*

*In everyday words:* a core point is somewhere genuinely crowded — it has enough neighbours nearby to
count as "in the thick of things" on its own merits. A border point isn't crowded itself, but it sits
right next to a crowded core point, so it gets pulled into that cluster on the strength of its
neighbour rather than its own local density. A noise point is neither — it's isolated, with too few
neighbours and no crowded core point nearby to attach to.

> 💡 **This three-way classification is the entire mechanism that gives DBSCAN "outlier detection for
> free."** No other algorithm in this lecture has a built-in notion of "this point doesn't belong to
> any cluster" — K-Means forces *every* point into its nearest centroid regardless of how far away
> that centroid is, and hierarchical clustering forces every point into the tree. DBSCAN's noise
> category is a structural feature of the definition, not a post-hoc addition.

---

## 19. The DBSCAN algorithm

> *"1. Pick an unvisited point p. 2. Find all points within $\epsilon$ of p. 3. If |neighbors| $\ge$
> MinPts: → p is core; start new cluster → Add all neighbors to cluster → Recursively expand from each
> core neighbor. 4. Else: label p as noise (for now). 5. Repeat until all points visited."*
> [slide 102]
>
> *"Complexity: $O(n\log n)$ with spatial index."*

The deck's three-panel figure walks the algorithm's execution: **1. All points unvisited** — a raw
scatter of points, no colouring. **2. Expand from seed** — one point is chosen, its $\epsilon$-neighbourhood
is examined, and a cascade of newly-labeled core and border points (in colour) spreads outward from it
while a few isolated points remain marked in red (candidate noise). **3. Final clusters** — the
recursive expansion has completed, producing two cleanly-separated curved clusters plus a small
residue of red noise points that never got pulled into either.

> ⚠️ **"label p as noise (for now)" is a precise phrase, not a loose one.** A point initially labeled
> noise because it fails the core-point density test *can still become a border point later*, if a
> different core point's expansion later reaches it within $\epsilon$. The algorithm's true noise
> points are only the ones that, after *every* core point in the dataset has been fully expanded, were
> never reached by any of them. This is why the deck writes "(for now)" — it's flagging that this
> label is provisional at the moment it's assigned, not final.

### 🧪 Worked example — running DBSCAN by hand

Six 1-D points: $\{1, 1.5, 2, 8, 8.5, 20\}$, $\epsilon = 1$, MinPts $= 2$ (counting the point itself, so
"MinPts=2" means at least 1 *other* point within $\epsilon$).

**Point 1:** neighbours within $\epsilon{=}1$: $\{1.5\}$ (distance 0.5). Count $=1 \ge$ MinPts$-1$
→ **core**. Start Cluster A.

**Point 1.5:** neighbours: $\{1, 2\}$ (distances 0.5, 0.5). Count $=2$ → **core**. Already in Cluster A
via point 1's expansion; recursively expand — point 2 gets added.

**Point 2:** neighbours: $\{1.5\}$ (distance 0.5). Count $=1$ → **core**. Confirms Cluster A
= $\{1, 1.5, 2\}$, fully expanded.

**Point 8:** neighbours: $\{8.5\}$ (distance 0.5). Count $=1$ → **core**. Start Cluster B.

**Point 8.5:** neighbours: $\{8\}$ (distance 0.5). Count $=1$ → **core**. Cluster B = $\{8, 8.5\}$.

**Point 20:** neighbours within $\epsilon{=}1$: none. Count $=0$ → **noise**.

**Final result:** Cluster A $=\{1, 1.5, 2\}$, Cluster B $=\{8, 8.5\}$, Noise $=\{20\}$ — two clusters
correctly separated by the large gap, with the far-off outlier correctly excluded from both, and no
$K$ was ever specified.

```python
from sklearn.cluster import DBSCAN
import numpy as np
X = np.array([1, 1.5, 2, 8, 8.5, 20]).reshape(-1, 1)
db = DBSCAN(eps=1, min_samples=2).fit(X)
print(db.labels_)   # [0 0 0 1 1 -1]   <- -1 marks noise
```

```interactive
type: animation
title: DBSCAN's core-point expansion, one point at a time
concept: Clusters grow by recursive ε-neighbourhood expansion from core points, and "noise" is
  provisional until every core point has been tried
control: A "next point" button that visits the deck's own two-crescent-plus-outliers dataset one
  unvisited point at a time, plus an ε slider
observe: Each visited point's ε-neighbourhood circle appearing, the point being labelled
  core/border/noise(-for-now), and — when a point is core — the cascade of newly-reached
  neighbours lighting up in the same cluster colour
insight: A point flagged "noise" early in the run can flip to "border" later once a different
  core point's expansion reaches it — the provisional-vs-final distinction the deck's own "(for
  now)" phrasing is making, seen happening rather than just stated
fallback: The three-panel figure description and the six-point worked example above already trace
  every point's core/border/noise decision and the order clusters form in, by hand.
```

---

## 20. DBSCAN in Python

> [slide 104]

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)  # important!

db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f'{n_clusters} clusters, {n_noise} noise points')
```

> *"Critical details: **Always scale first** — DBSCAN uses absolute distance, so units matter.
> **Noise points get label = -1.** Filter them for downstream analysis. **No `.predict()` method** —
> DBSCAN is transductive (fit-time only). Use `k_neighbors_graph`/distance plot to pick eps."*

> ⚠️ **"Always scale first" is not a generic best-practice reminder here — it is structurally
> necessary for DBSCAN specifically, in a way it isn't for K-Means.** K-Means at least implicitly
> re-normalizes distances through the relative comparison "which centroid is nearest" — a global
> rescaling of every feature changes the *absolute* distortion $J$ but often leaves the *assignments*
> largely unchanged if the scaling is uniform. DBSCAN's $\epsilon$ is an **absolute, fixed radius**: a
> feature measured in different units (say, income in dollars alongside age in years) will have the
> income differences numerically swamp the age differences, and a single global $\epsilon$ can never
> be simultaneously correct for both scales. Un-scaled DBSCAN is not merely suboptimal — it is often
> simply wrong.
>
> ⚠️ **"No `.predict()` method" — this is DBSCAN's single biggest practical limitation relative to
> K-Means, and it deserves to be stated plainly.** K-Means learns *centroids*, which are a compact,
> reusable summary you can compare any new point against forever after fitting. DBSCAN's clusters are
> defined entirely in terms of the density structure of the *specific dataset it was fit on* — there is
> no summary object to compare a brand-new point to. If a new point arrives after fitting, you cannot
> ask DBSCAN "which cluster does this belong to" without literally re-running the whole algorithm on
> the combined dataset. **This is what "transductive" means**, and it rules DBSCAN out for any
> streaming or online-scoring use case where new points must be classified in real time.

---

## 21. Choosing ε and MinPts

> *"MinPts (rule of thumb): MinPts $\ge$ dim + 1 (often 2·d). 2D data: MinPts = 4 or 5. Larger →
> smoother clusters, more noise."* [slide 111]
>
> *"$\epsilon$ (k-distance plot): 1. Compute dist to k-th neighbor. 2. Sort ascending and plot.
> 3. Find the elbow (sharp bend). 4. $\epsilon$ = distance at elbow."*

The deck's own k-distance plot figure shows exactly this procedure: the x-axis is *"Points (sorted by
distance to 4th neighbor)"*, the y-axis is *"4-distance."* The curve stays low and nearly flat for most
of its length — *"neighbors are close → points inside clusters"* — then rises sharply near the
right-hand end — *"neighbors are far → points between clusters or noise/outliers."* The elbow, marked
at $\epsilon = 0.41$, is read directly off where that sharp bend occurs. The paired result panel,
*"DBSCAN result ($\epsilon{=}0.41$, MinPts=4)"*, shows three cleanly-recovered clusters plus a scatter
of correctly-identified noise points marked with red x's.

> 💡 **Notice the structural parallel to the elbow method from §7 — same shape of reasoning, different
> quantity.** Both procedures work by sorting some quantity, plotting it, and looking for the point
> where the curve's behaviour changes qualitatively (diminishing returns for $K$; a sharp jump for
> $\epsilon$). This "sort, plot, find the bend" pattern recurs constantly in unsupervised
> hyperparameter selection precisely *because* there is no labeled validation signal to optimize
> against directly — you are always looking for a structural signature in the data itself.

---

## 22. DBSCAN: strengths & limitations

> [slide 113]

| **Strengths** | **Limitations** |
|---|---|
| ✓ No K required | ✗ Struggles with varying densities |
| ✓ Finds arbitrary-shape clusters | ✗ $\epsilon$ is global, one scale for all |
| ✓ Robust to outliers (labels noise) | ✗ High dimensions: distances converge |
| ✓ Deterministic for core points | ✗ Border points are non-deterministic |
| ✓ $O(n\log n)$ with spatial indexing | ✗ Hard boundaries with no uncertainty |

> ⚠️ **Two of these five limitations connect directly to earlier material in this course, and being
> able to make that connection is a strong interview signal.**
>
> *"High dimensions: distances converge"* is exactly the **distance concentration** result from
> [Dimensionality Reduction Part 1 §2](../Dimensionality%20Reduction/dimensionality-reduction-01.md):
> as dimensionality grows, the relative contrast between the nearest and farthest point shrinks toward
> zero, so a fixed $\epsilon$ radius stops meaningfully distinguishing "near" from "far" at all. DBSCAN
> in high dimensions typically needs dimensionality reduction first, exactly as the live demo in
> §29–§30 will demonstrate for K-Means.
>
> *"$\epsilon$ is global, one scale for all"* is the density-analogue of the same underlying limitation
> named for K-Means in §8: a single, fixed parameter cannot simultaneously be correct for regions of
> very different density, just as a single centroid-based objective cannot be simultaneously correct
> for clusters of very different size. **HDBSCAN** (mentioned in §23's Key Takeaways as the fix) allows
> $\epsilon$ to vary locally, precisely to address this.

---

## 23. Key takeaways: DBSCAN

> [slide 115]
>
> 1. DBSCAN defines clusters as dense regions — no K, no shape assumption, noise labeled for free.
> 2. Two parameters only: eps (neighborhood radius) and MinPts (density threshold). The k-distance
>    plot guides eps choice.
> 3. Struggles with varying density — one eps cannot serve both dense and sparse regions. Consider
>    HDBSCAN for this.
> 4. Always standardize features first. DBSCAN is distance-based and scale-sensitive.
> 5. No `predict()` for new points — it is a transductive algorithm. Use the learned labels to train a
>    classifier if needed.
>
> *"DBSCAN is the right first choice when shapes are unknown and you expect outliers."*

> 💡 **Point 5's suggested workaround deserves unpacking, because it's a genuinely useful practical
> pattern beyond DBSCAN itself.** If you need to classify new points against DBSCAN-discovered
> clusters, the standard trick is: run DBSCAN once on your training data to get cluster labels, discard
> the noise points, then **train an ordinary supervised classifier** (KNN is the natural choice, since
> it mirrors DBSCAN's own local-density reasoning) on the (point, cluster-label) pairs. You've converted
> a transductive, unsupervised problem into an inductive, supervised one — using the unsupervised
> algorithm purely to *generate* labels that a downstream supervised model can then generalize from.
> This pattern — "cluster once, then train a classifier on the resulting labels" — recurs throughout
> applied unsupervised learning.

---

# PART E — Evaluating Clusters Without Labels

*50:59 – end*

---

## 24. What's still missing?

> *"All three methods use hard boundaries: **K-Means**: one centroid. **Hierarchical**: one subtree.
> **DBSCAN**: one cluster or noise."* [slide 121]
>
> *"But real data has ambiguity: a purchase fits multiple segments. A document spans multiple topics.
> Border points have genuine uncertainty."*
>
> *"Next: Gaussian Mixture Models. $P(\text{point} \in \text{cluster } k)$ instead of hard assignment."*

The deck's paired figure makes the coming distinction visceral before naming the algorithm: *"Hard
Clustering — 'Each point = exactly one cluster'"* shows crisp Voronoi-style boundaries with every point
a single solid colour; *"Soft Clustering (GMM) — 'Each point = Pr(cluster k | point)'"* shows the same
points now coloured by a *blend* of two cluster colours where the boundary region falls, visually
representing genuine uncertainty about membership rather than an arbitrary hard cutoff.

> 📚 **This is exactly where this lecture ends and Part 2 begins.** GMM's mechanics — the Mahalanobis
> ellipse per cluster from §2, the soft, probabilistic assignment previewed here — are not covered in
> this document. What matters for this lecture is understanding *why* the transition is motivated: hard
> assignment is a genuine modeling limitation, not just a computational convenience, whenever real data
> exhibits ambiguity that a single deterministic label cannot represent.

---

## 25. Evaluating clusters: intrinsic metrics

> *"No ground-truth labels? Measure geometric quality instead."* [slide 117]

**Distortion (Inertia)**

$$J = \sum_{k}\sum_{x\in C_k}\|x-\mu_k\|^2$$

*"Sum of squared distances from each point to its centroid (K)."* — this is exactly K-Means' own
objective (§5), reused as a post-hoc quality metric.

**Silhouette Score**

$$s(x) = \frac{b(x)-a(x)}{\max(a(x),b(x))} \in [-1, +1]$$

*"$a$ = mean dist to own cluster. $b$ = mean dist to nearest other cluster."* [slide 51]

| Symbol | Read it as | What it means |
|---|---|---|
| $a(x)$ | "a of x" | The mean distance from point $x$ to every **other** point in its **own** cluster — how tightly packed its own cluster is around it |
| $b(x)$ | "b of x" | The mean distance from $x$ to every point in the **nearest neighbouring** cluster — how close the *next best* alternative cluster is |
| $s(x)$ | "s of x" | The silhouette score for point $x$: positive means $x$ sits comfortably in its own cluster ($a < b$); near zero means $x$ is right on a boundary; negative means $x$ is actually closer, on average, to a *different* cluster than the one it's assigned to |

The deck's own figure marks three worked examples directly on a scatter, with three genuinely distinct
values: a point *"deep inside"* its cluster scores $s = 0.83$ (near $+1$, clearly correct); a point *"on
boundary"* scores $s = 0.40$ (positive but modest); a point *"far from cluster"* scores $s = 0.45$
[slide 51].

> ⚠️ **The real oddity worth noticing is the ordering, not a labeling error.** "Far from cluster"
> ($s=0.45$) scores *higher* than "on boundary" ($s=0.40$) — the opposite of what a naive reading of
> the three captions ("deep inside" > "on boundary" > "far from cluster") would predict. This isn't a
> contradiction once you remember silhouette is a *relative* quantity, $s(x)=(b(x)-a(x))/\max(a(x),b(x))$:
> a point that is genuinely far from its own cluster's centroid can still score well if it is *even
> farther* from every other cluster, so $b(x)$ grows faster than $a(x)$ does. A point sitting exactly on
> the seam between two clusters, by contrast, has $a(x)\approx b(x)$ almost by construction, which is
> what pins "on boundary" to the smallest of the three scores. **Silhouette rewards isolation as much as
> compactness** — a lesson that resurfaces with real numbers in §29–§30's silhouette-paradox demo.

**Calinski-Harabasz Score**

$$CH = \frac{\mathrm{tr}(B_k)/(K-1)}{\mathrm{tr}(W_k)/(n-K)}$$

*"$B$ = between-cluster scatter, $W$ = within-cluster scatter. Large gaps + tight ellipses = high
ratio = good clustering."*

| Symbol | Read it as | What it means |
|---|---|---|
| $\mathrm{tr}(B_k)$ | "trace of B sub k" | Total dispersion **between** cluster centroids — how far apart the clusters' centres are from each other and the overall mean |
| $\mathrm{tr}(W_k)$ | "trace of W sub k" | Total dispersion **within** clusters — how spread out each cluster is internally |
| $K-1$, $n-K$ | — | Degrees-of-freedom normalizations, analogous to an F-statistic's denominator |

**Davies-Bouldin Index**

> *"Average similarity between each cluster and its most similar one. Lower = better. Penalizes
> overlapping clusters."* [slide 117] · `davies_bouldin_score(X, labels)`

The slide states the definition and the sklearn call but not the formula itself; the standard formula
behind "similarity between clusters" is worth having, since it's exactly what makes "lower is better"
true rather than asserted:

$$DB = \frac{1}{K}\sum_{i=1}^{K}\max_{j \ne i}\left(\frac{s_i + s_j}{d(c_i, c_j)}\right)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $s_i$ | "scatter of cluster i" | The mean distance from every point in cluster $i$ to its own centroid $c_i$ — how spread out cluster $i$ is internally |
| $d(c_i,c_j)$ | "distance between centroids i and j" | How far apart two clusters' centres sit |
| $\frac{s_i+s_j}{d(c_i,c_j)}$ | "similarity of i and j" | Large when either cluster is loose ($s_i$ or $s_j$ large) *or* the two centroids sit close together — exactly the two ways two clusters can look like they're overlapping |
| $\max_{j\ne i}(\cdot)$ | "worst-case partner" | For cluster $i$, its *most* similar (most confusable) other cluster — the single worst pairing, not an average over all of them |
| $DB$ | "the index" | The average, across all $K$ clusters, of each cluster's worst-case similarity to some other cluster |

> 💡 **Why "lower is better" here, unlike every other metric on this slide.** Silhouette and
> Calinski-Harabasz are both built so that a *larger* number means better separation — DB is built the
> opposite way on purpose: $DB$ is literally an average *similarity* (confusability) score, so a
> well-separated clustering, where every cluster's worst-case partner is still clearly distinct, drives
> the ratio $\frac{s_i+s_j}{d(c_i,c_j)}$ toward zero for every $i$. **Squinting at "higher = better"
> across all four intrinsic metrics is a common, avoidable interview mistake — DB is the one exception.**

> 💡 **Notice the shared shape across all four intrinsic metrics.** Distortion asks "how tight are
> the clusters?" alone. Silhouette asks "how tight *relative to the nearest alternative*?" — a ratio,
> not a raw number, which is why it's bounded in $[-1,+1]$ and comparable across datasets. Calinski-
> Harabasz asks essentially the same relative question as silhouette — between-cluster spread over
> within-cluster spread — but computed globally across all clusters at once rather than per-point,
> making it an F-statistic-like single summary number rather than a per-point diagnostic. Davies-Bouldin
> asks the identical relative question a third way — cluster looseness over inter-centroid distance —
> but reports it as a *confusability* score (lower is better) rather than a separation score (higher is
> better). **All four are answering some version of "compact clusters, well-separated from each other,"**
> which is a geometric proxy for quality in the total absence of ground-truth labels.

---

## 26. Evaluating clusters: extrinsic metrics

> *"When you have ground-truth labels (validation sets, benchmarks):"* [slide 118]

**Adjusted Rand Index (ARI)**

*"Measures agreement between predicted and true labels, corrected for chance. Range: $[-1, +1]$,
0 = random, 1 = perfect."*

**Normalized Mutual Information (NMI)**

*"How much knowing the cluster tells you about the true label (and vice versa). Range: $[0, 1]$,
1 = perfect agreement."*

> *"When to use extrinsic metrics: benchmarking algorithm choices on labeled subsets · [does the
> clustering] match business-defined segments · [comparing] different hyperparameters (K, eps,
> linkage)"*

> 📚 **Background the slide assumed — why "corrected for chance" matters for ARI.** A naive agreement
> score (what fraction of point-pairs are classified consistently between the predicted and true
> labeling) is misleadingly high even for a *completely random* clustering, simply because with enough
> clusters, many pairs will accidentally agree by chance alone. ARI subtracts off the *expected*
> agreement under a random labeling with the same cluster-size distribution, so that a score of exactly
> 0 genuinely means "no better than chance" rather than some arbitrary positive baseline — which is
> exactly why its range starts at $-1$ (worse than chance is possible) rather than at $0$.
>
> **NMI, recall, is literally the mutual information from Prerequisite 6**, normalized into $[0,1]$ by
> dividing by (a function of) the entropies of the two labelings. It answers a subtly different
> question from ARI: not "do the labelings agree pairwise" but "does knowing one labeling reduce your
> uncertainty about the other."

---

## 27. Which metric for which algorithm?

> *"No single metric is universal. Match the metric to the algorithm and use case."* [slide 119]

| | **K-Means** | **Hierarchical** | **DBSCAN** |
|---|---|---|---|
| Primary | Elbow plot (distortion vs K) | Dendrogram (visual cut) | Silhouette (non-noise only) |
| Confirm / Quantitative / Stability | Silhouette score | Cophenetic correlation | Stability: vary $\epsilon$ by $\pm10\%$ and confirm the clustering doesn't qualitatively change |
| Quick check / Parameter | Calinski-Harabasz | Validate: Silhouette at chosen cut | k-distance elbow for $\epsilon$ |
| Watch for | silhouette < 0.25 means clusters overlap badly | cophenetic r < 0.7 means tree poorly represents distances | >50% points as noise means eps is too small |

> *"Metrics can hide pathologies that a scatter plot reveals instantly."*

> 📚 **Background the slide assumed — cophenetic correlation.** For hierarchical clustering, the
> **cophenetic distance** between two points is the *height at which they first get merged into the
> same cluster* in the dendrogram. The cophenetic correlation coefficient then measures how well those
> merge-heights correlate with the *actual pairwise distances* in the original data — a value near 1
> means the dendrogram is a faithful summary of the true distance structure; a value below 0.7 (the
> deck's stated threshold) signals that the tree is distorting the real geometry, and you should not
> trust cuts made from it.
>
> **DBSCAN's own "Stability" check, named on the same slide, is a sanity check on $\epsilon$ itself, not
> a metric on the clustering.** Re-run DBSCAN with $\epsilon$ nudged $\pm10\%$ from your chosen value; if
> the resulting clusters and noise assignment stay qualitatively the same, $\epsilon$ was chosen from a
> stable region of the k-distance plot (§21) rather than from a knife-edge value that happens to work
> only at that exact number. This is the DBSCAN-specific analogue of running K-Means with multiple
> `n_init` restarts (§6) or checking a dendrogram's cophenetic correlation — a cheap, practical check
> that the result isn't an artefact of one specific hyperparameter choice.

> 🎯 **This table, plus the closing warning "metrics can hide pathologies that a scatter plot reveals
> instantly," is a strong two-part interview answer to "how do you know if your clustering is good?"**
> Part one: name the right *metric* for the algorithm you used, from this table. Part two: never trust
> the metric alone — a single scalar number cannot capture everything a 2-D scatter plot (or, as §29
> is about to demonstrate, a t-SNE projection) reveals to the eye in one glance.

---

## 28. 🧪 Live demo — clustering real handwritten digits

The deck now runs a complete Jupyter notebook, live, from scratch. Every printed number below is
transcribed directly from the captured screen output.

### Step 1 — Load data: 5 visually distinct digits

> *"Each digit is an 8×8 grayscale image → flattened to a 64-dimensional vector. We pick digits 0, 1,
> 4, 6, 7 — they have very different shapes, so K-Means should separate them cleanly."*

```python
digits = load_digits()
selected_digits = [0, 1, 4, 6, 7]
mask = np.isin(digits.target, selected_digits)
X = digits.data[mask]
y_raw = digits.target[mask]

# Remap labels to 0-4 for cleaner indexing
label_map = {d: i for i, d in enumerate(selected_digits)}
y = np.array([label_map[d] for d in y_raw])
```

**Output:**

```
Dataset: 901 images, each 64 dimensions (8x8 pixels flattened)
Selected digits: [0, 1, 4, 6, 7]
  Digit 0: 178 samples
  Digit 1: 182 samples
  Digit 4: 181 samples
  Digit 6: 181 samples
  Digit 7: 179 samples
```

*Check the total:* $178+182+181+181+179 = \mathbf{901}$ ✓, matching the printed dataset size exactly.

> 💡 **Why five specific digits, and why remap labels?** Choosing digits with genuinely different
> visual shapes (0, 1, 4, 6, 7 — versus, say, including both 3 and 8, which are easily confused even by
> humans) stacks the deck in K-Means' favour, which is deliberate: the whole point of the demo is to
> show that even in this *favourable* case, evaluation metrics can still surprise you. Remapping raw
> labels $\{0,1,4,6,7\}$ to $\{0,1,2,3,4\}$ is pure bookkeeping — it lets downstream code index cluster
> arrays with `range(5)` cleanly, with no effect on the clustering itself.

### Step 2 — K-Means clustering (K=5), with zero labels

> *"Can K-Means discover the 5 digit classes from pixel vectors alone — with **zero labels used**?"*

```python
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=30)
cluster_labels = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, cluster_labels)
print(f'Silhouette Score: {sil_score:.3f}  (1.0 = perfect, 0 = overlapping)\n')

# Cluster purity
print('Cluster purity (what digit dominates each cluster):')
print('=' * 55)
total_correct = 0
for c in range(5):
    c_mask = cluster_labels == c
    if c_mask.sum() == 0:
        continue
    digit_counts = np.bincount(y[c_mask], minlength=5)
    dominant_idx = digit_counts.argmax()
    dominant_digit = selected_digits[dominant_idx]
    purity = digit_counts[dominant_idx] / c_mask.sum()
    total_correct += digit_counts[dominant_idx]
    print(f'  Cluster {c}: mostly digit {dominant_digit} ({purity:.0%} pure, {c_mask.sum()} members)')

print(f'\nOverall accuracy: {total_correct/len(y):.1%}  (with ZERO labels given to K-Means!)')
```

**Output:**

```
Silhouette Score: 0.187  (1.0 = perfect, 0 = overlapping)

Cluster purity (what digit dominates each cluster):
=======================================================
  Cluster 0: mostly digit 6 (86% pure, 204 members)
  Cluster 1: mostly digit 7 (96% pure, 187 members)
  Cluster 2: mostly digit 0 (99% pure, 177 members)
  Cluster 3: mostly digit 1 (91% pure, 168 members)
  Cluster 4: mostly digit 4 (99% pure, 165 members)

Overall accuracy: 93.8%  (with ZERO labels given to K-Means!)
```

*Check the member counts sum to the dataset size:* $204+187+177+168+165 = \mathbf{901}$ ✓.

> 💡 **Sit with what just happened, because it's the deck's whole thesis in one printed block.**
> K-Means was given **only** 901 raw 64-dimensional pixel vectors — no labels, no hints about what a
> "0" or a "6" looks like — and correctly grouped 93.8% of them by *true digit identity*, purely from
> pixel-similarity geometry. That's the payoff §3's economic argument promised: structure genuinely is
> "free" when it exists in the data, and unsupervised clustering can recover it without ever seeing a
> label. **But look at the very first printed number: Silhouette Score = 0.187.** That is a low score
> by the "watch for silhouette < 0.25" threshold from §27's own table — and yet the clustering is
> *93.8% correct*. This is the exact paradox §29 exists to resolve.

### Step 3 — t-SNE visualisation (64D → 2D)

The demo projects the same 901 points, 64-dimensional, down to 2-D with t-SNE and produces two
side-by-side panels: *"Colored by TRUE Digit Label"* and *"Colored by K-Means Cluster
(unsupervised!)"*. The two panels are visually almost indistinguishable — the same five roughly
circular blobs appear in both, in matching positions, just recoloured — which is the visual
confirmation of the 93.8% purity number: K-Means' unsupervised grouping and the true digit identity
are, spatially, telling almost the identical story.

> ⚠️ Recall from [Dimensionality Reduction Part 2 §26](../Dimensionality%20Reduction/dimensionality-reduction-02.md):
> **t-SNE preserves local neighbourhood structure but distorts inter-cluster distances and cluster
> sizes.** The visual similarity of the two panels here is legitimate evidence that the *point-to-point
> neighbourhood assignments* agree between true labels and K-Means clusters — but you should not read
> anything into how far apart the five blobs are drawn from each other, or how large each blob appears,
> per that module's own warning.

### Step 4 — what did K-Means learn? (Cluster centroids as images)

```python
centroids_pixels = scaler.inverse_transform(kmeans.cluster_centers_)

for i, c in enumerate(centroids_pixels):
    ax = axes[i]
    ax.imshow(c.reshape(8, 8), cmap='hot', interpolation='nearest')
    dominant_digit = ...
    ax.set_title(f'"{dominant_digit}"', color='white')
```

**Output:** five small 8×8 heat-map images, captioned *"K-Means Cluster Centroids — the 'average'
digit per cluster,"* each one visually recognisable as a slightly blurred, averaged version of one of
the digits 6, 7, 0, 1, 4 — matching the dominant digit found in each cluster's purity breakdown above.

> 💡 **This is §5's convergence proof (the centroid update step) made visible.** Each centroid is, by
> construction, the *mean pixel vector* of every point assigned to its cluster (§5.2's derivation:
> $\mu_k$ minimizes $\sum\|x-\mu_k\|^2$, and that minimizer is exactly the arithmetic mean). Averaging
> together ~170–200 images of the digit "6," pixel by pixel, produces a smudged but still-recognisable
> average "6" — a direct, visual confirmation that K-Means' abstract mathematical update rule is doing
> something intuitively sensible.

### Step 5 — why the elbow method fails here (and that's OK)

```python
k_range = range(2, 15)
inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
```

> *"The elbow method looks for a 'kink' in inertia vs K. But for real data with non-spherical,
> overlapping clusters, the curve is often smooth — there is no clear elbow, and let's see this, and
> learn **when elbow works vs. when it doesn't**."*

The demo's own resulting plots — an inertia-vs-$K$ curve and a silhouette-vs-$K$ curve side by side —
show exactly the smooth, gently-decaying inertia curve the text predicts: no sharp visual "kink" is
apparent anywhere in the range $K{=}2$ to $K{=}14$. The silhouette-vs-$K$ curve, by contrast, does show
a genuine local structure — rising and falling with more visible texture — which is itself informative:
**when the elbow method's own diagnostic (a visible kink) fails to appear, that failure is a signal
you should fall back to a different intrinsic metric (silhouette) rather than force-reading a kink that
isn't really there.**

---

## 29. 🧪 Step 6: Deep dive — why is silhouette low despite 94% accuracy?

> *"This is one of the most important lessons in unsupervised learning."*

The demo poses the exact question raised in §28 and then answers it with real computed numbers, not
assertion.

> *"What silhouette actually measures: $s(i) = (b(i) - a(i)) / \max(a(i), b(i))$. $a(i)$ = average
> distance from point $i$ to all other points in **its own cluster**. $b(i)$ = average distance from
> point $i$ to all points in **the nearest other cluster**."*
>
> *"Why it's low here: **Within-cluster spread**. The problem isn't that clusters overlap (they don't
> — 94% accuracy proves it). The problem is **within-cluster variance**:"*
>
> - *"Handwritten '4's vary enormously: some are angular, some slanted, some have closed tops"*
> - *"So $a(i)$ (distance to your own cluster-mates) is already large (~8-9)"*
> - *"The nearest other cluster $b(i)$ might only be slightly farther (~10-11)"*
> - *"Silhouette $\approx (10-8)/10 = 0.2$ — mathematically correct!"*
>
> *"Silhouette asks: 'Is this point *much* closer to its own cluster than the nearest other?' Answer:
> 'Only slightly closer' — because each digit class is a spread-out blob, not a tight ball."*
>
> *"**The disconnect:** Purity/Accuracy = 'Is this point in the *right* cluster?' → YES (94%).
> Silhouette = 'Is this point *confidently* inside its cluster?' → Not really (0.19). **Both are
> correct. They measure different things. High accuracy + low silhouette = correct but not compact
> clusters.**"*

### 🧪 The exact numbers, computed

```python
print('SILHOUETTE DECOMPOSITION:')
print(f'  Average a(i) (intra-cluster distance): {a_values.mean():.2f}')
print(f'  Average b(i) (nearest-other distance): {b_values.mean():.2f}')
print(f'  Average gap (b - a): {(b_values - a_values).mean():.2f}')
print(f'  Average silhouette (b-a)/max(a,b): {((b_values-a_values)/np.maximum(a_values,b_values)).mean():.3f}')
```

**Output:**

```
Average a(i) (intra-cluster distance): 8.06
Average b(i) (nearest-other distance): 9.87
Average gap (b - a): 1.81
Average silhouette (b-a)/max(a,b): 0.187

→ The gap between b and a is small (~1.8) relative to a (~8.1)
  Points ARE in the right cluster, but their own cluster is spread out.
  Silhouette rewards COMPACT clusters, not just CORRECT assignments.
```

*Verify the silhouette formula on the averages directly:*
$\frac{b-a}{\max(a,b)} = \frac{9.87-8.06}{\max(8.06,9.87)} = \frac{1.81}{9.87} = \mathbf{0.1834}$ —
matching the printed 0.187 to rounding (the small discrepancy is because the printed 0.187 is the
*average of the per-point ratios*, not the ratio of the averages — a genuinely different, and
technically more correct, quantity, since silhouette is defined per-point and then averaged, not
computed from averaged $a$ and $b$ values).

> 🎯 **This is the single most valuable derivation in the whole lecture, and it deserves to be
> memorised almost verbatim.** The failure mode being demonstrated is *not* that K-Means put the wrong
> points together — purity/accuracy already proved that's not what's happening. The failure mode is
> that **silhouette and accuracy measure genuinely different properties of a clustering**, and a
> clustering can be excellent on one axis and mediocre on the other simultaneously, with no
> contradiction:
>
> $$\boxed{\textbf{Purity/Accuracy asks: "Is this point in the right group?" — a question about CORRECTNESS.}}$$
> $$\boxed{\textbf{Silhouette asks: "Is this point confidently, tightly inside its group?" — a question about COMPACTNESS.}}$$
>
> A digit class like handwritten "4" is, in reality, a genuinely *diffuse* population — different
> people write 4s with open or closed tops, different slants, different proportions — so **even the
> mathematically perfect clustering of real 4s together will still have a large $a(i)$**, because the
> real 4s themselves are spread out in pixel space. No clustering algorithm, however good, can make a
> genuinely diffuse population look compact. **The metric is correctly reporting a true property of the
> data's geometry, not a flaw in the algorithm.**

### 💼 Interview question

*"Your clustering has 90%+ purity against a labeled validation set, but silhouette score is only
0.2. Is something wrong?"* — Not necessarily. Compute $a(i)$ and $b(i)$ directly (as this demo does):
if the *gap* $b-a$ is small relative to $a$ itself, the classes are simply spread out (high
within-class variance) rather than misclustered. Purity/ARI/NMI measure *correctness against ground
truth*; silhouette measures *geometric compactness with no reference to ground truth at all*. The two
can and do diverge whenever the true classes are inherently diffuse rather than tightly clustered —
which is common for anything involving natural human variation (handwriting, natural images, free-text
categories). **The fix, if you want compactness too, is not to distrust the clustering — it's to
change the representation** (exactly what §30 does next).

---

## 30. 🧪 Step 7: the fix — PCA reduces noise, tightens clusters

> *"Why does PCA help? Many of the 64 pixel dimensions are **noise** — they vary randomly within a
> class without carrying useful signal. This noise inflates $a(i)$ (intra-cluster distance) without
> increasing $b(i)$ (inter-cluster distance). Result: clusters become tighter → silhouette jumps."*
>
> *"PCA keeps only the directions with real variance — the ones that actually distinguish digits."*

```python
dims_to_try = [2, 5, 10, 15, 20, 30, 50, 64]
results = []

for d in dims_to_try:
    if d >= X_scaled.shape[1]:
        X_reduced = X_scaled
    else:
        X_reduced = PCA(n_components=d, random_state=42).fit_transform(X_scaled)

    km = KMeans(n_clusters=5, random_state=42, n_init=30)
    cl = km.fit_predict(X_reduced)

    sil = silhouette_score(X_reduced, cl)
    ari = adjusted_rand_score(y, cl)
    nmi = normalized_mutual_info_score(y, cl)
    # accuracy via cluster purity (same procedure as Step 2)
    ...
```

**Output — the full accuracy-vs-silhouette table across every dimensionality tried:**

```
Dims | Silhouette | ARI   | NMI   | Accuracy
------------------------------------------------
   2 |      0.482 | 0.704 | 0.709 |    86.2%
   5 |      0.380 | 0.586 | 0.655 |    75.2%
  10 |      0.326 | 0.849 | 0.852 |    93.5%
  15 |      0.268 | 0.821 | 0.846 |    91.8%
  20 |      0.244 | 0.856 | 0.858 |    93.8%
  30 |      0.214 | 0.859 | 0.861 |    93.9%
  50 |      0.193 | 0.859 | 0.861 |    93.9%
  64 |      0.187 | 0.856 | 0.860 |    93.8%

→ Clustering ACCURACY stays ~94% across all dimensions!
  But silhouette jumps from 0.19 (64D) to 0.4+ (10D)
  The clusters are equally good — silhouette just needs lower dimensions to see it.
```

> 🧪 **Read this table as one continuous experiment, and the pattern is unmistakable.** Look down the
> **Accuracy** column first: it is essentially flat, hovering at 93–94% from $d{=}10$ all the way to
> $d{=}64$ (with $d{=}2$ and $d{=}5$ being genuine exceptions — too few dimensions to preserve enough
> signal, which is its own lesson: compression *can* go too far). **Now look at the Silhouette column
> over that identical stable-accuracy range**: it climbs steadily from 0.187 at $d{=}64$ to 0.244 at
> $d{=}20$ to 0.326 at $d{=}10$ — **without the clustering itself, or its correctness, changing at
> all.** This is direct, numeric proof of §29's claim: silhouette was never measuring whether K-Means
> got the *right answer*. It was measuring how *compact* the clusters looked in whatever
> dimensionality you happened to evaluate them in — and most of the 64 raw pixel dimensions are noise
> that inflates within-cluster spread ($a(i)$) without adding any real inter-cluster separation
> ($b(i)$), exactly as the deck's own explanation predicted.

The demo closes with a live plot — *"Same data, same K-Means — only the embedding dimension
changes"* — three panels titled Silhouette Score, ARI & NMI (need true labels), and Cluster Purity
(Accuracy), each plotted against PCA dimensions from 2 to 64. The Silhouette panel shows a smooth,
monotonic downward curve as dimensions increase. The ARI/NMI panel and the Purity panel both show a
sharp rise from $d{=}2$ to $d{=}10$ and then an essentially flat plateau thereafter — visually
confirming the table's own story in one glance.

### 🧪 Step 8: metric cheat sheet — when to use what

> [final summary slide]

| Metric | Needs True Labels? | Measures | Good For |
|---|---|---|---|
| Silhouette | No | How tight vs. separated clusters are geometrically | Choosing K, comparing algorithms — but **biased in high-D** |
| Inertia (elbow) | No | Total within-cluster variance | Finding the "elbow" for K |
| ARI (Adjusted Rand Index) | Yes | Agreement between clustering and true labels (chance-corrected) | Gold standard when you have labels |
| NMI (Normalized Mutual Info) | Yes | Information shared between clustering and true labels | Robust to different K values |
| Purity/Accuracy | Yes | Fraction of points matching their cluster's dominant class | Intuitive but inflates with large K |

> *"The deeper lesson: **In practice you rarely have true labels** (otherwise why cluster?) → you're
> stuck with silhouette/inertia. **Silhouette in high-D is misleading** → always consider PCA or UMAP
> first."*

> 🎯 **This is the practical workflow to carry away from the entire lecture, stated as a checklist.**
> When you cluster real, unlabeled, high-dimensional data and evaluate with silhouette: **(1)** don't
> trust a low silhouette score on its own as proof the clustering is bad — check accuracy/purity if
> *any* labels exist for even a validation subset; **(2)** if no labels exist at all, reduce
> dimensionality first (PCA, UMAP) before trusting silhouette's absolute value, since noise dimensions
> systematically depress it regardless of clustering quality; **(3)** always look at a 2-D projection
> (t-SNE or UMAP) directly, because — per §27's closing warning — a single scalar metric can hide
> exactly the kind of pathology (or non-pathology, as here) that a picture reveals instantly.

---

## Putting it together

```
                    "What is the SIMPLEST EXPLANATION of this data?"                    §Big picture
                                        │
        ┌───────────────────────────────┴────────────────────────────────┐
        ▼                                                                ▼
  GEOMETRIC ANSWER (today)                                    PROBABILISTIC/GENERATIVE
  "What are the natural groups?"                              (GMM, LDA, VAE, GAN — later parts)
        │
        │   distance itself is a MODELING CHOICE                                        §1–§3
        │   Lp norms (diamond/circle/square) · Mahalanobis = Euclidean after whitening
        │   Jaccard (sets) · DTW (time series)
        │
        ▼
  "What is a cluster?" — every algorithm answers differently                            §4
        │
   ┌────┴─────────────────┬─────────────────────────┐
   ▼                       ▼                         ▼
 CENTROID              CONNECTIVITY               DENSITY
 "one center point"    "built by merging"          "dense region,
   │                     │                          sparse gaps"
   ▼                     ▼                            ▼
 K-MEANS               HIERARCHICAL                 DBSCAN
 §5 J = Σ‖x-μ‖²         §12 O(n³), bottom-up          §17-19 core/border/noise
 §5.2 converges to      §13 dendrogram: ONE run       §19 the algorithm
   a LOCAL min only        → ALL values of K          §21 k-distance plot → eps
   (proof: both steps    §14 linkage: Ward = the       §22-23 strengths/limits
   only ever DECREASE J)    K-Means objective,          "no predict() — 
 §7 elbow: J always        applied incrementally         TRANSDUCTIVE"
   decreases, so pick    §16 greedy + IRREVERSIBLE
   the BEND not the min     merges — same local-
 §8 fails: non-spherical,   optimum caveat as K-Means
   varying density,
   outliers
 §9 K-Means++: d² -
   weighted seeding
   fixes "2 seeds, 1
   cluster"
        │                     │                         │
        └─────────────────────┴─────────────────────────┘
                                │
              ALL THREE PRODUCE HARD ASSIGNMENTS ONLY                                   §24
              "a purchase fits multiple segments" → Next: GMM (Part 2)
                                │
        ════════════════════════════════════════════════════════════
                  EVALUATING ANY OF THEM, WITH NO LABELS                                §25-27
        intrinsic: distortion · SILHOUETTE (a,b) · Calinski-Harabasz
        extrinsic: ARI (chance-corrected) · NMI (= mutual information, reused)
        "no single metric is universal — match metric to algorithm"
                                │
        ════════════════════════════════════════════════════════════
              LIVE DEMO: 901 real digits, K-Means, PCA                                  §28-30
        93.8% purity/accuracy  +  silhouette = 0.187  ← NOT a contradiction
        a(i)=8.06, b(i)=9.87, gap=1.81 — "correct but not compact"
        PCA to d=10: same 93.5% accuracy, silhouette jumps to 0.326
        "silhouette in high-D is misleading — PCA/UMAP first"
```

### Walking the diagram

**Everything in Parts B–D descends from one framing device: "what is the simplest explanation of this
data?", answered three different geometric ways.** K-Means says the simplest explanation is *a small
set of representative points* — every other point explained as "closest to point $\mu_k$." Hierarchical
clustering says the simplest explanation is *a merge history* — every grouping explained as "these two
things were, at some point, each other's nearest neighbours." DBSCAN says the simplest explanation is
*local density* — every grouping explained as "these points are packed closely enough together to
count as one region, and this other point is not." None of the three is more "correct" than the others
in the abstract; each is the right tool exactly when its specific geometric assumption matches the
true shape of your data, and wrong exactly when it doesn't — which is what §8's "when K-Means fails"
and its analogues in §16 and §22 are really cataloguing.

**Two proof techniques recur across all three algorithms, and noticing the recurrence is worth more
than memorising either proof alone.** K-Means' convergence guarantee (§5.2) rests on showing that
*both* of its alternating steps can only ever decrease the objective $J$, so the sequence of $J$ values
is non-increasing and bounded below, hence must converge — but to a *local*, not global, minimum,
because nothing rules out getting stuck at a point where neither step can improve things further.
Hierarchical clustering's greedy, irreversible merges (§16) carry the *identical* structural caveat in
different clothing: each merge is locally optimal given what's already been decided, with no mechanism
to revisit an early mistake. **"Locally optimal at every step, with no guarantee of global optimality"
is the load-bearing idea connecting §5.2 and §16**, and it is worth being able to state explicitly
rather than treating the two algorithms' limitations as unrelated facts to memorise separately.

**Evaluation (§25–§27) exists because none of the three algorithms comes with a built-in notion of
"how good was that?"** — they are all defined procedurally (run these steps) rather than in terms of
optimizing some externally-validated target. Intrinsic metrics substitute pure geometry (how tight, how
separated) for a ground truth that doesn't exist; extrinsic metrics become available only when you
happen to have *some* labels to check against, which — as the demo's own closing lesson states — is
rare precisely because if you had comprehensive labels, you likely wouldn't need to cluster in the
first place.

**And the live demo (§28–§30) is the payoff that ties every earlier thread together into one worked
example.** It uses K-Means (§5), evaluates with silhouette (§25) *and* purity/accuracy (an intrinsic
proxy the deck builds from scratch, playing the role extrinsic metrics like ARI would play with true
labels available), visualises with t-SNE (a direct callback to
[Dimensionality Reduction Part 2](../Dimensionality%20Reduction/dimensionality-reduction-02.md)), and
resolves its central paradox with PCA (another direct callback). **The single sentence worth carrying
out of the entire lecture is the demo's own closing line: "Purity/Accuracy asks 'is this point in the
right cluster?' — Silhouette asks 'is this point confidently inside its cluster?' — Both are correct.
They measure different things."** Every other fact in this document is context that makes that one
sentence land with its full force.

---

## Interview prep — Amazon Applied Scientist

### Core questions

<details>
<summary><b>1. (Easy)</b> What are the four families of clustering, and what does each assume a "cluster" is?</summary>

- **Centroid-based (K-Means)** — a cluster is the set of points closest to one representative centroid.
  Assumes spherical, similarly-sized, similarly-dense clusters.
- **Connectivity-based (Hierarchical)** — a cluster is built bottom-up by repeatedly merging the two
  nearest existing clusters. Still assumes roughly compact clusters; doesn't require choosing $K$
  upfront.
- **Density-based (DBSCAN)** — a cluster is a dense region of points, separated from other dense
  regions by sparse space. No shape assumption at all; naturally identifies outliers as noise.
- **Distribution-based (GMM)** — a cluster is a Gaussian distribution with its own mean and covariance
  (a Mahalanobis ellipse). Produces soft, probabilistic membership rather than hard assignment.
  *(Covered in Part 2, not this lecture.)*

**The strongest version of this answer** names the deck's own demonstration: run all four on identical
data containing two concentric rings, and K-Means and Hierarchical both fail (they split the rings into
wedges/fragments), while DBSCAN succeeds outright and GMM partially succeeds by fitting flexible
ellipses. The failure pattern on that one dataset is itself proof that "cluster" means something
genuinely different to each algorithm.
</details>

<details>
<summary><b>2. (Easy)</b> Why does K-Means always converge, and why doesn't that guarantee a good answer?</summary>

K-Means alternates two steps — assign each point to its nearest centroid, then move each centroid to
the mean of its assigned points — and **both steps can only ever decrease (or leave unchanged) the
objective $J = \sum_k\sum_{x\in C_k}\|x-\mu_k\|^2$.** Reassignment can't make a point's distance to its
new (nearest) centroid worse than to its old one; and for a fixed assignment, the mean is *provably*
the minimizer of the sum of squared distances (a one-line calculus argument: differentiate, set to
zero, solve for $\mu$). A non-increasing sequence bounded below by zero must converge.

**But that argument only proves convergence to *some* fixed point where neither step can improve
things further — a local minimum.** Nothing in the proof rules out landing in a bad local minimum
determined entirely by where the centroids started. This is exactly why initialization matters so much
(K-Means++, and sklearn's default of 10 restarts via `n_init`), and why the elbow method's minimum
$J$ is not itself the answer to "what's the best $K$" — $J$ is always minimized (trivially, at zero) by
$K=N$.
</details>

<details>
<summary><b>3. (Medium)</b> Explain Mahalanobis distance and derive its relationship to Euclidean distance.</summary>

Mahalanobis distance is $d_M(x,\mu) = \sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)}$ — it's Euclidean distance
that has been **normalised by the data's own covariance structure**, so that a point sitting along the
natural, high-variance direction of a correlated data cloud is treated as "close," while an equally
raw-distant point sitting *across* the cloud's natural spread is treated as genuinely far.

**Derive the connection to Euclidean distance via whitening.** Let $\tilde x = \Sigma^{-1/2}x$ be the
whitening transform (which turns any correlated data cloud into one with identity covariance — a
perfect sphere). Then:

$$d_M(x,\mu)^2 = (x-\mu)^\top\Sigma^{-1}(x-\mu) = \left(\Sigma^{-1/2}(x-\mu)\right)^\top\left(\Sigma^{-1/2}(x-\mu)\right) = \|\tilde x-\tilde\mu\|_2^2$$

using $\Sigma^{-1} = \Sigma^{-1/2}\Sigma^{-1/2}$. **Mahalanobis distance in the original space is
exactly Euclidean distance in the whitened space.** It isn't a different kind of distance — it's the
same distance, computed after first undoing correlation.

**And connect it forward to GMM**, since interviewers will often ask this as a lead-in: a Gaussian
Mixture Model, at bottom, differs from K-Means in exactly one structural way — it learns one full
covariance matrix (one Mahalanobis ellipse) per cluster instead of assuming a shared, spherical
distance metric for all clusters. Everything else about the two algorithms' machinery is a variation on
that single difference.
</details>

<details>
<summary><b>4. (Medium)</b> Compare the three clustering families on complexity, and explain what that implies for choosing between them at scale.</summary>

- **K-Means**: $O(nKdi)$ per run ($n$ points, $K$ clusters, $d$ dimensions, $i$ iterations) — scales
  to millions of points easily, which is why it's the "start here" default.
- **Hierarchical**: $O(n^3)$ time (naive), $O(n^2)$ space. The deck's own stated practical limit is
  **~10K points** — at $n=100{,}000$, $n^3=10^{15}$, simply not computable.
- **DBSCAN**: $O(n\log n)$ with a spatial index (e.g. a k-d tree), which is close to K-Means' scaling
  in practice for low-to-moderate dimensions — though high-dimensional data degrades spatial-index
  performance, connecting back to the distance-concentration problem.

**What this implies practically:** Hierarchical clustering's dendrogram is genuinely valuable (one run,
every value of $K$ available for free) but is essentially a small-to-medium-data tool. For anything at
real production scale, the choice is almost always between K-Means (fast, needs $K$, assumes
spherical/compact shape) and DBSCAN (comparably fast with indexing, no $K$ needed, handles arbitrary
shape, but scale-sensitive and unable to score new points without a full re-run).
</details>

<details>
<summary><b>5. (Medium)</b> Your clustering achieves 93.8% purity against a held-out labeled subset, but silhouette score is only 0.187. A colleague says the clustering must be bad. How do you respond?</summary>

**Not necessarily — and the two metrics are measuring genuinely different properties.** Purity/accuracy
asks "is each point assigned to the group matching its true label?" — a question about *correctness*.
Silhouette asks "is each point, on average, much closer to its own cluster than to the nearest
alternative cluster?" — a question about *geometric compactness*, computed with **no reference to
ground truth at all**.

**Decompose silhouette directly to check.** $s = (b-a)/\max(a,b)$: if $a$ (mean intra-cluster distance)
is *already large* even for correctly-clustered points — because the true underlying class is
genuinely diffuse in the raw feature space (handwriting varies a lot; natural categories rarely form
tight geometric balls) — and $b$ (distance to the nearest other cluster) is only modestly larger, the
*ratio* stays small even though every point sits in its objectively correct group. A concrete example:
$a\approx8.1$, $b\approx9.9$ gives $s\approx(9.9-8.1)/9.9\approx0.18$ — low, and mathematically correct,
for a clustering that is simultaneously 93.8% accurate against ground truth.

**What I'd actually do:** trust the labeled subset's purity/accuracy (or ARI/NMI, the proper
chance-corrected versions) as the primary signal, since it's measuring what actually matters — did the
model find the real groups. Then, if compactness genuinely matters for the downstream use case (e.g.
a nearest-centroid classifier that needs tight decision regions), reduce dimensionality with PCA first
— noise dimensions inflate $a(i)$ without adding real separation, and silhouette recovers dramatically
(in one real demonstration, from 0.187 at 64 raw dimensions to 0.326 at 10 PCA dimensions, with
accuracy essentially unchanged throughout).
</details>

<details>
<summary><b>6. (Medium)</b> Explain the difference between Adjusted Rand Index and Normalized Mutual Information, and when you'd reach for each.</summary>

Both require ground-truth labels, so both are *extrinsic* metrics — but they ask structurally different
questions.

**ARI** looks at every *pair* of points and asks: "did the predicted clustering and the true labeling
agree about whether these two points belong together?" — then chance-corrects that agreement rate so a
random clustering scores 0 and a perfect one scores 1 (range $[-1,+1]$).

**NMI** is literally **mutual information** — the same quantity used as a feature-selection filter in
[Dimensionality Reduction Part 1](../Dimensionality%20Reduction/dimensionality-reduction-01.md) —
normalized into $[0,1]$: "how much does knowing the predicted cluster reduce your uncertainty about the
true label, and vice versa?"

**When to reach for each:** ARI is the more standard "gold standard" comparison metric when cluster
counts roughly match between predicted and true labelings. NMI is more **robust when the number of
predicted clusters differs substantially from the number of true classes** — because it's measuring
shared information rather than pairwise agreement, it degrades more gracefully when, say, your
algorithm found 8 clusters against 5 true classes. In practice, reporting both together (as the demo's
own table does) is standard, since they occasionally diverge and that divergence itself is diagnostic.
</details>

<details>
<summary><b>7. (Medium–hard)</b> Design a clustering pipeline for grouping millions of Amazon delivery addresses by geographic + demand density, with no ground truth available.</summary>

**Start by choosing the family based on the expected shape, not by defaulting to K-Means.** Delivery
zones are naturally irregular — dense urban cores, sparse rural stretches, occasional isolated outlier
addresses (a single house at the end of a long rural road) — which is exactly DBSCAN's use case:
arbitrary shape, no need to pre-specify how many zones exist, and outlier addresses get flagged as
noise rather than forced into an inappropriate zone. **This is precisely one of the deck's own named
Amazon use cases** ("delivery station grouping — cluster delivery addresses by geography and route
density").

**Choosing $\epsilon$ and MinPts at this scale:** compute a k-distance plot (§21) on a representative
sample first — full $O(n\log n)$ DBSCAN with a spatial index (e.g. a ball tree on lat/long, or better,
haversine distance for genuine geographic accuracy rather than naive Euclidean on raw coordinates)
should still be tractable at millions of points, but the *parameter search* itself is best done on a
sample to keep iteration fast.

**Scale-sensitivity is a real trap here**, and worth naming proactively: raw latitude/longitude
degrees are not uniform distances (a degree of longitude is a different physical distance depending on
latitude), so either project to a locally-accurate flat coordinate system first or use a proper
geographic (haversine) distance metric — DBSCAN's "always scale first" warning (§20) applies with extra
force to geographic data specifically.

**Handling varying density** is the honest caveat to raise unprompted: urban and rural delivery
densities differ by orders of magnitude, and a single global $\epsilon$ cannot serve both well (§22's
named limitation). I'd evaluate **HDBSCAN** specifically for this reason — it's designed to handle
exactly this varying-density case by letting the effective density threshold adapt locally.

**Evaluation with no labels:** silhouette and Calinski-Harabasz as the primary intrinsic checks, but
cross-checked against a *business* metric that actually matters operationally — average intra-zone
delivery distance, or driver route efficiency on the resulting zones — since, per §27's closing warning,
a purely geometric metric can look fine while missing an operationally important pathology (e.g. a
zone that's geometrically tight but crosses a river with no bridge nearby).
</details>

<details>
<summary><b>8. (Hard — combines two concepts)</b> Derive why increasing K can never increase K-Means' distortion J, and explain why this makes the elbow method necessary rather than optional.</summary>

**The proof.** Take any optimal $K$-cluster solution with objective value $J_K$. Construct a *valid*
(if not necessarily optimal) $(K{+}1)$-cluster solution by taking any one existing cluster and
splitting it into two identical copies of itself (same points, same centroid, arbitrarily partitioned)
— this new configuration has exactly the same total sum of squared distances, so its objective is still
$J_K$. Since K-Means, run at $K{+}1$, is guaranteed (by §5.2's convergence proof) to find *some* local
minimum with objective **at most** as large as any specific valid configuration you can construct — and
you just constructed one with value $J_K$ — the true optimal $(K{+}1)$-cluster objective must satisfy
$J_{K+1} \le J_K$. Distortion is **monotonically non-increasing in $K$**, with the degenerate case
$K=n$ giving $J_n=0$ exactly (every point is its own cluster and its own centroid).

**Why this makes a naive "minimize $J$" selection rule useless, and the elbow method necessary.** If you
simply asked "which $K$ minimizes the objective?", the answer is *always* $K=n$ — a completely
uninformative clustering where every point is alone. There is no interior minimum to find, because the
curve is monotonic. **The elbow method exists specifically to substitute a different, more useful
question** — not "where is $J$ smallest?" but "where does *increasing $K$ further* stop buying you
much additional reduction in $J$?" — which is a statement about the curve's *rate of change* (its
second derivative, informally, the "bend"), not about its raw minimum. This is a genuinely different
optimization criterion, and understanding that the naive minimum is degenerate is what makes clear why
the more roundabout elbow heuristic is required at all rather than optional cleverness.
</details>

<details>
<summary><b>9. (Hard — combines two concepts)</b> You cluster a dataset with DBSCAN and get one enormous cluster containing 95% of all points, with almost nothing labeled as noise. Diagnose the likely cause and fix it.</summary>

**Two distinct, testable hypotheses, and the fix differs for each — a good answer distinguishes them
rather than guessing.**

**Hypothesis 1 — $\epsilon$ is too large relative to MinPts.** If $\epsilon$ is set too generously, the
density threshold for "core point" becomes trivially easy to satisfy almost everywhere, and DBSCAN's
recursive core-point expansion (§19) chains together what should be separate, sparser regions into one
giant blob — because *any* connected path of sufficiently-dense points, however winding, gets merged
into a single cluster (this is the same "finds chains" property that makes single linkage prone to
similar over-merging, §14). **Diagnostic:** rebuild the k-distance plot (§21) and check whether the
chosen $\epsilon$ actually sits at the genuine elbow, or whether it was set too high. **Fix:** lower
$\epsilon$, or raise MinPts (which raises the density bar for being called "core"), then re-examine the
k-distance elbow.

**Hypothesis 2 — the data genuinely lives in high dimensions with distance concentration.** Per §22's
own named limitation and its direct connection to the
[Dimensionality Reduction](../Dimensionality%20Reduction/dimensionality-reduction-01.md) distance-
concentration result: in high dimensions, the relative gap between "near" and "far" points shrinks
toward zero, so a single fixed $\epsilon$ can stop meaningfully distinguishing density variation across
the *entire* dataset — everything looks roughly equidistant, so everything looks equally "dense" (or
equally sparse), and the algorithm either merges everything or labels everything noise. **Diagnostic:**
check the raw dimensionality of the feature space, and check whether the k-distance plot has *any*
visible elbow at all versus a smooth, featureless curve (a flat curve with no elbow is itself the
signature of distance concentration, exactly as [Dimensionality Reduction Part 2
§5](../Dimensionality%20Reduction/dimensionality-reduction-02.md)'s spectral-plot diagnostic taught for
a different but structurally analogous problem). **Fix:** reduce dimensionality first — PCA or UMAP —
before re-running DBSCAN, exactly as this lecture's own live demo (§30) does to fix a *different*
symptom (low silhouette) of the *same* underlying cause (too many uninformative dimensions).

**Both hypotheses point toward the same general lesson**, worth stating explicitly to close the answer:
DBSCAN's parameters are not universal constants to tune blindly — they are meaningful only relative to
the specific geometry (dimensionality, density variation) of the space they're being applied in, and a
pathological result is evidence about that geometry, not just about parameter choice in isolation.
</details>

### Depth probes — the follow-up when your first answer is good

| Your good answer | The probe | What they want |
|---|---|---|
| "K-Means converges" | *"Converges to what, exactly, and can you prove it?"* | Local minimum only. Proof: both alternating steps are individually non-increasing on $J$; a bounded, non-increasing sequence converges. Nothing guarantees the *global* minimum. |
| "K-Means++ fixes bad initialization" | *"Why $d^2$ specifically, not $d$?"* | Squaring makes the preference for far points more aggressive without being fully deterministic (which would be brittle to a single outlier). It's a probabilistic compromise, with a stated $O(\log K)$-competitive guarantee. |
| "Ward linkage is the best default" | *"Why, mechanistically — not just 'it usually works'?"* | Ward's merge criterion — minimize the increase in within-cluster variance — is literally K-Means' own objective $J$, applied incrementally rather than globally. It inherits K-Means' well-behaved, general-purpose bias. |
| "DBSCAN finds outliers automatically" | *"What exactly makes a point noise, mechanically?"* | A point that never becomes a core point (< MinPts neighbours within $\epsilon$) **and** is never captured within $\epsilon$ of any core point's recursive expansion. Provisional "noise (for now)" labels can later flip to border if a later expansion reaches them. |
| "Silhouette measures cluster quality" | *"Quality of what, specifically — separation, or correctness?"* | Compactness/separation only — $a(i)$ vs $b(i)$, purely geometric, with zero reference to any ground truth. It can be low on a *provably correct* clustering if the true classes are inherently diffuse. |
| "PCA before clustering helps" | *"Helps what, mechanically — the clustering, or the metric?"* | Usually the **metric**, not necessarily the clustering itself. Noise dimensions inflate $a(i)$ (intra-cluster distance) without adding real inter-cluster separation $b(i)$; removing them can leave accuracy/purity essentially unchanged while silhouette rises sharply. |
| "ARI and NMI are both good extrinsic metrics" | *"When would they actually disagree?"* | When the predicted number of clusters differs substantially from the true number of classes — NMI (information-theoretic) tends to degrade more gracefully than ARI (pairwise-agreement-based) in that mismatch case. |
| "DBSCAN has no `.predict()`" | *"How would you score a genuinely new point against DBSCAN-found clusters, then?"* | It's transductive by construction — no summary object exists to compare against. Standard workaround: use DBSCAN's labels (discarding noise) as training labels for a downstream supervised classifier, typically KNN. |

### Whiteboard-ready derivations

**D1 — K-Means converges (but not to a global optimum).**
```
J = Σ_k Σ_{x∈C_k} ‖x − μ_k‖²

STEP 3 (assign):  reassigning x to its nearest centroid can only
                   keep or DECREASE ‖x − μ_{assigned}‖²   ⇒  J non-increasing

STEP 4 (update):  for fixed assignment, minimize Σ_x ‖x−μ‖² over μ:
                   ∂/∂μ Σ ‖x−μ‖² = Σ −2(x−μ) = 0  ⇒  μ* = mean(C_k)
                   ⇒ moving to the mean is PROVABLY optimal for this step
                   ⇒  J non-increasing

Both steps non-increasing, J ≥ 0  ⇒  J converges (bounded, monotonic)
but convergence is to a LOCAL fixed point only — no step ever
compares against a DIFFERENT starting configuration.
```

**D2 — Why K-Means' distortion is always minimized at K=n (hence: the elbow, not the minimum).**
```
Take any K-cluster solution with objective J_K.
Split ANY one cluster into 2 identical copies (same points, same
center) → valid (K+1)-cluster solution with objective STILL = J_K.

K-Means at K+1 finds a local minimum ≤ any valid configuration you
can construct  ⇒  J_{K+1} ≤ J_K.

⇒ J is monotonically non-increasing in K.
⇒ minimizing J directly always picks K = n (J_n = 0, useless).
⇒ must instead look for the BEND (diminishing returns), not the min.
```

**D3 — Mahalanobis = Euclidean after whitening.**
```
d_M(x,μ)² = (x−μ)ᵀ Σ⁻¹ (x−μ)

Σ⁻¹ = Σ⁻¹ᐟ² Σ⁻¹ᐟ²          (Σ symmetric positive-definite)

d_M(x,μ)² = (x−μ)ᵀ Σ⁻¹ᐟ² Σ⁻¹ᐟ² (x−μ)
          = [Σ⁻¹ᐟ²(x−μ)]ᵀ [Σ⁻¹ᐟ²(x−μ)]
          = ‖x̃ − μ̃‖²₂                where x̃ = Σ⁻¹ᐟ² x

⇒ Mahalanobis in original space = Euclidean in WHITENED space.
⇒ GMM's one-Mahalanobis-ellipse-per-cluster is K-Means' shared
  spherical assumption, relaxed to let each cluster learn its own Σ.
```

### Applied scenario — de-duplicating and clustering a fraud-review network

**The problem.** Amazon's trust-and-safety team suspects coordinated fake-review rings: groups of
accounts that review the same products in suspiciously similar patterns (timing, phrasing, rating
distributions). You have behavioural feature vectors for ~2M reviewer accounts and no labels — you
don't know in advance how many rings exist, and legitimate reviewers vastly outnumber fraudulent ones.

**Framing.** This maps almost exactly onto the deck's own named use case (*"Fraud Ring Detection —
identify clusters of coordinated fake reviews or fraudulent accounts by behavioral similarity
patterns"*), and the shape of the problem should drive the algorithm choice directly, not the other way
around. Three properties of the data rule out the naive default:

- **No known $K$** — you have no idea how many fraud rings exist, and forcing an upfront $K$ (as
  K-Means requires) would be guessing at exactly the number you're trying to discover.
- **Extreme class imbalance and non-convex structure** — the overwhelming majority of accounts are
  legitimate, scattered diffusely through feature space; a small number of fraud rings form tight,
  possibly oddly-shaped pockets. A centroid-based method would let the enormous legitimate majority
  dominate cluster placement and likely swallow small fraud pockets into whichever nearby "legitimate"
  centroid is closest.
- **Outliers are the whole point** — most accounts (the legitimate majority) *shouldn't* be forced into
  any tight ring-cluster at all. This is the single strongest argument for DBSCAN over K-Means or
  Hierarchical: DBSCAN's noise category isn't a bug to work around, it's exactly the mechanism that
  lets the vast legitimate population sit unclustered while only genuinely dense, suspicious pockets
  get flagged as rings.

**Distance metric, chosen deliberately (§1–§3), not defaulted to Euclidean.** Behavioural features here
are a mix of things that are *not* naturally comparable on a shared numeric scale — review timing
patterns (better suited to a time-series-aware distance, potentially DTW-flavoured for burst patterns),
categorical account attributes, and possibly set-valued features like "set of product categories
reviewed" (a natural fit for Jaccard similarity). I'd build a composite distance or, more practically,
engineer numeric features that summarise each of these (e.g. a burst-timing score, a Jaccard-based
category-overlap score) before feeding a single numeric feature vector into DBSCAN — rather than
force-fitting raw heterogeneous data into naive Euclidean.

**Parameter selection at 2M scale.** Compute the k-distance plot (§21) on a stratified sample, not the
full 2M — the elbow-finding procedure is cheap on a sample and the resulting $\epsilon$ transfers.
MinPts set conservatively low (a fraud ring might be small, even 3–5 coordinated accounts) — the deck's
rule of thumb (MinPts $\ge$ dim+1) is a floor, not a ceiling; domain knowledge about minimum ring size
should dominate.

**Evaluation with essentially no labels.** This is the genuinely hard part, and the honest answer is
that intrinsic metrics (silhouette, Calinski-Harabasz) are of *limited* use here specifically **because**
the "clusters" of interest are deliberately small, dense minority pockets embedded in a vast, diffuse
majority — silhouette computed globally would be dominated by how the huge unclustered/noise population
is scored, not by the quality of the small clusters that actually matter. I'd instead evaluate the
**found clusters themselves**, individually: for each DBSCAN-identified dense pocket, manually review a
sample and check for genuine behavioural coordination signals (near-identical review timing, review-text
similarity via a separate NLP pipeline) as a proxy ground truth on that specific subset — closer to
building a small labeled validation set after the fact than to trusting a single global unsupervised
metric.

**Failure modes to flag proactively.** Coordinated fraudsters adapt once they learn what gets flagged
(the same distributional-shift concern as any adversarial detection system) — so this needs to be a
**retrained, monitored pipeline**, not a one-off clustering. And a legitimate but unusual reviewer
(someone who genuinely reviews many products from one brand quickly, e.g. an enthusiast) risks a false
positive if the density threshold is too aggressive — human review of flagged clusters before any
account action is non-negotiable, not just good practice.

### Leadership Principles tie-in

**Dive Deep.** §29's silhouette investigation is the model. The shallow response to "silhouette is low"
is to conclude the clustering is bad and start re-tuning hyperparameters. The deep response decomposes
the metric into $a(i)$ and $b(i)$, computes both directly, and discovers the real cause (within-class
variance in the true population, not a clustering error) — a conclusion that would have been missed by
treating the metric as a black-box verdict. *"Rather than accept a low silhouette score as a failure
signal, I computed the decomposition directly and found the clustering was 94% correct against a
labeled subset — the low score was measuring true within-class diversity, not an algorithm failure."*

**Are Right, A Lot** fits the elbow-method monotonicity trap (D2): knowing in advance that "minimize $J$
directly" is a degenerate, useless selection rule — rather than discovering it only after shipping a
model that always picks $K=n$ — is exactly the kind of judgment call this LP rewards. Being right early,
analytically, is cheaper than being right late, empirically.

**Customer Obsession** fits the fraud-ring scenario directly: choosing DBSCAN specifically *because*
its noise category lets the vast legitimate majority go unflagged, rather than a method that forces
every account into some cluster (risking false-positive account actions against real customers), is a
modeling decision made in service of not harming legitimate users — a technical choice with a direct
customer-trust consequence.

**Insist on the Highest Standards** fits the "always scale first" and "check the k-distance elbow before
trusting $\epsilon$" disciplines throughout DBSCAN's section — the difference between a clustering that
looks plausible and one that is actually defensible often comes down to exactly these unglamorous
preprocessing checks that are easy to skip under time pressure.

> 🎯 **stretch — nice to know, not expected for an intern:** the formal proof that Ward's criterion is
> equivalent to minimizing the K-Means objective incrementally; HDBSCAN's hierarchical, variable-density
> extension to DBSCAN; the cophenetic correlation coefficient's exact formula; efficient $O(n^2\log n)$
> nearest-pair data structures for hierarchical clustering at scale; the formal chance-correction
> derivation behind the Adjusted Rand Index. Knowing these exist and roughly what they claim is enough.

---

## Glossary

| Term | Definition |
|---|---|
| **Agglomerative clustering** | Bottom-up hierarchical clustering: start with every point its own cluster, repeatedly merge the two nearest, until one cluster remains. $O(n^3)$ time. §12 |
| **ARI (Adjusted Rand Index)** | Extrinsic metric: pairwise agreement between predicted and true labels, chance-corrected. Range $[-1,+1]$, 0 = random. §26 |
| **Border point** | A point with fewer than MinPts neighbours within $\epsilon$, but itself within $\epsilon$ of some core point. Belongs to that core point's cluster. §18 |
| **Calinski-Harabasz score** | Intrinsic metric: ratio of between-cluster to within-cluster scatter (trace-based, F-statistic-like). Higher is better. §25 |
| **Davies-Bouldin index** | Intrinsic metric: average, over all clusters, of each cluster's worst-case similarity (scatter-over-centroid-distance) to another cluster. Lower is better — the one intrinsic metric in this lecture where that's true. §25 |
| **Centroid-based clustering** | Family where a cluster = the set of points nearest one representative center (K-Means). §4 |
| **Cluster purity / accuracy** | The fraction of points in a cluster matching that cluster's most common true label, when validation labels exist. §28 |
| **Cophenetic correlation** | How well a dendrogram's merge-heights correlate with the true pairwise distances. Below ~0.7 signals a poorly-representative tree. §27 |
| **Core point** | A point with $\ge$ MinPts neighbours within radius $\epsilon$. Forms the backbone of a DBSCAN cluster. §18 |
| **Connectivity-based clustering** | Family where a cluster is built by repeatedly merging nearby clusters (Hierarchical). §4 |
| **DBSCAN** | Density-Based Spatial Clustering of Applications with Noise. Defines clusters as dense regions separated by sparse space. No $K$ needed; labels outliers as noise. §17–§23 |
| **Dendrogram** | A tree recording the full merge history of agglomerative clustering. One run yields every value of $K$. §13 |
| **Density-based clustering** | Family where a cluster = a region of high point density, separated from other such regions by low density (DBSCAN). §4 |
| **Distortion (Inertia)** | K-Means' own objective, $J = \sum_k\sum_{x\in C_k}\|x-\mu_k\|^2$, reused as an intrinsic evaluation metric. Always monotonically decreases as $K$ grows. §5, §25 |
| **Distribution-based clustering** | Family where a cluster = a Gaussian distribution with its own mean and covariance, giving soft membership (GMM). §4, covered in Part 2 |
| **DTW (Dynamic Time Warping)** | A distance for time series that finds the optimal non-linear alignment between two sequences' indices, so shape-matching survives time shifts. §3 |
| **Elbow method** | Plot distortion $J$ vs $K$; pick the $K$ where the curve's rate of decrease sharply diminishes. Necessary because $J$ is minimized (uselessly) at $K=n$. §7 |
| **Extrinsic metric** | An evaluation metric requiring ground-truth labels (ARI, NMI, purity). §26 |
| **Hierarchical clustering** | See Agglomerative clustering. §11–§16 |
| **Intrinsic metric** | An evaluation metric requiring no labels — pure geometry (distortion, silhouette, Calinski-Harabasz). §25 |
| **Jaccard similarity** | $|A\cap B|/|A\cup B|$ — a distance for sets. §3 |
| **K-Means** | Partition $N$ points into $K$ groups by alternately assigning points to the nearest centroid and moving centroids to their assigned points' mean. Converges to a local, not global, minimum. §5 |
| **K-Means++** | Smarter initialization: seed centroids probabilistically, with probability $\propto d(x_i,C)^2$, to avoid two seeds landing in the same true cluster. $O(\log K)$-competitive guarantee. §9 |
| **Linkage** | The rule for measuring distance between two clusters in hierarchical clustering: single (min), complete (max), average (mean), or Ward (min variance increase). §14 |
| **Mahalanobis distance** | Euclidean distance after "whitening" by the data's covariance — $d_M = \sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)}$. Accounts for correlation between features. §2 |
| **MinPts** | DBSCAN's density threshold: the minimum number of neighbours (within $\epsilon$) required for a point to be a core point. §18, §21 |
| **NMI (Normalized Mutual Information)** | Extrinsic metric: mutual information between predicted and true labelings, normalized to $[0,1]$. More robust than ARI when cluster counts mismatch. §26 |
| **Noise point** | A DBSCAN point that is neither a core point nor within $\epsilon$ of one. Belongs to no cluster. §18 |
| **Silhouette score** | $s(x)=(b(x)-a(x))/\max(a(x),b(x))$, range $[-1,+1]$. Measures geometric compactness/separation, **not** correctness against ground truth. Biased low in high dimensions with noisy features. §25, §29–§30 |
| **Transductive** | An algorithm (like DBSCAN) with no `.predict()` for new data — clusters are defined only relative to the specific fitted dataset. §20 |
| **Ward linkage** | Merge the pair of clusters causing the smallest increase in total within-cluster variance — the incremental analogue of K-Means' own objective. Usually the best default. §14 |
| **Whitening** | The transform $\tilde x = \Sigma^{-1/2}x$ that turns a correlated data cloud into one with identity covariance (a perfect sphere). §2, Prereq 4 |

---

## Check yourself

Work these before reading the answers. Questions 9–12 require combining two concepts.

1. Why does K-Means' objective function use *squared* Euclidean distance rather than plain Euclidean
   distance?
2. A dataset has two clusters of very different density — one tight and compact, one loose and spread
   out. Which algorithm handles this better, K-Means or DBSCAN, and why?
3. You run agglomerative clustering once and get a dendrogram. A colleague asks you to also try $K=4$
   and $K=7$. Do you need to re-run the algorithm? Why or why not?
4. Compute the silhouette score for a point with $a(x)=3$ and $b(x)=9$. Is this point well-clustered?
5. In DBSCAN, MinPts=5 and $\epsilon=2$. A point has exactly 4 other points within distance 2 of it, one
   of which is a core point. What type of point is it?
6. Why can't you just pick the $K$ that minimizes K-Means' distortion $J$ directly?
7. Compute Mahalanobis distance for $x=(3,3)$, $\mu=(0,0)$, with covariance $\Sigma =
   \begin{pmatrix}4&0\\0&1\end{pmatrix}$ (i.e. feature 1 has 4× the variance of feature 2, no
   correlation). Compare to raw Euclidean distance.
8. Name one advantage and one disadvantage DBSCAN has relative to K-Means for streaming data (new
   points arriving continuously).
9. **(Combines two)** A clustering achieves 97% purity against a labeled validation subset but a
   silhouette score of only 0.15. Using the silhouette decomposition ($a$, $b$), explain how this is
   possible without contradiction, and propose one concrete fix if compactness matters for your
   downstream use case.
10. **(Combines two)** Explain why Ward linkage and K-Means, run on the same data with the same $K$,
    often (but not always) produce similar-looking clusters — connecting the mechanism of one to the
    objective of the other.
11. **(Combines two)** A DBSCAN run on high-dimensional data (500+ features, no dimensionality
    reduction) produces one giant cluster containing almost everything. Using the distance-concentration
    result from Dimensionality Reduction, explain the likely mechanism, and connect it to why the
    live demo in this lecture needed PCA for a *different* symptom.
12. **(Combines two)** You need to choose between K-Means and Hierarchical Clustering for a dataset of
    50,000 points where you don't know $K$ in advance and want to explore several candidate values.
    Using both the complexity numbers and the dendrogram's "one run, all $K$" property, make and
    justify a recommendation.

<details>
<summary><b>Answers</b></summary>

**1.** Squared distance is differentiable everywhere (plain Euclidean/absolute distance has a
non-differentiable kink at zero), which matters because the centroid-update step needs to find the
*minimizer* of the sum of distances to all points in a cluster by setting a derivative to zero. That
minimizer is provably the **mean** only for the *squared*-distance objective — differentiating
$\sum\|x-\mu\|^2$ and setting to zero gives $\mu=\text{mean}$; the equivalent for sum of plain
(unsquared) distances gives the **median** (or its multivariate generalization, the geometric median),
which has no closed-form update and requires iterative solving. K-Means uses squared distance
specifically so that "move the centroid to the mean" is exact, not approximate.

**2.** **DBSCAN handles this better in principle, but with an important caveat.** K-Means' single,
shared distortion objective implicitly assumes all clusters have comparable spread — a sparse cluster's
points, being farther from any centroid on average, distort where the K-Means boundary actually falls
(§8). DBSCAN's density-based definition is a better conceptual match for varying density *between*
clusters — but the deck's own limitations list (§22) flags that **DBSCAN itself struggles when density
varies substantially**, because a single global $\epsilon$ cannot be simultaneously correct for both a
tight and a loose cluster. The honest answer: DBSCAN is the better *starting point*, but genuinely
severe density variation may need HDBSCAN (which lets the density threshold adapt locally) rather than
plain DBSCAN.

**3.** **No — this is the entire point of the dendrogram (§13).** One agglomerative clustering run
builds the *complete* merge history; extracting a specific $K$-cluster solution is just a matter of
drawing a horizontal cut line at the appropriate height and reading off which branches it crosses. No
re-computation is needed for any value of $K$ from $1$ to $n$.

**4.** $s(x) = (b-a)/\max(a,b) = (9-3)/9 = 6/9 = \mathbf{0.667}$. This is a strongly positive score
(well above the deck's "watch for silhouette < 0.25" warning threshold), meaning the point's own
cluster is much tighter around it than the nearest alternative cluster is — a well-clustered point.

**5.** **Border point.** It has only 4 neighbours (fewer than MinPts=5, so it fails the core-point
test on its own), but it is within $\epsilon$ of at least one core point — which is exactly the
definition of a border point (§18): "< MinPts neighbours, but within $\epsilon$ of a core point."

**6.** K-Means' distortion $J$ is **monotonically non-increasing in $K$** (D2's proof: any $K$-cluster
solution can be turned into an equally-good $(K{+}1)$-cluster solution by splitting one cluster into
two identical copies, so the true optimum can only get better or stay the same). Minimizing $J$
directly therefore always selects the degenerate $K=n$ (every point its own cluster, $J=0$) — a
completely useless clustering. You need the elbow method's "diminishing returns" criterion instead,
which looks at the *rate of change* of the curve rather than its raw minimum.

**7.**

$$d_E = \sqrt{3^2+3^2} = \sqrt{18} = \mathbf{4.243}$$

$$d_M = \sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)} = \sqrt{\begin{pmatrix}3&3\end{pmatrix}\begin{pmatrix}1/4&0\\0&1\end{pmatrix}\begin{pmatrix}3\\3\end{pmatrix}} = \sqrt{3^2/4 + 3^2} = \sqrt{2.25+9} = \sqrt{11.25} = \mathbf{3.354}$$

Mahalanobis distance is *smaller* than Euclidean here, because the point's displacement along feature 1
(where variance is high, $\sigma^2=4$) is naturally discounted — a displacement of 3 along a
high-variance direction is less surprising, and Mahalanobis correctly treats it as "closer" to the
origin than raw Euclidean distance would.

**8.** **Advantage:** DBSCAN needs no pre-specified $K$ and naturally flags genuinely novel/anomalous
new points as noise rather than forcing them into an ill-fitting existing cluster — potentially useful
as a built-in anomaly signal for streaming data. **Disadvantage:** DBSCAN has no `.predict()` method
(§20) — it is transductive, so scoring each new streaming point technically requires re-running the
full algorithm on the combined dataset, which is completely impractical at streaming scale. K-Means'
learned centroids, by contrast, let you classify any new point with one cheap distance comparison, with
no re-fitting required — a large practical advantage for streaming/online use cases specifically.

**9.** This is not a contradiction — purity/accuracy and silhouette measure different properties
(§29). Decompose silhouette: if $a(i)$ (mean intra-cluster distance) is large relative to the gap
$b(i)-a(i)$, the ratio stays small even for objectively correctly-clustered points, whenever the true
classes are inherently spread out in feature space (e.g. natural variation within a category, like
handwriting styles for one digit). A concrete illustrative case: $a\approx8$, $b\approx10$ gives
$s=(10-8)/10=0.2$ — low, and simultaneously consistent with near-perfect accuracy. **Fix, if
compactness genuinely matters:** reduce dimensionality with PCA before evaluating — removing noise
dimensions that inflate $a(i)$ without adding real inter-cluster separation typically raises silhouette
substantially while leaving accuracy essentially unchanged (§30's demonstrated result: 0.187 → 0.326
going from 64 to 10 PCA dimensions, with accuracy staying at ~93–94% throughout).

**10.** Ward linkage's merge criterion is explicitly: at each step, merge whichever pair of clusters
causes the **smallest increase in total within-cluster variance**. That quantity — within-cluster
variance summed across all clusters — is *exactly* K-Means' objective function $J$ from §5. The two
algorithms are optimizing the same underlying quantity; they just do it via different mechanisms
(Ward: greedy bottom-up merging, locally optimal at each step; K-Means: alternating top-down
assignment/update, also only locally optimal). Since they share an objective, their results often
converge to visually similar cluster shapes — but "often" is doing real work in that sentence, since
neither algorithm's local-optimality guarantee ensures they land in the *same* local optimum, especially
on data where the true clusters violate the shared underlying assumption (spherical, compact) that both
objectives implicitly encode.

**11.** In high dimensions, distances between points concentrate — the relative gap between the
nearest and farthest point shrinks toward zero as dimensionality grows (the Dimensionality Reduction
distance-concentration result). Applied to DBSCAN: a single fixed $\epsilon$ radius that was meant to
distinguish "dense" from "sparse" regions stops being able to do so meaningfully, because at high
enough dimensionality *almost every* pairwise distance looks roughly similar — so either nearly every
point ends up within $\epsilon$ of nearly every other point (one giant merged cluster) or none do
(everything noise), depending on which side of the concentrated distance $\epsilon$ happens to fall.
**The connection to this lecture's own live demo:** §30's PCA fix addressed a *different* symptom of
the identical underlying cause — silhouette score depressed by noise dimensions inflating $a(i)$ without
adding real $b(i)$ separation. Both the giant-DBSCAN-cluster failure and the low-silhouette-K-Means
symptom trace back to the same root problem (too many uninformative dimensions diluting genuine
structure), and both share the same fix: reduce dimensionality (PCA/UMAP) before clustering or
evaluating.

**12.** **Recommendation: use Hierarchical Clustering for the exploration phase, then likely switch to
K-Means for the final production model, if one is needed at scale.** At $n=50{,}000$, hierarchical
clustering's $O(n^3)$ complexity is $\approx1.25\times10^{14}$ operations — right at or somewhat beyond
the deck's own stated practical ceiling of "~10K points," so a naive implementation may already be slow
(though optimized $O(n^2\log n)$ implementations extend this range somewhat). **But the specific
question here — "explore several candidate values of $K$ without knowing it in advance" — is *exactly*
the dendrogram's core value proposition (§13): one run produces every value of $K$ for free**, which is
a genuine advantage over K-Means' requirement to fully re-run for each candidate $K$ separately.
Practical compromise: run hierarchical clustering once (accepting the complexity cost, since it's a
one-time exploratory cost, not a repeated production cost) to visually inspect the dendrogram and settle
on a well-justified $K$, then — if the resulting model needs to run repeatedly in production or scale
beyond 50K points later — switch to K-Means with that now-known $K$, which is far cheaper to re-fit or
to score new points against going forward.

</details>

---

## Going deeper

Ranked by importance. Difficulty markers: `intro` / `solid` / `hard`.

1. **Arthur & Vassilvitskii, "k-means++: The Advantages of Careful Seeding" (SODA 2007)** — `solid`.
   The original K-Means++ paper, with the $O(\log K)$-competitive approximation guarantee proved
   rigorously. Directly cited by §9's slide.
2. **Ester, Kriegel, Sander & Xu, "A Density-Based Algorithm for Discovering Clusters in Large Spatial
   Databases with Noise" (KDD 1996)** — `solid`. The original DBSCAN paper. Short, clearly written, and
   the core/border/noise definitions in §18 are transcribed almost verbatim from it.
3. **Ward, "Hierarchical Grouping to Optimize an Objective Function" (JASA 1963)** — `hard`. The
   original Ward linkage paper. Dense reading, but it's the primary source for the variance-minimization
   criterion §14 derives an intuitive connection to K-Means for.
4. **Rousseeuw, "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis"
   (Journal of Computational and Applied Mathematics, 1987)** — `intro`. The original silhouette-score
   paper. Genuinely readable, and its worked examples are close in spirit to §29's own decomposition.
5. **Campello, Moulavi & Sander, "Density-Based Clustering Based on Hierarchical Density Estimates"
   (PAKDD 2013)** — `hard`. The HDBSCAN paper — the fix for DBSCAN's "one global $\epsilon$" limitation
   named in §22 and §23. Directly relevant if you ever hit varying-density data in practice.
6. **scikit-learn User Guide, "Clustering" (§2.3)** — `intro`, hands-on. Covers K-Means, Hierarchical,
   DBSCAN, and every metric in this lecture with runnable code and clear comparison figures. The
   fastest way to reproduce every worked example in this document yourself.
7. **Kaufman & Rousseeuw, *Finding Groups in Data: An Introduction to Cluster Analysis* (Wiley, 1990)**
   — `hard`. The classical textbook treatment. Dense but comprehensive; covers linkage methods, PAM
   (K-Medoids), and silhouette in far more mathematical depth than any single lecture can.
8. **Vinh, Epps & Bailey, "Information Theoretic Measures for Clusterings Comparison: Variants,
   Properties, Normalization and Correction for Chance" (JMLR 2010)** — `hard`. The rigorous treatment
   of NMI, ARI, and related chance-corrected clustering-comparison metrics — the paper behind §26's
   claims about when NMI is more robust than ARI.
9. **Distill.pub-style interactive: "Visualizing K-Means Clustering" (naftaliharris.com/blog/visualizing-k-means-clustering)**
   — `intro`, hands-on. A free, browser-based interactive that lets you place points and watch K-Means
   iterate step by step — the fastest way to build intuition for §5.2's convergence proof by watching it
   happen rather than reading it.
10. **Ankerst, Breunig, Kriegel & Sander, "OPTICS: Ordering Points To Identify the Clustering Structure"
    (SIGMOD 1999)** — `hard`. A generalization of DBSCAN that produces a reachability plot instead of a
    single hard clustering, letting you explore multiple density thresholds from one run — the
    density-based analogue of what a dendrogram gives hierarchical clustering. Good context for why
    HDBSCAN and OPTICS both exist as answers to the same underlying limitation.

---

## 📊 Summary

| | |
|---|---|
| **Source** | `output/Lecture_10 - Module 4 Unsupervised Learning Part 1` — 175 raw frames, ~122 distinct slide states plus an extensive live-coded Jupyter demo |
| **Runtime** | 1:02:54 · instructor not named in the recording |
| **Sections** | 30, across five parts (distance & families §1–§4 · K-Means §5–§10 · Hierarchical §11–§16 · DBSCAN §17–§23 · evaluation & live demo §24–§30) |
| **Worked examples** | 6, every one carried to a final number, plus a **complete, real Jupyter notebook walkthrough reproduced with its actual printed output** (901 real digit images, real purity/silhouette/PCA numbers) |
| **Derivations** | K-Means' convergence proof (both alternating steps non-increasing on $J$) · why distortion is monotonic in $K$ (hence the elbow method is necessary, not optional) · Mahalanobis = Euclidean after whitening, derived in one line · Ward linkage = K-Means' objective applied incrementally · $O(n^3)$ hierarchical-clustering complexity derived from first principles · the silhouette decomposition ($a(i)$, $b(i)$) computed on real data |
| **Interactive blocks** | 4 (§5/§6 the K-Means assign/update loop, §9 K-Means++ seed selection, §13 cutting the dendrogram, §19 DBSCAN's core-point expansion) — beyond these, the deck's own richest "interactivity" is the live-coded demo itself, reproduced in full in §28–§30 |
| **Interview questions** | 9 with model answers (2 combining concepts), 8 depth probes, 3 whiteboard derivations, 1 applied scenario (fraud-ring detection), 4 Leadership Principles |
| **Cross-references** | To [Dimensionality Reduction Part 1](../Dimensionality%20Reduction/dimensionality-reduction-01.md) (distance concentration, mutual information) and [Part 2](../Dimensionality%20Reduction/dimensionality-reduction-02.md) (t-SNE's local-structure-only guarantee, PCA as a pre-clustering step) |
| **⚠️ Flags left in the file** | GMM is previewed (§24) but **not covered** — it is Part 2's subject, and this document does not claim otherwise · the instructor is unnamed in the recording · the opening sentence quoted from slide 8 (Prerequisite 1) has its subject clipped by the raw capture's frame edge and is transcribed only from "lives inside..." onward, flagged inline · §25's three worked silhouette values ($s=0.83/0.40/0.45$) are transcribed exactly, with the genuinely counter-intuitive ordering (farther-from-cluster scoring higher than on-boundary) explained rather than smoothed over |
