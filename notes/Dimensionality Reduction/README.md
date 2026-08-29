# Dimensionality Reduction — Amazon ML Summer School

Self-study notes built from the verified slide extraction in
[`output/`](../../output/), following [`NOTES_PIPELINE.md`](../../NOTES_PIPELINE.md).

These are **teaching documents, not summaries**. They define every term before using it, derive
every formula rather than asserting it, and work every example through to a final number. They are
written to be the only thing you need to read to master the lecture.

> ✅ This module has completed a full [`QUALITY_REVIEW_PIPELINE.md`](../../QUALITY_REVIEW_PIPELINE.md)
> pass — see [`QUALITY_REVIEW.md`](QUALITY_REVIEW.md) for the audit trail of what was checked and
> fixed. The word counts, glossary/count figures, and capture-quality notes below reflect the post-fix
> state.

---

## Index

| # | Notes | Source deck | Status | Words | Covers |
|---|---|---|---|---|---|
| 01 | [dimensionality-reduction-01.md](dimensionality-reduction-01.md) | `Lecture_07 - Module 3 Dimensionality Reduction Part 1` (**40 slides**, 7 sections) | ✅ Complete · 2 quiz answers reconstructed | ~35,600 | Curse of dimensionality · why reduce · selection vs extraction · wrapper / filter / embedded methods · the linear algebra behind PCA |
| 02 | [dimensionality-reduction-02.md](dimensionality-reduction-02.md) | `Lecture_08 - Module 3 Dimensionality Reduction Part 2` (**35 slides**, 4 topics) | ✅ Complete · no capture gaps | ~31,100 | SVD · Eckart–Young · power iteration · PCA (centering, eigenfaces, kernel PCA) · matrix factorization · NMF · KL divergence · t-SNE · UMAP |
| 03 | [dimensionality-reduction-03.md](dimensionality-reduction-03.md) | `Lecture_09 - Module 3 Dimensionality Reduction Part 3` (**47 slides**, 4 parts) | ✅ Complete · no content gaps | ~26,400 | Autoencoders · denoising/sparse AE · VAE · VQ-VAE · latent diffusion · LoRA · DoRA · Matryoshka Representation Learning |

✅ Lecture 07's forward reference (*"Up next: PCA, SVD, and Beyond — Section 2 covers the algorithms,
kernels, and probabilistic extensions"*) held up for Lecture 08. Its guess for "Part 3" —
"nonlinear methods — t-SNE, UMAP, autoencoders" — turned out **wrong**: Lecture 08 already
delivered t-SNE/UMAP, and Lecture 09 is a different subject entirely ("Dimensionality Reduction in
the Age of LLMs & Generative AI") sharing only the autoencoder thread with the prediction. This is
the second confirmed case in this course of a lecture's own forward-reference slide being an
unreliable guide to what the *next* lecture actually covers — verify against the deck, don't plan
notes around a prior lecture's roadmap slide.

**Prerequisites.** This module leans on [`Supervised Learning`](../Supervised%20Learning/) for
cross-validation and the bias–variance framing, and on
[`Deep Neural Networks`](../Deep%20Neural%20Networks/) for L1-vs-L2 regularization — Part 1 §21–§23
re-derives Ridge and Lasso *geometrically* (constraint regions), where
[DNN Part 2 §10](../Deep%20Neural%20Networks/deep-neural-networks-02.md) derived them *analytically*
(gradients). Reading both gives you two independent arguments for the same result, which is exactly
what you want under interview pressure.

---

## Capture quality

### ✅ Lecture 07 — excellent

94 raw frames over 57 minutes. Every content slide has a fully-built state; every equation and
citation is legible. This deck is unusual in **citing its own sources on nearly every slide**
(Bellman 1961, Beyer 1999, Aggarwal 2001, Tenenbaum 2000, Pearson 1895 and 1901, Shannon 1948,
Kraskov 2004, Hoerl & Kennard 1970, Tibshirani 1996, Zou & Hastie 2005, Hotelling 1933), which makes
the notes' *Going deeper* section verifiable rather than reconstructed.

**Two small gaps, both the same kind.** The deck has three `Quick check` quiz slides. The third's
answer **was** captured as an on-click reveal at 46:04 (*"Answer: C"*); the first two's were not —
they were most likely revealed during a sampling gap. Both have answers fully determined by the
deck's own content, so §5 and §11 give them under a **🩹 badge** meaning *"answered from the deck's
own slides, not read off an answer line."*

**The instructor is not named** anywhere in the recording — the webcam tile carries no label — so
this module, unlike the others, cannot attribute its lecture.

### ✅ Lecture 08 — excellent

90 raw frames over 45 minutes. Every content slide has a fully-built state; every equation, figure
caption and citation is legible. Like Lecture 07's deck, this one **cites its own sources on nearly
every slide** (Koren–Bell–Volinsky 2009, Cattell 1966, Marchenko–Pastur 1967, Turk & Pentland 1991,
Schölkopf et al. 1998, Lee & Seung 1999/2000, Hofmann 1999, van der Maaten & Hinton 2008, McInnes
et al. 2018, Bishop PRML). **No content gaps.**

Three of the slides are live interactive demos (the SVD rank slider, the PCA axis rotator, the t-SNE
perplexity sweep); the capture caught each at multiple settings, so the notes reproduce real numbers
from two or more states of each rather than describing them.

⚠️ **This deck transposes the data matrix relative to Lecture 07** — features × samples rather than
samples × features — so the principal components are the columns of $U$ here, not $V$. The notes
flag this prominently in the front matter and at every crossing point.

The instructor is **Charul Paliwal** — named in the webcam tile, unlike Lecture 07's unattributed
deck.

### ✅ Lecture 09 — excellent, no content gaps

108 raw frames over 56 minutes. Every content slide has a fully-built state; every equation and
citation is legible, and — like both earlier decks in this module — this one cites its own
sources on nearly every slide (Weng 2018, van den Oord et al. 2017, Hu et al. 2022, Liu et al. 2024,
Rege 2024).

**One long single-slide stretch, explained rather than a gap.** Frames 77–98 (48:59–50:41, about 1
minute 42 seconds) are all the *same* content slide on Matryoshka Representation Learning — the
change-detector kept re-sampling on small cursor/webcam movements while the instructor spoke at
length over one static slide. Nothing is missing; the slide's five bullets are the complete written
content, and the notes teach them in full depth to compensate for the absent transcript.

Three interactive demos were captured at multiple slider settings and are reproduced with real
numbers rather than described: the linear-autoencoder-as-rank-$k$-PCA slider, the denoising
autoencoder's noise slider, and the two VQ-VAE/latent-diffusion quiz slides' worked answers.

The instructor is **Ravi Sankar Adepu** — named in the webcam tile throughout.

---

## What's in Part 1

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — the seven-section map with runtimes · capture notes · 16 capabilities · 6
prerequisites taught from zero (the $n$-vs-$p$ notation and the $i$-vs-$j$ trap · variance vs
covariance vs correlation, and why only the last is comparable · **a big-O table showing $2^{100} =
10^{30}$**, which is why feature selection is a research area · entropy in bits · the $X^\top X$
shape check · the two norms, with the diamond-vs-ball preview that is the whole of §21–§23) · a
full-lecture ASCII map

**ACT I — The curse of dimensionality** (§1–§5)
- §1 **The ball-in-cube ratio verified at four dimensions** — 78.5% → 52.4% → 0.249% → 0.0000025% ·
  📚 why $\Gamma$ beating $\pi^{d/2}$ means the unit ball's volume goes to **zero** · a second,
  sharper argument: at $d{=}100$, **99.997% of a cube is within 5% of its surface** · 💡 and that's
  $0.9^{100}$ — the same arithmetic that killed the RNN gradient in DNN Part 3
- §2 **Distance concentration derived from first principles** — $\mathbb{E}[D] = 0.408\sqrt d$ and
  $\mathrm{std}(D) = \mathbf{0.2415}$, *constant*, because the $\sqrt d$ cancels · so relative
  contrast is $0.592/\sqrt d$ · a table from $d{=}2$ to $d{=}1000$ where the middle column never
  changes · ⚠️ and the caveat that makes the deck's "practical test" the right framing:
  **correlated features don't count as separate dimensions**
- §3 **1,000 samples in 20 dimensions occupy $10^{-17}$ of the space** — one grain of sand to the
  Sahara · inverted: $n{=}1000$ densely supports **three** features · 💡 hence *there is no such
  thing as an assumption-free method in high dimensions*
- §4 The five failures, each with its mechanism · 📚 **why $p \gg n$ *guarantees* a perfect fit**,
  and therefore why training accuracy carries no information at all in that regime
- §5 🩹 The kNN quiz, with **why each wrong answer is wrong** — "nonparametric and flexible" is the
  reason it fails, not a defence; increasing $k$ reduces the variance of an estimate of the wrong
  quantity

**ACT II — Why, and how to select** (§6–§26)
- §6–§8 Three goals, and why they're three genuinely different constraints (human / resource /
  statistical) · 💡 **noise reduction is the counter-intuitive one** — throwing information away
  makes the model *better* · §7 the **manifold hypothesis**, with the generative argument (100K
  pixels, ~50 knobs) and ⚠️ **three ways it fails**, including PCA's silent low-variance-signal
  trap · ⚠️ a caveat on the deck's interpretability claim, which two of its own slides contradict
- §9–§11 The five-axis comparison, with §9.1 pointing out that **the discrete-vs-continuous row is
  why the module is organised as it is** · the taxonomy tree with every leaf mapped to a section ·
  💡 **the one distinction that organises everything: how much does the method know about the
  model?** · §11 🩹 the Pearson-vs-MI quiz, answered with the parabola computed to $r = 0$ exactly
- §12–§14 **Wrappers.** ⚠️ the *third* cost the slide omits — they overfit the selection, so you
  need nested CV · §13 **19,810 model fits** for $p{=}1000{\to}20$, versus $2^{1000}$ for exhaustive
  search · **the XOR construction worked out in full**, showing why greedy addition cannot find a
  pair whose parts are individually worthless · §14 **RFE is 202× cheaper**, and it's honestly a
  wrapper loop around an embedded ranking
- §15–§19 **Filters.** 💡 fast, model-agnostic and interaction-blind are **one property seen three
  ways** · ⚠️ **§15.3's leakage trap**, with the demonstration that 10,000 noise features and random
  labels produce above-chance accuracy · §16 the scale caveat worked in units — `VarianceThreshold`
  deletes a feature for being measured in metres · §17 **Pearson = 0 on a parabola, exactly**, with
  *why* the cancellation happens · Spearman as the cheap upgrade nobody names · §18 **MI computed by
  hand to 1.522 bits** on the same data · ⚠️ **the correction that matters: univariate MI does *not*
  find XOR** — MI removes the linearity assumption, not the univariate one · §19 **a χ² test worked
  to 19.20** with the small-expected-count caveat, plus a decision tree for choosing a filter
- §20–§26 **Embedded.** §20.1 💡 **L1 as the convex relaxation of an NP-hard $\ell_0$ problem** —
  the reason Lasso was a landmark · §21 why $(X^\top X + \lambda I)$ is *always* invertible, and
  **"Ridge divides, Lasso subtracts-and-clips"** · §22 the geometric argument done properly (a
  vertex has a *fan* of supporting lines; a $k$-face means exactly $k$ features) and **three
  explanations tabulated by which audience each suits** · §23 **the grouping effect proved in three
  lines** — $(c,0)$ costs $c^2$, $(c/2,c/2)$ costs $c^2/2$ · 💡 **Lasso can select at most $n$
  features**, a hard cap · §25 the death-order as a free importance ranking, and why LARS makes the
  whole path cost one fit · §26 ⚠️ **add a UUID column and Gini importance often ranks it near the
  top** — and what that implies about the *rest* of the ranking

**ACT III — The linear algebra behind PCA** (§27–§31)
- §27 ⚠️ "for centered $X$" is load-bearing, not a footnote · **$v^\top\Sigma v = \frac1n\|Xv\|^2$
  proved in two lines** — which simultaneously gives PSD, gives "eigenvalue = variance", and turns
  PCA into an eigenproblem · the data-ellipse table where **a tilted ellipse is the entire
  motivation for extraction over selection** · a full worked example: $\Sigma$, both eigenvalues,
  both eigenvectors, and the projection's variance verified to equal $\lambda_1$
- §28 💡 **the trace identity is what makes "explained variance ratio" honest** rather than
  double-counting · ⚠️ `np.linalg.eigh` returns *ascending* eigenvalues
- §29 $\Sigma = V\Lambda V^\top$ read right-to-left as **rotate · scale · rotate back** ·
  **the SVD connection derived in one line**, and the three reasons sklearn uses it (40 GB, squared
  condition number, cheap truncation) · ⚠️ `TruncatedSVD` ≠ `PCA`
- §30 **both selection rules worked on the same spectrum, and they disagree** (95% says $k{=}4$, the
  elbow says $k{=}3$) — which is the point · ⚠️ a high explained-variance ratio is **not** evidence
  you kept the useful part
- §31 **the two objectives are two halves of one fixed total**, which is why Pearson (1901) and
  Hotelling (1933) describe the same computation from opposite ends

**Closing** — a full-lecture ASCII dependency map · 💡 the thread: *high dimensions are only a
problem when they're real* · **12 interview questions** with model answers (3 combining concepts) ·
9 depth probes · **3 whiteboard derivations** · a delivery-time-prediction scenario including the
point that **PCA reduces model width, not data-collection cost** · 4 Leadership Principles ·
**46-term glossary** · 12 check-yourself questions · 18 ranked resources, 9 of them the deck's own
citations

**Interactive specs:** 4 blocks — the ball-in-cube ratio as $d$ grows · the distance histogram
sliding right without widening · circle-versus-diamond with a draggable OLS point · the data ellipse
and its eigenvectors.

</details>

---

## What's in Part 2

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — the four-topic roadmap with runtimes · ⚠️ **this deck transposes the data matrix
relative to Part 1** (features × samples, so the PCs are $U$'s columns rather than $V$'s — flagged
before it can bite) · 6 prerequisites (orthonormal matrices preserve length, hence total variance ·
rank-1 outer products as the entire economic argument for the SVD · the Frobenius-norm identity that
makes Eckart–Young checkable · KL prerequisites) · a full-lecture ASCII map

**PART 1 — SVD** (§1–§7)
- §1 The Netflix motivation, verified: **8.5 billion entries, 1.18% observed**, winning model used
  $k$ = 50–200 · 💡 the three questions the SVD answers are one question asked three ways
- §2 **Both key identities derived** — $X^\top X = V\Sigma^2V^\top$ and $XX^\top = U\Sigma^2U^\top$ —
  in two lines each, plus 💡 **three consequences**: $\sigma_i = \sqrt{\lambda_i}$, you never need to
  form $X^\top X$, and singular values can never be negative (why SVD exists for every matrix and
  eigendecomposition doesn't) · 📚 Eckart–Young stated, since the roadmap names it but no slide does
- §3 **Power iteration proved in three lines** and then **verified by hand** — the angle to the true
  eigenvector halves every step, exactly matching the predicted $\lambda_2/\lambda_1 = 0.5$ rate ·
  ⚠️ why naive deflation is a teaching algorithm, not a production one
- §4 **Truncation break-even derived**: $k^* = mn/(m+n+1)$ · the Netflix case at $k{=}50$: **343×**
  reduction that *also* fills in 98.8% missing data
- §5 The two-panel spectrum comparison — butterfly vs random noise — read as the empirical test of
  the manifold hypothesis · 📚 **Marchenko–Pastur**, upgrading the scree-plot elbow from heuristic to
  a principled noise threshold
- §6 🧪 **The deck's own live demo verified against Eckart–Young to three decimal places** — $k{=}5$:
  1,285 numbers, 0.304 relative error; $k{=}20$: 5,140 numbers, 0.146 error; both checked against
  $1 - \text{cumulative }\sigma^2$ and matching exactly
- §7 LSI: why co-occurring words end up close, worked · 💡 **LSI is a word embedding, 25 years before
  word2vec** — with the Levy & Goldberg equivalence cited to make the lineage rigorous rather than a
  rhetorical flourish

**PART 2 — PCA** (§8–§15)
- §8 **Why uncentred PCA points at the mean, derived and demonstrated** on three points where the
  correct PC1 is $(1,0)$ but the uncentred answer comes back at 45° · ⚠️ `TruncatedSVD` ≠ `PCA`
- §9 🧪 **The rotation demo's total variance verified invariant to the decimal** — 5947.5 at both
  captured angles — turned into 💡 **the reframing that matters**: PCA doesn't create variance, it
  redistributes a fixed total · ⚠️ PCA minimises *perpendicular* distance, not vertical — a genuinely
  different line from OLS
- §10 🧪 Iris worked: **72.8% + 23.0% = 95.8%**, and the standardisation swing to 92.5% dominated by
  petal length quantified · ⚠️ a first-decimal disagreement between the slide's text and its own plot,
  flagged rather than silently resolved
- §11 **Why PC1–PC3 are illumination, not identity** — the general mechanism, not just a fact about
  faces — with the counter-intuitive consequence: discard the leading components on purpose
- §12 **The denoiser argument quantified**: 97.6% of noise removed against ~10% of signal, ⇒ ~37×
  SNR improvement — not asserted, computed
- §13 **PCA is a linear autoencoder with tied weights — a theorem** (Baldi & Hornik 1989), not an
  analogy, stated and used to explain precisely what a deep autoencoder adds (nonlinearity, nothing
  else)
- §14 **Ambient vs geodesic distance**, with the London–Sydney analogy, as the one fault line that
  explains an entire literature (Isomap, LLE, t-SNE, UMAP, kernel PCA)
- §15 **The kernel trick derived from a concrete $\varphi$** (the degree-2 polynomial expansion,
  worked in full) so it stops looking like magic · ⚠️ **the cost inversion**: kernel PCA scales with
  *samples*, not features, plus the pre-image problem that rules it out for denoising round-trips

**PART 3 — Matrix & Non-Negative Matrix Factorization** (§16–§20)
- §16 💡 **Why MF is not truncated SVD** — fit only observed cells, forfeit Eckart–Young, need
  regularization because $p \gg n$ · 📚 the **biased predictor** ($\mu + b_u + b_i$) as the cheapest
  real win, cited to Koren–Bell–Volinsky
- §17 **Multiplicative updates explained** — why they enforce non-negativity by construction rather
  than by policing — and ⚠️ why that makes NMF **non-unique**, unlike PCA
- §18 **The mechanism of "parts not wholes" derived**, then demonstrated in miniature: rebuilding the
  digit 7 as $8-1$ (signed) versus $5+2$ (non-negative) — the entire Lee & Seung finding in four
  numbers
- §19 **Why a "topic" is what a rank-constrained non-negative factorisation of TF-IDF looks like** —
  explained as emergent, not defined · the LSI→NMF→pLSA→LDA lineage traced through one sklearn
  footnote
- §20 🧪 Six real topics from Amazon reviews read closely — four clean product categories, one
  cross-cutting value complaint, and two overlapping book topics used to explain **how you actually
  choose $r$ for NMF** (there's no scree plot; you read the topics)

**PART 4 — Nonlinear visualisation: KL, t-SNE, UMAP** (§21–§28)
- §21 💡 the reframing that makes the whole section necessary: "preserve distances" → "preserve
  neighbourhood probabilities," which is why a DR lecture spends three slides on information theory
- §22 **The crowding problem explained with a volume argument**, then the Gaussian-vs-Student-t decay
  tabulated (a ratio of $10^{41}$× at distance 10) · ⚠️ **t-SNE has no `transform`** and why that's
  disqualifying for production
- §23 **$H(P,Q) = H(P) + \mathrm{KL}(P\|Q)$ derived in three lines**, then made concrete with an actual
  4-message coding example (1.75 bits optimal, 2.0 bits with the wrong code, 0.25 bits wasted) ·
  **Gibbs' inequality proved via Jensen** · ⚠️ why "KL distance" is a phrase to avoid, with the two
  metric axioms it fails named explicitly
- §24 **The best slide in the lecture, unpacked**: forward-vs-reverse KL derived from *which
  distribution the expectation is under* — not asserted, shown term by term why one is zero-avoiding
  and the other zero-forcing — then connected to why t-SNE preserves local structure and nothing else
- §25 **Five real uses of KL cross-referenced back into the rest of the course** — VAEs (reverse,
  hence posterior collapse), distillation (with "dark knowledge" explained), PPO/TRPO, and 💡 mutual
  information *is* KL of the joint against the independence assumption, reconnecting to
  [Part 1 §18](dimensionality-reduction-01.md) explicitly
- §26 **Perplexity $=2^H$ derived to be exactly $k$ for a uniform-over-$k$ distribution** · 🧪 the
  deck's own perp-5 vs perp-80 demo compared · ⚠️ **the single most important warning in Part 4**:
  low-perplexity t-SNE invents clusters from pure noise (Wattenberg et al. 2016), with four things you
  must never conclude from a t-SNE plot
- §27 **UMAP's loss decomposed against t-SNE's** — the repulsive term is the whole difference, and is
  why UMAP keeps more global structure · ⚠️ its theoretical framing flagged as contested (Böhm et al.
  2022) rather than accepted at face value
- §28 The three-method digits comparison, with the 4/7/9 colocation read as UMAP telling you something
  true rather than an artefact · 💡 the closing rule: **PCA when the output feeds a model, t-SNE/UMAP
  when it feeds a human** — and the standard PCA→50D→t-SNE/UMAP pipeline justified by both speed and
  [Part 1 §2's](dimensionality-reduction-01.md) distance-concentration argument

**Closing** — a full-lecture ASCII dependency map tracing "SVD plus one constraint" through all four
parts · 💡 three threads (every constraint costs a guarantee · the largest variance is often a
nuisance · interpretability is a constraint you pay for) · **12 interview questions** with model
answers (3 combining concepts) · 10 depth probes · **3 whiteboard derivations** · a visual-search
retrieval-index scenario distinguishing the *serving* representation from the *analysis* one · 4
Leadership Principles · **39-term glossary** · 12 check-yourself questions · 19 ranked resources, 10
of them the deck's own citations.

**Interactive specs:** 6 blocks — rebuilding a matrix rank-1 piece at a time · the deck's own live
rank-slider demo on the butterfly image · rotating PCA's axes while total variance stays pinned ·
rebuilding a face with and without minus signs · fitting one Gaussian to two peaks in each KL
direction · a perplexity sweep on real data and on pure noise.

</details>

---

## What's in Part 3

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — ⚠️ **this is not the t-SNE/UMAP lecture Part 1 predicted, and not a repeat of Part
2's content** — resolved up front rather than left implicit; the deck's real title is
*"Dimensionality Reduction in the Age of LLMs & Generative AI"* · the four-part roadmap with runtimes ·
6 prerequisites (the bottleneck and KL divergence recalled explicitly from Part 2 · the
reparameterisation problem and straight-through estimation taught in miniature *before* they're needed
· rank-1 updates recalled as the bridge to LoRA) · a full-lecture ASCII map making the whole lecture's
argument explicit: every method here is either Part 1's bottleneck or Part 2's low-rank factorisation,
redeployed inside a system trained end-to-end

**PART 1 — Autoencoders & Variational Autoencoders** (§5–§28)
- §5–§9 The bottleneck as forced dimensionality reduction · 💡 **an autoencoder is nonlinear PCA** —
  stated immediately as the connection to make, not left implicit · 🧪 **the deck's own rank-slider
  demo verified**: $k{=}4$ gives error 0.131, $k{=}16$ and $k{=}18$ both give exactly 0.000 · 🎯 the
  Baldi–Hornik quiz, with **why each wrong answer is wrong** explained rather than just marked incorrect
- §12–§14 **Denoising.** Why the identity-function shortcut is closed off by comparing the loss target
  to the *original*, never-seen input · 👉 the same capacity-vs-memorisation overfitting story as
  [DNN Part 2 §9](../Deep%20Neural%20Networks/deep-neural-networks-02.md), reapplied to a bottleneck ·
  🧪 the noise-slider demo at two states (0.40: 0.404→0.291; 0.14: essentially flat) with an honest
  caveat that they aren't a controlled paired comparison
- §16–§18 **Sparse.** 🧪 **the KL sparsity penalty's endpoints derived** — proved to diverge to
  $+\infty$ at $\hat\rho\to0$ and $\hat\rho\to1$, explaining why it's a "soft wall" rather than a mild
  nudge · 📚 the bridge to **LLM interpretability** — polysemanticity, superposition, and
  monosemantic features, flagged 🔬 as a genuinely open research direction rather than settled
- §19–§25 **VAE.** Why a plain AE's latent space develops holes, argued from what the loss never
  evaluates, not asserted · 🧪 **the reparameterisation trick fully derived** — distributional
  correctness verified, both partial derivatives computed and their difference explained · the ELBO's
  tension named precisely · ⚠️ **the KL term identified as *reverse* KL**, connecting directly to
  [Part 2 §24](dimensionality-reduction-02.md) to *predict* posterior collapse rather than just report it
- §26–§28 **VQ-VAE.** The three-term loss unpacked term by term via stop-gradient placement · the
  straight-through estimator named explicitly as a biased approximation, not an exact gradient ·
  💡 why discrete latents are exactly a Transformer's native input format — the DALL·E/Jukebox
  connection made mechanistic, not just asserted

**PART 2 — Latent Diffusion Models** (§29–§30)
- §29 📚 diffusion taught from zero (forward/reverse process) before the cost argument · 🧪 **both
  costs computed exactly** — 786,432 values per step, and $6.87\times10^{10}$ pairwise attention
  scores — with the linear-vs-quadratic distinction flagged as the reason the two costs respond
  differently to compression
- §30 The two-stage training made explicit (train the VAE first and freeze it; run diffusion entirely
  inside its latent space) · 🎯 the quiz computing the exact 64×/4,096× saving · 💡 the payoff line
  connected directly back to §23's KL term — **latent diffusion only works because the VAE's latent
  space has no holes**
- §30.1 Video diffusion as the same idea with one more axis (spatiotemporal VAE, DiT), with the pixel
  count independently verified against the deck's own figure

**PART 3 — Low-Rank Methods in Transformers** (§31–§35)
- §31–§32 💡 **LoRA's key insight connected explicitly to Part 2's SVD** — the same low-rank
  hypothesis, transplanted from data matrices to a weight *update* · 🧪 parameter savings computed
  twice at different scales (256× at both $d{=}4096,r{=}8$ and $d{=}8192,r{=}16$) · the $B=0$
  initialisation derived from the forward pass, not just stated · 🎯 the LoRA quiz, with the correct
  answer shown to fold the wrong answer's premise into itself
- §35 **DoRA.** The magnitude/direction split derived as an algebraic identity, then used to show
  precisely what a single low-rank matrix conflates that DoRA separates · ⚠️ the "beats LoRA" result
  flagged as an empirical finding on tested benchmarks, not a proven guarantee · the LoRA→QLoRA→DoRA
  lineage clarified as three orthogonal, combinable fixes rather than competing alternatives

**PART 4 — Embedding Models & Matryoshka Representations** (§36–§39)
- §37 🧪 the 1.2TB storage figure verified, and **five distinct failure modes of a fixed-size
  embedding disentangled** rather than treated as one complaint — cost, quality, maintenance,
  inconsistency, and what's actually wanted
- §38 🧪 **the nested loss derived to show *why* it produces coarse-to-fine packing**, not just
  described · ⚠️ what "interpolates smoothly" does and doesn't guarantee about untrained sizes ·
  💡 **the precise mechanism-level comparison against post-hoc PCA**, using Part 2's own
  variance-vs-usefulness distinction rather than a generic "MRL is better" claim · 👉 see also
  [GenAI & LLM Part 3 §13](../GenAI%20&%20LLM/genai-llm-03.md) for the same method from the applied
  side (production truncation, MTEB curve, a worked retrieval example)
- §39 🧪 both headline numbers reframed as error-rate comparisons (MRL's BEIR shortfall is less than
  half the independent model's) · the 2-stage retrieval pattern connected back to §37's inconsistency
  problem as the reason it's even possible

**Closing** — a full-lecture ASCII map tracing the bottleneck and the low-rank factorisation through
all four parts as one argument · 💡 the single thread — every technique here is classified by what it
gives up to get a smaller representation · **12 interview questions** with model answers (3 combining
concepts, including one tracing a single idea — "the largest source of variation isn't always what
you care about" — through three separate sections and back to Part 2) · 8 depth probes · **3
whiteboard derivations** · an applied scenario building a production/analysis image-search split,
explicitly ruling out three plausible-but-wrong designs with reasons · 3 Leadership Principles ·
**23-term glossary** · 12 check-yourself questions · 17 ranked resources, 9 of them the deck's own
citations.

**Interactive specs:** 5 blocks — the linear-AE-as-rank-k-PCA slider · the denoising-AE noise slider ·
point vs distribution encoding · LoRA rank vs parameter count · truncating the Matryoshka dolls.

</details>

---

## Reading guide

The three parts are ~93,000 words together. Part 2 assumes Part 1's linear algebra throughout —
especially the $X^\top X$ shape rule and the eigenvalue-is-variance argument (Part 1 §27), which Part
2 §2 generalises to the SVD without re-deriving. Part 3 assumes Part 2 even more heavily — its entire
argument is "Part 1's bottleneck and Part 2's low-rank factorisation, redeployed inside a trained
system," and it says so explicitly rather than leaving the connection implicit.

**First pass (~10 hours across all three).** Read linearly. Do not skip *Before we start* in any
file — Part 1 §21–§23 need the two norms, Part 1 §27 needs the $X^\top X$ shape rule, Part 2 §2–§3
need orthonormal matrices and rank-1 outer products, and Part 3's prerequisites explicitly recall
Part 2's bottleneck theorem and KL divergence before building on them. Skip the `interactive` spec
blocks entirely.

**Second pass.** Work every 🧪 example on paper *before* reading the solution. The seven that matter
most, in order: Part 1 §2 (distance concentration), Part 1 §17.2 (Pearson = 0 on a parabola), Part 1
§23 (the grouping-effect proof), Part 2 §6 (the rank-slider demo verified against Eckart–Young), Part
2 §12 (the denoiser's SNR argument), Part 3 §22 (the reparameterisation trick, fully derived), and
Part 3 §38 (the Matryoshka nested loss, derived to show *why* it orders dimensions, not just recited).

**Before an interview.** Each file's *Putting it together*, then all 9 whiteboard derivations across
the three files, then the depth-probe tables. If you have time for only four derivations: **distance
concentration** (Part 1 §2), **why the eigenvalue is the variance** (Part 1 §27.2), **forward vs
reverse KL** (Part 2 §24), and **the reparameterisation trick** (Part 3 §22) — those four carry most
of the module's interview surface, and the fourth is the one most likely to come up if the role
touches generative modelling at all.

**The four questions this module is most likely to be examined on:** *"why does kNN fail in high
dimensions?"* (Part 1 §2 — derive $0.59/\sqrt d$, don't just recite "curse of dimensionality"),
*"why does L1 give sparsity and L2 doesn't?"* (Part 1 §21–§22 — give more than one of the three
arguments), *"why is KL asymmetric, and does it matter?"* (Part 2 §24 — derive it from which
distribution the expectation is under, then connect it to t-SNE's known failure mode), and *"how
does LoRA work, and why does B start at zero?"* (Part 3 §31–§33 — connect it explicitly to Part 2's
low-rank hypothesis, then derive the zero-disruption property from the forward pass).

**When something in your own pipeline looks too good.** Part 1 §15.3 first — check whether any
selection step was fit outside the cross-validation loop; permute the labels and rerun. If it's a
visualisation rather than a model, Part 2 §26.2–§26.3 next — sweep perplexity and seeds before
believing any cluster you see.

**Callout legend**

| | Meaning |
|---|---|
| 📚 **Background** | A prerequisite the slide assumed you already knew |
| 💡 **Key insight** | The one sentence from that section worth memorising |
| ⚠️ **Careful** | A slide gap, an ambiguity, or a place the standard presentation misleads |
| 🧪 **Worked example** | Real numbers computed to a real answer |
| 🎯 **Interview** | Phrased the way you would actually say it out loud |
| 🔬 **Research opportunity** | Where the field is genuinely still open |
| 🩹 **Reconstructed** | Content the capture missed; taught from the standard result, and said so |

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

---

## Key takeaway per lecture

| # | One-sentence takeaway |
|---|---|
| 01 | High-dimensional space is pathological in three measurable ways that are really one — volume grows exponentially in $d$ while your sample size does not, so a ball fills 0.25% of its cube by ten dimensions, pairwise distances have a *constant* spread of 0.24 around a mean that marches off as $0.41\sqrt d$ (making "nearest neighbour" meaningless), and a thousand samples occupy $10^{-17}$ of a twenty-dimensional grid; the escape is the manifold hypothesis — that data generated by a process with few degrees of freedom lies on a low-dimensional surface however many numbers you use to record it — and it is an **assumption that can fail silently**, most dangerously when the signal lives in a low-variance direction that unsupervised PCA discards while reporting 95% variance explained; given the assumption holds, you either **select** columns (filters know nothing about your model and cost nothing, wrappers know its score and cost thousands of fits, embedded methods *are* the model and cost one — you pay for knowledge in compute) or you **extract** directions, and the whole of extraction reduces to the fact that $v^\top\Sigma v$ is the variance along $v$, so the eigenvectors of the covariance matrix are the data cloud's principal axes and its eigenvalues are the variance along each. |
| 02 | Every feature-extraction method in this module is the SVD plus one added assumption, and each assumption buys a capability at the cost of a guarantee: centre the data and you get PCA — directions of maximum variance, provably optimal by Eckart–Young, but only because total variance is invariant under rotation, so PCA is choosing how to *concentrate* a fixed budget rather than creating anything (verified live: 5947.5 both before and after rotating onto the best axis); allow only some entries to be observed and you get matrix factorization — no longer a closed-form problem, forfeiting that same optimality guarantee, which is exactly why "just run SVD on the ratings matrix" is a real error; forbid subtraction and you get NMF — unable to cancel, so it is forced into localised, interpretable **parts** rather than PCA's signed, global **wholes**; and abandon linearity for curved manifolds and you get t-SNE and UMAP, which replace "preserve distances" with "preserve neighbourhood probabilities" and are optimised by minimising KL divergence — a choice whose direction is not incidental, since forward KL is provably mode-covering (why t-SNE protects *local* structure) while reverse KL is mode-seeking (why VAEs drop modes), and knowing that asymmetry is the difference between reciting "inter-cluster distances in a t-SNE plot are unreliable" and being able to derive why. |
| 03 | This lecture introduces no new mathematics of its own — every method in it is either Part 1's bottleneck or Part 2's low-rank factorisation, deployed inside a system trained end-to-end by gradient descent instead of solved in closed form, and recognising which of the two applies is most of the work of understanding any section in it: a linear autoencoder with MSE loss is, provably (Baldi–Hornik), exactly PCA, so nonlinearity is the one thing separating an autoencoder from the flat-subspace ceiling Part 2 §14 proved PCA cannot exceed, and every subsequent architecture — denoising, sparse, variational, vector-quantised — is that same bottleneck escalated once more to fix a specific weakness (memorisation, uninterpretability, an ungenerative latent space, a continuous latent a Transformer can't consume); the VAE's fix in particular is not a heuristic but a direct consequence of *reverse* KL's zero-forcing asymmetry (Part 2 §24), which is why its known failure mode, posterior collapse, is predictable rather than merely observed, and why the resulting smooth, hole-free latent space is precisely what makes running the entire diffusion process inside it — rather than in a 786,432-value, quadratic-attention pixel space — the trick that makes Stable Diffusion, Sora and Veo tractable at all; meanwhile LoRA and Matryoshka Representation Learning descend from Part 2's *other* theorem, Eckart–Young, along its two separate consequences — that truncation is optimal, applied to a weight *update* instead of to data, and that components are already ordered by importance, trained in directly instead of computed after the fact from a task-blind variance criterion — which is exactly why MRL beats post-hoc PCA on the identical base embedding. |
