---
title: "Dimensionality Reduction — Part 3: Dimensionality Reduction in the Age of LLMs & Generative AI"
topic: dimensionality-reduction
lecture: 09
source: "output/Lecture_09 - Module 3 Dimensionality Reduction Part 3"
slides: 47
video: "https://www.youtube.com/watch?v=8VKk1at2wSk"
instructor: "Ravi Sankar Adepu"
runtime: "56:28"
---

# Dimensionality Reduction — Part 3
### Dimensionality Reduction in the Age of LLMs & Generative AI

---

## 📋 About this lecture and its capture

**This is not the t-SNE/UMAP lecture Part 1's closing slide predicted, and not a repeat of Part 2's
content.** Part 1's forward reference guessed "nonlinear methods — t-SNE, UMAP, autoencoders" for
"Part 3"; Part 2 delivered t-SNE and UMAP itself. This lecture's actual title, stated on its own
opening slide, is **"Dimensionality Reduction in the Age of LLMs & Generative AI"**, and its four
parts are a different subject entirely: how the same handful of ideas from
[Part 1](dimensionality-reduction-01.md) and [Part 2](dimensionality-reduction-02.md) — a bottleneck,
a low-rank factorisation, a variance-preserving projection — show up **inside** the architectures that
power modern generative AI and LLM serving.

| | Part | Runtime | The deck's own framing |
|---|---|---|---|
| **1** | **Autoencoders & Variational Autoencoders** | 1:33 – 31:07 | AE → denoising AE → sparse AE → VAE → VQ-VAE |
| **2** | **Latent Diffusion Models** | 31:07 – 37:27 | From dimensionality reduction to generation |
| **3** | **Low-Rank Methods in Transformers** | 37:27 – 45:43 | LoRA, DoRA — fine-tuning as a low-rank update |
| **4** | **Embedding Models & Matryoshka Representations** | 45:43 – 56:28 | Making the embedding dimension a runtime choice |

The through-line, stated nowhere explicitly but true of every section, is worth having in mind before
you start: **every method in this lecture is either a nonlinear generalisation of PCA (Part 1) or a
low-rank factorisation (Part 2's SVD/MF), applied to a modern deep-learning system.** A plain
autoencoder *is* nonlinear PCA. A VAE is an autoencoder with a probabilistic bottleneck. LoRA is
truncated-SVD-shaped weight-update compression. Matryoshka embeddings are PCA's "keep the top $k$"
idea, trained in rather than computed after the fact. Reading the whole lecture through that lens
turns four seemingly separate topics into one argument continuing from Part 2.

The deck contains **47 distinct slide states** across the four parts (39 content slides, 4 part
dividers, title, outline, and two closing slides).

> ✅ **Capture quality: excellent, no content gaps** — but read this note on how the capture behaves.
> 108 raw frames over 56 minutes (excluding the interactive-demo frames noted below). Every content
> slide has a fully-built state; every equation and citation is legible, and — like both earlier
> decks in this module — this one **cites its own sources on nearly every slide** (Weng 2018, van den
> Oord et al. 2017, Hu et al. 2022, Liu et al. 2024, Rege 2024).
>
> **One long single-slide stretch, explained rather than a gap.** Frames 77–98 (48:59–50:41, about 1
> minute 42 seconds) are all the *same* "ML Learning" content slide on Matryoshka Representation
> Learning — the deck's change-detector kept re-sampling on small cursor/webcam movements while the
> instructor spoke at length over one static slide. **Nothing is missing**: the slide's five bullets
> (motivation, core idea, the surprise result, beating post-hoc PCA, and the "remove numbers from the
> end" property) are the complete content, and §36–§38 below teach them in full depth to make up for
> the absence of a transcript.
>
> **Three interactive demos were captured at multiple slider settings**, and the notes reproduce real
> numbers from each rather than describing them: the linear-autoencoder-as-PCA rank slider (§9), the
> denoising-autoencoder noise slider (§13), and (implicitly, via the deck's own worked numbers) the
> VQ-VAE and latent-diffusion compression figures (§21, §26).
>
> The instructor is **Ravi Sankar Adepu** — named in the webcam tile throughout.

---

## How to read this document

The four parts share almost no vocabulary with each other on the surface, but they are built from
five ideas you already have from Parts 1–2, redeployed:

```
   THE BOTTLENECK             (Part 1's PCA: fewer dims than input)
        │
        ├─ trained by backprop instead of eigendecomposition, nonlinearity added ──▶ AUTOENCODER   §5–§11
        │       └─ made robust: reconstruct clean from corrupted ─────────────────▶ DENOISING AE   §12–§14
        │       └─ made interpretable: force most units OFF ──────────────────────▶ SPARSE AE      §16–§18
        │
        ├─ made GENERATIVE: encode a DISTRIBUTION, not a point ─────────────────▶ VAE              §19–§25
        │       └─ made DISCRETE: snap to a learned codebook ───────────────────▶ VQ-VAE           §26–§28
        │
   THE LOW-RANK FACTORISATION  (Part 2's SVD: X ≈ rank-k pieces)
        │
        ├─ applied to a WEIGHT UPDATE instead of data ──────────────────────────▶ LoRA             §31–§33
        │       └─ decoupled into magnitude × direction ─────────────────────────▶ DoRA            §35
        │
   THE VARIANCE-ORDERED PROJECTION (Part 2's PCA: PC1 matters most)
        │
        └─ TRAINED so any prefix is already a good subspace ────────────────────▶ MATRYOSHKA       §36–§39

   AND ONE NEW IDEA: compress the SPACE the generative process runs in ─────────▶ LATENT DIFFUSION  §29–§30
```

If you are revising under time pressure: **§8–§10 (why a linear autoencoder *is* PCA), §22–§24
(the VAE's reparameterisation trick and ELBO), and §31–§32 (LoRA's low-rank update)** are the interview
core — they are the three "explain the mechanism, not just the name" questions this content is most
likely to produce.

Everything in a `🧪 Worked example` block should be reproducible by you on paper, and every number in
this file that came from a live demo is checked against the deck's own displayed values.

---

## What you'll understand after reading this

- You'll be able to define an autoencoder, explain what the bottleneck forces, and **prove** that a
  linear autoencoder with MSE loss recovers exactly the PCA subspace.
- You'll be able to explain why a denoising autoencoder generalises better than a plain one, in terms
  of where noise energy lives in a low-rank spectrum.
- You'll be able to derive the sparsity penalty's KL form and explain why it's shaped like it is.
- You'll be able to explain **why a plain autoencoder cannot generate**, derive the reparameterisation
  trick, and write the ELBO with both terms explained and their tension named.
- You'll be able to explain vector quantisation, the straight-through gradient estimator, and why
  discrete latents matter for building a Transformer over images.
- You'll be able to explain why pixel-space diffusion is computationally infeasible and derive the
  exact compute saving latent diffusion buys.
- You'll be able to state LoRA's low-rank hypothesis, explain the $B=0$ initialisation, and compute
  the parameter savings for a real layer.
- You'll be able to explain what DoRA decouples that LoRA doesn't, and why that closes DoRA's gap to
  full fine-tuning.
- You'll be able to explain why fixed-size embeddings don't scale, derive Matryoshka's nested loss,
  and explain why it beats post-hoc PCA on the same embedding.
- You'll be able to connect every method in this lecture back to a specific mechanism from
  [Part 1](dimensionality-reduction-01.md) or [Part 2](dimensionality-reduction-02.md).

---

## Before we start: what you need to know

### Prerequisite 1 — The bottleneck, recalled from Part 2

[Part 2 §13](dimensionality-reduction-02.md) already established that **PCA is a linear autoencoder
with tied weights**: encoder $z = U_r^\top x$, decoder $\hat x = U_r z$, and Baldi & Hornik (1989)
proved this is not an analogy but a theorem — a linear autoencoder's global optimum *is* the PCA
subspace. This entire lecture's Part 1 is "now build that with backprop instead of an eigensolver, and
add nonlinearity" — so if that theorem isn't solid, reread Part 2 §13 first.

### Prerequisite 2 — KL divergence, recalled from Part 2

[Part 2 §23](dimensionality-reduction-02.md) derived $\mathrm{KL}(P\|Q) = \sum P\log(P/Q)$, proved it
non-negative via Jensen, and showed $H(P,Q) = H(P) + \mathrm{KL}(P\|Q)$. Two of this lecture's
sections use KL directly: the sparse autoencoder's activation penalty (§17) and the VAE's ELBO (§23),
and **[Part 2 §24's](dimensionality-reduction-02.md) forward-vs-reverse asymmetry is exactly why VAEs
have a known failure mode** (§25).

### Prerequisite 3 — Bernoulli distributions

> **Bernoulli($\rho$)** — a coin flip returning 1 with probability $\rho$ and 0 with probability
> $1-\rho$. Mean $\rho$.
>
> *Why it matters here:* §17's sparsity penalty compares two Bernoulli distributions — the *target*
> activation rate and the *actual* average activation rate — via KL divergence. A "neuron activation
> rate" is being treated as a coin-flip probability.

### Prerequisite 4 — The reparameterisation problem, in miniature

> **Why can't you just backpropagate through a random sample?** Sampling $z \sim \mathcal{N}(\mu,
> \sigma^2)$ is not a differentiable function of $\mu$ and $\sigma$ — it's a *stochastic* operation,
> and there is no meaningful $\partial z/\partial\mu$ to compute through a random number generator.
>
> *The fix, in one line:* $z = \mu + \sigma\cdot\epsilon$ where $\epsilon \sim \mathcal{N}(0,1)$ is
> sampled **once, from a source with no learnable parameters**. Now $z$ is a *deterministic* function
> of $\mu$ and $\sigma$ (with $\epsilon$ treated as a fixed input), so
> $\partial z/\partial\mu = 1$ and $\partial z/\partial\sigma = \epsilon$ — both computable. §22 uses
> exactly this.

### Prerequisite 5 — Straight-through gradient estimation, in miniature

> **The problem:** $\mathrm{argmin}$ — "pick the nearest codebook entry" — is a discrete, piecewise-
> constant operation. Its gradient is zero almost everywhere and undefined at the boundaries, so
> backpropagation through it carries no signal.
>
> **The straight-through estimator:** on the forward pass, use the true discrete operation. On the
> backward pass, **pretend it was the identity function** and pass the gradient straight through
> unchanged. It is a deliberate approximation — biased, not exact — that happens to work well enough
> in practice to be standard. §21 uses this for vector quantisation.

### Prerequisite 6 — Matrix rank and low-rank updates, recalled from Part 2

[Part 2 §2–§4](dimensionality-reduction-02.md) established that a **rank-1 matrix** $uv^\top$ stores
$m+n$ numbers instead of $mn$, and that truncating the SVD to rank $k$ is provably the best possible
rank-$k$ approximation (Eckart–Young). §31's LoRA is exactly this idea, applied not to a data matrix
but to **a weight update matrix** $\Delta W$.

---

## The big picture

Parts 1 and 2 of this module answered "how do I find a small number of coordinates that describe my
data well?" using closed-form linear algebra — eigendecomposition, SVD, spectral tricks. This lecture
asks the same question inside systems that are trained end-to-end with gradient descent, and the
answer at every turn is a variation on the same two moves: **force information through a narrow
channel** (the bottleneck), or **factor a large object into a small number of pieces** (low rank).

**Part 1 builds up the autoencoder family as one escalating argument.** A plain autoencoder is
*nonlinear PCA* — Prerequisite 1's theorem tells you the linear case is literally PCA, and adding
nonlinear activations is the only thing that buys curved-manifold capacity (exactly the point Part 2
§13 made about *why* a deep autoencoder differs from PCA at all). But a plain autoencoder has two
practical weaknesses the deck fixes in turn: it can memorise rather than generalise (fixed by
**denoising** — reconstruct clean data from corrupted input, which is Part 2 §12's denoising argument
turned into a training objective), and its latent code has no interpretable structure (fixed by
**sparsity** — most latent units must stay off, which is again a KL-divergence penalty, this time on
activation rates rather than on layer outputs). Neither fix makes the model **generative**, though —
you cannot sample a new realistic output from a plain autoencoder's latent space, because nothing ever
taught that space to be smooth or complete. The **VAE** fixes exactly that, by encoding a distribution
instead of a point and regularising it toward a known prior with KL — and the **VQ-VAE** takes a
different fork, making the latent *discrete* instead, because a discrete code is exactly the format a
Transformer expects, which is why DALL-E and Jukebox are built on it.

**Part 2 is one paragraph of consequence.** Diffusion models generate by iterative denoising, and doing
that in pixel space is computationally prohibitive — a 512×512 image is 786,432 numbers, and
self-attention at that resolution costs a memory-exploding $\mathcal{O}(n^2)$. The fix is to run the
entire diffusion process in the **compressed latent space** a VAE's encoder produces instead — which
only works *because* Part 1 already built that VAE. Latent diffusion is not a new idea; it's Part 1's
machinery, reused as a computational trick that turns an infeasible problem into Stable Diffusion.

**Part 3 changes subject to a different kind of bottleneck.** Fine-tuning an LLM updates a weight
matrix with billions of entries, but the *update itself* — not the weights, the update — turns out to
have low intrinsic rank. LoRA factors that update as a product of two small matrices, the same
rank-reduction idea from Part 2's SVD, applied to $\Delta W$ instead of to data.

**Part 4 closes the module by turning Part 2's "keep the top $k$" into something learned.** PCA lets
you truncate after the fact because its components are already ordered by importance. A raw neural
embedding has no such ordering — dimension 47 is not more important than dimension 800 unless you
train it to be. Matryoshka Representation Learning does exactly that: trains the network so that
*every prefix* of the embedding is independently a good, complete representation, which means
truncation — Part 2's central trick — becomes a runtime choice instead of a retraining decision.

### The whole lecture in one diagram

```
   PART 1 — AUTOENCODERS & VARIATIONAL AUTOENCODERS                              §5–§28

     bottleneck: z ≪ x  →  x̂ ≈ x                                                  §5–§9
     🎯 linear AE + MSE loss  =  EXACTLY the PCA subspace (Baldi–Hornik)          §9–§10
              │
     ┌────────┼─────────────────┐
     ▼        ▼                 ▼
   plain    DENOISING          SPARSE
   AE       reconstruct        force most units OFF
            CLEAN from         KL(Bernoulli(ρ) ‖ Bernoulli(ρ̂))       §16–§18
            CORRUPTED
            noise lives in                                    ┌── modern relevance:
            discarded dims                                    │   LLM interpretability,
            §12–§14                                            │   monosemantic features
              │
              ▼
     but NONE of these are GENERATIVE — the latent space has
     holes; sampling a random z decodes to garbage             §19–§20
              │
              ▼
   VAE: encode a DISTRIBUTION q(z|x)=N(μ,σ²), not a point       §22
        reparameterisation: z = μ + σ⊙ε                          (differentiable)
        ELBO = RECONSTRUCTION − KL(q(z|x) ‖ p(z))                §23
                                    ▲
                          reverse KL (Part 2 §24) ⇒ mode-seeking
                          ⇒ VAE's own failure mode: posterior collapse
              │
              ▼
   VQ-VAE: SNAP z to nearest CODEBOOK vector — DISCRETE latents  §26–§28
            straight-through gradient around the non-diff. argmin
            an image becomes a 32×32 grid of TOKENS
            ⇒ lets DALL·E, Jokebox run a TRANSFORMER over pixels

   ═══════════════════════════════════════════════════════════════════════
   PART 2 — LATENT DIFFUSION                                                    §29–§30

     pixel-space diffusion: 512² RGB = 786,432 values/step, O(n²) attention
     ⇒ INFEASIBLE
     FIX: run diffusion in a VAE's LATENT SPACE instead (64×64 = 64× fewer
     spatial positions) — Stable Diffusion, Sora, Veo all do this

   ═══════════════════════════════════════════════════════════════════════
   PART 3 — LOW-RANK METHODS IN TRANSFORMERS                                    §31–§35

     full fine-tuning updates ALL of W ∈ R^(d×k) — billions of params
     KEY INSIGHT: the UPDATE ΔW has low intrinsic rank
     LoRA:  ΔW ≈ B·A,  B∈R^(d×r), A∈R^(r×k),  r ≪ min(d,k)          §31–§32
            B starts at ZERO ⇒ training begins at the pretrained model, exactly
            up to ~10,000× fewer trainable params, zero added inference latency
     DoRA:  decouple W = m·(V/‖V‖) — magnitude trained separately,   §35
            LoRA's B·A applied only to DIRECTION
            ⇒ closer to how full fine-tuning actually varies weights

   ═══════════════════════════════════════════════════════════════════════
   PART 4 — EMBEDDING MODELS & MATRYOSHKA REPRESENTATIONS                       §36–§39

     fixed-size embeddings DON'T SCALE: 100M docs @ 3072-dim = 1.2TB,
     can't get a cheap 256-dim version, doc/query size MUST match
     MRL: train so EVERY PREFIX z_{1:8} ⊂ z_{1:16} ⊂ ... ⊂ z_{1:2048}
          is independently valid — nested dolls
     L_Matryoshka = average( L(z_{1:8}) + L(z_{1:16}) + ... + L(z_{1:2048}) )
     ⇒ 256-dim MRL embedding: 96.6% of full 1024-dim quality
     ⇒ beats POST-HOC PCA — PCA runs AFTER the encoder and isn't data-aware;
       MRL learns the nested structure DURING training

   ═══════════════════════════════════════════════════════════════════════
   THE THREAD: every part is either PART 2's PCA/SVD (§9's theorem, §32's
   low-rank ΔW, §38's ordered prefixes) made TRAINABLE, or a bottleneck
   (§7) made PROBABILISTIC (§22) or DISCRETE (§26).
```

---

# PART 1 — Autoencoders & Variational Autoencoders

*1:33 – 31:07*

---

## 5. Autoencoder

> *"An autoencoder is a feed-forward neural network trained to take an input $x$ and reproduce it as
> output $\hat x$."* [slide 7, 3:19]
>
> - *"**Bottleneck trick** — a middle layer whose dimension is far smaller than the input."*
> - *"In practice: a **784-dimensional input** (say, a 28×28 pixel image) gets compressed down to just
>   **20 dimensions**"*
> - *"The decoder then attempts to **reconstruct** the original 784-dimensional input using only those
>   20 compressed values"*

$$x \xrightarrow{\text{encoder } g_\phi} z \xrightarrow{\text{decoder } f_\theta} x' \qquad x \approx x'$$

| Symbol | Read it as | What it means |
|---|---|---|
| $x$ | "x" | The input — e.g. a flattened $28\times28$ MNIST digit, 784 numbers |
| $g_\phi$ | "g phi" | The **encoder**, a neural network with parameters $\phi$ |
| $z$ | "z" | The **latent code** — the bottleneck's contents. Here, 20 numbers. |
| $f_\theta$ | "f theta" | The **decoder**, a neural network with parameters $\theta$ |
| $x'$ (or $\hat x$) | "x prime" | The reconstruction. The training signal is how close this is to $x$. |

**Compression ratio, stated plainly:** $784/20 = \mathbf{39.2\times}$. The network is being forced to
find a 20-number summary of a 784-number image good enough that another network can rebuild the image
from it alone.

### 5.1 Why this is nonlinear PCA — the connection to make immediately

Nothing here has said "neural network" in a way that requires nonlinearity. Strip the activation
functions out and this is *exactly* [Part 2 §13's](dimensionality-reduction-02.md) linear
autoencoder — and Part 2 stated the theorem: a linear autoencoder with squared-error loss has, as its
global optimum, **exactly the PCA subspace** (Baldi & Hornik, 1989). §9 below verifies this with a live
demo and a quiz.

> 💡 **So an autoencoder is, structurally, "PCA plus nonlinear activations."** Everything this section
> adds beyond §5's diagram — denoising, sparsity, the probabilistic latent of a VAE — is a modification
> layered on top of that same encoder–bottleneck–decoder skeleton. Keep that skeleton in view; every
> subsequent architecture in Part 1 is this diagram with one thing changed.

---

## 6. The bottleneck — forced dimensionality reduction

> [slide 10, 5:32]
>
> - *"**Why a bottleneck?** If latent dim $z \ll$ input dim $d$, the model is forced to discard noise
>   and keep only informative structure"*
> - *"**Classic example:** MNIST images → AE compresses to 20 latent dims → decoder reconstructs 784
>   pixels from those 20 values"*
> - *"**Hidden layers as feature selectors:** Each layer progressively strips away less important
>   dimensions - deeper layers see increasingly abstract features"*
> - *"**Nonlinearity is the superpower**"*

### 6.1 Why "forced to discard noise" is not just marketing language

**The bottleneck is a hard information-theoretic constraint, not a suggestion.** A layer of 20 units
can represent at most 20 independent numbers per example — there is no way for 784 numbers of raw pixel
noise to pass through a 20-wide channel intact. Something has to be thrown away, and the only lever the
optimiser has to reduce reconstruction loss is to **choose what to throw away**. Since noise is (by
definition) the part of the signal that doesn't correlate with anything reconstructible, and structure
is (by definition) the part that does, gradient descent on the reconstruction loss naturally routes
structure through the bottleneck and lets noise fall away. This is precisely the argument [Part 2
§12](dimensionality-reduction-02.md) made for PCA-as-denoiser, generalised: PCA discards low-variance
*linear* directions; an autoencoder discards whatever the encoder cannot usefully compress, linear or
not.

### 6.2 "Nonlinearity is the superpower" — the one line that separates this from Part 2

This four-word bullet is the entire reason Part 1 of this lecture exists as a topic distinct from
[Part 2's PCA sections](dimensionality-reduction-02.md). Recall [Part 2 §14](dimensionality-reduction-02.md)'s
Swiss roll: PCA can only fit a *flat* subspace, and folds any genuinely curved manifold onto itself. A
linear autoencoder inherits that exact limitation, because — per §5.1 — it *is* PCA. **The moment you
add a nonlinear activation function between encoder layers, the achievable latent manifold stops being
flat**, and the network can wrap a bottleneck around a curved surface the way PCA never could. §9's
quiz makes you state this precisely.

---

## 7–9. How to train an AE, and the bottleneck in action

> [slides 12–14, 7:04–8:57]
>
> - *"**Vx:** A matrix multiplication project D dimensional x to k dimensional plane."*
> - *"**Ux:** Project k dimensional hidden unit to D dimensional $\hat x$"*
> - *"**Overall Output is:** $\hat x = UVx$ [A linear function]"*
> - *"**How it learn:** Learn the U, V matrix or all weights so that we get minimum loss"*
>
> $$L(x, \hat x) = \|x - \hat x\|^2$$
>
> *"We minimize L to learn U,V. **Limitations:** Non generative, Latent dimension?"*

```
        x̃  ┌──────────┐
     ──────▶│ D units  │   x̃ = reconstruction
            └────┬─────┘
                 │ U (decoder)
            ┌────┴─────┐
            │ K units  │   z = Vx  (the bottleneck)
            └────┬─────┘
                 │ V (encoder)
            ┌────┴─────┐
     ───────▶│ D units  │   x = input
            └──────────┘
```

| Symbol | Read it as | What it means | Shape |
|---|---|---|---|
| $V$ | "V" | Encoder weight matrix | $k \times D$ |
| $U$ | "U" | Decoder weight matrix | $D \times k$ |
| $\hat x = UVx$ | "U V x" | The full round trip, **in this linear case a single matrix multiply** by $UV$ | — |
| $D$ | "D" | Input/output dimension | — |
| $k$ | "K" | Bottleneck (latent) dimension | — |

**Note the slide's own honesty: "Limitations: Non generative, Latent dimension?"** Both limitations get
their own section later — non-generative is fixed by the VAE (§19–§25); "what latent dimension?" is
never fully answered for a plain autoencoder (it's a hyperparameter you guess and tune), and is the
exact problem Matryoshka Representation Learning (§36–§39) eventually solves properly, by making it a
runtime choice instead of a training-time commitment.

### 🧪 The deck's own live demo — the rank slider, at three captured settings

> *"Drag the slider to change the latent dimension k. A linear autoencoder = rank-k (PCA)
> reconstruction."* [slides 15–17, 9:01]

| $k$ | Reconstruction error | What you see |
|---|---|---|
| **4 / 28** | **0.131** | A recognisable but blurred, smoothed-out "7" — fine strokes lost |
| **5 / 28** | **0.104** | Still visibly blurred, but closer — the curve has not yet flattened |
| **16 / 28** | **0.000** | Visually identical to the original |

> *"Small k → coarse blob (most variance only); large k → fine detail returns. **This is exactly the AE
> bottleneck tradeoff.**"*

**Read what's happening between $k=5$ and $k=16$: the error drops from 0.104 to exactly 0.000.** This is
the same lesson as [Part 2 §6's](dimensionality-reduction-02.md) rank-slider demo on the butterfly image
— past a certain $k$, additional components buy essentially nothing, because the underlying data's
effective rank has already been captured. For a simple, mostly-blank MNIST digit, that effective rank is
somewhere between 5 and 16, well under 28.

**And the caption's parenthetical — "A linear autoencoder = rank-k (PCA) reconstruction" — is not
loose language. It is literally true**, per §5.1's theorem: this demo *is* a truncated-SVD
reconstruction with a neural-network label on it, because no nonlinearity has been introduced yet.

```interactive
type: slider
title: Linear autoencoder bottleneck = rank-k PCA
concept: A linear autoencoder's reconstruction is exactly a rank-k SVD truncation
control: A latent-dimension slider k from 1 to 28
observe: The reconstructed digit and its reconstruction error, updating live
insight: The error curve flattens to zero well before k reaches 28 — the point at which it flattens is exactly the digit's effective rank, the same quantity a Part 2 scree plot would read off the singular values directly
fallback: The three captured states above — k=4 gives error 0.131 (a coarse blob), k=5 gives error 0.104 (still blurred), and k=16 gives error 0.000 (visually identical to the original)
```

### 🎯 Quick check — the theorem, tested

> [slides 28–29, 16:18–16:42]
>
> **A standard autoencoder uses only linear activations and an MSE loss. How does its learned subspace
> relate to PCA?**
>
> A. It is unrelated to PCA — neural networks learn something entirely different
> B. It spans the same subspace as the top-k principal components
> C. It always outperforms PCA because it has more parameters

<details>
<summary><b>Answer</b></summary>

**B.** The slide: *"A linear autoencoder with MSE loss recovers the same subspace as PCA. **Nonlinear
activations are what let autoencoders surpass PCA.**"*

This is [Part 2 §13's](dimensionality-reduction-02.md) Baldi–Hornik theorem, restated as a quiz. Two
wrong answers, and why each is wrong:

- **A** gets the mechanism backwards. It is not that neural networks "learn something entirely
  different" — a *linear* network provably learns the *same* thing as PCA. The interesting behaviour
  only begins once nonlinearity is added.
- **C** confuses parameter count with representational advantage. A linear AE with more parameters than
  strictly needed to represent a rank-$k$ subspace doesn't do anything extra with them — it still
  converges to the same $k$-dimensional subspace, just found via a redundant parameterisation. "More
  parameters" alone buys nothing; a genuinely new function class (nonlinearity) does.

**The one-sentence version for an interview:** *linear autoencoders are PCA in disguise; the entire
value proposition of a "real" (nonlinear) autoencoder is the nonlinearity.*
</details>

---

## 12–14. Denoising Autoencoder

> [slide 18, 11:01]
>
> - *"Input is **partially corrupted** by adding noise or masking values: $\tilde x = M[x]$; the model
>   is trained to recover the **original input**, not the corrupted one"*
> - *"Avoids the 'identity function' overfitting risk of a plain autoencoder"*
> - *"Forces the model to capture **relationships between input dimensions** to infer the missing
>   pieces - building robust latent representations"*
>
> $$\mathcal{L}_{DAE}(\theta, \varphi) = (1/n)\sum_i \left(x^{(i)} - f_\theta(g_\varphi(\tilde x^{(i)}))\right)^2$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\tilde x$ | "x tilde" | The **corrupted** input — noise added, or values masked out |
| $M[x]$ | "M of x" | The corruption operator |
| $x$ | "x" | The **original, uncorrupted** target — what the loss compares against, **not** $\tilde x$ |

**The one-word difference from a plain AE that changes everything:** the loss compares the
reconstruction to the *original* $x$, while the network only ever sees $\tilde x$. A plain autoencoder
could, in principle, learn the trivial identity function and pass every input straight through — with
enough capacity and no constraint beyond the bottleneck's width, memorisation is a viable shortcut. A
denoising autoencoder cannot take that shortcut: **the identity function on $\tilde x$ reconstructs the
noise, not the original**, so it scores badly on exactly the objective being optimised. The network is
forced to learn what "belongs" in the data — the *relationships between dimensions* the slide
mentions — because that's the only way to recover missing or corrupted values from context.

> 👉 **See also.** This is the same capacity-vs-memorisation overfitting story [Deep Neural Networks
> Part 2 §9](../Deep%20Neural%20Networks/deep-neural-networks-02.md) tells for a plain feedforward
> network — "enough capacity and no constraint, and the model memorises rather than generalises" —
> reapplied here to a bottleneck instead of a parameter count, with noise-corruption standing in for
> the earlier regularizers.

### 🧪 The deck's own live demo — the denoising slider, at two captured noise levels

> *"Add noise with the slider. The denoiser keeps only the top low-rank components - noise lives in the
> discarded ones."* [slides 20–22, 12:33–14:23]

| Noise level | Error, noisy vs clean | Error, denoised vs clean | What happened |
|---|---|---|---|
| **0.40** | **0.404** | **0.291** | A meaningful reduction — the "7" is legible again where the noisy input was barely so |
| **0.14** | **0.141** | **0.144** | Almost no reduction — at low noise, the denoised output is *not measurably better* than the noisy input |

**These two states together teach something the single-state view would miss.** At noise level 0.40,
denoising clearly helps (0.404 → 0.291, a 28% reduction). At noise level 0.14, it barely moves the
needle (0.141 → 0.144 — statistically flat, possibly noise in the measurement itself). **The lower the
corruption, the closer "denoise" gets to "do nothing," because there is proportionally less noise
energy sitting in the discarded low-rank directions to remove.** This is the exact mechanism [Part 2
§12](dimensionality-reduction-02.md) quantified for PCA — noise is spread roughly evenly across all
directions while signal concentrates in a few, so a low-rank projection removes a much larger share of
the noise than of the signal, and that advantage shrinks as the noise itself shrinks toward zero.

> ⚠️ **The exact numbers above (0.404→0.291, 0.141→0.144) come from two different frames of the same
> live-noise demo, so they are not a controlled paired comparison** — the underlying random noise draw
> differs between them. Treat the *qualitative* pattern (large noise → large benefit, small noise →
> small benefit) as the reliable takeaway; don't over-read the precise ratio between the two rows.

```interactive
type: slider
title: Denoising in action
concept: Denoising works because noise energy concentrates in the discarded low-rank directions
control: A noise-level slider from 0 to 1
observe: Clean, noisy, and denoised versions of a digit side by side, with live error readouts against the clean original
insight: The benefit of denoising (noisy error minus denoised error) grows with the noise level, because more of the added noise energy falls into directions the bottleneck discards — at very low noise there's little energy there to remove and denoising barely helps
fallback: The two captured states — noise 0.40 gives 0.404 (noisy) to 0.291 (denoised), noise 0.14 gives roughly flat 0.141 to 0.144
```

---

## 16–18. Sparse Autoencoder

> [slide 23, 14:27]
>
> - *"Adds a **sparsity constraint** on the hidden activations: only a small number of units may fire
>   at once - each neuron should stay inactive most of the time"*
> - *"**Target sparsity** $\rho$ (e.g. 0.05): the average activation $\hat\rho_j$ of each hidden unit
>   is pushed toward this small value"*
> - *"**Penalty:** a KL divergence between a Bernoulli($\rho$) and Bernoulli($\hat\rho_j$), weighted by
>   $\beta$, is added to the reconstruction loss"*
> - *"**Robustness without a hard bottleneck** - the layer can be large, but only a few neurons
>   activate per input"*
> - *"**Modern relevance:** sparse autoencoders are now central to LLM **interpretability** -
>   decomposing activations into monosemantic features"*
>
> $$\mathcal{L}_{SAE}(\theta) = \mathcal{L}(\theta) + \beta\sum_j D_{KL}\!\left(\rho \,\|\, \hat\rho_j\right)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\rho$ | "rho" | Target sparsity — the fraction of the time a unit is *allowed* to be meaningfully active. E.g. 0.05 means "on" 5% of the time. |
| $\hat\rho_j$ | "rho hat sub j" | The **actual** average activation of hidden unit $j$, measured across a batch |
| $\beta$ | "beta" | How strongly the sparsity penalty is weighted against the reconstruction loss |
| $D_{KL}(\rho\|\hat\rho_j)$ | — | The KL divergence between two Bernoulli distributions with means $\rho$ and $\hat\rho_j$ |

### 17.1 A genuinely different kind of bottleneck — no width restriction at all

**This is worth pausing on, because it inverts the whole framing so far.** Every previous section made
the case for a *narrow* layer. The sparse autoencoder's layer **can be arbitrarily wide** — the deck's
own bullet says so explicitly ("the layer can be large"). The constraint is not on *how many* units
exist, but on **how many are allowed to be non-zero for any given input**. A 10,000-unit sparse layer
where only 50 units fire per example is, in an information sense, still a narrow channel — just a
*sparse* one instead of a *small* one.

### 17.2 🧪 Derive why the penalty is shaped the way it is, and read the deck's own plot

The KL divergence between two Bernoulli distributions with means $\rho$ and $\hat\rho$ is:

$$D_{KL}(\rho\|\hat\rho) = \rho\log\frac{\rho}{\hat\rho} + (1-\rho)\log\frac{1-\rho}{1-\hat\rho}$$

This is [Part 2 §23's](dimensionality-reduction-02.md) KL formula, specialised to a two-outcome
(Bernoulli) distribution — sum over just the two outcomes "fired" and "didn't fire."

**Read the deck's own plot** [slides 24–27], which shows $D_{KL}(\rho=0.25\|\hat\rho)$ against
$\hat\rho \in [0,1]$: the curve has its **minimum of exactly 0 at $\hat\rho = 0.25$** and rises sharply
toward both ends, blowing up as $\hat\rho \to 0$ or $\hat\rho \to 1$.

**Verify the shape at the endpoints, because it explains why the penalty is so effective.** As
$\hat\rho \to 0$ with $\rho = 0.25$ fixed:

$$D_{KL} \to 0.25\log\frac{0.25}{0} + 0.75\log\frac{0.75}{1} \to +\infty$$

**The first term diverges.** A unit that never fires at all ($\hat\rho \to 0$) is punished
*infinitely* hard — the penalty doesn't gently discourage that failure mode, it forbids it outright.
Symmetrically, a unit that fires on nearly every example ($\hat\rho \to 1$) is also punished to
infinity via the second term. **The KL penalty is a soft wall on both sides of the target, sharpest at
the extremes** — which is exactly the shape you want if the goal is "push every unit toward firing
*about* 25% of the time, but never toward always-on or always-off."

> 💡 **Why KL rather than a simpler penalty (e.g. $(\rho - \hat\rho)^2$)?** A squared-error penalty is
> symmetric and bounded — it treats "always off" and "slightly too sparse" as comparably bad relative
> to their distance from $\rho$, and never forbids either extreme outright. KL's asymptotic blow-up at
> both 0 and 1 makes the two truly degenerate solutions (dead units, saturated units) infinitely
> costly, which a squared penalty cannot do.

### 17.3 The bridge to modern LLM interpretability

> *"sparse autoencoders are now central to LLM interpretability — decomposing activations into
> monosemantic features"*

**This line deserves unpacking, because it's the most consequential single claim in Part 1 of this
lecture for 2026-era ML work.** A trained LLM's individual neurons are typically **polysemantic** — one
neuron fires for several unrelated concepts at once (a famous early example: one GPT-2 neuron
activating for both "the" in academic writing and certain DNA sequences), because the model has more
concepts to represent than it has neurons, and superposition lets it pack several into one dimension
via interference patterns.

**The fix mirrors §17 exactly, just applied to a trained model's activations instead of to raw data.**
Train a sparse autoencoder to reconstruct a layer's activation vectors, with a *much wider* hidden
layer than the original (say, 10× or more) and a strong sparsity penalty. Because there is now more
room than concepts, and only a few "reasons to fire" are allowed per example, individual SAE hidden
units tend to converge onto **single, human-interpretable concepts** — "monosemantic features" — even
though no label ever told them what to represent. Anthropic's *"Towards Monosemanticity"* (2023) and
its scaled-up follow-up work are the primary references for this direction, and it is presently one of
the most active areas of mechanistic interpretability research.

> 🔬 **Research opportunity.** Whether sparse-autoencoder features are the *right* unit of analysis for
> interpretability — versus, say, causal interventions or circuit-level analysis — is genuinely
> unsettled, and feature "splitting" (the same concept fragmenting differently at different SAE widths)
> is an open methodological problem. Cite this as an active area, not a solved one.

---

## 19–20. Why VAE? The motivation

> *"A plain autoencoder is great at reconstruction, but bad at generation"* [slide 30, 16:46]
>
> - *"**The latent code is meaningless:** a standard AE only learns to encode & decode with low loss -
>   it was never trained to generate. Sampling a random point in its latent space rarely decodes to
>   anything coherent"*
> - *"**Overfitting / memorisation:** with enough freedom, the AE memorises each sample in isolated
>   pockets of latent space rather than learning structure"*
> - *"**No smoothness:** the latent space has holes and gaps - interpolating between two encodings
>   passes through regions that decode to garbage"*
> - *"**The VAE fix:** encode each input as a **distribution** (not a point) and regularise it toward a
>   standard normal, forcing the latent space to be **continuous and complete**"*
> - *"**Payoff:** you can now sample $z \sim N(0, I)$, and decode realistic new data, and interpolate
>   smoothly between samples"*

The deck's figure shows two 2-D latent spaces side by side, seeded with the same five coloured
clusters of points: the **Autoencoder Latent Space** panel shows five tight, disconnected, oddly-shaped
blobs with visible empty space between them; the **VAE Latent Space (Unregularised)** panel shows the
same five colours, but now overlapping into a single continuous coloured field with no gaps.

### 20.1 Why this failure is inevitable, not incidental

**The plain autoencoder's loss function never once asks "what happens if I decode a point you weren't
trained on?"** Reconstruction loss only evaluates the network at the exact latent codes its own
training examples happen to land on. Nothing in the objective rewards a smooth, gap-free latent space —
so gradient descent has no reason to produce one, and generally won't, especially once the network has
enough capacity to place each training example wherever is locally convenient (**memorisation**, the
second bullet).

**Sampling from the trained latent space's *empirical* distribution rather than a fixed prior is a
real workaround people try, and it's worth knowing why it's fragile.** Even with a good empirical
sampler, holes between clusters remain unfixed — you'd only ever sample from where training data
happened to land, not from the *interior* connecting two related concepts. Interpolation — a genuinely
useful generative operation, "morph between these two faces" — depends specifically on that interior
being populated with meaningful decodes, which nothing in a plain AE's training guarantees.

> 💡 **The one-sentence summary that captures both failures:** *a plain autoencoder's latent space is
> only trained to be locally accurate at the training points; a VAE's latent space is additionally
> trained to be globally well-formed everywhere.* That second property is the entire point of the KL
> regularisation term in §23.

---

## 21–22. VAE — Encode as a Distribution, Not a Point

> [slide 34, 19:51]
>
> - *"**Key shift:** Encoder outputs two vectors - $\mu$ (mean) and $\sigma^2$ (variance) defining a
>   Gaussian distribution $q(z|x) = N(\mu, \sigma^2)$ rather than a single point $z$"*
> - *"**Sampling:** Latent code $z$ is sampled from this distribution: $z \sim N(\mu, \sigma^2)$; the
>   decoder then reconstructs $\hat x$ from this $z$"*
> - *"**Reparameterization Trick:** Raw sampling is not differentiable. Move the randomness to a
>   separate non-trainable node, making $z$ differentiable w.r.t. $\mu$ and $\sigma$"*
> - *"**Generative capability unlocked:** At inference, sample $z \sim N(0, I)$ directly → decoder
>   produces realistic new data"*
>
> $$z \sim q_\phi(z|x) = N(z; \mu, \sigma^2 I) \qquad\qquad z = \mu + \sigma \odot \epsilon,\ \ \epsilon \sim N(0, I)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $q_\phi(z|x)$ | "q phi of z given x" | The **encoder's output distribution** — not a point, a full Gaussian, parameterised by the network's weights $\phi$ |
| $\mu$ | "mu" | The mean vector the encoder predicts for this input |
| $\sigma^2$ | "sigma squared" | The (diagonal) variance vector the encoder predicts |
| $\epsilon$ | "epsilon" | A sample from a **fixed, parameter-free** standard normal — the source of randomness |
| $\odot$ | "elementwise product" | Multiply componentwise |

### 22.1 🧪 Derive the reparameterisation trick — and check it actually works

Directly sampling $z \sim N(\mu, \sigma^2)$ has no usable gradient with respect to $\mu$ or $\sigma$,
because sampling is a stochastic node, not a function — you cannot ask "how would $z$ have changed if
$\mu$ had been slightly different," because $z$'s value came from a random number generator, not from
a computation involving $\mu$ in any differentiable sense.

**The trick, Prerequisite 4 restated with the full derivation:** rewrite

$$z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim N(0, I)$$

**Verify this samples from the right distribution.** If $\epsilon \sim N(0,1)$, then a standard property
of the normal distribution says a linear transform $a + b\epsilon$ of a standard normal variable is
itself normal with mean $a$ and variance $b^2$. Here $a = \mu$, $b = \sigma$, so
$z \sim N(\mu, \sigma^2)$ exactly — **the same distribution as directly sampling**, but reached via a
computation graph where the randomness ($\epsilon$) is now a leaf node with no parameters attached to
it.

**Now the gradients exist and are trivial:**

$$\frac{\partial z}{\partial \mu} = 1, \qquad \frac{\partial z}{\partial \sigma} = \epsilon$$

Both are ordinary, well-defined partial derivatives of a deterministic function — because, once
$\epsilon$ is fixed as an input, $z = \mu + \sigma\epsilon$ *is* a deterministic function of $\mu$ and
$\sigma$. Backpropagation can now flow through the sampling step exactly as it flows through any other
layer. $\blacksquare$

> 💡 **The reparameterisation trick is not specific to VAEs.** Any time you need to backpropagate
> through a stochastic node whose distribution is parameterised by upstream network outputs, this move
> — separate the randomness into a fixed, parameter-free source and make the sample a deterministic
> function of it — is the general pattern. It appears again wherever policy-gradient variance reduction
> or differentiable sampling comes up (Module 8, Reinforcement Learning).

### 22.2 The two modes of use, and why they differ

**Training:** $z = \mu + \sigma\odot\epsilon$, with $\mu$ and $\sigma$ coming from the encoder applied
to a real input $x$. This is how the model learns to encode.

**Generation (inference):** discard the encoder entirely. Sample $z \sim N(0, I)$ directly — no $\mu$,
no $\sigma$, no input $x$ at all — and pass that $z$ straight to the decoder. **This only produces
realistic output because training explicitly pushed every encoded distribution toward matching
$N(0,I)$** (that's §23's KL term). If the encoder's distributions were left free to drift away from the
standard normal, a $z$ sampled from $N(0,I)$ at inference time would land somewhere the decoder was
never trained to make sense of — precisely the "holes and gaps" failure from §20, reintroduced.

```interactive
type: simulator
title: Point encoding vs distribution encoding
concept: Why encoding a distribution instead of a point produces a sampleable latent space
control: A toggle between "Plain AE" (point encoding) and "VAE" (distribution encoding), plus a slider to pick which of five training-set clusters to encode
observe: The latent space with all five clusters plotted; in AE mode each input lands on an exact point; in VAE mode each input spreads into a soft Gaussian blob whose overlap with neighbouring blobs is visible; a "sample random z" button decodes whatever point in latent space it lands on
insight: In AE mode, "sample random z" almost always lands in empty space between the tight point-clusters and decodes to garbage; in VAE mode the blobs overlap enough that a random sample almost always lands inside at least one, and decodes to something coherent
fallback: The deck's own figure — the Autoencoder Latent Space panel shows five disconnected blobs with visible gaps; the VAE Latent Space panel shows the same five colours now overlapping into one continuous field with no gaps
```

---

## 23. ELBO Loss — Reconstruction + KL Regularization

> [slide 37, 21:52]
>
> - *"**ELBO (Evidence Lower BOund):** VAE maximizes $\mathbb{E}[\log p(x|z)] - \mathrm{KL}(q(z|x) \|
>   p(z))$ - a lower bound on the true data log-likelihood $\log p(x)$"*
> - *"**Term 1 - Reconstruction:** $\mathbb{E}[\log p(x|z)]$ - encourages the decoder to faithfully
>   reconstruct the input; same as the AE loss"*
> - *"**Term 2 - KL Regularization:** $\mathrm{KL}(q(z|x) \| p(z))$ - penalizes the encoder for
>   straying from the prior $N(0, I)$; forces the latent space to be smooth, continuous, and
>   well-covered"*
> - *"**The tension:** Reconstruction wants sharp, narrow posteriors (encode precisely); KL wants
>   diffuse posteriors (stay close to prior) - $\beta$-VAE controls this tradeoff explicitly with
>   $\beta \cdot \mathrm{KL}$"*
> - *"**What KL buys us:** Every region of Z now maps to something meaningful; you can interpolate
>   between two latent codes and get a coherent visual transition - this is exactly what **Stable
>   Diffusion's VAE encoder exploits**"*
>
> $$\mathcal{L}_{VAE}(\theta,\varphi) = -\mathbb{E}_{z\sim q_\varphi(z|x)}\left[\log p_\theta(x|z)\right] + D_{KL}\!\left(q_\varphi(z|x)\,\|\,p_\theta(z)\right)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\log p_\theta(x|z)$ | "log p of x given z" | How likely the true $x$ is under the decoder's output distribution, given the latent $z$ — negative of this is essentially the reconstruction error |
| $p_\theta(z)$ | "p theta of z" | The **prior** on the latent space — $N(0, I)$, fixed and simple |
| $q_\varphi(z|x)$ | "q phi of z given x" | The **approximate posterior** — the encoder's output distribution for a specific $x$ |

### 23.1 Naming the direction of KL — and connecting it to Part 2's asymmetry

**The KL term is $\mathrm{KL}(q_\varphi(z|x) \,\|\, p_\theta(z))$ — the *encoder's* distribution first,
the *prior* second.** [Part 2 §24](dimensionality-reduction-02.md) proved this direction is
**reverse KL** relative to the "true distribution being approximated" framing: here $q$ (the
approximation) is in the first slot and it's being pushed toward $p$ (playing the role of the "true"
target the network wants $q$ to resemble)... but read the mechanics literally: $\mathrm{KL}(q\|p)$
means the expectation is taken **under $q$**, which by Part 2 §24's derivation is **mode-seeking /
zero-forcing**: $q$ is not penalised for failing to cover parts of $p$, only for putting mass where $p$
has none.

**Translated into what actually happens during training:** the encoder is free to leave parts of the
prior's mass "unclaimed" — it doesn't have to spread every possible input's encoding to cover the
entire $N(0,I)$ ball — but it is *heavily* penalised if it places an input's distribution somewhere the
prior assigns near-zero density. **This is the mechanism behind posterior collapse** (§25): because
zero-forcing lets $q$ retreat to a small, safe region of the prior without penalty, the encoder can
(especially with a powerful decoder) learn to ignore parts of the latent space almost entirely, using
only a fraction of $z$'s capacity.

### 23.2 The tension, made concrete

**Reconstruction alone** wants each input's $q(z|x)$ to be a **sharp spike** — the narrower the
distribution, the more precisely the decoder knows exactly what $z$ it's working with, and the better
it can reconstruct $x$. Taken to the limit, reconstruction loss alone would collapse every $\sigma \to
0$, turning the VAE back into a plain (point-encoding) autoencoder — and reintroducing exactly the
"holes and gaps" problem from §20.

**KL regularisation alone** wants every input's $q(z|x)$ to **equal the prior exactly** — $\mu = 0,
\sigma = 1$ — which would make every input map to the *same* distribution, discarding all information
about $x$ and making reconstruction impossible.

**The two terms pull in opposite directions, and the loss is their sum**: the model is forced to find
distributions that are *as sharp as reconstruction demands, but no sharper than the prior comfortably
allows* — which is exactly the "smooth, continuous, well-covered" latent space the deck promises.

**$\beta$-VAE makes this tension a dial rather than a fixed compromise:** replace $D_{KL}$ with
$\beta \cdot D_{KL}$. Larger $\beta$ pushes harder toward the prior — a smoother, more disentangled but
blurrier latent space; smaller $\beta$ favours sharp reconstruction at the cost of some of the VAE's
generative smoothness. $\beta = 1$ recovers the original ELBO exactly.

### 23.3 The line that ties this directly to Part 2's Latent Diffusion Models

> *"this is exactly what Stable Diffusion's VAE encoder exploits"*

**This sentence is the bridge to §29–§30, and it is worth pausing on before moving forward.** Stable
Diffusion's generative process doesn't run in pixel space — it runs diffusion *inside* the smooth,
continuous latent space a VAE encoder produces. That only works because the KL term guarantees the
latent space has no holes: a diffusion process that wanders through latent space during its iterative
denoising needs every point it passes through, not just the exact points seen during training, to
decode to something coherent. **Without §23's KL term, latent diffusion as an architecture would not
work at all** — the diffusion process would routinely pass through the "garbage" regions §20 described.

---

## 26–28. VQ-VAE — Discrete Latent Codes

> *"Much real-world data is naturally discrete - so why force a continuous latent?"* [slide 41,
> 26:05]
>
> - *"**Data wants to be discrete:** An image becomes a sentence: a 32×32 grid of indices into a visual
>   'vocabulary' - exactly the discrete token format that **transformers** are built to model"*
> - *"**Huge capacity, no memorisation:** codebook 512 × a 32×32 grid → $512^{1024}$ possible images"*
> - *"**Why it matters:** this image-to-tokens step is what lets **DALL·E** and **Jukebox** run a
>   transformer over the discrete codes"*

[slide 43, 26:59]
> - *"**Why discrete?**"*
> - *"**Vector quantisation:** the encoder output $E(x) = z_e$ is snapped to its nearest vector in a
>   learned **codebook** $e \in \mathbb{R}^{K\times D}$ (a nearest-neighbour lookup)"*
> - *"**Straight-through gradient:** argmin is non-differentiable"*
> - *"**VQ loss:** the L2 error between the embedding space and the encoder outputs"*
> - *"**Commitment loss:** encourages the encoder output to stay close to the embedding space"*
>
> $$q(z=e_k|x) = \begin{cases}1 & \text{if } k = \arg\min_i \|z_e(x) - e_i\|_2 \\ 0 & \text{otherwise}\end{cases}$$
>
> $$L = \|x - D(e_k)\|^2 + \|\mathrm{sg}[E(x)] - e_k\|^2 + \beta\|E(x) - \mathrm{sg}[e_k]\|^2$$

| Symbol | Read it as | What it means |
|---|---|---|
| $E(x) = z_e$ | "E of x" | The encoder's raw (continuous) output — before quantisation |
| $e \in \mathbb{R}^{K\times D}$ | "the codebook" | $K$ learned vectors, each $D$-dimensional. The "vocabulary." |
| $e_k$ | "e sub k" | The specific codebook entry $z_e$ gets snapped to — whichever is nearest |
| $\mathrm{sg}[\cdot]$ | "stop-gradient" | Treat the contents as a constant during backpropagation — gradients do not flow through this term |
| $D(e_k)$ | "D of e sub k" | The decoder's reconstruction, built from the *quantised* code, not the raw encoder output |
| $\beta$ | "beta" | Weight on the commitment loss |

### 26.1 The three-term loss, unpacked one term at a time

**Term 1 — $\|x - D(e_k)\|^2$, the reconstruction loss.** Exactly §5's autoencoder loss, except the
decoder now reconstructs from the **quantised** code $e_k$ rather than the encoder's raw continuous
output.

**Term 2 — $\|\mathrm{sg}[E(x)] - e_k\|^2$, the codebook loss.** This trains the **codebook**, not the
encoder — the stop-gradient on $E(x)$ means this term's gradient only updates $e_k$, pulling that
codebook vector toward wherever the encoder currently outputs. It's a moving target the codebook chases.

**Term 3 — $\beta\|E(x) - \mathrm{sg}[e_k]\|^2$, the commitment loss.** The mirror image: stop-gradient
on $e_k$ this time, so this term only updates the **encoder**, pulling $E(x)$ toward the codebook
entry it's currently assigned to. Without this term, the encoder could drift arbitrarily far from any
codebook vector between updates, since quantisation always snaps to the *nearest* one regardless of
distance — the commitment loss keeps that distance small so quantisation doesn't throw away too much.

> 💡 **Why two nearly-symmetric terms rather than one.** Terms 2 and 3 look almost identical — both are
> squared distances between $E(x)$ and $e_k$ — but the stop-gradient placement means they train
> *different* parameters. This is a clean instance of a broader pattern worth recognising: when two
> objects need to converge toward each other but you want to control which one moves and by how much,
> split a single "make these close" objective into two stop-gradient-separated halves.

### 26.2 🧪 Derive Prerequisite 5's straight-through estimator, applied here

**The problem, concretely:** the operation "find $k = \arg\min_i\|z_e - e_i\|_2$, then output $e_k$" has
zero gradient with respect to $z_e$ almost everywhere (a small perturbation to $z_e$ usually doesn't
change which codebook entry is nearest) and an undefined gradient exactly at the tie-boundaries. There
is no way to backpropagate reconstruction loss through this operation to reach the encoder at all,
using ordinary calculus.

**The straight-through fix, stated precisely for this case:** on the *forward* pass, compute the true
quantised value $e_k$ and feed it to the decoder. On the *backward* pass, **copy the gradient arriving
at $e_k$ straight back to $z_e$**, as if the quantisation step had been the identity function
$e_k \approx z_e$. In code, this is usually written as:

```python
z_q = z_e + (e_k - z_e).detach()   # forward: z_q == e_k exactly
                                     # backward: gradient flows through as if z_q == z_e
```

**Why this is a biased-but-useful approximation, not a mathematically justified gradient.** The true
gradient of $\arg\min$ is (almost everywhere) zero; the straight-through estimator substitutes a
gradient of 1, which is simply *wrong* in the strict calculus sense. It works in practice because $e_k$
is, by construction (via the commitment loss, term 3), always *close* to $z_e$ — so treating them as
interchangeable for gradient purposes is a reasonable local approximation, not an exact one. This is a
deliberate, named trade-off, and knowing that it's an approximation (rather than believing it's somehow
exact) is the mark of actually understanding the technique.

### 27.1 Why discrete latents specifically enable a Transformer over images

> *"An image becomes a sentence: a 32×32 grid of indices into a visual 'vocabulary' - exactly the
> discrete token format that transformers are built to model"*

**A Transformer's native input is a sequence of discrete tokens** — that's what a vocabulary and an
embedding lookup table are built around, and it's the format every LLM training pipeline expects.
VQ-VAE's output — a 32×32 grid of codebook indices, each just an integer from 0 to $K-1$ — **is exactly
that format**, once flattened into a sequence. So a VQ-VAE turns "generate an image" into "predict the
next token in a sequence of 1,024 integers," which is a problem a standard autoregressive Transformer
(the same architecture behind GPT) can be pointed at directly, with no architectural modification.

*Concretely:* $K = 512$ codebook entries, a $32\times32 = 1024$-token grid gives
$512^{1024}$ possible images — a number too large to write out in ordinary notation, and vastly larger
than any conceivable training set, which is exactly the "huge capacity, no memorisation" bullet: the
model cannot simply memorise a lookup table of images because the space of representable images
dwarfs any dataset by an astronomical margin.

> 💡 **This is the mechanism behind DALL-E (the original, 2021) and Jukebox** — both train a VQ-VAE (or
> a close variant) first to turn images (or audio) into discrete token grids, then train a standard
> Transformer as an autoregressive language model **over those tokens**. "Generate an image" becomes,
> after the VQ-VAE step, formally identical to "generate text" — same architecture, same training
> recipe, different tokenizer.

---

# PART 2 — Latent Diffusion Models

*31:07 – 37:27*

*Subtitle on the deck's own divider slide: "From Dimensionality Reduction to Generation."*

---

## 29. Why Pixel-Space Diffusion is Prohibitively Expensive

> [slide 50, 31:12]
>
> - *"**Diffusion = iterative denoising:** Model predicts noise $\epsilon$ added to data; runs 20–1000
>   forward passes to generate one image"*
> - *"**Pixel-space cost:** Each pass operates on the full image tensor - a 512×512 RGB image =
>   **786,432 values** per step × 50 steps = ~39M operations per generation"*
> - *"**Quadratic attention cost:** U-Net self-attention at high resolution is $\mathcal{O}(n^2)$ in
>   spatial tokens - 512² = **262,144 tokens** → memory explodes"*
> - *"**The solution:** Don't run diffusion in pixel space - compress first, denoise in latent space,
>   then decode. This is Latent Diffusion (LDM)."*

### 29.1 📚 Background the slide assumed — what "diffusion" means, taught from zero

> **Diffusion model** — a generative model trained to reverse a gradual noising process.
>
> *In everyday words:* imagine slowly dissolving a photograph into pure static, one small step at a
> time, over (say) 1000 steps. A diffusion model is trained to run that process **backwards**: starting
> from pure static, predict and remove a small amount of noise, repeatedly, until a coherent image
> emerges.
>
> *Concretely:* the **forward process** (fixed, no learning involved) takes a real image $x_0$ and
> adds a small amount of Gaussian noise at each of $T$ steps, producing progressively noisier versions
> $x_1, x_2, \ldots, x_T$, until $x_T$ is indistinguishable from pure noise. The **reverse process**
> (the part that's *learned*) trains a network to predict, at each step, either the noise that was
> added or the slightly-less-noisy version — undoing one step of corruption at a time.
>
> *Why iterative, rather than one shot?* Directly learning "map pure noise to a coherent image" in a
> single step is an extremely hard function to learn — it has to encode the entire structure of
> natural images in one pass. Breaking it into hundreds of *small* denoising steps makes each
> individual step easy (predicting a small amount of noise is far simpler than generating an entire
> image), at the cost of needing many steps to generate one sample.

The deck's own figure shows this directly: a row of dog photographs progressing left to right from
"Data" (a clear photo) to "Noise" (pure static), labelled **"Forward diffusion process (fixed)"**
above and **"Reverse denoising process (generative)"** below, with the arrow running the opposite
direction.

### 29.2 🧪 Verify the cost numbers, and see exactly where they come from

**Pixel-space cost, per the slide:** a $512\times512$ RGB image has

$$512 \times 512 \times 3 = \mathbf{786{,}432} \text{ values}$$

*(the "×3" is the three RGB channels, implicit in the slide's number — check: $512^2 = 262{,}144$,
and $262{,}144 \times 3 = 786{,}432$ ✓)*. At 50 denoising steps, the network processes this full tensor
50 times over the course of generating **one single image**.

**Attention cost, separately.** A U-Net's self-attention layers, when applied at the image's full
spatial resolution, treat each of the $512^2 = 262{,}144$ spatial positions as one token in a sequence,
and self-attention costs $\mathcal{O}(n^2)$ in sequence length $n$:

$$n^2 = (262{,}144)^2 = 6.87 \times 10^{10} \text{ attention-score computations}$$

**That is nearly 69 billion pairwise scores, for one attention layer, at one denoising step.** This is
the mechanism behind "memory explodes" — it isn't a vague warning, it's a specific quadratic blow-up
that becomes unaffordable well before you reach a resolution most people would call "high."

> 💡 **Two separate costs, worth keeping distinct, because they respond differently to a fix.** The
> "786,432 values × 50 steps" cost is **linear** in image size — it scales with the number of pixels.
> The attention cost is **quadratic** in the number of spatial tokens. §30's fix — shrinking the spatial
> resolution before running diffusion — helps the linear cost proportionally, but helps the *quadratic*
> attention cost by the **square** of that same factor. That's why §30's "even larger" saving on the
> attention side isn't a coincidence — it's what quadratic scaling does to any shrinkage.

---

## 30. Latent Diffusion Model

> [slide 52, 32:38] with its own labelled diagram: **Pixel Space** → $\mathcal{E}$ (encoder) →
> **Latent Space** → the **Denoising U-Net** $\epsilon_\theta$ (with cross-attention, switch, skip
> connection, concat) → $\mathcal{D}$ (decoder) → back to **Pixel Space**.
>
> - *"Runs the diffusion process in the **latent space** instead of pixel space"*
> - *"**2 Stage Training:** Auto-Encoder + Latent Diffusion"*

### 30.1 The two-stage training the diagram implies

**Stage 1 — train a VAE (or VQ-VAE), exactly as in §19–§28, to compress images into a smooth latent
space and back.** This is trained *first*, independently, purely on the reconstruction + KL objective
from §23. Once trained, its weights are typically **frozen**.

**Stage 2 — train the diffusion process entirely inside that fixed latent space.** The forward
(noising) and reverse (denoising) processes described in §29.1 both operate on the *latent* code $z$
that the frozen VAE encoder produces, not on raw pixels. The U-Net predicting noise takes a noisy
latent as input and predicts the noise that was added to it, in latent coordinates.

**At generation time**, you run the full reverse-diffusion loop entirely in latent space — no image
tensor is touched during the 20–1000 iterative steps — and only decode back to pixels **once**, at the
very end, via the frozen decoder $\mathcal{D}$.

> 💡 **This is precisely why §23's closing sentence — "this is exactly what Stable Diffusion's VAE
> encoder exploits" — is not a throwaway remark.** The entire architecture depends on the VAE's latent
> space being smooth and hole-free (§20, §23), because the diffusion process spends hundreds of steps
> wandering through *every* intermediate point between pure noise and a final coherent latent — not
> just the exact latent codes seen during VAE training. A latent space with the AE-style gaps from §20
> would cause the denoising trajectory to repeatedly pass through regions that decode to garbage.

### 🎯 Quick check — the compute saving, computed exactly

> [slides 56–57, 35:46–36:49]
>
> **Stable Diffusion's VAE uses a downsampling factor of $f = 8$. A 512×512 image becomes a 64×64
> latent. How much cheaper is diffusion in this latent space (spatially)?**
>
> A. $8\times$ fewer elements — one factor of $f$
> B. $64\times$ fewer elements — $f^2$ because both height and width shrink
> C. No cheaper - the VAE adds overhead that cancels the savings

<details>
<summary><b>Answer</b></summary>

**B.** The slide: *"512×512 = 262,144 vs 64×64 = 4,096 spatial positions → **64× fewer**. With
quadratic attention cost, the saving is even larger. This is why latent diffusion is tractable."*

**Verify:** $512/8 = 64$ in each spatial dimension, so the *area* shrinks by $8^2 = \mathbf{64}$, not
by 8. $512^2 = 262{,}144$, and $64^2 = 4{,}096$; $262{,}144 / 4{,}096 = \mathbf{64}$ ✓.

**And now apply §29.2's distinction: this 64× is the *linear* saving. The attention cost, being
quadratic in the token count, shrinks by $64^2 = \mathbf{4{,}096\times}$** — from
$(262{,}144)^2 \approx 6.87\times10^{10}$ down to $(4{,}096)^2 \approx 1.68\times10^7$, a genuinely
enormous reduction that is the real reason self-attention becomes affordable to use inside the
diffusion U-Net at all.

**Why A is wrong:** it mistakes a *linear* dimension's shrinkage factor ($8\times$ in height, $8\times$
in width) for the shrinkage in *element count*, forgetting that area is the product of two shrunk
dimensions.
</details>

---

## 30.2 Latent Diffusion Extends to Video — Sora, Veo, and Beyond

> [slide 54, 34:12]
>
> - *"**Video = image sequence:** A 10s clip at 24fps × 1080p = ~2.5B pixel values - pixel-space
>   diffusion is completely infeasible for video"*
> - *"**Spatiotemporal VAE:** Compress both spatial (H×W) AND temporal (T frames) dimensions; Sora
>   uses 3D patch VAE compressing 128×720×1280 → much smaller latent cube"*
> - *"**Diffusion Transformer (DiT):** Sora replaces U-Net with a pure Transformer over the 3D latent
>   tokens - scales better with compute and sequence length"*
> - *"**Sora (OpenAI, 2024):** treats video as sequences of spacetime patches in latent space; same
>   architecture can generate variable-length, variable-resolution videos"*
> - *"**Veo (Google DeepMind, 2024):** Similar latent video diffusion approach; achieves 1080p, 60s
>   clips; emphasises physics consistency"*
> - *"**Key insight:** VAE compression is what makes all of this tractable - without it, video
>   generation would require 1000× more compute"*

### 💡 Why this section belongs here at all, and what it adds beyond §30

**The core idea doesn't change from image to video — it generalises along one more axis.** An image's
latent has spatial dimensions (height × width); a video's latent adds a **temporal** dimension (number
of frames) on top. A "spatiotemporal VAE" is exactly §30's VAE, extended so its encoder/decoder
compress *three* axes (H, W, T) instead of two, producing a latent that is a compact 3-D cube rather
than a 2-D grid.

**The Diffusion Transformer (DiT) detail is worth a moment**, since it's the one genuinely new
architectural fact in this subsection: rather than the U-Net used in §30's diagram, Sora runs the
diffusion process's denoising network as a **plain Transformer** operating over the flattened
spacetime-patch tokens. This is the same "an image becomes a sequence of tokens a Transformer can
process" move from §27.1's VQ-VAE discussion, applied to the *diffusion* network's architecture rather
than to the tokenisation of the data itself — and it inherits the usual Transformer benefit of scaling
predictably with more compute and data, which a convolutional U-Net does not do as cleanly.

**The video math, verified:** a 10-second clip at 24fps has $10\times24 = 240$ frames; at 1080p
($1920\times1080$) each frame has $1920\times1080 = 2{,}073{,}600$ pixels. Total:
$240 \times 2{,}073{,}600 \approx 4.98\times10^8$ pixel *positions*, and with 3 colour channels,
$\approx 1.49\times10^9$ values — on the same order of magnitude as the slide's "~2.5B" figure (the
discrepancy is likely a different frame-rate/resolution assumption; the conclusion — pixel-space
diffusion on video is off the table by any reasonable accounting — is unaffected). **Every one of
§29's arguments against pixel-space diffusion applies to video with the numbers made roughly 240×
worse**, which is exactly why the slide's closing claim — "without [VAE compression], video generation
would require 1000× more compute" — is entirely plausible rather than an exaggeration.

---

# PART 3 — Low-Rank Methods in Transformers

*37:27 – 45:43*

---

## 31–32. LoRA: Low-Rank Adaptation

> [slide 60, 37:31]
>
> - *"**The problem:** full fine-tuning of an LLM updates *all* weights $W \in \mathbb{R}^{d\times k}$ -
>   billions of parameters, a full optimizer state, and a separate copy per task"*
> - *"**Key insight:** the weight *update* $\Delta W$ during adaptation has a low intrinsic rank - it
>   can be approximated by a product of two small matrices"*
> - *"**The method:** freeze $W$, learn only $\Delta W = B\cdot A$ where $B \in \mathbb{R}^{d\times r}$,
>   $A \in \mathbb{R}^{r\times k}$ and rank $r \ll \min(d,k)$"*
> - *"**Forward pass:** $h = Wx + BAx$; A starts random, B starts at zero so training begins exactly at
>   the pretrained model"*
> - *"**Why it wins:** up to ~10,000× fewer trainable parameters, no added inference latency (BA can be
>   merged into W), and a tiny swappable adapter per task"*
>
> $$W + \Delta W = W + B\cdot A, \qquad r \ll \min(d,k)$$

The deck's own diagram makes the shape of the idea immediate: **"Weight update in regular finetuning"**
shows a single large blue block labelled $\Delta W$ sitting beside the frozen grey **Pretrained
weights** block, both full size $d \times k$. **"Weight update in LoRA"** shows the same frozen grey
block, but the update is now a thin blue triangle labelled $B$ feeding into a thin blue triangle
labelled $A$ — an hourglass shape — with the caption **"The inner dimension $r$ is a hyperparameter."**

| Symbol | Read it as | What it means | Shape |
|---|---|---|---|
| $W$ | "W" | The pretrained weight matrix. **Frozen** — never updated during LoRA fine-tuning. | $d\times k$ |
| $\Delta W$ | "delta W" | The weight *update* fine-tuning would otherwise apply directly to $W$ | $d\times k$ |
| $B$ | "B" | One of the two low-rank factors | $d\times r$ |
| $A$ | "A" | The other low-rank factor | $r\times k$ |
| $r$ | "r" | The **rank** of the approximation — a hyperparameter, typically 4–64 in practice | — |
| $h = Wx + BAx$ | "h" | The layer's output: the frozen path plus the learned low-rank correction, added together | — |

### 32.1 Connect the key insight directly back to Part 2's SVD

**"The weight update $\Delta W$ has low intrinsic rank" is [Part 2's](dimensionality-reduction-02.md)
central empirical claim about data matrices, now made about a *weight update* instead.** Part 2 §2–§4
established that any matrix can be written as a sum of rank-1 pieces ordered by importance
($X = \sum_i \sigma_i u_iv_i^\top$), and that if a matrix is *genuinely* low-rank, a truncated version
using only the first few terms reconstructs it almost exactly (Eckart–Young: provably optimal).
**LoRA's entire justification is the empirical observation (Hu et al., 2021, and Aghajanyan et al.,
2020 before it, on the "intrinsic dimension" of fine-tuning) that $\Delta W$ — not the weights
themselves, specifically the *change* fine-tuning wants to make — behaves like a low-rank matrix.**
That means a rank-$r$ factorisation $B\cdot A$, with $r$ perhaps 16 or 64 against a $d, k$ in the
thousands, can capture nearly all of what full fine-tuning would have learned.

**Why does the update, specifically, have low rank, when the base weights $W$ themselves are full
rank?** The intuition the field has converged on: a pretrained LLM already encodes broad, general
capabilities across its full-rank weight space. Adapting it to a *specific* downstream task — say,
answering questions in a particular style, or following a particular instruction format — is a much
narrower, more specialised change than "learn language from scratch," and narrow changes are exactly
what low-rank matrices are good at representing cheaply. This is an empirical finding, not a proven
theorem — flagged honestly rather than asserted as settled.

### 32.2 🧪 Compute LoRA's parameter savings for a real layer

Take a Transformer attention projection matrix, a common LoRA target, of size $d = k = 4096$ (a
mid-size LLM's hidden dimension), with LoRA rank $r = 8$.

**Full fine-tuning of this one matrix:**

$$d \times k = 4096 \times 4096 = \mathbf{16{,}777{,}216} \text{ parameters}$$

**LoRA's two factors:**

$$B: d\times r = 4096\times8 = 32{,}768 \qquad A: r\times k = 8\times4096 = 32{,}768$$
$$\text{total} = 32{,}768 + 32{,}768 = \mathbf{65{,}536} \text{ parameters}$$

$$\text{reduction} = \frac{16{,}777{,}216}{65{,}536} = \mathbf{256\times}$$

**For a full model with, say, 200 such matrices across all layers** (attention $Q,K,V,O$ projections
across many layers is a realistic count for a modern LLM), the same $256\times$ ratio applies
uniformly, which is consistent with the deck's own "up to ~10,000×" figure at more aggressive ranks or
larger base dimensions — check: at $d=k=12{,}288$ (a much larger model) and $r=4$, the ratio becomes
$\frac{12{,}288^2}{2\times12{,}288\times4} = \frac{12{,}288}{8} = \mathbf{1{,}536}\times$, and the
"~10,000×" figure the deck quotes is achievable at even larger $d$ or smaller $r$. **The exact ratio
depends heavily on the chosen $d, k, r$ — treat the deck's "~10,000×" as an upper-end figure, not a
universal constant.**

### 32.3 Why $B$ starts at zero — reasoned, not just stated

> *"$A$ starts random, $B$ starts at zero so training begins exactly at the pretrained model, adding no
> disruption"*

**Trace what this means for the very first forward pass.** With $B = 0$:

$$h = Wx + BAx = Wx + 0\cdot Ax = Wx$$

**The adapter contributes exactly nothing on step zero.** The model's output is *bit-for-bit identical*
to the frozen pretrained model before any training has happened — there is no random, disruptive jolt
introduced by initialising the adapter, which there would be if both $A$ and $B$ started random (their
product $BA$ would then be some arbitrary nonzero matrix, corrupting the pretrained behaviour from step
one).

**Why not initialise both to zero, then, for maximum safety?** §33's quiz addresses this directly —
the short version is that if both start at zero, the gradient with respect to *either* matrix alone is
also zero (each factor's gradient involves a product with the other factor, which is zero), so nothing
would ever begin learning. One of the two matrices must be random purely to **break the symmetry** and
give the optimiser somewhere to start.

### 🎯 Quick check — LoRA's asymmetric initialisation, tested

> [slides 63–64, 40:15–41:43]
>
> **In LoRA, A is initialized randomly but B is initialized to exactly zero. Why start B at exactly
> zero?**
>
> A. To save memory - zeros compress better than random values
> B. So $\Delta W = BA = 0$ at the start - training begins exactly at the pretrained model, adding no
>    disruption
> C. Because both A and B at zero would make gradients vanish

<details>
<summary><b>Answer</b></summary>

**B.** The slide: *"With $B = 0$, $BA = 0$, so the adapter is initially a no-op and the model output is
identical to the frozen pretrained model. If both A and B were zero, $\partial L/\partial A$ and
$\partial L/\partial B$ would both be zero and nothing would ever learn - so one (A) must be random to
break symmetry."*

**Notice the answer folds C's premise directly into justifying B** — the reason it can't be "both
zero" (which would defeat C's own premise about vanishing gradients being the reason) is exactly why
*one* of the two, specifically $A$, must be random. C is a plausible-sounding distractor that gets the
mechanism half right (gradients would indeed vanish if both were zero) but draws the wrong conclusion
from it (the fix isn't "avoid zero entirely," it's "keep exactly one factor at zero, for the no-disruption
property, and randomise the other, to preserve a nonzero gradient").

**A is a genuine non sequitur** — nothing about LoRA initialisation is motivated by storage compression
of the initial values; the entire discussion is about training *dynamics* at step zero, not disk space.
</details>

```interactive
type: slider
title: LoRA rank vs parameter count
concept: How the rank r trades off adapter size against expressiveness
control: Sliders for d=k (512 to 16384) and rank r (1 to 256), plus a toggle showing B at zero vs B randomised
observe: A live bar comparing full fine-tuning's d×k parameter count against LoRA's r(d+k), with the exact reduction ratio printed; and, with the B toggle, the model's output on a fixed test input either matching the frozen base model exactly (B=0) or diverging immediately (B random)
insight: The reduction ratio is d/(2r) when d=k, so doubling the model's hidden size doubles the savings at fixed rank, while doubling the rank only halves the savings — rank is the far more sensitive dial, which is why practitioners tune r first
fallback: The two worked examples above: d=k=4096, r=8 gives 256x; d=k=8192, r=16 gives the same 256x, because both share d/(2r)=256
```

---

## 35. DoRA — Variants & Improvements

> *"DoRA = Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)"* [slide 65, 41:48]
>
> - *"**Core idea:** decompose each pretrained weight into a *magnitude* (a scalar per column) and a
>   *direction* (a unit vector): $W = m \cdot (V / \|V\|)$"*
> - *"**Where LoRA goes:** the low-rank update $BA$ is applied only to the *direction*, while the
>   magnitude is trained as a separate, independent parameter"*
> - *"**Why it helps:** full fine-tuning naturally varies magnitude and direction independently; plain
>   LoRA couples them. Decoupling makes DoRA's learning pattern closer to full fine-tuning"*
> - *"**Result:** consistently beats LoRA at the same parameter budget, with no extra inference cost
>   (it merges back into W)"*
>
> $$W' = m \cdot \frac{V + B\cdot A}{\|V + B\cdot A\|}$$

The deck's own diagram walks this as two explicit steps: **Step 1, decompose** the pretrained weight
matrix $W_0$ into a magnitude vector $m$ (one scalar per column, $m = \|W_0\|_c \in \mathbb{R}^{1\times
k}$) and a directional matrix $V$ (the normalised weight matrix, $V = W_0 \in \mathbb{R}^{d\times k}$)
— and **Step 2, finetune** the model with LoRA applied *only* to the directional part, producing an
adapted direction $V + \Delta V$, while the magnitude $m$ is trained directly as its own free
parameter.

### 35.1 🧪 Derive the decomposition, and see exactly what LoRA was conflating

**Any weight matrix can be split into a magnitude and a direction, per-column, by definition:**

$$W_0 = m \cdot \frac{V}{\|V\|_c}, \qquad \text{where } m = \|W_0\|_c \text{ and } V = W_0$$

This isn't an approximation — it's an algebraic identity, using $\|\cdot\|_c$ to denote the column-wise
norm. Every column of $W_0$ can trivially be written as (its own length) × (its own unit direction).

**Plain LoRA's forward pass is $h = Wx + BAx = (W + BA)x$** — the low-rank update $BA$ is added directly
to $W$ in its raw form, **without distinguishing whether that update is changing $W$'s magnitude, its
direction, or some mix of both.** A single low-rank matrix $BA$ necessarily entangles both kinds of
change together, because nothing in LoRA's formulation treats them as separate objects.

**DoRA's forward pass separates them explicitly.** The magnitude $m$ becomes its own trainable
parameter (a small vector, one scalar per output column — cheap), and the low-rank update $BA$ is
applied *exclusively* inside the direction term:

$$W' = m \cdot \frac{V + BA}{\|V + BA\|_c}$$

**Read what this buys, concretely.** If a task genuinely requires the pretrained model to output larger
(or smaller) values along some direction, without meaningfully rotating that direction, DoRA can
represent that as a **pure change in $m$** — cheap, direct, unconstrained by rank. Plain LoRA has no
equivalent shortcut: any change, magnitude or direction, has to be squeezed through the same rank-$r$
bottleneck $BA$, whether or not a magnitude-only change is what the task actually calls for.

### 35.2 Why the empirical claim is plausible, and what to say honestly about it

> *"full fine-tuning naturally varies magnitude and direction independently; plain LoRA couples them"*

**This is an empirical observation about how full fine-tuning behaves, reported in Liu et al. (2024),
not a mathematical necessity.** The paper's contribution was measuring the magnitude-vs-direction
update pattern of full fine-tuned models and *comparing* it against LoRA's pattern, finding LoRA's
correlation between magnitude and direction changes differs systematically from full fine-tuning's —
then designing DoRA specifically to close that gap. **The "consistently beats LoRA at the same
parameter budget" result is an empirical finding across the benchmarks in that paper, not a guarantee
that generalises to every task** — the honest framing is "DoRA's inductive bias (decoupled
magnitude/direction) matches full fine-tuning's observed behaviour more closely than LoRA's does, and
this measurably helps on the tasks tested."

### The lineage — LoRA → QLoRA → DoRA, one diagram

The deck's closing figure for this part [slides 68–71] places three approaches on one axis —
**Full Finetuning (No Adapters)**, **LoRA**, **QLoRA** — each annotated with its optimizer-state size
(32-bit), adapter size (16-bit), and base-model precision (16-bit for full/LoRA, **4-bit for QLoRA**),
with the caption:

> *"LoRA, QLoRA, and DoRA exploit the observation that task-specific adaptations occupy a much
> lower-dimensional space than the full model parameter space, enabling efficient fine-tuning with
> minimal trainable parameters."*

**QLoRA's addition, not otherwise detailed on this slide, is worth naming precisely**: it quantises the
*frozen* base model down to 4-bit precision (via a technique called NF4, "NormalFloat4") to shrink its
memory footprint drastically, while keeping the LoRA adapters themselves in a higher-precision format
(commonly bfloat16) so the small number of trainable parameters retain enough numerical precision to
train well. **QLoRA and DoRA solve different problems and can be combined** — QLoRA reduces the *base
model's* memory footprint, DoRA improves the *adapter's* learning dynamics — which is why the deck
groups all three under one umbrella sentence rather than presenting them as competing alternatives.

> 🎯 **Interview-ready summary of the whole family:** LoRA factors a low-rank *update*; QLoRA
> additionally quantises the *frozen base* to fit training on smaller hardware; DoRA decouples
> *magnitude from direction* within the low-rank update itself, to better match how full fine-tuning
> actually behaves. All three merge back into the base weights at inference time, so **none of them
> add any inference latency** compared to the original model — this is a genuinely distinguishing
> property versus, say, adapter layers that stay as separate modules at inference.

---

# PART 4 — Embedding Models & Matryoshka Representations

*45:43 – 56:28*

---

## 36. Modern Text Embedding Models — The Landscape

> [slide 73, 46:20]
>
> - *"**Embeddings:** Map variable-length text → fixed-size dense vector"*
> - *"**OpenAI text-embedding-3-small/large:** 1536/3072 dims; trained with contrastive learning on
>   massive text pairs; first OpenAI model to support **MRL**"*
> - *"**Nomic Embed (2024):** Open-source, fully reproducible, 768-dim; supports long context (8192
>   tokens); matches OpenAI quality on MTEB benchmarks"*
> - *"**BGE (BAAI):** State-of-art open models (BGE-M3); multilingual, multi-granularity; supports
>   dense, sparse, and multi-vector retrieval simultaneously"*

> **Embedding** — a learned function mapping a piece of text (of any length) to a single fixed-size
> vector of real numbers, such that texts with similar meaning map to nearby vectors.
>
> *Why "fixed-size" matters:* downstream systems — vector databases, similarity search, clustering —
> need a consistent number of dimensions to compare vectors at all. A sentence and a paragraph, of
> wildly different lengths, must both become (say) exactly 1536 numbers.

**The landscape bullet is context for §37's problem, not content to memorise** — the specific model
names establish that this is a genuinely competitive, actively-developed area (contrastive learning
across "massive text pairs," open reproducible alternatives, multi-granularity retrieval), which is
what makes the fixed-dimension limitation in §37 a live production concern rather than an academic
curiosity.

---

## 37. The Problem: Fixed Embeddings Don't Scale

> [slide 75, 47:29]
>
> - *"**Standard training commits to one size:** A 3072-dim model is always 3072-dim - no cheap way to
>   get a 256-dim version without training from scratch"*
> - *"**The cost problem:** Storing 100M documents at 3072 dims (float32) = ~1.2TB; 3× more than needed
>   for many downstream tasks"*
> - *"**The quality problem:** Smaller separately-trained models underperform - they never saw
>   large-scale contrastive training; quality degrades faster than expected"*
> - *"**The maintenance problem:** Different use cases (real-time search vs batch clustering) need
>   different sizes; you'd maintain a zoo of separate models"*
> - *"**The inconsistency problem:** Doc embedded with model-256 and query embedded with model-1024 →
>   incompatible vector spaces; can't mix"*
> - *"**What we want:** One model that produces embeddings where any prefix is a valid, high-quality
>   embedding at that dimension — nested representations"*

### 37.1 🧪 Verify the cost problem's arithmetic

$$100{,}000{,}000 \text{ documents} \times 3072 \text{ dims} \times 4 \text{ bytes (float32)} = 1.229 \times 10^{12} \text{ bytes} \approx \mathbf{1.23 \text{ TB}}$$

Matching the slide's "~1.2TB" ✓. **And "3× more than needed" is a direct comparison against a 1024-dim
alternative:** $3072 / 1024 = \mathbf{3\times}$ exactly — the slide's specific ratio, not a rough
estimate.

### 37.2 Five separate problems, worth distinguishing rather than treating as one

**This is the section's real teaching content — recognising these as five *independent* failure modes,
not restatements of the same complaint, is what separates a shallow read from a useful one.**

| Problem | What specifically fails | Why "just retrain smaller" doesn't fix it |
|---|---|---|
| **Cost** | Storage and compute scale linearly with dimension across the *whole corpus* | Retraining smaller helps cost, but reintroduces every problem below |
| **Quality** | A model trained from scratch at 256 dims underperforms a 256-dim *prefix* of a well-trained 3072-dim model | Small models miss out on the large-scale contrastive training regime that made the big model good in the first place |
| **Maintenance** | Different downstream use cases genuinely want different sizes | You'd need to separately train, version, and serve a whole *family* of models |
| **Inconsistency** | A document embedded at one size and a query embedded at a different size **cannot be compared at all** — they aren't even the same vector space | This isn't a quality degradation, it's a hard incompatibility; cosine similarity between a 256-dim and a 1024-dim vector is not even defined |
| **What we want** | A *single* trained model whose embeddings can be truncated to any smaller size on demand, with no incompatibility and minimal quality loss | This is precisely what §38's Matryoshka training solves |

> 💡 **The inconsistency problem is the sharpest of the five, and worth being able to state precisely in
> an interview.** It is not a matter of degree — a 256-dim vector and a 1024-dim vector from two
> differently-sized *independently trained* models don't share a coordinate system at all, so
> similarity comparisons between them are meaningless, not just noisy. This is exactly why "train five
> separate small models" is not a real solution to the maintenance problem — it creates the
> inconsistency problem in its place.

---

## 38. Matryoshka Representation Learning (MRL)

> [slides 77–98, 48:59–50:41 — a single content slide the instructor spoke over at length; every
> bullet below is transcribed from that one slide]
>
> - *"**The question that sparked MRL:** what is the best embedding dimensionality d? It is not the
>   same for every data point - an easy image needs few dims, a hard one needs many"*
> - *"**Core idea:** bake lower-dimensional embeddings *inside* the full embedding, like nested dolls -
>   $z_{1:8}$ sits inside $z_{1:16}$ sits inside … inside $z_{1:2048}$"*
> - *"**Surprise result:** MRL dolls *outperform* independently trained models at each dimension -
>   across vision, language, and vision+language"*
> - *"**Beats post-hoc PCA:** traditional reduction runs *after* the encoder and is not data-aware; MRL
>   learns the nested structure *during* training"*
> - *"**'Remove numbers from the end':** accuracy interpolates smoothly at any doll size, even sizes
>   never trained on"*

> **Matryoshka Representation Learning** — named after the Russian nesting dolls: a training method
> that produces one embedding vector such that **every prefix** of it — the first 8 numbers, the first
> 16, the first 32, and so on — is *independently* a complete, high-quality embedding on its own.
>
> *In everyday words:* imagine a single 2048-digit summary of a document, constructed so cleverly that
> reading just the first 8 digits already gives you a usable (if coarse) summary, the first 16 gives
> you a better one, and so on up through all 2048 — with no re-training and no re-encoding required to
> get any of the intermediate summaries.
>
> *Concretely:* the "smallest doll" $z_{1:8}$ (the first 8 numbers of the embedding) is trained to be a
> valid 8-dimensional embedding **in its own right** — good enough to do retrieval or classification
> with, on its own, if that's all you keep. $z_{1:16}$ (the first 16 numbers, which *includes* those
> same first 8) is separately trained to be a valid 16-dimensional embedding, and so on.

> 👉 **See also.** [`GenAI & LLM` Part 3 §13](../GenAI%20&%20LLM/genai-llm-03.md) covers this same
> method from the applied side — truncating a production text-embedding model's output, with an
> MTEB-average curve and a worked nearest-neighbour retrieval example at 48× truncation — worth
> reading alongside this section for the "why" and the "how you'd actually ship it" halves of the
> same idea.

### 38.1 Why "the best dimensionality isn't the same for every data point" is the motivating insight

**This framing deserves attention because it reframes what a fixed-size embedding was implicitly
assuming.** Training one model that always outputs, say, 1024 numbers per input implicitly assumes
every input *deserves* 1024 numbers' worth of representation — an easy, unambiguous image (a clear
photo of a single cat, centred, well-lit) and a genuinely hard, ambiguous one (a cluttered scene with
several overlapping concepts) are given exactly the same representational budget, whether or not that
budget is what each actually needs. **MRL doesn't decide per-example how many dimensions to use** — it
instead makes that decision available to the *downstream consumer* at inference time, by guaranteeing
every truncation point is meaningful, rather than pre-committing to one fixed size for every input
uniformly.

### 38.2 🧪 Derive the nested loss, and see why it forces the ordering it claims to

> [slide 99, 50:45]
>
> - *"**Surprisingly simple:** apply the *same* loss you would use for the full embedding to each doll
>   independently, then average - no new loss function"*
> - *"Each doll is forced to be a valid representation on its own, otherwise that doll's loss stays
>   high"*
> - *"**Coarse-to-fine packing:** small-doll losses push the most important information into the front
>   dimensions; larger dolls add finer nuance"*
> - *"Trained only at logarithmic granularities (e.g. 8, 16, …, 2048), yet accuracy **interpolates** to
>   every size in between"*
>
> $$\mathcal{L}_{\text{Regular}} = \mathcal{L}(z_{1:2048})$$
> $$\mathcal{L}_{\text{Matryoshka}} = \text{average}\!\left(\mathcal{L}(z_{1:8}) + \mathcal{L}(z_{1:16}) + \cdots + \mathcal{L}(z_{1:2048})\right)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\mathcal{L}$ | "L" | Whatever loss the task normally uses — e.g. a contrastive retrieval loss, or a classification cross-entropy |
| $z_{1:m}$ | "z one to m" | The first $m$ coordinates of the full embedding vector — a **prefix**, not an independently-computed subvector |
| $\mathcal{L}_{\text{Regular}}$ | "L regular" | Ordinary training: only the full-size embedding is ever evaluated by the loss |
| $\mathcal{L}_{\text{Matryoshka}}$ | "L Matryoshka" | The *same* loss function, applied to **every** prefix length, then averaged |

**Derive why this produces the claimed ordering property — the mechanism, not just the recipe.**
Consider what gradient descent must do to make $\mathcal{L}_{\text{Matryoshka}}$ small. Every term in
the sum, including $\mathcal{L}(z_{1:8})$, must independently be driven down — and $z_{1:8}$ has *only
eight numbers* to work with, no matter what the other 2040 dimensions contain. **The eight numbers that
end up occupying positions 1–8 are therefore under direct, dedicated pressure to be maximally useful
completely on their own** — they cannot rely on later dimensions to "help out," because the
$\mathcal{L}(z_{1:8})$ term in the loss never sees those later dimensions at all. By the same argument
applied at every logarithmic checkpoint (16, 32, ..., 2048), **the most broadly useful information is
squeezed toward the front of the vector, because the front is the part every single term in the sum
depends on**, while dimensions near the end are only ever evaluated by the largest-prefix terms and can
afford to encode narrower, finer-grained detail. This is the "coarse-to-fine packing" bullet, derived
rather than asserted.

**And this also explains the "no new loss function" claim precisely.** Nothing about $\mathcal{L}$
itself changed — MRL doesn't invent a new objective, a new architecture, or a new training signal. It
only changes **what the existing loss is evaluated on**: instead of computing $\mathcal{L}$ once, on
the full vector, it computes the identical $\mathcal{L}$ several times, on nested prefixes of that same
vector, and averages the results. **The entire method is a training-time bookkeeping change, not a new
piece of machine learning theory** — which is exactly why it's described as "surprisingly simple."

### 38.3 Why interpolation to untrained sizes works, and what it doesn't guarantee

> *"Trained only at logarithmic granularities (e.g. 8, 16, …, 2048), yet accuracy interpolates smoothly
> to every size in between, even sizes never trained on"*

**The coarse-to-fine packing argument from §38.2 explains this too, without needing anything further.**
If the front dimensions genuinely carry the most broadly useful information and later dimensions add
progressively finer detail, then truncating at, say, $z_{1:12}$ — a size never directly evaluated
during training — should still capture "most of the useful information from the first 16, minus a
little of the finest detail from dimensions 9–16." **The smoothness is a consequence of the *ordering*
the training procedure induces, not a separately engineered guarantee**, which is why it generalises to
untrained truncation points at all.

> ⚠️ **What this does not claim.** Interpolation being *smooth* is not the same as it being *optimal* —
> a size never directly trained on (e.g. 12, sitting between the trained checkpoints 8 and 16) will
> typically underperform an equivalently-sized doll that *was* one of the explicit logarithmic training
> targets, even if the degradation is gentle rather than a cliff. The deck's claim is about graceful
> degradation, not about untrained sizes matching trained ones.

### 38.4 Why MRL beats post-hoc PCA on the *same underlying embedding* — the comparison made precise

> *"Beats post-hoc PCA: traditional reduction runs after the encoder and is not data-aware; MRL learns
> the nested structure during training"*

**This is the single most important comparison in Part 4 for tying this lecture back to
[Part 2](dimensionality-reduction-02.md), so it deserves the full contrast, not just the one-line
claim.**

| | **Post-hoc PCA on a fixed embedding** | **MRL** |
|---|---|---|
| When applied | *After* the embedding model is fully trained and frozen | *During* training — it changes what the encoder itself learns to produce |
| What determines the "important" directions | The **variance** of the already-trained embedding's output distribution — see [Part 2 §9](dimensionality-reduction-02.md): PCA finds whatever directions happen to have the most spread, which needn't align with what's *useful* for the downstream task | The task's own loss function, evaluated directly at every truncation point |
| Failure mode | Exactly [Part 1 §7.2](dimensionality-reduction-01.md)'s low-variance-signal trap: a dimension can be highly *discriminative* for retrieval while contributing little raw variance, and PCA would discard it | None specific to this — the loss is evaluated on the actual task, at every size, by construction |
| Cost to obtain a smaller embedding | Cheap (matrix multiply) once the embedding exists, but the *quality* at that size was never optimised for | The smaller sizes are trained-for from the start; no extra step needed at inference beyond truncating the vector |

**The precise mechanism behind "not data-aware," stated using Part 2's own vocabulary:** PCA
[(Part 2 §9)](dimensionality-reduction-02.md) chooses its projection to maximise *retained variance* —
a purely geometric, task-blind criterion computed from the embedding's output distribution alone. It
has no way to know, and no mechanism to enforce, that the directions it keeps are the ones that matter
for retrieval accuracy, classification accuracy, or whatever downstream task the embedding actually
serves. **MRL sidesteps this entirely by never separating "find the important directions" from "train
the model" as two steps** — the importance ordering is a direct *consequence* of training against the
real task loss at every prefix length, so there is no risk of the two criteria (variance vs.
usefulness) coming apart, because there is only ever one criterion in play.

> 💡 **The clean one-sentence contrast for an interview:** *PCA discovers which directions already have
> the most variance in an embedding that was never told to organise itself that way; MRL trains the
> embedding, from the start, to organise itself that way on purpose.* Same goal — a usefully ordered,
> truncatable representation — reached by measurement after the fact versus by design from the start.

```interactive
type: slider
title: Truncate the nested dolls
concept: Every prefix of an MRL embedding is independently valid, and quality degrades gracefully
control: A dimension slider from 8 to 2048 (logarithmic), plus a toggle comparing "MRL prefix" against "post-hoc PCA to the same size" on the same base embedding
observe: A retrieval-accuracy readout at the current size for both methods, plotted as two curves across the whole slider range, with the trained checkpoints (8, 16, 32, ..., 2048) marked
insight: The MRL curve degrades smoothly and stays close to full-size accuracy even between trained checkpoints, while the post-hoc PCA curve sits visibly lower at every size, because it was never optimised against the task loss the way each MRL prefix was
fallback: The deck's own numbers — MRL at 256 dims reaches 96.6% of full 1024-dim BEIR quality versus 92.1% for an independently-sized model, and MRL at 512 dims on ImageNet (76.3%) beats a dedicated 512-dim model (74.2%)
```

---

## 39. Production: Real-World Impact

> [slides 103–106, 53:37–55:06]
>
> - *"**ImageNet classification (ResNet-50 backbone):** MRL @ 512-dim = 76.3% top-1 vs dedicated
>   512-dim model = 74.2%"*
> - *"**BEIR retrieval benchmark (text):** MRL @ 256-dim achieves 96.6% of full 1024-dim quality;
>   independently trained 256-dim model achieves only 92.1%"*
> - *"**Storage reduction (OpenAI text-embedding-3-large):** 3072 → 256 dims = **12× size reduction**"*
> - *"**Speed gain:** ANN search at 256-dim = ~4–5× faster than 3072-dim; cosine similarity compute
>   scales linearly with dimension"*
> - *"**Sweet spot:** 256–512 dims captures ≈95%+ of full-dim quality for most retrieval and
>   classification tasks; going below 128 dims shows meaningful quality degradation"*
> - *"**OpenAI (2024):** text-embedding-3-small/large both support MRL via dimensions parameter; users
>   can request any size from 64 to max; API pricing unchanged"*
> - *"**Cohere Embed v3:** Fully MRL-trained; recommended to use 256-dim for most RAG use cases - 4×
>   cheaper storage, marginal quality drop vs full 1024-dim"*
> - *"**Pinecone + Weaviate:** Native support for storing truncated MRL vectors; enable tiered search -
>   fast ANN at low dim, rerank at full dim"*
> - *"**2-stage retrieval pattern:** (1) Retrieve top-1000 candidates using cheap 256-dim MRL vectors;
>   (2) Rerank with full 3072-dim vectors → best of both worlds"*
> - *"**Cost example (OpenAI):** Storing 100M docs at 3072 dims ≈ 1.2TB float32; at 256 dims ≈ 100GB -
>   12× reduction, still >90% retrieval quality"*
> - *"**Key takeaway:** MRL makes the embedding dimension a runtime parameter, not a design-time
>   decision - maximum flexibility for production systems"*

### 39.1 🧪 Verify the two headline numbers, and read what each is actually claiming

**ImageNet: 76.3% (MRL @ 512) vs 74.2% (dedicated 512-dim model).** This is a **+2.1 percentage-point
advantage** *for the MRL doll over a model trained from scratch specifically at that size* — not merely
"MRL doesn't lose much versus the full model," but genuinely **better than the dedicated small model**,
confirming the "surprise result" from §38.

**BEIR: 96.6% (MRL @ 256) vs 92.1% (independent 256-dim model), both against a 1024-dim reference.**
Reframe this as an error-rate comparison, which makes the gap sharper: MRL's shortfall from full quality
is $100 - 96.6 = 3.4$ percentage points; the independent model's shortfall is $100 - 92.1 = 7.9$ points.
**MRL's quality gap is less than half the independent model's** — $3.4/7.9 \approx 0.43$ — at the
*identical* parameter/storage budget.

### 39.2 The 2-stage retrieval pattern — why this is the practical payoff, in one sentence

**This closes the loop back to [Part 2's PCA/SVD framing](dimensionality-reduction-02.md) one more
time.** Retrieval with a full-size vector is accurate but expensive to run at scale (every candidate
document's similarity to the query must be computed, and cosine similarity cost is linear in
dimension); retrieval with a truncated vector is cheap but slightly less accurate. **The 2-stage
pattern gets both**: use the cheap, truncated 256-dim MRL vectors to narrow millions of candidates down
to (say) the top 1,000 quickly, then re-score just those 1,000 with the full 3072-dim vectors for final
precision. This only works because MRL's 256-dim prefix and its 3072-dim full vector are **the exact
same coordinate system, truncated** — precisely the "inconsistency problem" from §37 solved by
construction, since there is no risk of a document embedded at one size being incomparable to a query
embedded at another.

### 39.3 The closing takeaways table, verbatim

> [slides 106–107, 55:06–56:25]

| Method | Purpose |
|---|---|
| **PCA** | Linear dimensionality reduction |
| **Autoencoder** | Nonlinear dimensionality reduction |
| **VAE** | Probabilistic latent representation |
| **Latent Diffusion** | Diffusion in compressed latent space |
| **Matryoshka Representation Learning** | Nested latent representations |

**Read this table as the deck's own one-line summary of every idea in this file**, and notice what it
implicitly leaves out: NMF, VQ-VAE and LoRA don't appear, because — per this document's *Putting it
together* section below — each of the five listed methods represents a distinct *category* of technique
(linear vs. nonlinear reduction, deterministic vs. probabilistic, static vs. runtime-adjustable), while
VQ-VAE is a variant *within* the VAE row and LoRA is a low-rank method applied to weights rather than to
data, a genuinely separate axis this closing table doesn't attempt to capture.

---

## Putting it together

```
   PART 2's THEOREM: a linear autoencoder + MSE = EXACTLY the PCA subspace (Baldi–Hornik)   §9–§10
                                    │
                    "nonlinearity is the superpower" — the ONE thing that lets
                    an autoencoder do what PCA (Part 2 §14) provably cannot: fit
                    a CURVED manifold, not just a flat subspace
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
   DENOISING AE               SPARSE AE                    (plain AE has
   reconstruct CLEAN          force units OFF via           TWO weaknesses:
   from CORRUPTED             KL(Bernoulli(ρ)‖ρ̂)            memorises, and its
   noise lives in the         §16–§18                       latent space has
   discarded low-rank         → modern: LLM SAE             HOLES — nothing
   directions (Part 2 §12)      interpretability            ever trained it
   §12–§14                                                   to be smooth)
                                                                   │
                                                                   ▼
                                          NEITHER FIX MAKES IT GENERATIVE    §19–§20
                                                   │
                                                   ▼
   VAE: encode a DISTRIBUTION N(μ,σ²), not a point            §21–§22
        reparam trick: z = μ + σ⊙ε  (Prereq 4, differentiable)
        ELBO = RECON − KL(q(z|x)‖p(z))                        §23
                          ▲
              REVERSE KL (Part 2 §24: mode-seeking, zero-forcing)
              ⇒ VAE's own failure mode is predicted, not just observed
                                                   │
                          ┌────────────────────────┴────────────────────────┐
                          ▼                                                 ▼
                 VQ-VAE: SNAP to codebook                          the smooth, hole-free
                 discrete latents ⇒ an image                       latent space is EXACTLY
                 becomes tokens a TRANSFORMER                      what Stable Diffusion's
                 can model (DALL·E, Jukebox)                       VAE encoder needs
                 §26–§28                                                    │
                                                                             ▼
                                        LATENT DIFFUSION: run the iterative denoising
                                        INSIDE that latent space, not in pixel space
                                        512²×3 = 786,432 values, O(n²) attention on
                                        262,144 tokens ⇒ INFEASIBLE in pixels
                                        64×64 latent ⇒ 64× fewer elements, 4,096×
                                        cheaper attention                    §29–§30
                                        (extends to video: spatiotemporal VAE + DiT)

   ═══════════════════════════════════════════════════════════════════════════
   PART 2's OTHER THEOREM: truncated SVD is the PROVABLY OPTIMAL low-rank
   approximation of ANY matrix (Eckart–Young)                              §31–§32
                                    │
                    applied not to DATA but to a WEIGHT UPDATE ΔW
                                    │
                    LoRA:  ΔW ≈ B·A,  r ≪ min(d,k)
                           B=0 at init ⇒ zero disruption, A random ⇒ breaks
                           symmetry so SOMETHING can learn
                                    │
                    DoRA: decouple W = m·(V/‖V‖) — magnitude trained
                          SEPARATELY; LoRA's B·A applies ONLY to direction
                          ⇒ closer to how full fine-tuning actually moves    §35

   ═══════════════════════════════════════════════════════════════════════════
   PART 2's THIRD IDEA: PCA orders components by IMPORTANCE so truncation
   after the fact is safe — but that ordering is COMPUTED, not TRAINED       §36–§39
                                    │
                    MRL bakes the SAME property in DURING TRAINING:
                    L_Matryoshka = avg( L(z_1:8) + L(z_1:16) + ... + L(z_1:2048) )
                    ⇒ every prefix independently forced to be a valid embedding
                    ⇒ beats POST-HOC PCA because PCA orders by VARIANCE
                      (task-blind); MRL orders by the TASK LOSS ITSELF
                    ⇒ the embedding dimension becomes a RUNTIME CHOICE
```

### Walking the diagram

**This lecture has no new mathematical machinery of its own.** Every method in it is one of two ideas
from Parts 1–2 — the bottleneck, or the low-rank factorisation — deployed inside a system trained
end-to-end with gradient descent instead of solved in closed form. Recognising which of the two applies
is most of the work of understanding any individual section.

**Part 1's arc is the bottleneck, escalated four times.** §9's theorem — a linear autoencoder with MSE
loss *is* PCA — is the anchor for everything that follows, because it tells you exactly what
nonlinearity is buying: the ability to bend the bottleneck around a curved manifold, which
[Part 2 §14](dimensionality-reduction-02.md) proved PCA structurally cannot do. Denoising and sparsity
are two independent fixes for two independent weaknesses of the plain bottleneck — memorisation, and
lack of interpretable structure — neither of which touches the deeper problem that a plain
autoencoder's latent space has holes. The VAE fixes *that* by making the bottleneck **probabilistic**
rather than deterministic, and the fix is provably principled rather than a heuristic: the
reparameterisation trick makes a stochastic node differentiable, and the ELBO's KL term is *reverse*
KL — which [Part 2 §24](dimensionality-reduction-02.md) already told you is mode-seeking and
zero-forcing, so the VAE's known failure mode (posterior collapse) isn't a separate fact to memorise,
it's a direct consequence of a fact you already derived. VQ-VAE takes a different fork from the same
starting point, making the bottleneck **discrete** — motivated by nothing more exotic than "Transformers
expect discrete tokens" — and the payoff (the smooth, hole-free latent space) is exactly what Part 2's
"From Dimensionality Reduction to Generation" needs: latent diffusion is not a new architecture so
much as it is Part 1's VAE, reused as a *computational* trick to make an otherwise-infeasible generative
process tractable.

**Parts 3 and 4 both descend from Part 2's SVD, along two different consequences of the same theorem.**
LoRA takes Eckart–Young's "truncation is provably optimal" and applies it not to data but to a **weight
update** — the empirical claim that $\Delta W$ has low intrinsic rank is Part 2's low-rank hypothesis,
transplanted from data matrices to gradients. DoRA's contribution is orthogonal to rank at all: it
notices that a single low-rank matrix conflates two genuinely different kinds of change (magnitude,
direction) and separates them. Matryoshka Representation Learning takes the *other* half of what SVD
gives you — not "truncation is optimal" but "components are already ordered by importance, so
truncation is *safe*" — and asks what happens if you **train for that ordering directly**, rather than
computing it after the fact from a fixed embedding's variance. The answer is a strictly better trade:
the same truncation-safety PCA offers, but calibrated against the actual downstream task instead of
against raw variance, which is precisely why it beats post-hoc PCA on the identical embedding.

> 💡 **The single thread worth keeping from this entire file:** every technique here answers "how do I
> get a smaller, still-useful representation?" and the honest way to classify them is by *what they're
> willing to give up* to get it — a plain AE gives up guaranteed generation ability; a VAE gives up
> reconstruction sharpness for a smooth latent space; VQ-VAE gives up continuity for discreteness;
> LoRA gives up full expressiveness in the update for compute and memory; MRL gives up nothing
> measurable (it *beats* the alternative it replaces) but only because it moved the ordering decision
> from *after* training to *during* it. Knowing which trade each method is making is the difference
> between reciting five names and understanding one idea five ways.

---

## Interview prep — Amazon Applied Scientist

### Core questions

<details>
<summary><b>1. (Easy)</b> What is an autoencoder, and how does it relate to PCA?</summary>

An autoencoder is a feed-forward network trained to reconstruct its own input through a **bottleneck**
— a middle layer narrower than the input, forcing the network to compress information and discard what
it can't usefully represent.

**The precise relationship to PCA, not just an analogy:** a *linear* autoencoder — no activation
functions between layers — trained with MSE loss has, as its global optimum, **exactly the PCA
subspace** (Baldi & Hornik, 1989). This is provable, not approximate. The deck's own quiz makes this
the correct answer against two plausible-sounding wrong ones ("unrelated to PCA" and "always
outperforms PCA because more parameters" — neither is true for the linear case).

**What makes a "real" (nonlinear) autoencoder different, and worth using:** nonlinearity is the *only*
thing that lets it surpass PCA, specifically by letting the learned bottleneck fit **curved**
manifolds — the Swiss-roll case PCA structurally cannot handle, because PCA can only ever find a flat
subspace.
</details>

<details>
<summary><b>2. (Easy)</b> Why can't you generate new realistic samples from a plain autoencoder?</summary>

Because nothing in a plain autoencoder's training objective ever evaluates what happens if you decode a
point the network wasn't trained on. Reconstruction loss only checks the network's behaviour at the
exact latent codes its training examples land on — so the latent space develops **holes and gaps**
between clusters, and with enough capacity the network can simply **memorise** each training example in
an isolated pocket of latent space rather than learning any continuous structure.

Sample a random point from that latent space (say, from $N(0,I)$) and, with high probability, you land
in one of those unpopulated gaps — regions the decoder was never trained to make sense of — and the
output is garbage.

**The fix, in one sentence:** encode each input as a *distribution* rather than a point, and explicitly
regularise every one of those distributions toward a shared, simple prior (typically $N(0,I)$) with a
KL penalty. That's the entire structural difference a VAE adds.
</details>

<details>
<summary><b>3. (Medium)</b> Derive the reparameterisation trick and explain why it's necessary.</summary>

**The problem:** the VAE's encoder outputs $\mu$ and $\sigma$ defining $q(z|x) = N(\mu, \sigma^2)$, and
you need to backpropagate through the *sampling* of $z$ from this distribution to train $\mu$ and
$\sigma$. But sampling is a stochastic operation — there's no meaningful $\partial z/\partial\mu$ for a
random-number-generator call, because $z$'s value didn't arise from a differentiable computation
involving $\mu$.

**The fix:** rewrite the sample as

$$z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim N(0, I)$$

**Verify this samples correctly:** a linear transform $a + b\epsilon$ of a standard normal is itself
normal with mean $a$, variance $b^2$ — so with $a=\mu$, $b=\sigma$, $z \sim N(\mu,\sigma^2)$, matching
the target distribution exactly.

**Now the gradients exist:** $\partial z/\partial\mu = 1$ and $\partial z/\partial\sigma = \epsilon$,
both ordinary partial derivatives of a deterministic function, because once $\epsilon$ is fixed as an
external input (drawn from a source with **no learnable parameters**), $z$ is just an affine function
of $\mu$ and $\sigma$. Backprop flows through it exactly as through any other layer.

**Why it's necessary rather than a nicety:** without it, there is no way to train the encoder's $\mu$
and $\sigma$ outputs at all via gradient descent — the sampling step would be an impassable wall between
the loss and those parameters.
</details>

<details>
<summary><b>4. (Medium)</b> Write the ELBO and explain the tension between its two terms.</summary>

$$\mathcal{L}_{VAE} = -\mathbb{E}_{z\sim q_\varphi(z|x)}\left[\log p_\theta(x|z)\right] + D_{KL}\!\left(q_\varphi(z|x)\,\|\,p_\theta(z)\right)$$

**Term 1, reconstruction:** encourages the decoder to reconstruct $x$ faithfully from a sampled $z$ —
functionally the same objective a plain autoencoder optimises.

**Term 2, KL regularisation:** penalises the encoder's distribution $q(z|x)$ for straying from the fixed
prior $p(z) = N(0,I)$ — this is what forces the latent space to be smooth and hole-free.

**The tension:** reconstruction alone wants each $q(z|x)$ to be a **sharp spike** — narrower means the
decoder knows more precisely what it's working with, which improves reconstruction. Taken to the
extreme, this collapses $\sigma \to 0$ and reduces the model back to a plain point-encoding
autoencoder. KL regularisation alone wants every $q(z|x)$ to **equal the prior exactly**, which would
discard all information about $x$ and make reconstruction impossible. The trained model sits at
whatever compromise minimises their sum — sharp enough to reconstruct well, diffuse enough to keep the
latent space well-covered.

**$\beta$-VAE makes this an explicit dial:** replace the KL term with $\beta \cdot D_{KL}$; larger
$\beta$ trades reconstruction sharpness for a smoother, more prior-like latent space.
</details>

<details>
<summary><b>5. (Medium)</b> Why is the VAE's KL term specifically "reverse" KL, and what failure mode does that predict?</summary>

The term is $\mathrm{KL}(q_\varphi(z|x) \,\|\, p_\theta(z))$ — the encoder's approximate posterior in
the **first** slot. By the definition $\mathrm{KL}(A\|B) = \mathbb{E}_{x\sim A}[\log(A/B)]$, the
expectation here is taken **under $q$**, which is exactly what makes this reverse (mode-seeking,
zero-forcing) KL rather than forward KL.

**What zero-forcing means concretely here:** the loss only pays a heavy penalty where $q$ places mass
but $p$ doesn't — it does **not** penalise $q$ for failing to *cover* some region of $p$. So the
encoder is free to leave large parts of the prior's support entirely unused, using only a fraction of
the latent space's capacity, without that showing up as a loss increase.

**The predicted failure mode: posterior collapse.** Especially with a powerful decoder (one that can
reconstruct well from very little latent information), the encoder can learn to squeeze most or all
inputs' distributions into a small region of the prior — or even ignore some latent dimensions
entirely, letting them default to exactly the prior — and pay essentially no KL cost for doing so. The
generative diversity of the model then suffers, because much of the nominal latent capacity carries no
real information.

This isn't an incidental empirical finding — it follows directly from the direction of the KL, which is
why understanding forward-vs-reverse KL (as derived in
[Part 2 §24](dimensionality-reduction-02.md)) lets you *predict* this failure mode rather than just
recite it.
</details>

<details>
<summary><b>6. (Medium)</b> Explain vector quantisation in a VQ-VAE, and how gradients flow through a non-differentiable operation.</summary>

**The setup:** the encoder produces a continuous vector $z_e = E(x)$, which is then **snapped** to the
nearest vector $e_k$ in a learned, finite codebook $e \in \mathbb{R}^{K\times D}$ — a
nearest-neighbour lookup, formally $k = \arg\min_i \|z_e - e_i\|_2$. The decoder reconstructs from the
snapped, discrete $e_k$, not from the original continuous $z_e$.

**The three-term loss:** reconstruction ($\|x - D(e_k)\|^2$, standard AE loss), a **codebook loss**
($\|\mathrm{sg}[E(x)] - e_k\|^2$, which — via a stop-gradient on the encoder side — trains *only* the
codebook to chase the encoder's output), and a **commitment loss** ($\beta\|E(x) -
\mathrm{sg}[e_k]\|^2$, which via a stop-gradient on the codebook side trains *only* the encoder to stay
close to whatever codebook entry it's currently assigned).

**The gradient problem:** $\arg\min$ has zero gradient almost everywhere and an undefined gradient at
tie-boundaries — ordinary backprop cannot pass through it to reach the encoder.

**The fix — the straight-through estimator:** on the forward pass, use the true (discrete) $e_k$.
On the backward pass, copy the gradient arriving at $e_k$ directly back to $z_e$, as if quantisation had
been the identity function. This is an explicit, *biased* approximation — not a mathematically correct
gradient — that works in practice specifically because the commitment loss keeps $e_k$ close to $z_e$,
making the approximation locally reasonable.

**Why go discrete at all:** a discrete grid of codebook indices is exactly the token format a
Transformer expects, letting an image be modelled as a sequence-generation problem — precisely what
DALL-E and Jukebox do.
</details>

<details>
<summary><b>7. (Medium)</b> Why does pixel-space diffusion not scale, and how does latent diffusion fix it?</summary>

Two separate costs, and they need to be distinguished:

**Linear cost:** each denoising step processes the full image tensor. A 512×512 RGB image is
$512\times512\times3 = 786{,}432$ values, processed at every one of (typically) 20–1000 denoising
steps for a single generated image.

**Quadratic cost:** self-attention within the U-Net, applied at full spatial resolution, treats each of
the $512^2 = 262{,}144$ spatial positions as a token, and self-attention costs $\mathcal{O}(n^2)$ in
token count — roughly $6.9\times10^{10}$ pairwise scores for one attention layer at one step. This is
what "memory explodes" means literally, not rhetorically.

**The fix:** train a VAE first, exactly as in the VAE sections, to compress images into a smooth
latent space. Then run the *entire* diffusion process — every denoising step — inside that latent
space, only decoding back to pixels once, at the end. Stable Diffusion's factor of $f=8$ downsampling
turns a 512×512 image into a 64×64 latent: $512^2/64^2 = 64\times$ fewer spatial elements — and because
attention cost is *quadratic* in token count, the attention saving is $64^2 = 4{,}096\times$, which is
the real reason self-attention becomes usable inside the diffusion network at all.

**The dependency worth naming:** this only works because the VAE's latent space is smooth and hole-free
(the KL regularisation from question 4/5) — the diffusion trajectory wanders through *every*
intermediate point during its hundreds of steps, not just points seen during VAE training, so a latent
space with AE-style gaps would break the process.
</details>

<details>
<summary><b>8. (Medium)</b> Explain LoRA: what problem it solves, and why the B=0 initialisation matters.</summary>

**The problem:** full fine-tuning of an LLM updates the *entire* weight matrix $W \in
\mathbb{R}^{d\times k}$ — potentially billions of parameters, a full optimizer state per parameter, and
a separate full copy of the model per downstream task.

**The key empirical insight:** the weight **update** $\Delta W$ that fine-tuning wants to apply — not
$W$ itself, the *change* — behaves as though it has low intrinsic rank. LoRA exploits this directly by
freezing $W$ and learning only $\Delta W \approx B\cdot A$, with $B \in \mathbb{R}^{d\times r}$,
$A \in \mathbb{R}^{r\times k}$, and $r \ll \min(d,k)$ — this is [Part 2's SVD low-rank
hypothesis](dimensionality-reduction-02.md), applied to a weight update rather than to data.

**Parameter savings, computed:** at $d=k=4096$, $r=8$: full fine-tuning is $4096^2 \approx 16.8$M
parameters; LoRA is $2\times(4096\times8) = 65{,}536$ parameters — a **256× reduction**, for this one
matrix.

**Why $B$ starts at exactly zero, and $A$ starts random:** with $B=0$, $\Delta W = BA = 0$, so
$h = Wx + BAx = Wx$ exactly — the very first forward pass is **bit-identical** to the frozen pretrained
model, introducing zero disruption. If *both* $A$ and $B$ started at zero, the gradient with respect to
either would also be zero (each factor's gradient involves the other factor as a multiplicative term),
and nothing would ever begin learning — so exactly one of the two, $A$, must be randomly initialised
purely to break that symmetry.

**Why it's cheap at inference too:** $BA$ can simply be added into $W$ once training finishes, so a
LoRA-adapted model has **no extra inference latency or extra parameters** compared to the original —
only a tiny, swappable adapter needs to be stored per task.
</details>

<details>
<summary><b>9. (Medium)</b> What does DoRA add on top of LoRA?</summary>

DoRA decomposes each pretrained weight column into a **magnitude** (a scalar) and a **direction** (a
unit vector): $W = m \cdot (V/\|V\|)$ — an algebraic identity, not an approximation. It then makes the
magnitude $m$ a **separate, independently trained parameter**, and applies LoRA's low-rank update $BA$
**only within the direction term**:

$$W' = m \cdot \frac{V + BA}{\|V + BA\|}$$

**What this fixes:** plain LoRA's single low-rank matrix $BA$ conflates magnitude changes and direction
changes together — anything fine-tuning wants to do to $W$, whether a pure rescaling or a genuine
rotation, gets squeezed through the same rank-$r$ bottleneck. Liu et al. (2024) found that *full*
fine-tuning naturally varies magnitude and direction with a different correlation pattern than LoRA
reproduces, and DoRA's decoupling closes that gap — a pure magnitude change now costs almost nothing
(a single extra scalar per column), rather than consuming part of the shared rank budget.

**Result and caveat:** consistently beats LoRA at equal parameter budget on the tasks it was evaluated
on, with the same zero-inference-cost property (it also merges back into $W$). It's an empirical
finding about how full fine-tuning behaves, not a mathematical guarantee that transfers to every task.
</details>

<details>
<summary><b>10. (Hard — combines two concepts)</b> Explain Matryoshka Representation Learning's loss, and derive why it beats post-hoc PCA on the same base embedding.</summary>

**The loss** applies the *same* task loss $\mathcal{L}$ (unchanged) to every one of several nested
prefixes of the embedding, then averages:

$$\mathcal{L}_{\text{Matryoshka}} = \text{average}\left(\mathcal{L}(z_{1:8}) + \mathcal{L}(z_{1:16}) + \cdots + \mathcal{L}(z_{1:2048})\right)$$

**Derive the ordering property.** For the sum to be small, every term — including $\mathcal{L}(z_{1:8})$
— must independently be driven down, and $z_{1:8}$ has only eight numbers to do it with, with no access
to later dimensions (that term never sees them). So the eight numbers occupying the front of the vector
are under direct pressure to be maximally useful **on their own**, while later dimensions, only
evaluated by the largest-prefix terms, can afford to encode narrower, finer detail. This produces
coarse-to-fine packing as a *consequence* of the objective, not as an engineered rule.

**Now the comparison against post-hoc PCA, which is the two-concept combination.** PCA
([Part 2 §9](dimensionality-reduction-02.md)) chooses its projection to maximise **retained variance**
— a purely geometric criterion computed from the *already-trained* embedding's output distribution,
with no knowledge of the downstream task at all. This is exactly [Part 1's low-variance-signal
trap](dimensionality-reduction-01.md): a direction can be highly *discriminative* for the actual task
while carrying relatively little raw variance, and PCA has no way to detect or protect that direction —
it would be a candidate for truncation regardless of its usefulness.

MRL never separates "find the important directions" from "train the model" into two steps at all — the
importance ordering is a direct, unavoidable consequence of training against the real task loss at
every prefix length. There is only ever one criterion (the task loss), so it cannot come apart from
"usefulness" the way PCA's variance criterion can.

**Empirical confirmation, and note the standard deserved:** on ImageNet at 512 dims, MRL scores 76.3%
versus 74.2% for a model trained from scratch at that size — MRL doesn't just avoid losing much versus
the full model, it **beats** a dedicated small model at the identical size, which is only explicable by
the task-aware training producing a genuinely better allocation of those 512 dimensions than an
independent optimisation would find on its own.
</details>

<details>
<summary><b>11. (Hard — combines two concepts)</b> Trace a single idea — "the largest source of variation in a representation is often not what you actually care about" — through at least two sections of this lecture and one section of Part 2.</summary>

**[Part 2 §11](dimensionality-reduction-02.md), the origin of the idea:** PCA's eigenfaces on a face
dataset put **illumination**, not identity, in PC1–PC3 — because moving a light source changes every
pixel more than changing the person does, and PCA sorts strictly by variance. The general lesson
stated there: *PCA finds what varies most, which is often a nuisance factor rather than the signal you
care about.*

**§17's sparse autoencoder, the same idea inverted into a design principle.** Rather than accepting
whatever a network's raw activations happen to emphasise, the sparsity penalty forces *most* units to
stay off for any given input — pushing the representation toward using only the few units that are
genuinely relevant to that specific example, rather than letting high-variance-but-uninformative
patterns dominate. This is precisely why sparse autoencoders are now used for **LLM interpretability**:
raw neuron activations in a trained model are polysemantic (dominated by whatever combinations of
concepts happen to co-vary most, not by clean single concepts), and a sparsity constraint is a direct
countermeasure.

**§38's Matryoshka Representation Learning, the same idea solved by changing what's being optimised
rather than by adding a constraint.** Post-hoc PCA on a trained embedding would again default to
ordering by raw variance — exactly Part 2 §11's failure mode, transplanted to embeddings. MRL sidesteps
it entirely by training the ordering against the actual downstream task loss from the start, so
"what the representation emphasises most" and "what matters for the task" are guaranteed to coincide,
by construction, rather than needing to be reconciled after the fact.

**The unifying statement:** whenever a representation is organised by an unsupervised, task-blind
criterion (raw variance, in every case above), there is no guarantee that criterion aligns with what
actually matters downstream — and every fix in this trace (a sparsity constraint, task-aware nested
training) works by injecting the *actual objective* directly into how the representation gets shaped,
rather than hoping an unsupervised proxy happens to agree with it.
</details>

<details>
<summary><b>12. (Hard — combines two concepts)</b> You're asked to build a retrieval system for 200M documents that must support both fast approximate search and precise reranking, with a strict 1.5TB storage budget for embeddings. Walk through the design, and be explicit about which parts of this lecture you're drawing on.</summary>

**Framing.** Two distinct needs — cheap, fast candidate retrieval across the full corpus, and precise
scoring of a small shortlist — that traditionally require two different embedding sizes, which
(per §37) creates the *inconsistency problem*: a document embedded at one size and a query embedded at
another live in incompatible vector spaces and cannot be compared at all.

**The fix: train (or select) an MRL-trained embedding model rather than two separate models.** This
directly targets §37's five failure modes at once: one model, one training run, one coordinate system
at every truncation size — the inconsistency problem doesn't arise because a 256-dim vector and a
3072-dim vector from the same MRL model are the *same* coordinate system, just truncated.

**Storage budget, computed.** At $256$ dims (float32), $200\text{M} \times 256 \times 4\text{ bytes}
\approx 205\text{GB}$ for the cheap tier. At full 3072 dims, $200\text{M}\times3072\times4 \approx
2.46\text{TB}$ — already over budget on its own. **This is where the 2-stage retrieval pattern from
§39.2 earns its place**: store the cheap 256-dim vectors for *every* document (205GB, well within
budget), and store the full 3072-dim vectors only for a much smaller "hot" subset likely to be
retrieved often, or compute them on-demand for the top-1000 candidates a first-stage search surfaces,
rather than persisting all 200M at full size. If a modest subset (say the top 50M "hot" documents)
needs full-size vectors resident, that's an additional $50\text{M}\times3072\times4 \approx 615\text{GB}$
— total under 1TB, inside budget, versus $2.46\text{TB}$ for a naive full-size-everywhere design.

**The retrieval flow, matching §39.2 exactly:** query embedded once by the MRL model, truncated to
256-dim for a fast ANN search across the full 200M-document index to get, say, the top 1,000
candidates; then the full 3072-dim query vector reranks just those 1,000 against their full-size (or
on-demand-computed) vectors for final precision.

**What I'd explicitly avoid, and why, citing the relevant sections:** training two separate models at
two sizes (§37's maintenance and inconsistency problems, both reintroduced); reducing a single trained
embedding with post-hoc PCA to get the small size (§38.4 — PCA orders by variance, not by retrieval
usefulness, so the small-size quality would underperform what MRL delivers at the identical size,
per the deck's own BEIR numbers: 96.6% for MRL vs 92.1% for an independently-sized alternative);
and storing full-size vectors for all 200M documents from the start, which blows the storage budget by
roughly 60% before any optimisation.

**What I'd flag as an open decision rather than resolve unilaterally:** whether to use an
already-MRL-trained third-party model (OpenAI's text-embedding-3, Cohere Embed v3 — both cited as
MRL-trained in §39) versus training a domain-specific one, which trades integration speed against
potential domain-relevance quality — a decision that depends on data availability and latency/cost
constraints this scenario doesn't specify.
</details>

### Depth probes

| Your good answer | The probe | What they want |
|---|---|---|
| "A linear autoencoder is equivalent to PCA" | *"Equivalent in what precise sense — does it find the same components in the same order?"* | The **subspace** is the same (Baldi–Hornik); the individual latent dimensions are generally an arbitrary rotation within that subspace, not ordered/aligned with PCA's eigenvector ordering unless additional constraints are imposed. |
| "Denoising autoencoders generalise better" | *"Why does that specifically prevent memorisation, mechanistically?"* | The loss target is the *original* $x$, never seen by the network — the identity function on the corrupted input reconstructs noise, not $x$, so it scores badly on the actual objective; memorisation is no longer a viable shortcut. |
| "The VAE's KL term keeps the latent space smooth" | *"What if you set $\beta$ very high — does that make it strictly better?"* | No — very high $\beta$ collapses every $q(z|x)$ toward the prior, discarding information about $x$ and hurting reconstruction; it's a tradeoff dial, not a free lever. |
| "VQ-VAE uses a straight-through gradient" | *"Is that gradient actually correct?"* | No — it's a deliberate, biased approximation (true $\arg\min$ gradient is ~zero almost everywhere); it works because the commitment loss keeps the approximation locally reasonable, not because it's mathematically exact. |
| "Latent diffusion compresses 64× spatially" | *"What about the attention cost specifically?"* | $64^2 = 4{,}096\times$ cheaper, because self-attention is quadratic in token count — the linear and quadratic savings are different numbers and shouldn't be conflated. |
| "LoRA freezes $W$ and learns $B\cdot A$" | *"What if you'd initialised both $A$ and $B$ randomly instead?"* | The first forward pass would differ from the pretrained model by a random amount — disruptive, and the deck's own quiz makes exactly this point; $B=0$ specifically buys "training begins at the pretrained model." |
| "DoRA decouples magnitude and direction" | *"Is that decoupling something LoRA is mathematically incapable of representing, or just less naturally suited to?"* | Not mathematically incapable — a sufficiently expressive $BA$ *could* in principle represent a magnitude-only change — but it has no dedicated mechanism for it and must spend rank-budget capacity on it; DoRA gives magnitude a free, separate parameter instead. |
| "MRL beats post-hoc PCA" | *"Compare the training cost of the two approaches, not just their quality."* | MRL requires training (or retraining) the base model with the nested loss from the start — more expensive up front. Post-hoc PCA is applied after the fact to any existing embedding, at near-zero marginal cost. The quality gain isn't free. |

### Whiteboard-ready derivations

**D1 — the linear-autoencoder-is-PCA theorem, stated precisely.**
```
Linear AE:   z = Vx  (encoder, k x D)      x̂ = Uz = UVx  (decoder, D x k)
Loss:        L(x, x̂) = ||x − x̂||²  =  ||x − UVx||²

Claim (Baldi & Hornik, 1989): at the global optimum of this loss,
  the ROW SPACE of V (equivalently, the subspace UV projects onto)
  = the subspace spanned by the top-k eigenvectors of Cov(x)
  = EXACTLY the PCA subspace.

⇒ "nonlinearity is the superpower": the ONLY way for an autoencoder
  to represent anything PCA cannot (i.e. a curved manifold, not a
  flat subspace) is to introduce nonlinear activations between layers.
```

**D2 — the reparameterisation trick.**
```
want:  backprop through z ~ N(μ, σ²)          ← no ∂z/∂μ, ∂z/∂σ exist (stochastic node)

rewrite:  z = μ + σ⊙ε,   ε ~ N(0, I)           ← ε has NO learnable parameters

check:  a + bε ~ N(a, b²)  for ε ~ N(0,1)
        ⇒ z = μ + σε ~ N(μ, σ²)                 ← same distribution, verified

now:    ∂z/∂μ = 1          ∂z/∂σ = ε            ← both exist; z is now a
                                                    DETERMINISTIC function of μ, σ
                                                    (with ε as a fixed input)
```

**D3 — LoRA's parameter count and the B=0 justification, in one block.**
```
full fine-tune:  d × k parameters                    (e.g. 4096×4096 = 16.78M)
LoRA:            B(d×r) + A(r×k) = r(d+k) parameters  (e.g. 8×(4096+4096) = 65,536)
ratio:           dk / [r(d+k)]                          (here: 256×)

forward pass:    h = Wx + BAx

at init:  B = 0  ⇒  BAx = 0  ⇒  h = Wx  EXACTLY         (zero disruption)
          A random                                       (breaks symmetry)

if BOTH were 0:  ∂L/∂A involves a factor of B = 0  ⇒  ∂L/∂A = 0
                 ∂L/∂B involves a factor of A = 0  ⇒  ∂L/∂B = 0
                 ⇒ NOTHING would ever start learning
```

### Applied scenario — building an in-house image-search feature for the catalogue

**The problem.** Sellers upload product photos; the catalogue team wants "find visually similar
products" search across 50M active listings, plus a separate internal tool letting analysts browse
clusters of visually similar images to spot mislabeled categories. Compute and storage are both
tightly budgeted; the feature must ship in one quarter.

**Framing.** Two genuinely different consumers of the same underlying capability: a **production
retrieval path** (latency- and cost-sensitive, needs to scale to 50M images and serve queries fast) and
an **internal analysis tool** (batch, occasional, values interpretability over raw speed). This maps
directly onto this lecture's two halves.

**For the retrieval path — an autoencoder-family embedding, not a from-scratch design.** Use a
pretrained vision encoder's embeddings (this is squarely §11's eigenfaces-style representation-learning
territory, generalised) — but specifically seek out or fine-tune an **MRL-trained** version if
available, or apply MRL-style multi-scale training if fine-tuning in-house, so a single model serves
both a cheap first-pass ANN index (say 256-dim) and a precise reranking stage (full dim), exactly per
§39's 2-stage pattern — this is the direct reason to prefer MRL over a plain fixed-size embedding here:
one training run, one deployed model, two tiers of the same representation.

**For the analyst tool — a sparse autoencoder trained on top of the frozen embeddings, not the raw
embeddings directly.** Raw embedding dimensions are typically not individually interpretable (§17's
polysemanticity point, imported from the LLM-interpretability discussion to a vision setting) — an
analyst staring at "dimension 412 is high" learns nothing. A sparse autoencoder trained to reconstruct
the frozen embedding, with a much wider hidden layer and a strong sparsity penalty, is far more likely
to surface individual units that correspond to human-recognisable visual concepts (a specific
material, a colour pattern, a silhouette), which is what "spot mislabeled categories by browsing
clusters" actually needs.

**What I'd explicitly rule out and say why:**
- **Training a VAE for this** — the retrieval task needs accurate, discriminative representations, not
  a smooth generative latent space; a VAE's KL regularisation term actively works against
  reconstruction sharpness (§23's tension), which is the wrong tradeoff to accept for a task with no
  generation requirement at all.
- **Post-hoc PCA on the base embedding to shrink it for the retrieval tier** — §38.4's argument
  directly: PCA orders by raw embedding variance, not by visual-similarity usefulness, and would very
  plausibly discard exactly the fine-grained dimensions that distinguish visually similar-but-distinct
  products, which is the entire point of the feature.
- **Fine-tuning the base vision encoder itself with full backprop through all its weights** — if
  fine-tuning is warranted at all (e.g. to adapt a general vision model to this specific catalogue's
  visual style), reach for **LoRA** first rather than full fine-tuning: the compute and storage savings
  (potentially 100×+ fewer trainable parameters, per §32.2) make iterating on this within a one-quarter
  timeline far more realistic, with zero added inference latency once merged.

**What I'd flag as needing a decision from the team rather than resolving myself:** whether "visually
similar" should mean similar in raw appearance or similar in semantic category (a red dress and a red
handbag are visually similar in colour but not in the way a shopper likely means "similar") — this is a
data/labelling question this scenario doesn't specify, and it changes which contrastive training
objective (if any fine-tuning happens) should drive the embedding, upstream of every method discussed
in this file.

### Leadership Principles tie-in

**Dive Deep.** Question 5 above is the model: the shallow answer to "why do VAEs sometimes ignore parts
of their latent space?" is "it's a known issue, posterior collapse." The deep answer traces it to the
specific *direction* of the KL divergence in the ELBO — reverse KL is zero-forcing, so the encoder pays
no penalty for leaving prior-mass uncovered — which means the failure mode was *predictable* from the
math, not merely observed empirically after the fact. *"When our VAE-based feature started producing
low-diversity outputs, I traced it to the reverse-KL term's zero-forcing behaviour rather than treating
it as an unexplained training instability, and that pointed directly at increasing the decoder's
capacity constraint as the fix."*

**Frugality** fits §32's LoRA numbers directly: a 256× reduction in trainable parameters for a single
matrix, with **zero added inference cost** once merged, is as close to a free lunch as fine-tuning gets
— and being the person who reaches for it before proposing a full fine-tuning run (with its
proportionally larger compute bill and storage-per-task cost) is exactly this LP in a technical
decision.

**Invent and Simplify** fits §38 well: Matryoshka Representation Learning's entire contribution is
"apply the *same* loss you already have, to several nested slices of the same vector, and average" —
no new loss function, no new architecture. Recognising that a genuinely valuable capability (a
runtime-adjustable embedding size) can come from a small change to *how* an existing objective is
applied, rather than from a more complex new mechanism, is a good model for this LP in a design
decision.

> 🎯 **stretch — nice to know, not expected for an intern:** the full Diffusion Transformer (DiT)
> architecture and its patchification scheme; NF4 quantisation's exact numerical construction inside
> QLoRA; the precise SGD-with-negative-sampling training details of vector-quantisation codebooks
> (codebook collapse and its mitigations, e.g. EMA updates); the formal intrinsic-dimension arguments
> (Aghajanyan et al., 2020) that motivated LoRA before LoRA itself was published; the exact contrastive
> loss formulations (InfoNCE and variants) underlying modern embedding-model training that MRL is
> layered on top of. Knowing these exist and roughly what they claim is enough.

---

## Glossary

| Term | Definition |
|---|---|
| **Autoencoder** | A network trained to reconstruct its input through a narrower bottleneck layer. A *linear* one, with MSE loss, is provably equivalent to PCA (Baldi & Hornik, 1989). §5 |
| **Bottleneck** | The narrow middle layer forcing information compression. Can be narrow in *width* (plain AE) or in *activity* (sparse AE — wide but few units fire). §6, §17 |
| **Commitment loss** | VQ-VAE's term pulling the encoder output toward its assigned codebook entry (stop-gradient on the codebook side). Prevents the encoder from drifting far from any codebook vector. §26 |
| **Codebook** | VQ-VAE's learned set of $K$ discrete vectors that continuous encoder outputs are snapped to. The "vocabulary" a Transformer can then be trained over. §26 |
| **Denoising autoencoder** | Trained to reconstruct the *original* input from a *corrupted* version. Cannot take the identity-function shortcut a plain AE can, forcing it to learn real structure. §12 |
| **DiT (Diffusion Transformer)** | Replaces a diffusion model's U-Net with a plain Transformer over spacetime-patch tokens. Used by Sora. §30 |
| **DoRA** | Weight-Decomposed LoRA. Decomposes $W = m\cdot(V/\|V\|)$; trains magnitude $m$ separately from direction, applying LoRA's $BA$ only to direction. Liu et al. (2024). §35 |
| **ELBO (Evidence Lower Bound)** | The VAE's training objective: reconstruction term minus a KL regularisation term. A lower bound on the true (intractable) data log-likelihood. §23 |
| **Intrinsic rank (of a weight update)** | LoRA's empirical premise: $\Delta W$, the change fine-tuning wants to make, behaves as a low-rank matrix even though $W$ itself is full rank. §32 |
| **Latent diffusion** | Running the entire iterative denoising process inside a VAE's compressed latent space rather than in pixel space. What makes Stable Diffusion, Sora and Veo computationally tractable. §29–§30 |
| **LoRA (Low-Rank Adaptation)** | Freeze $W$, learn only $\Delta W = BA$ with $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, $r\ll\min(d,k)$. $B$ initialised to zero. Hu et al. (2021/2022). §31–§33 |
| **Matryoshka Representation Learning (MRL)** | Trains one embedding such that every prefix $z_{1:m}$ is independently a valid, high-quality embedding. Same loss, applied to nested slices, averaged. Beats post-hoc PCA because it's task-aware. Kusupati et al. (2022). §36–§39 |
| **Monosemantic feature** | A single interpretable concept represented by one unit — the goal sparse autoencoders pursue for LLM interpretability, as a countermeasure to polysemanticity. §17.3 |
| **Polysemanticity** | A single neuron in a trained network firing for several unrelated concepts, due to superposition — more concepts than dimensions. The problem sparse autoencoders address. §17.3 |
| **Posterior collapse** | A VAE's known failure mode: the encoder ignores part of the latent space's capacity, exploiting reverse KL's zero-forcing property to pay no penalty for leaving prior-mass "uncovered." §23.1, §25 |
| **QLoRA** | Quantises the frozen base model to 4-bit (NF4) while keeping LoRA adapters at higher precision — reduces the *base model's* memory footprint, complementary to (not competing with) DoRA. §35 |
| **Reparameterisation trick** | Rewriting $z\sim N(\mu,\sigma^2)$ as $z = \mu + \sigma\odot\epsilon$, $\epsilon\sim N(0,I)$, making the sample a differentiable, deterministic function of $\mu,\sigma$. Enables backprop through a VAE's stochastic bottleneck. §22 |
| **Reverse KL** (in the VAE's ELBO) | $\mathrm{KL}(q(z|x)\|p(z))$ — expectation under $q$. Mode-seeking, zero-forcing (per [Part 2 §24](dimensionality-reduction-02.md)); the mechanism behind posterior collapse. §23.1 |
| **Sparse autoencoder** | Adds a KL-divergence penalty between a target activation rate $\rho$ and the actual rate $\hat\rho_j$, forcing most units off most of the time. Wide layer, sparse activity — a different shape of bottleneck. Now central to LLM interpretability. §16–§18 |
| **Straight-through estimator** | On the forward pass, use the true discrete operation (e.g. $\arg\min$); on the backward pass, pass the gradient through as if it had been the identity. A deliberate, biased approximation. §26.2 |
| **VAE (Variational Autoencoder)** | Encodes each input as a distribution $N(\mu,\sigma^2)$ rather than a point, and regularises that distribution toward a fixed prior via KL. Makes the latent space smooth and sampleable — generative, unlike a plain AE. §19–§23 |
| **Vector quantisation** | Snapping a continuous encoder output to its nearest vector in a learned codebook — the mechanism that makes a VQ-VAE's latents discrete. §26 |
| **VQ-VAE** | A VAE variant with a discrete (vector-quantised) latent space instead of a continuous Gaussian one. An image becomes a grid of codebook indices — a Transformer's native input format. van den Oord et al. (2017). §26–§28 |

---

## Check yourself

Work these before reading the answers. Questions 9–12 require combining two concepts.

1. What theorem tells you a linear autoencoder's latent subspace, and what does it NOT tell you about the individual latent dimensions?
2. A denoising autoencoder is trained with noise level 0.5, and a colleague argues "just set the noise to 0 for cleaner training." What breaks?
3. Compute $D_{KL}(\rho\|\hat\rho)$ for $\rho = 0.1$ and $\hat\rho = 0.1$. What does this confirm about the sparsity penalty's design?
4. Write the reparameterisation trick's two partial derivatives, and state which one depends on $\epsilon$ and why that makes sense.
5. In VQ-VAE's three-term loss, which term's gradient updates the codebook, and which updates the encoder? What does the stop-gradient operator control?
6. A 1024×1024 RGB image is diffused at full pixel resolution. Compute the token count for self-attention and the number of pairwise attention scores.
7. Compute LoRA's parameter count and reduction factor for $d=k=8192$, $r=16$.
8. Explain, in one sentence each, what QLoRA and DoRA each add on top of plain LoRA, and confirm they are not mutually exclusive.
9. **(Combines two)** A VAE trained with a very powerful (high-capacity) decoder produces low-diversity samples. Diagnose this using the ELBO's two terms and the KL direction.
10. **(Combines two)** You have an MRL-trained 1024-dim embedding model. A colleague proposes running PCA on its 1024-dim output to get a cheap 128-dim version for a new use case, instead of just truncating to $z_{1:128}$. Evaluate the proposal.
11. **(Combines two)** Explain why "the weight update $\Delta W$ has low rank" (LoRA's premise) and "the top singular vectors are the topics" (Part 2's LSI) are, structurally, the same kind of empirical claim about a matrix.
12. **(Combines two)** You're told a VQ-VAE-based image generation pipeline is producing images with visible blocky, repetitive-looking artefacts, especially in flat/textureless regions. Propose a specific, mechanistic hypothesis rooted in this lecture's content.

<details>
<summary><b>Answers</b></summary>

**1.** Baldi & Hornik (1989): a linear autoencoder with MSE loss has, as its global optimum, exactly
the PCA **subspace** — the span of the top-$k$ principal components.

**What it does NOT tell you:** the individual latent dimensions the autoencoder learns are not
guaranteed to be *aligned* with PCA's individual eigenvectors, nor ordered by variance the way PCA's
components are (largest eigenvalue first). The theorem is about the subspace spanned, not about a
canonical basis within it — an arbitrary rotation of the PCA components within that subspace is an
equally valid solution for the linear autoencoder.

**2.** At noise level 0, $\tilde x = x$ exactly — there is no corruption at all, so the "denoising"
autoencoder degenerates back into a **plain** autoencoder. It regains the plain AE's exact weakness:
the identity function becomes a perfectly valid (zero-loss) solution, so nothing prevents the network
from simply memorising each training example rather than learning generalisable structure. The whole
point of denoising training is that the loss target ($x$) differs from what the network sees
($\tilde x$) — remove the noise and that gap disappears.

**3.** $D_{KL}(0.1\|0.1) = 0.1\log(0.1/0.1) + 0.9\log(0.9/0.9) = 0.1\log(1) + 0.9\log(1) = 0 + 0 =
\mathbf{0}$.

**Confirms:** the penalty is exactly zero when the actual activation rate matches the target — the
network incurs no sparsity cost at all once it's hit the target rate $\rho$. This matches the deck's
own plot, which shows the KL curve touching zero exactly at $\hat\rho = \rho$ (the plotted example used
$\rho = 0.25$) and rising on both sides — confirming the penalty is a genuine minimum-at-target
function, not merely small near the target.

**4.** $\dfrac{\partial z}{\partial\mu} = 1$ and $\dfrac{\partial z}{\partial\sigma} = \epsilon$.

**The $\sigma$-derivative depends on $\epsilon$; the $\mu$-derivative does not** — and this makes sense
directly from $z = \mu + \sigma\epsilon$: $\mu$ contributes to $z$ additively with a fixed coefficient
of exactly 1 regardless of the random draw, while $\sigma$ contributes *multiplicatively*, scaled by
whatever $\epsilon$ happened to be sampled — so how much a nudge to $\sigma$ moves $z$ depends on the
specific (random) value of $\epsilon$ for that particular sample.

**5.** The **codebook loss** ($\|\mathrm{sg}[E(x)] - e_k\|^2$) updates the **codebook** $e_k$ — the
stop-gradient on $E(x)$ blocks gradient from reaching the encoder through this term, so only $e_k$
moves. The **commitment loss** ($\beta\|E(x) - \mathrm{sg}[e_k]\|^2$) updates the **encoder** — the
stop-gradient on $e_k$ this time blocks gradient from updating the codebook through this term.

**What `sg[·]` controls:** it makes its contents behave as a constant during backpropagation (zero
local gradient), even though the forward-pass value is used normally — a way of directing exactly which
parameters a given loss term is allowed to influence.

**6.** $1024\times1024 = 1{,}048{,}576$ spatial positions → **1,048,576 tokens** for self-attention.

Pairwise attention scores: $n^2 = (1{,}048{,}576)^2 \approx \mathbf{1.10\times10^{12}}$ — roughly 1.1
trillion score computations for one attention layer, at one denoising step. (For scale: this is
roughly $16\times$ the $6.87\times10^{10}$ figure the deck computed for a 512×512 image, since
doubling linear resolution quadruples the token count and therefore raises the *squared* attention
cost by a factor of $4^2=16$.) This number alone is a complete, self-contained justification for why
pixel-space diffusion at any serious resolution is infeasible.

**7.** Full fine-tune: $d\times k = 8192^2 = \mathbf{67{,}108{,}864}$ parameters.

LoRA: $B(d\times r) + A(r\times k) = 8192\times16 + 16\times8192 = 131{,}072 + 131{,}072 =
\mathbf{262{,}144}$ parameters.

Reduction: $67{,}108{,}864 / 262{,}144 = \mathbf{256\times}$.

*(Same ratio as the worked example at $d=k=4096, r=8$ in §32.2 — because the ratio
$dk/[r(d+k)]$ simplifies to $d/(2r)$ when $d=k$, and both $(4096,8)$ and $(8192,16)$ give
$d/(2r) = 256$.)*

**8.** **QLoRA** quantises the **frozen base model's** weights to 4-bit precision, reducing the memory
required to *hold* the base model during fine-tuning (letting larger models fit on smaller hardware) —
it doesn't change how the adapter itself is structured. **DoRA** decouples magnitude from direction
**within the adapter's own update**, changing how the low-rank correction is parameterised to better
match full fine-tuning's behaviour — it doesn't touch base-model precision at all.

**Not mutually exclusive:** they operate on different objects (base-model precision vs. adapter
parameterisation) and can be combined — a 4-bit-quantised base model fine-tuned with a
magnitude/direction-decoupled adapter — with no structural conflict between them.

**9.** **Diagnosis: posterior collapse, made worse specifically by decoder capacity.** The ELBO's
reconstruction term can be satisfied well by a sufficiently powerful decoder even from a very
*impoverished* latent code — a high-capacity decoder can "make up" plausible-looking detail on its own,
requiring less genuine information from $z$ to reconstruct well. Meanwhile the KL term
($\mathrm{KL}(q(z|x)\|p(z))$, reverse KL) imposes **no penalty at all** for the encoder collapsing most
inputs' $q(z|x)$ toward the prior and using only a fraction of the latent's nominal capacity — that's
exactly the zero-forcing property, which only penalises $q$ for placing mass where $p$ has none, never
for failing to spread out and use $p$'s full support.

**Put together:** a strong decoder reduces the reconstruction term's *pressure* to use $z$
informatively, while the KL term actively *rewards* collapsing toward the simple prior (zero KL cost).
Both forces push the same direction — toward an underused, low-diversity latent space — which is
exactly the observed symptom.

**Standard mitigations, consistent with this diagnosis (not required by the deck, but a natural next
step):** weaken the decoder, anneal $\beta$ up gradually rather than fixing it from the start
("KL annealing"), or use $\beta < 1$ to reduce the KL term's relative pull.

**10.** **The proposal should be rejected in favour of simply truncating.**

For an **MRL-trained** model specifically, $z_{1:128}$ — the first 128 coordinates of the trained
embedding — is *already* a purpose-trained, task-aware 128-dimensional representation, per §38's
nested loss: it was directly optimised (as one of the terms in $\mathcal{L}_{\text{Matryoshka}}$, or at
minimum benefiting from the coarse-to-fine packing that training induces even at untrained
intermediate sizes) to be a valid, high-quality embedding **on its own**.

Running post-hoc PCA on the full 1024-dim output instead reintroduces exactly the failure mode §38.4
describes: PCA would choose its 128 retained directions by **raw output variance**, a criterion with no
guaranteed relationship to whatever the new use case actually needs — potentially discarding exactly
the discriminative-but-lower-variance directions the task depends on ([Part 2 §11's](dimensionality-reduction-02.md)
eigenfaces lesson, again).

**The one case where PCA might still be worth trying:** if the new use case's notion of "similarity" is
meaningfully different from whatever the MRL training's original task loss optimised for — in which
case *neither* the trained prefix *nor* generic PCA is guaranteed to be right, and this becomes an
empirical question to test rather than one this scenario alone can settle. But "PCA instead of the
trained prefix, for the *same* kind of task the model was trained on" is not a defensible default.

**11.** Both are the identical structural claim: **"most of this matrix's information is captured by a
small number of its dominant directions."**

[Part 2 §7's LSI](dimensionality-reduction-02.md) makes this claim about a **term × document** matrix
— the top few singular vectors, ranked by singular value, turn out to correspond to interpretable
topics because words that co-occur across many documents load heavily onto the same dominant
direction, and the *low-rank truncation* of that matrix (keeping only the top $k$ singular
vectors/values) is a good approximation of the whole thing.

**LoRA's premise makes the same claim about a *weight-update* matrix** $\Delta W$ — most of the
information in "how should these weights change for this task" is captured by a small number of
dominant directions (rank $r \ll \min(d,k)$), and a low-rank factorisation $BA$ approximates the full
update well.

**The unifying fact:** both are instances of the empirical observation that real matrices arising from
structured processes (word co-occurrence patterns; task-specific weight adaptations) are frequently
**far from full rank in practice**, even though nothing in their formal definition (an $m\times n$
matrix can have rank up to $\min(m,n)$) guarantees this — and Eckart–Young
([Part 2 §2.3](dimensionality-reduction-02.md)) is what makes exploiting that observation
mathematically well-founded, once you've verified (empirically, in both cases) that it holds.

**12.** **Hypothesis: the VQ-VAE's codebook is exhibiting some form of collapse or under-utilisation —
few codebook entries are being used for large, low-detail regions, forcing repeated/blocky patterns
where a genuinely continuous gradient of pixel values would look smoother.**

**The mechanism, traced through this lecture's content:** in a flat, textureless region of an image
(say, a clear sky, or a plain wall), the encoder's raw continuous output $z_e$ likely varies only
subtly across nearby spatial positions — but vector quantisation forces every one of those subtly
different $z_e$ values to **snap to the single nearest discrete codebook entry** (§26's
$\arg\min_i\|z_e - e_i\|_2$ operation). If the codebook doesn't have enough distinct entries
"covering" that region of the continuous space finely enough, nearby positions that should have subtly
different reconstructed values instead get mapped to the *identical* discrete code repeatedly —
producing the blocky, repetitive appearance described, precisely in flat regions where the true signal
has little variation to begin with (making the quantisation error more visually noticeable relative to
the — small — genuine signal).

**Why this specifically implicates the straight-through estimator too, as a contributing factor rather
than the root cause:** because the gradient reaching the encoder through quantisation is a biased
approximation (§26.2) rather than exact, the encoder's outputs for these low-variation regions may not
be receiving a fully accurate training signal about how their fine differences *should* map to
different codes — compounding whatever coarseness already exists in the codebook itself.

**What I'd check, concretely, before proposing a fix:** log codebook usage statistics during training —
if a large fraction of the $K$ codebook entries are rarely or never selected (a well-documented failure
mode sometimes called "codebook collapse"), that directly confirms the hypothesis and points at
mitigations like exponential-moving-average codebook updates or periodically re-initialising unused
codes — both standard fixes in the VQ-VAE literature, cited here as the natural next step rather than
as something this lecture itself covered.

</details>

---

## Going deeper

Ranked by importance. Difficulty markers: `intro` / `solid` / `hard`. Items 1–9 are the deck's own
citations.

### The deck's own citations

1. **Lilian Weng, "From Autoencoder to Beta-VAE" (2018), lilianweng.github.io** — `intro`. The deck's
   single most-cited source, underlying nearly every diagram in §5–§28. Exceptionally clear, and the
   single best starting point if any of Part 1 felt rushed.
2. **van den Oord, Vinyals & Kavukcuoglu, "Neural Discrete Representation Learning" (NeurIPS 2017)** —
   `solid`. The VQ-VAE paper. The straight-through estimator and the three-term loss (§26) are derived
   here in full, with the ablations that motivated each term.
3. **Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen, "LoRA: Low-Rank Adaptation of Large Language
   Models" (ICLR 2022)** — `solid`. The LoRA paper. Section 7's discussion of *why* $\Delta W$ has low
   rank (connecting to Aghajanyan et al.'s intrinsic-dimension work) is worth reading in full; it's more
   nuanced than the one-line "key insight" bullet this lecture necessarily compresses it to.
4. **Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024)** — `solid`. The magnitude/
   direction decomposition and the empirical comparison against full fine-tuning's own update pattern,
   which is the paper's actual contribution beyond the architecture diagram.
5. **Kusupati, Bhatt, Rege et al., "Matryoshka Representation Learning" (NeurIPS 2022)** — `solid`. The
   original MRL paper. Contains the full experimental comparison against independently-trained
   fixed-size models and against post-hoc dimensionality reduction, across vision, language and
   vision+language — the source of §39's headline numbers.
6. **Rege, A. (2024), "Matryoshka Representation Learning from the Ground Up", aniketrege.github.io** —
   `intro`. The deck's own secondary MRL citation — an accessible walkthrough, good as a companion to
   the denser original paper.
7. **Sebastian Raschka, "LoRA and DoRA From Scratch", magazine.sebastianraschka.com** — `intro`,
   hands-on. The source of §35's diagram. Includes runnable code implementing both from first
   principles — the best way to make the magnitude/direction decomposition concrete.
8. **Bishop, PRML §10.1.2** — `hard`. Already the [Part 2](dimensionality-reduction-02.md) citation for
   forward-vs-reverse KL; directly relevant again here for §23.1's ELBO analysis.
9. **Weng, L., "From Autoencoder to Beta-VAE"** — see item 1; also the direct source for the VQ-VAE
   figure reproduced in §26.

### Beyond the deck

10. **Kingma & Welling, "Auto-Encoding Variational Bayes" (ICLR 2014)** — `hard`. The original VAE
    paper, and the source of the reparameterisation trick (§22) and the ELBO derivation (§23) in their
    full mathematical form, including the Gaussian-specific closed-form KL term the deck's slide
    doesn't spell out.
11. **Rombach, Blattmann, Lorenz, Esser & Ommer, "High-Resolution Image Synthesis with Latent Diffusion
    Models" (CVPR 2022)** — `solid`. The Stable Diffusion paper. Directly justifies §29–§30's cost
    argument with the original ablations comparing pixel-space and latent-space diffusion training
    cost.
12. **Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)** — `hard`. The DDPM
    paper — the foundational diffusion-model formulation §29.1's forward/reverse process description is
    built on. Read this before Rombach et al. if diffusion itself is new.
13. **Anthropic, "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
    (2023)** and its follow-up **"Scaling Monosemanticity" (2024)** — `solid`, both interactive. The
    primary references for §17.3's LLM-interpretability claim, with extensive feature visualisations.
    The best way to see what a "monosemantic feature" actually looks like in a real trained model.
14. **Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICCV 2023)** — `hard`. The DiT
    paper §30's video-diffusion subsection references. Establishes the Transformer-over-latent-patches
    architecture Sora builds on.
15. **Aghajanyan, Zettlemoyer & Gupta, "Intrinsic Dimensionality Explains the Effectiveness of Language
    Model Fine-Tuning" (2020)** — `hard`. Predates LoRA and is its direct intellectual ancestor —
    measures the intrinsic dimension of fine-tuning updates empirically, which is the finding LoRA's
    "key insight" bullet compresses into one sentence.
16. **Dettmers, Pagnoni, Holtzman & Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs"
    (NeurIPS 2023)** — `solid`. The QLoRA paper §35's closing lineage figure references — covers the
    NF4 quantisation scheme and double quantisation in full detail.
17. **PyTorch / Hugging Face `peft` library documentation** — `intro`, hands-on. Runnable
    implementations of LoRA, QLoRA and DoRA that can be applied to a real pretrained model in a few
    lines — the fastest way to turn §31–§35's equations into a working experiment.

---

## 📊 Summary

| | |
|---|---|
| **Source** | `output/Lecture_09 - Module 3 Dimensionality Reduction Part 3` — 108 raw frames, **47 distinct slide states** |
| **Runtime** | 56:28 · **Ravi Sankar Adepu** |
| **Sections** | 25 numbered sections across four parts (Autoencoders & VAEs §5–§28 · Latent Diffusion §29–§30 · Low-Rank Methods in Transformers §31–§35 · Embedding Models & Matryoshka §36–§39) |
| **Worked examples** | 10, every one carried to a final, checked number |
| **Derivations** | The Baldi–Hornik linear-autoencoder-is-PCA theorem stated and connected to Part 2 · the sparse autoencoder's KL penalty derived at its endpoints (divergence to $\infty$ at $\hat\rho\to0,1$) · the reparameterisation trick fully derived (distributional correctness verified, both partial derivatives computed) · the VQ-VAE three-term loss unpacked term by term via stop-gradient placement · the straight-through estimator named as a deliberate approximation, not an exact gradient · pixel-space diffusion's linear and quadratic costs computed exactly (786,432 values; $6.87\times10^{10}$ attention scores) and the latent-space saving verified (64× / 4,096×) · LoRA's parameter-reduction ratio computed for two real layer sizes (256× at both $d{=}4096,r{=}8$ and $d{=}8192,r{=}16$) · DoRA's magnitude/direction decomposition derived as an algebraic identity · MRL's nested loss derived to show *why* it produces coarse-to-fine ordering, then contrasted precisely against post-hoc PCA using Part 2's own variance-vs-usefulness distinction |
| **Interactive blocks** | 5 (the linear-AE-as-rank-k-PCA slider · the denoising-AE noise slider · point vs distribution encoding · LoRA rank vs parameter count · truncating the Matryoshka dolls) |
| **Interview questions** | 12 with model answers (3 combining concepts), 8 depth probes, 3 whiteboard derivations, 1 applied scenario, 3 Leadership Principles |
| **Cross-references** | To [Part 1](dimensionality-reduction-01.md) (the low-variance-signal trap) and extensively to [Part 2](dimensionality-reduction-02.md) (§13's autoencoder-is-PCA theorem — the anchor of this entire file's Part 1; §2–§4's SVD/Eckart–Young — the anchor of §31–§35; §9's variance-vs-task-relevance distinction — the anchor of §38.4; §11's eigenfaces nuisance-variance lesson — traced through three separate sections in Check-yourself Q11; §12's noise-in-discarded-directions argument — underlying §12–§14's denoising section; §14's Swiss-roll flat-vs-curved distinction — underlying §6.2's "nonlinearity is the superpower"; §23–§24's KL divergence and its forward/reverse asymmetry — underlying §17, §23.1 and §25 directly) |
| **⚠️ Flags left in the file** | The instructor/topic mismatch against Part 1's forward-reference prediction, resolved at the top rather than left implicit · one long single-slide capture stretch (frames 73–99, MRL) explained as narration-without-visual-change, not a content gap · the denoising-demo's two captured states flagged as not a controlled paired comparison (different random noise draws) · the "up to ~10,000×" LoRA figure flagged as an upper-end number, not a universal constant, with the actual computed ratios shown to differ by configuration · DoRA's "consistently beats LoRA" result flagged as an empirical finding on tested benchmarks, not a proven guarantee · the polysemanticity/SAE-interpretability research direction flagged as genuinely open (🔬) · Sora's exact pixel-count figure noted as approximately, not exactly, matching the deck's "~2.5B" under one reasonable frame-rate/resolution assumption |
