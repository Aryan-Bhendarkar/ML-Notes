---
title: "Unsupervised Learning — Part 4: Diffusion Models and Flow Matching"
topic: unsupervised-learning
lecture: 13
source: "output/Lecture_13 - Module 4 Unsupervised Learning Part 4"
slides: 22
video: "https://www.youtube.com/watch?v=uMB483syr98"
runtime: "40:38"
---

# Unsupervised Learning — Part 4
### Diffusion Models and Flow Matching: Learning to Generate by Learning to Denoise

---

## 📋 About this lecture and its capture

[Part 3](unsupervised-learning-03.md) ended on a cliffhanger: VAEs and GANs had shown two
fundamentally different ways to learn a generative model — explicit density (ELBO) vs implicit
generative model (adversarial) — but neither was state-of-the-art for image generation. **This
lecture covers the two methods that overtook both: Diffusion Models and Flow Matching.**

| | Section | Runtime | Covers |
|---|---|---|---|
| **A** | Diffusion Models | 0:00 – ~25:00 | Forward process · reverse process · connection to hierarchical VAEs · ELBO and denoising matching · noise prediction vs score prediction · ancestral sampling · score functions |
| **B** | Flow Matching | ~25:00 – 40:38 | Probability paths · velocity fields · conditional flow matching (CFM) loss · Euler ODE solver · unifying connection to diffusion |

Reconstructed from the raw capture in `output/`, the deduplicated deck contains **22 distinct slide
states**. The original 52-frame raw capture was reduced to those 22 — **30 raw frames were not kept
as a slide's representative frame.** A frame-by-frame audit of all 30 dropped frames (as part of
this file's quality review) confirmed every one is a genuine intermediate build-state or
progressive-annotation duplicate of an adjacent kept slide, with no unique content lost to dedup
mechanics itself. The full YouTube transcript was obtained and verified against each slide; no
content gaps were identified. *(An earlier version of this note and the module README both stated
"12 out of 34 raw runs were dropped" — arithmetically inconsistent with the actual 52-raw/22-deduped
counts confirmed directly from the `output/` and `slides_deduped/` directories; corrected here to
the verified 52→22 figures.)*

> ✅ **Capture quality: good, with caveats.** 22 deduped slides over 40 minutes. The instructor
> explicitly states at the start: *"I'm going to cover these slides a bit quickly because we are
> short of time."* This is a rapid survey lecture — the notes therefore teach each concept in full
> depth far beyond what the slides show, filling in the background the lecture assumes but does not
> explain. The YouTube transcript is the primary source for intuition and caveats; slides provide
> the formal framework.
>
> > ⚠️ **Module position note:** Part 3's Table of Contents promised five sections beyond GMM/EM
> > that its captured video didn't deliver: Generative Modeling Overview, VAEs, GANs, Diffusion
> > Models, Flow Matching. Part 3 delivered the first three (Overview, VAEs, GANs). **This lecture
> > delivers the last two (Diffusion Models, Flow Matching).** The module is now complete.

---

## How to read this document

This lecture has a clean two-part structure, but the real shape is a *spiral* — Part B revisits
Part A's machinery from a more general perspective:

```
Part 3's VAE:    x → z in ONE step, learn encoder + decoder
                  │
                  ▼   "What if we break that one step into T tiny steps?"
                  │
Part A §1-3:     Diffusion = hierarchical VAE with FIXED forward (add noise T times)
                  │
Part A §4-8:     Learn the REVERSE: denoise T times, via ELBO → denoising matching → score
                  │
                  ▼   "The forward process is arbitrary — why stick to adding Gaussian noise?"
                  │
Part B §10-12:   Flow Matching = choose ANY smooth path from noise to data
                  │
Part B §13-14:   Solve the ODE along that path → deterministic generation in fewer steps
                  │
                  ▼
Part B §15:      Diffusion is a SPECIAL CASE of Flow Matching
```

If you are revising under time pressure: **Part A's score function equivalence (§6, §8) and
Part B's conditional flow matching loss (§12) are the interview core.** Both are non-obvious
ideas made rigorous, and both are increasingly asked in applied-scientist interviews as diffusion
models become production infrastructure.

Everything in a `🧪 Worked example` block should be reproducible by you on paper.

---

## What you'll understand after reading this

By the end of this document you will be able to:

1. **Explain why adding noise to data and learning to reverse it is a valid generative strategy**
   — and connect this idea directly back to [Part 3's](unsupervised-learning-03.md) VAE framework
   as a hierarchical generalisation.
2. **Derive the forward diffusion process** from first principles: how scaling and adding Gaussian
   noise at each step transforms any data distribution into a standard Gaussian, and why this is
   guaranteed by repeated convolution.
3. **State the reverse process** — how a neural network learns to undo each noise step — explain
   what the network predicts, why it needs the time step $t$, and derive the connection to the
   ELBO.
4. **Connect diffusion models to VAEs** via the ELBO: identify the reconstruction term, the KL
   term, and the novel **denoising matching term** that does all the useful learning.
5. **Explain the score function** of a probability distribution, compute it analytically for a
   Gaussian, and state precisely why learning the denoising direction is equivalent to learning
   the score.
6. **Describe ancestral sampling** — the iterative procedure that generates a sample from pure
   noise — and explain the role of the random noise term at each step.
7. **Define a probability path** from noise to data, explain what a velocity field is, and state
   how flow matching generalises diffusion in one sentence.
8. **Derive the conditional flow matching (CFM) loss** from first principles and explain why it
   simplifies to a mean-squared-error objective that trains a neural network to predict velocity.
9. **Solve an ODE using Euler's method** to generate a sample from a trained flow model, and
   explain the trade-off between step count and sample quality.
10. **State the precise relationship between diffusion models and flow models** — that diffusion
    is a special case of flow for a particular choice of probability path — and explain why flow
    matching typically needs fewer function evaluations.

---

## Before we start: what you need to know

### Prerequisite 1 — Everything from Parts 1–3, assumed fresh

This lecture does not re-derive any of the following:
- [Part 2's](unsupervised-learning-02.md) GMM/EM algorithm, the ELBO/KL decomposition, or the
  E-step/M-step logic
- [Part 3's](unsupervised-learning-03.md) VAE objective (reconstruction + KL regularisation), the
  reparameterization trick, or posterior collapse
- [Part 3's](unsupervised-learning-03.md) GAN framework (implicit generative models, the
  discriminator trick)

All three are used immediately. If any of these isn't fresh, this lecture will feel like a list of
formulas rather than a continuation of one argument.

### Prerequisite 2 — VAE as a hierarchical encoder

> **Hierarchical VAE** — a VAE where the encoding from input $x$ to latent $z$ happens in
> multiple stages, each stage producing an intermediate latent that is progressively more Gaussian.
>
> *In everyday words:* instead of forcing a single neural network to compress a complex image all
> the way to a simple Gaussian in one step, you break the compression into many small steps, each
> one making the data slightly more "random-looking" until it becomes pure noise.
>
> *Concretely:* in an ordinary VAE, $x \to z$ in one step. In a hierarchical VAE with $T$ stages,
> $x \to x_1 \to x_2 \to \cdots \to x_T = z$, where each $x_t$ is closer to Gaussian than
> $x_{t-1}$.
>
> *Why it exists:* a single-step encoding from complex data to a simple Gaussian is an extremely
> hard function to learn. Breaking it into many easy steps — each one making only a small change
> — is dramatically easier, and this is the core insight behind diffusion models.

### Prerequisite 3 — Adding independent Gaussian noise produces convolution

> **Convolution of distributions** — when you add two independent random variables, the
> distribution of their sum is the *convolution* of their individual distributions.
>
> *In everyday words:* if you take a sharp photograph and add TV static (random pixel noise), the
> result is the original photograph *blurred* by the static's distribution. The blurring *is* the
> convolution.
>
> *Concretely:* if $X$ has distribution $p_X$ and $Y$ has distribution $p_Y$ (independent), then
> $Z = X + Y$ has distribution $p_Z = p_X * p_Y$ (convolution).
>
> *Why it matters here:* when you add Gaussian noise to a data distribution at each step, the
> result is the data distribution *convolved with* a Gaussian — which smooths it. After enough
> steps of this smoothing, any distribution becomes indistinguishable from a Gaussian. This is the
> mathematical reason the forward process "works."

### Prerequisite 4 — Ordinary Differential Equations (ODEs)

> **Ordinary Differential Equation (ODE)** — an equation that relates a quantity to its rate of
> change. The simplest form: $\frac{dx}{dt} = v(x, t)$, meaning "the velocity of $x$ at time $t$
> is given by $v$."
>
> *In everyday words:* if you know how fast something is moving at every point in time and space,
> you can figure out where it ends up by adding up all those tiny movements. That "adding up" is
> what solving the ODE means.
>
> *Concretely:* if $\frac{dx}{dt} = 1$ (constant velocity), then starting at $x(0) = 0$, after
> time $t$ you have $x(t) = 0 + 1 \times t = t$. Simple enough. The hard case is when $v$ depends
> on $x$ itself — then you need numerical methods.
>
> *Why it matters here:* flow matching models learn a velocity field $v_\theta(x, t)$, and
> generating a sample means *solving the ODE* $\frac{dx}{dt} = v_\theta(x, t)$ from noise ($t=0$)
> to data ($t=1$). Euler's method is the simplest solver.

---

## The big picture

The entire unsupervised learning module has been answering one question with increasing
sophistication: **"What is the underlying distribution of this data, and can we generate new
samples from it?"**

| Lecture | Answer |
|---|---|
| [Part 1](unsupervised-learning-01.md) — K-Means, Hierarchical, DBSCAN | "The data falls into $K$ hard clusters" |
| [Part 2](unsupervised-learning-02.md) — GMM/EM | "Each cluster is a Gaussian with learnable shape; membership is soft" |
| [Part 3](unsupervised-learning-03.md) — VAEs and GANs | "Drop the parametric assumption entirely — learn the density with a neural network" |
| **This lecture** — Diffusion & Flow Matching | **"Don't learn the density. Learn to *undo* a process that destroys it."** |

This is a genuinely different philosophy from Parts 2 and 3. The VAE and GAN both try to build a
model of $p(x)$ directly (explicitly or implicitly). Diffusion models say: *forget $p(x)$*. Instead:
1. Define a process that *guarantees* turning any data distribution into a known Gaussian.
2. Train a network to *reverse* that process one step at a time.
3. Generate by starting from Gaussian noise and running the learned reverse process.

The elegance is that step 1 requires *no learning* — it's a fixed formula. All the learning
happens in step 2, and it reduces to predicting noise — arguably the simplest possible training
objective.

---

# PART A — Diffusion Models

*~0:00 – ~25:00*

---

## 1. The core idea: destroy then reconstruct

> *"Stable Diffusion, Midjourney, DALL-E — these all use diffusion models under the hood.
> The idea is simple: add noise until the data is gone, then learn to undo it."*
> [slide 1–2]
>
> *"Where you have already seen [diffusion]: Stable Diffusion, DALL-E 3, Midjourney — the dominant
> text-to-image systems today. Imagen Video, Sora — text-to-video generation. Audio (AudioLDM) and
> music (Stable Audio) — same recipe, different modality. Inverse problems in science — MRI
> reconstruction, protein design (RFdiffusion), molecular conformer generation."* [slide 2]

> 🎯 **The breadth of that list is worth being able to recite, not just the first two systems.**
> Diffusion is not "the image-generation trick" — it is the same forward-noise/learned-reverse
> recipe applied to video (Sora, Imagen Video), audio and music (AudioLDM, Stable Audio), and
> genuinely non-generative-media science problems (MRI reconstruction, protein design via
> RFdiffusion, molecular conformer generation) — anywhere "gradually corrupt, then learn to
> denoise" is a sensible way to frame the problem, regardless of modality.

Every generative model must answer: "how do I turn random numbers into realistic data?"
Diffusion models answer with a two-phase strategy:

**Phase 1 — Forward process (fixed, no learning):** Gradually add Gaussian noise to a data
sample $x_0$ over $T$ steps, producing a sequence $x_0, x_1, \ldots, x_T$ where:
- $x_0$ is clean data
- $x_T$ is pure Gaussian noise — indistinguishable from $\mathcal{N}(0, I)$
- Each step adds only a *tiny* amount of noise, so consecutive steps are similar

**Phase 2 — Reverse process (learned):** Train a neural network to undo each noise step. Given
$x_t$ (noisy), predict $x_{t-1}$ (less noisy). Starting from $x_T \sim \mathcal{N}(0, I)$, run
the reverse $T$ times to get back to clean data.

```interactive
type: simulator
title: Forward diffusion animation
concept: Gradual noise addition transforming data to Gaussian
control: Slider for time step t (0 to T)
observe: Image becoming progressively noisier until it's pure static
insight: Each step is only slightly noisier than the last — the process is smooth
fallback: Static figure showing x_0 (clean), x_{T/2} (half noisy), x_T (pure noise)
```

### Why this works: the information-theoretic argument

If you add noise *all at once* (one big step from $x_0$ to $x_T$), you've destroyed information
that no network can recover. But if you add noise *gradually*, each step only destroys a tiny
bit of information, and a neural network can learn to recover that tiny bit. The key insight:
**when the noise addition is small enough, the reverse step is well-approximated by a Gaussian
with a learnable mean.**

This is exactly what [Part 3's](unsupervised-learning-03.md) hierarchical VAE does — break a hard
one-step encoding into many easy small steps — except here the "encoding" is fixed (add noise)
rather than learned.

---

## 2. Diffusion models as hierarchical VAEs

> *"These can be looked upon as hierarchical VAEs."* [slide 3]

This is the formal bridge between Part 3 and this lecture. Recall:

| | Ordinary VAE (Part 3) | Diffusion Model |
|---|---|---|
| **Encoding** | $x \to z$ in **1 step** (learned encoder) | $x_0 \to x_1 \to \cdots \to x_T$ in **$T$ steps** (fixed noising) |
| **Dimension change** | $D \to M$ (compression to small latent) | $D \to D$ (same dimension — no compression) |
| **Encoder** | Learned neural network $q_\phi(z \mid x)$ | **Fixed** formula: add scaled Gaussian noise |
| **Decoder** | Learned neural network $p_\theta(x \mid z)$ | Learned neural network, **shared across all $T$ steps** |
| **Latent space** | Meaningful compressed representation | Noisy version of the input (not "compressed") |
| **Training objective** | ELBO = reconstruction $-$ KL | ELBO, but the reconstruction term becomes **denoising matching** |

The crucial differences:

1. **The encoder is fixed.** There's no learned encoder network — the "encoding" is just adding
   noise according to a predetermined schedule. This means no encoder collapse, no posterior
   collapse, and no need to balance encoder and decoder capacity.

2. **No compression.** The latent $x_t$ has the same dimensionality as the input $x_0$. It's not
   a compact representation — it's a *corrupted* version. This is a fundamentally different use of
   latent variables.

3. **The decoder is shared.** The same denoising network handles every noise level $t$, with $t$
   provided as an input so it knows how much denoising to do.

### Where people get confused

> 💡 **You might think:** "If there's no compression, how is this useful as a latent variable
> model?"
>
> **Actually:** Diffusion models don't care about compression. The "latent" $x_t$ isn't meant to
> be a compact code — it's a *waypoint* on a path from data to noise. The useful output is the
> *generated sample*, not the intermediate representations.

---

## 3. The forward diffusion process

> *"You take your input sample and repeatedly corrupt it until it becomes completely Gaussian."*
> [slide 6]

### Step-by-step: one noise addition

At each time step $t$ (from 1 to $T$), we transform $x_{t-1}$ into $x_t$ using:

$$x_t = \sqrt{\alpha_t} \, x_{t-1} + \sqrt{1 - \alpha_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $x_t$ | "the sample at time $t$" | The data after $t$ steps of noise addition. Starts as clean data ($x_0$), ends as noise ($x_T$). |
| $\alpha_t$ | "the signal retention rate" | A number in $(0, 1)$ chosen by the noise schedule. Higher $\alpha_t$ means less noise is added at step $t$. |
| $\epsilon$ | "the noise" | A random vector drawn from $\mathcal{N}(0, I)$ — standard Gaussian noise with the same dimensionality as the data. |
| $\sqrt{\alpha_t}$ | "signal scaling" | Keeps a fraction of the signal. Since $\alpha_t < 1$, this *shrinks* the original signal. |
| $\sqrt{1 - \alpha_t}$ | "noise scaling" | Scales the added noise so that the total variance stays at 1 (this is what "variance preserving" means). |

### The two scaling factors: why $\sqrt{\alpha_t}$ and $\sqrt{1 - \alpha_t}$?

The choice of these specific scalings is not arbitrary. It ensures **variance preservation**: if
$x_{t-1}$ has unit variance (i.e., $\text{Var}(x_{t-1}) = 1$), then $x_t$ also has unit
variance:

$$\text{Var}(x_t) = \alpha_t \cdot \text{Var}(x_{t-1}) + (1 - \alpha_t) \cdot \text{Var}(\epsilon) = \alpha_t \cdot 1 + (1 - \alpha_t) \cdot 1 = 1$$

This is why these are called **variance-preserving** diffusion models. The data doesn't
"blow up" or "shrink to zero" — it stays at unit variance throughout, just becoming
progressively more Gaussian.

### Closed form: skip to any time step

Since each step is a linear combination of a signal and independent noise, we can compose all $T$
steps into a single formula. Define:

$$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$$

Then:

$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\bar{\alpha}_t$ | "cumulative signal retention" | Product of all $\alpha_s$ from step 1 to $t$. Decreases monotonically toward 0. |

### 🧪 Worked example: forward diffusion on a scalar

Suppose $x_0 = 2.0$ (a single pixel), $T = 3$, with $\alpha_1 = 0.9$, $\alpha_2 = 0.8$,
$\alpha_3 = 0.7$.

**Step 1:** $\bar{\alpha}_1 = 0.9$
$$x_1 = \sqrt{0.9} \cdot 2.0 + \sqrt{1 - 0.9} \cdot \epsilon_1 = 1.897 + 0.316 \cdot \epsilon_1$$
If $\epsilon_1 = 0.5$: $x_1 = 1.897 + 0.158 = 2.055$

**Step 2:** $\bar{\alpha}_2 = 0.9 \times 0.8 = 0.72$
$$x_2 = \sqrt{0.72} \cdot 2.0 + \sqrt{1 - 0.72} \cdot \epsilon_2 = 1.697 + 0.529 \cdot \epsilon_2$$
If $\epsilon_2 = -0.3$: $x_2 = 1.697 - 0.159 = 1.538$

**Step 3:** $\bar{\alpha}_3 = 0.72 \times 0.7 = 0.504$
$$x_3 = \sqrt{0.504} \cdot 2.0 + \sqrt{1 - 0.504} \cdot \epsilon_3 = 1.420 + 0.704 \cdot \epsilon_3$$

Notice: the signal ($\sqrt{\bar{\alpha}_t} \cdot x_0$) shrinks from 2.0 → 1.897 → 1.697 → 1.420,
while the noise standard deviation grows from 0 → 0.316 → 0.529 → 0.704. The original signal is
being gradually drowned out.

### Why $x_T$ is Gaussian: the central limit argument

After enough steps, $\bar{\alpha}_T \approx 0$, so:

$$x_T \approx \sqrt{1 - \bar{\alpha}_T} \cdot \epsilon \approx \epsilon \sim \mathcal{N}(0, I)$$

> 📚 **Background the slide assumed — why this holds for *any* starting distribution $p(x_0)$, not
> just ones already close to Gaussian.** The deeper reason is the **central limit theorem applied
> to convolutions** (Prerequisite 3): each forward step convolves the current distribution with a
> Gaussian. Gaussian distributions are the *fixed point* of convolution — repeatedly convolving
> any distribution with Gaussians converges to a Gaussian, regardless of how oddly-shaped the
> starting distribution was. After $T$ steps (typically $T = 500$ to $1000$), the result is
> indistinguishable from $\mathcal{N}(0, I)$ regardless of $x_0$ — this is *why* the forward
> process's endpoint can be trusted to be a standard Gaussian for literally any dataset, not just
> verified empirically for one.

### The noise schedule: $\alpha_t$ vs $\bar{\alpha}_t$

The noise schedule defines how much noise is added at each step. Common schedules:

| Schedule | Formula | Behaviour |
|---|---|---|
| **Linear** | $\bar{\alpha}_t = 1 - t/T$ | Noise increases linearly. Simple but not optimal. |
| **Cosine** | $\bar{\alpha}_t = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$ | Slower at the start and end, faster in the middle. Better for images. |
| **Sigmoid** | Sigmoid-shaped $\bar{\alpha}$ curve | Similar idea to cosine. |

> 💡 **Key insight:** The schedule determines how information is destroyed. Too fast and the
> reverse process can't learn (consecutive steps are too different). Too slow and you waste
> compute on nearly-identical steps. The cosine schedule was a key innovation in the DDPM paper
> (Ho et al., 2020) that significantly improved sample quality.

---

## 4. The reverse process: learning to denoise

> *"Now comes the reverse process. This is where neural networks come in."* [slide 5]

The forward process is fixed — it's just adding noise. The reverse process is where all learning
happens. We need to learn to go from $x_t$ back to $x_{t-1}$.

### Why the reverse is Gaussian

A beautiful mathematical fact: if the forward steps are small enough (small $\alpha_t$ changes),
then the reverse conditional $p(x_{t-1} \mid x_t)$ is *approximately* Gaussian:

$$p(x_{t-1} \mid x_t) \approx \mathcal{N}\!\left(x_{t-1};\; \mu_\theta(x_t, t),\; \sigma_t^2 I\right)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\mu_\theta(x_t, t)$ | "the predicted mean" | A neural network that takes the noisy sample $x_t$ and the time step $t$, and predicts what the less-noisy version should look like. The subscript $\theta$ denotes learnable parameters. |
| $\sigma_t^2$ | "the reverse variance" | The uncertainty at each denoising step. Often fixed to $\beta_t = 1 - \alpha_t$ (the noise added at step $t$) or $(1 - \bar{\alpha}_{t-1})/(1 - \bar{\alpha}_t) \cdot \beta_t$. |

The proof that the reverse is Gaussian when steps are small comes from a Taylor expansion of the
joint distribution $p(x_{t-1}, x_t)$ — see Section 4 of Sohl-Dickstein et al. (2015), "Deep
Unsupervised Learning using Nonequilibrium Thermodynamics."

### The neural network: input, output, and what $t$ does

The denoising network $\epsilon_\theta(x_t, t)$ takes:

**Input:** the noisy sample $x_t$ (same dimension as the data — e.g., a $256 \times 256 \times 3$
image)

**Extra input:** the time step $t$ (typically encoded as a sinusoidal positional embedding, just
like in Transformers — this tells the network *how noisy* the input is)

**Output:** a prediction of the noise $\epsilon$ that was added to create $x_t$ from $x_0$

The network architecture is typically a **U-Net** — an encoder-decoder with skip connections that
preserves spatial detail. The time step $t$ is injected via additive embeddings at each layer.

### Why predict the noise instead of the clean image?

Two equivalent parameterisations exist:

| Parameterisation | Network predicts | Training target |
|---|---|---|
| **$\epsilon$-prediction** (standard) | The noise $\epsilon$ | $\|\epsilon_\theta(x_t, t) - \epsilon\|^2$ |
| **$x_0$-prediction** | The clean image $x_0$ | $\|x_{0,\theta}(x_t, t) - x_0\|^2$ |

They are equivalent because $x_0$ and $\epsilon$ are linearly related through the forward
process equation:

$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \, \epsilon}{\sqrt{\bar{\alpha}_t}}$$

So predicting one determines the other. **$\epsilon$-prediction is preferred** because:
1. The noise $\epsilon \sim \mathcal{N}(0, I)$ has a simpler distribution than $x_0$
2. The loss landscape is better conditioned
3. It connects naturally to the score function (see §6)

### Where people get confused

> 💡 **You might think:** "The network needs to learn a different function for every time step
> $t$."
>
> **Actually:** It's a *single* network that handles all $t$. The time step $t$ is provided as
> an input embedding, so the same weights $\theta$ produce the right output for any noise level.
> Think of it as one network that knows "at this noise level, the signal looks like *this*, so
> subtract *that*."

---

## 5. Training objective: the ELBO and denoising matching

> *"Reconstruction + KL (from VAE) + denoising matching (new, does all the learning)."*
> [slide 7]

### Deriving the objective from the VAE ELBO

Recall from [Part 3](unsupervised-learning-03.md) that the ELBO for a latent variable model is:

$$\log p(x_0) \geq \mathbb{E}_{q(z \mid x_0)}\!\left[\log p_\theta(x_0 \mid z) - \log \frac{q(z \mid x_0)}{p(z)}\right]$$

For diffusion models, the "latent" is the entire sequence $x_1, \ldots, x_T$, and the ELBO
decomposes into three named pieces, exactly as the deck's own handwritten-annotated slide shows
[slide 11]. After working through the algebra (which mirrors
[Part 2's](unsupervised-learning-02.md) GMM/EM decomposition), the ELBO becomes:

$$\log p(x_0) \geq \underbrace{\mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)]}_{\text{reconstruction term}} - \underbrace{D_{\mathrm{KL}}\big(q(x_T|x_0)\,\|\,p(x_T)\big)}_{\text{prior matching term}} - \sum_{t=2}^{T}\underbrace{\mathbb{E}_{q(x_t|x_0)}\big[D_{\mathrm{KL}}\big(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\big)\big]}_{\text{denoising matching term}}$$

| Term | What it means | Learnable? |
|---|---|---|
| Reconstruction term | How well $p_\theta(x_0\mid x_1)$ recovers the clean image from one step of noise — the diffusion analogue of a VAE decoder's reconstruction term, handled separately because $t=1$ has no "previous, noisier step" to compare against | ✅ Yes, but see below |
| Prior matching term | $D_{\mathrm{KL}}(q(x_T\mid x_0)\,\|\,p(x_T))$ — how far the forward process's final noisy state is from the fixed prior $p(x_T)=\mathcal{N}(0,I)$ | ❌ No — the forward process has no learned parameters, and by construction $q(x_T\mid x_0)\approx\mathcal{N}(0,I)$ already, so this term is close to zero and fixed |
| Denoising matching term | The sum, over every intermediate step $t=2,\ldots,T$, of how well the learned reverse step $p_\theta(x_{t-1}\mid x_t)$ matches the true reverse posterior $q(x_{t-1}\mid x_t,x_0)$ | ✅ Yes — **this is where essentially all of the learning happens** |

> ⚠️ **The reconstruction term is usually folded into the same simplified loss as the denoising
> matching term in practice**, which is why some derivations (and an earlier version of this
> section) treat $t=1$ as just another step of one unified per-step sum rather than naming it
> separately — but the deck's own slide is explicit that it *is* a separate, third named term,
> structurally identical to a VAE's reconstruction term (§2's comparison table), and it's worth
> keeping that name rather than silently merging it away.

Each denoising-matching term $\mathcal{L}_t := \mathbb{E}_{q(x_t|x_0)}\big[D_{\mathrm{KL}}\big(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\big)\big]$
is a KL divergence between two Gaussians (the true reverse posterior
$q(x_{t-1} \mid x_t, x_0)$ and the learned reverse $p_\theta(x_{t-1} \mid x_t)$). When you work
out this KL between two Gaussians, it simplifies beautifully:

$$\mathcal{L}_t = \mathbb{E}_{q}\!\left[\frac{1}{2\sigma_t^2}\left\|\mu_\theta(x_t, t) - \hat{\mu}_t(x_t, x_0)\right\|^2\right] + C_t$$

where $\hat{\mu}_t(x_t, x_0)$ is the *true* posterior mean (a function of $x_0$ and $x_t$ that
can be computed in closed form), and $C_t$ is a constant that doesn't depend on $\theta$.

| Term | What it means | Learnable? |
|---|---|---|
| $\mu_\theta(x_t, t)$ | The network's predicted mean | ✅ Yes — this is what we train |
| $\hat{\mu}_t(x_t, x_0)$ | The true posterior mean | ❌ No — computed from the forward process |
| $\sigma_t^2$ | The reverse variance | Usually fixed, not learned |

### The denoising matching reformulation

Since $\hat{\mu}_t(x_t, x_0)$ is a known function of $x_0$ and $x_t$, and $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$, we can reparameterise the target in terms of $\epsilon$:

$$\hat{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\epsilon\right)$$

The network $\mu_\theta$ predicts $x_t$'s mean — which is equivalent to predicting $\epsilon$. So
the training loss becomes:

$$\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\, x_0,\, \epsilon}\!\left[\left\|\epsilon - \epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\; t\right)\right\|^2\right]}$$

This is **denoising score matching** — you train a network to predict the noise that was added.
That's the entire training algorithm:

```
repeat:
    1. Sample x_0 from the dataset
    2. Sample t uniformly from {1, ..., T}
    3. Sample ε ~ N(0, I)
    4. Compute x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
    5. Compute loss = ||ε - ε_θ(x_t, t)||²
    6. Update θ by gradient descent
until converged
```

### Why this is remarkable

The training objective is **mean squared error between the added noise and the predicted
noise**. That's it. No adversarial training, no KL balancing, no careful reconstruction loss
weighting. Just: "here's a noisy image, what noise was added?" This simplicity is a major reason
diffusion models are so stable to train compared to GANs.

> 💡 **Interview golden nugget:** "The diffusion training objective is MSE on noise prediction.
> This connects directly to denoising score matching — the network learns the gradient of the
> log-density at each noise level, which tells it which direction to move to reach higher
> probability. That's the score function."

---

## 6. Noise prediction = score prediction

> *"Two equivalent formulations: predict $x_0$ or predict $\epsilon$. Standard: predict
> $\epsilon$. Equivalent to predicting the **score function** direction."* [slide 12]

### What is a score function?

> **Score function** — the gradient of the log-probability density: $s(x) = \nabla_x \log p(x)$.
>
> *In everyday words:* the score function points in the direction where probability increases
> fastest. It tells you "which way is uphill" on the probability landscape.
>
> *Concretely:* for a 1D Gaussian $\mathcal{N}(\mu, \sigma^2)$:
> $$s(x) = \nabla_x \log p(x) = \nabla_x \left[-\frac{(x - \mu)^2}{2\sigma^2}\right] = -\frac{x - \mu}{\sigma^2}$$
> This is a linear function that points toward $\mu$ — exactly what you'd expect: "move toward
> the mean."
>
> *Why it exists:* knowing the score tells you how to move samples toward higher probability.
> If you have a noisy sample and know the score, you can nudge it toward the data distribution.
> This is precisely what the reverse diffusion process does.

### 🧪 Worked example: score of a 2D Gaussian

Let $p(x) = \mathcal{N}\!\left(\begin{pmatrix} 1 \\ 2 \end{pmatrix}, \begin{pmatrix} 0.5 & 0 \\ 0 & 0.3 \end{pmatrix}\right)$

The score at $x = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$:

$$\nabla_x \log p(x) = -\Sigma^{-1}(x - \mu) = -\begin{pmatrix} 2 & 0 \\ 0 & 10/3 \end{pmatrix}\begin{pmatrix} 2 \\ -1 \end{pmatrix} = \begin{pmatrix} -4 \\ 10/3 \end{pmatrix}$$

The score says: "move left (decrease $x_1$) and move up (increase $x_2$)" — toward the mean
$(1, 2)$.

### The equivalence: $\epsilon$-prediction is score prediction

The deep connection: the noise prediction $\epsilon_\theta(x_t, t)$ is directly proportional to
the score of the noisy distribution $p(x_t)$:

$$\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar{\alpha}_t} \cdot \nabla_{x_t} \log p(x_t)$$

**Proof sketch:** From the forward process, $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 -
\bar{\alpha}_t}\,\epsilon$, so by Bayes' rule:

$$\nabla_{x_t} \log p(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}\,x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

The network learns $\epsilon_\theta \approx \epsilon$, so:

$$\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar{\alpha}_t} \cdot \nabla_{x_t} \log p(x_t)$$

Rearranging: **predicting the noise is the same as estimating the score up to a known
scaling factor.**

> ⚠️ **A step was silently taken above, and it's worth naming.** The derivation computes
> $\nabla_{x_t}\log p(x_t\mid x_0)$ — the score of the *conditional* distribution, given one
> specific clean image $x_0$ — but the final boxed result is stated in terms of
> $\nabla_{x_t}\log p(x_t)$, the score of the *marginal* distribution over all possible $x_0$'s.
> These are not the same quantity in general. The swap is valid because of an identity sometimes
> called **Tweedie's formula** (equivalently, the identity behind denoising score matching, Vincent
> 2011): $\nabla_{x_t}\log p(x_t) = \mathbb{E}_{x_0\mid x_t}\big[\nabla_{x_t}\log p(x_t\mid x_0)\big]$
> — the marginal score equals the *conditional* score **averaged over the posterior** $p(x_0\mid
> x_t)$. Training $\epsilon_\theta$ on the per-sample conditional target (the actual $\epsilon$
> used to generate each training $x_t$), averaged over the whole training set via ordinary
> expected-loss minimization, is exactly computing that expectation — which is why a network
> trained purely on individual conditional targets ends up estimating the marginal score, with no
> additional derivation needed.

| Symbol | Read it as | What it means |
|---|---|---|
| $\nabla_{x_t} \log p(x_t)$ | "the score at noise level $t$" | The direction of steepest increase in log-probability at noise level $t$ |
| $-\sqrt{1 - \bar{\alpha}_t}$ | "the scaling factor" | Converts between noise prediction and score. Deterministic given the schedule. |

### Why this matters

The score function interpretation reveals what the denoising network *actually* learns: not
just "what noise was added," but "how to move a sample toward higher probability at every noise
level." This insight led to **score-based generative models** (Song & Ermon, 2019), which
parameterise the score directly and use Langevin dynamics to generate samples — an alternative to
the iterative denoising approach.

> 💡 **The instructor's own summary of this section:** "The noise prediction and the score
> prediction are equivalent formulations of the same thing." This equivalence is the theoretical
> foundation that unifies diffusion models, score matching, and (as we'll see in Part B) flow
> matching.

---

## 7. Ancestral sampling: generating from noise

> *"Sample $x_T \sim \mathcal{N}(0, I)$, iteratively denoise with fresh noise. **Slow** — 50 to
> 1000 steps."* [slide 11]

### The algorithm

**Ancestral sampling** generates a new data sample by running the learned reverse process:

```
Algorithm: Ancestral Sampling
Input: trained noise predictor ε_θ, number of steps T, noise schedule {ᾱ_t}

1. Sample x_T ~ N(0, I)                    # Start from pure noise
2. For t = T, T-1, ..., 1:
     z ~ N(0, I) if t > 1, else z = 0      # Fresh random noise (none on last step)
     x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))·ε_θ(x_t, t)) + σ_t·z
3. Return x_0                               # Generated sample
```

| Symbol | Read it as | What it means |
|---|---|---|
| $\beta_t$ | "noise addition at step $t$" | $\beta_t = 1 - \alpha_t$ — the fraction of variance that's noise |
| $\sigma_t$ | "sampling noise scale" | Controls randomness. Often $\sigma_t = \sqrt{\beta_t}$ or $\sqrt{\beta_t \cdot \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}}$ |
| $z$ | "fresh noise" | New random sample each step. Without it ($z = 0$), sampling is deterministic. |

### Why the random noise $z$ matters

The noise term $z$ at each step injects *randomness* into the generation. Without it (setting
$z = 0$), the same $x_T$ always produces the same $x_0$ — deterministic mapping. With it, even
the same $x_T$ can produce different outputs, which improves sample diversity.

> 💡 **Key insight:** The stochastic sampling is also what makes diffusion models' samples
> higher quality than their ELBO would suggest. The ELBO is a lower bound on log-likelihood, and
> diffusion models actually have *worse* log-likelihoods than VAEs. But their *samples look
> better* because the stochastic sampling explores the distribution more thoroughly than a
> deterministic VAE decoder.

### The speed problem

Ancestral sampling requires running the neural network $T$ times (typically $T = 500$ to $1000$).
For a high-resolution image, this means 500–1000 forward passes through a large U-Net. This is
**the main practical limitation** of diffusion models.

Several acceleration methods exist:

| Method | Speedup | How |
|---|---|---|
| **DDIM** (Song et al., 2020) | 10–50× | Use a deterministic (non-Markovian) subset of the reverse process, allowing larger steps |
| **Distillation** | 4–8× | Train a smaller network to mimic multiple reverse steps at once |
| **Consistency models** (Song et al., 2023) | 1–2 steps | Learn to map any point on the trajectory directly to $x_0$ |

> 📚 **Background the slide names but doesn't unpack — what "non-Markovian" buys DDIM.** Ancestral
> sampling (§7's algorithm) is Markovian: computing $x_{t-1}$ uses only $x_t$, one step at a time,
> because that's how the forward process that defines $q(x_{t-1}\mid x_t,x_0)$ was built. DDIM
> defines a *different* reverse process that shares the same marginals $p(x_t)$ at every $t$ (so a
> network trained the ordinary way still works, unmodified) but is no longer required to only ever
> jump one step at a time — it can skip directly from $x_t$ to $x_{t-k}$ for $k>1$, because the
> alternative reverse process it defines is deterministic given $x_t$ and doesn't need the
> intermediate steps' randomness to stay consistent. That's the entire mechanism behind
> "use a subset of steps": the *skipped* steps' information was never load-bearing for this
> particular (non-Markovian) reverse process in the first place.

> ⚠️ **Note:** The instructor acknowledges the speed issue: diffusion models are "slow" compared
> to GANs (which generate in a single forward pass). This is a key motivation for flow matching
> (Part B), which typically needs fewer steps.

---

## 8. Score functions in depth: the full picture

> *"The score function tells you which direction to move to reach higher probability. The
> denoising network learns this at every noise level."* [slide 13]

### Why we need the score at *every* noise level

The score $\nabla_x \log p(x)$ depends on the distribution $p(x)$. At noise level $t$, the
distribution is $p(x_t)$ — which is different for each $t$. So we need a *family* of scores,
one for each noise level:

$$\nabla_{x_t} \log p(x_t), \quad t = 0, 1, \ldots, T$$

The neural network $\epsilon_\theta(x_t, t)$ learns this entire family simultaneously. The input
$t$ selects which "slice" of the score function to evaluate.

### The score at $t = 0$ vs $t = T$

| Noise level | Distribution $p(x_t)$ | Score $\nabla_x \log p(x_t)$ |
|---|---|---|
| $t = 0$ | The true data distribution $p(x_0) = p_{\text{data}}$ | Points toward modes of the data |
| $t = T$ | $\mathcal{N}(0, I)$ (nearly pure noise) | $= -x_T$ (a linear function — just points toward the origin) |
| Middle $t$ | Smoothed version of the data | Interpolates between the data score and the Gaussian score |

This interpolation is why the reverse process works: at high noise levels, the score is simple
(nearly linear), and the network can easily denoise a little. As noise decreases, the score
becomes more complex, but the network only needs to handle the *incremental* denoising from one
step to the next.

### Connection to Langevin dynamics

If you know the score, you can generate samples via **Langevin dynamics**:

$$x_{t+1} = x_t + \frac{\eta}{2} \nabla_x \log p(x_t) + \sqrt{\eta}\, z_t$$

This is a Markov chain that, given enough steps, converges to a sample from $p(x)$. Each step
moves toward higher probability (via the score) with some random exploration (via $z_t$). The
denoising reverse process can be seen as a *structured* version of Langevin dynamics, where the
score changes at each step (different noise levels).

---

## 9. Putting Part A together

```
                    THE DIFFUSION MODEL PIPELINE
                    
    TRAINING (one-time):
    ┌─────────────────────────────────────────────────────┐
    │ Dataset: {x_0}                                      │
    │                                                      │
    │ For each training step:                              │
    │   1. Sample x_0 from dataset                        │
    │   2. Sample t ~ Uniform({1,...,T})                   │
    │   3. Sample ε ~ N(0, I)                             │
    │   4. Compute x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε    │
    │   5. Loss = ‖ε - ε_θ(x_t, t)‖²                    │
    │   6. Gradient step on θ                             │
    └─────────────────────────────────────────────────────┘
                              │
                              │ θ converges
                              ▼
    GENERATION (per sample):
    ┌─────────────────────────────────────────────────────┐
    │ 1. Sample x_T ~ N(0, I)                            │
    │ 2. For t = T down to 1:                             │
    │      ε̂ = ε_θ(x_t, t)                              │
    │      x_{t-1} = denoise(x_t, ε̂, t)                 │
    │ 3. Return x_0                                       │
    │                                                      │
    │ Cost: T forward passes through ε_θ                  │
    └─────────────────────────────────────────────────────┘
```

**Key properties of diffusion models:**
- **Training:** Simple MSE on noise prediction. Stable, no adversarial dynamics.
- **Generation:** Iterative, requires $T$ network evaluations. Slow but high quality.
- **Connection to VAE:** Diffusion is a hierarchical VAE with a fixed encoder (add noise) and
  a shared decoder (the denoiser).
- **Connection to scores:** The noise predictor $\epsilon_\theta$ is proportional to the score
  function at each noise level.

👉 *See also:* this section derives diffusion's math; it doesn't cover how production systems make
it fast and controllable at scale — classifier-free guidance, samplers, latent diffusion, DiT, and
ControlNet are [GenAI & LLM Part 4](../GenAI%20%26%20LLM/genai-llm-04.md)'s territory.

> *"Diffusion = a Hierarchical VAE with T Gaussian denoising steps and shared decoder. Forward
> process is hand-coded noise injection. Reverse process is what you train — a single network,
> queried T times. The ELBO simplifies to a clean MSE on noise prediction. Training is
> regression-stable, no min-max games. The denoiser implicitly learns the score function
> $\nabla_x\log p(x)$ at every noise level — the gradient that points toward the data manifold.
> Sampling is iterative (slow). Trade-off: more steps — higher quality but more compute. **Modern
> role: state-of-the-art image, video, audio, and scientific generation — Stable Diffusion,
> DALL-E 3, Sora, AlphaFold 3.**"* [slide 14]

> 🎯 **That closing "modern role" line is the deck's own honest scorecard, worth quoting verbatim
> in an interview.** It names four systems spanning three different modalities and one science
> application — Stable Diffusion and DALL-E 3 (image), Sora (video), AlphaFold 3 (protein
> structure) — as concrete, current (2026) evidence that everything derived in §1–§8 (the ELBO
> decomposition, noise=score equivalence, ancestral sampling) is the actual mechanism inside
> production systems, not a toy algorithm that stopped mattering once something fancier arrived.

---

# PART B — Flow Matching

*~25:00 – 40:38*

---

## 10. The motivation: why not choose a better path?

> *"It is much simpler to actually just choose this trajectory."* [slide 16]
>
> *"Where you have already seen [flow matching]: Stable Diffusion 3 and Flux — flow matching is
> the training recipe under the hood. Movie Gen (Meta) — flow-based video generation. Protein
> structure (AlphaFold 3, RFdiffusion successors) — flow / diffusion for biological structures."*
> [slide 16]
>
> *"Plain-language intuition: think of every data point as a balloon you want to push from a
> Gaussian cloud to its target image. At each instant in time, you ask: which way and how fast
> should this balloon move? The neural network learns this 'wind field' (a velocity), and at
> inference you just integrate it forward. Why people switched: fewer sampling steps than
> diffusion (often <10), simpler regression objective, and a clean ODE view of generation."*
> [slide 16]

Diffusion models commit to one specific way of destroying data: adding Gaussian noise at each
step, following the schedule $\bar{\alpha}_t$. But this is an arbitrary choice. **What if we could
choose *any* smooth path from noise to data?**

This is the key idea of flow matching:
- Instead of a fixed noising process, define a **probability path** $\{p_t\}_{t \in [0,1]}$ that
  smoothly transitions from noise ($p_0 = \mathcal{N}(0, I)$) to data ($p_1 = p_{\text{data}}$)
- Learn a **velocity field** that transports samples along this path
- Generate by solving an ODE along the learned velocity field

> 💡 **One-sentence summary:** Flow matching generalises diffusion by letting you choose *any*
> smooth interpolation between noise and data, rather than being locked into Gaussian noise
> addition.

> 🎯 **The named production systems are worth being able to state cold, because this is exactly the
> kind of "is this still relevant in 2026" question an interviewer asks.** Flow matching is not a
> theoretical alternative to diffusion — it is the actual training recipe behind **Stable Diffusion
> 3** and **Flux** (image generation), **Movie Gen** (Meta's video generation system), and it sits
> alongside diffusion in **AlphaFold 3** and successors to **RFdiffusion** for biological structure
> generation. The deck's own reasons for the switch are concrete and practical, not aesthetic:
> **fewer sampling steps than diffusion** (often under 10, versus diffusion's hundreds), a
> **simpler regression objective** (§12's CFM loss), and a **clean ODE view of generation** (§13) —
> three separate, independently-verifiable engineering wins, not one vague "it's better" claim.

### Convention shift: $t \in [0, 1]$ instead of $t \in \{1, \ldots, T\}$

In flow matching, time is continuous: $t \in [0, 1]$.
- $t = 0$: pure noise ($x_0 \sim \mathcal{N}(0, I)$)
- $t = 1$: clean data ($x_1 \sim p_{\text{data}}$)

Note the **reversed convention** from Part A: here, $t=0$ is noise and $t=1$ is data. In
diffusion, $t=0$ was data and $t=T$ was noise. This is just notation — the physics is the same.

---

## 11. Probability paths and velocity fields

### Probability path

> **Probability path** — an ordered family of distributions $\{p_t\}_{t \in [0,1]}$ that
> smoothly interpolates from a simple prior $p_0 = \mathcal{N}(0, I)$ to the data distribution
> $p_1 = p_{\text{data}}$.
>
> *In everyday words:* imagine a movie where each frame is a probability distribution. The first
> frame is a blurry Gaussian blob. The last frame is the sharp data distribution. The probability
> path is the entire movie — every frame in between.
>
> *Concretely:* at each time $t$, $p_t$ is a well-defined distribution. At $t=0.5$, the data is
> "half-destroyed" — recognisable but noisy. At $t=0.9$, it's mostly clean with a hint of noise.

### Velocity field

> **Velocity field** — a function $v(x, t)$ that assigns a direction and speed to every point
> $x$ at every time $t$. Solving the ODE $\frac{dx}{dt} = v(x, t)$ moves points along the
> "flow" defined by the velocity field.
>
> *In everyday words:* imagine a fluid (like water) where the current at each point is
> $v(x, t)$. If you place a leaf on the surface, the velocity field tells you where it goes.
> Different starting positions lead to different trajectories, but they all follow the same
> current.
>
> *Concretely:* if $v(x, t) = (1 - t) \cdot (x_{\text{data}} - x)$, then every point moves
> toward $x_{\text{data}}$, slowing down as it approaches. This is a trivially simple velocity
> field — real ones are learned by a neural network.

### The transport equation

The probability path $\{p_t\}$ and the velocity field $v(x, t)$ are connected by the
**continuity equation** (also called the transport equation):

$$\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot \big(v(x, t) \, p_t(x)\big)$$

This says: "the way the distribution changes over time is determined by the velocity field
pushing probability mass around." It's the conservation law for probability — what flows out of
one region must flow into another.

> 💡 **The instructor's own simplification, stated directly:** In practice, we don't need to solve
> this PDE directly. Instead, we define the path implicitly through a formula for how individual
> samples move, and learn the velocity field from samples. That's the key simplification of flow
> matching.

---

## 12. Conditional flow matching (CFM) loss

> *"Remember learning the score function in diffusion models?"* [slide 20]
>
> *"It turns out the following objective is tractable and easier to compute (where $x_1$ is
> sampled from the training set): $\mathcal{L}_{CFM}(\theta) =
> \mathbb{E}_{t,p(x_1),p_t(x|x_1)}\|u_t^\theta(x_t) - u_t(x_t|x_1)\|^2$. In the simplest case, we
> define $p_t(x_t|x_1) := \mathcal{N}(x_t; x_1, \sigma_{1-t}^2 I)$. This results in a ground truth
> velocity as: $u_t(x|x_1) = \frac{\sigma'_{1-t}}{\sigma_{1-t}}(x_1-x_t)$."* [slide 20]

### The problem with learning the velocity directly

The velocity field $v(x, t)$ must satisfy the continuity equation for a given path $\{p_t\}$. But
computing $v(x, t)$ from the continuity equation requires knowing $p_t(x)$ — which is the thing
we're trying to learn. This circularity is what makes flow matching hard in general.

### The breakthrough: conditional flow matching

The key insight of Lipman et al. (2023): instead of learning the velocity of the *marginal*
path $p_t$, learn the velocity of a *conditional* path — one defined per data point $x_1$, which
is tractable, and which turns out to define the same marginal path in aggregate.

**The deck's own general formula, faithfully.** Define the conditional distribution of $x_t$
given a single data point $x_1$ as a Gaussian centred at $x_1$ with a *shrinking* standard
deviation $\sigma_{1-t}$ (small once $t$ is close to 1, i.e. $x_t$ is close to the real data
point; large near $t=0$, i.e. $x_t$ could be almost anywhere):

$$p_t(x_t\mid x_1) := \mathcal{N}(x_t;\, x_1,\, \sigma_{1-t}^2 I), \qquad u_t(x\mid x_1) = \frac{\sigma_{1-t}'}{\sigma_{1-t}}(x_1 - x_t)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $p_t(x_t\mid x_1)$ | "the conditional path density" | The distribution of the noisy point $x_t$ at time $t$, given the specific data point $x_1$ it's headed toward |
| $\sigma_{1-t}$ | "sigma sub one-minus-t" | The path's noise schedule: the standard deviation of the Gaussian around $x_1$, shrinking to (near) 0 as $t\to1$ and large as $t\to0$ — the free design choice that fixes which conditional path you get |
| $\sigma_{1-t}'$ | "sigma prime" | The derivative of $\sigma_{1-t}$ with respect to $t$ — how fast the noise schedule is shrinking at time $t$ |
| $u_t(x\mid x_1)$ | "the conditional velocity" | The instantaneous direction and speed of travel toward $x_1$ at point $x$, time $t$, for this schedule |

This is the deck's actual, general result — the "denoising strength" $\sigma_{1-t}'/\sigma_{1-t}$
scales how hard to push toward $x_1$, and it depends on *which* noise schedule $\sigma_{1-t}$ you
pick.

**The straight-line special case.** The deck's own accompanying diagram draws a specific,
concrete choice of path: $x_t = (1-t)x_0 + t x_1$, a literal straight line between one fixed noise
sample $x_0$ and one data point $x_1$, corresponding to letting $\sigma_{1-t}\to0$ as $t\to1$ in
the general formula above (the "optimal-transport" schedule in the flow-matching literature —
under this schedule $x_t$'s spread around $x_1$ shrinks to exactly zero at $t=1$, which is what
"the path ends exactly at the data point" means). For *this* specific path, the velocity is
easiest to get by differentiating the path equation directly rather than substituting into the
general $\sigma$-formula above — both routes agree, but direct differentiation is less error-prone
and is what the rest of this section does next.

> ⚠️ **Read what follows as a genuine, correct special case of the deck's own general formula
> above — not a different, unrelated result.** The deck's slide states the general
> Gaussian-conditional-path formula first, parameterized by an arbitrary noise schedule
> $\sigma_{1-t}$; the constant-velocity, straight-line result this section derives in detail next
> is what that general formula reduces to for the specific linear schedule its own diagram draws.
> Both are "the deck's formula" — one is just more general than the other, and this file only
> carries the second one through to a full worked derivation.

**Conditional path (the straight-line special case, worked in full from here on):** For a given
pair $(x_0, x_1)$ where $x_0 \sim \mathcal{N}(0, I)$ and $x_1 \sim p_{\text{data}}$, the simplest
possible path is a straight line:

$$x_t = (1 - t) \cdot x_0 + t \cdot x_1$$

This is a linear interpolation: at $t=0$ you're at $x_0$ (noise), at $t=1$ you're at $x_1$
(data), and at every intermediate $t$ you're on the straight line between them.

**Conditional velocity:** Since the path is a straight line, the velocity is constant:

$$u(x_t, t) = \frac{dx_t}{dt} = x_1 - x_0$$

The velocity is just "the difference between data and noise" — which is the same for every $t$.

### The training loss

The **conditional flow matching loss** trains a neural network $v_\theta(x_t, t)$ to predict this
velocity:

$$\boxed{\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\, x_0,\, x_1}\!\left[\left\|v_\theta(x_t, t) - (x_1 - x_0)\right\|^2\right]}$$

where $x_t = (1-t) \cdot x_0 + t \cdot x_1$.

| Symbol | Read it as | What it means |
|---|---|---|
| $v_\theta(x_t, t)$ | "the learned velocity" | Neural network predicting the direction and speed at each point along the path |
| $x_1 - x_0$ | "the target velocity" | The direction from noise to data — what the network should learn |
| $t$ | "time" | A scalar in $[0, 1]$ indicating position along the path |

### 🧪 Worked example: CFM on a simple 1D case

Suppose $x_0 = 0.5$ (a noise sample) and $x_1 = 2.0$ (a data sample).

The conditional path: $x_t = (1-t) \cdot 0.5 + t \cdot 2.0 = 0.5 + 1.5t$

At $t = 0$: $x_0 = 0.5$ (noise) ✓
At $t = 0.5$: $x_{0.5} = 1.25$ (halfway)
At $t = 1$: $x_1 = 2.0$ (data) ✓

The target velocity: $u = x_1 - x_0 = 2.0 - 0.5 = 1.5$ (constant for all $t$)

If the network predicts $v_\theta(1.25, 0.5) = 1.3$ at the midpoint, the loss is:
$(1.3 - 1.5)^2 = 0.04$

The network learns to predict velocity $1.5$ everywhere along this path. With many such pairs
$(x_0, x_1)$, it learns a velocity field that works for *all* data points simultaneously.

### Why straight lines are special

Straight-line paths are the simplest choice, and they have a crucial property: the
**intermediate distributions $p_t$ are Gaussian mixtures** (convolutions of the data distribution
with a scaled Gaussian). This makes them:
1. Easy to sample from (just interpolate)
2. Smooth (no sharp turns that would confuse the network)
3. Close to the marginal optimal transport path (the path that minimises total "work")

Other paths (curves, splines) are possible but straight lines are the default and work well in
practice.

### Where people get confused

> 💡 **You might think:** "CFM loss looks just like diffusion's noise-prediction loss — how are
> they different?"
>
> **Actually:** They *are* closely related, and that's the point. In diffusion, the network
> predicts $\epsilon$ (the noise), while in CFM it predicts $x_1 - x_0$ (the velocity). For the
> linear interpolation path, $\epsilon = x_0$ and velocity $= x_1 - x_0$, so predicting one
> determines the other. **The training objectives are algebraically similar because both are
> learning the same underlying thing: how to move from noise to data.** The difference is in the
> *generation* procedure (stochastic denoising vs ODE solving).

---

## 13. Inference: Euler's ODE solver

> *"Sample $x_0 \sim \mathcal{N}(0, I)$, then solve the ODE from $t=0$ to $t=1$."*

### The generation procedure

Once the velocity field $v_\theta(x, t)$ is trained, generating a sample means **solving the
initial value problem**:

$$\frac{dx}{dt} = v_\theta(x, t), \quad x(0) = x_0 \sim \mathcal{N}(0, I)$$

The simplest numerical solver is **Euler's method**:

$$x_{t + \Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t$$

### 🧪 Worked example: Euler's method in 1D

Suppose $v_\theta(x, t) = 2x$ (a simple learned velocity). Start at $x_0 = 1.0$, step size
$\Delta t = 0.25$.

| Step | $t$ | $x_t$ | $v_\theta(x_t, t)$ | $x_{t + \Delta t}$ |
|---|---|---|---|---|
| 0 | 0.00 | 1.000 | 2.000 | 1.500 |
| 1 | 0.25 | 1.500 | 3.000 | 2.250 |
| 2 | 0.50 | 2.250 | 4.500 | 3.375 |
| 3 | 0.75 | 3.375 | 6.750 | 5.063 |

Compare to the exact solution $x(t) = e^{2t}$: $x(1) = 7.389$. Euler gives $5.063$ — off by a
lot because the velocity grows fast. More steps (smaller $\Delta t$) would fix this.

### Algorithm: flow model sampling

```
Algorithm: Flow Matching Sampling (Euler's method)
Input: trained velocity v_θ, number of steps N, step size Δt = 1/N

1. Sample x_0 ~ N(0, I)                    # Start from noise
2. For i = 0, 1, ..., N-1:
     t = i / N                              # Current time
     x = x + v_θ(x, t) · Δt               # Euler step
3. Return x                                 # Generated sample
```

### The speed advantage over diffusion

| | Diffusion (ancestral) | Flow (Euler) |
|---|---|---|
| **Steps needed** | 500–1000 | 20–100 |
| **Cost per step** | 1 network forward pass | 1 network forward pass |
| **Total cost** | High | **10–50× lower** |
| **Generation** | Stochastic (random $z$ each step) | **Deterministic** (same $x_0$ → same $x_1$) |

The reason flow matching needs fewer steps: the straight-line path from noise to data is
"easier" to follow than the diffusion trajectory, which curves through high-dimensional space.
The velocity field along a straight line changes less, so bigger Euler steps are accurate.

> 💡 **Interview golden nugget:** "Flow matching is faster than diffusion because straight-line
> interpolation produces a smoother velocity field with lower curvature, allowing larger ODE
> solver steps without losing accuracy. Diffusion's noising schedule creates a more curved
> trajectory that needs finer discretisation."

### Determinism: a feature, not a bug

Since Euler's method is deterministic, the same noise sample $x_0$ always produces the same
output $x_1$. This is useful for:
- **Reproducibility:** same seed → same image
- **Latent space arithmetic:** interpolate between two $x_0$ values to interpolate outputs
- **Inversion:** given a real image, find the $x_0$ that generates it (run the ODE backwards)

---

## 14. The unifying connection: diffusion is a special case of flow

> 📚 **This section's framing is this document's own synthesis, not a verbatim on-slide quote.**
> No single frame in the raw capture states "diffusion models are a special case of flow models"
> as a standalone line — the closest the deck comes is §10's own motivating comparison (diffusion
> commits to one noise-addition path; flow matching lets you choose any path) and §12's general
> Gaussian-conditional-path formula, which the deck's own diffusion section (§3's forward process)
> is a specific instance of. The claim below is a true and standard connection in the flow-matching
> literature, made explicit here because it is exactly the kind of "how do these two things
> relate" question an interviewer asks — but it is being *derived* from the deck's separate pieces,
> not quoted from one slide that states it outright.

This is the deepest insight of the lecture. Let's make it precise.

### How diffusion fits into flow matching

Diffusion models define a specific probability path: $p_t = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\,x_0,\; (1 - \bar{\alpha}_t)\,I\right)$. This is a Gaussian centred at the scaled data point, with variance that grows with $t$.

The velocity field that transports samples along this path has a specific form (derived from the
SDE that defines the forward process). When you plug this specific path and velocity into the CFM
framework, you get back the noise-prediction loss.

In other words:

$$\text{Diffusion model} = \text{Flow model with Gaussian noise-addition path}$$

### Why flow matching is strictly more general

Flow matching lets you choose *any* path, not just the Gaussian noise-addition path. This opens
up:

| Choice of path | Resulting model |
|---|---|
| Gaussian noise addition (what diffusion does) | DDPM, score-based models |
| Straight-line interpolation | Rectified flow (the default CFM) |
| Optimal transport path | OT-Flow |
| Geodesic on the probability simplex | Riemannian flow matching |
| Any other smooth interpolation | Whatever works best for your data |

### Summary comparison table

| | VAE (Part 3) | GAN (Part 3) | Diffusion (Part A) | Flow (Part B) |
|---|---|---|---|---|
| **Density model** | Explicit (ELBO) | Implicit | Implicit (via ELBO lower bound) | Implicit |
| **Training** | Reconstruction + KL | Adversarial (minimax) | MSE on noise prediction | MSE on velocity prediction |
| **Generation** | Single forward pass | Single forward pass | Iterative denoising ($T$ steps) | Iterative ODE solve (fewer steps) |
| **Sample quality** | Blurry | Sharp but mode-collapsed | Sharp and diverse | Sharp and diverse |
| **Speed** | Fast | Fast | Slow | Medium |
| **Mode coverage** | Good | Poor | Excellent | Excellent |
| **Training stability** | Good | Poor (sensitive) | Excellent | Excellent |

---

## 15. Putting Part B together: the complete picture

```
                    THE FLOW MATCHING PIPELINE
                    
    TRAINING (one-time):
    ┌─────────────────────────────────────────────────────┐
    │ Dataset: {x_1}                                      │
    │                                                      │
    │ For each training step:                              │
    │   1. Sample x_1 from dataset                        │
    │   2. Sample x_0 ~ N(0, I)                           │
    │   3. Sample t ~ Uniform([0, 1])                     │
    │   4. Compute x_t = (1-t)·x_0 + t·x_1               │
    │   5. Loss = ‖v_θ(x_t, t) - (x_1 - x_0)‖²         │
    │   6. Gradient step on θ                             │
    └─────────────────────────────────────────────────────┘
                              │
                              │ θ converges
                              ▼
    GENERATION (per sample):
    ┌─────────────────────────────────────────────────────┐
    │ 1. Sample x_0 ~ N(0, I)                            │
    │ 2. Δt = 1/N  (e.g., N = 20 steps)                  │
    │ 3. For i = 0 to N-1:                                │
    │      t = i/N                                        │
    │      x = x + v_θ(x, t) · Δt                        │
    │ 4. Return x                                         │
    │                                                      │
    │ Cost: N forward passes through v_θ (N << T)         │
    └─────────────────────────────────────────────────────┘
```

### How the whole module connects

```
K-Means (Part 1)      →  hard clusters, assignment-based
        │
        ▼
GMM/EM (Part 2)       →  soft clusters, probabilistic, ELBO
        │
        ▼
VAE (Part 3 §B)       →  continuous latent, neural ELBO, reparameterization
        │
        ▼
GAN (Part 3 §C)       →  implicit model, adversarial training, no density
        │
        ▼
Diffusion (Part A)     →  hierarchical VAE, fixed forward, learned reverse, denoising = score
        │
        ▼
Flow Matching (Part B) →  generalise the path, ODE-based, fewer steps, diffusion is a special case
```

Every step generalises the previous one. The trajectory of the entire unsupervised learning
module is: from discrete to continuous, from parametric to neural, from explicit to implicit, and
from fixed paths to learnable ones.

> *"Flow matching frames generation as solving an Ordinary Differential Equation (ODE) — a
> deterministic flow from noise to data. Train a neural network $u_t^\theta$ to predict the
> velocity field (which way to push each point at each time). Conditional flow matching makes the
> target velocity tractable: simple linear paths from $x_0$ to data $x_1$ with a Gaussian along the
> way. Sampling: Euler or higher-order ODE solvers. Often <10 steps vs hundreds for diffusion.
> Tightly connected to diffusion — both can be cast as score-matching — but with a simpler
> regression target and fewer sampling steps. **Modern role: powering Stable Diffusion 3, Flux,
> and Movie Gen. The current favorite for fast, high-quality continuous generation.**"* [slide 22]

> 🎯 **Same discipline as §9's diffusion takeaways: name the current production systems, not just
> the mechanism.** Stable Diffusion 3 and Flux (image), Movie Gen (Meta's video system) are stated
> by the deck itself as flow matching's modern role — the direct, current-generation successors to
> the diffusion-era Stable Diffusion/DALL-E 3/Sora systems named in §9. **Interview signal:** being
> able to say "flow matching is what actually trains Stable Diffusion 3" is a stronger answer than
> reciting the CFM loss formula alone, because it demonstrates the derivation connects to something
> real and current rather than being a closed academic exercise.

---

## Interview prep — Amazon Applied Scientist

### Core questions

**Q1: Explain the forward diffusion process. Why is the data distribution eventually transformed into a Gaussian?**

<details>
<summary>Model answer</summary>

The forward process adds scaled Gaussian noise at each step: $x_t = \sqrt{\alpha_t}\,x_{t-1} +
\sqrt{1 - \alpha_t}\,\epsilon$. Each step convolves the current distribution with a Gaussian.
Since Gaussian distributions are the fixed point of convolution (by the central limit theorem),
repeated convolution of any distribution with Gaussians converges to a Gaussian. After $T$ steps
(typically 500–1000), $x_T \sim \mathcal{N}(0, I)$ regardless of the initial distribution
$p(x_0)$.
</details>

**Q2: What does the denoising network predict, and why does it need the time step $t$ as input?**

<details>
<summary>Model answer</summary>

The network $\epsilon_\theta(x_t, t)$ predicts the noise $\epsilon$ that was added to create $x_t$
from $x_0$. It needs $t$ because the appropriate denoising operation depends on the current noise
level: at high $t$ (lots of noise), the network should make large corrections; at low $t$ (little
noise), it should make small refinements. Without $t$, the network can't distinguish "very noisy"
from "barely noisy" inputs and can't calibrate its output. The time step is typically encoded as a
sinusoidal positional embedding (same technique as in Transformers) and injected additively at
each layer of the U-Net.
</details>

**Q3: How are diffusion models related to VAEs? What are the key differences?**

<details>
<summary>Model answer</summary>

Diffusion models are hierarchical VAEs with three key differences:
1. The encoder is *fixed* (add noise), not learned
2. The latent has the same dimensionality as the input (no compression)
3. The decoder (denoiser) is shared across all $T$ steps

The ELBO decomposition is the same: reconstruction term + KL terms. But in diffusion, the
reconstruction term becomes denoising score matching — MSE between predicted and actual noise —
which is simpler to optimise than the VAE's reconstruction loss. The fixed encoder eliminates
posterior collapse entirely.
</details>

**Q4: What is the score function, and how does it relate to the denoising network?**

<details>
<summary>Model answer</summary>

The score function is $s(x) = \nabla_x \log p(x)$ — the gradient of the log-density, pointing
toward higher probability. The denoising network $\epsilon_\theta(x_t, t)$ is proportional to the
score: $\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar{\alpha}_t} \cdot \nabla_{x_t} \log
p(x_t)$. So predicting the noise is equivalent to estimating the score up to a known scaling
factor. This means the network learns "which direction is uphill" on the probability landscape at
every noise level, and the reverse process follows these score vectors to move from noise toward
data.
</details>

**Q5: Explain ancestral sampling. What is the role of the random noise at each step?**

<details>
<summary>Model answer</summary>

Ancestral sampling starts from $x_T \sim \mathcal{N}(0, I)$ and iteratively applies the learned
reverse process: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 -
\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$ where $z \sim \mathcal{N}(0, I)$.
The random noise $z$ injects stochasticity, ensuring diversity: different $z$ values lead to
different samples even from the same starting point. Without $z$, sampling is deterministic (this
becomes the DDIM formulation). The stochasticity also improves sample quality by allowing the
process to explore the distribution, though it makes generation slow ($T$ network evaluations).
</details>

**Q6: What is a probability path, and how does flow matching use it?**

<details>
<summary>Model answer</summary>

A probability path $\{p_t\}_{t \in [0,1]}$ is a smooth family of distributions transitioning from
a simple prior $p_0 = \mathcal{N}(0, I)$ to the data distribution $p_1 = p_{\text{data}}$. Flow
matching defines this path implicitly through how individual samples move: for each pair $(x_0,
x_1)$, the sample at time $t$ is $x_t = (1-t)x_0 + tx_1$ (straight-line interpolation). The
velocity field $v(x, t)$ that moves samples along these paths is learned by minimising the CFM
loss: $\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$. At inference, starting from noise, the ODE
$\frac{dx}{dt} = v_\theta(x, t)$ is solved from $t=0$ to $t=1$ using Euler's method.
</details>

**Q7: Derive the conditional flow matching loss from first principles.**

<details>
<summary>Model answer</summary>

Given a data sample $x_1$ and noise sample $x_0 \sim \mathcal{N}(0, I)$, define the conditional
path: $x_t = (1-t)x_0 + tx_1$. The velocity along this path is constant: $u = \frac{dx_t}{dt} =
x_1 - x_0$. We train a network $v_\theta(x_t, t)$ to match this velocity. The loss is simply MSE:

$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - (x_1 -
x_0)\|^2\right]$

where $t \sim \text{Uniform}([0,1])$. This works because the straight-line path has constant
velocity, so the target doesn't depend on $t$ — only the input $x_t$ changes with $t$.
</details>

**Q8: How does a flow model generate a sample? Explain Euler's method.**

<details>
<summary>Model answer</summary>

Starting from $x_0 \sim \mathcal{N}(0, I)$, we solve the ODE $\frac{dx}{dt} = v_\theta(x, t)$
from $t=0$ to $t=1$ using Euler's method: $x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta
t$. With $N$ steps, $\Delta t = 1/N$. Each step evaluates the network once and nudges $x$ in the
direction of the learned velocity. The result is deterministic: same $x_0$ → same $x_1$. Flow
models need fewer steps (20–100) than diffusion (500–1000) because straight-line interpolation
produces a smoother velocity field with lower curvature, making larger steps accurate.
</details>

**Q9: Why is flow matching faster than diffusion?**

<details>
<summary>Model answer</summary>

Two reasons:
1. **Smoother trajectory:** Straight-line paths (flow) have less curvature than the curved
   trajectories created by diffusion's noise schedule. Lower curvature means the velocity field
   changes more slowly, so larger Euler steps can be used without accumulating error.
2. **Deterministic generation:** Flow matching doesn't need the per-step random noise $z$ that
   ancestral sampling uses. This simplifies the algorithm and can leverage adaptive ODE solvers
   that adjust step size based on local error estimates.

In practice, flow models achieve comparable sample quality to diffusion with 20–100 steps vs
500–1000, representing a 10–50× speedup.
</details>

**Q10: Compare VAE, GAN, diffusion, and flow matching on key properties.**

<details>
<summary>Model answer</summary>

| Property | VAE | GAN | Diffusion | Flow Matching |
|---|---|---|---|---|
| **Density** | Explicit (ELBO lower bound) | Implicit | Implicit | Implicit |
| **Training** | Reconstruction + KL | Minimax game | MSE on noise | MSE on velocity |
| **Generation speed** | 1 pass | 1 pass | T passes (500–1000) | N passes (20–100) |
| **Sample quality** | Blurry | Sharp but may miss modes | Sharp, diverse | Sharp, diverse |
| **Training stability** | Good | Poor (mode collapse, vanishing gradients) | Excellent | Excellent |
| **Mode coverage** | Good | Poor | Excellent | Excellent |
| **Connection** | Baseline | Density ratio estimation | Hierarchical VAE | Generalises diffusion |

The progression: VAEs are stable but blurry. GANs are sharp but unstable. Diffusion models get
the best of both (stable training + sharp samples) but are slow. Flow matching preserves those
advantages while reducing the speed gap.
</details>

### Depth probes

**"Why can't you just use a GAN discriminator to estimate the score function directly?"**

You can — and people do (this is called "discriminatorScore" or related approaches). But the
GAN's discriminator estimates a *divergence* between two distributions, while the score estimates
the *gradient* of one distribution. The score is a simpler object (a vector field, not a scalar),
and learning it via denoising is more stable than adversarial training.

**"What happens if the velocity field has high curvature — when would this fail?"**

If the learned velocity field changes rapidly (high curvature), Euler's method accumulates large
errors unless you use very small steps. This happens when: (1) the probability path has sharp
turns, (2) the data distribution has disconnected modes that require the path to "jump" between
them, or (3) the network is poorly trained and the velocity field is noisy. Higher-order ODE
solvers (Runge-Kutta) or adaptive step-size methods help.

**"If flow matching is strictly better, why do people still use diffusion?"**

Historical momentum, ecosystem maturity (more tools, pretrained models, and papers for
diffusion), and some cases where the stochastic sampling of diffusion is actually preferred
(multi-modal generation, diversity). In practice, the two are converging: many modern systems
use "rectified flow" which starts as a flow model but borrows diffusion's noise schedule for
training.

### Whiteboard-ready derivations

**Derivation 1: The CFM loss from the conditional velocity**

1. Define path: $x_t = (1-t)x_0 + tx_1$, so $\frac{dx_t}{dt} = x_1 - x_0$
2. Train $v_\theta$ to match: $\min_\theta \mathbb{E}_{t,x_0,x_1}\left[\|v_\theta(x_t, t) -
   (x_1 - x_0)\|^2\right]$
3. Sample $t \sim U([0,1])$, $x_0 \sim \mathcal{N}(0,I)$, $x_1 \sim p_{\text{data}}$
4. Compute $x_t$, evaluate loss, backprop

**Derivation 2: Noise prediction equals score prediction**

1. Forward process: $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$
2. Rearrange: $x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}$
3. Score: $\nabla_{x_t}\log p(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}
   = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$
4. Therefore: $\epsilon = -\sqrt{1-\bar{\alpha}_t}\nabla_{x_t}\log p(x_t|x_0)$
5. Network learns $\epsilon_\theta \approx \epsilon$, so $\epsilon_\theta \propto \nabla_{x_t}\log
   p(x_t)$

**Derivation 3: Euler's method from the ODE**

1. ODE: $\frac{dx}{dt} = v_\theta(x,t)$, $x(0) = x_0$
2. Taylor expand: $x(t + \Delta t) = x(t) + \frac{dx}{dt}\Delta t + O(\Delta t^2)$
3. First-order approximation: $x(t + \Delta t) \approx x(t) + v_\theta(x(t), t) \cdot \Delta t$
4. Discretise: $t_i = i/N$, $\Delta t = 1/N$, iterate $i = 0, \ldots, N-1$

### Applied scenario

**Amazon delivery time estimation with generative models:**

You need to generate realistic synthetic delivery time data for stress-testing a logistics model
when real data is scarce or privacy-restricted.

- **Framing:** Model the joint distribution of (weather, traffic, distance, time-of-day) → delivery
  time
- **Data:** Historical delivery records with features and actual delivery times
- **Model:** Train a conditional flow matching model that learns the conditional distribution
  $p(\text{delivery time} \mid \text{features})$
- **Metric:** Kolmogorov-Smirnov test between generated and real delivery time distributions;
  downstream model accuracy when trained on augmented data
- **Failure modes:** Mode collapse (GANs would miss rare but critical long-delivery scenarios);
  blurriness (VAEs would average out extreme cases). Diffusion/flow models handle both.
- **What you'd ship:** A flow matching model (faster inference than diffusion) generating synthetic
  delivery times, with quality validation against held-out real data

### Leadership Principles

- **Dive Deep:** Flow matching required understanding the mathematical connection between
  diffusion, ODEs, and optimal transport — not just using a library, but understanding *why* the
  loss works.
- **Learn and Be Curious:** The rapid evolution from VAE → GAN → Diffusion → Flow shows how
  quickly the field moves — staying current requires continuous learning.

---

## Glossary

| Term | Definition |
|---|---|
| **Ancestral sampling** | Iterative procedure that generates a data sample by running the reverse diffusion process from noise, adding fresh random noise at each step |
| **Conditional flow matching (CFM)** | Training method that learns a velocity field by matching the velocity of straight-line paths between noise and data samples |
| **Convolution of distributions** | When two independent random variables are added, their sum's distribution is the convolution of the individual distributions |
| **Denoising score matching** | Training a network to predict the noise added to a sample, which is equivalent to learning the score function |
| **Diffusion model** | Generative model that learns to reverse a fixed noising process, transforming Gaussian noise back into data |
| **Euler's method** | Simplest ODE solver: $x_{t+\Delta t} = x_t + v(x_t, t) \cdot \Delta t$ |
| **Flow matching** | Generalised framework for generative modelling that learns a velocity field along a probability path from noise to data |
| **Noise schedule** | The sequence $\{\alpha_t\}$ or $\{\bar{\alpha}_t\}$ controlling how much noise is added at each forward diffusion step |
| **Ordinary differential equation (ODE)** | Equation relating a quantity to its rate of change: $\frac{dx}{dt} = v(x, t)$ |
| **Probability path** | A smooth family of distributions $\{p_t\}_{t \in [0,1]}$ transitioning from noise to data |
| **Reverse process** | The learned denoising steps that transform a noisy sample back toward clean data |
| **Score function** | The gradient of the log-probability density: $s(x) = \nabla_x \log p(x)$, pointing toward higher probability |
| **Variance-preserving** | A diffusion process where the total variance of $x_t$ is kept constant (typically at 1) across all time steps |
| **Velocity field** | A function $v(x, t)$ assigning a direction and speed to every point in space at every time, whose integral curves transport noise to data |

---

## Check yourself

1. Why does repeatedly adding Gaussian noise to *any* distribution eventually produce a Gaussian?
   What mathematical property makes Gaussians special in this regard?

2. In the forward process $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon$, why
   are the scaling factors $\sqrt{\alpha_t}$ and $\sqrt{1-\alpha_t}$ chosen specifically (rather
   than, say, $\alpha_t$ and $1-\alpha_t$)?

3. Derive the closed-form expression for $x_t$ in terms of $x_0$ and $\epsilon$, starting from
   the iterative definition. What is $\bar{\alpha}_t$?

4. The denoising network takes $(x_t, t)$ as input. Explain precisely why $t$ is necessary —
   what goes wrong if you omit it?

5. Write out the complete training loop for a diffusion model (pseudocode). How many random
   samples are drawn per training step, and from which distributions?

6. What is the ELBO for a diffusion model? Identify the three types of terms (Term 0, the
   per-step terms, and the denoising matching term) and explain what each one does.

7. Explain why predicting the noise $\epsilon$ is equivalent to predicting the clean image $x_0$.
   Write the conversion formula.

8. Compute the score function $\nabla_x \log p(x)$ for $p(x) = \mathcal{N}(3, 0.5)$ at $x = 4$.

9. In ancestral sampling, what is the role of the random variable $z$ at each step? What
   happens to generation quality and diversity if you set $z = 0$?

10. Define a probability path. Give the formula for the straight-line path between $x_0$ and
    $x_1$, and compute the velocity along this path.

11. Write the conditional flow matching loss. Explain each symbol and why the target velocity
    $(x_1 - x_0)$ doesn't depend on $t$.

12. Apply Euler's method with 3 steps ($N = 3$) to solve $\frac{dx}{dt} = v(x, t) = 1 - x$,
    starting at $x(0) = 0$. Compare your answer to the exact solution $x(t) = 1 - e^{-t}$ at
    $t = 1$.

13. In one sentence, state the precise relationship between diffusion models and flow matching
    models. Why is this relationship true?

---

## Going deeper

1. **Ho, Jain, & Abbeel (2020) — "Denoising Diffusion Probabilistic Models" (DDPM)**
   The paper that made diffusion models practical for high-quality image generation. Introduces
   the simplified training objective and the cosine noise schedule. *Solid* — foundational reading.

2. **Song, Sohl-Dickstein, et al. (2021) — "Score-Based Generative Modeling through Stochastic
   Differential Equations"**
   Unifies diffusion models and score matching in a continuous-time SDE framework. Provides the
   theoretical backbone for understanding why noise prediction = score prediction. *Hard* — heavy
   on stochastic calculus, but deeply rewarding.

3. **Lipman, Chen, et al. (2023) — "Flow Matching for Generative Modeling"**
   The original flow matching paper. Introduces the CFM loss and demonstrates it on toy problems
   and images. *Solid* — accessible and clearly written.

4. **Liu, Gong, & Liu (2023) — "Flow Straight and Fast: Learning to Generate and Transfer Data
   with Rectified Flow" (ICLR 2023)**
   The follow-up that makes straight-line CFM paths practical at scale: iteratively "reflows" a
   learned coupling between noise and data so the resulting paths become straighter still, cutting
   the number of sampling steps needed. *Solid* — the key practical paper for straight-line paths.
   *(An earlier version of this entry cited a fabricated paper — "Lipman, Biloos, et al. (2023),
   Generative Modelling with Inverse Heat Dissipation" — under this description; that title
   belongs to a real but unrelated paper by Rissanen, Heinonen & Solin (ICLR 2023) on
   heat-dissipation-based generative models, not flow matching. Corrected here after a citation
   audit; verify this entry yourself before citing it in an interview.)*

5. **Song & Ermon (2019) — "Generative Modeling by Estimating Gradients of the Data
   Distribution"**
   The paper that established the score-matching perspective on generative models. Shows Langevin
   dynamics sampling. *Solid* — connects to the broader score-matching literature.

6. **Lillian Weng's blog post "What are Diffusion Models?"** (lilianweng.github.io)
   Excellent visual explainer covering DDPM, score matching, and the SDE perspective with clear
   diagrams. *Intro* — great for building intuition before diving into papers.

> ⚠️ **All citations above should be verified for exact titles and dates.** The paper titles and
> author lists are reconstructed from memory and may contain minor inaccuracies.

---

*End of Part 4. This completes the Unsupervised Learning module.*
