---
title: "Beyond Text: Multimodal Models and Diffusion"
topic: genai-llm
lecture: 04
source: "slides_deduped/Lecture_17 - Module 5 Generative AI and LLMs Part 4"
slides: 34
---

# Beyond Text: Multimodal Models and Diffusion

*How the same transformer learns to see, hear, and speak — and a second family that generates by
denoising.*

> ⚠️ **Capture note.** These notes were rebuilt against the verified slide extraction at
> `slides_deduped/Lecture_17 - Module 5 Generative AI and LLMs Part 4/` (**34 deduped slides**, up
> from the earlier PDF-screenshot draft's ~29 distinct slides). Every one of the 34 slides was read
> directly, and content coverage is essentially complete — this is the cleanest of the four files in
> this module by that measure. Confirmed directly from the title slide: this lecture's real topic is
> **multimodality and diffusion** (CLIP, modular vs. unified architectures, speech, DDPM math,
> guidance, samplers, distillation, latent diffusion, DiT, ControlNet, editing, evaluation) — not
> "deployment: quantization/distillation/KV-cache/safety/evaluation," which was this module's
> README's stale description of this file before this review (see `QUALITY_REVIEW.md` and the
> corrected module README); those deployment topics are Lecture 16's content
> (`genai-llm-03.md` §21–§28), not this file's.

---

## What you'll understand after reading this

By the end of this document you will be able to:

1. **Explain the one idea that makes multimodality possible** — and why a transformer does not care
   whether a token is a word, an image patch, or a millisecond of sound.
2. **Explain CLIP's contrastive objective** in full, including why it enables zero-shot
   classification with no labelled training.
3. **Compare modular and unified multimodal architectures** (LLaVA vs GPT-4o) and choose between
   them.
4. **Explain why native speech-to-speech beats a cascade** — in latency and in what survives the
   text bottleneck.
5. **Explain diffusion from scratch**: the fixed forward noising process, the learned reverse
   process, and why the loss is a simple regression.
6. **Write down the closed-form noising equation** and explain why it makes training cheap.
7. **Distinguish ε-prediction, $x_0$-prediction, and v-prediction**, and say when each is
   ill-conditioned.
8. **Explain noise schedules**, Min-SNR weighting, and the zero-terminal-SNR bug that stops models
   generating truly dark images.
9. **Explain classifier-free guidance**, compute the guided prediction, and describe the
   faithfulness-versus-diversity trade-off.
10. **Explain how ~1000 denoising steps became ~20, and then 1–4** — samplers versus distillation.
11. **Explain latent diffusion** and calculate the 48× compute saving that made Stable Diffusion
    possible.
12. **Contrast the U-Net and DiT backbones** and explain why the field converged on transformers
    again.
13. **Explain ControlNet's zero-convolution trick** and why it cannot damage the base model.
14. **Explain img2img, inpainting, and instruction editing**, and the one dial that trades
    faithfulness against edit size.
15. **Explain language diffusion** and argue honestly about where it beats autoregressive models
    and where it does not.
16. **Evaluate an image generator properly** — FID, CLIP-score, precision/recall, GenEval, and human
    preference — and explain why no single number suffices.

---

## Before we start: what you need to know

Diffusion needs more probability background than the previous three lectures. Everything is taught
here.

### Prerequisite 1 — Gaussian (normal) distribution

> **Gaussian / normal distribution** — the bell curve. The most common way randomness shows up in
> nature and in mathematics.
>
> *In everyday words:* most values cluster near the middle; extreme values are rare and get rarer
> fast as you move away.
>
> **Notation:** $\mathcal{N}(\mu, \sigma^2)$ means "a normal distribution with **mean** $\mu$ and
> **variance** $\sigma^2$."

| Symbol | Read it as | What it means |
|---|---|---|
| $\mu$ | "mu" | The **mean** — the centre of the bell. |
| $\sigma$ | "sigma" | The **standard deviation** — how wide the bell is. |
| $\sigma^2$ | "sigma squared" | The **variance**. Just the standard deviation squared. |
| $\mathcal{N}(0, 1)$ | "standard normal" | Centred at 0, standard deviation 1. **The default noise everywhere below.** |
| $I$ | "the identity matrix" | In $\mathcal{N}(\mu, \sigma^2 I)$, this means *"each dimension gets independent noise of the same size"* — no correlation between pixels. |

*Concretely:* drawing from $\mathcal{N}(0,1)$ gives numbers like $0.34, -1.21, 0.08, 1.77$ — usually
within ±2, occasionally further.

> 📚 **The one property that makes diffusion work: Gaussians compose.**
> If you add Gaussian noise, and then add Gaussian noise again, the result is **still Gaussian**,
> and you can compute exactly which one in a single step. Formally, adding $\mathcal{N}(0, a)$ then
> $\mathcal{N}(0, b)$ gives $\mathcal{N}(0, a + b)$ — you don't have to simulate both steps.
>
> **This is why you can jump straight from a clean image to step 700 of noising without simulating
> steps 1–699**, and it is the single fact that makes diffusion training affordable. It appears as
> "the shortcut" in section 8.

### Prerequisite 2 — Markov chain

> **Markov chain** — a sequence of steps where **the next state depends only on the current state**,
> not on the whole history.
>
> *In everyday words:* a board game where your next move depends only on where your piece is now,
> not how it got there.
>
> *Concretely:* in diffusion, image $x_5$ is produced from $x_4$ alone. Nothing about $x_0, x_1, x_2,
> x_3$ enters the computation.
>
> *Why it matters:* it makes the process analysable one step at a time. Both the noising and
> denoising processes in this lecture are Markov chains.

### Prerequisite 3 — MSE and regression

> **MSE (Mean Squared Error)** — the average of the squared differences between prediction and
> truth.
>
> $$\mathcal{L} = \mathbb{E}\big[\|\text{prediction} - \text{truth}\|^2\big]$$
>
> *Concretely:* predicted $[1.0, 2.0]$, truth $[1.5, 1.0]$. Differences: $[-0.5, 1.0]$. Squared:
> $[0.25, 1.0]$. Sum: $1.25$.
>
> *Why it matters here:* despite all the probability theory, **the actual training loss of a
> diffusion model is plain MSE** — predict a number, compare against the truth, square the
> difference. That is the punchline of section 7.

> 📚 **Notation: $\|\cdot\|^2$** means "square every component and add them up" — the squared length
> of the difference vector.

### Prerequisite 4 — Gradient of a log-density (the "score")

You'll meet $\nabla_x \log p(x)$ twice. Here's what it means without the machinery.

> **$\nabla_x \log p(x)$ — the score.** For any point in space, it's an arrow pointing in the
> direction that would make that point **more probable** under the data distribution.
>
> *In everyday words:* a compass needle pointing "toward more realistic". Standing on a blurry mess
> of pixels, the score tells you which way to nudge every pixel to make the picture look more like a
> real photograph.
>
> | Symbol | Read it as | What it means |
> |---|---|---|
> | $\nabla_x$ | "nabla sub x" / "gradient with respect to x" | The direction of steepest increase, as you vary $x$. |
> | $p(x)$ | "p of x" | How probable the data distribution considers $x$. |
> | $\log$ | "log" | Taken for numerical convenience; it doesn't change which direction is uphill. |
>
> *Why it matters:* it turns out that **predicting the noise in an image is mathematically
> equivalent to computing this score**. That equivalence links two research traditions
> (denoising diffusion and score matching) that were developed separately and turned out to be the
> same thing.

### Prerequisite 5 — ODE (in one paragraph)

> **ODE (Ordinary Differential Equation)** — an equation describing how something changes
> continuously, which you solve by stepping along it.
>
> *In everyday words:* if you know your velocity at every moment, an ODE solver figures out where
> you end up.
>
> *Why it appears:* the reverse diffusion process can be written as an ODE — a smooth trajectory
> from noise to image. Once you see it that way, you can apply **better numerical solvers** that
> take bigger, smarter steps. That is exactly what DDIM and DPM++ are (section 12), and it's why
> the step count fell from ~1000 to ~20 with **no retraining**.

### Prerequisite 6 — Recap: what you already have

From the previous lectures, you have everything else you need:

| From | Concept | Where it reappears |
|---|---|---|
| L1 | Attention, $\mathrm{softmax}(QK^\top/\sqrt{d})V$ | Cross-attention is how text steers image generation |
| L1 | Cross-attention | Section 17 — the bridge between prompt and denoiser |
| L1 | Transformer blocks, patches | Section 15 — DiT is literally a transformer on image patches |
| L1 | Residual connections | Sections 14, 16 — U-Net skips and DiT blocks |
| L2 | Distillation (teacher → student) | Section 13 — but for *steps*, not size |
| L2 | Cross-entropy loss | Section 4 — CLIP's InfoNCE is cross-entropy |
| L3 | Cosine similarity | Section 4 — CLIP's similarity matrix |
| L3 | Autoregressive generation and its sequential constraint | Section 20 — what language diffusion escapes |
| L3 | LLM-as-a-judge, benchmarks | Section 23 |

---

## The big picture

The first three lectures were about text. This one asks two questions that break out of it.

**Question 1: how does a model come to *see* and *hear*?** The answer is almost disappointingly
simple. A transformer does not care what a token *means* — it takes a sequence of vectors and mixes
them with attention. So if you can turn *anything* into a sequence of vectors, a transformer can
model it. Cut an image into **patches**, audio into **frames**, text into **sub-words**, and they
all become vectors flowing into the same attention. Multimodality is not a new architecture; it is
a new **tokenizer**.

**Question 2: how do you *generate* an image?** Here the answer is genuinely different. Text is
generated autoregressively — one token at a time, left to right. Images are not. Instead, a second
family of models learns to **destroy an image with noise and then reverse the destruction**. To
generate, you start from pure static and denoise your way to a picture that never existed. That is
**diffusion**, and it is the engine behind every modern image and video generator.

The lecture then follows diffusion's engineering story — how ~1000 denoising steps became 20 and
then 1, how diffusing in a compressed latent instead of pixels made it run on a consumer GPU, how
the convolutional backbone was replaced by a transformer and inherited LLM-style scaling, and how
text, structure, and style are wired in to steer it. Finally, in a neat closing of the loop,
**diffusion comes back for text** — generating whole sequences in parallel instead of left to right.

---

# PART 0 · The core idea

## 1. If you can turn it into tokens, a transformer can model it

*(Slide 2)*

> **Text, pixels, and sound all become sequences of vectors, then attention mixes them in one
> model.**

### The trick

> **A transformer does not care what a token *means*. Cut an image into patches, audio into frames,
> text into sub-words, and they all become vectors.**

This is the whole foundation of multimodality, and it's worth stating precisely why it's true.

Recall from lecture 1 what a transformer actually does: it takes a sequence of vectors, computes
$\mathrm{softmax}(QK^\top/\sqrt{d})V$ over them, passes each through an FFN, and repeats. **Nothing
in that computation refers to language.** The dot product doesn't know it's comparing words. The
architecture is a general-purpose sequence mixer that happened to be invented for translation.

So the only question is: **can you turn your data into a sequence of vectors?**

| Modality | How it becomes tokens | The slide's term |
|---|---|---|
| **Text** | Split into word-pieces (BPE, from lecture 1) | *sub-word tokens* |
| **Image** | Cut into a grid of small squares; flatten each and project to a vector | *patch embeddings* |
| **Audio** | Convert to a spectrogram; slice into short time windows | *spectrogram frames* |

> 📚 **Background the slide assumed — image patches.**
> Take a 224×224 pixel image and cut it into a grid of 16×16-pixel squares. That gives
> $(224/16)^2 = 14^2 = 196$ patches. Each patch is $16 \times 16 \times 3 = 768$ raw numbers
> (RGB), which a single learned linear layer projects into an embedding vector.
> **Result: an image is now a sequence of 196 vectors** — structurally identical to a 196-token
> sentence. This is the **Vision Transformer (ViT)** idea, and it is the entire bridge from pixels
> to transformers.

> 📚 **Background the slide assumed — spectrogram frames.**
> A **spectrogram** turns a sound wave into a picture: time on one axis, frequency on the other,
> loudness as brightness. It shows *which pitches are present at each moment*. Slice it into short
> time windows — typically 10–25 milliseconds — and each slice becomes a vector.
> **Result: audio is now a sequence of vectors too.** (And note: since a spectrogram is literally an
> image, you can also just patch it like one.)

### One mechanism

> **Self-attention then mixes every token with every other, so the model can ground a word in a
> region of an image, or a sound in a moment.**

The slide's diagram:

```
   text   ── sub-word tokens ───►  [The] [cat] [sat] ──┐
                                                       │
   image  ── patch embeddings ──►  [▪] [▪] [▪] [▪] ────┼──►  ONE TRANSFORMER
                                                       │      attention over
   audio  ── spectrogram frames ►  [♪] [♪] [♪] ────────┘      ALL tokens
```

> 💡 **The word "ground" in that bullet is doing a lot of work, so unpack it.** Because
> self-attention mixes *every* token with *every other* token — and image patches and text tokens
> are now in the same sequence — the word token `cat` can attend directly to the image patches
> containing the cat. **Nothing special was built to make that happen.** It falls out of attention
> operating over a sequence that happens to contain both kinds of token. The model learns which
> patches a word refers to the same way it learns which earlier word a pronoun refers to.

### The two questions, and why now

> **Two questions for this block: How do we *fuse* modalities (multimodal models), and how do we
> *generate* them, especially images and video (diffusion)?**
>
> **Why now: The same recipe that scaled text — more data and compute on one architecture — is what
> made GPT-4o, Gemini, and Sora possible.**

That second point closes the loop with lecture 1. Multimodal models didn't need a breakthrough. They
needed the **same** architecture, the **same** scaling behaviour, applied to data that had been
tokenized differently.

### Where people get confused

**You might think** multimodal models have separate "vision modules" and "language modules".
**Actually** in the unified design (section 5) there is **one** transformer and one set of weights.
The modalities differ only in how they were converted to tokens before entering.

**You might think** patching throws away spatial structure. **Actually** it's recovered exactly the
way word order is: **positional embeddings**. Each patch gets a vector encoding its grid position,
just as each word gets one encoding its place in the sentence (lecture 1, section 9).

---

# PART 1 · Teaching a model to see and hear

## 2. CLIP: aligning images and text in one space

*(Slides 4 and 5)*

> **Train two encoders so a picture and its caption land on the same point.**

### The goal

> **Put images and text in a shared embedding space, so "a photo of a dog" sits near actual dog
> photos.**

From lecture 3 you know embeddings: text becomes vectors where closeness means similar meaning.
CLIP's move is to build a space where **images and text share the same coordinate system** — where
you can compute the cosine similarity between a *photograph* and a *sentence* and get a meaningful
answer.

### Contrastive training

> **Take a batch of image-text pairs. Pull each true pair together; push all the mismatched pairs
> apart.**

> **Contrastive learning** — training by comparison rather than by labels. You show the model which
> pairs *belong together* and which don't, and it arranges its embedding space accordingly.
>
> *In everyday words:* teaching by "these two go together, those two don't", repeated millions of
> times, until the arrangement encodes meaning.
>
> *Why it exists:* it needs **no human labels at all**. Image-caption pairs already exist by the
> hundreds of millions on the web — every alt-text, every caption. **The supervision was lying
> around for free**, and that is why CLIP could train on a dataset no labelling budget could have
> produced.

### The similarity matrix

> **This is a similarity matrix: maximise the diagonal, minimise everything off it.**

$$\max\ \cos(I_i, T_i) \quad \text{vs} \quad \cos(I_i, T_{j \neq i})$$

| Symbol | Read it as | What it means |
|---|---|---|
| $I_i$ | "I sub i" | The embedding of the $i$-th **image** in the batch. |
| $T_i$ | "T sub i" | The embedding of the $i$-th **text** — the caption that actually belongs to $I_i$. |
| $T_{j \neq i}$ | "T sub j, j not equal i" | Every **other** caption in the batch. These are the negatives. |
| $\cos$ | "cosine similarity" | From lecture 3: $\frac{u \cdot v}{\|u\|\|v\|}$, measuring angle only. |

The slide's matrix, with five images down the side and five captions across the top:

```
              a dog   a car  sunset   pizza  a plane
   🐕 dog     [0.50]   0.40    0.60    0.30    0.50
   🚗 car      0.40   [0.55]   0.30    0.60    0.40
   🌅 sunset   0.60    0.30   [0.45]   0.40    0.60
   🍕 pizza    0.30    0.60    0.50   [0.50]   0.35
   ✈️ plane    0.50    0.40    0.60    0.35   [0.50]

   [boxed] = matching pair (should be HIGH)
   others  = mismatch      (should be LOW)
```

Note that the slide shows the **before training** state — the diagonal is *not* yet higher than the
off-diagonal entries (row 1: the diagonal is 0.50 but "sunset" scores 0.60). The slide's toggle
caption:

> **Training sharpens the diagonal, so matching pairs light up and the rest go dark.**

> 💡 **Why the batch is the supervision — this is the elegant part.** For a batch of $N$ pairs, you
> get **$N$ positives** (the diagonal) and **$N^2 - N$ negatives** (everything else) *for free*, just
> by pairing each image with the other captions. With $N = 32{,}768$, that's ~32k positives and over
> **a billion** negatives per batch, from data that required no annotation. **Bigger batches give
> more negatives, which give a harder and more informative task** — which is why CLIP-style training
> is unusually batch-size hungry.

```interactive
type: animation
title: The similarity matrix, before and after training
concept: Contrastive training sharpening a similarity matrix's diagonal
control: A before/after toggle (the deck's own two slide states)
observe: Every cell's shading updates — before training, diagonal and off-diagonal cells look similarly bright; after training, the diagonal cells light up brightly while every off-diagonal cell darkens
insight: Contrastive training isn't pulling matched pairs to some absolute similarity target — it's a purely relative, within-batch objective that pushes the diagonal up and everything else down at the same time, using the batch itself as the entire source of negative examples
fallback: The two 5×5 matrices already shown above (the "before" matrix, where row 1's diagonal 0.50 loses to sunset's 0.60; and the stated "after" behaviour, diagonal sharpens and the rest go dark) are the exact two toggle states this control switches between.
```

### The deep-dive slide

*(slide_006.jpg — "CLIP, in detail: contrastive alignment at scale")*

> **Two encoders, one shared space, trained on 400M image-text pairs with a symmetric contrastive
> loss.**

**Two encoders:**

> An image encoder (**ViT** or **ResNet**) and a text encoder (**Transformer**) each map their input
> to a vector, then a learned **linear projection** puts both into the **same $d$-dimensional
> space**; both are **L2-normalised**.

> 💡 **Note the L2 normalisation** — this is exactly lecture 3's lesson. Normalising to unit length
> means comparisons depend only on **angle**, not magnitude, so a "longer" image embedding can't win
> on length. It also makes cosine similarity equal to a plain dot product, which is faster.

**The batch is the supervision:**

> For a batch of $N$ pairs, form the $N \times N$ cosine-similarity matrix. The $N$ true pairs
> (diagonal) are positives; the $N^2 - N$ off-diagonal pairs are negatives.

**The loss — symmetric InfoNCE:**

> Softmax over each **row** (image→text) **and** each **column** (text→image), scaled by a learned
> temperature $\tau$; average the two cross-entropies.

**In words: treat each row of the similarity matrix as a classification problem — "which of these
$N$ captions belongs to this image?" — and score it with ordinary cross-entropy. Then do the same
for each column, asking "which image belongs to this caption?" Average the two.**

$$\mathcal{L} = \tfrac{1}{2}\Big[\mathrm{CE}\big(\tfrac{IT^\top}{\tau},\, y\big) + \mathrm{CE}\big(\tfrac{TI^\top}{\tau},\, y\big)\Big]$$

| Symbol | Read it as | What it means |
|---|---|---|
| $I$ | "I" | Matrix of image embeddings, $N \times d$. |
| $T$ | "T" | Matrix of text embeddings, $N \times d$. |
| $IT^\top$ | "I T transpose" | The $N \times N$ similarity matrix (image rows, text columns). |
| $TI^\top$ | "T I transpose" | Its transpose — the same matrix read the other way. |
| $\tau$ | "tau", the **temperature** | A **learned** scalar that scales the logits. Same role as lecture 3's decoding temperature: it controls how peaked the softmax is. |
| $\mathrm{CE}(\cdot, y)$ | "cross-entropy against y" | Ordinary cross-entropy (lecture 1). |
| $y$ | "y" | The labels — which are just $[0, 1, 2, \ldots, N-1]$, since pair $i$'s correct match is at position $i$. |
| $\tfrac{1}{2}[\cdot + \cdot]$ | "average the two" | **Symmetric**: both directions matter equally. |

> 💡 **Look at what the loss actually is.** It is **cross-entropy classification** — the same loss
> from lecture 1 — where the "classes" are the other items in the batch. There is no new machinery.
> CLIP is a classifier whose label set is generated on the fly from the batch itself.
>
> And **$\tau$ is learned, not set.** The model discovers for itself how sharply to separate
> positives from negatives. If $\tau$ is too large the softmax is flat and there's no gradient
> signal; too small and it saturates (lecture 1's $\sqrt{d}$ argument, again).

### Zero-shot classification — the payoff

> **Zero-shot transfer: embed class names as "a photo of a {class}", embed the image, pick the
> nearest text vector. No fine-tuning, no labelled head.**

This is what made CLIP famous, and the mechanism is beautifully simple.

**Worked example — classifying an image with zero training:**

```
Step 1 — write your classes as sentences:
  "a photo of a dog"
  "a photo of a cat"
  "a photo of a horse"

Step 2 — embed each with the TEXT encoder:
  T_dog   = [0.21, -0.44, ...]
  T_cat   = [0.19, -0.38, ...]
  T_horse = [-0.11, 0.62, ...]

Step 3 — embed your image with the IMAGE encoder:
  I = [0.20, -0.41, ...]

Step 4 — cosine similarity against each (all unit vectors, so just dot products):
  cos(I, T_dog)   = 0.31
  cos(I, T_cat)   = 0.28
  cos(I, T_horse) = 0.05

Answer: "a photo of a dog"
```

> 💡 **Why this is remarkable, stated plainly.** A traditional image classifier is trained on a
> **fixed** list of classes with a labelled output layer. Want to add "capybara"? Collect labelled
> capybara photos and retrain.
>
> CLIP has **no output layer and no class list.** You define the classes *at inference time, by
> typing them*. Add "capybara" by writing the string `"a photo of a capybara"`. Zero training, zero
> labels, works immediately.
>
> **The classification head became a sentence.** That reframing is what CLIP contributed, and it's
> why CLIP's text encoder ends up inside almost every text-to-image model in Part 2 — it is the
> component that already knows how words relate to pictures.

### The scale

**400 million image-text pairs.** Note this is roughly the same lesson as lecture 1's scaling
laws: the *architecture* is two ordinary encoders and a cross-entropy loss. **The result came from
the data scale**, which contrastive learning made possible by requiring no labels.

### Where people get confused

**You might think** CLIP generates images. **Actually** CLIP only **scores** how well an image and a
text match. It's an *understanding* model, not a generator. (It shows up inside generators as the
text encoder — sections 9 and 17.)

**You might think** the prompt template "a photo of a {class}" is arbitrary. **Actually** it
measurably matters, because it moves the text embedding closer to the distribution of real captions
CLIP trained on. Bare class names ("dog") perform worse than natural captions. This is
**prompt engineering for a classifier**, and it was one of the earliest demonstrations that prompt
wording has real effects.

**You might think** CLIP understands compositional structure. **Actually** it is notably weak at it
— it often scores *"a red cube on a blue sphere"* and *"a blue cube on a red sphere"* almost
identically. CLIP is closer to a **bag of concepts** than a parser of relationships. This limitation
propagates into image generators built on it, and is exactly what the **GenEval** and
**T2I-CompBench** benchmarks in section 22 were built to measure.

### 🔬 Research opportunity

CLIP's compositional weakness is a well-documented, unsolved problem with practical consequences —
it's a direct cause of text-to-image models mixing up attributes ("a red hat and blue scarf"
producing a blue hat). Work on contrastive objectives with **hard negatives** that differ only by
word order or attribute binding is active, tractable at small scale, and directly useful.

---

## 3. Two ways to make an LLM multimodal

*(slide_007.jpg and slide_008.jpg — one slide in two toggle states; slide_006.jpg belongs to §2's
CLIP deep-dive, not this section)*

> **Bolt an encoder onto a frozen text model, or train one model on all modalities at once.**

### Modular (e.g. LLaVA)

> **Keep a strong text LLM frozen. Add a vision encoder and a small projection that maps image
> features into the LLM's token space. Train only the bridge.**

```
Modular: a vision encoder bolted onto a frozen LLM

   🖼️  ──►  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐
            │   vision     │──►│  projection  │──►│  frozen LLM    │
            │ encoder(CLIP)│   │  → LLM tokens│   │ text + image   │
            └──────────────┘   └──────────────┘   │     tokens     │
                                      ▲            └────────────────┘
                              the ONLY trained part
```

*In everyday words:* hiring a translator rather than teaching your existing employee a new language.
The LLM already speaks "token"; the projection translates image features into that language.

**How it actually works, concretely:**

```
1. CLIP's image encoder turns the image into (say) 576 patch feature vectors,
   each of dimension 1024.

2. A small trained projection — often just one or two linear layers —
   maps each 1024-dim vector to the LLM's token dimension, say 4096.

3. Those 576 vectors are inserted into the LLM's input sequence
   as if they were 576 ordinary token embeddings.

4. The LLM, which has never been modified, simply attends over them
   alongside the text tokens.
```

> 💡 **The elegance is that step 3 requires nothing from the LLM.** The LLM's input layer accepts a
> sequence of vectors of the right dimension. It has no way to tell that 576 of them came from an
> image rather than a tokenizer. **The projection's entire job is to produce vectors that "look
> like" token embeddings to a model that was never told about images.**

**The trade-off the slide states:**

> **Cheap, fast, reuses an existing model, but the LLM was never *trained* to reason over pixels:
> vision is an add-on.**

Because the LLM's weights are frozen, its internal representations were shaped entirely by text.
Visual information has to be squeezed into a space built for language. It works surprisingly well —
LLaVA was trained for a few hundred GPU-hours — but there's a ceiling.

### Unified (e.g. GPT-4o, Gemini)

> **Tokenise every modality and train one model on all of them jointly. Any-to-any, deeper
> grounding.**

```
Unified: every modality is just tokens into one model

   text  → tokens  ──┐
                     │
   image → patches ──┼──►  ┌──────────────────┐
                     │     │  one transformer │
   audio → frames  ──┘     │  all tokens mixed│
                           └──────────────────┘
```

*In everyday words:* raising someone bilingual from birth rather than teaching them a second
language as an adult. The representations are shared from the ground up.

**The trade-off:**

> **More capable and more expensive: you pay for joint pretraining, but get native cross-modal
> reasoning and generation.**

> 📚 **What "any-to-any" means.** A modular model typically goes *image + text in → text out*. A
> unified model can take any combination in and produce any combination out: describe an image,
> generate an image from text, answer a spoken question in speech, edit an image from a spoken
> instruction. **Generation in every modality, not just understanding.**

### Comparison

| | **Modular (LLaVA)** | **Unified (GPT-4o, Gemini)** |
|---|---|---|
| What's trained | Only the projection bridge | Everything, jointly |
| Cost | Hundreds of GPU-hours | Full pre-training run |
| Base LLM | Frozen, reused | Trained from scratch on all modalities |
| Directions | Usually image+text → text | **Any-to-any** |
| Grounding depth | Vision is an add-on | Native |
| Can generate images/audio? | ❌ Generally no | ✅ Yes |
| Who can build one | Almost anyone | Frontier labs |

> 💡 **This is the same trade-off as lecture 3's PEFT-versus-full-training, one level up.** Modular
> is the PEFT of multimodality: freeze the expensive part, train a small bridge, accept a ceiling.
> Unified is full training: maximum capability, maximum cost. **And as in lecture 3, the right
> answer depends on your budget and how far you need to push quality — not on which is
> intrinsically better.**

```interactive
type: diagram
title: Modular vs. unified multimodal architectures
concept: Two different ways to give an LLM access to non-text modalities
control: A toggle between "Modular" and "Unified" (the deck's own two slide states)
observe: The pipeline diagram redraws — modular shows a frozen LLM block with a small trainable projection bridge feeding in; unified shows a single block ingesting text, image, and audio tokens jointly, all trainable together
insight: The architectural difference maps directly onto the comparison table's every row — cost, who can build one, and whether generation (not just understanding) is possible in non-text modalities — because "what gets trained" is the single variable that determines all of it
fallback: The comparison table already in this section (7 rows: what's trained, cost, base LLM, directions, grounding depth, generation capability, who can build one) is the complete static equivalent of what this toggle would visually contrast.
```

### Where people get confused

**You might think** modular models are obsolete now that unified ones exist. **Actually** modular is
how essentially every open-weight vision-language model is built, because it's the only approach
most organisations can afford. LLaVA-style training remains the standard recipe.

**You might think** the projection must be complex. **Actually** the original LLaVA used a **single
linear layer** and it worked. Later versions used a two-layer MLP for a modest gain. The heavy
lifting is done by the pre-trained encoder and the pre-trained LLM; the bridge just has to align
them.

---

## 4. From a pipeline of models to one that speaks

*(Slides 9–10 — one slide in two toggle states)*

> **Stitching speech-to-text, an LLM, and text-to-speech works, but a native model is faster and
> keeps the voice.**

### The cascade

> **Whisper transcribes speech, an LLM answers, a TTS model speaks. Three models in series.**

```
Cascaded: speech → text → LLM → text → speech

  👤 ──► ┌─────────┐ ──► ┌─────┐ ──► ┌─────┐ ──► 🔊
         │ Whisper │     │ LLM │     │ TTS │
         └─────────┘     └─────┘     └─────┘

  three models in series: errors and delay stack up

  latency ████████████████████████████████  ~2.2 s
```

> 📚 **Background — Whisper and TTS.**
> **Whisper** is OpenAI's speech-recognition model (an encoder–decoder transformer, from lecture 1's
> taxonomy). It converts audio to text.
> **TTS (Text-To-Speech)** does the reverse: text to audio waveform.

### The two problems

> **The problem: latency and errors *stack*, and everything in the voice (tone, emotion, who is
> speaking) is *thrown away* at the text bottleneck.**

**Problem 1 — latency stacks.** Three models in series means three waits:

```
Whisper transcription:   ~0.5 s   (and it must wait for you to stop talking)
LLM generation:          ~1.2 s
TTS synthesis:           ~0.5 s
                         ───────
Total:                   ~2.2 s      ✓ matches the slide
```

> 💡 **Why 2.2 seconds is a product-killing number.** In natural human conversation, the gap between
> turns is around **200 milliseconds**. At 2.2 seconds, every exchange has an awkward pause. You
> cannot interrupt it, you cannot have it back-channel ("mm-hmm"), and it never feels like talking
> to someone. **The latency budget for voice is set by human conversation, not by what's technically
> convenient.**

**Problem 2 — the text bottleneck destroys information.** This one is more interesting.

```
What you said:   "I'm FINE."   (clipped, rising pitch, clearly not fine)
                        ↓
What Whisper passes on:  "I'm fine."
                        ↓
What the LLM sees:  "I'm fine."     ← the sarcasm is gone. Forever.
```

Everything **paralinguistic** — tone, emotion, emphasis, hesitation, accent, who is speaking, whether
you're whispering or shouting — exists in the audio and **does not exist in the transcript**. The
text bottleneck is a lossy compression that throws away exactly the information that makes speech
different from writing.

And errors stack too: a Whisper mistranscription is invisible to the LLM, which answers the wrong
question with complete confidence.

### Native speech-to-speech

> **One model hears audio and emits audio directly: GPT-4o voice, Gemini Live. Sub-second,
> interruptible, keeps prosody.**

```
Native: one model hears and speaks directly

  🎤 ──► ┌───────────────────────────────┐ ──► 🔊
         │  single speech-to-speech model│
         └───────────────────────────────┘

  latency ████  ~0.3 s
```

**~2.2 s → ~0.3 s.** A **7.3×** reduction, and — crucially — it crosses the threshold from
"noticeably laggy" to "conversational".

> 📚 **Background — prosody.** The musical properties of speech: pitch, rhythm, stress, intonation.
> Prosody carries meaning that words alone don't. *"You're going."* versus *"You're **going**?"* are
> the same words and different sentences. A native model keeps this on input **and** produces it on
> output, so it can respond with matching tone.

**And "interruptible" matters more than it sounds.** A cascade cannot easily be interrupted —
Whisper is waiting for you to finish a complete utterance before it emits anything. A native model
processing a continuous audio stream can detect that you've started speaking and stop, which is how
real conversations work.

This is section 1's thesis applied directly: **audio is just another token stream.** Once the model
consumes and produces audio tokens natively, the text bottleneck simply doesn't exist.

### Why it matters for agents

> **Real-time voice is the interface for assistants you *talk to*, so the latency budget is
> human-conversation tight.**

The slide's closing framing, and the bridge to whatever comes after this lecture series: if the
interface is voice, latency is not an optimisation — it is the difference between a product that
feels alive and one that doesn't.

### Comparison

| | **Cascaded** | **Native speech-to-speech** |
|---|---|---|
| Models | 3 in series | 1 |
| Latency | **~2.2 s** | **~0.3 s** |
| Prosody preserved | ❌ Lost at the text bottleneck | ✅ |
| Interruptible | ❌ Hard | ✅ |
| Errors | Stack across three models | Single model |
| Debuggability | ✅ You can inspect the transcript | ❌ Opaque |
| Build difficulty | ✅ Off-the-shelf components | ❌ Requires a frontier-scale model |
| Examples | Whisper + GPT + TTS | GPT-4o voice, Gemini Live |

> 💡 **The cascade is not obsolete, and the last two rows are why.** You can build one this
> afternoon from three APIs, and when something goes wrong you can read the intermediate transcript
> and see exactly where. Native models are better and almost nobody can train one. **For most teams
> the real choice is "cascade you build" versus "native you rent".**

```interactive
type: animation
title: Cascaded versus native speech-to-speech
concept: Why collapsing three models into one collapses latency, not just architecture
control: A toggle between "Cascaded" and "Native" (the deck's own two slide states)
observe: The pipeline diagram redraws — cascaded shows three boxes in series (Whisper → LLM → TTS) with a latency bar filling to ~2.2 s; native shows one box (a single speech-to-speech model) with the latency bar collapsing to ~0.3 s
insight: The latency bar's length is a direct visual proxy for "how many models are in series", so the 7.3× drop is not a tuning win — it's the disappearance of two serial waits and the text bottleneck they created
fallback: The two ASCII pipeline diagrams already in this section (the three-box cascade with its ~2.2 s bar, and the single-box native model with its ~0.3 s bar) are the exact two toggle states this control switches between.
```

### 🎯 Interview question

*Your voice assistant feels sluggish and misses sarcasm. Diagnose both.* → Both are symptoms of a
cascaded architecture. **Sluggish**: three models in series stack latency to ~2.2 s against a
~200 ms conversational budget. **Missing sarcasm**: the speech-to-text step discards all
paralinguistic information — tone, pitch, emphasis — so the LLM only ever sees flat text. Neither is
fixable by improving the individual components; the text bottleneck is architectural. Move to a
native speech-to-speech model, or accept the ceiling.

---

# PART 2 · Generating by denoising

*Diffusion models: the engine behind modern image and video generation.*

## 5. Destroy an image with noise, then learn to rebuild it

*(Slide 12)*

> **The forward process is fixed and easy; all the learning is in reversing it.**

### The intuition first

Here is the whole idea, before any notation.

Take a photograph. Add a tiny bit of random static. Add a tiny bit more. Keep going, a thousand
times, until the photograph is **completely destroyed** — indistinguishable from television static.

That destruction process is **trivial**. It has no parameters, nothing to learn; it's a fixed recipe
anyone can run.

Now train a neural network to do **one step of the reverse**: given a slightly-noisy image, identify
the noise that was added. That's a well-posed supervised learning problem, because **you know the
answer** — you added the noise yourself.

Once the network can undo one step, you can undo all of them. And here's the leap: **start from
pure static that never contained an image at all**, and run the reverse process anyway. The network
removes "the noise" step by step, and what remains is a coherent image — one that has never
existed.

> 💡 **That last move is the conceptual jump worth sitting with.** The model was only ever trained
> to *clean up* images. But cleaning up random static, repeatedly, **hallucinates a picture out of
> nothing**. The model has no way to distinguish "static that used to be a photo" from "static that
> never was" — so it treats the second like the first and invents a photo to match.

```interactive
type: slider
title: Diffusion step t, from clean image to pure noise
concept: The forward process is a continuous dial between a real image and static, and the reverse process runs it backwards
control: Drag t from 0% noise (x_0, the clean image) to 100% noise (x_T, pure static) and back (the deck's own slider)
observe: The grid of pixels shown at each stopping point gets progressively noisier as t rises — sharp dark/light dots blur into uniform grey static — and reverses cleanly when dragged back down
insight: Because the forward process is Markov and has no parameters, every intermediate x_t is a well-defined, reproducible point on one fixed path — there's no "randomness in the destruction schedule" to worry about, only in the noise draw itself
fallback: The slide's static six-panel strip (x_0 through x_t, forward arrow labelled "+ noise" above, reverse arrow labelled "predict & subtract noise ε" below) already in this section shows the same six fixed stopping points this slider would let you scrub between continuously.
```

### The forward process

> **Start from a real image and add a little Gaussian noise at each of $T$ steps, until it is pure
> static. No learning here: it is just a fixed recipe.**

```
forward q(x_t | x_{t-1}) : + noise   ────────────────────────────►

  ▉▉▉▉      ▉▉░▉       ▉░░▉        ░▉░░       ░░░░       ░░░░
   x_0       x_1        x_2         x_3        x_4        x_T
 clean                                                pure noise

◄────────────────────  reverse p_θ(x_{t-1} | x_t) : predict & subtract noise ε
```

### The reverse process, and the loss

> **Train a network to take a noisy image and predict the noise that was added, so we can subtract
> it and step back toward a clean image.**

**In words, the loss says: add a known amount of noise to a real image, ask the network what noise
was added, and penalise it by the squared difference between its guess and the truth.**

$$\mathcal{L} = \mathbb{E}\big[\,\|\epsilon - \epsilon_\theta(x_t, t)\|^2\,\big]$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\mathcal{L}$ | "loss" | Minimised by gradient descent, as always. |
| $\mathbb{E}[\cdot]$ | "expectation" | Average over random images, random timesteps, and random noise draws. |
| $\epsilon$ | "epsilon" | The **actual noise** that was added. **We know this — we generated it.** |
| $\epsilon_\theta(x_t, t)$ | "epsilon theta of x-t and t" | The **network's prediction** of that noise. $\theta$ = its parameters. |
| $x_t$ | "x sub t" | The noisy image at step $t$. |
| $t$ | "t" | Which timestep — i.e. *how noisy* this image is. **The network is told this.** |
| $\|\cdot\|^2$ | "squared norm" | Square every component of the difference and sum. |

> 💡 **Stop and notice what this loss is: plain MSE regression.** After all the talk of Markov chains
> and Gaussians, the training objective is *"predict this number, minimise squared error"* — the
> most ordinary loss in machine learning. There is no adversarial game (unlike GANs), no likelihood
> bound to fight with, no instability. **That simplicity is a large part of why diffusion beat GANs**:
> it trains reliably.

> 💡 **Why the network needs $t$ as an input.** An image at $t = 10$ is barely noisy; at $t = 900$
> it's almost pure static. The *same* input pixels demand very different responses depending on how
> far along the chain you are. Telling the network $t$ lets one set of weights handle every noise
> level. (How $t$ is fed in — a **timestep embedding** — appears in sections 14 and 16.)

### The score view

> **Score matching: Equivalently, the model learns the gradient of the data density
> $\nabla_x \log p(x)$ — which direction makes the image more realistic.**

From Prerequisite 4: the **score** is an arrow pointing toward "more probable". The mathematical
result — and it's a genuinely deep one — is that **predicting the noise is the same thing as
computing the score**, up to a scale factor.

Intuitively: the noise you added is precisely what makes the image *less* realistic. So pointing at
the noise is pointing *away* from realism, and the negative of it points *toward* realism. **Noise
prediction and score estimation are the same arrow, opposite signs.**

> 💡 **Why this equivalence matters historically.** Two research communities arrived at diffusion
> independently — one from thermodynamics and denoising (Sohl-Dickstein, Ho et al.), one from
> score-based generative modelling (Song & Ermon). The recognition that they were describing the
> **same model** unified the field and gave it the continuous-time / ODE view that later made
> fast samplers possible (section 10).

### To generate

> **Start from pure noise and run the reverse steps. The model hallucinates a clean image out of
> static.**

```
GENERATION (sampling):

  1. Draw x_T ~ N(0, I)          ← pure random static, no image involved
  2. For t = T, T-1, ..., 1:
        predict the noise:  ε̂ = ε_θ(x_t, t)
        subtract a step of it to get x_{t-1}
  3. Return x_0                  ← a clean image that never existed
```

### Where people get confused

**You might think** the model memorises training images and replays them. **Actually** it learns a
*denoising function*, and every generation starts from a fresh random draw of $x_T$. Different noise
→ different image. (Memorisation *can* happen for images duplicated many times in training — a
real, documented problem — but it is a failure mode, not the mechanism.)

**You might think** the forward process is learned. **Actually** it has **zero parameters**. It's a
fixed schedule of "add this much noise at step $t$". All the learning is in the reverse direction.

**You might think** generation subtracts all the predicted noise at once. **Actually** each step
subtracts only a *portion*, moving one rung down the ladder, and (in the stochastic sampler) adds a
little fresh noise back. Jumping straight from $x_T$ to $x_0$ in one subtraction produces a blurry
average — the model's prediction at high noise is an average over *all* plausible images, and the
iterative process is what resolves that average into one specific image.

---

## 6. The forward and reverse processes, step by step

*(Slide 13)*

> **A fixed Markov chain destroys the image; a learned chain rebuilds it.**

### The forward chain

> **Forward ($q$): A *fixed* schedule adds a little Gaussian noise at each step $t$. After $T$ steps
> the image is indistinguishable from static. **No parameters, nothing to learn.**

```
The diffusion process: a fixed forward chain, a learned reverse

  ┌────┐   ┌────┐   ┌────┐   ┌────┐        ┌────┐
  │ x₀ │ → │ x₁ │ → │ x₂ │ → │ x₃ │ ⋯⋯⋯ → │ x_T│
  └────┘   └────┘   └────┘   └────┘        └────┘
   clean                                  pure noise
    ◄────────────────────────────────────────────
     reverse p_θ(x_{t-1}|x_t) — a network predicts & subtracts the noise (learned)
```

### The shortcut — the fact that makes training affordable

> **Because Gaussians compose, you can jump straight to any step:
> $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon$.
> Training samples a random $t$ directly.**

**In words: a noisy image at step $t$ is just the clean image shrunk by one factor, plus pure noise
scaled by another — and the two factors always combine to keep the total size constant.**

$$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon$$

| Symbol | Read it as | What it means |
|---|---|---|
| $x_t$ | "x sub t" | The noisy image at step $t$. |
| $x_0$ | "x nought" | The original clean image. |
| $\epsilon$ | "epsilon" | A fresh draw from $\mathcal{N}(0, I)$ — pure standard noise. |
| $\bar{\alpha}_t$ | "alpha-bar sub t" | **How much of the original signal survives** at step $t$. Runs from ≈1 (clean) down to ≈0 (destroyed). |
| $\sqrt{\bar{\alpha}_t}$ | "root alpha-bar" | The weight on the **signal**. |
| $\sqrt{1-\bar{\alpha}_t}$ | "root one minus alpha-bar" | The weight on the **noise**. |

> 💡 **Why the two coefficients are square roots that sum in squares.** Notice
> $(\sqrt{\bar\alpha_t})^2 + (\sqrt{1-\bar\alpha_t})^2 = \bar\alpha_t + 1 - \bar\alpha_t = 1$.
> Variances add, so this keeps the **total variance constant at every step**. The image never grows
> or shrinks in overall scale as it's destroyed — only the *ratio* of signal to noise changes. This
> is called a **variance-preserving** schedule, and it's what keeps the network's inputs in a stable
> numerical range throughout.

**Worked example — noising an image to three different levels:**

```
Take one pixel value  x_0 = 0.8   and one noise draw  ε = -1.2

At t = 100, suppose ᾱ = 0.90:
  x_t = √0.90 × 0.8  +  √0.10 × (-1.2)
      = 0.9487 × 0.8 + 0.3162 × (-1.2)
      = 0.7590 - 0.3795
      = 0.3795                      ← signal still dominant

At t = 500, suppose ᾱ = 0.50:
  x_t = √0.50 × 0.8  +  √0.50 × (-1.2)
      = 0.7071 × 0.8 + 0.7071 × (-1.2)
      = 0.5657 - 0.8485
      = -0.2828                     ← half and half

At t = 900, suppose ᾱ = 0.05:
  x_t = √0.05 × 0.8  +  √0.95 × (-1.2)
      = 0.2236 × 0.8 + 0.9747 × (-1.2)
      = 0.1789 - 1.1696
      = -0.9907                     ← almost entirely noise
```

> 💡 **This one equation is why diffusion training is cheap, and it deserves emphasis.** Without it,
> producing a training example at $t = 900$ would mean simulating 900 sequential noising steps.
> With it, you pick a random $t$, look up $\bar\alpha_t$ in a precomputed table, and get $x_t$ in
> **one line of arithmetic**.
>
> So a training step is: pick a random image, pick a random $t$, draw noise, compute $x_t$ in one
> shot, ask the network to predict $\epsilon$, take the MSE. **No sequential simulation anywhere.**
> Training is fully parallel across timesteps — the same property that made transformers trainable
> at scale (lecture 1).

### The reverse chain

> **Reverse ($p_\theta$): A network looks at $x_t$ and the step $t$, and predicts the noise
> $\epsilon$. Subtract it and you step one rung back toward a clean image.**

### The punchline

The slide's closing box states the thing to remember above everything else in Part 2:

> **The whole model is just an $\epsilon$-predictor. Everything else — guidance, samplers, latents,
> backbones — is built around this one regression target.**

> 💡 **Hold onto this as your organising frame for the next fifteen sections.** Every technique that
> follows is a modification *around* a noise-predicting network, not a change to it:
> - **Guidance** (section 9) — run the ε-predictor twice and combine the outputs.
> - **Samplers** (section 10) — take smarter steps between ε-predictions.
> - **Distillation** (section 11) — train a student to make bigger jumps.
> - **Latents** (section 12) — run the ε-predictor in a smaller space.
> - **Backbones** (sections 14–16) — change *what kind of network* predicts ε.
> - **Conditioning** (sections 17–18) — feed extra signals into the ε-predictor.
>
> **The regression target never changes.** If you keep that fixed point in mind, the rest of this
> lecture is a series of variations on one theme rather than a pile of unrelated tricks.

---

## 7. DDPM, in detail

*(Slide 15)*

> **A fixed Gaussian Markov chain forward; a learned Gaussian Markov chain in reverse.**

This slide formalises section 6. **DDPM** = **D**enoising **D**iffusion **P**robabilistic **M**odels
(Ho, Jain & Abbeel, NeurIPS 2020) — the paper that made diffusion work for images.

### The forward step

**In words: to get the next noisy image, shrink the current one slightly and add a small amount of
fresh Gaussian noise.**

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1 - \beta_t}\,x_{t-1},\ \beta_t I\big)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $q(x_t \mid x_{t-1})$ | "q of x-t given x-t-minus-1" | The forward (noising) distribution. $q$ by convention denotes the **fixed** process. |
| $\mathcal{N}(\mu, \sigma^2 I)$ | "normal with mean μ, variance σ²" | Prerequisite 1. |
| $\beta_t$ | "beta sub t" | **How much noise is added at step $t$.** A small number, typically 0.0001 → 0.02. |
| $\sqrt{1-\beta_t}\,x_{t-1}$ | | The mean: the previous image, **shrunk slightly**. |
| $\beta_t I$ | | The variance: independent noise of size $\beta_t$ in every dimension. |

> **The $\beta_t$ follow a fixed schedule (linear or cosine); no learning.**

### The closed form

> **With $\bar{\alpha}_t = \prod_{s \le t}(1 - \beta_s)$, you sample $x_t$ in one shot, which is what
> makes training cheap.**

| Symbol | Read it as | What it means |
|---|---|---|
| $\prod_{s \le t}$ | "product over s up to t" | Multiply together, rather than add. |
| $(1 - \beta_s)$ | "one minus beta-s" | The fraction of signal surviving step $s$. |
| $\bar{\alpha}_t$ | "alpha-bar sub t" | **The fraction surviving all $t$ steps** — the cumulative product. |

**Worked example — computing $\bar\alpha$ from a schedule:**

```
Suppose β = [0.02, 0.03, 0.04, 0.05]   (a very short 4-step schedule)

α_s = 1 - β_s:
  α_1 = 0.98,  α_2 = 0.97,  α_3 = 0.96,  α_4 = 0.95

Cumulative products:
  ᾱ_1 = 0.98
  ᾱ_2 = 0.98 × 0.97 = 0.9506
  ᾱ_3 = 0.9506 × 0.96 = 0.9126
  ᾱ_4 = 0.9126 × 0.95 = 0.8670

So at t = 4, 86.7% of the signal (in variance terms) remains,
and x_4 = √0.8670 x_0 + √0.1330 ε = 0.9311 x_0 + 0.3647 ε
```

With a real 1000-step schedule, $\bar\alpha_T$ ends up near **0** — the image is gone.

### The reverse step

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(\mu_\theta(x_t, t),\ \Sigma_t\big)$$

> **The network predicts the noise $\epsilon_\theta$; $\mu_\theta$ is an algebraic function of it.**

| Symbol | Read it as | What it means |
|---|---|---|
| $p_\theta$ | "p theta" | The **learned** reverse distribution. $\theta$ = network parameters. |
| $\mu_\theta(x_t, t)$ | "mu theta" | The predicted mean of the previous, cleaner image. |
| $\Sigma_t$ | "Sigma sub t" | The variance. In DDPM this is **fixed**, not learned. |

> 💡 **"$\mu_\theta$ is an algebraic function of $\epsilon_\theta$" is the practical point.** The
> network never outputs an image or a mean. It outputs **noise**, and a fixed formula converts that
> into where to step next. One output, one target, one loss.

### The simplified loss

> **The variational bound collapses to a plain noise-prediction MSE; train by adding known noise and
> regressing it back.**

$$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon \quad \Longrightarrow \quad \mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\big\|\epsilon - \epsilon_\theta(x_t, t)\big\|^2$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\mathbb{E}_{t, x_0, \epsilon}$ | "expectation over t, x-nought, and epsilon" | Average over random **timestep**, random **training image**, and random **noise draw**. All three are sampled fresh each step. |
| $\mathcal{L}_{\text{simple}}$ | "L simple" | The name from the paper — "simple" because it drops the theoretically-derived per-timestep weights. |

> 💡 **The word "collapses" is the story of the DDPM paper.** The principled derivation gives a
> **variational lower bound** — a complicated expression with a weighting term for each timestep.
> Ho et al.'s finding was that **dropping the weights and using plain, unweighted MSE worked
> better in practice.** The theory motivates the model; a simplification of it is what you actually
> train. (Section 8 shows that those weights are not entirely irrelevant after all — Min-SNR
> reintroduces a smarter version of them.)

**The complete training loop, in seven lines:**

```
repeat:
    x_0  ← sample a real image from the dataset
    t    ← sample a random timestep, uniform in 1..T
    ε    ← sample noise from N(0, I)
    x_t  ← √(ᾱ_t) x_0 + √(1-ᾱ_t) ε          # one-shot, no simulation
    ε̂    ← ε_θ(x_t, t)                       # network forward pass
    loss ← ‖ε - ε̂‖²                          # plain MSE
    backprop, update θ
```

**That is the entire training algorithm.** Compare it to lecture 2's PPO — four networks, a sampling
loop, a critic, clipping, KL penalties. Diffusion training is one network and squared error.

---

## 8. What the network predicts, and how the noise is scheduled

*(Slide 16)*

> **The same model can regress $\epsilon$, $x_0$, or a velocity $v$ — and the schedule decides which
> timesteps it ever sees.**

This slide is the most technically dense in the deck, and it contains three of the most practically
important facts in modern diffusion. Take it in three parts.

### Part A — three equivalent prediction targets

> **With $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$, the unknowns $x_0$ and
> $\epsilon$ are linearly related, so predicting any one fixes the others.**

Look at the equation. You know $x_t$ (it's the input) and $\bar\alpha_t$ (it's in a table). So if
the network tells you $\epsilon$, you can solve for $x_0$ algebraically — and vice versa. **They
carry identical information.**

So why does the choice matter? **Numerical conditioning.**

| Target | What the network predicts | Where it's **ill-conditioned** |
|---|---|---|
| **$\epsilon$-pred** (DDPM default) | The noise | **Noisy near $t = 0$** |
| **$x_0$-pred** | The clean image | **Noisy near $t = T$** |
| **$v$-pred** | A velocity, $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$ | **Well-scaled across all $t$** |

> 💡 **Work through *why* each fails at its bad end — this is the insight the slide compresses into
> one line.**
>
> **$\epsilon$-pred near $t = 0$:** the image is almost clean, so $\sqrt{1-\bar\alpha_t} \approx 0$.
> The noise contributes almost nothing to $x_t$. Asking the network to recover $\epsilon$ from
> $x_t$ is asking it to detect a signal that has been multiplied by ~0.03 — and then that tiny
> prediction gets divided by the same tiny number to recover $x_0$. **Any error is amplified
> enormously.**
>
> **$x_0$-pred near $t = T$:** now $\sqrt{\bar\alpha_t} \approx 0$. The clean image contributes
> almost nothing to $x_t$. Asking for $x_0$ from near-pure static is asking it to invent an image
> from nothing — the target is essentially unlearnable there.
>
> **$v$-pred:** it's a *rotation* of the $(\epsilon, x_0)$ pair — a mix whose weights change with
> $t$ so the target stays a well-scaled quantity everywhere. Neither end degenerates.

The slide notes v-prediction is used by **SD2.x, Imagen, and distillation** (section 11). That last
one matters: distillation makes very large steps, where $\epsilon$-pred's conditioning problems
become severe.

### Part B — the loss is a weighted regression

> **Every target is the same objective up to a per-timestep weight $w(t)$:**

$$\mathcal{L} = \mathbb{E}_t\, w(t)\,\big\|f_\theta(x_t, t) - \text{target}\big\|^2$$

| Symbol | Read it as | What it means |
|---|---|---|
| $w(t)$ | "w of t" | How much timestep $t$ contributes to the total loss. |
| $f_\theta(x_t, t)$ | "f theta" | The network's output, whatever target it predicts. |

> **$\epsilon$-pred implicitly sets $w(t) = 1$**; **Min-SNR-$\gamma$ caps
> $w(t) = \min(\mathrm{SNR}(t), \gamma)$ so high-SNR steps stop dominating the gradient.**

> 📚 **Background — SNR (signal-to-noise ratio).**
> $$\mathrm{SNR}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t}$$
> Signal variance over noise variance. **High SNR** = barely noisy ($t$ near 0). **Low SNR** =
> mostly noise ($t$ near $T$).
>
> *Concretely:* at $\bar\alpha_t = 0.99$, $\mathrm{SNR} = 0.99/0.01 = 99$. At
> $\bar\alpha_t = 0.5$, $\mathrm{SNR} = 1$. At $\bar\alpha_t = 0.01$,
> $\mathrm{SNR} = 0.01/0.99 = 0.0101$. **The range spans four orders of magnitude.**

$$\mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}, \qquad \mathcal{L} = \mathbb{E}_t\big[\min(\mathrm{SNR}(t), \gamma)\,\|\hat\epsilon - \epsilon\|^2\big]$$

> 💡 **The problem Min-SNR fixes, and why it's a real problem.** Different timesteps teach different
> things. **Low-noise steps** ($t$ near 0) teach fine texture and sharpness. **High-noise steps**
> ($t$ near $T$) teach global composition — where objects go, what the scene is.
>
> Under the natural weighting, the easy high-SNR steps produce far larger gradients and **dominate
> training**, so the model over-invests in polishing texture and under-invests in composition. This
> is a **multi-task balancing problem** in disguise.
>
> **Min-SNR-$\gamma$ caps the weight at $\gamma$** (typically 5), so no timestep can contribute more
> than that. Cheap to implement, and it measurably improves both convergence speed and sample
> quality.

### Part C — the schedule is a curriculum

> **The schedule $\beta_t$ is a curriculum.**

> **Noise schedule** — the sequence of $\beta_t$ values, which determines **how fast the signal is
> destroyed** and therefore how the model's training time is distributed across noise levels.

The slide's chart plots $\bar\alpha_t$ (signal retained) against diffusion time $t/T$, with three
curves:

| Curve | Shape | Consequence |
|---|---|---|
| **Linear** (orange) | Drops fast and early | **"Destroys low-res structure too fast"** — most timesteps are spent in near-pure noise, where little is learnable |
| **Cosine** (teal) | Gentle at both ends, steep in the middle | **"Spends more steps at useful noise levels and lifts sample quality at no extra cost"** (Nichol & Dhariwal, ICML 2021 — see "Going deeper" #22) |
| **$\sqrt{\bar\alpha}$ (signal)** (purple) | Reference curve | Shows the actual signal coefficient |

The cosine schedule's definition:

$$\bar\alpha_t = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $t/T$ | "t over T" | Fractional progress, 0 → 1. |
| $s$ | "s" | A small offset (≈0.008) preventing $\bar\alpha$ from being exactly 1 at $t = 0$. |
| $\cos^2(\cdot)$ | "cosine squared" | Runs smoothly from 1 down to 0 over a quarter period. |

> 💡 **"A curriculum" is exactly the right word.** The schedule decides **which noise levels the
> model spends its training budget on**. A linear schedule wastes many steps in the regime where
> the image is already destroyed and there's nothing to learn. The cosine schedule redistributes
> that budget to the middle range where structure is actually being formed. **Same model, same
> compute, better images — purely from how you allocate training time across difficulty.**

### Part D — the zero-terminal-SNR bug

This is the slide's most concrete practical finding, and it explains a bug you may have seen without
knowing its name.

> **Common schedules leave $\bar\alpha_T \neq 0$, so at $t = T$ a faint ghost of $x_0$ survives, yet
> sampling *starts* from pure noise. Rescaling $\beta_t$ so $\mathrm{SNR}(T) = 0$ (and switching to
> v-pred) closes this train/test gap and fixes the "can't make a truly dark image" bug.**

**The mechanism, step by step:**

> ⚠️ **verify this — not stated on the slide.** The slide (slide_016.jpg) shows only a qualitative
> chart with the endpoint labelled "SNR(T)=0 ✓" for the *corrected* schedule; it does not print a
> numeric value for the standard schedule's $\bar\alpha_T$. The **0.004** figure below is this
> document's own illustrative example, chosen to be a plausible, small-but-nonzero value
> demonstrating the mechanism — not a number read off the slide.

```
TRAINING: at t = T, the standard schedule leaves ᾱ_T ≈ 0.004 (illustrative — see caveat above), not 0.
          So  x_T = √0.004 x_0 + √0.996 ε
                  = 0.063 x_0 + 0.998 ε

          A 6% ghost of the real image survives. In particular, its
          MEAN BRIGHTNESS leaks through.

          The model learns: "at t = T, the overall brightness of the
          output is already hinted at in the input."

SAMPLING: we start from x_T ~ N(0, I) — PURE noise, mean exactly 0.

          There is no ghost. No brightness hint. But the model was
          trained to expect one, and a mean-zero input reads to it as
          "medium brightness".

RESULT:   The model can never produce a very dark or very bright image.
          Ask for "a photo of a black cat in a dark room" and you get
          a mid-grey room.
```

> 💡 **This is a textbook train/test mismatch, and it went unnoticed for years.** The model was
> trained on inputs that always carried a faint brightness signal and is sampled from inputs that
> never do. Nobody noticed because the images still looked fine — just never *very* dark or *very*
> bright, which is easy to mistake for a stylistic tendency rather than a bug.
>
> **The fix is a one-line schedule rescale**: force $\bar\alpha_T = 0$ exactly, so training and
> sampling see the same thing. Note the slide's parenthetical — **you must also switch to v-pred**,
> because at exactly $\mathrm{SNR} = 0$, $\epsilon$-prediction becomes degenerate (Part A's
> reasoning taken to its limit: with zero signal, there's nothing to condition the noise estimate
> on).

### 🎯 Interview questions

- *Why does v-prediction matter for distillation?* → Distilled models take very large steps,
  including steps that span the extreme ends of the schedule. $\epsilon$-pred is ill-conditioned
  near $t=0$ and $x_0$-pred near $t=T$; $v$-pred stays well-scaled across all $t$, so the
  regression target doesn't degenerate when the step size grows.
- *Your text-to-image model can't produce a truly dark image. Diagnose.* → Zero-terminal-SNR. The
  schedule leaves $\bar\alpha_T \neq 0$, so training inputs at $t=T$ retain a faint brightness cue
  that sampling from pure noise never provides. Rescale $\beta_t$ so $\mathrm{SNR}(T) = 0$ and
  train with v-prediction.

---

## 9. Classifier-free guidance: how hard to obey the prompt

*(Slide 17)*

> **One knob trades faithfulness to the text against diversity of the output.**

### The problem

> **Plain conditional diffusion follows a prompt only weakly. We want a dial for "take the prompt
> more seriously".**

Train a diffusion model on (image, caption) pairs and it *does* learn to use the caption — but
loosely. Ask for "a red fox in snow" and you might get a fox in a field, or a vaguely reddish
animal. The prompt is one influence among many, not a command.

### The idea

> **At each step run the model **twice** (once with the prompt, once without) and push **away** from
> the unconditioned prediction.**

**In words: compute what the model would denoise toward with no prompt at all, and what it denoises
toward with the prompt. Take the difference — that's the direction the prompt is pulling — and
exaggerate it.**

$$\hat\epsilon = \epsilon_\varnothing + w\,(\epsilon_{\text{text}} - \epsilon_\varnothing)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\hat\epsilon$ | "epsilon hat" | The **guided** noise prediction actually used for this step. |
| $\epsilon_\varnothing$ | "epsilon empty-set" | The **unconditional** prediction — the model run with an empty prompt. |
| $\epsilon_{\text{text}}$ | "epsilon text" | The **conditional** prediction — run with the real prompt. |
| $(\epsilon_{\text{text}} - \epsilon_\varnothing)$ | "the difference" | **The direction the prompt pulls in.** |
| $w$ | "w", the **guidance scale** | How far to amplify that direction. |

**Check the boundary cases — they make the formula obvious:**

```
w = 0:   ε̂ = ε_∅ + 0 × (…) = ε_∅         → prompt entirely IGNORED
w = 1:   ε̂ = ε_∅ + 1 × (ε_text - ε_∅) = ε_text   → ordinary conditional generation
w = 7:   ε̂ = ε_∅ + 7 × (ε_text - ε_∅)     → the prompt's pull, EXAGGERATED 7×
```

> 💡 **Notice that $w > 1$ extrapolates rather than interpolates.** You are not blending between
> unconditional and conditional — you are stepping *past* the conditional prediction, further in the
> direction the prompt indicated. **The model is pushed toward images that are more
> characteristically "red fox in snow" than the model's own honest estimate would be.** That's why
> it works, and also why it eventually breaks.

**Worked example, one pixel:**

```
ε_∅    = 0.30        (unconditional prediction)
ε_text = 0.42        (with the prompt)

difference = 0.42 - 0.30 = 0.12

w = 1:   ε̂ = 0.30 + 1 × 0.12 = 0.42       (ordinary conditional)
w = 7.5: ε̂ = 0.30 + 7.5 × 0.12 = 1.20     (strongly guided)
w = 15:  ε̂ = 0.30 + 15 × 0.12 = 2.10      (over-driven)
```

Watch the guided value leave the plausible range for a noise prediction. **That numerical
overshoot is exactly what the over-saturation artefacts look like in the image.**

### The trade-off

> **Higher $w$: tighter prompt adherence, but **less diverse** and eventually over-saturated,
> artefact-y images.**

The slide's chart plots two curves against guidance scale $w$ (0 to 15):

| $w$ | Prompt adherence (teal) | Diversity (orange) |
|---|---|---|
| **0** | ~20% | **~95%** |
| **2** | ~55% | ~78% |
| **4** | ~72% | ~62% |
| **6–10** | **~85–95%** ← *sweet spot band* | ~30% → ~5% |
| **15** | ~97% | ~5% |

The two curves **cross around $w \approx 3$**, and the slide shades a **sweet spot** band roughly
from $w = 6$ to $w = 10$.

> **Standard today; the sweet spot is usually $w \approx 7$–$8$.**

> 💡 **Why diversity collapses — the intuition.** Amplifying the prompt direction pushes every
> sample toward the *most prototypically* "red fox in snow" image. Generate a hundred samples at
> $w = 15$ and they look like a hundred slight variations of one image: same pose, same
> composition, same lighting. **Guidance concentrates probability mass on the modes.** You get
> exactly what you asked for, and only that.
>
> This is the same tension as lecture 3's decoding section — greedy decoding versus sampling.
> **Faithfulness and diversity trade off in both text and images, and for the same underlying
> reason: sharpening a distribution removes its tails.**

```interactive
type: slider
title: Guidance scale w
concept: Classifier-free guidance trades prompt faithfulness against output diversity
control: Drag the guidance scale w from 0 to 14 (the deck's own slider)
observe: Two curves move oppositely as w rises — "prompt adherence" climbs from ~20% toward ~97%, "diversity" falls from ~95% toward ~5% — crossing around w≈3, with a shaded "sweet spot" band roughly w=6–10
insight: The two curves are not independent measurements — they are the same underlying effect (extrapolating past the conditional prediction) read two ways, so pushing w past the sweet spot buys adherence you can already see and pays for it in diversity you can already see disappearing
fallback: The table and chart description already in this section (adherence and diversity at w = 0, 2, 4, 6–10, 15, with the crossover at w≈3 and the sweet-spot band at w=6–10) gives the same two curves this slider would let you scrub continuously.
```

### The cost

> **Samplers (DDIM, DPM++) then decide how many steps it takes.**

Note what CFG costs: **every denoising step now requires two network passes** instead of one — once
with the prompt, once without. **CFG doubles inference compute.** (In practice implementations batch
the two passes together, so it's ~2× the FLOPs but not 2× the wall-clock latency on a GPU with
spare capacity.)

That doubling is a large part of *why* the step-count reductions in the next two sections mattered
so much: 50 steps × 2 passes = 100 network evaluations per image.

### Where people get confused

**You might think** CFG requires a separate unconditional model. **Actually** it's the *same*
model, run with an empty prompt. During training, the caption is randomly dropped (typically 10% of
the time) so the model learns both the conditional and unconditional behaviours in one set of
weights. That trick — training one model to do both — is what "classifier-**free**" refers to.

**You might think** higher guidance is better prompt-following, full stop. **Actually** past the
sweet spot, images become over-saturated with blown-out colours and hard edges, and detail degrades.
The *adherence* metric keeps rising while perceived quality falls — which is lecture 2's Goodhart
problem in a new setting.

**You might think** CFG is only for text. **Actually** it applies to any conditioning signal —
class labels, reference images, depth maps. Anything you can drop during training can be guided
against at sampling time.

---

## 10. Fewer steps, same image: better samplers

*(Slide 18)*

> **The original recipe needed ~1000 denoising passes; smarter solvers reach quality in ~20.**

### The cost

> **Each denoising step is a full network pass. DDPM's ~1000 steps make generation slow.**

Do the arithmetic. At ~1000 steps, with CFG doubling each one, generating a single image is
**~2000 forward passes** of a large network. At even 50 ms each, that's **100 seconds per image**.
Diffusion was beautiful and unusably slow.

### DDIM

> **DDIM: A **deterministic** reformulation: same trained model, but you can skip steps and still
> land on a good image, in ~50 steps.**

> **DDIM (Denoising Diffusion Implicit Models)** — a reformulation of the reverse process that
> removes the random noise injected at each step, making the trajectory **deterministic** — and
> therefore skippable.
>
> *In everyday words:* DDPM's reverse process is a **random walk** that happens to drift toward an
> image; you must take small steps or you lose the path. DDIM is a **smooth curve** from noise to
> image; you can jump along it.
>
> *The crucial detail:* **it uses the same trained network.** No retraining. You change only the
> update rule at sampling time.

> 💡 **Two consequences of determinism worth knowing.** First, **skippability** — a smooth
> trajectory can be traversed in 50 jumps instead of 1000 small steps. Second, **invertibility** —
> because the path is deterministic, you can run it *backwards*, from a real image to the noise that
> would produce it. That's **DDIM inversion**, and it's what makes precise real-image editing
> possible (section 19).

### DPM++ and the ODE view

> **DPM++ and friends: Treat denoising as solving an **ODE** and use a higher-order solver. Quality
> in ~20 steps.**

From Prerequisite 5: an ODE describes continuous change, and you solve it by stepping along it.
Once DDIM revealed the reverse process is a smooth deterministic trajectory, the whole toolbox of
**numerical ODE solvers** — a mature field, centuries old — became available.

> 📚 **Background — what "higher-order" means, concretely.**
> **First-order (Euler):** look at the slope where you are, step straight along it. Simple; with big
> steps you drift off a curved path.
> **Higher-order:** evaluate the slope at several points and combine them to follow the curve's
> *bend*, not just its direction. Much more accurate per step, so you can take **far bigger steps**
> for the same error.
>
> *In everyday words:* Euler is steering a car by pointing at where the road heads right now.
> Higher-order is looking ahead to the corner and steering into it.

### The chart

Sample quality against denoising steps (log axis), three samplers:

| Steps | **DDPM** (grey) | **DDIM** (orange) | **DPM++** (teal) |
|---|---|---|---|
| 4 | ~1% | ~13% | ~36% |
| 10 | ~5% | ~45% | ~78% |
| 20 | ~10% | ~72% | **~93%** ← crosses "good enough" |
| 50 | ~28% | **~93%** ← crosses | ~97% |
| 100 | ~50% | ~97% | ~97% |
| 1000 | **~92%** ← crosses | ~97% | ~97% |

The dashed **"good enough"** line sits at ~90%.

**Steps to reach "good enough": DDPM ~1000 → DDIM ~50 → DPM++ ~20.** A **50× reduction** in
generation cost.

### The lesson

> **Training defines the destination; the sampler decides how few steps it takes to get there. Pure
> inference-time speed-up.**

> 💡 **This is the sentence to remember from the whole section.** The trained network is unchanged.
> The images it can produce are unchanged. **Only the route through the space changed.** A 50×
> speedup with zero retraining, zero quality loss, and no new data — obtained purely by recognising
> that a stochastic process could be rewritten as a smooth one and then applying 200-year-old
> numerical analysis.
>
> Compare this to lecture 3's FlashAttention and speculative decoding: the same pattern of a **free
> win from re-examining how a computation is performed** rather than what it computes.

---

## 11. From ~20 steps to one: distillation and consistency models

*(Slide 19)*

> **Better solvers shrink the step count; training a student to take giant steps collapses it to a
> handful.**

### Why solvers alone stall

> **DDIM/DPM++ integrate the same probability-flow ODE; below ~10 steps the discretisation error of
> *any* solver shows up as blur. To go lower you must change what the network **computes**, not
> just how you step it.**

> 📚 **Background — discretisation error.** Any ODE solver approximates a smooth curve with straight
> segments. Fewer, longer segments = worse approximation. Below ~10 steps you're approximating a
> strongly curved trajectory with a handful of straight lines, and no cleverness in the stepping
> rule fixes that. **The error is in the discretisation itself, and it shows up visually as blur.**

So to go lower, you must **retrain**. Three approaches.

### Progressive distillation

> **Train a **student** to reproduce in *one* step what the teacher does in *two*, then halve again.
> Each round doubles the step size: $N \to N/2 \to \ldots$. v-prediction is what keeps the target
> well-scaled as the steps grow huge.**

```
Round 0:  teacher = 1024 steps
Round 1:  student learns to do in 512 what teacher does in 1024
Round 2:  that student becomes the teacher; next student does 256
Round 3:  → 128
...
Round 8:  → 4 steps
```

*In everyday words:* teaching someone to take two stairs at a time, then four, then a whole flight.

> 💡 **Note the callback to section 8.** The slide explicitly says v-prediction is what makes this
> work — because a student taking one giant step must handle a trajectory spanning very different
> noise levels, exactly where $\epsilon$-pred and $x_0$-pred become ill-conditioned. **Part A of
> section 8 was not an academic detail; it's load-bearing for the fastest models in production.**

### Consistency models

> **Learn a function $f_\theta(x_t, t) \approx x_0$ that is **self-consistent** along the whole ODE
> trajectory: any point on a path maps to the same origin, $f_\theta(x_t, t) = f_\theta(x_{t'}, t')$.
> One forward pass jumps from noise to image; 2 to 4 steps refine. **LCM** applies this in latent
> space.**

$$f_\theta(x_t, t) = f_\theta(x_{t'}, t') \quad \forall\, t, t'$$

| Symbol | Read it as | What it means |
|---|---|---|
| $f_\theta(x_t, t)$ | "f theta of x-t and t" | The network's estimate of the **clean image** from any point on the trajectory. |
| $\forall\, t, t'$ | "for all t and t-prime" | **Any two points** on the same trajectory must give the same answer. |

*In everyday words:* every point on a river must know where the river's mouth is. Drop a leaf
anywhere along it and ask "where does this end up?" — every position must give the same answer.

> 💡 **Why this is a genuinely different training target.** A diffusion model learns *"which way is
> slightly downhill from here"* — a **local** question, answered one small step at a time. A
> consistency model learns *"where does this trajectory end"* — a **global** question, answered in
> one shot.
>
> That's why it can generate in one pass: it was never trained to take steps. **The self-consistency
> constraint is what makes it learnable** — you can't supervise "where does this end" directly for
> every noise level, but you *can* enforce that neighbouring points agree, and agreement propagates
> along the whole trajectory.

**LCM (Latent Consistency Models)** applies this inside the latent space of section 12, combining
both speedups.

### Adversarial distillation

> **SDXL-Turbo (ADD) and SD3-Turbo (LADD) add a GAN discriminator to the distillation loss,
> recovering sharpness lost by pure regression, for near-teacher quality in **1 to 4 steps**.
> SDXL-Lightning blends progressive + adversarial. This is what powers real-time, on-device
> generation.**

$$\mathcal{L}_{\text{ADD}} = \mathcal{L}_{\text{distill}} + \lambda\,\mathcal{L}_{\text{adv}}$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\mathcal{L}_{\text{distill}}$ | "distillation loss" | Match the teacher's output. Regression. |
| $\mathcal{L}_{\text{adv}}$ | "adversarial loss" | A **discriminator** network judges whether the output looks like a real image. |
| $\lambda$ | "lambda" | How much weight to give the adversarial term. |

> 📚 **Background — why pure regression produces blur, and why a GAN fixes it.**
> MSE regression, when uncertain, predicts the **average** of all plausible answers. Averaging many
> plausible sharp images gives a **blurry** image — averaging is what blur *is*. This is a
> fundamental property of squared-error losses, not a training failure.
>
> A **discriminator** doesn't ask "is this close to the target?" — it asks **"does this look real?"**
> A blurry average is instantly identifiable as fake, so the adversarial term actively penalises
> exactly the failure regression produces. **The two losses are complementary: regression keeps the
> student faithful to the teacher, the discriminator keeps it sharp.**

### The chart and the honest trade-off

Sample quality against network passes:

| Passes | **Solver** (DDIM/DPM++) | **Distilled / consistency** |
|---|---|---|
| 1 | ~11% | **~78%** |
| 2 | ~18% | ~88% |
| 4 | ~30% | **~93%** ← past "good enough" |
| 10 | ~72% | ~95% |
| 20 | ~88% | ~96% |
| 1000 | ~97% | ~96% |

The slide's closing box states the trade-off honestly:

> **Solvers are a free inference trick; distillation is a *second training stage* that trades a
> little quality and generality for a 10 to 50× speed-up.**

> 💡 **Read the table's right-hand end carefully — it's where the cost shows up.** At high step
> counts the solver reaches ~97% while the distilled model plateaus at ~96%. **Distillation gives up
> the top end.** It can't be pushed further by spending more compute, because it was trained to
> take giant steps.
>
> And "generality" is the subtler cost: distilled models are often less responsive to guidance
> scale, less good at unusual prompts, and less controllable. **You have traded flexibility for
> speed.** For real-time and on-device generation that's exactly the right trade; for a
> professional tool where quality is paramount, it may not be.

### The full progression

```
DDPM (2020)                      ~1000 steps    — the original
   ↓  reformulate as deterministic ODE
DDIM (2020)                        ~50 steps    — free, no retraining
   ↓  higher-order ODE solvers
DPM++ (2022)                       ~20 steps    — free, no retraining
   ↓  ═══ retraining required from here ═══
Progressive distillation            ~4 steps    — second training stage
Consistency models / LCM           2-4 steps    — different training target
Adversarial distillation (ADD)      1-4 steps   — + GAN discriminator

                     ~1000 → 1.   A 1000× reduction.
```

### 🎯 Interview question

*Why can't a better sampler get below ~10 steps?* → Below that, the discretisation error of *any*
ODE solver dominates — you're approximating a strongly curved trajectory with a few straight
segments, and the error appears as blur. Samplers only change *how you step along* a fixed
trajectory; to go lower you must change *what the network computes*, which requires retraining
(distillation or a consistency objective).

---

## 12. Diffuse in a small latent, not in pixels

*(Slide 20)*

> **The single change that made high-res generation cheap enough to run anywhere.**

### The problem

> **Pixel-space diffusion on a 512×512 image denoises ~790k numbers every step: far too expensive at
> high resolution.**

```
512 × 512 × 3 (RGB) = 786,432 numbers        ✓ matches the slide's "786,432 dims"

At 50 denoising steps, each a full network pass over 786k values.
And at 1024×1024, it quadruples to 3.1 million.
```

### The idea

> **Use a **VAE** to compress the image into a small latent (64×64×4), run the **entire diffusion**
> there, then decode back to pixels at the end.**

> 📚 **Background the slide assumed — what a VAE is.**
> A **VAE (Variational Autoencoder)** is a pair of networks:
> - an **encoder** that compresses an image into a small vector representation (the **latent**),
> - a **decoder** that reconstructs the image from that latent.
>
> Trained together so that decode(encode(image)) ≈ image.
>
> *In everyday words:* a very good, learned image compressor — like JPEG, but the compression scheme
> was learned from data rather than designed by hand.
>
> *Why it works here:* natural images are **enormously redundant**. Neighbouring pixels are almost
> always similar; most of the 786,432 numbers are predictable from their neighbours. The VAE strips
> that redundancy and keeps what's semantically meaningful.

### The saving

> **The latent keeps the **semantics** and throws away pixel-level redundancy, so the denoiser works
> in a space **~48× smaller**.**

```
Pixel space:   512 × 512 × 3 = 786,432 dims
Latent space:   64 ×  64 × 4 =  16,384 dims

Ratio: 786,432 / 16,384 = 48×                  ✓ matches the slide
```

> 💡 **Note where the 4 comes from.** The spatial dimensions shrink 8× each way ($512/64 = 8$), which
> alone would be 64×. But the channel count goes *up*, from 3 (RGB) to 4 — the latent needs a bit
> more depth per position to hold the compressed information. Net effect: 48×, not 64×.

```
Diffuse in a compressed latent, not in pixels

  ┌────────┐   ┌────────┐   ┌────────────────┐   ┌────────┐   ┌────────┐
  │ image  │──►│ VAE enc│──►│   diffusion    │──►│ VAE dec│──►│ image  │
  │ 512²×3 │   │   ↓    │   │   in latent    │   │   ↑    │   │  out   │
  └────────┘   └────────┘   │    64²×4       │   └────────┘   └────────┘
                            └────────────────┘
                            ALL 50 steps here

  pixels  ████████████████████████████████████  786,432 dims
  latent  ▌                                      16,384 dims (~48× smaller)
```

### The payoff

> **This is Stable Diffusion. Same algorithm, a fraction of the compute: image generation on a
> consumer GPU.**

> 💡 **Sit with why this was the decisive change.** Latent diffusion did not improve the
> *algorithm* at all — the diffusion mathematics is identical to section 7. It changed **where** the
> algorithm runs. But that 48× reduction is exactly what moved image generation from
> "a datacenter capability" to "runs on a gaming GPU in your bedroom" — and that shift is why the
> open image-generation ecosystem exists at all.
>
> **Combine it with section 11's step reduction and the total is remarkable:** ~1000 steps in pixel
> space → ~4 steps in latent space is roughly a **12,000×** reduction in generation compute, over
> about three years, with quality improving throughout.

### Why the VAE doesn't ruin quality

The natural worry: doesn't compressing 48× destroy detail?

Two answers. First, **the VAE is trained specifically to reconstruct images well** — its whole
objective is that decode(encode(x)) ≈ x, so it learns which information is worth keeping. Second,
what it discards is largely **pixel-level redundancy** — precise noise textures, exact values of
predictable pixels — not semantics.

> ⚠️ **That said, the VAE does impose a real ceiling, and it's visible in practice.** Latent
> diffusion cannot produce detail finer than the VAE can represent. This is a known cause of
> artefacts in **small text**, **faces at small scale**, and **fine repeating textures** — the
> classic "mangled text in AI images" problem is partly a VAE limitation, not only a model one. Each
> Stable Diffusion generation shipped an improved VAE partly for this reason.

---

## 13. The full text-to-image architecture, end to end

*(Slide 14)*

> **Three trained components, but the diffusion loop is the only one doing the generating.**

Now that latents (section 12) and guidance (section 9) are defined, here is the complete system.

```
   "a red fox in snow"
        (prompt)
           │
           ▼
  ┌──────────────────┐
  │  Text encoder    │   CLIP / T5.  FROZEN.
  │  CLIP / T5       │   Turns words into vectors.
  └────────┬─────────┘
           │ text embeddings
           │                    ┌── cross-attention ──┐
           │                    │                     │
           ▼                    ▼                     │
  ╔════════════════════════════════════════════════╗  │
  ║  LATENT SPACE · 64 × 64 × 4                    ║  │
  ║                                                ║  │
  ║   ┌────────┐    ┌─────────────┐   ┌────────┐  ║  │
  ║   │ noise  │───►│  Denoiser   │──►│ clean  │  ║  │
  ║   │ z_T ~  │    │ U-Net / DiT │   │  z_0   │  ║  │
  ║   │ N(0,I) │    │ predicts ε  │   └────┬───┘  ║  │
  ║   └────────┘    └──────┬──────┘        │      ║  │
  ║                        │               │      ║  │
  ║                        └───────────────┘      ║  │
  ║                     repeat T steps (= 20–50)  ║  │
  ╚════════════════════════════╤═══════════════════╝  │
                               │                      │
                               ▼                      │
                      ┌────────────────┐              │
                      │  VAE decoder   │  FROZEN.     │
                      │       ↑        │              │
                      └────────┬───────┘              │
                               ▼                      │
                    generated image · 512 × 512 × 3   │
                                                      │
   Training learns ONLY the denoiser; ────────────────┘
   the text encoder and VAE are pretrained and frozen.
```

### The three components

| Component | What it does | Trained during diffusion training? |
|---|---|---|
| **Text encoder** (CLIP / T5) | Turns the prompt into a sequence of embeddings | ❌ **Frozen** |
| **Denoiser** (U-Net or DiT) | Predicts the noise, 20–50 times | ✅ **The only part trained** |
| **VAE** (encoder + decoder) | Pixels ↔ latent | ❌ **Frozen** |

> **Text encoder (frozen):** CLIP or T5 turns the prompt into a sequence of embeddings. It is
> **pretrained and never updated** during diffusion training.
>
> **The latent loop (learned):** Start from pure noise $z_T$. The denoiser predicts the noise, we
> subtract a step, and **repeat 20–50 times**, all inside the small latent space.
>
> **Cross-attention is the bridge:** At every step the denoiser **attends to the text embeddings**,
> so the words steer each denoising move.
>
> **VAE decoder (frozen):** Once the latent is clean, a single decoder pass blows it back up to a
> full-resolution image.

> 💡 **Two things worth noticing about this diagram.**
>
> **(1) Only one component is trained.** The text encoder already knows language and how it relates
> to images (that's CLIP, section 2). The VAE already knows how to compress and reconstruct images.
> Diffusion training learns *only* the noise predictor. **This is the same modular reuse pattern as
> LLaVA in section 3** — freeze what already works, train the bridge.
>
> **(2) The text encoder runs ONCE; the denoiser runs 20–50 times.** The prompt is encoded a single
> time and those embeddings are reused at every step. **Essentially all the inference cost is the
> denoising loop** — which is why sections 10 and 11 (cutting steps) and 12 (shrinking the space)
> were where all the optimisation effort went.

### Cross-attention as the bridge

From lecture 1: cross-attention lets one sequence attend to a *different* sequence. Here:

```
Queries  ← the image latent patches      ("what am I looking at?")
Keys     ← the text embeddings           ("what does the prompt offer?")
Values   ← the text embeddings           ("what information to take")
```

Each patch of the latent asks the prompt what it should become. A patch in the region that will hold
the fox attends strongly to the `fox` token; a background patch attends to `snow`. **That is
mechanically how words steer pixels**, and section 17 develops it.

---

## 14. The U-Net backbone

*(Slide 21)*

> **The original biomedical-segmentation architecture, repurposed as the DDPM / Stable Diffusion
> denoiser.**

The denoiser has to be *some* network. The original choice was the **U-Net** — an architecture
designed in 2015 for segmenting biomedical images, which turned out to be almost perfectly shaped
for denoising.

### The shape

```
input                                                            output
image ──┐                                                    ┌── predicted
tile    │  64   64                              64   64   2  │    noise
        ▼  ██   ██ ──────────── skip ──────────► ██   ██   █ ▲
           │                                          ▲
           ▼ max pool 2×2                             │ up-conv 2×2
          128  128 ─────────── skip ──────────► 128  128
           │                                          ▲
           ▼                                          │
          256  256 ─────────── skip ──────────► 256  256
           │                                          ▲
           ▼                                          │
          512  512 ─────────── skip ──────────► 512  512
           │                                          ▲
           ▼                                          │
              1024   1024        (bottleneck)
       contracting path            expanding path
       (encoder, ↓ size ↑ channels)   (decoder, ↑ size)
```

> **Contracting path (left):** repeated $3\times3$ convs + downsampling. Spatial size halves at each
> level while channel count doubles ($64 \to 128 \to 256 \to 512 \ldots$); features get **coarser but
> richer**.
>
> **Expanding path (right):** up-convolutions restore resolution. At each level the matching encoder
> feature map is **concatenated** before more convs.

> 📚 **Background — convolutions and the coarse/rich trade.**
> A **convolution** slides a small learned filter across the image, computing a weighted sum at each
> position. Early layers detect edges; deeper layers, operating on downsampled maps, detect larger
> structures.
>
> **Downsampling** halves the spatial resolution, so each remaining position "sees" a larger region
> of the original image. **Doubling the channels** compensates: fewer positions, but more information
> per position. That's what "coarser but richer" means — you trade *where* for *what*.

### Skip connections are the point

> **They carry high-frequency spatial detail straight across, so the decoder is not forced to
> reconstruct edges from the bottleneck alone.**

> 💡 **Why the U-Net is *especially* right for denoising, more than for its original task.**
> Consider what the two paths do. The **bottleneck** has crushed a 512×512 image down to a small
> spatial map — it knows *"this is a fox in snow"* but has lost *"this exact edge is here, at this
> pixel"*. Yet a denoiser must output a noise prediction **at full resolution, pixel-aligned with
> its input**.
>
> Without skips, the decoder would have to hallucinate every edge position back from a semantic
> summary — and would produce a blurry, misaligned mess. **The skip connections hand the decoder the
> exact high-frequency detail from the corresponding encoder level**, so it only has to decide *what
> to do* with structure it can already see precisely.
>
> This is the same principle as lecture 1's residual connections — give information a direct path so
> it doesn't have to survive a long transformation — applied spatially instead of by depth.

### The two diffusion-specific additions

> **As a denoiser: diffusion adds two things the 2015 design lacked — a **timestep embedding** added
> in every block, and **self-attention** at the low-resolution levels for global coherence.**

**Timestep embedding.** Section 5 established the network must know $t$. It's encoded as a vector
(using the same sinusoidal scheme as positional encodings in lecture 1) and **injected into every
block**, so every layer knows how noisy the input is.

**Self-attention at low resolution.** Convolutions are **local** — a $3\times3$ filter sees nine
neighbouring pixels. But image coherence is global: if the left side of the image decides on evening
light, the right side must agree. Adding self-attention (lecture 1) at the low-resolution levels
lets distant regions communicate.

> 💡 **That second addition is the crack through which transformers entered.** Once you've admitted
> attention into the U-Net for global coherence, the obvious next question is: *why not attention
> everywhere?* That question is section 15.

---

## 15. From U-Net to Diffusion Transformer (DiT)

*(Slides 22–23 — one slide in two toggle states)*

> **Swapping the convolutional backbone for a transformer unlocked the same scaling magic as LLMs.**

### The shift

> **The original:** Stable Diffusion denoises with a **U-Net**: a convolutional encoder-decoder with
> skip connections, hierarchical by design.
>
> **The shift:** **DiT** replaces it with a **transformer over latent patches** — the same
> architecture as an LLM, just predicting noise.

```
backbone                                        quality vs scale

U-Net (conv, hierarchical)                    │              DiT ╱
   ███                                        │             ╱
    ███                                       │          ╱
     █████                                    │       ╱ ─────── U-Net
    ███                                       │    ╱ ──────
   ███                                        │ ╱──
downsample → bottleneck → up                  └─────────────────────
                                                1×    10×    100×
DiT (transformer on patches)                     training compute →
   □□□
   □□□   ↓ patchify
   □□□
   ▭▭▭▭
   ▭▭▭▭   N identical transformer blocks
   ▭▭▭▭
```

### Why it won

> **Transformers **scale more predictably**: add compute and quality keeps climbing, where the U-Net
> plateaus.**

> 💡 **This is lecture 1's scaling-laws lesson, arriving in a second field.** The U-Net is
> *hierarchical by design* — its structure encodes strong assumptions about images (locality,
> multi-scale organisation). Those assumptions are genuinely helpful at small scale: they're correct
> priors, so the model needs less data to learn.
>
> But at large scale, **built-in assumptions become constraints**. A transformer assumes almost
> nothing — every patch can attend to every other from layer one — so given enough data and compute
> it can learn whatever structure is actually there, including structure the U-Net's design forbids.
>
> **The same story played out in NLP** (hand-designed features → learned features), **in vision**
> (CNNs → ViT), and now **in generation** (U-Net → DiT). The pattern is consistent: *architectural
> priors help when data is scarce and hurt when it is abundant.* Look at the chart — the lines
> diverge only past ~10× compute.

```interactive
type: diagram
title: U-Net versus DiT backbone
concept: Swapping a hierarchical convolutional backbone for a flat transformer changes how quality scales with compute
control: A toggle between "U-Net" and "DiT" (the deck's own two slide states)
observe: The left diagram redraws — U-Net shows a shrinking-then-growing stack of blocks (downsample → bottleneck → up); DiT shows a uniform column of identical patch/transformer blocks. The right-hand quality-vs-scale chart is shared by both, with the U-Net and DiT curves nearly overlapping at 1× compute and visibly diverging by 100×
insight: The backbone toggle changes the left diagram completely but the right-hand chart is the same chart in both states — the two curves on it are what actually decide which backbone wins, and they only disagree once you can afford enough compute to see the difference
fallback: The two backbone diagrams and the single quality-vs-scale chart already described in this section (both curves starting together near 1× and the DiT curve pulling ahead past ~10×) are the exact two toggle states and shared chart this control would present.
```

### In the wild

> **Stable Diffusion 3, Flux, and Sora are all DiT-based. The image and text worlds converged on one
> backbone.**

> 💡 **Take that last sentence seriously — it's the closing argument of the whole lecture series.**
> Lecture 1 opened with a transformer predicting the next token. Four lectures later, image and
> video generation — a completely different mathematical framework, a completely different training
> objective — runs on **the same architecture**. Not a similar one. The same transformer blocks,
> the same attention, the same scaling behaviour.
>
> Section 1 claimed a transformer doesn't care what a token means. **DiT is that claim taken to its
> conclusion**: it doesn't care whether the task is predicting a word or predicting noise either.

---

## 16. Inside the denoiser: the DiT block

*(Slide 24)*

> **Replace the convolutions with a transformer over latent patches, and inherit LLM scaling.**

```
Inside the denoiser: the Diffusion Transformer (DiT)

  ┌──────────────┐
  │ noisy latent │  64 × 64 × 4
  │   □□□        │
  │   □□□        │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ patchify+pos │
  └──────┬───────┘
         ▼
     ▭ ▭ ▭ ▭   patch tokens
         │
         ▼
  ╔══════════════════════════════════╗       ┌──────────────┐
  ║        DiT block × N             ║       │  timestep t  │
  ║  ┌────────────────────────┐      ║ ◄─────│  + class/    │
  ║  │ adaLN-Zero (scale,shift)│     ║       │    text c    │
  ║  └───────────┬────────────┘      ║       └──────────────┘
  ║              ▼                    ║   adaLN-Zero injects t and c
  ║  ┌────────────────────────┐      ║   by modulating each block
  ║  │ Multi-Head Self-Attn   │      ║
  ║  └───────────┬────────────┘      ║
  ║              ⊕ ◄── residual      ║
  ║              ▼                    ║
  ║  ┌────────────────────────┐      ║
  ║  │ adaLN-Zero              │      ║
  ║  └───────────┬────────────┘      ║
  ║              ▼                    ║
  ║  ┌────────────────────────┐      ║
  ║  │ MLP (feed-forward)     │      ║
  ║  └───────────┬────────────┘      ║
  ║              ⊕ ◄── residual      ║
  ╚══════════════╤═══════════════════╝
                 ▼
        ┌─────────────────┐
        │linear+unpatchify│
        └────────┬────────┘
                 ▼
        predicted noise ε (per patch)
```

### The four steps

> **Patchify:** Cut the latent into a grid of patches, flatten each to a token, add positional
> embeddings. **The image is now a sequence, exactly like text.**

This is section 1's core idea, applied to a *latent* rather than an image. A 64×64×4 latent with
2×2 patches gives $(64/2)^2 = 1024$ tokens.

> **N identical blocks:** Each is a standard transformer block: **multi-head self-attention + MLP**,
> with residual connections. **No convolutions, no hierarchy.**

Compare to the U-Net: no downsampling, no upsampling, no skip connections between levels, no
bottleneck. **Just N identical blocks, exactly like GPT.** The architecture is *flat*.

> **adaLN-Zero conditioning:** Timestep $t$ and class/text $c$ are injected by **modulating the
> LayerNorm** (predicting per-block scale & shift), initialised to zero for stable training.

> 📚 **Background — adaptive LayerNorm (adaLN), from scratch.**
> Ordinary LayerNorm (lecture 1) normalises a vector, then applies a **learned scale and shift**:
> `output = γ · normalise(x) + β`, where γ and β are fixed parameters.
>
> **Adaptive** LayerNorm makes γ and β **depend on the conditioning**. A small network takes the
> timestep embedding and the text embedding and *outputs* γ and β for this block:
> ```
> γ, β = MLP(timestep_embedding + text_embedding)
> output = γ · normalise(x) + β
> ```
> **So the conditioning doesn't enter as extra tokens — it reshapes how every block processes its
> input.** Different $t$ → different γ and β → the same weights behave differently at different
> noise levels.
>
> **And "-Zero"?** The MLP producing γ and β is **initialised so its output makes the block an
> identity function** at step one. The block starts by doing nothing and learns to contribute
> gradually.

> 💡 **You have seen the "-Zero" trick twice now, and you'll see it again in section 18.** LoRA
> initialises $B$ to zero so the adapter starts as identity (lecture 3). adaLN-Zero initialises the
> modulation so each block starts as identity. ControlNet's zero-convolutions do the same. **It is
> a recurring, powerful design pattern: start a new component as a no-op, so adding it cannot damage
> what already works, and let gradient descent grow its contribution from zero.**

> **Unpatchify:** A final linear layer maps tokens back to per-patch noise, reassembled into the
> **predicted $\epsilon$**.

### The consequence

> **Because it *is* a transformer, DiT scales like an LLM: add compute and quality keeps climbing.
> This is the backbone of SD3, Flux, and Sora.**

---

## 17. Steering the image, and extending to video

*(Slides 25 and 26)*

> **Text guides through cross-attention; video adds a time axis to the patches.**

### Three signals, three doors

*(Slide 26 — "How conditioning is actually wired in")*

> **Three signals enter the same denoiser through different doors, and they compose.**

```
  "a red fox in snow"
        prompt
          │
          ▼
   ┌──────────────┐                              ┌──────────────┐
   │ ① Text encoder│                             │ ③ IP-Adapter │
   │  CLIP / T5    │──── cross-attn ──┐          │ reference img│
   └──────────────┘                   │          │    → style   │
                                      ▼          └──────┬───────┘
   ┌──────────────┐         ╔══════════════════╗        │ decoupled
   │ ② ControlNet │         ║    Denoiser      ║        │ cross-attn
   │ pose/depth/  │────────►║  ┌────────────┐  ║◄───────┘
   │    edges     │         ║  │ self-attn  │  ║
   │ a trainable  │         ║  ├────────────┤  ║
   │ copy of the  │         ║  │ cross-attn │  ║
   │ encoder,     │         ║  ├────────────┤  ║
   │ added back in│         ║  │ self-attn  │  ║
   └──────────────┘         ╚═════════╤════════╝
                                      ▼
                            conditioned image
```

**Door 1 — Text → cross-attention.**

> **The prompt embeddings are the **keys and values**; the image tokens are the **queries**. Every
> denoising step is pulled toward the words.**

Recall lecture 1's attention: queries ask, keys advertise, values deliver. Here **image patches ask
and text answers.** Each patch queries the prompt to find out what it should become. That single
sentence is the complete mechanical answer to "how do words control pixels".

**Door 2 — Structure → ControlNet.**

> **A **trainable copy** of the encoder takes a pose / depth / edge map and adds its features back
> into the frozen denoiser, pinning **layout**.**

*Concretely:* give it a stick-figure pose and the generated person adopts that exact pose. Give it a
depth map and the scene matches that 3-D layout. Give it a Canny edge map and the composition
follows those lines.

**Door 3 — Style → IP-Adapter.**

> **A reference image is encoded and fed through a **separate (decoupled) cross-attention**, so you
> can borrow a look without a text description.**

*Concretely:* supply a Van Gogh painting as reference and generations acquire that style — without
you having to write "in the style of Van Gogh", and capturing aspects of the look that no text
description could specify.

> 📚 **Why "decoupled" matters.** IP-Adapter adds a *second, separate* cross-attention path for the
> image reference, rather than concatenating the image features into the existing text
> cross-attention. Keeping them separate means the two signals don't compete for the same attention
> budget — you can dial the image influence up or down independently of the text.

### They compose

> **Text sets **content**, ControlNet pins **structure**, IP-Adapter borrows **style** — combine all
> three on one frozen base model.**

> 💡 **This decomposition is why controllable generation actually works in practice.** The three
> signals are largely **orthogonal**: *what is in the picture*, *where things are*, and *what it
> looks like*. Because they enter through different doors, you can specify each independently.
>
> ```
> Text:        "a woman reading a book in a library"     → content
> ControlNet:  a pose skeleton, arms raised              → structure
> IP-Adapter:  a reference photo with warm film grain    → style
> ```
>
> **And all three run on one frozen base model.** You are not training three models or even one —
> ControlNet and IP-Adapter are add-ons to a base that never changes. Same reuse pattern as
> LLaVA (section 3), LoRA (lecture 3), and the frozen text encoder (section 13).

### Extending to video

*(Slide 25)*

> **Video: Treat a clip as **spacetime patches** (patches across space *and* time) and denoise the
> whole volume. This is Sora.**

```
image: 2D patches                video: spacetime patches

  ▭ ▭ ▭ ▭                          ▭▭▭ ╲
  ▭ ▭ ▭ ▭                          ▭▭▭  ╲   patches across
  ▭ ▭ ▭ ▭                          ▭▭▭   ╲  space AND time
  ▭ ▭ ▭ ▭                           ╲▭▭▭  ╲
                                     ╲▭▭▭
  patches over (x, y)                patches over (x, y, t)
```

> 💡 **Notice how little had to change — that's the point.** An image is a 2-D grid of patches. A
> video is a 3-D **volume** of patches: two spatial axes plus time. Patchify the volume instead of
> the plane, and **everything else — the diffusion maths, the DiT blocks, cross-attention, CFG —
> works unchanged.**
>
> This is section 1's thesis at its most striking. Video is not a new kind of model. It's the same
> transformer over a token sequence that happens to have been cut from a 3-D volume. **Sora is a DiT
> on spacetime patches.**

### The hard part

> **The hard part: **Temporal consistency** (objects must persist) and **physics** (plausible
> motion). Temporal attention layers help but it is unsolved.**

> 💡 **Be precise about why this is hard, because it's the honest limitation of video generation
> today.** Nothing in the diffusion objective *requires* that the mug in frame 1 is the same mug in
> frame 60. The model is minimising noise-prediction error, not enforcing object permanence. Long-
> range consistency has to **emerge** from attention over a very long sequence — and attention is
> $O(N^2)$ (lecture 1), while a video has an enormous number of spacetime patches.
>
> The visible symptoms: objects morphing or vanishing, limbs multiplying, water flowing wrongly,
> objects passing through each other. **The slide's word is "unsolved", and that is accurate**, not
> modesty.

### 🔬 Research opportunity

Temporal consistency and physical plausibility in video generation are among the most active open
problems in the field. Approaches worth reading about: explicit 3-D or object-centric
representations, physics-informed losses, and hierarchical generation (keyframes first, then
interpolation). The evaluation side is equally open — we don't have good automatic metrics for
"does this video obey physics", which makes progress hard to measure.

---

## 18. ControlNet, in detail: the zero-convolution trick

*(Slide 27)*

> **Add spatial control to a frozen diffusion model without destroying it on step one.**

This slide is a small masterpiece of engineering design, and the idea generalises well beyond
diffusion.

### The problem

You have a base model that took millions of dollars to train and works beautifully. You want to add
a new input — a pose map — that it has never seen. Naively attaching a new module and training it
means that at step one, that module outputs **random garbage** which flows into a carefully tuned
network. The base model's behaviour is wrecked before learning begins.

### The design

```
   (a) Before                        (b) After
                                              c  (control map)
        x                                     │
        │                          x          ▼
        ▼                          │   ┌──────────────┐
  ┌──────────────┐                 │   │ zero convolution│
  │ neural       │                 │   └──────┬───────┘
  │ network      │                 │          ▼
  │ block        │                 ├────────► ⊕
  └──────┬───────┘                 │          │
         │                         ▼          ▼
         ▼                  ┌──────────────┐ ┌──────────────┐
         y                  │ neural network│ │  trainable   │
                            │ block (locked)│ │     copy     │
                            │      🔒       │ └──────┬───────┘
                            └──────┬───────┘        ▼
                                   │         ┌──────────────┐
                                   │         │zero convolution│
                                   │         └──────┬───────┘
                                   ▼                │
                                   ⊕ ◄──────────────┘
                                   │
                                   ▼
                                   y_c
```

**In words: the frozen block's output is added to the output of a trainable copy, where the copy is
wrapped in two convolutions whose weights start at exactly zero.**

$$y_c = \mathcal{F}(x) + \mathcal{Z}_2\big(\mathcal{F}_{\text{copy}}(x + \mathcal{Z}_1(c))\big), \qquad \mathcal{Z}_1, \mathcal{Z}_2 \text{ init} = 0$$

| Symbol | Read it as | What it means |
|---|---|---|
| $x$ | "x" | The input to this block. |
| $c$ | "c" | The **control map** — pose, depth, or edges. |
| $\mathcal{F}(x)$ | "F of x" | The **frozen** original block's output. Never changes. |
| $\mathcal{F}_{\text{copy}}$ | "F copy" | A **trainable clone** of that block, initialised from the same weights. |
| $\mathcal{Z}_1, \mathcal{Z}_2$ | "Z-one, Z-two" | Two $1\times1$ convolutions with **weights initialised to zero**. |
| $y_c$ | "y sub c" | The controlled output. |

### The four properties

> **Lock the base:** The pretrained denoiser block is **frozen**, all its learned generative ability
> is preserved untouched.

> **Trainable copy:** Clone that block; the copy receives the control map $c$ and learns to use it.
> **Cloning inherits strong features instead of training from scratch.**

> 💡 **Cloning rather than random-initialising is a real design choice, not an incidental one.** The
> copy starts as a network that *already knows how to process image features* — it inherits every
> learned edge detector, texture recogniser, and shape prior from the base. It only has to learn how
> to *incorporate the control map*, not how to see. That's why ControlNet trains in hours rather
> than weeks.

> **Zero convolutions:** Wrap the copy in two $1\times1$ convs whose weights start at **zero**. On
> step one they output 0, so $y_c = y$: the model is **identical to the original, no harm done.**

**Verify it algebraically at initialisation:**

```
At step 1,  Z_1 = 0  and  Z_2 = 0.

y_c = F(x) + Z_2( F_copy( x + Z_1(c) ) )
    = F(x) + Z_2( F_copy( x + 0 ) )        ← control map multiplied by zero
    = F(x) + Z_2( F_copy(x) )
    = F(x) + 0                              ← output of a zero-weight conv
    = F(x)
    = y                                     ← EXACTLY the original model
```

**Adding ControlNet to a model changes nothing until it has learned something.** There is no window
during which the base model is degraded.

### The subtle part

> **Gradients still flow: A zero-init conv has zero weight but **non-zero gradient**, so control is
> learned smoothly from a safe identity start. This is why ControlNet fine-tunes fast and never
> wrecks the base model.**

This is the detail that makes the trick work rather than merely be safe, and it deserves the
arithmetic.

> 📚 **Why a zero-weight layer still learns.**
> For a layer computing $y = w \cdot x$, the gradient with respect to the weight is:
> $$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial y} \cdot x$$
>
> **Note what's absent: $w$ itself.** The weight's gradient depends on the *input* $x$ and the
> *upstream* gradient — **not on the current weight's value.** So with $w = 0$:
> - The **output** is 0 (no effect on the model) ✓
> - The **gradient** is $\frac{\partial \mathcal{L}}{\partial y} \cdot x$, which is **non-zero** ✓
>
> The layer contributes nothing and still learns. After one update $w \neq 0$ and it begins to
> contribute — smoothly, from a safe start.
>
> ⚠️ **Contrast this with initialising an entire *network* to zero, which genuinely does fail** —
> there, every neuron in a layer receives the identical gradient and they can never differentiate
> (the "symmetry problem"). Zero-init works here specifically because it's a **single $1\times1$
> convolution sitting between components that are already asymmetric and richly initialised.**

> 💡 **This is the third appearance of the same pattern, so name it and keep it.**
> - **LoRA** (lecture 3): $B = 0$, so $BA = 0$, so $W' = W_0$ exactly at start.
> - **adaLN-Zero** (section 16): modulation initialised so each block is identity at start.
> - **ControlNet**: zero-convolutions, so the control branch contributes nothing at start.
>
> **The pattern: initialise a new component as a no-op so it cannot damage a working system, and
> rely on the fact that zero weights still receive gradients to let it grow into usefulness.** If
> you ever need to add capability to a trained model, this is the first thing to reach for.

---

## 19. Editing, not just generating: img2img, inpainting, instructions

*(Slide 28)*

> **The same denoiser edits a real image: start the reverse process from a partially-noised version
> of it.**

### SDEdit / img2img — the core trick

> **Don't start from pure noise. Encode the input image to $z_0$, noise it only to an intermediate
> $t_0 = \text{strength} \cdot T$, then run the reverse process from there with the **new prompt**.**

$$z_{t_0} = \sqrt{\bar\alpha_{t_0}}\,z_0 + \sqrt{1 - \bar\alpha_{t_0}}\,\epsilon, \qquad t_0 = \text{strength} \cdot T$$

| Symbol | Read it as | What it means |
|---|---|---|
| $z_0$ | "z nought" | The **real input image**, encoded to a latent by the VAE. |
| $t_0$ | "t nought" | **Where in the chain to start.** Not $T$ — partway. |
| $\text{strength}$ | "strength" | A dial from 0 to 1. **The one control.** |
| $z_{t_0}$ | "z sub t-nought" | The partially-noised latent you start denoising from. |

**Note this is exactly section 6's closed-form noising equation**, evaluated at $t_0$ instead of $T$.
No new mathematics at all.

> 💡 **The intuition in one line: partially destroy the image, then let the model rebuild it toward a
> different prompt.** Noise removes detail before it removes composition — a lightly-noised image
> still has its layout, just not its texture. So a small amount of noising erases only surface
> detail, and the reverse process fills it back in guided by your new prompt. A large amount erases
> the structure too, and you get something almost new.

> **Low strength gives a tiny change; high strength keeps only the rough layout. One dial trades
> **faithfulness against edit magnitude**.**

The slide's chart, two lines crossing:

| Strength ($t_0/T$) | Preserved from input | Regenerated |
|---|---|---|
| 0% | 100% | 0% |
| 20% | ~80% | ~20% |
| **40–80%** | — | — | ← **shaded "typical img2img" band** |
| 50% | ~50% | ~50% | ← the crossover |
| 100% | 0% | 100% (ignores the input entirely) |

**Worked example:**

```
strength = 0.3, T = 1000  →  t_0 = 300

Suppose ᾱ_300 = 0.65:
  z_300 = √0.65 z_0 + √0.35 ε
        = 0.806 z_0 + 0.592 ε

The input still contributes ~81% of the signal. Composition and
colour survive; fine texture does not. Denoising from step 300 with a
new prompt restyles the surface while keeping the scene.
```

### Inpainting

> **Given a binary mask $m$, at every reverse step **overwrite the known region** with a
> correctly-noised copy of the original and let the model only solve the hole.**

```
At each reverse step t:
    z_t ← model's denoising step output
    z_t ← m ⊙ z_t  +  (1 - m) ⊙ noise_to_level_t(z_0_original)
            └ the hole ┘        └─ everything outside the mask ─┘
                                   forcibly reset to the original
```

> 💡 **Why the known region is re-noised to level $t$ rather than pasted in clean.** The denoiser
> expects its input to be uniformly noisy at level $t$. Pasting a *clean* patch into a noisy latent
> creates an input the model has never seen, and it produces visible seams. **Re-noising the known
> region to the correct level keeps the whole latent statistically consistent**, so the model treats
> it as ordinary context.

> **Fine-tuned inpainting models add the mask and masked-latent as **extra U-Net input channels
> (4 to 9)** for cleaner seams. **Outpainting** is the same with the mask outside the canvas.**

The channel arithmetic: 4 (the noisy latent) + 4 (the masked original latent) + 1 (the mask itself)
= **9 input channels** instead of 4. The model now *knows* which region is the hole rather than
having to infer it, which measurably improves seam quality.

### Instruction editing

> **InstructPix2Pix trains on (image, instruction, edited-image) triples so you can say "make it
> winter" with no mask, using **two guidance scales**, one for the image and one for the
> instruction.**

The dual guidance is a direct extension of section 9's CFG:

```
ε̂ = ε_∅
   + w_image       × (ε_image - ε_∅)              ← how much to respect the input image
   + w_instruction × (ε_image,instr - ε_image)    ← how much to obey the instruction
```

Two dials: *how faithful to the original* and *how strongly to apply the edit*. Independently
tunable, which is exactly what editing needs.

### Exact inversion

> **Real-image editing that must preserve untouched detail needs the noise that *reconstructs* the
> input: **DDIM inversion** runs the ODE backward; **null-text inversion** and prompt-to-prompt then
> edit via the cross-attention maps without drifting.**

The problem with img2img: you noise with a **random** $\epsilon$. Denoising won't return exactly the
original — parts you wanted untouched drift.

**DDIM inversion** solves this using section 10's determinism. Because the DDIM trajectory is a
deterministic ODE, you can **run it backwards** from the real image to find the specific noise that
would regenerate it exactly. Start from *that* noise, change the prompt, and only the intended
aspect changes.

**Prompt-to-prompt** goes further: it edits by manipulating the **cross-attention maps** directly.
Since (section 17) each word's attention map shows which pixels it controls, you can swap the word
`cat` for `dog` while **retaining the original attention geometry** — so the dog appears in exactly
the cat's pose and position.

### Summary

| Method | Use when | The control |
|---|---|---|
| **img2img** | Restyle the whole image | One strength dial |
| **Inpaint** | Edit inside a mask, keep the rest exact | The mask |
| **Outpaint** | Extend beyond the canvas | Mask outside the frame |
| **Instruct** | "Make it winter", no mask | Two guidance scales |
| **DDIM inversion** | Precise edit, untouched detail must survive | Exact reconstructing noise |

> 💡 **Notice that every one of these reuses the same trained denoiser.** No editing model was
> trained (except InstructPix2Pix, which is a fine-tune). **Editing is a sampling-time technique** —
> the same pattern as samplers (section 10) and guidance (section 9). The ε-predictor really is the
> whole model.

---

# PART 3 · Diffusion comes for text

## 20. What if text were generated all at once?

*(Slide 30)*

> **Autoregressive models commit one token at a time; diffusion fills the whole sequence in
> parallel.**

### The constraint

> **An autoregressive LLM must produce tokens **left to right**: token N waits for token N−1.
> Inherently sequential.**

From lecture 1: this is the defining property of autoregressive generation, and from lecture 3, the
reason decoding is memory-bandwidth-bound — one full pass over the weights per token.

### The alternative

> **Start with a fully **masked** sequence and **unmask many tokens at once** over a few denoising
> rounds. Generation becomes parallel.**

The slide's comparison:

```
Same sentence, two ways to generate it

Autoregressive: one token at a time, left → right
 ┌─────┐┌───────┐┌───────┐┌─────┐┌───────┐┌─────┐┌───────┐
 │ The ││ quick ││ brown ││ fox ││ jumps ││ the ││ fence │
 └─────┘└───────┘└───────┘└─────┘└───────┘└─────┘└───────┘
        step 7 of 7: must wait for the previous token

Diffusion: all masked, unmask in parallel over a few steps
 ┌─────┐┌───────┐┌───────┐┌─────┐┌───────┐┌─────┐┌───────┐
 │ The ││ quick ││ brown ││ fox ││ jumps ││ the ││ fence │
 └─────┘└───────┘└───────┘└─────┘└───────┘└─────┘└───────┘
        step 3 of 3: many tokens at once
```

**7 sequential steps → 3 parallel rounds.** And the gap widens with length: a 500-token output takes
500 autoregressive steps but still only a handful of diffusion rounds.

*In everyday words:* autoregressive generation is writing a sentence word by word without ever
looking ahead. Diffusion is sketching the whole sentence roughly, then refining it — the way people
actually draft.

```interactive
type: animation
title: Replay generation — autoregressive vs. diffusion
concept: The same output sentence assembled by two structurally different generation processes
control: A "Replay generation" button (the deck's own control) that re-runs both animations from the start
observe: The autoregressive row lights up its seven word-tiles one at a time, left to right, ending on "step 7 of 7: must wait for the previous token"; the diffusion row's seven tiles (all masked at the start) resolve together across "step 3 of 3: many tokens at once"
insight: Watching them replay side by side makes the step count itself the point — the autoregressive row's step count is fixed to the sentence length (7 tokens, 7 steps), while the diffusion row's step count is decoupled from length (3 rounds regardless of how many tokens fill in per round), which is exactly why the gap widens for longer outputs
fallback: The slide's static two-row comparison already described in this section — the autoregressive row frozen at "step 7 of 7" and the diffusion row frozen at "step 3 of 3" — shows the same two end states this replay button would animate toward.
```

> 💡 **The connection to the image case.** In image diffusion the "noise" is Gaussian static. For
> discrete tokens, adding Gaussian noise makes no sense — a token is a category, not a number. So
> **masking plays the role of noising**: the forward process progressively replaces tokens with
> `[MASK]`, and the reverse process progressively unmasks. Same structure, discrete alphabet.

### Three approaches

> **Discrete** (mask & unmask: **D3PM, MDLM, SEDD**), like iterative BERT; **continuous** (diffuse
> embeddings, then round: **Diffusion-LM**); **flow matching** for text.

| Approach | How | Named examples |
|---|---|---|
| **Discrete** | Mask tokens, unmask progressively | D3PM, MDLM, SEDD |
| **Continuous** | Diffuse in *embedding* space, round to nearest token at the end | Diffusion-LM |
| **Flow matching** | A related continuous-time formulation | — |

> 💡 **"Like iterative BERT" is a genuinely useful anchor.** Recall lecture 1: BERT is trained by
> masking ~15% of tokens and predicting them, using context from **both directions**. Discrete
> language diffusion is that idea run **iteratively** — mask 100%, unmask a fraction, repeat. The
> bidirectional context is exactly what lets it fill a gap in the middle, which an autoregressive
> model structurally cannot do.

### It is working

> **Mercury** (real-time), **LLaDA** (rivals AR at scale), **Plaid**, **Dream** (diffusion +
> reasoning).

> ⚠️ **verify this** — these are recent and fast-moving systems. Treat the specific claims
> ("real-time", "rivals AR at scale") as the lecturer's characterisation at the time; check current
> results before relying on them.

---

## 21. Autoregressive vs diffusion: complementary strengths

*(slide_032.jpg)*

> **Diffusion wins on parallelism and editing; autoregressive still leads on open-ended quality.**

### The comparison table

| | Parallel decode | Native infilling | Streaming output | KV-cache friendly | Open-end quality |
|---|---|---|---|---|---|
| **Autoregressive** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Language diffusion** | ✅ | ✅ | ❌ | ❌ | ❌ |

**The two rows are almost exact complements** — which is the slide's point.

### Diffusion's edge

> **Parallel decoding (speed), native **infilling and editing** (no left-to-right constraint), and
> stronger **controllability** for constrained output.**

**Infilling** deserves unpacking. Given `"The quick ___ fox jumps over the ___ dog"`, an
autoregressive model must generate left to right — so when filling the first gap it cannot see
`dog`, and it has no direct mechanism to condition on the text that follows. Diffusion sees the
entire sequence at every step, so **filling a hole is its native operation, not a workaround.**

This is exactly why **code editing** is the flagship use case: modifying a function in the middle of
a file is precisely an infilling problem, and it is what autoregressive models handle least
naturally.

### Autoregressive's edge

> **Still ahead on **open-ended quality** at frontier scale, supports **streaming**, and benefits
> from mature **KV-cache optimisations**.**

**Streaming** is a genuine product concern. An autoregressive model emits token 1 immediately, so
text appears as it's generated and perceived latency is low. **A diffusion model has nothing to show
until the last round completes** — the whole sequence resolves at once. Even if total time is
shorter, the user waits in silence and then gets everything.

**KV-cache friendliness** is the deeper structural issue, and it connects directly to lecture 3.
Autoregressive decoding caches K and V because past tokens **never change**. In diffusion, tokens
are revised at every round, so **there is nothing stable to cache** — every round recomputes
everything. Diffusion loses the single most important inference optimisation in the LLM stack, plus
the years of engineering built on it (PagedAttention, continuous batching, speculative decoding).

> 💡 **That last point is easy to underestimate.** Language diffusion isn't just competing with
> autoregressive *models* — it's competing with a decade of accumulated systems engineering built on
> autoregression's assumptions. The theoretical parallelism advantage has to overcome a very large
> practical head start.

### Where it's heading

> **Hybrids: autoregressive to plan structure, diffusion to fill it in; and diffusion for
> **structured output** (code, JSON) where parallelism shines.**

> 💡 **Note that "diffusion for structured output" pairs naturally with lecture 3's constrained
> decoding.** When the output must satisfy a schema, you already know a great deal about it before
> generating — which fields exist, what types they take. **Diffusion can fill those slots in
> parallel with global consistency**, where an autoregressive model commits to field 1 before
> knowing field 7. For JSON and code, that's a real structural advantage.

### The verdict

> **Not a replacement yet, but a real second option, and the gap is closing fast.**

That is an appropriately calibrated statement, and it's the right one to carry: language diffusion
is not hype, and it is also not a solved replacement.

---

## 22. How do you score an image generator?

*(slide_033.jpg)*

> **No ground-truth pixel to match, so we measure distributions, alignment, and ultimately human
> preference.**

### The problem

For a language model you can measure perplexity against held-out text (lecture 1). For an image
generator there **is no correct image**. Ask for "a red fox in snow" and there are infinitely many
right answers. So what do you measure?

### FID

> **FID: does the output look like real data?** Embed real and generated images with an **Inception**
> network, fit a Gaussian to each, and take the **Fréchet distance** between them. Lower is better.
> (Heusel et al., NeurIPS 2017 — see "Going deeper" #23.)

$$\mathrm{FID} = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\big)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\mu_r, \mu_g$ | "mu real, mu generated" | Mean embedding vector of the real and generated image sets. |
| $\Sigma_r, \Sigma_g$ | "Sigma real, Sigma generated" | Covariance matrices — how the embeddings are spread out. |
| $\|\mu_r - \mu_g\|^2$ | "squared distance of means" | Are the two sets **centred** in the same place? |
| $\mathrm{Tr}(\cdot)$ | "trace" | Sum of the diagonal of a matrix. |
| The trace term | | Do the two sets have the same **shape and spread**? |

*In everyday words:* summarise both sets of images as clouds of points in a feature space, then
measure how far apart and how differently shaped the two clouds are.

> **It scores fidelity + diversity jointly, but is biased by sample count, sensitive to the
> backbone, and blind to the prompt.**

> 💡 **"Blind to the prompt" is FID's most important weakness.** FID compares *distributions of
> images*. It never looks at what you asked for. A model that produces gorgeous, realistic,
> diverse images **that completely ignore every prompt** scores an excellent FID. It measures
> "do these look like real photographs", not "did it do what I asked".

### CLIP-score

> **CLIP-score: does it match the prompt?** Cosine similarity between the CLIP image embedding and
> the text embedding. Captures **alignment** that FID ignores, but **saturates and is gameable**,
> since the model can be tuned toward the very encoder doing the grading.
> (Hessel et al., EMNLP 2021 — see "Going deeper" #24.)

This is section 2's CLIP used as a judge: embed the generated image, embed the prompt, take the
cosine similarity (lecture 3).

> ⚠️ **"The model can be tuned toward the very encoder doing the grading" is Goodhart's law again**
> (lecture 2). Most text-to-image models **use CLIP as their text encoder**. Scoring them with CLIP
> means the metric and the model share a component — so optimising the score partly optimises
> agreement with a shared blind spot rather than genuine quality. And recall section 2: CLIP is weak
> at composition, so CLIP-score is weak at detecting compositional failures.

### Precision and recall

> **FID conflates two failure modes; split them: **precision** = fraction of samples landing in the
> real manifold (quality), **recall** = fraction of the real manifold the model covers (diversity /
> mode collapse). A sharp but repetitive model scores high precision, low recall.**
> (Kynkäänniemi et al., NeurIPS 2019 — see "Going deeper" #25.)

> 💡 **This split is genuinely useful diagnostically.** A single FID number can be mediocre for two
> completely opposite reasons:
> - **Low precision, high recall:** produces varied images, many of which look wrong.
> - **High precision, low recall:** every image is beautiful, and they all look the same
>   (**mode collapse**).
>
> These demand opposite fixes. FID alone cannot distinguish them; precision/recall can. **Note this
> is the same faithfulness-versus-diversity axis as guidance scale in section 9** — cranking $w$ up
> raises precision and destroys recall.

### Compositional benchmarks

> **Compositional probes (**GenEval**, **T2I-CompBench**) test counting and spatial binding that
> CLIP-score misses.**

*Concretely:* "three red cubes to the left of two blue spheres" tests counting, colour binding, and
spatial relations — exactly CLIP's weaknesses from section 2. CLIP-score would rate a picture with
two red cubes and three blue spheres highly. GenEval would not.

### Human preference — the real target

> **Human preference is the real target. Reward models (**ImageReward, PickScore, HPSv2**) trained on
> human pairwise choices correlate far better than FID, and are used to both **rank** and **fine-tune**
> (RLHF / DPO).**

> 💡 **This is lecture 2 arriving in the image world, wholesale.** Collect human pairwise
> preferences → train a reward model → use it to rank outputs **and to fine-tune with DPO**. Exactly
> the pipeline from lecture 2, sections 5–8, applied to images instead of text. **And it carries the
> same risks**: the reward model is a *proxy*, and optimising it too hard produces reward hacking —
> in image models this shows up as an over-saturated, over-detailed "AI aesthetic" that scores well
> and looks synthetic.

### The verdict

> **No single number suffices: FID for realism, CLIP / GenEval for alignment, human preference as
> the tiebreaker. Each is gameable alone, so report a basket.**

| Metric | Measures | Blind to |
|---|---|---|
| **FID** | Realism + diversity | The prompt entirely |
| **CLIP-score** | Prompt alignment | Composition; gameable via shared encoder |
| **Precision / Recall** | Quality vs coverage, separately | Prompt alignment |
| **GenEval / T2I-CompBench** | Counting, spatial binding | General aesthetic quality |
| **Human preference** | What people actually choose | Expensive; can be hacked at scale |

---

## 23. Measuring these models, and a transition

*(slide_034.jpg)*

> **Benchmarks saturate, so evaluation is a moving target, and powerful generators raise the
> stakes.**

### Benchmarks saturate

> **Knowledge (**MMLU, GPQA**), reasoning (**GSM8K, MATH**), code (**HumanEval, SWE-bench**), chat
> (**Chatbot Arena, ELO**). They **saturate**, get contaminated, and get gamed.**

| Benchmark | Tests |
|---|---|
| **MMLU** | Multiple-choice knowledge across 57 subjects |
| **GPQA** | "Google-proof" graduate-level science questions |
| **GSM8K** | Grade-school multi-step math (lecture 3) |
| **MATH** | Competition mathematics |
| **HumanEval** | Function-level code generation, verified by tests |
| **SWE-bench** | Resolving real GitHub issues in real repositories |
| **Chatbot Arena** | Human pairwise votes, aggregated into **ELO** ratings |

The chart shows scores over time, with a human-performance line:

```
100% ┤                          ╭──────── GSM8K
     │                     ╭────┼──────── MMLU     ── human ──
 80% ┤          ╭──────────╯    │
     │      ╭───╯                │              ╭──── GPQA
 60% ┤   ╭──╯                    │          ╭───╯
     │╭──╯                       │      ╭───╯
 40% ┤╯                          │  ╭───╯
     │                           ╭──╯
 20% ┤                       ╭───╯
     └──┬────┬────┬────┬────┬────┬────┬───
       2019 2020 2021 2022 2023 2024 2025 2026

  As soon as a benchmark is solved, the field builds a harder one.
```

> 💡 **Read the caption as a structural claim, not a complaint.** MMLU and GSM8K both crossed human
> performance and flattened at the ceiling — once a benchmark saturates it **stops carrying
> information**, because every frontier model scores 92%, 93%, 94% and the differences are noise.
> GPQA was built precisely because the earlier ones stopped discriminating.
>
> **This means benchmark scores have a shelf life.** A model's MMLU score was informative in 2022
> and is nearly meaningless in 2026. When you read an evaluation, ask whether that benchmark is
> still in its discriminating range.

The three failure modes named:

- **Saturate** — everyone scores near the ceiling; the metric stops separating models.
- **Contaminated** — test questions leak into training data (lecture 1, section 14.2). Scores then
  measure memorisation.
- **Gamed** — models are tuned toward benchmark formats. Goodhart's law (lecture 2) once more.

> 💡 **And remember lecture 3's sampling-regime asterisk.** A score is meaningless without knowing
> how many samples were drawn and how the answer was selected — the same frozen weights scored 74.4
> or 93.0 on AIME. **Combine that with saturation and contamination and you should read every
> published benchmark number with three questions attached.**

### Hallucination

> **Factual, faithfulness, and reasoning errors. Mitigate with **grounding, citations,** and
> **constrained generation**.**

Three distinct kinds, worth separating:

| Type | What it is | Example |
|---|---|---|
| **Factual** | States something false about the world | Wrong date for a historical event |
| **Faithfulness** | Contradicts the provided source | RAG retrieves the right passage; the answer misstates it |
| **Reasoning** | Valid-looking chain with an invalid step | Arithmetic that is wrong in the middle |

The three mitigations map exactly onto lecture 3:

- **Grounding** → RAG (section 14 of lecture 3). Facts belong in context, not weights.
- **Citations** → answer with the source, so claims are auditable.
- **Constrained generation** → mask invalid outputs (section 3 of lecture 3).

### LLM-as-a-judge

> **Use a strong model to grade outputs at scale: cheap and fast, but inherits the judge's
> **biases**.**

From lecture 2's RLAIF: a model can evaluate faster and far more cheaply than humans. The costs:

- **Position bias** — judges often prefer whichever answer is shown first.
- **Length bias** — longer answers are rated higher, largely independent of quality.
- **Self-preference** — a model tends to rate its own outputs (or its family's) higher.
- **Style over substance** — confident, well-formatted answers score above hesitant correct ones.

> 💡 **All four are systematic, not random**, which is the problem. Random noise averages out over
> many judgements; **systematic bias does not.** And it compounds: if you *fine-tune* against a
> biased judge (RLAIF), you actively train the model to exploit those biases. Standard mitigations
> are randomising presentation order, controlling for length, and using a judge from a different
> model family than the one being evaluated.

### Safety and bias

> **Representation harms, **red-teaming**, and the safety-versus-capability trade-off, sharper now
> that models **generate** images, voice, and video.**

> **Red-teaming** — deliberately attacking your own model to find failures before deployment:
> adversarial prompts, jailbreaks, attempts to elicit harmful output.

> 💡 **Why "sharper now that models generate images, voice, and video" is the right closing note for
> this lecture.** A text model that produces something false produces a paragraph you can check. The
> models in *this* lecture produce **photorealistic images, cloned voices, and video** — artefacts
> that are evidence-shaped, that people believe by default, and whose harms (non-consensual imagery,
> voice-cloned fraud, fabricated footage) do not have text analogues.
>
> The **safety-versus-capability trade-off** is lecture 2's alignment tax, and it applies here with
> the same structure: over-restrict and the tool becomes useless; under-restrict and it becomes
> dangerous. Multimodality doesn't create the trade-off — it raises the stakes on both sides.

### The transition

> **Next: these models are powerful but **passive**. What if they could *act*? On to agents.**

> 💡 **The word "passive" is the whole framing.** Everything across these four lectures — text
> generation, alignment, retrieval, images, video, speech — **produces output and stops.** You ask,
> it answers, the interaction ends. An **agent** takes actions: calls tools, writes and runs code,
> browses, and loops on the results.
>
> And note how much of this series is prerequisite to that. Tool calls are JSON, so they need
> **constrained decoding** (lecture 3). Multi-step plans need **reasoning and test-time compute**
> (lecture 3). Acting safely on a user's behalf needs **alignment** (lecture 2). Voice interfaces
> need **native speech-to-speech** (section 4). **Agents are not a new topic so much as the
> composition of everything already covered.**

---

## Putting it together

### The dependency structure

```
              ONE IDEA: IF YOU CAN TOKENISE IT,
              A TRANSFORMER CAN MODEL IT
     text→sub-words · image→patches · audio→frames
                          │
        ╔═════════════════╪═════════════════╗
        ║  Q1: HOW DO WE FUSE MODALITIES?   ║
        ╚═════════════════╪═════════════════╝
                          │
              ┌───────────┴────────────┐
              │                        │
            CLIP                   FUSION PATTERNS
      contrastive: pull the            │
      diagonal, push the rest    ┌─────┴──────┐
      400M pairs, symmetric      │            │
      InfoNCE                MODULAR      UNIFIED
              │              (LLaVA)    (GPT-4o,
      → ZERO-SHOT:           frozen LLM   Gemini)
        the classifier       + projection  joint
        head became a        (cheap,       pretraining
        SENTENCE             capped)       (any-to-any)
              │                        │
              │                   AUDIO: cascade (2.2 s,
              │                   prosody LOST at the text
              │                   bottleneck) → native
              │                   speech-to-speech (0.3 s)
              │                        │
              └────────┬───────────────┘
                       │  CLIP's text encoder is reused below ↓
                       │
        ╔══════════════╪══════════════════════════╗
        ║  Q2: HOW DO WE GENERATE THEM?           ║
        ╚══════════════╪══════════════════════════╝
                       │
              DIFFUSION: destroy with noise,
              learn to reverse it
                       │
        forward q: FIXED, no parameters
        x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε   ← closed form
                    ↑                      = cheap training
        reverse p_θ: LEARNED
        L = E‖ε - ε_θ(x_t,t)‖²          ← plain MSE
                       │
        ┌──────────────┴─────────────────────────┐
        │  THE WHOLE MODEL IS AN ε-PREDICTOR.    │
        │  Everything below is built AROUND it.   │
        └──────────────┬─────────────────────────┘
                       │
   ┌───────┬───────────┼───────────┬─────────────┬──────────┐
   │       │           │           │             │          │
WHAT IT  HOW YOU    HOW HARD   WHERE IT      WHAT KIND   HOW YOU
PREDICTS  STEP      IT OBEYS    RUNS        OF NETWORK   STEER IT
   │       │           │           │             │          │
 ε/x₀/v  samplers     CFG      LATENT        U-Net       text →
 targets    │       ε̂=ε_∅+w(   diffusion    (skips carry  cross-attn
   │     DDPM 1000   ε_txt-ε_∅) 786k→16k    hi-freq        │
 schedules  ↓         │        = 48×        detail)     ControlNet
 (cosine  DDIM 50   w≈7-8         │            │        (structure)
  beats     ↓      faithful   ⇒ Stable      → DiT        │
  linear) DPM++ 20  vs        Diffusion    (patchify,   IP-Adapter
   │         ↓      diverse    on a         adaLN-Zero,  (style)
 Min-SNR  ── retrain ──         consumer     scales like    │
 caps     distill 4             GPU          an LLM)     they COMPOSE
 weights  consistency 1-4                       │           │
   │      ADD/GAN 1-4                    SD3, Flux, Sora    │
 zero-      │                                   │           │
 terminal  ~1000 → 1                     VIDEO = spacetime  │
 SNR fixes                               patches (same      │
 "no dark                                model, 3-D grid)   │
  images"                                temporal consistency
                                         UNSOLVED           │
                                                            │
                              EDITING (no new model): ──────┘
                              img2img (one strength dial)
                              inpaint (mask + re-noise)
                              instruct (dual guidance)
                              DDIM inversion (exact)
                       │
        ╔══════════════╪══════════════╗
        ║  DIFFUSION COMES BACK FOR   ║
        ║  TEXT                       ║
        ╚══════════════╪══════════════╝
                       │
          mask instead of Gaussian noise
          parallel unmasking, native infilling
                       │
          BUT: no streaming, no KV cache,
          behind on open-ended quality
          → hybrids; structured output
                       │
        ╔══════════════╪══════════════╗
        ║  HOW DO WE KNOW IT WORKS?   ║
        ╚══════════════╪══════════════╝
                       │
     FID (blind to prompt) · CLIP-score (gameable)
     precision/recall (quality vs coverage)
     GenEval (composition) · HUMAN PREFERENCE
     → report a BASKET, each is gameable alone
                       │
     benchmarks SATURATE, get contaminated, get gamed
                       │
                       ▼
              NEXT: these models are
              POWERFUL but PASSIVE.
              What if they could ACT?
                    → AGENTS
```

### Walking through it

**One idea underlies the whole first half.** A transformer takes a sequence of vectors and mixes
them with attention; **nothing in that computation refers to language**. So cut an image into
patches, audio into spectrogram frames, text into sub-words, and all three become vectors flowing
into the same attention. Multimodality is a tokenizer question, not an architecture question — and
because self-attention mixes *every* token with *every other*, a word token can attend directly to
the image patches it describes. That's what "grounding" means, and nothing special was built to
achieve it.

**CLIP builds the shared space.** Two encoders, one projection each, and a contrastive objective:
pull the true image-caption pairs together, push the $N^2 - N$ mismatched pairs apart. The batch is
the supervision, so 400 million web image-caption pairs need **no labels at all**. The payoff is
zero-shot classification — you define your classes by *typing them*, because **the classifier head
became a sentence**. CLIP's text encoder then reappears inside almost every image generator in
Part 2.

**Fusion has two patterns**, and they mirror lecture 3's PEFT-versus-full-training trade-off.
**Modular** (LLaVA) freezes a strong text LLM and trains only a projection that makes image features
look like token embeddings — cheap, reusable, and capped, because the LLM was never trained to
reason over pixels. **Unified** (GPT-4o, Gemini) tokenises every modality and trains one model
jointly — any-to-any, deeper grounding, frontier-scale cost. **Audio makes the trade-off vivid**: a
Whisper→LLM→TTS cascade takes ~2.2 s against a ~200 ms conversational budget, and the text
bottleneck **discards everything in the voice** — tone, emotion, sarcasm. A native speech-to-speech
model reaches ~0.3 s and keeps prosody, because audio is just another token stream.

**Generation works completely differently.** Diffusion adds Gaussian noise to an image over $T$
steps until it's static — a **fixed process with zero parameters** — and trains a network to reverse
one step. Because Gaussians compose, you can jump to any noise level in **one line of arithmetic**,
which is what makes training cheap. And the loss, after all the probability theory, is **plain MSE**:
predict the noise you added. To generate, start from static that never contained an image and denoise
anyway — the model hallucinates a picture out of nothing.

**Everything after that is built around one regression target.** *What* the network predicts —
ε, $x_0$, or $v$ — carries identical information but differs in conditioning, which is why v-pred
matters for distillation and why the zero-terminal-SNR bug stops models producing truly dark
images. The **schedule is a curriculum** deciding which noise levels get the training budget;
Min-SNR stops easy high-SNR steps dominating the gradient. **Guidance** runs the ε-predictor twice
and extrapolates past the conditional prediction, trading faithfulness against diversity at
$w \approx 7$–8. **Samplers** re-read the reverse process as an ODE and cut ~1000 steps to ~20 with
no retraining; **distillation and consistency models** retrain to reach 1–4, giving up a little top
-end quality and generality for a 10–50× speed-up. **Latent diffusion** runs the whole loop in a
48×-smaller VAE space — the change that put image generation on a consumer GPU.

**The backbone converged back on the transformer.** The U-Net's skip connections were exactly right
for denoising, carrying high-frequency detail past the bottleneck so the decoder never has to
hallucinate edge positions. But its hierarchy is a **built-in assumption**, and assumptions help
when data is scarce and constrain when it is abundant. **DiT** patchifies the latent, runs N
identical transformer blocks, and injects timestep and text by **modulating LayerNorm** — and
inherits LLM scaling. SD3, Flux, and Sora are all DiT. **Video needed almost no change**: patchify a
spacetime volume instead of a plane. Temporal consistency remains unsolved.

**Steering happens through three doors that compose.** Text enters by cross-attention (image patches
query, prompt answers) and sets **content**. ControlNet clones a block, wraps it in
**zero-initialised convolutions**, and pins **structure** — and because zero weights still receive
gradients, it starts as an exact identity and grows into usefulness without ever damaging the base.
IP-Adapter borrows **style** through a decoupled cross-attention. **Editing needs no new model at
all** — noise a real image partway and denoise toward a new prompt, with one strength dial trading
faithfulness against edit size.

**Then diffusion comes back for text.** Mask instead of Gaussian noise; unmask many tokens per
round instead of one per step. It wins on **parallel decoding, native infilling, and controllability
for structured output** — and loses on **streaming, KV-cache friendliness, and open-ended quality**,
which is a large practical head start to overcome. Not a replacement, a real second option.

**And evaluation is genuinely hard**, because there is no correct image. FID measures realism and
diversity but is **blind to the prompt**; CLIP-score measures alignment but is **gameable by the very
encoder doing the grading**; precision and recall separate quality from coverage; GenEval probes
composition; **human preference is the real target**, imported wholesale from lecture 2's RLHF —
along with its reward-hacking risks. **No single number suffices; report a basket.** And every
benchmark saturates, gets contaminated, and gets gamed, which means scores have a shelf life.

### The closing frame

> **These models are powerful but **passive**. What if they could *act*? On to agents.**

And notice how much of this series turns out to be prerequisite to that: tool calls need
**constrained decoding**, multi-step plans need **reasoning and test-time compute**, acting on a
user's behalf needs **alignment**, and talking to a person needs **native speech**. Agents are
mostly the composition of what you now know.

---

## Interview prep — Amazon Applied Scientist

### Core questions (easy → hard)

<details><summary><b>1. Why can the same transformer architecture handle text, images, and audio with no architectural change?</b></summary>

Self-attention operates on a sequence of vectors and never inspects what a vector "means" — it only
computes dot products. So the only requirement for a modality is a **tokenizer**: something that
turns raw data into a sequence of vectors. Text → sub-words (BPE), images → patches (ViT-style),
audio → spectrogram frames. Once tokenized, all three flow into the identical attention computation.
Multimodality is a data-representation problem, not a model-architecture problem.

</details>

<details><summary><b>2. Explain CLIP's contrastive loss in full, including why it needs no human labels.</b></summary>

Embed a batch of $N$ image-text pairs with two separate encoders into one L2-normalised space. Form
the $N \times N$ cosine-similarity matrix. The $N$ diagonal entries are the true pairs (positives);
the $N^2-N$ off-diagonal entries are negatives, generated for free just by pairing each image with
every *other* caption in the batch. Train with symmetric InfoNCE — row-wise cross-entropy
(image→text) averaged with column-wise cross-entropy (text→image), scaled by a learned temperature
$\tau$. No labels are needed because the supervision is *structural* (which pairs came from the same
alt-text on the web), not annotated.

</details>

<details><summary><b>3. You have a 512×512×3 image and want to run diffusion in a 64×64×4 latent. What's the compute saving, and why isn't it 64×?</b></summary>

$786{,}432 / 16{,}384 = 48\times$. It isn't the full $64\times$ spatial reduction ($8\times8$)
because the channel count *rises* from 3 (RGB) to 4 — the latent trades some of the spatial saving
for extra depth per position to hold the compressed semantic information.

</details>

<details><summary><b>4. Write the diffusion training loss and say, plainly, what kind of loss it is.</b></summary>

$$\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}\big[\|\epsilon - \epsilon_\theta(x_t,t)\|^2\big]$$

It is **plain MSE regression** — predict a number (the noise that was added), compare to the ground
truth (which you know, because you generated it), square the difference. No adversarial game, no
likelihood bound to balance. That simplicity is a large part of why diffusion trains more reliably
than GANs.

</details>

<details><summary><b>5. Why does the closed-form equation $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ matter for training cost?</b></summary>

Because Gaussians compose, you can jump straight to any noise level $t$ in one line of arithmetic
instead of simulating $t$ sequential noising steps. A training step becomes: sample a random image,
sample a random $t$, sample noise, compute $x_t$ in one shot, predict $\epsilon$, take MSE. There is
no sequential simulation anywhere, so training is fully parallel across timesteps — the same
property that made transformers trainable at scale.

</details>

<details><summary><b>6. What does classifier-free guidance actually compute, and why does $w>1$ work?</b></summary>

$\hat\epsilon = \epsilon_\varnothing + w(\epsilon_{\text{text}} - \epsilon_\varnothing)$. Run the same
model twice per step — once with the prompt, once with an empty prompt (the model was trained to do
both, by randomly dropping the caption during training) — and extrapolate *past* the conditional
prediction in the direction the prompt pulls. $w>1$ is extrapolation, not interpolation: the model is
pushed toward images more prototypically "on-prompt" than its own honest estimate, which is why
adherence rises — and why diversity collapses and over-saturation appears past the sweet spot
($w\approx7$–8).

</details>

<details><summary><b>7. Why can't a better ODE solver push diffusion sampling below ~10 steps?</b></summary>

DDIM and DPM++ both integrate the same underlying probability-flow ODE more accurately per step —
but below ~10 steps, the **discretisation error** of *any* solver dominates: you're approximating a
strongly curved trajectory with a handful of straight segments, and that shows up visually as blur.
Getting lower requires changing what the network *computes*, not how you step along a fixed
trajectory — i.e., retraining via distillation or a consistency objective.

</details>

<details><summary><b>8. Why do ControlNet's zero-initialised convolutions still learn, when zero-initialising an entire network famously fails?</b></summary>

For $y = w\cdot x$, the weight's gradient is $\partial\mathcal{L}/\partial w =
(\partial\mathcal{L}/\partial y)\cdot x$ — it does **not** depend on $w$'s current value. So at
$w=0$ the output is exactly zero (no damage to the frozen base model) but the gradient is non-zero,
so the layer starts learning immediately. Zero-initialising a whole *network* fails for a different
reason (the symmetry problem: every neuron gets an identical gradient and can never differentiate).
ControlNet avoids that because the zero-init sits on a single $1\times1$ conv between components
that are already richly, asymmetrically initialised.

</details>

<details><summary><b>9. (Combines two concepts) Your text-to-image model can never produce a genuinely dark image, no matter the prompt. Diagnose and fix it.</b></summary>

This is the **zero-terminal-SNR bug**. Standard noise schedules leave $\bar\alpha_T \neq 0$, so at
training time the "fully noised" input $x_T$ still carries a faint ghost of $x_0$ — in particular,
its mean brightness leaks through. The model learns "brightness is hinted at even at $t=T$." At
sampling time you start from $x_T \sim \mathcal{N}(0,I)$, which has **no** such hint, and a
mean-zero input reads to the model as "medium brightness" — so it can never commit to very dark or
very bright output. This connects **schedule design** (§8) with **prediction target** (also §8): the
fix is two-part — rescale $\beta_t$ so $\mathrm{SNR}(T)=0$ exactly, *and* switch to v-prediction,
because at exactly zero SNR there is no signal left to condition an $\epsilon$-prediction on.

</details>

<details><summary><b>10. (Combines two concepts) Explain the full progression from DDPM's ~1000 steps to a 1-step model, and say which transitions require retraining.</b></summary>

DDPM (~1000 steps, stochastic) → DDIM (~50 steps): reformulate the reverse process as a
**deterministic** ODE trajectory, so you can skip along it. Same trained network — **no retraining**.
DDIM → DPM++ (~20 steps): apply higher-order ODE solvers that use the trajectory's curvature, not
just its slope. Still **no retraining**. DPM++ → distilled/consistency models (1–4 steps): below
~10 steps, discretisation error dominates *any* solver, so you must change what the network computes
— progressive distillation, consistency models, or adversarial distillation. This step **requires
retraining**, and it costs something: distilled models plateau slightly below the solver's top-end
quality and lose generality (less responsive to guidance, weaker on unusual prompts).

</details>

<details><summary><b>11. (Combines three concepts) Design a real-time voice assistant that also generates images on request, on a tight latency budget. Name every technique and the two dominant costs.</b></summary>

**Voice path:** native speech-to-speech (not a cascade) — a Whisper→LLM→TTS cascade costs ~2.2 s
against a ~200 ms conversational budget and discards prosody at the text bottleneck; a native model
reaches ~0.3 s and is interruptible. **Image path, stacking every Part-2 latency technique:** latent
diffusion (48× less work per denoising step), distillation/consistency (LCM or SDXL-Turbo, 1–4 steps
instead of 20–50), and constrained decoding on any tool call that triggers generation so it can't
fail to parse. **The two dominant costs:** (1) the denoising loop itself — even at 4 steps with CFG
doubling each one, that's 8 network passes per image; (2) time-to-first-audio on the voice path,
since perceived responsiveness is set by when the user hears *something*, not total completion time.

</details>

<details><summary><b>12. (Combines four concepts) Your text-to-image product has excellent FID and excellent CLIP-score, but users say images look "samey" and often have the wrong object count. Diagnose all of it.</b></summary>

**"Samey" = low recall / mode collapse**, almost certainly from **guidance scale set too high** — CFG
extrapolates toward the most prototypical image for a prompt, collapsing diversity as $w$ rises.
**Wrong counts = a compositional failure traced to CLIP**, which is closer to a bag of concepts than
a relationship parser and is the text encoder for most of these models, so its weakness propagates
into generation. **Why both metrics missed it:** FID is blind to the prompt entirely (it only
compares image distributions) and conflates quality with diversity into one number; CLIP-score is
gameable because the model and the grader **share** a component. **Fixes:** lower $w$ toward 7–8 and
re-measure recall; add precision/recall (would have flagged low recall immediately) and GenEval
(would have flagged the counting failure) to the reporting basket; consider a stronger text encoder
(T5 alongside CLIP, as SD3 does) for composition.

</details>

### Depth probes

The follow-ups an interviewer asks after a good first answer:

- *"You said CFG doubles inference compute — does it double wall-clock latency too?"* → No, not
  necessarily: implementations typically batch the conditional and unconditional passes together, so
  it's ~2× FLOPs but not always ~2× wall-clock on a GPU with spare capacity.
- *"Why does v-prediction fix zero-terminal-SNR but $\epsilon$-prediction can't?"* → At exactly
  $\mathrm{SNR}(T)=0$ there's no signal left in $x_T$ at all, so recovering $\epsilon$ (which requires
  dividing out a near-zero signal coefficient) becomes numerically degenerate. $v$-prediction is a
  rotation of $(\epsilon, x_0)$ whose weighting stays well-scaled at that extreme.
- *"If contrastive batch size matters so much for CLIP, why not just use an infinite batch?"* →
  Diminishing returns plus quadratic memory cost of the similarity matrix; also, the number of hard
  negatives (semantically close but wrong pairs) matters more than raw batch size past a point —
  large batches mostly help by increasing the odds of hard negatives appearing.
- *"When would you choose a modular (LLaVA-style) architecture over a unified one, given unlimited
  budget?"* → Rarely, if capability is the only goal — unified is strictly more capable. But modular
  wins when you need to iterate fast on a narrow vision task without touching a frontier-scale
  pretraining run, or when the base LLM is a vendor model you can't retrain at all.
- *"FID is 'blind to the prompt' — could you fix that by conditioning FID on the prompt somehow?"* →
  That's effectively what CLIP-score and GenEval exist to do; FID's Fréchet-distance construction is
  inherently a two-set (real vs. generated) distributional comparison and doesn't have a natural slot
  for per-sample conditioning, which is exactly why it needs a *complementary* metric rather than a
  patch.

### Whiteboard-ready derivations

**1. Derive the closed-form noising equation from the one-step forward process.**
Given $q(x_t\mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I)$, and using the fact
that Gaussians compose (adding $\mathcal{N}(0,a)$ then $\mathcal{N}(0,b)$ gives $\mathcal{N}(0,a+b)$),
unroll the recursion with $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s\le t}\alpha_s$ to reach
$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$. Verify the coefficients sum to 1
in squares — $(\sqrt{\bar\alpha_t})^2+(\sqrt{1-\bar\alpha_t})^2=1$ — confirming the schedule is
variance-preserving.

**2. Derive the CFG update from "run twice and push away from unconditional."**
Start from the stated goal: exaggerate the direction the prompt pulls in. Define that direction as
$\epsilon_{\text{text}} - \epsilon_\varnothing$. Scale it by $w$ and add it back to the unconditional
baseline: $\hat\epsilon = \epsilon_\varnothing + w(\epsilon_{\text{text}}-\epsilon_\varnothing)$. Check
the boundary cases: $w=0$ recovers pure unconditional generation; $w=1$ recovers ordinary conditional
generation; $w>1$ extrapolates past it.

**3. Derive the 48× latent-diffusion compute ratio from first principles.**
Pixel space: $512\times512\times3 = 786{,}432$ dims. Latent space: $64\times64\times4 = 16{,}384$
dims (spatial dims each shrink by $512/64=8$, giving a $64\times$ spatial reduction, but channels
rise $3\to4$). Ratio: $786{,}432/16{,}384 = 48$. Be ready to explain *why* it isn't $64\times$: the
extra channel depth is the price paid for keeping enough information to reconstruct the image well.

### Applied scenario — Amazon-flavoured

**Problem:** Build a listing-image generator for third-party sellers on the Amazon catalog — given a
seller's product photo and a short description, generate polished "lifestyle" images (product in a
styled setting) at catalog scale.

- **Framing.** This is instruction-guided image editing, not generation from scratch: the seller's
  actual product must be preserved exactly (brand, shape, colour, printed text), while the
  *background and context* around it can be freely generated. That rules out plain img2img at high
  strength (which would drift the product itself) and points toward **inpainting** (mask everything
  except the product, regenerate the surroundings) combined with **ControlNet** on a depth or edge
  map extracted from the seller's original photo, to pin the product's exact silhouette.
- **Data.** No paired (raw product photo → styled lifestyle photo) dataset exists at scale. Start
  from a frozen, pretrained latent diffusion backbone (do not train one from scratch — full
  pretraining is a frontier-lab-scale cost this doesn't need); fine-tune only the ControlNet branch
  and, if needed, a small IP-Adapter for consistent brand "style" across a seller's catalog.
- **Model.** Base: a DiT-backbone latent diffusion model (SD3/Flux-class) for better scaling
  behaviour and text-encoder quality (composition matters — "the shoe on a wooden table in morning
  light" has spatial and attribute-binding requirements that plain CLIP-only text encoders handle
  poorly, per §2 and §22). Steering: ControlNet on the product's structure (frozen product,
  changeable background), CFG for text adherence, a distilled/consistency sampler (1–4 to 20 steps)
  to keep per-image cost low at catalog scale.
- **Metric.** No single metric suffices, per §22: **CLIP-score / GenEval** for "does the described
  setting actually appear," **precision/recall** or a dedicated "product fidelity" check (e.g.
  perceptual hash or embedding distance between the generated product region and the original) to
  catch drift, and **human preference** (seller and buyer click-through / conversion lift) as the
  real target, since the actual business metric is whether the image increases purchase intent, not
  whether it scores well on any proxy.
- **Failure modes.** (1) Product drift — the generated product no longer matches the real item
  (a returns/complaints risk, and a trust violation with sellers); mitigated by the ControlNet
  structure lock plus an automated fidelity gate before publishing. (2) Compositional failures
  (wrong count, wrong colour binding) inherited from a weak text encoder — mitigated by GenEval-style
  automated checks in the pipeline, not just eyeballing samples. (3) Reward-hacking an "AI aesthetic"
  if a preference model is used for fine-tuning — over-saturated, synthetic-looking images that score
  well but convert poorly; mitigated by keeping human preference (real conversion data) as the
  ultimate signal, not the proxy reward model score.
- **What you'd ship.** A staged system: ControlNet + inpainting on a frozen DiT backbone, gated by an
  automated product-fidelity check before any image reaches a seller, with a slow, metric-basket
  evaluation loop (not a single number) driving iteration — and a clear human-review escape hatch for
  low-confidence fidelity scores rather than silently auto-publishing.
- **Leadership Principle ties.** **Customer Obsession** — the actual target metric is buyer trust and
  conversion, not FID or CLIP-score; those are proxies, and treating a proxy as the target (Goodhart's
  law, referenced throughout §22 and §9) is exactly the failure this design guards against.
  **Insist on the Highest Standards** — refusing to publish an image that fails the automated product
  -fidelity gate, even though it would raise "throughput," is choosing the harder-right (a review
  queue) over the easier-wrong (auto-publish everything the model produces).

---

## Glossary

| Term | One-line definition |
|---|---|
| **adaLN-Zero** | Adaptive LayerNorm whose scale/shift are predicted from the conditioning, initialised so the block starts as identity. |
| **ADD / LADD** | Adversarial Diffusion Distillation — adds a GAN discriminator to distillation, recovering sharpness. |
| **$\bar{\alpha}_t$** | Fraction of the original signal surviving at step $t$; the cumulative product of $(1-\beta_s)$. |
| **$\beta_t$** | How much noise the forward process adds at step $t$; set by a fixed schedule. |
| **CFG (classifier-free guidance)** | Run the model with and without the prompt and extrapolate away from the unconditional prediction. |
| **CLIP** | Contrastive Language-Image Pre-training — two encoders trained so images and captions land in one space. |
| **CLIP-score** | Cosine similarity between a generated image's and the prompt's CLIP embeddings. |
| **Consistency model** | Learns a function mapping any point on an ODE trajectory to the same origin; generates in one pass. |
| **ControlNet** | A trainable clone of the denoiser's encoder, wrapped in zero-convs, that adds structural control. |
| **Contrastive learning** | Training by "these belong together, those don't" rather than by labels. |
| **Cosine schedule** | Noise schedule that destroys signal gently at both ends, spending more steps at useful noise levels. |
| **Cross-attention** | Attention where one sequence queries another; how text steers the denoiser. |
| **DDIM** | Deterministic reformulation of the reverse process; enables step-skipping and inversion. |
| **DDIM inversion** | Running the deterministic trajectory backwards to find the noise that reconstructs a real image. |
| **DDPM** | Denoising Diffusion Probabilistic Models — the 2020 paper that made image diffusion work. |
| **Diffusion** | Generating by learning to reverse a fixed noise-adding process. |
| **DiT** | Diffusion Transformer — a transformer over latent patches replacing the U-Net denoiser. |
| **DPM++** | Higher-order ODE solver for the reverse process; ~20 steps. |
| **$\epsilon$-prediction** | Training the network to output the noise that was added. DDPM's default target. |
| **FID** | Fréchet Inception Distance — distance between the feature distributions of real and generated images. |
| **Forward process** | The fixed, parameter-free chain that adds noise to an image. |
| **GenEval / T2I-CompBench** | Benchmarks probing counting and spatial/attribute binding. |
| **Guidance scale ($w$)** | How strongly to amplify the prompt's influence. Sweet spot ≈ 7–8. |
| **Img2img / SDEdit** | Editing by noising a real image partway and denoising toward a new prompt. |
| **Inpainting** | Editing inside a mask by overwriting the known region with a correctly-noised original each step. |
| **InfoNCE** | The contrastive loss CLIP uses — cross-entropy over a batch's similarity matrix. |
| **InstructPix2Pix** | Instruction-based editing trained on (image, instruction, edited-image) triples. |
| **IP-Adapter** | Conditions generation on a reference image's style via a decoupled cross-attention. |
| **Latent diffusion** | Running the entire diffusion process in a VAE's compressed latent space. |
| **LCM** | Latent Consistency Model — a consistency model operating in latent space. |
| **Markov chain** | A process where the next state depends only on the current one. |
| **Min-SNR-$\gamma$** | Caps the per-timestep loss weight at $\gamma$ so easy high-SNR steps don't dominate. |
| **Modular fusion** | Bolting a vision encoder + projection onto a frozen LLM (LLaVA-style). |
| **Native speech-to-speech** | One model that hears audio and emits audio, with no text bottleneck. |
| **Patchify** | Cutting an image or latent into a grid of patches, each flattened into a token. |
| **Precision / Recall (generative)** | Fraction of samples in the real manifold / fraction of the real manifold covered. |
| **Progressive distillation** | Training a student to do in one step what the teacher does in two, repeatedly. |
| **Prosody** | Pitch, rhythm, stress, and intonation — the meaning-bearing music of speech. |
| **Reverse process** | The learned chain that removes noise, step by step, to produce an image. |
| **Score / $\nabla_x \log p(x)$** | The direction that makes a sample more probable; equivalent to noise prediction. |
| **SNR** | Signal-to-noise ratio, $\bar\alpha_t/(1-\bar\alpha_t)$. |
| **Spacetime patches** | Video patches cut across two spatial axes and time. Sora's representation. |
| **Timestep embedding** | A vector encoding $t$, injected into every block so the network knows the noise level. |
| **U-Net** | Convolutional encoder-decoder with skip connections; the original diffusion denoiser. |
| **Unified fusion** | Tokenising every modality and training one model jointly (GPT-4o, Gemini). |
| **VAE** | Variational Autoencoder — the learned compressor between pixels and latent. |
| **$v$-prediction** | Predicting a velocity mixing $\epsilon$ and $x_0$; well-scaled at all timesteps. |
| **ViT** | Vision Transformer — a transformer over image patches. |
| **Whisper** | OpenAI's speech-recognition model. |
| **Zero-convolution** | A convolution initialised to zero weights: outputs nothing, still receives gradients. |
| **Zero-terminal-SNR** | Rescaling the schedule so $\bar\alpha_T = 0$, fixing the "can't make dark images" bug. |
| **Zero-shot classification** | Classifying by embedding class names as sentences; no labelled training. |

---

## Check yourself

Twelve questions, easy → hard. Questions **9–12** combine two or more concepts.

1. In one sentence, why can a transformer model images and audio without architectural change?

2. Explain CLIP's training objective, and why it needs no human labels.

3. You have a 512×512×3 image and a 64×64×4 latent. Compute the compression factor.

4. What is the forward diffusion process, and how many parameters does it have?

5. Write the diffusion training loss and explain each symbol. What kind of loss is it, plainly?

6. Compute the CFG-guided prediction for $\epsilon_\varnothing = 0.25$, $\epsilon_{\text{text}} =
   0.40$, at $w = 1$ and $w = 8$. What happens to diversity as $w$ rises?

7. Why do zero-initialised convolutions in ControlNet still learn, when zero-initialising a whole
   network fails?

8. Name the three conditioning "doors" and what each controls.

9. **(Combines two concepts)** Your text-to-image model cannot produce a genuinely dark image, no
   matter the prompt. Diagnose the cause precisely, explain the mechanism, and give the two-part
   fix — including *why* the second part is required.

10. **(Combines two concepts)** DDPM needs ~1000 steps, DPM++ ~20, and distilled models 1–4.
    Explain what changed at each transition, which required retraining, and what is given up.

11. **(Combines three concepts)** You must build a real-time voice assistant that also generates
    images on request, running on a tight latency budget. Design it, name every technique you'd use
    from this lecture, and identify the two dominant costs.

12. **(Combines four concepts)** Your team ships a text-to-image product. FID is excellent, CLIP-score
    is excellent, users are unhappy: images look "samey" and often get the number of objects wrong.
    Diagnose all three observations, explain why the metrics missed the problems, and give concrete
    fixes.

<details><summary><b>Answers</b></summary>

**1.** A transformer takes a sequence of vectors and mixes them with attention; **nothing in that
computation refers to language**. So any data you can turn into a sequence of vectors — image
patches, spectrogram frames, sub-words — can be modelled by the same architecture. Multimodality is
a **tokenizer** problem, not an architecture problem.

**2.** Take a batch of $N$ image-text pairs, embed both with separate encoders into one L2-normalised
space, and form the $N \times N$ cosine-similarity matrix. The $N$ **diagonal** entries are true
pairs (positives); the $N^2 - N$ **off-diagonal** entries are negatives. Train with symmetric
cross-entropy (InfoNCE) over rows and columns, scaled by a learned temperature $\tau$: **maximise
the diagonal, minimise everything off it.**
It needs no human labels because **image-caption pairs already exist on the web** — alt-text,
captions — and the negatives are generated for free by pairing each image with the *other* captions
in the batch. 400M pairs, zero annotation budget.

**3.**
```
Pixels:  512 × 512 × 3 = 786,432 dims
Latent:   64 ×  64 × 4 =  16,384 dims
Ratio:   786,432 / 16,384 = 48×
```
Note it isn't 64× (the spatial reduction of $8 \times 8$) because the channel count *rises* from 3 to
4 to hold the compressed information.

**4.** The forward process starts from a real image and adds a small amount of Gaussian noise at each
of $T$ steps, following a fixed schedule $\beta_t$, until the image is indistinguishable from static:
$q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I)$.
**It has zero parameters.** Nothing is learned in the forward direction — it is a fixed recipe. All
learning is in reversing it.

**5.**
$$\mathcal{L} = \mathbb{E}\big[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\big]$$
- $\epsilon$ — the **actual** noise added (we generated it, so we know it).
- $\epsilon_\theta(x_t, t)$ — the network's **prediction** of that noise.
- $x_t$ — the noisy image at step $t$.
- $t$ — which timestep, i.e. how noisy; given to the network as input.
- $\mathbb{E}$ — averaged over random images, timesteps, and noise draws.
- $\|\cdot\|^2$ — squared norm.

Plainly: **it is ordinary MSE regression.** Predict a number, compare to the truth, square the
difference. No adversarial game, no likelihood bound — which is a large part of why diffusion trains
so much more reliably than GANs.

**6.**
```
difference = 0.40 - 0.25 = 0.15

w = 1:  ε̂ = 0.25 + 1 × 0.15 = 0.40   ← identical to plain conditional generation
w = 8:  ε̂ = 0.25 + 8 × 0.15 = 1.45   ← strongly extrapolated past the conditional
```
As $w$ rises, **diversity collapses**. Guidance amplifies the prompt direction, pushing every sample
toward the most *prototypical* image for that prompt, so a hundred samples converge on slight
variations of one composition. It concentrates probability mass on the modes — the same
faithfulness-versus-diversity trade-off as greedy decoding versus sampling in lecture 3. Past the
sweet spot ($w \approx 7$–8) images also become over-saturated and artefact-y.

**7.** Because a weight's gradient does not depend on the weight's own value. For $y = w \cdot x$:
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial y} \cdot x$$
With $w = 0$, the **output** is zero (so the block is an exact identity and cannot damage the base
model), but the **gradient** is $\frac{\partial\mathcal{L}}{\partial y} \cdot x$, which is non-zero.
It learns from the first step.
Zero-initialising an **entire network** fails for a different reason — the **symmetry problem**:
every neuron in a layer receives an identical gradient and can never differentiate from its
siblings. ControlNet avoids this because the zero-init is a **single $1\times1$ conv sitting between
components that are already richly and asymmetrically initialised** (the frozen block and the cloned
copy).

**8.**
| Door | Mechanism | Controls |
|---|---|---|
| **Text** | Cross-attention (image patches = queries, prompt = keys/values) | **Content** |
| **ControlNet** | Trainable clone of the encoder, zero-conv wrapped, features added back | **Structure** (pose, depth, edges) |
| **IP-Adapter** | Separate, decoupled cross-attention on a reference image | **Style** |

They **compose** on one frozen base model, because the three signals are largely orthogonal: what is
in the picture, where things are, and what it looks like.

**9.** The cause is **zero-terminal-SNR**.

*Mechanism.* Standard schedules leave $\bar\alpha_T \neq 0$ — illustrated here with ~0.004 rather
than exactly 0 (⚠️ this specific number is this document's own illustrative choice, not printed on
the slide — see §8's caveat). So during training, at $t = T$:
```
x_T = √0.004 x_0 + √0.996 ε = 0.063 x_0 + 0.998 ε
```
A ~6% ghost of the real image survives, and in particular its **mean brightness leaks through**. The
model learns that at $t = T$ the overall brightness is already hinted at in its input.

At sampling time we start from $x_T \sim \mathcal{N}(0, I)$ — **pure** noise, mean exactly zero.
There is no brightness hint, and a mean-zero input reads to the model as "medium brightness". So it
can never commit to a very dark or very bright image. **A train/test mismatch**, invisible for years
because the images still looked fine — just never *very* dark.

*The two-part fix.* **(1)** Rescale $\beta_t$ so $\mathrm{SNR}(T) = 0$ exactly, making training and
sampling see the same thing. **(2)** Switch to **v-prediction** — and this part is *required*, not
optional, because at exactly $\mathrm{SNR} = 0$ there is no signal at all in the input, and
$\epsilon$-prediction becomes degenerate (it's already ill-conditioned near $t = 0$ for the mirror
reason; at zero terminal SNR the problem appears at the other end). v-pred stays well-scaled across
all $t$.

**10.**
```
DDPM ~1000  ──►  DDIM ~50     : reformulate the stochastic reverse process as a
                                DETERMINISTIC ODE trajectory. Smooth paths can be
                                skipped along. NO RETRAINING — same network, new
                                sampling rule. Nothing given up.

DDIM ~50    ──►  DPM++ ~20    : apply higher-order ODE solvers that follow the
                                trajectory's curvature rather than just its slope.
                                NO RETRAINING. Nothing given up.

DPM++ ~20   ──►  distilled 1-4: RETRAINING REQUIRED. Below ~10 steps the
                                discretisation error of *any* solver shows up as
                                blur — the error is in the discretisation itself.
                                You must change what the network COMPUTES:
                                progressive distillation (student does in 1 step
                                what teacher does in 2), consistency models (learn
                                the trajectory's endpoint directly), or adversarial
                                distillation (+ GAN discriminator to restore the
                                sharpness that MSE regression blurs away).
```
**What is given up:** distilled models plateau slightly *below* the solver's top-end quality
(~96% vs ~97% in the slide's chart) and cannot be pushed further with more compute, because they
were trained to take giant steps. They also lose **generality** — less responsive to guidance scale,
weaker on unusual prompts, less controllable. Solvers are a free inference trick; distillation is a
second training stage that trades a little quality and flexibility for 10–50× speed.

**11.** *Architecture.*

**Voice path — native speech-to-speech, not a cascade.** A Whisper→LLM→TTS cascade costs ~2.2 s
against a ~200 ms conversational budget, and the text bottleneck discards tone, emotion, and speaker
identity. A native model reaches ~0.3 s, preserves prosody, and is interruptible. If you cannot
train one, rent one (GPT-4o voice, Gemini Live) — the cascade is the fallback you can build yourself
and debug via transcripts, at a real quality cost.

**Image path — every latency technique in Part 2, stacked.**
- **Latent diffusion**: run the loop in a 64×64×4 latent, not 512×512×3 pixels. **48× less work per
  step.**
- **Distillation / consistency (LCM, SDXL-Turbo)**: 1–4 steps instead of 20–50. Accept the small
  quality and controllability cost — this is a real-time product.
- **DiT or U-Net backbone** as available; note DiT scales better if you're training your own.
- **Constrained decoding** (lecture 3) on any tool call the assistant makes to trigger image
  generation, so the call cannot fail to parse.

**The two dominant costs.**
1. **The denoising loop.** Even at 4 steps with CFG doubling each one, that's 8 network passes per
   image. It dominates image latency, which is why steps and latent size were the two things to
   attack.
2. **Time-to-first-token / first-audio on the voice path.** Perceived responsiveness is set by when
   the user hears *something*, not total completion time — so streaming the audio output matters as
   much as raw throughput.

*(Worth noting: CFG doubles per-step compute, so if latency is critical, distilled models that need
little or no guidance are doubly valuable.)*

**12.** Three observations, three distinct diagnoses.

**(a) "Samey" images = low recall / mode collapse.** Almost certainly the **guidance scale is too
high**. CFG extrapolates past the conditional prediction toward the most prototypical image for the
prompt, so diversity collapses as $w$ rises — at $w = 15$ every sample is a variation on one
composition. **Fix:** lower $w$ toward the 7–8 sweet spot and measure. Also check whether you are
using a **distilled** model, which is often less responsive to guidance and can be inherently less
diverse.

**(b) Wrong object counts = compositional failure.** This traces back to **CLIP**. CLIP is close to
a **bag of concepts** rather than a parser of relationships — it scores "three red cubes" and "two
red cubes" almost identically, and mixes up attribute binding. Since CLIP is the text encoder, that
weakness propagates straight into generation. **Fix:** a stronger text encoder (T5 alongside or
instead of CLIP, as SD3 does), and evaluate on **GenEval / T2I-CompBench**, which exist precisely to
probe counting and spatial binding.

**(c) Why both metrics missed it.**
- **FID is blind to the prompt entirely.** It compares the *distribution* of generated images against
  real ones. A model producing beautiful, diverse images that ignore every prompt scores an excellent
  FID. It cannot see a counting error.
- **CLIP-score is gameable and shares CLIP's blind spots.** Your model *uses CLIP as its text
  encoder*, and you're grading it with CLIP — so the metric and the model share a component and a
  weakness. CLIP-score cannot detect compositional errors because CLIP itself cannot. This is
  Goodhart's law (lecture 2): the measure became the target.
- **Neither metric separates quality from diversity.** FID conflates them into one number, so mode
  collapse can be masked by high per-image fidelity.

**Concrete fixes.**
1. **Report a basket, not a number.** Add **precision/recall** (which would have flagged the
   samey-ness immediately as high precision, low recall), **GenEval** (which would have flagged the
   counting), and **human preference** as the tiebreaker.
2. **Lower the guidance scale** and re-measure recall.
3. **Upgrade the text encoder** for composition.
4. **Train or adopt a human-preference reward model** (ImageReward, PickScore, HPSv2) and use it to
   rank and fine-tune — while remembering it is a *proxy* and can itself be reward-hacked into an
   over-saturated "AI aesthetic".

</details>

---

## Going deeper

### Tier 1 — read these

1. **"Denoising Diffusion Probabilistic Models"** — Ho, Jain & Abbeel, NeurIPS 2020.
   [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
   *Sections 5–7 in full — the paper that made image diffusion work.* **Difficulty: medium-hard**
   for the derivation, **easy** for the algorithm. Skip to Algorithm 1 and 2 (half a page) to see
   how simple the actual training and sampling loops are, then go back for the theory.

2. **"High-Resolution Image Synthesis with Latent Diffusion Models"** (Stable Diffusion) — Rombach
   et al., CVPR 2022. [arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
   *Section 12 — the change that put image generation on a consumer GPU.* **Difficulty: medium.**

3. **"Learning Transferable Visual Models From Natural Language Supervision"** (CLIP) — Radford et
   al., ICML 2021. [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
   *Section 2. Cited directly on slide 5.* **Difficulty: medium.** Long, but sections 2 and 3 carry
   the idea. The zero-shot results are worth studying for how surprising they were at the time.

4. **"Classifier-Free Diffusion Guidance"** — Ho & Salimans, 2022.
   [arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)
   *Section 9.* **Difficulty: easy-medium.** Short. The idea of training one model to be both
   conditional and unconditional (by randomly dropping the caption) is the elegant bit.

5. **"Scalable Diffusion Models with Transformers"** (DiT) — Peebles & Xie, ICCV 2023.
   [arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
   *Sections 15–16, including adaLN-Zero.* **Difficulty: medium.** The scaling plots are the whole
   argument — this is lecture 1's scaling laws re-derived for images.

### Tier 2 — the engineering

6. **"Denoising Diffusion Implicit Models"** (DDIM) — Song, Meng & Ermon. Preprint 2020, published
   ICLR 2021 (the slide's own footer cites the ICLR 2021 venue).
   [arxiv.org/abs/2010.02502](https://arxiv.org/abs/2010.02502)
   *Section 10 — determinism, step-skipping, and the inversion that section 19 depends on.*
   **Difficulty: medium-hard.**

7. **"Progressive Distillation for Fast Sampling of Diffusion Models"** — Salimans & Ho, ICLR 2022.
   [arxiv.org/abs/2202.00512](https://arxiv.org/abs/2202.00512)
   *Section 11, and where v-prediction was introduced. Cited on slide 19.* **Difficulty: medium.**

8. **"Consistency Models"** — Song et al., 2023.
   [arxiv.org/abs/2303.01469](https://arxiv.org/abs/2303.01469)
   *Section 11 — one-step generation via a self-consistency constraint.* **Difficulty: hard.** The
   idea is elegant; the theory is dense. Read the intro and Figure 1 at minimum.

9. **"Adding Conditional Control to Text-to-Image Diffusion Models"** (ControlNet) — Zhang, Rao &
   Agrawala, ICCV 2023. [arxiv.org/abs/2302.05543](https://arxiv.org/abs/2302.05543)
   *Section 18. Cited on slide 27.* **Difficulty: easy-medium.** The zero-convolution argument is a
   page and worth reading in the original.

10. **"Common Diffusion Noise Schedules and Sample Steps are Flawed"** — Lin et al., 2023.
    [arxiv.org/abs/2305.08891](https://arxiv.org/abs/2305.08891)
    *Section 8's zero-terminal-SNR bug, in full.* **Difficulty: medium.** A model of careful
    debugging — a subtle train/test mismatch nobody had noticed, found and fixed.

11. **"Efficient Diffusion Training via Min-SNR Weighting Strategy"** — Hang et al., ICCV 2023.
    [arxiv.org/abs/2303.09556](https://arxiv.org/abs/2303.09556)
    *Section 8's loss weighting.* **Difficulty: medium.**

12. **"SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations"** — Meng et
    al., ICLR 2022. [arxiv.org/abs/2108.01073](https://arxiv.org/abs/2108.01073)
    *Section 19 — the img2img trick.* **Difficulty: medium.** Remarkably simple once you see it.

### Tier 3 — multimodal and video

13. **"Visual Instruction Tuning"** (LLaVA) — Liu et al., NeurIPS 2023.
    [arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485)
    *Section 3's modular design.* **Difficulty: easy-medium.** Notable for how *little* it took —
    a linear projection and a few hundred GPU-hours.

14. **"Video generation models as world simulators"** (Sora technical report) — OpenAI, 2024.
    [openai.com/research/video-generation-models-as-world-simulators](https://openai.com/research/video-generation-models-as-world-simulators)
    *Section 17's spacetime patches.* **Difficulty: easy.** A report rather than a paper — light on
    detail, but it states the spacetime-patch idea clearly.

15. **"An Image is Worth 16x16 Words"** (ViT) — Dosovitskiy et al., ICLR 2021.
    [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
    *The origin of patchifying — section 1's whole premise.* **Difficulty: medium.**

16. **"U-Net: Convolutional Networks for Biomedical Image Segmentation"** — Ronneberger, Fischer &
    Brox, MICCAI 2015. [arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)
    *Section 14. Cited on slide 21.* **Difficulty: easy.** A 2015 medical-imaging paper that became
    the backbone of image generation — worth reading for that alone.

17. **"Large Language Diffusion Models"** (LLaDA) — Nie et al., 2025.
    [arxiv.org/abs/2502.09992](https://arxiv.org/abs/2502.09992)
    *Sections 20–21.* **Difficulty: medium.**
    > ⚠️ Fast-moving area — check for newer results before relying on specific claims.

### Tier 4 — build and see

18. **"What are Diffusion Models?"** — Lilian Weng.
    [lilianweng.github.io/posts/2021-07-11-diffusion-models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
    *Not a paper — the best single explanation of diffusion mathematics available, with every
    derivation worked.* **Difficulty: medium.** **If sections 6–8 felt dense, read this next.**

19. **"The Annotated Diffusion Model"** — Hugging Face.
    [huggingface.co/blog/annotated-diffusion](https://huggingface.co/blog/annotated-diffusion)
    *DDPM implemented line by line in PyTorch, with the maths beside each line.* **Difficulty:
    medium.** **The highest-value item here.** Training a tiny diffusion model on a small dataset
    and watching noise resolve into images teaches Part 2 faster than any paper.

20. **Hugging Face `diffusers`** — [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)
    *Every sampler (DDIM, DPM++, LCM), ControlNet, IP-Adapter, img2img, and inpainting, as working
    code.* **Difficulty: easy to use, medium to read.** Swap the scheduler on one pipeline and watch
    section 10's step-count claims hold up in real time.

21. **"Diffusion Models Beat GANs on Image Synthesis"** — Dhariwal & Nichol, 2021.
    [arxiv.org/abs/2105.05233](https://arxiv.org/abs/2105.05233)
    *The paper that established diffusion over GANs, and classifier guidance's origin.*
    **Difficulty: medium.** Useful historical context for why the field switched.
    > Note the author order carefully — this is a different paper from #22 below, by the same two
    > authors in reversed order.

22. **"Improved Denoising Diffusion Probabilistic Models"** — Nichol & Dhariwal, ICML 2021.
    [arxiv.org/abs/2102.09672](https://arxiv.org/abs/2102.09672)
    *Section 8. Named directly on slide_016.jpg's footer — the origin of the cosine noise
    schedule.* **Difficulty: medium.**

23. **"GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"** (FID) —
    Heusel et al., NeurIPS 2017. [arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)
    *Section 22. Named on slide_033.jpg's footer — the origin of the Fréchet Inception Distance,
    the most widely used image-generation quality metric.* **Difficulty: medium.**

24. **"CLIPScore: A Reference-free Evaluation Metric for Image Captioning"** — Hessel et al.,
    EMNLP 2021. [arxiv.org/abs/2104.08718](https://arxiv.org/abs/2104.08718)
    *Section 22. Named on slide_033.jpg's footer alongside FID — text-image alignment scoring
    without a reference caption.* **Difficulty: easy-medium.**

25. **"Improved Precision and Recall Metric for Assessing Generative Models"** — Kynkäänniemi et
    al., NeurIPS 2019. [arxiv.org/abs/1904.06991](https://arxiv.org/abs/1904.06991)
    *Section 22. Named on slide_033.jpg's footer — separates "does it look real" (precision) from
    "does it cover the real distribution" (recall), a distinction FID alone collapses into one
    number.* **Difficulty: medium.**

> 💡 **If you only do three things:** read Lilian Weng's diffusion post alongside sections 6–8 of
> these notes; work through *The Annotated Diffusion Model* and train a tiny model on a small dataset
> so you watch static become images; then load a Stable Diffusion pipeline in `diffusers` and change
> exactly three things one at a time — the **sampler** (DDIM vs DPM++ vs LCM), the **guidance scale**
> (2, 7, 15), and the **img2img strength** (0.2, 0.5, 0.9). Each of those three dials is a section of
> this document, and seeing them move is worth more than re-reading it.




