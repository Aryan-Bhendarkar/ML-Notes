# GenAI & LLM — Amazon ML Summer School

Self-study notes built from the verified slide extraction in
[`slides_deduped/`](../../slides_deduped/) (cross-checked against [`output/`](../../output/) raw
frames where noted), following [`NOTES_PIPELINE.md`](../../NOTES_PIPELINE.md).

These are **teaching documents, not summaries**. They define every term before using it, derive
every formula rather than asserting it, and work every example through to a final number. They are
written to be the only thing you need to read to master the lecture.

This module has gone through a full `QUALITY_REVIEW_PIPELINE.md` audit pass — see
[`QUALITY_REVIEW.md`](QUALITY_REVIEW.md) for the complete findings and fixes.

---

## Index

| # | Notes | Source deck | Status | Words | Covers |
|---|---|---|---|---|---|
| 01 | [genai-llm-01.md](genai-llm-01.md) | `Lecture_14 - Module 5 Generative AI and LLMs Part 1` (**31 real pages**, 66 deduped frames) | ✅ Complete · audit-fixed | ~30,700 | **Foundations, Scale, and the Transformer**: why GenAI now (adoption, labor-market, and capital-spend context) · self-attention, Q/K/V, scaled dot-product, multi-head attention derived from scratch · positional encoding · scaling laws (Kaplan/Chinchilla, the compute-optimal U-curve) · emergent abilities (the debate) · the quadratic-attention wall and its fixes (Mamba, RWKV, Jamba) · Mixture-of-Experts routing · the modern model taxonomy |
| 02 | [genai-llm-02.md](genai-llm-02.md) | `Lecture_15 - Module 5 Generative AI and LLMs Part 2` (**22 real slides**) | ✅ Complete · audit-fixed | ~21,700 | **Alignment & Training**: pre-training vs. instruction tuning vs. preference tuning as three distinct stages · RLHF (reward modeling, the KL-penalty tradeoff) · **DPO derived in full**, worked numerically off the slide's own chosen/rejected example · **GRPO derived in full** · PPO/DPO/GRPO compared head to head · IPO and KTO as DPO variants · the synthetic-data flywheel (WizardLM/Evol-Instruct) |
| 03 | [genai-llm-03.md](genai-llm-03.md) | `Lecture_16 - Module 5 Generative AI and LLMs Part 3` (**45 real slides**) | ✅ Complete · audit-fixed | ~32,000 | **Using & Serving LLMs — seven levers on three axes**: decoding strategies (temperature, top-k, top-p, nucleus sampling) · prompting and in-context learning · chain-of-thought and self-consistency · PEFT (LoRA, QLoRA, TIES) · embeddings and Matryoshka Representation Learning · RAG (the retrieval waterfall, RRF, lost-in-the-middle) · reasoning at inference time (test-time compute) · efficient serving (quantization, KV-cache and GQA, continuous batching, speculative decoding) |
| 04 | [genai-llm-04.md](genai-llm-04.md) | `Lecture_17 - Module 5 Generative AI and LLMs Part 4` (**34 real slides**) | ✅ Complete · audit-fixed | ~30,900 | **Beyond Text: Multimodal Models and Diffusion** — how the same transformer learns to see, hear, and speak (CLIP, modular vs. unified multimodal architectures, speech cascades), and a second generative family that produces by denoising: DDPM forward/reverse process derived in full, noise schedules, guidance, samplers (DDIM), distillation, latent diffusion, DiT, ControlNet, image editing, and generative-model evaluation (FID, CLIPScore, Precision/Recall) |
| | | | **Total** | **~115,300** | |

⚠️ **The old version of this README mis-described Lecture 17** as covering "efficiency
(quantization, distillation, KV-cache), safety..., and the emerging frontier of tool-augmented and
multimodal models." That is **factually wrong** — confirmed by directly viewing the deck's own title
slide, which reads *"Beyond text: multimodal models and diffusion."* The quantization/KV-cache/
efficient-serving material actually lives in **Lecture 16** (Part 3), not Lecture 17. This is now
corrected throughout this README and cross-checked against every lecture file's own frontmatter.

**Prerequisites.** This module leans on [`Deep Neural Networks`](../Deep%20Neural%20Networks/) for
backpropagation and optimizer mechanics, and on
[`Dimensionality Reduction`](../Dimensionality%20Reduction/) /
[`Unsupervised Learning`](../Unsupervised%20Learning/) for the generative-modeling foundations (VAEs,
GANs, diffusion, flow matching) that Lecture 17's diffusion section builds on directly rather than
re-deriving. Read those first if you haven't.

---

## Capture quality

### ✅ Lecture 14 — good, with a real early-slide gap now closed

66 deduped frames confirmed against the deck's own page counter to cover all 31 real pages — a
large upgrade from an earlier PDF-screenshot draft that had captured only ~20 of those 31 pages.
Every one of the 66 frames was read directly.

**Real gap found and closed by this module's quality review:** four previously-uncaptured pages
turned out to contain real taught content — page 2 ("You've already used Generative AI today"),
page 3 ("Everyone has feelings about this"), page 4 ("The biggest companies on Earth are betting
hundreds of billions on this," including Amazon/Microsoft/OpenAI/Anthropic funding and IPO figures,
now flagged for verification since they describe 2026 events), and page 7 ("You know deep learning.
Now meet the model that ate NLP."). These are now covered in a dedicated subsection early in the
notes. **Pages 22–25 are confirmed never presented** — the deck's own page counter jumps 21→26 — not
a capture gap.

One MoE slide (the router/experts diagram) was recaptured 30+ times across the deduped range as the
instructor progressively built up the animation — the notes call this out explicitly rather than
treating it as 30 distinct slides.

**The instructor is not named** anywhere in the recording — the title slide (`slide_001.jpg`) shows
only "AMAZON ML SUMMER SCHOOL 2026 · MODULE 6" branding and an unlabeled webcam tile. ("Module 6" is
the live course's own internal numbering and is unrelated to this project's module grouping.)

### ✅ Lecture 15 — excellent, instructor now identified

22 real content slides (23 deduped files, one of which is a non-content divider), title through
wrap-up. The gap versus an earlier 18-slide draft is mostly structural — three bare "Part N" divider
slides and two tail slides that are different interactive-stepper states of the opening example, not
new teaching content. Every slide was read directly.

This module's quality review corrected several numbers the earlier draft had transcribed from the
wrong animation state: a fabricated example chart in the RLHF section, the KL-penalty β value, and
the DPO worked example's chosen/rejected/gradient-weight numbers and "after" margin.

Both instructors named on this and Lecture 17's title slides are the same person: **Harsh Agarwal**,
confirmed directly from the nameplate overlay on `slide_001.jpg`.

### ✅ Lecture 16 — good, with three numeric errors caught and fixed

45 real content slides (46 deduped files), against an earlier PDF-screenshot draft's 34 unique
slides. Topic coverage against all 45 slides is essentially complete — every slide maps to an
existing section. This module's mandatory citation-and-number sweep found **three confirmed numeric
errors** where the earlier draft had transcribed a *different* interactive-slider state than the one
actually preserved by deduplication: the Matryoshka-embedding example (§13), the context-window
reach/influence figures (§18), and the KV-cache memory ledger (§24, propagated to four sites). All
four are now corrected and cross-checked against the slide images.

One genuinely unrecoverable-looking gap turned out to be recoverable: §12's "normalise: off"
embedding state doesn't survive deduplication in `slides_deduped/`, but the raw frame
`output/Lecture_16.../slide_044.jpg` does show it — the notes now cite that raw frame directly
instead of flagging the state as lost.

**The instructor is not named** — the title slide shows an Amazon logo and an unlabeled webcam tile,
matching Lecture 14.

### ✅ Lecture 17 — excellent, and the module's cleanest file

34 real content slides (35 deduped files), up from an earlier draft's ~29 distinct slides. Every
slide was read directly, and content coverage is essentially complete — this is the cleanest of the
four files in this module by that measure. The deck's own title slide reads exactly *"Beyond text:
multimodal models and diffusion"* — direct confirmation that this lecture is about multimodality and
diffusion, not the efficiency/deployment topics an earlier version of this README wrongly attributed
to it (those belong to Lecture 16).

This module's quality review found the file was missing its entire `## Interview prep — Amazon
Applied Scientist` section and had zero interactive spec blocks despite the deck shipping several
ready-to-convert widgets (CLIP toggle, fusion toggle, diffusion-step slider, guidance-scale slider,
U-Net-vs-DiT toggle, and more) — both gaps are now closed.

**The instructor is Harsh Agarwal** — same person as Lecture 15, confirmed on `slide_001.jpg`'s
nameplate. Two of this module's four lectures share an instructor; the other two (14 and 16) have no
instructor named in the recording at all — a genuine fact about this module's capture, not an
inconsistency to resolve.

---

## What's in Part 1 — Foundations, Scale, and the Transformer

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — capture notes (31 real pages, 66 deduped frames, four previously-missing early
pages now covered) · why GenAI now: adoption numbers, the labor-market debate (~500k tech layoffs
2023–2025 framed as "replace with LLM" then walked back to "humans + AI"), and the capital context
(Amazon, Microsoft, OpenAI, and Anthropic funding/IPO figures, flagged for verification as 2026
events) · the deck's own 5-part agenda

**Core sections**
- Vectors, dot products, and embeddings as the substrate everything else is built on
- Self-attention derived from first principles: queries, keys, values, scaled dot-product, why the
  scaling factor exists
- Multi-head attention: why one attention pattern isn't enough
- Positional encoding: why attention alone is permutation-invariant and needs help
- Scaling laws: the Kaplan and Chinchilla results, the compute-optimal U-curve, and why "bigger is
  better" is actually "bigger *and more data* is better, in a specific ratio"
- Emergent abilities: the capability-jump debate, both sides
- The quadratic-attention wall: why long context is expensive, and the alternatives (Mamba, RWKV,
  Jamba) that trade some expressiveness for linear scaling
- Mixture-of-Experts: sparse routing, why it lets you scale parameters without scaling compute
  proportionally — including the single router/experts slide that was recaptured 30+ times as the
  instructor built up its animation
- The modern model taxonomy: dense vs. MoE, open vs. closed, the current landscape

**Closing** — interview prep (Amazon Applied Scientist): depth probes, whiteboard-ready derivations,
an applied Amazon scenario with explicit Leadership Principle ties, ranked interview questions ·
glossary · check-yourself questions · ranked resources

**Interactive specs:** 4 — attention weights per token, the Chinchilla compute-optimal U-curve,
KV-cache-vs-context-length, MoE router top-k selection.

</details>

---

## What's in Part 2 — Alignment & Training

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — capture notes (22 real slides) · why pre-training alone isn't enough: a raw
next-token predictor vs. an instruction-following assistant

**Core sections**
- Instruction tuning (SFT): what it is, and its limits
- RLHF: reward modeling from human preference pairs, the KL-divergence penalty against the reference
  policy (β = 0.30, derived and worked numerically), and why unconstrained reward maximization
  degenerates
- DPO (Direct Preference Optimization): derived from the RLHF objective's closed-form optimal
  policy, avoiding the need for a separate reward model — worked fully numerically off the slide's
  own chosen = 1.4 / rejected = −1.2 example, gradient weight ≈ 0.07, margin improving to 1.8
- GRPO (Group Relative Policy Optimization): derived in full, worked numerically
- PPO vs. DPO vs. GRPO: a head-to-head comparison table
- IPO and KTO: two DPO variants addressing overfitting and requiring less paired data respectively
- The synthetic-data flywheel: WizardLM / Evol-Instruct as a case study in bootstrapping training
  data from a model's own outputs

**Closing** — interview prep (Amazon Applied Scientist) · glossary · check-yourself questions ·
ranked resources including Christiano et al. 2017 (the original RLHF paper), Xu et al. 2023/ICLR
2024 (WizardLM), and Azar et al. 2024 (IPO)

**Interactive specs:** present in the file (worked-example fallback content doubles as the primary
teaching device for the DPO/GRPO derivations).

</details>

---

## What's in Part 3 — Using & Serving LLMs: Seven Levers on Three Axes

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — capture notes (45 real slides, three confirmed numeric errors from
animation-state mistranscription found and fixed by this module's quality review) · the "seven
levers on three axes" framing that organizes the whole lecture: decoding, prompting/adaptation, and
serving

**Core sections**
- Decoding strategies: temperature, top-k, top-p (nucleus sampling), worked numerically
- Prompting and in-context learning: how few-shot examples steer a frozen model
- Chain-of-thought and self-consistency reasoning
- PEFT: LoRA, QLoRA, and TIES model-merging
- Embeddings and Matryoshka Representation Learning (MRL): truncating a 3072-d embedding to 64-d
  (48× smaller) while cosine similarity stays at 0.84 — because MRL training front-loads importance
  into the early dimensions, unlike naive truncation
- RAG: the retrieval waterfall, Reciprocal Rank Fusion (RRF), and the lost-in-the-middle effect
- Reasoning at inference time: test-time compute as a lever independent of model size
- Efficient serving: quantization and its accuracy cliff, the KV-cache memory formula derived and
  applied (32K-token context, 10.0 GiB per user at 70B/GQA), Grouped-Query Attention's 8× cache
  saving, continuous batching, and speculative decoding

**Closing** — interview prep (Amazon Applied Scientist), including a hard combines-two-concepts
question on serving Llama-3.1-70B on a single 80 GB GPU (quantization arithmetic + KV-cache formula
→ ≈4 concurrent users) · whiteboard-ready derivations (top-p sampling, the KV-cache formula, static
vs. continuous batching GPU-step accounting) · glossary · check-yourself questions · ranked resources

**Interactive specs:** 5 — temperature/top-p, the Matryoshka dimension slider, the quantization
bit-depth accuracy cliff, and continuous-batching-vs-static scheduling, among others.

</details>

---

## What's in Part 4 — Beyond Text: Multimodal Models and Diffusion

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — capture notes (34 real slides, the cleanest capture in the module) · the deck's
own framing, taken verbatim from its title slide: *"How the same transformer learns to see, hear,
and speak — and a second family that generates by denoising."*

**PART A — Multimodal Models**
- CLIP: contrastive image-text pretraining, and why it became the default vision-language bridge
- Modular vs. unified multimodal architectures
- Speech: cascade architectures (ASR → LLM → TTS) and their latency tradeoffs

**PART B — Diffusion Models**
- The forward diffusion process derived: variance preservation, the closed-form ᾱ_t shortcut, noise
  schedules (linear, cosine — Nichol & Dhariwal, ICML 2021), worked on a scalar example
- The reverse process: why it's Gaussian, ε-prediction vs. x₀-prediction
- The training objective: ELBO derived from the VAE framework down to the boxed simple MSE loss
- Noise prediction = score prediction, proved and computed on a 2D Gaussian example
- Sampling: the ancestral-sampling algorithm, and acceleration via DDIM (ICLR 2021), distillation,
  and consistency models
- Guidance, samplers, latent diffusion, DiT, ControlNet, and image editing
- Evaluating generative models: FID (Heusel et al., NeurIPS 2017), CLIPScore (Hessel et al., EMNLP
  2021), and Precision/Recall (Kynkäänniemi et al., NeurIPS 2019)

**Closing** — interview prep (Amazon Applied Scientist), added in full by this module's quality
review · glossary · check-yourself questions · ranked resources

**Interactive specs:** 7 — a CLIP toggle, a fusion toggle, an audio-cascade toggle with a latency
bar, a diffusion-step slider, a U-Net-vs-DiT toggle, a guidance-scale slider, and a "replay
generation" button — all added by this module's quality review against the deck's own
ready-to-convert widget slides.

</details>

---

## Reading guide

The four parts total ~115,300 words and form one escalating argument: the transformer's mechanics
(Part 1) → how it's turned into an assistant (Part 2) → how you actually use and serve it in
production (Part 3) → what it looks like once you go past pure text (Part 4).

**Prerequisites.** [`Deep Neural Networks`](../Deep%20Neural%20Networks/) for backprop/optimizer
mechanics; [`Unsupervised Learning`](../Unsupervised%20Learning/) for VAEs, GANs, and the
generative-modeling vocabulary Part 4's diffusion section assumes fresh.

**If you're new to transformers:** start with Part 1 in full, then Part 2's RLHF/DPO/GRPO sections.

**If you want to build with LLMs:** read Part 3 (prompting, RAG, serving) and Part 4 (multimodal
architectures).

**If you're preparing for interviews:** all four parts now have a full `## Interview prep — Amazon
Applied Scientist` section (depth probes, whiteboard derivations, an applied Amazon scenario tied to
explicit Leadership Principles, and ranked interview questions with model answers). Work all four
before an interview, not just the ones matching your strongest area — Part 3's serving-arithmetic
questions and Part 2's DPO/GRPO derivations are the two most likely to come up as whiteboard
questions.

**Second pass.** Work every 🧪 worked example on paper before reading the solution. The highest-value
ones: Part 1's attention mechanics, Part 2's DPO and GRPO derivations, Part 3's KV-cache formula and
Matryoshka truncation example, Part 4's noise-prediction-equals-score-prediction proof.

**The three questions this module is most likely to be examined on:**
1. *"Derive DPO from the RLHF objective"* (Part 2)
2. *"Walk through the KV-cache memory formula and what GQA saves"* (Part 3 §24)
3. *"How is a diffusion model related to a VAE, and what does the network actually predict?"*
   (Part 4)

**Callout legend**

| | Meaning |
|---|---|
| 📚 **Background** | A prerequisite the slide assumed you already knew |
| 💡 **Key insight** | The one sentence from that section worth memorising |
| ⚠️ **Careful** | A slide gap, an ambiguity, a place the standard presentation misleads, or a number flagged for independent verification |
| 🧪 **Worked example** | Real numbers computed to a real answer |
| 🎯 **Interview** | Phrased the way you would actually say it out loud |
| 🔬 **Research opportunity** | Where the field is genuinely still open |

---

## Key takeaway per lecture

| # | One-sentence takeaway |
|---|---|
| 01 | Self-attention is a differentiable, learned way to answer "which other tokens matter to me, and how much" for every token simultaneously, and everything else in this lecture is a consequence of that one mechanism's cost and behavior at scale: positional encoding exists because attention itself has no notion of order, scaling laws describe the (compute, data, parameter) triangle you must balance to spend a training budget optimally rather than just "make it bigger," the quadratic cost of attention in sequence length is the single structural reason alternative architectures (Mamba, RWKV, Jamba) exist at all, and Mixture-of-Experts routing is the mechanism that lets total parameter count keep growing without proportionally growing the compute spent per token. |
| 02 | Getting from a raw next-token predictor to a useful assistant is a two-stage alignment pipeline, not one step: instruction tuning teaches the model the *format* of being helpful, and preference optimization (RLHF, or its simpler closed-form descendants DPO and GRPO) teaches it *which* of two helpful-looking responses a human actually prefers — and the mathematically interesting part is that DPO's entire training signal collapses to a single number, the log-probability-ratio gap between the chosen and rejected response scaled by a sigmoid, meaning you never need to train a separate reward model or run reinforcement learning at all to get (approximately) the same alignment effect RLHF was designed to produce. |
| 03 | Once a model is trained, three genuinely separate levers determine whether it's actually useful in production — how you decode from it (temperature/top-p, chain-of-thought), how you adapt it to a task without retraining the whole thing (LoRA, RAG, in-context learning), and how you serve it efficiently at scale (quantization, KV-cache management, batching) — and the lecture's real "aha" is that the second and third levers are not independent: the exact same KV-cache memory formula that tells you how many tokens of context a GPU can hold is also the formula that tells you how many *concurrent users* it can serve, meaning a serving-efficiency decision (GQA, quantization) is simultaneously a product decision (how many customers this hardware supports). |
| 04 | The same transformer architecture that predicts the next token can be adapted, via contrastive pretraining like CLIP, to align an entirely different modality's representations with text — but generating new images, rather than just understanding existing ones, needs a fundamentally different mechanism, and diffusion models supply it by turning generation into hundreds of small, easy denoising steps whose training objective is provably equivalent to learning the score function (the gradient of the log-density) at every noise level, which is what makes both ancestral sampling and its accelerated variants (DDIM, distillation) actually work, and what connects this lecture's material all the way back to the score-based and VAE foundations built earlier in the course. |

---

*See [`QUALITY_REVIEW.md`](QUALITY_REVIEW.md) for this module's complete audit trail — every
finding, its source-slide verification, and its fix.*
