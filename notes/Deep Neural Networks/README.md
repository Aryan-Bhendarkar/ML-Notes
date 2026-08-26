# Deep Neural Networks — Amazon ML Summer School

Self-study notes built from the verified slide extraction in
[`output/`](../../output/), following [`NOTES_PIPELINE.md`](../../NOTES_PIPELINE.md).

These are **teaching documents, not summaries**. They define every term before using it, derive
every formula rather than asserting it, and work every example through to a final number. They are
written to be the only thing you need to read to master the lecture.

> ✅ This module has completed a full [`QUALITY_REVIEW_PIPELINE.md`](../../QUALITY_REVIEW_PIPELINE.md)
> pass — see [`QUALITY_REVIEW.md`](QUALITY_REVIEW.md) for the audit trail of what was checked and
> fixed. The word counts and glossary/count figures below reflect the post-fix state.

> ✅ This module has also completed an enhancement pass: all 16 existing `interactive` blocks
> (5 + 7 + 4 across the three files) were re-verified as well-formed and complete against a fresh
> scan of the raw slide captures in `output/` — no genuinely missed slider/toggle/animated-demo
> candidates were found. All 3 previously `⚠️`-flagged citations in Part 1 (Cybenko 1989, Hornik
> 1991, Dauphin et al. 2014, Dinh et al. 2017) were checked against primary sources and confirmed
> accurate. Three light cross-module "see also" pointers were added: Part 3 §6 and §13 (vanishing
> gradients / LSTM's fix) now point to `Sequential Learning` §3 and `GenAI & LLM`'s attention
> material, and Part 2 §9 (flat vs. sharp minima) now points forward to `Dimensionality Reduction`.

---

## Index

| # | Notes | Source deck | Status | Words | Covers |
|---|---|---|---|---|---|
| 01 | [deep-neural-networks-01.md](deep-neural-networks-01.md) | `Lecture_04 - Module 2 Deep Neural Network Part 1` (**43 slides**, 8 chapters) | ✅ Complete · verified | ~32,100 | Perceptron → MLP · Activations · Forward propagation · Loss functions · Backpropagation · Optimizers · Regularization · The training loop |
| 02 | [deep-neural-networks-02.md](deep-neural-networks-02.md) | `Lecture_05 - Module 2 Deep Neural Network Part 2` (**40 slides across two decks**) | ✅ Complete · 3 gaps flagged | ~36,400 | **Training Deep Networks:** initialization · vanishing/exploding gradients · BatchNorm & LayerNorm · regularization — then **CNNs:** convolution · padding/stride/pooling · receptive field · architectures · 1×1 convs · transfer learning · detection · style transfer |
| 03 | [deep-neural-networks-03.md](deep-neural-networks-03.md) | `Lecture_06 - Module 2 Deep Neural Network Part 3` (**35 slides** + a live notebook) | ✅ Complete · verified | ~30,000 | **Recurrent networks:** sequences · RNN & BPTT · LSTM & GRU · BiRNN · Seq2Seq — then **PyTorch:** tensors · autograd · `nn.Module` · `DataLoader` · the 5-step loop · a full hands-on demo |

**Prerequisites.** Part 1 of this module leans on [`Supervised Learning`](../Supervised%20Learning/):
bias–variance from Part 1 §8–11 (the entire regularization chapter is one application of it),
cross-entropy and softmax from Part 2 §6–7 (re-derived here from the deep-learning side, with the
gradient argument the earlier lecture omitted), and the optimiser lineage from Part 2 §13 (which
listed eight names; this lecture actually derives four of them).

**Read them in order.** Part 2 assumes Part 1's backpropagation, activation functions and optimizers
throughout, and Part 3 assumes Part 2's vanishing-gradient analysis — its central result (LSTM's
additive cell state) is explicitly the same trick as Part 2's residual connection, applied to time
instead of depth. The three files cross-reference each other by section number.

---

## Capture quality, lecture by lecture

### ✅ Lecture 04 — verifiable

Unlike Module 1's decks, **Lecture 04 is trivially verifiable**: every slide carries a page number in
its footer (`5 / 43`, `28 / 43`, …), so the deck length is known exactly rather than inferred. The
raw capture yields **42 of the 43 slides** — 33 content slides plus 9 chapter dividers.

The one gap is **slide 43**, which was never displayed: the recording ends at **44:11** with slide
42 (*Key takeaways*) still on screen. Given the deck's structure this is almost certainly a closing
card, but it has not been seen and is not described.

`slides_deduped/` held 49 images for this 43-slide deck — the same mid-build-duplicate failure mode
documented in [`../Supervised Learning/README.md`](../Supervised%20Learning/README.md). The notes were
written from `output/` using the timestamp-gap clustering procedure described there.

### ⚠️ Lecture 05 — three slides lost their bodies

This deck carries **no page numbers**, so its length is inferred rather than read off. It is also
**two decks in one session** — *Training Deep Networks* (0:00–20:41) and *Convolutional Neural
Networks* (20:41–48:08) — 40 distinct slide states between them.

Three slides animated their body in after their title landed, and the deck advanced before the next
sample: **Layer Normalization** (12:06), the **L2/Ridge half** of the L1 & L2 Regularization slide
(16:11), and **Output Size Formula** (27:34). All three are standard, unambiguous results that the
surrounding slides pin down, so the notes teach them in full under a **🩹 badge** meaning *"taught
from the standard result, not read off the slide."* Nothing is invented and nothing is skipped.

A weaker caveat: between **27:34 and 32:05** only the forced 90-second samples exist. They landed on
*Output Size Formula → Padding → Stride → Pooling*, a coherent sequence with no visible hole, but a
slide shown for well under 90 seconds in that window would not have been captured.

### ✅ Lecture 06 — the best capture in the module

153 raw frames over 57 minutes. Every content slide has a fully-built state, and the **10-minute live
notebook was captured densely enough that every cell's code *and every cell's printed output* is
legible** — which is why the notes reproduce the whole demo with its real numbers rather than
describing it.

Three minor notes: the *How an RNN Works* slide is a diagram and never writes the recurrence on
screen (reconstructed, 🩹); the GRU slide shows only the final interpolation equation, not the
reset-gate and candidate equations (reconstructed, 🩹); and the demo's last cell — a gradient-norm
check — was never run on camera, so its code is shown and its output is explicitly absent.

Note also that **Lecture 06 is not more feedforward material** despite its title: it is the module's
recurrent-networks and PyTorch lecture.

---

## What's in Part 1

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — capture note with the full chapter→slide map · 15 capabilities · 6 prerequisites
taught from zero (matrix shapes and the `nn.Linear` (in,out)-vs-(out,in) trap · derivatives, gradients
and **the chain rule** · exponentials and logs · this deck's layer-index notation · parameters vs
hyperparameters · **exponential moving averages and the $\frac{1}{1-\beta}$ window**) · the big
picture: what actually changed in 2012, and the eight chapters read as one argument where each exists
because the previous one left something broken

**Ch.1 — Architecture** (§1–4)
- The perceptron, and **XOR's non-separability proved** by contradiction rather than asserted
- Why the 1969 result cost the field a decade — and the three separate bottlenecks (algorithm 1986,
  data 2009, compute 2010) that a good "why so long?" answer names
- The MLP; **Universal Approximation stated with the three things it does not promise**
- **Why depth beats width at equal parameter count**, worked: 8,110 parameters buys either three
  layers of 50 or one layer of 73
- `h = σ(Wx+b)` with every shape named; why the matrix formulation *is* the GPU story
- **The 269,322-parameter count verified layer by layer**, and where 74.6% of it lives
- The concept check: three linear layers collapse to `Linear(100,10)` — **8,110 parameters buying
  1,010 parameters' worth of model** — and why "gradients won't flow" is the tempting wrong answer

**Ch.2 — Activations** (§5–8)
- The collapse blocked: why $\sigma(Wx) \ne W\sigma(x)$ is the whole mechanism
- **The four criteria**, and every historical transition read as one of them being fixed
- **The 0.25 derived** by maximising $s(1-s)$; why tanh's 1.0 still isn't enough past $|z|>2$
- The two legitimate sigmoid uses, and why the disqualifying property is what those uses need
- ReLU's four properties; **the dying-ReLU death spiral in five steps** + a runnable dead-unit check
- GELU computed at three points; ⚠️ an honest note that it beats ReLU in Transformers and not much else
- **The double-softmax bug worked numerically** — a 96%-confident prediction shown to the loss as
  56%, 16× the loss it deserved — and why it damages your *best-learned* examples hardest

**Ch.3 — Forward propagation** (§9–11)
- Inference vs training as the same computation with different stopping points
- **The complete forward pass by hand** to a final loss of 0.371, with a loss-scale table for
  reading the answer, and code to verify it
- The DAG, `requires_grad` semantics, and ⚠️ **the `total += loss` memory leak** that looks like
  your model being too big

**Ch.4 — Loss functions** (§12–13)
- Why "a single scalar" is a requirement, not a style choice
- The task→loss→output-layer table with **what breaks on each mismatch**
- **CE's $\hat y - y$ derived**, MSE's $2(\hat y-y)\sigma'(z)$ derived, and the gap computed: **50×
  at $\hat y=0.01$, 500× at $\hat y=0.001$** — MSE has a vanishing-gradient problem inside the loss
- **The $\ln K$ sanity check** turned into a four-row decision rule that separates a pipeline bug
  from a learning-rate problem in one `print`

**Ch.5 — Backpropagation** (§14–17)
- **The 270,000× speedup computed** — finite differences would need 4.5 minutes per gradient step
- **The δ recurrence derived**, with both factors given meanings: transposed weights routing *blame*
  backwards, and the local gate that zeroes a neuron that wasn't participating
- **Both weight-gradient formulas derived** from the index form — and the three separate things
  $\delta^l(a^{l-1})^\top$ explains at once
- The 3× memory claim itemised (and it's really 4× with Adam); why activations are the term that bites
- **Why `zero_grad()` exists**, and why accumulate-by-default is a feature (gradient accumulation)
- **$0.25^L$ tabulated to depth 50**; the six-step causal chain from large gradient to unrecoverable
  NaN; why clipping preserves direction; all five fixes with the mechanism each attacks

**Ch.6 — Optimizers** (§18–22)
- The four SGD failures, each mapped to its fix; why **saddle points, not local minima**, are the
  high-dimensional enemy
- **Momentum's cancellation worked numerically** — the useless direction damped, the useful one
  amplified 4.1× by step 5 — and the $\frac{1}{1-\beta}=10$ limit
- ⚠️ Why you must divide the learning rate when you turn momentum on
- **Adam's bias correction worked step by step**: without it the first step is **31.6× too large**;
  with it, exactly η
- Adam's memory cost: 4× the model, and 112 GB for a 7B-parameter LLM
- **AdamW's bug in numbers** — one parameter regularized **1,000×** more than another for no reason
- Warmup and cosine decay explained and endpoint-checked
- ⚠️ The Adam-vs-SGD generalisation trade, with the flat-minima story flagged as hypothesis not fact

**Ch.7 — Regularization** (§23–24)
- Dropout's inverted scaling and its $2^n$-ensemble framing; ⚠️ why L1 is essentially unused in DL
- ⚠️ **Why weight decay must skip biases and norm parameters**, with the two-parameter-group idiom
- ⚠️ What's missing from the deck's early-stopping snippet (it doesn't save the best weights)
- Why label smoothing exists — what a hard one-hot target demands of the logits
- **Batch norm: why γ and β mean normalisation costs zero representational power**; the three real
  train/eval failures; ⚠️ why a Linear's bias is redundant before BN; LayerNorm and why Transformers use it

**Training loop & hands-on** (§25–28)
- The six steps with **an omit-it-and-here's-what-breaks table** — including the two omissions that
  produce training which runs perfectly and learns nothing
- The full 20-line PyTorch loop with every chapter traced to its line
- The three bugs with their **distinctive signatures**; `no_grad()` vs `eval()` tabulated
- **A 10-row diagnostic table** consolidating every symptom in the lecture
- **"Overfit a single batch first"** expanded into a workflow, and ⚠️ when the diagnosis is
  underfitting instead

**Closing** — ASCII dependency map · **five threads** (including *"three separate things here are a
product of many factors"* and *"every gradient is (how wrong) × (what came in)"*) · **12 interview
questions** with model answers (5 combining concepts) · 14 depth probes · **3 whiteboard
derivations** · a full Alexa wake-word scenario including the two-stage cascade and the per-cohort
fairness metric · 4 Leadership Principles · **48-term glossary** · **69 check-yourself questions** ·
12 ranked resources with 3 citations that were flagged for verification and have since been
checked directly against primary sources (Cybenko 1989, Hornik 1991, Dauphin et al. 2014, Dinh et
al. 2017 — all confirmed accurate as stated, see the file's "Going deeper" section)

**Interactive specs:** 5 blocks — watch the layers collapse · the forward pass one step at a time ·
one backward sweep · five optimizers on one loss surface · the training loop with the pieces
removable.

</details>

---

## What's in Part 2

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — the two-deck structure with runtimes · 3 capture gaps tabulated and their handling
explained · 14 capabilities · 6 prerequisites taught from zero (**variance and its three rules**, which
are the entire mathematical content of Ch.1 · the $\mathcal{N}(\mu,\sigma^2)$-vs-`std` trap · the chain
rule as a product across layers · fan-in/fan-out including the conv counting people get wrong · the
loss landscape and what *poorly conditioned* means · Bernoulli) · the big picture: **four chapters that
read as one escalating argument**, each existing because the previous chapter's success is not
sufficient · a full-lecture ASCII map

**PART A — Training Deep Networks** (§1–§13)
- §1 **The variance recurrence derived** from two rules; the per-layer gain $g = n\cdot\mathrm{Var}(W)$ as
  a *geometric sequence in depth* · **20 layers at $\sigma$ = 0.05 vs 0.1: $1.3\times10^{-4}$ vs
  $1.5\times10^{8}$** — a factor of 2 in one hyperparameter · ⚠️ why too-large init causes *vanishing*
  with tanh
- §2 Xavier's compromise derived from two conflicting demands; **He's factor of 2 derived** from
  $\mathbb{E}[\mathrm{ReLU}(z)^2] = \tfrac12\mathbb{E}[z^2]$ · ⚠️ **the precision point the slide glosses**:
  the *variance* factor is 0.341, not 0.5 — He tracks the second moment for exactly that reason · the
  mismatch penalty tabulated in both directions
- §3–§4 The backward product; **$0.25^{20} = 9.1\times10^{-13}$ tabulated layer by layer** · why
  vanishing is *silent* and exploding is not · the four fixes with **which one fixes vanishing, which
  fixes exploding, and which fixes neither** · **the skip connection expanded**: $\prod(F'_i + I)$ has
  leading term $I$, independent of every $F'_i$ · the degradation problem
- §5–§8 Four observable symptoms → one mechanism (ill-conditioning) → the cure · **BatchNorm on four
  numbers by hand**, then with $\gamma$=2, $\beta$=3 recovering exactly those · why $\gamma,\beta$ mean BN
  can represent the identity · the train/eval trap and its diagnostic fingerprint · **🩹 LayerNorm taught
  in full** with the same numbers both ways — BatchNorm annihilating an example's structure while
  LayerNorm preserves it · §8's honest history: **ICS was tested and found wanting**
- §9–§13 Flat vs sharp minima with an ASCII proof-sketch · ⚠️ **Dinh et al.'s reparameterization
  objection** flagged · **🩹 L2 supplied**; L1 vs L2 settled by comparing gradients, then demonstrated:
  $w$=0.01 goes to **exactly 0 in one step** under L1 and never under L2 · **dropout's $\frac{1}{1-p}$
  derived** from the expectation requirement · early stopping's three design decisions read off the
  code · **a 4-step debugging checklist** where reaching for step 4 when your problem is step 2 is the
  classic wasted week

**PART B — Convolutional Neural Networks** (§14–§28)
- §14 The three failures quantified: **150,529,000 parameters for one FC layer vs 1,792 for the conv
  alternative** · ⚠️ **equivariance vs invariance** — the distinction interviewers actually probe
- §15 **All nine output cells of the deck's 5×5 convolution computed by hand** and matched to the slide
- §16 **🩹 The output-size formula derived** (including where the $+1$ comes from) and checked against
  *five* independent numbers in the deck, ending with **ResNet-50's real 224→7 trace**
- §17–§19 Padding's *two* costs · why kernels are odd-sized · three reasons to downsample · **both
  pooling modes computed** on the deck's 4×4, and what the two output matrices tell you · **Global
  Average Pooling: 102.8M parameters → 513K**
- §20 **The receptive-field recurrence derived**; the VGG claim verified in both parameters ($27C^2$ vs
  $49C^2$) and nonlinearity; a second example showing **a stride-2 layer doubling every layer above it**
- §21–§24 The canonical block with **all three ordering choices justified** · the evolution table with
  *"read the Params column, not the Depth column"* · **73,856 and 102,764,544 verified, ratio 1,391×** ·
  the FLOPs formula the deck omits and why memory and latency have different answers · **the bottleneck
  sandwich: 590,080 → 70,016 parameters**, and ⚠️ *most of the 36× is the channel cut, not the kernel*
- §25–§28 Transfer learning as a 2×2 decision grid · ⚠️ **three traps**, including *frozen ≠ inactive for
  BatchNorm* · **IoU computed to 0.143** and why that's a miss · NMS written as an algorithm · **the Gram
  matrix's spatial blindness derived** — and style transfer reframed as *the weights are frozen and the
  image is the parameter*

**Closing** — a full-lecture ASCII dependency map showing Part A's fixes feeding Part B's conv block ·
**12 interview questions** with model answers (3 combining concepts) · 8 depth probes · 3 whiteboard
derivations · an image-quality-gating scenario with the two-threshold routing policy · 4 Leadership
Principles · **56-term glossary** · 12 check-yourself questions · 23 ranked resources

**Interactive specs:** 7 blocks — variance propagation through depth · gradient magnitude by depth ·
BatchNorm on a live batch · dropout as an ensemble · the sliding kernel · receptive field traced
backwards · the content/style trade-off.

</details>

---

## What's in Part 3

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — ⚠️ **this is not more feedforward material**: it is the recurrent-networks and
PyTorch lecture · six parts with runtimes · capture notes · 14 capabilities · 6 prerequisites (sequence
notation · the product rule recap · **$\sigma$ = how much, $\tanh$ = what content** — the sentence that
makes all six LSTM equations readable · the Hadamard product as *the mathematical form of gating* ·
spectral radius · `batch_first`) · the big picture and a full-lecture ASCII map

**PART A — Vanilla RNN and BPTT** (§1–§7)
- §1–§2 Order as the defining property · 💡 **the deck's own opening example quietly violates its stated
  rule** — and §17 is where it pays that off · the three MLP failures, with **7.68M vs 416K parameters**
- §3 **🩹 The recurrence stated and unrolled**, making both key facts visible: $h_t$ genuinely contains
  $x_1$, and $W_{hh}$ appears $T$ times · **two RNN steps computed by hand**, showing $x_1$'s contribution
  *already* decaying after one step · 💡 an RNN as a very deep net whose layers are forced to share weights
- §4 The Quick Check, with **why each wrong answer is wrong** — "compressed" as the load-bearing word
- §5 **$0.9^n$ tabulated to 200 steps** · **effective memory computed at ≈44 steps** · a steps-to-1% table
  showing 0.99 buys 10× the memory of 0.9 — which is exactly what §13 delivers
- §6 **The BPTT product derived in three steps** · 💡 **the vicious property**: the better the RNN stores
  signal, the faster its gradients die · ⚠️ why fixing the initialization *cannot* work here, in sharp
  contrast with Part 2 §2
- §7 Why clipping is near-mandatory for RNNs · 💡 **truncated BPTT is a compute fix mistaken for a
  learning fix** — it turns a small gradient into a zero one and calls it solved

**PART B — LSTM and GRU** (§8–§16)
- §8 💡 the cleanest framing of vanishing gradients there is: **not that the model forgets, but that it
  can never learn to remember**
- §9–§11 The two states tabulated and **why there are two** · ⚠️ *LSTM does not eliminate the problem, it
  adds a clean path* · **ERASE / WRITE / REVEAL** · why gates are sigmoids and why they are vectors · all
  six equations with every symbol defined · the four gate-configuration cases · 💡 **why $C$ is not
  squashed — the squash is on the exit ramp, not the highway** · the 4× parameter count verified in code
- §12 **The deck's numeric example verified line by line** — one dimension replaced, one held, in a
  single step, which is the whole argument for per-dimension gates
- §13 **$\partial C_t/\partial C_{t-1} = f_t$ derived**, and the four-row table of what changes versus the
  RNN · **0.9 vs 0.99 vs 0.999 tabulated to 1000 steps** · the Kindle example quantified both ways ·
  📚 **forget-bias initialization** and why $b_f = 0$ is a chicken-and-egg problem
- §14–§16 The frozen-cell quiz and **why the second half of its answer is what separates a good answer
  from a correct one** · **🩹 GRU's reset and candidate equations supplied** · ⚠️ **the $z$-convention trap**,
  with the practical rule that beats memorising either side · a decision rule, and 💡 **constant memory
  per step** as the property keeping recurrent models alive on the edge · 🔬 state-space models

**PART C–D — BiRNN and Seq2Seq** (§17–§20)
- §17–§19 The information was there all along, one position to the right · **the shapes worked**, and
  ⚠️ **`h_n` vs `output[:, -1, :]` — a silent bug, not a crash** · one test decides legality, and *speech
  recognition appears on both sides of it* · ⚠️ **why "cannot" means label leakage** — and therefore why
  BERT masks
- §20 The encoder–decoder · **the bottleneck sized in bits** — 16,384 available vs ~745 needed, so *why
  does it fail?* — with **three distinct causes** · 📚 autoregressive generation, stop tokens, teacher
  forcing and exposure bias · ➕ **attention**, clearly marked as an addition, with each of its three
  effects mapped onto one of the three causes

**PART E–F — PyTorch** (§21–§26)
- §21 The graph verified by hand ($dz/dx = 12$) · 💡 **`loss.backward()` *is* §6's equation** — BPTT is
  not a separate algorithm and there is no flag · ⚠️ the `total_loss += loss` memory leak
- §22–§23 What `nn.Module` buys, line by line · ⚠️ **the plain-Python-list trap** and why it is silent ·
  why mini-batches beat both extremes, for two different reasons · the five steps with **every ordering
  constraint justified**, and why accumulate-by-default is a feature
- §24 **The entire hands-on notebook reproduced with its real printed output** — 22 batches verified,
  259 parameters verified layer by layer, **the $\ln K$ check turned into a four-row diagnostic**, the
  loss trajectory read honestly, and ⚠️ the one cell whose output was never captured, said plainly
- §25 Each common issue expanded into a procedure · 💡 **a 5-step debugging order where steps 1–4 are
  correctness and only step 5 is generalisation**

**Closing** — a full-lecture ASCII map · 💡 the thread tying LSTM, ResNet and attention into **one
insight** · **12 interview questions** with model answers (3 combining concepts) · 8 depth probes · 3
whiteboard derivations · a session-intent scenario including **the BiRNN trap that validates offline
and is unservable** · 3 Leadership Principles · **43-term glossary** · 12 check-yourself questions · 17
ranked resources

**Interactive specs:** 4 blocks — the unrolled RNN on "hello" · gradient decay through time · one LSTM
cell gate by gate · the 5-step loop stepped.

</details>

---

## Reading guide

The three files are ~98,000 words. Read them in order — Part 2 assumes Part 1 throughout, and Part 3's
central result is explicitly Part 2's residual connection applied to time instead of depth.

**First pass (~10 hours across the module).** Read linearly. Do not skip any *Before we start* section:
Part 1 §15 is unreadable without the chain rule and §20 without the EMA; Part 2 §1 is unreadable without
the three variance rules; Part 3 §11 is unreadable without knowing that $\sigma$ means *how much* and
$\tanh$ means *what content*. Skip the `interactive` spec blocks entirely on this pass.

**Second pass.** Work every 🧪 example on paper *before* reading the solution. The five that matter most,
in order: Part 1 §10 (a full forward pass), Part 1 §13.3 (the two gradient derivations), Part 2 §1
(variance propagation at 20 layers), Part 2 §15 (the 5×5 convolution, all nine cells), and Part 3 §12
(one LSTM cell update).

**Weekly.** Glossary + Check yourself, from all three files. Both are built for spaced repetition — the
three glossaries total 147 terms and the three question sets total 93 questions.

**Before an interview.** The *Putting it together* section of each file, then the 9 whiteboard
derivations, then the three depth-probe tables — in that order. If you have time for only three
derivations: **cross-entropy's gradient** (Part 1), **He's factor of 2** (Part 2 §2.2), and
**$\partial C_t/\partial C_{t-1} = f_t$** (Part 3 §13.1). Those are the three most-asked in this module's
territory.

**When something won't train.** Part 3 §25.2's five-step order is the module's consolidated version:
check the initial loss against $\ln K$ → overfit one batch → check gradient norms → then the learning
rate → and only then anything about generalisation. Steps 1–4 are correctness; only step 5 is
overfitting. Part 1 §26.1's diagnostic table and Part 2 §13's four-chapter checklist expand individual
steps.

**The single thread to carry across all three files.** A quantity that passes through many stages
becomes a *product*, and any per-stage factor other than 1 is fatal at depth. Part 1 meets it in
backprop, Part 2 in initialization and in ResNet's $F'(x) + I$, Part 3 in BPTT and in LSTM's
$\prod f_t$. **The fix is always the same: give the signal a path that is a sum.**

**Callout legend**

| | Meaning |
|---|---|
| 📚 **Background** | A prerequisite the slide assumed you already knew |
| 💡 **Key insight** | The one sentence from that section worth memorising |
| ⚠️ **Careful** | A slide gap, an ambiguity, or a place the standard presentation misleads |
| 🧪 **Worked example** | Real numbers computed to a real answer |
| 🎯 **Interview** | Phrased the way you would actually say it out loud |
| 🔬 **Research opportunity** | Where the field is genuinely still open |
| 🩹 **Reconstructed** | The slide's body was not captured; taught from the standard result, and said so |
| ➕ **Addition** | Not deck content — material added because the lecture's own argument leaves it hanging |

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
| 01 | A neural network is one block — multiply, add, bend — repeated, and the bend is load-bearing: without it depth collapses algebraically to a single matrix; training it is the chain rule arranged so that one backward sweep costs the same as one forward pass instead of one pass per weight, and everything else in the lecture — ReLU over sigmoid, Adam over SGD, batch norm, clipping, residuals — is a different answer to the same question of how to keep a product of many factors from collapsing to zero or running away to infinity. |
| 02 | Deep networks fail in four separate ways, and each of Part A's chapters exists because the previous one's success is not sufficient: initialization must set the per-layer gain to 1 ($\mathrm{Var}(W) = 1/n$, or $2/n$ for ReLU, because ReLU halves the second moment), live gradients must then survive the trip back (ReLU, clipping, and above all skip connections, whose $F'(x) + I$ leaves an identity term no depth can shrink), a live gradient must then walk on a well-conditioned landscape (which is what normalization actually buys — landscape smoothing, not the internal covariate shift the original paper claimed), and only then does *which* minimum you reach matter; Part B then shows that the architectures which made 2012–2015 famous are precisely the ones those four fixes made trainable — and that convolution's whole contribution is hard-coding three true assumptions about images (locality, stationarity, compositionality) so that 150 million parameters become 1,792. |
| 03 | Sharing one weight matrix across every position is what lets an RNN handle a sequence at all — and is exactly what makes its gradient a product of $T$ nearly-identical factors, so $0.9^{100} = 0.0000265$ and the effective memory is about 44 steps; the fix is not a better initialization (impossible here, since the $\tanh$ derivative shrinks *more* the better the state is doing its job) but an architectural one — LSTM's cell state updates by **addition**, so the gradient along it is $\prod f_t$, a product of *learned* numbers the network can push toward 1, and $0.99^{100} = 0.366$ survives; that is the same trick as Part 2's residual connection applied to time instead of depth, and attention is the same idea again taken to its limit, which is how this lecture's final slide sits one step away from the Transformer. |
