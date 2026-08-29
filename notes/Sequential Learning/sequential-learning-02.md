---
title: "Sequential Learning — Part 2: From Context to Attention"
topic: sequential-learning
lecture: 19
source: "output/Lecture_19 - Module 6 Sequential Learning Part 2"
slides: 54
---

# Sequential Learning — Part 2: From Context to Attention

> Runtime 1:12:00. Built from the raw capture in `output/Lecture_19 - Module 6 Sequential Learning
> Part 2/` (136 raw frames), not `slides_deduped/` — see project memory `slides-deduped-is-lossy`.
> The lecture opens by explicitly building on Part 1 ("From Part 1 you know: Word2Vec, GloVe,
> FastText give each word exactly ONE vector, regardless of context" [slide 3]), so this file
> assumes [`sequential-learning-01.md`](sequential-learning-01.md) has already been read. It closes
> with an explicit forward pointer to Part 3 ("assembling the Transformer"), so nothing here is a
> capture gap — the lecture itself stops at self-attention and multi-head attention, by design.

---

## What you'll understand after reading this

1. **Explain exactly why a static embedding is insufficient** for a word like "bank," and describe
   the architecture (ELMo) that was the field's first fix.
2. **Derive why RNNs/LSTMs fundamentally cannot parallelize across time**, and quantify how badly
   their gradients vanish over a long sequence using the chain rule.
3. **Draw and explain the encoder-decoder bottleneck** — why compressing an entire sentence into one
   fixed-size vector fails for long inputs, with the empirical BLEU-score evidence.
4. **Derive the attention mechanism's three-step computation** (score → normalize → aggregate) and
   compute a full attention output by hand for a 3-token toy example.
5. **Compare all four standard attention scoring functions** (additive, dot-product, general, scaled
   dot-product) and explain why the field converged on the last one.
6. **Explain self-attention as attention applied to one sequence**, and state precisely why it fixes
   both the RNN's parallelism problem and its long-range dependency problem.
7. **Derive Query/Key/Value from a database-lookup analogy**, and compute Q, K, V from raw token
   embeddings by hand.
8. **Derive why scaled dot-product attention divides by $\sqrt{d_k}$** from a variance argument, not
   just "because the paper says so."
9. **Explain multi-head attention as parallel, independent attention computations**, and state what
   different heads empirically specialize in.

---

## Before we start: what you need to know

### Prerequisite 1 — What this lecture assumes about RNNs and LSTMs

You do not need the internal gate equations of an LSTM for this lecture (that's assumed background,
built in an earlier module), but you do need the shape of the problem: an RNN/LSTM processes a
sequence **one token at a time**, maintaining a running hidden state $h_t$ that summarizes everything
seen so far. Each new hidden state is computed from the previous one: $h_t = f(h_{t-1}, x_t)$. This
single fact — that $h_t$ *depends on* $h_{t-1}$ — is the root cause of every RNN limitation this
lecture identifies.

### Prerequisite 2 — Variance, briefly

> **Variance** — a measure of how spread out a set of numbers is around their average. If $X$ has
> mean 0 and variance 1 ("unit variance"), its values typically range over roughly $[-2, 2]$; if its
> variance were $d$ instead, its typical range would grow to roughly $[-2\sqrt{d}, 2\sqrt{d}]$.
>
> *Why it matters here:* the derivation for why attention divides by $\sqrt{d_k}$ (§6) is a direct,
> short argument about how variance grows when you sum $d$ independent terms — you need this
> definition to follow it, not a full course in probability.

### Prerequisite 3 — Softmax saturation

> **Softmax saturation** — when the inputs to a softmax have very different magnitudes, the output
> becomes extremely "peaked" (close to one-hot: one entry near 1, the rest near 0), and the gradient
> almost everywhere else becomes vanishingly small.
>
> *In everyday words:* if one candidate's score is wildly higher than the others, softmax basically
> declares a winner with 100% confidence — and once it's that confident, nudging the losing scores up
> or down barely changes anything, so there's almost no gradient signal left to learn from.
>
> *Why it exists as a concept here:* this is the exact failure that motivates dividing attention
> scores by $\sqrt{d_k}$ in §6 — without it, raw dot products can grow large enough to saturate the
> softmax and stall training.

### Prerequisite 4 — Matrix multiplication as a set of dot products, briefly

Q, K, V (§7) are each produced by multiplying a matrix of token embeddings $X$ by a learned weight
matrix ($W^Q$, $W^K$, $W^V$). All you need: multiplying an $n\times d_{\text{model}}$ matrix of
token embeddings by a $d_{\text{model}}\times d_k$ weight matrix produces an $n\times d_k$ matrix —
one new, projected vector per token, each computed as a dot product between that token's original
embedding and a column of the weight matrix. This is the exact same "embedding = a row of a learned
matrix" fact from Part 1's Prerequisite 3, applied three times over (once each for $W^Q$, $W^K$,
$W^V$).

---

## The big picture

Part 1 answered "how do you turn a word into numbers?" This lecture answers the next question:
**those numbers were still fixed per word — how do you make a word's representation depend on the
sentence it's actually in?** The arc is a chain of "here's the fix, but it has a new problem" steps:

```
Static embeddings (Word2Vec/GloVe/FastText)
   │  problem: "bank" gets ONE vector regardless of context
   ▼
ELMo — contextual embeddings via a bidirectional LSTM
   │  problem: still sequential → can't parallelize, gradients still fade over long sequences
   ▼
Encoder-Decoder (Seq2Seq) — two RNNs, one reads, one writes
   │  problem: everything is squeezed through ONE fixed-size "context vector" — the bottleneck
   ▼
Attention — decoder looks back at ALL encoder states, weighted by relevance
   │  problem: attention as introduced here only connects TWO sequences (encoder ↔ decoder);
   │           it doesn't yet help a single sentence understand itself
   ▼
Self-Attention — the same weighted-lookup idea, but a sequence attends to itself
   │  fixes: RNN's O(sequence length) path between distant tokens → O(1); and the parallelism problem
   ▼
Scaled Dot-Product Attention + Multi-Head Attention — the actual mechanism, made numerically stable
   and given multiple "perspectives" at once
   │
   ▼
[Part 3, not yet covered here] Assembling the Transformer: + positional encoding, layer norm,
residual connections, feed-forward layers, masking — then BERT/GPT/T5 as configurations of it
```

Every step in this chain is introduced with the same rhetorical move: name a *concrete* failure of
the previous approach, then show the fix designed specifically to remove it. Track that pattern and
the whole lecture becomes predictable.

---

## 1. The Problem with Static Embeddings

> 💡 **The one sentence that reopens the whole problem.** "Word2Vec, GloVe, FastText give each word
> exactly ONE vector, regardless of context" [slide 3]. Two sentences from the deck make this
> concrete: *"I sat by the river bank"* vs. *"I deposited money at the bank."* **Same word,
> completely different meanings, but the same vector** [slide 3].

The slide's own diagram contrasts this directly: under Word2Vec (static), the single point labeled
`"bank"` sits at one fixed location, roughly equidistant from both the `river/stream/water` cluster
and the `money/loan/deposit` cluster — it can't commit to either meaning because it's the same vector
in every sentence. Under ELMo (contextual), there are **two separate points** — `"bank" (river)`
sitting near `river/water/stream`, and `"bank" (financial)` sitting near `deposit/money/loan` [slide
3] — because ELMo computes a *different* vector for "bank" depending on the sentence it's in.

> 🎯 **The question the rest of this lecture answers:** *"What if the representation could change
> based on the sentence?"* [slide 3]

---

## 2. ELMo — Embeddings from Language Models

> **ELMo (Embeddings from Language Models)** — Peters et al., 2018. Runs a deep **bidirectional**
> LSTM over the full sentence; the LSTM's hidden states *become* the word embeddings, so they change
> with context [slide 6].
>
> *In everyday words:* instead of looking a word up in a fixed dictionary of vectors, you read the
> whole sentence with two readers — one going left-to-right, one going right-to-left — and the
> word's representation is built from what both readers currently "know" at that word's position.

### 2.1 Architecture, top to bottom [slide 6]

1. **Character CNN** for initial embeddings — no fixed vocabulary lookup at all (detailed in §2.2)
2. **Forward LSTM** (left-to-right) over the sentence
3. **Backward LSTM** (right-to-left) over the sentence
4. **Stack 2 layers deep**
5. **Final embedding = weighted sum of all layers**

```
Embedding of "stick" in "Let's stick to" — Step #1     [slide 6]
(Image credit, per the slide's own caption: Jay Alammar, "The Illustrated BERT.")

           Forward Language Model              Backward Language Model
LSTM
Layer #2    ●───▶●───▶●                          ●◀───●◀───●
              ↑     ↑     ↑                          ↑     ↑     ↑
LSTM
Layer #1    ●───▶●───▶●                          ●◀───●◀───●
              ↑     ↑     ↑                          ↑     ↑     ↑
Embedding   [Let's][stick][to]                    [Let's][stick][to]
```

> 💡 **Key insight — why bidirectional matters.** A left-to-right-only reader knows "stick" is
> preceded by "Let's" but has no idea what comes after. A right-to-left-only reader knows the reverse.
> ELMo runs **both** and combines them, so the embedding for "stick" reflects the *entire* sentence —
> both what came before and what comes after — not just one direction.

### 2.2 Character CNN (Layer 0) — how ELMo handles any word at all

> **Character CNN** — before the biLSTM runs, every word is built from its individual characters
> using convolutional filters, rather than looked up in a fixed vocabulary table [slide 9].

```
                     [feeds into biLSTM layers...]
                              │
                Character-Level Word Embedding
                              │
                        [Concatenate]
                   ┌──────────┼──────────┐
                 [max]      [max]      [max]
                   │           │           │
             Width-3 Conv Width-4 Conv Width-5 Conv
                   │           │           │
             ┌─────┴─────────────────────┴─────┐
             p   l   a   y   i   n   g   (raw characters, no vocabulary lookup needed)
```

Several convolutional filters of different widths (3, 4, 5 characters) slide over the raw character
sequence of a word (here "playing"); each filter's outputs are **max-pooled** down to one scalar,
and all the filters' outputs are concatenated into the character-level word embedding [slide 9].

> 💡 **Key insight — this is why ELMo handles out-of-vocabulary words.** *"No vocabulary needed! Any
> word — even typos or new words — gets a vector. This is why ELMo handles out-of-vocabulary words"*
> [slide 9]. Notice this is the exact same structural fix FastText used in Part 1 §8 — build a
> word's representation from sub-word pieces (there: character n-grams; here: character convolutions)
> so an unseen word still produces *something*, rather than nothing.

### 2.3 What Different Layers Capture

> 💡 **Key insight — depth specializes.** Different layers of the stacked biLSTM learn qualitatively
> different things, without being told to [slide 18]:

| Layer | Captures |
|---|---|
| **Layer 0** (character CNN) | Morphology, word shape (prefix, suffix, capitalization) |
| **Layer 1** (lower biLSTM) | Syntax, POS tags, local grammatical structure |
| **Layer 2** (upper biLSTM) | Semantics, word sense disambiguation, coreference |

> 🔬 **Research opportunity — this finding outlived ELMo itself.** *"This insight persists in modern
> LLMs: lower layers = syntax, upper layers = semantics, final layers = task-specific"* [slide 18].
> The specific architecture (a stacked bidirectional LSTM) is now largely obsolete, but the emergent
> property — depth specializes from surface form toward meaning — reappears in Transformer-based
> models covered in Part 3 onward.

> ⚠️ **ELMo's own limitation, stated directly on the slide.** *"Limitation: Still sequential (biLSTM),
> so slow to train"* [slide 18]. ELMo fixed the *static-vector* problem but inherited the *sequential
> processing* problem — which is exactly the problem Part 3 (§3 below) develops in full, and which
> attention (§5 onward) is the actual fix for.

```interactive
type: diagram
title: ELMo's Two Fixes and One Remaining Problem
concept: Contextual embeddings via bidirectional LSTM, and what they still can't solve
control: Toggle between "Static (Word2Vec)" and "Contextual (ELMo)" views of the same sentence
  containing "bank"; separately toggle biLSTM layer 0/1/2 to see which captures morphology vs.
  syntax vs. semantics
observe: In contextual mode, "bank" (river) and "bank" (financial) separate into two different
  points; in layer mode, the same word's nearest neighbors shift from spelling-similar words (layer
  0) to meaning-similar words (layer 2)
insight: Fixing "one vector per word" (contextual embeddings) is orthogonal to fixing "sequential
  processing" (still unsolved here) — ELMo only fixes the first
fallback: The static-vs-contextual comparison in §1 and the layer table above cover the same two
  points in prose.
```

---

## 3. Why Not Just Use RNNs?

Having shown ELMo fixes the *static-vector* problem, the lecture turns to ELMo's own unfixed
weakness, in two parts.

### 3.1 The Long-Range Problem [slide 28]

Even LSTMs with gates (designed specifically to help information survive longer) suffer from three
distinct issues:

1. **Information fading over long sequences** — token 1's information is diluted by token 50
2. **Vanishing/exploding gradients** — gradients must flow through *every* timestep
3. **No direct connection between distant tokens** — word 1 and word 50 interact *only* through the
   chain of intermediate hidden states

> **Vanishing gradient** — when a gradient, computed via repeated multiplication across many layers
> or timesteps, shrinks toward zero, leaving the network unable to learn from signal that far back.
>
> *Why it exists:* backpropagation through time multiplies a gradient by (roughly) the recurrent
> weight matrix once per timestep. If that matrix's dominant eigenvalue has magnitude less than 1,
> the product shrinks geometrically with sequence length.

🧪 **Worked example — the chain rule made explicit** [slide 30]. The interactive demo sets a decay
factor $|W_h| = 0.70$ (i.e. each recurrent step multiplies the signal by a matrix whose dominant
eigenvalue has magnitude $0.70 < 1$) and traces the gradient of the *signal at $t=10$* with respect to
the input at $t=1$:

$$\text{Signal at } t{=}10 = W_h \times W_h \times \cdots \times W_h \quad (9 \text{ multiplications})$$

$$\text{Gradient magnitude} = 0.70^9 \approx \mathbf{0.040}$$

> 💡 **Key insight.** *"Each cell multiplies by $|W_h|$. If $|W_h| < 1$, the signal decays
> exponentially"* [slide 30]. After just 9 repeated multiplications, the gradient has shrunk to **4%**
> of its original magnitude — meaning a training signal trying to teach the network something about
> token 1 based on an error at token 10 arrives almost entirely destroyed. The demo's own caption:
> *"Medium sequences: gradient is weak but non-zero. LSTMs help but don't fully solve this"* [slide
> 30] — LSTMs' gating mechanism slows the decay but the exponential shrinkage is structural to any
> architecture that must multiply through every timestep.

> 📚 **Background the slide names directly** [slide 30]: *"LSTM solution: the cell state path
> $c_t = f\odot c_{t-1} + i\odot g$ avoids this repeated $W$ multiplication — gradients flow through
> addition, not matrix products."* This is *why* LSTMs are less prone to vanishing gradients than
> plain RNNs — but note the demo's own conclusion is that this mitigates, not eliminates, the problem.

```interactive
type: simulator
title: The Vanishing Gradient, Forward and Backward
concept: Why repeated multiplication by a sub-1 weight destroys long-range gradient signal
control: Three sliders — Sequence Length (tokens), Decay Factor $|W_h|$, Animation Speed — plus
  "Forward Pass," "Backward Pass," and "Run Forward"/"Run Backward" buttons
observe: A chain of ten token cells ($t{=}1 \ldots t{=}10$) lights up step by step. Forward Pass
  animates left to right (information flow); Backward Pass animates right to left (gradient flow)
  and prints the surviving gradient magnitude under every cell: **1.00, 0.70, 0.49, 0.34, 0.24, 0.17,
  0.12, 0.08, 0.06, 0.04** reading from $t{=}10$ back to $t{=}1$ — each step exactly one more
  multiplication by $|W_h|=0.70$
insight: The decay isn't a training artifact or a bug in one particular run — dragging the Decay
  Factor slider below 1 reproduces the same geometric shrinkage at any sequence length, which is the
  whole point of $|\lambda_{max}(W)|<1 \Rightarrow$ exponential vanishing. Dragging it above 1
  produces the mirror-image failure (exploding gradients), which the slide's own caption doesn't
  animate but the same chain-rule product explains
fallback: The worked example above already gives the $t{=}1$ endpoint exactly
  ($0.70^9\approx\mathbf{0.040}$); the demo's own backward-pass readout confirms every intermediate
  step matches the same $0.70^k$ formula ($k=0\ldots9$), captured directly from `slide_030.jpg`
  (forward-pass state) and `slide_031.jpg` (backward-pass state, showing the full per-cell decay
  chain: 1.00 → 0.70 → 0.49 → 0.34 → 0.24 → 0.17 → 0.12 → 0.08 → 0.06 → 0.04).
```

> 👉 **See also.** [`Deep Neural Networks` Part 3, §6](../Deep%20Neural%20Networks/deep-neural-networks-03.md)
> derives this same $|W_h|^n$ decay from the chain rule in full generality (with a spectral-radius
> argument), and [§13](../Deep%20Neural%20Networks/deep-neural-networks-03.md) derives exactly why
> the LSTM's additive cell-state path slows but doesn't eliminate it — the two modules' derivations
> agree and either can be read first, but Module 2's is the more general treatment.

### 3.2 The Parallelism Problem [slide 35]

> ⚠️ **This is a distinct problem from vanishing gradients — don't conflate them.** Vanishing
> gradients is about *training quality* over long sequences. The parallelism problem is about *raw
> compute cost*, and would exist even with a hypothetical RNN that never lost gradient signal.

RNNs are inherently sequential: $h_t$ depends on $h_{t-1}$, which depends on $h_{t-2}$, ... . **You
cannot compute $h_{50}$ until $h_{49}$ is done** [slide 35]. This means:

- Training cannot fully exploit GPU parallelism (GPUs are built to do many independent computations
  at once; a strict dependency chain forces them to wait)
- Longer sequences = linearly more (wall-clock) time
- Scaling to full documents or books is impractical

**What we want** [slide 35]:
- Process all positions in parallel
- Any token can directly attend to any other token
- Constant path length between distant positions

> 🎯 **The pivot sentence of the whole lecture:** *"This is exactly what self-attention will give
> us"* [slide 35]. Every remaining section is building toward satisfying these three bullet points.

---

## 4. Encoder-Decoder Architecture (Seq2Seq)

Before attention can be introduced, the lecture needs a concrete setting where it's obviously useful:
translating one sequence into another.

> **Encoder-Decoder (Seq2Seq)** — two separate RNNs working together: an **encoder** reads the input
> sequence token by token and produces a single final hidden state (the **context vector**); a
> **decoder** reads that context vector and generates the output sequence token by token [slide 41].

```
   ENCODER                              DECODER
   ┌───┐  ┌───┐  ┌───┐  ┌───┐    ┌──────────┐   ┌───┐  ┌───┐  ┌───┐  ┌───┐
   │ Je│─▶│suis│─▶│ un│─▶│étud│─▶│  Context  │──▶│ I │─▶│ am│─▶│ a │─▶│stud│
   └───┘  └───┘  └───┘  └───┘    │  Vector   │   └───┘  └───┘  └───┘  └───┘
                                  └──────────┘
                          Entire input compressed into one vector    [slide 41]
```

**Used for:** translation, summarization, question answering — any task where the input and output
sequences have different lengths.

---

## 5. The Bottleneck Problem

> ⚠️ **The named failure that motivates attention.** *"The entire input gets compressed into ONE
> fixed-size vector"* [slide 44]. For short sentences this works fine. **For long sentences,
> information gets lost.**

> Think about it: a 50-word paragraph compressed into 512 floats. The decoder must reconstruct
> *everything* from just that. **Empirically: BLEU scores drop sharply for sentences longer than 20
> tokens** [slide 44].

> **BLEU score** — a standard automatic metric for translation quality, comparing a model's output
> against reference translations; higher is better. Its mention here is used purely as empirical
> evidence: it's the number that dropped, giving a measurable symptom of the bottleneck problem, not
> something the lecture derives.

> 🎯 **The solution the slide poses as a question, then the rest of the lecture answers:** *"What if
> the decoder could look back at the full encoder output, not just the last hidden state?"* [slide
> 44] — this is, verbatim, the definition of attention.

```interactive
type: slider
title: The Bottleneck, Visualized
concept: Fixed-size context vector vs. input length
control: Slide input sentence length from 5 to 50 tokens, with context vector size fixed
observe: A "reconstruction difficulty" indicator rises with sentence length while the vector's
  capacity stays flat; overlay the empirical BLEU-score-vs-length curve
insight: The bottleneck isn't a bug in this particular model — it's a direct consequence of forcing
  a variable amount of information through a fixed-size channel
fallback: The "50-word paragraph into 512 floats" framing above states the same capacity mismatch
  in words.
```

---

## 6. Attention Mechanism

### 6.1 The Key Idea

> **Attention** — at each decoder step, compute a **weighted sum over ALL encoder hidden states**,
> instead of relying on a single fixed context vector [slide 51].
>
> *Words before symbols:* the weights say **"how relevant is each source word to the word I am
> generating right now?"** [slide 51]. When generating `'cat'`, attention puts high weight on the
> source word `'chat'` and low weight on `'Le'`, `'sur'`; when generating `'sat'`, high weight goes
> on `'assis'` and low weight on everything else [slide 51]. The decoder isn't limited to one
> compressed summary anymore — it can look back and reweight *which parts of the input matter* at
> every single output step.

### 6.2 The Computation — three steps [slide 55]

**Words first:** to generate the current output token, first *score* how relevant each encoder
position is to what the decoder currently needs; turn those scores into a proper probability
distribution; then take a weighted average of the encoder states using that distribution as the
weights.

$$e_{t,i} = \text{score}(\mathbf{s}_t, \mathbf{h}_i) \qquad \alpha_{t,i} = \text{softmax}(e_{t,i}) = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})} \qquad \mathbf{c}_t = \sum_{i=1}^n \alpha_{t,i}\,\mathbf{h}_i$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\mathbf{s}_t$ | "decoder state" | Decoder hidden state at output step $t$ |
| $\mathbf{h}_i$ | "encoder state" | Encoder hidden state at input position $i$ |
| $e_{t,i}$ | "raw score" | How relevant encoder position $i$ is to decoder step $t$, before normalizing |
| $\alpha_{t,i}$ | "attention weight" | The normalized (softmax) relevance — a proper probability, summing to 1 across $i$ |
| $\mathbf{c}_t$ | "context vector *at this step*" | The weighted sum of all encoder states — note this is now recomputed at **every** decoder step, unlike §4's single fixed context vector |

**The three steps, named** [slide 55]:
1. **Score:** how relevant is each encoder state to the current decoder state?
2. **Normalize:** softmax converts scores to probabilities
3. **Aggregate:** weighted sum of encoder states gives the (per-step) context

Then combine $[\mathbf{c}_t; \mathbf{s}_t]$ (concatenated) to actually predict the next output token
[slide 55].

> 💡 **Key insight — this is the direct fix for the bottleneck.** §5's fixed context vector is
> replaced by $\mathbf{c}_t$, a *different* weighted combination of *all* encoder states at every
> single decoder step. The decoder is no longer limited to one compressed summary of the whole
> input — it can re-examine the full input, differently, every time it generates a new word.

### 6.3 Scoring Functions — four options, each simpler than the last [slide 62]

$$\text{Additive (Bahdanau):}\quad \text{score}(\mathbf{s},\mathbf{h}) = \mathbf{v}^\top\tanh(\mathbf{W}_1\mathbf{s} + \mathbf{W}_2\mathbf{h})$$
$$\text{Dot-Product (Luong):}\quad \text{score}(\mathbf{s},\mathbf{h}) = \mathbf{s}^\top\mathbf{h}$$
$$\text{General (Luong):}\quad \text{score}(\mathbf{s},\mathbf{h}) = \mathbf{s}^\top\mathbf{W}\mathbf{h}$$
$$\text{Scaled Dot-Product:}\quad \text{score}(\mathbf{s},\mathbf{h}) = \frac{\mathbf{s}^\top\mathbf{h}}{\sqrt{d_k}}$$

> ⚠️ **Where people get confused.** These are ordered from *most* to *least* parameters, not from
> *worst* to *best*. Additive attention has its own learned weight matrices $\mathbf{W}_1,
> \mathbf{W}_2$ and vector $\mathbf{v}$ — expressive, but expensive (an extra small feedforward
> network per scoring computation). Plain dot-product has *zero* extra parameters but forces $s$ and
> $h$ into the same dimensionality. "General" adds back one learned matrix $\mathbf{W}$ to relax that
> constraint. Scaled dot-product is dot-product plus one fix, derived properly in §9 below — it isn't
> "simpler" in a way that costs accuracy, it's simpler *and* it solves a numerical problem the plain
> version has.

### 6.4 Summary So Far — the explicit pivot point [slide 68]

**What we have built:** encoder-decoder (two separate sequences); attention (decoder looks at all
encoder states, weighted sum); scoring (scaled dot-product is fast and effective).

**What is still missing:** *"This only helps BETWEEN two sequences. What about understanding WITHIN
a single sentence? RNNs process sequentially, so distant words struggle to interact"* [slide 68].

**Next: Self-Attention.** *"Apply the same attention idea, but a sequence attends to itself. Every
token can directly attend to every other token in one step"* [slide 68].

> ⚠️ **This is the hinge of the entire lecture — read it twice.** Everything in §6 was attention
> connecting *two different* sequences (source ↔ target). It does nothing to solve §3's parallelism
> and long-range problems *within* one sequence, because the encoder itself is still a sequential
> RNN. Self-attention (§7 onward) is what actually fixes §3 — attention *between sequences* and
> self-attention *within one sequence* are the same mathematical mechanism applied to different
> inputs, but they solve two genuinely different problems.

---

## 7. Self-Attention

### 7.1 Why?

🧪 **Motivating example** [slide 72, 84]: *"The animal didn't cross the street because it was too
tired."* What does `'it'` refer to? **An RNN needs 6 sequential steps to connect `'it'` to
`'animal'`** (it has to pass through every intervening hidden state). **Self-attention connects them
in ONE step.**

**The key shift:**
- Encoder-decoder attention (§6): one sequence attends to **another**
- Self-attention: a sequence attends to **itself**
- Every token computes a weighted sum over all other tokens (including itself)

**Benefits:** $O(1)$ path length (any two tokens are one hop apart, regardless of sentence length),
fully parallelizable (no $h_t$-depends-on-$h_{t-1}$ chain), captures long-range dependencies directly
[slide 72].

The interactive attention heatmap for this exact sentence [slide 90] shows the mechanism working:
the cell at row `"it"`, column `"animal"` lights up with a real, non-trivial attention weight — the
model has, without being told any grammar rule, learned to connect "it" back toward the noun phrase
it refers to.

### 7.2 Query, Key, Value

> **Query, Key, Value** — self-attention gives every token **three** different learned projections
> of its embedding, each answering a different question [slide 74]:
>
> - **Query (Q):** "What am I looking for?"
> - **Key (K):** "What do I contain?"
> - **Value (V):** "What do I give when matched?"
>
> *Analogy given directly on the slide:* **database lookup**. Q is your search. K is the index. V is
> the result you retrieve [slide 74].

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q \qquad \mathbf{K} = \mathbf{X}\mathbf{W}^K \qquad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

```
   X          W^Q         Q               X          W^K         K              X          W^V         V
 [────]   ×  [────]  =  [────]          [────]   ×  [────]  =  [────]        [────]   ×  [────]  =  [────]
 (n×d)      (d×d_k)     (n×d_k)         (n×d)      (d×d_k)     (n×d_k)       (n×d)      (d×d_v)     (n×d_v)
                                                                                                [slide 74]
```

$\mathbf{X}$ is the matrix of raw token embeddings (one row per token); $\mathbf{W}^Q, \mathbf{W}^K,
\mathbf{W}^V$ are three **separate, learned** weight matrices; multiplying projects every token's
embedding into its own query vector, key vector, and value vector.

> 💡 **Key insight — why three separate matrices, not one.** Using different learned matrices for Q,
> K, and V lets the model learn *different* things to look for, advertise, and hand over. If Q=K=V
> were forced, a token's "what I'm looking for" and "what I contain" would be mathematically
> identical — collapsing three distinct roles into one and removing the model's ability to separate
> "how I match" from "what I return once matched."

```interactive
type: diagram
title: Self-Attention vs. Encoder-Decoder Attention
concept: The same weighted-lookup mechanism applied to one sequence vs. two
control: Toggle between "Encoder-Decoder Attention" and "Self-Attention" tabs on the same sentence
observe: In encoder-decoder mode, the attention matrix is rectangular (decoder positions × encoder
  positions); in self-attention mode, it's square (sequence × itself), and the diagonal is visibly
  present
insight: Self-attention isn't a new formula — it's the identical score→normalize→aggregate recipe
  from §6.2, just applied with Q, K, V all drawn from the SAME sequence instead of two different ones
fallback: The Q/K/V analogy and formulas above, plus the "it → animal in one step" example, describe
  the same shift in words.
```

---

## 8. Attention Computation, Worked in Full

The lecture's interactive demo walks scaled dot-product self-attention through all 7 steps on a
toy 3-token sentence: **"I", "am", "happy"**, with $d_{\text{model}}=4$ for simplicity [slide 91].

### Step 1 — Input Embeddings [slide 91]

$$\mathbf{x}_{\text{I}} = [1,0,1,0] \qquad \mathbf{x}_{\text{am}} = [0,1,1,1] \qquad \mathbf{x}_{\text{happy}} = [1,1,0,0]$$

### Step 2 — Project to Q, K, V [slide 95]

Using $4\times2$ weight matrices (projecting $d_{\text{model}}=4$ down to $d_k=2$):

$$\mathbf{Q} = \begin{bmatrix}2&0\\0&2\\1&1\end{bmatrix} \qquad \mathbf{K} = \begin{bmatrix}0&2\\2&0\\1&1\end{bmatrix} \qquad \mathbf{V} = \begin{bmatrix}2&1\\0&1\\1&1\end{bmatrix}$$

(rows are $q_1,q_2,q_3$ / $k_1,k_2,k_3$ / $v_1,v_2,v_3$, in order "I", "am", "happy")

### Step 3 — Compute Attention Scores $\mathbf{Q}\cdot\mathbf{K}^\top$ [slide 99]

$$q_1\cdot k_1 = [2,0]\cdot[0,2] = 0 \qquad q_1\cdot k_2 = [2,0]\cdot[2,0] = 4 \qquad q_1\cdot k_3 = [2,0]\cdot[1,1] = 2$$
$$q_2\cdot k_1 = [0,2]\cdot[0,2] = 4 \qquad q_2\cdot k_2 = [0,2]\cdot[2,0] = 0 \qquad q_2\cdot k_3 = [0,2]\cdot[1,1] = 2$$
$$q_3\cdot k_1 = [1,1]\cdot[0,2] = 2 \qquad q_3\cdot k_2 = [1,1]\cdot[2,0] = 2 \qquad q_3\cdot k_3 = [1,1]\cdot[1,1] = 2$$

$$\text{Score matrix} = \begin{bmatrix}0&4&2\\4&0&2\\2&2&2\end{bmatrix}$$

*"Each query asks: 'how relevant is each key to me?' The score is the dot product of the query with
each key. Higher score = more relevant match."* [slide 99]

### Step 4 — Scale by $\sqrt{d_k}$ [slide 102]

With $d_k=2$: $\sqrt{d_k} = \sqrt{2}\approx1.414$.

$$\text{Scaled scores} = \begin{bmatrix}0.00&2.83&1.41\\2.83&0.00&1.41\\1.41&1.41&1.41\end{bmatrix}$$

*"Divide by $\sqrt{d_k}$ to prevent large values from saturating softmax. Without this, gradients
would vanish for large $d_k$ because softmax would produce near-one-hot distributions"* [slide 102]
— the exact softmax-saturation failure mode from Prerequisite 3, now shown concretely.

### Step 5 — Softmax (row-wise) [slide 103]

$$\text{Attention weights } \boldsymbol{\alpha} = \begin{bmatrix}0.045&0.769&0.186\\0.769&0.045&0.186\\0.333&0.333&0.333\end{bmatrix}$$

Each row sums to exactly 1.0 (verified on the slide: $0.045+0.769+0.186=1.0$, etc.). *"Darker amber =
higher attention weight. 'I' attends mostly to 'am'; 'am' attends mostly to 'I'; 'happy' attends
equally"* [slide 103] — note "happy" produces a **uniform** distribution here, a direct consequence
of its scaled scores against all three keys being tied at 1.41 in this toy example.

### Step 6 — Weighted Sum of Values [slide 106]

$$\mathbf{o}_1 = 0.045\cdot[2,1] + 0.769\cdot[0,1] + 0.186\cdot[1,1] = [0.28,\ 1.00]$$
$$\mathbf{o}_2 = 0.769\cdot[2,1] + 0.045\cdot[0,1] + 0.186\cdot[1,1] = [1.72,\ 1.00]$$
$$\mathbf{o}_3 = 0.333\cdot[2,1] + 0.333\cdot[0,1] + 0.333\cdot[1,1] = [1.00,\ 1.00]$$

*"The final output for each token is a weighted sum of all Value vectors, using the attention
weights. Tokens with higher attention weight contribute more to the output. Each output now contains
contextual information from the entire sequence"* [slide 106].

> 💡 **Key insight — trace what actually happened.** Token "I" (output $\mathbf{o}_1$) started as
> $[1,0,1,0]$ and ends as $[0.28, 1.00]$ — a genuinely new vector, built mostly from **"am"'s** value
> vector (weight 0.769), because "I" and "am" scored highly relevant to each other in Step 3. This is
> the entire point of self-attention made concrete: each output vector is *context-aware*, blending
> information from every other token, weighted by learned relevance — not a fixed lookup, and not
> something a static embedding (Part 1) or even ELMo's fixed-direction LSTM state could produce in a
> single step.

```interactive
type: simulator
title: Full Attention Computation, Step by Step
concept: The complete Q/K/V → score → scale → softmax → weighted-sum pipeline
control: Step through all 7 stages (embeddings → project → score → scale → softmax → weighted sum →
  encoder-decoder vs. self-attention comparison) on the "I am happy" toy example
observe: Each stage's output matrix updates and highlights which numbers from the previous stage fed
  into it
insight: There is no hidden step — the entire mechanism is five matrix operations applied in
  sequence, and every number in the final output is traceable back to the original embeddings
fallback: The full worked derivation above reproduces all six numbered steps by hand, ending in the
  same three output vectors.
```

---

## 9. Scaled Dot-Product Attention — deriving the $\sqrt{d_k}$

$$\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V} \qquad \text{where } \mathbf{Q}=\mathbf{X}\mathbf{W}^Q,\ \mathbf{K}=\mathbf{X}\mathbf{W}^K,\ \mathbf{V}=\mathbf{X}\mathbf{W}^V$$

**Step by step** [slide 113]: (1) $\mathbf{Q}\mathbf{K}^\top$ gives an $n\times n$ score matrix; (2)
scale by $\sqrt{d_k}$ to prevent softmax saturation; (3) softmax per row gives attention weights; (4)
multiply by $\mathbf{V}$ to aggregate.

**Why $\sqrt{d_k}$?** [slide 113] — words first: if the individual entries of $q$ and $k$ each have
unit variance (variance 1) and are independent, then their dot product — a sum of $d_k$ such
independent terms — has variance that grows to $d_k$ itself. Large variance means large-magnitude
scores, which saturate the softmax (Prerequisite 3). Dividing by $\sqrt{d_k}$ exactly restores the
variance to 1, keeping the softmax's gradients healthy.

$$q,k \text{ have unit variance} \implies \text{Var}(q\cdot k) = d_k \implies \text{dividing by } \sqrt{d_k} \text{ restores variance to } 1$$

🧪 **Worked example — the saturation demo made concrete** [slide 125]. At dimensionality $d=32$:

| | **Unscaled** | **Scaled by $\sqrt{d}$** |
|---|---|---|
| Raw dot products $(q\cdot k)$ | Range roughly $-15.5$ to $+9.9$ | Range roughly $-2.7$ to $+1.8$ |
| $\text{softmax}(q\cdot k)$ | Max probability **1.000**, entropy **0.00 / 2.08** — "nearly one-hot!" | Max probability **0.653**, entropy **1.24 / 2.08** — "healthy" |

*"As $d$ grows, dot products grow $\sim O(\sqrt d)$... softmax saturates → gradients vanish"*, versus
*"Dividing by $\sqrt d$ keeps the variance ≈ 1, softmax stays smooth, gradients flow"* [slide 125].

> 💡 **Key insight — connect this back to §3.1.** The unscaled column's softmax entropy of **0.00**
> means the distribution has collapsed to picking one key with 100% certainty — structurally the
> *same* failure as a saturated softmax anywhere else: once a distribution is this peaked, its
> gradient with respect to every non-winning entry is near zero, and the model stops learning from
> that comparison. Scaled dot-product attention is a small, deliberate design choice that exists
> entirely to keep this gradient alive as $d_k$ grows.

> 🎯 **Interview-ready one-liner:** "If $q$ and $k$'s components are independent with unit variance,
> their dot product sums $d_k$ independent unit-variance terms, so its variance is $d_k$ — not 1.
> Dividing the score by $\sqrt{d_k}$ before the softmax normalizes that variance back to 1, which is
> exactly what keeps softmax from saturating as the embedding dimension grows."

> 💡 **Cross-module note.** `GenAI & LLM Part 1 §8` independently derives this identical result —
> same variance argument, same conclusion — but uses plain $d$ for the projection dimension where
> this file uses $d_k$. Both are internally consistent within their own file; if you've read both
> modules, $d$ there and $d_k$ here name the same quantity.

---

## 10. Multi-Head Attention

> **Multi-Head Attention** — instead of computing attention once, compute it **$h$ times in
> parallel**, each with its own learned $\mathbf{W}^Q_i, \mathbf{W}^K_i, \mathbf{W}^V_i$, then
> concatenate the results and project back down [slide 127].
>
> *Why it exists:* *"One head captures one type of relationship. We want multiple perspectives
> simultaneously"* [slide 127].

$$\text{MultiHead}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,\mathbf{W}^O$$
$$\text{where head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}^Q_i,\ \mathbf{K}\mathbf{W}^K_i,\ \mathbf{V}\mathbf{W}^V_i)$$
$$d_k = d_v = d_{\text{model}}/h \quad\text{(e.g., } 512/8 = 64 \text{ per head)}$$

🧪 **Worked example, from the standard Transformer sizing** [slide 127]: $d_{\text{model}}=512$,
$h=8$ heads. So $d_k = d_v = 64$ per head. **Same total compute as one big head** — this is a
structural point, not an approximation: splitting into 8 heads of dimension 64 costs the same total
FLOPs as one head of dimension 512, because the per-head matrices are correspondingly smaller.

```
1) Input          2) Embed         3) Split into 8 heads       4) Attention        5) Concat + W^O
sentence*          each word        via W_i^Q, W_i^K, W_i^V     per head            → layer output

"Thinking            X          ┌─ W0^Q,W0^K,W0^V → Q0,K0,V0 → Attention → Z0 ─┐
 Machines"        [embeddings]  ├─ W1^Q,W1^K,W1^V → Q1,K1,V1 → Attention → Z1 ─┤
                                 │              ...                            ├─▶ Concat → ×W^O → Z
                                 └─ W7^Q,W7^K,W7^V → Q7,K7,V7 → Attention → Z7 ─┘
                                                                              [slide 129]
```

*(Footnote on the original slide: "In all encoders other than #0, we don't need embedding. We start
directly with the output of the encoder right below this one" — i.e. this diagram is showing the
*first* Transformer layer specifically; later layers take the previous layer's output as their
input $X$ instead of a fresh embedding lookup.)*

### 10.1 What Different Heads Learn [slide 131]

*"Different heads specialize without being told to"*:

| Head type | Specializes in |
|---|---|
| **Positional** | Attends to nearby tokens (local context) |
| **Syntactic** | Links subjects to verbs, modifiers to nouns |
| **Coreference** | Resolves pronouns (**"it" looks at "animal"** — the exact §7.1 example) |
| **Semantic** | Captures meaning similarity across distance |

> 💡 **Key insight — this closes the loop back to §7.1.** The "it" → "animal" coreference resolution
> that motivated self-attention in the first place isn't hand-coded anywhere in the architecture —
> it *emerges* as one head's specialization, purely from training on data, alongside three other
> distinct specializations no one explicitly programmed.

```interactive
type: diagram
title: One Head vs. Eight Heads
concept: Why multi-head attention captures more than single-head attention
control: Toggle a sentence's self-attention heatmap between "single head" and "8 heads, viewed one
  at a time"; a dropdown selects which of the 8 heads to view
insight: Different heads' heatmaps light up along genuinely different patterns (one along the
  diagonal/local-context, one connecting pronouns to their referents, etc.) — multi-head attention
  isn't redundancy, it's several different learned relationships computed side by side
fallback: The "what different heads learn" table above lists the four named specialization types
  directly.
```

---

## 11. Attention Is Still Evolving

The lecture closes Part 2 by explicitly flagging that everything just derived is a foundation, not
the final word [slide 133]:

**Making it faster:**
- **Flash Attention** — IO-aware, 2–4× speedup (same math, faster memory access pattern)
- **Sliding Window** (Mistral) — local context only, trading global reach for speed

**Shrinking memory at inference:**
- **GQA** (Llama 3) — groups share Key/Value projections
- **MLA** (DeepSeek) — compresses Key/Value to low-rank

**Beyond attention entirely:**
- **Mamba/SSMs** — $O(n)$, no attention at all
- **Hybrids** (Jamba) — mix attention + Mamba

> 💡 **Key insight, stated directly, and worth taking at face value:** *"The core idea you learned
> today is in every model you use. These are optimizations"* [slide 133]. Flash Attention, GQA, and
> MLA all compute the *same* mathematical object (softmax-weighted value aggregation) faster or with
> less memory — they are not alternative theories of attention, they're engineering refinements of
> exactly the mechanism derived in §9–10. Mamba/SSMs are the one genuine exception on this list (a
> different mechanism entirely, not an optimization of attention) — flagged here as the frontier,
> not covered further in this lecture.

The timeline's specific years — **Bahdanau Attention (2014)**, **Luong Attention (2015)**,
Transformer (2017), Flash Attention (2022), GQA/MQA (2023), MLA/Differential Attention (2024), Hybrid
Mamba+Attention (2025) — are each shown as individually-dated points directly on the lecture's own
timeline graphic [slide 133] and confirmed accurate as read; no hedge needed.

---

## Putting it together

```
                    ┌───────────────────────────────┐
                    │ The RNN's two structural flaws  │
                    │  (§3): sequential-only →         │
                    │  1. vanishing gradients over time│
                    │  2. no parallelism across tokens │
                    └────────────┬────────────────────┘
                                 │  attention (§6) fixes the
                                 │  BOTTLENECK between two sequences,
                                 │  but the encoder is still an RNN
                                 ▼
                    ┌───────────────────────────────┐
                    │ Self-attention (§7) applies the │
                    │ SAME mechanism WITHIN one       │
                    │ sequence — this is what actually │
                    │ fixes BOTH of §3's flaws:        │
                    │  • O(1) path length (no chain)   │
                    │  • fully parallel (no h_t→h_t+1) │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────┴────────────────────┐
                    ▼                                  ▼
        Scaled Dot-Product (§9)              Multi-Head (§10)
        makes ONE attention computation      runs SEVERAL attention
        numerically stable as d_k grows      computations in parallel,
        (÷√d_k keeps softmax's gradient      each free to specialize
        alive)                               (positional/syntactic/
                                              coreference/semantic)
                                 │
                                 ▼
                    [Part 3] + positional encoding, layer norm,
                    residual connections, feed-forward layers,
                    masked attention → the full Transformer
```

Three threads run through this lecture:

1. **Every architecture here is introduced by first naming a concrete failure of the one before it,
   with a specific, checkable piece of evidence** — a diagram of "bank" sitting at one ambiguous
   point (static embeddings), a $0.70^9\approx0.04$ gradient-decay calculation (RNN/LSTM),
   BLEU scores dropping past 20 tokens (bottleneck), a softmax entropy of 0.00 (unscaled dot-product).
   None of these fixes are motivated by "this is just how it's done" — each has a named, demonstrated
   problem it solves.
2. **Attention between two sequences and self-attention within one sequence are the same formula**
   ($\text{softmax}(\text{scaled scores})\cdot V$) applied to different inputs — §6.4's explicit
   pivot ("this only helps BETWEEN two sequences... apply the same idea, but a sequence attends to
   itself") is the moment this lecture makes that identity unmistakable.
3. **"Optimization of the same mechanism" vs. "a genuinely different mechanism" is a distinction the
   lecture draws explicitly in its closing section** — Flash Attention/GQA/MLA are the former, Mamba/
   SSMs are the latter — modeling good practice for evaluating any future "new attention variant" you
   encounter: is it the same softmax-weighted-sum computed more efficiently, or something else?

---

## Interview prep — Amazon Applied Scientist

### Core questions

<details><summary><b>1. Why does a static embedding fail on the word "bank," and what specifically does ELMo change to fix it?</b></summary>

A static embedding (Word2Vec/GloVe/FastText) assigns exactly one fixed vector to "bank" regardless of
sentence, so "river bank" and "savings bank" get the identical vector despite meaning different
things. ELMo instead computes the word's representation as the hidden state of a deep bidirectional
LSTM run over the *whole* sentence — since the hidden state at "bank" depends on every other word in
that specific sentence, "bank" gets a different vector in each context.
</details>

<details><summary><b>2. Derive, roughly, how much gradient signal survives after 9 timesteps in an RNN with recurrent weight magnitude 0.70.</b></summary>

Backpropagation through time multiplies the gradient by (roughly) the recurrent weight matrix once
per timestep, so after 9 steps the gradient magnitude is scaled by $0.70^9 \approx 0.040$ — about 4%
of its original size survives. This is a direct consequence of chaining $n$ multiplications by a
factor less than 1, and it's why very early tokens in a long sequence get almost no learning signal
from errors made many steps later.
</details>

<details><summary><b>3. Why can't an RNN's training be parallelized across time steps, and does this affect inference too?</b></summary>

$h_t$ is defined in terms of $h_{t-1}$, so computing $h_{50}$ strictly requires $h_{49}$ to already
exist — there's no way to compute both simultaneously. This affects both training (backprop through
time inherits the same dependency chain) and inference (generation is inherently step-by-step for an
RNN) — though for a Transformer specifically, it's *training* where full parallelism is the big win,
since autoregressive generation at inference is still sequential token-by-token regardless of
architecture.
</details>

<details><summary><b>4. What exactly is compressed in the encoder-decoder bottleneck, and why does it get worse for longer sentences specifically?</b></summary>

The encoder reduces the entire input sequence to a single fixed-size vector (e.g. 512 floats), and
the decoder must reconstruct the whole output from only that vector. A 5-word sentence and a 50-word
paragraph get compressed into the same-sized vector, so the *information per input token* that
survives into the context vector shrinks as the sentence grows — there's simply less room per token
to store what mattered, which is why BLEU scores empirically drop sharply past ~20 tokens.
</details>

<details><summary><b>5. Walk through attention's three-step computation and explain what changes between step 1 (score) and a plain dot-product similarity search.</b></summary>

(1) Score: compute how relevant each encoder state $h_i$ is to the current decoder state $s_t$, via
some scoring function (e.g. dot-product $s_t^\top h_i$). (2) Normalize: pass all the scores at this
decoder step through a softmax so they form a proper probability distribution summing to 1. (3)
Aggregate: take the weighted sum of the encoder states using those normalized weights, producing a
per-step context vector. The difference from plain similarity search is step 2 — softmax turns raw
similarity scores into a *distribution* used for a weighted average, rather than picking a single
best match.
</details>

<details><summary><b>6. What specifically distinguishes self-attention from the encoder-decoder attention introduced earlier in the lecture?</b></summary>

They use the identical score→normalize→aggregate mechanism; the difference is entirely in *where Q,
K, and V come from*. In encoder-decoder attention, the queries come from the decoder and the keys/
values come from the encoder — two different sequences. In self-attention, Q, K, and V are all
projections of the *same* sequence — every token attends to every other token (including itself)
within one sentence.
</details>

<details><summary><b>7. Derive why scaled dot-product attention divides by √d_k.</b></summary>

If the entries of $q$ and $k$ are independent with unit variance, their dot product $q\cdot k$ sums
$d_k$ independent unit-variance terms, giving the dot product a variance of $d_k$ (not 1). As $d_k$
grows, this makes the raw scores grow in typical magnitude, which saturates the softmax (drives it
toward a near-one-hot output) and kills the gradient almost everywhere. Dividing every score by
$\sqrt{d_k}$ rescales the variance back down to exactly 1, keeping the softmax's output — and its
gradient — well-behaved regardless of $d_k$.
</details>

<details><summary><b>8. What is Query, Key, and Value each computed from, and why does the lecture use a database analogy?</b></summary>

$Q=XW^Q$, $K=XW^K$, $V=XW^V$ — three separate learned linear projections of the same input token
embeddings $X$. The database analogy: Q is your search query ("what am I looking for"), K is the
index each stored item advertises itself under ("what do I contain"), and V is the actual content
retrieved once a query matches a key well ("what do I give when matched"). The analogy is apt because
the attention score is literally computed as a similarity between Q and K, and the *output* is built
only from V — matching and retrieving are structurally separated.
</details>

<details><summary><b>9. [Combines concepts] Explain, using the "the animal didn't cross the street because it was too tired" example, exactly why self-attention solves a problem that plain RNN-based encoder-decoder attention (§6) does not.</b></summary>

Encoder-decoder attention (§6) lets a decoder look back at all encoder states — but the encoder
itself is still a sequential RNN, so *within* the source sentence, connecting "it" to "animal" still
requires the RNN to pass information through every intervening timestep (6 sequential steps in this
example), inheriting the vanishing-gradient and no-parallelism problems from §3. Self-attention
instead lets "it," as a token in the sequence, directly compute a similarity score against every
other token — including "animal" — in a single matrix operation, with no dependency chain at all.
The path length from "it" to "animal" drops from $O(\text{distance})$ to $O(1)$.
</details>

<details><summary><b>10. [Combines concepts] Why does multi-head attention split d_model into h smaller heads rather than just running the same-size attention h separate times?</b></summary>

Running $h$ full-size ($d_{\text{model}}$-dimensional) attention computations in parallel would cost
$h\times$ the compute of one. Splitting $d_{\text{model}}$ into $h$ heads of size $d_k=d_v=
d_{\text{model}}/h$ instead keeps the *total* compute equal to one full-size attention computation
(the standard example: $512 = 8\times 64$), while still giving the model $h$ independent
Q/K/V projections that can each specialize in a different kind of relationship (positional,
syntactic, coreference, semantic) — more representational diversity at the same total cost, not at a
multiplied cost.
</details>

<details><summary><b>11. [Combines concepts] A colleague claims Flash Attention and Mamba/SSMs are "both just ways to make attention faster." Correct this.</b></summary>

They're in different categories. Flash Attention computes the *exact same* mathematical function —
softmax-weighted aggregation of values via scaled dot-product attention — just with an IO-aware
implementation that reduces memory movement, giving a 2-4× speedup with no change to the underlying
math or the model's outputs. Mamba/SSMs are a genuinely *different* sequence-modeling mechanism with
no attention computation at all (an $O(n)$ state-space model), trading away attention's explicit
pairwise token comparisons for a different way of propagating sequence information. One is an
engineering optimization of a fixed algorithm; the other is a different algorithm.
</details>

### Depth probes

- *"Why does the LSTM's cell-state path mitigate vanishing gradients but not eliminate them, per the
  demo's own conclusion?"* — the cell state update ($c_t = f\odot c_{t-1} + i\odot g$) replaces
  repeated *multiplication* with *addition*, which doesn't shrink geometrically the way repeated
  multiplication by a sub-unit factor does — but the forget gate $f$ is still a value that can itself
  push toward 0 over many steps under certain learned parameters, and gradients still flow through
  every intermediate timestep's computation graph, so the underlying "many sequential steps" problem
  isn't structurally removed, just softened.
- *"If multi-head attention costs the same total compute as one big head, why not just use one huge
  head?"* — the *compute* is equal, but the *representational* capacity differs: one head can only
  express a single learned notion of "relevance" per query-key comparison, while $h$ independent
  heads, each with their own learned $W^Q_i, W^K_i, W^V_i$, can each converge on a different
  specialization (as seen empirically in §10.1) — the split is a capacity/diversity choice, not a
  compute-saving trick.
- *"Self-attention is $O(1)$ path length between any two tokens — what's the hidden cost that trades
  against this?"* — computing the full $n\times n$ score matrix costs $O(n^2)$ in both compute and
  memory, growing quadratically with sequence length; this is exactly the cost Flash Attention,
  sliding-window attention, and ultimately Mamba/SSMs (§11) are each responding to in a different way.

### Whiteboard-ready derivations

1. **Vanishing gradient magnitude after $n$ steps** — with recurrent weight magnitude $|W_h| < 1$,
   the gradient after $n$ repeated multiplications is $|W_h|^n$; e.g. $0.70^9\approx0.040$ (§3.1).
2. **Why $\text{Var}(q\cdot k) = d_k$** — for independent, unit-variance components $q_i, k_i$, the
   dot product $q\cdot k = \sum_{i=1}^{d_k} q_i k_i$ sums $d_k$ independent terms, each contributing
   variance 1 to the sum (since $\text{Var}(q_i k_i)=1$ under these assumptions), giving total
   variance $d_k$; dividing the score by $\sqrt{d_k}$ therefore divides the variance by $d_k$,
   restoring it to 1 (§9).
3. **A full attention output computed by hand** — reproduce §8's 6-step derivation (embeddings →
   project to Q/K/V → score → scale → softmax → weighted sum) for the "I am happy" toy example,
   ending in the same $\mathbf{o}_1=[0.28,1.00]$, $\mathbf{o}_2=[1.72,1.00]$, $\mathbf{o}_3=[1.00,1.00]$.

### Applied scenario — Amazon customer review summarization

**Framing:** Amazon wants to generate a short, faithful summary of a product's thousands of customer
reviews, surfaced on the product detail page.

**Data:** Variable-length customer review text (from a few words to several paragraphs), at massive
scale across millions of products.

**Model:** This is exactly the encoder-decoder Seq2Seq setting (§4) — input: concatenated/aggregated
review text; output: a short summary — but the input can be *very* long (thousands of reviews), which
is precisely where §5's bottleneck problem would be most damaging: a single fixed-size context vector
cannot faithfully represent thousands of reviews' worth of information. This motivates attention (§6)
at minimum, and in practice a full Transformer encoder-decoder (self-attention throughout, per §7-§10)
so the model can directly relate any review sentence to any other, regardless of position, without
the RNN's sequential-distance penalty.

**Metric:** Not just BLEU/ROUGE against reference summaries — critically, **faithfulness** (does the
summary state anything not supported by the reviews?) and **coverage** (does it reflect the actual
distribution of opinions, not just the most verbose review?), since a hallucinated but fluent summary
is worse than a slightly awkward faithful one for a customer-facing feature.

**Failure modes:** The $O(n^2)$ cost of full self-attention (flagged as a depth probe above) becomes
a real constraint at "thousands of reviews" scale — this is exactly why §11's efficiency techniques
(sliding-window attention, or a two-stage architecture that first clusters/samples reviews before a
full-attention pass) matter operationally, not just academically. A model relying on one head's
learned coreference resolution could also mis-resolve a pronoun across review boundaries if reviews
are concatenated without clear separators.

**What you'd ship:** An encoder-decoder Transformer (or, in practice, a pretrained LLM used in a
summarization configuration — the direct target of Part 3's "BERT, GPT, T5 are just configurations of
this architecture" [slide 136]) with an efficient-attention variant for the long-input encoder side,
evaluated on faithfulness/coverage metrics with human spot-checks, not automatic metrics alone.

**Leadership Principle tie-in:** **Customer Obsession** — a summarization feature that hallucinates
a product claim not actually supported by reviews directly harms customer trust, which is exactly why
faithfulness has to be a first-class metric, not an afterthought to fluency. **Dive Deep** — diagnosing
*why* a long-review-set summary degrades (bottleneck vs. quadratic-attention memory limits vs.
coreference errors across review boundaries) requires distinguishing the specific mechanisms this
lecture built, not treating "the model got confused" as a sufficient explanation.

---

## Glossary

- **Additive (Bahdanau) attention** — scoring function using a small learned feedforward network:
  $\mathbf{v}^\top\tanh(\mathbf{W}_1\mathbf{s}+\mathbf{W}_2\mathbf{h})$.
- **Attention** — computing a weighted sum over a set of states, with weights determined by learned
  relevance to a query.
- **Bottleneck problem** — the failure mode of compressing an entire input sequence into one
  fixed-size vector; degrades with input length.
- **Character CNN** — convolutional filters over raw characters, producing a word embedding with no
  fixed vocabulary lookup; ELMo's Layer 0.
- **Context vector** — in plain Seq2Seq, the single fixed-size encoder output the decoder reads; in
  attention, a per-decoder-step weighted sum $\mathbf{c}_t$ instead.
- **Dot-Product (Luong) attention** — scoring function $\mathbf{s}^\top\mathbf{h}$, no extra learned
  parameters.
- **ELMo** — Embeddings from Language Models (Peters et al., 2018); contextual embeddings from a deep
  bidirectional LSTM.
- **Encoder-Decoder (Seq2Seq)** — two RNNs: one reads the input to a context vector, one generates
  output from it.
- **Flash Attention** — an IO-aware implementation of the same scaled dot-product attention
  computation, 2–4× faster, mathematically identical output.
- **General (Luong) attention** — scoring function $\mathbf{s}^\top\mathbf{W}\mathbf{h}$, one learned
  matrix.
- **GQA (Grouped Query Attention)** — inference-memory optimization where groups of query heads share
  Key/Value projections.
- **Key (K)** — a token's learned "what do I contain" projection; matched against queries.
- **Mamba / SSMs (State Space Models)** — a non-attention sequence architecture, $O(n)$ in sequence
  length.
- **MLA (Multi-head Latent Attention)** — inference-memory optimization compressing Key/Value into a
  low-rank representation.
- **Multi-head attention** — running $h$ independent attention computations in parallel (each with
  its own Q/K/V projections), then concatenating and projecting.
- **Query (Q)** — a token's learned "what am I looking for" projection.
- **Scaled dot-product attention** — $\text{score}=\dfrac{\mathbf{s}^\top\mathbf{h}}{\sqrt{d_k}}$;
  the dot-product score divided by $\sqrt{d_k}$ to prevent softmax saturation.
- **Self-attention** — attention applied within a single sequence (Q, K, V all from the same input),
  rather than between two different sequences.
- **Softmax saturation** — when large-magnitude inputs push a softmax toward a near-one-hot output,
  collapsing its gradient.
- **Value (V)** — a token's learned "what do I give when matched" projection; what's actually
  aggregated in the weighted sum.
- **Vanishing gradient** — a gradient that shrinks toward zero as it's backpropagated through many
  sequential steps or layers.

---

## Check yourself

1. Give the specific example sentence this lecture uses to show a static embedding failing, and
   explain in one sentence why it fails. *(§1)*
2. List ELMo's five architectural components in order, and explain what problem the character CNN
   (Layer 0) specifically solves. *(§2)*
3. Compute $0.80^9$ and explain what this number would represent in the RNN vanishing-gradient
   context. *(§3.1)*
4. Explain the difference between the "long-range problem" and the "parallelism problem" — are they
   the same failure described twice, or two distinct failures? *(§3)*
5. Why does the encoder-decoder bottleneck get worse specifically for *longer* sentences, rather than
   being a constant-severity problem? *(§5)*
6. Write out attention's three-step computation (score, normalize, aggregate) in your own words, then
   in the formulas from §6.2. *(§6.2)*
7. Name all four attention scoring functions and order them by number of extra learned parameters,
   fewest to most. *(§6.3)*
8. What exactly is "still missing" at the end of §6.4, and what section fixes it? *(§6.4)*
9. Explain the database-lookup analogy for Q, K, V, and state which of the three is used to compute
   the attention *score* versus which is used to compute the *output*. *(§7.2)*
10. Derive, from a variance argument, why scaled dot-product attention divides by $\sqrt{d_k}$ rather
    than by $d_k$ or by some other function of $d_k$. *(§9)*
11. In the multi-head attention formula, what does concatenating $h$ heads and multiplying by
    $\mathbf{W}^O$ accomplish, and why is $d_k=d_v=d_{\text{model}}/h$ chosen specifically? *(§10)*
12. Name two of the four head specializations observed empirically, and connect one of them back to
    a specific earlier example in this lecture. *(§10.1)*
13. Explain the distinction the lecture draws between "an optimization of attention" and "a different
    mechanism than attention," and classify Flash Attention, GQA, and Mamba/SSMs accordingly. *(§11)*

---

## Going deeper

1. **Peters et al. (2018), "Deep contextualized word representations"** — the original ELMo paper,
   cited by name and year directly on the lecture's own slides [slide 6]. `solid` · primary source
   for §2.
2. **Bahdanau, Cho, Bengio (2014/2015), "Neural Machine Translation by Jointly Learning to Align and
   Translate"** — the original additive-attention paper; named directly in the scoring-functions
   comparison [slide 62]. `solid` · primary source for §6.3's additive scoring function.
3. **Luong, Pham, Manning (2015), "Effective Approaches to Attention-based Neural Machine
   Translation"** — introduces dot-product and general attention, named directly [slide 62]. `solid`
   · primary source for the other two scoring functions in §6.3.
4. **Vaswani et al. (2017), "Attention Is All You Need"** — the Transformer paper; named directly on
   the evolution timeline [slide 133] as the point where scaled dot-product and multi-head attention
   (§9-§10) were introduced together as a complete architecture. `solid` · essential — this is where
   Part 3 of this course picks up.
5. **Jay Alammar, "The Illustrated Transformer"** — a widely used visual walkthrough of exactly the
   Q/K/V mechanism and multi-head splitting diagrammed in §10; the multi-head diagram in this lecture
   [slide 129] is itself in this illustration's style. `intro` · read before Part 3.
6. **Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** —
   named directly in §11's evolution timeline as the 2022 entry. `hard` · for understanding *why*
   attention is memory-bound in practice, beyond this lecture's scope.
7. **Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"** — named directly
   in §11 as the non-attention alternative on the evolution timeline. `hard` · genuinely different
   mechanism from everything else in this lecture; read only after the attention material is solid.

> **Externally verified** (module enhancement pass, 2026-08-30) — author lists, titles, and years for
> #1–7 above checked against primary listings and confirmed exact: Peters, Neumann, Iyyer, Gardner,
> Clark, Lee & Zettlemoyer, NAACL 2018 (ELMo); Bahdanau, Cho & Bengio, arXiv Sept 2014 / ICLR May
> 2015; Luong, Pham & Manning, EMNLP 2015; Vaswani et al., NeurIPS 2017 (Transformer); Dao, Fu,
> Ermon, Rudra & Ré, NeurIPS 2022 (FlashAttention); Gu & Dao, arXiv Dec 2023 (Mamba). The lecture's
> own slides gave author-surname-plus-year citations only [slides 6, 62, 133] — the fuller detail
> above comes from this external check, not from the slides themselves.
