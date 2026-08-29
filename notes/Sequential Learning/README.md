# Sequential Learning — Amazon ML Summer School

Self-study notes built from the raw slide capture in [`output/`](../../output/) (not
`slides_deduped/`, which is lossy — see the warning in
[`notes/Supervised Learning/README.md`](../Supervised%20Learning/README.md)), following
[`NOTES_PIPELINE.md`](../../NOTES_PIPELINE.md).

These are **teaching documents, not summaries**. They define every term before using it, derive
every formula rather than asserting it, and work every example through to a final number. They are
written to be the only thing you need to read to master the lecture.

This module has gone through a full `QUALITY_REVIEW_PIPELINE.md` audit pass — see
[`QUALITY_REVIEW.md`](QUALITY_REVIEW.md) for the complete findings and fixes. It is the cleanest
module audited in this project to date: one confirmed factual error across all three files (a single
mislabeled column in an attention heatmap, self-correcting two sections later in the same file), zero
fabricated numbers or citations, and every worked-example arithmetic chain independently re-derived
and confirmed correct.

---

## Index

| # | Notes | Source deck | Instructor | Status | Words | Covers |
|---|---|---|---|---|---|---|
| 01 | [sequential-learning-01.md](sequential-learning-01.md) | `Lecture_18 - Module 6 Sequential Learning Part 1` (83 deduped / **150 raw frames**) | Sayambhu Sen | ✅ Complete | ~11,590 | Tokenization & BPE · Stemming vs lemmatization · One-hot, Bag-of-Words, TF-IDF, N-grams · Why discrete representations fail (7 failure modes) · Word2Vec (Skip-Gram, CBOW, negative sampling) · GloVe · FastText |
| 02 | [sequential-learning-02.md](sequential-learning-02.md) | `Lecture_19 - Module 6 Sequential Learning Part 2` (54 deduped / **136 raw frames**) | — (not named in deck; confirmed by direct inspection of the title, closing, and webcam-tile frames) | ✅ Complete | ~9,420 | Static vs contextual embeddings · ELMo (biLSTM, character CNN) · RNN vanishing gradient & parallelism problems · Encoder-Decoder & the bottleneck problem · Attention (four scoring functions) · Self-attention & Q/K/V · Scaled dot-product attention (√d_k derivation) · Multi-head attention |
| 03 | [sequential-learning-03.md](sequential-learning-03.md) | `Lecture_20 - Module 6 Sequential Learning Part 3` (**79 deduped / 160 raw frames** — this file uses the raw set; a deduped folder exists but was not the note's source) | Ahmed Sanin MV | ✅ Complete | ~11,010 | Subword tokenization (BPE, WordPiece, Unigram/SentencePiece) · Positional encoding (sinusoidal, learned, RoPE/ALiBi) · Full Transformer block & architecture · Causal masking · BERT (MLM, NSP, fine-tuning) · GPT (next-token, sampling) · PEFT ladder (adapters→LoRA→QLoRA) · T5 · hands-on notebook (0.70 vs 0.88 test accuracy, confirmed against the notebook's own captured bar chart) |

**Prerequisites.** This module leans on [`GenAI & LLM`](../GenAI%20&%20LLM/) for substantially
overlapping content: `genai-llm-01.md` (lecture 14, taught *before* this module) already derives
self-attention, Q/K/V, scaled dot-product attention, multi-head attention, and positional encoding
(including RoPE/ALiBi) as infrastructure for LLMs, and `genai-llm-03.md` §8–9 covers LoRA/QLoRA in
depth with real numbers. Sequential Learning Parts 2–3 independently re-derive the identical
material as the general NLP-representation lineage (Word2Vec → attention → Transformer). Both
modules' independently-checked claims agree with each other everywhere — see `QUALITY_REVIEW.md`'s
"Module-wide observations" section for the specific cross-checks — so reading either module first is
fine, but expect the Transformer to be derived twice if you read both.

---

## Capture quality

### ✅ Lecture 18 — excellent

150 raw frames over 57:24. Every content slide has a fully-built state; the deck is a clean,
self-contained five-part arc (Preprocessing → Discrete Representations → Why Discrete Fails →
Distributed Representations/Word2Vec → GloVe & FastText) with nothing found truncated or missing.
This lecture's own quality-review pass found the module's cleanest results: zero factual errors, and
every worked-example arithmetic chain (BPE, TF-IDF, negative sampling, the GloVe ratio table)
independently re-derived and confirmed exact. The only findings were a dropped third example word in
one worked negative-sampling example and a stale frontmatter slide count — both fixed.

**Instructor: Sayambhu Sen**, named directly on the recording's webcam-tile overlay throughout.

### ✅ Lecture 19 — excellent, unusually rigorous derivations

136 raw frames over 1:12:00. Every content slide has a fully-built state. This lecture's own
quality-review pass independently re-derived its highest-fabrication-risk content by hand — the full
7-step "I am happy" self-attention computation (embeddings → Q/K/V → scores → scaling → softmax →
weighted sum) and the $\sqrt{d_k}$ variance argument at $d=32$ — and found every number exactly
correct against the source slides. The one confirmed factual error in this entire module lives here:
a misidentified attention-heatmap column (§7.1 said "tired," the slide shows "animal"), caught
because the file's own §10.1 table states the correct answer two sections later. Now fixed.

**No instructor is named** anywhere in this recording — independently reconfirmed during the quality
review by inspecting the title, closing, and webcam-tile frames directly (slides 1, 2, 4, 5, 135,
136); no name overlay appears anywhere, unlike Lectures 18 and 20.

### ✅ Lecture 20 — excellent, and the most code-heavy, interactive-demo-driven deck in the module

160 raw frames over ~1:03:45. Every precise number this lecture makes — the BPE/WordPiece/Unigram
worked examples, the causal-mask matrix, the GPT prediction-head softmax, and critically the hands-on
notebook's headline **0.70 (from-scratch) vs. 0.88 (pre-trained) test accuracy** result — was
independently re-derived by hand during the quality review and confirmed to match the source frames
digit for digit; the notebook figure is the literal captured bar-chart value, not a paraphrase.

A `slides_deduped/Lecture_20.../` folder exists (79 images) even though this note was built purely
from the 160-frame `output/` capture, matching Lectures 18–19's raw-frame sourcing discipline — the
deduped folder was simply never the note's source, now reflected consistently in the Index table
above. One genuine capture gap was found and disclosed: the interactive 4-step PEFT walkthrough
(`slide 132` → `133` → `134`) is missing its step-3/4 ("LoRA") frame — the presenter evidently
clicked through it too quickly for a stable frame to be captured (58 seconds elapse between slides
133 and 134, versus 1.5 minutes for the previous step). The note's LoRA description is corroborated
via `GenAI & LLM Part 1` (`genai-llm-03.md` §8) rather than this lecture's own slide, and this is now
disclosed explicitly in the file rather than silently passed through.

**Instructor: Ahmed Sanin MV** (a different instructor from Parts 1–2, with a different, more
code-heavy, interactive-demo-driven deck style), named directly on the recording's webcam-tile
overlay throughout.

---

## What's in Part 1

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — capture note (150 raw frames, five-part arc, nothing truncated) · 9 capabilities ·
4 prerequisites taught from zero (what a word embedding actually is · cosine similarity, including
why one-hot vectors are structurally orthogonal · a neural network hidden layer as a matrix lookup ·
logarithms, enough to read $\text{idf}=\log(N/n_t)$) · a full-lecture ASCII map tracing the five-act
arc from raw text to distributed representations

**1. Tokenization** — word/subword/character levels compared · 🧪 **Byte-Pair Encoding derived and
run by hand** on `"low lower lowest"` · 📚 why BPE's vocabulary size is a hyperparameter, not a fact
about English (the direct explanation for GPT-2's ~50K vs. GPT-4's ~100K vocabularies) · ⚠️ BPE
doesn't know linguistics — it only tracks co-occurrence frequency

**2. Stemming vs. Lemmatization** — Porter stemming vs. dictionary-aware lemmatization, compared
side by side · the five-stage preprocessing pipeline traced on one real sentence, with an explicit
note on why stage *order* is load-bearing (tokenize → lowercase → strip punctuation → remove
stopwords → stem/lemmatize)

**3. Discrete Representations** — One-Hot Encoding (🧪 why cosine similarity is 0 for *every* pair of
distinct words, by construction) · Bag-of-Words (🧪 "Dog bites man" = "Man bites dog," a fatal flaw
distinct from one-hot's) · **TF-IDF derived from words to symbols**, with a full worked example
(cat/the/fish, $N=1000$) and an interview-style quiz with the exact wrong-answer bugs people actually
write · N-grams, and the character-n-gram connection forward to FastText

**4. Why Discrete Representations Fall Short** — seven named failure modes traced back to one root
cause (no notion of similarity between symbols) · 🧪 compositionality worked in full (`"hot" +
"dog"` ≠ food, three more idiom examples) · the single question ("dense, low-dimensional vectors
where similar words are nearby?") the rest of the lecture answers

**5. Distributed Representations** — sparse vs. dense compared · the Distributional Hypothesis
(Firth, 1957) as the theoretical foundation for everything that follows · the king−man+woman≈queen
analogy, with an honest caveat that this is emergent, not guaranteed

**6. Word2Vec** — Skip-Gram and CBOW derived as mirror-image objectives · 📚 softmax explained from
scratch · 💡 why Word2Vec keeps two vectors per word · ⚠️ "faster" (CBOW) ≠ "better" — Skip-Gram's
$2m$ predictions per window is exactly why it learns better rare-word embeddings · **negative
sampling derived from the softmax cost problem**, with a full worked update on 'king'/'crown' plus
three negative words · the $3/4$-power noise-distribution smoothing trick

**7. GloVe** — the ice/steam/water/fashion co-occurrence-ratio insight, worked from a real 4×4 table
· the objective derived (reconstructing $\log X_{ij}$ via weighted least squares) · 💡 the same
$3/4$-power smoothing trick reappearing independently in a completely different algorithm

**8. FastText** — character n-gram decomposition derived as the fix for OOV that neither Word2Vec
nor GloVe has · 🧪 the "enviroment"/"environment" typo-robustness worked example · a three-way
comparison table (Word2Vec vs. GloVe vs. FastText) closing on when to reach for each

**9. Evolution of Word Representations** — a timeline from 1960s Vector Space Models through
Word2Vec/GloVe/FastText to ELMo/BERT/GPT · ⚠️ the lecture's own honest gap: every method here is
**static** — one vector per word regardless of sentence — setting up Part 2

**Closing** — a full-lecture ASCII dependency map (Distributional Hypothesis → three independent
routes → the OOV/context-fixing frontier) · **12 interview questions** with model answers (2
combining concepts) · 3 depth probes · **3 whiteboard derivations** · an Amazon product-search
scenario (TF-IDF + FastText + dense re-ranking, built around the "sneekers" typo case) · 2 Leadership
Principles · **25-term glossary** · 12 check-yourself questions · 8 ranked resources.

**Interactive specs:** 8 — BPE merge walkthrough, the five-stage preprocessing pipeline, TF-IDF vs.
document frequency, the seven-failures diagram, the Skip-Gram window explorer, negative sampling live,
the GloVe ratio explorer, and FastText's word-from-pieces decomposition.

</details>

---

## What's in Part 2

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — capture note (136 raw frames, opens by explicitly building on Part 1, closes with
an explicit forward pointer to Part 3 — nothing here is a capture gap) · 9 capabilities · 4
prerequisites taught from zero (what this lecture assumes about RNNs/LSTMs · variance, for the
$\sqrt{d_k}$ derivation · softmax saturation · matrix multiplication as a set of dot products) · a
full-lecture ASCII map tracing the "name a failure, then fix it" chain from static embeddings to
multi-head attention

**1. The Problem with Static Embeddings** — the "river bank" vs. "savings bank" motivating example,
read directly off the slide's own Word2Vec-vs-ELMo diagram

**2. ELMo** — architecture top to bottom (character CNN → forward/backward LSTM → 2 stacked layers →
weighted-sum embedding) · 💡 the character CNN as FastText's exact structural fix, one level up ·
what each of ELMo's three layers empirically captures (morphology → syntax → semantics), flagged as
an insight that outlived the architecture itself · ⚠️ ELMo's own stated limitation: still sequential

**3. Why Not Just Use RNNs?** — the long-range problem (🧪 $0.70^9\approx0.040$, the vanishing-
gradient decay made concrete) and the parallelism problem, explicitly distinguished as two different
failures, not one restated twice

**4. Encoder-Decoder Architecture** — the two-RNN Seq2Seq setup, as the concrete setting attention
needs to be motivated against

**5. The Bottleneck Problem** — why compressing an entire input into one fixed-size vector fails for
long sentences, with the empirical BLEU-score evidence the slide cites

**6. Attention Mechanism** — the three-step computation (score → normalize → aggregate) derived
symbol by symbol · four scoring functions (additive, dot-product, general, scaled dot-product)
compared by parameter count, not "quality" · 💡 the explicit pivot: attention *between* sequences
still doesn't fix an RNN encoder's *internal* sequential bottleneck

**7. Self-Attention** — the "it" → "animal" coreference motivating example · Query/Key/Value derived
from a database-lookup analogy, with the exact projection formulas

**8. Attention Computation, Worked in Full** — 🧪 the complete 7-step scaled dot-product self-attention
computation on a toy "I am happy" sentence, every number independently re-derived by hand during the
quality review and confirmed exact

**9. Scaled Dot-Product Attention** — 🧪 **the $\sqrt{d_k}$ divisor derived from a variance argument**,
not asserted — confirmed against the slide's own $d=32$ unscaled-vs-scaled entropy comparison

**10. Multi-Head Attention** — the formula and a worked $512=8\times64$ sizing example · what
different heads empirically specialize in (positional, syntactic, coreference, semantic), closing the
loop back to §7's motivating example

**11. Attention Is Still Evolving** — Flash Attention, sliding-window attention, GQA, MLA, and
Mamba/SSMs, with an explicit "optimization of the same mechanism vs. a genuinely different mechanism"
framing exercise

**Closing** — a full-lecture ASCII map tracing the RNN's two structural flaws through self-attention's
fix · **11 interview questions** with model answers (3 combining concepts) · 3 depth probes · **3
whiteboard derivations** · an Amazon customer-review-summarization scenario (long-input encoder-
decoder, faithfulness/coverage metrics) · 2 Leadership Principles · **21-term glossary** · 13
check-yourself questions · 7 ranked resources.

**Interactive specs:** 6 — ELMo's two fixes and one remaining problem, the vanishing gradient forward
and backward, the bottleneck visualized, self-attention vs. encoder-decoder attention, the full
attention computation step by step, and one head vs. eight heads.

</details>

---

## What's in Part 3

<details>
<summary><b>Full table of contents</b> (click to expand)</summary>

**Front matter** — capture note (160 raw frames; the deck's own three navigable tabs — Sub-word
Tokenization / Transformer Architecture / Language Models — mostly but not perfectly matched by this
file's structure, with one disclosed exception in §6) · 10 capabilities · 4 prerequisites taught from
zero (what this lecture assumes from Parts 1–2 · the notation table the deck fixes and reuses
throughout · LayerNorm, briefly · cross-entropy loss, briefly) · a three-act ASCII map matching the
deck's own three-tab structure

**PART 1 — Sub-word Tokenization** (§1–§5)
- §1 Why word-level and character-level both fail, quantified ($O(n^2)$ attention cost means a 6×
  longer character sequence costs ~36× the compute)
- §2 **BPE**, training and inference · 🧪 tokenizing an unseen word (`"bug"`) with learned merge
  rules · ⚠️ a genuine OOV case *inside* "OOV-free" BPE (`"mug"` → `[UNK]` if `m` never appeared) ·
  why byte-level BPE (GPT-2) makes this structurally impossible
- §3 **WordPiece** · 🧪 the score formula worked against BPE's frequency rule on the same corpus,
  showing the two algorithms genuinely disagree on which pair to merge first
- §4 **Unigram LM / SentencePiece** · 🧪 segmenting `"dogs"` by comparing four candidate splits'
  probabilities directly
- §5 All three compared side by side, closing on the thesis: one outer loop, three different
  selection rules (frequency / score / probability)

**PART 2 — Transformer Architecture** (§6–§13)
- §6 A model sees numbers, not text — the literal handoff from Part 1's output (token IDs) into the
  Transformer's input
- §7 Positional encoding — the problem (attention alone is a bag-of-tokens operation) and three
  fixes compared: sinusoidal, learned, and the field's move to **RoPE and ALiBi**
- §8 Inside one Transformer block — the full Add & Norm / attention / feed-forward diagram
- §9 Self-attention as Q/K/V, reusing (not re-deriving) Part 2's scaled dot-product formula · §9.1
  multi-head attention, with 🧪 a worked example showing exactly what $W_O$ contributes that plain
  concatenation cannot
- §10 The feed-forward sub-layer, with GELU derived as a smooth gate
- §11 Encoder vs. decoder — the one real difference (what each token may attend to) · §11.1 **causal
  masking derived**: why $-\infty$ before softmax, not zeroing weights after · §11.2 the two changes
  that turn an encoder block into a decoder block
- §12–§13 The full architecture diagram and the decoder's prediction head

**PART 3 — Language Models** (§14–§21)
- §14 Three families, one paradigm: pre-train once, adapt cheaply — "the text is its own label"
- §15 **BERT** — bidirectional encoding · the three-embedding input sum · 🧪 **the 80/10/10 masking
  trick derived from the train/inference mismatch it prevents** · Next Sentence Prediction, with the
  lecture's own honest flag that RoBERTa later dropped it · fine-tuning as "reuse the body, add a
  small task head"
- §16 **GPT** — the autoregressive factorization · 🧪 a full worked next-token prediction (hidden
  vector → logits → softmax → cross-entropy loss) · greedy, temperature, top-k, and top-p sampling
  compared
- §17 Adapting a pre-trained decoder — prompting, instruction tuning, and RLHF, ordered by how much
  of the model each one touches
- §18 **The PEFT ladder** — full fine-tuning → adapters → LoRA → QLoRA, each rung named as fixing
  the previous rung's biggest remaining cost · ⚠️ a disclosed capture gap (the LoRA-specific step of
  the deck's own interactive walkthrough was never captured as a stable frame; corroborated instead
  via `GenAI & LLM`)
- §19 **T5** — bolting BERT's encoder onto GPT's decoder via cross-attention, with span-denoising
  pre-training
- §20 Recap — one block, three families, one paradigm, in a single summary table
- §21 **The hands-on notebook** — a controlled from-scratch-vs-pre-trained experiment isolating
  pre-training's effect · 🧪 the headline **0.70 → 0.88 test-accuracy result**, confirmed against the
  notebook's own captured bar chart, not a paraphrase

**Closing** — a three-act ASCII map (tokenization → Transformer → language models, closing on "same
block, different mask, different objective") · **11 interview questions** with model answers (3
combining concepts) · 3 depth probes · **3 whiteboard derivations** · an Amazon customer-support-
ticket-triage scenario (encoder-only + LoRA fine-tuning) · 2 Leadership Principles · **24-term
glossary** · 14 check-yourself questions · 8 ranked resources.

**Interactive specs:** 5 — the BPE tokenizer live, multi-head attention head by head, applying the
causal mask live, sampling-strategy comparison, and the PEFT ladder step by step.

</details>

---

## Reading guide

The three parts total ~32,020 words and form a single escalating argument: from discrete text
representations (Part 1), through the sequential-processing problems that motivate attention (Part
2), to the full Transformer architecture and the three model families built on it (Part 3). Each part
assumes the previous ones — do not skip ahead.

**Prerequisites.** This module assumes basic neural network familiarity (Module 2, Deep Neural
Networks) and substantially overlaps [`GenAI & LLM`](../GenAI%20&%20LLM/)'s coverage of the
Transformer, attention, and PEFT — see the Index section above for the specific cross-references.

**Part 1 → Part 2** (~21,010 words together). Part 2 opens by explicitly assuming Part 1's static
embeddings (Word2Vec/GloVe/FastText) are fresh — its very first sentence is a direct callback.

**Part 2 → Part 3** (~20,430 words together). Part 3's Prerequisite 1 states outright that it does
not re-derive scaled dot-product attention — it uses Part 2's formula immediately as a settled
building block. Read Part 2 first if any of $\text{Attention}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d_k})V$
feels unfamiliar.

**First pass.** Read linearly: Part 1 → Part 2 → Part 3. Do not skip *Before we start* in any file.
The hardest derivations — Part 1 §6.4 (negative sampling), Part 2 §9 (the $\sqrt{d_k}$ variance
argument), and Part 3 §11.1 (causal masking) — are where each lecture's central mechanism actually
gets justified rather than asserted.

**Second pass.** Work every 🧪 example on paper *before* reading the solution. The most important
across all three parts:
1. Part 1 §6.4 — negative sampling's loss function and the direction of its gradient push
2. Part 1 §7.1 — the GloVe ratio-cancellation argument on the ice/steam table
3. Part 2 §8 — the full "I am happy" attention computation, all seven steps
4. Part 2 §9 — deriving $\sqrt{d_k}$ from the variance argument
5. Part 3 §11.1 — the causal mask, and why $-\infty$-before-softmax specifically
6. Part 3 §15 — BERT's 80/10/10 masking trick and the mismatch it prevents
7. Part 3 §21 — the from-scratch-vs-pre-trained notebook comparison, and what was held constant

**Before an interview.** Each file's *Putting it together*, then all 9 whiteboard derivations (3 per
part), then the depth-probe tables. The three highest-value derivations: **the $\sqrt{d_k}$ scaling
argument** (Part 2 §9), **the causal mask's $-\infty$-before-softmax mechanism** (Part 3 §11.1), and
**BERT's 80/10/10 trick** (Part 3 §15.3).

**The three questions this module is most likely to be examined on:**
1. *"Derive why scaled dot-product attention divides by $\sqrt{d_k}$"* (Part 2 §9)
2. *"Walk through self-attention's Q/K/V mechanism end to end"* (Part 2 §7–§8)
3. *"Compare BERT, GPT, and T5 by attention pattern and pre-training objective"* (Part 3 §20)

**When someone asks "why not just use an RNN?"** Part 2 §3 for the two distinct structural failures
(vanishing gradients vs. parallelism), then Part 2 §7's "it → animal" example for why self-attention
fixes both at once.

**When someone asks "why does fine-tuning need so little data?"** Part 3 §15.5's "one body, many
heads" framing, then §21's 0.70→0.88 notebook result as the direct empirical argument.

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
| 01 | Every discrete text representation (one-hot, Bag-of-Words, TF-IDF) fails for the same structural reason — no notion of similarity between symbols — and Word2Vec, GloVe, and FastText are three different statistical routes to the same fix: dense, learned vectors where distance actually means something, with FastText additionally solving the out-of-vocabulary problem the other two share. |
| 02 | RNN/LSTM architectures are structurally sequential (each hidden state depends on the last), which causes both vanishing gradients over long sequences and an inability to parallelize training — and attention, then self-attention, are the direct fix: replacing a fixed-size bottleneck with a learned, per-step weighted lookup over every position at once, giving any two tokens an O(1) path between them regardless of distance. |
| 03 | Subword tokenization, positional encoding, and parameter-efficient fine-tuning are three instances of the same "meet in the middle" move at three different levels of the stack — and BERT, GPT, and T5 are not three different architectures but the identical Transformer block configured along just two axes (attention pattern and pre-training objective), a reframing confirmed empirically when the hands-on notebook shows pre-training alone lifting test accuracy from 0.70 to 0.88 on an otherwise identical model. |
