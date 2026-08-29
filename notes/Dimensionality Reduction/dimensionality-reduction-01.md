---
title: "Dimensionality Reduction — Part 1: The Curse, Feature Selection, and the Linear Algebra Behind PCA"
topic: dimensionality-reduction
lecture: 07
source: "output/Lecture_07 - Module 3 Dimensionality Reduction Part 1"
slides: 40
video: "https://www.youtube.com/watch?v=cN_r6piA3PU"
runtime: "57:33"
---

# Dimensionality Reduction — Part 1
### Why high dimensions break everything, the three families of feature selection, and the linear algebra you need before PCA

---

## 📋 About this lecture and its capture

This is the module's foundation lecture, and it does two jobs that look unrelated but aren't:

| | Section | Runtime | Covers |
|---|---|---|---|
| **1** | **Curse of Dimensionality** | 3:37 – 21:28 | Volume concentration · distance concentration · empty space · what actually breaks |
| **2** | **Why Dimensionality Reduction?** | 21:28 – 26:54 | Three goals · the manifold hypothesis · production benefits |
| **3** | **Selection vs Extraction** | 26:54 – 31:40 | Two philosophies · the taxonomy of every method in this module |
| **4** | **Wrapper Methods** | 31:40 – 35:30 | Core idea · forward selection · RFE |
| **5** | **Filter Methods** | 35:30 – 41:58 | Core idea · variance threshold · Pearson · mutual information · χ²/ANOVA |
| **6** | **Embedded Methods** | 41:58 – 49:42 | Core idea · Ridge · Lasso · Elastic Net · regularization paths · tree importance |
| **7** | **Linear Algebra Refresher** | 49:42 – 57:33 | Covariance matrix · eigendecomposition · spectral decomposition · scree plot · **preview of PCA** |

Sections 1–6 are **feature selection** — keeping a subset of the columns you already have. Section 7
is the mathematical groundwork for **feature extraction**, which is Part 2's entire subject. The
lecture ends on a slide titled *"Up next: PCA, SVD, and Beyond"*, so it is explicitly a setup lecture
for the two that follow.

The deck contains **40 distinct slide states** (33 content slides plus 7 section dividers).

> ✅ **Capture quality: excellent.** 94 raw frames over 57 minutes. Every content slide has a
> fully-built state, every equation and citation is legible, and the deck cites its own sources on
> nearly every slide — which is unusual and makes the *Going deeper* section unusually trustworthy.
>
> **Two small gaps, both of the same kind.** The deck contains three `Quick check` quiz slides. The
> third one's **answer was captured** as an on-click reveal at 46:04. The first two quizzes'
> answers were **not** captured:
>
> | Quiz | Question | On screen | Answer captured? |
> |---|---|---|---|
> | Q1 [slide 24] | kNN on 1,000 samples × 500 features | 16:45 – ~21:00 | ❌ No |
> | Q2 [slide 46] | Zero Pearson, high mutual information | 30:56 – ~31:40 | ❌ No |
> | Q3 [slide 70] | 50 features correlated in groups | 45:38 – 46:04 | ✅ **Yes** — "Answer: C" |
>
> Because Q3's answer *is* a reveal, the deck's format clearly includes answer reveals — so Q1's and
> Q2's answers were most likely shown on screen during a sampling gap rather than never shown at all.
> Both questions have unambiguous answers that the deck's own slides fully determine, so §5 and §11
> give them in full under a **🩹 badge** meaning *"answered from the deck's own content, not read off
> an answer slide."*
>
> One more note: the instructor's name is **not displayed** anywhere in the recording — the webcam
> tile carries no name label. So this file, unlike the others in this course, cannot attribute the
> lecture.

---

## How to read this document

The lecture has a clean two-act structure, and knowing it up front makes everything easier:

```
ACT I  — WHY you must reduce dimensions        §1–§8
         high-D geometry is pathological, and real data doesn't need all those dimensions

ACT II — HOW, part one: keep a SUBSET of columns    §9–§26
         three families, in increasing order of how much they know about your model:
           filter    — knows nothing about the model        (fast, blind to interactions)
           wrapper   — knows only the model's score         (slow, sees everything)
           embedded  — is part of the model's training      (the practical default)

         HOW, part two: BUILD NEW columns  →  needs eigenvectors  →  §27–§31
         (and Part 2 of this module is that whole story)
```

If you are revising under time pressure: **§2 and §22–§23 are the interview core.** "Why does kNN
fail in high dimensions?" and "Why does L1 give sparsity but L2 doesn't?" are the two questions this
lecture is most likely to be examined on, and both have answers you can *derive* rather than recite.

Everything in a `🧪 Worked example` block should be reproducible by you on paper.

---

## What you'll understand after reading this

- You'll be able to **derive** the volume of a $d$-dimensional ball, compute what fraction of its
  bounding cube it fills at $d = 3, 10, 20$, and explain why "most of the volume is in the corners."
- You'll be able to **prove** that pairwise distances concentrate — showing that their spread stays
  *constant* while their mean grows like $\sqrt{d}$ — and use that to predict at what dimension kNN
  becomes unreliable.
- You'll be able to compute how many samples you'd need to fill a $d$-dimensional grid, and explain
  why 1,000 samples in 20 dimensions is effectively an empty space.
- You'll be able to name five things that break in high dimensions and give the mechanism for each.
- You'll be able to state the **manifold hypothesis** precisely and say what dimensionality reduction
  assumes about your data that could be false.
- You'll be able to distinguish **feature selection** from **feature extraction** on five axes, and
  place any named method into the module's taxonomy.
- You'll be able to explain wrapper, filter and embedded methods by **what each one knows about the
  model**, and compute the cost of forward selection and RFE in model fits.
- You'll be able to choose the right univariate filter for any feature-type/target-type combination,
  and explain what a Pearson correlation of exactly zero does *not* tell you.
- You'll be able to compute mutual information by hand on a small discrete example, and show a case
  where $r = 0$ while $I(X;Y) > 0$.
- You'll be able to run a χ² test of independence by hand to a final statistic.
- You'll be able to explain L1's sparsity **geometrically** (the diamond's corners lie on the axes)
  *and* **analytically** (the gradient has constant magnitude), and say when each explanation is the
  better one to give.
- You'll be able to explain why Elastic Net groups correlated features, by showing that L2 breaks a
  tie Lasso is indifferent to.
- You'll be able to read a regularization path and explain what the *order* of coefficient deaths means.
- You'll be able to compare Gini and permutation importance on speed, bias, and interaction capture,
  and say when Gini importance will mislead you.
- You'll be able to compute a covariance matrix, find its eigenvectors and eigenvalues by hand, and
  explain what each one means geometrically.
- You'll be able to state the spectral decomposition, connect it to the SVD, and explain in one
  sentence what PCA is — before Part 2 teaches it.

---

## Before we start: what you need to know

This lecture is more mathematically self-contained than most in the course, but it assumes six things.

### Prerequisite 1 — The data matrix, and the $n$ vs $p$ notation

> **Data matrix $X$** — your dataset as a grid: one **row** per example, one **column** per feature.

| Symbol | Read it as | What it means |
|---|---|---|
| $n$ | "n" | Number of **samples** (rows). |
| $p$ or $d$ | "p" / "d" | Number of **features** (columns). **This deck uses both** — $p$ in the curse-of-dimensionality and selection sections, $d$ in the geometry and linear-algebra sections. They mean the same thing. |
| $X$ | "X" | The $n \times p$ data matrix. |
| $x_i$ | "x sub i" | One **sample** — row $i$, a vector of length $p$. |
| $X_j$ or $x_j$ | "X sub j" | One **feature** — column $j$, a vector of length $n$. |
| $y$ | "y" | The target vector, length $n$. |
| $p \gg n$ | "p much greater than n" | More features than samples. The regime where everything breaks. |

> ⚠️ **The $i$-vs-$j$ convention matters and the deck switches freely.** $i$ indexes samples, $j$
> indexes features. When you see $\mathrm{Var}(X_j)$ that is "the variance of feature $j$ computed
> across all $n$ samples" — a column statistic. When you see $x_i$ it's one whole example. Reading
> $\Sigma_{jk} = \mathrm{Cov}(x_j, x_k)$ in §27 correctly depends on getting this right.

*Concretely:* a genomics dataset with 200 patients and 20,000 genes has $n = 200$, $p = 20{,}000$.
That is $p \gg n$ by a factor of 100, and it is exactly the regime this lecture exists for.

### Prerequisite 2 — Variance, covariance and correlation

You need all three, and the distinction between the last two is the whole content of §23.

> **Variance** — how spread out one variable is. $\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2]$.
>
> *Concretely:* $\{2, 4, 6, 8\}$ has mean 5, deviations $-3,-1,1,3$, squares $9,1,1,9$, so
> $\mathrm{Var} = \frac{20}{4} = 5$.

> **Covariance** — how two variables move *together*.
> $\mathrm{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$.
>
> *In everyday words:* when $X$ is above its average, is $Y$ usually above its average too?
>
> *Concretely:* height and weight have positive covariance. Height and shoe-size-measured-in-metres
> also have positive covariance, but a *smaller number* — because covariance carries the units of both
> variables multiplied together, so changing units changes the value.
>
> *Why that's a problem:* you cannot compare $\mathrm{Cov} = 3.2$ against $\mathrm{Cov} = 0.008$ and
> conclude anything, because they might be in different units.

> **Correlation** — covariance with the units divided out.
> $r_{XY} = \dfrac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y}$, always in $[-1, +1]$.
>
> *Why it exists:* it makes covariances comparable across variables measured in different units. It is
> the *only* one of the three that is a pure number.

**Note $\mathrm{Cov}(X,X) = \mathrm{Var}(X)$** — variance is just covariance with itself. That one
fact is why the covariance *matrix* in §27 has variances down its diagonal.

### Prerequisite 3 — Big-O notation, and why it decides which method you can use

> **$\mathcal{O}(f(p))$** — "grows no faster than $f(p)$, up to a constant." A statement about how
> cost scales as the problem grows, ignoring constant factors.
>
> *Concretely:* $\mathcal{O}(p)$ means doubling the features doubles the cost. $\mathcal{O}(p^2)$
> means doubling the features quadruples it. $\mathcal{O}(2^p)$ means **adding one feature doubles
> it** — which is why exhaustive subset search is impossible.

Put numbers on it, because this table decides §12–§26's entire argument:

| Cost | $p = 10$ | $p = 100$ | $p = 1{,}000$ |
|---|---|---|---|
| $\mathcal{O}(p)$ — filters | 10 | 100 | 1,000 |
| $\mathcal{O}(p^2)$ — pairwise correlations | 100 | 10,000 | 1,000,000 |
| $\mathcal{O}(p \cdot d)$ — forward selection ($d{=}20$) | 200 | 2,000 | 20,000 |
| $\mathcal{O}(2^p)$ — exhaustive search | 1,024 | $1.3\times10^{30}$ | $10^{301}$ |

**The last row is why feature selection is a research area rather than a solved problem.** There are
about $10^{80}$ atoms in the observable universe; $2^{100}$ subsets of 100 features is $10^{30}$,
which is tractable-sounding until you realise each one requires training a model.

### Prerequisite 4 — Entropy, for §11 and §18

> **Entropy $H(X)$** — the average number of bits needed to describe an outcome of $X$; equivalently,
> how *uncertain* you are about it.
>
> $$H(X) = -\sum_x p(x)\log_2 p(x)$$
>
> *In everyday words:* a fair coin has 1 bit of entropy — you need one yes/no answer to learn the
> result. A two-headed coin has 0 bits — you already know.
>
> *Concretely:* a fair coin: $H = -(0.5\log_2 0.5 + 0.5\log_2 0.5) = -(−0.5 − 0.5) = \mathbf{1}$ bit.
> A biased coin with $p = 0.9$: $H = -(0.9\log_2 0.9 + 0.1\log_2 0.1) = -(0.9 \times -0.152 + 0.1
> \times -3.322) = 0.137 + 0.332 = \mathbf{0.469}$ bits — less uncertain, so fewer bits.
> A fair die: $H = \log_2 6 = \mathbf{2.585}$ bits.
>
> *Why it exists:* it turns "how much do I not know?" into a number, which lets you then ask "how much
> did learning $Y$ reduce it?" — and that quantity is mutual information.

> **Conditional entropy $H(X \mid Y)$** — how uncertain you still are about $X$ *after* learning $Y$.
> If $Y$ determines $X$ completely, $H(X \mid Y) = 0$.

### Prerequisite 5 — Matrix transpose and the $X^\top X$ product

> **Transpose $X^\top$** — flip the matrix over its diagonal: rows become columns.
> If $X$ is $n \times p$, then $X^\top$ is $p \times n$.

**The one product you must be able to read:** if $X$ is $n \times p$, then $X^\top X$ is
$p \times p$ — a **feature-by-feature** matrix — and its $(j,k)$ entry is

$$\left(X^\top X\right)_{jk} = \sum_{i=1}^{n} X_{ij}\,X_{ik}$$

i.e. the dot product of column $j$ with column $k$: *"sum over all samples of feature $j$ times
feature $k$."* Divide by $n$ and, for centered data, that is exactly the covariance. §27 is that
sentence.

> 💡 **The shape check that saves you every time.** $(n \times p)^\top (n \times p) = (p \times n)(n
> \times p) = p \times p$. The inner dimensions ($n$) cancel; the outer ones survive. If you ever
> can't remember whether it's $X^\top X$ or $XX^\top$, ask which shape you want: $p\times p$ (a
> covariance over features) or $n \times n$ (a Gram matrix over samples). Both are used in this
> module — $X^\top X$ for PCA, $XX^\top$ for kernel PCA.

### Prerequisite 6 — Linear regression and the regularized objective

§21–§23 are all variations on one expression, so make sure you can read it:

$$\min_{\beta}\ \underbrace{\|y - X\beta\|^2}_{\text{fit the data}} + \underbrace{\lambda\,\Omega(\beta)}_{\text{stay simple}}$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\beta$ | "beta" | The coefficient vector, length $p$ — one number per feature. **This is what you're solving for.** |
| $X\beta$ | "X beta" | The model's predictions: a length-$n$ vector. |
| $\|y - X\beta\|^2$ | "norm squared of y minus X beta" | The **residual sum of squares** — total squared prediction error. Ordinary least squares minimises this alone. |
| $\Omega(\beta)$ | "omega of beta" | The **penalty**: some measure of how big $\beta$ is. Which one you pick is the entire difference between Ridge, Lasso and Elastic Net. |
| $\lambda$ | "lambda" | How much you care about the penalty relative to the fit. $\lambda = 0$ is plain OLS. |

> 📚 **Background the deck assumed** — *the two norms*
>
> $$\|\beta\|_1 = \sum_j |\beta_j| \qquad\qquad \|\beta\|_2^2 = \sum_j \beta_j^2$$
>
> The **L1 norm** sums absolute values; the **L2 norm squared** sums squares. *Concretely,* for
> $\beta = (3, -4)$: $\|\beta\|_1 = 3 + 4 = 7$, and $\|\beta\|_2^2 = 9 + 16 = 25$ (so
> $\|\beta\|_2 = 5$ — the ordinary straight-line length).
>
> **The whole of §21–§23 comes from one difference between them:** the set of $\beta$ with
> $\|\beta\|_2 \le t$ is a **round ball**, and the set with $\|\beta\|_1 \le t$ is a **diamond with
> corners on the axes**. Corners on the axes mean coordinates equal to zero. That's it — that's the
> sparsity story, and §22 makes it rigorous.

---

## The big picture

Every dataset you will ever meet has more columns than it needs. The lecture's job is to convince you
that this is not merely wasteful but actively *dangerous*, and then to give you two ways out.

**Act I: high-dimensional space is not what your intuition says it is.**

Your intuition for geometry was trained in two and three dimensions, where it works beautifully. In
500 dimensions it fails in specific, quantifiable ways, and the lecture picks three:

1. **Volume concentrates in the corners.** A ball inscribed in a cube fills 52% of it in 3D and
   **0.25%** in 10D. Almost all of a high-dimensional cube is in places a ball never reaches.
2. **Distances concentrate.** The gap between the nearest and farthest point shrinks *relative to* the
   distance itself. In high enough dimensions, everything is roughly the same distance from
   everything else — which makes "nearest neighbour" a meaningless phrase.
3. **Space is empty.** To sample a $d$-dimensional grid at 10 bins per axis you need $10^d$ points.
   At $d = 20$ that's $10^{20}$. You have a thousand.

These aren't three problems. **They are one problem seen from three angles**, and the angle is:
*volume grows exponentially in $d$, and your sample size does not.*

**Act II: real data doesn't actually live in all those dimensions.**

The escape is the **manifold hypothesis**: although your data is *represented* in 20,000 dimensions,
it *lives* on a much smaller curved surface inside that space. A 100×100-pixel face photo is a point
in 10,000-dimensional space, but the set of things that look like faces is maybe 50-dimensional —
because faces are generated by a modest number of underlying factors (pose, lighting, identity,
expression). If that's true, you can throw away most of the dimensions and lose nothing.

And there are two philosophies for doing it:

- **Feature selection** — *keep a subset of the original columns.* You end up with real, named
  features you can point at. Three families, differing in how much they know about the model:
  **filters** (nothing), **wrappers** (its score), **embedded** (they *are* the model). §12–§26.
- **Feature extraction** — *build new columns as combinations of the old ones.* More powerful, less
  interpretable. PCA is the canonical example, and everything it needs is eigenvectors of the
  covariance matrix — which is why §27–§31 exist and why the lecture ends where it does.

### The whole lecture in one diagram

```
                    p features, and p is too large
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
   WHY IT'S BAD  (Act I)                        WHAT TO DO  (Act II)
   ─────────────────────                        ────────────────────
   §1 volume concentration                      manifold hypothesis §7
      d=3: 52%  →  d=10: 0.25%                  "real data is low-D
   §2 distance concentration                     inside a high-D box"
      spread stays FIXED, mean grows √d                 │
      ⇒ contrast ~ 0.59/√d                              │
   §3 empty space                                       ▼
      10 bins/axis, d=20 ⇒ 10²⁰ points          ┌───────┴────────┐
   §4 what breaks: kNN, kernels,                ▼                ▼
      overfitting, plots, compute          SELECTION         EXTRACTION
                                        keep a subset      build new columns
                                        of the columns     from combinations
                                               │                │
                        ┌──────────────────────┼───────┐        │
                        ▼                      ▼       ▼        ▼
                   §15 FILTER            §12 WRAPPER  §20 EMBEDDED   §27–§31
                   score each feature    train a      penalty inside  LINEAR ALGEBRA
                   alone, then cut       model per    the loss        covariance Σ
                        │                 subset          │           eigen Σv = λv
                   knows NOTHING       knows the      IS the model    Σ = VΛVᵀ
                   about the model     model's score      │           scree plot
                        │                  │              │                │
                   O(p), blind to     O(p·d) fits,   Ridge / Lasso /       ▼
                   interactions       sees everything Elastic Net    ═════════════
                        │                  │              │          PCA = eigen-
                   variance thresh    forward sel.   ridge: circle   decomposition
                   Pearson r          RFE            lasso: diamond  of covariance
                   mutual info                       ⇒ CORNERS ON       (Part 2)
                   χ² / ANOVA                          THE AXES
                                                     ⇒ sparsity
```

---

# ACT I — The Curse of Dimensionality

*3:37 – 21:28*

> **Curse of dimensionality** — the collection of ways in which geometry, statistics and computation
> stop behaving sensibly as the number of features grows.
>
> *In everyday words:* everything you know about space was learned in a world with three dimensions.
> Almost none of it survives the trip to five hundred.
>
> *Why the name:* coined by **Richard Bellman (1961)** in *Adaptive Control Processes*, describing why
> dynamic programming became intractable as state variables were added. The deck cites him on the
> first content slide.

---

## 1. Volume concentration

> *"Hypersphere volume vanishes relative to hypercube"* [slide 13, 9:16]

**Intuition first.** Draw a circle inside a square. The circle takes up most of the square — about
79% — and the four corner regions are small. Now do it in 3D: a ball inside a cube fills 52%, and the
eight corners have grown. Keep going. **The corners keep multiplying (there are $2^d$ of them) while
the ball stays stubbornly round**, and by 10 dimensions the ball is a speck in a mostly-empty box.

The slide gives the volume of a unit-radius $d$-ball:

$$V_d = \frac{\pi^{d/2}}{\Gamma\!\left(\frac{d}{2} + 1\right)}$$

| Symbol | Read it as | What it means |
|---|---|---|
| $V_d$ | "V sub d" | Volume of a ball of radius 1 in $d$ dimensions. |
| $\pi^{d/2}$ | "pi to the d over two" | Grows fast, but only exponentially. |
| $\Gamma(\cdot)$ | "gamma function" | The factorial extended to non-integers — see the background box. It grows **faster than exponentially**, and that's the whole story. |

> 📚 **Background the slide assumed** — *the gamma function*
>
> $\Gamma$ is the factorial, generalised so it accepts halves. For positive integers,
> $\Gamma(m) = (m-1)!$; and $\Gamma(\tfrac12) = \sqrt{\pi}$, with the recurrence
> $\Gamma(z+1) = z\,\Gamma(z)$ generating everything else.
>
> *Concretely:* $\Gamma(1) = 1$, $\Gamma(2) = 1$, $\Gamma(3) = 2$, $\Gamma(4) = 6$, $\Gamma(6) = 120$;
> and $\Gamma(2.5) = 1.5 \times 0.5 \times \sqrt{\pi} = 1.3293$.
>
> **Why it matters here:** $\Gamma$ in the denominator grows *factorially* while $\pi^{d/2}$ in the
> numerator grows only *exponentially*. Factorial beats exponential, so $V_d \to 0$. **The volume of a
> unit ball goes to zero as the dimension grows** — which by itself is startling enough to be worth
> sitting with. It peaks at $d = 5$ ($V_5 = 5.264$) and falls away thereafter.

### 1.1 The three bullets

> - *"**Ratio** $V_{\text{sphere}}/V_{\text{cube}} \to 0$ as $d \to \infty$"*
> - *"Most volume concentrates in the **corners**"*
> - *"Uniform sampling misses the center"*

And the slide's two data points: **$d = 3$: sphere fills 52%**, **$d = 10$: sphere fills 0.25%**.

### 🧪 Worked example — verify both of the slide's numbers, then extend

Put a ball of radius 1 inside a cube of side 2 (so the ball just touches each face). The cube's volume
is $2^d$. So:

$$\text{ratio} = \frac{V_d}{2^d} = \frac{\pi^{d/2}}{\Gamma\!\left(\frac{d}{2}+1\right) 2^d}$$

**$d = 2$** (circle in a square):

$$\frac{\pi^{1}}{\Gamma(2)\cdot 4} = \frac{3.1416}{1 \times 4} = \mathbf{0.785} = 78.5\%$$

**$d = 3$** — the slide's first number:

$$\frac{\pi^{1.5}}{\Gamma(2.5)\cdot 8} = \frac{5.5683}{1.3293 \times 8} = \frac{5.5683}{10.635} = \mathbf{0.5236} = 52.4\% \quad\checkmark$$

**$d = 10$** — the slide's second number:

$$\frac{\pi^{5}}{\Gamma(6)\cdot 2^{10}} = \frac{306.02}{120 \times 1024} = \frac{306.02}{122{,}880} = \mathbf{0.00249} = 0.249\% \quad\checkmark$$

**$d = 20$**, to see where it goes:

$$\frac{\pi^{10}}{\Gamma(11)\cdot 2^{20}} = \frac{93{,}648}{3{,}628{,}800 \times 1{,}048{,}576} = \frac{93{,}648}{3.806\times10^{12}} = \mathbf{2.46\times10^{-8}}$$

| $d$ | Ball as a fraction of its cube |
|---|---|
| 2 | 78.5% |
| 3 | **52.4%** ✓ slide |
| 5 | 16.4% |
| 10 | **0.249%** ✓ slide |
| 20 | 0.0000025% |
| 50 | $\sim 10^{-28}$ |

```python
from math import pi, gamma
for d in (2, 3, 5, 10, 20, 50):
    print(d, pi**(d/2) / (gamma(d/2 + 1) * 2**d))
# 2  0.7853981633974483
# 3  0.5235987755982989
# 5  0.16449340668482262
# 10 0.0024903945701895653
# 20 2.4611373261172552e-08
# 50 1.5375216340608578e-28
```

### 1.2 🧪 A second, sharper way to see "most volume is in the corners"

The ball-in-a-cube picture is vivid but slightly indirect. Here is a cleaner statement of the same
fact, and one that is easier to state in an interview.

**Take a cube of side 1. What fraction of its volume lies within $\varepsilon$ of the surface?**

The "interior" — the set of points more than $\varepsilon$ from every face — is itself a cube, of side
$1 - 2\varepsilon$. So:

$$\Pr[\text{point is in the interior}] = (1 - 2\varepsilon)^d$$

With $\varepsilon = 0.05$ (within 5% of an edge counts as "near the surface"):

| $d$ | $(0.9)^d$ = fraction in the interior | So the fraction near the surface is |
|---|---|---|
| 1 | 0.900 | 10% |
| 10 | 0.349 | 65% |
| 50 | 0.0052 | **99.5%** |
| 100 | $2.65\times10^{-5}$ | **99.997%** |

**At $d = 100$, essentially every point in the cube is within 5% of the boundary.** There is no
"middle" of a high-dimensional cube in any meaningful sense — which is precisely the slide's third
bullet, *"uniform sampling misses the center"*, made quantitative.

> 💡 **And notice the number: $0.9^{100} = 2.65\times10^{-5}$.** That is the *same arithmetic* as
> [Deep Neural Networks Part 3 §5](../Deep%20Neural%20Networks/deep-neural-networks-03.md), where
> $0.9^{100}$ was the death of an RNN gradient. Different subject, identical mechanism: **a factor
> slightly below 1, raised to a large power.** Once you start noticing this pattern you find it
> everywhere in ML, and it's a genuinely good thing to say out loud.

### Where people get confused

- **"So high-dimensional balls are small."** Careful — *relative to their cube* they are vanishingly
  small, and the unit ball's absolute volume does go to zero too. But the ball still contains every
  point within distance 1 of the origin, which is a lot of points. The useful statement is the
  **ratio**, not the absolute volume.
- **"This is just a curiosity about spheres."** It isn't. Any method that reasons about a
  *neighbourhood* — kNN, kernel density estimation, RBF kernels, DBSCAN, local outlier factor — is
  implicitly putting a ball around a point and asking what's inside. In high dimensions, that ball
  captures essentially nothing, so those methods have nothing to work with. §4 spells this out.
- **"Then just use a cube-shaped neighbourhood instead."** A cube whose *volume* is a fixed fraction
  of the space has side length $\alpha^{1/d}$, which $\to 1$ as $d$ grows. To capture 1% of the space
  in 100 dimensions you need a cube of side $0.01^{1/100} = 0.955$ — i.e. spanning **95% of the range
  of every feature**. It isn't a "local" neighbourhood any more; it's nearly the whole dataset.

```interactive
type: slider
title: Ball inside a cube, as dimension grows
concept: Volume concentration
control: A slider for dimension d (2 to 50), and a second slider for the "near the surface" band width epsilon
observe: The ball-in-cube ratio plotted on a log axis with the current d marked, alongside a second readout of what fraction of the cube lies within epsilon of its surface
insight: Watching 78% collapse to 0.25% between d=2 and d=10 makes it visceral that this happens at dimensions you actually use — not at some exotic d=10,000
fallback: The two tables above: the ball fills 78.5% / 52.4% / 0.249% / 0.0000025% at d = 2 / 3 / 10 / 20; and at d=100, 99.997% of a cube is within 5% of its surface
```

---

## 2. Distance concentration

**This is the most important section in Act I**, because it is the one that directly breaks the most
algorithms — and because it has a clean proof you can reproduce.

> *"All pairwise distances become nearly equal"* [slide 17, 12:32]

$$\frac{d_{\max} - d_{\min}}{d_{\min}} \to 0 \quad \text{as } d \to \infty$$

| Symbol | Read it as | What it means |
|---|---|---|
| $d_{\max}$ | "d max" | The distance from a query point to the **farthest** point in your dataset. |
| $d_{\min}$ | "d min" | The distance to the **nearest** point. |
| The ratio | "relative contrast" | How much closer the nearest neighbour is than the farthest, **as a fraction**. If this is 0.5, the nearest is 33% closer than the farthest. If it's 0.01, they're basically the same distance. |

The slide's four consequences:

> - *"**Nearest** and **farthest** neighbors become indistinguishable"*
> - *"kNN decisions become arbitrary in high-D"*
> - *"Distance-based kernels lose discriminative power"*
> - *"Affects clustering (k-means) and anomaly detection"*

And a genuinely actionable box:

> *"Practical test: compute $\frac{d_{\max}-d_{\min}}{d_{\min}}$ on your dataset as you add features.
> **If it drops below 0.1, distances are unreliable.**"*

### 2.1 🧪 Why it happens — the derivation

This is worth doing properly, because the mechanism is more surprising than the conclusion.

**Setup.** Let $X$ and $Y$ be two independent points drawn uniformly from the cube $[0,1]^d$. Their
squared distance is

$$D^2 = \sum_{j=1}^{d} (X_j - Y_j)^2$$

— a **sum of $d$ independent identically distributed terms**. That's the key structural fact, and
everything follows from it.

**Step 1 — the mean of one term.** For $X_j, Y_j$ independent uniform on $[0,1]$:

$$\mathbb{E}\left[(X_j - Y_j)^2\right] = \mathbb{E}[X_j^2] - 2\mathbb{E}[X_j]\mathbb{E}[Y_j] + \mathbb{E}[Y_j^2] = \tfrac13 - 2\left(\tfrac12\right)\left(\tfrac12\right) + \tfrac13 = \tfrac23 - \tfrac12 = \boxed{\tfrac16}$$

**Step 2 — the variance of one term.** Let $Z = X_j - Y_j$, which has a *triangular* density
$f(z) = 1 - |z|$ on $[-1, 1]$. Then

$$\mathbb{E}[Z^4] = 2\int_0^1 z^4 (1-z)\,dz = 2\left[\frac{z^5}{5} - \frac{z^6}{6}\right]_0^1 = 2\left(\tfrac15 - \tfrac16\right) = \tfrac{1}{15}$$

$$\mathrm{Var}(Z^2) = \mathbb{E}[Z^4] - \left(\mathbb{E}[Z^2]\right)^2 = \tfrac{1}{15} - \tfrac{1}{36} = \tfrac{12 - 5}{180} = \boxed{\tfrac{7}{180}}$$

**Step 3 — sum over $d$ independent dimensions.** Means add, and (by independence) variances add:

$$\mathbb{E}[D^2] = \frac{d}{6}, \qquad \mathrm{Var}(D^2) = \frac{7d}{180}$$

**Step 4 — convert to the distance itself.** $D = \sqrt{D^2}$. Since $D^2$ is tightly concentrated
around $d/6$, the delta method gives $\mathrm{std}(D) \approx \dfrac{\mathrm{std}(D^2)}{2\,\mathbb{E}[D]}$:

$$\mathbb{E}[D] \approx \sqrt{\tfrac{d}{6}} = 0.4082\sqrt{d}$$

$$\mathrm{std}(D) \approx \frac{\sqrt{7d/180}}{2\sqrt{d/6}} = \frac{0.1972\sqrt{d}}{2 \times 0.4082\sqrt{d}} = \boxed{0.2415}$$

**Look at what just happened.** The $\sqrt{d}$ cancelled.

$$\textbf{The mean distance grows like } \sqrt{d}. \quad \textbf{The spread of distances stays CONSTANT.}$$

That is the entire phenomenon, and it is much more striking than "distances become similar." Distances
don't become similar in absolute terms — they become similar *relative to how large they are*:

$$\frac{\mathrm{std}(D)}{\mathbb{E}[D]} \approx \frac{0.2415}{0.4082\sqrt{d}} = \frac{0.592}{\sqrt{d}}$$

### 🧪 Worked example — put numbers on it

| $d$ | $\mathbb{E}[D]$ | $\mathrm{std}(D)$ | Relative spread $0.592/\sqrt{d}$ |
|---|---|---|---|
| 2 | 0.577 | 0.242 | **0.419** |
| 10 | 1.291 | 0.242 | **0.187** |
| 50 | 2.887 | 0.242 | 0.0837 |
| 100 | 4.082 | 0.242 | **0.0592** |
| 500 | 9.129 | 0.242 | **0.0265** |
| 1000 | 12.910 | 0.242 | 0.0187 |

**Read the middle column.** It never changes. Every pairwise distance in a 1000-dimensional uniform
cloud is $12.91 \pm 0.24$ — a band less than 4% wide. Ask "which point is nearest?" and you are asking
your algorithm to distinguish 12.7 from 13.1, on data that is noisy anyway.

**Now apply the slide's practical test.** The relative contrast
$\frac{d_{\max}-d_{\min}}{d_{\min}}$ is a *range* rather than a standard deviation, so it's a small
multiple of the relative spread above — the multiple depends on $n$, since more points means a wider
observed range. But the **scaling is the same $1/\sqrt{d}$**, so the threshold behaves predictably:
the relative spread crosses the slide's 0.1 line at around $d \approx 35$, and the contrast itself
somewhat later.

⚠️ Two honest caveats on that last sentence. First, the exact crossing point depends on $n$ and on the
data distribution — the derivation above assumed uniform, independent coordinates, which real data
never is. Second, and much more importantly: **correlated features don't count as separate
dimensions.** A dataset with 500 columns that really lives on a 6-dimensional manifold behaves like
$d = 6$, not $d = 500$. That is exactly why the deck's practical test says to *measure* the ratio on
your data rather than compute it from $p$ — and it is the bridge to §7's manifold hypothesis.

```python
import numpy as np
from scipy.spatial.distance import pdist

for d in (2, 10, 100, 500):
    X = np.random.rand(1000, d)
    D = pdist(X)
    print(f"d={d:4d}  mean={D.mean():.3f}  std={D.std():.3f}  "
          f"contrast={(D.max()-D.min())/D.min():.3f}")
# the mean tracks 0.408*sqrt(d); the std stays near 0.242 regardless of d
```

### 2.2 Why each of the four consequences follows

| The slide's claim | The mechanism |
|---|---|
| **Nearest and farthest become indistinguishable** | Direct restatement of the ratio $\to 0$. |
| **kNN decisions become arbitrary** | kNN's entire premise is that the $k$ closest points are *meaningfully* closer. When every distance is $12.91 \pm 0.24$, which $k$ you get is decided by measurement noise, not by similarity. Two runs with slightly different data give different neighbours and different predictions. |
| **Distance-based kernels lose discriminative power** | An RBF kernel is $\exp(-\gamma\|x - x'\|^2)$. If every $\|x-x'\|^2$ is nearly the same value $c$, then every kernel entry is nearly $e^{-\gamma c}$ — the kernel matrix becomes nearly constant, and a constant kernel carries no information about which points are similar. |
| **Affects clustering and anomaly detection** | k-means assigns each point to its nearest centroid — same failure as kNN. Anomaly detection typically flags points whose distance to their neighbours is unusually large; when all distances are nearly equal, nothing is unusual and nothing gets flagged. |

> 💡 **The unifying statement, and the one to say in an interview:** *every* one of these methods is
> built on the assumption that **the distance function is informative** — that "closer" means "more
> similar". Distance concentration doesn't make the distances wrong; it makes them **uninformative**,
> because they no longer vary enough to distinguish anything. And crucially it fails *silently*: your
> kNN still returns 5 neighbours, your k-means still returns clusters, and nothing errors.

### 🔬 Research opportunity

Aggarwal, Hinneburg & Keim (2001) — the deck's citation on this slide — showed that the *choice of
norm* matters: fractional norms $L_p$ with $p < 1$ concentrate more slowly than Euclidean ($L_2$), and
even the Manhattan norm ($L_1$) is measurably better than Euclidean in high dimensions. That is
actionable and underused: **if you must do nearest-neighbour work in high dimensions, try $L_1$ before
you try anything clever.** Whether fractional norms are worth their weirdness (they violate the
triangle inequality, which breaks indexing structures) is still an open practical question.

```interactive
type: graph
title: Distances concentrate as dimension grows
concept: Mean grows as sqrt(d) while spread stays constant
control: A slider for dimension d (2 to 1000) and one for sample count n
observe: A histogram of all pairwise distances, redrawn as d changes, with the mean marked and the band [mean - std, mean + std] shaded; a live readout of relative contrast
insight: Watching the histogram slide rightwards while never getting wider is a far stronger demonstration than the formula — the constancy of the width is the surprising part
fallback: The table above — mean distance goes 0.577 → 12.910 from d=2 to d=1000 while the standard deviation stays at 0.242 throughout, so relative spread falls as 0.592/sqrt(d)
```

---

## 3. Empty space

The third angle on the same problem, and the one that connects geometry to *sample size*.

> *"Exponential data requirements to fill high-D space"* [slide 20, 15:02]
>
> - *"To sample at density of **k bins per axis**, need $n \sim \mathcal{O}(k^d)$ points"*
> - *"Space becomes overwhelmingly empty with fixed n"*

| Dimensions ($d$) | Bins per axis ($k{=}10$) | Points needed |
|---|---|---|
| 2 | 10 | 100 |
| 5 | 10 | 100,000 |
| 10 | 10 | 10 billion |
| 20 | 10 | $10^{20}$ |

> *"With 1,000 samples and 20 features, your data occupies a vanishingly small fraction of the space."*

**The arithmetic is trivial and that's the point:** chop each of $d$ axes into $k$ pieces and you have
$k^d$ cells. One point per cell needs $k^d$ points. There is nothing subtle here — it's just that
exponentials are bigger than people's intuition allows.

### 🧪 Worked example — how empty is "vanishingly small"?

Take the slide's own scenario: $n = 1000$ samples, $d = 20$ features, $k = 10$ bins per axis.

$$\text{cells} = 10^{20} \qquad \text{occupied cells} \le 1000 = 10^{3}$$

$$\text{fraction occupied} \le \frac{10^3}{10^{20}} = \mathbf{10^{-17}}$$

**Make that concrete.** $10^{-17}$ is roughly the ratio of **one grain of sand to the entire Sahara
desert** — and that comparison is generous, because it assumes every sample lands in a different cell.

Turn it around and ask the more useful question: **how many features can 1,000 samples support?**
Requiring at least one point per cell:

$$10^d \le 1000 \implies d \le 3$$

**Three.** With $n = 1000$ and a 10-bin resolution, you can densely cover *three* features. This is
why purely local, nonparametric methods (kNN, kernel density estimation, histograms) are essentially
low-dimensional tools, and why everything that works in high dimensions makes a **structural
assumption** — linearity, smoothness, sparsity, a manifold — to avoid needing dense coverage.

> 💡 **This is the deepest sentence in Act I:** *there is no such thing as a truly assumption-free
> method in high dimensions.* Methods that appear assumption-free (kNN: "just look at nearby points")
> are really making the assumption that dense local coverage is available — and §3 shows it is not. A
> method that works in 500 dimensions is a method that has smuggled in a structural assumption
> somewhere. Being able to name that assumption is a genuinely high-level skill.

---

## 4. Practical consequences

The section's summary slide [slide 23, 16:42] — *"What breaks in high dimensions"*:

| The slide's claim | Which section explains it | The mechanism in one line |
|---|---|---|
| **kNN fails:** *all neighbors equidistant, predictions become random* | §2 | Distance concentration makes "nearest" meaningless. |
| **Kernel methods degrade:** *RBF bandwidth cannot cover sparse space* | §1, §3 | Any bandwidth small enough to be local encloses no points; any bandwidth large enough to enclose points is not local. |
| **Overfitting:** *with $p \gg n$, any model can perfectly separate noise* | see below | With more free parameters than constraints, a perfect fit always exists. |
| **Visualization impossible:** *cannot plot 100D data directly* | — | You have two, maybe three axes. This is the motivation for §6's first goal. |
| **Computational cost:** *many algorithms scale $\mathcal{O}(d^2)$ or worse* | Prereq 3 | Covariance is $\mathcal{O}(nd^2)$; eigendecomposition is $\mathcal{O}(d^3)$; a full pairwise distance matrix is $\mathcal{O}(n^2 d)$. |

### 4.1 📚 Background the slide assumed — why $p \gg n$ guarantees overfitting

The third bullet deserves unpacking, because it is a *theorem*, not a tendency.

**In $d$ dimensions, any $d+1$ points in general position can be shattered by a hyperplane** — that
is, for any assignment of $\pm$ labels to them, some hyperplane separates the positives from the
negatives. So if $p \ge n$, a linear model can fit **any** labelling of your training set, including
one you generated by flipping coins.

*Concretely:* take $n = 100$ samples and $p = 500$ features of pure Gaussian noise, and assign random
binary labels. A linear SVM will achieve **100% training accuracy**. There is no signal — you
generated the labels yourself with a coin — and yet the model separates them perfectly.

**The consequence for practice:** in the $p \gg n$ regime, *training accuracy carries no information
at all*. Not "less information" — none. Your only instrument is held-out validation, and with small
$n$ that estimate is itself noisy, so cross-validation stops being a nicety and becomes the only thing
standing between you and a fictional result.

This is the formal underpinning of [Deep Neural Networks Part 2 §9](../Deep%20Neural%20Networks/deep-neural-networks-02.md)'s
overfitting chapter, seen from the dimension side rather than the capacity side. It is also why the
whole of §20–§26 (embedded methods) exists: regularization is how you make $p \gg n$ survivable.

---

## 5. 🩹 Quick check — kNN on 1,000 × 500

> **You have 1,000 samples and 500 features. You train a kNN classifier ($k$=5). What happens?**
> [slide 24, 16:45]
>
> A) Works well because kNN is nonparametric and flexible
> B) Fails because distances concentrate and all neighbors are equidistant
> C) Works if you increase $k$ to 50

> ⚠️ **The answer reveal was not captured** (see the capture note). The answer below follows
> unambiguously from the deck's own §2 slide, but it is not transcribed from an answer line.

<details>
<summary><b>Answer</b></summary>

**B.**

**Verify it with §2's arithmetic.** At $d = 500$, the relative spread of pairwise distances is

$$\frac{0.592}{\sqrt{500}} = \frac{0.592}{22.36} = \mathbf{0.026}$$

The deck's own practical test says distances are unreliable below **0.1**. We are at 0.026 — a factor
of four past the line. The 5 "nearest" neighbours are not meaningfully nearer than the 500th.

**Why each wrong answer is wrong — this is the part that matters:**

- **A) "kNN is nonparametric and flexible."** Every word of that is true, and it is exactly why kNN
  fails here. Nonparametric means *the model makes no structural assumption and relies entirely on
  local density instead* — and §3 showed that with $n = 1000$, dense local coverage exists for about
  three features, not 500. **Flexibility is not free; it is paid for in samples, and the price is
  exponential in $d$.** This option is a trap for people who have learned that nonparametric methods
  are powerful without learning what they're powerful *at*.

- **C) "Works if you increase $k$ to 50."** Tempting, because averaging more neighbours does reduce
  variance. But it does not touch the problem: the issue is not that 5 neighbours is a noisy sample,
  it is that **the neighbours are not neighbours**. Increasing $k$ from 5 to 50 just averages 50
  effectively-random points instead of 5. You have reduced the variance of an estimate of the wrong
  quantity — and in the limit $k \to n$ you predict the global majority class for every input, which
  is at least stable but has learned nothing.

**What would actually work,** and worth adding as the follow-up: reduce the dimension first (PCA to
~10–20 components, or feature selection), then run kNN in the reduced space. Or check whether the data
has a low-dimensional manifold structure that makes the *effective* dimension far below 500 — measure
the relative contrast on your actual data rather than assuming $d = 500$ is the operative number
(§2.1's caveat).
</details>

---

# ACT II, part one — Why reduce dimensions

*21:28 – 31:40*

---

## 6. Three goals of dimensionality reduction

The deck gives three goals, each with two real-world examples [slide 34, 23:28]. Learn the three-way
split — it is the cleanest possible answer to "why would you use PCA?"

| Goal | The slide's subtitle | The slide's examples |
|---|---|---|
| **📊 Visualization** | *Project to 2D/3D for human insight* | *"A genomics team plots 20,000-gene tumor profiles in 2D to spot patient subgroups."* · *"A fraud team eyeballs a UMAP plot to find clusters of suspicious transactions."* |
| **📦 Compression** | *Fewer features, faster training and inference* | *"A streaming service stores 50-dim user embeddings instead of millions of raw click columns."* · *"An IoT pipeline shrinks 500 vibration sensors to 15 features so models fit on edge devices."* |
| **🎯 Noise Reduction** | *Signal lives in a low-rank subspace* | *"An MRI lab drops low-variance components to clear scanner noise before diagnosis."* · *"A quant team strips noisy directions from returns to get a stable covariance estimate."* |

### 6.1 Why these three and not others

They correspond to three genuinely different reasons the extra dimensions are a problem:

1. **Visualization** — the constraint is *human*. Your eyes take 2 or 3 dimensions. This goal doesn't
   care about model accuracy at all; a projection that distorts distances badly can still be the right
   choice if it makes the cluster structure visible. **This is why t-SNE and UMAP are legitimate
   despite being unusable as preprocessing** — they optimise for a picture, not for a downstream model.
2. **Compression** — the constraint is *resources*: memory, latency, bandwidth. Here the criterion is
   reconstruction quality per byte, and PCA is close to optimal by construction (§31).
3. **Noise reduction** — the constraint is *statistical*. Signal concentrates in the high-variance
   directions and noise spreads evenly across all of them, so discarding low-variance directions
   discards proportionally more noise than signal.

> 💡 **The third goal is the counter-intuitive one, and the best one to raise in an interview.**
> Dimensionality reduction is usually pitched as a *sacrifice*: you throw information away to gain
> speed. Noise reduction is the case where **throwing information away makes the model better**,
> because what you threw away was mostly noise. The quant example is the sharpest version: a
> covariance matrix estimated from $n$ samples in $p$ dimensions has $p(p+1)/2$ parameters, so with
> $p = 500$ you are estimating 125,250 numbers, and with $n < p$ the estimate is **singular** —
> mathematically unusable for portfolio optimisation. Truncating to the top few components isn't an
> approximation to a good answer; it's the only way to get an answer at all.

---

## 7. The manifold hypothesis

**This is the intellectual keystone of the entire module.** Everything after it is a technique; this
is the assumption that makes techniques possible.

> *"Real data lies near a low-dimensional manifold in high-D space"* [slide 37, 25:52]
>
> - *"Images of faces: ~100K pixels, but face space is ~50D"*
> - *"Natural language: vocabulary is huge, but topics are few"*
> - *"DR finds that low-D structure"*
>
> *"If the manifold hypothesis holds, DR can reduce dimensions without losing signal."*

The slide's diagram shows a curved 2D sheet threading through 3D space with data points sitting on it,
captioned *"2D manifold curving through 3D space"*.

> **Manifold** — a surface that is *locally* flat but may be *globally* curved, sitting inside a
> higher-dimensional space.
>
> *In everyday words:* the surface of the Earth. It lives in 3D, but you only need **two** numbers
> (latitude and longitude) to say where you are on it. Locally it looks like a flat map; globally it
> curves around into a sphere.
>
> *Concretely:* take a sheet of paper — a 2D object — and roll it into a tube in 3D. Every point on it
> still needs only two coordinates (how far along, how far around), even though it's now described by
> three numbers $(x, y, z)$. The **intrinsic** dimension is 2; the **ambient** dimension is 3.
>
> *Why it exists as a concept:* it separates *"how many numbers are used to represent the data"* from
> *"how many degrees of freedom the data actually has."* Dimensionality reduction is the business of
> replacing the first with the second.

### 7.1 Why the hypothesis is plausible — the generative argument

Take the face example and reason about where the data comes from.

A 100×100 greyscale photo is a point in $\mathbb{R}^{10{,}000}$. So there are $256^{10{,}000}$ possible
such images. **Essentially all of them are noise** — pick pixel values at random ten thousand times and
you will never once produce something that looks like a face.

The images that *are* faces were produced by a physical process with a limited number of knobs:
identity (bone structure, colouring), pose (3 rotations + 3 translations), expression (a few dozen
muscles), lighting (direction, intensity, colour), camera (focal length, distance). That's maybe a
hundred numbers, not ten thousand. **The set of face images is the image of a ~50–100-dimensional
parameter space under a smooth map** — which is the definition of a manifold sitting inside
$\mathbb{R}^{10{,}000}$.

> 💡 **The general principle, and the one to remember:** *data generated by a process with few degrees
> of freedom lies on a manifold whose dimension is the number of those degrees of freedom, no matter
> how many numbers you use to record it.* The recording format inflates the ambient dimension; it
> cannot inflate the intrinsic one.

Apply it to the deck's other examples and to your own work:

| Data | Ambient dimension | Plausible intrinsic dimension | The generating factors |
|---|---|---|---|
| Face photos | ~100,000 pixels | ~50 | identity, pose, lighting, expression |
| Documents (bag of words) | ~50,000 vocabulary | ~100 | topics |
| Vibration sensors on one machine | 500 channels | ~15 | a few physical failure modes |
| Customer purchase vectors | ~1M products | ~50 | taste, budget, life stage, season |

### 7.2 ⚠️ When the hypothesis fails — and why you must say this part

The manifold hypothesis is an **assumption about your data**, not a theorem. The deck words it
carefully — *"**If** the manifold hypothesis holds"* — and that conditional is the honest part.

**Where it genuinely does not hold:**

- **Truly independent features.** Twenty independent sensor readings measuring twenty unrelated
  physical quantities have intrinsic dimension 20. There is no manifold to find, and any reduction
  loses real signal. Tabular data assembled from unrelated sources is often like this — and it is one
  reason PCA disappoints on tabular problems where it dazzles on images.
- **The signal is in a low-variance direction.** PCA keeps high-variance directions on the assumption
  that variance = signal. If your target depends on a small-amplitude feature — a subtle sensor drift
  that predicts failure — PCA will discard it as noise. ⚠️ **This is PCA's single most dangerous
  failure mode, it is silent, and it is the best objection to raise when someone proposes PCA
  reflexively.** Note that PCA is *unsupervised*: it never looks at $y$, so it cannot know which
  directions matter for your task. (LDA, in the taxonomy at §10, is the supervised alternative that
  can.)
- **Non-smooth or disconnected structure.** Categorical variables and discrete jumps don't form
  smooth manifolds, and methods assuming smoothness handle them badly.

**How to test it rather than assume it.** Three practical checks, in increasing cost:

1. Run PCA and look at the **scree plot** (§30). A sharp elbow is direct evidence of low intrinsic
   dimension; a slowly-decaying spectrum says there isn't one.
2. Compute the deck's own **relative contrast** test from §2. If distances haven't concentrated
   despite a large $p$, the effective dimension is already low.
3. Estimate the **intrinsic dimension** directly — e.g. the correlation dimension or a
   maximum-likelihood estimator (Levina & Bickel, 2004). ⚠️ These estimators are themselves unreliable
   at small $n$; treat the number as an order of magnitude, not a measurement.

### 7.3 Where the citation points

The slide cites **Tenenbaum, de Silva & Langford (2000), "A Global Geometric Framework for Nonlinear
Dimensionality Reduction"** — the **Isomap** paper. Its contribution is directly relevant here: on a
curved manifold, *Euclidean* distance through the ambient space is the wrong measure (two points on
opposite sides of a rolled-up sheet are close in $\mathbb{R}^3$ but far apart along the sheet). Isomap
measures **geodesic** distance *along* the manifold instead, by building a neighbourhood graph and
taking shortest paths.

That distinction — ambient versus intrinsic distance — is the whole reason nonlinear methods exist,
and it is why the taxonomy in §10 splits extraction into *linear* and *nonlinear*.

---

## 8. Practical benefits

The section's closing slide [slide 39, 26:51] — *"Why DR matters in production ML"*:

| The slide's claim | Why |
|---|---|
| **Storage**: *1000 features to 50 means 20× less memory per sample* | $1000/50 = 20$. At 4 bytes per float and 100M users, that's 400 GB → 20 GB. |
| **Speed**: *training and inference scale with feature count* | Linear models are $\mathcal{O}(nd)$ per epoch; kernel methods and covariance work are $\mathcal{O}(d^2)$ or $\mathcal{O}(d^3)$. |
| **Generalization**: *fewer parameters, less overfitting* | §4.1 — fewer dimensions means the $p \gg n$ regime is further away. |
| **Interpretability**: *easier to understand 5 components than 500 raw features* | ⚠️ True only for *selection*. See the caveat below. |
| **Denoising**: *dropping low-variance directions removes noise* | §6, goal 3. |

> ⚠️ **The interpretability bullet is the one to push back on, and doing so is a good interview
> move.** For **feature selection** it is straightforwardly true: five named columns
> (`days_since_last_order`, `avg_basket_value`, …) are obviously easier to explain than five hundred.
> For **feature extraction** it is much weaker: PC1 is a linear combination of all 500 original
> features with 500 nonzero loadings. It is *lower-dimensional*, but "0.03 × feature_1 + 0.11 ×
> feature_2 + …" is not something you can put in front of a business stakeholder or a regulator.
>
> The §13 comparison table is honest about exactly this, listing extraction's interpretability as
> *"Lower (abstract components)"*. The two slides sit four minutes apart and mildly contradict each
> other; the §13 version is the correct one.

> 💡 **The benefit the slide omits, and it is often the biggest one: fewer features means fewer things
> to break in production.** Every feature is a pipeline that can fail, a schema that can drift, an
> upstream team that can change a definition without telling you. Going from 500 features to 50 cuts
> your operational surface area by 90%, and in a real system that is frequently worth more than the
> latency win.

---

# ACT II, part two — Selection vs Extraction

*26:54 – 31:40*

---

## 9. Two philosophies

> *"Two philosophies for reducing dimensionality"* [slide 42, 28:55]

| | **Feature Selection** | **Feature Extraction** |
|---|---|---|
| **Output** | Subset of original features | New features (combinations) |
| **Interpretability** | High (original meaning preserved) | Lower (abstract components) |
| **Search space** | Combinatorial ($2^p$ subsets) | Continuous optimization |
| **Examples** | Lasso, mutual information filter | PCA, autoencoders |
| **Best when** | Interpretability required, sparse true signal | Features correlated, manifold structure |

### 9.1 Reading the table properly

**Row 1 — Output.** This is the definition, and everything else follows from it.

- **Selection** returns a subset: you keep `age`, `income` and `n_orders`, and drop the other 497.
  Every surviving column still means what it always meant.
- **Extraction** returns combinations: you keep
  $\text{PC}_1 = 0.31\,\text{age} + 0.02\,\text{income} - 0.14\,\text{n\_orders} + \ldots$
  — a number with no name and no units.

**Row 3 — Search space is the most technically interesting row.** Selection is a *discrete* problem:
each feature is in or out, so there are $2^p$ candidate subsets and the objective is not
differentiable in any useful sense. Extraction is *continuous*: PCA's answer is the solution of an
eigenvalue problem with a closed form, and an autoencoder's is found by gradient descent.

**This is why the module's structure is what it is.** Discrete search over $2^p$ options is
intractable (Prereq 3), so §12–§26 are three different ways of *avoiding* it: wrappers search greedily,
filters skip search entirely by scoring features independently, and embedded methods **relax the
discrete problem into a continuous one** — which is exactly what the L1 penalty does, and it is the
single most elegant idea in this lecture.

**Row 5 — "Best when" is the row you'll actually use.**

- *"sparse true signal"* → **selection.** If only 10 of your 500 features genuinely matter and the rest
  are noise, you want the 10 named ones. A PCA component that mixes all 500 will drag the 490 noise
  features into every prediction.
- *"features correlated, manifold structure"* → **extraction.** If 100 features are all noisy
  measurements of 3 underlying quantities, no *subset* of them is as good as the right 3
  *combinations* — averaging correlated measurements cancels noise, and selection cannot average.

> 💡 **The one-sentence decision rule:** *if the true signal is a few of your columns, select; if it's
> a few directions through your columns, extract.* And a practical corollary that the deck doesn't
> state: **you can do both**, in that order. Filter out obviously dead features first (§16 costs
> nothing), then run PCA on what's left. Feeding 500 columns of which 300 are constant into PCA wastes
> the components on garbage.

### 9.2 ⚠️ The regulatory dimension the deck doesn't mention

For some applications, interpretability isn't a preference — it's a legal requirement. Credit
decisions, insurance pricing and hiring are subject to adverse-action and explainability
requirements in many jurisdictions, and *"your application was declined because principal component 3
was too low"* is not an explanation anyone will accept.

**In those settings feature extraction is effectively off the table regardless of accuracy**, and the
choice in §9's table is made for you before you look at any data. Knowing this — and asking about it
early — is a mark of someone who has shipped models rather than only trained them.

---

## 10. The taxonomy of DR methods

The map for the whole module [slide 45, 30:53]:

```
                        ┌────────────────────────┐
                        │ Dimensionality         │
                        │ Reduction              │
                        └───────────┬────────────┘
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
        ┌───────────────────────┐        ┌───────────────────────┐
        │  Feature Selection    │        │  Feature Extraction   │
        │  (keep a subset)      │        │  (build combinations) │
        └───────────┬───────────┘        └───────────┬───────────┘
          ┌─────────┼─────────┐              ┌───────┴────────┐
          ▼         ▼         ▼              ▼                ▼
      ┌───────┐ ┌───────┐ ┌────────┐    ┌────────┐    ┌─────────────┐
      │Filter │ │Wrapper│ │Embedded│    │ Linear │    │  Nonlinear  │
      └───┬───┘ └───┬───┘ └───┬────┘    └───┬────┘    └──────┬──────┘
          │         │         │             │                │
     MI, Chi²,  Forward,  Lasso,        PCA, LDA       t-SNE, UMAP,
     Variance   Backward, Ridge,                       Autoencoders
                RFE       Trees
          │         │         │             │                │
        §15–19    §12–14    §20–26     ◄─── Part 2 & 3 of this module ───►
```

**Everything named in that tree is covered somewhere in this module**, and the leaves tell you where:

| Leaf | Covered in |
|---|---|
| Filter: MI, χ², Variance | **§15–§19** (this file) |
| Wrapper: Forward, Backward, RFE | **§12–§14** (this file) |
| Embedded: Lasso, Ridge, Trees | **§20–§26** (this file) |
| Linear: PCA, LDA | Part 2 |
| Nonlinear: t-SNE, UMAP, Autoencoders | Part 3 (⚠️ expected — verify against the actual deck) |

> 💡 **The single distinction that organises the three selection families, and the thing to actually
> remember: *how much does the method know about the model you will eventually train?***
>
> | Family | Knows about the model | Cost | Sees feature interactions? |
> |---|---|---|---|
> | **Filter** | **Nothing.** Scores features against $y$ alone. | $\mathcal{O}(p)$, no training | ❌ No — scores are univariate |
> | **Wrapper** | **Its score.** Treats the model as a black box and asks "is this subset better?" | $\mathcal{O}(p \cdot d)$ **model fits** | ✅ Fully |
> | **Embedded** | **Everything — it *is* the model.** Selection happens inside training. | ~1 model fit | ✅ Within the model's own hypothesis class |
>
> Read down the "knows" column and the cost column together: **you pay for knowledge in compute.**
> That's the entire trade-off, and it is why §20's embedded methods — which get most of the knowledge
> for one model fit — are the practical default.

---

## 11. 🩹 Quick check — zero correlation, high mutual information

> **A feature has zero Pearson correlation with the target, but high mutual information. What does
> this mean?** [slide 46, 30:56]
>
> A) The feature is useless for prediction
> B) The relationship between feature and target is nonlinear
> C) The mutual information estimate is unreliable

> ⚠️ **The answer reveal was not captured.** The answer below is fully determined by the deck's own
> §17 and §18 slides ("Pearson: detects linear relationships only"; "MI: captures any statistical
> dependence"), but is not transcribed from an answer line.

<details>
<summary><b>Answer</b></summary>

**B.**

The two statistics measure different things, and the deck says so on its own slides:

- **Pearson $r$** — *"Detects linear relationships only."* $r = 0$ means *no **linear** trend*. It says
  nothing whatsoever about curves.
- **Mutual information** — *"Captures any statistical dependence."* And critically,
  $I(X;Y) = 0 \iff X \perp Y$: MI is zero **if and only if** the variables are genuinely independent.

So $r = 0$ with $I > 0$ is not a contradiction — it is the **signature of a nonlinear relationship**,
and it's the exact case MI exists to detect.

**Demonstrate it with three points and no hand-waving** (this is §17.2's example, and having it ready
turns a recited answer into a demonstrated one). Let $X \in \{-2,-1,0,1,2\}$ uniformly, and
$Y = X^2$:

| $X$ | $-2$ | $-1$ | $0$ | $1$ | $2$ |
|---|---|---|---|---|---|
| $Y = X^2$ | 4 | 1 | 0 | 1 | 4 |

$\bar{X} = 0$, $\bar{Y} = 2$, so

$$\mathrm{Cov}(X,Y) = \tfrac15\left[(-2)(2) + (-1)(-1) + (0)(-2) + (1)(-1) + (2)(2)\right] = \tfrac15\left[-4+1+0-1+4\right] = \mathbf{0}$$

**$r = 0$ exactly** — and yet $Y$ is a *deterministic function* of $X$. Knowing $X$ tells you $Y$ with
certainty. §18's worked example computes the mutual information here as **1.522 bits**.

**Why the wrong answers are wrong:**

- **A) "useless for prediction"** — the exact opposite. $Y$ is perfectly determined by $X$; a model
  that can represent curvature (a tree, a spline, a neural net, or linear regression on $X^2$) will
  predict it perfectly. **Discarding this feature on the basis of $r = 0$ throws away a perfect
  predictor**, and this is precisely why the deck warns that Pearson "misses nonlinear dependencies
  (parabola, XOR)".
- **C) "the MI estimate is unreliable"** — a reasonable instinct, since MI estimation *is* noisy on
  small samples (the deck flags exactly this: *"Noisier than Pearson for small samples"*). But it's
  the wrong diagnosis for this pattern. Unreliable MI estimates are **biased upward on noise** —
  they'd give you high MI on features with *no* relationship at all. Here we have a specific,
  reproducible structure. The right response to option C is: *"that's worth ruling out — permute the
  target and recompute; if MI stays high on shuffled labels, it's estimator bias. If it collapses,
  the dependence is real."* Saying that turns a wrong option into evidence you know how to check
  yourself.

> 🎯 **The interview version of this question is "when would you not use correlation for feature
> selection?"** — and the strongest answer names the failure mode (nonlinear and non-monotonic
> relationships: parabolas, XOR, U-shapes), gives the concrete parabola example above, and then adds
> the practical middle ground: **Spearman rank correlation** catches any *monotonic* relationship at
> almost the same cost as Pearson, so it's the cheap first upgrade before you reach for MI.
</details>

---

# ACT II, part three — Wrapper methods

*31:40 – 35:30*

---

## 12. Wrapper methods: the core idea

> *"Use model performance to evaluate feature subsets"* [slide 48, 31:45]

```
     ┌──────────────┐    ┌─────────────┐    ┌───────────────┐    ┌──────────┐
     │ Select Subset│───▶│ Train Model │───▶│ Evaluate (CV) │───▶│  Update  │
     └──────▲───────┘    └─────────────┘    └───────────────┘    └────┬─────┘
            │                                                          │
            └──────────────────────────────────────────────────────────┘
                                 try a different subset
```

> - *"**Scoring**: cross-validated accuracy, AUC, or any metric"*
> - *"**Search strategy**: forward, backward, genetic, random"*
> - *"**Expensive**: each candidate subset requires full model training"*

**The defining property: the model is a black box.** A wrapper never looks inside — it only asks *"how
well does it score with these columns?"* That has one enormous advantage and one fatal cost:

- ✅ **It optimises exactly what you care about.** If your metric is AUC on your model, the wrapper
  selects for AUC on your model. Filters optimise a proxy; wrappers optimise the real thing.
- ✅ **It sees interactions automatically.** Two features that are useless alone but powerful together
  will score well *as a subset*, and a wrapper evaluates subsets. No univariate method can do this.
- ❌ **Every candidate costs a full training run**, including cross-validation, so a single "score"
  might be five model fits.

> ⚠️ **And a third cost the slide doesn't name: wrappers overfit the selection.** You are choosing
> among thousands of subsets by their cross-validated score, so the winning subset's CV score is
> **optimistically biased** — you have effectively run thousands of hypothesis tests and reported the
> best one. (Cross-validation itself — splitting the data into folds so each one gets a turn as a
> held-out test set — is taught in full in
> [Supervised Learning Part 3 §11](../Supervised%20Learning/supervised-learning-03.md); if it's new to
> you, read that first.) The fix here is a **nested** cross-validation: an inner loop for selection, an
> outer loop for honest evaluation. It multiplies your cost by the outer fold count, which is painful on
> top of an already expensive method, but reporting the inner score as your performance estimate is simply
> wrong. This is the wrapper equivalent of test-set leakage and it is a genuinely common mistake.

---

## 13. Forward selection

> *"Greedy addition of features"* [slide 49, 33:15]
>
> - *"**Start** with empty set $S = \emptyset$"*
> - *"**Each step**: add feature $x_j$ that maximizes performance of $S \cup \{x_j\}$"*
> - *"**Stop** when no feature improves score (or budget reached)"*
> - *"**Cost**: $\mathcal{O}(p \cdot d)$ model evaluations, where $d$ = final subset size"*

And the honest warning:

> *"Greedy, so it may miss feature interactions. Example: two features useless alone but powerful
> together will be skipped."*

### 🧪 Worked example — the real cost of forward selection

$p = 1000$ candidate features, target subset size $d = 20$.

| Step | Current $|S|$ | Candidates to try | Model fits |
|---|---|---|---|
| 1 | 0 | 1000 | 1000 |
| 2 | 1 | 999 | 999 |
| 3 | 2 | 998 | 998 |
| … | … | … | … |
| 20 | 19 | 981 | 981 |

$$\text{Total} = \sum_{i=0}^{19}(1000 - i) = 20 \times 1000 - \frac{19 \times 20}{2} = 20{,}000 - 190 = \mathbf{19{,}810 \text{ model fits}}$$

At **1 second per fit**, that is **5.5 hours**. With 5-fold cross-validation on each candidate — which
you need, or you're selecting on noise — it is **99,050 fits, or 27.5 hours**.

**And compare the alternative you're avoiding:** exhaustive search over all subsets of 1000 features
is $2^{1000} \approx 10^{301}$ — more than the number of atoms in the observable universe raised to
the third power. Forward selection's 19,810 is a *staggering* saving. It is just still too slow to be
your default.

### 13.1 🧪 The greedy failure, made concrete

The slide's warning deserves a real example, because it is the reason forward selection is not simply
"the right answer, just slow."

**The XOR construction.** Let $x_1, x_2 \in \{0, 1\}$ be independent fair coins, and let the target be

$$y = x_1 \oplus x_2 \quad (\text{XOR: } 1 \text{ if exactly one of them is } 1)$$

| $x_1$ | $x_2$ | $y$ | probability |
|---|---|---|---|
| 0 | 0 | 0 | ¼ |
| 0 | 1 | 1 | ¼ |
| 1 | 0 | 1 | ¼ |
| 1 | 1 | 0 | ¼ |

**Look at $x_1$ alone.** When $x_1 = 0$, $y$ is 0 half the time and 1 half the time. When $x_1 = 1$,
same. So $x_1$ carries **exactly zero** information about $y$: $\Pr[y{=}1 \mid x_1] = 0.5$ regardless.
By symmetry the same is true of $x_2$. Any model using one of them alone achieves **50% accuracy —
chance.**

**Together they determine $y$ perfectly.** Accuracy 100%.

**So forward selection fails at step 1.** It evaluates every single feature on its own, finds that
$x_1$ and $x_2$ each score 50% — no better than a useless random column — and picks something else.
Having not selected either, it never evaluates the pair. **The perfect predictor is invisible to a
greedy search that only ever adds one feature at a time.**

> 💡 **This is the same structural problem as any greedy algorithm: it cannot pay a short-term cost for
> a long-term gain.** And it's worth noticing that it's the *same* problem the XOR example caused in
> [Deep Neural Networks Part 1](../Deep%20Neural%20Networks/deep-neural-networks-01.md), where XOR is
> the standard demonstration that a single-layer perceptron cannot represent everything. XOR is
> pathological for univariate and linear methods in general, and it shows up as the counterexample in
> three different chapters of this course for one reason: **it is the simplest function whose parts
> tell you nothing about the whole.**

**What to do about it:**

| Fix | How it helps | Cost |
|---|---|---|
| **Backward elimination** | Start with *all* features and remove the least useful. Both XOR features are present at the start, so the pair's value is visible — removing either causes a large accuracy drop, so neither is removed. | Even more expensive: early fits use all $p$ features |
| **Stepwise (forward + backward)** | Allow removal of previously-added features, escaping some greedy traps | ~2× forward |
| **Genetic / random search** | Evaluates whole subsets rather than incremental additions | Tunable, no guarantee |
| **Just use an embedded method** | Trees natively model interactions (§26) | 1 model fit |

> 🎯 **"Forward or backward selection?" is a real interview question, and the answer is the XOR
> argument.** Backward elimination handles interactions better because every feature's contribution is
> evaluated *in the presence of the others*. Forward selection is faster, especially when the final
> subset is much smaller than $p$ (compare: forward with $d = 20$ from $p = 1000$ trains mostly on
> tiny models; backward starts by training on all 1000 features). **Fast-and-blind versus
> slow-and-sighted** — and the fact that the trade-off has that shape is the point.

---

## 14. Recursive Feature Elimination (RFE)

> *"Use model internals to rank and prune"* [slide 51, 34:56]
>
> - *"**Train** model on current feature set"*
> - *"**Rank** features by importance (|coefficient|, Gini, etc.)"*
> - *"**Remove** bottom $k$ features"*
> - *"**Repeat** until desired feature count reached"*
>
> *"Example: SVM-RFE uses squared weights $w_j^2$ to rank features. Popular in bioinformatics for
> gene selection."*

### 14.1 Why RFE is dramatically cheaper than forward selection

**Because it prunes in batches rather than testing candidates one at a time.** Forward selection asks
$p$ separate questions per step ("does adding *this* one help?"). RFE asks **one** question per step
("which are the worst $k$, according to the model I just trained?") and uses the model's own internals
to answer it.

### 🧪 Worked example — RFE vs forward selection, same problem

Same setup: $p = 1000 \to d = 20$.

**RFE removing $k = 10$ features per round:**

$$\text{rounds} = \frac{1000 - 20}{10} = 98 \quad\Rightarrow\quad \mathbf{98 \text{ model fits}}$$

**Versus forward selection's 19,810.** RFE is **202× cheaper** for the same target size.

Removing more per round is cheaper still, at the cost of granularity:

| $k$ per round | Rounds (fits) | vs forward selection |
|---|---|---|
| 1 | 980 | 20× cheaper |
| 10 | **98** | **202× cheaper** |
| 50 | 20 | 990× cheaper |
| Halve each round | ~6 | 3,300× cheaper |

> ⚠️ **The trade-off with large $k$: importance rankings shift as features are removed.** If two
> features are highly correlated, a model splits the credit between them, so *both* look moderately
> unimportant — and removing $k=50$ at once can drop both, when keeping either one alone would have
> been fine. Removing fewer per round lets the survivor's importance rise before the next cut. This is
> the same correlated-feature pathology that §22 shows for Lasso and §23 fixes with Elastic Net; it
> recurs throughout the lecture.

### 14.2 Is RFE a wrapper or an embedded method?

**Genuinely both, and noticing that is a good sign.** The deck places it under wrappers, and that's
defensible — it wraps a full model-training loop and evaluates repeatedly. But it uses the model's
*internal* coefficients to rank, which is the defining move of an embedded method (§20). A filter
would score features against $y$ without a model; a pure wrapper would score subsets by held-out
performance without opening the model up.

**The honest classification: RFE is a wrapper *loop* around an embedded *ranking*.** If asked, say
that — the taxonomy is a teaching device, and methods that sit between categories are normal.

### 14.3 SVM-RFE, and why squared weights

The slide's example is **SVM-RFE** (Guyon, Weston, Barnhill & Vapnik, 2002), still a standard method
in gene selection.

For a linear SVM, the decision function is $f(x) = w^\top x + b$, so feature $j$'s influence on the
prediction is carried entirely by $w_j$. Ranking by $w_j^2$ ranks by squared influence. The specific
justification in the original paper is nicer than "big weights matter": $w_j^2$ approximates the
change in the SVM's objective function if feature $j$ were removed, so it is a **first-order estimate
of the cost of deletion** — exactly the quantity a pruning method should rank by.

> ⚠️ **Weight-based ranking requires scaled features, and this bites constantly.** If `income` is in
> dollars (range $10^5$) and `n_orders` is in units (range $10^1$), then to have comparable influence
> `income`'s coefficient must be ~10,000× smaller. Rank by $|w_j|$ without standardising and you will
> delete `income` for being unimportant, when it is merely measured in small units.
> **Always `StandardScaler` before any coefficient-based selection.** Same caveat as §16's variance
> threshold, for the same reason.

```python
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# the scaler is not optional — see the warning above
sel = make_pipeline(
    StandardScaler(),
    RFE(LinearSVC(dual='auto'), n_features_to_select=20, step=10),   # step=10 → 98 fits
)
sel.fit(X, y)

# RFECV picks the subset size for you by cross-validation, at proportionally more cost
from sklearn.feature_selection import RFECV
```

---

# ACT II, part four — Filter methods

*35:30 – 41:58*

---

## 15. Filter methods: the core idea

> *"Score each feature independently, then threshold"* [slide 54, 35:33]

```
   ┌──────┐         ┌────────────────┐        ┌───────────┐        ┌──────────────┐
   │  x₁  │         │  Score(xⱼ, y)  │        │ Threshold │        │   Selected   │
   │  x₂  │────────▶│  MI, χ²,       │───────▶│ top-k or  │───────▶│ subset S ⊂   │
   │  x₃  │         │  correlation   │        │  cutoff   │        │  {x₁,…,x_p}  │
   │  …   │         └────────────────┘        └───────────┘        └──────────────┘
   └──────┘
```

> - *"**Fast**: $\mathcal{O}(p)$ scoring, no model training"*
> - *"**Model-agnostic**: same filter for any downstream model"*
> - *"**Ignores feature interactions**: scores are univariate"*

**The defining property: a filter never trains a model.** It computes $p$ independent scores
$\text{Score}(x_j, y)$ and keeps the best ones.

### 15.1 The three bullets are one trade-off

Read them together and they are not three facts but one:

$$\textbf{filters are fast and model-agnostic } \textit{because} \textbf{ they are univariate}$$

Looking at one feature at a time is what makes it $\mathcal{O}(p)$ instead of $\mathcal{O}(2^p)$, and
what makes the score independent of any downstream model. **The blindness to interactions is not a bug
that could be fixed while keeping the speed — it is the same property viewed from the other side.**
Being able to state that connection, rather than listing three separate bullet points, is what a good
answer sounds like.

### 15.2 🧪 The cost comparison that justifies filters

| Method | Model fits for $p = 1000 \to d = 20$ | Wall clock at 1 s/fit |
|---|---|---|
| Exhaustive search | $2^{1000}$ | heat death of the universe |
| Forward selection | 19,810 | 5.5 hours |
| RFE (step 10) | 98 | 1.6 minutes |
| **Filter** | **0** | **~1 second total** |

A filter computes 1000 correlations. On a modern machine that is a single vectorised operation over
your data matrix. **This is why filters are the right first move on a wide dataset even if you intend
to use something better afterwards**: cutting 10,000 features to 500 in one second makes every
subsequent method 20× cheaper.

### 15.3 ⚠️ The leakage trap — the most common filter mistake in practice

**Filters must be fit inside your cross-validation loop, not before it.**

```python
# ❌ WRONG — the filter has seen the whole of y, including the validation folds
X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
scores = cross_val_score(model, X_sel, y, cv=5)     # optimistically biased

# ✅ RIGHT — selection is refit inside every fold
pipe = make_pipeline(SelectKBest(f_classif, k=20), model)
scores = cross_val_score(pipe, X, y, cv=5)          # honest
```

**How badly does it matter?** Enough to invent results from nothing. Generate $X$ as pure random noise
with $p = 10{,}000$ and $y$ as a random coin flip. Select the 20 features most correlated with $y$
across the full dataset, then cross-validate. **You will get accuracy well above chance** — because
with 10,000 noise features, some will correlate with $y$ by luck, and you chose them using the very
labels you are about to validate against. The signal is entirely manufactured by the selection step.

This is the feature-selection version of the same error as evaluating on your training set, and it is
the single most common way a published-looking result turns out to be nothing. **`Pipeline` exists
precisely to make this impossible** — use it.

---

## 16. Variance threshold

> *"Remove features with near-zero variance"* [slide 55, 36:34]

$$\mathrm{Var}(X_j) = \mathbb{E}\left[(X_j - \mu_j)^2\right]$$

> - *"If $\mathrm{Var}(X_j) < \varepsilon$, feature is nearly constant"*
> - *"A constant feature carries **zero information**"*
> - *"Simplest possible filter, always a good first step"*
> - *"Caveat: scale features first (or use coefficient of variation)"*
>
> *"In scikit-learn: `VarianceThreshold(threshold=0.01)`"*

**Note what makes this filter unique: it never looks at $y$.** Every other filter in §17–§19 scores
the feature *against the target*. Variance threshold is purely a property of the feature column, which
means (a) it can be applied to unlabelled data, and (b) **it cannot leak** — the §15.3 trap does not
apply. It is genuinely safe to run before your cross-validation loop.

**Why a constant feature carries zero information — properly.** If $X_j = c$ for every sample, then
$H(X_j) = 0$ (Prereq 4: no uncertainty, no bits), and since $I(X_j; Y) \le H(X_j)$, the mutual
information with *any* target is zero. It is not merely useless for this problem — it is provably
useless for every possible problem. And in a linear model, a constant column is perfectly collinear
with the intercept, making $X^\top X$ singular.

### 🧪 Worked example — the scale caveat, which is the whole point of this slide

The caveat is the interesting part, because variance is **not scale-invariant** and people forget.

Take one dataset of adult heights, recorded twice in different units:

| Feature | Values | Mean $\mu$ | Std $\sigma$ | **Variance** |
|---|---|---|---|---|
| `height_m` (metres) | ~1.7 | 1.70 | 0.10 | **0.01** |
| `height_mm` (millimetres) | ~1700 | 1700 | 100 | **10,000** |

**These are the same measurements of the same people.** Their variances differ by a factor of
$10^6$ — because $\mathrm{Var}(cX) = c^2\mathrm{Var}(X)$, and $c = 1000$ gives $c^2 = 10^6$.

Now apply the slide's own default, `VarianceThreshold(threshold=0.01)`:

- `height_m`: variance $0.01$, **not** $> 0.01$ → **dropped**
- `height_mm`: variance $10{,}000 \gg 0.01$ → **kept**

**You have deleted a feature for being measured in sensible units.** In a real dataset with mixed
units — prices in dollars, rates as fractions, counts as integers — a raw variance threshold
systematically deletes everything measured on a small scale, which is often the well-normalised
features you were most careful about.

**Two fixes, and the slide names both:**

**1. Standardise first.** After `StandardScaler`, every feature has variance exactly 1 — at which
point a variance threshold does nothing at all. So this "fix" only works for the *near-constant*
case, which standardising makes numerically obvious (a constant feature has $\sigma = 0$ and
standardising it is a division by zero — sklearn leaves it as zeros, and it's then trivially caught).

**2. Coefficient of variation**, $\mathrm{CV} = \sigma/\mu$ — a *relative* spread, and dimensionless:

$$\text{height\_m: } \frac{0.10}{1.70} = \mathbf{0.0588} \qquad \text{height\_mm: } \frac{100}{1700} = \mathbf{0.0588}$$

**Identical** ✓ — as they must be, since the units cancel. CV is the scale-invariant version of the
same idea. ⚠️ It is undefined at $\mu = 0$ and unstable near it, so it is only usable for strictly
positive features.

```python
import numpy as np
h_m  = np.array([1.60, 1.65, 1.70, 1.75, 1.80])
h_mm = h_m * 1000

print(h_m.var(), h_mm.var())                       # 0.005  5000.0     — differ by 1e6
print(h_m.std()/h_m.mean(), h_mm.std()/h_mm.mean())# 0.0417 0.0417     — identical
```

> 💡 **The practical recipe:** use variance threshold only to catch *literally* constant or
> near-constant columns (`threshold=0`, or a tiny epsilon after scaling). It is a **data-quality
> check**, not a feature-selection method — its job is to find the columns your ETL silently filled
> with a default value, not to rank informative features. Treat a nonzero hit as a bug report about
> your pipeline.

---

## 17. Pearson correlation

> *"Detects linear relationships only"* [slide 57, 37:12]

$$r_{XY} = \frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

The slide shows two scatter plots: one tight upward line labelled **$r \approx 1$**, and one scattered
cloud labelled **$r \approx 0$ (nonlinear!)**.

> - *"Misses nonlinear dependencies (parabola, XOR)"*
> - *"Use $|r| >$ threshold to select features correlated with target"*

### 17.1 What $r$ measures, and what it doesn't

| $r$ | Means |
|---|---|
| $+1$ | Perfect **increasing linear** relationship — all points exactly on an upward line |
| $0$ | **No linear** relationship. **Says nothing about other relationships.** |
| $-1$ | Perfect **decreasing linear** relationship |

The division by $\sigma_X\sigma_Y$ (Prereq 2) is what bounds $r$ to $[-1,1]$ and makes it comparable
across features in different units — which is exactly what covariance alone cannot do.

> ⚠️ **"$r$ measures the strength of the relationship" is wrong, and it's the most common
> misunderstanding.** $r$ measures the strength of the **linear** relationship. A perfect parabola
> has $r = 0$ and a perfect relationship.

### 17.2 🧪 Worked example — a perfect relationship with $r = 0$ exactly

$X \in \{-2, -1, 0, 1, 2\}$ each with probability $\tfrac15$, and $Y = X^2$:

| $X$ | $-2$ | $-1$ | $0$ | $1$ | $2$ |
|---|---|---|---|---|---|
| $Y$ | 4 | 1 | 0 | 1 | 4 |
| $X - \bar{X}$ | $-2$ | $-1$ | $0$ | $1$ | $2$ |
| $Y - \bar{Y}$ | $2$ | $-1$ | $-2$ | $-1$ | $2$ |
| product | $-4$ | $1$ | $0$ | $-1$ | $4$ |

$\bar{X} = 0$; $\bar{Y} = \frac{4+1+0+1+4}{5} = 2$.

$$\mathrm{Cov}(X,Y) = \frac{-4 + 1 + 0 - 1 + 4}{5} = \frac{0}{5} = \mathbf{0} \quad\Longrightarrow\quad r = \mathbf{0}$$

**Not approximately zero — exactly zero.** And $Y$ is a deterministic function of $X$: tell me $X$ and
I will tell you $Y$ with certainty.

**Why the cancellation happens, which is the insight rather than the arithmetic:** the products at
$X = -2$ and $X = +2$ are $-4$ and $+4$. The parabola is *symmetric*, so every point on the left
contributes exactly the negative of its mirror on the right. Correlation is a signed, linear
statistic; a symmetric relationship cancels it to zero by construction.

**The general lesson:** any relationship symmetric about $\bar{X}$ has $r = 0$. Parabolas, absolute
values, U-shapes, sine over a full period, XOR. These are not exotic — U-shaped relationships are
everywhere in real data (a drug's effect versus dose; churn versus tenure; conversion versus price).

```python
import numpy as np
X = np.array([-2, -1, 0, 1, 2]); Y = X**2
print(np.corrcoef(X, Y)[0, 1])          # 0.0  — exactly
from sklearn.feature_selection import mutual_info_regression
print(mutual_info_regression(X.reshape(-1,1), Y))   # clearly > 0
```

### 17.3 What to use instead

| Method | Catches | Cost | When |
|---|---|---|---|
| **Pearson $r$** | Linear only | $\mathcal{O}(n)$ | Default; fine when you expect monotone linear effects |
| **Spearman $\rho$** | Any **monotonic** relationship (rank-based) | $\mathcal{O}(n\log n)$ | **The cheap upgrade.** Catches saturating and exponential curves Pearson underestimates. Still misses U-shapes. |
| **Mutual information** | **Any** dependence | $\mathcal{O}(n \log n)$-ish, noisier | §18. Use when you genuinely expect non-monotonic structure |
| **Distance correlation** | Any dependence, and $= 0 \iff$ independent | $\mathcal{O}(n^2)$ | ⚠️ Beyond this deck. Cleaner theory than MI, worse scaling |

> 💡 **Spearman is the answer most people don't give and should.** It is Pearson computed on the
> *ranks* rather than the values, costs a sort, has none of MI's estimation difficulties, and catches
> every monotonic relationship. If you're going to name one alternative to Pearson in an interview,
> name this one first and MI second.

### 17.4 The other use of correlation: between features

The slide frames $r$ as a feature-versus-target score, but the other use matters just as much and the
deck doesn't mention it: **correlation between features** detects redundancy.

If $r(x_1, x_2) = 0.98$, they carry nearly the same information. Keeping both wastes a dimension,
destabilises linear-model coefficients (near-singular $X^\top X$ — see §21's motivation), and splits
importance scores between them so both look unimportant (the §14.1 pathology again).

**The standard recipe:** compute the $p \times p$ correlation matrix; for each pair above 0.95, drop
one. ⚠️ This is $\mathcal{O}(p^2)$ in both memory and time, so at $p = 50{,}000$ the matrix alone is
10 GB — apply a cheaper filter first. And note the greedy "drop one" is arbitrary about *which* one;
Elastic Net (§23) handles correlated groups more principledly.

---

## 18. Mutual information

> *"Captures any statistical dependence"* [slide 58, 37:15]

$$I(X;Y) = H(X) - H(X \mid Y) = \sum_{x,y} p(x,y) \log\frac{p(x,y)}{p(x)\,p(y)}$$

| Symbol | Read it as | What it means |
|---|---|---|
| $I(X;Y)$ | "the mutual information between X and Y" | How many bits knowing $Y$ tells you about $X$. Symmetric: $I(X;Y) = I(Y;X)$. |
| $H(X)$ | "entropy of X" | Your uncertainty about $X$ before seeing anything (Prereq 4). |
| $H(X \mid Y)$ | "conditional entropy" | Your remaining uncertainty about $X$ **after** learning $Y$. |
| $p(x,y)$ | "joint probability" | Probability of seeing $x$ and $y$ together. |
| $p(x)p(y)$ | — | What the joint *would* be if they were independent. **The ratio inside the log compares reality to independence.** |

> - *"**Non-parametric**: no linearity assumption"*
> - *"Measures **reduction in uncertainty** about X given Y"*
> - *"$I(X;Y) = 0 \iff X \perp Y$"*
> - *"Detects XOR, U-shape, and other nonlinear patterns that Pearson misses"*

### 18.1 The two forms say the same thing twice

**Form 1: $I = H(X) - H(X\mid Y)$** — *"how much your uncertainty about $X$ drops once you know $Y$."*
This is the intuitive reading. If $Y$ tells you nothing, $H(X \mid Y) = H(X)$ and the difference is 0.
If $Y$ determines $X$, $H(X\mid Y) = 0$ and $I = H(X)$ — the whole of $X$'s uncertainty is resolved.

**Form 2: $I = \sum p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$** — *"how far the joint distribution is from
what independence would predict."* This is the KL divergence between $p(x,y)$ and $p(x)p(y)$, and it
is the form that makes the third bullet obvious: if $X \perp Y$ then $p(x,y) = p(x)p(y)$ exactly, so
every log is $\log 1 = 0$, so $I = 0$. And KL divergence is zero *only* when the distributions match —
which gives the "if and only if".

> 💡 **That "if and only if" is MI's whole selling point, and it is exactly what Pearson lacks.**
> $r = 0$ does not imply independence. $I = 0$ **does**. So MI can never miss a dependence, of any
> shape. The price is in §18.4.

### 🧪 Worked example — MI on §17.2's parabola, computed by hand

Same data: $X$ uniform on $\{-2,-1,0,1,2\}$, $Y = X^2$. We showed $r = 0$ exactly. Now compute $I$.

**Step 1 — the distribution of $Y$.** $Y = 4$ when $X \in \{-2, 2\}$; $Y = 1$ when $X \in \{-1, 1\}$;
$Y = 0$ when $X = 0$:

$$p(Y{=}4) = \tfrac25, \qquad p(Y{=}1) = \tfrac25, \qquad p(Y{=}0) = \tfrac15$$

**Step 2 — the entropy of $Y$.**

$$H(Y) = -\left[\tfrac25\log_2\tfrac25 + \tfrac25\log_2\tfrac25 + \tfrac15\log_2\tfrac15\right]$$

$\log_2 0.4 = -1.3219$ and $\log_2 0.2 = -2.3219$, so:

$$H(Y) = -\left[0.4(-1.3219) + 0.4(-1.3219) + 0.2(-2.3219)\right] = 0.5288 + 0.5288 + 0.4644 = \mathbf{1.522 \text{ bits}}$$

**Step 3 — the conditional entropy.** $Y$ is a deterministic function of $X$: given $X$, there is no
uncertainty left about $Y$ at all. So

$$H(Y \mid X) = \mathbf{0}$$

**Step 4 — mutual information.**

$$I(X;Y) = H(Y) - H(Y\mid X) = 1.522 - 0 = \mathbf{1.522 \text{ bits}}$$

**The result, side by side:**

$$r = \mathbf{0} \qquad\qquad I(X;Y) = \mathbf{1.522 \text{ bits}}$$

Pearson says "no relationship." MI says "$X$ tells you **everything** there is to know about $Y$" —
because 1.522 bits is $H(Y)$, the *maximum possible* value. **This single pair of numbers is the
answer to §11's quiz, and it's worth memorising as a two-line demonstration.**

> 📚 **Why $I(X;Y) \le \min(H(X), H(Y))$**, which is the bound that makes 1.522 recognisable as
> maximal: $I = H(Y) - H(Y\mid X)$, and entropy is never negative, so $I \le H(Y)$. Symmetrically
> $I \le H(X)$. **You cannot learn more about $Y$ than $Y$ contains** — obvious once stated, and
> useful for sanity-checking an MI estimate.

### 18.2 Normalised MI, for comparing across features

Raw MI is in bits and is bounded by $H(Y)$ — so it is **not comparable across features with different
entropies**. A binary feature can never exceed 1 bit; a 100-category feature can reach 6.6. Ranking
raw MI systematically favours high-cardinality features. (The same bias appears in Gini importance in
§26 — it's a recurring theme.)

The standard fix is **normalised mutual information**:

$$\mathrm{NMI} = \frac{I(X;Y)}{\sqrt{H(X)H(Y)}} \in [0, 1]$$

`sklearn.metrics.normalized_mutual_info_score` implements this (with a choice of normaliser).

### 18.3 🧪 MI catches XOR, which no univariate linear method can

Recall §13.1's XOR: $y = x_1 \oplus x_2$ with $x_1, x_2$ fair coins.

- **Pearson** $r(x_1, y) = 0$, and $r(x_2, y) = 0$.
- **MI** $I(x_1; y) = 0$, and $I(x_2; y) = 0$ — **also zero!**
- **But** $I\big((x_1,x_2); y\big) = 1$ bit — the pair determines $y$ completely.

> ⚠️ **Read that carefully, because it corrects an over-claim it would be easy to make.** The deck says
> MI *"detects XOR ... that Pearson misses"*, and that is true for the *joint* MI of the pair. It is
> **not** true for univariate MI on each feature separately — which is what a filter computes. In XOR
> each feature alone is genuinely independent of $y$, so univariate MI is correctly zero.
>
> **MI removes the linearity assumption. It does not remove the univariate assumption.** A filter is
> still a filter: §15's third bullet ("ignores feature interactions") applies to MI exactly as much as
> to Pearson. If you need interactions, you need a wrapper or an embedded method — which is precisely
> the argument for §20.
>
> Being able to make this distinction is a strong signal in an interview, because it shows you're
> tracking *two* independent limitations rather than conflating them.

### 18.4 ⚠️ The estimation problem

> *"Estimation requires density estimation (kNN-based or binning). Noisier than Pearson for small
> samples."*

Pearson has a closed form: plug numbers in, get an answer. MI requires estimating $p(x,y)$, and that's
a density-estimation problem — which is hard for exactly §3's reason.

**Two approaches, both with failure modes:**

| Approach | How | Failure mode |
|---|---|---|
| **Binning** | Discretise into bins, count | Bin count is a hyperparameter with no good default. Too few bins → miss structure. Too many → every point in its own bin → **MI estimated as maximal for pure noise**. |
| **kNN-based** (Kraskov et al., 2004 — the deck's citation) | Estimate density from distances to the $k$ nearest neighbours | Better, and the default in sklearn. But it uses distances — so **§2's distance concentration degrades it in high dimensions**, which is a genuinely awkward circularity: the tool for high-dimensional feature selection is itself weakened by high dimensionality. |

**The practical consequence: MI estimates are biased *upward* on noise.** Estimator noise looks like
dependence. So a feature can score high in MI purely by chance, especially with few samples or many
categories.

**How to defend yourself** — the permutation test, and it takes three lines:

```python
from sklearn.feature_selection import mutual_info_classif
import numpy as np

real = mutual_info_classif(X, y, random_state=0)
null = np.array([mutual_info_classif(X, np.random.permutation(y), random_state=0)
                 for _ in range(50)])          # MI against deliberately destroyed labels

keep = real > np.percentile(null, 95, axis=0)  # beat the 95th percentile of pure chance
```

Shuffling $y$ destroys any real relationship while preserving both marginal distributions, so the
shuffled scores are exactly the estimator's noise floor. Anything not clearing it is not evidence.

---

## 19. Chi-square and ANOVA

The section's summary slide is a **lookup table**, and it is the single most immediately useful slide
in the lecture [slide 60, 38:53]:

> *"Statistical tests for different feature/target types"*

| Test | Feature type | Target type | Detects |
|---|---|---|---|
| **Chi-square (χ²)** | Categorical | Categorical | Association between categories |
| **ANOVA F-test** | Continuous | Categorical | Mean differences across groups |
| **Pearson r** | Continuous | Continuous | Linear correlation |
| **Mutual Info** | Any | Any | Any dependence |

> *"Rule of thumb: match the test to your data types. Using the wrong test yields meaningless
> scores."*

**Memorise this table.** It is a two-by-two on (feature type × target type) with MI as the universal
fallback, and getting it wrong is one of the most common errors in applied feature selection —
sklearn will happily run `f_classif` on a categorical feature and return numbers that mean nothing.

### 19.1 Chi-square: testing independence of two categoricals

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \qquad E_{ij} = \frac{(\text{row } i \text{ total})(\text{col } j \text{ total})}{n}$$

**The logic in one sentence:** compute what the counts *would* be if the feature and the target were
independent, compare to what you actually observed, and square the differences.

The expected count $E_{ij}$ is exactly independence written down: under independence,
$\Pr[\text{row } i \text{ and col } j] = \Pr[\text{row }i]\Pr[\text{col }j]$, so the expected count is
$n \cdot \frac{r_i}{n}\cdot\frac{c_j}{n} = \frac{r_i c_j}{n}$.

### 🧪 Worked example — does seeing an ad predict a purchase?

|  | Bought | Did not buy | **Row total** |
|---|---|---|---|
| **Saw ad** | 30 | 70 | **100** |
| **No ad** | 20 | 180 | **200** |
| **Column total** | **50** | **250** | **300** |

**Step 1 — expected counts under independence.** Overall, $50/300 = 16.7\%$ of people bought. If the
ad had no effect, that rate should hold in both rows:

$$E_{\text{saw, bought}} = \frac{100 \times 50}{300} = 16.67 \qquad E_{\text{saw, not}} = \frac{100 \times 250}{300} = 83.33$$
$$E_{\text{no, bought}} = \frac{200 \times 50}{300} = 33.33 \qquad E_{\text{no, not}} = \frac{200 \times 250}{300} = 166.67$$

*Sanity check:* the expected counts sum to $16.67 + 83.33 + 33.33 + 166.67 = 300$ ✓, and each row and
column total matches the observed ✓ — that's guaranteed by construction, and it's a good check that
you've built the table right.

**Step 2 — the deviations.** Every cell is off by exactly $\pm 13.33$ (in a 2×2 table it must be —
with the margins fixed, one cell determines the rest, which is why $\text{df} = 1$):

$$O - E: \quad +13.33, \quad -13.33, \quad -13.33, \quad +13.33$$

**Step 3 — sum the squared relative deviations.** $13.33^2 = 177.78$:

$$\chi^2 = \frac{177.78}{16.67} + \frac{177.78}{83.33} + \frac{177.78}{33.33} + \frac{177.78}{166.67}$$

$$= 10.667 + 2.133 + 5.333 + 1.067 = \mathbf{19.20}$$

**Step 4 — interpret.** Degrees of freedom $= (2-1)(2-1) = 1$. The critical value at $\alpha = 0.05$
is **3.841**. Our 19.20 is far beyond it ($p \approx 1.2 \times 10^{-5}$), so we reject independence:
**seeing the ad is strongly associated with buying.**

> ⚠️ **Association is not causation, and this example is a trap worth walking into deliberately.**
> People who saw the ad may differ systematically from those who didn't — they were browsing the
> category, or they're existing customers, or the ad was *targeted* at likely buyers in the first
> place. χ² establishes that the columns are not independent; it says nothing about which way the
> arrow points or whether there's a third variable driving both. **Module 7 (Causal Inference) is
> entirely about this gap.**

**Note also which cells drove the statistic:** the "saw ad / bought" cell contributed 10.67 of the
19.20 — over half — because it has the smallest expected count, and $E$ is in the denominator.
Small-expected-count cells dominate χ², which is why the standard rule of thumb is that **every cell
should have $E \ge 5$**; below that the χ² approximation degrades and you want Fisher's exact test.

```python
from scipy.stats import chi2_contingency
import numpy as np
table = np.array([[30, 70], [20, 180]])
chi2, p, dof, expected = chi2_contingency(table, correction=False)
print(chi2, p, dof)     # 19.2  1.18e-05  1
print(expected)         # [[ 16.67  83.33] [ 33.33 166.67]]
```

> ⚠️ **sklearn's `chi2` is not this test.** `sklearn.feature_selection.chi2` requires **non-negative**
> features and computes a chi-square between each feature and the class label treating the feature
> values as *counts* (it's designed for term-frequency data). Feeding it standardised or
> negative-valued features raises an error; feeding it arbitrary positive continuous features returns
> numbers that are not the test above. For genuinely categorical features, one-hot encode and use
> `chi2`, or use `scipy.stats.chi2_contingency` on the contingency table directly.

### 19.2 ANOVA F-test: a continuous feature against a categorical target

**The question:** does this continuous feature have different *means* across the target's classes?

$$F = \frac{\text{variance BETWEEN group means}}{\text{variance WITHIN groups}}$$

**The intuition, which is the part worth carrying:** if a feature separates the classes, the class
means will be far apart (large numerator) relative to the spread inside each class (small
denominator), so $F$ is large. If the feature is irrelevant, the class means are all about the same
and $F \approx 1$.

*Concretely:* to predict churn from `days_since_last_login`, compute the mean for churners and
non-churners. If churners average 45 days and non-churners 3 days, with a within-group spread of
about 5 days, the between-group difference dwarfs the within-group noise and $F$ is huge. If both
groups average 20 days, $F \approx 1$ and the feature is useless.

> ⚠️ **The F-test detects *mean* differences only** — so it inherits a Pearson-like blindness. A
> feature where class A is bimodal at $\{0, 100\}$ and class B is concentrated at 50 has **identical
> means** and is perfectly separable. $F \approx 1$; the feature is excellent. This is the same
> lesson as §17.2 in a different costume: **univariate summary statistics can be identical for very
> different distributions.**

```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
# continuous features, categorical target  → f_classif   (ANOVA F)
# categorical (count) features, cat. target → chi2
# anything at all                           → mutual_info_classif
# continuous features, continuous target    → f_regression / mutual_info_regression
```

### 19.3 The decision tree for choosing a filter

```
What is your TARGET?
│
├── Categorical (classification)
│   ├── Feature continuous  ──▶  ANOVA F-test        f_classif
│   └── Feature categorical ──▶  Chi-square           chi2 (on one-hot counts)
│
└── Continuous (regression)
    ├── Feature continuous  ──▶  Pearson r            f_regression
    │                             (or Spearman if you expect a monotone curve)
    └── Feature categorical ──▶  ANOVA F-test, roles swapped
                                  (group the target by the feature's categories)

    ...and if you suspect a NON-MONOTONIC relationship, or don't know:
                             ──▶  Mutual information  mutual_info_*
                                  (slower, noisier, but assumption-free)
```

---

# ACT II, part five — Embedded methods

*41:58 – 49:42*

---

## 20. Embedded methods: the core idea

> *"Feature selection happens during model training"* [slide 64, 40:25]
>
> - *"The **loss function** includes a penalty that shrinks or zeros coefficients"*
> - *"No separate feature selection step needed"*
> - *"Model and selection are **jointly optimized**"*
> - *"Faster than wrappers, richer than filters"*
>
> *"Key idea: add a regularization term $\Omega(\beta)$ to the objective that encourages sparsity."*

**The fourth bullet is the whole pitch, and it is why embedded methods are the practical default:**

| | Filter | Wrapper | **Embedded** |
|---|---|---|---|
| Cost | $\mathcal{O}(p)$, no fits | $\mathcal{O}(p\cdot d)$ fits | **~1 fit** |
| Sees interactions | ❌ | ✅ | **✅ (within its hypothesis class)** |
| Optimises your actual objective | ❌ (a proxy) | ✅ | **✅** |

It gets most of the wrapper's advantages for roughly a filter's price. The catch is in that
parenthesis — an embedded method sees whatever interactions *its own model class* can represent, so a
Lasso sees linear structure and a tree sees axis-aligned interactions. It cannot select for a model
you haven't trained.

### 20.1 The idea that makes it work: relaxing a discrete problem

**This is the most elegant thing in the lecture and it is worth stating explicitly**, because the deck
implies it without saying it.

Feature selection is naturally a **discrete** problem: each of $p$ features is in or out, giving $2^p$
options and no gradient to follow (§9.1). Written honestly, "select at most $k$ features" is

$$\min_\beta \|y - X\beta\|^2 \quad \text{subject to} \quad \|\beta\|_0 \le k$$

where $\|\beta\|_0$ counts the **nonzero** entries. This is NP-hard: you cannot do better than
searching.

**The L1 penalty is a *convex relaxation* of that count.** Replace $\|\beta\|_0$ with $\|\beta\|_1$
and the problem becomes convex — solvable in polynomial time, with a unique optimum and no local
minima — while still driving coefficients to exactly zero. **You get discrete selection out of a
continuous optimisation.**

$$\underbrace{\|\beta\|_0 \le k}_{\text{what you want; NP-hard}} \quad\longrightarrow\quad \underbrace{\|\beta\|_1 \le t}_{\text{what you solve; convex}} \quad\longrightarrow\quad \text{sparse } \beta$$

> 💡 **That relaxation is why Lasso was a landmark result rather than just another penalty.** Tibshirani
> (1996) is one of the most-cited papers in statistics because it turned an intractable combinatorial
> problem into a tractable convex one *without* giving up the property that made the combinatorial
> version desirable. §22 shows geometrically why the relaxation keeps the zeros.

---

## 21. Ridge regression (L2)

> *"Shrinks coefficients but never zeros them"* [slide 65, 40:28]

$$\min_{\beta}\ \|y - X\beta\|^2 + \lambda\|\beta\|_2^2$$

> - *"**Geometry**: constraint region is a circle (sphere in higher D)"*
> - *"Solution touches the circle, rarely at an axis"*
> - *"All coefficients shrink **proportionally**"*
> - *"Good for multicollinearity, not for feature selection"*
>
> Cites **Hoerl & Kennard (1970)**, *"Ridge Regression: Biased Estimation for Nonorthogonal Problems"*

### 21.1 Reading the geometry picture

The slide shows elliptical contours (the residual sum of squares) meeting a teal circle (the L2
constraint), touching at a point $\hat\beta$ that is off both axes.

> 📚 **Background the slide assumed** — *why a "constraint region" at all?*
>
> There are two equivalent ways to write a regularized problem:
>
> $$\underbrace{\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_2^2}_{\text{penalised form (what you code)}} \qquad\Longleftrightarrow\qquad \underbrace{\min_\beta \|y-X\beta\|^2 \ \text{ s.t. } \|\beta\|_2^2 \le t}_{\text{constrained form (what you draw)}}$$
>
> For every $\lambda$ there is a $t$ giving the same solution (Lagrangian duality), with **large
> $\lambda$ ↔ small $t$**. The penalised form is what you implement; **the constrained form is the one
> you can draw**, which is why every textbook picture — and this slide — uses it.
>
> **What's in the picture:** the ellipses are level sets of the residual sum of squares, centred on the
> unconstrained OLS solution and growing outward. The shaded region is the set of $\beta$ you're
> allowed to use. The answer is where **the smallest ellipse that still touches the region** meets it.
> Everything about L1 versus L2 comes from *what shape that region is.*

**Why the touch point is "rarely at an axis":** a circle is *smooth* — its boundary has no
distinguished points. An ellipse approaching from a random direction touches it at a random place, and
the axes are a measure-zero subset of the circle. So $\Pr[\beta_j = 0] = 0$: **Ridge produces small
coefficients, never zero ones.**

### 21.2 "Good for multicollinearity" — what that means and why

> 📚 **Multicollinearity** — two or more features are (nearly) linear combinations of each other.
>
> *Concretely:* you have `height_cm` and `height_inches` in the same model. They carry identical
> information.
>
> *Why it breaks OLS:* the OLS solution is $\hat\beta = (X^\top X)^{-1}X^\top y$. With perfectly
> collinear columns, $X^\top X$ is **singular** and the inverse does not exist. With *nearly*
> collinear columns it is nearly singular, so the inverse has enormous entries and the coefficients
> become wildly unstable — you might get $+500$ on height_cm and $-197$ on height_inches, a pair that
> nearly cancels and that will flip completely on a slightly different sample.

**How Ridge fixes it, exactly.** The Ridge solution has a closed form:

$$\hat\beta_{\text{ridge}} = \left(X^\top X + \lambda I\right)^{-1}X^\top y$$

**Adding $\lambda I$ adds $\lambda$ to every eigenvalue of $X^\top X$.** A singular matrix has a zero
eigenvalue; $X^\top X + \lambda I$ has a smallest eigenvalue of at least $\lambda > 0$, so it is
**always invertible**. The fix is not a heuristic — it is arithmetic, and it's where the name "ridge"
comes from (adding a ridge along the diagonal).

> 💡 **This is a good answer to "why would you use Ridge if it can't do feature selection?"** Ridge
> isn't a worse Lasso; it solves a *different* problem. When features are correlated and you want
> **stable, reliable coefficient estimates**, Ridge is the right tool and Lasso is the wrong one —
> because Lasso will arbitrarily pick one of a correlated pair and zero the other, and that choice
> flips between bootstrap samples (§22.3).

### 21.3 "All coefficients shrink proportionally"

For the special case of orthonormal features ($X^\top X = I$), the Ridge solution is exactly

$$\hat\beta_j^{\text{ridge}} = \frac{\hat\beta_j^{\text{OLS}}}{1 + \lambda}$$

Every coefficient is divided by the same number. A coefficient of 100 becomes $100/(1+\lambda)$; one
of 0.01 becomes $0.01/(1+\lambda)$. **Neither ever reaches zero, however large $\lambda$ gets** — you
can only ever divide by a bigger number.

Compare this to Lasso's orthonormal solution, which is the **soft-threshold**:

$$\hat\beta_j^{\text{lasso}} = \mathrm{sign}(\hat\beta_j^{\text{OLS}})\left(|\hat\beta_j^{\text{OLS}}| - \tfrac{\lambda}{2}\right)_+$$

where $(z)_+ = \max(z, 0)$. **Subtract a constant, and clip at zero.** Anything smaller than
$\lambda/2$ becomes *exactly* zero.

$$\textbf{Ridge divides. Lasso subtracts-and-clips.}$$

> 🎯 **That one line is the fastest correct answer to "why does L1 give sparsity and L2 doesn't?"**
> Division by a positive number can approach zero but never arrive; subtraction of a fixed amount
> arrives immediately. It is the same argument as the gradient comparison in
> [Deep Neural Networks Part 2 §10](../Deep%20Neural%20Networks/deep-neural-networks-02.md), and §22.2
> gives a third version (the geometric one). **Know all three and pick by audience:** subtract-vs-divide
> is fastest, the gradient argument is most rigorous, the diamond picture is most memorable.

---

## 22. Lasso regression (L1)

> *"Diamond constraint forces coefficients to zero"* [slide 67, 43:23]

$$\min_{\beta}\ \|y - X\beta\|^2 + \lambda\|\beta\|_1$$

> - *"**Geometry**: constraint region is a diamond (cross-polytope)"*
> - *"Corners lie on axes: solution hits a corner with $\beta_j = 0$"*
> - *"Produces **sparse** models: automatic feature selection"*
> - *"Struggles with correlated features (picks one, drops others)"*
>
> Cites **Tibshirani (1996)**, *"Regression Shrinkage and Selection via the Lasso"*

The slide's picture is the same as Ridge's, but the constraint region is now an orange **diamond**,
and the touch point sits **on the $\beta_1$ axis**, labelled $\beta_2 = 0$.

### 22.1 🧪 The geometric argument, done properly

**Why the L1 ball is a diamond.** In 2D, $\|\beta\|_1 = |\beta_1| + |\beta_2| \le t$ describes the set
bounded by four line segments: $\beta_1 + \beta_2 = t$ in the first quadrant, and its reflections.
That is a square rotated 45° — a diamond — with **vertices at $(\pm t, 0)$ and $(0, \pm t)$**.

**Those vertices are on the coordinate axes. That is the entire mechanism.** At the vertex $(t, 0)$,
$\beta_2$ is exactly zero — a feature has been deselected.

**Why an expanding ellipse tends to hit a vertex.** Contrast the two boundaries:

- **A circle** is smooth. At every boundary point there is exactly one tangent direction, so an
  ellipse must arrive at precisely the right angle to touch there. Every point is equally (im)probable,
  and the axis points are a measure-zero subset.
- **A diamond's vertex is a corner.** At a corner, there is a whole *fan* of supporting lines — a
  range of angles all of which touch there. So an ellipse arriving from any direction within that fan
  hits the corner. **A corner has positive probability of being the touch point; a smooth boundary
  point does not.**

**In higher dimensions this gets stronger, not weaker.** The L1 ball in $\mathbb{R}^p$ has $2p$
vertices (on the axes), plus edges, 2-faces, and so on. A $k$-dimensional face of the L1 ball is
exactly the set where $p - k$ coordinates are zero. So:

$$\text{touching a } k\text{-face} \quad\Longleftrightarrow\quad \text{selecting exactly } k \text{ features}$$

The low-dimensional faces (few features) are the "sharp" parts and attract the solution. **As $p$
grows, the fraction of the L1 ball's surface that is sharp grows too — so Lasso gets *more* sparse in
higher dimensions, which is exactly when you need it.**

### 22.2 The three explanations, and when to use each

| Explanation | The argument | Best for |
|---|---|---|
| **Geometric** | Diamond's corners lie on the axes; corners attract the touch point | Whiteboards. Most memorable. The deck's version. |
| **Gradient** | $\frac{d}{d\beta}\lambda\lvert\beta\rvert = \lambda\,\mathrm{sign}(\beta)$ — **constant** magnitude, so small coefficients are pushed to zero at the same absolute rate as large ones. L2's $\lambda\beta$ shrinks as $\beta$ shrinks. | Fastest to state. Fully worked in [DNN Part 2 §10](../Deep%20Neural%20Networks/deep-neural-networks-02.md). |
| **Soft-threshold** | $\hat\beta = \mathrm{sign}(\hat\beta^{\text{OLS}})(\lvert\hat\beta^{\text{OLS}}\rvert - \lambda/2)_+$ — subtract and clip | Most precise. Shows *exactly* which coefficients die and when. |

All three are the same fact. **The subgradient version deserves one extra line** because it explains
*why zero is special rather than merely reachable*: $|\beta|$ is not differentiable at $\beta = 0$, and
its subgradient there is the whole interval $[-\lambda, +\lambda]$. So $\beta_j = 0$ is optimal
whenever the data's pull on that coefficient is anywhere inside that interval — **a whole range of
data configurations map to exactly zero**, rather than a single knife-edge. That's the analytic
counterpart of "a corner has a fan of supporting lines."

### 22.3 ⚠️ "Struggles with correlated features" — the failure that motivates §23

Suppose $x_1$ and $x_2$ are **perfectly correlated** and jointly predictive, with the true relationship
$y = 2x_1$ (equivalently $y = 2x_2$, or any split between them).

**Every allocation with $\beta_1 + \beta_2 = 2$ fits the data identically.** And if both are positive,
they also have **identical L1 penalty**:

$$\|\beta\|_1 = |\beta_1| + |\beta_2| = \beta_1 + \beta_2 = 2 \quad \text{for all such splits}$$

$(2, 0)$, $(0, 2)$, $(1, 1)$, $(1.7, 0.3)$ — **Lasso is exactly indifferent between all of them.** The
optimum is not unique; it is a whole line segment.

**What happens in practice:** coordinate descent lands on a vertex, so you get $(2,0)$ or $(0,2)$ —
one feature selected, the other zeroed, **chosen essentially by numerical accident**. Resample the
data and the choice can flip. If you are using Lasso to decide which genes to investigate, that
instability is a serious problem: two runs give two different answers with equal justification.

**The right diagnostic:** bootstrap. Refit on 100 resamples and count how often each feature is
selected. Genuinely important features are selected consistently; members of a correlated group split
their selection frequency. (This is the idea behind **stability selection**, Meinshausen & Bühlmann,
2010.)

```interactive
type: simulator
title: Circle versus diamond — where the ellipse lands
concept: Why the L1 constraint region produces exact zeros and the L2 one does not
control: A toggle between L1 and L2 (and a blend for Elastic Net); a slider for the constraint budget t; and a draggable OLS solution that moves the RSS ellipse's centre
observe: The constraint region and the smallest touching ellipse, with the touch point marked and the two coefficient values read out live — highlighted in red whenever one is exactly zero
insight: Dragging the OLS point around a full circle, the L2 touch point moves smoothly and never hits an axis, while the L1 touch point SNAPS to a vertex over a wide range of angles — which is the "fan of supporting lines" argument seen rather than argued
fallback: The geometric argument in §22.1: a circle is smooth so the axes are a measure-zero subset of it, while a diamond's vertices lie on the axes and have a whole fan of supporting lines, hence positive probability of being the touch point

```

---

## 23. Elastic Net

> *"Combines L1 and L2 penalties"* [slide 69, 44:05]

$$\min_{\beta}\ \|y - X\beta\|^2 + \lambda_1\|\beta\|_1 + \lambda_2\|\beta\|_2^2$$

> - *"**L1 component**: drives coefficients to zero (sparsity)"*
> - *"**L2 component**: stabilizes when features are correlated"*
> - *"Selects **groups** of correlated features together (Lasso picks one arbitrarily)"*
> - *"Two hyperparameters: overall strength $\lambda$ and mixing ratio $\alpha$"*
>
> *"In scikit-learn: `ElasticNet(alpha=1.0, l1_ratio=0.5)`. Set `l1_ratio=1` for pure Lasso,
> `l1_ratio=0` for Ridge."*
>
> Cites **Zou & Hastie (2005)**, *"Regularization and Variable Selection via the Elastic Net"*

### 🧪 Worked example — proving the grouping effect

**This derivation is short and it completely explains the third bullet.** It is the best thing to have
ready if asked about Elastic Net.

Take §22.3's setup: $x_1, x_2$ perfectly correlated, and any split with $\beta_1 + \beta_2 = c$ fits
the data equally well. We showed Lasso is indifferent because $\|\beta\|_1 = c$ for every split.

**Now ask what the L2 penalty does to those same splits.** Minimise $\beta_1^2 + \beta_2^2$ subject to
$\beta_1 + \beta_2 = c$. By Lagrange multipliers, $2\beta_1 = \mu$ and $2\beta_2 = \mu$, so

$$\boxed{\beta_1 = \beta_2 = c/2}$$

**Compare the L2 penalty at the two extremes:**

| Split | L1 penalty | L2 penalty |
|---|---|---|
| $(c, 0)$ — Lasso's answer | $c$ | $c^2$ |
| $(c/2, c/2)$ — the even split | $c$ | $c^2/2$ ✅ **half as much** |

**L1 cannot tell them apart. L2 strictly prefers the even split, by a factor of two.** So adding *any*
amount of L2 breaks the tie in favour of sharing — and the solution becomes unique. That is the
"grouping effect", proved in three lines.

*Concretely with $c = 2$:* $(2,0)$ has L2 penalty $4 + 0 = 4$; $(1,1)$ has $1 + 1 = 2$. The even split
wins.

**And the general principle worth extracting:** the L2 penalty is *strictly convex*, so it has a unique
minimiser on any convex set. The L1 penalty is only *convex*, so it can be flat along a face — which is
exactly where Lasso's non-uniqueness lives. **Adding a strictly convex term to a merely convex one
makes the whole objective strictly convex, hence uniquely solved.** That sentence is the real theorem
behind Elastic Net.

### 23.1 The two hyperparameters, and sklearn's parameterisation

sklearn reparameterises the two lambdas into a strength and a mix, which is more practical:

$$\lambda_1 = \alpha \cdot \texttt{l1\_ratio}, \qquad \lambda_2 = \alpha \cdot (1 - \texttt{l1\_ratio})$$

| `l1_ratio` | Behaviour |
|---|---|
| **1.0** | Pure **Lasso** — maximum sparsity, unstable on correlated groups |
| **0.5** | Default. Balanced. |
| **0.0** | Pure **Ridge** — no sparsity at all |

> ⚠️ **`alpha` means different things in different sklearn estimators, and this catches everyone.** In
> `ElasticNet` and `Lasso`, `alpha` is the regularization *strength* (what the maths calls $\lambda$).
> In `Ridge` it is also the strength. But `l1_ratio` is the *mix* — and the deck's slide writes
> `alpha=1.0, l1_ratio=0.5`, which is strength 1.0 with a 50/50 mix. Meanwhile the maths literature
> often uses $\alpha$ for the mixing ratio. **Read the docstring, not your memory.**

```python
from sklearn.linear_model import ElasticNetCV
# tune BOTH: the mix over a grid, the strength by internal CV path
model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], cv=5)
model.fit(X, y)
print(model.l1_ratio_, model.alpha_, (model.coef_ != 0).sum())
```

Note the recommended `l1_ratio` grid is **skewed toward 1** — that's sklearn's own documented advice,
because the interesting behaviour (sparsity) lives near the Lasso end and you want resolution there.

### 23.2 When to use which — the decision table

| Situation | Use | Because |
|---|---|---|
| Few features truly matter, and they're uncorrelated | **Lasso** | Maximum sparsity, and the instability never bites |
| All features matter somewhat; features are correlated | **Ridge** | Stable estimates; you don't want zeros |
| Features correlated **in groups**, only some groups matter | **Elastic Net** | Zeros out whole irrelevant groups, keeps relevant ones **together** |
| $p \gg n$ and you need more than $n$ features | **Elastic Net** | ⚠️ **Lasso can select at most $n$ features** — a hard structural limit (Zou & Hastie prove it). Elastic Net has no such cap. |
| You need a stable, reproducible feature list | **Elastic Net** or Lasso + stability selection | §22.3 |

> 💡 **That $p \gg n$ row is the least-known and most impressive point.** In the $n = 200$ patients,
> $p = 20{,}000$ genes setting, Lasso cannot possibly return more than 200 genes no matter what
> $\lambda$ you choose — it's a property of the geometry, not a tuning issue. If the biology involves
> 500 relevant genes, Lasso *cannot* find them and Elastic Net can. This is one of the two headline
> motivations in the original Elastic Net paper.

---

## 24. 🎯 Quick check — 50 features correlated in groups

> **You have 50 features, many correlated in groups. Which regularizer?** [slide 72, 46:01]
>
> A) Lasso (L1) — sparsity, picks one from each group arbitrarily
> B) Ridge (L2) — shrinks all, never zeros, keeps all correlated features
> C) Elastic Net — groups correlated features together, zeros out irrelevant groups

<details>
<summary><b>Answer</b> — this one's answer <b>was</b> captured</summary>

**C.** The slide's own answer line reads:

> *"**Answer: C.** Elastic Net selects groups of correlated features while still achieving sparsity."*

**Why the other two fall short, in the language of §23's proof:**

- **A) Lasso** delivers the sparsity you want but is **indifferent** among splits within a correlated
  group ($\|\beta\|_1$ is identical for all of them), so it picks one arbitrarily and the choice is
  unstable across resamples. You get *a* sparse answer, but not a *reproducible* one — and when the
  point of feature selection is to tell someone which 10 features to care about, reproducibility is
  the deliverable.
- **B) Ridge** is perfectly stable on correlated groups — that's what §21.2 is about — but it produces
  **no zeros at all**. You still have all 50 features. That is not feature selection; it's coefficient
  shrinkage.

**C gets both**, and the mechanism is §23's three-line proof: the L1 term supplies the zeros, and the
strictly-convex L2 term breaks the within-group tie so members of a group are kept or dropped
*together*.

> 🎯 **The follow-up to be ready for: "how would you set `l1_ratio`?"** Cross-validate it, with a grid
> skewed toward 1 (e.g. `[0.1, 0.5, 0.9, 0.95, 0.99, 1.0]`) because the behaviour changes fastest near
> the Lasso end. And validate the *stability* of the selected set, not only the score: bootstrap the
> fit and check that the same features keep appearing. A model with 2% better CV that selects a
> different 10 features every run is worse than a stable one, for any purpose where the feature list
> is the product.
</details>

---

## 25. Regularization paths

> *"How coefficients evolve as lambda increases"* [slide 75, 47:41]
>
> - *"**Lasso path**: coefficients drop to zero one by one as $\lambda$ grows"*
> - *"**Ridge path**: all coefficients shrink toward zero but never reach it"*
> - *"The order features 'die' in Lasso reveals importance ranking"*
> - *"Cross-validate $\lambda$ to find the optimal sparsity level"*

The plot shows solid orange Lasso curves hitting the axis at different points, against dashed teal
Ridge curves that approach it asymptotically and never arrive.

```
   βⱼ │╲╲╲
      │ ╲ ╲╲ ─ ─ ─               Lasso (solid): each curve TOUCHES ZERO
      │  ╲  ╲╲     ─ ─ ─         Ridge (dashed): each curve APPROACHES zero
      │   ╲   ╲╲        ─ ─ ─
      │╲   ╲    ╲╲           ─ ─ ─
      │ ╲   ╲     ╲╲
      │  ╲   ╲      ╲╲
      │───●────●──────●──────────────  ← the axis: Lasso arrives, Ridge doesn't
      └────────────────────────────── λ →
        f₃   f₂      f₁
        dies dies    dies
        first        last  ⇒ f₁ is the most important
```

### 25.1 The third bullet is the useful one

> *"The order features 'die' in Lasso reveals importance ranking"*

This turns the path from a diagnostic plot into a **feature-ranking algorithm**, and it's better than
ranking by coefficient magnitude at a single $\lambda$:

- A feature that survives to a large $\lambda$ is one the model refuses to give up. It is important.
- A feature that dies at small $\lambda$ was marginal all along.
- **You get a complete ranking from a single path computation**, not one ranking per $\lambda$.

This is the same idea as RFE's recursive ranking (§14), obtained *for free* as a side effect of one
fit rather than by 98 refits. It's a nice illustration of §20's claim that embedded methods get
wrapper-like information at filter-like cost.

> 💡 **Why computing the whole path is cheap: LARS.** You might expect a path to require refitting at
> many $\lambda$ values. It doesn't. **Least Angle Regression** (Efron, Hastie, Johnstone & Tibshirani,
> 2004) exploits the fact that the Lasso solution is **piecewise linear in $\lambda$** — the
> coefficients trace straight lines, with kinks exactly where a feature enters or leaves. So you only
> need to compute the kink locations, and the entire path costs roughly the same as a single OLS fit.
> `sklearn.linear_model.lars_path` and `lasso_path` implement this.

### 25.2 The fourth bullet: cross-validate $\lambda$

$\lambda$ is not a free choice — it is a **bias–variance dial** (small $\lambda$ → low bias, high
variance, close to OLS; large $\lambda$ → high bias, low variance, everything shrinks toward zero; see
the trade-off derived in full in
[Supervised Learning Part 1 §9](../Supervised%20Learning/supervised-learning-01.md)) and must be tuned
on held-out data.

| $\lambda$ | Effect | Failure |
|---|---|---|
| 0 | Plain OLS | Overfits, and undefined if $p > n$ |
| Too small | Nearly all features kept | Overfitting; no useful selection |
| **CV-optimal** | The right sparsity for *your* data | — |
| Too large | Nearly all coefficients zero | Underfitting; at the limit, predicts the mean |

```python
from sklearn.linear_model import lasso_path, LassoCV
import numpy as np

alphas, coefs, _ = lasso_path(X, y)      # the whole path in one call
order = np.argsort([np.argmax(c != 0) for c in coefs])   # earliest to enter = most important

cv = LassoCV(cv=5).fit(X, y)             # pick lambda honestly
print(cv.alpha_, (cv.coef_ != 0).sum())
```

> ⚠️ **`LassoCV` returns the $\lambda$ with the best mean CV score, which is systematically too small.**
> The CV curve is flat near its minimum and noisy, so the argmin tends to land on the optimistic side.
> The standard correction is the **"one standard error" rule**: choose the *largest* $\lambda$ whose CV
> score is within one standard error of the best. That gives a sparser, more robust model at a
> statistically indistinguishable score. It's the default in R's `glmnet` and is not automatic in
> sklearn — you have to implement it from `cv.mse_path_` yourself.

---

## 26. Tree-based feature importance

> *"Embedded selection via tree structure"* [slide 78, 49:22]
>
> - *"**Gini importance**: total decrease in impurity from all splits on feature $j$"*
> - *"**Permutation importance**: drop in accuracy when feature $j$ is shuffled"*

| | **Gini** | **Permutation** |
|---|---|---|
| **Speed** | Free (from training) | Requires re-evaluation |
| **Bias** | Favors high-cardinality features | Unbiased |
| **Interactions** | Partial capture | Full capture |

> *"Random Forests average importance across many trees, reducing variance of the estimate."*

### 26.1 Gini importance — free, and biased

A decision tree splits to reduce **impurity** (Gini or entropy). Gini importance sums, over every
split on feature $j$ in the tree, the impurity reduction that split achieved, weighted by how many
samples reached it.

**Why it's free:** the tree already computed every one of those numbers while deciding where to split.
Reading them off costs nothing. This is `feature_importances_` in sklearn.

> ⚠️ **The cardinality bias is severe, and it is the thing to know about this metric** (Strobl et al.,
> 2007). A feature with many distinct values offers many possible split points, so **by chance alone**
> one of them will separate the training data somewhat — and that spurious reduction is counted as
> importance.
>
> **The demonstration that makes it undeniable:** add a column of random unique IDs (`customer_id`,
> a UUID, a row index) to your data. It carries *no* information — you generated it. Yet it will often
> rank **among the most important features** by Gini importance, because with $n$ distinct values it
> has $n-1$ candidate split points and can carve the training data almost arbitrarily.
>
> **The ordering of harm:** continuous features > high-cardinality categoricals > low-cardinality
> categoricals > binary. So Gini importance systematically under-rates your binary flags and
> over-rates your continuous ones. If your feature set mixes types — and it always does — **the
> ranking is comparing apples to oranges**, and it does so silently.

### 26.2 Permutation importance — honest, and slower

Shuffle one feature's column and measure how much performance drops.

```
1. Score the trained model on held-out data      → baseline
2. For each feature j:
       shuffle column j (destroying its relationship with y, keeping its marginal)
       re-score                                   → score_j
       importance_j = baseline − score_j
```

**Why shuffling rather than dropping:** dropping a column changes the input shape and would require
retraining; shuffling keeps the shape and the marginal distribution intact while destroying only the
*association*. So it measures exactly "how much did this model rely on this feature?" without
retraining.

**Why it's unbiased:** it measures actual predictive contribution rather than an artefact of split
counting. A random-ID column shuffles to no effect, because the model's reliance on it doesn't
generalise to held-out data — so it correctly gets ~zero importance.

**Cost:** one re-evaluation per feature (usually repeated a few times and averaged), so
$\mathcal{O}(p)$ **predictions** — much cheaper than $\mathcal{O}(p)$ *retrainings*, but not free.

> ⚠️ **Compute permutation importance on held-out data, not on the training set.** On training data a
> model that memorised a useless feature will show a large drop when it's shuffled — you'd be measuring
> memorisation, not signal. sklearn's `permutation_importance` accepts whatever you pass it and will
> not warn you.

> ⚠️ **And both methods mislead on correlated features, for the same reason but in opposite ways.**
> If $x_1$ and $x_2$ are duplicates, permuting $x_1$ barely hurts — the model just leans on $x_2$ —
> so **both look unimportant** even though the pair is essential. Gini splits the credit between them,
> with the same result. **Neither metric will tell you a correlated pair matters.** The fix is to
> permute correlated features *as a group*, or to cluster features by correlation first. This is the
> third appearance of the correlated-features pathology (after §14.1 and §22.3), and its recurrence is
> itself the lesson: **correlated features break almost every importance measure.**

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0)
for i in r.importances_mean.argsort()[::-1][:10]:
    print(f"{feature_names[i]:30s} {r.importances_mean[i]:.4f} ± {r.importances_std[i]:.4f}")
```

### 26.3 Why Random Forests improve the estimate

> *"Random Forests average importance across many trees, reducing variance of the estimate."*

A single decision tree is high-variance: change a few training rows and the top split can change,
cascading into a completely different tree and a completely different importance ranking. Averaging
over $B$ trees reduces the variance of the estimate by roughly $1/B$ for the independent part.

**And random feature subsampling helps in a second, subtler way.** Because each split only considers
a random subset of features, a feature that would always be overshadowed by a stronger correlated
partner sometimes gets evaluated *in its absence* — so it can accumulate importance rather than being
permanently masked. It's a partial mitigation of the correlated-features problem above, though not a
cure.

### 26.4 🎯 Which importance measure to report

| Situation | Use |
|---|---|
| Quick exploration, all features same type | Gini (free) |
| **Reporting importance to anyone** | **Permutation, on held-out data** |
| Features have mixed cardinality | **Permutation** — Gini is comparing apples to oranges |
| Correlated feature groups | Permutation **by group**, or cluster first |
| You need per-prediction attribution | SHAP ⚠️ *beyond this deck* |

> 🎯 **"How would you determine feature importance?" is a common interview question, and the strong
> answer starts by asking what for.** *Importance for what?* — model debugging, feature selection,
> stakeholder explanation and regulatory attribution have different right answers. Then name the
> Gini cardinality bias, prefer permutation on held-out data, and flag the correlated-feature caveat.
> That sequence demonstrates you've been burned by this before, which is exactly the signal being
> looked for.

---

# ACT III — The linear algebra behind PCA

*49:42 – 57:33*

Everything up to here has been **selection** — choosing among the columns you already have. The rest of
the module is **extraction** — building new ones. Extraction needs a way to decide *which directions
through the data matter*, and that answer is eigenvectors of the covariance matrix. These five slides
are the whole prerequisite, and the lecture ends the moment they're delivered.

---

## 27. The covariance matrix

> *"Encodes the shape of your data cloud"* [slide 82, 50:59]

$$\Sigma = \frac{1}{n}X^\top X \quad \text{(for centered } X)$$

> - *"**Diagonal** entries: $\Sigma_{jj} = \mathrm{Var}(x_j)$"*
> - *"**Off-diagonal** entries: $\Sigma_{jk} = \mathrm{Cov}(x_j, x_k)$"*
> - *"Symmetric and positive semi-definite"*
> - *"**Geometry**: defines the data ellipse (ellipsoid in higher D)"*
>
> *"The covariance matrix is the input to PCA. Understanding it geometrically is key."*
>
> Cites **Hotelling (1933)**, *"Analysis of a Complex of Statistical Variables into Principal Components"*

### 27.1 Reading the formula

| Symbol | Read it as | What it means | Shape |
|---|---|---|---|
| $X$ | "X" | The **centered** data matrix — every column's mean subtracted | $n \times p$ |
| $X^\top X$ | "X transpose X" | Feature-by-feature dot products (Prereq 5) | $p \times p$ |
| $\frac{1}{n}$ | — | Turns sums into averages | — |
| $\Sigma$ | "Sigma" (capital) | The covariance matrix | $p \times p$ |

> ⚠️ **"For centered $X$" is a load-bearing condition, not a footnote.** $\frac1n X^\top X$ is the
> matrix of *second moments*; it equals the covariance **only** when every column has mean zero.
> Forget to center, and $\Sigma_{jj} = \mathbb{E}[x_j^2]$ rather than $\mathrm{Var}(x_j)$ — inflated by
> $\mu_j^2$. A feature with mean 1000 and variance 1 would show up as 1,000,001, and PCA would return
> a first component that points at the data's *centroid* rather than at its direction of greatest
> spread. **Uncentered PCA is a classic silent bug**; `sklearn.decomposition.PCA` centers for you, but
> if you compute $\Sigma$ by hand or use `TruncatedSVD` (which deliberately does *not* center), you
> must handle it.

**Why the entries are what the slide says.** Expand the matrix product (Prereq 5):

$$\Sigma_{jk} = \frac{1}{n}\sum_{i=1}^{n} X_{ij}X_{ik}$$

For centered data, $X_{ij} = x_{ij} - \mu_j$, so this is exactly
$\frac1n\sum_i (x_{ij}-\mu_j)(x_{ik}-\mu_k) = \mathrm{Cov}(x_j, x_k)$ ✓. Set $j = k$ and it becomes
$\frac1n\sum_i(x_{ij}-\mu_j)^2 = \mathrm{Var}(x_j)$ ✓ — the diagonal.

### 27.2 The two structural properties, and why each matters

**Symmetric:** $\Sigma_{jk} = \Sigma_{kj}$, because $\mathrm{Cov}(a,b) = \mathrm{Cov}(b,a)$.

> 💡 **Symmetry is the property that makes PCA work at all.** The **spectral theorem** guarantees that
> a real symmetric matrix has (a) all **real** eigenvalues, and (b) an **orthonormal** basis of
> eigenvectors. Both matter enormously: real eigenvalues mean "variance along this direction" is a
> real number, and orthonormal eigenvectors mean the principal components are mutually perpendicular
> and unit-length — a proper coordinate system. For a general non-symmetric matrix neither is
> guaranteed. §29's $\Sigma = V\Lambda V^\top$ is the spectral theorem written out.

**Positive semi-definite (PSD):** $v^\top \Sigma v \ge 0$ for every vector $v$.

> 📚 **What PSD means here, in one line:** $v^\top \Sigma v$ is exactly **the variance of the data
> projected onto direction $v$**. Variance is an average of squares, so it cannot be negative — hence
> $\Sigma$ is PSD, necessarily.

**Prove it, because the proof is two lines and it explains the whole of PCA:**

$$v^\top \Sigma v = v^\top\!\left(\tfrac1n X^\top X\right)\!v = \tfrac1n (Xv)^\top (Xv) = \tfrac1n\|Xv\|^2 \ \ge 0$$

And $Xv$ is precisely the data projected onto $v$ — one number per sample. So $\frac1n\|Xv\|^2$ is the
mean squared projection, which (for centered data) is the **variance along $v$**. $\blacksquare$

**Two consequences you get for free:**
- All eigenvalues $\lambda_i \ge 0$ — which is why §30's scree plot only ever shows non-negative bars.
- **PCA's objective is now visible:** "find the direction of maximum variance" is
  $\max_{\|v\|=1} v^\top\Sigma v$, and the spectral theorem says the answer is the top eigenvector.
  §31 states exactly this.

### 27.3 The geometry — the "data ellipse"

The fourth bullet is the one the deck says is *"key"*, and it's the picture to carry into Part 2.

A cloud of centered points has a shape. Draw the ellipse that best captures it — the set
$\{x : x^\top \Sigma^{-1} x = c\}$ — and:

- Its **axes** point along the eigenvectors of $\Sigma$.
- Its **axis lengths** are proportional to $\sqrt{\lambda_i}$.

So $\Sigma$ doesn't just store variances; **it stores the full shape and orientation of the cloud.**

| $\Sigma$ | Ellipse | Meaning |
|---|---|---|
| $\begin{bmatrix}1 & 0\\0 & 1\end{bmatrix}$ | Circle | Both features equally spread, uncorrelated |
| $\begin{bmatrix}4 & 0\\0 & 1\end{bmatrix}$ | Axis-aligned ellipse, 2× wider than tall | Feature 1 has 4× the variance; still uncorrelated |
| $\begin{bmatrix}1 & 0.9\\0.9 & 1\end{bmatrix}$ | **Tilted** ellipse along $y = x$ | Strongly correlated — and the tilt is the correlation |

**That third row is the entire motivation for PCA.** The ellipse's long axis is *not* along either
original feature — it's along $(1,1)/\sqrt2$. Neither original coordinate captures the main direction
of variation, but a *combination* does. **Extraction finds that combination; selection cannot.**

### 🧪 Worked example — compute $\Sigma$, its eigenvectors, and its eigenvalues by hand

Four samples, two features, already centered:

$$X = \begin{bmatrix} 2 & 1 \\ -1 & -0.5 \\ -2 & -1 \\ 1 & 0.5 \end{bmatrix}$$

*Check it's centered:* column 1 sums to $2 - 1 - 2 + 1 = 0$ ✓; column 2 to $1 - 0.5 - 1 + 0.5 = 0$ ✓.

**Step 1 — form $X^\top X$.**

$$(X^\top X)_{11} = 2^2 + (-1)^2 + (-2)^2 + 1^2 = 4 + 1 + 4 + 1 = 10$$
$$(X^\top X)_{12} = (2)(1) + (-1)(-0.5) + (-2)(-1) + (1)(0.5) = 2 + 0.5 + 2 + 0.5 = 5$$
$$(X^\top X)_{22} = 1^2 + (-0.5)^2 + (-1)^2 + (0.5)^2 = 1 + 0.25 + 1 + 0.25 = 2.5$$

**Step 2 — divide by $n = 4$.**

$$\Sigma = \frac14\begin{bmatrix}10 & 5\\ 5 & 2.5\end{bmatrix} = \begin{bmatrix}2.5 & 1.25\\ 1.25 & 0.625\end{bmatrix}$$

Symmetric ✓. Diagonal entries are the two feature variances (2.5 and 0.625) ✓.

**Step 3 — eigenvalues.** For a $2\times2$ matrix, use trace and determinant:

$$\mathrm{tr}(\Sigma) = 2.5 + 0.625 = 3.125 = \lambda_1 + \lambda_2$$
$$\det(\Sigma) = (2.5)(0.625) - (1.25)^2 = 1.5625 - 1.5625 = \mathbf{0} = \lambda_1\lambda_2$$

The determinant is **exactly zero**, so one eigenvalue is zero and the other takes the whole trace:

$$\lambda_1 = 3.125, \qquad \lambda_2 = 0$$

> 💡 A zero eigenvalue means $\Sigma$ is **singular** — the data is **rank-deficient**, lying entirely
> within a lower-dimensional subspace. Look back at $X$ and you can see it: every row is a multiple of
> $(2, 1)$, namely $1\times$, $-0.5\times$, $-1\times$, $0.5\times$. **This "2-dimensional" data is
> genuinely 1-dimensional** — a perfect miniature of the manifold hypothesis (§7).

**Step 4 — eigenvectors.** For $\lambda_1 = 3.125$, solve $(\Sigma - 3.125 I)v = 0$:

$$\begin{bmatrix}-0.625 & 1.25\\ 1.25 & -2.5\end{bmatrix}\begin{bmatrix}v_1\\v_2\end{bmatrix} = 0 \implies -0.625v_1 + 1.25 v_2 = 0 \implies v_1 = 2v_2$$

So $v \propto (2, 1)$. Normalising by $\|(2,1)\| = \sqrt5 = 2.2361$:

$$\mathbf{v}_1 = \left(\tfrac{2}{\sqrt5}, \tfrac{1}{\sqrt5}\right) = (0.8944,\ 0.4472)$$

And the second eigenvector must be orthogonal to it (spectral theorem):

$$\mathbf{v}_2 = (-0.4472,\ 0.8944)$$

*Check orthogonality:* $(0.8944)(-0.4472) + (0.4472)(0.8944) = -0.4 + 0.4 = 0$ ✓

**Step 5 — project onto the first component and verify the variance.**

$$z_i = x_i^\top \mathbf{v}_1: \qquad \begin{aligned}
(2, 1) &\to 2(0.8944) + 1(0.4472) = \phantom{-}2.2361\\
(-1, -0.5) &\to -0.8944 - 0.2236 = -1.1180\\
(-2, -1) &\to -2.2361\\
(1, 0.5) &\to \phantom{-}1.1180
\end{aligned}$$

$$\mathrm{Var}(z) = \frac{2.2361^2 + 1.1180^2 + 2.2361^2 + 1.1180^2}{4} = \frac{5 + 1.25 + 5 + 1.25}{4} = \frac{12.5}{4} = \mathbf{3.125} = \lambda_1 \ \checkmark$$

**The eigenvalue *is* the variance along its eigenvector.** That is the sentence §28 states and this
example verifies numerically. And since $\lambda_2 = 0$, projecting onto $\mathbf{v}_1$ alone loses
**exactly nothing** — 2 features compressed to 1 with zero reconstruction error.

```python
import numpy as np
X = np.array([[2, 1], [-1, -0.5], [-2, -1], [1, 0.5]])
S = X.T @ X / len(X)
print(S)                             # [[2.5   1.25 ] [1.25  0.625]]
w, V = np.linalg.eigh(S)
print(w[::-1])                       # [3.125  0.   ]
print(V[:, ::-1])                    # first column ≈ [0.8944, 0.4472]
print((X @ V[:, -1]).var())          # 3.125  — matches lambda_1
```

---

## 28. Eigendecomposition

> *"Finding the principal axes of the data ellipse"* [slide 85, 53:22]

$$\Sigma \mathbf{v} = \lambda \mathbf{v}$$

> - *"**Eigenvectors** $\mathbf{v}$: directions of maximum spread"*
> - *"**Eigenvalues** $\lambda$: variance along each direction"*
> - *"Eigenvectors are **orthogonal** (for symmetric $\Sigma$)"*
> - *"Sorted by $\lambda_1 \ge \lambda_2 \ge \ldots \ge \lambda_d$"*
>
> Cites **Pearson (1901)**, *"On Lines and Planes of Closest Fit"*; **Hotelling (1933)**

The slide draws a tilted ellipse of scattered points with two arrows: a long orange one along the
major axis labelled $\mathbf{v}_1(\lambda_1)$, and a shorter teal one perpendicular to it labelled
$\mathbf{v}_2(\lambda_2)$.

### 28.1 What the equation says

> **Eigenvector** — a direction that a matrix does not *rotate*; it only stretches or shrinks it.
> **Eigenvalue** — the factor by which it stretches.
>
> *In everyday words:* most vectors get knocked sideways when you apply a matrix. Eigenvectors are the
> special ones that come out pointing the same way, just longer or shorter.
>
> *Concretely:* for $\Sigma = \begin{bmatrix}2.5 & 1.25\\1.25 & 0.625\end{bmatrix}$ from §27, take
> $v = (0.8944, 0.4472)$. Then
> $\Sigma v = (2.5 \times 0.8944 + 1.25\times 0.4472,\ 1.25\times 0.8944 + 0.625\times 0.4472) =
> (2.7951, 1.3975)$, which is exactly $3.125 \times (0.8944, 0.4472)$ ✓. Same direction, scaled by
> $\lambda_1 = 3.125$.

**But the deck's reading is the more useful one for PCA**, and it is worth stating separately from the
generic definition:

$$\textbf{eigenvector} = \textbf{a direction through the data} \qquad \textbf{eigenvalue} = \textbf{how much variance lies along it}$$

That reading is justified by §27.2's proof: the variance along a unit direction $v$ is $v^\top\Sigma v$,
and if $v$ is an eigenvector then $v^\top \Sigma v = v^\top(\lambda v) = \lambda\|v\|^2 = \lambda$.
**For a unit eigenvector, the eigenvalue simply *is* the variance along it.** §27's worked example
confirms it numerically (3.125 both ways).

### 28.2 Why orthogonality matters so much

The spectral theorem gives orthogonal eigenvectors for symmetric $\Sigma$, and this has three
consequences that PCA depends on entirely:

1. **The components form a valid coordinate system.** Orthonormal axes mean you can express any point
   as a combination of them, and recover it exactly.
2. **The components are uncorrelated.** In the new coordinates, the covariance matrix is *diagonal* —
   the redundancy between original features has been removed by construction. This is the deepest
   thing PCA does: **it decorrelates.**
3. **Variances add.** Because the directions are orthogonal, total variance is the sum of the
   per-component variances:

$$\sum_j \mathrm{Var}(x_j) = \mathrm{tr}(\Sigma) = \sum_i \lambda_i$$

> 💡 **That trace identity is what makes "explained variance ratio" meaningful.** Because the
> eigenvalues sum to the total variance, $\lambda_i / \sum_k\lambda_k$ is an honest *fraction of the
> total* — the pieces genuinely add to 100%. If the components weren't orthogonal, they'd overlap and
> the "fractions" would double-count. Every scree plot in §30 rests on this.

*Verify on §27's example:* $\mathrm{tr}(\Sigma) = 2.5 + 0.625 = 3.125$, and
$\lambda_1 + \lambda_2 = 3.125 + 0 = 3.125$ ✓.

### 28.3 The sorting convention

> *"Sorted by $\lambda_1 \ge \lambda_2 \ge \ldots \ge \lambda_d$"*

This is pure convention, but it is what makes "the first $k$ components" a meaningful phrase. Sorted
descending, **taking the top $k$ eigenvectors is provably the best possible $k$-dimensional linear
summary** — both in the sense of preserving the most variance and of minimising reconstruction error
(§31 states both, and Part 2 proves it).

> ⚠️ **NumPy does not sort for you, and this is a real bug source.** `np.linalg.eigh` returns
> eigenvalues in **ascending** order — the opposite of the convention. `np.linalg.eig` returns them in
> no guaranteed order at all. Always sort explicitly, and reorder the eigenvector *columns* to match:
>
> ```python
> w, V = np.linalg.eigh(Sigma)      # ascending!
> idx = w.argsort()[::-1]           # descending
> w, V = w[idx], V[:, idx]          # reorder BOTH, together
> ```
>
> Use `eigh` (not `eig`) whenever the matrix is symmetric: it's faster, and it guarantees real outputs
> instead of returning complex numbers with negligible imaginary parts.

```interactive
type: diagram
title: The data ellipse and its eigenvectors
concept: Eigenvectors are the ellipse's axes; eigenvalues are the variance along them
control: Drag the covariance entries (or drag the point cloud's shape directly); a toggle to project the data onto the first eigenvector only
observe: The point cloud with its fitted ellipse, the two eigenvector arrows scaled by sqrt(lambda), the covariance matrix updating live, and the explained-variance ratio as a two-segment bar
insight: Increasing the off-diagonal entry visibly TILTS the ellipse away from both original axes — making it concrete that when features are correlated, neither original coordinate captures the main direction of variation, which is exactly why extraction beats selection here
fallback: §27's worked example, where Sigma = [[2.5, 1.25], [1.25, 0.625]] has eigenvalues 3.125 and 0, eigenvector (0.894, 0.447), and the projected data has variance exactly 3.125 — confirming that the eigenvalue IS the variance along its eigenvector

```

---

## 29. Spectral decomposition

> *"Matrix factorization of the covariance"* [slide 88, 55:02]

$$\Sigma = V\Lambda V^\top$$

> - *"$V = [\mathbf{v}_1|\mathbf{v}_2|\ldots|\mathbf{v}_d]$: orthonormal eigenvectors as columns"*
> - *"$\Lambda = \mathrm{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)$: eigenvalues on diagonal"*
> - *"$V^\top V = I$: rotation matrix (no stretching, just reorientation)"*
> - *"**Connection to SVD**: if $X = USV^\top$, then $\Sigma = V(S^2/n)V^\top$"*
>
> *"PCA projects data onto the first $k$ columns of $V$. The truncated SVD achieves the same thing more
> efficiently."*

### 29.1 The factorization is $d$ separate eigen-equations, bundled

$\Sigma\mathbf{v}_i = \lambda_i\mathbf{v}_i$ holds for each $i$. Stack all $d$ of them side by side and
you get $\Sigma V = V\Lambda$; right-multiply by $V^\top$ and use $VV^\top = I$:

$$\Sigma = V\Lambda V^\top$$

**Read it right to left as three geometric operations** — this is the reading that makes it stick:

$$\underbrace{V}_{\text{3. rotate back}} \ \underbrace{\Lambda}_{\text{2. stretch along the axes}} \ \underbrace{V^\top}_{\text{1. rotate into the eigenbasis}}$$

1. $V^\top$ **rotates** your coordinates so the data's principal axes line up with the coordinate axes.
2. $\Lambda$ **stretches** each of those axes by its variance — and being diagonal, it acts on each
   independently, with no mixing.
3. $V$ **rotates back** to the original coordinates.

**So a covariance matrix is: rotate, scale, rotate back.** Its complexity is entirely in *where the
axes point*; once you're in the right frame it's just $d$ independent numbers. That is why "find the
right frame" — which is what PCA does — is so powerful.

### 29.2 $V^\top V = I$ means rotation, and why that's important

An orthogonal matrix preserves lengths and angles:
$\|Vx\|^2 = (Vx)^\top(Vx) = x^\top V^\top V x = x^\top x = \|x\|^2$.

**So PCA's change of basis does not distort your data.** Projecting onto *all* $d$ components is a
lossless relabelling — you could rotate straight back and recover $X$ exactly. **Information is only
lost when you truncate to $k < d$ components**, and §31 tells you exactly how much.

> 💡 **This is worth saying explicitly because it separates two things people conflate.** PCA does two
> distinct operations: (1) **rotate** into the eigenbasis — lossless, always safe; and (2)
> **truncate** to $k$ components — lossy, and the only place a decision is being made. When someone
> says "PCA loses information", the honest response is that the rotation loses nothing and the
> truncation loses exactly $\sum_{i>k}\lambda_i$ of the variance, which you can compute in advance.

### 29.3 The SVD connection, and why it's the practical route

> *"if $X = USV^\top$, then $\Sigma = V(S^2/n)V^\top$"*

**Derive it in one line.** Substitute $X = USV^\top$ into $\Sigma = \frac1n X^\top X$:

$$\Sigma = \tfrac1n (USV^\top)^\top(USV^\top) = \tfrac1n VS^\top U^\top U S V^\top = \tfrac1n VS^\top S V^\top = V\!\left(\tfrac{S^2}{n}\right)\!V^\top$$

using $U^\top U = I$ ($U$ is orthonormal) and $S^\top S = S^2$ ($S$ is diagonal). $\blacksquare$

**Match it against $\Sigma = V\Lambda V^\top$ term by term:**

$$\boxed{\ \text{the } V \text{ in the SVD of } X \ \textbf{is} \ \text{the } V \text{ in the eigendecomposition of } \Sigma, \quad\text{and}\quad \lambda_i = \frac{s_i^2}{n}\ }$$

**Why this matters practically — three reasons, in increasing importance:**

1. **You skip forming $\Sigma$.** With $p = 100{,}000$ features, $\Sigma$ is $100{,}000^2 = 10^{10}$
   entries — **40 GB** in float32. The SVD of $X$ never builds it.
2. **Better numerical conditioning.** Forming $X^\top X$ **squares the condition number**, so you lose
   roughly half your significant digits before you even start. Small eigenvalues — precisely the ones
   near the noise floor where you're deciding what to keep — are the worst affected. SVD works on $X$
   directly and avoids the squaring.
3. **Truncated SVD is cheap.** You rarely want all $d$ components. Randomised and Lanczos algorithms
   compute just the top $k$ singular vectors in roughly $\mathcal{O}(npk)$ instead of the
   $\mathcal{O}(p^3)$ a full eigendecomposition costs.

> 💡 **This is why `sklearn.decomposition.PCA` computes an SVD internally rather than an
> eigendecomposition**, and why its default `svd_solver='auto'` switches to a randomised solver for
> large inputs. If asked "how is PCA actually implemented?", the answer is *"SVD of the centered data
> matrix, because forming the covariance is both expensive and numerically worse"* — and that is a
> noticeably better answer than reciting the eigendecomposition definition.

> ⚠️ **`TruncatedSVD` is not `PCA`.** It runs SVD **without centering**, which is deliberate — it's
> designed for sparse data (like TF-IDF matrices) where centering would destroy sparsity and blow up
> memory. On dense data, use `PCA`. Using `TruncatedSVD` on uncentered dense data reproduces exactly
> the §27.1 bug.

---

## 30. Scree plot and effective rank

> *"How many dimensions do you actually need?"* [slide 90, 55:54]
>
> - *"Plot eigenvalues in **descending** order"*
> - *"The **elbow** marks effective dimensionality"*
> - *"Components after elbow contribute mostly noise"*
> - *"Cumulative variance: keep $k$ components covering 95% of total variance"*

The slide's plot shows eigenvalues falling steeply, then flattening, with a dashed vertical line
marking the **elbow**.

> **Scree plot** — eigenvalues plotted in descending order against component index.
>
> *Where the name comes from:* "scree" is the loose rubble at the foot of a cliff. The plot looks like
> a cliff face (the important components) with a rubble slope at its base (the noise).

### 🧪 Worked example — both methods on the same spectrum, and where they disagree

Ten eigenvalues from a 10-feature dataset:

$$\lambda = [8.2,\ 3.1,\ 1.4,\ 0.35,\ 0.20,\ 0.15,\ 0.10,\ 0.05,\ 0.03,\ 0.02]$$

**Total variance** $= \mathrm{tr}(\Sigma) = 8.2+3.1+1.4+0.35+0.20+0.15+0.10+0.05+0.03+0.02 = \mathbf{13.6}$

| $k$ | $\lambda_k$ | Individual % | Cumulative | **Cumulative %** |
|---|---|---|---|---|
| 1 | 8.20 | 60.3% | 8.20 | 60.3% |
| 2 | 3.10 | 22.8% | 11.30 | 83.1% |
| 3 | 1.40 | 10.3% | 12.70 | **93.4%** |
| 4 | 0.35 | 2.6% | 13.05 | **96.0%** ← first ≥ 95% |
| 5 | 0.20 | 1.5% | 13.25 | 97.4% |
| 6 | 0.15 | 1.1% | 13.40 | 98.5% |
| 7 | 0.10 | 0.7% | 13.50 | 99.3% |
| 8 | 0.05 | 0.4% | 13.55 | 99.6% |
| 9 | 0.03 | 0.2% | 13.58 | 99.9% |
| 10 | 0.02 | 0.1% | 13.60 | 100.0% |

**Method 1 — the 95% rule:** the first $k$ with cumulative $\ge 95\%$ is $k = 4$ (96.0%).

**Method 2 — the elbow:** look at the *drops* between consecutive eigenvalues:

$$8.20 \to 3.10\ (-5.10), \quad 3.10 \to 1.40\ (-1.70), \quad 1.40 \to 0.35\ (\mathbf{-1.05}), \quad 0.35 \to 0.20\ (-0.15), \quad \ldots$$

The last big drop is $1.40 \to 0.35$; after that the differences collapse to $\le 0.15$ and the curve
is flat. **The elbow is at $k = 3$.**

**The two methods disagree — 3 versus 4 — and that's the point of doing both.** Component 4 carries
2.6% of the variance, which is above the noise floor (components 5–10 are 0.1–1.5%) but not by much.
It's a genuine judgement call, and the honest answer is *"3 or 4; I'd take 3 for interpretability and
4 if downstream accuracy justifies it — and I'd check by measuring both."*

```python
import numpy as np
lam = np.array([8.2, 3.1, 1.4, .35, .20, .15, .10, .05, .03, .02])
cum = np.cumsum(lam) / lam.sum()
print(np.argmax(cum >= 0.95) + 1)     # 4
print(-np.diff(lam))                  # [5.1 1.7 1.05 0.15 0.05 0.05 0.05 0.02 0.01]
```

### 30.1 Three ways to choose $k$, ranked by how much I'd trust them

| Method | Rule | Verdict |
|---|---|---|
| **Cumulative variance** | Smallest $k$ with $\sum_{i\le k}\lambda_i / \sum\lambda_i \ge 0.95$ | Objective and reproducible, but **95% is arbitrary** — nothing makes it the right threshold, and it will hand you $k = 200$ on a flat spectrum without complaining |
| **Elbow** | Where the curve visibly flattens | Matches the noise/signal intuition best, but **subjective** and sometimes there is no elbow |
| **Downstream CV** ✅ | Try $k \in \{2,5,10,20,50\}$ and pick by the metric you actually care about | **The only one that optimises what you want.** Costs a few fits. |

> 💡 **The third row is the answer to give, and the reason is worth stating.** The first two methods
> optimise *reconstruction* — how well you can rebuild $X$. But you don't care about rebuilding $X$;
> you care about predicting $y$. And those can diverge badly: §7.2's warning applies here in full — if
> the signal lives in a low-variance direction, the 95% rule will discard it while reporting that you
> kept 95% of the variance. **A high explained-variance ratio is not evidence that you kept the
> useful part.**
>
> Use the scree plot to *understand* your data — it's the direct test of the manifold hypothesis
> (§7.2) — and use cross-validation to *choose* $k$.

> ⚠️ **Standardise before PCA when features are on different scales.** PCA maximises variance, and
> variance is scale-dependent (§16's `height_m` vs `height_mm` example). A feature measured in
> millimetres will dominate PC1 purely by unit choice. Standardising first means you're running PCA on
> the **correlation** matrix rather than the covariance matrix — a different and usually more sensible
> analysis. The exception is when all features share a unit and the relative scales are meaningful
> (pixel intensities, returns in percent), where you should *not* standardise.

---

## 31. Connection to PCA (preview)

The lecture's closing content slide [slide 92, 56:33] — *"Everything we need for Section 2"*:

> - *"**First PC** = eigenvector with largest eigenvalue $\lambda_1$"*
> - *"**Projection**: $z_i = V_k^\top(x_i - \bar{x})$, where $V_k$ has first $k$ eigenvectors"*
> - *"**Objective**: find subspace that **preserves maximum variance**"*
> - *"**Equivalently**: minimizes reconstruction error $\|X - XV_kV_k^\top\|^2$"*
>
> *"PCA = eigendecomposition of covariance = truncated SVD of centered data. Section 2 covers the
> algorithms, kernels, and probabilistic extensions."*

### 31.1 Reading the projection formula

$$z_i = V_k^\top\left(x_i - \bar{x}\right)$$

| Symbol | Read it as | What it means | Shape |
|---|---|---|---|
| $x_i$ | "x sub i" | One original sample | $p \times 1$ |
| $\bar{x}$ | "x bar" | The mean of all samples — **the centering step** (§27.1) | $p \times 1$ |
| $V_k$ | "V sub k" | The first $k$ eigenvectors as columns | $p \times k$ |
| $V_k^\top(x_i - \bar x)$ | — | $k$ dot products: how far along each principal direction this sample lies | $k \times 1$ |
| $z_i$ | "z sub i" | The reduced representation | $k \times 1$ |

**In words: subtract the mean, then take the dot product with each of the top $k$ directions.** That's
all PCA does at inference time — it is a matrix multiply, so it is fast and it is trivially applicable
to new data (which is *not* true of t-SNE or UMAP, a point Part 3 will need).

### 31.2 The two objectives are the same objective

The third and fourth bullets sound like different goals, and the word **"Equivalently"** is doing real
work. It is worth seeing why, because the duality is one of the most-asked PCA facts.

**The identity.** Total variance splits exactly into "kept" and "lost":

$$\underbrace{\sum_{i=1}^{d}\lambda_i}_{\text{total (fixed)}} = \underbrace{\sum_{i=1}^{k}\lambda_i}_{\text{variance preserved}} + \underbrace{\sum_{i=k+1}^{d}\lambda_i}_{\text{reconstruction error}}$$

The left side doesn't depend on your choice of subspace — it's $\mathrm{tr}(\Sigma)$, a property of the
data. So **maximising the middle term is exactly the same as minimising the right-hand one.** They are
one optimisation problem seen from two ends.

> 💡 **This duality is why PCA has two apparently independent origin stories.** Pearson (1901) — the
> deck's citation — derived it as *"lines and planes of closest fit"*: minimise the perpendicular
> distance from the points to a subspace. Hotelling (1933) derived it as maximising variance. **Both
> papers describe the same computation**, arrived at from opposite directions thirty years apart, and
> the deck cites both on the eigendecomposition slide for exactly that reason. That's a good piece of
> history to have.

**And note the reconstruction formula.** $XV_kV_k^\top$ is "project down to $k$ dimensions, then lift
back to $p$": $V_k^\top$ maps $p \to k$, and $V_k$ maps $k \to p$. The round trip lands you in the
$k$-dimensional subspace, and $\|X - XV_kV_k^\top\|^2$ measures how far the original points are from
it. If $k = d$, then $V_kV_k^\top = I$ and the error is exactly zero — the lossless case from §29.2.

### 31.3 What Part 2 will add

> *"Section 2 covers the algorithms, kernels, and probabilistic extensions."*

| Topic | The gap it fills |
|---|---|
| **Algorithms** | How to compute PCA at scale — randomised SVD, incremental/online PCA for streaming data |
| **Kernels** | **Kernel PCA.** PCA is linear, so it finds *flat* subspaces. §7's manifolds are *curved*. Kernel PCA applies PCA in a high-dimensional feature space via the kernel trick, capturing nonlinear structure. |
| **Probabilistic extensions** | **Probabilistic PCA** and factor analysis — PCA as a generative latent-variable model, which gives you likelihoods, principled handling of missing data, and a proper way to choose $k$ |

---

## Putting it together

```
                    ┌────────────────────────────────────────────┐
                    │  ONE FACT DRIVES ACT I:                    │
                    │  volume grows EXPONENTIALLY in d,          │
                    │  and your sample size does not.            │
                    └──────────────────┬─────────────────────────┘
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
    §1 VOLUME               §2 DISTANCE                §3 EMPTY SPACE
    ball/cube → 0           mean ~ √d, spread FIXED    need k^d points
    52% (d=3) → 0.25%       ⇒ contrast ~ 0.59/√d       n=1000, d=20
    (d=10)                  ⇒ "nearest" is meaningless ⇒ 10⁻¹⁷ of the space
              └────────────────────────┼────────────────────────┘
                                       ▼
                          §4 kNN · kernels · overfitting
                             p≫n ⇒ ANY labelling is fittable
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  §7 THE MANIFOLD HYPOTHESIS          │
                    │  ambient dim ≠ intrinsic dim         │
                    │  100K pixels, ~50 real knobs         │
                    │  ⚠️ an ASSUMPTION, and it can fail   │
                    └──────────────────┬───────────────────┘
                    ┌──────────────────┴───────────────────┐
                    ▼                                      ▼
        ═══ SELECTION: keep columns ═══        ═══ EXTRACTION: build columns ═══
                    │                                      │
    how much does it know about the model?                 │
    ┌───────────────┼───────────────┐                      │
    ▼               ▼               ▼                      │
 §15 FILTER    §12 WRAPPER    §20 EMBEDDED                 │
 nothing       its score      IS the model                 │
 O(p), 0 fits  O(p·d) fits    ~1 fit                       │
    │               │               │                      │
 §16 variance   §13 forward    §21 Ridge  (circle)          │
 §17 Pearson    (XOR fails!)   §22 Lasso  (DIAMOND)         │
 §18 MI         §14 RFE        §23 ElasticNet               │
 §19 χ²/ANOVA   (202× cheaper) §25 paths                    │
    │                          §26 tree importance          │
    │                               │                      │
 ⚠️ ALL univariate            ⚠️ correlated features        │
 ⇒ blind to XOR                 break: Lasso picks one      │
 (MI too! §18.3)                arbitrarily → ElasticNet    │
                                                            ▼
                            ┌───────────────────────────────────────────┐
                            │  §27  Σ = (1/n)XᵀX      symmetric, PSD    │
                            │       vᵀΣv = variance along v             │
                            │              ▼                            │
                            │  §28  Σv = λv    eigenvector = direction  │
                            │                  eigenvalue  = variance   │
                            │              ▼                            │
                            │  §29  Σ = VΛVᵀ   rotate·scale·rotate back │
                            │       = V(S²/n)Vᵀ  ⇐ SVD of X             │
                            │              ▼                            │
                            │  §30  scree plot ⇒ how many do you need?  │
                            │              ▼                            │
                            │  §31  PCA: max variance ≡ min recon error │
                            │       Σλᵢ = Σ_{i≤k}λᵢ + Σ_{i>k}λᵢ         │
                            └──────────────────┬────────────────────────┘
                                               ▼
                                        PART 2 OF THIS MODULE
```

### Walking the diagram

**Act I is one fact, seen three ways.** Volume in $d$ dimensions grows like $c^d$; your sample size
grows like whatever your budget allows. That mismatch is the curse, and §1, §2 and §3 are three
measurements of it. The one worth being able to derive is §2, because the mechanism is genuinely
surprising: **distances don't become similar in absolute terms — their spread stays exactly constant
at 0.24 while their mean marches off as $0.41\sqrt{d}$.** Everything that depends on a distance being
informative degrades as $1/\sqrt{d}$, silently.

**Act II's escape is an assumption, and saying so is the honest part.** The manifold hypothesis claims
that real data occupies a low-dimensional surface inside its high-dimensional box, because it was
generated by a process with few degrees of freedom. If that's true, reduction is nearly free. If it's
false — genuinely independent features, or signal hiding in a low-variance direction — reduction
destroys information and PCA will not tell you it did. **§7.2 and §30.1 are the same warning delivered
twice, and it is the most useful thing in this file to remember.**

**The three selection families are one spectrum, indexed by knowledge.** Filters know nothing about
your model and cost nothing. Wrappers know its score and cost $\mathcal{O}(p\cdot d)$ model fits.
Embedded methods *are* the model and cost about one fit. **You pay for knowledge in compute**, and
embedded methods are the practical default because they sit at the good corner of that trade.

**And one pathology recurs in every family, which is why it's worth naming:** correlated features.
They make forward selection's rankings unstable (§14.1), make Lasso pick arbitrarily among equals
(§22.3), and make *both* tree importance measures under-report a pair that matters (§26.2). Elastic
Net's grouping effect (§23) is the cleanest fix, and its proof is three lines — L2 is strictly convex,
so it breaks the tie that L1 is indifferent to.

**Act III is the bridge, and it inverts the problem.** Selection asks *which columns?* — a discrete
question with $2^p$ answers. Extraction asks *which directions?* — a continuous question with a closed
form. The covariance matrix stores the shape of the data cloud; its eigenvectors are the cloud's
principal axes; its eigenvalues are the variance along each. Sort them, keep the top $k$, and you have
provably the best linear $k$-dimensional summary — best in **two** senses simultaneously (max variance,
min reconstruction error), because $\sum\lambda_i$ is fixed and the two objectives are the two halves
of one identity.

> 💡 **The thread that runs through the whole lecture:** *high dimensions are only a problem when they
> are real.* Distance concentration is driven by *effective* dimension, not column count (§2.1). The
> manifold hypothesis says effective dimension is usually far smaller (§7). Selection finds it by
> discarding columns; extraction finds it by discarding directions. **Both are answers to the same
> question — how many numbers does this data actually need? — and §30's scree plot is the instrument
> that measures it.**

---

## Interview prep — Amazon Applied Scientist

### Core questions

<details>
<summary><b>1. (Easy)</b> What is the curse of dimensionality? Give three concrete manifestations.</summary>

The collection of ways geometry and statistics stop behaving usefully as $p$ grows. Three, with
numbers — the numbers are what separate a good answer from a definition:

1. **Volume concentration.** A ball inscribed in a cube fills 52% of it in 3D and **0.25%** in 10D.
   Equivalently: at $d = 100$, **99.997%** of a cube's volume lies within 5% of its surface — there is
   no "middle" of a high-dimensional space.
2. **Distance concentration.** For uniform points, the mean pairwise distance grows like
   $0.41\sqrt{d}$ while the standard deviation stays **constant at 0.24**. So relative contrast falls
   as $0.59/\sqrt{d}$, and "nearest neighbour" stops meaning anything.
3. **Empty space.** Sampling at 10 bins per axis needs $10^d$ points. At $d = 20$ that's $10^{20}$;
   with $n = 1000$ you occupy $10^{-17}$ of the space.

**The unifying statement:** volume grows exponentially in $d$ and your sample size doesn't.
</details>

<details>
<summary><b>2. (Easy)</b> Feature selection or feature extraction — how do you choose?</summary>

**Selection keeps a subset of the original columns; extraction builds new ones as combinations.**

| Choose selection when | Choose extraction when |
|---|---|
| You need interpretability (or a regulator does) | Features are correlated / on a manifold |
| The true signal is sparse — a few columns matter | The signal is spread across many correlated columns |
| You need to *stop collecting* the dropped features | You have all the features anyway |

The sharpest version: **if the signal is a few of your columns, select; if it's a few directions
through your columns, extract.** Selection cannot average correlated noisy measurements; extraction
can.

**And add the practical point most people miss:** they compose. Filter out dead and near-constant
features first (§16 costs nothing), *then* run PCA on what survives. Also mention the operational
argument — fewer features means fewer pipelines to break — and the regulatory one, which can make the
decision for you before you see any data.
</details>

<details>
<summary><b>3. (Medium)</b> Explain filter, wrapper and embedded methods. When would you use each?</summary>

**Organise the answer around one axis: how much does the method know about the model you'll train?**

| | Filter | Wrapper | Embedded |
|---|---|---|---|
| Knows | **Nothing** — scores features vs $y$ | **Its score** — treats it as a black box | **Everything — it *is* the model** |
| Cost ($p{=}1000 \to 20$) | 0 fits, ~1 s | 19,810 fits (forward), 98 (RFE) | ~1 fit |
| Sees interactions | ❌ univariate | ✅ fully | ✅ within its hypothesis class |
| Examples | variance, Pearson, MI, χ² | forward/backward, RFE | Lasso, Elastic Net, tree importance |

**When:** filters as a **first pass on very wide data** (10,000 → 500 in a second, making everything
downstream cheaper); embedded as the **default** (most of the benefit for one fit); wrappers only when
you have compute to burn and the metric is non-negotiable.

**Two caveats that show experience:** (a) filters must be fit *inside* the CV loop or you leak — with
10,000 noise features you can manufacture above-chance accuracy from nothing; (b) wrappers overfit the
selection itself, so the winning subset's CV score is optimistically biased and you need nested CV to
report honestly.
</details>

<details>
<summary><b>4. (Medium)</b> Why does L1 produce sparse solutions and L2 doesn't? Give more than one argument.</summary>

**Three arguments, all the same fact. Lead with whichever fits the audience.**

**1 — Geometric (most memorable).** In constrained form, you're finding where the smallest RSS ellipse
touches the constraint region. The L2 region is a **circle**: smooth, so the touch point is at a random
place on it and the axes are measure-zero — no exact zeros. The L1 region is a **diamond with vertices
on the axes**: a vertex has a whole *fan* of supporting lines, so it has positive probability of being
the touch point — and at a vertex, all but one coordinate is exactly zero. In $\mathbb{R}^p$, touching
a $k$-face means selecting exactly $k$ features.

**2 — Gradient (fastest to state).** $\frac{d}{d\beta}\lambda|\beta| = \lambda\,\mathrm{sign}(\beta)$
— **constant magnitude regardless of how small $\beta$ is** — so a weight at 0.001 is pushed toward
zero just as hard as one at 10, and it arrives. L2's gradient $\lambda\beta$ shrinks as $\beta$ does,
so it asymptotes and never arrives.

**3 — Soft-threshold (most precise).** With orthonormal features:
Ridge gives $\hat\beta^{\text{OLS}}/(1+\lambda)$ — **divide**. Lasso gives
$\mathrm{sign}(\hat\beta^{\text{OLS}})(|\hat\beta^{\text{OLS}}| - \lambda/2)_+$ — **subtract and
clip**. Division approaches zero forever; subtraction arrives immediately.

**One line if you only get one:** *Ridge divides, Lasso subtracts-and-clips.*

**And the deeper framing if there's room:** L1 is the **convex relaxation** of $\|\beta\|_0$ (the count
of nonzeros), which is NP-hard. It turns a combinatorial selection problem into a convex optimisation
while keeping the zeros — which is why Tibshirani (1996) is a landmark rather than just another penalty.
</details>

<details>
<summary><b>5. (Medium)</b> Your feature has zero correlation with the target. Should you drop it?</summary>

**No — not on that evidence alone.** Pearson $r$ measures the strength of the **linear** relationship
only. $r = 0$ rules out a linear trend and nothing else.

**Demonstrate it in three lines** (this is what makes the answer land): let $X \in \{-2,-1,0,1,2\}$
uniformly and $Y = X^2$. Then $\bar X = 0$, $\bar Y = 2$, and

$$\mathrm{Cov} = \tfrac15[(-2)(2) + (-1)(-1) + 0 + (1)(-1) + (2)(2)] = \tfrac15[-4+1+0-1+4] = 0$$

**$r = 0$ exactly, and yet $Y$ is a deterministic function of $X$.** Mutual information on the same
data is **1.522 bits** — the maximum possible, since it equals $H(Y)$.

**Why it cancels:** the parabola is symmetric about $\bar X$, so every left-hand product is the exact
negative of its mirror. Any symmetric relationship gives $r = 0$: parabolas, absolute values, U-shapes,
XOR. U-shaped relationships are common in real data (dose–response, churn vs tenure, conversion vs
price).

**What to do instead:** **Spearman** first — rank correlation, one sort, catches every monotonic
relationship. **Mutual information** if you suspect non-monotonic structure, with the caveat that MI
estimates are biased upward on noise, so validate with a permutation test.
</details>

<details>
<summary><b>6. (Medium)</b> What does the manifold hypothesis claim, and when does it fail?</summary>

**The claim:** real high-dimensional data lies near a low-dimensional **manifold** — a surface that is
locally flat but globally curved — embedded in the high-dimensional space. **Ambient dimension** (how
many numbers you record) is much larger than **intrinsic dimension** (how many degrees of freedom the
data has).

**Why it's plausible — the generative argument.** A 100×100 face photo is a point in
$\mathbb{R}^{10{,}000}$, but face images are produced by a process with maybe 50 knobs: identity, pose,
lighting, expression. The set of faces is the image of a ~50-dimensional parameter space under a smooth
map — which is a manifold. **The recording format inflates the ambient dimension; it cannot inflate the
intrinsic one.**

**When it fails — this is the part that matters:**
1. **Genuinely independent features.** Twenty unrelated sensors have intrinsic dimension 20. There is
   nothing to find, and reduction destroys signal. Common in assembled tabular data — and it's why
   PCA disappoints on tabular problems where it dazzles on images.
2. **Signal in a low-variance direction.** PCA keeps high-variance directions on the assumption that
   variance = signal. **PCA is unsupervised — it never sees $y$** — so if your target depends on a
   subtle small-amplitude feature, PCA discards it and reports 95% variance explained. Silent, and
   the single most dangerous PCA failure. (LDA is the supervised alternative.)
3. **Discrete or disconnected structure** — categoricals don't form smooth manifolds.

**How to test rather than assume:** scree plot (a sharp elbow *is* evidence of low intrinsic
dimension), the relative-contrast test, or a direct intrinsic-dimension estimator.
</details>

<details>
<summary><b>7. (Medium)</b> Walk me through what a covariance matrix is and why PCA needs it.</summary>

$\Sigma = \frac1n X^\top X$ for **centered** $X$ — and the centering is essential, not a detail: without
it you get second moments, so a feature with mean 1000 and variance 1 shows up as 1,000,001 and PC1
points at the centroid instead of the spread.

- **Diagonal:** $\Sigma_{jj} = \mathrm{Var}(x_j)$
- **Off-diagonal:** $\Sigma_{jk} = \mathrm{Cov}(x_j, x_k)$
- **Symmetric and PSD**

**Why PCA needs it — the key identity.** For a unit direction $v$:

$$v^\top \Sigma v = \tfrac1n\|Xv\|^2 = \textbf{the variance of the data projected onto } v$$

(Two lines: $v^\top(\frac1n X^\top X)v = \frac1n(Xv)^\top(Xv)$, and $Xv$ *is* the projection.)

So "find the direction of maximum variance" is exactly $\max_{\|v\|=1} v^\top \Sigma v$, and the
spectral theorem says the answer is the **top eigenvector**, with the maximum value being $\lambda_1$.
**The eigenvalue is the variance along its eigenvector.**

**Symmetry is what makes this work:** it guarantees real eigenvalues and an orthonormal eigenbasis. And
because the components are orthogonal, variances add — $\sum_j\mathrm{Var}(x_j) = \mathrm{tr}(\Sigma) =
\sum_i\lambda_i$ — which is what makes "explained variance ratio" an honest fraction rather than
double-counting.

**Geometrically:** $\Sigma$ defines the data ellipse. Eigenvectors are its axes; $\sqrt{\lambda_i}$ are
the axis lengths. When features are correlated the ellipse is *tilted*, so its long axis is along
neither original feature — which is precisely why you need extraction rather than selection.
</details>

<details>
<summary><b>8. (Medium–hard)</b> How is PCA actually implemented, and why not just eigendecompose the covariance?</summary>

**PCA is computed as an SVD of the centered data matrix, not an eigendecomposition of $\Sigma$.**

**The connection, derived in one line.** If $X = USV^\top$ then

$$\Sigma = \tfrac1n X^\top X = \tfrac1n VS^\top U^\top U S V^\top = V\!\left(\tfrac{S^2}{n}\right)\!V^\top$$

using $U^\top U = I$. Matching against $\Sigma = V\Lambda V^\top$: **the $V$ from the SVD of $X$ is
exactly the eigenvector matrix of $\Sigma$, and $\lambda_i = s_i^2/n$.**

**Three reasons to prefer the SVD route:**
1. **You never form $\Sigma$.** At $p = 100{,}000$ it's $10^{10}$ entries — 40 GB in float32.
2. **Conditioning.** Forming $X^\top X$ **squares the condition number**, costing you roughly half
   your significant digits — and it hits the *small* eigenvalues hardest, which are exactly the ones
   near the noise floor where you're deciding what to keep.
3. **Truncation is cheap.** Randomised/Lanczos methods get the top $k$ in $\mathcal{O}(npk)$ instead of
   $\mathcal{O}(p^3)$.

**The tell you've used it:** mention `svd_solver='auto'` switching to a randomised solver on large
inputs, and that `TruncatedSVD` is *not* `PCA` because it deliberately skips centering (for sparse
data, where centering would destroy sparsity).
</details>

<details>
<summary><b>9. (Hard — combines two concepts)</b> You have 200 patients and 20,000 genes. Design a feature selection pipeline, and say what could go wrong.</summary>

$p \gg n$ by a factor of 100 — the hardest regime, and the first thing to say is what that implies:
**training accuracy carries literally zero information here.** With $p \ge n$, a linear model can
perfectly fit *any* labelling including one you generated by coin flip, so a perfect training score is
guaranteed and meaningless. Held-out validation is your only instrument, and with $n = 200$ that
estimate is itself noisy.

**The pipeline:**

1. **Variance threshold + QC.** Drop constant and near-constant genes. Costs nothing, doesn't touch
   $y$, so it **cannot leak** and is safe outside the CV loop. Typically kills a third of the columns.
2. **Univariate filter** to ~1,000. ANOVA F-test (continuous feature, categorical target — §19's
   table). **Inside the CV loop.**
3. **Elastic Net**, not Lasso. Two reasons: genes come in **correlated co-expression modules**, and
   Lasso picks one arbitrarily from each — non-reproducible, which is fatal when the deliverable is a
   gene list a biologist will spend a year on. And **Lasso can select at most $n = 200$ features by
   construction** (Zou & Hastie); if the biology involves more, Lasso literally cannot find them.
4. **Stability selection.** Bootstrap 100 times, keep genes selected in >80% of runs. With $n = 200$
   the single-fit answer is not trustworthy on its own.
5. **Nested CV** for the honest score: inner loop selects and tunes, outer loop evaluates.

**What goes wrong:**
- **Leakage** — filtering on the full dataset before CV. With 20,000 noise columns you can manufacture
  above-chance accuracy from nothing. `Pipeline` makes it impossible; use it.
- **Batch effects** — if cases were sequenced in one batch and controls in another, the top "genes"
  are batch markers. Check whether your selected features predict *batch*. This is usually the real
  reason a genomics result doesn't replicate.
- **Selection instability** — see step 4.
- **Multiple testing** — 20,000 univariate tests at $\alpha = 0.05$ gives 1,000 false positives by
  chance. Use FDR (Benjamini–Hochberg), not raw p-values.

**Why not PCA?** It would work statistically, but a biologist cannot act on "principal component 3."
The deliverable here is a *list of genes to investigate*, so §9's interpretability row decides it.
</details>

<details>
<summary><b>10. (Hard — combines two concepts)</b> Your kNN model works in a 10-feature prototype and fails at 500 features. Diagnose it, and be specific about what you'd measure.</summary>

**Distance concentration — and the diagnosis should be measured, not assumed.**

**The mechanism.** kNN's premise is that the $k$ nearest points are *meaningfully* nearer. The mean
pairwise distance grows as $0.41\sqrt{d}$ while the spread stays constant at 0.24, so relative contrast
falls as $0.59/\sqrt{d}$:

- $d = 10$: $0.59/3.16 = \mathbf{0.187}$ — comfortably above the 0.1 reliability threshold ✅
- $d = 500$: $0.59/22.4 = \mathbf{0.026}$ — a factor of 4 past it ❌

**What I'd measure first, before changing anything:** compute
$\frac{d_{\max}-d_{\min}}{d_{\min}}$ on the actual data at both dimensionalities. This matters because
**correlated features don't count as separate dimensions** — if the 500 columns really live on a
6-dimensional manifold, distances haven't concentrated and my diagnosis is wrong. Measuring instead of
assuming is the whole point of the deck's practical test.

**If concentration is confirmed, in order of what I'd try:**
1. **Reduce dimension first, then kNN.** PCA to 10–20 components. Directly restores contrast.
2. **Try the $L_1$ metric.** Aggarwal et al. (2001) showed Manhattan concentrates measurably more
   slowly than Euclidean. One-line change, sometimes enough.
3. **Learn a metric** rather than assuming Euclidean — a supervised embedding, or a model that learns
   its own similarity.
4. **Use a different model class.** Trees don't use distances at all; a gradient-boosted model on 500
   features is often the pragmatic answer.

**What I would *not* do: increase $k$.** That reduces the variance of an estimate of the wrong
quantity. The neighbours aren't noisy — they aren't neighbours.

**And the meta-point worth making:** it "worked" at 10 features and "fails" at 500 with no error and no
warning. This class of failure is silent, which is why §2's practical test belongs in your monitoring,
not just your analysis.
</details>

<details>
<summary><b>11. (Hard — combines two concepts)</b> A stakeholder says "PCA kept 95% of the variance, so we lost almost nothing." Respond.</summary>

**That statement is true and potentially irrelevant, and the gap between those is worth explaining
carefully rather than dismissively.**

**What it does mean.** $\sum\lambda_i$ is fixed, so keeping 95% of the variance means the
reconstruction error $\sum_{i>k}\lambda_i$ is 5% of the total. If the goal is **compression** — rebuild
$X$ from fewer numbers — this is a real and sufficient guarantee.

**Why it may not mean what they think.** **PCA is unsupervised. It never looks at $y$.** It keeps
high-variance directions, on the assumption that variance = signal. But:

- **The target may depend on a low-variance direction.** A sensor drift of small amplitude that
  predicts failure sits in a direction PCA discards — and PCA will report 95% variance explained while
  having deleted the only thing you cared about.
- **95% is an arbitrary threshold.** Nothing makes it correct. On a flat spectrum it will return
  $k = 200$ without complaining.
- **Variance is scale-dependent.** If features weren't standardised, "95% of the variance" may mean
  "95% of the variance of whichever feature happens to be measured in the smallest units."

**What I'd actually do:** stop optimising reconstruction and measure the thing we care about. Sweep
$k \in \{2, 5, 10, 20, 50\}$ and pick by **downstream cross-validated performance**. Use the scree plot
to *understand* the data — it's the direct test of the manifold hypothesis — and CV to *choose* $k$.

**And offer the supervised alternative:** if the goal is prediction rather than compression, **LDA**
(or a supervised embedding) finds directions that separate the classes rather than directions of
maximum spread. Different objective, and it's the right one here.

> This is the single most useful pushback in the module, and it lands well precisely because the
> stakeholder's statement is *correct* — you're not contradicting them, you're pointing out that
> they've measured a different thing from the one they need.
</details>

<details>
<summary><b>12. (Hard)</b> Derive why Elastic Net selects correlated features as a group.</summary>

**Setup.** Let $x_1$ and $x_2$ be perfectly correlated and jointly predictive, so any split with
$\beta_1 + \beta_2 = c$ (both positive) fits the data identically.

**Step 1 — Lasso is exactly indifferent.** For all such splits,

$$\|\beta\|_1 = \beta_1 + \beta_2 = c$$

Identical penalty, identical fit ⇒ the optimum is a whole line segment, not a point. Coordinate descent
lands on a vertex, so you get $(c, 0)$ or $(0, c)$ **by numerical accident** — and the choice flips
across bootstrap samples.

**Step 2 — L2 strictly prefers the even split.** Minimise $\beta_1^2 + \beta_2^2$ subject to
$\beta_1 + \beta_2 = c$. Lagrange: $2\beta_1 = \mu$, $2\beta_2 = \mu$, hence

$$\beta_1 = \beta_2 = c/2$$

**Step 3 — compare the penalties.**

| Split | L1 | L2 |
|---|---|---|
| $(c, 0)$ | $c$ | $c^2$ |
| $(c/2, c/2)$ | $c$ | $c^2/2$ ✅ **half** |

**L1 cannot distinguish them. L2 prefers sharing, by a factor of two.** So adding *any* L2 breaks the
tie toward keeping correlated features together — and the solution becomes unique.

**The general theorem behind it:** L2 is **strictly convex**, so it has a unique minimiser on any
convex set. L1 is merely convex, so it can be flat along a face — which is exactly where Lasso's
non-uniqueness lives. Adding a strictly convex term to a convex one makes the whole objective strictly
convex, hence uniquely solved. **Sparsity comes from L1's corners; stability comes from L2's strict
convexity; Elastic Net gets both because it has both.**

**Add the second motivation** if there's room: **Lasso can select at most $n$ features**, a hard
structural limit. Elastic Net has no such cap — which is decisive in $p \gg n$ settings.
</details>

### Depth probes

| Your good answer | The probe | What they want |
|---|---|---|
| "Distances concentrate in high dimensions" | *"By how much? Derive it."* | Mean $\sim 0.41\sqrt d$, **std constant at 0.24**, so contrast $\sim 0.59/\sqrt d$. The constancy of the spread is the surprising half. |
| "The curse means you need more data" | *"How much more?"* | $\mathcal{O}(k^d)$. At 10 bins/axis and $d{=}20$, $10^{20}$ points. And inverted: $n{=}1000$ densely supports about **3** features. |
| "L1 gives sparsity" | *"Why is $\beta=0$ special rather than just reachable?"* | $\lvert\beta\rvert$ is non-differentiable at 0; its subgradient there is the whole interval $[-\lambda,\lambda]$, so a **range** of data configurations map to exactly zero. |
| "Use mutual information for nonlinear relationships" | *"Will MI find XOR?"* | **Univariate MI on each feature is zero for XOR** — MI removes the *linearity* assumption, not the *univariate* one. A filter is still a filter. |
| "PCA finds directions of maximum variance" | *"Prove that's the top eigenvector."* | Variance along unit $v$ is $v^\top\Sigma v = \frac1n\|Xv\|^2$; maximise over $\|v\|=1$; spectral theorem gives $v_1$ with value $\lambda_1$. |
| "Keep components covering 95% of variance" | *"Why 95? And what if the signal is low-variance?"* | It's arbitrary, and PCA is unsupervised so it can discard the predictive direction while reporting 95%. Choose $k$ by downstream CV. |
| "Gini importance ranks features" | *"What happens if I add a column of random UUIDs?"* | It often ranks near the **top** — high cardinality means many split points, so spurious impurity reduction. Use permutation importance on held-out data. |
| "Filter first, then model" | *"Where exactly does the filter go?"* | **Inside** the CV loop. Filtering on the full dataset leaks $y$ and manufactures above-chance accuracy from pure noise. |
| "Elastic Net handles correlated groups" | *"What else does it fix that Lasso can't?"* | Lasso selects at most $n$ features — a hard cap. Decisive when $p \gg n$ and the truth is denser than $n$. |

### Whiteboard-ready derivations

**D1 — Distance concentration.**
```
X, Y ~ Uniform([0,1]^d),  D² = Σⱼ (Xⱼ − Yⱼ)²        ← sum of d iid terms

one term:  E[(X−Y)²] = 1/3 − 2(1/4) + 1/3 = 1/6
           Z = X−Y is triangular on [−1,1], f(z)=1−|z|
           E[Z⁴] = 2∫₀¹ z⁴(1−z)dz = 2(1/5 − 1/6) = 1/15
           Var(Z²) = 1/15 − 1/36 = 7/180

sum:       E[D²] = d/6            Var(D²) = 7d/180
delta:     E[D] ≈ √(d/6) = 0.408√d
           std(D) ≈ std(D²)/(2E[D]) = 0.197√d / (2·0.408√d) = 0.2415   ← √d CANCELS

⇒ relative spread = 0.2415 / (0.408√d) = 0.592/√d  →  0
```

**D2 — Why the eigenvalue is the variance (and hence why PCA is an eigenproblem).**
```
variance along a unit direction v:
    vᵀΣv = vᵀ(1/n · XᵀX)v = (1/n)(Xv)ᵀ(Xv) = (1/n)‖Xv‖²        ← Xv IS the projection
                                                                 ≥ 0  ⇒ Σ is PSD

if v is a unit eigenvector:   vᵀΣv = vᵀ(λv) = λ‖v‖² = λ

⇒ maximise variance  ≡  max_{‖v‖=1} vᵀΣv  ≡  take the top eigenvector, value λ₁

orthogonality ⇒ variances add:   Σⱼ Var(xⱼ) = tr(Σ) = Σᵢ λᵢ
⇒ "explained variance ratio" λᵢ/Σλ is an honest fraction
⇒ max variance  ≡  min reconstruction error   (the two halves of one fixed total)
```

**D3 — Elastic Net's grouping effect.**
```
x₁ ≡ x₂ perfectly correlated;  any β₁+β₂ = c fits identically

L1:  ‖β‖₁ = β₁+β₂ = c   for EVERY split  ⇒ Lasso indifferent ⇒ arbitrary vertex

L2:  min β₁²+β₂²  s.t. β₁+β₂ = c
     Lagrange: 2β₁ = μ = 2β₂  ⇒  β₁ = β₂ = c/2

     (c, 0)      → L2 penalty c²
     (c/2, c/2)  → L2 penalty c²/2      ← strictly smaller

⇒ any L2 breaks the tie toward SHARING; L1 supplies the zeros
⇒ general reason: L2 strictly convex ⇒ unique minimiser; L1 only convex ⇒ flat faces
```

### Applied scenario — feature selection for delivery-time prediction

**The problem.** Predict the delivery time of a package at order placement, to show customers an
accurate promise date. The feature store exposes ~4,000 candidate features: origin/destination
geography, carrier, historical lane performance, package attributes, weather, calendar, warehouse
load, and a large block of one-hot encoded categoricals. Millions of predictions per day, latency
budget in the low tens of milliseconds.

**Framing.** This is a regression problem where **feature count is a production cost, not just a
statistical one**. Every feature is a lookup at serving time, a pipeline that can fail, and a schema
someone can change. So the objective isn't "best accuracy" — it's *"the smallest feature set within
$\epsilon$ of the best accuracy"*, which is exactly what §9 calls the sparse-signal regime and points
at **selection** rather than extraction. Extraction is additionally ruled out because a PCA component
over 4,000 features requires computing all 4,000 at serving time — **it reduces model width, not
data-collection cost**, which is a distinction people miss.

**Data.** Mixed types, so §19's table decides the filters: `f_regression` for continuous features,
and one-hot blocks handled as groups rather than individually. Two structural issues dominate:

- **Heavy correlation.** `distance_km`, `distance_miles`, `estimated_transit_days` and
  `historical_lane_median` are near-duplicates. This is the §22.3/§26.2 pathology and it will
  destabilise both Lasso and importance ranking if ignored.
- **Temporal structure.** Delivery performance drifts with season, carrier contracts and network
  changes. **Random k-fold CV would leak the future into the past** — split by time.

**Pipeline.**

1. **Variance threshold + QC** — kills constant columns and rare one-hot levels. Doesn't touch $y$,
   cannot leak, safe outside the CV loop. Expect ~4,000 → ~2,500.
2. **Correlation pruning** at $|r| > 0.95$, keeping the cheapest-to-serve member of each cluster —
   a genuinely useful tiebreak that pure statistics won't give you. ~2,500 → ~1,200.
3. **Univariate filter** to ~300 (`f_regression`), **inside** the CV loop.
4. **Embedded selection** with gradient-boosted trees + permutation importance on a held-out temporal
   slice. Trees because the relationships are non-monotonic (weather effects, weekday effects) and
   interaction-heavy (carrier × lane × season) — precisely where §13.1's greedy and §17's linear
   methods fail. **Permutation, not Gini**, because the feature set mixes high-cardinality continuous
   features with binary flags and Gini would systematically over-rate the former (§26.1).
5. **Permute correlated features as a group**, not individually — otherwise near-duplicates mask each
   other and both look unimportant.
6. **Sweep the feature count** (300 → 150 → 75 → 40 → 20) and plot accuracy against count. Ship the
   knee, not the maximum.

**Metric.** Not RMSE. The business cost is **asymmetric** — a package arriving early is mildly good, a
package arriving after the promised date is a customer-experience failure and a support contact. Use a
quantile loss (predict the 80th percentile, not the mean) and report **on-time rate at the promised
date** alongside it. Track it **per lane and per carrier**, because an aggregate improvement that
degrades a specific corridor is not shippable.

**Failure modes.**
- **Leakage via CV design** — random folds instead of temporal ones. The most likely way this project
  produces an unreproducible result.
- **Leakage via features** — anything computed after the order is placed (actual carrier scan times)
  is not available at prediction time. Audit every feature's availability timestamp.
- **Selection instability** across retrains — the feature list changing every week is an operational
  nightmare. Bootstrap the selection and prefer consistently-chosen features.
- **Drift** — a carrier changes its network and the selected features stop being predictive. Monitor
  feature distributions, not only the metric.
- **The correlated-group trap** at every stage, per §14.1, §22.3 and §26.2.

**What I'd ship.** V1: the pipeline above, targeting ~40 features, temporal CV, quantile loss,
per-lane monitoring, and the feature list version-controlled and diffed on every retrain. Explicitly
*not* in V1: PCA (doesn't reduce serving cost) and wrapper search (the compute is better spent on the
model than on the selection).

### Leadership Principles tie-in

**Dive Deep.** §2 is the model. The shallow answer to "why did kNN stop working?" is "curse of
dimensionality." The deep answer measures it: mean distance grows as $0.41\sqrt d$ while the spread
stays fixed at 0.24, so contrast falls as $0.59/\sqrt d$ — 0.187 at $d{=}10$, 0.026 at $d{=}500$, past
the reliability threshold. *"Rather than assume, I computed the relative contrast on our actual data
and found it was 0.03 — which told me the problem was the metric's resolution, not the model's
hyperparameters, and pointed straight at dimensionality reduction."*

**Insist on the Highest Standards.** §15.3's leakage trap. Fitting a filter outside the CV loop
produces a number that looks like a result and is manufactured from noise — and nothing errors. Being
the person who insists selection goes inside the `Pipeline`, and who demonstrates the failure by
running it on random data to show above-chance accuracy appearing from nothing, is exactly this LP.

**Are Right, A Lot** fits question 11's pushback: hearing "we kept 95% of the variance" and knowing to
ask *"variance of what, and does the target depend on it?"* — because PCA is unsupervised and can
discard the only predictive direction while reporting a reassuring number.

**Frugality** fits §15.2 and the applied scenario: a filter costs one second and makes everything
downstream 20× cheaper; forward selection costs 5.5 hours for the same problem RFE does in 98 seconds.
Knowing the cost table well enough to spend compute where it actually buys accuracy is this LP in its
technical form.

> 🎯 **stretch — nice to know, not expected for an intern:** proving the $\mathcal{O}(k^d)$ sample
> complexity bound rigorously; the LARS algorithm's piecewise-linearity proof; the exact statement of
> Zou & Hastie's "at most $n$ features" result; stability selection's error control; distance
> correlation; intrinsic-dimension estimators; SHAP values. Knowing these exist and what they claim is
> enough.

---

## Glossary

| Term | Definition |
|---|---|
| **ANOVA F-test** | Filter for a continuous feature against a categorical target. $F = \frac{\text{between-group variance}}{\text{within-group variance}}$. ⚠️ Detects *mean* differences only. §19.2 |
| **Ambient dimension** | The number of columns you record. Contrast with intrinsic dimension. §7 |
| **Chi-square test (χ²)** | Filter for a categorical feature against a categorical target. $\chi^2 = \sum(O-E)^2/E$ with $E_{ij} = r_ic_j/n$. ⚠️ sklearn's `chi2` is a different thing. §19.1 |
| **Coefficient of variation** | $\sigma/\mu$ — scale-invariant spread. The fix for variance threshold's unit-dependence. ⚠️ Undefined at $\mu = 0$. §16 |
| **Convex relaxation** | Replacing an intractable discrete constraint ($\|\beta\|_0 \le k$, NP-hard) with a tractable convex one ($\|\beta\|_1 \le t$) that keeps the property you wanted. **The reason Lasso is a landmark.** §20.1 |
| **Correlation ($r$)** | $\mathrm{Cov}(X,Y)/\sigma_X\sigma_Y$ — covariance with the units divided out, in $[-1,1]$. **Measures linear relationship only.** §17, Prereq 2 |
| **Covariance** | $\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$ — how two variables move together. Carries the units of both, so not comparable across features. Prereq 2 |
| **Covariance matrix ($\Sigma$)** | $\frac1n X^\top X$ for **centered** $X$. Variances on the diagonal, covariances off it. Symmetric and PSD. **The input to PCA.** §27 |
| **Curse of dimensionality** | The collection of ways geometry, statistics and computation fail as $d$ grows. Coined by Bellman (1961). §1–§4 |
| **Distance concentration** | $(d_{\max}-d_{\min})/d_{\min} \to 0$. The mean distance grows as $0.41\sqrt d$ while the spread stays **constant** at 0.24. §2 |
| **Effective rank** | How many components you actually need — read off the scree plot's elbow. §30 |
| **Eigendecomposition** | $\Sigma\mathbf{v} = \lambda\mathbf{v}$. Eigenvector = direction; **eigenvalue = variance along it.** §28 |
| **Elastic Net** | $\lambda_1\|\beta\|_1 + \lambda_2\|\beta\|_2^2$. Gets L1's sparsity and L2's stability; **selects correlated features as groups**; no $n$-feature cap. §23 |
| **Elbow** | The kink in a scree plot where eigenvalues stop falling steeply. Marks effective dimensionality. §30 |
| **Embedded method** | Selection happens **inside** model training via a penalty. ~1 model fit. Lasso, Elastic Net, tree importance. **The practical default.** §20 |
| **Empty space phenomenon** | Filling a $d$-dimensional grid at $k$ bins/axis needs $k^d$ points. $n{=}1000$, $d{=}20$ occupies $10^{-17}$ of the space. §3 |
| **Entropy $H(X)$** | $-\sum p(x)\log_2 p(x)$ — bits of uncertainty. Fair coin = 1 bit. Prereq 4 |
| **Feature extraction** | Build **new** features as combinations of the old. PCA, autoencoders. Continuous optimisation; lower interpretability. §9 |
| **Feature selection** | Keep a **subset** of the original features. Combinatorial ($2^p$); preserves meaning. §9 |
| **Filter method** | Score each feature **independently** against $y$, then threshold. $\mathcal{O}(p)$, no model training, **blind to interactions**. §15 |
| **Forward selection** | Greedily add the feature that most improves the score. $\mathcal{O}(p\cdot d)$ fits. ⚠️ **Misses XOR-type interactions.** §13 |
| **Gamma function $\Gamma$** | The factorial generalised to non-integers. Grows faster than exponentially — which is why $V_d \to 0$. §1 |
| **Gini importance** | Total impurity decrease from all splits on a feature. **Free** from training. ⚠️ **Biased toward high-cardinality features** — random UUIDs often rank near the top. §26.1 |
| **Intrinsic dimension** | The number of degrees of freedom the data actually has. §7 |
| **L1 norm** | $\sum_j\lvert\beta_j\rvert$. Its unit ball is a **diamond with corners on the axes** ⇒ sparsity. Prereq 6, §22 |
| **L2 norm (squared)** | $\sum_j\beta_j^2$. Its unit ball is a **smooth sphere** ⇒ shrinkage without zeros. **Strictly convex** ⇒ unique solutions. Prereq 6, §21 |
| **Lasso** | L1-penalised regression. Sparse; automatic feature selection. ⚠️ Picks arbitrarily among correlated features; **selects at most $n$**. Tibshirani (1996). §22 |
| **Manifold** | A surface that is locally flat but globally curved, embedded in a higher-dimensional space. §7 |
| **Manifold hypothesis** | Real data lies near a low-dimensional manifold in high-D space. **An assumption, not a theorem** — and it fails for independent features and for low-variance signal. §7 |
| **Multicollinearity** | Features that are near-linear combinations of each other. Makes $X^\top X$ near-singular and coefficients wildly unstable. Ridge fixes it by adding $\lambda I$. §21.2 |
| **Mutual information** | $I(X;Y) = H(X) - H(X\mid Y)$. Captures **any** dependence; $I = 0 \iff$ independence. ⚠️ Estimates are biased upward on noise; still **univariate** in a filter. §18 |
| **Nested cross-validation** | Inner loop selects/tunes, outer loop evaluates. Required for an honest score when selection is part of the pipeline. §12 |
| **Permutation importance** | Drop in held-out performance when a feature is shuffled. **Unbiased**, captures interactions, costs $\mathcal{O}(p)$ predictions. ⚠️ Use held-out data; permute correlated features as a group. §26.2 |
| **Positive semi-definite (PSD)** | $v^\top\Sigma v \ge 0$ for all $v$ — necessarily true here, because $v^\top\Sigma v$ *is* the variance along $v$. Implies all $\lambda_i \ge 0$. §27.2 |
| **Regularization path** | How coefficients evolve as $\lambda$ grows. **The order features die is an importance ranking.** Computed cheaply by LARS. §25 |
| **RFE** | Train, rank by model internals, remove the bottom $k$, repeat. **202× cheaper than forward selection** at $p{=}1000{\to}20$. A wrapper loop around an embedded ranking. §14 |
| **Ridge** | L2-penalised regression. $\hat\beta = (X^\top X + \lambda I)^{-1}X^\top y$ — always invertible. Shrinks proportionally, **never zeros**. Hoerl & Kennard (1970). §21 |
| **Scree plot** | Eigenvalues in descending order. Named for the rubble at a cliff's base. §30 |
| **Soft-threshold** | $\mathrm{sign}(\hat\beta)(\lvert\hat\beta\rvert - \lambda/2)_+$ — Lasso's orthonormal solution. **Subtract and clip**, versus Ridge's **divide**. §21.3 |
| **Spearman correlation** | Pearson on the ranks. Catches **any monotonic** relationship for the cost of a sort. **The cheap upgrade over Pearson.** §17.3 |
| **Spectral decomposition** | $\Sigma = V\Lambda V^\top$ — rotate, scale, rotate back. §29 |
| **Spectral theorem** | A real symmetric matrix has real eigenvalues and an orthonormal eigenbasis. **What makes PCA well-posed.** §27.2 |
| **SVD** | $X = USV^\top$. Gives $\Sigma = V(S^2/n)V^\top$, so the SVD's $V$ **is** $\Sigma$'s eigenvectors and $\lambda_i = s_i^2/n$. How PCA is actually computed. §29.3 |
| **Variance threshold** | Drop near-constant features. The only filter that never looks at $y$, so it **cannot leak**. ⚠️ Not scale-invariant. §16 |
| **Volume concentration** | A ball inscribed in its cube fills 52% at $d{=}3$ and 0.25% at $d{=}10$. At $d{=}100$, 99.997% of a cube is within 5% of its surface. §1 |
| **Wrapper method** | Search feature subsets by **held-out model score**, treating the model as a black box. Sees interactions; costs $\mathcal{O}(p\cdot d)$ fits. ⚠️ Overfits the selection. §12 |

---

## Check yourself

Work these before reading the answers. Questions 9–12 require combining two concepts.

1. Compute the fraction of a $d$-dimensional cube filled by its inscribed ball at $d = 4$. ($\Gamma(3) = 2$.)
2. At $d = 200$, what is the approximate relative spread of pairwise distances for uniform data? Is kNN reliable there by the deck's own test?
3. You have 5,000 samples. At 10 bins per axis, how many features can you densely cover?
4. Which filter would you use for: (a) continuous feature, continuous target; (b) categorical feature, categorical target; (c) continuous feature, categorical target; (d) you have no idea what shape the relationship is?
5. `feature_A` has variance 0.008 and `feature_B` has variance 4,500. Should you drop A with `VarianceThreshold(0.01)`? What would you check first?
6. Forward selection from $p = 500$ down to $d = 15$: how many model fits? RFE removing 25 per round: how many?
7. Given eigenvalues $[6.0, 2.5, 0.8, 0.4, 0.2, 0.1]$, compute the cumulative variance ratios and pick $k$ by the 95% rule. Where is the elbow?
8. $X = \begin{bmatrix} 3 & 6 \\ -1 & -2 \\ -2 & -4 \end{bmatrix}$ is centered. Compute $\Sigma$, its eigenvalues, and say what the data's intrinsic dimension is.
9. **(Combines two)** You run Lasso twice on bootstrap resamples of the same data and get two disjoint feature sets of the same size, with nearly identical CV scores. Explain, and say what you'd do.
10. **(Combines two)** A colleague reports 88% CV accuracy on a dataset with $n = 80$ and $p = 12{,}000$. They selected the top 50 features by correlation with $y$, then cross-validated a logistic regression on those 50. Diagnose, and estimate what the honest number is.
11. **(Combines two)** You apply PCA to 500 sensor features and keep 20 components covering 97% of the variance. Downstream accuracy *drops* versus using all 500. Give two explanations and say how you'd distinguish them.
12. **(Combines two)** Your random forest ranks `transaction_id` as the third most important feature for fraud detection. What happened, what does it tell you about the rest of the ranking, and what would you do?

<details>
<summary><b>Answers</b></summary>

**1.** $\dfrac{\pi^{d/2}}{\Gamma(d/2+1)2^d}$ at $d = 4$:

$$\frac{\pi^2}{\Gamma(3)\cdot 2^4} = \frac{9.8696}{2 \times 16} = \frac{9.8696}{32} = \mathbf{0.3084} = 30.8\%$$

Between $d{=}3$'s 52.4% and $d{=}5$'s 16.4% ✓ — the collapse is already well under way by four dimensions.

**2.** Relative spread $\approx \dfrac{0.592}{\sqrt{200}} = \dfrac{0.592}{14.14} = \mathbf{0.042}$.

That is **below the deck's 0.1 threshold**, so distances are unreliable and kNN is not trustworthy at
$d = 200$ **for independent uniform features**.

**The answer isn't complete without the caveat**, though: if those 200 features are heavily correlated
— living on, say, a 5-dimensional manifold — the *effective* dimension is 5, contrast is around
$0.592/\sqrt5 = 0.26$, and kNN is fine. **Measure the contrast on the real data rather than computing
it from $p$.** That's exactly why the deck frames it as a practical test.

**3.** $10^d \le 5000 \implies d \le \log_{10}5000 = 3.7$, so **3 features** (4 would need 10,000
points).

Five times the data bought you *zero* extra features versus §3's $n = 1000$ case, which also gave 3.
**That's the exponential in action:** each additional feature needs 10× more data, so a 5× increase in
$n$ doesn't even buy one.

**4.** From §19's table:
- (a) continuous / continuous → **Pearson $r$** (`f_regression`) — or **Spearman** if you suspect a monotone curve
- (b) categorical / categorical → **Chi-square**
- (c) continuous / categorical → **ANOVA F-test** (`f_classif`)
- (d) unknown shape → **Mutual information** — slower and noisier, but assumption-free

**5.** **No — not yet.** The two variances differ by a factor of ~560,000, which is far more likely to
be a **units** difference than an information difference (§16's `height_m` vs `height_mm`).

**What to check first:** the coefficient of variation, $\sigma/\mu$, which is scale-invariant. If
`feature_A` has $\mu = 0.09$ then $\mathrm{CV} = \sqrt{0.008}/0.09 = 0.089/0.09 \approx 0.99$ —
enormous relative spread, and dropping it would be a serious mistake. Or standardise everything first,
at which point all variances are 1 and only genuinely constant columns remain distinguishable.

**Use variance threshold as a data-quality check** (find columns your ETL filled with a default), not
as a feature-selection method.

**6.** **Forward selection:** $\sum_{i=0}^{14}(500 - i) = 15\times500 - \frac{14\times15}{2} = 7500 - 105 = \mathbf{7{,}395}$ fits.

**RFE** removing 25 per round from 500 to 15: $\lceil (500-15)/25 \rceil = \lceil 19.4 \rceil = \mathbf{20}$ fits.

**370× cheaper.** ⚠️ At 25 removed per round the granularity is coarse, and correlated pairs can both
be dropped in one cut when either alone would have been kept (§14.1).

**7.** Total $= 6.0 + 2.5 + 0.8 + 0.4 + 0.2 + 0.1 = 10.0$ — conveniently.

| $k$ | $\lambda$ | Cumulative | % |
|---|---|---|---|
| 1 | 6.0 | 6.0 | 60% |
| 2 | 2.5 | 8.5 | 85% |
| 3 | 0.8 | 9.3 | 93% |
| 4 | 0.4 | 9.7 | **97%** ← first ≥95% |
| 5 | 0.2 | 9.9 | 99% |
| 6 | 0.1 | 10.0 | 100% |

**95% rule: $k = 4$.**

**Elbow:** drops are $-3.5, -1.7, -0.4, -0.2, -0.1$. The last large drop is $2.5 \to 0.8$, after which
everything is $\le 0.4$. **Elbow at $k = 2$**, arguably 3.

The two methods disagree (2–3 vs 4), which is normal and is why §30.1 says to decide by downstream CV.

**8.** Rows: $(3,6)$, $(-1,-2)$, $(-2,-4)$. Centered ✓ ($3-1-2 = 0$; $6-2-4 = 0$).

$$(X^\top X)_{11} = 9 + 1 + 4 = 14 \qquad (X^\top X)_{12} = 18 + 2 + 8 = 28 \qquad (X^\top X)_{22} = 36 + 4 + 16 = 56$$

$$\Sigma = \frac13\begin{bmatrix}14 & 28\\ 28 & 56\end{bmatrix} = \begin{bmatrix}4.667 & 9.333\\ 9.333 & 18.667\end{bmatrix}$$

$\mathrm{tr} = 23.333$; $\det = (4.667)(18.667) - 9.333^2 = 87.11 - 87.11 = \mathbf{0}$.

$$\lambda_1 = 23.333, \qquad \lambda_2 = 0$$

**Intrinsic dimension = 1.** Every row is a multiple of $(1, 2)$ — $3\times$, $-1\times$, $-2\times$ —
so despite having two columns the data lies on a line. The zero eigenvalue detects it exactly, and PC1
would capture 100% of the variance with zero reconstruction error.

**9.** **Correlated features, and Lasso's indifference among them** (§22.3).

If the features fall into correlated groups, then for any group all splits of the coefficient satisfy
$\|\beta\|_1 = c$ and fit identically — **the optimum is a line segment, not a point**. Coordinate
descent lands on a vertex chosen by numerical accident, and a different bootstrap sample flips the
choice. Equal CV scores are the tell: the two sets are *genuinely equally good*, which is exactly what
the theory predicts.

**What I'd do:**
1. **Switch to Elastic Net.** Its L2 term is strictly convex, breaking the tie toward keeping group
   members together and making the solution unique (§23).
2. **Stability selection** — bootstrap 100 times, keep features chosen in >80% of runs.
3. **Report the correlated group, not one member of it.** If `distance_km` and `distance_miles` are
   interchangeable, the honest finding is "distance matters", not "`distance_km` matters".

**10.** **Selection leakage** (§15.3), and the reported number is close to meaningless.

They selected the top 50 features by correlation **using the full dataset including $y$**, then
cross-validated on those 50. Every validation fold's labels were already used to choose the features,
so the CV score is not an out-of-sample estimate of anything.

**How badly?** With $p = 12{,}000$ and $n = 80$, some features will correlate strongly with $y$ **by
pure chance** — that's 12,000 draws, so extreme correlations are expected even in noise. And §4.1
guarantees the rest: with $p \gg n$, a linear model can fit *any* labelling perfectly.

**The honest estimate:** rerun with selection inside the CV loop (`make_pipeline(SelectKBest(...),
LogisticRegression())`). ⚠️ I would not predict a specific number — but the diagnostic that settles it
is to **permute $y$ and rerun the leaky pipeline**. If the shuffled-label version also reports ~88%,
the entire result is an artefact of the leak. That test takes two minutes and is the thing to run first.

Also: with $n = 80$, even a correctly-run CV estimate has enormous variance. Repeated stratified CV,
and confidence intervals rather than a point estimate.

**11.** Two explanations:

- **The signal is in a low-variance direction** (§7.2, §30.1). PCA is unsupervised — it never sees
  $y$. If the predictive sensor drift is small in amplitude, it lands in components 21+ and was
  discarded, while the retained 97% is dominated by large-amplitude but irrelevant variation (a
  temperature cycle, a vibration mode).
- **Features weren't standardised** (§30.1). PCA maximises variance, and variance is scale-dependent.
  If one sensor reports in microvolts and another in volts, the first dominates every component for
  reasons of unit choice alone.

**How to distinguish them:**
1. **Check the scaling first** — it's the cheaper hypothesis. Standardise and refit. If accuracy
   recovers, it was units.
2. If not, **correlate each principal component with $y$** individually. If components 25, 40, 60 show
   real association while 1–20 don't, that's direct evidence the signal is low-variance.
3. **The decisive test:** swap PCA for a **supervised** reduction — LDA, or partial least squares. If
   supervised reduction to 20 dimensions recovers the accuracy, the problem was PCA's unsupervised
   objective, not the dimensionality.

**And the general lesson:** choose $k$ (and whether to use PCA at all) by **downstream CV**, not by
explained variance. 97% of the variance is a statement about reconstructing $X$, and you don't care
about reconstructing $X$.

**12.** **Gini importance's cardinality bias** (§26.1).

`transaction_id` is unique per row, so it offers $n-1$ candidate split points. With that many chances,
some split will separate the training data somewhat **by pure luck**, and that spurious impurity
reduction is counted as importance. The feature carries no information — it's an identifier — and it
ranks third.

**What it tells you about the rest of the ranking — this is the important part.** It is not one bad
row; it is evidence that **the whole ranking is systematically distorted by cardinality**. Every
continuous and high-cardinality feature is inflated, and every binary flag is deflated, by the same
mechanism. You cannot fix it by deleting `transaction_id` and trusting what's left.

**What I'd do:**
1. **Drop the ID from the model entirely** — it should never have been a feature. Also check for
   near-identifiers: timestamps at full resolution, sequential order numbers, hashes.
2. **Recompute with permutation importance on held-out data** (§26.2). `transaction_id` will correctly
   score ~zero, because reliance on it doesn't generalise.
3. **Permute correlated features as a group**, or cluster by correlation first — otherwise the
   correlated-feature masking problem replaces the cardinality problem.
4. **Check for leakage while I'm there.** An ID ranking third is a warning sign in its own right: if
   IDs are assigned sequentially and fraud rates drift over time, the ID is a **proxy for the
   timestamp**, and the model may be exploiting temporal ordering that won't exist at serving time.
   That would make this not merely a metric artefact but a genuine leak.

</details>

---

## Going deeper

Ranked by importance. Difficulty markers: `intro` / `solid` / `hard`. The deck cites its own sources on
nearly every slide, so items 1–9 are the lecture's own references — unusually verifiable.

### The deck's own citations

1. **Beyer, Goldstein, Ramakrishnan & Shaft (1999), "When Is 'Nearest Neighbor' Meaningful?"
   (ICDT)** — `solid`. The paper that made distance concentration rigorous. Its main theorem gives the
   precise condition under which the nearest/farthest ratio converges to 1. **Read this if you read one
   paper from Act I.**
2. **Aggarwal, Hinneburg & Keim (2001), "On the Surprising Behavior of Distance Metrics in High
   Dimensional Space" (ICDT)** — `solid`. The follow-up, cited on the deck's §2 slide. Shows the choice
   of $L_p$ norm materially changes how fast concentration bites, with fractional norms concentrating
   more slowly. Directly actionable.
3. **Tibshirani (1996), "Regression Shrinkage and Selection via the Lasso" (JRSS-B)** — `solid`. One of
   the most-cited papers in statistics. The geometric argument in §22 is Figure 2. Very readable.
4. **Zou & Hastie (2005), "Regularization and Variable Selection via the Elastic Net" (JRSS-B)** —
   `solid`. Gives both of §23's motivations: the grouping effect, and Lasso's hard "at most $n$
   features" limitation. Section 2 is the part to read.
5. **Hoerl & Kennard (1970), "Ridge Regression: Biased Estimation for Nonorthogonal Problems"
   (Technometrics)** — `hard`. The Ridge original. Mainly of historical interest now, but it is where
   the "add a ridge to the diagonal" framing comes from.
6. **Tenenbaum, de Silva & Langford (2000), "A Global Geometric Framework for Nonlinear Dimensionality
   Reduction" (Science)** — `solid`. **Isomap**, and the deck's manifold-hypothesis citation. Short
   (Science format), with the famous Swiss-roll figure. It's where "geodesic vs ambient distance"
   becomes concrete.
7. **Kraskov, Stögbauer & Grassberger (2004), "Estimating Mutual Information" (Phys. Rev. E)** —
   `hard`. The kNN-based MI estimator that sklearn implements. Read it for §18.4's estimation caveats
   rather than the derivation.
8. **Shannon (1948), "A Mathematical Theory of Communication"** — `solid`. Entropy and mutual
   information's origin. Sections 6–7 are the relevant ones and are more readable than their reputation.
9. **Pearson (1901), "On Lines and Planes of Closest Fit"** and **Hotelling (1933), "Analysis of a
   Complex of Statistical Variables into Principal Components"** — `hard`. PCA's two independent origin
   stories: minimum reconstruction error and maximum variance respectively. Reading both makes §31.2's
   equivalence land properly. Historical rather than practical.

### Beyond the deck

10. **Guyon & Elisseeff (2003), "An Introduction to Variable and Feature Selection" (JMLR)** — `intro`.
    **The best single overview of §12–§26 that exists**, and it's free. Introduces the
    filter/wrapper/embedded taxonomy the deck uses. If you read one thing after this file, read this.
11. **Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Ch. 3 (shrinkage) and
    Ch. 14.5 (PCA)** — `hard`, free online. §3.4 is the definitive treatment of Ridge/Lasso/Elastic Net
    including the soft-threshold formula from §21.3. The canonical reference for this entire lecture.
12. **Guyon, Weston, Barnhill & Vapnik (2002), "Gene Selection for Cancer Classification using Support
    Vector Machines"** — `solid`. **SVM-RFE**, the deck's §14 example, in its original setting — which
    is also question 9's applied scenario.
13. **Efron, Hastie, Johnstone & Tibshirani (2004), "Least Angle Regression" (Annals of Statistics)** —
    `hard`. Why the entire Lasso path costs about one OLS fit (§25.1). The piecewise-linearity result
    is elegant.
14. **Strobl, Boulesteix, Zeileis & Hothorn (2007), "Bias in random forest variable importance
    measures" (BMC Bioinformatics)** — `solid`. The paper behind §26.1's cardinality-bias warning, with
    the experiments that demonstrate it. Short and convincing.
15. **Meinshausen & Bühlmann (2010), "Stability Selection" (JRSS-B)** — `hard`. The principled version
    of §22.3's bootstrap suggestion, with actual error control. Relevant to the $p \gg n$ scenario.
16. **scikit-learn User Guide, §1.13 "Feature selection" and §2.5 "Decomposing signals in components"**
    — `intro`, hands-on. The API for everything in this file, with the leakage warning stated
    explicitly. Read §1.13's pipeline example before you write any selection code.
17. **Levina & Bickel (2004), "Maximum Likelihood Estimation of Intrinsic Dimension" (NeurIPS)** —
    `hard`. How to measure §7's intrinsic dimension rather than assuming it. ⚠️ These estimators are
    themselves unreliable at small $n$ — treat outputs as orders of magnitude.
18. **Blum & Langley (1997), "Selection of Relevant Features and Examples in Machine Learning"
    (*Artificial Intelligence*, vol. 97, pp. 245–271)** — `solid`. Older, but the clearest statement of
    what "relevant" even means for a feature — which is a surprisingly slippery definition and one the
    deck doesn't attempt.

---

## 📊 Summary

| | |
|---|---|
| **Source** | `output/Lecture_07 - Module 3 Dimensionality Reduction Part 1` — 94 raw frames, **40 distinct slide states** (33 content + 7 dividers) |
| **Runtime** | 57:33 · instructor not named on screen |
| **Sections** | 31, across three acts (I: the curse §1–§5 · II: why and how to select §6–§26 · III: the linear algebra behind PCA §27–§31) |
| **Worked examples** | 13, every one carried to a final number |
| **Derivations** | Ball-in-cube volume ratio at four dimensions · **distance concentration from first principles** (mean $0.41\sqrt d$, spread constant at 0.24) · the $(1-2\varepsilon)^d$ surface argument · why $p \gg n$ guarantees a perfect fit · forward-selection and RFE cost formulas · Pearson $= 0$ on a parabola, exactly · MI $= 1.522$ bits on the same data · a χ² test to 19.20 · Ridge's $(X^\top X + \lambda I)^{-1}$ invertibility · **Elastic Net's grouping effect in three lines** · $v^\top\Sigma v = \frac1n\|Xv\|^2$ ⇒ PSD ⇒ PCA is an eigenproblem · $\Sigma = V(S^2/n)V^\top$ from the SVD · max-variance ≡ min-reconstruction-error |
| **Interactive blocks** | 4 |
| **Interview questions** | 12 with model answers (3 combining concepts), 9 depth probes, 3 whiteboard derivations, 1 applied scenario, 4 Leadership Principles |
| **Cross-references** | To [Deep Neural Networks Part 1](../Deep%20Neural%20Networks/deep-neural-networks-01.md) (XOR and the perceptron), [Part 2](../Deep%20Neural%20Networks/deep-neural-networks-02.md) (L1 vs L2 gradients, flat minima, overfitting) and [Part 3](../Deep%20Neural%20Networks/deep-neural-networks-03.md) ($0.9^{100}$ as the same arithmetic) |
| **⚠️ Flags left in the file** | 2 unanswered quiz slides, answered under 🩹 from the deck's own content (§5, §11) · the instructor is unattributed · uncertainty flags at §2.1 (the contrast threshold's dependence on $n$ and on correlated features), §7.2 (the manifold hypothesis is an assumption), §10 (Part 3's contents are expected, not verified), §18.3 (MI does **not** solve XOR univariately — correcting a natural over-reading of the slide), and §30.1 (95% is arbitrary and PCA is unsupervised) |
