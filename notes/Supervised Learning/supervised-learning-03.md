---
title: "Supervised Learning — Part 3: Trees, Ensembles, and the Machinery Around the Model"
topic: supervised-learning
lecture: 03
source: "output/Lecture_03 - Module 1 Supervised Learning Part 3"
instructor: "Sandeep Chatterjee"
slides: 24
video: "https://www.youtube.com/watch?v=wIeQ-BwtDF0"
---

# Supervised Learning — Part 3
### Trees, Ensembles, and the Machinery Around the Model

> ⚠️ **Capture note — read once, then forget it.**
>
> These notes were written from the **raw capture** in
> `output/Lecture_03 - Module 1 Supervised Learning Part 3/` (163 frames), **not** from
> `slides_deduped/`. See [`README.md`](README.md) for why the deduped set cannot be trusted.
>
> The raw capture clusters into **21 distinct deck states** plus roughly **30 distinct states of
> the hands-on Jupyter notebook** that occupies the last 14 minutes. The deck itself has **24
> numbered slides** — we know this exactly, because the notebook's own table of contents maps its
> 13 sections onto deck slides 3–24 (`slide_056`, 44:47). The slides not separately captured are
> section dividers.
>
> Frames are cited throughout as `[f47]` = `slide_047.jpg`, with the video timestamp.
>
> **One genuine gap, and I will not paper over it:**
>
> > ⚠️ **The K-Fold half of the Cross Validation slide (deck slide 14) was never fully captured.**
> > `[f37]` at 31:00 shows the heading *"K-Fold Cross Validation:"* with its sub-bullets not yet
> > animated in; the next captured frame `[f38]` at 32:30 has already advanced to
> > *"Leave-One-Out CV"*. Ninety seconds of that slide's body — almost certainly the K-fold
> > mechanics and the Stratified K-Fold bullet — fell between two forced samples. I teach K-fold
> > and stratified K-fold in full in §11 from the notebook's Section 7, which restates them
> > (`[f130]`, 53:53), and from Part 1 §6 where the same material appears. Nothing in §11 is
> > invented, but **the deck's exact wording on that half-slide is unrecovered.**
>
> One smaller gap: the model-selection **leaderboard** cell (`df_leader`) is visible as
> *unexecuted* (`[ ]:`) in `[f142]`/`[f143]` at 55:11–55:29. Its output never appeared on screen,
> so §13 has no leaderboard numbers to quote. Everything else in the notebook ran and is
> transcribed with its real output.

---

## How to read this document

| Layer | What it is | How to use it |
|---|---|---|
| **Main body** | The teaching. Every concept built from zero. | Read linearly, once, slowly. |
| **🧪 Worked examples** | Real numbers to a real answer. | Do them on paper *first*. |
| **`interactive` blocks** | Machine-readable specs for the animated web version. | Skip on first read. |
| **Glossary + Check yourself** | Recall layer. | Come back weekly. |

Callout legend: 📚 background the slide assumed · 💡 key insight · ⚠️ careful · 🧪 worked example ·
🎯 interview · 🔬 research opportunity

**Where this sits.** Part 1 gave you problem formulation, splits, leakage, bias–variance, linear
and logistic regression. Part 2 gave you losses, optimisers, metrics, and three classical
classifiers (Naive Bayes, KNN, SVM). Part 3 does two things: it introduces the **model family that
actually wins on tabular data** (trees, then ensembles of trees), and it fills in **everything that
surrounds a model in production** — multi-class wrappers, imbalance handling, cross-validation,
hyperparameter search, model selection, probability calibration, and pipelines. Part 2's
bias–variance vocabulary is used on nearly every page here; if `ρσ²` means nothing to you yet, §6
re-derives it.

---

## What you'll understand after reading this

By the end of this document you will be able to:

1. **Derive Gini impurity from scratch** as the probability that two independent draws from a node
   disagree, and explain why it and entropy almost always choose the same split.
2. **Compute information gain by hand** for a real three-class split and say, in one sentence, what
   the number means.
3. **Explain why decision trees are grown greedily** — including the actual complexity result that
   forces it — and what that costs you.
4. **State precisely why a single tree is a high-variance model**, and trace that variance through
   to the two different fixes bagging and boosting represent.
5. **Derive the 63.2% / 36.8% bootstrap result** and explain why out-of-bag scoring is free
   validation.
6. **Write down the variance of an average of correlated estimators** and use it to explain why
   Random Forest subsamples features — the single best "do you actually understand bagging"
   answer.
7. **Run one round of AdaBoost by hand**, and prove the surprising fact that after reweighting, the
   just-trained learner has weighted error *exactly* 0.5.
8. **Explain gradient boosting as gradient descent in function space**, and run one round on real
   numbers.
9. **Choose between bagging and boosting from data properties alone**, and defend it with the label-noise
   experiment from the lecture's own notebook.
10. **Compute macro-F1 from a 3×3 confusion matrix** and explain why OvO beat softmax on this dataset.
11. **Compute `class_weight='balanced'` by hand** and explain why it lowers accuracy while improving
    the model.
12. **Say exactly when SMOTE is data leakage** and what to do instead.
13. **Explain why stratified K-fold has lower variance than plain K-fold**, with the numbers.
14. **Justify random search over grid search** with the actual probability argument, not folklore.
15. **Compute a Brier score and an Expected Calibration Error by hand**, and say when calibration
    changes nothing (and why ROC-AUC can't detect miscalibration).
16. **Build a leakage-proof `ColumnTransformer` + `Pipeline` + `SearchCV`** and explain what each of
    the three layers prevents.

---

## Before we start: what you need to know

The deck assumes seven things it never teaches. Six of them are taught here from zero. The seventh
(bias–variance) is recapped because §3–§8 are, in a real sense, one long application of it.

### Prerequisite 1 — A node, and the class distribution inside it

Everything about trees is about **nodes**.

> **Node** — a subset of your training rows. The **root** node is all of them. Every split cuts one
> node into two child nodes, and every row lands in exactly one child.
>
> *In everyday words:* a node is a bucket of students. The root bucket holds all 3,539 training
> students. A split like *"tuition fees up to date?"* pours them into two smaller buckets.
>
> *Concretely:* the root of the lecture's tree holds 3,539 students — 1,137 Dropout, 635 Enrolled,
> 1,767 Graduate `[f79]`.
>
> *Why it exists:* because a tree's prediction rule is *"find which bucket you fall in, and predict
> that bucket's majority class."* Nothing else.

Inside a node with $n$ rows and $K$ classes, write $n_i$ for the count of class $i$ and

$$p_i = \frac{n_i}{n}$$

for its **class proportion**. For the root above:

$$p_{\text{Dropout}} = \tfrac{1137}{3539} = 0.3213, \quad
p_{\text{Enrolled}} = \tfrac{635}{3539} = 0.1794, \quad
p_{\text{Graduate}} = \tfrac{1767}{3539} = 0.4993$$

These three numbers are *all* the impurity measures in §2 ever look at. A node's impurity does not
depend on which features got you there, how deep you are, or how many rows there are — only on the
class proportions.

### Prerequisite 2 — Logarithms base 2, and what a "bit" is

Entropy is written with $\log_2$. If that is rusty:

$$\log_2(x) = \text{the power you raise } 2 \text{ to, to get } x$$

So $\log_2(8)=3$, $\log_2(1)=0$, $\log_2(0.5)=-1$, $\log_2(0.25)=-2$. For anything else, convert:

$$\log_2(x) = \frac{\ln x}{\ln 2} = \frac{\ln x}{0.6931}$$

**Why base 2, and what a bit is.** If I have $x$ equally likely possibilities, I need $\log_2(x)$
yes/no questions to pin down which one it is. Eight possibilities → 3 questions. That count of
questions is measured in **bits**. Entropy generalises this to unequal probabilities: it is the
*average* number of yes/no questions needed to identify the class of a randomly drawn row.

That is not decoration. It is exactly why information gain is called *gain*: it is the number of
yes/no questions the split saved you.

### Prerequisite 3 — Bias, variance, and what is random

Part 1 derived this; here is the 90-second version, because §3–§8 use it constantly.

Imagine re-drawing your training set from the population, over and over, and refitting the model
each time. You get a *different* fitted model each time.

| Term | What it measures | The question it answers |
|---|---|---|
| **Bias** | How far the *average* of those models is from the truth | "Is my model family capable of this?" |
| **Variance** | How much the models differ *from each other* | "How much does my answer depend on which rows I happened to get?" |
| **Irreducible noise** | Randomness in $y$ itself | "What's the floor?" |

> 💡 **The thing to internalise:** the randomness in "variance" is **the training set**, not the
> test set and not the model's internal randomness. When I say a decision tree is high-variance, I
> mean: *change 5% of the training rows and you get a visibly different tree.*

That is the whole motivation for §6. Bagging is a variance-reduction device and nothing else.

### Prerequisite 4 — Expectation, and the variance of an average

📚 **Background the slide assumed.** The deck asserts "bagging reduces variance without increasing
bias" `[f19]` and never shows why. To show why in §6 you need one formula.

For $B$ random variables $X_1,\dots,X_B$ that each have variance $\sigma^2$ and each pair of which
has **correlation** $\rho$:

$$\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} X_b\right) \;=\; \rho\sigma^2 \;+\; \frac{1-\rho}{B}\,\sigma^2$$

| Symbol | Read it as | What it means |
|---|---|---|
| $B$ | "number of estimators" | How many trees are in the forest. |
| $\sigma^2$ | "sigma squared" | Variance of one individual tree's prediction. |
| $\rho$ | "rho" | Correlation between two trees' predictions. $0$ = independent, $1$ = identical. |

**Where it comes from.** $\operatorname{Var}(\frac1B\sum X_b) = \frac{1}{B^2}\left[\sum_b \operatorname{Var}(X_b) + \sum_{b\neq b'} \operatorname{Cov}(X_b,X_{b'})\right]$. There are $B$ variance terms each $\sigma^2$, and $B(B-1)$ covariance terms each $\rho\sigma^2$. So it equals $\frac{1}{B^2}\left[B\sigma^2 + B(B-1)\rho\sigma^2\right] = \frac{\sigma^2}{B} + \frac{B-1}{B}\rho\sigma^2$, which rearranges to the boxed form. ∎

> 💡 Read the formula: the second term **vanishes** as you add trees. The first term **does not**.
> $\rho\sigma^2$ is a floor you cannot buy your way past with more trees. Everything Random Forest
> does beyond bagging is an attack on $\rho$.

### Prerequisite 5 — The bootstrap

> **Bootstrap sample** — draw $n$ rows from your $n$ training rows **with replacement**.
>
> *In everyday words:* put all 3,539 student records in a hat, pull one out, **write it down and put
> it back**, repeat 3,539 times. You end up with 3,539 records, but some appear twice or three times
> and some do not appear at all.
>
> *Concretely:* from rows `[A,B,C,D]` a bootstrap sample might be `[B,A,D,B]` — `B` twice, `C`
> missing.
>
> *Why it exists:* it manufactures *different-looking* training sets out of one training set. That
> is the only way bagging can get diverse trees without collecting more data.

The rows left out are the **out-of-bag (OOB)** rows for that tree, and §6 shows there are always
about 36.8% of them.

### Prerequisite 6 — Residuals and gradient descent in function space

📚 **Background the slide assumed.** The deck's boosting slide says "each new model fits the residual
errors (negative gradient of loss)" `[f23]` and moves on. Two ideas are packed in there.

> **Residual** — for one row, $r_i = y_i - \hat y_i$: how much your current prediction *undershoots*
> the truth. Positive means you predicted too low.

For squared loss $L(y,F) = \tfrac12 (y-F)^2$, differentiate with respect to the *prediction* $F$:

$$\frac{\partial L}{\partial F} = -(y-F) \quad\Longrightarrow\quad -\frac{\partial L}{\partial F} = y - F = r$$

**The negative gradient of squared loss with respect to the prediction is exactly the residual.**
This is the hinge the whole of gradient boosting swings on: "fit the residuals" and "take a gradient
step" are the same instruction when the loss is squared error, and *only* the gradient version
generalises to other losses. §7 does the general case.

Ordinary gradient descent nudges **parameters**: $w \leftarrow w - \eta\,\nabla_w L$. Gradient
boosting nudges the **function itself**: $F \leftarrow F + \eta\,h$, where $h$ is a little tree
trained to approximate $-\nabla_F L$. Same algorithm, different space.

### Prerequisite 7 — Precision, recall, F1, macro-averaging

Part 2 §14–§18 covered these. The one thing Part 2 did *not* do, and §9 and §10 here need, is
**macro-averaging** across more than two classes.

> **Macro-F1** — compute F1 separately for each class (treating that class as "positive" and all
> others as "negative"), then take the plain unweighted mean.
>
> *Why it exists:* it gives a rare class the same vote as a common one. Accuracy and micro-F1 do
> not — they are dominated by whichever class has the most rows. On this lecture's dataset, macro-F1
> is the difference between noticing and not noticing that the model cannot find "Enrolled"
> students at all.

§9 computes one from a real 3×3 confusion matrix, digit by digit.

---

## The big picture

The deck's agenda slide `[f3]` (0:18) lists exactly one line — **"Decision Trees & Ensemble
Learning"** — and that undersells what actually gets covered. The lecture has two halves, and they
are about different things.

**The first half (0:00–24:25) is one idea developed three times.** A single decision tree is a model
you can read out loud — it is a flowchart of if-then questions — and that interpretability is real
and valuable. But it has one severe defect: it is *unstable*. Change a few training rows and you get
a structurally different tree. In bias–variance language, a fully grown tree has near-zero bias and
enormous variance.

Everything that follows is a response to that one defect:

- **Prune it** (§4) — deliberately make the tree worse on training data so it stops memorising.
  Trades a little bias for a lot of variance. Cheap, limited.
- **Average many of them** (§6, bagging / Random Forest) — build hundreds of trees on perturbed
  copies of the data and vote. Variance falls; bias barely moves. This is *the* default for tabular
  data.
- **Chain them** (§7, boosting) — build shallow, deliberately-underfit trees in sequence, each
  correcting what the previous ones got wrong. This attacks *bias*, and gets the best accuracy
  available on tabular data, at the cost of being able to overfit if you let it.

That's it. Trees → prune → bag → boost. Three answers to one problem.

**The second half (24:25–48:00) is everything that isn't the model.** Multi-class wrappers, class
imbalance, cross-validation, hyperparameter search, model selection, calibration, pipelines. These
feel like a grab-bag, but there is a spine: *a model that outputs a number is not yet a system that
makes a decision.* Each section closes one gap between the two.

```
                        MODEL                      SYSTEM
   ┌──────────────┐  ┌──────────────┐   ┌──────────────────────────┐
   │ Decision     │  │ It's binary  │→  │ §9  multi-class wrapper  │
   │ tree         │  │ only         │   └──────────────────────────┘
   │      ↓ prune │  │ Classes are  │→  │ §10 imbalance handling   │
   │      ↓ bag   │  │ lopsided     │   └──────────────────────────┘
   │      ↓ boost │  │ One split is │→  │ §11 cross-validation     │
   └──────────────┘  │ noisy        │   └──────────────────────────┘
                     │ Knobs unset  │→  │ §12 hyperparameter search│
                     │ Which family?│→  │ §13 model selection      │
                     │ Scores≠probs │→  │ §14 calibration          │
                     │ Leaks in prod│→  │ §15 pipelines            │
                     └──────────────┘   └──────────────────────────┘
```

The final 14 minutes run all of it on one real dataset — 4,424 Portuguese university students,
predicting Dropout / Enrolled / Graduate — and §16 walks that end to end with every number the
notebook actually printed.

---

## 1. Decision trees: the idea

The slide states it in one line `[f7]`, 5:04:

> **Idea:** Recursively partition feature space using if-then rules.
> A decision tree is a flowchart — each node asks one question about one feature.

Both halves of that matter.

**"Each node asks one question about one feature."** A tree does not form combinations. It cannot
ask *"is income − expenses > 0?"*. It can only ask *"is income ≤ 50k?"*, then, further down, *"is
expenses ≤ 50k?"*. The consequence is geometric: the decision boundary a tree draws is always made
of **axis-parallel** cuts. It carves feature space into rectangles (in 2-D), boxes (in 3-D),
hyper-rectangles in general.

```
   A tree's boundary                     What it cannot draw in one cut
   ─────────────────                     ──────────────────────────────
   x₂                                    x₂
    │  ┌───────┬────┐                     │        ╱
    │  │   A   │ B  │                     │   A   ╱    B
    │  ├───┬───┴────┤                     │      ╱
    │  │ B │   A    │                     │     ╱
    └──┴───┴────────── x₁                 └────╱─────────── x₁
      axis-parallel boxes                  a 45° line — a tree needs a
      only                                 staircase of many cuts to approximate it
```

> ⚠️ **Where people get confused.** "Trees capture nonlinearity" `[f11]` is true, and people over-read
> it. A tree captures nonlinearity in the *target* — it can represent an arbitrarily wiggly function
> of the features, given enough depth. It does **not** capture *rotated* structure efficiently. A
> boundary at 45° to your axes, which logistic regression nails with two weights, costs a tree a
> staircase of dozens of splits. This is exactly why feature engineering that creates ratios and
> differences still helps trees, despite the folklore that "trees don't need feature engineering."

**"Recursively partition."** The algorithm is three lines:

```
def grow(node):
    if stopping_condition(node):        # pure enough, too small, too deep
        make_leaf(node); return
    (feature, threshold) = best_split(node)   # ← §2 defines "best"
    left, right = split(node, feature, threshold)
    grow(left); grow(right)             # ← recursion
```

Prediction is even simpler: walk the questions from the root until you hit a leaf, and return that
leaf's majority class (classification) or mean target (regression).

The diagram on the slide `[f7]` shows the shape concretely — root `age ≤ 30? (Gini = 0.48)`,
children `income ≤ 50k? (Gini = 0.32)` and `owns_home? (Gini = 0.28)`, then four pure leaves of
sizes 42, 38, 51, 29. Note that the Gini values *fall* as you descend: 0.48 → 0.32 / 0.28 → 0.
That falling number is the whole training algorithm, and §2 is about what it is.

> 💡 **Key insight.** A decision tree is not really "a model" in the way linear regression is. It is
> a *partition of feature space* plus a lookup table of constants. All the learning is in choosing
> where to cut.

---

## 2. Splitting criteria: Gini, entropy, information gain

The tree needs to score candidate splits. It does so by measuring **impurity** — how mixed the
classes are in a node — and picking the split that reduces it most.

### 2.1 Gini impurity, derived

The slide gives the formula and two anchor points `[f7]`:

> **Gini Impurity:** $\text{Gini} = 1 - \sum_i p_i^2$ — Gini = 0 pure, Gini = 0.5 max impurity (binary)

Words first.

> The formula says: **pick two rows from this node at random, independently. Gini is the probability
> that they belong to different classes.**

Here is why that's the same thing. Draw one row; the probability it is class $i$ is $p_i$. Draw a
second, independently; the probability it is *also* class $i$ is $p_i$ again. So the probability
both are class $i$ is $p_i^2$, and the probability they **match** on any class is $\sum_i p_i^2$.
The probability they **differ** is therefore

$$\text{Gini} = 1 - \sum_{i=1}^{K} p_i^2$$

∎ That's the derivation. It is also identical to $\sum_i p_i(1-p_i)$, which reads as *"the expected
error rate if you labelled each row by drawing a class at random from the node's own
distribution."*

| Symbol | Read it as | What it means |
|---|---|---|
| $K$ | "kay" | Number of classes. |
| $p_i$ | "p sub i" | Fraction of the node's rows in class $i$. $\sum_i p_i = 1$. |
| $\text{Gini}$ | "Gini impurity" | 0 = all one class. Larger = more mixed. |

**The bounds.** Minimum is 0, when some $p_i = 1$: two draws always match. Maximum is when all
classes are equally likely, $p_i = 1/K$:

$$\text{Gini}_{\max} = 1 - K\cdot\left(\tfrac1K\right)^2 = 1 - \tfrac1K$$

For $K=2$ that is $1 - \tfrac12 = \boxed{0.5}$ — which is exactly the "Gini = 0.5 max impurity
(binary)" on the slide. For $K=3$, as in this lecture's dataset, the ceiling is $1-\tfrac13 = 0.667$,
**not** 0.5.

> ⚠️ **Where people get confused.** "Gini maxes out at 0.5" is a *binary-only* fact. Candidates repeat
> it in interviews on multi-class problems and it is wrong there. The general ceiling is $1-1/K$,
> which rises toward 1 as classes multiply.

### 2.2 The deck's own numeric check

Slide `[f15]` (16:22) gives three, and the notebook verifies them in code at `[f90]` (49:47):

```
Gini for [50/50  split] = 1 − (0.5² + 0.5²) = 0.500
Gini for [90/10  split] = 1 − (0.9² + 0.1²) = 0.180
Gini for [100/0  split] = 1 − (1.0² + 0.0²) = 0.000
```

Check the middle one yourself: $0.9^2 = 0.81$, $0.1^2 = 0.01$, sum $=0.82$, so Gini $= 0.18$. A 90/10
node is described by the slide as "mostly pure" — and 0.18 out of a possible 0.5 confirms it.

### 2.3 Entropy

The slide `[f7]`:

> **Entropy:** $H = -\sum_i p_i \log_2(p_i)$ — measures information content / disorder

Words first.

> The formula says: **on average, how many yes/no questions do I need to ask to learn the class of a
> row drawn from this node?**

The reasoning: an event of probability $p$ carries $\log_2(1/p) = -\log_2 p$ bits of "surprise" — a
1-in-8 event takes 3 questions to pin down. Entropy is the *average* surprise, weighting each class
by how often it occurs. Hence $H = \sum_i p_i \cdot (-\log_2 p_i)$.

| Symbol | Read it as | What it means |
|---|---|---|
| $H$ | "H" or "entropy" | Average bits needed to identify a row's class. |
| $p_i$ | "p sub i" | Class proportion, as before. |
| $\log_2$ | "log base two" | See Prerequisite 2. |

**Bounds.** $H = 0$ when one class has $p=1$ (no questions needed — you already know). Maximum at
uniform $p_i = 1/K$:

$$H_{\max} = -K \cdot \tfrac1K \log_2\tfrac1K = \log_2 K$$

For binary that's $\log_2 2 = 1$ bit. For three classes, $\log_2 3 = 1.585$ bits.

> ⚠️ The convention $0 \log_2 0 = 0$ is used (the limit is 0). Without it the formula would be
> undefined for pure nodes, which is exactly where you most want it to say zero.

### 2.4 Information gain

$$\text{IG} = H(\text{parent}) - \sum_{k} \frac{|N_k|}{|N|} H(k)$$

Words first.

> The formula says: **how many bits of uncertainty about the class did this split remove?** Take the
> parent's disorder; subtract the disorder still left in the children, counting each child in
> proportion to how many rows landed in it.

| Symbol | Read it as | What it means |
|---|---|---|
| $H(\text{parent})$ | "entropy of the parent" | Disorder before the split. |
| $k$ | "kay" | Index over the child nodes (2 for a binary split). |
| $|N_k|$ | "size of child k" | Number of rows in child $k$. |
| $|N|$ | "size of parent" | Number of rows in the parent. |
| $\frac{|N_k|}{|N|}$ | "the weight of child k" | Fraction of parent rows landing in $k$. |
| $H(k)$ | "entropy of child k" | Disorder remaining in child $k$. |

**The weighting is the part people skip, and it is the part that matters.** Without it, a split that
peels off one perfectly pure row would look brilliant. With it, that child gets weight $1/n$ and
contributes essentially nothing. The tree is therefore biased toward splits that are *both* pure
*and* balanced.

The same weighted-reduction idea applies with Gini in place of $H$; sklearn calls it *impurity
decrease* and it is what `criterion='gini'` (the default) optimises.

### 🧪 Worked example — one split, both criteria, to a final number

A node of **100 students**: 50 Graduate, 30 Dropout, 20 Enrolled. We test the split
*"tuition fees up to date?"*, which sends 70 students to **Yes** (45 G, 10 D, 15 E) and 30 to **No**
(5 G, 20 D, 5 E).

**Step 1 — parent impurity.**
$p_G = 0.5,\; p_D = 0.3,\; p_E = 0.2$.

$$\text{Gini}_{\text{parent}} = 1 - (0.5^2 + 0.3^2 + 0.2^2) = 1 - (0.25 + 0.09 + 0.04) = 1 - 0.38 = \mathbf{0.620}$$

$$H_{\text{parent}} = -\big[0.5\log_2 0.5 + 0.3\log_2 0.3 + 0.2\log_2 0.2\big]$$

$\log_2 0.5 = -1.0000$, $\log_2 0.3 = -1.7370$, $\log_2 0.2 = -2.3219$. So

$$H_{\text{parent}} = 0.5(1.0000) + 0.3(1.7370) + 0.2(2.3219) = 0.5000 + 0.5211 + 0.4644 = \mathbf{1.4855}\ \text{bits}$$

(Sanity: below the 3-class ceiling of 1.585. Good.)

**Step 2 — child impurities.**

*Yes* child, $n=70$: $p_G = 45/70 = 0.6429$, $p_D = 10/70 = 0.1429$, $p_E = 15/70 = 0.2143$.

$$\text{Gini}_{\text{Yes}} = 1 - (0.6429^2 + 0.1429^2 + 0.2143^2) = 1 - (0.4133 + 0.0204 + 0.0459) = \mathbf{0.5204}$$

$$H_{\text{Yes}} = 0.6429(0.6374) + 0.1429(2.8071) + 0.2143(2.2220) = 0.4097 + 0.4011 + 0.4762 = \mathbf{1.2870}\ \text{bits}$$

*No* child, $n=30$: $p_G = 5/30 = 0.1667$, $p_D = 20/30 = 0.6667$, $p_E = 5/30 = 0.1667$.

$$\text{Gini}_{\text{No}} = 1 - (0.1667^2 + 0.6667^2 + 0.1667^2) = 1 - (0.0278 + 0.4444 + 0.0278) = \mathbf{0.5000}$$

$$H_{\text{No}} = 0.1667(2.5850) + 0.6667(0.5850) + 0.1667(2.5850) = 0.4308 + 0.3900 + 0.4308 = \mathbf{1.2516}\ \text{bits}$$

**Step 3 — weighted children, and the gain.** Weights are $70/100 = 0.7$ and $30/100 = 0.3$.

$$\text{Gini}_{\text{children}} = 0.7(0.5204) + 0.3(0.5000) = 0.3643 + 0.1500 = 0.5143$$
$$\boxed{\text{Gini gain} = 0.6200 - 0.5143 = \mathbf{0.1057}}$$

$$H_{\text{children}} = 0.7(1.2870) + 0.3(1.2516) = 0.9009 + 0.3755 = 1.2764\ \text{bits}$$
$$\boxed{\text{IG} = 1.4855 - 1.2764 = \mathbf{0.2091}\ \text{bits}}$$

**What the numbers mean.** The split removed **0.209 bits** of uncertainty — about 14% of the 1.4855
bits we started with. In plain terms: knowing whether a student's fees are up to date gets you
roughly a fifth of the way to one yes/no question's worth of information about their outcome. Not
huge, but real. The tree computes this for *every feature at every candidate threshold* and takes
the maximum.

Notice the two criteria agreed the split was worth making, and they nearly always do.

### 2.5 Gini or entropy? The honest answer

| | Gini | Entropy |
|---|---|---|
| Formula | $1-\sum p_i^2$ | $-\sum p_i\log_2 p_i$ |
| Cost | Multiplications only | Logarithms — measurably slower |
| Max (binary) | 0.5 | 1.0 |
| Max ($K$ classes) | $1 - 1/K$ | $\log_2 K$ |
| Shape | Slightly flatter near $p=0.5$ | Slightly peakier |
| sklearn | `criterion='gini'` (default) | `criterion='entropy'` |

They pick the same split the overwhelming majority of the time — the two curves are close enough
that the argmax rarely differs. Gini is the sklearn default because it avoids logarithms. Entropy's
slightly sharper peak makes it marginally more inclined to split off pure sub-groups.

> 🎯 **Interview.** *"Gini or entropy?"* The correct answer is **"it almost never matters, and here's
> why"** — then draw both curves on $[0,1]$ and show they have the same shape, same zeros, same
> single maximum at $p=0.5$. Candidates who claim one is systematically better are guessing.
> If pushed for a preference: Gini, for speed, and tune `max_depth` instead — that knob is worth
> hundreds of times more.

```interactive
type: slider
title: Impurity as the class mix changes
concept: Gini impurity and entropy as functions of class proportion
control: A slider for p (proportion of class 1) in a two-class node, 0 to 1
observe: Two curves — Gini (scaled) and entropy — drawn over p, with the current point marked, and the node redrawn as a bag of coloured dots
insight: Both curves are zero at the ends and peak at p = 0.5; they have the same shape, so they choose the same split almost every time — which is why the Gini-vs-entropy question is not worth agonising over
fallback: The table in §2.5 plus the three verified numbers in §2.2 (0.500, 0.180, 0.000) give the same information statically.
```

---

## 3. Why a single tree fails: greedy splitting and high variance

Slide `[f11]` (9:31) is blunt about it:

> **Strengths:** Interpretable, handles mixed types, no scaling needed, captures nonlinearity
> **Weaknesses:** Prone to overfitting (high variance) — small data changes → very different tree.
> Greedy splitting — locally optimal, not globally optimal

Take the weaknesses one at a time, because both are deeper than they look.

### 3.1 Greedy splitting — and why there is no alternative

The algorithm picks the best split *at this node, right now*, with no lookahead. That can be
provably suboptimal: a split that looks mediocre now may enable two superb splits below it, and
greedy search will never find it.

📚 **Background the slide assumed — why not just search all trees?** Because you can't.

> **Constructing an optimal binary decision tree is NP-complete** (Hyafil & Rivest, 1976). There is
> no known algorithm that finds the globally best tree in time polynomial in the number of features
> and rows, and there almost certainly isn't one.

That is the actual reason every tree learner in production — CART, ID3, C4.5, sklearn, XGBoost — is
greedy. It is not laziness; it is the only tractable option. Greedy growth is $O(d \cdot n \log n)$
per level and gives you a good-enough tree.

> 🔬 **Research opportunity.** "Optimal decision trees" is an active area again: modern MILP and
> dynamic-programming formulations (e.g. the `OSDT` / `GOSDT` line of work) can now find provably
> optimal *small* trees — depth ≤ 4 or so — on moderate datasets, and they sometimes beat greedy
> trees of the same size by a useful margin. The open question is whether that ever scales to the
> depths that matter in practice, and whether the gap survives once you're allowed to ensemble.
> ✅ Confirmed — Lin, Zhong, Hu, Rudin & Seltzer, *"Generalized and Scalable Optimal Sparse Decision
> Trees"*, ICML 2020 (PMLR vol. 119, pp. 6150–6160; arXiv:2006.08690).

### 3.2 High variance — where it actually comes from

Here is the mechanism, which is more specific than "trees overfit."

Every split is chosen by an **argmax** over hundreds of candidate (feature, threshold) pairs. Suppose
at the root, feature A scores an impurity reduction of 0.1057 and feature B scores 0.1041. Feature A
wins. Now resample the training data slightly — a few rows in, a few out — and B might score 0.1053
against A's 0.1049. **B wins, and the entire tree below that node is different.** Not slightly
different: a different feature at the root means different children, which get different splits,
which get different children.

> 💡 **Key insight.** Tree variance is *argmax variance*, amplified by *recursion*. A near-tie at the
> root — which is common, because good features are correlated — flips the whole structure. This is
> why "small data changes → very different tree" is not hyperbole and why *no* amount of careful
> single-tree tuning fixes it. The fix has to be averaging (§6).

### 3.3 The strengths are real, and worth being precise about

| Claimed strength | Why it's true | The fine print |
|---|---|---|
| **Interpretable** | The path from root to leaf *is* the explanation, in the feature's own units | Only for shallow trees. A depth-20 tree with 3,000 leaves is not interpretable by any honest definition. |
| **Handles mixed types** | Splits are comparisons; a comparison works on ordered numbers and on categories | ⚠️ sklearn's `DecisionTreeClassifier` **cannot** take raw categorical columns — see below |
| **No scaling needed** | A split at `income ≤ 50000` is unchanged if you rescale income; only the *order* matters | Genuinely true, and a real operational advantage over KNN/SVM (Part 2 §21, §23) |
| **Captures nonlinearity** | Enough splits approximate any function | Axis-parallel only (§1) |

> ⚠️ **The categorical trap, straight from the lecture's own notebook.** The deck says trees "handle
> mixed types." sklearn does not. `[f79]` (48:30) shows the workaround the instructor had to write:
>
> ```python
> # Trees can't use pandas categorical dtype directly — encode categoricals as int codes
> def encode_cats(df):
>     df = df.copy()
>     for c in CATEGORICAL_COLS:
>         df[c] = df[c].cat.codes if hasattr(df[c], 'cat') else df[c].astype('category').cat.codes
>     return df
> ```
>
> This is integer-encoding a categorical, which imposes a **fake ordering** — it lets the tree split
> on `course_code ≤ 7`, a question with no meaning. sklearn trees tolerate it better than linear
> models do (a tree can isolate any single category with enough splits), but it is a real
> compromise. LightGBM and CatBoost handle categoricals natively and properly, which is a
> substantial part of why they win on business data. §15 shows the correct sklearn answer:
> `OneHotEncoder` inside a `ColumnTransformer`.

---

## 4. Controlling overfitting: pre-pruning and post-pruning

Slide `[f11]` lists the knobs; slide `[f15]` (16:22) names the two strategies.

> **Pre-pruning:** Set `max_depth`, `min_samples_split`, `min_samples_leaf` before training
> **Post-pruning (cost-complexity):** Grow full tree, then prune branches that add complexity > gain
> `ccp_alpha` in sklearn: higher → more aggressive pruning

### 4.1 Pre-pruning: stop early

| Knob | What it does | Symptom it treats |
|---|---|---|
| `max_depth` | Hard cap on levels below the root | The single most effective knob. Start here. |
| `min_samples_split` | Refuse to split a node with fewer than this many rows | Splits justified by 3 rows |
| `min_samples_leaf` | Refuse any split that would create a leaf smaller than this | Leaves of size 1 memorising individual students |
| `max_leaf_nodes` | Cap total leaves; grows best-first rather than depth-first | Direct control of model size |
| `min_impurity_decrease` | Refuse splits whose weighted impurity drop is below a threshold | Splits that are pure noise |

**Why this is bias–variance in disguise.** Each knob *removes* trees from the hypothesis space. That
raises bias (the family can express less) and lowers variance (fewer near-ties to flip on). You are
buying stability with expressiveness, and the optimum is empirical.

### 🧪 Worked example — the depth sweep from the lecture's own notebook

The notebook `[f96]` (50:20) sweeps `max_depth` from 1 to 20 and plots train and test accuracy:

```python
depths = range(1, 21)
train_scores, test_scores = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_train_enc, y_train)
    train_scores.append(t.score(X_train_enc, y_train))
    test_scores.append (t.score(X_test_enc,  y_test))
```

Printed result:

```
Best max_depth by test accuracy: 5   → test acc = 0.757
```

And the plotted curves are the canonical picture:

```
 acc
 1.00 ┤                        ●───●───●───●───●   train  → 100%: pure memorisation
 0.95 ┤                  ●───●
 0.90 ┤            ●───●
 0.85 ┤        ●─●
 0.80 ┤    ●─●
 0.75 ┤  ■─■─■■■                                    test peaks at depth 5 (0.757)
 0.70 ┤■■     ■─■■■─■■■■─■■■■■■■■─■■■               and then decays to ≈0.69
      └┬───┬───┬───┬───┬───┬───┬───┬───┬
       1   3   5   7  10  13  15  18  20   max_depth
```

**Read the three regimes off the plot.**

- **Depth 1–4:** both curves rise together. Underfitting — bias-limited. More capacity helps.
- **Depth 5:** test peaks at **0.757**. This is the sweet spot.
- **Depth 6–20:** train marches to 1.00 while test *falls* to about 0.69. The gap between the curves
  is variance, and it is being paid for with nothing.

The fully grown tree — the one you get with sklearn's defaults, since `max_depth=None` — scores
**0.697** on test `[f106]`. Depth-5 pruning bought **6 accuracy points for one hyperparameter.**

> 💡 That is the whole argument for pre-pruning in one number. `DecisionTreeClassifier()` with
> default settings is not a baseline; it is a bug.

### 4.2 Post-pruning: cost-complexity

Grow the tree fully, then remove subtrees that don't pay for themselves. The criterion:

> The formula says: **score a tree by its error *plus* a fine for every leaf it has. Prune whatever
> subtree lets you pay less total.**

$$R_\alpha(T) = R(T) + \alpha\,|\tilde T|$$

| Symbol | Read it as | What it means |
|---|---|---|
| $T$ | "T" | A tree (or subtree). |
| $R(T)$ | "R of T" | Training error (or total impurity) of $T$. |
| $\lvert\tilde T\rvert$ | "number of leaves of T" | Model complexity, counted in leaves. |
| $\alpha$ | "alpha" | Price per leaf. sklearn's `ccp_alpha`. |
| $R_\alpha(T)$ | "cost-complexity of T" | The thing being minimised. |

At $\alpha = 0$ the fine is free, so the full tree wins and nothing is pruned. As $\alpha$ rises,
leaves get progressively too expensive to keep, and branches collapse — always the weakest ones
first. At $\alpha$ large enough, the tree collapses to a single root node.

**This is a familiar shape.** $R_\alpha(T) = \text{error} + \alpha \cdot \text{complexity}$ is
exactly the structure of ridge and lasso from Part 1 §14: fit term plus penalty term, with a
knob trading them off. The penalty here counts leaves instead of summing squared or absolute
weights, but the logic — *and the fact that you tune $\alpha$ by cross-validation, never on test* —
is identical.

**Why it beats pre-pruning in principle.** Pre-pruning is myopic in the same way greedy splitting is:
it stops at a node that looks unpromising, and never discovers that two excellent splits lay just
beneath it. Post-pruning grows everything first, so it sees the whole picture before cutting.

**Why pre-pruning is used anyway.** Growing the full tree is expensive on large data, and in practice
`max_depth` tuned by CV gets you most of the benefit for a fraction of the compute. The notebook does
both — `[f102]` (50:45) plots the full `cost_complexity_pruning_path`, showing test accuracy against
`ccp_alpha` on a log scale alongside leaf count collapsing from thousands toward one.

```python
path   = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(X_train_enc, y_train)
alphas = path.ccp_alphas[:-1]          # drop the trivial root-only tree
```

> ⚠️ **A subtlety worth knowing.** `cost_complexity_pruning_path` returns the *finite* set of
> $\alpha$ values at which the pruned tree actually changes. Between two consecutive values, nothing
> happens. So you never grid-search `ccp_alpha` over a made-up linspace — you search the path the
> data gives you. That's what `alphas = path.ccp_alphas` is for.

```interactive
type: slider
title: Prune it and watch both curves
concept: Pre-pruning as an explicit bias–variance trade
control: A slider for max_depth from 1 to 20
observe: The tree diagram grows/shrinks on the left; train and test accuracy curves fill in on the right with the current depth highlighted; the train–test gap is shaded
insight: Train accuracy is monotone in depth and therefore useless for choosing it — the shaded gap IS the variance you are paying for, and it peaks in usefulness at depth 5 (test 0.757) before the tree starts memorising
fallback: The ASCII plot and the three-regime reading in §4.1, plus the two real numbers: depth-5 test 0.757 vs full-tree test 0.697.
```

---

## 5. Feature importance: what it means and what it doesn't

Slide `[f15]`:

> **Feature Importance:** Measured by total impurity reduction across all splits
> Features used higher in the tree (earlier splits) have more impact
> sklearn: `tree.feature_importances_` — normalized to sum to 1

**The mechanism, precisely.** For each feature $j$, walk every node in the tree that splits on $j$.
For each such node, compute the impurity decrease it achieved, **weighted by the fraction of samples
reaching that node**. Sum over all nodes. Then normalise across features so the vector sums to 1.

$$\text{Imp}(j) = \frac{1}{Z}\sum_{\text{nodes } t \text{ splitting on } j} \frac{n_t}{n}\Big[\,I(t) - \tfrac{n_L}{n_t}I(L) - \tfrac{n_R}{n_t}I(R)\Big]$$

where $n_t$ is the sample count at node $t$, $I$ is the impurity, $L,R$ are the children, and $Z$
normalises. The $n_t/n$ factor is why "features used higher in the tree have more impact" — the root
sees all $n$ rows and gets weight 1; a node at depth 8 might see 40 rows and get weight 0.011.

The notebook plots it for the real dataset `[f107]` (51:26), comparing a single tree against a
Random Forest on the top 10 features. The dominant feature by a wide margin is **Curricular units
2nd sem (approved)** — the number of courses a student passed in their second semester — followed by
*2nd sem (grade)*, *1st sem (approved)*, *1st sem (grade)*, *Admission grade*, *Tuition fees up to
date*, *Age at enrollment*.

That ordering is worth pausing on: **the model's top signal is academic performance in the most
recent term**, which is both intuitive and operationally awkward — see the interview scenario in
Phase 4.

### ⚠️ Three ways impurity importance lies

This is a place where the slide's one line hides real danger, and it is a favourite interview probe.

**1. It is biased toward high-cardinality features.** A feature with many distinct values (a
continuous variable, or an ID-like column) offers the split-finder many more thresholds to try. More
candidates means a higher chance one of them looks good *by luck*. So continuous and
high-cardinality features systematically absorb importance they haven't earned. In the extreme, give
a tree a random unique ID column and it will happily assign it importance.

**2. It splits credit arbitrarily between correlated features.** If *1st sem grade* and *2nd sem
grade* are strongly correlated, whichever one the greedy search happens to pick first takes the
credit, and the other looks unimportant — even though either alone would have worked. **Low
importance does not mean "irrelevant."** It can mean "redundant with something already used."

**3. It is computed on training data.** It tells you what the tree *used*, not what *generalises*.
A feature that drives splits which overfit gets high importance for it.

> 💡 **What to use instead.** **Permutation importance** — shuffle one feature's column in the
> *validation* set and measure how much the score drops — answers "does this feature help me predict
> unseen data?", which is usually the question you meant. It costs a full re-scoring per feature, is
> computed on held-out data, and is not biased by cardinality. `sklearn.inspection.permutation_importance`.
> For per-prediction attribution rather than global ranking, SHAP values are the standard.

> 🎯 **Interview.** *"Your Random Forest says feature X is the most important. What do you conclude?"*
> The strong answer names all three caveats above, then says what you'd actually do: **check
> permutation importance on validation, and check whether X is correlated with anything you dropped.**
> The weak answer says "X drives the target."

### The rest of slide `[f15]`

Two closing lines that set up the whole second half of the lecture:

> **Trees as building blocks:** Weak individually (high variance), powerful in ensembles
> **Amazon example:** Delivery delay root-cause analysis — interpretable rules for ops teams

The first is the bridge to §6. The second is the honest use case for a *single* tree: when the
output is meant to be **read by a human who will act on it**. An ops team investigating delivery
delays needs a rule like *"if warehouse = X and carrier = Y and weekday = Friday, delay risk is
3.4×"*, which they can take to the carrier. A Random Forest's 0.83 AUC is useless in that meeting.

> 💡 **Key insight.** Interpretability is not a nice-to-have you trade against accuracy on a smooth
> curve. It is a *requirement that comes from who reads the output*. If the consumer is a human
> making a decision, a depth-4 tree at 0.72 accuracy may be strictly more valuable than a boosted
> ensemble at 0.79.

---

## 6. Ensemble learning I: bagging and Random Forests

Slide `[f19]` (16:22) is dense; take it line by line.

> **Idea:** Combine multiple "weak" models to create a strong model
> **Bagging (Bootstrap Aggregating):** Train multiple models on bootstrap samples (sampling with
> replacement) · Combine: majority vote (classification) or average (regression) · Reduces variance
> without increasing bias
> **Random Forests = Bagging + random feature subsets at each split:** Each tree sees sqrt(d)
> features (classification) or d/3 (regression) · Decorrelates trees → better variance reduction
> than plain bagging
> **Hyperparameter Effects:** `n_estimators`: More trees → diminishing returns, no overfitting (safe
> to increase) · `max_features`: Lower → more decorrelation → less variance, slightly more bias ·
> `max_depth`: Deeper → lower bias per tree (bagging compensates the variance)
> **Out-of-Bag (OOB) Score:** Each tree validated on ~37% unseen data — free validation!
> **When to use:** Default go-to for tabular data; robust, parallelizable, few assumptions

### 6.1 Why averaging works, derived

The claim "reduces variance without increasing bias" is provable, and the proof is short.

**Bias is unchanged.** Each tree $T_b$ is fit to a bootstrap sample of the same distribution, so
every tree has the same expected prediction: $\mathbb{E}[T_b(x)] = \mu(x)$ for all $b$. The
ensemble's expectation is

$$\mathbb{E}\left[\frac1B\sum_b T_b(x)\right] = \frac1B\sum_b \mathbb{E}[T_b(x)] = \frac1B \cdot B\mu(x) = \mu(x)$$

Identical to a single tree's. So the bias — the gap between $\mu(x)$ and the truth — has not moved
one bit. ∎

**Variance falls.** Straight from Prerequisite 4:

$$\operatorname{Var}\!\left(\frac1B\sum_b T_b(x)\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

The second term shrinks like $1/B$. Add trees, variance falls. ∎

> 💡 **This is why deep trees are the right base learner for bagging.** A deep tree is low-bias,
> high-variance. Bagging cannot fix bias but *is* a variance-destroying machine. So you feed it the
> model whose only problem is the one bagging solves. That is exactly what the slide's
> "`max_depth`: Deeper → lower bias per tree (bagging compensates the variance)" means — and it is
> the **opposite** of the advice for boosting (§7), where shallow trees are mandatory.

### 6.2 The $\rho$ floor, and what Random Forest is actually for

Look at that formula again and put numbers in it. Take $\sigma^2 = 1$ and $B = 100$ trees:

| $\rho$ (tree correlation) | $\rho\sigma^2$ (floor) | $\frac{1-\rho}{B}\sigma^2$ | Total variance | vs. one tree |
|---|---|---|---|---|
| 1.00 (identical trees) | 1.000 | 0.000 | **1.000** | no reduction at all |
| 0.90 | 0.900 | 0.001 | **0.901** | 10% |
| 0.50 | 0.500 | 0.005 | **0.505** | 50% |
| 0.10 | 0.100 | 0.009 | **0.109** | 89% |
| 0.00 (independent) | 0.000 | 0.010 | **0.010** | 99% |

**Read the first column.** The floor $\rho\sigma^2$ is untouched by $B$. Going from 100 trees to
10,000 trees at $\rho = 0.9$ takes you from 0.901 to 0.900. You have burned 100× the compute for
nothing.

**Now the problem with plain bagging.** Bootstrap samples share about 63% of their rows (§6.3). Trees
grown on 63%-overlapping data, with the same greedy algorithm and the same features available,
mostly pick **the same feature at the root** — because one feature is genuinely the strongest. The
trees come out highly correlated. $\rho$ is large. The floor is high.

**Random Forest's fix, in one sentence:** at every split, hide all but a random handful of features,
so the dominant feature is *unavailable* in most trees and other features get to be the root.

$$\texttt{max\_features} = \sqrt{d} \ \text{(classification)}, \qquad d/3 \ \text{(regression)}$$

For this lecture's dataset, $d = 36$ features, so $\sqrt{36} = 6$: each split considers only 6
randomly chosen features out of 36.

> 💡 **Key insight — say this in an interview.** *Random Forest deliberately makes each individual
> tree worse in order to make the ensemble better.* Restricting features raises each tree's bias and
> variance. But it slashes $\rho$, and the $\rho\sigma^2$ floor is what actually limits the
> ensemble. Trading a small increase in $\sigma^2$ for a large decrease in $\rho$ is a winning trade,
> and the formula above is the proof.

That is also exactly what the slide means by "`max_features`: Lower → more decorrelation → less
variance, slightly more bias." The whole line falls out of one formula.

### 6.3 🧪 Out-of-bag: deriving the 37%

The slide claims "each tree validated on ~37% unseen data." Here is where 37% comes from.

Take one specific row and one bootstrap sample. Each of the $n$ draws is independent and uniform
over the $n$ rows, so:

- Probability this row is **not** picked on a single draw: $1 - \frac1n$
- Probability it is not picked on **any** of the $n$ draws: $\left(1-\frac1n\right)^n$

Now use the standard limit $\left(1-\frac1n\right)^n \to e^{-1}$ as $n \to \infty$:

$$P(\text{row is out-of-bag}) \longrightarrow e^{-1} = \mathbf{0.3679}$$

So **36.8% of rows are out-of-bag for any given tree, and 63.2% are in-bag.**

The convergence is fast — you don't need large $n$:

| $n$ | $(1-1/n)^n$ |
|---|---|
| 5 | 0.3277 |
| 10 | 0.3487 |
| 100 | 0.3660 |
| 1,000 | 0.3677 |
| **3,539** (this dataset) | **0.3678** |
| $\infty$ | 0.36788 |

**Why this is free validation.** Every row is out-of-bag for roughly 37% of the trees. To score row
$i$, poll only the trees that never saw it, and take their majority vote. Do that for every row and
you have an honest held-out estimate — **using no held-out data**. You have effectively run
cross-validation for free, as a by-product of training.

The notebook confirms it `[f106]` (51:16):

```
Single tree   · test acc = 0.697
Bagging       · test acc = 0.753
Random Forest · test acc = 0.769   (OOB score = 0.774 — free validation!)
```

**Read all four numbers.**

- Single tree → bagging: **+5.6 points** for nothing but averaging 100 copies.
- Bagging → Random Forest: **+1.6 points** purely from feature subsampling — i.e. purely from
  attacking $\rho$. That is §6.2 showing up as an actual number.
- OOB (0.774) vs test (0.769): **within half a point.** The OOB estimate is doing its job. If those
  two ever diverge badly, you have a distribution shift between train and test, and that is a
  finding, not a nuisance.

> ⚠️ **When OOB is not enough.** OOB assumes rows are exchangeable. With **time-ordered** data it is
> optimistic in exactly the way random K-fold is (§11.4) — the "unseen" rows are interleaved in time
> with the training rows, so the tree has effectively seen the future. With **grouped** data
> (multiple rows per customer) it leaks across the group. In both cases use a proper temporal or
> grouped split and ignore OOB.

### 6.4 The hyperparameters, and the one asymmetry that matters

| Knob | Effect | Guidance |
|---|---|---|
| `n_estimators` | More trees → variance falls toward the $\rho\sigma^2$ floor | **More is never worse for accuracy.** Raise until the curve flattens, then stop for compute reasons only. |
| `max_features` | Lower → lower $\rho$, higher per-tree bias | `sqrt(d)` default is usually near-optimal. Lower it if features are highly correlated. |
| `max_depth` | Deeper → lower bias per tree | Leave at `None`. Bagging handles the variance. |
| `min_samples_leaf` | Larger → smoother trees | Raise to 5–20 on noisy data. |
| `oob_score=True` | Turns on OOB scoring | Free. Turn it on. |
| `n_jobs=-1` | Use all cores | Free. Trees are independent — this is embarrassingly parallel. |

> ⚠️ **"More trees → no overfitting (safe to increase)" is true for Random Forest and *false* for
> boosting.** This is the single most important asymmetry in the whole lecture, and §8 turns on it.
> More bagged trees drive the estimator toward its own expectation — you converge, you don't
> diverge. More boosted rounds keep fitting residuals, including the residuals that are pure noise.
> Candidates who carry "more trees is always safe" across from RF to XGBoost get burned.

```interactive
type: simulator
title: The correlation floor
concept: Var(mean) = ρσ² + (1−ρ)σ²/B, and why Random Forest subsamples features
control: Two sliders — number of trees B (1 to 500) and tree correlation ρ (0 to 1)
observe: A variance bar splitting into its two components, with the ρσ² floor drawn as a horizontal line the total can approach but never cross; a scatter of individual tree predictions tightening as ρ falls
insight: Adding trees only ever drains the second term; the floor is set by correlation alone — so once B is a few hundred, the ONLY remaining lever is max_features
fallback: The five-row table in §6.2 computes the same thing at B = 100 for ρ ∈ {0, 0.1, 0.5, 0.9, 1.0}.
```

---

## 7. Ensemble learning II: boosting

Slide `[f23]` (19:35), with a KEY TAKEAWAY bar at the bottom:

> **Idea:** Sequentially train models, each correcting predecessor's errors
> **AdaBoost:** Increase weights on misclassified samples → focus on hard examples · Combine weak
> learners with weighted vote; sensitive to noise/outliers
> **Gradient Boosting:** Each new model fits the residual errors (negative gradient of loss) ·
> Additive model: $F(x) = \sum_m \alpha_m h_m(x)$; trees typically shallow (depth 3–8)
> **XGBoost / LightGBM / CatBoost:** Regularized gradient boosting with parallelized split finding ·
> Handle missing values, categorical features natively · State-of-the-art for tabular data
> **Key Hyperparameters:** `learning_rate` (shrinkage): Lower → more regularization, needs more
> trees · `n_estimators`: More rounds → risk of overfitting (unlike Random Forests!) ·
> `max_depth`: Shallow trees preferred (3-8) — boosting handles complexity through iterations
> **Boosting reduces both bias and variance; use early stopping to prevent overfitting**
>
> **KEY TAKEAWAY** — Sequential trees fit residuals; XGBoost / LightGBM are the tabular default —
> always pair with early stopping.

### 7.1 The reframing: boosting attacks bias

Bagging takes low-bias, high-variance trees and kills the variance. **Boosting does the reverse.** It
takes *deliberately underfit* trees — depth 3 to 8, sometimes single splits — and stacks them until
their combined bias is gone.

> **Weak learner** — a model only slightly better than random guessing.
>
> *In everyday words:* a rule of thumb that's right 55% of the time when coin-flipping gets 50%.
>
> *Concretely:* a **decision stump** — a tree of depth 1, asking exactly one question.
>
> *Why it exists:* the astonishing theoretical result behind all boosting is that **you can combine
> arbitrarily many weak learners into an arbitrarily strong one.** Being slightly-better-than-random
> is enough, provided you keep changing what "the problem" is between rounds. That result
> (Schapire, 1990; Freund & Schapire, 1997) is why boosting exists at all.

That last clause is the mechanism. Each round, the problem is *redefined* so that the previous
learner's mistakes are what matters. AdaBoost redefines it by **reweighting rows**; gradient boosting
redefines it by **changing the target to the residuals**. Two implementations of one idea.

### 7.2 AdaBoost, and a result that surprises people

The procedure:

1. Start with uniform weights $w_i = 1/n$ on every training row.
2. For $m = 1 \dots M$:
   a. Fit a weak learner $h_m$ to the *weighted* data.
   b. Compute its weighted error $\varepsilon_m = \sum_{i \text{ wrong}} w_i$.
   c. Compute its vote $\alpha_m = \tfrac12 \ln\!\frac{1-\varepsilon_m}{\varepsilon_m}$.
   d. Reweight: $w_i \leftarrow w_i e^{+\alpha_m}$ if wrong, $w_i e^{-\alpha_m}$ if right. Renormalise.
3. Predict $\operatorname{sign}\!\left(\sum_m \alpha_m h_m(x)\right)$.

**Why $\alpha_m$ has that form.** Read it as a log-odds of being right. If $\varepsilon = 0.5$
(useless), $\alpha = \tfrac12\ln 1 = 0$ — no vote. If $\varepsilon \to 0$ (perfect),
$\alpha \to +\infty$ — dominates the vote. If $\varepsilon > 0.5$ (worse than random), $\alpha < 0$ —
the learner is **inverted** and still contributes, because a reliably wrong classifier is a reliably
right one flipped.

### 🧪 Worked example — one AdaBoost round, and the 50/50 result

Ten training rows, uniform weights $w_i = 0.1$. The first stump misclassifies **3** of them.

**Step 1 — weighted error.** $\varepsilon_1 = 3 \times 0.1 = 0.300$

**Step 2 — the vote.**
$$\alpha_1 = \tfrac12 \ln\frac{1-0.3}{0.3} = \tfrac12 \ln(2.3333) = \tfrac12(0.8473) = \mathbf{0.4236}$$

**Step 3 — reweight.** $e^{\alpha_1} = e^{0.4236} = 1.5275$ and $e^{-\alpha_1} = 0.6547$.

- 3 misclassified rows: $0.1 \times 1.5275 = 0.15275$ each
- 7 correct rows: $0.1 \times 0.6547 = 0.06547$ each

**Step 4 — normalise.** Total $= 3(0.15275) + 7(0.06547) = 0.45826 + 0.45828 = 0.91654$.

- Misclassified: $0.15275 / 0.91654 = \mathbf{0.16667} = 1/6$
- Correct: $0.06547 / 0.91654 = \mathbf{0.07143} = 1/14$

**Step 5 — the punchline.** Total weight now on the misclassified set:

$$3 \times \tfrac16 = \mathbf{0.500}$$

and on the correct set, $7 \times \tfrac{1}{14} = \mathbf{0.500}$.

**Exactly half.** And this is not a coincidence of the numbers I chose — it is always true:

$$\text{misclassified weight after update} = \varepsilon\, e^{\alpha} = \varepsilon\sqrt{\tfrac{1-\varepsilon}{\varepsilon}} = \sqrt{\varepsilon(1-\varepsilon)}$$
$$\text{correct weight after update} = (1-\varepsilon)\,e^{-\alpha} = (1-\varepsilon)\sqrt{\tfrac{\varepsilon}{1-\varepsilon}} = \sqrt{\varepsilon(1-\varepsilon)}$$

Identical. ∎

> 💡 **What this means.** After reweighting, the stump you just trained has weighted error **exactly
> 0.5** on the new distribution — it is now *precisely useless*. AdaBoost has reshaped the problem so
> that the next learner gains nothing by repeating the last one. That is the mechanism that forces
> diversity, and it is a genuinely elegant piece of design. It is also the single best thing to say
> when an interviewer asks how AdaBoost avoids building the same stump twice.

> ⚠️ **And it is also the weakness.** "Sensitive to noise/outliers" `[f23]` follows directly. A
> mislabelled row can never be classified correctly, so it is wrong every round, so its weight grows
> **exponentially**. After 50 rounds a single corrupted label can carry a large share of the total
> weight and bend the entire ensemble toward fitting it. §8's label-noise experiment measures this.

### 7.3 Gradient boosting: gradient descent in function space

AdaBoost's reweighting is clever but hard to generalise past classification. Friedman's (2001)
reframing generalises to any differentiable loss.

**The model is additive:**

$$F_M(x) = \sum_{m=1}^{M} \alpha_m h_m(x)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $F_M$ | "F sub M" | The ensemble after $M$ rounds. |
| $h_m$ | "h sub m" | The $m$-th weak learner (a shallow tree). |
| $\alpha_m$ | "alpha sub m" | Its weight. In practice a fixed **learning rate** $\eta$. |
| $M$ | "M" | Number of boosting rounds = `n_estimators`. |

**The algorithm:**

1. $F_0(x) = $ a constant (the value minimising the loss — the mean for squared error, the log-odds
   of the base rate for log loss).
2. For $m = 1 \dots M$:
   a. Compute **pseudo-residuals** for every row:
      $\displaystyle r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$
   b. Fit a shallow tree $h_m$ to predict $r_{im}$ from $x_i$ (a plain regression tree, whatever the
      original task).
   c. Update: $F_m(x) = F_{m-1}(x) + \eta\, h_m(x)$.

**Step (a) is the whole idea.** From Prerequisite 6, for squared loss the negative gradient *is* the
residual — so "fit the residuals" is the squared-loss special case. But the general form works for
any differentiable loss: with log loss the pseudo-residual is $y_i - p_i$ (the same beautiful
cancellation Part 2 §6 derived for BCE); with absolute loss it is $\operatorname{sign}(y_i - F_i)$.
**Change the loss, and only step (a) changes.** Everything else — fit a regression tree, take a
small step — is untouched. That modularity is why gradient boosting handles regression,
classification, ranking, and survival analysis with the same code.

> 💡 **The name, unpacked.** Ordinary gradient descent: $w \leftarrow w - \eta \nabla_w L$, moving in
> parameter space. Gradient boosting: $F \leftarrow F + \eta h$ where $h \approx -\nabla_F L$, moving
> in **function** space. The trees are how you represent a step direction when your "parameter" is an
> entire function.

### 🧪 Worked example — one gradient boosting round on real numbers

Four rows, feature $x = [1,2,3,4]$, target $y = [2,4,6,8]$. Squared loss.

**Step 0 — initialise.** $F_0 = \bar y = \frac{2+4+6+8}{4} = \mathbf{5}$ for every row.

Initial MSE: $\frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = \frac{9+1+1+9}{4} = \mathbf{5.00}$

**Step 1 — pseudo-residuals.** $r = y - F_0 = [-3, -1, +1, +3]$

**Step 2 — fit a stump** to predict $r$ from $x$. The best single split is $x \le 2$:
- Left leaf ($x \in \{1,2\}$): mean residual $= \frac{-3 + -1}{2} = -2$
- Right leaf ($x \in \{3,4\}$): mean residual $= \frac{1+3}{2} = +2$

So $h_1(x) = -2$ if $x\le2$, $+2$ otherwise.

**Step 3 — update with $\eta = 0.1$.**
$$F_1 = [5 - 0.2,\; 5 - 0.2,\; 5 + 0.2,\; 5 + 0.2] = [4.8,\; 4.8,\; 5.2,\; 5.2]$$

New residuals: $[-2.8,\, -0.8,\, +0.8,\, +2.8]$

New MSE: $\frac{7.84 + 0.64 + 0.64 + 7.84}{4} = \frac{16.96}{4} = \mathbf{4.24}$

**A 15.2% reduction in one round.**

**Now do it with $\eta = 1.0$ instead:**
$$F_1 = [3,\, 3,\, 7,\, 7], \quad r = [-1,\, +1,\, -1,\, +1], \quad \text{MSE} = \frac{1+1+1+1}{4} = \mathbf{1.00}$$

**An 80% reduction in one round** — five times faster.

> 💡 **So why ever use $\eta = 0.1$?** Because the fast version fitted this *particular* stump's
> conclusion at full strength, including whatever part of it was noise. Shrinkage means every round
> commits only a little, so no single tree's idiosyncrasies dominate, and the ensemble averages over
> many partially-right corrections. It is regularisation by *hesitation*. This is exactly the
> slide's "`learning_rate` (shrinkage): Lower → more regularization, needs more trees."

The notebook makes the trade visible `[f114]` (52:10), sweeping $\eta \in \{0.01, 0.05, 0.1, 0.3\}$
against `n_estimators` $\in \{10, 25, 50, 100, 200, 400\}$:

```
 test
  acc
 0.75 ┤        ╭─────●────────●───────●────●    lr=0.05, 0.1, 0.3 all plateau ≈0.75
 0.70 ┤    ╭──╯
 0.65 ┤  ╭─╯
 0.60 ┤ ╱                                        lr=0.01 needs ~200 trees just
 0.55 ┤╱                                         to reach where lr=0.1 is at 25
 0.50 ┤●
      └┬────┬────┬────┬─────┬─────┬
       10   50  100  200   300   400   n_estimators
```

**Read it.** At `n_estimators=10`, $\eta = 0.01$ scores about **0.50** — no better than predicting the
majority class — because ten steps of size 0.01 have barely moved off $F_0$. By 400 trees it has
caught up. Meanwhile $\eta = 0.3$ is at its plateau by 25 trees. **Learning rate and number of trees
are not independent knobs; they are one knob viewed twice.** Their product is roughly how far you
travel. This is why you tune them jointly, or fix $\eta$ small and let early stopping choose $M$.

### 7.4 The modern implementations

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Tree growth | Level-wise (depth-first, balanced) | **Leaf-wise** (best-first) | Symmetric / oblivious trees |
| Speed on wide data | Fast | **Fastest** — histogram binning + GOSS | Moderate |
| Categoricals | Needs encoding | Native | **Native, with ordered target statistics** |
| Missing values | Learns a default direction per split | Same | Same |
| Typical edge | The safe default | Large datasets | Many high-cardinality categoricals |

Three genuine advances over Friedman's original that the slide compresses into "regularized gradient
boosting with parallelized split finding":

1. **Explicit regularisation in the objective.** XGBoost adds $\gamma T + \tfrac12\lambda\|w\|^2$ —
   penalties on the number of leaves and the magnitude of leaf values — *into the split-finding
   criterion itself*, not as a post-hoc prune.
2. **Second-order optimisation.** XGBoost uses both the gradient and the Hessian of the loss, giving
   a Newton-style step rather than a plain gradient step. Faster and more stable convergence.
3. **Histogram-based splitting.** Instead of testing every distinct value of a continuous feature,
   bucket it into ~256 bins and test bin boundaries. Turns split-finding from $O(n\log n)$ into
   $O(n + \text{bins})$ per feature, with negligible accuracy loss. This is what sklearn's
   `HistGradientBoostingClassifier` also does, and why it is dramatically faster than the classic
   `GradientBoostingClassifier` on large data.

> ⚠️ **"Parallelized" needs care.** Boosting is **sequential by construction** — round $m$ needs
> round $m-1$'s residuals, so rounds cannot be parallelised. What *is* parallelised is split-finding
> *within* one tree: evaluating candidate features and thresholds across cores. Random Forest, by
> contrast, parallelises across whole trees, which is a far bigger win. Saying "XGBoost is parallel
> like Random Forest" in an interview is a real error.

### 7.5 🧪 The lecture's own boosting comparison

`[f111]` (51:59), on the student dataset:

```
        AdaBoost: train=0.762  test=0.744  time=1.1s
GradientBoosting: train=0.882  test=0.763  time=3.9s
HistGradientBoosting: train=0.999  test=0.755  time=13.5s

(XGBoost skipped: ModuleNotFoundError — this is fine)
```

**This little table teaches more than it looks.**

- **AdaBoost** has the smallest train–test gap (0.762 → 0.744, **1.8 points**) and the worst test
  score. It is *underfitting*: 200 stumps aren't enough capacity here.
- **GradientBoosting** has a moderate gap (0.882 → 0.763, **11.9 points**) and the **best test score
  of the three, 0.763**. Depth-3 trees with `learning_rate=0.1` is a well-balanced configuration.
- **HistGradientBoosting** has a catastrophic gap (0.999 → 0.755, **24.4 points**). It has memorised
  the training set almost perfectly. `max_iter=300` with `max_depth=6` is simply **too much capacity
  for 3,539 rows**, and it is *slower* here (13.5s) because histogram binning's overhead doesn't pay
  off until you have far more data.

> 💡 **The lesson is the one the KEY TAKEAWAY bar states: *always pair with early stopping*.**
> HistGradientBoosting at train=0.999 is the exact failure mode boosting has and Random Forest does
> not. Had `early_stopping=True` been set (it is the default in recent sklearn when $n$ is large
> enough), it would have halted where validation stopped improving and very likely beaten 0.755.
> The fastest algorithm here lost to the slowest because nobody told it when to stop.

> ⚠️ **Also note what did *not* happen: XGBoost didn't run.** The notebook wrapped it in
> `try/except` and printed `XGBoost skipped: ModuleNotFoundError — this is fine`. So the deck's
> claim that XGBoost/LightGBM are "state-of-the-art for tabular data" is **not demonstrated by this
> lecture's own experiment**. It is a well-supported claim in the literature (see *Going deeper*),
> but on the numbers actually shown, plain sklearn `GradientBoostingClassifier` at 0.763 was the
> best boosting result — and Random Forest at **0.769** beat all three boosters.

---

## 8. Bagging vs boosting: choosing between them

Two slides, `[f26]` (21:25) and `[f28]` (22:52), with a shared diagram.

> **Bagging (Random Forest):** Parallel training — trees are independent, easily distributed ·
> Primarily reduces variance; low risk of overfitting · Robust to noise and outliers · Best when:
> High-variance base learner, noisy data, need parallelism
> **Boosting (XGBoost / LightGBM):** Sequential training — each tree depends on previous errors ·
> Reduces both bias and variance; risk of overfitting without early stopping · Sensitive to
> noise/outliers (upweights misclassified → may fit noise) · Best when: Want maximum accuracy, clean
> data, willing to tune carefully

Then the decision guide `[f28]`:

> **Quick Decision Guide:**
> Noisy data + quick baseline → **Random Forest**
> Clean data + maximum performance → **XGBoost/LightGBM**
> Need feature importance + interpretability → **Single tree or RF**
> Structured/tabular data (industry standard) → **XGBoost with early stopping**
> **Both dominate tabular data; neural networks dominate images, text, sequence data**
>
> **KEY TAKEAWAY** — Bagging → noisy data, parallel, variance reduction. Boosting → clean data,
> sequential, maximum accuracy.

### 8.1 The comparison, consolidated

| | **Bagging / Random Forest** | **Boosting** |
|---|---|---|
| Trees are built | In parallel, independently | Sequentially, each on the last's errors |
| Base learner | **Deep** (low bias, high variance) | **Shallow**, depth 3–8 (high bias, low variance) |
| Primarily reduces | Variance | Bias (and variance too) |
| Data each tree sees | Bootstrap sample, random feature subset | All rows, reweighted or re-targeted |
| More estimators | Safe — converges | **Dangerous** — can overfit |
| Noise/outliers | Robust — a bad row is in only 63% of trees and gets outvoted | **Fragile** — a bad row is chased every round |
| Parallelism | Across trees (huge) | Within a tree only (modest) |
| Tuning burden | Low — defaults are good | High — `learning_rate` × `n_estimators` × `max_depth` interact |
| Free validation | **Yes** (OOB) | No |
| Typical ceiling | Very good | **Best available on tabular data** |

### 8.2 🧪 The label-noise experiment — the deck's claim, measured

The slide asserts bagging is "robust to noise" and boosting is "sensitive." The notebook tests it
`[f122]` (52:50) by flipping a random fraction of training labels and re-fitting both:

```
noise=0%    RF=0.769   GBM=0.763
noise=5%    RF=0.783   GBM=0.754
noise=10%   RF=0.773   GBM=0.758
noise=20%   RF=0.775   GBM=0.753
noise=30%   RF=0.774   GBM=0.737
```

```
 test
 acc
0.78 ┤   ●      ●─────●─────●──────●        Random Forest — flat
     │  ╱ ╲    ╱
0.77 ┤ ●   ●──╯
0.76 ┤■
0.75 ┤  ╲   ■────■──────■                   Gradient Boosting — decays
0.74 ┤   ■─╯             ╲
0.73 ┤                    ■
     └┬─────┬─────┬──────┬──────┬
      0     5    10     20     30   label noise (%)
```

**Read it honestly, including the wobble.**

- **Random Forest is flat.** From 0.769 at 0% noise to 0.774 at 30% noise, it does not degrade at
  all. Corrupting *thirty percent of the training labels* cost it nothing measurable.
- **Gradient Boosting decays monotonically** after the first point: 0.763 → 0.754 → 0.758 → 0.753 →
  **0.737**. It loses 2.6 points, and the loss accelerates at the high end.
- **The crossover is immediate.** RF is behind at 0% (0.769 vs 0.763 — actually ahead here) and
  pulls further ahead at every subsequent level.

> ⚠️ **Be honest about the noise in the experiment itself.** RF at 5% noise (0.783) scores *higher*
> than at 0% (0.769) — a 1.4-point rise that cannot be a real effect of adding noise. That is
> **run-to-run variation on an 885-row test set.** One standard error on an accuracy of ~0.77 with
> $n=885$ is $\sqrt{0.77 \times 0.23 / 885} \approx 0.014$ — about **1.4 points**, exactly the size
> of the anomaly. So: RF's individual wiggles are noise, but its *flatness across the whole range*
> is real, and GBM's *2.6-point monotone decline* is about 2 standard errors and is probably real.
> Part 2 §18 made this point about metric uncertainty; here is where it bites.

**Why the mechanism produces this.** A mislabelled row is in ~63% of bootstrap samples, and in those
trees it distorts one leaf. The other ~37% of trees have never seen it, and majority vote drowns it
out. In boosting, a mislabelled row is *by construction* the row with the largest residual, so every
round targets it — and with AdaBoost's exponential reweighting, its influence compounds.

> 💡 **The decision rule that follows.** *How clean are your labels?* is a better guide to
> bagging-vs-boosting than *how much accuracy do I need?* If labels come from human annotation,
> user-reported categories, or noisy proxies (a click as a proxy for satisfaction), start with
> Random Forest. If labels are mechanically derived and near-certain (did the payment charge back?
> did the package arrive late?), boosting will win.

### 8.3 The last line, which is the most important one

> **Both dominate tabular data; neural networks dominate images, text, sequence data.**

This is true and worth saying plainly, because it cuts against a lot of noise. On structured
tabular data — rows and columns, mixed types, thousands to millions of rows — **gradient-boosted
trees still beat deep learning**, consistently, as of the current literature. Grinsztajn et al.
(2022) benchmarked this carefully and found tree ensembles ahead across a wide range of tabular
tasks, attributing it to three properties of tabular data: irregular, non-smooth target functions;
uninformative features that trees ignore and networks must learn to ignore; and no
rotation-invariance, which trees exploit and MLPs are penalised by.

> 🎯 **Interview.** *"Why not just use a neural network for everything?"* The strong answer names the
> three structural reasons above and then adds the operational ones: a `GradientBoostingClassifier`
> trains in seconds on a laptop, needs no GPU, no scaling, no architecture search, gives you feature
> importances for free, and has three hyperparameters that a practitioner can reason about. For a
> tabular problem at Amazon scale, that is not a compromise — it is the right engineering call.
>
> 🔬 **Where it's genuinely open:** TabPFN, FT-Transformer, and the tabular-foundation-model line of
> work are actively contesting this, particularly in the small-data regime. It's a live area, not a
> settled one.

---

## 9. Multi-class classification strategies

Slide `[f32]` (26:29). The problem: logistic regression and SVM are **binary** algorithms. Your
target has three classes. Three ways to bridge the gap.

### 9.1 One-vs-Rest (OvR)

> Train K binary classifiers (one per class vs. all others) · Predict class with highest confidence;
> simple, scalable · Default in sklearn for LogReg, SVM; works for any binary classifier

For $K=3$ classes {Dropout, Enrolled, Graduate}, train three models:

| Model | Positive class | Negative class |
|---|---|---|
| 1 | Dropout | Enrolled + Graduate |
| 2 | Enrolled | Dropout + Graduate |
| 3 | Graduate | Dropout + Enrolled |

At prediction time, run all three and take the **argmax of the confidence scores**.

- **Cost:** $K$ models, each trained on all $n$ rows. Linear in $K$. Scales to thousands of classes.
- **The catch:** each classifier sees an artificially imbalanced problem. Model 2 above sees 635
  positives against 2,904 negatives — an 18/82 split — even though the original problem is only
  moderately imbalanced. With $K = 100$ classes, every sub-problem is 1%-positive.
- **The other catch:** the three models are trained independently, so their scores are **not on a
  comparable scale**. Comparing them by argmax is a heuristic, not a principled decision rule.

### 9.2 One-vs-One (OvO)

> Train K(K-1)/2 classifiers (one per pair); predict by majority vote · Each classifier trained on
> less data — better for SVM (kernel computation) · For K>10 classes, becomes expensive — prefer OvR
> or Softmax

For $K=3$: $\frac{3 \times 2}{2} = 3$ models — Dropout vs Enrolled, Dropout vs Graduate, Enrolled vs
Graduate. Each is trained **only on the rows belonging to its two classes**.

**Why "better for SVM (kernel computation)" is a real and specific argument.** Kernel SVM training is
roughly $O(n^2)$ to $O(n^3)$ (Part 2 §23) because it must form the kernel matrix over all pairs.
Now compare, with $n$ rows split evenly over $K$ classes:

- **OvR:** $K$ problems each of size $n$ → cost $\propto K \cdot n^2$
- **OvO:** $\frac{K(K-1)}{2}$ problems each of size $\frac{2n}{K}$ → cost $\propto \frac{K^2}{2}\cdot\frac{4n^2}{K^2} = 2n^2$

**OvO's total cost doesn't grow with $K$ at all** in this accounting, while OvR's grows linearly.
For a superlinear base learner, many small problems beat few large ones. That is why sklearn's
`SVC` uses OvO internally even though `LogisticRegression` uses OvR.

For a **linear-time** base learner the argument reverses, which is why "for K>10 classes, becomes
expensive" — at $K=100$ you are training 4,950 models, and the bookkeeping alone dominates.

### 9.3 Softmax (multinomial logistic regression)

> The formula says: **score every class, exponentiate the scores so they're all positive, then divide
> by the total so they sum to one.**

$$P(y = k \mid x) = \frac{e^{\,w_k^\top x}}{\sum_j e^{\,w_j^\top x}}$$

| Symbol | Read it as | What it means |
|---|---|---|
| $k$ | "kay" | The class we're computing the probability of. |
| $w_k$ | "w sub k" | The weight vector for class $k$. One per class. |
| $w_k^\top x$ | "w-k transpose x" | The raw score (**logit**) for class $k$: $\sum_j w_{kj}x_j$. |
| $e^{(\cdot)}$ | "e to the" | Exponential. Forces positivity. |
| $\sum_j$ | "sum over j" | Over **all** classes, including $k$ itself. |

The slide's own gloss is exactly right:

> Intuition: exp() ensures positive; division normalizes to probability distribution

And its three other claims:
- **Native multi-class: single model, end-to-end trainable** — one optimisation, all classes fitted
  jointly against one loss.
- **Standard output layer in neural networks** — this is where you have met it before.
- **Outputs sum to 1** — by construction: the numerators are exactly the terms of the denominator.

**Why joint training matters.** OvR trains three models that never see each other. Softmax trains one
model whose loss couples the classes: pushing up $P(\text{Graduate})$ mechanically pushes down
$P(\text{Dropout})$, because they share a denominator. The probabilities are therefore *calibrated
against each other* in a way OvR's independent scores never are.

📚 **Background the slide assumed — two properties from Part 2 §7.** Softmax is **shift-invariant**:
adding a constant $c$ to every logit changes nothing, since $e^{z_k + c}/\sum_j e^{z_j+c} = e^c
e^{z_k} / (e^c \sum_j e^{z_j})$. Implementations exploit this by subtracting $\max_j z_j$ before
exponentiating, so nothing overflows — the **log-sum-exp trick**. And softmax with $K=2$ reduces
exactly to the sigmoid.

### 🧪 Worked example — softmax by hand

Logits for one student: $z = [z_{\text{Dropout}}, z_{\text{Enrolled}}, z_{\text{Graduate}}] = [1.2,\ 0.4,\ 2.0]$.

**Step 1 — subtract the max for stability.** $\max = 2.0$, so $z' = [-0.8,\ -1.6,\ 0.0]$.

**Step 2 — exponentiate.**
$e^{-0.8} = 0.4493$, $e^{-1.6} = 0.2019$, $e^{0} = 1.0000$

**Step 3 — sum.** $0.4493 + 0.2019 + 1.0000 = 1.6512$

**Step 4 — divide.**
$$P(\text{Dropout}) = \tfrac{0.4493}{1.6512} = \mathbf{0.272},\quad
P(\text{Enrolled}) = \tfrac{0.2019}{1.6512} = \mathbf{0.122},\quad
P(\text{Graduate}) = \tfrac{1.0000}{1.6512} = \mathbf{0.606}$$

Check: $0.272 + 0.122 + 0.606 = 1.000$ ✓. Prediction: **Graduate**.

Note the gap between logits 1.2 and 2.0 — only 0.8 — became a probability ratio of 2.2×. Softmax
amplifies differences exponentially, which is why a small change in logits can flip a confident
prediction.

### 9.4 🧪 The lecture's own three-way comparison — and computing macro-F1 by hand

`[f124]` (53:03) runs all three on the student data and prints three confusion matrices.

```
        OvR (default)            OvO                   Softmax (multinomial)
   acc=0.757 macro-F1=0.657  acc=0.774 macro-F1=0.695   acc=0.763 macro-F1=0.677

           D    E    G           D    E    G              D    E    G
    D  │ 216   25   43  │  D  │ 218   26   40  │   D  │ 217   26   41  │
    E  │  45   42   72  │  E  │  39   58   62  │   E  │  43   52   64  │
    G  │  15   15  412  │  G  │  13   20  409  │   G  │  15   21  406  │
       (rows = true, columns = predicted)
```

**First, verify the accuracy** — never trust a printed metric you haven't checked. Accuracy is the
diagonal over the total, and the total is 885 test rows:

$$\text{acc}_{\text{OvR}} = \frac{216 + 42 + 412}{885} = \frac{670}{885} = \mathbf{0.7571}\ ✓$$
$$\text{acc}_{\text{OvO}} = \frac{218 + 58 + 409}{885} = \frac{685}{885} = \mathbf{0.7740}\ ✓$$
$$\text{acc}_{\text{Softmax}} = \frac{217 + 52 + 406}{885} = \frac{675}{885} = \mathbf{0.7627}\ ✓$$

Row sums check too: OvR Dropout row $216+25+43 = 284$ ✓ (the test set has 284 Dropouts `[f79]`),
Enrolled $45+42+72 = 159$ ✓, Graduate $15+15+412 = 442$ ✓.

**Now compute OvR's macro-F1 from scratch.** For each class: TP is the diagonal cell, FP is the rest
of its *column*, FN is the rest of its *row*.

*Dropout:* TP $=216$, FP $= 45+15 = 60$, FN $= 25+43 = 68$
$$P = \tfrac{216}{276} = 0.7826,\quad R = \tfrac{216}{284} = 0.7606,\quad F_1 = \tfrac{2(0.7826)(0.7606)}{0.7826+0.7606} = \tfrac{1.1905}{1.5432} = \mathbf{0.7714}$$

*Enrolled:* TP $=42$, FP $= 25+15 = 40$, FN $= 45+72 = 117$
$$P = \tfrac{42}{82} = 0.5122,\quad R = \tfrac{42}{159} = 0.2642,\quad F_1 = \tfrac{2(0.5122)(0.2642)}{0.7764} = \tfrac{0.2707}{0.7764} = \mathbf{0.3486}$$

*Graduate:* TP $=412$, FP $= 43+72 = 115$, FN $= 15+15 = 30$
$$P = \tfrac{412}{527} = 0.7818,\quad R = \tfrac{412}{442} = 0.9321,\quad F_1 = \tfrac{2(0.7818)(0.9321)}{1.7139} = \tfrac{1.4574}{1.7139} = \mathbf{0.8503}$$

$$\boxed{\text{macro-F1} = \frac{0.7714 + 0.3486 + 0.8503}{3} = \frac{1.9703}{3} = \mathbf{0.6568}}$$

Which is the printed **0.657** ✓.

**And OvO's, the same way:**

*Dropout:* TP$=218$, FP$=52$, FN$=66$ → $P = 0.8074$, $R = 0.7676$, $F_1 = \mathbf{0.7870}$
*Enrolled:* TP$=58$, FP$=46$, FN$=101$ → $P = 0.5577$, $R = 0.3648$, $F_1 = \mathbf{0.4411}$
*Graduate:* TP$=409$, FP$=102$, FN$=33$ → $P = 0.8004$, $R = 0.9253$, $F_1 = \mathbf{0.8584}$

$$\text{macro-F1} = \frac{0.7870+0.4411+0.8584}{3} = \frac{2.0865}{3} = \mathbf{0.6955} \quad ✓\ (0.695)$$

### 💡 What those two calculations actually reveal

Line the per-class F1 scores up:

| Class | Test rows | OvR F1 | OvO F1 | Change |
|---|---|---|---|---|
| Dropout | 284 | 0.771 | 0.787 | +0.016 |
| **Enrolled** | **159** | **0.349** | **0.441** | **+0.092** |
| Graduate | 442 | 0.850 | 0.858 | +0.008 |
| **macro-F1** | | **0.657** | **0.695** | **+0.038** |

**Essentially the entire OvO advantage is on the minority class.** Dropout and Graduate barely move.
Enrolled jumps 9.2 points, and because macro-F1 weights all three equally, that one class drags the
headline number up by 3.8.

**Why?** OvR's Enrolled classifier faced 635 positives against 2,904 negatives and learned to mostly
say "no" — recall of just **0.264**, meaning it misses **three-quarters of all Enrolled students**.
OvO's Enrolled-vs-Dropout and Enrolled-vs-Graduate classifiers each faced a much more balanced
problem, and recall rose to 0.365. Still poor, but 38% better.

> 💡 **Key insight.** OvR's structural weakness — it manufactures imbalance — is not an abstraction.
> It cost 9 points of F1 on the class this dataset most needs help with. And notice that plain
> **accuracy would have told you almost nothing**: 0.757 vs 0.774 is a 1.7-point difference that
> looks like noise. Macro-F1 exposed a 3.8-point difference driven by a real, explainable mechanism.
> This is why §10's "never plain accuracy" rule exists.

> ⚠️ **Don't over-generalise this to "OvO is better."** It won here because $K = 3$ and one class was
> starved. At $K = 50$, OvO's 1,225 models and its ambiguous-vote regions (where several classes tie)
> would likely lose to softmax. The slide's own recommendation is the right default:

> **When to use:** OvR (scalable default) → OvO (small K + SVM) → Softmax (neural nets, end-to-end)

To which I'd add one row the slide omits: **if your base learner already supports multi-class
natively — and trees, Random Forests, and gradient boosting all do — use it directly.** None of this
section applies to them. Trees handle $K$ classes in one model by storing a class distribution at
each leaf. OvR/OvO/softmax exist to rescue *binary-only* algorithms.

---

## 10. Handling class imbalance

Slide `[f35]` (29:26).

> **Problem:** When one class dominates (e.g., fraud detection: 99.9% legitimate, 0.1% fraud) ·
> Accuracy becomes misleading — 99.9% by predicting all negative!
> **Data-Level Strategies:** SMOTE: Generate synthetic minority examples by interpolating between
> neighbors · Random undersampling of majority class (fast, loses information) · Combination: SMOTE
> + Tomek links (clean boundary after oversampling) · **Warning: Apply SMOTE only on training set —
> never on validation/test (data leakage!)**
> **Algorithm-Level Strategies:** `class_weight='balanced'`: $w_k = N/(K\cdot n_k)$ · Cost-sensitive
> learning: higher misclassification cost for minority · Focal Loss: Down-weights easy examples,
> focuses on hard ones (from CV, works for tabular)
> **When to use what:** Mild imbalance (80/20) → class weights sufficient · Severe imbalance (99/1)
> → SMOTE + undersampling + specialized metrics
> **Evaluation: Use F1, PR-AUC, or balanced accuracy — never plain accuracy**

### 10.1 The accuracy paradox, restated

Part 2 §15 worked this in detail. The one-line version: in the fraud example, the classifier
`return "legitimate"` — which contains no model at all — scores **99.9% accuracy**. It also catches
zero fraud. Accuracy is a weighted average dominated by the majority class, so it can be maximised by
ignoring the minority entirely.

On this lecture's dataset, the majority class (Graduate) is 1,767/3,539 = **49.9%** of training rows,
so `always predict Graduate` scores about 0.499. That's the number every model in §16 must beat.

### 10.2 Algorithm-level: class weights, derived

$$w_k = \frac{N}{K \cdot n_k}$$

| Symbol | Read it as | What it means |
|---|---|---|
| $w_k$ | "w sub k" | The weight every row of class $k$ gets in the loss. |
| $N$ | "N" | Total training rows. |
| $K$ | "K" | Number of classes. |
| $n_k$ | "n sub k" | Number of training rows in class $k$. |

**Why that formula and not some other.** We want each *class as a whole* to carry equal total weight
in the loss. Class $k$ has $n_k$ rows, each weighted $w_k$, so its total contribution is $n_k w_k$.
Setting that equal for all classes, and requiring the grand total to stay $N$ so the loss scale
doesn't shift:

$$n_k w_k = \frac{N}{K} \quad\Longrightarrow\quad w_k = \frac{N}{K\,n_k} \qquad ∎$$

### 🧪 Worked example — the real class weights for this dataset

From `[f79]`: $N = 3539$, $K = 3$, $n_{\text{Dropout}} = 1137$, $n_{\text{Enrolled}} = 635$,
$n_{\text{Graduate}} = 1767$.

$$w_{\text{Dropout}} = \frac{3539}{3 \times 1137} = \frac{3539}{3411} = \mathbf{1.0375}$$
$$w_{\text{Enrolled}} = \frac{3539}{3 \times 635} = \frac{3539}{1905} = \mathbf{1.8578}$$
$$w_{\text{Graduate}} = \frac{3539}{3 \times 1767} = \frac{3539}{5301} = \mathbf{0.6676}$$

**Verify the design goal.** Total weight per class:

| Class | $n_k$ | $w_k$ | $n_k w_k$ |
|---|---|---|---|
| Dropout | 1,137 | 1.0375 | 1,179.6 |
| Enrolled | 635 | 1.8578 | 1,179.7 |
| Graduate | 1,767 | 0.6676 | 1,179.6 |
| **Total** | **3,539** | | **3,538.9 ≈ N** ✓ |

All three equal $N/K = 3539/3 = 1179.67$. ✓ The formula does exactly what it promised.

**Read the numbers.** One Enrolled student now counts as much as **2.78 Graduate students**
($1.8578 / 0.6676$). Getting one Enrolled prediction right is worth nearly three times as much to the
loss as getting one Graduate right. The model will trade Graduate accuracy for Enrolled accuracy —
and that is precisely the intent.

In sklearn this is one keyword: `class_weight='balanced'`, supported by `LogisticRegression`, `SVC`,
`DecisionTreeClassifier`, `RandomForestClassifier`. For gradient boosting, pass `sample_weight` to
`.fit()`, or use `scale_pos_weight` in XGBoost.

### 10.3 Data-level: SMOTE, and its trap

> **SMOTE** (Synthetic Minority Over-sampling TEchnique) — create new minority-class rows by
> interpolating between a minority point and one of its minority-class nearest neighbours.
>
> *In everyday words:* rather than photocopying rare examples (which teaches nothing new), invent
> plausible ones *between* the rare examples you have.
>
> *Concretely:* take an Enrolled student $A$ with `[admission_grade=130, age=19]` and a neighbouring
> Enrolled student $B$ with `[admission_grade=140, age=23]`. Pick a random $\lambda \in [0,1]$, say
> 0.4. The synthetic point is $A + \lambda(B - A) = [130 + 0.4(10),\ 19 + 0.4(4)] = [134,\ 20.6]$.
>
> *Why it exists:* plain random oversampling duplicates rows, so the model sees the exact same point
> many times and overfits to it — it memorises those specific points rather than the region they
> occupy. SMOTE fills in the region.

**The three failure modes SMOTE has, which the slide doesn't mention:**

1. **It interpolates in feature space, including across categoricals.** Interpolating between
   `course_code = 3` and `course_code = 9` gives `course_code = 5.4`, which is meaningless. (Use
   `SMOTENC` for mixed types.)
2. **It can interpolate across the class boundary.** If a minority point sits inside a majority
   region, SMOTE happily generates synthetic minority points deeper into majority territory,
   *corrupting* the boundary. This is what "SMOTE + Tomek links" fixes — Tomek links identify pairs
   of opposite-class nearest neighbours and delete them, cleaning the boundary after oversampling.
3. **It degrades in high dimensions.** "Nearest neighbour" becomes meaningless as $d$ grows (Part 2
   §22's curse of dimensionality), so the interpolation partners are effectively random.

### ⚠️ The leakage warning is the most important line on the slide

> **Warning: Apply SMOTE only on training set — never on validation/test (data leakage!)**

Here is exactly what goes wrong, because "leakage" is too abstract to act on.

**The wrong order:**
```
1. SMOTE the whole dataset          ← synthetic points created from ALL rows
2. train_test_split
3. train, evaluate → glorious scores, model fails in production
```

Suppose real minority row $R$ ends up in the **test** set. But in step 1, SMOTE used $R$ to
manufacture synthetic rows, and some of those landed in the **training** set. Those synthetic rows
are literally $R + \lambda(\text{neighbour} - R)$ — they contain $R$'s coordinates. The model has
therefore seen (a blend of) a test point during training. Test performance is inflated, sometimes
enormously, and it is inflated *most* on the minority class — the one you were trying to measure.

**The right order:**
```
1. train_test_split (stratified)
2. SMOTE the TRAINING set only
3. train, evaluate on the untouched test set
```

**The right order, done properly:** put SMOTE inside an `imblearn.pipeline.Pipeline`, so that
cross-validation re-runs it inside each fold. If you SMOTE once before a 5-fold CV loop, you have
leaked across folds even if you never touched the test set. §15 is about exactly this class of bug.

> 💡 **Before you reach for SMOTE at all, remember Part 2 §16: threshold moving is free.** Your model
> already outputs a probability. Lowering the decision threshold from 0.5 to 0.2 trades precision
> for recall on the minority class **without retraining anything, without inventing data, and
> without any leakage risk**. Try that first. Every time. SMOTE and class weights change *what the
> model learns*; the threshold changes *how you act on it*, and only the latter is reversible in
> production.

### 10.4 Focal loss

> **Focal Loss:** Down-weights easy examples, focuses on hard ones (from CV, works for tabular)

Standard cross-entropy for one row is $-\log(p_t)$, where $p_t$ is the predicted probability of the
*true* class. Focal loss multiplies it by a modulating factor:

$$\text{FL}(p_t) = -(1-p_t)^{\gamma}\log(p_t)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $p_t$ | "p sub t" | Predicted probability of the correct class. |
| $\gamma$ | "gamma" | Focusing parameter, typically 2. $\gamma = 0$ gives plain cross-entropy. |
| $(1-p_t)^{\gamma}$ | "the modulating factor" | Near 0 for confident-correct rows; near 1 for badly-wrong rows. |

**Worked, with $\gamma = 2$:**

| $p_t$ | $-\log p_t$ (CE) | $(1-p_t)^2$ | Focal loss | Reduction |
|---|---|---|---|---|
| 0.99 (easy, right) | 0.0101 | 0.0001 | 0.0000010 | **10,000×** |
| 0.90 | 0.1054 | 0.01 | 0.001054 | 100× |
| 0.50 | 0.6931 | 0.25 | 0.1733 | 4× |
| 0.10 (hard, wrong) | 2.3026 | 0.81 | 1.8651 | **1.2×** |

**Read the last column.** Easy examples are silenced by four orders of magnitude; hard examples are
barely touched. With 99.9% easy negatives, plain cross-entropy's total loss is dominated by rows the
model already gets right, and their combined gradient drowns out the rare positives. Focal loss
removes them from the conversation.

The slide's "(from CV)" is a correct attribution — it was introduced by Lin et al. (2017) for dense
object detection, where the background/foreground imbalance is roughly 1000:1. Its use on tabular
data is a reasonable transfer but less established.

> ⚠️ **Focal loss vs class weights are not the same axis.** Class weights reweight by **class
> membership** — every Enrolled row gets 1.86× regardless of difficulty. Focal loss reweights by
> **difficulty** — a badly-misclassified *majority* row keeps its full weight, and an easy minority
> row loses its. They compose: the original paper uses $\alpha$-balanced focal loss, which does both.

### 10.5 🧪 The lecture's own imbalance experiment — read the *right* column

`[f130]` (53:53). Four strategies, four metrics:

```
                        accuracy   balanced_acc   macro_F1   minority_F1
Baseline                  0.763       0.670        0.677       0.403
class_weight='balanced'   0.736       0.711        0.698       0.509
SMOTE                     0.737       0.708        0.698       0.509
SMOTE + class_weight      0.737       0.708        0.698       0.509
```

**Every rebalancing strategy made accuracy worse.** Baseline 0.763 → 0.736, a drop of **2.7 points**.
If accuracy were your metric, you would conclude that rebalancing hurt and revert it.

**Now read the last column.** Minority-class (Enrolled) F1 went **0.403 → 0.509**, a gain of
**10.6 points** — a 26% relative improvement in the model's ability to identify the class you built
this for. Balanced accuracy rose 4.1 points; macro-F1 rose 2.1.

$$\text{Accuracy: } -2.7\ \text{points} \qquad \text{Minority F1: } +10.6\ \text{points}$$

> 💡 **This single table is the entire argument for the slide's closing line — "Use F1, PR-AUC, or
> balanced accuracy — never plain accuracy."** The trade is real and it is *deliberate*: you gave up
> some correct Graduate predictions (there are 442 of them, so you can afford it) to buy correct
> Enrolled predictions (there are 159, and you were missing most of them). Whether that trade is
> right depends on what an Enrolled student is *worth* to you — which is a business question, not a
> modelling one. But you cannot even *see* the trade if accuracy is the only number on your
> dashboard.

**Three more things this table says that are easy to miss:**

1. **SMOTE and class weights performed identically** — 0.737/0.708/0.698/0.509 versus
   0.736/0.711/0.698/0.509. Differences of 0.001–0.003 on 885 test rows are far inside the ±0.014
   standard error. **They are the same result.** So take the one that is one keyword
   (`class_weight='balanced'`) over the one that is an extra dependency, a fitted transformer, and a
   leakage hazard.
2. **SMOTE + class_weight is byte-identical to SMOTE alone.** Stacking them bought exactly nothing.
   After SMOTE has equalised the class counts, `'balanced'` computes $w_k = N/(K n_k)$ on
   already-equal $n_k$ and returns $w_k = 1$ for everything. It is a **no-op by construction**, not
   a coincidence — and it's a good demonstration that stacking imbalance fixes is not additive.
3. **The imbalance here is mild** (Enrolled is ~18%), which is exactly the regime the slide says
   class weights suffice for. The result confirms the slide's own guidance: *mild imbalance (80/20)
   → class weights sufficient.*

```interactive
type: simulator
title: The imbalance trade, made visible
concept: Why accuracy and minority-class F1 move in opposite directions under rebalancing
control: A toggle across the four strategies (baseline, class_weight, SMOTE, both) and a slider for the minority-class fraction
observe: A 3×3 confusion matrix updating live, with four metric bars beside it — accuracy, balanced accuracy, macro-F1, minority F1 — and arrows showing which way each moved
insight: Rebalancing moves predictions OUT of the majority diagonal cell and INTO the minority one; accuracy counts the loss and ignores the gain, which is exactly why it is the wrong metric here
fallback: The four-row table in §10.5, and the −2.7 vs +10.6 comparison beneath it.
```

---

## 11. Cross-validation

Two deck slides, 14 and 15. Slide 14 is the one with the capture gap (see the note at the top); slide
15 is fully captured at `[f39]` (33:55).

### 11.1 K-fold — the mechanics

> ⚠️ Reconstructed from `[f37]`'s heading ("K-Fold Cross Validation:"), the deck's diagram (fully
> visible in `[f37]`), and the notebook's Section 7 restatement at `[f130]`. The deck's exact
> sub-bullet wording is unrecovered.

The diagram `[f37]` is unambiguous and worth transcribing exactly, because it *is* the algorithm:

```
                5-Fold Cross Validation
        Validation fold rotates across all K splits

            Chunk 1  Chunk 2  Chunk 3  Chunk 4  Chunk 5
   Iter 1 │  [Val]    Train    Train    Train    Train
   Iter 2 │  Train    [Val]    Train    Train    Train
   Iter 3 │  Train    Train    [Val]    Train    Train
   Iter 4 │  Train    Train    Train    [Val]    Train
   Iter 5 │  Train    Train    Train    Train    [Val]

   Train = K−1 folds → fit model     Val = 1 fold → score
              Final metric = mean of the K validation scores
```

**The procedure:** shuffle, cut the training data into $K$ equal chunks, then $K$ times: hold one
chunk out, fit on the other $K-1$, score on the held-out chunk. Average the $K$ scores.

**What you gain over a single train/validation split:**
- **Every row is used for validation exactly once**, and for training $K-1$ times. Nothing is wasted.
- The reported metric is a **mean of $K$ estimates**, so its variance is roughly $1/K$ of a single
  split's — you get an error bar for free (the standard deviation across folds).
- The estimate no longer depends on which particular rows you happened to put in your validation set.

**What it costs:** $K$ model fits instead of one. With $K=5$ that is 5× the compute, which is why
`cv=3` shows up in the notebook's `GridSearchCV` calls `[f141]`.

📚 **Background the slide assumed — stratified K-fold.** The notebook `[f130]` states it directly:

> **Stratified K-Fold** — preserves class distribution in each fold (default for classification).

Plain K-fold cuts randomly, so with 18% Enrolled students a fold might get 14% or 22% just by chance.
Stratified K-fold cuts *within each class* and reassembles, so every fold has ≈32% Dropout, ≈18%
Enrolled, ≈50% Graduate — matching the whole. sklearn does this automatically when you pass a
classifier and an integer `cv`.

### 🧪 Worked example — why stratification lowers variance, with the lecture's numbers

`[f134]` (54:16) measures it on the real dataset:

```python
kf  = KFold(n_splits=5, shuffle=True, random_state=SEED)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
kf_scores  = cross_val_score(rf_model, X_train_enc, y_train, cv=kf,  scoring='f1_macro')
skf_scores = cross_val_score(rf_model, X_train_enc, y_train, cv=skf, scoring='f1_macro')
```

```
K-Fold             macro-F1: 0.702 ± 0.023
Stratified K-Fold  macro-F1: 0.688 ± 0.015   ← lower variance
```

**Two things happened, and only one of them is the point.**

**The spread shrank: ±0.023 → ±0.015, a 35% reduction.** That is the advertised effect, and here is
why it happens arithmetically. Each fold holds out $3539/5 = 708$ rows. The expected number of
Enrolled in a fold is $708 \times 0.1794 = 127$. Under *random* splitting that count is
Binomial(708, 0.1794), with standard deviation

$$\sigma = \sqrt{708 \times 0.1794 \times 0.8206} = \sqrt{104.2} = 10.2$$

So a typical fold has $127 \pm 10$ Enrolled students, and a 2σ fold has 107 or 147 — a **±16%
swing** in how much of the hardest class each fold contains. Since macro-F1 weights Enrolled equally
with everything else, and Enrolled's F1 is both the lowest and the least stable, that swing
propagates straight into the fold scores. Stratified splitting makes the count 127 in *every* fold,
$\sigma = 0$, and that source of variance disappears entirely.

**The mean also dropped, 0.702 → 0.688.** Don't skip past this. Plain K-fold's estimate was
**optimistic by 1.4 points**. Some folds got lucky with an easy class mix and scored high; the
average of a noisy estimator over an asymmetric metric is not the same as the estimator at the
average. Stratification gives the more trustworthy number even though it is the *lower* one.

> 💡 **Key insight.** A lower-variance estimator is not merely more comfortable — it is **more
> honest**. With ±0.023, two models differing by 0.02 macro-F1 are indistinguishable. With ±0.015,
> you can start to tell them apart. Every hyperparameter decision downstream (§12) is a comparison
> between CV scores, so shrinking their error bars directly improves every one of those decisions.
> Use `StratifiedKFold`. It is the default for a reason and costs nothing.

### 11.2 Leave-One-Out CV

`[f39]`:

> **Leave-One-Out CV (LOOCV):** K = n (each sample is a test set once) · unbiased but high variance
> estimate · Only practical when n < 100 (computationally expensive)

Set $K = n$: each fold is a **single row**. You fit $n$ models, each on $n-1$ rows.

**"Unbiased":** each model trains on $n-1$ rows, essentially the full dataset, so its performance
estimates the performance of a model trained on all $n$ rows with almost no pessimistic bias. (5-fold
CV trains on only 80% of the data, so it slightly *underestimates* what the final model will do.)

**"But high variance"** — and this is the part people get backwards. There are two contributions:

1. Each individual fold's score is a single 0/1 outcome — maximally noisy per fold.
2. More subtly: the $n$ training sets are **almost identical** to each other (they differ by one row
   out of $n$). So the $n$ fitted models are highly correlated, and by Prerequisite 4, averaging
   highly correlated estimates barely reduces variance. You did $n$ fits and got the variance
   reduction of far fewer. **It is the $\rho$ floor from §6.2 again**, showing up in a completely
   different context.

**"Only practical when n < 100":** $n$ model fits. For this lecture's 3,539 training rows, LOOCV would
mean 3,539 Random Forest fits — versus 5 for stratified 5-fold. At roughly 1 second per fit that is
an hour instead of five seconds, for a *worse* estimate.

> 💡 **The default is 5 or 10, and that is not arbitrary.** $K=5$ or $10$ sits at the sweet spot:
> enough folds that each training set is ≥80% of the data (low bias), few enough that the training
> sets differ meaningfully (low correlation, so averaging actually helps), and cheap enough to run
> inside a hyperparameter search.

### 11.3 The time-series caveat

`[f39]`, and this is the one that ends careers:

> **Time-Series Caveat:** Never use random K-Fold for temporal data — future leaks into past! · Use
> `TimeSeriesSplit` (expanding window) to respect temporal order

**Why random K-fold is catastrophic on temporal data.** Shuffle a year of daily sales and put January
15th in the validation fold while January 14th and 16th are in training. Your model isn't forecasting
— it is **interpolating between two neighbouring days it has already seen**. In production it will
have to *extrapolate* from the past into a future it has never seen, and it will be far worse.
Random CV will not warn you. The gap between your CV score and your production score can be enormous
and entirely invisible until launch.

**`TimeSeriesSplit`, the expanding window:**

```
Random K-Fold (WRONG for temporal)        TimeSeriesSplit (CORRECT)
──────────────────────────────────        ─────────────────────────
 Fold 1  T V T T V T V T T V T V           Split 1  [TRAIN][VAL]· · · · · · ·
 Fold 2  V T T V T T T V T T V T           Split 2  [TRAIN TRAIN][VAL]· · · ·
 Fold 3  T T V T T V T T V T T V           Split 3  [TRAIN TRAIN TRAIN][VAL]·
         ↑ validation rows are             Split 4  [TRAIN TRAIN TRAIN TR][VAL]
           surrounded by training                    →  time  →
           rows from the future            Every validation fold is strictly
                                           AFTER every training row it's scored against
```

Training always ends before validation begins, and the training window **expands** with each split —
mirroring reality, where you retrain on everything up to today and predict tomorrow.

> ⚠️ **Three more temporal traps the slide doesn't mention but that interviewers ask about.**
> **(1) Gaps.** If your label takes 30 days to materialise (e.g. "did this customer churn within a
> month?"), you need a 30-day gap between train and validation, or the last 30 days of training
> contain labels that weren't knowable then. sklearn supports `TimeSeriesSplit(gap=...)`.
> **(2) Grouped data.** Multiple rows per customer? Use `GroupKFold` so a customer never appears in
> both train and validation, or the model can memorise the customer instead of learning the pattern.
> **(3) Both at once** is common and needs a custom splitter.

### 11.4 The KEY TAKEAWAY

> **KEY TAKEAWAY** — 5-fold stratified CV is default. Use `TimeSeriesSplit` for temporal data —
> random K-fold leaks the future.

Nothing to add. That sentence is the correct default and the correct exception.

---

## 12. Hyperparameter tuning

Slide `[f42]` (36:36).

> **Hyperparameters: Settings NOT learned from data (set before training)** · Examples:
> `learning_rate`, `C`, `K` in KNN, `max_depth`, `n_estimators`
> **Grid Search:** Exhaustively try all combinations; guaranteed to find best in grid · Cost:
> exponential in # parameters — impractical for >3 hyperparameters
> **Random Search:** Sample random combinations from distributions · **60 random trials finds a
> top-5% config with 95% probability (Bergstra & Bengio, 2012)** · Often outperforms grid search for
> same compute budget
> **Bayesian Optimization:** Build surrogate model (Gaussian Process) of objective; choose next point
> intelligently · Most sample-efficient — use for expensive models (large NNs, long training) ·
> Tools: Optuna, SageMaker Automatic Model Tuning, HyperOpt
> **Practical Tips:** Not all hyperparameters matter equally — `learning_rate` > `max_depth` >
> `n_estimators` · Combine with early stopping (`HalvingRandomSearchCV`) — eliminate bad configs fast
> · **Always tune with cross-validation, never on test set**

### 12.1 Parameter vs hyperparameter

📚 **Background, though Part 1 covered it.** A **parameter** is fitted by the learning algorithm from
data — the weights $w$ in logistic regression, the split thresholds in a tree. A **hyperparameter** is
a setting you choose *before* training that controls how the fitting happens — `max_depth`,
`learning_rate`, `C`, `K`. You cannot fit a hyperparameter by minimising training loss, because
training loss is almost always monotone in capacity: it would tell you `max_depth = ∞`. That is
precisely why §11 exists — hyperparameters must be chosen against **held-out** performance.

### 12.2 Why grid search dies

Grid search takes the Cartesian product of your specified values. With $p$ hyperparameters and $v$
values each, that is $v^p$ configurations, times $K$ folds:

| Hyperparameters | Values each | Configs | × 5-fold | At 10s/fit |
|---|---|---|---|---|
| 2 | 4 | 16 | 80 | 13 min |
| 3 | 4 | 64 | 320 | 53 min |
| 4 | 4 | 256 | 1,280 | 3.6 hours |
| 5 | 4 | 1,024 | 5,120 | **14 hours** |
| 6 | 4 | 4,096 | 20,480 | **2.4 days** |

That is the slide's "exponential in # parameters — impractical for >3 hyperparameters." And four
values per hyperparameter is a *coarse* grid — for `learning_rate` it barely resolves the useful
range.

### 12.3 The random search argument, derived properly

The slide states the headline: **"60 random trials finds a top-5% config with 95% probability."** That
is a real theorem with a one-line proof, and being able to produce it separates people who read the
slide from people who understood it.

**Setup.** Define "good" as being in the top $5\%$ of the configuration space by validation score. Draw
$n$ configurations independently and uniformly at random.

- $P(\text{one draw is not good}) = 0.95$
- $P(\text{all } n \text{ draws miss}) = 0.95^n$
- $P(\text{at least one is good}) = 1 - 0.95^n$

Solve for the $n$ that gets this to 0.95:

$$1 - 0.95^n \ge 0.95 \;\Longleftrightarrow\; 0.95^n \le 0.05 \;\Longleftrightarrow\; n \ge \frac{\ln 0.05}{\ln 0.95} = \frac{-2.9957}{-0.05129} = 58.4$$

$$\boxed{n = 59 \text{ trials, so } 60 \text{ is comfortably enough}} \qquad ∎$$

Check: $1 - 0.95^{60} = 1 - 0.0461 = \mathbf{0.954}$ ✓

**The crucial property:** *nothing in that derivation mentions the number of hyperparameters.* Sixty
trials gets you into the top 5% whether you are tuning 2 knobs or 20. Grid search's cost explodes
with dimension; random search's does not. **That** is the argument, and it is far stronger than "it
often works better."

### 💡 The second, deeper reason random search wins

Bergstra & Bengio's actual contribution was subtler than the probability argument, and it is worth
knowing:

**Hyperparameters have wildly unequal importance** — the slide's own "`learning_rate` > `max_depth` >
`n_estimators`." Now consider a 9-point grid over two hyperparameters, one important and one not:

```
     GRID SEARCH (3×3 = 9 trials)          RANDOM SEARCH (9 trials)
   unimportant →                          unimportant →
     ●     ●     ●                          ●   ●        ●
     ●     ●     ●     ↑ important           ●      ●  ●   ↑ important
     ●     ●     ●                        ●     ●      ● ●

   Only 3 DISTINCT values of the          9 DISTINCT values of the
   important hyperparameter were          important hyperparameter
   ever tried. The other 6 trials         were tried. Every trial
   were duplicates along the axis         explored new ground on the
   that mattered.                         axis that mattered.
```

Grid search **wastes trials on duplicate values of the axis that matters**. With 9 trials on a 3×3
grid you learn about only 3 settings of the important knob. With 9 random trials you learn about 9.
And you don't need to know in advance *which* knob matters — random search gets the benefit
automatically.

### 12.4 Bayesian optimisation

Grid and random search are **memoryless**: trial 50 is chosen without reference to trials 1–49.
Bayesian optimisation isn't.

1. Fit a **surrogate model** — usually a Gaussian Process — mapping hyperparameters → validation
   score, using all trials so far. It gives a predicted score *and an uncertainty* at every
   unexplored point.
2. Maximise an **acquisition function** (e.g. Expected Improvement) over that surrogate to pick the
   next point. This balances **exploitation** (try where the surrogate predicts high) against
   **exploration** (try where it is uncertain).
3. Evaluate there, update the surrogate, repeat.

**When it is worth it.** The surrogate machinery costs seconds per iteration. If a training run takes
10 seconds, that overhead is significant and random search's simplicity wins. If a training run takes
6 hours on 8 GPUs, spending 30 seconds to choose a *better* next point is obviously correct. Hence
the slide's "use for expensive models (large NNs, long training)."

Tools named: **Optuna** (open source, the current default in research), **SageMaker Automatic Model
Tuning** (managed, the AWS-native one — worth naming in an Amazon interview), **HyperOpt** (older).

### 12.5 🧪 The lecture's own search — and the result that undercuts the slide

`[f141]` (54:44):

```python
grid_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20],
               'max_features': ['sqrt', 'log2']}
grid = GridSearchCV(rf_estimator, grid_params, cv=3, scoring='f1_macro', n_jobs=-1)

rand_params = {'n_estimators': randint(50, 400), 'max_depth': [None, 5, 10, 20, 40],
               'max_features': ['sqrt', 'log2', 0.5, 0.7], 'min_samples_leaf': randint(1, 20)}
rand = RandomizedSearchCV(rf_estimator, rand_params, n_iter=25, cv=3,
                          scoring='f1_macro', n_jobs=-1, random_state=SEED)
```

```
GridSearchCV     ·  12 configs  best F1 = 0.684  best = {'max_depth': 20, 'max_features': 'log2',
                                                          'n_estimators': 100}
RandomizedSearch ·  25 configs  best F1 = 0.681  best = {'max_depth': 20, 'max_features': 'sqrt',
                                                          'min_samples_leaf': 1, 'n_estimators': 363}
```

**Verify the grid size:** $2 \times 3 \times 2 = 12$ ✓. With `cv=3`, that is 36 model fits. Random
search did 25 configs × 3 folds = 75 fits.

**Now read the outcome honestly. Random search *lost*: 0.681 vs 0.684.** It used twice the compute
and searched a richer space (4 hyperparameters instead of 3, continuous `n_estimators`, a fourth
`max_features` option) and came out 0.003 behind.

> ⚠️ **The deck says random search "often outperforms grid search for same compute budget," and its
> own experiment does not show that.** Do not pretend otherwise. But do explain it, because the
> explanation is the actual lesson:
>
> 1. **0.003 is noise.** Both searches used `cv=3` on 3,539 rows; §11 measured the fold-to-fold
>    standard deviation of this exact metric at **±0.015–0.023**. A 0.003 gap is one-fifth of one
>    standard error. These two searches found *statistically indistinguishable* configurations, and
>    reporting either as "the winner" is over-reading the data.
> 2. **12 configs is far too small for the theorem to apply.** The 60-trial result needs ~60 trials.
>    At 25 trials, $1-0.95^{25} = 0.72$ — a 28% chance of missing the top 5% entirely.
> 3. **The search space was tiny.** Random search's advantage is *dimensional*: it grows with the
>    number of hyperparameters. With 3–4 knobs and a well-chosen grid, grid search is perfectly
>    competitive. Its collapse happens at 5+.
> 4. **Both landed on `max_depth: 20`,** from different search strategies over different spaces.
>    When two independent searches agree on a hyperparameter, that is real signal about the problem.
>
> 💡 The honest takeaway is not "random beats grid." It is: **at this scale neither beats the other,
> and the difference between them is smaller than your measurement error — so use the cheaper one
> and spend the saved compute on more CV folds instead**, which would actually shrink the error bars
> and let you tell configurations apart.

### 12.6 Successive halving

> Combine with early stopping (`HalvingRandomSearchCV`) — eliminate bad configs fast

The idea: don't give every candidate a full evaluation. Start all $N$ candidates on a small resource
budget (a fraction of the data, or few trees), keep the top $1/\eta$, multiply the survivors'
budget by $\eta$, repeat.

```
Round 1:  81 configs ×  100 rows   →  keep top 27
Round 2:  27 configs ×  300 rows   →  keep top  9
Round 3:   9 configs ×  900 rows   →  keep top  3
Round 4:   3 configs × 2700 rows   →  keep top  1
```

Total cost ≈ 4 × 8,100 resource-units, versus 81 × 2,700 = 218,700 for evaluating everything fully —
roughly a **6.7× saving**, and the saving grows with $N$. The assumption is that a config that is
bad on 100 rows will be bad on 2,700; usually true, occasionally not (a config needing lots of data
to shine gets killed early), which is what the more careful **Hyperband** algorithm hedges against by
running several bracket sizes.

### 12.7 The most important line on the slide

> **Always tune with cross-validation, never on test set**

Part 1 §5 quantified why: pick the best of 30 configurations by test score and you inflate your
reported number by roughly 2 points of pure selection bias, because you have optimised against the
test set's specific noise. The test set is a **one-shot instrument**. Every time you look at it and
change something, it becomes a validation set — and you no longer have a test set at all.

The correct hierarchy:

```
   TRAIN  ─────────────►  fit model parameters (weights, splits)
     │
     └─ CV folds inside ─►  choose hyperparameters (§12), choose model family (§13)
                              ▲ you may look at these as often as you like

   TEST   ─────────────►  report ONE number, ONCE, at the very end
                              ▲ look at this twice and it is no longer a test set
```

---

## 13. Model selection strategies

Slide `[f45]` (39:00).

> **No Free Lunch Theorem: No single algorithm dominates all problems** · Always compare multiple
> model families — never assume one is best
> **Workflow:** 1. Define candidate models (different algorithms, complexities) · 2. Tune each using
> cross-validation on training data · 3. Compare best version of each on validation set · 4. Final
> evaluation on test set (only once!)
> **Selection Criteria:** Performance on the business-relevant metric (not just accuracy) ·
> Computational cost (training time AND inference latency) · Interpretability requirements
> (regulatory, debugging) · Robustness to distribution shift (concept drift in production)
> **Occam's Razor: Among similar performers, prefer the simpler model** · Simpler models generalize
> better, are easier to debug, deploy, and maintain
> **Practical starting point: XGBoost for tabular, LogReg for interpretable baseline — then compare**

### 13.1 No Free Lunch, stated correctly

The theorem (Wolpert, 1996) says: **averaged over all possible problems, every learning algorithm has
identical expected performance.** Any algorithm that beats another on one class of problems must lose
by exactly as much on some other class.

> ⚠️ **Do not over-read it.** The average is over *all logically possible* target functions,
> including the overwhelming majority that are pure noise with no structure whatsoever. Real-world
> problems are a vanishingly small, highly structured corner of that space — they have smoothness,
> locality, hierarchy, sparsity. Within that corner, some algorithms *are* reliably better, which is
> why "XGBoost for tabular" is sound advice and not a contradiction of the theorem.
>
> The correct practical reading is **not** "all algorithms are equally good." It is: **"there is no
> algorithm you can pick without looking at the data, so always compare at least two families."**

### 13.2 The workflow, and the trap in step 3

The four steps are correct and the ordering is load-bearing:

1. **Define candidates.** Deliberately span *different* families — a linear model, a single tree, a
   bagged ensemble, a boosted ensemble. Five variants of XGBoost is not a comparison.
2. **Tune each using CV on training data.** Each family gets a *fair* fight — an untuned XGBoost
   losing to a tuned Random Forest tells you nothing.
3. **Compare best-of-each on validation.**
4. **Final evaluation on test, once.**

> ⚠️ **The trap in step 3.** After steps 2 and 3 you have now selected on validation *twice* — once
> per family internally, once across families. The winner's validation score is optimistically
> biased, by exactly the multiple-comparisons mechanism from Part 1 §5. That is why step 4 exists
> and why it must use data untouched by both. If you report the validation score as your expected
> production performance, you will be wrong in a predictable direction.

### 13.3 The selection criteria — where most candidates lose the interview

The slide lists four, and only the first is about accuracy. This is the part worth memorising,
because "which model is best?" answered purely with a metric is a weak answer at Amazon.

| Criterion | The question it asks | When it decides the outcome |
|---|---|---|
| **Business metric** | Not accuracy — the metric tied to money or customer experience | Almost always. See §10: accuracy and minority-F1 disagreed by 13 points. |
| **Computational cost** | Training time **and inference latency** | Inference latency is the one people forget. A 5ms p99 budget in a search-ranking path rules out a 500-tree forest before accuracy is discussed. |
| **Interpretability** | Can you explain a decision to a regulator, a customer, or an on-call engineer at 3am? | Credit, lending, hiring, medical. Also anywhere debugging matters. |
| **Robustness to drift** | Does it degrade gracefully when the input distribution moves? | Anything seasonal, anything with adversaries (fraud), anything with a changing catalogue. |

📚 **Background the slide assumed — concept drift.** Part 1 §4 covered this: **data drift** is
$P(x)$ changing (your users shifted younger); **concept drift** is $P(y \mid x)$ changing (the same
customer profile now behaves differently). Drift matters here because model families differ in how
they fail under it. A linear model degrades smoothly and predictably. A deep tree ensemble can fail
sharply, because a feature value outside the range it was trained on lands in whatever leaf the last
threshold happens to send it to — **trees cannot extrapolate at all**. If your feature distributions
move, that is a genuine argument for the simpler model.

### 13.4 Occam's Razor, made concrete

> Among similar performers, prefer the simpler model. Simpler models generalize better, are easier to
> debug, deploy, and maintain.

"Similar" is doing real work in that sentence, and §11 told you how to define it: **similar means
within the error bars.** With CV standard deviation ±0.015, a model at 0.771 and a model at 0.769 are
*the same model* as far as your evidence goes. At that point every other criterion — latency,
interpretability, maintenance, drift-robustness — becomes the tiebreaker, and simplicity wins them
all.

The three practical reasons, spelled out:
- **Generalisation:** fewer parameters, less capacity to fit noise. The same argument as pruning (§4).
- **Debugging:** when predictions go wrong at 3am, a logistic regression's coefficients tell you
  which feature moved. A 500-tree forest tells you nothing without extra tooling.
- **Deploy and maintain:** fewer dependencies, smaller artefacts, faster inference, less to retrain,
  fewer ways to silently break.

### 13.5 The practical starting point

> **XGBoost for tabular, LogReg for interpretable baseline — then compare**

This is genuinely good advice and the reasoning is worth making explicit. You run **two** models
first, not one, because they answer different questions:

- **LogReg** answers *"how much signal is there in a linear combination of these features?"* — it is
  your **floor**, and it is interpretable, fast, and nearly impossible to get wrong.
- **XGBoost** answers *"how much more is there in interactions and nonlinearity?"* — it is your
  **ceiling**.

The **gap between them** is the most informative number you will get in the first hour of a project.
Small gap → your problem is essentially linear; ship the simple thing and go improve your features
instead. Large gap → interactions matter, and complexity is buying something real.

> ⚠️ **The lecture's own model-selection cell never ran.** `[f142]`/`[f143]` (55:11–55:29) show the
> `df_leader` leaderboard cell with an empty prompt (`[ ]:`) — it was written but not executed on
> screen, so there are no comparison numbers to quote here. The candidate set it *would* have
> compared is visible and is a good template: `LogReg`, `Decision Tree` (at the tuned `best_depth`),
> `Random Forest` (`rand.best_estimator_` from §12), `GradientBoosting`, `HistGradientBoosting` —
> scored by 5-fold `f1_macro` and sorted by CV mean. Note the careful detail in the loop:
> `X_for = Xtr_s if name == 'LogReg' else X_train_enc` — **only the linear model gets scaled
> features**, because trees don't need scaling (§3.3) and scaling them would be wasted work.

---

## 14. Probability calibration

Two deck slides, `[f50]` (41:46) and `[f53]` (44:38), introduced by a section title card
`[f47]` (39:05): *"Probability Calibration & Hands-on Demo."*

> **Problem:** Many classifiers output scores that are NOT true probabilities · Random Forest pushes
> predictions toward 0.5; SVM outputs are unbounded scores
> **Calibration:** If model says P(y=1)=0.8 for 100 samples, ~80 should actually be positive ·
> Reliability diagram: plot predicted prob (x) vs actual frequency (y); perfect = diagonal
> **Methods:** Platt Scaling: Fit logistic regression on model's raw scores → sigmoid correction ·
> Isotonic Regression: Non-parametric monotone fit; more flexible, needs more data

### 14.1 What calibration means, precisely

> **Calibrated** — of all the cases where the model said "80% chance", about 80% actually turned out
> positive. For every probability value, not just 0.8.
>
> *In everyday words:* a weather forecaster is calibrated if, across all the days they said "70%
> chance of rain," it rained on about 70% of them. They can be calibrated while being nearly
> useless (always say "50%" in a climate that rains half the time), and they can be highly
> informative while being badly calibrated.
>
> *Concretely:* take every test student the model gave $P(\text{Dropout}) \in [0.75, 0.85]$. Count
> what fraction actually dropped out. If it's 0.62, the model is **overconfident** in that band.
>
> *Why it exists:* because the number gets used. `predict_proba()[:,1] > 0.5` only needs the
> *ranking* to be right. But *"offer a retention scholarship when dropout risk exceeds 60%"* needs
> the **60** to mean something.

**The formal statement:** a model is calibrated if $P(y=1 \mid \hat p = p) = p$ for all $p \in [0,1]$.

### 14.2 Why specific models are miscalibrated in specific directions

The slide names two, and the mechanisms are different and both worth knowing.

**Random Forest pushes toward 0.5.** Its probability is the fraction of trees voting positive. For a
genuinely certain case, you'd need *all* 200 trees to agree — but each tree saw a different bootstrap
sample and a different feature subset, so a handful will dissent on almost any input. That guarantees
you rarely see 0.00 or 1.00. The output is **compressed toward the middle**: the forest is
*under*-confident, and its reliability curve sits *above* the diagonal at low probabilities and
*below* it at high ones. This is exactly the shape drawn on the slide's own figure `[f50]`, annotated
`under-confident` on the left and `over-confident (predicts 0.9 → actual 0.7)` on the right.

**SVM outputs are unbounded scores.** $w^\top x + b$ is a signed distance from the hyperplane, on no
particular scale. It is not a probability at all and cannot be interpreted as one. (`SVC(probability=True)`
silently runs Platt scaling internally — which is why it is slow, and why its probabilities can
disagree with `decision_function`'s sign in rare cases.)

**Naive Bayes** (Part 2 §19) is the extreme case: multiplying many conditionally-dependent
likelihoods drives outputs to 0.999 or 0.001 almost always. Its *ranking* can be excellent while its
probabilities are worthless.

**Boosted ensembles** are typically **over**-confident: the exponential/logistic loss keeps pushing
margins outward long after the classification is correct.

> ⚠️ **The models that are naturally calibrated are the ones trained to minimise a proper scoring
> rule.** Logistic regression minimises log loss, and log loss is *minimised* by reporting your true
> belief — so its outputs are calibrated more or less for free. Neural nets with a softmax + cross-
> entropy head *should* be too, and classically were, but modern over-parameterised networks are
> badly over-confident (Guo et al., 2017) — the same problem, a different cause.

### 14.3 Reading a reliability diagram

Bin predictions by predicted probability. For each bin, plot mean predicted probability (x) against
observed positive frequency (y).

```
   1.0 ┤                                    ╱ ●
       │  perfect calibration = diagonal  ╱ ●
   0.8 ┤                                ╱ ●
       │                              ╱  ●          BELOW the line
 actual│                            ╱ ●             = OVERCONFIDENT
   0.6 ┤                          ╱●                  (said 0.9, got 0.7)
frequency                       ╱ ●
   0.4 ┤                      ╱●
       │                   ●╱                       ABOVE the line
   0.2 ┤              ● ●╱                          = UNDERCONFIDENT
       │        ●  ●╱                                 (said 0.2, got 0.35)
   0.0 ┤   ● ●╱
       └┬─────┬─────┬─────┬─────┬
        0.0  0.2   0.4   0.6   0.8   1.0
              mean predicted probability
```

A perfectly calibrated model traces the diagonal. **Above** the diagonal = underconfident. **Below** =
overconfident. The Random-Forest curve on the slide does both — above at the left, below at the right
— which is the signature of compression toward 0.5.

### 14.4 The two methods

**Platt scaling** — fit a one-dimensional logistic regression mapping the model's raw score $s$ to a
probability:

$$P(y=1 \mid s) = \frac{1}{1 + e^{-(As + B)}}$$

Two parameters, $A$ and $B$, fitted by maximum likelihood on held-out data. $A$ controls how sharply
the sigmoid rises (it fixes over/under-confidence); $B$ shifts it left or right (it fixes a base-rate
mismatch).

| | Platt scaling | Isotonic regression |
|---|---|---|
| Form | Sigmoid, 2 parameters | Arbitrary **monotone non-decreasing** step function |
| Flexibility | Low — can only fix sigmoid-shaped distortion | High — can fix any monotone distortion |
| Data needed | Works on a few hundred rows | **Needs thousands**; overfits below ~1,000 |
| Failure mode | Underfits complex miscalibration | Overfits; produces flat steps at the extremes |
| Rule of thumb | **< ~1,000 calibration rows** | **> ~1,000 calibration rows** |

**Isotonic regression** solves $\min \sum_i (y_i - f(s_i))^2$ subject to $f$ being non-decreasing —
via the Pool Adjacent Violators algorithm. The monotonicity constraint is what makes it a
*calibration* method rather than just a second model: it is allowed to change *how confident* the
scores are, but never to change their **ranking**. Which leads to the single most important
consequence in this section:

> 💡 **Calibration cannot change ROC-AUC.** Both Platt and isotonic apply a monotone transformation
> to the scores. ROC-AUC depends only on the *ordering* of scores (Part 2 §17 computed it two ways,
> both order-based). A monotone map preserves ordering exactly. Therefore **AUC before calibration =
> AUC after calibration, to the last decimal.**
>
> Two things follow, and both get asked about:
> 1. **ROC-AUC cannot detect miscalibration.** A model with perfect AUC 1.0 can have wildly wrong
>    probabilities. If your only metric is AUC, you are blind to this entire failure mode.
> 2. **If you only care about ranking, calibration is pointless.** Which is exactly the slide's
>    "Not needed: when you only care about ranking (ROC-AUC unaffected by calibration)."

### 14.5 Measuring calibration

Slide `[f53]` gives two metrics.

**Brier score.** Words first:

> The formula says: **the average squared distance between the probability you promised and what
> actually happened.**

$$\text{Brier} = \frac{1}{N}\sum_{i}(\hat p_i - y_i)^2 \qquad \text{lower = better},\ 0 = \text{perfect}$$

| Symbol | Read it as | What it means |
|---|---|---|
| $N$ | "N" | Number of examples scored. |
| $\hat p_i$ | "p-hat sub i" | Predicted probability of the positive class for row $i$. |
| $y_i$ | "y sub i" | Actual outcome, 0 or 1. |

It is simply **MSE applied to probabilities**. It is a **proper scoring rule**: it is minimised, in
expectation, precisely when you report your true beliefs — you cannot game it by shading your
answers. And it measures two things at once (this is the Murphy decomposition): **calibration** *and*
**resolution** (how far your predictions dare to stray from the base rate). A model that always
predicts the base rate is perfectly calibrated and has a mediocre Brier score, which is correct — it
is honest but useless.

### 🧪 Worked example — Brier score

Five students, predicted dropout probabilities and actual outcomes:

| $\hat p_i$ | $y_i$ | $(\hat p_i - y_i)^2$ |
|---|---|---|
| 0.9 | 1 | $(-0.1)^2 = 0.01$ |
| 0.8 | 1 | $(-0.2)^2 = 0.04$ |
| 0.6 | 0 | $(0.6)^2 = 0.36$ |
| 0.3 | 0 | $(0.3)^2 = 0.09$ |
| 0.1 | 0 | $(0.1)^2 = 0.01$ |
| | **Sum** | **0.51** |

$$\text{Brier} = \frac{0.51}{5} = \mathbf{0.102}$$

For reference: always predicting 0.5 gives Brier $=0.25$; a perfect confident model gives 0. So 0.102
is decent. Note the third row contributed 0.36 — **71% of the total** — from one confident mistake.
Brier punishes confident errors quadratically, which is the behaviour you want.

**Expected Calibration Error (ECE).** The slide's one-liner:

> Expected Calibration Error (ECE): avg gap between confidence and accuracy per bin

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}\Big|\,\text{acc}(b) - \text{conf}(b)\,\Big|$$

| Symbol | Read it as | What it means |
|---|---|---|
| $B$ | "B" | Number of probability bins (typically 10). |
| $n_b$ | "n sub b" | Rows falling in bin $b$. |
| $\text{conf}(b)$ | "confidence of bin b" | Mean predicted probability in the bin. |
| $\text{acc}(b)$ | "accuracy of bin b" | Actual fraction positive in the bin. |
| $\lvert\cdot\rvert$ | "absolute value" | Gap size, direction ignored. |

It is literally *"the average vertical distance from the diagonal on the reliability diagram,
weighted by how many points are at each x."*

### 🧪 Worked example — ECE with three bins

Ten students, sorted by predicted probability:

| Bin | Predictions | conf(b) | Actuals | acc(b) | $n_b$ | Gap |
|---|---|---|---|---|---|---|
| [0, 0.33) | 0.1, 0.2, 0.3 | 0.200 | 0, 0, 1 | 0.333 | 3 | 0.133 |
| [0.33, 0.67) | 0.4, 0.5, 0.6 | 0.500 | 0, 1, 1 | 0.667 | 3 | 0.167 |
| [0.67, 1.0] | 0.7, 0.8, 0.9, 1.0 | 0.850 | 1, 1, 1, 0 | 0.750 | 4 | 0.100 |

$$\text{ECE} = \tfrac{3}{10}(0.133) + \tfrac{3}{10}(0.167) + \tfrac{4}{10}(0.100) = 0.0399 + 0.0501 + 0.0400 = \mathbf{0.130}$$

**Interpretation: on average, this model's stated probability is off by 13 percentage points.** If
you were pricing a decision on those probabilities, that is the size of your systematic error.

Notice the first two bins are *under*confident (actual > predicted) and the third is *over*confident.
ECE takes absolute values, so **these do not cancel** — which is correct, but it also means ECE
cannot tell you the *direction* of miscalibration. Look at the reliability diagram for that.

> ⚠️ **ECE's known weaknesses,** worth a sentence in an interview: it depends on the binning scheme
> (10 equal-width bins is convention, not principle); with few samples per bin it is noisy; and
> equal-width bins are nearly empty in the middle for a confident model. Adaptive binning (equal
> *count* per bin) is the usual fix.

### 14.6 When to calibrate — and when not to

`[f53]`:

> **When to calibrate:** When decisions depend on probability magnitude (bidding, risk pricing,
> insurance) · After: SVMs, Naive Bayes, Random Forests, boosted ensembles · Not needed: when you
> only care about ranking (ROC-AUC unaffected by calibration)
> **In sklearn:** `CalibratedClassifierCV(base_model, method='sigmoid'|'isotonic', cv=5)`
>
> **KEY TAKEAWAY** — When decisions rely on probability values, calibrate — Platt or isotonic — and
> verify with Brier score / ECE.

The rule reduces to one question: **does anything downstream multiply by your probability?**

| Downstream use | Calibration needed? | Why |
|---|---|---|
| Rank ads/products and show the top 10 | **No** | Only ordering is consumed |
| Bid `value × P(click)` in an auction | **Yes — critically** | The probability is a multiplier; a 20% error is a 20% overbid on every impression |
| Expected-loss reserving: `exposure × P(default)` | **Yes** | It's an accounting number |
| Threshold at 0.5 and act | Weakly | Only the ordering near the threshold matters — but if you tune the threshold on calibrated probabilities the number is *interpretable* |
| Show "87% match" to a customer | **Yes** | You made a claim to a human |
| Combine several models' probabilities | **Yes** | Averaging miscalibrated probabilities compounds the error |

> ⚠️ **`CalibratedClassifierCV`'s `cv` parameter is the whole thing, and it is easy to get wrong.**
> Calibration must be fitted on data the base model did **not** train on. If you fit Platt scaling on
> the training set, you are learning to correct the model's *training* scores, which are already
> over-confident from memorisation — you will "fix" a distortion that doesn't exist at test time and
> make calibration worse. `cv=5` handles this correctly: it internally cross-fits, training the base
> model on 4 folds and the calibrator on the 5th, five times. **Never pass `cv='prefit'` with data
> the model has seen.**

### 14.7 🧪 The lecture's own calibration result

`[f145]` (55:56). The notebook binarises the target to Dropout-vs-rest and calibrates a Random
Forest three ways:

```
                 Brier      ECE
Uncalibrated     0.101     0.037
Platt (sigmoid)  0.098     0.019
Isotonic         0.098     0.018
```

**Read all six numbers.**

- **Brier barely moved** (0.101 → 0.098, a 3% improvement). That is because Brier bundles calibration
  *and* resolution, and calibration cannot change resolution — it can't make the model know more.
  A small Brier improvement is the expected result of a pure calibration step.
- **ECE nearly halved** (0.037 → 0.019, a **49% reduction**). That is the metric that isolates
  calibration, and it moved a lot. The model's stated probabilities went from being off by 3.7
  percentage points on average to 1.9.
- **Platt and isotonic tied** (0.019 vs 0.018 — well inside noise). With only 885 test rows and a
  base Random Forest whose miscalibration is the classic *sigmoid-shaped* compression toward 0.5,
  Platt's two parameters are enough. Isotonic's extra flexibility bought nothing and, at this sample
  size, risked overfitting. **This is the rule of thumb from §14.4 confirmed: below ~1,000 rows,
  prefer Platt.**

> 💡 **The pattern to remember: ECE improves dramatically, Brier improves slightly, AUC does not
> change at all.** If you ever see calibration change your AUC, you have a bug — most likely you
> refitted the base model instead of only the calibration map.

```interactive
type: graph
title: The reliability diagram, before and after
concept: What calibration does and does not fix
control: Toggle between uncalibrated / Platt / isotonic, and a slider for the number of bins (5 to 20)
observe: The reliability curve moving toward the diagonal; live Brier, ECE and ROC-AUC readouts beside it
insight: ECE nearly halves and the curve straightens, Brier barely moves, and AUC does not change AT ALL — because calibration is a monotone map and monotone maps preserve ranking
fallback: The three-row table in §14.7 and the ASCII reliability diagram in §14.3 make the same three points.
```

---

## 15. Pipelines and the end-to-end workflow

Two final deck slides, `[f154]` (57:42) and `[f160]` (59:39).

> **Why Pipelines?** Encapsulate preprocessing + model in a single object · **Prevent data leakage —
> transforms fit ONLY on training folds automatically** · Clean, reproducible, deployable: one
> artifact for production

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression())
])
pipe.fit(X_train, y_train)
```

### 15.1 The leakage bug a Pipeline makes impossible

Part 1 §6 introduced preprocessing leakage. Here is the version that survives a code review and still
inflates your score, because it *looks* correct:

```python
# ❌ WRONG — and it will not raise a single warning
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)          # looks fine: only X_train!
scores = cross_val_score(model, X_scaled, y_train, cv=5)
```

Spot it: `scaler` computed its mean and standard deviation from **all of `X_train`** — including the
rows that `cross_val_score` will later hold out as fold 1's validation set. Every validation fold was
standardised using statistics that were partly derived from *itself*. The leak is small per fold and
absolutely real, and it makes your CV score optimistic in a way you cannot detect from inside.

```python
# ✅ RIGHT — the Pipeline is re-fit inside every fold
pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
scores = cross_val_score(pipe, X_train, y_train, cv=5)
```

`cross_val_score` now calls `pipe.fit(train_fold)` five times. Each call runs
`scaler.fit_transform(train_fold)` — statistics from the training fold only — then
`scaler.transform(val_fold)`. **The correct behaviour is the default, and the wrong version is no
longer expressible.**

> 💡 **This is the real argument for pipelines, and it generalises.** The value is not tidiness. It
> is that a Pipeline makes an entire class of bug *structurally impossible* rather than merely
> discouraged. Anything with a `fit` — scalers, imputers, PCA, target encoders, feature selectors,
> SMOTE — leaks if fitted outside the CV loop, and every one of them is safe inside a Pipeline.
> Selectors are the worst offender: choosing the top-20 features on the full training set before CV
> is a classic and enormous leak, because the selection used the labels of the validation rows.

**And the deployment argument.** `joblib.dump(pipe, 'model.pkl')` saves preprocessing *and* model as
**one artifact**. The serving code calls `pipe.predict(raw_df)` and cannot possibly apply a different
scaler than the one training used — the single most common source of training/serving skew,
eliminated by construction.

### 15.2 ColumnTransformer: different treatment for different columns

Slide `[f160]`:

> **Real-world data has mixed types — ColumnTransformer handles this:**

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

pipe = Pipeline([("preprocess", preprocessor),
                 ("model", GradientBoostingClassifier(n_estimators=200))])

# Tune with RandomizedSearchCV (more efficient than Grid for many params)
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(pipe, param_distributions, n_iter=50, cv=5, scoring="f1")
search.fit(X_train, y_train)
```

> **Summary: ColumnTransformer + Pipeline + SearchCV = production-ready ML workflow**

**Why you need it.** `StandardScaler` on a categorical column is meaningless; `OneHotEncoder` on a
continuous one produces thousands of columns. Real data has both. `ColumnTransformer` routes each
group of columns to the right transformer and horizontally concatenates the results.

For this lecture's dataset that split is explicit `[f77]` (47:54): **18 numeric features** (Previous
qualification (grade), Admission grade, Age at enrollment, the twelve *Curricular units 1st/2nd sem*
variables, Unemployment rate, Inflation rate, GDP) and **18 categorical** (Marital status,
Application mode, Application order, Course, Daytime/evening attendance, Previous qualification,
Nacionality, Mother's/Father's qualification, Mother's/Father's occupation, Displaced, Educational
special needs, Debtor, Tuition fees up to date, Gender, Scholarship holder, International).
**Missing values total: 0.**

> ⚠️ **`handle_unknown="ignore"` is not boilerplate — it is the line that stops your service crashing
> at 2am.** In production a category will appear that wasn't in training: a new `Course` code, a new
> `Nacionality`. Without this flag, `OneHotEncoder` raises `ValueError` and the request fails.
> With it, the unseen category is encoded as all-zeros — the model degrades gracefully instead of
> throwing. Note the trade-off you're accepting: all-zeros is indistinguishable from "none of the
> known categories," so unseen categories quietly become a single pooled bucket. That is usually
> right, and it is always better than a 500.

**The three layers, and what each one prevents:**

| Layer | Prevents |
|---|---|
| `ColumnTransformer` | Applying the wrong transform to a column type |
| `Pipeline` | Fitting any transform on data the model will be evaluated on (leakage) |
| `SearchCV` **wrapping the pipeline** | Tuning hyperparameters against a leaked score |

The nesting order matters and is the thing to get right: **`SearchCV(Pipeline(ColumnTransformer, model))`**.
If you invert it — search over a model and paste preprocessing on outside — you have leaked at the
outermost level and the whole structure is decorative.

---

## 16. The hands-on notebook, end to end

The last 14 minutes run everything above on one dataset. This section is the whole demo with every
number it printed, because seeing the pieces compose is worth as much as the pieces.

### 16.1 The dataset

From `[f56]` (44:47):

> **Dataset:** UCI *"Predict Students' Dropout and Academic Success"* (Realinho et al., 2022) —
> 4,424 students × 36 features, 3-class target (`Graduate` / `Dropout` / `Enrolled`).
> Runtime target: end-to-end **< 5 minutes** on a laptop.

The notebook's own map from its 13 sections to the deck's 24 slides — useful as a revision index:

| § | Deck slide(s) | Concept |
|---|---|---|
| 1 | 3–5 | Decision Trees, splitting criteria, pruning |
| 2 | 6 | Practical diagnostics · numeric Gini intuition |
| 3 | 7 | Bagging & Random Forests |
| 4 | 8 | Boosting (AdaBoost, GBT, XGBoost) |
| 5 | 9–10 | Bagging vs Boosting |
| 6 | 12 | Multi-class strategies (OvR / OvO / Softmax) |
| 7 | 13 | Class imbalance (SMOTE, class_weight) |
| 8 | 14–15 | Cross-Validation (K-Fold, Stratified, TimeSeriesSplit) |
| 9 | 17 | Hyperparameter tuning (Grid / Random) |
| 10 | 18 | Model selection (No Free Lunch, Occam's Razor) |
| 11 | 20–21 | Probability calibration (Platt / isotonic, Brier, ECE) |
| 12 | 22–24 | End-to-end Pipeline + ColumnTransformer + GridSearchCV |
| 13 | — | Summary of key takeaways |

> ⚠️ **This table is the notebook's own printed TOC, transcribed verbatim — but the live notebook's
> own running section headers drift from it starting at row 3 ("Bagging & Random Forests").** By the
> time the notebook reaches the ensembles section its own in-cell headers read "Section 7" / "Section
> 8" where this TOC's numbering implies otherwise, because what this table lists as two separate rows
> (`1` and `2`) is folded into one "Section 1" with subsections `1.3`/`1.4` in the live notebook. If
> you're navigating a copy of the actual notebook rather than reading this table as a static index,
> expect the numbers from "Bagging & Random Forests" onward to be off by one — jump by concept name,
> not by the row number here.

### 16.2 The stratified split

`[f79]` (48:30):

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
```

```
Train: (3539, 36)  ·  Test: (885, 36)
Train class ratios: {'Dropout': 1137, 'Enrolled': 635, 'Graduate': 1767}
Test  class ratios: {'Dropout':  284, 'Enrolled': 159, 'Graduate':  442}
```

**Check the stratification held.** Dropout is $1137/3539 = 32.13\%$ of train and $284/885 = 32.09\%$
of test. Enrolled: $17.94\%$ vs $17.97\%$. Graduate: $49.93\%$ vs $49.94\%$. All three match to
within 0.04 percentage points — that is `stratify=y` doing exactly what §11.1 described.

**And note the baseline this sets.** Always predicting Graduate scores $442/885 = \mathbf{0.499}$.
Every model below must beat 0.499 to have earned its existence.

### 16.3 Every result, in one table

| § | What was measured | Result |
|---|---|---|
| 0 | Majority-class baseline | **0.499** |
| 1 | Full-depth single tree, test accuracy | **0.697** |
| 1 | Best `max_depth` by test accuracy | **5** → **0.757** |
| 2 | Gini verification | 50/50 → **0.500**, 90/10 → **0.180**, 100/0 → **0.000** |
| 3 | Bagging (100 trees) | **0.753** |
| 3 | Random Forest (200 trees) | **0.769** (OOB **0.774**) |
| 4 | AdaBoost (200 stumps) | train 0.762 / test **0.744** / 1.1 s |
| 4 | GradientBoosting (200, lr 0.1, depth 3) | train 0.882 / test **0.763** / 3.9 s |
| 4 | HistGradientBoosting (300, lr 0.05, depth 6) | train 0.999 / test **0.755** / 13.5 s |
| 4 | XGBoost | *skipped — `ModuleNotFoundError`* |
| 5 | Label noise 0% → 30% | RF **0.769 → 0.774**; GBM **0.763 → 0.737** |
| 6 | OvR / OvO / Softmax | acc 0.757 / **0.774** / 0.763 · macro-F1 0.657 / **0.695** / 0.677 |
| 7 | Imbalance: baseline → rebalanced | acc 0.763 → 0.736; **minority F1 0.403 → 0.509** |
| 8 | K-Fold vs Stratified K-Fold (macro-F1) | 0.702 ± 0.023 vs **0.688 ± 0.015** |
| 9 | GridSearchCV, 12 configs | best F1 **0.684** · `max_depth=20, max_features='log2', n_estimators=100` |
| 9 | RandomizedSearchCV, 25 configs | best F1 **0.681** · `max_depth=20, max_features='sqrt', min_samples_leaf=1, n_estimators=363` |
| 10 | Model selection leaderboard | *cell not executed on screen* |
| 11 | Calibration (Brier / ECE) | uncal 0.101/0.037 · Platt 0.098/**0.019** · isotonic 0.098/**0.018** |
| 12 | End-to-end pipeline | best CV F1 **0.707** · **test F1 0.679** |

### 16.4 The final pipeline, and the number that matters most

`[f152]` (56:56):

```python
grid_pipe = GridSearchCV(pipe, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
grid_pipe.fit(X_train, y_train)      # raw X_train — categorical dtype preserved
```

```
Best params : {'model__learning_rate': 0.1, 'model__max_depth': 4, 'model__max_iter': 200}
Best CV F1  : 0.707
Test F1     : 0.679
```

Then wrapped in calibration and saved as one artifact:

```python
best = grid_pipe.best_estimator_
calibrated_pipe = CalibratedClassifierCV(best, method='isotonic', cv=3)
calibrated_pipe.fit(X_train, y_train)
joblib.dump(calibrated_pipe, '/tmp/student_dropout_model.pkl')
```

And round-tripped — reload from disk, predict on three real test rows:

```
Saved: /tmp/student_dropout_model.pkl

     true       pred      P(Dropout)  P(Enrolled)  P(Graduate)
0    Graduate   Graduate     0.072       0.063        0.864
1    Graduate   Graduate     0.058       0.127        0.815
2    Enrolled   Dropout      0.675       0.250        0.075
```

> 💡 **Note the `model__` double-underscore prefix.** That is sklearn's syntax for reaching into a
> named step of a pipeline: `model__max_depth` means *"the `max_depth` of the step named `model`."*
> It is what lets `GridSearchCV` tune hyperparameters of a component buried three layers deep — and
> it means you can tune *preprocessing* hyperparameters the same way (`preprocess__num__with_mean`).

**Three things this output teaches that the slides do not:**

**1. The CV-to-test gap is 0.028 (0.707 → 0.679), and that is the expected, healthy outcome.** CV
scores are optimistically biased for exactly the reason §13.2 gave: the winning configuration was
*chosen* on those folds, so its CV score includes a lucky component. The test score is the honest
one. A gap of ~3 points after a 12-config search is normal. A gap of 15 points would mean you had
searched far too hard against the validation signal.

**2. Row 2 is a real, informative failure.** An `Enrolled` student was predicted `Dropout` with 67.5%
confidence. This is the model's characteristic error and every diagnostic above predicted it:
Enrolled's F1 was the lowest of the three classes in *every* strategy (§9.4), its recall was 0.264
under OvR, and rebalancing (§10.5) targeted precisely this class. "Enrolled" means *still studying,
outcome not yet determined* — it is genuinely the hardest class, because a student on a trajectory
toward dropping out and a student who is simply taking longer look nearly identical in the features.

**3. Row 0 and row 1 are confidently and correctly Graduate**, at 0.864 and 0.815 — and because the
model was calibrated with isotonic regression before saving, **those numbers can be used**. That is
the whole point of §14: `0.864` is a quantity a scholarship-allocation rule can multiply by, not just
a score to rank on.

### 16.5 The notebook's own closing summary

`[f158]` (59:26) — the lecture's last content slide, reproduced verbatim because it is a
ready-made revision card:

| Topic | One-line takeaway |
|---|---|
| **Decision Trees** | Interpretable but high-variance — control with `max_depth` / pruning, then ensemble. |
| **Bagging / RF** | Parallel trees on bootstrap samples → variance reduction; OOB gives free validation. |
| **Boosting** | Sequential trees fit residuals; XGBoost/LightGBM/HistGBM dominate tabular — always pair with early stopping. |
| **Bagging vs Boosting** | Noisy data → RF; clean data + max accuracy → boosting. |
| **Multi-class** | OvR (default), OvO (SVM/small K), Softmax (neural nets, end-to-end). |
| **Class imbalance** | Use `class_weight` for mild, SMOTE for severe. Never trust plain accuracy. |
| **Cross-Validation** | 5-fold stratified is default; `TimeSeriesSplit` for temporal data — random K-fold leaks the future. |
| **Hyperparameter tuning** | Random > Grid for same budget. |
| **Model selection** | No Free Lunch; compare families with CV. Occam's Razor — prefer simpler. |
| **Calibration** | Calibrate when decisions depend on probability values; Platt (small data) or isotonic (large). |
| **Pipelines** | `ColumnTransformer` + `Pipeline` + `SearchCV` = production-ready ML. One `joblib.dump` = deployable artifact. |

---

## Putting it together

### The dependency map

```
                    ┌───────────────────────────────┐
                    │  §1–2  DECISION TREE          │
                    │  recursive partitioning +     │
                    │  Gini / entropy / info gain   │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  §3  THE DEFECT               │
                    │  greedy (NP-completeness) +   │
                    │  argmax variance × recursion  │
                    └───────────────┬───────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
  ┌─────────────────┐   ┌────────────────────┐   ┌────────────────────┐
  │ §4 PRUNE        │   │ §6 BAG             │   │ §7 BOOST           │
  │ pre / post      │   │ deep trees +       │   │ shallow trees +    │
  │ ↓variance       │   │ bootstrap + vote   │   │ sequential residual│
  │ ↑bias           │   │ ↓↓variance         │   │ ↓↓bias             │
  │ 0.697→0.757     │   │ →0.753             │   │ →0.763             │
  └─────────────────┘   └─────────┬──────────┘   └─────────┬──────────┘
                                  │ decorrelate            │
                        ┌─────────▼──────────┐             │
                        │ RANDOM FOREST      │             │
                        │ + max_features     │             │
                        │ attacks ρσ² floor  │             │
                        │ →0.769, OOB 0.774  │             │
                        └─────────┬──────────┘             │
                                  └──────────┬─────────────┘
                                             │  §8 CHOOSE: how clean are the labels?
                                             ▼
        ══════════════════════ THE MACHINERY AROUND THE MODEL ══════════════════════
                                             │
     ┌──────────┬──────────┬─────────────────┼──────────┬───────────┬─────────────┐
     ▼          ▼          ▼                 ▼          ▼           ▼             ▼
  §9 multi-  §10 im-   §11 CROSS-       §12 hyper-  §13 model  §14 cali-    §15 PIPELINE
  class      balance   VALIDATION       parameter   selection  bration      the wrapper that
  OvR/OvO/   weights/  ┌──────────────┐ search      NFL +      Platt/       makes all of the
  softmax    SMOTE     │ every box    │ grid/random Occam      isotonic     above leak-proof
                       │ ABOVE and    │                                     ↓
                       │ BELOW needs  │                                     joblib.dump
                       │ this to be   │                              ONE DEPLOYABLE ARTIFACT
                       │ measured     │                                 CV 0.707 / test 0.679
                       └──────────────┘
```

**Read the diagram's spine.** §11 is drawn in the middle because it is not one topic among many —
it is the **measurement instrument every other decision depends on**. Choosing `max_depth` (§4),
choosing bagging vs boosting (§8), choosing a rebalancing strategy (§10), choosing hyperparameters
(§12), choosing a family (§13), choosing Platt vs isotonic (§14) — every one of those is a
comparison between cross-validated scores. Get §11 wrong and every decision downstream is made on
corrupted evidence.

### Five threads running through the lecture

**Thread 1 — Every ensemble method is a bias–variance decision, and you can read which from the base
learner's depth.** Bagging wants **deep** trees because it can only remove variance, so you feed it
the model whose only flaw is variance. Boosting wants **shallow** trees because it can only remove
bias by adding rounds, and deep base learners would leave it nothing to correct while adding variance
it cannot fix. If you ever forget which way round, ask: *what is this method able to reduce?* — then
give it a base learner with the opposite problem.

**Thread 2 — The correlation floor $\rho\sigma^2$ appears three times in this lecture, in disguise.**
(1) It's why Random Forest subsamples features (§6.2). (2) It's why LOOCV has high variance despite
averaging $n$ estimates — the $n$ training sets are 99.97% identical, so $\rho \approx 1$ (§11.2).
(3) It's why ensembling five near-identical XGBoost configs buys nothing. **Averaging only helps to
the extent that the things you average disagree.**

**Thread 3 — Accuracy lied at every single opportunity, and a different metric caught it each time.**
Rebalancing looked like a 2.7-point *loss* on accuracy and was a 10.6-point *gain* on minority F1
(§10.5). OvO looked 1.7 points better than OvR on accuracy and was 3.8 better on macro-F1, all of it
from the minority class (§9.4). Calibration didn't move accuracy at all and halved ECE (§14.7). Three
sections, three different right metrics, and accuracy wrong in all three.

**Thread 4 — Every measurement in this lecture has an error bar of about ±1.4 points, and several
"results" are inside it.** With 885 test rows at accuracy ≈0.77, one standard error is
$\sqrt{0.77 \times 0.23/885} = 0.014$. So: RF's noise-experiment wobble (§8.2) is noise. Grid's
0.003 win over random (§12.5) is noise. Platt vs isotonic's 0.001 ECE difference (§14.7) is noise.
SMOTE vs class weights (§10.5) is noise. **Knowing which differences are real is a skill, and it is
the one that separates a scientist from someone who reports leaderboards.**

**Thread 5 — Three separate mechanisms in this lecture are the same mechanism: hold out, then
evaluate.** OOB scoring holds out 36.8% per tree (§6.3). Cross-validation holds out $1/K$ per fold
(§11). `CalibratedClassifierCV(cv=5)` holds out a fold to fit the calibration map (§14.6). All three
exist because *a model's opinion of its own performance on data it has seen is worthless* — and all
three break in exactly the same way if you fit anything on the full training set beforehand, which is
why §15's Pipeline is the correct ending to the lecture.

---

## Interview prep — Amazon Applied Scientist

### Core questions

Ranked easy → hard. Questions 8–12 require combining two concepts.

---

<details>
<summary><b>1.</b> What is Gini impurity, and why is it 0.5 at maximum?</summary>

Gini impurity is the probability that two rows drawn independently at random from a node belong to
different classes. Draw once, get class $i$ with probability $p_i$; draw again independently, also
class $i$ with probability $p_i$. So they match with probability $\sum_i p_i^2$, and differ with
probability $1 - \sum_i p_i^2$ — that's Gini.

It is 0 when a node is pure, because two draws always match. Its maximum is at the uniform
distribution $p_i = 1/K$, giving $1 - K(1/K)^2 = 1 - 1/K$.

**The 0.5 figure is binary-specific** — that's $1 - 1/2$. For three classes the ceiling is 0.667, for
ten it's 0.9. I'd be careful about quoting 0.5 on a multi-class problem.
</details>

<details>
<summary><b>2.</b> Why do decision trees overfit, and what are the two ways to stop it?</summary>

Two mechanisms, and it's worth separating them.

**Structurally:** an unconstrained tree keeps splitting until every leaf is pure. With enough depth
it can isolate every training row in its own leaf, giving 100% training accuracy and zero
generalisation — it's memorised the data.

**Statistically, which is the deeper reason:** every split is an argmax over hundreds of candidate
(feature, threshold) pairs. When two candidates score nearly identically — which is common, since good
features are correlated — a small change in the training data flips the winner, and because the
algorithm is recursive, *the entire subtree below that node changes*. Tree variance is argmax
variance amplified by recursion.

**Two fixes.** **Pre-pruning** stops growth early via `max_depth`, `min_samples_split`,
`min_samples_leaf` — cheap, and `max_depth` alone does most of the work. **Post-pruning** grows the
full tree then removes subtrees that don't pay for themselves, minimising
$R_\alpha(T) = R(T) + \alpha|\tilde T|$ — same structure as ridge, error plus a complexity penalty.

Post-pruning is better in principle because pre-pruning is myopic in the same way greedy splitting
is — it can stop above two excellent splits it never discovers. Pre-pruning is used anyway because
it's cheaper, and `max_depth` tuned by CV captures most of the gain. In the lecture's demo, going
from the default full tree to `max_depth=5` moved test accuracy from 0.697 to 0.757 — six points from
one hyperparameter.
</details>

<details>
<summary><b>3.</b> Explain bagging. Why does it reduce variance but not bias?</summary>

Bagging trains $B$ models on bootstrap samples — draws of $n$ rows with replacement — and combines
them by majority vote or averaging.

**Bias is unchanged, provably.** Each tree is fit to a sample from the same distribution, so
$\mathbb{E}[T_b(x)] = \mu(x)$ for every $b$. The ensemble's expectation is
$\frac1B \sum_b \mathbb{E}[T_b(x)] = \mu(x)$ — identical to one tree's. Averaging doesn't move an
expectation.

**Variance falls** by the standard result for an average of correlated variables:
$\operatorname{Var}(\frac1B\sum T_b) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. The second term
shrinks like $1/B$.

That's why bagging wants **deep** base learners: it can only remove variance, so you feed it a model
whose only problem is variance. Fully grown trees are exactly that — near-zero bias, huge variance.
</details>

<details>
<summary><b>4.</b> What's the difference between Random Forest and plain bagging, and why does it help?</summary>

Random Forest is bagging **plus** a random subset of features considered at each split — typically
$\sqrt{d}$ for classification, $d/3$ for regression.

The reason is the variance formula: $\operatorname{Var} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$.
The second term vanishes as you add trees; **the first does not.** $\rho\sigma^2$ is a floor.

Plain bagging leaves $\rho$ high, because bootstrap samples overlap by about 63% and the greedy
algorithm, given the same features, mostly picks the same strong feature at the root. So the trees
come out similar. Feature subsampling makes the dominant feature *unavailable* in most trees, forcing
different roots and genuinely different structures — $\rho$ drops, and the floor drops with it.

The trade is that hiding features makes each individual tree worse. That's fine: you're trading a
small increase in $\sigma^2$ for a large decrease in $\rho$, and the formula says that's a winning
trade. **Random Forest deliberately handicaps every tree to improve the ensemble.**

Numerically, in the lecture's demo: bagging 0.753, Random Forest 0.769. That 1.6 points is purely
the $\rho$ effect.
</details>

<details>
<summary><b>5.</b> What is out-of-bag error and why is it "free"?</summary>

For any bootstrap sample, a given row has probability $(1-1/n)^n$ of never being drawn, which
converges to $e^{-1} = 0.368$. So about 36.8% of rows are out-of-bag for each tree.

To score row $i$, poll only the trees that never saw it and take their majority vote. Do that for
every row and you get an honest held-out estimate — using no held-out data. It's free because those
predictions are a by-product of training you'd have done anyway.

In the lecture's run, OOB was 0.774 against a true test score of 0.769 — within half a point.

**Caveat I'd raise:** OOB assumes exchangeable rows. With time-ordered data it's optimistic in
exactly the way random K-fold is, since the "unseen" rows are interleaved in time with training rows.
With grouped data (multiple rows per customer) it leaks across the group. In both cases I'd ignore
OOB and use a proper temporal or grouped split.
</details>

<details>
<summary><b>6.</b> Explain gradient boosting. Why is it called "gradient"?</summary>

It's an additive model $F_M(x) = \sum_m \alpha_m h_m(x)$ built greedily. Start with a constant. At
each round, compute the negative gradient of the loss **with respect to the current predictions**,
fit a shallow regression tree to those pseudo-residuals, and add a shrunken version of it:
$F_m = F_{m-1} + \eta h_m$.

The name is literal. Ordinary gradient descent moves in *parameter* space: $w \leftarrow w - \eta\nabla_w L$.
Gradient boosting moves in *function* space: $F \leftarrow F + \eta h$ where $h \approx -\nabla_F L$.
The trees are how you represent a step direction when your parameter is an entire function.

"Fit the residuals" is the squared-loss special case: for $L = \frac12(y-F)^2$,
$-\partial L/\partial F = y - F$, which *is* the residual. The gradient framing is what generalises —
with log loss the pseudo-residual is $y_i - p_i$, with absolute loss it's $\operatorname{sign}(y_i - F_i)$.
Change the loss and only that one step changes; everything else is the same code. That modularity is
why one algorithm handles regression, classification, ranking, and survival analysis.
</details>

<details>
<summary><b>7.</b> Bagging or boosting — how do you choose?</summary>

I'd ask one question first: **how clean are the labels?**

Boosting focuses on whatever it currently gets wrong. A mislabelled row is *by construction* the row
with the largest residual, so every round targets it, and with AdaBoost's exponential reweighting its
influence compounds. Bagging is the opposite: a bad row appears in only ~63% of bootstrap samples,
distorts one leaf in those trees, and gets outvoted by the rest.

The lecture measured exactly this. Flipping training labels from 0% to 30%: Random Forest went 0.769
→ 0.774, completely flat. Gradient boosting went 0.763 → 0.737, a monotone 2.6-point decline.

**So:** human-annotated labels, user-reported categories, or noisy proxies → start with Random
Forest. Mechanically-derived, near-certain labels — did the payment charge back, did the package
arrive late → boosting will win.

Secondary considerations: bagging parallelises across whole trees and boosting doesn't (only
split-finding within a tree parallelises), so bagging trains much faster on a cluster. Bagging gives
OOB validation free. And bagging has far fewer interacting hyperparameters — `learning_rate` and
`n_estimators` in boosting are really one knob viewed twice.

I'd typically run Random Forest first as a strong, robust, nearly-untuned baseline, then see whether
a properly early-stopped booster beats it enough to justify the tuning burden.
</details>

<details>
<summary><b>8.</b> 🔗 Why is "more trees never overfits" true for Random Forest and false for boosting?</summary>

*(Combines §6 and §7.)*

Because they're doing different things with each additional tree.

In **bagging**, each tree is an independent draw from the same procedure. Adding trees drives the
ensemble toward its own expectation — a *convergence* process. The bias is fixed at one tree's bias
and never moves; the variance falls monotonically toward $\rho\sigma^2$. There's nothing to overfit:
you're estimating a fixed quantity more precisely.

In **boosting**, each tree is fit to what the *current* ensemble gets wrong. Early rounds fit signal.
Later rounds fit whatever's left — which, once the signal is exhausted, is noise. The ensemble keeps
driving training loss down past the point where test loss has started rising. It's an *optimisation*
process against training error, and it doesn't know when to stop.

That's why boosting needs early stopping and Random Forest doesn't, and it's why `n_estimators` is a
safety knob in one and a regularisation knob in the other.

The lecture demonstrated it accidentally: `HistGradientBoosting` with `max_iter=300` hit **train
0.999, test 0.755** — a 24-point gap — while Random Forest with 200 trees was at 0.769 with no such
gap. The fastest algorithm lost to the slowest because nobody told it when to stop.
</details>

<details>
<summary><b>9.</b> 🔗 Your fraud model has 99.9% accuracy and ROC-AUC of 0.97. Ship it?</summary>

*(Combines §10 and §14, plus Part 2 §17.)*

No, and I'd want three more numbers before I could say anything useful.

**On the accuracy:** with 0.1% fraud, `return "legitimate"` scores 99.9%. The number is
uninformative by construction. I'd want **precision, recall, and PR-AUC** on the fraud class.

**On the AUC:** ROC-AUC is computed from TPR and FPR, and FPR's denominator is the huge negative
class. Catching 90% of fraud at a 1% false-positive rate sounds excellent — but 1% of a million
legitimate transactions is 10,000 false alarms against maybe 900 true catches. **Precision ≈ 8%.**
ROC-AUC structurally hides this on imbalanced data; PR-AUC does not, because precision's denominator
is the predicted-positive set.

**Third thing, which people miss:** what happens downstream? If the score feeds an expected-loss
calculation or a bidding decision, I need calibrated probabilities, and **ROC-AUC cannot detect
miscalibration at all** — calibration is a monotone map, and AUC depends only on ranking, so AUC is
literally invariant to it. I'd check Brier score and ECE, and if the base model is a Random Forest or
a booster (both reliably miscalibrated), wrap it in `CalibratedClassifierCV`.

So: PR-AUC and precision-at-operating-threshold for whether it works; Brier/ECE for whether the
numbers mean anything; and a cost analysis of a false positive against a missed fraud to set the
threshold.
</details>

<details>
<summary><b>10.</b> 🔗 Rebalancing dropped accuracy from 0.763 to 0.736. Did you make the model worse?</summary>

*(Combines §10 and §9.)*

Almost certainly better, and I'd show the other columns to prove it.

In the lecture's numbers, the same change moved **minority-class F1 from 0.403 to 0.509** — up 10.6
points — and balanced accuracy from 0.670 to 0.711. Accuracy fell 2.7 because rebalancing moved some
predictions out of the large Graduate class (442 test rows, plenty to spare) and into the small
Enrolled class (159 rows, and we were missing most of them).

Accuracy counts the loss and ignores the gain, because it weights every row equally and there are
2.8× more Graduates than Enrolled. Macro-F1 and balanced accuracy weight every *class* equally, which
is what you want when the rare class is the one you built the model for.

**But I'd also push back on the premise.** Whether the trade is right isn't a modelling question —
it's about what an Enrolled student is worth relative to a Graduate. If the intervention is a cheap
email, high recall is worth a lot of precision. If it's a €5,000 scholarship, precision matters far
more. I'd want the cost ratio, then set the **threshold** to hit it — which is free, needs no
retraining, and is more directly controllable than either class weights or SMOTE.
</details>

<details>
<summary><b>11.</b> 🔗 Why does calibration change ECE a lot, Brier a little, and AUC not at all?</summary>

*(Combines §14 with Part 2 §17.)*

Three different things are being measured.

**AUC doesn't move because it can't.** Platt and isotonic are both *monotone* transformations of the
score. ROC-AUC depends only on the ordering of scores — it's the probability a random positive
outranks a random negative. A monotone map preserves every pairwise ordering exactly, so AUC is
invariant. Not approximately: identically.

**ECE moves a lot** because it measures *only* calibration — the average gap between stated
confidence and observed frequency per bin. That gap is exactly what the calibration map is fitted to
close. In the lecture's run it went 0.037 → 0.019, a 49% reduction.

**Brier barely moves** because it bundles two things. By the Murphy decomposition it's calibration
*plus* resolution — how far your predictions dare to stray from the base rate. Calibration can fix
the first and cannot touch the second; it can't make the model know more. So only part of Brier is
addressable, and the lecture saw 0.101 → 0.098.

**The diagnostic use:** if you ever see AUC change after calibration, you have a bug — you refitted
the base model instead of only the calibration map.
</details>

<details>
<summary><b>12.</b> 🔗 You get 0.92 CV accuracy and 0.71 in production. Walk me through the diagnosis.</summary>

*(Combines §11, §15, and Part 1 §6.)*

A 21-point gap is far too large to be selection bias from hyperparameter search — that's worth 2–3
points. Something structural is wrong. I'd check four things, cheapest first.

**1. Preprocessing fitted outside the CV loop.** The classic:
`X = scaler.fit_transform(X_train)` then `cross_val_score(model, X, ...)`. The scaler saw every
validation fold's rows. Same bug, worse, with imputers, target encoders, and feature selectors —
selecting the top-20 features on the full training set before CV leaks the validation labels
themselves. **The fix and the test are the same:** put everything in a `Pipeline` and re-run. If the
score drops, that was it.

**2. Temporal structure with random K-fold.** If rows are time-ordered, random CV lets the model
interpolate between days it has seen, while production requires extrapolation into a genuinely unseen
future. This produces exactly this magnitude of gap. Test by switching to `TimeSeriesSplit` — and if
the label has a maturation delay, add a `gap`.

**3. Grouped rows.** Multiple rows per customer/session/device split across folds means the model can
memorise the entity instead of learning the pattern. `GroupKFold` fixes it, and the score will drop.

**4. A leaky feature.** Something in the feature set encodes the outcome — populated after the event,
or a proxy for it. The tell is a single feature with implausibly dominant importance. I'd audit
feature availability at *prediction time*, not at training time — for each feature, is its value
known before the label exists?

If all four come back clean, *then* I'd look at genuine distribution shift between the training
period and production, compare feature distributions directly, and check whether the deployed
preprocessing matches training — which, again, `joblib.dump` of a single Pipeline object prevents by
construction.
</details>

---

### Depth probes

The follow-ups that come after a good first answer.

| They asked | The probe | What a strong answer contains |
|---|---|---|
| "Gini or entropy?" | "So why does sklearn default to Gini?" | Logarithms cost more; the two curves have the same zeros, same shape, same maximum, so the argmax rarely differs. Tune `max_depth` instead — worth 100× more. |
| "Bagging reduces variance" | "By how much? Give me the formula." | $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$; the second term vanishes with $B$, the first doesn't, and that floor is what `max_features` attacks. |
| "Random Forest subsamples features" | "Doesn't that make each tree worse?" | Yes — deliberately. You trade a small $\sigma^2$ increase for a large $\rho$ decrease, and the formula shows that's a winning trade. |
| "OOB is free validation" | "When would you not trust it?" | Temporal data (interleaved in time = future leakage), grouped data (leaks across the group). Both need a proper split instead. |
| "Boosting fits residuals" | "What if the loss isn't squared error?" | Then it fits the *negative gradient* — $y_i - p_i$ for log loss, $\operatorname{sign}(y_i-F_i)$ for MAE. Only that step changes; the rest of the algorithm is identical. |
| "XGBoost is parallelised" | "Across what?" | Split-finding *within* a tree. Rounds are inherently sequential — round $m$ needs $m-1$'s residuals. Unlike RF, which parallelises across whole trees. |
| "Use random search" | "Your own experiment showed grid winning." | 0.003 is one-fifth of a standard error; 12 and 25 configs are both far below the 60 the theorem needs; random's advantage is *dimensional* and shows up at 5+ hyperparameters. |
| "60 trials, 95% probability" | "Derive it." | $1 - 0.95^n \ge 0.95 \Rightarrow n \ge \ln(0.05)/\ln(0.95) = 58.4$. And note $n$ is independent of dimension — that's the real point. |
| "Stratified K-fold is better" | "Quantify it." | Fold Enrolled count is Binomial(708, 0.179), $\sigma = 10.2$ — a ±16% swing per fold. Stratification zeroes that source. Measured: ±0.023 → ±0.015. |
| "Apply SMOTE" | "Where exactly in the code?" | Inside an `imblearn` Pipeline, so it re-runs per fold. SMOTE before CV leaks across folds even if you never touch the test set. And try threshold moving first — it's free. |
| "Feature X is most important" | "How confident are you?" | Impurity importance is cardinality-biased, splits credit arbitrarily among correlated features, and is computed on training data. Check permutation importance on validation. |
| "Calibrate the model" | "On which data?" | Data the base model didn't train on. `CalibratedClassifierCV(cv=5)` cross-fits automatically. Fitting on training scores "corrects" a distortion that isn't there at test time. |
| "Occam's razor — prefer simpler" | "Define 'similar performance'." | Within the CV error bars. At ±0.015, models at 0.771 and 0.769 are the same model, and every other criterion becomes the tiebreaker. |

---

### Whiteboard-ready derivations

The three you should be able to produce cold, in order of how often they're asked.

#### D1 — The bagging variance formula, and why Random Forest exists

*Two minutes. This is the highest-value derivation in the lecture.*

**Step 1.** $B$ trees, each with prediction variance $\sigma^2$, pairwise correlation $\rho$.

$$\operatorname{Var}\!\left(\frac1B\sum_{b}X_b\right) = \frac{1}{B^2}\Big[\underbrace{\textstyle\sum_b \operatorname{Var}(X_b)}_{B \text{ terms of } \sigma^2} + \underbrace{\textstyle\sum_{b\neq b'}\operatorname{Cov}(X_b,X_{b'})}_{B(B-1)\text{ terms of }\rho\sigma^2}\Big]$$

**Step 2.**
$$= \frac{1}{B^2}\left[B\sigma^2 + B(B-1)\rho\sigma^2\right] = \frac{\sigma^2}{B} + \frac{B-1}{B}\rho\sigma^2$$

**Step 3.** Rearrange to the standard form:
$$= \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

**Step 4 — say the punchline out loud.** *"As $B \to \infty$ the second term goes to zero and the
first doesn't. $\rho\sigma^2$ is a floor. So once you have a few hundred trees, adding more buys
nothing, and the only remaining lever is reducing $\rho$ — which is exactly what `max_features` does.
Random Forest makes each tree worse to make the trees more different, and the formula says that's the
right trade."*

#### D2 — The out-of-bag 36.8%

*Thirty seconds. Almost always a warm-up.*

$$P(\text{row } i \text{ missed on one draw}) = 1 - \tfrac1n$$
$$P(\text{missed on all } n \text{ draws}) = \left(1-\tfrac1n\right)^n$$
$$\lim_{n\to\infty}\left(1-\tfrac1n\right)^n = e^{-1} = 0.3679$$

So ~36.8% out-of-bag, ~63.2% in-bag, and convergence is fast — at $n=100$ it's already 0.366.

**Say the punchline:** *"Every row is unseen by about 37% of the trees, so poll only those trees to
score it. That's a held-out estimate using no held-out data."*

#### D3 — Random search: 60 trials, 95% probability, top 5%

*One minute. Impresses because most candidates only quote it.*

Define "good" = top 5% of configuration space. Draw $n$ configs independently and uniformly.

$$P(\text{one draw misses}) = 0.95 \quad\Rightarrow\quad P(\text{all } n \text{ miss}) = 0.95^n$$
$$P(\text{at least one good}) = 1 - 0.95^n \;\ge\; 0.95$$
$$\Longleftrightarrow\; 0.95^n \le 0.05 \;\Longleftrightarrow\; n \ge \frac{\ln 0.05}{\ln 0.95} = \frac{-2.9957}{-0.05129} = 58.4$$

So $n = 59$; 60 gives $1 - 0.95^{60} = 0.954$.

**Say the punchline:** *"Notice the dimension of the search space never appears. Sixty trials works
for 2 hyperparameters or 20, while grid search costs $v^p$. That's the real argument — and the
second one is that grid search wastes trials re-testing duplicate values of whichever knob actually
matters."*

---

### Applied scenario — Amazon delivery-delay prediction

*The deck's own suggested application, from `[f15]`: "Delivery delay root-cause analysis —
interpretable rules for ops teams."*

**The ask.** Predict, at the moment a package is dispatched, whether it will miss its promised
delivery date. Two consumers: an **automated system** that proactively notifies customers and issues
credits, and an **ops team** that needs to know *why* delays cluster so they can fix the causes.

**Framing.** Those two consumers want different models and I'd build both — this is the "LogReg for
interpretable baseline, XGBoost for tabular" split (§13.5) with a genuine business reason behind it.

- **Model A (automated):** binary classification, `P(late)`. Optimised for accuracy of the
  probability, because a credit decision multiplies by it.
- **Model B (root cause):** a depth-4 decision tree, deliberately weak, whose *rules* are the
  deliverable. A 0.72-accuracy readable tree is worth more in the ops meeting than a 0.83-AUC forest.

I'd resist merging them. Trying to serve both from one model produces something that is neither
accurate enough to bid on nor simple enough to read.

**Data and the leakage audit.** Features available *at dispatch*: origin FC, destination zip, carrier,
service level, package dimensions and weight, day-of-week, hour, historical lane on-time rate,
weather forecast at destination, current FC backlog.

The audit question for every feature is: **is this knowable at dispatch?** Two traps:
- *"Number of carrier scans"* — populated *during* transit. A delayed package accrues more scans. This
  is textbook leakage and would look like a fantastic feature.
- *"Lane on-time rate"* — must be computed from a window that **ends before** the dispatch timestamp,
  and recomputed per fold. Computed once over the whole dataset, it contains the outcome of the very
  packages you're predicting.

**Validation — this is where the scenario is won or lost.** Delivery data is temporal, so random
K-fold is disqualified (§11.3). I'd use `TimeSeriesSplit` with an **expanding window and a gap**:
the gap must exceed the maximum promise horizon, because a package dispatched on day $T$ doesn't have
a known label until day $T+3$. Without the gap, the last three days of every training window contain
labels that weren't knowable then.

I'd also check for grouping: many packages per seller or per lane. If seller identity is a feature,
`GroupKFold` on seller nested inside the temporal split.

**Model.** Start with Random Forest — labels here are *mechanically derived* (did the delivery scan
land after the promise timestamp?), so they're clean, which by §8's rule argues for boosting. But I'd
run RF first anyway as a robust baseline with free OOB validation, then see whether an early-stopped
`HistGradientBoostingClassifier` beats it enough to justify the tuning. Categorical features are
high-cardinality here (zip codes, FC identifiers, carrier codes), which is a strong argument for
LightGBM or CatBoost with native categorical handling over one-hot encoding thousands of zips.

**Metric.** Late deliveries are the minority class — say 4%. So:
- **PR-AUC**, not ROC-AUC, for model comparison (§10, and question 9 above).
- **Precision at the operating threshold**, because every false positive is a proactive credit issued
  to a customer whose package was fine. That's real money.
- **Recall at that threshold**, because every miss is a customer who found out from an empty doorstep.
- **Brier score and ECE**, because Model A's output gets multiplied by a credit amount.
  Both RF and boosted trees are reliably miscalibrated, so `CalibratedClassifierCV` is not optional
  here — it's the difference between an expected-cost calculation that's right and one that's off by
  a systematic 3.7 percentage points.

The threshold itself comes from the cost ratio: if a proactive credit costs \$5 and a surprise late
delivery costs \$40 in contact-centre time plus churn risk, the break-even is around
$P(\text{late}) > 0.125$, not 0.5. **Setting that threshold is free and I'd do it before touching
SMOTE.**

**Failure modes I'd instrument for.**

| Failure | Why it happens | What I'd watch |
|---|---|---|
| **Concept drift at peak** | Q4 volume changes lane behaviour entirely; a model trained on October is wrong in December | Weekly PR-AUC on a rolling window; retrain cadence tied to it |
| **New categories** | New FC opens, new carrier onboarded → unseen category crashes the encoder | `handle_unknown='ignore'`, plus an alert on unseen-category rate |
| **No extrapolation** | Trees can't extrapolate. A backlog level higher than anything in training lands in whatever leaf the last threshold points to | Monitor feature ranges against training ranges; flag out-of-range inputs |
| **Feedback loop** | Predicting "late" triggers expediting, which makes it on-time, which teaches the model it was wrong | This is the subtle one — the intervention changes the label. I'd hold out a small random control group that is never expedited, and train on that |
| **Calibration decay** | Base rate shifts seasonally; the calibration map fitted in October is wrong in December | Track ECE on a rolling window, refit the calibrator (cheap) more often than the model (expensive) |

That fourth row is the one I'd raise unprompted. It's the same failure Part 1 flagged in the churn
scenario — **once you act on a prediction, the label stops being an observation of the world and
starts being an observation of your own intervention.** Any system that both predicts and intervenes
needs a control group, or its training data degrades silently.

**What I'd ship.** Week one: the depth-4 tree, to the ops team, with its rules written out in plain
English. It costs almost nothing and it's the thing that actually *fixes* delays rather than
predicting them. Week two to four: the calibrated Random Forest behind a threshold set from the cost
ratio, shadow-mode first — logging predictions without acting — so I can measure real precision at
the operating point before spending a cent on credits. Everything as a single
`ColumnTransformer + Pipeline` artifact so serving cannot drift from training.

---

### Leadership Principles tie-in

**Dive Deep.** The lecture's own notebook printed `HistGradientBoosting: train=0.999 test=0.755` and
moved on. Diving deep is noticing that a 24-point train–test gap on the *fastest, most modern*
algorithm means it was never early-stopped, that this is the exact failure mode boosting has and
bagging doesn't, and that the "state-of-the-art" model lost to a default Random Forest for a reason
you can name and fix. *Evidence:* "I noticed our best-performing offline model had a 24-point
train–test gap, traced it to a missing early-stopping config, and recovered 2 points of test accuracy
by fixing one parameter rather than by trying a new architecture."

**Insist on the Highest Standards.** Three results in this lecture — random-vs-grid search, Platt vs
isotonic, SMOTE vs class weights — are differences smaller than the measurement error, and reporting
any of them as a win would be wrong. Insisting on the highest standards means computing the standard
error *before* declaring a winner. *Evidence:* "Our A/B dashboard was reporting a 0.3% model
improvement as a win; I computed the standard error on our sample size at 1.4% and we stopped
shipping changes we couldn't actually detect."

**Customer Obsession.** §10 is the whole principle in one table: rebalancing made accuracy *worse* and
made the model *better*, because accuracy measures the majority and the customer who needs help is in
the minority. *Evidence:* "The team was tracking overall accuracy on our at-risk-student model. I
showed that our recall on the students who most needed intervention was 26% — we were missing three
out of four — and moved the team's primary metric to minority-class recall at a fixed precision
budget."

**Bias for Action.** Threshold moving costs nothing, needs no retraining, and is fully reversible; it
should always be tried before SMOTE, synthetic data, or a new model. *Evidence:* "Rather than
scheduling a two-week rebalancing experiment, I shifted the decision threshold on the existing model
and got the recall the business asked for in an afternoon, then ran the experiment properly to see if
we could do better."

---

## Glossary

| Term | Definition |
|---|---|
| **AdaBoost** | Boosting variant that reweights misclassified rows upward each round and combines learners by weighted vote. Sensitive to label noise because a wrong row's weight grows exponentially. |
| **Bagging** | Bootstrap Aggregating. Train $B$ models on bootstrap samples, combine by vote or average. Reduces variance, leaves bias unchanged. |
| **Balanced accuracy** | Mean of per-class recall. Equals accuracy when classes are balanced; unlike accuracy, cannot be gamed by ignoring a rare class. |
| **Bayesian optimisation** | Hyperparameter search that fits a surrogate model (usually a Gaussian Process) of the objective and picks the next point by maximising an acquisition function. Most sample-efficient; worth its overhead only for expensive training runs. |
| **Bootstrap sample** | $n$ rows drawn from $n$ rows **with replacement**. About 63.2% of unique rows appear; 36.8% are out-of-bag. |
| **Boosting** | Sequential ensembling where each model corrects its predecessors' errors. Reduces bias primarily. Can overfit with too many rounds. |
| **Brier score** | $\frac1N\sum(\hat p_i - y_i)^2$. MSE applied to probabilities. A proper scoring rule; decomposes into calibration + resolution. Lower is better, 0 is perfect. |
| **Calibrated** | A model is calibrated if $P(y=1 \mid \hat p = p) = p$ — of all cases where it said 80%, about 80% are positive. |
| **CalibratedClassifierCV** | sklearn wrapper that fits Platt or isotonic calibration on cross-fitted held-out folds. The `cv` parameter is what prevents fitting the calibrator on data the base model memorised. |
| **CART** | Classification And Regression Trees (Breiman et al., 1984). The binary-split, Gini-based tree algorithm sklearn implements. |
| **`ccp_alpha`** | sklearn's cost-complexity pruning parameter — the price per leaf in $R_\alpha(T) = R(T) + \alpha\lvert\tilde T\rvert$. Higher → more aggressive pruning. |
| **Class weight** | Per-class multiplier in the loss. `'balanced'` sets $w_k = N/(K n_k)$, giving every class equal total weight. |
| **`ColumnTransformer`** | sklearn component routing different column groups to different transformers (scaling for numeric, one-hot for categorical) and concatenating the results. |
| **Concept drift** | $P(y \mid x)$ changing over time — the same input now implies a different outcome. Distinct from data drift, where $P(x)$ changes. |
| **Cost-complexity pruning** | Post-pruning that minimises $R_\alpha(T) = R(T) + \alpha\lvert\tilde T\rvert$: training error plus a per-leaf penalty. Same structure as ridge regularisation. |
| **Cross-validation** | Rotating hold-out: split into $K$ folds, train on $K-1$ and validate on the remaining one, $K$ times, and average. Gives a mean *and* a standard deviation. |
| **Decision stump** | A tree of depth 1 — one question, two leaves. The canonical weak learner for AdaBoost. |
| **ECE** (Expected Calibration Error) | $\sum_b \frac{n_b}{N}\lvert\text{acc}(b) - \text{conf}(b)\rvert$. The sample-weighted average vertical distance from the diagonal on a reliability diagram. |
| **Entropy** | $-\sum_i p_i\log_2 p_i$. Average number of yes/no questions needed to identify a row's class. 0 when pure, $\log_2 K$ at maximum. |
| **Feature importance (impurity-based)** | Total sample-weighted impurity reduction attributable to a feature, normalised to sum to 1. Biased toward high-cardinality features; splits credit arbitrarily among correlated ones; computed on training data. |
| **Focal loss** | $-(1-p_t)^\gamma \log p_t$. Down-weights easy examples so the loss is dominated by hard ones. Introduced for dense object detection (Lin et al., 2017). |
| **Gini impurity** | $1 - \sum_i p_i^2$. The probability two independent draws from a node have different classes. 0 when pure, $1 - 1/K$ at maximum. |
| **Gradient boosting** | Boosting where each new tree fits the negative gradient of the loss with respect to the current predictions. Gradient descent in function space. |
| **`GroupKFold`** | Cross-validation splitter that keeps all rows sharing a group id (customer, session, device) inside the same fold. |
| **`HistGradientBoosting`** | sklearn's histogram-binned gradient booster (LightGBM-style). Much faster on large data; needs early stopping. |
| **Hyperparameter** | A setting chosen before training that controls how fitting happens (`max_depth`, `learning_rate`, `C`). Cannot be fitted on training loss, which is monotone in capacity. |
| **Information gain** | $H(\text{parent}) - \sum_k \frac{\lvert N_k\rvert}{\lvert N\rvert} H(k)$. Bits of uncertainty removed by a split. The size-weighting is what stops the tree peeling off tiny pure children. |
| **Isotonic regression** | Non-parametric monotone-increasing fit used for calibration. Flexible; needs ~1,000+ rows or it overfits. Preserves ranking, so AUC is unchanged. |
| **LOOCV** | Leave-One-Out CV: $K = n$. Nearly unbiased but high variance (the $n$ training sets are almost identical, so averaging barely helps) and costs $n$ fits. |
| **Macro-F1** | Unweighted mean of per-class F1. Gives a rare class the same vote as a common one — which is why it detects failures accuracy hides. |
| **`max_features`** | Number of features considered at each split. $\sqrt d$ (classification) or $d/3$ (regression) in Random Forest. The knob that attacks tree correlation $\rho$. |
| **No Free Lunch theorem** | Averaged over *all* possible problems, every learning algorithm performs identically (Wolpert, 1996). Practical reading: always compare at least two model families — not "all algorithms are equally good." |
| **Occam's Razor** | Among models with statistically indistinguishable performance, prefer the simpler one. "Indistinguishable" means within the CV error bars. |
| **OOB score** | Out-of-bag. Score each row using only the trees whose bootstrap sample excluded it. Free held-out validation. Invalid for temporal or grouped data. |
| **OvO** (One-vs-One) | $K(K-1)/2$ binary classifiers, one per class pair, combined by majority vote. Each trains on less data — good for superlinear learners like kernel SVM. Expensive beyond $K\approx10$. |
| **OvR** (One-vs-Rest) | $K$ binary classifiers, each one class against all others; predict by argmax of confidence. Scalable, but manufactures class imbalance in every sub-problem. |
| **Permutation importance** | Shuffle one feature's column in *validation* data and measure the score drop. Answers "does this help on unseen data?" — unbiased by cardinality. |
| **`Pipeline`** | sklearn object chaining transformers and a final estimator. Makes fit-on-training-fold-only automatic inside CV, which converts an entire class of leakage bug from "discouraged" to "impossible". |
| **Platt scaling** | Calibration by fitting a 2-parameter logistic $1/(1+e^{-(As+B)})$ to raw scores. Works on small data; can only fix sigmoid-shaped distortion. |
| **Post-pruning** | Grow the full tree, then remove subtrees that don't justify their complexity. Less myopic than pre-pruning; more expensive. |
| **Pre-pruning** | Stop growth early via `max_depth`, `min_samples_split`, `min_samples_leaf`. Cheap; `max_depth` alone does most of the work. |
| **Pseudo-residual** | $-\partial L/\partial F$ evaluated at the current predictions. Equals the plain residual $y - F$ for squared loss; $y - p$ for log loss. What each gradient-boosting tree is fitted to. |
| **Random Forest** | Bagging + random feature subsets at each split. The feature subsampling decorrelates trees, lowering the $\rho\sigma^2$ variance floor. |
| **Random search** | Sample hyperparameter configurations from distributions. 60 trials finds a top-5% config with 95% probability — **independent of the number of hyperparameters**. |
| **Reliability diagram** | Predicted probability (x) against observed frequency (y), binned. Diagonal = calibrated; below = overconfident; above = underconfident. |
| **SMOTE** | Synthetic Minority Over-sampling: create minority rows by interpolating between a minority point and a minority neighbour. **Training set only** — before a split, or outside a CV loop, it leaks. |
| **Softmax** | $e^{w_k^\top x}/\sum_j e^{w_j^\top x}$. Turns $K$ logits into a probability distribution. Native multi-class; the standard neural-net output layer. Reduces to the sigmoid at $K=2$. |
| **Stratified K-fold** | K-fold that preserves class proportions in every fold. Default for classifiers in sklearn. Removes a real source of fold-to-fold variance — measured here as ±0.023 → ±0.015. |
| **Successive halving** | Search strategy that evaluates many configs on a small budget, keeps the best fraction, multiplies the budget, repeats. `HalvingRandomSearchCV`. |
| **`TimeSeriesSplit`** | Expanding-window splitter where every validation fold is strictly later than its training rows. Mandatory for temporal data; random K-fold leaks the future. |
| **Tomek links** | Pairs of nearest neighbours from opposite classes. Deleting them cleans the decision boundary — commonly paired with SMOTE. |
| **Weak learner** | A model only slightly better than random. Boosting's foundational result is that arbitrarily many weak learners can be combined into an arbitrarily strong one. |
| **$\rho\sigma^2$ floor** | The irreducible term in the variance of an average of correlated estimators. Not removable by adding more estimators — only by making them less correlated. |

---

## Check yourself

Each question names the section that answers it. Cover the section and answer out loud.

**Trees (§1–§5)**

1. Derive Gini impurity from the "two random draws" definition. *(§2.1)*
2. What is the maximum Gini for 4 classes? For 2? Why do people get this wrong? *(§2.1)*
3. Compute the entropy of a node with $p = (0.5, 0.3, 0.2)$. *(§2.4 worked example)*
4. In the information gain formula, why are the children weighted by $\lvert N_k\rvert/\lvert N\rvert$? What goes wrong without it? *(§2.4)*
5. What kind of decision boundary can a tree *not* draw efficiently, and why? *(§1)*
6. Give the two distinct reasons a single tree overfits — one structural, one statistical. *(§3.2)*
7. Why is greedy splitting used rather than searching for the optimal tree? *(§3.1)*
8. State cost-complexity pruning's objective and name the model from Part 1 it structurally resembles. *(§4.2)*
9. Why can't you choose `max_depth` by training accuracy? *(§4.1)*
10. Give three reasons impurity-based feature importance can mislead you. *(§5)*

**Ensembles (§6–§8)**

11. Prove that bagging leaves bias unchanged. *(§6.1)*
12. Write the variance of an average of $B$ correlated estimators and identify the floor. *(§6.2)*
13. Why does Random Forest restrict features per split, given that this makes each tree worse? *(§6.2)*
14. Derive the 36.8% out-of-bag fraction. *(§6.3)*
15. Name two situations where OOB scoring is invalid. *(§6.3)*
16. Why does bagging want deep trees and boosting want shallow ones? *(§6.1, §7.1)*
17. Run one AdaBoost round with $\varepsilon = 0.3$ and 10 rows. What is the total weight on the misclassified set afterwards, and why is that value not a coincidence? *(§7.2)*
18. Why is "fit the residuals" only the squared-loss special case? *(§7.3)*
19. Explain why `learning_rate` and `n_estimators` are effectively one knob. *(§7.3)*
20. In what sense is XGBoost "parallelised", and in what sense is it definitely not? *(§7.4)*
21. Random Forest was flat under 30% label noise and gradient boosting lost 2.6 points. Explain the mechanism behind each. *(§8.2)*
22. RF scored 0.783 at 5% noise and 0.769 at 0%. Is that a real effect? Show your reasoning. *(§8.2)*

**The machinery (§9–§15)**

23. Compute macro-F1 from the OvR confusion matrix in §9.4 and say which class drives it. *(§9.4)*
24. Why does OvO help the minority class more than OvR does? *(§9.4)*
25. Compute `class_weight='balanced'` for $N=3539$, $K=3$, $n_k = (1137, 635, 1767)$ and verify each class ends with equal total weight. *(§10.2)*
26. Rebalancing lowered accuracy 2.7 points and raised minority F1 10.6. Why does accuracy move the wrong way? *(§10.5)*
27. Why did SMOTE + class_weight give byte-identical results to SMOTE alone? *(§10.5)*
28. Describe precisely how SMOTE-before-split leaks, naming which rows contaminate which. *(§10.3)*
29. What should you always try before SMOTE, and why? *(§10.3)*
30. Compute the standard deviation of the Enrolled count per fold under random 5-fold splitting, and use it to explain why stratification lowered the CV std from ±0.023 to ±0.015. *(§11.1)*
31. Give the two reasons LOOCV has high variance — and note which one is the $\rho\sigma^2$ floor in disguise. *(§11.2)*
32. Why is random K-fold catastrophic on time-series data, and what does `gap` fix? *(§11.3)*
33. Derive $n \ge 59$ for the random-search result, and say why the derivation's independence from dimension is the actual point. *(§12.3)*
34. Draw the 3×3 grid vs 9 random points picture and explain what grid search wastes. *(§12.3)*
35. The lecture's grid search *beat* random search, 0.684 to 0.681. Give three reasons that doesn't contradict the slide. *(§12.5)*
36. State the No Free Lunch theorem, then state what it does *not* imply. *(§13.1)*
37. Define "similar performance" in Occam's Razor operationally. *(§13.4)*
38. Why can calibration never change ROC-AUC? What two consequences follow? *(§14.4)*
39. Compute the Brier score for $\hat p = (0.9, 0.8, 0.6, 0.3, 0.1)$, $y = (1,1,0,0,0)$. *(§14.5)*
40. Compute ECE for the three-bin table in §14.5 and state what the number means in plain English. *(§14.5)*
41. Calibration halved ECE, barely moved Brier, and left AUC identical. Explain all three. *(§14.7)*
42. Why is Random Forest under-confident and boosting over-confident? *(§14.2)*
43. Write the CV code that leaks despite only touching `X_train`, and the one-line fix. *(§15.1)*
44. What does `handle_unknown='ignore'` prevent, and what does it cost you? *(§15.2)*
45. Give the correct nesting order of `SearchCV`, `Pipeline`, and `ColumnTransformer`, and say what breaks if you invert it. *(§15.2)*

**Synthesis (§16, Putting it together)**

46. The final pipeline scored CV F1 0.707 and test F1 0.679. Is a 0.028 gap a problem? *(§16.4)*
47. Name the three places the $\rho\sigma^2$ floor appears in this lecture. *(Thread 2)*
48. Name three results in this lecture that are inside the measurement error. *(Thread 4)*
49. Which three mechanisms in this lecture are the same "hold out, then evaluate" idea? *(Thread 5)*
50. Row 2 of the final round-trip was an Enrolled student predicted Dropout at 67.5%. Name three earlier results that predicted this exact failure. *(§16.4)*

---

## Going deeper

Ranked by importance. Difficulty: `intro` · `solid` · `hard`.

### Tier 1 — read these

1. **Breiman, "Random Forests" (2001), *Machine Learning* 45(1)** · `solid`
   The original. Section 2's variance analysis is the source of the $\rho\sigma^2$ argument in §6.2 —
   reading Breiman's own version of it is the fastest way to make that formula permanent.

2. **Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" (2001), *Annals of
   Statistics* 29(5)** · `hard`
   The paper that reframed boosting as gradient descent in function space. Dense, but §3–4 are where
   "fit the residuals" becomes "fit the negative gradient", and seeing that transition first-hand is
   worth the effort.

3. **Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Ch. 9, 10, 15** ·
   `solid`→`hard`
   Ch. 9 = trees, Ch. 10 = boosting, Ch. 15 = random forests. Free PDF from Stanford. The single best
   treatment of everything in §1–§8, by three of the people who invented it. Ch. 15.2's correlation
   analysis is D1 done properly.

4. **Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization" (2012), *JMLR* 13** · `solid`
   Short and highly readable. The deck's "60 trials / 95%" claim is from here, but the *real*
   contribution is the low-effective-dimensionality argument in §12.3 — that hyperparameters have
   unequal importance and grid search wastes trials on duplicate values of the important one.

### Tier 2 — the modern practice

5. **Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (KDD 2016)** · `solid`
   The three advances of §7.4 — regularisation inside the split criterion, second-order
   optimisation, and the approximate/sparsity-aware split finder — all in one clearly written paper.

6. **Niculescu-Mizil & Caruana, "Predicting Good Probabilities With Supervised Learning" (ICML 2005)**
   · `solid`
   The empirical study behind §14.2. It's where the "Random Forest pushes toward 0.5, boosting pushes
   outward" characterisations come from, with reliability diagrams for ten model families. Read this
   before you ever claim a model is or isn't calibrated.

7. **sklearn User Guide §1.11 (Ensembles), §3.1 (Cross-validation), §3.4 (Calibration), §6.1
   (Pipelines)** · `intro`
   Unusually good documentation — the calibration and pipeline pages in particular are better
   tutorials than most blog posts, with runnable examples and the reliability diagrams already
   plotted.

8. **Chawla, Bowyer, Hall & Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique" (2002),
   *JAIR* 16** · `solid`
   The original. Worth reading mainly to see how carefully the authors scoped their claims compared
   to how SMOTE is used today.

### Tier 3 — hands-on and current

9. **Grinsztajn, Oyallon & Varoquaux, "Why do tree-based models still outperform deep learning on
   tabular data?" (NeurIPS 2022)** · `solid`
   The careful benchmark behind §8.3's final claim, and — more usefully — three *mechanisms* for why:
   irregular target functions, uninformative features, and non-rotation-invariance.

10. **Kaggle: "Home Credit Default Risk" or "Santander Customer Transaction" — read the top
    solutions** · `intro`→`solid`
    Every technique in §9–§15 appears in the winning write-ups, applied under real constraints. The
    validation-strategy discussions are the most valuable part; the feature engineering is
    competition-specific.

11. **`imbalanced-learn` documentation, "Common pitfalls and recommended practices"** · `intro`
    Short, and directly about the §10.3 leakage trap. It shows the wrong and right code side by side
    and quantifies the inflation.

12. **Lin, Goyal, Girshick, He & Dollár, "Focal Loss for Dense Object Detection" (ICCV 2017)** · `solid`
    §10.4's source. Section 3's derivation of the modulating factor is two pages and completely clear.

**On citations:** these were checked against primary sources during the enhancement pass:

> ✅ **Confirmed** — Hyafil & Rivest (1976), *"Constructing Optimal Binary Decision Trees is
> NP-Complete"*, *Information Processing Letters* 5(1), pp. 15–17 (DOI 10.1016/0020-0190(76)90095-8).
> Result, authors, venue, and year all confirmed.
>
> ✅ **Confirmed** — Wolpert's No Free Lunch paper is *"The Lack of A Priori Distinctions Between
> Learning Algorithms"*, *Neural Computation* 8(7), 1996, pp. 1341–1390 — this is indeed the
> classification-side result cited in this section (confirmed the specific volume/issue/pages against
> the paper's own record, distinguishing it from the related but separate 1997 *"No Free Lunch
> Theorems for Optimization"* (Wolpert & Macready, *IEEE Transactions on Evolutionary Computation*),
> which is the optimisation-side companion this file correctly does not conflate it with).
>
> ✅ **Confirmed** — the GOSDT / optimal-decision-tree line of work referenced in §3.1's research note
> is Lin, Zhong, Hu, Rudin & Seltzer, *"Generalized and Scalable Optimal Sparse Decision Trees"*,
> ICML 2020 (see §3.1 for the full citation).
