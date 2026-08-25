---
title: "Supervised Learning — Part 2: Losses, Optimisation, Metrics & Three Classical Classifiers"
topic: supervised-learning
lecture: 02
source: "output/Lecture_02 - Module 1 Supervised Learning Part 2"
instructor: "Vikas Raykar"
slides: 35
video: "https://www.youtube.com/watch?v=0oFJgieacFQ"
---

# Supervised Learning — Part 2
### Losses, Optimisation, Metrics & Three Classical Classifiers

> ⚠️ **Capture note — read once, then forget it.**
>
> `slides_deduped/Lecture_02/` contains **21 images**, but the deck actually has **35 distinct
> states** (21 kept by the dedup step + the 14 it dropped, listed below). The de-duplication step
> merged genuinely different slides. The following were missing from the deduped set and were
> recovered from the full capture in `output/Lecture_02 - .../`, cited below by raw frame number and
> video timestamp:
>
> **Two numbering systems appear in this file, and they are not interchangeable.** An un-prefixed
> `slide_NNN` cited without the word "raw" refers to `slides_deduped/` numbering (only used for the 21
> slides that survived dedup intact); a citation marked **raw** `slide_NNN` (as in the recovery table
> immediately below, and in every citation later in this file) refers to `output/Lecture_02.../`
> numbering — the complete, un-deduplicated capture. If you open a frame yourself to check a citation,
> open it from the folder the citation says: `output/` for anything marked "raw", `slides_deduped/`
> only for the rare un-prefixed one.
>
> | Slide | Raw frame | Timestamp |
> |---|---|---|
> | Loss Functions (definition + the list of five) | `slide_009` | 2:02 |
> | Mean Squared Error (MSE) | `slide_011` | 3:15 |
> | Mean Absolute Error (MAE) | `slide_013` | 4:40 |
> | Comparing loss functions for regression (table) | `slide_019` | 7:03 |
> | Binary Cross-Entropy (BCE) | `slide_022` | 9:56 |
> | Mini-batch Stochastic Gradient Descent | `slide_042` | 18:51 |
> | Evaluation metrics (overview) | `slide_056` | 22:14 |
> | Classification: Confusion Matrix | `slide_060` | 25:43 |
> | Precision vs. Recall Trade-off | `slide_062` | 26:29 |
> | ROC-AUC & PR-AUC | `slide_065` | 28:36 |
> | K-Nearest Neighbors (KNN) | `slide_082` | 35:49 |
> | KNN: Practical Considerations | `slide_084` | 36:30 |
> | SVM: Core Idea | `slide_087` | 36:43 |
> | SVM: Kernel Trick | `slide_088` | 38:13 |
>
> **Every slide in this lecture is now accounted for** (21 + 14 = 35). Nothing below is reconstructed
> or guessed. Where I teach something the slide did not, it is explicitly marked `📚 Background the
> slide assumed`.

---

## How to read this document

| Layer | What it is | How to use it |
|---|---|---|
| **Main body** | The teaching. Every concept built from zero. | Read linearly, once, slowly. |
| **`interactive` blocks** | Machine-readable specs for the animated web version. | Skip on first read. |
| **Glossary + Check yourself** | Recall layer. | Come back weekly. |

Callout legend: 📚 background the slide assumed · 💡 key insight · ⚠️ careful · 🧪 worked example ·
🎯 interview · 🔬 research opportunity

---

## What you'll understand after reading this

By the end of this document you will be able to:

1. **Explain what a loss function *is* structurally** — a scalar-valued function of one prediction
   and one target — and why choosing it is the single most consequential modelling decision after
   choosing the model class.
2. **Derive, not recall, that MSE optimises toward the mean and MAE toward the median**, and use
   that fact to pick a loss from a business requirement.
3. **Show that MSE is maximum likelihood under Gaussian noise and MAE is maximum likelihood under
   Laplacian noise**, and explain why that reframing tells you which loss matches your data.
4. **Prove Huber loss is continuous and differentiable at its own kink**, and choose δ deliberately.
5. **Read the BCE and cross-entropy formulas out loud in English**, explain why the softmax must
   accompany cross-entropy, and compute both by hand.
6. **Explain why hinge loss produces sparse models and logistic loss does not**, from the shape of
   their gradients.
7. **Walk the whole optimiser family tree** — closed form → batch GD → SGD → mini-batch → momentum
   → adaptive → Adam/AdamW — and say what problem each step solves.
8. **Derive the learning-rate divergence threshold** on a quadratic and explain why "too big" fails.
9. **Build a confusion matrix, compute all six derived metrics by hand, and expose the accuracy
   paradox** on an imbalanced problem.
10. **Construct an ROC curve from raw scores and compute AUC two independent ways** that agree.
11. **Classify a document with Naive Bayes end to end**, including Laplace smoothing and the
    log-space computation, and explain why a false independence assumption still classifies well.
12. **Pick K in KNN from a bias–variance argument** and explain why KNN collapses in high dimensions.
13. **Compute an SVM margin by hand** and explain the kernel trick as an inner-product substitution
    rather than an explicit mapping.

---

## Before we start: what you need to know

The deck opens straight into loss functions and assumes the whole supervised-learning scaffold.
Here it is, taught from zero. If you already know a section, read the **bold** line and move on.

### Prerequisite 1 — The supervised learning setup and its notation

> **Supervised learning** — learning a function from examples where you were told the right answer.
>
> *In everyday words:* someone hands you a stack of flashcards with the question on the front and
> the answer on the back. You study them, then get tested on cards you've never seen.
>
> *Concretely:* 50,000 product listings, each with a title, price, and image, each labelled
> "counterfeit" or "genuine". You learn the mapping from listing to label.
>
> *Why it exists:* the alternative is writing the rules by hand. Nobody can write down the rule that
> separates counterfeit from genuine listings; but you can collect examples of both.

The notation used everywhere below, and on every slide in this deck:

| Symbol | Read it as | What it means |
|---|---|---|
| $x_i$ | "x sub i" | The $i$-th **input** (also: example, instance, feature vector). Usually a vector of $d$ numbers. |
| $y_i$ | "y sub i" | The **true target** for example $i$. A number for regression, a class label for classification. |
| $\hat{y}_i$ | "y-hat sub i" | The model's **prediction** for example $i$. The hat always means "estimated". |
| $n$ | "n" | Number of training examples. |
| $d$ | "d" | Number of features per example. |
| $C$ | "C" | Number of classes (classification only). |
| $\theta$ or $\mathbf{w}$ | "theta" / "w" | The **parameters** — the numbers the model learns. $\mathbf{w}$ specifically means a weight vector. |
| $b$ | "b" | The **bias** / intercept — a single number added to every prediction. |
| $f(x; \theta)$ | "f of x, given theta" | The **model**: a function that turns an input into a prediction using parameters $\theta$. |
| $\mathcal{L}$ | "script L" | The **loss**. |
| $\eta$ | "eta" | The **learning rate**. |
| $\nabla$ | "del" or "nabla" | The **gradient** operator. |

Two conventions that trip people up constantly:

- $\mathbf{w}^\top \mathbf{x}$ (read "w transpose x") is the **dot product**: multiply matching
  entries and add them up. If $\mathbf{w} = (2, -1, 3)$ and $\mathbf{x} = (1, 4, 2)$ then
  $\mathbf{w}^\top\mathbf{x} = 2(1) + (-1)(4) + 3(2) = 2 - 4 + 6 = \mathbf{4}$.
- The subscript $i$ indexes **examples**; the subscript $j$ indexes **features**; the subscript $c$
  indexes **classes**. When you see a double sum $\sum_i \sum_c$ you are summing over every example
  and, inside that, over every class.

### Prerequisite 2 — Model, parameters, training, inference

> **Model** — a family of functions with adjustable knobs.
>
> *Concretely:* $f(x; w, b) = wx + b$ is a model. Every choice of $(w, b)$ is one specific line.
> The model is the *set of all lines*; training picks one.

- **Parameters** are the knobs ($w$, $b$). Learned from data.
- **Hyperparameters** are knobs you set *before* training and the algorithm never touches — the
  learning rate $\eta$, the Huber threshold $\delta$, the SVM's $C$, KNN's $K$. Every one of these
  appears in this lecture, and the difference matters: parameters are fit, hyperparameters are
  *tuned* (usually by trying values on a validation set).
- **Training** is searching for the parameter values that make the loss small.
- **Inference** is using the fitted parameters on new inputs.

### Prerequisite 3 — Derivatives, gradients, and why they point downhill

The entire optimisation half of this lecture is one idea: *the gradient points uphill, so walk the
other way.* You need three things.

**A derivative is a slope.** $\frac{d}{dw}L(w)$ answers: if I nudge $w$ up by a tiny amount, how much
does $L$ change, and in which direction? Positive derivative → increasing $w$ increases $L$.

**A partial derivative is a slope along one axis.** When $L$ depends on many parameters,
$\frac{\partial L}{\partial w_j}$ asks the same question about $w_j$ only, holding everything else
fixed. Notation changes from $d$ to $\partial$; the meaning does not.

**A gradient is all the partial derivatives stacked into a vector.**

$$\nabla L(\mathbf{w}) = \left( \frac{\partial L}{\partial w_1},\ \frac{\partial L}{\partial w_2},\ \dots,\ \frac{\partial L}{\partial w_d} \right)$$

> 💡 **The one fact that makes gradient descent work:** among all directions you could step, the
> gradient is the direction of **steepest increase**. So $-\nabla L$ is the direction of steepest
> *decrease*. That is the whole algorithm.

🧪 **Worked example.** $L(w_1, w_2) = w_1^2 + 3w_1w_2$. Then
$\frac{\partial L}{\partial w_1} = 2w_1 + 3w_2$ and $\frac{\partial L}{\partial w_2} = 3w_1$.
At the point $(w_1, w_2) = (1, 2)$: $\nabla L = (2(1) + 3(2),\ 3(1)) = (8, 3)$.
So from $(1,2)$, the fastest way to *reduce* $L$ is to step in direction $(-8, -3)$.

### Prerequisite 4 — Convexity, local minima, global minima

> **Convex function** — one whose graph is bowl-shaped: the straight line joining any two points on
> the curve never dips below the curve.
>
> *Why you care:* **a convex function has exactly one minimum, and it is global.** Gradient descent
> on a convex loss cannot get stuck. MSE with a linear model, logistic loss, hinge loss, and the SVM
> objective are all convex. Neural network losses are not.

- **Local minimum** — lower than everything nearby, but maybe not lower than somewhere far away.
- **Global minimum** — the lowest point anywhere.
- A **saddle point** has zero gradient but is a minimum in one direction and a maximum in another.
  In high dimensions saddles vastly outnumber local minima, and they are the real obstacle for deep
  networks — not local minima.

### Prerequisite 5 — Probability: Gaussian, Laplace, likelihood, and MLE

Slides 5 and 6 both end with a line like "equivalent to MLE under Gaussian noise" and then move on.
That line is the deepest idea in the loss section, so here is everything needed to unpack it.

**Probability density.** For continuous quantities you cannot ask "what is the probability the error
is exactly 0.3?" — that's zero. You ask about density, written $p(\cdot)$: higher density means
values near there are more likely.

**The Gaussian (normal) distribution** with mean 0 and variance $\sigma^2$:

$$p(\varepsilon) = \frac{1}{\sqrt{2\pi\sigma^2}}\, \exp\!\left(-\frac{\varepsilon^2}{2\sigma^2}\right)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\varepsilon$ | "epsilon" | The error / noise: the gap between truth and the clean signal. |
| $\sigma^2$ | "sigma squared" | Variance — how spread out the noise is. |
| $\exp(\cdot)$ | "e to the" | The exponential function. |

The shape that matters: **the exponent contains $\varepsilon^2$**. Squared. Remember that.

**The Laplace distribution** with mean 0 and scale $b$:

$$p(\varepsilon) = \frac{1}{2b}\, \exp\!\left(-\frac{|\varepsilon|}{b}\right)$$

**The exponent contains $|\varepsilon|$**. Absolute value. Compared to a Gaussian, the Laplace has a
sharp peak at zero and **fatter tails** — it considers a huge error far less shocking than a
Gaussian does. That single fact is why MAE is robust to outliers, as you'll see.

**Likelihood** — given a candidate parameter value, how probable is the data I actually observed?
Note the inversion: probability asks "given the parameters, how likely is this data?", likelihood
takes the same number and treats it as a function of the *parameters* with the data fixed.

**Maximum Likelihood Estimation (MLE)** — pick the parameters that make the observed data most
probable. Because the examples are assumed independent, the likelihood of the whole dataset is a
*product*:

$$\mathcal{L}(\theta) = \prod_{i=1}^{n} p(y_i \mid x_i; \theta)$$

Products of many small numbers underflow to zero in floating point, and products are painful to
differentiate. So we always take the **log-likelihood**, because $\log$ turns products into sums and
is monotonically increasing (so it doesn't move the maximum):

$$\log \mathcal{L}(\theta) = \sum_{i=1}^{n} \log p(y_i \mid x_i; \theta)$$

Finally, optimisers minimise by convention, so we minimise the **negative** log-likelihood.

> 💡 **Every loss function in this lecture is a negative log-likelihood in disguise.** Choosing a
> loss is choosing a noise model. That is the sentence to take into an interview.

### Prerequisite 6 — Mean, median, and why they differ

- **Mean** — add up and divide by the count. Pulled hard by extreme values.
- **Median** — sort and take the middle. Ignores how extreme the extremes are.

For $[2, 4, 6, 8, 100]$: mean $= 120/5 = \mathbf{24}$, median $= \mathbf{6}$. One outlier moved the
mean by a factor of four and the median not at all. Hold on to these two numbers — they reappear in
the MSE/MAE worked example.

### Prerequisite 7 — Big-O notation

$O(\cdot)$ describes how cost grows with input size, ignoring constants. $O(nd)$ means "proportional
to the number of points times the number of dimensions". You need it for exactly two claims in this
lecture: the normal equation costs $O(d^3)$ (which is why it doesn't scale), and KNN prediction
costs $O(nd)$ per query (which is why it doesn't scale either, for a different reason).

---

## The big picture

Strip away the specifics and every supervised learning system in this lecture — and in industry — is
four boxes:

```
   ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
   │  MODEL   │───▶│   LOSS   │───▶│ OPTIMISER │───▶│  METRIC  │
   │ f(x; θ)  │    │  L(y,ŷ)  │    │   find θ* │    │ is it    │
   │          │    │          │    │           │    │  good?   │
   │ what     │    │ what     │    │ how do we │    │ did it   │
   │ shapes   │    │ counts   │    │ search?   │    │ work for │
   │ can I    │    │ as       │    │           │    │ the      │
   │ express? │    │ "wrong"? │    │           │    │ business?│
   └──────────┘    └──────────┘    └───────────┘    └──────────┘
        ▲                                                 │
        └──────────── if the metric says no ──────────────┘
```

This lecture is boxes 2, 3 and 4 — and then three complete models (Naive Bayes, KNN, SVM) that fill
box 1 in three radically different ways.

The single most important structural point, and the one the slides state explicitly at 22:14:
**box 2 and box 4 are different functions and they are allowed to disagree.** You train on a loss
because it is differentiable and smooth; you are judged on a metric because it reflects money,
safety, or user experience. Accuracy is not differentiable. You cannot descend it. So you minimise
cross-entropy and *hope* accuracy follows. Most of the hard engineering in applied ML lives in that
gap.

---

# Part 1 — Loss Functions

## 1. What a loss function is

The slide [raw `slide_009`, 2:02] gives the definition:

> A function $\mathcal{L}(y, \hat{y})$ that measures the discrepancy between the true target $y$ and
> the model prediction $\hat{y}$.
>
> - Defines the optimization objective.
> - Encodes our assumptions about the problem.
> - Different losses result in different model behaviors.

Take those three bullets seriously, because they are three genuinely different claims.

**"Defines the optimization objective"** — the loss is the *only* thing the optimiser can see. The
model does not know what a counterfeit listing is, or that a delivery estimate three days late is
worse than one day early. Everything you want the model to care about must be expressed as a number
the loss returns.

**"Encodes our assumptions"** — from Prerequisite 5: picking a loss is picking a noise model. Pick
MSE and you have *asserted* that your errors are Gaussian. If they aren't, you've quietly told the
optimiser something false about your data.

**"Different losses result in different model behaviors"** — the same model class, same data, same
optimiser, different loss gives you a genuinely different fitted model. §3 proves this with numbers.

> 📚 **Background the slide assumed — loss vs cost vs objective vs risk.** These four words get used
> interchangeably and shouldn't be:
>
> | Term | Means |
> |---|---|
> | **Loss** | The error on **one** example: $\mathcal{L}(y_i, \hat{y}_i)$. |
> | **Cost** | The average over the **training set**: $\frac{1}{n}\sum_i \mathcal{L}(y_i, \hat{y}_i)$. |
> | **Objective** | What you actually minimise — cost **plus** any regularisation term. |
> | **Risk** | The expected loss over the **true data distribution** — what you actually want, and can never compute. **Empirical risk** is the cost: the sample estimate of it. |
>
> Most people (and these slides) say "loss" for all four. Context disambiguates. But in an interview,
> using them precisely is a cheap signal of depth.

The deck then lists the five losses it will teach [`slide_009`]:

| # | Loss | Task |
|---|---|---|
| 1 | Mean Squared Error (MSE) | regression |
| 2 | Mean Absolute Error (MAE) | regression |
| 3 | Huber Loss | robust regression |
| 4 | Binary Cross-Entropy | binary classification |
| 5 | Hinge Loss | max-margin classification, used in SVMs |

Cross-entropy for multi-class is taught too, making six. We take them in order.

---

## 2. Mean Squared Error (MSE)

### Intuition first

You predicted 10, the truth was 13. You were off by 3. How bad is that? MSE's answer: **square it.**
Being off by 3 is nine times as bad as being off by 1, not three times.

That is the entire design decision, and everything else about MSE follows from it.

### The formula in words, then symbols

The formula says: **for each example, take the gap between truth and prediction, square it so that
sign doesn't matter and big gaps hurt disproportionately, then average over all examples.**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

| Symbol | Read it as | What it means |
|---|---|---|
| $n$ | "n" | Number of examples averaged over. Makes the loss comparable across dataset sizes. |
| $\sum_{i=1}^n$ | "sum from i equals 1 to n" | Add up the term that follows, once per example. |
| $y_i$ | "y sub i" | True target for example $i$. |
| $\hat{y}_i$ | "y-hat sub i" | Predicted value for example $i$. |
| $(y_i - \hat{y}_i)$ | "the residual" | The signed error. Positive means the model under-predicted. |

The slide [raw `slide_011`, 3:15] lists four properties. All four are worth unpacking.

### Property 1 — Differentiable everywhere

$(y - \hat y)^2$ is a parabola. Parabolas have no kinks. So the gradient exists at every single point
including the minimum, and gradient-based optimisers never hit an undefined value. Compare MAE,
which fails exactly this test.

### Property 2 — Penalises large errors quadratically

Errors of $1, 2, 10$ contribute $1, 4, 100$.

### Property 3 — Hence sensitive to outliers

This is Property 2 restated as a warning. Because a single error of 100 contributes 10,000 while a
hundred errors of 1 contribute 100 total, **one bad label can outvote a hundred good ones.**

### Property 4 — Equivalent to MLE under Gaussian noise

The slide states $y = f(x) + \varepsilon,\ \varepsilon \sim \mathcal{N}(0, \sigma^2)$ and stops.
Here is the derivation, which takes four lines.

Assume the target is the model output plus Gaussian noise. Then the density of observing $y_i$ given
$x_i$ is the Gaussian density evaluated at the residual:

$$p(y_i \mid x_i; \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i - f(x_i;\theta))^2}{2\sigma^2}\right)$$

Take the log of the whole dataset's likelihood (product → sum):

$$\log \mathcal{L}(\theta) = \sum_{i=1}^{n} \left[ -\frac{(y_i - f(x_i;\theta))^2}{2\sigma^2} - \log\sqrt{2\pi\sigma^2} \right]$$

Negate to get something to minimise, and drop every term that does not contain $\theta$ (the
$\log\sqrt{2\pi\sigma^2}$ term is a constant; the $\frac{1}{2\sigma^2}$ is a positive constant
multiplier which cannot move the location of the minimum):

$$-\log \mathcal{L}(\theta) \;\propto\; \sum_{i=1}^{n} (y_i - f(x_i;\theta))^2$$

Which is $n$ times MSE. **Minimising MSE and maximising Gaussian likelihood are the same
optimisation problem.** ∎

> 💡 The $\varepsilon^2$ in the Gaussian exponent became the $(\cdot)^2$ in MSE. The square did not
> come from nowhere — it came from the noise model.

### The gradient

The formula says: **the push on each prediction is proportional to how wrong it was.**

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{y}_i} = -\frac{2}{n}(y_i - \hat{y}_i)$$

The factor $(y_i - \hat y_i)$ is the key: gradient magnitude **scales with the error**. Big error →
big correction. This gives fast convergence when you're far away, and gentle steps near the optimum
— which is exactly the behaviour you want, and is why the slide later calls MSE's convergence "fast".

> ⚠️ You will often see MSE written with a $\frac{1}{2}$ in front. That is purely cosmetic: it makes
> the 2 from the derivative cancel. It does not change where the minimum is. This deck uses $\frac12$
> in the Huber formula and not in MSE, which is a genuine inconsistency in the slides — see §5.

### 🧪 Worked example — proving MSE optimises toward the mean

The claim on the MAE slide is "MSE optimizes toward the **mean** of the target distribution." Let's
prove it rather than trust it, using the simplest possible model: predict a single constant $c$ for
everything.

Data: $y = [2, 4, 6, 8, 100]$.

$$\mathcal{L}(c) = \frac{1}{5}\sum_{i=1}^{5}(y_i - c)^2$$

Differentiate with respect to $c$ and set to zero:

$$\frac{d\mathcal{L}}{dc} = \frac{1}{5}\sum_{i=1}^{5} -2(y_i - c) = 0
\quad\Longrightarrow\quad \sum_i y_i - 5c = 0
\quad\Longrightarrow\quad c = \frac{1}{5}\sum_i y_i$$

That is the definition of the mean. So $c^\star = 120/5 = \mathbf{24}$. ∎

Sanity-check numerically. At $c = 24$:
$$\mathcal{L} = \tfrac{1}{5}\left[22^2 + 20^2 + 18^2 + 16^2 + 76^2\right] = \tfrac{1}{5}\left[484 + 400 + 324 + 256 + 5776\right] = \tfrac{7240}{5} = \mathbf{1448}$$

At $c = 6$ (the median):
$$\mathcal{L} = \tfrac{1}{5}\left[16 + 4 + 0 + 4 + 8836\right] = \tfrac{8860}{5} = \mathbf{1772}$$

1448 < 1772, so MSE genuinely prefers 24. **And 24 is a terrible summary of this data** — four of the
five points are below 8. That is outlier sensitivity, in one number.

### Where people get confused

**You might think** MSE's outlier sensitivity is always a bug. **Actually** it is sometimes exactly
what you want. If you're predicting server load and being wrong by 10× causes an outage, you *want*
the loss to scream about large errors. Outlier sensitivity is a bug when outliers are **label noise**
and a feature when they are **real events with real cost**. Diagnose which before switching losses.

**You might think** "sensitive to outliers" and "not robust" mean the model will be worse. **Actually**
if your noise really is Gaussian, MSE is the *statistically optimal* choice — no other loss extracts
more information from the data. Robustness is insurance, and like all insurance it costs something.

```interactive
type: slider
title: One outlier, two fits
concept: MSE's outlier sensitivity vs MAE's robustness
control: Drag a single data point vertically away from an otherwise linear cloud of 20 points.
observe: Two fitted lines redraw live — the MSE line swings toward the dragged point, the MAE line barely moves. A readout shows both losses' fitted intercepts.
insight: The MSE line chases the outlier because the penalty grows quadratically; the MAE line ignores it because the penalty grows linearly and the point is only one vote.
fallback: The worked example in §2 — data [2,4,6,8,100] gives MSE optimum 24 and MAE optimum 6.
```

---

## 3. Mean Absolute Error (MAE)

### Intuition first

Same setup, one change: **don't square, just take the size.** Off by 3 is exactly three times as bad
as off by 1.

### Words, then symbols

The formula says: **average the absolute size of the errors.**

$$\mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^{n}\left|y_i - \hat{y}_i\right|$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\lvert \cdot \rvert$ | "absolute value of" | Throw away the sign. $\lvert -3 \rvert = 3$. |

The slide [raw `slide_013`, 4:40] gives four bullets and then two summary lines. Take them in turn.

### "Penalises all errors linearly — large errors are not disproportionately punished"

Errors of $1, 2, 10$ contribute $1, 2, 10$. Compare MSE's $1, 4, 100$.

### "Robust to outliers (unlike MSE)"

Follows directly. A wildly wrong point contributes its error once, not squared, so it cannot dominate.

### "Equivalent to MLE under Laplacian noise"

Same derivation shape as MSE. With $\varepsilon \sim \text{Laplace}(0, b)$:

$$p(y_i \mid x_i;\theta) = \frac{1}{2b}\exp\!\left(-\frac{|y_i - f(x_i;\theta)|}{b}\right)$$

$$-\log\mathcal{L}(\theta) = \sum_i \frac{|y_i - f(x_i;\theta)|}{b} + n\log(2b) \;\propto\; \sum_i |y_i - f(x_i;\theta)|$$

which is $n \times$ MAE. ∎

> 💡 **The pattern.** Gaussian exponent has $\varepsilon^2$ → squared loss. Laplace exponent has
> $|\varepsilon|$ → absolute loss. The loss is *literally the exponent of your assumed noise
> distribution, with the sign flipped.* Once you see this, you can invent a loss for any noise model
> you can write down. This is the single highest-leverage idea in the loss section.

### "Not differentiable at zero — use sub-gradient or smooth approximations"

$|e|$ has a sharp corner at $e = 0$. Approaching from the right the slope is $+1$; from the left it's
$-1$. There is no single slope at the corner, so the derivative is undefined there.

> 📚 **Background the slide assumed — sub-gradients.** For a convex function with a kink, a
> **sub-gradient** is any slope of a line that touches the function at that point and stays below it
> everywhere. At the kink of $|e|$, *any* value in $[-1, 1]$ qualifies. Optimisers simply pick one
> — PyTorch's `torch.abs` returns 0 at exactly zero — and everything works, because you land exactly
> on the kink with probability ~0 in floating point anyway.
>
> The "smooth approximations" the slide mentions are functions that agree with $|e|$ away from zero
> but round off the corner. Huber (§4) is precisely that, and it's why Huber comes next.

The gradient away from zero:

$$\frac{\partial\mathcal{L}_{\text{MAE}}}{\partial\hat{y}_i} = -\frac{1}{n}\,\text{sign}(y_i - \hat{y}_i)$$

**The magnitude is constant.** $\pm\frac{1}{n}$ regardless of whether you're off by 0.001 or by
1,000. This is why the comparison table calls MAE's convergence "slower (constant gradient)" — near
the optimum, MAE keeps taking full-size steps and bounces around the minimum instead of settling.
The standard fix is a decaying learning rate (§13).

### 🧪 Worked example — proving MAE optimises toward the median

Same data, $y = [2, 4, 6, 8, 100]$, same constant model.

$$\mathcal{L}(c) = \frac{1}{5}\sum_i |y_i - c|$$

The derivative with respect to $c$ is $\frac{1}{5}\sum_i -\text{sign}(y_i - c)$, i.e.
$\frac{1}{5}\left[(\#\text{points below } c) - (\#\text{points above } c)\right]$.

Setting this to zero means: **the number of points below $c$ must equal the number above.** That is
the definition of the median. So $c^\star = \mathbf{6}$. ∎

Numerically, at $c = 6$:
$$\mathcal{L} = \tfrac{1}{5}\left[4 + 2 + 0 + 2 + 94\right] = \tfrac{102}{5} = \mathbf{20.4}$$
At $c = 24$ (the mean):
$$\mathcal{L} = \tfrac{1}{5}\left[22 + 20 + 18 + 16 + 76\right] = \tfrac{152}{5} = \mathbf{30.4}$$

20.4 < 30.4. MAE prefers 6. **Same data, same model, different loss, different answer — 6 versus 24.**
That is bullet three of the loss-function slide made concrete.

> 🎯 **The interview version of this.** "MSE targets the mean, MAE the median" is a fact. The
> *reason* is that the derivative of a square is proportional to the error (so points vote with
> weight equal to their distance) while the derivative of an absolute value is $\pm 1$ (so every
> point gets exactly one vote regardless of distance). Say the reason, not the fact.

### Where people get confused

**You might think** you should always use MAE because it's robust. **Actually** if you are forecasting
demand and your metric is total inventory cost, the mean is what you want, because costs add up
linearly across the whole distribution, not around its middle. Match the loss to the **decision**,
not to a general preference for robustness.

---

## 4. Huber Loss

### The problem it solves

You now have two losses with complementary flaws:

| | MSE | MAE |
|---|---|---|
| Near zero error | smooth, gradient shrinks → converges nicely | kinked, gradient constant → bounces |
| Far from zero | penalty explodes → outliers dominate | penalty linear → robust |

The obvious question: can we have MSE's behaviour where errors are small and MAE's where they're
large? Huber is exactly that, welded together at a chosen threshold.

### Words, then symbols

The formula says: **if the error is small enough, use the squared penalty; once it exceeds a
threshold $\delta$, switch to a straight line — and shift that line down by just enough that the two
pieces meet with no jump and no kink.**

$$\mathcal{L}_\delta(y, \hat y) = \begin{cases}
\frac{1}{2}(y - \hat y)^2 & \text{if } |y - \hat y| \le \delta \\[6pt]
\delta\left(|y - \hat y| - \frac{\delta}{2}\right) & \text{otherwise}
\end{cases}$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\delta$ | "delta" | The **transition threshold** — a hyperparameter you choose. Errors below it are "normal", above it are "outliers". |
| $\frac12(y-\hat y)^2$ | — | The quadratic (MSE-like) branch. The $\frac12$ is *not* cosmetic here; it makes the pieces join smoothly. |
| $\delta(|y-\hat y| - \frac\delta2)$ | — | The linear (MAE-like) branch, scaled by $\delta$ and offset by $-\delta^2/2$. |

The slide [`slide_004`, 4:43] lists exactly the four properties we just motivated:

- Quadratic for small errors (like MSE) — smooth, fast convergence near optimum
- Linear for large errors (like MAE) — robust to outliers
- **Differentiable everywhere** (unlike MAE)
- $\delta$ controls the transition point between quadratic and linear regimes

### Derivation — why those specific constants?

The slide asserts "differentiable everywhere". Let's prove it, because the proof explains where the
$\frac{\delta}{2}$ came from.

Write $e = y - \hat y$ and check the boundary $e = \delta$ from both sides.

**Continuity — do the two pieces meet?**

- Quadratic branch at $e = \delta$: $\ \frac12\delta^2$
- Linear branch at $e = \delta$: $\ \delta(\delta - \frac\delta2) = \delta \cdot \frac\delta2 = \frac{\delta^2}{2}$

Equal. ✓ No jump.

**Differentiability — do the slopes match?**

- Quadratic branch: $\frac{d}{de}\left[\frac12 e^2\right] = e$, which at $e = \delta$ equals $\delta$.
- Linear branch (for $e > 0$): $\frac{d}{de}\left[\delta(e - \frac\delta2)\right] = \delta$.

Equal. ✓ No kink. **Huber is $C^1$ — continuous with a continuous first derivative — everywhere.** ∎

> 💡 That $-\frac{\delta}{2}$ offset is not decoration. It is the unique constant that makes the
> linear piece land exactly on the parabola's value at the crossover. Remove it and the loss jumps
> discontinuously at $|e| = \delta$, and the optimiser sees a cliff.

The gradient — and here is the practical payoff:

$$\frac{\partial \mathcal{L}_\delta}{\partial e} = \begin{cases} e & |e| \le \delta \\ \delta \cdot \text{sign}(e) & |e| > \delta \end{cases}$$

**The gradient is capped at $\pm\delta$.** No matter how catastrophically wrong a single example is,
it can never push the parameters harder than $\delta$. This is *identical in effect to gradient
clipping*, a technique you'll meet again in deep learning — Huber achieves it by construction rather
than by post-hoc surgery on the gradient.

### 🧪 Worked example

Four examples, residuals $e = [0.5,\ 1.5,\ 3,\ 10]$, with $\delta = 2$.

| $e$ | $\lvert e\rvert \le \delta$? | Branch | Value |
|---|---|---|---|
| 0.5 | yes | $\frac12(0.5)^2$ | $0.125$ |
| 1.5 | yes | $\frac12(1.5)^2$ | $1.125$ |
| 3 | no | $2(3 - 1)$ | $4$ |
| 10 | no | $2(10 - 1)$ | $18$ |

$$\mathcal{L}_{\text{Huber}} = \frac{0.125 + 1.125 + 4 + 18}{4} = \frac{23.25}{4} = \mathbf{5.8125}$$

Compare, on the same residuals, using the matching $\frac12 e^2$ convention for MSE:

$$\mathcal{L}_{\text{MSE}} = \frac{0.125 + 1.125 + 4.5 + 50}{4} = \frac{55.75}{4} = \mathbf{13.9375}$$
$$\mathcal{L}_{\text{MAE}} = \frac{0.5 + 1.5 + 3 + 10}{4} = \frac{15}{4} = \mathbf{3.75}$$

**5.8125 sits between 3.75 and 13.9375**, exactly as the design intends. And look where the
difference comes from: the $e = 10$ point contributes 50 to MSE, 18 to Huber, 10 to MAE. Huber let
that point matter — but not run the show.

### Choosing δ

- **Too small** → almost every residual is in the linear regime → Huber degenerates into MAE (times a
  constant), and you lose the smooth convergence you came for.
- **Too large** → almost every residual is quadratic → Huber degenerates into MSE, and you lose
  robustness.
- **Practical rule:** set $\delta$ to roughly the level of error you consider "normal" — a common
  starting point is the median absolute residual of a quick baseline fit, or $1.345\hat\sigma$, which
  is the classical choice giving ~95% efficiency relative to MSE under genuinely Gaussian noise.
  ✅ **Confirmed and corrected.** The commonly-quoted constant is **1.345**, not 1.35 (the file
  previously rounded it) — confirmed against Huber, *"Robust Estimation of a Location Parameter"*,
  *Annals of Mathematical Statistics* 35(1), 1964, pp. 73–101, and the standard robust-statistics
  literature that cites the 95%-efficiency result from it.
- **Scale sensitivity:** $\delta$ is in the units of $y$. If you rescale your target from dollars to
  cents, you must rescale $\delta$ by 100. This catches people constantly.

> 🔬 **Research opportunity.** $\delta$ is a fixed hyperparameter, but the "right" boundary between
> signal and outlier is not constant across a dataset — it varies by region, by segment, by time.
> Adaptive-$\delta$ and learned robust losses (e.g. Barron's general robust loss, which places
> MSE, MAE, Huber and Cauchy on a single continuous family with a learnable shape parameter) are an
> active area. ✅ Confirmed — Jonathan T. Barron, *"A General and Adaptive Robust Loss Function"*,
> CVPR 2019 (arXiv:1701.03077).

```interactive
type: slider
title: The delta dial
concept: How δ interpolates Huber between MSE and MAE
control: A slider for δ from 0.1 to 10, replicating the slide's own interactive figure [slide 5].
observe: The Huber curve is drawn over fixed dashed MSE (½e²) and MAE (|e|) references, with the quadratic zone |e| ≤ δ shaded. As δ → 0 the Huber curve collapses onto MAE; as δ grows it hugs MSE.
insight: δ is not a tuning nuisance — it is the literal dividing line between "this is noise I should learn from" and "this is an outlier I should cap".
fallback: The table in §4 — at δ=2 the residual e=10 contributes 50 under MSE, 18 under Huber, 10 under MAE.
```

---

## 5. Comparing MSE, MAE and Huber

Slide [raw `slide_019`, 7:03] gives this table directly. It is worth memorising as a unit.

| | **MSE** | **MAE** | **Huber** |
|---|---|---|---|
| **Penalty growth** | Quadratic | Linear | Quadratic (small $r$), Linear (large $r$) |
| **Outlier sensitivity** | High | Low | Low (controlled by $\delta$) |
| **Differentiable** | Yes | No (at 0) | Yes |
| **Optimizes toward** | Mean | Median | Mean (small errors) / Median-like (large errors) |
| **Noise assumption** | Gaussian | Laplacian | Gaussian core + heavy tails |
| **Convergence** | Fast | Slower (constant gradient) | Fast near optimum |

Three extra rows the slide didn't include, which you should carry anyway:

| | **MSE** | **MAE** | **Huber** |
|---|---|---|---|
| **Units of the loss** | $y^2$ (take $\sqrt{}$ → RMSE to fix) | same as $y$ | mixed |
| **Extra hyperparameter** | none | none | $\delta$ |
| **sklearn / PyTorch** | `squared_error` / `nn.MSELoss` | `absolute_error` / `nn.L1Loss` | `huber` / `nn.HuberLoss(delta=)` |

> ⚠️ **A genuine inconsistency in the deck.** The MSE slide writes $\mathcal{L} = \frac1n\sum(y-\hat
> y)^2$ with **no** $\frac12$, but the Huber slide's quadratic branch is $\frac12(y-\hat y)^2$
> **with** one. Both conventions are standard; the deck just switches between them. It does not
> affect any minimiser, but it *does* mean you cannot directly compare the numeric values of "MSE"
> and "Huber" across the two slides without picking one convention. The worked example in §4 uses
> $\frac12 e^2$ for both so the comparison is apples to apples.

### A decision procedure

```
Are large errors genuinely more costly than proportionally?
├── YES → MSE. (outages, safety margins, physical blowups)
└── NO
    ├── Do you have label noise / data-entry outliers?
    │   ├── YES, lots  → MAE
    │   └── YES, some  → Huber  ◀── the default when unsure
    └── NO → MSE (it's the MLE under Gaussian, and converges fastest)

Then always: is your reported *metric* the same as your training loss?
If you train MSE but report MAE, expect the model to look worse than
its loss curve suggests — you optimised for the mean and are being
scored on the median.
```

---

## 6. Binary Cross-Entropy (BCE)

Regression is done. Classification needs an entirely different shape of loss, and the reason is
worth stating before any formula: **classifiers output probabilities, and you cannot score a
probability with a squared error without breaking the gradient.**

### Words, then symbols

The formula says: **for each example, look only at the probability the model assigned to the class
that actually occurred, take its logarithm (which is a large negative number when the probability is
small), and average the negatives of those.**

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\Big[\,y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i)\,\Big]$$

| Symbol | Read it as | What it means |
|---|---|---|
| $y_i$ | "y sub i" | The true label, exactly **0 or 1**. |
| $\hat p_i$ | "p-hat sub i" | The model's predicted probability that $y_i = 1$. A number in $(0,1)$. |
| $\log$ | "log" | Natural logarithm (base $e$) throughout ML unless stated otherwise. |
| the minus sign | — | Turns a log-likelihood (which we'd maximise) into a loss (which we minimise). |

**The switch mechanism.** Because $y_i$ is exactly 0 or 1, one of the two terms always vanishes:

- If $y_i = 1$: the bracket is $1\cdot\log\hat p_i + 0 \cdot \log(1-\hat p_i) = \log \hat p_i$.
- If $y_i = 0$: the bracket is $0 + 1\cdot\log(1-\hat p_i) = \log(1 - \hat p_i)$.

So the formula is really just **"$-\log$ (probability assigned to the correct class)"**, written in a
way that avoids an `if` statement. Once you see that, BCE stops looking intimidating.

### The slide's three claims

The slide [raw `slide_022`, 9:56] says BCE:

1. **Heavily penalises confident wrong predictions.**
2. **Is equivalent to minimising negative log-likelihood.**
3. **Can be interpreted in terms of KL divergence.**

**Claim 1** is the shape of $-\log$. As $\hat p \to 0$, $-\log \hat p \to \infty$. The slide's own
worked numbers, which I've verified:

| True label | Predicted $\hat p$ | Loss $=-\log \hat p$ |
|---|---|---|
| 1 | 0.01 | $-\log(0.01) = \mathbf{4.605}$ |
| 1 | 0.99 | $-\log(0.99) = \mathbf{0.01005}$ |

**Being confidently wrong costs 458× more than being confidently right saves.** That asymmetry is
deliberate: a classifier that says "99% sure" had better be right 99% of the time.

**Claim 2** — the derivation. A binary label is a **Bernoulli** random variable: it takes value 1 with
probability $p$ and 0 with probability $1-p$. Its probability mass function can be written in one
line, using the same exponent trick as above:

$$P(y \mid \hat p) = \hat p^{\,y}(1-\hat p)^{1-y}$$

(Check: $y=1$ gives $\hat p^1 (1-\hat p)^0 = \hat p$ ✓; $y=0$ gives $\hat p^0(1-\hat p)^1 = 1 - \hat p$ ✓.)

Log-likelihood over the dataset:

$$\log\mathcal{L} = \sum_i \Big[ y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)\Big]$$

Negate, divide by $n$, and that is exactly $\mathcal{L}_{\text{BCE}}$. ∎

**Claim 3** needs three definitions the slide never gives.

> 📚 **Background the slide assumed — entropy, cross-entropy, KL divergence.**
>
> **Entropy** $H(p) = -\sum_c p_c \log p_c$ — the average number of nats (bits, if you use $\log_2$)
> needed to encode outcomes drawn from $p$, using the *best possible* code for $p$. It measures
> genuine uncertainty. A fair coin has $H = \log 2 \approx 0.693$ nats; a two-headed coin has $H=0$.
>
> **Cross-entropy** $H(p, q) = -\sum_c p_c \log q_c$ — the average code length when the truth is $p$
> but you built your code assuming $q$. Always $\ge H(p)$, with equality only when $q = p$.
>
> **KL divergence** $D_{\text{KL}}(p \parallel q) = H(p,q) - H(p) = \sum_c p_c \log\frac{p_c}{q_c}$ —
> the *excess* code length you pay for being wrong about the distribution. It is $\ge 0$, and $=0$
> only when $p = q$. It is **not** symmetric: $D_{\text{KL}}(p\|q) \ne D_{\text{KL}}(q\|p)$, so it
> is not a distance despite constantly being described as one.
>
> **Why this makes claim 3 true:** in supervised learning the "true distribution" $p$ for one example
> is a point mass — all probability on the observed label. For a point mass, $H(p) = 0$. Therefore
> $$D_{\text{KL}}(p \parallel \hat p) = H(p, \hat p) - \underbrace{H(p)}_{=0} = H(p, \hat p)$$
> **Minimising cross-entropy is minimising KL divergence, exactly, because the entropy term is a
> constant zero.** That is the whole content of claim 3. (If your labels are soft — from label
> smoothing or distillation — then $H(p) \ne 0$ and the two differ by that constant, which still
> doesn't change the minimiser.)

### 🧪 Worked example

Three examples: $y = [1, 0, 1]$, $\hat p = [0.9,\ 0.2,\ 0.4]$.

| $i$ | $y_i$ | $\hat p_i$ | probability of the **true** class | contribution $-\log(\cdot)$ |
|---|---|---|---|---|
| 1 | 1 | 0.9 | $0.9$ | $0.10536$ |
| 2 | 0 | 0.2 | $1 - 0.2 = 0.8$ | $0.22314$ |
| 3 | 1 | 0.4 | $0.4$ | $0.91629$ |

$$\mathcal{L}_{\text{BCE}} = \frac{0.10536 + 0.22314 + 0.91629}{3} = \frac{1.24479}{3} = \mathbf{0.41493}$$

Note example 3: the model said 40% for a positive — not confidently wrong, just wrong — and it
contributes more than the other two combined.

### 📚 The gradient, and the miracle that makes it work

This is the most important thing in the section and no slide mentions it.

A binary classifier produces a raw score $z$ (a **logit**, any real number) and squashes it through
the **sigmoid** to get a probability:

$$\hat p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Now differentiate BCE with respect to the *logit* $z$, not the probability. Using
$\frac{d\sigma}{dz} = \sigma(1-\sigma)$:

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat p}\cdot\frac{\partial \hat p}{\partial z} = \left(\frac{-y}{\hat p} + \frac{1-y}{1-\hat p}\right)\cdot \hat p(1-\hat p)$$

Expand the product:

$$= -y(1-\hat p) + (1-y)\hat p = -y + y\hat p + \hat p - y\hat p = \boxed{\hat p - y}$$

> 💡 **The gradient is just "prediction minus truth".** Every $\sigma$, every $\log$, cancels. This is
> why sigmoid+BCE are always paired, and why using MSE on a sigmoid output is a well-known mistake:
> with MSE the gradient picks up an extra $\sigma(1-\sigma)$ factor, which is ~0 whenever the model
> is confident. A confidently *wrong* MSE-trained classifier gets almost no gradient and cannot
> recover. This is the **vanishing gradient** problem in its simplest form.

> ⚠️ **Numerical stability.** Never compute `log(sigmoid(z))` literally. If $z = -50$, `sigmoid(z)`
> underflows to `0.0` and `log(0.0)` is `-inf`, which poisons the whole batch to `nan`. Every
> framework provides a fused, stable version: PyTorch's `nn.BCEWithLogitsLoss` (feed it logits, not
> probabilities) and `nn.CrossEntropyLoss`. **Using `BCELoss` after a manual `sigmoid` is the single
> most common numerical bug in beginner PyTorch code.**

```interactive
type: slider
title: The cost of confidence
concept: Why BCE punishes confident errors asymptotically
control: A slider for the predicted probability p̂ from 0.001 to 0.999, mirroring the slide's own figure [slide 10].
observe: Two curves — loss if the true label is 1 (−log p̂) and loss if it is 0 (−log(1−p̂)) — with the current point marked and both losses displayed numerically.
insight: The curve is finite and flat in the middle but vertical at the ends. There is no upper bound on the cost of being certain and wrong.
fallback: The table in §6 — p̂=0.01 with y=1 costs 4.605; p̂=0.99 with y=1 costs 0.01005.
```

---

## 7. Cross-Entropy Loss (CE) and softmax

### Words, then symbols

BCE handles two classes. Cross-entropy generalises it to $C$ classes.

The formula says: **for every example and every class, multiply the true indicator for that class by
the log of the predicted probability for that class, add them all up, negate, and average. Because
the true indicator is 1 for exactly one class and 0 elsewhere, this is again just "minus the log of
the probability you gave the right answer".**

$$\mathcal{L}_{\text{CE}} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C} y_{i,c}\,\log \hat p_{i,c}$$

| Symbol | Read it as | What it means |
|---|---|---|
| $C$ | "C" | Number of classes. |
| $y_{i,c}$ | "y sub i,c" | 1 if example $i$ truly belongs to class $c$, else 0. This vector across $c$ is the **one-hot encoding**. |
| $\hat p_{i,c}$ | "p-hat sub i,c" | Predicted probability that example $i$ is class $c$. Must sum to 1 across $c$. |

> 📚 **Background the slide assumed — one-hot encoding.** A label like "class 2 of 4" is stored as
> the vector $(0, 1, 0, 0)$: a 1 in the true slot, 0 elsewhere. It is used because class indices
> $\{0,1,2,3\}$ are not numbers you can do arithmetic on — class 3 is not "three times" class 1 —
> and one-hot removes any implied ordering.

Because exactly one $y_{i,c}$ is 1, the inner sum collapses to a single term, and CE with $C=2$ is
algebraically identical to BCE. They are the same loss.

### Softmax — where the probabilities come from

A network's final layer outputs $C$ raw scores $z_1,\dots,z_C$ (**logits**), which can be any real
numbers — negative, huge, whatever. Cross-entropy needs probabilities. **Softmax** is the converter.

The formula says: **exponentiate every score so they're all positive, then divide each by the total
so they sum to one.**

$$\hat p_c = \text{softmax}(z)_c = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}$$

| Symbol | Read it as | What it means |
|---|---|---|
| $z_c$ | "z sub c" | The raw logit for class $c$. |
| $e^{z_c}$ | "e to the z_c" | Makes it positive, and *amplifies differences* — a logit 2 larger becomes $e^2 \approx 7.4\times$ larger. |
| $\sum_j e^{z_j}$ | "the partition function" | The normaliser. Guarantees the outputs sum to 1. |

### 🧪 Worked example

Three classes, logits $z = (2.0,\ 1.0,\ 0.1)$, true class is class 1 (the first one).

**Step 1 — exponentiate.**
$$e^{2.0} = 7.389,\qquad e^{1.0} = 2.718,\qquad e^{0.1} = 1.105$$

**Step 2 — sum.**
$$7.389 + 2.718 + 1.105 = 11.212$$

**Step 3 — divide.**
$$\hat p = \left(\tfrac{7.389}{11.212},\ \tfrac{2.718}{11.212},\ \tfrac{1.105}{11.212}\right) = (\mathbf{0.6590},\ \mathbf{0.2424},\ \mathbf{0.0986})$$

(Sum: $0.6590 + 0.2424 + 0.0986 = 1.0000$ ✓)

**Step 4 — the loss** is $-\log$ of the probability given to the true class:
$$\mathcal{L}_{\text{CE}} = -\log(0.6590) = \mathbf{0.4170}$$

**Sanity anchors worth memorising:** a uniform prediction over $C$ classes gives loss $\log C$. For
$C = 3$ that is $\log 3 = 1.0986$. Our 0.4170 is well below it, so the model has learned something.
For $C = 50{,}000$ (a language-model vocabulary) random guessing is $\log 50000 \approx 10.8$ — which
is why "loss 10.8" is the number people quote for an untrained LLM.

### 📚 Two things about softmax that will be asked

**Shift invariance.** Adding the same constant $k$ to every logit changes nothing:

$$\frac{e^{z_c + k}}{\sum_j e^{z_j+k}} = \frac{e^k e^{z_c}}{e^k\sum_j e^{z_j}} = \frac{e^{z_c}}{\sum_j e^{z_j}}$$

This is not a curiosity — it is the **fix for numerical overflow**. If your logits are around 1000,
$e^{1000}$ is `inf`. Every implementation therefore subtracts $\max_j z_j$ first, making the largest
exponent exactly $e^0 = 1$. That is the **log-sum-exp trick**, and it is why you should call
`F.log_softmax` rather than `log(softmax(x))`.

**Temperature.** Divide logits by $T$ before the softmax: $\hat p_c \propto e^{z_c/T}$.
- $T < 1$ sharpens the distribution toward the argmax (more confident, more deterministic).
- $T > 1$ flattens it (more uncertain, more diverse).
- $T \to 0$ recovers a hard argmax; $T \to \infty$ recovers uniform.

You will meet this again as the sampling temperature in LLMs and as the softening parameter in
knowledge distillation.

> ⚠️ **Softmax vs sigmoid — get this right.** Softmax forces the probabilities to sum to 1, so it
> encodes **"exactly one of these classes"** (multi-*class*). If an example can carry several labels
> at once — a product that is both "electronics" and "gift" — you need $C$ independent sigmoids and
> $C$ independent BCE terms (multi-*label*). Using softmax for a multi-label problem forces the
> classes to compete for a fixed budget of probability, which is simply the wrong model.

---

## 8. Hinge Loss

### The problem it solves

BCE asks the model to output a well-calibrated probability. But often you don't need a probability —
you need a **decision**, with a comfortable safety buffer. Hinge loss is the loss for that, and it is
the loss that defines the SVM (§23).

### Words, then symbols

The formula says: **if the example is on the correct side of the boundary by a comfortable distance,
charge nothing at all. Otherwise charge in proportion to how far short of comfortable it falls.**

$$\mathcal{L}_{\text{hinge}} = \frac{1}{n}\sum_{i=1}^{n}\max\big(0,\ 1 - y_i \cdot f(x_i)\big)$$

with, as the slide states, $y_i \in \{-1, +1\}$ and $f(x_i) = \mathbf{w}^\top x_i + b$.

| Symbol | Read it as | What it means |
|---|---|---|
| $y_i \in \{-1,+1\}$ | — | The label convention **changes here**: not $\{0,1\}$ but $\{-1,+1\}$. |
| $f(x_i)$ | "f of x i" | The raw signed score. Positive → predict class $+1$. Its **sign** is the prediction; its **magnitude** is the confidence. |
| $y_i \cdot f(x_i)$ | "the **functional margin**" | Positive when the prediction is correct, negative when wrong, and larger when more confident. |
| $\max(0, \cdot)$ | "the hinge" | Clamps negatives to zero — this is what creates the flat region. |

> 💡 **Why $\{-1,+1\}$ instead of $\{0,1\}$?** So that the single product $y \cdot f(x)$ encodes both
> correctness and confidence. If $y = +1$ and $f = 3$, the product is $+3$: right, confidently. If
> $y = -1$ and $f = 3$, the product is $-3$: wrong, confidently. One number, both facts. With
> $\{0,1\}$ labels you'd need a case split.

### The slide's four bullets

Slide [`slide_008`, 10:39] states:

1. **Zero loss when $y_i \cdot f(x_i) \ge 1$** (correct with sufficient margin)
2. **Not differentiable at the hinge point.**
3. **Encourages a margin between classes.**
4. **Sparse solution: only support vectors contribute to gradient.**

**Bullet 1 — the "1" is the point.** Note that the threshold is not 0. Getting the *sign* right
(margin > 0) is not enough; the loss keeps pushing until the margin exceeds **1**. That gap between
0 and 1 is the safety buffer, and it is where the word "margin" comes from.

**Bullet 2** — at $y f(x) = 1$ exactly, the function switches from a downward slope of $-1$ to a flat
0. Same sub-gradient story as MAE (§3). Handled identically.

**Bullet 3** follows from bullet 1: because loss is only zero past margin 1, the optimiser is
actively rewarded for pushing points *away* from the boundary, not merely onto the right side.

**Bullet 4 — the sparsity result, derived.** The gradient with respect to $\mathbf{w}$ is:

$$\frac{\partial}{\partial \mathbf{w}}\max(0, 1 - y_i f(x_i)) = \begin{cases} -y_i x_i & \text{if } y_i f(x_i) < 1 \\ \mathbf{0} & \text{if } y_i f(x_i) > 1 \end{cases}$$

**Exactly zero** for every comfortably-correct point. Those points contribute *nothing* to the update
— you could delete them from the training set and get the identical model. The only points that
matter are those with $y_i f(x_i) \le 1$: the ones on or inside the margin. Those are the **support
vectors**. ∎

### The comparison with logistic loss

The slide's own comparison, verbatim:

> Hinge is "flat" for well-classified points; logistic loss always has a nonzero gradient → hinge
> produces sparser models.

This is worth sitting with, because it explains a real architectural difference between SVMs and
logistic regression.

| | Hinge | Logistic (BCE) |
|---|---|---|
| Loss at margin 5 | exactly **0** | $\log(1+e^{-5}) \approx 0.0067$ — small but **not** zero |
| Gradient at margin 5 | exactly **0** | $\approx -0.0067$ — tiny but **not** zero |
| Effect | far-away points are ignored entirely | far-away points keep nudging the boundary forever |
| Result | solution depends only on support vectors | solution depends on all $n$ points |
| Output | a score; no calibrated probability | a calibrated probability |

> 🎯 If asked "when would you prefer an SVM over logistic regression?", the sparsity property is the
> real answer: with $n$ large but the decision boundary determined by few points, the SVM's solution
> is compact and its prediction cost depends only on the number of support vectors. If you need
> **probabilities** — for ranking, for expected-value decisions, for thresholding at a business-chosen
> operating point — logistic regression gives them natively and the SVM does not (Platt scaling can
> retrofit them, at the cost of a second fitting stage).

### 🧪 Worked example

| $y$ | $f(x)$ | margin $y f(x)$ | $1 - yf(x)$ | $\max(0,\cdot)$ | reading |
|---|---|---|---|---|---|
| $+1$ | $2.0$ | $+2.0$ | $-1.0$ | $\mathbf{0}$ | correct, comfortably — free |
| $+1$ | $0.5$ | $+0.5$ | $+0.5$ | $\mathbf{0.5}$ | correct, but inside the margin — still charged |
| $+1$ | $-0.3$ | $-0.3$ | $+1.3$ | $\mathbf{1.3}$ | wrong |
| $-1$ | $-2.0$ | $+2.0$ | $-1.0$ | $\mathbf{0}$ | correct, comfortably — free |
| $-1$ | $0.3$ | $-0.3$ | $+1.3$ | $\mathbf{1.3}$ | wrong |

$$\mathcal{L}_{\text{hinge}} = \frac{0 + 0.5 + 1.3 + 0 + 1.3}{5} = \frac{3.1}{5} = \mathbf{0.62}$$

Row 2 is the row to study: **the model got that example right and was still penalised.** No other
loss in this lecture does that. It is the entire idea of a margin.

(The slide's own interactive figure shows exactly this case: $y\cdot f(x) = 0.50$, hinge loss
$0.500$, "Classified correctly? Yes", "Inside margin? Yes".)

> 📚 **Background the slide assumed — 0/1 loss and surrogates.** What you *actually* want to minimise
> for classification is the **0/1 loss**: 1 if wrong, 0 if right. It is exactly the error rate. But
> it is piecewise constant, so its gradient is zero everywhere it's defined and undefined at the
> jump — completely useless for gradient descent, and minimising it exactly is NP-hard in general.
>
> So we minimise a **surrogate**: a convex function that sits *above* the 0/1 loss and can be
> descended. Hinge, logistic and exponential (AdaBoost's) losses are the three classical surrogates.
> All three are upper bounds on 0/1 loss, so driving the surrogate down drives the error rate down.
> **This is the reason classification losses look the way they do**, and the slides never say it.

---

# Part 2 — Optimisation & Solvers

## 9. The optimisation problem

Slide [`slide_010`, 12:46] states the goal:

> **Goal:** Find parameters $\theta^*$ that minimize the empirical risk.
>
> $$\theta^* = \arg\min_{\theta}\ \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}\big(y_i,\ f(x_i;\theta)\big)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\arg\min_\theta$ | "arg min over theta" | Not the minimum *value* — the $\theta$ **at which** the minimum occurs. |
| $\theta^*$ | "theta star" | The optimal parameters. The star always means "the best one". |
| $\frac1n\sum_i \mathcal{L}(\cdot)$ | "the empirical risk" | Average loss over the training data. |

> 📚 **Background the slide assumed — Empirical Risk Minimisation (ERM), and its central flaw.**
> What you *want* is the parameters minimising **true risk** — expected loss over the real
> distribution of listings/customers/queries you'll face in production. You cannot compute it; you
> have a finite sample. So you minimise the **empirical** risk on that sample and hope it transfers.
>
> **This hope is exactly what fails when you overfit.** Empirical risk goes to zero, true risk goes
> up, because the parameters started explaining the sample's noise rather than its signal. The
> defences — regularisation, early stopping, validation sets, cross-validation — all exist to patch
> this one gap, and the gap is created the moment you write "$\frac1n\sum_i$" instead of
> "$\mathbb{E}$".
>
> - **Overfitting** — training loss low, validation loss high. Model memorised.
> - **Underfitting** — both high. Model too simple, or trained too little.

The slide then names the two families and asks the organising question:

> **Two broad approaches:**
> 1. **Closed-form (Normal Equation)** — exact solution, limited to specific forms
> 2. **Iterative (Gradient-based)** — general purpose, scalable
>
> **Key question:** How much data do we use to compute the gradient at each step?
> - All data → Batch Gradient Descent
> - One sample → Stochastic Gradient Descent (SGD)
> - m samples → Mini-batch SGD

That "key question" is the spine of the next four sections. Everything is one trade-off: **gradient
accuracy versus gradient cost.**

### 📚 The closed-form solution, and why we abandon it

The slide names the normal equation and moves on. Here it is, derived, because knowing *why* it fails
is what justifies everything after it.

For linear regression with MSE, write the data as a matrix $X$ ($n \times d$) and targets as a vector
$y$. The cost is $\|y - Xw\|^2$. Expand:

$$\mathcal{L}(w) = (y - Xw)^\top(y - Xw) = y^\top y - 2w^\top X^\top y + w^\top X^\top X w$$

Differentiate with respect to $w$ and set to zero:

$$\nabla_w \mathcal{L} = -2X^\top y + 2X^\top X w = 0 \quad\Longrightarrow\quad X^\top X w = X^\top y$$

$$\boxed{\ w^* = (X^\top X)^{-1} X^\top y\ }$$

This is the **normal equation**, and it is the exact global optimum in one shot — no learning rate,
no iterations, no convergence criterion. It is also, in the slide's words, "limited to specific
forms". Three reasons it doesn't scale:

1. **Cost.** $X^\top X$ is $d \times d$; inverting it is $O(d^3)$. With $d = 100{,}000$ features (a
   bag-of-words model, say) that is $10^{15}$ operations. Not happening.
2. **Memory.** You must hold $X^\top X$: $d^2$ numbers. At $d=100{,}000$ in float32 that's 40 GB.
3. **Existence.** $(X^\top X)^{-1}$ requires $X^\top X$ to be invertible. It is not, whenever two
   features are perfectly correlated or $d > n$. Both are routine.

**Ridge regression is the fix for (3),** and it is worth seeing because it explains why L2
regularisation is numerically as well as statistically motivated:

$$w^*_{\text{ridge}} = (X^\top X + \lambda I)^{-1}X^\top y$$

Adding $\lambda I$ inflates every diagonal entry, which guarantees invertibility for any $\lambda > 0$.
The regulariser you added to prevent overfitting *also* fixed your linear algebra. That is not a
coincidence — both problems are the same problem (directions in which the data carries no
information).

And crucially: **the normal equation only exists because MSE + linear model happens to have a
closed-form stationary point.** Change the loss to MAE, or the model to a neural network, and there
is no such formula. Iterative methods work for everything. That is why the rest of the lecture — and
essentially all of modern ML — is about them.

---

## 10. Batch Gradient Descent

### Words, then symbols

Slide [`slide_011`, 14:16] states it directly:

> We take a small step in the direction of the **negative gradient**.

$$\mathbf{w}^t \leftarrow \mathbf{w}^{t-1} - \eta \nabla L(\mathbf{w}^{t-1})$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\mathbf{w}^t$ | "w at step t" | Parameters **after** update $t$. Superscript is an iteration counter, **not** an exponent. |
| $\leftarrow$ | "is assigned" | Assignment, not equality. |
| $\eta$ | "eta" | The **learning rate**: how far to step. The slide notes $\eta > 0$. |
| $\nabla L(\mathbf{w}^{t-1})$ | "gradient of L at w" | Direction of steepest **increase**, computed at the current parameters. |
| the minus sign | — | Why we go *down*hill. Delete it and you have gradient **ascent**. |

The slide's code:

```python
for epoch in range(n_epochs):
    dw = gradient(loss, data, w)   # ← uses ALL of `data`
    w  = w - lr * dw
```

And its verdict: *"Each step requires that the **entire training data** be processed to compute the
gradient. For large datasets this is not computationally efficient."*

### 🧪 Worked example — five steps by hand

Minimise $L(w) = (w - 3)^2$. Gradient: $\nabla L = 2(w-3)$. Take $\eta = 0.1$, start at $w^0 = 0$.

| $t$ | $w^{t-1}$ | $\nabla L = 2(w-3)$ | $\eta\nabla L$ | $w^t$ | distance to 3 |
|---|---|---|---|---|---|
| 1 | 0 | $-6$ | $-0.6$ | $\mathbf{0.6}$ | 2.4 |
| 2 | 0.6 | $-4.8$ | $-0.48$ | $\mathbf{1.08}$ | 1.92 |
| 3 | 1.08 | $-3.84$ | $-0.384$ | $\mathbf{1.464}$ | 1.536 |
| 4 | 1.464 | $-3.072$ | $-0.3072$ | $\mathbf{1.7712}$ | 1.2288 |
| 5 | 1.7712 | $-2.4576$ | $-0.24576$ | $\mathbf{2.01696}$ | 0.98304 |

Look at the last column: $2.4,\ 1.92,\ 1.536,\ 1.2288,\ 0.98304$. Each is exactly $0.8\times$ the one
before. That is not a coincidence.

### Derivation — the learning rate divergence threshold

Subtract 3 from both sides of the update to track the *error* $e^t = w^t - 3$:

$$e^t = e^{t-1} - \eta \cdot 2e^{t-1} = (1 - 2\eta)\,e^{t-1}$$

So the error is multiplied by $(1-2\eta)$ every step. Three regimes:

| Condition | $\lvert 1-2\eta \rvert$ | Behaviour |
|---|---|---|
| $0 < \eta < 0.5$ | $< 1$, positive | smooth convergence (our case: $1 - 0.2 = 0.8$ ✓) |
| $\eta = 0.5$ | $0$ | converges in **one step** |
| $0.5 < \eta < 1$ | $<1$, negative | converges while **oscillating** across the minimum |
| $\eta = 1$ | $=1$ | oscillates forever between two points, never converging |
| $\eta > 1$ | $>1$ | **diverges** |

Generalising: for a quadratic with second derivative $L''$, the update multiplier is $(1 - \eta L'')$,
so convergence requires

$$\boxed{\ \eta < \frac{2}{L''}\ }$$

For a general multi-dimensional loss, $L''$ becomes the **largest eigenvalue of the Hessian** (the
matrix of all second derivatives), often written $L$ and called the **smoothness constant**. The
condition $\eta < 2/L$ is one of the most-quoted results in optimisation.

> 📚 **Background — the Hessian and eigenvalues, in one paragraph.** The Hessian is the matrix of all
> second partial derivatives; it describes the local curvature of the loss surface in every
> direction. Its **eigenvalues** are the curvatures along its principal axes. The largest eigenvalue
> is the steepest, most sharply-curved direction — and that is the direction that will blow up first
> if $\eta$ is too large. The ratio of largest to smallest eigenvalue is the **condition number**: when
> it's big the loss surface is a long narrow valley, gradient descent zig-zags across the walls
> instead of running down the floor, and this is precisely the problem momentum and the adaptive
> methods in §13 exist to solve.

> ⚠️ **Learning-rate failure modes, and how to recognise them from a loss curve:**
>
> | Symptom | Cause | Fix |
> |---|---|---|
> | Loss decreases painfully slowly, nearly linear | $\eta$ far too small | raise by 3–10× |
> | Loss decreases then plateaus high | $\eta$ slightly too small, or a real local structure | LR schedule; check the data |
> | Loss oscillates but trends down | $\eta$ near the stability edge | halve it |
> | Loss goes to `NaN` within a few steps | $\eta > 2/L$; gradients exploded | drop 10×, add gradient clipping |
>
> The standard practical recipe is an **LR range test**: train for a few hundred steps while
> exponentially increasing $\eta$, plot loss against $\eta$, and pick roughly an order of magnitude
> below where it turns upward.

> 📚 **Background the slide assumed — what "till convergence" actually means.** The slide says the
> update "is repeated multiple times (till covergence)" [sic] and never defines it. In code, you stop
> when one of these fires:
>
> 1. **Max iterations** — a hard budget. Always have one.
> 2. **Gradient norm** $\|\nabla L\| < \epsilon$ — you're at a flat point. The most principled test.
> 3. **Parameter change** $\|w^t - w^{t-1}\| < \epsilon$ — you've stopped moving.
> 4. **Loss change** $|L^t - L^{t-1}| < \epsilon$ — cheapest, and the most misleading: a plateau on a
>    saddle looks identical to a minimum.
> 5. **Early stopping on validation loss** — stop when *validation* loss stops improving for
>    `patience` epochs. This is the one you actually use in practice, because it targets
>    generalisation rather than optimisation.
>
> ⚠️ The **patience trap**: with `patience=1` you'll stop on the first noisy uptick. Typical values
> are 5–20 epochs, and you must **restore the best checkpoint**, not the last one — otherwise you
> keep the worse model you trained during the patience window.

---

## 11. Stochastic Gradient Descent (SGD)

### The setup the slide gives

Slide [`slide_012`, 15:57]:

> In general most loss functions can be written as sum over each training instance.
> $$L(\mathbf{w}) = \sum_{i=1}^{N} L_i(\mathbf{w})$$
> In Stochastic Gradient Descent (SGD) we update the parameters **one data point at a time**.
> $$\mathbf{w}^t \leftarrow \mathbf{w}^{t-1} - \eta\nabla L_i(\mathbf{w}^{t-1})$$
> A complete passthrough of the whole dataset is called an **epoch**.

That first line is the enabling observation. Because the total loss **decomposes as a sum over
examples**, its gradient decomposes too:

$$\nabla L(\mathbf{w}) = \sum_{i=1}^{N}\nabla L_i(\mathbf{w})$$

So one example's gradient is a *sample* of the full gradient. That is the entire justification.

### Why a single noisy sample works — the unbiasedness argument

The slide asserts SGD works but doesn't say why it's legitimate. Here's the one-line reason.

If you pick index $i$ uniformly at random, the **expected value** of the single-example gradient is:

$$\mathbb{E}_i\left[\nabla L_i(\mathbf{w})\right] = \frac{1}{N}\sum_{i=1}^{N}\nabla L_i(\mathbf{w}) = \frac{1}{N}\nabla L(\mathbf{w})$$

**The stochastic gradient is an unbiased estimator of the true gradient** (up to the constant $1/N$,
absorbed into $\eta$). It is wrong on every individual step, but right *on average*. Over many steps
the errors cancel and you move in the correct direction — for $1/N$ of the compute per step.

> 📚 **Background — epochs vs iterations.** With $N = 1{,}000{,}000$ examples:
> - **Batch GD**: 1 epoch = 1 parameter update, using 1,000,000 gradient computations.
> - **SGD**: 1 epoch = 1,000,000 parameter updates, using 1,000,000 gradient computations.
>
> Same compute, **a million times more progress**. That is the whole argument for SGD, and it is
> overwhelming.

### The cost, which the slide states plainly

> SGD is much faster and more computationally efficient, but it has **noise in the estimation of the
> gradient**. Since it updates the weight frequently, it can lead to big oscillations and that makes
> the training process highly unstable.

Both halves are true and both matter. The noise is not purely a cost, though:

> 💡 **Noise can help.** A noisy gradient can knock you out of a shallow local minimum or off a
> saddle point that a clean batch gradient would sit on forever. There is also strong empirical
> evidence that SGD's noise biases training toward *flatter* minima, which generalise better than
> sharp ones. So SGD is not merely a cheap approximation to batch GD — on non-convex problems it
> often finds *better* solutions. ⚠️ The flat-minima-generalise-better claim is well supported
> empirically but still debated theoretically; present it as evidence, not theorem.

> 📚 **Background — Robbins–Monro conditions.** For SGD to provably converge on a convex problem, the
> learning rate must decay, satisfying $\sum_t \eta_t = \infty$ (so you can still reach the optimum
> from anywhere) and $\sum_t \eta_t^2 < \infty$ (so the noise is eventually damped out).
> $\eta_t = \eta_0/t$ satisfies both. In deep learning people rarely use exactly this, but every
> learning-rate schedule you'll meet is descended from it.

The slide's code makes the structure explicit:

```python
for epoch in range(n_epochs):
    for i in range(n_data):
        dw = gradient(loss, data[i], w)   # ← ONE example
        w  = w - lr * dw
```

---

## 12. Mini-batch Stochastic Gradient Descent

Slide [raw `slide_042`, 18:51] — one of the slides missing from the deduped set, and the single most
practically important slide in the optimisation section, because **this is what everyone actually
runs.**

> Using a single example results in a very noisy estimate of the gradient. So we use a small random
> subset of data called **mini-batch** of size $B$ (**batch size**) to compute the gradient.
>
> $$\mathbf{w}^t \leftarrow \mathbf{w}^{t-1} - \eta\nabla L_{batch}(\mathbf{w}^{t-1})$$
>
> Mini-batch SGD is the most commonly used method and is sometimes referred to as just SGD.
> Typical choices of the batch size are $B = 32, 64, 128, 256, \dots$
> In practice we do a random shuffle of the data per epoch. In practice, mini-batch SGD is the most
> frequently used variation because it is both computationally cheap and results in more robust
> convergence.

```python
for epoch in range(n_epochs):
    for mini_batch in get_batches(data, batch_size):
        dw = gradient(loss, mini_batch, w)
        w  = w - lr * dw
```

### The three things this slide is really telling you

**1. Batch size is the noise dial, and the law is $1/\sqrt{B}$.**

Averaging $B$ independent unbiased estimates reduces the standard deviation of the estimate by
$\sqrt{B}$:

$$\text{std}\left(\frac1B\sum_{i=1}^B g_i\right) = \frac{\sigma}{\sqrt{B}}$$

| $B$ | Gradient noise vs $B=1$ | Compute vs $B=1$ |
|---|---|---|
| 1 | $1.00\times$ | $1\times$ |
| 32 | $0.177\times$ | $32\times$ |
| 128 | $0.088\times$ | $128\times$ |
| 1024 | $0.031\times$ | $1024\times$ |

**Going from 32 to 1024 costs 32× the compute for 5.7× less noise.** That diminishing return is
exactly why batch sizes settle in the hundreds and not the millions.

**2. Why the sizes are powers of two.** $B \in \{32, 64, 128, 256\}$ is not superstition. GPUs
execute in fixed-width warps (32 threads on NVIDIA hardware) and their matrix kernels are tiled to
power-of-two shapes. A batch of 32 and a batch of 33 take nearly the same wall-clock time; the 33rd
example is close to free but forces a second, almost-empty tile. **Mini-batching is the operation
that turns a sequence of vector operations into one matrix multiply**, which is the only reason GPUs
help with training at all.

**3. Why shuffle every epoch.** If your data is sorted by label — all the class-0 examples first —
then without shuffling, every batch in the first half is pure class 0. The model will lurch toward
predicting 0, then lurch back. Shuffling makes each mini-batch a representative sample, which is
exactly the condition the unbiasedness argument in §11 requires. Reshuffling *each* epoch also
prevents the model from memorising a fixed batch composition.

> 💡 **The linear scaling rule.** When you increase batch size by $k$, multiply the learning rate by
> $k$ as well. Reasoning: each step's gradient is $\sqrt{k}$ times less noisy but you're taking $k$
> times fewer steps per epoch, so to cover the same distance you need proportionally larger steps.
> In practice this holds well up to a few thousand and then breaks, requiring a **warmup** period
> (start with a small $\eta$ and ramp up over the first few hundred steps) to avoid diverging early
> while the gradients are still large and uninformative. ✅ Confirmed — Goyal, Dollár, Girshick,
> Noordhuis, Wesolowski, Kyrola, Tulloch, Jia & He, *"Accurate, Large Minibatch SGD: Training ImageNet
> in 1 Hour"*, arXiv:1706.02677 (2017). The paper's own warmup schedule ramps over the first 5 epochs,
> not a few hundred steps — the shorter warmup window this file states is the more commonly used
> convention in later practice, not a claim about this specific paper's exact numbers.

### The three-way comparison

| | Batch GD | SGD ($B=1$) | Mini-batch SGD |
|---|---|---|---|
| Gradient per step | exact | very noisy | moderately noisy |
| Updates per epoch | 1 | $N$ | $N/B$ |
| Memory per step | whole dataset | one example | $B$ examples |
| GPU utilisation | good but rarely fits | terrible (no parallelism) | **excellent** |
| Escapes saddles/shallow minima | no | yes | yes |
| Convergence path | smooth | very jagged | moderately smooth |
| **In practice** | almost never | almost never | **always** |

> ⚠️ **A vocabulary trap that catches people in interviews.** When a paper or a colleague says "SGD",
> they mean **mini-batch SGD**, exactly as this slide warns. `torch.optim.SGD` is mini-batch SGD —
> the batching happens in the DataLoader, not the optimiser. True $B=1$ SGD is essentially never
> used. If you're asked "what's the difference between SGD and mini-batch SGD?", the correct answer
> starts by noting that the terms have collapsed in practice, and *then* gives the textbook
> distinction.

```interactive
type: simulator
title: Three descents on one surface
concept: The variance/cost trade-off between batch, stochastic and mini-batch gradient descent
control: Choose batch size B ∈ {1, 8, 32, full} and learning rate; press run.
observe: Three trajectories trace over a contour plot of a 2-D loss surface with a narrow valley — batch GD glides smoothly, B=1 scribbles wildly, B=32 sits between. A counter shows updates-per-epoch and wall-clock estimate alongside.
insight: B=1 reaches the neighbourhood of the optimum in the fewest gradient computations but never settles; batch GD settles precisely but takes one step per epoch. B=32 is not a compromise — it is strictly better on a GPU than either.
insight2: Raising B by 4× only halves the noise — the 1/√B law is visible as the trajectory getting smoother far more slowly than the compute grows.
fallback: The 1/√B table in §12 and the three-way comparison table above it.
```

---

## 13. Improvements: momentum, adaptive rates, and schedules

Slide [`slide_013`, 20:17] is a bare list of names:

> - One of the basic improvements over SGD comes from adding a **momentum** term.
> - Different learning rate for each parameter: **Adagrad**, **RMSProp**
> - Combine momentum with adaptive learning rate: **Adam**, **AdamW**, **Muon**, **SOAP**
> - Learning rate schedule: **Cosine annealing**

Eight names, zero explanation. This section is that explanation. They form a clean lineage where each
one fixes the previous one's specific failure.

### The problem all of them solve

Plain SGD uses **one** learning rate for **every** parameter and has **no memory** of previous steps.
Both are bad:

- **No memory** → in a long narrow valley (a badly conditioned loss surface), the gradient points
  mostly *across* the valley, not along it. You zig-zag between the walls and creep along the floor.
- **One rate for all** → a parameter that appears in every example (say, a bias) and a parameter that
  appears in one example in ten thousand (a rare word's embedding) get the same step size. The rare
  one barely ever learns.

### Momentum — fixing "no memory"

The formula says: **keep a running average of recent gradients, and step along that instead of the
raw gradient.**

$$v^t = \beta v^{t-1} + \nabla L(\mathbf{w}^{t-1}) \qquad \mathbf{w}^t = \mathbf{w}^{t-1} - \eta\, v^t$$

| Symbol | Read it as | What it means |
|---|---|---|
| $v^t$ | "velocity" | The accumulated direction. Initialised to $\mathbf{0}$. |
| $\beta$ | "beta" | Decay factor, typically **0.9**. How much of the past to keep. |

The physical analogy is exact: a ball rolling downhill has inertia. Consistent gradient components
(down the valley floor) accumulate and grow; oscillating components (across the valley) cancel
against their own past and shrink.

**How much speedup?** If the gradient were constant $g$, the velocity converges to the geometric
series $v_\infty = g(1 + \beta + \beta^2 + \cdots) = \frac{g}{1-\beta}$. At $\beta = 0.9$ that is
$\mathbf{10g}$ — **a 10× effective step size in consistent directions**, and no speedup at all in
directions that keep flipping sign. That is exactly the selectivity you wanted.

**Nesterov momentum** is a refinement: evaluate the gradient at $\mathbf{w}^{t-1} - \eta\beta v^{t-1}$
— i.e. *where momentum is about to take you* — rather than where you currently are. It is a
"look-ahead" correction that lets the ball start braking before it overshoots.

### Adagrad — fixing "one rate for all"

The formula says: **divide each parameter's step by the square root of the total squared gradient it
has accumulated so far. Parameters that have been updated a lot get small steps; rarely-updated ones
keep large steps.**

$$G^t_j = G^{t-1}_j + \left(\nabla L_j\right)^2 \qquad w^t_j = w^{t-1}_j - \frac{\eta}{\sqrt{G^t_j} + \epsilon}\,\nabla L_j$$

| Symbol | Read it as | What it means |
|---|---|---|
| $G^t_j$ | "G sub j" | Running **sum** of squared gradients for parameter $j$ only. |
| $\epsilon$ | "epsilon" | A tiny constant ($10^{-8}$) preventing division by zero. |

This was a genuine breakthrough for sparse data — NLP with bag-of-words features, recommender
systems with rare item IDs.

**Its fatal flaw:** $G$ is a *sum* and gradients are squared, so $G$ **only ever increases**. The
effective learning rate $\eta/\sqrt{G}$ therefore decays monotonically toward zero, and on a long
training run learning stops entirely — even if you haven't converged.

### RMSProp — fixing Adagrad's decay

One change: make $G$ an **exponentially weighted moving average** instead of a sum, so old gradients
are forgotten.

$$G^t_j = \rho\,G^{t-1}_j + (1-\rho)\left(\nabla L_j\right)^2, \qquad \rho \approx 0.9$$

Now $G$ tracks the *recent* gradient magnitude and can go down as well as up. Learning never dies.

### Adam — combining both

Adam = momentum (a running average of the gradient) + RMSProp (a running average of the squared
gradient). "Adam" is short for **Adaptive Moment Estimation**, and the two "moments" are exactly
those two averages.

$$m^t = \beta_1 m^{t-1} + (1-\beta_1)\nabla L \qquad\text{(1st moment — the mean, i.e. momentum)}$$
$$v^t = \beta_2 v^{t-1} + (1-\beta_2)(\nabla L)^2 \qquad\text{(2nd moment — uncentred variance, i.e. RMSProp)}$$
$$\hat m^t = \frac{m^t}{1-\beta_1^t}, \qquad \hat v^t = \frac{v^t}{1-\beta_2^t} \qquad\text{(bias correction)}$$
$$\mathbf{w}^t = \mathbf{w}^{t-1} - \eta\,\frac{\hat m^t}{\sqrt{\hat v^t}+\epsilon}$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.

> 💡 **Why bias correction exists.** $m$ and $v$ start at zero, so early on they are biased toward
> zero — at $t=1$ with $\beta_1=0.9$, $m^1 = 0.1\nabla L$, ten times too small. Dividing by
> $(1-\beta_1^t)$ exactly undoes this: at $t=1$ that divisor is $0.1$, restoring the full magnitude.
> As $t$ grows, $\beta_1^t \to 0$ and the correction fades to 1. **Without it, Adam takes tiny,
> useless steps for the first several hundred iterations.**

🧪 **Three Adam steps by hand.** Say $\nabla L = 0.1$ at every step (constant gradient),
$\eta = 0.001$, defaults as above.

| $t$ | $m^t$ | $v^t$ | $\hat m^t$ | $\hat v^t$ | step $= \eta\hat m/(\sqrt{\hat v}+\epsilon)$ |
|---|---|---|---|---|---|
| 1 | $0.1\cdot0.1 = 0.01$ | $0.001\cdot0.01 = 10^{-5}$ | $0.01/0.1 = 0.1$ | $10^{-5}/0.001 = 0.01$ | $0.001 \cdot 0.1/0.1 = \mathbf{0.001}$ |
| 2 | $0.9(0.01)+0.01 = 0.019$ | $0.999(10^{-5})+10^{-5} = 1.999\times10^{-5}$ | $0.019/0.19 = 0.1$ | $1.999\!\times\!10^{-5}/0.001999 = 0.01$ | $\mathbf{0.001}$ |
| 3 | $0.9(0.019)+0.01 = 0.0271$ | $\approx 2.997\times 10^{-5}$ | $0.0271/0.271 = 0.1$ | $\approx 0.01$ | $\mathbf{0.001}$ |

**The step is exactly $\eta$, every time.** That is the defining property of Adam: for a
constant gradient of *any* magnitude, the ratio $\hat m/\sqrt{\hat v}$ is 1, so the step is $\eta$.
Adam is nearly **scale-invariant** — multiply your loss by 1000 and Adam takes the same steps, while
plain SGD would take 1000× larger ones. This is why Adam "just works" without tuning $\eta$ per
problem, and it is the whole reason for its dominance.

### AdamW — fixing Adam's weight decay

> **The bug.** L2 regularisation adds $\lambda\|w\|^2$ to the loss, which adds $2\lambda w$ to the
> gradient. In Adam, that added term then goes through the $/\sqrt{\hat v}$ division — so parameters
> with large historical gradients get *less* weight decay than parameters with small ones. That is
> the opposite of the intent: weight decay is meant to be a uniform pull toward zero.
>
> **The fix (AdamW).** *Decouple* it — don't put decay in the gradient at all; subtract it directly
> from the weights after the Adam step:
> $$\mathbf{w}^t = \mathbf{w}^{t-1} - \eta\frac{\hat m^t}{\sqrt{\hat v^t}+\epsilon} - \eta\lambda \mathbf{w}^{t-1}$$
>
> **AdamW is the default for training transformers.** If you use `torch.optim.Adam` with a
> `weight_decay` argument, you are getting the buggy coupled version; `torch.optim.AdamW` is what you
> want. This is a very common interview question and an even more common real-world bug.

### Muon and SOAP

The slide names these two without comment. Be honest about what they are:

- **Muon** — a recent optimiser that applies an orthogonalisation step (via a Newton–Schulz
  iteration) to the momentum matrix before updating weight *matrices*, treating them as matrices
  rather than flattened vectors. It has drawn attention for strong wall-clock results on
  transformer pre-training.
- **SOAP** — a recent method connecting Shampoo-style second-order preconditioning with Adam, running
  Adam in the eigenbasis of a preconditioner.

> ✅ **Confirmed via primary source.** **Muon** ("MomentUm Orthogonalized by Newton-Schulz") was
> released by Keller Jordan in October 2024 (initially as a blog post / open-source implementation
> rather than a traditional peer-reviewed paper). **SOAP** is Vyas, Morwani, Zhao, Shapira, Brandfonbrener,
> Janson & Kakade, *"SOAP: Improving and Stabilizing Shampoo using Adam"*, arXiv:2409.11321, published
> at ICLR 2025. What you *can* say safely, and what the slide's inclusion of them is signalling:
> **optimiser design is not a solved, closed subject.** Adam has been the default for a decade and is
> now being seriously challenged.
>
> 🎯 **stretch — nice to know, not expected.** An intern is not expected to know Muon or SOAP.
> Knowing that "the field is actively revisiting Adam, and matrix-aware/second-order methods are
> where the action is" is a *better* answer than a half-remembered equation.

### Cosine annealing — the schedule

The learning rate need not be constant. **Cosine annealing** decays it along a cosine curve from
$\eta_{\max}$ to $\eta_{\min}$ over $T$ steps:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$$

Check the endpoints: at $t=0$, $\cos 0 = 1$ so $\eta = \eta_{\max}$ ✓. At $t=T$, $\cos\pi = -1$ so
$\eta = \eta_{\min}$ ✓.

**Why a cosine rather than a straight line?** The cosine is flat at both ends and steepest in the
middle. So you get a long high-LR exploration phase early, a rapid transition, and a long low-LR
refinement phase at the end — which is exactly the shape you want, and empirically beats linear decay
and step decay on most large-scale training runs.

In practice it is paired with **linear warmup**: ramp $\eta$ from ~0 up to $\eta_{\max}$ over the
first few hundred to few thousand steps, *then* cosine down. The warmup exists because early
gradients are large and the Adam second-moment estimate $\hat v$ is still unreliable; a big step
taken on a bad estimate can destabilise training permanently.

### The family tree

```
                     Gradient Descent
                            │
              ┌─────────────┴──────────────┐
       (how much data?)              (how to step?)
              │                             │
     Batch → SGD → Mini-batch       ┌───────┴────────┐
                                    │                │
                              add MEMORY       adapt PER-PARAMETER
                                    │                │
                               Momentum          Adagrad
                                    │           (decays to 0)
                              Nesterov               │
                                    │             RMSProp
                                    │           (EMA instead
                                    │            of sum)
                                    └───────┬────────┘
                                            │
                                          ADAM
                                     (1st + 2nd moment
                                      + bias correction)
                                            │
                                          AdamW
                                   (decouple weight decay)
                                            │
                                    Muon / SOAP  ⚠️ new, verify
                                    (matrix-aware /
                                     second-order)

   Orthogonal to all of the above: LEARNING RATE SCHEDULES
   warmup → cosine annealing → (optional) restarts
```

> 🎯 **What to actually use, asked and answered.** Default to **AdamW** with cosine annealing and a
> short warmup. Use **SGD with momentum** if you are training a convolutional vision model where the
> literature says it generalises slightly better and you can afford to tune. Never use plain SGD
> without momentum on anything real. Never use Adagrad for a long run.

---

# Part 3 — Evaluation Metrics

## 14. Loss is not the metric

Slide [raw `slide_056`, 22:14] — recovered from the raw capture — makes the framing explicit:

> How do we measure whether a model is "good enough" for deployment?
>
> The loss function (training objective) **≠** the evaluation metric (what we actually care about).
>
> **Classification:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
> **Regression:** RMSE, MAE, R²

This distinction deserves to be hammered:

| | Loss | Metric |
|---|---|---|
| Who consumes it | the optimiser | you, and your product manager |
| Must be differentiable | **yes** | no |
| Must be interpretable | no | **yes** |
| Computed on | training batches | validation / test sets |
| Example | cross-entropy = 0.34 | "we catch 82% of fraud at a 3% false-alarm rate" |

**Nobody ships a cross-entropy of 0.34.** They ship a recall at a fixed precision. The loss is
scaffolding; the metric is the product.

> 📚 **Background the slide assumed — train / validation / test, and data leakage.** You need three
> disjoint splits:
> - **Train** — the optimiser sees it. Loss computed here is not evidence of anything.
> - **Validation** — you see it, repeatedly, while choosing hyperparameters, architectures, and when
>   to stop. Because you make decisions using it, **you slowly overfit to it too.**
> - **Test** — looked at **once**, at the end. Every additional look degrades it into a validation set.
>
> **Data leakage** is any path by which test information reaches the model. The classic instances:
> normalising using statistics computed over the *whole* dataset before splitting; imputing missing
> values before splitting; having duplicate rows spanning train and test; and, in time series,
> shuffling randomly so the model trains on the future and predicts the past. Symptom: a validation
> score that is excellent and utterly fails to reproduce in production. Fix: **split first,
> preprocess after**, and wrap every transform in a pipeline object fitted on train only.

---

## 15. The confusion matrix

Slide [raw `slide_060`, 25:43]:

> **Confusion Matrix for binary classification:**
> - True Positives (TP): Correctly predicted positive
> - True Negatives (TN): Correctly predicted negative
> - False Positives (FP): Incorrectly predicted positive (**Type I error**)
> - False Negatives (FN): Incorrectly predicted negative (**Type II error**)
>
> **Derived metrics:**
> - Accuracy = (TP + TN) / (TP + TN + FP + FN)
> - Precision = TP / (TP + FP) — "Of predicted positives, how many correct?"
> - Recall = TP / (TP + FN) — "Of actual positives, how many found?"
> - F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

### The naming rule that makes this stop being confusing

Every one of the four names is **two words**, and each word answers a separate question:

| Word | Question it answers |
|---|---|
| **first** word (True / False) | *Was the model right?* |
| **second** word (Positive / Negative) | *What did the model **say**?* |

So a **False Negative** = the model **said negative**, and it was **wrong**. Therefore the truth was
positive. You never need to memorise a grid again — you can derive any cell in two seconds.

- **Type I error = False Positive** = a false alarm. You rejected a true null hypothesis.
- **Type II error = False Negative** = a miss. You failed to reject a false null hypothesis.

(Mnemonic if you need one: Type **I** has one letter in the roman numeral and FP is the "cry wolf"
error; Type **II** has two, and FN is the "miss it" error.)

> ⚠️ **The orientation trap.** The diagram on this slide puts **Predicted** on the rows and **Actual**
> on the columns. `sklearn.metrics.confusion_matrix` does the **opposite** — rows are actual, columns
> are predicted, and it returns `[[TN, FP], [FN, TP]]`. Reading an sklearn matrix with the slide's
> layout in your head silently swaps FP and FN, which swaps precision and recall. **Always check the
> axis labels.** This has burned a lot of people in real post-mortems.

### 🧪 Worked example — the accuracy paradox

A screening model for a disease affecting **1%** of a 1,000-patient population. So 10 patients are
genuinely positive, 990 negative.

**Model A: predict "negative" for everyone.**

| | Actual + | Actual − |
|---|---|---|
| **Predicted +** | TP = 0 | FP = 0 |
| **Predicted −** | FN = 10 | TN = 990 |

$$\text{Accuracy} = \frac{0 + 990}{1000} = \mathbf{99.0\%}$$

**99% accurate, and it has never once identified a sick patient.** Precision is undefined (0/0),
recall is $0/10 = \mathbf{0}$. This model is worthless and its headline number is excellent. That is
the accuracy paradox, and it is why nobody reports accuracy on imbalanced problems.

**Model B: a real model.**

| | Actual + | Actual − |
|---|---|---|
| **Predicted +** | TP = 8 | FP = 50 |
| **Predicted −** | FN = 2 | TN = 940 |

$$\text{Accuracy} = \frac{8 + 940}{1000} = \frac{948}{1000} = \mathbf{94.8\%}$$
$$\text{Precision} = \frac{8}{8+50} = \frac{8}{58} = \mathbf{0.1379}$$
$$\text{Recall} = \frac{8}{8+2} = \frac{8}{10} = \mathbf{0.80}$$
$$F_1 = 2\cdot\frac{0.1379 \times 0.80}{0.1379 + 0.80} = 2\cdot\frac{0.11034}{0.93793} = \frac{0.22069}{0.93793} = \mathbf{0.2353}$$

**Model B has lower accuracy than Model A (94.8% vs 99.0%) and is enormously better.** It finds 8 of
10 sick patients. Its precision is poor — 86% of its alarms are false — but for a *screening* test
followed by a cheap confirmatory test, that is exactly the right trade.

> 💡 **The single sentence to remember:** accuracy asks "how often is the model right?", which is the
> wrong question whenever one class is rare or the two error types have different costs. Both are
> true in almost every problem worth solving.

### The other derived metrics worth knowing

| Metric | Formula | Reads as |
|---|---|---|
| **Specificity** (TNR) | $\frac{TN}{TN+FP}$ | Of actual negatives, how many correctly cleared? |
| **TPR** (= Recall, = Sensitivity) | $\frac{TP}{TP+FN}$ | Of actual positives, how many caught? |
| **FPR** | $\frac{FP}{FP+TN} = 1 - \text{Specificity}$ | Of actual negatives, how many falsely flagged? |
| **NPV** | $\frac{TN}{TN+FN}$ | Of predicted negatives, how many correct? |
| **Balanced accuracy** | $\frac{TPR + TNR}{2}$ | Accuracy that is immune to class imbalance. |
| **MCC** | $\frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | A correlation coefficient between prediction and truth; $-1$ to $+1$; the most honest single number on imbalanced data. |

For Model B, balanced accuracy $= \frac{0.80 + 940/990}{2} = \frac{0.80 + 0.9495}{2} = \mathbf{0.8747}$
— far more informative than 94.8%.

---

## 16. The precision–recall trade-off

Slide [raw `slide_062`, 26:29]:

> **Adjusting the threshold changes precision and recall inversely:**
> - Higher threshold → fewer positives → higher precision, lower recall
> - Lower threshold → more positives → lower precision, higher recall
>
> **Which to prioritize?**
> - **Precision:** When false positives are costly (spam detection)
> - **Recall:** When false negatives are costly (cancer screening, fraud)
>
> **F1 Score:** Harmonic mean — balances precision and recall.
> Useful when classes are imbalanced and accuracy is misleading.

### Why the trade-off is mechanical, not empirical

A classifier does not output a class. It outputs a **score** — a probability, or a raw margin — and
you compare it to a **threshold** $\tau$ to get a class. That threshold is a free parameter you
choose *after* training, at zero cost, without retraining anything.

- Raise $\tau$ → you flag fewer things → the ones you do flag are the ones you were most confident
  about → **precision up**. But you now miss things you'd previously have caught → **recall down**.
- Lower $\tau$ → the mirror image.

At $\tau = 0$ you flag everything: recall $= 1$, precision $=$ the base rate. At $\tau = 1$ you flag
nothing: recall $= 0$, precision undefined. **The whole curve between those two points belongs to a
single trained model.** Which is why:

> 💡 **Threshold moving is the first thing to try on an imbalanced problem, and it is free.** Before
> you reach for class weights, resampling, SMOTE, or focal loss, sweep the threshold on your
> validation set and pick the operating point your business actually wants. It costs one line of code
> and no retraining. People skip it constantly and go straight to SMOTE.

### 🧪 Worked example — a threshold sweep

Ten validation examples, sorted by the model's score, with true labels:

| Score | 0.95 | 0.90 | 0.80 | 0.75 | 0.70 | 0.60 | 0.50 | 0.40 | 0.30 | 0.20 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Label** | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 |

Five positives, five negatives. Now sweep $\tau$:

| $\tau$ | Predicted + | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 0.95 | 1 | 1 | 0 | 4 | $1/1 = 1.000$ | $1/5 = 0.200$ | $0.333$ |
| 0.75 | 4 | 3 | 1 | 2 | $3/4 = 0.750$ | $3/5 = 0.600$ | $0.667$ |
| 0.70 | 5 | 4 | 1 | 1 | $4/5 = 0.800$ | $4/5 = 0.800$ | $\mathbf{0.800}$ |
| 0.50 | 7 | 5 | 2 | 0 | $5/7 = 0.714$ | $5/5 = 1.000$ | $0.833$ |
| 0.20 | 10 | 5 | 5 | 0 | $5/10 = 0.500$ | $5/5 = 1.000$ | $0.667$ |

Verifying one F1 by hand, at $\tau = 0.70$:
$F_1 = 2\cdot\frac{0.8 \times 0.8}{0.8+0.8} = 2 \cdot \frac{0.64}{1.6} = \mathbf{0.800}$ ✓

**One model. Five completely different products.** At $\tau = 0.95$ you have a high-confidence
auto-blocking system; at $\tau = 0.50$ you have a review queue that misses nothing. Choosing $\tau$
is a business decision, not a modelling one.

### Why F1 uses the harmonic mean

The slide says "harmonic mean" without saying why it isn't the ordinary average. Here is why, and it
is a genuinely good interview answer.

$$F_1 = 2\cdot\frac{P \cdot R}{P + R} \qquad\text{vs}\qquad \text{arithmetic mean} = \frac{P+R}{2}$$

Consider the degenerate classifier that flags **everything**: $P = 0.01$ (the base rate), $R = 1.0$.

- Arithmetic mean: $\frac{0.01 + 1.0}{2} = \mathbf{0.505}$ — looks like a coin flip. Respectable!
- Harmonic mean: $2\cdot\frac{0.01 \times 1.0}{1.01} = \frac{0.02}{1.01} = \mathbf{0.0198}$ — correctly
  identifies it as garbage.

**The harmonic mean is dominated by the smaller of the two values.** It cannot be gamed by maxing one
metric at the other's expense — which is precisely what you need, because both trivial classifiers
(flag everything / flag nothing) max exactly one.

**$F_\beta$** generalises it when you care about them unequally:

$$F_\beta = (1+\beta^2)\cdot\frac{P\cdot R}{\beta^2 P + R}$$

$\beta > 1$ weights **recall** more ($F_2$ is standard for fraud and medical screening); $\beta < 1$
weights **precision** more ($F_{0.5}$ for spam filtering, where a false positive means a lost
legitimate email).

> 📚 **Background — macro vs micro vs weighted averaging.** For $C > 2$ classes you compute
> precision/recall per class and must combine them:
> - **Macro** — unweighted mean across classes. Every class counts equally, so a rare class you get
>   badly wrong tanks the score. Use when rare classes matter.
> - **Micro** — pool all TP/FP/FN across classes, then compute once. Dominated by frequent classes.
>   For single-label multi-class, micro-F1 equals accuracy exactly.
> - **Weighted** — mean weighted by class support. A compromise; the most commonly reported and the
>   easiest to misread.
>
> Reporting "F1 = 0.86" without saying which average is meaningless. Say "macro-F1 = 0.86".

---

## 17. ROC-AUC and PR-AUC

Slide [raw `slide_065`, 28:36]:

> **ROC Curve (Receiver Operating Characteristic):**
> - Plots True Positive Rate vs. False Positive Rate at all thresholds
> - AUC = Area Under ROC Curve (1.0 = perfect, 0.5 = random)
> - Threshold-independent measure of discriminative ability
>
> **PR Curve (Precision-Recall):**
> - Plots Precision vs. Recall at all thresholds
> - PR-AUC preferred for highly imbalanced datasets
> - More informative when negatives dominate
>
> **When to use which?**
> - ROC-AUC: Balanced classes, care about both classes equally
> - PR-AUC: Imbalanced data, focus on positive class performance

### What "threshold-independent" buys you

§16 showed that a single model produces a whole family of (precision, recall) pairs. That makes model
*comparison* awkward: if model A beats model B at $\tau=0.5$ but loses at $\tau=0.8$, which is
better? AUC answers this by summarising performance **across every threshold at once**, giving you
one number that describes the model's *ranking ability* independently of where you eventually set
the cut.

### 🧪 Worked example — build an ROC curve and compute AUC two ways

Eight examples, four positive and four negative:

| Score | 0.95 | 0.90 | 0.80 | 0.70 | 0.60 | 0.50 | 0.40 | 0.20 |
|---|---|---|---|---|---|---|---|---|
| **Label** | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 |

**Method 1 — trace the curve.** Start at the origin and walk down the score list. Each **positive**
you pass steps you **up** by $1/P = 0.25$; each **negative** steps you **right** by $1/N = 0.25$.

| After | Label | TP | FP | FPR = FP/4 | TPR = TP/4 | Point |
|---|---|---|---|---|---|---|
| start | — | 0 | 0 | 0.00 | 0.00 | (0.00, 0.00) |
| 0.95 | 1 | 1 | 0 | 0.00 | 0.25 | (0.00, 0.25) |
| 0.90 | 1 | 2 | 0 | 0.00 | 0.50 | (0.00, 0.50) |
| 0.80 | 0 | 2 | 1 | 0.25 | 0.50 | (0.25, 0.50) |
| 0.70 | 1 | 3 | 1 | 0.25 | 0.75 | (0.25, 0.75) |
| 0.60 | 0 | 3 | 2 | 0.50 | 0.75 | (0.50, 0.75) |
| 0.50 | 1 | 4 | 2 | 0.50 | 1.00 | (0.50, 1.00) |
| 0.40 | 0 | 4 | 3 | 0.75 | 1.00 | (0.75, 1.00) |
| 0.20 | 0 | 4 | 4 | 1.00 | 1.00 | (1.00, 1.00) |

Area, by summing rectangles over each rightward move (each of width 0.25, at the current TPR):

$$\text{AUC} = 0.25(0.50) + 0.25(0.75) + 0.25(1.00) + 0.25(1.00) = 0.125 + 0.1875 + 0.25 + 0.25 = \mathbf{0.8125}$$

**Method 2 — the pairwise ranking interpretation.** AUC equals the probability that a randomly
chosen positive scores higher than a randomly chosen negative. Count winning pairs out of
$P \times N = 4 \times 4 = 16$:

| Positive | Beats which negatives (0.80, 0.60, 0.40, 0.20)? | Count |
|---|---|---|
| 0.95 | all four | 4 |
| 0.90 | all four | 4 |
| 0.70 | 0.60, 0.40, 0.20 | 3 |
| 0.50 | 0.40, 0.20 | 2 |
| | **Total** | **13** |

$$\text{AUC} = \frac{13}{16} = \mathbf{0.8125}$$

**Both methods give 0.8125.** They are the same quantity; the geometric one is how you plot it, the
combinatorial one is what it *means*.

> 💡 **The one-sentence definition of AUC worth memorising:** *the probability that the model scores
> a random positive above a random negative.* This immediately explains why AUC = 0.5 is random
> (a coin flip on every pair), why AUC = 1.0 is perfect separation, and why AUC is invariant to any
> monotonic rescaling of your scores — it only ever looks at the **ordering**.

### Why ROC lies on imbalanced data, and PR doesn't

This is the most important practical point in the metrics section.

Fraud detection: 1,000,000 transactions, **1,000 fraudulent** (0.1%). Your model, at its chosen
threshold, catches 900 frauds and raises 10,000 false alarms.

$$\text{TPR} = \frac{900}{1000} = 0.90 \qquad \text{FPR} = \frac{10{,}000}{999{,}000} = \mathbf{0.010}$$

An ROC point of $(0.010,\ 0.90)$ is spectacular — hard against the top-left corner. AUC will look
superb. But:

$$\text{Precision} = \frac{900}{900 + 10{,}000} = \frac{900}{10{,}900} = \mathbf{0.0826}$$

**Fewer than 9% of the alerts your fraud team investigates are real.** They will drown.

**The reason ROC hides this:** FPR divides by $TN + FP$, and with 999,000 negatives that denominator
is enormous, so 10,000 false positives barely register. Precision divides by $TP + FP$, where the
10,000 false positives are compared against only 900 true ones — and the disaster is immediately
visible.

| | Denominator | Contains the huge negative class? |
|---|---|---|
| **FPR** (ROC's x-axis) | $FP + TN$ | **yes** → dilutes FP |
| **Precision** (PR's y-axis) | $FP + TP$ | **no** → exposes FP |

> ⚠️ **Rule:** whenever the positive class is under ~10% of your data, report **PR-AUC** (also called
> **average precision**), not ROC-AUC. Also note that the PR curve's random-guess baseline is not
> 0.5 — it is the **positive base rate**. A PR-AUC of 0.30 on a 0.1%-positive problem is a 300×
> improvement over random, and looks bad to anyone expecting 0.5 to be the floor. Always report the
> baseline alongside it.

```interactive
type: graph
title: Drag the threshold, watch three views move together
concept: One model, one score distribution, and how threshold choice moves the confusion matrix, the ROC point and the PR point simultaneously
control: A single threshold slider over a pair of overlapping score histograms (positives and negatives), plus a class-imbalance slider from 50% to 0.1% positive.
observe: Three linked panels update live — a 2×2 confusion matrix with counts, a dot moving along the ROC curve, and a dot moving along the PR curve, with precision/recall/F1 read out numerically.
insight: Dragging the imbalance slider toward 0.1% barely moves the ROC dot while the PR dot collapses. That divergence, seen live, is the entire argument for PR-AUC on imbalanced data.
fallback: The fraud worked example in §17 — FPR = 0.010 (looks excellent) while precision = 0.083 (a disaster), from the same confusion matrix.
```

---

## 18. Regression metrics

Slide [`slide_015`, 29:07]:

> **Error Metrics**
> - **RMSE (Root Mean Squared Error):** `sqrt(1/n * sum(yi - y_hat_i)^2)`
>   - Same units as target variable
>   - Penalizes large errors more heavily
> - **MAE (Mean Absolute Error):** `(1/n) * sum(|yi - y_hat_i|)`
>   - Robust to outliers
>   - Easier to interpret
>
> **Goodness of Fit**
> - **R-squared (Coefficient of Determination):** `1 - SS_res / SS_tot`
>   - Proportion of variance explained by model
>   - 1.0 = perfect, 0 = baseline (predict mean)
>   - Can be negative for very poor models
>
> **Choosing metrics:** Outlier-sensitive tasks → RMSE · Outlier-robust tasks → MAE ·
> Relative model fit → R-squared

### RMSE vs MSE — why the square root matters

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_i (y_i - \hat y_i)^2}$$

MSE and RMSE are minimised at exactly the same parameters (square root is monotonic), so **as a
training loss they are interchangeable.** As a *metric* they are not, for one reason the slide
states: **units**. If $y$ is in dollars, MSE is in dollars-squared, which is meaningless to a
stakeholder. RMSE is in dollars. "Our average error is about \$47" is a sentence you can say in a
review; "our MSE is 2209" is not.

### R² explained properly

The formula says: **measure how much error your model leaves, measure how much error the dumbest
reasonable baseline leaves, take the ratio, and subtract it from one — so the result is the fraction
of the baseline's error that you managed to remove.**

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$$

| Symbol | Read it as | What it means |
|---|---|---|
| $SS_{res}$ | "sum of squares, residual" | How much error **your model** leaves. |
| $SS_{tot}$ | "sum of squares, total" | How much error the **dumbest reasonable baseline** leaves — always predicting the mean $\bar y$. |
| $\bar y$ | "y bar" | The mean of the true targets. |

So $R^2$ literally reads: **"what fraction of the baseline's error did I eliminate?"**

- $R^2 = 1$ → $SS_{res} = 0$ → perfect.
- $R^2 = 0$ → $SS_{res} = SS_{tot}$ → **exactly as good as always guessing the mean.**
- $R^2 < 0$ → **worse than guessing the mean.** Entirely possible, and the slide correctly says so.
  It happens routinely on a test set when a model has overfit.

### 🧪 Worked example

$y = [10,\ 20,\ 30,\ 40]$, $\hat y = [12,\ 18,\ 33,\ 39]$. Residuals: $[-2,\ 2,\ -3,\ 1]$.

$$\text{MAE} = \frac{2+2+3+1}{4} = \frac{8}{4} = \mathbf{2.0}$$
$$\text{RMSE} = \sqrt{\frac{4+4+9+1}{4}} = \sqrt{\frac{18}{4}} = \sqrt{4.5} = \mathbf{2.1213}$$

For $R^2$: $\bar y = \frac{10+20+30+40}{4} = 25$.

$$SS_{tot} = (10-25)^2 + (20-25)^2 + (30-25)^2 + (40-25)^2 = 225 + 25 + 25 + 225 = 500$$
$$SS_{res} = 4 + 4 + 9 + 1 = 18$$
$$R^2 = 1 - \frac{18}{500} = 1 - 0.036 = \mathbf{0.964}$$

**Now the negative case.** Same $y$, but a bad model that always predicts 30: $\hat y = [30,30,30,30]$.

$$SS_{res} = (10-30)^2 + (20-30)^2 + (30-30)^2 + (40-30)^2 = 400 + 100 + 0 + 100 = 600$$
$$R^2 = 1 - \frac{600}{500} = 1 - 1.2 = \mathbf{-0.2}$$

Negative, as promised. Predicting the constant 30 is worse than predicting the constant 25.

> ⚠️ **Note RMSE (2.1213) > MAE (2.0) here, and that is always true** (by the QM–AM inequality), with
> equality only when every error has the same magnitude. So the **gap between them is a free
> diagnostic**: RMSE ≈ MAE means your errors are uniform; RMSE ≫ MAE means a few examples are
> contributing most of your error, and you should go look at them.

### Three more regression metrics worth having

| Metric | Formula | When |
|---|---|---|
| **MAPE** | $\frac{100}{n}\sum_i \left\lvert\frac{y_i - \hat y_i}{y_i}\right\rvert$ | When relative error is what matters. ⚠️ Explodes when any $y_i \approx 0$, and asymmetrically punishes over-prediction. |
| **Adjusted R²** | $1 - (1-R^2)\frac{n-1}{n-d-1}$ | Comparing models with different numbers of features. Plain $R^2$ **never decreases** when you add a feature, even a random one; the adjustment penalises $d$. |
| **RMSLE** | $\sqrt{\frac{1}{n}\sum_i(\log(1+y_i) - \log(1+\hat y_i))^2}$ | Targets spanning orders of magnitude (demand, prices). Penalises under-prediction more than over-prediction, which often matches inventory economics. |

> 📚 **Background the slide assumed — is a 2-point difference real?** If model A scores 0.86 and
> model B scores 0.84 on a 100-example test set, **that is almost certainly noise.** The standard
> error of an accuracy estimate is $\sqrt{p(1-p)/n} = \sqrt{0.85 \times 0.15/100} \approx 0.036$, so
> a 95% interval is roughly $\pm 0.07$. The two models are indistinguishable.
>
> Fixes, in increasing order of rigour: (1) report **mean ± std across cross-validation folds**, not
> a single number; (2) **bootstrap** the test set — resample with replacement 1,000 times, recompute
> the metric, take the 2.5th and 97.5th percentiles; (3) for comparing two models, use a **paired**
> test on the *same* examples (McNemar's test for classification), which removes the variance from
> example difficulty and is far more sensitive than comparing two independent intervals.
>
> Nothing damages credibility faster than shipping a model because it won by one point on 200 test
> examples.

---

# Part 4 — Naive Bayes

## 19. Bayes' theorem, the naive assumption, and the decision rule

Slide [`slide_017`, 30:04] gives three formulas and almost no words. Every one deserves unpacking.

### Bayes' theorem applied to classification

The formula says: **the probability that this example belongs to class $k$, given what I observed,
is proportional to how likely class $k$ was to begin with, times how likely class $k$ was to produce
what I observed.**

$$P(C_k \mid \mathbf{x}) = \frac{P(\mathbf{x}\mid C_k)\,P(C_k)}{P(\mathbf{x})} \ \propto\ P(\mathbf{x}\mid C_k)\,P(C_k)$$

| Symbol | Read it as | Name | What it means |
|---|---|---|---|
| $P(C_k \mid \mathbf{x})$ | "P of C_k given x" | **posterior** | What we want: probability of class $k$ after seeing the evidence. |
| $P(\mathbf{x}\mid C_k)$ | "P of x given C_k" | **likelihood** | If this really were class $k$, how likely is this exact input? |
| $P(C_k)$ | "P of C_k" | **prior** | How common class $k$ is, before looking at anything. |
| $P(\mathbf{x})$ | "P of x" | **evidence** | How likely this input is under any class. |
| $\propto$ | "is proportional to" | — | Equal up to a constant factor. |

**Deriving it takes two lines.** The definition of conditional probability gives, two ways:

$$P(C_k \cap \mathbf{x}) = P(C_k\mid\mathbf{x})P(\mathbf{x}) \qquad\text{and}\qquad P(C_k \cap \mathbf{x}) = P(\mathbf{x}\mid C_k)P(C_k)$$

Set them equal and divide by $P(\mathbf{x})$. ∎

**Why we can drop $P(\mathbf{x})$.** It does not depend on $k$ — it is the same number for every
class we're comparing. Since we only want the $\arg\max$ over $k$, a common positive factor cannot
change which class wins. That's what the $\propto$ is doing, and it saves us from computing the
hardest term in the formula.

> ⚠️ You may only drop it because you want the **argmax**. If you want a calibrated probability, you
> must restore it: $P(\mathbf{x}) = \sum_k P(\mathbf{x}\mid C_k)P(C_k)$, i.e. normalise the scores to
> sum to 1. The worked example below does exactly this.

### The "naive" assumption

The formula says: **assume that, once you know the class, the features are independent of each other
— so the probability of the whole feature vector is just the product of each feature's probability.**

$$P(\mathbf{x}\mid C_k) = \prod_{j=1}^{d} P(x_j \mid C_k)$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\prod_{j=1}^d$ | "product from j equals 1 to d" | Multiply the terms together, once per feature. |
| $x_j$ | "x sub j" | The value of feature $j$. |

**Why this is called "naive":** it is usually false. In an email, "free" and "money" co-occur far
more than independently. The assumption says knowing one tells you nothing about the other, given
that the email is spam. That's simply wrong.

### 📚 Why make a false assumption? The counting argument

The slide never justifies it, and the justification is the entire point of the method.

Suppose you have $d$ binary features and you want $P(\mathbf{x}\mid C_k)$ **without** the assumption.
You'd need one probability for every possible feature combination: $2^d - 1$ free parameters, per
class.

| $d$ | Parameters without the assumption | Parameters with it |
|---|---|---|
| 10 | $2^{10} - 1 = 1{,}023$ | $10$ |
| 20 | $1{,}048{,}575$ | $20$ |
| 100 | $\approx 1.3\times10^{30}$ | $100$ |
| 10,000 (a vocabulary) | astronomically infeasible | $10{,}000$ |

**Exponential becomes linear.** With 10,000 vocabulary words you would need more training emails than
there are atoms in the observable universe to estimate the joint distribution; with the naive
assumption you need enough to estimate 10,000 individual word frequencies, which is a few thousand
emails. That is the trade: **accept a wrong model in exchange for one you can actually estimate.**

> 💡 **Why a false assumption still classifies well.** Naive Bayes' probability *estimates* are badly
> wrong — usually pushed toward 0 or 1, because correlated features double-count evidence. But
> classification only needs the **argmax**, and the argmax is far more robust than the values. If the
> true posterior is $(0.6, 0.4)$ and NB estimates $(0.99, 0.01)$, the decision is identical. This is
> the classic result: NB is a poor probability estimator and a surprisingly good classifier, and it's
> exactly what the slide means by "poor probability calibration".

### The decision rule

The formula says: **for each class, multiply its prior by every feature's likelihood under that
class; then pick whichever class produced the biggest product.**

$$\hat y = \arg\max_k\ P(C_k)\prod_{j=1}^{d}P(x_j\mid C_k)$$

### 📚 Two things you must do that the slide omits

**1. Laplace (add-one) smoothing.** If a word never appeared in any spam email in training, then
$P(\text{word}\mid \text{spam}) = 0$, and a single zero **annihilates the entire product** — the
class gets probability zero no matter how much other evidence supports it. One unseen word vetoes
everything.

The fix is to pretend you saw everything once more than you did:

$$P(x_j \mid C_k) = \frac{\text{count}(x_j, C_k) + \alpha}{\text{count}(C_k) + \alpha \lvert V \rvert}$$

| Symbol | What it means |
|---|---|
| $\alpha$ | Smoothing strength. $\alpha=1$ is Laplace; $\alpha<1$ is Lidstone. |
| $\lvert V\rvert$ | Vocabulary size. Needed in the denominator so the probabilities still sum to 1. |

**This is not optional.** `sklearn`'s Naive Bayes classes default to `alpha=1.0` precisely because
without it the method is broken on any real text corpus.

**2. Work in log space.** Multiplying 10,000 numbers each around $10^{-4}$ gives $10^{-40000}$, which
underflows a float64 (min ~$10^{-308}$) to exactly zero, for **every** class. Take logs, turning the
product into a sum:

$$\hat y = \arg\max_k\ \left[\log P(C_k) + \sum_{j=1}^{d}\log P(x_j\mid C_k)\right]$$

Logs are monotonic, so the argmax is unchanged. Every real implementation does this.

### 🧪 Worked example — spam classification, end to end

**Training data.** 10 emails: **4 spam**, **6 ham**. Vocabulary $V = \{\text{free},
\text{money}, \text{meeting}\}$, so $|V| = 3$.

Word occurrence counts:

| | free | money | meeting | **total words** |
|---|---|---|---|---|
| **Spam** | 6 | 4 | 0 | 10 |
| **Ham** | 1 | 2 | 7 | 10 |

**Step 1 — priors.**
$$P(\text{spam}) = \frac{4}{10} = 0.4 \qquad P(\text{ham}) = \frac{6}{10} = 0.6$$

**Step 2 — smoothed likelihoods** ($\alpha=1$, $|V|=3$, so every denominator is $10 + 3 = 13$):

| | $P(\cdot\mid\text{spam})$ | $P(\cdot\mid\text{ham})$ |
|---|---|---|
| free | $\frac{6+1}{13} = \frac{7}{13} = 0.5385$ | $\frac{1+1}{13} = \frac{2}{13} = 0.1538$ |
| money | $\frac{4+1}{13} = \frac{5}{13} = 0.3846$ | $\frac{2+1}{13} = \frac{3}{13} = 0.2308$ |
| meeting | $\frac{0+1}{13} = \frac{1}{13} = 0.0769$ | $\frac{7+1}{13} = \frac{8}{13} = 0.6154$ |

Note the smoothing at work: "meeting" never appeared in spam, and without $\alpha$ its likelihood
would be **0**, making *any* email containing "meeting" impossible to classify as spam regardless of
what else it said. With smoothing it is 0.0769 — small, but not fatal.

**Step 3 — classify a new email: "free money".**

$$\text{score}(\text{spam}) = 0.4 \times 0.5385 \times 0.3846 = 0.4 \times 0.20710 = \mathbf{0.082840}$$
$$\text{score}(\text{ham}) = 0.6 \times 0.1538 \times 0.2308 = 0.6 \times 0.035497 = \mathbf{0.021298}$$

**Step 4 — normalise to a real probability** (restoring the evidence term):

$$P(\text{spam}\mid \text{"free money"}) = \frac{0.082840}{0.082840 + 0.021298} = \frac{0.082840}{0.104138} = \mathbf{0.7955}$$

**Verdict: spam, with 79.6% posterior probability.**

**Step 5 — verify in log space**, as a real implementation would:

$$\log\text{score}(\text{spam}) = \log 0.4 + \log 0.5385 + \log 0.3846 = -0.9163 - 0.6190 - 0.9555 = \mathbf{-2.4908}$$
$$\log\text{score}(\text{ham}) = \log 0.6 + \log 0.1538 + \log 0.2308 = -0.5108 - 1.8718 - 1.4663 = \mathbf{-3.8489}$$

Spam wins ($-2.4908 > -3.8489$) ✓. Recovering the probability: the difference is $1.3581$, so the
odds ratio is $e^{1.3581} = 3.889$, giving $P(\text{spam}) = \frac{3.889}{1 + 3.889} = \mathbf{0.7955}$ ✓.

**Both routes agree exactly.**

---

## 20. Naive Bayes variants and trade-offs

Slide [`slide_018`, 32:20]:

> **Variants:**
> - **Gaussian NB:** Features are continuous, assumed normally distributed
> - **Multinomial NB:** Features are counts (e.g., word frequencies in text)
> - **Bernoulli NB:** Features are binary (e.g., word presence/absence)
>
> **Strengths:** Fast, works well with high-dimensional data, good baseline
> **Weaknesses:** Independence assumption rarely holds; poor probability calibration
> **Classic use case:** Spam classification, text categorization

The three variants differ in exactly one place: **what distribution you assume for
$P(x_j \mid C_k)$.** Everything else — the prior, the product, the argmax — is identical.

| Variant | $P(x_j \mid C_k)$ is | Fitted by | Use for |
|---|---|---|---|
| **Gaussian** | $\frac{1}{\sqrt{2\pi\sigma_{jk}^2}}\exp\!\left(-\frac{(x_j-\mu_{jk})^2}{2\sigma_{jk}^2}\right)$ | the mean $\mu_{jk}$ and variance $\sigma^2_{jk}$ of feature $j$ within class $k$ | continuous features (height, price, latency) |
| **Multinomial** | $\propto \theta_{jk}^{x_j}$, with $\theta_{jk}$ the probability of token $j$ in class $k$ | smoothed relative frequency (our worked example) | counts — word frequencies, tf-idf |
| **Bernoulli** | $\theta_{jk}^{x_j}(1-\theta_{jk})^{1-x_j}$ | fraction of class-$k$ documents containing the feature | binary presence/absence |

> 💡 **Multinomial vs Bernoulli on the same text task.** Multinomial counts how many times a word
> appears; Bernoulli only whether it appeared, **and it explicitly penalises words that are absent**
> (the $(1-\theta)^{1-x_j}$ factor). Bernoulli tends to win on short documents (tweets, subject
> lines) where absence is informative; multinomial wins on longer documents where repetition carries
> signal.

### 📚 Background — generative vs discriminative

Naive Bayes is a **generative** classifier: it models $P(\mathbf{x}\mid C_k)$ and $P(C_k)$ — the full
story of how the data was produced — and then uses Bayes' rule to *derive* $P(C_k\mid\mathbf{x})$.

Logistic regression, SVMs and neural networks are **discriminative**: they model the boundary
$P(C_k\mid\mathbf{x})$ directly and never learn what the data looks like.

| | Generative (NB) | Discriminative (LogReg, SVM) |
|---|---|---|
| Models | $P(\mathbf{x}, y)$ | $P(y\mid\mathbf{x})$ or just the boundary |
| Training | closed-form counting, **one pass** | iterative optimisation |
| Small data | **better** — the assumption acts as a strong prior | worse — overfits |
| Large data | plateaus (the wrong assumption caps it) | **better** — keeps improving |
| Can generate samples | **yes** | no |
| Handles missing features | naturally (drop the term) | needs imputation |

> 💡 There's a well-known result (Ng & Jordan) that a generative model reaches its — higher —
> asymptotic error much faster, so it wins in the low-data regime and loses in the high-data regime.
> ✅ **Confirmed and corrected.** The citation is Ng & Jordan, *"On Discriminative vs. Generative
> Classifiers: A Comparison of Logistic Regression and Naive Bayes"*, NIPS 2001 — that part was
> already right. The sample-complexity rates were not: this file's earlier draft stated "$O(d)$ for
> NB vs $O(d)$" for both, which is not the paper's result and not even internally distinct. The
> paper's actual finding is that **Naive Bayes' generalisation error approaches its asymptote in
> $O(\log d)$ examples, while logistic regression needs $\Omega(d)$** — a genuinely large gap, not
> the same rate. That asymmetry is exactly why NB can win decisively in the low-data regime and why
> the crossover point scales with the number of features $d$.

### 📚 Naive Bayes is secretly a linear classifier

Take logs of the decision rule for two classes and subtract:

$$\log\frac{P(C_1\mid\mathbf{x})}{P(C_0\mid\mathbf{x})} = \log\frac{P(C_1)}{P(C_0)} + \sum_{j=1}^d \log\frac{P(x_j\mid C_1)}{P(x_j\mid C_0)}$$

For Bernoulli features, each term in the sum is linear in $x_j$. So the decision boundary is
$b + \sum_j w_j x_j = 0$ — **a hyperplane, exactly like logistic regression.** The difference is
entirely in *how the weights are chosen*: NB sets them by counting each feature independently,
logistic regression fits them jointly to minimise loss. That is why logistic regression usually wins
with enough data — it can compensate for correlated features by shrinking their weights, and NB
structurally cannot.

> 🎯 A great answer to "why is Naive Bayes still used?" — it trains in a single pass with no
> hyperparameter search, needs no GPU, handles 100,000 features on a laptop, and gives you a
> published baseline number in ten minutes. **Its job is to be the number your fancy model has to
> beat.** If your transformer doesn't beat Naive Bayes, something is wrong with your pipeline, not
> your architecture.

---

# Part 5 — K-Nearest Neighbors

## 21. KNN: the algorithm

Slide [raw `slide_082`, 35:49] — recovered:

> **Idea:** Classify a point by majority vote of its K nearest neighbors.
> No explicit training phase — "lazy learner"
>
> **Distance Metrics:**
> - Euclidean: `sqrt(sum(xi - yi)^2)` — most common
> - Manhattan: `sum(|xi - yi|)` — for grid-like data
> - Minkowski: generalization of both
> - Cosine similarity: for text/high-dimensional sparse data
>
> **Choice of K:**
> - Small K (e.g., 1): Low bias, high variance (overfitting)
> - Large K: High bias, low variance (underfitting)
> - Typically odd K for binary classification (avoid ties)
> - Use cross-validation to find optimal K

The slide also displays the three distance formulas:

$$\text{Euclidean (L2):}\quad d(\mathbf{x},\mathbf{z}) = \sqrt{\sum_{j=1}^d (x_j - z_j)^2}$$
$$\text{Manhattan (L1):}\quad d(\mathbf{x},\mathbf{z}) = \sum_{j=1}^d \lvert x_j - z_j\rvert$$
$$\text{Minkowski (Lp):}\quad d(\mathbf{x},\mathbf{z}) = \left(\sum_{j=1}^d \lvert x_j - z_j\rvert^p\right)^{1/p}$$

Setting $p=2$ recovers Euclidean, $p=1$ recovers Manhattan. That is what "generalization of both"
means, and it is worth checking yourself: at $p=1$ the outer exponent is $1/1 = 1$, so the whole
thing collapses to the sum of absolute differences ✓.

### "No explicit training phase — lazy learner"

This is the strangest and most important property. KNN's "training" is: **store the data.** That's
it. All the work happens at prediction time, when it must compute the distance from the query to
every stored point.

| | Eager learner (LogReg, SVM, NN) | Lazy learner (KNN) |
|---|---|---|
| Train cost | high | $O(1)$ — just store |
| Predict cost | $O(d)$ — one dot product | $O(nd)$ — compare to everything |
| Model size | $d$ parameters | the **entire training set** |
| Adding new data | retrain | just append it |
| Decision boundary | fixed functional form | arbitrary, data-shaped |

> 💡 **The trade is exact:** KNN pays nothing up front and everything at inference. For a 100M-item
> catalogue that is fatal for online serving, which is why production nearest-neighbour systems
> (embedding retrieval, recommendations) all use approximate indexes — HNSW, IVF, ScaNN — that trade
> a small amount of recall for orders of magnitude in query speed. **The KNN you learn here is the
> exact version of what vector databases do approximately.**

### 🧪 Worked example — K changes the answer

Five training points in 2-D:

| Point | Coordinates | Class |
|---|---|---|
| A | (1, 1) | 0 |
| B | (2, 1) | 0 |
| C | (4, 5) | 1 |
| D | (5, 4) | 1 |
| E | (4, 2) | 1 |

Query: $q = (3, 2)$. Euclidean distances:

| Point | Computation | Distance | Class |
|---|---|---|---|
| **E** | $\sqrt{(4-3)^2 + (2-2)^2} = \sqrt{1+0}$ | $\mathbf{1.000}$ | 1 |
| **B** | $\sqrt{(2-3)^2+(1-2)^2} = \sqrt{1+1}$ | $\mathbf{1.414}$ | 0 |
| **A** | $\sqrt{(1-3)^2+(1-2)^2} = \sqrt{4+1}$ | $\mathbf{2.236}$ | 0 |
| **D** | $\sqrt{(5-3)^2+(4-2)^2} = \sqrt{4+4}$ | $\mathbf{2.828}$ | 1 |
| **C** | $\sqrt{(4-3)^2+(5-2)^2} = \sqrt{1+9}$ | $\mathbf{3.162}$ | 1 |

Now vary K:

| K | Neighbours (nearest first) | Votes | **Prediction** |
|---|---|---|---|
| 1 | E | 1×class 1 | **class 1** |
| 3 | E, B, A | 1×class 1, 2×class 0 | **class 0** |
| 5 | E, B, A, D, C | 3×class 1, 2×class 0 | **class 1** |

**Same query, same data, three different answers.** K is not a minor tuning knob — it is the model.

Check the Manhattan distances give the same $K=3$ answer: E $= 1+0 = 1$; B $= 1+1 = 2$; A $= 2+1 = 3$;
D $=2+2=4$; C $=1+3=4$. Nearest three are still E, B, A → **class 0** ✓. The metric didn't change the
outcome here, but it can and does.

### Choice of K — the bias–variance argument

> 📚 **Background the slide assumed — bias and variance.**
> - **Bias** — error from the model being too simple to represent the truth. It's wrong the *same
>   way* every time.
> - **Variance** — error from the model being too sensitive to the particular training sample. Retrain
>   on a different sample and you get a very different model.
> - Total expected error $=$ bias$^2$ $+$ variance $+$ irreducible noise. You are always trading one
>   against the other.

| K | Boundary shape | Bias | Variance | Failure mode |
|---|---|---|---|---|
| $K=1$ | maximally jagged; every training point owns a cell | **low** | **high** | one mislabelled point creates a wrong island → overfits |
| $K$ moderate | smooth but responsive | balanced | balanced | — |
| $K=n$ | a single flat prediction: the majority class everywhere | **high** | **low** | ignores the input entirely → underfits |

Note the extreme case: at $K=1$, **training accuracy is always exactly 100%**, because each training
point's nearest neighbour is itself. This makes training accuracy a completely useless signal for
choosing K — you *must* use cross-validation, which is exactly what the slide says.

**Why odd K:** with two classes and even K, you can get 2–2 and need a tie-break rule. Odd K makes
ties impossible. (For $C > 2$ classes odd K doesn't guarantee anything, and you break ties by
smallest total distance.)

> 💡 **Distance weighting** is the upgrade the slide doesn't mention: instead of one vote per
> neighbour, weight each vote by $1/d$ or $1/d^2$. Closer neighbours count more, which makes the
> method far less sensitive to K and removes most tie situations.
> (`KNeighborsClassifier(weights='distance')`.)

**Which distance to use:**

| Metric | Use when |
|---|---|
| **Euclidean (L2)** | Default. Continuous features on comparable scales. |
| **Manhattan (L1)** | Grid-like movement (city blocks), or when you want robustness to a single wildly-off coordinate — L1 doesn't square it. |
| **Minkowski (Lp)** | When you want to tune the interpolation between them. Rarely worth it. |
| **Cosine similarity** | **Text and embeddings.** Measures the *angle*, ignoring magnitude — so a 100-word and a 1000-word document about the same topic come out similar, which Euclidean would deny. |

---

## 22. KNN: practical considerations

Slide [raw `slide_084`, 36:30] — recovered:

> **Curse of Dimensionality:**
> - In high dimensions, all points become equidistant
> - Distance metrics lose discriminative power
> - Need exponentially more data as dimensions grow
>
> **Mitigation strategies:**
> - Feature selection / dimensionality reduction (PCA)
> - Feature scaling (normalization) is essential for KNN
>
> **Computational cost:**
> - Prediction is O(n*d) for n training points, d dimensions
> - KD-trees, Ball-trees for faster neighbor lookup
>
> **KNN for regression:** Average (or weighted average) of K neighbors' target values

### The curse of dimensionality, made concrete

"All points become equidistant" is a startling claim. Here is why it's true.

Take $n$ points drawn uniformly at random in the unit hypercube $[0,1]^d$ and look at the ratio

$$\frac{d_{\max} - d_{\min}}{d_{\min}}$$

— i.e. how much further the farthest point is than the nearest, in relative terms. As $d$ grows, this
ratio **goes to zero**. The nearest and farthest neighbours become indistinguishable.

The intuition: squared Euclidean distance is a **sum of $d$ independent per-dimension contributions**.
By concentration of measure, a sum of many independent terms concentrates tightly around its mean —
its standard deviation grows like $\sqrt{d}$ while its mean grows like $d$, so the *relative* spread
shrinks like $1/\sqrt d$. Every pair of points ends up at almost exactly the same distance.

**And "nearest neighbour" is a meaningless concept when every neighbour is equally near.** KNN doesn't
degrade gracefully in high dimensions; it stops being an algorithm.

Second framing — **"need exponentially more data".** To cover 10% of the range along each axis you
need a fraction $0.1^d$ of the volume:

| $d$ | Fraction of the space within 10% on every axis | Points needed for one neighbour in that box |
|---|---|---|
| 1 | $0.1$ | 10 |
| 2 | $0.01$ | 100 |
| 10 | $10^{-10}$ | 10 billion |
| 20 | $10^{-20}$ | $10^{20}$ |

At $d=20$ your "nearest" neighbour is nowhere near you. It is just the least-far point in a space
where everything is far.

### Feature scaling is not optional

The slide says scaling is "essential for KNN", and it is the most common way people silently break it.

🧪 Predict whether a customer churns from `[age (years), annual_income (dollars)]`.

Two candidate neighbours for a query customer aged 40 earning \$50,000:

| Neighbour | Age | Income | $\Delta$age | $\Delta$income | Euclidean distance |
|---|---|---|---|---|---|
| P | 41 | 50,500 | 1 | 500 | $\sqrt{1 + 250{,}000} = 500.001$ |
| Q | 70 | 50,010 | 30 | 10 | $\sqrt{900 + 100} = 31.62$ |

**Q is "closer"** — despite being 30 years older — because a \$500 income difference outweighs a
30-year age difference purely from the units. Income is measured in tens of thousands and age in
tens, so income contributes ~$10^6\times$ more to the squared distance.

The fix — apply **one** of these to every feature:

- **Standardisation:** $x' = \frac{x - \mu}{\sigma}$ → mean 0, std 1. The default.
- **Min-max:** $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ → range $[0,1]$. Sensitive to outliers.

> ⚠️ **Fit the scaler on the training set only, then apply it to validation and test.** Fitting on
> everything leaks test statistics into training — a textbook instance of the data leakage described
> in §14. Use `sklearn.pipeline.Pipeline` so this cannot be got wrong by accident.

### Computational cost and the tree structures

Naive prediction is $O(nd)$ **per query**: compute $d$ subtractions for each of $n$ points. With
$n = 10^6$ and $d = 100$ that's $10^8$ operations for one prediction.

- **KD-trees** recursively split space on one axis at a time. They give roughly $O(\log n)$ lookups
  — but only in low dimensions. Above roughly $d \approx 20$ they degrade to worse-than-brute-force,
  because the curse of dimensionality means the search cannot prune any branches.
- **Ball-trees** partition into nested hyperspheres instead of axis-aligned boxes, and tolerate
  somewhat higher $d$ and non-Euclidean metrics — but they too eventually lose.

> 💡 **What is actually used at scale:** approximate methods that give up exactness. HNSW
> (hierarchical navigable small-world graphs) and IVF-PQ (inverted file with product quantisation)
> underpin FAISS, ScaNN, and every vector database. They answer "find me 10 of the ~15 true nearest
> neighbours, in a millisecond, over a billion vectors". Semantic search and RAG retrieval are KNN —
> at a scale where exactness was traded away deliberately.

### KNN for regression

One line change: instead of taking the majority **class** of the K neighbours, take the **average of
their target values**. With distance weighting:

$$\hat y = \frac{\sum_{i \in \mathcal{N}_K} w_i\, y_i}{\sum_{i \in \mathcal{N}_K} w_i}, \qquad w_i = \frac{1}{d(x, x_i)}$$

⚠️ A structural limitation worth knowing: **KNN regression cannot extrapolate.** Its prediction is
always an average of observed targets, so it is bounded by $[\min y, \max y]$ in the training set.
Ask it about an input beyond the range of your data and it will confidently return the average of the
nearest points you *do* have. Linear regression would at least extend the trend.

```interactive
type: simulator
title: K, scaled and unscaled
concept: How K controls the bias–variance trade-off, and how unscaled features destroy the metric
control: A 2-D scatter of two overlapping classes; sliders for K (1→25) and for a multiplier on the x-axis feature's scale (1× → 1000×). A toggle applies standardisation.
observe: The decision boundary is shaded live. Raising K smooths jagged islands into a clean curve. Raising the scale multiplier collapses the boundary into vertical stripes — the y-feature stops mattering at all. Turning on standardisation restores it instantly.
insight: K=1 gives 100% training accuracy and a boundary full of noise-driven islands, which is why training accuracy cannot be used to pick K. And an unscaled feature doesn't bias KNN slightly — it deletes the other features.
fallback: The K=1/3/5 worked example in §21 (three different answers for one query) and the age-vs-income table in §22.
```

---

# Part 6 — Support Vector Machines

## 23. SVM: the core idea

Slide [raw `slide_087`, 36:43] — recovered:

> **Goal:** Find the hyperplane that maximizes the margin between classes
>
> **Margin:** Distance between decision boundary and nearest data points
> - Support vectors: data points closest to boundary (define the margin)
>
> **Hard Margin SVM:**
> - Assumes data is linearly separable
> - Maximize margin subject to all points correctly classified
>
> **Soft Margin SVM (C-SVM):**
> - Allows some misclassification via slack variables
> - C parameter controls the trade-off:
>   - Large C → narrow margin, fewer violations (may overfit)
>   - Small C → wide margin, more violations (may underfit)

### Why "maximum margin" is the right objective

If the data is linearly separable, there are **infinitely many** lines that separate it. Logistic
regression will find one; which one depends on initialisation and regularisation. The SVM asks a
sharper question: *of all separating hyperplanes, which is furthest from both classes?*

The justification is a generalisation argument. A boundary that squeaks past the training points has
no room for error — a test point drawn slightly differently lands on the wrong side. A boundary with
a wide buffer tolerates perturbation. **Maximising the margin is minimising sensitivity to the exact
positions of the training points**, which is a form of regularisation you get from the geometry
rather than from a penalty term.

> 📚 **Background the slide assumed — what a hyperplane is.** $\mathbf{w}^\top\mathbf{x} + b = 0$
> defines a flat surface one dimension below the space it lives in: a point in 1-D, a line in 2-D, a
> plane in 3-D, a "hyperplane" in $d$-D. $\mathbf{w}$ is the **normal vector** — it points
> perpendicular to the surface, and its direction is what "which way is positive" means. $b$ shifts
> the surface away from the origin. The signed distance from a point $\mathbf{x}_0$ to the hyperplane
> is $\frac{\mathbf{w}^\top\mathbf{x}_0 + b}{\|\mathbf{w}\|}$ — which is the formula the whole margin
> derivation rests on.

### Deriving the margin — and where the "1" in hinge loss came from

Fix the scale so that the closest points on each side satisfy $\mathbf{w}^\top\mathbf{x} + b = \pm 1$.
(We're allowed to do this: scaling $\mathbf{w}$ and $b$ together by any positive constant describes
the *same* hyperplane, so we choose the constant that makes this true.)

The two margin boundaries are $\mathbf{w}^\top\mathbf{x}+b = +1$ and $\mathbf{w}^\top\mathbf{x}+b=-1$.
The distance from the decision boundary to each is $\frac{1}{\|\mathbf{w}\|}$, so the full margin
width is

$$\boxed{\ \text{margin} = \frac{2}{\|\mathbf{w}\|}\ }$$

**Maximising $\frac{2}{\|\mathbf{w}\|}$ is minimising $\|\mathbf{w}\|$**, which for convenience is
written as minimising $\frac12\|\mathbf{w}\|^2$. So the hard-margin SVM is:

$$\min_{\mathbf{w},b}\ \tfrac12\|\mathbf{w}\|^2 \quad\text{subject to}\quad y_i(\mathbf{w}^\top\mathbf{x}_i + b)\ \ge\ 1 \ \ \forall i$$

> 💡 **Look at that constraint.** It is exactly the condition under which hinge loss (§8) is zero.
> The "1" in $\max(0, 1 - y f(x))$ is not arbitrary — it is the margin normalisation. **Hinge loss is
> the SVM constraint, converted from a hard requirement into a penalty.** That connection is the
> single best thing to be able to state about SVMs.

### Soft margin — what happens when the data isn't separable

Real data overlaps. The hard constraint then has **no feasible solution at all** — the optimiser
returns nothing. The fix is to allow violations and charge for them, via **slack variables**
$\xi_i \ge 0$:

$$\min_{\mathbf{w},b,\xi}\ \tfrac12\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\xi_i \quad\text{s.t.}\quad y_i(\mathbf{w}^\top\mathbf{x}_i+b) \ge 1 - \xi_i,\ \ \xi_i \ge 0$$

| $\xi_i$ | Meaning |
|---|---|
| $0$ | correctly classified, outside the margin — a "free" point |
| $0 < \xi_i \le 1$ | correct side, but inside the margin |
| $\xi_i > 1$ | **misclassified** |

**And the slack is exactly the hinge loss.** The smallest $\xi_i$ satisfying the constraint is
$\xi_i = \max(0,\ 1 - y_i(\mathbf{w}^\top\mathbf{x}_i+b))$. Substituting it in:

$$\min_{\mathbf{w},b}\ \underbrace{\tfrac12\|\mathbf{w}\|^2}_{\text{L2 regularisation}} + C\sum_i \underbrace{\max(0, 1 - y_i f(\mathbf{x}_i))}_{\text{hinge loss}}$$

> 💡 **The soft-margin SVM *is* hinge loss plus L2 regularisation.** Nothing more. Every piece of
> this lecture's loss section and its optimisation section converges on this one line. An SVM is not
> an exotic algorithm — it is a specific choice of loss and regulariser, and you could train one with
> the mini-batch SGD from §12.

### What C does

$C$ is the exchange rate between "wide margin" and "few violations".

| | Small $C$ | Large $C$ |
|---|---|---|
| Violations are | cheap | expensive |
| Optimiser prefers | a wide margin, tolerating errors | classifying everything right, even at a razor margin |
| $\|\mathbf{w}\|$ | small | large |
| Bias / variance | high bias, low variance → **underfit** | low bias, high variance → **overfit** |
| As it goes to the extreme | $C\to 0$: ignores the data entirely | $C\to\infty$: recovers the hard-margin SVM |

Note that $C$ is the **inverse** of the usual regularisation strength $\lambda$: in the standard
$\text{loss} + \lambda R(\theta)$ form, bigger $\lambda$ means more regularisation, but bigger $C$
means *less*. This trips people up constantly. (Indeed, dividing the objective through by $C$ gives
$\frac{1}{2C}\|w\|^2 + \sum_i\text{hinge}$, so $\lambda = \frac{1}{2C}$.)

### 🧪 Worked example — compute a margin by hand

Two points: $\mathbf{x}_+ = (2,2)$ with $y=+1$, and $\mathbf{x}_- = (0,0)$ with $y=-1$.

By symmetry, the maximum-margin boundary is perpendicular to the line joining them and passes through
their midpoint $(1,1)$. So $\mathbf{w} = (a, a)$ for some $a > 0$.

Impose the two margin constraints as equalities (both points are support vectors — they're the only
points):

$$\mathbf{w}^\top(2,2) + b = 4a + b = +1$$
$$\mathbf{w}^\top(0,0) + b = b = -1$$

From the second, $b = -1$. Substituting: $4a - 1 = 1 \Rightarrow a = \mathbf{0.5}$.

So $\mathbf{w} = (0.5,\ 0.5)$, $b = -1$.

$$\|\mathbf{w}\| = \sqrt{0.5^2 + 0.5^2} = \sqrt{0.5} = 0.7071$$
$$\text{margin} = \frac{2}{\|\mathbf{w}\|} = \frac{2}{0.7071} = \mathbf{2.828} = 2\sqrt2$$

**Verification:** the actual Euclidean distance between $(0,0)$ and $(2,2)$ is
$\sqrt{4+4} = 2\sqrt2 = 2.828$ ✓. Since both points are support vectors, the margin spans the entire
gap between them, exactly as it should.

**Cross-check with hinge loss:**
- $y_+ f(\mathbf{x}_+) = (+1)(0.5\cdot2 + 0.5\cdot2 - 1) = (+1)(1) = 1 \Rightarrow \max(0, 1-1) = \mathbf{0}$ ✓
- $y_- f(\mathbf{x}_-) = (-1)(0 + 0 - 1) = (-1)(-1) = 1 \Rightarrow \max(0,1-1) = \mathbf{0}$ ✓

Both support vectors sit exactly at margin 1 with zero hinge loss. The solution is consistent.

---

## 24. SVM: the kernel trick

Slide [raw `slide_088`, 38:13] — recovered. This is the slide the previous version of these notes had
to guess at; here it is verbatim:

> **Problem:** Many real datasets are not linearly separable
>
> **Solution:** Map data to higher-dimensional space where it IS separable
> - Kernel trick: compute dot products in high-D space without explicit mapping
>
> **Common Kernels:**
> - Linear: `K(x,y) = x^T y` — use when linearly separable
> - RBF (Gaussian): `K(x,y) = exp(-gamma * ||x-y||^2)` — most versatile
> - Polynomial: `K(x,y) = (x^T y + c)^d` — captures feature interactions
>
> **Hyperparameters:**
> - `gamma` (RBF): Higher = more complex boundary (overfitting risk)
> - `C`: Regularization strength (trade-off margin vs. violations)
> - `d` (Polynomial): Degree of polynomial kernel

### 🧪 Step 1 — see that mapping up actually works: XOR

Four points, the classic non-separable case:

| Point | $x_1$ | $x_2$ | Label |
|---|---|---|---|
| A | 0 | 0 | $-1$ |
| B | 1 | 1 | $-1$ |
| C | 0 | 1 | $+1$ |
| D | 1 | 0 | $+1$ |

In 2-D no line separates $\{A,B\}$ from $\{C,D\}$ — the positives sit on one diagonal and the
negatives on the other. Try it; it's impossible.

Now map to 3-D with $\phi(x_1,x_2) = (x_1,\ x_2,\ x_1 x_2)$:

| Point | $\phi(\mathbf{x})$ | Label |
|---|---|---|
| A | $(0,0,0)$ | $-1$ |
| B | $(1,1,1)$ | $-1$ |
| C | $(0,1,0)$ | $+1$ |
| D | $(1,0,0)$ | $+1$ |

Take $\mathbf{w} = (2, 2, -4)$, $b = -1$ and evaluate $\mathbf{w}^\top\phi(\mathbf{x}) + b$:

| Point | Computation | Score | Sign | Label | ✓ |
|---|---|---|---|---|---|
| A | $0+0-0-1$ | $-1$ | $-$ | $-1$ | ✓ |
| B | $2+2-4-1$ | $-1$ | $-$ | $-1$ | ✓ |
| C | $0+2-0-1$ | $+1$ | $+$ | $+1$ | ✓ |
| D | $2+0-0-1$ | $+1$ | $+$ | $+1$ | ✓ |

**Perfectly separated in 3-D, with all four points exactly at margin 1.** The interaction feature
$x_1x_2$ was the missing ingredient — and note it's exactly what a degree-2 polynomial kernel
supplies.

### Step 2 — why we don't build $\phi$ explicitly

The mapping worked, but it's expensive. A degree-$p$ polynomial map of $d$ features produces
$\binom{d+p}{p}$ dimensions. For $d = 1000$, $p=3$ that is ~167 million features per example. And the
RBF kernel corresponds to an **infinite-dimensional** feature space — you literally cannot build it.

**The escape.** Look back at the SVM: written in its dual form, both training and prediction depend on
the data *only* through inner products $\mathbf{x}_i^\top\mathbf{x}_j$. Never through individual
coordinates. So if you map to $\phi$, you only ever need $\phi(\mathbf{x}_i)^\top\phi(\mathbf{x}_j)$
— a single number.

**A kernel is a function that returns that number directly, without ever constructing $\phi$.**

$$K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^\top\phi(\mathbf{z})$$

### 🧪 Step 3 — proving the trick actually saves work

Take $d=2$ and the polynomial kernel $K(\mathbf{x},\mathbf{z}) = (\mathbf{x}^\top\mathbf{z})^2$.
Expand it algebraically:

$$(\mathbf{x}^\top\mathbf{z})^2 = (x_1z_1 + x_2z_2)^2 = x_1^2z_1^2 + 2x_1x_2z_1z_2 + x_2^2z_2^2$$

Now regroup as an inner product:

$$= \underbrace{(x_1^2,\ \sqrt2\,x_1x_2,\ x_2^2)}_{\phi(\mathbf{x})} \cdot \underbrace{(z_1^2,\ \sqrt2\,z_1z_2,\ z_2^2)}_{\phi(\mathbf{z})}$$

**So $K(\mathbf{x},\mathbf{z}) = (\mathbf{x}^\top\mathbf{z})^2$ is exactly the inner product in that
3-D quadratic feature space.** ∎

Now count the arithmetic with real numbers, $\mathbf{x} = (1,2)$, $\mathbf{z} = (3,4)$:

**Via the explicit map:**
- $\phi(\mathbf{x}) = (1,\ \sqrt2 \cdot 2,\ 4) = (1,\ 2.828,\ 4)$
- $\phi(\mathbf{z}) = (9,\ \sqrt2\cdot 12,\ 16) = (9,\ 16.971,\ 16)$
- Inner product $= 9 + 47.999 + 64 = \mathbf{121}$
- Cost: build two 3-vectors (6 multiplications + 2 square roots), then 3 mults + 2 adds.

**Via the kernel:**
- $\mathbf{x}^\top\mathbf{z} = 1(3) + 2(4) = 11$
- $K = 11^2 = \mathbf{121}$
- Cost: 2 mults, 1 add, 1 square.

**Same answer, 121, from a fraction of the work — and the saving grows explosively with $d$ and the
degree.** For the RBF kernel the explicit route isn't merely slower, it's impossible: the feature
space is infinite-dimensional, yet $\exp(-\gamma\|\mathbf{x}-\mathbf{z}\|^2)$ is three lines of code.

### The three kernels, and when to use them

| Kernel | Formula | Feature space | Use when |
|---|---|---|---|
| **Linear** | $\mathbf{x}^\top\mathbf{z}$ | the original space | $d$ is large relative to $n$ (text, bag-of-words), or the data really is separable. Fastest by far. |
| **RBF / Gaussian** | $\exp(-\gamma\|\mathbf{x}-\mathbf{z}\|^2)$ | infinite-dimensional | The default when you don't know. Can fit any boundary. |
| **Polynomial** | $(\mathbf{x}^\top\mathbf{z} + c)^d$ | all monomials up to degree $d$ | You specifically believe feature *interactions* matter (e.g. price × category). |

🧪 **RBF, computed.** With $\gamma=1$, $\mathbf{x}=(0,0)$, $\mathbf{z}=(1,1)$:
$\|\mathbf{x}-\mathbf{z}\|^2 = 1 + 1 = 2$, so $K = e^{-2} = \mathbf{0.1353}$.
With $\mathbf{z} = (0.1, 0.1)$: $\|\cdot\|^2 = 0.02$, $K = e^{-0.02} = \mathbf{0.9802}$.

> 💡 **The RBF kernel is a similarity score.** It is 1 when the points coincide and decays smoothly
> toward 0 as they separate. Read that way, an RBF-SVM's prediction
> $f(\mathbf{x}) = \sum_i \alpha_i y_i K(\mathbf{x}, \mathbf{x}_i) + b$ is a **weighted vote of the
> support vectors, where each one's influence fades with distance**. Which makes RBF-SVM a close
> cousin of distance-weighted KNN (§21) — except that the SVM *learns* which training points get to
> vote ($\alpha_i \ne 0$ only for support vectors) and how loudly, instead of using all of them
> equally.

### The two hyperparameters, and how they interact

$\gamma$ controls how fast similarity decays with distance:

| $\gamma$ | Each support vector's influence | Boundary | Failure |
|---|---|---|---|
| **small** | reaches far | smooth, nearly linear | underfit |
| **large** | very local | wiggly, wraps individual points | **overfit** — in the limit each training point gets its own island |

$\gamma$ and $C$ **interact**, which is why they must be tuned *jointly* on a 2-D grid rather than one
at a time. Both large is a near-guaranteed overfit; both small underfits. The standard recipe is a
logarithmic grid, e.g. $C \in \{0.1, 1, 10, 100\}$ × $\gamma \in \{0.001, 0.01, 0.1, 1\}$, scored by
cross-validation.

> ⚠️ **Scale your features before an RBF-SVM**, for the same reason as KNN (§22): $\gamma$ multiplies
> a squared Euclidean distance, so an unscaled large-magnitude feature dominates the kernel and the
> others become invisible. `sklearn`'s default `gamma='scale'` divides by
> `n_features * X.var()`, which partially compensates — but it is not a substitute for standardising.

> 🔬 **Research opportunity.** Kernel methods have an elegant theory and a hard ceiling: computing the
> $n \times n$ kernel matrix costs $O(n^2)$ memory and $O(n^2 d)$ time, so exact kernel SVMs stop at
> roughly $10^5$ examples. Random-feature approximations (Random Fourier Features) and Nyström
> sampling trade a little accuracy for linear scaling, and the Neural Tangent Kernel line of work
> shows infinitely-wide neural networks behave like kernel machines — which makes "when does a
> network do something a kernel provably cannot?" a live, open, and genuinely deep question.
> ✅ Confirmed — Rahimi & Recht, *"Random Features for Large-Scale Kernel Machines"*, NIPS 2007
> (Random Fourier Features); Jacot, Gabriel & Hongler, *"Neural Tangent Kernel: Convergence and
> Generalization in Neural Networks"*, NeurIPS 2018.

```interactive
type: simulator
title: Kernel, C, gamma
concept: How kernel choice and hyperparameters reshape an SVM decision boundary, and what over/underfitting looks like
control: Kernel selector (linear / RBF / polynomial), sliders for C (0.01→100, log) and γ (0.001→10, log) and polynomial degree, over a fixed non-separable 2-D dataset with a toggle for the XOR layout.
observe: The boundary, both margin lines, and the highlighted support vectors redraw live. A readout shows the support-vector count and train/validation accuracy side by side.
insight: Push γ high and the boundary shatters into islands around individual points while training accuracy hits 100% and validation accuracy collapses — overfitting made visible. On the XOR toggle, the linear kernel cannot do better than chance no matter how C is set, and RBF solves it instantly.
insight2: As C rises, the support-vector count falls — with violations expensive, only the truly-nearest points remain on the margin.
fallback: The XOR worked example in §24 (separable in 3-D via x₁x₂, impossible in 2-D) and the C table in §23.
```

---

## Putting it together

Everything in this lecture hangs off one skeleton. Here it is, with the dependencies drawn.

```
                        ┌─────────────────────────┐
                        │  What noise do I think   │
                        │  my data has?            │
                        └───────────┬─────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │ Gaussian                  │ Laplacian                 │ Bernoulli /
        │                           │                           │ Categorical
        ▼                           ▼                           ▼
    ┌───────┐                  ┌───────┐                   ┌──────────┐
    │  MSE  │                  │  MAE  │                   │ BCE / CE │
    └───┬───┘                  └───┬───┘                   └────┬─────┘
        │  outliers?               │  need smooth?              │  need a margin
        │  ↓                       │  ↓                         │  instead of a
        └──────► ┌───────┐ ◄───────┘                            │  probability?
                 │ HUBER │                                      ▼
                 └───┬───┘                                  ┌────────┐
                     │                                      │ HINGE  │
                     │                                      └───┬────┘
                     └────────────┬─────────────────────────────┘
                                  │  all of these are functions
                                  │  you must MINIMISE
                                  ▼
                    ┌─────────────────────────────┐
                    │   θ* = argmin  (1/n) Σ L     │   ← §9 ERM
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │ closed form                             │ iterative
              │ (MSE + linear only)                     │ (everything)
              ▼                                         ▼
        Normal equation                        How much data per step?
        w = (XᵀX)⁻¹Xᵀy                          all → Batch GD
        O(d³), doesn't scale                    one → SGD
                                                B   → MINI-BATCH ◀ what you run
                                                          │
                                          add memory ─────┼───── adapt per-parameter
                                          Momentum        │       Adagrad → RMSProp
                                                    ╲     │     ╱
                                                     ╲    │    ╱
                                                      ADAM → AdamW
                                                       + cosine schedule
                                                          │
                                                          ▼
                                            ┌──────────────────────────┐
                                            │  Did it actually work?   │  ← the loss
                                            │  METRIC ≠ LOSS           │    cannot
                                            └────────────┬─────────────┘    tell you
                                                         │
                                 ┌───────────────────────┼──────────────────┐
                                 │ classification        │                  │ regression
                                 ▼                       ▼                  ▼
                        Confusion matrix         threshold sweep      RMSE · MAE · R²
                        ↓                        ↓
                   P / R / F1              ROC-AUC  (balanced)
                                           PR-AUC   (imbalanced) ◀ use this on fraud


   THREE MODELS, THREE PHILOSOPHIES, ALL FITTING THE SAME PIPELINE:

   NAIVE BAYES   models P(x|y) — generative. Counting, one pass, no optimiser at all.
                 Wins on tiny data and huge d. Caps out early.

   KNN           models nothing. Stores everything, decides at query time.
                 Zero training cost, O(nd) inference, dies in high dimensions.

   SVM           models the boundary — discriminative. And it is EXACTLY
                 hinge loss (§8) + L2 regularisation, which means it plugs
                 straight into the optimiser stack above.
                 └── kernel trick: same algorithm, richer geometry, no extra cost.
```

### The five threads that run through the whole lecture

**1. Every loss is a negative log-likelihood.** Gaussian → MSE, Laplace → MAE, Bernoulli → BCE,
Categorical → CE. Choosing a loss *is* asserting a noise model. Huber is the one exception in this
deck — a pragmatic weld rather than a likelihood — and it corresponds to a Gaussian core with heavy
tails.

**2. Every design choice is a bias–variance dial.** $\delta$ in Huber, $\eta$ in gradient descent,
$B$ in mini-batching, the threshold $\tau$ in classification, $K$ in KNN, $C$ and $\gamma$ in the
SVM. Small/large in each case maps onto underfit/overfit. Once you see them as the same dial in
different costumes, you stop memorising them.

**3. Gradient shape determines model behaviour.** MSE's gradient scales with error → fast, outlier-
chasing. MAE's is constant → robust, bouncy. Huber's is capped → free gradient clipping. Hinge's is
exactly zero past the margin → sparsity, support vectors. BCE-through-sigmoid collapses to
$\hat p - y$ → no vanishing gradient. **Ask "what does the gradient look like?" and every property
falls out.**

**4. Scale matters, silently.** Feature scaling breaks KNN and RBF-SVM. Target scaling breaks
Huber's $\delta$. Loss scaling breaks plain SGD but not Adam. None of these throw an error — they
just quietly give you a worse model.

**5. The loss you minimise is not the number you're judged on.** Cross-entropy vs recall-at-fixed-
precision. This gap is where threshold moving, class weighting, $F_\beta$, and PR-AUC all live, and
it is where most real applied-science work actually happens.

---

## Interview prep — Amazon Applied Scientist

### Core questions, easy → hard

<details>
<summary><b>1. What's the difference between MSE and MAE, and when would you use each?</b></summary>

MSE squares the error, MAE takes its absolute value. Three consequences:

- **Outliers:** MSE's quadratic penalty means one bad point can outvote a hundred good ones. MAE is
  robust.
- **What they estimate:** MSE's optimum is the **mean** of the conditional target distribution,
  MAE's is the **median**. I can show that in two lines — set the derivative to zero and the MSE
  condition is "sum of residuals is zero" (the mean), while the MAE condition is "equal counts above
  and below" (the median).
- **Optimisation:** MSE is differentiable everywhere and its gradient shrinks near the optimum, so it
  converges cleanly. MAE has a kink at zero requiring sub-gradients, and its gradient is constant, so
  it bounces around the minimum unless you decay the learning rate.

I'd use MSE when large errors are genuinely disproportionately costly — predicting server capacity,
where being 10× under causes an outage. I'd use MAE when I have label noise or heavy-tailed targets
and I want a typical-case estimate. If unsure, Huber, which is quadratic below $\delta$ and linear
above.
</details>

<details>
<summary><b>2. Why do we use cross-entropy for classification instead of MSE?</b></summary>

Two reasons, and the gradient one is the real one.

**The gradient.** If you put a sigmoid output through MSE, the gradient with respect to the logit
picks up a $\sigma(z)(1-\sigma(z))$ factor, which goes to zero whenever the model is confident. A
confidently *wrong* model then receives almost no gradient and cannot correct itself — a vanishing
gradient. With cross-entropy, everything cancels and the gradient is exactly $\hat p - y$:
prediction minus truth, linear in the error, largest exactly when you're most wrong.

**The principle.** Cross-entropy is the negative log-likelihood of a Bernoulli/categorical
distribution, which is the correct probabilistic model for a discrete label. MSE is the NLL of a
Gaussian, which is the wrong model for a variable that only takes values 0 and 1. And because the
true label distribution is a point mass with zero entropy, minimising cross-entropy is *exactly*
minimising the KL divergence to the truth.
</details>

<details>
<summary><b>3. Your fraud model has 0.97 ROC-AUC. Are you happy?</b></summary>

Not yet — I'd need the class balance first, and on fraud it's typically well under 1%.

ROC-AUC uses FPR on its x-axis, and FPR's denominator is $FP + TN$. With, say, 999,000 legitimate
transactions, tens of thousands of false positives barely move FPR. So the curve can hug the
top-left while precision is catastrophic.

Concretely: 1,000 frauds in 1,000,000 transactions, catching 900 with 10,000 false alarms gives
FPR = 0.010 and TPR = 0.90 — a beautiful ROC point — but precision = 900/10,900 = **8.3%**. Over 91%
of what the investigation team opens is noise.

So I'd ask for **PR-AUC (average precision)** instead, and report it against the correct baseline —
which for PR curves is the positive base rate, not 0.5. Then I'd pick the operating threshold from
the business constraint, usually "what precision do we need for the review team's capacity?", and
report recall at that precision.
</details>

<details>
<summary><b>4. Explain the kernel trick to someone who knows linear algebra but not ML.</b></summary>

Some datasets can't be separated by a straight line but can be separated by a curve. One fix is to
add engineered features — for XOR, adding $x_1x_2$ as a third coordinate makes it linearly separable
in 3-D, and I can show that with a concrete weight vector.

The problem is cost: a degree-3 polynomial expansion of 1,000 features is ~167 million dimensions,
and the Gaussian kernel's feature space is infinite-dimensional, so you can't build it at all.

The trick is that the SVM, written in its dual form, touches the data **only through inner products**
$\mathbf{x}_i^\top\mathbf{x}_j$ — never through individual coordinates. So you don't need
$\phi(\mathbf{x})$; you only need $\phi(\mathbf{x})^\top\phi(\mathbf{z})$, a single scalar. A kernel
function computes that scalar directly.

Example: $K(\mathbf{x},\mathbf{z}) = (\mathbf{x}^\top\mathbf{z})^2$ equals the inner product in the
3-D space $(x_1^2, \sqrt2 x_1x_2, x_2^2)$ — you can verify it by expanding the square. For
$\mathbf{x}=(1,2), \mathbf{z}=(3,4)$: the kernel is $11^2 = 121$ in three operations; the explicit
route builds two 3-vectors and gets the same 121. Same answer, a fraction of the work, and the gap
grows explosively with dimension.
</details>

<details>
<summary><b>5. Walk me from batch gradient descent to Adam, saying what problem each step fixes.</b></summary>

- **Batch GD** — exact gradient, but one parameter update per full pass over the data. With a million
  examples that's a million gradient computations for one step.
- **SGD** — use one example. The gradient is an unbiased estimator of the true gradient (its
  expectation over a uniformly random index is the full gradient), so it's right on average. Same
  compute per epoch, a million times more updates. Cost: very noisy, unstable.
- **Mini-batch SGD** — average $B$ examples. Noise falls as $1/\sqrt{B}$ while cost rises linearly,
  so returns diminish fast and $B$ settles in the hundreds. Critically, it's the version that maps
  onto a GPU matrix multiply. This is what "SGD" means in practice.
- **Momentum** — SGD has no memory, so it zig-zags across narrow valleys. Keep an EMA of gradients;
  consistent directions accumulate to $\frac{1}{1-\beta} = 10\times$ at $\beta=0.9$, oscillating ones
  cancel.
- **Adagrad** — one learning rate for all parameters starves rare features. Scale each parameter's
  step by $1/\sqrt{\text{accumulated squared gradient}}$. But that accumulator only grows, so
  learning dies.
- **RMSProp** — make it an EMA instead of a sum, so it can decrease. Learning never dies.
- **Adam** — combine momentum (1st moment) and RMSProp (2nd moment), plus bias correction because
  both start at zero and are biased low early. Result: for a constant gradient the step is exactly
  $\eta$ regardless of gradient magnitude — near scale-invariance, which is why it needs almost no
  tuning.
- **AdamW** — Adam's L2 penalty goes through the $/\sqrt{\hat v}$ division, so parameters with large
  gradient history get less decay, which inverts the intent. Decouple it: subtract $\eta\lambda w$
  directly. This is the default for transformers, and `torch.optim.Adam(weight_decay=...)` still has
  the old behaviour.
</details>

<details>
<summary><b>6. Naive Bayes assumes features are independent. That's obviously false for text. Why does it work? [combines two concepts]</b></summary>

Three parts.

**Why we make the false assumption:** without it, modelling $P(\mathbf{x}\mid C_k)$ for $d$ binary
features needs $2^d - 1$ parameters per class. At $d = 10{,}000$ vocabulary words that's beyond
astronomical. With it, it's $d$ parameters. Exponential becomes linear — we accept a wrong model in
exchange for one we can actually estimate from a few thousand documents.

**Why it still classifies well:** the assumption wrecks the *probability estimates* — correlated
features like "free" and "money" double-count the same evidence, pushing posteriors toward 0 and 1.
But classification only needs the **argmax**, and the argmax is far more robust than the magnitudes.
If the true posterior is (0.6, 0.4) and NB says (0.99, 0.01), the decision is identical. That's
exactly what "poor probability calibration" on the slide means — and why you should never use raw NB
scores for ranking or expected-value decisions without calibrating them first (Platt scaling or
isotonic regression).

**Where it breaks:** when the errors are systematic rather than random. Heavily duplicated features
— near-synonyms, n-grams overlapping their unigrams — get counted many times, and that *can* flip
the argmax. Feature selection or de-duplication helps.
</details>

<details>
<summary><b>7. You have 100 features and 500 examples. Would you use KNN? [combines two concepts]</b></summary>

Almost certainly not, because of the curse of dimensionality.

Squared Euclidean distance is a sum of $d$ independent per-dimension contributions. By concentration
of measure, its mean grows like $d$ while its standard deviation grows like $\sqrt{d}$, so the
*relative* spread shrinks like $1/\sqrt{d}$. At $d=100$, the nearest and farthest neighbours are
nearly the same distance away — "nearest neighbour" stops carrying information.

The data-density framing is starker: to have a neighbour within 10% of the range on every axis you'd
need on the order of $10^{100}$ points. With 500, every point is effectively isolated.

What I'd do instead: **reduce the dimension first** — PCA, or a learned embedding — down to something
like 10–20 dimensions, standardise, and *then* KNN is reasonable. Or skip it and use a method that
doesn't rely on a distance metric in the raw space — regularised logistic regression or a linear SVM,
both of which handle $d$ comparable to $n$ well. And whatever I use, with $n=500$ I'd report
cross-validated performance with error bars, because a single split of 500 examples has a standard
error big enough to hide most differences I'd care about.
</details>

<details>
<summary><b>8. Show me that a soft-margin SVM is just a regularised linear model. [combines two concepts]</b></summary>

Start from the soft-margin problem:

$$\min_{\mathbf{w},b,\xi}\ \tfrac12\|\mathbf{w}\|^2 + C\sum_i \xi_i \quad\text{s.t.}\quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1-\xi_i,\ \xi_i \ge 0$$

For any fixed $\mathbf{w}, b$, the objective is increasing in $\xi_i$, so the optimum takes the
smallest $\xi_i$ satisfying both constraints:

$$\xi_i = \max\big(0,\ 1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)\big)$$

which is exactly the hinge loss. Substituting eliminates the constraints entirely:

$$\min_{\mathbf{w},b}\ \tfrac12\|\mathbf{w}\|^2 + C\sum_i \max\big(0,\ 1 - y_i f(\mathbf{x}_i)\big)$$

That's **L2 regularisation plus hinge loss** — an unconstrained convex objective I could train with
the mini-batch SGD from earlier. Dividing through by $C$ puts it in the familiar
$\lambda\|\mathbf{w}\|^2 + \sum \text{loss}$ form with $\lambda = \frac{1}{2C}$, which also explains
why large $C$ means *less* regularisation.

So the difference between an SVM and logistic regression is entirely the loss function: hinge versus
logistic. Hinge is exactly zero past margin 1, which is what makes the solution depend only on the
support vectors; logistic loss is never zero, so every point influences the boundary forever.
</details>

<details>
<summary><b>9. Your model trains fine but the loss goes to NaN after 200 steps. Debug it.</b></summary>

I'd work down this list, cheapest first:

1. **Learning rate too high.** The most common cause. On a quadratic, convergence requires
   $\eta < 2/L$ where $L$ is the largest Hessian eigenvalue; above it the error grows geometrically
   and overflows. Drop $\eta$ 10× and see if the NaN moves later or disappears.
2. **`log(0)` in the loss.** If I'm computing `log(sigmoid(z))` manually, a logit of $-50$ underflows
   sigmoid to exactly 0.0 and `log(0)` is `-inf`, which becomes NaN on the next operation. Fix:
   `BCEWithLogitsLoss` / `CrossEntropyLoss`, which fuse the operations stably.
3. **Softmax overflow.** `exp(1000)` is `inf`. Fix: log-sum-exp — subtract the max logit first, which
   is legitimate because softmax is shift-invariant. Frameworks do this; hand-rolled code often
   doesn't.
4. **Bad data.** A NaN or inf in one input feature, or a division by a zero-valued feature during
   normalisation. Assert on the inputs.
5. **Exploding gradients**, particularly with recurrent structures. Add gradient clipping — or note
   that Huber loss gives you exactly this by construction, since its gradient is capped at $\pm\delta$.

Diagnostically I'd log the gradient norm each step. If it grows steadily before the NaN, it's (1) or
(5). If it's fine and then jumps to inf in one step, it's (2), (3) or (4).
</details>

<details>
<summary><b>10. When is accuracy the right metric?</b></summary>

When two conditions both hold: the classes are roughly **balanced**, and the two error types have
roughly **equal cost**.

That's rarer than people assume. It's true for, say, a balanced 10-class image classifier where
confusing a cat for a dog is no worse than the reverse. It's false for fraud, medical screening,
content moderation, defect detection, churn — anything where the interesting class is rare or where
a miss costs differently from a false alarm.

The demonstration: on a 1% disease prevalence, a model that predicts "healthy" for all 1,000 patients
is **99.0% accurate** and has recall 0. A real model catching 8 of 10 sick patients with 50 false
alarms is **94.8% accurate** — worse by accuracy, enormously better in fact.

What I'd report instead depends on the problem: precision/recall at a chosen threshold with the
threshold justified by business cost; $F_\beta$ with $\beta$ chosen to reflect the asymmetry; PR-AUC
if I need a threshold-free summary on imbalanced data; balanced accuracy or MCC if I want one number
that isn't fooled by the base rate.
</details>

<details>
<summary><b>11. Derive why hinge loss gives sparse solutions and logistic loss does not. [combines two concepts]</b></summary>

Compare the gradients with respect to $\mathbf{w}$ for a single example with margin $m = y f(x)$.

**Hinge**, $\max(0, 1-m)$:
$$\frac{\partial}{\partial \mathbf{w}} = \begin{cases} -y\mathbf{x} & m < 1 \\ \mathbf{0} & m > 1\end{cases}$$

**Exactly zero** past margin 1. Not small — zero. That point contributes nothing to any update; you
could delete it from the dataset and get the identical model.

**Logistic**, $\log(1 + e^{-m})$:
$$\frac{\partial}{\partial \mathbf{w}} = \frac{-y\mathbf{x}}{1+e^{m}}$$

At $m=5$ this is $\frac{-y\mathbf{x}}{1+148.4} \approx -0.0067\,y\mathbf{x}$ — small, but **never
zero**. Every point in the training set keeps nudging the boundary forever.

**Consequence:** the SVM's solution is determined entirely by the points with $m \le 1$ — the support
vectors, typically a small fraction of the data. Prediction cost scales with the number of support
vectors, not $n$. Logistic regression's boundary depends on all $n$ points.

**The trade-off:** what you buy with sparsity you pay for in calibration. Hinge is not a proper
scoring rule, so an SVM's raw score is not a probability, and you need Platt scaling to get one.
Logistic loss *is* a proper scoring rule and gives calibrated probabilities natively. So: SVM when
you want a compact decision rule, logistic regression when you need probabilities to threshold or to
feed into an expected-value calculation.
</details>

<details>
<summary><b>12. 🎯 stretch — Why does mini-batch SGD often generalise better than full-batch GD, given full-batch has the exact gradient?</b></summary>

Marked stretch because the honest answer includes "this is still debated".

What's solid: mini-batch gradients are unbiased but noisy, and that noise acts as an implicit
regulariser. It lets the iterate escape sharp, narrow minima and shallow local minima that an exact
gradient would sit in permanently. Empirically, large-batch training reliably finds *sharper* minima
and generalises worse unless you compensate — with the linear scaling rule, warmup, and longer
training.

The proposed mechanism is the flat-minima hypothesis: flat minima are robust to small parameter
perturbations, so a small train/test shift moves the loss less, and SGD's noise preferentially
settles in flat regions because it can't stay in a narrow one.

⚠️ I'd flag that the flat-minima story has real counterarguments — sharpness isn't
reparameterisation-invariant, so "flat" isn't a well-defined property of a solution without fixing a
parameterisation, and there are constructed counterexamples of sharp minima that generalise fine. So
I'd present it as a strong empirical regularity with a plausible but contested explanation, rather
than a theorem.
</details>

### Depth probes — the follow-up after a good answer

| Your answer | The probe | What they want |
|---|---|---|
| "MAE is robust to outliers" | *"Why, mechanically?"* | The gradient is $\pm 1$ regardless of error size, so every point gets one vote. MSE's gradient scales with error, so distant points vote with proportional weight. |
| "Use cross-entropy for classification" | *"What breaks if I use MSE?"* | The $\sigma(1-\sigma)$ factor in the gradient vanishes when the model is confident, so a confidently wrong model can't recover. |
| "Adam is the default optimiser" | *"When would you not use it?"* | Convolutional vision models, where SGD+momentum often generalises slightly better; and any case where you've observed Adam converge to a worse solution — it does happen. Also: use AdamW, not Adam, if you want weight decay. |
| "Higher K smooths the boundary" | *"So why not K = n?"* | You'd predict the majority class for every input, ignoring the features entirely. Maximum bias, zero variance. |
| "The kernel trick avoids computing φ" | *"What's the cost you're paying?"* | The $n\times n$ kernel matrix: $O(n^2)$ memory, $O(n^2 d)$ time. Exact kernel SVMs stop scaling around $10^5$ examples. |
| "PR-AUC for imbalanced data" | *"What's the random baseline for a PR curve?"* | The positive base rate, not 0.5. Must be reported alongside. |
| "I'd tune C and gamma" | *"Jointly or separately?"* | Jointly, on a log grid — they interact, and both-large is a guaranteed overfit. |
| "Model A beat model B by 2 points" | *"Is that significant?"* | On 100 test examples the standard error is ~3.6 points. Report CV mean ± std, bootstrap the interval, and use a paired test (McNemar) to compare. |

### Whiteboard-ready derivations

Three results you should be able to produce cold, with chalk, in under three minutes each.

**① MSE optimises toward the mean; MAE toward the median.**
```
Model: predict a single constant c.
MSE:  L(c) = (1/n) Σ (yᵢ − c)²
      dL/dc = (1/n) Σ −2(yᵢ − c) = 0
      ⟹ Σ yᵢ − nc = 0  ⟹  c = (1/n) Σ yᵢ        ← the MEAN
MAE:  L(c) = (1/n) Σ |yᵢ − c|
      dL/dc = (1/n) Σ −sign(yᵢ − c)
            = (1/n)[ #(below c) − #(above c) ] = 0
      ⟹ #below = #above                          ← the MEDIAN
Sanity: y = [2,4,6,8,100] → mean 24, median 6.
        MSE(24)=1448 < MSE(6)=1772 ✓
        MAE(6)=20.4  < MAE(24)=30.4 ✓
```

**② The BCE gradient through a sigmoid is $\hat p - y$.**
```
p̂ = σ(z) = 1/(1+e⁻ᶻ),   dσ/dz = σ(1−σ)
L  = −[ y log p̂ + (1−y) log(1−p̂) ]

∂L/∂p̂ = −y/p̂ + (1−y)/(1−p̂)

∂L/∂z = ∂L/∂p̂ · ∂p̂/∂z
      = [ −y/p̂ + (1−y)/(1−p̂) ] · p̂(1−p̂)
      = −y(1−p̂) + (1−y)p̂
      = −y + y p̂ + p̂ − y p̂
      = p̂ − y                                    ∎
Everything cancels. This is why sigmoid+BCE are always paired,
and why sigmoid+MSE suffers vanishing gradients.
```

**③ The soft-margin SVM is hinge loss + L2, and margin $= 2/\|w\|$.**
```
Normalise so the closest points sit at wᵀx + b = ±1.
Distance from boundary to each margin plane = 1/‖w‖.
⟹ margin = 2/‖w‖.
Maximise 2/‖w‖  ⟺  minimise ½‖w‖².

Hard margin:  min ½‖w‖²   s.t.  yᵢ(wᵀxᵢ+b) ≥ 1
Soft margin:  min ½‖w‖² + C Σ ξᵢ
              s.t. yᵢ(wᵀxᵢ+b) ≥ 1 − ξᵢ,  ξᵢ ≥ 0

Objective increases in ξᵢ ⟹ take the smallest feasible one:
              ξᵢ = max(0, 1 − yᵢ f(xᵢ))          ← hinge loss

Substitute:   min ½‖w‖² + C Σ max(0, 1 − yᵢ f(xᵢ))
              └─ L2 reg ─┘   └──── hinge loss ────┘
Divide by C:  λ = 1/(2C).  Large C ⟹ weak regularisation.   ∎
```

### Applied scenario — counterfeit product detection on the Amazon marketplace

**Framing.** Sellers list products; some are counterfeit. Removing a genuine listing hurts a
legitimate seller and their customers; leaving a counterfeit up hurts a buyer and the brand. This is
binary classification on a **severely imbalanced** problem — counterfeits are well under 1% of
listings — with **asymmetric, and importantly *tiered*, error costs.** Not one model: a **cascade**.

**Data.** Listing text (title, bullets, description), price relative to the category and to the
brand's own listings, seller tenure and history, image embeddings, review-text signals, and
return/complaint rates. Labels come from enforcement actions, brand-registry reports, and test buys.

> ⚠️ **The label problem is the hard part and I'd say so out loud.** Labels are only available for
> listings someone *looked at*, so they're not missing at random — the training set is biased toward
> what previous enforcement already caught. I'd budget for a small randomly-audited holdout to get an
> unbiased estimate of true prevalence and recall, and I'd treat that audit set as the real test set.

**Model, in stages.**

1. **Baseline first, same day:** Multinomial Naive Bayes on title + description n-grams with Laplace
   smoothing. One pass over the data, no GPU, no tuning. This is the number everything else must
   beat, and it usually catches the crude cases (§20).
2. **Production candidate:** gradient-boosted trees over the tabular + engineered features, plus a
   fine-tuned text encoder; blend the scores. Train with **cross-entropy** — I need calibrated
   probabilities, because the whole system is threshold-driven, so hinge/SVM is the wrong choice
   here despite the tempting sparsity.
3. **Near-duplicate retrieval:** counterfeits cluster — the same bad actor relists under new seller
   IDs. An approximate-KNN index over image and text embeddings (HNSW) surfaces "this listing is
   nearly identical to one we removed last week", which is a signal no per-listing classifier
   produces. This is §21–22 at production scale, with feature scaling and dimensionality reduction
   applied exactly as the slides warn.

**Optimisation.** AdamW, linear warmup then cosine annealing, mini-batch $B=256$. Class imbalance
handled *first* by threshold moving on the validation set — it's free — and only then by class
weighting if that isn't enough.

**Metric.** **Not** accuracy: predicting "genuine" always would score 99.5%. I'd report:
- **PR-AUC** as the threshold-free summary, always quoted against the base-rate baseline;
- **Recall at the precision the enforcement team's capacity allows** as the headline operating number
  — this is the number I'd put in the doc;
- **Recall on the randomly-audited holdout**, since that's the only unbiased estimate;
- Segmented by category and seller tenure, because an aggregate hides the fact that the model may be
  systematically wrong for new sellers.

**Two thresholds, not one.** High threshold → auto-suppress. Middle band → human review queue. Below
→ leave. That converts the precision/recall trade-off (§16) from a single painful compromise into a
routing decision, and it's where the money is.

**Failure modes I'd monitor.**
- **Distribution shift** — counterfeiters adapt within weeks. Track score distribution drift and
  schedule retraining; a static model decays fast here.
- **Feedback loops** — suppressed listings never generate the outcome data that would have labelled
  them, so the training set narrows around what the model already believes. Hold out a small random
  fraction from enforcement to keep collecting unbiased labels.
- **Proxy leakage** — if "seller was previously enforced against" is a feature, the model learns to
  predict past enforcement rather than counterfeiting. Audit feature importances for this.
- **Calibration drift** — the thresholds are set on probabilities, so if calibration drifts the
  business meaning of the threshold silently changes even when AUC looks stable. Monitor a
  reliability curve, not just AUC.

**What I'd ship first.** The Naive Bayes baseline plus threshold-moved logistic regression on tabular
features, behind the human-review queue. It is explainable to the enforcement team, deployable in
days, and establishes the measurement infrastructure — the audit holdout, the segmented dashboards,
the drift monitors — that every later model needs. Then iterate on the model with the harness already
in place.

### Leadership Principles tie-in

**Dive Deep.** The whole diagnostic chain in this lecture is Dive Deep in miniature: "ROC-AUC is
0.97" is a surface number; noticing that FPR's denominator contains 999,000 negatives, recomputing
precision at 8.3%, and telling the team the model would flood their queue — that's refusing to accept
a good-looking metric at face value. *Concretely:* on the counterfeit project, insisting on a
randomly-audited holdout because the enforcement-derived labels are biased by past enforcement, even
though it costs audit budget and makes the reported numbers look worse.

**Insist on the Highest Standards.** Reporting "macro-F1 = 0.86 ± 0.03 across 5 folds, versus 0.84 ±
0.04 for the baseline — not a significant difference on this sample size" instead of "we improved F1
by 2 points". Shipping a model on a difference smaller than its own error bar is the standard this
principle exists to prevent.

**Bias for Action.** Shipping the Naive Bayes baseline and a threshold-moved logistic regression in
week one, behind a human review queue, rather than spending a quarter on the sophisticated model.
It's reversible, it's measurable, and it builds the evaluation harness the better model will need
anyway.

---

## Glossary

| Term | Definition |
|---|---|
| **Accuracy** | $(TP+TN)/n$. Fraction correct. Misleading under class imbalance. |
| **Adagrad** | Per-parameter learning rate scaled by $1/\sqrt{\text{cumulative squared gradient}}$. Decays to zero. |
| **Adam** | Momentum (1st moment) + RMSProp (2nd moment) + bias correction. Near scale-invariant. |
| **AdamW** | Adam with weight decay decoupled from the gradient. The transformer default. |
| **AUC** | Area under a curve. For ROC: the probability a random positive outscores a random negative. |
| **Batch size ($B$)** | Examples per gradient step. Noise falls as $1/\sqrt B$. |
| **BCE** | Binary cross-entropy. NLL of a Bernoulli. Gradient through a sigmoid is $\hat p - y$. |
| **Bias (statistical)** | Error from a model too simple to represent the truth. |
| **Convex** | Bowl-shaped; exactly one minimum, and it's global. |
| **Cosine annealing** | LR schedule decaying along a cosine from $\eta_{\max}$ to $\eta_{\min}$. |
| **Cross-entropy** | $-\sum_c p_c \log q_c$. Coding cost of using $q$ when the truth is $p$. |
| **Curse of dimensionality** | In high $d$, all pairwise distances concentrate; "nearest" loses meaning. |
| **Data leakage** | Test information reaching the model. Symptom: great validation, bad production. |
| **Empirical risk** | Average loss on the training sample. The computable stand-in for true risk. |
| **Epoch** | One complete pass over the training data. |
| **$F_1$ / $F_\beta$** | Harmonic mean of precision and recall; $\beta>1$ weights recall. |
| **Functional margin** | $y\cdot f(x)$. Positive if correct; magnitude is confidence. |
| **Generative vs discriminative** | Models $P(\mathbf{x},y)$ vs $P(y\mid\mathbf{x})$. NB vs LogReg/SVM. |
| **Gradient** | Vector of all partial derivatives. Points in the direction of steepest increase. |
| **Hinge loss** | $\max(0, 1 - y f(x))$. Zero past margin 1 → sparse solutions. |
| **Huber loss** | Quadratic below $\delta$, linear above. $C^1$ everywhere; gradient capped at $\pm\delta$. |
| **Hyperparameter** | Set before training and never updated by the optimiser ($\eta$, $\delta$, $K$, $C$, $\gamma$). |
| **Kernel** | $K(\mathbf{x},\mathbf{z}) = \phi(\mathbf{x})^\top\phi(\mathbf{z})$ computed without building $\phi$. |
| **KL divergence** | $\sum_c p_c\log(p_c/q_c)$. Excess coding cost. $\ge0$, not symmetric. |
| **Laplace smoothing** | Add $\alpha$ to every count so no likelihood is zero. Mandatory for NB on text. |
| **Lazy learner** | No training phase; all work at query time. KNN. |
| **Learning rate ($\eta$)** | Step size. Convergence on a quadratic requires $\eta < 2/L$. |
| **Logit** | The raw pre-activation score, before sigmoid or softmax. |
| **MAE** | $\frac1n\sum\lvert y-\hat y\rvert$. MLE under Laplace. Targets the median. Robust. |
| **Margin (SVM)** | $2/\|\mathbf{w}\|$. The buffer between the boundary and the nearest points. |
| **MLE** | Choose parameters maximising the probability of the observed data. |
| **Momentum** | EMA of gradients. Effective step $\times\frac{1}{1-\beta}$ in consistent directions. |
| **MSE** | $\frac1n\sum(y-\hat y)^2$. MLE under Gaussian. Targets the mean. Outlier-sensitive. |
| **Naive assumption** | Features conditionally independent given the class. Turns $2^d$ parameters into $d$. |
| **One-hot** | A label as a vector with 1 in the true slot, 0 elsewhere. |
| **PR-AUC** | Area under the precision–recall curve. Baseline is the positive base rate, not 0.5. |
| **Precision** | $TP/(TP+FP)$. Of what I flagged, how much was right? |
| **$R^2$** | $1 - SS_{res}/SS_{tot}$. Fraction of the mean-baseline's error eliminated. Can be negative. |
| **Recall / TPR / Sensitivity** | $TP/(TP+FN)$. Of what was there, how much did I find? |
| **RMSE** | $\sqrt{\text{MSE}}$. Same units as the target. Always $\ge$ MAE. |
| **RMSProp** | Adagrad with an EMA instead of a sum, so learning doesn't die. |
| **ROC curve** | TPR vs FPR across all thresholds. |
| **Slack variable ($\xi$)** | Permitted margin violation in a soft-margin SVM. Equals the hinge loss. |
| **Softmax** | $e^{z_c}/\sum_j e^{z_j}$. Logits → a probability distribution. Shift-invariant. |
| **Sub-gradient** | A valid slope at a kink. Any value in $[-1,1]$ at $\lvert e\rvert$'s corner. |
| **Support vector** | A training point with $y f(x) \le 1$. The only points that affect the solution. |
| **Surrogate loss** | A convex upper bound on 0/1 loss that can actually be descended. |
| **Type I / Type II error** | False positive / false negative. |
| **Variance (statistical)** | Error from over-sensitivity to the particular training sample. |

---

## Check yourself

Answer without looking. If you can't, the section number tells you where to go.

1. Without looking anything up, write MSE, MAE, Huber, BCE, CE and hinge, and name the noise
   distribution each corresponds to. *(§2–8)*
2. Data is $[1, 3, 5, 7, 200]$. What constant minimises MSE? What constant minimises MAE? *(§2–3)*
3. Prove Huber is differentiable at $\lvert e\rvert = \delta$. Where did the $-\delta/2$ come from?
   *(§4)*
4. Why does using MSE on a sigmoid output cause vanishing gradients, and what exactly cancels when
   you use BCE instead? *(§6)*
5. Logits are $(3, 1, 0)$ and the true class is the second. Compute the softmax and the cross-entropy
   loss. *(§7)*
6. A point has $y=+1$ and $f(x)=0.8$. It's classified correctly. What is its hinge loss, and why
   isn't it zero? *(§8)*
7. Minimise $L(w)=(w-5)^2$ from $w=0$ with $\eta=0.2$. Give $w$ after three steps. For what $\eta$
   does this diverge? *(§10)*
8. Why is the batch-size sweet spot in the hundreds and not the millions? Give the scaling law.
   *(§12)*
9. Adam's bias correction — what goes wrong without it, and why does the problem disappear as $t$
   grows? *(§13)*
10. What bug does AdamW fix, and is `torch.optim.Adam(weight_decay=0.01)` affected? *(§13)*
11. 1,000 patients, 20 sick. A model flags 60, of which 16 are genuinely sick. Compute accuracy,
    precision, recall and F1. Which number would you put in the report? *(§15)*
12. Why does F1 use the harmonic mean? Show a case where the arithmetic mean gives a misleading
    answer. *(§16)*
13. Compute AUC for scores $(0.9, 0.8, 0.6, 0.3)$ with labels $(1, 0, 1, 0)$ — both by tracing the
    curve and by counting pairs. Do they agree? *(§17)*
14. When does an excellent ROC-AUC coexist with a useless model? Give the numbers. *(§17)*
15. Why can $R^2$ be negative? Construct an example. *(§18)*
16. Why does Naive Bayes make an assumption everyone knows is false? Give the parameter-count
    argument with a number. *(§19)*
17. What breaks without Laplace smoothing, and why does one zero destroy the whole prediction? *(§19)*
18. Classify "free meeting" with the spam model in §19. Show the log-space computation. *(§19)*
19. A KNN query has neighbours at distances $(1.0, 1.4, 2.2, 2.8, 3.2)$ with classes
    $(1, 0, 0, 1, 1)$. Give the prediction for $K = 1, 3, 5$. *(§21)*
20. Why is training accuracy useless for choosing $K$? *(§21)*
21. Explain "in high dimensions all points become equidistant" using the mean and standard deviation
    of a sum of $d$ terms. *(§22)*
22. Compute the max-margin hyperplane for $\mathbf{x}_+=(3,0)$ with $y=+1$ and $\mathbf{x}_-=(1,0)$
    with $y=-1$. What is the margin? *(§23)*
23. Show that soft-margin SVM $=$ hinge $+$ L2. Why does *large* $C$ mean *weak* regularisation?
    *(§23)*
24. Verify that $K(\mathbf{x},\mathbf{z}) = (\mathbf{x}^\top\mathbf{z})^2$ equals
    $\phi(\mathbf{x})^\top\phi(\mathbf{z})$ for $\phi(\mathbf{x}) = (x_1^2, \sqrt2x_1x_2, x_2^2)$,
    both algebraically and with $\mathbf{x}=(2,1)$, $\mathbf{z}=(1,3)$. *(§24)*
25. Which of the six losses in this lecture would you use for: house prices with data-entry errors ·
    a 1,000-class image classifier · click-through prediction · a maximum-margin text classifier?
    Justify each in one sentence. *(§2–8)*

---

## Going deeper

Ranked by what will move your understanding most, per hour spent.

### Read first

1. **Hastie, Tibshirani & Friedman — *The Elements of Statistical Learning*, ch. 4 (linear
   classification) and ch. 12 (SVMs and kernels).** `hard` · Free PDF from the authors' Stanford
   page. The canonical treatment of everything in Parts 4–6, with the SVM dual derived properly.
   Ch. 12's margin geometry is worth the price of admission alone.

2. **Bishop — *Pattern Recognition and Machine Learning*, ch. 1.5 (decision theory), ch. 4.2
   (probabilistic generative models), ch. 6–7 (kernels and sparse kernel machines).** `hard` ·
   The best explanation anywhere of *why* losses take the shapes they do, via decision theory. Ch. 4.2
   is the rigorous version of §19–20.

3. **Sebastian Ruder — "An overview of gradient descent optimization algorithms."** `solid` ·
   A blog post that covers momentum → Adagrad → RMSProp → Adam with the equations and clear
   intuitions, in the same lineage order as §13. The single best hour you can spend on the
   optimisation half of this lecture. ⚠️ It predates AdamW and everything after, so pair it with the
   AdamW paper below.

### The original papers, where they matter

4. **Kingma & Ba — "Adam: A Method for Stochastic Optimization" (ICLR 2015).** `solid` ·
   Short and readable. Read it specifically for the bias-correction derivation in §2, which is the
   part everyone skips and the part that's actually subtle.

5. **Loshchilov & Hutter — "Decoupled Weight Decay Regularization" (ICLR 2019).** `solid` ·
   The AdamW paper. It exists because of a bug that every framework had shipped for years, which
   makes it a genuinely instructive read about how the field self-corrects. Directly answers
   interview question 5's last bullet.

6. **Cortes & Vapnik — "Support-Vector Networks" (Machine Learning, 1995).** `hard` ·
   The soft-margin SVM paper. Reading a foundational paper that is still exactly how people describe
   the method 30 years later is worth doing once.

7. **Huber — "Robust Estimation of a Location Parameter"** (*Annals of Mathematical Statistics*
   35(1), 1964, pp. 73–101). `hard` · Where Huber loss comes from, and where the efficiency arguments
   behind the $\delta$ choice live. ✅ Confirmed venue, year, and page range; the $1.345\sigma$
   constant (see §4) is confirmed and was corrected from an earlier $1.35\sigma$ rounding in this
   file.

> ✅ **On the two optimisers the slide names without explaining, now confirmed:** **Muon**
> ("MomentUm Orthogonalized by Newton-Schulz"), released by Keller Jordan in October 2024. **SOAP**
> is Vyas et al., *"SOAP: Improving and Stabilizing Shampoo using Adam"*, arXiv:2409.11321, ICLR
> 2025. Both are genuinely recent — worth reading the original sources directly rather than relying
> on a summary, since the field is still moving.

### Do, don't just read

8. **scikit-learn User Guide §1.4 (SVM), §1.6 (nearest neighbors), §1.9 (naive Bayes), §3.3 (metrics).**
   `intro` · Unusually good documentation — each section explains the method, not just the API. The
   metrics page is the fastest way to internalise §14–18, and the SVM page has the definitive
   plots of what `C` and `gamma` do.

9. **Build all six losses from scratch in NumPy, then check against PyTorch.** `intro` ·
   Concretely: write `mse`, `mae`, `huber`, `bce_with_logits`, `cross_entropy`, `hinge`, plus their
   gradients. Verify each gradient numerically with a finite-difference check
   ($\frac{L(w+\epsilon)-L(w-\epsilon)}{2\epsilon}$). You will discover the log-sum-exp and
   `log(0)` problems from §6–7 personally, which is worth more than reading about them.

10. **Distill.pub — "Why Momentum Really Works."** `solid` ·
    An interactive article where you drag the momentum coefficient and watch the trajectory on an
    ill-conditioned quadratic. It makes the condition-number argument in §10 and §13 visual in a way
    no static text achieves.

11. **The `plot_roc_curve` / `plot_precision_recall_curve` examples in scikit-learn, run on a
    deliberately imbalanced dataset you construct.** `intro` · Generate 10,000 negatives and 50
    positives, fit anything, and plot both curves side by side. Seeing the ROC look excellent while
    the PR curve collapses — on data you made yourself — fixes §17 permanently.



