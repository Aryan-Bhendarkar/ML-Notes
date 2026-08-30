---
title: "Causal Inference — Part 3: Causal ML Methods for Treatment Effect Estimation"
topic: causal-inference
lecture: 23
source: "output/Lecture_23 - Module 7 Causal Inference Part 3"
slides: 102
---

# Causal Inference — Part 3: Causal ML Methods for Treatment Effect Estimation

> Runtime ~1:20:00. Built from the raw capture in `output/Lecture_23 - Module 7 Causal Inference Part
> 3/` (102 raw frames), not `slides_deduped/` — see project memory `slides-deduped-is-lossy`.
> Instructor: **Pranita Khandelwal**, Senior Data Scientist, International Machine Learning team at
> Amazon (buyer abuse prevention and payment experiences) — confirmed from the slide nameplate. This
> is the third and final lecture in the causal inference series, completing the progression from
> foundational theory (Part 1) through classical estimation (Part 2) to modern ML-based methods for
> heterogeneous treatment effect estimation and debiased causal inference.

---

## What you'll understand after reading this

1. **Explain why ATE alone is insufficient** for decision-making and state the precise definition
   of CATE (τ(x)), relating it back to the potential outcomes framework from Part 1.
2. **Build an S-Learner, T-Learner, and X-Learner from first principles**, explaining the
   information-sharing properties and failure modes of each, and choose among them given the
   imbalance structure of a dataset.
3. **Explain why honest estimation in causal trees** prevents overfitting and enables valid
   confidence intervals, and describe how causal forests reduce variance through ensembling.
4. **Explain why representation learning is necessary** for deep causal methods, and describe how
   TarNet and CFRNet achieve covariate balance in a learned latent space.
5. **State the three key ideas of the DML framework** (partialing out, Neyman orthogonality,
   cross-fitting) and explain how each addresses a specific failure mode of naive ML-based
   causal estimation.
6. **Derive the Robinson decomposition** and explain why it eliminates the unknown confounding
   function g(x) from the partially linear model.
7. **Explain the Neyman orthogonality condition** and why it makes the treatment effect estimator
   insensitive to first-order nuisance function errors, reducing bias from O(δ) to O(δ²).
8. **Describe the R-Learner's objective function** and explain why it generalizes OLS from fixed
   treatment effects to heterogeneous ones.
9. **Explain the DR-Learner's doubly robust and Neyman orthogonal properties**, and describe how
   it generates pseudo-outcomes from AIPW for subsequent meta-learning.
10. **Map the entire causal ML landscape** — from meta-learners to representation learning to
    orthogonal learners — onto the canonical causal DAG, explaining which edges each method
    targets and how.

---

## Before we start: what you need to know

### Prerequisite 1 — Potential outcomes and CATE (from Parts 1–2)

This lecture assumes familiarity with the potential outcomes framework (§3 of Part 1): $Y_i(1)$,
$Y_i(0)$, the switching equation, the fundamental problem of causal inference, and the five
estimands (ITE, ATE, ATT, ATC, CATE). If $\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$ is
unfamiliar, read Part 1 §3 and Part 2 §4 first.

### Prerequisite 2 — The classical estimation toolkit (from Part 2)

This lecture builds directly on Part 2's classical methods: stratification, matching, propensity
scores, IPW, regression adjustment, and doubly robust (AIPW) estimation. In particular, the
**nuisance functions** $e(x) = P(T=1|X=x)$ (propensity) and $\mu_t(x) = \mathbb{E}[Y|T=t, X=x]$
(outcome model) from Part 2 §16 are used throughout this lecture without re-derivation.

### Prerequisite 3 — Regularization bias

> **Regularization bias** — when a flexible ML model (random forest, neural network, etc.) is
> optimized for predictive accuracy using techniques like L1/L2 penalties, dropout, early stopping,
> or tree pruning, the resulting model's predictions can be systematically biased in ways that don't
> vanish with more data. This is the central problem Part 2 §16 flagged and Part 3 solves.

In standard supervised learning, regularization bias is acceptable because it trades bias for
variance, often improving generalization — see [Supervised Learning Part 1
§9](../Supervised%20Learning/supervised-learning-01.md), "The bias–variance trade-off," for the full
derivation of why that trade is normally a good one. In causal inference, this same bias propagates
directly into treatment effect estimates and **does not vanish with more data** — making it a much
more serious problem that requires specialized solutions (DML, orthogonal learners).

### Prerequisite 4 — The factual MSE loss

> **Factual MSE** — a loss function that computes mean squared error only on the *observed*
> (factual) outcomes, not the counterfactual ones. Since we only observe $Y_i(T_i)$ and never
> $Y_i(1-T_i)$ for any unit, this is the only loss directly computable from observational data.

$$\mathcal{L}_{\text{factual}} = \sum_{i: T_i=1} \big(Y_i - \hat\mu_1(X_i)\big)^2 + \sum_{i: T_i=0} \big(Y_i - \hat\mu_0(X_i)\big)^2$$

---

## The big picture

Parts 1 and 2 established *why* causal inference is hard and *what* classical methods can do with
low-dimensional data. This lecture asks: **what happens when the data is high-dimensional, the
treatment effects are heterogeneous, and we need ML's flexibility without ML's regularization
bias?**

```
Part 1 — Foundations (Lecture 21)
   Why P(Y|X) ≠ P(Y|do(X)). DAGs, do-calculus, backdoor/frontdoor. RCTs as gold standard.
        │
        ▼
Part 2 — Classical Estimation (Lecture 22)
   Six recipes (stratification → AIPW) scored against a known $1,794 truth.
   Finding: overlap matters more than estimator choice.
   Bridge: every method reduces to estimating e(x) and μ_t(x) — exactly what ML can replace.
        │
        ▼
Part 3 — Causal ML Methods (Lecture 23, this file)
   │
   ├── Heterogeneous treatment effects: from ATE to CATE (§1–§2)
   │
   ├── Meta-learners: S, T, X (§3) — flexible ML plug-ins, but regularization bias
   │
   ├── Causal trees & forests (§4) — honest estimation, valid CIs, interpretable
   │
   ├── Deep representation learning: TarNet, CFRNet (§5) — neural-net-based balancing
   │
   ├── DML framework (§6–§8): the theoretical backbone
   │   ├── Partialing out (Robinson decomposition) — remove confounders
   │   ├── Neyman orthogonality — suppress nuisance-function bias
   │   └── Cross-fitting — prevent overfitting
   │
   ├── R-Learner (§9) — DML + heterogeneous effects via meta-learner interface
   │
   ├── DR-Learner (§10) — doubly robust + orthogonal + meta-learner
   │
   └── The full landscape mapped onto the canonical DAG (§11)
        │
        ▼
   Beyond treatment effects: robust ML, causal generative models, causal RL,
   fairness, explainability, LLMs (§12)
```

---

## 1. From Average to Heterogeneous Treatment Effects

### 1.1 Why ATE is not enough

The ATE answers *"on average, what is the impact of a treatment?"* But for decisions — which
patients get a drug, which customers get a coupon, which fulfillment centers get a new algorithm —
we need to know **who benefits most and by how much**.

$$\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$$

*The conditional average treatment effect: the expected effect for units sharing covariate value
$x$.*

- CATE is a **function of x**, not a single number.
- ATE is the marginal: $\tau = \mathbb{E}_X[\tau(x)]$.
- When treatment effects are **homogeneous** (every unit benefits equally), $\tau(x) = \tau$ for
  all $x$ and CATE collapses to ATE. In most real-world settings, effects are heterogeneous.

```interactive
type: diagram
title: Homogeneous vs. heterogeneous effects, same ATE
concept: ATE can hide a completely different underlying reality — this is the whole reason CATE exists
control: Toggle between "Homogeneous" and "Heterogeneous" τ(x) distributions
observe: Both distributions share the same average effect (ATE ≈ the same number in both states, per the lecture's own toggle), but the homogeneous case is a narrow spike around that number while the heterogeneous case is spread wide — some units near zero or negative, others far above
insight: Two populations can report the identical ATE while one has a treatment that works uniformly and the other has a treatment that helps some people a lot, does nothing for others, and actively harms a few — a single average number cannot distinguish these, which is exactly why targeting decisions require τ(x), not τ
fallback: The lecture's own slide shows this as two side-by-side distribution plots of τ(x): a "Homogeneous" narrow spike and a "Heterogeneous" wide spread, both centered on the same ATE (≈6.0 in the lecture's example) — the reader can picture the same story from the coupon example in §1.2, where Alice/Bob/Charlie/Dana's four very different CATEs (+0%, +12%, +1%, −4%) would still average out to some single unremarkable ATE if you only looked at the population mean.
```

### 1.2 When heterogeneity matters — two worked examples

**Precision medicine.** A population-level study finds a blood-pressure drug lowers systolic
pressure by 8 mmHg on average. But that average hides the fact that some patients benefit by 15
mmHg while others are actively *harmed*. Only CATE can identify which patients should receive the
drug and which should not.

**Marketing coupons.** Consider four customers [slide 16/18, "Who should receive the offer?"]:

| Customer | Buy prob. | CATE $\hat\tau(x)$ | Coupon? |
|----------|------------|---------------------|---------|
| Alice | 60% | +0% | No |
| Bob | 60% | **+12%** | **Yes** |
| Charlie | 90% | +1% | No |
| Dana | 8% | **−4%** | No |

*Who gets the coupon?* Not Charlie — his high baseline means the coupon barely moves him (+1%). Not
**Dana** — she isn't merely a weak candidate, she's **actively harmed**: her CATE is *negative*
(−4%), meaning the coupon would *reduce* her purchase probability (perhaps a discount cheapens the
perceived value of a product she was already inclined to buy at full price, or the offer arrives at
the wrong moment). **Bob** is the right target: same baseline purchase probability as Alice, but a
genuinely large positive incremental effect. This is exactly why CATE-based targeting must do more
than rank customers by expected benefit — it must also screen out units the treatment would actively
harm, something an ATE-only view can never see. The ATE is meaningless here; only the per-unit CATE
tells you both who to target and who to protect from the treatment entirely.

### 1.3 Policy learning with budget constraints

If treating unit $i$ costs $c$, then the optimal policy assigns treatment to exactly those units
where $\tau(x_i) > c$:

$$\pi^*(x) = \mathbb{1}[\tau(x) > c]$$

This is the **value-to-cost** decision rule: rank all units by their CATE and treat the top $K$
units whose benefit exceeds the treatment cost.

> 💡 **Key insight — CATE estimation is the prerequisite for personalization.** ATE tells you
> whether a treatment works *on average*; CATE tells you *for whom* it works. The entire
> progression from ATE → CATE → policy learning is the analytical backbone of precision medicine,
> targeted marketing, and adaptive resource allocation — and it all requires estimating a
> *function* $\tau(x)$, not a single number.

---

## 2. From CATE Definition to Estimation — the Identification Bridge

### 2.1 The causal-to-statistical translation

Under the same three identification assumptions from Part 2 §7 (SUTVA, unconfoundedness,
positivity), the causal quantity $\tau(x)$ can be expressed as a statistical one:

$$\tau(x) = \mathbb{E}[Y|T=1, X=x] - \mathbb{E}[Y|T=0, X=x] = \mu_1(x) - \mu_0(x)$$

**Words before symbols:** the CATE for people with covariates $x$ equals the difference between
what treated people with covariates $x$ actually earned ($\mu_1(x)$) and what control people with
covariates $x$ actually earned ($\mu_0(x)$). Both quantities are *directly estimable from data*
— we observe $Y$ for both treated and control units at each $x$, so we can fit models to predict
$Y$ given $(T, X)$ and then compute the difference.

### 2.2 Why ML is both the solution and the problem

ML models excel at estimating $\mu_t(x) = \mathbb{E}[Y|T=t, X=x]$ when $X$ is high-dimensional
and the relationship is nonlinear. But Part 2 §16 flagged the catch: **ML models are optimized for
prediction, not causal estimation**, and the regularization that makes them good at prediction
introduces bias that doesn't vanish with more data.

The rest of this lecture is a catalog of strategies for getting ML's flexibility while avoiding
its regularization bias:

| Strategy | Methods | How it handles bias |
|----------|---------|-------------------|
| **Naive plug-in** | S-Learner, T-Learner | Doesn't — regularization bias propagates directly |
| **Imputation + re-estimation** | X-Learner | Reduces bias via propensity-weighted blending |
| **Honest partitioning** | Causal trees/forests | Data splitting prevents overfitting |
| **Balanced representations** | TarNet, CFRNet | Learns a space where treated/control are indistinguishable |
| **Debiasing via orthogonality** | DML, R-Learner, DR-Learner | Neyman orthogonality + cross-fitting suppress bias to second order |

---

## 3. Meta-Learners — Flexible ML Plug-ins

Meta-learners use standard supervised-learning models as building blocks to estimate CATE. They
differ in *how many models* are fit and *how the CATE is extracted*.

### 3.1 S-Learner (Single Learner)

Train **one model** $\hat{s}(X, T)$ that predicts $Y$ from covariates $X$ *and* treatment $T$ as a
feature:

$$\hat\tau(x) = \hat{s}(x, T{=}1) - \hat{s}(x, T{=}0)$$

**Strength:** simple — just one model, uses all data.

**Weakness:** regularization can suppress the treatment variable's importance. If $T$ is a weak
predictor of $Y$ relative to $X$, the model may assign $T$ near-zero weight, making
$\hat{s}(x, 1) \approx \hat{s}(x, 0)$ for all $x$ — **shrinking CATE toward zero** (regularization
bias in the causal direction).

> ⚠️ **The S-Learner's failure mode is the canonical regularization-bias problem.** The model is
> optimized to minimize overall prediction error, and if the treatment effect is small relative to
> the outcome variance, regularization may treat $T$ as noise and effectively zero it out. This
> produces a CATE estimate of approximately zero everywhere — not because the treatment has no
> effect, but because the model wasn't asked to find one.

### 3.2 T-Learner (Two Learners)

Train **two separate models** — one on treated units, one on controls:

$$\hat\tau(x) = \hat\mu_1(x) - \hat\mu_0(x)$$

**Strength:** avoids S-Learner's treatment-drowning problem — each model only needs to predict
well within its own arm.

**Weaknesses:**
1. **No information sharing** between arms — each model sees only half the data.
2. **Imbalanced arms** force different regularization strengths, creating differential bias that
   doesn't cancel in the difference.

### 3.3 X-Learner (Cross-Learner)

A four-stage procedure that combines T-Learner's flexibility with propensity-based information
sharing:

**Stage 1:** Fit T-Learner (two models: $\hat\mu_1$, $\hat\mu_0$).

**Stage 2:** Impute individual treatment effects:
- For treated units ($T_i=1$): $D_i^{(1)} = Y_i - \hat\mu_0(X_i)$ (observed outcome minus imputed
  counterfactual)
- For control units ($T_i=0$): $D_i^{(0)} = \hat\mu_1(X_i) - Y_i$ (imputed counterfactual minus
  observed outcome)

**Stage 3:** Fit second-stage models: $\hat\tau_1(x)$ from $\{(X_i, D_i^{(1)}) : T_i=1\}$ and
$\hat\tau_0(x)$ from $\{(X_i, D_i^{(0)}) : T_i=0\}$.

**Stage 4:** Combine with propensity-weighted blending:

$$\hat\tau(x) = e(x) \cdot \hat\tau_0(x) + (1 - e(x)) \cdot \hat\tau_1(x)$$

**Why the propensity weighting works:** look at the formula again — the weight on $\hat\tau_0(x)$
is $e(x)$ itself, and the weight on $\hat\tau_1(x)$ is $1-e(x)$. So when $e(x)$ is **low** (treated
units are scarce at $x$), the estimate leans *toward* $\hat\tau_1(x)$, not $\hat\tau_0(x)$. This is
correct, and the reason is subtler than "more raw data points": $\hat\tau_1(x)$ is fit on the
pseudo-outcomes $D_i^{(1)} = Y_i - \hat\mu_0(X_i)$ for treated units — and $\hat\mu_0$, the *control*
outcome model, is well-estimated *precisely because* controls are abundant when $e(x)$ is low. So
even though $\hat\tau_1$ is fit on relatively few (treated) points, each of those points' target
value is a high-quality imputation, built from a reliably-estimated counterfactual. Symmetrically,
when $e(x)$ is **high** (controls are scarce), $\hat\mu_1$ is the reliably-estimated model, so
$\hat\tau_0$'s pseudo-outcomes $D_i^{(0)}=\hat\mu_1(X_i)-Y_i$ are high quality, and the weight
$e(x)$ on $\hat\tau_0$ is correspondingly large. **The propensity score weights toward whichever
$\hat\tau$'s pseudo-outcome construction leaned on the better-estimated nuisance model** — not
toward "the arm with more rows."

> 💡 **Key insight — X-Learner recycles all the data.** T-Learner effectively discards cross-arm
> information (the treated model never sees control outcomes and vice versa). X-Learner's
> imputation step feeds *all* units' information into both second-stage models, with the propensity
> score acting as a reliability weight — it's a principled way to share information across arms
> without requiring a single model to handle both simultaneously (which is what causes the
> S-Learner's drowning problem).

---

## 4. Causal Trees and Forests

### 4.1 Causal trees — honest estimation

> Causal trees and the honest-estimation idea originate with **Athey & Imbens (2016), "Recursive
> Partitioning for Heterogeneous Causal Effects"** — confirmed directly on the slide footer for this
> section.

A **causal tree** partitions the covariate space so that each leaf contains a subset of units
where the treatment effect is approximately homogeneous. Unlike a standard decision tree (which
predicts $Y$), each leaf directly estimates $\tau(x)$:

$$\hat\tau_{\text{leaf}} = \bar{Y}_{1,\text{leaf}} - \bar{Y}_{0,\text{leaf}}$$

The average treated outcome minus the average control outcome within that leaf.

**Splitting criterion:** maximize heterogeneity in treatment effects between child nodes, subject
to a balance constraint (neither child should be dramatically smaller than the other).

🧪 **Worked example — which split would a causal tree favor?** Estimating the effect of a discount
coupon (treatment) on customer spend (outcome). Parent node: $\hat\tau=5\%$. Two candidate splits:

| Split | Left leaf | Right leaf | Spread |
|---|---|---|---|
| **A — on "weeks since last purchase"** (< 4 wks vs. ≥ 4 wks) | $\hat\tau=1\%$ | $\hat\tau=12\%$ | **11 points** |
| **B — on "device"** (mobile vs. web) | $\hat\tau=4.8\%$ | $\hat\tau=5.2\%$ | 0.4 points |

The causal tree favors **Split A**. Split B's two children are both close to the parent's 5% — device
barely moderates the coupon's effect, so splitting on it buys almost no new information. Split A's
children diverge sharply from each other (1% vs. 12%) — recency of last purchase is a real driver of
*how much* the coupon effect varies, which is exactly what the splitting criterion is built to find.
A standard decision tree, by contrast, would split on whatever best predicts $Y$ directly — it has
no reason to prefer A over B unless recency also happens to predict raw spend, which is a different
question entirely from predicting the coupon's *effect*.

**Honest estimation** — the key innovation:
- Split the data into two halves: one for **structure** (deciding where to split), one for
  **estimation** (computing leaf-level treatment effects).
- The structure sample never estimates treatment effects; the estimation sample never influences
  splits.

**Why honesty matters:** using the same data for both splitting and estimation creates overfitting
bias — the tree will find splits that look good on the data used to estimate them, just as a
standard decision tree overfits to its training data. Honest estimation breaks this cycle,
producing **unbiased treatment effect estimates** and enabling valid confidence intervals.

### 4.2 Causal forests — reducing variance

A single causal tree has high variance (different splits of the data produce very different trees).
**Causal forests** solve this by:
1. Growing many trees on random subsamples (bagging).
2. Using random feature subsets at each split (decorrelation, like random forests).
3. Averaging treatment effect estimates across all trees for each data point.

The result: low-bias (from honesty), low-variance (from ensembling), interpretable (tree
structure) treatment effect estimates with valid confidence intervals.

> ⚠️ **Not every causal forest is Neyman orthogonal.** Orthogonality (§7.2) holds for the
> **local-centering (residual-on-residual) causal forest of Athey, Tibshirani & Wager (2019)**,
> which folds a DML-style residualization step into each split — it does **not** hold for the
> original Wager & Athey (2018) formulation. If you need the orthogonality guarantee specifically
> (robustness to small nuisance-model errors, per §7.2), make sure you're using the 2019
> local-centering variant, not just "a causal forest."

> 🎯 **Interview framing — why causal forests are well-regarded.** Causal forests combine three
> desirable properties that no other method in this lecture offers simultaneously:
> 1. **Unbiasedness** via honest estimation (unlike meta-learners).
> 2. **Valid confidence intervals** (unlike neural-net methods).
> 3. **Interpretability** — the tree structure shows *which features drive effect heterogeneity*
>    (unlike black-box representation learners).

---

## 5. Deep Representation Learning for Causal Inference

### 5.1 The representation-learning idea

When $X$ is high-dimensional (images, text, graphs, or tabular data with hundreds of features),
estimating $\mu_1(x)$ and $\mu_0(x)$ directly is hard — the treated and control distributions in
the original covariate space may barely overlap (Part 2 §10's curse of dimensionality).

**Representation learning** addresses this by learning a mapping $\phi: \mathcal{X} \to \mathcal{Z}$
that projects the original covariates into a lower-dimensional space $\mathcal{Z}$ where:

1. Treated and control units are **hard to distinguish** (balanced distributions — this removes
   confounding bias).
2. The representation is **rich enough** to predict outcomes well (this preserves statistical
   efficiency).

Once $\phi(x)$ is learned, the CATE is extracted via a simple meta-learner head:

$$\hat\tau(x) = \hat{h}_1(\phi(x)) - \hat{h}_0(\phi(x))$$

### 5.2 TarNet (Treatment-Agnostic Representation Network)

> **TarNet** — Shalit, Johansson & Sontag (2017), building on the balanced-representation framing
> of Johansson, Shalit & Sontag (2016). A neural network that learns a balanced representation and
> estimates per-arm outcome models simultaneously.
>
> ⚠️ TarNet and CFRNet (§5.3) are introduced in the **same paper** by the same author team — not two
> independent works. The slide's own footer cites both as "Johansson, Shalit & Sontag (2016) ·
> Shalit et al. (2017), ICML."

**Architecture:**
```
X → [Shared Encoder φ] → [Treatment Head h₁] → μ̂₁(x)
                                 ↘
                                  z = φ(x)
                                 ↗
                  [Control Head h₀] → μ̂₀(x)
```

- **Shared trunk (encoder):** MLP layers that produce a latent representation $z = \phi(x)$
  common to both treatment arms.
- **Separate heads:** each arm gets its own MLP head that predicts $Y$ from $z$.
- **Loss:** factual MSE only — the model never sees counterfactual outcomes.

**Why the shared representation learns balance:** both heads must predict well using the *same*
latent code $z$. If $z$ retains information that distinguishes treated from control units (i.e.,
information about confounders), then each head would need to compensate for the different
distributions — wasting capacity. The most efficient $z$ for joint prediction is one where the
treated and control distributions *overlap* as much as possible, because then neither head needs
to extrapolate far from its training data.

**Limitation:** the balance is *implicit* — it emerges as a side effect of optimizing the factual
MSE, not as an explicit objective. If the factual MSE can be minimized well enough without
achieving balance, the model may not bother.

### 5.3 CFRNet (Counterfactual Regression Network)

> **CFRNet** — Shalit, Johansson & Sontag (2017) — the same paper that introduces TarNet (§5.2).
> Extends TarNet with an explicit distributional balance penalty.

$$\mathcal{L}_{\text{CFR}} = \underbrace{\text{Factual MSE}}_{\text{prediction}} + \underbrace{\lambda \cdot \text{IPM}(\hat{\mu}_1, \hat{\mu}_0)}_{\text{balance penalty}}$$

The **Integral Probability Metric (IPM)** measures the distance between the representation
distributions of treated and control units:

$$\text{IPM}(\hat{\mu}_1, \hat{\mu}_0) = \sup_{f \in \mathcal{F}} \left| \mathbb{E}_{z \sim \hat{\mu}_1}[f(z)] - \mathbb{E}_{z \sim \hat{\mu}_0}[f(z)] \right|$$

**Two common IPM choices:**
- **MMD (Maximum Mean Discrepancy):** projects both distributions into a kernel space and computes
  the difference in means. If MMD = 0, the distributions are identical.
- **Wasserstein distance:** the minimum "transport cost" to convert one distribution into the other
  (inspired by optimal transport / earth-mover's distance, also used in GANs).

**Double robustness of CFRNet:** the authors proved that the counterfactual prediction error is
bounded by:

$$\text{Error}_{\text{counterfactual}} \leq \text{Error}_{\text{factual}} + \text{IPM}$$

If the factual prediction error is small AND the distributions are balanced (IPM ≈ 0), the
counterfactual error is also small — because the model is interpolating rather than extrapolating.

> 💡 **Key insight — TarNet vs. CFRNet is implicit vs. explicit balance.** TarNet hopes the shared
> representation will be balanced as a *side effect* of joint prediction. CFRNet *demands* balance
> via an explicit penalty. In practice, the penalty gives more control over the bias-variance
> trade-off: increasing $\lambda$ improves balance (reducing bias) but may worsen prediction
> (increasing variance), while decreasing $\lambda$ does the opposite.

### 5.4 DragonNet — Adding a Propensity Head

> **DragonNet** — a third member of the TarNet/CFRNet family: deep semiparametric learning that
> adds a **propensity head** $\hat e(x)$ alongside the shared representation and the two outcome
> heads $\hat\mu_0(x), \hat\mu_1(x)$.

*"TARNet + propensity head + targeted regularization."*

```
X → [Shared Encoder φ] → [Treatment Head h₁] → μ̂₁(x)
                  ↘             ↘
                    [Propensity Head] → ê(x)
                  ↗             ↗
        [Control Head h₀] → μ̂₀(x)
```

**Why add a propensity head at all, when the goal is the outcome heads $\hat\mu_0,\hat\mu_1$?**
Predicting $e(x)$ is an *auxiliary task* — the representation $\phi(x)$ is trained so that it's
simultaneously useful for predicting the outcome **and** the treatment assignment. This is a form
of multi-task learning: forcing $\phi(x)$ to retain enough information to predict $e(x)$ prevents
the representation from over-compressing away exactly the covariates that matter for confounding
adjustment — the same covariates the propensity-score machinery in Part 2 §11 relies on.

**Targeted regularization:** DragonNet adds a small correction term to the loss, in the spirit of
Part 2 §14's doubly robust AIPW correction — a one-step update that nudges the outcome predictions
using the fitted propensity, specifically to reduce the *final* CATE estimate's bias, not just the
outcome-prediction loss. This targets the actual estimand ($\tau(x)$) rather than only the
intermediate nuisance functions.

> 💡 **Key insight — DragonNet closes the loop TarNet/CFRNet leave open.** TarNet balances
> implicitly; CFRNet balances explicitly via an IPM penalty on the *representation*; DragonNet adds
> a *propensity* signal and a *targeted* correction so that the final effect estimate, not just the
> intermediate representation, is what the extra machinery is optimizing for.

---

## 6. Orthogonal Causal ML — The DML Framework

### 6.1 The problem: regularization bias in nuisance functions

Parts 1–2 established that every causal estimator reduces to estimating nuisance functions —
$e(x)$ and $\mu_t(x)$. When these are estimated with ML models (random forests, neural nets,
etc.), the regularization that makes them good *predictors* introduces bias that propagates into
the treatment effect estimate.

**The critical insight:** even with more data, this bias doesn't vanish. The ML model's
regularization creates a systematic error in the nuisance estimates, and that systematic error
translates directly into a systematic error in $\hat\tau$. Increasing $n$ reduces variance but
not this bias — the sampling distribution of the estimator remains centered away from the true
value.

The **DML (Double/Debiased Machine Learning)** framework, introduced by Chernozhukov et al.
(2018), solved this with three key ideas:

1. **Partialing out** — remove confounders by regressing out their effects first.
2. **Neyman orthogonality** — make the estimator insensitive to first-order nuisance errors.
3. **Cross-fitting** — prevent overfitting from using the same data for nuisance estimation and
   treatment effect estimation.

---

## 7. The Three Ideas of DML

### 7.1 Partialing Out (Robinson Decomposition)

Consider the partially linear structural model:

$$Y = \tau_0 T + g(X) + \epsilon$$

where $\tau_0$ is the treatment effect (a constant), $g(X)$ is the unknown confounding function,
and $\epsilon \perp (T, X)$.

**The problem:** $g(X)$ is unknown, and any error in estimating it biases $\hat\tau_0$.

**Robinson's trick:** take conditional expectations of the structural equation:

$$\mathbb{E}[Y|X] = \tau_0 \cdot \mathbb{E}[T|X] + g(X)$$

Subtract:

$$Y - \mathbb{E}[Y|X] = \tau_0 \cdot (T - \mathbb{E}[T|X])$$

Define **residuals:**

$$\tilde{Y} = Y - m(X), \quad \tilde{T} = T - e(X)$$

where $m(X) = \mathbb{E}[Y|X]$ and $e(X) = \mathbb{E}[T|X]$ are the nuisance functions. Then:

$$\tilde{Y} = \tau_0 \cdot \tilde{T} + \epsilon$$

**Words before symbols:** the confounding function $g(X)$ has completely disappeared. Whatever
relationship $X$ has with $Y$ that's *not* through $T$ has been removed by subtracting the
conditional expectation. Whatever relationship $X$ has with $T$ (confounding) has been removed
by subtracting the propensity. What remains — the relationship between $\tilde{Y}$ and
$\tilde{T}$ — is the direct causal effect $\tau_0$.

This is a **residual-on-residual regression**: regress the outcome residual on the treatment
residual. The slope is $\tau_0$.

**Connection to the DAG:** the confounders $X$ impact both $T$ and $Y$. By removing their
predictable influence on both (via $m(X)$ and $e(X)$), we eliminate the backdoor path. Whatever
relationship survives between the residuals is the causal (front-door) effect.

> 💡 **Key insight — partialing out is the algebraic realization of blocking the backdoor path.**
> Part 1 §8's backdoor adjustment formula sums over confounder strata to remove their effect.
> Robinson's decomposition achieves the same goal through subtraction rather than summation — it's
> a continuous, ML-compatible version of the same insight.

### 7.2 Neyman Orthogonality

> The term "Neyman orthogonality" traces back to **Neyman (1959), "Optimal Asymptotic Tests of
> Composite Hypotheses"** — confirmed directly on the slide footer for this section — decades before
> its modern application to ML-based causal inference by Chernozhukov et al. (2018).

The residual-on-residual estimator requires knowing $m(X)$ and $e(X)$ exactly. In practice, we
estimate them with ML, introducing errors $\delta_m$ and $\delta_e$. The question is: **how do
these errors affect $\hat\tau_0$?**

Define the **moment condition** (score function):

$$\psi(\tilde{Y}, \tilde{T}, \tau) = (\tilde{Y} - \tau \cdot \tilde{T}) \cdot \tilde{T} = 0$$

This is the "set the derivative of the loss to zero" condition from OLS — at the correct $\tau_0$,
the expected score is zero.

**Without orthogonality:** errors in the nuisance estimates propagate linearly — if the nuisance
error is $\delta$, the bias in $\hat\tau_0$ is $O(\delta)$.

**Neyman orthogonality** requires the derivative condition:

$$\left. \frac{\partial \mathbb{E}[\psi]}{\partial \eta} \right|_{\eta=\eta_0} = 0$$

where $\eta = (m, e)$ represents the nuisance functions. If this holds, the bias drops to
$O(\delta^2)$ — the product of the two nuisance errors rather than their sum.

**Intuition:** the moment condition is designed so that first-order errors in $m$ and $e$ cancel
out. Only second-order interactions between the two errors remain. This is like a Taylor expansion
where the first derivative is zero — the function is "flat" at the true parameter, so small
perturbations have negligible effect.

```
Nuisance error δ on x-axis, bias in τ̂ on y-axis:

Without orthogonality:  bias ∝ O(δ)        — linear relationship
With orthogonality:     bias ∝ O(δ²)       — quadratic, much smaller for small δ
```

> ⚠️ **Why this is the theoretical breakthrough.** Before DML, using ML for nuisance estimation
> meant accepting irreducible regularization bias in the treatment effect. Neyman orthogonality
> provides a *principled way to suppress that bias* without requiring the ML models to be unbiased
> themselves — you just need them to be "not too wrong" (small $\delta$), and the second-order
> suppression does the rest.

### 7.3 Cross-Fitting

Even with Neyman orthogonality, there's one remaining problem: if the *same data* is used to both
train the nuisance models AND estimate $\tau_0$, the fitted nuisance models' errors become
**correlated with the outcomes**, reintroducing overfitting bias.

**Cross-fitting** fixes this by splitting the data into $K$ folds:

```
For each fold k = 1, ..., K:
    1. Train nuisance models m(X), e(X) on ALL data EXCEPT fold k
    2. Compute residuals Ỹ, T̃ on fold k using the out-of-fold predictions
    3. Accumulate the moment conditions across folds

Final: solve the averaged moment condition for τ̂₀
```

**Why this works:** the residuals on fold $k$ are computed using models that *never saw* fold
$k$'s data — so the estimation errors are uncorrelated with the outcomes in fold $k$. This
breaks the overfitting channel.

> 💡 **Key insight — cross-fitting is to causal ML what k-fold cross-validation is to supervised
> learning.** In supervised learning, you hold out data to get an unbiased estimate of
> generalization error. In causal ML, you hold out data to get an unbiased estimate of the
> treatment effect — the underlying principle is identical, but the target is different.

---

## 8. DML in Summary

| DML Idea | Problem It Solves | Mechanism |
|----------|------------------|-----------|
| **Partialing out** | Confounders contaminate the T-Y relationship | Regress out confounders' effects via residualization |
| **Neyman orthogonality** | Nuisance estimation errors bias $\hat\tau$ at $O(\delta)$ | Design the score function so first-order errors cancel |
| **Cross-fitting** | Overfitting from same-data nuisance estimation | Train nuisance models out-of-fold to decorrelate errors |

The combined DML estimator is: **consistent**, **asymptotically normal**, and admits **valid
confidence intervals** — even when nuisance functions are estimated by flexible ML models with
non-negligible regularization bias.

---

## 9. R-Learner — DML for Heterogeneous Effects

> **R-Learner** — Nie & Wager (2021), "Quasi-Oracle Estimation of Heterogeneous Treatment Effects,"
> *Biometrika* — confirmed directly on the slide's own footer and body text.

The DML framework (§6–§8) estimates a **constant** treatment effect $\tau_0$. The **R-Learner**
extends it to estimate a **function** $\tau(x)$ — heterogeneous effects.

Starting from the partially linear model with a *functional* treatment effect:

$$Y = \tau(X) \cdot T + g(X) + \epsilon$$

After Robinson's partialing-out transformation (§7.1), the residual-on-residual equation becomes:

$$\tilde{Y} = \tau(X) \cdot \tilde{T} + \epsilon$$

**R-Learner's objective:** find the function $\tau(x)$ that minimizes the weighted sum of squared
residuals, using the treatment residual as a weight:

$$\hat\tau(x) = \underset{\tau}{\arg\min} \sum_{i=1}^{n} \big(\tilde{Y}_i - \tau(X_i) \cdot \tilde{T}_i\big)^2$$

> 🟡 **Simplification, for fidelity.** The slide's own objective includes an additional, explicitly
> **optional** regularizer term $\Lambda_n(\tau)$ added to this sum (e.g., a penalty controlling the
> complexity of $\tau(x)$) — omitted above because the slide itself labels it optional and the loss
> above is the core R-Learner idea. If you're implementing this from scratch, know that a
> regularization term is standard practice for controlling the flexibility of the fitted $\tau(x)$.

**Words before symbols:** this is OLS, but instead of fitting a constant slope, we're fitting a
*function* $\tau(x)$ that maps each covariate value to its own treatment effect. The "regressors"
are the treatment residuals $\tilde{T}_i$ (confounders removed), and the "weights" are proportional
to $\tilde{T}_i^2$ — units where the treatment residual is large (high variation in treatment
assignment) get more weight.

> 🎯 **Interview framing — how the R-Learner generalizes OLS.** In standard OLS, $\hat\beta = (X^TX)^{-1}X^TY$ —
> the slope is a constant. The R-Learner replaces the constant slope with a *function* $\tau(x)$,
> optimized via a similar least-squares criterion but with the critical modification that both the
> "regressors" ($\tilde{T}$) and the "targets" ($\tilde{Y}$) have been residualized to remove
> confounding. The $\tilde{T}_i^2$ weighting naturally downweights units where treatment
> assignment was deterministic (low $\tilde{T}$) and upweights units where it was near-random
> (high $\tilde{T}$).

---

## 10. DR-Learner — Doubly Robust + Orthogonal + Meta-Learner

> **DR-Learner** — Kennedy (2023), "Towards optimal doubly robust estimation of heterogeneous causal
> effects" — confirmed directly on the slide's own footer ("Doubly-robust learner · Kennedy (2023)").

The **DR-Learner** combines the doubly robust property from Part 2 §14 (AIPW) with the
orthogonal/cross-fitting machinery of DML, wrapped in a meta-learner interface:

**Stage 1:** Estimate nuisance functions $\hat\mu_1(x)$, $\hat\mu_0(x)$, $\hat e(x)$ using ML.

**Stage 2:** For each unit, compute the **AIPW pseudo-outcome:**

$$\hat{\Delta}_i = \underbrace{\hat\mu_1(X_i) - \hat\mu_0(X_i)}_{\text{regression component}} + \underbrace{\frac{T_i}{\hat e(X_i)}\big(Y_i - \hat\mu_1(X_i)\big) - \frac{1-T_i}{1-\hat e(X_i)}\big(Y_i - \hat\mu_0(X_i)\big)}_{\text{IPW correction}}$$

This pseudo-outcome is a **noisy, per-unit estimate of $\tau(x_i)$** — a "doubly robust"
denoising of the raw treatment effect.

**Stage 3:** Regress the pseudo-outcomes on $X$:

$$\hat\tau(x) = \underset{\tau}{\arg\min} \sum_i \big(\hat\Delta_i - \tau(X_i)\big)^2$$

**Two key properties:**
1. **Doubly robust** (from Part 2 §14): consistent if *either* the outcome model OR the propensity
   model is correctly specified — you don't need both to be right.
2. **Neyman orthogonal** (from §7.2): nuisance estimation errors propagate at second order,
   $O(\delta^2)$, not first order.

> 💡 **Key insight — DR-Learner is the "best of both worlds" estimator.** It inherits AIPW's
> double robustness (insurance against one model being wrong), DML's orthogonality (suppression of
> regularization bias), and the meta-learner's ability to estimate *heterogeneous* effects via the
> final-stage regression on $X$.

---

## 11. The Full Landscape

### 11a. The lecture's own comparison table

This is the deck's actual closing summary table ("Comparison of HTE estimators"), reproduced
directly — a complete side-by-side of every method covered in this lecture, including **DragonNet**
(§5.4), which the synthesis view in §11b below doesn't separately break out:

| Method | Generation | Models learned | Main objective | Neyman orthogonal? | Doubly robust? | Handles imbalance |
|---|---|---|---|---|---|---|
| **S-learner** | Meta-learner | $f(x,t)$ | Single outcome model; $\hat\tau(x)=f(x,1)-f(x,0)$ | No | No | Moderate |
| **T-learner** | Meta-learner | $\mu_0(x),\mu_1(x)$ | Separate outcome models; $\hat\tau(x)=\mu_1(x)-\mu_0(x)$ | No | No | Poor |
| **X-learner** | Meta-learner | $\mu_0(x),\mu_1(x),\tau_0(x),\tau_1(x)$, optionally $e(x)$ | Impute treatment effects and learn CATE directly | No | No | Excellent |
| **R-learner** | Orthogonal / DML-style | $m(x),e(x),\tau(x)$ | Minimize residualized R-loss | **Yes** | No | Good |
| **DR-learner** | Orthogonal + doubly robust | $\mu_0(x),\mu_1(x),e(x),\tau(x)$ | Regress doubly-robust pseudo-outcome | **Yes** | **Yes** | Good |
| **Causal forest** | Forest-based orthogonal estimator | $\tau(x)$ (plus nuisances internally) | Honest splitting to maximize treatment-effect heterogeneity | Yes\* | No | Good |
| **TARNet** | Deep representation learning | $\Phi(x),\mu_0(x),\mu_1(x)$ | Shared representation + treatment-specific outcome heads | No | No | Moderate |
| **CFRNet** | Deep representation learning | $\Phi(x),\mu_0(x),\mu_1(x)$ | TARNet + representation balancing (MMD / Wasserstein) | No | No | Good |
| **DragonNet** | Deep semiparametric learning | $\Phi(x),\mu_0(x),\mu_1(x),e(x)$ | TARNet + propensity head + targeted regularization | No (efficiency-inspired) | No | Good |

*\* Orthogonality holds for the local-centering (residual-on-residual) causal forest of Athey,
Tibshirani & Wager (2019), not the original Wager & Athey (2018) formulation — see the ⚠️ callout
in §4.2.*

### 11b. A synthesis view — mapped onto the canonical DAG

The table above tells you *what* each method does; this is my own reorganization of *why* — every
method in Parts 1–3 can also be understood as a different strategy for removing the causal
influence of confounders on the treatment-outcome relationship. Here's how each method maps onto
the canonical DAG ($X \to T$, $X \to Y$, $T \to Y$):

| Method | How it removes confounding | Edge targeted |
|--------|--------------------------|---------------|
| **RCT** | Randomizes $T$, breaking the $X \to T$ edge | Eliminates confounding by design |
| **Matching/stratification** | Conditions on $X$, blocking the backdoor path | Blocks $X \to T$ and $X \to Y$ |
| **Propensity weighting** | Reweights to create a pseudo-population where $T \perp X$ | Eliminates $X \to T$ influence |
| **Meta-learners (S/T/X)** | Models $\mu_t(x)$ to adjust for $X$ | Conditions on $X$ via outcome models |
| **Causal forests** | Partitions $X$-space, estimates within leaves | Stratifies on $X$ with honest estimation |
| **TarNet/CFRNet/DragonNet** | Learns balanced $\phi(x)$ where $T \perp \phi(X)$ | Transforms $X$ to eliminate confounding in $\mathcal{Z}$ |
| **DML / R-Learner** | Residualizes both $T$ and $Y$ on $X$, estimates on residuals | Removes $X$'s influence on both arms simultaneously |
| **DR-Learner** | AIPW pseudo-outcomes + orthogonal regression | Combines residualization with double robustness |

> 🎯 **Interview framing — the unified view.** Whether you're doing propensity weighting, matching,
> representation learning, or DML, the goal is always the same: make the comparison between
> treated and control groups as if the confounders $X$ didn't exist. Each method does this through
> a different mechanism — weighting, conditioning, residualizing, or learning a balanced
> representation — but they're all solving the same problem of blocking the backdoor path.

---

## 12. Beyond Treatment Effects — Frontiers of Causal ML

The lecture concludes with a survey of applications where causal reasoning enhances standard ML:

### 12.1 Robust and generalizable ML

Standard supervised learning assumes train and test distributions match. When they don't (domain
shift), predictions fail — e.g., a cow detector trained on grassy backgrounds fails on beach
backgrounds. Causal ML seeks to learn **invariant features** (the causal mechanisms) rather than
spurious correlations.

### 12.2 Causal generative modeling

Instead of sampling from a learned distribution, causal generative models ask: *"what would this
MRI look like if the patient didn't have the disease?"* or *"how would this face change if age
increased while everything else stayed fixed?"* Applications include controllable diffusion models
and counterfactual data augmentation.

### 12.3 Causal RL

- **Dynamic treatment regimes:** sequences of treatments (e.g., titrating drug dosage over time).
- **Causal bandits:** exploiting causal structure to predict outcomes under interventions.
- **World models:** learning environment dynamics via causal reasoning.
- **Off-policy evaluation:** estimating a new policy's value from data collected under an old
  policy — a fundamentally causal problem.
- **Counterfactual credit assignment:** determining which specific action in a sequence caused
  the reward.

### 12.4 Fairness and explainability

- **Counterfactual fairness:** *"would this person's loan application have been approved if their
  race were different, all else equal?"* — a causal question, not a statistical one.
- **Explainability:** identifying which variables causally drive a model's decisions, beyond
  correlational feature importance.

### 12.5 LLMs and causal reasoning

- Automatically extracting causal knowledge from scientific literature to build causal graphs.
- Studying whether LLMs reason about interventions and counterfactuals, or merely exploit
  statistical associations (and hallucinate).

---

## Putting it together

```
        THE CAUSAL ML METHOD LANDSCAPE
        ════════════════════════════════

        Problem: high-dimensional observational data → heterogeneous treatment effects
        Goal: estimate τ(x) without regularization bias

                    ┌────────────────────────────┐
                    │   Meta-Learners (§3)        │
                    │   S-Learner: one model       │ ← simple, but regularization bias
                    │   T-Learner: two models      │ ← no information sharing
                    │   X-Learner: four stages     │ ← propensity-weighted blending
                    └──────────┬─────────────────┘
                               │
                    ┌──────────▼─────────────────┐
                    │   Causal Trees/Forests (§4) │
                    │   Honest estimation          │ ← unbiased + valid CIs
                    │   Ensemble → low variance    │ ← interpretable
                    └──────────┬─────────────────┘
                               │
                    ┌──────────▼─────────────────┐
                    │   Deep Representation (§5)  │
                    │   TarNet: implicit balance   │ ← neural-net flexibility
                    │   CFRNet: explicit IPM       │ ← distributional balance
                    └──────────┬─────────────────┘
                               │
                    ┌──────────▼─────────────────┐
                    │   DML Framework (§6–§8)      │
                    │   Partialing out             │ ← remove confounders
                    │   Neyman orthogonality       │ ← suppress bias to O(δ²)
                    │   Cross-fitting              │ ← prevent overfitting
                    └──────────┬─────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌──────────────┐     ┌──────────────────┐
          │  R-Learner   │     │  DR-Learner      │
          │  DML + CATE  │     │  AIPW + DML +    │
          │  via residual│     │  meta-learner     │
          │  regression  │     │  double robust    │
          └──────────────┘     └──────────────────┘

        All methods: target the same causal DAG edge (T → Y),
        differ only in HOW they block the confounders (X → T, X → Y)
```

Three threads run through this lecture:

1. **Every method is a different mechanism for blocking the backdoor path** — whether through
   weighting (propensity methods), conditioning (stratification, meta-learners), residualizing
   (DML), or representation learning (TarNet/CFRNet). The goal never changes; only the mechanism
   does.

2. **The progression from meta-learners to DML is the progression from "ML models are flexible"
   to "ML models' biases are controllable."** S/T/X-Learners use ML freely but accept
   regularization bias. Causal forests bound the bias via honest estimation. DML suppresses it to
   second order via orthogonality. The theoretical sophistication increases monotonically.

3. **Causal inference extends far beyond treatment effect estimation.** The lecture's final
   survey (§12) makes clear that the same DAG-based reasoning, the same backdoor/frontdoor logic,
   and the same doubly-robust estimators appear in fairness, RL, generative modeling, and LLM
   evaluation — causal thinking is not a subfield of statistics but a *mode of reasoning* that
   improves any domain where actions change outcomes.

---

## Interview prep — Amazon Applied Scientist

### Core questions

<details><summary><b>1. What is CATE and why can't you just use ATE for targeting decisions?</b></summary>

CATE, $\tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]$, is the expected treatment effect for units
with covariates $x$. ATE averages over the whole population and tells you whether a treatment
works "on average," but real decisions require knowing *who* benefits (and who is harmed). In the
coupon example from §1.2, CATE identifies Bob (+12%) as the optimal target, Charlie (+1%) as not
worth targeting, and — critically — Dana (−4%) as someone the coupon would actively *harm*, a
distinction an ATE-only view could never surface. Targeting based on ATE alone would waste resources
on units where the effect is negligible, or worse, push the offer onto units it damages.
</details>

<details><summary><b>2. Explain the regularization bias problem in the S-Learner and how the X-Learner addresses it.</b></summary>

The S-Learner trains one model on $(X, T) \to Y$. When the treatment effect is small relative to
outcome variance, regularization may assign $T$ near-zero weight, making $\hat{s}(x, 1) \approx
\hat{s}(x, 0)$ and shrinking CATE toward zero — a systematic bias that doesn't vanish with more
data. The X-Learner avoids this by: (1) training separate models per arm (T-Learner step, avoiding
treatment drowning), (2) imputing individual treatment effects for all units, and (3) combining
second-stage estimates via propensity-weighted blending that automatically leans on the more
data-rich arm.
</details>

<details><summary><b>3. What is honest estimation in causal trees and why does it matter?</b></summary>

Honest estimation splits the data: one half decides *where* to split the tree (structure), the
other half estimates *what* the treatment effect is in each leaf (estimation). This prevents the
overfitting that occurs when the same data both selects splits and evaluates them — the tree would
find splits that look good on its own training data, producing biased leaf-level estimates. With
honest estimation, the leaf estimates are unbiased, and valid confidence intervals can be
constructed.
</details>

<details><summary><b>4. Explain TarNet's implicit balance mechanism — why does sharing a representation between treatment heads help?</b></summary>

TarNet's shared encoder produces a latent code $z$ used by both the treatment and control heads.
If $z$ retains information that distinguishes treated from control units (confounders), each head
must accommodate different distributions, wasting capacity. The most efficient shared code is one
where treated and control distributions overlap maximally — so the model learns balance as a
*side effect* of optimizing factual prediction accuracy. CFRNet makes this explicit via an IPM
penalty, giving direct control over the bias-variance trade-off.
</details>

<details><summary><b>5. State the three key ideas of DML and the specific failure mode each addresses.</b></summary>

1. **Partialing out (Robinson decomposition):** removes confounders by regressing both $Y$ and $T$
   on $X$, then estimating the treatment effect on the residuals — addresses the confounding
   problem ($X \to T$, $X \to Y$ edges).

2. **Neyman orthogonality:** designs the moment condition so first-order nuisance estimation errors
   cancel, reducing bias from $O(\delta)$ to $O(\delta^2)$ — addresses regularization bias from
   ML-estimated nuisance functions.

3. **Cross-fitting:** trains nuisance models out-of-fold to prevent overfitting when the same data
   is used for both nuisance estimation and treatment effect estimation — addresses the
   overfitting-induced correlation between nuisance errors and outcomes.
</details>

<details><summary><b>6. Derive the Robinson decomposition from the partially linear model. Why does g(x) disappear?</b></summary>

Starting from $Y = \tau_0 T + g(X) + \epsilon$, take conditional expectations: $\mathbb{E}[Y|X]
= \tau_0 \cdot \mathbb{E}[T|X] + g(X)$, so $m(X) = \tau_0 \cdot e(X) + g(X)$. Subtracting:
$Y - m(X) = \tau_0(T - e(X)) + \epsilon$, which is $\tilde{Y} = \tau_0 \tilde{T} + \epsilon$.
The function $g(X)$ disappears because it enters both the original equation and the conditional
expectation equally, canceling in the subtraction. This is why residualization eliminates
confounding: whatever $X$ predicts about $Y$ (beyond its effect through $T$) is subtracted out.
</details>

<details><summary><b>7. Why does increasing the sample size NOT fix regularization bias in naive ML-based causal estimation?</b></summary>

Regularization bias is *systematic*, not random — it comes from the model's optimization objective
(penalizing complexity) biasing the nuisance function estimates in a consistent direction. More
data reduces *variance* (the estimates become more precise) but the *systematic offset* remains,
because the regularization term's influence doesn't diminish. The sampling distribution of $\hat\tau$
becomes tighter but stays centered away from the true value — exactly the phenomenon the DML
framework's orthogonality condition (§7.2) was designed to suppress.
</details>

<details><summary><b>8. Explain the R-Learner's objective function and why $\tilde{T}_i^2$ appears as an implicit weight.</b></summary>

The R-Learner minimizes $\sum_i (\tilde{Y}_i - \tau(X_i) \cdot \tilde{T}_i)^2$. This is
regression of $\tilde{Y}$ on $\tilde{T}$, where $\tilde{T}$ is the treatment residual (confounders
removed). The $\tilde{T}_i^2$ weighting emerges because the loss is quadratic in $\tilde{T}_i$:
units with large treatment residuals (near-random treatment assignment) contribute more to the
objective, naturally upweighting the most informative observations. Units where treatment was
almost deterministic ($\tilde{T} \approx 0$) contribute almost nothing — they can't help
distinguish one $\tau(x)$ from another because the treatment residual provides no leverage.
</details>

<details><summary><b>9. How does the DR-Learner combine double robustness with orthogonality?</b></summary>

The DR-Learner's Stage 2 computes AIPW pseudo-outcomes, which inherit double robustness from
Part 2 §14: they are consistent if either $\hat\mu_t$ or $\hat e$ is correctly specified.
Stage 3 regresses these pseudo-outcomes on $X$, and the full pipeline is designed to satisfy
Neyman orthogonality — so nuisance errors propagate at second order. The result is an estimator
that is simultaneously robust to one nuisance model being wrong (double robustness) AND robust to
both models having small regularization biases (orthogonality), while estimating *heterogeneous*
effects (meta-learner structure).
</details>

<details><summary><b>10. [Combines concepts] A colleague says "just use a neural network to predict Y from (X, T) and take the difference at T=1 vs T=0." Name three problems with this approach and map each to a specific method from this lecture that addresses it.</b></summary>

1. **Regularization bias** — the neural net may suppress $T$'s importance, shrinking the estimated
   effect toward zero → addressed by **T-Learner** (separate models per arm prevent drowning) or
   **DML** (orthogonality suppresses bias to second order).

2. **No covariate balance** — the model learns in a space where treated/control distributions may
   not overlap, causing poor generalization to counterfactuals → addressed by **TarNet/CFRNet**
   (learns a balanced representation explicitly).

3. **No valid uncertainty quantification** — a standard neural net's predictions have no confidence
   intervals for the *causal effect* → addressed by **causal forests** (honest estimation enables
   valid CIs) or **DML** (asymptotic normality of the orthogonal estimator enables CIs).
</details>

### Depth probes

- *"CFRNet's bound says counterfactual error ≤ factual error + IPM. What happens when you increase
  λ (the balance penalty) too much?"* — increasing λ forces the representations to be more
  balanced (lower IPM), reducing bias, but at the cost of prediction quality (higher factual
  error) because the encoder is forced to sacrifice discriminative information to satisfy the
  balance constraint. There's an optimal λ that minimizes the total bound; too much balance
  produces representations that are balanced but useless for prediction, while too little balance
  produces good predictions on the factual data that don't generalize to counterfactuals.

- *"Why is cross-fitting necessary even when Neyman orthogonality already suppresses first-order
  errors?"* — orthogonality suppresses the *systematic* bias from nuisance estimation errors, but
  when the same data trains both the nuisance models and estimates τ, the *residual* overfitting
  creates a correlation between the fitted nuisance values and the outcomes that orthogonality
  doesn't address (it's a finite-sample phenomenon, not an asymptotic one). Cross-fitting
  eliminates this correlation by construction.

- *"The lecture says causal thinking applies beyond treatment effects — give a concrete example of
  how doubly robust estimation solves an RL problem."* — off-policy evaluation: given trajectories
  collected under old policy π₀, estimate the value of a new policy π₁. The "treatment" is the
  new policy's action, the "outcome" is the reward, and confounding arises because the old policy's
  actions are correlated with states. A doubly robust estimator combines a model of the new
  policy's expected reward (outcome model) with importance sampling weights (propensity model) —
  consistent if either model is correct, exactly as in the ATE setting.

### Whiteboard-ready derivations

1. **Robinson decomposition** — §7.1: start from $Y = \tau_0 T + g(X) + \epsilon$, take
   conditional expectations, subtract, derive $\tilde{Y} = \tau_0 \tilde{T} + \epsilon$, and
   explain why $g(X)$ vanishes.

2. **The R-Learner objective as generalized OLS** — §9: write the R-Learner loss
   $\sum_i (\tilde{Y}_i - \tau(X_i) \tilde{T}_i)^2$, take the derivative with respect to
   $\tau(x)$, and show that the solution at each $x$ is a weighted least-squares estimate with
   weights $\tilde{T}_i^2$.

3. **IPM intuition for CFRNet** — §5.3: define IPM as the supremum over a function class of the
   difference in expectations, explain why MMD projects into a kernel space and computes mean
   difference, and state the double-robustness bound Error ≤ factual error + IPM.

### Applied scenario — Amazon promotional targeting

**Framing:** Amazon wants to estimate the causal effect of a discount coupon on purchase conversion
for each customer segment, then target only those segments where the effect exceeds the coupon's
cost.

**Data:** Observational — customer features (purchase history, browsing behavior, device type,
account age), treatment (coupon offered/not offered), outcome (purchased within 7 days).

**Model:** This maps directly onto the lecture's framework. The naive comparison (coupon recipients
vs. non-recipients) suffers from selection bias (Part 2 §5) — customers who receive coupons may
systematically differ from those who don't. Apply the full pipeline:
1. Estimate propensity $e(x)$ and outcome models $\mu_t(x)$ with flexible ML.
2. Use **DR-Learner** (§10) for double robustness + orthogonality + heterogeneous effect
   estimation.
3. Target only customers where $\hat\tau(x) > \text{coupon cost}$ (§1.3's policy learning rule).

**Metric:** The ATT (effect on customers who actually received the coupon) if the question is
"was the coupon worth it for those who got it?" or CATE if the question is "which segments should
we target in the next campaign?"

**Failure modes:** Unmeasured confounders (e.g., a customer's underlying purchase intent, which
makes them both likely to receive a coupon *and* likely to buy) — address via sensitivity
analysis. Distribution shift between training and deployment periods — monitor for this using the
invariant-feature ideas from §12.1.

**What you'd ship:** A DR-Learner estimate with explicit propensity overlap diagnostics (Part 2
§7.1), confidence intervals (from DML's asymptotic normality), and a targeting policy that assigns
coupons only where $\hat\tau(x) > c$ with sufficient statistical confidence.

**Leadership Principle tie-in:** **Customer Obsession** — targeting coupons to customers who
actually benefit (high CATE) rather than sending them to everyone improves the customer experience
(relevant coupons, not spam). **Dive Deep** — correctly distinguishing confounders (past purchase
frequency) from mediators (post-coupon browsing behavior) before deciding what to adjust for
requires the causal-graph reasoning from Part 1 §5–§6.

---

## Glossary

- **AIPW (Augmented Inverse Probability Weighting)** — doubly robust estimator combining outcome
  regression with IPW correction; foundation for DR-Learner.
- **CATE (Conditional Average Treatment Effect)** — $\tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]$,
  the heterogeneous, covariate-dependent treatment effect.
- **Causal forest** — ensemble of honest causal trees with random subsampling and feature
  subsetting; provides unbiased CATE estimates with valid confidence intervals.
- **Causal tree** — decision tree that estimates treatment effects in each leaf via honest
  estimation (separate data for structure vs. estimation).
- **CFRNet (Counterfactual Regression Network)** — TarNet extended with an explicit IPM balance
  penalty for distributional alignment.
- **Cross-fitting** — training nuisance models on data folds held out from the moment-condition
  computation to decorrelate estimation errors from outcomes.
- **DragonNet** — TarNet/CFRNet extended with a propensity head and targeted regularization; deep
  semiparametric learning aimed directly at the final CATE estimate.
- **DML (Double/Debiased Machine Learning)** — framework combining partialing out, Neyman
  orthogonality, and cross-fitting for unbiased treatment effect estimation with ML nuisance
  functions.
- **Double robustness** — property of AIPW/DR-Learner: consistent if either nuisance model
  (propensity or outcome) is correctly specified.
- **Factual MSE** — loss computed only on observed outcomes; the standard training objective for
  causal models.
- **Honest estimation** — splitting data into separate samples for tree structure and leaf-level
  estimation to prevent overfitting bias.
- **IPM (Integral Probability Metric)** — family of distribution distance measures including MMD
  and Wasserstein; used in CFRNet's balance penalty.
- **MMD (Maximum Mean Discrepancy)** — kernel-based IPM that projects distributions into a
  reproducing kernel Hilbert space and computes mean difference.
- **Neyman orthogonality** — property of a moment condition where first-order nuisance estimation
  errors cancel, reducing bias from O(δ) to O(δ²).
- **Partialing out** — Robinson decomposition: regressing both $Y$ and $T$ on $X$ to produce
  residuals free of confounding influence.
- **R-Learner** — DML-based meta-learner that estimates heterogeneous effects via weighted
  residual regression.
- **DR-Learner** — doubly robust + Neyman orthogonal meta-learner using AIPW pseudo-outcomes.
- **Regularization bias** — systematic error introduced by ML model regularization that doesn't
  vanish with more data; the core problem DML solves.
- **Representation learning** — learning a mapping $\phi: \mathcal{X} \to \mathcal{Z}$ that
  balances treated/control distributions for causal inference.
- **Robinson decomposition** — the partialing-out transformation $\tilde{Y} = Y - m(X)$,
  $\tilde{T} = T - e(X)$ that eliminates the confounding function.
- **S-Learner** — single ML model with treatment as a feature; vulnerable to regularization
  shrinking the treatment effect.
- **TarNet (Treatment-Agnostic Representation Network)** — neural network with shared encoder and
  per-arm heads; learns balanced representations implicitly.
- **T-Learner** — two separate ML models per treatment arm; avoids treatment drowning but loses
  information sharing.
- **X-Learner** — four-stage meta-learner combining T-Learner imputation with propensity-weighted
  blending.

---

## Check yourself

1. Why is the ATE insufficient for coupon-targeting decisions, and what does CATE provide that ATE
   doesn't? Use the worked example from §1.2. *(§1)*
2. Explain the S-Learner's regularization bias failure mode. What happens to $\hat\tau(x)$ when
   the treatment variable $T$ is down-weighted by regularization? *(§3.1)*
3. How does the T-Learner avoid the S-Learner's drowning problem, and what new problem does it
   introduce? *(§3.2)*
4. Trace the X-Learner's four stages and explain why propensity weighting in Stage 4 is the
   correct way to blend the two second-stage models. *(§3.3)*
5. What is honest estimation in causal trees, and why can't you use the same data for both
   splitting and estimation? *(§4.1)*
6. Explain how TarNet's shared encoder implicitly learns a balanced representation. Why does the
   joint training objective encourage this? *(§5.2)*
7. State CFRNet's double-robustness bound and explain what happens when you increase the balance
   penalty $\lambda$ too much. *(§5.3)*
8. Derive the Robinson decomposition from $Y = \tau_0 T + g(X) + \epsilon$ and explain why
   $g(X)$ disappears. *(§7.1)*
9. What is Neyman orthogonality, and how does it change the relationship between nuisance error
   and treatment effect bias? *(§7.2)*
10. Why is cross-fitting necessary even when Neyman orthogonality is satisfied? *(§7.3)*
11. Write the R-Learner's objective function and explain why it can be interpreted as generalized
    OLS with $\tilde{T}_i^2$ as implicit weights. *(§9)*
12. Describe the DR-Learner's three stages and explain how it combines double robustness,
    orthogonality, and heterogeneous effect estimation. *(§10)*
13. Map five different causal ML methods onto the canonical DAG, specifying which edge or path each
    method targets and how. *(§11)*

---

## Going deeper

1. **Künzel, Sekhon, Bickel & Yu (2019), "Metalearners for estimating heterogeneous treatment
   effects using machine learning," PNAS** — ⚠️ **confirmed directly on the slide footer** for the
   entire meta-learner section (S/T/X-learner). `solid`. The original X-learner paper — read this
   first among the meta-learner references.

2. **Chernozhukov et al. (2018), "Double/Debiased Machine Learning for Treatment and Structural
   Parameters"** — ⚠️ **confirmed directly on the slide footer**, across the DML-framework slide
   range (e.g. the slide introducing §6's "regularization bias in nuisance functions," footer reads
   "Chernozhukov et al. (2018) · Robinson (1988) · Wager (2024)") — **not** slide 85 as an earlier
   pass of this file stated (slide 85's own footer is "Doubly-robust learner · Kennedy (2023)," the
   DR-Learner citation, see item 10 below). `hard`. The foundational DML paper; essential reading for
   understanding partialing out, Neyman orthogonality, and cross-fitting in full technical detail.

3. **Shalit, Johansson & Sontag (2017), "Estimating Individual Treatment Effect: Generalization
   Bounds and Algorithms," ICML** (companion: Johansson, Shalit & Sontag, 2016) — ⚠️ **confirmed
   directly on the slide footer** as the single paper introducing both TarNet and CFRNet. `solid`.
   The representation-learning paper for causal inference — covers both §5.2 and §5.3.

4. **Athey & Imbens (2016), "Recursive Partitioning for Heterogeneous Causal Effects"** — ⚠️
   **confirmed directly on the slide footer** for the causal-trees-and-forests section (the same
   footer that names items 5's paper). `hard`. The original causal-tree paper — §4.1's splitting
   criterion and honest-estimation idea trace directly to this work.

5. **Athey, Tibshirani & Wager (2019), "Generalized Random Forests," Annals of Statistics** — ⚠️
   **confirmed directly on the slide footer** (same causal-trees-and-forests slide as item 4) — this
   is the local-centering causal forest variant that actually achieves Neyman orthogonality, per the
   ⚠️ callout in §4.2. `hard`. (An earlier pass of this file incorrectly flagged this as "not named
   on the slides" — corrected here after direct re-verification.)

6. **Robinson (1988), "Root-N-Consistent Semiparametric Regression"** — ⚠️ **confirmed directly on
   the slide footer**, across the same DML-framework slide range as item 2 — this is the paper §7.1's
   Robinson-decomposition formulas are named for. `hard`. (An earlier pass of this file incorrectly
   flagged this as "not named on the slides" — corrected here after direct re-verification.)

7. **Neyman (1959), "Optimal Asymptotic Tests of Composite Hypotheses"** — ⚠️ **confirmed directly on
   the slide footer** for the Neyman-orthogonality slides (§7.2's source range). `hard`. The origin of
   the term "Neyman orthogonality" that §7.2 is built entirely around — worth reading for the
   original statistical idea behind the modern causal-ML usage.

8. **Wager (2024)** — ⚠️ **confirmed directly on the slide footer**, appearing across both the
   heterogeneous-treatment-effects framing (§1's source range) and the DML-framework slides (item 2's
   range) — a recent synthesis/lecture-note-style reference tying the two topics together. `solid`.

9. **Nie & Wager (2021), "Quasi-Oracle Estimation of Heterogeneous Treatment Effects," Biometrika**
   — ⚠️ **confirmed directly on the slide footer and body text** of the R-Learner slide. `hard`. The
   actual R-Learner paper — §9's objective function traces directly to this work, which had no
   citation in this section before this review.

10. **Kennedy (2023), "Towards optimal doubly robust estimation of heterogeneous causal effects"**
    — ⚠️ **confirmed directly on the slide footer** ("Doubly-robust learner · Kennedy (2023)") for
    the DR-Learner slide. `hard`. The actual DR-Learner citation — §10's pseudo-outcome formula
    traces directly to this work, which had no citation in this section before this review.

11. **Imbens & Rubin (2015), "Causal Inference for Statistics, Social, and Biomedical Sciences"**
    — ⚠️ not named on the slides — comprehensive reference for the potential outcomes framework
    underpinning the entire series. `solid`.

12. **Athey & Imbens (2019), "Machine Learning Methods That Economists Should Know About,"
    Annual Review of Economics** — ⚠️ not named on the slides — influential survey connecting ML
    methods (including causal forests) to econometric causal inference. `solid`. (Corrected from a
    stated year of 2018 — confirmed via WebSearch: published in *Annual Review of Economics*, Vol.
    11, 2019, pp. 685–725.)

> ⚠️ **verify this** — items 1–10 are confirmed directly on the lecture's own slide footers (items
> 4–5's earlier "not named on the slides" flags, and item 2's slide-85 attribution, were themselves
> errors from an earlier review pass — both corrected above after direct re-verification against the
> raw slide images). Items 11–12 are added from general causal inference field knowledge to support concepts the lecture uses
> but does not cite by author/year — treat them as suggested background reading, not transcribed
> lecture citations.
