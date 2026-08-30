---
title: "Causal Inference — Part 1: Foundations, Graphical Models, and do-Calculus"
topic: causal-inference
lecture: 21
source: "output/Lecture_21 - Module 7 Causal Inference Part 1"
slides: 53
---

# Causal Inference — Part 1: Foundations, Graphical Models, and do-Calculus

> Runtime ~53:23. Built from the raw capture in `output/Lecture_21 - Module 7 Causal Inference Part
> 1/` (53 raw frames), not `slides_deduped/` — see project memory `slides-deduped-is-lossy`. This
> deck is unusually citation-dense for the course — every major concept is traced to its original
> paper (Simpson 1951, Rubin 1974, Pearl 1988/1995/2000, Dawid 1979, Verma & Pearl 1988, Fisher
> 1935) — and this file preserves those citations rather than dropping them. Instructor not named in
> the webcam tile (same gap as some earlier lectures).

---

## What you'll understand after reading this

1. **State precisely why $P(Y\mid X) \ne P(Y\mid\text{do}(X))$**, and explain in your own words why a
   model that predicts well can still give the wrong answer to "what happens if we act?"
2. **Diagnose Simpson's paradox** in a real dataset — explain exactly how a confounder can flip the
   sign of an association between the aggregate and every subgroup.
3. **Write and interpret the Rubin potential-outcomes switching equation**, and explain why causal
   inference is fundamentally a missing-data problem.
4. **Read a DAG as a factorization of a joint distribution**, and distinguish what deleting an edge
   means (an intervention) from what conditioning on a node means (an observation).
5. **Classify any three-node substructure as a chain, fork, or collider**, and predict — correctly,
   including the counterintuitive collider case — whether conditioning on the middle node blocks or
   opens the path between the outer two.
6. **Apply the three rules of do-calculus** to reduce an interventional expression to a purely
   observational one, and recognize when that reduction is (and isn't) possible.
7. **State the backdoor and frontdoor adjustment formulas**, explain when each applies, and choose
   between them given a specific causal graph.
8. **Explain exactly what randomization buys you in an RCT**, and name the four practical limitations
   that push real problems back toward observational methods.

---

## Before we start: what you need to know

### Prerequisite 1 — Conditional probability and expectation, briefly

You need to be comfortable reading $P(Y\mid X)$ ("the probability of $Y$ given $X$") and
$\mathbb{E}[Y\mid X]$ ("the expected value of $Y$ given $X$") as *statements about a fixed,
unmanipulated population* — you are looking at whichever rows of a dataset happen to have that value
of $X$, and asking what $Y$ looks like among them. This is the single most important habit to bring
into this lecture, because the entire subject exists to explain why that quantity is not always the
one you actually want.

### Prerequisite 2 — What "i.i.d." means, and why it matters here

> **i.i.d. (independent and identically distributed)** — the standard assumption that every data
> point is drawn independently from the same fixed underlying distribution.
>
> *Why it exists:* almost every supervised learning guarantee (generalization bounds, consistency of
> the empirical risk minimizer) depends on this assumption. *Why it matters here:* the moment someone
> **acts** on a model's output (a pricing team raises prices for items predicted to sell well), the
> distribution the next batch of data is drawn from is no longer the one the model was trained on —
> the i.i.d. assumption silently breaks, and this lecture is largely about what to do once it has.

### Prerequisite 3 — Random variables and joint distributions

A **joint distribution** $P(X_1,\ldots,X_p)$ describes the probability of every combination of values
several random variables could simultaneously take. This lecture's graphical models section (§4)
is entirely about a compact way to *write down* a joint distribution using a graph — if joint
distributions are unfamiliar, know only that $P(X,Y,Z)$ is "the probability of seeing this particular
$X$, this particular $Y$, and this particular $Z$, all at once."

### Prerequisite 4 — Expectation notation, $\mathbb{E}[\cdot]$

$\mathbb{E}[Y]$ means "the average value of $Y$, weighted by how likely each value is." $\mathbb{E}[Y
\mid T{=}1]$ means "the average value of $Y$, computed only over the subpopulation where $T=1$." This
notation appears in nearly every formula in this lecture (the ATE, the do-calculus rules, the RCT
formula) and always means exactly this.

---

## The big picture

This lecture answers one question with three progressively more powerful tools: **given data where
you only ever *observed* what happened, how do you answer a question about what *would* happen if
you *acted*?**

```
Prediction ("what will Y be?")          Causal question ("what if we set X = x?")
        │                                          │
        │ P(Y | X)  ≠  P(Y | do(X))  — these are DIFFERENT quantities, in general
        │
        ▼
Why they differ: confounding — a common cause Z of both X and Y creates an association
between X and Y even when X has no effect on Y at all (Simpson's paradox, §2)
        │
        ▼
Two formal languages for the same problem:
        │
        ├──▶ Potential Outcomes (Rubin) — define τ = Y(1) - Y(0) per unit; the
        │    "fundamental problem" is that you only ever observe ONE of Y(1), Y(0) (§3)
        │
        └──▶ Structural Causal Models / DAGs (Pearl) — encode causal structure as a
             graph; P(Y | do(X)) means "delete X's incoming edges, then read off Y" (§4)
                        │
                        ▼
             Conditional independence in a graph — chains, forks, colliders (§5)
                        │
                        ▼
             d-separation — a complete, mechanical procedure for reading every
             conditional independence directly off the graph (§6)
                        │
                        ▼
             do-calculus — three rules that let you REDUCE a do(·) expression to a
             plain observational one, when the graph permits it (§7)
                        │
                        ▼
             Backdoor / frontdoor adjustment — the two most common, ready-made
             recipes built from those three rules (§8)
                        │
                        ▼
             RCTs — sidestep the entire identification problem by randomizing,
             but at a real cost in money, ethics, and generalizability (§9)
```

Every later technique in this lecture is a way of answering the same question — *"what is
$P(Y\mid\text{do}(X))$?"* — with progressively fewer assumptions about what you're allowed to
manipulate directly.

---

## 1. Why Causality Matters

### 1.1 The central distinction

> *"Observing $P(Y\mid X)$ is not the same as intervening $P(Y\mid\text{do}(X))$."*

$$P(Y\mid X) \;\ne\; P(Y\mid\text{do}(X))$$

*Left: conditioning on observed data. Right: forcing $X$ to a value.*

- **Prediction** optimizes $\hat y$; breaks when we change the system.
- **Causal question:** *"if we **set** $X=x$, how does $Y$ change?"*
- 🧪 **Example, as given:** A model finds patients on Drug A have lower readmission. But sicker
  patients were prescribed Drug B. The correlation is confounded by severity.

> **do-operator, $\text{do}(X{=}x)$** — notation for a *forced* intervention: every unit in the
> population is set to $X=x$, overriding whatever would normally have determined $X$, and everything
> downstream is then observed.
>
> *In everyday words:* $P(Y\mid X{=}x)$ asks "among people who happened to have $X=x$, what does $Y$
> look like?" $P(Y\mid\text{do}(X{=}x))$ asks "if I reached in and *set* everyone's $X$ to $x$
> (regardless of what they would have chosen), what would $Y$ look like?" These can give completely
> different numbers whenever something else influences *both* who ends up with $X=x$ *and* what $Y$
> turns out to be.
>
> *Why it exists:* it is the formal notation that separates "what we happened to see" from "what
> would happen if we acted," and — as the rest of this lecture shows — that gap is exactly where a
> **confounder** lives.

### 1.2 Three paradigms, contrasted directly

**Supervised learning:** Learn $f:\mathcal{X}\to\mathcal{Y}$ minimizing expected loss under
$P_{\text{data}}$.

$$\hat f = \underset{f\in\mathcal{F}}{\arg\min}\ \mathbb{E}_{(X,Y)\sim P}\big[\ell(f(X),Y)\big]$$

- Assumes data is i.i.d. from a fixed distribution $P$.
- Optimizes a single objective (accuracy, AUC, MSE).
- **No notion of actions, interventions, or counterfactuals.**
- **Breaks when $P$ shifts due to deployment decisions** (covariate shift, concept drift).
- 🧪 **Example:** A demand forecast predicts sales accurately. But when the pricing team **acts** on
  it (raises price for high-demand items), the input distribution changes and the forecast degrades.

**Causal inference:** Estimate effects of interventions from observational (or experimental) data.

$$\tau = \mathbb{E}[Y(1)-Y(0)] = \mathbb{E}[Y\mid\text{do}(T{=}1)] - \mathbb{E}[Y\mid\text{do}(T{=}0)]$$

- Target is a **treatment effect**, not a prediction.
- Requires **identification**: assumptions that link the causal quantity to a statistical one that
  can actually be computed from data.
- **Invariant under intervention by construction** — this is the entire point: a correctly identified
  causal effect does not degrade the way the demand forecast above did.
- **Two frameworks:** potential outcomes (Rubin) and structural causal models (Pearl) — the two this
  lecture builds, in §3 and §4 respectively.
- 🧪 **Example:** Does a new blood-pressure medication reduce stroke risk? Estimate $\tau$ from an RCT
  or, when randomization is infeasible, from observational EHR (electronic health record) data with
  appropriate adjustment.

**Reinforcement learning:** Learn a policy $\pi$ that maximizes cumulative reward through sequential
decisions.

$$\pi^* = \underset{\pi}{\arg\max}\ \mathbb{E}_\pi\!\left[\sum_{t=0}^{T}\gamma^t R_t\right]$$

- Agent interacts with an environment over time.
- Actions affect future states (sequential, non-i.i.d.).
- Exploration/exploitation trade-off.
- **Uses causal structure implicitly** (actions cause state transitions).
- 🧪 **Example:** An RL agent decides daily replenishment quantities for a warehouse. Each decision
  affects future inventory and costs. **Off-policy evaluation (a causal problem)** is needed before
  deploying a new policy.

### 1.3 Reconciliation — same data, different questions

| | **Supervised ML** | **Causal Inference** | **RL** |
|---|---|---|---|
| Question | What will $Y$ be? | What if we set $T$? | Which action sequence? |
| Target | $\mathbb{E}[Y\mid X]$ | $\mathbb{E}[Y\mid\text{do}(T)]$ | $V^\pi = \mathbb{E}_\pi[\sum\gamma^tR_t]$ |
| Data | i.i.d. samples | Obs. or experimental | Trajectories |
| Key challenge | Generalization | **Identification** | Exploration |

- **CI appears inside RL** (off-policy evaluation) **and inside ML** (domain adaptation, fairness).
- **All three share estimation machinery:** IPW (inverse propensity weighting), doubly-robust
  estimators, semiparametric efficiency.

> 💡 **Key insight — this table is the thesis of the whole "motivation" section.** These three fields
> are not separate disciplines that happen to share notation; they are three different *questions*
> asked of related kinds of data, and causal inference's central machinery (identification,
> confounding adjustment) shows up as a genuine subroutine inside both of the other two — off-policy
> evaluation in RL and domain-shift/fairness analysis in supervised ML are, underneath, causal
> inference problems in disguise.

### Knowledge check — correlation or causation?

The lecture poses three statements as a self-check, without revealing worked answers on-screen — work
through each using the definitions above before reading on:

1. *"Countries spending more on healthcare have higher life expectancy."*
2. *"In an RCT, patients receiving the new antibiotic cleared infection 2 days faster."*
3. *"Factories that adopted predictive maintenance had 15% fewer breakdowns last year."*

> 🎯 **Working through it.** Statement 2 is the only one built from *randomization* — an RCT
> guarantees $T\perp\{Y(0),Y(1)\}$ (§9), so the observed difference is a valid causal effect: the new
> antibiotic *causes* faster clearance. Statements 1 and 3 are both **observational** associations
> with an obvious confounder available: wealthier countries can afford both more healthcare spending
> *and* have other life-expectancy-improving factors (diet, sanitation, education) — spending and life
> expectancy could easily share a common cause without spending having a large direct effect.
> Similarly, factories that *adopt* predictive maintenance may systematically differ (better-funded,
> better-managed, newer equipment) from factories that don't — the 15% figure is a correlation between
> **adopting the practice** and **fewer breakdowns**, not necessarily a measurement of the practice's
> isolated causal effect. Both 1 and 3 need the adjustment machinery in §7–§8 (or an RCT) before their
> numbers can be read causally.

---

## 2. Correlation ≠ Causation

> *"A confounder $Z$ induces $\rho_{XY}\ne0$ even when $X\not\to Y$."*

### 2.1 Spurious correlation — firefighters vs. property damage

A scatter plot of firefighters deployed vs. property damage shows a strong, clean positive
relationship. *"Larger fires cause both more firefighters and more damage. Sending fewer firefighters
would **increase** damage."* The confounder here is **fire severity** — it drives both axes of the
plot simultaneously, producing a strong correlation between two quantities where intervening on one
(sending fewer firefighters) would not reduce, and would likely worsen, the other.

### 2.2 Simpson's Paradox — hospital recovery

> **Simpson's paradox** — a pattern where an association observed in the aggregate population
> **reverses** when the same population is broken into subgroups, or vice versa. Named for Simpson
> (1951), *"The Interpretation of Interaction in Contingency Tables."*

The lecture's example, hospital recovery rates:

| | **Aggregate** | **Mild cases** | **Severe cases** |
|---|---|---|---|
| Hospital A | **70%** | 93% | 73% |
| Hospital B | **80%** | 87% | 69% |

*"Hospital A treats more severe patients."*

> 🧪 **Working through the *qualitative* mechanism (an illustrative reconstruction — the slide shows
> only the percentages above, no raw counts).** Suppose Hospital A sees mostly severe cases (say, 800
> severe patients and 200 mild) while Hospital B sees mostly mild cases (say, 200 severe and 800
> mild). If Hospital A had the **higher** recovery rate in *both* the mild subgroup and the severe
> subgroup, its aggregate rate could still come out **lower** than Hospital B's simply because
> Hospital A's patient mix is weighted much more heavily toward the harder-to-treat severe category,
> pulling its blended average down further than Hospital B's easier mix pulls Hospital B's average
> down. This is the general *mechanism* Simpson's paradox relies on — a confounder (case severity)
> correlated with both which hospital a patient ends up at and the outcome.
>
> ⚠️ **verify this — the slide's own numbers are themselves arithmetically inconsistent, not just an
> unreconstructed teaching device.** An aggregate rate must always lie *between* its subgroup rates
> (it is a weighted average of them). But `slide_032.jpg`'s own aggregate figure for Hospital A (70%)
> sits *below both* of Hospital A's own subgroup rates (93% mild, 73% severe) — no patient-count split
> whatsoever could reproduce a 70% aggregate from subgroups of 93% and 73% (the true weighted average
> would always be somewhere between 73% and 93%). The 800/200 counts above were chosen only to
> illustrate the *direction* of the mechanism (a severity-skewed mix can pull an aggregate below where
> naive intuition expects) — plugging them into the slide's own subgroup rates actually yields ≈77%
> (mild-heavy weighting would be needed for ≈83%), not the slide's stated 70%, confirming the
> inconsistency is in the source slide itself, not an error introduced here. Treat the qualitative
> lesson (case-mix confounding can reverse an aggregate comparison) as sound, but do not treat the
> specific "70%" as arithmetically derivable from 73%/93% — it isn't, on any patient split.

> 💡 **Key insight.** Case severity is the confounder $Z$: it affects *both* which hospital a patient
> is more likely to end up at (severe cases may be preferentially referred to Hospital A, a more
> specialized center) *and* the outcome $Y$ directly (severe cases are simply harder to treat,
> regardless of hospital). Comparing raw aggregate recovery rates conflates "which hospital is
> better" with "which hospital sees harder cases" — exactly the trap Simpson's paradox names. The fix
> is to compare **within** severity strata (which the mild/severe breakdown already does), or to use
> the adjustment formulas in §8.

> ⚠️ **Where people get confused.** Simpson's paradox is not a statistical anomaly or a rare edge
> case — it is the *direct, generic consequence* of a confounder that is correlated with the exposure
> at unequal rates across subgroups. Any time group sizes and outcome rates both vary across strata,
> the aggregate can reverse the within-stratum pattern. The lesson is structural: **always ask what
> variable determines both who gets the treatment/exposure and what the outcome tends to be**, before
> trusting an aggregate comparison.

---

## 3. Potential Outcomes Framework

> **Rubin causal model** — formalizes counterfactuals: every unit $i$ has *two* potential outcomes,
> $Y_i(1)$ (what would happen if treated) and $Y_i(0)$ (what would happen if not treated), only one of
> which is ever actually observed. Rubin (1974), *"Estimating causal effects of treatments in
> randomized and nonrandomized studies."*

### 3.1 The switching equation

$$Y_i = T_i\cdot Y_i(1) + (1-T_i)\cdot Y_i(0)$$

*Switching equation: observed outcome depends on assignment $T_i\in\{0,1\}$.*

**Words before symbols:** the outcome you actually *see* for unit $i$ is just whichever potential
outcome corresponds to the treatment that unit actually received — if $T_i=1$ (treated), you see
$Y_i(1)$; if $T_i=0$ (untreated), you see $Y_i(0)$. The equation is a bookkeeping device: it says
"the observed $Y$ is a *selector* that reveals one of the two potential outcomes and hides the
other," nothing more.

### 3.2 The individual treatment effect and the fundamental problem

$$\tau_i = Y_i(1) - Y_i(0) \quad\text{(never jointly observed)}$$

> ⚠️ **The fundamental problem of causal inference, stated directly.** *"Only one potential outcome
> per unit is observed. Causal inference is a missing-data problem."* You can never compute $\tau_i$
> for any single unit $i$, because you never see both $Y_i(1)$ and $Y_i(0)$ for the same person at the
> same time — the other one is, by construction, a counterfactual that did not happen. Every method in
> this lecture is a strategy for estimating some *aggregate* of $\tau_i$ across many units, without
> ever needing any single unit's individual $\tau_i$.

### 3.3 Three aggregate estimands

$$\text{ATE} = \mathbb{E}[\tau_i] \qquad \text{ATT} = \mathbb{E}[\tau_i\mid T_i{=}1] \qquad \text{CATE}(x) = \mathbb{E}[\tau_i\mid X_i{=}x]$$

| Symbol | Read it as | What it means |
|---|---|---|
| **ATE** | "average treatment effect" | The average effect across the *whole* population, treated and untreated alike |
| **ATT** | "average treatment effect on the treated" | The average effect, but only among units that *actually* received treatment |
| **CATE($x$)** | "conditional average treatment effect" | The average effect *within* the subgroup defined by covariate value $X{=}x$ — a personalized or subgroup-specific effect |

> 🎯 **Interview framing — why three, not one.** ATE answers "should this treatment be rolled out to
> everyone?" ATT answers "for the people who got it, did it actually help?" (useful when treatment
> assignment wasn't random, so the treated group may differ systematically from the general
> population). CATE answers "for a person with *these specific characteristics*, how much would this
> help?" — the estimand behind personalized/precision interventions. They can differ substantially
> whenever treatment effects vary across the population and treatment assignment correlates with those
> same characteristics.

---

## 4. Probabilistic Graphical Models

> **DAG (Directed Acyclic Graph), $\mathcal{G}=(V,E)$** — encodes a factorization of the joint
> distribution. Pearl (1988), *Probabilistic Reasoning in Intelligent Systems*, Morgan Kaufmann.

$$P(X_1,\ldots,X_p) = \prod_{j=1}^{p} P\big(X_j \mid \text{Pa}(X_j)\big)$$

*Markov property: each node is conditionally independent of its non-descendants given its parents.*

**Words before symbols:** the probability of the whole system taking on a particular joint
configuration factors into a product of small, local pieces — one factor per variable, each factor
depending only on that variable's own **parents** (its direct causes in the graph) rather than on the
entire rest of the system. This is what makes a graph a compact, checkable representation of a
complicated joint distribution.

🧪 **Worked example — a 3-node graph** [Z → X, Z → Y, X → Y]:

$$P(Z,X,Y) = P(Z)\,P(X\mid Z)\,P(Y\mid X,Z)$$

- **Nodes** = random variables.
- **Directed edges** = direct causal influence.
- **Acyclicity** allows topological ordering (you can always list the variables so every arrow points
  from earlier to later in the list — this is what makes the factorization well-defined).
- **Causal semantics:** **deleting edges = intervening.** This is the single sentence that connects
  graphs back to §1's do-operator: computing $P(Y\mid\text{do}(X{=}x))$ means *literally erasing every
  arrow pointing into $X$* (since an intervention overrides whatever normally determines $X$), fixing
  $X=x$, and then reading off the resulting (now-modified) joint distribution over everything else.

---

## 5. Conditional Independence — the Atomic Building Block

$$X\perp Y \mid Z \quad\Longleftrightarrow\quad P(X,Y\mid Z) = P(X\mid Z)\,P(Y\mid Z)$$

Dawid (1979), *"Conditional Independence in Statistical Theory."*

There are exactly three shapes a path between two variables can take through a third, and they behave
**oppositely** with respect to conditioning:

| | **Fork (confounder)** | **Chain (mediator)** | **Collider** |
|---|---|---|---|
| Diagram | $Z\to X,\ Z\to Y$ | $X\to M\to Y$ | $X\to C \leftarrow Y$ |
| Marginally | $X\not\perp Y$ | $X\not\perp Y$ | $X\perp Y$ |
| Given the middle node | $X\perp Y \mid Z$ | $X\perp Y \mid M$ | $X\not\perp Y \mid C$ |
| Effect of conditioning | **Conditioning blocks path** | **Conditioning blocks path** | **Conditioning opens path** |

- **Conditioning can create or destroy associations.**
- **Collider bias (Berkson's paradox):** *"among selected units, causes of selection appear
  negatively correlated."*

> ⚠️ **The single most counterintuitive fact in this lecture, and worth sitting with.** A **fork**
> ($Z$ causes both $X$ and $Y$) and a **chain** ($X$ causes $M$ causes $Y$) both behave the way
> intuition expects: $X$ and $Y$ start out associated, and conditioning on the variable in the middle
> removes that association (in the fork's case, because you've controlled for the shared cause; in
> the chain's case, because you've blocked the only causal pathway from $X$ to $Y$). A **collider**
> does the *opposite*: $X$ and $Y$ start out **independent** — genuinely unrelated — but conditioning
> on their common effect $C$ *creates* an association between them that did not exist before.
>
> *Concretely — a classic collider-bias example (celebrity attractiveness and talent):* suppose being
> famous ($C$) requires either being unusually attractive ($X$) or unusually talented ($Y$), and
> attractiveness and talent are genuinely unrelated in the general population. Restrict attention only
> to famous people (i.e., condition on $C$) and attractiveness and talent will appear *negatively*
> correlated — because among the famous, a person low on one trait most likely made it in on the
> other. This is Berkson's paradox exactly as named on the slide: *conditioning on a shared effect
> manufactures a spurious negative correlation between two otherwise independent causes.*
>
> **How to apply:** before conditioning on (or "controlling for") any variable in an analysis, check
> whether it's a collider on the causal path you care about. Controlling for a collider is a common,
> genuine mistake — it does not remove confounding, it *introduces* a new spurious association where
> none existed.

---

## 6. D-Separation Theorem — Reading Conditional Independences from the Graph

> *"Reading conditional independences from the graph."* Verma & Pearl (1988), *"Causal Networks:
> Semantics and Expressiveness."*

**Definition.** A path between $X$ and $Y$ is **blocked** by $Z$ if it contains: **(i)** a
non-collider in $Z$, or **(ii)** a collider whose descendants are all outside $Z$.

$$X \perp_{\mathcal{G}} Y \mid Z \quad\Longrightarrow\quad X\perp Y\mid Z \quad\text{(under faithfulness)}$$

- **d-separation is sound and complete** for reading off conditional independencies from the graph.
- **Faithfulness:** no exact parameter cancellations (a generic assumption — it rules out the
  freak coincidence where two causal effects happen to numerically cancel each other out exactly,
  producing an independence in the data that the graph itself doesn't predict).
- **Practical use:** determine the **minimal adjustment set** for identification.

> 💡 **Key insight — this generalizes §5's three cases into one rule.** D-separation is the single
> algorithm that correctly handles *any* path of *any* length, built out of chains, forks, and
> colliders in any combination: walk every path between $X$ and $Y$; a path is blocked if it hits a
> non-collider that's in the conditioning set $Z$ (fork or chain logic — conditioning on it blocks the
> path) *or* if it hits a collider that's *not* in $Z$ and has no descendant in $Z$ either (collider
> logic — an unconditioned collider already blocks the path on its own). If **every** path between $X$
> and $Y$ is blocked given $Z$, the graph implies $X\perp Y\mid Z$. This single mechanical procedure
> is "sound and complete" — it never gives a wrong answer, and it finds every independence the graph
> actually implies.

---

## 7. do-Calculus — From Graph to Intervention

> *"Three rules that reduce $\text{do}(\cdot)$ expressions to observational quantities."* Pearl
> (1995), *"Causal Diagrams for Empirical Research."*

$$P(Y\mid\text{do}(X),Z) \;\longrightarrow\; \text{observational expression (if identifiable)}$$

**Rule 1 (insertion/deletion of observations):**
$$P(Y\mid\text{do}(X),Z,W) = P(Y\mid\text{do}(X),W) \quad\text{if}\quad Y\perp_{\mathcal{G}_{\bar X}} Z \mid X,W$$

**Rule 2 (action/observation exchange):**
$$P(Y\mid\text{do}(X),\text{do}(Z),W) = P(Y\mid\text{do}(X),Z,W) \quad\text{if}\quad Y\perp_{\mathcal{G}_{\bar X\underline Z}} Z\mid X,W$$

**Rule 3 (insertion/deletion of actions):**
$$P(Y\mid\text{do}(X),\text{do}(Z),W) = P(Y\mid\text{do}(X),W) \quad\text{if}\quad Y\perp_{\mathcal{G}_{\bar X\overline{Z(W)}}} Z\mid X,W$$

**Words before symbols, for each rule:**

- **Rule 1** says: an *observed* variable $Z$ can be dropped from (or added to) what you're
  conditioning on, without changing the answer, exactly when $Z$ gives no extra information about $Y$
  once you already know $X$ and $W$ — checked by d-separation in the graph with $X$'s incoming edges
  deleted (written $\mathcal{G}_{\bar X}$, "the graph as it looks after intervening on $X$").
- **Rule 2** says: an *action* $\text{do}(Z)$ can be replaced by simply *observing* $Z{=}z$ — turning
  an intervention into a plain conditioning statement — whenever $Z$'s only relevant relationship with
  $Y$ runs entirely through observable paths, once $X$ is intervened on.
- **Rule 3** says: an action $\text{do}(Z)$ can be dropped entirely (as if it were never performed)
  whenever forcing $Z$ wouldn't have changed anything relevant to $Y$ anyway, given $X$ and $W$.

> 💡 **Key insight — what these three rules are *for*.** Individually the notation is dense, but the
> collective purpose is simple: **do-calculus is a proof system for turning an unmeasurable quantity
> (something with a $\text{do}(\cdot)$ in it) into a measurable one (a plain conditional probability,
> computable from observational data), one legal graph-surgery step at a time.** If a sequence of
> Rule 1/2/3 applications can fully eliminate every $\text{do}(\cdot)$ from an expression, the causal
> effect is **identifiable** from observational data; Pearl proved these three rules are complete —
> if an effect *is* identifiable, some sequence of these rules will find it.

---

## 8. Backdoor and Frontdoor Criteria — Two Ready-Made Corollaries

> *"Two commonly used corollaries of the do-calculus."* Pearl (2000), *Causality*, Ch. 3; Huang &
> Valtorta (2006), *"Identifiability in Causal Bayesian Networks."*

### 8.1 Backdoor adjustment

$$P(Y\mid\text{do}(X)) = \sum_{z} P(Y\mid X, Z{=}z)\,P(Z{=}z)$$

*Valid when $Z$ blocks all backdoor paths and contains no descendant of $X$.*

> **Backdoor path** — any path between $X$ and $Y$ that starts with an arrow *into* $X$ (i.e., a
> path running "backward" out of $X$ through a common cause, rather than forward along $X$'s own
> causal effect). These are exactly the paths that create confounding.
>
> **Words before symbols:** if you can find a set of variables $Z$ that blocks every backdoor path
> from $X$ to $Y$ (and doesn't itself get affected by $X$), then you can compute the causal effect by
> a **weighted average**: look at $Y$'s distribution within each stratum of $Z$ (that's the
> "adjustment" step — comparing like-with-like within each stratum, exactly as §2.2's Simpson's
> paradox fix required), then average those stratum-specific numbers back together, weighting by how
> common each stratum is. This is the formal generalization of "compare recovery rates *within* each
> severity stratum, then combine."

### 8.2 Frontdoor adjustment

$$P(Y\mid\text{do}(X)) = \sum_m P(m\mid X)\sum_{x'} P(Y\mid m,x')\,P(x')$$

*Valid when mediator $M$ satisfies: $X\to M\to Y$, no direct $X\to Y$ path unblocked by $M$.*

- **Backdoor: most common in practice** (adjust for confounders).
- **Frontdoor: useful when confounders are unobserved but a mediator is measured.**
- **Both are derivable via the three rules of do-calculus** — they are not separate assumptions
  bolted on to the theory; they are specific, named *consequences* of §7's three rules, packaged as
  reusable formulas because they come up so often in practice.

> 🎯 **Interview framing — when do you reach for which?** Reach for **backdoor adjustment** whenever
> you can plausibly list and measure the common causes of treatment and outcome (the usual
> "control for confounders" instinct in applied statistics). Reach for **frontdoor adjustment**
> specifically in the harder case where the confounder between $X$ and $Y$ is *unmeasured or
> unmeasurable* (e.g., an unobserved genetic predisposition), but you happen to have measured a
> **mediator** — a variable that fully carries $X$'s effect on $Y$ — whose own relationship to $Y$ is
> *not* confounded by that same unmeasured variable. This is a substantially less common situation in
> practice, which is exactly why the slide calls backdoor "most common" and frontdoor "useful when."

---

## 9. RCTs — The Gold Standard

> *"Randomisation guarantees $T\perp\{Y(0),Y(1)\}$."* Fisher (1935), *The Design of Experiments*,
> Oliver & Boyd.

$$\text{ATE} = \mathbb{E}[Y\mid T{=}1] - \mathbb{E}[Y\mid T{=}0]$$

*Unbiased by design: no confounders (observed **or unobserved**).*

> 💡 **Key insight — why randomization is uniquely powerful.** Every technique in §7–§8
> (do-calculus, backdoor, frontdoor) works by **adjusting for confounders you can identify and
> measure**. Randomization sidesteps the entire problem differently: by assigning $T$ purely by chance
> (a coin flip, independent of everything about the unit), it guarantees that treatment assignment is
> statistically unrelated to *both* potential outcomes — **including confounders you never thought to
> measure, or couldn't measure even in principle.** This is the one thing no amount of clever
> observational adjustment can promise, because adjustment can only control for variables you actually
> have in your dataset.

Four practical limitations, each of which pushes real-world problems back toward observational
methods:

- **Cost:** experiments are expensive or slow.
- **Ethics:** cannot randomize harmful exposures (you cannot ethically randomize people to smoke, to
  receive a known-inferior treatment, or to be denied a beneficial intervention).
- **External validity:** trial population $\ne$ deployment population — an effect measured in a
  controlled trial population may not transfer cleanly to the full population it will eventually be
  deployed to.
- **Compliance:** intention-to-treat $\ne$ per-protocol effect — the effect of *being assigned* to
  treatment (what randomization directly guarantees) is not the same quantity as the effect of
  *actually receiving* treatment, once some assigned participants don't comply.

> *"When randomisation is infeasible, we need observational methods + identification assumptions"* —
> the explicit handoff back to §4–§8's machinery, and the sentence that frames the entire lecture as
> one coherent argument: **RCTs are the gold standard exactly because they need none of the
> assumptions §4–§8 require, but the moment an RCT isn't possible, that assumption-laden machinery is
> the only path back to a valid causal answer.**

---

## Putting it together

```
                    ┌──────────────────────────────┐
                    │  P(Y|X) ≠ P(Y|do(X))           │
                    │  the problem this whole         │
                    │  lecture exists to solve         │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                                ▼
        ┌─────────────────────┐         ┌─────────────────────────┐
        │ Potential Outcomes    │         │ Structural Causal Models │
        │ (Rubin, 1974)          │         │ (Pearl, DAGs)             │
        │ τ = Y(1) - Y(0)        │         │ do(X) = delete X's        │
        │ fundamental problem:   │         │ incoming edges            │
        │ never jointly observed │         └────────────┬─────────────┘
        └───────────┬─────────────┘                      │
                    │                       ┌─────────────┴─────────────┐
                    │                       ▼                           ▼
                    │           Conditional independence      d-separation: the
                    │           (chain/fork/collider) —       complete, mechanical
                    │           §5's atomic building block     rule built from §5
                    │                       │                           │
                    │                       └─────────────┬─────────────┘
                    │                                     ▼
                    │                          do-calculus's three rules:
                    │                          reduce do(·) to observational
                    │                          quantities, when possible
                    │                                     │
                    │                       ┌─────────────┴─────────────┐
                    │                       ▼                           ▼
                    │              Backdoor adjustment          Frontdoor adjustment
                    │              (measured confounders)       (unmeasured confounders,
                    │                                            measured mediator)
                    │                                     │
                    └─────────────────────┬───────────────┘
                                          ▼
                          RCTs: sidestep identification entirely by
                          randomizing — but cost/ethics/generalizability
                          push most real problems back to the machinery above
```

Three threads run through this lecture:

1. **Every technique here is answering the same single question** — "how do I compute $P(Y\mid
   \text{do}(X))$ when I can't literally perform the intervention?" — from increasingly specific
   angles: potential outcomes names the target quantity and its fundamental unobservability; DAGs give
   a language for *when* observational data is enough; do-calculus gives the mechanical procedure;
   backdoor/frontdoor package the two most common special cases; RCTs are the one setting where the
   question never needs asking, because you *can* perform the intervention directly.
2. **Confounding is the single villain, and it takes exactly three graphical shapes** (§5): a fork
   (the classic confounder), and — much less intuitively — conditioning on a collider can
   *manufacture* the same kind of spurious association a fork creates naturally. Half of applied
   causal inference is correctly identifying which shape you're looking at before you decide whether
   to control for a variable or leave it alone.
3. **"Identifiability" is the load-bearing word of the entire lecture.** A causal effect being
   well-defined (as a quantity, via potential outcomes or the do-operator) is a completely separate
   question from whether it is *identifiable* — expressible in terms of quantities you can actually
   estimate from the data you have. Do-calculus, backdoor, and frontdoor are all, at bottom,
   identification results; RCTs matter because they make identification trivial by construction.

---

## Interview prep — Amazon Applied Scientist

### Core questions

<details><summary><b>1. In one sentence, why is P(Y|X) not the same quantity as P(Y|do(X))?</b></summary>

$P(Y\mid X)$ conditions on units that *happened* to have a particular value of $X$, which can be
correlated with other factors (confounders) that also affect $Y$; $P(Y\mid\text{do}(X))$ describes
what happens if $X$ is *forced* to that value independent of whatever normally determines it, cutting
off any confounding influence — the two coincide only when $X$ is not confounded with $Y$ (e.g.,
under randomization).
</details>

<details><summary><b>2. Explain how Simpson's paradox can make Hospital A look worse in aggregate while being better in every subgroup.</b></summary>

If Hospital A treats a much higher proportion of severe (harder-to-treat, lower-recovery) cases than
Hospital B, its aggregate recovery rate is a weighted average pulled down more heavily by that harder
case mix — even if Hospital A outperforms Hospital B within *both* the mild and severe subgroups
individually. Case severity is a confounder affecting both which hospital treats a patient and the
probability of recovery, and the aggregate comparison fails to account for the unequal case mix.
</details>

<details><summary><b>3. What is the "fundamental problem of causal inference," and why does it make τ_i uncomputable for a single unit?</b></summary>

Each unit has two potential outcomes, $Y_i(1)$ and $Y_i(0)$, but the switching equation guarantees
only one is ever observed, depending on which treatment the unit actually received — the other is a
counterfactual that never happened. Since $\tau_i = Y_i(1) - Y_i(0)$ requires both values for the same
unit, it can never be directly computed; every practical estimator instead targets an *aggregate*
(ATE, ATT, or CATE) across many units, not any individual unit's exact effect.
</details>

<details><summary><b>4. What does "deleting an edge" in a causal DAG correspond to, and why is this the graphical definition of intervention?</b></summary>

Deleting the edges pointing *into* a node $X$ removes every causal influence that would normally
determine $X$'s value, then fixes $X$ at the intervened value — exactly capturing the idea of forcing
$X=x$ regardless of what would ordinarily cause it. This graph-surgery operation is the formal,
graphical counterpart of the do-operator: $P(Y\mid\text{do}(X{=}x))$ is read off the graph after this
edge deletion, whereas $P(Y\mid X{=}x)$ is read off the original, unmodified graph.
</details>

<details><summary><b>5. Classify X→M→Y and X←Z→Y by whether conditioning on the middle node blocks or opens the path, and explain the collider case, which behaves oppositely.</b></summary>

In both the chain ($X\to M\to Y$) and the fork ($X\leftarrow Z\to Y$), $X$ and $Y$ start out
associated, and conditioning on the middle node (M or Z) **blocks** that association — in the chain
because you've cut the only causal pathway, in the fork because you've controlled for the shared
cause. A collider ($X\to C\leftarrow Y$) behaves oppositely: $X$ and $Y$ start out independent, and
conditioning on $C$ (their common effect) **opens** a spurious association between them that did not
exist unconditionally — this is collider bias / Berkson's paradox.
</details>

<details><summary><b>6. State d-separation's blocking rule and explain why it needs two different cases (collider vs. non-collider).</b></summary>

A path is blocked by conditioning set $Z$ if it contains either (i) a non-collider that is *in* $Z$,
or (ii) a collider whose descendants are all *outside* $Z$. Two cases are needed because non-colliders
(chain/fork middle nodes) and colliders behave oppositely under conditioning (per Q5) — a single
uniform rule would get one of the two cases backward; the two-part definition correctly captures both
behaviors within one mechanical procedure.
</details>

<details><summary><b>7. What does it mean for a causal effect to be "identifiable," and how does do-calculus relate to that definition?</b></summary>

A causal effect $P(Y\mid\text{do}(X))$ is identifiable if it can be re-expressed purely in terms of
the *observational* joint distribution $P(V)$ over the measured variables — i.e., computed from data
you could actually collect, without ever needing to perform the intervention. Do-calculus's three
rules are a proof system for this: if some sequence of Rule 1/2/3 applications eliminates every
$\text{do}(\cdot)$ from the expression, the effect is identifiable, and the rules are provably complete
(they will find such a sequence whenever one exists).
</details>

<details><summary><b>8. When would you use frontdoor adjustment instead of backdoor adjustment?</b></summary>

Backdoor adjustment requires measuring and conditioning on a set of variables that blocks every
backdoor (confounding) path between treatment and outcome — it fails if a relevant confounder is
unobserved. Frontdoor adjustment is the fallback for exactly that failure case: it requires instead a
measured *mediator* through which the entire treatment effect flows, whose own relationship to the
outcome is not confounded by the same unmeasured variable — letting you identify the effect without
ever needing to measure the confounder directly.
</details>

<details><summary><b>9. [Combines concepts] Why does an RCT not need any of the identification machinery in §4–§8?</b></summary>

Randomization makes treatment assignment $T$ statistically independent of both potential outcomes,
$T\perp\{Y(0),Y(1)\}$, by construction — including confounders that were never measured or even
identified as relevant. Backdoor/frontdoor adjustment and do-calculus all exist specifically to handle
confounding when treatment assignment is *not* independent of potential outcomes; since randomization
guarantees that independence directly, the raw difference in group means, $\mathbb{E}[Y\mid T{=}1] -
\mathbb{E}[Y\mid T{=}0]$, is already an unbiased estimate of the ATE with no adjustment required.
</details>

<details><summary><b>10. [Combines concepts] A colleague wants to "control for" every available variable in a regression to be safe. Using the collider concept, explain what could go wrong.</b></summary>

If any of those variables is a collider on a causal path between the treatment and outcome of
interest — a common *effect* of both, rather than a common cause — conditioning on it does not remove
confounding but *introduces* a new spurious association where none existed (collider bias / Berkson's
paradox), potentially biasing the estimated effect in either direction. "Control for everything
available" is not a safe default; each variable needs to be classified by its position in the causal
graph (confounder to adjust for vs. mediator or collider to leave alone) before deciding whether
conditioning on it helps or hurts identification.
</details>

### Depth probes

- *"Backdoor adjustment requires Z to block all backdoor paths AND contain no descendant of X — why
  is the second condition necessary?"* — conditioning on a descendant of $X$ can partially or fully
  block $X$'s own *causal* effect on $Y$ if that descendant lies on the causal pathway (effectively
  turning part of the causal chain into a controlled-for mediator), which would bias the estimate of
  $X$'s total effect downward, or introduce collider bias if the descendant is also a collider with
  some other cause of $Y$.
- *"Why is faithfulness described as a 'generic' assumption rather than something that's always
  true?"* — faithfulness fails only when causal effects happen to cancel exactly (e.g., a direct
  positive effect and an indirect negative effect through a mediator that sum to precisely zero) —
  a measure-zero coincidence in the space of possible parameter values, which is why it's treated as
  a reasonable default assumption rather than a guaranteed property of every real system.
- *"RCTs guarantee internal validity — what's a concrete mechanism by which external validity can
  still fail even with perfect randomization?"* — the trial population itself may be recruited from a
  narrower or systematically different group than the eventual deployment population (e.g., clinical
  trial volunteers skew healthier or more compliant than the general patient population), so the
  measured ATE, though unbiased *for that trial population*, need not transfer to a different target
  population — this is a population-mismatch problem randomization does nothing to solve.

### Whiteboard-ready derivations

1. **The switching equation and why $\tau_i$ is unobservable** — §3.1–3.2: write
   $Y_i=T_iY_i(1)+(1-T_i)Y_i(0)$, then note that observing $T_i=1$ reveals $Y_i(1)$ but leaves
   $Y_i(0)$ permanently counterfactual for that unit (and vice versa), so $\tau_i=Y_i(1)-Y_i(0)$ can
   never be directly computed for any single unit.
2. **Classifying a three-node structure and predicting the effect of conditioning** — reproduce §5's
   table from memory: for chain and fork, conditioning on the middle node blocks the path; for
   collider, conditioning on the middle node opens it — and explain each case's mechanism, not just
   the label.
3. **The backdoor adjustment formula, and why it requires summing over strata of Z** — §8.1:
   $P(Y\mid\text{do}(X))=\sum_z P(Y\mid X,Z{=}z)P(Z{=}z)$ — derive in words why this is a
   *weighted average of within-stratum comparisons*, directly generalizing the Simpson's-paradox fix
   of comparing recovery rates within each severity stratum before combining.

### Applied scenario — Amazon marketing-attribution

**Framing:** Amazon wants to know whether a specific email marketing campaign *causes* increased
purchase rates, using historical data where customers who opened the email are compared to customers
who didn't.

**Data:** Observational — customer purchase history, email engagement logs, and available customer
covariates (past purchase frequency, account age, browsing activity).

**Model:** This is a textbook confounding setup (§2): customers who are already more engaged with
Amazon (frequent browsers, frequent past purchasers) are *both* more likely to open marketing emails
*and* more likely to purchase regardless of the email — engagement level is a confounder $Z$ inducing
a spurious correlation between "opened email" ($X$) and "purchased" ($Y$) even if the email's true
causal effect were small or zero. The naive comparison $\mathbb{E}[Y\mid X{=}1]-\mathbb{E}[Y\mid
X{=}0]$ (§1's exact "prediction vs. causal question" trap) would overstate the campaign's effect.
Applying **backdoor adjustment** (§8.1) using measured engagement covariates as the adjustment set $Z$
— provided they block all backdoor paths — gives a properly identified estimate of
$P(\text{purchase}\mid\text{do}(\text{opened email}))$. Where feasible and ethical, a **randomized
holdout** (an RCT, §9) — randomly withholding the campaign from a control group — sidesteps the
identification problem entirely and is the gold-standard design Amazon marketing teams typically
prefer when cost and customer experience allow it.

**Metric:** The estimated ATE (incremental purchase rate caused by the campaign), not the raw
open-vs-non-open purchase-rate gap, since only the former isolates the campaign's actual causal
contribution from pre-existing engagement differences.

**Failure modes:** Unmeasured confounders (e.g., a customer's underlying purchase intent that both
makes them likely to check email *and* likely to buy) would violate the backdoor criterion if not
captured by the measured covariates — this is exactly the scenario where frontdoor adjustment (§8.2)
or a genuine randomized holdout becomes necessary instead of backdoor adjustment on the covariates at
hand.

**What you'd ship:** A randomized holdout test as the primary evidence where feasible, cross-validated
against an observational backdoor-adjusted estimate using engagement covariates as a sanity check —
flagging any large divergence between the two as evidence of residual, unmeasured confounding.

**Leadership Principle tie-in:** **Dive Deep** — refusing to accept "email openers purchase more" as
evidence the email *works*, and instead explicitly identifying engagement level as the confounder
driving both, is the precise discipline this lecture is built to instill. **Insist on the Highest
Standards** — preferring a randomized holdout over a purely observational estimate whenever a real
business decision (marketing spend) rides on the answer reflects exactly the RCT-vs-observational
trade-off §9 lays out.

---

## Glossary

- **ATE (Average Treatment Effect)** — $\mathbb{E}[\tau_i]$, the average causal effect across the
  whole population.
- **ATT (Average Treatment effect on the Treated)** — $\mathbb{E}[\tau_i\mid T_i{=}1]$, the average
  effect among units that actually received treatment.
- **Backdoor path** — a path between treatment and outcome that starts with an arrow into the
  treatment node; the source of confounding.
- **CATE (Conditional Average Treatment Effect)** — $\mathbb{E}[\tau_i\mid X_i{=}x]$, the average
  effect within a covariate-defined subgroup.
- **Chain** — a three-node structure $X\to M\to Y$; conditioning on $M$ blocks the $X$–$Y$
  association.
- **Collider** — a three-node structure $X\to C\leftarrow Y$; conditioning on $C$ *opens* a spurious
  $X$–$Y$ association (Berkson's paradox).
- **Confounder** — a variable that causally influences both the treatment and the outcome, creating
  an association between them even absent a direct causal effect.
- **DAG (Directed Acyclic Graph)** — a graph encoding a factorization of a joint distribution, with
  causal semantics: edges are direct causal influences, edge deletion represents intervention.
- **d-separation** — the sound and complete graphical procedure for reading conditional
  independencies directly off a DAG.
- **do-calculus** — Pearl's three-rule proof system for reducing $\text{do}(\cdot)$ expressions to
  purely observational quantities.
- **do-operator, $\text{do}(X{=}x)$** — notation for forcing $X=x$ via intervention, as opposed to
  observing $X=x$.
- **Fork** — a three-node structure $X\leftarrow Z\to Y$ (confounder); conditioning on $Z$ blocks the
  $X$–$Y$ association.
- **Frontdoor adjustment** — an identification formula using a measured mediator, valid when
  confounders are unobserved.
- **Fundamental problem of causal inference** — the fact that only one of a unit's two potential
  outcomes is ever observed.
- **Identification** — the process of expressing an unobservable causal quantity in terms of an
  estimable statistical one.
- **Potential outcomes** — $Y_i(1)$ and $Y_i(0)$, the two possible outcomes a unit could have under
  treatment or control.
- **RCT (Randomized Controlled Trial)** — an experimental design where treatment assignment is
  randomized, guaranteeing $T\perp\{Y(0),Y(1)\}$.
- **Simpson's paradox** — an association that reverses between the aggregate population and every
  subgroup, caused by an unequally-distributed confounder.
- **Switching equation** — $Y_i = T_iY_i(1) + (1-T_i)Y_i(0)$; formalizes that the observed outcome
  reveals exactly one potential outcome.

---

## Check yourself

1. Write $P(Y\mid X)\ne P(Y\mid\text{do}(X))$ from memory and explain, in one sentence each, what the
   left and right sides mean. *(§1.1)*
2. Explain how a confounder can produce a strong correlation (firefighters vs. property damage) with
   zero or even reversed causal effect. *(§2.1)*
3. Given hospital recovery rates that reverse between aggregate and subgroup level, identify the
   confounder and explain the mechanism using stratum weights. *(§2.2)*
4. Write the switching equation and explain why it makes $\tau_i$ fundamentally unobservable for any
   single unit. *(§3.1–3.2)*
5. Distinguish ATE, ATT, and CATE, and give a business question each one specifically answers. *(§3.3)*
6. Write the DAG factorization formula, and explain what "deleting an edge" means in causal terms.
   *(§4)*
7. Classify each of the three three-node structures (chain, fork, collider) and state whether
   conditioning on the middle node blocks or opens the path, for each. *(§5)*
8. State d-separation's blocking definition, including both the non-collider and collider cases.
   *(§6)*
9. Name the three rules of do-calculus in one sentence each, without the formal notation. *(§7)*
10. State the backdoor adjustment formula and its validity condition. When would you reach for
    frontdoor adjustment instead? *(§8)*
11. Explain precisely what randomization guarantees, and list the four practical reasons RCTs aren't
    always available. *(§9)*

---

## Going deeper

1. **Simpson (1951), "The Interpretation of Interaction in Contingency Tables"** — the original
   Simpson's paradox paper, named directly on the lecture's slide [slide 32]. `solid` · primary
   source for §2.2.
2. **Rubin (1974), "Estimating causal effects of treatments in randomized and nonrandomized
   studies"** — the foundational potential-outcomes paper, named directly [slide 36]. `solid` ·
   primary source for §3.
3. **Pearl (1988), *Probabilistic Reasoning in Intelligent Systems*, Morgan Kaufmann** — the
   foundational text on graphical causal models, named directly [slide 38]. `hard` · primary source
   for §4, dense but the definitive reference.
4. **Dawid (1979), "Conditional Independence in Statistical Theory"** — named directly [slide 40] as
   the source for §5's formal conditional-independence framework. `hard`.
5. **Verma & Pearl (1988), "Causal Networks: Semantics and Expressiveness"** — the original
   d-separation paper, named directly [slide 43]. `hard` · primary source for §6.
6. **Pearl (1995), "Causal Diagrams for Empirical Research"** — introduces the three rules of
   do-calculus, named directly [slide 45]. `hard` · primary source for §7.
7. **Pearl (2000), *Causality*, Ch. 3** — the comprehensive textbook treatment of backdoor/frontdoor
   criteria, named directly [slide 47] exactly as shown (the slide does not display a publisher).
   `solid` · the standard reference for §8, more accessible than the original papers as a starting
   point. (Published by Cambridge University Press, for reference — but that detail is not itself
   shown on the slide, so it's omitted here to match what item 7 actually transcribes.)
8. **Fisher (1935), *The Design of Experiments*, Oliver & Boyd** — the foundational text on
   randomized experimentation, named directly [slide 51]. `solid` · primary source for §9.
9. **Pearl, Glymour, Jewell (2016), *Causal Inference in Statistics: A Primer*** — a widely used,
   more approachable companion to Pearl's denser texts, covering DAGs, d-separation, and do-calculus
   with worked examples. `intro` · a good next read before tackling Pearl (1988/2000) directly.
   ⚠️ not itself named on the lecture's slides — included here as a commonly recommended companion.

> ⚠️ **verify this** — all citations except item 9 are transcribed directly from the lecture's own
> slides (author, year, title, and — where shown — publisher), so their accuracy reflects the slide's
> accuracy rather than independent verification. Item 9 was not shown on any slide and is added from
> general knowledge of the field; treat it as a recommendation, not a lecture citation.
