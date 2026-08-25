---
title: "Supervised Learning — Part 1: Problem Formulation, Data Discipline, Bias–Variance & the Two Linear Models"
topic: supervised-learning
lecture: 01
source: "output/Lecture_01 - Module 1 Supervised Learning Part 1"
instructor: "Apoorva Singh (Applied Scientist II, Amazon)"
slides: 26
video: "https://www.youtube.com/watch?v=Kmc09aTiA3w"
---

# Supervised Learning — Part 1
### Problem Formulation, Data Discipline, Bias–Variance & the Two Linear Models

> ✅ **Capture note.** This deck is **complete**: 26 distinct slide states, title through
> *Thank you*, recovered from the raw capture in `output/` by taking the last frame of each stable
> run. That breaks down as **11 content slides**, **5 "Key Takeaways" slides** (the deck closes every
> section with one), 5 section dividers, the agenda, the instructor slide, a generic "ML Summer
> School" branding/intro card (`f1`–`f2`, distinct from the lecture's own title card), the lecture
> title card, and the close (11+5+5+1+1+1+1+1 = 26). Nothing below is reconstructed or guessed. Where
> I teach something the slide did not, it is marked `📚 Background the slide assumed`.
>
> ⚠️ Note that `slides_deduped/Lecture_01/` also holds 26 images, but they are the **first** frame of
> each run rather than the last — so animated builds may appear half-finished there. Prefer the
> frames cited here.

---

## Where this sits

This is the **first** lecture of Module 1 and it is the prerequisite for everything after it. It is
taught by a different instructor from Part 2 and has a completely different character: Part 1 is
about **how to set a problem up and not fool yourself**; Part 2 is about the machinery (losses,
optimisers, metrics, classifiers).

Concretely, this lecture introduces four things that Part 2 then leans on without re-explaining:

| Introduced here | Used in [Part 2](supervised-learning-02.md) |
|---|---|
| Train/validation/test, data leakage | §14, throughout the metrics section |
| Bias–variance trade-off | Every hyperparameter: δ, η, B, K, C, γ |
| OLS, the normal equation, ridge | §9 (why closed form doesn't scale), §23 (SVM = hinge + L2) |
| Sigmoid, logistic regression, MLE | §6 (BCE), §7 (softmax), §20 (generative vs discriminative) |

Where Part 2 goes deeper on something, I say so and link rather than repeating the derivation.

---

## What you'll understand after reading this

By the end of this document you will be able to:

1. **State what machine learning is and is not**, and place any problem into one of the three
   learning paradigms from its data alone.
2. **Walk the seven-stage ML workflow** and explain, with an example, why it is a loop rather than a
   pipeline.
3. **Formulate a vague business request as a concrete $(X, Y, \text{hypothesis space})$ triple**, and
   explain why that step matters more than the algorithm choice.
4. **Explain why three splits are needed rather than two**, and quantify how much test data you need
   for a result to mean anything.
5. **Identify all four leakage mechanisms in a described pipeline** and state the one rule that
   prevents most of them.
6. **Diagnose underfitting vs. overfitting from a train/validation error pair alone**, and name the
   correct fix for each.
7. **Derive the bias–variance decomposition** $\mathbb{E}[(\hat y - y)^2] = \text{Var}(\hat y) +
   \text{Bias}^2$ from scratch, including why the cross-term vanishes.
8. **Read a learning curve** and say whether more data will help.
9. **Fit an OLS line by hand** and verify the answer via the normal equation.
10. **State all four OLS assumptions, say what breaks when each fails**, and name the fix.
11. **Explain why L1 produces exact zeros and L2 never does**, from the shape of the penalty.
12. **Derive the logistic regression decision boundary** and explain why a model with "regression" in
    its name is a classifier.

---

## Before we start: what you need to know

The deck assumes a programmer's comfort with data but not much mathematics. Everything the slides
lean on is taught here from zero. If a section looks familiar, read the **bold** line and move on.

### Prerequisite 1 — The vocabulary of a dataset

> **Dataset** — a table. Rows are things you observed; columns are facts about them.

| Term | Means | In a table |
|---|---|---|
| **Example** / instance / sample / observation | One thing you observed | one **row** |
| **Feature** / attribute / predictor / covariate | One measured property | one **column** |
| **Label** / target / response / ground truth | The thing you're trying to predict | one special **column** |
| **Feature vector** | All features for one example | one row, minus the label |

Notation used throughout, and on the *Problem Formulation* slide:

| Symbol | Read it as | What it means |
|---|---|---|
| $X$ | "big X" | The whole feature matrix — $n$ rows by $d$ columns. |
| $x_i$ | "x sub i" | The $i$-th example's feature vector. |
| $x_{ij}$ | "x sub i j" | Feature $j$ of example $i$ — one cell. |
| $Y$ | "big Y" | The label column, all $n$ of them. |
| $y_i$ | "y sub i" | The true label for example $i$. |
| $\hat y_i$ | "y-hat sub i" | The model's **prediction**. The hat always means "estimated, not observed". |
| $n$ | "n" | Number of examples (rows). |
| $d$ | "d" | Number of features (columns). |

### Prerequisite 2 — Function, parameter, hyperparameter

> **Model** — a family of functions with adjustable knobs, plus a rule for setting them from data.
>
> *In everyday words:* a shape with dials on the side. The shape is fixed by you; the dials are set
> by the data.
>
> *Concretely:* $\hat y = w_1 x + w_0$ is the family of all straight lines. Training picks one.

- **Parameters** ($w$, $b$, $\beta$) are set by the training algorithm from the data.
- **Hyperparameters** ($\lambda$ in ridge, the polynomial degree, the split ratio) are set by **you**,
  before training, and the algorithm never touches them.

That distinction runs through this entire lecture: the validation set exists specifically to choose
**hyper**parameters, because using training data for that would be circular and using test data for
it would be cheating.

### Prerequisite 3 — Expectation, variance, and what "$\mathbb{E}$" means

The bias–variance slide shows $\mathbb{E}[\,\cdot\,]$ three times without ever defining it.

> **Expectation** $\mathbb{E}[Z]$ — the long-run average value of a random quantity $Z$.
>
> *Concretely:* a fair six-sided die has $\mathbb{E}[Z] = \frac{1+2+3+4+5+6}{6} = 3.5$. Note you can
> never *roll* 3.5 — an expectation is an average, not an outcome.

> **Variance** $\text{Var}(Z) = \mathbb{E}\big[(Z - \mathbb{E}[Z])^2\big]$ — the average squared
> distance from the mean. How spread out $Z$ is.
>
> *Concretely:* for $Z \in \{4, 5, 6\}$ each equally likely, $\mathbb{E}[Z] = 5$ and
> $\text{Var}(Z) = \frac{1 + 0 + 1}{3} = 0.667$.

**The crucial and non-obvious part:** in the bias–variance decomposition, the random thing is *the
model itself*. Imagine drawing a fresh training set from the world, fitting a model, and recording
its prediction at one fixed input $x$. Do that a thousand times and you get a thousand predictions.
$\mathbb{E}[\hat y]$ is their average; $\text{Var}(\hat y)$ is their spread. **The randomness comes
from which training data you happened to get**, not from the model being non-deterministic.

Two properties used in the derivation:
- $\mathbb{E}[c] = c$ for a constant $c$ — averaging something that never changes gives it back.
- $\mathbb{E}[Z - \mathbb{E}[Z]] = 0$ — deviations from the mean average out to zero, by definition.

### Prerequisite 4 — Matrix notation, just enough for the normal equation

The linear-regression slide states $w = (X^\top X)^{-1}X^\top y$ and moves on. You need three things:

- $X^\top$ ("X transpose") — flip rows and columns. An $n \times d$ matrix becomes $d \times n$.
- $AB$ — matrix multiplication. Legal only when $A$'s column count equals $B$'s row count.
  $X^\top X$ is $(d \times n)(n \times d) = d \times d$: a small square matrix regardless of how many
  examples you have.
- $A^{-1}$ ("A inverse") — the matrix that undoes $A$, i.e. $A^{-1}A = I$. **It does not always
  exist.** When it doesn't, the matrix is called *singular*, and §10 explains exactly when that
  happens and what to do.

### Prerequisite 5 — Derivatives, in one paragraph

$\frac{d}{dw}f(w)$ is the slope of $f$ at $w$. At a minimum the slope is zero, so **the standard way
to minimise something is to differentiate it and set the result to zero.** That single move derives
the OLS solution (§10), the normal equation (§10), the ridge shrinkage formula (§11), and the
bias–variance-optimal complexity (§8). If you can do $\frac{d}{dw}(y - wx)^2 = -2x(y - wx)$ by the
chain rule, you have enough.

---

## The big picture

Almost everyone learning ML starts at the algorithm — *which model should I use?* This lecture
argues, correctly, that the algorithm is the **last** and **least** important decision.

The deck's own agenda tells the story:

```
   1. Introduction to ML & Problem Formulation   ← what are you even asking?
   2. Data Splits & Data Leakage                 ← can you trust your own numbers?
   3. Model Fitting, Bias-Variance Trade-off     ← is it too simple or too complex?
   4. Linear Regression and Logistic Regression  ← ...and only now, an algorithm
```

Three of the four sections are about things that happen **before and around** the model. That
ordering is the lecture's thesis, and its *Key Takeaways* slide states it outright:

> Good problem formulation (choosing the right features X, labels Y, and model family) is often more
> important than the algorithm itself.

The reason is that the failure modes are asymmetric. Choosing logistic regression when gradient
boosting would have been better costs you a few points of AUC. Choosing the wrong label, or leaking
test data into training, produces a model that looks excellent in development and is worthless in
production — and you won't find out for months.

**So the arc of this lecture is: formulate honestly → measure honestly → diagnose the failure →
then, finally, fit something.**

---

# Part 1 — Introduction to ML & Problem Formulation

## 1. What machine learning is

Slide [raw `slide_012`, 3:43]:

> - A field of study that gives computers the ability to learn without being explicitly programmed
>   (**Arthur Samuel, 1959**)
> - Learning patterns from data to make predictions or decisions
> - **Types of Machine Learning:**
>   1. **Supervised Learning:** learn from labelled examples (input → output)
>   2. **Unsupervised Learning:** find hidden patterns in unlabelled data
>   3. **Reinforcement Learning:** learn through trial and error with rewards
> - **Real-world Applications at Amazon:** Product recommendations, fraud detection, Alexa NLU,
>   delivery routing, demand forecasting

### The definition, taken seriously

> **Machine learning** — writing a program that improves at a task by being shown examples, rather
> than by being told the rules.
>
> *In everyday words:* instead of writing "if the email contains 'viagra' and comes from an unknown
> sender, mark it spam", you show the program 100,000 emails already marked spam or not, and it works
> out the rule.
>
> *Concretely:* nobody can write down the rule that distinguishes a photo of a cat from a photo of a
> dog. You genuinely cannot enumerate it. But you can collect a million labelled photos.
>
> *Why it exists:* for a large class of problems the rule is real but **not expressible**. Humans
> apply it effortlessly and cannot articulate it. ML is the technique for extracting an inarticulable
> rule from examples of its output.

> 💡 The **Arthur Samuel (1959)** attribution is genuine — Samuel coined "machine learning" while
> building a checkers program at IBM that improved by playing against itself. The deck's own *Key
> Takeaways* slide makes the sharper point: *"the definition hasn't changed since Arthur Samuel
> (1959), but the scale has."* Everything that has changed in 65 years is compute, data volume, and
> model size. The idea is the same.

### The three paradigms

| | **Supervised** | **Unsupervised** | **Reinforcement** |
|---|---|---|---|
| You have | inputs **and** correct outputs | inputs only | an environment that returns rewards |
| It learns | a mapping $X \to Y$ | structure within $X$ | a policy: state → action |
| Feedback | the right answer, immediately, for every example | none | a scalar reward, often **delayed** |
| Example | "is this listing counterfeit?" | "group these customers into segments" | "choose a delivery route" |
| Fails when | labels are expensive or wrong | there is no objective way to check the answer | the reward signal is misspecified |

> 📚 **Background the slide assumed — the labelling bottleneck.** Supervised learning is the most
> reliable of the three and the most expensive, because someone has to produce the labels. That cost
> shapes the whole field: it's why **semi-supervised** learning (a few labels plus lots of unlabelled
> data) and **self-supervised** learning (invent labels from the data's own structure — "predict the
> next word") exist. The Venn diagram on this slide shows exactly those overlaps, with
> *Semi-supervised Learning* between supervised and unsupervised, *Self-supervised Exploration*
> between unsupervised and RL, *Imitation Learning* between supervised and RL, and **Deep Learning**
> in the centre — because a deep network is a *model class*, not a paradigm, and can be trained under
> any of the three.

> 🎯 The Amazon applications list is not decoration — it's a map of the rest of the course, and worth
> memorising as a set of concrete anchors:
>
> | Application | Paradigm | Framed as |
> |---|---|---|
> | Product recommendations | supervised / RL | ranking, or a bandit over slots |
> | Fraud detection | supervised | severely imbalanced binary classification |
> | Alexa NLU | supervised | intent classification + slot filling |
> | Delivery routing | RL / optimisation | sequential decisions under uncertainty |
> | Demand forecasting | supervised | time-series regression |

---

## 2. The ML workflow

Slide [raw `slide_015`, 6:05]:

> 1. **Problem Definition:** What are we trying to predict/classify?
> 2. **Data Collection:** Gather relevant, representative data
> 3. **Feature Engineering:** Transform raw data into useful inputs
> 4. **Model Selection:** Choose an appropriate algorithm
> 5. **Training:** Fit the model to training data
> 6. **Evaluation:** Measure performance on held-out data
> 7. **Deployment & Monitoring:** Serve predictions, track drift
>
> **Key Insight: ML is iterative — expect to revisit earlier steps as you learn more about the
> problem.**

The accompanying figure draws these as a **circle**, not a line, which is the entire point.

> ⚠️ **A discrepancy worth noticing.** The bulleted list and the circular diagram on this slide do
> not use identical stage names — the diagram reads *Problem Definition → Data Collection → Data
> Preparation → Feature Engineering → Model Training → Model Evaluation → Model Deployment*, i.e. it
> has a separate "Data Preparation" step and folds "Model Selection" into training. This is a minor
> inconsistency in the deck, not a real disagreement; the substance is identical. I use the bulleted
> list's names below.

### Where each stage actually goes wrong

The list is uncontroversial. What makes it useful is knowing the characteristic failure of each.

| Stage | The characteristic failure | What it looks like |
|---|---|---|
| **1. Problem Definition** | Solving a proxy for the real problem | You optimise click-through; the business wanted revenue; the model learns to promote clickbait |
| **2. Data Collection** | Data that isn't **representative** | Trained on desktop traffic, deployed to mobile |
| **3. Feature Engineering** | Leakage (§5) | A feature that won't exist at prediction time |
| **4. Model Selection** | Reaching for complexity first | A transformer where regularised logistic regression matched it |
| **5. Training** | Optimising the wrong objective | See [Part 2 §14](supervised-learning-02.md) — loss ≠ metric |
| **6. Evaluation** | Measuring on data the model has effectively seen | An optimistic number that won't reproduce |
| **7. Deployment & Monitoring** | Assuming the world is static | Silent decay as the distribution shifts |

### Why it's a loop, concretely

The slide says "expect to revisit earlier steps". Here is a realistic trace of that happening:

1. You define the problem as *predict whether a delivery will be late*.
2. You collect data, engineer features, train, evaluate. AUC is 0.95. Suspiciously good.
3. You investigate and find a feature called `delivery_exception_code` — which is only populated
   **after** a delivery has already gone wrong. Leakage.
4. → back to **stage 3**. Drop the feature. AUC falls to 0.71.
5. Now the model is honest but weak. Talking to operations, you learn that "late" is defined
   differently for same-day and standard shipping, and you've been mixing them.
6. → back to **stage 1**. Redefine the target per shipping class.
7. That needs a field you didn't collect. → back to **stage 2**.

Three loops before a single honest model. **This is the normal case, not a sign of incompetence** —
and the reason the diagram is a circle.

> 💡 **The one deployment concept the slide names and doesn't explain: drift.**
>
> - **Data drift** (covariate shift): the inputs change. New product categories, a new device mix,
>   a marketing campaign bringing a different customer population.
> - **Concept drift**: the *relationship* between inputs and label changes. Fraudsters adapt to your
>   detector, so the same features now mean something different.
>
> They need different responses. Data drift often just needs retraining on fresh data. Concept drift
> means your problem definition may have aged out, and you may be looping back to stage 1. Monitoring
> the input distribution catches the first; only monitoring **outcomes** catches the second.

---

## 3. Problem formulation

Slide [raw `slide_018`, 8:38]:

> - **Features (X):** Input variables the model uses to make predictions
>   *e.g., age, income, purchase history, page views*
> - **Labels (Y):** The target variable we want to predict
>   *e.g., "will this customer churn?" (binary), "expected revenue?" (continuous)*
> - **Hypothesis Space:** The set of all possible functions the model can learn
>   - Linear models → hyperplanes
>   - Decision trees → axis-aligned partitions
>   - Neural networks → highly flexible nonlinear mappings
> - **A good formulation = choosing the right X, Y, and model family for the business problem**

### The hypothesis space is the idea to actually absorb

Features and labels are intuitive. **Hypothesis space** is the one that repays thought.

> **Hypothesis space** $\mathcal{H}$ — the complete set of functions your chosen model class is
> capable of representing. Training is a **search within $\mathcal{H}$**; it can never leave it.
>
> *In everyday words:* it's the vocabulary your model is allowed to speak in. Training finds the best
> sentence it can construct — but if the truth cannot be said in that vocabulary, no amount of
> training or data will get you there.
>
> *Concretely:* $\mathcal{H}$ for simple linear regression is every line $\{w_1x + w_0\}$. If the
> true relationship is $y = x^2$, **no line is the right answer.** The best line is still wrong, and
> that irreducible wrongness is exactly what §7 will call **bias**.

The three examples on the slide, made concrete in 2-D:

| Model family | $\mathcal{H}$ contains | Boundary shape | Cannot represent |
|---|---|---|---|
| **Linear** | $\{ \mathbf{w}^\top\mathbf{x} + b \}$ — hyperplanes | one straight cut | XOR; anything curved |
| **Decision trees** | axis-aligned rectangular partitions | staircases | a 45° diagonal boundary, except as an infinite staircase |
| **Neural networks** | compositions of weighted sums and nonlinearities | essentially arbitrary | very little — which is both the appeal and the danger |

> 💡 **The trade-off in one line:** a bigger hypothesis space can express more truths *and* more
> falsehoods. Search a large enough space and you will find something that fits your training data
> perfectly by accident. That is overfitting (§6), and it is why the answer to "which model family?"
> is never automatically "the most expressive one".

### 🧪 Worked example — formulating a vague request

**The request from the business:** *"Can you help us reduce customer churn?"*

That is not a machine learning problem. Turning it into one means making six decisions, and every one
of them is a place to go wrong.

| Decision | Options | Consequence of choosing wrong |
|---|---|---|
| **What is churn?** | no purchase in 30 / 90 / 180 days? cancelled subscription? | 30 days over-flags seasonal buyers; 180 days means you learn about it too late to act |
| **Prediction horizon** | will they churn in the next 30 days? ever? | "ever" is nearly unactionable — everyone churns eventually |
| **$Y$ type** | binary (will churn) vs. continuous (days until churn) vs. ranking (who's most at risk) | Binary is simplest. Ranking is often what the retention team actually needs, since they can only call 500 people |
| **$X$: what's available at prediction time?** | last-90-day activity, tenure, support tickets, category mix | Including `cancellation_reason` leaks the answer (§5) |
| **Population** | all customers? only active ones? only paid? | Including already-churned customers makes the task trivially easy and useless |
| **Model family** | logistic regression / GBDT / neural net | Least important of the six. Start with the simplest that could work |

**A defensible formulation:**
- **Population:** customers with ≥1 purchase in the last 180 days.
- **$Y$:** 1 if zero purchases in the **next** 60 days, else 0.
- **$X$:** features computed strictly from data available on the prediction date — purchase
  frequency, recency, category diversity, support-contact count, tenure.
- **$\mathcal{H}$:** regularised logistic regression first, because it's interpretable and the
  retention team will ask *why* a customer was flagged.
- **Output used as:** a ranked list, top 500 per week, because that is the team's calling capacity.

Notice that the last row changed the problem: because the output is consumed as a **ranking**, the
right metric is precision@500, not accuracy — a Part 2 concern that was determined here, in
formulation.

---

## 4. Key Takeaways — Section 1

The deck's own summary [raw `slide_020`, 9:19], verbatim:

> - ML is about learning patterns from data rather than explicit programming: the definition hasn't
>   changed since Arthur Samuel (1959), but the scale has.
> - The ML workflow is iterative, not linear. You should expect to loop back to earlier steps.
> - Good problem formulation (choosing the right features X, labels Y, and model family) is often
>   more important than the algorithm itself.

---

# Part 2 — Data Splits & Data Leakage

## 5. Train / validation / test

Slide [raw `slide_025`, 11:13]:

> - **Training Set (~60-80%):** Model learns patterns from this data
> - **Validation Set (~10-20%):** Used to tune hyperparameters, select models
> - **Test Set (~10-20%):** Final, unbiased evaluation — touched only once
>
> - **Why split?**
>   - Simulates how the model will perform on unseen real-world data
>   - Prevents overfitting to training noise
>
> - **Best Practices:**
>   - Stratified splits for imbalanced classes
>   - Temporal splits for time-series data
>   - Never use test data for any model decision

### Why three, and not two

Almost everyone understands train vs. test. The **validation** set is the one people skip, and the
reason it exists is subtle and important.

Suppose you have only train and test. You train 30 models with different hyperparameters, score each
on test, and ship the best. What have you done?

**You used the test set 30 times to make a decision.** The winner won partly on merit and partly
because it happened to suit that particular test sample. Your reported number is now optimistic, and
you have no way to know by how much.

This is **multiple-comparisons bias**, and it is quantifiable. If you evaluate 30 genuinely
equivalent models, the *best-looking* one is on average roughly 2 standard errors above the truth. On
a 1,000-example test set at ~90% accuracy, one standard error is
$\sqrt{0.9 \times 0.1 / 1000} \approx 0.0095$ — so the winner is inflated by about **2 percentage
points** purely by selection. That is exactly the size of improvement people write documents about.

**The fix is a third split.** Choose everything on validation; touch test once, at the very end, to
get one honest number.

> 💡 **The right mental model:** every time you look at a dataset and *change something* as a result,
> you spend a little of that dataset's ability to tell you the truth. The training set is spent
> immediately and completely. The validation set is spent slowly, over dozens of decisions — which is
> why on long projects you should periodically re-split it. The test set is spent the moment you look
> at it. The *Key Takeaways* slide's phrasing is exactly right: **"The test set is sacred."**

### 🧪 Worked example — sizing the splits

10,000 labelled examples, 70/15/15:

$$\text{train} = 7{,}000 \qquad \text{validation} = 1{,}500 \qquad \text{test} = 1{,}500$$

**Is 1,500 test examples enough?** That depends on the difference you need to detect. For an accuracy
around 90%, the standard error is

$$\text{SE} = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0.9 \times 0.1}{1500}} = \sqrt{0.00006} = 0.00775$$

A 95% confidence interval is roughly $\pm 1.96 \times 0.00775 = \pm 0.0152$, i.e. **±1.5 percentage
points**. So with 1,500 test examples you can confidently detect a 3-point improvement and cannot
distinguish a 1-point one.

> ⚠️ **Turn that around before you start.** If the business cares about a 1-point improvement, you
> need roughly $\left(\frac{1.96 \times 0.3}{0.01}\right)^2 \approx 3{,}500$ — and more like 14,000
> to resolve half a point. **Decide the detectable effect size first, then size the test set**, not
> the other way round. Splitting 70/15/15 by reflex and then trying to defend a 0.8-point win is a
> conversation that does not go well.

### The three best practices, unpacked

**Stratified splits for imbalanced classes.** A stratified split preserves the class proportions in
every split.

🧪 1,000 examples, 5% positive (50 positives). A plain random 20% test split *expects* 10 positives,
but the actual count is binomial with standard deviation
$\sqrt{200 \times 0.05 \times 0.95} = \sqrt{9.5} = 3.08$. So a $\pm2\sigma$ range is roughly **4 to
16 positives**. Your test-set recall would be computed from as few as 4 examples — statistically
meaningless, and it would swing wildly with the random seed. Stratification pins it to exactly 10.
(`train_test_split(..., stratify=y)`.)

**Temporal splits for time-series data.** If your data has a time dimension, a random split lets the
model train on Thursday and predict Wednesday. In production it will only ever predict *forward*.

```
   RANDOM SPLIT ON TIME-SERIES DATA — WRONG
   Jan  Feb  Mar  Apr  May  Jun  Jul  Aug
   [T][V][T][T][V][T][V][T][T][V][T][T][V][T]     ← train and val interleaved
        ▲ the model sees June while predicting March

   TEMPORAL SPLIT — CORRECT
   Jan  Feb  Mar  Apr  May  Jun  Jul  Aug
   [────── train ──────][── val ──][─ test ─]
                        ▲ every evaluation is strictly in the future
```

The random version reports a wonderful score and then fails in production, because it was evaluated
on a task (interpolation) that is fundamentally easier than the one it will actually do
(extrapolation).

**Never use test data for any model decision.** Including — and people do all of these — choosing the
number of epochs, selecting features, deciding a threshold, or picking between two model families.

> 📚 **Background the slide assumed — cross-validation, and when you'd use it instead.** With little
> data, holding out 15% for validation is painful: you lose training data *and* your validation
> estimate is noisy. **k-fold cross-validation** solves both. Split the training data into $k$ parts;
> train $k$ times, each time holding out a different part; average the $k$ scores.
>
> ```
>    5-FOLD CROSS-VALIDATION
>    Fold 1:  [VAL][ tr ][ tr ][ tr ][ tr ]   → score₁
>    Fold 2:  [ tr ][VAL][ tr ][ tr ][ tr ]   → score₂
>    Fold 3:  [ tr ][ tr ][VAL][ tr ][ tr ]   → score₃
>    Fold 4:  [ tr ][ tr ][ tr ][VAL][ tr ]   → score₄
>    Fold 5:  [ tr ][ tr ][ tr ][ tr ][VAL]   → score₅
>                                        report mean ± std
> ```
>
> Every example is used for training ($k-1$ times) and for validation (once). You get an **error bar
> for free**, which is the real prize — reporting "0.86 ± 0.02" is a far stronger claim than "0.86".
> Cost: $k$ times the training compute, which is why you use a single validation split on large data
> and cross-validation on small.
>
> ⚠️ The **test set still stays out of this entirely.** Cross-validation replaces the validation
> split, not the test split.
>
> 💡 **See also.** This k-fold machinery reappears, largely unchanged, as the *inner loop* of nested
> cross-validation in Dimensionality Reduction's feature-selection chapter (§12) and as hyperparameter
> search's evaluation harness in Deep Neural Networks (Part 3 §12); Causal Inference's *cross-fitting*
> for double machine learning is the same idea adapted so nuisance models don't leak into the
> treatment-effect estimate.

---

## 6. Data leakage

Slide [raw `slide_028`, 13:03]:

> - **Definition:** When information from outside the training set leaks into the model
> - **Common causes:**
>   - Using future data to predict the past (temporal leakage)
>   - Feature derived from the target variable
>   - Preprocessing (e.g., normalization) on full dataset before split
>   - Duplicate records spanning train and test
> - **How to prevent:**
>   - Always split FIRST, preprocess AFTER
>   - Use pipelines (sklearn Pipeline) to encapsulate transforms
>   - Think carefully about what would be available at prediction time
> - **Leakage leads to overly optimistic metrics that collapse in production!**

This is the most operationally important slide in the lecture. Leakage is the single most common way
a competent ML project silently fails, and it fails in the worst possible way: **it looks like
success.**

### The four mechanisms, each with a concrete instance

**① Temporal leakage — using future data to predict the past.**

You're predicting whether a customer will churn this month. One of your features is
`support_tickets_last_90_days`, computed over a window that — because of how you assembled the table
— extends past the prediction date. Customers about to churn file complaints first. Your model
learns "people who will complain next week are about to churn", which is true, useful, and
**unavailable at prediction time.**

*Symptom:* the feature has suspiciously high importance and the model is far better than domain
experts expected.

**② A feature derived from the target.**

The purest form. Predicting delivery lateness using `delivery_exception_code`, which is only
populated *because* the delivery went wrong. Or predicting fraud using `chargeback_flag`, which is
set by the fraud team after they investigate.

*Symptom:* near-perfect performance. **AUC above ~0.98 on a genuinely hard problem is not a
triumph, it is a bug report.**

**③ Preprocessing before splitting.**

The subtlest, and the one the slide's "always split FIRST" rule targets.

🧪 You have 10 values of a feature: `[1, 2, 3, 4, 5, 6, 7, 8, 100, 200]`, and the last two land in
the test set. You standardise before splitting:

$$\mu_{\text{all}} = \frac{1+2+\cdots+8+100+200}{10} = \frac{336}{10} = \mathbf{33.6}$$

But the correct, leakage-free statistic uses training data only:

$$\mu_{\text{train}} = \frac{1+2+\cdots+8}{8} = \frac{36}{8} = \mathbf{4.5}$$

A training value of 4 is transformed to $(4 - 33.6)/\sigma_{\text{all}}$ in the leaky version and
$(4 - 4.5)/\sigma_{\text{train}}$ in the correct one — completely different numbers. **The training
features now encode information about the test set's distribution.**

This applies to every fitted transform, not just standardisation: min-max scaling, PCA, missing-value
imputation, target encoding of categoricals, feature selection, SMOTE. Anything with a `.fit()` must
be fitted on training data only.

*Symptom:* validation looks fine, production is a few points worse and nobody can explain why. This
one leaks *less* dramatically than ① and ②, which makes it harder to catch.

**④ Duplicate records spanning train and test.**

The same listing scraped twice under different IDs. The same customer appearing under two accounts.
Data-augmented copies of the same underlying image landing on both sides. The model memorises the
example in training and "recognises" it in test.

*Symptom:* test performance far above validation performance, or a mysterious gap between offline and
online metrics. Fix by de-duplicating **before** splitting, and by splitting on the **entity**
(customer, listing, patient), not on the row.

### The one rule, and how to make it structural

The slide gives three preventions. The second is the one that turns a rule you have to remember into
one you cannot violate:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# 1. Split FIRST — before touching the data in any way
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# 2. Every fitted transform lives INSIDE the pipeline
pipe = Pipeline([
    ("scale", StandardScaler()),          # fitted on train folds only
    ("clf",   LogisticRegression()),
])

# 3. Cross-validation now re-fits the scaler on each fold's training part.
#    Doing this by hand is exactly where people leak.
scores = cross_val_score(pipe, X_train, y_train, cv=5)

pipe.fit(X_train, y_train)                # scaler fitted on train only
pipe.score(X_test, y_test)                # test transformed with TRAIN statistics
```

> 💡 **Why the `Pipeline` matters more than the rule.** Someone who knows "split first" will still
> leak, because they'll standardise once at the top of the notebook and then run cross-validation
> underneath it — at which point every fold's "training" data has already seen every other fold. The
> `Pipeline` makes the correct behaviour the *default*, and that is the only kind of rule that
> survives contact with a real deadline.

### The third prevention is the one that generalises

> **Think carefully about what would be available at prediction time.**

This is the master test, and it subsumes the others. For every feature, ask: *at the exact moment the
model runs in production, does this value exist yet?*

Run it on the churn example:

| Feature | Available at prediction time? | Verdict |
|---|---|---|
| `tenure_days` | yes | ✅ |
| `purchases_last_90d` (window ending on prediction date) | yes | ✅ |
| `purchases_last_90d` (window ending *today*, in the training table) | **no** | ❌ temporal leakage |
| `cancellation_reason` | no — only exists after churning | ❌ derived from target |
| `support_tickets_next_30d` | no | ❌ future data |
| `avg_order_value` (computed over all history including future orders) | no | ❌ subtle temporal leak |

Row 3 and row 6 are the instructive ones: the *same feature name* is fine or fatal depending on how
its window was computed. This is why leakage survives code review — the bug is in the SQL that built
the table, not in the model.

```interactive
type: simulator
title: Leak it, then fix it
concept: How each leakage mechanism inflates the reported metric, and by how much
control: Four toggles — temporal window, target-derived feature, scale-before-split, duplicate rows — over a fixed synthetic dataset with a known true AUC of 0.74.
observe: Two bars update live: "reported validation AUC" and "true held-out AUC on genuinely unseen data". Toggling target-derived pushes reported to 0.99 while true stays at 0.74; toggling scale-before-split moves reported by only ~0.02.
insight: The dangerous leaks are not the dramatic ones. A target-derived feature gives 0.99 and someone will notice. Scale-before-split gives +2 points — exactly the size of a result you would celebrate and ship.
fallback: The µ_all = 33.6 vs µ_train = 4.5 calculation in §6, and the feature-availability table above.
```

---

## 7. Key Takeaways — Section 2

Verbatim [raw `slide_030`, 13:38]:

> - Always split first, preprocess after. This single rule prevents most leakage.
> - The test set is sacred. Touch it only once for final evaluation.
> - Data leakage produces metrics that look great in development but collapse in production. Think:
>   "What information would actually be available at prediction time?"

---

# Part 3 — Model Fitting & the Bias–Variance Trade-off

## 8. Underfitting vs. overfitting

Slide [raw `slide_035`, 16:08]:

> - **Underfitting (High Bias):**
>   - Model is too simple to capture underlying patterns
>   - High training error AND high test error
>   - *Example:* fitting a straight line to quadratic data
> - **Overfitting (High Variance):**
>   - Model memorizes training data including noise
>   - Low training error BUT high test error
>   - *Example:* high-degree polynomial on sparse data
> - **Generalization:**
>   - The sweet spot where the model captures true patterns without memorizing noise
>   - *Goal:* minimize the gap between train and test performance

### The two-number diagnostic

This slide contains a genuinely powerful practical tool, hidden in plain sight. **From the training
error and the validation error alone, you can name the problem.**

| Training error | Validation error | Diagnosis | What to do |
|---|---|---|---|
| **High** | **High** (similar) | **Underfitting** — high bias | Add features, increase complexity, reduce regularisation, train longer |
| **Low** | **High** (big gap) | **Overfitting** — high variance | More data, fewer features, more regularisation, early stopping |
| Low | Low | **You're done.** Ship it. | — |
| **High** | **Low** | Something is broken | Check for a bug: is validation easier than train? leaked? mis-shuffled? are you applying dropout at eval time? |

That fourth row is not in the deck but you should know it: validation better than training is not a
triumph, it is a signal that your evaluation is wrong. (The one benign exception is regularisation
that is active during training and disabled at evaluation — dropout, or data augmentation applied
only to the training set — which genuinely can make training loss look worse.)

### 🧪 Worked example — the same data, three model complexities

Seven points sampled from $y = 2x + 1$ plus noise:

| $x$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| $y$ | 3.1 | 4.9 | 7.2 | 8.8 | 11.3 | 12.9 | 15.1 |

| Model | Fits training data | Training error | Validation error (new points) | Diagnosis |
|---|---|---|---|---|
| **Constant** $\hat y = \bar y = 9.04$ | badly — it's a flat line through a rising trend | **very high** | **very high** | underfits: $\mathcal{H}$ cannot express a slope |
| **Linear** $\hat y \approx 2.01x + 1.03$ | well | **low** | **low** | ✅ correct complexity |
| **Degree-6 polynomial** | *perfectly* — 7 free coefficients, 7 points, error exactly **0** | **0** | **very high** | overfits: it fitted the noise exactly |

The degree-6 row is the one to sit with. **A degree-$(n-1)$ polynomial can pass exactly through any
$n$ points.** Training error of zero, achieved with certainty, carrying zero information about
whether the model learned anything. Between the observed points that curve oscillates wildly, and at
$x = 8$ it will return something absurd.

> 💡 **Zero training error is not a goal, it is a warning.** Any model with enough capacity can
> memorise the training set. The question is never "did it fit the training data?" but "what did it
> fit the training data *with*?"

### 📚 The bit the slide's own diagram gets slightly wrong

The figure on this slide labels a **wiggly curve through scattered points** as "Overfitting" and a
**straight line through scattered points** as "Underfitting". That's the standard picture and it's
fine as far as it goes — but notice it shows both training data (blue) and validation data (tan)
on the same axes, and the wiggly curve visibly passes through blue points while missing tan ones.
That detail is the actual content: **overfitting is only visible once you plot the held-out data.**
On training data alone, the wiggly curve looks strictly better.

---

## 9. The bias–variance trade-off

Slide [raw `slide_038`, 18:15]:

> - **Total Error = Bias² + Variance + Irreducible Noise**
> - **Bias:** Error from overly simplistic assumptions
>   - High bias → underfitting, misses relevant patterns
> - **Variance:** Error from sensitivity to training data fluctuations
>   - High variance → overfitting, captures noise as signal
> - **The trade-off:**
>   - Increasing model complexity → decreases bias, increases variance
>   - Decreasing model complexity → increases bias, decreases variance
> - **Practical strategies:** regularization, cross-validation, ensemble methods

The slide also shows the decomposition with each term annotated:

$$\underbrace{\mathbb{E}\big[(\hat y - y)^2\big]}_{\text{How far is a model from the ground truth}} = \underbrace{\mathbb{E}\big[(\hat y - \mathbb{E}[\hat y])^2\big]}_{\text{How far is a model from the "average model"}} + \underbrace{\big(\mathbb{E}[\hat y] - y\big)^2}_{\text{How far is the "average model" from the ground truth}}$$

$$\text{MSE} \;=\; \text{Variance} \;+\; \text{Bias}^2$$

### Words before symbols

The formula says: **if you were to retrain your model on many different training sets and look at
its prediction for one fixed input, the average squared error splits cleanly into two parts — how
much the predictions scatter around their own average, and how far that average sits from the
truth.**

| Symbol | Read it as | What it means |
|---|---|---|
| $\hat y$ | "y-hat" | The model's prediction at one fixed $x$. **Random**, because the training set is random. |
| $y$ | "y" | The true value at that $x$. Fixed. |
| $\mathbb{E}[\hat y]$ | "expected y-hat" | The average prediction across all possible training sets — the "average model". |
| $\mathbb{E}[(\hat y - \mathbb{E}[\hat y])^2]$ | "variance" | Scatter of the predictions around their own mean. |
| $(\mathbb{E}[\hat y] - y)^2$ | "bias squared" | How far the average model is from the truth. |

### Deriving it

The slide asserts the decomposition. Here it is in four lines — one of the cleanest derivations in
all of ML, and worth being able to reproduce.

Start with the total squared error and **add and subtract $\mathbb{E}[\hat y]$**, which changes
nothing:

$$\mathbb{E}\big[(\hat y - y)^2\big] = \mathbb{E}\Big[\big(\underbrace{\hat y - \mathbb{E}[\hat y]}_{A} + \underbrace{\mathbb{E}[\hat y] - y}_{B}\big)^2\Big]$$

Expand the square as $(A+B)^2 = A^2 + 2AB + B^2$:

$$= \underbrace{\mathbb{E}\big[(\hat y - \mathbb{E}[\hat y])^2\big]}_{\text{Variance}} + 2\,\mathbb{E}\big[(\hat y - \mathbb{E}[\hat y])(\mathbb{E}[\hat y] - y)\big] + \underbrace{\mathbb{E}\big[(\mathbb{E}[\hat y] - y)^2\big]}_{\text{Bias}^2}$$

**The cross term vanishes.** The factor $(\mathbb{E}[\hat y] - y)$ is a **constant** — neither piece
depends on which training set you drew — so it pulls out of the expectation:

$$2\,\mathbb{E}\big[(\hat y - \mathbb{E}[\hat y])(\mathbb{E}[\hat y] - y)\big] = 2\big(\mathbb{E}[\hat y] - y\big)\cdot\underbrace{\mathbb{E}\big[\hat y - \mathbb{E}[\hat y]\big]}_{=\ 0} = 0$$

using the identity from Prerequisite 3: deviations from the mean average to zero. The third term is
also constant, so its expectation is itself. Therefore

$$\boxed{\ \mathbb{E}\big[(\hat y - y)^2\big] = \text{Var}(\hat y) + \text{Bias}(\hat y)^2\ }\qquad\blacksquare$$

> ⚠️ **Where did the "Irreducible Noise" go?** The bulleted line says *Total Error = Bias² + Variance
> + Irreducible Noise*, but the boxed formula has only two terms. Both are correct — they are
> answering slightly different questions, and the deck doesn't flag the difference.
>
> The boxed derivation treats $y$ as a **fixed** true value. In reality the label you observe is
> $y = f(x) + \varepsilon$ where $f$ is the true function and $\varepsilon$ is measurement noise with
> variance $\sigma^2$. Redo the derivation against the *observed* $y$ rather than $f(x)$ and a third
> term appears:
> $$\mathbb{E}\big[(\hat y - y)^2\big] = \text{Var}(\hat y) + \text{Bias}^2 + \sigma^2$$
> $\sigma^2$ is **irreducible**: it is the noise in the labels themselves. No model, no amount of
> data, no architecture can go below it. Two identical listings with different human-assigned labels
> put a hard floor under your achievable error.
>
> 🎯 This is worth saying in an interview: "our model is at 8% error and the label-noise floor from
> our annotator-agreement study is about 6%, so there is at most 2 points left on the table" is a far
> more sophisticated statement than "we're at 8% and trying to improve."

### 🧪 Worked example — three models, decomposed numerically

At one fixed input $x$, the true value is $y = 5$. We train each model on three different training
samples and record its prediction:

| Model | Predictions across 3 samples | $\mathbb{E}[\hat y]$ | Bias | Bias² | Variance | **Total** |
|---|---|---|---|---|---|---|
| **A** (too simple) | 3.0, 3.1, 2.9 | 3.00 | $-2.00$ | **4.000** | **0.007** | **4.007** |
| **B** (too complex) | 2.0, 5.0, 8.0 | 5.00 | $0.00$ | **0.000** | **6.000** | **6.000** |
| **C** (balanced) | 4.2, 4.8, 5.4 | 4.80 | $-0.20$ | **0.040** | **0.240** | **0.280** |

Working Model C explicitly:
- $\mathbb{E}[\hat y] = \frac{4.2 + 4.8 + 5.4}{3} = \frac{14.4}{3} = 4.8$
- $\text{Bias}^2 = (4.8 - 5)^2 = (-0.2)^2 = 0.04$
- $\text{Var} = \frac{(4.2-4.8)^2 + (4.8-4.8)^2 + (5.4-4.8)^2}{3} = \frac{0.36 + 0 + 0.36}{3} = \frac{0.72}{3} = 0.24$
- Sum: $0.04 + 0.24 = \mathbf{0.28}$

**Verifying directly**, without the decomposition:
$$\frac{(4.2-5)^2 + (4.8-5)^2 + (5.4-5)^2}{3} = \frac{0.64 + 0.04 + 0.16}{3} = \frac{0.84}{3} = \mathbf{0.28}\ ✓$$

The two routes agree, which is the decomposition working.

> 💡 **Look at Model B.** Its bias is **exactly zero** — on average it is perfectly correct. And it is
> the **worst** of the three, with total error 6.0. An unbiased model can be useless. This is why
> "unbiased" is not a synonym for "good", and why the trade-off is a trade-off: you will often
> **accept bias deliberately** to buy a larger reduction in variance. That is precisely what ridge
> regression does (§11), what a small $K$ does not do and a large $K$ does in KNN, and what every
> regularisation technique in ML is for.

### Reading the trade-off curve

The slide's second figure plots three curves against **Model Complexity**: Bias² falling, Variance
rising, and Total Error as a U-shape with a dotted line at "Optimum Model Complexity".

```
   error
     ▲
     │╲                                              ╱ Total Error
     │ ╲                                          ╱
     │  ╲                                      ╱
     │   ╲            ┌ optimum ┐          ╱
     │    ╲___        │         │      ╱          ╱ Variance
     │        ╲___    │         │  ╱          ╱
     │            ╲___│_________│____     ╱
     │                ╲             ╲ ╱
     │                 ╲         ╱   ╲
     │                  ╲____╱        ╲______  Bias²
     └────────────────────┼──────────────────────────▶
                          ▲                  model complexity
                     the sweet spot
```

Three things to take from it:

1. **The minimum of the total is not where either component is minimised.** It's where their
   *derivatives* cancel — where one more unit of complexity buys less bias reduction than it costs in
   variance.
2. **The curve is asymmetric in practice.** The left side (underfitting) rises steeply and is easy to
   detect; the right side is shallower and much easier to sit on without noticing.
3. **"Model complexity" is whatever knob you're turning.** Polynomial degree, tree depth, number of
   features, $1/\lambda$, network width, $1/K$ in KNN, training epochs. The same picture applies to
   all of them, which is why this one diagram covers most hyperparameter tuning you will ever do.

> ⚠️ **The honest caveat: deep learning breaks this picture.** The classical U-curve predicts that
> massively over-parameterised models should be catastrophically bad. They are not — a network with
> far more parameters than training examples often generalises *better* than a smaller one. The
> phenomenon is called **double descent**: test error rises to a peak at the interpolation threshold
> (where the model has just enough capacity to fit the training data exactly), then **falls again**
> as capacity grows further.
>
> The classical trade-off is still the right mental model for the models in this lecture — linear,
> regularised linear, trees, KNN, SVMs. Do not carry it uncritically into very large networks.
> ✅ **Confirmed via primary source.** Double descent is a well-replicated empirical finding. The two
> usual references are Belkin, Hsu, Ma & Mandal, *"Reconciling modern machine-learning practice and
> the classical bias–variance trade-off"*, PNAS 116(32), 2019, pp. 15849–15854; and Nakkiran, Kaplun,
> Bansal, Yang, Barak & Sutskever, *"Deep Double Descent: Where Bigger Models and More Data Hurt"*,
> ICLR 2020 (arXiv:1912.02292). The theoretical *account* of why it happens is still unsettled — that
> part of the honesty flag stands — but the citations themselves are correct.
>
> 🔬 **Research opportunity.** *Why* over-parameterised models descend a second time is still an open
> question — the empirical curve is not in dispute, but there is no single agreed mechanistic
> explanation the way there is for the classical U-curve's bias–variance trade-off. Competing accounts
> invoke implicit regularisation from SGD, benign overfitting in high dimensions, and the geometry of
> the loss landscape near the interpolation threshold; none is yet the settled textbook answer the way
> bias–variance is. If you want a research-flavoured interview answer, this is the place to have one.

### The three practical strategies the slide names

**Regularisation** — add a penalty for complexity to the objective. Reduces variance, adds a little
bias. This is §11, and it's the main lever.

**Cross-validation** — doesn't change bias or variance; it lets you *measure* where you are on the
curve reliably enough to choose. Without it you cannot find the optimum, you can only guess at it.

**Ensemble methods** — combine several models. Two distinct mechanisms, and knowing which is which
is a common interview question:

| | **Bagging** (e.g. Random Forest) | **Boosting** (e.g. XGBoost) |
|---|---|---|
| Trains models | in parallel, on bootstrap resamples | sequentially, each on the previous one's errors |
| Primarily reduces | **variance** | **bias** |
| Base learner should be | low-bias, high-variance (deep trees) | high-bias, low-variance (shallow "stumps") |
| Overfits if | rarely — more trees is roughly safe | yes — more rounds eventually overfits |

The variance reduction in bagging is quantifiable: averaging $M$ models with individual variance
$\sigma^2$ and pairwise correlation $\rho$ gives variance $\rho\sigma^2 + \frac{1-\rho}{M}\sigma^2$.
As $M \to \infty$ the second term vanishes and you're left with $\rho\sigma^2$ — **which is why
Random Forests randomise the feature subset at each split.** It isn't for speed; it's to drive $\rho$
down, because $\rho$ is the floor.

```interactive
type: slider
title: The U-curve, built from scratch
concept: How bias and variance move in opposite directions as complexity changes, and why total error is U-shaped
control: A slider for polynomial degree (1→15) fitting a fixed noisy sample from a cubic; a button re-draws 20 fresh training samples.
observe: Top panel shows all 20 fitted curves overlaid — at degree 1 they lie almost on top of each other but far from the truth; at degree 15 they scatter wildly. Bottom panel plots measured Bias², Variance and Total against degree, with the running minimum marked.
insight: Variance is literally visible as the spread of the 20 curves, and bias as the offset of their average from the true curve. The U-curve is not a diagram to memorise — it is a measurement you just made.
fallback: The three-model table in §9 — Model B has zero bias and the worst total error (6.000); Model C has non-zero bias and the best (0.280).
```

---

## 10. Bias–variance: practical diagnostics

Slide [raw `slide_041`, 19:50]:

> - **High Bias (underfitting) signals:**
>   - Training accuracy is low
>   - Training and validation errors are both high and similar
>   - *Fix:* more features, more complex model, less regularization
> - **High Variance (overfitting) signals:**
>   - Training accuracy is high, validation accuracy is much lower
>   - Large gap between train and validation performance
>   - *Fix:* more data, fewer features, more regularization, early stopping
> - **Learning curves (error vs. training size) are the best diagnostic tool**

### 📚 Learning curves — the slide's headline claim, explained

The slide bolds this and doesn't define it. It's the most actionable idea in the section.

> **Learning curve** — a plot of training error and validation error **against the number of training
> examples used**. You train the model repeatedly on 10%, 20%, … 100% of your data and plot both
> errors each time.

It answers a question nothing else answers cleanly: **will collecting more data help?** That is
usually the most expensive decision on the table, and guessing at it is costly.

```
   HIGH BIAS                          HIGH VARIANCE
   error                              error
     ▲                                  ▲
     │                                  │╲
     │   ╭──────── validation           │ ╲______ validation
     │  ╱                               │        ╲______
     │ ╱                                │               ╲___
     │╱  ╭──────── training             │
     ├───╯                              │      ___________ training
     │   both plateau HIGH              │   ╱
     │   and CLOSE together             │ ╱   big persistent GAP
     └──────────────────▶               └──────────────────▶
        training set size                  training set size

   → More data will NOT help.          → More data WILL help.
     The curves have already met.        The gap is still closing.
     Increase capacity instead.
```

**How to read it:**

| What you see | Means | Action |
|---|---|---|
| Both curves plateau, high, close together | High bias. The model has hit its capacity ceiling. | More data is **wasted money**. Add features or capacity. |
| Large gap, validation still falling at the right edge | High variance, not yet saturated | More data **will** help. Collect it. |
| Large gap, validation curve **flat** at the right edge | High variance, saturated | More data won't help either. Regularise or simplify. |
| Curves have converged, both low | Done. | Ship. |

> 💡 **Why this is worth the compute.** "Should we spend three months and \$200k labelling more data?"
> is a question you can answer in an afternoon by training on 10%, 25%, 50% and 100% of what you
> already have and extrapolating the validation curve. Answering it by intuition is how teams spend
> quarters collecting data for a model that was bias-limited all along.

### The fixes, and why each one works

The slide gives seven fixes across the two conditions. Each maps to a specific mechanism:

| Fix | For | Mechanism |
|---|---|---|
| More features | bias | Enlarges $\mathcal{H}$ so the truth might now be inside it |
| More complex model | bias | Same — a bigger hypothesis space |
| Less regularisation | bias | Stops artificially constraining the search |
| **More data** | variance | The model can no longer fit noise that doesn't repeat across examples |
| Fewer features | variance | Shrinks $\mathcal{H}$, fewer ways to fit noise |
| More regularisation | variance | Penalises the extreme parameter values that noise-fitting requires |
| Early stopping | variance | Stops before the optimiser has had time to fit the noise |

> ⚠️ **Two traps in this list.**
>
> **"More data" never fixes bias.** If a straight line cannot represent your quadratic relationship,
> ten million examples still cannot make it. You will get a *very precisely estimated* wrong line.
> This is the single most common misallocation in applied ML, and the learning curve is precisely the
> instrument that catches it.
>
> **"Add features" and "remove features" are both on this list**, for opposite conditions. Which one
> you need depends on the diagnosis — which is why you diagnose first. Reaching for feature
> engineering when you have a variance problem makes it worse.

```interactive
type: graph
title: Will more data help?
concept: Reading a learning curve to decide between collecting data and changing the model
control: A model-complexity selector (constant / linear / degree-3 / degree-12) over a fixed noisy cubic dataset, plus a draggable marker for "how much data could we realistically collect?" extending past the current dataset size.
observe: Training and validation error are plotted against training-set size, redrawn per model. A verdict panel reads "MORE DATA WILL HELP — projected validation error at 3× data: 0.14" or "MORE DATA WON'T HELP — curves already converged at 0.31; increase capacity instead."
insight: The two shapes are unmistakable once seen side by side — converged-and-high means you are capacity-limited and buying data is wasted money; a still-closing gap means the data is the binding constraint. This is the diagram that answers a six-figure question in an afternoon.
fallback: The ASCII learning-curve sketches and the four-row reading table in §10.
```

> 💡 **See also.** This bias–variance framework is not a one-lecture idea — it reappears as the reason
> PCA's retained-variance/reconstruction-error trade-off has a "how many components" question at all
> (Dimensionality Reduction), as the lens Deep Neural Networks uses to explain *why* its regularisation
> chapter's fixes (weight decay, dropout, early stopping) work, and as the same decomposition applied
> to *value estimates instead of model predictions* in Reinforcement Learning's Monte Carlo vs.
> temporal-difference comparison (§7.1 there).

---

## 11. Key Takeaways — Section 3

Verbatim [raw `slide_043`, 20:34]:

> - Total Error = Bias² + Variance + Irreducible Noise. You can only control the first two.
> - Underfitting (high bias): model is too simple → both train and test errors are high.
> - Overfitting (high variance): model memorizes noise → low train error, high test error.
> - Learning curves are your best diagnostic tool to distinguish these.
> - Practical levers: regularization reduces variance; more features/complexity reduces bias; more
>   data helps with variance.

---

# Part 4 — Linear Regression

## 12. Ordinary Least Squares

Slide [raw `slide_048`, 23:47]:

> - **Goal:** Model a linear relationship between features X and continuous target Y
> - **Ordinary Least Squares (OLS):**
>   - `y_hat = w0 + w1*x1 + w2*x2 + ... + wn*xn`
>   - Minimize: sum of `(yi - y_hat_i)^2`
> - **Key Assumptions:**
>   - Linearity of the relationship
>   - Independence of errors
>   - Homoscedasticity (constant variance of errors)
>   - No multicollinearity among features
> - **Closed-form solution:** `w = (X^T X)^(-1) X^T y` (Normal Equation)

The figure shows a scatter with a best-fit line, a residual $\varepsilon$ marked as the vertical gap
from a point to the line, and the model written as $y = \beta_0 + \beta_1 x + \varepsilon$.

### Words before symbols

The model says: **the prediction is a weighted sum of the features, plus a constant.**

$$\hat y = w_0 + w_1x_1 + w_2x_2 + \cdots + w_dx_d = w_0 + \sum_{j=1}^{d} w_jx_j$$

| Symbol | Read it as | What it means |
|---|---|---|
| $w_0$ | "w nought", the **intercept** or **bias** | The prediction when every feature is zero. |
| $w_j$ | "w sub j", a **coefficient** or **weight** | How much $\hat y$ changes per one-unit increase in $x_j$, **holding all other features fixed**. |
| $\hat y$ | "y-hat" | The prediction. |

The fitting rule says: **choose the weights that make the total squared vertical distance from the
points to the line as small as possible.**

$$\mathbf{w}^\star = \arg\min_{\mathbf{w}} \sum_{i=1}^{n}\big(y_i - \hat y_i\big)^2$$

> 📚 **Background the slide assumed — why *squared*, and why *vertical*?**
>
> **Squared** rather than absolute: three reasons, and Part 2 §2 derives them properly. It's
> differentiable everywhere (so calculus works and a closed form exists), it's the maximum-likelihood
> estimate under Gaussian noise, and it has a unique minimum. The cost is sensitivity to outliers.
>
> **Vertical** distance rather than perpendicular: because the model treats $x$ as **given** and only
> $y$ as uncertain. You are asking "given this $x$, what is $y$?" — so only the error in $y$ counts.
> (If both variables are noisy, the right tool is Total Least Squares / orthogonal regression, which
> minimises perpendicular distance and gives a different answer.)

### 🧪 Worked example — fit a line by hand

| $x$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $y$ | 2 | 4 | 5 | 4 | 5 |

**Step 1 — means.** $\bar x = \frac{15}{5} = 3$, $\bar y = \frac{20}{5} = 4$.

**Step 2 — the slope.** For simple regression, minimising the squared error gives

$$w_1 = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2}$$

| $x_i$ | $x_i - \bar x$ | $y_i$ | $y_i - \bar y$ | product | $(x_i-\bar x)^2$ |
|---|---|---|---|---|---|
| 1 | $-2$ | 2 | $-2$ | $4$ | $4$ |
| 2 | $-1$ | 4 | $0$ | $0$ | $1$ |
| 3 | $0$ | 5 | $1$ | $0$ | $0$ |
| 4 | $1$ | 4 | $0$ | $0$ | $1$ |
| 5 | $2$ | 5 | $1$ | $2$ | $4$ |
| | | | **Σ** | $\mathbf{6}$ | $\mathbf{10}$ |

$$w_1 = \frac{6}{10} = \mathbf{0.6}$$

**Step 3 — the intercept.** The fitted line always passes through $(\bar x, \bar y)$:

$$w_0 = \bar y - w_1\bar x = 4 - 0.6(3) = 4 - 1.8 = \mathbf{2.2}$$

$$\boxed{\ \hat y = 2.2 + 0.6x\ }$$

**Step 4 — check it.**

| $x$ | $\hat y = 2.2 + 0.6x$ | $y$ | residual $y - \hat y$ | squared |
|---|---|---|---|---|
| 1 | 2.8 | 2 | $-0.8$ | 0.64 |
| 2 | 3.4 | 4 | $+0.6$ | 0.36 |
| 3 | 4.0 | 5 | $+1.0$ | 1.00 |
| 4 | 4.6 | 4 | $-0.6$ | 0.36 |
| 5 | 5.2 | 5 | $-0.2$ | 0.04 |
| | | | **Σ = 0.0** ✓ | **SSE = 2.40** |

**The residuals sum to exactly zero.** That is not luck — it's a mathematical consequence of fitting
an intercept. Setting $\partial/\partial w_0 \sum(y_i - w_0 - w_1x_i)^2 = 0$ gives
$\sum(y_i - \hat y_i) = 0$ directly. **If your residuals don't sum to ~0, you have a bug.**

### Deriving and verifying the normal equation

The slide states $\mathbf{w} = (X^\top X)^{-1}X^\top y$ without derivation. It takes three lines.

Write the data as a matrix $X$ (with a leading column of 1s for the intercept) and the targets as a
vector $y$. The objective is $\|y - X\mathbf{w}\|^2$. Expand:

$$\mathcal{L}(\mathbf{w}) = (y - X\mathbf{w})^\top(y - X\mathbf{w}) = y^\top y - 2\mathbf{w}^\top X^\top y + \mathbf{w}^\top X^\top X\mathbf{w}$$

Differentiate with respect to $\mathbf{w}$ and set to zero:

$$\nabla_\mathbf{w}\mathcal{L} = -2X^\top y + 2X^\top X\mathbf{w} = 0 \quad\Longrightarrow\quad X^\top X\mathbf{w} = X^\top y$$

$$\boxed{\ \mathbf{w}^\star = (X^\top X)^{-1}X^\top y\ }\qquad\blacksquare$$

🧪 **Now verify it reproduces our hand-fitted line.** With the intercept column:

$$X = \begin{pmatrix}1&1\\1&2\\1&3\\1&4\\1&5\end{pmatrix},\qquad y = \begin{pmatrix}2\\4\\5\\4\\5\end{pmatrix}$$

$$X^\top X = \begin{pmatrix} n & \sum x \\ \sum x & \sum x^2\end{pmatrix} = \begin{pmatrix}5 & 15\\15 & 55\end{pmatrix},\qquad X^\top y = \begin{pmatrix}\sum y \\ \sum xy\end{pmatrix} = \begin{pmatrix}20\\66\end{pmatrix}$$

(Checking $\sum xy = 1(2)+2(4)+3(5)+4(4)+5(5) = 2+8+15+16+25 = 66$ ✓)

The determinant is $5(55) - 15(15) = 275 - 225 = 50$, so

$$(X^\top X)^{-1} = \frac{1}{50}\begin{pmatrix}55 & -15\\-15 & 5\end{pmatrix}$$

$$\mathbf{w} = \frac{1}{50}\begin{pmatrix}55 & -15\\-15 & 5\end{pmatrix}\begin{pmatrix}20\\66\end{pmatrix} = \frac{1}{50}\begin{pmatrix}1100 - 990\\-300 + 330\end{pmatrix} = \frac{1}{50}\begin{pmatrix}110\\30\end{pmatrix} = \begin{pmatrix}\mathbf{2.2}\\\mathbf{0.6}\end{pmatrix}$$

**Identical to the hand calculation.** ✓

> 💡 **When you would *not* use the normal equation.** It requires inverting a $d \times d$ matrix,
> which costs $O(d^3)$ and needs $O(d^2)$ memory. At $d = 100{,}000$ features that's $10^{15}$
> operations and 40 GB. And it only exists at all because MSE + a linear model happens to have a
> closed-form stationary point — change the loss or the model and there is no formula.
> [Part 2 §9–13](supervised-learning-02.md) is the general answer: iterative gradient-based
> optimisation, which works for everything.
>
> (In practice, even when $d$ is small, numerical libraries don't literally invert the matrix — they
> solve $X^\top X\mathbf{w} = X^\top y$ by QR or SVD decomposition, which is more numerically stable.
> `np.linalg.lstsq` does this; explicitly computing `inv(X.T @ X)` is a mild code smell.)

### The four assumptions, and what each failure does to you

The slide lists them. Here is what actually goes wrong.

| Assumption | Plain meaning | What breaks if violated | How to detect | Fix |
|---|---|---|---|---|
| **Linearity** | The true relationship is a weighted sum | The model is **biased** — systematically wrong, no matter how much data | Plot residuals vs. $\hat y$: a curve or arc means non-linearity | Add polynomial/interaction terms, transform features, or change model family |
| **Independence of errors** | One example's error tells you nothing about another's | Coefficients are still unbiased but **standard errors are wrong** — you'll believe results that aren't real | Plot residuals in collection order; look for runs. Durbin–Watson for time series | Time-series models; cluster-robust standard errors; mixed models |
| **Homoscedasticity** | Error spread is the same everywhere | Same as above — estimates OK, **uncertainty understated**. OLS also over-weights the noisy region | Residuals vs. $\hat y$ shows a **funnel/cone** shape | Transform $y$ (e.g. $\log$), or use weighted least squares |
| **No multicollinearity** | No feature is a near-copy of a combination of others | $X^\top X$ is near-singular → coefficients become **huge, unstable, and sign-flip** on tiny data changes | Variance Inflation Factor > 5–10; a near-zero eigenvalue of $X^\top X$ | Drop one of the correlated features, or use **ridge** (§13) |

> 💡 **The residual plot is the single highest-value diagnostic**, and it checks three of the four at
> once. Plot residuals on the y-axis against fitted values on the x-axis. You want a **structureless
> horizontal band**. A curve means non-linearity; a cone means heteroscedasticity; visible clumps
> mean dependence. Five seconds of looking at that plot catches more than any summary statistic.

> ⚠️ **Multicollinearity is the assumption that connects this section to the next.** When two features
> are nearly identical, $X^\top X$ is nearly singular, its inverse blows up, and the fitted
> coefficients become enormous with opposite signs that cancel. The *predictions* may still be fine;
> the *coefficients* are meaningless, and the model is extremely unstable. That is exactly the
> disease ridge regression cures — and it cures it by making the matrix invertible again. Read §13
> with this in mind.

---

## 13. Regularised linear regression

Slide [raw `slide_051`, 26:41]:

> - **Ridge Regression (L2):**
>   - `Loss = sum(yi - y_hat_i)^2 + lambda * sum(wj^2)`
>   - Shrinks coefficients toward zero; handles multicollinearity
>   - All features retained (none set exactly to zero)
> - **Lasso Regression (L1):**
>   - `Loss = sum(yi - y_hat_i)^2 + lambda * sum(|wj|)`
>   - Can set coefficients exactly to zero → feature selection
> - **Elastic Net (L1 + L2):**
>   - `Loss = MSE + lambda1 * sum(|wj|) + lambda2 * sum(wj^2)`
>   - Combines sparsity of Lasso with stability of Ridge
> - **lambda (regularization strength): higher → simpler model, more bias, less variance**

The figure shows the classic geometry: an $L^1$-norm **diamond** and an $L^2$-norm **circle**, each
with the elliptical contours of the loss touching them, and a red dot at the solution.

### Words before symbols

The objective says: **don't just fit the data — fit the data *and* keep the coefficients small,
trading a little accuracy on the training set for a model that doesn't swing wildly.**

$$\mathcal{L}_{\text{ridge}} = \underbrace{\sum_{i=1}^n (y_i - \hat y_i)^2}_{\text{fit the data}} + \underbrace{\lambda\sum_{j=1}^d w_j^2}_{\text{keep weights small}}$$

$$\mathcal{L}_{\text{lasso}} = \sum_{i=1}^n (y_i - \hat y_i)^2 + \lambda\sum_{j=1}^d |w_j|$$

| Symbol | Read it as | What it means |
|---|---|---|
| $\lambda$ | "lambda" | **Regularisation strength**, a hyperparameter. $\lambda = 0$ recovers plain OLS; $\lambda \to \infty$ drives every weight to zero. |
| $\sum w_j^2$ | "the squared L2 norm" | Penalises large weights **quadratically** — one weight of 10 costs 100, ten weights of 1 cost 10. |
| $\sum \lvert w_j\rvert$ | "the L1 norm" | Penalises **linearly** — one weight of 10 costs 10, ten weights of 1 also cost 10. |

> ⚠️ **The intercept $w_0$ is not penalised.** It represents the baseline level of $y$, not a
> relationship, and shrinking it would bias every prediction toward zero for no reason. Every
> implementation excludes it. Similarly, **features must be standardised before regularising** —
> otherwise a feature measured in dollars gets a tiny coefficient and escapes the penalty while one
> measured in thousands of dollars gets crushed. The penalty compares raw coefficient magnitudes, so
> the units have to be comparable.

### 🧪 Worked example — ridge shrinks, and how much

A one-feature model $\hat y = wx$ with data $x = [1,2,3]$, $y = [2,4,6]$ — a perfect $w = 2$.

Minimise $\sum (y_i - wx_i)^2 + \lambda w^2$. Differentiate and set to zero:

$$-2\sum x_i(y_i - wx_i) + 2\lambda w = 0 \quad\Longrightarrow\quad w\Big(\sum x_i^2 + \lambda\Big) = \sum x_iy_i$$

$$\boxed{\ w_{\text{ridge}} = \frac{\sum x_iy_i}{\sum x_i^2 + \lambda}\ }$$

With $\sum x_iy_i = 2 + 8 + 18 = 28$ and $\sum x_i^2 = 1 + 4 + 9 = 14$:

| $\lambda$ | $w = \dfrac{28}{14 + \lambda}$ | |
|---|---|---|
| 0 | $28/14 = \mathbf{2.000}$ | the OLS answer |
| 1 | $28/15 = \mathbf{1.867}$ | |
| 14 | $28/28 = \mathbf{1.000}$ | exactly halved when $\lambda = \sum x^2$ |
| 100 | $28/114 = \mathbf{0.246}$ | |
| $10^6$ | $28/1000014 = \mathbf{0.000028}$ | tiny — but **not zero** |

**That last row is the whole point.** The denominator grows without bound but never reaches infinity,
so $w$ approaches zero **asymptotically and never arrives**. This is exactly the slide's claim that
ridge retains all features.

### Deriving why L1 gives *exact* zeros and L2 never does

The slide asserts the difference. Here's the mechanism, which is the best interview answer in this
lecture.

**Look at the derivative of the penalty near $w = 0$.**

| | Penalty | Derivative | Value as $w \to 0^+$ |
|---|---|---|---|
| **L2 (ridge)** | $\lambda w^2$ | $2\lambda w$ | $\to \mathbf{0}$ |
| **L1 (lasso)** | $\lambda\lvert w\rvert$ | $\lambda \cdot \text{sign}(w)$ | $\to \boldsymbol{\lambda}$ — a **constant** |

**Ridge's pull toward zero vanishes as the weight approaches zero.** It's always pushing, but ever
more gently, so it never quite arrives — the same asymptote we just computed.

**Lasso's pull is a constant $\lambda$ all the way in.** So if the data's own pull on that weight
(its gradient contribution) is smaller than $\lambda$, the penalty wins outright and pins the weight
at exactly 0.

Making that precise: with standardised features ($\sum x_j^2 = 1$), the lasso solution is the
**soft-thresholding** operator

$$w_{\text{lasso}} = \text{sign}(w_{\text{OLS}}) \cdot \max\big(0,\ |w_{\text{OLS}}| - \lambda\big)$$

🧪 With $w_{\text{OLS}} = 0.8$:

| $\lambda$ | Lasso: $\max(0,\ 0.8 - \lambda)$ | Ridge: $0.8/(1+\lambda)$ |
|---|---|---|
| 0.0 | $\mathbf{0.800}$ | $\mathbf{0.800}$ |
| 0.3 | $\mathbf{0.500}$ | $\mathbf{0.615}$ |
| 0.6 | $\mathbf{0.200}$ | $\mathbf{0.500}$ |
| 0.8 | $\mathbf{0.000}$ ← **exactly zero** | $\mathbf{0.444}$ |
| 5.0 | $\mathbf{0.000}$ | $\mathbf{0.133}$ |
| 100 | $\mathbf{0.000}$ | $\mathbf{0.0079}$ |

**Lasso hits exactly zero at $\lambda = 0.8$ and stays there. Ridge is still at 0.0079 when
$\lambda = 100$.** ∎

### The geometry on the slide, explained

The diamond-vs-circle figure is the same fact seen visually. Both problems can be written as
"minimise the squared error, subject to the coefficient vector staying inside a budget region":

- **L2 budget region:** $w_1^2 + w_2^2 \le t$ — a **circle** (a sphere in higher dimensions).
- **L1 budget region:** $|w_1| + |w_2| \le t$ — a **diamond**, with sharp corners **on the axes**.

The loss contours are ellipses centred on the OLS solution. The regularised answer is where the
outermost ellipse first touches the budget region.

```
        L1 (Lasso)                          L2 (Ridge)
            w₂                                  w₂
            ▲                                   ▲
            │    ╱ loss contour                 │    ╱ loss contour
         ╱╲ │  ╱                             ___│  ╱
        ╱  ╲│╱                              ╱   │╱
   ─────◆───┼────────▶ w₁              ────(────●───)──────▶ w₁
       ╱ ╲  │╲                              ╲___│  ╲
      ╱   ╲ │  ╲                                │    ╲
            │                                   │
   Touches at a CORNER                  Touches on a SMOOTH ARC
   → the corner sits ON an axis         → generically no coordinate
   → w₂ = 0 EXACTLY                       is exactly zero
```

**A circle has no corners, so a tangent point almost never lands exactly on an axis. A diamond's
corners lie *on* the axes, and corners are where a moving contour is most likely to make first
contact.** Same conclusion as the calculus, arrived at geometrically.

### Choosing between them

| | **Ridge (L2)** | **Lasso (L1)** | **Elastic Net** |
|---|---|---|---|
| Sets weights to exactly 0 | never | yes | yes |
| Feature selection | no | **yes, automatically** | yes |
| With correlated features | **shares** the weight between them, stably | picks **one arbitrarily**, drops the rest — unstable | shares among the correlated group |
| $d > n$ | works | selects at most $n$ features | works, no cap |
| Closed form | **yes**: $(X^\top X + \lambda I)^{-1}X^\top y$ | no — needs iterative solvers | no |
| Use when | all features plausibly matter; multicollinearity is the problem | you want a sparse, interpretable model | many correlated features **and** you want sparsity |

> 💡 **Ridge's closed form is also its numerical justification.** Adding $\lambda I$ inflates every
> diagonal entry of $X^\top X$, which **guarantees invertibility for any $\lambda > 0$** — even when
> features are perfectly collinear or $d > n$. That's why the slide says ridge "handles
> multicollinearity". The statistical fix and the linear-algebra fix are the same fix, because they
> address the same underlying condition: directions in feature space along which the data carries no
> information.

> ⚠️ **The lasso instability that the comparison table hides.** If two features are 0.99 correlated,
> lasso will keep one and zero the other — and *which* one it keeps can flip if you resample the data
> or change the random seed. If you're using lasso for feature selection and reporting "these are the
> important features", that claim is much less stable than it looks. Elastic Net exists largely to
> fix this: the L2 component makes correlated features share, so the *group* is selected together.

**Choosing $\lambda$:** by cross-validation, on a **logarithmic** grid — $\lambda \in \{10^{-4},
10^{-3}, \ldots, 10^{2}\}$. Never linear, because the effect of $\lambda$ is multiplicative.
(`RidgeCV`, `LassoCV`, `ElasticNetCV` do this for you and are strictly better than a hand-rolled
loop.)

And note the last line of the slide connects this whole section back to §9:

> lambda (regularization strength): higher → simpler model, **more bias, less variance**

$\lambda$ **is** the complexity dial from the U-curve. Regularisation is not a separate topic; it is
the bias–variance trade-off with a knob attached.

---

## 14. Key Takeaways — Section 4

Verbatim [raw `slide_053`, 27:29]:

> - OLS minimizes squared residuals and has a closed-form solution but assumes linearity, independent
>   errors, constant variance, and no multicollinearity.
> - Ridge (L2) shrinks all coefficients toward zero while Lasso (L1) can zero out coefficients
>   entirely.
> - The regularization parameter λ directly controls the bias-variance trade-off: higher λ = simpler
>   model.

---

# Part 5 — Logistic Regression

## 15. From a line to a probability

Slide [raw `slide_059`, 31:51]:

> - **Goal:** Model the probability of a binary outcome (classification)
> - **Sigmoid function:** `sigma(z) = 1 / (1 + exp(-z))` maps any real value to [0,1]
> - `P(y=1|x) = sigma(w^T x + b)`
> - **Training via Maximum Likelihood Estimation (MLE):**
>   - Maximize the likelihood of observed labels given the model
>   - Equivalent to minimizing binary cross-entropy loss
> - **Decision Boundary:**
>   - The hyperplane where P(y=1) = 0.5 (i.e., `w^T x + b = 0`)
>   - Linear in feature space
>   - Threshold can be adjusted for precision/recall trade-off

### Why linear regression cannot do this

Start from the obvious bad idea, because seeing why it fails motivates everything else. You want to
predict a 0/1 label. Why not just run linear regression on it and round at 0.5?

Three reasons, and only the third is fatal:

1. **The output isn't bounded.** $\mathbf{w}^\top\mathbf{x} + b$ ranges over all real numbers. A
   customer with a long purchase history gets "probability 3.7", which is not a probability.
2. **The errors are not homoscedastic.** For a binary target, the variance of $y$ given $x$ is
   $p(1-p)$, which depends on $p$ — so OLS assumption 3 (§12) is violated by construction.
3. **It is not robust to the position of extreme points.** Add a few examples far out on the $x$
   axis, all correctly labelled 1, and the fitted line **rotates** to reduce their squared error —
   dragging the 0.5 crossing point sideways and *misclassifying points it previously got right.*
   Being more confidently correct about a distant point makes the model worse, which is absurd.

The fix is to keep the linear part — it's interpretable and cheap — and pass it through a function
that squashes it into $[0,1]$.

### The sigmoid

The formula says: **take any real number and map it smoothly onto the interval between 0 and 1,
sending large negatives toward 0, large positives toward 1, and zero to exactly one half.**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| Symbol | Read it as | What it means |
|---|---|---|
| $z$ | "z", the **logit** or **score** | The linear part: $\mathbf{w}^\top\mathbf{x} + b$. Any real number. |
| $e^{-z}$ | "e to the minus z" | Large when $z$ is very negative, near 0 when $z$ is very positive. |
| $\sigma(z)$ | "sigma of z" | The output probability, strictly between 0 and 1. |

Check the three anchor points, which is all you need to remember its shape:

| $z$ | $e^{-z}$ | $\sigma(z)$ | |
|---|---|---|---|
| $-\infty$ | $\infty$ | $\to \mathbf{0}$ | certainly class 0 |
| $0$ | $1$ | $\frac{1}{1+1} = \mathbf{0.5}$ | maximally uncertain |
| $+\infty$ | $0$ | $\frac{1}{1+0} = \mathbf{1}$ | certainly class 1 |

So the full model is:

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top\mathbf{x} + b)}}$$

> 📚 **Background the slide assumed — the log-odds interpretation, which is where the name comes
> from.** Solve $p = \sigma(z)$ for $z$:
>
> $$p(1 + e^{-z}) = 1 \;\Rightarrow\; e^{-z} = \frac{1-p}{p} \;\Rightarrow\; z = \log\frac{p}{1-p}$$
>
> $\frac{p}{1-p}$ is the **odds** (3:1 odds means $p = 0.75$), and $\log\frac{p}{1-p}$ is the
> **log-odds** or **logit**. So:
>
> $$\log\frac{P(y=1\mid \mathbf{x})}{P(y=0\mid\mathbf{x})} = \mathbf{w}^\top\mathbf{x} + b$$
>
> **Logistic regression is a linear model of the log-odds.** That's the sentence, and it's what makes
> the coefficients interpretable: increasing $x_j$ by one unit multiplies the *odds* by $e^{w_j}$. A
> coefficient of $0.7$ means $e^{0.7} = 2.01$ — that feature roughly **doubles the odds** per unit.
> This is exactly how coefficients get reported in medicine and credit risk, and it is a common
> interview question.

### Training: MLE, and why it equals cross-entropy

The slide says training maximises likelihood and that this is "equivalent to minimizing binary
cross-entropy loss". Here is the equivalence in three lines.

A binary label is **Bernoulli**: it's 1 with probability $p$ and 0 with probability $1-p$. That's
written compactly as

$$P(y \mid p) = p^{\,y}(1-p)^{1-y}$$

(Check: $y=1$ gives $p^1(1-p)^0 = p$ ✓; $y=0$ gives $p^0(1-p)^1 = 1-p$ ✓.)

Assuming examples are independent, the likelihood of the whole dataset is the product; taking logs
turns it into a sum:

$$\log \mathcal{L}(\mathbf{w}) = \sum_{i=1}^n \Big[y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)\Big]$$

Negate it (optimisers minimise) and divide by $n$:

$$-\frac{1}{n}\log\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n\Big[y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)\Big] = \mathcal{L}_{\text{BCE}}$$

**Maximising the likelihood and minimising binary cross-entropy are the same optimisation, exactly.**
∎

> 💡 **There is no closed form here.** Unlike OLS, setting the gradient to zero gives a system with no
> analytic solution, so logistic regression is fit **iteratively** — gradient descent, or Newton-type
> methods like IRLS. The objective is convex, so any decent optimiser reaches the global optimum.
> [Part 2 §6](supervised-learning-02.md) derives the gradient and shows the remarkable cancellation
> that makes it $\hat p - y$.

### 🧪 Worked example — one prediction, end to end

Model: $\mathbf{w} = (0.8,\ -0.5)$, $b = 0.2$. New example: $\mathbf{x} = (2,\ 1)$.

**Step 1 — the linear part.**
$$z = 0.8(2) + (-0.5)(1) + 0.2 = 1.6 - 0.5 + 0.2 = \mathbf{1.3}$$

**Step 2 — the sigmoid.**
$$\sigma(1.3) = \frac{1}{1 + e^{-1.3}} = \frac{1}{1 + 0.27253} = \frac{1}{1.27253} = \mathbf{0.7858}$$

**Step 3 — the decision.** At the default threshold 0.5: $0.7858 > 0.5$, so **predict class 1**.

**Step 4 — cross-check via the log-odds.**
$$\log\frac{0.7858}{1 - 0.7858} = \log\frac{0.7858}{0.2142} = \log(3.6685) = \mathbf{1.30}\ ✓$$

The log-odds recovers $z$ exactly, as it must.

**Step 5 — move the threshold.** Suppose false positives are expensive and you set the threshold at
0.9. Now $0.7858 < 0.9$ → **predict class 0**. *Same model, same example, opposite decision.* This is
the slide's last bullet, and it's the hinge between this lecture and
[Part 2 §16](supervised-learning-02.md): threshold choice is a post-training business decision that
costs nothing and changes everything.

### The decision boundary is a line

The slide says the boundary is "the hyperplane where $P(y=1) = 0.5$, i.e. $\mathbf{w}^\top\mathbf{x}
+ b = 0$" and that it is "linear in feature space". Here's why those are the same statement.

$\sigma(z) = 0.5$ exactly when $z = 0$ — that was the middle row of the anchor table. And $z = 0$
means $\mathbf{w}^\top\mathbf{x} + b = 0$, which is the equation of a **hyperplane**: a line in 2-D,
a plane in 3-D, a flat $(d{-}1)$-dimensional surface in general.

🧪 With our model, the boundary is
$$0.8x_1 - 0.5x_2 + 0.2 = 0 \quad\Longrightarrow\quad x_2 = \frac{0.8x_1 + 0.2}{0.5} = 1.6x_1 + 0.4$$

A straight line. **Verifying our point** $(2, 1)$: at $x_1 = 2$ the boundary sits at
$x_2 = 1.6(2) + 0.4 = 3.6$. Our point has $x_2 = 1$, which is **below** the line — and since the
coefficient on $x_2$ is negative, below means the positive side. Consistent with $z = 1.3 > 0$ ✓.

> 💡 **The sigmoid is non-linear; the boundary is not.** This confuses people constantly. The squash
> changes *how confident* the model is as you move away from the boundary — the confidence gradient
> — but it is a **monotonic** function of $z$, so the set of points where it crosses any fixed
> threshold is still $\{z = \text{const}\}$, which is still a hyperplane. **Logistic regression can
> only ever draw a straight boundary.** To get a curved one you must engineer non-linear features
> (add $x_1^2$, $x_1x_2$), which is exactly the same move as the kernel trick in
> [Part 2 §24](supervised-learning-02.md).

> 🎯 **"Why is it called logistic *regression* if it's a classifier?"** — a genuinely common interview
> opener, and the deck's own *Key Takeaways* slide flags it. Two-part answer: it performs
> **regression on the log-odds** (a continuous quantity, modelled linearly — that part of the name is
> accurate), and the "logistic" refers to the logistic function $\sigma$. Classification only happens
> when you apply a threshold, which is a separate, post-hoc step and not part of the model. Saying
> "it outputs a probability, and thresholding is a decision I make afterwards" shows you understand
> where the model ends and the application begins.

```interactive
type: slider
title: The sigmoid, the boundary, and the threshold
concept: How w, b and the decision threshold each move a logistic classifier — and which of them requires retraining
control: A 2-D scatter of two overlapping classes with sliders for w₁, w₂, b and the decision threshold (0.01→0.99).
observe: Left panel shows the data with the decision boundary line and a confidence gradient shading. Right panel shows the sigmoid curve with the current threshold marked. Precision, recall and accuracy update live.
insight: Moving w rotates the boundary and moving b translates it — both require retraining. Moving the threshold slides the boundary along w's direction with no retraining at all, and it is the only one of the three you get to change after the model is shipped.
fallback: The §15 worked example — z = 1.3 gives p̂ = 0.7858, which is class 1 at threshold 0.5 and class 0 at threshold 0.9.
```

---

## 16. Key Takeaways — Section 5

Verbatim [raw `slide_061`, 32:25]:

> - Logistic regression models probabilities (not just class labels) via the sigmoid function mapping
>   ℝ → [0, 1].
> - The decision boundary is a linear hyperplane where P(y=1) = 0.5, but you can shift the threshold
>   to trade off precision vs. recall based on business needs.
> - Despite the name, it's a classification algorithm, not regression!

---

## Putting it together

```
                    ┌──────────────────────────────────┐
                    │  1. PROBLEM FORMULATION          │
                    │     What is X? What is Y?        │
                    │     Which hypothesis space H?    │
                    └───────────────┬──────────────────┘
                                    │  ← the highest-leverage decision,
                                    │    and the one people skip
                    ┌───────────────▼──────────────────┐
                    │  2. DATA DISCIPLINE              │
                    │     split FIRST, preprocess AFTER│
                    │     train / val / test           │
                    │     ▲ every fitted transform     │
                    │       goes inside the Pipeline   │
                    └───────────────┬──────────────────┘
                                    │  ← without this, everything
                                    │    downstream is a lie
                    ┌───────────────▼──────────────────┐
                    │  3. FIT, THEN DIAGNOSE           │
                    │                                  │
                    │   train err ─┬─ high, val high   │
                    │              │   → HIGH BIAS     │
                    │              │   → more capacity │
                    │              │                   │
                    │              └─ low, val high    │
                    │                  → HIGH VARIANCE │
                    │                  → more data /   │
                    │                    regularise    │
                    │                                  │
                    │   Total = Bias² + Var + σ²       │
                    │   learning curve tells you which │
                    └───────────────┬──────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
   ┌──────────▼──────────┐                   ┌────────────▼───────────┐
   │  CONTINUOUS Y       │                   │  BINARY Y              │
   │  Linear Regression  │                   │  Logistic Regression   │
   │                     │                   │                        │
   │  ŷ = wᵀx + b        │                   │  P(y=1) = σ(wᵀx + b)   │
   │  minimise Σ(y-ŷ)²   │                   │  maximise likelihood   │
   │  w = (XᵀX)⁻¹Xᵀy     │                   │   ≡ minimise BCE       │
   │  ▲ closed form      │                   │  ▲ NO closed form      │
   │                     │                   │    → iterative         │
   │  4 assumptions:     │                   │                        │
   │  linearity          │                   │  boundary: wᵀx + b = 0 │
   │  independence       │                   │  ▲ still LINEAR        │
   │  homoscedasticity   │                   │                        │
   │  no multicollinearity                   │  threshold is a FREE   │
   │        │            │                   │  post-hoc business dial│
   └────────┼────────────┘                   └────────────────────────┘
            │
            │ multicollinearity breaks (XᵀX)⁻¹
            ▼
   ┌─────────────────────────────────────────┐
   │  REGULARISATION — λ is the complexity   │
   │  dial from the U-curve, with a knob     │
   │                                         │
   │  Ridge  L2  λΣw²   → shrinks, never 0   │
   │              ▲ penalty gradient → 0     │
   │  Lasso  L1  λΣ|w|  → EXACT zeros        │
   │              ▲ penalty gradient → λ     │
   │  Elastic    both   → sparse AND stable  │
   └─────────────────────────────────────────┘
                     │
                     ▼
        ═══════════════════════════════
         Part 2: losses, optimisers,
         metrics, and three more models
        ═══════════════════════════════
```

### The four threads

**1. Every decision in this lecture is the bias–variance dial in a different costume.** Model family
(§3), polynomial degree (§8), $\lambda$ (§13), the number of features (§10). Once you see them as one
dial, hyperparameter tuning stops being a bag of tricks.

**2. The value of a number is set by how it was measured, not how large it is.** A 0.99 AUC from a
leaked feature is worth less than a 0.71 from a clean pipeline, because the second one is *true*. The
whole of Part 2 exists downstream of that.

**3. Linear models are the floor you must beat, not the ceiling you aspire to.** OLS and logistic
regression are fast, interpretable, need almost no tuning, and are frequently within a few points of
a heavily-engineered model. Their real job is to be the baseline number that tells you whether your
sophisticated approach is actually adding anything, or whether you have a pipeline bug.

**4. There is always a floor you cannot go below.** Irreducible noise $\sigma^2$ is real. Knowing
roughly where it sits — from annotator agreement, or from repeated measurements — turns "we should
keep improving the model" into "we are 2 points from the ceiling and should go work on something
else."

---

## Interview prep — Amazon Applied Scientist

### Core questions, easy → hard

<details>
<summary><b>1. Why do you need a validation set — isn't train/test enough?</b></summary>

Because every time you use a dataset to make a decision, you spend some of its ability to tell you
the truth.

With only train and test, you'd tune hyperparameters by scoring repeatedly on test. After 30 model
variants, the winner won partly on merit and partly because it happened to suit that particular test
sample. That's multiple-comparisons bias, and it's quantifiable: with 30 equivalent models the
best-looking one sits roughly 2 standard errors above the truth. On 1,000 test examples at ~90%
accuracy, one SE is about 0.95 points — so you're inflated by ~2 points purely by selection, which is
exactly the size of improvement people write documents about.

The validation set absorbs that cost so the test set stays clean. Choose everything on validation;
touch test once, at the end, for a single honest number. On small data I'd use k-fold
cross-validation instead of a single validation split — every example gets used for both training and
validation, and I get an error bar for free, which matters more than the point estimate.
</details>

<details>
<summary><b>2. Name four ways data leakage happens and the one rule that prevents most of them.</b></summary>

1. **Temporal leakage** — a feature window that extends past the prediction date. Predicting churn
   using support tickets filed after the prediction cutoff.
2. **A feature derived from the target** — predicting delivery lateness using an exception code that
   only gets set because the delivery went wrong.
3. **Preprocessing before splitting** — standardising, imputing, or fitting PCA on the full dataset,
   so training features encode test-set statistics.
4. **Duplicate records spanning splits** — the same listing under two IDs, so the model memorises it
   in train and "recognises" it in test.

The rule is **split first, preprocess after** — and the way to make it stick is a `sklearn.Pipeline`,
because someone who knows the rule will still standardise once at the top of a notebook and then run
cross-validation underneath it, at which point every fold has already seen every other fold.

The master test that subsumes all four: *for every feature, at the exact moment the model runs in
production, does this value exist yet?*
</details>

<details>
<summary><b>3. Your model has 99% training accuracy and 71% validation accuracy. What's wrong and what do you do?</b></summary>

That's a 28-point gap — classic high variance, overfitting. The model has enough capacity to memorise
the training set including its noise.

Before reaching for fixes I'd rule out the boring explanations: is there a duplicate-record leak
inflating train? Is validation drawn from a different distribution or time period? Is the split
stratified?

Assuming it's genuinely overfitting, I'd plot a **learning curve** — validation error against
training-set size — because it tells me which fix will work:

- If validation error is still falling at the right edge, **more data will help** and that's the
  cleanest fix.
- If it's flat, more data won't help either, and I need to reduce variance: more regularisation
  (increase λ), fewer features, a simpler model family, or early stopping.

I'd start with regularisation because it's a one-line change and I can sweep λ on a log grid with
cross-validation in an afternoon, versus months to collect data.
</details>

<details>
<summary><b>4. Derive the bias–variance decomposition.</b></summary>

Take the expected squared error at a fixed input, where the randomness is over which training set you
drew. Add and subtract the mean prediction:

$$\mathbb{E}[(\hat y - y)^2] = \mathbb{E}\big[\big((\hat y - \mathbb{E}[\hat y]) + (\mathbb{E}[\hat y] - y)\big)^2\big]$$

Expand as $(A+B)^2 = A^2 + 2AB + B^2$:

$$= \underbrace{\mathbb{E}[(\hat y - \mathbb{E}[\hat y])^2]}_{\text{Variance}} + 2\mathbb{E}[(\hat y - \mathbb{E}[\hat y])(\mathbb{E}[\hat y]-y)] + \underbrace{(\mathbb{E}[\hat y]-y)^2}_{\text{Bias}^2}$$

The cross term dies because $(\mathbb{E}[\hat y] - y)$ is a constant — it pulls out of the
expectation — leaving $\mathbb{E}[\hat y - \mathbb{E}[\hat y]]$, which is zero by definition of the
mean. So $\text{MSE} = \text{Var} + \text{Bias}^2$.

I'd add that this version treats $y$ as fixed. If the observed label is itself noisy,
$y = f(x) + \varepsilon$, a third term $\sigma^2$ appears — the irreducible noise. That one matters
practically: it's the floor you can't go below, and knowing roughly where it sits tells you when to
stop working on the model.
</details>

<details>
<summary><b>5. Ridge vs. Lasso — why does one give exact zeros and the other never does? [combines two concepts]</b></summary>

Look at the derivative of the penalty as the weight approaches zero.

- **Ridge**, penalty $\lambda w^2$, derivative $2\lambda w$ → **goes to 0** as $w \to 0$. The pull
  toward zero weakens exactly as you approach zero, so you converge asymptotically and never arrive.
- **Lasso**, penalty $\lambda|w|$, derivative $\lambda \cdot \text{sign}(w)$ → **stays at $\lambda$**
  all the way in. If the data's pull on that coefficient is weaker than $\lambda$, the penalty wins
  outright and pins it at exactly zero.

Made precise, with standardised features the lasso solution is soft-thresholding:
$w = \text{sign}(w_{\text{OLS}})\max(0, |w_{\text{OLS}}| - \lambda)$. With $w_{\text{OLS}} = 0.8$ and
$\lambda = 0.8$ it's exactly zero; ridge at $\lambda = 100$ is still at 0.0079.

Geometrically it's the same fact: the L1 constraint region is a diamond whose corners sit on the
axes, and a moving loss contour tends to touch a corner first. The L2 region is a circle, which has
no corners, so the tangent point almost never lands on an axis.

Practically I'd choose ridge when all features plausibly matter and multicollinearity is the problem
— it shares weight between correlated features stably, and it has a closed form. Lasso when I want
sparsity and interpretability, with the caveat that with correlated features it picks one arbitrarily
and that choice can flip on resampling. Elastic Net when I have correlated groups and still want
sparsity.
</details>

<details>
<summary><b>6. Why can't you just use linear regression for a binary target? [combines two concepts]</b></summary>

Three reasons, escalating.

First, the output isn't bounded — $\mathbf{w}^\top\mathbf{x}+b$ can return 3.7, which isn't a
probability.

Second, it violates OLS's homoscedasticity assumption by construction: for a binary target the
conditional variance is $p(1-p)$, which depends on $p$, so the error spread is not constant.

Third, and the fatal one: it's not robust to the *position* of correctly-classified points. Add
examples far out on the x-axis, all correctly labelled 1, and OLS rotates the line to reduce their
squared error — dragging the 0.5 crossing sideways and misclassifying points it previously got right.
Being more confidently correct about a distant point makes the model worse. That's not a tuning
issue, it's the wrong objective.

Logistic regression fixes all three by keeping the linear part and passing it through the sigmoid,
then training by maximum likelihood under a Bernoulli model rather than by squared error. That
Bernoulli NLL is exactly binary cross-entropy.
</details>

<details>
<summary><b>7. Logistic regression's boundary is linear even though the sigmoid is non-linear. Explain. [combines two concepts]</b></summary>

The boundary is the set of points where the predicted probability equals the threshold. Since
$\sigma$ is **monotonic**, $\sigma(z) = 0.5$ happens exactly when $z = 0$ — and $z = \mathbf{w}^\top
\mathbf{x} + b = 0$ is the equation of a hyperplane.

The sigmoid changes *how fast confidence grows* as you move away from the boundary; it doesn't change
*where* the boundary is. And this holds for any threshold, not just 0.5 — a different threshold just
means a different constant $c$ in $z = c$, which is a parallel hyperplane.

So logistic regression can only ever draw a straight boundary. To get a curved one you engineer
non-linear features — add $x_1^2$, $x_1 x_2$ — at which point the boundary is linear in the *expanded*
space and curved in the original one. That's the same move as the kernel trick in an SVM, just done
explicitly rather than implicitly.
</details>

<details>
<summary><b>8. You have a fraud model. Business says "catch more fraud." What do you actually change?</b></summary>

First thing, and it's free: **move the threshold**. The model outputs a score, and the class comes
from comparing it to a cutoff I chose. Lowering it catches more fraud immediately, with no
retraining, no new data, no code beyond a config value.

But that's a trade, so I'd want the second half of the sentence before I did it. Lowering the
threshold raises recall and lowers precision — more false alarms into the investigation queue. So
I'd ask: what's the review team's capacity, and what does a missed fraud cost versus a false alarm?
Those two numbers determine the threshold, and they're business inputs, not modelling ones.

Then I'd sweep the threshold on validation and bring back a table — recall at each precision level —
and let them pick the operating point. Often the right answer isn't one threshold but two: a high one
that auto-blocks, a middle band that goes to human review, and below that nothing. That turns a
painful either/or into a routing decision.

Only if no achievable operating point is good enough would I go back to the model — and then I'd
check the learning curve first to see whether more data or more capacity is the lever.
</details>

<details>
<summary><b>9. Walk me through the four OLS assumptions and what actually goes wrong. [combines two concepts]</b></summary>

**Linearity** — if the true relationship is curved, the model is systematically biased and no amount
of data fixes it. That's a pure bias problem in the bias–variance sense: the truth isn't in the
hypothesis space. Detect it with a residuals-vs-fitted plot showing a curve; fix with polynomial or
interaction terms, or change model family.

**Independence of errors** — with correlated errors (time series, clustered data) the coefficients
are still unbiased, but the *standard errors* are wrong, usually too small. So you believe results
that aren't real. Detect with residuals in collection order, or Durbin–Watson; fix with time-series
models or cluster-robust standard errors.

**Homoscedasticity** — non-constant error variance. Again the estimates are fine but the uncertainty
is understated, and OLS over-weights the noisy regions. Detect with a funnel shape in the residual
plot; fix by transforming $y$ or using weighted least squares.

**No multicollinearity** — this is the one that actually breaks the math. Near-duplicate features
make $X^\top X$ near-singular, its inverse blows up, and coefficients become enormous with
cancelling signs. Predictions can still be fine; coefficients are meaningless and unstable. Detect
with VIF > 5–10; fix by dropping a feature or using ridge — and note ridge fixes it *literally*, by
adding $\lambda I$ to the diagonal, which guarantees invertibility for any $\lambda > 0$. The
statistical fix and the linear-algebra fix are the same fix.

The efficient move is that a single residuals-vs-fitted plot checks three of the four at once — you
want a structureless horizontal band.
</details>

<details>
<summary><b>10. 🎯 stretch — Massively over-parameterised neural nets should be catastrophic per the bias–variance curve. They aren't. What's going on?</b></summary>

Marked stretch because the honest answer includes "this is unsettled".

The observed phenomenon is **double descent**. As capacity grows, test error follows the classical
U-curve up to the *interpolation threshold* — where the model has just enough capacity to fit the
training data exactly — and peaks there. Then, as capacity grows further, it **falls again**, often
below the classical minimum.

The usual account is that in the over-parameterised regime there are many parameter settings that
fit the training data perfectly, and the optimiser doesn't pick among them randomly — SGD has an
implicit bias toward low-norm / "simpler" solutions. So effective complexity stops tracking parameter
count, and parameter count stops being the right x-axis.

I'd be careful about two things. The classical trade-off is still exactly right for the models in
this lecture — linear, regularised linear, trees, KNN, SVMs — and I wouldn't carry double descent
into a conversation about ridge regression. And the theory is genuinely contested — the *mechanism*
is still debated even though the Belkin et al. (PNAS 2019) and Nakkiran et al. (ICLR 2020) citations
themselves are confirmed (see §9's callout for full references).

The practically useful version: **stop using parameter count as a proxy for complexity**, and measure
the thing you care about — the validation curve — directly.
</details>

### Depth probes

| Your answer | The probe | What they want |
|---|---|---|
| "Split first, preprocess after" | *"You did — but you cross-validated. Where's the leak?"* | If the scaler was fitted before CV, each fold's train has seen every other fold. It must live inside the `Pipeline`. |
| "More data fixes overfitting" | *"Always?"* | Only while the validation curve is still falling. If it's flat, more data is wasted; regularise instead. And more data never fixes bias. |
| "Ridge handles multicollinearity" | *"Mechanically, how?"* | $\lambda I$ inflates the diagonal of $X^\top X$, guaranteeing invertibility for any $\lambda > 0$. |
| "Total error = bias² + variance" | *"You dropped a term."* | Irreducible noise $\sigma^2$. It appears when you decompose against the *observed* noisy label rather than the true function. |
| "Logistic regression is linear" | *"So how would you fit a circular boundary?"* | Engineer $x_1^2 + x_2^2$ as a feature. Linear in the expanded space, circular in the original — same idea as a kernel. |
| "I'd use the test set to pick the threshold" | *"Would you?"* | No — that's a model decision. Threshold goes on validation. |
| "AUC is 0.98" | *"On a hard problem? Check your features."* | Near-perfect performance on a genuinely hard task is a leakage bug report, not a result. |
| "I used a 70/15/15 split" | *"What effect size can 15% detect?"* | Compute the SE. Size the test set from the difference you need to detect, not by reflex. |

### Whiteboard-ready derivations

**① The normal equation.**
```
L(w) = ‖y − Xw‖²  =  yᵀy − 2wᵀXᵀy + wᵀXᵀXw
∇_w L = −2Xᵀy + 2XᵀXw = 0
     ⟹  XᵀXw = Xᵀy
     ⟹  w = (XᵀX)⁻¹Xᵀy                                  ∎

Verify on x=[1..5], y=[2,4,5,4,5]:
  XᵀX = [[5,15],[15,55]]   Xᵀy = [20,66]   det = 50
  (XᵀX)⁻¹ = (1/50)[[55,−15],[−15,5]]
  w = (1/50)[1100−990, −300+330] = (1/50)[110,30] = [2.2, 0.6]
Matches the hand fit ŷ = 2.2 + 0.6x ✓
```

**② The bias–variance decomposition.**
```
E[(ŷ − y)²]
  = E[((ŷ − E[ŷ]) + (E[ŷ] − y))²]                add and subtract E[ŷ]
  = E[(ŷ − E[ŷ])²]  +  2E[(ŷ−E[ŷ])(E[ŷ]−y)]  +  (E[ŷ] − y)²
      └── Variance ──┘   └──── cross term ────┘   └── Bias² ──┘

Cross term:  (E[ŷ] − y) is CONSTANT → pulls out
             leaving E[ŷ − E[ŷ]] = 0
             ⟹ cross term = 0

  ⟹  MSE = Var(ŷ) + Bias(ŷ)²                            ∎

With noisy labels y = f(x) + ε:   MSE = Var + Bias² + σ²
```

**③ Sigmoid ⟺ log-odds, and the boundary.**
```
p = σ(z) = 1/(1 + e⁻ᶻ)
  ⟹ p(1 + e⁻ᶻ) = 1
  ⟹ e⁻ᶻ = (1 − p)/p
  ⟹ z = log( p / (1 − p) )        ← z IS the log-odds

So:  log[ P(y=1|x) / P(y=0|x) ] = wᵀx + b
     "linear model of the log-odds"

Boundary:  p = 0.5  ⟺  z = 0  ⟺  wᵀx + b = 0   → a hyperplane
σ is monotonic ⟹ ANY threshold gives z = c ⟹ a PARALLEL hyperplane
```

### Applied scenario — churn prediction for Amazon subscriptions

**Framing.** The retention team can call ~500 customers a week. So this is not "predict churn" — it
is **rank customers by churn risk and hand back the top 500**, which changes the metric from accuracy
to **precision@500** and makes calibrated probabilities valuable (they let you compute expected value
saved per call).

**Formulation**, made explicit because §3 says this is the highest-leverage step:
- **Population:** customers with ≥1 purchase in the last 180 days.
- **$Y$:** 1 if zero purchases in the **next 60 days**. Sixty days because 30 over-flags seasonal
  buyers and 180 gives the team no time to act.
- **$X$:** purchase recency/frequency, category diversity, support-contact count, tenure, delivery
  issues — **all windowed strictly to end on the prediction date**.
- **$\mathcal{H}$:** regularised logistic regression first. Not because it's the most accurate, but
  because the retention team will ask *why* a customer was flagged, and $e^{w_j}$ gives them an
  odds-ratio answer they can act on.

**Data discipline.** Temporal split, not random — train on months 1–9, validate on 10–11, test on 12
— because in production the model only ever predicts forward. Everything fitted lives in a
`Pipeline`. Before anything else I'd audit the feature list against the "does this exist at
prediction time?" test; on a churn problem `cancellation_reason` and any post-cutoff support-ticket
window are the two that always sneak in.

**Diagnosis before iteration.** Fit, then read the train/validation pair. If both errors are high and
close, the fix is features and capacity, and collecting more history is wasted. If there's a large
gap, I plot the learning curve to see whether more data will actually close it before proposing that
we buy any.

**Metric.** Precision@500 as the headline, since that's the operating point. AUC-PR as the
threshold-free summary — not ROC-AUC, since churners are maybe 5% and ROC would flatter us
(see [Part 2 §17](supervised-learning-02.md)). Reported as mean ± std across temporal folds, because a
single number on one quarter is not evidence.

**Failure modes.**
- **The intervention changes the label.** If the team calls a flagged customer and saves them, that
  customer now looks like a non-churner in next month's training data — so the model learns that its
  own high-risk signals *predict retention*. This is the nastiest failure here. Fix: hold out a
  random control group who are never called, and train on them.
- **Temporal drift** — a pricing change or a competitor launch shifts behaviour. Monitor the input
  distribution and retrain on a rolling window.
- **Proxy leakage** — if "previously flagged" ends up in the feature set, the model learns to predict
  its own past output.
- **Calibration drift** — the expected-value calculation depends on the probabilities being real, so
  I'd monitor a reliability curve, not just AUC.

**What I'd ship first.** Regularised logistic regression on windowed features, temporal split, output
as a ranked list with the top 500 per week, plus a randomised control group from day one. It's
explainable, it's deployable in a week, and — critically — the control group is what makes it
possible to ever measure whether the programme works. A better model added later is easy; a control
group added later is not, because you've already contaminated the data.

### Leadership Principles tie-in

**Dive Deep.** This entire lecture is Dive Deep as a method. *Concretely:* seeing AUC 0.95 on the
delivery-lateness model, not celebrating, and tracing it to `delivery_exception_code` — a field
populated only *after* the failure. The instinct to distrust a good number and go find out where it
came from is the whole principle.

**Insist on the Highest Standards.** Sizing the test set from the effect you need to detect *before*
running the experiment, and then declining to claim a 0.8-point win that sits inside a ±1.5-point
confidence interval. The standard isn't "did the number go up", it's "would this replicate".

**Are Right, A Lot.** Not about being confident — it's about calibration. Reporting "0.86 ± 0.02
across 5 folds" instead of "0.86", and stating "we're at 8% error and the label-noise floor is ~6%,
so there are at most 2 points left" rather than promising open-ended improvement. Being right a lot
requires knowing how right you actually are.

---

## Glossary

| Term | Definition |
|---|---|
| **Bagging** | Train models in parallel on bootstrap resamples and average. Reduces **variance**. |
| **Bias (statistical)** | Error from a model too simple to represent the truth: $(\mathbb{E}[\hat y] - y)^2$. |
| **Boosting** | Train models sequentially, each on the previous one's errors. Reduces **bias**. |
| **Concept drift** | The input→label *relationship* changes over time. |
| **Cross-validation** | Rotate the validation split across $k$ folds; average. Gives an error bar for free. |
| **Data drift** | The input distribution changes, but the relationship doesn't. |
| **Data leakage** | Information reaching the model that won't exist at prediction time. |
| **Decision boundary** | Where the predicted class changes. For logistic regression, $\mathbf{w}^\top\mathbf{x}+b = 0$ — a hyperplane. |
| **Double descent** | Test error rises to a peak at the interpolation threshold, then falls again as capacity grows. Breaks the classical U-curve. |
| **Elastic Net** | L1 + L2 penalties together. Sparsity of lasso, stability of ridge. |
| **Expectation $\mathbb{E}[\cdot]$** | Long-run average. In bias–variance, averaged over *training sets*. |
| **Feature ($X$)** | An input variable. |
| **Homoscedasticity** | Constant error variance across the range of $\hat y$. OLS assumption 3. |
| **Hypothesis space $\mathcal{H}$** | Every function the model class can represent. Training searches inside it and can never leave. |
| **Hyperparameter** | Set by you before training; the algorithm never updates it ($\lambda$, degree, split ratio). |
| **Irreducible noise ($\sigma^2$)** | Randomness in the labels themselves. A hard floor on achievable error. |
| **Label ($Y$)** | The target variable. |
| **Lasso (L1)** | Penalty $\lambda\sum\lvert w_j\rvert$. Sets coefficients **exactly** to zero → feature selection. |
| **Learning curve** | Error vs. **training-set size**. The tool that tells you whether more data will help. |
| **Logit** | The raw linear score $\mathbf{w}^\top\mathbf{x}+b$; equals the log-odds. |
| **Log-odds** | $\log\frac{p}{1-p}$. Logistic regression is linear in this quantity. |
| **MLE** | Choose parameters maximising the probability of the observed data. For a Bernoulli label, equals minimising BCE. |
| **Multicollinearity** | Features that are near-linear-combinations of each other. Makes $X^\top X$ near-singular. |
| **Normal equation** | $\mathbf{w} = (X^\top X)^{-1}X^\top y$. Exact OLS solution; $O(d^3)$, so it doesn't scale. |
| **Odds ratio** | $e^{w_j}$ — the multiplicative effect on the odds per unit increase in $x_j$. |
| **OLS** | Ordinary Least Squares: minimise $\sum(y_i - \hat y_i)^2$. |
| **Overfitting** | Low training error, high validation error. High variance. |
| **Parameter** | Learned from data by the training algorithm ($w$, $b$). |
| **Problem formulation** | Choosing $X$, $Y$ and $\mathcal{H}$. The highest-leverage step. |
| **Regularisation** | Adding a complexity penalty to the objective. Trades bias for variance. |
| **Residual** | $y_i - \hat y_i$. With a fitted intercept, they sum to exactly zero. |
| **Ridge (L2)** | Penalty $\lambda\sum w_j^2$. Shrinks toward zero, never reaches it. Fixes multicollinearity. |
| **Sigmoid** | $\sigma(z) = 1/(1+e^{-z})$. Maps $\mathbb{R} \to (0,1)$. $\sigma(0) = 0.5$. |
| **Soft thresholding** | The lasso solution: $\text{sign}(w)\max(0, \lvert w\rvert - \lambda)$. |
| **Stratified split** | A split preserving class proportions. Essential when classes are imbalanced. |
| **Temporal split** | Train on the past, evaluate on the future. Mandatory for time-dependent data. |
| **Test set** | Touched **once**, at the end. Every additional look degrades it into a validation set. |
| **Underfitting** | High training *and* validation error. High bias. |
| **Validation set** | Used to choose hyperparameters and models. Spent slowly over many decisions. |
| **Variance (statistical)** | Error from over-sensitivity to the particular training sample: $\mathbb{E}[(\hat y - \mathbb{E}[\hat y])^2]$. |
| **VIF** | Variance Inflation Factor. > 5–10 signals multicollinearity. |

---

## Check yourself

Answer without looking. The section number tells you where to go.

1. State the three learning paradigms and classify each of: customer segmentation · delivery route
   selection · predicting tomorrow's demand. *(§1)*
2. Why is the ML workflow drawn as a circle? Give a concrete three-loop trace. *(§2)*
3. What is a hypothesis space? Give one function that is **not** in a linear model's. *(§3)*
4. You have 8,000 examples. Split them and justify the test-set size in terms of the smallest
   improvement you could detect. *(§5)*
5. Why isn't train/test enough? Quantify the inflation from selecting the best of 30 models. *(§5)*
6. Name all four leakage mechanisms. For each, give the symptom you would actually observe. *(§6)*
7. You standardise, then cross-validate. Where exactly is the leak, and what's the fix? *(§6)*
8. Training error 4%, validation error 5%. Overfitting or underfitting? What do you do? *(§8)*
9. A degree-6 polynomial fits 7 points with zero error. Why is that not good news? *(§8)*
10. Derive $\mathbb{E}[(\hat y-y)^2] = \text{Var} + \text{Bias}^2$. Why does the cross term vanish?
    *(§9)*
11. Model P predicts $[6, 6, 6]$ where the truth is 5; model Q predicts $[2, 5, 8]$. Compute bias²,
    variance and total for each. Which is better, and does that surprise you? *(§9)*
12. Where did the "irreducible noise" term go in the boxed formula, and how do you get it back?
    *(§9)*
13. Sketch the learning curves for a high-bias and a high-variance model. Which one justifies buying
    more data? *(§10)*
14. Which of these fix bias and which fix variance: more data · more features · early stopping ·
    larger $\lambda$ · a deeper tree? *(§10)*
15. Fit OLS by hand to $x=[1,2,3,4]$, $y=[3,5,6,9]$. Then verify via the normal equation. *(§12)*
16. Why must the residuals sum to zero, and what does it mean if yours don't? *(§12)*
17. State the four OLS assumptions. For each: what breaks, how you detect it, how you fix it. *(§12)*
18. Which single plot checks three of the four assumptions, and what should it look like? *(§12)*
19. With $\sum x_iy_i = 40$ and $\sum x_i^2 = 20$, give the ridge coefficient at
    $\lambda = 0, 5, 20, 1000$. Does it ever reach zero? *(§13)*
20. Prove that lasso can zero a coefficient and ridge cannot, from the penalty derivatives. *(§13)*
21. Why does the L1 diamond produce sparsity where the L2 circle doesn't? *(§13)*
22. You have 50 features, 20 of them near-duplicates of each other. Ridge, lasso, or elastic net?
    Why? *(§13)*
23. Give three reasons not to run OLS on a binary target. Which is fatal? *(§15)*
24. $\mathbf{w} = (1.2, -0.4)$, $b = -0.5$, $\mathbf{x} = (1, 2)$. Compute $z$, $\hat p$, and the
    prediction at thresholds 0.5 and 0.7. *(§15)*
25. Show that $z$ equals the log-odds, and use it to state what a coefficient of $-0.7$ means in
    plain English. *(§15)*
26. Why is logistic regression's decision boundary linear despite the sigmoid being non-linear? How
    would you get a circular one? *(§15)*

---

## Going deeper

Ranked by return per hour.

### Read first

1. **James, Witten, Hastie & Tibshirani — *An Introduction to Statistical Learning* (ISLR), ch. 2–4
   and 6.1–6.2.** `intro` · Free PDF from the authors. This is the **exact** book for this lecture:
   ch. 2 is bias–variance and model assessment, ch. 3 is linear regression and its assumptions, ch. 4
   is logistic regression, ch. 6 is ridge and lasso. If you read one thing, read this. Every figure
   in this deck has a more careful cousin in it.

2. **Hastie, Tibshirani & Friedman — *The Elements of Statistical Learning* (ESL), ch. 3 and 7.**
   `hard` · Also free. The graduate version of ISLR. Ch. 3.4 has the definitive treatment of ridge
   and lasso including the soft-thresholding derivation in §13; ch. 7 is the rigorous bias–variance
   and model-selection chapter.

3. **scikit-learn User Guide §3.1 (cross-validation) and §6.1 (pipelines).** `intro` · Read these
   two together, specifically for the worked demonstration of how preprocessing outside a pipeline
   leaks during cross-validation. It is the §6 leakage material with runnable code, and it will
   change how you write notebooks.

### Deeper on specific pieces

4. **Sebastian Raschka — "Model Evaluation, Model Selection, and Algorithm Selection in Machine
   Learning" (a four-part series, also on arXiv:1811.12808).** `solid` · The most thorough accessible
   treatment of exactly the §5 material: why three splits, nested cross-validation, confidence
   intervals on metrics, and how to compare two models honestly. ✅ Confirmed — arXiv number verified
   directly against the paper's own abstract page.

5. **Kaufman, Rosset & Perlich — "Leakage in Data Mining: Formulation, Detection, and Avoidance"
   (KDD 2011, with a later ACM TKDD 6(4) journal version).** `solid` · The paper that made leakage a
   named, systematic concept rather than folklore, with real case studies from data-mining
   competitions. Worth it for the taxonomy alone. ✅ Confirmed venue and year via ACM/DBLP.

6. **Belkin, Hsu, Ma & Mandal — "Reconciling modern machine-learning practice and the classical
   bias–variance trade-off"** (PNAS 116(32), 2019, pp. 15849–15854). `hard` · The double-descent
   paper referenced in §9's caveat. Read it *after* you're solid on the classical picture, not before.
   ✅ Confirmed authors, venue, and year via PNAS.

### Do, don't just read

7. **Implement OLS three ways on the same data and check they agree.** `intro` · (a) the
   $\sum(x-\bar x)(y-\bar y)/\sum(x-\bar x)^2$ formula, (b) the normal equation with an explicit
   matrix inverse, (c) `np.linalg.lstsq`. Then make two features nearly collinear and watch (b) blow
   up while (c) survives — that's §12's multicollinearity assumption failing in front of you, and
   it's also why nobody inverts the matrix explicitly.

8. **Plot a learning curve on a dataset you already have.** `intro` ·
   `sklearn.model_selection.learning_curve` does it in five lines. Do it for a model you know is
   underfitting and one you know is overfitting, and see the two shapes from §10 with your own data.
   This is the single most transferable habit in the lecture.

9. **Build the regularisation path.** `solid` · Fit ridge and lasso across
   $\lambda \in \{10^{-4} \ldots 10^{2}\}$ on a 20-feature dataset and plot every coefficient against
   $\log\lambda$. You will see lasso's coefficients hit exactly zero one by one while ridge's decay
   smoothly and never arrive — §13's central claim, as a picture you generated.

10. **Deliberately leak, then measure the damage.** `solid` · Take a clean pipeline, record the
    honest score, then introduce each of §6's four leaks one at a time and record the inflated
    score. Knowing that scale-before-split buys you ~2 points — exactly the size of a result you'd
    celebrate — is the thing that makes you careful forever.


