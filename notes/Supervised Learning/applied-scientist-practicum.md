# Applied Scientist practicum — make supervised learning operational

> **Purpose.** Parts 1–3 teach the model-level mechanics. This practicum is the missing bridge from
> “I can optimise F1” to “I can make and defend a product decision as an Applied Scientist.” Complete
> it after `supervised-learning-03.md`; it intentionally reuses its split, calibration, pipeline and
> uncertainty discipline rather than introducing another collection of algorithms.

## What interviewers and strong teams actually look for

An Applied Scientist is not hired for naming a model. They are hired to turn an ambiguous product
question into a measurable intervention, choose an honest evaluation, understand the failure modes,
and communicate a decision with evidence. For every project, be able to answer all eight questions
below without slides:

| Question | Evidence that makes the answer credible |
|---|---|
| What decision changes because of this model? | A named user, action, decision threshold or top-$K$ budget, and the cost of FP/FN. |
| What is the target, and when is it observable? | An operational label definition, label delay, observation window, and point-in-time feature cutoff. |
| What is the right offline metric? | A metric tied to the decision: PR-AUC/recall at capacity for triage, NDCG/Recall@$K$ for ranking, calibration for expected-value decisions. |
| Why should an offline win ship? | A pre-registered online hypothesis, primary metric, guardrails, minimum detectable effect, and randomisation unit. |
| Can the comparison be trusted? | Time-appropriate split, leakage audit, baseline, confidence interval, and a paired comparison where applicable. |
| Will it work in the product? | p50/p95 latency, throughput, cost, dependency and fallback requirements. |
| Who could be harmed or excluded? | Slice metrics, proxy-feature review, appeal/fallback path, and an explicit statement of what cannot be inferred. |
| What happens after launch? | Data/label/performance/calibration monitoring, alert thresholds, owner, rollback and retraining policy. |

💡 **Key insight.** A useful default model with a rigorous decision and experiment plan is more senior
than a complex model with only an offline score. Use complexity only after the evaluation harness is
credible.

## Four gaps to close alongside these notes

### 1. Experimental science and causal thinking

Predictive accuracy answers, “who is likely to do $Y$?” Product impact asks, “what happens if we do
$A$?” These are different questions. A churn model can accurately identify customers likely to leave
and still fail to prove that a discount prevents churn: high-risk customers may receive the discount
and leave anyway. The treatment changes the outcome and creates selection bias.

For each proposal, write a one-sentence causal hypothesis: **“Showing intervention $A$ to eligible
unit $u$ increases primary metric $M$ over window $W$, without worsening guardrails $G$.”** Then state:

- the unit of randomisation (customer, session, listing, seller, or region),
- the eligibility criteria set *before* assignment,
- the treatment and control experience,
- the primary metric, direction, and guardrails (e.g. cancellations, latency, complaints),
- the analysis window and contamination risk, and
- the decision rule: ship, iterate, or stop.

Do not peek daily and declare victory at the first $p < 0.05$. Fix the duration and minimum detectable
effect before launch, or use a sequential design with an explicitly chosen stopping rule. Report an
effect size and confidence interval, not only a p-value. For networked products, discuss interference:
one seller’s treatment can affect another seller’s demand, invalidating independent-user assumptions.

### 2. Retrieval and ranking, not only classification

Many Amazon-shaped problems have a candidate-generation stage followed by a ranker. A binary label is
often an approximation to a scarce exposure decision: which 20 products, ads, documents or cases should
be shown first?

| Layer | Question | Typical metric | Critical failure |
|---|---|---|---|
| Retrieval | Did the true/useful item enter the candidate set? | Recall@$K$ | A perfect ranker cannot recover an omitted item. |
| Ranking | Are the best candidates ordered first? | NDCG@$K$, MRR, Precision@$K$ | Optimising AUC can ignore the top-of-list experience. |
| Decision | Should this item be acted on at all? | calibrated expected value, precision at capacity | A score is treated as a probability without calibration. |

> **NDCG@$K$** (Normalised Discounted Cumulative Gain) — how good the ordering of the top $K$ results
> is, not just whether the right items are present. *In everyday words:* like Recall@$K$, but it also
> cares *where* in the list a good item lands — a relevant item at rank 1 counts for more than the
> same item at rank 10. *Concretely:* each item's relevance is discounted by $1/\log_2(\text{rank}+1)$
> before summing, then divided by the score of the best possible ordering (the "ideal DCG") so the
> metric sits in $[0,1]$. *Why it exists:* a binary "did we retrieve it" metric can't distinguish a
> ranker that puts the best item first from one that buries it at rank 20 — NDCG can.
>
> **MRR** (Mean Reciprocal Rank) — averaged over queries, $1/(\text{rank of the first relevant
> item})$. *In everyday words:* "how far down the list did I have to scroll before I saw one good
> result?", averaged across many queries. *Concretely:* if the first correct answer is at rank 1,
> that query scores 1.0; at rank 4, it scores 0.25; average those scores across all queries. *Why it
> exists:* for tasks where only the *first* correct answer matters (a single autocomplete suggestion,
> the first relevant support article), MRR is a cleaner match to the user experience than NDCG, which
> credits every relevant item in the list.

For the practicum project, define a capacity-constrained metric: “Among the 500 items a team can review
each day, maximise expected prevented harm” is clearer than “maximise F1.” Include a no-model baseline
(recency, popularity, rules, or current production logic), because an ML baseline is not a business
baseline.

### 3. Production ML and reproducibility

The `Pipeline` lesson in Part 3 is the start, not the finish. A deployable project needs a data contract:
feature names/types/ranges, timestamp availability, null policy, owners, and behaviour on unseen values.
Every training run should record code version, data snapshot/query, feature version, random seed,
configuration, metrics by slice, artifact URI, and approver. This is how a surprising metric becomes
debuggable rather than folklore.

At serving time, specify the latency budget as a percentile: “p95 < 30 ms” is actionable; “fast” is not.
Name a fallback (rules, cached ranking, human review, or no action) and test it. Avoid training-serving
skew by sharing transformations or by validating online features against their offline definitions.

### 4. Responsible and robust ML

Never claim a model is fair because its aggregate metric improved. Inspect relevant, legally and
ethically appropriate slices; compare sample sizes and uncertainty; and identify whether a feature is a
proxy for protected status or access. A gap can arise from representation, label quality, measurement,
or the decision policy—not only the model.

For high-impact predictions, keep a human appeal/review route, avoid using the score beyond its stated
purpose, and document known non-use cases. A model card is a compact record of intended use, data and
labels, metrics/slices, limitations, monitoring, owners, and rollback. It is an engineering handoff,
not compliance theatre.

## The portfolio capstone — one deep case study beats five notebooks

Build one end-to-end **capacity-constrained risk or quality-ranking** project. You may use a public
tabular dataset, but write the product framing as if the system will allocate a limited daily review
queue. Do not invent production results; label assumptions clearly.

### Required deliverables

1. **One-page design memo.** Problem, user decision, target and label delay, constraints, baseline,
   success metric, guardrails, experiment hypothesis, and launch decision rule.
2. **Reproducible training pipeline.** A point-in-time split; all preprocessing inside a pipeline;
   a simple baseline and a tuned model; seed/config/data version captured; no test-set selection.
3. **Evaluation report.** Overall and slice metrics with confidence intervals; PR and calibration plots;
   threshold/top-$K$ table tied to daily capacity; at least five manually inspected errors.
4. **Launch and monitoring plan.** Data quality, distribution, delayed-label performance and calibration
   monitors; alert thresholds; on-call owner; retraining trigger; rollback/fallback.
5. **Six-minute verbal defence.** State the decision first, then evidence, trade-off, caveat, and next
   experiment. Record yourself. If you cannot explain it simply, the project is not yet interview-ready.

### Score yourself before calling it complete

| Dimension | 0 — not ready | 1 — credible | 2 — exceptional |
|---|---|---|---|
| Framing | “Predict $Y$.” | Decision, metric and costs explicit. | Counterfactual, capacity and guardrails explicit. |
| Validity | Random split and one score. | Leakage audit, temporal logic, CI. | Paired/slice analysis and clear limitations. |
| Modelling | One complex model. | Baseline plus tuned pipeline. | Simplicity justified; calibration and error analysis drive changes. |
| Product | Offline result only. | Latency/fallback/monitoring named. | Online experiment and rollback are operationally specified. |
| Communication | Tool-first walkthrough. | Clear six-minute narrative. | Anticipates objections and changes a decision with evidence. |

Target **8/10**, with no zero. A model with 0.02 more AUC does not compensate for an invalid split or
no rollout plan.

## Interview drill — answer in this order

When given an open-ended ML case, use this reliable sequence:

1. Clarify the user, action, capacity constraint and horizon.
2. Define the label and audit whether features exist before the prediction time.
3. Select a simple baseline and split strategy that matches production.
4. Choose offline metric(s), threshold or $K$, and uncertainty method.
5. State the first model and the error analysis that would justify iteration.
6. Design the online experiment: unit, primary metric, guardrails, duration/MDE, decision rule.
7. Explain serving, monitoring, safety/fairness slices, and fallback.
8. End with the trade-off and the next evidence you would collect.

🎯 **Answer habit.** Say “I would first verify…” rather than pretending missing information exists.
Senior answers surface uncertainty, make a bounded assumption, and state how it will be checked.

## Interactive specifications for the study environment

```interactive
type: simulator
title: Offline score versus online impact
concept: Prediction is not a causal effect
control: Adjust treatment targeting strength, treatment effectiveness, and label delay
observe: Offline PR-AUC can rise while the estimated treatment lift is zero or negative
insight: A model selects likely outcomes; randomised evaluation estimates the change caused by an intervention
fallback: Side-by-side table showing a high-risk targeted group, untreated control, and the selection-bias trap
```

```interactive
type: simulator
title: Capacity-constrained ranking
concept: Retrieval recall, ranking quality, and decision thresholds are separate layers
control: Adjust candidate recall, ranking quality, and daily review capacity K
observe: Missed candidates cap all downstream value; a fixed K makes top-of-list quality dominate aggregate accuracy
insight: Optimise the metric that matches the scarce action, not a convenient global score
fallback: Three-layer flow diagram with retrieval Recall@K, ranking NDCG@K, and calibrated decision value
```

```interactive
type: simulator
title: Monitoring incident drill
concept: Data drift, performance drift, calibration drift, and outages need different responses
control: Choose an incident signal and inspect its symptom, likely causes, owner, and safe action
observe: The correct response changes when inputs shift, labels degrade, probabilities miscalibrate, or serving fails
insight: Monitoring is a decision system with an owner and rollback path, not a dashboard of lines
fallback: Incident-response table mapping each signal to diagnosis, action, and escalation condition
```

## Check yourself

1. Why does a high-AUC churn model not prove that an offer prevents churn?
2. A review team can inspect 300 of 1 million listings per day. Which metric and decision table would you present, and why is accuracy inadequate?
3. What timestamp rule prevents point-in-time leakage? Give one feature that commonly violates it.
4. Name three cases where a random train/test split is invalid or misleading.
5. What are the randomisation unit, primary metric, two guardrails, and stop rule for your capstone experiment?
6. Why can a ranker with excellent NDCG still have poor user impact?
7. List the four monitor categories and one action each: data quality, data drift, delayed-label performance, calibration/serving health.
8. What evidence would make you choose a simpler model over a better offline score?

## Definition of “ready for the next module”

You are ready to move on when you can reproduce a project from raw data to a decision memo, defend its
evaluation for ten minutes, identify the next experiment, and explain a safe rollback. Keep this file
alive: replace assumptions with evidence as you build the capstone, and link every claim to an artifact
you could show an interviewer.
