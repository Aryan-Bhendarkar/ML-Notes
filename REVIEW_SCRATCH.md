> ✅ **STATUS: ALL FINDINGS BELOW HAVE BEEN APPLIED.** This file is kept as an audit trail of what
> was found and fixed across the consolidated edit pass, not as a pending to-do list. Additional
> fixes applied beyond what's logged below: instructor names corrected in all three module
> `README.md` index files (Causal Inference, Reinforcement Learning, Agentic AI), stray OCR working
> files (`ocr_rl24/25/26.txt`, `ocr_ai27/28/29.txt`) removed from the Reinforcement Learning and
> Agentic AI folders, word counts updated across all three READMEs, and the Agentic AI README's
> "Capture quality" table corrected (it previously understated the true `output/` raw-frame counts
> by ~45%, citing a different/coarser capture).

# Review scratch — notes written by other agent (Causal Inf. Part 3, RL 24-26, Agentic AI 27-29)

Purpose: catalog concrete, actionable fixes per file before doing a single consolidated edit pass.
Findings are grouped by file. Each item cites the section/line to fix and what to change.
Severity: 🔴 factual error/fabrication (must fix) · 🟠 real content gap (should fix) ·
🟡 pedagogy/polish (nice to fix).

---

## `notes/Causal Inference/causal-inference-03.md` (Lecture 23) — REVIEWED

Source: `output/Lecture_23 - Module 7 Causal Inference Part 3/` (102 raw frames). Cross-checked
against slides 2, 30, 45, 49, 55, 89, 93 at full resolution + both contact sheets (full 102-frame
structure mapped).

### 🔴 Factual errors — must fix

1. **Instructor name is wrong.** File says "Pranitaka Kandelar" (hedged with a verify-flag). The
   slide nameplate (visible on slides 2, 49, 55, 89, 93 etc.) clearly reads **"Pranita Khandelwal"**.
   Fix the header note and remove the verify-flag — this is now confirmed, not approximate.

2. **TarNet citation is fabricated.** File's Going Deeper item 2 attributes TarNet to "Shah, Bose,
   Chen (2017)" — this paper does not exist / is not what the slide says. Slide 49/55 footer reads:
   **"Johansson, Shalit & Sontag (2016) · Shalit et al. (2017), ICML"**, and slide 55's body text
   states directly: *"TARNet (Shalit, Johansson & Sontag, 2017) = a shared trunk Φ plus two outcome
   heads."* TarNet and CFRNet are **the same paper/author team**, not two separate works. Fix:
   - §5.2's TarNet definition block: change author attribution to **Shalit, Johansson & Sontag
     (2017)** (companion: Johansson, Shalit & Sontag 2016 for the earlier balanced-representation
     framing).
   - §5.3's CFRNet attribution is approximately right already but should be reconciled so both
     methods cite the *same* paper/team, not implied as separate.
   - Going Deeper items 2 and 3: merge or clearly cross-reference — do not present as two
     independent citations by different authors.

### 🟠 Real content gaps — should fix

3. **DragonNet is entirely missing.** Slide 93 ("Comparison of HTE estimators") lists **DragonNet**
   (deep semiparametric learning: Φ(x), μ₀(x), μ₁(x), e(x); "TARNet + propensity head + targeted
   regularization") as a distinct method in the same family as TarNet/CFRNet. It appears nowhere in
   the notes — not in §5, not in the glossary, not in the comparison content. Add a short §5.4
   introducing DragonNet (what the propensity head buys over CFRNet — likely targeted regularization
   for the propensity-outcome interaction) and add it to the glossary.

4. **The actual "Comparison of HTE estimators" table (slide 93) was never transcribed.** The notes'
   §11 substitutes an invented "DAG edge targeted" table that is a reasonable synthesis but is *not*
   the real slide content, and it drops real, precise information the slide has: a **Method ×
   Generation × Models-learned × Main-objective × Neyman-orthogonal? × Doubly-robust? ×
   Handles-imbalance** grid covering S-learner, T-learner, X-learner, R-learner, DR-learner, Causal
   forest, TARNet, CFRNet, **and DragonNet** — including the pointed footnote that *"Orthogonality
   holds for the local-centering (residual-on-residual) causal forest of Athey, Tibshirani & Wager
   (2019), not the original Wager & Athey (2018) formulation."* This footnote is a genuinely
   sophisticated, citable nuance about causal forests that's completely absent from §4.2's causal
   forest discussion. **Action:** add the real table (reproduced faithfully) either replacing or
   supplementing the current §11 table, and pull the Athey/Tibshirani/Wager (2019) vs. Wager/Athey
   (2018) distinction into §4.2 as a ⚠️ callout.

5. **The causal-tree splitting-criterion worked example (slides 45–46) is missing entirely.** This
   is a concrete, numeric, high-value teaching example that directly demonstrates §4.1's abstract
   "maximize heterogeneity between child nodes" claim:
   - Setup: discount coupon (treatment) → customer spend (outcome), parent node τ̂ = 5%.
   - **Split A — on "weeks since last purchase":** children have τ̂ = 1% and τ̂ = 12% (an 11-point
     spread).
   - **Split B — on "device" (mobile/web):** children have τ̂ = 4.8% and τ̂ = 5.2% (a 0.4-point
     spread).
   - The causal tree favors **Split A** — it's the split that actually reveals differential treatment
     effect; Split B's near-identical child leaves tell you device doesn't moderate the coupon's
     effect at all.
   **Action:** add this as a 🧪 worked example directly inside §4.1, right after the splitting
   criterion is stated in words — it is exactly the concrete instance the current prose is missing
   (the whole section currently has zero worked numbers).

6. **Künzel et al. (2019), PNAS is never cited**, despite being the explicit footer citation for the
   entire meta-learner section (S/T/X-learner, confirmed directly on slide 30's footer). This is the
   actual X-learner paper ("Künzel, Sekhon, Bickel, Yu — Metalearners for estimating heterogeneous
   treatment effects using machine learning," PNAS 2019) and should replace or supplement whatever
   general background citation currently stands in for meta-learners in Going Deeper. Add it as item
   1 (or near top) of Going Deeper, marked as directly slide-sourced, not inferred.

### 🟡 Pedagogy / polish

7. **X-Learner stage-count inconsistency — CONFIRMED, fix to "four-stage."** §3.3 header text says
   *"A five-stage procedure"* but the body only enumerates Stage 1 through Stage 4. Checked directly
   against slide 34 ("X-learner: share information across arms"): the slide's own bullets are labeled
   **Stage 1** (outcome models), **Stage 2** (impute pseudo-effects), an unlabeled "regress the
   imputed effects on X" bullet, and **Stage 4** (combine) — four stages total, matching the notes'
   own body content exactly. Fix: change "five-stage" → "four-stage" in §3.3's opening sentence.

### 🔴 Additional factual error — mechanism explanation is backwards (self-contradicts its own formula)

9. **§3.3's "why the propensity weighting works" explanation has the weighting direction backwards,
   and contradicts its own stated formula.** The notes give:
   $$\hat\tau(x) = e(x)\cdot\hat\tau_0(x) + (1-e(x))\cdot\hat\tau_1(x)$$
   then claim: *"when $e(x)$ is low (few treated units at $x$), the estimate leans on $\hat\tau_0(x)$
   — which was trained on the larger control group and is therefore more reliable."* This is
   arithmetically backwards: when $e(x)$ is low, the weight on $\hat\tau_0$ (which is $e(x)$ itself)
   is *low*, and the weight on $\hat\tau_1$ (which is $1-e(x)$) is *high* — so the formula actually
   leans toward $\hat\tau_1$, not $\hat\tau_0$, when treated units are scarce. Confirmed against
   slide 34: the real mechanism is that $\hat\tau_1(x)$ is built from treated units' pseudo-effects
   $\tilde D^1 = Y-\hat\mu_0(X)$, and $\hat\mu_0$ (the control outcome model) is well-estimated
   *precisely because* controls are abundant when $e(x)$ is low — so $\hat\tau_1$'s pseudo-outcomes
   are high-quality (reliable counterfactual imputation) even though $\hat\tau_1$ itself is fit on
   few (treated) data points. The correct explanation: **the weight leans toward whichever
   $\hat\tau$'s pseudo-outcome imputation drew on the *better-estimated* nuisance model** — i.e.,
   toward $\hat\tau_1$ when controls are abundant (e(x) low), toward $\hat\tau_0$ when treated units
   are abundant (e(x) high). **Action:** rewrite this paragraph entirely; do not just swap the two
   labels — re-derive the direction from the formula and explain *why* via the pseudo-outcome
   construction (μ̂0 reliability → τ̂1 quality), not via "which arm has more raw rows."

8. Section §11's invented DAG-edge table is fine to *keep* as an additional synthesis/teaching aid
   (it's not wrong, just not slide-sourced) — but it should be clearly distinguished from the
   slide-sourced comparison table added per item 4, e.g. relabel it "§11a — a synthesis view" vs.
   "§11b — the lecture's own comparison table" so a reader doesn't mistake the author's own
   reorganization for a transcription.

### Not yet checked (lower priority, spot-check if time allows before final edit)

- R-Learner (§9) and DR-Learner (§10) formula transcription against slides ~80–85 (contact sheet
  labeled these "residualize, then fit the effect" and "robust if either nuisance model is correct"
  — formulas in the note look internally consistent with Part 2's AIPW notation, but were not
  individually re-verified pixel-for-pixel against the slide LaTeX).
- Cross-fitting section (§7.3) pseudocode block — plausible and consistent with standard DML
  presentations, not independently re-checked against slide 77–79's exact wording.
- §12 "Beyond Treatment Effects" frontiers section — broadly matches the contact-sheet thumbnails
  (robust/generalizable ML, causal generative modeling, causal RL, fairness/explainability, LLMs;
  cow-detector domain-shift example visible in thumbnails) but individual bullet sub-claims (e.g.
  "counterfactual credit assignment," "dynamic treatment regimes") were not verified word-for-word.

**Overall verdict for this file:** strong structurally and pedagogically (good worked examples
elsewhere, honest ⚠️ flags used correctly, interview questions are solid) — but it has one clear
fabricated citation (TarNet authors) that must be fixed before publishing, plus a real, avoidable
"slide fidelity" gap (missing DragonNet, missing the actual comparison table, missing the causal-tree
numeric example) that a teacher would flag as under-mining the "did you actually read every slide"
standard the pipeline requires.

---

## `notes/Reinforcement Learning/reinforcement-learning-01.md` (Lecture 24) — REVIEWED

Source: `output/Lecture_24 - Module 8 Reinforcement Learning Part 1/` (150 raw frames). Cross-checked
contact sheets (full 150-frame structure) + slides 78 (discount-factor demo), 86 (CartPole spec).

### 🔴 Factual error — must fix

1. **Instructor name — now confirmed, currently wrong/hedged.** File says *"Samu"* / *"Sayanbhu"*
   with a verify-flag. Slide 78's nameplate clearly reads **"Sayambhu Sen"** — full name confirmed,
   remove the hedge. **Bonus continuity note:** this is very likely the *same* instructor as Lecture
   18 (Sequential Learning Part 1), whose slide nameplate I confirmed directly as "Sayambhu Sen" while
   writing that file myself. Worth a one-line cross-reference in the header if it reads naturally
   ("same instructor as Lecture 18"), but not essential.

### 🟡 Pedagogy / polish

2. **Duplicated paragraph — copy-paste bug.** §5.2 (Markov property) has the *exact same* ⚠️ callout
   paragraph ("The Markov property is an assumption, not always true...") printed twice back-to-back
   (immediately consecutive blockquotes). Delete the duplicate.

3. **CartPole terminal conditions ($|\theta|>12°$, $|x|>2.4$, 500-step cap) are stated in §7.1 as if
   read off the slide, but slide 86 ("Example: Grid World MDP and Cartpole") only states states/
   actions/reward — no termination thresholds are shown on that slide.** The numbers themselves are
   correct (they match the real Gymnasium `CartPole-v1` defaults), so this isn't a fabrication in the
   harmful sense, but it's presented with the same confidence as slide-sourced content elsewhere in
   the file. **Action:** either find the actual slide that states these thresholds (check nearby
   slides 87–96, the CartPole interactive-demo range, before assuming it's not shown anywhere) or add
   a light "these are the standard Gymnasium CartPole-v1 defaults, not explicitly stated on this
   slide" caveat.

### Verified accurate (no action needed)

- The full "How Farsighted Is Your Agent?" discount-factor demo table (§7.3) was checked against the
  actual interactive-demo screenshots for γ = 0.68 (slide 78: G₀(Path A)=6.82, G₀(Path B)=5.58 —
  **exact match**) and the γ=0.9/0.99/0.1/0.0 rows are visible with matching values across slides
  61–84. This table is accurately transcribed — good work by the other agent here, no fix needed.
- RL milestones timeline (§4.2) matches slide 45 ("Milestones Timeline") exactly, including the 2022
  ChatGPT/RLHF and 2024 "large-scale reasoning models" rows.
- Frontmatter `slides: 73` (deduped count) vs. header prose "150 raw frames" is not a bug — matches
  this project's established convention of frontmatter using the deduped count while the prose cites
  the raw capture count actually used.

### Not yet checked (lower priority)

- §9–§13 (Bellman equations, policy iteration, value iteration) worked examples and derivations
  were not independently re-verified against the corresponding slides (~slides 97–150, "Part 5:
  Policies & Value Functions" and "Part 6: Bellman Equations" per the contact sheet) — the math is
  internally consistent and standard, low risk, but do a pass before final publish if time allows.

**Overall verdict:** structurally sound, only one real fix needed (instructor name) plus a
copy-paste duplication to clean up. Much better shape than the causal-inference file — no fabricated
citations found.

---

## `notes/Reinforcement Learning/reinforcement-learning-02.md` (Lecture 25) — REVIEWED

Source: `output/Lecture_25 - Module 8 Reinforcement Learning Part 2/` (134 raw frames vs. 70 in
`slides_deduped/`). Contact sheets built and read for all 134 raw frames.

### 🟠 Process/methodology flag — should prompt a broader re-check

1. **This file's own header states it was built from "70 deduped slides... with OCR verification
   and the full YouTube transcript," not the raw `output/` capture.** This directly contradicts the
   established project methodology (see project memory `slides-deduped-is-lossy`, and every other
   file in this course including my own — Lectures 18–23, RL Lecture 24 — which explicitly avoid
   `slides_deduped/` because dedup measurably drops or merges distinct slides, sometimes ~40% of
   content). Confirmed directly: `output/Lecture_25.../` has **134 raw frames** vs. **70** in
   `slides_deduped/` — a ~48% reduction, right in line with the dedup-loss pattern the project has
   already documented twice. **This means the file may have systematic, silent content gaps** that a
   raw-frame read would have caught, even though the OCR+transcript approach is a reasonable-sounding
   alternative safeguard. Practical outcome of my spot-check below: I did *not* find large blocks of
   entirely-missing content when scanning all 134 raw frames' structure, so the OCR+transcript
   approach seems to have caught most of the substance — but one real gap did surface (item 2 below),
   and the extremely detailed grid-world sweep-by-sweep numeric tables (raw frames 16–45, ~30 frames)
   are compressed into a single vague sentence in the notes (item 3) — exactly the kind of density
   loss dedup-based or transcript-only summarization tends to produce. **Recommend:** re-verify this
   file against the raw 134-frame capture before final publish, using the same procedure as every
   other lecture in this course, rather than treating the OCR+transcript method as equivalent.

### 🟠 Real content gap

2. **Thompson Sampling is completely missing.** The lecture's own closing summary slide (raw frame
   131/132, "Summary: The Complete Journey") states directly: *"Multi-Armed Bandits: Simplest RL,
   **Thompson Sampling dominates in industry**."* Thompson Sampling is a named, industry-standard
   bandit algorithm that the lecture itself flags as dominant in practice — yet it appears nowhere in
   §4 (exploration strategies: ε-greedy, softmax, UCB), §5 (bandits), the glossary, or anywhere else
   in the file. This is the same shape of gap as Lecture 23's missing DragonNet: a method the source
   material names explicitly in its own summary, silently dropped from the notes. **Action:** find
   the slide(s) that actually introduce Thompson Sampling (likely inside the bandits section, raw
   frames ~65–72) and add a proper subsection — Bayesian posterior sampling over arm-success
   probabilities, why it typically outperforms UCB/ε-greedy in practice, and its relationship to the
   other three strategies in §4.5's comparison table (which should grow a fourth row).

### 🟡 Pedagogy / enrichment opportunity

3. **The grid-world policy-evaluation demo (§3.1) is under-mined relative to how much real data the
   source provides.** Raw frames 16–45 (~30 frames) show a detailed sweep-by-sweep numeric
   walkthrough of iterative policy evaluation on a 3×3 grid — actual V-table values evolving sweep by
   sweep from all-zero toward converged values (visible directly in the contact sheet: sequences like
   0.00→-1.00→-1.23→1.79→...→+10 at the goal cell, with the start-cell arithmetic shown explicitly
   each sweep). The current notes compress this into one sentence ("starts near -20... converges
   after many sweeps"), which is a plausible paraphrase but wastes a large amount of good, concrete,
   already-worked material. **Action:** pull 2–3 actual sweep snapshots (e.g. sweep 1, sweep 3, final
   converged) with real numbers into a 🧪 worked example in §3.1 — this is exactly the kind of
   concrete numeric grounding the pipeline's teaching contract calls for and the source material
   already has, it just wasn't transcribed.

### Verified accurate (no action needed)

- DQN architecture, the two tricks (experience replay + target network), the deadly triad, and the
  four DQN variants (Double DQN/Dueling/Prioritized Replay/Rainbow) all check out structurally
  against raw frames 106–130 — no missing methods in this part of the deck, citations (van Hasselt
  2016, Wang 2016, Schaul 2016, Hessel 2017) match the "Key Improvements Since 2015" slide.
- SARSA vs. Q-learning cliff-style comparison (§7.3) matches the "SARSA vs Q-Learning" interactive
  demo slides (96–105) in substance.
- Agenda/section time-allocations on raw frame 5 match the notes' own section structure and ordering
  closely (DP → explore/exploit → bandits → TD/SARSA/Q-learning → on/off-policy → DQN).

---

## `notes/Reinforcement Learning/reinforcement-learning-03.md` (Lecture 26) — REVIEWED

Source: `output/Lecture_26 - Module 8 Reinforcement Learning Part 3/` (**83 raw frames** vs. **31**
in `slides_deduped/` — a 63% reduction, the worst dedup-loss ratio found in this review round).
Same "OCR + transcript, deduped slides" methodology note as Lecture 25 applies here — see that
file's item 1, which generalizes to this file too. Contact sheets built and read for all 83 frames.

### 🟠 Real content gap — PPO is used throughout but never actually explained

1. **PPO (Proximal Policy Optimization) is referenced constantly — in the glossary, 3+ interview
   questions, a depth probe about its clipping mechanism, the applied scenario, AND used as the
   fine-tuning algorithm in §9.5's RLHF pipeline — but is never once explained in the main body.**
   No section defines what PPO actually is or derives its clipped objective; a reader hits "PPO's
   clipping mechanism" in a depth probe having never been told what PPO clips or why. Checked the
   source: **slide 78** ("Example: Evaluation & Visualization," in the hands-on demo section) has a
   real, if brief, PPO definition box sitting right next to a matching DQN one:
   > **PPO (Proximal Policy Optimization)** — Type: On-policy, policy gradient. How it works:
   > Directly learns a policy (state → action probability). Uses a "clipped" objective to prevent
   > too-large policy updates, making training stable. Key params: `learning_rate=3e-4`,
   > `gae_lambda=0.95`, `clip_range=0.2`.

   **Action:** add a proper §7.5 or new §8 "PPO" section — even a short one — introducing the clipped
   surrogate objective $r(\theta)=\pi_\theta(a|s)/\pi_{\theta_{\text{old}}}(a|s)$ clipped to
   $[1-\epsilon, 1+\epsilon]$ (this exact ratio is already used, unexplained, in the file's own depth
   probe), *before* it's used in §9.5 and the interview/depth-probe content. This closes a real
   "used before defined" gap, not just a missing citation.

### 🟠 Real content gap — the hands-on demo's actual quantitative result is missing

2. **§12 "Hands-on Demo Summary" reports only a vague qualitative comparison** ("DQN: Low average
   reward, unstable pole angles" / "PPO: Converges to maximum reward, stable pole angles") when the
   source slides (73–80, "Example: Evaluation & Visualization") show **the actual head-to-head
   numbers**, plotted and stated directly:
   - **DQN:** −9.2 avg reward — *"Failed to learn... in 20k steps"* (exploration_fraction=0.2, 20%
     of training spent exploring).
   - **PPO:** 500.0 avg reward — *"Perfect score, after ~8k steps"* (learning_rate=3e-4,
     gae_lambda=0.95, clip_range=0.2).
   - Plus pole-angle-over-time plots comparing the two directly (slides 79–80).

   This is real, concrete, already-worked evidence for exactly the claim the notes are trying to make
   ("PPO converges, DQN doesn't") — reducing it to adjectives ("low," "unstable") when the source has
   precise numbers is a direct instance of the pipeline's "worked examples land... every example
   ending in a final number" standard being unmet. **Action:** rewrite §12 as a proper 🧪 worked
   example with the real numbers, hyperparameters, and the qualitative pole-angle-stability
   description as supporting color, not the entire content.

### Verified accurate / no action needed

- Policy gradient theorem derivation (§2.1), REINFORCE (§2.2–2.3), baselines (§3), actor-critic
  (§4–5), A3C (§6), GAE (§7) all check out structurally against slides 6–38 — no missing methods in
  this range, math is standard and correctly presented.
- Multi-agent RL three-paradigm table (§8) matches slides 41–45 ("Agent RL — Approaches": Cooperative
  CTDE/QMIX/MAPPO, Competitive, Self-Play) closely.
- RLHF five-stage pipeline, Bradley-Terry model, and the DPO/RLAIF/Constitutional AI "beyond RLHF"
  table (§9) match slides 46–56 well.
- Challenges table (§10) and frameworks table (§11) match slides 57–68 well.

**Overall verdict:** structurally sound and well-organized; the two gaps found (PPO never formally
introduced despite heavy reliance on it, and the demo's real numbers dropped) are both "compressed
away good source material" issues rather than fabrications — consistent with the OCR+transcript
methodology losing density even where it didn't lose whole topics.

---

## `notes/Agentic AI/agentic-ai-01.md` (Lecture 27) — REVIEWED

Source: `output/Lecture_27 - Bonus Module Agentic AI Part 1/` (53 raw frames vs. 23 deduped — 57%
loss, same dedup-based-file pattern as RL Lectures 25–26). Full 53-frame contact sheet read, plus
slides 17, 35, 47 at full resolution.

### 🟠 Major structural gap — an entire named architecture is missing

1. **"Plan-and-Execute" is completely absent from the notes, despite being one of only four
   architectures the lecture itself names as "Four ways to build an agent"** (slide 22/24: *"ReAct,
   Reflection, Plan-and-Execute, search over thoughts"*). The notes cover ReAct, Reflection, and Tree
   of Thought (§5.1–5.3) but skip Plan-and-Execute entirely — not in the body, not in the glossary,
   not in "putting it together." This is the single largest gap found in this review round: a whole
   named framework, with its own dedicated, richly worked slide (35, "Plan-and-execute"):
   - **Architecture:** a strong-model **Planner** drafts the whole plan in one call → a cheap-model
     **Executor** runs each step with tools → a **Replan** edge repairs on a failed step, looping back
     to the planner.
   - **The benchmark that makes the point unmissable:** a 3-block Blocksworld stacking problem, 600
     instances, zero-shot, plans checked by the VAL verifier. Success rate: **Fast Downward (classical
     planner) 100.0%**, **GPT-4 34.6%**, **Claude 3.5 Sonnet 54.8%**, **o1-preview 97.8%**.
   - **The punchline, stated directly on the slide:** renaming the blocks/actions (familiar names →
     obfuscated names) leaves the underlying problem *mathematically identical*, yet the LLM bars
     **collapse to ~0%** under obfuscation while the classical planner **still holds 100%**. *"The LLM
     is a good plan **proposer** but a poor plan **guarantor**."*

   **Action:** add a proper §5.2a (or renumber) "Plan-and-Execute" section with this architecture
   diagram and the full benchmark — it's exactly the kind of concrete, surprising, real-model-named
   evidence the pipeline's teaching contract wants, and currently zero such content survived into the
   notes for this entire architecture.

2. **LATS (Language Agent Tree Search) is missing.** The same "four ways" slide and the Tree-of-Thought
   section (slides 37–40) present **"ToT / LATS"** together, with an actual success-rate figure
   ("ToT 74%") on a game-of-24 tree diagram. The notes' §5.3 covers Tree of Thought only, with an
   invented-looking game-of-24 example that doesn't cite the slide's actual 74% figure. **Action:** add
   a short LATS mention (tree search + backtracking, but using a learned/value-guided policy over
   which branch to expand next, per its name) alongside ToT, and pull in the real 74% success figure.

### 🟠 Numeric fidelity concern — confidence-threshold table likely has invented intermediate values

3. **§6.2's confidence-threshold table's anchor point is correct but the surrounding rows don't
   reconcile.** Confirmed accurate against slide 47 ("Asking versus proceeding"): the optimal threshold
   **τ\* = 0.43** and **minimum total cost = 6** are both stated explicitly on the slide, matching the
   notes' "0.43 (sweet spot)" row. But checking a second concrete point on the same slide — at
   **τ=0.72**, the demo shows **9 human interrupts, 0 wrong actions, total cost 9** — this does not
   fit smoothly with the notes' adjacent row "τ=0.7 → 5 interrupts, 0 wrong, total cost 5" (a jump
   from 9 to 5 interrupts for a 0.02 change in τ is implausible given the sparse point cloud visible in
   the demo). This suggests the table's non-anchor rows (τ=1.0/0.7/0.2 and their interrupt/wrong/cost
   counts) were estimated or invented for illustration rather than read off actual demo states.
   **Action:** either re-derive the table from actual slider positions in the demo (screenshot 2–3 more
   τ values precisely) or clearly caveat the non-anchor rows as illustrative, not measured — don't
   present estimated numbers with the same confidence as the verified τ*=0.43/cost=6 anchor.

### 🔴 Instructor name — likely wrong, cross-reference from Lecture 28

5. **File says instructor is "Deep Nyak" (hedged "name approximate").** No name overlay was visible
   in any frame I checked in this deck, so I could not directly confirm/deny it here — but Lecture 28
   (same Agentic AI module, reviewed next) has a **clearly visible nameplate reading "Harsh
   Agarwal"** on its slides. Since both lectures are almost certainly delivered by the same
   instructor (same module, consecutive parts, same slide-deck house style), **"Deep Nyak" is very
   likely wrong and should probably be "Harsh Agarwal."** Action: re-check a wider sample of Lecture
   27's frames for a name overlay before finalizing; if none is found, change the attribution to
   "Harsh Agarwal (same instructor as Part 2, confirmed there)" rather than the invented-sounding
   "Deep Nyak."

### 🟡 Structural completeness — missing required pipeline sections

4. **This file is missing several sections `NOTES_PIPELINE.md` requires of every lecture file:** no
   "Before we start: what you need to know" prerequisites section, no "Applied scenario" (an
   Amazon-flavored end-to-end problem), no "Depth probes," and no "Whiteboard-ready derivations." The
   interview-prep section also has only 6 questions with none marked "[Combines concepts]," versus the
   8–12 with several combining questions the pipeline specifies. This file reads as noticeably thinner
   than Lectures 18–26 structurally, independent of the content-accuracy issues above. **Action:** add
   the missing sections before this file is considered complete — a Prerequisites section (LLM basics,
   tool-calling/function-calling as a concept, prompt engineering fundamentals would be reasonable
   candidates) and an Applied Amazon scenario are the highest-value additions.

### Verified accurate / no action needed

- The error-compounding math (§3.1, the 0.95ⁿ table) is fully correct — independently recomputed
  0.95^5≈77%, 0.95^10=60%, 0.95^20≈36%, 0.95^40≈12%, 0.95^100≈0.6%, and the 60%/0.6% anchor points
  match slide 17's interactive demo exactly. No fix needed here despite initially looking like a
  candidate for fabricated-number concern.
- The chatbot/copilot/agent spectrum (§1.1), autonomy ladder (§2), when-to-use-an-agent table (§4),
  and ReAct/Reflection substance (§5.1–5.2) all match the corresponding slides structurally.

---

## `notes/Agentic AI/agentic-ai-02.md` (Lecture 28) — REVIEWED

Source: `output/Lecture_28 - Bonus Module Agentic AI Part 2/` (52 raw frames vs. 21 deduped — 60%
loss). Full 52-frame contact sheet read, plus slide 46 at full resolution.

### 🔴 Factual error — instructor name

1. **Instructor name confirmed wrong.** File says *"Deep Nyak (name approximate)."* Slide 46's
   nameplate clearly reads **"Harsh Agarwal."** Fix directly (no more hedging needed) — and this
   also strongly suggests Lecture 27's identical "Deep Nyak" attribution should be corrected to the
   same name (see that file's item 5).

### 🟡 Structural completeness — same gap as Lecture 27

2. **Missing the same required sections as Lecture 27:** no "Before we start" prerequisites, no
   Applied Amazon scenario, no Depth probes, no Whiteboard-ready derivations. Interview prep has only
   7 questions, none marked "[Combines concepts]." **Action:** add these before publish — a
   prerequisites section here could reasonably cover: what a JSON schema is, what "context window"
   means at a basic level, and REST API basics (since tool-calling and MCP both assume the reader
   knows what an API call is).

### Verified accurate / no action needed

- Full-deck structural scan (52 frames) found **no missing named framework or method** analogous to
  Lecture 27's Plan-and-Execute gap — the notes' three-part structure (Tools §1–6, Memory §7–10, MCP
  §11–15) matches the deck's own section breaks (slides 1–24 tools, 22–35 memory, 36–51 MCP) closely,
  and no additional named pattern/algorithm was spotted in the thumbnails that isn't already covered.
- MCP adoption claim (§14.3: "created by Anthropic in 2024... adopted by OpenAI, Google, Microsoft,
  Amazon... native in Claude Code, VS Code, JetBrains") verified **word-for-word accurate** against
  slide 46's "Adoption" box.
- Tool-scaling cost/error-rate curves (§5), the escalation ladder (§6.1), and the four memory tiers
  (§8) all match their corresponding slides (13–16, 17–20, 25–26) structurally.

### Not yet checked (lower priority)

- The specific "~12 tools / ~100 tools / 1000+ tools" scaling thresholds in §5's table were not
  independently re-measured against the slide's axis labels (the curves are visible in the contact
  sheet but exact crossover points weren't read at full resolution) — low risk, since the qualitative
  ordering (direct < router < retrieval scaling) is clearly correct either way.

---

## `notes/Agentic AI/agentic-ai-03.md` (Lecture 29) — REVIEWED

Source: `output/Lecture_29 - Bonus Module Agentic AI Part 3/` (84 raw frames vs. 35 deduped — 58%
loss). Full 84-frame contact sheet read, plus slides 69 and 71 at full resolution.

### 🔴 Fabricated numbers — must fix

1. **§15.3's sandbox limits "(30s, 52MB)" do not appear anywhere on the source slide and appear to
   be invented.** Checked directly against slide 69 ("Contain the blast radius" / Sandboxing &
   Guardrails): the actual bullet reads only *"CPU, memory, and time limits"* — **no specific numbers
   are given anywhere on this slide.** The notes state "(30s, 52MB)" with the same confidence as
   genuinely slide-sourced content. "52MB" in particular is an implausibly small memory limit for a
   real code-execution sandbox (typical defaults run hundreds of MB to several GB), reinforcing that
   this was invented rather than misread. **Action:** remove the specific numbers or replace with
   "CPU, memory, and time limits (specific values are deployment-dependent — not given on the
   slide)."

### 🟠 Real content gaps

2. **The "Evaluation Benchmarks" block is only partially covered.** The notes mention SWE-bench (~55%
   SOTA, correctly matching slide 71's "SWE-bench Verified: real GitHub issues (~55% SOTA)"), but the
   slide's own "Frameworks & Evaluation" section names **three more benchmarks entirely absent from
   the notes**: **WebArena** (browser task completion, ~40%), **GAIA** (general AI assistant
   reasoning), and **τ-bench** (tool-use + multi-step evaluation). The same slide also has a "Where
   Agents Are Heading" block (more autonomy/less micro-prompting; agent ecosystems via MCP + interop
   protocols; specialization beats generality; always-on background agents as teammates) that's
   missing entirely. **Action:** expand §10 (or add a new §16a) with all four benchmarks and the
   "where heading" bullets — this is good, concrete, forward-looking closing material the notes
   currently cut.

3. **The production-gap "hook" statistics are missing.** Slides 46–47 open Part D with a punchy,
   quotable framing that never made it into §11: *"Great demo. Ships to production. Costs $47 per
   user per day. Crashes at 2am. **90% of agent projects never leave the prototype stage.** The gap
   isn't engineering — it's cost control, reliability, observability. The boring stuff makes it
   real."* This is exactly the kind of concrete, memorable opening statistic that motivates *why*
   §11–13 (reliability/cost/observability) matter, and it's currently absent — the notes launch
   straight into the reliability-patterns table with no framing.

4. **The "amplifying engineers 3–10x" before/after table (slides 43–45) is missing**, and would be a
   strong concrete addition to §9: a direct **before (manual) / after (agent-assisted)** comparison —
   *"write boilerplate by hand" → "add pagination to this endpoint"* (one-line prompt); *"debug at
   staging for hours" → "agent reads code, writes tests, runs tests"*; *"context-switch between 12
   files manually" → agent handles the context-switching*; closing with the concrete punchline **"10
   PRs/day instead of 2 — same quality."** This is a better, more concrete illustration of §9's
   "unfair advantage" claim than anything currently in that section.

### 🔴 Instructor name — same issue as Lectures 27/28

5. **Same "Deep Nyak (name approximate)" attribution as Lecture 27.** No name overlay was visible in
   this deck's frames either (matches Lecture 27's situation, not Lecture 28's). Given Lecture 28
   confirmed "Harsh Agarwal" for the same 3-part module, this file's attribution should very likely
   also be corrected to Harsh Agarwal — apply the same fix as recommended for Lecture 27.

### 🟡 Structural completeness — same gap as Lectures 27–28

6. Missing "Before we start" prerequisites, a dedicated Applied Amazon scenario (though interview Q10
   partially substitutes), Depth probes, and Whiteboard-ready derivations — consistent with the other
   two Agentic AI files. Lower priority than the factual/content fixes above but should be closed
   before final publish for consistency with the rest of the course.

### Verified accurate / no action needed

- Multi-agent topologies (§2), communication patterns (§3), A2A protocol (§4), all four workflow
  patterns (§5), the PR review pipeline (§6), human-in-the-loop patterns (§8), the code-agent
  landscape table (§10), reliability patterns (§11), cost management (§12), observability (§13), and
  the framework comparison (§16) all check out structurally against the corresponding slides — no
  further missing sections found beyond items 2–4 above.
- SWE-bench ~55% figure independently confirmed accurate against slide 71.

---

## Summary — priority order for the consolidated edit pass

Across all 7 files reviewed, group fixes into three passes:

**Pass 1 — 🔴 factual errors (do first, these are wrong/misleading as written):**
- L23: TarNet citation fabrication; X-learner propensity-weighting explanation backwards
- L23/L24: instructor names (Pranita Khandelwal; Sayambhu Sen)
- L27/L28/L29: instructor name — very likely "Harsh Agarwal" for all three (confirmed on L28,
  inferred for L27/29)
- L29: fabricated sandbox limit numbers "(30s, 52MB)"

**Pass 2 — 🟠 real content gaps (add missing material, all have concrete source content ready to
transcribe):**
- L23: DragonNet, the real HTE comparison table, causal-tree split worked example, Künzel citation
- L25: Thompson Sampling
- L26: PPO's own definition/derivation section; the real DQN-vs-PPO demo numbers (500.0 vs −9.2)
- L27: Plan-and-Execute (a whole missing architecture + its Blocksworld benchmark), LATS
- L27: confidence-threshold table's non-anchor rows (verify or caveat)
- L29: WebArena/GAIA/τ-bench benchmarks + "where agents are heading"; the "$47/day, 90% never ship"
  hook stats; the "3–10x, 10 PRs/day" before/after table
- L25/L26: methodology note — both were built from `slides_deduped/` + OCR/transcript rather than
  raw `output/` frames; worth a final raw-frame diff pass given the pattern of real gaps found

**Pass 3 — 🟡 polish (quick, low-risk):**
- L23: X-learner "five-stage" → "four-stage"; relabel the synthesis DAG table vs. the real slide table
- L24: duplicated Markov-property paragraph; CartPole terminal-condition sourcing caveat
- L25: enrich the grid-world sweep-by-sweep demo with real numbers
- L27/L28/L29: add the missing pipeline-required sections (Prerequisites, Applied scenario, Depth
  probes, Whiteboard-ready derivations) for structural consistency with Lectures 18–26

All findings above are written so each can be applied directly to its file without re-deriving
anything — section references, exact quotes/numbers, and the specific fix are inline with each item.
