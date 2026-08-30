# Reinforcement Learning — Amazon ML Summer School

> ✅ **Module status: quality-review complete, enhancement pass complete.** All three lecture files
> and this README have been through a full `QUALITY_REVIEW_PIPELINE.md` pass (see
> [`QUALITY_REVIEW.md`](QUALITY_REVIEW.md)) and a follow-up enhancement pass. The enhancement pass
> added **6 `interactive` spec blocks** reproducing the lecture's own live browser demos that had
> only ever been taught as static prose (Lecture 24 §7.3's discount-factor path-choice slider;
> Lecture 25 §3.1's policy-evaluation sweep, §3.2's policy-iteration crash-then-fix, §3.3's value-
> iteration spread, §6.2's TD(0) step-by-step walk — with a newly-transcribed 10-step numeric trace
> that didn't exist in prose before — and §7.3's SARSA-vs-Q-learning side-by-side comparison, also
> newly transcribed with real demo numbers); independently re-verified 12 citations across the
> module against primary sources (all confirmed accurate, zero corrections needed); added 4 more
> `| Symbol | Read it as | What it means |` tables to Lecture 26 (§3.2 baseline, §5.1 advantage,
> §5.2 actor/critic losses, §9.5 KL penalty) plus one to Lecture 25 (§4.4 UCB's previously-unbound
> $c$ and $t$); and added two light cross-references to Causal Inference Part 3 §12.3's
> off-policy-evaluation/credit-assignment framing. Lecture 26's demo (§12) was checked for a
> possible interactive block and judged not to need one — it is a static output chart, not a live
> browser tool, and is already fully described in prose. See the per-file summaries below.

Self-study notes built from the raw slide capture in [`output/`](../../output/) (not
`slides_deduped/`, which is lossy — see the warning in
[`notes/Supervised Learning/README.md`](../Supervised%20Learning/README.md)), following
[`NOTES_PIPELINE.md`](../../NOTES_PIPELINE.md).

These are **teaching documents, not summaries**. They define every term before using it, derive
every formula rather than asserting it, and work every example through to a final number. They are
written to be the only thing you need to read to master the lecture.

This module has gone through a full `QUALITY_REVIEW_PIPELINE.md` audit pass — see
[`QUALITY_REVIEW.md`](QUALITY_REVIEW.md) for the complete findings and fixes. Highlights: a swapped
transition in Lecture 24's central 2-state MDP worked example (which had inverted the
policy-iteration worked example's conclusion) was found and corrected; two numeric errors in
Lecture 26's hands-on demo that survived an earlier ad-hoc fix pass were corrected; and Lecture 25's
instructor identity was re-investigated (see below).

---

## Index

| # | Notes | Source deck | Instructor | Status | Covers |
|---|---|---|---|---|---|
| 01 | [reinforcement-learning-01.md](reinforcement-learning-01.md) | `Lecture_24 - Module 8 Reinforcement Learning Part 1` (150 raw frames) | Sayambhu Sen (Applied Scientist, Alexa, Amazon) | ✅ Complete · quality-reviewed | RL vs. supervised/unsupervised learning · Agent-environment loop · Policy π (deterministic/stochastic) · Episodic vs. continuing tasks · Real-world applications (AlphaGo, Atari, OpenAI Five, LLMs) · MDP 5-tuple (S, A, P, R, γ) · Markov property · Transition probabilities · Reward functions & discount factor γ · Effective planning horizon · State value function V^π(s) & action-value function Q^π(s,a) · Bellman expectation equations, worked on a 2-state MDP · Bellman optimality equations · Policy evaluation with a numeric convergence trace · Policy improvement & policy iteration, worked to its correct (policy-unchanged) conclusion · Value iteration with a numeric convergence trace · Contraction mapping theorem |
| 02 | [reinforcement-learning-02.md](reinforcement-learning-02.md) | `Lecture_25 - Module 8 Reinforcement Learning Part 2` (134 raw frames) | Instructor identity uncertain — see note below | ✅ Complete · quality-reviewed | Dynamic programming limitations · Policy iteration & value iteration (with real, independently re-verified sweep-by-sweep worked examples) · Exploration vs. exploitation (ε-greedy, softmax, UCB, Thompson Sampling) · Multi-armed bandits & contextual bandits · TD learning (TD(0) update rule, bootstrapping) · SARSA (on-policy TD control) · Q-learning (off-policy TD control) · SARSA vs Q-learning cliff-world comparison · On-policy vs off-policy distinction · Experience replay & the deadly triad · DQN (architecture, two tricks, Atari results) · DQN variants (Double, Dueling, Prioritized, Rainbow) |
| 03 | [reinforcement-learning-03.md](reinforcement-learning-03.md) | `Lecture_26 - Module 8 Reinforcement Learning Part 3` (83 raw frames) | Vijay Neeluru (Applied Scientist, Amazon) | ✅ Complete · quality-reviewed | Why policy gradients (discrete action limitation of DQN) · Policy gradient theorem (log-likelihood trick, full derivation, now with a bound symbol table) · REINFORCE algorithm (strengths/weaknesses) · Baselines for variance reduction, now with the unbiasedness proof derived · Actor-critic (A2C: advantage function, actor/critic losses) · A3C (asynchronous workers, decorrelated experience) · GAE (bias-variance tradeoff via λ, now with a bound symbol table) · **PPO (clipped objective, derived)** · Multi-agent RL (cooperative/competitive/self-play, CTDE) · RLHF (5-stage pipeline, Bradley-Terry model with a bound symbol table, PPO fine-tuning, KL penalty) · Beyond RLHF (DPO, RLAIF, Constitutional AI) · Nine core challenges (sample efficiency, reward shaping, sparse rewards, stability, exploration, credit assignment, sim-to-real transfer, safety, reproducibility) · Frameworks (Gymnasium, SB3, RLlib, CleanRL) · Hands-on demo (CartPole: DQN ~9.2 avg reward vs. PPO 500.0 avg reward, corrected from a prior sign/step-count error) |

⚠️ **Instructor identity for Lecture 25 is not fully resolved.** The file previously credited
"Naman Kanan," a name with no supporting nameplate anywhere in the 134-frame raw capture. This
review found the only direct evidence available: the interactive-demo slides repeatedly show a
browser address bar reading `file:///Users/nkhetan/RL_Interactive_Examples/...` — the macOS account
name **"nkhetan"** suggests a surname **Khetan** (first initial "N"), but no slide shows a full
first name, so neither "Naman Kanan" nor "N. Khetan" is stated as fully confirmed. See the file's
own header and `QUALITY_REVIEW.md` for the full account.

## Key takeaway per lecture

| # | One-sentence takeaway |
|---|---|
| 01 | Reinforcement learning is the only ML paradigm where the agent's own actions determine what data it sees next — and the Bellman equations provide the universal recursive decomposition that every RL algorithm, from tabular Q-learning to deep policy gradients, is ultimately trying to solve (approximately). |
| 02 | Every RL algorithm is a version of the same idea — update toward reward plus discounted future value — and the journey from DP (full model) through TD (model-free) to DQN (neural + replay + target net) progressively removes assumptions until one algorithm learns to play 49 Atari games from raw pixels at superhuman level. |
| 03 | Policy gradients unlock continuous actions that DQN can't handle; actor-critic is the backbone of modern RL combining policy flexibility with value stability; and RLHF is how LLMs are aligned with human preferences — the entire journey from tabular Q-learning to ChatGPT is one continuous thread of the same core idea: update toward reward plus discounted future value. |

---

## What's in Part 1

**File:** [`reinforcement-learning-01.md`](reinforcement-learning-01.md) (~7,800 words)

**Instructor:** Sayambhu Sen, Applied Scientist, Alexa team at Amazon — the same instructor as
Lecture 18 (Sequential Learning Part 1).

**This lecture establishes the foundational vocabulary and mathematical framework** for the entire
RL series: the agent-environment loop, MDPs, value functions, and the Bellman equations.

| Section | Topic | Key concept |
|---------|-------|-------------|
| §1 | What is RL? | Three paradigms, reward hypothesis |
| §2 | The RL framework | Agent-environment loop, policy π |
| §3 | Episodic vs. continuing | Finite vs. infinite horizons, γ mandatory for continuing |
| §4 | Applications | AlphaGo, Atari, OpenAI Five, LLMs, robotics |
| §5 | MDPs | 5-tuple (S, A, P, R, γ), Markov property |
| §6 | Transition probabilities | Grid world example, unknown in practice |
| §7 | Rewards & discount factor | Effective horizon 1/(1-γ) |
| §8 | Value functions | V^π(s) and Q^π(s,a), their relationship |
| §9 | Bellman expectation | V and Q equations, backup diagrams, 2-state MDP worked example |
| §10 | Bellman optimality | Max operator makes it nonlinear |
| §11 | Policy evaluation | Direct solution vs. iterative |
| §12 | Policy iteration | Evaluate → improve → evaluate → ... |
| §13 | Value iteration | Combine eval + improve, contraction mapping |
| §14 | Summary | Foundation for Q-Learning, DQN, Actor-Critic, ... |

### Capture quality

**Part 1 (Lecture 24) — ✅ excellent, quality-reviewed.** The raw capture has **150 raw frames**
(from `output/`). Built from the raw capture, following the project memory
`slides-deduped-is-lossy`. The deduped version has 73 slides. Instructor: Sayambhu Sen, confirmed
from the slide nameplate. Runs ~53:00. A `QUALITY_REVIEW_PIPELINE.md` pass found and fixed a
swapped transition in the §9.2 2-state MDP worked example that had inverted §12.2's
policy-iteration conclusion, plus several smaller citation and labeling fixes — see
`QUALITY_REVIEW.md` for the full account. No content gaps. **Interactive specs: 1** — §7.3's γ
discount-factor path-choice slider, added during the enhancement pass (the underlying browser demo,
`rl_discount_factor.html`, was previously taught only as a static table).

**Part 2 (Lecture 25) — ✅ excellent, quality-reviewed, instructor identity still uncertain.**
Originally drafted from 70 deduped slides with OCR + transcript verification, then **audited
against the raw 134-frame capture** per this project's standard methodology — one real gap
(Thompson Sampling, named in the lecture's own closing summary but never taught on a dedicated
slide) and one under-transcribed demo (the grid-world policy-evaluation sweeps) were found and
fixed. A subsequent `QUALITY_REVIEW_PIPELINE.md` pass independently re-verified every sweep number
in §3.1's and §3.3's worked examples pixel-for-pixel against the raw frames and confirmed all of
them exact, and — following up on a citation that first looked like a false alarm — found that
§3.2 was missing its own numeric worked example despite real, dramatic source material for one (a
deterministic policy that evaluates to nearly −100 before a single arrow-flip fixes it); that gap is
now filled. Instructor: see
the note above the Index table — the credited name "Naman Kanan" has no supporting nameplate
anywhere in the capture; the strongest available evidence is a screen-shared file path pointing to
a macOS account "nkhetan," suggesting a surname Khetan. Runs ~55:00. **Interactive specs: 5** —
added during the enhancement pass, one per live browser demo the deck actually runs: §3.1 policy
evaluation, §3.2 policy iteration, §3.3 value iteration, §6.2 TD(0) learning (with a newly
transcribed 10-step numeric trace that had no prose worked example before), and §7.3 SARSA vs.
Q-learning (also newly transcribed with the demo's own "one concrete update" numbers). All five
`fallback`s were verified against the actual raw frames (`slide_016`–`slide_052`, `slide_081`–
`slide_090`, `slide_096`–`slide_105`).

**Part 3 (Lecture 26) — ✅ excellent, quality-reviewed.** Originally drafted from 31 deduped
slides with OCR + transcript verification, then **audited against the raw 83-frame capture** — PPO's
own definition/derivation (used throughout the lecture but never previously explained) and the real
DQN-vs-PPO demo numbers were found missing and added. A subsequent `QUALITY_REVIEW_PIPELINE.md`
pass found that two of those demo numbers had been transcribed incorrectly (a reward sign and a
step count) and corrected them, along with three stale citation slide-numbers left over from the
deduped-to-raw renumbering and several missing symbol tables — see `QUALITY_REVIEW.md`. Instructor:
Vijay Neeluru, confirmed from the slide nameplate. Runs ~40:00. **Interactive specs: 0.** The
enhancement pass checked §12's CartPole DQN-vs-PPO demo and the rest of the 83-frame deck for a
live browser tool like Lecture 25's — none exists; §12's chart is a static matplotlib output, not
an interactive slider/toggle, and is already fully described in prose. §3.2's baseline, §5.1–5.2's
advantage/actor/critic losses, and §9.5's KL penalty — flagged by `QUALITY_REVIEW.md` as a residual
symbol-table gap — each now have a bound `| Symbol | Read it as | What it means |` table.

---

## Reading guide

Start with Part 1 (Lecture 24) for the foundational vocabulary and MDP framework, then proceed to
Parts 2–3 for the learning algorithms (Q-learning, policy gradients, deep RL). The three parts
form a progression from *what* RL is and *how* to formalize it, through *how to solve* small
MDPs, to *how to scale* to large state spaces with deep learning.

- **Part 1** (~7,800 words): what is RL, MDPs, value functions, Bellman equations, policy/value
  iteration — including numeric convergence traces for both
- **Part 2** (~7,600 words): dynamic programming, exploration-exploitation, multi-armed bandits, TD learning, SARSA, Q-learning, on/off-policy, experience replay, DQN
- **Part 3** (~7,000 words): policy gradients, actor-critic, PPO, multi-agent RL, RLHF, continuous actions

**Second pass — work these worked examples on paper before reading the solution:**
1. Part 1 §9.2/§10.2/§12.2 — the 2-state MDP solved three ways (direct solution, Bellman
   optimality, policy iteration) — all three now agree exactly ($V^*(s_1)=20$, $V^*(s_2)=21$)
2. Part 1 §11/§13 — the same MDP's convergence traces under iterative policy evaluation and value
   iteration
3. Part 2 §3.1–§3.3 — three separate grid-world demos: uniform-random policy evaluation, a
   deterministic bad-policy crash toward −100 fixed by one arrow-flip, and value iteration
4. Part 2 §7.3 — the SARSA-vs-Q-learning cliff-world comparison
5. Part 3 §2.1 — the policy gradient theorem's full 8-step derivation
6. Part 3 §7.3 — PPO's clipped objective, derived from the trust-region motivation
7. Part 3 §12 — the CartPole DQN-vs-PPO hands-on demo, with the corrected numbers

**Before an interview.** Each file's *Putting it together*, then all whiteboard derivations, then
the depth-probe questions. The three highest-value derivations across the module: **the Bellman
expectation equation from first principles** (Part 1 §9), **the policy gradient theorem's
log-likelihood trick** (Part 3 §2.1), and **PPO's clipped surrogate objective** (Part 3 §7.3).

**Callout legend**

| | Meaning |
|---|---|
| 📚 **Background** | A prerequisite the slide assumed you already knew |
| 💡 **Key insight** | The one sentence from that section worth memorising |
| ⚠️ **Careful** | A slide gap, an ambiguity, or a place a previous version of this file was wrong |
| 🧪 **Worked example** | Real numbers computed to a real answer |
| 🎯 **Interview** | Phrased the way you would actually say it out loud |
| 🔬 **Research opportunity** | Where the field is genuinely still open |

---

## What's in Part 2

**File:** [`reinforcement-learning-02.md`](reinforcement-learning-02.md) (~7,600 words)

**Instructor:** identity uncertain — see the note above the Index table.

**This lecture bridges from Part 1's definitions to actual algorithms**, starting from DP (full
model known) and progressively removing assumptions until DQN learns Atari from raw pixels.

| Section | Topic | Key concept |
|---------|-------|-------------|
| §1 | Recap | RL definition, two unique features |
| §2 | Dynamic programming | Three limitations: model needed, curse of dimensionality, discrete only |
| §3 | Policy/value iteration | Review with three separate real demos: uniform-random policy evaluation (§3.1), a deterministic bad-policy crash-then-fix (§3.2), value iteration converging in ~5 sweeps (§3.3) |
| §4 | Exploration vs. exploitation | ε-greedy, softmax, UCB strategies |
| §5 | Multi-armed bandits | Simplest RL, regret metric, contextual bandits |
| §6 | TD learning | TD(0) update, bootstrapping, best of both worlds |
| §7 | SARSA & Q-learning | On-policy vs off-policy, cliff-world comparison |
| §8 | On/off-policy | Behavior vs target policy, experience replay enables data efficiency |
| §9 | Deadly triad | Off-policy + function approx + bootstrapping = instability |
| §10 | DQN | Neural Q-function, 84×84×4 architecture, Atari results |
| §11 | DQN tricks | Experience replay (fixes correlated data) + target network (fixes moving target) |
| §12 | DQN results | 29/49 games superhuman, same hyperparameters, raw pixels |
| §13 | DQN variants | Double DQN, Dueling, Prioritized Replay, Rainbow (230% human median) |
| §14 | Summary | The complete journey: DP → TD → DQN |

---

## What's in Part 3

**File:** [`reinforcement-learning-03.md`](reinforcement-learning-03.md) (~7,000 words)

**Instructor:** Vijay Neeluru, Applied Scientist, Amazon.

**This lecture fills the gap DQN leaves** — continuous actions — via policy gradients, builds up to
actor-critic and PPO, then extends to multi-agent RL and RLHF for LLM alignment.

| Section | Topic | Key concept |
|---------|-------|-------------|
| §1 | Why policy gradients | DQN's argmax requires discrete actions |
| §2 | Policy gradient theorem | Log-likelihood trick, full derivation |
| §3 | Baselines | Reduce variance without introducing bias |
| §4 | Actor-critic | Actor (policy) + critic (value), every-step updates |
| §5 | A2C | Advantage function, actor/critic loss functions |
| §6 | A3C | Asynchronous workers decorrelate experience |
| §7 | GAE | λ dials between 1-step TD (A2C) and Monte Carlo (REINFORCE) |
| §7.3 | PPO | Clipped objective, derived — the practical default used in §12's demo and §9's RLHF |
| §8 | Multi-agent RL | Cooperative (CTDE)/competitive/self-play |
| §9 | RLHF | 5-stage pipeline, Bradley-Terry model, PPO fine-tuning, KL penalty |
| §10 | Challenges | Nine challenges: sample efficiency, reward shaping, sparse rewards, stability, exploration, credit assignment, sim-to-real, safety, reproducibility |
| §11 | Frameworks | Gymnasium, Stable-Baselines3, RLlib, CleanRL |
| §12 | Hands-on demo | CartPole: DQN (~9.2 avg reward, failed) vs. PPO (500.0 avg reward, perfect after ~9k steps) |
