---
title: "Reinforcement Learning — Part 1: Foundations, MDPs, and Bellman Equations"
topic: reinforcement-learning
lecture: 24
source: "output/Lecture_24 - Module 8 Reinforcement Learning Part 1"
slides: 73
---

# Reinforcement Learning — Part 1: Foundations, MDPs, and Bellman Equations

> Runtime ~53:00. Built from the raw capture in `output/Lecture_24 - Module 8 Reinforcement
> Learning Part 1/` (150 raw frames), not `slides_deduped/` — see project memory
> `slides-deduped-is-lossy`. Instructor: **Sayambhu Sen**, Applied Scientist, Alexa team at Amazon —
> confirmed from the slide nameplate; the same instructor as Lecture 18 (Sequential Learning Part
> 1). This is the first lecture in the reinforcement learning series, establishing the foundational
> vocabulary, the Markov Decision Process framework, and the Bellman equations that underpin all RL
> algorithms.

---

## What you'll understand after reading this

1. **State precisely how reinforcement learning differs from supervised and unsupervised learning** — no supervisor, delayed scalar rewards, sequential decisions, and the agent's actions affect future data.
2. **Define the agent-environment loop** and name its components: state, action, reward, policy π, and transition dynamics.
3. **Distinguish episodic from continuing tasks** and explain why the discount factor γ is mandatory for continuing tasks (to prevent divergent sums) but optional for episodic ones.
4. **Explain the reward hypothesis** — that all goals can be described as maximizing expected cumulative reward — and give examples of how reward functions encode goals.
5. **Define a Markov Decision Process (MDP) as a 5-tuple** (S, A, P, R, γ) and state the Markov property and its implications for modeling.
6. **Explain why transition probabilities are usually unknown** and how the agent must learn them from interaction.
7. **Compute the discounted return G_t** for a given reward sequence and explain the effective planning horizon 1/(1-γ).
8. **Define and distinguish the state value function V^π(s) and the action-value function Q^π(s,a)**, and explain their relationship V^π(s) = Σ_a π(a|s) Q^π(s,a).
9. **State the Bellman expectation equation for both V and Q**, explain what "backup" means graphically, and solve a 2-state MDP example by matrix inversion.
10. **State the Bellman optimality equation** and explain why the max operator makes it nonlinear, requiring iterative solution rather than matrix inversion.
11. **Explain policy iteration** — the alternating cycle of policy evaluation and policy improvement — and prove by example that a greedy policy improvement is guaranteed to be at least as good.
12. **Explain value iteration** and its relationship to dynamic programming, and state the contraction mapping theorem guaranteeing convergence when γ < 1.

---

## Before we start: what you need to know

### Prerequisite 1 — Expected value and summation

You need to be comfortable with $\mathbb{E}[X] = \sum_x x \cdot P(X=x)$ (discrete expectation)
and infinite geometric series: $\sum_{k=0}^{\infty} \gamma^k = \frac{1}{1-\gamma}$ when
$0 \leq \gamma < 1$. These appear in every Bellman equation.

### Prerequisite 2 — Probability distributions

A probability distribution $P(s'|s,a)$ gives the probability of transitioning to state $s'$ from
state $s$ under action $a$. The normalization property $\sum_{s'} P(s'|s,a) = 1$ for all $s,a$
is required and used throughout.

### Prerequisite 3 — Matrix inversion (conceptual)

For a small system of linear equations $Ax = b$, the solution is $x = A^{-1}b$. This is used to
solve Bellman expectation equations for small MDPs (§8). You don't need to compute inverses by
hand — the point is that the system is *linear* and therefore solvable in closed form, unlike the
Bellman *optimality* equation.

---

## The big picture

This lecture establishes the mathematical language of reinforcement learning — the vocabulary
(agents, environments, policies, rewards), the formal framework (MDPs), and the core equations
(Bellman expectation and optimality) that every subsequent RL algorithm builds on.

```
Three ML paradigms (§1)
   Supervised: input-output pairs, correct labels
   Unsupervised: no labels, discover structure
   RL: no supervisor, delayed rewards, sequential decisions
        │
        ▼
The RL framework (§2–§4)
   Agent-environment loop: observe state → take action → get reward + next state
   Policy π(a|s): rule for choosing actions (deterministic or stochastic)
   Episodic vs. continuing tasks
        │
        ▼
Markov Decision Process (§5–§7)
   5-tuple (S, A, P, R, γ)
   Markov property: future depends only on present
   Discount factor γ: controls planning horizon
   Return G_t = Σ γ^k R_{t+k+1}
        │
        ▼
Value functions (§8)
   V^π(s): how good is state s under policy π?
   Q^π(s,a): how good is taking action a in state s under π?
   Relationship: V^π(s) = Σ_a π(a|s) Q^π(s,a)
        │
        ▼
Bellman equations (§9–§12)
   Expectation: V^π(s) = Σ_a π(a|s) [R + γ Σ_{s'} P(s'|s,a) V^π(s')]
   Optimality: V*(s) = max_a [R + γ Σ_{s'} P(s'|s,a) V*(s')]
   Policy iteration: evaluate → improve → evaluate → ...
   Value iteration: combine evaluate + improve in one step
        │
        ▼
Foundation for all RL algorithms: Q-learning, DQN, actor-critic, PPO, ...
```

---

## 1. What Is Reinforcement Learning?

### 1.1 The three ML paradigms

| | **Supervised** | **Unsupervised** | **Reinforcement** |
|---|---|---|---|
| Data | Input-output pairs | No labels | No labels, no correct answers |
| Feedback | Correct label at each step | None | Scalar reward signal, often delayed |
| Learning signal | Prediction error (loss) | Structure (clusters, manifolds) | Reward (cumulative) |
| Time matters? | No (i.i.d.) | No (i.i.d.) | **Yes** — actions affect future data |
| Analogy | Student with answer key | Student exploring patterns | Trial-and-error explorer |

> *"In supervised learning the input directly gets the output. In reinforcement learning you take
> a set of actions and at the end you know whether you got the reward or not."*

**Key distinction:** RL has no supervisor providing correct actions at each step — only a scalar
reward that may come much later. The agent must discover *which* actions led to good outcomes
through trial and error.

### 1.2 The reward hypothesis

> **Reward hypothesis** (Sutton & Barto) — all goals can be described by the maximization of
> expected cumulative reward.

Every objective in RL — winning a game, balancing a pole, maximizing revenue — is encoded as a
scalar reward signal that the agent tries to maximize over time. The reward function is the
*only* communication channel between the environment and the agent about what "good" means.

---

## 2. The RL Framework

### 2.1 The agent-environment loop

```
    ┌────────────────────────────────────────────┐
    │                                            │
    │  ┌─────────┐  action a_t   ┌────────────┐  │
    │  │         │──────────────▶│            │  │
    │  │  Agent  │               │ Environment│  │
    │  │         │◀──────────────│            │  │
    │  └─────────┘  state s_{t+1}, reward r_{t+1} └─┘
    │                                            │
    └────────────────────────────────────────────┘
```

At each time step $t$:
1. Agent observes state $s_t$.
2. Agent selects action $a_t$ based on policy $\pi(a_t | s_t)$.
3. Environment transitions to $s_{t+1}$ with probability $P(s_{t+1} | s_t, a_t)$.
4. Environment emits reward $r_{t+1} = R(s_t, a_t, s_{t+1})$.
5. Repeat.

### 2.2 Policy π

> **Policy** $\pi$ — a rule for choosing actions. Maps states to actions (deterministic) or
> states to probability distributions over actions (stochastic).

- **Deterministic:** $\pi(s) = a$ — given state $s$, always take action $a$.
- **Stochastic:** $\pi(a|s)$ = probability of taking action $a$ in state $s$.
  - Example: $\pi(\text{left}|s) = 0.6$, $\pi(\text{right}|s) = 0.3$, $\pi(\text{up}|s) = 0.1$.

**Why stochastic policies exist:** exploration — trying new actions reveals new states and new
reward structures. A purely deterministic policy may never discover the best actions because it
never tries them.

### 2.3 Actions: discrete vs. continuous

- **Discrete:** finite set of actions (up, down, left, right, fire).
- **Continuous:** real-valued actions (torque applied to a motor joint, amount of investment).
- Within discrete actions: deterministic policies (one action per state) or stochastic
  (probability distribution).

---

## 3. Episodic vs. Continuing Tasks

### 3.1 Episodic tasks

The interaction breaks naturally into **episodes** — finite sequences from a start state to a
terminal state. After reaching the terminal state, the environment resets.

**Examples:** chess, maze navigation, a single game of Go, one deal of cards. In the grid world:
the episode runs from START → hallway → fork → treasure (reward +10) or trap (reward -10).

Return: $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$ where $T$ is the terminal time step.
$\gamma = 1$ is allowed because the sum is finite.

### 3.2 Continuing tasks

No natural endpoint — the agent interacts forever (or until an arbitrary cutoff).

**Examples:** stock trading, server control, robot motor control, a robot balancing a pole
indefinitely. In the grid world: the agent can roam indefinitely among non-terminal states
with no natural endpoint.

Return: $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$. **γ < 1 is mandatory** — if
$\gamma = 1$ and rewards are non-zero, the sum diverges to infinity.

> ⚠️ **The γ = 1 rule:** for continuing tasks, you can have $\gamma = 1$ or infinite horizon,
> but **not both**. At least one must be finite for the return to be well-defined.

### 3.3 The OpenAI Gym interface

```python
env = gym.make('CartPole-v1')
obs, info = env.reset()
for step in range(max_steps):
    action = agent.act(obs)              # policy π
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

- `terminated`: the episode ended naturally (pole fell, goal reached).
- `truncated`: the episode was cut short by a time limit.
- `obs`: the state (e.g., cart position, velocity, pole angle, angular velocity).

---

## 4. Real-World Applications

### 4.1 Gaming — where RL first showed its power

- **TD-Gammon (1992):** the earliest major RL milestone — learned to play backgammon at
  world-class level via self-play.
- **Atari games (2013–2015):** DeepMind's DQN learned to play 49 Atari games from raw pixels
  alone, beating human experts on many. No feature engineering — just pixels in, game actions out.
- **AlphaGo (2016):** DeepMind defeated world champion Lee Sedol at Go, first using human data +
  RL.
- **AlphaGo Zero (2017):** beat AlphaGo 100-0 after 3 days of self-play with **no human data**.
- **OpenAI Five (2018–2019):** defeated professional Dota 2 players using self-play, no human
  demonstrations.

**The common thread:** all three used a **simulation environment** where the model could play
millions of games against itself, learning from failures and rare successes.

### 4.2 RL milestones timeline

| Year | Milestone |
|------|-----------|
| 1992 | TD-Gammon (backgammon via self-play) |
| 2013 | DQN (Atari from raw pixels) |
| 2016 | AlphaGo defeats Lee Sedol |
| 2017 | AlphaGo Zero (no human data) |
| 2018 | OpenAI Five (Dota 2) |
| 2022 | ChatGPT (RLHF for language models) |
| 2024 | Large-scale reasoning models (DeepSeek et al.) |

### 4.3 Beyond gaming

- **LLMs:** ChatGPT, Claude, and DeepSeek use RL extensively (RLHF — reinforcement learning from
  human feedback) to align model outputs with human preferences beyond what supervised fine-tuning
  alone can achieve.
- **Robotics:** sim-to-real transfer — train in simulation, deploy on physical robots.
- **Self-driving cars:** lane keeping, navigation.
- **Finance:** portfolio optimization, trading strategies.
- **Recommendations:** YouTube, Netflix.
- **Data centers:** 40% cooling savings (DeepMind).

---

## 5. Markov Decision Processes (MDPs)

### 5.1 The 5-tuple

> **MDP (Markov Decision Process)** — defined by a 5-tuple $(\mathcal{S}, \mathcal{A}, P, R,
> \gamma)$:

| Symbol | Meaning | Example |
|--------|---------|---------|
| $\mathcal{S}$ | Set of all possible states | Grid coordinates {(1,1), (1,2), ...} |
| $\mathcal{A}$ | Set of all possible actions | {up, down, left, right} |
| $P(s'|s,a)$ | Transition probability | P(right \| (1,1), right) = 0.8 |
| $R(s,a,s')$ | Reward function | R((1,1), right, (2,1)) = 0 |
| $\gamma$ | Discount factor ∈ [0,1] | 0.9 |

**The power of the MDP framework:** once you define these five elements, *any* RL problem can be
reduced to the same equations — the math is universal regardless of whether the domain is chess,
robotics, or stock trading.

### 5.2 The Markov property

> **Markov property** — the future is independent of the past given the present. The current
> state $s_t$ contains all the information needed to predict the future.

$$P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_1, a_1, s_2, a_2, \ldots, s_t, a_t)$$

**What this buys you:** you don't need to remember history. As long as the state is
*sufficiently informative* (encodes everything relevant about the past), you can make optimal
decisions based only on the current state.

**Examples:**
- ✅ **Chess board position:** the current position fully determines what moves are legal and
  what outcomes are possible — you don't need to know how the game got there.
- ❌ **Driving duration:** "how long have I been driving" requires remembering the start time —
  the current state alone isn't enough.

> ⚠️ **The Markov property is an assumption, not always true.** In practice, many problems are
  "approximately Markov" — the current state captures *most* of the relevant information. The
  trade-off: assuming Markov makes modeling tractable; violating it means the agent's decisions
  may be suboptimal.

---

## 6. Transition Probabilities

$$P(s'|s,a) \geq 0 \quad \text{for all } s, a, s'$$

$$\sum_{s'} P(s'|s,a) = 1 \quad \text{for all } s, a$$

**Grid world example:** from state (1,1) with action "right" (slippery stochastic transitions):
- P(right | (1,1), right) = 0.8 (moves right as intended)
- P(up | (1,1), right) = 0.1 (slips up)
- P(down | (1,1), right) = 0.1 (slips down)
- P(left | (1,1), right) = 0 (cannot move left when trying to move right)
- Sum = 0.8 + 0.1 + 0.1 = 1.0 ✓
- **Intended travel hits a wall → agent stays in current cell** (not listed as a separate
  transition; the 0.8 probability of "right" only succeeds if there's no wall).

**Key point:** in most real problems, transition probabilities are **unknown** — the agent must
learn the dynamics of the environment through interaction, not from a known model.

---

## 7. Reward Functions and the Discount Factor

### 7.1 Reward function

> **Reward** $R(s,a,s')$ — a scalar signal emitted at each transition, defining the goal.

- Usually **zero** until a terminal or desired state is reached.
- In CartPole: +1 per step the pole stays upright, terminal when the pole tips too far or the cart
  drifts too far (exact thresholds given just below — see the sourcing caveat there).
- The reward function is the environment designer's way of telling the agent what to optimize.

**CartPole state vector (4 continuous variables):**
- Cart position $x$ (continuous)
- Cart velocity $\dot{x}$ (continuous)
- Pole angle $\theta$ (continuous, radians from vertical)
- Pole angular velocity $\dot{\theta}$ (continuous)
- Actions: {push_left, push_right} (discrete)
- Terminal conditions: $|x| > 2.4$, $|\theta| > 12°$, or episode length > 500 steps — these are the
  standard Gymnasium `CartPole-v1` defaults; the slide itself states states/actions/reward but not
  these exact thresholds, so treat them as the well-known environment spec rather than a slide quote.
- Physics: deterministic transitions (not stochastic like the grid world)

### 7.2 Discount factor γ

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

| $\gamma$ | Interpretation | Effective horizon |
|-----------|---------------|-------------------|
| 0 | Only cares about immediate reward | 1 step |
| 0.9 | Far-sighted but still convergent | $\frac{1}{1-0.9} = 10$ steps |
| 1 | Values all future rewards equally | $\infty$ (diverges for continuing tasks) |

**Why discounting matters — three reasons:**
1. **Convergence:** ensures infinite sums converge (required for continuing tasks).
2. **Uncertainty:** far-future rewards are less certain — discounting reflects this.
3. **Myopia:** "cash now is better than cash later" — immediate rewards are naturally more
   valuable than delayed ones of the same magnitude.

### 7.3 Interactive demo — which path does the agent choose?

The lecture's interactive demo presents multiple scenarios with different γ values:

| γ | Effective horizon | Path A (immediate +5, then +1/step) | Path B (delayed +10 after 10 steps) | Agent's choice |
|---|-------------------|-------------------------------------|-------------------------------------|----------------|
| 0.90 | 10 steps | 8.69 | 15.90 | **Path B** (delayed reward still visible) |
| 0.99 | 100 steps | 9.85 | 23.92 | **Path B** (far-sighted, values delayed reward highly) |
| 0.68 | ~3 steps | 6.82 | 5.58 | **Path A** (myopic, immediate reward dominates) |
| 0.14 | ~1.2 steps | 5.16 | 1.16 | **Path A** (very myopic) |
| 0.00 | 1 step | 5.00 | 1.00 | **Path A** (only sees immediate reward) |

> ⚠️ **A previous version of this table labeled the 5.16/1.16 row "γ = 0.10."** The lecture's own
> demo screen [slide 72] shows these exact values (5.16, 1.16) computed at **γ = 0.14** (its own
> annotation reads "Eff. horizon: 1.2 steps"), not γ = 0.10 — corrected above.

**Key insight from the demo:** as γ decreases, the agent becomes progressively more myopic —
it can no longer "see" the delayed +10 reward because it's discounted to near-zero. At γ = 0.9,
the +10 after 10 steps is still worth 10 × 0.9¹⁰ ≈ 3.5, which combined with the intermediate
+1/step rewards totals 15.90, beating Path A's 8.69. But at γ = 0.1, the same +10 is discounted
to 10 × 0.1¹⁰ ≈ 0 — invisible.

**Effective planning horizon:** $\frac{1}{1-\gamma}$. At $\gamma = 0.9$, the agent effectively
plans 10 steps ahead. At $\gamma = 0.99$, it plans 100 steps ahead. At $\gamma = 0.1$, only
~1 step ahead.

> 💡 **Key insight — γ controls the agent's time preference.** A low $\gamma$ makes the agent
> myopic (greedy for immediate reward); a high $\gamma$ makes it farsighted (willing to endure
> short-term pain for long-term gain). The choice of $\gamma$ is a design decision that encodes
> how much the problem designer values future versus immediate rewards.

```interactive
type: slider
title: γ and the Agent's Time Horizon
concept: The discount factor γ sets the agent's effective planning horizon and decides which of two reward paths looks better
control: Drag γ continuously from 0.00 to 0.99
observe: The per-step decay bars (γ⁰, γ¹, γ²...) reshape, Path A's and Path B's G₀ recompute live, and the "Agent chooses" banner flips between Path A and Path B as γ crosses the point where they're equal
insight: Path B's delayed +10 only "counts" once γ is high enough that γ¹⁰ isn't already ≈0 — below that crossover the same +10 reward is mathematically invisible to the agent, no matter how large it is
fallback: The table above gives five sampled γ values (0.00, 0.14, 0.68, 0.90, 0.99) with both paths' computed G₀ and which path wins at each — γ=0.90 and 0.99 favor Path B (15.90, 23.92 vs. 8.69, 9.85); γ=0.68, 0.14, and 0.00 favor Path A (6.82, 5.16, 5.00 vs. 5.58, 1.16, 1.00), with the crossover between γ=0.68 and γ=0.90.
```

---

## 8. Value Functions

### 8.1 State value function V^π(s)

> **State value function** — the expected cumulative discounted reward starting from state $s$
> and following policy $\pi$ forever.

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\Big|\; S_t = s\right]$$

**Interpretation:** "how good is it to be in state $s$ under policy $\pi$?" — a single number
summarizing the long-term value of that state.

### 8.2 Action-value function Q^π(s,a)

> **Action-value function (Q-function)** — the expected cumulative discounted reward starting from
> state $s$, taking action $a$, and then following policy $\pi$.

$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

**Interpretation:** "how good is it to take action $a$ in state $s$ under policy $\pi$?"

### 8.3 The relationship between V and Q

$$V^\pi(s) = \sum_a \pi(a|s) \cdot Q^\pi(s,a)$$

**Words before symbols:** the value of being in state $s$ is the weighted average of the values
of all possible actions from that state, weighted by the policy's probability of choosing each
action. If the policy is deterministic ($\pi(a^*|s) = 1$ for one action $a^*$), then
$V^\pi(s) = Q^\pi(s, a^*)$.

---

## 9. Bellman Expectation Equations

### 9.1 The Bellman equation for V^π

$$V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) \cdot V^\pi(s') \right]$$

**Words before symbols:** the value of state $s$ under policy $\pi$ equals:
1. **Average over actions** (weighted by $\pi(a|s)$): for each action $a$ that $\pi$ might take...
2. **Immediate reward** $R(s,a)$ plus...
3. **Discounted expected value of the next state**: averaged over all possible next states $s'$
   weighted by transition probabilities $P(s'|s,a)$, each valued at $V^\pi(s')$.

**The "backup" diagram:**
```
          s
         /|\
        / | \
      a₁  a₂  a₃    ← weighted by π(a|s)
      |   |   |
      s'₁ s'₂ s'₃  ← weighted by P(s'|s,a)
      |   |   |
     V(s'₁) V(s'₂) V(s'₃)
```

The value at $s$ "backs up" from the values of the states it can reach.

**Key property:** this is a **system of linear equations** — if you know $P$, $R$, and $\pi$,
you can solve for $V^\pi$ by matrix inversion.

### 9.2 Worked example: 2-state MDP

Two states $s_1, s_2$. Actions: stay or go. $\gamma = 0.9$. Policy: $\pi(\text{stay}|s_1) = 1$,
$\pi(\text{go}|s_2) = 1$ (deterministic).

| State | Action | Reward | Next state |
|-------|--------|--------|------------|
| $s_1$ | stay | 2 | $s_1$ |
| $s_1$ | go | 0 | $s_2$ |
| $s_2$ | stay | 1 | $s_2$ |
| $s_2$ | go | 3 | $s_1$ |

Under the policy $\pi$, $s_1$ always "stays" (self-loop) and $s_2$ always "goes" (transitions to
$s_1$) — this is the row of the table each state actually follows, shown in **bold** logic below.
Bellman equations under this policy [slide 137]:

$$V^\pi(s_1) = R(s_1,\text{stay}) + \gamma \cdot V^\pi(s_1) = 2 + 0.9 \cdot V^\pi(s_1)$$
$$V^\pi(s_2) = R(s_2,\text{go}) + \gamma \cdot V^\pi(s_1) = 3 + 0.9 \cdot V^\pi(s_1)$$

Simplifying:
$$V^\pi(s_1) = 2 + 0.9 \cdot V^\pi(s_1) \implies 0.1 \cdot V^\pi(s_1) = 2 \implies V^\pi(s_1) = 20$$
$$V^\pi(s_2) = 3 + 0.9 \cdot V^\pi(s_1) = 3 + 0.9 \times 20 = 21$$

**Matrix form** [slide 137]: $V = b + \gamma P V \implies (I - \gamma P) V = b \implies V = (I -
\gamma P)^{-1} b$. In this MDP, $\begin{bmatrix} 0.1 & 0 \\ -0.9 & 1\end{bmatrix}
\begin{bmatrix}V(s_1)\\V(s_2)\end{bmatrix} = \begin{bmatrix}2\\3\end{bmatrix}$, giving the same
result: $V^\pi(s_1)=20$, $V^\pi(s_2)=21$.

This works because the system is linear — a direct consequence of the expectation (weighted
average) in the Bellman equation.

> ⚠️ **A previous version of this file had the $s_2$ row of the transition table backwards**
> (claiming "$s_2$, stay → $s_1$" and "$s_2$, go → $s_2$"), which produced the wrong value
> $V^\pi(s_2)=30$. The lecture's own slides [137, 139, 140] agree unambiguously: under this policy,
> "stay" from $s_2$ is a **self-loop** back into $s_2$ (reward 1), and "go" from $s_2$ transitions
> **to $s_1$** (reward 3) — so $V^\pi(s_2)=21$, not 30. This correction propagates through §10.2 and
> §12.2 below.

### 9.3 Bellman equation for Q^π

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') \cdot Q^\pi(s',a')$$

**Key difference from V:** the Q-backup averages over the **next action** $a'$ (weighted by
$\pi(a'|s')$), not the current action (which is given as the argument to Q).

```
V-backup:  s → {a} → s'         (average over next s')
Q-backup:  s, a → s' → {a'}     (average over next s' AND next a')
```

> 💡 **Key insight — V vs. Q reflects two different questions.** $V^\pi(s)$ asks *"how good is
> this state?"* — useful for evaluating a state. $Q^\pi(s,a)$ asks *"how good is this action in
> this state?"* — useful for choosing actions. Policy improvement (§12) requires Q-values because
> you need to compare actions, not just states.

---

## 10. Bellman Optimality Equations

### 10.1 From expectation to optimality

The Bellman *expectation* equation averages over actions weighted by $\pi$. The Bellman
*optimality* equation replaces the average with **max**:

$$V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) \cdot V^*(s') \right]$$

$$Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \cdot \max_{a'} Q^*(s',a')$$

**The critical difference:** the max operator makes the equation **nonlinear** — you cannot solve
it by matrix inversion. Iterative methods (value iteration, policy iteration) are required.

### 10.2 Worked example: the same 2-state MDP

$$V^*(s_1) = \max\begin{cases} 2 + 0.9 \cdot V^*(s_1) & \text{(stay)} \\ 0 + 0.9 \cdot V^*(s_2) & \text{(go)} \end{cases}$$

$$V^*(s_2) = \max\begin{cases} 1 + 0.9 \cdot V^*(s_2) & \text{(stay)} \\ 3 + 0.9 \cdot V^*(s_1) & \text{(go)} \end{cases}$$

**Interactive exploration:**
- With default rewards (stay at $s_1$: reward 2, go from $s_2$: reward 3): optimal policy is
  "stay at $s_1$, go from $s_2$" — confirmed below in §12.2 by computing every Q-value directly
  [slides 139–140]: $V^*(s_1)=20$, $V^*(s_2)=21$.
- Increase the "go" reward at $s_1$: at some point, the optimal policy switches to "go from $s_1$"
  because the immediate reward outweighs staying.
- Decrease the "stay" reward at $s_2$: eventually "go from $s_2$" becomes better because $s_2$'s
  stay value drops below the transition to $s_1$.

> ⚠️ **The optimal value always dominates:** $V^*(s) \geq V^\pi(s)$ for any policy $\pi$ and
> state $s$. No policy can exceed the optimal value — you can only match it.

---

## 11. Policy Evaluation

Given a fixed policy $\pi$, compute $V^\pi(s)$ for all $s$.

**Method 1 — Direct solution (matrix inversion):** when $P$ and $\pi$ are known and the state
space is small, solve the linear system $(I - \gamma P^\pi) V = R^\pi$ directly.

**Method 2 — Iterative evaluation:** start with $V_0(s) = 0$ for all $s$, then repeatedly apply
the Bellman expectation update:

$$V_{k+1}(s) \leftarrow \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) \cdot V_k(s') \right]$$

Converges to $V^\pi$ as $k \to \infty$ (guaranteed by the contraction mapping theorem when
$\gamma < 1$).

🧪 **Worked example — iterating toward §9.2's known answer.** Applying this update to §9.2's
2-state MDP under $\pi_0$ (equations $V(s_1) \leftarrow 2 + 0.9\,V(s_1)$,
$V(s_2) \leftarrow 3 + 0.9\,V(s_1)$), starting from $V_0(s_1)=V_0(s_2)=0$:

| $k$ | $V_k(s_1)$ | $V_k(s_2)$ |
|---|---|---|
| 0 | 0.000 | 0.000 |
| 1 | 2.000 | 3.000 |
| 2 | 3.800 | 4.800 |
| 3 | 5.420 | 6.420 |
| 4 | 6.878 | 7.878 |
| 5 | 8.190 | 9.190 |
| $\to\infty$ | **20.000** | **21.000** |

Each step closes the gap to the true fixed point $(20, 21)$ by a factor of exactly $\gamma=0.9$
(e.g. the gap at $s_1$ shrinks $18 \to 16.2 \to 14.58 \to \ldots$, each term $0.9\times$ the last)
— this *is* the contraction mapping theorem made concrete: convergence is geometric with rate
$\gamma$, so full precision takes many iterations, but the direction and rate are exact from
iteration 1.

---

## 12. Policy Improvement and Policy Iteration

### 12.1 Policy improvement

Given $V^\pi$, construct a new policy $\pi'$ that is **greedy** with respect to $Q^\pi$:

$$\pi'(s) = \arg\max_a Q^\pi(s,a)$$

**The policy improvement theorem:** $V^{\pi'}(s) \geq V^{\pi}(s)$ for all $s$ — the greedy
policy is always at least as good as the original.

### 12.2 Policy iteration

Alternate between evaluation and improvement until convergence:

```
π₀ → evaluate → V^π₀ → improve → π₁ → evaluate → V^π₁ → improve → π₂ → ...
```

**Each improvement step is guaranteed to be at least as good.** Since the number of deterministic
policies is finite (for finite MDPs), policy iteration converges in a finite number of steps.

**Worked example** [slides 137, 139, 140] — starting with $\pi_0$ = "stay at $s_1$, go from
$s_2$":
1. **Evaluate:** $V^{\pi_0}(s_1) = 20$, $V^{\pi_0}(s_2) = 21$ (from §9.2).
2. **Compute Q-values for ALL actions** (not just the ones $\pi_0$ takes):
   - $Q^{\pi_0}(s_1, \text{stay}) = 2 + 0.9 \times V^{\pi_0}(s_1) = 2 + 0.9 \times 20 = 20.0$
   - $Q^{\pi_0}(s_1, \text{go}) = 0 + 0.9 \times V^{\pi_0}(s_2) = 0 + 0.9 \times 21 = 18.9$
   - $Q^{\pi_0}(s_2, \text{stay}) = 1 + 0.9 \times V^{\pi_0}(s_2) = 1 + 0.9 \times 21 = 19.9$
   - $Q^{\pi_0}(s_2, \text{go}) = 3 + 0.9 \times V^{\pi_0}(s_1) = 3 + 0.9 \times 20 = 21.0$
3. **Improve** (greedy w.r.t. these Q-values):
   $\pi_1(s_1) = \arg\max\{Q(s_1,\text{stay})=20.0,\ Q(s_1,\text{go})=18.9\} = \text{stay}$;
   $\pi_1(s_2) = \arg\max\{Q(s_2,\text{stay})=19.9,\ Q(s_2,\text{go})=21.0\} = \text{go}$.
4. **Conclusion:** $\pi_1 = \{s_1:\text{stay},\ s_2:\text{go}\} = \pi_0$ — **the policy is
   unchanged.** By the policy improvement theorem, an unchanged policy means $\pi_0$ was *already*
   optimal: $\pi_0 = \pi^*$, with $V^*(s_1)=20$, $V^*(s_2)=21$.

> 💡 **This is the honest, common outcome of policy iteration, not a special case.** The initial
> policy in this example happens to already be greedy with respect to its own value function, so
> the loop converges in **zero** improvement steps — you evaluate once, check that no action beats
> the current one anywhere, and stop. If $\pi_1 \ne \pi_0$ had come out of step 3, you would repeat:
> evaluate $\pi_1$ → improve → evaluate → ... until the policy stops changing, exactly as §12.2's
> diagram above shows.
>
> ⚠️ **A previous version of this file's step 1 used the wrong value $V^{\pi_0}(s_2)=30$** (see the
> §9.2 correction above) and, compounding the error, mis-transcribed the Q-values so that the
> "improved" policy appeared to switch both actions to "go" — the opposite of the correct,
> slide-verified result that the original policy was already optimal.

---

## 13. Value Iteration

Combine evaluation and improvement in one step — update V directly using the Bellman optimality
equation:

$$V_{k+1}(s) \leftarrow \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) \cdot V_k(s') \right]$$

After convergence, recover the optimal policy:

$$\pi^*(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) \cdot V^*(s') \right]$$

**Relationship to dynamic programming:** value iteration is the RL analogue of dynamic programming's
"solve subproblems bottom-up." Each iteration solves a one-step-ahead subproblem and uses the
result to improve the overall solution.

**Convergence guarantee:** the contraction mapping theorem guarantees that value iteration
converges to $V^*$ as long as $\gamma < 1$. The contraction factor is $\gamma$, so the error
shrinks by a factor of $\gamma$ each iteration.

🧪 **Worked example — value iteration on §9.2's 2-state MDP.** Using
$V(s_1)\leftarrow\max\{2+0.9V(s_1),\ 0.9V(s_2)\}$ and
$V(s_2)\leftarrow\max\{1+0.9V(s_2),\ 3+0.9V(s_1)\}$, starting from $V_0=(0,0)$:

| $k$ | $V_k(s_1)$ | argmax at $s_1$ | $V_k(s_2)$ | argmax at $s_2$ |
|---|---|---|---|---|
| 1 | 2.000 | stay | 3.000 | go |
| 2 | 3.800 | stay | 4.800 | go |
| 3 | 5.420 | stay | 6.420 | go |
| 4 | 6.878 | stay | 7.878 | go |
| 5 | 8.190 | stay | 9.190 | go |
| $\to\infty$ | **20.000** | stay | **21.000** | go |

The argmax at each state stabilizes on "stay"/"go" from the very first sweep and never flips —
value iteration here rediscovers $\pi^*=\{s_1:\text{stay}, s_2:\text{go}\}$ well before the value
estimates themselves finish converging, which is typical: the *policy* often stabilizes long
before the *values* do.

---

## 14. Putting it together

```
THE RL FOUNDATION STACK
════════════════════════

   ┌─────────────────────────────────────────────┐
   │  Applications: games, LLMs, robotics, finance │
   └──────────────────┬──────────────────────────┘
                      │ all rest on
   ┌──────────────────▼──────────────────────────┐
   │  RL Algorithms: Q-Learning, DQN, Actor-Critic │
   │  (covered in Parts 2–3 of this module)        │
   └──────────────────┬──────────────────────────┘
                      │ all solve
   ┌──────────────────▼──────────────────────────┐
   │  Bellman Equations                            │
   │  Expectation: V^π(s) = Σ π(a|s)[R + γV(s')] │
   │  Optimality:  V*(s) = max_a [R + γV*(s')]    │
   └──────────────────┬──────────────────────────┘
                      │ which define
   ┌──────────────────▼──────────────────────────┐
   │  MDP = (S, A, P, R, γ)                       │
   │  Markov property: future ⊥ past | present    │
   └──────────────────┬──────────────────────────┘
                      │ which captures
   ┌──────────────────▼──────────────────────────┐
   │  Agent-Environment Loop                       │
   │  observe s → take a → get r, s' → repeat     │
   └─────────────────────────────────────────────┘
```

Three threads run through this lecture:

1. **RL is fundamentally different from supervised learning because the agent's actions change
   the data distribution.** In supervised learning, the i.i.d. assumption holds — each training
   example is independent. In RL, choosing action $a$ in state $s$ transitions to a *different*
   state $s'$, which determines what data the agent sees next. This sequential, non-i.i.d.
   structure is what makes RL both harder and more powerful than supervised learning.

2. **The Bellman equations are the foundational recursive decomposition of RL.** Every RL
   algorithm — from tabular Q-learning to deep policy gradients — is ultimately a method for
   solving (approximately) some Bellman equation. Understanding the exact form (expectation vs.
   optimality, V vs. Q) is prerequisite to understanding what each algorithm is actually computing.

3. **The discount factor γ is not just a mathematical convenience — it encodes a design decision
   about time preference.** The choice of γ determines how far ahead the agent plans, how it
   trades off immediate versus delayed rewards, and whether the return is even well-defined (for
   continuing tasks). Changing γ changes the *problem*, not just the solution.

---

## Interview prep — Amazon Applied Scientist

### Core questions

<details><summary><b>1. How does RL differ from supervised learning in three specific ways?</b></summary>

(1) **No supervisor:** RL has no correct-label teacher — only a scalar reward signal that may be
delayed. (2) **Delayed feedback:** the agent takes a sequence of actions before knowing whether
the outcome was good; in supervised learning, the loss is computed immediately after each
prediction. (3) **Non-i.i.d. data:** the agent's actions change the state, which determines the
next observation — the data distribution depends on the agent's own policy, violating the i.i.d.
assumption that supervised learning relies on.
</details>

<details><summary><b>2. Define an MDP and explain why the Markov property matters for computational tractability.</b></summary>

An MDP is a 5-tuple (S, A, P, R, γ). The Markov property states that P(s_{t+1} | s_t, a_t)
doesn't depend on the history s_1, a_1, ..., s_{t-1}, a_{t-1} — the current state is sufficient.
This matters because without it, the agent would need to track arbitrarily long histories, making
the state space (and therefore the value function) grow exponentially with time. With the Markov
property, the value function is just V(s): a function of the current state alone, which is
tractable to learn and store.
</details>

<details><summary><b>3. Why is γ < 1 mandatory for continuing tasks but optional for episodic tasks?</b></summary>

For continuing tasks (infinite horizon), the return G_t = Σ_{k=0}^∞ γ^k R_{t+k+1} is an
infinite sum. If γ = 1 and rewards are nonzero, this sum diverges — the return is infinite and
the value function is undefined. With γ < 1, the geometric series converges to a finite value
(1/(1-γ) times the average reward). For episodic tasks, the sum is finite (T - t terms), so it
converges regardless of γ.
</details>

<details><summary><b>4. State the Bellman expectation equation for V^π and explain what "backup" means graphically.</b></summary>

V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]. Graphically, the value at state s
"backs up" to the values of the next states s' it can reach: each next state's value is weighted
by the transition probability, each action is weighted by the policy, and the immediate reward is
added. The value function is recursively defined in terms of the values of successor states.
</details>

<details><summary><b>5. Explain why the Bellman optimality equation is nonlinear and requires iterative solution.</b></summary>

The optimality equation replaces the policy-weighted average (Σ_a π(a|s) · ...) with a max
operator (max_a [...]). The max makes the equation nonlinear — you cannot write it as a linear
system Ax = b and solve by matrix inversion. Instead, iterative methods like value iteration
(repeatedly apply the max update until convergence) or policy iteration (alternate evaluation and
improvement) are needed.
</details>

<details><summary><b>6. In the worked 2-state MDP, what happens to the optimal policy when you increase the "go" reward at s₁?</b></summary>

Initially, staying at s₁ (reward 2) dominates going (reward 0). As the "go" reward increases,
there's a crossover point where 0 + 0.9·V*(s₂) > 2 + 0.9·V*(s₁) — the discounted future value
of being in s₂ outweighs the immediate staying reward. At that point, the optimal policy at s₁
switches from "stay" to "go." The policy change is discontinuous (a threshold effect) while the
value function changes smoothly.
</details>

<details><summary><b>7. What is the effective planning horizon and how does it relate to γ?</b></summary>

The effective planning horizon is 1/(1-γ). At γ = 0.9, the agent plans ~10 steps ahead (rewards
beyond 10 steps contribute less than ~35% of their undiscounted value). At γ = 0.99, it plans
~100 steps ahead. This directly controls how "farsighted" the agent is — a low γ makes it myopic
(greedy for immediate reward), a high γ makes it willing to endure short-term pain for long-term
gain.
</details>

<details><summary><b>8. Explain the difference between policy evaluation, policy improvement, and policy iteration.</b></summary>

Policy evaluation: given a fixed policy π, compute V^π(s) for all states (by solving the linear
system or iterating the Bellman expectation equation). Policy improvement: given V^π, construct a
new greedy policy π'(s) = argmax_a Q^π(s,a) that is guaranteed to be at least as good. Policy
iteration: alternate evaluation → improvement → evaluation → ... until the policy stops changing,
at which point you've found π*.
</details>

<details><summary><b>9. Why is γ = 0.9 the "typical choice"? What goes wrong with γ too low or too high?</b></summary>

γ = 0.9 gives an effective horizon of 10 steps — long enough to plan meaningfully, short enough
for convergence. γ too low (e.g., 0.1, horizon ~1 step): the agent is too myopic to discover
delayed rewards; it may never find the treasure at the end of a maze because it only sees the
immediate +1 steps. γ too high (e.g., 0.999, horizon ~1000 steps): convergence is very slow (the
contraction factor is γ, so error shrinks slowly), and the agent may overvalue distant, uncertain
outcomes.
</details>

<details><summary><b>10. [Combines concepts] A grid-world agent with γ = 0.9 consistently chooses the action leading to a +5 immediate reward over a path that yields +10 after 3 steps. Is this a bug or a feature? How would you change the behavior?</b></summary>

With γ = 0.9, the +10 after 3 steps is discounted to 10 × 0.9³ ≈ 7.3, which exceeds the +5
immediate reward — so the agent *should* prefer the delayed path if the value function is correct.
If it doesn't, either (1) the value function hasn't converged yet (run more iterations), or (2)
the environment dynamics are being learned incorrectly (the agent doesn't yet know that the
+10 path is reachable). Increasing γ (e.g., to 0.95, horizon 20) would make the delayed reward
even more attractive, but the root cause is likely insufficient training, not γ being wrong.
</details>

### Depth probes

- *"The Markov property says the current state suffices. But what if it doesn't — for example,
  in a POMDP where the state is partially observable? How would the framework need to change?"* —
  in a POMDP (Partially Observable MDP), the agent doesn't observe the true state s_t directly
  but instead receives an observation o_t drawn from P(o_t | s_t). The agent must maintain a
  *belief state* — a probability distribution over possible states — which is itself Markov. The
  framework extends from MDPs to POMDPs by replacing the state value function V(s) with a
  value function over belief distributions V(b), dramatically increasing computational complexity.

- *"Why does the contraction mapping theorem require γ < 1 specifically? What mathematical
  property does γ < 1 provide?"* — the Bellman operator T defined by the expectation equation is
  a contraction in the sup-norm with factor γ: ||TV₁ - TV₂||_∞ ≤ γ ||V₁ - V₂||_∞. The Banach
  fixed-point theorem then guarantees a unique fixed point (V^π) and convergence from any
  starting point. When γ = 1, the operator is no longer a contraction (it's a non-expansive map),
  and the fixed point may not exist or may not be unique (e.g., in continuing tasks with
  non-zero rewards, every state has infinite value).

- *"The lecture says policy iteration converges in a finite number of steps for finite MDPs.
  Why finite rather than just 'converges'?"* — because there are only |A|^|S| possible
  deterministic policies for a finite MDP. Each improvement step either changes the policy (to a
  strictly better one, by the policy improvement theorem) or leaves it unchanged (meaning it's
  optimal). Since the number of policies is finite and each step is strictly improving, the
  process must terminate in at most |A|^|S| steps — not just converging asymptotically, but
  reaching the optimum in finite time.

### Whiteboard-ready derivations

1. **The Bellman expectation equation for V^π** — §9.1: derive from the definition V^π(s) =
   E_π[G_t | S_t = s] by splitting G_t into R_{t+1} + γG_{t+1}, conditioning on the first
   action and first transition, and recognizing that E_π[G_{t+1} | S_{t+1} = s'] = V^π(s').

2. **The relationship V^π(s) = Σ_a π(a|s) Q^π(s,a)** — §8.3: derive by definition — V^π(s) is
   the expectation of G_t given S_t = s, which decomposes over actions via the law of total
   expectation: E[G_t | S_t = s] = Σ_a P(A_t = a | S_t = s) E[G_t | S_t = s, A_t = a] = Σ_a
   π(a|s) Q^π(s,a).

3. **Policy improvement theorem (sketch)** — §12.1: starting from V^π, define π' as greedy w.r.t.
   Q^π. Show V^π'(s) ≥ V^π(s) by expanding V^π'(s) = Q^π(s, π'(s)) ≥ Q^π(s, π(s)) = V^π(s),
   where the inequality holds because π' chooses the argmax action.

### Applied scenario — Amazon warehouse robot navigation

**Framing:** An Amazon warehouse robot must navigate from a pickup location to a shelf, grab a
package, and deliver it to a packing station. The robot receives +1 for each step toward the goal,
+10 for successful delivery, -5 for collisions, and 0 otherwise.

**MDP formulation:**
- **States:** robot position (x, y), orientation, package held (yes/no), target shelf ID.
- **Actions:** move forward, turn left, turn right, grab, release.
- **Transitions:** stochastic — the robot may slip on wet floors, encounter other robots, or
  have grasp failures (P(grab succeeds) = 0.9).
- **Discount factor:** γ = 0.95 (planning ~20 steps ahead — appropriate for a warehouse aisle
  that's ~15-20 steps long).
- **Reward function:** encodes the business goal — fast, collision-free delivery.

**Policy learning:** start with a random policy, use policy iteration to find the optimal
navigation policy. The Bellman optimality equation's max operator naturally selects the fastest
collision-free path.

**Failure modes:**
- γ too low: the robot becomes myopic, preferring short-term collision avoidance over reaching
  the delivery goal.
- Reward misspecification: if +1 per step is too high relative to +10 for delivery, the robot
  may wander indefinitely to collect step rewards.
- Partial observability: if the robot can't see around corners, the Markov property is violated —
  the robot needs a belief state or memory.

**What you'd ship:** a value-iteration-trained policy with γ = 0.95, combined with a safety layer
that overrides the learned policy when a collision is imminent (constraining the max operator to
safe actions only).

**Leadership Principle tie-in:** **Invent and Simplify** — the MDP framework reduces a complex
warehouse navigation problem to five well-defined elements (S, A, P, R, γ), making it tractable.
**Bias for Action** — value iteration provides a concrete, computable policy rather than a vague
"navigate well" directive.

---

## Glossary

- **Action-value function Q^π(s,a)** — expected cumulative discounted reward from state $s$,
  taking action $a$, then following policy $\pi$.
- **Bellman equation** — recursive equation expressing the value of a state (or state-action
  pair) in terms of the values of successor states.
- **Bellman expectation equation** — the Bellman equation averaged over actions weighted by the
  policy $\pi$; used for policy evaluation.
- **Bellman optimality equation** — the Bellman equation with max over actions; defines the
  optimal value function $V^*$.
- **Contraction mapping theorem** — guarantees convergence of iterative Bellman updates when
  $\gamma < 1$; the Bellman operator is a contraction with factor $\gamma$.
- **Continuing task** — an RL task with no natural terminal state; the agent interacts forever.
- **Discount factor γ** — controls how much future rewards are valued relative to immediate
  rewards; $\gamma \in [0,1)$.
- **Effective planning horizon** — $\frac{1}{1-\gamma}$; approximately how many steps ahead the
  agent "cares about."
- **Episodic task** — an RL task that breaks into finite episodes from start to terminal state.
- **Greedy policy** — a policy that always chooses the action with the highest Q-value:
  $\pi(s) = \arg\max_a Q(s,a)$.
- **Markov Decision Process (MDP)** — 5-tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$
  defining a sequential decision problem.
- **Markov property** — the future depends only on the present state, not the history.
- **Policy $\pi$** — a rule for choosing actions, either deterministic ($\pi(s) = a$) or
  stochastic ($\pi(a|s)$ = probability).
- **Policy improvement** — constructing a new greedy policy from a value function; guaranteed to
  be at least as good.
- **Policy iteration** — alternating policy evaluation and policy improvement until convergence.
- **Return $G_t$** — cumulative discounted reward from time $t$ onward:
  $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$.
- **Reward hypothesis** — all goals can be described as maximizing expected cumulative reward.
- **State value function V^π(s)** — expected cumulative discounted reward from state $s$ under
  policy $\pi$.
- **Transition probability $P(s'|s,a)$** — probability of reaching state $s'$ from state $s$
  under action $a$.
- **Value iteration** — iteratively applying the Bellman optimality update to converge to $V^*$.

---

## Check yourself

1. Name three specific ways RL differs from supervised learning and explain why each matters.
   *(§1)*
2. Define an MDP as a 5-tuple and explain what each element represents in a grid-world example.
   *(§5)*
3. State the Markov property and give one example where it holds and one where it doesn't.
   *(§5.2)*
4. Explain why γ < 1 is mandatory for continuing tasks but optional for episodic tasks. Compute
   the return for a reward sequence [1, 1, 1, ...] with γ = 0.9 and γ = 1. *(§3, §7.2)*
5. Define V^π(s) and Q^π(s,a) and state their mathematical relationship. *(§8)*
6. Write the Bellman expectation equation for V^π and solve the 2-state MDP example from §9.2.
   *(§9)*
7. Explain why the Bellman optimality equation is nonlinear and cannot be solved by matrix
   inversion. *(§10)*
8. Describe policy iteration's alternating steps and explain why the policy improvement theorem
   guarantees convergence. *(§12)*
9. What is the effective planning horizon for γ = 0.99? For γ = 0.5? *(§7.2)*
10. A robot in a maze gets -1 per step and +100 at the goal. With γ = 0.9 and a path that's 5
    steps long, what is the discounted return? With γ = 0.5? *(§7)*
11. Explain why a purely deterministic policy may fail in exploration-heavy environments and how
    stochastic policies address this. *(§2.2)*
12. State the contraction mapping theorem's role in RL and explain what happens when its
    assumption (γ < 1) is violated. *(§13)*
13. In the grid-world simulation, explain why a random policy produces uniformly poor state values
    while the greedy policy produces meaningful values. *(§9, interactive demo)*

---

## Going deeper

1. **Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed., 2018)** — ⚠️ **named on
   the slides** [slide 148] as the primary textbook reference. `solid`. The standard RL textbook;
   covers everything in this lecture and all subsequent topics in the module.

2. **David Silver's RL course (UCL/DeepMind)** — ⚠️ **named on the slides** [slide 148] as
   recommended video lectures. `solid`. Free lecture series that mirrors this module's structure
   closely.

3. **Stanford CS234: Reinforcement Learning** — ⚠️ **named on the slides** [slide 148]. `solid`.
   More mathematically rigorous than Silver's course.

4. **Berkeley CS285: Deep Reinforcement Learning** — ⚠️ **named on the slides** [slide 148]. `hard`.
   Focuses on deep RL methods (DQN, policy gradients, model-based RL) — the natural next step after
   this foundational lecture.

5. **Mnih et al. (2015), "Human-level control through deep reinforcement learning" (DQN)** —
   ⚠️ not named on the slides but referenced implicitly as the Atari result. `hard`. The paper
   that launched deep RL.

6. **OpenAI Five (2019)** — ⚠️ not named on the slides but referenced as the Dota 2 result.
   `solid`. Demonstrates self-play at scale.

7. **OpenAI Spinning Up (spinningup.openai.com)** — ⚠️ **named on the slides** [slide 148,
   "References & Further Reading" — corrected from a previous, wrong citation to slide 72, which is
   actually the discount-factor demo] as recommended online resource. `solid`. A practical
   introduction to deep RL algorithms.

> ⚠️ **verify this** — items 1–4 are explicitly recommended on the lecture's own slides. Items
> 5–6 are referenced implicitly through the applications discussion but not cited by author —
> title/venue/year not on a slide, so only the standard bibliographic record backs them.
>
> ✅ **Citation check (enhancement pass).** Independently verified against primary/secondary sources:
> Sutton & Barto's *Reinforcement Learning: An Introduction* is the 2018 2nd edition (MIT Press);
> Mnih et al. (2015) "Human-level control through deep reinforcement learning" is *Nature* **518**,
> 529–533 (DOI 10.1038/nature14236) — title, author list, journal, volume, and year all confirmed
> exact. No corrections needed to any citation in this list.
> treat as suggested background reading.
