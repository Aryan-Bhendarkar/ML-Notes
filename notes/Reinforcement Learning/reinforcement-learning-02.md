---
title: "Reinforcement Learning — Part 2: From Planning to Learning"
topic: reinforcement-learning
lecture: 25
source: "output/Lecture_25 - Module 8 Reinforcement Learning Part 2"
slides: 70
---

# Reinforcement Learning — Part 2: From Planning to Learning

> Runtime ~55:00. Originally drafted from `slides_deduped/` (70 slides) with OCR + transcript
> verification, then **audited against the raw 134-frame capture** in
> `output/Lecture_25 - Module 8 Reinforcement Learning Part 2/` per this project's standard
> methodology (see project memory `slides-deduped-is-lossy`) — one real gap (Thompson Sampling) and
> one under-transcribed demo (the grid-world policy-evaluation sweeps, now in §3.1) were found and
> fixed; a subsequent quality-review pass independently re-verified the sweep numbers in §3.1 and
> §3.3 pixel-for-pixel against the raw frames and confirmed both are exact (see
> `QUALITY_REVIEW.md`). Instructor: previously credited as **"Naman Kanan"**, but no slide in the
> deck shows a nameplate with that name. The lecture's own interactive-demo slides
> (`slide_015.jpg`–`slide_052.jpg`) repeatedly show a browser address bar reading
> `file:///Users/nkhetan/RL_Interactive_Examples/...` — the macOS account name **"nkhetan"** is the
> only direct evidence of the instructor's identity anywhere in the raw capture, suggesting a surname
> **Khetan** (first initial "N") rather than "Naman Kanan." ⚠️ **This is still not a full
> confirmation** — no slide shows a full first name — so the credit below is stated as inferred, not
> confirmed. This lecture is the bridge from Part 1's definitions (MDPs, Bellman equations) to actual
> algorithms that solve problems — starting from the ideal case (full model known) and progressively
> removing assumptions until arriving at DQN, which learns Atari games from raw pixels.

---

## What you'll understand after reading this

1. **Explain the three limitations of dynamic programming** (needs complete model, curse of
   dimensionality, discrete spaces only) and why we need model-free methods.
2. **Solve a small MDP using value iteration** and explain why it converges (contraction mapping).
3. **Define the exploration-exploitation tradeoff** and implement epsilon-greedy, softmax, and
   UCB strategies, explaining when each is preferred.
4. **Explain why multi-armed bandits are the simplest RL problem** and compute regret for a given
   strategy.
5. **Distinguish contextual bandits from full RL** and explain why industry prefers the bandit
   setting (immediate rewards, no sequential consequences).
6. **Derive the TD(0) update rule** from the Bellman equation and explain bootstrapping,
   the TD target, and the TD error δ.
7. **Explain why TD learning is "the best of both worlds"** compared to DP and Monte Carlo.
8. **Implement SARSA and Q-learning** and explain the single-word difference (max vs. actual a')
   that changes everything.
9. **Explain the on-policy vs. off-policy distinction** using behavior and target policies, and
   why off-policy enables experience replay.
10. **State the deadly triad** (off-policy + function approximation + bootstrapping) and explain
    why DQN uses all three yet remains stable.
11. **Describe the DQN architecture** (84×84×4 input, 3 conv layers, FC layer, Q-output) and
    explain each design choice.
12. **Explain the two DQN tricks** (experience replay + target network) and how each solves a
    specific problem.
13. **Name and describe four DQN variants** (Double DQN, Dueling, Prioritized Replay, Rainbow)
    and what each improves.

---

## Before we start: what you need to know

### Prerequisite 1 — Part 1 vocabulary

This lecture assumes familiarity with MDPs (5-tuple), Bellman expectation and optimality
equations, V(s) and Q(s,a), policy iteration, and value iteration from Part 1. If these are
unfamiliar, read Part 1 first.

### Prerequisite 2 — Neural network basics

DQN (§12) replaces a Q-table with a neural network. You need to understand: forward pass
(input → output), backpropagation (computing gradients), and loss functions (mean squared error).
No deep architectural knowledge is needed — the DQN network is just 3 convolutional layers +
1 fully connected layer.

### Prerequisite 3 — Gradient descent

The DQN training loop uses gradient descent to minimize a loss. You need to know: we adjust
weights θ in the direction that reduces loss, controlled by a learning rate α.

---

## The big picture

This lecture follows an **organized journey where each step removes an assumption:**

```
Dynamic Programming (§2–§3)
   Assumption: COMPLETE model (P, R known)
   Remove it → need model-free methods
        │
        ▼
Exploration vs. Exploitation (§4)
   Problem: what to try when you don't know the world
   Strategies: ε-greedy, softmax, UCB
        │
        ▼
Multi-Armed Bandits (§5)
   Simplest RL: no states, no transitions, pure exploration
   Bandits are the most widely deployed RL in industry
        │
        ▼
Temporal Difference Learning (§6–§7)
   The HEART of the course
   Learn from experience, no model, update every step
   TD(0), SARSA (on-policy), Q-learning (off-policy)
        │
        ▼
On-policy vs. Off-policy (§8)
   The distinction that unlocks data efficiency
   Off-policy → experience replay → DQN
        │
        ▼
Deep Q-Networks (§9–§11)
   Replace Q-table with neural network
   Two tricks: replay + target network
   Result: superhuman Atari from raw pixels
```

Two threads to hold the entire time:
1. **Almost every algorithm is a version of the same idea:** update my estimate toward reward +
   discounted future value.
2. **There is constant tension** between using what you already know (exploit) and discovering
   something better (explore).

---

## 1. Recap: What Is Reinforcement Learning?

> *"RL is learning what to do — how to map situations to actions — so as to maximize a numerical
> reward signal."* — Sutton & Barto

Two features unique to RL:
- **Trial-and-error search** — no labels, must discover by trying.
- **Delayed reward** — actions affect not just immediate reward but all future rewards.

Part 1 gave us the language (MDPs, Bellman equations). Part 2 is where the equations turn into
**algorithms that actually solve problems.**

---

## 2. Dynamic Programming — Planning with a Known Model

### 2.1 What DP means in RL

DP computes the optimal policy when you have **full knowledge** of the environment — the complete
MDP (S, A, P, R, γ).

> **Analogy:** a GPS with a complete map and live traffic data. It doesn't drive around trying
> routes (that would be learning). It computes the shortest route directly from the map. That's
> DP — planning on a known model.

### 2.2 The three limitations (why DP isn't enough)

| Limitation | Why it's a problem | Example |
|-----------|-------------------|---------|
| **Needs complete model** | Real agents don't know P(s'\|s,a) | A robot doesn't have physics equations for every surface |
| **Curse of dimensionality** | Sweeps over ALL states every iteration | Chess: ~10⁴⁷ states. Go: ~10¹⁷⁰. Atari: effectively infinite |
| **Discrete spaces only** | Continuous control has infinite states | Robot arm joint angles are continuous |

> ⚠️ **The punchline that sets up the entire rest of the talk:** both DP methods require the
> full model P(s'|s,a). In the real world, we almost never have that. We need methods that
> **learn from experience without a model.**

---

## 3. Policy Iteration and Value Iteration (Review + Demo)

### 3.1 Policy evaluation (iterative)

Given a fixed policy π, compute V^π(s) by repeatedly applying the Bellman expectation update
until convergence. Each sweep pushes reward information one step backward through the state space.

🧪 **Worked example — evaluating a bad policy honestly.** The demo's grid: γ=0.9, step reward −1,
GOAL +10, PIT −10, policy = **uniform random** (¼ probability each direction, so it often stumbles
into the pit). Every $V^\pi(s)$ starts at 0.00, then the start cell's value is recomputed each
sweep by averaging its four neighbors' *current* values:

| Sweep | $V^\pi(\text{start})$ | Sweep | $V^\pi(\text{start})$ |
|---|---|---|---|
| 1 | $\tfrac14(-1.00-1.00-1.00-10.00)=$ **−3.25** | 10 | **−15.58** |
| 2 | $\tfrac14(-1.90-3.93-3.93-12.93)=$ **−5.67** | 15 | **−17.96** |
| 5 | **−10.88** | 20 | **−19.23** |

**This is policy evaluation working exactly as designed, on a genuinely bad policy.** A uniform
random walker in a grid with a −10 pit spends much of its (infinite-horizon, γ=0.9) future
blundering into that pit, so its true value keeps drifting more negative for many sweeps before
settling — the algorithm isn't broken, the *policy* is bad, and the numbers say so honestly. This
is the same iterative Bellman-expectation update from Part 1 §11, just watched sweep by sweep on
a policy worth evaluating critically instead of trusting.

```interactive
type: simulator
title: Policy Evaluation — Sweeping a Bad Policy Honestly
concept: Iterative policy evaluation under a fixed policy (here: uniform random)
control: Step through sweeps one at a time (or auto-run) on the 3×3 grid
observe: Every cell's V^π(s) updates from its neighbors' current values each sweep; the start cell drifts from 0.00 down toward its converged value around −20.7 as the pit's influence propagates backward
insight: A worsening value estimate isn't a broken algorithm — it's the algorithm correctly discovering that a uniform-random walker in a grid with a −10 pit is, in expectation, a bad plan
fallback: The sweep table above (sweeps 1, 2, 5, 10, 15, 20 → −3.25, −5.67, −10.88, −15.58, −17.96, −19.23 at the start cell) shows the same convergence a static reader can follow by hand.
```

### 3.2 Policy iteration

Alternate two steps:
1. **Evaluate:** compute V^π for current policy.
2. **Improve:** at every state, switch to the greedy action: π'(s) = argmax_a Q^π(s,a).

**Policy Improvement Theorem:** the new policy is guaranteed to be ≥ the old one. Since there
are finitely many policies and each round strictly improves, the process **must converge to π\*.**

**Downside:** Step 1 (full evaluation) is expensive — many sweeps over all states before each
improvement.

🧪 **Worked example — a genuinely bad policy, evaluated honestly, then fixed in one flip.** This
is a *different* demo from §3.1's (same grid: γ=0.9, step −1, GOAL +10, PIT −10, but this one
starts from a **deterministic** policy: every cell points **all → Right**). The bottom-left cell's
arrow therefore points straight at the pit:

| Sweep | $V(\text{start})$ | Sweep | $V(\text{start})$ |
|---|---|---|---|
| 1 | $-10+0.9(0.00)=$ **−10.00** | 4 | **−34.39** |
| 2 | $-10+0.9(-10.00)=$ **−19.00** | 56 | −99.73 |
| 3 | $-10+0.9(-19.00)=$ **−27.10** | 57 | **≈ −100.00** (converged) |

**Evaluation on this policy crashes almost to $-100$** — the theoretical floor for a state that
never escapes a $-10$ step (its own discounted self-loop value is $-10/(1-0.9)=-100$), because the
policy sends the agent to fall into the pit, restart, and fall in again, forever. **Improve** then
looks at this converged $V^\pi$ and asks, for the bottom-left cell, "would any other action beat
'go right into the pit forever'?" — yes: going up leads toward better-valued cells instead. The
arrow flips from → to ↑. Re-evaluating under the new policy and repeating Improve converges, by
iteration 5, to exactly the same optimal values §3.3's Value Iteration demo reaches independently:
$V^*(\text{start})=4.58$ (and $6.20$, $8.00$, $10.00$ elsewhere in the grid) — **two structurally
different algorithms (alternate full evaluation vs. one-step max) landing on the identical answer**,
which is exactly the guarantee the policy improvement theorem promises.

> ⚠️ **This is a separate interactive demo from §3.1's**, not the same one re-run: §3.1 shows a
> **uniform-random** policy settling to a *moderately* bad value (~−20.7) because random wandering
> only occasionally wanders into the pit; this section's policy is *deterministically* aimed at the
> pit, so its evaluated value crashes far lower, toward the theoretical worst case of −100. Both are
> real, both are useful teaching points, and neither is a contradiction of the other.

```interactive
type: simulator
title: Policy Iteration — Crash, Then One Flip Fixes It
concept: Policy iteration's evaluate → improve loop, on a policy that starts genuinely bad
control: Step through evaluation sweeps under the deterministic "all → Right" policy, then trigger Improve
observe: V(start) crashing sweep by sweep toward roughly −100 while evaluation faithfully reports the policy's true (terrible) worth; then, after Improve, the bottom-left cell's arrow flips from → to ↑ and the value climbs back up over the next few evaluate/improve rounds
insight: Evaluation's job is only to report the truth about whatever policy it's given, however bad — Improve is what fixes it, and here a single flipped arrow is enough to send the policy to the same optimum value iteration reaches independently
fallback: The worked-example table above (sweeps 1–57: −10.00 → −19.00 → −27.10 → −34.39 → roughly −100.00) followed by the post-Improve convergence to 4.58/6.20/8.00/10.00, matching §3.3's value-iteration result exactly.
```

### 3.3 Value iteration — the efficient shortcut

**Insight:** why fully evaluate before improving? Fold both into one step:

$$V(s) \leftarrow \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) \cdot V(s') \right]$$

Replace the policy-weighted average with a **max** — merges evaluation and improvement into a
single update.

| | Policy Iteration | Value Iteration |
|---|---|---|
| Outer loops | Fewer | More |
| Each loop | Full evaluation (expensive) | Single sweep (cheap) |
| Preferred for | Small problems | Large problems |

**Demo:** value iteration on the 3×3 grid — watch values spread outward from +10 goal, one ring
per sweep, converging in ~5 sweeps to V(start) ≈ 4.58.

```interactive
type: animation
title: Value Iteration — Values Spreading From the Goal
concept: Value iteration folds evaluation and improvement into one max-based update
control: Step forward sweep by sweep on the 3×3 grid
observe: Values spread outward from the +10 goal cell one ring per sweep, converging in ~5 sweeps
insight: Because every sweep already takes the max over actions, there's no separate improvement phase to wait for — the optimal policy falls out for free the moment V converges, unlike §3.2's alternating evaluate/improve loop
fallback: The description above — values spread outward from the goal one ring per sweep, converging in ~5 sweeps to V(start) ≈ 4.58, the same fixed point §3.2's policy iteration reaches by a structurally different route.
```

---

## 4. Exploration vs. Exploitation

### 4.1 The fundamental tension

| | **Exploit** | **Explore** |
|---|---|---|
| What | Pick the action with highest estimated value | Pick a more uncertain action to gather information |
| Upside | Maximize immediate expected reward | Might discover something much better |
| Downside | Stuck in local optimum forever | Costs reward now (new thing might be worse) |
| Analogy | Always go to your favorite restaurant | Try the new restaurant that just opened |

**The practical answer:** explore more early (when you know nothing and information is precious),
exploit more later (once you understand the landscape).

**Amazon framing:** "Do we show the customer the product we *know* they click on (exploit), or
test a new listing that might convert even better (explore)?"

### 4.2 Strategy 1: ε-Greedy

With probability (1-ε): pick the greedy action (highest Q-value). With probability ε: pick a
random action.

**Decaying ε:** start near 1 (almost pure exploration), slowly decay toward 0.01 (explore early,
exploit late). DQN uses ε decaying from 1.0 to 0.1 over 1 million frames.

**Pros:** trivially simple, no assumptions about rewards, shockingly strong baseline.
**Cons:** exploration is *uniform* — a clearly terrible action gets the same exploration
probability as a promising one. Wastes exploration budget.

### 4.3 Strategy 2: Softmax (Boltzmann exploration)

$$P(a) = \frac{e^{Q(a)/\tau}}{\sum_{a'} e^{Q(a')/\tau}}$$

τ (temperature) controls the balance: τ → ∞ means uniform random (explore); τ → 0 means always
best (exploit).

**Advantage over ε-greedy:** explores in proportion to value — a slightly worse action gets
slightly less probability, not equal probability.

### 4.4 Strategy 3: UCB (Upper Confidence Bound)

**Words before symbols:** pick the action whose estimated value, plus a bonus for how little we
know about it, is highest — the bonus shrinks the more that action has already been tried and
grows the more total rounds have passed.

$$a^* = \arg\max_a \left[ Q(a) + c \sqrt{\frac{\ln t}{N(a)}} \right]$$

| Symbol | Read it as | What it means |
|---|---|---|
| $Q(a)$ | "estimated value of arm $a$" | Average reward observed from arm $a$ so far (the exploit term) |
| $c$ | "exploration constant" | A tunable constant controlling how large the uncertainty bonus is; higher $c$ → more exploration |
| $t$ | "round number" | How many pulls (across all arms) have happened so far |
| $N(a)$ | "times arm $a$ was pulled" | The count in the bonus's denominator — small $N(a)$ inflates the bonus |

The second term is an **uncertainty bonus:** N(a) is how many times action a has been tried.
Rarely tried → big bonus → try it. Frequently tried → small bonus → exploit it.

**Slogan:** "Be optimistic in the face of uncertainty."

**No ε schedule, no temperature** — self-adapting. Provable regret bounds of O(√(KT ln T)).

### 4.4a Thompson Sampling

> ⚠️ **Source note.** The lecture's own closing summary states directly: *"Multi-Armed Bandits:
> Simplest RL. **Thompson Sampling dominates in industry.**"* — but the deck has no dedicated
> slide walking through its mechanism the way it does for ε-greedy/softmax/UCB. This subsection is
> written from standard background knowledge to fill that gap, not transcribed from a slide.

> **Thompson Sampling** — a Bayesian exploration strategy: maintain a probability distribution
> (a *belief*) over each arm's true payout rate, and at each step, **sample** one plausible payout
> value from each arm's belief distribution, then pull the arm whose sampled value is highest.

For a bandit with binary (success/fail) rewards, each arm's belief is typically a **Beta
distribution** $\text{Beta}(\alpha_a,\beta_a)$, updated after every pull: a success increments
$\alpha_a$, a failure increments $\beta_a$. Arms with more data have tighter (more confident)
distributions; arms with little data have wide, uncertain distributions.

**Why it explores automatically, with no tunable schedule:** an arm that's been pulled rarely has a
wide belief distribution, so its *sampled* value is highly variable — it will occasionally sample
very high and get tried, even if its current average looks mediocre. An arm that's been pulled
often and performs well has a narrow, high-centered distribution — it gets sampled high (and thus
chosen) consistently. Exploration and exploitation fall out of the *same* sampling step, rather than
being controlled by a separate hyperparameter like $\varepsilon$ or $\tau$.

> 🎯 **Why it tends to beat UCB and ε-greedy in practice.** Thompson Sampling's exploration is
> calibrated to genuine posterior uncertainty rather than a hand-tuned bonus term or a fixed random
> rate — it naturally explores *more* for arms with little data and *less* for arms it's already
> confident about, without any schedule to tune. This is consistent with the strong, repeated
> empirical results favoring it in real-world bandit deployments (ad selection, recommendations)
> that the lecture's summary slide is referring to.

### 4.5 Comparison

| Strategy | Complexity | Tuning | Exploration quality |
|----------|-----------|--------|-------------------|
| ε-greedy | Trivial | ε schedule | Uniform (wasteful) |
| Softmax | Simple | Temperature τ | Proportional to value |
| UCB | Moderate | None (self-adapting) | Optimal regret bounds |
| Thompson Sampling | Moderate (needs a posterior model) | None (self-calibrating) | Posterior-calibrated; often best empirically |

---

## 5. Multi-Armed Bandits

### 5.1 The simplest RL problem

K slot machines ("arms"), each with an unknown payout rate. Pull one arm, get a reward. Goal:
maximize total reward over T pulls.

**Why it's the simplest RL:** no states (same situation every time), no transitions (pulling
doesn't change the world), no planning — pure exploration vs. exploitation.

**Scoreboard:** Regret = (best arm's mean × T) − (your total reward). Lower regret = better.
Best algorithms achieve O(√T) regret → average regret per pull shrinks to zero.

**Industry applications:** A/B testing, ad selection, clinical trials, product recommendations,
search ranking, dynamic pricing — anywhere you repeatedly choose under uncertainty with quick
feedback.

### 5.2 Contextual bandits

A plain bandit gives the same answer to everyone. A **contextual bandit** asks: "which arm is
best *for this context?*"

1. Observe context (user features, time of day, location).
2. Choose action from K arms.
3. Receive **immediate** reward.
4. Learn which actions work best for which contexts.

**Key distinction from full RL:** the reward is immediate — your action does NOT change the
future state. No delayed consequences, no long-horizon credit assignment.

**The spectrum:**

```
MAB (no context) → Contextual Bandits (context, immediate reward) → Full RL (actions affect future)
```

**Amazon example:** user browsed electronics, it's evening, they're in Seattle → context goes
into model → picks which of 5 recommendation widgets to show → user clicks → model updates.

---

## 6. Temporal Difference Learning — The Heart of the Course

### 6.1 TD between DP and Monte Carlo

| | **DP** | **Monte Carlo** | **TD Learning** |
|---|---|---|---|
| Needs model? | Yes | **No** | **No** |
| Wait for episode end? | No (every step) | **Yes** (full episode) | **No** (every step) |
| Bootstraps? | Yes | No | **Yes** |

TD takes the **best of both worlds:** model-free like MC, updates every step like DP, and
bootstraps.

**Bootstrapping:** updating an estimate using another estimate (rather than waiting for the true
final outcome). DP bootstraps off a known model. TD bootstraps off its own current guesses about
the future.

### 6.2 The TD(0) update rule

> *"If you remember one equation from today, make it this one."*

$$V(s) \leftarrow V(s) + \alpha \big[ R + \gamma V(s') - V(s) \big]$$

| Symbol | Meaning |
|--------|---------|
| V(s) | Current estimate of state s's value |
| α | Learning rate (small step, e.g., 0.1) |
| R | Actual reward received |
| γ | Discount factor |
| V(s') | Current estimate of the next state's value |
| **R + γV(s')** | **TD target** — fresh, usually better estimate of what s is worth |
| **R + γV(s') − V(s)** | **TD error δ** — the surprise, the learning signal |

**Plain English:** "I thought state s was worth V(s). Then I took a step, got reward R, and
landed in s' which I currently think is worth V(s'). So actually s is worth about R + γV(s').
I was off by δ. Let me nudge V(s) a small fraction α toward the better target."

**Key:** we update **immediately** after one step, using our own estimate of the future. No
true return needed. No waiting for episode end.

🧪 **Worked example — the demo's own first 10 steps.** Same grid as §3 (α=0.1, γ=0.9, step −1,
GOAL +10, PIT −10), random-walk policy, all $V(s)=0.00$ initially, agent starting bottom-left:

```
All values 0. Agent at the start cell.
step 1 ←: δ = (−1 + 0.9×−0.10) − 0.00  = −1.00; V(cell) ← −0.10
step 2 ↑: δ = (−1 + 0.9×0.00)  − −0.10 = −0.90; V(cell) ← −0.19
step 3 ←: δ = (−1 + 0.9×−0.10) − 0.00  = −1.00; V(cell) ← −0.10
step 4 →: δ = (−1 + 0.9×0.00)  − −0.10 = −0.90; V(cell) ← −0.19
step 5 →: δ = (−1 + 0.9×0.00)  − 0.00  = −1.00; V(cell) ← −0.10
step 6 →: δ = (−1 + 0.9×−0.10) − 0.00  = −1.00; V(cell) ← −0.10
step 7 ←: δ = (−1 + 0.9×−0.10) − −0.10 = −0.99; V(cell) ← −0.20
step 8 ↑: δ = (−1 + 0.9×0.00)  − −0.10 = −0.90; V(cell) ← −0.19
step 9 ↓: δ = (−1 + 0.9×−0.19) − 0.00  = −1.17; V(cell) ← −0.12
step 10 ↓: δ = (−10 + 0.9×−0.19) − −0.19 = −9.98; V(cell) ← −1.19  (fell in PIT → back to start)
```

Notice how tiny, ordinary steps (δ ≈ −1) nudge V(cell) by a fraction α of the error, but falling
in the pit produces one sharp δ ≈ −10 update — the same update rule handles both without any
special-casing. After 80 steps (1 full episode) of this random walk, the V-table reads: start
cell −5.52, the cell directly above the pit −3.58, the cell adjacent to the goal −0.05 — negative
almost everywhere except right next to the goal, the same overall shape §3.1's full-sweep policy
evaluation reaches, but built up here from one noisy real step at a time instead of an averaged
sweep over the whole grid.

```interactive
type: simulator
title: TD(0) Learning — One Step at a Time
concept: The TD(0) update learns V(s) online from one step of real experience, with no model and no waiting for the episode to end
control: Click "One step" to take a single step of the random-walk policy on the 3×3 grid, or "Auto-walk" to run many steps continuously
observe: The V-table cell the agent just left nudges toward its TD target after every single step; an ordinary step produces a small nudge, falling into the pit produces one sharp negative jump
insight: TD(0) never waits for the episode to end or needs a model of the grid — every step, however small, moves exactly one cell's estimate a little closer to the truth, and the same rule handles both routine steps and the rare disaster
fallback: The step-by-step trace above (steps 1–10, with the fall into the PIT at step 10) plus the resulting V-table after 80 steps (start −5.52, above-pit −3.58, next-to-goal −0.05).
```

### 6.3 Why TD is powerful

- **Model-free:** no transition probabilities needed.
- **Online:** learn mid-episode, don't wait for end.
- **Works in continuing tasks** where MC simply can't (infinite episodes).
- **Lower variance than MC** (but introduces some bias from bootstrapping).
- **Most widely used family of RL algorithms in practice.**

---

## 7. SARSA and Q-Learning — TD Control

### 7.1 SARSA — On-Policy TD Control

To choose actions, we need Q(s,a) — the value of *taking action a in state s*.

$$Q(s,a) \leftarrow Q(s,a) + \alpha \big[ R + \gamma Q(s', a') - Q(s,a) \big]$$

**The name spells out the update:** State, Action, Reward, next State, next Action.

**The crucial word: ACTUALLY.** a' is the action we *will* take next, chosen by our real
ε-greedy policy. If the policy explores, SARSA values *include* that exploration cost. It
learns the value of the policy it's actually following — that's what **on-policy** means.

**Algorithm sketch:**
1. In state s, pick action a (ε-greedy).
2. Take a, get reward R, arrive at s'.
3. In s', pick next action a' (ε-greedy) — uses **same** policy.
4. Update Q(s,a).
5. s ← s', a ← a', repeat.

**Consequence:** SARSA learns a **safe policy** that accounts for its own exploration.

### 7.2 Q-Learning — Off-Policy TD Control

> *"The most important single algorithm in the course."*

$$Q(s,a) \leftarrow Q(s,a) + \alpha \big[ R + \gamma \max_{a'} Q(s', a') - Q(s,a) \big]$$

**One word different from SARSA:** max instead of a'.

- SARSA uses Q(s', a') — value of what we'll *actually* do next.
- Q-learning uses max Q(s', a') — value of the *best possible* next action.

**What this unlocks:**
- The agent can explore widely (even random actions) and still learn the optimal policy,
  because the learning target ignores exploration.
- Can learn from data collected by other agents or humans.
- The thing it's learning *about* (optimal policy) is different from the thing *generating*
  its behavior (exploratory policy) — that's exactly what **off-policy** means.

**Watkins 1989** — one of the most influential papers in RL history.

### 7.3 SARSA vs. Q-Learning — the cliff example

| | SARSA | Q-Learning |
|---|---|---|
| Learns | Longer, safer path (stays away from edge) | Shortest optimal path (right along edge) |
| Accounts for | Its own ε% exploration risk | Assumes optimal behavior afterward |
| During training | Safe — avoids cliff edge | Risky — random 10% occasionally falls off |
| Final policy | Suboptimal but safe | Optimal but risky training |

> **One-liner:** SARSA = safe learner. Q-learning = optimal learner.

🧪 **Worked example — one concrete update, both ways.** In the demo's grid, the agent at a
centre cell takes **Right** (reward −1) and lands one cell from the goal, where the four
Q-values are $Q(\uparrow)=3.0$, $Q(\downarrow)=8.0$, $Q(\leftarrow)=4.0$, $Q(\rightarrow)=3.5$.
Because of ε-exploration, the action **actually** sampled next is $\uparrow$ ($Q=3.0$) — not the
best one. Current $Q(\text{centre},\text{Right})=5.0$, with $\alpha=0.4$, $\gamma=0.95$:

- **SARSA:** target $= -1 + 0.95\times Q(\uparrow) = -1+0.95\times3.0 = 1.85$;
  $Q \leftarrow 5.0 + 0.4\times(1.85-5.0) = \mathbf{3.74}$
- **Q-learning:** target $= -1 + 0.95\times\max_{a'}Q = -1+0.95\times8.0 = 6.60$;
  $Q \leftarrow 5.0 + 0.4\times(6.60-5.0) = \mathbf{5.64}$

**Same experience, two different updated values** — SARSA lands lower (3.74) because its target
"expects" the ε-exploratory move that actually happened, while Q-learning lands higher (5.64)
because its target always assumes the best move happens next, regardless of what the agent
actually does. After training both for 800 episodes on this grid, the cell **just above the
pit** ends up valued at **7.32** under SARSA vs. **8.50** under Q-learning — SARSA keeps pricing
in its own ε-chance of slipping into the pit from that cell, so it stays measurably more cautious
near the edge, exactly the safe-vs-optimal split the table above describes in words.

```interactive
type: simulator
title: SARSA vs. Q-Learning — the Same Experience, Two Different Updates
concept: The single max-vs-actual difference between SARSA's and Q-learning's targets, made concrete on one shared transition
control: Trigger "Show both updates" on the one fixed transition above, or "Train both" for 800 episodes and compare the two learned policies side by side on the grid
observe: SARSA's target uses whichever action ε-greedy actually samples next (even a bad one); Q-learning's target always uses the best next action on paper, regardless of what's actually taken
insight: Identical experience, identical starting Q-value, two different updated numbers (3.74 vs. 5.64) — the gap is entirely the max vs. actual difference, and it's what makes SARSA hug a safer path around hazards while Q-learning cuts the optimal but riskier one
fallback: The one-concrete-update numbers above (SARSA → 3.74, Q-learning → 5.64) and the trained-policy comparison (cell above the pit: 7.32 under SARSA vs. 8.50 under Q-learning).
```

---

## 8. On-Policy vs. Off-Policy

### 8.1 The core distinction

Two policies quietly at play in any RL algorithm:

- **Behavior policy b:** the one *generating* the data (what the agent actually does).
- **Target policy π:** the one you're trying to *learn/improve.*

| | **On-policy (b = π)** | **Off-policy (b ≠ π)** |
|---|---|---|
| Learning | About the strategy I'm following | The optimal strategy while doing something else |
| Algorithm | SARSA | Q-learning |
| Pro | Stable; accounts for exploration noise | Can learn from old data, other agents, human demos |
| Con | Data goes stale when policy changes; can't reuse old experience | Can be unstable (deadly triad) |
| Key enabler | — | **Experience replay** |

> **Theme to carry forward:** off-policy is harder but far more powerful because it unlocks
> data efficiency.

> 👉 *See also:* estimating a **new** policy's value from data collected under an **old**
> behavior policy — off-policy evaluation — is itself a causal-inference problem (the "treatment"
> is the new policy's action, the "outcome" is the reward); see
> [Causal Inference Part 3, §12.3](../Causal%20Inference/causal-inference-03.md) for the
> counterfactual framing of off-policy evaluation and credit assignment.

---

## 9. Experience Replay and the Deadly Triad

### 9.1 Experience Replay — the superpower

Only possible because Q-learning is off-policy:

1. Store every experience (s, a, r, s') in a big memory buffer (~1 million transitions).
2. To learn: randomly sample old experiences and train on them.
3. Each experience gets reused 10–100 times — incredibly data efficient.

**Why on-policy can't do this:** their data becomes "stale" the instant the policy shifts.
SARSA needs the action its current policy would take next — old data has the wrong a'.

### 9.2 The Deadly Triad — the warning

When you combine ALL THREE of:
1. **Off-policy** learning (data from different policy than what you're learning)
2. **Function approximation** (neural network instead of table)
3. **Bootstrapping** (updating estimates from other estimates)

...the learning can **diverge** — Q-values exploding to infinity.

**The subtle part:** any TWO of three are fine. It's only when all THREE come together that
things blow up.

> ⚠️ **DQN uses all three** — and adds two clever tricks to keep it stable (§11).

---

## 10. Deep Q-Networks (DQN) — The Payoff

### 10.1 Why neural networks?

Tabular Q-learning keeps one entry for every (state, action) pair:
- Tic-tac-toe: ~5,000 states ✓
- Grid world: 500 states ✓
- **Atari screen: 210×160×3 pixels = more states than atoms in the universe** ✗

A table gives **zero generalization** — two nearly identical screens get completely unrelated
values. The solution: replace the table with a neural network that **generalizes** — similar
states get similar Q-values.

$$Q(s, a; \theta) \approx Q^*(s, a)$$

One forward pass gives Q-values for ALL actions at once.

### 10.2 Three problems with naive deep Q-learning

Just swapping the table for a neural net **does not work.** Here's why:

| Problem | What goes wrong | Result |
|---------|----------------|--------|
| **Correlated training data** | Consecutive game frames are nearly identical; neural nets need diverse, i.i.d. samples | Catastrophic forgetting of earlier lessons |
| **Non-stationary targets** | Q-learning target R + γ max Q(s',a'; θ) depends on θ — the very weights we're updating. Every update moves the target. | Oscillation, divergence |
| **Deadly triad** | Off-policy + function approximation + bootstrapping | Instability |

---

## 11. The Two DQN Tricks

### 11.1 Trick 1: Experience Replay

**Fixes Problem 1 (correlated data).**

- Store transitions in a circular buffer (~1 million).
- Train on random mini-batches of 32.
- A random draw from 1M experiences is **diverse and decorrelated** — a single batch might
  contain frames from level 1, a death moment, and a powerup, all mixed together.
- **Bonus:** data efficiency (each experience reused many times) and smoother learning.

### 11.2 Trick 2: Target Network

**Fixes Problem 2 (moving targets).**

Two copies of the network:
- **Online network θ:** updated every training step (the "student").
- **Target network θ⁻:** frozen copy, updated only every 10,000 steps (the "teacher").

$$\text{Loss} = \big( R + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \big)^2$$

The frozen teacher provides a **stable target** for 10,000 steps, giving the student time to
converge.

> **Analogy:** studying from a textbook that only gets a new edition each semester, not one that
> rewrites itself every night.

### 11.3 The two tricks together

| Trick | Problem it fixes | Mechanism |
|-------|-----------------|-----------|
| Experience replay | Correlated data | Random sampling from diverse buffer |
| Target network | Moving target | Frozen copy for stable objectives |

Together: stable deep Q-learning that actually works.

---

## 12. DQN Architecture and Results

### 12.1 The network

```
Input:  84 × 84 × 4  (last 4 frames stacked, grayscale)
  ↓
Conv2d: 32 filters, 8×8, stride 4, ReLU
  ↓
Conv2d: 64 filters, 4×4, stride 2, ReLU
  ↓
Conv2d: 64 filters, 3×3, stride 1, ReLU
  ↓
Fully connected: 512 neurons, ReLU
  ↓
Output: Q(s, a) for each action
```

**Design choices explained:**
- **4 stacked frames:** a single frame is ambiguous (is the ball going up or down?). 4 frames
  encode velocity and direction.
- **Grayscale 84×84:** color rarely matters for gameplay; smaller = faster training.
- **Convolutions:** images have spatial structure; CNNs detect patterns regardless of position.
- **All Q-values at once:** one forward pass gives Q for every action — pick argmax. Much faster
  than running the network once per action.
- **Same architecture for all 49 games.** No per-game features or tuning.

### 12.2 The complete algorithm

```
Initialize replay buffer D (capacity 1,000,000)
Initialize online Q-network θ with random weights
Initialize target network θ⁻ = θ

Loop forever:
  1. OBSERVE: stack last 4 frames as input (84×84×4)
  2. CHOOSE: ε-greedy from Q(s; θ), ε decaying from 1.0 to 0.1
  3. EXECUTE: take action in game, get reward R and next screen s'
  4. STORE: transition (s, a, R, s') in buffer D
  5. SAMPLE: random mini-batch of 32 from D
  6. COMPUTE targets: y = R + γ max_a' Q(s', a'; θ⁻)  [frozen net]
  7. LEARN: gradient descent to minimize (y - Q(s,a; θ))²
  8. Every 10,000 steps: copy θ → θ⁻
```

### 12.3 Results — DeepMind 2015 (Nature)

| Game | DQN Score | Human Score | DQN/Human |
|------|-----------|-------------|-----------|
| Breakout | 4,815 | 368 | **13× superhuman** |
| Pong | 20.9 | 20.2 | Above human |
| Space Invaders | 1,976 | 1,652 | Above human |
| Enduro | 831 | 309 | **2.7× human** |
| Seaquest | 5,286 | 20,182 | Below human |

**Headline:** beats human-level on **29 of 49 games** using the same algorithm, architecture,
and hyperparameters for all games. From raw pixels only — no game rules, no hand-crafted features.

**Notable learned strategies:**
- **Breakout:** learned to tunnel behind the wall for maximum points.
- **Pong:** learned to angle shots to be unreturnable.

**Why it mattered:**
- First general-purpose agent mastering diverse tasks from raw perception.
- Proved deep learning + RL can work at scale.
- Led directly to AlphaGo (2016), OpenAI Five (2019), and all of modern deep RL.
- Published in Nature — one of the most cited AI papers of the decade.

---

## 13. DQN Variants

| Variant | Year | Key improvement |
|---------|------|----------------|
| **Double DQN** (van Hasselt) | 2016 | Use online network to SELECT action, target network to EVALUATE it — fixes Q-value overestimation from max operator |
| **Dueling DQN** (Wang) | 2016 | Split network into "how good is this state" + "how much does each action help" — faster learning when action choice barely matters |
| **Prioritized Replay** (Schaul) | 2016 | Sample transitions with high TD error (surprising, informative) more often — better use of buffer |
| **Rainbow** (Hessel) | 2017 | Combined 6 improvements → **230% of human median** on Atari. Shows improvements are complementary, not competing. |

---

## 14. Putting it together

```
THE JOURNEY FROM PLANNING TO LEARNING
══════════════════════════════════════

   Dynamic Programming          "I know the world perfectly"
   (policy/value iteration)     → compute answer exactly
           │
           │  Remove: complete model
           ▼
   Exploration vs. Exploitation "What should I try?"
   (ε-greedy, softmax, UCB)    → balance knowledge vs. discovery
           │
           │  Remove: states, transitions
           ▼
   Multi-Armed Bandits          "Simplest RL problem"
   (MAB, contextual bandits)    → pure exploration, most deployed in industry
           │
           │  Add: states, sequential decisions
           ▼
   TD Learning                  "Learn from every step"
   (TD(0), SARSA, Q-learning)   → no model, no waiting, bootstrap
           │
           │  Key distinction: on-policy vs. off-policy
           ▼
   Experience Replay            "Reuse old data"
   (only possible off-policy)   → data efficiency
           │
           │  Replace table with neural network
           ▼
   DQN                          "Superhuman Atari from pixels"
   (replay + target network)    → one algorithm, 49 games, raw pixels in
           │
           │  DQN limitation: discrete actions only
           ▼
   Part 3: Policy Gradients     "Continuous actions, RLHF, ..."
```

**Five things to remember:**
1. DP is planning with a full model. TD is learning from experience.
2. The TD error δ is the surprise signal that drives all TD-based learning.
3. Q-learning is off-policy → enables replay → enables DQN.
4. DQN = Q-network + experience replay + target network.
5. Result: one algorithm, raw pixels in, superhuman on 29 Atari games.

---

## Interview prep — Amazon Applied Scientist

### Core questions

<details><summary><b>1. What are the three limitations of dynamic programming and why do they matter?</b></summary>

(1) DP needs the complete model P(s'|s,a) — real agents rarely have this. (2) Curse of
dimensionality — DP sweeps over ALL states every iteration; chess has ~10⁴⁷ states, Go ~10¹⁷⁰,
Atari is effectively infinite. (3) Only works for discrete, finite state/action spaces — continuous
control (robot arm angles) has infinite states and is simply out of scope. Together, these
limitations motivate the entire rest of the course: model-free methods that learn from experience.
</details>

<details><summary><b>2. Explain the exploration-exploitation tradeoff with an Amazon example.</b></summary>

Exploit: show the customer the product we know they click on — maximizes immediate expected
revenue. Explore: test a new listing that might convert even better — gathers information but
costs revenue now if the new listing is worse. Pure exploitation gets stuck showing the same
products forever (local optimum). Pure exploration wastes revenue testing bad products. The
practical answer: explore more early (when you know nothing), exploit more later (once you
understand the landscape). Every recommendation engine and ad system on earth lives on this
tradeoff.
</details>

<details><summary><b>3. What is the TD(0) update rule and what do the TD target and TD error mean?</b></summary>

V(s) ← V(s) + α[R + γV(s') − V(s)]. The TD target is R + γV(s') — a fresh estimate of what
state s is really worth based on one step of experience. The TD error δ = R + γV(s') − V(s) is
the "surprise" — the gap between what we expected and what we experienced. The update nudges V(s)
by a fraction α toward the better target. Key: we update immediately after one step, using our
own estimate of the future — no model needed, no waiting for episode end.
</details>

<details><summary><b>4. What is the single-word difference between SARSA and Q-learning, and why does it change everything?</b></summary>

SARSA uses Q(s', a') — the value of the action we'll actually take next (following our ε-greedy
policy). Q-learning uses max_a' Q(s', a') — the value of the best possible next action. This
one-word change makes SARSA on-policy (learns about the strategy it's following, including
exploration costs) and Q-learning off-policy (learns the optimal policy regardless of behavior).
Off-policy enables experience replay, which enables DQN — so this single word is the foundation
of the entire deep RL revolution.
</details>

<details><summary><b>5. What is the deadly triad and why doesn't DQN collapse?</b></summary>

The deadly triad is the combination of (1) off-policy learning, (2) function approximation
(neural network), and (3) bootstrapping — which can cause Q-values to diverge to infinity. Any
two of three are fine; all three together is unstable. DQN uses all three but adds two tricks:
experience replay (decorrelates the training data, fixing the correlated-data problem) and a
target network (frozen copy updated every 10,000 steps, fixing the moving-target problem). These
two tricks stabilize what would otherwise be an unstable combination.
</details>

<details><summary><b>6. Describe the DQN network architecture and explain why each design choice was made.</b></summary>

Input: 84×84×4 (4 stacked grayscale frames). Why 4 frames: a single frame is ambiguous about
motion direction; 4 frames encode velocity. Why grayscale: color rarely matters for gameplay,
and smaller input trains faster. Three conv layers (32 filters 8×8 stride 4, 64 filters 4×4
stride 2, 64 filters 3×3 stride 1) with ReLU, then FC 512 with ReLU, then output Q(s,a) for
each action. Why convolutions: images have spatial structure; CNNs detect patterns regardless of
position. Why output all Q-values at once: one forward pass gives Q for every action — pick
argmax. Same architecture for all 49 games.
</details>

<details><summary><b>7. Why is experience replay only possible with off-policy methods?</b></summary>

Experience replay stores old (s, a, r, s') transitions and samples them randomly for training.
On-policy methods like SARSA need a' — the action the current policy would take in s'. But old
transitions have the a' from whatever policy was active when they were collected, which may be
different from the current policy. Using stale a' values would corrupt the learning target.
Q-learning is off-policy: its target max_a' Q(s', a') doesn't depend on what action was actually
taken, so old transitions are always valid training data.
</details>

<details><summary><b>8. Compare policy iteration and value iteration in terms of computational cost and when each is preferred.</b></summary>

Policy iteration: fewer outer loops (often converges in very few iterations), but each loop
requires a full policy evaluation — many sweeps over all states until V^π converges. Value
iteration: more outer loops, but each loop is just a single sweep (one pass over all states).
For small problems, policy iteration can be faster (fewer total sweeps). For large problems,
value iteration is often preferred because each sweep is cheap and you avoid the expensive inner
evaluation loop. Both require the full model P(s'|s,a).
</details>

<details><summary><b>9. What makes contextual bandits the "sweet spot" for industry applications?</b></summary>

Contextual bandits add user features (context) to plain multi-armed bandits, enabling
personalization — "which action is best for THIS user in THIS situation?" The key advantage over
full RL: rewards are immediate, so your action doesn't change the future state. No delayed
consequences, no long-horizon credit assignment, no need for temporal-difference learning. This
makes the problem tractable at scale (Amazon shows different recommendations to millions of users
based on context) while still being more useful than plain bandits (which give the same answer
to everyone). The trade-off: you miss sequential effects that full RL captures.
</details>

<details><summary><b>10. [Combines concepts] DQN achieves superhuman performance on 29 of 49 Atari games with the SAME hyperparameters. Why is this remarkable, and what limitation does it expose?</b></summary>

It's remarkable because it demonstrates genuine generalization — the same network architecture,
same learning algorithm, same hyperparameters work across 49 completely different games with no
game-specific engineering. The agent learns to play each game from raw pixels alone, discovering
strategies (like Breakout's tunnel trick) that no human programmed. But DQN has a hard
limitation: it requires DISCRETE actions because it takes argmax over a finite set. For continuous
actions (steering wheel angle, robot joint torque, investment allocation), you can't enumerate
all options — this limitation motivates Part 3's policy gradient methods.
</details>

### Depth probes

- *"The lecture says TD has lower variance than Monte Carlo but introduces bias from
  bootstrapping. Explain this tradeoff concretely."* — MC waits until the episode ends and uses
  the true return G_t, which is an unbiased estimate of V(s) but has high variance (different
  episodes from the same state can produce very different returns). TD uses R + γV(s'), which is
  a biased estimate (because V(s') is itself an estimate, not the true value) but has much lower
  variance (it only depends on one step of experience, not the entire rest of the episode). In
  practice, the variance reduction from TD usually outweighs the small bias, which is why TD is
  more widely used.

- *"Why does the target network get updated every 10,000 steps specifically? What would happen
  with a different number?"* — 10,000 is a heuristic that balances stability vs. adaptation
  speed. Too frequent (e.g., every step): the target moves too fast, causing oscillation (the
  original problem). Too infrequent (e.g., every 100,000 steps): the target is very stable but
  becomes outdated — the student network converges to a target that's now stale, slowing
  learning. 10,000 was empirically found to work well for Atari; the right number depends on the
  problem's complexity and learning rate.

- *"The lecture says Rainbow combines 6 improvements and reaches 230% of human median. What does
  'median' mean here and why is it a better metric than 'average'?"* — the median across 57
  Atari games means DQN performs better than half of human-level scores. Using median instead of
  mean is important because a few extremely high scores (e.g., 100× human on one game) would
  inflate the mean and mask poor performance on other games. Median is robust to outliers and
  gives a more honest picture of typical performance.

### Whiteboard-ready derivations

1. **The TD(0) update from the Bellman equation** — §6.2: start from V^π(s) = E_π[R + γV^π(s')],
   replace the expectation with a single sample (R + γV(s')), and use the learning rate α to
   take a small step toward this sample: V(s) ← V(s) + α[R + γV(s') − V(s)].

2. **SARSA vs. Q-learning update** — §7: write both updates side by side, highlight the single
   difference (Q(s',a') vs. max_a' Q(s',a')), and explain why this makes one on-policy and the
   other off-policy.

3. **DQN loss function** — §11.2: write L = (R + γ max_a' Q(s',a'; θ⁻) − Q(s,a; θ))²,
   explain each term, and show why the frozen θ⁻ makes the target stable for 10,000 steps.

### Applied scenario — Amazon product recommendation

**Framing:** Amazon wants to recommend products to users, balancing showing known popular items
(exploit) with testing new recommendations (explore).

**Contextual bandit approach (§5.2):** observe user context (browsing history, time of day,
location, device), choose from K recommendation widgets, receive immediate reward (click,
purchase). No delayed consequences — the reward is immediate. This is the industry-preferred
setting because it's tractable at scale.

**Full RL approach (§6–§11):** model the sequential nature of browsing — each recommendation
changes what the user sees next, which affects future purchases. Use Q-learning with experience
replay to learn which recommendation sequences maximize long-term customer value.

**DQN-style approach:** replace the Q-table with a neural network that takes user features +
current page state as input and outputs Q-values for each possible recommendation. Use
experience replay to learn from millions of past browsing sessions. The target network stabilizes
training.

**Failure modes:**
- **Cold start:** new users/items have no history → high uncertainty → need exploration (ε-greedy
  or UCB).
- **Reward misspecification:** optimizing for clicks may not maximize long-term customer value
  (clickbait problem).
- **Deadly triad:** off-policy + function approximation + bootstrapping can diverge → need both
  tricks (replay + target network).

**Leadership Principle tie-in:** **Customer Obsession** — the exploration-exploitation tradeoff
is fundamentally about whether to show customers what we *know* they like or discover what they
*would* like even more. **Invent and Simplify** — contextual bandits simplify the full RL
problem to its most tractable form while still enabling personalization at scale.

---

## Glossary

- **Behavior policy b** — the policy that generates the data (what the agent actually does).
- **Bootstrapping** — updating an estimate using another estimate (not the true final outcome).
- **Contextual bandit** — a bandit where the action choice depends on observed context features.
- **Curse of dimensionality** — the exponential growth in state space size as dimensions increase,
  making tabular methods infeasible.
- **Deadly triad** — the unstable combination of off-policy + function approximation +
  bootstrapping.
- **Double DQN** — uses online network to select actions, target network to evaluate them,
  reducing Q-value overestimation.
- **Dueling DQN** — splits the network into state-value and advantage streams for faster learning.
- **Dynamic programming (RL)** — methods that compute optimal policies given a complete MDP.
- **ε-greedy** — exploration strategy: with probability ε take a random action, otherwise greedy.
- **Experience replay** — storing and randomly sampling past transitions for training; enables
  data efficiency.
- **Multi-armed bandit (MAB)** — the simplest RL problem: K arms, no states, pure exploration.
- **Off-policy** — learning the target policy from data generated by a different behavior policy.
- **On-policy** — learning about the same policy that generates the data.
- **Prioritized experience replay** — sampling transitions with high TD error more often.
- **Q-learning** — off-policy TD control using max_a' Q(s',a') as the target.
- **Rainbow** — combined 6 DQN improvements, reaching 230% of human median on Atari.
- **Regret** — the difference between optimal reward and actual reward; lower = better.
- **SARSA** — on-policy TD control: (State, Action, Reward, next State, next Action).
- **Softmax (Boltzmann)** — exploration strategy: action probabilities proportional to e^{Q/τ}.
- **Target network** — frozen copy of the DQN network, updated every C steps for stability.
- **TD error δ** — R + γV(s') − V(s), the "surprise" driving learning.
- **TD target** — R + γV(s'), the updated estimate of V(s).
- **Temporal difference (TD) learning** — model-free learning that updates after every step using
  bootstrapping.
- **Thompson Sampling** — Bayesian exploration strategy: sample a plausible payout from each arm's
  belief distribution, pull the arm with the highest sample; self-calibrating, often best in practice.
- **UCB (Upper Confidence Bound)** — exploration strategy: pick action maximizing Q(a) + uncertainty
  bonus; self-adapting with optimal regret bounds.
- **Value iteration** — combines policy evaluation and improvement in one sweep using max instead
  of policy-weighted average.

---

## Check yourself

1. Name the three limitations of dynamic programming and explain why each prevents DP from solving
   real-world RL problems. *(§2)*
2. Solve a 2-state MDP using value iteration: S={s1,s2}, A={stay,go}, γ=0.9, rewards as in Part
   1. Show 3 iterations. *(§3)*
3. Explain the exploration-exploitation tradeoff using a concrete Amazon recommendation example.
   *(§4)*
4. Compare ε-greedy, softmax, and UCB strategies. When would you choose each? *(§4)*
5. Why are multi-armed bandits the "simplest RL problem"? List the three things they strip away.
   *(§5)*
6. What is the key difference between contextual bandits and full RL? Why does industry prefer
   contextual bandits? *(§5)*
7. Write the TD(0) update rule and define every symbol. What is the TD target? What is the TD
   error? *(§6)*
8. Explain why TD is "the best of both worlds" compared to DP and Monte Carlo. *(§6)*
9. What is the single-word difference between SARSA and Q-learning? Explain why this makes one
   on-policy and the other off-policy. *(§7)*
10. Explain the cliff-world example: why does SARSA learn a safe path while Q-learning learns the
    optimal path? *(§7)*
11. What is experience replay and why is it only possible with off-policy methods? *(§9)*
12. State the deadly triad. Why does DQN use all three yet remain stable? *(§9)*
13. Describe the DQN network architecture and explain why 4 frames are stacked. *(§10)*
14. Name the two DQN tricks and explain which problem each solves. *(§11)*
15. What are Double DQN, Dueling DQN, Prioritized Replay, and Rainbow? One sentence each. *(§13)*

---

## Going deeper

1. **Sutton & Barto (2018), "Reinforcement Learning: An Introduction"** — ⚠️ **named on the
   slides** as the primary textbook. `solid`. Chapters 4–8 cover everything in this lecture.

2. **David Silver's RL course (UCL/DeepMind)** — ⚠️ **named on the slides**. `solid`. Lectures
   3–5 cover DP, MC, and TD in more depth.

3. **Mnih et al. (2015), "Human-level control through deep reinforcement learning" (DQN)**
   — ⚠️ **named on the slides** as the DeepMind Nature paper. `hard`. The foundational DQN paper.

4. **Watkins (1989), "Learning from Delayed Rewards" (Q-learning)** — ⚠️ **named on the slides**
   as "one of the most influential RL papers ever written." `hard`. The original Q-learning paper.

5. **OpenAI Spinning Up (spinningup.openai.com)** — ⚠️ **named on the slides** in Part 1 as
   recommended resource. `solid`. Practical implementations of all algorithms covered here.

6. **Van Hasselt et al. (2016), "Deep Reinforcement Learning with Double Q-learning"** — ⚠️ not
   named on slides by title but referenced as the Double DQN paper. `hard`.

> ⚠️ **verify this** — items 1–4 are explicitly named on the lecture's slides. Items 5–6 are
> referenced implicitly — treat as suggested background reading.
>
> ✅ **Citation check (enhancement pass).** Independently verified against primary sources: Mnih et
> al. (2015) is *Nature* **518**, 529–533 (DOI 10.1038/nature14236); Watkins (1989), "Learning from
> Delayed Rewards," is his PhD thesis, King's College, Cambridge (May 1989); van Hasselt, Guez &
> Silver (2016), "Deep Reinforcement Learning with Double Q-learning," is AAAI 2016, pp. 2094–2100.
> Also independently checked (cited in-body rather than in this list): Wang et al. (2016), "Dueling
> Network Architectures for Deep Reinforcement Learning," ICML 2016; Schaul et al. (2016),
> "Prioritized Experience Replay," ICLR 2016 (arXiv Nov 2015); Hessel et al., "Rainbow: Combining
> Improvements in Deep Reinforcement Learning" — arXiv-dated 2017 (as this file cites it) but the
> AAAI-2018 proceedings version is the same paper, so both years are seen in the literature; no
> correction needed. All titles, author lists, and years confirmed exact — no corrections to any
> citation in this file.
