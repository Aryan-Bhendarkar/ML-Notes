---
title: "Agentic AI — Part 1: Fundamentals, Architectures, and Reasoning"
topic: agentic-ai
lecture: 27
source: "output/Lecture_27 - Bonus Module Agentic AI Part 1"
slides: 23
---

# Agentic AI — Part 1: Fundamentals, Architectures, and Reasoning

> Runtime ~32:00. Originally drafted from `slides_deduped/` (23 slides) with OCR + transcript
> verification, then **audited against the raw 53-frame capture** in
> `output/Lecture_27 - Bonus Module Agentic AI Part 1/` per this project's standard methodology
> (see project memory `slides-deduped-is-lossy`) — a whole missing architecture (Plan-and-Execute,
> §5.2a) was found and added. Instructor: **Harsh Agarwal**, Applied Scientist, International ML
> team at Amazon — no name overlay was visible in this deck's own frames, but Lecture 28 (same
> module, consecutive part) has a confirmed nameplate reading "Harsh Agarwal," and both lectures
> are almost certainly delivered by the same instructor. This module covers how agents work on top
> of LLM architectures — the loop that turns a language model from a one-shot answerer into a
> system that perceives, reasons, and acts.

---

## What you'll understand after reading this

1. **Distinguish chatbot, copilot, and agent** by who controls the next step in the loop.
2. **Describe the agent loop** (think → act → observe) and explain why closing this loop is what
   makes an agentic system.
3. **Explain the autonomy ladder** — L1 tool call → L2 multi-step (ReAct) → L3 plan & adapt → L4
   multi-agent — and identify where the "do I need an agent?" threshold actually sits.
4. **Calculate how errors compound** in multi-step agents (the 0.95^n problem) and explain why
   a "95% reliable step" does NOT mean a "95% reliable agent."
5. **Decide when to use an agent** vs. a simple LLM call vs. a fixed workflow, based on path
   complexity and action reversibility.
6. **Implement the ReAct framework** (Reasoning + Acting) and explain why combining thought,
   action, and observation prevents hallucination.
7. **Explain the Reflection framework** — how agents learn from mistakes without retraining the
   LLM, via memory banks.
8. **Implement Tree of Thought** — exploring multiple reasoning branches with backtracking —
   and explain when it outperforms single-chain reasoning.
9. **Design guardrails** — thinking vs. acting decision, confidence thresholds for human
   interruption, and validate-retry-escalate patterns.

---

## Before we start: what you need to know

### 📚 Prerequisite 1 — What an LLM call actually is

An **LLM (large language model)** takes a sequence of text (a "prompt") and produces a probability
distribution over what text should come next, then samples from it — one token at a time. A single
"call" to an LLM is stateless: it has no memory of anything beyond what's in the prompt you send it
this time. Everything this lecture builds — tools, loops, memory — exists because a raw LLM call,
by itself, can only do this one thing: text in, text out.

### 📚 Prerequisite 2 — Tokens and context window, briefly

A **token** is roughly a word-piece — the unit an LLM actually processes (see Lecture 18's
tokenization material for the full mechanics). The **context window** is the maximum number of
tokens a single LLM call can take as input. Every idea in this lecture about "injecting a memory"
or "the schema is the prompt" ultimately means: some text gets added to what fits inside this
window before the next call.

### 📚 Prerequisite 3 — What "calling a function" means, for a non-programmer reader

When this lecture says an agent "calls a tool" or "calls `search_database()`," it means: the LLM
outputs a piece of structured text specifying a function name and arguments (e.g. `search("green
parks Mumbai")`), and separate code — not the LLM itself — actually runs that function and returns
a result. The LLM never executes anything; it only ever *requests* that something be executed. This
distinction becomes important in Lecture 28's tool-calling material and is assumed throughout this
lecture's "Act" step.

---

## The big picture

LLMs are one-shot: you ask, they answer, they stop. **Agents close the loop** — the LLM
decides what to do next based on what it just observed:

```
Standard LLM:     Input → LLM → Output (done)

Agent:             Input → LLM → Think → Act (tool call) → Observe → LLM → Think → Act → ... → Done
```

This module covers:
1. What makes something an agent (vs. chatbot vs. copilot)
2. How agents think (ReAct, Reflection, Tree of Thought)
3. When agents fail (error compounding, infinite loops)
4. How to guardrail them (confidence thresholds, human gates)

---

## 1. What Is an Agent?

### 1.1 The chatbot → copilot → agent spectrum

| | **Chatbot** | **Copilot** | **Agent** |
|---|---|---|---|
| Who drives? | You drive every turn | Model suggests, you approve | Model owns the loop |
| Interaction | Q&A, open-ended text | Autocomplete, suggestions | Autonomous task execution |
| Example | FAQ chatbot | Copilot autocomplete | **ReAct web researcher**, **SWE-bench coder**, **Multi-agent orchestrator** |
| Loop control | Human closes the loop | Human approves each step | **Agent decides when to stop** |

> *"A chatbot answers questions. A coding agent runs the test, reads the failure, then edits and
> re-runs on its own — nobody approves each step."*

> ⚠️ **A RAG bot with tools is still a chatbot** [slide 12] — retrieving documents and answering
> is still "human drives each turn": the human asks a new question every turn, the bot never
> decides its own next step. A copilot and a true agent differ only on *who owns the loop*, not on
> whether tools are involved at all.

### 1.2 The agent loop: think → act → observe

Every agentic system follows this three-phase cycle:

1. **Think (Reason):** The LLM analyzes the task, the available tools, and what it knows so far.
   It creates a plan or chain of thought.
2. **Act (Execute):** The LLM calls a tool (search, database lookup, code execution, API call)
   with specific parameters.
3. **Observe (Feedback):** The tool returns results. The LLM reads them, decides if the task is
   complete, and either stops or loops back to Think.

**The key insight:** the LLM needs access to *tools* — specialized functions that do one thing
well (search, read file, run code, send email). The agent's prompt includes descriptions of all
available tools, and the LLM decides which to call and with what parameters.

---

## 2. The Autonomy Ladder [slide 13–14]

**How much of the control flow you hand to the model, one rung at a time.** The deck's own ladder
is keyed to a single worked task, so you can see exactly where each rung succeeds or fails rather
than reasoning about autonomy in the abstract:

> **THE TASK** — *"I can come Monday but my passport delay may push me to Wednesday. Can I still
> surf Tuesday with cancellation insurance?"*

| Level | What runs | Control-flow code | On this task |
|-------|-----------|--------------------|---------------|
| **L1 — Tool call** | A single tool call, no loop | `run_function(llm_tool, llm_args)` | ❌ **Partial / wrong answer** — one lookup can't reconcile "Monday," "Wednesday," and "Tuesday" against an insurance policy in one shot |
| **L2 — Multi-step (ReAct)** | A self-directed loop — the model decides when to stop | `while llm_should_continue(): step()` | ✅ **Answered** — this is the deck's own **"do I need an agent?" threshold**: below this line, hardcode it; at or above, you need an agent |
| **L3 — Plan & adapt** | Autonomous goal pursuit | `plan → execute → replan` | Handles multi-step plans that must revise themselves mid-request |
| **L4 — Multi-agent** | Delegation / collaboration | `if llm_trigger(): run_agent()` | One agent spawns others for sub-tasks |

**The principle:** *"A plan that changes mid-request can't be answered by one fixed lookup. Give
the model more say over what runs next: a tool call becomes a loop. Agency is a dial, not a
switch: more autonomy, harder to control."* L2 (ReAct) is the practical floor for most real
agentic tasks — it's the first rung on this ladder that actually *answers* the worked example
above, not just partially addresses it.

> 📚 **Note on the earlier chatbot/copilot/agent framing (§1.1)** — that spectrum and this ladder
> describe the same underlying idea (how much control the model has) from two different angles:
> §1.1 is about who *drives the conversation*, this ladder is about what *code structure* each
> level of control actually compiles down to (a single call vs. a loop vs. a replanning loop vs.
> agent-spawns-agent).

> ⚠️ **Agents fail fast and silently.** If you give an autonomous agent a complex task without
> guardrails, a single error at step 3 poisons every subsequent step. The overall success rate
> drops exponentially with the number of steps — this is the most critical thing to understand
> about agentic systems.

---

## 3. Why Agents Are Hard: Error Compounding

### 3.1 The 0.95^n problem [slide 16–18]

The formula says: **if every step succeeds independently with the same probability, the chance
all of them succeed together shrinks fast — multiplying probabilities less than 1 always makes the
result smaller, and doing it n times compounds that shrinkage.**

$$P(\text{all } n \text{ steps succeed}) = p^n$$

| Symbol | Read it as | What it means |
|---|---|---|
| $p$ | "per-step success probability" | The chance a single step of the agent succeeds (e.g. 0.95 for "95% reliable") |
| $n$ | "number of steps" | How many sequential steps the agent's task requires |

If each step has 95% success rate, the end-to-end success rate over n steps is $0.95^n$:

| Steps (n) | End-to-end success rate |
|-----------|------------------------|
| 1 | 95% |
| 5 | 77% |
| 10 | 60% |
| 20 | 36% |
| 40 | **12.9%** |
| 100 | **0.6%** |

*"A 95% reliable step does NOT make a 95% reliable agent."*

```interactive
type: slider
title: Error compounding — 0.95^n
concept: Why per-step reliability doesn't translate into agent reliability
control: Drag n (number of steps) from 1 to 100
observe: End-to-end success rate falls off a cliff well before n=100
insight: A step that "almost always works" still makes a long agent almost always fail — the fix is guardrails, not a marginally better model
fallback: The static table above (n=1→95%, n=40→12.9%, n=100→0.6%) already shows the full curve; read down the rows in order to see the same shape a slider would animate.
```

### 3.2 Common failure modes

| Failure mode | What happens | Example |
|-------------|-------------|---------|
| **Infinite loop** | Agent repeats the same validation cycle forever, burning compute | Agent can't satisfy its own acceptance criteria |
| **Hallucinated action** | Agent assumes a tool exists that doesn't, then hallucinates its output | Agent calls `search_database()` but the function doesn't exist |
| **Poisoned pipeline** | One wrong step makes all subsequent steps invalid | Agent writes wrong function signature, all callers break |
| **Cost explosion** | Each retry adds API costs; errors compound into massive bills | Agent retries 50 times on an impossible task |

> 💡 **Key insight — error compounding is why guardrails matter more than raw capability.** An
> agent that's brilliant at each individual step but has no way to detect or recover from errors
> will fail catastrophically on multi-step tasks. The solution isn't a smarter LLM — it's
> unit tests, confidence thresholds, and human gates at critical points.

---

## 4. When to Use an Agent

| Task type | Best approach | Why |
|-----------|--------------|-----|
| Simple lookup (search Google) | Direct LLM call or manual | No need for the agent overhead |
| Fixed workflow (always 3 steps) | Hardcoded pipeline | Cheaper, deterministic, debuggable |
| **Dynamic, unknown path** | **Agent** | Path can't be scripted in advance |
| Irreversible action (drop DB, cancel flight) | Agent + **human gate** | Cost of failure too high for full automation |

**The rule of thumb:** reach for an agent *only* when the path is unknown AND the task has
multiple moving parts that benefit from automation. If you can predict every step in advance,
hardcode it — it's cheaper and more reliable.

---

## 5. Agentic Reasoning Frameworks

### 5.1 ReAct (Reasoning + Acting)

The most fundamental agentic framework. Every step follows the cycle:

```
Thought: "I need to find green parks in Mumbai. Let me search for that."
Action: search("green parks Mumbai")
Observation: [3 results: Park A, Park B, Park C]
Thought: "Park B is actually a garden, not a park. Let me verify the others."
Action: lookup("Park A details")
Observation: [3.2 acres, public, open daily]
Thought: "Park A matches. Let me check Park C too."
...
Thought: "I have enough information. Stopping here."
```

**Why it works:**
- **Thought** prevents hallucination by grounding every action in reasoning.
- **Action** prevents the LLM from making things up (it must call a real tool).
- **Observation** feeds real-world data back, closing the loop.

**Without observation:** the LLM hallucinates tool outputs.
**Without thought:** the LLM calls tools randomly without reasoning about results.

> ⚠️ **ReAct isn't an unqualified win** [slide 26] — on the HotpotQA benchmark, ReAct still trails
> plain Chain-of-Thought reasoning (**27.4 vs. 29.4 exact-match**). Grounding every step in a real
> tool call trades one failure mode for another: it eliminates *hallucination* but introduces
> *retrieval errors* (the tool call itself can return the wrong or an unhelpful result). ReAct wins
> when grounding matters more than raw single-shot reasoning accuracy — it is not strictly better
> in every setting.

### 5.2 Reflection — Learning from Mistakes Without Retraining

The agent learns from its own failures by:

1. **Retrospective analysis:** After a task fails, the LLM examines the entire chain of events
   and identifies where/why it went wrong.
2. **Memory bank:** Store the lesson (e.g., "don't use multiplication when not allowed") in a
   persistent memory store.
3. **Context injection:** On the next attempt, inject the memory into the prompt so the LLM
   knows to avoid the same mistake.

**Key point:** the LLM's weights are never updated. Learning happens entirely through context
engineering — the memory bank is just text appended to future prompts.

**Result:** success rate improves dramatically with each trial, without any retraining. This is
a form of RL operating at the prompt level.

🧪 **The number behind "dramatically"** [slide 28]: with reflection **on**, the agent climbs from
~22% of tasks solved at trial 0 to **130/134 solved (≈97%) by trial 12** — a smooth, steady climb.
With reflection **off** (retry only, no lesson stored), the curve **stalls near ~55–58% by trial
7** and plateaus there — the agent keeps retrying but keeps making the same mistake, so extra
attempts stop buying any improvement. The gap between these two curves *is* what "learning without
retraining" is worth in practice.

### 5.2a Plan-and-Execute

The lecture's own "four ways to build an agent" slide names ReAct, Reflection, Plan-and-Execute,
and search-over-thoughts (§5.3) as the four architectures — this is the third.

> **Plan-and-Execute** — a **strong-model Planner** drafts the entire plan in a single call, a
> **cheap-model Executor** carries out each step with tools, and a **Replan** edge repairs the plan
> when a step fails, looping back to the planner.

```
PLANNER (strong model, 1 call) → EXECUTOR (cheap model + tools) → REPLAN (on a failed step)
        ▲                                                              │
        └──────────────────────────────────────────────────────────────┘
```

**Why split planner and executor at all?** A single expensive model call drafts the whole plan
once; a cheap model then just *executes* each already-decided step, which is a much easier job than
planning. This is a cost optimization on top of a capability one — most of the work (running tools,
checking outputs) doesn't need the strong model at all.

🧪 **The benchmark that makes the risk concrete — a 3-block Blocksworld stacking problem, 600
instances, zero-shot, plans checked by the VAL verifier:**

| Planner | Success rate |
|---|---|
| **Fast Downward** (classical, symbolic planner) | **100.0%** |
| GPT-4 | 34.6% |
| Claude 3.5 Sonnet | 54.8% |
| o1-preview | 97.8% |

The block-stacking problem itself never changes — only the **names** of the blocks and actions do
(familiar block-stacking words like `pickup`, `stack(A,B)`, `on(C,A)` vs. obfuscated equivalents).
**Renaming the actions leaves the problem mathematically identical, yet the LLM bars collapse toward
~0% under obfuscation while the classical planner still holds 100%.**

> ⚠️ **"The LLM is a good plan *proposer* but a poor plan *guarantor*."** A plan that *reads* well
> — plausible-sounding steps, familiar terminology — can still be logically infeasible, and an LLM
> planner has no built-in way to verify its own plan's validity the way a symbolic planner does.
> This is exactly why the **Replan** edge exists: don't trust the plan blindly, check each step's
> actual outcome and loop back when it fails, rather than assuming a good-looking plan is a correct
> one.

### 5.3 Tree of Thought — Exploring Multiple Branches [slide 38]

Instead of one chain of thought, explore *multiple* reasoning branches, score each one, and
backtrack from dead ends — this is the deck's own "Game of 24" example (reach 24 using the four
numbers `4 4 6 8` exactly once each), redrawn here to match the slide's actual tree:

```
                              [Root: 4 4 6 8]
                    ┌──────────────┼──────────────┐
              6-4=2 (maybe)   4+8=12 (sure)   4×6=24 ✗ (impossible —
                    │               │          uses only 2 of 4 tiles,
              2×8=? ✗ (dead end,    │           strands the other two)
               backtrack)           │
                                6-4=2 (sure)
                                     │
                                  =24 ✓ (reached — correct)
```

**Reading the tree:** the root generates three candidate first moves. `4×6=24` is scored
**impossible** and pruned immediately — it "solves" the arithmetic but leaves two tiles (`4`, `8`)
unused, so it cannot be a valid path to a Game-of-24 answer that must use all four numbers.
`6-4=2` looks promising ("maybe") but its own follow-up `2×8` dead-ends, forcing a **backtrack**.
The surviving branch, `4+8=12`, evaluates its own next step `6-4=2` and reaches `=24` — a
confirmed, correct solution.

**When to use:** tasks with multiple valid paths where the answer is not deterministic
(puzzles, planning, creative problem-solving).

**Advantage over best-of-N chain-of-thought:** Tree of Thought with backtracking explores more
efficiently because it prunes invalid branches early rather than sampling complete chains and
keeping the best.

🧪 **The comparison that makes this concrete** — at the *same* ~6,000-token budget per problem:

| Method | Success rate |
|---|---|
| Single chain of thought | **4%** |
| CoT best-of-100 (i.i.d. sampling, pick best) | **49%** |
| ToT, branching factor b=5 (branch · score · backtrack) | **74%** |

Structured search gains **25 points over best-of-100 sampling at the same token budget** — the
budget buys much more when it's spent exploring and pruning a tree than when it's spent generating
100 independent guesses and hoping one is right.

**Tradeoff:** higher token cost per problem than a single chain. The gap between Tree of Thought
and best-of-N decreases as token budget increases (given enough budget, sampling enough independent
chains eventually catches up).

**LATS (Language Agent Tree Search)** — the lecture's own slide presents this alongside Tree of
Thought as "ToT / LATS." Where plain ToT explores and prunes branches by having the LLM directly
judge which look promising, LATS layers in a more structured **tree search** — expanding, scoring,
and backtracking from the most promising nodes more systematically, closer to how Monte Carlo Tree
Search picks which branch to explore next.

---

## 6. Agent Guardrails

### 6.1 Think vs. Act decision

| Action type | Cost of mistake | Strategy |
|-------------|----------------|----------|
| **Reversible** (read file, search) | Low | Execute more, think less |
| **Moderate cost** (send email) | Medium | Ask human if confidence < threshold |
| **Irreversible** (drop DB, cancel flight) | Very high | Think extensively, require human gate |

### 6.2 Confidence-based human interruption [slide 46–48]

Set a confidence threshold τ for when the agent should ask for human approval instead of acting —
the interactive demo lets you drag τ and watch the trade-off. *"Unsure: ship a wrong action, or
pause and ask? A wrong write costs far more. So gate on confidence."*

> ⚠️ **This only works if the confidence number is honest** [slide 46–48] — the deck makes this a
> precondition, not a footnote: **raw RLHF log-probability confidence is over-confident**, but
> *asking the model to verbalize* its own confidence in words (rather than reading off the model's
> internal probability) **cuts Expected Calibration Error (ECE) by a measured 50%**. Gating on a
> badly-calibrated confidence number is worse than not gating at all — you'd interrupt on the wrong
> cases and act confidently on the wrong ones too. Everything below assumes the "verbalized"
> confidence source, not raw log-probs.

| Threshold | Human interruptions | Wrong actions shipped | Total cost |
|-----------|-------------------|---------------|------------|
| τ = 0.72 | **9** | **0** | **9** |
| τ\* = 0.43 (**cost-minimizing threshold**) | **6** | 0 | **6 (the minimum, over this run's data)** |

> ⚠️ **Both rows are verified precisely against the demo** [slide 48] — τ\*=0.43's 6 interrupts / 0
> wrong / total cost 6, and τ=0.72's 9 interrupts / 0 wrong / total cost 9, are both read directly
> off the same interactive chart. Values at other thresholds (e.g. τ=1.0 or τ=0.2) would need to be
> read from the live demo directly — don't treat any specific interrupt/wrong-action count at an
> untested τ as precise.

```interactive
type: slider
title: Asking versus proceeding — the confidence threshold τ
concept: Gating actions on (calibrated) confidence trades human interruptions against wrong actions shipped
control: Drag the threshold τ from 0.00 to 1.00
observe: Below τ, low-confidence actions route to a human ("ask"); above τ, the agent acts on its own ("act") — the count of dots on each side of the line changes as τ moves
insight: There's a single τ* that minimizes total cost (interruptions + weighted wrong actions) for a given cost trade-off — moving τ away from it in either direction makes things worse, either by asking too often or by shipping too many wrong actions
fallback: The table above already gives the two verified reference points — τ=0.72 (9 interrupts, cost 9) and τ*=0.43 (6 interrupts, cost 6, the minimum) — read them as the two ends of what a slider would show moving.
```

**The sweet spot:** ask the human only when confidence is below a calibrated threshold. Too
high (ask too often, e.g. τ=0.72's 9 interrupts) → you interrupt constantly for little benefit,
since wrong actions were already at 0. Too low → the agent starts shipping wrong, uncorrected
actions, and a single wrong action typically costs far more than one human interruption would have.
τ\*=0.43 is the point on *this run's* data where total cost (interruptions + weighted wrong actions)
is minimized — it's a property of the specific cost trade-off and data, not a universal constant.

### 6.3 Validate → Retry → Escalate [slide 50–51]

*"Sampling the answer many times is easy; selecting the right one is the hard part."*

1. **Validate:** Run unit tests or sanity checks on each action's output.
2. **Retry:** If validation fails, try a different approach (up to a limit).
3. **Escalate:** If retries are exhausted, hand off to the human with context about what was
   tried and why it failed.

Without this pattern, agents enter infinite loops — they keep retrying forever because there's
no signal that says "stop and ask for help."

🧪 **Why "retry" alone isn't enough — the SWE-bench-Lite worked example** [slide 50]: retrying and
sampling k attempts on the same bug makes the chance that *a* correct fix is somewhere in the batch
climb steeply — coverage rises well past **55%** as k grows to 250. But if you then **vote** to pick
the winner with no independent checker, the selected answer barely moves off **~38–40%**, staying
close to the single-attempt SOTA of **43%** even at k=250. The gap between "found in the batch" and
"correctly selected" is the whole lesson: *"a vote barely moves — it found the answer, yet can't
tell which."* A **unit test acts as an independent verifier** that actually *selects* the correct
fix out of the batch, which is exactly what closes that gap; a vote can't, because the model
grading its own attempts is not independent of the attempts themselves. **The rule this implies:**
retry has a budget — when it's exhausted, escalate to a human rather than trusting a vote to have
silently picked the right answer.

---

## 7. Putting it together

> 📚 **A fourth pillar, named on the deck's own closing slide but not yet taught as its own idea:
> memory.** The closing diagram groups this module's ideas into four pillars — **Architecture**
> (§5's ReAct/Reflection/Plan-and-Execute/ToT), **Decision policy** (§4/§6's when-to-use-an-agent
> and think-vs-act rules), **Guardrails** (§6's confidence thresholds and validate-retry-escalate),
> and **Memory**. This file has so far only ever discussed memory narrowly, inside Reflection's
> "memory bank" (§5.2) — as a standalone concept in its own right (working memory vs. short-term vs.
> long-term vs. episodic memory, and how each is implemented), memory gets its full dedicated
> treatment in **Part 2 (`agentic-ai-02.md`, §7–§10)**. Flagging it here so the closing diagram's
> own vocabulary is grounded somewhere in this file, even though the deep-dive lives in the next
> lecture.

```
AGENTIC AI FUNDAMENTALS
════════════════════════

   LLM (one-shot) → Agent (closed loop)
        │
        ├── Chatbot: human drives every turn
        ├── Copilot: model suggests, human approves
        └── Agent: model owns the loop
             │
             ├── ReAct: Think → Act → Observe
             │   (prevents hallucination + random tool calls)
             │
             ├── Reflection: learn from mistakes via memory
             │   (no retraining, just context engineering)
             │
             └── Tree of Thought: explore multiple branches
                 (backtracking > best-of-N)
                      │
                      ▼
              Guardrails are essential:
              • Error compounding: 0.95^n → 0.6% at n=100
              • Confidence thresholds for human gates
              • Validate → Retry → Escalate
              • Irreversible actions ALWAYS need human approval
```

---

## Interview prep — Amazon Applied Scientist

<details><summary><b>1. What distinguishes an agent from a chatbot and a copilot?</b></summary>

A chatbot is driven by the human — every turn requires human input. A copilot suggests changes
but the human approves each one before it's applied. An agent owns the entire loop — it decides
what to do next, executes tool calls, observes results, and determines when the task is complete
without human intervention at each step. The key distinction is who controls the next step.
</details>

<details><summary><b>2. Why does error compounding make long-horizon agents unreliable, and what's the mathematical relationship?</b></summary>

If each step has success probability p, the end-to-end success over n steps is p^n. At p=0.95
and n=40, success drops to ~12.9%. At n=100, it's 0.6%. This exponential decay means that even
individually reliable steps produce unreliable agents over long horizons. The solution isn't
making each step slightly better — it's adding guardrails (unit tests, confidence gates, human
escalation) that catch and correct errors before they compound.
</details>

<details><summary><b>3. Explain the ReAct framework and why all three components (thought, action, observation) are necessary.</b></summary>

ReAct cycles through Thought → Action → Observation. Thought prevents random tool calls by
grounding each action in reasoning. Action prevents hallucination by requiring the LLM to call
a real tool instead of making things up. Observation feeds real-world results back, preventing
the LLM from operating on stale or incorrect information. Remove thought → random tool calls.
Remove action → hallucinated outputs. Remove observation → the LLM never sees real results and
can't correct course.
</details>

<details><summary><b>4. How does the Reflection framework enable learning without retraining?</b></summary>

After a task fails, the LLM performs retrospective analysis to identify where/why it went wrong.
This analysis is stored in a memory bank (just text). On subsequent attempts, the memory is
injected into the prompt, giving the LLM explicit knowledge of past failures. The model weights
never change — all learning happens through context engineering. This is essentially a form of
in-context learning / prompt-level RL.
</details>

<details><summary><b>5. When should you use Tree of Thought instead of a single chain of thought?</b></summary>

Tree of Thought is better when: (1) the answer is not deterministic — multiple valid paths exist;
(2) early steps can be validated to prune bad branches before committing; (3) backtracking is
possible (you can undo a wrong step). It outperforms best-of-N chain-of-thought because it
prunes invalid branches early rather than sampling complete chains. The tradeoff is higher token
cost, though the gap narrows with larger budgets.
</details>

<details><summary><b>6. Design the confidence threshold for a customer-service agent that can issue refunds.</b></summary>

Issuing refunds is semi-reversible (can be corrected but costs trust/money). Set a confidence
threshold based on the cost asymmetry: asking the human for approval costs ~1 unit (time),
while a wrong refund costs ~10 units (money + trust). The sweet spot is where total cost
(minimize human interruptions + minimize wrong actions × 10) is minimized — typically around
0.4-0.5 threshold. Below this: ask human. Above: execute automatically. Always escalate for
refunds above a dollar threshold regardless of confidence.
</details>

### Depth probes

- *"§5.2a shows a classical planner beating every LLM on a plan-validity benchmark. Does this mean
  LLMs should never be used for planning?"* — no; the benchmark isolates plan *feasibility*
  specifically, which classical planners are built to guarantee. LLMs remain far better at the step
  the classical planner can't do at all: proposing a plausible plan from an ambiguous, natural-
  language goal in an environment with no formal symbolic model. The practical answer the lecture
  itself gives is Plan-and-Execute's Replan loop — use the LLM to propose, then verify/repair rather
  than trusting the proposal outright.
- *"Error compounding (§3.1) assumes each step's failure is independent. When would that assumption
  break down, and which direction would it bias the 0.95ⁿ estimate?"* — if failures are *correlated*
  (e.g. one bad early tool-call result systematically confuses every later step, rather than each
  step failing independently), the true end-to-end success rate is typically *worse* than the
  independent-steps estimate suggests, since a single upstream mistake can poison many downstream
  steps at once rather than contributing one independent 5% failure chance.

### Whiteboard-ready derivation

**The 0.95ⁿ error-compounding curve** — §3.1: if each of $n$ steps succeeds independently with
probability $p$, the probability all $n$ succeed is $p^n$ (independent events multiply). At $p=0.95$:
$n=10 \to 0.95^{10}\approx0.60$; $n=40\to0.95^{40}\approx0.129$; $n=100\to0.95^{100}\approx0.006$ —
reproduce these by hand and confirm against §3.1's table.

### Applied scenario — Amazon returns-processing agent

**Framing:** Amazon wants an agent that reads a customer's return request, checks it against policy,
and either approves a refund automatically or routes it to a human.

**Why this needs an agent, not a fixed workflow (§4):** the *path* is genuinely unpredictable — a
return might need a policy lookup, an order-history check, an image-quality check on a "damaged
item" photo, and possibly a follow-up question to the customer, in an order that depends on what's
found at each step. This isn't a fixed 3-step pipeline; it's exactly the "dynamic, unknown path"
case §4 says warrants an agent.

**Architecture (§5.2a):** Plan-and-Execute fits well here — a stronger model drafts the overall plan
("check policy → check order history → check photo if applicable → decide"), and a cheaper model
executes each step's tool calls. A Replan edge handles the case where, say, the photo-quality check
comes back ambiguous and the plan needs to add a customer follow-up step.

**Guardrails (§6):** issuing a refund is a **moderate-cost, not fully reversible** action — apply
§6.1's think-vs-act framing: cheap, reversible steps (looking up order history) execute immediately;
the final refund decision goes through a confidence threshold (§6.2) tuned to this specific cost
asymmetry (a wrongly-denied legitimate return costs customer trust; a wrongly-approved fraudulent
return costs money directly) — not a threshold borrowed from a different task.

**Failure modes:** error compounding (§3) is a real risk across a 4–5 step pipeline — even at 95%
per-step reliability, a 5-step return-processing chain succeeds only ~77% of the time end to end, so
validate-retry-escalate (§6.3) at each step matters more than it would for a 1–2 step task.

**What you'd ship:** Plan-and-Execute with a confidence-gated final approval step, unit-test-style
validation on each tool call's output (§6.3), and an escalation path to a human agent whenever
confidence falls below the calibrated threshold or a step fails validation twice.

**Leadership Principle tie-in:** **Customer Obsession** — an agent that escalates an ambiguous return
rather than guessing protects the customer relationship more than one that always resolves
automatically. **Insist on the Highest Standards** — the validate-retry-escalate pattern (§6.3)
exists specifically so a plausible-looking but wrong decision doesn't ship silently.

---

## Glossary

- **Agent** — an LLM-based system that closes the loop: thinks, acts (tool calls), observes
  results, and decides the next step autonomously.
- **Autonomy ladder** — the four rungs of control-flow structure: L1 tool call (single call) → L2
  multi-step/ReAct (self-directed loop — where the "do I need an agent?" threshold sits) → L3 plan
  & adapt (plan → execute → replan) → L4 multi-agent (delegation).
- **Copilot** — a model that suggests actions but requires human approval before each step.
- **Error compounding** — the exponential degradation of success probability over multi-step
  tasks (0.95^n).
- **Hallucinated action** — an agent assuming a tool exists that doesn't, then fabricating its
  output.
- **LATS (Language Agent Tree Search)** — Tree of Thought with a more structured, search-guided
  expansion/scoring/backtracking step, closer to Monte Carlo Tree Search.
- **Memory bank** — persistent storage for lessons learned from past failures; injected into
  prompts on retry.
- **Plan-and-Execute** — a strong-model Planner drafts the full plan once; a cheap-model Executor
  runs each step; a Replan edge repairs failures. Good plan *proposer*, poor plan *guarantor*.
- **ReAct** — Reasoning + Acting framework: Think → Act → Observe cycle.
- **Reflection** — learning from mistakes via retrospective analysis and memory, without
  retraining.
- **Tree of Thought** — exploring multiple reasoning branches with backtracking, pruned by
  validation at each step.
- **Validate → Retry → Escalate** — the guardrail pattern that prevents infinite loops.

---

## Check yourself

1. What is the key difference between a chatbot, a copilot, and an agent? *(§1)*
2. Calculate the end-to-end success rate for a 10-step agent where each step is 90% reliable.
   What about 95%? *(§3)*
3. Why does a "95% reliable step" NOT mean a "95% reliable agent" for a 40-step task? *(§3)*
4. When should you use an agent vs. a hardcoded workflow? *(§4)*
5. Explain the ReAct framework. What goes wrong if you remove the "observation" step? *(§5.1)*
6. How does Reflection enable learning without retraining the LLM? *(§5.2)*
6a. Why did the classical planner hold 100% success on the Blocksworld benchmark under both
    familiar and obfuscated block names, while GPT-4 and Claude 3.5 Sonnet's success rates collapsed
    under obfuscation? *(§5.2a)*
7. When is Tree of Thought better than best-of-N chain-of-thought? *(§5.3)*
8. Design a confidence threshold system for an agent that can send customer emails. *(§6)*
9. What is the validate-retry-escalate pattern and why is it essential? *(§6.3)*

---

## Going deeper

1. **Yao et al. (2023), "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"**
   — ✅ verified via WebSearch against the paper's own listing: authors Shunyu Yao, Dian Yu, Jeffrey
   Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan; arXiv:2305.10601;
   published NeurIPS 2023. Named on the slides as the Tree of Thought paper. `solid`.

2. **Yao et al. (2023), "ReAct: Synergizing Reasoning and Acting in Language Models"** — ✅ verified
   via WebSearch: authors Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik
   Narasimhan, Yuan Cao; arXiv:2210.03629; published ICLR 2023. Named on the slides as the ReAct
   framework. `hard`. Foundational paper for agentic reasoning.

3. **Shinn et al. (2023), "Reflexion: Language Agents with Verbal Reinforcement Learning"** —
   ✅ verified via WebSearch: authors Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath,
   Karthik Narasimhan, Shunyu Yao; arXiv:2303.11366; published NeurIPS 2023. Referenced as the
   reflection framework. `solid`. The paper on agent self-reflection.

4. **Zhou et al. (2023), "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in
   Language Models" (LATS)** — ✅ verified via WebSearch: authors Andy Zhou, Kai Yan, Michal
   Shlapentokh-Rothman, Haohan Wang, Yu-Xiong Wang; arXiv:2310.04406, posted 2023 (published ICML
   2024). Not named on the slides by full citation, but the deck's own "ToT / LATS" slide names the
   method directly. `hard`.

5. **Valmeekam et al. (2023), "PlanBench: An Extensible Benchmark for Evaluating Large Language
   Models on Planning and Reasoning about Change"** — ✅ verified via WebSearch: authors Karthik
   Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, Subbarao Kambhampati;
   arXiv:2206.10498 (posted 2022, published NeurIPS 2023 Datasets & Benchmarks track — the "2023"
   citation year tracks its NeurIPS publication). Not named on the slides directly — this remains
   the likely source of the Blocksworld/VAL-verified benchmark shown in §5.2a, given the matching
   methodology (block-renaming, VAL plan verification), but the attribution to §5.2a's specific
   numbers is still an inference, not a slide-confirmed citation. `hard`.

> The citations above (authors, arXiv IDs, venues) are now confirmed against the papers' own arXiv
> listings. What remains unconfirmed is only the *attribution* of item 5 (PlanBench) as the exact
> source of §5.2a's Blocksworld benchmark — the deck itself never names this paper, so treat that
> specific link as a plausible, well-matched inference rather than a slide-stated fact.
