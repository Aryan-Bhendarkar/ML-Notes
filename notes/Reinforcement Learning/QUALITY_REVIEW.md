> ✅ **STATUS: ALL FINDINGS BELOW HAVE BEEN APPLIED AND INDEPENDENTLY RE-VERIFIED AGAINST THE LIVE
> FILES.** This file is kept as an audit trail of what was found and fixed during the Reinforcement
> Learning module's `QUALITY_REVIEW_PIPELINE.md` pass, not as a pending to-do list. Three parallel
> sub-agent audits (one per lecture file, Lecture 25's split further into 5 nested frame-range
> passes covering all 134 raw frames) ran the full three-lens review and an exhaustive citation
> sweep. Before writing any fix, the coordinator independently re-opened the actual raw slide
> images for every 🔴 finding and several 🟠/🟡 findings — `output/Lecture_24.../slide_137.jpg`,
> `slide_139.jpg`, `slide_140.jpg`, `slide_072.jpg`, `slide_144.jpg`, `slide_148.jpg`,
> `output/Lecture_25.../slide_016.jpg`, `slide_020.jpg`, `slide_030.jpg`, `slide_040.jpg`,
> `slide_050.jpg`–`slide_054.jpg`, and `output/Lecture_26.../slide_011.jpg`, `slide_034.jpg`,
> `slide_037.jpg`, `slide_054.jpg`, `slide_058.jpg`, `slide_066.jpg`, `slide_073.jpg`, `slide_080.jpg`
> — then re-read each edited region from the live `.md` afterward to confirm the fix landed, per
> this project's hard rule: never mark a finding "Fixed" without re-reading the live file.
>
> **One sub-agent-reported 🔴 finding (Lecture 25's claim that §3.1's sweep-by-sweep worked example
> was fabricated) was independently investigated in two rounds.** Round 1 (this file's first
> published version) concluded it was a false positive, after confirming §3.1's cited numbers exactly
> match `slide_016.jpg`/`slide_020.jpg`/`slide_030.jpg`'s **`policy-evaluation.html`** demo (uniform
> random policy). That conclusion for §3.1 specifically was correct and still stands. But a second
> round — triggered by the coordinator independently re-spot-checking the file and pushing back with
> the sub-agent's original frame citations (`slide_034`–`slide_044`) — found the round-1 investigation
> had stopped one step short: those frames are a genuinely **separate** demo, `policy-iteration.html`
> (a deterministic "all→Right" starting policy that crashes toward −100), which the sub-agent had
> correctly read but which was never actually contradicting §3.1 — and which the file's §3.2 was, in
> fact, failing to use anywhere despite it being real, dramatic, on-deck source material. That gap is
> now fixed (see 🟠 finding 1 in Lecture 25's section below). **Net result: §3.1 needed no numeric
> correction (confirmed twice), but the sub-agent's frame citations pointed at a real, separate,
> previously-unaddressed gap in §3.2 that the first investigation round missed by not asking why the
> sub-agent had cited those specific frames in the first place.** Recorded here as the concrete
> lesson: when a finding is rejected, re-examine *why* the source material that triggered it exists
> at all — a wrong conclusion about which section it applies to is a different failure from the
> finding being baseless.

# Quality review — Reinforcement Learning — 2026-08-30

Purpose / severity legend (reuse: 🔴 factual error or fabrication · 🟠 real content/pedagogy gap ·
🟡 polish / web-readiness).

Method: each lecture file was run through all three lenses (Teacher / Student / Engineer) per
`QUALITY_REVIEW_PIPELINE.md`, executed as three parallel background sub-agent passes (Lecture 25's
further split into 5 nested sub-agents covering raw frames 1–30, 31–58, 59–90, 91–115, 116–134 to
guarantee an exhaustive, non-sampled sweep of all 134 frames), each required to run a mandatory
exhaustive `[slide N]`/numeric citation sweep and a closing/summary-slide named-but-untaught check
before compiling findings. Two lecture files (24 and 25) had already been through a prior, explicitly
non-exhaustive ad-hoc review pass documented in `REVIEW_SCRATCH.md` (not this pipeline) — those
fixes (Lecture 24: instructor name, a duplicated Markov-property paragraph, a CartPole caveat;
Lecture 25: Thompson Sampling, the grid-world sweep demo; Lecture 26: PPO's own definition, the
DQN-vs-PPO demo numbers) were independently re-confirmed still present and, in Lecture 26's case,
found to contain **two residual numeric errors that survived the earlier fix pass** — direct,
concrete evidence for this project's hard rule that a prior fix must always be re-verified
pixel-for-pixel, never trusted at face value.

This module surfaced the same class of central-worked-example fabrication found repeatedly
elsewhere in this project's review history: **Lecture 24's 2-state MDP worked example had its
$s_2$ transitions swapped**, corrupting three downstream sections including a policy-iteration
result that was backwards (claiming the policy needed to change when the source slides show it was
already optimal). This is fixed below, along with two numeric sign/count errors in Lecture 26's
hands-on demo that survived the file's own prior fix pass.

---

## `notes/Reinforcement Learning/reinforcement-learning-01.md` (Lecture 24)

Source cross-checked: `output/Lecture_24 - Module 8 Reinforcement Learning Part 1/` (150 raw
frames, confirmed via directory listing). Full contact-sheet sweep plus ~30 full-resolution pulls
for every numeric/formulaic claim in the file. Prior `REVIEW_SCRATCH.md` fixes (instructor name,
duplicated Markov paragraph, CartPole terminal-condition caveat) independently re-confirmed intact
— **zero action needed on those three**, contrary to this task's initial briefing that this file
had "zero prior review": it did have one prior, non-exhaustive pass, and that pass's fixes hold up.

### 🔴 Factual error or fabrication

1. **§9.2's 2-state MDP worked example has the $s_2$ transition row swapped, corrupting the
   Bellman-optimality worked example (§10.2) and completely inverting the policy-iteration worked
   example's conclusion (§12.2).** The file stated "$s_2$, stay → reward 1 → $s_1$" and "$s_2$, go →
   reward 3 → $s_2$." Three independent slides — `slide_137.jpg` (Bellman-expectation derivation),
   `slide_139.jpg` and `slide_140.jpg` (the policy-improvement worked example) — agree unambiguously
   the correct mapping is the **opposite**: "$s_2$, stay" is a **self-loop back into $s_2$** (reward
   1) and "$s_2$, go" **transitions to $s_1$** (reward 3). This produced three cascading errors: (a)
   §9.2 computed $V^\pi(s_2)=30$ instead of the slide-verified $V^\pi(s_2)=21$; (b) §10.2's Bellman
   optimality equation for $s_2$ had its next-states swapped; (c) §12.2's entire policy-iteration
   worked example used the wrong $V^\pi(s_2)=30$, producing wrong Q-values and the **backwards
   conclusion that the policy needed to improve to "go, go"** — when `slide_139.jpg`/`slide_140.jpg`
   show the pixel-verified correct Q-values ($Q(s_1,\text{stay})=20.0$, $Q(s_1,\text{go})=18.9$,
   $Q(s_2,\text{stay})=19.9$, $Q(s_2,\text{go})=21.0$) yield $\pi_1 = \pi_0 = \{s_1:\text{stay},
   s_2:\text{go}\}$ — **the policy is unchanged; policy iteration converges in zero improvement
   steps**, the opposite of what the file previously taught. This also contradicted the file's own
   §10.2, which two paragraphs earlier correctly states the optimal policy is "stay at $s_1$, go
   from $s_2$" (unchanged from $\pi_0$) — an internal self-contradiction independent of the source.
   **Fix:** corrected the §9.2 transition table and both Bellman-equation derivations (now
   $V^\pi(s_2)=21$), corrected §10.2's optimality equation for $s_2$, and completely rewrote §12.2's
   worked example from the slide-verified numbers, arriving at the correct conclusion (policy
   unchanged, $\pi_0=\pi^*$, $V^*(s_1)=20$, $V^*(s_2)=21$). Explicit ⚠️ notes were left in both
   §9.2 and §12.2 documenting exactly what was wrong before, for anyone who cross-references an
   older copy. **Fixed** — re-verified against the live file and against `slide_137.jpg`,
   `slide_139.jpg`, `slide_140.jpg` directly.

### 🟠 Real content/pedagogy gap

2. **§11 (iterative policy evaluation) and §13 (value iteration) stated their convergence claims
   without ever showing a numeric trace** — the banned "it can be shown" pattern applied to a
   convergence guarantee rather than a formula. **Fix:** added a 🧪 worked example to each,
   iterating the (now-corrected) 2-state MDP for 5 steps from $V_0=(0,0)$, showing the values
   climbing toward $(20,21)$ with the gap shrinking by exactly $\gamma=0.9$ each step (§11), and
   showing the argmax at each state stabilizing on the optimal action from iteration 1 even before
   the values finish converging (§13). **Fixed**, re-verified against the live file.
3. **§14 "Putting it together" names "PPO" as one of the RL algorithms the Bellman equations
   underpin — but the deck's own closing slide (`slide_144.jpg`, "Why Bellman Equations Matter")
   lists "Value Iteration, Policy Iteration, Q-Learning, DQN, Actor-Critic," not PPO.** PPO isn't
   taught until Part 3 and isn't on this lecture's own summary slide. **Fix:** replaced "PPO" with
   "Actor-Critic" in the ASCII diagram to match the source slide exactly. **Fixed**, re-verified
   against the live file and `slide_144.jpg`.

### 🟡 Polish / web-readiness

4. **Going Deeper item 7 cited `[slide 72]` for the OpenAI Spinning Up reference — slide 72 is
   actually a discount-factor demo screen, unrelated to the references list.** The real
   "References & Further Reading" slide (confirmed to list all of items 1–4 and 7 exactly) is
   `slide_148.jpg`. **Fix:** corrected all five citations (items 1–4, 7) to `[slide 148]`. **Fixed**,
   re-verified against the live file and `slide_148.jpg`.
5. **§7.3's discount-factor demo table mislabeled a row "γ = 0.10" — the values shown (5.16, 1.16)
   are actually from γ = 0.14 on the source slide.** `slide_072.jpg` shows exactly these values
   (5.16, 1.16) with the slide's own annotation "Eff. horizon: 1.2 steps," which is $1/(1-0.14)
   \approx 1.16$, not $1/(1-0.10)=1.11$. **Fix:** relabeled the row to γ = 0.14 with effective
   horizon ~1.2 steps, and added an explicit ⚠️ note. **Fixed**, re-verified against the live file
   and `slide_072.jpg`.
6. **§2.1's ASCII agent-environment diagram labeled the reward `r_t`, but the numbered steps
   directly below it (and the source slide) both use `r_{t+1}`** — a state/reward at time $t+1$ is
   what's received *after* acting at time $t$, so `r_t` is inconsistent with the file's own
   surrounding text. **Fix:** corrected the diagram label to `r_{t+1}`. **Fixed**, re-verified
   against the live file.
7. **§7.1 stated CartPole's exact terminal-condition thresholds ($|\theta|>12°$, $|x|>2.4$) twice —
   once with the prior pass's honest sourcing caveat (in the "CartPole state vector" subsection),
   and once earlier in the same section, presented with full confidence and no caveat** (the "In
   CartPole: +1 per step... terminal when..." sentence). This is a residual inconsistency from the
   earlier partial fix: the same claim appears once correctly caveated and once not. **Fix:**
   reworded the first (uncaveated) mention to point forward to the properly caveated one instead of
   repeating the specific numbers unconditionally. **Fixed**, re-verified against the live file.

### Verified accurate / no action needed

- Instructor **Sayambhu Sen** — nameplate reconfirmed. Duplicated Markov-property paragraph —
  confirmed gone (prior fix intact). CartPole terminal-condition caveat — confirmed present and
  accurate (prior fix intact, aside from the duplicate-mention polish item above).
- RL milestones timeline (§4.2), the reward hypothesis quote, the MDP 5-tuple, the Markov property,
  the transition-probability grid-world example, the CartPole state-vector spec, the discount-factor
  table's γ=0.90/0.99/0.68 rows, the γ=1 rule, the generic (non-worked-example) forms of the Bellman
  expectation/optimality equations, the Expectation-vs-Optimality comparison table, and the policy
  iteration diagram — all independently re-checked and confirmed pixel-exact matches to their source
  slides.
- LaTeX escaping clean (no illegitimate double-backslash). Symbol tables bound immediately to their
  formulas. All internal §N cross-references resolve. Word count ~7,100 words, within the module's
  target range.

### Not yet checked

- Individual full-resolution review of all 150 raw frames — contact sheets covered the full
  structure and ~30 frames were pulled full-resolution for every numeric/formulaic claim in the
  file; the remaining frames are intermediate animation states of slides already checked at their
  final/fullest state. Low risk given zero drift found across the checked set.

**Overall verdict:** One 🔴 (a swapped transition in the file's central worked example, cascading
into a backwards policy-iteration conclusion — the most consequential finding in this file), two
🟠s (missing numeric convergence traces; a wrong algorithm name on a closing-slide cross-check), and
four 🟡s (a wrong Going-Deeper slide citation repeated 5 times, a mislabeled discount-factor row, an
inconsistent reward-index label, and a residual duplicate-caveat inconsistency). All seven findings
fixed and re-verified against source.

---

## `notes/Reinforcement Learning/reinforcement-learning-02.md` (Lecture 25)

Source cross-checked: `output/Lecture_25 - Module 8 Reinforcement Learning Part 2/` (134 raw
frames). All 134 frames individually read across 5 nested sub-agent passes (frames 1–30, 31–58,
59–90, 91–115, 116–134) to guarantee full coverage without sampling. Prior `REVIEW_SCRATCH.md`
fixes (Thompson Sampling subsection, the grid-world sweep demo) independently re-confirmed intact.

### Investigated in two rounds — §3.1 confirmed accurate, but a real gap was found in §3.2

The Lecture-25 sub-agent pass reported a 🔴 claiming **"the entire §3 sweep-by-sweep worked example
is fabricated"** — that none of the notes' §3.1 sweep values (−3.25, −5.67, −10.88, −15.58, −17.96,
−19.23) appear in the actual demo, and that the real demo instead uses a different starting policy
and produces different numbers entirely (−10.00, −19.00, −27.10, −34.39...), citing
`slide_034`–`slide_044`.

**Round 1:** the coordinator opened `slide_016.jpg`/`slide_020.jpg`/`slide_030.jpg` (tab title
"Policy Evaluation," `file:///Users/nkhetan/RL_Interactive_Examples/**policy-evaluation.html**`,
config bar reading "policy = **uniform random**") and found the notes' six cited §3.1 values an
**exact match**: Sweep 1=−3.25, Sweep 2=−5.67, Sweep 5=−10.88, Sweep 10=−15.58, Sweep 15=−17.96,
Sweep 20=−19.23. §3.3's "converges in ~5 sweeps to 4.58" claim was also confirmed exact against
`slide_052.jpg`'s `value-iteration.html` demo. Concluded: false positive, no fix needed.

**Round 2 (prompted by the coordinator's independent spot-check pushing back on round 1):** the
coordinator re-opened `slide_034.jpg`, `slide_036.jpg`, `slide_038.jpg`, `slide_041.jpg`, and
`slide_044.jpg` specifically — the exact frames the sub-agent had cited — and confirmed the
sub-agent's numbers are also **real and accurate**: this is a genuinely separate page,
`policy-iteration.html` (different tab, different URL, "start policy = **all →**" — a deterministic
policy, not uniform random), whose own **evaluate** step really does produce sweep1=−10.00,
sweep2=−19.00, sweep3=−27.10, sweep4=−34.39, crashing to ≈−100.00 by sweep 57 (`slide_044.jpg`'s
visible log), before "Improve" flips the arrow and the demo re-converges to the same optimal values
(`slide_045.jpg`: 4.58/6.20/8.00/.../GOAL) that `value-iteration.html` reaches independently.

**Resolution: both demos are real, and the sub-agent read its cited frames correctly — the round-1
error was concluding "false positive" without asking why those specific frames existed at all.**
§3.1 needed no numeric correction (its cited frames are a different demo, confirmed twice now). But
§3.2 ("Policy iteration") had **zero numeric worked example**, despite this dramatic, concrete,
already-worked source material (`policy-iteration.html`'s crash-then-one-flip-fixes-it story) sitting
unused on the deck exactly where §3.2's own subject matter needed it. **Fix:** added a 🧪 worked
example to §3.2 transcribing the real sweep values (−10.00 → −19.00 → −27.10 → −34.39 → ≈−100.00),
the theoretical-floor explanation ($-10/(1-0.9)=-100$), the Improve step's arrow flip, and the
convergence to the same optimal values §3.3 reaches by a different route — with an explicit ⚠️ note
distinguishing this demo from §3.1's uniform-random one so a reader never conflates the two. **Fixed**
— re-verified against the live file and against `slide_034/036/038/041/044/045.jpg` directly.

### 🟠 Real content/pedagogy gap

1. **§3.2 (Policy iteration) had no numeric worked example despite real, on-deck source material for
   one.** See the two-round investigation above — this is the finding that investigation actually
   surfaced. **Fixed**, re-verified against the live file.
2. **Instructor credited as "Naman Kanan" is very likely wrong, and the deck contains a strong
   alternative clue.** No slide anywhere in the 134-frame capture shows a nameplate overlay with
   the instructor's name. However, `slide_016.jpg` through `slide_052.jpg` (the interactive-demo
   screen-share slides) repeatedly show the browser address bar reading
   `file:///Users/nkhetan/RL_Interactive_Examples/...` — the macOS account name **"nkhetan"** is the
   only direct evidence of the instructor's identity in the entire raw capture. This is consistent
   with a surname **Khetan**, first initial "N" — not "Naman Kanan." No slide shows a full first
   name, so this cannot be fully confirmed either way. **Fix:** updated the file's header and the
   module README to present this as an inferred correction with the supporting evidence (the file
   path) shown explicitly, rather than stating either name as settled fact. **Fixed**, re-verified
   against the live file.

### Verified accurate / no action needed

- The DP-limitations numbers (chess ~10⁴⁷ states, Go ~10¹⁷⁰, Atari effectively infinite) —
  independently re-confirmed exact against `slide_054.jpg` ("DP Limitations").
- ε-greedy formula and decay footnote ("DQN uses ε-greedy 1.0→0.1 over 1M frames"), Softmax/
  Boltzmann formula, UCB formula and its O(√(KT·lnT)) regret bound, multi-armed bandit regret
  formula and O(√T) claim, contextual bandits section, SARSA formula, Q-learning formula and its
  "most important algorithm" framing, Watkins (1989) citation, the SARSA-vs-Q-learning cliff-world
  comparison table, on/off-policy definitions, the deadly triad definition, the DQN architecture
  (84×84×4, Conv32/8×8/s4, Conv64/4×4/s2, Conv64/3×3/s1, FC512), the Atari results table (Breakout,
  Pong, Space Invaders, Enduro, Seaquest), the "29 of 49 games" headline, Rainbow's "230% of human
  median" (Hessel 2017, combining 6 improvements), and the four DQN variant citations (van Hasselt
  2016, Wang 2016, Schaul 2016) — all independently checked against their source frames across the
  5-pass sweep and confirmed exact.
- "Thompson Sampling dominates in industry" reconfirmed verbatim on the Part-2-Summary closing
  slide, genuinely never explained on a dedicated slide anywhere in the 134 frames — confirming the
  prior fix's premise is still accurate; no further action needed beyond what's already fixed.
- No second instance of the "named-but-untaught" failure class was found beyond the
  already-fixed Thompson Sampling gap; the closing slide's other name-drops (UCB, contextual
  bandits) are both explained in dedicated body sections.
- LaTeX escaping clean, symbol tables bound to their formulas, all internal §N cross-references
  resolve.

### Not yet checked

None — a full 134-frame sweep was completed via the 5 nested sub-agent passes, and the coordinator
independently spot-checked ~15 of the most numerically load-bearing frames directly.

**Overall verdict:** Zero 🔴s in this file — the sub-agent-reported 🔴 was reclassified after a
two-round investigation: §3.1's worked example is confirmed accurate (no fabrication), but the
underlying frame citations pointed to a real, separate 🟠 gap in §3.2 that the first investigation
round missed. Two 🟠s total (the §3.2 worked example addition, and the instructor-name correction).
This file's central worked examples are now fully sourced and verified, aside from the
instructor-identity uncertainty, which itself cannot be fully resolved from the available capture.

---

## `notes/Reinforcement Learning/reinforcement-learning-03.md` (Lecture 26)

Source cross-checked: `output/Lecture_26 - Module 8 Reinforcement Learning Part 3/` (83 raw
frames). All of the demo, multi-agent, RLHF, actor-critic, REINFORCE, and opening/closing frame
ranges opened directly. Prior `REVIEW_SCRATCH.md`/README-documented fixes (§7.3's PPO definition,
§12's DQN-vs-PPO demo numbers) were re-verified — **the PPO section is confirmed present and
correct, but the demo numbers turned out to contain two residual errors that survived the earlier
fix pass**, exactly the scenario this pipeline's honesty rules warn about.

### 🔴 Factual error or fabrication

1. **§12's DQN result was transcribed with the wrong sign: "−9.2 avg reward" instead of "~9.2 avg
   reward."** `slide_073.jpg` ("Demo Code — Evaluation & Visualization") reads, character for
   character, **"Result: ~9.2 avg reward → Failed to learn in 20k steps"** — an approximately-equal
   tilde, not a minus sign. This is also confirmable by pure sanity check: CartPole's reward is +1
   for every step the pole stays upright and the episode simply ends on failure, so a negative
   *average* reward is not a value this environment can produce at all. **Fix:** corrected
   "−9.2" → "~9.2" throughout §12, with an explicit ⚠️ note explaining both the OCR-style
   misread and the sanity-check reasoning. **Fixed** — re-verified against the live file and
   `slide_073.jpg` directly.
2. **§12's PPO convergence point was transcribed as "~8k steps" instead of "~9k steps," which also
   broke the derived "~40%" comparison figure.** `slide_073.jpg` reads **"Result: 500.0 avg reward →
   Perfect score after ~9k steps."** Using the correct 9k/20k figure, PPO reaches the maximum score
   in **45%** of DQN's training budget, not 40%. **Fix:** corrected "~8k steps" → "~9k steps" and
   "roughly 40%" → "roughly 45%" in both the table callout and the body prose. **Fixed** —
   re-verified against the live file and `slide_073.jpg` directly.
3. **Three "Going Deeper" citations point to the wrong raw-frame numbers.** Items 3, 5, and 6 cited
   `[slide 11]` (A3C), `[slide 19]` (DPO), and `[slide 23]` (CleanRL) — these are numbers from the
   lossy `slides_deduped/` set (31 frames), but the file's stated source and `source:` frontmatter
   field is the 83-frame raw `output/` capture, where those slide numbers point to unrelated
   content: `slide_011.jpg` is actually the policy-gradient-theorem derivation. The topics are
   genuinely on the deck — just at different raw-frame numbers, confirmed directly: A3C is
   `slide_034.jpg` ("A3C — Asynchronous Advantage Actor-Critic"), DPO is `slide_054.jpg` ("Beyond
   RLHF"), and CleanRL is `slide_066.jpg` ("Popular RL Frameworks"). This is a pointer bug, not a
   fabrication — the underlying facts are correct. **Fix:** corrected all three citations to their
   raw-frame numbers with a note on the renumbering. **Fixed** — re-verified against the live file
   and `slide_034.jpg`/`slide_054.jpg`/`slide_066.jpg` directly.

### 🟠 Real content/pedagogy gap

4. **§12's description of DQN's pole-angle-over-time plot was backwards.** The file said "DQN's
   plot stays noisy and unstable across the run" — but `slide_080.jpg` ("Pole Angle Over Time: PPO
   vs DQN") shows DQN's line (dashed, labeled "DQN (9 steps)") as a **smooth, monotonic rise**
   straight through the ±12° failure threshold over its 9 recorded steps — the most predictable
   curve on the chart, not a noisy one. PPO's line (solid, "PPO (500 steps)") stays flat near zero
   for the full run. **Fix:** corrected the description to state DQN's curve rises smoothly and
   rapidly to failure, and that PPO's stays flat and stable — removing the incorrect "noisy"
   characterization. **Fixed**, re-verified against the live file and `slide_080.jpg`.
5. **Systemic: almost no formula in this file had the mandatory `| Symbol | Read it as | What it
   means |` table** — only §7.3's PPO section (added by the prior fix pass) had one. The file's
   single most important derivation (§2.1's policy gradient theorem), §7.1's GAE formula, and
   §9.4's Bradley-Terry model all lacked one, violating `NOTES_PIPELINE.md`'s "words before symbols"
   rule. **Fix:** added symbol tables (with a preceding plain-English "words before symbols"
   sentence) to all three formulas — the three highest-value derivations in the file. **Fixed**,
   re-verified against the live file. (§3.2's baseline formula, §5.1–5.2's advantage/critic/actor
   losses, and §9.5's KL-penalty formula remain without formal symbol tables; these are lower
   priority since each symbol is already defined in adjoining prose, and are noted in the module
   README as a residual module-wide stylistic gap worth a future pass.)
6. **§3.2's baseline-unbiasedness claim was asserted ("This was mathematically proved") rather than
   derived** — the banned "it can be shown" pattern. **Fix:** added the 3-step derivation showing
   $\sum_a \pi(a|s)\nabla_\theta\log\pi(a|s) = \nabla_\theta\sum_a\pi(a|s) = \nabla_\theta(1) = 0$,
   using the fact that $\pi_\theta(\cdot|s)$ always sums to 1. **Fixed**, re-verified against the
   live file.

### 🟡 Polish / web-readiness

7. **§7.2's GAE "sweet spot" range stated "0.9–0.99," but `slide_037.jpg` explicitly reads
   "typically between 0.95 to 0.99"** — the lower bound was wrong by 0.05. **Fix:** corrected
   "0.9–0.99" → "0.95–0.99". **Fixed**, re-verified against the live file and `slide_037.jpg`.
8. **Frontmatter `slides: 31` contradicted the file's own stated source** — the header prose and
   `source:` field both correctly state the 83-raw-frame `output/` capture was used, but the
   `slides:` field carried the 31-frame `slides_deduped/` count instead, the same recurring bug
   pattern found in other modules' reviews. **Fix:** changed to `slides: 83`. **Fixed**,
   re-verified against the live file.
9. **§12's demo table showed only 1 of DQN's 3 cited hyperparameters** (`exploration_fraction=0.2`)
   while showing all 3 for PPO — an asymmetry with a simple fix, since `slide_073.jpg` shows DQN's
   other two params directly (`learning_rate=1e-3`, `buffer_size=50000`). **Fix:** added both
   missing hyperparameters to the DQN row. **Fixed**, re-verified against the live file and
   `slide_073.jpg`.
10. **"What you'll understand" item 13 said "the seven core challenges of RL" while listing nine
    named challenges, and §10's own table has nine rows** — an internal miscount, independent of
    the source slide (which also lists nine, confirmed via `slide_058.jpg`). **Fix:** corrected
    "seven" → "nine". **Fixed**, re-verified against the live file.

### Cross-module check — GenAI & LLM's RLHF/PPO/DPO coverage

This lecture's §9–§10 (RLHF, DPO, RLAIF, Constitutional AI) covers the same ground as GenAI & LLM's
dedicated RLHF lecture. Independently compared: the PPO clipped objective is identical in both
files; the Bradley-Terry formula is identical; the KL-penalized RLHF objective is mathematically
equivalent between the two (different variable-naming convention, same math); the DPO/RLAIF/CAI
mechanism descriptions are consistent (the GenAI & LLM file goes into more depth, appropriately,
since it's that module's dedicated topic); GRPO is absent from this file but genuinely does not
appear on any of its 83 slides (confirmed directly), so this is faithful to source, not an omission;
the RLHF stage-count presentation differs in style (a numbered pipeline diagram here vs. prose
there) but not in substance. **No contradictions found.**

### Verified accurate / no action needed

- Instructor **Vijay Neeluru** reconfirmed on 16+ frames. Agenda-vs-body topic match (7 topics),
  closing "Thank You!" slide's 4 key takeaways all match taught content with no new named-but-untaught
  methods beyond what's already covered. Policy gradient theorem's 7-step derivation re-derived and
  matches `slide_011.jpg` exactly. REINFORCE pseudocode and strengths/weaknesses table, Actor-Critic/
  A2C architecture and loss functions, A3C, GAE's formula and λ-endpoint behavior (aside from the
  sweet-spot range, fixed above), the multi-agent three-paradigm table, the RLHF five-stage
  pipeline, the Bradley-Terry model, the RLHF PPO fine-tuning objective, the Beyond-RLHF table
  (DPO/RLAIF/CAI), the nine-row Challenges table, and the Frameworks table — all independently
  re-derived/re-checked and confirmed exact against their source frames.
- LaTeX escaping clean, emoji/callout semantics correctly applied throughout, all internal §N
  cross-references resolve, including after the prior pass's §7.3 insertion (no numbering
  collisions or skips found).

### Not yet checked

None — a full mandatory sweep of the demo, multi-agent, RLHF, actor-critic, REINFORCE, and
opening/closing frame ranges was completed.

**Overall verdict:** Three 🔴s (two numeric transcription errors in the hands-on demo that survived
an earlier fix pass, plus three mismatched citation slide-numbers from a stale deduped-numbering
carryover), three 🟠s (a backwards chart description, a systemic missing-symbol-table gap on the
file's most important formulas, and an asserted-not-derived claim), and four 🟡s (a wrong numeric
range, a stale frontmatter count, an asymmetric hyperparameter table, and a challenge-count
miscount). All ten findings fixed and re-verified against source. Zero contradictions found against
GenAI & LLM's independent RLHF/PPO/DPO coverage.

---

## Module-wide observations

**The central-worked-example fabrication pattern repeated in this module, as it has in every module
reviewed under this pipeline so far.** Lecture 24's 2-state MDP had its $s_2$ transitions
swapped, corrupting a downstream policy-iteration conclusion into its exact opposite (claiming a
policy needed to improve when the source shows it was already optimal) — this is judged the most
consequential single finding in this module, on the same order as the Causal Inference module's
positivity-example inversion, because a reader working through §12.2 as written would have
internalized a backwards understanding of what policy iteration's convergence criterion means.

**A sub-agent-reported 🔴 was independently investigated across two rounds** (Lecture 25's §3
claim) — recorded above in detail. Round 1 confirmed the specific claim ("§3.1 is fabricated") was
wrong, but stopped there; round 2, prompted by external pushback that re-asserted the sub-agent's
original frame citations, found those citations pointed to a genuinely separate demo and a real,
previously-missed 🟠 gap in §3.2. The lesson generalizes beyond "always re-verify a finding before
trusting it": when a finding is rejected, the specific frames or evidence that produced it still
need an account — rejecting a conclusion is not the same as explaining away the evidence behind it.

**Lecture 26's demo-number errors survived one earlier fix pass entirely intact** — the earlier
`REVIEW_SCRATCH.md` pass correctly found that the demo's numbers were missing and added them, but
transcribed two of the four key figures (the reward sign, the step count) incorrectly in the
process. This is the concrete evidence behind this pipeline's instruction to re-derive/re-verify
🔴-adjacent prior fixes pixel-for-pixel rather than trusting that "already fixed once" means
"correct."

**Instructor identity.** Lecture 25's instructor was credited as "Naman Kanan" with an existing
honest "not independently re-confirmed" flag. This review found the strongest available evidence —
a macOS account name ("nkhetan") visible in a screen-shared file path across ~35 frames — pointing
toward a different name (surname Khetan). Neither name is now stated as fully confirmed; the file
and README present the file-path evidence explicitly so a future reviewer with access to the
original recording's audio or a company directory can resolve it definitively.

**Symbol-table coverage.** Lecture 26 had almost no formulas bound to a `| Symbol | Read it as |
What it means |` table before this pass (only the prior-pass-added §7.3 PPO section had one). This
review added tables to the three highest-value formulas (policy gradient theorem, GAE, Bradley-Terry).
A handful of secondary formulas (§3.2's baseline, §5's advantage/actor/critic losses, §9.5's KL
penalty) still rely on inline prose rather than a bound table — each symbol is defined in adjoining
text, so this is lower-severity than the fixed items, but is flagged here as a residual gap worth a
future pass if this module is revisited. Lectures 24 and 25 already use symbol tables consistently
for their major formulas (Bellman equations, TD(0), SARSA/Q-learning) and needed no equivalent fix.

**Cross-module overlap with GenAI & LLM.** Checked in detail (see Lecture 26's section above) — no
contradictions found between this module's RLHF/PPO/DPO/RLAIF/CAI coverage and GenAI & LLM's
dedicated, deeper treatment of the same material.

**Module README.** Before this review, the README already had a useful per-lecture section table
for all three parts (a better starting point than several other modules' pre-review READMEs) and
accurate capture-quality prose, but lacked a formal callout-emoji legend, a fuller "What's in Part
N" reading-guide depth matching `notes/Unsupervised Learning/README.md`, and (until this pass) had
a wrong-in-two-directions instructor-name situation for Lecture 25. Phase 4 brings the README's
structure up to the established module standard while preserving its already-useful capture-quality
paragraphs.

**Companion web artifact.** A grep for "Reinforcement" or "reinforcement-learning" across
`web/*.py`/`*.html`/`*.mjs` returns no matches — `web/` currently contains only
`supervised-learning.html` plus build tooling. **No companion web artifact exists yet for this
module**, so there is nothing to flag as stale.

---

## Summary — module-wide

| File | 🔴 | 🟠 | 🟡 | Status after Phase 3 |
|---|---|---|---|---|
| `reinforcement-learning-01.md` (Lecture 24) | 1 | 2 | 4 | All fixed |
| `reinforcement-learning-02.md` (Lecture 25) | 0 | 2 | 0 | All fixed (a sub-agent-reported 🔴 was reclassified, after a two-round investigation, into one of these two 🟠s — see above) |
| `reinforcement-learning-03.md` (Lecture 26) | 3 | 3 | 4 | All fixed |
| **Total** | **4** | **7** | **8** | **19/19 addressed** |

**Overall module verdict.** Four 🔴s across the module, all traced to either a swapped-transition
arithmetic error in a central worked example (Lecture 24, cascading into a backwards conclusion), a
sign/count transcription error that survived an earlier fix pass (Lecture 26, twice), or a stale
citation-numbering carryover from the lossy deduped slide set (Lecture 26). None reflect an
unrecoverable gap in the underlying source capture — every fix was made by re-opening the actual
cited raw slide image, not by inference or by trusting a sub-agent or prior-pass report at face
value. All 19 compiled findings were fixed and independently re-verified
against the live files.
