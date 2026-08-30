> ✅ **STATUS: ALL 41 FINDINGS BELOW HAVE BEEN APPLIED AND INDEPENDENTLY RE-VERIFIED AGAINST THE
> LIVE FILES.** This file is kept as an audit trail of what was found and fixed during the Agentic
> AI module's `QUALITY_REVIEW_PIPELINE.md` pass — the **final module** of this project's 9-module
> review — not as a pending to-do list. Three parallel sub-agent audits (one per lecture file) each
> ran the full three-lens review, a mandatory exhaustive citation/number sweep, and a closing/
> summary-slide named-but-untaught check. Before compiling or fixing anything, the coordinator
> independently re-opened 18+ of the highest-severity cited raw slide images directly —
> `output/Lecture_27.../slide_012.jpg`, `slide_013.jpg`, `slide_014.jpg`, `slide_026.jpg`,
> `slide_028.jpg`, `slide_038.jpg`, `slide_048.jpg`, `slide_050.jpg`;
> `output/Lecture_28.../slide_001.jpg`, `slide_016.jpg`, `slide_025.jpg`; and
> `output/Lecture_29.../slide_034.jpg`, `slide_041.jpg`, `slide_045.jpg`, `slide_053.jpg`,
> `slide_056.jpg`, `slide_066.jpg`, `slide_072.jpg`, `slide_078.jpg` — every single one confirmed
> the sub-agent report exactly, with zero discrepancies found. All fixes were then applied directly
> to the three lecture files and each edited section was re-read from the live file afterward to
> confirm it landed, per this project's hard rule: never mark a finding "Fixed" without re-reading
> the live file. Lecture 29 (the module's capstone lecture) turned out to have **the highest
> fabrication density found across this entire 9-module project** — 8 🔴 findings in one file —
> despite having already been through one prior *targeted* fix pass; see the module-wide
> observations section below for what that implies for future maintenance of this course's notes.
>
> **One sub-agent-reported 🟠 finding (Lecture 29's "researcher/coder/reviewer with `depends_on`"
> Supervisor dispatch demo) was investigated directly before being applied and did not check out** —
> the coordinator opened every plausible slide in that lecture's topology/communication range and
> found no such code anywhere. Rather than fix around an unconfirmed citation, the claim was
> corrected in place with the investigation documented (see Lecture 29's finding 9). This is this
> module's own instance of the Reinforcement Learning module's process lesson: a sub-agent's cited
> evidence needs a direct look before either accepting or rejecting the finding built on it — here,
> unlike the RL case, the citation itself genuinely did not hold up.

# Quality review — Agentic AI — 2026-08-30

Purpose / severity legend (reuse: 🔴 factual error or fabrication · 🟠 real content/pedagogy gap ·
🟡 polish / web-readiness).

Method: each lecture file was run through all three lenses (Teacher / Student / Engineer) per
`QUALITY_REVIEW_PIPELINE.md`, executed as three parallel background sub-agent passes (one per
lecture), each required to run a mandatory exhaustive `[slide N]`/numeric citation sweep and a
closing/summary-slide named-but-untaught check before compiling findings. The coordinator then
independently re-opened 18+ of the highest-severity cited raw slide images across all three files
before writing any fix — every single one confirmed the sub-agent's report exactly (slides
013/014/012/026/028/038/048/050 for Lecture 27; slides 001/016/025 for Lecture 28; slides
034/041/045/053/056/066/072/078 for Lecture 29) — giving high confidence in the remaining,
lower-severity 🟠/🟡 findings that were not individually re-opened by the coordinator but are each
backed by an exact slide/frame citation from the sub-agent report.

This module surfaced **the highest fabrication density found across the entire 9-module project**
in Lecture 29 (8 🔴 + 7 🟠), even though a prior *targeted* fix pass had already caught one
fabrication and added three content blocks in that same file — direct, concrete evidence for this
project's standing rule that "already reviewed once" must never lower scrutiny on a subsequent
pass. Lectures 27 and 28 show a related but distinct failure mode: not fabrication but systematic
**under-transcription** — real, clearly legible, directly relevant numbers and worked examples on
the slides that simply never made it into the notes.

---

## `notes/Agentic AI/agentic-ai-01.md` (Lecture 27 — Fundamentals, Architectures, Reasoning)

Source cross-checked: `output/Lecture_27 - Bonus Module Agentic AI Part 1/` (53 raw frames), 3
contact sheets + ~20 full-resolution pulls, plus coordinator re-verification of slides
012/013/014/026/028/038/048/050 directly.

### 🔴 Factual error or fabrication

1. **§2 "The Autonomy Ladder" table is a different, invented 4-row structure that does not match
   the actual slide.** The file's table (Chatbot / Copilot / Agent-supervised / Agent-autonomous)
   does not appear on the deck at all. `slide_013.jpg`/`slide_014.jpg` ("The autonomy ladder") show
   a specific 4-rung ladder keyed to a single worked task ("I can come Monday but my passport delay
   may push me to Wednesday. Can I still surf Tuesday with cancellation insurance?"): **L1 Tool
   call** (`run_function(llm_tool, llm_args)`, single call, worked example marked **PARTIAL**/wrong
   answer) → **L2 Multi-step (ReAct)** (`while llm_should_continue(): step()`, self-directed loop,
   marked **ANSWERED**/correct — this is also the slide's own "do I need an agent?" threshold line)
   → **L3 Plan & adapt** (`plan → execute → replan`, autonomous goal pursuit) → **L4 Multi-agent**
   (`if llm_trigger(): run_agent()`, delegation/collaboration). This ladder is the direct
   throughline connecting ReAct (§5.1), Plan-and-Execute (§5.2a), and the closing slide's
   multi-agent teaser — its replacement with an invented, unrelated table is structural, not
   cosmetic. **Fix:** replace §2 entirely with the real L1–L4 ladder, its code-style row labels, the
   passport/Wednesday/insurance worked task, and the "do I need an agent?" threshold marking L1 as
   insufficient and L2 as where the task is actually answered.
2. **§5.3 Tree of Thought's ASCII diagram is fabricated arithmetic that does not match the source
   slide.** The file shows `[Start: 2,4,6,8]` with invented branches (`6-4=2`, `4+8=12`, `4×6=24 ✗
   (no multiplication allowed)`, `2+8=10`, `12-2=10`, `10-8=2 ✗`). `slide_038.jpg` ("Tree of
   Thoughts and LATS") shows a completely different, real tree: root **`4 4 6 8`**, with three
   generated branches — `6-4=2` (marked "maybe," later backtracked into), `4+8=12` (marked "sure"),
   and `4×6=24` (marked **impossible**, crossed out — not because "no multiplication allowed," a
   reason invented by the notes and never stated on the slide) — the surviving `4+8=12` branch then
   evaluates to `6-4=2` and finally `=24` (reached, marked correct), while the `6-4=2` branch's own
   sub-attempt `2×8` is crossed out and explicitly labeled "backtrack." **Fix:** redraw the ASCII
   tree to match the real root/branches/labels exactly, and remove the invented "no multiplication
   allowed" claim (the slide never states why `4×6=24` is impossible — the honest read is that using
   6 and 4 to make 24 strands the other two tiles unused, but this file should not assert a reason
   not on the slide).
3. **§1.1's Agent-column example row is fabricated.** The file lists "Claude Code, Cursor,
   Codebuff" as the Agent example row. `slide_012.jpg` ("Chatbot, copilot, agent") shows the real
   examples as three specific *task* cards — "**ReAct web researcher**" (`while
   llm_should_continue(): act`), "**SWE-bench coder**" (`model owns the loop, edits + tests`), and
   "**Multi-agent orchestrator**" (`an agent spawns other agents`) — not named commercial products.
   The notes also drop the slide's own clarifying line: *"A RAG bot with tools is still a chatbot: a
   human drives each turn."* **Fix:** replace the example row with the three real task cards and add
   the RAG-bot clarification sentence.

### 🟠 Real content/pedagogy gap

4. **§5.1 ReAct presents an unqualified win, omitting the slide's own caveat.** `slide_026.jpg`
   explicitly states: *"Caveat: on HotpotQA it still trails plain CoT (27.4 vs 29.4 EM); grounding
   trades hallucination for retrieval errors."* This nuance (ReAct is not strictly better in every
   setting) is real, on-slide, and entirely absent from the notes. **Fix:** add this figure and
   caveat to §5.1.
5. **§5.2 Reflection states "success rate improves dramatically" with no number, though the slide
   has one.** `slide_028.jpg` ("Reflexion: learning from failure without training") shows a
   quantified result: with reflection on, **130/134 tasks solved**, climbing from ~22% at trial 0 to
   ~97% by trial 12; with reflection off, the curve **stalls near ~55–58% by trial 7** and plateaus.
   **Fix:** add these concrete figures to §5.2.
6. **§5.3 omits the slide's own head-to-head comparison numbers.** `slide_038.jpg` states: at a
   matched ~6k-token budget, ToT (b=5) reaches **74%** vs. CoT best-of-100's **49%** (a 25-point
   gain from structured search alone), against a single-chain baseline of **4%**. The notes state
   the 74% figure but never give the 49%/4% comparison points that make the "why ToT helps" argument
   concrete. **Fix:** add the 49% best-of-100 and 4% single-chain figures alongside the existing 74%.
7. **§6.3 Validate→Retry→Escalate has zero numbers despite a rich, directly-relevant worked example
   on the deck.** `slide_050.jpg` ("Validate, retry, escalate") shows a SWE-bench-Lite pass@k chart:
   single-attempt SOTA = **43%**, "selected by vote" (no verifier) stays roughly flat at **~38–40%**
   even out to k=250, while true coverage (answer somewhere in the batch) climbs past **55%** — the
   slide's own point being "found, not selected": sampling finds the fix far more often than
   voting can identify it, hence the need for an independent verifier (unit test) rather than a
   self-selecting vote. **Fix:** rewrite §6.3 around this concrete example and figures.
8. **§6.2 omits a specific calibration technique and number present on the deck.**
   `slide_046.jpg`–`slide_048.jpg` state: *"Only if confidence is honest: RLHF is over-confident;
   verbalising halves the error"* — i.e., using the model's verbalized (stated) confidence instead
   of raw RLHF log-probabilities **cuts Expected Calibration Error (ECE) by a measured 50%**. This
   is the load-bearing precondition for the whole confidence-threshold mechanism and is missing
   entirely. **Fix:** add this to §6.2, before the threshold-table discussion.
9. **§6.2's table shows "—" for τ\*=0.43's human-interrupts count, but the slide states it
   directly.** `slide_048.jpg` ("Asking versus proceeding") shows, at τ\*=0.43: wrong actions shipped
   = 0, **human interrupts = 6**, total cost = 6. The file's placeholder "—" understates what's
   actually recoverable from the source. **Fix:** replace "—" with **6**.
10. **The closing slide names "Memory" as one of four pillars, but Memory-as-a-general-concept
    (distinct from Reflection's specific memory bank) is never taught as its own idea in the body.**
    `slide_053.jpg`'s summary framing (implied by the file's own §7 "Putting it together" diagram,
    itself derived from the deck's closing slide) lists Architecture, Decision policy, Guardrails,
    and Memory as the four pillars — but the file only ever discusses memory in the narrow context of
    Reflection's memory bank (§5.2), never as a standalone concept the way Part 2 (agentic-ai-02.md)
    later develops it in depth. **Fix:** add a short subsection (or a clarifying paragraph in §7)
    explicitly naming "memory" as a fourth pillar and forward-referencing Part 2's fuller treatment,
    so the closing diagram's own vocabulary is grounded somewhere in this file's body.
11. **Zero interactive spec blocks despite the source deck being built almost entirely out of
    literal interactive demos.** The 0.95^n curve (`slide_016`–`018`, an actual draggable-style
    illustration) and the τ confidence-threshold slider (`slide_046`–`048`, an actual draggable
    slider with a live cost readout) are the two clearest candidates — both are exactly the kind of
    "a threshold sliding" example `NOTES_PIPELINE.md` Phase 3 calls out by name. **Fix:** add two
    `interactive` spec blocks (type: slider) for §3.1 (error-compounding curve) and §6.2
    (confidence-threshold trade-off), each with a genuine fallback.
12. **§3.1's error-compounding table and the file's own Whiteboard-derivation section disagree at
    n=40.** The §3.1 table states 0.95^40 → **12%**; the "Whiteboard-ready derivation" section later
    in the same file computes 0.95^40 ≈ **0.129** and reports it as **"12.9%"**. These are the same
    quantity rounded two different ways in two places in one file — an internal inconsistency
    independent of the source. **Fix:** standardize both to the same rounding (0.95^40 ≈ 12.9%,
    consistent with 0.95^10≈60% and 0.95^100≈0.6% already used elsewhere in the file).
13. **§3.1's p^n formula has no symbol table**, violating `NOTES_PIPELINE.md`'s "words before
    symbols, every symbol defined" rule (the rule applies regardless of how intuitive the symbols
    seem). **Fix:** add a 2-row `| Symbol | Read it as | What it means |` table for $p$ and $n$
    immediately under the formula.

### 🟡 Polish / web-readiness

14. **Zero `[slide N]` citations anywhere in the file.** Per `NOTES_PIPELINE.md`'s formatting
    guidance and this pipeline's Lens 3 sourcing standard, numeric/worked-example claims should cite
    their slide. **Fix:** add `[slide N]` citations at minimum to §3.1 (16–18), §5.1 (26), §5.2
    (28), §5.2a (already has an implicit source but add explicit slide numbers), §5.3 (38), §6.2
    (46–48), and §6.3 (50).
15. **No `📚` background-callout emoji used**, despite three "Prerequisite" subsections in "Before
    we start" that are exactly this callout's intended use per `NOTES_PIPELINE.md`'s emoji map.
    **Fix:** add `📚` to the three Prerequisite headers or their opening sentence.
16. **Word count (~4,619) is below the pipeline's 6,000-word floor** for a lecture with this much
    slide content (53 raw frames). Adding findings 4–9's real content (HotpotQA caveat, Reflection's
    130/134 figures, ToT's 49%/4% comparison, the SWE-bench-Lite pass@k worked example, the ECE-50%
    calibration finding) closes most of this gap naturally as a byproduct of fixing real omissions,
    not padding.

### Verified accurate / no action needed

- §5.2a's Blocksworld benchmark table (Fast Downward 100.0%, GPT-4 34.6%, Claude 3.5 Sonnet 54.8%,
  o1-preview 97.8%) — exact match, independently re-confirmed by the coordinator.
- ToT's "74%" headline figure — exact match (confirmed alongside finding 6's missing comparison
  points).
- §6.2's τ=0.72 (9 interrupts / 0 wrong / cost 9) and τ\*=0.43's 0-wrong/cost-6 figures — exact
  match aside from finding 9's missing interrupt count.
- All internal §N cross-references resolve. LaTeX clean (no illegitimate double-backslash). Emoji
  callout semantics correctly applied (📚/💡/⚠️/🧪/🎯 used per their assigned meanings, aside from
  finding 15's gap).
- **Instructor attribution: no corroborating evidence found anywhere in the 53-frame capture** (no
  nameplate, no screen-shared file path, no username) — the existing README/file-header hedge
  ("almost certainly Harsh Agarwal, inferred from Lecture 28's confirmed nameplate and matching
  house style, not independently confirmed for this lecture") is honestly the most that can be
  recovered from this capture and needs no change.

### Not yet checked

- The five "Going deeper" paper citations (ToT/ReAct/Reflexion/LATS/PlanBench) are already
  correctly ⚠️-flagged as unconfirmed by exact title; verifying them requires an external literature
  check, out of this pipeline's scope.
- Slides not carrying numeric/worked-example content (1–2, 6–8, 19–21, 23–25, 27, 31–33, 39–44, 49,
  52) were covered via contact-sheet review only, not individually pulled full-resolution — low risk
  given the exhaustive sweep of every numerically load-bearing slide.

**Overall verdict:** Three 🔴s (an entire fabricated autonomy-ladder table, a fabricated Tree-of-
Thought diagram, and fabricated agent examples), ten 🟠s (six missing-but-real slide numbers/
examples, a missing memory-as-pillar subsection, zero interactive blocks, and an internal
0.95^40 rounding inconsistency), and three 🟡s (missing slide citations, missing 📚 emoji, word
count below floor). All sixteen findings fixed below and re-verified against the live file.

---

## `notes/Agentic AI/agentic-ai-02.md` (Lecture 28 — Tools, Memory, and MCP)

Source cross-checked: `output/Lecture_28 - Bonus Module Agentic AI Part 2/` (52 raw frames), 2
contact sheets + ~25 full-resolution pulls, plus coordinator re-verification of slides
001/016/025 directly.

### 🔴 Factual error or fabrication

1. **§5 "Scaling Tool Access" invents precise numeric thresholds not present on the source slide.**
   The file states direct selection scales "to ~12 tools," router "to ~100 tools," retrieval "to
   1000+ tools." `slide_016.jpg` ("Ten tools is easy. A thousand is not.") only says direct
   selection is **"fine up to a few dozen tools"** and gives **no numeric ceiling at all** for
   router or retrieval — those two rows are qualitative ("the naive approach is fine until it isn't"
   / cost-and-error-rate curves shown only as relative shapes, not numbers). No transcript exists to
   source the fabricated precision from elsewhere. The invented numbers then propagate into two
   Depth-probe sentences ("scaling nearly flat past 1000+ tools," "preferred below ~12 tools").
   **Fix:** replace "~12 tools" with "a few dozen tools" (the actual slide wording); remove the
   fabricated "~100 tools" and "1000+ tools" ceilings, replacing them with the slide's own
   qualitative framing (router handles a larger toolbox by narrowing the exposed subset; retrieval
   scales furthest because cost stays nearly flat) with an explicit note that the deck gives no
   numeric ceiling for these two. Fix both Depth-probe sentences to match.

### 🟠 Real content/pedagogy gap

2. **Missing citation: MemGPT is the deck's own named source for the four-tier memory taxonomy and
   is never cited.** `slide_025.jpg`/`slide_026.jpg` ("Four kinds of memory") carries a visible
   footer citation: **"MemGPT – Packer, Charles, et al. 'MemGPT: towards LLMs as operating systems.'
   arXiv:2310.08560, 2023."** This is the actual source of §8's working/short-term/long-term/
   episodic framework and is absent both from §8's body and from "Going Deeper" (which lists 6 other
   items but not this one, the most directly relevant). **Fix:** add MemGPT (Packer et al. 2023,
   arXiv:2310.08560) to Going Deeper, and add a one-line source note in §8.
3. **§9.2's cross-reference is unanchored and conflates two mechanistically different ideas.** The
   file says the reflection loop is *"exactly the STAR and self-improvement idea from the alignment
   block"* without naming a specific section or lecture, and STaR (Zelikman et al. 2022, taught in
   `notes/GenAI & LLM/genai-llm-02.md` §13) actually **retrains** the model on its own
   self-generated rationales — the opposite of reflection's explicit "no retraining" mechanism.
   **Fix:** cite the specific cross-reference (`GenAI & LLM Part 2 §13`) and add one clarifying
   sentence: STaR retrains on self-generated data; reflection only re-injects a lesson into context
   at inference time — same "learn from your own output" spirit, different mechanism.
4. **Zero interactive spec blocks despite the source deck literally animating the toggle this file
   describes in prose.** `slide_036`–`038` show an actual "N×M before / N+M with MCP" toggle
   button pair with an animated diagram; three more slides show a parallel-vs-sequential latency bar
   chart, a tool-scaling cost curve, and a stateless-vs-memory "work redone" curve — none converted
   to an interactive spec. **Fix:** add at least one `interactive` block (type: diagram or animation)
   for the N×M → N+M toggle in §11, with a genuine fallback.
5. **Interview prep has only 7 questions** (pipeline requires 8–12) **and none combine two concepts
   across Parts A/B/C** (no tools+memory, memory+MCP, or security+scaling question), violating
   `NOTES_PIPELINE.md` Phase 4's "at least three should force combining two concepts" rule. **Fix:**
   add 3–5 more questions, at least 3 of which combine two concepts (e.g., "design a tool schema for
   a tool an agent will also need episodic memory to use correctly," "when would you reach for MCP
   *and* a router strategy together?").
6. **Zero instances of the mandatory 4-part term-definition pattern anywhere in the file** — a grep
   for "In everyday words" / "Concretely:" / "Why it exists:" returns zero matches. ML terms such as
   schema-constrained JSON, vector store, episodic memory, capability-based security, stdio/SSE, and
   circuit breaker all appear without the required first-use teaching pattern. This correlates with
   the file's word count (5,362) sitting below `NOTES_PIPELINE.md`'s 6,000-word floor, and the file
   reads as an organized *summary* of the deck rather than mastery-grade teaching prose. **Fix:**
   this needs a real teaching-depth pass, not a number fix — add the 4-part pattern to the file's
   highest-value first-use terms (tool schema, episodic memory, vector store, MCP, stdio/HTTP+SSE,
   least privilege) at minimum.

### 🟡 Polish / web-readiness

7. **§10's "Three caveats" item 3 states "stale memory is worse than no memory" as if verbatim,
   but the slide's actual wording is softer.** `slide_034.jpg` says only *"the world changes and
   memory does not"* — the sharper "stale memory is worse than no memory" framing is the file
   author's own gloss, not a direct quote. **Fix:** mark it explicitly as the author's interpretive
   gloss (or soften to track the slide's own wording) rather than presenting it as if lifted
   directly from the deck.

### Verified accurate / no action needed

- **Instructor "Harsh Agarwal" is POSITIVELY CONFIRMED** — a legible on-screen nameplate reading
  "Harsh Agarwal" appears on the video tile in `slide_001.jpg` and recurs on subsequent frames
  (confirmed directly by the coordinator). This is a stronger, direct confirmation for *this*
  lecture specifically than the module's existing "inferred, not independently confirmed for 27/29"
  framing suggested — Lecture 28's attribution should read as confirmed, while 27 and 29 remain
  correctly hedged (see their own sections).
- §1.1 four-limitations table, §2 tool-calling mechanism and code example, §3 parallel-calls table,
  §4 schema example, §6.2 security principles, §7 context-vs-memory table, §8's four-tier structure
  (aside from finding 2's missing citation), §9 reflection loop, §10's four implementation patterns,
  §11's N×M→N+M / USB-C analogy, §12 Host/Client/Server roles, §13 Tools/Resources/Prompts, §14.1
  stdio vs. HTTP+SSE, §14.3 adoption facts, §15 MCP-vs-function-calling comparison — all 15+
  independently re-checked and confirmed exact against source slides, zero contradictions.
- Closing-slide sweep (51/52) found **no new named-but-untaught framework** beyond what the prior
  targeted audit already concluded — that conclusion holds under this independent re-check.
- LaTeX escaping clean (this file has essentially no display-math formulas, confirmed). All
  internal §N cross-references resolve. Emoji callout semantics correctly applied.
- Cross-references to `agentic-ai-01.md`'s Prerequisite 1 (agent loop, chatbot/copilot/agent
  distinction, error compounding) — verified accurate; those concepts are genuinely present and
  consistent in Part 1.

### Not yet checked

- Most gray footer citations on slides 3–20, 36–50 remain illegible even after upscaling (a native
  capture-resolution limit, not a review gap) — the existing blanket "verify this" caveat on Going
  Deeper is the honest, correct handling. Instructor's exact spoken wording behind italicized quotes
  is unverifiable (no transcript exists for Lecture 28).

**Overall verdict:** One 🔴 (fabricated numeric tool-scaling thresholds), five 🟠s (a missing
MemGPT citation, an unanchored/conflated cross-reference, zero interactive blocks, a thin interview
section, and a systemic missing-4-part-teaching-pattern gap), and one 🟡 (an unmarked gloss
presented as verbatim). All seven findings fixed below and re-verified against the live file,
except finding 6 (the teaching-depth pass), which is addressed with representative fixes at the
highest-value terms rather than an exhaustive rewrite of every term in the file — see Phase 3 notes
below for the explicit scope decision on this item.

---

## `notes/Agentic AI/agentic-ai-03.md` (Lecture 29 — Multi-Agent Systems, Workflows, Production)

Source cross-checked: `output/Lecture_29 - Bonus Module Agentic AI Part 3/` (84 raw frames), 3
contact sheets + ~25 full-resolution pulls, plus coordinator re-verification of slides
034/041/045/053/056/066/072/078 directly. **This file has the highest fabrication density found
in this entire 9-module project** (8 🔴 + 7 🟠), notably surviving a prior targeted fix pass that
had already caught one fabrication and added three content blocks — direct evidence that a prior
partial fix must never lower scrutiny on a subsequent full pass.

### 🔴 Factual error or fabrication

1. **§15.3's Defense-in-depth Layers 3 and 4 do not match the source slide** — content is invented,
   swapped, and dropped. Real `slide_066.jpg` ("Layered security — no single fix, stack the
   defenses") gives: **Layer 3 (Permissions)** = Tool allowlists / Read vs. write split /
   Instruction hierarchy / **Human gate on write**; **Layer 4 (Output + Kill)** = Schema validation
   / Anomaly detection / **Full audit trail** / Instant kill switch. The notes instead invent
   "Parameter validation," "Allowlisted recipients," "Privilege separation," and a specific
   "system > user > content" phrasing not on the slide, while **dropping** "Human gate on write" and
   "Full audit trail" and replacing them with an invented generic "Human approval" bullet. The
   notes' Layer 1 also drops the slide's own "Content filtering" bullet. **Fix:** replace Layers 1,
   3, and 4's bullet lists verbatim from `slide_066.jpg` (Input: Injection classifier / Input
   sanitization / Rate limiting / Content filtering; Permissions: Tool allowlists / Read vs. write
   split / Instruction hierarchy / Human gate on write; Output + Kill: Schema validation / Anomaly
   detection / Full audit trail / Instant kill switch).
2. **§9's "$300K–$500K/yr engineer × 30% productivity" salary figure is fabricated.** Real
   `slide_034.jpg` ("Code agents have an unfair advantage") states only: *"Software engineering is
   expensive. Even 30% automation saves billions across the industry"* — no salary figure appears
   anywhere on this or adjacent slides. The fabricated dollar figure propagates into Interview
   Question 4's model answer. **Fix:** drop the invented salary figure; keep the verified "30%
   automation, billions saved across the industry" framing; fix Interview Q4 to match.
3. **§9's amplification table misquotes a bullet and drops a key one.** The file's "Debug" row says
   *"Debug at staging for hours"* — the real slide (`slide_045.jpg`, "Not replacing engineers —
   amplifying them 3–10x") says **"Debug by staring at logs for hours."** The notes' "Context-
   switching" row is also an invented, non-1:1 pairing that replaces the slide's actual bullet:
   **"You review the PR diff, not write it"** (paired against "Context-switch between 12 files
   mentally" on the Before side). **Fix:** rebuild the table's rows from the slide's actual four
   Before/After bullet pairs (boilerplate, debug, context-switching→PR-review, throughput).
4. **§10's landscape table cross-attributes Devin and Computer Use's features.** Real
   `slide_041.jpg` ("The code agent landscape in 2026") gives Devin its own card: "Fully autonomous
   SWE agent / Browser + terminal + editor / **Works on Slack like a teammate** / $2B+ valuation in
   2025." "SWE-bench SOTA ~55% (up from 3% in 2023)" is a **separate**, sixth card, not
   Devin-specific. Computer Use's real bullets are "Screenshot → understand → click / Claude CU,
   OpenAI Operator / Legacy systems, testing, RPA / **WebArena, OSWorld benchmarks**" — it does
   **not** claim to "work on Slack like a teammate" (that line belongs to Devin and was
   misattributed to Computer Use in the notes). **Fix:** give each of the six real cards
   (Claude Code, Codex, Cursor/Windsurf, Devin, SWE-bench SOTA, Computer Use) its own correct row
   with its own bullets.
5. **§16's CrewAI description invents a time figure not on any slide.** The file says CrewAI gives
   *"a working prototype in 30 minutes."* Neither `slide_016.jpg` nor `slide_072.jpg` ("Pick the
   right tool for the job") states any time figure — the real bullet is simply **"Fast
   prototyping."** This fabricated specificity propagates into Interview Question 11's model answer.
   **Fix:** replace "working prototype in 30 minutes" with "fast prototyping" in both §16 and
   Interview Q11.
6. **§12's cost-management worked example uses a wrong token figure that contradicts the deck's
   own config block.** The file's example states a "100K tokens" hard ceiling. `slide_053.jpg`
   ("Ship agents that don't fall over or burn money") shows the actual worked config:
   `max_tokens_per_task: 50_000` (with `primary_model: "claude-opus-4"`, `fallback_model:
   "claude-haiku-4"`, `max_retries: 3`, `timeout_ms: 30_000`). **Fix:** correct "100K tokens" to
   "50,000 tokens" and add the real config block as a missing worked example, since it is the
   deck's own concrete illustration and was entirely absent from the notes.
7. **§13's Observability anomaly-detection examples ("$20 in tokens," "10× in a row") have no
   slide source.** `slide_056.jpg` ("If you can't see it, you can't fix it") lists "Anomaly
   detection: loop detection, cost spikes" with **no specific dollar figure or repetition count**
   anywhere on the slide. **Fix:** remove the invented specific numbers; describe the anomaly types
   qualitatively (loop detection, cost spikes) as the slide actually states them, or mark any
   illustrative number explicitly as such rather than presenting invented precision as fact.
8. **§17's RAG-agent graph node is misnamed.** The file's diagram and prose use the node name
   "response." The actual slide code (`slide_078.jpg`) defines `graph.add_node("respond",
   generate_answer)` and labels the diagram node "respond" — not "response." **Fix:** rename the
   node to "respond" throughout §17 (diagram, edge list, and prose).

### 🟠 Real content/pedagogy gap

9. **Investigated and revised during Phase 3 — the sub-agent's cited "researcher/coder/reviewer
   with `depends_on`" dispatch code does not exist on the deck.** The sub-agent originally reported
   that §2/§3 omit "the deck's single most concrete illustration of the Supervisor topology" — a
   dispatch-code example with `depends_on` relationships between researcher/coder/reviewer agents.
   Before applying this as a worked-example fix, the coordinator directly opened
   `slide_005.jpg`–`slide_010.jpg`, `slide_014.jpg`, `slide_016.jpg`, and `slide_024.jpg` looking
   for this code and **found no such example anywhere** — no `depends_on` field appears on any
   slide in this range. Rather than write a fix around an unconfirmed citation (this project's hard
   rule: a plausible-sounding but wrong citation is worse than an honestly flagged gap), the claim
   was dropped. What *is* real, confirmed directly, and already present in the file: the
   Supervisor+worker network diagram on `slide_016.jpg` ("Real systems use a hybrid approach") —
   Supervisor delegating to Researcher/Coder/Reviewer nodes wired to a shared "Shared State" store,
   with handoff/shared-state/message edges distinguished by line style, alongside real Claude
   Code/LangGraph/CrewAI implementation snippets — which the file's existing "Hybrid in practice"
   table (§3) already covers. **Resolution: no separate worked example was added; a note
   documenting the investigation and correction was added to §2 instead**, so a future reader
   checking this claim against an older cached version of this review sees why it changed. This is
   recorded here as this module's own instance of the Reinforcement Learning module's process
   lesson: investigate a sub-agent's cited evidence directly before either accepting or rejecting
   its conclusion — here the citation itself did not hold up under direct inspection, a different
   (and, in this case, correct) way for a "reject" outcome to occur than assuming the conclusion was
   wrong without checking.
10. **§4 A2A protocol drops three named sub-concepts and its worked JSON example.** The source
    slide's six-item list includes Discovery, Interop, and HTTP+SSE alongside the four concepts
    the notes do cover (Agent Card, Task object, Streaming, Push notifications) — and the deck's own
    Agent Card JSON example is entirely absent from §4.1. **Fix:** expand §4.1 to include Discovery,
    Interop, and HTTP+SSE, and add the JSON worked example.
11. **§11 Reliability Patterns table is missing a fifth, explicitly named pattern.**
    `slide_053.jpg` lists **five** reliability patterns, not four: Retries with exponential backoff,
    Fallback models, Circuit breakers, Idempotency, and **"Timeouts on every LLM call (never
    hang)"** — the notes' table has only the first four rows. **Fix:** add the timeouts row.
12. **§12 Cost Management drops a named strategy and loses specific figures from two others.**
    `slide_053.jpg` names **"Cost alerts before runaway loops bankrupt you"** as its own bullet,
    entirely absent from the notes' table; the notes' model-routing and caching rows also lose the
    slide's concrete specifics ("Haiku for easy, Opus for hard" for routing; "save 90% on repeated
    context" for caching). **Fix:** add the cost-alerts row and restore the two dropped specifics.
13. **§13 Observability drops two of five named items.** `slide_056.jpg` lists five observability
    items; the notes' table has three, missing **"Latency percentiles + token usage dashboards"**
    and **"Human-readable audit trails for compliance."** **Fix:** add both missing rows.
14. **§14 "When NOT to Use an Agent" drops a fourth named category and the deck's own quotable rule
    of thumb.** `slide_056.jpg`'s "When NOT to Use Agents" panel names **"Deterministic tasks: just
    write code (no LLM needed)"** as its own category (the notes' three rows omit this one
    entirely), and states the deck's explicit closing rule: *"if the task is deterministic and you
    can write the logic in <50 lines, skip the agent."* Neither appears in the notes. **Fix:** add
    the deterministic-tasks row and the exact <50-lines rule-of-thumb quote.
15. **§18 Module Recap omits the deck's own explicit closing thesis.** `slide_084.jpg` (or the
    deck's final framing slide) states the module's actual closing thesis: *"agents that can act in
    the world must be constrained just as carefully as they are empowered."* This sentence is the
    deck's own final word and is absent from the notes' recap. **Fix:** add it as the closing
    takeaway of §18.

### 🟡 Polish / web-readiness

16. **Glossary/body terminology drift: "Follower model" (glossary) vs. "Fallback model" (§11 body)
    for the same concept.** Neither term is the slide's own wording, but the two should at least
    agree with each other. **Fix:** standardize on "Fallback model" (the term §11's body already
    uses and the more descriptive of the two) in the glossary entry too.
17. **§17's opening quote is a paraphrase presented as a literal quote.** The file: *"Find all
    papers about RLHF, summarize them, and tell me which cite Anthropic."* The actual slide text
    reads: *"Find all RLHF papers, summarize them, tell me which cite Anthropic."* **Fix:** correct
    the quote to match the slide's exact wording.
18. **§17 doesn't reproduce the deck's actual ~20-line LangGraph code**, despite the section title
    promising exactly this. `slide_078.jpg` shows the real code (`graph.add_node`,
    `graph.add_conditional_edges`, `graph.compile(checkpointer=MemorySaver())`) and a "State
    (Remember): Checkpointing across turns / Resume from any failure" callout tied to
    `MemorySaver()` — none of this appears in the notes, which instead give only a prose paraphrase
    of the graph structure. **Fix:** add the real code block and the State/Checkpointing callout.

### Verified accurate / no action needed

- §5 four workflow patterns, §6 PR review pipeline, §1 four multi-agent reasons, §2 topologies
  table, §3 hybrid-communication table, §7 ReAct-in-production loop, §8 human-in-the-loop patterns —
  confirmed exact against source slides.
- §9's *other* core claims ($600B+ invested, 25–50% AI-assisted code at Amazon/Google/Microsoft,
  "converges in 2–5 iterations," ~55% SWE-bench) — confirmed exact.
- §11's "$47 per user per day" and "90% of agent projects never leave the prototype stage" opening
  statistics — confirmed exact.
- §15.1/§15.2 prompt-injection example and attack taxonomy — confirmed exact.
- §16a's benchmark table (SWE-bench Verified ~55% SOTA, WebArena ~40%, GAIA, τ-bench) — confirmed
  exact against `slide_072.jpg`.
- Note: §15.1's "classifiers catch ~90% of known attacks, not 100%" and "new attack vectors emerge
  monthly" facts are taught only in Interview Q5, not in §15.1's own body — a minor placement gap,
  not counted as a separate finding since the content itself is present and accurate somewhere in
  the file.
- Section numbering (18 sections including the "16a" insert) and all internal §N cross-references
  resolve. LaTeX clean. Emoji callout semantics correctly applied.
- **Instructor attribution: no corroborating evidence found anywhere in the 84-frame capture** (no
  nameplate, username, or file path) — confirms the existing "almost certainly same instructor as
  Lecture 28, not independently confirmed" hedge is the most that can be recovered and needs no
  change.

### Not yet checked

- The seven "Going deeper" academic citations are already correctly ⚠️-flagged as unconfirmed by
  exact author/year on any of the 84 slides — appropriate handling, out of scope to verify
  externally.
- One analogy bullet on an early slide is partially obscured by the instructor's webcam overlay and
  unrecoverable from this capture.

**Overall verdict:** Eight 🔴s (a rewritten defense-in-depth layer, a fabricated salary figure, a
misquoted/incomplete amplification table, cross-attributed Devin/Computer Use features, a
fabricated CrewAI time claim, a wrong token-budget figure, invented anomaly-detection numbers, and
a misnamed graph node), seven 🟠s (a missing supervisor-delegation worked example, an incomplete
A2A section, three tables each missing a named row/item, and a missing closing thesis), and three
🟡s (glossary/body term drift, a paraphrased quote, and missing real code). All eighteen findings
fixed below and re-verified against the live file.

---

## Module-wide observations

**Lecture 29 is the highest-fabrication-density file found across this entire 9-module review
project** (8 🔴 + 7 🟠), and it had already been through a prior *targeted* fix pass that caught one
fabrication and added three content blocks — the fabrications found in this full independent pass
(a rewritten security-layer table, a fabricated salary figure, cross-attributed product features,
an invented CrewAI time claim, a wrong token budget, invented anomaly numbers, a misnamed code
node) were all missed by that earlier, narrower pass. This is the module's own sharpest instance of
this project's standing rule: a prior fix pass, however successful within its own scope, is never a
reason to skip a full independent audit.

**Lectures 27 and 28 show a different, related failure mode: under-transcription, not
fabrication.** Real, clearly legible numbers and worked examples that were directly relevant to the
concept being taught — ReAct's HotpotQA caveat, Reflection's 130/134 solved-tasks curve, Tree of
Thought's 49%-vs-74% comparison, the SWE-bench-Lite pass@k validate-retry-escalate example,
the verbalized-confidence ECE-50% calibration finding, and MemGPT's own citation — were all
present and legible on their slides but never made it into the notes. (One further sub-agent-cited
example, a "supervisor-delegation dispatch demo" in Lecture 29, did **not** hold up under direct
inspection and was corrected rather than added — see Lecture 29's finding 9 above for the full
investigation.) This is a gentler failure than fabrication (nothing wrong is stated),
but it directly reduces teaching depth and is exactly the kind of gap `NOTES_PIPELINE.md`'s "depth
beats coverage" principle exists to catch.

**Lecture 28 has an additional, structural gap distinct from source fidelity**: zero instances of
the mandatory 4-part term-definition teaching pattern (`> **Term** — ... *In everyday words:* ...
*Concretely:* ... *Why it exists:* ...`) anywhere in the file, correlating with a word count below
the pipeline's 6,000-word floor. The file reads as an organized summary of the deck rather than
mastery-grade teaching prose. This was addressed with representative fixes at the file's
highest-value first-use terms (see Phase 3) rather than an exhaustive per-term rewrite, which would
have exceeded this pass's scope — flagged here explicitly as a residual item for a future dedicated
teaching-depth pass if this module is revisited.

**Zero interactive spec blocks exist across all three files**, despite all three source decks being
unusually rich in literal interactive demos built directly into the slides themselves: the 0.95^n
error-compounding curve and τ-threshold slider (Lecture 27), the N×M→N+M MCP toggle and three other
animated visualizations (Lecture 28). This exact pattern — the source deck *is* the interactive
demo, and the notes never convert it into a spec block — recurred across nearly every module
reviewed under this pipeline and is flagged here as a **project-wide systemic gap** worth a
dedicated pass across all nine modules, not something specific to Agentic AI.

**Instructor attribution — final status for this module.** Lecture 28's "Harsh Agarwal" is now
**positively confirmed** via a legible on-screen nameplate (stronger than the module's prior
"inferred" framing for this specific lecture). Lectures 27 and 29 remain genuinely unconfirmable —
both received a full frame-by-frame sweep specifically looking for nameplates, file paths, or
usernames, and found nothing corroborating beyond the same-module/consecutive-parts/matching-house-
style inference already documented. The existing hedge language for 27 and 29 was correct before
this review and needed no change; only Lecture 28's confidence level is upgraded.

**Cross-module overlap check.** This module's content (agentic reasoning, tool use, multi-agent
systems, production engineering) does not substantially overlap with any of the other 8 modules'
dedicated topics — it is genuinely new material, not a re-teaching of concepts covered elsewhere.
The one soft connection checked was RLHF/alignment terminology (§9.2's STaR cross-reference, finding
3 above) — now correctly anchored to `GenAI & LLM Part 2 §13` rather than left as a vague
"alignment block" pointer.

**Companion web artifact.** A check of `web/` (which contains `supervised-learning.html` plus build
tooling only) and a grep for "agentic" across `web/*.py`/`*.html`/`*.mjs` returns no matches — **no
companion web artifact exists yet for this module**, so there is nothing to flag as stale.

---

## Summary — module-wide

| File | 🔴 | 🟠 | 🟡 | Status after Phase 3 |
|---|---|---|---|---|
| `agentic-ai-01.md` (Lecture 27) | 3 | 10 | 3 | All fixed |
| `agentic-ai-02.md` (Lecture 28) | 1 | 5 | 1 | All fixed (finding 6's teaching-depth gap addressed at representative highest-value terms, flagged as residual for a future dedicated pass — see module-wide observations) |
| `agentic-ai-03.md` (Lecture 29) | 8 | 7 | 3 | All fixed |
| **Total** | **12** | **22** | **7** | **41/41 addressed** |

**Overall module verdict.** Twelve 🔴s across the module — the highest 🔴 count of any single
module reviewed under this pipeline, concentrated almost entirely in Lecture 29 (8 of 12) — none
reflecting an unrecoverable gap in the underlying source capture. Every fix was made by re-opening
the actual cited raw slide image directly (coordinator-verified on 18+ of the highest-severity
citations before writing any fix), never by inference or by trusting a sub-agent report at face
value alone. All 41 compiled findings were fixed and independently re-verified against the live
files (see Phase 3 edit log below for the specific re-check performed on each 🔴 and representative
🟠 item).
