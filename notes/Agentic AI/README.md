# Agentic AI

> **Bonus module — ML Summer School 2026.** Three lectures on turning language models into
> agents that act, coordinate, and ship to production. Instructor: **Harsh Agarwal**, Applied
> Scientist, International ML team at Amazon — **positively confirmed** via a legible on-screen
> nameplate on Lecture 28 (`slide_001.jpg`'s video tile). Lectures 27 and 29 have no visible name
> overlay anywhere in their raw frame captures (both were swept in full for a nameplate, file
> path, or username, per `QUALITY_REVIEW.md`) and remain an inference — almost certainly the same
> instructor (same module, consecutive parts, matching slide house style) but not independently
> confirmed for those two lectures specifically.
>
> ✅ **This module has completed a full `QUALITY_REVIEW_PIPELINE.md` audit** — see
> [`QUALITY_REVIEW.md`](QUALITY_REVIEW.md) for the complete findings and fixes (41 findings across
> all three lectures, all applied and re-verified against the live files).

---

## Index

| # | File | Lecture | Topic | Slides | Words |
|---|------|---------|-------|--------|-------|
| 01 | [agentic-ai-01.md](agentic-ai-01.md) | 27 | Fundamentals, Architectures, Reasoning | 23 | ~6,100 |
| 02 | [agentic-ai-02.md](agentic-ai-02.md) | 28 | Tools, Memory, and MCP | 21 | ~7,200 |
| 03 | [agentic-ai-03.md](agentic-ai-03.md) | 29 | Multi-Agent Systems, Workflows, Production | 35 | ~7,700 |
| | | | **Total** | **79** | **~21,000** |

---

## Key takeaway per lecture

- **Part 1:** An agent is a language model that closes the loop — Think → Act → Observe —
  and the ReAct framework prevents hallucination by grounding every action in reasoning +
  real-world feedback. Error compounding (0.95^n) makes guardrails more important than raw
  capability.

- **Part 2:** Three capabilities turn a passive LLM into an active agent: tools to act (schema
  design is prompt engineering), memory to persist (episodic memory + reflection = learning
  without retraining), and MCP to connect (N×M → N+M, like USB-C for AI tools).

- **Part 3:** Single agents become orchestrated teams via topologies (supervisor, peer-to-peer,
  hierarchical) and workflows (sequential, parallel, conditional, loop). Code agents have an
  unfair advantage (verifiable output, rich tooling) — the edit-test-fix loop converges in
  2–5 iterations. Production requires reliability (backoff, circuit breakers), cost management
  (token budgets, model routing), observability (trajectory logging), and defense-in-depth
  against prompt injection.

---

## What's in Part 1

| Section | Topic |
|---------|-------|
| §1 | What Is an Agent — chatbot → copilot → agent spectrum |
| §2 | The Autonomy Ladder — L1 tool call → L2 multi-step (ReAct) → L3 plan & adapt → L4 multi-agent, with the deck's own worked task showing exactly where the "do I need an agent?" threshold sits |
| §3 | Error Compounding — the 0.95^n problem |
| §4 | When to Use an Agent — dynamic path + tool use threshold |
| §5 | ReAct, Reflection, **Plan-and-Execute** (with the Blocksworld benchmark), Tree of Thought / LATS |
| §6 | Guardrails — confidence thresholds, validate-retry-escalate |
| §7 | Putting it together |

## What's in Part 2

| Section | Topic |
|---------|-------|
| §1 | Why LLMs Need Tools — four limitations |
| §2 | How Tool Calling Works — declare, select, execute, return |
| §3 | Parallel Tool Calls — independence detection |
| §4 | Tool Schema Design — schema = prompt |
| §5 | Scaling Tool Access — direct, router, retrieval |
| §6 | Tool Failure Handling and Security |
| §7 | Why Memory Matters — context ≠ memory |
| §8 | Four Tiers of Memory — working, short-term, long-term, episodic |
| §9 | Episodic Memory and Reflection |
| §10 | Memory Implementation Patterns |
| §11 | The Problem MCP Solves |
| §12 | MCP Architecture — host, client, server |
| §13 | Three MCP Capabilities — tools, resources, prompts |
| §14 | MCP Transports and Ecosystem |
| §15 | MCP vs. Function Calling |
| §16 | Module Recap |

## What's in Part 3

| Section | Topic |
|---------|-------|
| §1 | Why Multiple Agents — specialization, parallelism, robustness, scalability |
| §2 | Multi-Agent Topologies — supervisor, peer-to-peer, hierarchical |
| §3 | Communication Patterns — message passing, blackboard, handoff |
| §4 | A2A Protocol (Google) |
| §5 | Four Workflow Patterns — sequential, parallel, conditional, loop |
| §6 | PR Review Pipeline — chaining all four patterns |
| §7 | ReAct Framework in Production |
| §8 | Human-in-the-Loop Patterns |
| §9 | Why Code Agents Have an Unfair Advantage |
| §10 | The Code Agent Landscape in 2026 |
| §11 | Reliability Patterns |
| §12 | Cost Management |
| §13 | Observability |
| §14 | When NOT to Use an Agent |
| §15 | Safety and Guardrails — prompt injection, defense-in-depth |
| §16 | Framework Comparison + **Evaluation Benchmarks (SWE-bench/WebArena/GAIA/τ-bench) & Where Agents Are Heading** |
| §17 | Building a RAG Agent in 20 Lines |
| §18 | Module Recap |

---

## Reading guide

**If you're brand new to agents:** Start with Part 1 (§1–§4) to understand what agents are
and when to use them. Then read Part 2 §1–§6 for tools, §7–§9 for memory.

**If you want to build agents:** Read all of Part 2 (tools + memory + MCP), then Part 3
§5–§7 (workflow patterns + ReAct) and §17 (20-line RAG agent).

**If you're preparing for interviews:** Read the interview prep in all three parts. Part 3's
questions cover production systems, which are the most commonly asked.

**Full read time:** ~3.5–4 hours for all three parts (~21,000 words).

---

## Capture quality

| Lecture | Raw frames (`output/`) | Deduped slides | OCR + transcript draft | Raw-frame audit (targeted, pre-pipeline) | `QUALITY_REVIEW_PIPELINE.md` full audit |
|---------|-----------|---------------|-----------------|-------------|-------------|
| 27 (Part 1) | 53 | 23 | ✅ | ✅ — found & added Plan-and-Execute (a whole missing architecture) and LATS | ✅ — 3 🔴, 10 🟠, 3 🟡 found and fixed (fabricated autonomy-ladder table, fabricated ToT diagram, fabricated agent examples, plus 10 real-but-untranscribed slide numbers/examples) |
| 28 (Part 2) | 52 | 21 | ✅ | ✅ — no missing framework found | ✅ — 1 🔴, 5 🟠, 1 🟡 found and fixed (fabricated tool-scaling thresholds, missing MemGPT citation, systemic missing-4-part-teaching-pattern gap) |
| 29 (Part 3) | 84 | 35 | ✅ | ✅ — found & fixed one fabricated detail, added 3 missing content blocks | ✅ — 8 🔴, 7 🟠, 3 🟡 found and fixed (the highest fabrication density found across this entire 9-module review project — a rewritten security-layer table, a fabricated salary figure, cross-attributed product features, a fabricated CrewAI time claim, a wrong token budget, invented anomaly numbers, a misnamed code node, and more) |
| **Total** | **189** | **79** | **✅** | **✅** | **✅ — 41/41 findings fixed, see [`QUALITY_REVIEW.md`](QUALITY_REVIEW.md)** |

> ⚠️ Earlier drafts of this README reported raw-frame counts of 30/28/45 — those numbers came from
> a different, coarser capture than this project's actual `output/` directory and understated the
> real raw capture by roughly 45%. The table above reflects the true `output/` frame counts (see
> project memory `slides-deduped-is-lossy` for why this matters — dedup and coarse sampling both
> lose real content).

All three files were originally drafted from `slides_deduped/` with OCR (EasyOCR) cross-referenced
against the YouTube transcript, then audited twice: once in a targeted, pre-pipeline pass (see each
file's own header note), and again as a full independent three-lens `QUALITY_REVIEW_PIPELINE.md`
audit (see [`QUALITY_REVIEW.md`](QUALITY_REVIEW.md)) — the second pass found substantially more
issues than the first, including in Lecture 29, which had already been through the targeted pass.
This is recorded there as this project's clearest evidence that a prior fix pass, however
successful within its own narrower scope, is never a substitute for a full independent audit.
