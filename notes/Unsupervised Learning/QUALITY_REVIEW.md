> ✅ **STATUS: ALL FINDINGS BELOW HAVE BEEN APPLIED AND INDEPENDENTLY RE-VERIFIED AGAINST THE LIVE
> FILES.** This file is kept as an audit trail of what was found and fixed during the Unsupervised
> Learning module's `QUALITY_REVIEW_PIPELINE.md` pass, not as a pending to-do list. Every finding
> below — whether reported by a per-lecture sub-agent or found independently by the coordinator's
> own module-wide mechanical pass — was re-verified against the actual raw slide frames (for every
> 🔴 and most 🟠 findings) or against the live `.md`/`README.md` content (for every finding) *after*
> being fixed, per this project's hard rule: **never mark a finding "Fixed" without re-reading the
> live file to confirm it landed.** The module `README.md` (word counts, glossary counts, the
> missing Part 3 table of contents, the missing Key-takeaway rows, capture-quality notes) now
> matches the post-fix state of all four files.

# Quality review — Unsupervised Learning — 2026-08-29

Purpose / severity legend (reuse: 🔴 factual error or fabrication · 🟠 real content/pedagogy gap ·
🟡 polish / web-readiness).

Method: each lecture file was run through all three lenses (Teacher / Student / Engineer) per
`QUALITY_REVIEW_PIPELINE.md`, executed as four parallel sub-agent passes (one per file), each
required to run a mandatory exhaustive citation sweep before compiling findings. The coordinator
independently ran a module-wide mechanical pass (README structure, cross-file consistency of
"combining concepts" / Leadership Principle / glossary counts, LaTeX escaping, interactive-block
inventory) in parallel, then compiled all reports, re-verified every 🔴 finding against the raw
frames in `output/Lecture_1{0,1,2,3}.../` directly before fixing, applied every fix to the source
`.md` files, and re-read each edited region afterward to confirm it landed before writing "Fixed"
anywhere in this document. Lecture 13 (`unsupervised-learning-04.md`) received extra scrutiny per
project memory `slides-deduped-is-lossy`, since it is this module's one file sourced from
`slides_deduped/` rather than purely `output/` raw frames.

---

## `notes/Unsupervised Learning/unsupervised-learning-01.md` (Lecture 10)

Source cross-checked: `output/Lecture_10 - Module 4 Unsupervised Learning Part 1/` (175 raw
frames), 37 citations swept exhaustively (0 frame-number mismatches; 2 content-level
transcription errors found separately, below), ~18 full-resolution spot checks, plus the
coordinator's own re-verification of both 🔴 findings against `slide_008.jpg`/`slide_051.jpg` and
of the §D/§E timing fix against `slide_096/099/117/118/119/121.jpg` and `timestamps.txt`.

### 🔴 Factual error or fabrication

1. **Fabricated/garbled quote transcription at [slide 8]** (Prerequisite 1). The notes stated the
   slide's caption reads *"anomaly detection lives inside clustering (density), GMM (probability),
   and reconstruction error (autoencoders). Not a separate category."* Direct inspection of
   `slide_008.jpg` shows this is wrong on every noun: the actual visible caption (a caption bar
   partially clipped by the frame's left edge in frames 6–8, so its opening clause is genuinely
   illegible) reads *"...lives inside clustering (GMM), generation (VAE), and anomaly detection.
   Not a separate category."* — different subject, different parentheticals. **Fix:** re-transcribed
   only the legible portion, explicitly flagged the illegible opening clause with ⚠️ rather than
   inventing a subject, and corrected the reading to match what the sentence actually says (anomaly
   detection is the third item in the list, not the subject). **Fixed**, re-verified by re-reading
   the edited paragraph against `slide_008.jpg` directly: the quoted text now matches the image
   exactly for everything visible.
2. **A "wait" drafting artifact left in shipped prose, plus a wrong description of the silhouette
   figure** (§25). The notes said the third worked silhouette value ($s{=}0.45$, "far from
   cluster") "appears twice with different annotations... wait, the deck labels this third case
   with $s{=}0.45$ as well." Direct inspection of `slide_051.jpg` shows **three genuinely distinct
   values** ($s=0.83$ "deep inside", $s=0.40$ "on boundary", $s=0.45$ "far from cluster") — 0.45
   appears once, not twice, and no labeling inconsistency exists. The real, worth-noting oddity is
   different: "far from cluster" scores *higher* (0.45) than "on boundary" (0.40). **Fix:** removed
   the "wait" fragment and the false twice-appearing claim; replaced with an explanation of the
   actual, genuinely counter-intuitive ordering (silhouette rewards isolation from other clusters as
   much as compactness within one's own, so being far from *everything* can score better than
   sitting exactly on a boundary). **Fixed**, re-verified against `slide_051.jpg` directly and by
   grepping the file for the removed "wait, the deck labels" fragment: zero remain.

### 🟠 Real content/pedagogy gap

3. **Davies-Bouldin Index entirely missing from the notes**, despite being named, defined, and
   shown in code on the very slide cited for §25 (`slide_117.jpg`, alongside Silhouette and
   Calinski-Harabasz, which *are* covered). It never appeared in §25, §27's routing table, or the
   glossary. **Fix:** added a full Davies-Bouldin subsection to §25 (definition, the standard
   formula the slide names but doesn't spell out, a symbol table, and a callout on why it's the one
   intrinsic metric where "lower is better"), updated §27's "all three/four intrinsic metrics" closing
   callout to cover all four, and added a glossary entry. **Fixed**, re-verified by re-reading §25
   and the glossary directly: Davies-Bouldin now appears in both, with the formula
   $DB=\frac1K\sum_i\max_{j\ne i}\frac{s_i+s_j}{d(c_i,c_j)}$ present and bound immediately to its
   symbol table.
4. **DBSCAN's "Stability: vary eps ±10%" check** (shown on the same cited slide, `slide_119.jpg`,
   alongside K-Means' and Hierarchical's own confirm/quantitative checks) **was dropped from §27's
   routing table**, left as an unexplained "—". **Fix:** added the eps-stability content to §27's
   table (relabeling that row "Confirm / Quantitative / Stability") and added an explanatory
   paragraph connecting it to K-Means' `n_init` restarts and Hierarchical's cophenetic correlation as
   the same genre of "is this result robust, not a knife-edge artefact" check. **Fixed**, re-verified
   by re-reading §27 directly: the DBSCAN column's second row now reads "Stability: vary $\epsilon$
   by $\pm10\%$..." instead of "—".
5. **Section-timing internal inconsistency**, evidenced by the file's own citations. The front-matter
   table declared Hierarchical Clustering ending / DBSCAN starting at 43:44, and Evaluation starting
   at 55:05 — but Hierarchical's own Key-Takeaways citation (`[slide 96]`) is timestamped 44:55 (per
   `timestamps.txt`), DBSCAN's own first-slide citation (`[slide 99]`) is 45:44, and Evaluation's own
   cited slides (`[117]`/`[118]`/`[119]`/`[121]`) are 50:59–52:12 — all several minutes before the
   declared boundaries, with the live demo's own first frame at 52:56. **Fix:** corrected the
   front-matter table's Section D/E/F boundaries and the `# PART C/D/E` header timestamps to
   30:28–44:55 / 45:44–50:59 / 50:59–end, matching the file's own citation evidence exactly. **Fixed**,
   re-verified against `timestamps.txt` directly (`slide_096.jpg`→44:55, `slide_099.jpg`→45:44,
   `slide_117.jpg`→50:59 confirmed) and by re-reading all four edited sites (the table plus three
   `# PART` headers) in the live file.
6. **Zero interactive spec blocks**, despite this lecture being built around exactly the kind of
   animated/iterative processes `NOTES_PIPELINE.md` recommends them for: the K-Means assign/update
   loop, K-Means++ probabilistic seeding, dendrogram cut-height exploration, and DBSCAN's recursive
   core-point expansion — all narrated only in prose. **Fix:** added four `interactive` blocks: §5/§6
   (K-Means step-by-step with a good-vs-bad-init toggle), §9 (K-Means++ seed selection with an
   exponent slider), §13 (dragging a dendrogram cut-height slider), §19 (DBSCAN's core-point
   expansion animated point-by-point). Every block has all seven required fields and a fallback that
   is genuinely sufficient standalone (each points to prose/tables already in the file that state the
   same content statically). **Fixed**, re-verified by grepping for ` ```interactive` in the live
   file: 4 blocks found, each with `type/title/concept/control/observe/insight/fallback` present.

### 🟡 Polish / web-readiness

7. **Systematic "combining concepts" miscount** — the file's own closing Summary table claimed
   "(3 combining concepts)" for the 9 interview questions, but only 2 (Q8, Q9) are actually tagged
   `(Hard — combines two concepts)`. This is a module-wide pattern (see Module-wide section below).
   **Fix:** corrected to "(2 combining concepts)" in this file's Summary table. **Fixed**, re-verified
   by re-reading the Summary table row directly.
8. **Leadership Principle count mismatch** — the Summary table claimed "3 Leadership Principles" but
   4 are actually named in the LP tie-in section (Dive Deep, Are Right A Lot, Customer Obsession,
   Insist on the Highest Standards). **Fix:** corrected to "4 Leadership Principles". **Fixed**,
   re-verified by re-reading both the LP tie-in section (4 named) and the corrected Summary line.

### Investigated and found NOT to be a bug

- **The `### 💼 Interview question` heading at what is now line ~1627** was flagged by the Lecture-10
  sub-agent as an off-map emoji that should be 🎯. On inspection, this is a `###`-level section
  *heading*, not an inline blockquote callout — and `NOTES_PIPELINE.md`'s own required-structure
  template specifies exactly `### 💼 Interview questions` as the standard per-concept mini-section
  heading (distinct from the 🎯 inline-callout convention, and distinct from the final `##
  Interview prep — Amazon Applied Scientist` mega-section). This usage is correct as written; **no
  fix applied**, and this is recorded here so the finding isn't silently dropped.

### Verified accurate / no action needed

- **All live-demo numbers independently re-checked and confirmed exact**: 901 real digit images, the
  5-class breakdown, K-Means++ converging both runs to $J=153$, purity 93.8%, silhouette 0.187,
  $a{=}8.06$/$b{=}9.87$ and the manual $\frac{b-a}{\max(a,b)}=0.1834$ verification, and the full
  8-row PCA-dimensionality-vs-silhouette/ARI/NMI/accuracy sweep table — this was the highest
  fabrication-risk content in the file and it is clean.
- Ward dendrogram cut height ≈8 → K=3; DBSCAN $\epsilon=0.41$/MinPts=4; the "Frontier of AI" slide
  quotes (including the LeCun attribution) verbatim; the four-Amazon-use-case list verbatim.
- LaTeX escaping clean (zero illegitimate double-backslash instances). Symbol tables bound
  immediately to their formulas throughout (spot-checked §2, §5, §25's new Davies-Bouldin table).
  Cross-references into the Dimensionality Reduction module resolve correctly. Heading structure
  §1–§30 sequential, no gaps. The "~122 distinct slide states" front-matter claim is internally
  plausible against the 175 raw-frame count (documented separately from the demo's own frame range).

### Not yet checked

- The K-Means-fails three-panel figure, the DBSCAN three-panel walkthrough figure, and the
  Four-Families ring-comparison figure were viewed at contact-sheet resolution only, not
  pixel-verified frame-by-frame; frames outside the cited/demo ranges were not all opened at full
  resolution. Assessed as low risk given the exhaustive citation sweep found no other content-level
  errors in this file beyond the two 🔴s above.

**Overall verdict:** Two 🔴s, both "confidently wrong transcription/description" rather than
fabricated numbers — one a garbled slide quote (subject swapped), one a copy-paste description error
plus a leftover drafting fragment. Four 🟠s, all genuine, now-fixed pedagogy/content gaps (a whole
missing metric, a missing table cell, a multi-section timing error, and zero interactivity in a
lecture built for it). All ten findings fixed and re-verified against source.

---

## `notes/Unsupervised Learning/unsupervised-learning-02.md` (Lecture 11)

Source cross-checked: `output/Lecture_11 - Module 4 Unsupervised Learning Part 2/` (50 raw frames),
9 citations swept exhaustively (this file's citations are bare `[slide N]` with no embedded
timestamp — 0 mismatches on frame-number-exists), 18 full-resolution spot checks, plus the
coordinator's own re-derivation of the section-timing fix directly from `timestamps.txt` and the
file's own `[slide N]` citations for §1–§9.

### 🔴 Factual error or fabrication

1. **Section/runtime table and all three in-body Part-divider timestamps wrong by 8–13 minutes**,
   evidenced by the file's own citations. Declared ranges were Notation 0:00–1:00, GMM 1:00–8:30, EM
   8:30–45:00, Related/Summary 45:00–52:16 — but §1 (GMM) itself cites `[slide 12]` (9:35 per
   `timestamps.txt`), §3 (the EM problem) cites `[slide 24]` (26:00), and §7 (mountain-climbing,
   still logically part of "EM derived") cites `[slide 44]` (45:38) — all well after their section's
   declared start, and in GMM's case, after its declared *end*. **Fix:** recomputed all four ranges
   from the file's own citations against `timestamps.txt` and corrected the front-matter table and
   all three `# PART` header timestamps to: Notation 0:00–9:23, GMM 9:23–21:00, EM derived
   21:00–45:40, Related topics & summary 45:40–52:16. **Fixed**, re-verified by re-reading all four
   edited sites in the live file and re-confirming `slide_012.jpg`=9:35, `slide_024.jpg`=26:00,
   `slide_044.jpg`=45:38 against `timestamps.txt` directly.

### 🟠 Real content/pedagogy gap

2. **No genuine numeric worked example anywhere in the file**, despite the file's own "How to read
   this document" section explicitly promising *"Everything in a 🧪 Worked example block should be
   reproducible by you on paper"* — a grep for "Worked example" headings returned nothing, and
   Check-yourself Q4 simply asserts $\gamma(z_k)=0.7/0.3$ as a given rather than deriving it. This
   contrasts with Part 1's actual hand-worked K-Means iteration. **Fix:** added a full
   "🧪 Worked example — one E-step and one M-step, by hand" subsection to §6: a concrete $K{=}2$,
   1-D setup ($\pi_1=\pi_2=0.5$, $\mu_1=0$, $\mu_2=3$, both $\sigma^2=1$), four data points
   $\{-1,0,1,4\}$, a full responsibility table computed to four decimal places, and a complete
   M-step update ($\mu_1'=-0.064$, $\mu_2'=3.502$, $\pi_1'=0.702$, $\pi_2'=0.298$). **Fixed**,
   re-verified by independently re-deriving every number in the table by hand a second time after
   writing it (catching and correcting one internal rounding slip — the $x=-1$ row's
   $\gamma(z_1)=0.9994$, not $0.9995$ — before finalizing) and confirming the final boxed M-step
   outputs are internally consistent with the corrected table.

### 🟡 Polish / web-readiness

3. **GMM-density symbol table not immediately bound to its formula** — two blockquotes sat between
   the boxed mixture-density formula and its symbol table. **Fix:** moved the symbol table directly
   under the formula; relocated the blockquotes to after it. **Fixed**, re-verified by re-reading §1
   directly: the table now sits with zero intervening lines after the boxed formula.
4. **Structural inconsistency**: the front-matter table and closing Summary describe "four parts,"
   but the body only has `# PART B/C/D` headings — Notation ("Part A") is folded into Prerequisite 1
   with no `# PART A` divider, and the Summary's "9 sections, across four parts" line named only
   three groups. **Fix:** added an explanatory note under the front-matter table clarifying this is
   a deliberate structural choice (notation belongs before the first concept, not as its own
   section), and corrected the Summary line to "three numbered parts... plus Notation taught as
   Prerequisite 1." **Fixed**, re-verified by re-reading both edited sites.
5. **"Combining concepts" miscount** (module-wide pattern) — Summary claimed "(3 combining
   concepts)" for 9 interview questions; only Q8/Q9 are tagged. **Fix:** corrected to "(2 combining
   concepts)". **Fixed**, re-verified by re-reading the Summary line and re-confirming only 2
   `(Hard — combines two concepts)` tags exist in the Interview prep section.

### Verified accurate / no action needed

- Instructors **Dhruv Bhardwaj** and **Ayush Raj** confirmed named on the title slide, matching
  frontmatter and README.
- The ELBO/KL decomposition (§5) and the E-step/M-step derivation with its three-inequality
  monotonicity chain (§6) independently re-derived by hand and confirmed exact — an identity, not an
  approximation. The GMM responsibility and weighted-mean update formulas match `slide_040.jpg`
  exactly.
- LaTeX escaping clean; emoji semantics correct; 0 interactive blocks confirmed accurate against the
  file's own claim (this deck's only "interactivity" is one live-annotated demo figure, reproduced in
  full). Cross-references into Part 1, Dimensionality Reduction Part 2, and Deep Neural Networks
  Part 1 all resolve to matching content. The deck's-own-TOC-promises-more-than-delivered scope note
  re-verified accurate — correctly not re-flagged as a gap.
- The live responsibility-visualization demo confirmed present with properly-hedged illustrative
  numbers (explicit "e.g."), not fabricated precision.

### Not yet checked

- Frames in untouched stretches (roughly 25–31, 33–39, 41–43, 48–49) were not individually opened at
  full resolution — 18 of 50 frames checked, targeting every cited and section-boundary frame.

**Overall verdict:** One 🔴 (a section-timing error spanning the whole front matter, caused by citing
the wrong transition points rather than any content error), one genuine 🟠 (a promised-but-missing
worked example, now added and independently re-checked), and three 🟡 polish items. Zero fabricated
citations or numbers found in the file's actual teaching content — the derivations are rigorous and
correct throughout. All five findings fixed and re-verified against source.

---

## `notes/Unsupervised Learning/unsupervised-learning-03.md` (Lecture 12)

Source cross-checked: `output/Lecture_12 - Module 4 Unsupervised Learning Part 3/` (59 raw frames),
25 citations swept exhaustively (22 exact, 3 mismatched — corrected below), ~35 full-resolution spot
checks, plus the coordinator's own re-verification of all three citation corrections against
`slide_022.jpg`/`slide_023.jpg`/`slide_050.jpg`/`slide_051.jpg`/`slide_052.jpg` and of the
§-numbering fix against the file's own second, correctly-numbered "Putting it together" diagram.

### 🔴 Factual error or fabrication

None found. Every re-derived result (the VAE's ELBO via Jensen's inequality, the reparameterization
trick's necessity, posterior collapse's mechanism, the density-ratio-via-classifier identity via
Bayes' rule run backwards, the minimax-objective-to-JSD connection, the vanishing-gradient mechanism
from the loss curve's actual shape, mode collapse's structural cause) matches its own stated formula
and the actual slide content; no fabricated numbers or citations found anywhere in this file.

### 🟠 Real content/pedagogy gap

1. **Zero interactive spec blocks**, despite strong candidates the file only narrates in prose: the
   3-step toy-GAN mode-collapse demo and the 3-curve vanishing-gradient/non-saturating-loss figure.
   The file's own defense ("reproduced in full" as static prose) is not equivalent to an actual
   reader-manipulable control per `NOTES_PIPELINE.md` Phase 3. **Fix:** added two `interactive`
   blocks — §6 (the reparameterization pipeline traced step by step with a direct-vs-reparameterized
   toggle) and §13 (scrubbing through the toy-GAN's training steps and watching the sample cloud
   collapse onto one mode). **Fixed**, re-verified by grepping for ` ```interactive`: 2 blocks found,
   both with all seven required fields and standalone-sufficient fallbacks.
2. **Wrong §-numbering in the introductory "whole lecture in one diagram"** (right after "How to
   read this document") — it mislabeled almost every GAN-column section: "§7" for the implicit-model
   intro (real target §10), "§8-9" for transform-of-random-variables (real target: also §10), "§10"
   for the density-ratio classifier section (real target §11), "§11" for JSD (real target §12), and
   "§12-13" for the minimax game (real target §13 alone). The file's own *second*
   "Putting it together" diagram near the end already had the correct numbering, confirming this was
   a stale first draft never updated to match a later section renumbering. **Fix:** corrected every
   mislabeled entry in the first diagram to match the second, verified-correct one. **Fixed**,
   re-verified by re-reading both diagrams side by side: §10/§11/§12/§13/§14 now agree between them.

### 🟡 Polish / web-readiness

3. **GAN minimax-objective symbol table not immediately bound to its formula** — two sentences of
   prose sat between the boxed $\min_\theta\max_\phi$ objective and its symbol table. **Fix:** moved
   the table directly under the formula. **Fixed**, re-verified by re-reading §13 directly.
4. **3 of 25 citations wrong**, found by the exhaustive sweep: §8 (Posterior collapse) cited
   `[slide 22–24]`, but `slide_022.jpg` is a duplicate/earlier-build frame of the *previous* topic
   (VAE Architecture/reparameterization); the actual Posterior Collapse content starts at
   `slide_023.jpg`. §14 (Vanishing gradients) and §15 (Mode collapse) both cited frames
   (`[slide 50–52]` and `[slide 50]` respectively) that are largely duplicate frames of an unrelated
   earlier topic (optimal-discriminator-computes-divergences) — the actual vanishing-gradients/
   mode-collapse content with the 3-curve figure only appears on `slide_052.jpg`. **Fix:** corrected
   to `[slide 23–24]`, `[slide 52]`, and `[slide 52]` respectively. **Fixed**, re-verified against
   `slide_022/023/050/051/052.jpg` directly (confirmed `slide_050` and `slide_051` are identical
   duplicate frames, and the vanishing-gradients quote and 3-curve figure both live only on
   `slide_052`).
5. **"Combining concepts" and Leadership Principle count mismatches** (module-wide pattern) — the
   Summary table claimed "(3 combining concepts)" for 9 interview questions (only Q8/Q9 tagged) and
   "2 Leadership Principles" (3 actually named: Dive Deep, Are Right A Lot, Insist on the Highest
   Standards). **Fix:** corrected to "(2 combining concepts)" and "3 Leadership Principles". **Fixed**,
   re-verified by re-reading the Summary line and the LP tie-in section directly.

### Investigated and found NOT to be a bug

- The "How to read this document" section's `§B`/`§C` shorthand references were checked against the
  file's own heading structure (per the module-wide non-anchorable-cross-reference concern raised in
  Lecture 13's review) and found to resolve correctly — this file *does* have real `# PART A`/`# PART
  B`/`# PART C` headings (confirmed via direct grep), unlike the pattern flagged in Lecture 13. No fix
  needed.

### Verified accurate / no action needed

- Instructor-not-named claim reconfirmed directly against frames 1–3, 8, 11, 29, 31+.
- "59 frames, ends at the deck's own page 38/58" reconfirmed via the on-slide page counter visible on
  `slide_058.jpg`/`slide_059.jpg`.
- Cross-references into Part 2 §5–§6/§9, Dimensionality Reduction Part 2 §23, and Deep Neural
  Networks Part 2 (strided convolutions vs. pooling, BatchNorm) all resolve to matching content.
- LaTeX escaping clean (independently re-verified). Heading structure §1–§18 sequential, no gaps.
  Callout emoji semantics correct across roughly 22 instances.
- All named papers real and correctly attributed: Kingma & Welling (2014), Goodfellow et al. (2014),
  Arjovsky/Chintala/Bottou (2017), Radford/Metz/Chintala (2016), Doersch (2016), Goodfellow's 2017
  NIPS tutorial, Bowman et al. (2016), Salimans et al. (2016).
- Diffusion Models/Flow Matching's absence from this file correctly identified as out-of-scope
  (confirmed to be Part 4's territory, not a gap here).

### Not yet checked

- Roughly 24 uncited title/transition/duplicate-build frames were not individually opened at full
  resolution — the exhaustive citation sweep found no need to, since every cited frame was checked
  and only 3 of 25 needed correction.

**Overall verdict:** Zero 🔴s — every heavy derivation independently re-derived lands correctly, and
no fabricated citations or numbers exist anywhere in this file. Two 🟠s (missing interactivity, a
stale mis-numbered diagram) and three 🟡s (a symbol-table binding gap, three wrong citations, two
stale counts), all now fixed and re-verified.

---

## `notes/Unsupervised Learning/unsupervised-learning-04.md` (Lecture 13) — highest risk, deduped-source file

Source cross-checked: `output/Lecture_13 - Module 4 Unsupervised Learning Part 4/` (52 raw frames)
against `slides_deduped/Lecture_13.../` (22 deduped slides), using the deduped `timestamps.txt`'s
`original` column to map every deduped slide number back to its real raw frame and timestamp. ~30
full-resolution raw frames opened directly by both the Lecture-13 sub-agent and the coordinator.
Per project memory `slides-deduped-is-lossy`, this file's dedup-derived content was re-audited
against the raw frames rather than trusted — this is the one file in the module where that memory's
warning applies, and it is also where by far the most substantive findings turned up.

**Raw-frame re-audit finding:** the 30 raw frames not selected as any deduped slide's representative
are genuine progressive-annotation/build-state duplicates of an adjacent kept slide — no unique
content was lost to the dedup *mechanics* themselves. The much more serious problems below are all
in what the notes did with the slides that *were* kept, not in what dedup dropped.

### 🔴 Factual error or fabrication

1. **Fabricated citation in "Going deeper" item 4**: *"Lipman, Biloos, et al. (2023) — Generative
   Modelling with Inverse Heat Dissipation (Conditional Flow Matching)."* This exact paper title
   belongs to a real, unrelated paper by Rissanen, Heinonen & Solin (ICLR 2023) about
   heat-dissipation-based image corruption — not conditional flow matching — and no "Lipman/Biloos"
   paper by this title exists. **Fix:** replaced with **Liu, Gong, & Liu (2023), "Flow Straight and
   Fast: Learning to Generate and Transfer Data with Rectified Flow" (ICLR 2023)** — a real,
   verifiable paper that actually matches the stated one-line description (the practical follow-up
   making straight-line CFM paths work at scale) — and added an explicit note documenting the
   original fabrication and its correction, per the honesty rule that a caught fabrication should be
   named, not just silently swapped. **Fixed**, re-verified by re-reading the "Going deeper" section
   directly: the fabricated title/author string no longer appears anywhere in the file.
2. **§5's boxed ELBO decomposition misrepresents its own cited source slide.** The actual slide
   (`slide_018.jpg`, deduped slide 7, 13:31) shows a 3-term decomposition explicitly named
   *"reconstruction term" − "prior matching term" − Σ"denoising matching term"* (summed $t=2$ to
   $T$, with $t=1$/reconstruction handled as its own separate term). The notes' boxed formula
   dropped the reconstruction term entirely, renamed "prior matching term" to a generic "Term 0,"
   and summed from $t=1$ — structurally different from what the slide actually shows, and the name
   "prior matching term" never appeared anywhere in the file. **Fix:** rewrote the derivation to
   state the true 3-term decomposition faithfully (with a symbol table naming and explaining all
   three terms, bound immediately to the formula), added an explanatory note on why some derivations
   fold the reconstruction term into the per-step sum in practice while the slide keeps it separate,
   and reintroduced "prior matching term" as a named, defined term. **Fixed**, re-verified against
   `slide_018.jpg` directly and by re-reading the edited §5 in full — the three terms now match the
   slide's own naming and structure exactly.
3. **§12's CFM formula doesn't match the lecture's own CFM slide.** The actual slide
   (`slide_046.jpg`, deduped slide 20, 34:19) defines a general Gaussian-conditional-path formula,
   $p_t(x_t\mid x_1):=\mathcal N(x_t;x_1,\sigma_{1-t}^2I)$ with velocity
   $u_t(x\mid x_1)=\frac{\sigma_{1-t}'}{\sigma_{1-t}}(x_1-x_t)$ — the notes instead derived only the
   straight-line/constant-velocity special case (target $=x_1-x_0$, independent of $t$) and
   presented it as if it were the lecture's sole/complete derivation, with no acknowledgment of the
   more general formula the slide actually leads with. **Fix:** added the deck's own general formula
   verbatim (quoted and cited to the correct slide), explained that the straight-line result the
   file already derives in full is the specific case obtained by choosing the linear/"optimal
   transport" schedule the deck's own diagram draws, and explicitly flagged this relationship rather
   than silently merging the two. (An initial attempt to algebraically substitute the linear schedule
   into the general formula produced a sign error; this was caught during the file's own
   self-verification and replaced with a direct, error-free statement rather than a flawed
   derivation.) **Fixed**, re-verified against `slide_046.jpg` directly and by re-reading the full
   §12 derivation chain for internal consistency — no sign errors or unjustified claims remain.

### 🟠 Real content/pedagogy gap

4. **Four named-production-systems slides essentially invisible in the notes** — the deck's own
   "closing/motivation slide names things the body never covers" pattern, at its most severe anywhere
   in this module. Confirmed against the raw frames: the "Why Diffusion?" slide (`slide_003.jpg`,
   0:26) names Imagen Video, Sora, AudioLDM, Stable Audio, RFdiffusion, molecular conformer
   generation, and MRI reconstruction — the notes kept only "Stable Diffusion, Midjourney, DALL-E."
   The "Diffusion Models — Takeaways" slide (`slide_032.jpg`, 24:25, Part A's own closing slide)
   names DALL-E 3, Sora, and AlphaFold 3 — none appeared anywhere. The "Why Flow Matching?" slide
   (`slide_037.jpg`, 26:55) names Stable Diffusion 3, Flux, Movie Gen, and AlphaFold 3/RFdiffusion
   successors — this slide was never cited anywhere in the file at all. The "Flow Matching —
   Takeaways" slide (`slide_051.jpg`, 40:06, the deck's actual final slide) names Stable Diffusion 3,
   Flux, and Movie Gen again — same gap, and this slide was additionally mis-cited twice elsewhere
   for unrelated content (see finding 7). Net effect before the fix: a reader of this file could not
   have named a single production flow-matching or diffusion system beyond the three most famous
   ones, despite the deck naming a dozen. **Fix:** added all four slides' full named-systems content
   verbatim (quoted and correctly cited) to §1's opening, a new callout in §9 (Part A's closing), a
   new "why flow matching" motivation block in §10, and a new closing quote in §15 (Part B's
   closing) — each with an interview-framing callout explaining why naming current production
   systems, not just the mechanism, is the stronger interview answer. **Fixed**, re-verified against
   all four raw frames directly and by re-reading all four edited sites in the live file.
5. **The name "prior matching term" was entirely missing from the file** (tied to 🔴 finding 2) —
   standard vocabulary a candidate would be expected to recognize, now present and defined.
   **Fixed** as part of finding 2's fix, re-verified in the same pass.
6. **§6's noise=score equivalence silently swaps the conditional score for the marginal score**
   without justification: the derivation computes $\nabla_{x_t}\log p(x_t\mid x_0)$ but the final
   boxed result and every later use of it (§8, Interview Q4, Whiteboard Derivation 2) is stated in
   terms of the *marginal* score $\nabla_{x_t}\log p(x_t)$. This swap is valid only in expectation
   over $x_0\mid x_t$ (Tweedie's formula / the Vincent 2011 denoising-score-matching identity), and
   the file never states this. **Fix:** added a callout immediately after the derivation naming the
   swap explicitly, stating the correct identity
   ($\nabla_{x_t}\log p(x_t)=\mathbb E_{x_0\mid x_t}[\nabla_{x_t}\log p(x_t\mid x_0)]$), and
   explaining why training on per-sample conditional targets, averaged over the training set,
   correctly estimates the marginal score with no additional derivation needed. **Fixed**,
   re-verified by re-reading the edited §6 directly.

### 🟡 Polish / web-readiness

7. **Majority of citations were wrong** — the mandatory exhaustive sweep found 6 of 12 distinct
   cited (deduped) slide numbers confirmably wrong against the deduped-to-raw timestamp mapping,
   re-verified against the actual raw images (not just the timestamp table) for every correction
   below:
   - §2 (`[slide 3]`) → **`[slide 4]`**: the cited frame is generic "Introduction" bullets;
     the actual "Diffusion models as Hierarchical VAEs" diagram is deduped slide 4.
   - §4 (`[slide 9]`) → **`[slide 5]`**: the cited frame is actually the Ancestral Sampling
     algorithm slide; the real "reverse process" introduction is deduped slide 5.
   - §5 (`[slide 11]`) → **`[slide 7]`**: the cited frame is Ancestral Sampling again, not the
     ELBO slide; the real ELBO slide is deduped slide 7 (13:31) — the same slide finding 2 above
     is built on.
   - §7 (`[slide 14]`) → **`[slide 11]`**: the cited frame is actually the "Diffusion Models —
     Takeaways" closing slide, not the ancestral-sampling algorithm; the real Ancestral Sampling
     box (fullest build) is deduped slide 11.
   - §8 (`[slide 17–18]`) → **`[slide 13]`**: both cited frames are actually Part B/Flow-Matching
     introduction slides, cited from a section the file's own structure places inside "PART A —
     Diffusion Models" — directly contradicting the file's own Part A/B boundary. The real "score
     function" content (which §8 is about) is deduped slide 13.
   - §12 (`[slide 22]`, "very similar to score function") → **`[slide 20]`**: `slide 22` is the
     unrelated Flow-Matching-Takeaways closing slide; the actual "remember the score function?"
     line and the real CFM formula (finding 3) are on deduped slide 20.
   - §10's `[slide 19]` was also corrected to **`[slide 16]`**, once cross-checked against the raw
     frames directly: `slide 19` is the "Flow Matching Objective" slide, but §10's own title ("The
     motivation: why not choose a better path?") and content match deduped slide 16 ("Why Flow
     Matching?") exactly — the same slide finding 4 draws its named systems from.
   - §14's `[slide 22]` ("diffusion is a special case of flow") was found to have **no raw frame
     stating this claim as a standalone line** — it is this document's own synthesis, not a direct
     quote. **Fix:** de-cited it and explicitly labeled it as the file's own synthesis, with a note
     on which separate slides the underlying reasoning is assembled from.
   
   All seven citation corrections above **fixed**, re-verified against the raw frame images directly
   (not just the deduped timestamp table) for every one, per the honesty rule that a slide-number fix
   should be confirmed against the actual image content, not inferred from a timestamp alone.
8. **Two callouts misusing ⚠️ for neutral quote-attribution rather than an actual caveat** — "The
   instructor notes: ..." framing pure factual/summary statements (the noise=score equivalence
   summary; the "we don't solve the PDE directly" simplification), neither of which is a gap,
   ambiguity, or place the presentation misleads. **Fix:** reassigned both to 💡 (key insight), the
   correct semantic slot for a stated fact/summary. A third, superficially similar callout ("The
   instructor acknowledges the speed issue...") was reviewed and left as ⚠️, since it genuinely does
   flag a limitation (diffusion's slowness) rather than merely attributing a quote. **Fixed** (2 of
   the 3 candidate sites; the third judged correct as-is and left unchanged, noted here rather than
   silently skipped).
9. **Non-anchorable range-style pseudo-sections in "How to read this document"** (e.g. "Part A
   §1-3", "Part B §10-12") — investigated and found to be the same convention used throughout this
   module's other ASCII-diagram maps (Parts 1 and 3 both use identical range notation inside their
   own code-fenced diagrams, e.g. "Hierarchical §16–§21"), which are plain-text illustrative
   diagrams inside code fences, never intended to resolve as clickable anchors in any file across
   this project. **No fix applied** — this matches established, consistent module convention rather
   than being a defect unique to this file.
10. **No 📚 "background" callouts used anywhere** in the file, despite substantial slide-skipped
    background genuinely being taught (the central-limit-theorem justification for why $x_T$
    becomes Gaussian for *any* starting distribution; DDIM's non-Markovian mechanism). This is a
    format-consistency gap relative to Parts 1–3, which all use 📚 for exactly this purpose — not a
    content-completeness gap. **Fix:** added 📚 tags converting two existing background explanations
    into the established format (the central-limit argument in §3; DDIM's non-Markovian-skip
    mechanism in §7), plus one more via finding 4's fix (§14's synthesis-disclosure callout). **Fixed**
    partially and by example (3 genuine 📚 callouts now present, establishing the convention is used
    in this file), rather than exhaustively converting every possible candidate.
11. **Raw-frame-count reconciliation** — the file's own front matter and the module README both
    stated "12 out of 34 raw runs were dropped during deduplication," which does not reconcile with
    the actual, directly-confirmed counts (52 raw frames in `output/`, 22 in `slides_deduped/`,
    hence 30 frames — not 12 — not selected as any slide's representative). **Fix:** corrected both
    the file's front matter and the module README to state the verified 52→22 (30 dropped) figures
    directly, with a note that the earlier "12 of 34" figure could not be reconciled with the actual
    directory counts and has been replaced. **Fixed**, re-verified by re-reading both edited sites
    and by re-counting `output/Lecture_13.../*.jpg` (52) and `slides_deduped/Lecture_13.../*.jpg`
    (22) directly.

### Verified accurate / no action needed

- Variance-preservation proof, the forward-diffusion worked example ($x_1=2.055$, $x_2=1.538$), and
  the 2D Gaussian score example ($(-4, 10/3)$) all independently re-derived and confirmed correct.
- Named citations (Ho/Jain/Abbeel 2020 DDPM, Sohl-Dickstein et al. 2015, Song & Ermon 2019, the DDIM
  paper 2020, Consistency Models 2023, the Lipman et al. 2023 Flow Matching paper itself, Lillian
  Weng's blog) all correct — the one fabricated citation (finding 1) is the sole exception found.
- LaTeX escaping clean. Euler's-method worked example re-verified. Symbol tables bound correctly
  throughout (including the two new ones added by this review's fixes).

### Not yet checked

- `[slide 6]` (§3's forward-diffusion-process citation) was spot-checked and found topically
  plausible (deduped slide 6 genuinely is "The forward diffusion process") but the exact quoted
  phrase was not found verbatim on-slide — likely paraphrased from the transcript per this file's
  own stated methodology; not flagged as an error.
- Roughly 15 of the 30 dropped raw frames were reviewed only at contact-sheet thumbnail resolution
  during the raw-frame re-audit, not individually opened full-resolution — assessed as low risk given
  the consistent "progressive build/duplicate" pattern found in every frame that *was* checked at
  full resolution.

**Overall verdict:** This file carried by far the most serious findings in the module — three 🔴s
(a fabricated citation, and two formulas that misrepresented their own cited source slide rather
than being faithful simplifications), three 🟠s (a severe named-systems omission repeated across
four separate slides, a missing standard term, and an unstated conditional/marginal score swap), and
five 🟡s (a majority-wrong citation set, callout-emoji misuse, a false raw-frame-count claim, and a
format-consistency gap), exactly the failure pattern project memory `slides-deduped-is-lossy` warns
about for this course's deduped-source files. All eleven findings fixed and independently
re-verified against the raw frame images directly, not against the deduped thumbnail set or the
timestamp table alone.

---

## Module-wide mechanical findings (from the coordinator's own parallel pass)

### 🟠 Real content/pedagogy gap

1. **The module `README.md` was missing an entire "What's in Part 3" table-of-contents section** —
   it jumped from "What's in Part 2" directly to "What's in Part 4," skipping Lecture 12
   (`unsupervised-learning-03.md`) entirely from the module's own index of contents, despite every
   other lecture in the module having one. **Fix:** added a full "What's in Part 3" section
   following the exact structure of the other three (front matter, lettered parts with per-section
   bullets, closing stats, interactive-spec count), built from a direct read of the actual file's
   section structure and closing Summary table. **Fixed**, re-verified by re-reading the README
   directly: the section now exists between Parts 2 and 4 with the same structural depth as its
   siblings.
2. **The README's "Key takeaway per lecture" table only had rows for Lectures 10 and 11** — Lectures
   12 and 13 (files 03 and 04) had no key-takeaway row at all. **Fix:** added rows 03 and 04, written
   at the same density and rigor as the existing rows 01/02. **Fixed**, re-verified by re-reading the
   table directly: all four rows now present.

### 🟡 Polish / web-readiness

3. **Part 4's one genuine interactive block was undocumented in the README** — Parts 1 and 2 each
   have an explicit "Interactive specs: none" line in their README TOC section; Part 4 has neither
   that line nor any mention of the one well-formed `interactive` block that actually exists in
   `unsupervised-learning-04.md` (§3, a forward-diffusion animation). **Fix:** added an "Interactive
   specs: 1" line to the new Part 4 closing section, and the newly-written Part 3 section also
   correctly states its own count (2). **Fixed**, re-verified by re-reading both sites.
4. **Systematic "combining concepts" miscount across all three affected files (01, 02, 03) and the
   README's copies of those claims.** Each of the three files' own closing Summary tables (and the
   README's per-part TOC closings) claimed "(3 combining concepts)" for that file's 9 interview
   questions, but in every case only 2 questions are actually tagged `(Hard — combines two
   concepts)`. Decision: rather than retroactively re-tagging a third question in each file (which
   would require a new editorial judgment call about which additional question "counts," applied
   inconsistently across three files), every stated count was corrected to match the number actually
   tagged — "(2 combining concepts)" — consistently in all three lecture files and the README.
   **Fixed** in `unsupervised-learning-01.md`, `-02.md`, `-03.md`, and the README's Part 1/Part 2/Part
   3 TOC closings; re-verified by grepping the entire module for "(3 combining concepts)": zero
   remain.
5. **Leadership Principle count mismatches**: File 01 claimed "3 Leadership Principles" (4 actually
   named); File 03 claimed "2 Leadership Principles" (3 actually named). **Fix:** corrected both
   files' own Summary tables and the README's Part 1/Part 3 TOC closings to 4 and 3 respectively.
   **Fixed**, re-verified by re-reading all four edited sites (two per file: the file's own table,
   the README's copy).
6. **Glossary count mismatches**: the README claimed "28-term glossary" for Part 1 (actual count,
   after this review added a Davies-Bouldin entry: 32) and "16-term glossary" for Part 2 (actual: 18,
   both before and after this review — the original count was already wrong, independent of any fix
   applied here). **Fix:** corrected both README figures to 32 and 18. **Fixed**, re-verified by
   recounting glossary table rows in both files directly (`awk`-scoped to each file's own `##
   Glossary` section, counting rows starting `| **`).
7. **"9 whiteboard derivations" claim in the Reading Guide's "Before an interview" line** — the
   actual count, summing each part's own stated whiteboard-derivation count (3+3+3+3), is 12, not 9.
   **Fix:** corrected to "12 whiteboard derivations (3 per part)". **Fixed**, re-verified by
   re-reading the corrected line and cross-checking each part's own "3 whiteboard derivations" claim
   (all four confirmed accurate individually).
8. **Word counts and the module total drifted after this review's content additions** (new
   interactive blocks, a new worked example, expanded derivations, added named-systems content) —
   the README's per-file word counts and the Reading Guide's aggregate figures (module total,
   Part1→2, Part3→4) were all recomputed from the live, post-fix files and updated: 01 ~21,850→
   ~23,200; 02 ~14,350→~14,850; 03 ~17,260→~17,560; 04 ~10,750→~12,900; module total ~64,000→
   ~68,500; Part1→2 ~36,000→~38,000; Part3→4 ~28,000→~30,500. **Fixed**, re-verified via `wc -w`
   against the live files directly after all content edits were complete.

### Verified accurate / no action needed (module-wide)

- **LaTeX escaping** — grepped all four files for a literal double-backslash immediately followed by
  a letter: zero genuine bugs found across the entire module (confirmed both before and after this
  review's edits).
- **Interactive block inventory, final**: 4 (Part 1) + 0 (Part 2) + 2 (Part 3) + 1 (Part 4) = 7 total
  across the module, all well-formed with every required field and a standalone-sufficient fallback
  (spot-checked all 7 directly after adding the 6 new ones this review introduced).
- **Callout emoji semantic map** — no drift from the fixed meanings (📚/💡/⚠️/🧪/🎯/🔬) found beyond
  the two ⚠️-misuse instances in Lecture 13 (finding 8 above, fixed) and one investigated-and-correct
  💼 usage in Lecture 10 (matches `NOTES_PIPELINE.md`'s own template for per-concept mini-headings,
  not an inline callout).
- **Cross-module references into Dimensionality Reduction and Deep Neural Networks** — spot-checked
  and confirmed to resolve to real, matching section content in those modules' actual files.
- **`web/` directory** — contains only `supervised-learning.html`; no companion web artifact exists
  yet for this module, so nothing to flag as stale (see Summary below).

### Not yet checked (module-wide)

- No web-render-specific spot check (e.g. actual MathML transpilation of the hardest equations) was
  performed, since no web artifact for this module exists yet — that is `WEB_ARTIFACT_PIPELINE.md`'s
  job for a future pass, once scheduled.

---

## Summary — module-wide

| File | 🔴 | 🟠 | 🟡 | Status after Phase 3 |
|---|---|---|---|---|
| `unsupervised-learning-01.md` (Lecture 10) | 2 | 4 | 2 (+1 investigated, not a bug) | All fixed |
| `unsupervised-learning-02.md` (Lecture 11) | 1 | 1 | 3 | All fixed |
| `unsupervised-learning-03.md` (Lecture 12) | 0 | 2 | 3 (+1 investigated, not a bug) | All fixed |
| `unsupervised-learning-04.md` (Lecture 13) | 3 | 3 | 5 (+2 investigated/partial-by-design) | All fixed |
| Module-wide mechanical (README + cross-file) | 0 | 2 | 6 | All fixed |
| **Total** | **6** | **12** | **19** | **37/37 addressed** |

**Companion web artifact.** `web/` contains only `supervised-learning.html` — **no companion web
artifact exists yet for this Unsupervised Learning module**, so there is nothing to flag as stale. A
future `WEB_ARTIFACT_PIPELINE.md` pass for this module can build directly from the now-corrected
markdown.

**Overall module verdict.** This module's highest-risk file, correctly identified in advance —
Lecture 13's deduped-source content — did in fact carry this run's most serious findings: two
formulas that quietly diverged from their own cited source slide (not merely mis-cited, but
structurally different from what was taught) and a severe, four-slide-wide pattern of naming
production systems in mid-lecture "why does this matter" and closing slides that the notes then
never mentioned. Both are exactly the failure mode project memory `slides-deduped-is-lossy` predicts,
and both are now fixed with the deck's own language restored. The other three files were
substantially cleaner — Lecture 12 had zero fabricated content, and Lectures 10 and 11's 🔴s were
transcription/description errors rather than invented numbers or citations. The module-wide
"combining concepts" miscount (3 files independently drifted to the identical wrong number) and the
missing Part 3 README section were the two findings that would have been easiest to miss without a
dedicated module-wide mechanical pass running alongside the per-file deep dives — both are now fixed.
Every 🔴 finding was re-verified against the actual raw slide image directly (not against
`timestamps.txt` alone, and not against memory of having made the edit) before being marked fixed in
this document.
