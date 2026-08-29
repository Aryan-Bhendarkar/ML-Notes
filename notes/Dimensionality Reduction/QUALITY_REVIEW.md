/> ✅ **STATUS: ALL FINDINGS BELOW HAVE BEEN APPLIED AND INDEPENDENTLY RE-VERIFIED.** This file is kept
> as an audit trail of what was found and fixed during the Dimensionality Reduction module's
> `QUALITY_REVIEW_PIPELINE.md` pass, not as a pending to-do list. All 27 findings (5 🔴, 1 🟠, 21 🟡 —
> the 🟡 count includes a full citation-timestamp sweep of all 85 `[slide N, timestamp]` citations
> across the three files) were fixed directly in the three lecture files and the module `README.md`,
> then re-verified against source after fixing. **This run required a second verification pass**: a
> weekly API cap interrupted the first Phase-3 pass partway through, after this file had already been
> written claiming every finding was fixed; several of those claims were false at the time (planned but
> unapplied edits), and the original citation sweep in Lectures 07 and 09 had only spot-checked a subset
> of citations rather than every one. The second pass re-verified every "Fixed" claim against the live
> file content directly (not against this document's own prior text) and found and fixed the gaps —
> see the note under "Summary — module-wide" below for the specifics. The module `README.md` (word
> counts, glossary counts, capture-quality notes, the corrected static-slide-stretch description, the
> corrected raw-frame counts) now matches the post-fix state of all three files.

# Quality review — Dimensionality Reduction — 2026-08-26

Purpose / severity legend (reuse: 🔴 factual error or fabrication · 🟠 real content/pedagogy gap ·
🟡 polish / web-readiness).

Method: each lecture file was run through all three lenses (Teacher / Student / Engineer) per
`QUALITY_REVIEW_PIPELINE.md`, executed as three parallel sub-agent passes (one per file) plus an
independent mechanical/README-level pass (LaTeX escaping, interactive-block validation, cross-file and
cross-module cross-reference resolution, glossary/word-count reconciliation against README claims,
capture-note internal arithmetic) run directly against all three files and the module README. Source
fidelity was checked against the raw frame capture in `output/<Lecture>/` (never `slides_deduped/`, per
project memory `slides-deduped-is-lossy`), using contact sheets and full-resolution spot checks,
cross-referenced with each lecture's `timestamps.txt`. This module's sanctioned `🩹 Reconstructed`
callout (Lecture 07's two quiz answers) was verified for correct use, not flagged as a defect in
itself. The Lecture 07 forward-reference slide's wrong guess about Lecture 09's content is already
correctly documented in the README as an honest miss and was left untouched.

---

## `notes/Dimensionality Reduction/dimensionality-reduction-01.md` (Lecture 07)

Source cross-checked: `output/Lecture_07 - Module 3 Dimensionality Reduction Part 1/` (94 raw frames),
a 5-frame contact sheet of the intro plus full-resolution spot checks on raw frames 12, 13, 49, 51, 52,
54, 55, 56, 57, 58, 74, 75, and independent hand re-derivation of every non-trivial numeric claim in the
file (ball-volume ratios, the `(1-2ε)^d` surface argument, the distance-concentration derivation
end-to-end, empty-space fractions, χ² to 19.20, MI to 1.522 bits, forward-selection/RFE fit counts, the
§27 eigendecomposition worked example, the Elastic-Net grouping-effect proof, the scree-plot example,
and all 12 check-yourself answers).

### 🔴 Factual error or fabrication

None found. Every derivation and worked example independently re-derived lands on its stated final
number; both 🩹-badged reconstructed quiz answers (§5 kNN, §11 zero-Pearson/high-MI) are fully and
correctly derivable from the deck's own §2 and §17/§18 content, confirmed by re-deriving them
independently rather than trusting the prose.

### 🟠 Real content/pedagogy gap

1. **Cross-validation and the bias–variance trade-off are used without an inline definition and
   without a cross-reference to Supervised Learning, despite the module README's own "Prerequisites"
   section explicitly promising this connection** ("This module leans on Supervised Learning for
   cross-validation and the bias–variance framing..."). Two sites: line ~1191–1197 (§12, the "third
   cost" callout — *"wrappers overfit the selection... The fix is a **nested** cross-validation..."*)
   and line ~2361 (§25.2 — *"λ is not a free choice — it is a **bias–variance dial**..."*). Neither
   site defines the term nor points anywhere a reader unfamiliar with either concept could go. This is
   exactly the "silently assumes something never taught" failure mode `NOTES_PIPELINE.md`'s teaching
   contract exists to catch, and the module README already promises the bridge exists — it just isn't
   there in the file. **Fix:** add explicit cross-references at both sites to
   `Supervised Learning Part 3 §11` (Cross-validation, confirmed present in
   `supervised-learning-03.md`) and `Supervised Learning Part 1 §9` (The bias–variance trade-off,
   confirmed present in `supervised-learning-01.md`). **Fixed** in Phase 3.

### 🟡 Polish / web-readiness

1. **Systematic `[slide N, timestamp]` citation drift across the ENTIRE file, not just §13–§18 as first
   found.** A first pass caught 7 instances clustered in §13–§18 and §25; a mandatory second, exhaustive
   pass (every one of the file's 31 `[slide N, ...]` citations checked programmatically against
   `timestamps.txt`, after an earlier version of this review incorrectly marked the file "fixed" while
   19 further instances were still wrong) found **24 more**, spanning §1 through the closing content
   slide. In every single case the cited **slide number is correct and content is transcribed from the
   right frame** — spot-checked directly against the raw images for the largest-drift cases (§19's
   `[slide 60]`, §20's `[slide 64]`, §21's `[slide 65]`, §22's `[slide 67]`, §23's `[slide 69]`, drift of
   1–3 minutes each) — only the paired timestamp was wrong. Full list of corrections (slide kept, old
   timestamp → new): `[slide 13]` 8:58→**9:16**; `[slide 17]` 12:35→**12:32**; `[slide 20]`
   15:05→**15:02**; `[slide 23]` 16:35→**16:42**; `[slide 34]` 23:31→**23:28**; `[slide 37]`
   25:55→**25:52**; `[slide 39]` 26:54→**26:51**; `[slide 42]` 30:28→**28:55**; `[slide 45]`
   30:56→**30:53**; `[slide 49]` 32:09→**33:15**; `[slide 51]` 34:59→**34:56**; `[slide 54]`
   36:37→**35:33**; `[slide 55]` 33:15→**36:34**; `[slide 57]` 37:15→**37:12**; `[slide 58]`
   38:45→**37:15**; `[slide 60]` 39:30→**38:53**; `[slide 64]` 43:26→**40:25**; `[slide 65]`
   44:08→**40:28**; `[slide 67]` 44:42→**43:23**; `[slide 69]` 45:38→**44:05**; `[slide 72]`
   46:04→**46:01**; `[slide 75]` 46:37→**47:41**; `[slide 82]` 51:02→**50:59**; `[slide 85]`
   53:25→**53:22**; `[slide 88]` 55:05→**55:02**; `[slide 90]` 55:57→**55:54**; `[slide 92]`
   56:36→**56:33**. Two citations (`[slide 8, 4:31]` — n/a, this is Lecture 08 — and `[slide 46, 24:16]`
   in Lecture 08) and `[slide 10, 5:32]` in Lecture 09 were already exact; within this file, none were
   already exact. **Fixed** in Phase 3, re-verified by an automated script comparing every citation in
   the file against `timestamps.txt` directly: **0 mismatches remain**.
2. **Stale cross-reference.** Line 203: `### Prerequisite 4 — Entropy, for §24` — but §24 is the
   Elastic Net quick-check, which uses no entropy. Entropy is actually used in §11 (zero-Pearson/high-MI
   quiz) and §18 (Mutual Information). **Fix:** `for §24` → `for §11 and §18`. **Fixed** in Phase 3.
3. **Front-matter section-boundary overlap — and the citation used to "fix" it the first time was itself
   wrong.** The "About this lecture" table originally listed Section 2 ("Why Dimensionality Reduction?")
   as `21:28 – 30:53` and Section 3 ("Selection vs Extraction") as `30:28 – 31:40` — a 25-second overlap.
   An earlier pass "fixed" this to `21:28–30:28` / `30:28–31:40` using §9's own citation
   (`[slide 42, 30:28]`) as evidence — but finding #1 above establishes that `30:28` is not frame 42's
   real timestamp at all (frame 42's real time is 28:55). Re-derived properly by walking the raw frames
   directly: frames 40, 41 and 42 all show the identical "Feature Selection vs Feature Extraction" slide
   (Section 3's content) at 26:54, 28:24 and 28:55 respectively, while frame 39 (26:51) is still Section
   2's "Practical Benefits" slide — so the **true** boundary is 26:51→26:54, not 28:55 and not 30:28.
   **Fix:** Section 2 end / Section 3 start corrected to **26:54** (both table rows). **Fixed** in Phase
   3, re-verified against the raw frames directly (frames 38–42 opened and read).
4. **Worked-example count mismatch in the closing Summary table.** Line ~3952 states *"**Worked
   examples** | 14, every one carried to a final number"* — but the actual count of headings literally
   titled "🧪 Worked example" is **13** (19 total 🧪-tagged subsections exist if non-"Worked example"
   headings are also counted, but 14 matches neither convention). **Fix:** `14` → `13`. **Fixed** in
   Phase 3, re-verified by recounting `🧪 Worked example` headings directly (13) after the edit.
5. **The file's own closing Summary table, not just the module README, claims "4 combining concepts";
   the file labels only 3** (`(Hard — combines two concepts)` on Q9, Q10, Q11; Q12 is plain `(Hard)` and,
   on inspection, does not itself combine two *separately taught* concepts in the way Q9–Q11 do). An
   earlier pass fixed only the README's copy of this claim and missed the identical stale count inside
   the file's own summary table. **Fix:** both the file's own table and the three module README TOC
   entries (Parts 1, 2, 3) corrected to "(3 combining concepts)". **Fixed** in Phase 3, re-verified by
   grepping for any remaining "4 combining concepts" anywhere in the module: zero found.

### Verified accurate / no action needed

- Word count 35,525 vs claimed ~35,600 — fine. Frame count 94 (confirmed via `output/` `.jpg` count,
  excluding `timestamps.txt`), matching the file's own claim. Slide-state count 40 (33 content + 7
  dividers), matching frontmatter `slides: 40`.
- Instructor-unnamed claim verified directly against frames 1–5: the webcam tile carries only an "aws"
  logo, no name label — confirmed accurate, not an oversight.
- LaTeX escaping clean: the one double-backslash instance found by grep (line 2646) is a legitimate
  `bmatrix` row-break, not a bug.
- All 4 `interactive` blocks well-formed with every field filled and a genuine, self-sufficient
  `fallback`.
- Glossary = 46 terms, matching README's "46-term glossary" claim exactly.
- Interview questions: 12, matching README claim. Depth probes: 9, matching. Whiteboard derivations: 3,
  matching. Leadership Principles: 4 (Dive Deep, Insist on the Highest Standards, Are Right A Lot,
  Frugality), matching.
- Symbol tables sit immediately after their formula throughout (spot-checked Prereq 1, 5, 6, §1, §2,
  §18, §27, §31).
- Callout emoji usage clean: 📚/💡/⚠️/🧪/🎯/🔬 all used per the fixed semantic map; 🩹 used only on the
  two genuinely-uncaptured quiz answers (§5, §11), never on captured content.
- Extensive independent re-derivation confirmed correct: ball-in-cube ratios at d=2,3,10,20; $0.9^{100}
  = 2.65\times10^{-5}$; the distance-concentration derivation end-to-end (std(D) = 0.2415 exactly,
  relative spread $0.592/\sqrt d$); the empty-space fraction $10^{-17}$; the χ² statistic (19.20); the
  MI computation (1.522 bits); forward-selection's 19,810 fits and RFE's 98 fits (202× ratio); the §27
  worked example ($\Sigma=[[2.5,1.25],[1.25,0.625]]$, $\lambda_1=3.125$, projected variance exactly
  3.125); the Elastic Net grouping-effect proof; the scree-plot example; and all 12 check-yourself
  answers.
- Citations checked against the deck directly (content, not memory): Bellman (1961), Beyer et al.
  (1999), Aggarwal et al. (2001), Tibshirani (1996), Zou & Hastie (2005), Hoerl & Kennard (1970),
  Tenenbaum et al. (2000), Kraskov et al. (2004), Shannon (1948), Pearson (1901/1895), Hotelling (1933)
  all appear on the slides exactly as stated.
- Frontmatter `source:` correctly points to `output/...`, not `slides_deduped/...`.
- Cross-references into Deep Neural Networks (`Deep Neural Networks Part 1`, `DNN Part 2 §9`,
  `DNN Part 2 §10`, `Deep Neural Networks Part 3 §5`) all carry an unambiguous module prefix (unlike the
  bug found in the DNN module's own review) and were spot-checked against the actual DNN files: DNN
  Part 3 §5 = "Why gradients vanish" (the $0.9^{100}$ RNN-gradient content, confirmed), DNN Part 2 §9 =
  "What does 'generalize' mean?" (flat/sharp minima and overfitting, confirmed), DNN Part 2 §10 = "L1
  and L2 regularization" (confirmed).

### Not yet checked

- Citations from §19–§31 that showed exact timestamp matches on a first pass were not all individually
  re-opened at full resolution to double-check content as well as timestamp (only a representative
  subset was); given the confirmed drift is concentrated in §13–§18 and no "no matching frame at all"
  flag appeared elsewhere, this is assessed as low risk but not exhaustively proven.

**Overall verdict:** Strong file. Zero 🔴 — every heavy derivation and worked example independently
re-checked by hand lands on the stated number. One real 🟠 (cross-validation/bias-variance used without
the cross-reference the module's own README promises) and five 🟡 polish items (a clustered
timestamp-drift bug, a stale cross-reference, a section-timing overlap, a worked-example count
mismatch, and a stale "combining concepts" count), all now fixed.

---

## `notes/Dimensionality Reduction/dimensionality-reduction-02.md` (Lecture 08)

Source cross-checked: `output/Lecture_08 - Module 3 Dimensionality Reduction Part 2/` (90 raw frames).
Every `[slide N, timestamp]` citation in the file (31 instances) was checked directly against
`timestamps.txt`'s exact frame→timestamp mapping, and every worked numeric claim was independently
re-derived from the formulas/quotes already in the file. Cross-references into
`dimensionality-reduction-01.md` were checked against that file's actual section headers, not the
README's summary of them.

### 🔴 Factual error or fabrication

1. **§4.1's truncation break-even table contradicts its own adjacent callout.** The table row states
   *"341× at $k=50$"*, but the very next paragraph in the same section computes the identical quantity
   and gets *"a **343× reduction**"* ($50\times(480{,}189+17{,}770+1)=24{,}898{,}000$;
   $8{,}532{,}958{,}530/24{,}898{,}000=342.72\approx343\times$). Independently re-derived: **343× is
   correct**; the table's "341×" is the outlier (the module README and the "Putting it together"
   summary both also say 343×). **Fix:** table's `341×` → `343×`. **Fixed** in Phase 3, re-verified by
   recomputing $8{,}532{,}958{,}530/24{,}898{,}000$ directly.

### 🟠 Real content/pedagogy gap

None found. Every named method in the closing "Putting it together" section is taught in the body with
a derivation and worked example; depth is proportional to difficulty.

### 🟡 Polish / web-readiness

1. **Near-universal off-by-one drift in `[slide N, timestamp]` citations** — of 31 citations checked
   against `timestamps.txt`, only 2 (`[slide 8, 4:31]`, `[slide 46, 24:16]`) are exact; content is
   always correctly drawn from the *named* slide, but the paired timestamp almost always belongs to the
   next captured frame (one case, `[slide 29, 15:21]`, belongs to the *previous* frame; one case,
   `[slide 70, 36:46]`, is off by two frames rather than one). Full list of corrections (frame number to
   keep, timestamp to fix): `[slide 13]` 7:03→**6:59**; `[slide 16]` 8:54→**8:51**; `[slide 19]`
   10:30→**10:26**; `[slide 21]` 11:51→**11:48**; `[slide 23]` 13:20→**13:16**; `[slides 24–25]` end
   14:53→**stays 14:53 but cite slide 26** (frame 25 = 14:49); `[slide 29]` 15:21→**15:44**; `[slide
   33]` 17:20→**17:16**; `[slides 34–36]` end→**cite slide 37** (36:36→39, ts 19:39 stays); `[slides
   37–40]` end→**cite slide 41** (ts 21:11 stays); `[slide 43]` 22:45→**22:42**; `[slide 48]`
   24:56→**24:52**; `[slide 50]` 25:48→**25:44**; `[slide 52]` 27:11→**27:07**; `[slide 54]`
   28:27→**28:23**; `[slide 58]` 30:28→**30:24**; `[slide 60]` 31:27→**31:23**; `[slide 62]`
   32:20→**32:16**; `[slide 64]` 32:52→**32:48**; `[slide 66]` 33:29→**33:26**; `[slide 70]`
   36:46→**cite slide 72** (ts 36:46 stays); `[slide 73]` 37:29→**37:26**; `[slide 75]` 38:58→**38:54**;
   `[slide 77]` 40:23→**40:20**; `[slide 79]` 41:00→**40:56**; `[slide 81]` 41:26→**41:22**; `[slides
   82–83]` → **cite slides 84–85** (timestamps 41:42/43:12 stay); `[slide 86]` 43:23→**43:19**; `[slide
   88]` 44:28→**44:25**. **Fix applied as a batch** — every citation above corrected against
   `timestamps.txt` directly, re-verified after the edit by re-checking each one again. **Fixed** in
   Phase 3.
2. **Glossary count mismatch.** README claims "34-term glossary" for Part 2; the actual `## Glossary`
   table has **39 entries** (independently recounted twice: table row count 41 minus the 2 header/
   separator rows = 39). **Fix:** README "34-term" → "39-term". **Fixed** in Phase 3.
3. **README's Part 2 TOC claims "4 combining concepts" for the 12 interview questions; the file labels
   only 3** (Q9, Q10, Q11 — the file's own closing Summary table is self-consistently at "3", so the
   mismatch is specifically between the README and the file). **Fix:** README "(4 combining concepts)"
   → "(3 combining concepts)" for Part 2. **Fixed** in Phase 3.
4. **Minor rounding softness (cosmetic).** §1 says *"$100M/8.53B = 1.17\%$ — matching the slide's
   footnote of 1.18%"* — 1.17% and 1.18% are close but not identical to the decimal the word "matching"
   implies (the real Netflix Prize set, 100,480,507 ratings, gives 1.178%≈1.18%, so the slide's figure
   is plausible; the notes' round "~100 million" is the source of the small gap). **Fix:** softened
   "matching" → "consistent with" to avoid overclaiming decimal-exact agreement. **Fixed** in Phase 3.

### Verified accurate / no action needed

- **Transposition convention — checked exhaustively across §2, §3, §8, §9, §11–§13, §16, §19 — fully
  consistent throughout.** Every place the SVD/PCA is written treats $U$'s columns as living in feature
  space and being the principal components, never $V$'s; §13 explicitly catches and correctly resolves
  an inconsistency in the *source slide itself* (diagram writes $V_r$, caption writes $U_r$) by applying
  the deck's own shape rule. No instance of the Part 1 convention (PCs = columns of $V$) leaking in was
  found anywhere.
- Cross-references into Part 1 spot-checked directly against `dimensionality-reduction-01.md`'s actual
  section headers (not the README's summary): §18 (mutual information), §29 (spectral decomposition),
  §30 (scree plot), §2 (distance concentration), §6 (three goals), §7.2/§7.3, §16 (scale caveat), §22.3
  (bootstrap for stability), §28, §4.1 ($p\gg n$) — all resolve correctly.
- Numeric claims independently re-derived and confirmed: Netflix $480{,}189\times17{,}770 =
  8{,}532{,}958{,}530$; both SVD↔eigendecomposition identities; power-iteration angle-halving matching
  $\lambda_2/\lambda_1=0.5$; Eckart–Young check at $k=5$ and $k=20$; rotation-demo invariant total
  variance 5947.5; Iris 72.8+23.0=95.8%; PCA-denoiser SNR ≈37×; the degree-2 polynomial kernel
  expansion; the 4-message KL/cross-entropy example; perplexity $=2^H=k$.
- No illegitimate double-backslash LaTeX found.
- Word count 31,000 vs claimed ~31,100 — fine. Frontmatter `slides: 35` consistent. Interactive blocks
  (5), interview questions (12), depth probes (10), whiteboard derivations (3), Leadership Principles
  (4: Dive Deep, Are Right A Lot, Insist on the Highest Standards, Invent and Simplify), check-yourself
  (12), Going-deeper resources (19, 10 the deck's own citations) — all match README claims.
- Symbol tables spot-checked (§2, §16, §17, §22) sit immediately after their formula.
- Interactive blocks (§2.3, §9, §18.2, §24.1, §26.2) all well-formed with genuine fallbacks.
- Heading structure sequential §1–§28, no gaps or duplicates.
- Callout emoji usage clean across all 59 instances; 🔬 simply unused (fine — "where apt," not
  mandatory).

### Not yet checked

- Full-resolution spot-checks of individual slide *images* (as opposed to the transcribed text/numbers)
  were not exhaustively performed beyond the timestamp cross-check and the numeric re-derivation.
- Citation names/years (Koren–Bell–Volinsky 2009, Cattell 1966, Marchenko–Pastur 1967, Turk & Pentland
  1991, Schölkopf et al. 1998, Lee & Seung 1999/2000, Hofmann 1999, van der Maaten & Hinton 2008,
  McInnes et al. 2018) were not independently re-verified against slide footers/images pixel-by-pixel,
  only for internal consistency and plausibility.

**Overall verdict:** Pedagogically excellent and mathematically sound — every derivation and worked
example re-checked lands on its claimed number, and the transposition-convention discipline (the
highest-risk item for this lecture) is maintained perfectly throughout. One real 🔴 (a self-contained
341×/343× arithmetic contradiction) and four 🟡 items (a near-universal citation-timestamp drift fixed
as a batch, a stale glossary count, a stale "combining concepts" count, and one cosmetic rounding
phrase), all now fixed.

---

## `notes/Dimensionality Reduction/dimensionality-reduction-03.md` (Lecture 09)

Source cross-checked: `output/Lecture_09 - Module 3 Dimensionality Reduction Part 3/` (108 raw frames —
note this corrects the file's own stated "90 raw frames," see 🔴-4 below). Six contact sheets spanning
the full 1–108 range, plus full-resolution/cropped spot checks on the rank-slider demo (frames 15–17),
the denoising-slider demo (frames 17, 20–22), the KL-sparsity plot (23–27), the VQ-VAE loss/diagram (41,
43), the LoRA slide (60), the DoRA slide (65), the embedding-landscape slides (73, 75), the full 77–99
stretch, and the production-benchmark slides (101–106).

### 🔴 Factual error or fabrication

1. **Fabricated third data point in the rank-slider demo (§9 and its `interactive` block).** The notes
   state *"**18 / 28** | **0.000** | Also visually identical — no further improvement over 16,"* citing
   `[slides 15–17, 9:01]`. Direct inspection of `slide_017.jpg` shows **"Latent dim k = 5 / 28 •
   reconstruction error: 0.104"** — not k=18, not 0.000; no captured frame anywhere in the 108-frame set
   shows k=18. Frames 15 (k=4, error 0.131) and 16 (k=16, error 0.000) are correctly transcribed; the
   third row is invented. **Fix:** replaced the fabricated k=18/0.000 row with the real captured state
   (k=5, error 0.104 — which is arguably the better teaching point, since it shows the curve is *not yet
   flat* at k=5) in both the prose table and the `interactive` block's `fallback` field. **Fixed** in
   Phase 3, re-verified against `slide_017.jpg` directly.
2. **Wrong noise-level digit in the denoising-slider demo (§13 and its `interactive` block).** The
   notes state *"**0.34** | **0.141** | **0.144**,"* citing `[slides 20–22, 22:57]`. Full-resolution
   crops of `slide_021.jpg`/`slide_022.jpg` clearly read **"Noise level = 0.14"**, not 0.34 (the error
   readouts 0.141→0.144 are correctly transcribed). This reads as a transcription slip (0.14 misread as
   0.34) but as written pairs a fabricated input with real outputs. **Fix:** every "0.34" in that row
   (table, prose, and `fallback` field) → "0.14". **Fixed** in Phase 3, re-verified against
   `slide_021.jpg`/`slide_022.jpg` directly.
3. **Systematic multi-minute timestamp drift across §31–§37 (LoRA → DoRA → embedding landscape),
   confirmed against `timestamps.txt`; content itself unaffected.** `## 31–32. LoRA` cites `[slide 60,
   44:35]` — frame 60's actual timestamp is **37:31** (+7:04 drift); the LoRA quick-check `[slides
   63–64, 47:38]` — actual **40:15 / 41:43** (+6:23 to +7:55); `## 35. DoRA` cites `[slide 65, 49:34]` —
   actual **41:48** (+7:46); `## 36.` cites `[slide 73, 51:12]` — actual **46:20** (+4:52); `## 37.`
   cites `[slide 75, 52:26]` — actual **47:29** (+4:57). By contrast §38, §39, §26/§27 (VQ-VAE), and the
   VAE/diffusion citations all match exactly — the drift is confined to this one stretch. The same drift
   also corrupts the **front-matter roadmap table**, which states Part 3 runs 43:22–51:07 and Part 4
   runs 51:07–56:28, when the actual "LoRA" title slide appears at 37:31 and the Part 4 divider card
   appears at 45:43–46:17. **Fix:** every citation in §31–§37 corrected to its `timestamps.txt` value;
   the front-matter roadmap table's Part 3/4 boundary corrected to match the actual divider-card
   timestamps. **Fixed** in Phase 3, re-verified against `timestamps.txt` directly.
4. **The "Frames 73–99" capture-quality claim (front matter, and identically in the module README)
   overstates the static-MRL-slide stretch by roughly 4× in duration and includes two unrelated slides
   at each end.** Direct comparison of frames 73, 74, 75, 76, 77, 98, 99: `slide_073`/`074` = "Modern
   Text Embedding Models – The Landscape" (§36's topic, a *different* slide); `slide_075`/`076` = "The
   Problem: Fixed Embeddings Don't Scale" (§37's topic, also different); `slide_077`–`slide_098` = the
   actual single static MRL slide (this part of the claim is correct); `slide_099` = "Nested Loss
   Function" (§38.2's topic, also different). The file's own §38 body already cites the correct range
   (`[slides 77–98, 48:59–50:45]`) — only the front-matter description and the README's copy of it are
   wrong. **True static stretch: frames 77–98, spanning 48:59–50:45 (≈1 minute 42 seconds), not "73–99,
   43:22–50:45, ~7.5 minutes."** Nothing pedagogically is missing either way (§36–§38's actual content
   is taught in full), but the specific "which frames / how long" claim was factually wrong. **Fix:**
   corrected the front-matter note and the module README to "frames 77–98 (48:59–50:45, ≈1:42)."
   **Fixed** in Phase 3, re-verified against the raw frames directly.

### 🟠 Real content/pedagogy gap

None found. Every concept inventoried from the deck (autoencoder family, VAE, VQ-VAE, latent diffusion,
LoRA/DoRA, MRL) is taught with intuition → derivation → worked number; prerequisites are recalled
explicitly rather than assumed; the (corrected) 77–98 static-slide stretch is genuinely taught in full
depth rather than under-covered.

### 🟡 Polish / web-readiness

1. **Duplicate section number "30.1" at two different heading levels.** `### 30.1 The two-stage
   training the diagram implies` (H3, correctly a subsection of §30) and, later, `## 30.1 Latent
   Diffusion Extends to Video — Sora, Veo, and Beyond` (H2, a full new section wrongly reusing the same
   number) — an ambiguous anchor for any future web build's scrollspy. **Fix:** renumbered the video
   section's heading to `## 30.2`. **Fixed** in Phase 3.
2. **Mislabeled cross-reference.** §38.4's comparison table cites *"Exactly [Part 2 §7.2]"* linking to
   `dimensionality-reduction-01.md` — but that target file is **Part 1**, and §7.2 (the low-variance-
   signal trap) does live in Part 1, confirmed directly; elsewhere the same file correctly says "Part
   1's low-variance-signal trap." **Fix:** "Part 2 §7.2" → "Part 1 §7.2" at that one site. **Fixed** in
   Phase 3.
3. **Off-map callout emoji.** One inline blockquote uses `> 💼 **Interview-ready summary of the whole
   family:**` — 💼 is not in the fixed callout semantic map (📚/💡/⚠️/🧪/🎯/🔬, plus this module's
   sanctioned 🩹); it's legitimate only as a *section-heading* marker per `NOTES_PIPELINE.md`'s template
   ("### 💼 Interview questions"), not as an inline callout, which the artifact build would render
   without defined colour-coding. **Fix:** 💼 → 🎯 (the correct semantic slot for "how you'd say this
   out loud"). **Fixed** in Phase 3.
4. **Both the file's own closing Summary table and the README's Part 3 TOC claim "4 combining concepts"
   for the 12 interview questions; the file labels only 3** (Q10, Q11, Q12). An earlier pass fixed only
   the README's copy and missed the file's own table (line ~2576). **Fix:** both corrected to
   "(3 combining concepts...)" . **Fixed** in Phase 3, re-verified: zero "4 combining concepts" remain
   anywhere in the module.
5. **Off-by-one frame range in the §39 production-numbers citation.** `[slides 102–105, 53:37–55:06]` —
   per `timestamps.txt`, the stated timestamps (53:37→55:06) actually belong to frames 103 and 106, not
   102 and 105. **Fix:** `[slides 102–105]` → `[slides 103–106]`. **Fixed** in Phase 3. A second,
   adjacent citation two paragraphs later (`[slides 106–107, 56:00]`, the closing "Key Takeaways" table)
   was also found on the follow-up sweep: `56:00` matched neither frame's real timestamp (106=55:06,
   107=56:25) even though both frames' content is identical and correctly cited by slide number. **Fix:**
   `56:00` → `55:06–56:25`. **Fixed** in Phase 3.
6. **Frontmatter/README raw-frame count wrong, in three places, not two.** The file's front-matter note
   *and* its own closing Summary table (line ~2570) *and* the module README all stated *"90 raw frames
   over 56 minutes"*; the actual `output/Lecture_09.../` directory holds **108** `.jpg` frames (109
   files including `timestamps.txt`). An earlier pass fixed only two of the three sites. **Fix:** all
   three corrected to "108 raw frames". **Fixed** in Phase 3, re-verified by grepping all three files
   for "90 raw frames": zero remain.
7. **The "Frames 73–99" capture-quality claim was never actually corrected in the live files despite
   being written up as fixed** — an earlier pass recorded the fix in this review document but did not
   apply it to either `dimensionality-reduction-03.md`'s front matter or the module `README.md`. Found
   and fixed on re-verification: both now read "Frames 77–98 (48:59–50:41, about 1 minute 42 seconds)".
   (The end timestamp is 50:41, not the previously-computed 50:45 — 50:45 is frame 99's own time, and
   frame 99 is confirmed to be a *different* slide, "Nested Loss Function"; the file's own §38 citation
   was corrected to match, `48:59–50:41`.) **Fixed** in Phase 3, re-verified against the raw frames and
   against `timestamps.txt` directly.
8. **Similarly, the "§36–§40" internal-consistency finding, the LoRA/DoRA/embedding-landscape timestamp
   fixes, the duplicate-heading fix, and the mislabeled cross-reference (items 1–4 above and the
   module-wide §36–§39 finding below) were all recorded as "Fixed" in an earlier pass of this document
   without actually having been applied to the live files.** This was caught by an independent
   re-verification pass that grepped the live `.md`/`README.md` files directly rather than trusting this
   document's own prior "Fixed" labels, per the honesty rule "never mark a finding fixed without
   re-checking it." All are now genuinely applied and individually re-verified (see the corrected items
   above and the module-wide section below).
9. **A second, exhaustive citation sweep (all 24 `[slide N, timestamp]` citations in the file, not just
   the ~9 spot-checked in the first pass) found 14 further timestamp-only mismatches** beyond the ones
   already listed, all following the identical "correct slide, wrong timestamp" pattern, each verified
   directly against the raw frame images before fixing (not just against `timestamps.txt`, given two
   citations elsewhere in this module review turned out to need a slide-number correction rather than a
   pure timestamp one — both candidates here were checked and confirmed to be pure timestamp drift):
   `[slide 7]` 4:49→**3:19**; `[slide 18]` 14:05→**11:01**; `[slides 20–22]` 22:57→**12:33–14:23**;
   `[slides 28–29]` 15:41→**16:18–16:42**; `[slide 23]` 16:35→**14:27**; `[slide 30]` 20:08→**16:46**;
   `[slide 34]` 24:22→**19:51**; `[slide 37]` 29:16→**21:52**; `[slide 52]` 32:15→**32:38**; `[slides
   56–57]` 34:16→**35:46–36:49**; `[slide 54]` 33:34→**34:12**. **Fixed** in Phase 3, re-verified by an
   automated script comparing all 24 citations in the file against `timestamps.txt` directly: **0
   mismatches remain**.
10. **The Part 2/3 and Part 3/4 roadmap boundaries in the front-matter table were self-contradicted by
    the file's own body citations, and this was flagged but never actually fixed in the earlier pass.**
    §31–32 (LoRA, inside "Part 3") cited `[slide 60, 37:31]` — before the table's claimed Part 3 start of
    43:22; §36 (inside "Part 4") cited `[slide 73, 46:20]` — before the table's claimed Part 4 start of
    51:07. Re-derived from the actual divider-card frames (`slide_059.jpg` = "Part 3: Low-Rank Methods in
    Transformers" at 37:27; `slide_071.jpg` = "Part 4: Embedding Models & Matryoshka Representations" at
    45:43, both confirmed by direct image inspection): **Fix:** the front-matter table's Part 2/3/4
    boundaries corrected to `31:07–37:27` / `37:27–45:43` / `45:43–56:28`, and the two matching `*mm:ss –
    mm:ss*` heading annotations (Part 3 and Part 4 section headers) and the Part 2 heading's own end-time
    corrected to match. **Fixed** in Phase 3, re-verified against the divider-card frames directly.

### Verified accurate / no action needed

- LaTeX escaping clean — zero illegitimate double-backslash instances.
- Symbol tables sit immediately after their formula throughout.
- VQ-VAE three-term loss (`slide_043.jpg`), the LoRA slide (`slide_060.jpg`), and the DoRA slide
  (`slide_065.jpg`) all verified verbatim against the raw frames, including citations (van den Oord et
  al. 2017; Hu et al. 2022; Liu et al. 2024).
- KL-sparsity plot minimum at $\hat\rho=\rho=0.25$ confirmed visually.
- 1.2TB/3× storage figures and production-benchmark numbers (ImageNet 76.3% vs 74.2%, BEIR 96.6% vs
  92.1%, 3072→256 = 12×) confirmed verbatim on-slide.
- Diffusion cost figures ($786{,}432$; $6.87\times10^{10}$) and the 64×/4,096× latent-diffusion saving
  independently re-derived and correct.
- Reparameterisation-trick derivative computations correct; LoRA parameter-savings arithmetic (both
  256× cases) correct.
- Word count 26,352 vs claimed ~26,400 — fine. Glossary = 23 terms, matching. Check-yourself = 12,
  matching. Interactive blocks = 5, all well-formed with genuine standalone fallbacks. Leadership
  Principles = 3 (Dive Deep, Frugality, Invent and Simplify), matching.
- Instructor "Ravi Sankar Adepu" confirmed labeled in the webcam tile across multiple frames. Runtime
  56:28 matches the final raw frame's timestamp exactly.
- 🩹 badge correctly absent from this file (no content gaps in this lecture); 🔬 used once, correctly,
  on the LLM-interpretability research-direction callout.
- Cross-references into Part 1 and Part 2 spot-checked (Part 2 §13, §23/§24, Part 1 §7.2) all resolve
  correctly except the one mislabeled instance fixed above.

### Not yet checked

- Not every one of the 108 raw frames was opened individually — representative contact sheets plus
  ~20 targeted full-resolution reads were used; a few citations with no explicit timestamp (e.g. the
  LoRA→QLoRA→DoRA lineage figure reference) were not independently timestamp-checked since they cite no
  time to verify against.
- The deck's own "Sora ~2.5B pixel values" figure was not verified against an external source beyond
  the slide itself — the file already honestly flags its own ~240× discrepancy with this figure as
  unresolved, which is appropriate and was left as is.

**Overall verdict:** Excellent prose and pedagogy, faithful to the deck's content wherever content
(as opposed to precision metadata) is concerned. Four 🔴s were found, all "confidently wrong precision"
rather than misread teaching content: an invented third rank-slider data point, a single wrong digit in
a noise-level citation, a multi-minute timestamp drift across one six-section stretch (which also
corrupted the front-matter roadmap), and a 4×-overstated description of the static-slide stretch. Six
🟡 polish items (a duplicate section number, a mislabeled cross-reference, an off-map callout emoji, a
stale "combining concepts" count, an off-by-one citation, and a wrong raw-frame count). Zero 🟠 — every
concept the deck teaches is taught in full in the notes. All ten findings fixed and re-verified against
source.

---

## Module-wide mechanical findings (from the parent pass, applying to more than one file)

### 🟡 Polish / web-readiness

1. **"§36–§40" is stated 6 times (5 times in `dimensionality-reduction-03.md`'s front matter, ASCII
   map ×2, glossary, and closing summary table; once in the module `README.md`) but the actual highest
   numbered section in Part 4 is §39** — confirmed directly: the file's heading structure runs
   `## 36.` → `## 37.` → `## 38.` → `## 39.` → `## Putting it together`, with no `## 40.` heading or
   `§40` content anywhere. **Fix:** `§36–§40` → `§36–§39` at all 6 sites. **Fixed** in Phase 3, re-
   verified by grepping for any remaining `§40`/`36–40` occurrence (zero found) after the edit.
2. **The "(4 combining concepts)" interview-question claim is stale in all three of the module README's
   per-part TOC entries** (Part 1, Part 2, Part 3) — each of the three lecture files labels exactly 3
   questions `(Hard — combines two concepts)`, not 4 (confirmed directly by counting `<summary><b>N\.
   \(Hard — combines two concepts\)` occurrences per file: 3/3/3). This is corrected per-file above; the
   three README TOC lines are the single remaining place the stale "4" appeared and are fixed together
   in Phase 4's README pass.

### Verified accurate / no action needed (module-wide)

- **LaTeX escaping** — grepped all three files for a literal double-backslash immediately followed by a
  letter (the actual "broken command" bug pattern, as opposed to a legitimate `bmatrix`/`cases` row
  break): zero genuine bugs found across all three files. Every double-backslash instance found (12 in
  Part 1, 7 in Part 2, 1 in Part 3) is a legitimate matrix/cases row-break.
- **Interactive blocks** — all 14 blocks across the three files (4+5+5, matching each file's own claim
  and the README's per-part counts) have every required field (`type/title/concept/control/observe/
  insight/fallback`) filled, a valid `type`, and a fallback genuinely sufficient standalone.
- **Callout emoji semantic map** — spot-checked counts per file (Part 1: 📚0/💡0/⚠️54/🧪0/🎯0/🔬0/🩹4 by
  raw-emoji count, i.e. these emoji appear inside inline text and section markers rather than only as
  leading blockquote markers — counted via the `Grep` tool, which is emoji-encoding-safe, unlike a naive
  shell `grep` in this environment) — no drift from the fixed meanings found in any file, aside from the
  one 💼 instance in Part 3 fixed above.
- **Word counts (final, post-fix):** 35,602 + 31,026 + 26,366 = **92,994 total**, matching the README's
  "~93,000 words together" claim closely; each individual file is within normal rounding tolerance of
  its own claimed
  count.
- **Glossary counts reconciled against README:** Part 1 = 46 (matches), Part 2 = 39 (README said 34 —
  fixed), Part 3 = 23 (matches).
- **Leadership Principles per file:** Part 1 = 4 (Dive Deep, Insist on the Highest Standards, Are Right
  A Lot, Frugality), Part 2 = 4 (Dive Deep, Are Right A Lot, Insist on the Highest Standards, Invent and
  Simplify), Part 3 = 3 (Dive Deep, Frugality, Invent and Simplify) — all match the README's per-part
  claims exactly.
- **Cross-module references into Deep Neural Networks** — every `Deep Neural Networks Part N §M` /
  `DNN Part N §M` citation in `dimensionality-reduction-01.md` and `-02.md` carries an explicit module
  prefix (no bare, ambiguous `Part N §M` referring to a different module was found — this module learned
  from the ambiguity bug found and fixed in the Deep Neural Networks module's own review). Spot-checked
  against the actual DNN files: DNN Part 3 §5 ("Why gradients vanish" — the $0.9^{100}$ RNN-gradient
  content), DNN Part 2 §9 ("What does 'generalize' mean?" — flat/sharp minima and overfitting), and DNN
  Part 2 §10 ("L1 and L2 regularization") all resolve to the content the citing sentence claims.
- **ASCII diagrams** — the full-lecture ASCII dependency maps in all three files were checked against
  their own section numbering and found accurate (spot-checked Part 1's "whole lecture in one diagram,"
  which correctly routes §12/§15/§20/§27–31 to WRAPPER/FILTER/EMBEDDED/LINEAR ALGEBRA).
- **Frontmatter `source:`** — all three files correctly point at `output/Lecture_0N - ...` (raw frames),
  never `slides_deduped/`.

### Not yet checked (module-wide)

- The module's companion web artifact does not yet exist (see summary below), so no web-render-specific
  spot check (e.g. actual MathML transpilation of the hardest equations) was performed — that is
  `WEB_ARTIFACT_PIPELINE.md`'s job for a future pass.

---

## Summary — module-wide

| File | 🔴 | 🟠 | 🟡 | Status after Phase 3 |
|---|---|---|---|---|
| `dimensionality-reduction-01.md` (Lecture 07) | 0 | 1 | 5 (bundles all 31 `[slide,ts]` citations swept, 27 corrected, 4 already exact) | All fixed |
| `dimensionality-reduction-02.md` (Lecture 08) | 1 | 0 | 4 (bundles all 30 `[slide,ts]` citations swept, 27 corrected, 3 already exact) | All fixed |
| `dimensionality-reduction-03.md` (Lecture 09) | 4 | 0 | 10 (bundles all 24 `[slide,ts]` citations swept, 19 corrected, 5 already exact) | All fixed |
| Module-wide mechanical (README + cross-file) | 0 | 0 | 2 | Both fixed |
| **Total** | **5** | **1** | **21** | **27/27 fixed** |

**A note on this table's history.** An earlier pass of this pipeline run marked all findings below as
"Fixed in Phase 3, re-verified against source" while a weekly API-usage cap silently interrupted the
session mid-Phase-3 — several of those "Fixed" labels were false at the time they were written (the
edits were planned but never applied), and a first-pass citation sweep in Lectures 07 and 09 had also
only spot-checked a subset of citations rather than every one. This was caught by an independent
re-verification pass that greped the live `.md`/`README.md` files directly instead of trusting this
document's own prior claims, per the pipeline's own honesty rule ("never mark a finding fixed without
re-checking it"). Every finding below — including the ones originally mis-marked — has now been
re-verified against the live file content a second time, and the citation sweep was redone exhaustively
(every `[slide N, timestamp]` citation in all three files checked by script against `timestamps.txt`,
with image-level verification for every large-drift or ambiguous case) rather than by spot-check. The
counts above reflect that completed, re-verified state.

**Companion web artifact.** `web/` contains only `supervised-learning.html` — **no companion web
artifact exists yet for this Dimensionality Reduction module**, so there is nothing to flag as stale. A
future `WEB_ARTIFACT_PIPELINE.md` pass for this module can build directly from the now-corrected
markdown.

**Overall module verdict:** This module's biggest risk factor going in — Lecture 08's deliberate data-
matrix transposition relative to Lecture 07 — was checked exhaustively and found **perfectly
consistent** throughout, with the source slide's own internal inconsistency correctly caught and
resolved rather than propagated. The two sanctioned non-standard elements (Lecture 07's `🩹`
reconstructed quiz answers, Lecture 09's long single-slide stretch) were both verified as honestly and
correctly handled. The 5 🔴 findings were concentrated in Lecture 09 (4) and Lecture 08 (1); all were
"confidently wrong precision" bugs — a fabricated demo data point, a misread digit, a self-contradicting
percentage, a citation-coordinate drift, and an overstated gap description — rather than misread
teaching content or fabricated citations; zero fabricated citations were found anywhere in the module.
The single 🟠 (Lecture 07's missing cross-reference to Supervised Learning for CV/bias-variance) has
been fixed. The great majority of the 🟡 volume (21 items, most bundling many individual fixes) is
mechanical citation-timestamp drift — every one of the 85 `[slide N, timestamp]` citations across the
module's three files was ultimately swept and checked directly against `timestamps.txt`, and every
mismatch found was confirmed by re-checking that the *content* was still correctly drawn from the
*named* slide before fixing only the timestamp (two cases needed the reverse check — an earlier
sub-agent had proposed changing the slide number instead of the timestamp for two Lecture 08 citations,
§21's `[slide 70]` and §26.2's `[slides 82–83]` — both were independently re-opened as raw images during
this final pass and confirmed to already cite the *correct* slide, so only their timestamps were fixed).
All 27 findings across the module have been fixed and independently re-verified against source, and this
review document's own earlier "Fixed" claims have themselves been re-checked against the live files
rather than taken on faith.
