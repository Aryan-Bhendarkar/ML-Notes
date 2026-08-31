# Amazon ML Summer School — Study Notes & Site

Mastery-grade, self-study notes for all **9 modules** of the Amazon ML Summer
School, rebuilt from a verified slide-extraction pipeline and published as a
self-contained static study site.

**Live site:** https://aryan-bhendarkar.github.io/ML-Notes/

These are teaching documents, not summaries: every term is defined before it is
used, every formula is derived rather than asserted, and every example is worked
through to a final number. ~652,000 words across 29 lectures.

## Modules

| # | Module | Lectures | Words |
|---|--------|:--------:|------:|
| 1 | Supervised Learning       | 3 + practicum | ~94k |
| 2 | Deep Neural Networks      | 3 | ~108k |
| 3 | Dimensionality Reduction  | 3 | ~106k |
| 4 | Unsupervised Learning     | 4 | ~81k |
| 5 | GenAI & LLM               | 4 | ~122k |
| 6 | Sequential Learning       | 3 | ~40k |
| 7 | Causal Inference          | 3 | ~37k |
| 8 | Reinforcement Learning    | 3 | ~33k |
| 9 | Agentic AI *(bonus)*      | 3 | ~30k |

Each module folder under [`notes/`](notes/) has a `README.md` index, the lecture
notes, and a `QUALITY_REVIEW.md` audit trail of what was fact-checked and fixed.

## Repository layout

```
notes/<Module>/        Source markdown — the single source of truth
web/                   Markdown → self-contained HTML build (Node)
docs/                  Built site, served by GitHub Pages (main / docs)
scripts/, *.py         Slide extraction / dedup / OCR / verification pipeline
*_PIPELINE.md          How each stage of the pipeline works
```

`output/` and `slides_deduped/` (raw video frames, ~500 MB) are kept locally as
provenance and are intentionally not committed.

## Build the site locally

```
cd web
npm install
node build.mjs "Supervised Learning"   # build one module (+ refresh the home)
node build.mjs --home                   # refresh only the home page
```

The build prints a fidelity report — source vs. rendered counts for `##`
sections, `<details>` blocks, table rows, ⚠️ callouts and interactive blocks.
They must match; the build is the contract that nothing is dropped or paraphrased.

## Pipeline

1. **Extract** — `process_lecture.py`: download the lecture video, detect slide
   changes by frame diff, grab a full-resolution frame at each change.
2. **Dedupe** — `dedupe_slides.py`: collapse near-identical frames, then
   `verify_dedup.py` checks no distinct slide was lost.
3. **Author** — `NOTES_PIPELINE.md`: slides → mastery notes, one lecture per pass.
4. **Review** — `QUALITY_REVIEW_PIPELINE.md`: fact-check every claim, fix, log the
   findings in the module's `QUALITY_REVIEW.md`.
5. **Publish** — `web/build.mjs`: render the markdown to `docs/`.

## Deployment

GitHub Pages serves this site. `docs/.nojekyll` disables Jekyll so every asset is
served as-is, and `docs/404.html` catches bad URLs.

A GitHub Actions workflow (`.github/workflows/pages.yml`) rebuilds every module and
deploys on any push that touches `notes/**` or `web/**`. To publish by hand
instead, re-run `node build.mjs "<Module>"` and commit the changed `docs/*.html`.

## License & attribution

- **Code** (`*.py`, `scripts/`, `web/`) — [MIT](LICENSE).
- **Notes** (`notes/`, `docs/`) — [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

These are **unofficial, personal study notes**. Not affiliated with, endorsed by,
or produced by Amazon. Lecture and slide content, and the "Amazon ML Summer
School" name, belong to their respective owners.
