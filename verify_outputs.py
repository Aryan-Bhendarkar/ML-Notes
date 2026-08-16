#!/usr/bin/env python3
"""
Post-run QA over every extracted lecture folder.

Checks, per lecture:
  * folder + zip exist, zip opens and its member count matches the folder
  * slide numbering is contiguous (slide_001..slide_NNN, no holes)
  * every image opens, is non-trivial in size, and reports its resolution
  * near-blank frames (e.g. the black frame many videos open on at t=0)
  * near-duplicate neighbours, for a sense of the duplicate rate
Exit code is non-zero if any hard problem (missing/corrupt/hole) is found.
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTROOT = ROOT / "output"
META = ROOT / "playlist_meta.tsv"

BLANK_STD = 6.0     # std-dev below this => essentially featureless frame
DUP_DIFF = 1.0      # mean abs diff below this => visually identical neighbours
SPARSE_INK = 4.0    # edge density below this => title-only / near-empty slide


def thumb(path, size=(160, 90)):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize(size), dtype=np.float32)


def ink(path):
    """Edge density at moderate resolution: a proxy for 'how much content'.

    Text and diagrams create edges; a slide showing only its title scores far
    lower than the same slide once its body has animated in. Used to find
    title-only captures that never got a fully-built counterpart -- the failure
    mode where a gradual reveal slips under the change threshold.
    """
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((480, 270)), dtype=np.float32)
    return float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())


def main():
    expected = {}
    for line in META.read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = line.split("\t")
            expected[int(p[0])] = p[2]

    folders = sorted(d for d in OUTROOT.iterdir() if d.is_dir())
    print(f"{'lecture':<52} {'slides':>6} {'res':>11} {'blank':>5} {'dup':>4} {'zip':>6}")
    print("-" * 92)

    problems = []
    warnings = []
    total_slides = 0
    seen_idx = set()

    for d in folders:
        try:
            idx = int(d.name.split("_")[1].split(" ")[0])
        except (IndexError, ValueError):
            problems.append(f"{d.name}: cannot parse lecture number")
            continue
        seen_idx.add(idx)

        slides = sorted(d.glob("slide_*.jpg"))
        total_slides += len(slides)
        if not slides:
            problems.append(f"{d.name}: NO SLIDES")
            continue

        # contiguity
        nums = [int(s.stem.split("_")[1]) for s in slides]
        if nums != list(range(1, len(nums) + 1)):
            problems.append(f"{d.name}: non-contiguous numbering "
                            f"(min={min(nums)} max={max(nums)} count={len(nums)})")

        # images readable + resolution + blanks + dupes + content density
        res = set()
        blanks, dupes = [], 0
        inks = []
        prev = None
        for s in slides:
            try:
                with Image.open(s) as im:
                    res.add(im.size)
                g = thumb(s)
                inks.append(ink(s))
            except Exception as exc:
                problems.append(f"{d.name}/{s.name}: unreadable ({exc})")
                inks.append(None)
                prev = None
                continue
            if float(g.std()) < BLANK_STD:
                blanks.append(s.name)
            if prev is not None and float(np.mean(np.abs(g - prev))) < DUP_DIFF:
                dupes += 1
            prev = g

        # A sparse (title-only) slide is expected -- it is the head of a stable
        # segment. It is only a problem if NO nearby slide is substantially
        # richer, which would mean the built-out version was never captured.
        orphans = []
        for i, v in enumerate(inks):
            if v is None or v >= SPARSE_INK:
                continue
            window = [x for x in inks[max(0, i - 2):i + 3] if x is not None]
            if max(window) < SPARSE_INK * 2.0:
                orphans.append(slides[i].name)
        # Reported for human review rather than failing the run: section
        # dividers are legitimately sparse, so this list is a review queue,
        # not a defect list.
        if orphans:
            warnings.append(f"{d.name}: {len(orphans)} sparse slide(s) with no "
                            f"richer neighbour -- check these are dividers, not "
                            f"lost content: {', '.join(orphans[:8])}"
                            + (" ..." if len(orphans) > 8 else ""))

        res_s = "/".join(f"{w}x{h}" for w, h in sorted(res)) if res else "?"
        if len(res) > 1:
            problems.append(f"{d.name}: mixed resolutions {res_s}")

        # zip
        zpath = OUTROOT / f"{d.name}.zip"
        zstat = "MISSING"
        if zpath.exists():
            try:
                with zipfile.ZipFile(zpath) as zf:
                    bad = zf.testzip()
                    n_img = sum(1 for n in zf.namelist() if n.endswith(".jpg"))
                    if bad:
                        problems.append(f"{d.name}.zip: corrupt member {bad}")
                        zstat = "CORRUPT"
                    elif n_img != len(slides):
                        problems.append(f"{d.name}.zip: has {n_img} images, "
                                        f"folder has {len(slides)}")
                        zstat = f"{n_img}!={len(slides)}"
                    else:
                        zstat = f"{zpath.stat().st_size/(1024*1024):.1f}M"
            except Exception as exc:
                problems.append(f"{d.name}.zip: unreadable ({exc})")
                zstat = "ERROR"
        else:
            problems.append(f"{d.name}: zip missing")

        if not (d / ".done").exists():
            problems.append(f"{d.name}: no .done marker (incomplete run?)")

        name = d.name if len(d.name) <= 51 else d.name[:48] + "..."
        print(f"{name:<52} {len(slides):>6} {res_s:>11} "
              f"{len(blanks):>5} {dupes:>4} {zstat:>6}")
        if blanks:
            print(f"{'':<52}   blank frames: {', '.join(blanks)}")

    missing = sorted(set(expected) - seen_idx)
    print("-" * 92)
    print(f"lectures present: {len(seen_idx)}/29   total slides: {total_slides}")
    if missing:
        print(f"MISSING LECTURES: {missing}")
        for m in missing:
            problems.append(f"lecture {m} ({expected[m][:40]}) not processed")

    if warnings:
        print(f"\n{len(warnings)} item(s) for review:")
        for w in warnings:
            print(f"  ? {w}")

    if problems:
        print(f"\n!!! {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nAll hard checks passed.")


if __name__ == "__main__":
    main()
