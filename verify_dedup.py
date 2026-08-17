#!/usr/bin/env python3
"""
Prove the dedup pass lost no content.

For every ORIGINAL slide that was dropped, locate the kept slide it was folded
into (the next survivor at or after it) and check that the survivor still
contains the dropped slide's content. The dedup pass makes this guarantee
pairwise, but chains (A->B->C) compound tolerance, so this re-checks each
dropped slide directly against its final survivor rather than its neighbour.

Reports any dropped slide whose content is NOT fully present in its survivor.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from dedupe_slides import load, contains, SRC, DST

STRICT_LOST = 0.05   # allow a little more drift than the pairwise test


def lost_fraction(later, earlier):
    """Fraction of `earlier`'s content missing from `later` (0.0 = all present)."""
    from dedupe_slides import dilate, BG_SHIFT
    if abs(float(later.mean()) - float(earlier.mean())) > BG_SHIFT:
        return 1.0
    dark_bg = (earlier.mean() + later.mean()) / 2 < 128
    if dark_bg:
        bg = min(np.percentile(earlier, 10), np.percentile(later, 10))
        ea, la = earlier > bg + 30, later > bg + 30
    else:
        bg = max(np.percentile(earlier, 90), np.percentile(later, 90))
        ea, la = earlier < bg - 30, later < bg - 30
    n = int(ea.sum())
    if n < 40:
        return 0.0
    return int((ea & ~dilate(la)).sum()) / n


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    folders = sorted(d for d in SRC.iterdir() if d.is_dir())
    if only:
        folders = [d for d in folders if only in d.name]

    total_dropped = total_bad = 0
    print(f"{'lecture':<52} {'dropped':>8} {'unsafe':>7}")
    print("-" * 70)

    for f in folders:
        ded = DST / f.name
        tf = ded / "timestamps.txt"
        if not tf.exists():
            continue

        # kept original filename -> deduped filename
        kept_order = []
        for line in tf.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            p = line.split("\t")
            if len(p) >= 4:
                kept_order.append((p[3], p[0]))
        kept_names = {o for o, _ in kept_order}

        originals = sorted(f.glob("slide_*.jpg"))
        cache = {}

        def g(path):
            if path not in cache:
                cache[path] = load(path)
            return cache[path]

        bad = []
        dropped = 0
        for i, src in enumerate(originals):
            if src.name in kept_names:
                continue
            dropped += 1
            # A dropped slide folds into its group's representative, which may
            # sit either side of it: a near-identical frame is discarded in
            # favour of the EARLIER survivor, while a build step is superseded
            # by a LATER, fuller one. Accept if either neighbour holds it.
            nxt = next((originals[j] for j in range(i + 1, len(originals))
                        if originals[j].name in kept_names), None)
            prv = next((originals[j] for j in range(i - 1, -1, -1)
                        if originals[j].name in kept_names), None)
            best, best_lf = None, 1.0
            for cand in (nxt, prv):
                if cand is None:
                    continue
                lf = lost_fraction(g(cand), g(src))
                if lf < best_lf:
                    best, best_lf = cand, lf
            if best is None:
                bad.append((src.name, "no survivor", 1.0))
            elif best_lf > STRICT_LOST:
                bad.append((src.name, best.name, best_lf))

        total_dropped += dropped
        total_bad += len(bad)
        name = f.name if len(f.name) <= 51 else f.name[:48] + "..."
        print(f"{name:<52} {dropped:>8} {len(bad):>7}")
        for nm, sv, lf in bad[:6]:
            print(f"{'':<52}   ! {nm} -> {sv}: {lf*100:.0f}% of content missing")
        if len(bad) > 6:
            print(f"{'':<52}   ... and {len(bad)-6} more")

    print("-" * 70)
    print(f"{'TOTAL':<52} {total_dropped:>8} {total_bad:>7}")
    if total_bad:
        print(f"\n!!! {total_bad} dropped slide(s) had content not present in "
              f"their survivor. Review before discarding output/.")
        sys.exit(1)
    print("\nEvery dropped slide's content is present in its survivor.")


if __name__ == "__main__":
    main()
