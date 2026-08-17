#!/usr/bin/env python3
"""
Collapse each lecture's slides down to a minimal, complete set.

The capture pipeline deliberately over-captures: every stable stretch is
photographed at both ends, so an animated slide appears first as a title-only
frame and again once its body has filled in. That is the right trade for
completeness, but it roughly doubles the image count for the notes pass.

This pass removes redundancy WITHOUT losing content, using two rules:

  1. near-identical neighbours          -> keep one
  2. later frame CONTAINS the earlier   -> keep only the later (fuller) one

Rule 2 is the important one, and it is checked rather than assumed. A frame B
"contains" A only if almost none of A's content pixels are missing from B --
so a progressive build (title -> +bullet -> +bullet) collapses to its final
state, while a slide whose content was REPLACED (annotations cleared, a
different diagram swapped in) keeps both frames.

Non-destructive: writes to a separate tree and leaves `output/` untouched.
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "output"
DST = ROOT / "slides_deduped"

W, H = 640, 360
SAME = 2.0        # mean abs diff below this => visually the same frame
GROUP = 45.0      # above this => a wholly different frame, never merge
BG_SHIFT = 25.0   # background brightness change => different slide template
LOST_MAX = 0.02   # >2% of content pixels vanished => not a superset

# Containment, not GROUP, is the real safeguard here: a genuinely different
# slide cannot retain the previous slide's text, so it fails contains(). GROUP
# and BG_SHIFT only rule out comparisons where the containment test itself
# would be meaningless (e.g. a light content slide following a dark divider,
# where "ink" flips polarity).


def load(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((W, H)), dtype=np.float32)


def dilate(mask):
    """1-pixel dilation via shifted ORs (no scipy dependency).

    Absorbs anti-aliasing and sub-pixel jitter, so a glyph that shifted by a
    hair between two frames is not scored as 'content that disappeared'.
    """
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def contains(later, earlier):
    """True if `later` retains essentially all of `earlier`'s content."""
    if abs(float(later.mean()) - float(earlier.mean())) > BG_SHIFT:
        return False            # background changed -> different slide template

    # One threshold for BOTH frames, anchored to the shared background level.
    # Deriving it per-frame would move the cutoff as content is added, flipping
    # edge pixels between categories and faking 'lost' content.
    dark_bg = (earlier.mean() + later.mean()) / 2 < 128
    if dark_bg:
        bg = min(np.percentile(earlier, 10), np.percentile(later, 10))
        ea, la = earlier > bg + 30, later > bg + 30
    else:
        bg = max(np.percentile(earlier, 90), np.percentile(later, 90))
        ea, la = earlier < bg - 30, later < bg - 30

    n = int(ea.sum())
    if n < 40:                      # earlier frame is essentially empty
        return True
    lost = int((ea & ~dilate(la)).sum()) / n
    return lost <= LOST_MAX


def dedupe_lecture(folder, dst_root, verbose=False):
    slides = sorted(folder.glob("slide_*.jpg"))
    if not slides:
        return None

    # original timestamps, keyed by filename
    ts = {}
    tf = folder / "timestamps.txt"
    if tf.exists():
        for line in tf.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            p = line.split("\t")
            if len(p) >= 3:
                ts[p[0]] = (p[1], p[2])

    grays = [load(s) for s in slides]

    keep = []            # indices retained
    cur = 0
    for nxt in range(1, len(slides)):
        d = float(np.mean(np.abs(grays[nxt] - grays[cur])))
        if d < SAME:
            # identical frame -- prefer whichever holds more content
            if contains(grays[nxt], grays[cur]) and not contains(grays[cur], grays[nxt]):
                cur = nxt
            continue
        if d < GROUP and contains(grays[nxt], grays[cur]):
            cur = nxt            # build step: the later frame supersedes
            continue
        keep.append(cur)
        cur = nxt
    keep.append(cur)

    out = dst_root / folder.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    lines = ["# " + folder.name]
    src_head = (folder / "timestamps.txt")
    if src_head.exists():
        for line in src_head.read_text(encoding="utf-8").splitlines()[:2]:
            if line.startswith("# source:"):
                lines.append(line)
    lines.append("# filename\tseconds\ttimestamp\toriginal")

    for i, idx in enumerate(keep, start=1):
        src = slides[idx]
        name = f"slide_{i:03d}.jpg"
        shutil.copy2(src, out / name)
        sec, hms = ts.get(src.name, ("", ""))
        lines.append(f"{name}\t{sec}\t{hms}\t{src.name}")
    (out / "timestamps.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zp = dst_root / f"{folder.name}.zip"
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for f in sorted(out.iterdir()):
            zf.write(f, arcname=f"{folder.name}/{f.name}")

    return len(slides), len(keep), zp.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="substring filter, e.g. 'Lecture_06'")
    args = ap.parse_args()

    DST.mkdir(exist_ok=True)
    folders = sorted(d for d in SRC.iterdir() if d.is_dir())
    if args.only:
        folders = [d for d in folders if args.only in d.name]
    if not folders:
        sys.exit("no lecture folders matched")

    tot_before = tot_after = tot_bytes = 0
    print(f"{'lecture':<52} {'before':>7} {'after':>6} {'cut':>6} {'zip':>7}")
    print("-" * 82)
    for d in folders:
        r = dedupe_lecture(d, DST)
        if r is None:
            print(f"{d.name:<52}   (no slides)")
            continue
        before, after, nbytes = r
        tot_before += before
        tot_after += after
        tot_bytes += nbytes
        name = d.name if len(d.name) <= 51 else d.name[:48] + "..."
        print(f"{name:<52} {before:>7} {after:>6} {100*(1-after/before):>5.0f}% "
              f"{nbytes/(1024*1024):>6.1f}M")

    print("-" * 82)
    print(f"{'TOTAL':<52} {tot_before:>7} {tot_after:>6} "
          f"{100*(1-tot_after/tot_before):>5.0f}% {tot_bytes/(1024*1024):>6.1f}M")
    print(f"\nWritten to: {DST}")
    print("Originals in output/ are untouched.")


if __name__ == "__main__":
    main()
