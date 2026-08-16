#!/usr/bin/env python3
"""Detect slide-capture timestamps from a sequence of 1fps low-res thumbnails.

v2 -- segment-based.

The v1 algorithm captured the moment the picture *settled* after a change. On
decks that animate (title lands first, body fades in afterwards) that yields a
title-only frame, and any body fade gentle enough to stay under `high` is then
never captured at all -- the finished slide is lost.

v2 instead splits the video into "moving" and "stable" stretches and captures
BOTH ENDS of every stable stretch:

  * the head  -- the slide as it first settles (matches v1 behaviour)
  * the tail  -- the last frame before the next change, i.e. the slide in its
                 most complete state, after every build/animation has finished

The tail capture is what makes this robust: a gradual reveal that never trips
the threshold still gets photographed in full, because the tail of the stable
segment is by definition the final state of that slide.

Long stable stretches additionally get filled every `max_gap` seconds.
"""
import sys
import glob
import numpy as np
from PIL import Image


def main():
    thumbs_dir = sys.argv[1]
    high = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    low = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    max_gap = int(sys.argv[4]) if len(sys.argv) > 4 else 90

    files = sorted(glob.glob(f"{thumbs_dir}/f_*.jpg"))
    if not files:
        print("NO_FRAMES", file=sys.stderr)
        sys.exit(1)

    grays = []
    for f in files:
        img = Image.open(f).convert("L")
        grays.append(np.asarray(img, dtype=np.float32))

    n = len(grays)
    diffs = [0.0]
    for i in range(1, n):
        diffs.append(float(np.mean(np.abs(grays[i] - grays[i - 1]))))

    # ---- classify each frame as moving or still -----------------------------
    # A frame is "moving" if it differs sharply from its predecessor (> high).
    # Once moving, we stay moving until the picture has been quiet (< low) for
    # two consecutive frames -- this rides out multi-second transitions and
    # animation builds without cutting them into fragments.
    moving = [False] * n
    unstable = False
    low_streak = 0
    for i in range(1, n):
        d = diffs[i]
        if d > high:
            unstable = True
            low_streak = 0
        elif unstable:
            if d < low:
                low_streak += 1
                if low_streak >= 2:
                    unstable = False
                    low_streak = 0
            else:
                low_streak = 0
        moving[i] = unstable

    # ---- group the still frames into stable segments -----------------------
    segments = []
    start = None
    for i in range(n):
        if not moving[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None:
        segments.append((start, n - 1))

    # ---- capture head + tail (+ periodic fills) of every segment -----------
    capture_idx = set()
    for a, b in segments:
        capture_idx.add(a)
        capture_idx.add(b)          # <-- the completed state of this slide
        t = a + max_gap
        while t < b:
            capture_idx.add(t)
            t += max_gap

    capture_idx.add(0)
    capture_idx.add(n - 1)
    capture_idx = sorted(i for i in capture_idx if 0 <= i < n)

    n_tails = sum(1 for a, b in segments if b != a)
    print(f"numFrames={n} numSegments={len(segments)} "
          f"numTails={n_tails} numCaptures={len(capture_idx)}", file=sys.stderr)
    for idx in capture_idx:
        print(idx)


if __name__ == "__main__":
    main()
