#!/usr/bin/env python3
"""
Threshold diagnostic: quantify what the high=6 cutoff might be discarding.

Re-derives the frame-to-frame diff series for an already-downloaded video's
thumbnails, then reports:
  * the diff distribution
  * "sub-threshold events" -- local peaks with 'lo' < peak <= 'hi', i.e. changes
    too small to trigger a capture but too large to be webcam/cursor noise.
Those are exactly the candidates for missed slides.
"""
import glob
import sys

import numpy as np
from PIL import Image

thumbs = sys.argv[1]
hi = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
lo = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5

files = sorted(glob.glob(f"{thumbs}/f_*.jpg"))
grays = [np.asarray(Image.open(f).convert("L"), dtype=np.float32) for f in files]
diffs = [0.0] + [float(np.mean(np.abs(grays[i] - grays[i - 1])))
                 for i in range(1, len(grays))]
d = np.array(diffs)

print(f"frames={len(files)}")
print("\n--- diff distribution ---")
for p in (50, 75, 90, 95, 98, 99, 99.5):
    print(f"  p{p:<5} = {np.percentile(d, p):8.3f}")
print(f"  max    = {d.max():8.3f}")

bands = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 30), (30, 1e9)]
print("\n--- frames per diff band ---")
for a, b in bands:
    n = int(((d > a) & (d <= b)).sum())
    label = f"{a}-{b}" if b < 1e8 else f">{a}"
    print(f"  {label:>8} : {n:5d}")

# local peaks strictly between lo and hi = candidate missed transitions
print(f"\n--- sub-threshold local peaks ({lo} < peak <= {hi}) ---")
cands = []
for i in range(1, len(d) - 1):
    if lo < d[i] <= hi and d[i] >= d[i - 1] and d[i] >= d[i + 1]:
        cands.append((i, float(d[i])))
# suppress peaks within 3s of a supra-threshold spike (already captured)
supra = {i for i in range(len(d)) if d[i] > hi}
cands = [(i, v) for i, v in cands if not any(abs(i - s) <= 3 for s in supra)]
print(f"count={len(cands)}")
for i, v in cands:
    print(f"  t={i:5d}s ({i//60}:{i%60:02d})  diff={v:6.2f}")
