#!/usr/bin/env python3
"""Build a labelled contact sheet from a list of images (handles ragged sizes).

Usage: contact_sheet.py <out.png> <cols> <cell_w> <label_prefix> <start_num> <img1> ...
"""
import sys
from PIL import Image, ImageDraw, ImageFont

out, cols, cell_w, prefix = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
start = int(sys.argv[5])
paths = sys.argv[6:]

try:
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 30)
except OSError:
    font = ImageFont.load_default()

imgs = []
for p in paths:
    im = Image.open(p).convert("RGB")
    w, h = im.size
    im = im.resize((cell_w, max(1, round(h * cell_w / w))), Image.LANCZOS)
    imgs.append(im)

cell_h = max(i.height for i in imgs)
rows = (len(imgs) + cols - 1) // cols
pad = 6
sheet = Image.new("RGB", (cols * (cell_w + pad) + pad, rows * (cell_h + pad) + pad), "black")
draw = ImageDraw.Draw(sheet)

for n, im in enumerate(imgs):
    r, c = divmod(n, cols)
    x = pad + c * (cell_w + pad)
    y = pad + r * (cell_h + pad)
    sheet.paste(im, (x, y))
    label = f"{prefix}{n + start}"
    tw = draw.textlength(label, font=font)
    draw.rectangle([x, y, x + tw + 14, y + 40], fill="black")
    draw.text((x + 7, y + 4), label, fill="yellow", font=font)

sheet.save(out)
print(f"{out}  {sheet.size[0]}x{sheet.size[1]}  ({len(imgs)} cells)")
