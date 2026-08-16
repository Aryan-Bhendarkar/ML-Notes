#!/usr/bin/env python3
"""
Extract every slide as a full-resolution screenshot from one lecture video.

Windows-native port of the validated process_lecture.sh pipeline:
  download (video-only, <=1080p)  ->  1fps low-res thumbnails
  -> diff-based slide-change detection  ->  full-res frame grab at each change
  -> cleanup  ->  zip

Usage:
  python process_lecture.py 1              # one lecture (line 1 of playlist_meta.tsv)
  python process_lecture.py 2 5            # inclusive range, sequential, paced
  python process_lecture.py all            # every lecture, sequential, paced
  python process_lecture.py 7 --high 4.5   # retune detection thresholds
  python process_lecture.py 7 --force      # redo a lecture already marked .done
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
META = ROOT / "playlist_meta.tsv"
OUTROOT = ROOT / "output"
WORKROOT = ROOT / "work"
STATUS = ROOT / "status.log"
DETECT = ROOT / "detect_changes.py"

# Boilerplate repeated on all 29 titles; stripped so folder names read like
# "Lecture_01 - Module 1 Supervised Learning Part 1".
BOILERPLATE = re.compile(r"\s*\|\s*Amazon ML Summer School 2026\s*$", re.I)


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sanitize(title):
    t = BOILERPLATE.sub("", title)
    t = re.sub(r'[\\/:*?"<>|]', "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def read_meta():
    rows = []
    for raw in META.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) < 4:
            continue
        idx, vid, title, dur = parts[0], parts[1], parts[2], parts[3]
        rows.append({
            "idx": int(idx),
            "id": vid,
            "title": title,
            "duration": int(dur) if dur.isdigit() else 0,
        })
    return rows


def run(cmd, **kw):
    """Run a command, streaming nothing; return CompletedProcess with captured text."""
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw
    )


def hms(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def download(video_id, workdir, cookies_from_browser=None, retries=3):
    """Download video-only stream up to 1080p. Returns path to the media file."""
    out_tmpl = str(workdir / "video.%(ext)s")
    base = ["yt-dlp"]
    if cookies_from_browser:
        base += ["--cookies-from-browser", cookies_from_browser]
    base += [
        "--sleep-requests", "2",
        "--sleep-interval", "3",
        "--max-sleep-interval", "8",
        "--retries", "10",
        "--fragment-retries", "10",
        "--no-warnings",
        "--newline",
        "-f", "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]/best[height<=1080]",
        "-o", out_tmpl,
        f"https://www.youtube.com/watch?v={video_id}",
    ]

    for attempt in range(1, retries + 1):
        proc = run(base)
        found = sorted(workdir.glob("video.*"))
        found = [f for f in found if f.suffix.lower() != ".part"]
        if found and found[0].stat().st_size > 0:
            return found[0]
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        log(f"    download attempt {attempt}/{retries} failed: " + " | ".join(tail))
        blocked = "bot" in (proc.stderr or "").lower() or "sign in" in (proc.stderr or "").lower()
        if blocked:
            log("    !!! YouTube is challenging this request (bot check).")
        if attempt < retries:
            backoff = 30 * attempt
            log(f"    backing off {backoff}s before retry ...")
            time.sleep(backoff)
    return None


def probe_dims(path):
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)])
    return p.stdout.strip()


def process(row, args):
    idx, vid, duration = row["idx"], row["id"], row["duration"]
    safe = sanitize(row["title"])
    lecdir = f"Lecture_{idx:02d} - {safe}"
    outdir = OUTROOT / lecdir
    donefile = outdir / ".done"

    if donefile.exists() and not args.force:
        log(f"SKIP (already done): {lecdir}")
        return True

    log(f"=== START {lecdir} (id={vid}, duration={duration}s / {hms(duration)}) ===")
    t0 = time.time()

    workdir = WORKROOT / f"lec{idx:02d}"
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    thumbs = workdir / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    if outdir.exists() and args.force:
        for old in outdir.glob("slide_*.jpg"):
            old.unlink()
        if donefile.exists():
            donefile.unlink()
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1. download -------------------------------------------------------
    log(f"  [1/5] Downloading video {idx} ...")
    video = download(vid, workdir, args.cookies_from_browser)
    if video is None:
        log(f"  !!! DOWNLOAD FAILED for {lecdir}")
        return False
    size_mb = video.stat().st_size / (1024 * 1024)
    dims = probe_dims(video)
    log(f"        got {video.name} ({size_mb:.1f} MiB, {dims})")

    # ---- 2. 1fps low-res thumbnails for diff analysis ----------------------
    log("  [2/5] Extracting 1fps thumbnails for diff analysis ...")
    p = run(["ffmpeg", "-nostdin", "-i", str(video), "-vf", "fps=1,scale=160:90",
             "-q:v", "5", str(thumbs / "f_%05d.jpg"), "-hide_banner", "-loglevel", "error"])
    nframes = len(list(thumbs.glob("f_*.jpg")))
    if nframes == 0:
        log(f"  !!! THUMBNAIL EXTRACTION FAILED: {p.stderr.strip()[:400]}")
        return False
    log(f"        got {nframes} thumbnails (video is {duration}s)")
    if duration and nframes < duration * 0.9:
        log(f"  !!! WARNING: only {nframes} frames for a {duration}s video "
            f"— download may be truncated.")

    # ---- 3. detect slide changes ------------------------------------------
    log(f"  [3/5] Detecting slide changes (high={args.high} low={args.low} "
        f"max_gap={args.max_gap}) ...")
    times_file = workdir / "capture_times.txt"
    p = run([sys.executable, str(DETECT), str(thumbs),
             str(args.high), str(args.low), str(args.max_gap)])
    if p.returncode != 0:
        log(f"  !!! DETECTION FAILED: {p.stderr.strip()[:400]}")
        return False
    times = [int(x) for x in p.stdout.split() if x.strip().isdigit()]
    times_file.write_text("\n".join(str(t) for t in times), encoding="utf-8")
    log(f"        {p.stderr.strip()}")
    log(f"        detected {len(times)} capture points")

    # ---- 4. full-resolution frame grabs from the ORIGINAL video ------------
    log(f"  [4/5] Extracting {len(times)} full-resolution slides ...")
    manifest = []
    failed = 0
    for i, t in enumerate(times, start=1):
        dest = outdir / f"slide_{i:03d}.jpg"
        run(["ffmpeg", "-nostdin", "-ss", str(t), "-i", str(video),
             "-frames:v", "1", "-q:v", "2", str(dest),
             "-hide_banner", "-loglevel", "error", "-y"])
        if not dest.exists() or dest.stat().st_size == 0:
            failed += 1
            log(f"        !! failed to grab frame at t={t}s ({hms(t)})")
            if dest.exists():
                dest.unlink()
        else:
            manifest.append((dest.name, t))
        if i % 25 == 0:
            log(f"        ... {i}/{len(times)}")

    if not manifest:
        log(f"  !!! NO SLIDES EXTRACTED for {lecdir}")
        return False

    # timestamp map — lets the later note-taking pass jump back to the recording
    with open(outdir / "timestamps.txt", "w", encoding="utf-8") as fh:
        fh.write(f"# {lecdir}\n")
        fh.write(f"# source: https://www.youtube.com/watch?v={vid}\n")
        fh.write("# filename\tseconds\ttimestamp\n")
        for name, t in manifest:
            fh.write(f"{name}\t{t}\t{hms(t)}\n")

    first = outdir / manifest[0][0]
    log(f"        extracted {len(manifest)} slides"
        + (f" ({failed} grabs failed)" if failed else "")
        + f", native resolution {probe_dims(first)}")

    # ---- 5. cleanup + zip --------------------------------------------------
    shutil.rmtree(workdir, ignore_errors=True)

    zip_path = OUTROOT / f"{lecdir}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for f in sorted(outdir.iterdir()):
            if f.name == ".done":
                continue
            zf.write(f, arcname=f"{lecdir}/{f.name}")
    log(f"  [5/5] Zipped: {zip_path.name} ({zip_path.stat().st_size/(1024*1024):.1f} MiB)")

    donefile.write_text(
        f"slides={len(manifest)}\nhigh={args.high}\nlow={args.low}\n"
        f"max_gap={args.max_gap}\nfinished={datetime.now():%Y-%m-%d %H:%M:%S}\n",
        encoding="utf-8")
    log(f"=== DONE {lecdir} — {len(manifest)} slides in {time.time()-t0:.0f}s ===")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", help="line number 1-29, or 'all'")
    ap.add_argument("end", nargs="?", help="optional inclusive end line number")
    ap.add_argument("--high", type=float, default=6.0)
    ap.add_argument("--low", type=float, default=3.0)
    ap.add_argument("--max-gap", type=int, default=90)
    ap.add_argument("--force", action="store_true", help="reprocess even if .done exists")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="fallback if YouTube starts bot-challenging, e.g. 'firefox'")
    ap.add_argument("--pace", type=int, default=15,
                    help="seconds to wait between lectures (default 15)")
    args = ap.parse_args()

    rows = read_meta()
    if not rows:
        log("playlist_meta.tsv is empty or missing")
        sys.exit(1)

    if args.start == "all":
        todo = rows
    else:
        s = int(args.start)
        e = int(args.end) if args.end else s
        todo = [r for r in rows if s <= r["idx"] <= e]
    if not todo:
        log(f"No lectures matched {args.start}..{args.end}")
        sys.exit(1)

    OUTROOT.mkdir(exist_ok=True)
    WORKROOT.mkdir(exist_ok=True)

    ok, bad = [], []
    for n, row in enumerate(todo):
        try:
            (ok if process(row, args) else bad).append(row["idx"])
        except KeyboardInterrupt:
            log("Interrupted by user.")
            raise
        except Exception as exc:
            log(f"  !!! ERROR on lecture {row['idx']}: {exc!r}")
            bad.append(row["idx"])
        if n < len(todo) - 1:
            time.sleep(args.pace)  # gentle pacing so YouTube doesn't flag the run

    log(f"BATCH COMPLETE — ok={len(ok)} {ok} failed={len(bad)} {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
