"""
Step 1/2: videos se frames + rolling temporal median backgrounds nikaalo.

Temporal median hi model ka asli input hai - static CCTV pe ye moving
mazdooron ko hata deta hai aur spill ko chhodta hai (analysis/FINDINGS.md).

NOTE: cv2.VideoCapture is machine pe .mkv/.avi nahi kholta, isliye
      video_io.py (ffmpeg pipe) use kar rahe hain.

Usage:
    python extract_frames.py --fps 2 --median-window 2.0
Output:
    dataset/frames_raw/<subset>/<video>/<video>_f00000.jpg   raw frame
    dataset/frames_med/<subset>/<video>/<video>_f00000.jpg   median background
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_io import read_frames, probe

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "dataset" / "raw_videos"
OUT_RAW = ROOT / "dataset" / "frames_raw"
OUT_MED = ROOT / "dataset" / "frames_med"


def rolling_median(arr, half):
    """Har frame ke liye +-half frames ka median. (T,H,W,3) -> (T,H,W,3)"""
    T = len(arr)
    out = np.empty_like(arr)
    for i in range(T):
        lo, hi = max(0, i - half), min(T, i + half + 1)
        out[i] = np.median(arr[lo:hi], axis=0).astype(np.uint8)
    return out


def process(video, subset, target_fps, med_seconds, max_width):
    w, h, src_fps = probe(video)
    scale = max_width / w if w > max_width else None

    # median ke liye zyada dense frames chahiye, saving ke liye kam
    med_fps = max(target_fps, 5.0)
    arr, _ = read_frames(video, fps=med_fps, scale=scale)
    if len(arr) < 5:
        print(f"  {video.name}: only {len(arr)} frames, skip"); return 0

    half = max(1, int(round(med_seconds * med_fps)))
    med = rolling_median(arr, half)

    keep = max(1, int(round(med_fps / target_fps)))
    raw_dir = OUT_RAW / subset / video.stem
    med_dir = OUT_MED / subset / video.stem
    raw_dir.mkdir(parents=True, exist_ok=True)
    med_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for i in range(0, len(arr), keep):
        name = f"{video.stem}_f{i:05d}.jpg"
        cv2.imwrite(str(raw_dir / name), arr[i], [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(med_dir / name), med[i], [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    print(f"  {video.name:16s} {w}x{h}@{src_fps:.0f}  ->  {n} frames "
          f"(median window +-{half} @ {med_fps:.0f}fps)")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--median-window", type=float, default=2.0,
                    help="seconds each side for rolling median")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="HD frames ko isse zyada wide nahi rakhenge")
    args = ap.parse_args()

    total = 0
    for subset in ["train_videos", "test_videos"]:
        print(f"\n{subset}:")
        for v in sorted((VIDEOS / subset).iterdir()):
            if v.suffix.lower() in {".mp4", ".avi", ".mkv"}:
                total += process(v, subset, args.fps, args.median_window, args.max_width)
    print(f"\nTotal: {total} frames")
    print(f"  raw    -> {OUT_RAW}")
    print(f"  median -> {OUT_MED}   <- ismein annotate karna hai")


if __name__ == "__main__":
    main()
