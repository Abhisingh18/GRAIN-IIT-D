"""
Verify: (a) camera fixed hai ya nahi, (b) temporal median se moving log hat-te
hain ya nahi. Agar dono TRUE hain to temporal background ko model ka extra
input channel banaya ja sakta hai - spill static hai, mazdoor nahi.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from video_io import read_frames

ROOT = Path(__file__).resolve().parents[1]
VID = ROOT / "dataset" / "raw_videos"
OUT = ROOT / "analysis" / "static_check"
OUT.mkdir(parents=True, exist_ok=True)


def analyse(rel):
    path = VID / rel
    w, _, _ = __import__("video_io").probe(path)
    sc = 640.0 / w if w > 900 else None      # HD ko downscale, SD as-is
    arr, src_fps = read_frames(path, fps=5, scale=sc)
    if len(arr) < 8:
        print(f"{path.name}: only {len(arr)} frames"); return

    median = np.median(arr, axis=0).astype(np.uint8)

    # (a) camera motion: pehla vs aakhri frame phase correlation
    g = lambda f: cv2.cvtColor(cv2.resize(f, (320, 240)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(g(arr[0]), g(arr[-1]))

    # (b) har frame ka median se deviation
    dev = np.abs(arr.astype(np.int16) - median.astype(np.int16)).mean(axis=(1, 2, 3))
    moving_pct = (np.abs(arr.astype(np.int16) - median.astype(np.int16)).max(axis=3) > 40).mean() * 100

    print(f"{path.name:16s} n={len(arr):3d}  cam_shift=({dx:+5.2f},{dy:+5.2f})px  "
          f"dev mean={dev.mean():5.2f} max={dev.max():5.2f}  moving_px={moving_pct:5.2f}%")

    mid = arr[len(arr) // 2]
    h = 420
    s = h / mid.shape[0]
    a = cv2.resize(mid, None, fx=s, fy=s)
    b = cv2.resize(median, None, fx=s, fy=s)
    cv2.putText(a, "SINGLE FRAME", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(b, "TEMPORAL MEDIAN", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(OUT / f"{path.stem}_cmp.jpg"), np.hstack([a, b]), [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["train_videos/train_7.mkv", "train_videos/train_1.avi",
                               "train_videos/train_12.mkv", "test_videos/test_1.mkv",
                               "test_videos/test_3.mkv"]):
        analyse(t)
    print(f"\nimages -> {OUT}")
