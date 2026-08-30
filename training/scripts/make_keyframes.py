"""
Annotation ka bojh 567 frames se ~18 tak girata hai.

Kyun ye chalta hai: camera static hai AUR spill static hai. 12 me se 9 videos
me poore video ka median background <3% badalta hai (analysis/FINDINGS.md).
Isliye har video ke liye 1 mask kaafi hai; jo videos badalte hain unke liye 3.

Output: dataset/keyframes/<video>__<frame>.jpg   <- yahi annotate karna hai
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MED = ROOT / "dataset" / "frames_med" / "train_videos"
OUT = ROOT / "dataset" / "keyframes"

CHANGE_THRESH = 3.0     # % pixels badle to video ko "dynamic" maano


def change_pct(fs):
    a = cv2.imread(str(fs[0])).astype(np.int16)
    b = cv2.imread(str(fs[-1])).astype(np.int16)
    return (np.abs(a - b).max(axis=2) > 40).mean() * 100


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for d in sorted(MED.iterdir(), key=lambda p: (len(p.name), p.name)):
        fs = sorted(d.glob("*.jpg"))
        pct = change_pct(fs)
        idxs = ([0, len(fs) // 2, len(fs) - 1] if pct > CHANGE_THRESH
                else [len(fs) // 2])
        for i in idxs:
            src = fs[i]
            cv2.imwrite(str(OUT / f"{d.name}__{src.stem}.jpg"),
                        cv2.imread(str(src)), [cv2.IMWRITE_JPEG_QUALITY, 95])
            total += 1
        print(f"  {d.name:10s} change={pct:5.2f}%  -> {len(idxs)} keyframe(s)")
    print(f"\n{total} keyframes -> {OUT}")
    print("Inhe annotate karo (1 class: grain_spill, polygon/mask).")


if __name__ == "__main__":
    main()
