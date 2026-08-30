"""
Annotated keyframe masks ko us video ke saare frames pe propagate karo.

Chalta kyun hai: camera static hai aur spill static hai, isliye ek keyframe ka
mask us video ke baaki frames pe bhi valid hai. Jo videos badalte hain
(train_1, train_7, train_12) unke 3 keyframes hain - har frame ko uske
sabse nazdeek keyframe ka mask milta hai.

Input : dataset/keyframes/<video>__<frame>.txt   (YOLO-seg, tumne annotate kiya)
Output: dataset/frames_med/train_videos/<video>/<frame>.txt

Usage:
    python propagate_masks.py
"""
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEY = ROOT / "dataset" / "keyframes"
MED = ROOT / "dataset" / "frames_med" / "train_videos"


def frame_idx(stem):
    m = re.search(r"_f(\d+)$", stem)
    return int(m.group(1)) if m else 0


def main():
    labels = sorted(KEY.glob("*.txt"))
    if not labels:
        print(f"Koi .txt label nahi mila {KEY} me.")
        print("Pehle keyframes annotate karo (YOLO-seg format export).")
        return

    by_video = defaultdict(list)
    for lb in labels:
        video, frame = lb.stem.split("__", 1)
        by_video[video].append((frame_idx(frame), lb))

    total = 0
    for video, keys in sorted(by_video.items()):
        keys.sort()
        vd = MED / video
        if not vd.exists():
            print(f"  ! {video}: frames folder nahi mila, skip"); continue
        n = 0
        for img in sorted(vd.glob("*.jpg")):
            i = frame_idx(img.stem)
            _, src = min(keys, key=lambda k: abs(k[0] - i))   # nazdeek keyframe
            shutil.copy2(src, img.with_suffix(".txt"))
            n += 1
        print(f"  {video:10s} {len(keys)} keyframe(s) -> {n} frames")
        total += n

    print(f"\n{total} labels propagate hue.")
    print("Ab: python split_dataset.py --val <video stems>")


if __name__ == "__main__":
    main()
