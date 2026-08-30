"""
Step 2: video-wise train/val split (frame-wise NAHI - leakage se bachne ke liye).

Annotated frames + YOLO .txt labels ko dataset/images/{train,val} aur
dataset/annotations/{train,val} me daalta hai.

Usage:
    python split_dataset.py --val train_3 train_8 train_11
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "dataset" / "frames_med" / "train_videos"   # median backgrounds pe train karte hain
IMG_OUT = ROOT / "dataset" / "images"
LBL_OUT = ROOT / "dataset" / "annotations"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="+", required=True,
                    help="val ke liye video stems, e.g. train_3 train_8 train_11")
    args = ap.parse_args()
    val_set = set(args.val)

    counts = {"train": 0, "val": 0}
    for vid_dir in sorted(FRAMES.iterdir()):
        if not vid_dir.is_dir():
            continue
        split = "val" if vid_dir.name in val_set else "train"
        (IMG_OUT / split).mkdir(parents=True, exist_ok=True)
        (LBL_OUT / split).mkdir(parents=True, exist_ok=True)

        for img in sorted(vid_dir.glob("*.jpg")):
            lbl = img.with_suffix(".txt")
            if not lbl.exists():
                continue          # bina annotate kiya frame skip
            shutil.copy2(img, IMG_OUT / split / img.name)
            shutil.copy2(lbl, LBL_OUT / split / lbl.name)
            counts[split] += 1
        print(f"{vid_dir.name:16s} -> {split}")

    total = counts["train"] + counts["val"]
    print(f"\ntrain={counts['train']}  val={counts['val']}  total={total}")
    if total:
        print(f"split ratio = {counts['train']/total:.2f} / {counts['val']/total:.2f}")


if __name__ == "__main__":
    main()
