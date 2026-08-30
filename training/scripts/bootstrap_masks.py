"""
Annotation bootstrap: CLIPSeg se zero-shot spill masks propose karo.

Ye FINAL labels nahi hain - ye human verification ka starting point hain.
Insaan inko CVAT/Roboflow me import karke theek karega, phir train karenge.

Usage:
    python bootstrap_masks.py --preview        # sirf 6 frames pe dekho
    python bootstrap_masks.py --all            # saare train frames
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

ROOT = Path(__file__).resolve().parents[2]
MED = ROOT / "dataset" / "frames_med" / "train_videos"
OUT = ROOT / "dataset" / "bootstrap"

PROMPTS = [
    "grain spilled on the floor",
    "a pile of loose white grain on the ground",
    "scattered wheat grain on concrete floor",
]
NEG_PROMPTS = [
    "stacked jute sacks",
    "concrete floor",
    "a person",
    "wooden pallet",
]


def load_model(device):
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined").to(device).eval()
    return proc, model


@torch.no_grad()
def predict(proc, model, bgr, device):
    """spill probability map (H,W) float 0-1"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    texts = PROMPTS + NEG_PROMPTS
    inputs = proc(text=texts, images=[rgb] * len(texts),
                  padding=True, return_tensors="pt").to(device)
    logits = model(**inputs).logits            # (T,352,352)
    if logits.ndim == 2:
        logits = logits[None]
    pos = logits[:len(PROMPTS)].mean(0)
    neg = logits[len(PROMPTS):].max(0).values
    score = torch.sigmoid(pos - neg)
    m = score.cpu().numpy().astype(np.float32)
    return cv2.resize(m, (bgr.shape[1], bgr.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda:10")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    proc, model = load_model(device)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.preview:
        vids = ["train_7", "train_4", "train_2", "train_6", "train_11", "train_3"]
        tiles = []
        for v in vids:
            fs = sorted((MED / v).glob("*.jpg"))
            f = fs[len(fs) // 2]
            im = cv2.imread(str(f))
            prob = predict(proc, model, im, device)
            mask = (prob > args.thresh).astype(np.uint8)

            vis = cv2.resize(im, (320, 240))
            mk = cv2.resize(mask, (320, 240), interpolation=cv2.INTER_NEAREST)
            ov = vis.copy()
            ov[mk > 0] = (0.45 * ov[mk > 0] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.putText(ov, f"{v} {mk.mean()*100:.0f}%", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(np.hstack([vis, ov]))
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
        cv2.imwrite(str(ROOT / "analysis" / "clipseg_preview.jpg"), np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("preview ->", ROOT / "analysis" / "clipseg_preview.jpg")
        return

    if args.all:
        n = 0
        for vd in sorted(MED.iterdir()):
            od = OUT / vd.name
            od.mkdir(parents=True, exist_ok=True)
            for f in sorted(vd.glob("*.jpg")):
                im = cv2.imread(str(f))
                prob = predict(proc, model, im, device)
                cv2.imwrite(str(od / f"{f.stem}_prob.png"),
                            (prob * 255).astype(np.uint8))
                n += 1
            print(f"  {vd.name}: {len(list(vd.glob('*.jpg')))} frames")
        print(f"\n{n} probability maps -> {OUT}")


if __name__ == "__main__":
    main()
