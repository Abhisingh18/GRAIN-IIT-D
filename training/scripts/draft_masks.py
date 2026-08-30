"""
18 keyframes ke liye DRAFT spill masks - classical color/texture heuristic.

Ye final labels NAHI hain. Insaan inko theek karega. Logic:
  spill  = bright + low saturation + smooth (granular parat)
  sack   = tan/brown + strong periodic edges
  floor  = grey + darker + smooth

Output:
  dataset/keyframes/<name>.txt        YOLO-seg polygons
  analysis/draft_masks_preview.jpg    overlay contact sheet
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KEY = ROOT / "dataset" / "keyframes"


def spill_mask(bgr, bright_pct=72, sat_max=60, edge_max=0.22, min_area=0.004):
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (640, int(640 * h / w)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    V, S = hsv[..., 2].astype(np.float32), hsv[..., 1].astype(np.float32)

    # 1) bright: image ke apne distribution ke hisaab se
    thr = np.percentile(V, bright_pct)
    bright = V > thr

    # 2) low saturation - grain fiika hota hai, boriyan/kapde rangeen
    lowsat = S < sat_max

    # 3) smooth - boriyon ke beech tez edges hote hain, grain me nahi
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(cv2.GaussianBlur(g, (5, 5), 0), cv2.CV_32F)
    edge_dens = cv2.blur(np.abs(edges), (25, 25))
    edge_dens /= (edge_dens.max() + 1e-6)
    smooth = edge_dens < edge_max

    m = (bright & lowsat & smooth).astype(np.uint8)

    # cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    # chhote blobs hatao
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] / m.size >= min_area:
            out[lab == i] = 1
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)


def to_yolo_seg(mask, cls=0, eps_frac=0.004):
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in cnts:
        if cv2.contourArea(c) < 0.002 * h * w:
            continue
        approx = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= w
        pts[:, 1] /= h
        lines.append(f"{cls} " + " ".join(f"{x:.6f}" for x in pts.flatten()))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bright-pct", type=float, default=72)
    ap.add_argument("--sat-max", type=float, default=60)
    ap.add_argument("--edge-max", type=float, default=0.22)
    args = ap.parse_args()

    imgs = sorted(KEY.glob("*.jpg"))
    tiles, total_poly = [], 0
    for f in imgs:
        im = cv2.imread(str(f))
        m = spill_mask(im, args.bright_pct, args.sat_max, args.edge_max)
        lines = to_yolo_seg(m)
        f.with_suffix(".txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        total_poly += len(lines)

        vis = cv2.resize(im, (300, 225))
        mk = cv2.resize(m, (300, 225), interpolation=cv2.INTER_NEAREST)
        ov = vis.copy()
        ov[mk > 0] = (0.45 * ov[mk > 0] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.putText(ov, f"{f.stem.split('__')[0]} {mk.mean()*100:.0f}% n={len(lines)}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        tiles.append(ov)
        print(f"  {f.stem:32s} area={m.mean()*100:5.1f}%  polys={len(lines)}")

    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(str(ROOT / "analysis" / "draft_masks_preview.jpg"), np.vstack(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\n{len(imgs)} keyframes, {total_poly} polygons")
    print("preview -> analysis/draft_masks_preview.jpg")


if __name__ == "__main__":
    main()
