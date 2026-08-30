"""
Step 4: Testing - test videos pe inference + post-processing.

Model rolling temporal median pe train hua hai, isliye inference me bhi wahi
representation banate hain (moving mazdoor hat jaate hain).

Video I/O ffmpeg CLI se hai - is machine pe cv2 ka video backend toota hai.

Output:
  predictions/<video>.json          per-frame severity, confidence
  predictions/summary.csv           inference time, FPS, mean severity
  output_videos/<video>_pred.mp4    mask overlay + severity label

Usage:
    source ../env.sh && $PY predict.py --config inference_config.yaml
"""
import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from video_io import read_frames, probe


def rolling_median(arr, half):
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out[i] = np.median(arr[lo:hi], axis=0).astype(np.uint8)
    return out


def writer(path, w, h, fps):
    """ffmpeg pipe - cv2.VideoWriter is machine pe bharosemand nahi."""
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)],
        stdin=subprocess.PIPE)


def run_video(model, video, cfg, out_dir):
    pp = cfg["post_processing"]
    w0, h0, src_fps = probe(video)
    scale = 1280 / w0 if w0 > 1280 else None

    arr, _ = read_frames(video, fps=5, scale=scale)
    med = rolling_median(arr, half=10)
    h, w = med.shape[1:3]

    proc = writer(out_dir / f"{video.stem}_pred.mp4", w, h, 5)
    hist = deque(maxlen=pp["smoothing_window"])
    records = []
    t0 = time.time()

    for i, frame in enumerate(med):
        res = model.predict(frame, imgsz=cfg["imgsz"], conf=cfg["conf"],
                            iou=cfg["iou"], device=cfg["device"], verbose=False)[0]

        severity, confs = 0.0, []
        if res.masks is not None and len(res.masks) > 0:
            union = np.zeros((h, w), bool)
            for m, c in zip(res.masks.data.cpu().numpy(),
                            res.boxes.conf.cpu().numpy()):
                mm = cv2.resize(m.astype(np.float32), (w, h)) > 0.5
                if mm.mean() < pp["min_area_ratio"]:
                    continue
                union |= mm
                confs.append(float(c))
            severity = float(union.mean())

        hist.append(severity)
        sm = float(np.median(hist)) if pp["temporal_smoothing"] else severity

        records.append({"frame": i, "severity": round(sm, 5),
                        "raw_severity": round(severity, 5),
                        "n_instances": len(confs),
                        "confidences": [round(c, 3) for c in confs],
                        "spill_detected": bool(sm > 0)})

        vis = res.plot(img=frame.copy())
        lab = f"SPILL  {sm*100:.1f}% of frame" if sm > 0 else "no spill"
        col = (0, 0, 255) if sm > 0 else (0, 190, 0)
        cv2.rectangle(vis, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(vis, lab, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)
        proc.stdin.write(np.ascontiguousarray(vis).tobytes())

    elapsed = time.time() - t0
    proc.stdin.close(); proc.wait()
    return records, len(med), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="inference_config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / args.config).read_text())

    model = YOLO(str((HERE / cfg["weights"]).resolve()))
    src = (HERE / cfg["source"]).resolve()
    pred_dir, vid_dir = HERE / "predictions", HERE / "output_videos"
    pred_dir.mkdir(exist_ok=True); vid_dir.mkdir(exist_ok=True)

    rows = ["video,frames,inference_time_s,fps,mean_severity,max_severity,spill_frames"]
    for video in sorted(src.iterdir()):
        if video.suffix.lower() not in {".mp4", ".avi", ".mkv"}:
            continue
        recs, n, el = run_video(model, video, cfg, vid_dir)
        (pred_dir / f"{video.stem}.json").write_text(
            json.dumps({"video": video.name, "frames": n, "records": recs}, indent=2))
        sev = [r["severity"] for r in recs]
        spill = sum(r["spill_detected"] for r in recs)
        rows.append(f"{video.name},{n},{el:.2f},{n/el:.2f},"
                    f"{np.mean(sev):.5f},{max(sev):.5f},{spill}")
        print(f"{video.name:14s} {n:4d} frames  {n/el:5.1f} FPS  "
              f"mean_sev={np.mean(sev):.4f}  max_sev={max(sev):.4f}  spill_frames={spill}/{n}")

    (pred_dir / "summary.csv").write_text("\n".join(rows) + "\n")
    print(f"\nsummary -> {pred_dir/'summary.csv'}")


if __name__ == "__main__":
    main()
