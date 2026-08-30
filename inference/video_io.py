"""
Robust video reader - ffmpeg CLI pipe se.

Is env me cv2.VideoCapture .mkv/.avi open nahi kar pa raha (opencv 5.0.0
ka ffmpeg backend toota hai), isliye frames seedhe ffmpeg se rawvideo
pipe karke numpy me le rahe hain.
"""
import subprocess
from pathlib import Path

import numpy as np


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0:s=,", str(path)],
        capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(out[0]), int(out[1])
    num, den = out[2].split("/")
    return w, h, float(num) / float(den)


def read_frames(path, fps=None, max_frames=None, scale=None):
    """BGR frames ka numpy array (T,H,W,3) return karta hai."""
    w, h, src_fps = probe(path)
    vf = []
    if fps:
        vf.append(f"fps={fps}")
    if scale:
        w, h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
        vf.append(f"scale={w}:{h}")

    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3), src_fps
