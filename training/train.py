"""
Grain Spillage Identification - IITD Challenge
Step 3: Model Development and Training

Usage:
    python train.py --config config_files/train_config.yaml
"""
import argparse
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_files/train_config.yaml")
    args = ap.parse_args()

    cfg_path = (HERE / args.config).resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # data yaml ko config_files/ ke relative resolve karo
    cfg["data"] = str((cfg_path.parent / cfg["data"]).resolve())
    cfg["project"] = str((cfg_path.parent / cfg["project"]).resolve())

    model = YOLO(cfg.pop("model"))

    t0 = time.time()
    model.train(**cfg)
    mins = (time.time() - t0) / 60
    print(f"\nTotal training time: {mins:.1f} min")

    # best weights ko models/ me copy karo (Deliverable #2)
    run_dir = Path(cfg["project"]) / cfg["name"]
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        dest = HERE.parent / "models" / "trained_model_weights" / "best.pt"
        dest.write_bytes(best.read_bytes())
        print(f"Weights copied -> {dest}")


if __name__ == "__main__":
    main()
