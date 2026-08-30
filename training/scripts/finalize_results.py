"""
Training outputs ko PDF ke maange hue results/ folders me rakho (Deliverable #5).

Usage:  python finalize_results.py
"""
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "training" / "results" / "yolo11s_seg_run1"
RES = ROOT / "results"


def main():
    if not RUN.exists():
        print(f"run folder nahi mila: {RUN}"); return
    graphs = RES / "training_graphs"
    metrics = RES / "evaluation_metrics"
    samples = RES / "sample_predictions"
    for d in (graphs, metrics, samples):
        d.mkdir(parents=True, exist_ok=True)

    for p in RUN.glob("*.png"):
        shutil.copy2(p, (samples if p.name.startswith("val_batch") else graphs) / p.name)
    for p in RUN.glob("*.jpg"):
        shutil.copy2(p, samples / p.name)

    csv_src = RUN / "results.csv"
    if csv_src.exists():
        shutil.copy2(csv_src, metrics / "results.csv")
        rows = list(csv.DictReader(csv_src.open()))
        rows = [{k.strip(): v for k, v in r.items()} for r in rows]
        key = "metrics/mAP50-95(M)"
        best = max(rows, key=lambda r: float(r.get(key) or 0))
        lines = [
            "# Evaluation Metrics — YOLO11s-seg, class `grain_spill`", "",
            f"Epochs run: {len(rows)}   |   Best epoch: {best['epoch']}", "",
            "| Metric | Box | Mask |", "|---|---|---|",
            f"| Precision | {float(best['metrics/precision(B)']):.4f} | {float(best['metrics/precision(M)']):.4f} |",
            f"| Recall | {float(best['metrics/recall(B)']):.4f} | {float(best['metrics/recall(M)']):.4f} |",
            f"| mAP50 | {float(best['metrics/mAP50(B)']):.4f} | {float(best['metrics/mAP50(M)']):.4f} |",
            f"| mAP50-95 | {float(best['metrics/mAP50-95(B)']):.4f} | {float(best['metrics/mAP50-95(M)']):.4f} |",
            "", f"Total training time: {float(rows[-1]['time'])/60:.1f} min", "",
            "Validation split (video-wise): train_2, train_8 (spill) + train_6, train_10 (clean).",
        ]
        (metrics / "metrics.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))

    for w in ("best.pt", "last.pt"):
        src = RUN / "weights" / w
        if src.exists():
            shutil.copy2(src, ROOT / "models" / "trained_model_weights" / w)
            print(f"  weights -> models/trained_model_weights/{w}")
    print(f"\ngraphs={len(list(graphs.glob('*')))} samples={len(list(samples.glob('*')))}")


if __name__ == "__main__":
    main()
