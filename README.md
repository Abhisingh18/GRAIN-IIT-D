# Grain Spillage Identification — IITD Challenge

CCTV footage se anaaj ke spillage ko identify karne ka end-to-end computer
vision pipeline. Approach: **YOLO11s-seg instance segmentation**, single class
`grain_spill`, plus temporal post-processing jo per-video **severity score**
deta hai.

---

## Folder Structure

```
Grain_Spillage_Identification/
├── env.sh                           # GPU 10 isolate karta hai - har script se pehle source karo
│
├── dataset/
│   ├── images/{train,val}/          # 362 / 205 annotated frames
│   ├── annotations/{train,val}/     # YOLO-seg polygon labels (567)
│   ├── labels -> annotations        # Ultralytics is naam se dhoondta hai
│   ├── keyframes/                   # 18 masks jo manually banaye (source of truth)
│   ├── frames_med/                  # rolling temporal median - model ka asli input
│   ├── frames_raw/                  # raw frames (reference)
│   ├── raw_videos -> ../../IITD-Grain_Spillage_Identification_Challenge/Dataset
│   └── dataset_description.txt      # annotation methodology + statistics
│
├── training/
│   ├── train.py
│   ├── scripts/
│   │   ├── extract_frames.py        # videos -> frames + median backgrounds
│   │   ├── make_keyframes.py        # 567 frames -> 18 keyframes
│   │   ├── propagate_masks.py       # 18 masks -> 567 labels
│   │   ├── split_dataset.py         # video-wise train/val split
│   │   ├── ingest_labels.py         # annotation tool ka export -> .txt
│   │   ├── draft_masks.py           # (fail hua heuristic - reference ke liye rakha)
│   │   ├── bootstrap_masks.py       # (fail hua CLIPSeg - reference ke liye rakha)
│   │   └── finalize_results.py      # outputs -> results/
│   ├── config_files/
│   │   ├── grain_spill.yaml         # dataset config
│   │   └── train_config.yaml        # hyperparameters
│   ├── results/yolo11s_seg_run1/    # raw Ultralytics run output
│   └── requirements.txt
│
├── models/
│   └── trained_model_weights/       # best.pt, last.pt
│
├── inference/
│   ├── predict.py                   # test videos pe inference + post-processing
│   ├── video_io.py                  # ffmpeg reader (cv2 is machine pe toota hai)
│   ├── inference_config.yaml
│   ├── predictions/                 # per-video JSON + summary.csv
│   └── output_videos/               # annotated .mp4 x3
│
├── results/
│   ├── RESULTS.md                   # metrics, threshold calibration, limitations
│   ├── evaluation_metrics/
│   ├── training_graphs/
│   ├── sample_predictions/
│   └── train.log
│
├── analysis/                        # dataset analysis jisse architecture decide hua
│   ├── FINDINGS.md                  # static camera + static spill ka proof
│   ├── check_static_camera.py
│   └── video_io.py
│
├── docs/
│   ├── TRAINING_DETAILS.md          # Deliverable #4
│   └── DECISIONS.md                 # saare decisions + environment gotchas
│
└── README.md
```

---

## 1. Environment Setup

```bash
conda create -n grain python=3.10 -y
conda activate grain
pip install -r training/requirements.txt
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Hardware used:** NVIDIA RTX 6000 Ada (48 GB), device 2 on `gpu17` — 128 CPU
cores, 1007 GB RAM, CUDA 12.1.

---

## 2. Dataset Preparation

```bash
source env.sh          # GPU 10 isolate karta hai (zaroori)

# Step 1: frames + rolling temporal median backgrounds
$PY training/scripts/extract_frames.py --fps 2 --median-window 2.0

# Step 2: 18 keyframes nikaalo (567 nahi - camera aur spill dono static hain)
$PY training/scripts/make_keyframes.py

# Step 3: dataset/keyframes/ ke 18 images annotate karo
#         1 class: grain_spill | YOLO-seg polygon export
#         .txt ko usi naam se dataset/keyframes/ me rakho

# Step 4: keyframe masks ko saare frames pe propagate karo
$PY training/scripts/propagate_masks.py

# Step 5: video-wise split
$PY training/scripts/split_dataset.py --val train_3 train_8 train_11
```

> **Model median background pe train hota hai, raw frame pe nahi.** Static CCTV
> pe median moving mazdooron ko hata deta hai aur spill ko chhodta hai — yahi
> is dataset ka #1 false-positive source tha. Dekho `analysis/FINDINGS.md`.

Annotation rules aur statistics: [`dataset/dataset_description.txt`](dataset/dataset_description.txt)


---

## 3. Training Procedure

```bash
cd training
$PY train.py --config config_files/train_config.yaml
```

Best weights apne aap `models/trained_model_weights/best.pt` me copy ho jaate
hain. Training curves `results/yolo11s_seg_run1/` me aayenge.

Hyperparameters: [`training/config_files/train_config.yaml`](training/config_files/train_config.yaml)

---

## 4. Inference Procedure

```bash
cd inference
$PY predict.py --config inference_config.yaml
```

Output:
- `predictions/<video>.json` — per-frame severity, confidence, instance count
- `predictions/summary.csv` — inference time, **FPS**, mean severity per video
- `output_videos/<video>_pred.mp4` — mask overlay + severity label

**Post-processing:** confidence threshold 0.35 → 5-frame rolling median
(single-frame flicker hatane ke liye) → area filter (`min_area_ratio 0.005`)
→ severity = `spill_pixels / frame_pixels`.

---

## 5. Results

| Metric (mask) | Value |
|---|---|
| Precision | 0.915 |
| Recall | 0.658 |
| mAP50 | 0.670 |
| mAP50-95 | 0.465 |

Frame-level: precision 1.000 / recall 0.971 on val at conf=0.10.
Inference 29-76 FPS. Training 5.7 min on one RTX 6000 Ada.

Full results and a documented false-positive failure case on `test_3`:
[`results/RESULTS.md`](results/RESULTS.md)

---

## 6. Dependencies

Dekho [`training/requirements.txt`](training/requirements.txt). Main:
`torch 2.5.1`, `torchvision 0.20.1`, `ultralytics 8.4.133`,
`opencv-python-headless 4.11.0.86`, `numpy 1.26.4`.

---

## Deliverables Map

| PDF Deliverable | Kahan hai |
|---|---|
| 1. Complete project folder | ye poora repo |
| 2. Trained model weights | `models/trained_model_weights/` |
| 3. Dataset + annotations | `dataset/` + `dataset_description.txt` |
| 4. Training details document | `docs/TRAINING_DETAILS.md` |
| 5. Training & evaluation results | `results/` |
| 6. README | ye file |
