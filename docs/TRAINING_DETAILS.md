# Training Details Document
*Mandatory Deliverable #4 — bharna hai training ke baad*

## 1. Model Architecture
- **Model:** YOLO11s-seg (instance segmentation), single class `grain_spill`
- **Backbone:** CSPDarknet + C3k2 blocks, SPPF, C2PSA attention
- **Neck:** PANet (FPN top-down + bottom-up), P3/P4/P5 scales
- **Heads:** anchor-free decoupled detection head + YOLACT-style mask head
  (32 prototype masks @160×160 + per-instance coefficients)
- **Params:** ~10M | **Init:** COCO-pretrained `yolo11s-seg.pt`
- **Kyun segmentation:** spill amorphous region hai — bbox me 40-60% floor/bori
  aa jaata hai. Mask se `spill_px / frame_px` = severity metric milta hai.

## 2. Dataset Statistics
Dekho: `dataset/dataset_description.txt`

| | |
|---|---|
| Train videos | 12 |
| Test videos | 3 |
| Total footage | ~3.9 min |
| Extraction FPS | 2 fps |
| Extracted frames | 743 (567 train + 176 test) |
| Keyframes annotated | 18 (9 spill, 9 clean) |
| Annotated images | 567 (propagated) |
| Train / Val split | 362 / 205 frames = 0.64 / 0.36 (video-wise) |
| Val videos | train_2, train_8 (spill), train_6, train_10 (clean) |

## 3. Hyperparameters
Full config: `training/config_files/train_config.yaml`

| Param | Value |
|---|---|
| imgsz | 640 |
| epochs | 150 (patience 30) |
| batch | 16 |
| optimizer | AdamW, lr0=0.001, lrf=0.01, cosine |
| warmup | 3 epochs |
| freeze | first 10 layers |
| close_mosaic | last 15 epochs |

## 4. Loss Functions
- Classification: BCE
- Box regression: CIoU + DFL (Distribution Focal Loss)
- Mask: BCE over prototype-combined masks

## 5. Training Procedure
1. `extract_frames.py --fps 2` → `dataset/frames_raw/`
2. SAM-assisted polygon annotation (CVAT / Roboflow) → YOLO-seg `.txt`
3. `split_dataset.py --val <video stems>` → video-wise split
4. `train.py --config config_files/train_config.yaml`
5. Best weights → `models/trained_model_weights/best.pt`

**Total training time:** 5.7 min (111 epochs, early stop at patience 30; best epoch 63)

## 6. Hardware Specifications
- **GPU:** NVIDIA RTX 6000 Ada Generation, 48 GB VRAM (device 2 on `gpu17`)
- **CPU:** 128 cores
- **RAM:** 1007 GB
- **CUDA:** 12.1 | **PyTorch:** 2.5.1+cu121+cu121 | **Ultralytics:** 8.4.133
- **OS:** Linux 6.8.0-50-generic

## 7. Results

Full write-up: **`results/RESULTS.md`**

| Metric | Box | Mask |
|---|---|---|
| Precision | 0.9149 | 0.9149 |
| Recall | 0.6581 | 0.6581 |
| mAP50 | 0.6697 | 0.6697 |
| mAP50-95 | 0.5085 | 0.4649 |

Frame-level detection on val at conf=0.10: precision 1.000, recall 0.971,
zero false positives on clean videos.

Test inference: 29-76 FPS. `test_3` produced 15 false positives on a blue
tarpaulin (out-of-distribution object) - documented in `results/RESULTS.md`.
