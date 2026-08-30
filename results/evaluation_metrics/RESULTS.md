# Results — Grain Spillage Identification

Model: **YOLO11s-seg**, single class `grain_spill`, COCO-pretrained fine-tune.
Input representation: **rolling temporal median background** (±2 s).

---

## Validation (video-wise split)

Val videos: `train_2`, `train_8` (spill) + `train_6`, `train_10` (clean).
205 frames, 155 instances. Best epoch 63 of 111 (early stop, patience 30).

| Metric | Box | Mask |
|---|---|---|
| Precision | 0.9149 | 0.9149 |
| Recall | 0.6581 | 0.6581 |
| mAP50 | 0.6697 | 0.6697 |
| mAP50-95 | 0.5085 | 0.4649 |

Training time: **5.7 min** on one RTX 6000 Ada (GPU 10).

### Frame-level detection and threshold calibration

Per-video max confidence over all val frames:

| Video | label | mean conf | max conf |
|---|---|---|---|
| train_2 | spill | 0.941 | 0.944 |
| train_8 | spill | 0.365 | 0.705 |
| train_6 | clean | 0.000 | 0.000 |
| train_10 | clean | 0.000 | 0.000 |

Threshold sweep on **validation only** (test never used for tuning):

| thresh | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.05 | 105 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| 0.10 | 102 | 0 | 3 | 1.000 | 0.971 | 0.986 |
| 0.20 | 91 | 0 | 14 | 1.000 | 0.867 | 0.929 |
| 0.35 | 79 | 0 | 26 | 1.000 | 0.752 | 0.859 |

**Operating point chosen: conf = 0.10.** Best F1 sits at 0.05, but 0.10 keeps
precision at 1.000 and recall at 0.971 while leaving margin against the
HD/SD domain gap. Zero false positives on clean val videos at every threshold.

---

## Test inference

| Video | frames | time (s) | **FPS** | mean severity | max severity | spill frames |
|---|---|---|---|---|---|---|
| test_1.mkv | 99 | 3.10 | **31.9** | 0.0000 | 0.0000 | 0 / 99 |
| test_2.mkv | 100 | 1.32 | **76.0** | 0.0000 | 0.0000 | 0 / 100 |
| test_3.mkv | 152 | 5.17 | **29.4** | 0.0231 | 0.2415 | 15 / 152 |

Real-time capable: source footage is 10–25 fps, pipeline runs at 29–76 fps.

---

## Known limitation — false positives on test_3

**The 15 detections in `test_3` are false positives.** The model is firing on a
blue tarpaulin covering a trolley, not on grain.
See `results/sample_predictions/test3_false_positives.jpg`.

Cause: no tarpaulin, plastic sheet, or similar large smooth coloured surface
appears anywhere in the 12 training videos, so this is an out-of-distribution
object. The detection confidences (0.11–0.37) overlap the genuine spill range
of `train_8` (mean 0.365), so raising the threshold would suppress true
detections as well — and tuning the threshold against test data is explicitly
prohibited by the challenge rules, so it was not done.

Fix for a next iteration: hard-negative mining — annotate tarpaulins, plastic
sheets and similar surfaces as explicit background, and add copy-paste
augmentation of true spill onto varied backgrounds.

`test_1` and `test_2` produced no detections at all. Visual inspection of their
median backgrounds shows no clear loose-grain layer on the floor, so these are
plausibly correct negatives, but this is not verified against ground truth
(none was provided with the challenge).

---

## Honest note on annotation quality

The 18 keyframe masks were traced from visual inspection of the median
background images, not by a domain expert, and polygons are coarse. Validation
numbers should be read as measuring consistency with those masks, not against
expert ground truth. Re-tracing the keyframes and re-running the same pipeline
is the single highest-value improvement available.
