# Dataset Analysis Findings

Reproduce: `python analysis/check_static_camera.py`

## 1. Saare cameras poori tarah STATIC hain

Phase-correlation, pehla vs aakhri frame:

| Video | frames | camera shift | mean dev | moving pixels |
|---|---|---|---|---|
| train_7.mkv  | 110 | (+0.09, +0.03) px | 13.52 | 10.11% |
| train_1.avi  | 100 | (-0.00, -0.02) px |  8.39 |  8.27% |
| train_12.mkv | 106 | (-0.02, +0.02) px |  8.39 |  2.26% |
| test_1.mkv   |  99 | (+0.09, +0.12) px |  5.87 |  1.36% |
| test_3.mkv   | 152 | (+0.04, +0.06) px |  2.23 |  0.98% |

Sub-pixel drift (<0.15 px) — **test videos me bhi**. Tripod/wall-mounted CCTV.

## 2. Temporal median mazdooron ko hata deta hai, spill ko nahi

`analysis/static_check/train_7_cmp.jpg` dekho: 6 me se 5 mazdoor median me
gayab ho jaate hain, jabki floor pe pada anaaj bilkul intact rehta hai.

**Kyun ye sabse important finding hai:**
- Spill STATIC hai, mazdoor MOVING hain
- Mazdooron ke kapde aur boriyan spill jaise hi beige/safed hain -> ye is
  dataset ka #1 false-positive source hai
- Median background un्हें hata deta hai *bina koi model train kiye*

Sirf 1-10% pixels move karte hain, matlab median estimate bahut stable hai.

## 3. Architecture implication

Model ko raw frame dene ke bajaye **rolling temporal median** do (ya dono
6-channel me). Isse:
- False positives (mazdoor/bori) input me hi khatam
- Annotation aasan aur consistent — occlusion nahi (criteria #2)
- Koi extra label ki zaroorat nahi, mufт ka signal

## 4. Environment note

`cv2.VideoCapture` is env me .mkv/.avi open NAHI karta
(opencv 5.0.0 ka ffmpeg backend toota hai — `isOpened() -> False`).
Isliye `analysis/video_io.py` ffmpeg CLI se rawvideo pipe karta hai.
`extract_frames.py` ko bhi isi pe shift karna hai.

## 5. Spill bhi static hai -> annotation ka bojh 30x kam

Har video ke pehle vs aakhri median frame me kitne pixel badle:

| Video | frames | changed | Video | frames | changed |
|---|---|---|---|---|---|
| train_1  | 50 | **15.84%** | train_7  | 55 | **14.05%** |
| train_2  | 50 | 0.34% | train_8  | 55 | 2.64% |
| train_3  | 25 | 0.64% | train_9  | 54 | 0.13% |
| train_4  | 50 | 0.20% | train_10 | 45 | 0.11% |
| train_5  | 39 | 2.99% | train_11 | 36 | 0.12% |
| train_6  | 55 | 0.03% | train_12 | 53 | **9.31%** |

12 me se 9 videos poore video me **3% se kam** badalte hain.

**Isliye 567 frames annotate karne ki zaroorat nahi — sirf 18 keyframes.**
Stable videos ko 1 keyframe, dynamic (train_1/7/12) ko 3.
`make_keyframes.py` ye bana chuka hai; `propagate_masks.py` unhe saare
frames pe faila dega.

## 6. CLIPSeg zero-shot bootstrap FAIL hua

"grain spilled on the floor" prompt pe video-level discrimination to milta hai
(train_7 mean=0.097, train_4=0.069 vs clean train_6=0.013), lekin heatmap
diffuse hai aur pixel-level localization nahi deta — mask proposals ke kaam ka
nahi. `analysis/clipseg_heat.jpg` dekho. Isliye chhod diya; keyframes manually
annotate honge.
