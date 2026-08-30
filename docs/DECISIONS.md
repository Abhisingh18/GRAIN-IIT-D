# Project Decisions & Standing Instructions

Ye file har wo decision aur instruction rakhti hai jo is project pe lagu hai.
Koi bhi naya kaam shuru karne se pehle ye padho.

---

## Standing instructions (user ne diye)

| # | Instruction |
|---|---|
| 1 | Kaam sirf `/speech/abhishek/abhi_iitd/Grain_Spillage_Identification` me hoga |
| 2 | **GPU 10** use karni hai (training + inference dono) |
| 3 | Folder structure PDF ke "Suggested folder structure" ke exactly hisaab se |
| 4 | Best possible accuracy target hai |

---

## Environment

```bash
/speech/abhishek/miniconda3/envs/grain/bin/python
```

- Python 3.10 | torch 2.5.1+cu121 | ultralytics 8.4.133 | CUDA available, 11 devices
- Banaya: `conda create -n grain python=3.10`

**Zaroori:** `conda` PATH me nahi hai. Full path use karo:
`/speech/abhishek/miniconda3/bin/conda`

**Zaroori:** is machine pe `cv2.VideoCapture` .mkv/.avi open NAHI karta
(opencv ka ffmpeg backend toota hai, `isOpened() -> False`).
Video padhne ke liye hamesha `analysis/video_io.py` use karo — wo ffmpeg CLI
se rawvideo pipe karta hai.

---

## Hardware

- GPU: NVIDIA RTX 6000 Ada Generation, 48 GB (device **10** on `gpu17`)
- CPU: 128 cores | RAM: 1007 GB | OS: Linux 6.8.0-50-generic

---

## Task formulation

**Instance segmentation**, single class `grain_spill`, YOLO-seg polygon format.

Bbox kyun nahi: spill amorphous region hai — box me 40-60% floor aur boriyan
aa jaati hain, jisse training signal kharab hota hai aur IoU model quality
reflect nahi karta. Mask se `spill_px / frame_px` = severity metric milta hai.

---

## Architecture (final)

```
Videos (static CCTV)
   ↓
Rolling temporal median      <- mazdoor hat-te hain, spill rehta hai
   ↓
SAM2-assisted mask annotation (1 class)
   ↓
Copy-paste augmentation + multi-scale
   ↓
YOLO11s-seg  (COCO-pretrained, fine-tune, freeze=10)
   ↓
Temporal filter (5-frame median) + area filter
   ↓
severity = spill_px / frame_px  +  annotated videos
```

**Core model:** `yolo11s-seg`, COCO init.
`n` is texture task ke liye kamzor; `m`/`l` ~500 frames pe overfit honge.

---

## Kyun temporal median (verified, guess nahi)

`analysis/FINDINGS.md` dekho. Measure kiya:
- Saare cameras static hain — drift < 0.15 px, **test videos me bhi**
- Temporal median mazdooron ko hataata hai, spill ko nahi (train_7: 6 me 5 gayab)
- Sirf 1-10% pixels move karte hain

Spill STATIC hai, confusing cheezein (mazdoor, uthaayi jaa rahi boriyan) MOVING
hain — aur unke kapde/boriyan spill jaise hi beige hain, yani ye #1
false-positive source hai. Median unhe input me hi hata deta hai, bina model.

---

## Data ke baare me tay baatein

- **Split VIDEO-WISE hoga, frame-wise NAHI.** Ek video ke frames ~identical
  hote hain; random frame split se val accuracy jhooti aayegi.
- **Test videos kabhi bhi** training / validation / threshold tuning me nahi
  (PDF me explicit mana hai — pseudo-labeling bhi nahi).
- Do camera types training me dono honi chahiye:
  HD 2592x1944 (train_1..5, test_3) | SD 704x576 (train_6..12, test_1,2)
- `test_3` HD hai — scale mismatch #1 generalization risk hai.

### Annotation rule (har frame pe yahi)
```
IN : floor pe pada loose grain jahan continuous parat ho
OUT: bori ke upar jama dhool/grain
OUT: patli dusting jahan floor ka texture saaf dikhta ho
NOTE: mazdoor/object upar ho to mask continuous rakho, kaato mat
```

---

## Accuracy levers (priority order)

| Lever | Mehnat | Faayda |
|---|---|---|
| Temporal median input | Low | ****  |
| Annotation quality (SAM2-assisted) | High | *** |
| Copy-paste augmentation | Medium | *** |
| Scale fix (imgsz 1024 ya native crops) | Low/Med | ** |
| Leave-one-video-out CV (12 folds) | Low | ** (report) |
| Ensemble (+SegFormer / DINOv2-frozen) | Medium | ** |
| TTA (hflip + multiscale) | Low | * |

---

## Augmentation ke tay faisle

```
hsv_v=0.5        lighting 09:32-22:01 tak badalti hai
scale=0.5        do resolutions ka gap
fliplr=0.5       safe
degrees=5        halka camera tilt
flipud=0.0       NAHI - CCTV angle hamesha top-down
close_mosaic=15  aakhri 15 epochs pe mosaic band (spill ka context tootta hai)
```

---

## GPU addressing gotcha (important)

`nvidia-smi` ka index aur torch ka device index **match nahi karte** is machine
pe — beech me ek NVIDIA T1000 `Exclusive_Process` mode me hai, aur usko touch
karte hi `CUDA error: CUDA-capable device(s) is/are busy or unavailable` aata
hai.

Isliye GPU hamesha isolate karke use karo:

```bash
source env.sh          # CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=10
$PY training/train.py  # code ke andar GPU 10 = cuda:0
```

Configs me `device: 0` likha hai — wo physical GPU 10 hi hai, `env.sh` ke saath.
Verified: `cuda:0 -> RTX 6000 Ada, free 42.1 GB`.
