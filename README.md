# Semantic Feature Visualisation for Autonomous Driving
### DINOv2 Vision Transformer on the Waymo Open Dataset v2.0.1

---

## Overview

One of the most fundamental challenges in autonomous driving perception is
teaching a machine to understand what it sees — not just detect objects, but
genuinely comprehend the semantic structure of a scene: which parts are road,
which are sky, which are other vehicles, which are people.

Traditional approaches solve this with supervision: you label thousands of
images, train a model on those labels, and hope it generalises. This works,
but it is expensive, slow, and brittle. The labels are always finite. The
real world is not.

This project takes a different direction. Instead of labelling anything, it
asks a simpler question: **what does a model that was never trained on
autonomous driving already understand about driving scenes?**

The model in question is **DINOv2**, a Vision Transformer pretrained entirely
through self-supervision on a curated dataset of internet images — no manual
labels, no domain-specific fine-tuning. Yet when its internal representations
are visualised on real Waymo driving footage, something striking emerges: the
model has spontaneously learned to separate roads from sky from vegetation
from vehicles, purely from the statistical structure of visual data.

This project builds a lightweight pipeline to extract, project, and visualise
these representations on three real driving segments from the Waymo Open
Dataset v2.0.1. The pipeline was built from scratch, working directly with
Waymo's parquet data format, handling frame extraction, DINOv2 inference,
dimensionality reduction via PCA, and cosine similarity analysis — all
running on a standard CPU laptop without any GPU.

The results connect directly to an active research direction in autonomous
driving: **weakly-supervised and self-supervised 4D occupancy forecasting**,
where the goal is to predict how a scene will evolve in 3D space and time
without relying on dense, expensive 3D annotations. Foundation models like
DINOv2 are increasingly seen as the key to unlocking this — and this project
is a practical, hands-on exploration of why.

---

## Results

Three driving segments from the Waymo validation set were processed.
Each image shows: **Input frame | DINOv2 PCA heatmap | Cosine similarity map**

**Segment 1 — Highway / construction zone**

<img width="2985" height="855" alt="segment_01" src="https://github.com/user-attachments/assets/dcd7fcf2-ac33-4c1b-b2c7-9418273c26d2" />


**Segment 2 — Urban intersection with pedestrians**

<img width="2985" height="855" alt="segment_02" src="https://github.com/user-attachments/assets/e395c9bf-b0f4-401f-be80-b4dfd6b363e6" />



**Segment 3 — Residential street with vehicles**

<img width="2985" height="855" alt="segment_03" src="https://github.com/user-attachments/assets/63bde6dc-4a96-4ef1-9966-9bc6c92272c6" />


> Same colour = same semantic class. No labels used — entirely self-supervised.

---

## Method

```
Waymo parquet  →  JPEG decode  →  PNG frames
                                      ↓
                              DINOv2 ViT-S/14
                                      ↓
                          Patch tokens [N × 384]
                                      ↓
                           PCA (3 components)
                                      ↓
                         RGB semantic heatmap
```

- **DINOv2 patch tokens** encode local semantics at 14×14 px resolution
- **Joint PCA** across frames → consistent colour-to-class mapping
- **Cosine similarity** maps reveal which regions share the same feature identity
- Motivates 2D→3D feature distillation in semantic 4D occupancy forecasting

---

## Setup

```bash
git clone https://github.com/<your-username>/waymo-dinov2.git
cd waymo-dinov2
pip install -r requirements.txt
```

### Download data (Waymo v2.0.1)

Requires Google account + Waymo terms acceptance at
[waymo.com/open](https://waymo.com/open).

```bash
# Install Google Cloud SDK, then:
gcloud auth login

# Download 3 validation segments (~1 GB)
gsutil -m cp \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_image/10203656353524179475_7625_000_7645_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_image/1024360143612057520_3580_000_3600_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_image/10247954040621004675_2180_000_2200_000.parquet" \
  ./data/waymo/camera_image/
```

---

## Usage

### Full pipeline (parquet → frames → heatmaps)
```bash
cd src
python dinov2_waymo.py \
    --parquet_dir C:/Users/Lindsay/data/waymo/camera_image \
    --out_dir     ../outputs \
    --camera      1 \
    --max_frames  15
```

### If frames already extracted
```bash
cd src
python dinov2_waymo.py \
    --frames_dir ../outputs/frames \
    --out_dir    ../outputs
```

### Inspect parquet schema
```bash
cd src
python waymo_reader.py \
    --parquet_dir C:/Users/Lindsay/data/waymo/camera_image \
    --out_dir     ../outputs/frames \
    --inspect
```

---

## Camera IDs

| ID | Camera |
|----|--------|
| 1  | FRONT (default) |
| 2  | FRONT_LEFT |
| 3  | FRONT_RIGHT |
| 4  | SIDE_LEFT |
| 5  | SIDE_RIGHT |

---

## Project structure

```
waymo-dinov2/
├── src/
│   ├── dinov2_waymo.py    # Main pipeline: parquet → DINOv2 → heatmaps
│   └── waymo_reader.py    # Waymo v2.0.1 parquet reader + PNG extractor
├── outputs/               # Generated heatmaps + extracted frames
└── requirements.txt
```

---

## Connection to 4D Occupancy Forecasting

This visualisation is the foundation for **Semantic 4D Occupancy Forecasting**:

1. DINOv2 produces rich, open-vocabulary 2D semantic patch features
2. These features are **lifted into 3D/4D** via camera projection + temporal fusion
3. A Transformer prediction head forecasts future voxel occupancy without dense 3D labels

Related work:
- [DINOv2](https://arxiv.org/abs/2304.07193) — Oquab et al., TMLR 2024
- [OccFormer](https://arxiv.org/abs/2304.05316) — Zhang et al., ICCV 2023
- [UniOcc](https://arxiv.org/abs/2306.09117) — Pan et al., 2023
- [Waymo Open Dataset](https://arxiv.org/abs/1912.04838) — Sun et al., CVPR 2020
