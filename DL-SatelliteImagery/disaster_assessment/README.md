# Satellite Disaster Assessment & Analysis System (Phase 1 & Phase 2)

> **Modular extension** for the DL-SatelliteImagery project.  
> All 8 original DeepWorks notebooks, models, and workflows remain 100% untouched.

---

## 1. System Architecture

```
disaster_assessment/
├── models/
│   ├── base_unet.py               # Shared encoder/decoder building blocks
│   ├── siamese_unet.py            # Pre/post change detection (~3.54M params)
│   ├── flood_unet.py              # Binary flood segmentation (~2.36M params)
│   └── building_damage_model.py   # Siamese 4-class building damage (~3.84M params)
│
├── datasets/
│   └── building_damage_dataset.py # xBD / triplet dataset loader with augmentations
│
├── training/
│   └── train_building_damage.py   # Full training loop (Dice + CE loss, mixed precision, checkpoints)
│
├── inference/
│   └── building_damage_inference.py # Dual-mode inference (Trained vs. Estimated fallback)
│
├── preprocessing/
│   ├── image_validation.py        # Integrity, dimension, and channel checks
│   └── image_alignment.py         # Dimension matching, ORB alignment, CLAHE normalization
│
├── analytics/
│   ├── geospatial_area.py         # GeoTIFF metadata, Projected/Geographic CRS, resolution engine
│   ├── area_calculator.py         # Pixel-based and relative area calculator
│   ├── damage_metrics.py          # Phase 1 structural change heuristic
│   ├── land_change_metrics.py     # 6-class land cover transition matrix & metrics
│   └── severity_engine.py         # Configurable 0-100 disaster severity scorer
│
├── visualization/
│   ├── overlays.py                # Land cover, change, flood, 3-class & 4-class damage overlays
│   ├── heatmap.py                 # Fused disaster intensity heatmaps with Gaussian smoothing
│   └── comparison.py              # Side-by-side and multi-panel comparison grids
│
├── pipeline/
│   └── disaster_analyzer.py       # Main 10-step orchestrator pipeline
│
├── configs/
│   ├── severity_config.yaml       # Severity weights & thresholds
│   └── damage_model_config.yaml   # Building damage training/model config
│
└── weights/
    ├── siamese_unet.pth           # Change detection weights
    ├── flood_unet.pth             # Flood segmentation weights
    └── building_damage/
        └── best_damage_model.pth  # Trained 4-class building damage weights
```

---

## 2. Resource Requirements & Optimization

| Model | Parameters | VRAM (fp32) | VRAM (fp16 autocast) | Device |
|---|---|---|---|---|
| **Siamese U-Net** (Change Detection) | 3,539,281 (~3.54M) | ~14.2 MB | ~7.1 MB | CUDA / CPU |
| **Flood U-Net** (Binary Flood) | 2,359,521 (~2.36M) | ~9.4 MB | ~4.7 MB | CUDA / CPU |
| **BuildingDamageUNet** (4-Class Damage) | 3,842,756 (~3.84M) | ~15.4 MB | ~7.7 MB | CUDA / CPU |
| **Land-Cover U-Net** (Original Keras) | 1,939,686 (~1.94M) | ~7.8 MB | N/A | CPU / GPU |
| **Total Peak Inference Footprint** | **~11.7M total** | **~550 MB** | **~380 MB** | **GTX 1650 (4GB VRAM)** |

---

## 3. Two Damage Assessment Modes

The system clearly distinguishes between real trained classifications and heuristic estimates:

### Mode 1 — Trained Damage Assessment
- **Trigger**: `damage_mode="trained"` or `damage_mode="auto"` when weights exist in `weights/building_damage/best_damage_model.pth`.
- **Outputs**: 4 discrete classes:
  - `0`: **Undamaged** (Green `#00C800`)
  - `1`: **Minor Damage** (Yellow `#FFD700`)
  - `2`: **Major Damage** (Orange `#FF8C00`)
  - `3`: **Destroyed** (Red `#FF0000`)
- **Labeling**: Explicitly reported as `"Trained Neural Model (4-Class)"`.

### Mode 2 — Estimated Damage (Phase 1 Fallback)
- **Trigger**: `damage_mode="estimated"` or `damage_mode="auto"` when trained weights are not present on disk.
- **Outputs**: Structural change heuristic based on pre/post building mask comparison (`Undamaged`, `Possible Damage`, `Severe Damage`).
- **Labeling**: Explicitly reported as `"Estimated Damage: structural change heuristic"`.

---

## 4. Georeferenced Flood Area Engine

Supports 3 distinct operational modes:

1. **Mode 1 — GeoTIFF (.tif/.tiff)**:
   - Automated CRS and affine transform parsing via `rasterio`.
   - **Projected CRS** (e.g. UTM): Uses meter-based pixel dimensions.
   - **Geographic CRS** (e.g. EPSG:4326): Calculates geodesic metric resolution based on central scene latitude ($dx = \cos(\text{lat}) \cdot 111319.5\text{ m}$, $dy = 111132.9\text{ m}$).
2. **Mode 2 — User-Supplied Resolution (`meters_per_pixel`)**:
   - For standard PNG/JPG files (e.g. `meters_per_pixel=0.5` for WorldView-3). Computes exact $\text{m}^2$ and $\text{km}^2$.
3. **Mode 3 — No Georeference Available**:
   - Reports pixel counts and percentages only. **Never invents $\text{km}^2$**.

---

## 5. Dataset Structure & Training

### Dataset Structure (xBD / Custom)
```
data/damage_dataset/
├── train/
│   ├── pre/     (e.g., img_001.png, img_002.png)
│   ├── post/    (e.g., img_001.png, img_002.png)
│   └── masks/   (e.g., img_001.png, img_002.png with values 0, 1, 2, 3)
└── val/
    ├── pre/
    ├── post/
    └── masks/
```

### Train Building Damage Model
```powershell
.\.venv\Scripts\python.exe DL-SatelliteImagery\disaster_assessment\training\train_building_damage.py --data_dir data/damage_dataset --epochs 20 --batch_size 4 --lr 0.001
```

### Dry Run on Synthetic Samples
```powershell
.\.venv\Scripts\python.exe DL-SatelliteImagery\disaster_assessment\training\train_building_damage.py --dry_run
```

---

## 6. How to Launch and Use

### Launch Unified Gradio Web UI
```powershell
.\.venv\Scripts\python.exe DL-SatelliteImagery\disaster_gradio_app.py
```
Open `http://localhost:7860`:
- **Tab 1:** Land-Cover Segmentation
- **Tab 2:** Disaster Assessment (Select Damage Mode, input optional GSD or GeoTIFF)
- **Tab 3:** Quick Surface Change Detection

### Run Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest DL-SatelliteImagery/tests -v
```

```text
======================== 92 passed, 2 skipped in 14.45s ========================
```

### Python API
```python
from disaster_assessment.pipeline import DisasterAnalyzer
import numpy as np
from PIL import Image

analyzer = DisasterAnalyzer()
pre = np.array(Image.open("pre.jpg"))
post = np.array(Image.open("post.jpg"))

report = analyzer.analyze(
    pre_image=pre,
    post_image=post,
    damage_mode="auto",
    meters_per_pixel=0.5,
)

print(report.summary())
print(report.to_dict())
```
