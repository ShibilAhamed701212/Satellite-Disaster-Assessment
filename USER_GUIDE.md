# Satellite Disaster Assessment — User Execution Guide

Welcome to the **Satellite Disaster Assessment & Remote Sensing Intelligence System**, an end-to-end deep learning platform for automated post-disaster damage assessment, surface change detection, flood inundation mapping, and multi-hazard severity scoring.

This guide provides step-by-step instructions for environment configuration, Web UI launching, command-line utilities, training pipelines, and troubleshooting.

---

## 📖 Table of Contents

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Quick Start Guide](#2-quick-start-guide)
3. [Satellite Imagery & Disaster Assessment (`DL-SatelliteImagery`)](#3-satellite-imagery--disaster-assessment-dl-satelliteimagery)
   - [3.1 Baseline Land-Cover Segmentation (6 Classes)](#31-baseline-land-cover-segmentation-6-classes)
   - [3.2 Satellite Disaster Assessment System (Phase 1 & Phase 2)](#32-satellite-disaster-assessment-system-phase-1--phase-2)
   - [3.3 Trained Building Damage Assessment (4 Classes)](#33-trained-building-damage-assessment-4-classes)
   - [3.4 Georeferenced Flood Area Engine (GeoTIFF & GSD)](#34-georeferenced-flood-area-engine-geotiff--gsd)
   - [3.5 Launching the Unified Disaster Gradio Web UI](#35-launching-the-unified-disaster-gradio-web-ui)
   - [3.6 Training the Building Damage Model](#36-training-the-building-damage-model)
   - [3.7 Running Automated Verification Tests](#37-running-automated-verification-tests)
   - [3.8 Real Satellite Imagery Reference (`sample_images/`)](#38-real-satellite-imagery-reference-sample_images)
   - [3.9 Python API Usage Example](#39-python-api-usage-example)
4. [Troubleshooting & Common Issues](#4-troubleshooting--common-issues)

---

## 1. Prerequisites & Environment Setup

### System Requirements
- **Operating System**: Windows 10/11, Ubuntu 20.04/22.04, or macOS (Intel / Apple Silicon M-Series).
- **Python Version**: Python **3.10** or **3.11** (Active environment: Python 3.11).
- **Package Manager**: [Conda](https://docs.conda.io/) or Python standard `venv`.
- **GPU Acceleration**: NVIDIA GPU with CUDA 12.1 / 11.8 (tested on NVIDIA GeForce GTX 1650 4GB VRAM).

### Setting Up a Virtual Environment

#### Option A: Using Conda
```bash
conda create -n deepworks python=3.11 -y
conda activate deepworks
```

#### Option B: Using Python `venv` (Current Setup)
```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### Installing Dependencies

Run the following command from the repository root directory (`d:\var-codes\satelite`):

```powershell
# Core PyTorch and Remote Sensing Stack via requirements.txt
pip install -r requirements.txt

# Or install package in editable mode with all dependencies
pip install -e .
```

---

## 2. Quick Start Guide

1. **Open PowerShell / Terminal** in the project directory:
   ```powershell
   cd d:\var-codes\satelite
   ```

2. **Verify GPU / CUDA Acceleration**:
   ```powershell
   .\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

3. **Launch the Unified Disaster Web UI**:
   ```powershell
   .\.venv\Scripts\python.exe DL-SatelliteImagery\disaster_gradio_app.py
   ```
   Open `http://localhost:7860` in your web browser.

4. **Launch JupyterLab** (for interactive notebooks):
   ```powershell
   jupyter lab
   ```
   Access notebooks at `http://localhost:8888`.

---

## 3. Satellite Imagery & Disaster Assessment (`DL-SatelliteImagery`)

The **`DL-SatelliteImagery`** module contains the complete land-cover segmentation notebooks as well as the **Satellite Disaster Assessment and Analysis System (Phase 1 & Phase 2)**.

```
DL-SatelliteImagery/
├── [ALL 8 ORIGINAL NOTEBOOKS PRESERVED]
│
├── disaster_assessment/
│   ├── models/
│   │   ├── base_unet.py              # Shared encoder/decoder (16→32→64→128→256)
│   │   ├── siamese_unet.py           # Change detection (~3.54M params)
│   │   ├── flood_unet.py             # Binary flood mapping (~2.36M params)
│   │   └── building_damage_model.py  # 4-class Siamese damage network (~3.84M params)
│   │
│   ├── datasets/
│   │   └── building_damage_dataset.py # xBD / triplet loader with augmentations
│   │
│   ├── training/
│   │   └── train_building_damage.py   # Training loop (Dice + CE loss, mixed precision)
│   │
│   ├── inference/
│   │   └── building_damage_inference.py # Dual-mode engine (Trained vs. Estimated)
│   │
│   ├── analytics/
│   │   ├── geospatial_area.py        # GeoTIFF metadata & geodesic resolution engine
│   │   ├── area_calculator.py        # Metric / percentage area calculator
│   │   ├── damage_metrics.py         # Structural change heuristic (Phase 1)
│   │   ├── land_change_metrics.py    # 6-class land cover transition matrix
│   │   └── severity_engine.py        # Configurable 0-100 severity scorer
│   │
│   ├── visualization/
│   │   ├── overlays.py               # Land cover, change, flood & 4-class damage overlays
│   │   ├── heatmap.py                # Fused multi-hazard disaster heatmaps
│   │   └── comparison.py             # Side-by-side & grid comparison panels
│   │
│   └── pipeline/
│       └── disaster_analyzer.py      # Unified 10-step orchestrator pipeline
│
└── disaster_gradio_app.py            # Unified 3-tab Gradio Web UI
```

---

### 3.1 Baseline Land-Cover Segmentation (6 Classes)

The baseline model classifies satellite image pixels into 6 land-cover categories:

| Class Code | Label | Color Code | Description |
| :---: | :--- | :---: | :--- |
| **`0`** | **`Water`** | `#E2A929` | Water bodies, rivers, lakes, flooded zones |
| **`1`** | **`Land`** | `#8429F6` | Bare soil, unbuilt terrain |
| **`2`** | **`Road`** | `#6EC1E4` | Transportation networks, highways |
| **`3`** | **`Building`** | `#3C1098` | Structural footprints, urban areas |
| **`4`** | **`Vegetation`** | `#FEDD3A` | Forests, green canopy, agricultural fields |
| **`5`** | **`Unlabeled`** | `#9B9B9B` | Background / unclassified pixels |

#### Notebook Reference:
- **Patching & Masking**: [`DL-SatelliteImagery/Satellite_Imagery_Segmentation.ipynb`](./DL-SatelliteImagery/Satellite_Imagery_Segmentation.ipynb)
- **Base Training**: [`DL-SatelliteImagery/Satellite_Imagery_DeepLearning-Base.ipynb`](./DL-SatelliteImagery/Satellite_Imagery_DeepLearning-Base.ipynb)
- **Local Diagnostics**: [`DL-SatelliteImagery/Satellite_Imagery_DeepLearning-LocalDiag.ipynb`](./DL-SatelliteImagery/Satellite_Imagery_DeepLearning-LocalDiag.ipynb)
- **Model Checkpoints**: [`DL-SatelliteImagery/Satellite_Imagery_DeepLearning-SaveLoadModel.ipynb`](./DL-SatelliteImagery/Satellite_Imagery_DeepLearning-SaveLoadModel.ipynb)
- **WandB Tracking**: [`DL-SatelliteImagery/Satellite_Imagery_DeepLearning-WandB.ipynb`](./DL-SatelliteImagery/Satellite_Imagery_DeepLearning-WandB.ipynb)
- **Activation Heatmaps**: [`DL-SatelliteImagery/Satellite_Imagery_DeepLearning_ActivationHeatmap.ipynb`](./DL-SatelliteImagery/Satellite_Imagery_DeepLearning_ActivationHeatmap.ipynb)
- **Interactive Web App**: [`DL-SatelliteImagery/Satellite_segmentation_Prediction-GradioUI.ipynb`](./DL-SatelliteImagery/Satellite_segmentation_Prediction-GradioUI.ipynb)

---

### 3.2 Satellite Disaster Assessment System (Phase 1 & Phase 2)

The system compares pre- and post-disaster satellite images to produce:
1. **Surface Change Detection**: Highlighted binary difference mask using `SiameseUNet`.
2. **Flood Inundation**: Flooded area segmentation using `FloodUNet`.
3. **Vegetation Loss & Water Expansion**: Calculated via class-to-class transition matrices.
4. **Disaster Heatmap**: Multi-hazard fused intensity map with Gaussian smoothing.
5. **Severity Scoring Engine**: Configurable 0–100 score (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`) configured in [`severity_config.yaml`](file:///d:/var-codes/satelite/DL-SatelliteImagery/disaster_assessment/configs/severity_config.yaml).

---

### 3.3 Trained Building Damage Assessment (4 Classes)

The system supports two distinct damage assessment modes:

#### Mode 1: Trained Neural Model (4 Classes)
When trained weights are loaded (`best_damage_model.pth`), the Siamese `BuildingDamageUNet` classifies damage into:
- 🟢 **`0: Undamaged`** (`#00C800`)
- 🟡 **`1: Minor Damage`** (`#FFD700`)
- 🟠 **`2: Major Damage`** (`#FF8C00`)
- 🔴 **`3: Destroyed`** (`#FF0000`)

#### Mode 2: Estimated Structural Change (Fallback)
If trained weights are unavailable or explicit estimated mode is requested, the system compares pre/post building masks using structural change heuristics:
- `Undamaged`, `Possible Damage`, `Severe Damage`
- Clearly labeled as *"Estimated damage based on structural change"*.

---

### 3.4 Georeferenced Flood Area Engine (GeoTIFF & GSD)

Supports 3 real-world area calculation modes:

1. **Mode 1: GeoTIFF (.tif/.tiff)**
   - Automatically reads CRS and affine transform using `rasterio`.
   - **Projected CRS** (e.g. UTM): Uses direct meter resolution.
   - **Geographic CRS** (EPSG:4326): Calculates geodesic metric resolution using central scene latitude ($dx = \cos(\text{lat}) \cdot 111319.5\text{ m}$).
2. **Mode 2: User-Supplied Resolution (`meters_per_pixel`)**
   - For standard JPG/PNG images (e.g. `meters_per_pixel=0.5` for WorldView-3). Computes exact $\text{m}^2$ and $\text{km}^2$.
3. **Mode 3: No Georeference Available**
   - Reports pixel counts and percentages only. **Never invents $\text{km}^2$**.

---

### 3.5 Launching the Unified Disaster Gradio Web UI

To launch the multi-tab interactive web interface:

```powershell
.\.venv\Scripts\python.exe DL-SatelliteImagery\disaster_gradio_app.py
```
Open `http://localhost:7860` in your browser:
- **Tab 1: Land-Cover Segmentation**: Upload a single image for 6-class color overlay.
- **Tab 2: Disaster Assessment**: Upload PRE and POST disaster images. Select Damage Mode (`Auto`, `Trained`, `Estimated`), optionally input Ground Resolution (`meters/pixel`) or upload a GeoTIFF raster.
- **Tab 3: Quick Change Detection**: Fast binary surface change highlighting.

---

### 3.6 Training the Building Damage Model

#### 1. Expected Dataset Directory Structure (xBD format):
```
data/damage_dataset/
├── train/
│   ├── pre/     # e.g., tile_001.png
│   ├── post/    # e.g., tile_001.png
│   └── masks/   # e.g., tile_001.png (pixel values: 0, 1, 2, 3)
└── val/
    ├── pre/
    ├── post/
    └── masks/
```

#### 2. Run Training:
```powershell
.\.venv\Scripts\python.exe DL-SatelliteImagery\disaster_assessment\training\train_building_damage.py --data_dir data/damage_dataset --epochs 20 --batch_size 4 --lr 0.001
```

#### 3. Run Self-Contained Dry Run (Synthetic Data):
```powershell
.\.venv\Scripts\python.exe DL-SatelliteImagery\disaster_assessment\training\train_building_damage.py --dry_run
```
The dry run writes synthetic data and a throwaway checkpoint under `runs/dry_run/` (git-ignored). The checkpoint is flagged as synthetic, so the app refuses to load it as a trained model.

---

### 3.7 Running Automated Verification Tests

To run the complete automated test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest DL-SatelliteImagery/tests -v
```

```
======================= 240 passed, 2 skipped (CPU, Python 3.11) =======================
```

---

### 3.8 Real Satellite Imagery Reference (`sample_images/`)

The repository includes authentic **NASA MODIS / VIIRS** satellite imagery (250m GSD) ready for testing and demonstrations:

| Directory / File | Description | Recommended Usage |
| :--- | :--- | :--- |
| **`pre_disaster.png` & `post_disaster.png`** | Pakistan Indus Valley flood (May vs. Sep 2022) | Primary pre/post disaster pair for Tab 2 |
| **`landcover_tiles/`** | Cairo (urban), Punjab (farms), Amazon (forest), Florida Keys (water), Dubai (infrastructure) | Single-image analysis in Tab 1 |
| **`disaster_pairs/`** | Australia bushfires, Amazon deforestation, Hurricane Milton, Turkey earthquake pairs | Before/after pairs for Tab 2 and Tab 3 |
| **`real_satellite_samples/`** | Nile Delta, Turkey wildfires, Ganges Delta, VIIRS continental views | Large-scale evaluation & visual inspection |
| **`geotiff_rasters/`** | Clean projected & geographic GeoTIFF rasters | Metric area calculations and CRS validation |

---

### 3.9 Python API Usage Example

```python
from disaster_assessment.pipeline import DisasterAnalyzer
import numpy as np
from PIL import Image

# Initialize analyzer (auto-detects CUDA / CPU)
analyzer = DisasterAnalyzer()

# Load image pair
pre_image = np.array(Image.open("pre_event.jpg"))
post_image = np.array(Image.open("post_event.jpg"))

# Run full disaster assessment with ground resolution
report = analyzer.analyze(
    pre_image=pre_image,
    post_image=post_image,
    damage_mode="auto",         # "auto", "trained", or "estimated"
    meters_per_pixel=0.5,       # 0.5 meters per pixel
)

# Print human-readable report & extract structured dictionary
print(report.summary())
results_dict = report.to_dict()
```

---

## 4. Troubleshooting & Common Issues

| Error / Symptom | Probable Cause | Recommended Fix / Solution |
| :--- | :--- | :--- |
| `CUDA out of memory` | Batch size too large for 4GB VRAM. | Reduce `batch_size=2` or `batch_size=1` in `damage_model_config.yaml` or training script. |
| `[WinError 1114] DLL initialization routine failed` | Using an external or unactivated Python environment. | Execute scripts using the repository's dedicated virtual environment: `.\.venv\Scripts\python.exe <script.py>` or activate via `.\.venv\Scripts\Activate.ps1`. |
| `rasterio not installed` | Missing optional geospatial package. | Run `pip install rasterio --only-binary :all:`. Fallback to user `meters_per_pixel` input is always supported. |
| `Trained damage model weights unavailable` | `best_damage_model.pth` not present. | Normal behavior in `auto` mode. The system automatically falls back to estimated structural change without crashing. |
| `Torch not compiled with CUDA enabled` | Installed CPU-only wheel. | Reinstall PyTorch with explicit CUDA index: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`. |
| `OSError: [Errno 98] / [WinError 10048] Address in use` | Port 7860 is occupied by another app. | Terminate previous process or launch on another port: `app.launch(server_port=7865)`. |

---

> [!TIP]
> All disaster assessment modules operate on both CUDA and CPU. For questions or enhancements, explore [`DL-SatelliteImagery/disaster_assessment/README.md`](./DL-SatelliteImagery/disaster_assessment/README.md).

