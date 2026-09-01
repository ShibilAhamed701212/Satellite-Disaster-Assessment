# Deep Learning for Satellite Imagery & Disaster Assessment

A comprehensive repository for aerial and satellite remote sensing, land-cover semantic segmentation, and automated disaster impact analysis.

---

## 🚀 Quick Launch: Satellite Disaster Assessment Web UI

To launch the multi-tab interactive dashboard:

```powershell
.\.venv\Scripts\python.exe DL-SatelliteImagery\disaster_gradio_app.py
```
Access at `http://localhost:7860`:
- **Tab 1 — Land-Cover Segmentation**: Upload a satellite image to view 6-class semantic segmentation (`Water`, `Land`, `Road`, `Building`, `Vegetation`, `Unlabeled`).
- **Tab 2 — Disaster Assessment**: Upload PRE and POST disaster images. Select Damage Assessment Mode (`Auto`, `Trained`, `Estimated`), optionally input Ground Resolution (`meters/pixel`) or upload a GeoTIFF raster.
- **Tab 3 — Surface Change Detection**: Binary surface change highlighting between image pairs.

---

## 🛰️ System Capabilities (Phase 1 & Phase 2)

| Feature | Model / Module | Purpose | Output / Classes |
| :--- | :--- | :--- | :--- |
| **Land-Cover Segmentation** | Keras U-Net / Color Estimator | Baseline semantic land-cover mapping | 6 Classes: `Water`, `Land`, `Road`, `Building`, `Vegetation`, `Unlabeled` |
| **Surface Change Detection** | [`SiameseUNet`](./disaster_assessment/models/siamese_unet.py) (~3.54M params) | Detects surface alterations between pre/post image pairs | Binary Mask (0=No Change, 1=Changed) |
| **Flood Inundation Mapping** | [`FloodUNet`](./disaster_assessment/models/flood_unet.py) (~2.36M params) | Segments flooded regions and submerged infrastructure | Binary Flood Mask |
| **Georeferenced Metric Area** | [`geospatial_area.py`](./disaster_assessment/analytics/geospatial_area.py) | GeoTIFF metadata parsing & geodesic latitude scaling | Metric $m^2$ and $km^2$ flood area |
| **Building Damage Assessment** | [`BuildingDamageUNet`](./disaster_assessment/models/building_damage_model.py) (~3.84M params) | 4-class Siamese neural damage segmentation | 4 Classes: `Undamaged`, `Minor Damage`, `Major Damage`, `Destroyed` |
| **Disaster Severity Engine** | [`severity_engine.py`](./disaster_assessment/analytics/severity_engine.py) | Rule-based multi-hazard severity scoring | 0–100 Score (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`) |

---

## 📁 Project Structure

```
DL-SatelliteImagery/
├── disaster_gradio_app.py            # Unified 3-Tab Gradio Web Dashboard
│
├── disaster_assessment/              # Modular Disaster Assessment Engine
│   ├── models/                       # PyTorch U-Net & Siamese architectures
│   │   ├── base_unet.py
│   │   ├── siamese_unet.py
│   │   ├── flood_unet.py
│   │   └── building_damage_model.py
│   ├── datasets/                     # Dataset loader, augmentations & synthetic generator
│   │   └── building_damage_dataset.py
│   ├── training/                     # Building damage training pipeline
│   │   └── train_building_damage.py
│   ├── inference/                    # Dual-mode damage inference (Trained vs. Estimated)
│   │   └── building_damage_inference.py
│   ├── analytics/                    # Area, damage, land change & severity engines
│   │   ├── geospatial_area.py
│   │   ├── area_calculator.py
│   │   ├── damage_metrics.py
│   │   ├── land_change_metrics.py
│   │   └── severity_engine.py
│   ├── visualization/                # Overlays, heatmaps & comparison grids
│   │   ├── overlays.py
│   │   ├── heatmap.py
│   │   └── comparison.py
│   ├── configs/                      # Configuration files
│   │   ├── severity_config.yaml
│   │   └── damage_model_config.yaml
│   └── weights/                      # Trained model weight checkpoints
│       └── building_damage/
│
├── tests/                            # Comprehensive automated test suite (94 tests)
│   ├── test_models.py
│   ├── test_preprocessing.py
│   ├── test_analytics.py
│   ├── test_pipeline.py
│   ├── test_phase2_damage.py
│   └── test_phase2_geospatial.py
│
└── [ORIGINAL WORKSHOP NOTEBOOKS]
    ├── Satellite_Imagery_Segmentation.ipynb
    ├── Satellite_Imagery_DeepLearning-Base.ipynb
    ├── Satellite_Imagery_DeepLearning-LocalDiag.ipynb
    ├── Satellite_Imagery_DeepLearning-SaveLoadModel.ipynb
    ├── Satellite_Imagery_DeepLearning-WandB.ipynb
    ├── Satellite_Imagery_DeepLearning_ActivationHeatmap.ipynb
    ├── Satellite_segmentation_Prediction-Base.ipynb
    └── Satellite_segmentation_Prediction-GradioUI.ipynb
```

---

## 📺 Original Deep Learning Workshop Series

### Part 1: Satellite Images Data Processing
<table class="table table-striped table-bordered table-vcenter">
  <tr>
    <td align="center"><b>🔥&nbsp;Deep Learning Workshop for Satellite Imagery - Data Processing (Part 1/3)</b></td>
  </tr>
  <tr>
    <td>
      <div>
        <a href="https://www.youtube.com/watch?v=3Xn21RT-y7Y"><img src="https://img.youtube.com/vi/3Xn21RT-y7Y/0.jpg" alt="Part 1 Video"></a>
      </div>
    </td>
  </tr>
</table>

- **Notebook**: [`Satellite_Imagery_Segmentation.ipynb`](./Satellite_Imagery_Segmentation.ipynb)

---

### Part 2: Deep Learning for Satellite Imagery
<table class="table table-striped table-bordered table-vcenter">
  <tr>
    <td align="center"><b>🔥&nbsp;Deep Learning Workshop for Satellite Imagery - Training & Prediction (Part 2/3)</b></td>
  </tr>
  <tr>
    <td>
      <div>
        <a href="https://www.youtube.com/watch?v=UBzMgr6yfpw"><img src="https://img.youtube.com/vi/UBzMgr6yfpw/0.jpg" alt="Part 2 Video"></a>
      </div>
    </td>
  </tr>
</table>

- **Base Training**: [`Satellite_Imagery_DeepLearning-Base.ipynb`](./Satellite_Imagery_DeepLearning-Base.ipynb)
- **Local Diagnostics**: [`Satellite_Imagery_DeepLearning-LocalDiag.ipynb`](./Satellite_Imagery_DeepLearning-LocalDiag.ipynb)
- **Checkpoint Management**: [`Satellite_Imagery_DeepLearning-SaveLoadModel.ipynb`](./Satellite_Imagery_DeepLearning-SaveLoadModel.ipynb)

---

### Part 3: Advanced Concepts & MLOps
- **W&B Experiment Tracking**: [`Satellite_Imagery_DeepLearning-WandB.ipynb`](./Satellite_Imagery_DeepLearning-WandB.ipynb)
- **Activation Heatmaps (XAI)**: [`Satellite_Imagery_DeepLearning_ActivationHeatmap.ipynb`](./Satellite_Imagery_DeepLearning_ActivationHeatmap.ipynb)
- **Prediction Web UI**: [`Satellite_segmentation_Prediction-GradioUI.ipynb`](./Satellite_segmentation_Prediction-GradioUI.ipynb)

---

## 🧪 Testing & Verification

Run the full automated test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest DL-SatelliteImagery/tests -v
```

```text
======================== 92 passed, 2 skipped in 14.45s ========================
```

---

## 📚 Dataset References & Resources

- **Dubai Aerial Segmentation Dataset**: [Humans in the Loop](https://humansintheloop.org/resources/datasets/semantic-segmentation-dataset-2/) | [Kaggle](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)
- **xBD / xView2 Building Damage Challenge**: [xView2.org](https://xview2.org/)
- **Sen1Floods11 Dataset**: [Sen1Floods11 GitHub](https://github.com/cloudtostreet/Sen1Floods11)
- **SpaceNet 6 Challenge (38GB)**: [SpaceNet AI](https://spacenet.ai/sn6-challenge/)
- **Segmentation Models Library**: [qubvel/segmentation_models](https://github.com/qubvel/segmentation_models)
- **Keract Activation Library**: [philipperemy/keract](https://github.com/philipperemy/keract)
