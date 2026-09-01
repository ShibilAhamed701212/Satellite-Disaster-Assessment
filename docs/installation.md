# Installation

## Quick Start (Windows PowerShell)

```powershell
# 1. Activate virtual environment
.\\.venv\\Scripts\\Activate.ps1

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install in editable mode
pip install -e .

# 4. (Optional) Install geospatial extras
pip install -e ".[geospatial]"
```

## Linux / macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Optional Dependency Groups

```bash
# Core only (existing functionality)
pip install -e .

# Geospatial (shapely, geopandas, folium, scipy, sklearn)
pip install -e ".[geospatial]"

# SAR processing
pip install -e ".[sar]"

# RAG copilot
pip install -e ".[rag]"

# ONNX export/optimization
pip install -e ".[optimization]"

# Everything
pip install -e ".[all]"
```

## Verify Installation

```python
# Check GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Check feature status
from disaster_assessment.feature_status import check_feature_status
registry = check_feature_status()
for feature in registry.get_all():
    print(f"[{feature.status.value}] {feature.feature}")
```

## Run Tests

```powershell
pytest
# Expected: 228 passed, 2 skipped
```

## Launch UI

```powershell
python app.py
# Opens at http://localhost:7860
```

## CLI Usage

```powershell
# Disaster assessment
python cli.py --pre pre.png --post post.png --gsd 250.0

# Single image segmentation
python cli.py --single satellite.jpg
```
