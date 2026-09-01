# Configuration

## Centralized Configuration

All configuration is in `configs/` directory:

### default.yaml
Main pipeline configuration controlling which modules are active.

### models.yaml
Model definitions, channel configurations, and weight paths.

### pipeline.yaml
Pipeline stage ordering and memory management settings.

### data_sources.yaml
External data provider configuration (Sentinel, weather, MOSDAC, Bhoonidhi).

## Configuration Example

```yaml
# configs/default.yaml
analysis:
  land_cover: true
  change_detection: true
  flood_detection: true
  building_damage: true
  building_damage_mode: "auto"

spectral:
  enabled: true
  indices: [ndvi, ndwi, nbr]

sar:
  enabled: false
  speckle_filter:
    enabled: true
    method: "median"
    kernel_size: 5

dem:
  enabled: false
  flood_depth: false

tiling:
  tile_size: 512
  overlap: 0.25
  blend_mode: "gaussian"

vector_export:
  enabled: true
  format: "geojson"

device: "auto"  # auto, cpu, cuda
```

## Environment Variables

No hardcoded machine-specific paths. All paths are configurable via YAML or constructor parameters.

## Hardware Requirements

- **Minimum**: CPU-only, 8GB RAM
- **Recommended**: NVIDIA GPU with 4GB+ VRAM (GTX 1650 compatible)
- **Optimal**: NVIDIA GPU with 8GB+ VRAM for mixed precision
