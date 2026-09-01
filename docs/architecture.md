# System Architecture

## Overview

The Satellite Disaster Assessment & Remote Sensing Intelligence Platform v3.0 is a modular, production-oriented geospatial disaster intelligence system built on the existing working foundation.

## Architecture Diagram

```
INPUT DATA (GeoTIFF / PNG / JPG / SAR)
    |
VALIDATION (ImageValidator)
    |
DATA TYPE DETECTION (Optical / SAR / Multispectral)
    |
    +---> TILING (sliding window for large rasters)
    |
OPTICAL / SAR / MULTISPECTRAL PROCESSING
    |
    +---> Preprocessing (normalization, speckle filter, band extraction)
    |
MODEL INFERENCE
    +---> Land-cover segmentation (6-class U-Net)
    +---> Change detection (Siamese U-Net)
    +---> Flood detection (Flood U-Net)
    +---> Building damage (4-class Siamese UNet)
    +---> SAR-optical fusion (optional)
    |
PREDICTION BLENDING (Gaussian/uniform overlap averaging)
    |
GEOSPATIAL ANALYTICS
    +---> Area calculation (pixel-based + geodesic)
    +---> Land change metrics (6x6 transition matrix)
    +---> Damage metrics
    +---> Severity engine (configurable YAML weights)
    |
SPECTRAL ANALYTICS (optional)
    +---> NDVI, NDWI, MNDWI, NBR, dNBR
    |
DEM ANALYSIS (optional)
    +---> Terrain (slope, aspect, hillshade)
    +---> Flood depth estimation
    +---> Volume calculation
    |
VECTOR GENERATION
    +---> Polygonize -> GeoJSON / Shapefile / KML
    |
DIGITAL TWIN STATE UPDATE
    |
RAG / COPILOT ACCESS
    |
UI + API + CLI + EXPORT
```

## Directory Structure

```
DL-SatelliteImagery/disaster_assessment/
    __init__.py                  # Package init (v3.0.0)
    feature_status.py            # Feature status tracking system
    models/
        base_unet.py             # Shared encoder/decoder (existing)
        siamese_unet.py          # Change detection (existing)
        flood_unet.py            # Flood segmentation (existing)
        building_damage_model.py # 4-class damage (existing)
        backbones/
            base.py              # Backbone interface
            registry.py          # Model registry
            unet_backbone.py     # Base UNet adapter
            foundation_backbone.py # SegFormer/Swin/Prithvi adapters
        fusion/
            sar_optical_fusion.py # SAR+Optical fusion U-Net
            feature_fusion.py    # Early/feature-level fusion
    preprocessing/
        image_validation.py      # Image validation (existing)
        image_alignment.py       # Image alignment (existing)
        tiling/
            tile_config.py       # Tiling configuration
            tile_generator.py    # Sliding-window tile generator
            blending.py          # Prediction overlap blending
            raster_reader.py     # Large raster streaming reader
        sar/
            sentinel1.py         # Sentinel-1 processor
            normalization.py     # SAR normalization
            speckle_filter.py    # Median/Lee speckle filters
            coherence.py         # SAR change features
    analytics/
        area_calculator.py       # Area calculation (existing)
        damage_metrics.py        # Damage metrics (existing)
        geospatial_area.py       # GeoTIFF area (existing)
        land_change_metrics.py   # Land change (existing)
        severity_engine.py       # Severity scoring (existing)
        vectorization/
            polygonizer.py       # Raster-to-vector conversion
            geojson_export.py    # GeoJSON export
            shapefile_export.py  # ESRI Shapefile export
            kml_export.py        # KML export
        spectral/
            indices.py           # NDVI/NDWI/NBR/dNBR
            band_mapping.py      # Sensor band mappings
            validators.py        # Band validation
        elevation/
            dem_loader.py        # DEM loading
            terrain_analysis.py  # Slope/aspect/hillshade
            flood_depth.py       # Flood depth estimation
            volume.py            # Water volume calculation
    inference/
        building_damage_inference.py # Dual-mode damage (existing)
    visualization/
        overlays.py              # Overlay rendering (existing)
        heatmap.py               # Heatmap generation (existing)
        comparison.py            # Comparison views (existing)
        map_viewer.py            # Folium geospatial map viewer
    training/
        train_building_damage.py # Training pipeline (existing)
    configs/
        severity_config.yaml     # Severity weights (existing)
        damage_model_config.yaml # Model config (existing)
        default.yaml             # Centralized default config
        models.yaml              # Model definitions
        pipeline.yaml            # Pipeline stage config
        data_sources.yaml        # External data provider config
    copilot/
        agent.py                 # RAG copilot agent
        rag/
            ingestion.py         # Document ingestion
            retriever.py         # Document retrieval
    data/
        ingestion/
            base.py              # Data provider interface
            cache.py             # Data cache
            scheduler.py         # Ingestion scheduler
            sources/
                sentinel.py      # Sentinel data provider
                weather.py       # Weather data provider
                mosdac.py        # MOSDAC provider
                bhoonidhi.py     # Bhoonidhi provider
    twin/
        state.py                 # Digital twin state engine
        entities.py              # Entity definitions
        versioning.py            # State versioning
    deployment/
        export_onnx.py           # ONNX export/validation
        benchmark.py             # Model benchmarking
```

## Backward Compatibility

All existing 92 tests continue to pass. The existing `DisasterAnalyzer`, `DisasterReport`, CLI, Gradio UI, and all public APIs are preserved without breaking changes.
