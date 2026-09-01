"""
Centralized Configuration System for the Disaster Assessment Platform.

Loads YAML configuration files and provides a single source of truth
for all pipeline behavior, model settings, and feature flags.

Configuration precedence:
1. Explicit overrides
2. Environment variables
3. YAML config files
4. Built-in defaults
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str = ""
    weights_path: Optional[str] = None
    enabled: bool = True
    threshold: float = 0.5
    minimum_iou: Optional[float] = None
    minimum_f1: Optional[float] = None
    minimum_epoch: Optional[int] = None


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    tile_size: int = 512
    overlap: int = 128
    target_size: tuple = (256, 256)
    auto_tile_threshold: int = 1024
    device: str = "auto"


@dataclass
class FeatureConfig:
    """Configuration for optional feature flags."""
    land_cover: bool = True
    change_detection: bool = True
    flood_detection: bool = True
    building_damage: bool = True
    spectral_analysis: bool = True
    sar: bool = False
    dem: bool = False
    vector_export: bool = True
    digital_twin: bool = True
    copilot: bool = False
    interactive_map: bool = True
    onnx_export: bool = False
    tensorrt: bool = False


@dataclass
class ExportConfig:
    """Configuration for export settings."""
    formats: List[str] = field(default_factory=lambda: ["geojson"])
    output_dir: str = "results"


@dataclass
class DisasterConfig:
    """Top-level configuration object."""
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    severity_config_path: Optional[str] = None
    gsd_meters: Optional[float] = None

    @classmethod
    def from_yaml(cls, config_path: str) -> "DisasterConfig":
        """Load configuration from a YAML file."""
        if not os.path.isfile(config_path):
            print(f"[Config] Warning: Config file not found at {config_path}. Using defaults.")
            return cls()

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "DisasterConfig":
        """Create configuration from a dictionary."""
        config = cls()

        # Parse model configs
        models_data = data.get("models", {})
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
        for name, model_data in models_data.items():
            if isinstance(model_data, dict):
                wp = model_data.get("weights", None)
                # Resolve relative paths from weights dir
                if wp and not os.path.isabs(wp):
                    wp = os.path.join(weights_dir, wp)
                config.models[name] = ModelConfig(
                    name=name,
                    weights_path=wp if wp and os.path.isfile(wp) else None,
                    enabled=model_data.get("enabled", True),
                    threshold=model_data.get("threshold", 0.5),
                    minimum_iou=model_data.get("minimum_iou"),
                    minimum_f1=model_data.get("minimum_f1"),
                    minimum_epoch=model_data.get("minimum_epoch"),
                )

        # Parse processing config
        proc = data.get("processing", {})
        if proc:
            config.processing = ProcessingConfig(
                tile_size=proc.get("tile_size", 512),
                overlap=proc.get("overlap", 128),
                target_size=tuple(proc.get("target_size", [256, 256])),
                auto_tile_threshold=proc.get("auto_tile_threshold", 1024),
                device=proc.get("device", "auto"),
            )

        # Parse feature flags
        features = data.get("features", {})
        if features:
            config.features = FeatureConfig(
                land_cover=features.get("land_cover", True),
                change_detection=features.get("change_detection", True),
                flood_detection=features.get("flood_detection", True),
                building_damage=features.get("building_damage", True),
                spectral_analysis=features.get("spectral_analysis", True),
                sar=features.get("sar", False),
                dem=features.get("dem", False),
                vector_export=features.get("vector_export", True),
                digital_twin=features.get("digital_twin", True),
                copilot=features.get("copilot", False),
                interactive_map=features.get("interactive_map", True),
                onnx_export=features.get("onnx_export", False),
                tensorrt=features.get("tensorrt", False),
            )

        # Parse export config
        export = data.get("export", {})
        if export:
            config.export = ExportConfig(
                formats=export.get("formats", ["geojson"]),
                output_dir=export.get("output_dir", "results"),
            )

        config.severity_config_path = data.get("severity_config_path")
        config.gsd_meters = data.get("gsd_meters")

        return config

    @classmethod
    def default(cls) -> "DisasterConfig":
        """Load from default.yaml if it exists, otherwise use built-in defaults."""
        default_path = os.path.join(DEFAULT_CONFIG_DIR, "default.yaml")
        if os.path.isfile(default_path):
            return cls.from_yaml(default_path)
        return cls()

    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.models.get(name)

    def get_quality_config(self) -> Dict[str, Dict[str, float]]:
        """Extract quality thresholds for the model validator."""
        result = {}
        for name, mc in self.models.items():
            thresholds = {}
            if mc.minimum_iou is not None:
                thresholds["minimum_iou"] = mc.minimum_iou
            if mc.minimum_f1 is not None:
                thresholds["minimum_f1"] = mc.minimum_f1
            if mc.minimum_epoch is not None:
                thresholds["minimum_epoch"] = mc.minimum_epoch
            if thresholds:
                result[name] = thresholds
        return result


# Global configuration singleton
_config: Optional[DisasterConfig] = None


def get_config() -> DisasterConfig:
    """Get the global configuration, loading default.yaml if available."""
    global _config
    if _config is None:
        _config = DisasterConfig.default()
    return _config


def load_config(config_path: str) -> DisasterConfig:
    """Load and set the global configuration from a specific path."""
    global _config
    _config = DisasterConfig.from_yaml(config_path)
    return _config


def reset_config():
    """Reset to default configuration."""
    global _config
    _config = None
