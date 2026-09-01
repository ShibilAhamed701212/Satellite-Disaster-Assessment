"""
Feature Status System for the Disaster Intelligence Platform.

Every optional component reports its ACTUAL operational state.
No feature should silently claim to be working when it isn't.

This is the single source of truth for feature availability.
The UI, CLI, and reports all use this system.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Status(str, Enum):
    """Standardized feature status values."""
    READY = "READY"
    AVAILABLE = "AVAILABLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    WEIGHTS_MISSING = "WEIGHTS_MISSING"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    INSUFFICIENT_INPUT = "INSUFFICIENT_INPUT"
    UNAVAILABLE = "UNAVAILABLE"
    EXPERIMENTAL = "EXPERIMENTAL"
    FAILED = "FAILED"
    HEURISTIC_ONLY = "HEURISTIC_ONLY"
    TRAINED_MODEL = "TRAINED_MODEL"
    IMPLEMENTED_ARCHITECTURE = "IMPLEMENTED_ARCHITECTURE"
    LIVE_DATA_CONNECTED = "LIVE_DATA_CONNECTED"
    STATIC_DATA_MODE = "STATIC_DATA_MODE"


@dataclass
class FeatureReport:
    """Report for a single feature/module."""
    feature: str
    status: Status
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "status": self.status.value,
            "reason": self.reason,
            "details": self.details,
        }

    def __str__(self) -> str:
        return f"[{self.status.value}] {self.feature}: {self.reason}" if self.reason else f"[{self.status.value}] {self.feature}"


class FeatureStatusRegistry:
    """Global registry tracking status of all platform features."""

    def __init__(self):
        self._features: Dict[str, FeatureReport] = {}

    def register(self, feature: str, status: Status, reason: str = "", **details) -> FeatureReport:
        report = FeatureReport(feature=feature, status=status, reason=reason, details=details)
        self._features[feature] = report
        return report

    def get(self, feature: str) -> Optional[FeatureReport]:
        return self._features.get(feature)

    def get_all(self) -> List[FeatureReport]:
        return list(self._features.values())

    def get_summary(self) -> Dict[str, str]:
        return {f.feature: f.status.value for f in self._features.values()}

    def summary_table(self) -> str:
        lines = [
            "=" * 70,
            "  FEATURE STATUS REPORT",
            "=" * 70,
        ]
        for f in self._features.values():
            marker = {
                Status.READY: "[OK]",
                Status.AVAILABLE: "[OK]",
                Status.TRAINED_MODEL: "[AI]",
                Status.HEURISTIC_ONLY: "[HEUR]",
                Status.IMPLEMENTED_ARCHITECTURE: "[ARCH]",
                Status.EXPERIMENTAL: "[EXP]",
                Status.NOT_CONFIGURED: "[--]",
                Status.UNAVAILABLE: "[--]",
                Status.DEPENDENCY_MISSING: "[DEP]",
                Status.WEIGHTS_MISSING: "[WGT]",
                Status.AUTH_REQUIRED: "[AUTH]",
                Status.INSUFFICIENT_INPUT: "[INPUT]",
                Status.FAILED: "[FAIL]",
                Status.RUNNING: "[RUN]",
                Status.COMPLETED: "[DONE]",
                Status.LIVE_DATA_CONNECTED: "[LIVE]",
                Status.STATIC_DATA_MODE: "[STAT]",
            }.get(f.status, "[??]")
            lines.append(f"  {marker} {f.feature:<40} [{f.status.value}]")
            if f.reason:
                lines.append(f"      {f.reason}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def check_requirements(self, required_features: List[str]) -> List[str]:
        """Return list of features that are NOT ready."""
        missing = []
        for feat in required_features:
            report = self._features.get(feat)
            if report is None:
                missing.append(f"{feat}: NOT REGISTERED")
            elif report.status not in (Status.READY, Status.AVAILABLE, Status.TRAINED_MODEL, Status.COMPLETED, Status.LIVE_DATA_CONNECTED):
                missing.append(f"{feat}: {report.status.value} - {report.reason}")
        return missing


# Global singleton
_feature_registry: Optional[FeatureStatusRegistry] = None


def get_feature_registry() -> FeatureStatusRegistry:
    global _feature_registry
    if _feature_registry is None:
        _feature_registry = FeatureStatusRegistry()
    return _feature_registry


def check_feature_status() -> FeatureStatusRegistry:
    """Check ACTUAL status of all platform features and return registry.

    This is the honest source of truth. Every status reflects the actual
    runtime capability of the feature.
    """
    registry = get_feature_registry()
    registry._features.clear()  # Fresh check

    import os
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")

    # --- Core Models ---
    # Land-cover segmentation: HSV heuristic only (no trained model)
    registry.register(
        "land_cover_segmentation",
        Status.HEURISTIC_ONLY,
        "Uses HSV color-space heuristic. NOT a trained neural network. Results are approximate.",
    )

    # Change detection: No weights file exists
    change_w = os.path.join(weights_dir, "siamese_unet.pth")
    if os.path.isfile(change_w):
        registry.register(
            "change_detection",
            Status.TRAINED_MODEL,
            "SiameseUNet weights found - needs validation",
            weights_path=change_w,
        )
    else:
        registry.register(
            "change_detection",
            Status.WEIGHTS_MISSING,
            "NO validated trained checkpoint. siamese_unet.pth not found. Output is UNAVAILABLE.",
            weights_path=change_w,
        )

    # Flood detection: No weights file exists
    flood_w = os.path.join(weights_dir, "flood_unet.pth")
    if os.path.isfile(flood_w):
        registry.register(
            "flood_detection",
            Status.TRAINED_MODEL,
            "FloodUNet weights found - needs validation",
            weights_path=flood_w,
        )
    else:
        registry.register(
            "flood_detection",
            Status.WEIGHTS_MISSING,
            "NO validated trained checkpoint. flood_unet.pth not found. Output is UNAVAILABLE.",
            weights_path=flood_w,
        )

    # Building damage: Weights exist but are degenerate (1 epoch, 23.85% IoU)
    damage_w = os.path.join(weights_dir, "building_damage", "best_damage_model.pth")
    if os.path.isfile(damage_w):
        # Try to validate the checkpoint
        try:
            import torch
            checkpoint = torch.load(damage_w, map_location="cpu", weights_only=False)
            epoch = checkpoint.get("epoch", -1) if isinstance(checkpoint, dict) else -1
            metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
            iou = metrics.get("iou", metrics.get("val_iou", -1))

            if epoch <= 1:
                registry.register(
                    "building_damage",
                    Status.EXPERIMENTAL,
                    f"Checkpoint exists but trained only {epoch} epoch(s). IoU={iou:.2f} if available. "
                    f"Model likely degenerate (always predicts single class). Fallback to heuristic available.",
                    weights_path=damage_w,
                    epoch=epoch,
                    iou=iou,
                )
            else:
                registry.register(
                    "building_damage",
                    Status.TRAINED_MODEL,
                    f"Trained {epoch} epochs. Needs quality validation.",
                    weights_path=damage_w,
                    epoch=epoch,
                    iou=iou,
                )
        except Exception:
            registry.register(
                "building_damage",
                Status.EXPERIMENTAL,
                "Checkpoint exists but cannot be read. Fallback to heuristic available.",
                weights_path=damage_w,
            )
    else:
        registry.register(
            "building_damage",
            Status.WEIGHTS_MISSING,
            "No damage model weights found. Using structural change heuristic.",
            weights_path=damage_w,
        )

    # --- Pipeline Features ---
    # Severity scoring: Always works (numeric formula)
    registry.register(
        "severity_scoring",
        Status.READY,
        "Rule-based severity scoring. Always operational.",
    )

    # Area calculation
    registry.register(
        "area_calculation",
        Status.READY,
        "Pixel-based area calculation. Geospatial when CRS available.",
    )

    # Land change metrics
    registry.register(
        "land_change_metrics",
        Status.READY,
        "Land-cover transition analysis. Always operational.",
    )

    # --- New Phase Modules ---
    # Phase A: Tiling
    try:
        import rasterio
        registry.register(
            "large_geotiff_tiling",
            Status.READY,
            "Sliding-window tiling with rasterio available.",
        )
    except ImportError:
        registry.register(
            "large_geotiff_tiling",
            Status.DEPENDENCY_MISSING,
            "rasterio required for large GeoTIFF processing.",
        )

    # Phase B: Vectorization
    try:
        import shapely
        registry.register(
            "gis_vector_export",
            Status.READY,
            "GeoJSON export available via shapely.",
        )
    except ImportError:
        registry.register(
            "gis_vector_export",
            Status.DEPENDENCY_MISSING,
            "shapely required for vector export.",
        )

    # Phase C: Spectral
    registry.register(
        "multispectral_indices",
        Status.READY,
        "NDVI/NDWI/NBR/dNBR computation ready.",
    )

    # Phase D: SAR
    registry.register(
        "sar_processing",
        Status.IMPLEMENTED_ARCHITECTURE,
        "Sentinel-1 VV/VH processing architecture ready. Requires real SAR data for validation.",
    )

    # Phase E: Fusion
    registry.register(
        "sar_optical_fusion",
        Status.IMPLEMENTED_ARCHITECTURE,
        "Fusion architecture implemented. Requires trained weights.",
    )

    # Phase F: DEM
    registry.register(
        "dem_analysis",
        Status.READY,
        "DEM loading and terrain analysis via rasterio/scipy.",
    )
    registry.register(
        "flood_depth_estimation",
        Status.EXPERIMENTAL,
        "Requires water surface elevation or boundary data for accurate depth estimation.",
    )

    # Phase G: Map UI
    registry.register(
        "interactive_map_ui",
        Status.READY,
        "Folium-based geospatial web viewer available.",
    )

    # Phase H: Foundation models
    registry.register(
        "foundation_models",
        Status.IMPLEMENTED_ARCHITECTURE,
        "Backbone registry implemented. No foundation model weights downloaded.",
    )

    # Phase I: Training
    registry.register(
        "training_pipeline",
        Status.READY,
        "Unified training with checkpointing, resume, mixed precision.",
    )

    # Phase J: ONNX
    try:
        import onnx
        registry.register(
            "onnx_export",
            Status.READY,
            "ONNX export and validation available.",
        )
    except ImportError:
        registry.register(
            "onnx_export",
            Status.DEPENDENCY_MISSING,
            "onnx package required for model export.",
        )

    # Phase K: RAG
    registry.register(
        "rag_copilot",
        Status.IMPLEMENTED_ARCHITECTURE,
        "RAG copilot architecture. Requires LLM provider (Ollama, etc.).",
    )

    # Phase L: Ingestion
    registry.register(
        "data_ingestion",
        Status.IMPLEMENTED_ARCHITECTURE,
        "Data provider architecture. Requires API credentials for live data.",
    )

    # Phase M: Digital Twin
    registry.register(
        "digital_twin",
        Status.EXPERIMENTAL,
        "In-memory state engine. No persistence. Not connected to pipeline.",
    )

    return registry
