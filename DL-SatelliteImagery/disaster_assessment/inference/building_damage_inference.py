"""
Building Damage Inference Engine supporting both Trained and Estimated Assessment.

Modes:
    1. MODE "trained":
       - Executes the validated 4-class BuildingDamageUNet
       - Requires validated trained weights that pass quality checks
       - Returns: Undamaged, Minor Damage, Major Damage, Destroyed

    2. MODE "estimated":
       - Executes Phase 1 structural change heuristic on pre/post building masks
       - Returns: Undamaged, Possible Damage, Severe Damage
       - Label clearly states this is NOT a trained model

    3. MODE "auto":
       - Checks if trained weights exist AND pass validation
       - If validated -> Executes "trained" mode
       - If not validated -> Falls back to "estimated" with clear notification

Safety:
    - Degenerate models are detected and refused
    - Random weights never produce fake inference
    - Heuristic results clearly labeled
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..analytics.damage_metrics import DamageMetrics, BuildingDamageResult
from ..models.base_unet import get_device
from ..models.building_damage_model import (
    DAMAGE_CLASSES,
    NUM_DAMAGE_CLASSES,
    BuildingDamageUNet,
    create_building_damage_model,
)
from ..models.validation import ModelValidator, ModelStatus, ModelInferenceMode, ValidationResult


# Color mapping for the 4 trained classes (RGB)
TRAINED_DAMAGE_COLORS = {
    0: (0, 200, 0),     # Undamaged - Green
    1: (255, 215, 0),   # Minor Damage - Gold/Yellow
    2: (255, 140, 0),   # Major Damage - Dark Orange
    3: (255, 0, 0),     # Destroyed - Red
}


@dataclass
class DamageAssessmentOutput:
    """Standardized output structure for building damage assessment."""

    damage_mode: str = "estimated"  # "trained" or "estimated"
    model_available: bool = False
    warning: Optional[str] = None

    # 4-class trained breakdown
    undamaged_percentage: float = 0.0
    minor_damage_percentage: float = 0.0
    major_damage_percentage: float = 0.0
    destroyed_percentage: float = 0.0

    # Counts
    class_pixel_counts: Dict[str, int] = field(default_factory=dict)
    total_building_pixels: int = 0

    # Overall score (0-100)
    overall_damage_score: float = 0.0

    # Visual map
    damage_map: Optional[np.ndarray] = None
    disclaimer: str = ""

    # Embedded Phase 1 result if estimated mode was used
    estimated_details: Optional[BuildingDamageResult] = None

    def to_dict(self) -> dict:
        data = {
            "damage_mode": self.damage_mode,
            "model_available": self.model_available,
            "overall_damage_score": round(self.overall_damage_score, 2),
            "total_building_pixels": self.total_building_pixels,
            "disclaimer": self.disclaimer,
        }
        if self.warning:
            data["warning"] = self.warning

        if self.damage_mode == "trained":
            data["classes"] = {
                "undamaged_percentage": round(self.undamaged_percentage, 2),
                "minor_damage_percentage": round(self.minor_damage_percentage, 2),
                "major_damage_percentage": round(self.major_damage_percentage, 2),
                "destroyed_percentage": round(self.destroyed_percentage, 2),
            }
            data["pixel_counts"] = self.class_pixel_counts
        else:
            if self.estimated_details:
                data["estimated_classes"] = {
                    "undamaged_percentage": round(self.estimated_details.undamaged_percentage, 2),
                    "possible_damage_percentage": round(self.estimated_details.possible_damage_percentage, 2),
                    "severe_damage_percentage": round(self.estimated_details.severe_damage_percentage, 2),
                }

        return data

    def summary(self) -> str:
        lines = [
            "=" * 50,
            f"  BUILDING DAMAGE ASSESSMENT ({self.damage_mode.upper()} MODE)",
            "=" * 50,
            f"  Model Source: {'Trained Neural Model' if self.damage_mode == 'trained' else 'Structural Change Heuristic'}",
            f"  Overall Damage Score: {self.overall_damage_score:.1f}/100",
            f"  Total Building Pixels: {self.total_building_pixels:,}",
            "",
        ]

        if self.damage_mode == "trained":
            lines.extend([
                "  4-Class Trained Distribution:",
                f"    * Undamaged:    {self.undamaged_percentage:>6.2f}% ({self.class_pixel_counts.get('Undamaged', 0):,} px)",
                f"    * Minor Damage: {self.minor_damage_percentage:>6.2f}% ({self.class_pixel_counts.get('Minor Damage', 0):,} px)",
                f"    * Major Damage: {self.major_damage_percentage:>6.2f}% ({self.class_pixel_counts.get('Major Damage', 0):,} px)",
                f"    * Destroyed:    {self.destroyed_percentage:>6.2f}% ({self.class_pixel_counts.get('Destroyed', 0):,} px)",
            ])
        else:
            if self.estimated_details:
                lines.extend([
                    "  Estimated Change Distribution:",
                    f"    * Undamaged:       {self.estimated_details.undamaged_percentage:>6.2f}%",
                    f"    * Possible Damage: {self.estimated_details.possible_damage_percentage:>6.2f}%",
                    f"    * Severe Damage:   {self.estimated_details.severe_damage_percentage:>6.2f}%",
                ])

        if self.warning:
            lines.extend(["", f"  ! Note: {self.warning}"])

        lines.extend(["", f"  i {self.disclaimer}"])
        return "\n".join(lines)


class BuildingDamageInference:
    """
    Orchestrates building damage assessment across trained and estimated modes.

    Safety:
    - Validates checkpoint before inference
    - Detects degenerate weights
    - Refuses to produce fake results
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.weights_path = weights_path
        self.device = device or get_device()
        self.model: Optional[BuildingDamageUNet] = None
        self.fallback_metrics = DamageMetrics()
        self._validation: Optional[ValidationResult] = None
        self._validator = ModelValidator()

    def has_trained_weights(self) -> bool:
        """Check if valid weights file exists on disk."""
        return self.weights_path is not None and os.path.isfile(self.weights_path)

    def validate(self) -> dict:
        """Validate the damage model checkpoint and return status."""
        if self._validation is not None:
            return self._validation.to_dict()

        self._validation = self._validator.validate_checkpoint(
            weights_path=self.weights_path,
            model_class=BuildingDamageUNet,
            model_name="building_damage",
            device=self.device,
        )
        return self._validation.to_dict()

    def _load_model_if_needed(self) -> bool:
        """Load model only if validated."""
        if self.model is not None:
            return True

        # Validate first
        if self._validation is None:
            self.validate()

        if self._validation is None or not self._validation.is_usable:
            return False

        try:
            self.model = create_building_damage_model(
                weights_path=self.weights_path,
                device=self.device,
            )
            return True
        except Exception as e:
            print(f"[BuildingDamageInference] Error loading weights: {e}")
            self.model = None
            return False

    def assess(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        pre_segmentation: Optional[np.ndarray] = None,
        post_segmentation: Optional[np.ndarray] = None,
        mode: str = "auto",
    ) -> DamageAssessmentOutput:
        """
        Assess building damage using validated model or estimation heuristic.

        Args:
            pre_image: [H, W, 3] uint8 pre-disaster image.
            post_image: [H, W, 3] uint8 post-disaster image.
            pre_segmentation: [H, W] land-cover mask (used for estimation fallback).
            post_segmentation: [H, W] land-cover mask (used for estimation fallback).
            mode: "auto", "trained", or "estimated".

        Returns:
            Standardized DamageAssessmentOutput.
        """
        target_mode = mode.lower()

        # Handle "auto" mode
        if target_mode == "auto":
            if self.has_trained_weights() and self._load_model_if_needed():
                return self._run_trained_inference(pre_image, post_image)
            else:
                reason = ""
                if self._validation:
                    if self._validation.status == ModelStatus.DEGENERATE:
                        reason = "Trained model weights are degenerate (single-class collapse). "
                    elif self._validation.status == ModelStatus.WEIGHTS_MISSING:
                        reason = "No trained damage model weights found. "
                    elif self._validation.status == ModelStatus.VALIDATION_FAILED:
                        reason = "Trained model failed validation (possibly untrained or quality below threshold). "
                    elif self._validation.status == ModelStatus.CHECKPOINT_INVALID:
                        reason = "Checkpoint file is invalid or corrupted. "
                    else:
                        reason = f"Model validation status: {self._validation.status.value}. "
                else:
                    reason = "No trained damage model configured. "

                return self._run_estimated_fallback(
                    pre_segmentation,
                    post_segmentation,
                    warning=reason + "Using structural change estimation.",
                )

        # Handle explicit "trained" mode
        elif target_mode == "trained":
            if self.has_trained_weights() and self._load_model_if_needed():
                return self._run_trained_inference(pre_image, post_image)
            else:
                return self._run_estimated_fallback(
                    pre_segmentation,
                    post_segmentation,
                    warning="Explicit 'trained' mode requested, but no validated weights found. Fallback to estimated change.",
                )

        # Handle explicit "estimated" mode
        else:
            return self._run_estimated_fallback(
                pre_segmentation,
                post_segmentation,
                warning=None,
            )

    def _run_trained_inference(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
    ) -> DamageAssessmentOutput:
        """Execute validated trained 4-class neural segmentation."""
        self.model.eval()

        # Convert numpy (H, W, 3) to tensor (1, 3, H, W)
        pre_tensor = torch.from_numpy(pre_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        post_tensor = torch.from_numpy(post_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    class_map_tensor, probs = self.model.predict(pre_tensor, post_tensor)
            else:
                class_map_tensor, probs = self.model.predict(pre_tensor, post_tensor)

        damage_map = class_map_tensor.squeeze().cpu().numpy().astype(np.uint8)

        # Calculate class distribution
        total_pixels = damage_map.size
        counts = {}
        for cls_id, cls_name in DAMAGE_CLASSES.items():
            counts[cls_name] = int((damage_map == cls_id).sum())

        undamaged_pct = (counts["Undamaged"] / max(total_pixels, 1)) * 100.0
        minor_pct = (counts["Minor Damage"] / max(total_pixels, 1)) * 100.0
        major_pct = (counts["Major Damage"] / max(total_pixels, 1)) * 100.0
        destroyed_pct = (counts["Destroyed"] / max(total_pixels, 1)) * 100.0

        # Weighted severity score across the 4 classes:
        # Undamaged: 0%, Minor: 25%, Major: 75%, Destroyed: 100%
        damage_score = min(
            100.0,
            (minor_pct * 0.25) + (major_pct * 0.75) + (destroyed_pct * 1.0),
        )

        # Build disclaimer with validation info
        disclaimer_parts = ["TRAINED DAMAGE MODEL: 4-class classification"]
        if self._validation and self._validation.metadata:
            m = self._validation.metadata
            if m.epoch >= 0:
                disclaimer_parts.append(f"trained epoch {m.epoch}")
            if m.metrics:
                metric_strs = [f"{k}={v:.3f}" for k, v in m.metrics.items()]
                disclaimer_parts.append(f"metrics: {', '.join(metric_strs)}")

        return DamageAssessmentOutput(
            damage_mode="trained",
            model_available=True,
            undamaged_percentage=undamaged_pct,
            minor_damage_percentage=minor_pct,
            major_damage_percentage=major_pct,
            destroyed_percentage=destroyed_pct,
            class_pixel_counts=counts,
            total_building_pixels=total_pixels,
            overall_damage_score=damage_score,
            damage_map=damage_map,
            disclaimer=". ".join(disclaimer_parts),
        )

    def _run_estimated_fallback(
        self,
        pre_segmentation: Optional[np.ndarray],
        post_segmentation: Optional[np.ndarray],
        warning: Optional[str],
    ) -> DamageAssessmentOutput:
        """Execute Phase 1 heuristic fallback."""
        if pre_segmentation is None or post_segmentation is None:
            h, w = (256, 256)
            pre_segmentation = np.zeros((h, w), dtype=np.uint8)
            post_segmentation = np.zeros((h, w), dtype=np.uint8)

        est_result = self.fallback_metrics.estimate_damage(pre_segmentation, post_segmentation)

        return DamageAssessmentOutput(
            damage_mode="estimated",
            model_available=False,
            warning=warning,
            undamaged_percentage=est_result.undamaged_percentage,
            minor_damage_percentage=0.0,
            major_damage_percentage=0.0,
            destroyed_percentage=0.0,
            class_pixel_counts={
                "Undamaged": est_result.undamaged_pixels,
                "Possible Damage": est_result.possible_damage_pixels,
                "Severe Damage": est_result.severe_damage_pixels,
            },
            total_building_pixels=est_result.total_building_pixels_pre,
            overall_damage_score=est_result.overall_damage_score,
            damage_map=est_result.damage_map,
            disclaimer=est_result.disclaimer,
            estimated_details=est_result,
        )
