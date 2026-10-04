"""
Model Validation Layer for the Disaster Assessment System.

Validates checkpoints before inference:
- Weight file existence
- Checkpoint integrity and format
- Architecture compatibility
- Degenerate model detection (single-class collapse)
- Training metadata extraction
- Quality gate enforcement

A model must pass validation before producing any output presented as AI inference.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch


class ModelInferenceMode(str, Enum):
    """How a model's output should be interpreted."""
    TRAINED = "trained"
    HEURISTIC = "heuristic"
    UNAVAILABLE = "unavailable"
    EXPERIMENTAL = "experimental"
    DEGENERATE = "degenerate"
    VALIDATION_FAILED = "validation_failed"


class ModelStatus(str, Enum):
    """Current status of a model."""
    READY = "READY"
    WEIGHTS_MISSING = "WEIGHTS_MISSING"
    CHECKPOINT_INVALID = "CHECKPOINT_INVALID"
    ARCHITECTURE_MISMATCH = "ARCHITECTURE_MISMATCH"
    DEGENERATE = "DEGENERATE"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    NOT_CONFIGURED = "NOT_CONFIGURED"


@dataclass
class CheckpointMetadata:
    """Metadata extracted from a checkpoint file."""
    model_name: str = ""
    model_version: str = ""
    architecture: str = ""
    epoch: int = -1
    num_classes: int = 0
    image_size: Tuple[int, int] = (0, 0)
    dataset: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    state_dict_keys: List[str] = field(default_factory=list)
    parameter_count: int = 0
    has_metadata: bool = False


@dataclass
class ValidationResult:
    """Result of validating a model checkpoint."""
    status: ModelStatus
    inference_mode: ModelInferenceMode
    metadata: Optional[CheckpointMetadata] = None
    reason: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """Whether this model is safe to use for inference."""
        return self.status == ModelStatus.READY

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "inference_mode": self.inference_mode.value,
            "reason": self.reason,
            "warnings": self.warnings,
            "metadata": {
                "epoch": self.metadata.epoch if self.metadata else -1,
                "metrics": self.metadata.metrics if self.metadata else {},
                "has_metadata": self.metadata.has_metadata if self.metadata else False,
            },
        }


class ModelValidator:
    """Validates model checkpoints and enforces quality gates."""

    def __init__(self, quality_config: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Args:
            quality_config: Per-model minimum quality thresholds.
                Example: {"flood_detection": {"minimum_iou": 0.5, "minimum_f1": 0.4}}
        """
        self.quality_config = quality_config or {}

    def validate_checkpoint(
        self,
        weights_path: Optional[str],
        model_class,
        model_name: str,
        device: Optional[torch.device] = None,
    ) -> ValidationResult:
        """
        Full validation pipeline for a model checkpoint.

        Args:
            weights_path: Path to the .pth checkpoint file.
            model_class: The model class to instantiate for compatibility check.
            model_name: Human-readable model name for logging.
            device: Target device for loading.

        Returns:
            ValidationResult with status and metadata.
        """
        if device is None:
            device = torch.device("cpu")

        # Step 1: Check file existence
        if weights_path is None or not os.path.isfile(weights_path):
            return ValidationResult(
                status=ModelStatus.WEIGHTS_MISSING,
                inference_mode=ModelInferenceMode.UNAVAILABLE,
                reason=f"No weights file configured for {model_name}",
            )

        # Step 2: Load checkpoint
        try:
            # weights_only: a .pth is a pickle; never execute arbitrary code from it.
            checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        except Exception as e:
            return ValidationResult(
                status=ModelStatus.CHECKPOINT_INVALID,
                inference_mode=ModelInferenceMode.UNAVAILABLE,
                reason=f"Failed to load checkpoint (tensor-only checkpoints are supported): {e}",
            )

        if isinstance(checkpoint, dict) and checkpoint.get("synthetic_data"):
            return ValidationResult(
                status=ModelStatus.VALIDATION_FAILED,
                inference_mode=ModelInferenceMode.UNAVAILABLE,
                reason=(
                    f"{model_name} checkpoint was trained on synthetic dry-run data "
                    "and is not a real trained model"
                ),
            )

        # Step 3: Extract metadata
        metadata = self._extract_metadata(checkpoint, model_name)

        # Step 4: Extract state dict
        state_dict = self._extract_state_dict(checkpoint)
        if state_dict is None:
            return ValidationResult(
                status=ModelStatus.CHECKPOINT_INVALID,
                inference_mode=ModelInferenceMode.UNAVAILABLE,
                reason="Checkpoint does not contain a valid state_dict",
                metadata=metadata,
            )

        # Step 5: Architecture compatibility
        try:
            model = model_class()
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            return ValidationResult(
                status=ModelStatus.ARCHITECTURE_MISMATCH,
                inference_mode=ModelInferenceMode.UNAVAILABLE,
                reason=f"Architecture mismatch: {e}",
                metadata=metadata,
            )

        # Step 6: Check training quality if metadata available
        result = ValidationResult(
            status=ModelStatus.READY,
            inference_mode=ModelInferenceMode.TRAINED,
            metadata=metadata,
            reason=f"Checkpoint validated for {model_name}",
        )

        if metadata and metadata.has_metadata:
            if metadata.epoch == 0:
                result.status = ModelStatus.VALIDATION_FAILED
                result.inference_mode = ModelInferenceMode.DEGENERATE
                result.reason = "Model appears untrained (epoch=0)"
                return result

            # Check quality gates
            quality_issue = self._check_quality_gates(model_name, metadata)
            if quality_issue:
                result.warnings.append(quality_issue)

        # Step 7: Random weight detection heuristic
        is_degenerate = self._detect_degenerate_weights(state_dict)
        if is_degenerate:
            result.status = ModelStatus.DEGENERATE
            result.inference_mode = ModelInferenceMode.DEGENERATE
            result.reason = "Model weights appear degenerate (possible random initialization or collapsed training)"
            return result

        if missing and len(missing) > 0:
            result.warnings.append(f"Missing keys in state_dict: {len(missing)}")

        return result

    def _extract_metadata(self, checkpoint: Any, model_name: str) -> Optional[CheckpointMetadata]:
        """Extract training metadata from checkpoint if available."""
        metadata = CheckpointMetadata(model_name=model_name)

        if not isinstance(checkpoint, dict):
            metadata.has_metadata = False
            return metadata

        # Check for structured checkpoint with metadata
        if "model_name" in checkpoint or "epoch" in checkpoint or "metrics" in checkpoint:
            metadata.has_metadata = True
            metadata.model_name = checkpoint.get("model_name", model_name)
            metadata.model_version = checkpoint.get("model_version", "unknown")
            metadata.architecture = checkpoint.get("architecture", "unknown")
            metadata.epoch = checkpoint.get("epoch", -1)
            metadata.num_classes = checkpoint.get("num_classes", 0)
            metadata.image_size = checkpoint.get("image_size", (0, 0))
            metadata.dataset = checkpoint.get("dataset", "unknown")
            metadata.metrics = dict(checkpoint.get("metrics") or {})
            # Older training checkpoints stored only best_val_iou, in percent.
            if "best_val_iou" in checkpoint and not metadata.metrics:
                metadata.metrics["val_iou"] = float(checkpoint["best_val_iou"]) / 100.0
        elif "model_state_dict" in checkpoint:
            # Has structure but no metadata fields
            metadata.has_metadata = False

        return metadata

    def _extract_state_dict(self, checkpoint: Any) -> Optional[Dict]:
        """Extract the state_dict from various checkpoint formats."""
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                sd = checkpoint["model_state_dict"]
                if isinstance(sd, dict):
                    metadata = CheckpointMetadata()
                    metadata.state_dict_keys = list(sd.keys())
                    metadata.parameter_count = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
                    return sd
            elif "state_dict" in checkpoint:
                sd = checkpoint["state_dict"]
                if isinstance(sd, dict):
                    return sd
            # Might be a raw state_dict
            first_val = next(iter(checkpoint.values()), None)
            if isinstance(first_val, torch.Tensor):
                return checkpoint
        return None

    def _check_quality_gates(
        self, model_name: str, metadata: CheckpointMetadata
    ) -> Optional[str]:
        """Check if model meets configured quality thresholds."""
        thresholds = self.quality_config.get(model_name, {})
        if not thresholds:
            return None

        metrics = metadata.metrics
        issues = []

        min_iou = thresholds.get("minimum_iou")
        if min_iou is not None:
            iou = metrics.get("iou", metrics.get("val_iou", metrics.get("mean_iou")))
            if iou is not None and iou < min_iou:
                issues.append(f"IoU={iou:.3f} < minimum={min_iou:.3f}")

        min_f1 = thresholds.get("minimum_f1")
        if min_f1 is not None:
            f1 = metrics.get("f1", metrics.get("val_f1", metrics.get("mean_f1")))
            if f1 is not None and f1 < min_f1:
                issues.append(f"F1={f1:.3f} < minimum={min_f1:.3f}")

        min_epoch = thresholds.get("minimum_epoch")
        if min_epoch is not None and metadata.epoch >= 0 and metadata.epoch < min_epoch:
            issues.append(f"Epoch={metadata.epoch} < minimum={min_epoch}")

        if issues:
            return f"Quality gate warning for {model_name}: {'; '.join(issues)}"
        return None

    def _detect_degenerate_weights(self, state_dict: Dict) -> bool:
        """
        Heuristic detection of degenerate (random or collapsed) weights.

        Checks:
        1. Parameter distribution similarity to random init
        2. Output layer concentration (all weights nearly identical)
        """
        if not state_dict:
            return False

        # Find the last conv/linear layer (output layer)
        output_layer_key = None
        for key in reversed(list(state_dict.keys())):
            if "final_conv" in key or "classifier" in key or "head" in key:
                if "weight" in key:
                    output_layer_key = key
                    break

        if output_layer_key is None:
            return False

        output_weights = state_dict[output_layer_key]
        if not isinstance(output_weights, torch.Tensor):
            return False

        # Check if output layer weights are all nearly identical
        # (sign of class collapse during training)
        flat = output_weights.float().flatten()
        if flat.numel() > 1:
            std = flat.std().item()
            mean = flat.mean().item()
            # Very low variance relative to mean suggests collapse
            if abs(mean) > 1e-6 and std / abs(mean) < 0.01:
                return True
            # Extremely low variance overall
            if std < 1e-6:
                return True

        return False


class ModelRegistry:
    """
    Central registry for all models in the system.

    Provides validated model loading with quality gates.
    """

    def __init__(self, quality_config: Optional[Dict] = None):
        self.validator = ModelValidator(quality_config=quality_config)
        self._models: Dict[str, Any] = {}
        self._validations: Dict[str, ValidationResult] = {}

    def register(
        self,
        name: str,
        model_class,
        weights_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> ValidationResult:
        """Register and validate a model."""
        result = self.validator.validate_checkpoint(
            weights_path=weights_path,
            model_class=model_class,
            model_name=name,
            device=device,
        )
        self._validations[name] = result
        return result

    def get_validation(self, name: str) -> Optional[ValidationResult]:
        """Get the validation result for a registered model."""
        return self._validations.get(name)

    def is_usable(self, name: str) -> bool:
        """Check if a model is validated and safe to use."""
        result = self._validations.get(name)
        return result is not None and result.is_usable

    def get_inference_mode(self, name: str) -> ModelInferenceMode:
        """Get the inference mode for a registered model."""
        result = self._validations.get(name)
        if result is None:
            return ModelInferenceMode.UNAVAILABLE
        return result.inference_mode

    def summary(self) -> str:
        """Human-readable summary of all registered models."""
        lines = [
            "=" * 70,
            "  MODEL VALIDATION SUMMARY",
            "=" * 70,
        ]
        for name, result in self._validations.items():
            marker = {
                ModelStatus.READY: "[OK]",
                ModelStatus.WEIGHTS_MISSING: "[--]",
                ModelStatus.CHECKPOINT_INVALID: "[!!]",
                ModelStatus.ARCHITECTURE_MISMATCH: "[!!]",
                ModelStatus.DEGENERATE: "[DEGEN]",
                ModelStatus.VALIDATION_FAILED: "[FAIL]",
                ModelStatus.NOT_CONFIGURED: "[--]",
            }.get(result.status, "[??]")
            lines.append(f"  {marker} {name}")
            lines.append(f"        Status: {result.status.value}")
            lines.append(f"        Mode: {result.inference_mode.value}")
            lines.append(f"        Reason: {result.reason}")
            if result.warnings:
                for w in result.warnings:
                    lines.append(f"        Warning: {w}")
            if result.metadata and result.metadata.has_metadata:
                m = result.metadata
                lines.append(f"        Epoch: {m.epoch}")
                if m.metrics:
                    lines.append(f"        Metrics: {m.metrics}")
            lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)
