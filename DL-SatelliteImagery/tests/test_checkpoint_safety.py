"""
Checkpoint validation: dry-run models must not be reported as trained, and
checkpoint files must never be unpickled as arbitrary Python objects.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.inference.building_damage_inference import BuildingDamageInference
from disaster_assessment.models.building_damage_model import BuildingDamageUNet
from disaster_assessment.models.validation import ModelStatus, ModelValidator


def _save_checkpoint(path, **extra):
    model = BuildingDamageUNet()
    torch.save({"epoch": 3, "model_state_dict": model.state_dict(), **extra}, path)
    return str(path)


def test_synthetic_dry_run_checkpoint_is_refused(tmp_path):
    path = _save_checkpoint(tmp_path / "dry.pth", best_val_iou=23.9, synthetic_data=True)

    inference = BuildingDamageInference(weights_path=path, device=torch.device("cpu"))
    status = inference.validate()

    assert status["status"] == ModelStatus.VALIDATION_FAILED.value
    assert "synthetic" in status["reason"]


def test_legacy_best_val_iou_is_exposed_as_fraction(tmp_path):
    path = _save_checkpoint(tmp_path / "legacy.pth", best_val_iou=42.0)

    result = ModelValidator().validate_checkpoint(path, BuildingDamageUNet, "building_damage")

    assert result.metadata.metrics["val_iou"] == 0.42


class _Exploit:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        # On unpickle this would create the marker file.
        return (open, (self.marker, "w"))


def test_pickled_objects_are_not_executed(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "evil.pth"
    torch.save({"epoch": 3, "payload": _Exploit(str(marker))}, path)

    result = ModelValidator().validate_checkpoint(str(path), BuildingDamageUNet, "building_damage")

    assert result.status == ModelStatus.CHECKPOINT_INVALID
    assert not marker.exists()
