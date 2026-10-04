"""
Tests for Phase 2: Real Trained Building Damage Assessment.
"""

import os
import shutil
import sys
import tempfile
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.datasets.building_damage_dataset import (
    BuildingDamageDataset,
    create_damage_dataloaders,
    create_synthetic_damage_data,
)
from disaster_assessment.inference.building_damage_inference import (
    BuildingDamageInference,
)
from disaster_assessment.models.base_unet import count_parameters
from disaster_assessment.models.building_damage_model import (
    BuildingDamageUNet,
    create_building_damage_model,
)
from disaster_assessment.training.train_building_damage import (
    CombinedDamageLoss,
    compute_metrics,
    train_building_damage,
)
from disaster_assessment.visualization.overlays import (
    OverlayRenderer,
    TRAINED_DAMAGE_COLORS,
    TRAINED_DAMAGE_LABELS,
)


class TestBuildingDamageModel:
    def test_forward_pass_shape(self):
        model = BuildingDamageUNet(in_channels=3, num_classes=4)
        pre = torch.randn(2, 3, 256, 256)
        post = torch.randn(2, 3, 256, 256)
        logits = model(pre, post)

        assert logits.shape == (2, 4, 256, 256)

    def test_predict_classes(self):
        model = BuildingDamageUNet(in_channels=3, num_classes=4)
        pre = torch.randn(1, 3, 256, 256)
        post = torch.randn(1, 3, 256, 256)

        class_map, probs = model.predict(pre, post)
        assert class_map.shape == (1, 256, 256)
        assert probs.shape == (1, 4, 256, 256)
        unique_classes = torch.unique(class_map).tolist()
        assert all(c in [0, 1, 2, 3] for c in unique_classes)

    def test_parameter_count(self):
        model = BuildingDamageUNet()
        params = count_parameters(model)
        assert 1_000_000 < params < 6_000_000  # ~3.8M parameters

    def test_factory_function(self):
        model = create_building_damage_model(device=torch.device("cpu"))
        assert isinstance(model, BuildingDamageUNet)


class TestBuildingDamageDataset:
    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.temp_dir = tempfile.mkdtemp()
        create_synthetic_damage_data(self.temp_dir, num_samples=4, size=(128, 128))
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dataset_loading(self):
        train_dir = os.path.join(self.temp_dir, "train")
        ds = BuildingDamageDataset(train_dir, target_size=(128, 128), is_train=True)
        assert len(ds) == 4

        item = ds[0]
        assert "pre_image" in item
        assert "post_image" in item
        assert "mask" in item
        assert item["pre_image"].shape == (3, 128, 128)
        assert item["mask"].shape == (128, 128)

    def test_dataloaders(self):
        train_loader, val_loader = create_damage_dataloaders(
            self.temp_dir, batch_size=2, target_size=(128, 128)
        )
        assert train_loader is not None
        batch = next(iter(train_loader))
        assert batch["pre_image"].shape == (2, 3, 128, 128)


class TestDamageTrainingPipeline:
    @pytest.fixture(autouse=True)
    def setup_env(self):
        self.temp_dir = tempfile.mkdtemp()
        self.weights_dir = os.path.join(self.temp_dir, "weights")
        create_synthetic_damage_data(self.temp_dir, num_samples=4, size=(128, 128))
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_loss_computation(self):
        criterion = CombinedDamageLoss(num_classes=4)
        logits = torch.randn(2, 4, 64, 64)
        targets = torch.randint(0, 4, (2, 64, 64), dtype=torch.long)
        loss = criterion(logits, targets)
        assert loss.item() > 0

    def test_compute_metrics(self):
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])
        metrics = compute_metrics(preds, targets, num_classes=4)
        assert metrics["accuracy"] == 100.0
        assert metrics["mean_iou"] == 100.0

    def test_train_dry_run(self):
        best_path = train_building_damage(
            data_dir=self.temp_dir,
            output_dir=self.weights_dir,
            epochs=1,
            batch_size=2,
            lr=1e-3,
            device=torch.device("cpu"),
        )
        assert os.path.isfile(best_path)


class TestBuildingDamageInference:
    @pytest.fixture(autouse=True)
    def setup_weights(self):
        self.temp_dir = tempfile.mkdtemp()
        self.weights_path = os.path.join(self.temp_dir, "test_weights.pth")
        model = BuildingDamageUNet()
        torch.save({"model_state_dict": model.state_dict()}, self.weights_path)
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_auto_fallback_without_weights(self):
        engine = BuildingDamageInference(weights_path="non_existent.pth")
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        out = engine.assess(pre, post, mode="auto")
        assert out.damage_mode == "estimated"
        assert not out.model_available
        assert out.warning is not None

    def test_trained_mode_with_weights(self):
        engine = BuildingDamageInference(weights_path=self.weights_path)
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        out = engine.assess(pre, post, mode="trained")
        assert out.damage_mode == "trained"
        assert out.model_available
        assert 0 <= out.undamaged_percentage <= 100
        assert 0 <= out.destroyed_percentage <= 100
        assert out.damage_map.shape == (256, 256)

    def test_explicit_estimated_mode(self):
        engine = BuildingDamageInference(weights_path=self.weights_path)
        pre = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        post = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        out = engine.assess(pre, post, mode="estimated")
        assert out.damage_mode == "estimated"


class TestTrainedDamageVisualization:
    def test_trained_overlay(self):
        renderer = OverlayRenderer()
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        dmg_map = np.random.randint(0, 4, (256, 256), dtype=np.uint8)

        overlay = renderer.trained_damage_overlay(img, dmg_map)
        assert overlay.shape == (256, 256, 3)

    def test_trained_legend(self):
        legend = OverlayRenderer.create_legend(TRAINED_DAMAGE_LABELS, TRAINED_DAMAGE_COLORS)
        assert legend.shape[2] == 3
