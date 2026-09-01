"""
Tests for SAR-Optical fusion models.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.models.fusion.sar_optical_fusion import SAROpticalFusionUNet
from disaster_assessment.models.fusion.feature_fusion import EarlyFusion, FeatureLevelFusion


class TestSAROpticalFusionUNet:
    def test_feature_fusion_forward(self):
        model = SAROpticalFusionUNet(
            optical_channels=3,
            sar_channels=2,
            fusion_type="feature",
        )
        optical = torch.randn(1, 3, 128, 128)
        sar = torch.randn(1, 2, 128, 128)
        output = model(optical, sar)
        assert output.shape == (1, 1, 128, 128)

    def test_early_fusion_forward(self):
        model = SAROpticalFusionUNet(
            optical_channels=3,
            sar_channels=2,
            fusion_type="early",
        )
        optical = torch.randn(1, 3, 128, 128)
        sar = torch.randn(1, 2, 128, 128)
        output = model(optical, sar)
        assert output.shape == (1, 1, 128, 128)

    def test_multiclass(self):
        model = SAROpticalFusionUNet(
            optical_channels=3,
            sar_channels=2,
            num_classes=4,
            fusion_type="feature",
        )
        optical = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 2, 64, 64)
        output = model(optical, sar)
        assert output.shape == (1, 4, 64, 64)

    def test_predict(self):
        model = SAROpticalFusionUNet(
            optical_channels=3,
            sar_channels=2,
            fusion_type="feature",
        )
        optical = torch.randn(1, 3, 64, 64)
        sar = torch.randn(1, 2, 64, 64)
        mask = model.predict(optical, sar)
        assert mask.shape == (1, 1, 64, 64)
        unique = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique.tolist())


class TestFeatureFusion:
    def test_early_fusion(self):
        fusion = EarlyFusion()
        tensors = [torch.randn(1, 3, 64, 64), torch.randn(1, 2, 64, 64)]
        result = fusion(tensors)
        assert result.shape == (1, 5, 64, 64)

    def test_feature_level_concat(self):
        fusion = FeatureLevelFusion(in_channels=[64, 64], fusion_method="concat")
        t1 = torch.randn(1, 64, 32, 32)
        t2 = torch.randn(1, 64, 32, 32)
        result = fusion([t1, t2])
        assert result.shape == (1, 128, 32, 32)

    def test_feature_level_add(self):
        fusion = FeatureLevelFusion(in_channels=[64], fusion_method="add")
        t1 = torch.randn(1, 64, 32, 32)
        t2 = torch.randn(1, 64, 32, 32)
        result = fusion([t1, t2])
        assert result.shape == (1, 64, 32, 32)
