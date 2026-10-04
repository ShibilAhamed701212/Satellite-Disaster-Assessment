"""
Tests for disaster assessment models.

Uses synthetic 256×256 images — no large dataset downloads required.
"""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.models.base_unet import (
    ConvBlock,
    LightweightEncoder,
    LightweightDecoder,
    get_device,
    count_parameters,
)
from disaster_assessment.models.siamese_unet import SiameseUNet, create_siamese_unet
from disaster_assessment.models.flood_unet import FloodUNet, create_flood_unet


class TestConvBlock:
    def test_output_shape(self):
        block = ConvBlock(3, 16)
        x = torch.randn(1, 3, 64, 64)
        out = block(x)
        assert out.shape == (1, 16, 64, 64)

    def test_different_channels(self):
        block = ConvBlock(32, 64)
        x = torch.randn(2, 32, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)


class TestLightweightEncoder:
    def test_output_shapes(self):
        encoder = LightweightEncoder(in_channels=3)
        x = torch.randn(1, 3, 256, 256)
        bottleneck, skips = encoder(x)

        assert bottleneck.shape == (1, 256, 16, 16)  # 256/16 = 16
        assert len(skips) == 4
        assert skips[0].shape == (1, 16, 256, 256)   # enc1
        assert skips[1].shape == (1, 32, 128, 128)   # enc2
        assert skips[2].shape == (1, 64, 64, 64)     # enc3
        assert skips[3].shape == (1, 128, 32, 32)    # enc4

    def test_parameter_count(self):
        encoder = LightweightEncoder()
        params = count_parameters(encoder)
        assert params > 0
        assert params < 2_000_000  # 1.18M parameters


class TestLightweightDecoder:
    def test_output_shape(self):
        encoder = LightweightEncoder()
        decoder = LightweightDecoder(out_channels=1)

        x = torch.randn(1, 3, 256, 256)
        bn, skips = encoder(x)
        out = decoder(bn, skips)

        assert out.shape == (1, 1, 256, 256)


class TestSiameseUNet:
    def test_forward_pass(self):
        model = SiameseUNet()
        pre = torch.randn(1, 3, 256, 256)
        post = torch.randn(1, 3, 256, 256)
        out = model(pre, post)

        assert out.shape == (1, 1, 256, 256)
        assert out.min() >= 0.0
        assert out.max() <= 1.0  # Sigmoid output

    def test_predict(self):
        model = SiameseUNet()
        pre = torch.randn(1, 3, 256, 256)
        post = torch.randn(1, 3, 256, 256)
        mask = model.predict(pre, post, threshold=0.5)

        assert mask.shape == (1, 1, 256, 256)
        unique_values = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_values.tolist())

    def test_batch_size(self):
        model = SiameseUNet()
        pre = torch.randn(4, 3, 256, 256)
        post = torch.randn(4, 3, 256, 256)
        out = model(pre, post)
        assert out.shape == (4, 1, 256, 256)

    def test_parameter_count(self):
        model = SiameseUNet()
        params = count_parameters(model)
        assert params > 1_000_000
        assert params < 10_000_000  # Should be 3-4M

    def test_factory_function_cpu(self):
        model = create_siamese_unet(device=torch.device("cpu"))
        assert next(model.parameters()).device.type == "cpu"


class TestFloodUNet:
    def test_forward_pass(self):
        model = FloodUNet()
        x = torch.randn(1, 3, 256, 256)
        out = model(x)

        assert out.shape == (1, 1, 256, 256)
        assert out.min() >= 0.0
        assert out.max() <= 1.0  # Sigmoid output

    def test_predict(self):
        model = FloodUNet()
        x = torch.randn(1, 3, 256, 256)
        mask = model.predict(x, threshold=0.5)

        assert mask.shape == (1, 1, 256, 256)
        unique_values = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_values.tolist())

    def test_batch_size(self):
        model = FloodUNet()
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_parameter_count(self):
        model = FloodUNet()
        params = count_parameters(model)
        assert params > 500_000
        assert params < 5_000_000  # Should be ~1.9M

    def test_factory_function_cpu(self):
        model = create_flood_unet(device=torch.device("cpu"))
        assert next(model.parameters()).device.type == "cpu"


class TestCUDAInference:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_siamese_cuda(self):
        model = create_siamese_unet(device=torch.device("cuda"))
        pre = torch.randn(1, 3, 256, 256, device="cuda")
        post = torch.randn(1, 3, 256, 256, device="cuda")
        out = model(pre, post)
        assert out.device.type == "cuda"
        assert out.shape == (1, 1, 256, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flood_cuda(self):
        model = create_flood_unet(device=torch.device("cuda"))
        x = torch.randn(1, 3, 256, 256, device="cuda")
        out = model(x)
        assert out.device.type == "cuda"
        assert out.shape == (1, 1, 256, 256)


class TestDeviceDetection:
    def test_get_device(self):
        device = get_device()
        assert device.type in ["cpu", "cuda"]
