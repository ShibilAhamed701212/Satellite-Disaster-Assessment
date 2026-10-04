"""
Tests for deployment tools (benchmarking, ONNX export).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.deployment.benchmark import benchmark_model, BenchmarkResult
from disaster_assessment.models.flood_unet import FloodUNet


class TestBenchmark:
    def test_benchmark_flood_unet(self):
        model = FloodUNet()
        result = benchmark_model(
            model,
            input_shape=(1, 3, 128, 128),
            device="cpu",
            num_iterations=5,
            warmup_iterations=2,
            model_name="FloodUNet",
        )
        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "FloodUNet"
        assert result.mean_latency_ms > 0
        assert result.parameters > 0
        assert result.throughput_fps > 0

    def test_benchmark_result_to_dict(self):
        model = FloodUNet()
        result = benchmark_model(
            model,
            input_shape=(1, 3, 64, 64),
            device="cpu",
            num_iterations=3,
            warmup_iterations=1,
        )
        d = result.to_dict()
        assert "mean_latency_ms" in d
        assert "parameters" in d
        assert "throughput_fps" in d

    def test_benchmark_result_summary(self):
        result = BenchmarkResult(
            model_name="test",
            mean_latency_ms=10.0,
            std_latency_ms=2.0,
            parameters=100000,
            throughput_fps=100.0,
        )
        s = result.summary()
        assert "test" in s
        assert "100,000" in s
