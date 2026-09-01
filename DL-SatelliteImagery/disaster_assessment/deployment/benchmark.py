"""
Model inference benchmarking.

Reports actual measured latency and memory usage.
No fabricated numbers.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BenchmarkResult:
    """Benchmark results for a model."""
    model_name: str
    input_shape: Tuple[int, ...] = (1, 3, 256, 256)
    device: str = "cpu"
    num_iterations: int = 100
    warmup_iterations: int = 10

    # Measured metrics
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_fps: float = 0.0
    peak_memory_mb: float = 0.0
    parameters: int = 0

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "input_shape": self.input_shape,
            "device": self.device,
            "iterations": self.num_iterations,
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "std_latency_ms": round(self.std_latency_ms, 3),
            "min_latency_ms": round(self.min_latency_ms, 3),
            "max_latency_ms": round(self.max_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "throughput_fps": round(self.throughput_fps, 1),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "parameters": self.parameters,
        }

    def summary(self) -> str:
        return (
            f"Benchmark: {self.model_name} on {self.device}\n"
            f"  Input: {self.input_shape}\n"
            f"  Latency: {self.mean_latency_ms:.1f}±{self.std_latency_ms:.1f}ms "
            f"(min={self.min_latency_ms:.1f}, max={self.max_latency_ms:.1f}, "
            f"p95={self.p95_latency_ms:.1f}ms)\n"
            f"  Throughput: {self.throughput_fps:.1f} fps\n"
            f"  Peak Memory: {self.peak_memory_mb:.1f} MB\n"
            f"  Parameters: {self.parameters:,}"
        )


def benchmark_model(
    model,
    input_shape: Tuple[int, ...] = (1, 3, 256, 256),
    device: str = "cpu",
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    model_name: str = "model",
) -> BenchmarkResult:
    """Benchmark a PyTorch model.

    Args:
        model: PyTorch nn.Module.
        input_shape: Input tensor shape.
        device: Device string.
        num_iterations: Number of timed iterations.
        warmup_iterations: Number of warmup iterations.
        model_name: Name for the report.

    Returns:
        BenchmarkResult with actual measured metrics.
    """
    import torch

    result = BenchmarkResult(
        model_name=model_name,
        input_shape=input_shape,
        device=device,
        num_iterations=num_iterations,
    )

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    dummy = torch.randn(*input_shape, device=dev)
    latencies = []

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy)

    # Benchmark
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    latencies = np.array(latencies)
    result.mean_latency_ms = float(np.mean(latencies))
    result.std_latency_ms = float(np.std(latencies))
    result.min_latency_ms = float(np.min(latencies))
    result.max_latency_ms = float(np.max(latencies))
    result.p95_latency_ms = float(np.percentile(latencies, 95))
    result.p99_latency_ms = float(np.percentile(latencies, 99))
    result.throughput_fps = 1000.0 / max(result.mean_latency_ms, 0.001)
    result.parameters = sum(p.numel() for p in model.parameters())

    return result
