"""
ONNX model export and validation.

Status: DEPENDENCY_MISSING - requires onnx package.
"""

import os
from typing import Tuple

import numpy as np


def export_to_onnx(
    model,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    opset_version: int = 17,
) -> str:
    """Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch nn.Module.
        output_path: Path for the .onnx file.
        input_shape: Input tensor shape (B, C, H, W).
        opset_version: ONNX opset version.

    Returns:
        Path to exported ONNX file.

    Raises:
        ImportError: If onnx or torch.onnx is not available.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for ONNX export")

    try:
        import onnx
    except ImportError:
        raise ImportError("onnx package is required. Install with: pip install onnx")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    return output_path


def validate_onnx_output(
    onnx_path: str,
    pytorch_model,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> dict:
    """Validate ONNX output against PyTorch output.

    Returns:
        dict with 'passed', 'max_diff', 'mean_diff'.
    """
    result = {"passed": False, "max_diff": 0.0, "mean_diff": 0.0, "error": None}

    try:
        import torch
        import onnxruntime as ort
    except ImportError as e:
        result["error"] = str(e)
        return result

    try:
        dummy = torch.randn(*input_shape)

        # PyTorch output
        pytorch_model.eval()
        with torch.no_grad():
            pt_output = pytorch_model(dummy).numpy()

        # ONNX output
        session = ort.InferenceSession(onnx_path)
        onnx_output = session.run(None, {"input": dummy.numpy()})[0]

        diff = np.abs(pt_output - onnx_output)
        result["max_diff"] = float(np.max(diff))
        result["mean_diff"] = float(np.mean(diff))
        result["passed"] = np.allclose(pt_output, onnx_output, rtol=rtol, atol=atol)
    except Exception as e:
        result["error"] = str(e)

    return result
