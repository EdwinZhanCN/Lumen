"""Validate a single ONNX model with ONNX Runtime."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def _normalize_shape(raw_shape: list[Any]) -> list[int]:
    """Normalize symbolic ONNX dimensions into concrete dummy dimensions."""
    normalized: list[int] = []
    for dim in raw_shape:
        if isinstance(dim, int) and dim > 0:
            normalized.append(dim)
        else:
            normalized.append(1)
    return normalized


def _dummy_input(type_name: str, shape: list[int]) -> np.ndarray:
    """Build a deterministic dummy array for one ONNX input."""
    if "float16" in type_name:
        return np.zeros(shape, dtype=np.float16)
    if "float" in type_name or "double" in type_name:
        return np.zeros(shape, dtype=np.float32)
    if "int64" in type_name:
        return np.zeros(shape, dtype=np.int64)
    if "int32" in type_name:
        return np.zeros(shape, dtype=np.int32)
    if "bool" in type_name:
        return np.zeros(shape, dtype=np.bool_)
    raise ValueError(f"Unsupported ONNX input type for dummy execution: {type_name}")


def validate_onnx_model(
    *,
    model_path: Path,
    task: str,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Load and execute one ONNX model with deterministic dummy inputs."""
    requested_providers = providers or ["CPUExecutionProvider"]
    available_providers_fn = getattr(ort, "get_available_providers", None)
    available_providers = (
        list(available_providers_fn()) if callable(available_providers_fn) else []
    )
    chosen_providers = [
        provider for provider in requested_providers if provider in available_providers
    ]
    if not chosen_providers:
        chosen_providers = ["CPUExecutionProvider"]

    result: dict[str, Any] = {
        "task": task,
        "model": str(model_path),
        "requested_providers": requested_providers,
        "available_providers": available_providers,
        "providers": chosen_providers,
    }

    try:
        session = ort.InferenceSession(str(model_path), providers=chosen_providers)
        input_specs: list[dict[str, Any]] = []
        feed_dict: dict[str, np.ndarray] = {}
        for input_arg in session.get_inputs():
            shape = _normalize_shape(input_arg.shape)
            input_specs.append(
                {
                    "name": input_arg.name,
                    "type": input_arg.type,
                    "shape": shape,
                }
            )
            feed_dict[input_arg.name] = _dummy_input(input_arg.type, shape)

        start = time.perf_counter()
        outputs = session.run(None, feed_dict)
        latency_ms = (time.perf_counter() - start) * 1000.0

        output_specs: list[dict[str, Any]] = []
        for output_arg, value in zip(session.get_outputs(), outputs, strict=True):
            output_specs.append(
                {
                    "name": output_arg.name,
                    "type": output_arg.type,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )

        result.update(
            {
                "success": True,
                "input_specs": input_specs,
                "output_specs": output_specs,
                "latency_ms": latency_ms,
            }
        )
    except Exception as exc:
        result.update({"success": False, "error": str(exc)})

    return result
