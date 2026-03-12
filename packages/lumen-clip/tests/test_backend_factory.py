"""Unit tests for backend factory runtime normalization."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, cast

import pytest
from lumen_resources.lumen_config import BackendSettings

from lumen_clip.backends import factory


class _DummyONNXRTBackend:
    """Lightweight stand-in for ONNXRTBackend used in factory tests."""

    def __init__(
        self,
        resources,
        providers,
        device_preference,
        max_batch_size,
        precision,
    ) -> None:
        self.resources = resources
        self.providers = providers
        self.device_preference = device_preference
        self.max_batch_size = max_batch_size
        self.precision = precision


def test_create_backend_accepts_onnx_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should accept `onnx` as the public runtime value."""
    dummy_module = cast(Any, ModuleType("lumen_clip.backends.onnxrt_backend"))
    dummy_module.ONNXRTBackend = _DummyONNXRTBackend

    monkeypatch.setattr(factory, "_BACKEND_REGISTRY", {}, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "lumen_clip.backends.onnxrt_backend",
        dummy_module,
    )

    backend_settings = BackendSettings(
        device="cpu",
        batch_size=2,
        onnx_providers=cast(Any, ["CPUExecutionProvider"]),
    )
    resources = object()

    backend = factory.create_backend(
        backend_config=backend_settings,
        resources=resources,
        runtime="onnx",
        precision="fp16",
    )

    assert isinstance(backend, _DummyONNXRTBackend)
    assert backend.resources is resources
    assert backend.providers == ["CPUExecutionProvider"]
    assert backend.device_preference == "cpu"
    assert backend.max_batch_size == 2
    assert backend.precision == "fp16"


def test_create_backend_rejects_onnxrt_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should expose only `onnx` as the public runtime value."""
    monkeypatch.setattr(
        factory,
        "_BACKEND_REGISTRY",
        {factory.RuntimeKind.ONNX: _DummyONNXRTBackend},
        raising=False,
    )
    monkeypatch.setattr(factory, "get_available_backends", lambda: ["onnx"])

    backend_settings = BackendSettings(
        device="cpu",
        batch_size=2,
        onnx_providers=cast(Any, ["CPUExecutionProvider"]),
    )

    with pytest.raises(ValueError, match="Runtime 'onnxrt' is not available"):
        factory.create_backend(
            backend_config=backend_settings,
            resources=object(),
            runtime="onnxrt",
        )
