"""Unit tests for FastVLM backend factory behavior."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from lumen_vlm.backends import factory


class _DummyFastVLMONNXBackend:
    """Minimal stand-in for the ONNX backend constructor."""

    def __init__(
        self,
        resources,
        *,
        providers,
        device_preference,
        max_new_tokens,
        prefer_fp16,
    ) -> None:
        self.resources = resources
        self.providers = providers
        self.device_preference = device_preference
        self.max_new_tokens = max_new_tokens
        self.prefer_fp16 = prefer_fp16


def test_create_backend_uses_fixed_max_new_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should not source generation limits from backend settings."""
    dummy_module = cast(Any, ModuleType("lumen_vlm.backends.onnxrt_backend"))
    dummy_module.FastVLMONNXBackend = _DummyFastVLMONNXBackend

    monkeypatch.setattr(factory, "_BACKEND_REGISTRY", {}, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "lumen_vlm.backends.onnxrt_backend",
        dummy_module,
    )

    backend_settings = cast(
        Any,
        SimpleNamespace(
            device="cuda",
            onnx_providers=["CUDAExecutionProvider"],
            max_new_tokens=2048,
        ),
    )

    backend = factory.create_backend(
        backend_config=backend_settings,
        resources=object(),
        runtime="onnx",
        prefer_fp16=True,
    )

    assert isinstance(backend, _DummyFastVLMONNXBackend)
    assert backend.device_preference == "cuda"
    assert backend.providers == ["CUDAExecutionProvider"]
    assert backend.max_new_tokens == factory.DEFAULT_MAX_NEW_TOKENS
    assert backend.prefer_fp16 is True


def test_create_backend_rejects_unknown_runtime() -> None:
    """Factory should expose only supported public runtime values."""
    with pytest.raises(ValueError, match="Runtime 'torch' is not available"):
        factory.create_backend(
            backend_config=None,
            resources=object(),
            runtime="torch",
        )
