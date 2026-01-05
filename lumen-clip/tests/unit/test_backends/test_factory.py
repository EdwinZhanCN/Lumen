"""
Unit tests for backend factory precision parameter passing.

Tests that the create_backend function correctly passes the precision
parameter to backend instances.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock onnxruntime before importing
with patch.dict("sys.modules", {"onnxruntime": MagicMock()}):
    from lumen_clip.backends import RuntimeKind, create_backend
    from lumen_clip.backends.onnxrt_backend import ONNXRTBackend


class TestCreateBackendPrecision:
    """Test precision parameter passing through create_backend factory."""

    def test_factory_accepts_precision_string(self, tmp_path: Path) -> None:
        """Test that create_backend accepts precision string parameter."""
        # Create mock model directory structure
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)
        (model_dir / "vision.onnx").touch()
        (model_dir / "text.onnx").touch()

        resources = MagicMock()
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        backend_settings = MagicMock()
        backend_settings.device = "cpu"
        backend_settings.batch_size = 8
        backend_settings.onnx_providers = None

        # Should not raise any type errors
        backend = create_backend(
            backend_settings,
            resources,
            RuntimeKind.ONNXRT,
            precision="fp16",
        )

        assert isinstance(backend, ONNXRTBackend)

    def test_factory_passes_precision_to_onnxrt_backend(self, tmp_path: Path) -> None:
        """Test that precision parameter is passed to ONNXRTBackend constructor."""
        # Create mock model directory structure
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)
        (model_dir / "vision.onnx").touch()
        (model_dir / "text.onnx").touch()

        resources = MagicMock()
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        backend_settings = MagicMock()
        backend_settings.device = "cpu"
        backend_settings.batch_size = 8
        backend_settings.onnx_providers = None

        backend = create_backend(
            backend_settings,
            resources,
            RuntimeKind.ONNXRT,
            precision="fp16",
        )

        assert backend._precision == "fp16"

    def test_factory_passes_none_precision_to_onnxrt_backend(
        self, tmp_path: Path
    ) -> None:
        """Test that precision=None is passed to ONNXRTBackend constructor."""
        # Create mock model directory structure
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)
        (model_dir / "vision.onnx").touch()
        (model_dir / "text.onnx").touch()

        resources = MagicMock()
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        backend_settings = MagicMock()
        backend_settings.device = "cpu"
        backend_settings.batch_size = 8
        backend_settings.onnx_providers = None

        backend = create_backend(
            backend_settings,
            resources,
            RuntimeKind.ONNXRT,
            precision=None,
        )

        assert backend._precision is None

    @pytest.mark.parametrize(
        "precision",
        ["fp32", "fp16", "int8", "q4fp16", None],
    )
    def test_factory_various_precision_values(
        self, precision: str | None, tmp_path: Path
    ) -> None:
        """Test that various precision values are correctly passed to backend."""
        # Create mock model directory structure
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)
        (model_dir / "vision.onnx").touch()
        (model_dir / "text.onnx").touch()

        resources = MagicMock()
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        backend_settings = MagicMock()
        backend_settings.device = "cpu"
        backend_settings.batch_size = 8
        backend_settings.onnx_providers = None

        backend = create_backend(
            backend_settings,
            resources,
            RuntimeKind.ONNXRT,
            precision=precision,
        )

        assert backend._precision == precision

    def test_factory_does_not_have_prefer_fp16_parameter(self) -> None:
        """Test that create_backend no longer has prefer_fp16 parameter."""
        import inspect

        from lumen_clip.backends import factory

        sig = inspect.signature(factory.create_backend)
        params = list(sig.parameters.keys())

        # Should have precision parameter
        assert "precision" in params

        # Should NOT have prefer_fp16 parameter
        assert "prefer_fp16" not in params
