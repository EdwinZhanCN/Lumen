"""
Unit tests for ONNXRTBackend precision-based model file selection.

Tests that the backend correctly loads ONNX model files based on the
configured precision parameter.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock onnxruntime before importing ONNXRTBackend
with patch.dict("sys.modules", {"onnxruntime": MagicMock()}):
    from lumen_clip.backends.onnxrt_backend import ONNXRTBackend
    from lumen_clip.resources import ModelResources


class TestONNXRTBackendPrecision:
    """Test precision-based file selection in ONNXRTBackend."""

    def test_precision_fp16_selects_fp16_files(self, tmp_path: Path) -> None:
        """Test that precision='fp16' selects vision.fp16.onnx and text.fp16.onnx files."""
        # Create mock model directory structure
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)

        # Create fp16 model files
        (model_dir / "vision.fp16.onnx").touch()
        (model_dir / "text.fp16.onnx").touch()

        # Create mock resources
        resources = MagicMock(spec=ModelResources)
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        # Create backend with fp16 precision
        backend = ONNXRTBackend(resources=resources, precision="fp16")

        # Test _select_model_file
        vision_path, vision_prec = backend._select_model_file("vision")
        text_path, text_prec = backend._select_model_file("text")

        assert vision_prec == "fp16"
        assert text_prec == "fp16"
        assert vision_path.name == "vision.fp16.onnx"
        assert text_path.name == "text.fp16.onnx"

    def test_precision_fp32_selects_default_files(self, tmp_path: Path) -> None:
        """Test that precision='fp32' selects vision.onnx and text.onnx (default) files."""
        # Create mock model directory structure
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)

        # Create default model files (fp32)
        (model_dir / "vision.onnx").touch()
        (model_dir / "text.onnx").touch()

        # Create mock resources
        resources = MagicMock(spec=ModelResources)
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        # Create backend with fp32 precision
        backend = ONNXRTBackend(resources=resources, precision="fp32")

        # Test _select_model_file
        vision_path, vision_prec = backend._select_model_file("vision")
        text_path, text_prec = backend._select_model_file("text")

        assert vision_prec == "fp32"
        assert text_prec == "fp32"
        assert vision_path.name == "vision.onnx"
        assert text_path.name == "text.onnx"

    def test_precision_none_selects_default_files(self, tmp_path: Path) -> None:
        """Test that precision=None selects default vision.onnx and text.onnx files."""
        # Create mock model directory structure
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)

        # Create default model files
        (model_dir / "vision.onnx").touch()
        (model_dir / "text.onnx").touch()

        # Create mock resources
        resources = MagicMock(spec=ModelResources)
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        # Create backend with no precision (default)
        backend = ONNXRTBackend(resources=resources, precision=None)

        # Test _select_model_file
        vision_path, vision_prec = backend._select_model_file("vision")
        text_path, text_prec = backend._select_model_file("text")

        assert vision_prec == "fp32"
        assert text_prec == "fp32"
        assert vision_path.name == "vision.onnx"
        assert text_path.name == "text.onnx"

    def test_precision_fallback_when_not_found(self, tmp_path: Path) -> None:
        """Test that configured precision files not found falls back to default files."""
        # Create mock model directory structure
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)

        # Only create default (fp32) files, not int8
        (model_dir / "vision.onnx").touch()
        (model_dir / "text.onnx").touch()

        # Create mock resources
        resources = MagicMock(spec=ModelResources)
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        # Create backend with int8 precision (files don't exist)
        backend = ONNXRTBackend(resources=resources, precision="int8")

        # Test _select_model_file - should fall back to default
        vision_path, vision_prec = backend._select_model_file("vision")
        text_path, text_prec = backend._select_model_file("text")

        assert vision_prec == "fp32"
        assert text_prec == "fp32"
        assert vision_path.name == "vision.onnx"
        assert text_path.name == "text.onnx"

    def test_precision_int8_selects_int8_files(self, tmp_path: Path) -> None:
        """Test that precision='int8' selects vision.int8.onnx and text.int8.onnx files."""
        # Create mock model directory structure
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)

        # Create int8 model files
        (model_dir / "vision.int8.onnx").touch()
        (model_dir / "text.int8.onnx").touch()

        # Create mock resources
        resources = MagicMock(spec=ModelResources)
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        # Create backend with int8 precision
        backend = ONNXRTBackend(resources=resources, precision="int8")

        # Test _select_model_file
        vision_path, vision_prec = backend._select_model_file("vision")
        text_path, text_prec = backend._select_model_file("text")

        assert vision_prec == "int8"
        assert text_prec == "int8"
        assert vision_path.name == "vision.int8.onnx"
        assert text_path.name == "text.int8.onnx"

    def test_precision_q4fp16_selects_q4fp16_files(self, tmp_path: Path) -> None:
        """Test that precision='q4fp16' selects vision.q4fp16.onnx and text.q4fp16.onnx files."""
        # Create mock model directory structure
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)

        # Create q4fp16 model files
        (model_dir / "vision.q4fp16.onnx").touch()
        (model_dir / "text.q4fp16.onnx").touch()

        # Create mock resources
        resources = MagicMock(spec=ModelResources)
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        # Create backend with q4fp16 precision
        backend = ONNXRTBackend(resources=resources, precision="q4fp16")

        # Test _select_model_file
        vision_path, vision_prec = backend._select_model_file("vision")
        text_path, text_prec = backend._select_model_file("text")

        assert vision_prec == "q4fp16"
        assert text_prec == "q4fp16"
        assert vision_path.name == "vision.q4fp16.onnx"
        assert text_path.name == "text.q4fp16.onnx"

    def test_raises_error_when_no_model_files_found(self, tmp_path: Path) -> None:
        """Test that ONNXRTModelLoadingError is raised when no model files exist."""
        # Create mock model directory structure (empty)
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)

        # Create mock resources
        resources = MagicMock(spec=ModelResources)
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        # Create backend with fp16 precision (files don't exist)
        backend = ONNXRTBackend(resources=resources, precision="fp16")

        # Test _select_model_file raises error
        try:
            backend._select_model_file("vision")
            assert False, "Expected ONNXRTModelLoadingError to be raised"
        except Exception as e:
            # Check exception type name and message
            assert type(e).__name__ == "ONNXRTModelLoadingError"
            assert "No vision model found" in str(e)

    def test_precision_attribute_stored(self, tmp_path: Path) -> None:
        """Test that the precision value is stored in the backend instance."""
        # Create mock model directory structure
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)

        # Create default model files
        (model_dir / "vision.onnx").touch()
        (model_dir / "text.onnx").touch()

        # Create mock resources
        resources = MagicMock(spec=ModelResources)
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        # Test with various precision values
        for prec in [None, "fp32", "fp16", "int8", "q4fp16"]:
            backend = ONNXRTBackend(resources=resources, precision=prec)
            assert backend._precision == prec

    def test_no_gpu_available_check_in_initialization(self, tmp_path: Path) -> None:
        """Test that GPU availability check is NOT called during initialization."""
        # Create mock model directory structure
        model_dir = tmp_path / "models" / "test-model" / "onnx"
        model_dir.mkdir(parents=True)

        # Create default model files
        (model_dir / "vision.onnx").touch()
        (model_dir / "text.onnx").touch()

        # Create mock resources
        resources = MagicMock(spec=ModelResources)
        resources.model_name = "test-model"
        resources.model_root_path = tmp_path / "models" / "test-model"
        resources.get_embedding_dim = MagicMock(return_value=512)
        resources.get_image_size = MagicMock(return_value=(224, 224))
        resources.get_normalization_stats = MagicMock(
            return_value={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        resources.tokenizer_config = False
        resources.config = {}

        # Create backend with fp16 precision but CPU providers
        # The old implementation would have fallen back to fp32 based on GPU check
        # The new implementation should still try to use fp16 files
        backend = ONNXRTBackend(
            resources=resources,
            providers=["CPUExecutionProvider"],
            precision="fp16",
        )

        # Verify precision is stored as-is (not modified by GPU check)
        assert backend._precision == "fp16"
        assert not hasattr(backend, "_prefer_fp16")
