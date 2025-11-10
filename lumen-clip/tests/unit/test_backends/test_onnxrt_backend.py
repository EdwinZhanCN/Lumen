"""
Unit tests for ONNX Runtime backend.

Tests the ONNX backend including:
- Model initialization
- Float16/Float32 precision handling
- Image preprocessing
- Embedding generation
- Error handling
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import onnxruntime as ort

from lumen_clip.backends.onnxrt_backend import ONNXRuntimeBackend
from lumen_clip.models import ModelInfo


class TestONNXRuntimeBackend:
    """Test cases for ONNX Runtime Backend."""

    @pytest.fixture
    def mock_model_path(self):
        """Create a mock model path."""
        return "/path/to/mock/model.onnx"

    @pytest.fixture
    def mock_onnx_session(self):
        """Create a mock ONNX session."""
        session = Mock()

        # Mock model input/output info
        input_info = Mock()
        input_info.name = "image"
        input_info.shape = [None, 3, 224, 224]  # batch, channels, height, width
        input_info.type = "tensor(float16)"

        output_info = Mock()
        output_info.name = "embedding"
        output_info.shape = [None, 512]

        session.get_inputs.return_value = [input_info]
        session.get_outputs.return_value = [output_info]

        # Mock run method
        def mock_run(output_names, input_feed):
            # Simulate ONNX model output with potential precision issues
            image_input = input_feed["image"]
            batch_size = image_input.shape[0]

            # Generate embeddings that depend on input characteristics
            embeddings = np.random.randn(batch_size, 512).astype(np.float16)

            # Simulate potential NaN issues with float16
            if image_input.dtype == np.float16:
                # Occasionally introduce NaN for testing error handling
                if np.random.random() < 0.1:  # 10% chance
                    embeddings[0, 0] = np.nan

            return {output_info.name: embeddings}

        session.run = mock_run
        return session

    @pytest.fixture
    def onnx_backend(self, mock_model_path):
        """Create an ONNX backend instance."""
        backend = ONNXRuntimeBackend(mock_model_path, device="cpu")
        return backend

    def test_backend_creation(self, onnx_backend, mock_model_path):
        """Test backend creation."""
        assert onnx_backend.model_path == mock_model_path
        assert onnx_backend.device == "cpu"
        assert not onnx_backend.is_initialized
        assert onnx_backend.session is None

    @patch('onnxruntime.InferenceSession')
    def test_initialization_float32(self, mock_session_class, onnx_backend, mock_onnx_session):
        """Test backend initialization with float32 model."""
        mock_session_class.return_value = mock_onnx_session

        # Modify mock to return float32
        mock_onnx_session.get_inputs.return_value[0].type = "tensor(float)"

        onnx_backend.initialize()

        assert onnx_backend.is_initialized
        assert onnx_backend.session is mock_onnx_session
        assert onnx_backend.input_size == (224, 224)
        assert onnx_backend.embedding_dim == 512
        assert onnx_backend.precision == "float32"

    @patch('onnxruntime.InferenceSession')
    def test_initialization_float16(self, mock_session_class, onnx_backend, mock_onnx_session):
        """Test backend initialization with float16 model."""
        mock_session_class.return_value = mock_onnx_session

        onnx_backend.initialize()

        assert onnx_backend.is_initialized
        assert onnx_backend.precision == "float16"

    @patch('onnxruntime.InferenceSession')
    def test_initialization_dynamic_batch_size(self, mock_session_class, onnx_backend, mock_onnx_session):
        """Test backend initialization with dynamic batch size."""
        mock_session_class.return_value = mock_onnx_session

        # Modify mock to have dynamic batch dimension
        mock_onnx_session.get_inputs.return_value[0].shape = ['batch_size', 3, 256, 256]

        onnx_backend.initialize()

        assert onnx_backend.is_initialized
        assert onnx_backend.input_size == (256, 256)

    @patch('onnxruntime.InferenceSession')
    def test_get_info(self, mock_session_class, onnx_backend, mock_onnx_session):
        """Test getting backend information."""
        mock_session_class.return_value = mock_onnx_session

        onnx_backend.initialize()
        info = onnx_backend.get_info()

        assert info.runtime == "onnx"
        assert info.model_id == str(onnx_backend.model_path)
        assert info.model_name == "onnx_model"
        assert info.image_embedding_dim == 512
        assert info.text_embedding_dim == 512
        assert info.device == "cpu"
        assert info.precisions == ["float16"]

    def test_preprocess_image_uint8(self, onnx_backend):
        """Test image preprocessing with uint8 input."""
        # Create a simple 224x224 RGB image
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        image_bytes = image.tobytes()

        onnx_backend.input_size = (224, 224)
        processed = onnx_backend._preprocess_image(image_bytes)

        assert processed.shape == (1, 3, 224, 224)
        assert processed.dtype == np.float32  # Should default to float32

    def test_preprocess_image_with_alpha_channel(self, onnx_backend):
        """Test image preprocessing with RGBA input."""
        # Create a simple 224x224 RGBA image
        image = np.random.randint(0, 256, (224, 224, 4), dtype=np.uint8)
        image_bytes = image.tobytes()

        onnx_backend.input_size = (224, 224)
        processed = onnx_backend._preprocess_image(image_bytes)

        assert processed.shape == (1, 3, 224, 224)  # Should drop alpha channel
        assert processed.dtype == np.float32

    def test_preprocess_image_wrong_size(self, onnx_backend):
        """Test image preprocessing with wrong image size."""
        # Create an image with wrong dimensions
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image_bytes = image.tobytes()

        onnx_backend.input_size = (224, 224)

        with pytest.raises(ValueError, match="Input image must be RGB with shape"):
            onnx_backend._preprocess_image(image_bytes)

    def test_preprocess_image_corrupt_data(self, onnx_backend):
        """Test image preprocessing with corrupt data."""
        # Create incomplete image data
        image_bytes = b"incomplete_image_data"

        onnx_backend.input_size = (224, 224)

        with pytest.raises(ValueError, match="Input image must be RGB with shape"):
            onnx_backend._preprocess_image(image_bytes)

    def test_normalize_simple_float16_fallback(self, onnx_backend):
        """Test simple normalization with float16 fallback."""
        # Create input that would cause overflow in float16
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        onnx_backend.precision = "float16"

        # This should work with float16 fallback to float32
        normalized = onnx_backend._normalize_simple(image.astype(np.float32))

        assert normalized.shape == (3, 224, 224)
        assert normalized.dtype == np.float16  # Should convert back to float16
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))

    def test_normalize_simple_check_invalid_values(self, onnx_backend):
        """Test simple normalization checks for invalid values."""
        # Create array with NaN and Inf
        image = np.full((3, 224, 224), 128.0, dtype=np.float32)
        image[0, 0, 0] = np.nan
        image[0, 0, 1] = np.inf
        image[0, 0, 2] = -np.inf

        onnx_backend.precision = "float32"

        with pytest.raises(ValueError, match="contains NaN or infinite values"):
            onnx_backend._normalize_simple(image)

    @patch('onnxruntime.InferenceSession')
    def test_image_to_vector_float32(self, mock_session_class, onnx_backend, mock_onnx_session):
        """Test image to vector conversion with float32."""
        mock_session_class.return_value = mock_onnx_session
        mock_onnx_session.get_inputs.return_value[0].type = "tensor(float)"

        # Modify mock run to return float32
        def mock_run_float32(output_names, input_feed):
            batch_size = input_feed["image"].shape[0]
            embeddings = np.random.randn(batch_size, 512).astype(np.float32)
            return {"embedding": embeddings}

        mock_onnx_session.run = mock_run_float32

        onnx_backend.initialize()

        # Create a test image
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        image_bytes = image.tobytes()

        embedding = onnx_backend.image_to_vector(image_bytes)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32

    @patch('onnxruntime.InferenceSession')
    def test_image_to_vector_float16_with_nan(self, mock_session_class, onnx_backend, mock_onnx_session):
        """Test image to vector conversion with float16 and NaN handling."""
        mock_session_class.return_value = mock_onnx_session

        # Modify mock to return NaN
        def mock_run_with_nan(output_names, input_feed):
            batch_size = input_feed["image"].shape[0]
            embeddings = np.random.randn(batch_size, 512).astype(np.float16)
            embeddings[0, 0] = np.nan  # Inject NaN
            return {"embedding": embeddings}

        mock_onnx_session.run = mock_run_with_nan

        onnx_backend.initialize()

        # Create a test image
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        image_bytes = image.tobytes()

        # Should raise ValueError due to NaN detection
        with pytest.raises(ValueError, match="contains NaN or infinite values"):
            onnx_backend.image_to_vector(image_bytes)

    @patch('onnxruntime.InferenceSession')
    def test_image_to_vector_float16_fallback(self, mock_session_class, onnx_backend, mock_onnx_session):
        """Test image to vector conversion with float16 fallback to float32."""
        mock_session_class.return_value = mock_onnx_session

        # First call returns NaN, second call succeeds
        call_count = [0]

        def mock_run_with_fallback(output_names, input_feed):
            call_count[0] += 1
            batch_size = input_feed["image"].shape[0]

            if call_count[0] == 1:
                # First call with float16 - return NaN
                embeddings = np.random.randn(batch_size, 512).astype(np.float16)
                embeddings[0, 0] = np.nan
            else:
                # Second call with float32 - succeed
                embeddings = np.random.randn(batch_size, 512).astype(np.float32)

            return {"embedding": embeddings}

        mock_onnx_session.run = mock_run_with_fallback

        onnx_backend.initialize()

        # Create a test image
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        image_bytes = image.tobytes()

        # Should succeed after fallback
        embedding = onnx_backend.image_to_vector(image_bytes)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32  # Should be float32 after fallback
        assert call_count[0] == 2  # Should have made two calls

    def test_image_to_vector_not_initialized(self, onnx_backend):
        """Test image to vector conversion when not initialized."""
        with pytest.raises(RuntimeError, match="Backend not initialized"):
            onnx_backend.image_to_vector(b"fake_image")

    def test_text_to_vector_not_supported(self, onnx_backend):
        """Test that text to vector is not supported."""
        with pytest.raises(NotImplementedError, match="Text encoding not supported"):
            onnx_backend.text_to_vector("test")

    def test_text_batch_to_vectors_not_supported(self, onnx_backend):
        """Test that text batch encoding is not supported."""
        with pytest.raises(NotImplementedError, match="Batch text encoding not supported"):
            onnx_backend.text_batch_to_vectors(["test1", "test2"])

    @patch('onnxruntime.InferenceSession')
    def test_embedding_normalization(self, mock_session_class, onnx_backend, mock_onnx_session):
        """Test that output embeddings are properly normalized."""
        mock_session_class.return_value = mock_onnx_session

        # Create non-normalized embedding
        def mock_run_unnormalized(output_names, input_feed):
            batch_size = input_feed["image"].shape[0]
            # Create embedding with norm != 1
            embedding = np.random.randn(batch_size, 512).astype(np.float16) * 10
            return {"embedding": embedding}

        mock_onnx_session.run = mock_run_unnormalized

        onnx_backend.initialize()

        # Create a test image
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        image_bytes = image.tobytes()

        embedding = onnx_backend.image_to_vector(image_bytes)

        # Check that embedding is normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6, f"Embedding norm is {norm}, expected 1.0"

    def test_different_input_sizes(self, onnx_backend):
        """Test preprocessing with different input sizes."""
        test_sizes = [(224, 224), (256, 256), (384, 384)]

        for size in test_sizes:
            onnx_backend.input_size = size

            # Create image of correct size
            image = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            image_bytes = image.tobytes()

            processed = onnx_backend._preprocess_image(image_bytes)

            assert processed.shape == (1, 3, *size)
            assert processed.dtype == np.float32

    def test_device_providers_detection(self):
        """Test device providers detection."""
        with patch('onnxruntime.get_available_providers') as mock_providers:
            mock_providers.return_value = ['CPUExecutionProvider', 'CUDAExecutionProvider']

            # Test CPU device
            backend_cpu = ONNXRuntimeBackend("/fake/path", device="cpu")
            assert backend_cpu.device == "cpu"

            # Test CUDA device
            backend_cuda = ONNXRuntimeBackend("/fake/path", device="cuda")
            assert backend_cuda.device == "cuda"