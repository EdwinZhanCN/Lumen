"""
Tests for image preprocessing and normalization.

Tests the critical preprocessing pipeline that was causing classification issues:
- Normalization methods (ImageNet vs simple scaling)
- Float16 vs Float32 precision handling
- Value range validation
- NaN/Inf detection and handling
"""

import pytest
import numpy as np
from typing import Tuple

from lumen_clip.backends.onnxrt_backend import ONNXRuntimeBackend


class TestNormalizationMethods:
    """Test different normalization methods and their effects."""

    def test_imagenet_normalization(self):
        """Test ImageNet normalization values."""
        # Standard ImageNet statistics
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        # Create test image in [0, 255] range
        image = np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8).astype(np.float32)

        # Normalize to [0, 1]
        image_normalized = image / 255.0

        # Apply ImageNet normalization
        imagenet_result = (image_normalized - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

        # Check result properties
        assert imagenet_result.shape == (3, 224, 224)
        assert imagenet_result.dtype == np.float32

        # ImageNet normalization typically produces values in approximately [-2.4, 2.6] range
        assert np.min(imagenet_result) < -1.0  # Should have negative values
        assert np.max(imagenet_result) > 1.0   # Should have values > 1

        # Check for NaN/Inf
        assert not np.any(np.isnan(imagenet_result))
        assert not np.any(np.isinf(imagenet_result))

    def test_simple_normalization(self):
        """Test simple [-1, 1] normalization."""
        # Create test image in [0, 255] range
        image = np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8).astype(np.float32)

        # Simple normalization to [-1, 1]
        simple_result = (image / 127.5) - 1.0

        # Check result properties
        assert simple_result.shape == (3, 224, 224)
        assert simple_result.dtype == np.float32

        # Simple normalization should produce values in [-1, 1] range
        assert np.min(simple_result) >= -1.0
        assert np.max(simple_result) <= 1.0

        # Check boundaries
        assert np.min(simple_result) == -1.0  # Black pixels (0) -> -1
        assert np.max(simple_result) == 1.0   # White pixels (255) -> 1

        # Check for NaN/Inf
        assert not np.any(np.isnan(simple_result))
        assert not np.any(np.isinf(simple_result))

    def test_imagenet_vs_simple_comparison(self):
        """Compare ImageNet vs simple normalization outputs."""
        # Create a fixed test image for comparison
        np.random.seed(42)
        image = np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8).astype(np.float32)

        # ImageNet normalization
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        imagenet_result = ((image / 255.0) - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

        # Simple normalization
        simple_result = (image / 127.5) - 1.0

        # Compare statistics
        print(f"ImageNet - mean: {np.mean(imagenet_result):.4f}, std: {np.std(imagenet_result):.4f}")
        print(f"Simple - mean: {np.mean(simple_result):.4f}, std: {np.std(simple_result):.4f}")

        # ImageNet should have different distribution than simple
        assert not np.allclose(np.mean(imagenet_result, axis=(1, 2)), np.mean(simple_result, axis=(1, 2)), atol=0.1)

        # Both should be valid
        assert not np.any(np.isnan(imagenet_result))
        assert not np.any(np.isnan(simple_result))


class TestFloatPrecisionHandling:
    """Test float16 vs float32 precision handling."""

    def test_float16_overflow_detection(self):
        """Test detection of float16 overflow conditions."""
        # Create values that would cause overflow in float16
        large_values = np.array([65504.0, 100000.0, -100000.0], dtype=np.float32)

        # Convert to float16
        float16_values = large_values.astype(np.float16)

        # Check for overflow
        assert float16_values[0] == 65504.0  # Max float16 value
        assert float16_values[1] == np.inf   # Should overflow to inf
        assert float16_values[2] == -np.inf  # Should overflow to -inf

        # Detection function should find these
        def has_overflow_issues(arr: np.ndarray) -> bool:
            if arr.dtype == np.float16:
                return np.any(np.isinf(arr))
            return False

        assert has_overflow_issues(float16_values)

    def test_float16_precision_loss(self):
        """Test precision loss in float16."""
        # Create values with high precision
        precise_values = np.array([1.123456789, 2.987654321, 3.141592653], dtype=np.float32)

        # Convert to float16 and back
        float16_values = precise_values.astype(np.float16).astype(np.float32)

        # Check precision loss
        precision_loss = np.abs(precise_values - float16_values)
        print(f"Precision loss: {precision_loss}")

        # Should have some precision loss
        assert np.any(precision_loss > 1e-3)

        # But not too much
        assert np.all(precision_loss < 1e-1)

    def test_safe_float16_conversion(self):
        """Test safe conversion from float16 to float32."""
        def safe_float16_to_float32(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
            """Convert float16 to float32, checking for issues."""
            if arr.dtype != np.float16:
                return arr.astype(np.float32), False

            # Check for invalid values
            has_nan = np.any(np.isnan(arr))
            has_inf = np.any(np.isinf(arr))

            if has_nan or has_inf:
                return None, True

            return arr.astype(np.float32), False

        # Test valid float16
        valid_arr = np.random.randn(10, 10).astype(np.float16)
        converted, has_issues = safe_float16_to_float32(valid_arr)
        assert not has_issues
        assert converted.dtype == np.float32

        # Test invalid float16
        invalid_arr = np.array([1.0, np.nan, 2.0, np.inf], dtype=np.float16)
        converted, has_issues = safe_float16_to_float32(invalid_arr)
        assert has_issues
        assert converted is None


class TestONNXBackendNormalization:
    """Test ONNX backend normalization implementation."""

    def test_normalize_simple_implementation(self):
        """Test the simple normalization implementation used in ONNX backend."""
        # Create test image
        image = np.random.randint(0, 256, (3, 224, 224), dtype=np.float32)

        # Replicate ONNX backend's simple normalization
        def normalize_simple(image: np.ndarray, precision: str = "float32") -> np.ndarray:
            # Normalize to [-1, 1] range
            normalized = (image / 127.5) - 1.0

            # Handle precision
            if precision == "float16":
                try:
                    normalized = normalized.astype(np.float16)
                    # Check for overflow/underflow
                    if np.any(np.isinf(normalized)) or np.any(np.isnan(normalized)):
                        raise ValueError("Float16 precision issues detected")
                except Exception:
                    # Fallback to float32
                    normalized = (image / 127.5) - 1.0
                    normalized = normalized.astype(np.float32)

            return normalized

        # Test float32
        result_f32 = normalize_simple(image, "float32")
        assert result_f32.shape == image.shape
        assert result_f32.dtype == np.float32
        assert np.min(result_f32) >= -1.0
        assert np.max(result_f32) <= 1.0

        # Test float16
        result_f16 = normalize_simple(image, "float16")
        assert result_f16.shape == image.shape
        # Should be float16 if successful, or float32 if fallback occurred
        assert result_f16.dtype in [np.float16, np.float32]

    def test_normalization_with_edge_values(self):
        """Test normalization with edge case values."""
        # Test with all black image (all zeros)
        black_image = np.zeros((3, 224, 224), dtype=np.float32)
        normalized_black = (black_image / 127.5) - 1.0
        assert np.all(normalized_black == -1.0)

        # Test with all white image (all 255)
        white_image = np.full((3, 224, 224), 255.0, dtype=np.float32)
        normalized_white = (white_image / 127.5) - 1.0
        assert np.all(normalized_white == 1.0)

        # Test with gray image (all 128)
        gray_image = np.full((3, 224, 224), 128.0, dtype=np.float32)
        normalized_gray = (gray_image / 127.5) - 1.0
        assert np.allclose(normalized_gray, 0.0039)  # Approximately 0

    def test_normalization_statistics(self):
        """Test normalization preserves image statistics appropriately."""
        # Create a test image with known statistics
        np.random.seed(42)
        image = np.random.randint(0, 256, (3, 224, 224), dtype=np.float32)

        # Original statistics
        orig_mean = np.mean(image)
        orig_std = np.std(image)

        # Simple normalization
        normalized = (image / 127.5) - 1.0

        # Normalized statistics
        norm_mean = np.mean(normalized)
        norm_std = np.std(normalized)

        print(f"Original - mean: {orig_mean:.2f}, std: {orig_std:.2f}")
        print(f"Normalized - mean: {norm_mean:.4f}, std: {norm_std:.4f}")

        # Normalization should preserve relative relationships
        # The mean should be close to 0 (since input is centered around 127.5)
        assert abs(norm_mean) < 0.1

        # Standard deviation should be scaled appropriately
        expected_std = orig_std / 127.5
        assert abs(norm_std - expected_std) < 0.01


class TestPreprocessingValidation:
    """Test validation functions for preprocessing pipeline."""

    def test_check_embedding_quality(self):
        """Test embedding quality validation."""
        def validate_embedding(embedding: np.ndarray) -> Tuple[bool, str]:
            """Validate embedding quality."""
            if embedding is None:
                return False, "Embedding is None"

            if not isinstance(embedding, np.ndarray):
                return False, "Embedding is not numpy array"

            if len(embedding.shape) != 1:
                return False, f"Embedding should be 1D, got shape {embedding.shape}"

            if np.any(np.isnan(embedding)):
                return False, "Embedding contains NaN values"

            if np.any(np.isinf(embedding)):
                return False, "Embedding contains Inf values"

            norm = np.linalg.norm(embedding)
            if abs(norm - 1.0) > 1e-5:
                return False, f"Embedding not normalized: norm={norm}"

            return True, "Valid embedding"

        # Test valid embedding
        valid_embedding = np.random.randn(512).astype(np.float32)
        valid_embedding = valid_embedding / np.linalg.norm(valid_embedding)
        is_valid, message = validate_embedding(valid_embedding)
        assert is_valid
        assert message == "Valid embedding"

        # Test invalid embeddings
        invalid_embeddings = [
            None,
            "not_array",
            np.random.randn(10, 10),  # Wrong shape
            np.array([1.0, np.nan, 2.0]),  # Contains NaN
            np.array([1.0, np.inf, 2.0]),  # Contains Inf
            np.random.randn(512),  # Not normalized
        ]

        for invalid_emb in invalid_embeddings:
            is_valid, message = validate_embedding(invalid_emb)
            assert not is_valid
            assert message != "Valid embedding"

    def test_preprocessing_pipeline_validation(self):
        """Test complete preprocessing pipeline validation."""
        def validate_preprocessing_pipeline(
            image_bytes: bytes,
            target_size: Tuple[int, int],
            backend_precision: str = "float32"
        ) -> Tuple[bool, str]:
            """Validate the complete preprocessing pipeline."""
            try:
                # Mock image preprocessing
                image_shape = (target_size[1], target_size[0], 3)  # height, width, channels
                expected_length = np.prod(image_shape)

                if len(image_bytes) != expected_length:
                    return False, f"Image size mismatch: expected {expected_length}, got {len(image_bytes)}"

                # Convert to numpy array
                image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(image_shape)

                # Check for alpha channel (4 channels instead of 3)
                if image_shape[2] == 4:
                    image = image[:, :, :3]  # Drop alpha channel
                    image_shape = (image_shape[0], image_shape[1], 3)

                # Normalize
                image_float = image.astype(np.float32)
                normalized = (image_float / 127.5) - 1.0

                # Handle precision
                if backend_precision == "float16":
                    try:
                        normalized = normalized.astype(np.float16)
                        if np.any(np.isinf(normalized)) or np.any(np.isnan(normalized)):
                            normalized = normalized.astype(np.float32)
                    except Exception:
                        normalized = normalized.astype(np.float32)

                # Validate output
                if np.any(np.isnan(normalized)):
                    return False, "Preprocessing produced NaN values"

                if np.any(np.isinf(normalized)):
                    return False, "Preprocessing produced Inf values"

                return True, "Preprocessing successful"

            except Exception as e:
                return False, f"Preprocessing failed: {str(e)}"

        # Test with valid image
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        image_bytes = image.tobytes()

        is_valid, message = validate_preprocessing_pipeline(image_bytes, (224, 224))
        assert is_valid
        assert "successful" in message

        # Test with invalid image size
        wrong_size_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        wrong_size_bytes = wrong_size_image.tobytes()

        is_valid, message = validate_preprocessing_pipeline(wrong_size_bytes, (224, 224))
        assert not is_valid
        assert "size mismatch" in message