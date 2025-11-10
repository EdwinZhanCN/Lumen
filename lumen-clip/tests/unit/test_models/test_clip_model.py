"""
Unit tests for CLIP model functionality.

Tests the CLIP model manager including:
- Model initialization
- Image and text encoding
- Classification accuracy
- Scene classification
- Error handling
"""

import pytest
import numpy as np
from typing import Any

from lumen_clip.general_clip.clip_model import CLIPModelManager
from lumen_clip.backends import BaseClipBackend


class MockBackend(BaseClipBackend):
    """Mock backend for testing CLIP model manager."""

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.is_initialized = False

    def initialize(self) -> None:
        self.is_initialized = True

    def image_to_vector(self, image_bytes: bytes) -> np.ndarray:
        # Return deterministic embeddings based on image length for testing
        np.random.seed(len(image_bytes))
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    def text_to_vector(self, text: str) -> np.ndarray:
        # Return deterministic embeddings based on text for testing
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    def text_batch_to_vectors(self, texts: list[str]) -> np.ndarray:
        embeddings = np.array([self.text_to_vector(text) for text in texts])
        return embeddings

    def get_info(self) -> Any:
        from lumen_clip.models import BackendInfo
        return BackendInfo(
            runtime="mock",
            model_id="mock_model",
            model_name="MockCLIP",
            version="1.0.0",
            image_embedding_dim=self.embedding_dim,
            text_embedding_dim=self.embedding_dim,
            device="cpu"
        )


class MockResources:
    """Mock resources for testing."""

    def __init__(self, num_labels: int = 1000, has_labels: bool = True):
        self.model_name = "mock_clip"
        self.runtime = "mock"
        self.num_labels = num_labels

        if has_labels:
            # Generate mock labels
            self.labels = np.array([f"label_{i}" for i in range(num_labels)])
            # Generate mock text embeddings
            np.random.seed(42)
            self.label_embeddings = np.random.randn(num_labels, 512).astype(np.float32)
            # Normalize embeddings
            self.label_embeddings = self.label_embeddings / np.linalg.norm(
                self.label_embeddings, axis=1, keepdims=True
            )
        else:
            self.labels = None
            self.label_embeddings = None

    def has_classification_support(self) -> bool:
        return self.labels is not None


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    return MockBackend()


@pytest.fixture
def mock_resources():
    """Create mock resources with labels."""
    return MockResources(num_labels=1000, has_labels=True)


@pytest.fixture
def mock_resources_no_labels():
    """Create mock resources without labels."""
    return MockResources(num_labels=1000, has_labels=False)


@pytest.fixture
def clip_model_manager(mock_backend, mock_resources):
    """Create a CLIP model manager for testing."""
    manager = CLIPModelManager(mock_backend, mock_resources)
    manager.initialize()
    return manager


@pytest.fixture
def clip_model_manager_no_labels(mock_backend, mock_resources_no_labels):
    """Create a CLIP model manager without classification support."""
    manager = CLIPModelManager(mock_backend, mock_resources_no_labels)
    manager.initialize()
    return manager


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return b"fake_image_data_for_testing"


class TestCLIPModelManager:
    """Test cases for CLIP Model Manager."""

    def test_initialization_with_labels(self, mock_backend, mock_resources):
        """Test model manager initialization with classification support."""
        manager = CLIPModelManager(mock_backend, mock_resources)

        assert not manager.is_initialized
        assert manager.supports_classification

        manager.initialize()

        assert manager.is_initialized
        assert len(manager.labels) == 1000
        assert manager.text_embeddings is not None
        assert manager.scene_prompt_embeddings is not None
        assert manager._load_time is not None

    def test_initialization_without_labels(self, mock_backend, mock_resources_no_labels):
        """Test model manager initialization without classification support."""
        manager = CLIPModelManager(mock_backend, mock_resources_no_labels)

        assert not manager.supports_classification

        manager.initialize()

        assert manager.is_initialized
        assert len(manager.labels) == 0
        assert manager.text_embeddings is None
        assert manager.scene_prompt_embeddings is not None  # Should still have scene classification

    def test_double_initialization(self, clip_model_manager):
        """Test that double initialization doesn't cause issues."""
        clip_model_manager.initialize()  # Should not raise or reinitialize

        assert clip_model_manager.is_initialized

    def test_encode_image(self, clip_model_manager, sample_image):
        """Test image encoding."""
        embedding = clip_model_manager.encode_image(sample_image)

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding) == 512
        # Check unit normalization
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-6

    def test_encode_text(self, clip_model_manager):
        """Test text encoding."""
        text = "cat"
        embedding = clip_model_manager.encode_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding) == 512
        # Check unit normalization
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-6

    def test_classify_image(self, clip_model_manager, sample_image):
        """Test image classification."""
        results = clip_model_manager.classify_image(sample_image, top_k=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        assert len(results) > 0

        # Check result structure
        for label, probability in results:
            assert isinstance(label, str)
            assert isinstance(probability, (float, np.floating))
            assert 0.0 <= probability <= 1.0

        # Check sorting (should be descending by probability)
        probabilities = [prob for _, prob in results]
        assert probabilities == sorted(probabilities, reverse=True)

        # Check probability sum (should be close to 1.0 for all labels)
        assert abs(sum(probabilities) - 1.0) < 0.1  # Allow some tolerance due to top-k

    def test_classify_image_different_top_k(self, clip_model_manager, sample_image):
        """Test classification with different top_k values."""
        # Test top_k=1
        results_1 = clip_model_manager.classify_image(sample_image, top_k=1)
        assert len(results_1) == 1

        # Test top_k=10
        results_10 = clip_model_manager.classify_image(sample_image, top_k=10)
        assert len(results_10) <= 10

        # top_k=1 should be the same as first element of top_k=10
        assert results_1[0][0] == results_10[0][0]  # Same label
        assert abs(results_1[0][1] - results_10[0][1]) < 1e-6  # Same probability

    def test_classify_image_without_support(self, clip_model_manager_no_labels, sample_image):
        """Test classification when not supported."""
        with pytest.raises(RuntimeError, match="Classification not supported"):
            clip_model_manager_no_labels.classify_image(sample_image)

    def test_classify_scene(self, clip_model_manager, sample_image):
        """Test scene classification."""
        scene_label, confidence = clip_model_manager.classify_scene(sample_image)

        assert isinstance(scene_label, str)
        assert isinstance(confidence, (float, np.floating))
        assert 0.0 <= confidence <= 1.0

        # Check that scene label is one of expected scene types
        expected_scenes = ["person", "animal", "vehicle", "food", "building", "nature", "object", "landscape"]
        assert scene_label in expected_scenes

    def test_classify_scene_without_embeddings(self, mock_backend, mock_resources_no_labels):
        """Test scene classification when scene embeddings fail to initialize."""
        manager = CLIPModelManager(mock_backend, mock_resources_no_labels)
        # Force scene embeddings to None
        manager.scene_prompt_embeddings = None
        manager.initialize()

        with pytest.raises(RuntimeError, match="Scene embeddings are not available"):
            manager.classify_scene(b"fake_image")

    def test_info(self, clip_model_manager):
        """Test model info retrieval."""
        info = clip_model_manager.info()

        assert info.model_name == "mock_clip"
        assert info.model_id == "mock_clip_mock"
        assert info.supports_classification is True
        assert info.is_initialized is True
        assert info.load_time is not None
        assert info.num_labels == 1000
        assert info.scene_classification_available is True
        assert info.backend_info is not None

    def test_info_without_labels(self, clip_model_manager_no_labels):
        """Test model info without classification support."""
        info = clip_model_manager_no_labels.info()

        assert info.supports_classification is False
        assert info.num_labels == 0
        assert info.scene_classification_available is True  # Scene classification should still work

    def test_ensure_initialized(self, mock_backend, mock_resources):
        """Test that operations fail when model is not initialized."""
        manager = CLIPModelManager(mock_backend, mock_resources)
        # Don't initialize

        with pytest.raises(RuntimeError, match="Model manager not initialized"):
            manager.encode_image(b"fake_image")

        with pytest.raises(RuntimeError, match="Model manager not initialized"):
            manager.encode_text("test")

        with pytest.raises(RuntimeError, match="Model manager not initialized"):
            manager.classify_image(b"fake_image")

        with pytest.raises(RuntimeError, match="Model manager not initialized"):
            manager.classify_scene(b"fake_image")


class TestEmbeddingQuality:
    """Test embedding quality and normalization."""

    def test_image_embedding_normalization(self, clip_model_manager, sample_image):
        """Test that image embeddings are properly normalized."""
        embedding = clip_model_manager.encode_image(sample_image)

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6, f"Embedding norm is {norm}, expected 1.0"

        # Check for NaN or Inf values
        assert not np.any(np.isnan(embedding)), "Embedding contains NaN values"
        assert not np.any(np.isinf(embedding)), "Embedding contains Inf values"

    def test_text_embedding_normalization(self, clip_model_manager):
        """Test that text embeddings are properly normalized."""
        embedding = clip_model_manager.encode_text("test text")

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6, f"Embedding norm is {norm}, expected 1.0"

        # Check for NaN or Inf values
        assert not np.any(np.isnan(embedding)), "Embedding contains NaN values"
        assert not np.any(np.isinf(embedding)), "Embedding contains Inf values"

    def test_classification_probability_distribution(self, clip_model_manager, sample_image):
        """Test that classification probabilities form a valid distribution."""
        results = clip_model_manager.classify_image(sample_image, top_k=100)

        probabilities = [prob for _, prob in results]

        # All probabilities should be between 0 and 1
        for prob in probabilities:
            assert 0.0 <= prob <= 1.0

        # Probabilities should be in descending order
        assert probabilities == sorted(probabilities, reverse=True)

        # No NaN or Inf in probabilities
        assert not any(np.isnan(probabilities)), "Probabilities contain NaN values"
        assert not any(np.isinf(probabilities)), "Probabilities contain Inf values"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_image(self, clip_model_manager):
        """Test classification with empty image data."""
        results = clip_model_manager.classify_image(b"", top_k=5)
        assert len(results) == 5  # Should still work

    def test_very_large_image(self, clip_model_manager):
        """Test classification with large image data."""
        large_image = b"x" * 1000000  # 1MB of fake data
        results = clip_model_manager.classify_image(large_image, top_k=3)
        assert len(results) == 3

    def test_top_k_larger_than_labels(self, clip_model_manager, sample_image):
        """Test when top_k is larger than number of labels."""
        results = clip_model_manager.classify_image(sample_image, top_k=2000)
        assert len(results) <= 1000  # Should not exceed number of labels

    def test_top_k_zero(self, clip_model_manager, sample_image):
        """Test with top_k=0."""
        results = clip_model_manager.classify_image(sample_image, top_k=0)
        assert len(results) == 0

    def test_negative_top_k(self, clip_model_manager, sample_image):
        """Test with negative top_k."""
        results = clip_model_manager.classify_image(sample_image, top_k=-1)
        assert len(results) == 0