"""
Pytest configuration and shared fixtures for Lumen-CLIP tests.

This file provides common fixtures and configuration for all test modules.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import lumen_resources
from lumen_resources import ResourceConfig


@pytest.fixture(scope="session")
def test_cache_dir():
    """Create a temporary cache directory for testing."""
    cache_dir = tempfile.mkdtemp(prefix="lumen_test_cache_")
    yield cache_dir
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def mock_lumen_resources(test_cache_dir):
    """Mock lumen-resources configuration for testing."""
    # Create a mock config that uses our test cache
    config = ResourceConfig(
        cache_dir=test_cache_dir,
        version="test-1.0.0"
    )

    # Patch the global config
    with patch.object(lumen_resources, '_config', config):
        yield config


@pytest.fixture
def sample_image_bytes():
    """Generate sample image bytes for testing."""
    # Create a simple 224x224 RGB image
    np.random.seed(42)  # For reproducible tests
    image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return image.tobytes()


@pytest.fixture
def sample_images():
    """Generate multiple sample images for testing."""
    np.random.seed(42)
    images = {}

    # Different image types
    images["cat"] = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    images["dog"] = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    images["small"] = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    images["large"] = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)

    # Convert to bytes
    return {name: img.tobytes() for name, img in images.items()}


@pytest.fixture
def sample_text_labels():
    """Generate sample text labels for testing."""
    return [
        "cat", "dog", "bird", "fish", "car", "tree", "house", "person",
        "mountain", "ocean", "flower", "book", "computer", "phone"
    ]


@pytest.fixture
def mock_onnx_model_file(tmp_path):
    """Create a mock ONNX model file for testing."""
    model_path = tmp_path / "mock_model.onnx"

    # Create a minimal valid ONNX file
    # For testing purposes, we'll create a simple file that exists
    # In real tests, this would be mocked at the ONNX session level
    model_path.write_bytes(b"mock_onnx_model_content")

    return str(model_path)


@pytest.fixture
def mock_torch_model_file(tmp_path):
    """Create a mock PyTorch model file for testing."""
    model_path = tmp_path / "mock_model.pt"

    # Create a minimal valid PyTorch file
    # For testing purposes, we'll create a simple file that exists
    # In real tests, this would be mocked at the torch load level
    model_path.write_bytes(b"mock_torch_model_content")

    return str(model_path)


@pytest.fixture
def mock_labels_file(tmp_path):
    """Create a mock labels file for testing."""
    labels_path = tmp_path / "mock_labels.npz"

    # Create mock labels
    labels = [f"label_{i}" for i in range(1000)]
    label_embeddings = np.random.randn(1000, 512).astype(np.float32)

    # Normalize embeddings
    label_embeddings = label_embeddings / np.linalg.norm(
        label_embeddings, axis=1, keepdims=True
    )

    np.savez(labels_path, labels=labels, embeddings=label_embeddings)

    return str(labels_path)


class MockONNXSession:
    """Mock ONNX session for testing."""

    def __init__(self, model_path, providers=None):
        self.model_path = model_path
        self.providers = providers or ['CPUExecutionProvider']

        # Mock input/output info
        self.input_info = Mock()
        self.input_info.name = "image"
        self.input_info.shape = [None, 3, 224, 224]
        self.input_info.type = "tensor(float16)"

        self.output_info = Mock()
        self.output_info.name = "embedding"
        self.output_info.shape = [None, 512]

    def get_inputs(self):
        return [self.input_info]

    def get_outputs(self):
        return [self.output_info]

    def run(self, output_names, input_feed):
        # Generate deterministic embeddings based on input
        image_input = input_feed["image"]
        batch_size = image_input.shape[0]

        # Create embeddings based on input statistics
        input_mean = np.mean(image_input)
        np.random.seed(int(input_mean * 1000) % 2**32)

        embeddings = np.random.randn(batch_size, 512).astype(np.float16)
        return {self.output_info.name: embeddings}


@pytest.fixture
def mock_onnx_session_class():
    """Mock ONNXRuntime InferenceSession class."""
    with patch('onnxruntime.InferenceSession', MockONNXSession) as mock_session:
        yield mock_session


# Custom pytest markers for categorizing tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "preprocessing: marks tests for preprocessing pipeline"
    )
    config.addinivalue_line(
        "markers", "normalization: marks tests for normalization methods"
    )
    config.addinivalue_line(
        "markers", "precision: marks tests for float precision handling"
    )
    config.addinivalue_line(
        "markers", "model_loading: marks tests for model loading"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location and names."""
    for item in items:
        # Add unit marker based on location
        if "unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker based on location
        elif "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add specific markers based on test content
        if "preprocessing" in str(item.fspath) or "normalize" in item.name.lower():
            item.add_marker(pytest.mark.preprocessing)
            item.add_marker(pytest.mark.normalization)

        if "float16" in item.name.lower() or "precision" in item.name.lower():
            item.add_marker(pytest.mark.precision)

        if "load" in item.name.lower() or "initialize" in item.name.lower():
            item.add_marker(pytest.mark.model_loading)


# Skip slow tests by default
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests that require GPU"
    )


def pytest_runtest_setup(item):
    """Skip tests based on command line options."""
    # Skip slow tests unless --run-slow is specified
    if pytest.mark.slow in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")

    # Skip GPU tests unless --run-gpu is specified
    if pytest.mark.gpu in item.keywords and not item.config.getoption("--run-gpu"):
        pytest.skip("need --run-gpu option to run")