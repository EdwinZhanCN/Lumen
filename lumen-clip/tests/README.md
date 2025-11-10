# Lumen-CLIP Testing Framework

This comprehensive test suite systematically tests the CLIP model functionality, backend implementations, and preprocessing pipelines that were identified as sources of inference problems.

## 🎯 Test Coverage Areas

### 1. **Preprocessing Pipeline** (`test_preprocessing/`)
- **Normalization Methods**: Tests ImageNet vs simple [-1,1] scaling
- **Float Precision Handling**: Tests float16 vs float32 compatibility
- **Value Range Validation**: Detects NaN/Inf values in preprocessing
- **Edge Case Handling**: Tests overflow/underflow conditions

### 2. **CLIP Model Manager** (`test_models/test_clip_model.py`)
- **Model Initialization**: Tests proper setup with/without labels
- **Image/Text Encoding**: Validates embedding generation and normalization
- **Classification Pipeline**: Tests probability distribution and sorting
- **Error Handling**: Tests invalid inputs and edge cases

### 3. **ONNX Backend** (`test_backends/test_onnxrt_backend.py`)
- **Model Loading**: Tests ONNX session initialization
- **Precision Handling**: Tests float16 fallback to float32
- **Image Preprocessing**: Tests input size detection and validation
- **NaN Detection**: Tests detection of invalid model outputs

## 🚀 Quick Start

### Run All Tests
```bash
python tests/run_tests.py --all
```

### Run Specific Test Categories
```bash
# Test preprocessing pipeline (critical for debugging classification issues)
python tests/run_tests.py --preprocessing

# Test CLIP model functionality
python tests/run_tests.py --model

# Test backend implementations
python tests/run_tests.py --backend

# Run unit tests only
python tests/run_tests.py --unit
```

### Test Specific Runtimes
```bash
# Test Torch runtime only
python tests/run_tests.py --preprocessing --torch

# Test ONNX runtime only
python tests/run_tests.py --preprocessing --onnx

# Test both runtimes
python tests/run_tests.py --all --both
```

### Quick Development Tests
```bash
# Run a quick subset of tests for development
python tests/run_tests.py --quick

# Run tests with debug output
python tests/run_tests.py --debug
```

## 📋 Configuration-Based Testing

The test framework uses the same lumen-resources configuration system as the server:

### Test Configuration Files
- `test_clip_torch.yaml` - Torch runtime configuration
- `test_clip_onnx.yaml` - ONNX runtime configuration

### Using Real Configurations
Tests load real configurations from `fixtures/configs/` using lumen-resources:
- Same validation logic as server
- Real model paths and runtime settings
- Tests fail if config is invalid (same as server would)

### Environment Variables
```bash
# Set test cache directory
export LUMEN_CACHE_DIR=./test_cache

# Override config path
export LUMEN_CONFIG=./tests/fixtures/configs/test_clip_torch.yaml
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)
- **Component Isolation**: Test individual components in isolation
- **Mock Dependencies**: Use mocks for external dependencies
- **Fast Execution**: Designed for rapid iteration during development
- **Edge Cases**: Comprehensive coverage of error conditions

### Integration Tests (`tests/integration/`)
- **Component Interactions**: Test how components work together
- **Real Configurations**: Use actual configuration files
- **End-to-End Workflows**: Test complete inference pipelines
- **Performance Validation**: Test latency and throughput

### Property Tests
- **Embedding Quality**: Validate that embeddings are unit-normalized
- **Probability Distributions**: Ensure classification results form valid distributions
- **Preprocessing Consistency**: Validate that preprocessing produces expected ranges

## 🔍 Key Test Scenarios

### 1. **Normalization Method Comparison**
Tests the two normalization approaches that were causing classification issues:

```python
# ImageNet normalization (was causing issues)
imagenet_normalized = ((image / 255.0) - imagenet_mean) / imagenet_std

# Simple normalization (works better)
simple_normalized = (image / 127.5) - 1.0
```

### 2. **Float16 Precision Handling**
Tests the critical float16 overflow/NaN detection:

```python
# Detect float16 precision issues
if np.any(np.isinf(float16_output)) or np.any(np.isnan(float16_output)):
    # Fallback to float32
    output = process_with_float32(input)
```

### 3. **Embedding Validation**
Comprehensive validation of model outputs:

```python
def validate_embedding(embedding):
    # Check for invalid values
    assert not np.any(np.isnan(embedding))
    assert not np.any(np.isinf(embedding))

    # Check unit normalization
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-6

    return True
```

## 📊 Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.preprocessing` - Preprocessing pipeline tests
- `@pytest.mark.normalization` - Normalization method tests
- `@pytest.mark.precision` - Float precision handling tests
- `@pytest.mark.model_loading` - Model loading and initialization tests
- `@pytest.mark.slow` - Tests that take a long time
- `@pytest.mark.gpu` - Tests that require GPU access

### Filter Tests by Markers
```bash
# Run only preprocessing tests
pytest -m "preprocessing"

# Run normalization and precision tests
pytest -m "normalization or precision"

# Run fast tests only (exclude slow tests)
pytest -m "not slow"
```

## 🐛 Debugging Classification Issues

### Step 1: Test Preprocessing Pipeline
```bash
python tests/run_tests.py --preprocessing --debug
```

### Step 2: Test Backend Implementation
```bash
python tests/run_tests.py --backend --onnx
```

### Step 3: Test Complete Model Pipeline
```bash
python tests/run_tests.py --model --onnx
```

### Step 4: Run Integration Tests
```bash
python tests/run_tests.py --integration --onnx
```

## 📈 Coverage Reports

Generate coverage reports to see what code is being tested:

```bash
python tests/run_tests.py --all --coverage
```

This generates:
- Terminal coverage summary
- HTML coverage report in `htmlcov/`

## 🔧 Test Configuration

### pytest.ini
Main pytest configuration file with:
- Test discovery patterns
- Custom markers
- Default options

### conftest.py
Shared fixtures and utilities:
- `sample_image_bytes` - Generates test images
- `mock_onnx_session` - Mocks ONNX runtime
- `sample_text_labels` - Test labels
- Test configuration and markers

## 📝 Writing New Tests

### Unit Test Template
```python
import pytest
from lumen_clip.module import ClassToTest

class TestClassToTest:
    def test_method_with_valid_input(self):
        # Test normal operation
        instance = ClassToTest()
        result = instance.method(valid_input)
        assert result == expected_result

    def test_method_with_edge_case(self):
        # Test edge cases
        with pytest.raises(ExpectedException):
            instance = ClassToTest()
            instance.method(invalid_input)
```

### Using Configuration
```python
def test_with_real_config(mock_lumen_resources):
    # Load real configuration
    config_path = "tests/fixtures/configs/test_clip_torch.yaml"
    # Test with real config...
```

### Mocking External Dependencies
```python
def test_with_mock_backend(mock_onnx_session_class):
    # Test with mocked ONNX session
    backend = ONNXRuntimeBackend("fake_path.onnx")
    backend.initialize()
    # Test backend behavior...
```

## 🚨 Important Testing Notes

1. **Use Real Configurations**: Tests should use the same configuration system as the server
2. **Mock External Dependencies**: Use mocks for ONNX runtime, torch models, etc.
3. **Test Edge Cases**: Comprehensive testing of error conditions
4. **Validate Outputs**: Always validate that model outputs are reasonable
5. **Test Both Runtimes**: Ensure both Torch and ONNX work correctly
6. **Check for NaN/Inf**: Critical for detecting the classification issues we encountered

## 📚 Related Documentation

- [Lumen-CLIP README](../README.md)
- [Examples Configuration](../examples/config/)
- [lumen-resources Documentation](https://github.com/your-org/lumen-resources)