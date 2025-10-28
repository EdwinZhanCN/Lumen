# Lumen Resources

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://doc.lumilio.org)

**Lumen Resources** is a unified model resource management tool for Lumen ML services. It provides configuration-driven downloading and validation of ML model resources from platforms like Hugging Face and ModelScope, with support for multiple runtimes and deployment modes. The tool ensures production-grade reliability through YAML configuration, JSON Schema validation, and Pydantic models.

## 🚀 Features

### Core Capabilities
- **Multi-Platform Support**: Unified interface for Hugging Face Hub and ModelScope
- **Runtime Flexibility**: Support for PyTorch, ONNX, RKNN, and custom runtimes
- **Configuration-Driven**: Declarative YAML configuration for reproducible deployments
- **Production-Grade Validation**: JSON Schema and Pydantic model validation
- **Efficient Downloads**: Pattern-based file filtering to minimize bandwidth usage
- **Automatic Caching**: Intelligent model caching with resume capability

### Platform Integration
- **Hugging Face Hub**: Full integration with the world's largest model repository
- **ModelScope**: China-based model platform with optimized access for Chinese users
- **Automatic Platform Selection**: Region-based platform selection (cn=ModelScope, other=HuggingFace)

### Runtime Support
- **PyTorch**: Native PyTorch model loading and inference
- **ONNX**: Optimized cross-platform inference with multiple execution providers
- **RKNN**: Rockchip NPU acceleration for edge devices
- **Custom**: Extensible runtime system for specialized hardware

## 📦 Installation

### Standard Installation

Install the package from the Lumen repository:

```bash
pip install lumen-resources @ git+https://github.com/EdwinZhanCN/Lumen.git@main#subdirectory=lumen-resources
```

### Development Installation

For development or contributions, clone the repository and install with optional dependencies:

```bash
git clone https://github.com/EdwinZhanCN/Lumen.git
cd lumen-resources
pip install -e ".[dev,config]"
```

### Platform Dependencies

Depending on your target platforms, install the required SDKs:

```bash
# For Hugging Face Hub
pip install huggingface_hub

# For ModelScope (recommended for China region)
pip install modelscope

# For ONNX Runtime
pip install onnxruntime

# For PyTorch
pip install torch torchvision
```

## 🎯 Quick Start

### CLI Usage

The CLI provides commands for downloading, validating, and listing model resources:

```bash
# Download models from configuration
lumen-resources download config.yaml

# Validate configuration file
lumen-resources validate config.yaml

# List cached models
lumen-resources list ~/.lumen/
```

### Python API Usage

Import and use the package programmatically:

```python
from lumen_resources import load_and_validate_config, Downloader

# Load and validate configuration
config = load_and_validate_config("config.yaml")

# Initialize downloader
downloader = Downloader(config, verbose=True)

# Download all enabled models
results = downloader.download_all(force=False)

# Check results
for model_type, result in results.items():
    if result.success:
        print(f"✅ Downloaded: {model_type} to {result.model_path}")
    else:
        print(f"❌ Failed: {model_type} - {result.error}")
```

## 📖 Detailed Usage

### Configuration Format

Lumen Resources uses YAML configuration files to define model deployments. Here's a comprehensive example:

```yaml
# Lumen Services Configuration
metadata:
  version: "1.0.0"
  region: "other"  # "cn" for ModelScope, "other" for HuggingFace
  cache_dir: "~/.lumen/models"

deployment:
  mode: "single"
  service: "clip"

server:
  port: 50051
  host: "0.0.0.0"
  mdns:
    enabled: true
    service_name: "lumen-clip"

services:
  clip:
    enabled: true
    package: "lumen_clip"
    import:
      registry_class: "lumen_clip.service_registry.ClipService"
      add_to_server: "lumen_clip.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server"

    backend_settings:
      device: "cuda"  # or "mps", "cpu"
      batch_size: 16
      onnx_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]

    models:
      default:
        model: "ViT-B-32"
        runtime: "torch"
        dataset: "ImageNet_1k"

      cn_clip:
        model: "CN-CLIP-ViT-B-16"
        runtime: "torch"
        dataset: "ImageNet_1k"

      onnx_version:
        model: "ViT-B-32"
        runtime: "onnx"
        dataset: "ImageNet_1k"
```

### Configuration Sections

#### Metadata
- `version`: Configuration version (semantic versioning)
- `region`: Platform region selection (`cn` for ModelScope, `other` for HuggingFace)
- `cache_dir`: Local directory for model caching (supports `~` expansion)

#### Deployment
- `mode`: Deployment mode (`single` or `hub`)
- `service`: Service name for single mode
- `services`: List of services for hub mode

#### Server
- `port`: gRPC server port (1024-65535)
- `host`: Server bind address
- `mdns`: Optional mDNS service discovery configuration

#### Services
Each service contains:
- `enabled`: Whether the service should be loaded
- `package`: Python package name
- `import`: Dynamic import configuration
- `backend_settings`: Optional runtime optimization settings
- `models`: Dictionary of model configurations

#### Model Configuration
Each model contains:
- `model`: Repository name/identifier
- `runtime`: Runtime type (`torch`, `onnx`, `rknn`)
- `rknn_device`: Required device identifier for RKNN runtime
- `dataset`: Optional dataset for zero-shot classification

### CLI Commands Reference

#### Download Command

```bash
lumen-resources download <config> [--force]
```

Downloads all enabled models from the specified configuration file.

**Arguments:**
- `config`: Path to configuration YAML file

**Options:**
- `--force`: Force re-download even if models are already cached

**Example:**
```bash
lumen-resources download config.yaml --force
```

#### Validate Command

```bash
lumen-resources validate <config> [--strict | --schema-only]
```

Validates configuration files against the schema.

**Arguments:**
- `config`: Path to configuration YAML file

**Options:**
- `--strict`: Use strict Pydantic validation (default)
- `--schema-only`: Use JSON Schema validation only (less strict)

**Example:**
```bash
lumen-resources validate config.yaml --strict
```

#### Validate Model Info Command

```bash
lumen-resources validate-model-info <model_info> [--strict | --schema-only]
```

Validates `model_info.json` files.

**Arguments:**
- `model_info`: Path to model_info.json file

**Options:**
- `--strict`: Use strict Pydantic validation (default)
- `--schema-only`: Use JSON Schema validation only (less strict)

**Example:**
```bash
lumen-resources validate-model-info model_info.json
```

#### List Command

```bash
lumen-resources list [cache_dir]
```

Lists all models in the cache directory.

**Arguments:**
- `cache_dir`: Cache directory path (default: `~/.lumen/`)

**Example:**
```bash
lumen-resources list ~/.lumen/models
```

### Python API Reference

#### Core Classes

##### `LumenConfig`
Root configuration model for Lumen services.

```python
from lumen_resources import LumenConfig, Runtime, Region

config = LumenConfig(
    metadata={
        "version": "1.0.0",
        "region": Region.other,
        "cache_dir": "~/.lumen/models"
    },
    deployment=Deployment(mode="single", service="clip"),
    server=Server(port=50051),
    services={
        "clip": Services(
            enabled=True,
            package="lumen_clip",
            import_=Import(
                registry_class="lumen_clip.service_registry.ClipService",
                add_to_server="lumen_clip.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server"
            ),
            models={
                "default": ModelConfig(
                    model="ViT-B-32",
                    runtime=Runtime.torch
                )
            }
        )
    }
)
```

##### `Downloader`
Main resource downloader with validation and caching.

```python
from lumen_resources import Downloader

downloader = Downloader(config, verbose=True)
results = downloader.download_all(force=False)

for model_type, result in results.items():
    if result.success:
        print(f"✅ {model_type}: {result.model_path}")
    else:
        print(f"❌ {model_type}: {result.error}")
```

##### `DownloadResult`
Result object containing download status and information.

```python
result = DownloadResult(
    model_type="clip:default",
    model_name="ViT-B-32",
    runtime="torch",
    success=True,
    model_path=Path("/models/clip_vit_b32")
)

print(f"Success: {result.success}")
print(f"Path: {result.model_path}")
```

#### Utility Functions

##### `load_and_validate_config()`
Load and validate a configuration file.

```python
from lumen_resources import load_and_validate_config

config = load_and_validate_config("config.yaml")
print(config.metadata.version)
```

##### `load_and_validate_model_info()`
Load and validate a model_info.json file.

```python
from lumen_resources import load_and_validate_model_info

model_info = load_and_validate_model_info("model_info.json")
print(model_info.name)
print(model_info.version)
```

## 🔧 Advanced Configuration

### Multi-Service Hub Configuration

```yaml
metadata:
  version: "1.0.0"
  region: "other"
  cache_dir: "~/.lumen/models"

deployment:
  mode: "hub"
  services: ["clip", "face"]

services:
  clip:
    enabled: true
    package: "lumen_clip"
    # ... clip service configuration

  face:
    enabled: true
    package: "lumen_face"
    # ... face service configuration
```

### Backend Optimization

```yaml
services:
  clip:
    backend_settings:
      device: "cuda"           # Preferred device
      batch_size: 32          # Maximum batch size
      onnx_providers:         # ONNX execution providers
        - "CUDAExecutionProvider"
        - "CPUExecutionProvider"
```

### Platform-Specific Configuration

For Chinese users, configure ModelScope:

```yaml
metadata:
  region: "cn"  # Uses ModelScope platform
  cache_dir: "~/.lumen/models"
```

For international users, configure HuggingFace:

```yaml
metadata:
  region: "other"  # Uses HuggingFace Hub
  cache_dir: "~/.lumen/models"
```

## 🏗️ Architecture

### System Design

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Layer     │    │   Python API     │    │  Validation     │
│                 │    │                  │    │                 │
│ • download      │    │ • Downloader     │    │ • JSON Schema   │
│ • validate      │    │ • Config Models  │    │ • Pydantic      │
│ • list          │    │ • Utilities      │    │ • Custom Rules  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                    Platform Layer                                 │
│                                                                 │
│ • HuggingFace Hub    • ModelScope    • Unified Interface         │
│ • Pattern Filtering  • Caching       • Error Handling            │
└─────────────────────────────────┼─────────────────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     Model Repositories     │
                    │                           │
                    │  HuggingFace  ModelScope   │
                    └───────────────────────────┘
```

### Component Interaction

1. **Configuration Loading**: YAML files are parsed and validated using JSON Schema and Pydantic
2. **Platform Selection**: Based on region configuration, appropriate platform adapter is selected
3. **Pattern-Based Downloads**: Only required files are downloaded using efficient filtering
4. **Validation**: Downloaded models are validated against their metadata
5. **Caching**: Models are cached locally with integrity verification

## 🚨 Error Handling

The package provides comprehensive error handling with specific exception types:

```python
from lumen_resources import (
    ConfigError, DownloadError, ValidationError,
    ModelInfoError, PlatformUnavailableError
)

try:
    config = load_and_validate_config("config.yaml")
except ConfigError as e:
    print(f"Configuration error: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")

try:
    downloader = Downloader(config)
    results = downloader.download_all()
except DownloadError as e:
    print(f"Download error: {e}")
except PlatformUnavailableError as e:
    print(f"Platform unavailable: {e}")
```

## 📝 Model Information Format

Each model includes a `model_info.json` file with metadata:

```json
{
  "name": "ViT-B-32",
  "version": "1.0.0",
  "description": "Vision Transformer for CLIP",
  "model_type": "vision-transformer",
  "embedding_dim": 512,
  "source": {
    "format": "huggingface",
    "repo_id": "openai/clip-vit-base-patch32"
  },
  "runtimes": {
    "torch": {
      "available": true,
      "files": ["pytorch_model.bin", "config.json"],
      "devices": ["cuda", "cpu"]
    },
    "onnx": {
      "available": true,
      "files": ["model.onnx", "config.json"],
      "devices": ["cpu", "cuda"]
    }
  },
  "datasets": {
    "ImageNet_1k": {
      "labels": "imagenet_labels.txt",
      "embeddings": "imagenet_embeddings.npy"
    }
  },
  "metadata": {
    "license": "MIT",
    "author": "OpenAI",
    "tags": ["computer-vision", "multimodal", "clip"]
  }
}
```

## 🔍 Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: `PlatformUnavailableError: HuggingFace Hub SDK not available`
**Solution**: Install the required SDK:
```bash
pip install huggingface_hub
```

**Issue**: `ValidationError: Configuration validation failed`
**Solution**: Check your YAML syntax and required fields:
```bash
lumen-resources validate config.yaml --strict
```

#### Download Issues

**Issue**: `DownloadError: Failed to download model`
**Solution**: Check network connectivity and repository availability:
```bash
# Test with verbose output
lumen-resources download config.yaml --force
```

**Issue**: Missing model files after download
**Solution**: Verify model_info.json exists and is valid:
```bash
lumen-resources validate-model-info path/to/model_info.json
```

#### Configuration Issues

**Issue**: Models not downloading
**Solution**: Ensure models are enabled in configuration:
```yaml
services:
  your_service:
    enabled: true  # Must be true
    models:
      your_model:  # Model must be defined
        model: "repo/name"
        runtime: "torch"
```

### Performance Optimization

#### Cache Management
- Set appropriate `cache_dir` with sufficient disk space
- Use `--force` flag sparingly to avoid unnecessary downloads
- Monitor cache size and clean up unused models

#### Download Optimization
- Configure appropriate `allow_patterns` to download only required files
- Use region-appropriate platforms (ModelScope for China)
- Enable parallel downloads where supported

#### Runtime Optimization
- Configure appropriate `batch_size` for your hardware
- Select optimal `device` (cuda/mps/cpu) based on availability
- Configure ONNX providers for best performance

## 🤝 Contributing

We welcome contributions to Lumen Resources! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/EdwinZhanCN/Lumen.git
cd lumen-resources
```

2. Install in development mode:
```bash
pip install -e ".[dev,config]"
```

3. Run tests:
```bash
pytest tests/
```

4. Run code formatting:
```bash
black src/
isort src/
```

### Adding New Platforms

To add support for a new model platform:

1. Create a new platform adapter in `platform.py`
2. Add platform type to `PlatformType` enum
3. Implement download and cleanup methods
4. Add platform-specific tests
5. Update documentation

### Adding New Runtimes

To add support for a new runtime:

1. Add runtime to `Runtime` enum in `lumen_config.py`
2. Update file patterns in `downloader.py`
3. Add runtime-specific validation
4. Update model_info schema
5. Add documentation and examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the amazing model hub and libraries
- [ModelScope](https://modelscope.cn/) for providing access to models in China
- [Pydantic](https://pydantic-docs.helpmanual.io/) for excellent data validation
- The open-source community for inspiration and feedback

## 📚 Additional Resources

- [Lumen Documentation](https://doc.lumilio.org/)
- [Configuration Schema Reference](https://doc.lumilio.org/schema/config-schema.yaml)
- [Model Information Schema](https://doc.lumilio.org/schema/model_info-schema.json)
- [GitHub Repository](https://github.com/EdwinZhanCN/Lumen)
- [Issue Tracker](https://github.com/EdwinZhanCN/Lumen/issues)

---

**Lumen Resources** - Making model resource management simple, reliable, and efficient. 🚀
