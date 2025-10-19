# Lumen CLIP

A Python Machine Learning (PML) module for Lumilio Photos, providing CLIP-based image understanding capabilities through gRPC services.

## Overview

Lumen CLIP is a modular machine learning service that provides:

- **General CLIP** - Standard image-text understanding
- **BioCLIP** - Specialized for biological and scientific imagery
- **Unified SmartCLIP** - Multi-modal intelligent image analysis

## Features

- 🚀 High-performance inference with multiple backend support (CPU, CUDA, MPS, OpenVINO)
- 🔧 Modular architecture with specialized model managers
- 📡 gRPC-based service interface for easy integration
- 🎯 Type-safe configuration with Pydantic models
- 📦 Automatic model downloading and management
- 🔍 mDNS service discovery for distributed deployments

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.1.0+
- ONNX Runtime 1.16.0+

### Basic Installation

```bash
pip install lumen-clip
```

### Platform-Specific Installation

Choose the appropriate installation for your hardware:

```bash
# CPU-only
pip install lumen-clip[cpu]

# CUDA (NVIDIA GPU)
pip install lumen-clip[cuda]

# MPS (Apple Silicon)
pip install lumen-clip[mps]

# OpenVINO (Intel)
pip install lumen-clip[openvino2404]
```

## Quick Start

### Starting the Service

```python
from lumen_clip.server import serve

# Start the gRPC server
serve(port=50051, host="0.0.0.0")
```

Or from command line:

```bash
python -m lumen_clip.server
```

### Client Usage

```python
import grpc
import lumen_clip.proto.ml_service_pb2 as pb
import lumen_clip.proto.ml_service_pb2_grpc as pb_grpc

# Connect to the service
channel = grpc.insecure_channel('localhost:50051')
stub = pb_grpc.InferenceStub(channel)

# Create inference request
request = pb.InferRequest(
    correlation_id="test-001",
    task="embed",
    payload=image_bytes,
    payload_mime="image/jpeg"
)

# Stream inference
for response in stub.Infer(iter([request])):
    print(f"Result: {response.result}")
```

## Configuration

Lumen CLIP uses `lumen-resources` for configuration management. Create a configuration file:

```yaml
# Lumen Services Configuration - Single Service Mode
# yaml-language-server: $schema=https://doc.lumilio.org/schema/config-schema.yaml

metadata:
    version: "1.0.0"
    region: "cn" # "cn" for ModelScope, "other" for HuggingFace
    cache_dir: "~/Lumen-Resources"

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
    # CLIP Service
    clip:
        enabled: true
        package: "lumen_clip"
        import:
            registry_class: "lumen_clip.service_registry.CLIPService"
            add_to_server: "lumen_clip.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server"
        backend_settings:
            device: "mps"
            onnx_providers:
                - "CoreMLExecutionProvider"
            batch_size: 8
        models:
            general:
                model: "MobileCLIP2-S2"
                runtime: onnx
                dataset: ImageNet_1k
```

## Service Architecture

### Core Components

- **GeneralCLIPService** - Standard CLIP embeddings and similarity
- **BioCLIPService** - Biological image analysis and classification
- **UnifiedCLIPService** - Multi-modal intelligent analysis
- **ResourceLoader** - Model loading and management
- **gRPC Server** - High-performance inference API

### Backend Support

- **PyTorch** - Primary inference engine
- **ONNX Runtime** - Optimized execution
- **Multiple Hardware** - CPU, CUDA, MPS, OpenVINO

## API Reference

### Inference Service

The service provides a bidirectional gRPC stream for inference:

```protobuf
service Inference {
  rpc Infer(stream InferRequest) returns (stream InferResponse);
  rpc GetCapabilities(Empty) returns (Capability);
  rpc StreamCapabilities(Empty) returns (stream Capability);
  rpc Health(Empty) returns (Empty);
}
```

### Supported Tasks

- `embed` - Generate image/text embeddings
- `classify` - Image classification
- `similarity` - Text-image similarity scoring
- Custom expert tasks for specialized domains

## Development

### Project Structure

```
src/lumen_clip/
├── general_clip/      # Standard CLIP implementation
├── expert_bioclip/    # Biological CLIP specialization
├── unified_smartclip/ # Multi-modal analysis
├── resources/         # Model loading utilities
├── proto/            # gRPC service definitions
└── backends/         # Hardware backend implementations
```

### Building from Source

```bash
git clone https://github.com/EdwinZhanCN/Lumen.git
cd Lumen/lumen-clip
pip install -e .
```

### Running Tests

```bash
python -m pytest test/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is part of the Lumilio ecosystem. See the main repository for license information.

## Support

- 📖 Documentation: [Lumilio Docs](https://docs.lumilio.org)
- 🐛 Issues: [GitHub Issues](https://github.com/EdwinZhanCN/Lumen/issues)
