# Lumen-CLIP: Production-Ready CLIP Model Serving

A robust, production-grade gRPC service for serving CLIP and BioCLIP models with strict resource management, configuration-driven deployment, and multi-runtime support.

## Overview

Lumen-CLIP provides a unified interface for:
- **General CLIP**: Image classification (ImageNet), text/image embeddings, scene analysis
- **BioCLIP**: Species identification (TreeOfLife-10M), biological embeddings
- **Unified Service**: Smart routing between CLIP and BioCLIP based on context

### Key Features

- **Config-Driven**: All deployment controlled via YAML configuration
- **Strict Resource Management**: Uses `lumen-resources` for model/dataset validation
- **Multi-Runtime Support**: PyTorch, ONNX Runtime (RKNN planned)
- **All-or-None Principle**: Service refuses to start if required files are missing
- **Dynamic Capabilities**: Only exposes tasks supported by loaded resources
- **Production Ready**: Graceful shutdown, mDNS advertisement, comprehensive error handling

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      YAML Configuration                      │
│  (Service selection, models, runtime, datasets, etc.)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Server (server.py)                      │
│  • Config validation & single-service enforcement            │
│  • Dynamic service loading & gRPC registration               │
│  • mDNS advertisement & graceful shutdown                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Service Layer (from_config)                  │
│  CLIPService / BioCLIPService / UnifiedCLIPService           │
│  • Resource validation & loading                             │
│  • Backend initialization                                    │
│  • Dynamic task exposure                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Model Manager Layer                         │
│  CLIPModelManager / BioCLIPModelManager                      │
│  • Classification (if dataset available)                     │
│  • Embedding (always available)                              │
│  • Batching & preprocessing                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     Backend Layer                            │
│  TorchBackend / ONNXRTBackend / RKNNBackend                  │
│  • Model inference                                           │
│  • Device management                                         │
│  • Runtime-specific optimizations                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Resource Management                          │
│  ResourceLoader / ModelResources                             │
│  • Strict cache structure validation                         │
│  • Config/weights/tokenizer/dataset loading                  │
│  • Region-aware (ModelScope/HuggingFace)                     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- pip or uv (recommended)

### Install from source

```bash
# Clone the repository
git clone https://github.com/EdwinZhanCN/Lumen.git
cd Lumen/lumen-clip

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Install dependencies

```bash
# Core dependencies (automatically installed)
- torch / torchvision
- grpcio / grpcio-tools
- protobuf
- pyyaml
- pillow
- numpy

# Optional for ONNX Runtime
pip install onnxruntime  # CPU
pip install onnxruntime-gpu  # GPU

# Optional for mDNS
pip install zeroconf
```

## Quick Start

### 1. Prepare Resources

All models, configs, and datasets must be managed via `lumen-resources`:

```bash
# Download CLIP model (MobileCLIP2-S2) for torch runtime
lumen-resources download MobileCLIP2-S2 --runtime torch --region cn

# Download BioCLIP model with dataset
lumen-resources download bioclip-2 --runtime torch --dataset TreeOfLife-10M --region cn

# Validate resources
lumen-resources validate config/clip_only.yaml
```

Resources are stored in: `~/.lumen/models/{model_name}/{runtime}/`

Required files per model:
- `config.json` - Model configuration
- `model.pt` or `*.onnx` - Model weights
- `tokenizer.json` (optional, falls back to SimpleTokenizer)
- Dataset files (e.g., `ImageNet_1k.npz`, `TreeOfLife-10M.npy`) for classification

### 2. Configure Service

Choose a deployment scenario:

**Scenario 1: CLIP Only** (`config/clip_only.yaml`)
```yaml
metadata:
    version: "1.0"
    region: "cn"
    cache_dir: "~/.lumen/models"

services:
    clip:
        enabled: true
        package: "lumen-clip"
        import:
            registry_class: "image_classification.clip_service.CLIPService"
            add_to_server: "ml_service_pb2_grpc.add_InferenceServicer_to_server"
        models:
            default:
                model: "MobileCLIP2-S2"
                runtime: "torch"
        env:
            BATCH_SIZE: "8"
            DEVICE: "cuda"  # or "cpu"
        server:
            port: 50051
            mdns:
                enabled: true
                name: "CLIP-Service"
                type: "_lumen-clip._tcp.local."
```

**Scenario 2: BioCLIP Only** (`config/bioclip_only.yaml`)
```yaml
services:
    bioclip:
        enabled: true
        import:
            registry_class: "biological_atlas.bioclip_service.BioCLIPService"
            add_to_server: "ml_service_pb2_grpc.add_InferenceServicer_to_server"
        models:
            default:
                model: "bioclip-2"
                runtime: "torch"
                dataset: "TreeOfLife-10M"
        server:
            port: 50052
```

**Scenario 3: Unified Service** (`config/unified_service.yaml`)
```yaml
services:
    clip-unified:
        enabled: true
        import:
            registry_class: "service_registry.service.UnifiedCLIPService"
            add_to_server: "ml_service_pb2_grpc.add_InferenceServicer_to_server"
        models:
            clip_default:
                model: "MobileCLIP2-S2"
                runtime: "torch"
            bioclip_default:
                model: "bioclip-2"
                runtime: "torch"
                dataset: "TreeOfLife-10M"
        server:
            port: 50053
```

### 3. Start Service

```bash
# Start CLIP service
python src/server.py --config config/clip_only.yaml

# Start on custom port
python src/server.py --config config/clip_only.yaml --port 50060

# Enable debug logging
python src/server.py --config config/clip_only.yaml --log-level DEBUG
```

### 4. Test Service

```bash
# Using grpcurl
grpcurl -plaintext localhost:50051 inference.Inference/GetCapabilities

# Using Python client
python examples/client_test.py
```

## Service Types & Supported Tasks

### CLIP Service

**Always Available:**
- `embed`: Text → embedding vector
- `image_embed`: Image → embedding vector

**Available if ImageNet dataset present:**
- `classify`: Image → ImageNet class labels
- `classify_scene`: Image → scene category

### BioCLIP Service

**Always Available:**
- `embed`: Text → embedding vector
- `image_embed`: Image → embedding vector

**Available if TreeOfLife-10M dataset present:**
- `classify`: Image → species identification

### Unified Service

**Always Available:**
- `clip_embed`, `bioclip_embed`
- `clip_image_embed`, `bioclip_image_embed`

**Conditional:**
- `clip_classify`: If CLIP has ImageNet
- `bioclip_classify`: If BioCLIP has TreeOfLife-10M
- `smart_classify`: If both models have datasets (auto-routes between CLIP/BioCLIP)

## Multi-Service Deployment

**Important**: Only one service per process is allowed. For multi-service deployment, run multiple processes:

```bash
# Terminal 1: CLIP service
python src/server.py --config config/clip_only.yaml --port 50051

# Terminal 2: BioCLIP service
python src/server.py --config config/bioclip_only.yaml --port 50052

# Terminal 3: Unified service
python src/server.py --config config/unified_service.yaml --port 50053
```

Or use a process manager (systemd, supervisor, docker-compose):

```yaml
# docker-compose.yml
services:
  clip:
    image: lumen-clip:latest
    command: python src/server.py --config config/clip_only.yaml
    ports:
      - "50051:50051"
  
  bioclip:
    image: lumen-clip:latest
    command: python src/server.py --config config/bioclip_only.yaml
    ports:
      - "50052:50052"
```

## Runtime Selection

### PyTorch (Default)

```yaml
models:
    default:
        model: "MobileCLIP2-S2"
        runtime: "torch"
env:
    DEVICE: "cuda"  # or "cpu", "mps"
```

### ONNX Runtime

```yaml
models:
    default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
env:
    ONNX_PROVIDERS: "CUDAExecutionProvider,CPUExecutionProvider"
```

Required files:
- `image_encoder.onnx`
- `text_encoder.onnx`

### RKNN (Rockchip NPU)

```yaml
models:
    default:
        model: "MobileCLIP2-S2"
        runtime: "rknn"
env:
    RKNN_TARGET: "rk3588"
```

## Resource Management

### Cache Structure

```
~/.lumen/models/
├── MobileCLIP2-S2/
│   ├── torch/
│   │   ├── config.json          # Model configuration
│   │   ├── model.pt             # PyTorch weights
│   │   ├── tokenizer.json       # (optional)
│   │   └── ImageNet_1k.npz      # Classification dataset
│   └── onnx/
│       ├── config.json
│       ├── image_encoder.onnx
│       ├── text_encoder.onnx
│       └── ImageNet_1k.npz
└── bioclip-2/
    └── torch/
        ├── config.json
        ├── model.pt
        ├── tokenizer.json
        └── TreeOfLife-10M.npy
```

### Validation

```bash
# Validate config before deployment
lumen-resources validate config/clip_only.yaml

# Check specific model
lumen-resources check MobileCLIP2-S2 --runtime torch

# List available models
lumen-resources list
```

### Download from Different Regions

```yaml
metadata:
    region: "cn"    # Use ModelScope (China)
    # region: "other" # Use HuggingFace
```

## Configuration Reference

### Metadata Section

```yaml
metadata:
    version: "1.0"              # Config version
    region: "cn"                # Download region: "cn" or "other"
    cache_dir: "~/.lumen/models"  # Resource cache directory
```

### Service Section

```yaml
services:
    <service-name>:
        enabled: true/false     # Enable this service
        package: "lumen-clip"   # Package name
        import:
            registry_class: "module.path.ClassName"
            add_to_server: "module.path.function_name"
        models:
            default:            # Model identifier
                model: "model-name"
                runtime: "torch|onnx|rknn"
                dataset: "dataset-name"  # Optional
        env:                    # Environment variables
            BATCH_SIZE: "8"
            DEVICE: "cuda"
        server:
            port: 50051
            mdns:
                enabled: true
                name: "Service-Name"
                type: "_service._tcp.local."
```

## gRPC API

### Methods

```protobuf
service Inference {
    // Bidirectional streaming for inference tasks
    rpc Infer(stream InferRequest) returns (stream InferResponse);
    
    // Get service capabilities
    rpc GetCapabilities(google.protobuf.Empty) returns (CapabilityList);
    
    // Stream capabilities (for future expansion)
    rpc StreamCapabilities(google.protobuf.Empty) returns (stream Capability);
    
    // Health check
    rpc Health(google.protobuf.Empty) returns (HealthResponse);
}
```

### Request Format

```json
{
    "task_name": "classify",
    "request_id": "unique-id",
    "inputs": {
        "image": "<base64-encoded-image>"
    },
    "params": {
        "top_k": 5
    }
}
```

### Response Format

```json
{
    "request_id": "unique-id",
    "outputs": {
        "labels": [
            {"label": "golden_retriever", "score": 0.95},
            {"label": "labrador", "score": 0.03}
        ]
    },
    "metadata": {
        "model_id": "MobileCLIP2-S2",
        "processing_time_ms": 45
    }
}
```

## Development

### Project Structure

```
lumen-clip/
├── config/                      # Configuration examples
│   ├── clip_only.yaml
│   ├── bioclip_only.yaml
│   ├── unified_service.yaml
│   ├── clip_onnx.yaml
│   └── clip_rknn_rk3588.yaml
├── src/
│   ├── server.py               # Main server entry point
│   ├── backends/               # Runtime backends
│   │   ├── base.py
│   │   ├── torch_backend.py
│   │   ├── onnxrt_backend.py
│   │   └── rknn_backend.py
│   ├── resources/              # Resource management
│   │   ├── loader.py
│   │   └── exceptions.py
│   ├── image_classification/   # CLIP service
│   │   ├── clip_service.py
│   │   └── clip_model.py
│   ├── biological_atlas/       # BioCLIP service
│   │   ├── bioclip_service.py
│   │   └── bioclip_model.py
│   ├── service_registry/       # Unified service
│   │   └── service.py
│   └── proto/                  # gRPC definitions
│       └── ml_service.proto
└── examples/                   # Example clients
    └── client_test.py
```

### Adding a New Model

1. **Download resources via lumen-resources**
2. **Create config file**
3. **Validate**: `lumen-resources validate config/new_model.yaml`
4. **Test**: `python src/server.py --config config/new_model.yaml`

### Testing

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Load testing
python examples/load_test.py --config config/clip_only.yaml
```

## Troubleshooting

### Service refuses to start

**Error**: `ResourceNotFoundError: Model files not found`
- **Solution**: Run `lumen-resources download <model> --runtime <runtime>`

**Error**: `ConfigError: Multiple services enabled`
- **Solution**: Enable only one service per config file

### Classification task not available

**Error**: `Task 'classify' not supported`
- **Solution**: Download the required dataset:
  - CLIP: `lumen-resources download <model> --dataset ImageNet_1k`
  - BioCLIP: `lumen-resources download <model> --dataset TreeOfLife-10M`

### Model loading is slow

- Use ONNX runtime for faster inference
- Enable GPU: Set `DEVICE: cuda` in config
- Reduce batch size if memory limited

### mDNS not working

- Install zeroconf: `pip install zeroconf`
- Set `ADVERTISE_IP` environment variable to your LAN IP
- Check firewall settings

## Performance Tuning

### Batch Size

```yaml
env:
    BATCH_SIZE: "16"  # Increase for better throughput, decrease for lower latency
```

### Device Selection

```yaml
env:
    DEVICE: "cuda"     # Use GPU
    DEVICE: "cuda:0"   # Specific GPU
    DEVICE: "cpu"      # Force CPU
```

### ONNX Optimization

```yaml
env:
    ONNX_PROVIDERS: "CUDAExecutionProvider,CPUExecutionProvider"
    ONNX_INTRA_OP_THREADS: "4"
    ONNX_INTER_OP_THREADS: "2"
```

## Citations

If you use Lumen-CLIP in your research, please cite:

### OpenCLIP

```bibtex
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={ICML},
  year={2021}
}
```

### BioCLIP

```bibtex
@article{gu2025bioclip,
  title={BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning},
  author={Gu, Yuning and Van Horn, Grant and others},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}
```

## License

This project is part of the Lumen ecosystem and follows the same license terms. See LICENSE file for details.

## Support

- **Issues**: https://github.com/EdwinZhanCN/Lumen/issues
- **Documentation**: https://lumen-docs.example.com
- **Community**: [Discord/Slack link]

---

**Built with ❤️ for the Lumen ecosystem**