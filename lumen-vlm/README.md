# Lumen VLM

FastVLM multimodal vision-language understanding service for the Lumen ecosystem.

## Overview

Lumen VLM provides state-of-the-art vision-language model capabilities including:
- **Multimodal Understanding**: Process images and text together
- **Text Generation**: Generate coherent text responses from visual input
- **Streaming Support**: Real-time streaming generation for interactive applications
- **Chat Templates**: Support for conversation context and structured dialogues
- **Multiple Runtimes**: ONNX Runtime support with GPU/CPU acceleration

## Architecture

This implementation follows the Lumen development protocol with a clean 5-layer architecture:

```
Service Registry (协调层)
    ↓
Service (API层) - FastVLMService
    ↓
Model Manager (业务逻辑层) - FastVLMModelManager  
    ↓
Backend Implementation (具体实现) - FastVLMONNXBackend
    ↓
Base Backend (抽象层) - BaseFastVLMBackend
```

### Key Components

#### 1. **Base Backend (`src/lumen_vlm/backends/base.py`)**
- Abstract interface for all FastVLM runtime backends
- Defines standard data structures and protocols
- Provides utilities for chat templates and tokenization

#### 2. **ONNX Backend (`src/lumen_vlm/backends/onnxrt_backend.py`)**  
- Complete ONNX Runtime implementation
- Multi-GPU support with automatic provider selection
- Optimized inference for vision encoder + text decoder pipeline

#### 3. **Model Manager (`src/lumen_vlm/fastvlm/fastvlm_model.py`)**
- Business logic layer managing VLM model lifecycle
- Handles multimodal preprocessing and postprocessing
- Provides clean API for generation tasks with dependency injection

#### 4. **Service Layer (`src/lumen_vlm/fastvlm/fastvlm_service.py`)**
- gRPC service implementing the Lumen Inference protocol
- Task registry for capability reporting and routing
- Support for both streaming and non-streaming generation

#### 5. **Server (`src/lumen_vlm/server.py`)**
- Complete server implementation with configuration management
- Graceful shutdown and error handling
- Performance optimizations for production use

## Features

### ✅ Implemented
- **Complete Architecture**: Full 5-layer Lumen-compliant implementation
- **Dependency Injection**: Service-managed backend creation for testability
- **Multimodal Input**: Support for images (JPEG/PNG/WebP) + text
- **Generation Tasks**: Core `vlm_generate` and `vlm_generate_stream` tasks
- **Configuration**: YAML-based configuration with backend settings
- **Error Handling**: Comprehensive error handling with proper gRPC status codes
- **Resource Management**: Model loading, caching, and lifecycle management
- **Protocol Compliance**: Full Lumen Inference protocol implementation

### 🚀 Generation Capabilities
- **Chat Templates**: Support for conversation history and structured prompts
- **Streaming**: Real-time text generation for interactive applications  
- **Parameters**: Temperature, top_p, repetition_penalty, stop_sequences
- **Multimodal Fusion**: Integrated vision encoder + text decoder pipeline

### 🔧 Technical Features
- **Type Safety**: Full type annotations throughout the codebase
- **Logging**: Structured logging with colorized console output
- **Performance**: Optimized ONNX Runtime with multi-provider support
- **Extensibility**: Clean interfaces for adding new backends and tasks

## Quick Start

### Installation
```bash
# From the monorepo root
cd lumen-vlm
pip install -e .
```

### Configuration
Create a `config.yaml` file:
```yaml
cache_dir: "~/.cache/lumen"
services:
  vlm:
    models:
      fastvlm:
        model: "fastvlm-2b"
        runtime: "onnx"
    backend_settings:
      device: "cuda"  # or "cpu", "mps"
      max_new_tokens: 512
```

### Running the Service
```bash
# Start the VLM service
lumen-vlm --config config.yaml --port 50051

# Or with Python directly
python -m lumen_vlm.server config.yaml 50051
```

### Client Usage
```python
import grpc
from lumen_vlm.proto import ml_service_pb2, ml_service_pb2_grpc

# Connect to service
channel = grpc.insecure_channel('localhost:50051')
stub = ml_service_pb2_grpc.InferenceStub(channel)

# Prepare multimodal request
with open('image.jpg', 'rb') as f:
    image_data = f.read()

messages = [
    {"role": "user", "content": "What do you see in this image?"}
]

request = ml_service_pb2.InferRequest(
    task="vlm_generate",
    payload=image_data,
    payload_mime="image/jpeg",
    meta={
        "messages": json.dumps(messages),
        "max_new_tokens": "256",
        "temperature": "0.7"
    }
)

# Get response
response = stub.Infer(iter([request]))
for resp in response:
    if resp.is_final:
        result = json.loads(resp.result)
        print(f"Generated: {result['text']}")
```

## Architecture Benefits

### 🎯 **Dependency Injection**
- Service layer creates and manages Backend lifecycle
- Model Manager receives Backend interface, enabling easy testing
- Clean separation of concerns following SOLID principles

### 🔧 **Extensibility** 
- New backends (TensorRT, CoreML) can be added without modifying Model Manager
- Task Registry allows easy addition of new capabilities
- Protocol-based design supports diverse client implementations

### 🧪 **Testability**
- Interface-based design enables comprehensive unit testing
- Mock backends can be injected for isolated testing
- Clear boundaries between layers simplify test setup

### 📈 **Production Ready**
- Comprehensive error handling and logging
- Resource management and cleanup
- Performance optimizations and configuration options
- Graceful shutdown and signal handling

## Development

### Project Structure
```
src/lumen_vlm/
├── __init__.py                 # Package initialization and CLI entry point
├── cli.py                      # Command-line interface
├── server.py                   # Server startup and configuration
├── registry.py                 # Task registry for capability management
├── backends/                   # Backend abstraction layer
│   ├── base.py                 # Abstract backend interface
│   ├── backend_exceptions.py   # Backend-specific exceptions
│   └── onnxrt_backend.py       # ONNX Runtime implementation
├── fastvlm/                    # VLM-specific implementation
│   ├── fastvlm_model.py        # Model Manager (business logic)
│   └── fastvlm_service.py      # Service layer (API)
├── proto/                      # gRPC protocol definitions
└── resources/                  # Resource management
    ├── exceptions.py           # Resource-level exceptions
    └── loader.py               # Model resource loading
```

### Adding New Backends
1. Implement `BaseFastVLMBackend` interface
2. Add backend creation logic in `FastVLMService.from_config()`
3. Register supported runtime in configuration schema

### Adding New Tasks
1. Implement task handler method in `FastVLMService`
2. Register task in `_setup_task_registry()`
3. Update capability metadata and documentation

## Dependencies

### Core Dependencies
- `grpcio`: gRPC framework and protocol buffers
- `protobuf`: Protocol buffer implementation
- `numpy`: Numerical computing and array operations
- `pillow`: Image processing and format handling
- `jinja2`: Template engine for chat templates
- `tokenizers`: Hugging Face tokenizers library
- `colorlog`: Colored logging output

### Runtime Dependencies
- `onnxruntime`: CPU inference (default)
- `onnxruntime-gpu`: CUDA accelerated inference
- `lumen-resources`: Shared Lumen resource management

## License

This project is part of the Lumen ecosystem and follows the same licensing terms.