# Lumen Clip

`lumen-clip` provides text and image embedding services using CLIP (Contrastive Language-Image Pre-training) inside the Lumen AI inference server family. It follows the shared monorepo conventions (task registry, protobuf APIs, configuration-driven runtime) so it can be routed and orchestrated alongside other services.

## Overview

- **Service name:** `clip-general` (or `bioclip`, `smartclip` depending on config)
- **Package entry point:** `lumen_clip.general_clip.clip_service.GeneralCLIPService`, `lumen_clip.unified_smartclip.smartclip_service.SmartCLIPService`, `lumen_clip.expert_bioclip.bioclip_service.BioCLIPService`
- **gRPC interface:** `lumen_clip.proto.ml_service_pb2_grpc.InferenceServicer`
- **Task routing:** All requests are routed by keyword via the task registry—clients set `task` in `InferRequest` to select behavior.

## Backends

| Backend | Runtime | Notes |
|---------|---------|-------|
| `TorchBackend` | `torch` (CPU, MPS, CUDA) | Native PyTorch implementation. Supports dynamic batching and device selection (`cpu`, `mps`, `cuda`). |
| `ONNXRTBackend` | `onnxruntime` (CPU, CUDA, CoreML, DirectML, OpenVINO) | Optimized inference using ONNX Runtime. Provider priority is detected automatically, or you can pass `onnx_providers` in config to bias selection. |

## Keyword-Based Tasks

Tasks are registered in `TaskRegistry` and exposed through gRPC streaming inference. The available tasks depend on which service mode (`general`, `bioclip`, or `unified`) is active.

### General CLIP Tasks
Available when running `GeneralCLIPService`.

| Task keyword | Description | Input MIME | Output MIME |
|--------------|-------------|------------|-------------|
| `clip_text_embed` | Creates a vector embedding from a text string. | `application/json`, `text/plain` | `application/json;schema=embedding_v1` |
| `clip_image_embed` | Creates a vector embedding from an input image. | `image/jpeg`, `image/png`, `image/webp` | `application/json;schema=embedding_v1` |
| `clip_classify` | Zero-shot classification against the ImageNet dataset (requires dataset). | `image/jpeg`, `image/png` | `application/json;schema=labels_v1` |
| `clip_scene_classify` | Performs high-level scene analysis on an image. | `image/jpeg`, `image/png` | `application/json;schema=labels_v1` |

### BioCLIP Tasks
Available when running `BioCLIPService`.

| Task keyword | Description | Input MIME | Output MIME |
|--------------|-------------|------------|-------------|
| `bioclip_text_embed` | Creates a text embedding using the BioCLIP model. | `application/json`, `text/plain` | `application/json;schema=embedding_v1` |
| `bioclip_image_embed` | Creates an image embedding using the BioCLIP model. | `image/jpeg`, `image/png`, `image/webp` | `application/json;schema=embedding_v1` |
| `bioclip_classify` | Classifies an image using the TreeOfLife dataset (requires dataset). | `image/jpeg`, `image/png` | `application/json;schema=labels_v1` |

### Unified SmartCLIP Tasks
Available when running `SmartCLIPService` (loads both General and BioCLIP models).

| Task keyword | Description | Input MIME | Output MIME |
|--------------|-------------|------------|-------------|
| `smartclip_text_embed` | Creates text embedding using intelligent model selection. | `application/json`, `text/plain` | `application/json;schema=embedding_v1` |
| `smartclip_image_embed` | Creates image embedding using intelligent model selection. | `image/jpeg`, `image/png`, `image/webp` | `application/json;schema=embedding_v1` |
| `smartclip_classify` | Intelligent classification using best available model. | `image/jpeg`, `image/png` | `application/json;schema=labels_v1` |
| `smartclip_scene_classify` | Scene analysis using the general CLIP model. | `image/jpeg`, `image/png` | `application/json;schema=labels_v1` |
| `smartclip_bioclassify` | Biological classification using the BioCLIP model. | `image/jpeg`, `image/png` | `application/json;schema=labels_v1` |

Each task returns a protobuf-defined schema serialized as JSON and includes per-request metadata such as processing time, embedding dimension, or model version.

## Configuration & Models

Service configuration is provided through the shared Lumen YAML schema (see `examples/config/clip_cn.yaml`):

```yaml
metadata:
  region: "cn"
  cache_dir: "~/.lumen"

deployment:
  mode: "single"
  service: "clip"

server:
  port: 50051
  mdns:
    enabled: true
    service_name: "lumen-clip"

services:
  clip:
    enabled: true
    package: lumen_clip
    import:
      registry_class: "lumen_clip.general_clip.clip_service.GeneralCLIPService"
      add_to_server: "lumen_clip.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server"
    backend_settings:
      device: "mps"         # or cpu/cuda
      batch_size: 8
    models:
      # Define 'general' for GeneralCLIPService
      # Supported keys: "general", "clip", "general_clip"
      general:
        model: "CN-CLIP_ViT-B-16"
        runtime: torch
        dataset: ImageNet_1k # Optional, enables classification tasks

      # Define 'bioclip' for BioCLIPService
      # Define BOTH for SmartCLIPService
      # Supported keys: "bioclip", "bio", "bioclip2"
      # bioclip:
      #   model: "bioclip-2"
      #   runtime: torch
      #   dataset: TreeOfLife-200M
```

### Supported Model Configuration Keys

The service automatically selects which service class to instantiate based on the model keys present in `services.clip.models`:

| Service Type | Supported Keys | Description |
|--------------|----------------|-------------|
| **GeneralCLIPService** | `general`, `clip`, `general_clip` | Loads a general-purpose CLIP model for text/image embeddings and ImageNet classification |
| **BioCLIPService** | `bioclip`, `bio`, `bioclip2` | Loads BioCLIP for biological and scientific image analysis |
| **SmartCLIPService** | Both general AND bioclip keys | Unified service with both models loaded, enabling intelligent model selection |

Key points:

1. **Model caching** is handled by `lumen-resources`: it downloads model weights, tokenizers, and optional datasets (like `ImageNet_1k.npz` or `TreeOfLife`).
2. **Service Selection**: The server automatically instantiates `GeneralCLIPService`, `BioCLIPService`, or `SmartCLIPService` based on which model keys are defined in the config.
3. **Backend settings** allow you to bias provider selection (`device`) and tweak batch sizes for high-throughput scenarios.

## Supported Models

| Model ID | Type | Embedding Dim | Notes |
|----------|------|---------------|-------|
| `CN-CLIP_ViT-B-16` | General | 512 | Chinese-English bilingual CLIP, ViT-B/16 backbone. |
| `CN-CLIP_ViT-L-14` | General | 768 | Larger bilingual model, higher accuracy. |
| `MobileCLIP2-S2` | General | 512 | Efficient mobile-optimized CLIP model. |
| `MobileCLIP2-S4` | General | 768 | Robust mobile-optimized CLIP model. |
| `bioclip-2` | Expert | 768 | Specialized for biology/nature (requires `bioclip` config). |

All models support both text and image inputs. The `dataset` configuration option enables zero-shot classification tasks if the corresponding dataset artifacts are present.

## Running the Server

```bash
uv run python -m lumen_clip.server \
  --config lumen-clip/examples/config/clip_cn.yaml \
  --log-level INFO
```

The runner will:

1. Validate and download resources via `lumen-resources`.
2. Instantiate the appropriate service (`GeneralCLIPService`, `BioCLIPService`, or `SmartCLIPService`) based on config.
3. Initialize the backend (`TorchBackend` or `ONNXRTBackend`).
4. Start the gRPC server (and optional mDNS advertisement).

You can override the port with `--port` and dynamically scale logging via `--log-level`.

---

For shared contributor guidelines, lint rules, and deployment workflows, refer to the root `README.md` and `docs/` in the Lumen monorepo.
