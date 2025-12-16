Lumen/lumen-vlm/README.md
# Lumen VLM

`lumen-vlm` provides vision-language model services inside the Lumen AI inference server family. It follows the shared monorepo conventions (task registry, protobuf APIs, configuration-driven runtime) so it can be routed and orchestrated alongside other services.

## Overview

- **Service name:** `vlm-fast`
- **Package entry point:** `lumen_vlm.fastvlm.GeneralFastVLMService`
- **gRPC interface:** `lumen_vlm.proto.ml_service_pb2_grpc.InferenceServicer`
- **Task routing:** All requests are routed by keyword via the task registry—clients set `task` in `InferRequest` to select behavior.

## Backends

| Backend | Runtime | Notes |
|---------|---------|-------|
| `ONNXRTBackend` | `onnxruntime` (CPU, CUDA, CoreML, DirectML, OpenVINO) | Loads FastVLM-style models (vision encoder, text embedder, causal decoder). Provider priority is detected automatically, or you can pass `onnx_providers` in config to bias selection. |

## Keyword-Based Tasks

Tasks are registered in `TaskRegistry` and exposed through gRPC streaming inference. The available tasks depend on the active service mode.

### VLM Generation Tasks
Available when running `GeneralFastVLMService`.

| Task keyword | Description | Input MIME | Output MIME |
|--------------|-------------|------------|-------------|
| `vlm_generate` | Generate text from image and text input. | `image/jpeg`, `image/png`, `image/webp` | `application/json;schema=text_generation_v1` |
| `vlm_generate_stream` | Generate text from image and text input with streaming output. | `image/jpeg`, `image/png`, `image/webp` | `application/json;schema=text_generation_v1` |

Each task returns a protobuf-defined schema serialized as JSON and includes per-request metadata such as processing time, generated tokens, or finish reason.

## Configuration & Models

Service configuration is provided through the shared Lumen YAML schema (see `examples/config/vlm_cn.yaml`):

```yaml
metadata:
  region: "cn"
  cache_dir: "~/.lumen"

deployment:
  mode: "single"
  service: "vlm"

server:
  port: 50051
  mdns:
    enabled: true
    service_name: "lumen-vlm"

services:
  vlm:
    enabled: true
    package: lumen_vlm
    import:
      registry_class: "lumen_vlm.fastvlm.GeneralFastVLMService"
      add_to_server: "lumen_vlm.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server"
    backend_settings:
      device: null          # auto-detect (cpu/cuda/coreml/...)
      max_new_tokens: 512
      prefer_fp16: true
    models:
      general:
        model: "FastVLM-0.5B"
        runtime: onnx
```

Key points:

1. **Model caching** is handled by `lumen-resources`: it downloads model weights, tokenizers, and metadata.
2. **Service Selection**: The server automatically instantiates `GeneralFastVLMService` based on the config.
3. **Backend settings** allow you to bias provider selection (`device`), set generation limits (`max_new_tokens`), and enable optimizations (`prefer_fp16`).

## Supported Models

| Model ID | Type | Embedding Dim | Notes |
|----------|------|---------------|-------|
| `FastVLM-0.5B` | General | N/A | Efficient vision-language model for multimodal understanding and generation. |

All models support image inputs (JPEG, PNG, WebP) and text generation with configurable parameters like temperature, top_p, and repetition penalty.

## Running the Server

```bash
lumen-vlm --config lumen-vlm/examples/config/vlm_cn.yaml --log-level INFO
```

The runner will:

1. Validate and download resources via `lumen-resources`.
2. Instantiate `GeneralFastVLMService` with the configured model/runtime.
3. Initialize `ONNXRTBackend`.
4. Start the gRPC server (and optional mDNS advertisement).

You can override the port with `--port` and dynamically scale logging via `--log-level`.

---

For shared contributor guidelines, lint rules, and deployment workflows, refer to the root `README.md` and `docs/` in the Lumen monorepo.
