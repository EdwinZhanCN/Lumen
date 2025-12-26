# Lumen OCR

`lumen-ocr` provides optical character recognition services inside the Lumen AI inference server family. It follows the shared monorepo conventions (task registry, protobuf APIs, configuration-driven runtime) so it can be routed and orchestrated alongside other services.

## Overview

- **Service name:** `ocr-general`
- **Package entry point:** `lumen_ocr.general_ocr.GeneralOcrService`
- **gRPC interface:** `lumen_ocr.proto.ml_service_pb2_grpc.InferenceServicer`
- **Task routing:** All requests are routed by keyword via the task registry—clients set `task` in `InferRequest` to select behavior.

## Backends

| Backend | Runtime | Notes |
|---------|---------|-------|
| `ONNXRTBackend` | `onnxruntime` (CPU, CUDA, CoreML, DirectML, OpenVINO) | Loads PaddleOCR-style models (DBNet + SVTR/CRNN). Provider priority is detected automatically, or you can pass `device` in config to bias selection. |

> Additional runtimes (RKNN, Torch, etc.) follow the same backend contract and can be added in the future through `BaseOcrBackend`.

### macOS / CoreML Note

On macOS, `onnxruntime` may default to `CoreMLExecutionProvider`. If you encounter warnings like `Context leak detected, CoreAnalytics returned false` or stability issues with dynamic shapes in OCR models, it is recommended to force the CPU provider in your configuration:

```yaml
backend_settings:
    device: cpu # auto-detect
    onnx_providers:
        - CPUExecutionProvider
```

## Keyword-Based Tasks

Registered in `TaskRegistry`, exposed through gRPC streaming inference:

| Task keyword | Handler | Description | Input MIME | Output MIME |
|--------------|---------|-------------|------------|-------------|
| `ocr` | `_handle_ocr` | End-to-end text detection and recognition. Returns bounding boxes (polygon), text content, and confidence scores. Thresholds (`det_thresh`, `rec_thresh`, `box_thresh`, `unclip_ratio`) come from request metadata. | `image/jpeg`, `image/png`, `image/bmp`, `image/webp` | `application/json;schema=ocr_v1` |

Each task returns a protobuf-defined schema serialized as JSON and includes per-request metadata such as processing time.

## Configuration & Models

Service configuration is provided through the shared Lumen YAML schema (see `examples/config/ocr.yaml`):

```yaml
metadata:
  region: "cn"
  cache_dir: "~/.lumen"

deployment:
  mode: "single"
  service: "ocr"

server:
  port: 50051
  mdns:
    enabled: true
    service_name: "lumen-ocr"

services:
  ocr:
    enabled: true
    package: lumen_ocr
    import:
      registry_class: "lumen_ocr.general_ocr.GeneralOcrService"
      add_to_server: "lumen_ocr.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server"
    backend_settings:
      device: null          # auto-detect (cpu/cuda/coreml/...)
    models:
      # Supported keys: "general", "ocr", "ppocr"
      general:
        model: "PP-OCRv5"   # or other supported model IDs
        runtime: onnx
```

### Supported Model Configuration Keys

| Service Type | Supported Keys | Description |
|--------------|----------------|-------------|
| **GeneralOcrService** | `general`, `ocr`, `ppocr` | Loads a PaddleOCR-style model for text detection and recognition |

Key points:

1. **Model caching** is handled by `lumen-resources`: it downloads `model_info.json`, ONNX files (`detection.fp32.onnx`, `recognition.fp32.onnx`), and dictionary files (`ppocr_keys_v1.txt`).
2. **Runtime metadata** (preprocessing stats, image shapes, thresholds) is loaded from `model_info.json`'s `extra_metadata` field.
3. **Backend settings** allow you to bias provider selection (`device`).

## Supported Models

| Model ID | Detection | Recognition | Notes |
|----------|-----------|-------------|-------|
| `PP-OCRv5` | DBNet (Mobile), `detection.fp32.onnx` | SVTR_LCNet (Mobile), `recognition.fp32.onnx` | Default general model (Chinese/English) |

Models typically use standard PaddleOCR preprocessing (Resize, Normalize) and post-processing (DB unclip, CTC decode). Configuration for specific models (e.g., input shapes, mean/std) is defined in the model's metadata.

## Running the Server

```bash
lumen-ocr --config lumen-ocr/examples/config/ocr.yaml --log-level INFO
```

The runner will:

1. Validate and download resources via `lumen-resources`.
2. Instantiate `GeneralOcrService` with the configured model/runtime.
3. Initialize `ONNXRTBackend`.
4. Start the gRPC server (and optional mDNS advertisement).

You can override the port with `--port` and dynamically scale logging via `--log-level`.

---

For shared contributor guidelines, lint rules, and deployment workflows, refer to the root `README.md` and `docs/` in the Lumen monorepo.
