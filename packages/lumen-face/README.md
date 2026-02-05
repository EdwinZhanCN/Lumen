# Lumen Face

`lumen-face` provides face detection and recognition services inside the Lumen AI inference server family. It follows the shared monorepo conventions (task registry, protobuf APIs, configuration-driven runtime) so it can be routed and orchestrated alongside other services.

## Overview

- **Service name:** `face-general`
- **Package entry point:** `lumen_face.general_face.GeneralFaceService`
- **gRPC interface:** `lumen_face.proto.ml_service_pb2_grpc.InferenceServicer`
- **Task routing:** All requests are routed by keyword via the task registry—clients set `task` in `InferRequest` to select behavior.

## Backends

| Backend | Runtime | Notes |
|---------|---------|-------|
| `ONNXRTBackend` | `onnxruntime` (CPU, CUDA, CoreML, DirectML, OpenVINO) | Loads InsightFace-style model packs such as `buffalo_l`, `antelopev2`. Provider priority is detected automatically, or you can pass `onnx_providers` in config to bias selection. |

> Additional runtimes (RKNN, Torch, etc.) follow the same backend contract and can be added in the future through `FaceRecognitionBackend`.

## Keyword-Based Tasks

Registered in `TaskRegistry`, exposed through gRPC streaming inference:

| Task keyword | Handler | Description | Input MIME | Output MIME |
|--------------|---------|-------------|------------|-------------|
| `face_detect` | `_handle_detect` | SCRFD detection with bounding boxes, confidence, and optional landmarks. Thresholds (`detection_confidence_threshold`, `nms_threshold`, `face_size_min/max`) come from request metadata. | `image/jpeg`, `image/png` | `application/json;schema=face_v1` |
| `face_embed` | `_handle_embed` | Embedding extraction (ArcFace-style, 512-dim) with optional alignment when `landmarks` metadata is provided. | `image/jpeg`, `image/png` | `application/json;schema=embedding_v1` |
| `face_detect_and_embed` | `_handle_detect_and_embed` | One-shot detect plus embedding per face, reusing detection metadata knobs and optional `max_faces`. | `image/jpeg`, `image/png` | `application/json;schema=face_v1` |

Each task returns a protobuf-defined schema serialized as JSON and includes per-request metadata such as processing time, embedding dimension, or face counts.

## Configuration & Models

Service configuration is provided through the shared Lumen YAML schema (see `examples/config/face_cn.yaml`):

```yaml
metadata:
  region: "cn"
  cache_dir: "~/.lumen"

deployment:
  mode: "single"
  service: "face"

server:
  port: 50051
  mdns:
    enabled: true
    service_name: "lumen-face"

services:
  face:
    enabled: true
    package: lumen_face
    import:
      registry_class: "lumen_face.general_face.GeneralFaceService"
      add_to_server: "lumen_face.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server"
    backend_settings:
      device: null          # auto-detect (cpu/cuda/coreml/...)
      batch_size: 8
    models:
      # Supported keys: "general", "face", "recognition"
      general:
        model: "buffalo_l"  # or antelopev2
        runtime: onnx
```

### Supported Model Configuration Keys

| Service Type | Supported Keys | Description |
|--------------|----------------|-------------|
| **GeneralFaceService** | `general`, `face`, `recognition` | Loads an InsightFace-style model pack for face detection, recognition, and embedding extraction |

Key points:

1. **Model caching** is handled by `lumen-resources`: it downloads `model_info.json`, ONNX files (`detection.fp32.onnx`, `recognition.fp32.onnx`), and metadata.
2. **Runtime metadata** (embedding dimension, SCRFD specs, preprocessing) is static overrides in `insightface_specs.py`.
3. **Backend settings** allow you to bias provider selection (`device`) and tweak batch sizes for future multi-image support.

## Supported Models

| Model ID | Detection | Recognition | Embedding Dim | Notes |
|----------|-----------|-------------|---------------|-------|
| `buffalo_l` | SCRFD 640, SCRFD anchors, `detection.fp32.onnx` | ArcFace 112, `recognition.fp32.onnx` | 512 | Default general model |
| `antelopev2` | SCRFD 640 with pack-specific heads | ArcFace | 512 | Alternative InsightFace pack |

Both shipped packs use `letterbox` preprocessing, configurable thresholds, and return optional 5-point landmarks when provided by the detector. Configurations for other InsightFace packs (e.g., `buffalo_m`, `buffalo_s`, `buffalo_sc`) remain in the repository for future expansion but their model artifacts are not currently bundled.

## Running the Server

```bash
uv run python -m lumen_face.server \
  --config lumen-face/examples/config/face_cn.yaml \
  --log-level INFO
```

The runner will:

1. Validate and download resources via `lumen-resources`.
2. Instantiate `GeneralFaceService` with the configured model/runtime.
3. Initialize `ONNXRTBackend`.
4. Start the gRPC server (and optional mDNS advertisement).

You can override the port with `--port` and dynamically scale logging via `--log-level`.

---

For shared contributor guidelines, lint rules, and deployment workflows, refer to the root `README.md` and `docs/` in the Lumen monorepo.
