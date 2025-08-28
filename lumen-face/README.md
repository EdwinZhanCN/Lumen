# lumen-face

A lightweight, modular face embedding service with pluggable backends: ONNX Runtime (cross-platform) and RKNN (typically Linux-only). It resolves model artifacts from either a local directory or a Hugging Face layout under Lumilio-Photos, and provides a simple API for generating L2-normalized face embeddings.

Status:
- Ready for programmatic use and CLI-based configuration checks
- Detection is scaffolded (stub); recognition/embedding is implemented
- Designed to be embedded into a unified AI service later (gRPC/FastAPI)

Python: 3.13+
License: MIT (or your choice; update as needed)

---

## Highlights

- Backends
  - onnxrt: Cross-platform via ONNX Runtime
  - rknn: Rockchip RKNN Toolkit (Linux-only in practice)
- Models: Resolved from either local paths or URLs with this layout:
  - https://huggingface.co/Lumilio-Photos/{model_name}/{runtime}/{precision}/detection.{ext}
  - https://huggingface.co/Lumilio-Photos/{model_name}/{runtime}/{precision}/recognition.{ext}
  - runtime ∈ {onnxrt, rknn}
  - precision ∈ {fp16, int8}
  - ext = onnx for onnxrt, rknn for rknn
- Stateless: Returns per-face embeddings. Compare/group client-side.
- Environment-driven configuration with a clean `Config` dataclass API.

---

## Installation

This package aims to keep core dependencies minimal. Install runtime-specific packages explicitly:

- Base (array/image I/O):
  - pip install numpy
  - pip install pillow  # optional; improves image resizing/format handling

- ONNX Runtime backend:
  - pip install onnxruntime        # CPU
  - or: pip install onnxruntime-gpu  # if you want CUDA EP on Linux/Windows

- RKNN backend:
  - Install the appropriate RKNN Toolkit for your device from Rockchip (often Linux-only).
  - Python package name and install method vary by platform/toolkit release.

Note: The project itself does not force-install these extras so you can choose the minimal set you need.

---

## Model Layout

You can point to local files or remote artifacts following this layout:

- Local directory (preferred if present):
  - {LOCAL_DIR}/{model_name}/{runtime}/{precision}/detection.{onnx|rknn}
  - {LOCAL_DIR}/{model_name}/{runtime}/{precision}/recognition.{onnx|rknn}

- Hugging Face (default remote base):
  - https://huggingface.co/Lumilio-Photos/{model_name}/{runtime}/{precision}/detection.{onnx|rknn}
  - https://huggingface.co/Lumilio-Photos/{model_name}/{runtime}/{precision}/recognition.{onnx|rknn}

Defaults:
- repo_base: https://huggingface.co/Lumilio-Photos
- model_name: scrfd_10g_iresnet50
- runtime: onnxrt
- precision: fp16

Downloads from remote URIs are cached to:
- ${LUMEN_FACE_CACHE_DIR} or ~/.cache/lumen_face

---

## Environment Variables

All are optional; sensible defaults apply.

- LUMEN_FACE_RUNTIME or LUMEN_FACE_BACKEND
  - onnxrt (default) | rknn
- LUMEN_FACE_DEVICE
  - Device hint string, e.g., cpu, cuda:0, rknpu0
- LUMEN_FACE_MODEL_NAME
  - Model pack name under repo; default: scrfd_10g_iresnet50
- LUMEN_FACE_PRECISION
  - fp16 (default) | int8
- LUMEN_FACE_REPO_BASE
  - Base repo URL (default: https://huggingface.co/Lumilio-Photos)
- LUMEN_FACE_LOCAL_DIR
  - Local directory to prefer for artifacts; if files exist locally, remote is skipped
- LUMEN_FACE_TIMEOUT
  - Remote fetch timeout seconds; default: 30
- LUMEN_FACE_MAX_BATCH_SIZE
  - Integer hint for batching (not enforced by scaffold)
- LUMEN_FACE_SUPPORTS_IMAGE_BATCH
  - "1"/"true" to indicate backend supports image batch input (hint only)
- LUMEN_FACE_EXTRA
  - JSON object string for arbitrary extras. For RKNN, common keys:
    - rknn_target: "rk3588" | "rk356x" | "rk3399pro" | "android" | ...
    - rknn_device_id: "0" | "1" | ...
    - rknn_perf_debug: "1" | "0"
    - rec_input_h: "112" (recognition height override)
    - rec_input_w: "112" (recognition width override)
- LUMEN_FACE_CACHE_DIR
  - Directory to cache downloaded artifacts; default: ~/.cache/lumen_face

---

## Programmatic Usage

Basic embedding with auto-selected backend (from environment):

```/dev/null/example.py#L1-200
from PIL import Image
import numpy as np

from face_recognition import create_backend

# Configure via environment variables (see README)
backend = create_backend()

# Prepare an image (numpy array HxWxC in RGB or a PIL.Image)
img = Image.open("face.jpg").convert("RGB")

# Whole image embedding (no boxes)
embeddings = backend.embed(img)
print(f"Generated {len(embeddings)} embedding(s). D = {embeddings[0].shape[0]}")

# With bounding boxes (x1, y1, x2, y2). Coordinates are pixels.
boxes = [(50, 60, 200, 240)]
embeddings = backend.embed(img, boxes=boxes)
for i, e in enumerate(embeddings):
    print(f"Face {i}: norm = {np.linalg.norm(e):.4f} (should be ~1.0)")

backend.close()
```

For a specific backend:

```/dev/null/example_specific_backend.py#L1-200
from face_recognition import RuntimeKind, create_backend

backend = create_backend(
    runtime=RuntimeKind.ONNXRT,
    device_preference="cpu",
    pack_name="scrfd_10g_iresnet50",
    precision="fp16",
)
# ... use backend.embed(...)
backend.close()
```

Notes:
- Detection is currently a stub (returns no boxes). The `embed` API uses provided boxes or the full image. You can extend `detect` to wire your detector’s pre/post-processing.
- Recognition input size:
  - ONNX: inferred from the model’s input if static; otherwise defaults to 112x112.
  - RKNN: defaults to 112x112 and can be overridden via `LUMEN_FACE_EXTRA` (rec_input_h, rec_input_w).

---

## CLI

The standalone CLI has been removed. Use face_recognition.create_backend(...) within your service or app.





---

## Backend Behavior and Notes

- onnxrt
  - Providers chosen from `LUMEN_FACE_DEVICE`:
    - "cuda[:id]" -> CUDAExecutionProvider (+ CPU fallback)
    - otherwise -> CPUExecutionProvider
  - Downloads remote models to cache directory and loads via `onnxruntime.InferenceSession`.

- rknn
  - Requires RKNN Toolkit matching your device (e.g., RKNN Toolkit 2 for RK3588).
  - Initializes runtime using extras (target, device_id, perf_debug).
  - Many RKNN models are compiled for batch=1; the scaffold runs crops sequentially.

- Embedding Output
  - The last output tensor is taken as the embedding. If needed, adapt the selection.
  - Embeddings are L2-normalized.

- Detection Scaffold
  - `detect(image)` currently returns an empty list.
  - To support automatic face detection, inspect the detection model’s IO, add proper preprocessing, and decode outputs into pixel boxes.

---

## Configuration Cheatsheet

Quick environment example for onnxrt with local models:

```/dev/null/env_example_onxxrt.sh#L1-200
export LUMEN_FACE_RUNTIME=onnxrt
export LUMEN_FACE_DEVICE=cpu
export LUMEN_FACE_LOCAL_DIR=/opt/models
export LUMEN_FACE_MODEL_NAME=scrfd_10g_iresnet50
export LUMEN_FACE_PRECISION=fp16
# Optional:
export LUMEN_FACE_CACHE_DIR=/var/cache/lumen_face
```

For RKNN on RK3588:

```/dev/null/env_example_rknn.sh#L1-200
export LUMEN_FACE_RUNTIME=rknn
export LUMEN_FACE_LOCAL_DIR=/opt/models
export LUMEN_FACE_MODEL_NAME=scrfd_10g_iresnet50
export LUMEN_FACE_PRECISION=int8
export LUMEN_FACE_EXTRA='{"rknn_target":"rk3588","rknn_device_id":"0","rec_input_h":"112","rec_input_w":"112"}'
```

---

## FAQ

- Q: Do I need Hugging Face credentials?
  - A: Not for public artifacts at Lumilio-Photos. The scaffold uses plain HTTP(S) download. For private models or advanced features, integrate huggingface_hub yourself.

- Q: How do I compare embeddings?
  - A: Compute cosine similarity or Euclidean distance between normalized vectors. Group/compare client-side as needed.

- Q: Can I run multiple services in one process?
  - A: Yes. This package is designed to be registered within a unified server. You can build a gRPC/FastAPI layer over these backends.

---

## Roadmap

- Implement standardized detection (e.g., SCRFD postprocessing)
- Add gRPC/HTTP server with batching and health endpoints
- Optional integration with huggingface_hub for robust caching/versioning
- Dockerfiles for CPU/CUDA/ROCm/RKNN images



## Contributing

- Keep the backend interface consistent
- Add environment variables and README updates when you add features
- Prefer dependency-light utilities and make heavy deps optional by backend
