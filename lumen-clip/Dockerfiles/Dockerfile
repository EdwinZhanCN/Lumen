# Stage 1: Build stage with uv on Python 3.12 (CPU)
FROM python:3.12-slim AS builder

# Install uv
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

# Copy only dependency descriptors to leverage Docker layer caching
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install project (and its dependencies) for CPU
# This installs extras defined under [project.optional-dependencies].cpu
# Note: RKNN support is Linux-only and excluded in this CPU image. For Linux GPU images
# (CUDA/ROCm), install with '.[gpu,rknn]' to include rknn-toolkit2.
RUN uv pip install --system --no-cache '.[cpu]'

# Stage 2: Final runtime image (Python 3.12, CPU)
FROM python:3.12-slim

WORKDIR /app

# Copy installed dependencies from builder
# Correct site-packages path for Python 3.12
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy application code
COPY . .

# Ensure Python can import from src/ so absolute imports like "from proto import ..." resolve
ENV PYTHONPATH=/app/src

# Expose the gRPC port
# Environment variables (runtime selection & discovery)
# Backend selection:
#   - CLIP_BACKEND: torch (default) | onnxrt | rknn (Linux-only)
#   - BIOCLIP_BACKEND: torch (default). If unset, falls back to CLIP_BACKEND
# Devices and batching:
#   - CLIP_DEVICE, BIOCLIP_DEVICE: cuda | mps | cpu (hints; if available)
#   - CLIP_MAX_BATCH_SIZE, BIOCLIP_MAX_BATCH_SIZE: optional integer hints
# ONNX Runtime:
#   - CLIP_ONNX_IMAGE, CLIP_ONNX_TEXT (and BIOCLIP_* equivalents): paths to encoder ONNX files
#   - CLIP_ORT_PROVIDERS, BIOCLIP_ORT_PROVIDERS: comma-separated providers (e.g. "CUDAExecutionProvider,CPUExecutionProvider")
# RKNN (Linux-only):
#   - Requires rknn-toolkit2; include via extras on Linux images only.
#   - In CUDA/ROCm Dockerfiles, change install to: uv pip install --system --no-cache '.[gpu,rknn]'
#   - CLIP_RKNN_MODEL, BIOCLIP_RKNN_MODEL: path to compiled .rknn model
#   - CLIP_RKNN_TARGET, BIOCLIP_RKNN_TARGET: SoC target (default "rk3588")
# Models:
#   - CLIP_MODEL_NAME (e.g. "ViT-B-32"), CLIP_PRETRAINED (e.g. "laion2b_s34b_b79k")
#   - BIOCLIP_MODEL_NAME (e.g. "hf-hub:imageomics/bioclip-2")
# Service:
#   - BATCH_SIZE: server-side request batch size (default 8)
# mDNS discovery:
#   - CLIP_MDNS_TYPE (default "_homenative-node._tcp.local."), CLIP_MDNS_NAME (default "CLIP-Image-Processor")
#   - CLIP_MDNS_UUID, CLIP_MDNS_STATUS=ready, CLIP_MDNS_VERSION=1.0.0
#   - ADVERTISE_IP: override advertised LAN IP
EXPOSE 50051

# Run the unified server module
CMD ["python", "-m", "src.server"]
