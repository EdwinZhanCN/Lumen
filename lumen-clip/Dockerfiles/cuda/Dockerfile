# Stage 1: Build stage with uv on Python 3.12 (CUDA 12.6)
FROM nvidia/cuda:12.6.0-base-ubuntu22.04 AS builder

# Install Python 3.12 and pip for that interpreter, then install uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    software-properties-common \
    gnupg && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.12 /tmp/get-pip.py && rm -f /tmp/get-pip.py && \
    python3.12 -m pip install --no-cache-dir --upgrade pip && \
    python3.12 -m pip install --no-cache-dir uv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency descriptors first to leverage caching
COPY pyproject.toml uv.lock ./

# Install project dependencies (GPU build) into the system interpreter (Python 3.12)
# Use PyTorch CUDA 12.6 wheel index
RUN uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple '.[gpu]'

# Stage 2: Runtime image (CUDA 12.6 + Python 3.12)
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Install Python 3.12 runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    software-properties-common \
    gnupg && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from the builder (installed into /usr/local by pip)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy application code
COPY . .

# Ensure the source tree is importable (src/ layout)
ENV PYTHONPATH=/app/src

# Environment variables (runtime selection & discovery)
# Backend selection:
#   - CLIP_BACKEND: torch (default) | onnxrt | rknn
#   - BIOCLIP_BACKEND: torch (default). If unset, falls back to CLIP_BACKEND
# Devices and batching:
#   - CLIP_DEVICE, BIOCLIP_DEVICE: cuda | mps | cpu (hints; if available)
#   - CLIP_MAX_BATCH_SIZE, BIOCLIP_MAX_BATCH_SIZE: optional integer hints
# ONNX Runtime:
#   - CLIP_ONNX_IMAGE, CLIP_ONNX_TEXT (and BIOCLIP_* equivalents): paths to encoder ONNX files
#   - CLIP_ORT_PROVIDERS, BIOCLIP_ORT_PROVIDERS: comma-separated providers (e.g. "CUDAExecutionProvider,CPUExecutionProvider")
# RKNN:
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
# Expose the gRPC port
EXPOSE 50051

# Run the unified server module
CMD ["python", "-m", "src.server"]
