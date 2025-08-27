# Stage 1: Build stage with uv on Python 3.12 (ROCm 6.3)
FROM rocm/pytorch:rocm6.3_ubuntu22.04_py3.10_pytorch_release-2.5.0 AS builder

# Install Python 3.12 and uv, make python3 -> python3.12
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

# Copy dependency descriptors first for better build caching
COPY pyproject.toml uv.lock ./

# Install project + deps (ROCm wheels for PyTorch)
# Uses optional dependency group [gpu] defined in pyproject.toml
RUN uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/rocm6.3 '.[gpu]'

# Stage 2: Runtime image (ROCm 6.3 + Python 3.12)
FROM rocm/pytorch:rocm6.3_ubuntu22.04_py3.10_pytorch_release-2.5.0

# Install Python 3.12 runtime and set python3 -> python3.12
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

# Copy installed Python packages from builder (Python 3.12 site-packages)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy application source
COPY . .

# Ensure the src/ layout is importable and set common ROCm env
ENV PYTHONPATH=/app/src
# Override as needed for your GPU architecture; 10.3.0 is a common fallback
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0

# Expose the gRPC port
EXPOSE 50051

# Run the unified server module
CMD ["python", "-m", "src.server"]
