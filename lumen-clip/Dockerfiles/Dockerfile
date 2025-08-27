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
EXPOSE 50051

# Run the unified server module
CMD ["python", "-m", "src.server"]
