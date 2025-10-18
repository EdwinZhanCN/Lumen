# Lumen-CLIP Quick Start Guide

Get up and running with Lumen-CLIP in 5 minutes.

## Prerequisites

- Python 3.10+
- pip or uv
- 4GB+ RAM (8GB+ recommended for GPU)

## Step 1: Install

```bash
# Clone repository
git clone https://github.com/EdwinZhanCN/Lumen.git
cd Lumen/lumen-clip

# Install dependencies
pip install -e .

# Optional: Install ONNX Runtime for better performance
pip install onnxruntime

# Optional: Install mDNS support
pip install zeroconf
```

## Step 2: Download Model Resources

Use `lumen-resources` to download models and datasets:

```bash
# Download (MobileCLIP2-S2) with ImageNet dataset
lumen-resources download MobileCLIP2-S2 --runtime torch --dataset ImageNet_1k --region cn

# Or download BioCLIP with TreeOfLife dataset
lumen-resources download bioclip-2 --runtime torch --dataset TreeOfLife-10M --region cn
```

**Note**: Use `--region cn` for China (ModelScope) or `--region other` for international (HuggingFace).

Resources will be downloaded to: `~/.lumen/models/`

## Step 3: Validate Configuration

Before starting, validate that all resources are available:

```bash
lumen-resources validate config/clip_only.yaml
```

Expected output:
```
✓ Configuration syntax valid
✓ Model 'MobileCLIP2-S2' found
✓ Runtime 'torch' supported
✓ Dataset 'ImageNet_1k' found
✓ All required files present
```

## Step 4: Start Service

### Option A: Using Python directly

```bash
python src/server.py --config config/clip_only.yaml
```

### Option B: Using startup script

```bash
./scripts/start_service.sh --config config/clip_only.yaml
```

Expected output:
```
INFO - Loading configuration from: config/clip_only.yaml
INFO - ✓ Configuration validated: clip service enabled
INFO - Initializing clip service...
INFO - Loading models and resources... This may take a moment.
INFO - ✓ Service initialized successfully
INFO - ✓ Supported tasks: embed, image_embed, classify, classify_scene
INFO - 🚀 clip service listening on [::]:50051
INFO - Server running. Press Ctrl+C to stop.
```

## Step 5: Test the Service

### Using grpcurl

```bash
# Check health
grpcurl -plaintext localhost:50051 inference.Inference/Health

# Get capabilities
grpcurl -plaintext localhost:50051 inference.Inference/GetCapabilities
```

### Using Python client

Create `test_client.py`:

```python
import grpc
import base64
from pathlib import Path
import ml_service_pb2 as pb
import ml_service_pb2_grpc as rpc

# Connect to service
channel = grpc.insecure_channel('localhost:50051')
stub = rpc.InferenceStub(channel)

# Test text embedding
def test_embed():
    request = pb.InferRequest(
        correlation_id="test-1",
        task="embed",
        payload=b"a photo of a golden retriever",
        payload_mime="text/plain",
    )

    responses = stub.Infer(iter([request]))
    for response in responses:
        if response.is_final:
            print(f"Embedding dimension: {response.meta.get('dim')}")
            print("Success!")

# Test image classification
def test_classify():
    # Load image
    image_path = "path/to/your/image.jpg"
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    request = pb.InferRequest(
        correlation_id="test-2",
        task="classify",
        payload=image_bytes,
        payload_mime="image/jpeg",
        meta={"topk": "5"},
    )

    responses = stub.Infer(iter([request]))
    for response in responses:
        if response.is_final:
            import json
            result = json.loads(response.result)
            print("Classification results:")
            for label_info in result['labels']:
                print(f"  {label_info['label']}: {label_info['score']:.3f}")

if __name__ == "__main__":
    test_embed()
    test_classify()
```

Run:
```bash
python test_client.py
```

## Common Deployment Scenarios

### Scenario 1: CLIP Only (General Purpose)

**Use case**: Image classification, scene analysis, general embeddings

**Config**: `config/clip_only.yaml`

```yaml
services:
    clip:
        enabled: true
        models:
            default:
                model: "MobileCLIP2-S2"
                runtime: "torch"
        server:
            port: 50051
```

**Start**:
```bash
python src/server.py --config config/clip_only.yaml
```

**Supported tasks**:
- `embed`: Text → embedding
- `image_embed`: Image → embedding
- `classify`: Image → ImageNet labels
- `classify_scene`: Image → scene category

---

### Scenario 2: BioCLIP Only (Species Identification)

**Use case**: Biological/species identification, nature photography

**Config**: `config/bioclip_only.yaml`

```yaml
services:
    bioclip:
        enabled: true
        models:
            default:
                model: "bioclip-2"
                runtime: "torch"
                dataset: "TreeOfLife-10M"
        server:
            port: 50052
```

**Download resources**:
```bash
lumen-resources download bioclip-2 --runtime torch --dataset TreeOfLife-10M --region cn
```

**Start**:
```bash
python src/server.py --config config/bioclip_only.yaml
```

**Supported tasks**:
- `embed`: Text → biological embedding
- `image_embed`: Image → biological embedding
- `classify`: Image → species identification

---

### Scenario 3: Unified Service (Smart Classification)

**Use case**: Automatic model selection, best-of-both-worlds

**Config**: `config/unified_service.yaml`

```yaml
services:
    clip-unified:
        enabled: true
        models:
            clip_default:
                model: "MobileCLIP2-S2"
                runtime: "torch"
            bioclip_default:
                model: "bioclip-2"
                runtime: "torch"
                dataset: "TreeOfLife-10M"
        server:
            port: 50053
```

**Start**:
```bash
python src/server.py --config config/unified_service.yaml
```

**Supported tasks**:
- `clip_embed`, `bioclip_embed`
- `clip_image_embed`, `bioclip_image_embed`
- `clip_classify`, `bioclip_classify`
- `smart_classify`: Automatically routes to best model

**Smart classify behavior**:
1. Analyzes image with CLIP scene classification
2. If scene is "animal-like" → routes to BioCLIP for species ID
3. Otherwise → returns CLIP scene classification

---

### Scenario 4: Multi-Service Deployment

Run multiple services on different ports:

**Terminal 1 - CLIP**:
```bash
python src/server.py --config config/clip_only.yaml --port 50051
```

**Terminal 2 - BioCLIP**:
```bash
python src/server.py --config config/bioclip_only.yaml --port 50052
```

**Terminal 3 - Unified**:
```bash
python src/server.py --config config/unified_service.yaml --port 50053
```

## Performance Optimization

### Use GPU

```yaml
env:
    DEVICE: "cuda"        # Use NVIDIA GPU
    BATCH_SIZE: "16"      # Increase batch size for better throughput
```

### Use ONNX Runtime

ONNX Runtime offers ~2-3x faster inference:

```bash
# Download ONNX models
lumen-resources download MobileCLIP2-S2 --runtime onnx --region cn

# Update config
# config/clip_onnx.yaml
services:
    clip:
        models:
            default:
                model: "MobileCLIP2-S2"
                runtime: "onnx"
        env:
            ONNX_PROVIDERS: "CUDAExecutionProvider,CPUExecutionProvider"
```

### Tune Batch Size

Experiment with batch sizes based on your workload:

- **Low latency** (single requests): `BATCH_SIZE: "1"`
- **High throughput** (batch processing): `BATCH_SIZE: "32"`
- **Balanced**: `BATCH_SIZE: "8"` (default)

## Troubleshooting

### Problem: "ResourceNotFoundError: Model files not found"

**Solution**:
```bash
# Download the model
lumen-resources download <model-name> --runtime torch --region cn

# Verify
lumen-resources list
```

---

### Problem: "Task 'classify' not supported"

**Solution**: Dataset is missing. Download it:
```bash
# For CLIP
lumen-resources download MobileCLIP2-S2 --dataset ImageNet_1k --region cn

# For BioCLIP
lumen-resources download bioclip-2 --dataset TreeOfLife-10M --region cn
```

---

### Problem: "Multiple services enabled"

**Solution**: Only one service can be enabled per config file.

**Wrong** ❌:
```yaml
services:
    clip:
        enabled: true
    bioclip:
        enabled: true  # ERROR!
```

**Correct** ✅:
```yaml
# config/clip_only.yaml
services:
    clip:
        enabled: true
    bioclip:
        enabled: false
```

For multiple services, use multiple config files and run separate processes.

---

### Problem: Service is slow

**Solutions**:
1. **Enable GPU**: Set `DEVICE: "cuda"` in config
2. **Use ONNX**: Download ONNX models and switch runtime
3. **Increase batch size**: Set `BATCH_SIZE: "16"` or higher
4. **Use smaller model**: Try MobileCLIP variants for faster inference

---

### Problem: Out of memory

**Solutions**:
1. **Reduce batch size**: Set `BATCH_SIZE: "4"` or lower
2. **Use CPU**: Set `DEVICE: "cpu"`
3. **Use smaller model**: Switch to a lighter model
4. **Close other services**: Free up GPU/RAM

---

### Problem: mDNS not advertising

**Solutions**:
1. Install zeroconf: `pip install zeroconf`
2. Set advertised IP: `export ADVERTISE_IP=192.168.1.100`
3. Check firewall settings
4. Verify mDNS is enabled in config:
```yaml
server:
    mdns:
        enabled: true
```

## Next Steps

- **Production Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **API Reference**: See [API.md](API.md)
- **Custom Models**: See [CUSTOM_MODELS.md](CUSTOM_MODELS.md)
- **Performance Tuning**: See [PERFORMANCE.md](PERFORMANCE.md)

## Getting Help

- **Issues**: https://github.com/EdwinZhanCN/Lumen/issues
- **Discussions**: https://github.com/EdwinZhanCN/Lumen/discussions
- **Documentation**: Full README at [README.md](README.md)

---

**Happy inferencing! 🚀**
