## 安装说明

根据平台选择

### CPU Only
```bash
pip install lumen-clip[cpu]
```

### CUDA GPU 支持（以 CUDA 11.8 为例）

torch:
- CPU
- CUDA
onnxruntime:
- CPU EP
- CUDA EP
- TensorRT EP

```bash
pip install lumen-clip[cuda] \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
```

### MPS / macOS 支持：

torch:
- CPU
- MPS
onnxruntime:
- CPU EP

```bash
pip install lumen-clip[mps]
```

### OpenVINO 支持：

torch:
- CPU
onnxruntime:
- CPU EP
- OpenVINO EP

```bash
pip install lumen-clip[openvino2404]
```

### Jetson / 嵌入式支持：

torch:
- CPU
- CUDA
onnxruntime:
- CPU EP
- CUDA EP
- TensorRT EP

**China Mainland**

```bash
pip install lumen-clip[jp61] \
  --index-url https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/ \
  --extra-index-url https://pypi.org/simple
```

**Other Region**

```bash
pip install lumen-clip[jp61] \
  --index-url https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/ \
  --extra-index-url https://pypi.org/simple
```

### RKNN NPU 支持（Arm Linux Only）：
