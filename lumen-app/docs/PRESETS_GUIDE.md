# Lumen Configuration Presets Guide

## Overview

Lumen provides two-level configuration presets:

1. **DeviceConfig Presets** - Hardware-specific optimization settings
2. **LumenConfig Presets** - Application-specific feature combinations

---

## DeviceConfig Presets

Hardware configuration presets that define runtime, execution providers, and precision constraints.

### Precision Behavior

- **`precision = None`**: Device supports multiple precisions (fp32/fp16/int8/q4). Actual precision is determined by LumenConfig Presets.
- **`precision = "value"`**: Hardware constraint or optimal precision. Overrides LumenConfig Presets.

### Available Presets

#### Rockchip NPU
```python
DeviceConfig.rockchip(rknn_device="rk3588")
```
- **Runtime**: `rknn`
- **Precision**: `int8` (fixed)
- **Batch Size**: `1` (NPU constraint)
- **Use Case**: RK3588 and other Rockchip NPU devices

#### Apple Silicon
```python
DeviceConfig.apple_silicon()
```
- **Runtime**: `onnx` with CoreML
- **Precision**: `None` (flexible)
- **Batch Size**: `1`
- **Execution Providers**: CoreML → CPU
- **Use Case**: M1/M2/M3/M4 chips

#### NVIDIA GPU (Low RAM)
```python
DeviceConfig.nvidia_gpu()
```
- **Runtime**: `onnx` with CUDA
- **Precision**: `None` (flexible)
- **Batch Size**: `4`
- **Use Case**: GPUs with < 12GB VRAM

#### NVIDIA GPU (High RAM)
```python
DeviceConfig.nvidia_gpu_high()
```
- **Runtime**: `onnx` with TensorRT
- **Precision**: `fp16` (optimal)
- **Batch Size**: `None` (dynamic)
- **Features**: TensorRT cache enabled, 2GB workspace
- **Use Case**: GPUs with >= 12GB VRAM

#### Intel GPU
```python
DeviceConfig.intel_gpu()
```
- **Runtime**: `onnx` with OpenVINO
- **Precision**: `fp16` (optimal)
- **Batch Size**: `None` (dynamic)
- **Use Case**: Intel iGPU, Arc GPU

#### AMD GPU (Windows)
```python
DeviceConfig.amd_gpu_win()
```
- **Runtime**: `onnx` with DirectML
- **Precision**: `None` (flexible)
- **Execution Providers**: DML → CPU
- **Use Case**: AMD Radeon GPUs on Windows

#### AMD NPU
```python
DeviceConfig.amd_npu()
```
- **Runtime**: `onnx` with VitisAI
- **Precision**: `None` (flexible)
- **Features**: VitisAI cache enabled
- **Use Case**: AMD Ryzen AI NPUs

#### NVIDIA Jetson (Low RAM)
```python
DeviceConfig.nvidia_jetson()
```
- **Runtime**: `onnx` with CUDA
- **Precision**: `None` (flexible)
- **Use Case**: Jetson devices with < 12GB RAM

#### NVIDIA Jetson (High RAM)
```python
DeviceConfig.nvidia_jetson_high()
```
- **Runtime**: `onnx` with TensorRT
- **Precision**: `fp16` (optimal)
- **Features**: TensorRT cache enabled
- **Use Case**: Jetson devices with >= 12GB RAM

#### CPU
```python
DeviceConfig.cpu()
```
- **Runtime**: `onnx`
- **Precision**: `None` (flexible)
- **Batch Size**: `1`
- **Execution Providers**: CPU only
- **Use Case**: General-purpose CPUs

---

## LumenConfig Presets

Application-level presets that combine multiple AI services.

### Common Parameters

- **clip_model**: Model variant for CLIP service
  - `light_weight`: `"MobileCLIP2-S2"` or `"CN-CLIP_ViT-B-16"`
  - `basic`: `"MobileCLIP2-S4"` or `"CN-CLIP_ViT-L-14"`

### Available Presets

#### Minimal
```python
Config.minimal()
```
- **Services**: OCR only
- **OCR Model**: PP-OCRv5
- **Batch Size**: 1
- **Use Case**: Lightweight OCR-only applications

#### Light Weight
```python
Config.light_weight(clip_model="MobileCLIP2-S2")
```
- **Services**: OCR, CLIP, Face
- **Models**:
  - OCR: PP-OCRv5 (fp32)
  - CLIP: MobileCLIP2-S2 / CN-CLIP_ViT-B-16 (int8)
  - Face: buffalo_l (int8)
- **Batch Size**: 1
- **Use Case**: Edge devices, resource-constrained environments

#### Basic
```python
Config.basic(clip_model="MobileCLIP2-S4")
```
- **Services**: OCR, CLIP, Face, VLM
- **Models**:
  - OCR: PP-OCRv5 (fp32)
  - CLIP: MobileCLIP2-S4 / CN-CLIP_ViT-L-14 (int8)
  - Face: antelopev2 (int8)
  - VLM: FastVLM-0.5B (int8)
- **Batch Size**: 5
- **Use Case**: Standard AI applications with vision-language capabilities

#### Brave
```python
Config.brave()
```
- **Services**: OCR, CLIP, Face, VLM
- **Models**:
  - OCR: PP-OCRv5 (fp32)
  - CLIP: bioclip-2 (fp16)
  - Face: antelopev2 (fp16)
  - VLM: FastVLM-0.5B (int8)
- **Batch Size**: 8-10 (varies by service)
- **Use Case**: High-performance applications with biological recognition

---

## Usage Example

```python
from lumen_app.core.config import Config, DeviceConfig

# Step 1: Choose device preset
device = DeviceConfig.nvidia_gpu_high()

# Step 2: Create config with device preset
config = Config(
    cache_dir="./cache",
    device_config=device,
    region="cn",
    service_name="lumen-service",
    port=50051
)

# Step 3: Choose application preset
lumen_config = config.basic(clip_model="MobileCLIP2-S4")

# Step 4: Use the configuration
# lumen_config contains all settings for deployment
```

---

## Precision Override Rules

1. **DeviceConfig.precision = None**: Use precision from LumenConfig preset
2. **DeviceConfig.precision = "value"`: Override LumenConfig preset

### Examples

```python
# Rockchip NPU: Always uses int8 (hardware constraint)
DeviceConfig.rockchip("rk3588")  # precision="int8"
Config.light_weight(...)  # CLIP uses int8 (forced by device)

# NVIDIA GPU: Uses preset-defined precision
DeviceConfig.nvidia_gpu()  # precision=None
Config.light_weight(...)  # CLIP uses int8 (from preset)
Config.brave()            # CLIP uses fp16 (from preset)

# Intel GPU: Always uses fp16 (optimal)
DeviceConfig.intel_gpu()  # precision="fp16"
Config.basic(...)  # All models use fp16 (forced by device)
```

---

## Best Practices

1. **Match preset to hardware**: Choose appropriate device preset based on available hardware
2. **Consider memory constraints**: Use batch size overrides for low-memory devices
3. **Precision trade-offs**:
   - `fp32`: Best accuracy, slowest
   - `fp16`: Good accuracy, faster
   - `int8`: Moderate accuracy, fastest
   - `q4fp16`: Lowest accuracy, minimal memory usage
4. **Customization**: All presets can be further customized by modifying the returned `LumenConfig`
