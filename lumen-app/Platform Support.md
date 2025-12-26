# Platform Support (Presets Included)

## CPU

## NVIDIA GPU

**System Requirements:** Linux, Windows  10/11

**ONNXRuntime Execution Provider:** CUDA Execution Provider, TensorRT Execution Provider (High RAM)

**Python Version:** 3.11

**Python Package:** `onnxruntime-gpu`

**Steps:**

1. Install micromamba.
2. Run micromamba with `utils/mamba/cuda.yaml` or `utils/mamba/tensorrt.yaml` (High RAM).
3. Run `uv pip install` all required `lumen-*` modules.
4. Lumen app will start the inference server then.

## Intel iGPU

**System Requirements:** Linux, Windows  10/11

**ONNXRuntime Execution Provider:** OpenVINO Execution Provider

**Python Version:** 3.11

**Python Package:** `onnxruntime-openvino`

## Intel Arc

**System Requirements:** Linux, Windows  10/11

**ONNXRuntime Execution Provider:** OpenVINO Execution Provider

**Python Version:** 3.11

**Python Package:** `onnxruntime-openvino`

## AMD Ryzen GPU

**System Requirements:** Windows  10/11 (with DirectX 12)

**ONNXRuntime Execution Provider:** DirectML Execution Provider

**Python Version:** 3.11

**Python Package:** `onnxruntime`

## Apple Silicon

**System Requirements:** macOS 13+

**ONNXRuntime Execution Provider:** CoreML Execution Provider

**Python Version:** 3.11

**Python Package:** `onnxruntime`

## Rokchip RKNPU



