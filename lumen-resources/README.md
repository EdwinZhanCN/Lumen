# Lumen Resources

统一的模型资源管理工具，用于从 HuggingFace 和 ModelScope 平台下载和管理 ML 模型资源。

## 安装

```bash

pip install "lumen-resources[all]"

# 只安装 HuggingFace 支持
pip install "lumen-resources[huggingface]"

# 只安装 ModelScope 支持
pip install "lumen-resources[modelscope]"
```

## 快速开始

### 1. 创建配置文件

配置文件采用 YAML 格式，主要字段说明：

#### metadata 元数据配置
- `version`: 配置版本号（如 "1.0"）
- `region`: `"cn"` (使用 ModelScope) 或 `"other"` (使用 HuggingFace)
- `cache_dir`: 本地缓存目录

#### dependencies 依赖配置（可选）
- 列表格式，指定需要安装的 Python 包依赖

#### services 服务配置
每个服务包含以下字段：
- `enabled`: 是否启用服务
- `package`: Python 包名
- `import`: 导入配置
  - `registry_class`: 服务注册类路径
  - `add_to_server`: gRPC 服务添加函数路径
- `models`: 模型配置字典
  - `model`: 模型仓库名称
  - `runtime`: `"torch"` | `"onnx"` | `"rknn"`
  - `rknn_device`: 仅当 `runtime` 为 `"rknn"` 时需要，指定设备型号如 `"rk3588"`
  - `dataset`: 可选，数据集名称（用于 BioCLIP 等）
- `default_model`: 默认使用的模型名称
- `env`: 环境变量配置（可选）
- `server`: 服务器配置
  - `port`: 服务端口
  - `mdns`: mDNS 配置
    - `enabled`: 是否启用 mDNS
    - `name`: 服务名称
    - `type`: 服务类型（可选）

#### hub Hub 模式配置（可选）
- `enabled`: 是否启用 Hub 模式
- `server`: Hub 服务器配置

参考 `examples/` 目录中的配置模板：
- `config_minimal.yaml` - 最小化配置（仅 CLIP 服务）
- `config_multi_models.yaml` - 多模型配置示例
- `config_hub.yaml` - Hub 模式配置
- `config_rknn_edge.yaml` - RKNN 边缘设备配置

#### 完整配置示例

```yaml
# 完整的服务配置示例
metadata:
  version: "1.0"
  region: "cn"  # "cn" 使用 ModelScope，"other" 使用 HuggingFace
  cache_dir: "~/.lumen/models"

# 可选的依赖包配置
dependencies:
  - "lumen-clip @ git+https://github.com/lumilio/lumen.git@main#subdirectory=lumen-clip"
  - "lumen-face @ git+https://github.com/lumilio/lumen.git@main#subdirectory=lumen-face"

services:
  clip:
    enabled: true
    package: "lumen-clip"
    import:
      registry_class: "lumen_clip.service_registry.CLIPService"
      add_to_server: "lumen_clip.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server"
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "onnx"
      expert:
        model: "bioclip-2"
        runtime: "torch"
        dataset: "TreeOfLife-200M"  # BioCLIP 专用数据集
    default_model: "default"
    env:
      CLIP_BACKEND: "onnx"
      CLIP_BATCH_SIZE: "8"
    server:
      port: 50051
      mdns:
        enabled: true
        name: "CLIP-Service"
        type: "_homenative-node._tcp.local."

  face:
    enabled: true
    package: "lumen-face"
    import:
      registry_class: "lumen_face.service_registry.FaceService"
      add_to_server: "lumen_face.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server"
    models:
      default:
        model: "antelopev2"
        runtime: "onnx"
    server:
      port: 50052
      mdns:
        enabled: true
        name: "Face-Service"

# Hub 模式配置（可选）
hub:
  enabled: true
  server:
    port: 50050
    mdns:
      enabled: true
      name: "Lumen-AI-Hub"
      type: "_homenative-node._tcp.local."
```

#### RKNN 边缘设备配置示例

```yaml
# 针对 RK3588 等 ARM 边缘设备的优化配置
metadata:
  version: "1.0"
  region: "cn"
  cache_dir: "/data/lumen/models"

services:
  clip:
    enabled: true
    package: "lumen-clip"
    import:
      registry_class: "lumen_clip.service_registry.CLIPService"
      add_to_server: "lumen_clip.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server"
    models:
      default:
        model: "MobileCLIP2-S2"
        runtime: "rknn"
        rknn_device: "rk3588"  # 指定 RKNN 设备型号
    env:
      CLIP_BACKEND: "rknn"
      RKNN_BATCH_SIZE: "1"  # 边缘设备通常使用 batch_size=1
    server:
      port: 50051
      mdns:
        enabled: true
        name: "CLIP-Edge-RK3588"
```

#### 配置验证和注意事项

**配置文件验证：**
- 使用 `lumen-resources validate <config_file>` 验证配置文件语法和字段
- 确保所有必需字段都已正确配置
- 检查模型名称和运行时组合是否有效

**常见配置注意事项：**
- `region` 字段影响下载源：`"cn"` 使用 ModelScope，`"other"` 使用 HuggingFace
- RKNN 运行时需要指定 `rknn_device` 字段，支持的设备有：`rk3566`, `rk3588`, `rk3576` 等
- BioCLIP 模型可以配置 `dataset` 字段来指定专用数据集
- 端口配置避免冲突，建议使用 50050-50099 范围
- mDNS 服务名称应具有唯一性，避免网络中的命名冲突

**性能优化建议：**
- 内存受限设备（<4GB）：使用 ONNX 运行时，避免同时加载多个大模型
- ARM 边缘设备：优先使用 RKNN 运行时以获得 NPU 加速
- 批处理优化：根据硬件能力调整 `BATCH_SIZE` 环境变量

### 2. 命令行使用

```bash
# 验证配置文件
lumen-resources validate examples/config_hub.yaml

# 下载资源
lumen-resources download examples/config_hub.yaml

# 强制重新下载
lumen-resources download examples/config_hub.yaml --force

# 列出已缓存的模型
lumen-resources list ~/.lumen/
```

### 3. 作为 Python 库使用

```python
from pathlib import Path
from lumen_resources import ResourceConfig, Downloader

# 解析配置
config = ResourceConfig.from_yaml(Path("config.yaml"))

# 创建下载器
downloader = Downloader(config)

# 下载所有模型
results = downloader.download_all()

# 检查结果
for model_type, result in results.items():
    if result.success:
        print(f"Success: {model_type} -> {result.model_path}")
    else:
        print(f"Failed: {model_type} -> {result.error}")
```

## 目录结构规范

### 本地缓存结构

```
~/.lumen/
└── models/
    ├── MobileCLIP-L-14/
    │   ├── model_info.json
    │   ├── onnx/
    │   │   ├── text.onnx
    │   │   └── vision.onnx
    │   └── ImageNet_10k.npz
    ├── antelopev2/
    ├── PP-OCRv5/
    └── bioclip-2/
        ├── model_info.json
        ├── torch/
        │   └── model.pt
        └── TreeOfLife-200M.npz
```

### 远程仓库结构规范

所有模型仓库必须遵守以下结构：

```
{repo_name}/
├── model_info.json          # 必需，模型元数据
├── torch/                   # 可选
│   └── model.pt
├── onnx/                    # 可选
│   ├── text.onnx
│   └── vision.onnx
├── rknn/                    # 可选
│   ├── rk3566/
│   │   ├── text.rknn
│   │   └── vision.rknn
│   └── rk3588/
│       ├── text.rknn
│       └── vision.rknn
└── {dataset_name}.npz       # 可选
```

## model_info.json 规范

每个模型仓库根目录必须包含 `model_info.json` 文件，定义模型的元数据和支持的运行时：

```json
{
    "name": "MobileCLIP-L-14",
    "version": "1.0.0",
    "description": "MobileCLIP model for efficient CLIP inference",
    "model_type": "clip",
    "embedding_dim": 768,
    "runtimes": {
        "torch": {
            "available": true,
            "files": ["torch/model.pt"]
        },
        "onnx": {
            "available": true,
            "files": ["onnx/text.onnx", "onnx/vision.onnx"]
        },
        "rknn": {
            "available": true,
            "devices": ["rk3566", "rk3588"],
            "files": {
                "rk3566": ["rknn/rk3566/text.rknn", "rknn/rk3566/vision.rknn"],
                "rk3588": ["rknn/rk3588/text.rknn", "rknn/rk3588/vision.rknn"]
            }
        }
    },
    "datasets": {
        "ImageNet_10k": "ImageNet_10k.npz"
    },
    "metadata": {
        "license": "MIT",
        "author": "Lumilio Photos",
        "created_at": "2024-01-01"
    }
}
```
