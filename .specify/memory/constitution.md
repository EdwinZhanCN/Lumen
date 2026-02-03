# Lumen 项目 Constitution

本文宪法文档定义了 Lumen ML 微服务项目的核心开发原则、架构契约和技术标准，所有开发工作必须严格遵循。

## 核心原则

### I. 分层架构契约 (Layered Architecture Contract)

Lumen 采用严格的三层架构设计，每层都有明确的职责边界和接口规范。所有 ML 服务模块（除 lumen-resources 和 lumen-app 外）必须遵循此架构。

#### 1. Backend 层 (运行时抽象层)

**职责**：定义 ML 服务的标准接口，在特定运行时上实现模型加载和推理

**目录结构**：
```
<module>/backends/
├── base.py              # 抽象基类定义
├── onnxrt_backend.py    # ONNX Runtime 实现（必需）
├── torch_backend.py     # PyTorch 实现（可选）
├── rknn_backend.py      # RKNN 实现（可选）
├── factory.py           # Backend 工厂
└── backend_exceptions.py # 统一异常定义
```

**契约规范**：

```python
@requires:
  - 必须实现 Base<Module>Backend 抽象基类
  - 必须提供 initialize() 方法进行模型加载和设备初始化
  - 必须提供运行时信息查询接口（get_info() 或 get_runtime_info()）
  - 必须提供 is_initialized() 状态检查
  - 输入必须是标准类型 (bytes, str, np.ndarray等)
  
@returns:
  - 输出必须是标准类型 (np.ndarray, List[Tuple]等)
  - 必须保证输出格式一致性
  - 向量输出必须是 L2 归一化的 float32 类型
  - 必须提供 BackendInfo 数据类包含运行时元数据

@errors:
  - BackendNotInitializedError: 后端未初始化
  - InvalidInputError: 输入数据无效
  - InferenceError: 推理过程失败
  - ModelLoadingError: 模型加载失败
  - DeviceUnavailableError: 设备不可用

@mandatory:
  - 每个模块必须至少实现 onnxrt_backend.py（ONNX Runtime 后端）
  - torch_backend.py、rknn_backend.py 等其他后端为可选
```

**实现要求**：
- ONNX Runtime 后端必须支持多种 Execution Provider（CPU、CUDA、CoreML、OpenVINO 等）
- 必须自动检测模型精度（FP32/FP16）并优先使用合适版本
- 必须支持批处理（batch processing）以提高吞吐量
- 必须正确处理动态和静态 batch 维度

#### 2. Model Manager 层 (业务逻辑层)

**职责**：封装业务逻辑，管理模型特定数据和配置，为特定模型准备数据、管理缓存

**目录结构**：
```
<module>/general_<task>/
├── <task>_model.py      # Model Manager 实现
└── <task>_service.py    # gRPC Service 实现
```

**契约规范**：

```python
@requires:
  - 必须接收一个 Backend 实例（通过依赖注入）
  - 必须接收一个 ModelResources 实例（来自 lumen-resources）
  - 必须提供 initialize() 方法
  - 必须管理模型特定的标签/配置/缓存
  - 必须处理数据预处理/后处理的业务逻辑
  
@returns:
  - 业务级别的结果 (分类结果、嵌入向量等)
  - 模型元数据信息（ModelInfo）
  - 必须返回标准化的响应格式（使用 lumen-resources 的 Schema）

@errors:
  - ModelDataNotFoundError: 模型数据缺失
  - CacheCorruptionError: 缓存数据损坏
  - LabelMismatchError: 标签不匹配

@responsibilities:
  - 封装业务逻辑（如分类阈值、场景提示词等）
  - 管理预计算的嵌入和标签
  - 提供高层次的推理接口（如 classify_image、detect_faces）
  - 处理批量推理和结果聚合
```

**实现要求**：
- 必须使用依赖注入接收 Backend 实例
- 必须使用 lumen-resources 提供的 ModelResources
- 必须提供 is_initialized 属性检查初始化状态
- 必须在 initialize() 中初始化 Backend
- 必须提供 info() 或 get_info() 方法返回模型元数据

#### 3. Service 层 (API 协议层)

**职责**：处理协议转换、请求路由、输入验证、任务注册

**目录结构**：
```
<module>/general_<task>/
├── <task>_service.py       # gRPC Service 实现
└── <module>/
    ├── proto/              # Protocol Buffers 定义
    │   └── ml_service.proto
    ├── registry.py         # 任务注册表
    ├── server.py           # gRPC 服务器启动
    └── models.py           # 数据模型定义
```

**契约规范**：

```python
@requires:
  - 必须实现 InferenceServicer 接口（来自 ml_service.proto）
  - 必须接收一个 Model Manager 实例
  - 必须使用 TaskRegistry 管理任务路由
  - 必须验证输入协议格式和 MIME 类型
  
@returns:
  - 标准化的 gRPC 响应（InferResponse）
  - 统一的错误格式（Error 消息）
  - 服务能力描述（Capability 消息）

@errors:
  - InvalidRequestError: 请求格式无效
  - ServiceUnavailableError: 服务未就绪
  - TimeoutError: 请求超时

@methods:
  - Infer(): 双向流式推理 RPC（必需）
  - GetCapabilities(): 获取服务能力（必需）
  - StreamCapabilities(): 流式返回所有能力（必需）
  - Health(): 健康检查（必需）

@task_registration:
  - 必须使用 TaskRegistry 注册所有任务
  - 每个任务必须提供：name、handler、description、input_mimes、output_mime
```

**实现要求**：
- 必须支持双向流式 RPC（bidirectional streaming）
- 必须支持数据块重组（chunk reassembly）
- 必须使用 correlation_id 进行请求追踪
- 必须在响应中包含 lat_ms 元数据
- 必须使用 lumen-resources 定义的响应 Schema（EmbeddingV1、LabelsV1、FaceV1 等）

#### 4. 通信协议层 (Protocol Layer)

**职责**：定义统一的 gRPC 通信协议和消息格式

**协议定义**：
- 所有模块必须使用统一的 `ml_service.proto`
- 必须支持以下消息类型：
  - `InferRequest`: 推理请求
  - `InferResponse`: 推理响应
  - `Capability`: 服务能力描述
  - `IOTask`: 任务 I/O 描述
  - `Error`: 错误信息

**核心要求**：
```protobuf
service Inference {
  rpc Infer(stream InferRequest) returns (stream InferResponse);
  rpc GetCapabilities(google.protobuf.Empty) returns (Capability);
  rpc StreamCapabilities(google.protobuf.Empty) returns (stream Capability);
  rpc Health(google.protobuf.Empty) returns (google.protobuf.Empty);
}
```

**MIME 类型约定**：
- 输入：`image/jpeg`, `image/png`, `text/plain`, `application/json`
- 输出：`application/json;schema=embedding_v1`, `application/json;schema=labels_v1`, `application/json;schema=face_v1`

### II. 依赖方向原则 (Dependency Direction)

**单向依赖流**：Service → Model Manager → Backend → lumen-resources

```
┌─────────────────────────────────────────────────────────────┐
│  Service Layer (API 协议层)                                  │
│  - Protocol Buffers 定义                                     │
│  - gRPC 服务实现                                             │
│  - 任务路由和注册                                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Model Manager Layer (业务逻辑层)                            │
│  - 业务逻辑封装                                              │
│  - 数据管理（标签、缓存、嵌入）                              │
│  - 高层次推理接口                                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Backend Layer (运行时抽象层)                                │
│  - Base Backend 定义                                         │
│  - ONNX Runtime 实现（必需）                                 │
│  - PyTorch/RKNN 等实现（可选）                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  lumen-resources (基础设施层)                                │
│  - 配置管理 (YAML)                                           │
│  - 模型下载 (HuggingFace/ModelScope)                        │
│  - 响应 Schema 定义                                          │
└─────────────────────────────────────────────────────────────┘
```

**依赖规则**：
- 上层可以依赖下层，下层不能依赖上层
- 禁止跨层依赖（如 Service 直接依赖 Backend Implementation）
- 所有模块都可以依赖 lumen-resources

### III. 数据类型转换原则 (Data Type Transformation)

**越底层越通用，越上层越业务化**

| 层级 | 输入类型 | 输出类型 | 说明 |
|------|---------|---------|------|
| **Backend** | bytes, str, np.ndarray | np.ndarray (float32, L2归一化) | 使用标准 NumPy 类型，确保 L2 归一化 |
| **Model Manager** | bytes, str | 业务对象 (List[Tuple], Schema) | 返回业务级别的分类结果、标签等 |
| **Service** | InferRequest (protobuf) | InferResponse (protobuf) | 使用 Protocol Buffers 消息 |
| **lumen-resources** | YAML, bytes | Pydantic Models, Schema | 提供配置管理和响应 Schema |

**类型规范**：
- Backend 层必须返回 `np.ndarray[np.float32]`
- 所有向量输出必须 L2 归一化（norm = 1.0）
- Model Manager 必须使用 lumen-resources 定义的 Schema（如 EmbeddingV1、LabelsV1）
- Service 层必须将 Schema 序列化为 JSON 返回

### IV. 错误处理原则 (Error Handling)

**分层错误处理，向上传递**

```python
# Backend 层错误
class BackendError(Exception): pass
class BackendNotInitializedError(BackendError): pass
class InvalidInputError(BackendError): pass
class InferenceError(BackendError): pass
class ModelLoadingError(BackendError): pass
class DeviceUnavailableError(BackendError): pass

# Model Manager 层错误
class ModelDataNotFoundError(Exception): pass
class CacheCorruptionError(Exception): pass
class LabelMismatchError(Exception): pass

# Service 层错误（映射到 gRPC）
# 内部错误 → Error 消息（protobuf）
```

**错误处理规则**：
- 每层只处理本层的错误，向上层传递底层错误
- Backend 错误应该被捕获并转换为适当的 gRPC 错误码
- Service 层必须使用 `Error` 消息返回错误信息
- 禁止在上层处理底层实现细节的错误

**gRPC 错误码映射**：
- `ERROR_CODE_INVALID_ARGUMENT`: 输入验证失败
- `ERROR_CODE_UNAVAILABLE`: 服务/设备不可用
- `ERROR_CODE_DEADLINE_EXCEEDED`: 请求超时
- `ERROR_CODE_INTERNAL`: 内部错误

### V. 扩展性原则 (Extensibility)

**新的 ML 服务只需实现新的 Backend + Model Manager + Service**

**扩展新服务的步骤**：
1. 在 `<new-module>/backends/` 实现 `Base<New>Backend` 和 `ONNXRTBackend`
2. 在 `<new-module>/general_<task>/` 实现 `<Task>ModelManager`
3. 在 `<new-module>/general_<task>/` 实现 `<Task>Service` 并注册任务
4. 复用统一的 `ml_service.proto`
5. 使用 lumen-resources 管理配置和资源

**接口稳定性**：
- 公共接口一旦发布，必须保持向后兼容
- 如需破坏性变更，必须增加主版本号
- 废弃的接口必须保留至少一个次要版本

## 技术栈要求

### 依赖管理

**工具链**：
- **包管理**：使用 `uv` 和 `pyproject.toml` 管理所有依赖
- **工作区**：使用 uv workspace 管理 lumen-resources 依赖
- **禁止**：不允许使用 pip、poetry 或其他工具管理依赖

**依赖声明**（pyproject.toml）：
```toml
[project]
dependencies = [
    "lumen-resources",  # Workspace 依赖
    "grpcio>=1.76.0",
    "protobuf>=6.33.0",
    "numpy>=1.26.0",
    # ... 其他依赖
]

[tool.uv.sources]
lumen-resources = { workspace = true }
```

**可选依赖分组**：
- `cpu`: ONNX Runtime CPU 版本
- `cuda`: ONNX Runtime GPU 版本（CUDA 12.6）
- `apple`: ONNX Runtime CoreML 版本（Apple Silicon）
- `openvino`: ONNX Runtime OpenVINO 版本
- `torch`: PyTorch 和相关依赖
- `rknn`: RKNN Toolkit Lite2

### 代码质量检查

**强制要求**：
- **Lint 检查**：每次编码后必须运行 `ruff check`（已全局安装）
- **类型检查**：每次编码后必须运行 `ty check`（已全局安装）进行严格类型检查
- **格式化**：建议使用 `ruff format` 格式化代码
- **代码提交前**：必须通过所有检查

**Ruff 配置**（pyproject.toml）：
```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP"]
ignore = ["E501", "B008", "C901"]

[tool.ruff.per-file-ignores]
"*_pb2.py" = ["ALL"]
"*_pb2_grpc.py" = ["ALL"]
```

### 通信协议

**统一协议**：
- **协议定义**：Protocol Buffers 3（proto3）
- **RPC 框架**：gRPC >= 1.76.0
- **协议文件**：每个模块的 `proto/ml_service.proto`
- **代码生成**：使用 protoc 生成 Python 代码

**核心特性**：
- 双向流式 RPC（bidirectional streaming）
- 数据块传输支持（chunking）
- 服务发现和能力查询（Capability）
- 健康检查（Health check）

### 支持的运行时

**必需运行时**：
- **ONNX Runtime**（onnxruntime >= 1.16.0）：所有模块必须支持

**可选运行时**：
- **PyTorch**（torch >= 2.6.0）：仅在需要时支持
- **RKNN**（rknn-toolkit-lite2）：仅在需要时支持（Linux + Rockchip NPU）

**执行提供者（Execution Providers）**：
- CPUExecutionProvider（必需）
- CUDAExecutionProvider（NVIDIA GPU）
- CoreMLExecutionProvider（Apple Silicon）
- OpenVINOExecutionProvider（Intel NPU）
- DmlExecutionProvider（DirectML）

## 开发规范

### 模块职责单一性

- 每个模块只做一件事（如 lumen-clip 只做 CLIP 嵌入和分类）
- 一个模块不应该包含多个不相关的功能
- 功能复杂时考虑拆分为多个子模块（如 general_clip、expert_bioclip、unified_smartclip）

### 接口稳定性

- 公共接口一旦发布，必须保持向后兼容
- Base Backend 的抽象方法不能随意修改
- 如需破坏性变更，必须增加主版本号
- 废弃的接口必须保留至少一个次要版本

### 可测试性

- 每个模块必须包含单元测试
- 测试应该覆盖所有公开接口
- 测试应该独立运行，不依赖外部服务
- 使用 pytest 框架进行测试

### 可替换性

- 实现应该基于接口而非具体实现
- 通过依赖注入实现松耦合
- 任何符合契约的 Backend 都应该可以替换
- Model Manager 不应该依赖具体的 Backend 实现

### 文档要求

- 所有公开模块必须有 docstring
- 使用 Google 风格的 docstring
- 复杂的算法必须有注释说明
- README.md 必须包含使用示例

## 性能标准

### 推理性能

- **单次推理**：典型延迟 < 100ms（CPU），< 50ms（GPU/NPU）
- **批处理**：必须支持批处理以提高吞吐量
- **内存使用**：推理期间内存峰值应该合理
- **并发处理**：服务必须支持并发请求

### 资源管理

- **模型加载**：初始化时间 < 10s（CPU），< 15s（GPU）
- **内存占用**：模型加载后内存占用应该稳定
- **设备利用**：应该充分利用硬件加速（GPU/NPU）
- **资源释放**：close() 方法必须正确释放资源

### 优化要求

- 必须使用批处理减少推理次数
- 必须预计算和缓存标签嵌入
- 必须使用向量化操作（NumPy/ONNX）
- 避免不必要的数据复制和类型转换

## 安全要求

### 输入验证

- **Service 层验证**：必须验证输入格式和 MIME 类型
- **Backend 层验证**：必须验证输入数据的有效性
- **大小限制**：必须限制输入数据的大小
- **类型检查**：必须检查输入数据的类型

### 错误处理

- **异常捕获**：必须捕获所有异常并转换为适当的错误响应
- **敏感信息**：禁止在日志和错误消息中暴露敏感信息
- **堆栈跟踪**：堆栈跟踪只应该在 detail 字段中返回
- **错误码**：必须使用统一的错误码

### 资源保护

- **并发限制**：必须限制并发请求数量
- **超时控制**：必须设置推理超时
- **资源清理**：必须正确释放模型和设备资源
- **内存泄漏**：必须避免内存泄漏

## lumen-app 应用层规范

lumen-app 是 Lumen 项目的桌面管理应用（GUI），负责服务安装、配置、启动和监控。作为应用层项目，它不遵循 ML 服务模块的三层架构契约，但必须遵守以下规范。

### I. 架构设计原则

#### 1. 职责边界

**核心职责**：
- **服务生命周期管理**：安装、配置、启动、停止、监控 ML 服务
- **用户界面**：提供直观的桌面应用界面（基于 Flet）
- **配置管理**：管理设备配置、服务配置、环境配置
- **依赖管理**：处理 Python 依赖、系统依赖的安装和解析
- **服务发现**：通过 mDNS 发现和管理本地服务实例

**不负责**：
- 不实现任何 ML 推理逻辑
- 不直接处理模型加载和执行
- 不实现 gRPC 服务端功能（仅作为客户端调用）

#### 2. 技术栈契约

**必需技术栈**：
```toml
[project]
dependencies = [
    "flet[all]>=0.28.3",      # UI 框架
    "lumen-resources",        # Workspace 依赖
    "colorlog>=6.10.1",       # 日志
    "python-i18n>=0.3.9",     # 国际化
    "pyyaml>=6.0.3",          # 配置文件解析
]

[tool.uv.sources]
lumen-resources = { workspace = true }
```

**禁止使用的技术**：
- 禁止使用其他 UI 框架（如 PyQt、Tkinter）
- 禁止直接管理 ML 模型文件
- 禁止绕过 lumen-resources 直接访问服务配置

#### 3. 目录结构规范

```
lumen-app/src/lumen_app/
├── main.py                 # 应用入口点
├── ui/                     # UI 层
│   ├── app.py             # 主应用布局和导航
│   ├── views/             # 视图组件
│   │   ├── installer_view.py      # 安装向导视图
│   │   ├── runner_view.py         # 服务运行器视图
│   │   └── monitor_view.py        # 监控仪表板视图
│   ├── components/        # 可复用 UI 组件
│   ├── locales/           # 国际化翻译文件
│   └── i18n_manager.py    # 国际化管理器
├── core/                   # 核心业务逻辑
│   ├── config.py          # 配置管理（设备配置、服务配置）
│   ├── service.py         # 服务管理（启动、停止、状态查询）
│   ├── router.py          # 服务路由和通信
│   └── loader.py          # 动态加载器和模块发现
├── utils/                  # 工具模块
│   ├── installation/      # 安装工具（依赖解析、环境配置）
│   ├── package_resolver.py # 包解析器
│   └── logger.py          # 日志配置
└── proto/                  # gRPC 协议定义（从服务模块复制）
    └── ml_service.proto
```

### II. 分层架构（应用层）

lumen-app 采用简化的事件驱动架构，分为四层：

#### 1. UI 层（Presentation Layer）

**职责**：渲染界面、处理用户交互、展示状态

**组件**：
- **Views**：独立的功能视图（Installer、Runner、Monitor）
- **Components**：可复用的 UI 组件（按钮、卡片、表单）
- **I18n Manager**：国际化管理和动态切换

**契约规范**：
```python
@requires:
  - 必须使用 Flet 框架构建 UI
  - 必须支持国际化（i18n）
  - 必须实现响应式布局
  - 必须提供 loading 状态反馈
  
@returns:
  - Flet 控件树
  - 用户交互事件回调

@prohibitions:
  - 禁止在 UI 层直接调用 gRPC 服务
  - 禁止在 UI 层处理配置数据
```

**实现要求**：
- 每个 View 必须是一个独立的函数或类
- View 必须通过回调与 Core 层交互
- 必须支持语言热切换（不重启应用）
- 必须提供友好的错误提示

#### 2. Core 层（Business Logic Layer）

**职责**：业务逻辑、状态管理、服务编排

**模块**：
- **Config Manager**：管理所有配置（设备、服务、环境）
- **Service Manager**：管理服务生命周期（启动、停止、重启）
- **Router**：服务发现和 gRPC 客户端管理
- **Loader**：动态加载服务模块和插件

**契约规范**：
```python
@requires:
  - 必须使用 lumen-resources 的 LumenConfig
  - 必须通过 Router 与 ML 服务通信
  - 必须管理应用状态（State）
  
@returns:
  - 业务数据对象
  - 服务状态信息
  - 配置对象

@responsibilities:
  - 封装复杂的业务逻辑
  - 提供统一的错误处理
  - 管理全局状态
  - 协调多个服务调用
```

**实现要求**：
- 必须使用 dataclass 定义配置对象
- 必须提供异步接口（避免阻塞 UI）
- 必须实现状态缓存和更新机制
- 必须处理服务发现和连接管理

#### 3. Utils 层（Utility Layer）

**职责**：通用工具函数、辅助功能

**模块**：
- **Installation Utils**：依赖解析、环境配置、安装器
- **Package Resolver**：Python 包和依赖关系解析
- **Logger**：日志配置和管理

**契约规范**：
```python
@requires:
  - 必须是无状态的（纯函数）
  - 必须可独立测试
  
@returns:
  - 工具函数返回值
  - 辅助数据结构

@prohibitions:
  - 禁止依赖 UI 层
  - 禁止直接调用 gRPC 服务
```

**实现要求**：
- 必须提供清晰的函数签名
- 必须包含完整的类型注解
- 必须处理边界情况和错误输入
- 必须提供文档字符串

#### 4. Protocol 层（通信协议层）

**职责**：定义与 ML 服务通信的协议

**内容**：
- Protocol Buffers 定义（.proto 文件）
- 生成的 Python gRPC 代码

**契约规范**：
```python
@requires:
  - 必须与 ML 服务模块使用相同的 proto 定义
  - 必须使用生成的 gRPC 客户端代码
  
@prohibitions:
  - 禁止修改 proto 定义（必须从服务模块复制）
  - 禁止直接使用原始 HTTP 或 TCP 通信
```

### III. 数据流原则

**单向数据流**：UI → Core → Protocol → ML Services

```
┌─────────────────────────────────────────────────────────────┐
│  UI Layer (Flet Views)                                      │
│  - 用户交互触发事件                                          │
│  - 显示服务状态和配置信息                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Core Layer (Business Logic)                                │
│  - Config Manager: 管理配置对象                             │
│  - Service Manager: 控制服务生命周期                        │
│  - Router: 服务发现和 gRPC 客户端                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Protocol Layer (gRPC Client)                               │
│  - ml_service.proto 定义                                    ���
│  - 生成的 gRPC 客户端代码                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  ML Services (lumen-clip, lumen-vlm, etc.)                  │
│  - 实现三层架构的服务                                       │
│  - 提供 gRPC 服务端                                        │
└─────────────────────────────────────────────────────────────┘
```

**数据流规则**：
- UI 只能通过 Core 层访问服务
- Core 层只能通过 Protocol 层与 ML 服务通信
- 禁止反向依赖（Core 不能依赖 UI 的具体实现）
- 状态更新必须通过事件通知机制

### IV. 配置管理规范

#### 1. 配置层次

**全局配置**：
- 应用设置（语言、主题、日志级别）
- 用户偏好（默认设备、默认服务）

**设备配置**：
- 硬件平台配置（Apple Silicon、NVIDIA GPU、Rockchip NPU）
- 运行时配置（ONNX、PyTorch、RKNN）
- 依赖元数据（Python 版本、额外依赖、PyPI 索引）

**服务配置**：
- 服务实例配置（端口、模型路径、批处理大小）
- 环境变量配置
- 启动参数配置

#### 2. 配置存储

**存储位置**：
```
~/.lumen/
├── config/
│   ├── devices.yaml        # 设备配置
│   ├── services.yaml       # 服务配置
│   └── app.yaml            # 应用配置
├── cache/                  # 缓存目录
├── logs/                   # 日志目录
└── models/                 # 模型存储（软链接）
```

**配置格式**：
- 必须使用 YAML 格式
- 必须通过 lumen-resources 的 LumenConfig 加载
- 必须支持配置验证（Pydantic）

#### 3. 配置类设计

**DeviceConfig 类**：
```python
@dataclass
class DeviceConfig:
    """设备配置类"""
    runtime: Runtime              # 推理运行时
    onnx_providers: list | None   # ONNX 执行提供者
    batch_size: int | None        # 批处理大小
    precision: str | None         # 精度（fp32/fp16/int8）
    description: str              # 设备描述
    env: str                      # 环境配置标识
    os: str | None                # 操作系统约束
    dependency_metadata: DependencyMetadata | None  # 依赖元数据
```

**依赖元数据**：
```python
@dataclass
class DependencyMetadata:
    """Python 依赖元数据"""
    extra_index_url: list[str] | None  # 额外的 PyPI 索引
    extra_deps: list[str] | None       # 可选依赖标识符
    python_version: str                # Python 版本要求
    install_args: list[str] | None     # 额外的安装参数
```

### V. 服务管理规范

#### 1. 服务生命周期

**安装阶段**：
1. 检测设备和环境
2. 解析依赖关系
3. 安装 Python 包（使用 uv）
4. 配置服务参数
5. 验证安装结果

**启动阶段**：
1. 加载服务配置
2. 启动服务进程
3. 等待服务就绪（健康检查）
4. 注册服务发现（mDNS）
5. 更新服务状态

**运行阶段**：
1. 监控服务健康状态
2. 收集服务指标
3. 处理服务请求（通过 gRPC）
4. 记录服务日志

**停止阶段**：
1. 发送停止信号
2. 等待服务优雅关闭
3. 清理资源
4. 更新服务状态

#### 2. 服务发现

**mDNS 协议**：
- 服务启动后必须注册 mDNS
- 服务名称格式：`lumen-{service-name}._tcp.local.`
- 必须包含服务元数据（端口、能力、设备类型）

**服务查询**：
- 支持浏览所有本地 Lumen 服务
- 支持按服务类型过滤
- 支持按设备类型过滤

#### 3. 服务监控

**监控指标**：
- 服务状态（运行/停止/错误）
- 服务健康度（健康检查响应时间）
- 资源使用（CPU、内存、GPU）
- 请求统计（请求数、成功率、延迟）

**监控方式**：
- 定期健康检查（gRPC Health RPC）
- mDNS 服务状态监听
- 日志文件监控

### VI. 用户界面规范

#### 1. 视图设计

**Runner 视图**（服务运行器）：
- 显示所有可用的 Lumen 服务
- 提供启动/停止控制
- 显示服务状态和配置
- 提供服务配置编辑

**Monitor 视图**（监控仪表板）：
- 显示全局服务网格状态
- 显示服务健康度和性能指标
- 提供服务日志查看
- 提供故障诊断工具

**Installer 视图**（安装向导）：
- 引导用户完成服务安装
- 检测设备和环境
- 提供依赖安装进度
- 显示安装结果和错误

#### 2. 国际化（i18n）

**支持语言**：
- English（en）
- 中文（zh）

**实现要求**：
- 所有用户可见文本必须使用 `t()` 函数
- 翻译文件存储在 `ui/locales/` 目录
- 必须支持语言热切换
- 必须提供完整的翻译覆盖

**使用示例**：
```python
from .i18n_manager import t

title = ft.Text(t("views.runner.title"))
button = ft.ElevatedButton(t("actions.start"))
```

#### 3. 响应式设计

**布局要求**：
- 支持不同窗口大小（最小 800x600）
- 使用 Navigation Rail 进行主导航
- 内容区域自适应窗口大小
- 支持滚动和分页

**主题支持**：
- 支持浅色/深色主题（跟随系统）
- 使用统一的颜色方案
- 使用一致的组件样式

### VII. 错误处理规范

#### 1. 错误分类

**UI 层错误**：
- 用户输入验证错误
- 界面渲染错误

**Core 层错误**：
- 服务启动失败
- 服务连接失败
- 配置加载失败
- 依赖安装失败

**服务层错误**：
- gRPC 调用失败
- 服务返回错误

#### 2. 错误处理策略

**UI 层**：
- 显示友好的错误消息
- 提供重试或恢复选项
- 记录错误日志

**Core 层**：
- 捕获并包装底层错误
- 提供错误恢复机制
- 通知 UI 层更新状态
- 记录详细的错误日志

**Utils 层**：
- 抛出明确的异常类型
- 提供错误上下文信息

#### 3. 日志规范

**日志级别**：
- DEBUG：详细的调试信息
- INFO：一般信息（启动、停止）
- WARNING：警告信息（非致命错误）
- ERROR：错误信息（服务失败）
- CRITICAL：严重错误（应用崩溃）

**日志存储**：
- 日志文件：`~/.lumen/logs/lumen-app.log`
- 日志轮转：按大小（10MB）或日期
- 日志格式：包含时间戳、级别、模块、消息

### VIII. 性能标准

#### 1. 启动性能

- 应用启动时间：< 3s
- UI 渲染时间：< 1s
- 配置加载时间：< 500ms

#### 2. 响应性能

- UI 响应时间：< 100ms（用户交互）
- 服务发现时间：< 2s（mDNS 查询）
- 健康检查时间：< 500ms（gRPC 调用）

#### 3. 资源使用

- 内存占用：< 200MB（空闲）
- CPU 使用：< 5%（空闲）
- 网络流量：< 1KB/s（空闲）

### IX. 安全要求

#### 1. 输入验证

- 必须验证所有用户输入
- 必须限制输入长度和格式
- 必须转义特殊字符

#### 2. 权限管理

- 只请求必要的系统权限
- 必须正确处理文件权限
- 必须保护敏感配置信息

#### 3. 通信安全

- gRPC 通信必须使用本地回环（localhost）
- 禁止未加密的网络通信
- 必须验证服务身份

### X. 开发规范

#### 1. 代码质量

**强制要求**：
- 每次编码后必须运行 `ruff check`
- 每次编码后必须运行 `ty check`
- 必须通过所有检查才能提交代码

**配置要求**：
```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP"]
ignore = ["E501", "B008", "C901"]
```

#### 2. 测试要求

- 必须包含单元测试
- 必须包含集成测试（UI 自动化测试）
- 测试覆盖率目标：> 70%

#### 3. 文档要求

- 所有公开模块必须有 docstring
- 复杂的业务逻辑必须有注释
- README.md 必须包含安装和使用说明

## 例外项目

以下项目不强制遵循上述 ML 服务模块的三层架构契约：
- **lumen-resources**：工具类和基础设施项目
- **lumen-app**：桌面管理应用（GUI），遵循应用层规范

## 治理 (Governance)

### 宪法效力

- 本宪法优先于所有其他开发实践
- 所有代码审查必须验证是否符合本宪法
- 任何与宪法冲突的实践必须修改

### 修订流程

- 宪法的修订需要团队讨论和批准
- 修订必须有明确的理由和影响分析
- 修订后必须更新所有相关文档和模板
- 重大修订需要通知所有开发者

### 违规处理

- 发现不符合契约的代码必须指出并要求修改
- 严重违规应该阻止代码合并
- 重复违规应该进行团队培训和流程改进
- 违规记录应该被保留以供参考

### 合规检查

- 所有 PR 必须通过 ruff 和 ty check
- 所有新模块必须遵循三层架构
- 所有新功能必须包含单元测试
- 所有变更必须更新文档

**Version**: 1.0.0  
**Ratified**: 2025-12-28  
**Last Amended**: 2025-12-28  
**Maintainer**: Lumen Development Team
