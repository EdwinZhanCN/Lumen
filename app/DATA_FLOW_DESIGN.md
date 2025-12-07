# Data Flow Design - Service Configuration

## 问题分析

当前存在的问题：
1. 用户选择 Device Type = Rockchip，芯片型号 = rk3588
2. 添加模型后，想改成 rk3566
3. 修改 Rockchip Model 选择器，但模型配置没有更新

根本原因：**数据流混乱，没有明确的 Single Source of Truth**

---

## 数据关系图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户界面层                             │
├─────────────────────────────────────────────────────────────┤
│ ▶ Device Type Selector                                      │
│   ├─ NVIDIA GPU (CUDA)                                      │
│   ├─ Apple Silicon (M1/M2/M3)                               │
│   ├─ CPU (Generic)                                          │
│   ├─ Intel GPU (OpenVINO)                                   │
│   └─ Rockchip NPU  ◀── 选择这个时                           │
│       └─ ▶ Rockchip Chip Model Selector                     │
│           ├─ rk3566                                          │
│           └─ rk3588                                          │
│                                                              │
│ ▶ Models Section                                            │
│   ├─ Add Model (alias + model name)                         │
│   └─ Edit Model (只能改 model name)                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    数据存储层 (Config)                        │
├─────────────────────────────────────────────────────────────┤
│ services:                                                    │
│   [serviceName]:                                             │
│     backend_settings:                                        │
│       device: null | string                                  │
│       batch_size: number                                     │
│       onnx_providers: string[] | null                        │
│                                                              │
│     models:                                                  │
│       [alias]:                                               │
│         model: string        ◀── 用户可以修改                │
│         runtime: "onnx"|"rknn" ◀── 由 Device Type 决定       │
│         rknn_device?: string   ◀── 由 Rockchip Model 决定   │
└─────────────────────────────────────────────────────────────┘
```

---

## 设计方案：UI State 作为 Single Source of Truth

### 核心思想

**DeviceSelection (UI State)** 是用户选择的直接表现，是 Single Source of Truth。

```typescript
interface DeviceSelection {
  type: DeviceType; // "nvidia" | "apple" | "cpu" | "intel-gpu" | "rockchip"
  rockchipModel?: RockchipModel; // "rk3566" | "rk3588" (仅当 type="rockchip")
}
```

### 数据流向（单向）

```
用户操作
   ↓
更新 deviceSelection (setState)
   ↓
同步到 backend_settings
   ↓
同步到所有 models
   ↓
更新 config (onChange)
   ↓
渲染 UI
```

---

## 实现细节

### 1. 初始化 DeviceSelection

```typescript
// 组件加载时，从 config 推导初始的 deviceSelection
const [deviceSelection, setDeviceSelection] = useState<DeviceSelection | null>(() => {
  return deriveDeviceSelectionFromConfig(serviceConfig);
});

function deriveDeviceSelectionFromConfig(config): DeviceSelection | null {
  const bs = config.backend_settings;
  
  // 检查是否是 Rockchip (从 models 判断)
  const firstModel = Object.values(config.models || {})[0];
  if (firstModel?.runtime === "rknn") {
    return {
      type: "rockchip",
      rockchipModel: firstModel.rknn_device
    };
  }
  
  // 从 onnx_providers 推导
  if (!bs?.onnx_providers || bs.onnx_providers.length === 0) {
    return null; // 未配置
  }
  
  const providers = bs.onnx_providers;
  if (providers.includes("CUDAExecutionProvider")) return { type: "nvidia" };
  if (providers.includes("CoreMLExecutionProvider")) return { type: "apple" };
  if (providers.includes("OpenVINOExecutionProvider")) return { type: "intel-gpu" };
  if (providers.includes("CPUExecutionProvider") && providers.length === 1) return { type: "cpu" };
  
  return null;
}
```

### 2. 改变 Device Type

```typescript
function handleDeviceTypeChange(newSelection: DeviceSelection) {
  // 1. 更新 UI state
  setDeviceSelection(newSelection);
  
  // 2. 同步 backend_settings
  const backendUpdates = getBackendSettingsForDeviceType(newSelection.type);
  onBackendSettingsChange(backendUpdates);
  
  // 3. 同步所有 models
  syncAllModelsWithDeviceSelection(newSelection);
}
```

### 3. 改变 Rockchip Model（关键！）

```typescript
function handleRockchipModelChange(model: RockchipModel) {
  // 1. 更新 UI state
  const newSelection = { type: "rockchip", rockchipModel: model };
  setDeviceSelection(newSelection);
  
  // 2. backend_settings 不需要改（onnx_providers 还是 null）
  
  // 3. 更新所有 models 的 rknn_device
  Object.keys(serviceConfig.models || {}).forEach((alias) => {
    const currentModel = serviceConfig.models[alias];
    onModelChange(alias, {
      ...currentModel,
      rknn_device: model
    });
  });
}
```

### 4. 添加/编辑 Model

```typescript
function handleSaveModel(alias: string, modelName: string) {
  if (!deviceSelection) {
    alert("Please select device type first");
    return;
  }
  
  // 从当前 deviceSelection 推导 runtime 和 rknn_device
  const runtime = deviceSelection.type === "rockchip" ? "rknn" : "onnx";
  
  const modelConfig: any = {
    model: modelName,
    runtime: runtime
  };
  
  if (runtime === "rknn") {
    if (!deviceSelection.rockchipModel) {
      alert("Please select Rockchip chip model first");
      return;
    }
    modelConfig.rknn_device = deviceSelection.rockchipModel;
  }
  
  onAddModel(alias, modelConfig); // 或 onModelChange
}
```

---

## 关键点

### ✅ 单一数据源

- **deviceSelection (state)** 是用户选择的唯一来源
- config 中的 backend_settings 和 models 是 deviceSelection 的**派生数据**
- 任何修改都通过 setDeviceSelection 触发

### ✅ 同步机制

- 不使用 useEffect 自动同步（避免循环和依赖问题）
- 在每个修改点**主动同步** backend_settings 和 models
- 同步函数要**幂等**，多次调用结果一致

### ✅ 数据一致性保证

```typescript
// 核心原则：
// 1. deviceSelection.type 决定 runtime
deviceSelection.type === "rockchip" → runtime = "rknn"
deviceSelection.type !== "rockchip" → runtime = "onnx"

// 2. deviceSelection.type 决定 onnx_providers
deviceSelection.type === "rockchip" → onnx_providers = null
deviceSelection.type === "nvidia" → onnx_providers = [CUDA, CPU]
// ... 其他类型

// 3. deviceSelection.rockchipModel 决定 rknn_device
deviceSelection.rockchipModel → 所有 models 的 rknn_device
```

### ✅ 边界情况

1. **初始化时 config 为空**
   - deviceSelection = null
   - 显示 "Select your device..." 提示

2. **添加第一个 service**
   - deviceSelection = null
   - 用户必须先选择 Device Type

3. **Rockchip 未选择芯片型号**
   - 显示选择器，但显示警告
   - 不允许添加模型

4. **切换 Device Type**
   - 清除 rockchipModel (如果不是 rockchip)
   - 更新所有 models

5. **Advanced Settings 启用时**
   - Device Type 选择器禁用
   - 但 deviceSelection 状态保持，用于显示

---

## 实现步骤

### Step 1: 重构 deviceSelection 为组件 state
- 移除 deriveDeviceSelection()，改为初始化 state
- 所有地方使用 state，不再动态推导

### Step 2: 实现同步函数
```typescript
function syncBackendSettings(selection: DeviceSelection) { ... }
function syncAllModels(selection: DeviceSelection) { ... }
```

### Step 3: 在所有修改点调用同步
- handleDeviceTypeChange
- handleRockchipModelChange (新增)
- handleSaveModel

### Step 4: 移除 useEffect 自动同步
- 删除可能导致循环的 useEffect
- 所有同步都是主动触发

### Step 5: 测试场景
- ✅ 添加 service，选择 Rockchip，选择 rk3588，添加模型
- ✅ 改变 Rockchip Model 为 rk3566，检查所有模型的 rknn_device
- ✅ 切换到 NVIDIA，检查 runtime 变为 onnx
- ✅ 切换回 Rockchip，选择 rk3566，检查恢复正确

---

## 预期行为

### 场景：修改 Rockchip Model

```
用户操作：
1. Device Type = Rockchip NPU
2. Rockchip Model = rk3588
3. 添加 model: general → buffalo_l

Config:
backend_settings:
  device: null
  onnx_providers: null
models:
  general:
    model: buffalo_l
    runtime: rknn
    rknn_device: rk3588

--- 用户修改 ---
Rockchip Model 选择器改为 rk3566

预期结果：
models:
  general:
    model: buffalo_l
    runtime: rknn
    rknn_device: rk3566  ← 自动更新
```

---

## 总结

- **Single Source of Truth**: deviceSelection (UI state)
- **数据流**: 单向，从 deviceSelection → config
- **同步**: 主动触发，不依赖 useEffect
- **一致性**: 所有 models 的 runtime 和 rknn_device 始终与 deviceSelection 一致