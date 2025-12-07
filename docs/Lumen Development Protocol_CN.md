## 1. Base Backend (抽象层)

**职责**：定义某类ML服务的标准接口和数据类型

**契约**：
```
@requires: 
  - 输入必须是标准类型 (bytes, str, np.ndarray等)
  - 必须提供初始化状态检查
  - 必须提供运行时信息
  
@returns:
  - 输出必须是标准类型 (np.ndarray, List[Tuple]等) 
  - 必须保证输出格式一致性
  - 必须提供错误类型定义

@errors:
  - BackendNotInitializedError
  - InvalidInputError  
  - InferenceError
```

## 2. Backend Implementation (具体实现)

**职责**：在特定运行时上实现Base Backend接口

**契约**：
```
@requires:
  - 必须实现Base Backend的所有抽象方法
  - 必须处理特定运行时的资源管理
  - 必须提供设备/精度配置
  
@returns: 
  - 必须符合Base Backend的输出规范
  - 可以提供运行时特有的优化选项

@errors:
  - RuntimeSpecificError (如CUDA OOM)
  - ModelLoadingError
  - DeviceUnavailableError
```

## 3. Model Manager (业务逻辑层)

**职责**：为特定模型准备数据、管理缓存、封装业务逻辑

**契约**：
```
@requires:
  - 必须接收一个Base Backend实例
  - 必须管理模型特定的标签/配置
  - 必须处理数据预处理/后处理
  
@returns:
  - 业务级别的结果 (分类结果、嵌入向量等)
  - 模型元数据信息

@errors:
  - ModelDataNotFoundError
  - CacheCorruptionError  
  - LabelMismatchError
```

## 4. Service (API层)

**职责**：处理协议转换、请求路由、输入验证

**契约**：
```
@requires:
  - 必须接收一个或多个Model Manager实例
  - 必须验证输入协议格式
  - 必须处理并发和批处理
  
@returns:
  - 标准化的API响应
  - 统一的错误格式
  - 服务状态信息

@errors:
  - InvalidRequestError
  - ServiceUnavailableError
  - TimeoutError
```

## 5. Service Registry (协调层)

**职责**：跨ML服务的API路由、服务发现、负载均衡

**契约**：
```
@requires:
  - 必须注册多个不同类型的Service
  - 必须提供服务发现机制
  - 必须处理跨服务调用
  
@returns:
  - 统一的服务端点
  - 服务能力描述
  - 路由决策结果

@errors:
  - ServiceNotFoundError
  - RoutingError
  - CapacityExceededError
```

## 关键设计原则

1. **依赖方向**：Service -> Model Manager -> Backend (单向依赖)
2. **数据类型**：越底层越通用，越上层越业务化
3. **错误处理**：每层只处理本层的错误，向上层传递底层错误
4. **扩展性**：新的ML服务只需实现新的Base Backend + Model Manager
