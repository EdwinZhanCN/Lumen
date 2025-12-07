## 1. Base Backend (Abstract Layer)

**Responsibilities**: Define the standard interface and data types for a class of ML services.

**Contract**:
```
@requires:
  - Inputs must use standard types (bytes, str, np.ndarray, etc.)
  - Must provide an initialization status check
  - Must expose runtime information
  
@returns:
  - Outputs must use standard types (np.ndarray, List[Tuple], etc.)
  - Must guarantee consistent output formats
  - Must provide error type definitions

@errors:
  - BackendNotInitializedError
  - InvalidInputError  
  - InferenceError
```

## 2. Backend Implementation (Concrete Implementation)

**Responsibilities**: Implement the Base Backend interface on a specific runtime.

**Contract**:
```
@requires:
  - Must implement all abstract methods of the Base Backend
  - Must handle runtime-specific resource management
  - Must provide device/precision configuration
  
@returns: 
  - Must conform to the Base Backend output specification
  - May provide runtime-specific optimization options

@errors:
  - RuntimeSpecificError (e.g. CUDA OOM)
  - ModelLoadingError
  - DeviceUnavailableError
```

## 3. Model Manager (Business Logic Layer)

**Responsibilities**: Prepare data for a specific model, manage caches, and encapsulate business logic.

**Contract**:
```
@requires:
  - Must accept a Base Backend instance
  - Must manage model-specific labels/configuration
  - Must perform data preprocessing/postprocessing
  
@returns:
  - Business-level results (classification results, embedding vectors, etc.)
  - Model metadata

@errors:
  - ModelDataNotFoundError
  - CacheCorruptionError  
  - LabelMismatchError
```

## 4. Service (API Layer)

**Responsibilities**: Handle protocol translation, request routing, and input validation.

**Contract**:
```
@requires:
  - Must accept one or more Model Manager instances
  - Must validate input protocol formats
  - Must handle concurrency and batching
  
@returns:
  - Standardized API responses
  - Unified error format
  - Service status information

@errors:
  - InvalidRequestError
  - ServiceUnavailableError
  - TimeoutError
```

## 5. Service Registry (Coordination Layer)

**Responsibilities**: Route APIs across ML services, perform service discovery, and handle load balancing.

**Contract**:
```
@requires:
  - Must register multiple different types of Services
  - Must provide a service discovery mechanism
  - Must handle cross-service calls
  
@returns:
  - Unified service endpoints
  - Service capability descriptions
  - Routing decision results

@errors:
  - ServiceNotFoundError
  - RoutingError
  - CapacityExceededError
```

## Key Design Principles

1. Dependency direction: Service -> Model Manager -> Backend (one-way dependency)
2. Data types: lower layers use more generic types; higher layers use more business-oriented types
3. Error handling: each layer should handle errors at its level and propagate lower-level errors upward
4. Extensibility: adding a new ML service only requires implementing a new Base Backend + Model Manager
