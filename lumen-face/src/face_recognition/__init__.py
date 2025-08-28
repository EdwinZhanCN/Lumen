"""
face_recognition package: lightweight facade and factory for face embedding backends.

This package provides:
- A small factory that selects and instantiates a backend based on environment variables
  or an explicit runtime argument.
- Convenience imports of common base types (RuntimeKind, BackendInfo, BaseFaceBackend).

Environment variables (optional):
- LUMEN_FACE_BACKEND or LUMEN_FACE_RUNTIME: 'onnxrt' (default) or 'rknn'
- LUMEN_FACE_DEVICE: device string (e.g., 'cpu', 'cuda:0', 'rknpu0')
- LUMEN_FACE_PACK_NAME: model pack name (if applicable)
- LUMEN_FACE_MAX_BATCH_SIZE: integer hint for batching
- LUMEN_FACE_PRECISION: preferred precision hint (e.g., 'fp16', 'int8') [optional]
- LUMEN_FACE_SUPPORTS_IMAGE_BATCH: '1'/'true' for a boolean hint [optional]
- LUMEN_FACE_EXTRA: JSON string for extra key-values [optional]

Notes:
- This package only wires the selection/factory. Individual backends should be implemented
  under the 'backends' package (e.g., backends.onnxrt, backends.rknn) and expose a class
  named 'Backend' or a more specific class (e.g., 'OnnxRuntimeFaceBackend').
- If a selected backend module/class is not available, a clear ImportError is raised.

Example:
    from face_recognition import create_backend, RuntimeKind

    backend = create_backend()  # uses env to select runtime (onnxrt default)
    # or
    backend = create_backend(runtime=RuntimeKind.RKNN, device_preference="rknpu0")

    # Use the backend...
    # embeddings = backend.embed(image, boxes=[(x1,y1,x2,y2), ...])

    backend_info = backend_info_from_env()
    print(backend_info.as_dict())
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple, Type, Union

# Re-export common types from the shared base module
from backends.base import BaseFaceBackend, BackendInfo, RuntimeKind


def _normalize_runtime(runtime: Optional[Union[str, RuntimeKind]]) -> str:
    if runtime is None:
        # Default to environment values or 'onnxrt'
        runtime = os.environ.get("LUMEN_FACE_BACKEND", os.environ.get("LUMEN_FACE_RUNTIME", "onnxrt"))
    if isinstance(runtime, RuntimeKind):
        return runtime.value
    return str(runtime).strip().lower()


def _import_backend_class(runtime: str) -> Type[BaseFaceBackend]:
    """
    Import and return the backend class for the specified runtime.

    The backend module should expose either:
      - a class named 'Backend' (preferred), or
      - a specific class (e.g., 'OnnxRuntimeFaceBackend' / 'RKNNFaceBackend').

    Raises:
        ValueError: if runtime is unknown
        ImportError: if module or class cannot be imported
    """
    if runtime == RuntimeKind.ONNXRT.value:
        module_name_candidates = [
            "backends.onnxrt",            # preferred module path
            "face_recognition.backends.onnxrt",  # fallback if nested
        ]
        class_name_candidates = ["Backend", "OnnxRuntimeFaceBackend"]
    elif runtime == RuntimeKind.RKNN.value:
        module_name_candidates = [
            "backends.rknn",
            "face_recognition.backends.rknn",
        ]
        class_name_candidates = ["Backend", "RKNNFaceBackend"]
    else:
        raise ValueError(f"Unknown runtime: {runtime!r}. Expected one of: {[e.value for e in RuntimeKind]}")

    last_import_error: Optional[Exception] = None

    for module_name in module_name_candidates:
        try:
            mod = __import__(module_name, fromlist=["*"])
        except Exception as e:
            last_import_error = e
            continue

        # Try class name candidates
        for cls_name in class_name_candidates:
            backend_cls = getattr(mod, cls_name, None)
            if backend_cls is not None:
                # Basic sanity: ensure it's a subclass of BaseFaceBackend if possible
                if isinstance(backend_cls, type):
                    return backend_cls  # type: ignore[return-value]

    # If we get here, imports failed
    if last_import_error is not None:
        raise ImportError(
            f"Failed to import backend for runtime '{runtime}'. "
            f"Tried modules: {module_name_candidates}. Last error: {last_import_error}"
        ) from last_import_error
    raise ImportError(
        f"Backend class for runtime '{runtime}' not found. "
        f"Tried modules {module_name_candidates} with classes {class_name_candidates}."
    )


def create_backend(
    runtime: Optional[Union[str, RuntimeKind]] = None,
    pack_name: Optional[str] = None,
    device_preference: Optional[str] = None,
    max_batch_size: Optional[int] = None,
    **kwargs: Any,
) -> BaseFaceBackend:
    """
    Instantiate a face backend using a selected runtime and common constructor hints.

    Args:
        runtime: 'onnxrt' | 'rknn' | RuntimeKind enum (default from env or 'onnxrt')
        pack_name: Optional model pack name (if the backend supports it)
        device_preference: Device hint (e.g., 'cpu', 'cuda:0', 'rknpu0')
        max_batch_size: Optional batch size hint
        **kwargs: Forwarded to backend constructor for backend-specific options

    Returns:
        An instance of BaseFaceBackend (backend-specific subtype).

    Raises:
        ValueError: if runtime is unknown
        ImportError: if backend class cannot be imported
        Exception: any error raised by backend constructor
    """
    rt = _normalize_runtime(runtime)
    backend_cls = _import_backend_class(rt)
    return backend_cls(
        pack_name=pack_name,
        device_preference=device_preference,
        max_batch_size=max_batch_size,
        **kwargs,
    )


def _parse_bool(s: Optional[str], default: bool = False) -> bool:
    if s is None:
        return default
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(s: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if s is None or s.strip() == "":
        return default
    try:
        return int(s.strip())
    except Exception:
        return default


def _parse_json_dict(s: Optional[str]) -> Dict[str, str]:
    if not s:
        return {}
    try:
        value = json.loads(s)
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
    except Exception:
        pass
    return {}


def backend_info_from_env() -> BackendInfo:
    """
    Build a lightweight BackendInfo using environment variables.

    This does not instantiate any backend or import heavy dependencies. It only
    reports the intended runtime/device/pack_name/etc. Backends can expose more
    accurate info via their own runtime objects if needed.
    """
    runtime = _normalize_runtime(None)
    device = os.environ.get("LUMEN_FACE_DEVICE") or None
    pack_name = os.environ.get("LUMEN_FACE_PACK_NAME") or None
    max_batch_size = _parse_int(os.environ.get("LUMEN_FACE_MAX_BATCH_SIZE"), None)
    supports_image_batch = _parse_bool(os.environ.get("LUMEN_FACE_SUPPORTS_IMAGE_BATCH"), False)

    # precision hints; some backends might support multiple precisions.
    # Accept either a single precision or a CSV list.
    precisions_env = os.environ.get("LUMEN_FACE_PRECISIONS") or os.environ.get("LUMEN_FACE_PRECISION") or ""
    precisions = [p.strip() for p in precisions_env.split(",") if p.strip()] if precisions_env else []

    extra = _parse_json_dict(os.environ.get("LUMEN_FACE_EXTRA"))

    return BackendInfo(
        runtime=runtime,
        device=device,
        pack_name=pack_name,
        face_embedding_dim=None,  # unknown here (backend-specific)
        precisions=precisions,
        max_batch_size=max_batch_size,
        supports_image_batch=supports_image_batch,
        extra=extra,
    )


__all__ = [
    # Types
    "RuntimeKind",
    "BackendInfo",
    "BaseFaceBackend",
    # Factory and helpers
    "create_backend",
    "backend_info_from_env",
]
