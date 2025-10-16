"""
backend_factory.py

Backend factory for creating CLIP backend instances according to Lumen architecture.
This module separates backend creation logic from Model Manager layer,
ensuring proper separation of concerns.
"""

import os
from typing import Optional

from backends import BaseClipBackend, TorchBackend, ONNXRTBackend, RKNNBackend


class BackendCreationError(Exception):
    """Raised when backend creation fails."""

    pass


def create_clip_backend(
    model_name: str,
    pretrained: Optional[str] = None,
    backend_type: Optional[str] = None,
    device_preference: Optional[str] = None,
    max_batch_size: int = 512,
    **kwargs,
) -> BaseClipBackend:
    """
    Create a CLIP backend instance based on configuration.

    Args:
        model_name: Model architecture name
        pretrained: Pretrained weights identifier
        backend_type: Backend type ('torch', 'onnxrt', 'rknn')
        device_preference: Preferred device (e.g., 'cuda', 'cpu')
        max_batch_size: Maximum batch size for processing
        **kwargs: Additional backend-specific parameters

    Returns:
        Configured backend instance

    Raises:
        BackendCreationError: If backend creation fails
    """
    # Determine backend type from environment or parameter
    backend_type = backend_type or os.getenv("CLIP_BACKEND", "torch")
    backend_type = backend_type.lower()

    # Get device preference from environment if not specified
    device_preference = device_preference or os.getenv("CLIP_DEVICE")

    try:
        if backend_type == "torch":
            return TorchBackend(
                model_name=model_name,
                pretrained=pretrained,
                device_preference=device_preference,
                max_batch_size=max_batch_size,
                **kwargs,
            )

        elif backend_type == "onnxrt":
            # ONNX Runtime specific configuration
            onnx_image = kwargs.get("onnx_image_path") or os.getenv("CLIP_ONNX_IMAGE")
            onnx_text = kwargs.get("onnx_text_path") or os.getenv("CLIP_ONNX_TEXT")
            providers = kwargs.get("providers") or os.getenv("CLIP_ORT_PROVIDERS")

            providers_list: Optional[list[str]] = None
            if providers:
                providers_list = [p.strip() for p in providers.split(",")]

            return ONNXRTBackend(
                model_name=model_name,
                pretrained=pretrained,
                onnx_image_path=onnx_image,
                onnx_text_path=onnx_text,
                providers=providers_list,
                device_preference=device_preference,
                max_batch_size=max_batch_size,
                **kwargs,
            )

        elif backend_type == "rknn":
            # RKNN specific configuration
            rknn_path = kwargs.get("rknn_model_path") or os.getenv("CLIP_RKNN_MODEL")
            target = kwargs.get("target") or os.getenv("CLIP_RKNN_TARGET", "rk3588")

            return RKNNBackend(
                model_name=model_name,
                pretrained=pretrained,
                rknn_model_path=rknn_path,
                target=target,
                device_preference=device_preference,
                max_batch_size=max_batch_size,
                **kwargs,
            )

        else:
            raise BackendCreationError(f"Unsupported backend type: {backend_type}")

    except Exception as e:
        raise BackendCreationError(
            f"Failed to create {backend_type} backend: {e}"
        ) from e


def create_bioclip_backend(
    model: str = "hf-hub:imageomics/bioclip-2",
    backend_type: Optional[str] = None,
    device_preference: Optional[str] = None,
    max_batch_size: int = 512,
    **kwargs,
) -> BaseClipBackend:
    """
    Create a BioCLIP backend instance based on configuration.

    Args:
        model: BioCLIP model identifier
        backend_type: Backend type ('torch', 'onnxrt', 'rknn')
        device_preference: Preferred device (e.g., 'cuda', 'cpu')
        max_batch_size: Maximum batch size for processing
        **kwargs: Additional backend-specific parameters

    Returns:
        Configured backend instance

    Raises:
        BackendCreationError: If backend creation fails
    """
    # Determine backend type from environment or parameter
    backend_type = backend_type or os.getenv("BIOCLIP_BACKEND", "torch")
    backend_type = backend_type.lower()

    # Get device preference from environment if not specified
    device_preference = device_preference or os.getenv("BIOCLIP_DEVICE")

    try:
        if backend_type == "torch":
            return TorchBackend(
                model_name=model,
                pretrained=None,  # BioCLIP uses specific model IDs
                device_preference=device_preference,
                max_batch_size=max_batch_size,
                **kwargs,
            )

        elif backend_type == "onnxrt":
            # ONNX Runtime specific configuration
            onnx_image = kwargs.get("onnx_image_path") or os.getenv(
                "BIOCLIP_ONNX_IMAGE"
            )
            onnx_text = kwargs.get("onnx_text_path") or os.getenv("BIOCLIP_ONNX_TEXT")
            providers = kwargs.get("providers") or os.getenv("BIOCLIP_ORT_PROVIDERS")

            providers_list: Optional[list[str]] = None
            if providers:
                providers_list = [p.strip() for p in providers.split(",")]

            return ONNXRTBackend(
                model_name=model,
                pretrained=None,
                onnx_image_path=onnx_image,
                onnx_text_path=onnx_text,
                providers=providers_list,
                device_preference=device_preference,
                max_batch_size=max_batch_size,
                **kwargs,
            )

        elif backend_type == "rknn":
            # RKNN specific configuration
            rknn_path = kwargs.get("rknn_model_path") or os.getenv("BIOCLIP_RKNN_MODEL")
            target = kwargs.get("target") or os.getenv("BIOCLIP_RKNN_TARGET", "rk3588")

            return RKNNBackend(
                model_name=model,
                pretrained=None,
                rknn_model_path=rknn_path,
                target=target,
                device_preference=device_preference,
                max_batch_size=max_batch_size,
                **kwargs,
            )

        else:
            raise BackendCreationError(f"Unsupported backend type: {backend_type}")

    except Exception as e:
        raise BackendCreationError(
            f"Failed to create {backend_type} backend: {e}"
        ) from e


def get_available_backends() -> list[str]:
    """
    Get list of available backend types.

    Returns:
        List of available backend names
    """
    available = []

    # Check if torch backend is available
    try:
        import torch

        available.append("torch")
    except ImportError:
        pass

    # Check if ONNX Runtime backend is available
    try:
        import onnxruntime

        available.append("onnxrt")
    except ImportError:
        pass

    # Check if RKNN backend is available
    try:
        # This would depend on RKNN toolkit availability
        # For now, we'll assume it's available if the backend class exists
        available.append("rknn")
    except ImportError:
        pass

    return available


def validate_backend_config(backend_type: str, **config) -> bool:
    """
    Validate backend configuration before creation.

    Args:
        backend_type: Backend type to validate
        **config: Configuration parameters

    Returns:
        True if configuration is valid

    Raises:
        BackendCreationError: If configuration is invalid
    """
    backend_type = backend_type.lower()

    if backend_type not in get_available_backends():
        raise BackendCreationError(f"Backend type '{backend_type}' is not available")

    if backend_type == "onnxrt":
        # Validate ONNX specific configuration
        if not config.get("onnx_image_path") and not os.getenv("CLIP_ONNX_IMAGE"):
            raise BackendCreationError(
                "ONNX image model path is required for ONNX backend"
            )
        if not config.get("onnx_text_path") and not os.getenv("CLIP_ONNX_TEXT"):
            raise BackendCreationError(
                "ONNX text model path is required for ONNX backend"
            )

    elif backend_type == "rknn":
        # Validate RKNN specific configuration
        if not config.get("rknn_model_path") and not os.getenv("CLIP_RKNN_MODEL"):
            raise BackendCreationError("RKNN model path is required for RKNN backend")

    return True
