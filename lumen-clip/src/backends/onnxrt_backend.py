"""
ONNXRTBackend: ONNX Runtime backend for CLIP-like models.

This backend:
- Loads ONNX models from local files with precision-aware naming:
  - FP32: vision.onnx, text.onnx (default fallback)
  - FP16: vision.fp16.onnx, text.fp16.onnx (preferred on GPU)
- Auto-detects model input/output precision and adapts preprocessing
- Uses config.json for model configuration
- Uses tokenizer.json or falls back to SimpleTokenizer
- Produces unit-normalized float32 embeddings for both text and images
- Supports configurable execution providers (CPU, CUDA, CoreML, etc.)

Notes:
- Returned vectors are always L2-normalized float32
- Image preprocessing adapts to model's expected input dtype
- On GPU, FP16 models are preferred if available; otherwise falls back to FP32
- On CPU, FP32 models are used by default
"""

from __future__ import annotations

from collections.abc import Sequence
import io
import logging
from typing import Callable, cast
from typing_extensions import override

import numpy as np
from numpy.typing import NDArray
from PIL import Image, Image as PILImage
from pathlib import Path

from resources import ModelResources

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is required for ONNXRTBackend. "
        + "Install with: pip install onnxruntime"
    )

from .base import (
    BaseClipBackend,
    BackendInfo,
    BackendError,
    ModelLoadingError,
    InvalidInputError,
    InferenceError,
)

logger = logging.getLogger(__name__)


class ONNXRTBackendError(BackendError):
    """Base class for ONNXRTBackend specific errors."""

    pass


class ONNXRTModelLoadingError(ONNXRTBackendError, ModelLoadingError):
    """Raised when ONNX model loading fails."""

    pass


class ONNXRTBackend(BaseClipBackend):
    # Class-level attribute annotations to satisfy static type checkers
    _providers: list[str]
    _prefer_fp16: bool
    _initialized: bool
    """
    ONNX Runtime backend implementing BaseClipBackend.

    Args:
        resources: ModelResources object containing model files and configs
        providers: ONNX Runtime execution providers (e.g., ["CPUExecutionProvider"])
        device_preference: Optional hint for device selection
        max_batch_size: Optional hint for batch size
        prefer_fp16: If True and GPU available, prefer FP16 models over FP32

    Behavior:
        - initialize() loads precision-aware model files:
          - GPU: tries vision.fp16.onnx first, falls back to vision.onnx
          - CPU: uses vision.onnx (FP32) by default
        - Auto-detects model input dtype and adapts preprocessing accordingly
        - text_to_vector() tokenizes and encodes text via the text encoder
        - image_to_vector() preprocesses and encodes images via the vision encoder
        - All outputs are converted to unit-normalized float32 vectors
    """

    def __init__(
        self,
        resources: "ModelResources",
        providers: list[str] | None = None,
        device_preference: str | None = None,
        max_batch_size: int | None = None,
        prefer_fp16: bool = True,
    ) -> None:
        super().__init__(
            resources=resources,
            device_preference=device_preference,
            max_batch_size=max_batch_size,
        )

        # Execution providers
        self._providers = providers or self._default_providers(device_preference)
        self._prefer_fp16 = prefer_fp16 and self._is_gpu_available(self._providers)

        # Runtime objects
        self._sess_vision: ort.InferenceSession | None = None
        self._sess_text: ort.InferenceSession | None = None
        self._tokenizer: Callable[[Sequence[str]], np.ndarray] | None = None
        self._image_preprocessor: Callable[[Image.Image], np.ndarray] | None = None
        self._load_time_seconds: float | None = None

        # Precision tracking
        self._vision_precision: str = "unknown"
        self._text_precision: str = "unknown"
        self._vision_input_dtype: np.dtype = np.dtype(np.float32)
        self._text_input_dtype: np.dtype = np.dtype(np.int64)

    @override
    def initialize(self) -> None:
        """Load ONNX models and prepare preprocessing."""
        if self._initialized:
            return

        import time

        t0 = time.time()

        try:
            logger.info(f"Initializing ONNXRTBackend for {self.resources.model_name}")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # 1. Load vision encoder with precision selection
            vision_path, vision_precision = self._select_model_file("vision")
            logger.info(
                f"Loading vision encoder from {vision_path} ({vision_precision})"
            )

            self._sess_vision = ort.InferenceSession(
                str(vision_path),
                sess_options,
                providers=self._providers,
            )
            self._vision_precision = vision_precision

            # Detect vision input dtype
            vision_input = self._sess_vision.get_inputs()[0]
            self._vision_input_dtype = self._onnx_type_to_numpy(vision_input.type)

            # 2. Load text encoder with precision selection
            text_path, text_precision = self._select_model_file("text")
            logger.info(f"Loading text encoder from {text_path} ({text_precision})")

            self._sess_text = ort.InferenceSession(
                str(text_path),
                sess_options,
                providers=self._providers,
            )
            self._text_precision = text_precision

            # Detect text input dtype
            text_input = self._sess_text.get_inputs()[0]
            self._text_input_dtype = self._onnx_type_to_numpy(text_input.type)

            # 3. Setup tokenizer
            # Cast to the declared callable signature to satisfy static type checkers.
            self._tokenizer = cast(
                Callable[[Sequence[str]], np.ndarray], self._load_tokenizer()
            )

            # 4. Setup image preprocessor
            self._image_preprocessor = self._create_image_preprocessor()

            self._load_time_seconds = time.time() - t0
            self._initialized = True

            logger.info(
                f"✅ ONNXRTBackend initialized in {self._load_time_seconds:.2f}s"
            )
            logger.info(f"   Providers: {self._providers}")
            logger.info(
                f"   Vision: {self._vision_precision} (input: {self._vision_input_dtype})"
            )
            logger.info(
                f"   Text: {self._text_precision} (input: {self._text_input_dtype})"
            )

        except Exception as e:
            raise ONNXRTModelLoadingError(f"ONNX model loading failed: {e}") from e

    def _select_model_file(self, model_type: str) -> tuple[Path, str]:
        """
        Select model file based on precision preference and availability.

        Args:
            model_type: "vision" or "text"

        Returns:
            Tuple of (file_path, precision_string)

        Raises:
            ONNXRTModelLoadingError: If no suitable model file is found
        """
        runtime_dir = self.resources.model_root_path / "onnx"

        # Try FP16 first if GPU and preferred
        if self._prefer_fp16:
            fp16_path = runtime_dir / f"{model_type}.fp16.onnx"
            if fp16_path.exists():
                return fp16_path, "fp16"
            logger.info(f"FP16 model not found at {fp16_path}, falling back to FP32")

        # Fall back to FP32 (default naming)
        fp32_path = runtime_dir / f"{model_type}.onnx"
        if fp32_path.exists():
            return fp32_path, "fp32"

        # If still not found, check if there's an fp16 variant as last resort
        fp16_path = runtime_dir / f"{model_type}.fp16.onnx"
        if fp16_path.exists():
            logger.warning("Only FP16 model found, using it despite CPU preference")
            return fp16_path, "fp16"

        raise ONNXRTModelLoadingError(
            f"No {model_type} model found. Expected {fp32_path} or {fp16_path}"
        )

    @staticmethod
    def _onnx_type_to_numpy(onnx_type: str) -> np.dtype:
        """Convert ONNX type string to numpy dtype."""
        type_lower = onnx_type.lower()
        if "float16" in type_lower or "half" in type_lower:
            return np.dtype(np.float16)
        if "float" in type_lower:
            return np.dtype(np.float32)
        if "int64" in type_lower or "long" in type_lower:
            return np.dtype(np.int64)
        if "int32" in type_lower or "int" in type_lower:
            return np.dtype(np.int32)
        # Default to float32 for unknown types
        logger.warning(f"Unknown ONNX type '{onnx_type}', defaulting to float32")
        return np.dtype(np.float32)

    @staticmethod
    def _is_gpu_available(providers: list[str]) -> bool:
        """Check if any GPU execution provider is in the list."""
        gpu_providers = {
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "TensorrtExecutionProvider",
        }
        return any(p in gpu_providers for p in providers)

    def _load_tokenizer(self) -> Callable[[Sequence[str]], np.ndarray]:
        """Load tokenizer from tokenizer.json or fallback to SimpleTokenizer."""
        if self.resources.tokenizer_config:
            try:
                from tokenizers import Tokenizer as HFTokenizer

                tokenizer_path = self.resources.model_root_path / "tokenizer.json"
                hf_tokenizer = HFTokenizer.from_file(str(tokenizer_path))

                logger.info("Using custom tokenizer from tokenizer.json")

                def tokenize_fn(texts: Sequence[str]) -> np.ndarray:
                    texts_list = list(texts)
                    encoded = hf_tokenizer.encode_batch(texts_list)
                    tokens = [enc.ids for enc in encoded]
                    CLIP_MAX_LEN = 77
                    padded = [
                        t[:CLIP_MAX_LEN] + [0] * max(0, CLIP_MAX_LEN - len(t))
                        for t in tokens
                    ]
                    return np.array(padded, dtype=np.int64)

                return tokenize_fn

            except ImportError:
                logger.warning(
                    "tokenizers library not available, falling back to SimpleTokenizer"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load custom tokenizer: {e}, falling back to SimpleTokenizer"
                )

        # Fallback to SimpleTokenizer
        try:
            from open_clip.tokenizer import SimpleTokenizer

            logger.info("Using SimpleTokenizer (fallback)")

            simple_tokenizer = SimpleTokenizer()

            def tokenize_fn(texts: Sequence[str]) -> np.ndarray:
                tokens = simple_tokenizer(list(texts))
                return tokens.numpy().astype(np.int64)

            return tokenize_fn
        except ImportError:
            raise ONNXRTModelLoadingError(
                "Neither tokenizers nor open_clip is available for tokenization"
            )

    def _create_image_preprocessor(self) -> Callable[[Image.Image], np.ndarray]:
        """Create image preprocessing function based on model config and input dtype."""
        # Get image size from config
        image_size = self.resources.get_image_size() or (224, 224)
        target_dtype = self._vision_input_dtype

        # Standard CLIP preprocessing
        def preprocess(image: Image.Image) -> np.ndarray:
            # Resize and center crop
            image = image.convert("RGB")
            image = image.resize(image_size, PILImage.Resampling.BICUBIC)

            # Convert to numpy array with target dtype
            img_array = np.array(image).astype(np.float32) / 255.0

            # Normalize with ImageNet stats
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

            img_array = (img_array - mean) / std

            # CHW format and add batch dimension
            img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

            # Convert to target dtype (fp16 or fp32)
            img_array = img_array.astype(target_dtype)

            return img_array

        return preprocess

    @override
    def text_to_vector(self, text: str) -> NDArray[np.float32]:
        """Encode text into a unit-normalized float32 vector."""
        self._ensure_initialized()

        if not text or not text.strip():
            raise InvalidInputError("text cannot be empty or whitespace only")

        try:
            assert self._tokenizer is not None
            assert self._sess_text is not None

            # Tokenize (always int64)
            tokens = self._tokenizer([text]).astype(self._text_input_dtype)

            # Run inference
            input_name = self._sess_text.get_inputs()[0].name
            output_name = self._sess_text.get_outputs()[0].name

            outputs = self._sess_text.run([output_name], {input_name: tokens})
            embedding = np.asarray(outputs[0]).squeeze(0).astype(np.float32)

            # Normalize
            embedding = (embedding / np.linalg.norm(embedding)).astype(np.float32)

            return embedding

        except Exception as e:
            raise InferenceError(f"Text encoding failed: {e}") from e

    @override
    def image_to_vector(self, image_bytes: bytes) -> NDArray[np.float32]:
        """Encode image bytes into a unit-normalized float32 vector."""
        self._ensure_initialized()

        if not image_bytes:
            raise InvalidInputError("image_bytes cannot be empty")

        try:
            assert self._image_preprocessor is not None
            assert self._sess_vision is not None

            # Decode and preprocess image
            img = Image.open(io.BytesIO(image_bytes))
            img_tensor = self._image_preprocessor(img)

            # Run inference
            input_name = self._sess_vision.get_inputs()[0].name
            output_name = self._sess_vision.get_outputs()[0].name

            outputs = self._sess_vision.run([output_name], {input_name: img_tensor})
            embedding = np.asarray(outputs[0]).squeeze(0).astype(np.float32)

            # Normalize
            embedding = (embedding / np.linalg.norm(embedding)).astype(np.float32)

            return embedding

        except Exception as e:
            raise InferenceError(f"Image encoding failed: {e}") from e

    @override
    def image_batch_to_vectors(self, images: Sequence[bytes]) -> NDArray[np.float32]:
        """Encode a batch of images using batched ONNX inference."""
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        self._ensure_initialized()
        assert self._image_preprocessor is not None
        assert self._sess_vision is not None

        try:
            # Preprocess all images
            img_tensors = []
            for img_bytes in images:
                if not img_bytes:
                    raise InvalidInputError("image bytes cannot be empty")
                img = Image.open(io.BytesIO(img_bytes))
                img_tensor = self._image_preprocessor(img)
                img_tensors.append(img_tensor)

            # Stack into batch
            batch = np.concatenate(img_tensors, axis=0)

            # Run batched inference
            input_name = self._sess_vision.get_inputs()[0].name
            output_name = self._sess_vision.get_outputs()[0].name

            outputs = self._sess_vision.run([output_name], {input_name: batch})
            embeddings = np.asarray(outputs[0]).astype(np.float32)

            # Normalize each vector
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

            return embeddings

        except Exception as e:
            raise InferenceError(f"Batch image encoding failed: {e}") from e

    @override
    def text_batch_to_vectors(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode a batch of texts using batched ONNX inference."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        self._ensure_initialized()
        assert self._tokenizer is not None
        assert self._sess_text is not None

        try:
            # Validate inputs
            for text in texts:
                if not text or not text.strip():
                    raise InvalidInputError("text cannot be empty or whitespace only")

            # Tokenize all texts (always int64)
            tokens = self._tokenizer(list(texts)).astype(self._text_input_dtype)

            # Run batched inference
            input_name = self._sess_text.get_inputs()[0].name
            output_name = self._sess_text.get_outputs()[0].name

            outputs = self._sess_text.run([output_name], {input_name: tokens})
            embeddings = np.asarray(outputs[0]).astype(np.float32)

            # Normalize each vector
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

            return embeddings

        except Exception as e:
            raise InferenceError(f"Batch text encoding failed: {e}") from e

    @override
    def get_info(self) -> BackendInfo:
        """Report ONNX Runtime metadata and configuration."""
        version = getattr(ort, "__version__", None)
        device = self._infer_device_from_providers(self._providers)

        # Get embedding dimension from resources
        embed_dim = self.resources.get_embedding_dim()

        # Get image size from resources
        image_size = self.resources.get_image_size()
        image_size_str = f"{image_size[0]}x{image_size[1]}" if image_size else None

        # Determine precision list
        precisions = list(set([self._vision_precision, self._text_precision]))
        if "unknown" in precisions:
            precisions.remove("unknown")
        if not precisions:
            precisions = ["fp32"]

        return BackendInfo(
            runtime="onnx",
            device=device,
            model_id=self.resources.model_name,
            model_name=self.resources.config.get(
                "model_name", self.resources.model_name
            ),
            pretrained=None,  # Local weights
            version=str(version) if version else None,
            image_embedding_dim=embed_dim,
            text_embedding_dim=embed_dim,
            precisions=precisions,
            max_batch_size=self._max_batch_size,
            supports_image_batch=True,
            extra={
                "providers": ",".join(self._providers),
                "image_size": image_size_str,
                "vision_precision": self._vision_precision,
                "text_precision": self._text_precision,
                "vision_input_dtype": str(self._vision_input_dtype),
                "text_input_dtype": str(self._text_input_dtype),
            },
        )

    @staticmethod
    def _default_providers(device_pref: str | None) -> list[str]:
        """Select default ONNX Runtime providers based on device preference."""
        pref = (device_pref or "").lower().strip()

        if pref == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if pref == "coreml":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        if pref == "directml":
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        if pref == "openvino":
            return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

        # Default: CPU only
        return ["CPUExecutionProvider"]

    @staticmethod
    def _infer_device_from_providers(providers: list[str]) -> str:
        """Infer device string from provider list."""
        provs = [p.lower() for p in providers]

        if any("cuda" in p for p in provs):
            return "cuda"
        if any("coreml" in p for p in provs):
            return "coreml"
        if any("dml" in p for p in provs):
            return "directml"
        if any("openvino" in p for p in provs):
            return "openvino"

        return "cpu"

    def _ensure_initialized(self) -> None:
        """Ensure backend is initialized before inference."""
        if (
            not self._initialized
            or self._sess_vision is None
            or self._sess_text is None
        ):
            raise RuntimeError(
                "ONNXRTBackend is not initialized. Call initialize() first."
            )
