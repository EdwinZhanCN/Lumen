"""
ONNXRTBackend: ONNX Runtime backend for CLIP-like models.

This backend:
- Loads ONNX models from local files with precision-aware naming:
  - FP32: vision.onnx, text.onnx (default fallback)
  - FP16: vision.fp16.onnx, text.fp16.onnx
  - INT8: vision.int8.onnx, text.int8.onnx
  - Q4FP16: vision.q4fp16.onnx, text.q4fp16.onnx
- Auto-detects model input/output precision and adapts preprocessing
- Uses config.json for model configuration
- Uses tokenizer.json or falls back to SimpleTokenizer
- Produces unit-normalized float32 embeddings for both text and images
- Supports configurable execution providers (CPU, CUDA, CoreML, etc.)

Notes:
- Returned vectors are always L2-normalized float32
- Image preprocessing adapts to model's expected input dtype
- Model file selection uses the precision parameter directly (e.g., "fp16", "int8")
- Falls back to default precision (fp32) if specified precision files not found
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL import Image as PILImage
from tokenizers import Tokenizer as HFTokenizer
from typing_extensions import override

from lumen_clip.resources import ModelResources

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .backend_exceptions import *
from .base import BackendInfo, BaseClipBackend

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
    _precision: str | None
    _initialized: bool
    """
    ONNX Runtime backend implementing BaseClipBackend.

    Args:
        resources: ModelResources object containing model files and configs
        providers: ONNX Runtime execution providers (e.g., ["CPUExecutionProvider"])
        device_preference: Optional hint for device selection
        max_batch_size: Optional hint for batch size
        precision: Model precision for file selection (e.g., "fp32", "fp16", "int8", "q4fp16").
                    If None, uses default precision (fp32).

    Behavior:
        - initialize() loads precision-aware model files based on the precision parameter:
          - Tries {component}.{precision}.onnx if precision is specified
          - Falls back to {component}.onnx (fp32 default) if precision files not found
        - Auto-detects model input dtype and adapts preprocessing accordingly
        - text_to_vector() tokenizes and encodes text via the text encoder
        - image_to_vector() preprocesses and encodes images via the vision encoder
        - All outputs are converted to unit-normalized float32 vectors
    """

    def __init__(
        self,
        resources: ModelResources,
        providers: list[str] | None = None,
        device_preference: str | None = None,
        max_batch_size: int | None = None,
        precision: str | None = None,
    ) -> None:
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ONNXRTBackend. "
                + "Install with: pip install onnxruntime"
            )

        super().__init__(
            resources=resources,
            device_preference=device_preference,
            max_batch_size=max_batch_size,
        )

        # Execution providers
        self._providers = providers or self._default_providers(device_preference)
        self._precision = precision

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

        # Batching capability flags
        self._vision_batch_dynamic: bool = False
        self._text_batch_dynamic: bool = False

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

            # Smart loading: Patch to static shape if using CoreML
            vision_model_payload = self._load_and_patch_model(vision_path)

            self._sess_vision = ort.InferenceSession(
                vision_model_payload,
                sess_options,
                providers=self._providers,
            )
            self._vision_precision = vision_precision

            # Detect vision input dtype
            vision_input = self._sess_vision.get_inputs()[0]
            self._vision_input_dtype = self._onnx_type_to_numpy(vision_input.type)

            # Detect if vision model supports dynamic batching
            v_shape = vision_input.shape
            self._vision_batch_dynamic = len(v_shape) > 0 and not isinstance(
                v_shape[0], int
            )
            if not self._vision_batch_dynamic:
                logger.info(f"   Vision model has static batch size: {v_shape[0]}")

            # 2. Load text encoder with precision selection
            text_path, text_precision = self._select_model_file("text")
            logger.info(f"Loading text encoder from {text_path} ({text_precision})")

            # Smart loading: Patch to static shape if using CoreML
            text_model_payload = self._load_and_patch_model(text_path)

            self._sess_text = ort.InferenceSession(
                text_model_payload,
                sess_options,
                providers=self._providers,
            )
            self._text_precision = text_precision

            # Detect text input dtype
            text_input = self._sess_text.get_inputs()[0]
            self._text_input_dtype = self._onnx_type_to_numpy(text_input.type)

            # Detect if text model supports dynamic batching
            t_shape = text_input.shape
            self._text_batch_dynamic = len(t_shape) > 0 and not isinstance(
                t_shape[0], int
            )
            if not self._text_batch_dynamic:
                logger.info(f"   Text model has static batch size: {t_shape[0]}")

            # Detect context length from model input shape
            self._context_length = 77  # Default CLIP context length
            if len(text_input.shape) == 2 and isinstance(text_input.shape[1], int):
                self._context_length = text_input.shape[1]
                logger.info(
                    f"Detected context length from ONNX model: {self._context_length}"
                )

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
        Select model file based on precision configuration and availability.

        Args:
            model_type: "vision" or "text"

        Returns:
            Tuple of (file_path, precision_string)

        Raises:
            ONNXRTModelLoadingError: If no suitable model file is found
        """
        runtime_dir = self.resources.model_root_path / "onnx"

        # Try specified precision first
        if self._precision:
            precision_path = runtime_dir / f"{model_type}.{self._precision}.onnx"
            if precision_path.exists():
                return precision_path, self._precision
            logger.warning(
                f"Precision {self._precision} model file not found at {precision_path}, "
                f"falling back to default precision"
            )

        # Fall back to default (fp32, no extension)
        default_path = runtime_dir / f"{model_type}.onnx"
        if default_path.exists():
            return default_path, "fp32"

        # If still not found and we had a precision, check for fp16 as last resort
        if self._precision and self._precision != "fp16":
            fp16_path = runtime_dir / f"{model_type}.fp16.onnx"
            if fp16_path.exists():
                logger.warning(
                    f"Only FP16 model found for {model_type}, using it instead of {self._precision}"
                )
                return fp16_path, "fp16"

        # Build error message with both expected paths
        precision_filename = f"{model_type}.{self._precision or 'fp32'}.onnx"
        precision_path = runtime_dir / precision_filename
        raise ONNXRTModelLoadingError(
            f"No {model_type} model found. Expected {default_path} or {precision_path}"
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

    def _load_tokenizer(self) -> Callable[[Sequence[str]], np.ndarray]:
        """Load tokenizer from tokenizer.json, transformers, or fallback to SimpleTokenizer."""
        context_length = getattr(self, "_context_length", 77)

        # 1. Try loading from tokenizer.json using tokenizers library (fastest)
        if self.resources.tokenizer_config:
            try:
                tokenizer_path = self.resources.model_root_path / "tokenizer.json"
                hf_tokenizer = HFTokenizer.from_file(str(tokenizer_path))

                logger.info(
                    f"Using custom tokenizer from tokenizer.json (context_length={context_length})"
                )

                def tokenize_fn(texts: Sequence[str]) -> np.ndarray:
                    texts_list = list(texts)
                    encoded = hf_tokenizer.encode_batch(texts_list)
                    tokens = [enc.ids for enc in encoded]
                    padded = [
                        t[:context_length] + [0] * max(0, context_length - len(t))
                        for t in tokens
                    ]
                    # Truncate if longer than context_length
                    padded = [t[:context_length] for t in padded]
                    return np.array(padded, dtype=np.int64)

                return tokenize_fn

            except ImportError:
                logger.warning(
                    "tokenizers library not available, falling back to transformers"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load custom tokenizer: {e}, falling back to transformers"
                )

        # 2. Try loading using transformers AutoTokenizer
        try:
            from transformers import AutoTokenizer

            logger.info(
                f"Attempting to load tokenizer via transformers.AutoTokenizer from {self.resources.model_root_path}"
            )
            # Use local_files_only to ensure we use the downloaded resources
            transformers_tokenizer = AutoTokenizer.from_pretrained(
                str(self.resources.model_root_path), local_files_only=True
            )

            def tokenize_fn_transformers(texts: Sequence[str]) -> np.ndarray:
                res = transformers_tokenizer(
                    list(texts),
                    padding="max_length",
                    truncation=True,
                    max_length=context_length,
                    return_tensors="np",
                )
                return res["input_ids"].astype(np.int64)

            return tokenize_fn_transformers

        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer via transformers: {e}. Falling back to SimpleTokenizer."
            )

        # 3. Fallback to SimpleTokenizer (requires open_clip_torch)
        try:
            from open_clip.tokenizer import SimpleTokenizer

            logger.info("Using SimpleTokenizer (fallback)")

            simple_tokenizer = SimpleTokenizer()

            def tokenize_fn_simple(texts: Sequence[str]) -> np.ndarray:
                # SimpleTokenizer usually handles 77 context length internally
                tokens = simple_tokenizer(list(texts), context_length=context_length)
                return tokens.numpy().astype(np.int64)

            return tokenize_fn_simple
        except ImportError as exc:
            raise ONNXRTModelLoadingError(
                "Tokenization failed: 'tokenizer.json' not found/valid, 'transformers' failed, and 'open_clip' not installed. "
                "Install with `pip install lumen-clip[cpu]` (or other extras) to include open-clip-torch."
            ) from exc

    def _create_image_preprocessor(self) -> Callable[[Image.Image], np.ndarray]:
        """Create image preprocessing function based on model config and input dtype."""
        # Get image size from config first, then from model input shape
        image_size = self.resources.get_image_size()
        if image_size is None:
            # Extract image size from vision model input shape
            # Input shape is typically [batch, channels, height, width] or [batch, height, width, channels]
            vision_input = self._sess_vision.get_inputs()[0]  # type: ignore
            input_shape = vision_input.shape
            # Find height and width dimensions (skip batch and channels)
            spatial_dims = [
                dim for dim in input_shape if isinstance(dim, int) and dim > 3
            ]
            if len(spatial_dims) >= 2:
                height, width = spatial_dims[-2], spatial_dims[-1]
                image_size = (height, width)
            else:
                # Fallback if we can't determine shape
                image_size = (224, 224)
                logger.warning(
                    f"Could not determine image size from input shape {input_shape}, using fallback {image_size}"
                )

        # Get normalization stats from resources
        norm_stats = self.resources.get_normalization_stats()
        target_dtype = self._vision_input_dtype

        logger.info(
            f"Using normalization stats - mean: {norm_stats['mean']}, std: {norm_stats['std']}"
        )

        # Configurable preprocessing
        def preprocess(image: Image.Image) -> np.ndarray:
            # Resize and center crop
            image = image.convert("RGB")
            image = image.resize(image_size, PILImage.Resampling.BICUBIC)

            # Convert to numpy array with target dtype
            img_array = np.array(image).astype(np.float32) / 255.0

            # Normalize with configured stats
            mean = np.array(norm_stats["mean"], dtype=np.float32)
            std = np.array(norm_stats["std"], dtype=np.float32)

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

            # Handle static batch size model (e.g. exported for CoreML/NPU)
            # If model is static (batch=1) but we have N images, we must loop.
            if not self._vision_batch_dynamic and batch.shape[0] > 1:
                # Fallback to sequential execution for static models
                embeddings_list = []
                for i in range(batch.shape[0]):
                    # Slice to keep dims: (1, C, H, W)
                    single_input = batch[i : i + 1]
                    out = self._sess_vision.run(
                        [output_name], {input_name: single_input}
                    )
                    embeddings_list.append(out[0])

                # Stack results: (N, D)
                raw_embeddings = np.concatenate(embeddings_list, axis=0)
                embeddings = np.asarray(raw_embeddings).astype(np.float32)
            else:
                # Standard batched execution
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

            # Handle static batch size model
            if not self._text_batch_dynamic and tokens.shape[0] > 1:
                embeddings_list = []
                for i in range(tokens.shape[0]):
                    single_input = tokens[i : i + 1]
                    out = self._sess_text.run([output_name], {input_name: single_input})
                    embeddings_list.append(out[0])

                raw_embeddings = np.concatenate(embeddings_list, axis=0)
                embeddings = np.asarray(raw_embeddings).astype(np.float32)
            else:
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
        precisions = list({self._vision_precision, self._text_precision})
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
        """Select default ONNX Runtime providers based on availability and priority."""
        available = set(ort.get_available_providers())

        # Priority order: GPU first, then CPU
        priority = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "TensorrtExecutionProvider",
            "CPUExecutionProvider",
        ]

        selected = [prov for prov in priority if prov in available]

        pref_map = {
            "cuda": "CUDAExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
            "directml": "DmlExecutionProvider",
            "openvino": "OpenVINOExecutionProvider",
        }
        desired = pref_map.get((device_pref or "").lower())
        if desired and desired in selected:
            selected.insert(0, selected.pop(selected.index(desired)))

        return selected or ["CPUExecutionProvider"]

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

    def _load_and_patch_model(self, model_path: Path) -> str | bytes:
        """
        Load model, optionally patching dynamic shapes to static for CoreML.
        Returns either a file path (str) or model bytes.
        """
        # Only patch if using CoreML
        if "CoreMLExecutionProvider" not in self._providers:
            return str(model_path)

        try:
            import onnx
        except ImportError:
            logger.warning(
                "CoreMLExecutionProvider detected but 'onnx' library not found. "
                "Cannot patch dynamic shapes to static. Install 'onnx' for better CoreML support."
            )
            return str(model_path)

        try:
            logger.info(
                f"Patching model {model_path.name} to static batch=1 for CoreML..."
            )
            model = onnx.load(model_path)

            # Iterate over all inputs and fix dynamic batch dim (index 0) to 1
            patched = False
            for input_tensor in model.graph.input:
                dims = input_tensor.type.tensor_type.shape.dim
                if len(dims) > 0:
                    # Check if dim 0 is dynamic (has param string or value <= 0)
                    dim0 = dims[0]
                    if dim0.dim_param or dim0.dim_value <= 0:
                        dim0.dim_value = 1
                        if dim0.HasField("dim_param"):
                            dim0.ClearField("dim_param")
                        patched = True

            if patched:
                return model.SerializeToString()
            else:
                return str(model_path)

        except Exception as e:
            logger.warning(
                f"Failed to patch model for CoreML: {e}. Fallback to file path."
            )
            return str(model_path)

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
