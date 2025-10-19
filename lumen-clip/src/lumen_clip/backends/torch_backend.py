"""
TorchBackend: an OpenCLIP-based backend that implements the BaseClipBackend
interface using PyTorch.

This backend:
- Loads models from local files (no automatic downloads)
- Uses config.json for model architecture
- Uses tokenizer.json or falls back to SimpleTokenizer
- Produces unit-normalized float32 embeddings for both text and images
- Supports true batched image embedding on GPU/CPU

Notes:
- Returned vectors are always L2-normalized float32.
- image_batch_to_vectors performs a single forward pass for efficiency.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable, Sequence
from typing import cast
from typing_extensions import override

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from lumen_clip.resources.loader import ModelResources

import torch
import open_clip
from .backend_exceptions import *
from .base import BaseClipBackend, BackendInfo

logger = logging.getLogger(__name__)


class TorchBackendError(BackendError):
    """Base class for TorchBackend specific errors."""

    pass


class CUDAMemoryError(TorchBackendError):
    """Raised when CUDA out of memory occurs."""

    pass


class TorchModelLoadingError(TorchBackendError, ModelLoadingError):
    """Raised when PyTorch model loading fails."""

    pass


class TorchDeviceUnavailableError(TorchBackendError, DeviceUnavailableError):
    """Raised when requested PyTorch device is not available."""

    pass


class TorchBackend(BaseClipBackend):
    """
    A PyTorch/OpenCLIP backend implementing BaseClipBackend.

    Args:
        resources: ModelResources object containing model files and configs
        device_preference: Optional hint for device selection ("cuda", "mps", "cpu").
        max_batch_size: Optional hint for batch size; not enforced by backend.

    Behavior:
        - initialize() loads the model from local files, tokenizer, and preprocess pipeline
        - Always loads FP32 weights (model.pt)
        - Uses mixed precision (AMP) on GPU (CUDA/MPS) for inference
        - text_to_vector() encodes a text prompt via the model's text encoder
        - image_to_vector() decodes and preprocesses image bytes, then encodes via the image encoder
        - image_batch_to_vectors() performs a single batched forward pass for multiple images
    """

    def __init__(
        self,
        resources: "ModelResources",
        device_preference: str | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        super().__init__(
            resources=resources,
            device_preference=device_preference,
            max_batch_size=max_batch_size,
        )

        # Runtime objects
        self._device: torch.device = self._select_device(device_preference)
        self._model: torch.nn.Module | None = None
        self._preprocess: Callable[[Image.Image], torch.Tensor] | None = None
        self._tokenizer: Callable[[list[str]], torch.Tensor] | None = None
        self._load_time_seconds: float | None = None

        # Mixed precision settings
        self._use_amp: bool = self._device.type in ("cuda")
        self._amp_dtype: torch.dtype = torch.float16
        self._current_precision: str = "fp32"  # Will be updated after init

    # ---------- Lifecycle ----------

    @override
    def initialize(self) -> None:
        if self._initialized:
            return

        t0 = time.time()
        try:
            logger.info(f"Initializing TorchBackend for {self.resources.model_name}")

            # 1. Get model architecture name from config
            config = self.resources.config
            model_name = config.get("model_name", self.resources.model_name)

            # 2. Create model architecture without pretrained weights
            logger.info(f"Creating model architecture: {model_name}")
            model_obj, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=None,  # No automatic download
            )

            # 3. Load local weights
            model_file = self.resources.get_model_file("model.pt")
            if not model_file.exists():
                raise TorchModelLoadingError(f"Model file not found: {model_file}")

            logger.info(f"Loading weights from {model_file}")
            state_dict = torch.load(model_file, map_location=self._device)

            # Handle potential state_dict wrapper (e.g., {"model": ...})
            if "model" in state_dict:
                state_dict = state_dict["model"]
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            model_obj.load_state_dict(state_dict)

            model_module = cast(torch.nn.Module, model_obj)
            model_module.eval().to(self._device)
            self._model = model_module
            self._preprocess = cast(Callable[[Image.Image], torch.Tensor], preprocess)

            # 4. Load tokenizer
            self._tokenizer = self._load_tokenizer(model_name)

            self._load_time_seconds = time.time() - t0
            self._initialized = True

            # Update precision info
            if self._use_amp:
                self._current_precision = "fp32+amp(fp16)"
                logger.info(f"Mixed precision (AMP) enabled on {self._device}")
            else:
                self._current_precision = "fp32"

            logger.info(
                f"✅ TorchBackend initialized in {self._load_time_seconds:.2f}s"
            )
            logger.info(f"   Precision: {self._current_precision}")
            logger.info(f"   Device: {self._device}")

        except ImportError as e:
            raise TorchModelLoadingError(f"Required dependencies not found: {e}") from e
        except Exception as e:
            raise TorchModelLoadingError(f"Model loading failed: {e}") from e

    def _load_tokenizer(self, model_name: str) -> Callable[[list[str]], torch.Tensor]:
        """Load tokenizer from tokenizer.json or fallback to SimpleTokenizer."""
        if self.resources.tokenizer_config:
            try:
                # Try to use HuggingFace tokenizers
                from tokenizers import Tokenizer as HFTokenizer

                tokenizer_path = self.resources.model_root_path / "tokenizer.json"
                hf_tokenizer = HFTokenizer.from_file(str(tokenizer_path))

                logger.info("Using custom tokenizer from tokenizer.json")

                # Wrap HF tokenizer to match open_clip interface
                def tokenize_fn(texts: list[str]) -> torch.Tensor:
                    encoded = hf_tokenizer.encode_batch(texts)
                    tokens = [enc.ids for enc in encoded]
                    # Pad to max length
                    max_len = max(len(t) for t in tokens)
                    padded = [t + [0] * (max_len - len(t)) for t in tokens]
                    return torch.tensor(padded, dtype=torch.long)

                return tokenize_fn

            except ImportError:
                logger.warning(
                    "tokenizers library not available, falling back to SimpleTokenizer"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load custom tokenizer: {e}, "
                    f"falling back to SimpleTokenizer"
                )

        # Fallback to SimpleTokenizer
        logger.info("Using SimpleTokenizer (fallback)")
        tokenizer_fun = open_clip.get_tokenizer(model_name)
        return cast(Callable[[list[str]], torch.Tensor], tokenizer_fun)

    @override
    def close(self) -> None:
        # Best-effort release; PyTorch will free on GC.
        m = self._model
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        if m is not None and self._device.type == "cuda":
            # Free cached memory for the device
            torch.cuda.empty_cache()

    # ---------- Encoding API ----------

    @torch.inference_mode()
    @override
    def text_to_vector(self, text: str) -> NDArray[np.float32]:
        """
        Encode a text string into a unit-normalized float32 embedding vector.
        Uses mixed precision (AMP) on GPU for better performance.
        """
        self._ensure_initialized()

        if not text or not text.strip():
            raise InvalidInputError("text cannot be empty or whitespace only")

        if len(text) > 10000:  # Reasonable length limit
            raise InvalidInputError("text too long (max 10000 characters)")

        try:
            # Create a simple CLIP-compatible prompt
            prompt = text
            assert self._tokenizer is not None
            assert self._model is not None

            tokenizer = cast(Callable[[list[str]], torch.Tensor], self._tokenizer)
            tokens = tokenizer([prompt]).to(self._device)

            # Use AMP if enabled (GPU only)
            if self._use_amp:
                with torch.autocast(
                    device_type=self._device.type, dtype=self._amp_dtype
                ):
                    feats = self._model.encode_text(tokens)  # type: ignore[attr-defined]
            else:
                feats = self._model.encode_text(tokens)  # type: ignore[attr-defined]

            feats = feats / feats.norm(dim=-1, keepdim=True)

            vec = feats.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            return vec

        except torch.cuda.OutOfMemoryError as e:
            raise CUDAMemoryError(
                f"CUDA out of memory during text encoding: {e}"
            ) from e
        except Exception as e:
            raise InferenceError(f"Text encoding failed: {e}") from e

    @torch.inference_mode()
    @override
    def image_to_vector(self, image_bytes: bytes) -> NDArray[np.float32]:
        """
        Encode image bytes (RGB) into a unit-normalized float32 embedding vector.
        Uses mixed precision (AMP) on GPU for better performance.
        """
        self._ensure_initialized()

        if not image_bytes:
            raise InvalidInputError("image_bytes cannot be empty")

        try:
            assert self._preprocess is not None
            assert self._model is not None

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            preprocess = cast(Callable[[Image.Image], torch.Tensor], self._preprocess)
            tensor = preprocess(img).unsqueeze(0).to(self._device)

            # Use AMP if enabled (GPU only)
            if self._use_amp:
                with torch.autocast(
                    device_type=self._device.type, dtype=self._amp_dtype
                ):
                    feats = self._model.encode_image(tensor)  # type: ignore[attr-defined]
            else:
                feats = self._model.encode_image(tensor)  # type: ignore[attr-defined]

            feats = feats / feats.norm(dim=-1, keepdim=True)

            vec = feats.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            return vec

        except torch.cuda.OutOfMemoryError as e:
            raise CUDAMemoryError(
                f"CUDA out of memory during image encoding: {e}"
            ) from e
        except Exception as e:
            raise InferenceError(f"Image encoding failed: {e}") from e

    @torch.inference_mode()
    @override
    def image_batch_to_vectors(self, images: Sequence[bytes]) -> NDArray[np.float32]:
        """
        Encode a list of image bytes using a single batched forward pass.
        Uses mixed precision (AMP) on GPU for better performance.
        Falls back to BaseClipBackend's sequential implementation if images is empty.
        """
        if not images:
            return super().image_batch_to_vectors(images)

        self._ensure_initialized()
        assert self._preprocess is not None
        assert self._model is not None
        preprocess = cast(Callable[[Image.Image], torch.Tensor], self._preprocess)

        try:
            # Decode and preprocess to a batch tensor [N, C, H, W]
            tensors: list[torch.Tensor] = []
            for b in images:
                if not b:
                    raise InvalidInputError("image bytes cannot be empty")

                img = Image.open(io.BytesIO(b)).convert("RGB")
                t = preprocess(img)
                tensors.append(t)

            batch = torch.stack(tensors, dim=0).to(self._device)

            # Use AMP if enabled (GPU only)
            if self._use_amp:
                with torch.autocast(
                    device_type=self._device.type, dtype=self._amp_dtype
                ):
                    feats = self._model.encode_image(batch)  # type: ignore[attr-defined]
            else:
                feats = self._model.encode_image(batch)  # type: ignore[attr-defined]

            feats = feats / feats.norm(dim=-1, keepdim=True)

            arr = feats.detach().cpu().numpy().astype(np.float32, copy=False)
            return arr

        except torch.cuda.OutOfMemoryError as e:
            raise CUDAMemoryError(
                f"CUDA out of memory during batch image encoding: {e}"
            ) from e
        except Exception as e:
            raise InferenceError(f"Batch image encoding failed: {e}") from e

    @torch.inference_mode()
    @override
    def text_batch_to_vectors(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """
        Encode a batch of text strings into unit-normalized float32 embedding vectors.
        Uses mixed precision (AMP) on GPU for better performance.

        Args:
            texts: Sequence of text strings to encode.

        Returns:
            np.ndarray with shape (N, D) and dtype float32, each row L2-normalized.
        """
        self._ensure_initialized()
        assert self._tokenizer is not None
        assert self._model is not None

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        try:
            # Validate inputs
            for text in texts:
                if not text or not text.strip():
                    raise InvalidInputError("text cannot be empty or whitespace only")
                if len(text) > 10000:  # Reasonable length limit
                    raise InvalidInputError("text too long (max 10000 characters)")

            tokenizer = cast(Callable[[list[str]], torch.Tensor], self._tokenizer)
            tokens = tokenizer(list(texts)).to(self._device)

            # Use AMP if enabled (GPU only)
            if self._use_amp:
                with torch.autocast(
                    device_type=self._device.type, dtype=self._amp_dtype
                ):
                    feats = self._model.encode_text(tokens)  # type: ignore[attr-defined]
            else:
                feats = self._model.encode_text(tokens)  # type: ignore[attr-defined]

            feats = feats / feats.norm(dim=-1, keepdim=True)

            vecs = feats.detach().cpu().numpy().astype(np.float32, copy=False)
            return vecs

        except torch.cuda.OutOfMemoryError as e:
            raise CUDAMemoryError(
                f"CUDA out of memory during batch text encoding: {e}"
            ) from e
        except Exception as e:
            raise InferenceError(f"Batch text encoding failed: {e}") from e

    @override
    def get_info(self) -> BackendInfo:
        """
        Report runtime and model metadata.
        """
        # Report precision: always loads fp32 weights, uses AMP on GPU
        if self._use_amp:
            precisions = ["fp32", "fp16(amp)"]
        else:
            precisions = ["fp32"]

        version = getattr(open_clip, "__version__", None)

        # Get embedding dimension from resources
        embed_dim = self.resources.get_embedding_dim()

        # Get image size from resources
        image_size = self.resources.get_image_size()
        image_size_str = f"{image_size[0]}x{image_size[1]}" if image_size else None

        return BackendInfo(
            runtime="torch",
            device=str(self._device),
            model_id=self.resources.model_name,
            model_name=self.resources.config.get(
                "model_name", self.resources.model_name
            ),
            pretrained=None,  # Local weights, no pretrained tag
            version=str(version) if version is not None else None,
            image_embedding_dim=embed_dim,
            text_embedding_dim=embed_dim,
            precisions=precisions,
            max_batch_size=self._max_batch_size,
            supports_image_batch=True,
            extra={
                "library": "open-clip-torch",
                "image_size": image_size_str,
                "config_path": str(self.resources.model_root_path / "config.json"),
            },
        )

    # ---------- Helpers ----------

    @staticmethod
    def _select_device(preference: str | None) -> torch.device:
        """
        Choose a torch.device based on an optional preference and availability.
        """
        pref = (preference or "").lower().strip()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")

        # Auto-detect best available
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _ensure_initialized(self) -> None:
        if (
            not self._initialized
            or self._model is None
            or self._preprocess is None
            or self._tokenizer is None
        ):
            raise RuntimeError(
                "TorchBackend is not initialized. Call initialize() first."
            )
