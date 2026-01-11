"""
TorchBackend: an OpenCLIP-based backend that implements the BaseClipBackend
interface using PyTorch.

This backend loads models from local files without automatic downloads, uses
config.json for model architecture, and produces unit-normalized float32 embeddings.

Notes:
    - Returned vectors are always L2-normalized float32.
    - image_batch_to_vectors performs a single forward pass for efficiency.

Args:
    resources (ModelResources): ModelResources object containing model files and configs.
    device_preference (str, optional): Hint for device selection ("cuda", "mps", "cpu").
    max_batch_size (int, optional): Hint for batch size; not enforced by backend.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import open_clip
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    ChineseCLIPProcessor,
    CLIPProcessor,
)
from typing_extensions import override

from lumen_clip.resources.loader import ModelResources

from .backend_exceptions import *
from .base import BackendInfo, BaseClipBackend

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

    This backend loads models from local files without automatic downloads, uses
    config.json for model architecture, and produces unit-normalized float32 embeddings.

    Notes:
        - Returned vectors are always L2-normalized float32.
        - image_batch_to_vectors performs a single forward pass for efficiency.

    Args:
        resources (ModelResources): ModelResources object containing model files and configs.
        device_preference (str, optional): Hint for device selection ("cuda", "mps", "cpu").
        max_batch_size (int, optional): Hint for batch size; not enforced by backend.
    """

    def __init__(
        self,
        resources: ModelResources,
        device_preference: str | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for TorchBackend. "
                "Install with `pip install lumen-clip[cpu]` (or [cu126], [mps])."
            )

        super().__init__(
            resources=resources,
            device_preference=device_preference,
            max_batch_size=max_batch_size,
        )

        # Runtime objects
        self._device: torch.device = self._select_device(device_preference)
        # OpenCLIP model and preprocess
        self._openclip_model: torch.nn.Module | None = None
        self._openclip_preprocess: Callable[[Image.Image], torch.Tensor] | None = None
        self._openclip_tokenizer: Callable[[list[str]], torch.Tensor] | None = None
        # HuggingFace model and processor (Auto classes)
        self._hf_model: AutoModel | None = None
        self._hf_processor: AutoProcessor | None = None
        self._hf_tokenizer: AutoTokenizer | None = None

        self._load_time_seconds: float | None = None

        # Mixed precision settings
        self._use_amp: bool = self._device.type in "cuda"
        self._amp_dtype: torch.dtype = torch.float16
        self._current_precision: str = "fp32"  # Will be updated after init

    # ---------- Lifecycle ----------

    @override
    def initialize(self) -> None:
        """
        Initialize the TorchBackend by loading the model, tokenizer, and preprocess pipeline.

        This method loads the model from local files based on the source format (OpenCLIP or HuggingFace),
        sets up the device, and configures mixed precision if applicable.

        Raises:
            TorchModelLoadingError: If model loading fails or dependencies are missing.
        """
        if self._initialized:
            return

        t0 = time.time()
        try:
            logger.info(f"Initializing TorchBackend for {self.resources.model_name}")
            logger.info(f"Model source format: {self.resources.source_format}")

            if self.resources.source_format == "openclip":
                self._initialize_openclip()
            elif self.resources.source_format == "huggingface":
                self._initialize_huggingface()
            else:
                # This should have been caught by the loader, but as a safeguard:
                raise TorchModelLoadingError(
                    f"Unsupported source_format: {self.resources.source_format}"
                )

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

    def _initialize_openclip(self) -> None:
        """
        Initialize an OpenCLIP model from local files or remote repo.

        Attempts to load the model architecture and weights, with fallbacks for local path and remote repo.

        Raises:
            TorchModelLoadingError: If model loading fails.
        """
        # 1. Get model architecture name from config
        model_name = self.resources.model_name
        model_remote_repo = self.resources.source_repo
        model_runtime_files_path = self.resources.runtime_files_path

        # 2. Create model architecture without pretrained weights
        logger.info(f"Creating OpenCLIP model architecture: {model_name}")

        try:
            model_obj, _, preprocess = open_clip.create_model_and_transforms(
                "local-dir:" + str(model_runtime_files_path),
                pretrained=None,  # No automatic download
            )
            logger.info(
                f"Successfully created OpenCLIP model architecture using local path'{model_runtime_files_path}'"
            )
        except Exception as e:
            logger.warning(
                f"Failed to create OpenCLIP model architecture with name '{model_name}': {e}. "
                "Attempting to create from remote repo fallback"
            )
            try:
                model_obj, _, preprocess = open_clip.create_model_and_transforms(
                    "hf-hub:" + model_remote_repo,
                    pretrained=None,  # No automatic download
                )
                logger.info(
                    f"Successfully created OpenCLIP model architecture using remote repo'{model_remote_repo}'"
                )
            except Exception as e2:
                raise TorchModelLoadingError(
                    f"Failed to create OpenCLIP model architecture with both "
                    f"'{model_name}' and fallback '{model_remote_repo}': {e2}"
                ) from e

        # 3. Load local openclip weights
        model_file = self.resources.get_model_file("open_clip_pytorch_model.bin")
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
        self._openclip_model = model_module
        self._openclip_preprocess = cast(
            Callable[[Image.Image], torch.Tensor], preprocess
        )

        # 4. Load tokenizer
        self._openclip_tokenizer = self._load_tokenizer_openclip(model_name)

    def _initialize_huggingface(self) -> None:
        """
        Initialize a HuggingFace AutoModel from local files or remote repo.

        Uses AutoModel, AutoProcessor, and AutoTokenizer for universal compatibility.

        Raises:
            TorchModelLoadingError: If model loading fails.
        """
        model_runtime_files_path = self.resources.runtime_files_path
        model_remote_repo = self.resources.source_repo
        logger.info(
            f"Loading HuggingFace AutoModel from local disk: {model_runtime_files_path}"
        )

        try:
            # Try AutoModel first to detect the model type
            temp_model = AutoModel.from_pretrained(model_runtime_files_path)
            logger.info(f"AutoModel detected: {type(temp_model).__name__}")

            # Check if it's a CLIP model or needs special handling
            if (
                "CLIP" in type(temp_model).__name__
                or "clip" in type(temp_model).__name__.lower()
            ):
                # For CLIP models, use the specific classes for better control
                del temp_model
                if self.resources.config["model_type"] == "chinese_clip":
                    from transformers import ChineseCLIPModel

                    model = ChineseCLIPModel.from_pretrained(model_runtime_files_path)
                    processor = ChineseCLIPProcessor.from_pretrained(
                        model_runtime_files_path
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_runtime_files_path)
                    logger.info("Using ChineseCLIPModel and ChineseCLIPProcessor")
                else:
                    from transformers import CLIPModel

                    model = CLIPModel.from_pretrained(model_runtime_files_path)
                    processor = CLIPProcessor.from_pretrained(model_runtime_files_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_runtime_files_path)
                    logger.info("Using CLIPModel and CLIPProcessor")
            else:
                # For non-CLIP models, use the AutoModel
                model = temp_model
                processor = AutoProcessor.from_pretrained(model_runtime_files_path)
                tokenizer = AutoTokenizer.from_pretrained(model_runtime_files_path)
                logger.info(
                    f"Using AutoModel ({type(model).__name__}) and AutoProcessor"
                )

        except Exception as e:
            logger.warning(
                f"Failed to load model from local disk: {e}. "
                + "Attempting to load from remote repo fallback"
            )
            # Fallback logic similar to above...
            if self.resources.config["model_type"] == "chinese_clip":
                from transformers import ChineseCLIPModel

                model = ChineseCLIPModel.from_pretrained(model_remote_repo)
                processor = ChineseCLIPProcessor.from_pretrained(
                    model_runtime_files_path
                )
                tokenizer = AutoTokenizer.from_pretrained(model_remote_repo)
                logger.info(
                    "Using ChineseCLIPModel and ChineseCLIPProcessor from remote repo"
                )
            else:
                from transformers import CLIPModel

                model = CLIPModel.from_pretrained(model_remote_repo)
                processor = CLIPProcessor.from_pretrained(model_runtime_files_path)
                tokenizer = AutoTokenizer.from_pretrained(model_remote_repo)
                logger.info("Using CLIPModel and CLIPProcessor from remote repo")

        self._hf_model = model.eval().to(self._device)
        self._hf_processor = processor
        self._hf_tokenizer = tokenizer

        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Processor type: {type(processor).__name__}")
        logger.info(f"Tokenizer type: {type(tokenizer).__name__}")
        logger.info(f"Model has text_embeds: {hasattr(model, 'text_embeds')}")
        logger.info(f"Model has image_embeds: {hasattr(model, 'image_embeds')}")

    def _get_text_features_from_auto_model(self, text: str) -> torch.Tensor:
        """
        Get text features from CLIP model using the correct methods.

        ChineseCLIP models have get_text_features() method that handles everything correctly.
        """
        assert self._hf_model is not None
        assert self._hf_tokenizer is not None

        # Use tokenizer for text-only processing
        if hasattr(self._hf_tokenizer, "__call__"):
            inputs = self._hf_tokenizer(text, return_tensors="pt", padding=True)
        else:
            # Fallback to processor for tokenization
            inputs = self._hf_processor(text=text, return_tensors="pt", padding=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        # For ChineseCLIP models, get_text_features() has a bug where pooled_output is None
        # Always use manual processing which works correctly
        if hasattr(self._hf_model, "text_model") and hasattr(
            self._hf_model, "text_projection"
        ):
            # Manual processing (works reliably for ChineseCLIP)
            outputs = self._hf_model.text_model(**inputs)

            # Try to get pooler output, fallback to CLS token from last_hidden_state
            pooled_output = getattr(outputs, "pooler_output", None)
            if pooled_output is None and hasattr(outputs, "last_hidden_state"):
                pooled_output = outputs.last_hidden_state[:, 0, :]

            if pooled_output is None:
                raise RuntimeError(
                    "Text model did not provide pooler output or CLS token."
                )

            # Apply manual projection to match CLIP standard dimensions
            features = self._hf_model.text_projection(pooled_output)
        elif hasattr(self._hf_model, "get_text_features"):
            # Fallback to get_text_features for non-ChineseCLIP models
            try:
                features = self._hf_model.get_text_features(**inputs)

            except Exception as e:
                logger.error(f"get_text_features failed: {e}")
                raise TorchBackendError(
                    f"Cannot extract text features from model: {type(self._hf_model)}"
                )
        else:
            raise TorchBackendError(
                f"Cannot extract text features from model: {type(self._hf_model)}"
            )

        return features

    def _get_image_features_from_auto_model(self, image: Image.Image) -> torch.Tensor:
        """
        Get image features from CLIP model using the correct methods.

        ChineseCLIP models have get_image_features() method that handles everything correctly.
        """
        assert self._hf_model is not None
        assert self._hf_processor is not None

        # Use processor to process image
        inputs = self._hf_processor(images=image, return_tensors="pt", padding=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        # Use the specific get_image_features method if available
        if hasattr(self._hf_model, "get_image_features"):
            features = self._hf_model.get_image_features(**inputs)
        else:
            raise TorchBackendError(
                f"Cannot extract image features from model: {type(self._hf_model)}"
            )

        return features

    def _load_tokenizer_openclip(
        self, model_name: str
    ) -> Callable[[list[str]], torch.Tensor]:
        """
        Load tokenizer for OpenCLIP from model name, local path, or remote repo.

        Args:
            model_name (str): The model name.

        Returns:
            Callable[[list[str]], torch.Tensor]: The tokenizer function.

        Raises:
            TorchModelLoadingError: If loading fails.
        """
        model_remote_repo = self.resources.source_repo
        model_runtime_files_path = self.resources.runtime_files_path
        try:
            tokenizer_fun = open_clip.get_tokenizer(model_name)
        except Exception as e1:
            logger.warning(
                f"Failed to load OpenCLIP tokenizer with name '{model_name}': {e1}. "
                + "Attempting to load from local directory fallback"
            )
            try:
                tokenizer_fun = open_clip.get_tokenizer(str(model_runtime_files_path))
                logger.info(
                    f"Successfully loaded OpenCLIP tokenizer using local path'{model_runtime_files_path}'"
                )
            except Exception as e2:
                logger.warning(
                    f"Failed to load OpenCLIP tokenizer with name '{model_name}': {e2}. "
                    + "Attempting to load from remote repo fallback"
                )
                try:
                    tokenizer_fun = open_clip.get_tokenizer(
                        "hf-hub:" + model_remote_repo
                    )
                    logger.info(
                        f"Successfully loaded OpenCLIP tokenizer using remote repo'{model_remote_repo}'"
                    )
                except Exception as e3:
                    raise TorchModelLoadingError(
                        f"Failed to load OpenCLIP tokenizer with both "
                        f"'{model_name}' and fallback '{model_remote_repo}': {e3}"
                    ) from e1
        return cast(Callable[[list[str]], torch.Tensor], tokenizer_fun)

    def _load_tokenizer_huggingface(self) -> Callable[[list[str]], torch.Tensor]:
        """deprecated: Use self._hf_tokenizer directly."""
        raise TorchModelLoadingError("_HuggingFace tokenizer loading not implemented")

    @override
    def close(self) -> None:
        """
        Close the backend and free resources.

        Frees CUDA cached memory if using GPU.
        """
        if self._device.type == "cuda":
            # Free cached memory for the device
            torch.cuda.empty_cache()

    # ---------- Encoding API ----------

    @torch.inference_mode()
    @override
    def text_to_vector(self, text: str) -> NDArray[np.float32]:
        """
        Encode a text string into a unit-normalized float32 embedding vector.

        Uses mixed precision (AMP) on GPU for better performance.

        Args:
            text (str): The text string to encode.

        Returns:
            NDArray[np.float32]: The unit-normalized embedding vector.

        Raises:
            InvalidInputError: If text is empty or too long.
            CUDAMemoryError: If CUDA out of memory.
            InferenceError: If encoding fails.
        """
        self._ensure_initialized()

        if not text or not text.strip():
            raise InvalidInputError("text cannot be empty or whitespace only")

        if len(text) > 10000:  # Reasonable length limit
            raise InvalidInputError("text too long (max 10000 characters)")

        def _compute_text_features() -> torch.Tensor:
            if self.resources.source_format == "openclip":
                assert self._openclip_tokenizer is not None
                tokens = self._openclip_tokenizer([text]).to(self._device)
                features = self._openclip_model.encode_text(tokens)
            elif self.resources.source_format == "huggingface":
                features = self._get_text_features_from_auto_model(text)
            else:
                raise TorchModelLoadingError(
                    f"Unsupported source_format: {self.resources.source_format}"
                )
            return features

        try:
            # Use AMP if enabled (GPU only)
            if self._use_amp:
                with torch.autocast(
                    device_type=self._device.type, dtype=self._amp_dtype
                ):
                    feats = _compute_text_features()
            else:
                feats = _compute_text_features()
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

        Args:
            image_bytes (bytes): The image bytes to encode.

        Returns:
            NDArray[np.float32]: The unit-normalized embedding vector.

        Raises:
            InvalidInputError: If image_bytes is empty.
            CUDAMemoryError: If CUDA out of memory.
            InferenceError: If encoding fails.
        """
        self._ensure_initialized()

        if not image_bytes:
            raise InvalidInputError("image_bytes cannot be empty")

        def _compute_image_features() -> torch.Tensor:
            if self.resources.source_format == "openclip":
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                assert self._openclip_preprocess is not None
                image_tensor = (
                    self._openclip_preprocess(image).unsqueeze(0).to(self._device)
                )
                features = self._openclip_model.encode_image(image_tensor)
            elif self.resources.source_format == "huggingface":
                # Convert bytes to PIL Image before passing to AutoModel
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                features = self._get_image_features_from_auto_model(image)
            else:
                raise TorchModelLoadingError(
                    f"Unsupported source_format: {self.resources.source_format}"
                )
            return features

        try:
            # Use AMP if enabled (GPU only)
            if self._use_amp:
                with torch.autocast(
                    device_type=self._device.type, dtype=self._amp_dtype
                ):
                    feats = _compute_image_features()
            else:
                feats = _compute_image_features()

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

        Args:
            images (Sequence[bytes]): Sequence of image bytes to encode.

        Returns:
            NDArray[np.float32]: Array of unit-normalized embedding vectors.

        Raises:
            CUDAMemoryError: If CUDA out of memory.
            InferenceError: If encoding fails.
        """
        if not images:
            return np.empty((0, self.get_info().image_embedding_dim), dtype=np.float32)

        self._ensure_initialized()

        try:
            pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]

            logger.debug("Image Batch To Vectors: Encoding images")

            if self.resources.source_format == "huggingface":
                assert self._hf_model is not None
                assert self._hf_processor is not None

                # Process all images with AutoModel processor
                inputs = self._hf_processor(
                    images=pil_images, return_tensors="pt", padding=True
                ).to(self._device)

                if self._use_amp:
                    with torch.autocast(
                        device_type=self._device.type, dtype=self._amp_dtype
                    ):
                        outputs = self._hf_model(**inputs)
                        if hasattr(outputs, "image_embeds"):
                            feats = outputs.image_embeds
                        elif hasattr(outputs, "pooler_output"):
                            feats = outputs.pooler_output
                        else:
                            feats = outputs.last_hidden_state[:, 0]  # CLS token
                else:
                    outputs = self._hf_model(**inputs)
                    if hasattr(outputs, "image_embeds"):
                        feats = outputs.image_embeds
                    elif hasattr(outputs, "pooler_output"):
                        feats = outputs.pooler_output
                    else:
                        feats = outputs.last_hidden_state[:, 0]  # CLS token
            else:  # openclip
                assert self._openclip_preprocess is not None
                batch = torch.stack(
                    [self._openclip_preprocess(img) for img in pil_images]
                ).to(self._device)
                if self._use_amp:
                    with torch.autocast(
                        device_type=self._device.type, dtype=self._amp_dtype
                    ):
                        feats = self._openclip_model.encode_image(batch)  # type: ignore
                else:
                    feats = self._openclip_model.encode_image(batch)  # type: ignore

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
            texts (Sequence[str]): Sequence of text strings to encode.

        Returns:
            NDArray[np.float32]: Array of unit-normalized embedding vectors.

        Raises:
            InvalidInputError: If any text is empty or too long.
            CUDAMemoryError: If CUDA out of memory.
            InferenceError: If encoding fails.
        """
        self._ensure_initialized()
        if not texts:
            return np.empty((0, self.get_info().text_embedding_dim), dtype=np.float32)

        try:
            # Validate inputs
            for text in texts:
                if not text or not text.strip():
                    raise InvalidInputError("text cannot be empty or whitespace only")
                if len(text) > 10000:  # Reasonable length limit
                    raise InvalidInputError("text too long (max 10000 characters)")

            if self.resources.source_format == "huggingface":
                assert self._hf_model is not None
                assert self._hf_tokenizer is not None

                # Process all texts using get_text_features method
                if hasattr(self._hf_tokenizer, "__call__"):
                    inputs = self._hf_tokenizer(
                        texts, return_tensors="pt", padding=True
                    )
                else:
                    inputs = self._hf_processor(
                        text=texts, return_tensors="pt", padding=True
                    )
                inputs = {key: value.to(self._device) for key, value in inputs.items()}

                if self._use_amp:
                    with torch.autocast(
                        device_type=self._device.type, dtype=self._amp_dtype
                    ):
                        if hasattr(self._hf_model, "get_text_features"):
                            feats = self._hf_model.get_text_features(**inputs)
                        else:
                            raise TorchBackendError(
                                "Model does not have get_text_features method"
                            )
                else:
                    if hasattr(self._hf_model, "get_text_features"):
                        feats = self._hf_model.get_text_features(**inputs)
                    else:
                        raise TorchBackendError(
                            "Model does not have get_text_features method"
                        )
            else:
                # OpenCLIP path
                assert self._openclip_tokenizer is not None
                tokens = self._openclip_tokenizer(list(texts)).to(self._device)
                # Use AMP if enabled (GPU only)
                if self._use_amp:
                    with torch.autocast(
                        device_type=self._device.type, dtype=self._amp_dtype
                    ):
                        feats = self._openclip_model.encode_text(tokens)  # type: ignore
                else:
                    feats = self._openclip_model.encode_text(tokens)  # type: ignore

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

        Returns:
            BackendInfo: Object containing backend information.
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
                "image_size": image_size_str,
                "config_path": str(self.resources.model_root_path / "config.json"),
            },
        )

    @override
    def get_temperature(self) -> float | None:
        """
        Get model temperature (logit scale) for classification calibration.

        Extracts the logit_scale parameter from the loaded model (OpenCLIP or HuggingFace)
        and returns its exponential value as temperature.

        Returns:
            Temperature value if model provides logit_scale, None otherwise.
            Typical CLIP models use values between 1.0 and 10.0.
        """
        try:
            # Try OpenCLIP model first
            if self._openclip_model is not None:
                if hasattr(self._openclip_model, "logit_scale"):
                    return self._openclip_model.logit_scale.exp().item()

            # Try HuggingFace model
            if self._hf_model is not None:
                if hasattr(self._hf_model, "logit_scale"):
                    return self._hf_model.logit_scale.exp().item()

        except Exception as e:
            logger.debug(f"Could not extract temperature from model: {e}")

        return None

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
        if not self._initialized:
            raise RuntimeError(
                "TorchBackend is not initialized. Call initialize() first."
            )
