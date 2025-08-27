"""
TorchBackend: an OpenCLIP-based backend that implements the BaseClipBackend
interface using PyTorch.

This backend mirrors the behavior of the current OpenCLIP usage in the project:
- Loads a CLIP-like model via open_clip.create_model_and_transforms
- Uses open_clip.get_tokenizer for text tokenization
- Produces unit-normalized float32 embeddings for both text and images
- Supports true batched image embedding on GPU/CPU

Notes:
- Returned vectors are always L2-normalized float32.
- image_batch_to_vectors performs a single forward pass for efficiency.
"""

from __future__ import annotations

import io
import time
from typing import List, Optional, Sequence, Callable, cast

import numpy as np
from PIL import Image

import torch
import open_clip

from .base import BaseClipBackend, BackendInfo


class TorchBackend(BaseClipBackend):
    """
    A PyTorch/OpenCLIP backend implementing BaseClipBackend.

    Args:
        model_name: OpenCLIP model architecture (default "ViT-B-32").
        pretrained: OpenCLIP pretrained weights tag (default "laion2b_s34b_b79k").
        model_id: Optional stable model identifier. Defaults to "{model_name}_{pretrained}".
        device_preference: Optional hint for device selection ("cuda", "mps", "cpu").
        max_batch_size: Optional hint for batch size; not enforced by backend.
        cache_dir: Unused by this backend (included for interface parity).

    Behavior:
        - initialize() loads the model, tokenizer, and preprocess pipeline, and moves model to device.
        - text_to_vector() encodes a text prompt via the model's text encoder.
        - image_to_vector() decodes and preprocesses image bytes, then encodes via the image encoder.
        - image_batch_to_vectors() performs a single batched forward pass for multiple images.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        pretrained: Optional[str] = None,
        model_id: Optional[str] = None,
        device_preference: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        # Provide defaults consistent with current CLIPModelManager
        model_name = model_name or "ViT-B-32"
        pretrained = pretrained or "laion2b_s34b_b79k"
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            model_id=model_id,
            device_preference=device_preference,
            max_batch_size=max_batch_size,
            cache_dir=cache_dir,
        )

        # Runtime objects
        self._device: torch.device = self._select_device(device_preference)
        self._model: Optional[torch.nn.Module] = None
        self._preprocess: Optional[Callable[[Image.Image], torch.Tensor]] = None  # torchvision-like transform returned by open_clip
        self._tokenizer: Optional[Callable[[List[str]], torch.Tensor]] = None
        self._load_time_seconds: Optional[float] = None

        # Cache derived identifiers
        if self._model_id is None:
            self._model_id = f"{self._model_name}_{self._pretrained}"

    # ---------- Lifecycle ----------

    def initialize(self) -> None:
        if self._initialized:
            return

        t0 = time.time()
        # Load model, transforms, tokenizer
        model_name_str: str = self._model_name or "ViT-B-32"
        pretrained_str: str = self._pretrained or "laion2b_s34b_b79k"
        model_obj, _, preprocess = open_clip.create_model_and_transforms(
            model_name_str, pretrained=pretrained_str
        )
        tokenizer_fun = open_clip.get_tokenizer(model_name_str)

        model_module = cast(torch.nn.Module, model_obj)
        model_module.eval().to(self._device)
        self._model = model_module
        self._preprocess = cast(Callable[[Image.Image], torch.Tensor], preprocess)
        self._tokenizer = cast(Callable[[List[str]], torch.Tensor], tokenizer_fun)

        self._load_time_seconds = time.time() - t0
        self._initialized = True

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
    def text_to_vector(self, text: str) -> np.ndarray:
        """
        Encode a text string into a unit-normalized float32 embedding vector.
        Uses a CLIP-style prompt template for consistency with the current implementation.
        """
        self._ensure_initialized()

        # Create a simple CLIP-compatible prompt
        prompt = text
        assert self._tokenizer is not None
        assert self._model is not None

        tokenizer = cast(Callable[[List[str]], torch.Tensor], self._tokenizer)
        tokens = tokenizer([prompt]).to(self._device)
        feats = self._model.encode_text(tokens)  # type: ignore[attr-defined]
        feats = feats / feats.norm(dim=-1, keepdim=True)

        vec = feats.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        return vec

    @torch.inference_mode()
    def image_to_vector(self, image_bytes: bytes) -> np.ndarray:
        """
        Encode image bytes (RGB) into a unit-normalized float32 embedding vector.
        """
        self._ensure_initialized()
        assert self._preprocess is not None
        assert self._model is not None

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        preprocess = cast(Callable[[Image.Image], torch.Tensor], self._preprocess)
        tensor = preprocess(img).unsqueeze(0).to(self._device)
        feats = self._model.encode_image(tensor)  # type: ignore[attr-defined]
        feats = feats / feats.norm(dim=-1, keepdim=True)

        vec = feats.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        return vec

    @torch.inference_mode()
    def image_batch_to_vectors(self, images: Sequence[bytes]) -> np.ndarray:
        """
        Encode a list of image bytes using a single batched forward pass.
        Falls back to BaseClipBackend's sequential implementation if images is empty.
        """
        if not images:
            return super().image_batch_to_vectors(images)

        self._ensure_initialized()
        assert self._preprocess is not None
        assert self._model is not None
        preprocess = cast(Callable[[Image.Image], torch.Tensor], self._preprocess)

        # Decode and preprocess to a batch tensor [N, C, H, W]
        tensors: List[torch.Tensor] = []
        for b in images:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            t = preprocess(img)
            tensors.append(t)

        batch = torch.stack(tensors, dim=0).to(self._device)
        feats = self._model.encode_image(batch)  # type: ignore[attr-defined]
        feats = feats / feats.norm(dim=-1, keepdim=True)

        arr = feats.detach().cpu().numpy().astype(np.float32, copy=False)
        return arr

    # ---------- Metadata ----------

    def get_info(self) -> BackendInfo:
        """
        Report runtime and model metadata. Embedding dims are left optional; they
        can vary by model and are discoverable at runtime if needed.
        """
        precisions = ["fp32"]
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            precisions.append("fp16")

        version = getattr(open_clip, "__version__", None)
        return BackendInfo(
            runtime="torch",
            device=str(self._device),
            model_id=self._model_id,
            model_name=self._model_name,
            pretrained=self._pretrained,
            version=str(version) if version is not None else None,
            image_embedding_dim=None,
            text_embedding_dim=None,
            precisions=precisions,
            max_batch_size=self._max_batch_size,
            supports_image_batch=True,
            extra={"library": "open-clip-torch"},
        )

    # ---------- Helpers ----------

    @staticmethod
    def _select_device(preference: Optional[str]) -> torch.device:
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
        if not self._initialized or self._model is None or self._preprocess is None or self._tokenizer is None:
            raise RuntimeError("TorchBackend is not initialized. Call initialize() first.")
