"""
clip_model.py

Refactored CLIPModelManager to align with BioCLIP's elegant design.
- Implements a clear initialize() pattern.
- Downloads and caches ImageNet text embeddings to a local NPZ file.
- Uses cached text embeddings for fast, default classification.
- Standardizes API for encoding and classification.
"""

import json
import logging
import os
import time
import hashlib
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from backends import BaseClipBackend, TorchBackend, ONNXRTBackend, RKNNBackend

logger = logging.getLogger(__name__)


class CLIPModelManager:
    """
    Manages an OpenCLIP model for image classification using cached ImageNet labels.

    This class mirrors the design of the BioCLIPModelManager, providing a streamlined
    interface for loading a model, caching text embeddings for ImageNet classes,
    and performing inference.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        batch_size: int = 512,
        backend: Optional[BaseClipBackend] = None,
    ) -> None:
        # Resolve config from args with environment overrides
        self.model_name = os.getenv("CLIP_MODEL_NAME", model_name)
        self.pretrained = os.getenv("CLIP_PRETRAINED", pretrained)
        self.model_id = f"{self.model_name}_{self.pretrained}"
        bs_env = os.getenv("CLIP_MAX_BATCH_SIZE")
        self.batch_size = int(bs_env) if bs_env and bs_env.isdigit() else batch_size

        # Define local data paths (names shared; vectors per-backend)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self._base_data_dir = os.path.join(base_dir, "data", "clip")
        os.makedirs(self._base_data_dir, exist_ok=True)
        self.names_filename = os.path.join(self._base_data_dir, "imagenet_class_names.json")

        # Legacy cache path (for migration only)
        self._legacy_vectors_filename = os.path.join(self._base_data_dir, f"{self.model_id}_imagenet_vectors.npz")

        # Per-backend cache paths (resolved after backend.initialize())
        self._cache_runtime_dir: Optional[str] = None
        self._vectors_npz_path: Optional[str] = None
        self._vectors_meta_path: Optional[str] = None

        # Backend (env-driven selection if not provided)
        if backend is None:
            env_backend = (os.getenv("CLIP_BACKEND", "torch") or "torch").lower()
            device_pref = os.getenv("CLIP_DEVICE")
            if env_backend == "onnxrt":
                onnx_image = os.getenv("CLIP_ONNX_IMAGE")
                onnx_text = os.getenv("CLIP_ONNX_TEXT")
                providers = os.getenv("CLIP_ORT_PROVIDERS")
                providers_list = [p.strip() for p in providers.split(",")] if providers else None
                self.backend = ONNXRTBackend(
                    model_name=self.model_name,
                    pretrained=self.pretrained,
                    onnx_image_path=onnx_image,
                    onnx_text_path=onnx_text,
                    providers=providers_list,
                    device_preference=device_pref,
                    max_batch_size=self.batch_size,
                )
            elif env_backend == "rknn":
                rknn_path = os.getenv("CLIP_RKNN_MODEL")
                target = os.getenv("CLIP_RKNN_TARGET", "rk3588")
                self.backend = RKNNBackend(
                    model_name=self.model_name,
                    pretrained=self.pretrained,
                    rknn_model_path=rknn_path,
                    target=target,
                    device_preference=device_pref,
                    max_batch_size=self.batch_size,
                )
            else:
                self.backend = TorchBackend(
                    model_name=self.model_name,
                    pretrained=self.pretrained,
                    device_preference=device_pref,
                    max_batch_size=self.batch_size,
                )
        else:
            self.backend = backend

        # Model and data components
        self.device = self._choose_device()
        self._model: Optional[torch.nn.Module] = None  # preserved for service batch path when using TorchBackend
        self.is_initialized = False

        self.labels: List[str] = []
        self.text_embeddings: Optional[torch.Tensor] = None
        self._load_time: Optional[float] = None

        # Scene classification prompts and embeddings
        self.scene_prompts = [
            "a photo of an animal",
            "a photo of a bird",
            "a photo of an insect",
            "a photo of a human-made object",
            "a photo of a landscape",
            "an abstract painting",
        ]
        self.scene_prompt_embeddings: Optional[torch.Tensor] = None

    @staticmethod
    def _choose_device() -> torch.device:
        """Chooses the best available device (CUDA, MPS, or CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def initialize(self) -> None:
        """
        Loads the backend/model, downloads/caches labels, and computes/loads text embeddings.
        This method must be called before any inference.
        """
        if self.is_initialized:
            return

        t0 = time.time()
        logger.info(f"Initializing backend for {self.model_name} ({self.pretrained})...")

        # 1) Initialize backend
        self.backend.initialize()
        binfo = self.backend.get_info()

        # Derive runtime cache dir and vector paths
        runtime_id = binfo.runtime or "unknown"
        model_id = binfo.model_id or self.model_id
        self._cache_runtime_dir = os.path.join(self._base_data_dir, runtime_id, model_id)
        os.makedirs(self._cache_runtime_dir, exist_ok=True)
        self._vectors_npz_path = os.path.join(self._cache_runtime_dir, "text_vectors.npz")
        self._vectors_meta_path = os.path.join(self._cache_runtime_dir, "text_vectors.meta.json")

        # Preserve torch model handle for service batch path if using TorchBackend
        if isinstance(self.backend, TorchBackend):
            # Access internal model only for service compatibility; safe in our codebase
            self._model = getattr(self.backend, "_model", None)  # type: ignore[attr-defined]



        # 2) Load labels and embeddings (with migration)
        self._load_label_names()
        self._load_or_compute_text_embeddings()

        # 3) Initialize scene embeddings via backend
        self._initialize_scene_embeddings()

        self.is_initialized = True
        self._load_time = time.time() - t0
        logger.info(f"Model initialized in {self._load_time:.2f} seconds (runtime={runtime_id}).")

    def _load_label_names(self) -> None:
        """
        Downloads and caches ImageNet class names from a canonical source.
        The file is a simple JSON list of strings.
        """
        if not os.path.exists(self.names_filename):
            logger.info("Downloading ImageNet class names...")
            try:
                # Using a known reliable source for ImageNet labels as a simple list.
                # Replace with a different repo/file if needed.
                path = hf_hub_download(
                    repo_id="huggingface/label-files",
                    repo_type="dataset",
                    filename="imagenet-1k-id2label.json",
                )
                import shutil
                shutil.copy(path, self.names_filename)
            except Exception as e:
                raise RuntimeError(f"Failed to download label names: {e}")

        with open(self.names_filename, 'r') as f:
            self.labels = json.load(f)
        logger.info(f"Loaded {len(self.labels)} ImageNet class names.")

    def _build_vectors_meta(self, embed_dim: Optional[int] = None) -> Dict[str, Any]:
        binfo = self.backend.get_info()
        model_id = binfo.model_id or self.model_id
        model_name = binfo.model_name or self.model_name
        pretrained = binfo.pretrained or self.pretrained
        labels_str = json.dumps(self.labels, ensure_ascii=False, separators=(",", ":"))
        labels_hash = hashlib.sha256(labels_str.encode("utf-8")).hexdigest()
        meta: Dict[str, Any] = {
            "runtime": binfo.runtime,
            "model_id": model_id,
            "model_name": model_name,
            "pretrained": pretrained,
            "backend_version": binfo.version,
            "labels_hash": labels_hash,
        }
        if embed_dim is not None:
            meta["embed_dim"] = int(embed_dim)
        return meta

    def _compute_and_cache_text_embeddings(self) -> None:
        """Computes text embeddings for labels via backend and saves them to per-backend cache."""
        assert self._vectors_npz_path is not None and self._vectors_meta_path is not None
        logger.info(f"Computing text embeddings for {len(self.labels)} labels via backend...")

        # Use simple prompt template consistent with encode_text
        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in self.labels]
        # TODO: Consider adding a backend batch text API to accelerate this loop.
        vec_list: List[np.ndarray] = []
        for text in prompts:
            vec = self.backend.text_to_vector(text)
            vec_list.append(vec.astype(np.float32, copy=False))
        vecs = np.vstack(vec_list).astype(np.float32, copy=False)

        # Save NPZ and META
        np.savez(self._vectors_npz_path, names=np.array(self.labels, dtype=object), vecs=vecs)
        meta = self._build_vectors_meta(embed_dim=int(vecs.shape[1]))
        with open(self._vectors_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))

        self.text_embeddings = torch.tensor(vecs)
        logger.info(f"Computed and cached text embeddings at {self._vectors_npz_path}.")

    def _load_or_compute_text_embeddings(self) -> None:
        """Loads text embeddings from per-backend cache or computes them. Migrates legacy cache if present."""
        assert self._vectors_npz_path is not None and self._vectors_meta_path is not None

        # 0) Migrate legacy torch cache if exists and new cache missing
        if os.path.exists(self._legacy_vectors_filename) and not os.path.exists(self._vectors_npz_path):
            os.makedirs(os.path.dirname(self._vectors_npz_path), exist_ok=True)
            shutil.move(self._legacy_vectors_filename, self._vectors_npz_path)
            meta = self._build_vectors_meta(embed_dim=None)  # embed_dim inferred on load
            with open(self._vectors_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
            logger.info(f"Migrated legacy vectors to {self._vectors_npz_path} and wrote meta.")

        # 1) Try loading existing cache
        if os.path.exists(self._vectors_npz_path):
            data = np.load(self._vectors_npz_path, allow_pickle=True)
            names = data.get("names")
            vecs = data.get("vecs")
            if names is not None and vecs is not None and names.tolist() == self.labels:
                self.text_embeddings = torch.tensor(vecs.astype(np.float32, copy=False))
                logger.info(f"Loaded cached text embeddings from {self._vectors_npz_path}.")
                return
            else:
                logger.warning("Cached labels mismatch or invalid format; recomputing embeddings.")

        # 2) Compute and save
        self._compute_and_cache_text_embeddings()

    def _initialize_scene_embeddings(self) -> None:
        """Initialize embeddings for simple scene prompts using backend; store on CPU."""
        logger.info("Initializing scene classification embeddings via backend.")
        try:
            vecs = [self.backend.text_to_vector(p) for p in self.scene_prompts]
            arr = np.vstack([v.astype(np.float32, copy=False) for v in vecs])
            self.scene_prompt_embeddings = torch.tensor(arr, device="cpu")
        except Exception as e:
            logger.error(f"Failed to initialize scene embeddings: {e}")
            self.scene_prompt_embeddings = None

    @staticmethod
    def _unit_normalize(t: torch.Tensor) -> torch.Tensor:
        """Normalizes a tensor to unit length."""
        return t / t.norm(dim=-1, keepdim=True)

    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """Encodes image bytes into a unit-normalized embedding vector via backend."""
        self._ensure_initialized()
        return self.backend.image_to_vector(image_bytes).astype(np.float32, copy=False)

    def encode_text(self, text: str) -> np.ndarray:
        """Encodes a single string of text into a unit-normalized embedding vector via backend."""
        self._ensure_initialized()
        prompt = f"a photo of a {text}"
        return self.backend.text_to_vector(prompt).astype(np.float32, copy=False)

    def classify_image(
        self,
        image_bytes: bytes,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Classifies an image against the cached ImageNet labels.

        Args:
            image_bytes: The image to classify, in bytes.
            top_k: The number of top results to return.

        Returns:
            A list of (label, probability) tuples.
        """
        self._ensure_initialized()
        img_vec = torch.tensor(self.encode_image(image_bytes), device=self.device).unsqueeze(0)

        if self.text_embeddings is None:
            raise RuntimeError("Text embeddings are not available.")

        text_emb = self.text_embeddings.to(self.device)
        with torch.no_grad():
            # Similarities -> Probabilities
            sims = (100.0 * img_vec @ text_emb.T).softmax(dim=-1).squeeze(0)
            probs, idxs = sims.topk(min(top_k, sims.numel()))

        return [(self.labels[idx], float(prob)) for prob, idx in zip(probs, idxs)]

    def classify_scene(self, image_bytes: bytes) -> Tuple[str, float]:
        """
        Performs a high-level scene classification on an image.

        Args:
            image_bytes: The image to classify, in bytes.

        Returns:
            A tuple of (scene_label, confidence_score).
        """
        self._ensure_initialized()
        if self.scene_prompt_embeddings is None:
            raise RuntimeError("Scene embeddings not initialized.")

        assert self.scene_prompt_embeddings is not None
        img_vec = torch.tensor(self.encode_image(image_bytes), device='cpu').unsqueeze(0)

        with torch.no_grad():
            sims = (img_vec @ self.scene_prompt_embeddings.T).softmax(-1).squeeze(0)
            confidence, idx = sims.max(dim=0)

        return self.scene_prompts[int(idx.item())], float(confidence.item())

    def info(self) -> Dict[str, Any]:
        """Returns a dictionary with information about the loaded model."""
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "device": str(self.device),
            "is_initialized": self.is_initialized,
            "load_time_seconds": self._load_time,
            "label_count": len(self.labels),
            "vectors_cache_path": self._vectors_npz_path,
        }

    def _ensure_initialized(self) -> None:
        """Raises a RuntimeError if the model is not initialized."""
        if not self.is_initialized:
            raise RuntimeError("Model is not initialized. Call initialize() before inference.")

    def _ensure_initialized_for_computation(self) -> None:
        """Kept for backward compatibility; backend handles model readiness."""
        if not self.backend.is_initialized:
            raise RuntimeError("Backend must be initialized before this operation.")
