"""
clip_model.py

Refactored CLIPModelManager to align with Lumen architecture contracts.
- Separates concerns between Model Manager and Backend layers
- Uses standard types (np.ndarray) instead of torch-specific types
- Implements proper error handling according to Lumen contracts
- Focuses on business logic: data preparation, caching, and result formatting
"""

import json
import logging
import os
import time
from typing import Any, Optional

import numpy as np
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray

from backends import BaseClipBackend

logger = logging.getLogger(__name__)


class ModelDataNotFoundError(Exception):
    """Raised when model-specific data (labels, embeddings) cannot be found or loaded."""

    pass


class CacheCorruptionError(Exception):
    """Raised when cached data is corrupted or incompatible."""

    pass


class LabelMismatchError(Exception):
    """Raised when cached embeddings don't match current labels."""

    pass


class CLIPModelManager:
    """
    Manages CLIP model business logic for image classification using cached ImageNet labels.

    This class follows Lumen architecture contracts:
    - Receives a BaseClipBackend instance for inference
    - Manages model-specific labels and embeddings cache
    - Handles data preprocessing/postprocessing
    - Returns business-level results using standard types
    """

    def __init__(
        self,
        backend: BaseClipBackend,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        batch_size: int = 512,
    ) -> None:
        """
        Initialize CLIP Model Manager.

        Args:
            backend: Backend instance implementing BaseClipBackend interface
            model_name: Model architecture name
            pretrained: Pretrained weights identifier
            batch_size: Batch size for processing
        """
        # Backend dependency injection (following Lumen contracts)
        self.backend: BaseClipBackend = backend

        # Model configuration from args with environment overrides
        self.model_name: str = os.getenv("CLIP_MODEL_NAME", model_name)
        self.pretrained: str = os.getenv("CLIP_PRETRAINED", pretrained)
        self.model_id: str = f"{self.model_name}_{self.pretrained}"

        bs_env = os.getenv("CLIP_MAX_BATCH_SIZE")
        self.batch_size: int = (
            int(bs_env) if bs_env and bs_env.isdigit() else batch_size
        )

        # Data management paths
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self._base_data_dir: str = os.path.join(base_dir, "data", "clip")
        os.makedirs(self._base_data_dir, exist_ok=True)

        # ImageNet labels file (shared across backends)
        self.names_filename: str = os.path.join(
            self._base_data_dir, "imagenet_labels.json"
        )

        # Per-backend cache paths (resolved after backend initialization)
        self._cache_runtime_dir: str | None = None
        self._vectors_npz_path: str | None = None
        self._vectors_meta_path: str | None = None

        # Business data (using standard types as per Lumen contracts)
        self.labels: list[str] = []
        self.text_embeddings: NDArray[np.float32] | None = None
        self._load_time: float | None = None
        self.is_initialized: bool = False

        # Scene classification prompts (business logic)
        self.scene_prompts: list[str] = [
            "a photo of a person",
            "a photo of an animal",
            "a photo of a vehicle",
            "a photo of food",
            "a photo of a building",
            "a photo of nature",
            "a photo of an object",
            "a photo of a landscape",
            "an abstract painting",
        ]
        self.scene_prompt_embeddings: NDArray[np.float32] | None = None

    def initialize(self) -> None:
        """
        Initialize the model manager: load backend, cache labels, and compute embeddings.
        Must be called before any inference operations.

        Raises:
            ModelDataNotFoundError: If required model data cannot be loaded
            CacheCorruptionError: If cached data is corrupted
        """
        if self.is_initialized:
            return

        t0 = time.time()
        logger.info(
            f"Initializing CLIP Model Manager for {self.model_name} ({self.pretrained})..."
        )

        try:
            # 1) Initialize backend (delegate device/runtime concerns to backend)
            self.backend.initialize()

            # 2) Setup per-backend cache paths
            self._setup_cache_paths()

            # 3) Load/download labels and compute/load embeddings
            self._load_label_names()
            self._load_or_compute_text_embeddings()

            # 4) Initialize scene embeddings
            self._initialize_scene_embeddings()

            self.is_initialized = True
            self._load_time = time.time() - t0
            logger.info(
                f"CLIP Model Manager initialized in {self._load_time:.2f} seconds"
            )

        except Exception as e:
            logger.error(f"Failed to initialize CLIP Model Manager: {e}")
            raise ModelDataNotFoundError(f"Model initialization failed: {e}") from e

    def _setup_cache_paths(self) -> None:
        """Setup per-backend cache directory structure."""
        backend_info = self.backend.get_info()
        runtime_id = backend_info.runtime or "unknown"
        model_id = backend_info.model_id or self.model_id

        self._cache_runtime_dir = os.path.join(
            self._base_data_dir, runtime_id, model_id
        )
        os.makedirs(self._cache_runtime_dir, exist_ok=True)

        self._vectors_npz_path = os.path.join(
            self._cache_runtime_dir, "text_vectors.npz"
        )
        self._vectors_meta_path = os.path.join(
            self._cache_runtime_dir, "text_vectors.meta.json"
        )

    def _load_label_names(self) -> None:
        """
        Download and cache ImageNet class names from a canonical source.

        Raises:
            ModelDataNotFoundError: If labels cannot be downloaded or loaded
        """
        if not os.path.exists(self.names_filename):
            logger.info("Downloading ImageNet class names...")
            try:
                # Download from huggingface repository
                path = hf_hub_download(
                    repo_id="huggingface/label-files",
                    filename="imagenet-1k-id2label.json",
                    repo_type="dataset",
                )

                # Convert to simple list format and save
                with open(path, "r") as f:
                    id2label = json.load(f)

                # Extract labels in order (assuming keys are "0", "1", "2", ...)
                labels = [id2label[str(i)] for i in range(len(id2label))]

                with open(self.names_filename, "w") as f:
                    json.dump(labels, f, ensure_ascii=False, indent=2)

            except Exception as e:
                raise ModelDataNotFoundError(
                    f"Failed to download ImageNet labels: {e}"
                ) from e

        # Load labels
        try:
            with open(self.names_filename, "r") as f:
                self.labels = json.load(f)
            logger.info(f"Loaded {len(self.labels)} ImageNet class names")
        except Exception as e:
            raise ModelDataNotFoundError(f"Failed to load ImageNet labels: {e}") from e

    def _compute_and_cache_text_embeddings(self) -> None:
        """
        Compute text embeddings for all labels and cache to disk.

        Raises:
            CacheCorruptionError: If caching fails
        """
        assert (
            self._vectors_npz_path is not None and self._vectors_meta_path is not None
        )

        try:
            logger.info(f"Computing text embeddings for {len(self.labels)} labels...")

            # Compute embeddings via backend (using standard interface)
            prompts = [f"a photo of a {label}" for label in self.labels]
            embeddings_list: list[NDArray[np.float32]] = []

            for prompt in prompts:
                embedding = self.backend.text_to_vector(prompt)
                embeddings_list.append(embedding)

            # Stack into single array
            embeddings_array = np.vstack(embeddings_list).astype(np.float32, copy=False)

            # Cache embeddings and metadata
            np.savez(
                self._vectors_npz_path,
                names=np.array(self.labels, dtype=object),
                vecs=embeddings_array,
            )

            backend_info = self.backend.get_info()
            metadata = {
                "runtime": backend_info.runtime,
                "model_id": self.model_id,
                "num_labels": len(self.labels),
                "embedding_dim": embeddings_array.shape[1],
            }

            with open(self._vectors_meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, separators=(",", ":"))

            # Store in standard format (np.ndarray)
            self.text_embeddings = embeddings_array
            logger.info(
                f"Computed and cached text embeddings to {self._vectors_npz_path}"
            )

        except Exception as e:
            raise CacheCorruptionError(
                f"Failed to compute and cache embeddings: {e}"
            ) from e

    def _load_or_compute_text_embeddings(self) -> None:
        """
        Load cached text embeddings or compute them if cache is invalid/missing.

        Raises:
            LabelMismatchError: If cached embeddings don't match current labels
            CacheCorruptionError: If cached data is corrupted
        """
        assert self._vectors_npz_path is not None

        # Try to load from cache first
        if os.path.exists(self._vectors_npz_path):
            try:
                data = np.load(self._vectors_npz_path, allow_pickle=True)
                cached_names = data.get("names")
                cached_vecs = data.get("vecs")

                if (
                    cached_names is not None
                    and cached_vecs is not None
                    and cached_names.tolist() == self.labels
                ):
                    self.text_embeddings = cached_vecs.astype(np.float32, copy=False)
                    logger.info(
                        f"Loaded cached text embeddings from {self._vectors_npz_path}"
                    )
                    return
                else:
                    logger.warning(
                        "Cached labels mismatch current labels; recomputing embeddings"
                    )
                    raise LabelMismatchError(
                        "Cached embeddings don't match current labels"
                    )

            except (Exception, KeyError, LabelMismatchError) as e:
                logger.warning(f"Cache validation failed: {e}; recomputing embeddings")

        # Compute embeddings if cache is invalid/missing
        self._compute_and_cache_text_embeddings()

    def _initialize_scene_embeddings(self) -> None:
        """Initialize scene classification embeddings using backend."""
        try:
            embeddings_list = []
            for prompt in self.scene_prompts:
                embedding = self.backend.text_to_vector(prompt)
                embeddings_list.append(embedding)

            self.scene_prompt_embeddings = np.vstack(embeddings_list).astype(
                np.float32, copy=False
            )
            logger.info("Initialized scene classification embeddings")

        except Exception as e:
            logger.error(f"Failed to initialize scene embeddings: {e}")
            self.scene_prompt_embeddings = None

    def encode_image(self, image_bytes: bytes) -> NDArray[np.float32]:
        """
        Encode image bytes into a unit-normalized embedding vector.

        Args:
            image_bytes: Raw image data in bytes

        Returns:
            Unit-normalized embedding vector as numpy array

        Raises:
            RuntimeError: If model is not initialized
        """
        self._ensure_initialized()
        return self.backend.image_to_vector(image_bytes)

    def encode_text(self, text: str) -> NDArray[np.float32]:
        """
        Encode text into a unit-normalized embedding vector.

        Args:
            text: Text to encode

        Returns:
            Unit-normalized embedding vector as numpy array

        Raises:
            RuntimeError: If model is not initialized
        """
        self._ensure_initialized()
        prompt = f"a photo of a {text}"
        return self.backend.text_to_vector(prompt)

    def classify_image(
        self, image_bytes: bytes, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Classify an image against cached ImageNet labels.

        Args:
            image_bytes: Raw image data in bytes
            top_k: Number of top results to return

        Returns:
            List of (label, probability) tuples sorted by probability (descending)

        Raises:
            RuntimeError: If model is not initialized
        """
        self._ensure_initialized()

        if self.text_embeddings is None:
            raise RuntimeError("Text embeddings are not available")

        # Get image embedding via backend
        img_embedding = self.encode_image(image_bytes)

        # Compute similarities using numpy (standard operations)
        img_embedding = img_embedding / np.linalg.norm(
            img_embedding
        )  # Ensure unit norm
        text_embeddings_norm = self.text_embeddings / np.linalg.norm(
            self.text_embeddings, axis=1, keepdims=True
        )

        # Compute cosine similarities
        similarities = np.dot(img_embedding, text_embeddings_norm.T)

        # Convert to probabilities (softmax)
        similarities_scaled = similarities * 100.0  # Temperature scaling
        exp_sims = np.exp(
            similarities_scaled - np.max(similarities_scaled)
        )  # Numerical stability
        probabilities = exp_sims / np.sum(exp_sims)

        # Get top-k results
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        results = [(self.labels[idx], float(probabilities[idx])) for idx in top_indices]

        return results

    def classify_scene(self, image_bytes: bytes) -> tuple[str, float]:
        """
        Perform high-level scene classification on an image.

        Args:
            image_bytes: Raw image data in bytes

        Returns:
            Tuple of (scene_label, confidence_score)

        Raises:
            RuntimeError: If model is not initialized or scene embeddings unavailable
        """
        self._ensure_initialized()

        if self.scene_prompt_embeddings is None:
            raise RuntimeError("Scene embeddings are not available")

        # Get image embedding via backend
        img_embedding = self.encode_image(image_bytes)

        # Compute similarities using numpy
        img_embedding = img_embedding / np.linalg.norm(img_embedding)
        scene_embeddings_norm = self.scene_prompt_embeddings / np.linalg.norm(
            self.scene_prompt_embeddings, axis=1, keepdims=True
        )

        similarities = np.dot(img_embedding, scene_embeddings_norm.T)

        # Convert to probabilities
        exp_sims = np.exp(similarities - np.max(similarities))
        probabilities = exp_sims / np.sum(exp_sims)

        # Get best match
        best_idx = np.argmax(probabilities)
        scene_label = (
            self.scene_prompts[best_idx].replace("a photo of ", "").replace("an ", "")
        )
        confidence = float(probabilities[best_idx])

        return scene_label, confidence

    def info(self) -> dict[str, str | int | float | bool | dict[str, Any]]:
        """
        Return model manager information.

        Returns:
            Dictionary containing model metadata and performance info
        """
        backend_info: dict[str, Any] = {}
        if hasattr(self, "backend"):
            info = self.backend.get_info()
            backend_info = {
                "runtime": info.runtime,
                "model_id": info.model_id,
                "model_name": info.model_name,
                "version": info.version,
            }

        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "model_id": self.model_id,
            "num_labels": len(self.labels),
            "load_time": self._load_time,
            "is_initialized": self.is_initialized,
            "backend_info": backend_info,
            "scene_classification_available": self.scene_prompt_embeddings is not None,
        }

    def _ensure_initialized(self) -> None:
        """
        Ensure the model manager is initialized before inference.

        Raises:
            RuntimeError: If model is not initialized
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Model manager not initialized. Call initialize() first."
            )
