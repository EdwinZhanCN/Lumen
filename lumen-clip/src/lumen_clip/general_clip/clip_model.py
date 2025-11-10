"""
CLIP Model Management Module

This module provides the `CLIPModelManager` class, which encapsulates the business logic for managing CLIP-based image classification and embedding tasks. It handles model initialization, label and embedding management, image and text encoding, and both fine-grained and high-level scene classification. The module also defines custom exceptions for error handling related to model data, cache integrity, and label mismatches.

Classes:
    - ModelDataNotFoundError: Raised when required model data is missing.
    - CacheCorruptionError: Raised when cached data is corrupted or incompatible.
    - LabelMismatchError: Raised when cached embeddings do not match current labels.
    - CLIPModelManager: Main class for managing CLIP model inference and classification.

Dependencies:
    - Requires a backend implementing the BaseClipBackend interface.
    - Uses ModelResources for pre-loaded model data and embeddings.
"""

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lumen_clip.backends import BaseClipBackend
from lumen_clip.resources.loader import ModelResources
from lumen_clip.models import ModelInfo, BackendInfo

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
    Manages CLIP model business logic for image classification.

    This class:
    - Receives a BaseClipBackend instance for inference
    - Uses pre-loaded labels and embeddings from ModelResources
    - Handles classification and embedding tasks
    - Returns business-level results using standard types
    """

    def __init__(
        self,
        backend: BaseClipBackend,
        resources: ModelResources,
    ) -> None:
        """
        Initialize CLIP Model Manager.

        Args:
            backend: Backend instance implementing BaseClipBackend interface
            resources: ModelResources with pre-loaded data
        """
        # Backend dependency injection
        self.backend: BaseClipBackend = backend
        self.resources: ModelResources = resources

        # Use pre-loaded labels and embeddings from resources
        self.labels: list[str] = (
            resources.labels.tolist() if resources.labels is not None else []
        )
        self.text_embeddings: NDArray[np.float32] | None = resources.label_embeddings

        # Model identification
        self.model_id: str = f"{resources.model_name}_{resources.runtime}"
        self.supports_classification: bool = resources.has_classification_support()

        # Initialization state
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
        ]
        self.scene_prompt_embeddings: NDArray[np.float32] | None = None

    def initialize(self) -> None:
        """
        Initialize the model manager: initialize backend and compute embeddings if needed.
        Must be called before any inference operations.
        """
        if self.is_initialized:
            return

        t0 = time.time()
        logger.info(
            f"Initializing CLIP Model Manager for {self.resources.model_name}..."
        )

        try:
            # 1) Initialize backend
            self.backend.initialize()

            # 2) If no pre-computed text embeddings and we have labels, compute them
            if self.supports_classification and self.text_embeddings is None:
                logger.info("No pre-computed embeddings found, computing on-the-fly...")
                self._compute_text_embeddings()

            # 3) Initialize scene embeddings (always computed, not dataset-dependent)
            self._initialize_scene_embeddings()

            self.is_initialized = True
            self._load_time = time.time() - t0

            if self.supports_classification:
                logger.info(
                    f"✅ CLIP Model Manager initialized in {self._load_time:.2f}s "
                    f"({len(self.labels)} classes)"
                )
            else:
                logger.info(
                    f"✅ CLIP Model Manager initialized in {self._load_time:.2f}s "
                    f"(embed-only mode)"
                )

        except Exception as e:
            logger.error(f"Failed to initialize CLIP Model Manager: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def _compute_text_embeddings(self) -> None:
        """
        Compute text embeddings for all labels (called if not pre-computed).
        """
        if not self.labels:
            logger.warning("No labels available, skipping embedding computation")
            return

        logger.info(f"Computing text embeddings for {len(self.labels)} labels...")

        # Use batch processing if backend supports it
        prompts = [f"a photo of a {label}" for label in self.labels]

        try:
            # Try batch processing first
            embeddings_array = self.backend.text_batch_to_vectors(prompts)
            self.text_embeddings = embeddings_array.astype(np.float32, copy=False)
            logger.info("Text embeddings computed successfully")
        except Exception as e:
            logger.warning(f"Batch processing failed: {e}, falling back to sequential")
            # Fallback to sequential processing
            embeddings_list: list[NDArray[np.float32]] = []
            for prompt in prompts:
                embedding = self.backend.text_to_vector(prompt)
                embeddings_list.append(embedding)
            self.text_embeddings = np.vstack(embeddings_list).astype(
                np.float32, copy=False
            )

    def _initialize_scene_embeddings(self) -> None:
        """Initialize scene classification embeddings using backend."""
        try:
            # Use batch processing if available
            try:
                self.scene_prompt_embeddings = self.backend.text_batch_to_vectors(
                    self.scene_prompts
                )
                logger.info("Initialized scene classification embeddings (batched)")
            except Exception as e:
                logger.warning(f"Batch processing failed: {e}")
                # Fallback to sequential
                embeddings_list = []
                for prompt in self.scene_prompts:
                    embedding = self.backend.text_to_vector(prompt)
                    embeddings_list.append(embedding)
                self.scene_prompt_embeddings = np.vstack(embeddings_list).astype(
                    np.float32, copy=False
                )
                logger.info("Initialized scene classification embeddings (sequential)")

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
        Classify an image against ImageNet labels.

        Args:
            image_bytes: Raw image data in bytes
            top_k: Number of top results to return

        Returns:
            List of (label, probability) tuples sorted by probability (descending)

        Raises:
            RuntimeError: If model is not initialized or classification not supported
        """
        # Temporarily enable debug logging for this method
        logging.getLogger("lumen_clip.backends.onnxrt_backend").setLevel(logging.DEBUG)
        logging.getLogger("lumen_clip.general_clip.clip_model").setLevel(logging.DEBUG)

        self._ensure_initialized()

        if not self.supports_classification:
            raise RuntimeError(
                "Classification not supported: no dataset loaded. "
                "Ensure ImageNet_1k.npz exists in the model directory."
            )

        if self.text_embeddings is None:
            raise RuntimeError("Text embeddings are not available")

        # Get image embedding via backend
        img_embedding = self.encode_image(image_bytes)

        # Check for invalid values in image embedding
        if np.any(np.isnan(img_embedding)) or np.any(np.isinf(img_embedding)):
            logger.error(f"Invalid values in image embedding: NaN={np.any(np.isnan(img_embedding))}, Inf={np.any(np.isinf(img_embedding))}")
            logger.error(f"Embedding stats: min={np.min(img_embedding):.6f}, max={np.max(img_embedding):.6f}, mean={np.mean(img_embedding):.6f}")
            raise RuntimeError(f"Image embedding contains invalid values (NaN/Inf)")

        # Compute similarities using numpy (standard operations)
        img_norm = np.linalg.norm(img_embedding)
        if img_norm == 0:
            raise RuntimeError("Image embedding has zero norm - cannot normalize")

        img_embedding = img_embedding / img_norm  # Ensure unit norm

        logger.debug(f"Image embedding norm after normalization: {np.linalg.norm(img_embedding)}")

        # Check text embeddings for invalid values
        if np.any(np.isnan(self.text_embeddings)) or np.any(np.isinf(self.text_embeddings)):
            logger.error(f"Invalid values in text embeddings: NaN={np.any(np.isnan(self.text_embeddings))}, Inf={np.any(np.isinf(self.text_embeddings))}")
            raise RuntimeError(f"Text embeddings contain invalid values (NaN/Inf)")

        text_embeddings_norm = self.text_embeddings / np.linalg.norm(
            self.text_embeddings, axis=1, keepdims=True
        )

        # Check for zero norms in text embeddings
        text_norms = np.linalg.norm(self.text_embeddings, axis=1)
        zero_norm_count = np.sum(text_norms == 0)
        if zero_norm_count > 0:
            logger.error(f"Found {zero_norm_count} text embeddings with zero norm")
            raise RuntimeError(f"{zero_norm_count} text embeddings have zero norm - cannot normalize")

        # Compute cosine similarities
        similarities = np.dot(img_embedding, text_embeddings_norm.T)

        logger.debug(f"Image embedding norm: {np.linalg.norm(img_embedding)}")
        logger.debug(f"Text embeddings shape: {text_embeddings_norm.shape}")
        logger.debug(f"Similarities stats: min={np.min(similarities):.4f}, max={np.max(similarities):.4f}, mean={np.mean(similarities):.4f}")
        logger.debug(f"Any NaN in similarities: {np.any(np.isnan(similarities))}")

        # Convert to probabilities (softmax)
        similarities_scaled = similarities * 100.0  # Temperature scaling
        exp_sims = np.exp(
            similarities_scaled - np.max(similarities_scaled)
        )  # Numerical stability
        probabilities = exp_sims / np.sum(exp_sims)

        logger.debug(f"Probabilities stats: min={np.min(probabilities):.4f}, max={np.max(probabilities):.4f}, sum={np.sum(probabilities):.4f}")
        logger.debug(f"Any NaN in probabilities: {np.any(np.isnan(probabilities))}")

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

    def info(self) -> ModelInfo:
        """
        Return model manager information.

        Returns:
            ModelInfo containing model metadata and performance info
        """
        backend_info = None
        if hasattr(self, "backend"):
            info = self.backend.get_info()
            backend_info = BackendInfo(
                runtime=info.runtime,
                model_id=info.model_id,
                model_name=info.model_name,
                version=info.version,
                image_embedding_dim=info.image_embedding_dim,
                text_embedding_dim=info.text_embedding_dim,
                device=getattr(info, 'device', None),
                precisions=getattr(info, 'precisions', None),
            )

        return ModelInfo(
            model_name=self.resources.model_name,
            model_id=self.model_id,
            supports_classification=self.supports_classification,
            is_initialized=self.is_initialized,
            load_time=self._load_time,
            num_labels=len(self.labels) if self.labels else None,
            backend_info=backend_info,
            scene_classification_available=self.scene_prompt_embeddings is not None,
        )

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
