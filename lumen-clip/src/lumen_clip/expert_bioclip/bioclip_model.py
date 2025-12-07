"""
bioclip_model.py

Refactored BioCLIPModelManager to use ModelResources from lumen-resources.
- Uses pre-loaded labels and embeddings from ModelResources
- No longer downloads from HuggingFace
- Simplified initialization (no caching logic needed)
- Focuses on business logic: classification and embedding
"""

from __future__ import annotations
import logging
import time
from typing import Any

import numpy as np
from lumen_clip.backends import BaseClipBackend
from lumen_clip.resources.loader import ModelResources
from lumen_clip.models import ModelInfo, BackendInfo
from numpy.typing import NDArray

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


class BioCLIPModelManager:
    """
    Manages BioCLIP model business logic for biological species classification.

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
        Initialize BioCLIP Model Manager.

        Args:
            backend: Backend instance implementing BaseClipBackend interface
            resources: ModelResources with pre-loaded data
        """
        # Backend dependency injection
        self.backend: BaseClipBackend = backend
        self.resources: ModelResources = resources

        # Fixed BioCLIP version
        self.model_version: str = "bioclip2"

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

    def initialize(self) -> None:
        """
        Initialize the model manager: initialize backend and compute embeddings if needed.
        Must be called before any inference operations.
        """
        if self.is_initialized:
            return

        t0 = time.time()
        logger.info(
            f"Initializing BioCLIP Model Manager for {self.resources.model_name}..."
        )

        try:
            # 1) Initialize backend
            self.backend.initialize()

            # 2) If no pre-computed text embeddings and we have labels, compute them
            if self.supports_classification and self.text_embeddings is None:
                logger.info("No pre-computed embeddings found, computing on-the-fly...")
                self._compute_text_embeddings()

            self.is_initialized = True
            self._load_time = time.time() - t0

            if self.supports_classification:
                logger.info(
                    f"✅ BioCLIP Model Manager initialized in {self._load_time:.2f}s "
                    f"({len(self.labels)} species)"
                )
            else:
                logger.info(
                    f"✅ BioCLIP Model Manager initialized in {self._load_time:.2f}s "
                    f"(embed-only mode)"
                )

        except Exception as e:
            logger.error(f"Failed to initialize BioCLIP Model Manager: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def _compute_text_embeddings(self) -> None:
        """
        Compute text embeddings for all labels (called if not pre-computed).
        """
        if not self.labels:
            logger.warning("No labels available, skipping embedding computation")
            return

        logger.info(f"Computing text embeddings for {len(self.labels)} species...")

        # Use batch processing if backend supports it
        prompts = [f"a photo of {name}" for name in self.labels]

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
        Encode text into a unit-normalized embedding vector for biological queries.

        Args:
            text: Text to encode (biological context will be added)

        Returns:
            Unit-normalized embedding vector as numpy array

        Raises:
            RuntimeError: If model is not initialized
        """
        self._ensure_initialized()
        prompt = f"a photo of a {text}"
        return self.backend.text_to_vector(prompt)

    @staticmethod
    def extract_name(label_data: str | list[Any]) -> str:
        """
        Extract the preferred name from TreeOfLife label structure.
        Format: [[... 'Genus', 'species'], 'Common Name']
        FallBack:
        - Common Name
        - Scientific Name (Genus species)
        - Full label as string

        Args:
            label_data: The label data from TreeOfLife

        Returns:
            Preferred name string
        """
        if (
                isinstance(label_data, list)
                and len(label_data) == 2
        ):
            taxonomy, common_name = label_data
            if isinstance(common_name, str) and common_name.strip() and common_name != "":
                return common_name
            if isinstance(taxonomy, list) and len(taxonomy) >= 2:
                return f"{taxonomy[-2]} {taxonomy[-1]}"
        return str(label_data)

    def classify_image(
        self, image_bytes: bytes, top_k: int = 3
    ) -> list[tuple[str, float]]:
        """
        Classify an image against TreeOfLife species labels.

        Args:
            image_bytes: Raw image data in bytes
            top_k: Number of top results to return

        Returns:
            List of (scientific_name, probability) tuples sorted by probability (descending)

        Raises:
            RuntimeError: If model is not initialized or classification not supported
        """
        self._ensure_initialized()

        if not self.supports_classification:
            raise RuntimeError(
                "Classification not supported: no dataset loaded. "
                "Ensure TreeOfLife dataset exists in the model directory."
            )

        if self.text_embeddings is None:
            raise RuntimeError("Text embeddings are not available")

        # Get image embedding via backend (already unit-normalized from backend)
        img_embedding = self.encode_image(image_bytes)

        # Normalize text embeddings if not already normalized
        # Check if text embeddings are unit-normalized
        text_norms = np.linalg.norm(self.text_embeddings, axis=1, keepdims=True)
        if not np.allclose(text_norms.flatten(), 1.0, atol=1e-6):
            logger.info("Normalizing text embeddings to unit vectors")
            text_embeddings_norm = self.text_embeddings / text_norms
        else:
            text_embeddings_norm = self.text_embeddings

        # Get temperature scaling from model if available (for better calibration)
        temperature = 1.0
        try:
            # Try to get logit scale from OpenCLIP model (TorchBackend specific)
            if hasattr(self.backend, '_openclip_model') and self.backend._openclip_model is not None:
                raw_temp = self.backend._openclip_model.logit_scale.exp().item()
                # Clamp temperature to reasonable range (CLIP models typically use 1-10)
                temperature = max(0.1, min(10.0, raw_temp))
                if raw_temp != temperature:
                    logger.warning(f"Clamped model temperature from {raw_temp:.4f} to {temperature:.4f}")
                else:
                    logger.debug(f"Using model temperature: {temperature:.4f}")
            elif hasattr(self.backend, '_hf_model') and hasattr(self.backend._hf_model, 'logit_scale'):
                raw_temp = self.backend._hf_model.logit_scale.exp().item()
                temperature = max(0.1, min(10.0, raw_temp))
                if raw_temp != temperature:
                    logger.warning(f"Clamped HF model temperature from {raw_temp:.4f} to {temperature:.4f}")
                else:
                    logger.debug(f"Using HF model temperature: {temperature:.4f}")
        except Exception as e:
            logger.debug(f"Could not get model temperature, using 1.0: {e}")

        # Log embedding dimensions for debugging
        logger.debug(f"Image embedding dim: {img_embedding.shape}")
        logger.debug(f"Text embeddings shape: {text_embeddings_norm.shape}")
        logger.debug(f"Image embedding norm: {np.linalg.norm(img_embedding):.6f}")
        logger.debug(f"Text embeddings avg norm: {np.mean(np.linalg.norm(text_embeddings_norm, axis=1)):.6f}")

        # Auto-detect and fix text embedding axis order for robust compatibility
        # Text embeddings should be (num_classes, embedding_dim)
        if self.text_embeddings.shape[0] == len(self.labels) and self.text_embeddings.shape[1] == img_embedding.shape[0]:
            # Correct orientation: (num_classes, embedding_dim)
            text_embeddings_correct = self.text_embeddings
        elif self.text_embeddings.shape[1] == len(self.labels) and self.text_embeddings.shape[0] == img_embedding.shape[0]:
            # Wrong orientation: (embedding_dim, num_classes) - transpose it
            logger.warning(f"Text embeddings have wrong axis order {self.text_embeddings.shape}, transposing to ({self.text_embeddings.shape[1]}, {self.text_embeddings.shape[0]})")
            text_embeddings_correct = self.text_embeddings.T
        else:
            logger.error(f"Unexpected text embeddings shape: {self.text_embeddings.shape}, expected ({len(self.labels)}, {img_embedding.shape[0]})")
            raise RuntimeError(f"Text embeddings shape incompatible: {self.text_embeddings.shape}")

        # Use cosine similarities directly (no softmax for large datasets)
        # Compute cosine similarities (already unit-normalized)
        similarities = np.dot(img_embedding, text_embeddings_correct.T)

        # Get top-k results using raw cosine similarities
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Extract scientific names from the label data
        results = [
            (self.extract_name(self.labels[idx]), float(similarities[idx]))
            for idx in top_indices
        ]

        # Log similarity statistics for debugging
        logger.debug(f"Similarity stats - max: {np.max(similarities):.6f}, min: {np.min(similarities):.6f}")

        # Log top confidence for debugging
        if results:
            logger.debug(f"Top result: {results[0][0]} with similarity {results[0][1]:.4f}")

        return results

    def info(self) -> ModelInfo:
        """
        Return model manager information including fixed version and performance data.

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
                text_embedding_dim=self.text_embeddings,
                image_embedding_dim=info.image_embedding_dim,
                device=getattr(info, 'device', None),
                precisions=getattr(info, 'precisions', None),
            )

        return ModelInfo(
            model_name=self.resources.model_name,
            model_id=self.model_id,
            model_version=self.model_version,
            supports_classification=self.supports_classification,
            is_initialized=self.is_initialized,
            load_time=self._load_time if self._load_time is not None else 0.0,
            num_labels=len(self.labels) if self.labels else None,
            backend_info=backend_info,
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
