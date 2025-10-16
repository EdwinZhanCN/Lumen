"""
bioclip_model.py

Refactored BioCLIPModelManager to align with Lumen architecture contracts.
- Separates concerns between Model Manager and Backend layers
- Uses standard types (np.ndarray) instead of torch-specific types
- Implements proper error handling according to Lumen contracts
- Focuses on business logic: data preparation, caching, and result formatting
"""

from __future__ import annotations
import os
import json
import time
from typing import Any

import numpy as np
from backends import BaseClipBackend
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray


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

    This class follows Lumen architecture contracts:
    - Receives a BaseClipBackend instance for inference
    - Manages model-specific TreeOfLife-10M labels and embeddings cache
    - Handles data preprocessing/postprocessing for biological classification
    - Returns business-level results using standard types
    """

    def __init__(
        self,
        backend: BaseClipBackend,
        model: str = "hf-hub:imageomics/bioclip-2",
        text_repo_id: str = "imageomics/TreeOfLife-10M",
        remote_names_path: str = "embeddings/txt_emb_species.json",
        batch_size: int = 512,
    ) -> None:
        """
        Initialize BioCLIP Model Manager.

        Args:
            backend: Backend instance implementing BaseClipBackend interface
            model: BioCLIP model identifier
            text_repo_id: HuggingFace repository ID for species labels
            remote_names_path: Path to species names file in the repository
            batch_size: Batch size for processing
        """
        # Backend dependency injection (following Lumen contracts)
        self.backend: BaseClipBackend = backend

        # Fixed BioCLIP version and configuration
        self.model_version: str = "bioclip2"

        # Environment overrides for model/config
        self.model_id: str = os.getenv("BIOCLIP_MODEL_NAME", model)
        self.text_repo_id: str = os.getenv("BIOCLIP_TEXT_REPO_ID", text_repo_id)
        self.remote_names_path: str = os.getenv(
            "BIOCLIP_REMOTE_NAMES_PATH", remote_names_path
        )

        bs_env = os.getenv("BIOCLIP_MAX_BATCH_SIZE")
        self.batch_size: int = (
            int(bs_env) if bs_env and bs_env.isdigit() else batch_size
        )

        # Data management paths
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self._base_data_dir: str = os.path.join(base_dir, "data", "bioclip")
        os.makedirs(self._base_data_dir, exist_ok=True)

        # Species labels file (shared across backends)
        self.names_filename: str = os.path.join(
            self._base_data_dir, "txt_emb_species.json"
        )

        # Legacy vectors path (for migration only)
        self._legacy_vectors_filename: str = os.path.join(
            self._base_data_dir, "text_vectors.npz"
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
        print(f"Initializing BioCLIP Model Manager for {self.model_id}...")

        try:
            # 1) Initialize backend (delegate device/runtime concerns to backend)
            self.backend.initialize()

            # 2) Setup per-backend cache paths
            self._setup_cache_paths()

            # 3) Load/download labels and compute/load embeddings (with migration)
            self._load_label_names()
            self._load_or_compute_text_embeddings()

            self.is_initialized = True
            self._load_time = time.time() - t0
            print(f"BioCLIP Model Manager initialized in {self._load_time:.2f} seconds")

        except Exception as e:
            print(f"Failed to initialize BioCLIP Model Manager: {e}")
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
        Download and cache TreeOfLife-10M label names as a JSON list.

        Raises:
            ModelDataNotFoundError: If labels cannot be downloaded or loaded
        """
        # Create parent dir if needed
        dirpath = os.path.dirname(self.names_filename)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

        # Download if missing
        if not os.path.exists(self.names_filename):
            try:
                path = hf_hub_download(
                    repo_id=self.text_repo_id,
                    repo_type="dataset",
                    filename=self.remote_names_path,
                )
                import shutil

                _ = shutil.copy(path, self.names_filename)
            except Exception as e:
                raise ModelDataNotFoundError(
                    f"Failed to download TreeOfLife-10M labels: {e}"
                ) from e

        # Load label names
        try:
            with open(self.names_filename, "r") as f:
                self.labels = json.load(f)
            print(f"Loaded {len(self.labels)} TreeOfLife-10M species labels")
        except Exception as e:
            raise ModelDataNotFoundError(
                f"Failed to load TreeOfLife-10M labels: {e}"
            ) from e

    def _compute_and_cache_text_embeddings(self) -> None:
        """
        Compute text embeddings for all species labels and cache to disk.

        Raises:
            CacheCorruptionError: If caching fails
        """
        assert (
            self._vectors_npz_path is not None and self._vectors_meta_path is not None
        )

        try:
            print(f"Computing text embeddings for {len(self.labels)} species labels...")

            # Compute embeddings via backend (using standard interface)
            prompts = [f"a photo of {name}" for name in self.labels]
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
            print(f"Computed and cached text embeddings to {self._vectors_npz_path}")

        except Exception as e:
            raise CacheCorruptionError(
                f"Failed to compute and cache embeddings: {e}"
            ) from e

    def _load_or_compute_text_embeddings(self) -> None:
        """
        Load cached text embeddings or compute them if cache is invalid/missing.
        Handles migration from legacy cache format.

        Raises:
            LabelMismatchError: If cached embeddings don't match current labels
            CacheCorruptionError: If cached data is corrupted
        """
        assert (
            self._vectors_npz_path is not None and self._vectors_meta_path is not None
        )

        # Migrate legacy cache if exists and new cache is missing
        if os.path.exists(self._legacy_vectors_filename) and not os.path.exists(
            self._vectors_npz_path
        ):
            os.makedirs(os.path.dirname(self._vectors_npz_path), exist_ok=True)
            import shutil

            _ = shutil.move(self._legacy_vectors_filename, self._vectors_npz_path)

            # Create metadata for migrated cache
            backend_info = self.backend.get_info()
            metadata = {
                "runtime": backend_info.runtime,
                "model_id": self.model_id,
                "migrated": True,
            }
            with open(self._vectors_meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, separators=(",", ":"))
            print(f"Migrated legacy cache to {self._vectors_npz_path}")

        # Try to load from cache
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
                    print(
                        f"Loaded cached text embeddings from {self._vectors_npz_path}"
                    )
                    return
                else:
                    print(
                        "Cached labels mismatch current labels; recomputing embeddings"
                    )
                    raise LabelMismatchError(
                        "Cached embeddings don't match current labels"
                    )

            except (Exception,) as e:
                print(f"Cache validation failed: {e}; recomputing embeddings")

        # Compute embeddings if cache is invalid/missing
        self._compute_and_cache_text_embeddings()

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
    def extract_scientific_name(label_data: str | list[Any]) -> str:
        """
        Extract the scientific name from the complex TreeOfLife label structure.

        Args:
            label_data: The label data from TreeOfLife-10M

        Returns:
            The scientific name as a string
        """
        if (
            isinstance(label_data, list)
            and len(label_data) == 2
            and isinstance(label_data[0], list)
        ):
            # Format: [['Animalia', ..., 'Genus', 'species'], 'Common Name']
            taxonomy = label_data[0]
            if len(taxonomy) >= 2:
                # Scientific name is genus + species (last two elements)
                return f"{taxonomy[-2]} {taxonomy[-1]}"
        # Fallback to string representation if we can't extract properly
        return str(label_data)

    def classify_image(
        self, image_bytes: bytes, top_k: int = 3
    ) -> list[tuple[str, float]]:
        """
        Classify an image against TreeOfLife-10M species labels.

        Args:
            image_bytes: Raw image data in bytes
            top_k: Number of top results to return

        Returns:
            List of (scientific_name, probability) tuples sorted by probability (descending)

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
        exp_sims = np.exp(similarities - np.max(similarities))  # Numerical stability
        probabilities = exp_sims / np.sum(exp_sims)

        # Get top-k results
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        # Extract scientific names from the label data
        results = [
            (self.extract_scientific_name(self.labels[idx]), float(probabilities[idx]))
            for idx in top_indices
        ]

        return results

    def info(self) -> dict[str, str | int | float | bool | dict[str, Any]]:
        """
        Return model manager information including fixed version and performance data.

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
            "model_version": self.model_version,
            "model_id": self.model_id,
            "text_repo_id": self.text_repo_id,
            "num_species": len(self.labels),
            "load_time": self._load_time if self._load_time is not None else "",
            "is_initialized": self.is_initialized,
            "backend_info": backend_info,
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
