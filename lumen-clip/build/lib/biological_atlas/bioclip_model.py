from __future__ import annotations
import os
import json
import time
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from backends import BaseClipBackend, TorchBackend, ONNXRTBackend, RKNNBackend
from huggingface_hub import hf_hub_download
from typing import Tuple


class BioCLIPModelManager:
    """
    Simplified BioCLIP-2 manager using OpenCLIP and TreeOfLife-10M labels.
    Provides model loading, text embedding computation/caching, and image classification.
    """

    def __init__(
        self,
        model: str = "hf-hub:imageomics/bioclip-2",
        text_repo_id: str = "imageomics/TreeOfLife-10M",
        remote_names_path: str = "embeddings/txt_emb_species.json",  # Path in the HF repo
        batch_size: int = 512,
        backend: Optional[BaseClipBackend] = None,
    ) -> None:
        # Fixed BioCLIP2 model version
        self.model_version = "bioclip2"
        # Environment overrides for model/config
        self.model_id = os.getenv("BIOCLIP_MODEL_NAME", model)
        self.text_repo_id = os.getenv("BIOCLIP_TEXT_REPO_ID", text_repo_id)
        self.remote_names_path = os.getenv("BIOCLIP_REMOTE_NAMES_PATH", remote_names_path)  # Keep track of the remote path
        bs_env = os.getenv("BIOCLIP_MAX_BATCH_SIZE")
        self.batch_size = int(bs_env) if bs_env and bs_env.isdigit() else batch_size
        # Backend selection (defaults to Torch if none provided)
        if backend is None:
            env_backend = (os.getenv("BIOCLIP_BACKEND", "torch") or "torch").lower()
            device_pref = os.getenv("BIOCLIP_DEVICE")
            if env_backend == "onnxrt":
                onnx_image = os.getenv("BIOCLIP_ONNX_IMAGE")
                onnx_text = os.getenv("BIOCLIP_ONNX_TEXT")
                providers = os.getenv("BIOCLIP_ORT_PROVIDERS")
                providers_list = [p.strip() for p in providers.split(",")] if providers else None
                self.backend = ONNXRTBackend(
                    model_name=self.model_id,
                    pretrained=None,
                    onnx_image_path=onnx_image,
                    onnx_text_path=onnx_text,
                    providers=providers_list,
                    device_preference=device_pref,
                    max_batch_size=self.batch_size,
                )
            elif env_backend == "rknn":
                rknn_path = os.getenv("BIOCLIP_RKNN_MODEL")
                target = os.getenv("BIOCLIP_RKNN_TARGET", "rk3588")
                self.backend = RKNNBackend(
                    model_name=self.model_id,
                    pretrained=None,
                    rknn_model_path=rknn_path,
                    target=target,
                    device_preference=device_pref,
                    max_batch_size=self.batch_size,
                )
            else:
                self.backend = TorchBackend(
                    model_name=self.model_id,
                    pretrained=None,
                    device_preference=device_pref,
                    max_batch_size=self.batch_size,
                )
        else:
            self.backend = backend

        # Use absolute paths based on the location of this script
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self._base_data_dir = os.path.join(base_dir, "data", "bioclip")
        os.makedirs(self._base_data_dir, exist_ok=True)

        # Local filenames for storing data (labels shared; vectors per-backend)
        self.names_filename = os.path.join(self._base_data_dir, "txt_emb_species.json")

        # Legacy vectors path (for migration only)
        self._legacy_vectors_filename = os.path.join(self._base_data_dir, "text_vectors.npz")

        # Per-backend cache paths (resolved after backend.initialize())
        self._cache_runtime_dir: Optional[str] = None
        self._vectors_npz_path: Optional[str] = None
        self._vectors_meta_path: Optional[str] = None

        self.device = self._choose_device()
        self._model: torch.nn.Module | None = None
        self._preprocess = None
        self._tokenizer = None
        self.is_initialized = False

        self.labels: List[str] = []
        self.text_embeddings: torch.Tensor | None = None
        self._load_time: float | None = None

    @staticmethod
    def _choose_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def initialize(self) -> None:
        if self.is_initialized:
            return
        t0 = time.time()
        # Initialize backend and derive per-backend cache paths
        self.backend.initialize()
        binfo = self.backend.get_info()
        runtime_id = binfo.runtime or "torch"
        model_id = binfo.model_id or self.model_id
        self._cache_runtime_dir = os.path.join(self._base_data_dir, runtime_id, model_id)
        os.makedirs(self._cache_runtime_dir, exist_ok=True)
        self._vectors_npz_path = os.path.join(self._cache_runtime_dir, "text_vectors.npz")
        self._vectors_meta_path = os.path.join(self._cache_runtime_dir, "text_vectors.meta.json")
        # Preserve torch model handle for service batch path if using TorchBackend
        if isinstance(self.backend, TorchBackend):
            self._model = getattr(self.backend, "_model", None)
        self.is_initialized = True
        self._load_time = time.time() - t0

        # Load labels and embeddings (with migration)
        self._load_label_names()
        self._load_or_compute_text_embeddings()

    def _load_label_names(self) -> None:
        """
        Download and cache TreeOfLife-10M label names as a JSON list.
        Ensures the local directory for the JSON exists before moving.
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
                shutil.copy(path, self.names_filename)
            except Exception as e:
                raise RuntimeError(f"Failed to download label names: {e}")

        # Load label names
        with open(self.names_filename, 'r') as f:
            self.labels = json.load(f)

    def _compute_and_cache_text_embeddings(self) -> None:
        assert self._vectors_npz_path is not None and self._vectors_meta_path is not None
        prompts = [f"a photo of {name}" for name in self.labels]
        vec_list: List[np.ndarray] = []
        for text in prompts:
            vec = self.backend.text_to_vector(text)
            vec_list.append(vec.astype(np.float32, copy=False))
        vecs = np.vstack(vec_list).astype(np.float32, copy=False)
        np.savez(self._vectors_npz_path, names=np.array(self.labels, dtype=object), vecs=vecs)
        meta = {
            "runtime": self.backend.get_info().runtime,
            "model_id": self.model_id,
        }
        with open(self._vectors_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
        self.text_embeddings = torch.tensor(vecs)

    def _load_or_compute_text_embeddings(self) -> None:
        assert self._vectors_npz_path is not None and self._vectors_meta_path is not None
        # Migrate legacy cache if exists and new cache is missing
        if os.path.exists(self._legacy_vectors_filename) and not os.path.exists(self._vectors_npz_path):
            os.makedirs(os.path.dirname(self._vectors_npz_path), exist_ok=True)
            import shutil
            shutil.move(self._legacy_vectors_filename, self._vectors_npz_path)
            meta = {
                "runtime": self.backend.get_info().runtime,
                "model_id": self.model_id,
            }
            with open(self._vectors_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
        if os.path.exists(self._vectors_npz_path):
            data = np.load(self._vectors_npz_path, allow_pickle=True)
            names = data.get("names")
            vecs = data.get("vecs")
            if names is not None and vecs is not None and names.tolist() == self.labels:
                self.text_embeddings = torch.tensor(vecs.astype(np.float32, copy=False))
                return
        self._compute_and_cache_text_embeddings()



    @staticmethod
    def _unit_normalize(t: torch.Tensor) -> torch.Tensor:
        return t / t.norm(dim=-1, keepdim=True)

    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        self._ensure_initialized()
        return self.backend.image_to_vector(image_bytes).astype(np.float32, copy=False)

    @staticmethod
    def extract_scientific_name(label_data: Any) -> str:
        """
        Extract the scientific name from the complex label structure.

        Args:
            label_data: The label data from the model

        Returns:
            The scientific name as a string
        """
        if isinstance(label_data, list) and len(label_data) == 2 and isinstance(label_data[0], list):
            # Format: [['Animalia', ..., 'Genus', 'species'], 'Common Name']
            taxonomy = label_data[0]
            if len(taxonomy) >= 2:
                # Scientific name is genus + species (last two elements)
                return f"{taxonomy[-2]} {taxonomy[-1]}"
        # Fallback to string representation if we can't extract properly
        return str(label_data)

    def classify_image(
        self,
        image_bytes: bytes,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        self._ensure_initialized()
        img_vec = torch.tensor(self.encode_image(image_bytes), device=self.device).unsqueeze(0)
        assert self.text_embeddings is not None
        text_emb = self.text_embeddings.to(self.device)
        with torch.no_grad():
            sims = (img_vec @ text_emb.T).softmax(dim=-1).squeeze(0)
            probs, idxs = sims.topk(min(top_k, sims.numel()))

        # Extract scientific names from the label data
        return [(self.extract_scientific_name(self.labels[idx]), float(probs[i]))
                for i, idx in enumerate(idxs)]

    def encode_text(self, text: str) -> np.ndarray:
        """Encodes a single string of text into a unit-normalized embedding vector via backend."""
        self._ensure_initialized()
        prompt = f"a photo of a {text}"
        return self.backend.text_to_vector(prompt).astype(np.float32, copy=False)

    def info(self) -> Dict[str, Any]:
        """
        Return model information including fixed version, device, and load time.
        """
        return {
            "model_version": self.model_version,
            "device": str(self.device),
            "load_time": self._load_time,
        }

    def _ensure_initialized(self) -> None:
        if not self.is_initialized:
            raise RuntimeError("Call initialize() before inference.")
