# insightface_backend.py
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, NamedTuple,
import enum

import numpy as np
import cv2
import io

__all__ = [
    "RuntimeKind",
    "BackendInfo",
    "BaseInsightFaceBackend"
]

class RuntimeKind(str, enum.Enum):
    """Enumerates the primary runtime families for backends."""

    TORCH = "torch"
    ONNXRT = "onnxrt"
    RKNN = "rknn"

# ------------- Types -------------
class BBox(NamedTuple):
    """x1, y1, x2, y2 (pixel coords, int)"""
    x1: int
    y1: int
    x2: int
    y2: int

class Landmarks(NamedTuple):
    """(5,2) typical five-point landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)"""
    pts: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]

@dataclass
class DetectedFace:
    bbox: BBox
    score: float
    landmarks: Optional[Landmarks] = None
    extra: Dict[str, object] = field(default_factory=dict)

@dataclass
class BackendInfo:
    runtime: str
    device: Optional[str] = None
    detector_model: Optional[str] = None
    recognizer_model: Optional[str] = None
    recognizer_embedding_dim: Optional[int] = None
    precisions: List[str] = field(default_factory=list)
    max_batch_size: Optional[int] = None
    supports_batch_recognition: bool = False
    extra: Dict[str, str] = field(default_factory=dict)

# ------------- Base InsightFace Backend -------------
class BaseInsightFaceBackend(abc.ABC):
    """
    Abstract base for InsightFace-style detection + recognition backends.

    Responsibilities:
    - initialize()/close()
    - detect_faces(image_bytes) -> List[DetectedFace]
    - align_and_crop(image_bytes, DetectedFace) -> np.ndarray (RGB image or preprocessed tensor)
    - face_to_vector(cropped_face_bytes_or_array) -> np.ndarray (D,), float32, L2=1
    - batch helper image_batch_to_vectors for lists of face crops or full images (optional override)
    - get_info() -> BackendInfo
    """

    def __init__(
        self,
        detector_path: Optional[str] = None,
        recognizer_path: Optional[str] = None,
        device_preference: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._initialized = False
        self._detector_path = detector_path
        self._recognizer_path = recognizer_path
        self._device_pref = device_preference
        self._max_batch_size = max_batch_size
        self._cache_dir = cache_dir

    # --- lifecycle ---
    @abc.abstractmethod
    def initialize(self) -> None:
        """Load model(s) into the runtime. Idempotent."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources."""
        return

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # --- core capabilities ---

    @abc.abstractmethod
    def detect_faces(self, image_bytes: bytes, min_face_size: int = 20, max_results: int = 10) -> List[DetectedFace]:
        """
        Detect faces in an image.

        Args:
            image_bytes: encoded image bytes (webp/jpg/png)
            min_face_size: heuristic
            max_results: cap number of detections returned

        Returns:
            List[DetectedFace] sorted by score desc (prefer highest-confidence first)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def align_and_crop(self, image_bytes: bytes, face: DetectedFace, output_size: int = 112) -> np.ndarray:
        """
        Given an image and a DetectedFace, return a cropped/aligned face image
        suitable for the recognizer (RGB uint8 or float32 depending on implementation).

        Should follow typical InsightFace preprocessing: crop using landmarks, then resize to 112x112.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def face_to_vector(self, face_image: np.ndarray) -> np.ndarray:
        """
        Convert a preprocessed face image (e.g., 112x112 RGB, float32 normalized) to a unit-normalized vector.

        Input:
            face_image: np.ndarray HWC or CHW depending on backend contract. Document expected dtype and range.

        Returns:
            np.ndarray shape (D,), dtype float32, L2-normalized.
        """
        raise NotImplementedError

    def face_batch_to_vectors(self, face_images: Sequence[np.ndarray]) -> np.ndarray:
        """
        Optional batch implementation. Default fallback: sequential apply face_to_vector.
        Returns shape (N, D).
        """
        if not face_images:
            return np.empty((0, 0), dtype=np.float32)

        vecs = [self.face_to_vector(img) for img in face_images]
        dim = vecs[0].shape[0]
        out = np.stack(vecs, axis=0).astype(np.float32, copy=False)
        if out.ndim != 2 or out.shape[1] != dim:
            raise RuntimeError("face_batch_to_vectors: inconsistent vector dims.")
        return out

    def image_to_face_vectors(self, image_bytes: bytes, top_k: Optional[int] = None) -> np.ndarray:
        """
        Convenience: detect -> align crops -> get embeddings for up to top_k faces.
        Returns (N, D).
        """
        faces = self.detect_faces(image_bytes)
        if top_k:
            faces = faces[:top_k]
        crops = [self.align_and_crop(image_bytes, f, output_size=112) for f in faces]
        return self.face_batch_to_vectors(crops)

    # --- metadata ---
    @abc.abstractmethod
    def get_info(self) -> BackendInfo:
        raise NotImplementedError

    # --- utilities ---
    @staticmethod
    def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
        """
        Decode image bytes (webp/png/jpg) -> HWC BGR uint8 as returned by cv2.
        """
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("decode_image_bytes: failed to decode image")
        return img

    @staticmethod
    def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def preprocess_for_recognizer(rgb_img: np.ndarray, size: int = 112) -> np.ndarray:
        """
        Common InsightFace recognizer preprocessing:
        - resize to (size, size)
        - convert to float32
        - (img - 127.5) / 127.5  -> range [-1,1]
        - change to CHW float32 if backend expects CHW
        Return HWC float32 in default; specific backend can reorder.
        """
        img = cv2.resize(rgb_img, (size, size), interpolation=cv2.INTER_LINEAR)
        arr = img.astype(np.float32)
        arr = (arr - 127.5) / 127.5
        return arr

    @staticmethod
    def unit_normalize(vec: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
        v = vec.astype(np.float32, copy=False)
        norm = np.linalg.norm(v, ord=2, axis=axis, keepdims=True)
        norm = np.maximum(norm, eps)
        return v / norm
