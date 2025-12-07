# L1-625
"""
ONNX Runtime backend for InsightFace model packages.

This module implements a generic ONNX Runtime backend that can serve any
InsightFace package (buffalo_l, buffalo_m, antelopev2, etc.) that follows the
standard naming convention:

- detection.fp32.onnx  (RetinaFace-style detector)
- recognition.fp32.onnx (ArcFace-/PartialFC-style recognizer)

The backend honors the abstract interface defined in `base.py` and exposes
detection plus embedding extraction capabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from ..resources.loader import ModelResources
from .backend_exceptions import (
    BackendError,
    BackendNotInitializedError,
    InferenceError,
    InvalidInputError,
    ModelLoadingError,
)
from .base import BackendInfo, FaceDetection, FaceRecognitionBackend
from .insightface_specs import PACK_SPECS

# onnxruntime is an external dependency; we import lazily to surface a clear error
try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


logger = __import__("logging").getLogger(__name__)


class ONNXRTBackendError(BackendError):
    """Base class for ONNXRTBackend specific errors."""


class ONNXRTModelLoadingError(ONNXRTBackendError, ModelLoadingError):
    """Raised when ONNX model loading fails."""


# --------------------------------------------------------------------------- #
# Data structures describing InsightFace model behaviour
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResizeMeta:
    """Metadata produced when resizing an image for detection."""

    orig_size: tuple[int, int]
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    letterbox: bool


@dataclass(frozen=True)
class DetectionOutputs:
    """Indices of detection outputs produced by the ONNX graph."""

    boxes: int = 0
    scores: int | None = 1
    landmarks: int | None = 2


@dataclass(frozen=True)
class ScrfdHeadSpec:
    """Describes the tensor indices for a SCRFD head at a given stride."""

    stride: int
    score_idx: int
    bbox_idx: int
    kps_idx: int | None = None


@dataclass
class DetectionSpec:
    """Configuration describing how to run and decode the detector."""

    filename: str = "detection.fp32.onnx"
    input_size: tuple[int, int] = (640, 640)
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    letterbox: bool = True
    normalized_boxes: bool = False
    output_indices: DetectionOutputs = field(default_factory=DetectionOutputs)
    detector_type: str = "retinaface"
    strides: tuple[int, ...] = ()
    scrfd_heads: tuple[ScrfdHeadSpec, ...] = ()
    score_threshold: float = 0.5
    nms_threshold: float = 0.4
    min_face_size: int = 32
    max_face_size: int = 1000

    @property
    def uses_scrfd(self) -> bool:
        return self.detector_type.lower() == "scrfd" and bool(self.scrfd_heads)


@dataclass
class RecognitionSpec:
    """Configuration describing how to run the embedding model."""

    filename: str = "recognition.fp32.onnx"
    input_size: tuple[int, int] = (112, 112)
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    model_color_order: str = "bgr"  # color order expected by the model
    input_color_order: str = "rgb"  # color order of input image
    align_landmarks: bool = True  # whether to align face using landmarks


@dataclass
class InsightFaceSpec:
    """Aggregated configuration for an InsightFace package."""

    package_name: str
    detection: DetectionSpec = field(default_factory=DetectionSpec)
    recognition: RecognitionSpec = field(default_factory=RecognitionSpec)
    embedding_dim: int = 512


def _ensure_triplet(values: Sequence[float]) -> tuple[float, float, float]:
    channel_values = list(values)
    if not channel_values:
        return (0.0, 0.0, 0.0)
    while len(channel_values) < 3:
        channel_values.append(channel_values[-1])
    if len(channel_values) > 3:
        channel_values = channel_values[:3]
    return (
        float(channel_values[0]),
        float(channel_values[1]),
        float(channel_values[2]),
    )


def _coerce_hw_tuple(size: Sequence[int]) -> tuple[int, int]:
    values = list(size)
    if not values:
        raise ValueError("input_size must specify at least one dimension")
    if len(values) == 1:
        values = [values[0], values[0]]
    return int(values[0]), int(values[1])


def _parse_scrfd_heads(
    config: Sequence[Mapping[str, Any]] | None,
) -> tuple[ScrfdHeadSpec, ...]:
    if not config:
        return ()
    heads: list[ScrfdHeadSpec] = []
    for head in config:
        if not isinstance(head, Mapping):
            continue
        try:
            stride = int(head["stride"])
            score_idx = int(head["score"])
            bbox_idx = int(head["bbox"])
        except (KeyError, TypeError, ValueError):
            continue
        kps_val = head.get("kps")
        heads.append(
            ScrfdHeadSpec(
                stride=stride,
                score_idx=score_idx,
                bbox_idx=bbox_idx,
                kps_idx=None if kps_val is None else int(kps_val),
            )
        )
    return tuple(heads)


def _merge_detection_spec(
    spec: DetectionSpec, cfg: Mapping[str, Any] | None
) -> DetectionSpec:
    if not cfg:
        return spec

    kwargs: dict[str, Any] = {}
    if "filename" in cfg:
        kwargs["filename"] = str(cfg["filename"])
    if "input_size" in cfg:
        kwargs["input_size"] = _coerce_hw_tuple(cfg["input_size"])
    if "mean" in cfg:
        kwargs["mean"] = _ensure_triplet(cfg["mean"])
    if "std" in cfg:
        kwargs["std"] = _ensure_triplet(cfg["std"])
    if "letterbox" in cfg:
        kwargs["letterbox"] = bool(cfg["letterbox"])
    if "normalized_boxes" in cfg:
        kwargs["normalized_boxes"] = bool(cfg["normalized_boxes"])
    outputs_cfg = cfg.get("outputs")
    if isinstance(outputs_cfg, Mapping):
        kwargs["output_indices"] = DetectionOutputs(
            boxes=outputs_cfg.get("boxes", spec.output_indices.boxes),
            scores=outputs_cfg.get("scores", spec.output_indices.scores),
            landmarks=outputs_cfg.get("landmarks", spec.output_indices.landmarks),
        )
    elif isinstance(outputs_cfg, Sequence) and not isinstance(
        outputs_cfg, (str, bytes)
    ):
        kwargs["scrfd_heads"] = _parse_scrfd_heads(outputs_cfg)
    if "type" in cfg:
        kwargs["detector_type"] = str(cfg["type"])
    if "strides" in cfg:
        kwargs["strides"] = tuple(int(s) for s in cfg["strides"])
    if "score_threshold" in cfg:
        kwargs["score_threshold"] = float(cfg["score_threshold"])
    if "nms_threshold" in cfg:
        kwargs["nms_threshold"] = float(cfg["nms_threshold"])
    if "min_face" in cfg:
        kwargs["min_face_size"] = int(cfg["min_face"])
    if "max_face" in cfg:
        kwargs["max_face_size"] = int(cfg["max_face"])

    return replace(spec, **kwargs)


def _merge_recognition_spec(
    spec: RecognitionSpec, cfg: Mapping[str, Any] | None
) -> RecognitionSpec:
    if not cfg:
        return spec

    kwargs: dict[str, Any] = {}
    if "filename" in cfg:
        kwargs["filename"] = str(cfg["filename"])
    if "input_size" in cfg:
        kwargs["input_size"] = _coerce_hw_tuple(cfg["input_size"])
    if "mean" in cfg:
        kwargs["mean"] = _ensure_triplet(cfg["mean"])
    if "std" in cfg:
        kwargs["std"] = _ensure_triplet(cfg["std"])
    if "channels_last" in cfg:
        # Backward compatibility: channels_last=False means model expects BGR
        kwargs["model_color_order"] = "bgr" if not bool(cfg["channels_last"]) else "rgb"
    if "color_order" in cfg:
        kwargs["input_color_order"] = str(cfg["color_order"])
    if "align_landmarks" in cfg:
        kwargs["align_landmarks"] = bool(cfg["align_landmarks"])

    return replace(spec, **kwargs)


def _apply_pack_overrides(
    package_name: str,
    det_spec: DetectionSpec,
    rec_spec: RecognitionSpec,
    embedding_dim: int,
) -> tuple[DetectionSpec, RecognitionSpec, int]:
    pack_key = (package_name or "").lower()
    pack_cfg = PACK_SPECS.get(pack_key)
    if not pack_cfg:
        return det_spec, rec_spec, embedding_dim

    detection_cfg = pack_cfg.get("detection")
    recognition_cfg = pack_cfg.get("recognition")

    det_spec = _merge_detection_spec(det_spec, detection_cfg)
    rec_spec = _merge_recognition_spec(rec_spec, recognition_cfg)
    if recognition_cfg and "embedding_dim" in recognition_cfg:
        embedding_dim = int(recognition_cfg["embedding_dim"])

    return det_spec, rec_spec, embedding_dim


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _get_nested(mapping: Mapping[str, Any], path: Sequence[str], default: Any) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return current if current is not None else default


def _load_spec(resources: ModelResources) -> InsightFaceSpec:
    """Derive an InsightFaceSpec from model_info metadata (with sane defaults)."""
    info = resources.model_info
    package_name = info.name or resources.model_name

    extra = getattr(info, "extra", None) or {}
    insight_extra = extra.get("insightface", {})

    # Detection overrides
    det_cfg = _get_nested(insight_extra, ("detection",), {})
    outputs_cfg = det_cfg.get("outputs", {})
    scrfd_heads: tuple[ScrfdHeadSpec, ...] = ()
    if isinstance(outputs_cfg, Mapping):
        output_indices = DetectionOutputs(
            boxes=outputs_cfg.get("boxes", 0),
            scores=outputs_cfg.get("scores", 1),
            landmarks=outputs_cfg.get("landmarks", 2),
        )
    else:
        output_indices = DetectionOutputs()
        if isinstance(outputs_cfg, Sequence) and not isinstance(
            outputs_cfg, (str, bytes)
        ):
            scrfd_heads = _parse_scrfd_heads(outputs_cfg)
    det_spec = DetectionSpec(
        filename=det_cfg.get("filename", "detection.fp32.onnx"),
        input_size=_coerce_hw_tuple(det_cfg.get("input_size", (640, 640))),
        mean=_ensure_triplet(det_cfg.get("mean", (0.5, 0.5, 0.5))),
        std=_ensure_triplet(det_cfg.get("std", (0.5, 0.5, 0.5))),
        letterbox=bool(det_cfg.get("letterbox", True)),
        normalized_boxes=bool(det_cfg.get("normalized_boxes", False)),
        output_indices=output_indices,
        detector_type=str(det_cfg.get("type", "retinaface")),
        strides=tuple(int(s) for s in det_cfg.get("strides", ())),
        scrfd_heads=scrfd_heads,
        score_threshold=float(det_cfg.get("score_threshold", 0.5)),
        nms_threshold=float(det_cfg.get("nms_threshold", 0.4)),
        min_face_size=int(det_cfg.get("min_face", 32)),
        max_face_size=int(det_cfg.get("max_face", 1000)),
    )

    # Recognition overrides
    rec_cfg = _get_nested(insight_extra, ("recognition",), {})
    rec_spec = RecognitionSpec(
        filename=rec_cfg.get("filename", "recognition.fp32.onnx"),
        input_size=_coerce_hw_tuple(rec_cfg.get("input_size", (112, 112))),
        mean=_ensure_triplet(rec_cfg.get("mean", (0.5, 0.5, 0.5))),
        std=_ensure_triplet(rec_cfg.get("std", (0.5, 0.5, 0.5))),
        model_color_order="rgb" if bool(rec_cfg.get("channels_last", True)) else "bgr",
        input_color_order=str(rec_cfg.get("color_order", "rgb")),
        align_landmarks=bool(rec_cfg.get("align_landmarks", True)),
    )

    embedding_dim = resources.get_embedding_dim() or rec_cfg.get("embedding_dim", 512)

    det_spec, rec_spec, embedding_dim = _apply_pack_overrides(
        package_name,
        det_spec,
        rec_spec,
        embedding_dim,
    )

    return InsightFaceSpec(
        package_name=package_name,
        detection=det_spec,
        recognition=rec_spec,
        embedding_dim=embedding_dim,
    )


def _convert_image_to_uint8(image: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    """Ensure the image array is contiguous uint8 in RGB order."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(arr.max()) if arr.size else 1.0
            scale = 255.0 if max_val <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Perform NMS on boxes in (x1, y1, x2, y2) format."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def _scrfd_anchor_centers(
    stride: int, height: int, width: int, num_anchors: int = 1
) -> np.ndarray:
    """Generate flattened anchor centers for a SCRFD stride level."""
    grid_y, grid_x = np.mgrid[:height, :width]
    centers = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2).astype(np.float32)
    centers *= float(stride)
    if num_anchors > 1:
        centers = np.repeat(centers, num_anchors, axis=0)
    return centers


def _scrfd_distance2bbox(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Decode SCRFD bbox distances (l, t, r, b) back to XYXY boxes."""
    left = distances[:, 0]
    top = distances[:, 1]
    right = distances[:, 2]
    bottom = distances[:, 3]

    x1 = centers[:, 0] - left
    y1 = centers[:, 1] - top
    x2 = centers[:, 0] + right
    y2 = centers[:, 1] + bottom

    return np.stack((x1, y1, x2, y2), axis=-1)


def _scrfd_distance2kps(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Decode SCRFD landmark offsets (dx, dy) to absolute coordinates."""
    if distances.shape[1] % 2 != 0:
        raise ValueError(f"Invalid SCRFD landmark shape: {distances.shape}")

    num_points = distances.shape[1] // 2
    coords: list[np.ndarray] = []
    for idx in range(num_points):
        dx = distances[:, 2 * idx]
        dy = distances[:, 2 * idx + 1]
        px = centers[:, 0] + dx
        py = centers[:, 1] + dy
        coords.append(px)
        coords.append(py)

    stacked = np.stack(coords, axis=-1)
    return stacked.reshape(-1, num_points, 2)


def _scrfd_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for SCRFD classification heads."""
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=-1, keepdims=True)
    return exp / np.clip(denom, a_min=1e-8, a_max=None)


# --------------------------------------------------------------------------- #
# Backend implementation
# --------------------------------------------------------------------------- #


class ONNXRTBackend(FaceRecognitionBackend):
    """Generic InsightFace backend powered by ONNX Runtime."""

    def __init__(
        self,
        resources: ModelResources,
        providers: list[str] | None = None,
        device_preference: str | None = None,
        max_batch_size: int | None = None,
        prefer_fp16: bool = True,  # kept for signature compatibility
    ) -> None:
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ONNXRTBackend. Install with `pip install onnxruntime`."
            )
        super().__init__()
        self.resources = resources
        self.spec = _load_spec(resources)

        self._providers = providers or self._default_providers(device_preference)
        self._sess_detection: ort.InferenceSession | None = None
        self._sess_recognition: ort.InferenceSession | None = None

        self._detection_input_size = self.spec.detection.input_size
        self._recognition_input_size = self.spec.recognition.input_size
        self._embedding_dim = self.spec.embedding_dim
        self._load_time_seconds: float | None = None
        self._max_batch_size = max_batch_size or 1
        self._prefer_fp16 = prefer_fp16

    # ------------------------------------------------------------------ #
    # Provider utilities
    # ------------------------------------------------------------------ #

    def _default_providers(self, device_pref: str | None) -> list[str]:
        available = set(ort.get_available_providers())
        priority = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "TensorrtExecutionProvider",
            "CPUExecutionProvider",
        ]
        selected = [prov for prov in priority if prov in available]

        pref_map = {
            "cuda": "CUDAExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
            "directml": "DmlExecutionProvider",
            "openvino": "OpenVINOExecutionProvider",
        }
        desired = pref_map.get((device_pref or "").lower())
        if desired and desired in selected:
            selected.insert(0, selected.pop(selected.index(desired)))

        return selected or ["CPUExecutionProvider"]

    @staticmethod
    def _infer_device(providers: list[str]) -> str:
        provs = [p.lower() for p in providers]
        if any("cuda" in p for p in provs):
            return "cuda"
        if any("coreml" in p for p in provs):
            return "coreml"
        if any("dml" in p for p in provs):
            return "directml"
        if any("openvino" in p for p in provs):
            return "openvino"
        return "cpu"

    # ------------------------------------------------------------------ #
    # Initialization & runtime info
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        if self._initialized:
            return

        import time

        start = time.time()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        try:
            det_path = self.resources.get_model_file(self.spec.detection.filename)
            if not det_path.exists():
                raise ONNXRTModelLoadingError(
                    f"Detection model not found: {det_path.name}"
                )
            self._sess_detection = ort.InferenceSession(
                str(det_path), sess_options, providers=self._providers
            )

            rec_path = self.resources.get_model_file(self.spec.recognition.filename)
            if not rec_path.exists():
                raise ONNXRTModelLoadingError(
                    f"Recognition model not found: {rec_path.name}"
                )
            self._sess_recognition = ort.InferenceSession(
                str(rec_path), sess_options, providers=self._providers
            )

            # Update input sizes from actual models (fallback to existing spec)
            self._detection_input_size = self._infer_input_hw(
                self._sess_detection, fallback=self.spec.detection.input_size
            )
            self._recognition_input_size = self._infer_input_hw(
                self._sess_recognition, fallback=self.spec.recognition.input_size
            )

            self._load_time_seconds = time.time() - start
            self._initialized = True
            logger.info(
                "ONNXRTBackend ready in %.2fs (providers=%s)",
                self._load_time_seconds,
                ",".join(self._providers),
            )
        except Exception as exc:  # pragma: no cover
            raise ONNXRTModelLoadingError(
                f"Failed to initialize ONNX sessions: {exc}"
            ) from exc

    @staticmethod
    def _infer_input_hw(
        session: ort.InferenceSession | None,
        fallback: tuple[int, int],
    ) -> tuple[int, int]:
        if not session:
            return fallback
        try:
            shape = session.get_inputs()[0].shape
            if len(shape) >= 4:
                h = int(shape[2])
                w = int(shape[3])
                if h > 0 and w > 0:
                    return (h, w)
        except Exception:  # pragma: no cover
            pass
        return fallback

    def get_runtime_info(self) -> BackendInfo:
        version = getattr(ort, "__version__", None)
        return BackendInfo(
            runtime="onnx",
            device=self._infer_device(self._providers),
            model_id=self.resources.model_name,
            model_name=self.resources.model_info.description,
            pretrained=self.spec.package_name,
            version=version,
            face_embedding_dim=self._embedding_dim,
            precisions=["fp32"],
            max_batch_size=self._max_batch_size,
            supports_image_batch=False,
            extra={
                "providers": ",".join(self._providers),
                "detection_model": self.spec.detection.filename,
                "recognition_model": self.spec.recognition.filename,
                "detection_input": f"{self._detection_input_size[0]}x{self._detection_input_size[1]}",
                "recognition_input": f"{self._recognition_input_size[0]}x{self._recognition_input_size[1]}",
            },
        )

    # ------------------------------------------------------------------ #
    # Detection pipeline
    # ------------------------------------------------------------------ #

    def image_to_faces(
        self,
        image_bytes: bytes,
        detection_confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        face_size_min: int = 50,
        face_size_max: int = 1000,
    ) -> list[FaceDetection]:
        if not self._initialized:
            raise BackendNotInitializedError("Backend not initialized")

        if not image_bytes:
            raise InvalidInputError("image_bytes cannot be empty")

        try:
            decoded = cv2.imdecode(
                np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if decoded is None:
                raise InvalidInputError("Failed to decode image bytes")
            rgb_image = np.ascontiguousarray(
                cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            ).astype(np.uint8, copy=False)

            det_input, resize_meta = self._preprocess_detection(rgb_image)

            assert self._sess_detection is not None
            input_name = self._sess_detection.get_inputs()[0].name
            outputs = self._sess_detection.run(
                None,
                {input_name: det_input},
            )

            return self._postprocess_detection(
                outputs,
                resize_meta,
                detection_confidence_threshold,
                nms_threshold,
                face_size_min,
                face_size_max,
            )
        except InvalidInputError:
            raise
        except Exception as exc:  # pragma: no cover
            raise InferenceError(f"Face detection failed: {exc}") from exc

    def _preprocess_detection(
        self, image: npt.NDArray[np.uint8]
    ) -> tuple[npt.NDArray[np.float32], ResizeMeta]:
        spec = self.spec.detection
        target_h, target_w = self._detection_input_size
        orig_h, orig_w = image.shape[:2]

        if spec.letterbox:
            # Match InsightFace SCRFD preprocessing exactly:
            # - Maintain aspect ratio
            # - Pad with 0 (black), not 128 (gray)
            # - Place image at top-left, not centered
            im_ratio = orig_h / orig_w
            model_ratio = target_h / target_w
            if im_ratio > model_ratio:
                new_h = target_h
                new_w = int(new_h / im_ratio)
            else:
                new_w = target_w
                new_h = int(new_w * im_ratio)

            scale = new_h / orig_h
            resized = cv2.resize(image, (new_w, new_h))

            # Create black canvas and place resized image at top-left
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            canvas[:new_h, :new_w, :] = resized
            working = canvas

            meta = ResizeMeta(
                orig_size=(orig_h, orig_w),
                scale_x=scale,
                scale_y=scale,
                pad_x=0,
                pad_y=0,
                letterbox=True,
            )
        else:
            working = cv2.resize(image, (target_w, target_h))
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            meta = ResizeMeta(
                orig_size=(orig_h, orig_w),
                scale_x=scale_x,
                scale_y=scale_y,
                pad_x=0,
                pad_y=0,
                letterbox=False,
            )

        mean = np.array(spec.mean, dtype=np.float32)
        std = np.array(spec.std, dtype=np.float32)

        # InsightFace SCRFD preprocessing: (pixel - mean) / std
        # Mean and std are applied to [0-255] range, not [0-1]
        normalized = working.astype(np.float32)
        normalized = (normalized - mean) / np.maximum(std, 1e-6)

        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        return tensor.astype(np.float32), meta

    def _decode_detection_outputs(
        self,
        outputs: list[npt.NDArray],
        spec: DetectionSpec,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if spec.uses_scrfd:
            return self._decode_scrfd(outputs, spec)

        idx = spec.output_indices

        def _safe_get(slot: int | None, default_idx: int) -> npt.NDArray:
            if slot is None:
                return outputs[default_idx]
            return outputs[slot]

        boxes = np.asarray(_safe_get(idx.boxes, 0)).squeeze()
        scores = (
            np.asarray(outputs[idx.scores]).squeeze()
            if idx.scores is not None
            else None
        )
        landmarks = (
            np.asarray(outputs[idx.landmarks]).squeeze()
            if idx.landmarks is not None and len(outputs) > idx.landmarks
            else None
        )

        if boxes.ndim == 1:
            if boxes.size % 4 == 0:
                boxes = boxes.reshape(-1, 4)
            else:
                boxes = boxes.reshape(1, -1)
        if boxes.size == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                None,
            )

        if scores is None and boxes.shape[1] >= 5:
            scores = boxes[:, 4]
            boxes = boxes[:, :4]
        if scores is None:
            scores = np.ones(len(boxes), dtype=np.float32)
        else:
            scores = np.asarray(scores).reshape(-1)

        row_count = min(len(boxes), len(scores))
        boxes = np.asarray(boxes[:row_count], dtype=np.float32)
        scores = np.asarray(scores[:row_count], dtype=np.float32)

        if landmarks is not None:
            landmarks = np.asarray(landmarks)
            if (
                landmarks.ndim == 1
                and row_count > 0
                and landmarks.size % row_count == 0
            ):
                landmarks = landmarks.reshape(row_count, -1)
            elif landmarks.ndim == 1 and landmarks.size % 10 == 0:
                landmarks = landmarks.reshape(-1, 10)
            landmarks = landmarks[:row_count]
            if len(landmarks) != row_count:
                logger.warning(
                    "Discarding landmarks due to row mismatch (landmarks=%s rows=%d)",
                    getattr(landmarks, "shape", None),
                    row_count,
                )
                landmarks = None

        return boxes, scores, landmarks

    def _decode_scrfd(
        self,
        outputs: list[npt.NDArray],
        spec: DetectionSpec,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Decode SCRFD outputs.

        SCRFD models output tensors in (N, C) format where:
        - N = spatial_h * spatial_w * num_anchors
        - C = channels (1 or 2 for scores, 4 for bbox, 10 for 5-point landmarks)

        The outputs are grouped by type across all strides:
        [score_s8, score_s16, score_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32]
        """
        box_chunks: list[np.ndarray] = []
        score_chunks: list[np.ndarray] = []
        landmark_chunks: list[np.ndarray] = []

        # Calculate input dimensions for spatial size calculation
        input_h, input_w = spec.input_size

        for head in spec.scrfd_heads:
            # Get score tensor
            score_raw = outputs[head.score_idx]
            logger.debug(
                "SCRFD head (stride=%d) raw score output shape=%s",
                head.stride,
                getattr(score_raw, "shape", None),
            )
            score_array = np.asarray(score_raw, dtype=np.float32)
            # Remove batch dimension if present
            score_array = np.squeeze(score_array)

            logger.debug(
                "SCRFD head (stride=%d) after squeeze score shape=%s",
                head.stride,
                getattr(score_array, "shape", None),
            )

            if score_array.size == 0:
                logger.debug(
                    "SCRFD head (stride=%d) has empty score tensor, skipping",
                    head.stride,
                )
                continue

            # Get bbox tensor
            bbox_raw = outputs[head.bbox_idx]
            logger.debug(
                "SCRFD head (stride=%d) raw bbox output shape=%s",
                head.stride,
                getattr(bbox_raw, "shape", None),
            )
            bbox_array = np.asarray(bbox_raw, dtype=np.float32)
            bbox_array = np.squeeze(bbox_array)

            logger.debug(
                "SCRFD head (stride=%d) after squeeze bbox shape=%s",
                head.stride,
                getattr(bbox_array, "shape", None),
            )

            # Handle tensor shapes - SCRFD outputs are (N, C) format
            # where N = spatial_h * spatial_w * num_anchors

            # Ensure score is 2D (N, C)
            if score_array.ndim == 1:
                # Reshape to (N, 1) - single class score
                score_array = score_array.reshape(-1, 1)
            elif score_array.ndim != 2:
                logger.debug(
                    "Skipping SCRFD head (stride=%d) - unexpected score shape: %s",
                    head.stride,
                    score_array.shape,
                )
                continue

            # Ensure bbox is 2D (N, C)
            if bbox_array.ndim == 1:
                # Try to reshape as (N, 4)
                if bbox_array.size % 4 != 0:
                    logger.debug(
                        "Skipping SCRFD head (stride=%d) - bbox size %d not divisible by 4",
                        head.stride,
                        bbox_array.size,
                    )
                    continue
                bbox_array = bbox_array.reshape(-1, 4)
            elif bbox_array.ndim != 2:
                logger.debug(
                    "Skipping SCRFD head (stride=%d) - unexpected bbox shape: %s",
                    head.stride,
                    bbox_array.shape,
                )
                continue

            # Validate bbox has 4 channels
            if bbox_array.shape[1] != 4:
                logger.debug(
                    "Skipping SCRFD head (stride=%d) - bbox should have 4 channels, got %d",
                    head.stride,
                    bbox_array.shape[1],
                )
                continue

            # Calculate spatial dimensions
            spatial_h = input_h // head.stride
            spatial_w = input_w // head.stride

            # Infer number of anchors
            num_positions = bbox_array.shape[0]
            expected_positions_per_anchor = spatial_h * spatial_w

            if num_positions % expected_positions_per_anchor != 0:
                logger.debug(
                    "Skipping SCRFD head (stride=%d) - num_positions %d not divisible by spatial size %d",
                    head.stride,
                    num_positions,
                    expected_positions_per_anchor,
                )
                continue

            num_anchors = num_positions // expected_positions_per_anchor

            logger.debug(
                "SCRFD head (stride=%d) spatial=(%d,%d) anchors=%d positions=%d",
                head.stride,
                spatial_h,
                spatial_w,
                num_anchors,
                num_positions,
            )

            # Verify score tensor matches
            if score_array.shape[0] != num_positions:
                logger.debug(
                    "Skipping SCRFD head (stride=%d) - score positions %d != bbox positions %d",
                    head.stride,
                    score_array.shape[0],
                    num_positions,
                )
                continue

            # Process scores
            score_channels = score_array.shape[1]

            # Debug: show raw score values
            logger.debug(
                "SCRFD head (stride=%d) score values: min=%.6f max=%.6f mean=%.6f channels=%d",
                head.stride,
                float(np.min(score_array)),
                float(np.max(score_array)),
                float(np.mean(score_array)),
                score_channels,
            )

            # SCRFD ONNX models output already-activated scores (sigmoid/softmax applied)
            # Just extract the scores directly without additional activation
            if score_channels == 1:
                # Single class score (already sigmoid-activated)
                cls_scores = score_array[:, 0]
            elif score_channels == 2:
                # Binary classification (already softmax-activated), take positive class
                cls_scores = score_array[:, 1]
            else:
                logger.debug(
                    "Skipping SCRFD head (stride=%d) - unexpected score channels: %d",
                    head.stride,
                    score_channels,
                )
                continue

            logger.debug(
                "SCRFD head (stride=%d) extracted %d scores, range=[%.6f, %.6f]",
                head.stride,
                len(cls_scores),
                float(np.min(cls_scores)),
                float(np.max(cls_scores)),
            )

            # Scale bbox distances by stride
            bbox_matrix = bbox_array * float(head.stride)

            # Generate anchor centers
            centers = _scrfd_anchor_centers(
                head.stride, spatial_h, spatial_w, num_anchors=num_anchors
            )

            # Decode bboxes from distance predictions
            decoded_boxes = _scrfd_distance2bbox(centers, bbox_matrix)

            box_chunks.append(decoded_boxes.astype(np.float32))
            score_chunks.append(cls_scores.astype(np.float32))

            # Process keypoints if available
            if head.kps_idx is None:
                continue

            kps_raw = outputs[head.kps_idx]
            logger.debug(
                "SCRFD head (stride=%d) raw kps output shape=%s",
                head.stride,
                getattr(kps_raw, "shape", None),
            )
            kps_array = np.asarray(kps_raw, dtype=np.float32)
            kps_array = np.squeeze(kps_array)

            logger.debug(
                "SCRFD head (stride=%d) after squeeze kps shape=%s",
                head.stride,
                getattr(kps_array, "shape", None),
            )

            # Ensure kps is 2D (N, C)
            if kps_array.ndim == 1:
                # Try to reshape - typically 10 channels for 5 landmarks
                if kps_array.size % 10 != 0 and kps_array.size % 2 != 0:
                    logger.debug(
                        "Skipping SCRFD head (stride=%d) - unexpected kps size: %d",
                        head.stride,
                        kps_array.size,
                    )
                    continue
                # Determine channels per position
                if kps_array.size % num_positions == 0:
                    kps_channels = kps_array.size // num_positions
                    kps_array = kps_array.reshape(num_positions, kps_channels)
                else:
                    logger.debug(
                        "Skipping SCRFD head (stride=%d) - kps size %d not compatible with positions %d",
                        head.stride,
                        kps_array.size,
                        num_positions,
                    )
                    continue
            elif kps_array.ndim != 2:
                logger.debug(
                    "Skipping SCRFD head (stride=%d) - unexpected kps shape: %s",
                    head.stride,
                    kps_array.shape,
                )
                continue

            # Verify kps positions match
            if kps_array.shape[0] != num_positions:
                logger.debug(
                    "Skipping SCRFD head (stride=%d) - kps positions %d != bbox positions %d",
                    head.stride,
                    kps_array.shape[0],
                    num_positions,
                )
                continue

            # Scale keypoint distances by stride
            kps_matrix = kps_array * float(head.stride)

            # Decode keypoints from distance predictions
            decoded_kps = _scrfd_distance2kps(centers, kps_matrix)
            landmark_chunks.append(decoded_kps.astype(np.float32))

        if not box_chunks:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                None,
            )

        boxes = np.concatenate(box_chunks, axis=0)
        scores = np.concatenate(score_chunks, axis=0)
        landmarks = np.concatenate(landmark_chunks, axis=0) if landmark_chunks else None

        return boxes, scores, landmarks

    def _postprocess_detection(
        self,
        outputs: list[npt.NDArray],
        resize_meta: ResizeMeta,
        detection_confidence_threshold: float,
        nms_threshold: float,
        face_size_min: int,
        face_size_max: int,
    ) -> list[FaceDetection]:
        spec = self.spec.detection

        boxes, scores, landmarks = self._decode_detection_outputs(outputs, spec)
        if boxes.size == 0 or scores.size == 0:
            logger.debug("Detection decode produced no candidates")
            return []

        row_count = min(len(boxes), len(scores))
        boxes = boxes[:row_count]
        scores = scores[:row_count]
        if row_count == 0:
            return []

        if landmarks is not None:
            landmarks = np.asarray(landmarks)[:row_count]
            if landmarks.shape[0] != row_count:
                logger.warning(
                    "Discarding landmarks due to row mismatch (landmarks=%s rows=%d)",
                    getattr(landmarks, "shape", None),
                    row_count,
                )
                landmarks = None

        logger.debug(
            "Detection decoded to %d rows (boxes=%s scores=%s landmarks=%s)",
            row_count,
            boxes.shape,
            scores.shape,
            None if landmarks is None else getattr(landmarks, "shape", None),
        )

        # Log score distribution for debugging
        if len(scores) > 0:
            logger.debug(
                "Score distribution: min=%.6f max=%.6f mean=%.6f median=%.6f",
                float(np.min(scores)),
                float(np.max(scores)),
                float(np.mean(scores)),
                float(np.median(scores)),
            )
            # Show top 10 scores
            top_scores = np.sort(scores)[-10:][::-1]
            logger.debug("Top 10 scores: %s", [f"{s:.6f}" for s in top_scores])

        mask = scores >= detection_confidence_threshold
        logger.debug(
            "Detection confidence mask kept %d/%d rows (threshold=%.3f)",
            int(mask.sum()),
            len(mask),
            detection_confidence_threshold,
        )
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        scores = scores[mask]
        landmarks = landmarks[mask] if landmarks is not None else None

        # Convert coordinates back to original image space
        orig_h, orig_w = resize_meta.orig_size
        if spec.normalized_boxes:
            boxes[:, [0, 2]] *= orig_w
            boxes[:, [1, 3]] *= orig_h
        elif resize_meta.letterbox:
            boxes[:, [0, 2]] = (
                boxes[:, [0, 2]] - resize_meta.pad_x
            ) / resize_meta.scale_x
            boxes[:, [1, 3]] = (
                boxes[:, [1, 3]] - resize_meta.pad_y
            ) / resize_meta.scale_y
        else:
            boxes[:, [0, 2]] /= resize_meta.scale_x
            boxes[:, [1, 3]] /= resize_meta.scale_y

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        size_mask = (
            (widths >= face_size_min)
            & (heights >= face_size_min)
            & (widths <= face_size_max)
            & (heights <= face_size_max)
        )
        if not np.any(size_mask):
            logger.debug(
                "Detection size filter removed all boxes (min=%d max=%d)",
                face_size_min,
                face_size_max,
            )
            return []

        boxes = boxes[size_mask]
        scores = scores[size_mask]
        landmarks = landmarks[size_mask] if landmarks is not None else None

        selected_indices = _nms(boxes, scores, nms_threshold)
        if selected_indices.size == 0:
            return []

        boxes = boxes[selected_indices]
        scores = scores[selected_indices]
        if landmarks is not None:
            landmarks = landmarks[selected_indices]

        faces: list[FaceDetection] = []
        for idx_val, bbox in enumerate(boxes):
            lm_list: list[tuple[float, float]] | None = None
            if landmarks is not None:
                lm = landmarks[idx_val].reshape(-1, 2)
                lm_list = [(float(x), float(y)) for x, y in lm]

            faces.append(
                FaceDetection(
                    bbox=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    landmarks=lm_list,
                    confidence=float(scores[idx_val]),
                )
            )

        return faces

    # ------------------------------------------------------------------ #
    # Embedding pipeline
    # ------------------------------------------------------------------ #

    def face_to_embedding(
        self,
        face_image: bytes | None = None,
        cropped_face_array: npt.NDArray[np.float32] | None = None,
        landmarks: list[tuple[float, float]] | None = None,
    ) -> npt.NDArray[np.float32]:
        if not self._initialized:
            raise BackendNotInitializedError("Backend not initialized")
        if face_image is None and cropped_face_array is None:
            raise InvalidInputError(
                "Either face_image or cropped_face_array must be provided"
            )

        try:
            if face_image is not None:
                array = cv2.imdecode(
                    np.frombuffer(face_image, np.uint8), cv2.IMREAD_COLOR
                )
                if array is None:
                    raise InvalidInputError("Failed to decode face_image bytes")
                rgb = np.ascontiguousarray(
                    cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                ).astype(np.uint8, copy=False)
            else:
                if cropped_face_array is None:
                    raise InvalidInputError(
                        "cropped_face_array cannot be None when face_image is not provided"
                    )
                rgb = _convert_image_to_uint8(cropped_face_array)

            # Apply face alignment only if align_landmarks is True and landmarks are provided
            if (
                self.spec.recognition.align_landmarks
                and landmarks
                and len(landmarks) == 5
            ):
                rgb = self._align_face_5points(rgb, landmarks)

            rec_input = self._preprocess_recognition(rgb)

            assert self._sess_recognition is not None
            input_name = self._sess_recognition.get_inputs()[0].name
            output = self._sess_recognition.run(None, {input_name: rec_input})[0]
            embedding = np.asarray(output, dtype=np.float32).reshape(-1)

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm

            return embedding
        except InvalidInputError:
            raise
        except Exception as exc:  # pragma: no cover
            raise InferenceError(f"Face embedding extraction failed: {exc}") from exc

    def _preprocess_recognition(
        self, image: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:
        target_h, target_w = self._recognition_input_size
        resized = cv2.resize(image, (target_w, target_h))

        # Convert input color order to model's expected color order
        spec = self.spec.recognition
        if spec.input_color_order != spec.model_color_order:
            if spec.input_color_order == "rgb" and spec.model_color_order == "bgr":
                resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            elif spec.input_color_order == "bgr" and spec.model_color_order == "rgb":
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                logger.warning(
                    "Unsupported color order conversion: %s -> %s",
                    spec.input_color_order,
                    spec.model_color_order,
                )

        mean = np.array(spec.mean, dtype=np.float32)
        std = np.array(spec.std, dtype=np.float32)
        if float(np.max(mean)) > 1.0 or float(np.max(std)) > 1.0:
            mean /= 255.0
            std /= 255.0
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - mean) / np.maximum(std, 1e-6)

        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        return tensor.astype(np.float32)

    def _align_face_5points(
        self, image: npt.NDArray[np.uint8], landmarks: list[tuple[float, float]]
    ) -> npt.NDArray[np.uint8]:
        if len(landmarks) != 5:
            return image

        dst = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        if self._recognition_input_size != (112, 112):
            scale_x = self._recognition_input_size[1] / 112.0
            scale_y = self._recognition_input_size[0] / 112.0
            dst[:, 0] *= scale_x
            dst[:, 1] *= scale_y

        src = np.array(landmarks, dtype=np.float32)
        transform, _ = cv2.estimateAffinePartial2D(src, dst)
        if transform is None:
            return image

        aligned = cv2.warpAffine(
            image,
            transform,
            (self._recognition_input_size[1], self._recognition_input_size[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        aligned_uint8: np.ndarray = _convert_image_to_uint8(aligned)
        return aligned_uint8
