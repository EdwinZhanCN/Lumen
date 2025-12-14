"""
ONNX Runtime Backend for Lumen-OCR.

This module implements the OCR backend using ONNX Runtime, supporting
PaddleOCR-style detection (DBNet) and recognition (SVTR/CRNN) models.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
from shapely.geometry import Polygon

from ..backends.backend_exceptions import (
    BackendNotInitializedError,
    InvalidInputError,
    ModelLoadingError,
)
from ..backends.base import BackendInfo, BaseOcrBackend, OcrResult
from ..resources.loader import ModelResources

logger = logging.getLogger(__name__)


class OnnxOcrBackend(BaseOcrBackend):
    """
    OCR Backend implementation using ONNX Runtime.
    Supports DBNet for detection and SVTR/CRNN for recognition.
    """

    def __init__(
        self,
        resources: ModelResources,
        providers: list[str] | None = None,
        device_preference: str | None = None,
        prefer_fp16: bool = True,
    ):
        """
        Initialize the ONNX backend.

        Args:
            resources: ModelResources object to access model files and config.
            device_preference: Optional device preference (e.g., "cuda", "cpu").
            prefer_fp16: Whether to prefer FP16 model files over FP32 when available.
        """
        super().__init__()
        self.resources = resources
        self.device_preference = device_preference or "cpu"
        self.providers = providers
        self._prefer_fp16 = prefer_fp16

        # Sessions
        self.det_sess = None
        self.rec_sess = None
        self.cls_sess = None  # Angle classifier (optional, future support)

        # Configs
        self.det_config = {}
        self.rec_config = {}
        self.character_str = []

        # Runtime info
        self._runtime_info = BackendInfo(
            runtime="onnxruntime",
            device=self.device_preference,
            precisions=["fp32"],  # Default assumption
        )

    def initialize(self) -> None:
        """Load models and initialize ONNX sessions."""
        if self.is_initialized:
            return

        try:
            model_info = self.resources.model_info
            extra_meta = getattr(model_info, "extra_metadata", {}) or {}

            # 1. Load Configurations
            self.det_config = extra_meta.get("det_config", {})
            self.rec_config = extra_meta.get("rec_config", {})

            # Set defaults if not present
            self._set_default_configs()

            # 2. Load Vocabulary
            vocab_file = self.rec_config.get("character_dict_path", "ppocr_keys_v1.txt")
            vocab_path = self.resources.get_file_path(vocab_file)
            self.character_str = self._load_vocab(vocab_path)

            # Add space char if configured
            if self.rec_config.get("use_space_char", True):
                self.character_str.append(" ")

            # Add blank char (CTC requirement)
            # For PP-OCRv5 ONNX models, blank is at index 0
            self.character_str.insert(0, "blank")

            # 3. Initialize ONNX Sessions
            providers = self.providers or self._default_providers(
                self.device_preference
            )

            # Detection Model
            det_model_path = self._find_model_file("det")
            self.det_sess = ort.InferenceSession(
                str(det_model_path), providers=providers
            )

            # Recognition Model
            rec_model_path = self._find_model_file("rec")
            self.rec_sess = ort.InferenceSession(
                str(rec_model_path), providers=providers
            )

            # Update info
            self._runtime_info.det_model_id = f"{model_info.name}_det"
            self._runtime_info.rec_model_id = f"{model_info.name}_rec"
            self._runtime_info.model_name = model_info.name
            self._runtime_info.version = model_info.version
            self._runtime_info.extra = extra_meta

            self._initialized = True
            logger.info(f"Initialized OnnxOcrBackend with providers: {providers}")

        except Exception as e:
            logger.error(f"Failed to initialize ONNX backend: {e}")
            raise ModelLoadingError(f"Initialization failed: {e}") from e

    def get_info(self) -> BackendInfo:
        return self._runtime_info

    def predict(
        self,
        image_bytes: bytes,
        det_threshold: float = 0.3,
        rec_threshold: float = 0.5,
        use_angle_cls: bool = False,
        **kwargs: Any,
    ) -> list[OcrResult]:
        """
        Perform end-to-end OCR.
        """
        if not self._initialized:
            raise BackendNotInitializedError("Backend not initialized")

        # 1. Decode Image
        img = self._decode_image(image_bytes)
        if img is None:
            raise InvalidInputError("Failed to decode image bytes")

        # 2. Text Detection
        # Merge kwargs into det_config for this run
        run_det_config = self.det_config.copy()
        run_det_config["thresh"] = det_threshold
        # Allow overriding other params via kwargs
        if "box_thresh" in kwargs:
            run_det_config["box_thresh"] = kwargs["box_thresh"]
        if "unclip_ratio" in kwargs:
            run_det_config["unclip_ratio"] = kwargs["unclip_ratio"]

        dt_boxes = self._detect(img, run_det_config)

        if dt_boxes is None or len(dt_boxes) == 0:
            return []

        # Sort boxes (top-down, left-to-right)
        dt_boxes = self._sorted_boxes(dt_boxes)

        # 3. Text Recognition
        results = []
        for box in dt_boxes:
            # Crop image
            crop_img = self._get_rotate_crop_image(img, box)

            # Recognize
            text, score = self._recognize(crop_img, self.rec_config)

            # Filter by threshold
            if score >= rec_threshold:
                # Convert box to list of tuples
                box_points = [tuple(pt) for pt in box.astype(int).tolist()]
                results.append(
                    OcrResult(box=box_points, text=text, confidence=float(score))
                )

        return results

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _find_model_file(self, keyword: str) -> Path:
        """
        Find model file based on keyword (det/rec) and precision preference.
        Follows naming convention: {keyword}ection.{precision}.onnx
        e.g., detection.fp32.onnx, recognition.fp16.onnx
        """
        # Map short keyword to full filename prefix
        prefix = "detection" if keyword == "det" else "recognition"

        # Determine precision preference based on prefer_fp16 setting
        if self._prefer_fp16:
            precisions = ["fp16", "fp32", "int8"]
        else:
            precisions = ["fp32", "fp16", "int8"]

        for prec in precisions:
            filename = f"{prefix}.{prec}.onnx"
            p = self.resources.runtime_files_path / filename
            if p.exists():
                logger.info(f"Found model file: {filename}")
                return p

        # Fallback to simple name if precision-specific files not found
        fallback = f"{prefix}.onnx"
        p = self.resources.runtime_files_path / fallback
        if p.exists():
            return p

        raise FileNotFoundError(
            f"Could not find model file for '{prefix}' in {self.resources.runtime_files_path}"
        )

    def _set_default_configs(self):
        """Set default values for configs if missing."""
        # Detection Defaults (ImageNet stats)
        if "mean" not in self.det_config:
            self.det_config["mean"] = [0.485, 0.456, 0.406]
        if "std" not in self.det_config:
            self.det_config["std"] = [0.229, 0.224, 0.225]
        if "scale" not in self.det_config:
            self.det_config["scale"] = 1.0 / 255.0
        if "limit_side_len" not in self.det_config:
            self.det_config["limit_side_len"] = 960
        if "thresh" not in self.det_config:
            self.det_config["thresh"] = 0.3
        if "box_thresh" not in self.det_config:
            self.det_config["box_thresh"] = 0.6
        if "unclip_ratio" not in self.det_config:
            self.det_config["unclip_ratio"] = 1.5

        # Recognition Defaults (0.5 stats)
        if "mean" not in self.rec_config:
            self.rec_config["mean"] = [0.5, 0.5, 0.5]
        if "std" not in self.rec_config:
            self.rec_config["std"] = [0.5, 0.5, 0.5]
        if "scale" not in self.rec_config:
            self.rec_config["scale"] = 1.0 / 255.0
        if "image_shape" not in self.rec_config:
            self.rec_config["image_shape"] = [3, 48, 320]

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

    def _load_vocab(self, path: Path) -> list[str]:
        """Load character dictionary from file."""
        try:
            with open(path, "rb") as f:
                lines = f.readlines()
                return [
                    line.decode("utf-8").strip("\n").strip("\r\n") for line in lines
                ]
        except Exception as e:
            raise ModelLoadingError(f"Failed to load vocab file {path}: {e}")

    def _decode_image(self, image_bytes: bytes) -> np.ndarray | None:
        """Decode image bytes to BGR numpy array."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Detection Logic
    # -------------------------------------------------------------------------

    def _detect(self, img: np.ndarray, config: dict) -> np.ndarray:
        """Run DBNet detection."""
        if self.det_sess is None:
            raise BackendNotInitializedError("Detection session not initialized")

        # 1. Preprocess
        preprocessed_img, ratio_h, ratio_w = self._det_preprocess(img, config)

        # 2. Inference
        input_name = self.det_sess.get_inputs()[0].name
        output_name = self.det_sess.get_outputs()[0].name

        preds = self.det_sess.run([output_name], {input_name: preprocessed_img})[0]

        # 3. Postprocess
        boxes = self._det_postprocess(
            np.array(preds), ratio_h, ratio_w, img.shape[:2], config
        )
        return boxes

    def _det_preprocess(
        self, img: np.ndarray, config: dict
    ) -> tuple[np.ndarray, float, float]:
        """Resize and normalize image for detection."""
        h, w, _ = img.shape
        limit_side_len = config["limit_side_len"]

        # Resize logic
        ratio = 1.0
        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        # Ensure multiple of 32
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        img_resize = cv2.resize(img, (resize_w, resize_h))

        # Normalize
        img_resize = img_resize.astype("float32")
        scale = float(config["scale"])
        mean = np.array(config["mean"], dtype="float32").reshape(1, 1, 3)
        std = np.array(config["std"], dtype="float32").reshape(1, 1, 3)

        img_resize = (img_resize * scale - mean) / std
        img_resize = img_resize.astype("float32")

        # HWC -> CHW -> NCHW
        img_resize = img_resize.transpose((2, 0, 1))
        img_resize = np.expand_dims(img_resize, axis=0)

        return img_resize, ratio_h, ratio_w

    def _det_postprocess(
        self,
        preds: np.ndarray,
        ratio_h: float,
        ratio_w: float,
        src_shape: tuple[int, int],
        config: dict,
    ) -> np.ndarray:
        """Convert probability map to boxes."""
        pred = preds[0, 0, :, :]  # (H, W)
        segmentation = pred > config["thresh"]

        boxes_batch = []
        # Find contours
        contours, _ = cv2.findContours(
            (segmentation * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        src_h, src_w = src_shape

        for contour in contours:
            # 1. Get min area rect
            points, sside = self._get_mini_boxes(contour)
            if sside < 3:
                continue

            # 2. Check score
            points = np.array(points)
            score = self._box_score_fast(pred, points.astype(np.int16))
            if score < config["box_thresh"]:
                continue

            # 3. Unclip
            box = self._unclip(points, config["unclip_ratio"])
            if len(box) > 1:
                continue

            # 4. Get result box
            box = np.array(box).reshape(-1, 1, 2)
            points, sside = self._get_mini_boxes(box)
            if sside < 5:
                continue

            # 5. Clip to image size and rescale
            box = np.array(points)
            box[:, 0] = np.clip(np.round(box[:, 0] / ratio_w), 0, src_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / ratio_h), 0, src_h)

            boxes_batch.append(box.astype(np.int16))

        return np.array(boxes_batch)

    def _get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def _box_score_fast(self, bitmap, _box):
        """Calculate mean score of the box region in bitmap."""
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]  # type: ignore

    def _unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()  # type: ignore
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)  # type: ignore
        expanded = offset.Execute(distance)
        return expanded

    def _sorted_boxes(self, dt_boxes):
        """Sort boxes by Y coordinate first, then X."""
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def _get_rotate_crop_image(self, img, points):
        """Crop image based on box points, handling rotation."""
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.array(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)  # type: ignore
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )

        # Handle vertical text (height > width * 1.5)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)

        return dst_img

    # -------------------------------------------------------------------------
    # Recognition Logic
    # -------------------------------------------------------------------------

    def _recognize(self, img: np.ndarray, config: dict) -> tuple[str, float]:
        """Run SVTR/CRNN recognition."""
        if self.rec_sess is None:
            raise BackendNotInitializedError("Recognition session not initialized")

        # 1. Preprocess
        norm_img = self._rec_preprocess(img, config)

        # 2. Inference
        input_name = self.rec_sess.get_inputs()[0].name
        output_name = self.rec_sess.get_outputs()[0].name

        preds = self.rec_sess.run([output_name], {input_name: norm_img})[0]

        # 3. Decode
        text, score = self._rec_decode(np.array(preds))
        return text, score

    def _rec_preprocess(self, img: np.ndarray, config: dict) -> np.ndarray:
        """Resize and normalize for recognition."""
        imgC, imgH, imgW = config["image_shape"]
        max_wh_ratio = imgW / imgH

        h, w = img.shape[:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)

        imgW = int(imgH * max_wh_ratio)

        # Resize height to imgH, width scaled
        ratio = float(imgH) / float(h)
        resize_w = math.ceil(w * ratio)

        # If resize_w > imgW, we might need to cap it, but usually we expand imgW
        # For batch inference, we need padding. For single image, we can just use resize_w.
        # Here we implement simple resize for single image inference.

        resized_image = cv2.resize(img, (resize_w, imgH))

        # Normalize
        resized_image = resized_image.astype("float32")
        scale = float(config["scale"])
        mean = np.array(config["mean"], dtype="float32").reshape(1, 1, 3)
        std = np.array(config["std"], dtype="float32").reshape(1, 1, 3)

        resized_image = (resized_image * scale - mean) / std
        resized_image = resized_image.astype("float32")

        # HWC -> CHW -> NCHW
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = np.expand_dims(resized_image, axis=0)

        # Pad to expected width if needed (optional for ONNX dynamic shape, but good practice)
        # For now, we return as is, assuming model handles dynamic width or we are close enough.

        return resized_image

    def _rec_decode(self, preds: np.ndarray) -> tuple[str, float]:
        """CTC Decode."""
        # preds: (Batch, SeqLen, NumClasses)
        preds_idx = preds.argmax(axis=2)  # (Batch, SeqLen)
        preds_prob = preds.max(axis=2)  # (Batch, SeqLen)

        text_index = preds_idx[0]
        text_prob = preds_prob[0]

        char_list = []
        conf_list = []

        # CTC decoding: drop blanks and duplicates
        # Blank is at index 0
        blank_idx = 0
        ignored_tokens = [blank_idx]

        for idx in range(len(text_index)):
            current_idx = int(text_index[idx])

            if current_idx in ignored_tokens:
                continue

            # Merge duplicates
            if idx > 0 and text_index[idx - 1] == text_index[idx]:
                continue

            if current_idx < 0 or current_idx >= len(self.character_str):
                continue

            char_list.append(self.character_str[current_idx])
            conf_list.append(text_prob[idx])

        text = "".join(char_list)
        score = float(np.mean(conf_list)) if conf_list else 0.0

        return text, score
