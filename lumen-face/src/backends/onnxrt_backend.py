# onnx_insightface_ab.py

import os
from typing import List, Optional, Sequence
import numpy as np
import cv2
import onnxruntime as ort

from .base import (
    BaseInsightFaceBackend,
    DetectedFace, BBox, Landmarks, BackendInfo
)

class ONNXInsightFaceBackend(BaseInsightFaceBackend):
    def __init__(
        self,
        detector_onnx: str,
        recognizer_onnx: str,
        device_preference: Optional[str] = None,
        max_batch_size: Optional[int] = None,
    ):
        super().__init__(detector_path=detector_onnx, recognizer_path=recognizer_onnx,
                         device_preference=device_preference,
                         max_batch_size=max_batch_size)
        self.det_session: Optional[ort.InferenceSession] = None
        self.rec_session: Optional[ort.InferenceSession] = None
        self.det_input_name = None
        self.det_output_names = None
        self.rec_input_name = None
        self.rec_output_name = None

    def initialize(self) -> None:
        if self._initialized:
            return
        so = ort.SessionOptions()
        # 可开启图优化等
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        if self._device_pref and "cuda" in self._device_pref.lower():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # 加载检测模型
        self.det_session = ort.InferenceSession(self._detector_path, so, providers=providers)
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_output_names = [o.name for o in self.det_session.get_outputs()]

        # 加载识别模型
        self.rec_session = ort.InferenceSession(self._recognizer_path, so, providers=providers)
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.rec_output_name = self.rec_session.get_outputs()[0].name

        self._initialized = True

    def close(self) -> None:
        self.det_session = None
        self.rec_session = None
        self._initialized = False

    def get_info(self) -> BackendInfo:
        return BackendInfo(
            runtime="onnxruntime",
            device=self._device_pref,
            detector_model=os.path.basename(self._detector_path),
            recognizer_model=os.path.basename(self._recognizer_path),
            recognizer_embedding_dim=None,  # 若你能 introspect 输出维度可填
            precisions=["fp32"],
            max_batch_size=self._max_batch_size,
            supports_batch_recognition=True,
            extra={}
        )

    def detect_faces(self, image_bytes: bytes, min_face_size: int = 20, max_results: int = 10) -> List[DetectedFace]:
        # decode
        img_bgr = self.decode_image_bytes(image_bytes)
        h0, w0 = img_bgr.shape[:2]

        # 假设检测模型用输入 640×640（需根据实际模型调整）
        det_size = (640, 640)
        img_resized = cv2.resize(img_bgr, det_size)
        # 转 float，归一化到 [0,1]
        img_in = img_resized.astype(np.float32) / 255.0
        # CHW + batch
        img_in = img_in.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)

        outputs = self.det_session.run(self.det_output_names, {self.det_input_name: img_in})

        # —— decode 部分（根据你实际的 detector 模型格式替换） ——
        # 假设 outputs = [bboxes, landmarks, scores] 或其他结构
        # 下面是一个示例假设，实际要你根据模型改写：

        # 示例：outputs[0] = bboxes (1, N, 5), outputs[1] = landmarks (1, N, 10)
        bboxes = outputs[0][0]  # shape (N,5): x1,y1,x2,y2,score
        landmarks = outputs[1][0]  # shape (N,10): 5 pts x,y

        faces: List[DetectedFace] = []
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2, score = bboxes[i].tolist()
            if score < 0.3:
                continue
            # 将 bbox 映射回原图尺度
            scale_x = w0 / det_size[0]
            scale_y = h0 / det_size[1]
            bx1 = max(int(x1 * scale_x), 0)
            by1 = max(int(y1 * scale_y), 0)
            bx2 = min(int(x2 * scale_x), w0 - 1)
            by2 = min(int(y2 * scale_y), h0 - 1)
            # landmarks
            lm = landmarks[i]
            pts = []
            for j in range(5):
                lx = lm[2*j] * scale_x
                ly = lm[2*j + 1] * scale_y
                pts.append((lx, ly))
            lm_tuple = Landmarks(tuple(pts))
            faces.append(DetectedFace(bbox=BBox(bx1, by1, bx2, by2), score=score, landmarks=lm_tuple))

        faces = sorted(faces, key=lambda f: f.score, reverse=True)
        if max_results is not None:
            faces = faces[:max_results]
        return faces

    def align_and_crop(self, image_bytes: bytes, face: DetectedFace, output_size: int = 112) -> np.ndarray:
        img_bgr = self.decode_image_bytes(image_bytes)
        img_rgb = self.to_rgb(img_bgr)

        # 模型训练时对齐模板（5 点）需要和你训练配置一致
        # 以下是一个常见的 5 点模板（与很多 InsightFace 示例一致）
        # 但你要核对 antelopev2 / buffalo_l 使用的模板
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ], dtype=np.float32)
        dst = src.copy()
        # 如果输出是 112×112，则常见在 x 方向 +8 偏移
        dst[:, 0] += 8.0
        # 若 output_size 不等于 112，需要 scale
        scale = output_size / 112.0
        dst *= scale

        src_pts = np.array(face.landmarks.pts, dtype=np.float32)

        # 估计仿射变换
        tform, _ = cv2.estimateAffinePartial2D(src_pts, dst, method=cv2.LMEDS)
        if tform is None:
            # fallback：直接中心裁剪
            x1, y1, x2, y2 = face.bbox
            crop = img_rgb[y1:y2, x1:x2]
            crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
            return crop

        warped = cv2.warpAffine(img_rgb, tform, (output_size, output_size), flags=cv2.INTER_LINEAR)
        return warped  # RGB uint8

    def face_to_vector(self, face_image: np.ndarray) -> np.ndarray:
        # face_image 是 RGB uint8 或者你也可以先转 float32
        arr = face_image.astype(np.float32)
        # 预处理： (img - 127.5) / 127.5
        arr = (arr - 127.5) / 127.5
        # CHW + batch
        arr = arr.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
        out = self.rec_session.run([self.rec_output_name], {self.rec_input_name: arr})[0]  # (1, D)
        vec = out[0]
        return self.unit_normalize(vec, axis=0)

    def face_batch_to_vectors(self, face_images: Sequence[np.ndarray]) -> np.ndarray:
        if not face_images:
            return np.empty((0, 0), dtype=np.float32)
        arrs = []
        for img in face_images:
            a = img.astype(np.float32)
            a = (a - 127.5) / 127.5
            a = a.transpose(2, 0, 1)
            arrs.append(a)
        batch = np.stack(arrs, axis=0)  # (N,3,H,W)
        out = self.rec_session.run([self.rec_output_name], {self.rec_input_name: batch})[0]  # (N, D)
        return self.unit_normalize(out, axis=1)

    def create_session_with_multi_ep(self, onnx_path: str, preferred_eps: Optional[Sequence[str]] = None):
        # 优先 EP 顺序，让你最期望的 EP 在前面
        default_order = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
        if preferred_eps:
            # 你可以让用户指定优先 EP 顺序
            providers = preferred_eps + [ep for ep in default_order if ep not in preferred_eps]
        else:
            providers = default_order

        # 创建 session
        sess_options = ort.SessionOptions()
        # 可禁用 fallback（如果你希望 strict fail 而不是悄悄落回 CPU）：
        # sess_options.disable_fallback()

        # 用 providers 和（可选的 provider_options）来创建 session
        sess = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        # 查看实际启用的 EP
        enabled = sess.get_providers()
        print("Enabled EPs:", enabled)

        return sess
