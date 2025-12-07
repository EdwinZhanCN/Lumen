"""Static InsightFace pack specifications.

This module exposes hard-coded detection and recognition specs for the
InsightFace model bundles we support (antelopev2, buffalo_l/m/s/sc).
The backend can import `PACK_SPECS` to configure preprocessing and
post-processing without extra metadata files.
"""

from __future__ import annotations

PACK_SPECS = {
    "antelopev2": {
        "detection": {
            "type": "scrfd",
            "input_size": (640, 640),
            "mean": (127.5, 127.5, 127.5),
            "std": (128.0, 128.0, 128.0),
            "letterbox": True,
            "normalized_boxes": False,
            "strides": [8, 16, 32],
            # outputs are grouped by type: [scores...], [bboxes...], [keypoints...]
            # NOT grouped by stride
            "outputs": [
                {"stride": 8, "score": 0, "bbox": 3, "kps": 6},
                {"stride": 16, "score": 1, "bbox": 4, "kps": 7},
                {"stride": 32, "score": 2, "bbox": 5, "kps": 8},
            ],
            "score_threshold": 0.4,
            "nms_threshold": 0.4,
            "min_face": 32,
            "max_face": 1000,
        },
        "recognition": {
            "input_size": (112, 112),
            "mean": (127.5, 127.5, 127.5),
            "std": (127.5, 127.5, 127.5),
            "channels_last": False,
            "color_order": "rgb",
            "align_landmarks": True,
            "embedding_dim": 512,
        },
    },
    "buffalo_l": {
        "detection": {
            "type": "scrfd",
            "input_size": (640, 640),
            "mean": (127.5, 127.5, 127.5),
            "std": (128.0, 128.0, 128.0),
            "letterbox": True,
            "normalized_boxes": False,
            "strides": [8, 16, 32],
            "outputs": [
                {"stride": 8, "score": 0, "bbox": 3, "kps": 6},
                {"stride": 16, "score": 1, "bbox": 4, "kps": 7},
                {"stride": 32, "score": 2, "bbox": 5, "kps": 8},
            ],
            "score_threshold": 0.4,
            "nms_threshold": 0.4,
            "min_face": 32,
            "max_face": 1000,
        },
        "recognition": {
            "input_size": (112, 112),
            "mean": (127.5, 127.5, 127.5),
            "std": (127.5, 127.5, 127.5),
            "channels_last": False,
            "color_order": "rgb",
            "align_landmarks": True,
            "embedding_dim": 512,
        },
    },
    "buffalo_m": {
        "detection": {
            "type": "scrfd",
            "input_size": (640, 640),
            "mean": (127.5, 127.5, 127.5),
            "std": (128.0, 128.0, 128.0),
            "letterbox": True,
            "normalized_boxes": False,
            "strides": [8, 16, 32],
            "outputs": [
                {"stride": 8, "score": 0, "bbox": 3, "kps": 6},
                {"stride": 16, "score": 1, "bbox": 4, "kps": 7},
                {"stride": 32, "score": 2, "bbox": 5, "kps": 8},
            ],
            "score_threshold": 0.4,
            "nms_threshold": 0.4,
            "min_face": 32,
            "max_face": 1000,
        },
        "recognition": {
            "input_size": (112, 112),
            "mean": (127.5, 127.5, 127.5),
            "std": (127.5, 127.5, 127.5),
            "channels_last": False,
            "color_order": "rgb",
            "align_landmarks": True,
            "embedding_dim": 512,
        },
    },
    "buffalo_s": {
        "detection": {
            "type": "scrfd",
            "input_size": (640, 640),
            "mean": (127.5, 127.5, 127.5),
            "std": (128.0, 128.0, 128.0),
            "letterbox": True,
            "normalized_boxes": False,
            "strides": [8, 16, 32],
            "outputs": [
                {"stride": 8, "score": 0, "bbox": 3, "kps": 6},
                {"stride": 16, "score": 1, "bbox": 4, "kps": 7},
                {"stride": 32, "score": 2, "bbox": 5, "kps": 8},
            ],
            "score_threshold": 0.4,
            "nms_threshold": 0.4,
            "min_face": 32,
            "max_face": 1000,
        },
        "recognition": {
            "input_size": (112, 112),
            "mean": (127.5, 127.5, 127.5),
            "std": (127.5, 127.5, 127.5),
            "channels_last": False,
            "color_order": "rgb",
            "align_landmarks": True,
            "embedding_dim": 512,
        },
    },
    "buffalo_sc": {
        "detection": {
            "type": "scrfd",
            "input_size": (640, 640),
            "mean": (127.5, 127.5, 127.5),
            "std": (128.0, 128.0, 128.0),
            "letterbox": True,
            "normalized_boxes": False,
            "strides": [8, 16, 32],
            "outputs": [
                {"stride": 8, "score": 0, "bbox": 3, "kps": 6},
                {"stride": 16, "score": 1, "bbox": 4, "kps": 7},
                {"stride": 32, "score": 2, "bbox": 5, "kps": 8},
            ],
            "score_threshold": 0.4,
            "nms_threshold": 0.4,
            "min_face": 32,
            "max_face": 1000,
        },
        "recognition": {
            "input_size": (112, 112),
            "mean": (127.5, 127.5, 127.5),
            "std": (127.5, 127.5, 127.5),
            "channels_last": False,
            "color_order": "rgb",
            "align_landmarks": True,
            "embedding_dim": 512,
        },
    },
}
