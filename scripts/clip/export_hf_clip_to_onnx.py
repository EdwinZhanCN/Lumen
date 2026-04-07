"""Backward-compatible import shim for CLIP export tooling."""

from scripts.clip.export_clip_to_onnx import (  # noqa: F401
    CLIP_HF_ALLOW_PATTERNS,
    export_clip_to_onnx,
)

export_clip_assets = export_clip_to_onnx
