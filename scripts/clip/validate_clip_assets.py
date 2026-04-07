"""Validate CLIP asset repositories generated for Lumen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from scripts.common._shared import ensure_dir, read_json, write_json
from scripts.common.validate_onnx import validate_onnx_model


def _load_and_check_model_info(model_root: Path) -> tuple[dict[str, Any], bool]:
    """Load `model_info.json` and verify the expected CLIP runtime shape."""
    model_info = read_json(model_root / "model_info.json")
    runtimes = model_info.get("runtimes", {})
    onnx_runtime = runtimes.get("onnx", {})
    consistent = (
        model_info.get("model_type") == "clip"
        and onnx_runtime.get("available") is True
        and onnx_runtime.get("components") == ["vision", "text"]
        and onnx_runtime.get("precisions") == ["fp32", "fp16"]
    )
    return model_info, consistent


def validate_clip_assets(*, model_root: Path) -> dict[str, Any]:
    """Validate CLIP ONNX assets, model info, and tokenizer presence."""
    model_root = model_root.resolve()
    model_info, model_info_consistent = _load_and_check_model_info(model_root)
    tokenizer_path = model_root / "upstream" / "tokenizer.json"
    tokenizer_loadable = False
    tokenizer_mode = "runtime-bound"
    if tokenizer_path.exists():
        Tokenizer.from_file(str(tokenizer_path))
        tokenizer_loadable = True
        tokenizer_mode = "tokenizers-json"

    reports: dict[str, Any] = {}
    validation_dir = ensure_dir(model_root / "validation" / "onnx")
    for precision in ("fp32", "fp16"):
        component_reports = {
            "vision": validate_onnx_model(
                model_path=model_root / "onnx" / f"vision.{precision}.onnx",
                task="clip-vision",
            ),
            "text": validate_onnx_model(
                model_path=model_root / "onnx" / f"text.{precision}.onnx",
                task="clip-text",
            ),
        }
        success = all(report.get("success", False) for report in component_reports.values())
        payload = {
            "precision": precision,
            "success": success and model_info_consistent,
            "model_info_consistent": model_info_consistent,
            "tokenizer": {
                "path": str(tokenizer_path),
                "loadable": tokenizer_loadable,
                "mode": tokenizer_mode,
            },
            "components": component_reports,
        }
        write_json(validation_dir / f"{precision}.json", payload)
        reports[precision] = payload

    return {
        "model_info": model_info,
        "reports": reports,
    }
