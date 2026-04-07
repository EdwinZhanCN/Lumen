"""Export CLIP assets into fixed-shape ONNX and generate repository metadata."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import onnx
import torch
from onnxruntime.transformers.float16 import convert_float_to_float16
from torch import nn
from transformers import CLIPModel

from scripts.clip.validate_clip_assets import validate_clip_assets
from scripts.common._shared import DEFAULT_ONNX_OPSET, parse_shape, read_json
from scripts.common.generate_model_card import generate_model_card
from scripts.common.hf_download import download_hf_artifacts
from scripts.common.write_model_info import write_model_metadata

try:
    import open_clip as _open_clip  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    open_clip: Any = SimpleNamespace()
else:  # pragma: no cover - exercised when optional dependency is installed
    open_clip = _open_clip

try:
    from timm.utils import reparameterize_model as _reparameterize_model
except ImportError:  # pragma: no cover - optional dependency
    reparameterize_model: Any = None
else:  # pragma: no cover - exercised when optional dependency is installed
    reparameterize_model = _reparameterize_model


CLIP_HF_ALLOW_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "*.bin",
    "*.safetensors",
)


@dataclass(frozen=True, slots=True)
class _SourceSpec:
    """Resolved CLIP upstream source information."""

    format: str
    repo_id: str
    revision: str | None = None
    pretrained: str | None = None
    reparameterize: bool = False


class _HFVisionWrapper(nn.Module):
    """Expose `get_image_features()` for Hugging Face CLIP export."""

    def __init__(self, model: CLIPModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return projected image features."""
        return self.model.get_image_features(pixel_values=pixel_values)


class _HFTextWrapper(nn.Module):
    """Expose `get_text_features()` for Hugging Face CLIP export."""

    def __init__(self, model: CLIPModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return projected text features."""
        return self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


class _OpenCLIPVisionWrapper(nn.Module):
    """Expose `encode_image()` for OpenCLIP export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model: Any = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return projected image features."""
        return self.model.encode_image(pixel_values)


class _OpenCLIPTextWrapper(nn.Module):
    """Expose `encode_text()` for OpenCLIP export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model: Any = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return projected text features."""
        return self.model.encode_text(input_ids)


def _read_source_spec(model_root: Path) -> _SourceSpec:
    """Resolve CLIP source metadata from `upstream/source.json`."""
    source_metadata = read_json(model_root / "upstream" / "source.json")
    export_options = source_metadata.get("export_options", {})
    return _SourceSpec(
        format=str(source_metadata["format"]),
        repo_id=str(source_metadata["repo_id"]),
        revision=source_metadata.get("revision"),
        pretrained=source_metadata.get("pretrained"),
        reparameterize=bool(export_options.get("reparameterize", False)),
    )


def _maybe_download_hf_upstream(model_root: Path, source_spec: _SourceSpec) -> None:
    """Download missing CLIP Hugging Face artifacts into `upstream/`."""
    if source_spec.format != "huggingface":
        return

    upstream_dir = model_root / "upstream"
    has_weight = any(upstream_dir.glob("*.safetensors")) or any(upstream_dir.glob("*.bin"))
    required_metadata = (
        upstream_dir / "config.json",
        upstream_dir / "tokenizer.json",
        upstream_dir / "tokenizer_config.json",
        upstream_dir / "preprocessor_config.json",
        upstream_dir / "special_tokens_map.json",
    )
    if has_weight and all(path.exists() for path in required_metadata):
        return

    download_hf_artifacts(
        repo_id=source_spec.repo_id,
        revision=source_spec.revision or "main",
        output_dir=model_root,
        allow_patterns=list(CLIP_HF_ALLOW_PATTERNS),
    )


def _resolve_hf_source(model_root: Path) -> tuple[str | Path, str | None, bool]:
    """Return the Hugging Face CLIP source and whether it should be loaded locally."""
    upstream_dir = model_root / "upstream"
    if any(upstream_dir.glob(pattern) for pattern in ("*.safetensors", "*.bin")):
        return upstream_dir, None, True

    source_metadata = _read_source_spec(model_root)
    return (
        str(source_metadata.repo_id),
        source_metadata.revision or "main",
        False,
    )


def _load_hf_clip_model(model_root: Path) -> CLIPModel:
    """Load a Hugging Face CLIPModel from local artifacts or the upstream repo."""
    source, revision, local_files_only = _resolve_hf_source(model_root)
    model = CLIPModel.from_pretrained(
        source,
        revision=revision,
        local_files_only=local_files_only,
    )
    model.eval()
    return model


def _load_openclip_model(model_root: Path) -> nn.Module:
    """Load an OpenCLIP model using its factory APIs."""
    spec = _read_source_spec(model_root)
    if not hasattr(open_clip, "create_model_and_transforms"):
        raise RuntimeError(
            "OpenCLIP export requires the optional 'open_clip_torch' dependency."
        )
    model, _, _ = open_clip.create_model_and_transforms(
        spec.repo_id,
        pretrained=spec.pretrained,
    )
    model.eval()
    if spec.reparameterize:
        if reparameterize_model is None:
            raise RuntimeError(
                "OpenCLIP reparameterization requires the optional 'timm' dependency."
            )
        model = reparameterize_model(model)
        model.eval()
    return model


def _load_clip_model(model_root: Path) -> tuple[str, nn.Module]:
    """Load a CLIP-family model and return `(source_format, model)`."""
    source_spec = _read_source_spec(model_root)
    if source_spec.format == "huggingface":
        return source_spec.format, _load_hf_clip_model(model_root)
    if source_spec.format == "openclip":
        return source_spec.format, _load_openclip_model(model_root)
    raise ValueError(
        f"Unsupported CLIP source format '{source_spec.format}'. "
        "Expected 'huggingface' or 'openclip'."
    )


def _export_vision_model(
    *,
    source_format: str,
    model: nn.Module,
    output_path: Path,
    opset: int,
    image_shape: tuple[int, ...],
) -> None:
    """Export the CLIP vision branch to ONNX."""
    if source_format == "huggingface":
        wrapper = _HFVisionWrapper(cast(CLIPModel, model)).eval()
    else:
        wrapper = _OpenCLIPVisionWrapper(model).eval()
    dummy = torch.zeros(image_shape, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["image_features"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def _export_text_model(
    *,
    source_format: str,
    model: nn.Module,
    output_path: Path,
    opset: int,
    text_shape: tuple[int, ...],
) -> None:
    """Export the CLIP text branch to ONNX."""
    if source_format == "huggingface":
        wrapper = _HFTextWrapper(cast(CLIPModel, model)).eval()
        input_ids = torch.zeros(text_shape, dtype=torch.long)
        attention_mask = torch.ones(text_shape, dtype=torch.long)
        args: tuple[torch.Tensor, ...] = (input_ids, attention_mask)
        input_names = ["input_ids", "attention_mask"]
    else:
        wrapper = _OpenCLIPTextWrapper(model).eval()
        input_ids = torch.zeros(text_shape, dtype=torch.long)
        args = (input_ids,)
        input_names = ["input_ids"]

    torch.onnx.export(
        wrapper,
        args,
        str(output_path),
        input_names=input_names,
        output_names=["text_features"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def _convert_to_fp16(src_path: Path, dst_path: Path) -> int:
    """Convert an FP32 ONNX file into FP16."""
    model = onnx.load(str(src_path))
    converted = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, str(dst_path))
    return int(converted.ir_version)


def _default_model_name(source_spec: _SourceSpec) -> str:
    """Derive a default model name from the upstream identifier."""
    return source_spec.repo_id.rsplit("/", maxsplit=1)[-1].lower()


def _default_description(model_name: str) -> str:
    """Build a default model description."""
    return f"CLIP deployment assets for {model_name}."


def _build_clip_runtimes() -> dict[str, Any]:
    """Return the CLIP runtime inventory."""
    return {
        "onnx": {
            "available": True,
            "components": ["vision", "text"],
            "precisions": ["fp32", "fp16"],
        }
    }


def _build_clip_task_metadata(
    *,
    model_root: Path,
    source_format: str,
    model: nn.Module,
    image_shape: tuple[int, ...],
    text_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Build task-specific weak metadata for CLIP assets."""
    tokenizer_path = model_root / "upstream" / "tokenizer.json"
    task_metadata: dict[str, Any] = {
        "input_contract": {
            "vision": {"fixed_shape": list(image_shape)},
            "text": {"fixed_shape": list(text_shape)},
        },
        "tokenizer": {
            "mode": "tokenizers-json" if tokenizer_path.exists() else "runtime-bound",
        },
    }
    if tokenizer_path.exists():
        task_metadata["tokenizer"]["file"] = "upstream/tokenizer.json"

    if source_format == "huggingface" and isinstance(model, CLIPModel):
        task_metadata["projection_dim"] = int(model.config.projection_dim)
        task_metadata["embedding_dim"] = int(model.config.projection_dim)
        task_metadata["vision_config"] = {
            "image_size": int(model.config.vision_config.image_size),
            "patch_size": int(model.config.vision_config.patch_size),
        }
        task_metadata["text_config"] = {
            "max_position_embeddings": int(
                model.config.text_config.max_position_embeddings
            ),
            "vocab_size": int(model.config.text_config.vocab_size),
        }

    return task_metadata


def export_clip_to_onnx(
    *,
    model_root: Path,
    model_name: str | None = None,
    description: str | None = None,
    version: str = "1.0.0",
    opset: int = DEFAULT_ONNX_OPSET,
    image_shape: tuple[int, ...] = (1, 3, 224, 224),
    text_shape: tuple[int, ...] = (1, 77),
) -> dict[str, Any]:
    """Build the full CLIP asset package with a single entrypoint."""
    model_root = model_root.resolve()
    source_spec = _read_source_spec(model_root)
    _maybe_download_hf_upstream(model_root, source_spec)

    onnx_dir = model_root / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    source_format, model = _load_clip_model(model_root)

    vision_fp32 = onnx_dir / "vision.fp32.onnx"
    text_fp32 = onnx_dir / "text.fp32.onnx"
    vision_fp16 = onnx_dir / "vision.fp16.onnx"
    text_fp16 = onnx_dir / "text.fp16.onnx"

    _export_vision_model(
        source_format=source_format,
        model=model,
        output_path=vision_fp32,
        opset=opset,
        image_shape=image_shape,
    )
    _export_text_model(
        source_format=source_format,
        model=model,
        output_path=text_fp32,
        opset=opset,
        text_shape=text_shape,
    )
    _convert_to_fp16(vision_fp32, vision_fp16)
    _convert_to_fp16(text_fp32, text_fp16)

    resolved_model_name = model_name or _default_model_name(source_spec)
    resolved_description = description or _default_description(resolved_model_name)
    task_metadata = _build_clip_task_metadata(
        model_root=model_root,
        source_format=source_format,
        model=model,
        image_shape=image_shape,
        text_shape=text_shape,
    )
    model_info_path = write_model_metadata(
        model_root=model_root,
        task="clip",
        model_name=resolved_model_name,
        description=resolved_description,
        source_format=source_format,
        source_repo_id=source_spec.repo_id,
        runtimes=_build_clip_runtimes(),
        task_metadata=task_metadata,
        version=version,
    )
    validation = validate_clip_assets(model_root=model_root)
    readme_path = generate_model_card(model_root=model_root, template="clip")

    return {
        "model_info": model_info_path,
        "readme": readme_path,
        "validation": validation,
        "artifacts": {
            "vision_fp32": vision_fp32,
            "vision_fp16": vision_fp16,
            "text_fp32": text_fp32,
            "text_fp16": text_fp16,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--description")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--opset", type=int, default=DEFAULT_ONNX_OPSET)
    parser.add_argument("--image-shape", default="1,3,224,224")
    parser.add_argument("--text-shape", default="1,77")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    export_clip_to_onnx(
        model_root=args.model_root.resolve(),
        model_name=args.model_name,
        description=args.description,
        version=args.version,
        opset=args.opset,
        image_shape=parse_shape(args.image_shape),
        text_shape=parse_shape(args.text_shape),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
