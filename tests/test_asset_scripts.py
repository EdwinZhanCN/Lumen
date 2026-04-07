"""Tests for model asset tooling scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import onnx
import torch
from onnx import TensorProto, helper
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

from scripts.clip.export_clip_to_onnx import export_clip_to_onnx
from scripts.common.generate_model_card import generate_model_card
from scripts.common.hf_download import download_hf_artifacts
from scripts.common.validate_onnx import validate_onnx_model
from scripts.common.write_model_info import write_model_metadata


def _create_identity_onnx(path: Path) -> None:
    """Create a tiny identity ONNX model for validation tests."""
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph(
        [node],
        "identity_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(
        graph,
        producer_name="lumen-tests",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    model.ir_version = 10
    onnx.save(model, str(path))


def _write_tokenizer(upstream_dir: Path) -> None:
    """Write a tiny tokenizer.json and companion tokenizer files."""
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "hello": 2,
                "world": 3,
            },
            unk_token="[UNK]",
        )
    )
    cast(Any, tokenizer).pre_tokenizer = Whitespace()
    tokenizer.save(str(upstream_dir / "tokenizer.json"))

    (upstream_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 8, "unk_token": "[UNK]", "pad_token": "[PAD]"}),
        encoding="utf-8",
    )
    (upstream_dir / "special_tokens_map.json").write_text(
        json.dumps({"unk_token": "[UNK]", "pad_token": "[PAD]"}),
        encoding="utf-8",
    )
    (upstream_dir / "preprocessor_config.json").write_text(
        json.dumps({"do_resize": True, "size": {"shortest_edge": 8}}),
        encoding="utf-8",
    )


def _build_tiny_clip_repo(model_root: Path) -> Path:
    """Create a tiny local Hugging Face CLIP-style repository."""
    upstream_dir = model_root / "upstream"
    upstream_dir.mkdir(parents=True, exist_ok=True)

    text_config = CLIPTextConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=1,
    )
    vision_config = CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
    )
    config = CLIPConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=8,
    )
    model = CLIPModel(config)
    model.save_pretrained(upstream_dir)
    _write_tokenizer(upstream_dir)
    (upstream_dir / "source.json").write_text(
        json.dumps(
            {
                "format": "huggingface",
                "repo_id": "local/tiny-clip",
                "revision": "test",
            }
        ),
        encoding="utf-8",
    )
    return upstream_dir


def test_hf_download_filters_files_and_writes_source(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "repo"

    def _fake_snapshot_download(
        *,
        repo_id: str,
        revision: str,
        local_dir: str,
        allow_patterns,
    ):
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        for name in allow_patterns:
            (target / name).write_text(f"{repo_id}@{revision}:{name}", encoding="utf-8")
        return str(target)

    monkeypatch.setattr("scripts.common.hf_download.snapshot_download", _fake_snapshot_download)

    manifest = download_hf_artifacts(
        repo_id="openai/clip-vit-base-patch32",
        revision="main",
        output_dir=output_dir,
        allow_patterns=["config.json", "tokenizer.json", "*.bin"],
    )

    assert (output_dir / "upstream" / "config.json").exists()
    source = json.loads((output_dir / "upstream" / "source.json").read_text(encoding="utf-8"))
    assert source["repo_id"] == "openai/clip-vit-base-patch32"
    assert source["revision"] == "main"
    assert "config.json" in source["downloaded_files"]
    assert manifest["format"] == "huggingface"


def test_write_model_info_generates_runtime_facing_file(tmp_path: Path) -> None:
    model_info_path = write_model_metadata(
        model_root=tmp_path,
        task="clip",
        model_name="clip-vit-base-patch32",
        description="CLIP deployment assets",
        source_format="huggingface",
        source_repo_id="openai/clip-vit-base-patch32",
        runtimes={
            "onnx": {
                "available": True,
                "components": ["vision", "text"],
                "precisions": ["fp32", "fp16"],
            }
        },
        task_metadata={
            "embedding_dim": 512,
            "input_contract": {
                "vision": {"fixed_shape": [1, 3, 224, 224]},
                "text": {"fixed_shape": [1, 77]},
            },
        },
    )

    model_info = json.loads(model_info_path.read_text(encoding="utf-8"))
    assert model_info["name"] == "clip-vit-base-patch32"
    assert model_info["runtimes"]["onnx"]["components"] == ["vision", "text"]
    assert model_info["task_metadata"]["input_contract"]["vision"]["fixed_shape"] == [
        1,
        3,
        224,
        224,
    ]
    assert model_info["task_metadata"]["embedding_dim"] == 512


def test_generate_model_card_renders_required_sections(tmp_path: Path) -> None:
    write_model_metadata(
        model_root=tmp_path,
        task="clip",
        model_name="clip-vit-base-patch32",
        description="CLIP deployment assets",
        source_format="huggingface",
        source_repo_id="openai/clip-vit-base-patch32",
        runtimes={
            "onnx": {
                "available": True,
                "components": ["vision", "text"],
                "precisions": ["fp32", "fp16"],
            }
        },
        task_metadata={
            "input_contract": {
                "vision": {"fixed_shape": [1, 3, 224, 224]},
                "text": {"fixed_shape": [1, 77]},
            }
        },
    )
    upstream_dir = tmp_path / "upstream"
    upstream_dir.mkdir(parents=True, exist_ok=True)
    (upstream_dir / "source.json").write_text(
        json.dumps(
            {
                "format": "huggingface",
                "repo_id": "openai/clip-vit-base-patch32",
                "revision": "main",
            }
        ),
        encoding="utf-8",
    )
    readme_path = generate_model_card(model_root=tmp_path, template="clip")
    content = readme_path.read_text(encoding="utf-8")

    assert "## What This Repository Contains" in content
    assert "## Upstream Source" in content
    assert "## Deployment Artifacts" in content
    assert "## Runtime Support" in content
    assert "## Input Contract" in content
    assert "## Validation" in content
    assert "## Known Limitations" in content


def test_validate_onnx_outputs_execution_report(tmp_path: Path) -> None:
    model_path = tmp_path / "identity.onnx"
    _create_identity_onnx(model_path)

    payload = validate_onnx_model(model_path=model_path, task="identity")

    assert payload["success"] is True
    assert payload["providers"] == ["CPUExecutionProvider"]
    assert payload["output_specs"][0]["shape"] == [1, 4]


def test_clip_asset_pipeline_exports_and_validates(tmp_path: Path) -> None:
    _build_tiny_clip_repo(tmp_path)
    result = export_clip_to_onnx(
        model_root=tmp_path,
        model_name="tiny-clip",
        description="Tiny CLIP deployment assets",
        image_shape=(1, 3, 8, 8),
        text_shape=(1, 8),
    )

    assert (tmp_path / "onnx" / "vision.fp32.onnx").exists()
    assert (tmp_path / "onnx" / "vision.fp16.onnx").exists()
    assert (tmp_path / "onnx" / "text.fp32.onnx").exists()
    assert (tmp_path / "onnx" / "text.fp16.onnx").exists()
    assert (tmp_path / "validation" / "onnx" / "fp32.json").exists()
    assert (tmp_path / "validation" / "onnx" / "fp16.json").exists()
    validation = result["validation"]
    assert validation["reports"]["fp32"]["model_info_consistent"] is True
    assert validation["reports"]["fp32"]["components"]["vision"]["success"] is True
    assert validation["reports"]["fp16"]["components"]["text"]["success"] is True
    model_info = json.loads((tmp_path / "model_info.json").read_text(encoding="utf-8"))
    assert model_info["runtimes"]["onnx"]["components"] == ["vision", "text"]
    assert model_info["task_metadata"]["embedding_dim"] == 8
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "tiny-clip" in readme
    assert "tokenizer.json" in readme


class _FakeOpenCLIPModel(torch.nn.Module):
    """Tiny OpenCLIP-like module for exporter tests."""

    def __init__(self) -> None:
        super().__init__()
        self.image_proj = torch.nn.Linear(3 * 8 * 8, 8)
        self.text_emb = torch.nn.Embedding(8, 8)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Return fake image embeddings."""
        return self.image_proj(image.reshape(image.shape[0], -1))

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Return fake text embeddings."""
        return self.text_emb(text).mean(dim=1)


def test_openclip_asset_pipeline_exports_with_reparameterize(
    monkeypatch,
    tmp_path: Path,
) -> None:
    upstream_dir = tmp_path / "upstream"
    upstream_dir.mkdir(parents=True, exist_ok=True)
    (upstream_dir / "source.json").write_text(
        json.dumps(
            {
                "format": "openclip",
                "repo_id": "MobileCLIP2-S4",
                "pretrained": "dfndr2b",
                "export_options": {"reparameterize": True},
            }
        ),
        encoding="utf-8",
    )
    marker = {"reparameterized": False}

    def _fake_create_model_and_transforms(repo_id: str, pretrained: str | None = None):
        assert repo_id == "MobileCLIP2-S4"
        assert pretrained == "dfndr2b"
        return _FakeOpenCLIPModel(), None, None

    def _fake_reparameterize_model(model: torch.nn.Module) -> torch.nn.Module:
        marker["reparameterized"] = True
        return model

    monkeypatch.setattr(
        "scripts.clip.export_clip_to_onnx.open_clip.create_model_and_transforms",
        _fake_create_model_and_transforms,
        raising=False,
    )
    monkeypatch.setattr(
        "scripts.clip.export_clip_to_onnx.reparameterize_model",
        _fake_reparameterize_model,
        raising=False,
    )

    result = export_clip_to_onnx(
        model_root=tmp_path,
        model_name="mobileclip2-s4",
        description="OpenCLIP deployment assets",
        image_shape=(1, 3, 8, 8),
        text_shape=(1, 8),
    )

    assert marker["reparameterized"] is True
    assert (tmp_path / "onnx" / "vision.fp32.onnx").exists()
    assert (tmp_path / "onnx" / "text.fp16.onnx").exists()
    assert result["validation"]["reports"]["fp32"]["tokenizer"]["loadable"] is False
    assert result["validation"]["reports"]["fp32"]["components"]["vision"]["success"] is True
    model_info = json.loads((tmp_path / "model_info.json").read_text(encoding="utf-8"))
    assert model_info["source"]["format"] == "openclip"
