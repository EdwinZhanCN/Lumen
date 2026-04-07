"""Generate a model repository README from model assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.common._shared import read_json


def _render_runtime_support(model_info: dict[str, Any]) -> str:
    """Render runtime support lines."""
    runtimes = model_info.get("runtimes", {})
    lines: list[str] = []
    for runtime_name, runtime_info in runtimes.items():
        components = ", ".join(runtime_info.get("components", [])) or "none"
        precisions = ", ".join(runtime_info.get("precisions", [])) or "none"
        lines.append(
            f"- `{runtime_name}`: components=`{components}`, precisions=`{precisions}`"
        )
    return "\n".join(lines) if lines else "- None recorded"


def _render_artifacts(model_info: dict[str, Any]) -> str:
    """Render artifact lines grouped by runtime and component."""
    artifact_lines: list[str] = []
    runtimes = model_info.get("runtimes", {})
    for runtime_name, runtime_info in runtimes.items():
        artifact_lines.append(f"- `{runtime_name}`")
        for component in runtime_info.get("components", []):
            files = [
                f"{runtime_name}/{component}.{precision}.{runtime_name}"
                for precision in runtime_info.get("precisions", [])
            ]
            joined = ", ".join(f"`{file_name}`" for file_name in files)
            artifact_lines.append(f"  - `{component}`: {joined}")
    return "\n".join(artifact_lines) if artifact_lines else "- None recorded"


def _render_input_contract(model_info: dict[str, Any]) -> str:
    """Render input contract lines."""
    task_metadata = model_info.get("task_metadata", {})
    contract = task_metadata.get("input_contract", {})
    lines: list[str] = []
    for component, metadata in contract.items():
        fixed_shape = metadata.get("fixed_shape")
        if fixed_shape is not None:
            lines.append(f"- `{component}` fixed shape: `{fixed_shape}`")
    return "\n".join(lines) if lines else "- No explicit input contract recorded"


def _render_validation(model_root: Path) -> str:
    """Render validation lines from the validation directory."""
    validation_dir = model_root / "validation" / "onnx"
    if not validation_dir.exists():
        return "- No validation reports found"
    lines = []
    for report_path in sorted(validation_dir.glob("*.json")):
        report = read_json(report_path)
        precision = report.get("precision", report_path.stem)
        status = "ok" if report.get("success", True) else "failed"
        lines.append(f"- `{precision}`: {status}")
    return "\n".join(lines) if lines else "- No validation reports found"


def render_model_card(
    *,
    model_root: Path,
    template: str = "generic",
) -> str:
    """Render a README body for a model repository."""
    model_info = read_json(model_root / "model_info.json")
    source_path = model_root / "upstream" / "source.json"
    source = read_json(source_path) if source_path.exists() else {}

    title = f"# {model_info['name']}"
    sections = [
        title,
        "",
        "## What This Repository Contains",
        f"This repository contains Lumen deployment assets for the `{model_info['model_type']}` task.",
        "",
        "## Upstream Source",
        f"- Format: `{source.get('format', model_info['source']['format'])}`",
        f"- Repository: `{source.get('repo_id', model_info['source']['repo_id'])}`",
        f"- Revision: `{source.get('revision', 'unknown')}`",
        "",
        "## Deployment Artifacts",
        _render_artifacts(model_info),
        "",
        "## Runtime Support",
        _render_runtime_support(model_info),
        "",
        "## Input Contract",
        _render_input_contract(model_info),
        "",
        "## Validation",
        _render_validation(model_root),
        "",
        "## Known Limitations",
        "- Upstream truth lives in `upstream/`; deployment truth lives in `onnx/`.",
        "- Fixed batch and fixed shape are explicit deployment constraints.",
        "- `tokenizer.json` remains outside ONNX and is expected to be loaded at runtime.",
    ]

    if template == "clip":
        sections.insert(
            3,
            "This repository is generated for a CLIP family model with separate `vision` and `text` ONNX exports.",
        )

    return "\n".join(sections).strip() + "\n"


def generate_model_card(*, model_root: Path, template: str = "generic") -> Path:
    """Generate the README file for a model repository."""
    readme_path = model_root / "README.md"
    readme_path.write_text(
        render_model_card(model_root=model_root, template=template),
        encoding="utf-8",
    )
    return readme_path
