"""Generate `model_info.json` for model assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.common._shared import write_json


def build_model_info_payload(
    *,
    task: str,
    model_name: str,
    description: str,
    source_format: str,
    source_repo_id: str,
    runtimes: dict[str, Any],
    task_metadata: dict[str, Any] | None = None,
    version: str = "1.0.0",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a runtime-facing `model_info.json` payload."""
    payload: dict[str, Any] = {
        "name": model_name,
        "version": version,
        "description": description,
        "model_type": task,
        "source": {
            "format": source_format,
            "repo_id": source_repo_id,
        },
        "runtimes": runtimes,
    }
    if task_metadata:
        payload["task_metadata"] = task_metadata
    if metadata:
        payload["metadata"] = metadata
    return payload


def write_model_metadata(
    *,
    model_root: Path,
    task: str,
    model_name: str,
    description: str,
    source_format: str,
    source_repo_id: str,
    runtimes: dict[str, Any],
    task_metadata: dict[str, Any] | None = None,
    version: str = "1.0.0",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write `model_info.json` and return its path."""
    model_info = build_model_info_payload(
        task=task,
        model_name=model_name,
        description=description,
        source_format=source_format,
        source_repo_id=source_repo_id,
        runtimes=runtimes,
        task_metadata=task_metadata,
        version=version,
        metadata=metadata,
    )

    model_info_path = model_root / "model_info.json"
    write_json(model_info_path, model_info)
    return model_info_path
