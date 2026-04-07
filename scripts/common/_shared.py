"""Shared helpers for asset tooling scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_ONNX_OPSET = 17


def ensure_dir(path: Path) -> Path:
    """Create a directory and return its path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv(value: str) -> list[str]:
    """Parse a comma-separated list into trimmed strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_shape(value: str) -> tuple[int, ...]:
    """Parse a comma-separated shape string into an integer tuple."""
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("Shape must contain at least one dimension.")
    return parsed


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON dictionary with stable formatting."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def collect_relative_files(base_dir: Path) -> list[str]:
    """Return a sorted list of files relative to a base directory."""
    if not base_dir.exists():
        return []
    return sorted(
        str(path.relative_to(base_dir))
        for path in base_dir.rglob("*")
        if path.is_file()
    )
