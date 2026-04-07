"""Download upstream Hugging Face artifacts into a model repository."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from scripts.common._shared import collect_relative_files, ensure_dir, write_json


def download_hf_artifacts(
    *,
    repo_id: str,
    revision: str,
    output_dir: Path,
    allow_patterns: list[str],
) -> dict[str, object]:
    """Download a curated subset of Hugging Face artifacts into `upstream/`."""
    output_dir = output_dir.resolve()
    upstream_dir = ensure_dir(output_dir / "upstream")

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(upstream_dir),
        allow_patterns=allow_patterns,
    )

    manifest = {
        "format": "huggingface",
        "repo_id": repo_id,
        "revision": revision,
        "downloaded_files": collect_relative_files(upstream_dir),
    }
    write_json(upstream_dir / "source.json", manifest)
    return manifest
