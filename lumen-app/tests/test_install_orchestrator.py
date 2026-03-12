"""Tests for installation orchestration cancellation behavior."""

from pathlib import Path

import pytest

from lumen_app.schemas.install import InstallSetupRequest
from lumen_app.services.install_orchestrator import InstallOrchestrator
from lumen_app.services.install_task_repository import InMemoryInstallTaskRepository


@pytest.mark.asyncio
async def test_cancel_installation_clears_cache_dir(tmp_path: Path):
    """Cancelling an installation should clear all files under cache_dir."""
    (tmp_path / "lumen-config.yaml").write_text("version: 1\n", encoding="utf-8")
    (tmp_path / "wheels").mkdir()
    (tmp_path / "wheels" / "dummy.whl").write_text("wheel", encoding="utf-8")
    (tmp_path / "micromamba").mkdir()

    orchestrator = InstallOrchestrator(InMemoryInstallTaskRepository())
    request = InstallSetupRequest(
        preset="cpu",
        cache_dir=str(tmp_path),
        environment_name="lumen_env",
        force_reinstall=False,
    )

    task = await orchestrator.create_install_task(request)
    cancelled_task = await orchestrator.cancel_installation(task.task_id)

    assert cancelled_task is not None
    assert cancelled_task.status == "cancelled"
    assert cancelled_task.progress == 0
    assert list(tmp_path.iterdir()) == []
