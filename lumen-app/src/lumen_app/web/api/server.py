"""Server management API endpoints."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException

from lumen_app.utils.logger import get_logger
from lumen_app.web.core.state import app_state
from lumen_app.web.models.server import (
    ServerConfig,
    ServerLogs,
    ServerRestartRequest,
    ServerStartRequest,
    ServerStatus,
    ServerStopRequest,
)

logger = get_logger("lumen.web.api.server")
router = APIRouter()

# Track server start time for uptime calculation
_server_start_time: float | None = None


@router.get("/status", response_model=ServerStatus)
async def get_server_status():
    """Get the ML server status."""
    status = app_state.server_status

    uptime = None
    if status.running and _server_start_time:
        uptime = time.time() - _server_start_time

    return ServerStatus(
        running=status.running,
        pid=status.pid,
        port=status.port,
        host="0.0.0.0",  # TODO: Get from actual config
        uptime_seconds=uptime,
        service_name="lumen-ai",
        config_path=status.config_path,
        environment="lumen_env",  # TODO: Get from actual config
        health="healthy" if status.running else "unknown",
        last_error=None,
    )


@router.post("/start", response_model=ServerStatus)
async def start_server(request: ServerStartRequest):
    """Start the ML server."""
    logger.info("Starting ML server")

    if app_state.server_status.running:
        raise HTTPException(status_code=400, detail="Server is already running")

    # TODO: Implement actual server startup logic
    # This would involve:
    # 1. Loading the configuration
    # 2. Creating the gRPC server subprocess
    # 3. Monitoring the process

    # For now, simulate startup
    await asyncio.sleep(1)

    success = await app_state.start_server(config_path=request.config_path)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to start server")

    global _server_start_time
    _server_start_time = time.time()

    return ServerStatus(
        running=True,
        pid=app_state.server_status.pid,
        port=request.port or 50051,
        host=request.host or "0.0.0.0",
        uptime_seconds=0,
        service_name=request.service_name or "lumen-ai",
        config_path=request.config_path,
        environment=request.environment,
        health="healthy",
    )


@router.post("/stop", response_model=ServerStatus)
async def stop_server(request: ServerStopRequest):
    """Stop the ML server."""
    logger.info("Stopping ML server")

    if not app_state.server_status.running:
        raise HTTPException(status_code=400, detail="Server is not running")

    # TODO: Implement graceful shutdown
    # This would involve:
    # 1. Sending shutdown signal to subprocess
    # 2. Waiting for graceful shutdown
    # 3. Force kill if necessary

    success = await app_state.stop_server()

    if not success:
        raise HTTPException(status_code=500, detail="Failed to stop server")

    global _server_start_time
    _server_start_time = None

    return ServerStatus(
        running=False,
        pid=None,
        port=app_state.server_status.port,
        host="0.0.0.0",
        uptime_seconds=None,
        service_name=app_state.server_status.service_name,
        config_path=app_state.server_status.config_path,
        environment="lumen_env",
        health="unknown",
    )


@router.post("/restart", response_model=ServerStatus)
async def restart_server(request: ServerRestartRequest):
    """Restart the ML server."""
    logger.info("Restarting ML server")

    # Stop if running
    if app_state.server_status.running:
        stop_request = ServerStopRequest(force=request.force, timeout=request.timeout)
        await stop_server(stop_request)

    # Start with new config
    start_request = ServerStartRequest(
        config_path=request.config_path,
        port=request.port,
        host=request.host,
        environment=request.environment,
    )
    return await start_server(start_request)


@router.get("/logs", response_model=ServerLogs)
async def get_server_logs(lines: int = 100, since: float | None = None):
    """Get server logs."""
    # TODO: Implement log retrieval from file or buffer
    return ServerLogs(
        logs=[],
        total_lines=0,
        new_lines=0,
    )
