"""Server management API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from lumen_app.schemas.server import (
    ServerLogs,
    ServerRestartRequest,
    ServerStartRequest,
    ServerStatus,
    ServerStopRequest,
)
from lumen_app.services.state import app_state
from lumen_app.utils.logger import get_logger

logger = get_logger("lumen.web.api.server")
router = APIRouter()


@router.get("/status", response_model=ServerStatus)
async def get_server_status():
    """
    Get the current ML server status.

    Returns detailed information about the running server including:
    - Running state and PID
    - Port and host configuration
    - Uptime in seconds
    - Health status
    - Configuration path
    """
    manager = app_state.server_manager

    # Get basic status
    running = manager.is_running
    pid = manager.pid
    uptime = manager.uptime_seconds

    # Perform health check if running
    health = "unknown"
    if running:
        health = await manager.health_check()

    return ServerStatus(
        running=running,
        pid=pid,
        port=manager.port or 50051,
        host="0.0.0.0",
        uptime_seconds=uptime,
        service_name="lumen-ai",  # TODO: Get from config
        config_path=manager.config_path,
        environment=manager.environment or "lumen_env",
        health=health,
        last_error=None,  # TODO: Track last error
    )


@router.post("/start", response_model=ServerStatus)
async def start_server(request: ServerStartRequest):
    """
    Start the ML server with specified configuration.

    Args:
        request: Server start configuration including:
            - config_path: Path to the Lumen YAML configuration
            - port: Optional port override
            - host: Host address (currently unused, always 0.0.0.0)
            - environment: Conda environment name

    Returns:
        Current server status after startup

    Raises:
        HTTPException 400: If server is already running
        HTTPException 404: If config file not found
        HTTPException 500: If server fails to start
    """
    logger.info(f"Starting ML server with config: {request.config_path}")

    manager = app_state.server_manager

    # Check if already running
    if manager.is_running:
        raise HTTPException(
            status_code=400,
            detail="Server is already running. Stop it first or use restart.",
        )

    if request.config_path is None:
        raise HTTPException(
            status_code=400, detail="config_path is required to start the server."
        )

    try:
        # Start the server
        success = await manager.start(
            config_path=request.config_path,
            port=request.port,
            log_level="INFO",  # TODO: Make configurable
            environment=request.environment,
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Server failed to start. Check logs for details.",
            )

        logger.info("✓ ML server started successfully")

        # Return current status
        return await get_server_status()

    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    except RuntimeError as e:
        logger.error(f"Runtime error starting server: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error starting server: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")


@router.post("/stop", response_model=ServerStatus)
async def stop_server(request: ServerStopRequest):
    """
    Stop the running ML server.

    Args:
        request: Stop configuration including:
            - force: If True, force kill immediately without graceful shutdown
            - timeout: Maximum seconds to wait for graceful shutdown

    Returns:
        Current server status after shutdown

    Raises:
        HTTPException 400: If server is not running
        HTTPException 500: If server fails to stop
    """
    logger.info("Stopping ML server")

    manager = app_state.server_manager

    # Check if running
    if not manager.is_running:
        raise HTTPException(
            status_code=400, detail="Server is not running. Nothing to stop."
        )

    try:
        # Stop the server
        success = await manager.stop(timeout=request.timeout, force=request.force)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Server failed to stop gracefully. Check logs for details.",
            )

        logger.info("✓ ML server stopped successfully")

        # Return current status
        return await get_server_status()

    except Exception as e:
        logger.error(f"Error stopping server: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stop server: {str(e)}")


@router.post("/restart", response_model=ServerStatus)
async def restart_server(request: ServerRestartRequest):
    """
    Restart the ML server with optional new configuration.

    This is equivalent to stop + start, but handles the sequencing automatically.

    Args:
        request: Restart configuration including:
            - config_path: Optional new config path (uses existing if not provided)
            - port: Optional new port (uses existing if not provided)
            - host: Host address (currently unused)
            - environment: Environment name
            - force: If True, force kill during stop
            - timeout: Maximum seconds to wait for graceful shutdown

    Returns:
        Current server status after restart

    Raises:
        HTTPException 400: If no config path available
        HTTPException 500: If restart fails
    """
    logger.info("Restarting ML server")

    manager = app_state.server_manager

    try:
        # Restart the server
        success = await manager.restart(
            config_path=request.config_path,
            port=request.port,
            log_level="INFO",  # TODO: Make configurable
            timeout=request.timeout,
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Server failed to restart. Check logs for details.",
            )

        logger.info("✓ ML server restarted successfully")

        # Return current status
        return await get_server_status()

    except ValueError as e:
        logger.error(f"Invalid restart configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error restarting server: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to restart server: {str(e)}"
        )


@router.get("/logs", response_model=ServerLogs)
async def get_server_logs(lines: int = 100, since: float | None = None):
    """
    Get recent server logs.

    Args:
        lines: Number of recent log lines to return (default: 100, 0 for all)
        since: Unix timestamp to filter logs (currently unused)

    Returns:
        Server logs with metadata

    Note:
        The 'since' parameter is reserved for future filtering implementation.
        Currently returns the most recent N lines from the log buffer.
    """
    manager = app_state.server_manager

    # Get logs from manager
    log_lines = manager.get_logs(tail=lines)

    return ServerLogs(
        logs=log_lines,
        total_lines=len(log_lines),
        new_lines=0,  # TODO: Implement incremental log fetching
    )
