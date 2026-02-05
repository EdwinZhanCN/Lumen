"""
Server Manager for gRPC ML Server Process.

This module provides lifecycle management for the gRPC ML server subprocess,
including starting, stopping, health checking, and log capture.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from pathlib import Path
from typing import Literal

from lumen_app.utils.installation import MicromambaInstaller
from lumen_app.utils.logger import get_logger

logger = get_logger("lumen.web.server_manager")


class ServerManager:
    """
    Manages the gRPC ML server as a subprocess.

    This class handles:
    - Starting the server with proper configuration
    - Monitoring server health and status
    - Capturing and buffering logs
    - Graceful shutdown with timeout handling
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        env_name: str = "lumen_env",
        max_log_lines: int = 1000,
    ):
        """
        Initialize the server manager.

        Args:
            cache_dir: Cache directory containing micromamba installation
            env_name: Micromamba environment name
            max_log_lines: Maximum number of log lines to keep in memory
        """
        self.process: asyncio.subprocess.Process | None = None
        self.config_path: str | None = None
        self.port: int | None = None
        self.start_time: float | None = None
        self.log_buffer: deque[str] = deque(maxlen=max_log_lines)
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.env_name = env_name
        self.environment: str | None = None
        self._log_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    def update_cache_dir(
        self, cache_dir: str | Path, env_name: str | None = None
    ) -> None:
        """Update cache_dir (and optionally env_name) for micromamba resolution."""
        self.cache_dir = Path(cache_dir).expanduser()
        if env_name is not None:
            self.env_name = env_name

    @property
    def is_running(self) -> bool:
        """Check if the server process is running."""
        if self.process is None:
            return False

        # Check if process is still alive
        return self.process.returncode is None

    @property
    def pid(self) -> int | None:
        """Get the server process PID."""
        if self.process:
            return self.process.pid
        return None

    @property
    def uptime_seconds(self) -> float | None:
        """Get server uptime in seconds."""
        if self.start_time and self.is_running:
            return time.time() - self.start_time
        return None

    async def start(
        self,
        config_path: str,
        port: int | None = None,
        log_level: str = "INFO",
        environment: str | None = None,
    ) -> bool:
        """
        Start the gRPC ML server as a subprocess.

        Args:
            config_path: Path to the Lumen configuration YAML file
            port: Port number (overrides config file)
            log_level: Logging level for the server
            environment: Conda environment name (if using conda)

        Returns:
            True if server started successfully, False otherwise

        Raises:
            RuntimeError: If server is already running or config is invalid
        """
        if self.is_running:
            raise RuntimeError("Server is already running")

        # Validate config path
        config_file = Path(config_path).expanduser()
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Starting ML server with config: {config_path}")

        if self.cache_dir is None:
            raise RuntimeError("cache_dir is required to start the server")
        self.environment = environment or self.env_name

        cmd = [
            "python",
            "-m",
            "lumen.server",
            "--config",
            str(config_file),
            "--log-level",
            log_level,
        ]

        if port:
            cmd.extend(["--port", str(port)])

        env_name = self.environment or self.env_name
        installer = MicromambaInstaller(self.cache_dir)
        micromamba_exe = installer.get_executable()
        env_path = self.cache_dir / "micromamba" / "envs" / env_name
        cmd = [
            str(micromamba_exe),
            "run",
            "-p",
            str(env_path),
            *cmd,
        ]

        logger.debug(f"Server command: {' '.join(cmd)}")

        try:
            # Start the subprocess
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                stdin=asyncio.subprocess.DEVNULL,
            )

            self.config_path = config_path
            self.port = port
            self.start_time = time.time()

            logger.info(f"Server process started with PID: {self.process.pid}")

            # Start log capture task
            self._log_task = asyncio.create_task(self._capture_logs())

            # Wait for server to be ready (with timeout)
            ready = await self._wait_for_ready(timeout=30.0)

            if not ready:
                logger.error("Server failed to start within timeout")
                await self.stop(force=True)
                return False

            logger.info("✓ ML server is ready")
            return True

        except Exception as e:
            logger.error(f"Failed to start server: {e}", exc_info=True)
            if self.process:
                await self.stop(force=True)
            return False

    async def stop(self, timeout: float = 30.0, force: bool = False) -> bool:
        """
        Stop the gRPC ML server.

        Args:
            timeout: Maximum time to wait for graceful shutdown
            force: If True, skip graceful shutdown and kill immediately

        Returns:
            True if server stopped successfully, False otherwise
        """
        if not self.process:
            logger.warning("No server process to stop")
            return True

        if not self.is_running:
            logger.info("Server process already stopped")
            self._cleanup()
            return True

        logger.info(f"Stopping ML server (PID: {self.process.pid})")

        try:
            if force:
                # Force kill immediately
                logger.warning("Force killing server process")
                self.process.kill()
            else:
                # Try graceful shutdown first
                logger.info("Sending SIGTERM for graceful shutdown")
                self.process.terminate()

                # Wait for graceful shutdown with timeout
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=timeout)
                    logger.info("Server stopped gracefully")
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Server did not stop within {timeout}s, force killing"
                    )
                    self.process.kill()
                    await self.process.wait()

            self._cleanup()
            logger.info("✓ Server stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping server: {e}", exc_info=True)
            return False

    async def restart(
        self,
        config_path: str | None = None,
        port: int | None = None,
        log_level: str = "INFO",
        timeout: float = 30.0,
    ) -> bool:
        """
        Restart the server.

        Args:
            config_path: New config path (or use existing)
            port: New port (or use existing)
            log_level: Logging level
            timeout: Timeout for stop operation

        Returns:
            True if restart successful, False otherwise
        """
        logger.info("Restarting ML server")

        # Use existing config if not provided
        config_path = config_path or self.config_path
        port = port or self.port

        if not config_path:
            raise ValueError("No config path specified for restart")

        # Stop if running
        if self.is_running:
            success = await self.stop(timeout=timeout)
            if not success:
                logger.error("Failed to stop server for restart")
                return False

            # Wait a bit for resources to be freed
            await asyncio.sleep(1.0)

        # Start with new/existing config
        return await self.start(
            config_path=config_path,
            port=port,
            log_level=log_level,
            environment=self.environment,
        )

    async def health_check(self) -> Literal["healthy", "unhealthy", "unknown"]:
        """
        Perform a health check on the server.

        Returns:
            "healthy" if server is running and responsive
            "unhealthy" if server process exists but not responsive
            "unknown" if server is not running
        """
        if not self.is_running:
            return "unknown"

        # TODO: Implement actual gRPC health check
        # For now, just check if process is running
        return "healthy"

    def get_logs(self, tail: int = 100) -> list[str]:
        """
        Get recent log lines.

        Args:
            tail: Number of recent lines to return (0 for all)

        Returns:
            List of log lines
        """
        if tail > 0:
            # Return last N lines
            return list(self.log_buffer)[-tail:]
        else:
            # Return all lines
            return list(self.log_buffer)

    async def _capture_logs(self):
        """Background task to capture server stdout/stderr."""
        if not self.process or not self.process.stdout:
            return

        logger.debug("Starting log capture")

        try:
            async for line in self.process.stdout:
                try:
                    log_line = line.decode("utf-8").rstrip()
                    self.log_buffer.append(log_line)

                    # Also log to our logger for debugging
                    logger.debug(f"[Server] {log_line}")

                except Exception as e:
                    logger.warning(f"Error decoding log line: {e}")

        except asyncio.CancelledError:
            logger.debug("Log capture cancelled")
        except Exception as e:
            logger.error(f"Error capturing logs: {e}", exc_info=True)

    async def _wait_for_ready(self, timeout: float = 30.0) -> bool:
        """
        Wait for the server to be ready.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if server is ready, False if timeout or error
        """
        logger.info(f"Waiting for server to be ready (timeout: {timeout}s)...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process died
            if not self.is_running:
                logger.error("Server process died during startup")
                return False

            # Look for startup success indicators in logs
            recent_logs = self.get_logs(tail=10)
            for log_line in recent_logs:
                if (
                    "listening on" in log_line.lower()
                    or "server running" in log_line.lower()
                ):
                    logger.info("Server startup detected in logs")
                    # Give it a moment to fully initialize
                    await asyncio.sleep(1.0)
                    return True

            # Wait a bit before checking again
            await asyncio.sleep(0.5)

        logger.error(f"Server did not start within {timeout}s")
        return False

    def _cleanup(self):
        """Clean up server state after shutdown."""
        if self._log_task and not self._log_task.done():
            self._log_task.cancel()

        self.process = None
        self.start_time = None
        self._log_task = None
