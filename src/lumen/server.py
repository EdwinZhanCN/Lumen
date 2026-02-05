"""
Server startup module for Lumen App Service (Hub Mode).

This module provides the main server initialization and startup logic for running
multiple Lumen services in hub mode, integrating with lumen-resources for
configuration and model loading.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import uuid
from concurrent import futures
from typing import Any

import colorlog
import grpc
from lumen_resources import Downloader, DownloadResult, load_and_validate_config
from lumen_resources.lumen_config import LumenConfig, Mdns

from lumen.service import AppService
from lumen.utils.logger import get_logger

try:
    from zeroconf import ServiceInfo, Zeroconf
except ImportError:
    ServiceInfo = None  # type: ignore
    Zeroconf = None  # type: ignore

logger = get_logger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid."""

    pass


def setup_logging(log_level: str = "INFO"):
    """
    Configure the root logger for the application.

    This function clears any pre-existing handlers, sets the requested log
    level, and attaches a single colorized stream handler for console output.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any handlers that may have been pre-configured
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add a colorized console handler
    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(cyan)s[%(name)s]%(reset)s %(message)s",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def setup_mdns(port: int, mdns_config: Mdns | None) -> tuple[Any, Any]:
    """
    Set up mDNS advertisement for hub service.

    Args:
        port: Service port number
        mdns_config: mDNS configuration

    Returns:
        Tuple of (zeroconf, service_info) or (None, None) if setup fails
    """
    if Zeroconf is None or ServiceInfo is None:
        logger.warning("zeroconf package not installed, mDNS unavailable")
        return None, None

    try:
        # Determine advertised IP
        ip = os.getenv("ADVERTISE_IP")
        if not ip:
            try:
                # Best-effort LAN IP detection
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
            except Exception:
                ip = socket.gethostbyname(socket.gethostname())

        if ip.startswith("127."):
            logger.warning(
                f"mDNS advertising loopback IP {ip}; "
                "other devices may not reach the service. "
                "Set ADVERTISE_IP to a LAN IP."
            )

        # Build TXT record properties
        props = {
            "uuid": os.getenv("SERVICE_UUID", str(uuid.uuid4())),
            "status": os.getenv("SERVICE_STATUS", "ready"),
            "version": os.getenv("SERVICE_VERSION", "1.0.0"),
        }

        # Get service type and name from config
        service_type = (
            getattr(mdns_config, "type", None)
            or getattr(mdns_config, "service_type", None)
            or "_lumen._tcp.local."
        )
        instance_name = (
            getattr(mdns_config, "name", None)
            or getattr(mdns_config, "service_name", None)
            or "Lumen-Hub"
        )
        full_name = f"{instance_name}.{service_type}"

        # Create service info
        service_info = ServiceInfo(
            type_=service_type,
            name=full_name,
            addresses=[socket.inet_aton(ip)],
            port=port,
            properties=props,
            server=f"{socket.gethostname()}.local.",
        )

        # Register service
        zeroconf = Zeroconf()
        zeroconf.register_service(service_info)
        logger.info(f"✓ mDNS advertised: {full_name} at {ip}:{port}")

        return zeroconf, service_info

    except Exception as e:
        logger.warning(f"mDNS advertisement failed: {e}")
        return None, None


def handle_download_results(results: dict[str, DownloadResult]):
    """
    Processes download results, logs them, and exits if critical failures occurred.

    Args:
        results: Dictionary of model_type to DownloadResult
    """
    successful_downloads = []
    failed_downloads = []

    for _model_type, result in results.items():
        if result.success:
            successful_downloads.append(result)
        else:
            failed_downloads.append(result)

    # CRITICAL: If any model failed to download, abort the server startup.
    if failed_downloads:
        logger.error(
            "💥 Critical error: Model download failed. Cannot start the server."
        )
        for res in failed_downloads:
            logger.error(f"  - Model '{res.model_type}': {res.error}")
        sys.exit(1)

    # If all downloads were successful, log a summary.
    logger.info("✅ All required models are successfully downloaded and verified.")
    for res in successful_downloads:
        # Also check for non-critical warnings, like missing optional files.
        if res.missing_files:
            logger.warning(
                f"  - Model '{res.model_type}' is ready, but has missing optional files: "
                f"{', '.join(res.missing_files)}"
            )


def serve(config_path: str, port_override: int | None = None) -> None:
    """
    Initializes and starts the gRPC hub server based on a validated configuration.

    This server runner acts as an orchestrator for multiple services:
    1. Loads and validates the master lumen_config.yaml.
    2. Verifies deployment mode is 'hub'.
    3. Downloads all required model assets.
    4. Initializes all enabled services via AppService.
    5. Attaches all services to the gRPC server with routing support.
    6. Starts listening and handles graceful shutdown.

    Args:
        config_path: Path to the YAML configuration file
        port_override: Optional port number to override config setting
    """
    try:
        # Step 1: Load and validate the main configuration file
        logger.info(f"Loading configuration from: {config_path}")
        config: LumenConfig = load_and_validate_config(config_path)

        # Step 2: Ensure we are running in hub mode
        if config.deployment.mode != "hub":
            logger.error(
                f"This server is designed for 'hub' deployment mode only. "
                f"Current mode: {config.deployment.mode}"
            )
            sys.exit(1)

        # Step 3: Verify and download all required model assets
        logger.info("Verifying and downloading model assets...")
        downloader = Downloader(config, verbose=False)
        download_results = downloader.download_all()
        handle_download_results(download_results)

        # Step 4: Initialize AppService with all enabled services
        # Note: AppService.from_app_config() already initializes all services
        # via their from_config() factory methods, so no additional initialization needed
        logger.info("Initializing Lumen Hub service...")
        app_service = AppService.from_app_config(config)
        logger.info(f"✓ Loaded {len(app_service.services)} service(s)")

        # Step 5: Set up and start the gRPC server
        logger.info("Setting up gRPC server...")
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[("grpc.so_reuseport", 0)],
        )

        # Attach the hub router to the server
        # The router will handle dispatching requests to appropriate services
        app_service.router.attach_to_server(server)

        # Determine port: CLI override > config file > default
        preferred_port = port_override or config.server.port or 50051
        requested_addr = f"0.0.0.0::{preferred_port}"
        try:
            bound_port = server.add_insecure_port(requested_addr)
        except RuntimeError as exc:
            logger.warning(
                f"Port {preferred_port} bind raised {exc}; requesting OS-assigned port."
            )
            bound_port = 0

        if bound_port == 0:
            try:
                bound_port = server.add_insecure_port("0.0.0.0:0")
            except RuntimeError as exc:
                logger.error(f"Unable to bind gRPC server to any port: {exc}")
                sys.exit(1)

        if bound_port == 0:
            logger.error("Unable to bind gRPC server to any port.")
            sys.exit(1)

        port = bound_port
        listen_addr = f"0.0.0.0:{port}"
        server.start()

        # Log server startup info
        logger.info(f"🚀 Lumen Hub service listening on {listen_addr}")
        logger.info(f"✓ Running {len(app_service.services)} service(s):")
        for service in app_service.services:
            service_name = service.__class__.__name__
            logger.info(f"    - {service_name}")

        # Log capabilities of each service
        try:
            from google.protobuf import empty_pb2

            for service in app_service.services:
                try:
                    # Create a minimal context for capability probing
                    class _StartupServicerContext:
                        def abort(self, code, details):
                            pass

                        def set_code(self, code):
                            pass

                        def set_details(self, details):
                            pass

                    startup_context = _StartupServicerContext()
                    # Check if service has GetCapabilities method
                    if not hasattr(service, "GetCapabilities"):
                        continue
                    get_capabilities = getattr(service, "GetCapabilities")
                    capabilities = get_capabilities(empty_pb2.Empty(), startup_context)
                    supported_tasks = [task.name for task in capabilities.tasks]
                    service_name = service.__class__.__name__
                    logger.info(
                        f"    ✓ {service_name} tasks: {', '.join(supported_tasks)}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not retrieve capabilities for {service.__class__.__name__}: {e}"
                    )
        except Exception as e:
            logger.warning(f"Could not retrieve service capabilities: {e}")

        # Step 6: Set up mDNS and graceful shutdown
        zeroconf, service_info = None, None
        if config.server.mdns and config.server.mdns.enabled:
            zeroconf, service_info = setup_mdns(port, config.server.mdns)

        def handle_shutdown(signum, frame):
            logger.info("Shutdown signal received. Stopping server...")
            if zeroconf and service_info:
                logger.info("Unregistering mDNS service...")
                zeroconf.unregister_service(service_info)
                zeroconf.close()
            server.stop(grace=5.0)

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        logger.info("Server running. Press Ctrl+C to stop.")
        server.wait_for_termination()
        logger.info("Server shutdown complete.")

    except (ConfigError, FileNotFoundError) as e:
        logger.error(f"Service startup failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)


def main():
    """Main entry point for Lumen Hub server."""
    parser = argparse.ArgumentParser(
        description="Run Lumen Hub gRPC Service (multiple services in one process)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run hub service with default config settings
  python -m lumen.server --config config/lumen-hub.yaml

  # Override port
  python -m lumen.server --config config/lumen-hub.yaml --port 50052

  # Enable debug logging
  python -m lumen.server --config config/lumen-hub.yaml --log-level DEBUG

  # Validate config without starting server
  lumen-resources validate config/lumen-hub.yaml
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Port number (overrides config file setting)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set logging level
    setup_logging(args.log_level)

    # Start server
    serve(config_path=args.config, port_override=args.port)


if __name__ == "__main__":
    main()
