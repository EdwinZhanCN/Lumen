"""
Hub server startup module for Lumen Application.

This module provides a unified gRPC server that can host multiple AI services
(OCR, CLIP, Face, VLM) in a single process. It adapts the proven server architecture
from individual Lumen service packages to support hub deployment mode.

Key features:
- Load and validate hub configuration
- Download and verify all required model assets
- Initialize multiple ML services via AppService
- Route inference requests via HubRouter
- Support mDNS service discovery
- Graceful shutdown with signal handling
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
from typing import cast

import colorlog
import grpc
from google.protobuf import empty_pb2
from lumen_resources import Downloader, DownloadResult, load_and_validate_config
from lumen_resources.lumen_config import LumenConfig, Mdns
from zeroconf import ServiceInfo, Zeroconf

from lumen_app.core.service import AppService
from lumen_app.proto import ml_service_pb2_grpc

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid."""

    pass


class _StartupServicerContext:
    """Minimal ServicerContext stub for startup-time capability probing."""

    def abort(self, code, details):
        raise RuntimeError(f"Capability probe aborted ({code}): {details}")

    def abort_with_status(self, status):
        raise RuntimeError(f"Capability probe aborted: {status}")

    def set_code(self, code):
        pass

    def set_details(self, details):
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
        # Include the logger name in cyan and keep a fixed-width field to
        # improve alignment of log source labels.
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


def setup_mdns(
    port: int, mdns_config: Mdns | None
) -> tuple[Zeroconf | None, ServiceInfo | None]:
    """
    Set up mDNS advertisement.

    Args:
        port: Service port number
        mdns_config: mDNS configuration dictionary

    Returns:
        Tuple of (zeroconf, service_info) or (None, None) if setup fails
    """

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
                + "other devices may not reach the service. "
                + "Set ADVERTISE_IP to a LAN IP."
            )

        # Build TXT record properties
        props = {
            "uuid": os.getenv("SERVICE_UUID", str(uuid.uuid4())),
            "status": os.getenv("SERVICE_STATUS", "ready"),
            "version": os.getenv("SERVICE_VERSION", "1.0.0"),
        }

        # Get service type and name from config (Mdns may be None or have different attribute names)
        service_type = (
            getattr(mdns_config, "type", None)
            or getattr(mdns_config, "service_type", None)
            or "_lumen._tcp.local."
        )
        instance_name = (
            getattr(mdns_config, "name", None)
            or getattr(mdns_config, "service_name", None)
            or "Lumen-Hub"  # Changed from service-specific names
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

    This server runner acts as an orchestrator:
    1. Loads and validates the master lumen_config.yaml.
    2. Downloads all required model assets.
    3. Initializes all enabled ML services via AppService.
    4. Starts gRPC server with HubRouter for request routing.
    5. Sets up mDNS service discovery (optional).
    6. Handles graceful shutdown.
    """
    try:
        # Step 1: Load and validate the main configuration file.
        config: LumenConfig = load_and_validate_config(config_path)

        # Step 1a: Verify and download all required model assets before proceeding.
        logger.info("Verifying and downloading model assets...")
        downloader = Downloader(config, verbose=False)
        download_results = downloader.download_all()
        handle_download_results(download_results)

        # Ensure we are running in the correct deployment mode.
        if config.deployment.mode != "hub":
            logger.error("This server is designed for 'hub' deployment mode only.")
            sys.exit(1)

        # Step 2: Initialize all enabled services via AppService
        logger.info("Initializing Lumen Hub services...")
        app_service = AppService.from_app_config(config)
        logger.info(f"Loaded {len(app_service.services)} services")

        # Step 3: Set up and start the gRPC server.
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[("grpc.so_reuseport", 0)],
        )

        # Register HubRouter as the gRPC servicer
        ml_service_pb2_grpc.add_InferenceServicer_to_server(app_service.router, server)

        # Determine port: CLI override > config file > default.
        preferred_port = port_override or config.server.port or 50051
        requested_addr = f"[::]:{preferred_port}"
        try:
            bound_port = server.add_insecure_port(requested_addr)
        except RuntimeError as exc:
            logger.warning(
                f"Port {preferred_port} bind raised {exc}; requesting OS-assigned port."
            )
            bound_port = 0

        if bound_port == 0:
            try:
                bound_port = server.add_insecure_port("[::]:0")
            except RuntimeError as exc:
                logger.error(f"Unable to bind gRPC server to any port: {exc}")
                sys.exit(1)

        if bound_port == 0:
            logger.error("Unable to bind gRPC server to any port.")
            sys.exit(1)

        port = bound_port
        listen_addr = f"[::]:{port}"
        server.start()
        logger.info(f"🚀 Lumen Hub server listening on {listen_addr}")

        # Log service capabilities now that all services are initialized.
        try:
            startup_context = _StartupServicerContext()
            capabilities = app_service.router.GetCapabilities(
                empty_pb2.Empty(), cast(grpc.ServicerContext, startup_context)
            )
            supported_tasks = [task.name for task in capabilities.tasks]
            logger.info(f"✓ Supported tasks: {', '.join(supported_tasks)}")
        except Exception as e:
            logger.warning(f"Could not retrieve service capabilities: {e}")

        # Step 4: Set up mDNS and graceful shutdown.
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
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Lumen Hub gRPC Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run hub server with default config settings
  python -m lumen_app.server --config config/hub.yaml

  # Run hub server on custom port
  python -m lumen_app.server --config config/hub.yaml --port 50052

  # Validate config without starting server
  lumen-resources validate config/hub.yaml
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
