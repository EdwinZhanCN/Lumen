"""
Server startup module for Lumen Face Service.

This module provides the main server initialization and startup logic,
integrating with lumen-resources for configuration and model loading.
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
from pathlib import Path

import colorlog
import grpc
from lumen_resources import Downloader, DownloadResult, load_and_validate_config
from lumen_resources.lumen_config import LumenConfig, Mdns
from zeroconf import ServiceInfo, Zeroconf

from lumen_face.general_face.face_service import GeneralFaceService
from lumen_face.proto import ml_service_pb2_grpc as pb_rpc

logger = logging.getLogger(__name__)


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
    Set up mDNS general_face advertisement.

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
                + "other devices may not reach the general_face. "
                + "Set ADVERTISE_IP to a LAN IP."
            )

        # Build TXT record properties
        props = {
            "uuid": os.getenv("SERVICE_UUID", str(uuid.uuid4())),
            "status": os.getenv("SERVICE_STATUS", "ready"),
            "version": os.getenv("SERVICE_VERSION", "1.0.0"),
        }

        # Get general_face type and name from config (Mdns may be None or have different attribute names)
        service_type = (
            getattr(mdns_config, "type", None)
            or getattr(mdns_config, "service_type", None)
            or "_lumen._tcp.local."
        )
        instance_name = (
            getattr(mdns_config, "name", None)
            or getattr(mdns_config, "service_name", None)
            or "Face-Service"
        )
        full_name = f"{instance_name}.{service_type}"

        # Create general_face info
        service_info = ServiceInfo(
            type_=service_type,
            name=full_name,
            addresses=[socket.inet_aton(ip)],
            port=port,
            properties=props,
            server=f"{socket.gethostname()}.local.",
        )

        # Register general_face
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
    Initializes and starts the gRPC server based on a validated configuration.

    This server runner acts as an orchestrator:
    1. Loads and validates the master lumen_config.yaml.
    2. Determines which general_face to run based on the models defined in LumenConfig.
    3. Delegates the creation of the general_face instance to the appropriate Service.from_config factory.
    4. Initializes the general_face, which loads the ML models.
    5. Attaches the general_face to the gRPC server and starts listening.
    """
    try:
        # Step 1: Load and validate the main configuration file.
        config: LumenConfig = load_and_validate_config(config_path)

        # Step 1a: Verify and download all required model assets before proceeding.
        logger.info("Verifying and downloading model assets...")
        # verbose=False because we now have a much better result handler.
        downloader = Downloader(config, verbose=False)
        download_results = downloader.download_all()
        handle_download_results(download_results)

        # Ensure we are running in the correct deployment mode.
        if config.deployment.mode != "single":
            logger.error("This server is designed for 'single' deployment mode only.")
            sys.exit(1)

        # Step 2: Determine which general_face to instantiate based on model definitions.
        service_config = config.services.get("face")
        if not service_config or not service_config.models:
            logger.error("Configuration error: 'services.face.models' is not defined.")
            sys.exit(1)

        model_config = service_config.models.get("general")
        if not model_config or not model_config.model:
            logger.error(
                "Configuration error: 'services.face.models.general' is not defined."
            )
            sys.exit(1)

        service_instance = None
        service_display_name = "Unknown"
        cache_dir = Path(config.metadata.cache_dir).expanduser()
        backend_settings = service_config.backend_settings
        # Decision logic: choose the general_face based on which models are configured.
        if model_config:
            service_display_name = "General Face Recognition"
            logger.info(
                f"Configuration for '{service_display_name}' general_face identified."
            )

            service_instance = GeneralFaceService.from_config(
                model_config=model_config,
                cache_dir=cache_dir,
                backend_settings=backend_settings,
            )

        else:
            logger.error(
                "Configuration error: No valid model 'general' found under 'services.face'."
            )
            sys.exit(1)

        # Step 3: Initialize the chosen general_face (this loads the ML models).
        logger.info(f"Initializing {service_display_name} general_face...")
        service_instance.initialize()

        # Step 4: Set up and start the gRPC server.
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        # All our services implement the same gRPC interface.
        pb_rpc.add_InferenceServicer_to_server(service_instance, server)

        # Determine port: CLI override > config file > default.
        port = port_override or config.server.port or 50051
        listen_addr = f"[::]:{port}"
        server.add_insecure_port(listen_addr)
        server.start()
        logger.info(
            f"🚀 {service_display_name} general_face listening on {listen_addr}"
        )

        # Log general_face capabilities now that it's initialized.
        try:
            from google.protobuf import empty_pb2

            capabilities = service_instance.GetCapabilities(empty_pb2.Empty(), None)
            supported_tasks = [cap.name for cap in capabilities.tasks]
            logger.info(f"✓ Supported tasks: {', '.join(supported_tasks)}")
        except Exception as e:
            logger.warning(f"Could not retrieve general_face capabilities: {e}")

        # Step 5: Set up mDNS and graceful shutdown.
        zeroconf, service_info = None, None
        if config.server.mdns and config.server.mdns.enabled:
            zeroconf, service_info = setup_mdns(port, config.server.mdns)

        def handle_shutdown(signum, frame):
            logger.info("Shutdown signal received. Stopping server...")
            if zeroconf and service_info:
                logger.info("Unregistering mDNS general_face...")
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
        description="Run Lumen CLIP gRPC Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run General Face general_face with default config settings
  python server.py --config config/face_cn.yaml

  # Validate config without starting server
  lumen-resources validate config/face_cn.yaml
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
