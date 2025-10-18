"""
gRPC Server Runner for Lumen CLIP Services

This script initializes and runs CLIP-based services (CLIP, BioCLIP, or Unified)
based on YAML configuration files. It supports:
- Dynamic service loading from config
- Single-service-per-process enforcement
- mDNS advertisement
- Graceful shutdown
- Resource validation before startup

Usage:
    python server.py --config config/clip_only.yaml
    python server.py --config config/bioclip_only.yaml --port 50052
"""

import argparse
import importlib
import logging
import os
import signal
import socket
import sys
import uuid
from concurrent import futures
from pathlib import Path
from typing import Optional
from lumen_resources import LumenServicesConfiguration, load_and_validate_config
import colorlog

import grpc
from lumen_resources.lumen_config import Services

# Try to import zeroconf for mDNS support
try:
    from zeroconf import ServiceInfo, Zeroconf

    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False

# --- Logging Configuration ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
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
logger.addHandler(handler)


class ConfigError(Exception):
    """Raised when configuration is invalid."""

    pass


# Simplified helpers that work with LumenServicesConfiguration from lumen_resources
def import_from_string(dotted: str, package: Optional[str] = None):
    """
    Import attribute (class or function) by dotted path. If direct import fails,
    try to prefix the module path with the service `package`.
    Raises ConfigError on failure (consistent with existing error handling).
    """
    try:
        module_path, attr_name = dotted.rsplit(".", 1)
    except ValueError:
        raise ConfigError(f"Invalid import string: '{dotted}'")

    # Try direct import first
    last_exc = None
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        last_exc = e
        # If package provided, try with package prefix
        if package:
            try:
                module = importlib.import_module(f"{package}.{module_path}")
                last_exc = None
            except Exception as e2:
                last_exc = e2
                module = None
        else:
            module = None

    if module is None:
        raise ConfigError(f"Failed to import module for '{dotted}': {last_exc}")

    try:
        return getattr(module, attr_name)
    except AttributeError as e:
        raise ConfigError(
            f"Module '{module.__name__}' has no attribute '{attr_name}': {e}"
        )


def select_enabled_service(cfg: LumenServicesConfiguration) -> tuple[str, Services]:
    """
    From a LumenServicesConfiguration instance, pick the single enabled service.
    Returns (service_name, services_model).
    """
    enabled = [
        name for name, svc in cfg.services.items() if getattr(svc, "enabled", False)
    ]
    if len(enabled) == 0:
        raise ConfigError("No service is enabled in configuration")
    if len(enabled) > 1:
        raise ConfigError(
            f"Multiple services enabled: {enabled}. Only one service is supported per process."
        )
    name = enabled[0]
    service_config = cfg.services[name]
    return name, service_config


def setup_mdns(port: int, mdns_config: dict) -> tuple:
    """
    Set up mDNS service advertisement.

    Args:
        port: Service port number
        mdns_config: mDNS configuration dictionary

    Returns:
        Tuple of (zeroconf, service_info) or (None, None) if setup fails
    """
    if not ZEROCONF_AVAILABLE:
        logger.warning("Zeroconf not available; skipping mDNS advertisement")
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
        service_type = mdns_config.get("type", "_lumen-clip._tcp.local.")
        instance_name = mdns_config.get("name", "CLIP-Service")
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


def serve(config_path: str, port_override: int | None = None) -> None:
    """
    Initialize and start the gRPC server.

    Args:
        config_path: Path to YAML configuration file
        port_override: Optional port override from command line

    Raises:
        SystemExit: On fatal errors
    """
    try:
        # Load and validate configuration (returns Pydantic model)
        config = load_and_validate_config(config_path)
        # deployment mode
        deployment_mode: str = config.deployment.mode

        # Top-level server / mDNS config
        server_cfg = getattr(config, "server", None)
        mdns_cfg = getattr(server_cfg, "mdns", None) if server_cfg is not None else None
        mdns_service_name = (
            getattr(mdns_cfg, "service_name", None) if mdns_cfg is not None else "clip"
        )
        mdns_enabled = (
            getattr(mdns_cfg, "enabled", False) if mdns_cfg is not None else False
        )

        # Select the single enabled service definition
        service_name, services_config = select_enabled_service(config)
        service_package = getattr(services_config, "package", None)

        # import_ contains registry_class and add_to_server
        service_import = services_config.import_
        registry_class_path = service_import.registry_class
        add_to_server_path = service_import.add_to_server

        logger.info(
            f"Configuration validated: {mdns_service_name} service enabled (service={service_name})"
        )
        if mdns_enabled:
            logger.info("mDNS service enabled")
        else:
            logger.warning("mDNS service disabled")
        logger.info(f"Deployment mode: {deployment_mode}")

        # Import service class and registration function (with package fallback)
        service_class = import_from_string(registry_class_path, package=service_package)
        add_to_server_func = import_from_string(
            add_to_server_path, package=service_package
        )
        logger.info(f"✓ Imported service class: {registry_class_path}")
        logger.info(f"✓ Imported gRPC registration function: {add_to_server_path}")

        # Create gRPC server
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        # Prepare cache_dir from metadata
        cache_dir = Path(config.metadata.cache_dir).expanduser()

        # Initialize service instance from config
        try:
            # pass service config as dict with aliases to preserve 'import' alias if needed
            svc_config_dict = services_config.model_dump(by_alias=True)
            logger.info(f"Initializing service '{service_name}' from config...")
            service_instance = service_class.from_config(
                config=svc_config_dict, cache_dir=cache_dir
            )
        except Exception as e:
            logger.exception(f"Failed to create service from config: {e}")
            sys.exit(1)

        # Register service with gRPC server
        try:
            add_to_server_func(service_instance, server)
            logger.info("✓ Service registered with gRPC server")
        except Exception as e:
            logger.exception(f"Failed to register service with gRPC server: {e}")
            sys.exit(1)

        # Initialize the service (load models, etc.)
        try:
            logger.info("Loading models and resources... This may take a moment.")
            service_instance.initialize()
            logger.info("✓ Service initialized successfully")
        except Exception as e:
            logger.exception(f"Fatal error during service initialization: {e}")
            sys.exit(1)

        # Get capabilities for logging (best-effort)
        try:
            from google.protobuf import empty_pb2

            capabilities = service_instance.GetCapabilities(empty_pb2.Empty(), None)
            supported_tasks = [cap.task_name for cap in capabilities.capabilities]
            logger.info(f"✓ Supported tasks: {', '.join(supported_tasks)}")
        except Exception as e:
            logger.warning(f"Could not retrieve capabilities: {e}")

        # Decide port: CLI override > top-level server.port > default 50051
        port = (
            port_override
            if port_override is not None
            else (server_cfg.port if server_cfg is not None else 50051)
        )
        listen_addr = f"[::]:{port}"
        server.add_insecure_port(listen_addr)
        server.start()
        logger.info(f"🚀 {service_name} service listening on {listen_addr}")

        # Set up mDNS advertisement if enabled
        zeroconf = None
        service_info = None
        if mdns_cfg and getattr(mdns_cfg, "enabled", False):
            # convert mdns model to dict for setup_mdns (which expects a mapping)
            zeroconf, service_info = setup_mdns(port, mdns_cfg.dict(by_alias=True))

        # Graceful shutdown handler
        def handle_shutdown(signum, frame):
            logger.info("Shutdown signal received. Stopping server...")
            try:
                if zeroconf and service_info:
                    zeroconf.unregister_service(service_info)
                    zeroconf.close()
                    logger.info("✓ mDNS service unregistered")
            except Exception as e:
                logger.warning(f"mDNS unregistration failed: {e}")

            server.stop(grace=5.0)

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        # Wait for termination
        logger.info("Server running. Press Ctrl+C to stop.")
        server.wait_for_termination()
        logger.info("Server shutdown complete.")

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Lumen CLIP gRPC Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CLIP service with default config settings
  python server.py --config config/clip_only.yaml

  # Run BioCLIP service on custom port
  python server.py --config config/bioclip_only.yaml --port 50052

  # Run unified service
  python server.py --config config/unified_service.yaml

  # Validate config without starting server
  lumen-resources validate config/clip_only.yaml
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
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Start server
    serve(config_path=args.config, port_override=args.port)


if __name__ == "__main__":
    main()
