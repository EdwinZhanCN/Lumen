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
from typing import Any, Dict, Optional

import grpc
import yaml

# Try to import zeroconf for mDNS support
try:
    from zeroconf import ServiceInfo, Zeroconf

    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid."""

    pass


class ServerConfig:
    """Parsed and validated server configuration."""

    def __init__(self, config_dict: dict):
        """
        Initialize ServerConfig from configuration dictionary.

        Args:
            config_dict: Parsed YAML configuration

        Raises:
            ConfigError: If configuration is invalid
        """
        self.raw = config_dict
        self._validate()

    def _validate(self):
        """Validate configuration structure and enforce single-service rule."""
        # Check metadata
        if "metadata" not in self.raw:
            raise ConfigError("Configuration must contain 'metadata' section")

        metadata = self.raw["metadata"]
        if "version" not in metadata:
            raise ConfigError("metadata.version is required")
        if "region" not in metadata:
            raise ConfigError("metadata.region is required")

        # Check services
        if "services" not in self.raw:
            raise ConfigError("Configuration must contain 'services' section")

        services = self.raw["services"]
        enabled_services = [
            name for name, cfg in services.items() if cfg.get("enabled", False)
        ]

        if len(enabled_services) == 0:
            raise ConfigError("No service is enabled in configuration")

        if len(enabled_services) > 1:
            raise ConfigError(
                f"Multiple services enabled: {enabled_services}. "
                "Only one service can be enabled per process. "
                "For multi-service deployment, run multiple processes with separate configs."
            )

        self.service_name = enabled_services[0]
        self.service_config = services[self.service_name]

        # Validate service configuration
        required_fields = ["package", "import"]
        for field in required_fields:
            if field not in self.service_config:
                raise ConfigError(
                    f"Service '{self.service_name}' missing required field '{field}'"
                )

        import_cfg = self.service_config["import"]
        if "registry_class" not in import_cfg:
            raise ConfigError(
                f"Service '{self.service_name}' missing 'import.registry_class'"
            )
        if "add_to_server" not in import_cfg:
            raise ConfigError(
                f"Service '{self.service_name}' missing 'import.add_to_server'"
            )

    @property
    def cache_dir(self) -> Path:
        """Get cache directory path."""
        cache_path = self.raw["metadata"].get("cache_dir", "~/.lumen/models")
        return Path(cache_path).expanduser()

    @property
    def region(self) -> str:
        """Get region setting."""
        return self.raw["metadata"]["region"]

    @property
    def port(self) -> int:
        """Get server port from config or default."""
        return self.service_config.get("server", {}).get("port", 50051)

    @property
    def mdns_config(self) -> Optional[Dict[str, Any]]:
        """Get mDNS configuration if enabled."""
        mdns = self.service_config.get("server", {}).get("mdns", {})
        if mdns.get("enabled", False):
            return mdns
        return None

    def get_env_vars(self) -> Dict[str, str]:
        """Get environment variables from config."""
        env = self.service_config.get("env", {})
        return {k: str(v) for k, v in env.items()}


def load_config(config_path: str) -> ServerConfig:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated ServerConfig instance

    Raises:
        ConfigError: If configuration is invalid
        FileNotFoundError: If config file doesn't exist
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML: {e}")

    return ServerConfig(config_dict)


def import_service_class(service_config: dict):
    """
    Dynamically import service class from configuration.

    Args:
        service_config: Service configuration dictionary

    Returns:
        Tuple of (ServiceClass, add_to_server_function)

    Raises:
        ConfigError: If import fails
    """
    import_cfg = service_config["import"]
    registry_class = import_cfg["registry_class"]
    add_to_server = import_cfg["add_to_server"]

    try:
        # Import service class (e.g., "image_classification.clip_service.CLIPService")
        module_path, class_name = registry_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        service_class = getattr(module, class_name)
        logger.info(f"✓ Imported service class: {registry_class}")
    except (ImportError, AttributeError) as e:
        raise ConfigError(f"Failed to import service class '{registry_class}': {e}")

    try:
        # Import add_to_server function (e.g., "ml_service_pb2_grpc.add_InferenceServicer_to_server")
        module_path, func_name = add_to_server.rsplit(".", 1)
        module = importlib.import_module(module_path)
        add_func = getattr(module, func_name)
        logger.info(f"✓ Imported gRPC registration function: {add_to_server}")
    except (ImportError, AttributeError) as e:
        raise ConfigError(
            f"Failed to import add_to_server function '{add_to_server}': {e}"
        )

    return service_class, add_func


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


def serve(config_path: str, port_override: Optional[int] = None) -> None:
    """
    Initialize and start the gRPC server.

    Args:
        config_path: Path to YAML configuration file
        port_override: Optional port override from command line

    Raises:
        SystemExit: On fatal errors
    """
    try:
        # Load and validate configuration
        config = load_config(config_path)
        logger.info(f"✓ Configuration validated: {config.service_name} service enabled")

        # Set environment variables from config
        env_vars = config.get_env_vars()
        if env_vars:
            logger.info(f"Setting environment variables: {list(env_vars.keys())}")
            os.environ.update(env_vars)

        # Import service class and registration function
        service_class, add_to_server_func = import_service_class(config.service_config)

        # Create gRPC server
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        # Initialize service from config
        logger.info(f"Initializing {config.service_name} service...")
        try:
            service_instance = service_class.from_config(
                config=config.service_config, cache_dir=config.cache_dir
            )
        except Exception as e:
            logger.exception(f"Failed to create service from config: {e}")
            sys.exit(1)

        # Register service with gRPC server
        add_to_server_func(service_instance, server)
        logger.info("✓ Service registered with gRPC server")

        # Initialize the service (load models, etc.)
        try:
            logger.info("Loading models and resources... This may take a moment.")
            service_instance.initialize()
            logger.info("✓ Service initialized successfully")
        except Exception as e:
            logger.exception(f"Fatal error during service initialization: {e}")
            sys.exit(1)

        # Get capabilities for logging
        try:
            from google.protobuf import empty_pb2

            capabilities = service_instance.GetCapabilities(empty_pb2.Empty(), None)
            supported_tasks = [cap.task_name for cap in capabilities.capabilities]
            logger.info(f"✓ Supported tasks: {', '.join(supported_tasks)}")
        except Exception as e:
            logger.warning(f"Could not retrieve capabilities: {e}")

        # Start server
        port = port_override if port_override is not None else config.port
        listen_addr = f"[::]:{port}"
        server.add_insecure_port(listen_addr)
        server.start()
        logger.info(f"🚀 {config.service_name} service listening on {listen_addr}")

        # Set up mDNS advertisement
        zeroconf = None
        service_info = None
        mdns_config = config.mdns_config
        if mdns_config:
            zeroconf, service_info = setup_mdns(port, mdns_config)

        # Set up graceful shutdown
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
