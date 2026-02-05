"""
Command-line interface for Lumen VLM Service.
"""

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version


def main():
    try:
        ver = version("lumen-vlm")
    except PackageNotFoundError:
        ver = "0.0.0"

    parser = argparse.ArgumentParser(
        description="Run Lumen VLM gRPC Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run VLM service with default config settings
  lumen-vlm --config config/vlm.yaml

  # Check version
  lumen-vlm --version
""",
    )

    # Note: required=False here to allow --version to work without config.
    # We manually check for config existence later.
    parser.add_argument(
        "--config",
        type=str,
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

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {ver}",
        help="Show program's version number and exit",
    )

    args = parser.parse_args()

    # Manually check for required config since we made it optional for --version
    if not args.config:
        parser.error("the following arguments are required: --config")

    # Lazy import server module to avoid heavy dependencies (ONNXRuntime, Transformers) on startup
    try:
        from lumen_vlm.server import serve, setup_logging
    except ImportError as e:
        print(f"Error importing server module: {e}", file=sys.stderr)
        sys.exit(1)

    # Set logging level
    setup_logging(args.log_level)

    # Start server
    serve(config_path=args.config, port_override=args.port)


if __name__ == "__main__":
    main()
