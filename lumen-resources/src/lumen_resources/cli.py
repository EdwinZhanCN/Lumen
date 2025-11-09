"""
Command Line Interface for Lumen Resources

Provides user-friendly commands for downloading and managing model resources.
"""

import argparse
import sys
from pathlib import Path

from .downloader import Downloader, DownloadResult
from .lumen_config_validator import ConfigValidator, load_and_validate_config
from .model_info_validator import (
    ModelInfoValidator,
    load_and_validate_model_info,
)


def print_banner():
    """Print welcome banner.

    Displays the Lumen Resources application banner with formatting
    to provide a professional command-line interface experience.
    """
    print("=" * 60)
    print("  Lumen Resources - Model Resource Manager")
    print("=" * 60)


def print_summary(results: dict[str, DownloadResult]) -> None:
    """Print download summary with results and statistics.

    Args:
        results: Dictionary mapping model type identifiers to DownloadResult objects.
            Contains information about download success status, file paths, and errors.
    """
    print("\n" + "=" * 60)
    print("📊 Download Summary")
    print("=" * 60)

    success_count = sum(1 for r in results.values() if r.success)
    total_count = len(results)

    for model_type, result in results.items():
        status = "✅" if result.success else "❌"
        print(f"{status} {model_type.upper()}")
        if result.success:
            print(f"   Path: {result.model_path}")
            if result.missing_files:
                print(f"   ⚠️  Missing: {', '.join(result.missing_files)}")
        else:
            print(f"   Error: {result.error}")

    print("\n" + "=" * 60)
    print(f"🎉 Completed: {success_count}/{total_count} successful")
    print("=" * 60)


def cmd_download(args: argparse.Namespace) -> None:
    """Handle download command for model resources.

    Downloads all enabled models from the configuration file, with support for
    forced re-downloading and detailed progress reporting. Validates configuration
    before downloading and provides a comprehensive summary of results.

    Args:
        args: Parsed command line arguments containing:
            - config: Path to configuration YAML file
            - force: Whether to force re-download even if models are cached

    Raises:
        SystemExit: If configuration validation fails or downloads encounter errors.
    """
    config_path = Path(args.config)

    try:
        # Load and validate configuration
        print("📋 Loading configuration...")
        config = load_and_validate_config(config_path)

        print(f"🌍 Region: {config.metadata.region.value}")
        print(f"📁 Cache directory: {config.metadata.cache_dir}")

        # Count enabled models
        enabled_models = []
        for service_name, service_config in config.services.items():
            if service_config.enabled:
                for alias in service_config.models.keys():
                    enabled_models.append(f"{service_name}:{alias}")
        print(f"🎯 Enabled models: {', '.join(enabled_models)}")

        # Download resources
        print("\n🚀 Starting download...")
        downloader = Downloader(config, verbose=True)
        results = downloader.download_all(force=args.force)

        # Print summary
        print_summary(results)

        # Exit with error if any downloads failed
        if not all(r.success for r in results.values()):
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def cmd_validate(args: argparse.Namespace) -> None:
    """Handle validate command for configuration files.

    Validates YAML configuration files against the Lumen configuration schema
    with options for strict Pydantic validation or flexible JSON Schema validation.
    Displays detailed configuration information when validation succeeds.

    Args:
        args: Parsed command line arguments containing:
            - config: Path to configuration YAML file
            - strict: Whether to use strict Pydantic validation

    Raises:
        SystemExit: If configuration validation fails.
    """
    config_path = Path(args.config)

    try:
        print("📋 Validating configuration...")

        # Use new Pydantic-based validator
        validator = ConfigValidator()
        is_valid, errors = validator.validate_file(config_path, strict=args.strict)

        if not is_valid:
            print("❌ Validation failed:\n")
            for error in errors:
                print(f"  • {error}")
            sys.exit(1)

        # Load the validated config
        config = load_and_validate_config(config_path)

        print("✅ Configuration is valid!")
        print(f"\n🌍 Region: {config.metadata.region.value}")
        print(f"📁 Cache directory: {config.metadata.cache_dir}")
        print(f"🚀 Deployment mode: {config.deployment.mode}")

        if config.deployment.mode == "single":
            print(f"   Service: {config.deployment.service}")
        else:
            services_list = [s.root for s in (config.deployment.services or [])]
            print(f"   Services: {', '.join(services_list)}")

        print("\n🌐 Server:")
        print(f"   Port: {config.server.port}")
        print(f"   Host: {config.server.host}")
        if config.server.mdns and config.server.mdns.enabled:
            print(f"   mDNS: {config.server.mdns.service_name}")

        print("\n📦 Services:")
        for service_name, service_config in config.services.items():
            status = "✅ enabled" if service_config.enabled else "⚪ disabled"
            print(f"  • {service_name} ({status})")
            if service_config.enabled:
                print(f"    Package: {service_config.package}")
                print(f"    Models: {', '.join(service_config.models.keys())}")
                for alias, model in service_config.models.items():
                    print(f"      - {alias}: {model.model} ({model.runtime.value})")
                    if model.dataset:
                        print(f"        Dataset: {model.dataset}")

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)


def cmd_validate_model_info(args: argparse.Namespace) -> None:
    """Handle validate-model-info command for model metadata files.

    Validates model_info.json files against the model information schema
    with options for strict Pydantic validation or flexible JSON Schema validation.
    Displays comprehensive model information when validation succeeds.

    Args:
        args: Parsed command line arguments containing:
            - model_info: Path to model_info.json file
            - strict: Whether to use strict Pydantic validation

    Raises:
        SystemExit: If model_info.json validation fails.
    """
    model_info_path = Path(args.model_info)

    try:
        print("📋 Validating model_info.json...")

        # Use ModelInfoValidator
        validator = ModelInfoValidator()
        is_valid, errors = validator.validate_file(model_info_path, strict=args.strict)

        if not is_valid:
            print("❌ Validation failed:\n")
            for error in errors:
                print(f"  • {error}")
            sys.exit(1)

        # Load the validated model info
        model_info = load_and_validate_model_info(model_info_path)

        print("✅ Model info is valid!")
        print(f"\n📦 Model: {model_info.name}")
        print(f"   Version: {model_info.version}")
        print(f"   Type: {model_info.model_type}")
        print(f"   Embedding dimension: {model_info.embedding_dim}")

        print("\n📥 Source:")
        print(f"   Format: {model_info.source.format.value}")
        print(f"   Repository: {model_info.source.repo_id}")

        print("\n🔧 Runtimes:")
        for runtime_name, runtime_config in model_info.runtimes.items():
            status = "✅ available" if runtime_config.available else "⚪ not available"
            print(f"  • {runtime_name} ({status})")
            if runtime_config.available and runtime_config.files:
                if isinstance(runtime_config.files, list):
                    print(f"    Files: {len(runtime_config.files)} file(s)")
                else:
                    total_files = sum(
                        len(files) for files in runtime_config.files.values()
                    )
                    devices_count = len(runtime_config.files)
                    print(
                        f"    Files: {total_files} file(s) across {devices_count} device(s)"
                    )
            if runtime_config.devices:
                print(f"    Devices: {', '.join(runtime_config.devices)}")

        if model_info.datasets:
            print("\n📊 Datasets:")
            for dataset_name, dataset_file in model_info.datasets.items():
                print(f"  • {dataset_name}: {dataset_file}")

        if model_info.metadata:
            print("\n📝 Metadata:")
            if model_info.metadata.license:
                print(f"   License: {model_info.metadata.license}")
            if model_info.metadata.author:
                print(f"   Author: {model_info.metadata.author}")
            if model_info.metadata.created_at:
                print(f"   Created: {model_info.metadata.created_at}")
            if model_info.metadata.tags:
                print(f"   Tags: {', '.join(model_info.metadata.tags)}")

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    """Handle list command for cached models.

    Lists all models currently cached in the specified cache directory,
    showing model information including version, available runtimes, and
    file contents when model_info.json files are available.

    Args:
        args: Parsed command line arguments containing:
            - cache_dir: Path to cache directory (defaults to ~/.lumen/)
    """
    cache_dir = Path(args.cache_dir).expanduser()
    models_dir = cache_dir / "models"

    if not models_dir.exists():
        print(f"No models found in {cache_dir}")
        return

    print(f"📦 Models in {cache_dir}:")
    print()

    model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])

    if not model_dirs:
        print("  (empty)")
        return

    for model_dir in model_dirs:
        print(f"  📁 {model_dir.name}")

        # Check for model_info.json
        info_file = model_dir / "model_info.json"
        if info_file.exists():
            import json

            try:
                with open(info_file, "r") as f:
                    info = json.load(f)
                print(f"     Version: {info.get('version', 'unknown')}")
                runtimes = info.get("runtimes", {})
                available_runtimes = [
                    r for r, data in runtimes.items() if data.get("available")
                ]
                if available_runtimes:
                    print(f"     Runtimes: {', '.join(available_runtimes)}")
            except Exception:
                pass

        # List subdirectories
        subdirs = [d.name for d in model_dir.iterdir() if d.is_dir()]
        if subdirs:
            print(f"     Contents: {', '.join(subdirs)}")

        print()


def main() -> None:
    """Main CLI entry point.

    Sets up the argument parser with subcommands for different operations
    (download, validate, validate-model-info, list) and dispatches to the
    appropriate handler functions. Handles help display and error cases.

    Raises:
        SystemExit: If no command is provided or if a command handler exits with an error.
    """
    parser = argparse.ArgumentParser(
        prog="lumen-resources",
        description="Lumen Resources - Model Resource Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser(
        "download", help="Download model resources from configuration"
    )
    _ = download_parser.add_argument("config", help="Path to configuration YAML file")
    _ = download_parser.add_argument(
        "--force", action="store_true", help="Force re-download even if cached"
    )
    download_parser.set_defaults(func=cmd_download)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate configuration file"
    )
    _ = validate_parser.add_argument("config", help="Path to configuration YAML file")
    _ = validate_parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Use strict Pydantic validation (default: True)",
    )
    _ = validate_parser.add_argument(
        "--schema-only",
        action="store_false",
        dest="strict",
        help="Use JSON Schema validation only (less strict)",
    )
    validate_parser.set_defaults(func=cmd_validate)

    # Validate model info command
    validate_model_parser = subparsers.add_parser(
        "validate-model-info", help="Validate model_info.json file"
    )
    _ = validate_model_parser.add_argument(
        "model_info", help="Path to model_info.json file"
    )
    _ = validate_model_parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Use strict Pydantic validation (default: True)",
    )
    _ = validate_model_parser.add_argument(
        "--schema-only",
        action="store_false",
        dest="strict",
        help="Use JSON Schema validation only (less strict)",
    )
    validate_model_parser.set_defaults(func=cmd_validate_model_info)

    # List command
    list_parser = subparsers.add_parser("list", help="List cached models")
    _ = list_parser.add_argument(
        "cache_dir", nargs="?", default="~/.lumen/", help="Cache directory path"
    )
    list_parser.set_defaults(func=cmd_list)

    args: argparse.Namespace = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    print_banner()
    print()

    args.func(args)


if __name__ == "__main__":
    main()
