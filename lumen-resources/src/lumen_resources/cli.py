"""
Command Line Interface for Lumen Resources

Provides user-friendly commands for downloading and managing model resources.
"""

import argparse
import sys
from pathlib import Path

from . import ResourceConfig, Downloader, ResourceError


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("  Lumen Resources - Model Resource Manager")
    print("=" * 60)


def print_summary(results: dict):
    """Print download summary."""
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


def cmd_download(args):
    """Handle download command."""
    config_path = Path(args.config)

    try:
        # Parse configuration
        print("📋 Parsing configuration...")
        config = ResourceConfig.from_yaml(config_path)

        print(f"🌍 Platform: {config.platform_type.value}")
        print(f"📁 Cache directory: {config.cache_dir}")
        print(f"🎯 Enabled models: {', '.join(config.models.keys())}")

        # Download resources
        print("\n🚀 Starting download...")
        downloader = Downloader(config, verbose=True)
        results = downloader.download_all(force=args.force)

        # Print summary
        print_summary(results)

        # Exit with error if any downloads failed
        if not all(r.success for r in results.values()):
            sys.exit(1)

    except ResourceError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def cmd_validate(args):
    """Handle validate command."""
    config_path = Path(args.config)

    try:
        print("📋 Validating configuration...")
        config = ResourceConfig.from_yaml(config_path)

        print("✅ Configuration is valid!")
        print(f"\n🌍 Platform: {config.platform_type.value}")
        print(f"📁 Cache directory: {config.cache_dir}")
        print(f"\n📦 Configured models:")

        for model_type, model_config in config.models.items():
            print(f"  • {model_type.upper()}")
            print(f"    Model: {model_config.model}")
            print(f"    Runtime: {model_config.runtime.value}")
            if model_config.rknn_device:
                print(f"    Device: {model_config.rknn_device}")
            if model_config.dataset:
                print(f"    Dataset: {model_config.dataset}")

    except ResourceError as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)


def cmd_list(args):
    """Handle list command."""
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


def main():
    """Main CLI entry point."""
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
    download_parser.add_argument("config", help="Path to configuration YAML file")
    download_parser.add_argument(
        "--force", action="store_true", help="Force re-download even if cached"
    )
    download_parser.set_defaults(func=cmd_download)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate configuration file"
    )
    validate_parser.add_argument("config", help="Path to configuration YAML file")
    validate_parser.set_defaults(func=cmd_validate)

    # List command
    list_parser = subparsers.add_parser("list", help="List cached models")
    list_parser.add_argument(
        "cache_dir", nargs="?", default="~/.lumen/", help="Cache directory path"
    )
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    print_banner()
    print()

    args.func(args)


if __name__ == "__main__":
    main()
