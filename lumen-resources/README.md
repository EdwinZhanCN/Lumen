# Lumen Resources

## Short Description

Lumen Resources is a unified model resource management tool for Lumen ML services. It provides configuration-driven downloading and validation of ML model resources from platforms like Hugging Face and ModelScope, with support for multiple runtimes and deployment modes. The tool ensures production-grade reliability through YAML configuration, JSON Schema validation, and Pydantic models.

## Installation

Install the package using pip:

```bash
pip install lumen-resources @ git+https://github.com/EdwinZhanCN/Lumen.git@main#subdirectory=lumen-resources
```

For development, clone the repository and install with optional dependencies:

```bash
git clone https://github.com/EdwinZhanCN/Lumen.git
cd lumen-resources
pip install -e ".[dev,config]"
```

## How to Use CLI

The CLI provides commands for downloading, validating, and listing model resources. The entry point is `lumen-resources`.

### Download Model Resources

Download models based on a YAML configuration file:

```bash
lumen-resources download path/to/config.yaml [--force]
```

- `--force`: Force re-download even if models are already cached.

This command loads the configuration, validates it, and downloads enabled models to the specified cache directory.

### Validate Configuration

Validate a YAML configuration file:

```bash
lumen-resources validate path/to/config.yaml [--strict]
```

- `--strict`: Use strict Pydantic validation (default: True). Use `--schema-only` for JSON Schema validation only.

Outputs validation results and configuration details if valid.

### Validate Model Info

Validate a `model_info.json` file:

```bash
lumen-resources validate-model-info path/to/model_info.json [--strict]
```

- `--strict`: Use strict Pydantic validation (default: True). Use `--schema-only` for JSON Schema validation only.

Outputs validation results and model details if valid.

### List Cached Models

List models in the cache directory:

```bash
lumen-resources list [cache_dir]
```

- `cache_dir`: Path to cache directory (default: `~/.lumen/`).

Displays available models, versions, runtimes, and contents.

## How to Use as a Python Package

Import and use the package programmatically for configuration loading, validation, and downloading.

### Basic Usage

```python
from lumen_resources import load_and_validate_config, Downloader

# Load and validate configuration
config = load_and_validate_config("path/to/config.yaml")

# Initialize downloader
downloader = Downloader(config, verbose=True)

# Download all enabled models
results = downloader.download_all(force=False)

# Check results
for model_type, result in results.items():
    if result.success:
        print(f"Downloaded: {model_type} to {result.model_path}")
    else:
        print(f"Failed: {model_type} - {result.error}")
```

### Key Classes and Functions

- `LumenServicesConfiguration`: Pydantic model for the full configuration.
- `load_and_validate_config(path)`: Load and validate a YAML config file.
- `load_and_validate_model_info(path)`: Load and validate a `model_info.json` file.
- `Downloader`: Class for downloading models.
- `DownloadResult`: Result object from downloads.
- Exceptions: `ConfigError`, `DownloadError`, `ValidationError`, etc.

### Advanced Example

```python
from lumen_resources import LumenServicesConfiguration, Runtime, Region

# Build config programmatically
config = LumenServicesConfiguration(
    metadata={
        "region": Region.US_WEST,
        "cache_dir": "/path/to/cache"
    },
    services={
        "my_service": {
            "enabled": True,
            "package": "my_package",
            "models": {
                "model_alias": {
                    "model": "org/model-name",
                    "runtime": Runtime.ONNX
                }
            }
        }
    }
)

downloader = Downloader(config)
results = downloader.download_all()
```

## Configuration Format

The YAML configuration includes metadata (region, cache dir), deployment settings (mode, services), server config (port, host, mDNS), and service definitions with models and runtimes.

See `docs/examples` for detailed schemas and examples.

## License

MIT License
