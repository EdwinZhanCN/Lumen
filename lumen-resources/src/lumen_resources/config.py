"""
Configuration Parser and Validator (YAML)

@requires: Valid unified YAML configuration matching the schema
@returns: Structured ResourceConfig object (derived from unified config)
@errors: ConfigError
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

from .exceptions import ConfigError


class RuntimeType(str, Enum):
    """Supported runtime types."""

    TORCH = "torch"
    ONNX = "onnx"
    RKNN = "rknn"


class PlatformType(str, Enum):
    """Supported platforms."""

    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"


# Platform configuration mapping
PLATFORM_CONFIG = {
    "cn": {"type": PlatformType.MODELSCOPE, "owner": "LumilioPhotos"},
    "other": {"type": PlatformType.HUGGINGFACE, "owner": "Lumilio-Photos"},
}


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    model: str
    runtime: RuntimeType
    rknn_device: Optional[str] = None
    dataset: Optional[str] = None

    def get_runtime_patterns(self) -> list[str]:
        """
        Get file patterns to download based on runtime.

        Downloads runtime-specific model files plus all JSON configuration files
        (model_info.json, config.json, tokenizer.json, openclip.json, etc.)

        Returns:
            List of glob patterns for allow_patterns in snapshot_download
        """
        # Common files for all runtimes
        common_files = [
            "*.json"
        ]  # Includes model_info.json, config.json, tokenizer.json, openclip.json, etc.

        if self.runtime == RuntimeType.TORCH:
            return ["torch/*"] + common_files
        elif self.runtime == RuntimeType.ONNX:
            return ["onnx/*"] + common_files
        elif self.runtime == RuntimeType.RKNN:
            device = self.rknn_device or "rk3588"
            return [f"rknn/{device}/*"] + common_files
        return common_files

    def get_dataset_pattern(self) -> Optional[str]:
        """Get dataset file pattern if specified."""
        if self.dataset:
            return f"{self.dataset}.npz"
        return None


@dataclass
class ResourceConfig:
    """Complete reference for ResourceConfig.

    This dataclass represents the unified, validated runtime configuration used by
    the resource management utilities. It is produced by parsing a high-level
    YAML configuration file and provides convenient accessors and helpers for
    downstream operations (such as constructing repository IDs and determining
    which files to download per model runtime).

    Core behavior
    - Instances are created via ResourceConfig.from_yaml(config_path, only_services)
      which performs full validation and parsing of the YAML file.
    - The 'cache_dir' path is expanded (Path.expanduser) during parsing.
    - Model entries are keyed in the 'models' mapping by "service:alias".
    - If no enabled services with models are found, from_yaml will raise ConfigError.
    - Validation errors or YAML parsing issues raise ConfigError with a descriptive
      message. PyYAML is imported lazily and a helpful error is raised if absent.

    Attributes:
        region (str):
            One of "cn" or "other". Determines the platform mapping looked up in
            PLATFORM_CONFIG to derive platform_type and platform_owner.
        cache_dir (Path):
            Filesystem path where model artifacts and caches are stored. Converted
            to a pathlib.Path and expanded during parsing.
        models (Dict[str, ModelConfig]):
            Mapping from "service:alias" to ModelConfig objects. Each ModelConfig
            contains the repository name, runtime type, optional rknn_device, and
            optional dataset name.

    Properties and helpers:
        platform_type -> PlatformType:
            Derived property returning the PlatformType enum from PLATFORM_CONFIG
            for the configured region.
        platform_owner -> str:
            Derived property returning the platform owner/organization string from
            PLATFORM_CONFIG for the configured region.
        get_repo_id(model_name: str) -> str:
            Convenience method that returns the full repo id in the form
            "<platform_owner>/<model_name>".

    YAML structure expected (high level):
        metadata:
          region: "other"         # required, "cn" or "other"
          cache_dir: "~/lumen_cache"  # required
        services:
          clip:
            enabled: true
            models:
              default:
                model: "MobileCLIP-L-14"    # required
                runtime: "torch"            # required, one of "torch","onnx","rknn"
                rknn_device: "rk3588"       # optional, only for rknn runtime
                dataset: "some_dataset"     # optional

    Notes:
    - Model keys in the 'models' mapping are auto-constructed as "service:alias".
    - Use ModelConfig.get_runtime_patterns() to obtain glob patterns for
      downloading runtime-specific files as well as JSON metadata files.
    - Use ModelConfig.get_dataset_pattern() to obtain dataset artifact name
      (returns e.g. "<dataset>.npz" or None).
    - The class relies on helper static methods (_validate_structure and
      _parse_model_config) to perform structural checks and to convert raw model
      dicts into ModelConfig instances.

    Example usage:
        cfg = ResourceConfig.from_yaml(Path("resources.yaml"))
        print(cfg.cache_dir)
        for key, model in cfg.models.items():
            print(key, model.model, model.runtime)
            patterns = model.get_runtime_patterns()
            ds = model.get_dataset_pattern()
            repo_id = cfg.get_repo_id(model.model)

    Raises:
        ConfigError: on missing/invalid config file, invalid YAML,
                     or schema validation errors.
    """

    region: str
    cache_dir: Path
    models: Dict[str, ModelConfig]

    @property
    def platform_type(self) -> PlatformType:
        """Get platform type based on region."""
        return PLATFORM_CONFIG[self.region]["type"]

    @property
    def platform_owner(self) -> str:
        """Get platform organization/owner name."""
        return PLATFORM_CONFIG[self.region]["owner"]

    def get_repo_id(self, model_name: str) -> str:
        """
        Construct full repository ID.

        Args:
            model_name: Model repository name

        Returns:
            Full repo ID (e.g., "Lumilio-Photos/MobileCLIP-L-14")
        """
        return f"{self.platform_owner}/{model_name}"

    @classmethod
    def from_yaml(
        cls, config_path: Path, only_services: Optional[list[str]] = None
    ) -> "ResourceConfig":
        """
        Parse and validate configuration from unified YAML file.

        @requires: Valid YAML file at config_path
        @returns: Validated ResourceConfig instance derived from unified config
        @errors: ConfigError

        Args:
            config_path: Path to YAML configuration file
            only_services: Optional list of service names to filter models (e.g. ["clip"])

        Returns:
            Parsed and validated ResourceConfig

        Raises:
            ConfigError: If configuration is invalid
        """
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        # Lazy import to keep PyYAML optional for consumers who don't use YAML
        try:
            import yaml  # type: ignore
        except Exception:
            raise ConfigError(
                "YAML support requires PyYAML. Install with: pip install pyyaml "
                "or: pip install 'lumen-resources[config]'"
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ConfigError(f"Invalid YAML format: {e}")

        # Static validation - Phase 1
        cls._validate_structure(data)

        # Parse cache directory and region from metadata
        metadata = data["metadata"]
        cache_dir = Path(metadata["cache_dir"]).expanduser()
        region = metadata["region"]

        # Collect model configurations from enabled services
        services = data.get("services", {})
        models: Dict[str, ModelConfig] = {}

        for service_name, service_conf in services.items():
            # Service filter (single module mode)
            if only_services and service_name not in only_services:
                continue

            if not service_conf.get("enabled", False):
                continue

            service_models = service_conf.get("models", {}) or {}
            for alias, model_spec in service_models.items():
                key = f"{service_name}:{alias}"
                try:
                    models[key] = cls._parse_model_config(model_spec)
                except (KeyError, ValueError) as e:
                    raise ConfigError(
                        f"Invalid configuration for service '{service_name}', "
                        f"model alias '{alias}': {e}"
                    )

        if not models:
            raise ConfigError("No enabled services with models found in configuration")

        return cls(region=region, cache_dir=cache_dir, models=models)

    @staticmethod
    def _validate_structure(data: Dict[str, Any]) -> None:
        """
        Validate the basic structure of unified YAML configuration.

        Args:
            data: Raw configuration dictionary loaded from YAML

        Raises:
            ConfigError: If structure is invalid
        """
        # metadata checks
        if not isinstance(data, dict):
            raise ConfigError("Configuration root must be a mapping")

        if "metadata" not in data or not isinstance(data["metadata"], dict):
            raise ConfigError("Missing required field: metadata")

        metadata = data["metadata"]
        if "region" not in metadata:
            raise ConfigError("Missing required field: metadata.region")

        if metadata["region"] not in ["cn", "other"]:
            raise ConfigError(
                f"Invalid region: {metadata['region']}, must be 'cn' or 'other'"
            )

        if "cache_dir" not in metadata or not metadata["cache_dir"]:
            raise ConfigError("Missing required field: metadata.cache_dir")

        # services checks
        if "services" not in data or not isinstance(data["services"], dict):
            raise ConfigError("Missing required field: services")

        services = data["services"]
        # At least one enabled service with models
        has_enabled_with_models = False
        for svc_name, svc_conf in services.items():
            if not isinstance(svc_conf, dict):
                continue
            if not svc_conf.get("enabled", False):
                continue
            models = svc_conf.get("models", {})
            if isinstance(models, dict) and len(models) > 0:
                has_enabled_with_models = True
                # Light validation of each model spec
                for alias, spec in models.items():
                    if not isinstance(spec, dict):
                        raise ConfigError(
                            f"Invalid model spec for service '{svc_name}', alias '{alias}'"
                        )
                    if "model" not in spec:
                        raise ConfigError(
                            f"Missing 'model' in service '{svc_name}', alias '{alias}'"
                        )
                    if "runtime" not in spec:
                        raise ConfigError(
                            f"Missing 'runtime' in service '{svc_name}', alias '{alias}'"
                        )
        if not has_enabled_with_models:
            raise ConfigError(
                "At least one enabled service with non-empty models is required"
            )

    @staticmethod
    def _parse_model_config(model_data: Dict[str, Any]) -> ModelConfig:
        """
        Parse a single model configuration.

        Args:
            model_data: Raw model configuration dictionary

        Returns:
            Parsed ModelConfig instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If field values are invalid
        """
        if "model" not in model_data:
            raise KeyError("Missing required field: model")
        if "runtime" not in model_data:
            raise KeyError("Missing required field: runtime")

        try:
            runtime = RuntimeType(model_data["runtime"])
        except ValueError:
            raise ValueError(
                f"Invalid runtime: {model_data['runtime']}, "
                f"must be one of {[r.value for r in RuntimeType]}"
            )

        return ModelConfig(
            model=model_data["model"],
            runtime=runtime,
            rknn_device=model_data.get("rknn_device"),
            dataset=model_data.get("dataset"),
        )
