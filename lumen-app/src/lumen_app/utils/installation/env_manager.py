"""Python environment manager for creating and managing micromamba environments."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class PythonEnvManager:
    """Manages Python environment creation and pip operations."""

    ENV_NAME = "lumen_env"

    def __init__(self, cache_dir: Path | str, micromamba_exe: Path | str):
        """Initialize environment manager.

        Args:
            cache_dir: Cache directory
            micromamba_exe: Path to micromamba executable
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.micromamba_exe = Path(micromamba_exe)
        self.mamba_configs_dir = Path(__file__).parent.parent / "mamba"
        self.target_name = "micromamba"

        logger.debug(
            f"[PythonEnvManager] Initialized with cache_dir={self.cache_dir}, "
            f"micromamba_exe={self.micromamba_exe}"
        )

    def create_env(self, yaml_config: str, python_version: str = "3.11") -> Path:
        """Create Python environment using micromamba.

        Args:
            yaml_config: Mamba yaml config identifier (e.g., "cuda", "default")
            python_version: Python version (e.g., "3.11")

        Returns:
            Path to created environment

        Raises:
            Exception: If environment creation fails
        """
        logger.info(
            f"[PythonEnvManager] Creating environment '{self.ENV_NAME}' "
            f"with yaml_config={yaml_config}, python={python_version}"
        )

        env_path = self.get_env_path()

        # Check if environment already exists
        if env_path.exists():
            logger.info(f"[PythonEnvManager] Environment already exists: {env_path}")
            return env_path

        # Ensure parent directory exists
        env_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine which yaml file to use
        yaml_file = self.mamba_configs_dir / f"{yaml_config}.yaml"

        if yaml_config == "default" or not yaml_file.exists():
            # Create basic environment
            cmd = [
                str(self.micromamba_exe),
                "create",
                "-p",  # Use path instead of name
                str(env_path),
                f"python={python_version}",
                "-y",
            ]
            logger.debug(
                f"[PythonEnvManager] Creating basic environment: {' '.join(cmd)}"
            )
        else:
            # Use yaml config file
            if not yaml_file.exists():
                raise Exception(f"YAML config file not found: {yaml_file}")

            cmd = [
                str(self.micromamba_exe),
                "install",
                "-y",
                "-p",  # Use path instead of name
                str(env_path),
                "-f",
                str(yaml_file),
            ]
            logger.debug(
                f"[PythonEnvManager] Creating environment from yaml: {' '.join(cmd)}"
            )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                logger.error(
                    f"[PythonEnvManager] Environment creation failed: {error_msg}"
                )
                raise Exception(f"Failed to create environment: {error_msg}")

            logger.info(f"[PythonEnvManager] Environment created: {env_path}")
            return env_path

        except subprocess.TimeoutExpired:
            logger.error("[PythonEnvManager] Environment creation timed out")
            raise Exception("Environment creation timed out")

    def get_env_path(self) -> Path:
        """Get the path to the lumen_env environment.

        Returns:
            Path to environment directory
        """
        env_path = self.cache_dir / self.target_name / "envs" / self.ENV_NAME
        return env_path

    def run_pip(self, *args: str) -> subprocess.CompletedProcess:
        """Run pip command in the lumen_env environment.

        Args:
            *args: Pip command arguments (e.g., "install", "package")

        Returns:
            Completed process result

        Raises:
            Exception: If command fails
        """
        env_path = self.get_env_path()
        cmd = [
            str(self.micromamba_exe),
            "run",
            "-p",  # Use path instead of name
            str(env_path),
            "pip",
            *args,
        ]

        logger.debug(f"[PythonEnvManager] Running pip: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes
            )
            return result
        except subprocess.TimeoutExpired:
            logger.error("[PythonEnvManager] Pip command timed out")
            raise Exception("Pip command timed out")

    def env_exists(self) -> bool:
        """Check if lumen_env exists.

        Returns:
            True if environment exists
        """
        return self.get_env_path().exists()
