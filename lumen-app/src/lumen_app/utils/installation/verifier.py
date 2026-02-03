"""Installation verifier for validating Lumen package installations."""

import logging
import subprocess

from .env_manager import PythonEnvManager

logger = logging.getLogger(__name__)


class InstallationVerifier:
    """Verifies Lumen package installations."""

    def __init__(self, env_manager: PythonEnvManager):
        """Initialize verifier.

        Args:
            env_manager: Python environment manager
        """
        self.env_manager = env_manager
        logger.debug("[InstallationVerifier] Initialized")

    def verify_imports(self, packages: list[str]) -> dict[str, bool]:
        """Verify that packages can be imported.

        Args:
            packages: List of package names (e.g., ["lumen_ocr", "lumen_clip"])

        Returns:
            Dictionary mapping package names to import success status
        """
        logger.info(f"[InstallationVerifier] Verifying imports for: {packages}")
        results = {}

        for package in packages:
            # Convert lumen-ocr to lumen_ocr
            module_name = package.replace("-", "_")
            success = self._verify_single_import(module_name)
            results[package] = success

            status = "✓" if success else "✗"
            logger.info(f"[InstallationVerifier] {status} {package}")

        return results

    def _verify_single_import(self, module_name: str) -> bool:
        """Verify that a single module can be imported.

        Args:
            module_name: Module name (e.g., "lumen_ocr")

        Returns:
            True if import successful
        """
        cmd = [
            str(self.env_manager.micromamba_exe),
            "run",
            "-n",
            PythonEnvManager.ENV_NAME,
            "python",
            "-c",
            f"from {module_name} import *",
        ]

        logger.debug(f"[InstallationVerifier] Verifying import: {module_name}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                logger.debug(f"[InstallationVerifier] Import successful: {module_name}")
                return True
            else:
                logger.warning(
                    f"[InstallationVerifier] Import failed for {module_name}: "
                    f"{result.stderr[:200]}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"[InstallationVerifier] Import timed out: {module_name}")
            return False
        except Exception as e:
            logger.error(f"[InstallationVerifier] Import error for {module_name}: {e}")
            return False

    def verify_env_exists(self) -> bool:
        """Verify that lumen_env exists.

        Returns:
            True if environment exists
        """
        exists = self.env_manager.env_exists()
        logger.info(f"[InstallationVerifier] Environment exists: {exists}")
        return exists
