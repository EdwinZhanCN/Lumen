"""Tests for install API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lumen_app.web.api.install import _check_installation_components


class TestCheckInstallationComponents:
    """Tests for _check_installation_components function."""

    def test_empty_directory_returns_no_components(self, tmp_path):
        """Test that an empty directory returns all components as False."""
        components = _check_installation_components(tmp_path)

        assert components["micromamba"] is False
        assert components["environment"] is False
        assert components["config"] is False
        assert components["drivers"] is False

    def test_micromamba_detected(self, tmp_path):
        """Test that micromamba is detected when present."""
        # Create a mock micromamba executable
        micromamba_dir = tmp_path / "micromamba" / "bin"
        micromamba_dir.mkdir(parents=True)
        micromamba_exe = micromamba_dir / "micromamba"
        micromamba_exe.write_text("#!/bin/sh\necho 'micromamba'")
        micromamba_exe.chmod(0o755)

        with patch(
            "lumen_app.utils.installation.MicromambaInstaller"
        ) as mock_installer_class:
            mock_installer = MagicMock()
            mock_result = MagicMock()
            mock_result.status = "INSTALLED"
            mock_result.executable_path = str(micromamba_exe)
            mock_installer.check.return_value = mock_result
            mock_installer_class.return_value = mock_installer

            components = _check_installation_components(tmp_path)

            assert components["micromamba"] is True

    def test_environment_detected(self, tmp_path):
        """Test that environment is detected when present."""
        env_dir = tmp_path / "envs" / "lumen_env"
        env_dir.mkdir(parents=True)
        (env_dir / "bin").mkdir()

        components = _check_installation_components(tmp_path)

        assert components["environment"] is True

    def test_config_detected(self, tmp_path):
        """Test that config is detected when present."""
        config_file = tmp_path / "lumen-config.yaml"
        config_file.write_text("model: test\n")

        components = _check_installation_components(tmp_path)

        assert components["config"] is True


class TestCheckInstallationPathEndpoint:
    """Tests for the /check-path endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from lumen_app.web.main import create_app

        app = create_app()
        return TestClient(app)

    def test_empty_path_returns_configure_new(self, client):
        """Test that empty path returns configure_new action."""
        response = client.get("/api/v1/install/check-path?path=")
        assert response.status_code == 200

        data = response.json()
        assert data["recommended_action"] == "configure_new"
        assert data["has_existing_service"] is False
        assert data["ready_to_start"] is False

    def test_no_path_returns_configure_new(self, client):
        """Test that missing path query param returns configure_new."""
        response = client.get("/api/v1/install/check-path")
        assert response.status_code == 200

        data = response.json()
        assert data["recommended_action"] == "configure_new"

    def test_complete_installation_detected(self, client, tmp_path):
        """Test that a complete installation is detected."""
        # Create complete installation
        (tmp_path / "envs" / "lumen_env" / "bin").mkdir(parents=True)
        (tmp_path / "lumen-config.yaml").write_text("model: test\n")

        # Create mock micromamba
        micromamba_exe = tmp_path / "micromamba" / "bin" / "micromamba"
        micromamba_exe.parent.mkdir(parents=True)
        micromamba_exe.write_text("#!/bin/sh\necho 'micromamba'")
        micromamba_exe.chmod(0o755)

        with patch("lumen_app.utils.installation.MicromambaInstaller") as mock_class:
            mock_installer = MagicMock()
            mock_result = MagicMock()
            mock_result.status = "INSTALLED"
            mock_result.executable_path = str(micromamba_exe)
            mock_installer.check.return_value = mock_result
            mock_class.return_value = mock_installer

            response = client.get(f"/api/v1/install/check-path?path={tmp_path}")
            assert response.status_code == 200

            data = response.json()
            assert data["recommended_action"] == "start_existing"
            assert data["has_existing_service"] is True
            assert data["ready_to_start"] is True
            assert data["service_status"]["micromamba"] is True
            assert data["service_status"]["environment"] is True
            assert data["service_status"]["config"] is True

    def test_partial_installation_detected(self, client, tmp_path):
        """Test that a partial installation (only micromamba) is detected."""
        # Create only micromamba executable (no env or config)
        micromamba_exe = tmp_path / "micromamba" / "bin" / "micromamba"
        micromamba_exe.parent.mkdir(parents=True)
        micromamba_exe.write_text("#!/bin/sh\necho 'micromamba'")
        micromamba_exe.chmod(0o755)

        with patch("lumen_app.utils.installation.MicromambaInstaller") as mock_class:
            mock_installer = MagicMock()
            mock_result = MagicMock()
            mock_result.status = "INSTALLED"
            mock_result.executable_path = str(micromamba_exe)
            mock_installer.check.return_value = mock_result
            mock_class.return_value = mock_installer

            response = client.get(f"/api/v1/install/check-path?path={tmp_path}")
            assert response.status_code == 200

            data = response.json()
            assert data["recommended_action"] == "configure_new"
            assert data["has_existing_service"] is True  # Has micromamba
            assert data["ready_to_start"] is False  # But not complete
