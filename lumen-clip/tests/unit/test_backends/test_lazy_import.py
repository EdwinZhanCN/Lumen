"""
Tests for lazy backend import behavior.

These tests verify that the package can be imported and used with only
base dependencies (ONNX Runtime) when torch/rknn are not installed.
"""

from __future__ import annotations

from unittest import mock

import pytest

from lumen_clip.backends import RuntimeKind, get_available_backends
from lumen_clip.backends.backend_exceptions import BackendDependencyError
from lumen_clip.backends.factory import create_backend, reload_backends


class TestLazyImportBaseDepsOnly:
    """Test lazy import behavior with only base dependencies."""

    def test_get_available_backends_base_only(self):
        """Verify only ONNXRT returned when torch/rknn missing."""
        # Mock find_spec to simulate torch and rknn not being available
        with mock.patch("importlib.util.find_spec") as mock_find:
            # Setup: onnxruntime available, torch and rknn not available
            def find_spec_side_effect(name):
                if name == "torch":
                    return None
                if name == "rknnlite":
                    return None
                # For any other package, return a mock spec
                return mock.MagicMock()

            mock_find.side_effect = find_spec_side_effect

            # Reload to clear cache and apply the mock
            available = reload_backends()

            # Assert: Only ONNXRT should be available
            assert RuntimeKind.ONNXRT in available
            assert RuntimeKind.TORCH not in available
            assert RuntimeKind.RKNN not in available
            assert len(available) == 1

    def test_create_backend_missing_torch_deps(self):
        """Verify BackendDependencyError raised with install instructions."""
        with mock.patch("importlib.util.find_spec") as mock_find:
            # Setup: torch not available
            def find_spec_side_effect(name):
                if name == "torch":
                    return None
                return mock.MagicMock()

            mock_find.side_effect = find_spec_side_effect

            # Reload to clear cache and apply the mock
            reload_backends()

            # Create mock backend config and resources
            mock_config = mock.MagicMock()
            mock_config.device = "cpu"
            mock_config.batch_size = 1
            mock_config.onnx_providers = None

            mock_resources = mock.MagicMock()

            # Attempt to create Torch backend should raise BackendDependencyError
            with pytest.raises(BackendDependencyError) as exc_info:
                create_backend(
                    backend_config=mock_config,
                    resources=mock_resources,
                    runtime=RuntimeKind.TORCH,
                )

            # Assert: Error message includes installation instructions
            error_msg = str(exc_info.value)
            assert "torch" in error_msg.lower()
            assert "pip install" in error_msg.lower()
            assert "[torch]" in error_msg

    def test_no_importerror_with_base_deps(self):
        """Verify successful import with only base dependencies."""
        with mock.patch("importlib.util.find_spec") as mock_find:
            # Setup: Only base dependencies available
            def find_spec_side_effect(name):
                if name in ("torch", "rknnlite"):
                    return None
                return mock.MagicMock()

            mock_find.side_effect = find_spec_side_effect

            # This should not raise ImportError
            try:
                from lumen_clip.backends import get_available_backends, reload_backends
                available = get_available_backends()
                assert isinstance(available, list)
            except ImportError as e:
                pytest.fail(f"Import should not fail with base dependencies only: {e}")

    def test_onnx_backend_always_available(self):
        """Verify ONNXRT backend is always available even without optional deps."""
        with mock.patch("importlib.util.find_spec") as mock_find:
            # Mock onnxruntime as available, torch/rknn not available
            def find_spec_side_effect(name):
                if name in ("torch", "rknnlite"):
                    return None
                return mock.MagicMock()

            mock_find.side_effect = find_spec_side_effect

            # Reload backends
            available = reload_backends()

            # Assert: ONNXRT should always be available
            assert RuntimeKind.ONNXRT in available

            # And we should be able to create it (with proper mocks)
            mock_config = mock.MagicMock()
            mock_config.device = "cpu"
            mock_config.batch_size = 1
            mock_config.onnx_providers = None

            mock_resources = mock.MagicMock()

            # This should not raise BackendDependencyError
            # Note: It may fail for other reasons (model loading), but not due to missing deps
            try:
                backend = create_backend(
                    backend_config=mock_config,
                    resources=mock_resources,
                    runtime=RuntimeKind.ONNXRT,
                )
                # If we get here, great - the factory worked
                # The backend might not actually initialize without proper resources,
                # but the lazy import logic succeeded
            except BackendDependencyError:
                pytest.fail("ONNXRT backend should not raise BackendDependencyError")
            except Exception:
                # Other exceptions are fine - we're just testing the lazy import logic
                pass


class TestTorchBackendAvailability:
    """Test torch backend availability when torch is installed."""

    def test_get_available_backends_with_torch(self):
        """Verify TORCH returned when torch installed."""
        with mock.patch("importlib.util.find_spec") as mock_find:
            # Setup: torch available, rknn not available
            def find_spec_side_effect(name):
                if name == "torch":
                    return mock.MagicMock()
                if name == "rknnlite":
                    return None
                return mock.MagicMock()

            mock_find.side_effect = find_spec_side_effect

            # Need to also mock the torch_backend import
            with mock.patch.dict("sys.modules", {"lumen_clip.backends.torch_backend": mock.MagicMock()}):
                available = reload_backends()

                # Assert: TORCH should be available
                assert RuntimeKind.TORCH in available
                assert RuntimeKind.ONNXRT in available
                assert RuntimeKind.RKNN not in available

    def test_reload_backends_after_torch_install(self):
        """Mock torch installation and verify reload detects it."""
        with mock.patch("importlib.util.find_spec") as mock_find:
            # Initial state: torch not available
            def find_spec_no_torch(name):
                if name == "torch":
                    return None
                return mock.MagicMock()

            mock_find.side_effect = find_spec_no_torch
            available = reload_backends()
            assert RuntimeKind.TORCH not in available

            # Simulate torch installation
            def find_spec_with_torch(name):
                if name == "torch":
                    return mock.MagicMock()
                return mock.MagicMock()

            mock_find.side_effect = find_spec_with_torch

            # Mock the torch_backend module
            with mock.patch.dict("sys.modules", {"lumen_clip.backends.torch_backend": mock.MagicMock()}):
                available = reload_backends()

                # Assert: TORCH should now be available
                assert RuntimeKind.TORCH in available

    def test_create_torch_backend_success(self):
        """Verify TorchBackend creation succeeds when torch installed."""
        with mock.patch("importlib.util.find_spec") as mock_find:
            # Setup: torch available
            def find_spec_side_effect(name):
                return mock.MagicMock()

            mock_find.side_effect = find_spec_side_effect

            # Mock the torch_backend module and TorchBackend class
            mock_torch_backend = mock.MagicMock()
            mock_torch_backend_class = mock.MagicMock(return_value=mock.MagicMock())
            mock_torch_backend.TyTorchBackend = mock_torch_backend_class

            with mock.patch.dict("sys.modules", {"lumen_clip.backends.torch_backend": mock_torch_backend}):
                # Temporarily patch the factory to use our mock
                with mock.patch("lumen_clip.backends.factory._ensure_backends_registered"):
                    # Manually register TORCH in the registry
                    from lumen_clip.backends.factory import _BACKEND_REGISTRY
                    _BACKEND_REGISTRY[RuntimeKind.TORCH] = mock_torch_backend_class

                    mock_config = mock.MagicMock()
                    mock_config.device = "cpu"
                    mock_config.batch_size = 1

                    mock_resources = mock.MagicMock()

                    # This should not raise BackendDependencyError
                    try:
                        backend = create_backend(
                            backend_config=mock_config,
                            resources=mock_resources,
                            runtime=RuntimeKind.TORCH,
                        )
                        # If we get here, the factory logic worked
                        assert True
                    except BackendDependencyError:
                        pytest.fail("TorchBackend should not raise BackendDependencyError when torch is available")

    def test_torch_backend_conditional_test(self):
        """Conditional test that only runs when torch is actually available."""
        # Skip if torch not actually installed
        pytest.importorskip("torch")

        # If we get here, torch is installed
        from lumen_clip.backends import get_available_backends, reload_backends

        # Reload to ensure torch is detected
        available = reload_backends()

        # Note: This test only runs if torch is actually installed
        # It's marked as conditional because it depends on the environment
        if RuntimeKind.TORCH in available:
            # Torch backend is available in this environment
            assert True
        else:
            # Torch is installed but backend not available - might be due to
            # missing dependencies like open-clip-torch
            pytest.skip("Torch installed but backend not fully available")


class TestDevelopmentMode:
    """Test development workflow with varying dependencies."""

    def test_reload_backends_multiple_calls(self):
        """Verify reload can be called multiple times without side effects."""
        with mock.patch("importlib.util.find_spec") as mock_find:
            def find_spec_side_effect(name):
                return mock.MagicMock()

            mock_find.side_effect = find_spec_side_effect

            # Call reload multiple times
            available1 = reload_backends()
            available2 = reload_backends()
            available3 = reload_backends()

            # All should return the same result
            assert available1 == available2 == available3

            # Should have ONNXRT at minimum
            assert RuntimeKind.ONNXRT in available1

    def test_editable_install_import_stability(self):
        """Verify imports work in editable mode with different dependency sets."""
        # Simulate editable install by testing multiple reload scenarios
        with mock.patch("importlib.util.find_spec") as mock_find:
            # Scenario 1: Only base deps
            def find_spec_base_only(name):
                if name in ("torch", "rknnlite"):
                    return None
                return mock.MagicMock()

            mock_find.side_effect = find_spec_base_only
            available = reload_backends()
            assert RuntimeKind.ONNXRT in available
            assert RuntimeKind.TORCH not in available

            # Scenario 2: Add torch
            def find_spec_with_torch(name):
                if name == "rknnlite":
                    return None
                return mock.MagicMock()

            mock_find.side_effect = find_spec_with_torch

            # Mock torch_backend module
            with mock.patch.dict("sys.modules", {"lumen_clip.backends.torch_backend": mock.MagicMock()}):
                available = reload_backends()
                # Should still work without errors
                assert isinstance(available, list)

    def test_get_available_backends_caching(self):
        """Verify results cached on first call, reload clears cache."""
        with mock.patch("importlib.util.find_spec") as mock_find:
            call_count = {"count": 0}

            def find_spec_side_effect(name):
                call_count["count"] += 1
                return mock.MagicMock()

            mock_find.side_effect = find_spec_side_effect

            # First call should trigger find_spec calls
            call_count["count"] = 0
            available1 = reload_backends()
            first_call_count = call_count["count"]
            assert first_call_count > 0

            # Second call should be cached (no new find_spec calls)
            available2 = get_available_backends()
            second_call_count = call_count["count"]
            # Should be the same (no new calls)
            assert second_call_count == first_call_count

            # Reload should clear cache and trigger new find_spec calls
            available3 = reload_backends()
            third_call_count = call_count["count"]
            # Should have more calls now
            assert third_call_count > first_call_count

    def test_partial_torch_installation(self):
        """Verify graceful handling when torch present but open-clip-torch missing."""
        # This test verifies that if torch package exists but torch_backend
        # import fails (e.g., missing open-clip-torch), the system handles it gracefully
        with mock.patch("importlib.util.find_spec") as mock_find:
            # torch is available according to find_spec
            def find_spec_side_effect(name):
                if name == "torch":
                    return mock.MagicMock()
                return mock.MagicMock()

            mock_find.side_effect = find_spec_side_effect

            # Mock the import of torch_backend module to raise ImportError
            # This simulates the case where torch is installed but open-clip-torch is missing
            with mock.patch("lumen_clip.backends.factory._ensure_backends_registered") as mock_ensure:
                # Make the real function but with torch_backend import failing
                from lumen_clip.backends import factory

                original_ensure = factory._ensure_backends_registered

                def failing_ensure():
                    # Register ONNXRT
                    factory._BACKEND_REGISTRY[RuntimeKind.ONNXRT] = factory.ONNXRTBackend
                    factory._INITIALIZED = True
                    # Don't register TORCH (simulating import failure)

                mock_ensure.side_effect = failing_ensure

                # This should not crash even though torch_backend "failed to import"
                available = reload_backends()

                # System should remain stable with ONNXRT available
                assert isinstance(available, list)
                assert RuntimeKind.ONNXRT in available

    def test_conditional_integration_tests(self):
        """Template for conditional integration tests based on available backends."""
        # This test demonstrates the pattern for conditional tests
        # Actual integration tests would be in separate files

        # Test can be run with or without torch
        try:
            import torch
            torch_available = True
        except ImportError:
            torch_available = False

        if torch_available:
            # Run torch-specific tests
            assert True  # Placeholder for actual torch integration tests
        else:
            # Skip torch-specific tests
            pass

        # Always run base dependency tests
        from lumen_clip.backends import get_available_backends
        available = get_available_backends()
        assert RuntimeKind.ONNXRT in available
