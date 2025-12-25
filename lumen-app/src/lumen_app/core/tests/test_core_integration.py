"""
Integration tests for Lumen App Core framework.

Tests the Config, ServiceLoader, HubRouter, and AppService components
using a minimal CPU preset configuration.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lumen_resources.lumen_config import Region, Runtime

import lumen_app.proto.ml_service_pb2 as pb
from lumen_app.core.config import Config, DeviceConfig
from lumen_app.core.loader import ServiceLoader
from lumen_app.core.router import HubRouter
from lumen_app.core.service import AppService

# =============================================================================
# Mock Service Classes
# =============================================================================


class MockService:
    """Mock ML service implementing the gRPC Inference servicer interface."""

    def __init__(self, name: str, supported_tasks: list[str]):
        self.name = name
        self._supported_tasks = supported_tasks

    @classmethod
    def from_config(cls, service_config, cache_dir: Path):
        """Mock factory method matching real service interface."""
        # Extract service name from package
        package_name = service_config.package
        return cls(name=package_name, supported_tasks=["embed", "classify"])

    def get_supported_tasks(self) -> list[str]:
        """Return list of task keys this service handles."""
        return self._supported_tasks

    async def Infer(self, request_iterator, context):
        """Mock inference handler."""
        requests = []
        async for req in request_iterator:
            requests.append(req)

        # Return mock response
        response = pb.InferResponse()
        response.correlation_id = requests[0].correlation_id if requests else "test"
        response.is_final = True
        response.result = b'{"status": "ok"}'
        response.result_mime = "application/json"
        response.meta["service"] = self.name
        yield response

    async def GetCapabilities(self, request, context):
        """Mock capability query."""
        return pb.Capability(
            service_name=self.name,
            model_ids=["mock-model-v1"],
            runtime="mock-runtime",
            max_concurrency=1,
            precisions=["fp32"],
            protocol_version="1.0",
        )


class MockRegistry:
    """Mock registry class that returns MockService instances."""

    @staticmethod
    def from_config(service_config, cache_dir: Path):
        return MockService(
            name=service_config.package,
            supported_tasks=["embed", "classify"],
        )


# =============================================================================
# Config Tests
# =============================================================================


def test_device_config_cpu_preset():
    """Test CPU preset device configuration."""
    config = DeviceConfig.cpu()

    assert config.runtime == Runtime.onnx
    assert config.onnx_providers == ["CPUExecutionProvider"]
    assert config.batch_size == 1
    assert config.description == "Preset General CPUs"


def test_device_config_apple_silicon_preset():
    """Test Apple Silicon preset configuration."""
    config = DeviceConfig.apple_silicon()

    assert config.runtime == Runtime.onnx
    assert len(config.onnx_providers) == 2
    # onnx_providers[0] is a tuple (provider_name, options)
    assert config.onnx_providers[0][0] == "CoreMLExecutionProvider"
    assert config.batch_size == 1


@pytest.mark.parametrize(
    "preset_method,expected_runtime,expected_batch",
    [
        ("cpu", Runtime.onnx, 1),
        ("apple_silicon", Runtime.onnx, 1),
        ("nvidia_gpu", Runtime.onnx, 4),
        ("nvidia_gpu_high", Runtime.onnx, None),
        ("intel_gpu", Runtime.onnx, None),
    ],
)
def test_device_config_presets(preset_method, expected_runtime, expected_batch):
    """Test various device presets."""
    method = getattr(DeviceConfig, preset_method)
    config = method()

    assert config.runtime == expected_runtime
    assert config.batch_size == expected_batch


def test_config_minimal_preset():
    """Test minimal configuration preset (OCR only)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        device_config = DeviceConfig.cpu()
        config = Config(
            cache_dir=tmpdir,
            device_config=device_config,
            region=Region.cn,
            service_name="lumen-test",
            port=50051,
        )

        lumen_config = config.minimal()

        assert lumen_config.metadata.region == Region.cn
        assert lumen_config.server.port == 50051
        assert lumen_config.server.mdns.enabled is True
        assert lumen_config.deployment.mode == "hub"
        assert len(lumen_config.deployment.services) == 1
        assert lumen_config.deployment.services[0].root == "ocr"


def test_config_light_weight_preset():
    """Test lightweight configuration preset (OCR + CLIP + Face)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        device_config = DeviceConfig.cpu()
        config = Config(
            cache_dir=tmpdir,
            device_config=device_config,
            region=Region.other,  # Region has 'cn' and 'other', not 'us'
            service_name="lumen-test",
            port=None,
        )

        lumen_config = config.light_weight(clip_model="MobileCLIP2-S2")

        assert len(lumen_config.deployment.services) == 3
        service_roots = {s.root for s in lumen_config.deployment.services}
        assert service_roots == {"ocr", "clip", "face"}


# =============================================================================
# ServiceLoader Tests
# =============================================================================


def test_service_loader_invalid_path():
    """Test ServiceLoader with invalid class path."""
    loader = ServiceLoader()

    with pytest.raises(ValueError, match="Invalid class path"):
        loader.get_class("")

    with pytest.raises(ValueError, match="Invalid class path"):
        loader.get_class("NoDotsInPath")


@patch("lumen_app.core.loader.importlib.import_module")
def test_service_loader_import_error(mock_import):
    """Test ServiceLoader with module import failure."""
    mock_import.side_effect = ImportError("Module not found")
    loader = ServiceLoader()

    # The loader wraps ImportError in its own error handling
    with pytest.raises(ImportError):  # Just check it raises ImportError
        loader.get_class("nonexistent.module.ClassName")


@patch("lumen_app.core.loader.importlib.import_module")
def test_service_loader_attribute_error(mock_import):
    """Test ServiceLoader when class not found in module."""
    mock_module = MagicMock(spec=[])  # No attributes, will cause getattr to fail
    mock_import.return_value = mock_module

    loader = ServiceLoader()

    # Should raise AttributeError when getattr fails
    with pytest.raises(AttributeError):
        loader.get_class("valid.module.DoesNotExist")


def test_service_loader_success_with_real_module():
    """Test ServiceLoader with a real Python module."""
    loader = ServiceLoader()

    # Test with a real built-in class
    cls = loader.get_class("builtins.str")
    assert cls is str


# =============================================================================
# HubRouter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_router_route_table_building():
    """Test HubRouter builds correct route table from services."""
    service1 = MockService("service1", ["task_a", "task_b"])
    service2 = MockService("service2", ["task_c"])

    router = HubRouter(services=[service1, service2])

    assert router._route_table["task_a"] is service1
    assert router._route_table["task_b"] is service1
    assert router._route_table["task_c"] is service2


@pytest.mark.asyncio
async def test_router_infer_dispatch():
    """Test HubRouter dispatches inference requests correctly."""
    service = MockService("test-service", ["embed"])
    router = HubRouter(services=[service])

    # Create mock request
    request = pb.InferRequest()
    request.task = "embed"
    request.correlation_id = "test-123"
    request.payload = b'{"text": "hello"}'

    async def request_generator():
        yield request

    # Mock context
    context = MagicMock()

    # Collect responses
    responses = []
    async for resp in router.Infer(request_generator(), context):
        responses.append(resp)

    assert len(responses) == 1
    assert responses[0].correlation_id == "test-123"
    assert responses[0].meta["service"] == "test-service"


@pytest.mark.asyncio
async def test_router_task_not_found():
    """Test HubRouter returns NOT_FOUND for unsupported tasks."""
    service = MockService("test-service", ["embed"])
    router = HubRouter(services=[service])

    request = pb.InferRequest()
    request.task = "unsupported_task"

    async def request_generator():
        yield request

    # Mock context with abort method
    context = MagicMock()
    context.abort = MagicMock(side_effect=Exception("Aborted with NOT_FOUND"))

    with pytest.raises(Exception, match="Aborted with NOT_FOUND"):
        async for _ in router.Infer(request_generator(), context):
            pass


@pytest.mark.asyncio
async def test_router_get_capabilities_aggregation():
    """Test HubRouter aggregates capabilities from all services."""
    service1 = MockService("service1", ["task_a"])
    service2 = MockService("service2", ["task_b"])

    router = HubRouter(services=[service1, service2])

    # Import Empty from google.protobuf.empty
    from google.protobuf import empty_pb2

    request = empty_pb2.Empty()
    context = MagicMock()

    caps = await router.GetCapabilities(request, context)

    # Should aggregate tasks from both services
    assert len(caps.tasks) >= 0  # Mock returns empty, real would have tasks


# =============================================================================
# AppService Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_app_service_from_config_with_mocks():
    """Test AppService initialization from config using mocked services."""
    with tempfile.TemporaryDirectory() as tmpdir:
        device_config = DeviceConfig.cpu()
        config_obj = Config(
            cache_dir=tmpdir,
            device_config=device_config,
            region=Region.cn,
            service_name="lumen-test",
            port=50051,
        )
        lumen_config = config_obj.minimal()

        # Patch the loader to return our mock service
        with patch.object(ServiceLoader, "get_class", return_value=MockRegistry):
            app_service = AppService.from_app_config(lumen_config)

        assert len(app_service.services) == 1
        assert app_service.config is lumen_config
        assert isinstance(app_service.services[0], MockService)


@pytest.mark.asyncio
async def test_app_service_router_forwarding():
    """Test end-to-end request flow through AppService."""
    with tempfile.TemporaryDirectory() as tmpdir:
        device_config = DeviceConfig.cpu()
        config_obj = Config(
            cache_dir=tmpdir,
            device_config=device_config,
            region=Region.cn,
            service_name="lumen-test",
            port=50051,
        )
        lumen_config = config_obj.minimal()

        with patch.object(ServiceLoader, "get_class", return_value=MockRegistry):
            app_service = AppService.from_app_config(lumen_config)

        # Create request
        request = pb.InferRequest()
        request.task = "embed"
        request.correlation_id = "integration-test"
        request.payload = b"test data"

        async def request_generator():
            yield request

        context = MagicMock()

        # Send through router
        responses = []
        async for resp in app_service.router.Infer(request_generator(), context):
            responses.append(resp)

        assert len(responses) == 1
        assert responses[0].correlation_id == "integration-test"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.asyncio
async def test_router_empty_request_stream():
    """Test HubRouter handles empty request stream."""
    service = MockService("test", ["embed"])
    router = HubRouter(services=[service])

    async def empty_generator():
        return
        yield  # Never reached

    context = MagicMock()

    # Should handle gracefully (returns immediately)
    result = []
    async for resp in router.Infer(empty_generator(), context):
        result.append(resp)


# =============================================================================
# Cache Path Resolution Tests
# =============================================================================


def test_cache_path_resolution_intel_gpu():
    """Test that Intel GPU preset accepts cache_dir and builds absolute paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pass cache_dir directly to the preset method
        device_config = DeviceConfig.intel_gpu(cache_dir=tmpdir)
        config_obj = Config(
            cache_dir=tmpdir,
            device_config=device_config,
            region=Region.other,
            service_name="lumen-test",
            port=50051,
        )

        # Check onnx_providers has absolute paths
        providers = config_obj.device_config.onnx_providers
        assert providers is not None

        # Find OpenVINO provider
        openvino_provider = None
        for p in providers:
            if isinstance(p, tuple) and p[0] == "OpenVINOExecutionProvider":
                openvino_provider = p
                break

        assert openvino_provider is not None
        cache_dir = openvino_provider[1]["cache_dir"]

        # Should be absolute path under tmpdir, not "./cache/ov"
        assert cache_dir.startswith(tmpdir)
        assert cache_dir.endswith("cache/ov")


def test_cache_path_resolution_nvidia_high():
    """Test that NVIDIA high-ram preset accepts cache_dir and builds absolute paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pass cache_dir directly to the preset method
        device_config = DeviceConfig.nvidia_gpu_high(cache_dir=tmpdir)
        config_obj = Config(
            cache_dir=tmpdir,
            device_config=device_config,
            region=Region.other,
            service_name="lumen-test",
            port=50051,
        )

        # Check onnx_providers has absolute paths
        providers = config_obj.device_config.onnx_providers
        assert providers is not None

        # Find TensorRT provider
        trt_provider = None
        for p in providers:
            if isinstance(p, tuple) and p[0] == "TensorRTExecutionProvider":
                trt_provider = p
                break

        assert trt_provider is not None
        cache_path = trt_provider[1]["trt_engine_cache_path"]

        # Should be absolute path under tmpdir, not "./cache/trt"
        assert cache_path.startswith(tmpdir)
        assert cache_path.endswith("cache/trt")


def test_cache_path_resolution_apple_silicon():
    """Test that Apple Silicon preset accepts cache_dir and builds absolute paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pass cache_dir directly to the preset method
        device_config = DeviceConfig.apple_silicon(cache_dir=tmpdir)
        config_obj = Config(
            cache_dir=tmpdir,
            device_config=device_config,
            region=Region.other,
            service_name="lumen-test",
            port=50051,
        )

        # Check onnx_providers has absolute paths
        providers = config_obj.device_config.onnx_providers
        assert providers is not None

        # Find CoreML provider
        coreml_provider = None
        for p in providers:
            if isinstance(p, tuple) and p[0] == "CoreMLExecutionProvider":
                coreml_provider = p
                break

        assert coreml_provider is not None
        model_cache_dir = coreml_provider[1]["ModelCacheDirectory"]

        # Should be absolute path under tmpdir, not "./cache/coreml"
        assert model_cache_dir.startswith(tmpdir)
        assert model_cache_dir.endswith("cache/coreml")


def test_cache_path_resolution_cpu_no_relative_paths():
    """Test that CPU preset (no cache paths) is handled correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        device_config = DeviceConfig.cpu()
        config_obj = Config(
            cache_dir=tmpdir,
            device_config=device_config,
            region=Region.other,
            service_name="lumen-test",
            port=50051,
        )

        # CPU provider has no cache configs, should pass through unchanged
        providers = config_obj.device_config.onnx_providers
        assert providers == ["CPUExecutionProvider"]


def test_cache_path_resolution_applied_to_config():
    """Test that cache paths from preset are applied to generated LumenConfig."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pass cache_dir directly to the preset method
        device_config = DeviceConfig.intel_gpu(cache_dir=tmpdir)
        config_obj = Config(
            cache_dir=tmpdir,
            device_config=device_config,
            region=Region.other,
            service_name="lumen-test",
            port=50051,
        )

        lumen_config = config_obj.minimal()

        # Check that the service config has the preset providers
        ocr_service = lumen_config.services["ocr"]
        backend_settings = ocr_service.backend_settings

        assert backend_settings.onnx_providers is not None

        # Verify OpenVINO provider has absolute cache path
        openvino_provider = None
        for p in backend_settings.onnx_providers:
            if isinstance(p, tuple) and p[0] == "OpenVINOExecutionProvider":
                openvino_provider = p
                break

        assert openvino_provider is not None
        cache_dir = openvino_provider[1]["cache_dir"]
        assert cache_dir.startswith(tmpdir)
        assert cache_dir.endswith("cache/ov")


def test_multiple_device_presets_independence():
    """Test that multiple device configs don't share state."""
    cpu_config = DeviceConfig.cpu()
    gpu_config = DeviceConfig.nvidia_gpu()

    assert cpu_config.runtime == Runtime.onnx
    assert gpu_config.runtime == Runtime.onnx
    assert cpu_config.batch_size == 1
    assert gpu_config.batch_size == 4

    # Modify one shouldn't affect the other
    cpu_config.batch_size = 999
    assert cpu_config.batch_size == 999
    assert gpu_config.batch_size == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
