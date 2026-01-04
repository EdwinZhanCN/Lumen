from dataclasses import dataclass
from typing import Literal

from lumen_resources import LumenConfig
from lumen_resources.lumen_config import (
    BackendSettings,
    Deployment1,
    ImportInfo,
    Mdns,
    Metadata,
    ModelConfig,
    Region,
    Runtime,
    Server,
    Service,
    Services,
)


@dataclass
class DependencyMetadata:
    """Python 依赖元数据（pip 管理的部分）。

    用于描述特定平台的 Python 依赖安装要求，包括可选依赖标识符、
    额外的 PyPI 索引 URL 等。

    Attributes:
        extra_index_url: 额外的 PyPI 索引 URL（如 CUDA PyTorch 索引）
        extra_deps: 可选依赖标识符（对应 pyproject.toml 中的 [project.optional-dependencies]）
        python_version: Python 版本要求
        install_args: 额外的 pip 安装参数
    """

    extra_index_url: list[str] | None = None
    extra_deps: list[str] | None = None
    python_version: str = "3.11"
    install_args: list[str] | None = None


@dataclass
class DeviceConfig:
    """设备配置类。

    定义特定硬件平台的推理配置，包括运行时、ONNX 提供者、
    批处理大小、精度，以及相关的依赖元数据。

    Attributes:
        runtime: 推理运行时类型（torch/onnx/rknn）
        onnx_providers: ONNX 执行提供者列表
        rknn_device: RKNN 设备标识（如 "rk3588"）
        batch_size: 设备硬编码批处理大小（如 CPU/NPU 固定为 1）
        description: 设备描述
        precision: 设备硬编码精度（根据设备支持情况自动选择）
        env: mamba yaml 配置文件标识符（default/cuda/openvino/tensorrt）
        os: 操作系统约束（linux/win/darwin）
        dependency_metadata: Python 依赖元数据
    """

    runtime: Runtime
    onnx_providers: list | None
    rknn_device: str | None = None  # 如 "rk3588"
    batch_size: int | None = (
        None  # 设备硬编码batch_size，如cpu, npu设备均为1，如果是动态batch_size 设备，则为None，可以在后续根据任务类型/期望配置灵活调整。将会覆盖LumenConig中的batch_size字段。
    )
    description: str = ""
    precision: str | None = (
        None  # 设备硬编码精度，根据设备的支持情况，自动选择最优的精度，将会覆盖LumenConfig中的precision字段。
    )
    env: str = "default"  # mamba yaml 配置文件标识符
    os: str | None = None  # 操作系统约束 (linux/win/darwin)
    dependency_metadata: DependencyMetadata | None = None  # Python 依赖元数据

    @classmethod
    def rockchip(cls, rknn_device: str):
        return cls(
            runtime=Runtime.rknn,
            onnx_providers=None,
            rknn_device=rknn_device,
            batch_size=1,  # NPU fixed to 1 batch size
            description="Preset for Rockchip NPU",
            precision="int8",
            env="default",
            dependency_metadata=DependencyMetadata(extra_deps=["rknn"]),
        )

    @classmethod
    def apple_silicon(cls, cache_dir: str = "."):
        return cls(
            runtime=Runtime.onnx,
            onnx_providers=[
                (
                    "CoreMLExecutionProvider",
                    {
                        "ModelFormat": "MLProgram",
                        "MLComputeUnits": "ALL",
                        "RequireStaticInputShapes": "0",
                        "EnableOnSubgraphs": "0",
                        "SpecializationStrategy": "FastPrediction",
                        "ModelCacheDirectory": f"{cache_dir}/cache/coreml",
                    },
                ),
                "CPUExecutionProvider",
            ],
            batch_size=1,  # NPU fixed to 1 batch size
            description="Preset for Apple Silicon",
            env="default",
            dependency_metadata=DependencyMetadata(extra_deps=["apple"]),
        )

    @classmethod
    def nvidia_gpu(cls):
        return cls(
            runtime=Runtime.onnx,
            onnx_providers=[
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                "CPUExecutionProvider",
            ],
            batch_size=4,
            description="Preset for low RAM (< 12GB) Nvidia GPUs",
            env="cuda",
            dependency_metadata=DependencyMetadata(
                extra_deps=["cuda"],
            ),
        )

    @classmethod
    def nvidia_gpu_high(cls, cache_dir: str = "."):
        return cls(
            runtime=Runtime.onnx,
            onnx_providers=[
                (
                    "TensorRTExecutionProvider",
                    {
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": f"{cache_dir}/cache/trt",
                        "trt_max_workspace_size": 2147483648,  # 2GB
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            description="Preset for high RAM (>= 12GB) Nvidia GPUs",
            env="tensorrt",
            dependency_metadata=DependencyMetadata(
                extra_deps=["cuda"],
            ),
        )

    @classmethod
    def intel_gpu(cls, cache_dir: str = "."):
        return cls(
            runtime=Runtime.onnx,
            onnx_providers=[
                (
                    "OpenVINOExecutionProvider",
                    {
                        "device_type": "GPU",
                        "precision": "FP16",
                        "num_streams": 1,
                        "cache_dir": f"{cache_dir}/cache/ov",
                    },
                ),
                "CPUExecutionProvider",
            ],
            description="Preset for Intel iGPU or Arc GPU",
            precision="fp16",
            env="openvino",
            dependency_metadata=DependencyMetadata(extra_deps=["openvino"]),
        )

    # ROCm Support is under evaluation.
    # @classmethod
    # def amd_gpu(cls):
    #     return cls(
    #         runtime=Runtime.onnx,
    #         onnx_providers=[
    #             (
    #                 "MIGraphXExecutionProvider",
    #                 {
    #                     "migraphx_fp16_enable": 1,
    #                     "migraphx_int8_enable": 0,
    #                     "migraphx_exhaustive_tune": 0,
    #                 },
    #             ),
    #             "CPUExecutionProvider",
    #         ],
    #         description="Preset for AMD Radeon GPUs",
    #         precision="fp16",
    #     )

    @classmethod
    def amd_gpu_win(cls):
        """
        Installation instruction: Refer to https://ryzenai.docs.amd.com/en/latest/gpu/ryzenai_gpu.html
        """
        return cls(
            runtime=Runtime.onnx,
            onnx_providers=[
                "DMLExecutionProvider",
                "CPUExecutionProvider",
            ],
            description="Preset for AMD Ryzen GPUs",
            env="default",
            dependency_metadata=DependencyMetadata(extra_deps=["cpu"]),
        )

    @classmethod
    def amd_npu(cls, cache_dir: str = "."):
        """
        Installation instruction: Refer to https://ryzenai.docs.amd.com/en/latest/inst.html
        """
        return cls(
            runtime=Runtime.onnx,
            onnx_providers=[
                ("VitisAIExecutionProvider", {"cache_dir": f"{cache_dir}/cache/amd"}),
                "CPUExecutionProvider",
            ],
            description="Preset for AMD Ryzen NPUs",
            env="default",
            dependency_metadata=DependencyMetadata(extra_deps=["cpu"]),
        )

    @classmethod
    def nvidia_jetson(cls):
        return cls(
            runtime=Runtime.onnx,
            onnx_providers=[
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                "CPUExecutionProvider",
            ],
            description="Preset for low RAM (< 12GB) Nvidia Jetson Devices",
            env="default",
            os="linux",
            dependency_metadata=DependencyMetadata(
                extra_index_url=["https://pypi.jetson-ai-lab.io/jp6/cu126"],
                extra_deps=["cuda"],
            ),
        )

    @classmethod
    def nvidia_jetson_high(cls):
        return cls(
            runtime=Runtime.onnx,
            onnx_providers=[
                (
                    "TensorRTExecutionProvider",
                    {
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "./cache/trt",
                        "trt_max_workspace_size": 2147483648,
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            description="Preset for high RAM (>= 12GB) Nvidia Jetson Devices",
            env="default",
            os="linux",
            dependency_metadata=DependencyMetadata(
                extra_index_url=["https://pypi.jetson-ai-lab.io/jp6/cu126"],
                extra_deps=["cuda"],
            ),
        )

    @classmethod
    def cpu(cls):
        return cls(
            runtime=Runtime.onnx,
            onnx_providers=[
                "CPUExecutionProvider",
            ],
            batch_size=1,
            description="Preset General CPUs",
            env="default",
            dependency_metadata=DependencyMetadata(extra_deps=["cpu"]),
        )


class Config:
    def __init__(
        self,
        cache_dir: str,
        device_config: DeviceConfig,
        region: Region,
        service_name: str,
        port: int | None,
    ):
        self.cache_dir: str = cache_dir
        self.region: Region = region
        self.port: int = port or 50051
        self.service_name: str = service_name

        self.unified_runtime: Runtime = device_config.runtime
        self.unified_rknn_device: str | None = device_config.rknn_device
        self.device_config: DeviceConfig = device_config

    def minimal(self) -> LumenConfig:
        return LumenConfig(
            metadata=Metadata(
                version="1.0.0",
                region=self.region,
                cache_dir=self.cache_dir,
            ),
            deployment=Deployment1(
                mode="hub", services=[Service(root="ocr")], service=None
            ),
            server=Server(
                port=self.port,
                host="0.0.0.0",
                mdns=Mdns(enabled=True, service_name=self.service_name),
            ),
            services={
                "ocr": Services(
                    enabled=True,
                    package="lumen_ocr",
                    import_info=ImportInfo(
                        registry_class="lumen_ocr.general_ocr.GeneralOcrService",
                        add_to_server="lumen_ocr.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server",
                    ),
                    backend_settings=BackendSettings(
                        device=None,  # Auto Detect For Non-PyTorch Backend
                        batch_size=self.device_config.batch_size or 1,
                        onnx_providers=self.device_config.onnx_providers,
                    ),
                    models={
                        "general": ModelConfig(
                            model="PP-OCRv5",
                            runtime=self.unified_runtime,
                            rknn_device=self.unified_rknn_device,
                            precision="fp32",
                            dataset=None,
                        )
                    },
                )
            },
        )

    def light_weight(
        self, clip_model: Literal["MobileCLIP2-S2", "CN-CLIP_ViT-B-16"]
    ) -> LumenConfig:
        return LumenConfig(
            metadata=Metadata(
                version="1.0.0",
                region=self.region,
                cache_dir=self.cache_dir,
            ),
            deployment=Deployment1(
                mode="hub",
                services=[
                    Service(root="ocr"),
                    Service(root="clip"),
                    Service(root="face"),
                ],
                service=None,
            ),
            server=Server(
                port=self.port,
                host="0.0.0.0",
                mdns=Mdns(enabled=True, service_name=self.service_name),
            ),
            services={
                "ocr": Services(
                    enabled=True,
                    package="lumen_ocr",
                    import_info=ImportInfo(
                        registry_class="lumen_ocr.general_ocr.GeneralOcrService",
                        add_to_server="lumen_ocr.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server",
                    ),
                    backend_settings=BackendSettings(
                        device=None,  # Auto Detect For Non-PyTorch Backend
                        batch_size=self.device_config.batch_size or 1,
                        onnx_providers=self.device_config.onnx_providers,
                    ),
                    models={
                        "general": ModelConfig(
                            model="PP-OCRv5",
                            runtime=self.unified_runtime,
                            rknn_device=self.unified_rknn_device,
                            precision="fp32",
                            dataset=None,
                        )
                    },
                ),
                "clip": Services(
                    enabled=True,
                    package="lumen_clip",
                    import_info=ImportInfo(
                        registry_class="lumen_clip.general_clip.clip_service.GeneralCLIPService",
                        add_to_server="lumen_clip.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server",
                    ),
                    backend_settings=BackendSettings(
                        device=None,  # Auto Detect For Non-PyTorch Backend
                        batch_size=self.device_config.batch_size or 1,
                        onnx_providers=self.device_config.onnx_providers,
                    ),
                    models={
                        "general": ModelConfig(
                            model=clip_model,
                            runtime=self.unified_runtime,
                            rknn_device=self.unified_rknn_device,
                            dataset="ImageNet_1k",
                            precision=self.device_config.precision or "fp16",
                        )
                    },
                ),
                "face": Services(
                    enabled=True,
                    package="lumen_face",
                    import_info=ImportInfo(
                        registry_class="lumen_face.general_face.GeneralFaceService",
                        add_to_server="lumen_face.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server",
                    ),
                    backend_settings=BackendSettings(
                        device=None,  # Auto Detect For Non-PyTorch Backend
                        batch_size=self.device_config.batch_size or 1,
                        onnx_providers=self.device_config.onnx_providers,
                    ),
                    models={
                        "general": ModelConfig(
                            model="buffalo_l",
                            runtime=self.unified_runtime,
                            rknn_device=self.unified_rknn_device,
                            precision=self.device_config.precision or "fp16",
                            dataset=None,
                        )
                    },
                ),
            },
        )

    def basic(
        self, clip_model: Literal["MobileCLIP2-S4", "CN-CLIP_ViT-L-14"]
    ) -> LumenConfig:
        return LumenConfig(
            metadata=Metadata(
                version="1.0.0",
                region=self.region,
                cache_dir=self.cache_dir,
            ),
            deployment=Deployment1(
                mode="hub",
                services=[
                    Service(root="ocr"),
                    Service(root="clip"),
                    Service(root="face"),
                    Service(root="vlm"),
                ],
                service=None,
            ),
            server=Server(
                port=self.port,
                host="0.0.0.0",
                mdns=Mdns(enabled=True, service_name=self.service_name),
            ),
            services={
                "ocr": Services(
                    enabled=True,
                    package="lumen_ocr",
                    import_info=ImportInfo(
                        registry_class="lumen_ocr.general_ocr.GeneralOcrService",
                        add_to_server="lumen_ocr.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server",
                    ),
                    backend_settings=BackendSettings(
                        device=None,  # Auto Detect For Non-PyTorch Backend
                        batch_size=self.device_config.batch_size or 5,
                        onnx_providers=self.device_config.onnx_providers,
                    ),
                    models={
                        "general": ModelConfig(
                            model="PP-OCRv5",
                            runtime=self.unified_runtime,
                            rknn_device=self.unified_rknn_device,
                            precision=self.device_config.precision or "fp32",
                            dataset=None,
                        )
                    },
                ),
                "clip": Services(
                    enabled=True,
                    package="lumen_clip",
                    import_info=ImportInfo(
                        registry_class="lumen_clip.general_clip.clip_service.GeneralCLIPService",
                        add_to_server="lumen_clip.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server",
                    ),
                    backend_settings=BackendSettings(
                        device=None,  # Auto Detect For Non-PyTorch Backend
                        batch_size=self.device_config.batch_size or 5,
                        onnx_providers=self.device_config.onnx_providers,
                    ),
                    models={
                        "general": ModelConfig(
                            model=clip_model,
                            runtime=self.unified_runtime,
                            rknn_device=self.unified_rknn_device,
                            dataset="ImageNet_1k",
                            precision=self.device_config.precision or "fp16",
                        )
                    },
                ),
                "face": Services(
                    enabled=True,
                    package="lumen_face",
                    import_info=ImportInfo(
                        registry_class="lumen_face.general_face.GeneralFaceService",
                        add_to_server="lumen_face.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server",
                    ),
                    backend_settings=BackendSettings(
                        device=None,  # Auto Detect For Non-PyTorch Backend
                        batch_size=self.device_config.batch_size or 5,
                        onnx_providers=self.device_config.onnx_providers,
                    ),
                    models={
                        "general": ModelConfig(
                            model="antelopev2",
                            runtime=self.unified_runtime,
                            rknn_device=self.unified_rknn_device,
                            precision=self.device_config.precision or "fp16",
                            dataset=None,
                        )
                    },
                ),
                "vlm": Services(
                    enabled=True,
                    package="lumen_vlm",
                    import_info=ImportInfo(
                        registry_class="lumen_vlm.fastvlm.GeneralFastVLMService",
                        add_to_server="lumen_vlm.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server",
                    ),
                    backend_settings=BackendSettings(
                        device=None,  # Auto Detect For Non-PyTorch Backend
                        batch_size=self.device_config.batch_size or 1,
                        onnx_providers=self.device_config.onnx_providers,
                    ),
                    models={
                        "general": ModelConfig(
                            model="FastVLM-0.5B",
                            runtime=self.unified_runtime,
                            rknn_device=self.unified_rknn_device,
                            precision=self.device_config.precision or "int8",
                            dataset=None,
                        )
                    },
                ),
            },
        )

    def brave(self) -> LumenConfig:
        return LumenConfig(
            metadata=Metadata(
                version="1.0.0",
                region=self.region,
                cache_dir=self.cache_dir,
            ),
            deployment=Deployment1(
                mode="hub",
                service=None,  # None for hub mode
                services=[
                    Service(root="ocr"),
                    Service(root="clip"),
                    Service(root="face"),
                    Service(root="vlm"),
                ],
            ),
            server=Server(
                port=self.port,
                host="0.0.0.0",
                mdns=Mdns(enabled=True, service_name=self.service_name),
            ),
            services={
                "ocr": Services(
                    enabled=True,
                    package="lumen_ocr",
                    import_info=ImportInfo(
                        registry_class="lumen_ocr.general_ocr.GeneralOcrService",
                        add_to_server="lumen_ocr.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server",
                    ),
                    backend_settings=BackendSettings(
                        device=None,  # Auto Detect For Non-PyTorch Backend
                        batch_size=self.device_config.batch_size or 10,
                        onnx_providers=self.device_config.onnx_providers,
                    ),
                    models={
                        "general": ModelConfig(
                            model="PP-OCRv5",
                            runtime=self.unified_runtime,
                            rknn_device=self.unified_rknn_device,
                            precision=self.device_config.precision or "fp32",
                            dataset=None,
                        )
                    },
                ),
                "clip": Services(
                    enabled=True,
                    package="lumen_clip",
                    import_info=ImportInfo(
                        registry_class="lumen_clip.expert_bioclip.BioCLIPService",
                        add_to_server="lumen_clip.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server",
                    ),
                    backend_settings=BackendSettings(
                        device=None,  # Auto Detect For Non-PyTorch Backend
                        batch_size=self.device_config.batch_size or 8,
                        onnx_providers=self.device_config.onnx_providers,
                    ),
                    models={
                        "bioclip": ModelConfig(
                            model="bioclip-2",
                            runtime=self.unified_runtime,
                            rknn_device=self.unified_rknn_device,
                            dataset="TreeOfLife-200M",
                            precision=self.device_config.precision or "fp16",
                        )
                    },
                ),
                "face": Services(
                    enabled=True,
                    package="lumen_face",
                    import_info=ImportInfo(
                        registry_class="lumen_face.general_face.GeneralFaceService",
                        add_to_server="lumen_face.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server",
                    ),
                    backend_settings=BackendSettings(
                        device=None,  # Auto Detect For Non-PyTorch Backend
                        batch_size=self.device_config.batch_size or 8,
                        onnx_providers=self.device_config.onnx_providers,
                    ),
                    models={
                        "general": ModelConfig(
                            model="antelopev2",
                            runtime=self.unified_runtime,
                            rknn_device=self.unified_rknn_device,
                            precision=self.device_config.precision or "fp16",
                            dataset=None,
                        )
                    },
                ),
                "vlm": Services(
                    enabled=True,
                    package="lumen_vlm",
                    import_info=ImportInfo(
                        registry_class="lumen_vlm.fastvlm.GeneralFastVLMService",
                        add_to_server="lumen_vlm.proto.ml_service_pb2_grpc.add_InferenceServicer_to_server",
                    ),
                    backend_settings=BackendSettings(
                        device=None,  # Auto Detect For Non-PyTorch Backend
                        batch_size=self.device_config.batch_size or 1,
                        onnx_providers=self.device_config.onnx_providers,
                    ),
                    models={
                        "general": ModelConfig(
                            model="FastVLM-0.5B",
                            runtime=self.unified_runtime,
                            rknn_device=self.unified_rknn_device,
                            precision=self.device_config.precision or "fp16",
                            dataset=None,
                        )
                    },
                ),
            },
        )
