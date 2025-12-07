import type { LumenConfig } from "../types/lumen-config";

/**
 * Default configuration template for lumen-face service
 */
export const defaultFaceConfig: LumenConfig = {
  metadata: {
    version: "1.0.0",
    region: "other",
    cache_dir: "~/.lumen",
  },
  deployment: {
    mode: "single",
  },
  server: {
    port: 50051,
    host: "0.0.0.0",
    mdns: {
      enabled: false,
      service_name: "lumen-face",
    },
  },
  services: {
    face: {
      enabled: true,
      package: "lumen_face",
      import: {
        registry_class: "lumen_face.service.FaceServiceRegistry",
        add_to_server: "lumen_face.service.add_to_server",
      },
      backend_settings: {
        device: null,
        batch_size: 1,
        onnx_providers: null,
      },
      models: {
        default: {
          model: "buffalo_l",
          runtime: "onnx",
        },
      },
    },
  },
};

/**
 * Default configuration template for lumen-clip service
 */
export const defaultClipConfig: LumenConfig = {
  metadata: {
    version: "1.0.0",
    region: "other",
    cache_dir: "~/.lumen",
  },
  deployment: {
    mode: "single",
  },
  server: {
    port: 50052,
    host: "0.0.0.0",
    mdns: {
      enabled: false,
      service_name: "lumen-clip",
    },
  },
  services: {
    clip: {
      enabled: true,
      package: "lumen_clip",
      import: {
        registry_class: "lumen_clip.service.CLIPServiceRegistry",
        add_to_server: "lumen_clip.service.add_to_server",
      },
      backend_settings: {
        device: null,
        batch_size: 1,
        onnx_providers: null,
      },
      models: {
        default: {
          model: "ViT-B-32",
          runtime: "onnx",
        },
      },
    },
  },
};

/**
 * Get default config for a specific service
 */
export function getDefaultConfig(service: "face" | "clip"): LumenConfig {
  switch (service) {
    case "face":
      return JSON.parse(JSON.stringify(defaultFaceConfig));
    case "clip":
      return JSON.parse(JSON.stringify(defaultClipConfig));
    default:
      return JSON.parse(JSON.stringify(defaultFaceConfig));
  }
}

/**
 * Create an empty config template
 */
export function createEmptyConfig(): Partial<LumenConfig> {
  return {
    metadata: {
      version: "1.0.0",
      region: "other",
      cache_dir: "~/.lumen",
    },
    deployment: {
      mode: "single",
    },
    server: {
      port: 50051,
      host: "0.0.0.0",
    },
    services: {},
  };
}

/**
 * Service model configurations with hardcoded aliases
 */
export interface ModelAlias {
  alias: string;
  label: string;
  models: { value: string; label: string }[];
}

/**
 * Available models for CLIP service
 */
export const clipModelConfig: ModelAlias[] = [
  {
    alias: "general",
    label: "General Purpose",
    models: [
      { value: "MobileCLIP2-S2", label: "MobileCLIP2-S2 (Mobile, Fast)" },
      { value: "MobileCLIP2-S4", label: "MobileCLIP2-S4 (Mobile, Balanced)" },
      { value: "CN-CLIP_ViT-B-16", label: "CN-CLIP ViT-B-16 (Chinese)" },
      { value: "CN-CLIP_ViT-L-14", label: "CN-CLIP ViT-L-14 (Chinese, Large)" },
    ],
  },
  {
    alias: "bioclip",
    label: "Biology/Nature",
    models: [{ value: "bioclip-2", label: "BioCLIP-2 (Biology Specialized)" }],
  },
];

/**
 * Available models for Face service
 */
export const faceModelConfig: ModelAlias[] = [
  {
    alias: "general",
    label: "General Purpose",
    models: [
      { value: "antelopev2", label: "Antelope V2 (High Quality)" },
      { value: "buffalo_l", label: "Buffalo L (Recommended)" },
    ],
  },
];

/**
 * Available models for OCR service
 */
export const ocrModelConfig: ModelAlias[] = [
  {
    alias: "general",
    label: "General Purpose",
    models: [{ value: "PP-OCRv5", label: "PaddleOCR V5 (Latest)" }],
  },
];

/**
 * Runtime options
 */
export const runtimeOptions = [
  {
    value: "onnx",
    label: "ONNX Runtime",
    description: "Cross-platform, recommended",
  },
  { value: "torch", label: "PyTorch", description: "Native PyTorch support" },
  { value: "rknn", label: "RKNN", description: "Rockchip NPU acceleration" },
];

/**
 * Device options
 */
export const deviceOptions = [
  {
    value: null,
    label: "Auto-detect",
    description: "Automatically select best device",
  },
  { value: "cpu", label: "CPU", description: "Use CPU only" },
  { value: "cuda", label: "CUDA", description: "NVIDIA GPU acceleration" },
  { value: "mps", label: "MPS", description: "Apple Silicon GPU acceleration" },
];

/**
 * ONNX Execution Providers
 */
export const onnxProviders = [
  "CPUExecutionProvider",
  "CUDAExecutionProvider",
  "CoreMLExecutionProvider",
  "TensorrtExecutionProvider",
  "OpenVINOExecutionProvider",
  "DmlExecutionProvider",
];

/**
 * Region options
 */
export const regionOptions = [
  {
    value: "other",
    label: "International (HuggingFace)",
    description: "Use HuggingFace for model downloads",
  },
  {
    value: "cn",
    label: "China (ModelScope)",
    description: "Use ModelScope for model downloads",
  },
];
