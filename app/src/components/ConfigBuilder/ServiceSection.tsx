import { useState, useEffect } from "react";
import type {
  LumenConfig,
  ModelConfig,
  BackendSettings,
} from "../../types/lumen-config";
import {
  clipModelConfig,
  faceModelConfig,
  ocrModelConfig,
  onnxProviders,
  type ModelAlias,
} from "../../utils/defaultConfig";

type DeviceType = "nvidia" | "apple" | "cpu" | "intel-gpu" | "rockchip";
type RockchipModel = "rk3566" | "rk3588";

interface DeviceSelection {
  type: DeviceType;
  rockchipModel?: RockchipModel;
}

interface ServiceSectionProps {
  services: LumenConfig["services"];
  deployment: LumenConfig["deployment"];
  onChange: (services: LumenConfig["services"]) => void;
}

export function ServiceSection({
  services,
  deployment,
  onChange,
}: ServiceSectionProps) {
  const deploymentMode = (deployment as any).mode || "single";
  const isSingleMode = deploymentMode === "single";
  const serviceCount = Object.keys(services).length;
  const canAddService = !isSingleMode || serviceCount === 0;
  const [selectedService, setSelectedService] = useState<string | null>(
    Object.keys(services)[0] || null,
  );

  const handleAddService = (serviceType: "face" | "clip" | "ocr") => {
    const newServices = { ...services };

    // Default backend settings - device=null for auto-detection, onnx_providers=null for defaults
    const backendSettings: Partial<BackendSettings> = {
      device: null,
      batch_size: 1,
      onnx_providers: null,
    };

    // Service configurations without initial models
    // Users will add models themselves after selecting device type
    const serviceConfigs: Record<string, any> = {
      face: {
        enabled: true,
        package: "lumen_face",
        import: {
          registry_class: "lumen_face.service.FaceServiceRegistry",
          add_to_server: "lumen_face.service.add_to_server",
        },
        backend_settings: backendSettings,
        models: {},
      },
      clip: {
        enabled: true,
        package: "lumen_clip",
        import: {
          registry_class: "lumen_clip.service.CLIPServiceRegistry",
          add_to_server: "lumen_clip.service.add_to_server",
        },
        backend_settings: backendSettings,
        models: {},
      },
      ocr: {
        enabled: true,
        package: "lumen_ocr",
        import: {
          registry_class: "lumen_ocr.service.OCRServiceRegistry",
          add_to_server: "lumen_ocr.service.add_to_server",
        },
        backend_settings: backendSettings,
        models: {},
      },
    };

    newServices[serviceType] = serviceConfigs[serviceType];
    onChange(newServices);
    setSelectedService(serviceType);
  };

  const handleRemoveService = (serviceName: string) => {
    const newServices = { ...services };
    delete newServices[serviceName];
    onChange(newServices);

    const remaining = Object.keys(newServices);
    setSelectedService(remaining.length > 0 ? remaining[0] : null);
  };

  const handleToggleService = (serviceName: string) => {
    const newServices = { ...services };
    newServices[serviceName] = {
      ...newServices[serviceName],
      enabled: !newServices[serviceName].enabled,
    };
    onChange(newServices);
  };

  const handleBackendSettingsChange = (
    serviceName: string,
    settings: Partial<BackendSettings>,
  ) => {
    const newServices = { ...services };
    newServices[serviceName] = {
      ...newServices[serviceName],
      backend_settings: {
        ...newServices[serviceName].backend_settings,
        ...settings,
      },
    };
    onChange(newServices);
  };

  const handleModelChange = (
    serviceName: string,
    alias: string,
    modelConfig: Partial<ModelConfig>,
  ) => {
    const newServices = { ...services };
    const currentModel = newServices[serviceName].models?.[alias] || {};

    newServices[serviceName] = {
      ...newServices[serviceName],
      models: {
        ...newServices[serviceName].models,
        [alias]: {
          ...currentModel,
          ...modelConfig,
        },
      },
    };
    onChange(newServices);
  };

  const handleDeviceChange = (
    serviceName: string,
    backendSettings: Partial<BackendSettings>,
    modelUpdates: Record<string, Partial<ModelConfig>>,
  ) => {
    const newServices = { ...services };

    // Update backend settings
    if (Object.keys(backendSettings).length > 0) {
      newServices[serviceName] = {
        ...newServices[serviceName],
        backend_settings: {
          ...newServices[serviceName].backend_settings,
          ...backendSettings,
        },
      };
    }

    // Update models
    if (Object.keys(modelUpdates).length > 0) {
      const currentModels = newServices[serviceName].models || {};
      const updatedModels = { ...currentModels };

      Object.entries(modelUpdates).forEach(([alias, config]) => {
        const merged = {
          ...updatedModels[alias],
          ...config,
        };

        // Remove undefined keys
        Object.keys(merged).forEach((key) => {
          if ((merged as any)[key] === undefined) {
            delete (merged as any)[key];
          }
        });

        updatedModels[alias] = merged;
      });

      newServices[serviceName] = {
        ...newServices[serviceName],
        models: updatedModels,
      };
    }

    onChange(newServices);
  };

  const handleAddModel = (
    serviceName: string,
    alias: string,
    modelConfig: ModelConfig,
  ) => {
    const newServices = { ...services };
    newServices[serviceName] = {
      ...newServices[serviceName],
      models: {
        ...newServices[serviceName].models,
        [alias]: modelConfig,
      },
    };
    onChange(newServices);
  };

  const handleRemoveModel = (serviceName: string, alias: string) => {
    const newServices = { ...services };
    const models = { ...newServices[serviceName].models };
    delete models[alias];
    newServices[serviceName] = {
      ...newServices[serviceName],
      models,
    };
    onChange(newServices);
  };

  const serviceNames = Object.keys(services);

  return (
    <div className="card bg-base-100 shadow-lg">
      <div className="card-body">
        <div className="flex justify-between items-center mb-4">
          <h2 className="card-title text-xl">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
              />
            </svg>
            Services
          </h2>

          <div className="dropdown dropdown-end">
            <label
              tabIndex={0}
              className={`btn btn-primary btn-sm ${!canAddService ? "btn-disabled" : ""}`}
              title={
                !canAddService
                  ? "Single mode allows only one service"
                  : "Add a new service"
              }
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                />
              </svg>
              Add Service
            </label>
            <ul
              tabIndex={0}
              className="dropdown-content z-1 menu p-2 shadow bg-base-200 rounded-box w-52 mt-2"
            >
              <li>
                <a onClick={() => handleAddService("face")}>Face Recognition</a>
              </li>
              <li>
                <a onClick={() => handleAddService("clip")}>
                  CLIP (Vision-Language)
                </a>
              </li>
              <li>
                <a onClick={() => handleAddService("ocr")}>
                  OCR (Text Recognition)
                </a>
              </li>
            </ul>
          </div>
        </div>

        {serviceNames.length === 0 ? (
          <div className="alert alert-info">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              className="stroke-current shrink-0 w-6 h-6"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span>
              No services configured. Click "Add Service" to get started.
            </span>
          </div>
        ) : (
          <>
            {isSingleMode && serviceCount > 1 && (
              <div className="alert alert-warning mb-4">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="stroke-current shrink-0 h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
                <span className="text-sm">
                  Single mode allows only one service. Switch to Hub mode or
                  remove extra services.
                </span>
              </div>
            )}
            <div className="tabs tabs-boxed mb-4">
              {serviceNames.map((name) => (
                <a
                  key={name}
                  className={`tab ${selectedService === name ? "tab-active" : ""}`}
                  onClick={() => setSelectedService(name)}
                >
                  {name}
                  {!services[name].enabled && (
                    <span className="badge badge-ghost badge-sm ml-2">
                      disabled
                    </span>
                  )}
                </a>
              ))}
            </div>

            {selectedService && services[selectedService] && (
              <ServiceDetails
                serviceName={selectedService}
                serviceConfig={services[selectedService]}
                onToggle={() => handleToggleService(selectedService)}
                onRemove={() => handleRemoveService(selectedService)}
                onBackendSettingsChange={(settings) =>
                  handleBackendSettingsChange(selectedService, settings)
                }
                onModelChange={(alias, config) =>
                  handleModelChange(selectedService, alias, config)
                }
                onAddModel={(alias, config) =>
                  handleAddModel(selectedService, alias, config)
                }
                onRemoveModel={(alias) =>
                  handleRemoveModel(selectedService, alias)
                }
                onDeviceChange={(backendSettings, modelUpdates) =>
                  handleDeviceChange(
                    selectedService,
                    backendSettings,
                    modelUpdates,
                  )
                }
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}

interface ServiceDetailsProps {
  serviceName: string;
  serviceConfig: any;
  onToggle: () => void;
  onRemove: () => void;
  onBackendSettingsChange: (settings: Partial<BackendSettings>) => void;
  onModelChange: (alias: string, config: Partial<ModelConfig>) => void;
  onAddModel: (alias: string, config: ModelConfig) => void;
  onRemoveModel: (alias: string) => void;
  onDeviceChange: (
    backendSettings: Partial<BackendSettings>,
    modelUpdates: Record<string, Partial<ModelConfig>>,
  ) => void;
}

function ServiceDetails({
  serviceName,
  serviceConfig,
  onToggle,
  onRemove,
  onBackendSettingsChange,
  onModelChange,
  onAddModel,
  onRemoveModel,
  onDeviceChange,
}: ServiceDetailsProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [editingModel, setEditingModel] = useState<{
    alias: string;
    isNew: boolean;
  } | null>(null);
  const [selectedAlias, setSelectedAlias] = useState("");
  const [selectedModel, setSelectedModel] = useState("");

  // Initialize deviceSelection from config (single source of truth for UI)
  const deriveInitialDeviceSelection = (): DeviceSelection | null => {
    const bs = serviceConfig.backend_settings;
    const models = serviceConfig.models || {};
    const firstModel = Object.values(models)[0] as any;

    // Check if this is a Rockchip setup (runtime=rknn)
    if (firstModel && firstModel.runtime === "rknn") {
      return {
        type: "rockchip",
        rockchipModel: firstModel.rknn_device as RockchipModel | undefined,
      };
    }

    // Otherwise, derive from onnx_providers
    if (!bs?.onnx_providers || bs.onnx_providers.length === 0) {
      return null; // Not configured yet
    }

    const providers = bs.onnx_providers;
    if (providers.includes("CUDAExecutionProvider")) {
      return { type: "nvidia" };
    } else if (providers.includes("CoreMLExecutionProvider")) {
      return { type: "apple" };
    } else if (providers.includes("OpenVINOExecutionProvider")) {
      return { type: "intel-gpu" };
    } else if (
      providers.includes("CPUExecutionProvider") &&
      providers.length === 1
    ) {
      return { type: "cpu" };
    }

    return null;
  };

  // Device selection is the single source of truth for UI state
  const [deviceSelection, setDeviceSelection] =
    useState<DeviceSelection | null>(deriveInitialDeviceSelection());

  // Update deviceSelection when serviceConfig changes (e.g., loading a different service)
  useEffect(() => {
    const derived = deriveInitialDeviceSelection();
    setDeviceSelection(derived);
  }, [serviceName]);

  // Get backend settings for a device type
  const getBackendSettingsForDeviceType = (
    type: DeviceType,
  ): Partial<BackendSettings> => {
    const settings: Partial<BackendSettings> = {
      device: null,
      batch_size: serviceConfig.backend_settings?.batch_size || 1,
    };

    switch (type) {
      case "nvidia":
        settings.onnx_providers = [
          "CUDAExecutionProvider",
          "CPUExecutionProvider",
        ];
        break;
      case "apple":
        settings.onnx_providers = [
          "CoreMLExecutionProvider",
          "CPUExecutionProvider",
        ];
        break;
      case "cpu":
        settings.onnx_providers = ["CPUExecutionProvider"];
        break;
      case "intel-gpu":
        settings.onnx_providers = [
          "OpenVINOExecutionProvider",
          "CPUExecutionProvider",
        ];
        break;
      case "rockchip":
        settings.onnx_providers = null; // No ONNX for RKNN
        break;
    }

    return settings;
  };

  // Get model updates for device selection
  const getModelUpdatesForDeviceSelection = (selection: DeviceSelection) => {
    const runtime = selection.type === "rockchip" ? "rknn" : "onnx";
    const updates: Record<string, any> = {};

    Object.keys(serviceConfig.models || {}).forEach((alias) => {
      const currentModel = serviceConfig.models[alias];

      // Check if update is needed
      const needsUpdate =
        currentModel.runtime !== runtime ||
        (runtime === "rknn" &&
          currentModel.rknn_device !== selection.rockchipModel) ||
        (runtime === "onnx" && currentModel.rknn_device !== undefined);

      if (!needsUpdate) return;

      const modelUpdates: any = {
        model: currentModel.model,
        runtime: runtime,
      };

      // Add or remove rknn_device
      if (runtime === "rknn" && selection.rockchipModel) {
        modelUpdates.rknn_device = selection.rockchipModel;
      } else {
        modelUpdates.rknn_device = undefined;
      }

      // Preserve dataset if exists
      if (currentModel.dataset) {
        modelUpdates.dataset = currentModel.dataset;
      }

      updates[alias] = modelUpdates;
    });

    return updates;
  };

  // Handle device type change
  const handleDeviceTypeChange = (type: DeviceType) => {
    const newSelection: DeviceSelection = {
      type,
      rockchipModel: type === "rockchip" ? undefined : undefined,
    };

    // Update UI state
    setDeviceSelection(newSelection);

    // Sync backend settings
    const backendUpdates = getBackendSettingsForDeviceType(type);

    // Sync all models (including the initial general model)
    const modelUpdates = getModelUpdatesForDeviceSelection(newSelection);

    onDeviceChange(backendUpdates, modelUpdates);
  };

  // Handle Rockchip model change (critical!)
  const handleRockchipModelChange = (model: RockchipModel) => {
    const newSelection: DeviceSelection = {
      type: "rockchip",
      rockchipModel: model,
    };

    // Update UI state
    setDeviceSelection(newSelection);

    // Backend settings don't need to change (onnx_providers is still null)

    // Update all models' rknn_device
    const modelUpdates: Record<string, any> = {};
    Object.keys(serviceConfig.models || {}).forEach((alias) => {
      const currentModel = serviceConfig.models[alias];
      modelUpdates[alias] = {
        ...currentModel,
        rknn_device: model,
      };
    });

    onDeviceChange({}, modelUpdates);
  };

  // Get model configuration for current service
  const getModelConfigForService = (): ModelAlias[] => {
    switch (serviceName) {
      case "clip":
        return clipModelConfig;
      case "face":
        return faceModelConfig;
      case "ocr":
        return ocrModelConfig;
      default:
        return [];
    }
  };

  const serviceModelConfig = getModelConfigForService();

  // Get available models for selected alias
  const getModelsForAlias = (alias: string) => {
    const aliasConfig = serviceModelConfig.find((a) => a.alias === alias);
    return aliasConfig?.models || [];
  };

  const handleStartAddModel = () => {
    if (!deviceSelection) {
      alert("Please select a device type first");
      return;
    }

    if (deviceSelection.type === "rockchip" && !deviceSelection.rockchipModel) {
      alert("Please select a Rockchip chip model first");
      return;
    }

    // Find first available alias not already used
    const usedAliases = Object.keys(serviceConfig.models || {});
    const availableAlias = serviceModelConfig.find(
      (a) => !usedAliases.includes(a.alias),
    );

    if (!availableAlias) {
      alert("All model aliases are already configured");
      return;
    }

    setSelectedAlias(availableAlias.alias);
    setSelectedModel("");
    setEditingModel({ alias: availableAlias.alias, isNew: true });
  };

  const handleStartEditModel = (alias: string) => {
    const currentModel = serviceConfig.models[alias];
    setSelectedAlias(alias);
    setSelectedModel(currentModel.model);
    setEditingModel({ alias, isNew: false });
  };

  const handleSaveModel = () => {
    if (!selectedAlias || !selectedModel) {
      alert("Please select both alias and model");
      return;
    }

    if (!deviceSelection) {
      alert("Please select device type first");
      return;
    }

    // Build model config from current deviceSelection (single source of truth)
    const runtime = deviceSelection.type === "rockchip" ? "rknn" : "onnx";

    const modelConfig: any = {
      model: selectedModel,
      runtime: runtime,
    };

    // Add rknn_device for Rockchip
    if (runtime === "rknn") {
      if (!deviceSelection.rockchipModel) {
        alert("Please select a Rockchip chip model first");
        return;
      }
      modelConfig.rknn_device = deviceSelection.rockchipModel;
    }

    // Preserve dataset if editing existing model
    if (!editingModel?.isNew && serviceConfig.models[selectedAlias]?.dataset) {
      modelConfig.dataset = serviceConfig.models[selectedAlias].dataset;
    }

    if (editingModel?.isNew) {
      onAddModel(selectedAlias, modelConfig);
    } else {
      onModelChange(selectedAlias, modelConfig);
    }

    setSelectedAlias("");
    setSelectedModel("");
    setEditingModel(null);
  };

  const handleCancelEdit = () => {
    setSelectedAlias("");
    setSelectedModel("");
    setEditingModel(null);
  };

  return (
    <div className="space-y-6">
      {/* Service Header */}
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-lg font-semibold">{serviceName}</h3>
          <p className="text-sm text-base-content/60">
            Package: <code className="text-xs">{serviceConfig.package}</code>
          </p>
        </div>
        <div className="flex gap-2">
          <button
            className={`btn btn-sm ${serviceConfig.enabled ? "btn-success" : "btn-ghost"}`}
            onClick={onToggle}
          >
            {serviceConfig.enabled ? "Enabled" : "Disabled"}
          </button>
          <button
            className="btn btn-sm btn-error btn-outline"
            onClick={onRemove}
          >
            Remove
          </button>
        </div>
      </div>

      {/* Simplified Device Selection */}
      <div>
        <h4 className="font-medium mb-3 flex items-center gap-2">
          Hardware Configuration
          <div
            className="tooltip"
            data-tip="Select your hardware type for optimized performance"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-4 w-4 text-base-content/40 hover:text-base-content/70"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
        </h4>

        <div className="form-control">
          <label className="label pb-2">
            <span className="label-text">Device Type</span>
          </label>
          <select
            className="select select-bordered w-full"
            value={deviceSelection?.type || ""}
            disabled={showAdvanced}
            onChange={(e) => {
              const value = e.target.value;
              if (value) {
                handleDeviceTypeChange(value as DeviceType);
              }
            }}
          >
            <option value="">Select your device...</option>
            <option value="nvidia">NVIDIA GPU (CUDA)</option>
            <option value="apple">Apple Silicon (M1/M2/M3)</option>
            <option value="cpu">CPU (Generic)</option>
            <option value="intel-gpu">Intel GPU (OpenVINO)</option>
            <option value="rockchip">Rockchip NPU</option>
          </select>
          {showAdvanced && (
            <label className="label">
              <span className="label-text-alt text-warning">
                Disabled when Advanced Settings are enabled
              </span>
            </label>
          )}
        </div>

        {deviceSelection?.type === "rockchip" && !showAdvanced && (
          <div className="form-control mt-3">
            <label className="label pb-2">
              <span className="label-text">Rockchip Chip Model</span>
            </label>
            <select
              className="select select-bordered w-full"
              value={deviceSelection.rockchipModel || ""}
              onChange={(e) => {
                const value = e.target.value;
                if (value) {
                  handleRockchipModelChange(value as RockchipModel);
                }
              }}
            >
              <option value="">Select chip model...</option>
              <option value="rk3566">RK3566</option>
              <option value="rk3588">RK3588</option>
            </select>
            <label className="label">
              <span className="label-text-alt text-warning">
                Required for RKNN runtime
              </span>
            </label>
          </div>
        )}

        {/* Advanced Settings Toggle */}
        <div className="form-control mt-4">
          <label className="label cursor-pointer justify-start gap-3">
            <input
              type="checkbox"
              className="toggle toggle-primary"
              checked={showAdvanced}
              onChange={(e) => setShowAdvanced(e.target.checked)}
            />
            <div className="flex items-center gap-2">
              <span className="label-text font-medium">Advanced Settings</span>
              <div
                className="tooltip"
                data-tip="Manually configure device, batch size, and ONNX providers. Disables automatic hardware configuration."
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-4 w-4 text-base-content/40 hover:text-base-content/70"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
            </div>
          </label>
        </div>

        {showAdvanced && (
          <div className="card bg-base-200 p-4 mt-2 space-y-3">
            <div className="alert alert-info text-xs">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                className="stroke-current shrink-0 w-4 h-4"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>
                Device is null by default for auto-detection. Set manually only
                if needed for torch runtime.
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="form-control">
                <label className="label pb-2">
                  <span className="label-text text-sm">
                    Device Override (for torch)
                  </span>
                </label>
                <input
                  type="text"
                  className="input input-bordered input-sm font-mono w-full"
                  placeholder="null (auto)"
                  value={
                    serviceConfig.backend_settings?.device === null
                      ? ""
                      : serviceConfig.backend_settings?.device || ""
                  }
                  onChange={(e) =>
                    onBackendSettingsChange({
                      device: e.target.value || null,
                    })
                  }
                />
              </div>

              <div className="form-control">
                <label className="label pb-2">
                  <span className="label-text text-sm">Batch Size</span>
                </label>
                <input
                  type="number"
                  className="input input-bordered input-sm w-full"
                  value={serviceConfig.backend_settings?.batch_size || 1}
                  onChange={(e) =>
                    onBackendSettingsChange({
                      batch_size: parseInt(e.target.value) || 1,
                    })
                  }
                  min={1}
                />
              </div>
            </div>

            <div className="form-control">
              <label className="label pb-2">
                <span className="label-text text-sm">
                  ONNX Execution Providers
                </span>
              </label>
              <select
                multiple
                className="select select-bordered select-sm h-20 font-mono text-xs w-full"
                value={serviceConfig.backend_settings?.onnx_providers || []}
                onChange={(e) => {
                  const selected = Array.from(
                    e.target.selectedOptions,
                    (opt) => opt.value,
                  );
                  onBackendSettingsChange({
                    onnx_providers: selected.length > 0 ? selected : null,
                  });
                }}
              >
                {onnxProviders
                  .filter((p) => p !== "DmlExecutionProvider")
                  .map((provider) => (
                    <option key={provider} value={provider}>
                      {provider}
                    </option>
                  ))}
              </select>
              <label className="label">
                <span className="label-text-alt text-xs">
                  Hold Ctrl/Cmd to select multiple
                </span>
              </label>
            </div>
          </div>
        )}
      </div>

      {/* Models */}
      <div>
        <div className="flex justify-between items-center mb-3">
          <h4 className="font-medium">Models</h4>
          <button
            className="btn btn-sm btn-outline"
            onClick={handleStartAddModel}
            disabled={!!editingModel}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-4 w-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 6v6m0 0v6m0-6h6m-6 0H6"
              />
            </svg>
            Add Model
          </button>
        </div>

        {editingModel && (
          <div className="card bg-base-200 p-4 mb-4">
            <h5 className="font-medium mb-3">
              {editingModel.isNew ? "Add Model" : "Edit Model"}
            </h5>
            <div className="space-y-3">
              <div className="form-control">
                <label className="label pb-2">
                  <span className="label-text">Model Alias</span>
                </label>
                <select
                  className="select select-bordered w-full"
                  value={selectedAlias}
                  onChange={(e) => {
                    setSelectedAlias(e.target.value);
                    setSelectedModel(""); // Reset model when alias changes
                  }}
                  disabled={!editingModel.isNew}
                >
                  <option value="">Select alias...</option>
                  {serviceModelConfig.map((config) => (
                    <option
                      key={config.alias}
                      value={config.alias}
                      disabled={
                        editingModel.isNew &&
                        Object.keys(serviceConfig.models || {}).includes(
                          config.alias,
                        )
                      }
                    >
                      {config.label} ({config.alias})
                    </option>
                  ))}
                </select>
              </div>

              {selectedAlias && (
                <div className="form-control">
                  <label className="label pb-2">
                    <span className="label-text">Model</span>
                  </label>
                  <select
                    className="select select-bordered w-full"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    <option value="">Select model...</option>
                    {getModelsForAlias(selectedAlias).map((model) => (
                      <option key={model.value} value={model.value}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div className="alert alert-info text-xs mt-2">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  className="stroke-current shrink-0 w-4 h-4"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <div className="text-left">
                  <div className="font-medium">
                    Runtime:{" "}
                    {deviceSelection?.type === "rockchip" ? "rknn" : "onnx"}
                  </div>
                  {deviceSelection?.type === "rockchip" && (
                    <div>
                      RKNN Device:{" "}
                      {deviceSelection.rockchipModel || "Not selected"}
                    </div>
                  )}
                  <div className="text-xs opacity-70 mt-1">
                    Automatically set by Device Type selection
                  </div>
                </div>
              </div>
            </div>

            <div className="flex gap-2 mt-4">
              <button
                className="btn btn-primary btn-sm"
                onClick={handleSaveModel}
                disabled={!selectedAlias || !selectedModel}
              >
                {editingModel.isNew ? "Add" : "Save"}
              </button>
              <button
                className="btn btn-ghost btn-sm"
                onClick={handleCancelEdit}
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {Object.keys(serviceConfig.models || {}).length === 0 && (
          <div className="alert alert-warning">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="stroke-current shrink-0 h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <span>
              No models configured. Please select a device type and add a model.
            </span>
          </div>
        )}

        <div className="space-y-2">
          {Object.entries(serviceConfig.models || {}).map(
            ([alias, config]: [string, any]) => {
              const aliasConfig = serviceModelConfig.find(
                (a) => a.alias === alias,
              );
              const modelInfo = aliasConfig?.models.find(
                (m) => m.value === config.model,
              );

              return (
                <div key={alias} className="card bg-base-200 p-3">
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-sm">{alias}</span>
                        {aliasConfig && (
                          <span className="badge badge-ghost badge-xs">
                            {aliasConfig.label}
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-base-content/60 space-y-1">
                        <div>
                          Model:{" "}
                          <code>
                            {modelInfo ? modelInfo.label : config.model}
                          </code>
                        </div>
                        <div>
                          Runtime:{" "}
                          <span className="badge badge-sm">
                            {config.runtime || "onnx"}
                          </span>
                        </div>
                        {config.rknn_device && (
                          <div>
                            RKNN Device: <code>{config.rknn_device}</code>
                          </div>
                        )}
                        {config.dataset && (
                          <div>
                            Dataset: <code>{config.dataset}</code>
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="flex gap-1">
                      <button
                        className="btn btn-ghost btn-xs"
                        onClick={() => handleStartEditModel(alias)}
                        disabled={!!editingModel}
                        title="Edit model"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-4 w-4"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                          />
                        </svg>
                      </button>
                      <button
                        className="btn btn-ghost btn-xs text-error"
                        onClick={() => onRemoveModel(alias)}
                        disabled={!!editingModel}
                        title="Remove model"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-4 w-4"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M6 18L18 6M6 6l12 12"
                          />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              );
            },
          )}
        </div>
      </div>
    </div>
  );
}
