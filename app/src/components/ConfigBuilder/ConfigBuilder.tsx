import { useState } from "react";
import type { LumenConfig } from "../../types/lumen-config";
import { MetadataSection } from "./MetadataSection";
import { DeploymentSection } from "./DeploymentSection";
import { ServerSection } from "./ServerSection";
import { ServiceSection } from "./ServiceSection";
import { YamlPreview } from "../YamlPreview";
import { createEmptyConfig } from "../../utils/defaultConfig";
import { validateConfig, configToYaml } from "../../utils/yaml";
import { saveConfigFile } from "../../utils/tauri";

export function ConfigBuilder() {
  const [config, setConfig] =
    useState<Partial<LumenConfig>>(createEmptyConfig());
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);

  const handleMetadataChange = (metadata: LumenConfig["metadata"]) => {
    setConfig((prev) => ({ ...prev, metadata }));
  };

  const handleDeploymentChange = (deployment: LumenConfig["deployment"]) => {
    setConfig((prev) => ({ ...prev, deployment }));
  };

  const handleServerChange = (server: LumenConfig["server"]) => {
    setConfig((prev) => ({ ...prev, server }));
  };

  const handleServicesChange = (services: LumenConfig["services"]) => {
    setConfig((prev) => ({ ...prev, services }));
  };

  const handleSave = async () => {
    const validation = validateConfig(config);

    if (!validation.valid) {
      setSaveStatus({
        type: "error",
        message: `Validation failed: ${validation.errors.join(", ")}`,
      });
      return;
    }

    setSaving(true);
    setSaveStatus(null);

    try {
      const filename = "lumen.yaml";
      const yamlContent = configToYaml(config);
      const cacheDir = config.metadata?.cache_dir;

      const savedPath = await saveConfigFile(filename, yamlContent, cacheDir);

      setSaveStatus({
        type: "success",
        message: `Configuration saved to ${savedPath}`,
      });
    } catch (error) {
      setSaveStatus({
        type: "error",
        message: `Failed to save: ${error}`,
      });
    } finally {
      setSaving(false);
    }
  };

  const handleExport = () => {
    const validation = validateConfig(config);

    if (!validation.valid) {
      setSaveStatus({
        type: "error",
        message: `Cannot export invalid config: ${validation.errors.join(", ")}`,
      });
      return;
    }

    const yamlContent = configToYaml(config);
    const filename = "lumen.yaml";
    const blob = new Blob([yamlContent], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    setSaveStatus({
      type: "success",
      message: `Configuration exported as ${filename}`,
    });
  };

  return (
    <div className="min-h-screen bg-base-200 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">
            Lumen Configuration Builder
          </h1>
          <p className="text-base-content/70">
            Create and manage configuration files for Lumen ML services
          </p>
        </div>

        {/* Main Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Configuration Forms */}
          <div className="space-y-6">
            {config.metadata && (
              <MetadataSection
                metadata={config.metadata}
                onChange={handleMetadataChange}
              />
            )}

            {config.deployment && (
              <DeploymentSection
                deployment={config.deployment}
                onChange={handleDeploymentChange}
              />
            )}

            {config.server && (
              <ServerSection
                server={config.server}
                onChange={handleServerChange}
              />
            )}

            {config.services && config.deployment && (
              <ServiceSection
                services={config.services}
                deployment={config.deployment}
                onChange={handleServicesChange}
              />
            )}

            {/* Action Buttons */}
            <div className="card bg-base-100 shadow-lg">
              <div className="card-body">
                <h3 className="card-title text-lg mb-4">Actions</h3>
                <div className="flex flex-wrap gap-3">
                  <button
                    className="btn btn-primary flex-1"
                    onClick={handleSave}
                    disabled={saving}
                  >
                    {saving ? (
                      <>
                        <span className="loading loading-spinner loading-sm"></span>
                        Saving...
                      </>
                    ) : (
                      <>
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
                            d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"
                          />
                        </svg>
                        Save to Disk
                      </>
                    )}
                  </button>
                  <button
                    className="btn btn-secondary flex-1"
                    onClick={handleExport}
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
                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                      />
                    </svg>
                    Export YAML
                  </button>
                </div>
                <div className="text-sm text-base-content/60 mt-2">
                  <p>
                    💾 <strong>Save to Disk</strong> writes to{" "}
                    <span className="badge badge-sm badge-ghost font-mono">
                      ~/.lumen/lumen.yaml
                    </span>
                  </p>
                  <p className="mt-1">
                    📦 <strong>Export YAML</strong> downloads the config file
                    for deployment
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - YAML Preview */}
          <div className="lg:sticky lg:top-6 h-fit">
            <YamlPreview config={config} />
          </div>
        </div>
      </div>

      {/* Toast Notifications */}
      {saveStatus && (
        <div className="toast toast-end toast-bottom z-50">
          <div
            className={`alert ${saveStatus.type === "success" ? "alert-success" : "alert-error"}`}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="stroke-current shrink-0 h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
            >
              {saveStatus.type === "success" ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              )}
            </svg>
            <span>{saveStatus.message}</span>
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => setSaveStatus(null)}
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
