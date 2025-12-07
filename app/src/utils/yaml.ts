import type { LumenConfig } from "../types/lumen-config";

/**
 * Convert a LumenConfig object to YAML string
 * Uses a simple manual serializer to maintain control over formatting
 */
export function configToYaml(config: Partial<LumenConfig>): string {
  const lines: string[] = [];

  // Helper to add indented lines
  const indent = (level: number) => "  ".repeat(level);

  // Metadata section
  if (config.metadata) {
    lines.push("metadata:");
    lines.push(`${indent(1)}version: "${config.metadata.version}"`);
    lines.push(`${indent(1)}region: ${config.metadata.region}`);
    lines.push(`${indent(1)}cache_dir: "${config.metadata.cache_dir}"`);
    lines.push("");
  }

  // Deployment section
  if (config.deployment) {
    lines.push("deployment:");
    const mode = (config.deployment as any).mode || "single";
    lines.push(`${indent(1)}mode: ${mode}`);
    lines.push("");
  }

  // Server section
  if (config.server) {
    lines.push("server:");
    lines.push(`${indent(1)}port: ${config.server.port}`);
    if (config.server.host) {
      lines.push(`${indent(1)}host: "${config.server.host}"`);
    }
    if (config.server.mdns) {
      lines.push(`${indent(1)}mdns:`);
      lines.push(`${indent(2)}enabled: ${config.server.mdns.enabled ?? false}`);
      if (config.server.mdns.service_name) {
        lines.push(
          `${indent(2)}service_name: "${config.server.mdns.service_name}"`,
        );
      }
    }
    lines.push("");
  }

  // Services section
  if (config.services) {
    lines.push("services:");

    for (const [serviceName, serviceConfig] of Object.entries(
      config.services,
    )) {
      lines.push(`${indent(1)}${serviceName}:`);
      lines.push(`${indent(2)}enabled: ${serviceConfig.enabled}`);
      lines.push(`${indent(2)}package: "${serviceConfig.package}"`);

      // Import section
      lines.push(`${indent(2)}import:`);
      lines.push(
        `${indent(3)}registry_class: "${serviceConfig.import.registry_class}"`,
      );
      lines.push(
        `${indent(3)}add_to_server: "${serviceConfig.import.add_to_server}"`,
      );

      // Backend settings (optional)
      if (serviceConfig.backend_settings) {
        const bs = serviceConfig.backend_settings;
        const hasSettings =
          bs.device !== undefined ||
          bs.batch_size !== undefined ||
          bs.onnx_providers !== undefined;

        if (hasSettings) {
          lines.push(`${indent(2)}backend_settings:`);

          if (bs.device !== undefined) {
            lines.push(
              `${indent(3)}device: ${bs.device === null ? "null" : `"${bs.device}"`}`,
            );
          }
          if (bs.batch_size !== undefined) {
            lines.push(`${indent(3)}batch_size: ${bs.batch_size}`);
          }
          if (bs.onnx_providers !== undefined) {
            if (bs.onnx_providers === null) {
              lines.push(`${indent(3)}onnx_providers: null`);
            } else if (bs.onnx_providers.length > 0) {
              lines.push(`${indent(3)}onnx_providers:`);
              bs.onnx_providers.forEach((provider) => {
                lines.push(`${indent(4)}- "${provider}"`);
              });
            }
          }
        }
      }

      // Models section
      if (
        serviceConfig.models &&
        Object.keys(serviceConfig.models).length > 0
      ) {
        lines.push(`${indent(2)}models:`);

        for (const [modelAlias, modelConfig] of Object.entries(
          serviceConfig.models,
        )) {
          lines.push(`${indent(3)}${modelAlias}:`);
          lines.push(`${indent(4)}model: "${modelConfig.model}"`);
          lines.push(`${indent(4)}runtime: ${modelConfig.runtime}`);

          if (modelConfig.rknn_device) {
            lines.push(`${indent(4)}rknn_device: "${modelConfig.rknn_device}"`);
          }
          if (modelConfig.dataset) {
            lines.push(`${indent(4)}dataset: "${modelConfig.dataset}"`);
          }
        }
      }
    }
  }

  return lines.join("\n");
}

/**
 * Validate that a config object has required fields
 */
export function validateConfig(config: Partial<LumenConfig>): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (!config.metadata) {
    errors.push("Metadata section is required");
  } else {
    if (!config.metadata.version) errors.push("metadata.version is required");
    if (!config.metadata.region) errors.push("metadata.region is required");
    if (!config.metadata.cache_dir)
      errors.push("metadata.cache_dir is required");
  }

  if (!config.deployment) {
    errors.push("Deployment section is required");
  }

  if (!config.server) {
    errors.push("Server section is required");
  } else {
    if (!config.server.port) errors.push("server.port is required");
  }

  if (!config.services || Object.keys(config.services).length === 0) {
    errors.push("At least one service must be configured");
  } else {
    // Validate deployment mode constraints
    const deploymentMode = (config.deployment as any)?.mode || "single";
    const serviceCount = Object.keys(config.services).length;

    if (deploymentMode === "single" && serviceCount > 1) {
      errors.push(
        "Single mode allows only one service. Please remove extra services or switch to Hub mode.",
      );
    }

    // Validate that each service has at least one model
    for (const [serviceName, serviceConfig] of Object.entries(
      config.services,
    )) {
      if (
        !serviceConfig.models ||
        Object.keys(serviceConfig.models).length === 0
      ) {
        errors.push(
          `Service '${serviceName}' must have at least one model configured`,
        );
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Generate a filename for the config
 *
 * Note: Each Lumen service (lumen-face, lumen-clip, etc.) reads only its own
 * section from the config file, so a single lumen.yaml can contain multiple services.
 */
export function generateConfigFilename(): string {
  return "lumen.yaml";
}
