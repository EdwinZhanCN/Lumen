import type { paths, components } from "@/types/schema";

// Base URL for API requests
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// Helper to build URL with query parameters
function buildUrl(
  path: string,
  query?: Record<string, string | number | boolean | undefined | null>,
): string {
  const url = new URL(path, BASE_URL);
  if (query) {
    Object.entries(query).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        url.searchParams.set(key, String(value));
      }
    });
  }
  return url.toString();
}

// Generic fetch wrapper
async function fetchApi<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      message: `HTTP ${response.status}: ${response.statusText}`,
    }));
    throw new Error(error.message || `HTTP ${response.status}`);
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

// ===== Health API =====

export type HealthCheckResponse =
  paths["/health"]["get"]["responses"]["200"]["content"]["application/json"];

export async function healthCheck(): Promise<HealthCheckResponse> {
  return fetchApi<HealthCheckResponse>(buildUrl("/health"), {
    method: "GET",
  });
}

// ===== Config APIs =====

export type ConfigRequest = components["schemas"]["ConfigRequest"];
export type ConfigResponse = components["schemas"]["ConfigResponse"];

export async function generateConfig(
  data: ConfigRequest,
): Promise<ConfigResponse> {
  return fetchApi<ConfigResponse>(buildUrl("/api/v1/config/generate"), {
    method: "POST",
    body: JSON.stringify(data),
  });
}

// Current config response - matches actual backend response structure
export type CurrentConfigResponse =
  | {
      loaded: false;
      message: string;
    }
  | {
      loaded: true;
      cache_dir: string;
      region: string;
      port: number;
      service_name: string;
      device: {
        runtime: string;
        batch_size: number;
        precision: string;
        rknn_device: string | null;
        onnx_providers: string[];
      };
    };

export async function getCurrentConfig(): Promise<CurrentConfigResponse> {
  return fetchApi<CurrentConfigResponse>(buildUrl("/api/v1/config/current"), {
    method: "GET",
  });
}

export async function loadConfig(configPath: string): Promise<{
  loaded: boolean;
  config_path?: string;
  cache_dir?: string;
  region?: string;
  port?: number;
  service_name?: string;
}> {
  return fetchApi(
    buildUrl("/api/v1/config/load", { config_path: configPath }),
    {
      method: "POST",
    },
  );
}

export async function getConfigYaml(): Promise<{
  loaded: boolean;
  yaml?: string;
  cache_dir?: string;
}> {
  return fetchApi(buildUrl("/api/v1/config/yaml"), {
    method: "GET",
  });
}

export async function validateConfig(
  config: Record<string, unknown>,
): Promise<unknown> {
  return fetchApi<unknown>(buildUrl("/api/v1/config/validate"), {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export type PathValidationResponse = {
  valid: boolean;
  error?: string;
  warning?: string;
  free_space_gb?: number;
  exists?: boolean;
};

export async function validatePath(
  data: Record<string, unknown>,
): Promise<PathValidationResponse> {
  return fetchApi<PathValidationResponse>(
    buildUrl("/api/v1/config/validate-path"),
    {
      method: "POST",
      body: JSON.stringify(data),
    },
  );
}

// ===== Hardware APIs =====

export type HardwareInfo = components["schemas"]["HardwareInfoResponse"];
export type HardwarePreset = components["schemas"]["HardwarePresetResponse"];
export type DriverStatus = components["schemas"]["DriverCheckResponse"];

export async function getHardwareInfo(): Promise<HardwareInfo> {
  return fetchApi<HardwareInfo>(buildUrl("/api/v1/hardware/info"), {
    method: "GET",
  });
}

export async function listHardwarePresets(): Promise<HardwarePreset[]> {
  return fetchApi<HardwarePreset[]>(buildUrl("/api/v1/hardware/presets"), {
    method: "GET",
  });
}

export async function checkPresetDrivers(
  presetName: string,
): Promise<DriverStatus[]> {
  return fetchApi<DriverStatus[]>(
    buildUrl(
      `/api/v1/hardware/presets/${encodeURIComponent(presetName)}/check`,
    ),
    {
      method: "GET",
    },
  );
}

export type DetectHardwareResponse =
  paths["/api/v1/hardware/detect"]["post"]["responses"]["200"]["content"]["application/json"];

export async function detectHardware(): Promise<DetectHardwareResponse> {
  return fetchApi<DetectHardwareResponse>(buildUrl("/api/v1/hardware/detect"), {
    method: "POST",
  });
}

// ===== Install APIs =====

export type InstallStatusResponse =
  components["schemas"]["InstallStatusResponse"];
export type InstallSetupRequest = components["schemas"]["InstallSetupRequest"];
export type InstallTaskResponse = components["schemas"]["InstallTaskResponse"];
export type InstallTaskListResponse =
  components["schemas"]["InstallTaskListResponse"];
export type InstallLogsResponse = components["schemas"]["InstallLogsResponse"];
export type InstallStep = components["schemas"]["InstallStep"];

// New types for check-path endpoint
export type ServiceStatus = {
  micromamba: boolean;
  environment: boolean;
  config: boolean;
  drivers: boolean;
};

export type CheckInstallationPathResponse = {
  has_existing_service: boolean;
  service_status: ServiceStatus;
  ready_to_start: boolean;
  recommended_action: "start_existing" | "configure_new" | "repair";
  message: string;
};

export async function getInstallStatus(): Promise<InstallStatusResponse> {
  return fetchApi<InstallStatusResponse>(buildUrl("/api/v1/install/status"), {
    method: "GET",
  });
}

export async function checkInstallationPath(
  path: string,
): Promise<CheckInstallationPathResponse> {
  return fetchApi<CheckInstallationPathResponse>(
    buildUrl("/api/v1/install/check-path", { path }),
    {
      method: "GET",
    },
  );
}

export async function startInstallation(
  data: InstallSetupRequest,
): Promise<InstallTaskResponse> {
  return fetchApi<InstallTaskResponse>(buildUrl("/api/v1/install/setup"), {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function listInstallTasks(): Promise<InstallTaskListResponse> {
  return fetchApi<InstallTaskListResponse>(buildUrl("/api/v1/install/tasks"), {
    method: "GET",
  });
}

export async function getInstallTask(
  taskId: string,
): Promise<InstallTaskResponse> {
  return fetchApi<InstallTaskResponse>(
    buildUrl(`/api/v1/install/tasks/${encodeURIComponent(taskId)}`),
    {
      method: "GET",
    },
  );
}

export type GetInstallLogsQuery = {
  tail?: number;
};

export async function getInstallLogs(
  taskId: string,
  query?: GetInstallLogsQuery,
): Promise<InstallLogsResponse> {
  return fetchApi<InstallLogsResponse>(
    buildUrl(
      `/api/v1/install/tasks/${encodeURIComponent(taskId)}/logs`,
      query as Record<string, string | number>,
    ),
    {
      method: "GET",
    },
  );
}

// ===== Server APIs =====

export type ServerStatus = components["schemas"]["ServerStatus"];
export type ServerStartRequest = components["schemas"]["ServerStartRequest"];
export type ServerStopRequest = components["schemas"]["ServerStopRequest"];
export type ServerRestartRequest =
  components["schemas"]["ServerRestartRequest"];
export type ServerLogs = components["schemas"]["ServerLogs"];

export async function getServerStatus(): Promise<ServerStatus> {
  return fetchApi<ServerStatus>(buildUrl("/api/v1/server/status"), {
    method: "GET",
  });
}

export async function startServer(
  data: ServerStartRequest,
): Promise<ServerStatus> {
  return fetchApi<ServerStatus>(buildUrl("/api/v1/server/start"), {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function stopServer(
  data: ServerStopRequest,
): Promise<ServerStatus> {
  return fetchApi<ServerStatus>(buildUrl("/api/v1/server/stop"), {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function restartServer(
  data: ServerRestartRequest,
): Promise<ServerStatus> {
  return fetchApi<ServerStatus>(buildUrl("/api/v1/server/restart"), {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export type GetServerLogsQuery = {
  lines?: number;
  since?: number | null;
};

export async function getServerLogs(
  query?: GetServerLogsQuery,
): Promise<ServerLogs> {
  return fetchApi<ServerLogs>(
    buildUrl("/api/v1/server/logs", query as Record<string, string | number>),
    {
      method: "GET",
    },
  );
}
