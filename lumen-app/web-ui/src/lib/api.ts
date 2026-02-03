import type {
  paths,
  components,
} from "@/types/schema";

// Base URL for API requests
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// Helper to build URL with query parameters
function buildUrl(
  path: string,
  query?: Record<string, string | number | boolean | undefined | null>
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
async function fetchApi<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
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
  data: ConfigRequest
): Promise<ConfigResponse> {
  return fetchApi<ConfigResponse>(buildUrl("/api/v1/config/generate"), {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export type GetCurrentConfigResponse =
  paths["/api/v1/config/current"]["get"]["responses"]["200"]["content"]["application/json"];

export async function getCurrentConfig(): Promise<GetCurrentConfigResponse> {
  return fetchApi<GetCurrentConfigResponse>(
    buildUrl("/api/v1/config/current"),
    {
      method: "GET",
    }
  );
}

export async function validateConfig(
  config: Record<string, unknown>
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
  data: Record<string, unknown>
): Promise<PathValidationResponse> {
  return fetchApi<PathValidationResponse>(buildUrl("/api/v1/config/validate-path"), {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function loadConfig(configPath: string): Promise<unknown> {
  return fetchApi<unknown>(
    buildUrl("/api/v1/config/load", { config_path: configPath }),
    {
      method: "POST",
    }
  );
}

// ===== Hardware APIs =====

export type HardwareInfo = components["schemas"]["HardwareInfo"];
export type HardwarePreset = components["schemas"]["HardwarePreset"];
export type DriverStatus = components["schemas"]["DriverStatus"];

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
  presetName: string
): Promise<DriverStatus[]> {
  return fetchApi<DriverStatus[]>(
    buildUrl(`/api/v1/hardware/presets/${encodeURIComponent(presetName)}/check`),
    {
      method: "GET",
    }
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

export type InstallListResponse = components["schemas"]["InstallListResponse"];
export type InstallRequest = components["schemas"]["InstallRequest"];
export type InstallStatus = components["schemas"]["InstallStatus"];

export async function listTasks(): Promise<InstallListResponse> {
  return fetchApi<InstallListResponse>(buildUrl("/api/v1/install/tasks"), {
    method: "GET",
  });
}

export async function createInstallTask(
  data: InstallRequest
): Promise<InstallStatus> {
  return fetchApi<InstallStatus>(buildUrl("/api/v1/install/tasks"), {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function getTaskStatus(taskId: string): Promise<InstallStatus> {
  return fetchApi<InstallStatus>(
    buildUrl(`/api/v1/install/tasks/${encodeURIComponent(taskId)}`),
    {
      method: "GET",
    }
  );
}

export async function cancelTask(taskId: string): Promise<unknown> {
  return fetchApi<unknown>(
    buildUrl(`/api/v1/install/tasks/${encodeURIComponent(taskId)}`),
    {
      method: "DELETE",
    }
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
  data: ServerStartRequest
): Promise<ServerStatus> {
  return fetchApi<ServerStatus>(buildUrl("/api/v1/server/start"), {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function stopServer(
  data: ServerStopRequest
): Promise<ServerStatus> {
  return fetchApi<ServerStatus>(buildUrl("/api/v1/server/stop"), {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function restartServer(
  data: ServerRestartRequest
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

export async function getServerLogs(query?: GetServerLogsQuery): Promise<ServerLogs> {
  return fetchApi<ServerLogs>(
    buildUrl("/api/v1/server/logs", query as Record<string, string | number>),
    {
      method: "GET",
    }
  );
}
